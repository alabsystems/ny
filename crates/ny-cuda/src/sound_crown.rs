// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound, **f64-native** CROWN backward on CUDA.
//!
//! ny's wgpu sound CROWN is forced to carry f32 coefficients (WGSL has no f64),
//! paying a per-layer f32 rounding error `γ_k = k·2⁻²⁴`. cuBLAS gives exact
//! IEEE-f64, so this keeps the WHOLE backward in **f64** — coefficients AND their
//! certified error stay f64 across every layer, with NO per-layer f32 cast. Only
//! the FINAL concretized bounds are rounded outward to f32. The intermediate error
//! is therefore just the f64 GEMM accumulation `γ_k = k·2⁻⁵³` (plus the propagated
//! incoming error) — ~2²⁹× tighter per layer than the wgpu f32 path, and tighter
//! than the CPU f32-coefficient path too, so it verifies AT LEAST as much.
//!
//! Soundness model (order-independent, so cuBLAS's reduction order is irrelevant —
//! the same property validated for the f64 A·W oracle):
//!  - Linear `A·W`: `a_new = fl_f64(a@w)`, certified `|a_new − a_exact@w| ≤
//!    γ_k·(|a|@|w|) + a_err@|w|` for every `a_exact ∈ [a−a_err, a+a_err]`, all f64.
//!  - Activation: sign-routed slope compose; error `in_err·(|ls|+|us|) + |coeff|·u`
//!    (the `slope_sum` covers a sign flip of `a` under its error; `|coeff|·u` the
//!    f64 multiply rounding).
//!  - Bias/intercept folds: `lb += Σ a·b` certified by `γ_{n+1}` over the fold's
//!    magnitude sum (incoming bound included) + the propagated coefficient error.
//!  - Concretize: worst-case `coeff·x` over the coefficient-interval × input-box
//!    (min/max of 4 corners), summed in f64, rounded OUTWARD to f32.
//!
//! A size-gate keeps small nets (where the GPU launch/transfer dominates the tiny
//! GEMMs) on the proven CPU path via `UnsupportedOp`.

use ny_core::{
    GemmEngine, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuCrownTrajectoryResult,
    GpuResidentCoeffBatched, GpuResnetBatchedDomainRef, GpuResnetSegment, NyError, Result,
};

/// f64 unit roundoff, 2⁻⁵³ (exact).
const U64: f64 = f64::from_bits(0x3CA0_0000_0000_0000);
/// Smallest positive f64 subnormal, 2⁻¹⁰⁷⁴ — additive underflow floor.
const ETA64: f64 = f64::from_bits(0x0000_0000_0000_0001);
/// Largest single-Linear MAC count below which the GPU loses to the CPU (launch /
/// transfer bound) — route small nets to the proven CPU sound path. Matches the
/// f64-GEMM-seam crossover (~16M MACs on the GB10).
const MIN_RESIDENT_MACS: u128 = 1 << 24;

/// Two shared-weight slices belong to the same wide proof frontier when they are
/// literally shared, or (until graph extraction interns weights) value-identical.
/// The latter is load-bearing today: extraction may mint a fresh `Arc` per BaB
/// child even though every child refers to the same network.
fn arc_slice_eq(a: &std::sync::Arc<[f32]>, b: &std::sync::Arc<[f32]>) -> bool {
    std::sync::Arc::ptr_eq(a, b) || a == b
}

fn arc_opt_slice_eq(a: &Option<std::sync::Arc<[f32]>>, b: &Option<std::sync::Arc<[f32]>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => arc_slice_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

/// Fail-closed structural equality used before domain rows are stacked.  Dynamic
/// activation values may differ; topology, dimensions, and affine weights may not.
fn layer_skeleton_matches(a: &GpuCrownLayer, b: &GpuCrownLayer) -> bool {
    use GpuCrownLayer::{Activation, Conv2d, Linear};
    match (a, b) {
        (
            Linear {
                weight: wa,
                bias: ba,
                out_features: oa,
                in_features: ia,
            },
            Linear {
                weight: wb,
                bias: bb,
                out_features: ob,
                in_features: ib,
            },
        ) => oa == ob && ia == ib && arc_slice_eq(wa, wb) && arc_opt_slice_eq(ba, bb),
        (
            Activation {
                num_neurons: na, ..
            },
            Activation {
                num_neurons: nb, ..
            },
        ) => na == nb,
        (
            Conv2d {
                weight_col: wa,
                bias_expanded: ba,
                out_channels: oca,
                in_channels: ica,
                kernel_h: kha,
                kernel_w: kwa,
                stride_h: sha,
                stride_w: swa,
                pad_h: pha,
                pad_w: pwa,
                out_h: oha,
                out_w: owa,
                in_h: iha,
                in_w: iwa,
            },
            Conv2d {
                weight_col: wb,
                bias_expanded: bb,
                out_channels: ocb,
                in_channels: icb,
                kernel_h: khb,
                kernel_w: kwb,
                stride_h: shb,
                stride_w: swb,
                pad_h: phb,
                pad_w: pwb,
                out_h: ohb,
                out_w: owb,
                in_h: ihb,
                in_w: iwb,
            },
        ) => {
            oca == ocb
                && ica == icb
                && kha == khb
                && kwa == kwb
                && sha == shb
                && swa == swb
                && pha == phb
                && pwa == pwb
                && oha == ohb
                && owa == owb
                && iha == ihb
                && iwa == iwb
                && arc_slice_eq(wa, wb)
                && arc_opt_slice_eq(ba, bb)
        }
        // Dual-alpha and max-pool need domain-indexed kernels that this first
        // CUDA-wide increment deliberately does not provide.
        _ => false,
    }
}

fn layers_skeleton_match(a: &[GpuCrownLayer], b: &[GpuCrownLayer]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| layer_skeleton_matches(a, b))
}

fn segment_skeleton_matches(a: &GpuResnetSegment, b: &GpuResnetSegment) -> bool {
    match (a, b) {
        (GpuResnetSegment::Chain(a), GpuResnetSegment::Chain(b))
        | (GpuResnetSegment::Residual(a), GpuResnetSegment::Residual(b)) => {
            layers_skeleton_match(a, b)
        }
        (GpuResnetSegment::ResidualProj(af, ap), GpuResnetSegment::ResidualProj(bf, bp)) => {
            layers_skeleton_match(af, bf) && layers_skeleton_match(ap, bp)
        }
        _ => false,
    }
}

fn resnet_skeleton_matches(a: &[GpuResnetSegment], b: &[GpuResnetSegment]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| segment_skeleton_matches(a, b))
}

fn segments_are_wide_batchable(segments: &[GpuResnetSegment]) -> bool {
    let layers_ok = |layers: &[GpuCrownLayer]| {
        !layers.is_empty()
            && layers.iter().all(|layer| {
                matches!(
                    layer,
                    GpuCrownLayer::Linear { .. }
                        | GpuCrownLayer::Activation { .. }
                        | GpuCrownLayer::Conv2d { .. }
                )
            })
    };
    !segments.is_empty()
        && segments.iter().all(|segment| match segment {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => layers_ok(l),
            GpuResnetSegment::ResidualProj(f, p) => layers_ok(f) && layers_ok(p),
        })
}

/// Stack one corresponding layer chain from every child.  Affine layers retain
/// the shared weights; activation tables become `[domain, neuron]`.  The f64
/// fold below maps each specification row back to its domain block.
fn stack_wide_layers(per_domain: &[&[GpuCrownLayer]]) -> Option<Vec<GpuCrownLayer>> {
    let template = per_domain.first()?;
    let mut out = Vec::with_capacity(template.len());
    for (li, layer) in template.iter().enumerate() {
        match layer {
            GpuCrownLayer::Activation { num_neurons, .. } => {
                let n = *num_neurons;
                let mut lower_slope = Vec::with_capacity(per_domain.len() * n);
                let mut upper_slope = Vec::with_capacity(per_domain.len() * n);
                let mut lower_intercept = Vec::with_capacity(per_domain.len() * n);
                let mut upper_intercept = Vec::with_capacity(per_domain.len() * n);
                for layers in per_domain {
                    let GpuCrownLayer::Activation {
                        lower_slope: ls,
                        upper_slope: us,
                        lower_intercept: li,
                        upper_intercept: ui,
                        num_neurons: dn,
                    } = layers.get(li)?
                    else {
                        return None;
                    };
                    if *dn != n || ls.len() != n || us.len() != n || li.len() != n || ui.len() != n
                    {
                        return None;
                    }
                    lower_slope.extend_from_slice(ls);
                    upper_slope.extend_from_slice(us);
                    lower_intercept.extend_from_slice(li);
                    upper_intercept.extend_from_slice(ui);
                }
                out.push(GpuCrownLayer::Activation {
                    lower_slope,
                    upper_slope,
                    lower_intercept,
                    upper_intercept,
                    num_neurons: n,
                });
            }
            GpuCrownLayer::Linear { .. } | GpuCrownLayer::Conv2d { .. } => {
                out.push(layer.clone());
            }
            _ => return None,
        }
    }
    Some(out)
}

fn stack_wide_segments(domains: &[GpuResnetBatchedDomainRef<'_>]) -> Option<Vec<GpuResnetSegment>> {
    let template = domains.first()?.segments;
    let mut out = Vec::with_capacity(template.len());
    for (si, segment) in template.iter().enumerate() {
        let collect = |pick: fn(&GpuResnetSegment) -> Option<&[GpuCrownLayer]>| {
            domains
                .iter()
                .map(|d| pick(d.segments.get(si)?))
                .collect::<Option<Vec<_>>>()
        };
        out.push(match segment {
            GpuResnetSegment::Chain(_) => {
                let layers = collect(|s| match s {
                    GpuResnetSegment::Chain(l) => Some(l),
                    _ => None,
                })?;
                GpuResnetSegment::Chain(stack_wide_layers(&layers)?)
            }
            GpuResnetSegment::Residual(_) => {
                let layers = collect(|s| match s {
                    GpuResnetSegment::Residual(l) => Some(l),
                    _ => None,
                })?;
                GpuResnetSegment::Residual(stack_wide_layers(&layers)?)
            }
            GpuResnetSegment::ResidualProj(_, _) => {
                let f = collect(|s| match s {
                    GpuResnetSegment::ResidualProj(f, _) => Some(f),
                    _ => None,
                })?;
                let p = collect(|s| match s {
                    GpuResnetSegment::ResidualProj(_, p) => Some(p),
                    _ => None,
                })?;
                GpuResnetSegment::ResidualProj(stack_wide_layers(&f)?, stack_wide_layers(&p)?)
            }
        });
    }
    Some(out)
}

fn stack_wide_table(per_domain: &[&[Vec<f32>]]) -> Option<Vec<Vec<f32>>> {
    let first = per_domain.first()?;
    if per_domain.iter().any(|d| d.len() != first.len()) {
        return None;
    }
    (0..first.len())
        .map(|entry| {
            let block_len = first[entry].len();
            let mut wide = Vec::with_capacity(per_domain.len() * block_len);
            for domain in per_domain {
                if domain[entry].len() != block_len {
                    return None;
                }
                wide.extend_from_slice(&domain[entry]);
            }
            Some(wide)
        })
        .collect()
}

/// `γ_k = k·u / (1 − k·u)` for an f64 length-`k` dot product (`u = 2⁻⁵³`).
#[inline]
fn gamma_k_f64(k: usize) -> f64 {
    let ku = (k as f64) * U64;
    if ku < 0.5 {
        ku / (1.0 - ku)
    } else {
        2.0 * ku
    }
}

/// Round `x` DOWN to f32 (toward −∞): a lower bound must never round up. Handles
/// the ±0 boundary explicitly so the bit step never underflows `u32`.
#[inline]
fn down(x: f64) -> f32 {
    let n = x as f32;
    if !n.is_finite() || f64::from(n) <= x {
        return n;
    }
    if n == 0.0 {
        return f32::from_bits(0x8000_0001);
    }
    let bits = n.to_bits();
    f32::from_bits(if n > 0.0 { bits - 1 } else { bits + 1 })
}

/// Round `x` UP to f32 (toward +∞): an upper bound / error must never round down.
#[inline]
fn up(x: f64) -> f32 {
    let n = x as f32;
    if !n.is_finite() || f64::from(n) >= x {
        return n;
    }
    if n == 0.0 {
        return f32::from_bits(0x0000_0001);
    }
    let bits = n.to_bits();
    f32::from_bits(if n > 0.0 { bits + 1 } else { bits - 1 })
}

/// One SOUND f64-native linear step: `a_new = fl_f64(a@w)` (kept f64) with
/// certified f64 error bounding `|a_new − a_exact@w|` for every exact coefficient
/// in `[a − a_err, a + a_err]`. `a`/`a_err` are f64; `w` is the f32 weight.
fn linear_step_f64<E: GemmEngine + ?Sized>(
    eng: &E,
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    a_err: &[f64],
    w: &[f32],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let w64: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    let abs_a: Vec<f64> = a.iter().map(|x| x.abs()).collect();
    let abs_w: Vec<f64> = w64.iter().map(|x| x.abs()).collect();

    let [a_new, s, prop] =
        eng.gemm_f64_triplet(m, k, n, [a, &abs_a, a_err], [&w64, &abs_w, &abs_w])?;

    let gamma_k = gamma_k_f64(k);
    let real_factor = 1.0 + 2.0 * gamma_k;
    let additive = 8.0 * (k as f64) * ETA64;
    let mut a_err_new = vec![0.0f64; m * n];
    for i in 0..(m * n) {
        a_err_new[i] = (gamma_k * s[i] + prop[i]) * real_factor + additive;
    }
    Ok((a_new, a_err_new))
}

/// One SOUND f64-native activation step (elementwise, sign-routed slopes), the f64
/// analogue of `ny_core::crown_activation_error_step`. Returns
/// `(new_lower_a, new_upper_a, new_lower_err, new_upper_err)`, all f64.
#[allow(clippy::too_many_arguments)]
fn activation_step_f64(
    num_outputs: usize,
    num_neurons: usize,
    lower_a: &[f64],
    upper_a: &[f64],
    lower_err: &[f64],
    upper_err: &[f64],
    lower_slope: &[f32],
    upper_slope: &[f32],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if num_neurons == 0
        || lower_slope.len() != upper_slope.len()
        || !lower_slope.len().is_multiple_of(num_neurons)
    {
        return Err(NyError::InvalidSpec(
            "cuda f64 activation: malformed domain-stacked slope table".into(),
        ));
    }
    let n_domains = lower_slope.len() / num_neurons;
    if n_domains == 0 || !num_outputs.is_multiple_of(n_domains) {
        return Err(NyError::InvalidSpec(
            "cuda f64 activation: specification rows do not partition into domains".into(),
        ));
    }
    let specs_per_domain = num_outputs / n_domains;
    let n = num_outputs * num_neurons;
    let mut nla = vec![0.0f64; n];
    let mut nua = vec![0.0f64; n];
    let mut nle = vec![0.0f64; n];
    let mut nue = vec![0.0f64; n];
    for j in 0..num_outputs {
        let domain = j / specs_per_domain;
        let param_base = domain * num_neurons;
        for i in 0..num_neurons {
            let idx = j * num_neurons + i;
            let ls = f64::from(lower_slope[param_base + i]);
            let us = f64::from(upper_slope[param_base + i]);
            let slope_sum = ls.abs() + us.abs();

            let la = lower_a[idx];
            let lsel = if la >= 0.0 { ls } else { us };
            let lcoeff = la * lsel;
            nla[idx] = lcoeff;
            nle[idx] = lower_err[idx] * slope_sum + lcoeff.abs() * U64 + ETA64;

            let ua = upper_a[idx];
            let usel = if ua >= 0.0 { us } else { ls };
            let ucoeff = ua * usel;
            nua[idx] = ucoeff;
            nue[idx] = upper_err[idx] * slope_sum + ucoeff.abs() * U64 + ETA64;
        }
    }
    Ok((nla, nua, nle, nue))
}

/// SOUND fold of a per-output bias into a running bound accumulator:
/// `acc[s] += Σ_k a[s,k]·b_k`, certifying BOTH the propagated coefficient error
/// (`a_err·|b_k|`) AND the fold's own f64 rounding. Each product rounds once and
/// the sum chains `n` adds onto the incoming `acc[s]`, so the fold error is
/// bounded by Higham's `γ_{n+1}·(|acc| + Σ|a·b|)`, charged over the COMPUTED
/// magnitudes with the same `(1+2γ)` slack + underflow floor as
/// [`linear_step_f64`].
fn bias_fold_f64(
    num_specs: usize,
    n: usize,
    a: &[f64],
    a_err: &[f64],
    bias: &[f32],
    acc: &mut [f64],
    acc_err: &mut [f64],
) {
    let gamma = gamma_k_f64(n + 1);
    let real_factor = 1.0 + 2.0 * gamma;
    let additive = 8.0 * ((n + 1) as f64) * ETA64;
    for s in 0..num_specs {
        let mut mag = acc[s].abs();
        for k in 0..n {
            let bk = f64::from(bias[k]);
            let prod = a[s * n + k] * bk;
            acc[s] += prod;
            mag += prod.abs();
            acc_err[s] += a_err[s * n + k] * bk.abs();
        }
        acc_err[s] += gamma * mag * real_factor + additive;
    }
}

/// SOUND f64 concretization → outward-rounded f32 `(lower, upper)` per spec.
#[allow(clippy::too_many_arguments)]
fn concretize_f64(
    num_specs: usize,
    dim: usize,
    lower_a: &[f64],
    upper_a: &[f64],
    lower_err: &[f64],
    upper_err: &[f64],
    input_lower: &[f32],
    input_upper: &[f32],
    lb: &[f64],
    ub: &[f64],
    lb_err: &[f64],
    ub_err: &[f64],
) -> (Vec<f32>, Vec<f32>) {
    let gamma = gamma_k_f64(8 * dim + 8);
    let additive = 8.0 * ((dim + 1) as f64) * ETA64;
    let mut lo = vec![0.0f32; num_specs];
    let mut hi = vec![0.0f32; num_specs];
    for s in 0..num_specs {
        let mut lacc = lb[s] - lb_err[s];
        let mut labs = lb[s].abs() + lb_err[s].abs();
        let mut uacc = ub[s] + ub_err[s];
        let mut uabs = ub[s].abs() + ub_err[s].abs();
        for i in 0..dim {
            let xl = f64::from(input_lower[i]);
            let xu = f64::from(input_upper[i]);
            let a = lower_a[s * dim + i];
            let e = lower_err[s * dim + i];
            let cl = ((a - e) * xl)
                .min((a - e) * xu)
                .min((a + e) * xl)
                .min((a + e) * xu);
            lacc += cl;
            labs += cl.abs();
            let a2 = upper_a[s * dim + i];
            let e2 = upper_err[s * dim + i];
            let cu = ((a2 - e2) * xl)
                .max((a2 - e2) * xu)
                .max((a2 + e2) * xl)
                .max((a2 + e2) * xu);
            uacc += cu;
            uabs += cu.abs();
        }
        lo[s] = down(lacc - gamma * labs - additive);
        hi[s] = up(uacc + gamma * uabs + additive);
    }
    (lo, hi)
}

/// Gate-free f64-native sound CROWN backward core (Linear/Activation chains).
/// `UnsupportedOp` on conv/pool/dual-alpha layers ⇒ caller falls back to CPU.
fn backward_f64_core<E: GemmEngine + ?Sized>(
    eng: &E,
    layers: &[GpuCrownLayer],
    spec: &[f32],
    num_specs: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Result<GpuCrownResult> {
    if num_specs == 0 {
        return Ok(GpuCrownResult {
            lower_bounds: vec![],
            upper_bounds: vec![],
        });
    }
    if spec.is_empty() || !spec.len().is_multiple_of(num_specs) {
        return Err(NyError::shape_mismatch(vec![num_specs], vec![spec.len()]));
    }
    let dim = spec.len() / num_specs;
    let lower_a: Vec<f64> = spec.iter().map(|&x| f64::from(x)).collect();
    let upper_a = lower_a.clone();
    let lb = vec![0.0f64; num_specs];
    let ub = vec![0.0f64; num_specs];
    // Spec frontier: lower row == upper row == spec, zero bias. Delegate to the
    // shared backward from that initialised frontier.
    run_backward_f64_from(
        eng,
        layers,
        lower_a,
        upper_a,
        lb,
        ub,
        dim,
        num_specs,
        input_lower,
        input_upper,
    )
}

/// SOUND f64 Conv2d CROWN backward step: `A_x = col2im(A ⊛ᵀ W)` (transposed conv).
/// Conv is LINEAR ⇒ the coefficient map is EXACT; only the f64 reshape→GEMM→col2im
/// rounding is certified. Mirrors `linear_step_f64` but with the conv connectivity:
/// the GEMM contracts over `oc` (`A_reshaped[batch·oh·ow, oc] @ W_col[oc, ic·kh·kw]`)
/// and col2im scatters the result into `[batch, ic, in_h, in_w]`, accumulating
/// overlapping windows. Each `A_x` entry sums at most `oc·kh·kw` f64 terms, so the
/// certified error uses `γ_{oc·kh·kw}` over the col2im'd magnitude `S = col2im(|A|⊛|W|)`
/// plus the incoming error amplified `prop = col2im(err⊛|W|)`. Lower & upper rows are
/// batched into one set of GEMMs. Returns `(a_new, err_new)` flat `batch·ic·in_h·in_w`.
#[allow(clippy::too_many_arguments)]
fn conv_step_f64<E: GemmEngine + ?Sized>(
    eng: &E,
    batch: usize, // 2·num_specs (lower rows then upper rows)
    a: &[f64],    // [batch, oc, oh, ow]
    a_err: &[f64],
    weight_col: &[f32], // [oc, ic·kh·kw]
    oc: usize,
    ic: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
    in_h: usize,
    in_w: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let ohw = oh * ow;
    let n = ic * kh * kw;
    let out_d = oc * ohw;
    let in_d = ic * in_h * in_w;
    if a.len() != batch * out_d || a_err.len() != batch * out_d {
        return Err(NyError::shape_mismatch(vec![batch * out_d], vec![a.len()]));
    }
    if weight_col.len() != oc * n {
        return Err(NyError::shape_mismatch(
            vec![oc * n],
            vec![weight_col.len()],
        ));
    }

    // Reshape [batch, oc, oh·ow] → [batch·oh·ow, oc] (transpose oc ↔ spatial per row).
    let m = batch * ohw;
    let mut a_r = vec![0.0f64; m * oc];
    let mut ae_r = vec![0.0f64; m * oc];
    for b in 0..batch {
        for c in 0..oc {
            for p in 0..ohw {
                let src = b * out_d + c * ohw + p;
                let dst = (b * ohw + p) * oc + c;
                a_r[dst] = a[src];
                ae_r[dst] = a_err[src];
            }
        }
    }
    let w64: Vec<f64> = weight_col.iter().map(|&x| f64::from(x)).collect();
    let abs_ar: Vec<f64> = a_r.iter().map(|x| x.abs()).collect();
    let abs_w: Vec<f64> = w64.iter().map(|x| x.abs()).collect();

    // im2col-space backward coefficient, magnitude, and error propagation.
    let [acol, scol, pcol] =
        eng.gemm_f64_triplet(m, oc, n, [&a_r, &abs_ar, &ae_r], [&w64, &abs_w, &abs_w])?;

    // col2im scatter into [batch, ic, in_h, in_w] (f64 accumulate over windows).
    let mut a_new = vec![0.0f64; batch * in_d];
    let mut s_x = vec![0.0f64; batch * in_d];
    let mut p_x = vec![0.0f64; batch * in_d];
    for b in 0..batch {
        for oh_i in 0..oh {
            for ow_i in 0..ow {
                let row = (b * ohw + oh_i * ow + ow_i) * n;
                for c in 0..ic {
                    for kh_i in 0..kh {
                        let ih = (oh_i * sh + kh_i) as isize - ph as isize;
                        if ih < 0 || ih >= in_h as isize {
                            continue;
                        }
                        for kw_i in 0..kw {
                            let iw = (ow_i * sw + kw_i) as isize - pw as isize;
                            if iw < 0 || iw >= in_w as isize {
                                continue;
                            }
                            let wc = c * (kh * kw) + kh_i * kw + kw_i;
                            let dst =
                                b * in_d + c * (in_h * in_w) + (ih as usize) * in_w + (iw as usize);
                            a_new[dst] += acol[row + wc];
                            s_x[dst] += scol[row + wc];
                            p_x[dst] += pcol[row + wc];
                        }
                    }
                }
            }
        }
    }

    // Certified error over the full receptive contraction (γ_{oc·kh·kw}).
    let gamma = gamma_k_f64(oc * kh * kw);
    let real_factor = 1.0 + 2.0 * gamma;
    let additive = 8.0 * ((oc * kh * kw) as f64) * ETA64;
    let mut err_new = vec![0.0f64; batch * in_d];
    for i in 0..(batch * in_d) {
        err_new[i] = (gamma * s_x[i] + p_x[i]) * real_factor + additive;
    }
    Ok((a_new, err_new))
}

/// A CROWN backward frontier in f64: distinct lower/upper coefficient rows + biases,
/// each with its certified (Higham γ_n·S accumulated) error, at width `dim`. Carried
/// across resnet segment boundaries so residual blocks compose soundly.
#[derive(Clone)]
struct Frontier {
    lower_a: Vec<f64>,
    upper_a: Vec<f64>,
    lower_err: Vec<f64>,
    upper_err: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    lb_err: Vec<f64>,
    ub_err: Vec<f64>,
    dim: usize,
}

impl Frontier {
    /// A copy of this frontier's COEFFICIENTS (+ their error) with ZERO bias — the
    /// starting frontier for a residual sub-branch (`out = F(z) + skip(z)`: the outer
    /// bias `b_out` is added ONCE by the merge, never inside a branch).
    fn coeffs_zero_bias(&self, num_specs: usize) -> Frontier {
        Frontier {
            lower_a: self.lower_a.clone(),
            upper_a: self.upper_a.clone(),
            lower_err: self.lower_err.clone(),
            upper_err: self.upper_err.clone(),
            lb: vec![0.0; num_specs],
            ub: vec![0.0; num_specs],
            lb_err: vec![0.0; num_specs],
            ub_err: vec![0.0; num_specs],
            dim: self.dim,
        }
    }
}

fn empty_resident_coeff() -> GpuResidentCoeffBatched {
    GpuResidentCoeffBatched {
        lower_a: Vec::new(),
        upper_a: Vec::new(),
        lower_err: Vec::new(),
        upper_err: Vec::new(),
        lower_b: Vec::new(),
        upper_b: Vec::new(),
        lower_b_err: Vec::new(),
        upper_b_err: Vec::new(),
        dim: 0,
        num_specs: 0,
        num_specs_per_dom: 0,
    }
}

/// Next representable f64 toward +infinity for a finite non-negative value.
/// Used to make the f64 bookkeeping additions below directed, rather than
/// assuming their round-to-nearest result rounded upward.
fn next_up_nonnegative_f64(x: f64) -> f64 {
    debug_assert!(x.is_finite() && x >= 0.0);
    if x == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

/// Convert one f64 center/error enclosure into the public f32 center/error
/// representation.  The f32 error includes BOTH the inherited certified error
/// and the center's f64→f32 cast displacement, each rounded outward.
fn widen_center_error_f64_to_f32(center: f64, error: f64) -> Result<(f32, f32)> {
    if !center.is_finite() || !error.is_finite() || error < 0.0 {
        return Err(NyError::NumericalInstability(
            "cuda wide trajectory: non-finite center or invalid certified error".into(),
        ));
    }
    let center_f32 = center as f32;
    if !center_f32.is_finite() {
        return Err(NyError::NumericalInstability(
            "cuda wide trajectory: f64 coefficient cannot be represented by finite f32".into(),
        ));
    }
    // The subtraction and following addition are correctly rounded f64 ops.
    // Advancing one f64 ULP after each non-zero result bounds either rounding
    // direction before the final directed f32 conversion.
    let cast_delta_raw = (center - f64::from(center_f32)).abs();
    let cast_delta = if cast_delta_raw == 0.0 {
        0.0
    } else {
        next_up_nonnegative_f64(cast_delta_raw)
    };
    let total_raw = error + cast_delta;
    if !total_raw.is_finite() {
        return Err(NyError::NumericalInstability(
            "cuda wide trajectory: certified error overflow".into(),
        ));
    }
    let total = if total_raw == 0.0 {
        0.0
    } else {
        next_up_nonnegative_f64(total_raw)
    };
    let error_f32 = up(total);
    if !error_f32.is_finite() {
        return Err(NyError::NumericalInstability(
            "cuda wide trajectory: certified error cannot be represented by finite f32".into(),
        ));
    }
    Ok((center_f32, error_f32))
}

fn widen_center_error_vec(
    centers: &[f64],
    errors: &[f64],
    site: &'static str,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if centers.len() != errors.len() {
        return Err(NyError::shape_mismatch(
            vec![centers.len()],
            vec![errors.len()],
        ));
    }
    let mut center_out = Vec::new();
    let mut error_out = Vec::new();
    center_out
        .try_reserve_exact(centers.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: centers.len().saturating_mul(size_of::<f32>()),
            budget_bytes: usize::MAX,
            site,
        })?;
    error_out
        .try_reserve_exact(errors.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: errors.len().saturating_mul(size_of::<f32>()),
            budget_bytes: usize::MAX,
            site,
        })?;
    for (&center, &error) in centers.iter().zip(errors) {
        let (c, e) = widen_center_error_f64_to_f32(center, error)?;
        center_out.push(c);
        error_out.push(e);
    }
    Ok((center_out, error_out))
}

/// Export the f64 frontier without discarding any of its certified enclosure.
/// Every cast displacement is charged into the corresponding f32 error field.
fn frontier_to_resident_coeff(
    frontier: &Frontier,
    num_specs: usize,
    num_specs_per_dom: usize,
) -> Result<GpuResidentCoeffBatched> {
    if frontier.dim == 0 || num_specs == 0 || num_specs_per_dom == 0 {
        return Err(NyError::InvalidSpec(
            "cuda wide trajectory: zero coefficient shape".into(),
        ));
    }
    let expected_a = num_specs
        .checked_mul(frontier.dim)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide trajectory: coeff size overflow".into()))?;
    if frontier.lower_a.len() != expected_a
        || frontier.upper_a.len() != expected_a
        || frontier.lower_err.len() != expected_a
        || frontier.upper_err.len() != expected_a
        || frontier.lb.len() != num_specs
        || frontier.ub.len() != num_specs
        || frontier.lb_err.len() != num_specs
        || frontier.ub_err.len() != num_specs
        || !num_specs.is_multiple_of(num_specs_per_dom)
    {
        return Err(NyError::InvalidSpec(
            "cuda wide trajectory: malformed f64 frontier shape".into(),
        ));
    }
    let (lower_a, lower_err) = widen_center_error_vec(
        &frontier.lower_a,
        &frontier.lower_err,
        "cuda::wide_trajectory/lower_a",
    )?;
    let (upper_a, upper_err) = widen_center_error_vec(
        &frontier.upper_a,
        &frontier.upper_err,
        "cuda::wide_trajectory/upper_a",
    )?;
    let (lower_b, lower_b_err) = widen_center_error_vec(
        &frontier.lb,
        &frontier.lb_err,
        "cuda::wide_trajectory/lower_b",
    )?;
    let (upper_b, upper_b_err) = widen_center_error_vec(
        &frontier.ub,
        &frontier.ub_err,
        "cuda::wide_trajectory/upper_b",
    )?;
    Ok(GpuResidentCoeffBatched {
        lower_a,
        upper_a,
        lower_err,
        upper_err,
        lower_b,
        upper_b,
        lower_b_err,
        upper_b_err,
        dim: frontier.dim,
        num_specs,
        num_specs_per_dom,
    })
}

/// Elementwise `dst += src` in f64 with the merge rounding certified into `dst_err`
/// (`+= |src_err| + U64·|dst|`, `U64 = 2^-53` the f64 unit roundoff — a sound bound
/// on one correctly-rounded add `|fl(a+b) - (a+b)| <= U64·|fl(a+b)|`). Sound: widens
/// the error, never tightens.
fn merge_add(dst: &mut [f64], src: &[f64], dst_err: &mut [f64], src_err: &[f64]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
        dst_err[i] += src_err[i].abs() + U64 * dst[i].abs();
    }
}

/// Optional per-Activation auxiliaries, consumed in ReLU FOLD order (each branch's
/// Activations in order, F before P) by a single shared `cursor` advanced once per
/// Activation. `None` ⇒ the base backward — byte-identical to the pre-aux path. Any
/// present input list must have exactly one entry per Activation.
struct ActAux<'a> {
    /// Per-ReLU `β·sign`, folded into the POST-slope coefficient (`lower −= β·sign`,
    /// `upper += β·sign`). Sound for any β≥0; the add is certified outward.
    beta_signed: Option<&'a [Vec<f32>]>,
    /// Per-ReLU masked pre-activation lower bounds → analytic alpha-gradient capture.
    grad_pre_lower: Option<&'a [Vec<f32>]>,
    /// Per-ReLU split-neuron column indices → pre-transform lower-A gather (β-grad).
    beta_gather_idx: Option<&'a [Vec<u32>]>,
    /// OUT: per-ReLU alpha gradients `pre_lower[i]·Σ_j max(A_lower_pre[j,i],0)`.
    relu_grads: Vec<Vec<f32>>,
    /// OUT: per-ReLU gathered pre-transform lower-A values, row-major `specs × |idx|`.
    beta_gather: Vec<Vec<f32>>,
    /// Shared fold cursor (advances once per Activation).
    cursor: usize,
}

impl<'a> ActAux<'a> {
    fn beta(signed: &'a [Vec<f32>]) -> Self {
        ActAux {
            beta_signed: Some(signed),
            grad_pre_lower: None,
            beta_gather_idx: None,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            cursor: 0,
        }
    }
    fn grad(pre_lower: &'a [Vec<f32>]) -> Self {
        ActAux {
            beta_signed: None,
            grad_pre_lower: Some(pre_lower),
            beta_gather_idx: None,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            cursor: 0,
        }
    }
    fn beta_grad(signed: &'a [Vec<f32>], gather_idx: &'a [Vec<u32>]) -> Self {
        ActAux {
            beta_signed: Some(signed),
            grad_pre_lower: None,
            beta_gather_idx: Some(gather_idx),
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            cursor: 0,
        }
    }

    fn beta_grad_alpha(
        signed: &'a [Vec<f32>],
        gather_idx: &'a [Vec<u32>],
        pre_lower: &'a [Vec<f32>],
    ) -> Self {
        ActAux {
            beta_signed: Some(signed),
            grad_pre_lower: Some(pre_lower),
            beta_gather_idx: Some(gather_idx),
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            cursor: 0,
        }
    }

    fn beta_alpha(signed: &'a [Vec<f32>], pre_lower: &'a [Vec<f32>]) -> Self {
        ActAux {
            beta_signed: Some(signed),
            grad_pre_lower: Some(pre_lower),
            beta_gather_idx: None,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            cursor: 0,
        }
    }
}

/// Propagate a frontier BACKWARD through a plain layer sub-chain (Linear/Activation),
/// WITHOUT concretizing — the shared primitive for both the whole-net backward and a
/// resnet sub-branch. `UnsupportedOp` on any other layer kind (⇒ CPU fallback). When
/// `beta` is `Some`, the per-ReLU β-CROWN dual term is folded in (sound for any β≥0;
/// the add is over-bounded outward in the certified error).
fn backward_layers_f64<E: GemmEngine + ?Sized>(
    eng: &E,
    layers: &[GpuCrownLayer],
    f: Frontier,
    num_specs: usize,
    aux: &mut Option<ActAux<'_>>,
) -> Result<Frontier> {
    let Frontier {
        mut lower_a,
        mut upper_a,
        mut lower_err,
        mut upper_err,
        mut lb,
        mut ub,
        mut lb_err,
        mut ub_err,
        mut dim,
    } = f;

    for layer in layers {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
            } => {
                if dim != *out_features {
                    return Err(NyError::shape_mismatch(vec![*out_features], vec![dim]));
                }
                if let Some(bias) = bias {
                    bias_fold_f64(
                        num_specs,
                        *out_features,
                        &lower_a,
                        &lower_err,
                        bias,
                        &mut lb,
                        &mut lb_err,
                    );
                    bias_fold_f64(
                        num_specs,
                        *out_features,
                        &upper_a,
                        &upper_err,
                        bias,
                        &mut ub,
                        &mut ub_err,
                    );
                }
                // Batch the lower + upper bounds into one set of (2·specs)-row
                // GEMMs: 3 bigger cuBLAS calls per layer instead of 6 (better GPU
                // utilization, fewer launches than the per-bound CPU path).
                let two = 2 * num_specs;
                let mut a_stack = Vec::with_capacity(two * out_features);
                a_stack.extend_from_slice(&lower_a);
                a_stack.extend_from_slice(&upper_a);
                let mut e_stack = Vec::with_capacity(two * out_features);
                e_stack.extend_from_slice(&lower_err);
                e_stack.extend_from_slice(&upper_err);
                let (na, ne) = linear_step_f64(
                    eng,
                    two,
                    *out_features,
                    *in_features,
                    &a_stack,
                    &e_stack,
                    weight,
                )?;
                let half = num_specs * in_features;
                lower_a = na[..half].to_vec();
                upper_a = na[half..].to_vec();
                lower_err = ne[..half].to_vec();
                upper_err = ne[half..].to_vec();
                dim = *in_features;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                if dim != *num_neurons {
                    return Err(NyError::shape_mismatch(vec![*num_neurons], vec![dim]));
                }
                if *num_neurons == 0
                    || lower_slope.len() != upper_slope.len()
                    || lower_slope.len() != lower_intercept.len()
                    || lower_slope.len() != upper_intercept.len()
                    || !lower_slope.len().is_multiple_of(*num_neurons)
                {
                    return Err(NyError::InvalidSpec(
                        "cuda f64 activation: malformed domain-stacked relaxation table".into(),
                    ));
                }
                let activation_domains = lower_slope.len() / *num_neurons;
                if activation_domains == 0 || !num_specs.is_multiple_of(activation_domains) {
                    return Err(NyError::InvalidSpec(
                        "cuda f64 activation: rows do not partition into relaxation domains".into(),
                    ));
                }
                let specs_per_domain = num_specs / activation_domains;
                // Intercept fold: sign-routed `Σ_i a·intercept` into the running
                // bound. As in `bias_fold_f64`, the fold's own f64 rounding is
                // certified by `γ_{n+1}` over the computed magnitudes (incoming
                // bound included), on top of the propagated coefficient error
                // (`err·(|li|+|ui|)`, covering a sign flip of `a` under its error).
                let gamma = gamma_k_f64(*num_neurons + 1);
                let real_factor = 1.0 + 2.0 * gamma;
                let additive = 8.0 * ((*num_neurons + 1) as f64) * ETA64;
                for s in 0..num_specs {
                    let param_base = (s / specs_per_domain) * *num_neurons;
                    let mut lmag = lb[s].abs();
                    let mut umag = ub[s].abs();
                    for i in 0..*num_neurons {
                        let la = lower_a[s * num_neurons + i];
                        let ua = upper_a[s * num_neurons + i];
                        let p = param_base + i;
                        let li = if la >= 0.0 {
                            lower_intercept[p]
                        } else {
                            upper_intercept[p]
                        };
                        let ui = if ua >= 0.0 {
                            upper_intercept[p]
                        } else {
                            lower_intercept[p]
                        };
                        let lprod = la * f64::from(li);
                        let uprod = ua * f64::from(ui);
                        lb[s] += lprod;
                        ub[s] += uprod;
                        lmag += lprod.abs();
                        umag += uprod.abs();
                        let int_sum = f64::from(lower_intercept[p]).abs()
                            + f64::from(upper_intercept[p]).abs();
                        lb_err[s] += lower_err[s * num_neurons + i] * int_sum;
                        ub_err[s] += upper_err[s * num_neurons + i] * int_sum;
                    }
                    lb_err[s] += gamma * lmag * real_factor + additive;
                    ub_err[s] += gamma * umag * real_factor + additive;
                }
                // Non-soundness-critical captures from the PRE-transform lower coeff
                // (before the slope transform), in ReLU fold order.
                if let Some(a) = aux.as_mut() {
                    let r = a.cursor;
                    if let Some(pl) = a.grad_pre_lower {
                        // relu_grads[r][i] = pre_lower[r][i]·Σ_j max(A_lower[j,i], 0).
                        let plr = pl.get(r).ok_or_else(|| {
                            NyError::InvalidSpec(
                                "cuda grad resnet CROWN: relu_pre_lower shorter than the ReLU fold"
                                    .into(),
                            )
                        })?;
                        if plr.len() != activation_domains * *num_neurons {
                            return Err(NyError::shape_mismatch(
                                vec![activation_domains * *num_neurons],
                                vec![plr.len()],
                            ));
                        }
                        let mut g = vec![0.0f32; activation_domains * *num_neurons];
                        for d in 0..activation_domains {
                            for i in 0..*num_neurons {
                                let mut acc = 0.0f64;
                                let start = d * specs_per_domain;
                                for s in start..start + specs_per_domain {
                                    let v = lower_a[s * *num_neurons + i];
                                    if v > 0.0 {
                                        acc += v;
                                    }
                                }
                                let p = d * *num_neurons + i;
                                g[p] = (f64::from(plr[p]) * acc) as f32;
                            }
                        }
                        a.relu_grads.push(g);
                    }
                    if let Some(gidx) = a.beta_gather_idx {
                        // beta_gather[r]: row-major specs × |idx| of pre-transform A_lower.
                        let idxs = gidx.get(r).ok_or_else(|| {
                            NyError::InvalidSpec(
                                "cuda beta-grad resnet CROWN: gather_idx shorter than the ReLU fold"
                                    .into(),
                            )
                        })?;
                        let mut gath = Vec::with_capacity(num_specs * idxs.len());
                        for s in 0..num_specs {
                            for &col in idxs {
                                let c = col as usize;
                                if c >= *num_neurons {
                                    return Err(NyError::InvalidSpec(
                                        "cuda beta-grad resnet CROWN: gather col out of range"
                                            .into(),
                                    ));
                                }
                                gath.push(lower_a[s * *num_neurons + c] as f32);
                            }
                        }
                        a.beta_gather.push(gath);
                    }
                }

                let (nla, nua, nle, nue) = activation_step_f64(
                    num_specs,
                    *num_neurons,
                    &lower_a,
                    &upper_a,
                    &lower_err,
                    &upper_err,
                    lower_slope,
                    upper_slope,
                )?;
                lower_a = nla;
                upper_a = nua;
                lower_err = nle;
                upper_err = nue;
                // β-CROWN split-constraint dual, folded into the POST-slope
                // coefficient. Sound for ANY β≥0 (a valid Lagrangian dual); the f64
                // add is over-bounded outward (U64·|·|) into the certified error.
                if let Some(a) = aux.as_mut() {
                    if let Some(signed) = a.beta_signed {
                        let k = a.cursor;
                        let bk = signed.get(k).ok_or_else(|| {
                            NyError::InvalidSpec(
                                "cuda beta resnet CROWN: beta_signed shorter than the ReLU fold"
                                    .into(),
                            )
                        })?;
                        if bk.len() != activation_domains * *num_neurons {
                            return Err(NyError::shape_mismatch(
                                vec![activation_domains * *num_neurons],
                                vec![bk.len()],
                            ));
                        }
                        for s in 0..num_specs {
                            let param_base = (s / specs_per_domain) * *num_neurons;
                            for i in 0..*num_neurons {
                                let idx = s * *num_neurons + i;
                                let b = f64::from(bk[param_base + i]);
                                lower_a[idx] -= b;
                                upper_a[idx] += b;
                                lower_err[idx] += U64 * lower_a[idx].abs();
                                upper_err[idx] += U64 * upper_a[idx].abs();
                            }
                        }
                    }
                    // One Activation processed: advance the shared fold cursor.
                    a.cursor += 1;
                }
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                in_h,
                in_w,
            } => {
                let (oc, ic, kh, kw) = (*out_channels, *in_channels, *kernel_h, *kernel_w);
                let out_d = oc * out_h * out_w;
                if dim != out_d {
                    return Err(NyError::shape_mismatch(vec![out_d], vec![dim]));
                }
                // Fold the per-output bias into the running bound (like Linear).
                if let Some(bias) = bias_expanded {
                    if bias.len() != out_d {
                        return Err(NyError::shape_mismatch(vec![out_d], vec![bias.len()]));
                    }
                    bias_fold_f64(
                        num_specs,
                        out_d,
                        &lower_a,
                        &lower_err,
                        bias,
                        &mut lb,
                        &mut lb_err,
                    );
                    bias_fold_f64(
                        num_specs,
                        out_d,
                        &upper_a,
                        &upper_err,
                        bias,
                        &mut ub,
                        &mut ub_err,
                    );
                }
                // Batch lower + upper rows through the transposed-conv coefficient map.
                let mut a_stack = Vec::with_capacity(2 * num_specs * out_d);
                a_stack.extend_from_slice(&lower_a);
                a_stack.extend_from_slice(&upper_a);
                let mut e_stack = Vec::with_capacity(2 * num_specs * out_d);
                e_stack.extend_from_slice(&lower_err);
                e_stack.extend_from_slice(&upper_err);
                let (na, ne) = conv_step_f64(
                    eng,
                    2 * num_specs,
                    &a_stack,
                    &e_stack,
                    weight_col,
                    oc,
                    ic,
                    kh,
                    kw,
                    *stride_h,
                    *stride_w,
                    *pad_h,
                    *pad_w,
                    *out_h,
                    *out_w,
                    *in_h,
                    *in_w,
                )?;
                let in_d = ic * in_h * in_w;
                let half = num_specs * in_d;
                lower_a = na[..half].to_vec();
                upper_a = na[half..].to_vec();
                lower_err = ne[..half].to_vec();
                upper_err = ne[half..].to_vec();
                dim = in_d;
            }
            _ => {
                return Err(NyError::UnsupportedOp(
                    "cuda f64 sound CROWN: Linear/Activation/Conv2d only (CPU fallback otherwise)"
                        .into(),
                ));
            }
        }
    }

    Ok(Frontier {
        lower_a,
        upper_a,
        lower_err,
        upper_err,
        lb,
        ub,
        lb_err,
        ub_err,
        dim,
    })
}

/// Concretize a FINAL frontier against the input box — the last backward step,
/// folding the accumulated coefficient error into the bound via `|input|`.
fn concretize_frontier(
    f: &Frontier,
    num_specs: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> GpuCrownResult {
    let (lower_bounds, upper_bounds) = concretize_f64(
        num_specs,
        f.dim,
        &f.lower_a,
        &f.upper_a,
        &f.lower_err,
        &f.upper_err,
        input_lower,
        input_upper,
        &f.lb,
        &f.ub,
        &f.lb_err,
        &f.ub_err,
    );
    GpuCrownResult {
        lower_bounds,
        upper_bounds,
    }
}

/// Shared f64 CROWN backward from an ALREADY-INITIALISED frontier (spec or seed):
/// run the plain sub-chain then concretize. Sound by construction (ONE proven
/// backward). The frontier is treated as EXACT (zero error to start); only the
/// suffix's own f64 rounding is tracked and certified OUTWARD at concretization.
#[allow(clippy::too_many_arguments)]
fn run_backward_f64_from<E: GemmEngine + ?Sized>(
    eng: &E,
    layers: &[GpuCrownLayer],
    lower_a: Vec<f64>,
    upper_a: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    dim: usize,
    num_specs: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Result<GpuCrownResult> {
    let init = Frontier {
        lower_err: vec![0.0f64; num_specs * dim],
        upper_err: vec![0.0f64; num_specs * dim],
        lb_err: vec![0.0f64; num_specs],
        ub_err: vec![0.0f64; num_specs],
        lower_a,
        upper_a,
        lb,
        ub,
        dim,
    };
    let f = backward_layers_f64(eng, layers, init, num_specs, &mut None)?;
    Ok(concretize_frontier(&f, num_specs, input_lower, input_upper))
}

/// Gate-free SOUND f64 RESNET CROWN backward (T1.3): propagate the alpha-suffix
/// `seed` frontier BACKWARD through the decomposed `segments` (plain chains +
/// identity/projection residual blocks), carrying the certified error ACROSS
/// segment/residual-block boundaries, then concretize ONCE at the input.
///
/// Residual composition (proven-sound, mirrors the wgpu/CPU path):
/// - `Chain(F)`: frontier = backward(F, frontier).
/// - `Residual(F)`: `out = F(z) + z` ⇒ A_z = backward_F(A) + A (skip identity),
///   bias `b_F + b_out`, errors summed + merge-rounding certified.
/// - `ResidualProj(F,P)`: `out = F(z) + P(z)` ⇒ A_z = backward_F(A) + backward_P(A),
///   outer bias `b_out` counted ONCE.
///
/// Each sub-branch starts from the current COEFFICIENTS with ZERO bias
/// ([`Frontier::coeffs_zero_bias`]); the outer bias is re-added by the merge. This
/// backend handles the BASE (un-concretized-frontier) path — the verdict default for
/// non-exploding nets; the optional per-segment error concretization / auto-fallback
/// (`frontier_abs`/`node_abs`, gated on `NY_RESNET_ERR_CONCRETIZE*`) is a TIGHTENING
/// the wgpu path adds for exploding nets, so ignoring it here stays sound (looser).
fn resnet_fold_f64_core<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    // Per-Activation auxiliaries (β fold / grad / gather), threaded in fold order
    // across ALL segment branches (F before P) by one shared cursor. The captured
    // `relu_grads`/`beta_gather` are read back from `aux` by the caller.
    aux: &mut Option<ActAux<'_>>,
) -> Result<Frontier> {
    use ny_core::GpuResnetSegment;
    let num_specs = seed.num_specs;
    let dim = seed.current_dim;
    if num_specs == 0 {
        return Ok(Frontier {
            lower_a: Vec::new(),
            upper_a: Vec::new(),
            lower_err: Vec::new(),
            upper_err: Vec::new(),
            lb: Vec::new(),
            ub: Vec::new(),
            lb_err: Vec::new(),
            ub_err: Vec::new(),
            dim,
        });
    }
    let expect_a = num_specs.checked_mul(dim).ok_or_else(|| {
        NyError::InvalidSpec("cuda resnet sound CROWN: seed size overflow".into())
    })?;
    if seed.lower_a.len() != expect_a || seed.upper_a.len() != expect_a {
        return Err(NyError::shape_mismatch(
            vec![expect_a],
            vec![seed.lower_a.len()],
        ));
    }
    if seed.lower_b.len() != num_specs || seed.upper_b.len() != num_specs {
        return Err(NyError::shape_mismatch(
            vec![num_specs],
            vec![seed.lower_b.len()],
        ));
    }

    let mut f = Frontier {
        lower_a: seed.lower_a.iter().map(|&x| f64::from(x)).collect(),
        upper_a: seed.upper_a.iter().map(|&x| f64::from(x)).collect(),
        lower_err: vec![0.0f64; expect_a],
        upper_err: vec![0.0f64; expect_a],
        lb: seed.lower_b.iter().map(|&x| f64::from(x)).collect(),
        ub: seed.upper_b.iter().map(|&x| f64::from(x)).collect(),
        lb_err: vec![0.0f64; num_specs],
        ub_err: vec![0.0f64; num_specs],
        dim,
    };

    for seg in segments {
        match seg {
            GpuResnetSegment::Chain(layers) => {
                f = backward_layers_f64(eng, layers, f, num_specs, aux)?;
            }
            GpuResnetSegment::Residual(fbranch) => {
                // out = F(z) + z: branch through F from the coeffs (zero bias), then
                // add the identity skip's coeffs (the incoming A) back in.
                let mut branch = backward_layers_f64(
                    eng,
                    fbranch,
                    f.coeffs_zero_bias(num_specs),
                    num_specs,
                    aux,
                )?;
                if branch.dim != f.dim {
                    return Err(NyError::shape_mismatch(vec![f.dim], vec![branch.dim]));
                }
                merge_add(
                    &mut branch.lower_a,
                    &f.lower_a,
                    &mut branch.lower_err,
                    &f.lower_err,
                );
                merge_add(
                    &mut branch.upper_a,
                    &f.upper_a,
                    &mut branch.upper_err,
                    &f.upper_err,
                );
                // Outer bias b_out (in f.lb/ub) added once; skip contributes none.
                merge_add(&mut branch.lb, &f.lb, &mut branch.lb_err, &f.lb_err);
                merge_add(&mut branch.ub, &f.ub, &mut branch.ub_err, &f.ub_err);
                f = branch;
            }
            GpuResnetSegment::ResidualProj(fbranch, pbranch) => {
                // out = F(z) + P(z): both branches from the coeffs (zero bias); sum
                // their coeffs+biases; the outer bias b_out is added ONCE.
                let mut bf = backward_layers_f64(
                    eng,
                    fbranch,
                    f.coeffs_zero_bias(num_specs),
                    num_specs,
                    aux,
                )?;
                let bp = backward_layers_f64(
                    eng,
                    pbranch,
                    f.coeffs_zero_bias(num_specs),
                    num_specs,
                    aux,
                )?;
                if bf.dim != bp.dim {
                    return Err(NyError::shape_mismatch(vec![bf.dim], vec![bp.dim]));
                }
                merge_add(
                    &mut bf.lower_a,
                    &bp.lower_a,
                    &mut bf.lower_err,
                    &bp.lower_err,
                );
                merge_add(
                    &mut bf.upper_a,
                    &bp.upper_a,
                    &mut bf.upper_err,
                    &bp.upper_err,
                );
                merge_add(&mut bf.lb, &bp.lb, &mut bf.lb_err, &bp.lb_err);
                merge_add(&mut bf.ub, &bp.ub, &mut bf.ub_err, &bp.ub_err);
                // Outer bias b_out (in f.lb/ub), counted once.
                merge_add(&mut bf.lb, &f.lb, &mut bf.lb_err, &f.lb_err);
                merge_add(&mut bf.ub, &f.ub, &mut bf.ub_err, &f.ub_err);
                f = bf;
            }
        }
    }

    // Every present per-ReLU aux list must be consumed exactly once (fold-order
    // agreement between the caller's lists and the ReLUs actually processed).
    if let Some(a) = aux.as_ref() {
        let check = |name: &str, len: usize| -> Result<()> {
            if a.cursor != len {
                return Err(NyError::InvalidSpec(format!(
                    "cuda resnet CROWN: consumed {} of {len} {name} entries (fold mismatch)",
                    a.cursor
                )));
            }
            Ok(())
        };
        if let Some(s) = a.beta_signed {
            check("beta_signed", s.len())?;
        }
        if let Some(p) = a.grad_pre_lower {
            check("relu_pre_lower", p.len())?;
        }
        if let Some(g) = a.beta_gather_idx {
            check("gather_idx", g.len())?;
        }
    }

    Ok(f)
}

/// Serial wrapper around [`resnet_fold_f64_core`].  Keeping concretization out of
/// the fold is what lets the CUDA-wide path slice the common coefficient frontier
/// and discharge each domain against its own input box without approximation.
fn resnet_backward_f64_core<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    aux: &mut Option<ActAux<'_>>,
) -> Result<GpuCrownResult> {
    let f = resnet_fold_f64_core(eng, segments, seed, aux)?;
    Ok(concretize_frontier(
        &f,
        seed.num_specs,
        input_lower,
        input_upper,
    ))
}

struct WideResnetBatch {
    segments: Vec<GpuResnetSegment>,
    seed: GpuCrownSeed,
    beta_signed: Vec<Vec<f32>>,
    input_lower: Vec<f32>,
    input_upper: Vec<f32>,
    n_domains: usize,
    specs_per_domain: usize,
    input_dim: usize,
}

fn finite_slice(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn validate_wide_layer_finite(layer: &GpuCrownLayer) -> bool {
    match layer {
        GpuCrownLayer::Linear { weight, bias, .. } => {
            finite_slice(weight) && bias.as_ref().is_none_or(|b| finite_slice(b))
        }
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            ..
        } => {
            finite_slice(lower_slope)
                && finite_slice(upper_slope)
                && finite_slice(lower_intercept)
                && finite_slice(upper_intercept)
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            ..
        } => finite_slice(weight_col) && bias_expanded.as_ref().is_none_or(|b| finite_slice(b)),
        _ => false,
    }
}

fn validate_wide_segments_finite(segments: &[GpuResnetSegment]) -> bool {
    let layers_ok = |layers: &[GpuCrownLayer]| layers.iter().all(validate_wide_layer_finite);
    segments.iter().all(|segment| match segment {
        GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => layers_ok(l),
        GpuResnetSegment::ResidualProj(f, p) => layers_ok(f) && layers_ok(p),
    })
}

/// Assemble domain-major rows for Hydra's CUDA proof forest.  This is all
/// validation and copying; any mismatch returns `Err`, causing the caller to use
/// the pre-existing WGPU/serial sound path rather than risk a ragged wide fold.
fn prepare_wide_resnet_batch(
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    skeleton_prevalidated: bool,
) -> Result<WideResnetBatch> {
    let first = domains
        .first()
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: empty domain batch".into()))?;
    let n_domains = domains.len();
    let specs_per_domain = seed.num_specs;
    if specs_per_domain == 0 || seed.current_dim == 0 {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: zero specification rows or output width".into(),
        ));
    }
    let expected_seed_a = specs_per_domain
        .checked_mul(seed.current_dim)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: seed size overflow".into()))?;
    if seed.lower_a.len() != expected_seed_a
        || seed.upper_a.len() != expected_seed_a
        || seed.lower_b.len() != specs_per_domain
        || seed.upper_b.len() != specs_per_domain
        || !finite_slice(&seed.lower_a)
        || !finite_slice(&seed.upper_a)
        || !finite_slice(&seed.lower_b)
        || !finite_slice(&seed.upper_b)
    {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: malformed or non-finite seed".into(),
        ));
    }
    if !segments_are_wide_batchable(first.segments) {
        return Err(NyError::UnsupportedOp(
            "cuda wide resnet: only Linear/Activation/Conv2d segments are batchable".into(),
        ));
    }
    if !skeleton_prevalidated {
        for domain in &domains[1..] {
            if !resnet_skeleton_matches(first.segments, domain.segments) {
                return Err(NyError::UnsupportedOp(
                    "cuda wide resnet: heterogeneous network skeleton".into(),
                ));
            }
        }
    }

    let segments = stack_wide_segments(domains).ok_or_else(|| {
        NyError::InvalidSpec("cuda wide resnet: failed to stack activation tables".into())
    })?;
    if !validate_wide_segments_finite(&segments) {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: non-finite network or relaxation value".into(),
        ));
    }

    let beta_tables: Vec<&[Vec<f32>]> = domains.iter().map(|d| d.beta_signed).collect();
    let beta_signed = stack_wide_table(&beta_tables)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: malformed beta table".into()))?;
    if beta_signed.iter().any(|row| !finite_slice(row)) {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: non-finite beta table".into(),
        ));
    }

    let input_dim = first.input_lower.len();
    if input_dim == 0 {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: empty input box".into(),
        ));
    }
    let total_rows = n_domains
        .checked_mul(specs_per_domain)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: row count overflow".into()))?;
    let total_a = total_rows
        .checked_mul(seed.current_dim)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: frontier size overflow".into()))?;
    let mut lower_a = Vec::new();
    let mut upper_a = Vec::new();
    let mut lower_b = Vec::new();
    let mut upper_b = Vec::new();
    lower_a
        .try_reserve_exact(total_a)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: total_a.saturating_mul(size_of::<f32>()),
            budget_bytes: usize::MAX,
            site: "cuda::prepare_wide_resnet_batch/lower_a",
        })?;
    upper_a
        .try_reserve_exact(total_a)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: total_a.saturating_mul(size_of::<f32>()),
            budget_bytes: usize::MAX,
            site: "cuda::prepare_wide_resnet_batch/upper_a",
        })?;
    lower_b
        .try_reserve_exact(total_rows)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: total_rows.saturating_mul(size_of::<f32>()),
            budget_bytes: usize::MAX,
            site: "cuda::prepare_wide_resnet_batch/lower_b",
        })?;
    upper_b
        .try_reserve_exact(total_rows)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: total_rows.saturating_mul(size_of::<f32>()),
            budget_bytes: usize::MAX,
            site: "cuda::prepare_wide_resnet_batch/upper_b",
        })?;
    for _ in 0..n_domains {
        lower_a.extend_from_slice(&seed.lower_a);
        upper_a.extend_from_slice(&seed.upper_a);
        lower_b.extend_from_slice(&seed.lower_b);
        upper_b.extend_from_slice(&seed.upper_b);
    }

    let total_input = n_domains
        .checked_mul(input_dim)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: input size overflow".into()))?;
    let mut input_lower = Vec::with_capacity(total_input);
    let mut input_upper = Vec::with_capacity(total_input);
    for domain in domains {
        if domain.input_lower.len() != input_dim
            || domain.input_upper.len() != input_dim
            || !finite_slice(domain.input_lower)
            || !finite_slice(domain.input_upper)
            || domain
                .input_lower
                .iter()
                .zip(domain.input_upper)
                .any(|(l, u)| l > u)
        {
            return Err(NyError::InvalidSpec(
                "cuda wide resnet: malformed, non-finite, or inverted input box".into(),
            ));
        }
        input_lower.extend_from_slice(domain.input_lower);
        input_upper.extend_from_slice(domain.input_upper);
    }

    Ok(WideResnetBatch {
        segments,
        seed: GpuCrownSeed {
            lower_a: lower_a.into(),
            upper_a: upper_a.into(),
            lower_b: lower_b.into(),
            upper_b: upper_b.into(),
            num_specs: total_rows,
            current_dim: seed.current_dim,
        },
        beta_signed,
        input_lower,
        input_upper,
        n_domains,
        specs_per_domain,
        input_dim,
    })
}

/// Conservative process-wide cap for one CUDA proof-forest call.  The estimate
/// includes retained outputs plus a deliberately padded f64 working set.  Users
/// may lower this to fit a smaller accelerator; malformed values fail closed.
const DEFAULT_CUDA_WIDE_MAX_BYTES: usize = 512 * 1024 * 1024;
// Peak row-dependent storage, per maximum-width cell, is bounded by roughly:
//  32 B live frontier (lower/upper center+error),
//  32 B lower/upper stack + error stack,
//  16 B |A| staging,
//  32 B cached unified A/C buffers,
//  64 B simultaneous GEMM outputs/magnitude/error,
//  32 B residual-branch frontier overlap.
// That is 208 B; 256 B adds >23% headroom for vector headers, split copies, and
// backend bookkeeping without collapsing practical CIFAR/Tiny batches to D=1.
const CUDA_WIDE_BYTES_PER_ROW_CELL: usize = 256;
const CUDA_WIDE_BYTES_PER_WEIGHT: usize = 32;

fn parse_cuda_wide_max_bytes(raw: Option<&std::ffi::OsStr>) -> Result<usize> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_CUDA_WIDE_MAX_BYTES);
    };
    let raw = raw.to_str().ok_or_else(|| {
        NyError::InvalidSpec("NY_CUDA_WIDE_MAX_BYTES must be a positive base-10 integer".into())
    })?;
    // Keep the runtime grammar identical to the sealed measurement manifest:
    // non-empty ASCII decimal digits only. In particular, Rust's integer
    // parser accepts a leading `+`, while provenance intentionally rejects it.
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(NyError::InvalidSpec(
            "NY_CUDA_WIDE_MAX_BYTES must be a positive base-10 integer".into(),
        ));
    }
    raw.parse::<usize>().ok().filter(|&v| v > 0).ok_or_else(|| {
        NyError::InvalidSpec("NY_CUDA_WIDE_MAX_BYTES must be a positive base-10 integer".into())
    })
}

fn cuda_wide_max_bytes() -> Result<usize> {
    let raw = std::env::var_os("NY_CUDA_WIDE_MAX_BYTES");
    parse_cuda_wide_max_bytes(raw.as_deref())
}

#[derive(Default)]
struct WideMemoryShape {
    max_work_width: usize,
    max_weight_elems: usize,
    relu_neurons: usize,
}

fn update_wide_memory_shape(layers: &[GpuCrownLayer], shape: &mut WideMemoryShape) -> Result<()> {
    for layer in layers {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
            } => {
                shape.max_work_width = shape.max_work_width.max(*out_features).max(*in_features);
                shape.max_weight_elems = shape.max_weight_elems.max(
                    weight
                        .len()
                        .saturating_add(bias.as_ref().map_or(0, |b| b.len())),
                );
            }
            GpuCrownLayer::Activation { num_neurons, .. } => {
                shape.max_work_width = shape.max_work_width.max(*num_neurons);
                shape.relu_neurons =
                    shape
                        .relu_neurons
                        .checked_add(*num_neurons)
                        .ok_or_else(|| {
                            NyError::InvalidSpec("cuda wide resnet: ReLU size overflow".into())
                        })?;
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                out_h,
                out_w,
                in_h,
                in_w,
                ..
            } => {
                let output = out_channels
                    .checked_mul(*out_h)
                    .and_then(|v| v.checked_mul(*out_w))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("cuda wide resnet: conv output overflow".into())
                    })?;
                let input = in_channels
                    .checked_mul(*in_h)
                    .and_then(|v| v.checked_mul(*in_w))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("cuda wide resnet: conv input overflow".into())
                    })?;
                // im2col workspace per spec row; often larger than either tensor.
                let col = out_h
                    .checked_mul(*out_w)
                    .and_then(|v| v.checked_mul(*in_channels))
                    .and_then(|v| v.checked_mul(*kernel_h))
                    .and_then(|v| v.checked_mul(*kernel_w))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("cuda wide resnet: conv workspace overflow".into())
                    })?;
                shape.max_work_width = shape.max_work_width.max(output).max(input).max(col);
                shape.max_weight_elems = shape.max_weight_elems.max(
                    weight_col
                        .len()
                        .saturating_add(bias_expanded.as_ref().map_or(0, |b| b.len())),
                );
            }
            _ => {
                return Err(NyError::UnsupportedOp(
                    "cuda wide resnet: unbatchable layer in memory estimate".into(),
                ));
            }
        }
    }
    Ok(())
}

fn wide_memory_shape(segments: &[GpuResnetSegment], seed_dim: usize) -> Result<WideMemoryShape> {
    let mut shape = WideMemoryShape {
        max_work_width: seed_dim,
        ..WideMemoryShape::default()
    };
    for segment in segments {
        match segment {
            GpuResnetSegment::Chain(layers) | GpuResnetSegment::Residual(layers) => {
                update_wide_memory_shape(layers, &mut shape)?;
            }
            GpuResnetSegment::ResidualProj(f, p) => {
                update_wide_memory_shape(f, &mut shape)?;
                update_wide_memory_shape(p, &mut shape)?;
            }
        }
    }
    Ok(shape)
}

fn checked_size_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| NyError::InvalidSpec(format!("cuda wide resnet: {what} overflow")))
}

fn checked_size_add(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| NyError::InvalidSpec(format!("cuda wide resnet: {what} overflow")))
}

const CUDA_WIDE_CHUNK_PLAN_MARKER: &str = "NY_CUDA_WIDE_CHUNK_PLAN_V1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaWideChunkPlan {
    cap_bytes: usize,
    fixed_bytes: usize,
    per_domain_bytes: usize,
    requested_domains: usize,
    chunk_domains: usize,
}

fn cuda_wide_chunk_plan_line(plan: CudaWideChunkPlan) -> String {
    let CudaWideChunkPlan {
        cap_bytes,
        fixed_bytes,
        per_domain_bytes,
        requested_domains,
        chunk_domains,
    } = plan;
    format!(
        "{CUDA_WIDE_CHUNK_PLAN_MARKER} cap_bytes={cap_bytes} fixed_bytes={fixed_bytes} \
         per_domain_bytes={per_domain_bytes} requested_domains={requested_domains} \
         chunk_domains={chunk_domains}"
    )
}

/// Compute a device-safe domain chunk plan without emitting it. This performs
/// the whole-batch homogeneity check before chunking, so differing weights
/// cannot hide across a chunk boundary. Distinct-but-equal Arcs require a
/// by-value scan here; graph extraction should eventually intern them to make
/// this pointer-fast.
fn cuda_wide_chunk_plan_for_cap(
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    union_gather_idx: &[&[u32]],
    relu_pre_lower: &[&[Vec<f32>]],
    capture_coeff: bool,
    cap: usize,
) -> Result<CudaWideChunkPlan> {
    let first = domains
        .first()
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: empty domain batch".into()))?;
    if !segments_are_wide_batchable(first.segments) {
        return Err(NyError::UnsupportedOp(
            "cuda wide resnet: only Linear/Activation/Conv2d segments are batchable".into(),
        ));
    }
    for domain in &domains[1..] {
        if !resnet_skeleton_matches(first.segments, domain.segments) {
            return Err(NyError::UnsupportedOp(
                "cuda wide resnet: heterogeneous network skeleton".into(),
            ));
        }
    }
    if !relu_pre_lower.is_empty() && relu_pre_lower.len() != domains.len() {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: alpha table/domain count mismatch".into(),
        ));
    }
    let shape = wide_memory_shape(first.segments, seed.current_dim)?;
    if shape.max_work_width == 0 || seed.num_specs == 0 {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: zero row or working width".into(),
        ));
    }
    let total_rows = checked_size_mul(domains.len(), seed.num_specs, "total row count")?;
    let input_dim = first.input_lower.len();

    // Retained outputs live while every subsequent chunk runs.
    let mut retained = checked_size_mul(total_rows, 2 * size_of::<f32>(), "bound output")?;
    if !relu_pre_lower.is_empty() {
        retained = checked_size_add(
            retained,
            checked_size_mul(
                checked_size_mul(domains.len(), shape.relu_neurons, "alpha output")?,
                size_of::<f32>(),
                "alpha output bytes",
            )?,
            "retained alpha output",
        )?;
    }
    let gather_cols = union_gather_idx.iter().try_fold(0usize, |acc, idx| {
        checked_size_add(acc, idx.len(), "gather columns")
    })?;
    retained = checked_size_add(
        retained,
        checked_size_mul(
            checked_size_mul(total_rows, gather_cols, "gather output")?,
            size_of::<f32>(),
            "gather output bytes",
        )?,
        "retained gather output",
    )?;
    if capture_coeff {
        let coeff_cells = checked_size_mul(total_rows, input_dim, "captured coeff cells")?;
        // Four coefficient arrays and four scalar-bias arrays, all f32.
        let coeff_values = checked_size_add(
            checked_size_mul(coeff_cells, 4, "captured coeff values")?,
            checked_size_mul(total_rows, 4, "captured bias values")?,
            "captured trajectory values",
        )?;
        retained = checked_size_add(
            retained,
            checked_size_mul(coeff_values, size_of::<f32>(), "captured trajectory bytes")?,
            "retained trajectory output",
        )?;
    }
    let static_bytes = checked_size_mul(
        shape.max_weight_elems,
        CUDA_WIDE_BYTES_PER_WEIGHT,
        "static weight workspace",
    )?;
    let per_domain_cells = checked_size_mul(
        seed.num_specs,
        shape.max_work_width,
        "per-domain working cells",
    )?;
    let mut per_domain = checked_size_mul(
        per_domain_cells,
        CUDA_WIDE_BYTES_PER_ROW_CELL,
        "per-domain working bytes",
    )?;
    // Domain-stacked activation/β/intercept tables and gradient staging.
    per_domain = checked_size_add(
        per_domain,
        checked_size_mul(shape.relu_neurons, 64, "per-domain activation workspace")?,
        "per-domain workspace",
    )?;
    let fixed = checked_size_add(retained, static_bytes, "fixed wide memory")?;
    let chunk_domains = cap
        .checked_sub(fixed)
        .map_or(0, |available| available / per_domain.max(1))
        .min(domains.len());
    Ok(CudaWideChunkPlan {
        cap_bytes: cap,
        fixed_bytes: fixed,
        per_domain_bytes: per_domain,
        requested_domains: domains.len(),
        chunk_domains,
    })
}

fn cuda_wide_chunk_domains_for_cap(
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    union_gather_idx: &[&[u32]],
    relu_pre_lower: &[&[Vec<f32>]],
    capture_coeff: bool,
    cap: usize,
) -> Result<usize> {
    let plan = cuda_wide_chunk_plan_for_cap(
        domains,
        seed,
        union_gather_idx,
        relu_pre_lower,
        capture_coeff,
        cap,
    )?;
    // Exactly one decision line per logical wide call, including cap rejection.
    eprintln!("{}", cuda_wide_chunk_plan_line(plan));
    if cap < plan.fixed_bytes {
        let fixed = plan.fixed_bytes;
        return Err(NyError::UnsupportedOp(format!(
            "cuda wide resnet: retained/static estimate {fixed} exceeds {cap}-byte cap"
        )));
    }
    if plan.chunk_domains == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "cuda wide resnet: one domain exceeds {cap}-byte conservative memory cap"
        )));
    }
    Ok(plan.chunk_domains)
}

fn validate_resident_coeff(coeff: &GpuResidentCoeffBatched) -> Result<()> {
    let expected_a = coeff
        .num_specs
        .checked_mul(coeff.dim)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide trajectory: coeff overflow".into()))?;
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    let errors_valid = |v: &[f32]| v.iter().all(|x| x.is_finite() && *x >= 0.0);
    if coeff.dim == 0
        || coeff.num_specs == 0
        || coeff.num_specs_per_dom == 0
        || !coeff.num_specs.is_multiple_of(coeff.num_specs_per_dom)
        || coeff.lower_a.len() != expected_a
        || coeff.upper_a.len() != expected_a
        || coeff.lower_err.len() != expected_a
        || coeff.upper_err.len() != expected_a
        || coeff.lower_b.len() != coeff.num_specs
        || coeff.upper_b.len() != coeff.num_specs
        || coeff.lower_b_err.len() != coeff.num_specs
        || coeff.upper_b_err.len() != coeff.num_specs
        || !finite(&coeff.lower_a)
        || !finite(&coeff.upper_a)
        || !finite(&coeff.lower_b)
        || !finite(&coeff.upper_b)
        || !errors_valid(&coeff.lower_err)
        || !errors_valid(&coeff.upper_err)
        || !errors_valid(&coeff.lower_b_err)
        || !errors_valid(&coeff.upper_b_err)
    {
        return Err(NyError::InvalidSpec(
            "cuda wide trajectory: malformed captured coefficient frontier".into(),
        ));
    }
    Ok(())
}

fn append_resident_coeff(
    dst: &mut Option<GpuResidentCoeffBatched>,
    mut src: GpuResidentCoeffBatched,
) -> Result<()> {
    validate_resident_coeff(&src)?;
    let Some(dst) = dst.as_mut() else {
        *dst = Some(src);
        return Ok(());
    };
    validate_resident_coeff(dst)?;
    if dst.dim != src.dim || dst.num_specs_per_dom != src.num_specs_per_dom {
        return Err(NyError::InvalidSpec(
            "cuda wide trajectory: incompatible coefficient chunks".into(),
        ));
    }
    macro_rules! reserve_append {
        ($field:ident) => {
            dst.$field
                .try_reserve_exact(src.$field.len())
                .map_err(|_| NyError::CpuMemoryExceeded {
                    required_bytes: dst
                        .$field
                        .len()
                        .saturating_add(src.$field.len())
                        .saturating_mul(size_of::<f32>()),
                    budget_bytes: usize::MAX,
                    site: "cuda::wide_trajectory/append",
                })?;
        };
    }
    reserve_append!(lower_a);
    reserve_append!(upper_a);
    reserve_append!(lower_err);
    reserve_append!(upper_err);
    reserve_append!(lower_b);
    reserve_append!(upper_b);
    reserve_append!(lower_b_err);
    reserve_append!(upper_b_err);
    dst.lower_a.append(&mut src.lower_a);
    dst.upper_a.append(&mut src.upper_a);
    dst.lower_err.append(&mut src.lower_err);
    dst.upper_err.append(&mut src.upper_err);
    dst.lower_b.append(&mut src.lower_b);
    dst.upper_b.append(&mut src.upper_b);
    dst.lower_b_err.append(&mut src.lower_b_err);
    dst.upper_b_err.append(&mut src.upper_b_err);
    dst.num_specs = dst
        .num_specs
        .checked_add(src.num_specs)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide trajectory: row overflow".into()))?;
    validate_resident_coeff(dst)
}

#[allow(clippy::type_complexity)]
fn resnet_backward_f64_wide_chunked_with_cap<E: GemmEngine + ?Sized>(
    eng: &E,
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    union_gather_idx: &[&[u32]],
    relu_pre_lower: &[&[Vec<f32>]],
    enforce_size_gate: bool,
    capture_coeff: bool,
    cap: usize,
) -> Result<(
    Vec<GpuCrownResult>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Option<GpuResidentCoeffBatched>,
)> {
    let chunk_domains = cuda_wide_chunk_domains_for_cap(
        domains,
        seed,
        union_gather_idx,
        relu_pre_lower,
        capture_coeff,
        cap,
    )?;
    let total_rows = domains
        .len()
        .checked_mul(seed.num_specs)
        .ok_or_else(|| NyError::InvalidSpec("cuda wide resnet: row count overflow".into()))?;
    if enforce_size_gate && resnet_max_macs(domains[0].segments, total_rows) < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "cuda wide beta resnet CROWN: batch below GPU size-gate".into(),
        ));
    }

    let mut bounds = Vec::new();
    bounds
        .try_reserve_exact(domains.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: domains.len().saturating_mul(size_of::<GpuCrownResult>()),
            budget_bytes: cap,
            site: "cuda::wide_trajectory/bounds",
        })?;
    let mut alpha_grads: Vec<Vec<f32>> = Vec::new();
    let mut beta_gather: Vec<Vec<f32>> = Vec::new();
    let mut coeff = None;
    for start in (0..domains.len()).step_by(chunk_domains) {
        let end = (start + chunk_domains).min(domains.len());
        let pre_lower_chunk = if relu_pre_lower.is_empty() {
            &[][..]
        } else {
            &relu_pre_lower[start..end]
        };
        let mut chunk_coeff = empty_resident_coeff();
        let coeff_out = capture_coeff.then_some(&mut chunk_coeff);
        let (chunk_bounds, chunk_alpha, chunk_gather) = resnet_backward_f64_wide_core(
            eng,
            &domains[start..end],
            seed,
            union_gather_idx,
            pre_lower_chunk,
            false,
            coeff_out,
            true,
        )?;
        bounds.extend(chunk_bounds);
        if alpha_grads.is_empty() {
            alpha_grads = chunk_alpha;
        } else {
            if alpha_grads.len() != chunk_alpha.len() {
                return Err(NyError::InvalidSpec(
                    "cuda wide resnet: alpha chunk shape mismatch".into(),
                ));
            }
            for (dst, mut src) in alpha_grads.iter_mut().zip(chunk_alpha) {
                dst.append(&mut src);
            }
        }
        if beta_gather.is_empty() {
            beta_gather = chunk_gather;
        } else {
            if beta_gather.len() != chunk_gather.len() {
                return Err(NyError::InvalidSpec(
                    "cuda wide resnet: gather chunk shape mismatch".into(),
                ));
            }
            for (dst, mut src) in beta_gather.iter_mut().zip(chunk_gather) {
                dst.append(&mut src);
            }
        }
        if capture_coeff {
            append_resident_coeff(&mut coeff, chunk_coeff)?;
        }
    }
    if bounds.len() != domains.len() {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: bound chunk count mismatch".into(),
        ));
    }
    Ok((bounds, alpha_grads, beta_gather, coeff))
}

#[allow(clippy::type_complexity)]
fn resnet_backward_f64_wide_chunked<E: GemmEngine + ?Sized>(
    eng: &E,
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    union_gather_idx: &[&[u32]],
    relu_pre_lower: &[&[Vec<f32>]],
    capture_coeff: bool,
) -> Result<(
    Vec<GpuCrownResult>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Option<GpuResidentCoeffBatched>,
)> {
    resnet_backward_f64_wide_chunked_with_cap(
        eng,
        domains,
        seed,
        union_gather_idx,
        relu_pre_lower,
        true,
        capture_coeff,
        cuda_wide_max_bytes()?,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn resnet_backward_f64_wide_core<E: GemmEngine + ?Sized>(
    eng: &E,
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    union_gather_idx: &[&[u32]],
    relu_pre_lower: &[&[Vec<f32>]],
    enforce_size_gate: bool,
    coeff_full_out: Option<&mut GpuResidentCoeffBatched>,
    skeleton_prevalidated: bool,
) -> Result<(Vec<GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let wide = prepare_wide_resnet_batch(domains, seed, skeleton_prevalidated)?;
    if enforce_size_gate && resnet_max_macs(&wide.segments, wide.seed.num_specs) < MIN_RESIDENT_MACS
    {
        return Err(NyError::UnsupportedOp(
            "cuda wide beta resnet CROWN: batch below GPU size-gate".into(),
        ));
    }

    let gather_owned: Vec<Vec<u32>> = union_gather_idx.iter().map(|v| v.to_vec()).collect();
    let pre_lower_owned = if relu_pre_lower.is_empty() {
        Vec::new()
    } else {
        if relu_pre_lower.len() != wide.n_domains {
            return Err(NyError::InvalidSpec(
                "cuda wide resnet: alpha table/domain count mismatch".into(),
            ));
        }
        stack_wide_table(relu_pre_lower).ok_or_else(|| {
            NyError::InvalidSpec("cuda wide resnet: malformed alpha pre-lower table".into())
        })?
    };
    if pre_lower_owned.iter().any(|v| !finite_slice(v)) {
        return Err(NyError::InvalidSpec(
            "cuda wide resnet: non-finite alpha pre-lower table".into(),
        ));
    }

    let mut aux = Some(
        match (gather_owned.is_empty(), pre_lower_owned.is_empty()) {
            (true, true) => ActAux::beta(&wide.beta_signed),
            (false, true) => ActAux::beta_grad(&wide.beta_signed, &gather_owned),
            (true, false) => ActAux::beta_alpha(&wide.beta_signed, &pre_lower_owned),
            (false, false) => {
                ActAux::beta_grad_alpha(&wide.beta_signed, &gather_owned, &pre_lower_owned)
            }
        },
    );
    let frontier = resnet_fold_f64_core(eng, &wide.segments, &wide.seed, &mut aux)?;
    if frontier.dim != wide.input_dim {
        return Err(NyError::shape_mismatch(
            vec![wide.input_dim],
            vec![frontier.dim],
        ));
    }
    if let Some(out) = coeff_full_out {
        *out = frontier_to_resident_coeff(&frontier, wide.seed.num_specs, wide.specs_per_domain)?;
    }

    let rows = wide.specs_per_domain;
    let mut results = Vec::with_capacity(wide.n_domains);
    for domain in 0..wide.n_domains {
        let row_start = domain * rows;
        let row_end = row_start + rows;
        let coeff_start = row_start * frontier.dim;
        let coeff_end = row_end * frontier.dim;
        let box_start = domain * wide.input_dim;
        let box_end = box_start + wide.input_dim;
        let (lower_bounds, upper_bounds) = concretize_f64(
            rows,
            frontier.dim,
            &frontier.lower_a[coeff_start..coeff_end],
            &frontier.upper_a[coeff_start..coeff_end],
            &frontier.lower_err[coeff_start..coeff_end],
            &frontier.upper_err[coeff_start..coeff_end],
            &wide.input_lower[box_start..box_end],
            &wide.input_upper[box_start..box_end],
            &frontier.lb[row_start..row_end],
            &frontier.ub[row_start..row_end],
            &frontier.lb_err[row_start..row_end],
            &frontier.ub_err[row_start..row_end],
        );
        if lower_bounds
            .iter()
            .chain(&upper_bounds)
            .any(|v| !v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "cuda wide resnet: non-finite concretized bound".into(),
            ));
        }
        results.push(GpuCrownResult {
            lower_bounds,
            upper_bounds,
        });
    }

    let aux = aux.expect("wide CROWN always installs activation auxiliaries");
    Ok((results, aux.relu_grads, aux.beta_gather))
}

/// Gate-free SOUND f64 seeded CROWN backward: the alpha-CROWN suffix counterpart of
/// [`backward_f64_core`]. Initialises the backward from the alpha-suffix frontier
/// `seed` (distinct lower/upper coefficient rows + biases) INSTEAD of the spec
/// identity, then runs the SAME proven f64 layer backward. The seed is treated as
/// EXACT (its f32→f64 widening loses nothing, and the CPU sound suffix likewise
/// carries no coefficient-error frontier), so only the suffix's own f64 rounding is
/// tracked — the returned bounds are a sound enclosure of `A·f(x)+b` over the box.
fn seeded_backward_f64_core<E: GemmEngine + ?Sized>(
    eng: &E,
    layers: &[GpuCrownLayer],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Result<GpuCrownResult> {
    let num_specs = seed.num_specs;
    let dim = seed.current_dim;
    if num_specs == 0 {
        return Ok(GpuCrownResult {
            lower_bounds: vec![],
            upper_bounds: vec![],
        });
    }
    let expect_a = num_specs.checked_mul(dim).ok_or_else(|| {
        NyError::InvalidSpec("cuda seeded sound CROWN: seed size overflow".into())
    })?;
    if seed.lower_a.len() != expect_a || seed.upper_a.len() != expect_a {
        return Err(NyError::shape_mismatch(
            vec![expect_a],
            vec![seed.lower_a.len()],
        ));
    }
    if seed.lower_b.len() != num_specs || seed.upper_b.len() != num_specs {
        return Err(NyError::shape_mismatch(
            vec![num_specs],
            vec![seed.lower_b.len()],
        ));
    }
    let lower_a: Vec<f64> = seed.lower_a.iter().map(|&x| f64::from(x)).collect();
    let upper_a: Vec<f64> = seed.upper_a.iter().map(|&x| f64::from(x)).collect();
    let lb: Vec<f64> = seed.lower_b.iter().map(|&x| f64::from(x)).collect();
    let ub: Vec<f64> = seed.upper_b.iter().map(|&x| f64::from(x)).collect();
    run_backward_f64_from(
        eng,
        layers,
        lower_a,
        upper_a,
        lb,
        ub,
        dim,
        num_specs,
        input_lower,
        input_upper,
    )
}

/// SOUND f64-native GPU-resident SEEDED CROWN backward with the same small-net
/// size-gate as [`crown_backward_gpu_sound_impl`]. Below the gate → `UnsupportedOp`
/// so the caller keeps the CPU sound suffix (where the GPU would lose to overhead).
pub(crate) fn crown_backward_gpu_seeded_sound_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    layers: &[GpuCrownLayer],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Result<GpuCrownResult> {
    let max_macs: u128 = layers
        .iter()
        .map(|l| layer_macs(l, seed.num_specs))
        .max()
        .unwrap_or(0);
    if max_macs < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "cuda seeded sound CROWN: net below GPU size-gate (CPU is faster here)".into(),
        ));
    }
    seeded_backward_f64_core(eng, layers, seed, input_lower, input_upper)
}

/// SOUND f64-native GPU-resident RESNET-decomposed seeded CROWN backward (T1.3),
/// with the same small-net size-gate. Below the gate (or on an unsupported layer in
/// a segment) → `UnsupportedOp` so the caller keeps the CPU sound suffix. Handles
/// the BASE path; `frontier_abs`/`node_abs` (the exploding-net error-concretization
/// TIGHTENING) are accepted for signature parity but not required for soundness.
pub(crate) fn crown_backward_gpu_resnet_sound_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Result<GpuCrownResult> {
    if resnet_max_macs(segments, seed.num_specs) < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "cuda resnet sound CROWN: net below GPU size-gate (CPU is faster here)".into(),
        ));
    }
    resnet_backward_f64_core(eng, segments, seed, input_lower, input_upper, &mut None)
}

/// SOUND f64-native GPU-resident β-CROWN RESNET seeded backward (T1.3): the
/// [`crown_backward_gpu_resnet_sound_impl`] path with the per-domain β-CROWN split
/// dual `beta_signed` (per-ReLU `β·sign`, fold order: each branch's Activations in
/// order, F before P) folded into each POST-slope coefficient. A β-CROWN bound is a
/// valid Lagrangian dual for ANY β≥0, so this is SOUND regardless of the β values;
/// the fold add is over-bounded outward in the certified error. Same size-gate.
pub(crate) fn crown_backward_gpu_resnet_sound_beta_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    beta_signed: &[Vec<f32>],
) -> Result<GpuCrownResult> {
    if resnet_max_macs(segments, seed.num_specs) < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "cuda beta resnet sound CROWN: net below GPU size-gate (CPU is faster here)".into(),
        ));
    }
    resnet_backward_f64_core(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        &mut Some(ActAux::beta(beta_signed)),
    )
}

/// Guard-only form of [`crown_backward_gpu_resnet_sound_beta_impl`].
///
/// The ordinary serial entry's [`MIN_RESIDENT_MACS`] threshold is purely a
/// throughput decision. A wide batch can clear that threshold while each
/// one-row domain used by the runtime re-fold guard does not. The guard still
/// needs an independent serial execution, so force the identical sound f64 core
/// here without changing the ordinary dispatch policy. This result is used only
/// as a comparison oracle, never directly as a verdict bound.
pub(crate) fn crown_backward_gpu_resnet_sound_beta_refold_oracle_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    beta_signed: &[Vec<f32>],
) -> Result<GpuCrownResult> {
    resnet_backward_f64_core(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        &mut Some(ActAux::beta(beta_signed)),
    )
}

/// f64-GEMM MAC estimate for one CROWN layer's backward (drives the size-gate):
/// Linear = `specs·out·in`; Conv2d = `specs·oh·ow·oc·ic·kh·kw` (the transposed-conv
/// GEMM+col2im work). Others (Activation, …) do no GEMM ⇒ 0.
fn layer_macs(l: &GpuCrownLayer, num_specs: usize) -> u128 {
    match l {
        GpuCrownLayer::Linear {
            out_features,
            in_features,
            ..
        } => (num_specs as u128) * (*out_features as u128) * (*in_features as u128),
        GpuCrownLayer::Conv2d {
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            out_h,
            out_w,
            ..
        } => {
            (num_specs as u128)
                * (*out_h as u128)
                * (*out_w as u128)
                * (*out_channels as u128)
                * (*in_channels as u128)
                * (*kernel_h as u128)
                * (*kernel_w as u128)
        }
        _ => 0,
    }
}

/// Largest single-layer GEMM MAC count across a resnet's segments (drives the gate).
fn resnet_max_macs(segments: &[GpuResnetSegment], num_specs: usize) -> u128 {
    use ny_core::GpuResnetSegment;
    let seg = |layers: &[GpuCrownLayer]| -> u128 {
        layers
            .iter()
            .map(|l| layer_macs(l, num_specs))
            .max()
            .unwrap_or(0)
    };
    segments
        .iter()
        .map(|s| match s {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => seg(l),
            GpuResnetSegment::ResidualProj(f, p) => seg(f).max(seg(p)),
        })
        .max()
        .unwrap_or(0)
}

/// SOUND multi-domain CUDA β-CROWN.  Domains/specifications are one algebraic
/// batch dimension, so every affine layer executes three wide DGEMMs (center,
/// magnitude, propagated error) instead of `3 * domains` small synchronized
/// DGEMMs.  Dynamic ReLU and β tables stay domain-indexed; final rows are sliced
/// and concretized against each child's own box.
pub(crate) fn crown_backward_gpu_resnet_sound_beta_batched_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
) -> Result<Vec<GpuCrownResult>> {
    resnet_backward_f64_wide_chunked(eng, domains, seed, &[], &[], false)
        .map(|(bounds, _, _, _)| bounds)
}

/// Gradient-capturing sibling used by the production wide β-ascent loop.  β
/// gathers and optional per-domain α gradients are read from the same wide
/// coefficient stream; neither channel participates in the verdict bound.
#[allow(clippy::type_complexity)]
pub(crate) fn crown_backward_gpu_resnet_sound_beta_batched_grad_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    union_gather_idx: &[&[u32]],
    relu_pre_lower: &[&[Vec<f32>]],
) -> Result<(Vec<GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    resnet_backward_f64_wide_chunked(eng, domains, seed, union_gather_idx, relu_pre_lower, false)
        .map(|(bounds, alpha_grads, beta_gather, _)| (bounds, alpha_grads, beta_gather))
}

/// Combined trajectory capture: the f64 frontier produced for these bounds is
/// widened once into the public f32 center/error representation.  No second
/// backward is issued for coefficients; memory-safe domain subchunks are
/// concatenated in domain-major order when the conservative cap requires it.
pub(crate) fn crown_backward_gpu_resnet_sound_beta_batched_trajectory_impl<
    E: GemmEngine + ?Sized,
>(
    eng: &E,
    domains: &[GpuResnetBatchedDomainRef<'_>],
    seed: &GpuCrownSeed,
    union_gather_idx: &[&[u32]],
    relu_pre_lower: &[&[Vec<f32>]],
) -> Result<GpuCrownTrajectoryResult> {
    let (bounds, alpha_grads, beta_gather, coeff) = resnet_backward_f64_wide_chunked(
        eng,
        domains,
        seed,
        union_gather_idx,
        relu_pre_lower,
        true,
    )?;
    let coeff = coeff.ok_or_else(|| {
        NyError::InternalError("cuda wide trajectory: coefficient capture missing".into())
    })?;
    validate_resident_coeff(&coeff)?;
    Ok(GpuCrownTrajectoryResult {
        bounds,
        alpha_grads,
        beta_gather,
        coeff,
    })
}

/// SOUND f64-native GPU-resident GRADIENT-capturing resnet backward (T1.3): the
/// same sound bounds as [`crown_backward_gpu_resnet_sound_impl`], plus each ReLU's
/// analytic alpha gradient `pre_lower[i]·Σ_j max(A_lower_pre[j,i], 0)` (fold order),
/// captured from the PRE-transform lower coefficient. Gradients are
/// NON-soundness-critical (they only steer the warmup alpha); a wrong gradient can
/// never affect a verdict, only convergence speed. Same size-gate.
pub(crate) fn crown_backward_gpu_resnet_sound_grad_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    relu_pre_lower: &[Vec<f32>],
) -> Result<ny_core::GpuCrownGradResult> {
    if resnet_max_macs(segments, seed.num_specs) < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "cuda grad resnet sound CROWN: net below GPU size-gate (CPU is faster here)".into(),
        ));
    }
    let mut aux = Some(ActAux::grad(relu_pre_lower));
    let bounds = resnet_backward_f64_core(eng, segments, seed, input_lower, input_upper, &mut aux)?;
    let relu_grads = aux.map(|a| a.relu_grads).unwrap_or_default();
    Ok(ny_core::GpuCrownGradResult {
        lower_bounds: bounds.lower_bounds,
        upper_bounds: bounds.upper_bounds,
        relu_grads,
    })
}

/// SOUND f64-native GPU-resident β-GRADIENT resnet backward (T1.3): the same sound
/// β-folded bounds as [`crown_backward_gpu_resnet_sound_beta_impl`], plus each
/// requested ReLU's PRE-transform lower A-coefficients gathered at the split neuron
/// columns (`beta_gather_idx`, fold order), row-major `num_specs × |idx|` — the
/// inputs to the CPU analytic β-gradient rule `∂lb_row/∂β_k = −sign_k·A_lower[row,k]`.
/// The gather reads the coefficient buffer only, so the BOUNDS are identical to the
/// non-gather beta path; gathered values are NON-soundness-critical. Same size-gate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn crown_backward_gpu_resnet_sound_beta_grad_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    beta_signed: &[Vec<f32>],
    beta_gather_idx: &[Vec<u32>],
) -> Result<ny_core::GpuCrownBetaGradResult> {
    if resnet_max_macs(segments, seed.num_specs) < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "cuda beta-grad resnet sound CROWN: net below GPU size-gate (CPU is faster here)"
                .into(),
        ));
    }
    let mut aux = Some(ActAux::beta_grad(beta_signed, beta_gather_idx));
    let bounds = resnet_backward_f64_core(eng, segments, seed, input_lower, input_upper, &mut aux)?;
    let beta_gather = aux.map(|a| a.beta_gather).unwrap_or_default();
    Ok(ny_core::GpuCrownBetaGradResult {
        lower_bounds: bounds.lower_bounds,
        upper_bounds: bounds.upper_bounds,
        beta_gather,
    })
}

/// SOUND f64-native GPU-resident CROWN backward with a small-net size-gate. Routes
/// nets whose largest Linear is below [`MIN_RESIDENT_MACS`] to the CPU sound path
/// (`UnsupportedOp`), where the GPU would lose to launch/transfer overhead.
pub(crate) fn crown_backward_gpu_sound_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    layers: &[GpuCrownLayer],
    spec: &[f32],
    num_specs: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Result<GpuCrownResult> {
    let max_macs: u128 = layers
        .iter()
        .map(|l| layer_macs(l, num_specs))
        .max()
        .unwrap_or(0);
    if max_macs < MIN_RESIDENT_MACS {
        tracing::debug!(
            "cuda CROWN: GATED OUT → CPU (max_macs={max_macs}, gate={MIN_RESIDENT_MACS}, specs={num_specs}, layers={})",
            layers.len()
        );
        return Err(NyError::UnsupportedOp(
            "cuda sound CROWN: net below GPU size-gate (CPU is faster here)".into(),
        ));
    }
    tracing::debug!(
        "cuda CROWN: RESIDENT f64 backward (max_macs={max_macs}, specs={num_specs}, layers={})",
        layers.len()
    );
    backward_f64_core(eng, layers, spec, num_specs, input_lower, input_upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaGemmEngine;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct RecordingCpuGemm {
        calls: Mutex<Vec<(usize, usize, usize)>>,
    }

    impl GemmEngine for RecordingCpuGemm {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    for p in 0..k {
                        out[i * n + j] += a[i * k + p] * b[p * n + j];
                    }
                }
            }
            Ok(out)
        }

        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            self.calls.lock().expect("recording lock").push((m, k, n));
            let mut out = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    for p in 0..k {
                        out[i * n + j] += a[i * k + p] * b[p * n + j];
                    }
                }
            }
            Ok(out)
        }
    }

    struct OwnedDomain {
        segments: Vec<GpuResnetSegment>,
        lo: Vec<f32>,
        hi: Vec<f32>,
        beta: Vec<Vec<f32>>,
    }

    fn tiny_wide_domain(domain: usize) -> OwnedDomain {
        // Fresh, equal-by-value Arcs deliberately exercise the homogeneity gate's
        // value fallback (real graph extraction currently behaves this way).
        let output = GpuCrownLayer::Linear {
            weight: Arc::from(vec![0.8, -0.3, 0.2, 0.9]),
            bias: Some(Arc::from(vec![0.1, -0.2])),
            out_features: 2,
            in_features: 2,
        };
        let d = domain as f32;
        let activation = GpuCrownLayer::Activation {
            lower_slope: vec![0.15 + 0.1 * d, 0.75 - 0.05 * d],
            upper_slope: vec![0.8, 0.9],
            lower_intercept: vec![0.02 * d, -0.01 * d],
            upper_intercept: vec![0.25 + 0.03 * d, 0.1],
            num_neurons: 2,
        };
        let input = GpuCrownLayer::Linear {
            weight: Arc::from(vec![1.1, -0.4, -0.6, 0.7]),
            bias: Some(Arc::from(vec![0.05, 0.12])),
            out_features: 2,
            in_features: 2,
        };
        OwnedDomain {
            segments: vec![GpuResnetSegment::Chain(vec![output, activation, input])],
            lo: vec![-1.0 + 0.1 * d, -0.7 - 0.05 * d],
            hi: vec![0.9 + 0.05 * d, 1.2 - 0.1 * d],
            beta: vec![vec![0.03 * (d + 1.0), -0.02 * d]],
        }
    }

    struct OwnedResidualDomain {
        segments: Vec<GpuResnetSegment>,
        lo: Vec<f32>,
        hi: Vec<f32>,
        beta: Vec<Vec<f32>>,
        pre_lower: Vec<Vec<f32>>,
    }

    fn tiny_wide_residual_domain(domain: usize) -> OwnedResidualDomain {
        let d = domain as f32;
        let activation = |width: usize, stage: usize| {
            let q = stage as f32;
            GpuCrownLayer::Activation {
                lower_slope: (0..width)
                    .map(|i| 0.12 + 0.04 * d + 0.03 * q + 0.05 * i as f32)
                    .collect(),
                upper_slope: (0..width)
                    .map(|i| 0.72 + 0.02 * d + 0.03 * i as f32)
                    .collect(),
                lower_intercept: (0..width)
                    .map(|i| -0.015 * d + 0.01 * q * i as f32)
                    .collect(),
                upper_intercept: (0..width)
                    .map(|i| 0.08 + 0.025 * d + 0.02 * q + 0.01 * i as f32)
                    .collect(),
                num_neurons: width,
            }
        };
        let linear =
            |weight: Vec<f32>, bias: Vec<f32>, out_features, in_features| GpuCrownLayer::Linear {
                weight: Arc::from(weight),
                bias: Some(Arc::from(bias)),
                out_features,
                in_features,
            };
        let conv1x1 = |weight: Vec<f32>, bias: Vec<f32>| GpuCrownLayer::Conv2d {
            weight_col: Arc::from(weight),
            bias_expanded: Some(Arc::from(bias)),
            out_channels: 2,
            in_channels: 3,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: 1,
            in_h: 1,
            in_w: 1,
        };

        // Backward order: output chain, identity residual, projection residual,
        // input chain. The three ReLUs have widths 2, 2, and 3, respectively,
        // which exercises independent domain-block strides at each fold cursor.
        let segments = vec![
            GpuResnetSegment::Chain(vec![linear(
                vec![0.8, -0.3, 0.2, 0.9],
                vec![0.1, -0.2],
                2,
                2,
            )]),
            GpuResnetSegment::Residual(vec![
                activation(2, 0),
                linear(vec![0.5, -0.7, 0.4, 0.6], vec![0.03, -0.04], 2, 2),
            ]),
            GpuResnetSegment::ResidualProj(
                vec![
                    activation(2, 1),
                    conv1x1(vec![0.3, -0.2, 0.7, -0.5, 0.8, 0.1], vec![0.02, -0.06]),
                ],
                vec![conv1x1(
                    vec![0.9, 0.1, -0.4, 0.2, -0.6, 0.7],
                    vec![-0.03, 0.05],
                )],
            ),
            GpuResnetSegment::Chain(vec![
                activation(3, 2),
                linear(
                    vec![0.6, -0.2, -0.3, 0.9, 0.5, 0.4],
                    vec![0.04, -0.01, 0.07],
                    3,
                    2,
                ),
            ]),
        ];
        OwnedResidualDomain {
            segments,
            lo: vec![-1.0 + 0.08 * d, -0.6 - 0.04 * d],
            hi: vec![0.85 + 0.03 * d, 1.1 - 0.06 * d],
            beta: vec![
                vec![0.02 * (d + 1.0), -0.01 * d],
                vec![-0.015 * (d + 1.0), 0.025 * d],
                vec![0.01 * d, -0.02 * (d + 1.0), 0.03],
            ],
            pre_lower: vec![
                vec![-0.7 - 0.1 * d, -0.25 + 0.03 * d],
                vec![-0.5 + 0.02 * d, -0.9 - 0.04 * d],
                vec![-0.3 - 0.05 * d, -0.8, -0.15 + 0.02 * d],
            ],
        }
    }

    struct CifarWideEstimatorFixture {
        segments: Vec<GpuResnetSegment>,
        lo: Vec<f32>,
        hi: Vec<f32>,
        beta: Vec<Vec<f32>>,
        seed: GpuCrownSeed,
        gather: Vec<Vec<u32>>,
    }

    /// Aggregate dimensions from the sealed CIFAR100 ResNet-medium row-52 path.
    /// Only estimator metadata is exercised: final Linear weights contribute the
    /// 204900-element static maximum, the representative conv contributes the
    /// 73728-cell im2col maximum, and activations total 55460 ReLUs.
    fn cifar_wide_estimator_fixture() -> CifarWideEstimatorFixture {
        let final_linear = GpuCrownLayer::Linear {
            weight: Arc::from(vec![0.0; 100 * 2048]),
            bias: Some(Arc::from(vec![0.0; 100])),
            out_features: 100,
            in_features: 2048,
        };
        let activation = GpuCrownLayer::Activation {
            lower_slope: vec![0.0; 55_460],
            upper_slope: vec![0.0; 55_460],
            lower_intercept: vec![0.0; 55_460],
            upper_intercept: vec![0.0; 55_460],
            num_neurons: 55_460,
        };
        let max_workspace_conv = GpuCrownLayer::Conv2d {
            weight_col: Arc::from(vec![0.0; 8 * 8 * 3 * 3]),
            bias_expanded: None,
            out_channels: 8,
            in_channels: 8,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 32,
            out_w: 32,
            in_h: 34,
            in_w: 34,
        };
        CifarWideEstimatorFixture {
            segments: vec![GpuResnetSegment::Chain(vec![
                final_linear,
                activation,
                max_workspace_conv,
            ])],
            lo: vec![-1.0; 3072],
            hi: vec![1.0; 3072],
            beta: Vec::new(),
            seed: GpuCrownSeed {
                lower_a: Arc::from(vec![0.0; 53 * 100]),
                upper_a: Arc::from(vec![0.0; 53 * 100]),
                lower_b: Arc::from(vec![0.0; 53]),
                upper_b: Arc::from(vec![0.0; 53]),
                num_specs: 53,
                current_dim: 100,
            },
            gather: vec![vec![0]],
        }
    }

    #[test]
    fn cuda_wide_cap_parser_is_fail_closed() {
        use std::ffi::OsStr;

        assert_eq!(
            parse_cuda_wide_max_bytes(None).expect("default cap"),
            512 * 1024 * 1024
        );
        assert_eq!(
            parse_cuda_wide_max_bytes(Some(OsStr::new("2147483648"))).expect("2GiB cap"),
            2_147_483_648
        );
        assert_eq!(
            parse_cuda_wide_max_bytes(Some(OsStr::new("000128"))).expect("leading zeros"),
            128
        );
        assert_eq!(
            parse_cuda_wide_max_bytes(Some(OsStr::new(&usize::MAX.to_string())))
                .expect("native usize maximum"),
            usize::MAX
        );
        for malformed in [
            "",
            "0",
            "+1",
            "-1",
            "1.0",
            " 1024",
            "1024 ",
            "１２８",
            "not-a-number",
        ] {
            let error = parse_cuda_wide_max_bytes(Some(OsStr::new(malformed)))
                .expect_err("malformed cap must fail closed");
            assert_eq!(
                error.to_string(),
                "Invalid specification: NY_CUDA_WIDE_MAX_BYTES must be a positive base-10 integer"
            );
        }
        let overflow = format!("{}0", usize::MAX);
        assert!(parse_cuda_wide_max_bytes(Some(OsStr::new(&overflow))).is_err());
    }

    #[test]
    fn cuda_wide_cifar_chunk_plan_exact_boundaries_and_marker() {
        const FIXED: usize = 6_558_072;
        const PER_DOMAIN: usize = 1_003_890_944;
        const ONE_DOMAIN: usize = FIXED + PER_DOMAIN;
        const TWO_DOMAINS: usize = FIXED + 2 * PER_DOMAIN;
        const MIB_512: usize = 512 * 1024 * 1024;
        const GIB_1: usize = 1024 * 1024 * 1024;
        const GIB_2: usize = 2 * 1024 * 1024 * 1024;
        const GIB_4: usize = 4 * 1024 * 1024 * 1024;

        let fixture = cifar_wide_estimator_fixture();
        let empty: Vec<Vec<f32>> = Vec::new();
        let domains = (0..2)
            .map(|_| GpuResnetBatchedDomainRef {
                segments: &fixture.segments,
                input_lower: &fixture.lo,
                input_upper: &fixture.hi,
                beta_signed: &fixture.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect::<Vec<_>>();
        let gather_refs = fixture.gather.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let plan_for = |cap| {
            cuda_wide_chunk_plan_for_cap(&domains, &fixture.seed, &gather_refs, &[], false, cap)
                .expect("CIFAR estimator plan")
        };

        let rejected = plan_for(MIB_512);
        assert_eq!(
            rejected,
            CudaWideChunkPlan {
                cap_bytes: MIB_512,
                fixed_bytes: FIXED,
                per_domain_bytes: PER_DOMAIN,
                requested_domains: 2,
                chunk_domains: 0,
            }
        );
        let error = cuda_wide_chunk_domains_for_cap(
            &domains,
            &fixture.seed,
            &gather_refs,
            &[],
            false,
            MIB_512,
        )
        .expect_err("512MiB must reject one CIFAR domain");
        assert_eq!(
            error.to_string(),
            "Unsupported operation: cuda wide resnet: one domain exceeds 536870912-byte \
             conservative memory cap"
        );

        assert_eq!(plan_for(ONE_DOMAIN - 1).chunk_domains, 0);
        assert_eq!(plan_for(ONE_DOMAIN).chunk_domains, 1);
        assert_eq!(plan_for(GIB_1).chunk_domains, 1);
        assert_eq!(plan_for(TWO_DOMAINS - 1).chunk_domains, 1);
        assert_eq!(plan_for(TWO_DOMAINS).chunk_domains, 2);
        assert_eq!(plan_for(GIB_2).chunk_domains, 2);
        assert_eq!(plan_for(GIB_4).chunk_domains, 2);

        assert_eq!(
            cuda_wide_chunk_plan_line(plan_for(GIB_2)),
            "NY_CUDA_WIDE_CHUNK_PLAN_V1 cap_bytes=2147483648 fixed_bytes=6558072 \
             per_domain_bytes=1003890944 requested_domains=2 chunk_domains=2"
        );
    }

    #[test]
    fn cuda_wide_beta_matches_serial_and_collapses_launches() {
        let engine = RecordingCpuGemm::default();
        let owned: Vec<OwnedDomain> = (0..3).map(tiny_wide_domain).collect();
        let empty: Vec<Vec<f32>> = Vec::new();
        let refs: Vec<GpuResnetBatchedDomainRef<'_>> = owned
            .iter()
            .map(|d| GpuResnetBatchedDomainRef {
                segments: &d.segments,
                input_lower: &d.lo,
                input_upper: &d.hi,
                beta_signed: &d.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect();
        let seed = GpuCrownSeed {
            lower_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0]),
            upper_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0]),
            lower_b: Arc::from(vec![0.0, 0.0]),
            upper_b: Arc::from(vec![0.0, 0.0]),
            num_specs: 2,
            current_dim: 2,
        };

        let gather_cols = [0_u32, 1];
        let (wide, _alpha, wide_gather) = resnet_backward_f64_wide_core(
            &engine,
            &refs,
            &seed,
            &[gather_cols.as_slice()],
            &[],
            false,
            None,
            false,
        )
        .expect("wide fold");
        let wide_calls = engine.calls.lock().expect("recording lock").len();
        assert_eq!(wide_calls, 6, "two affine layers x three sound DGEMMs");

        engine.calls.lock().expect("recording lock").clear();
        let mut serial_gather = Vec::new();
        for (idx, domain) in owned.iter().enumerate() {
            let gather = vec![gather_cols.to_vec()];
            let mut aux = Some(ActAux::beta_grad(&domain.beta, &gather));
            let serial = resnet_backward_f64_core(
                &engine,
                &domain.segments,
                &seed,
                &domain.lo,
                &domain.hi,
                &mut aux,
            )
            .expect("serial fold");
            assert_eq!(wide[idx].lower_bounds, serial.lower_bounds);
            assert_eq!(wide[idx].upper_bounds, serial.upper_bounds);
            serial_gather.extend(aux.expect("serial aux").beta_gather.remove(0));
        }
        let serial_calls = engine.calls.lock().expect("recording lock").len();
        assert_eq!(serial_calls, 18);
        assert_eq!(wide_gather, vec![serial_gather]);
    }

    /// The proof-forest can clear the CUDA work-size gate on the aggregate row
    /// count while its one-domain re-fold sample cannot. The ordinary serial
    /// entry must retain that performance gate; the guard-only oracle must run
    /// the same sound serial core anyway, or `serial_ok=false` discards every
    /// otherwise-valid one-row wide result in the scored CIFAR route.
    #[test]
    fn cuda_refold_oracle_bypasses_only_the_serial_performance_gate() {
        let engine = RecordingCpuGemm::default();
        let domain = tiny_wide_domain(0);
        let seed = GpuCrownSeed {
            lower_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0]),
            upper_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0]),
            lower_b: Arc::from(vec![0.0, 0.0]),
            upper_b: Arc::from(vec![0.0, 0.0]),
            num_specs: 2,
            current_dim: 2,
        };

        let gated = match crown_backward_gpu_resnet_sound_beta_impl(
            &engine,
            &domain.segments,
            &seed,
            &domain.lo,
            &domain.hi,
            &domain.beta,
        ) {
            Ok(_) => panic!("tiny ordinary serial fold must retain its performance gate"),
            Err(error) => error,
        };
        assert_eq!(
            gated.to_string(),
            "Unsupported operation: cuda beta resnet sound CROWN: net below GPU size-gate \
             (CPU is faster here)"
        );
        assert!(
            engine.calls.lock().expect("recording lock").is_empty(),
            "performance-gated entry must not launch GEMMs"
        );

        let oracle = crown_backward_gpu_resnet_sound_beta_refold_oracle_impl(
            &engine,
            &domain.segments,
            &seed,
            &domain.lo,
            &domain.hi,
            &domain.beta,
        )
        .expect("guard-only oracle must force the sound serial core");
        assert!(
            oracle
                .lower_bounds
                .iter()
                .chain(oracle.upper_bounds.iter())
                .all(|v| v.is_finite()),
            "forced oracle must retain finite-result validation"
        );
        assert_eq!(
            engine.calls.lock().expect("recording lock").len(),
            6,
            "two affine layers x three certified-error GEMMs"
        );
    }

    /// Device qualification for the public trait seam used by the production
    /// guard. This is intentionally below `MIN_RESIDENT_MACS`: the ordinary
    /// entry must refuse it, while the refold oracle must execute on CUDA and
    /// agree with the same sound core on the recording CPU engine.
    #[test]
    fn cuda_refold_oracle_executes_on_device_below_size_gate() {
        use ny_core::GpuCrownBackward;

        let engine = match CudaGemmEngine::new() {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("skipping CUDA refold-oracle test (no device): {error}");
                return;
            }
        };
        let domain = tiny_wide_domain(1);
        let seed = GpuCrownSeed {
            lower_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0]),
            upper_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0]),
            lower_b: Arc::from(vec![0.0, 0.0]),
            upper_b: Arc::from(vec![0.0, 0.0]),
            num_specs: 2,
            current_dim: 2,
        };

        let ordinary = engine.crown_backward_gpu_resnet_sound_beta(
            &domain.segments,
            &seed,
            &domain.lo,
            &domain.hi,
            &domain.beta,
            &[],
            &[],
        );
        assert!(
            matches!(ordinary, Err(NyError::UnsupportedOp(_))),
            "ordinary below-gate CUDA entry must retain its dispatch policy"
        );

        let device = engine
            .crown_backward_gpu_resnet_sound_beta_refold_oracle(
                &domain.segments,
                &seed,
                &domain.lo,
                &domain.hi,
                &domain.beta,
                &[],
                &[],
            )
            .expect("refold oracle must execute the sound serial core on CUDA");
        let cpu = crown_backward_gpu_resnet_sound_beta_refold_oracle_impl(
            &RecordingCpuGemm::default(),
            &domain.segments,
            &seed,
            &domain.lo,
            &domain.hi,
            &domain.beta,
        )
        .expect("recording CPU oracle");
        let close = |a: f32, b: f32| {
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()))
        };
        assert_eq!(device.lower_bounds.len(), cpu.lower_bounds.len());
        assert_eq!(device.upper_bounds.len(), cpu.upper_bounds.len());
        assert!(
            device
                .lower_bounds
                .iter()
                .zip(cpu.lower_bounds.iter())
                .chain(device.upper_bounds.iter().zip(cpu.upper_bounds.iter()))
                .all(|(&a, &b)| close(a, b)),
            "device refold must meet the production wide/serial comparison contract"
        );
    }

    #[test]
    fn cuda_wide_residual_projection_alpha_and_gather_match_serial() {
        let engine = RecordingCpuGemm::default();
        let owned: Vec<OwnedResidualDomain> = (0..2).map(tiny_wide_residual_domain).collect();
        let empty: Vec<Vec<f32>> = Vec::new();
        let refs = owned
            .iter()
            .map(|d| GpuResnetBatchedDomainRef {
                segments: &d.segments,
                input_lower: &d.lo,
                input_upper: &d.hi,
                beta_signed: &d.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect::<Vec<_>>();
        let seed = GpuCrownSeed {
            lower_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0, 0.7, -0.4]),
            upper_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0, 0.6, -0.3]),
            lower_b: Arc::from(vec![0.02, -0.03, 0.04]),
            upper_b: Arc::from(vec![0.05, 0.01, 0.08]),
            num_specs: 3,
            current_dim: 2,
        };
        let gather = vec![vec![0_u32, 1], vec![1], vec![0, 2]];
        let gather_refs = gather.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let pre_lower_refs = owned
            .iter()
            .map(|d| d.pre_lower.as_slice())
            .collect::<Vec<_>>();

        let (wide, wide_alpha, wide_gather) = resnet_backward_f64_wide_core(
            &engine,
            &refs,
            &seed,
            &gather_refs,
            &pre_lower_refs,
            false,
            None,
            false,
        )
        .expect("wide residual/projection fold");
        assert_eq!(
            engine.calls.lock().expect("recording lock").len(),
            15,
            "five affine branch layers x three sound DGEMMs"
        );

        engine.calls.lock().expect("recording lock").clear();
        let mut serial_alpha = vec![Vec::new(); gather.len()];
        let mut serial_gather = vec![Vec::new(); gather.len()];
        for (domain_idx, domain) in owned.iter().enumerate() {
            let mut aux = Some(ActAux::beta_grad_alpha(
                &domain.beta,
                &gather,
                &domain.pre_lower,
            ));
            let serial = resnet_backward_f64_core(
                &engine,
                &domain.segments,
                &seed,
                &domain.lo,
                &domain.hi,
                &mut aux,
            )
            .expect("serial residual/projection fold");
            assert_eq!(wide[domain_idx].lower_bounds, serial.lower_bounds);
            assert_eq!(wide[domain_idx].upper_bounds, serial.upper_bounds);
            let aux = aux.expect("serial aux");
            for (dst, src) in serial_alpha.iter_mut().zip(aux.relu_grads) {
                dst.extend(src);
            }
            for (dst, src) in serial_gather.iter_mut().zip(aux.beta_gather) {
                dst.extend(src);
            }
        }
        assert_eq!(
            engine.calls.lock().expect("recording lock").len(),
            30,
            "two domains x five affine branch layers x three sound DGEMMs"
        );
        assert_eq!(wide_alpha, serial_alpha);
        assert_eq!(wide_gather, serial_gather);
    }

    #[test]
    fn cuda_wide_trajectory_coeff_layout_concretization_and_subchunks_match() {
        let engine = RecordingCpuGemm::default();
        let owned: Vec<OwnedResidualDomain> = (0..2).map(tiny_wide_residual_domain).collect();
        let empty: Vec<Vec<f32>> = Vec::new();
        let refs = owned
            .iter()
            .map(|d| GpuResnetBatchedDomainRef {
                segments: &d.segments,
                input_lower: &d.lo,
                input_upper: &d.hi,
                beta_signed: &d.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect::<Vec<_>>();
        let seed = GpuCrownSeed {
            lower_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0, 0.7, -0.4]),
            upper_a: Arc::from(vec![1.0, 0.0, 0.0, 1.0, 0.6, -0.3]),
            lower_b: Arc::from(vec![0.02, -0.03, 0.04]),
            upper_b: Arc::from(vec![0.05, 0.01, 0.08]),
            num_specs: 3,
            current_dim: 2,
        };
        let gather = vec![vec![0_u32, 1], vec![1], vec![0, 2]];
        let gather_refs = gather.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let pre_lower_refs = owned
            .iter()
            .map(|d| d.pre_lower.as_slice())
            .collect::<Vec<_>>();

        let (wide_bounds, wide_alpha, wide_gather, wide_coeff) =
            resnet_backward_f64_wide_chunked_with_cap(
                &engine,
                &refs,
                &seed,
                &gather_refs,
                &pre_lower_refs,
                false,
                true,
                1 << 20,
            )
            .expect("one-pass trajectory capture");
        assert_eq!(
            engine.calls.lock().expect("recording lock").len(),
            15,
            "coefficient capture must not issue a second backward"
        );
        let wide_coeff = wide_coeff.expect("captured coeff");
        assert_eq!(wide_coeff.dim, 2);
        assert_eq!(wide_coeff.num_specs, 6);
        assert_eq!(wide_coeff.num_specs_per_dom, 3);
        validate_resident_coeff(&wide_coeff).expect("valid resident coeff");

        // A tiny cap forces one domain per pass.  Concatenation must preserve
        // byte-identical domain-major bounds, captures, and coefficient rows.
        assert_eq!(
            cuda_wide_chunk_domains_for_cap(
                &refs,
                &seed,
                &gather_refs,
                &pre_lower_refs,
                true,
                6_000,
            )
            .expect("chunk estimate"),
            1
        );
        engine.calls.lock().expect("recording lock").clear();
        let (chunk_bounds, chunk_alpha, chunk_gather, chunk_coeff) =
            resnet_backward_f64_wide_chunked_with_cap(
                &engine,
                &refs,
                &seed,
                &gather_refs,
                &pre_lower_refs,
                false,
                true,
                6_000,
            )
            .expect("subchunked trajectory capture");
        assert_eq!(
            engine.calls.lock().expect("recording lock").len(),
            30,
            "two one-domain chunks x five affine layers x three sound DGEMMs"
        );
        for (wide, chunk) in wide_bounds.iter().zip(&chunk_bounds) {
            assert_eq!(wide.lower_bounds, chunk.lower_bounds);
            assert_eq!(wide.upper_bounds, chunk.upper_bounds);
        }
        assert_eq!(wide_alpha, chunk_alpha);
        assert_eq!(wide_gather, chunk_gather);
        let chunk_coeff = chunk_coeff.expect("subchunked coeff");
        assert_eq!(wide_coeff.lower_a, chunk_coeff.lower_a);
        assert_eq!(wide_coeff.upper_a, chunk_coeff.upper_a);
        assert_eq!(wide_coeff.lower_err, chunk_coeff.lower_err);
        assert_eq!(wide_coeff.upper_err, chunk_coeff.upper_err);
        assert_eq!(wide_coeff.lower_b, chunk_coeff.lower_b);
        assert_eq!(wide_coeff.upper_b, chunk_coeff.upper_b);
        assert_eq!(wide_coeff.lower_b_err, chunk_coeff.lower_b_err);
        assert_eq!(wide_coeff.upper_b_err, chunk_coeff.upper_b_err);

        // Each domain-major f32 interval encloses its independently folded f64
        // frontier, including the cast displacement charged into the error.
        for (domain_idx, domain) in owned.iter().enumerate() {
            let mut aux = Some(ActAux::beta_grad_alpha(
                &domain.beta,
                &gather,
                &domain.pre_lower,
            ));
            let serial = resnet_fold_f64_core(&engine, &domain.segments, &seed, &mut aux)
                .expect("serial f64 frontier");
            for row in 0..seed.num_specs {
                let wide_row = domain_idx * seed.num_specs + row;
                for col in 0..serial.dim {
                    let wi = wide_row * serial.dim + col;
                    let si = row * serial.dim + col;
                    let lc = f64::from(wide_coeff.lower_a[wi]);
                    let le = f64::from(wide_coeff.lower_err[wi]);
                    assert!(lc - le <= serial.lower_a[si] - serial.lower_err[si]);
                    assert!(lc + le >= serial.lower_a[si] + serial.lower_err[si]);
                    let uc = f64::from(wide_coeff.upper_a[wi]);
                    let ue = f64::from(wide_coeff.upper_err[wi]);
                    assert!(uc - ue <= serial.upper_a[si] - serial.upper_err[si]);
                    assert!(uc + ue >= serial.upper_a[si] + serial.upper_err[si]);
                }
                let lb = f64::from(wide_coeff.lower_b[wide_row]);
                let lbe = f64::from(wide_coeff.lower_b_err[wide_row]);
                assert!(lb - lbe <= serial.lb[row] - serial.lb_err[row]);
                assert!(lb + lbe >= serial.lb[row] + serial.lb_err[row]);
                let ub = f64::from(wide_coeff.upper_b[wide_row]);
                let ube = f64::from(wide_coeff.upper_b_err[wide_row]);
                assert!(ub - ube <= serial.ub[row] - serial.ub_err[row]);
                assert!(ub + ube >= serial.ub[row] + serial.ub_err[row]);
            }

            // Re-concretizing the exported enclosure through the same f64 oracle
            // can only widen the verdict bounds, never tighten them.
            let row_start = domain_idx * seed.num_specs;
            let row_end = row_start + seed.num_specs;
            let coeff_start = row_start * wide_coeff.dim;
            let coeff_end = row_end * wide_coeff.dim;
            let to_f64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
            let (lo, hi) = concretize_f64(
                seed.num_specs,
                wide_coeff.dim,
                &to_f64(&wide_coeff.lower_a[coeff_start..coeff_end]),
                &to_f64(&wide_coeff.upper_a[coeff_start..coeff_end]),
                &to_f64(&wide_coeff.lower_err[coeff_start..coeff_end]),
                &to_f64(&wide_coeff.upper_err[coeff_start..coeff_end]),
                &domain.lo,
                &domain.hi,
                &to_f64(&wide_coeff.lower_b[row_start..row_end]),
                &to_f64(&wide_coeff.upper_b[row_start..row_end]),
                &to_f64(&wide_coeff.lower_b_err[row_start..row_end]),
                &to_f64(&wide_coeff.upper_b_err[row_start..row_end]),
            );
            for row in 0..seed.num_specs {
                assert!(lo[row] <= wide_bounds[domain_idx].lower_bounds[row]);
                assert!(hi[row] >= wide_bounds[domain_idx].upper_bounds[row]);
            }
        }
    }

    #[test]
    fn cuda_trajectory_cast_and_shape_error_invariants_fail_closed() {
        let center = 1.0 + 2.0f64.powi(-30);
        let inherited = 2.0f64.powi(-60);
        let (rounded, error) =
            widen_center_error_f64_to_f32(center, inherited).expect("finite widening");
        assert_eq!(rounded, 1.0);
        assert!(f64::from(rounded) - f64::from(error) <= center - inherited);
        assert!(f64::from(rounded) + f64::from(error) >= center + inherited);
        assert!(widen_center_error_f64_to_f32(f64::NAN, 0.0).is_err());
        assert!(widen_center_error_f64_to_f32(0.0, -1.0).is_err());
        assert!(widen_center_error_f64_to_f32(f64::MAX, 0.0).is_err());

        let malformed = Frontier {
            lower_a: vec![0.0],
            upper_a: vec![0.0, 1.0],
            lower_err: vec![0.0],
            upper_err: vec![0.0, 0.0],
            lb: vec![0.0],
            ub: vec![0.0],
            lb_err: vec![0.0],
            ub_err: vec![0.0],
            dim: 2,
        };
        assert!(frontier_to_resident_coeff(&malformed, 1, 1).is_err());

        let invalid_error = Frontier {
            lower_a: vec![0.0],
            upper_a: vec![0.0],
            lower_err: vec![-1.0],
            upper_err: vec![0.0],
            lb: vec![0.0],
            ub: vec![0.0],
            lb_err: vec![0.0],
            ub_err: vec![0.0],
            dim: 1,
        };
        assert!(frontier_to_resident_coeff(&invalid_error, 1, 1).is_err());
    }

    #[test]
    fn cuda_wide_rejects_cross_domain_weight_or_box_mismatch() {
        let engine = RecordingCpuGemm::default();
        let mut owned: Vec<OwnedDomain> = (0..2).map(tiny_wide_domain).collect();
        if let GpuResnetSegment::Chain(layers) = &mut owned[1].segments[0] {
            if let GpuCrownLayer::Linear { weight, .. } = &mut layers[0] {
                *weight = Arc::from(vec![0.81, -0.3, 0.2, 0.9]);
            }
        }
        let empty: Vec<Vec<f32>> = Vec::new();
        let refs = owned
            .iter()
            .map(|d| GpuResnetBatchedDomainRef {
                segments: &d.segments,
                input_lower: &d.lo,
                input_upper: &d.hi,
                beta_signed: &d.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect::<Vec<_>>();
        let seed = GpuCrownSeed {
            lower_a: Arc::from(vec![1.0, 0.0]),
            upper_a: Arc::from(vec![1.0, 0.0]),
            lower_b: Arc::from(vec![0.0]),
            upper_b: Arc::from(vec![0.0]),
            num_specs: 1,
            current_dim: 2,
        };
        assert!(
            resnet_backward_f64_wide_core(&engine, &refs, &seed, &[], &[], false, None, false)
                .is_err()
        );

        let mut owned: Vec<OwnedDomain> = (0..2).map(tiny_wide_domain).collect();
        owned[1].lo[0] = f32::NAN;
        let refs = owned
            .iter()
            .map(|d| GpuResnetBatchedDomainRef {
                segments: &d.segments,
                input_lower: &d.lo,
                input_upper: &d.hi,
                beta_signed: &d.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect::<Vec<_>>();
        assert!(
            resnet_backward_f64_wide_core(&engine, &refs, &seed, &[], &[], false, None, false)
                .is_err()
        );
    }

    #[test]
    fn up_down_outward_at_zero_boundary() {
        let tiny = 1e-60_f64;
        assert!(up(tiny) > 0.0);
        assert!(down(-tiny) < 0.0);
        assert_eq!(up(0.0), 0.0);
        assert_eq!(down(0.0), 0.0);
        assert!(f64::from(up(0.3)) >= 0.3 && f64::from(down(0.3)) <= 0.3);
        assert!(f64::from(up(-0.3)) >= -0.3 && f64::from(down(-0.3)) <= -0.3);
    }

    /// A linear-only (relu-free) net is exactly affine, so the SOUND f64 backward's
    /// `(lower, upper)` must enclose every sampled forward output. Uses the
    /// gate-free core (the test net is intentionally small).
    #[test]
    fn cuda_sound_crown_f64_linear_only_encloses_forward() {
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping (no CUDA device): {e}");
                return;
            }
        };
        let mut state: u64 = 0x50FA_1234;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..8 {
            let (din, dh, dout) = (6usize, 7usize, 4usize);
            let w1: Vec<f32> = (0..dh * din).map(|_| rng() * 0.8).collect();
            let b1: Vec<f32> = (0..dh).map(|_| rng() * 0.5).collect();
            let w2: Vec<f32> = (0..dout * dh).map(|_| rng() * 0.8).collect();
            let b2: Vec<f32> = (0..dout).map(|_| rng() * 0.5).collect();
            let layers = vec![
                GpuCrownLayer::Linear {
                    weight: Arc::from(w2.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b2.clone().into_boxed_slice())),
                    out_features: dout,
                    in_features: dh,
                },
                GpuCrownLayer::Linear {
                    weight: Arc::from(w1.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b1.clone().into_boxed_slice())),
                    out_features: dh,
                    in_features: din,
                },
            ];
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xr: Vec<f32> = (0..din).map(|_| (rng() * 0.3).abs() + 0.05).collect();
            let xl: Vec<f32> = (0..din).map(|i| xc[i] - xr[i]).collect();
            let xu: Vec<f32> = (0..din).map(|i| xc[i] + xr[i]).collect();
            let r =
                backward_f64_core(&eng, &layers, &spec, dout, &xl, &xu).expect("sound backward");
            let eval = |x: &[f32]| -> Vec<f32> {
                let mut h = vec![0.0f32; dh];
                for j in 0..dh {
                    let mut s = b1[j];
                    for i in 0..din {
                        s += w1[j * din + i] * x[i];
                    }
                    h[j] = s;
                }
                let mut o = vec![0.0f32; dout];
                for j in 0..dout {
                    let mut s = b2[j];
                    for i in 0..dh {
                        s += w2[j * dh + i] * h[i];
                    }
                    o[j] = s;
                }
                o
            };
            for t in 0..300 {
                let x: Vec<f32> = (0..din)
                    .map(|i| {
                        let f = ((t * 31 + i * 17) % 100) as f32 / 99.0;
                        xl[i] + f * (xu[i] - xl[i])
                    })
                    .collect();
                let o = eval(&x);
                for k in 0..dout {
                    assert!(
                        r.lower_bounds[k] <= o[k] + 1e-4 && o[k] <= r.upper_bounds[k] + 1e-4,
                        "UNSOUND: out[{k}]={} not in [{}, {}]",
                        o[k],
                        r.lower_bounds[k],
                        r.upper_bounds[k]
                    );
                }
            }
        }
    }

    /// SEEDED sound backward: the alpha-suffix frontier `A·y + b` (distinct lower &
    /// upper rows) must be enclosed by the returned bounds at every sampled forward
    /// output — i.e. `lower[s] <= A_l[s]·f(x)+b_l[s]` and `A_u[s]·f(x)+b_u[s] <=
    /// upper[s]`. Uses the gate-free seeded core (the test net is intentionally
    /// small). T1.3.
    #[test]
    fn cuda_seeded_sound_crown_f64_encloses_frontier() {
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping (no CUDA device): {e}");
                return;
            }
        };
        let mut state: u64 = 0x0DDF_ACE5;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..8 {
            let (din, dh, dout) = (6usize, 7usize, 4usize);
            let w1: Vec<f32> = (0..dh * din).map(|_| rng() * 0.8).collect();
            let b1: Vec<f32> = (0..dh).map(|_| rng() * 0.5).collect();
            let w2: Vec<f32> = (0..dout * dh).map(|_| rng() * 0.8).collect();
            let b2: Vec<f32> = (0..dout).map(|_| rng() * 0.5).collect();
            let layers = vec![
                GpuCrownLayer::Linear {
                    weight: Arc::from(w2.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b2.clone().into_boxed_slice())),
                    out_features: dout,
                    in_features: dh,
                },
                GpuCrownLayer::Linear {
                    weight: Arc::from(w1.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b1.clone().into_boxed_slice())),
                    out_features: dh,
                    in_features: din,
                },
            ];
            // Random alpha-suffix frontier (distinct lower & upper rows + biases).
            let num_specs = dout;
            let la: Vec<f32> = (0..num_specs * dout).map(|_| rng()).collect();
            let ua: Vec<f32> = (0..num_specs * dout).map(|_| rng()).collect();
            let lbv: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();
            let ubv: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();
            let seed = GpuCrownSeed {
                lower_a: Arc::from(la.clone().into_boxed_slice()),
                upper_a: Arc::from(ua.clone().into_boxed_slice()),
                lower_b: Arc::from(lbv.clone().into_boxed_slice()),
                upper_b: Arc::from(ubv.clone().into_boxed_slice()),
                num_specs,
                current_dim: dout,
            };
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xr: Vec<f32> = (0..din).map(|_| (rng() * 0.3).abs() + 0.05).collect();
            let xl: Vec<f32> = (0..din).map(|i| xc[i] - xr[i]).collect();
            let xu: Vec<f32> = (0..din).map(|i| xc[i] + xr[i]).collect();
            let r = seeded_backward_f64_core(&eng, &layers, &seed, &xl, &xu)
                .expect("seeded sound backward");
            let eval = |x: &[f32]| -> Vec<f32> {
                let mut h = vec![0.0f32; dh];
                for j in 0..dh {
                    let mut s = b1[j];
                    for i in 0..din {
                        s += w1[j * din + i] * x[i];
                    }
                    h[j] = s;
                }
                let mut o = vec![0.0f32; dout];
                for j in 0..dout {
                    let mut s = b2[j];
                    for i in 0..dh {
                        s += w2[j * dh + i] * h[i];
                    }
                    o[j] = s;
                }
                o
            };
            for t in 0..300 {
                let x: Vec<f32> = (0..din)
                    .map(|i| {
                        let f = ((t * 31 + i * 17) % 100) as f32 / 99.0;
                        xl[i] + f * (xu[i] - xl[i])
                    })
                    .collect();
                let o = eval(&x);
                for s in 0..num_specs {
                    let mut lo_frontier = f64::from(lbv[s]);
                    let mut hi_frontier = f64::from(ubv[s]);
                    for k in 0..dout {
                        lo_frontier += f64::from(la[s * dout + k]) * f64::from(o[k]);
                        hi_frontier += f64::from(ua[s * dout + k]) * f64::from(o[k]);
                    }
                    assert!(
                        f64::from(r.lower_bounds[s]) <= lo_frontier + 1e-3,
                        "UNSOUND lower: spec {s} bound {} > frontier {lo_frontier}",
                        r.lower_bounds[s]
                    );
                    assert!(
                        hi_frontier <= f64::from(r.upper_bounds[s]) + 1e-3,
                        "UNSOUND upper: frontier {hi_frontier} > spec {s} bound {}",
                        r.upper_bounds[s]
                    );
                }
            }
        }
    }

    /// RESNET seeded backward, identity skip `out = W·z + b + z`: the frontier
    /// `A·out + b_seed` (distinct lower/upper rows) must be enclosed at every sampled
    /// z. Exercises the `Residual` identity merge `A_z = backward_F(A) + A`. Affine
    /// branch ⇒ exact concrete forward, no ReLU relaxation to set up. Gate-free core.
    /// T1.3.
    #[test]
    fn cuda_resnet_identity_sound_f64_encloses_frontier() {
        use ny_core::GpuResnetSegment;
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping (no CUDA device): {e}");
                return;
            }
        };
        let mut state: u64 = 0xBEEF_5AFE;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..8 {
            let d = 5usize;
            let wf: Vec<f32> = (0..d * d).map(|_| rng() * 0.6).collect();
            let bf: Vec<f32> = (0..d).map(|_| rng() * 0.4).collect();
            // out = F(z) + z, F(z) = wf·z + bf  ⇒  out = (wf + I)·z + bf.
            let segments = vec![GpuResnetSegment::Residual(vec![GpuCrownLayer::Linear {
                weight: Arc::from(wf.clone().into_boxed_slice()),
                bias: Some(Arc::from(bf.clone().into_boxed_slice())),
                out_features: d,
                in_features: d,
            }])];
            let num_specs = 3usize;
            let la: Vec<f32> = (0..num_specs * d).map(|_| rng()).collect();
            let ua: Vec<f32> = (0..num_specs * d).map(|_| rng()).collect();
            let lbv: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();
            let ubv: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();
            let seed = GpuCrownSeed {
                lower_a: Arc::from(la.clone().into_boxed_slice()),
                upper_a: Arc::from(ua.clone().into_boxed_slice()),
                lower_b: Arc::from(lbv.clone().into_boxed_slice()),
                upper_b: Arc::from(ubv.clone().into_boxed_slice()),
                num_specs,
                current_dim: d,
            };
            let xc: Vec<f32> = (0..d).map(|_| rng()).collect();
            let xr: Vec<f32> = (0..d).map(|_| (rng() * 0.3).abs() + 0.05).collect();
            let xl: Vec<f32> = (0..d).map(|i| xc[i] - xr[i]).collect();
            let xu: Vec<f32> = (0..d).map(|i| xc[i] + xr[i]).collect();
            let r = resnet_backward_f64_core(&eng, &segments, &seed, &xl, &xu, &mut None)
                .expect("resnet identity backward");
            for t in 0..300 {
                let z: Vec<f32> = (0..d)
                    .map(|i| {
                        let f = ((t * 31 + i * 17) % 100) as f32 / 99.0;
                        xl[i] + f * (xu[i] - xl[i])
                    })
                    .collect();
                // out = wf·z + bf + z
                let mut out = vec![0.0f64; d];
                for (j, o) in out.iter_mut().enumerate() {
                    let mut s = f64::from(bf[j]) + f64::from(z[j]);
                    for (i, &zi) in z.iter().enumerate() {
                        s += f64::from(wf[j * d + i]) * f64::from(zi);
                    }
                    *o = s;
                }
                for s in 0..num_specs {
                    let mut lo = f64::from(lbv[s]);
                    let mut hi = f64::from(ubv[s]);
                    for k in 0..d {
                        lo += f64::from(la[s * d + k]) * out[k];
                        hi += f64::from(ua[s * d + k]) * out[k];
                    }
                    assert!(
                        f64::from(r.lower_bounds[s]) <= lo + 1e-3,
                        "UNSOUND lower spec {s}: {} > {lo}",
                        r.lower_bounds[s]
                    );
                    assert!(
                        hi <= f64::from(r.upper_bounds[s]) + 1e-3,
                        "UNSOUND upper spec {s}: {hi} > {}",
                        r.upper_bounds[s]
                    );
                }
            }
        }
    }

    /// RESNET projection skip `out = F(z) + P(z)` (both affine): exercises the
    /// `ResidualProj` merge `A_z = backward_F(A) + backward_P(A)` with the outer bias
    /// counted once. Gate-free core. T1.3.
    #[test]
    fn cuda_resnet_proj_sound_f64_encloses_frontier() {
        use ny_core::GpuResnetSegment;
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping (no CUDA device): {e}");
                return;
            }
        };
        let mut state: u64 = 0x1DEA_C0DE;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..8 {
            let (din, dout) = (5usize, 4usize);
            let wf: Vec<f32> = (0..dout * din).map(|_| rng() * 0.6).collect();
            let bf: Vec<f32> = (0..dout).map(|_| rng() * 0.4).collect();
            let wp: Vec<f32> = (0..dout * din).map(|_| rng() * 0.6).collect();
            let bp: Vec<f32> = (0..dout).map(|_| rng() * 0.4).collect();
            // out = F(z) + P(z) = (wf+wp)·z + (bf+bp).
            let segments = vec![GpuResnetSegment::ResidualProj(
                vec![GpuCrownLayer::Linear {
                    weight: Arc::from(wf.clone().into_boxed_slice()),
                    bias: Some(Arc::from(bf.clone().into_boxed_slice())),
                    out_features: dout,
                    in_features: din,
                }],
                vec![GpuCrownLayer::Linear {
                    weight: Arc::from(wp.clone().into_boxed_slice()),
                    bias: Some(Arc::from(bp.clone().into_boxed_slice())),
                    out_features: dout,
                    in_features: din,
                }],
            )];
            let num_specs = 3usize;
            let la: Vec<f32> = (0..num_specs * dout).map(|_| rng()).collect();
            let ua: Vec<f32> = (0..num_specs * dout).map(|_| rng()).collect();
            let lbv: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();
            let ubv: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();
            let seed = GpuCrownSeed {
                lower_a: Arc::from(la.clone().into_boxed_slice()),
                upper_a: Arc::from(ua.clone().into_boxed_slice()),
                lower_b: Arc::from(lbv.clone().into_boxed_slice()),
                upper_b: Arc::from(ubv.clone().into_boxed_slice()),
                num_specs,
                current_dim: dout,
            };
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xr: Vec<f32> = (0..din).map(|_| (rng() * 0.3).abs() + 0.05).collect();
            let xl: Vec<f32> = (0..din).map(|i| xc[i] - xr[i]).collect();
            let xu: Vec<f32> = (0..din).map(|i| xc[i] + xr[i]).collect();
            let r = resnet_backward_f64_core(&eng, &segments, &seed, &xl, &xu, &mut None)
                .expect("resnet proj backward");
            for t in 0..300 {
                let z: Vec<f32> = (0..din)
                    .map(|i| {
                        let f = ((t * 31 + i * 17) % 100) as f32 / 99.0;
                        xl[i] + f * (xu[i] - xl[i])
                    })
                    .collect();
                let mut out = vec![0.0f64; dout];
                for (j, o) in out.iter_mut().enumerate() {
                    let mut s = f64::from(bf[j]) + f64::from(bp[j]);
                    for (i, &zi) in z.iter().enumerate() {
                        s += (f64::from(wf[j * din + i]) + f64::from(wp[j * din + i]))
                            * f64::from(zi);
                    }
                    *o = s;
                }
                for s in 0..num_specs {
                    let mut lo = f64::from(lbv[s]);
                    let mut hi = f64::from(ubv[s]);
                    for k in 0..dout {
                        lo += f64::from(la[s * dout + k]) * out[k];
                        hi += f64::from(ua[s * dout + k]) * out[k];
                    }
                    assert!(
                        f64::from(r.lower_bounds[s]) <= lo + 1e-3,
                        "UNSOUND lower spec {s}: {} > {lo}",
                        r.lower_bounds[s]
                    );
                    assert!(
                        hi <= f64::from(r.upper_bounds[s]) + 1e-3,
                        "UNSOUND upper spec {s}: {hi} > {}",
                        r.upper_bounds[s]
                    );
                }
            }
        }
    }

    /// β-CROWN resnet backward MECHANICS: a stable-ACTIVE single-ReLU chain (relu ≡
    /// identity, slope 1) with an identity seed makes the base bound exactly the input
    /// box `[xl, xu]`. Folding `beta_signed[0][k] = β_k` must scale the k-th diagonal
    /// coefficient to `1 ∓ β_k`, so the bound becomes `[(1−β_k)·xl_k, (1+β_k)·xu_k]`
    /// (outward-rounded). Validates: right neuron (diagonal k), right sign
    /// (lower−=β / upper+=β), right magnitude — and that the folded bound still
    /// ENCLOSES the concrete forward on the full active box (β on a vacuously-satisfied
    /// active constraint only widens ⇒ sound). β=0 must reproduce the base EXACTLY.
    /// T1.3.
    #[test]
    fn cuda_resnet_beta_fold_is_applied_soundly() {
        use ny_core::{GpuCrownLayer as L, GpuResnetSegment};
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping (no CUDA device): {e}");
                return;
            }
        };
        let d = 4usize;
        // Stable-active ReLU: relu(x) = x for x >= 0. slope 1, intercept 0.
        let relu = L::Activation {
            lower_slope: vec![1.0f32; d],
            upper_slope: vec![1.0f32; d],
            lower_intercept: vec![0.0f32; d],
            upper_intercept: vec![0.0f32; d],
            num_neurons: d,
        };
        let segments = vec![GpuResnetSegment::Chain(vec![relu])];
        // Active box: xl >= 0 (all neurons active ⇒ relu is exact identity).
        let xl: Vec<f32> = (0..d).map(|i| 0.2 + i as f32 * 0.1).collect();
        let xu: Vec<f32> = (0..d).map(|i| 1.0 + i as f32 * 0.2).collect();

        // (1) MECHANICAL: single spec on neuron 0 (seed row e_0), β only at neuron 0.
        // Then the folded lower row is [1−β_0, 0, …] ⇒ lower = (1−β_0)·xl_0, and
        // upper = (1+β_0)·xu_0 — isolating the fold's sign & magnitude at one neuron.
        let mut e0 = vec![0.0f32; d];
        e0[0] = 1.0;
        let seed1 = GpuCrownSeed {
            lower_a: Arc::from(e0.clone().into_boxed_slice()),
            upper_a: Arc::from(e0.into_boxed_slice()),
            lower_b: Arc::from(vec![0.0f32].into_boxed_slice()),
            upper_b: Arc::from(vec![0.0f32].into_boxed_slice()),
            num_specs: 1,
            current_dim: d,
        };
        let base1 = resnet_backward_f64_core(&eng, &segments, &seed1, &xl, &xu, &mut None).unwrap();
        assert!((f64::from(base1.lower_bounds[0]) - f64::from(xl[0])).abs() < 1e-4);
        assert!((f64::from(base1.upper_bounds[0]) - f64::from(xu[0])).abs() < 1e-4);
        // β=0 reproduces the base EXACTLY (plumbing no-op).
        let z1 = vec![vec![0.0f32; d]];
        let b0 = resnet_backward_f64_core(
            &eng,
            &segments,
            &seed1,
            &xl,
            &xu,
            &mut Some(ActAux::beta(&z1)),
        )
        .unwrap();
        assert_eq!(base1.lower_bounds[0], b0.lower_bounds[0]);
        assert_eq!(base1.upper_bounds[0], b0.upper_bounds[0]);
        // β_0 only.
        let b0v = 0.3f32;
        let mut bs = vec![0.0f32; d];
        bs[0] = b0v;
        let one = vec![bs];
        let bet1 = resnet_backward_f64_core(
            &eng,
            &segments,
            &seed1,
            &xl,
            &xu,
            &mut Some(ActAux::beta(&one)),
        )
        .unwrap();
        let exp_lo = f64::from(1.0 - b0v) * f64::from(xl[0]);
        let exp_hi = f64::from(1.0 + b0v) * f64::from(xu[0]);
        assert!(
            (f64::from(bet1.lower_bounds[0]) - exp_lo).abs() < 1e-3,
            "β lower {} != expected {exp_lo}",
            bet1.lower_bounds[0]
        );
        assert!(
            (f64::from(bet1.upper_bounds[0]) - exp_hi).abs() < 1e-3,
            "β upper {} != expected {exp_hi}",
            bet1.upper_bounds[0]
        );

        // (2) SOUNDNESS: a full identity seed + a β at EVERY neuron. β on a
        // vacuously-satisfied active constraint only widens ⇒ the folded bound must
        // still enclose the concrete forward (out_s = x_s) over the whole active box.
        let mut la = vec![0.0f32; d * d];
        for i in 0..d {
            la[i * d + i] = 1.0;
        }
        let seedd = GpuCrownSeed {
            lower_a: Arc::from(la.clone().into_boxed_slice()),
            upper_a: Arc::from(la.into_boxed_slice()),
            lower_b: Arc::from(vec![0.0f32; d].into_boxed_slice()),
            upper_b: Arc::from(vec![0.0f32; d].into_boxed_slice()),
            num_specs: d,
            current_dim: d,
        };
        let betas: Vec<f32> = (0..d).map(|i| 0.1 + i as f32 * 0.15).collect();
        let bd = vec![betas];
        let betd = resnet_backward_f64_core(
            &eng,
            &segments,
            &seedd,
            &xl,
            &xu,
            &mut Some(ActAux::beta(&bd)),
        )
        .unwrap();
        for s in 0..d {
            // out_s = x_s in [xl_s, xu_s]; the sound bound must bracket the whole range.
            assert!(
                f64::from(betd.lower_bounds[s]) <= f64::from(xl[s]) + 1e-4,
                "UNSOUND β lower[{s}]={} > xl {}",
                betd.lower_bounds[s],
                xl[s]
            );
            assert!(
                f64::from(betd.upper_bounds[s]) >= f64::from(xu[s]) - 1e-4,
                "UNSOUND β upper[{s}]={} < xu {}",
                betd.upper_bounds[s],
                xu[s]
            );
        }

        // (3) A β list of the wrong fold length is rejected.
        let bad = vec![vec![0.0f32; d + 1]];
        assert!(
            resnet_backward_f64_core(
                &eng,
                &segments,
                &seed1,
                &xl,
                &xu,
                &mut Some(ActAux::beta(&bad))
            )
            .is_err(),
            "wrong-length beta must be rejected"
        );
    }

    /// GRAD + β-GRAD capture MECHANICS on a stable-active single-ReLU chain with an
    /// identity seed: the pre-transform lower coeff AT the ReLU is exactly `I`, so
    /// `Σ_j max(A_lower[j,i],0) = 1` ⇒ `relu_grads[0][i] = pre_lower[0][i]`, and the
    /// β-gather of column c for row r is `I[r,c] = (r==c)`. Also checks the captures
    /// leave the BOUNDS identical to the base/beta path. Non-soundness-critical, but
    /// the layout must match the CPU gradient rule. T1.3.
    #[test]
    fn cuda_resnet_grad_and_beta_grad_capture_layout() {
        use ny_core::{GpuCrownLayer as L, GpuResnetSegment};
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping (no CUDA device): {e}");
                return;
            }
        };
        let d = 4usize;
        let relu = L::Activation {
            lower_slope: vec![1.0f32; d],
            upper_slope: vec![1.0f32; d],
            lower_intercept: vec![0.0f32; d],
            upper_intercept: vec![0.0f32; d],
            num_neurons: d,
        };
        let segments = vec![GpuResnetSegment::Chain(vec![relu])];
        let mut la = vec![0.0f32; d * d];
        for i in 0..d {
            la[i * d + i] = 1.0;
        }
        let seed = GpuCrownSeed {
            lower_a: Arc::from(la.clone().into_boxed_slice()),
            upper_a: Arc::from(la.into_boxed_slice()),
            lower_b: Arc::from(vec![0.0f32; d].into_boxed_slice()),
            upper_b: Arc::from(vec![0.0f32; d].into_boxed_slice()),
            num_specs: d,
            current_dim: d,
        };
        let xl: Vec<f32> = (0..d).map(|i| 0.2 + i as f32 * 0.1).collect();
        let xu: Vec<f32> = (0..d).map(|i| 1.0 + i as f32 * 0.2).collect();
        let base = resnet_backward_f64_core(&eng, &segments, &seed, &xl, &xu, &mut None).unwrap();

        // GRAD: relu_grads[0][i] == pre_lower[0][i].
        let pre_lower = vec![(0..d).map(|i| 0.3 + i as f32 * 0.2).collect::<Vec<f32>>()];
        let mut aux_g = Some(ActAux::grad(&pre_lower));
        let rg = resnet_backward_f64_core(&eng, &segments, &seed, &xl, &xu, &mut aux_g).unwrap();
        let grads = aux_g.unwrap().relu_grads;
        assert_eq!(grads.len(), 1, "one ReLU ⇒ one grad row");
        for i in 0..d {
            assert!(
                (grads[0][i] - pre_lower[0][i]).abs() < 1e-5,
                "grad[{i}]={} != pre_lower {}",
                grads[0][i],
                pre_lower[0][i]
            );
            assert_eq!(
                rg.lower_bounds[i], base.lower_bounds[i],
                "grad must not change bounds"
            );
            assert_eq!(
                rg.upper_bounds[i], base.upper_bounds[i],
                "grad must not change bounds"
            );
        }

        // β-GRAD: gather cols {0, 2}; beta_gather[0] = row-major d×2 of I[row, col].
        let gather_idx: Vec<Vec<u32>> = vec![vec![0, 2]];
        let zero_beta = vec![vec![0.0f32; d]];
        let mut aux_bg = Some(ActAux::beta_grad(&zero_beta, &gather_idx));
        let rbg = resnet_backward_f64_core(&eng, &segments, &seed, &xl, &xu, &mut aux_bg).unwrap();
        let gather = aux_bg.unwrap().beta_gather;
        assert_eq!(gather.len(), 1);
        assert_eq!(gather[0].len(), d * 2, "row-major specs × |idx|");
        for row in 0..d {
            for (j, &col) in [0u32, 2].iter().enumerate() {
                let expect = if row == col as usize { 1.0 } else { 0.0 };
                assert!(
                    (gather[0][row * 2 + j] - expect).abs() < 1e-5,
                    "gather[row={row}, col={col}]={} != {expect}",
                    gather[0][row * 2 + j]
                );
            }
        }
        // β=0 ⇒ bounds unchanged from base.
        for i in 0..d {
            assert_eq!(rbg.lower_bounds[i], base.lower_bounds[i]);
            assert_eq!(rbg.upper_bounds[i], base.upper_bounds[i]);
        }
    }

    /// SOUND f64 Conv2d CROWN backward: a single conv layer (affine) with a
    /// full-identity output seed makes each spec's bound the reachable range of one
    /// conv output neuron. The bounds must ENCLOSE the concrete conv forward at every
    /// sampled input. Exercises the reshape→GEMM→col2im transposed-conv + bias fold +
    /// certified error. Gate-free core (small net). T1.3 conv.
    #[test]
    fn cuda_conv_sound_f64_encloses_forward() {
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping (no CUDA device): {e}");
                return;
            }
        };
        let mut state: u64 = 0xC02F_1234;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let (ic, oc, kh, kw) = (2usize, 3usize, 3usize, 3usize);
        let (sh, sw, ph, pw) = (1usize, 1usize, 1usize, 1usize);
        let (ih, iw) = (4usize, 4usize);
        let oh = (ih + 2 * ph - kh) / sh + 1;
        let ow = (iw + 2 * pw - kw) / sw + 1;
        let n = ic * kh * kw;
        let out_d = oc * oh * ow;
        let in_d = ic * ih * iw;
        for _ in 0..4 {
            let w_col: Vec<f32> = (0..oc * n).map(|_| rng() * 0.5).collect();
            let bias: Vec<f32> = (0..out_d).map(|_| rng() * 0.3).collect();
            let layers = vec![GpuCrownLayer::Conv2d {
                weight_col: Arc::from(w_col.clone().into_boxed_slice()),
                bias_expanded: Some(Arc::from(bias.clone().into_boxed_slice())),
                out_channels: oc,
                in_channels: ic,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                pad_h: ph,
                pad_w: pw,
                out_h: oh,
                out_w: ow,
                in_h: ih,
                in_w: iw,
            }];
            // Full identity seed: one spec per conv-output neuron.
            let mut spec = vec![0.0f32; out_d * out_d];
            for i in 0..out_d {
                spec[i * out_d + i] = 1.0;
            }
            let xc: Vec<f32> = (0..in_d).map(|_| rng()).collect();
            let xr: Vec<f32> = (0..in_d).map(|_| (rng() * 0.3).abs() + 0.05).collect();
            let xl: Vec<f32> = (0..in_d).map(|i| xc[i] - xr[i]).collect();
            let xu: Vec<f32> = (0..in_d).map(|i| xc[i] + xr[i]).collect();
            let r =
                backward_f64_core(&eng, &layers, &spec, out_d, &xl, &xu).expect("conv backward");

            let conv_fwd = |x: &[f32]| -> Vec<f64> {
                let mut out = vec![0.0f64; out_d];
                for oc_i in 0..oc {
                    for oh_i in 0..oh {
                        for ow_i in 0..ow {
                            let k = oc_i * (oh * ow) + oh_i * ow + ow_i;
                            let mut acc = f64::from(bias[k]);
                            for ic_i in 0..ic {
                                for kh_i in 0..kh {
                                    let y = (oh_i * sh + kh_i) as isize - ph as isize;
                                    if y < 0 || y >= ih as isize {
                                        continue;
                                    }
                                    for kw_i in 0..kw {
                                        let xx = (ow_i * sw + kw_i) as isize - pw as isize;
                                        if xx < 0 || xx >= iw as isize {
                                            continue;
                                        }
                                        let wc = ic_i * (kh * kw) + kh_i * kw + kw_i;
                                        let xi =
                                            ic_i * (ih * iw) + (y as usize) * iw + (xx as usize);
                                        acc += f64::from(w_col[oc_i * n + wc]) * f64::from(x[xi]);
                                    }
                                }
                            }
                            out[k] = acc;
                        }
                    }
                }
                out
            };
            for t in 0..150 {
                let x: Vec<f32> = (0..in_d)
                    .map(|i| {
                        let f = ((t * 31 + i * 17) % 100) as f32 / 99.0;
                        xl[i] + f * (xu[i] - xl[i])
                    })
                    .collect();
                let o = conv_fwd(&x);
                for k in 0..out_d {
                    assert!(
                        f64::from(r.lower_bounds[k]) <= o[k] + 1e-3,
                        "UNSOUND conv lower[{k}]={} > {}",
                        r.lower_bounds[k],
                        o[k]
                    );
                    assert!(
                        o[k] <= f64::from(r.upper_bounds[k]) + 1e-3,
                        "UNSOUND conv upper[{k}]={} < {}",
                        r.upper_bounds[k],
                        o[k]
                    );
                }
            }
        }
    }

    /// The size-gate routes tiny nets to CPU (`UnsupportedOp`).
    #[test]
    fn size_gate_rejects_small_nets() {
        let eng = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        let layers = vec![GpuCrownLayer::Linear {
            weight: Arc::from(vec![0.1f32; 4 * 4].into_boxed_slice()),
            bias: None,
            out_features: 4,
            in_features: 4,
        }];
        let spec = vec![1.0f32; 4 * 4];
        let r = crown_backward_gpu_sound_impl(&eng, &layers, &spec, 4, &[0.0; 4], &[1.0; 4]);
        assert!(
            matches!(r, Err(NyError::UnsupportedOp(_))),
            "tiny net must hit the size-gate"
        );
    }
}
