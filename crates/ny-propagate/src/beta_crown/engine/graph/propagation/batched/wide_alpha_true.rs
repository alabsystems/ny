// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! TRUE per-subdomain alpha gradient for the wide alpha+beta ascent
//! (#cifar100 task 11, dark gate `NY_WIDE_ALPHA_TRUE=1`).
//!
//! The 2026-07-08 prop_1498 A/B refuted the LOCAL gradient rule
//! (`pre_lower[i] * sum_rows max(A_lower[j,i], 0)`): it degraded bounds
//! proportionally to lr in BOTH signs. The finite-difference oracle
//! (`network/graph_alpha/backward/true_grad_oracle_tests.rs`) shows the true
//! gradient of the critical spec row's lower bound w.r.t. the lower slope
//! `alpha_i` of ReLU `k` is
//!
//! ```text
//!   dlb/dalpha_i = nu_k[i] * hhat_k[i](x*)     if nu_k[i] > 0, else 0
//! ```
//!
//! where `nu_k` is the PRE-relaxation lower coefficient row when the backward
//! reaches ReLU `k` (branch select `a >= 0 -> lower_slope`, matching
//! CROWN_ACTIVATION_RESIDENT_SHADER), `x*` is the concretization argmin corner
//! of the FINAL folded row (`x*_j = xl_j` if `final_A[j] > 0` else `xu_j`),
//! and `hhat_k[i](x*)` is the RELAXED-linear forward evaluation of neuron i's
//! pre-activation at `x*` (earlier ReLUs apply the backward-selected affine
//! relaxation `slope*z + intercept`, NOT the concrete ReLU; the beta fold adds
//! per-neuron CONSTANTS to coefficient rows, so it does not enter the forward
//! derivative path).
//!
//! This module replays the CRITICAL row's backward on the HOST over the
//! domain's own `GpuResnetSegment`s (the same slopes/intercepts/beta the wide
//! GPU fold consumes — write_back_alpha keeps them current), harvesting `nu`
//! at every Activation and the final input-level row, then runs the relaxed
//! forward at `x*`. One (row backward + forward) per participating domain per
//! replay. THROUGHPUT (#task-11 gate follow-up): the conv walks run as faer
//! GEMMs (transpose-conv GEMM + col2im backward; im2col + GEMM forward — the
//! naive scalar loops cost ~0.5s/replay on the cifar100 resnet and starved
//! BaB), the per-domain replays run in parallel on rayon, and the caller can
//! throttle replays with `NY_WIDE_ALPHA_TRUE_EVERY=k` /
//! `NY_WIDE_ALPHA_TRUE_DOMS=worst` (Adam steps on the last captured gradient
//! between replays — sound, α stays in [0,1]).
//!
//! FAIL-CLOSED VALIDATION: the replayed row concretizes to a lower bound that
//! must agree with the GPU fold's bound for the same row (the fold is
//! error-widened, so `gpu_lb <= replay_lb + eps`; gross disagreement means a
//! walk/convention mismatch). On any mismatch the domain's alpha step is
//! skipped for that iteration (gradients only steer alpha, so this is purely
//! a quality guard — soundness always comes from the GPU fold itself).

use ny_core::head_f64_fold::HeadF64Fold;
use ny_core::{GpuCrownLayer, GpuResnetSegment};

/// Env gate for the TRUE wide-alpha gradient (dark, default off). Only
/// meaningful on top of `NY_BAB_RESNET_WIDE_ALPHA=1`.
pub(in crate::beta_crown::engine::graph) fn wide_alpha_true_enabled() -> bool {
    matches!(
        std::env::var("NY_WIDE_ALPHA_TRUE").ok().as_deref(),
        Some("1")
    )
}

/// STEP gate (`NY_WIDE_ALPHA_TRUE_STEP=1`, dark, default off — BLOCKER-1 fix).
///
/// ROOT CAUSE (2026-07-14): under `NY_WIDE_ALPHA_TRUE=1` the FD-oracle-validated
/// host-replay gradient (`true_alpha_grads_for_row`, the CRITICAL/obj-row
/// `∂lb_crit/∂α`) is COMPUTED and cached, then used ONLY as a pass/skip GATE —
/// the α Adam step is actually taken with a DIFFERENT gradient `joint_d`
/// (`crown_joint_alpha_gradient_resident` / `joint_alpha_gradient`), which is the
/// SUM-over-all-specs adjoint `∂(Σ_s lb_s)/∂α`. Steering the ONE shared per-neuron
/// α by the all-spec sum lands a compromise slope that is slack for the binding
/// worst row, so the subdomain ascent never converges to the worst row's LP
/// (the +0.09 prop1498 residual). The host-optimized direction is literally not
/// the one the fold consumes — the "step not realized by the fold".
///
/// With this gate on (and `true_mode`), the α step consumes `true_grads` — the
/// exact FD-validated crit/obj-row gradient — so the direction the host replay
/// optimized IS the direction the next wide fold sees baked into `seg_store[d]`.
/// This also makes the documented cadence semantics true (between replays Adam
/// steps on the LAST captured HOST gradient, not a fresh all-spec `joint_d`) and
/// makes the `NY_AB_PARITY` per-spec round-robin actually reach the α step.
///
/// SOUND: any α ∈ [0,1] is a valid ReLU lower slope; the reported bound is always
/// the sound wide fold, element-wise best-iterate merged. Default off ⇒ the
/// applied gradient is `joint_d` exactly as before ⇒ byte-identical.
pub(in crate::beta_crown::engine::graph) fn wide_alpha_true_step_enabled() -> bool {
    matches!(
        std::env::var("NY_WIDE_ALPHA_TRUE_STEP").ok().as_deref(),
        Some("1")
    )
}

/// Parse the replay cadence knob: replay only on ascent iterations where
/// `iter % k == 0`. Invalid/absent/0 ⇒ 1 (replay every iteration — the
/// original behavior).
fn parse_true_every(v: Option<&str>) -> usize {
    v.and_then(|s| s.parse::<usize>().ok())
        .filter(|&k| k >= 1)
        .unwrap_or(1)
}

/// Replay cadence (`NY_WIDE_ALPHA_TRUE_EVERY=k`, default 1): the host replay
/// runs only on ascent iterations with `iter % k == 0`. Between replays Adam
/// keeps stepping on the LAST captured gradient — stale-gradient steps are
/// sound (any α ∈ [0,1] is valid; the best-iterate merge keeps only sound
/// improvements) and only affect step quality (#task-11 throughput lever 1).
pub(in crate::beta_crown::engine::graph) fn wide_alpha_true_every() -> usize {
    parse_true_every(std::env::var("NY_WIDE_ALPHA_TRUE_EVERY").ok().as_deref())
}

/// Domain-selection knob (`NY_WIDE_ALPHA_TRUE_DOMS=worst|all`, default all):
/// `worst` replays only the active domain with the smallest critical margin
/// each replay iteration; the others keep stepping on their cached gradient
/// (if any) or skip the α step (β still steps).
pub(in crate::beta_crown::engine::graph) fn wide_alpha_true_worst_only() -> bool {
    matches!(
        std::env::var("NY_WIDE_ALPHA_TRUE_DOMS").ok().as_deref(),
        Some("worst")
    )
}

/// Per-domain UNSHARED-α persistence gate (`NY_WIDE_ALPHA_UNSHARED=1`, dark,
/// default off — #hard-six prop8945 survivor lever 2).
///
/// RECON FINDING (2026-07-12): `GraphDomainAlphaState` is already PER-NEURON
/// (per-`(node, neuron_idx)` params, gradients, and Adam moments — no spatial
/// tying anywhere in the ascent), BUT `gpu_beta_optimize_wide` DISCARDS the
/// Adam-ascended α clones at the end of every batch call: only β persists to
/// the children (`optimized_betas`). Every domain at every depth therefore
/// re-seeds its α ascent from the ROOT-inherited α — functionally a SHARED
/// root α across the whole tree, with only `iterations-1` steps of per-domain
/// specialization that evaporate each batch.
///
/// With this gate on, each participating domain's best-margin α snapshot is
/// returned alongside the β (same snapshot point as `best_beta`) and written
/// back onto the evaluated child, so `GraphDomainAlphaState::from_parent`
/// inheritance ACCUMULATES the ascent along the branch — per-domain unshared
/// α at the pinned tail, compounding with depth like β does.
///
/// SOUND regardless: α only parameterizes the lower-relaxation slope in
/// [0, 1]; every bound the verifier reports is the sound wide fold for the
/// α/β actually used. Persistence only changes the ascent's starting point.
pub(in crate::beta_crown::engine::graph) fn wide_alpha_unshared_enabled() -> bool {
    matches!(
        std::env::var("NY_WIDE_ALPHA_UNSHARED").ok().as_deref(),
        Some("1")
    )
}

/// auto_LiRPA per-spec α parity gate (`NY_AB_PARITY=1`, dark, default off).
///
/// STUDY-1/STUDY-2 both isolate the one structural difference behind αβ-CROWN's
/// ~0.25-tighter per-subdomain bound: auto_LiRPA keeps a SEPARATE optimizable
/// lower slope α per (target-spec, source-ReLU) pair (`self.alpha[start_node]`,
/// shape `(2, spec_dim, batch, *relu_shape)`), so the sum-of-margins objective
/// `Σ_i lb_i` DECOUPLES and every spec reaches its own α optimum. NY holds ONE
/// α per (node, neuron) and steers it only toward the single worst (`crit_row`)
/// spec, landing a compromise slope that is slack for the OTHER binding specs.
///
/// This gate realizes the decoupling WITHOUT a per-spec α tensor (which would
/// require a ny-core GPU-kernel spec axis): the wide ascent already returns
/// `ds.best_lo` = the ELEMENT-WISE per-row best lower bound across iterations,
/// so if the α-optimization objective row ROUND-ROBINS over the active
/// (unverified) specs — one spec targeted per replay iteration — then each
/// spec's own-α iterate is captured into its `best_lo[s]`. The reported
/// per-domain margin `min_s best_lo[s]` then reflects each spec optimized with
/// its own α, exactly auto_LiRPA's per-spec decoupling, reusing the existing
/// FD-oracle-validated host replay (`true_alpha_grads_for_row`).
///
/// SOUND by construction: any α ∈ [0, 1] is a valid ReLU lower slope; the
/// objective-row choice only steers which α direction Adam takes. Every iterate
/// is the sound wide fold and `best_lo` keeps only element-wise sound maxima, so
/// the bound can never regress below the crit-only trajectory's captured maxima.
/// Turning the gate ON also forces the true host-replay lane so the per-row
/// steering has a per-row gradient to follow.
pub(in crate::beta_crown::engine::graph) fn ab_parity_enabled() -> bool {
    matches!(std::env::var("NY_AB_PARITY").ok().as_deref(), Some("1"))
}

/// Ascent-iteration floor under `NY_AB_PARITY` (`NY_AB_PARITY_ITERS=k`, default
/// 10). The round-robin needs `iterations - 1 ≥ n_active` to target every
/// active spec at least once (the last iteration merges but does not step), so
/// the floor guarantees typical active-spec counts (~8) are all covered even
/// when the base `NY_MO_GPU_BETA_ITERS` budget is the parity-default 3.
pub(in crate::beta_crown::engine::graph) fn ab_parity_iters() -> usize {
    std::env::var("NY_AB_PARITY_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&k| k >= 2)
        .unwrap_or(10)
}

/// Profiling gate (`NY_WIDE_ALPHA_TRUE_PROF=1`): per-replay phase/layer-kind
/// timing breakdown on stderr. Dark; costs two `Instant::now()` per layer when
/// on, nothing when off.
fn prof_enabled() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("NY_WIDE_ALPHA_TRUE_PROF").ok().as_deref() == Some("1"))
}

thread_local! {
    /// Per-thread accumulator (each domain's replay runs whole on one rayon
    /// worker): [conv_bwd, lin_bwd, act_bwd, conv_fwd, lin_fwd, act_fwd] secs.
    static PROF_ACC: std::cell::Cell<[f64; 6]> = const { std::cell::Cell::new([0.0; 6]) };
}

fn prof_add(slot: usize, dt: f64) {
    PROF_ACC.with(|a| {
        let mut v = a.get();
        v[slot] += dt;
        a.set(v);
    });
}

/// Result of replaying one spec row's backward over a domain's segments.
pub(super) struct CriticalRowReplay {
    /// PRE-relaxation lower coefficient row at each Activation (fold order).
    pub nu: Vec<Vec<f32>>,
    /// Final input-level coefficient row.
    pub final_a: Vec<f32>,
    /// Final accumulated bias.
    pub final_b: f32,
}

impl CriticalRowReplay {
    /// Concretized lower bound of the replayed row over the input box
    /// (matches concretize.rs: `lb += a>0 ? a*xl : a*xu`).
    pub fn lower_bound(&self, in_lo: &[f32], in_hi: &[f32]) -> f32 {
        let mut acc = self.final_b as f64;
        for (j, &a) in self.final_a.iter().enumerate() {
            let x = if a > 0.0 { in_lo[j] } else { in_hi[j] };
            acc += a as f64 * x as f64;
        }
        acc as f32
    }

    /// Concretization argmin corner of the replayed row.
    pub fn argmin_corner(&self, in_lo: &[f32], in_hi: &[f32]) -> Vec<f32> {
        self.final_a
            .iter()
            .enumerate()
            .map(|(j, &a)| if a > 0.0 { in_lo[j] } else { in_hi[j] })
            .collect()
    }
}

/// Backward one branch (layers already in backward order: output -> input).
/// `beta` holds the per-Activation signed-beta slices for THIS branch's
/// Activations in slice order; `nu_out` collects the pre-relaxation rows.
fn backward_branch(
    layers: &[GpuCrownLayer],
    mut coeff: Vec<f32>,
    bias: &mut f64,
    beta: &[Option<&[f32]>],
    nu_out: &mut Vec<Vec<f32>>,
) -> Option<Vec<f32>> {
    let mut act_idx = 0usize;
    for layer in layers {
        let t0 = prof_enabled().then(std::time::Instant::now);
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias: lbias,
                out_features,
                in_features,
            } => {
                if coeff.len() != *out_features {
                    return None;
                }
                if let Some(b) = lbias {
                    for (a, &bv) in coeff.iter().zip(b.iter()) {
                        *bias += *a as f64 * bv as f64;
                    }
                }
                let mut next = vec![0.0f32; *in_features];
                for (o, &a) in coeff.iter().enumerate() {
                    if a == 0.0 {
                        continue;
                    }
                    let row = &weight[o * in_features..(o + 1) * in_features];
                    for (n, &w) in next.iter_mut().zip(row.iter()) {
                        *n += a * w;
                    }
                }
                coeff = next;
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
                let (oc, ic) = (*out_channels, *in_channels);
                let (kh, kw) = (*kernel_h, *kernel_w);
                let ohw = out_h * out_w;
                if coeff.len() != oc * ohw {
                    return None;
                }
                if let Some(b) = bias_expanded {
                    for (a, &bv) in coeff.iter().zip(b.iter()) {
                        *bias += *a as f64 * bv as f64;
                    }
                }
                let n = ic * kh * kw;
                // #task-11 replay cost cut: the transpose-conv contraction as a
                // single faer GEMM `cols[n × ohw] = W_colᵀ[n × oc] · C[oc × ohw]`
                // plus a cheap col2im scatter — the naive per-element scatter
                // over the 128×128×3×3 stacks dominated the ~0.5s/replay wall.
                // faer's blocked f32 accumulation reorders the sums vs the
                // scalar loop; fine here — the gradient is advisory and the
                // fail-closed lb check has wide tolerance. `mat_mul` degrades
                // to Par::Seq inside the domain-parallel rayon workers.
                let wt = faer::Mat::<f32>::from_fn(n, oc, |i, c| weight_col[c * n + i]);
                let cm = faer::Mat::<f32>::from_fn(oc, ohw, |c, s| coeff[c * ohw + s]);
                let cols = crate::faer_parallelism::mat_mul(&wt, &cm);
                // col2im: s outer / r inner keeps the col-major `cols` reads
                // contiguous (column s is contiguous in r).
                let mut next = vec![0.0f32; ic * in_h * in_w];
                for oh_i in 0..*out_h {
                    for ow_i in 0..*out_w {
                        let s = oh_i * out_w + ow_i;
                        for ci in 0..ic {
                            let base = ci * (in_h * in_w);
                            for kh_i in 0..kh {
                                let ih = (oh_i * stride_h + kh_i) as isize - *pad_h as isize;
                                if ih < 0 || ih >= *in_h as isize {
                                    continue;
                                }
                                let row = base + ih as usize * in_w;
                                for kw_i in 0..kw {
                                    let iw = (ow_i * stride_w + kw_i) as isize - *pad_w as isize;
                                    if iw < 0 || iw >= *in_w as isize {
                                        continue;
                                    }
                                    let r = ci * (kh * kw) + kh_i * kw + kw_i;
                                    next[row + iw as usize] += cols[(r, s)];
                                }
                            }
                        }
                    }
                }
                coeff = next;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                if coeff.len() != *num_neurons {
                    return None;
                }
                nu_out.push(coeff.clone());
                let bs = beta.get(act_idx).copied().flatten();
                for (i, a) in coeff.iter_mut().enumerate() {
                    // Lower-row branch select: a >= 0 -> lower relaxation
                    // (CROWN_ACTIVATION_RESIDENT_SHADER / relu/mod.rs:620-641).
                    let (sel, sel_int) = if *a >= 0.0 {
                        (lower_slope[i], lower_intercept[i])
                    } else {
                        (upper_slope[i], upper_intercept[i])
                    };
                    *bias += *a as f64 * sel_int as f64;
                    let mut v = *a * sel;
                    // Beta folds post-slope: lower rows SUBTRACT beta_signed.
                    if let Some(b) = bs {
                        if let Some(&bv) = b.get(i) {
                            v -= bv;
                        }
                    }
                    *a = v;
                }
                act_idx += 1;
            }
            // Dual-alpha / MaxPool are gated out of the wide lane (HOLE 8);
            // decline rather than guess.
            _ => return None,
        }
        if let Some(t0) = t0 {
            let slot = match layer {
                GpuCrownLayer::Conv2d { .. } => 0,
                GpuCrownLayer::Linear { .. } => 1,
                _ => 2,
            };
            prof_add(slot, t0.elapsed().as_secs_f64());
        }
    }
    Some(coeff)
}

/// Number of Activation layers in a branch.
fn n_act(layers: &[GpuCrownLayer]) -> usize {
    layers
        .iter()
        .filter(|l| matches!(l, GpuCrownLayer::Activation { .. }))
        .count()
}

/// Replay ONE spec row's lower backward over a domain's segments, mirroring the
/// resident fold walk exactly (segments in vec order = output -> input; per
/// segment F then P; Residual adds the skip coefficients; ResidualProj's P
/// branch carries coefficients only, bias counted once in F).
///
/// `beta_signed[r]` is the fold-order per-Activation signed beta (empty slice
/// or missing entries mean zero). Returns `None` on any dim/layer mismatch —
/// the caller skips the alpha step (fail-closed).
pub(super) fn replay_critical_row(
    segments: &[GpuResnetSegment],
    spec_row: &[f32],
    beta_signed: &[Vec<f32>],
) -> Option<CriticalRowReplay> {
    let mut nu: Vec<Vec<f32>> = Vec::new();
    let mut coeff = spec_row.to_vec();
    let mut bias = 0.0f64;
    let mut fold_idx = 0usize;
    let beta_for = |start: usize, count: usize| -> Vec<Option<&[f32]>> {
        (start..start + count)
            .map(|r| {
                beta_signed
                    .get(r)
                    .map(|v| v.as_slice())
                    .filter(|v| !v.is_empty())
            })
            .collect()
    };
    for seg in segments {
        match seg {
            GpuResnetSegment::Chain(branch) => {
                let nf = n_act(branch);
                let beta = beta_for(fold_idx, nf);
                coeff = backward_branch(branch, coeff, &mut bias, &beta, &mut nu)?;
                fold_idx += nf;
            }
            GpuResnetSegment::Residual(branch) => {
                let nf = n_act(branch);
                let beta = beta_for(fold_idx, nf);
                let skip = coeff.clone();
                let mut through = backward_branch(branch, coeff, &mut bias, &beta, &mut nu)?;
                if through.len() != skip.len() {
                    return None;
                }
                for (t, s) in through.iter_mut().zip(skip.iter()) {
                    *t += s;
                }
                coeff = through;
                fold_idx += nf;
            }
            GpuResnetSegment::ResidualProj(f_branch, p_branch) => {
                let nf = n_act(f_branch);
                let np = n_act(p_branch);
                let beta_f = beta_for(fold_idx, nf);
                let beta_p = beta_for(fold_idx + nf, np);
                let seed_p = coeff.clone();
                let through = backward_branch(f_branch, coeff, &mut bias, &beta_f, &mut nu)?;
                // P carries coefficients only; its bias contributions are real
                // (a projection conv can carry a bias) so keep accumulating
                // into the same scalar — matches the fold, which seeds P's
                // bias to 0 and ADDS both streams' biases at the join.
                let mut p_bias = 0.0f64;
                let p_coeff = backward_branch(p_branch, seed_p, &mut p_bias, &beta_p, &mut nu)?;
                bias += p_bias;
                if through.len() != p_coeff.len() {
                    return None;
                }
                coeff = through
                    .iter()
                    .zip(p_coeff.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                fold_idx += nf + np;
            }
        }
    }
    Some(CriticalRowReplay {
        nu,
        final_a: coeff,
        final_b: bias as f32,
    })
}

// ============================================================================
// BARRIER-1 SOUND f64 LINEAGE RECOVERY (dark gate `NY_F64_LINEAGE_RECOVER=1`).
//
// DIAGNOSIS (errprobe, prop1498 worst class-55 subdomains, n=60): the GPU f32
// *sound* fold reports mean 0.088 BELOW the ideal (f64) fold, but the ACTUAL
// f32-vs-f64 fold difference is only ~1e-5 (max 1.16e-5). 95% of the tax is the
// DEEP err-channel worst-case (Higham |A|@|W| no-cancellation) accumulation, not
// true f32 rounding ⇒ the certified-error bound is CONSERVATIVE. Metal has no
// f64, so the sound recovery is a CPU f64 host fold of the worst subdomains'
// binding rows: fold the critical row backward in f64, carrying the SAME sound
// certified-error accounting but with u = 2⁻⁵³ (so the recovered penalty is
// ~(2⁻⁵³/2⁻²⁴) ≈ 6e-9× smaller ≈ 1e-9, negligible). `max(gpu_lb, f64_sound_lb)`
// is sound (both are valid lower bounds). Recovers ~0.088 for the pinned rows.
// ============================================================================

/// f64 unit roundoff `u = 2⁻⁵³`.
const U_F64: f64 = f64::from_bits(0x3CA0_0000_0000_0000);

/// γ_k = k·u/(1−k·u) in f64 (the length-k reduction backward-error factor),
/// clamped to `2·k·u` past the half-way point — the f64 twin of `gamma_k_f32`.
fn gamma_k_f64(k: usize) -> f64 {
    let ku = (k as f64) * U_F64;
    if ku < 0.5 {
        ku / (1.0 - ku)
    } else {
        2.0 * ku
    }
}

/// A row's f64 coefficient frontier with a per-coefficient certified-error
/// bound. `err[j] ≥ |a_exact[j] − a[j]|` is maintained OUTWARD (over-estimated)
/// at every op so the final concretized bound stays a valid lower bound.
struct SoundRowF64 {
    a: Vec<f64>,
    err: Vec<f64>,
    bias: f64,
    /// Certified error on the per-contribution bias dots (rounding + propagated
    /// coefficient error of each layer's `A·bias` / `A·τ` dot).
    berr: f64,
    /// Σ|bias contribution| — bounds the cross-layer running-sum accumulation of
    /// `bias` (each `bias +=` rounds; over all adds ≤ γ_M·babs, penalized once at
    /// concretize with a safe over-count M).
    babs: f64,
}

/// Fold ONE spec row's lower backward over a domain's segments entirely in f64,
/// carrying the sound certified-error channel (u = 2⁻⁵³), and concretize to a
/// SOUND f32 lower bound over the input box. Mirrors [`replay_critical_row`]'s
/// walk and beta handling exactly. `None` on any dim/layer mismatch (caller then
/// keeps the GPU bound — never unsound). The result is a valid lower bound of the
/// spec row's minimum over the box, so `max(gpu_lb, this)` is sound.
pub(super) fn sound_f64_lower_bound(
    segments: &[GpuResnetSegment],
    spec_row: &[f32],
    beta_signed: &[Vec<f32>],
    in_lo: &[f32],
    in_hi: &[f32],
    // #mn-head-facet increment 1: an optional HEAD coupling-facet fold to embed
    // (`+Σβ·g_i` post, `+Σβ·a_i` pre, `−Σβ·b` bias) at its `target_act` head ReLU,
    // with certified build error injected into the outward err channels. `None`
    // (gate off / unregistered) ⇒ the fold arm is NEVER entered ⇒ byte-identical.
    head_fold: Option<&HeadF64Fold>,
) -> Option<f32> {
    let mut row = SoundRowF64 {
        a: spec_row.iter().map(|&x| f64::from(x)).collect(),
        err: vec![0.0f64; spec_row.len()],
        bias: 0.0,
        berr: 0.0,
        babs: 0.0,
    };
    let mut fold_idx = 0usize;
    let beta_for = |start: usize, count: usize| -> Vec<Option<&[f32]>> {
        (start..start + count)
            .map(|r| {
                beta_signed
                    .get(r)
                    .map(|v| v.as_slice())
                    .filter(|v| !v.is_empty())
            })
            .collect()
    };
    for seg in segments {
        match seg {
            GpuResnetSegment::Chain(branch) => {
                let nf = n_act_all(branch);
                let beta = beta_for(fold_idx, nf);
                sound_f64_branch(branch, &mut row, &beta, fold_idx, head_fold)?;
                fold_idx += nf;
            }
            GpuResnetSegment::Residual(branch) => {
                let nf = n_act_all(branch);
                let beta = beta_for(fold_idx, nf);
                let skip_a = row.a.clone();
                let skip_err = row.err.clone();
                sound_f64_branch(branch, &mut row, &beta, fold_idx, head_fold)?;
                if row.a.len() != skip_a.len() {
                    return None;
                }
                for j in 0..row.a.len() {
                    let sum = row.a[j] + skip_a[j];
                    // add rounding ≤ u·|sum|, plus both incoming errors.
                    row.err[j] += skip_err[j] + U_F64 * sum.abs();
                    row.a[j] = sum;
                }
                fold_idx += nf;
            }
            GpuResnetSegment::ResidualProj(f_branch, p_branch) => {
                let nf = n_act_all(f_branch);
                let np = n_act_all(p_branch);
                let beta_f = beta_for(fold_idx, nf);
                let beta_p = beta_for(fold_idx + nf, np);
                let seed_a = row.a.clone();
                let seed_err = row.err.clone();
                // F branch mutates `row` in place (keeps bias accumulator).
                sound_f64_branch(f_branch, &mut row, &beta_f, fold_idx, head_fold)?;
                // P branch on its own coeff/err, bias seeded to 0 then added.
                let mut prow = SoundRowF64 {
                    a: seed_a,
                    err: seed_err,
                    bias: 0.0,
                    berr: 0.0,
                    babs: 0.0,
                };
                sound_f64_branch(p_branch, &mut prow, &beta_p, fold_idx + nf, head_fold)?;
                row.bias += prow.bias;
                row.berr += prow.berr;
                row.babs += prow.babs;
                if row.a.len() != prow.a.len() {
                    return None;
                }
                for j in 0..row.a.len() {
                    let sum = row.a[j] + prow.a[j];
                    row.err[j] += prow.err[j] + U_F64 * sum.abs();
                    row.a[j] = sum;
                }
                fold_idx += nf + np;
            }
        }
    }
    // Concretize: lb = bias + Σ_j φ(a_j) − penalty. penalty widens OUTWARD by the
    // per-coeff certified error, the concretize dot's own γ_n rounding, and the
    // accumulated bias error. n = input_dim.
    let n = row.a.len();
    if n != in_lo.len() || n != in_hi.len() {
        return None;
    }
    let gamma_n = gamma_k_f64(n);
    let mut lb = row.bias;
    let mut penalty = row.berr;
    for j in 0..n {
        let a = row.a[j];
        let (xl, xu) = (f64::from(in_lo[j]), f64::from(in_hi[j]));
        lb += if a >= 0.0 { a * xl } else { a * xu };
        let xmax = xl.abs().max(xu.abs());
        penalty += (row.err[j] + gamma_n * a.abs()) * xmax;
    }
    // The concretize sum itself rounds (length-n f64 reduction): ≤ γ_n·Σ|a·x|.
    let mut l1_ax = 0.0f64;
    for j in 0..n {
        let xmax = f64::from(in_lo[j]).abs().max(f64::from(in_hi[j]).abs());
        l1_ax += row.a[j].abs() * xmax;
    }
    penalty += gamma_n * l1_ax + U_F64 * lb.abs();
    // Cross-layer running-sum accumulation of `bias`: over M sequential f64 adds
    // the rounding is ≤ γ_M·Σ|contribution|. M is safely over-counted at 2²⁴
    // (γ = 2⁻²⁹ ≈ 1.86e-9) — far above this net's per-row bias-add count, still
    // negligible vs the ~0.085 recovered.
    penalty += gamma_k_f64(1 << 24) * row.babs;
    let sound = lb - penalty;
    if !sound.is_finite() {
        return None;
    }
    // Round DOWN to f32 (outward for a lower bound): if the round-to-nearest cast
    // landed ABOVE the true f64 value, step one ULP toward −∞.
    let f = sound as f32;
    if !f.is_finite() {
        return None;
    }
    Some(if f64::from(f) > sound {
        if f > 0.0 {
            f32::from_bits(f.to_bits() - 1)
        } else if f < 0.0 {
            f32::from_bits(f.to_bits() + 1)
        } else {
            // f == ±0.0: next value below zero = −smallest subnormal.
            -f32::from_bits(1)
        }
    } else {
        f
    })
}

/// Count ALL Activation layers in a branch (twin of the private `n_act`).
fn n_act_all(layers: &[GpuCrownLayer]) -> usize {
    layers
        .iter()
        .filter(|l| matches!(l, GpuCrownLayer::Activation { .. }))
        .count()
}

/// f64 sound backward of ONE branch (output→input), mutating `row` in place.
/// Mirrors [`backward_branch`]'s walk; carries the certified-error channel.
///
/// `act_base` is the GLOBAL fold-order index of this branch's FIRST Activation
/// (the caller's `fold_idx` accumulator), so `act_base + act_idx` is the global
/// activation index. `head_fold` (when `Some`) embeds the #mn-head-facet HEAD
/// coupling-facet Lagrangian at the single activation whose global index equals
/// `head_fold.target_act` AND whose width equals `head_fold.head_width` — see the
/// Activation arm. `None` ⇒ this is byte-identical to the pre-facet walk.
fn sound_f64_branch(
    layers: &[GpuCrownLayer],
    row: &mut SoundRowF64,
    beta: &[Option<&[f32]>],
    act_base: usize,
    head_fold: Option<&HeadF64Fold>,
) -> Option<()> {
    let mut act_idx = 0usize;
    for layer in layers {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias: lbias,
                out_features,
                in_features,
            } => {
                if row.a.len() != *out_features {
                    return None;
                }
                let k = *out_features;
                let gk = gamma_k_f64(k);
                if let Some(b) = lbias {
                    if b.len() != *out_features {
                        return None;
                    }
                    let mut dot = 0.0f64;
                    let mut absdot = 0.0f64;
                    let mut errdot = 0.0f64;
                    for (i, &a) in row.a.iter().enumerate() {
                        let bv = f64::from(b[i]);
                        dot += a * bv;
                        absdot += a.abs() * bv.abs();
                        errdot += row.err[i] * bv.abs();
                    }
                    row.bias += dot;
                    row.babs += dot.abs();
                    row.berr += gk * absdot + errdot;
                }
                // na[j] = Σ_o a[o]·w[o,j]; err_new[j] = γ_k·(Σ|a|·|w|) + Σ err·|w|.
                let mut na = vec![0.0f64; *in_features];
                let mut ne = vec![0.0f64; *in_features]; // Σ err·|w|
                let mut sprod = vec![0.0f64; *in_features]; // Σ |a|·|w|
                for (o, &a) in row.a.iter().enumerate() {
                    let e = row.err[o];
                    let av = a.abs();
                    if a == 0.0 && e == 0.0 {
                        continue;
                    }
                    let wrow = &weight[o * in_features..(o + 1) * in_features];
                    for (j, &w) in wrow.iter().enumerate() {
                        let wf = f64::from(w);
                        let wa = wf.abs();
                        na[j] += a * wf;
                        ne[j] += e * wa;
                        sprod[j] += av * wa;
                    }
                }
                for j in 0..*in_features {
                    ne[j] += gk * sprod[j];
                }
                row.a = na;
                row.err = ne;
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
                let (oc, ic) = (*out_channels, *in_channels);
                let (kh, kw) = (*kernel_h, *kernel_w);
                let ohw = out_h * out_w;
                if row.a.len() != oc * ohw {
                    return None;
                }
                let klen = oc * kh * kw;
                let gk = gamma_k_f64(klen);
                if let Some(b) = bias_expanded {
                    if b.len() != oc * ohw {
                        return None;
                    }
                    let mut dot = 0.0f64;
                    let mut absdot = 0.0f64;
                    let mut errdot = 0.0f64;
                    for (i, &a) in row.a.iter().enumerate() {
                        let bv = f64::from(b[i]);
                        dot += a * bv;
                        absdot += a.abs() * bv.abs();
                        errdot += row.err[i] * bv.abs();
                    }
                    row.bias += dot;
                    row.babs += dot.abs();
                    row.berr += gk * absdot + errdot;
                }
                let in_dim = ic * in_h * in_w;
                let mut na = vec![0.0f64; in_dim];
                let mut ne = vec![0.0f64; in_dim];
                let mut sprod = vec![0.0f64; in_dim];
                for oh_i in 0..*out_h {
                    for ow_i in 0..*out_w {
                        for c in 0..oc {
                            let s = (c * out_h + oh_i) * out_w + ow_i;
                            let a = row.a[s];
                            let e = row.err[s];
                            let av = a.abs();
                            if a == 0.0 && e == 0.0 {
                                continue;
                            }
                            for kh_i in 0..kh {
                                let ih = (oh_i * stride_h + kh_i) as isize - *pad_h as isize;
                                if ih < 0 || ih >= *in_h as isize {
                                    continue;
                                }
                                for kw_i in 0..kw {
                                    let iw = (ow_i * stride_w + kw_i) as isize - *pad_w as isize;
                                    if iw < 0 || iw >= *in_w as isize {
                                        continue;
                                    }
                                    for cin in 0..ic {
                                        let wv = f64::from(
                                            weight_col[c * (ic * kh * kw)
                                                + cin * (kh * kw)
                                                + kh_i * kw
                                                + kw_i],
                                        );
                                        let idx = (cin * in_h + ih as usize) * in_w + iw as usize;
                                        na[idx] += a * wv;
                                        ne[idx] += e * wv.abs();
                                        sprod[idx] += av * wv.abs();
                                    }
                                }
                            }
                        }
                    }
                }
                for j in 0..in_dim {
                    ne[j] += gk * sprod[j];
                }
                row.a = na;
                row.err = ne;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                if row.a.len() != *num_neurons {
                    return None;
                }
                let bs = beta.get(act_idx).copied().flatten();
                // #mn-head-facet increment 1: this activation carries the HEAD
                // coupling-facet fold iff its GLOBAL fold index matches the fold's
                // `target_act` AND its width matches `head_width` (the belt against
                // a mislabeled target — folding onto the wrong ReLU would be an
                // invalid Lagrangian). `None`/mismatch ⇒ `hf` is `None` ⇒ every
                // guarded add below is skipped ⇒ byte-identical to the base walk.
                let hf: Option<&HeadF64Fold> = head_fold
                    .filter(|f| act_base + act_idx == f.target_act && *num_neurons == f.head_width);
                // Fold `bias_shift` (`−Σβ·b`) into the lower bias ONCE, outward:
                // the certified accumulation error `bias_err` plus this add's own
                // f64 rounding widen `berr` (subtracted at concretize). Guarded so
                // an all-zero fold performs NO arithmetic (oracle (a) byte-identity).
                if let Some(f) = hf {
                    if f.bias_shift != 0.0 || f.bias_err != 0.0 {
                        let nb = row.bias + f.bias_shift;
                        row.berr += f.bias_err + U_F64 * nb.abs();
                        row.babs += f.bias_shift.abs();
                        row.bias = nb;
                    }
                }
                for i in 0..*num_neurons {
                    let mut a = row.a[i];
                    // step 1 (BEFORE the sign-select): add the post-activation term
                    // `+Σβ·g_i` to the ReLU-OUTPUT coefficient, so it rides the same
                    // relaxation the margin's own y-coefficient does. The certified
                    // build error `ge` (deviation of the f64-summed coeff from the
                    // exact `Σβ·g_i`) plus this add's f64 rounding widen `err[i]`,
                    // which then propagates through the relaxation to the input.
                    if let Some(f) = hf {
                        if let Some(&(g, ge)) = f.post.get(&(i as u32)) {
                            let na = a + g;
                            row.err[i] += ge + U_F64 * na.abs();
                            a = na;
                        }
                    }
                    let (sel, sel_int) = if a >= 0.0 {
                        (f64::from(lower_slope[i]), f64::from(lower_intercept[i]))
                    } else {
                        (f64::from(upper_slope[i]), f64::from(upper_intercept[i]))
                    };
                    row.bias += a * sel_int;
                    row.babs += (a * sel_int).abs();
                    row.berr += U_F64 * (a * sel_int).abs() + row.err[i] * sel_int.abs();
                    let mut v = a * sel;
                    if let Some(b) = bs {
                        if let Some(&bv) = b.get(i) {
                            v -= f64::from(bv);
                        }
                    }
                    // err: slope multiply |sel|·err[i] + rounding of a·sel + subtract.
                    row.err[i] = sel.abs() * row.err[i] + U_F64 * (a * sel).abs() + U_F64 * v.abs();
                    // step 2 (AFTER the relaxation): add the pre-activation term
                    // `+Σβ·a_i` DIRECTLY to the ReLU-INPUT (`x_i`) coefficient,
                    // bypassing the ReLU exactly where a single-neuron β-split adds
                    // its `±β`. Its build error `ae` + this add's rounding widen the
                    // now-input-side `err[i]`.
                    if let Some(f) = hf {
                        if let Some(&(pa, ae)) = f.pre.get(&(i as u32)) {
                            let nv = v + pa;
                            row.err[i] += ae + U_F64 * nv.abs();
                            v = nv;
                        }
                    }
                    row.a[i] = v;
                }
                act_idx += 1;
            }
            _ => return None,
        }
    }
    Some(())
}

/// Forward one branch of the RELAXED network (branch layers are stored in
/// backward order, so iterate reversed). At the `k`-th Activation seen in
/// forward order (fold index `fold_start + (n_branch_acts - 1 - k)`), record
/// the incoming pre-activation into `pre_out[fold_idx]` and apply the affine
/// relaxation selected by `nu[fold_idx]`'s sign.
fn forward_branch(
    layers: &[GpuCrownLayer],
    mut x: Vec<f32>,
    fold_start: usize,
    nu: &[Vec<f32>],
    pre_out: &mut [Vec<f32>],
) -> Option<Vec<f32>> {
    let n_acts = n_act(layers);
    let mut seen = 0usize;
    for layer in layers.iter().rev() {
        let t0 = prof_enabled().then(std::time::Instant::now);
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
            } => {
                if x.len() != *in_features {
                    return None;
                }
                let mut y = vec![0.0f32; *out_features];
                for (o, yo) in y.iter_mut().enumerate() {
                    let row = &weight[o * in_features..(o + 1) * in_features];
                    let mut acc = 0.0f64;
                    for (&w, &xv) in row.iter().zip(x.iter()) {
                        acc += w as f64 * xv as f64;
                    }
                    if let Some(b) = bias {
                        acc += b[o] as f64;
                    }
                    *yo = acc as f32;
                }
                x = y;
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
                let (oc, ic) = (*out_channels, *in_channels);
                let (kh, kw) = (*kernel_h, *kernel_w);
                if x.len() != ic * in_h * in_w {
                    return None;
                }
                let n = ic * kh * kw;
                let ohw = out_h * out_w;
                // #task-11 replay cost cut: im2col gather + one faer GEMM
                // `Y[oc × ohw] = W_col[oc × n] · P[n × ohw]` replaces the naive
                // per-output gather (same reasoning as the backward arm: the
                // gradient is advisory, f32 GEMM reordering is fine, Par::Seq
                // inside rayon workers). P is filled s outer / r inner so the
                // col-major column writes stay contiguous; padding stays 0.
                let mut p_mat = faer::Mat::<f32>::zeros(n, ohw);
                for oh_i in 0..*out_h {
                    for ow_i in 0..*out_w {
                        let s = oh_i * out_w + ow_i;
                        for ci in 0..ic {
                            let base = ci * (in_h * in_w);
                            for kh_i in 0..kh {
                                let ih = (oh_i * stride_h + kh_i) as isize - *pad_h as isize;
                                if ih < 0 || ih >= *in_h as isize {
                                    continue;
                                }
                                let row = base + ih as usize * in_w;
                                for kw_i in 0..kw {
                                    let iw = (ow_i * stride_w + kw_i) as isize - *pad_w as isize;
                                    if iw < 0 || iw >= *in_w as isize {
                                        continue;
                                    }
                                    let r = ci * (kh * kw) + kh_i * kw + kw_i;
                                    p_mat[(r, s)] = x[row + iw as usize];
                                }
                            }
                        }
                    }
                }
                let w = faer::Mat::<f32>::from_fn(oc, n, |c, i| weight_col[c * n + i]);
                let ym = crate::faer_parallelism::mat_mul(&w, &p_mat);
                let mut y = vec![0.0f32; oc * ohw];
                for s in 0..ohw {
                    for c in 0..oc {
                        let p = c * ohw + s;
                        let mut acc = ym[(c, s)];
                        if let Some(b) = bias_expanded {
                            acc += b[p];
                        }
                        y[p] = acc;
                    }
                }
                x = y;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                if x.len() != *num_neurons {
                    return None;
                }
                let fold_idx = fold_start + (n_acts - 1 - seen);
                let nu_r = nu.get(fold_idx)?;
                if nu_r.len() != *num_neurons {
                    return None;
                }
                pre_out.get_mut(fold_idx)?.clone_from(&x);
                for (i, v) in x.iter_mut().enumerate() {
                    let (s, t) = if nu_r[i] >= 0.0 {
                        (lower_slope[i], lower_intercept[i])
                    } else {
                        (upper_slope[i], upper_intercept[i])
                    };
                    *v = s * *v + t;
                }
                seen += 1;
            }
            _ => return None,
        }
        if let Some(t0) = t0 {
            let slot = match layer {
                GpuCrownLayer::Conv2d { .. } => 3,
                GpuCrownLayer::Linear { .. } => 4,
                _ => 5,
            };
            prof_add(slot, t0.elapsed().as_secs_f64());
        }
    }
    Some(x)
}

/// Relaxed-linear forward evaluation at `x` through the whole segment stack.
/// Returns per-Activation (fold order) pre-activation values, plus the final
/// network output (used by the adjointness self-test).
pub(super) fn relaxed_forward(
    segments: &[GpuResnetSegment],
    nu: &[Vec<f32>],
    x: &[f32],
) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
    let n_relu = nu.len();
    let mut pre = vec![Vec::new(); n_relu];
    // Fold-order start index of each segment's activations.
    let mut starts: Vec<usize> = Vec::with_capacity(segments.len());
    let mut idx = 0usize;
    for seg in segments {
        starts.push(idx);
        idx += match seg {
            GpuResnetSegment::Chain(b) | GpuResnetSegment::Residual(b) => n_act(b),
            GpuResnetSegment::ResidualProj(f, p) => n_act(f) + n_act(p),
        };
    }
    if idx != n_relu {
        return None;
    }
    // Forward = reverse segment order (fold order is output -> input).
    let mut h = x.to_vec();
    for (seg, &start) in segments.iter().zip(starts.iter()).rev() {
        h = match seg {
            GpuResnetSegment::Chain(branch) => forward_branch(branch, h, start, nu, &mut pre)?,
            GpuResnetSegment::Residual(branch) => {
                let through = forward_branch(branch, h.clone(), start, nu, &mut pre)?;
                if through.len() != h.len() {
                    return None;
                }
                through.iter().zip(h.iter()).map(|(&a, &b)| a + b).collect()
            }
            GpuResnetSegment::ResidualProj(f_branch, p_branch) => {
                let nf = n_act(f_branch);
                let through = forward_branch(f_branch, h.clone(), start, nu, &mut pre)?;
                let proj = forward_branch(p_branch, h, start + nf, nu, &mut pre)?;
                if through.len() != proj.len() {
                    return None;
                }
                through
                    .iter()
                    .zip(proj.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
        };
    }
    Some((pre, h))
}

/// The TRUE per-neuron alpha gradients of ONE spec row's lower bound, per
/// Activation in fold order: `g_r[i] = max(nu_r[i], 0) * hhat_r[i](x*)`.
///
/// `gpu_lb` is the wide fold's lower bound for the SAME row and domain; the
/// replayed row must reproduce it (up to the fold's certified error widening
/// and f32 reorder noise) or the caller gets `None` (fail-closed: a walk or
/// convention mismatch must not steer alpha).
pub(crate) fn true_alpha_grads_for_row(
    segments: &[GpuResnetSegment],
    spec_row: &[f32],
    beta_signed: &[Vec<f32>],
    in_lo: &[f32],
    in_hi: &[f32],
    n_relu_expected: usize,
    gpu_lb: f32,
    probe: bool,
) -> Option<Vec<Vec<f32>>> {
    let prof = prof_enabled();
    if prof {
        PROF_ACC.with(|a| a.set([0.0; 6]));
    }
    let t_bwd = std::time::Instant::now();
    let replay = replay_critical_row(segments, spec_row, beta_signed)?;
    let bwd_s = t_bwd.elapsed().as_secs_f64();
    if replay.nu.len() != n_relu_expected || replay.final_a.len() != in_lo.len() {
        if probe {
            eprintln!(
                "[wide-alpha-true] replay shape mismatch: n_relu={} (want {}) in_dim={} (want {})",
                replay.nu.len(),
                n_relu_expected,
                replay.final_a.len(),
                in_lo.len()
            );
        }
        return None;
    }
    let replay_lb = replay.lower_bound(in_lo, in_hi);
    // The GPU fold subtracts certified error, so gpu_lb <= replay_lb + noise;
    // a large gap either way means the replay walked the net differently.
    let scale = 1.0 + replay_lb.abs().max(gpu_lb.abs());
    let diff = replay_lb - gpu_lb;
    if !diff.is_finite() || diff < -0.05 * scale || diff > 0.5 * scale {
        if probe {
            eprintln!(
                "[wide-alpha-true] replay lb {replay_lb:.5} vs gpu {gpu_lb:.5} out of tolerance — skipping alpha step"
            );
        }
        return None;
    }
    let x_star = replay.argmin_corner(in_lo, in_hi);
    let t_fwd = std::time::Instant::now();
    let (pre, _out) = relaxed_forward(segments, &replay.nu, &x_star)?;
    if prof {
        let fwd_s = t_fwd.elapsed().as_secs_f64();
        let a = PROF_ACC.with(|acc| acc.get());
        eprintln!(
            "[wide-alpha-true-prof] bwd={:.1}ms (conv={:.1} lin={:.1} act={:.1}) fwd={:.1}ms (conv={:.1} lin={:.1} act={:.1})",
            bwd_s * 1e3,
            a[0] * 1e3,
            a[1] * 1e3,
            a[2] * 1e3,
            fwd_s * 1e3,
            a[3] * 1e3,
            a[4] * 1e3,
            a[5] * 1e3,
        );
    }
    if probe {
        // BLOCKER-1 error-decomposition probe (dark, NY_WIDE_ALPHA_ERRPROBE=1):
        // classify `diff = replay_lb − gpu_lb` as CERTIFIED-ERROR widening vs an
        // f32-fold BUG. `joint_lower_bound_debug` is the error-free, no-beta host
        // fold of the SAME segments/seed; with beta≈0 it must reproduce `replay_lb`
        // to f32-reorder noise (~1e-4). If it does, the two error-free host folds
        // agree and the ~0.1 gap to `gpu_lb` is entirely the fold's SOUND certified
        // error channel (one-sided, gpu lower) — not a walk/convention mismatch.
        // Cap the decomposition spam (this fires per replay×domain×iter): a few
        // dozen lines fully characterize the one-sided gap; unbounded stderr would
        // itself slow BaB and skew the depth reached.
        static ERRPROBE_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if std::env::var("NY_WIDE_ALPHA_ERRPROBE").ok().as_deref() == Some("1")
            && ERRPROBE_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 60
        {
            let seed_b = [0.0f32];
            let joint_lb = ny_core::joint_alpha_grad::joint_lower_bound_debug(
                segments,
                spec_row,
                &seed_b,
                1,
                spec_row.len(),
                in_lo,
                in_hi,
            )
            .and_then(|v| v.first().copied());
            // IDEAL-arithmetic reference: the SAME fold in f64. `f64_fold − jlb` is
            // the ACTUAL f32 rounding drift of the whole backward; if it is ≪ the
            // sound-fold deficit `jlb − gpu`, that deficit is CONSERVATIVE certified-
            // error accounting (recoverable in f32), not true f32 rounding.
            let f64_lb = ny_core::joint_alpha_grad::joint_lower_bound_debug_f64(
                segments,
                spec_row,
                &seed_b,
                1,
                spec_row.len(),
                in_lo,
                in_hi,
            )
            .and_then(|v| v.first().copied());
            // Decompose the sound deficit: the concretize's own γ_n dot-product
            // penalty `Σ_j γ_n·|a⁰[j]|·max(|x_l|,|x_u|)` (one length-n reduction)
            // vs the DEEP accumulated err channel (the remainder). n = input_dim.
            let input_dim = replay.final_a.len();
            let u64c: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24
            let nu = (input_dim as f64) * u64c;
            let gamma_n = if nu < 0.5 { nu / (1.0 - nu) } else { 2.0 * nu };
            let gamma_n_term: f64 = replay
                .final_a
                .iter()
                .enumerate()
                .map(|(j, &a)| {
                    let xmax = (in_lo[j].abs()).max(in_hi[j].abs()) as f64;
                    gamma_n * (a.abs() as f64) * xmax
                })
                .sum();
            let l1_ax: f64 = replay
                .final_a
                .iter()
                .enumerate()
                .map(|(j, &a)| (a.abs() as f64) * (in_lo[j].abs().max(in_hi[j].abs()) as f64))
                .sum();
            // SOUND f64 recovery (beta-aware) — the actual recovered bound. Must be a
            // valid lower bound (≥ ~gpu+0.088) and ≈ replay_lb (both f64-ish, beta-aware)
            // minus a ~1e-9 penalty.
            let sound_f64 =
                sound_f64_lower_bound(segments, spec_row, beta_signed, in_lo, in_hi, None);
            let beta_nz = beta_signed.iter().flatten().any(|&b| b != 0.0);
            match joint_lb {
                Some(jlb) => {
                    let sf = sound_f64
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "NA".into());
                    let recov = sound_f64
                        .map(|v| format!("{:.6}", v - gpu_lb))
                        .unwrap_or_else(|| "NA".into());
                    eprintln!(
                        "[wide-alpha-recover] sound_f64={sf} recovered(sound_f64-gpu)={recov} \
                         vs_replay={:.2e}",
                        sound_f64.map(|v| v - replay_lb).unwrap_or(f32::NAN)
                    );
                    let f64s = f64_lb
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "NA".into());
                    let actual_f32 = f64_lb
                        .map(|v| format!("{:.2e}", v - jlb as f64))
                        .unwrap_or_else(|| "NA".into());
                    eprintln!(
                        "[wide-alpha-errprobe] replay={replay_lb:.6} joint_nobeta={jlb:.6} \
                         f64_fold={f64s} gpu={gpu_lb:.6} replay-joint={:.2e} \
                         actual_f32(f64-jlb)={actual_f32} joint-gpu={:.6} \
                         gamma_n_term={gamma_n_term:.6} err_channel={:.6} L1|a·x|={l1_ax:.3} \
                         beta_nz={beta_nz}",
                        replay_lb - jlb,
                        jlb - gpu_lb,
                        (jlb - gpu_lb) as f64 - gamma_n_term,
                    );
                }
                None => eprintln!("[wide-alpha-errprobe] joint_lower_bound_debug declined"),
            }
        }
        eprintln!("[wide-alpha-true] replay ok: lb={replay_lb:.5} gpu={gpu_lb:.5} diff={diff:.2e}");
    }
    Some(
        replay
            .nu
            .iter()
            .zip(pre.iter())
            .map(|(nu_r, z_r)| {
                nu_r.iter()
                    .zip(z_r.iter())
                    .map(|(&nu, &z)| if nu > 0.0 { nu * z } else { 0.0 })
                    .collect()
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn lin(w: Vec<f32>, b: Option<Vec<f32>>, out_f: usize, in_f: usize) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from(w.into_boxed_slice()),
            bias: b.map(|v| Arc::from(v.into_boxed_slice())),
            out_features: out_f,
            in_features: in_f,
        }
    }

    fn act(ls: Vec<f32>, us: Vec<f32>, li: Vec<f32>, ui: Vec<f32>) -> GpuCrownLayer {
        let n = ls.len();
        GpuCrownLayer::Activation {
            lower_slope: ls,
            upper_slope: us,
            lower_intercept: li,
            upper_intercept: ui,
            num_neurons: n,
        }
    }

    fn conv(
        weight_col: Vec<f32>,
        bias_expanded: Option<Vec<f32>>,
        oc: usize,
        ic: usize,
        k: usize,
        stride: usize,
        pad: usize,
        out_hw: usize,
        in_hw: usize,
    ) -> GpuCrownLayer {
        GpuCrownLayer::Conv2d {
            weight_col: Arc::from(weight_col.into_boxed_slice()),
            bias_expanded: bias_expanded.map(|v| Arc::from(v.into_boxed_slice())),
            out_channels: oc,
            in_channels: ic,
            kernel_h: k,
            kernel_w: k,
            stride_h: stride,
            stride_w: stride,
            pad_h: pad,
            pad_w: pad,
            out_h: out_hw,
            out_w: out_hw,
            in_h: in_hw,
            in_w: in_hw,
        }
    }

    fn rngf(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A conv+residual+linear segment stack with interior slopes (as after an
    /// alpha write-back): fold order = [Chain(lin2, act2), Residual(act1a?, ..)] etc.
    /// Built INPUT-side last (fold order is output -> input; branch layers are
    /// backward-ordered within each segment).
    fn build_stack(state: &mut u64) -> (Vec<GpuResnetSegment>, usize) {
        let ic = 2usize;
        let hw = 4usize; // 2x4x4 = 32 input dims
        let in_dim = ic * hw * hw;
        let oc = 3usize;
        let ohw = 2usize; // stride 2, k3, pad 1: 4 -> 2
        let conv_dim = oc * ohw * ohw;

        // Innermost (input-side) segment: Chain(act1, lin_mid, act0, conv0) in
        // backward order, i.e. forward x -> conv0 -> act0 -> lin_mid -> act1.
        let wc: Vec<f32> = (0..oc * ic * 9).map(|_| rngf(state) * 0.5).collect();
        let cb: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.2).collect();
        let conv0 = conv(wc, Some(cb), oc, ic, 3, 2, 1, ohw, hw);
        let slopes0: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.5 + 0.5).collect();
        let us0: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.4 + 0.55).collect();
        let ui0: Vec<f32> = (0..conv_dim).map(|_| rngf(state).abs() * 0.3).collect();
        let act0 = act(slopes0, us0, vec![0.0; conv_dim], ui0);
        let wm: Vec<f32> = (0..conv_dim * conv_dim)
            .map(|_| rngf(state) * 0.4)
            .collect();
        let bm: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.1).collect();
        let lin_mid = lin(wm, Some(bm), conv_dim, conv_dim);
        let slopes1: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.5 + 0.5).collect();
        let us1: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.4 + 0.55).collect();
        let ui1: Vec<f32> = (0..conv_dim).map(|_| rngf(state).abs() * 0.3).collect();
        let act1 = act(slopes1, us1, vec![0.0; conv_dim], ui1);
        let seg_in = GpuResnetSegment::Chain(vec![act1, lin_mid, act0, conv0]);

        // Middle segment: Residual over a Linear+Activation branch (dims equal).
        let wr: Vec<f32> = (0..conv_dim * conv_dim)
            .map(|_| rngf(state) * 0.4)
            .collect();
        let br: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.1).collect();
        let lin_r = lin(wr, Some(br), conv_dim, conv_dim);
        let slopes_r: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.5 + 0.5).collect();
        let us_r: Vec<f32> = (0..conv_dim).map(|_| rngf(state) * 0.4 + 0.55).collect();
        let ui_r: Vec<f32> = (0..conv_dim).map(|_| rngf(state).abs() * 0.3).collect();
        let act_r = act(slopes_r, us_r, vec![0.0; conv_dim], ui_r);
        // Forward: branch = act_r(lin_r(x)); backward order: [act_r, lin_r].
        let seg_mid = GpuResnetSegment::Residual(vec![act_r, lin_r]);

        // Output segment: Chain(lin_out) 1-row spec-shaped output (od=2).
        let wo: Vec<f32> = (0..2 * conv_dim).map(|_| rngf(state) * 0.6).collect();
        let bo: Vec<f32> = (0..2).map(|_| rngf(state) * 0.1).collect();
        let seg_out = GpuResnetSegment::Chain(vec![lin(wo, Some(bo), 2, conv_dim)]);

        (vec![seg_out, seg_mid, seg_in], in_dim)
    }

    /// ADJOINTNESS: with beta = 0, the replayed row is exactly the affine
    /// function `spec . F_relaxed(x)` where `F_relaxed` applies the affine
    /// relaxation branch selected by the SAME nu signs the backward chose —
    /// the relaxed forward must reproduce `final_a . x + final_b` at random
    /// points. This ties the backward scatter and forward gather (independent
    /// code paths incl. conv indexing and the Residual walks) to each other.
    #[test]
    fn replay_backward_and_relaxed_forward_are_adjoint() {
        let mut state = 0x5EED_1234u64;
        for trial in 0..4 {
            let (segments, in_dim) = build_stack(&mut state);
            let spec: Vec<f32> = (0..2).map(|_| rngf(&mut state)).collect();
            let replay = replay_critical_row(&segments, &spec, &[]).expect("replay must succeed");
            assert_eq!(replay.nu.len(), 3, "three Activations in the stack");
            assert_eq!(replay.final_a.len(), in_dim);

            for _ in 0..4 {
                let x: Vec<f32> = (0..in_dim).map(|_| rngf(&mut state)).collect();
                let (_pre, out) =
                    relaxed_forward(&segments, &replay.nu, &x).expect("forward must succeed");
                let lhs: f64 = spec
                    .iter()
                    .zip(out.iter())
                    .map(|(&s, &o)| s as f64 * o as f64)
                    .sum();
                let rhs: f64 = replay.final_b as f64
                    + replay
                        .final_a
                        .iter()
                        .zip(x.iter())
                        .map(|(&a, &xv)| a as f64 * xv as f64)
                        .sum::<f64>();
                assert!(
                    (lhs - rhs).abs() <= 1e-3 * (1.0 + rhs.abs()),
                    "trial {trial}: spec.F(x) = {lhs} != final_a.x + b = {rhs}"
                );
            }
        }
    }

    /// BARRIER-1 SOUNDNESS: `sound_f64_lower_bound` must be a valid LOWER bound of
    /// the relaxed affine function `spec·F_relaxed(x)` at EVERY point of the box
    /// (the adjoint test proves that equals `final_a·x + b`), including through
    /// beta != 0. It must never exceed the affine minimum over the box (=
    /// `replay.lower_bound`); and it must TRACK it (recovery), differing only by
    /// the tiny f64 certified penalty + f32-vs-f64 fold noise (≪ 1e-2 here).
    #[test]
    fn sound_f64_lower_bound_is_a_valid_lower_bound() {
        let mut state = 0x1357_9BDFu64;
        for trial in 0..4 {
            let (segments, in_dim) = build_stack(&mut state);
            let spec: Vec<f32> = (0..2).map(|_| rngf(&mut state)).collect();
            let in_lo: Vec<f32> = (0..in_dim).map(|_| -0.5 + 0.1 * rngf(&mut state)).collect();
            let in_hi: Vec<f32> = in_lo.iter().map(|&l| l + 0.9).collect();
            // Exercise both beta=0 and beta!=0 (fold index 1 = middle Activation).
            let beta = if trial % 2 == 0 {
                vec![Vec::new(), Vec::new(), Vec::new()]
            } else {
                vec![
                    Vec::new(),
                    (0..12)
                        .map(|_| rngf(&mut state) * 0.05)
                        .collect::<Vec<f32>>(),
                    Vec::new(),
                ]
            };
            let sound = sound_f64_lower_bound(&segments, &spec, &beta, &in_lo, &in_hi, None)
                .expect("sound f64 fold must succeed on the supported stack");
            let replay = replay_critical_row(&segments, &spec, &beta).expect("replay");
            let affine_min = replay.lower_bound(&in_lo, &in_hi);
            // (1) Never above the affine minimum over the box (SOUND).
            assert!(
                sound <= affine_min + 1e-4,
                "trial {trial}: UNSOUND sound_f64 {sound} > affine min {affine_min}"
            );
            // (2) Never above the affine value at ANY point in the box (SOUND,
            // stronger — catches an argmin-corner mismatch). Random interior pts.
            for _ in 0..8 {
                let x: Vec<f32> = (0..in_dim)
                    .map(|j| {
                        let t = f32::midpoint(rngf(&mut state), 1.0);
                        in_lo[j] + t * (in_hi[j] - in_lo[j])
                    })
                    .collect();
                let fx: f64 = replay.final_b as f64
                    + replay
                        .final_a
                        .iter()
                        .zip(x.iter())
                        .map(|(&a, &xv)| a as f64 * xv as f64)
                        .sum::<f64>();
                assert!(
                    f64::from(sound) <= fx + 1e-4,
                    "trial {trial}: UNSOUND sound_f64 {sound} > spec·F_relaxed(x) {fx}"
                );
            }
            // (3) Recovery: tracks the ideal to a tiny margin (penalty ~1e-9 on this
            // small stack + f32 fold noise) — not pessimistically far below.
            assert!(
                f64::from(affine_min - sound) < 1e-2,
                "trial {trial}: sound_f64 {sound} too far below affine min {affine_min}"
            );
        }
    }

    /// FINITE-DIFFERENCE self-check on the full conv/residual stack: the
    /// gradient `max(nu,0) * hhat(x*)` must match the central difference of
    /// the replayed row's concretized lower bound when a lower_slope is
    /// perturbed — including through beta != 0 (beta shifts nu rows and x*
    /// but not the derivative rule).
    #[test]
    fn true_grads_match_finite_difference_on_segments() {
        let mut state = 0xABCD_EF01u64;
        let (segments, in_dim) = build_stack(&mut state);
        let spec: Vec<f32> = vec![0.8, -0.6];
        let in_lo: Vec<f32> = (0..in_dim).map(|_| -0.4 + 0.1 * rngf(&mut state)).collect();
        let in_hi: Vec<f32> = in_lo.iter().map(|&l| l + 0.7).collect();
        // Nonzero beta on the middle Activation (fold index 1).
        let beta = vec![
            Vec::new(),
            (0..12)
                .map(|_| rngf(&mut state) * 0.05)
                .collect::<Vec<f32>>(),
            Vec::new(),
        ];

        let lb = |segs: &[GpuResnetSegment]| -> f32 {
            replay_critical_row(segs, &spec, &beta)
                .expect("replay")
                .lower_bound(&in_lo, &in_hi)
        };
        let replay = replay_critical_row(&segments, &spec, &beta).expect("replay");
        let x_star = replay.argmin_corner(&in_lo, &in_hi);
        let (pre, _) = relaxed_forward(&segments, &replay.nu, &x_star).expect("forward");

        // Perturb each Activation's lower_slope one neuron at a time.
        let mut checked = 0usize;
        let mut skipped_kink = 0usize;
        let h = 2e-3f32;
        let mut act_fold = 0usize;
        for seg_i in 0..segments.len() {
            let branches: Vec<usize> = match &segments[seg_i] {
                GpuResnetSegment::Chain(b) | GpuResnetSegment::Residual(b) => vec![b.len()],
                GpuResnetSegment::ResidualProj(f, p) => vec![f.len(), p.len()],
            };
            let mut branch_starts = vec![act_fold];
            if branches.len() == 2 {
                let nf = match &segments[seg_i] {
                    GpuResnetSegment::ResidualProj(f, _) => n_act(f),
                    _ => 0,
                };
                branch_starts.push(act_fold + nf);
            }
            for (b_i, _) in branches.iter().enumerate() {
                let layer_count = branches[b_i];
                for l_i in 0..layer_count {
                    let is_act = {
                        let branch = match &segments[seg_i] {
                            GpuResnetSegment::Chain(b) | GpuResnetSegment::Residual(b) => {
                                b.as_slice()
                            }
                            GpuResnetSegment::ResidualProj(f, p) => {
                                if b_i == 0 {
                                    f.as_slice()
                                } else {
                                    p.as_slice()
                                }
                            }
                        };
                        matches!(branch[l_i], GpuCrownLayer::Activation { .. })
                    };
                    if !is_act {
                        continue;
                    }
                    // Fold index of this Activation = branch_start + (number of
                    // activations before it in slice order).
                    let branch = match &segments[seg_i] {
                        GpuResnetSegment::Chain(b) | GpuResnetSegment::Residual(b) => b.as_slice(),
                        GpuResnetSegment::ResidualProj(f, p) => {
                            if b_i == 0 {
                                f.as_slice()
                            } else {
                                p.as_slice()
                            }
                        }
                    };
                    let acts_before = n_act(&branch[..l_i]);
                    let r = branch_starts[b_i] + acts_before;
                    let nn = match &branch[l_i] {
                        GpuCrownLayer::Activation { num_neurons, .. } => *num_neurons,
                        _ => unreachable!(),
                    };
                    for i in (0..nn).step_by(3) {
                        // Analytic: max(nu,0) * hhat.
                        let expected = if replay.nu[r][i] > 0.0 {
                            replay.nu[r][i] * pre[r][i]
                        } else {
                            0.0
                        };
                        // FD on cloned segments.
                        let perturb = |delta: f32| -> f32 {
                            let mut segs = segments.clone();
                            let target = match &mut segs[seg_i] {
                                GpuResnetSegment::Chain(b) | GpuResnetSegment::Residual(b) => {
                                    &mut b[l_i]
                                }
                                GpuResnetSegment::ResidualProj(f, p) => {
                                    if b_i == 0 {
                                        &mut f[l_i]
                                    } else {
                                        &mut p[l_i]
                                    }
                                }
                            };
                            if let GpuCrownLayer::Activation { lower_slope, .. } = target {
                                lower_slope[i] += delta;
                            }
                            lb(&segs)
                        };
                        let fd = (perturb(h) - perturb(-h)) / (2.0 * h);
                        // Near-kink neurons (nu ~ 0 or a downstream selection
                        // flip within +-h) legitimately disagree; skip only
                        // when the disagreement co-occurs with a tiny nu.
                        let tol = 2e-3 + 0.03 * fd.abs();
                        if (fd - expected).abs() > tol && replay.nu[r][i].abs() < 5e-2 {
                            skipped_kink += 1;
                            continue;
                        }
                        assert!(
                            (fd - expected).abs() <= tol,
                            "relu {r} neuron {i}: fd {fd} != analytic {expected} (nu={}, hhat={})",
                            replay.nu[r][i],
                            pre[r][i]
                        );
                        checked += 1;
                    }
                }
            }
            act_fold += match &segments[seg_i] {
                GpuResnetSegment::Chain(b) | GpuResnetSegment::Residual(b) => n_act(b),
                GpuResnetSegment::ResidualProj(f, p) => n_act(f) + n_act(p),
            };
        }
        assert!(
            checked >= 10,
            "FD self-check must exercise a meaningful neuron count (checked={checked}, skipped={skipped_kink})"
        );
    }

    /// Cadence knob parsing: default/garbage/0 ⇒ 1 (replay every iteration);
    /// a positive integer passes through.
    #[test]
    fn true_every_parser_is_fail_safe() {
        assert_eq!(parse_true_every(None), 1);
        assert_eq!(parse_true_every(Some("")), 1);
        assert_eq!(parse_true_every(Some("nope")), 1);
        assert_eq!(parse_true_every(Some("0")), 1);
        assert_eq!(parse_true_every(Some("1")), 1);
        assert_eq!(parse_true_every(Some("2")), 2);
        assert_eq!(parse_true_every(Some("16")), 16);
    }

    /// The fail-closed lb validation: a gpu_lb far from the replayed lb must
    /// yield None (no alpha step), a close one must yield gradients.
    #[test]
    fn lb_validation_gates_the_gradients() {
        let mut state = 0x1111_2222u64;
        let (segments, in_dim) = build_stack(&mut state);
        let spec = vec![1.0f32, -0.5];
        let in_lo = vec![-0.3f32; in_dim];
        let in_hi = vec![0.5f32; in_dim];
        let replay = replay_critical_row(&segments, &spec, &[]).expect("replay");
        let lb = replay.lower_bound(&in_lo, &in_hi);

        let ok = true_alpha_grads_for_row(&segments, &spec, &[], &in_lo, &in_hi, 3, lb, false);
        assert!(ok.is_some(), "matching gpu_lb must pass validation");
        let bad = true_alpha_grads_for_row(
            &segments,
            &spec,
            &[],
            &in_lo,
            &in_hi,
            3,
            lb - 10.0 * (1.0 + lb.abs()),
            false,
        );
        assert!(bad.is_none(), "wildly divergent gpu_lb must fail closed");
    }

    // =======================================================================
    // #mn-head-facet increment 1 SOUNDNESS ORACLES (THE MOAT).
    //
    // Fixture: a small REAL-shaped dense head `Gemm→ReLU→Gemm` whose two hidden
    // pre-activations are the "diamond" `x1 = u1+u2`, `x2 = u1−u2` over the input
    // box `u ∈ [−1,1]²` (both crossing). margin = spec·out = −(y1+y2). The HEAD
    // coupling facet is the CLOSED-FORM one emitted by `coupling_facets` on this
    // group (`−0.5·x1 −0.5·x2 + y1 + y2 ≤ 1`), validated independently by the
    // multineuron enclosure/adversarial/routing tests (oracle (d)).
    // =======================================================================

    /// Segments (backward order) + spec row + input box for the diamond head.
    fn diamond_head_segments() -> (Vec<GpuResnetSegment>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // W2 = [−1,−1] (1×2), b2 = [0]  ⇒ out = −(y1+y2).
        let l2 = lin(vec![-1.0, -1.0], Some(vec![0.0]), 1, 2);
        // ReLU relaxation of pre ∈ [−2,2]²: lower α=0, upper slope 0.5 / intercept 1.
        let a = act(
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        );
        // W1 = [[1,1],[1,−1]] (2×2), b1 = [0,0] ⇒ pre = (u1+u2, u1−u2).
        let l1 = lin(vec![1.0, 1.0, 1.0, -1.0], Some(vec![0.0, 0.0]), 2, 2);
        let seg = vec![GpuResnetSegment::Chain(vec![l2, a, l1])];
        (seg, vec![1.0], vec![-1.0, -1.0], vec![1.0, 1.0])
    }

    /// The TRUE (unrelaxed) network margin `−(relu(u1+u2) + relu(u1−u2))`.
    fn true_diamond_margin(u: &[f32]) -> f32 {
        let x1 = u[0] + u[1];
        let x2 = u[0] - u[1];
        -(x1.max(0.0) + x2.max(0.0))
    }

    /// The REAL diamond coupling facet, reduced to a [`HeadF64Fold`] at `beta`
    /// (post `+β·g_i`, pre `+β·a_i`, bias `−β·b`; per-term certified add error),
    /// mirroring `pool_to_head_f64_fold`. `target_act`/`head_width` are explicit so
    /// the byte-identity oracle can drive mismatches.
    fn diamond_fold(beta: f64, target_act: usize, head_width: usize) -> HeadF64Fold {
        let p = crate::multineuron::Octahedron2::from_affine(
            &[1.0, 1.0],
            &[1.0, -1.0],
            0.0,
            0.0,
            &[-1.0, -1.0],
            &[1.0, 1.0],
        );
        let facet = crate::multineuron::coupling_facets(&p)
            .into_iter()
            .find(|f| f.a[2] > 0.1 && f.a[3] > 0.1 && (f.a[2] - f.a[3]).abs() < 1e-3)
            .expect("y1+y2 coupling facet must exist for the diamond group");
        let mut post = std::collections::HashMap::new();
        let mut pre = std::collections::HashMap::new();
        for i in 0..2usize {
            let g = beta * f64::from(facet.a[2 + i]); // post (ReLU-OUTPUT) g_i
            let a = beta * f64::from(facet.a[i]); //     pre  (ReLU-INPUT)  a_i
            if g != 0.0 {
                post.insert(i as u32, (g, U_F64 * g.abs()));
            }
            if a != 0.0 {
                pre.insert(i as u32, (a, U_F64 * a.abs()));
            }
        }
        let bias_shift = -beta * f64::from(facet.b);
        HeadF64Fold {
            target_act,
            relu_name: "head".to_string(),
            head_width,
            post,
            pre,
            bias_shift,
            bias_err: U_F64 * bias_shift.abs(),
        }
    }

    /// ORACLE (a) — GATE-OFF / all-zero fold BYTE-IDENTICAL. With no fold, an
    /// all-zero fold at the matching target, a fold at a NON-matching target, or a
    /// fold whose width disagrees with the layer, `sound_f64_lower_bound` returns
    /// the bit-exact baseline: the default path never perturbs the recovery.
    #[test]
    fn head_facet_oracle_a_byte_identical_when_inert() {
        let (seg, spec, lo, hi) = diamond_head_segments();
        let beta_none: Vec<Vec<f32>> = vec![Vec::new()];
        let base = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, None)
            .expect("baseline recovery");

        // all-zero fold entered at the matching target ⇒ no arithmetic performed.
        let zero = HeadF64Fold {
            target_act: 0,
            relu_name: "head".to_string(),
            head_width: 2,
            post: Default::default(),
            pre: Default::default(),
            bias_shift: 0.0,
            bias_err: 0.0,
        };
        let z = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, Some(&zero)).unwrap();
        assert_eq!(
            base.to_bits(),
            z.to_bits(),
            "all-zero fold must be byte-identical to no fold"
        );

        // A nonzero fold at a NON-matching global target act (99) is never applied.
        let far = diamond_fold(1.0, 99, 2);
        let f = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, Some(&far)).unwrap();
        assert_eq!(
            base.to_bits(),
            f.to_bits(),
            "fold at a non-matching target_act must be byte-identical"
        );

        // A width mismatch (fold expects 3, layer has 2) fails closed to baseline.
        let wrong_w = diamond_fold(1.0, 0, 3);
        let w = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, Some(&wrong_w)).unwrap();
        assert_eq!(
            base.to_bits(),
            w.to_bits(),
            "head_width mismatch must fail closed to byte-identical"
        );
    }

    /// ORACLE (b) — SOUNDNESS / NEVER-A-FALSE-LOWER-BOUND (the critical one). On
    /// the real diamond head, the facet-augmented f64 lower bound must be `≤` the
    /// TRUE min of the margin over the box (dense Monte-Carlo) for EVERY β, WITH
    /// the facet active and nonzero — and must MATERIALLY raise the bound at a good
    /// β (a multi-neuron tightening single-neuron α/β cannot buy). A negative
    /// control proves the oracle has teeth: a wrong-signed (bug) fold DOES exceed
    /// the true min, which is exactly what this test would catch.
    #[test]
    fn head_facet_oracle_b_montecarlo_soundness() {
        let (seg, spec, lo, hi) = diamond_head_segments();
        let beta_none: Vec<Vec<f32>> = vec![Vec::new()];
        let base = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, None).unwrap();

        // Dense Monte-Carlo min of the TRUE network over the box (grid + randoms).
        let mut true_min = f32::INFINITY;
        for gi in 0..=120 {
            for gj in 0..=120 {
                let u = [
                    -1.0 + 2.0 * gi as f32 / 120.0,
                    -1.0 + 2.0 * gj as f32 / 120.0,
                ];
                true_min = true_min.min(true_diamond_margin(&u));
            }
        }
        let mut state = 0xC0FF_EE42u64;
        for _ in 0..300_000 {
            let u = [rngf(&mut state), rngf(&mut state)];
            true_min = true_min.min(true_diamond_margin(&u));
        }

        // (SOUND) folded bound ≤ true min at EVERY β; track the best (tightest).
        let mut best = base;
        for &b in &[0.25f64, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0] {
            let fold = diamond_fold(b, 0, 2);
            let v = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, Some(&fold)).unwrap();
            assert!(
                f64::from(v) <= f64::from(true_min) + 1e-4,
                "UNSOUND: facet-folded bound {v:.6} EXCEEDS true_min {true_min:.6} at beta={b}"
            );
            if v > best {
                best = v;
            }
        }
        eprintln!(
            "[mn-head-facet oracle-b] base={base:.6} best_folded={best:.6} true_min={true_min:.6}"
        );
        // (MATERIAL) the facet lifts the bound toward the true min (base≈−3 → −2).
        assert!(
            best > base + 0.5,
            "facet must MATERIALLY tighten the bound: base={base:.6} best={best:.6}"
        );
        assert!(
            f64::from(best) <= f64::from(true_min) + 1e-4,
            "tightest folded bound must stay sound: best={best:.6} true_min={true_min:.6}"
        );
        assert!(
            (f64::from(best) - f64::from(true_min)).abs() < 1e-1,
            "coupling facet should ~recover the true min: best={best:.6} true_min={true_min:.6}"
        );

        // (NEGATIVE CONTROL — the oracle discriminates) a wrong-signed bias fold
        // (simulating a fold-sign bug) lifts the bound ABOVE the true min. THIS is
        // the failure this Monte-Carlo test exists to catch.
        let bad = HeadF64Fold {
            target_act: 0,
            relu_name: "head".to_string(),
            head_width: 2,
            post: Default::default(),
            pre: Default::default(),
            bias_shift: 5.0,
            bias_err: 0.0,
        };
        let badv = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, Some(&bad)).unwrap();
        assert!(
            f64::from(badv) > f64::from(true_min) + 0.5,
            "negative control: an unsound (+bias) fold MUST exceed true_min — proving \
             the MC oracle would catch a broken fold (badv={badv:.6} true_min={true_min:.6})"
        );
    }

    /// ORACLE (c) — MONOTONE MAX / INTERSECT. The batched merge is
    /// `best_lo[crit] = max(best_lo[crit], folded)`. Assert the facet path enters
    /// ONLY through that max: a tightening fold raises the merged value (never above
    /// the true min), and a LOOSER fold is discarded — the baseline is retained
    /// bit-exact. So best_lo can only rise or stay.
    #[test]
    fn head_facet_oracle_c_monotone_max_intersect() {
        let (seg, spec, lo, hi) = diamond_head_segments();
        let beta_none: Vec<Vec<f32>> = vec![Vec::new()];
        let base = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, None).unwrap();

        // A tightening fold: max(base, good) == good > base (best_lo RISES).
        let good = diamond_fold(1.0, 0, 2);
        let goodv = sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, Some(&good)).unwrap();
        let merged = base.max(goodv);
        assert!(merged >= base, "max can only raise best_lo");
        assert!(
            merged > base,
            "the tightening fold must raise the merged bound: base={base:.6} good={goodv:.6}"
        );

        // A looser fold (extra-negative bias) LOWERS the candidate; the max keeps
        // the baseline bit-exact — the fold contribution can never lower best_lo.
        let loose = HeadF64Fold {
            target_act: 0,
            relu_name: "head".to_string(),
            head_width: 2,
            post: Default::default(),
            pre: Default::default(),
            bias_shift: -2.0,
            bias_err: 0.0,
        };
        let loosev =
            sound_f64_lower_bound(&seg, &spec, &beta_none, &lo, &hi, Some(&loose)).unwrap();
        assert!(
            loosev < base,
            "the constructed loose fold should be below baseline (loosev={loosev:.6} base={base:.6})"
        );
        assert_eq!(
            base.max(loosev).to_bits(),
            base.to_bits(),
            "a looser fold must be DISCARDED by the max (baseline retained bit-exact)"
        );
    }
}
