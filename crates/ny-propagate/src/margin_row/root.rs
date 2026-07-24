// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Frozen root gates from a forward DeepPoly (M/D) tableau (#twinwall).
//!
//! Port of the verified Python reference (`core.py::RootTableau`): a forward
//! tableau `A_l = M - D`, `A_u = M + D` over augmented input rows, frozen
//! DeepPoly gates per trunk relu chosen from the concretized boxes. In
//! [`RoundMode::Outward`] the `D` lane additionally ABSORBS a certified bound
//! on every rounding committed so far (Higham `gamma_n` per conv + elementwise
//! widening), the boxes are concretized outward, and the upper gate lines are
//! REPAIRED so `s*y + c >= relu(y)` holds in real arithmetic on the certified
//! box (checked at the three kink points; `c` is bumped up on any deficit).
//!
//! The tableau stops after the LAST trunk relu: the backward engine owns
//! everything from there to the margin.

use ndarray::Array2;
use ny_core::{NyError, Result};
use rayon::prelude::*;
use std::time::Instant;

use super::net::{conv_apply_forward, conv_apply_forward_prec, ConvOp, TwinNet, TwinOp};
use super::rounding::{
    certify_up, gamma_n, gamma_n_f32, next_down, next_up, RoundMode, SUBNORMAL_F32, UNIT,
};

/// Frozen gates of one trunk relu layer.
pub struct LayerGates {
    /// Relu op index in the net.
    pub op: usize,
    /// Layer width.
    pub n: usize,
    /// Pre-activation box (certified outward in `Outward` mode).
    pub l: Vec<f64>,
    /// Upper box side.
    pub u: Vec<f64>,
    /// Lower-line slope per neuron (0.0 or 1.0).
    pub alpha: Vec<f64>,
    /// Upper-line slope per neuron.
    pub s: Vec<f64>,
    /// Upper-line intercept per neuron (>= 0).
    pub c: Vec<f64>,
    /// `max(alpha, s)` per neuron (error-carry contraction factor).
    pub ms: Vec<f64>,
    /// Unstable neuron indices (l < 0 < u).
    pub unst: Vec<usize>,
}

/// Root state shared by every pass: input box + frozen gates.
pub struct RootGates {
    /// Rounding mode the gates (and every consumer pass) run in.
    pub mode: RoundMode,
    /// Box midpoint `(lo + hi) / 2` (Python-parity formula).
    pub mid: Vec<f64>,
    /// Box radius: parity `(hi - lo) / 2`; outward `next_up(max(hi - mid,
    /// mid - lo))` so `mid ± rad` covers `[lo, hi]` despite midpoint rounding.
    pub rad: Vec<f64>,
    /// `|mid| + rad` per input (magnitude weights for error-penalty dots).
    pub xabs: Vec<f64>,
    /// Per trunk relu layer gates, in execution order.
    pub layers: Vec<LayerGates>,
    /// Original input box lower bounds (kept so epoch rebuilds — #epoch-bab
    /// Tier 2 — reconstruct gates from the DECLARED box, not `mid ± rad`).
    pub lo: Vec<f64>,
    /// Original input box upper bounds.
    pub hi: Vec<f64>,
}

/// Retention policy for the Tier-0 tableau rows (#epoch-bab).
#[derive(Debug, Clone, Copy)]
pub struct RetainCfg {
    /// Max retained unstable neurons per trunk relu layer (top by
    /// `c * (u - l)`, the relaxation-slack split-worthiness score).
    pub per_layer: usize,
    /// Global byte budget across all layers (f32 rows; retention stops
    /// adding layers once exceeded — later layers are dropped first-come).
    pub budget_bytes: usize,
}

impl Default for RetainCfg {
    fn default() -> Self {
        Self {
            per_layer: 128,
            budget_bytes: 256 << 20,
        }
    }
}

/// One trunk layer's retained pre-activation sandwich rows (#epoch-bab).
///
/// RANKER-ONLY: rows are f32 and are consumed exclusively by the
/// nearest-mode trunk variant ranker (`bounds::trunk_variant`). No Outward
/// (verdict-grade) pass ever reads them.
pub struct RetainedLayer {
    /// Absolute neuron index per retained row (ascending).
    pub idx: Vec<usize>,
    /// Position of each retained neuron in the layer's `unst` list.
    pub unst_pos: Vec<usize>,
    /// Lower sandwich rows `M - D`, `(n_ret, n_in + 1)` row-major.
    pub a_l: Vec<f32>,
    /// Upper sandwich rows `M + D`, `(n_ret, n_in + 1)` row-major.
    pub a_u: Vec<f32>,
    /// Augmented row width (`n_in + 1`).
    pub naug: usize,
}

/// Retained rows for all trunk layers, aligned with `RootGates::layers`
/// (entry `li` may be empty when the layer had no unstable neurons or the
/// budget ran out).
pub struct RetainedRows {
    /// Per-layer retained rows.
    pub layers: Vec<RetainedLayer>,
}

impl RetainedLayer {
    fn empty(naug: usize) -> Self {
        Self {
            idx: Vec::new(),
            unst_pos: Vec::new(),
            a_l: Vec::new(),
            a_u: Vec::new(),
            naug,
        }
    }

    /// Bytes held by this layer's rows.
    pub fn bytes(&self) -> usize {
        (self.a_l.len() + self.a_u.len()) * size_of::<f32>()
    }
}

/// Is the SOUND f32 root-tableau conv fast path requested? Opt-in
/// (`NY_MARGIN_ROW_ROOT_F32=1`), default OFF — the bit-for-bit f64 lane. When
/// on, the two bandwidth-bound forward-conv lanes (M and D) run in f32 and a
/// certified additive concretize slack (accumulated per op) dominates the
/// worst-case effect of that f32 rounding on every box endpoint. Pure loosening:
/// `dj` can only shrink, never a false-UNSAT (moat-safe).
fn root_f32_requested() -> bool {
    matches!(
        std::env::var("NY_MARGIN_ROW_ROOT_F32").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    )
}

impl RootGates {
    /// Build the tableau and gates. `deadline`: fail with `Timeout` when
    /// exceeded (checked per op). Honors `NY_MARGIN_ROW_ROOT_F32`.
    pub fn build(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
    ) -> Result<Self> {
        Ok(Self::build_retaining(net, lo, hi, mode, deadline, None, &[])?.0)
    }

    /// Build with an explicit f32-fast-path override (bypasses the env gate;
    /// used by the differential/enclosure oracles to compare f32-ON vs f64-OFF
    /// deterministically). `use_f32` only takes effect in [`RoundMode::Outward`]
    /// (the verdict mode); `Parity` always runs bit-for-bit f64.
    pub fn build_prec(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        use_f32: bool,
    ) -> Result<Self> {
        Ok(Self::build_retaining_inner(net, lo, hi, mode, deadline, None, &[], use_f32)?.0)
    }

    /// Build with optional Tier-0 row retention (#epoch-bab) and optional
    /// baked trunk splits (#epoch-bab Tier 2).
    ///
    /// `retain`: when set, the pre-activation sandwich rows `M ± D` of the
    /// top unstable neurons (by `c * (u - l)`) are copied out per layer,
    /// f32, ranker-only.
    ///
    /// `splits`: `(trunk_layer, absolute_neuron, dir)` piece-fixes BAKED
    /// into the forward tableau: the neuron's gates are overridden to the
    /// exact fixed lines ((1,1,0) active / (0,0,0) inactive) and the neuron
    /// is removed from the layer's `unst` list, so every downstream tableau
    /// row and box tightens. The resulting gates are valid exactly on the
    /// split-halfspace intersection (the calling subtree's domain) — the
    /// same soundness contract as `engine::domain_gates`, moved into the
    /// forward build.
    pub fn build_retaining(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            splits,
            root_f32_requested(),
        )
    }

    /// Full build implementation with an EXPLICIT f32-fast-path flag
    /// (`NY_MARGIN_ROW_ROOT_F32`). `use_f32` only takes effect in
    /// [`RoundMode::Outward`]; when off, every bound is bit-identical to the
    /// pure-f64 lane (epoch-bab retention/splits included). When on, the two
    /// bandwidth-bound forward-conv lanes (M and D) run in f32 and the
    /// per-tensor certified error accumulator `g_err[j]` is threaded through
    /// each op and consumed as a pure additive concretize slack.
    #[allow(clippy::too_many_arguments)]
    fn build_retaining_inner(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        use_f32: bool,
    ) -> Result<(Self, Option<RetainedRows>)> {
        // f32 lanes + the certified error slack are only meaningful in the
        // certified-outward verdict mode.
        let use_f32 = use_f32 && mode.outward();
        let n_in = net.n_in;
        if lo.len() != n_in || hi.len() != n_in {
            return Err(NyError::shape_mismatch(vec![n_in], vec![lo.len()]));
        }
        let mut mid = vec![0.0; n_in];
        let mut rad = vec![0.0; n_in];
        for i in 0..n_in {
            if !(lo[i].is_finite() && hi[i].is_finite() && lo[i] <= hi[i]) {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: bad input box at {i}: [{}, {}]",
                    lo[i], hi[i]
                )));
            }
            // Parity-critical formula (core.py: mid = (lo+hi)/2): do NOT
            // replace with f64::midpoint (different rounding on some inputs).
            #[allow(clippy::manual_midpoint)]
            {
                mid[i] = (lo[i] + hi[i]) / 2.0;
            }
            rad[i] = if mode.outward() {
                next_up((hi[i] - mid[i]).max(mid[i] - lo[i]))
            } else {
                (hi[i] - lo[i]) / 2.0
            };
        }
        let xabs: Vec<f64> = mid.iter().zip(&rad).map(|(m, r)| m.abs() + r).collect();
        let naug = n_in + 1;

        // Sum of the augmented input magnitude weights (xabs on the input
        // columns, 1.0 on the bias column). Constant across the tableau; scales
        // the per-conv f32 FTZ/subnormal absolute floor in the g_err accumulator.
        let s_xabs: f64 = next_up(xabs.iter().sum::<f64>() + 1.0);

        // Consumer counts over the processed prefix (up to last trunk relu).
        let last_relu = *net.trunk_relus.last().expect("validated: non-empty");
        let mut consumers = vec![0usize; net.ops.len() + 1];
        for op in &net.ops[..=last_relu] {
            match op {
                TwinOp::Conv(c) => consumers[c.input] += 1,
                TwinOp::Relu { input, .. }
                | TwinOp::Flatten { input }
                | TwinOp::ChannelAffine { input, .. } => consumers[*input] += 1,
                TwinOp::Add { lhs, rhs } => {
                    consumers[*lhs] += 1;
                    consumers[*rhs] += 1;
                }
                TwinOp::Gemm { .. } => {
                    return Err(NyError::InvalidSpec(
                        "margin_row: gemm before last trunk relu".into(),
                    ))
                }
            }
        }

        // Identity tableau at the input.
        let mut m0 = Array2::<f64>::zeros((n_in, naug));
        for i in 0..n_in {
            m0[[i, i]] = 1.0;
        }
        let d0 = Array2::<f64>::zeros((n_in, naug));
        // Per-tensor certified f32-error accumulator `g_err[j]` (`Outward`+`use_f32`
        // only; empty otherwise). Invariant: `g_err[j] >=` the exact worst-case
        // perturbation of neuron j's concretized endpoints caused by running the
        // forward-conv lanes in f32 vs exact arithmetic, i.e.
        // `sum_i (|dM_ji| + |dD_ji|) * xabs_ext_i` (bias column weight 1). Consumed
        // as a pure additive concretize slack. Identity input carries zero error.
        let g0: Vec<f64> = if use_f32 { vec![0.0; n_in] } else { Vec::new() };
        let mut state: Vec<Option<(Array2<f64>, Array2<f64>, Vec<f64>)>> =
            vec![None; net.ops.len() + 1];
        state[0] = Some((m0, d0, g0));
        let mut layers = Vec::new();

        // Baked splits per trunk layer (#epoch-bab Tier 2).
        let mut split_by_layer: std::collections::BTreeMap<usize, Vec<(usize, i8)>> =
            std::collections::BTreeMap::new();
        for &(li, idx, dir) in splits {
            split_by_layer.entry(li).or_default().push((idx, dir));
        }
        // Tier-0 retention accumulator (#epoch-bab).
        let mut retained = retain.map(|_| RetainedRows { layers: Vec::new() });
        let mut retained_bytes = 0usize;

        // Optional per-op wall-clock breakdown (NY_MARGIN_ROW_ROOT_TIMING);
        // pure diagnostics, no effect on any bound.
        let timing = std::env::var("NY_MARGIN_ROW_ROOT_TIMING").is_ok();
        let build_t0 = Instant::now();
        let mut op_times: Vec<(usize, &'static str, usize, f64)> = Vec::new();

        for (k, op) in net.ops.iter().enumerate().take(last_relu + 1) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    if timing {
                        print_root_timing(&op_times, build_t0.elapsed().as_secs_f64(), false);
                    }
                    return Err(NyError::DeadlineExceeded(
                        "margin_row root tableau deadline".into(),
                    ));
                }
            }
            let op_t0 = Instant::now();
            let take = |st: &mut Vec<Option<(Array2<f64>, Array2<f64>, Vec<f64>)>>,
                        cons: &mut Vec<usize>,
                        id: usize|
             -> Result<(Array2<f64>, Array2<f64>, Vec<f64>, bool)> {
                let last = cons[id] == 1;
                cons[id] -= 1;
                let entry = st[id]
                    .as_ref()
                    .ok_or_else(|| NyError::InvalidSpec("margin_row: dead tensor".into()))?;
                if last {
                    let owned = st[id].take().expect("checked above");
                    Ok((owned.0, owned.1, owned.2, true))
                } else {
                    Ok((entry.0.clone(), entry.1.clone(), entry.2.clone(), false))
                }
            };
            match op {
                TwinOp::Conv(c) => {
                    let (mi, di, gerr_in, _) = take(&mut state, &mut consumers, c.input)?;
                    let n_out = net.tsize[k + 1];
                    let mut mo = Array2::<f64>::zeros((n_out, naug));
                    // M lane (f64 default, or f32 fast path stored back exactly).
                    conv_apply_forward_prec(c, &mi, &mut mo, false, use_f32);
                    let mut do_ = Array2::<f64>::zeros((n_out, naug));
                    // g_err accumulation for the f32 fast path (see conv_f32_gerr
                    // doc): g_err_out = conv_abs(g_err_in + gamma_f32 * B) + FTZ
                    // floor, with B[t] = sum_i (|mi_ti| + din_ti) * xabs_ext_i.
                    // Needs `din`, so compute it here and reuse for the D lane.
                    let mut gerr_out: Vec<f64> = Vec::new();
                    if mode.outward() {
                        // D input absorbs the M-product error: Din = D + g(|M| + D).
                        let g = next_up(
                            gamma_n(c.k_fwd + 2)
                                + c.weight_rel_err
                                + gamma_n(c.k_fwd + 2) * c.weight_rel_err,
                        );
                        let mut din = Array2::<f64>::zeros(mi.raw_dim());
                        par_zip3(&mut din, &mi, &di, |dst, m, d| *dst = d + g * (m.abs() + d));
                        if use_f32 {
                            gerr_out = conv_f32_gerr(c, &mi, &din, &gerr_in, &xabs, s_xabs, n_in);
                        }
                        conv_apply_forward_prec(c, &din, &mut do_, true, use_f32);
                        let g2 = gamma_n(c.k_fwd + 8);
                        do_.par_mapv_inplace(|v| certify_up(v, g2));
                    } else {
                        conv_apply_forward_prec(c, &di, &mut do_, true, use_f32);
                    }
                    // Bias into the M bias column (+ certified bias error into D).
                    let p = c.oshape.1 * c.oshape.2;
                    for j in 0..n_out {
                        let ch = j / p;
                        mo[[j, n_in]] += c.bias[ch];
                        if mode.outward() {
                            let extra = next_up(c.bias_err[ch] + UNIT * mo[[j, n_in]].abs());
                            do_[[j, n_in]] = next_up(do_[[j, n_in]] + extra);
                        }
                    }
                    state[k + 1] = Some((mo, do_, gerr_out));
                }
                TwinOp::Add { lhs, rhs } => {
                    let (ma, da, ga, _) = take(&mut state, &mut consumers, *lhs)?;
                    let (mb, db, gb, _) = take(&mut state, &mut consumers, *rhs)?;
                    let mut mo = ma;
                    mo += &mb;
                    let mut do_ = da;
                    if mode.outward() {
                        // do_ still holds da: dst = widen(da + db + 2u|mo|).
                        par_zip3(&mut do_, &db, &mo, |dst, dbv, mv| {
                            *dst = next_up(((*dst + dbv) + 2.0 * UNIT * mv.abs()) * (1.0 + 1e-15));
                        });
                    } else {
                        do_ += &db;
                    }
                    // f32 error is additive across the two branches: the elementwise
                    // sum's coefficient errors add (its own f64 add rounding is a
                    // second-order effect on the errors, absorbed by next_up).
                    let gerr_out: Vec<f64> = if use_f32 {
                        ga.iter().zip(&gb).map(|(a, b)| next_up(a + b)).collect()
                    } else {
                        Vec::new()
                    };
                    state[k + 1] = Some((mo, do_, gerr_out));
                }
                TwinOp::Flatten { input } => {
                    let (mi, di, gi, _) = take(&mut state, &mut consumers, *input)?;
                    state[k + 1] = Some((mi, di, gi));
                }
                TwinOp::ChannelAffine {
                    input,
                    scale,
                    shift,
                    scale_rel_err,
                    shift_err,
                } => {
                    // Diagonal affine on the tableau: M' = s ⊙ M (+ shift in
                    // the bias column); D' = |s| ⊙ D widened by the certified
                    // parameter/rounding envelope in Outward mode.
                    let (mut mi, mut di, gerr_in, _) = take(&mut state, &mut consumers, *input)?;
                    let g = next_up(*scale_rel_err + 4.0 * UNIT);
                    let msl = mi.as_slice_mut().expect("standard layout");
                    let dsl = di.as_slice_mut().expect("standard layout");
                    for (j, (&sj, &tj)) in scale.iter().zip(shift.iter()).enumerate() {
                        let sa = sj.abs();
                        let mrow = &mut msl[j * naug..(j + 1) * naug];
                        let drow = &mut dsl[j * naug..(j + 1) * naug];
                        for i in 0..naug {
                            let m2 = sj * mrow[i];
                            if mode.outward() {
                                drow[i] = next_up(
                                    (sa * drow[i] + g * (m2.abs() + sa * drow[i])) * (1.0 + 1e-15),
                                );
                            } else {
                                drow[i] *= sa;
                            }
                            mrow[i] = m2;
                        }
                        let b2 = mrow[n_in] + tj;
                        if mode.outward() {
                            drow[n_in] = next_up(drow[n_in] + shift_err[j] + 2.0 * UNIT * b2.abs());
                        }
                        mrow[n_in] = b2;
                    }
                    // f32 error carries through the diagonal scale: the new
                    // coefficient errors are |dM'| + |dD'| <= |scale_j| * (|dM| +
                    // |dD|) (the shift enters only the bias column with no f32
                    // error; the affine's own f64 rounding on the error-carrying
                    // coefficients is second order, absorbed by next_up + 1e-15).
                    let gerr_out: Vec<f64> = if use_f32 {
                        scale
                            .iter()
                            .zip(&gerr_in)
                            .map(|(&sj, &gj)| next_up(sj.abs() * gj * (1.0 + 1e-15)))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    state[k + 1] = Some((mi, di, gerr_out));
                }
                TwinOp::Relu { input, layer } => {
                    let (mi, di, gerr_in, _) = take(&mut state, &mut consumers, *input)?;
                    let n = net.tsize[k + 1];
                    let f32_gerr = if use_f32 {
                        Some(gerr_in.as_slice())
                    } else {
                        None
                    };
                    let (l, u) = concretize_box(
                        &mi,
                        &di,
                        &mid,
                        &rad,
                        &xabs,
                        mode,
                        net.trunk_relus.len(),
                        f32_gerr,
                    );
                    for v in l.iter().chain(u.iter()) {
                        if !v.is_finite() {
                            return Err(NyError::NumericalInstability(
                                "margin_row: non-finite tableau box".into(),
                            ));
                        }
                    }
                    let mut gates = gates_from_box(&l, &u);
                    if mode.outward() {
                        repair_upper_lines(&l, &u, &mut gates.1, &mut gates.2);
                    }
                    let (mut alpha, mut s, mut c) = gates;
                    let li = layers.len();
                    // Bake this layer's splits: exact fixed lines, valid on
                    // the split halfspaces (the calling subtree's domain).
                    let mut baked: std::collections::BTreeSet<usize> =
                        std::collections::BTreeSet::new();
                    if let Some(list) = split_by_layer.get(&li) {
                        for &(idx, dir) in list {
                            if idx >= n {
                                return Err(NyError::InvalidSpec(format!(
                                    "margin_row: baked split neuron {idx} out of range {n}"
                                )));
                            }
                            if dir > 0 {
                                alpha[idx] = 1.0;
                                s[idx] = 1.0;
                                c[idx] = 0.0;
                            } else {
                                alpha[idx] = 0.0;
                                s[idx] = 0.0;
                                c[idx] = 0.0;
                            }
                            baked.insert(idx);
                        }
                    }
                    let ms: Vec<f64> = alpha.iter().zip(&s).map(|(a, b)| a.max(*b)).collect();
                    let unst: Vec<usize> = (0..n)
                        .filter(|&j| l[j] < 0.0 && u[j] > 0.0 && !baked.contains(&j))
                        .collect();
                    // Tier-0 retention: pre-activation sandwich rows M ± D of
                    // the top unstable neurons by relaxation slack c*(u-l).
                    // f32, RANKER-ONLY (see `RetainedLayer` docs).
                    if let (Some(ret), Some(cfg)) = (retained.as_mut(), retain) {
                        let mut layer_ret = RetainedLayer::empty(naug);
                        if cfg.per_layer > 0 && retained_bytes < cfg.budget_bytes {
                            let mut scored: Vec<(f64, usize, usize)> = unst
                                .iter()
                                .enumerate()
                                .map(|(pos, &j)| (c[j] * (u[j] - l[j]), j, pos))
                                .collect();
                            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                            scored.truncate(cfg.per_layer);
                            // Ascending neuron index for cache-friendly reads.
                            scored.sort_by_key(|&(_, j, _)| j);
                            let msrc = mi.as_slice().expect("standard layout");
                            let dsrc = di.as_slice().expect("standard layout");
                            for &(_, j, pos) in &scored {
                                let mrow = &msrc[j * naug..(j + 1) * naug];
                                let drow = &dsrc[j * naug..(j + 1) * naug];
                                layer_ret.idx.push(j);
                                layer_ret.unst_pos.push(pos);
                                #[allow(clippy::cast_possible_truncation)]
                                for i in 0..naug {
                                    layer_ret.a_l.push((mrow[i] - drow[i]) as f32);
                                    layer_ret.a_u.push((mrow[i] + drow[i]) as f32);
                                }
                            }
                            retained_bytes += layer_ret.bytes();
                        }
                        ret.layers.push(layer_ret);
                    }
                    layers.push(LayerGates {
                        op: k,
                        n,
                        l,
                        u,
                        alpha: alpha.clone(),
                        s: s.clone(),
                        c: c.clone(),
                        ms,
                        unst,
                    });
                    debug_assert_eq!(layers.len() - 1, *layer);
                    if k == last_relu {
                        if timing {
                            op_times.push((k, "relu", n, op_t0.elapsed().as_secs_f64()));
                        }
                        break; // downstream tableau unused beyond retention
                    }
                    // Apply gates: L = (M-D)*alpha; U = (M+D)*s + c@bias.
                    let (mo, do_) = apply_gates(&mi, &di, &alpha, &s, &c, n_in, mode);
                    // Propagate the f32 error through the gate transform. Per
                    // neuron j the new coefficient errors are
                    // |dM'| + |dD'| <= (alpha_j + s_j) * (|dM| + |dD|), so the
                    // error functional scales by (alpha_j + s_j) <= 2 (the intercept
                    // c carries no f32 error; the gate's own f64 rounding is second
                    // order, absorbed by next_up). alpha, s in [0, 1] (baked splits
                    // are (1,1) or (0,0), both within this bound).
                    let gerr_out: Vec<f64> = if use_f32 {
                        (0..n)
                            .map(|j| next_up((alpha[j] + s[j]) * gerr_in[j]))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    state[k + 1] = Some((mo, do_, gerr_out));
                }
                TwinOp::Gemm { .. } => unreachable!("guarded above"),
            }
            if timing {
                let kind = match op {
                    TwinOp::Conv(_) => "conv",
                    TwinOp::Add { .. } => "add",
                    TwinOp::Flatten { .. } => "flatten",
                    TwinOp::ChannelAffine { .. } => "chaffine",
                    TwinOp::Relu { .. } => "relu",
                    TwinOp::Gemm { .. } => "gemm",
                };
                op_times.push((k, kind, net.tsize[k + 1], op_t0.elapsed().as_secs_f64()));
            }
        }
        if timing {
            print_root_timing(&op_times, build_t0.elapsed().as_secs_f64(), true);
        }
        Ok((
            Self {
                mode,
                mid,
                rad,
                xabs,
                layers,
                lo: lo.to_vec(),
                hi: hi.to_vec(),
            },
            retained,
        ))
    }
}

/// Diagnostic dump of the per-op tableau build times (NY_MARGIN_ROW_ROOT_TIMING).
/// `completed` distinguishes a finished build from a deadline-truncated one.
fn print_root_timing(op_times: &[(usize, &'static str, usize, f64)], total: f64, completed: bool) {
    let mut by_kind: std::collections::BTreeMap<&'static str, (usize, f64)> =
        std::collections::BTreeMap::new();
    for &(_, kind, _, secs) in op_times {
        let e = by_kind.entry(kind).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += secs;
    }
    let tag = if completed {
        "complete"
    } else {
        "DEADLINE-truncated"
    };
    eprintln!(
        "[root-timing] {tag}: total={total:.2}s over {} ops",
        op_times.len()
    );
    for (kind, (cnt, secs)) in &by_kind {
        eprintln!("[root-timing]   {kind:<8} {secs:7.2}s  (x{cnt})");
    }
    let mut slow: Vec<_> = op_times.to_vec();
    slow.sort_by(|a, b| b.3.total_cmp(&a.3));
    for &(k, kind, n_out, secs) in slow.iter().take(8) {
        eprintln!("[root-timing]   slow: op {k:<3} {kind:<8} n_out={n_out:<7} {secs:6.2}s");
    }
}

/// Elementwise 3-way zip with rayon over rows.
fn par_zip3(
    dst: &mut Array2<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
    f: impl Fn(&mut f64, f64, f64) + Sync,
) {
    let cols = dst.ncols();
    let ds = dst.as_slice_mut().expect("standard layout");
    let asl = a.as_slice().expect("standard layout");
    let bs = b.as_slice().expect("standard layout");
    ds.par_chunks_mut(cols)
        .zip(asl.par_chunks(cols).zip(bs.par_chunks(cols)))
        .for_each(|(d, (ar, br))| {
            for ((dv, &av), &bv) in d.iter_mut().zip(ar).zip(br) {
                f(dv, av, bv);
            }
        });
}

/// f32-fast-path g_err propagation across ONE forward conv (`NY_MARGIN_ROW_ROOT_F32`).
///
/// Returns the per-output-neuron certified error functional
/// `g_err_out[j] = sum_i (|dM_out_ji| + |dD_out_ji|) * xabs_ext_i`, bounding the
/// worst-case effect on neuron j's concretized endpoints of running BOTH conv
/// lanes in f32 vs exact arithmetic. Derivation (Higham sec. 3.5, per output
/// coefficient, valid for ANY tap/SIMD order):
///
/// * f32 M-lane error at `(j,i)`: `<= gamma_f32 * (|W| @ |mi|)_ji` — the f32
///   rounding of an accumulation of `k_fwd` terms plus the input f64->f32 and
///   weight->f32 conversions (all folded into `gamma_n_f32(k_fwd+8)`).
/// * f32 D-lane error at `(j,i)`: `<= gamma_f32 * (|W| @ din)_ji`.
/// * propagated input error: `<= (|W| @ g_err_in-as-coeffs)`.
///
/// Summing `E_out_ji * xabs_ext_i` over `i` collapses the per-coefficient matrix
/// into a per-neuron VECTOR conv (`|W|` gather, ONE column) over
/// `g_err_in[t] + gamma_f32 * B[t]`, with `B[t] = sum_i (|mi_ti| + din_ti) *
/// xabs_ext_i` (bias column weight 1). Cost: one length-`n` reduction + one
/// single-column `conv_abs` — negligible beside the two full-width conv lanes.
/// Everything is rounded outward (upper bound). A per-output additive floor
/// covers f32 subnormal/FTZ rounding. Pure over-estimate -> pure loosening.
fn conv_f32_gerr(
    c: &ConvOp,
    mi: &Array2<f64>,
    din: &Array2<f64>,
    gerr_in: &[f64],
    xabs: &[f64],
    s_xabs: f64,
    n_in: usize,
) -> Vec<f64> {
    let n_src = mi.nrows();
    let naug = n_in + 1;
    debug_assert_eq!(gerr_in.len(), n_src);
    // gamma_f32 covers: input f64->f32 conv, weight f32 conv, the f32 multiply,
    // and the k_fwd-1 f32 additions (+ generous headroom). Order-independent.
    let gf32 = next_up(gamma_n_f32(c.k_fwd + 8));
    // f64 rounding envelope of the length-naug B reduction below.
    let bwiden = gamma_n(naug + 2);
    let mis = mi.as_slice().expect("standard layout");
    let dins = din.as_slice().expect("standard layout");
    // ginp[t] = g_err_in[t] + gamma_f32 * B[t]  (a per-input-neuron scalar).
    let mut ginp = Array2::<f64>::zeros((n_src, 1));
    ginp.as_slice_mut()
        .expect("standard layout")
        .par_iter_mut()
        .enumerate()
        .for_each(|(t, gt)| {
            let mrow = &mis[t * naug..(t + 1) * naug];
            let drow = &dins[t * naug..(t + 1) * naug];
            let mut braw = 0.0;
            for i in 0..n_in {
                braw += (mrow[i].abs() + drow[i]) * xabs[i];
            }
            // Bias column (index n_in): xabs_ext weight = 1.
            braw += mrow[n_in].abs() + drow[n_in];
            let b_up = next_up(braw * (1.0 + bwiden));
            *gt = next_up(gerr_in[t] + gf32 * b_up);
        });
    // g_err_out = |W| (gather) @ ginp  — a single-column forward conv_abs (f64).
    let n_out = c.oshape.0 * c.oshape.1 * c.oshape.2;
    let mut gout = Array2::<f64>::zeros((n_out, 1));
    conv_apply_forward(c, &ginp, &mut gout, true);
    // Widen for the vector conv's own f64 rounding, then add the per-output f32
    // subnormal/FTZ absolute floor (M and D lanes each <= (k_fwd+2)*2^-149 per
    // output coefficient; summed over i weighted by xabs_ext gives *S).
    let gwiden = gamma_n(c.k_fwd + 2);
    let ftz = next_up(2.0 * (c.k_fwd as f64 + 2.0) * SUBNORMAL_F32 * s_xabs);
    gout.as_slice()
        .expect("standard layout")
        .iter()
        .map(|&v| next_up(v * (1.0 + gwiden) + ftz))
        .collect()
}

/// Concretize the (M, D) tableau to per-neuron boxes over `mid ± rad`.
/// Python-parity formula: `L = M - D; l = L@mid + L_bias - |L|@rad` (and the
/// mirrored upper). Outward: additionally subtract/add a Higham envelope over
/// the whole accumulation and round the endpoints outward.
///
/// `f32_gerr` (the `NY_MARGIN_ROW_ROOT_F32` fast path): a per-neuron certified
/// upper bound on `sum_i (|dM_ji| + |dD_ji|) * xabs_ext_i + |dM_bias| +
/// |dD_bias|`, i.e. the exact worst-case perturbation of THIS neuron's `low`/
/// `upp` caused by the f32 conv rounding vs exact arithmetic (for BOTH signs of
/// `M-D` and of `mid`, since `|Δ(vl+bl-rl)| <= sum |Δ coeff| * (|mid_i|+rad_i)`
/// and `xabs_ext_i = |mid_i|+rad_i`). Added to the existing `gam` envelope as a
/// pure additive slack: SUBTRACTED from `low`, ADDED to `upp` — the box only
/// GROWS, so `dj` can only shrink (never a false-UNSAT). The base `gam` term
/// still covers the identical concretize-rounding + depth-leak for the f32
/// arrays exactly as for f64 (the f32 error is orthogonal, charged separately).
#[allow(clippy::too_many_arguments)]
fn concretize_box(
    m: &Array2<f64>,
    d: &Array2<f64>,
    mid: &[f64],
    rad: &[f64],
    xabs: &[f64],
    mode: RoundMode,
    relu_depth: usize,
    f32_gerr: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let n = m.nrows();
    let n_in = mid.len();
    // Depth-scaled outward headroom. The concretize `rl` term (|M-D|*rad below) already
    // re-absorbs each layer's ~6u coefficient-rounding D-widening, so unlike the backward
    // bias there is no first-order unbounded running sum. What remains is a second-order
    // ~8u*tabs per-trunk-relu leak that is NOT re-absorbed when a relu is separated from
    // the next concretize only by Add/Flatten (no intervening conv g-term). A fixed +16u
    // covered this only while 6*R << n_in; scaling the slack by the trunk relu count R
    // makes the soundness margin DEPTH-INDEPENDENT (root-tableau adversarial oracle +
    // formal analysis, wf_af482867). Pure widening: dj can only shrink -> never a
    // false-UNSAT (moat-safe).
    let gam = gamma_n(n_in + 16 + 8 * relu_depth);
    let ms = m.as_slice().expect("standard layout");
    let ds = d.as_slice().expect("standard layout");
    let naug = n_in + 1;
    let mut l = vec![0.0; n];
    let mut u = vec![0.0; n];
    l.par_iter_mut()
        .zip(u.par_iter_mut())
        .enumerate()
        .for_each(|(j, (lj, uj))| {
            let mrow = &ms[j * naug..(j + 1) * naug];
            let drow = &ds[j * naug..(j + 1) * naug];
            let mut vl = 0.0;
            let mut rl = 0.0;
            let mut vu = 0.0;
            let mut ru = 0.0;
            let mut tabs = 0.0;
            for i in 0..n_in {
                let lo_c = mrow[i] - drow[i];
                let up_c = mrow[i] + drow[i];
                vl += lo_c * mid[i];
                rl += lo_c.abs() * rad[i];
                vu += up_c * mid[i];
                ru += up_c.abs() * rad[i];
                tabs += (lo_c.abs() + up_c.abs()) * xabs[i];
            }
            let bl = mrow[n_in] - drow[n_in];
            let bu = mrow[n_in] + drow[n_in];
            let low = vl + bl - rl;
            let upp = vu + bu + ru;
            if mode.outward() {
                let base = next_up(gam * (tabs + bl.abs() + bu.abs()));
                // f32-fast-path additive term (0 on the f64 default lane).
                let slack = match f32_gerr {
                    Some(g) => next_up(base + g[j]),
                    None => base,
                };
                *lj = next_down(next_down(low - slack));
                *uj = next_up(next_up(upp + slack));
            } else {
                *lj = low;
                *uj = upp;
            }
        });
    (l, u)
}

/// DeepPoly gates from a box — exact Python-parity formulas
/// (`core.py::gates_from_box`): active `l >= 0` => (1,1,0); inactive
/// (else, `u <= 0`) => (0,0,0); unstable => `s = u/(u-l)`,
/// `c = -u*l/(u-l)`, `alpha = [u >= -l]`.
pub fn gates_from_box(l: &[f64], u: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = l.len();
    let mut alpha = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut c = vec![0.0; n];
    for j in 0..n {
        if l[j] >= 0.0 {
            alpha[j] = 1.0;
            s[j] = 1.0;
        } else if u[j] > 0.0 {
            let (uu, ll) = (u[j], l[j]);
            s[j] = uu / (uu - ll);
            c[j] = -uu * ll / (uu - ll);
            alpha[j] = if uu >= -ll { 1.0 } else { 0.0 };
        }
    }
    (alpha, s, c)
}

/// Certified upper-line repair (outward mode): the f64 chord `(s, c)` must
/// satisfy `s*y + c >= relu(y)` in REAL arithmetic for all `y` in `[l, u]`.
/// Linear-over-convex reduces this to the kinks: `c >= 0`, `s*l + c >= 0`,
/// `s*u + c >= u`. Each f64 product is within 1 ulp of real, so a deficit
/// bump of `max(0, -(s*l + c), u - (s*u + c))` plus 4-ulp headroom on the
/// checked magnitudes makes both hold; bumping `c` upward only loosens the
/// relaxation (always sound).
pub fn repair_upper_lines(l: &[f64], u: &[f64], s: &mut [f64], c: &mut [f64]) {
    for j in 0..l.len() {
        if s[j] == 0.0 && c[j] == 0.0 {
            continue; // inactive: exact
        }
        if s[j] == 1.0 && c[j] == 0.0 {
            continue; // active: exact (identity line dominates relu everywhere above l>=0)
        }
        if c[j] < 0.0 {
            c[j] = 0.0;
        }
        let sl = s[j] * l[j];
        let su = s[j] * u[j];
        let head = 4.0 * UNIT * (sl.abs() + su.abs() + c[j].abs() + u[j].abs());
        let d1 = -(sl + c[j]) + head;
        let d2 = u[j] - (su + c[j]) + head;
        let bump = d1.max(d2).max(0.0);
        if bump > 0.0 {
            c[j] = next_up(next_up(c[j] + bump));
        }
    }
}

/// Gate application on the tableau (Python parity):
/// `L = (M-D)*alpha; U = (M+D)*s; U_bias += c; M' = (L+U)/2; D' = (U-L)/2`.
/// Outward: widen `D'` by the elementwise rounding envelope.
fn apply_gates(
    m: &Array2<f64>,
    d: &Array2<f64>,
    alpha: &[f64],
    s: &[f64],
    c: &[f64],
    n_in: usize,
    mode: RoundMode,
) -> (Array2<f64>, Array2<f64>) {
    let naug = n_in + 1;
    let n = m.nrows();
    let mut mo = Array2::<f64>::zeros((n, naug));
    let mut do_ = Array2::<f64>::zeros((n, naug));
    let msrc = m.as_slice().expect("standard layout");
    let dsrc = d.as_slice().expect("standard layout");
    let mdst = mo.as_slice_mut().expect("standard layout");
    let ddst = do_.as_slice_mut().expect("standard layout");
    mdst.par_chunks_mut(naug)
        .zip(ddst.par_chunks_mut(naug))
        .enumerate()
        .for_each(|(j, (mrow, drow))| {
            let a_j = alpha[j];
            let s_j = s[j];
            let msr = &msrc[j * naug..(j + 1) * naug];
            let dsr = &dsrc[j * naug..(j + 1) * naug];
            for i in 0..naug {
                let lo_c = (msr[i] - dsr[i]) * a_j;
                let mut up_c = (msr[i] + dsr[i]) * s_j;
                if i == n_in {
                    up_c += c[j];
                }
                // Bit-identical (a+b)*0.5 anchor: midpoint's overflow-edge branch
                // would move the produced center on this bound path.
                #[allow(clippy::manual_midpoint)]
                let mm = (lo_c + up_c) * 0.5;
                let dd = (up_c - lo_c) * 0.5;
                if mode.outward() {
                    mrow[i] = mm;
                    drow[i] = next_up((dd + 6.0 * UNIT * (mm.abs() + dd)) * (1.0 + 1e-15));
                } else {
                    mrow[i] = mm;
                    drow[i] = dd;
                }
            }
        });
    (mo, do_)
}
