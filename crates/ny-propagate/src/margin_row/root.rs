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

use super::net::{conv_apply_forward, conv_apply_forward_prec_masked, ConvOp, TwinNet, TwinOp};
use super::rounding::{
    certify_up, gamma_n, gamma_n_f32, next_down, next_up, RoundMode, SUBNORMAL_F32, UNIT,
};

// --------------------------------------------------------------------------
//  #tableau-support-mask: exact column-block occupancy of the (M, D) tableau
// --------------------------------------------------------------------------
//
// WHAT THIS IS. Every tableau row is `naug = n_in + 1` wide (9409 on
// tinyimagenet), but for the first ~two thirds of a conv stack most of that
// width is EXACTLY ZERO: a neuron's row can only be nonzero on its receptive
// field, and the receptive field does not cover the whole image until late.
// The mask records, per tableau ROW, which of [`SUPPORT_BLOCKS`] equal column
// blocks can hold a nonzero coefficient. Blocks outside it are skipped by the
// forward conv, the gate application and the concretize reduction.
//
// THE INVARIANT (this is the whole soundness argument):
//
//   for every row j and every block b NOT set in mask[j],
//   `M[j, i] == 0.0` and `D[j, i] == 0.0` EXACTLY for every column i in b.
//
// It is established at the identity input and preserved by every op:
//   * conv  — out row (oc, sp) is a weighted sum of the gathered src rows, so
//     its support is contained in their union; the D lane uses |W| over the
//     same gather, so it has the same containment. The bias column is always
//     retained (the conv folds `c.bias` / `c.bias_err` into it).
//   * add   — union of the two operand masks.
//   * relu/gate application, channel affine, flatten — per-row elementwise, so
//     the mask is unchanged (plus the bias column, which `c[j]` / `shift[j]`
//     writes into).
//
// WHY SKIPPING IS EXACT, NOT A RELAXATION. Inside a masked-out block both
// operands of every one of these ops are exactly `0.0`, so the exact result is
// exactly `0.0` and writing `0.0` is not an approximation — it IS the value.
// The one place where today's code differs is the outward widening: `next_up`
// maps `0.0` to `5e-324` (its smallest-subnormal branch), so the shipped lane
// currently sprinkles subnormal dust across the provably-zero region and then
// carries it. Dropping that dust makes `D` equal to its EXACT value rather than
// a subnormal over-estimate of it, so the tableau stays a valid outward
// enclosure — `D` is defined as an upper bound on the coefficient error, and
// the error is exactly zero there. It is a change of at most ~1e-300 against
// accumulators holding ~1e-1 with a ~1e-12 certified `gam` slack, i.e. below
// the ulp of every sum it enters; the root bound is expected to print
// identically, and that identity is the acceptance test for this change.
//
// Kill switch: `NY_MARGIN_ROW_SUPPORT_MASK=0` restores the dense passes.

/// Number of equal column blocks tracked per tableau row (one `u64` of bits).
///
/// 64 is not a tuning parameter with headroom: replaying the real gather
/// geometry of `TinyImageNet_resnet_medium` gives 64.4% mean occupancy at 64
/// blocks and 62.5% at 128, against a ~60% floor for exact per-coefficient
/// support. One machine word already captures essentially all of the available
/// work elimination.
const SUPPORT_BLOCKS: usize = 64;

/// Column-block width for a tableau of `naug` columns.
#[inline]
const fn support_blk(naug: usize) -> usize {
    naug.div_ceil(SUPPORT_BLOCKS)
}

/// Is block `b` live in `mask`?
#[inline]
const fn support_has(mask: u64, b: usize) -> bool {
    mask & (1u64 << b) != 0
}

/// Mask with only the bias column's block set (every op writes there).
#[inline]
const fn support_bias_bit(n_in: usize, blk: usize) -> u64 {
    1u64 << (n_in / blk)
}

/// Per-row masks of the identity input tableau: row `i` holds `1.0` at column
/// `i` and nothing else; the bias block is retained so downstream bias folds
/// are always in range.
fn support_input(n_in: usize, blk: usize) -> Vec<u64> {
    let bias = support_bias_bit(n_in, blk);
    (0..n_in).map(|i| (1u64 << (i / blk)) | bias).collect()
}

/// Per-out-row masks of a conv: the union of the gathered source rows' masks.
/// All output channels at one spatial position gather the same rows, so the
/// union is computed once per position and broadcast.
fn support_conv(c: &ConvOp, src: &[u64], n_out: usize, bias: u64) -> Vec<u64> {
    let p = c.oshape.1 * c.oshape.2;
    let k = c.k_fwd;
    let mut per_sp = vec![bias; p];
    per_sp.par_iter_mut().enumerate().for_each(|(sp, m)| {
        for t in 0..k {
            let gi = c.gather[t * p + sp];
            if gi != usize::MAX {
                *m |= src[gi];
            }
        }
    });
    (0..n_out).map(|j| per_sp[j % p]).collect()
}

/// Is the support-mask work elimination armed? Default ON; exact `0` disarms.
fn support_mask_enabled() -> bool {
    !matches!(
        std::env::var("NY_MARGIN_ROW_SUPPORT_MASK").as_deref(),
        Ok("0")
    )
}

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
    /// Lower-line slope per neuron, in `[0, 1]`.
    ///
    /// Born binary from the area heuristic (`gates_from_box`: `[u >= -l]`);
    /// `alpha_opt` (#alpha-opt, env-gated, default OFF) may move UNSTABLE
    /// neurons to fractional values. Any `alpha in [0, 1]` is a valid DeepPoly
    /// lower line (`alpha*y <= relu(y)` pointwise on all of R), and the
    /// backward engine's certified error carry contracts by
    /// [`Self::ms`] `= max(alpha, s)` — whoever writes `alpha` MUST re-derive `ms` or the
    /// carried error is under-scaled (false-UNSAT risk). Stable and
    /// split-baked neurons always keep their exact fixed lines.
    pub alpha: Vec<f64>,
    /// Upper-line slope per neuron.
    pub s: Vec<f64>,
    /// Upper-line intercept per neuron (>= 0).
    pub c: Vec<f64>,
    /// `max(alpha, s)` per neuron (error-carry contraction factor).
    pub ms: Vec<f64>,
    /// Unstable neuron indices (l < 0 < u).
    pub unst: Vec<usize>,
    /// #clip-rows: INPUT-RELATIVE affine rows for the UNSTABLE neurons only,
    /// `(unst.len(), n_in)` midpoint coefficients and matching deviation.
    ///
    /// These are the halfspace coefficients Clip-and-Verify needs: a split
    /// `z_k,i(x) <= 0` plus this row gives an input-space constraint that
    /// over-approximates the subdomain (a NECESSARY condition, hence the safe
    /// direction), which lets every OTHER unstable neuron be re-concretized over
    /// `box INTERSECT halfspaces`. Without it a split tightens only its own
    /// neuron, children never close, and the frontier explodes -- the measured
    /// cause of cifar100's timeouts (idx_8600 open domains 18 -> 415 at depth 30
    /// while idx_6659 drains 26 -> 4 and proves).
    ///
    /// The root forward pass ALREADY computes these (`mi`/`di` at each ReLU's
    /// input) and previously discarded them, so this is retention, not new work.
    /// `None` when unretained, which is byte-identical to the historical struct
    /// for every existing consumer.
    ///
    /// FRAME: input-relative, `n_in` columns. NOT the tier-0 capture, which is
    /// output-relative — substituting it is the frame error that quarantined the
    /// sequential clip and is a false-`unsat` generator.
    pub clip_rows: Option<ClipRows>,
}

/// #clip-rows: retained input-relative affine rows for one layer's unstable
/// neurons. See [`LayerGates::clip_rows`].
#[derive(Debug, Clone)]
pub struct ClipRows {
    /// Midpoint coefficients, `(unst.len(), n_in)`.
    pub m: Array2<f64>,
    /// Deviation, same shape.
    pub d: Array2<f64>,
    /// Certified DOWNWARD slack of the lower line, one per retained row.
    ///
    /// `concretize_box` does not stop at the box-minimum of the `m - d` line: it
    /// subtracts `gam * (tabs + |bl| + |bu|)` plus, on the f32 fast path, a
    /// per-neuron error term `gerr[j]` that GROWS WITH DEPTH (measured on
    /// cifar100 idx_8600: 2.4e-5 at layer 0, 2.9e-3 at layer 2). So the line
    /// bounds the neuron pointwise only up to that error, and any halfspace
    /// derived from it must pay the same slack or it cuts into the true
    /// subdomain — a false-`unsat` generator.
    ///
    /// Stored as the gap between the line's own box-minimum and the bound the
    /// lane actually published, which needs no re-derivation of `gam`, `tabs`
    /// or `gerr` and is therefore exact by construction.
    pub sl_lo: Vec<f64>,
    /// Certified UPWARD slack of the upper line, one per retained row.
    pub sl_up: Vec<f64>,
    /// Midpoint CONSTANT term, one per retained row.
    ///
    /// The forward tableau is AUGMENTED (`naug = n_in + 1`) and
    /// `concretize_box` reads the bias out of the last column as
    /// `bl = m[n_in] - d[n_in]`, `bu = m[n_in] + d[n_in]`. Dropping it would
    /// make every halfspace pass through the origin, which is simply a
    /// different (and wrong) constraint.
    pub bm: Vec<f64>,
    /// Deviation of the constant term.
    pub bd: Vec<f64>,
}

/// Root state shared by every pass: input box + frozen gates.
/// #clip-rows: retain input-relative affine rows at each trunk ReLU?
///
/// Default OFF, so `LayerGates::clip_rows` is `None` and the struct is
/// byte-identical to its history for every existing consumer. Retention is pure
/// memory — nothing reads these rows yet.
fn clip_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS)
            .value
            .as_bool()
    })
}

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

/// Is the SOUND f32 root-tableau conv fast path requested? Default OFF — the
/// bit-for-bit f64 lane. When on, the two bandwidth-bound forward-conv lanes (M
/// and D) run in f32 and a certified additive concretize slack (accumulated per
/// op) dominates the worst-case effect of that f32 rounding on every box
/// endpoint. Pure loosening: `dj` can only shrink, never a false-UNSAT
/// (moat-safe).
///
/// Resolution order — the env var is authoritative in BOTH directions wherever
/// it is present, so a sealed A/B can pin either arm regardless of the preset:
///   * `NY_MARGIN_ROW_ROOT_F32=1|true|on`  => f32 root, whatever the preset says;
///   * `NY_MARGIN_ROW_ROOT_F32=<anything else>` => f64 root (kill switch);
///   * absent => [`super::root_f32_preset`], the typed `margin_row.root_f32`
///     route the CLI sets once from the category preset.
///
/// Neither route can make the tableau TIGHTER, so an accidental arming costs
/// proofs and can never manufacture one.
fn root_f32_requested() -> bool {
    match std::env::var("NY_MARGIN_ROW_ROOT_F32") {
        Ok(raw) => matches!(raw.trim(), "1" | "true" | "on"),
        Err(_) => super::root_f32_preset(),
    }
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
        // `bi: None`: the precision differential/enclosure oracles compare the
        // f32 and f64 tableau lanes in isolation; the (env-gated) backward
        // intermediate phase is exercised by its own tests instead.
        Ok(Self::build_retaining_inner(net, lo, hi, mode, deadline, None, &[], use_f32, None)?.0)
    }

    /// Test entry with an EXPLICIT backward-intermediate config
    /// (#backward-interm), bypassing the `NY_MARGIN_ROW_BACKWARD_INTERM` env
    /// gate so unit tests are order-independent of process environment.
    #[cfg(test)]
    pub(crate) fn build_retaining_bi(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        bi: Option<super::backward_interm::BiCfg>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::build_retaining_inner(net, lo, hi, mode, deadline, None, &[], false, bi)
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
            super::backward_interm::from_env(mode),
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
        bi: Option<super::backward_interm::BiCfg>,
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
        // #tableau-support-mask. Armed only alongside the f32 kernel (the only
        // conv lane that reads it) and only when the kill switch is absent.
        // Empty vectors elsewhere, which every consumer reads as "dense".
        let blk = support_blk(naug);
        let bias_bit = support_bias_bit(n_in, blk);
        let use_mask = use_f32 && support_mask_enabled();
        let s0: Vec<u64> = if use_mask {
            support_input(n_in, blk)
        } else {
            Vec::new()
        };
        let mut state: Vec<Option<(Array2<f64>, Array2<f64>, Vec<f64>, Vec<u64>)>> =
            vec![None; net.ops.len() + 1];
        state[0] = Some((m0, d0, g0, s0));
        let mut layers = Vec::new();
        // #backward-interm: only on the ROOT build (`splits.is_empty()`) —
        // Tier-2 epoch rebuilds must not re-spend the budget mid-search — and
        // only in outward mode (enforced by `from_env`; the test entry passes
        // an explicit config). `None` ⇒ byte-identical build.
        let mut bi_state = if splits.is_empty() && mode.outward() {
            bi.map(|cfg| super::backward_interm::BiState::new(cfg, deadline))
        } else {
            None
        };

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
            let take =
                |st: &mut Vec<Option<(Array2<f64>, Array2<f64>, Vec<f64>, Vec<u64>)>>,
                 cons: &mut Vec<usize>,
                 id: usize|
                 -> Result<(Array2<f64>, Array2<f64>, Vec<f64>, Vec<u64>, bool)> {
                    let last = cons[id] == 1;
                    cons[id] -= 1;
                    let entry = st[id]
                        .as_ref()
                        .ok_or_else(|| NyError::InvalidSpec("margin_row: dead tensor".into()))?;
                    if last {
                        let owned = st[id].take().expect("checked above");
                        Ok((owned.0, owned.1, owned.2, owned.3, true))
                    } else {
                        Ok((
                            entry.0.clone(),
                            entry.1.clone(),
                            entry.2.clone(),
                            entry.3.clone(),
                            false,
                        ))
                    }
                };
            match op {
                TwinOp::Conv(c) => {
                    let (mi, di, gerr_in, smask_in, _) = take(&mut state, &mut consumers, c.input)?;
                    let n_out = net.tsize[k + 1];
                    // #tableau-support-mask: the output row support is the union
                    // of the gathered source rows' supports (plus the bias column
                    // the folds below write into).
                    let smask_out: Vec<u64> = if use_mask {
                        support_conv(c, &smask_in, n_out, bias_bit)
                    } else {
                        Vec::new()
                    };
                    let cmask: Option<(&[u64], &[u64], usize)> = if use_mask {
                        Some((&smask_in, &smask_out, SUPPORT_BLOCKS))
                    } else {
                        None
                    };
                    let mut mo = Array2::<f64>::zeros((n_out, naug));
                    // M lane (f64 default, or f32 fast path stored back exactly).
                    conv_apply_forward_prec_masked(c, &mi, &mut mo, false, use_f32, cmask);
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
                        // Masked-out columns have `m == d == 0.0`, so the exact
                        // result is `0.0` and the zeroed `din` already holds it.
                        par_zip3_masked(&mut din, &mi, &di, &smask_in, blk, |dst, m, d| {
                            *dst = d + g * (m.abs() + d);
                        });
                        if use_f32 {
                            gerr_out = conv_f32_gerr(
                                c,
                                &mi,
                                &din,
                                &gerr_in,
                                &xabs,
                                s_xabs,
                                n_in,
                                if use_mask { Some(&smask_in) } else { None },
                                blk,
                            );
                        }
                        conv_apply_forward_prec_masked(c, &din, &mut do_, true, use_f32, cmask);
                        let g2 = gamma_n(c.k_fwd + 8);
                        // Same argument: `certify_up` of an exact `0.0` is `0.0`
                        // (its `next_up` would otherwise return 5e-324 dust).
                        map_masked(&mut do_, &smask_out, blk, |v| certify_up(v, g2));
                    } else {
                        conv_apply_forward_prec_masked(c, &di, &mut do_, true, use_f32, cmask);
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
                    state[k + 1] = Some((mo, do_, gerr_out, smask_out));
                }
                TwinOp::Add { lhs, rhs } => {
                    let (ma, da, ga, sa_, _) = take(&mut state, &mut consumers, *lhs)?;
                    let (mb, db, gb, sb_, _) = take(&mut state, &mut consumers, *rhs)?;
                    // Support of an elementwise sum is the union of the operands'.
                    let smask_out: Vec<u64> = if use_mask {
                        sa_.iter().zip(&sb_).map(|(a, b)| a | b).collect()
                    } else {
                        Vec::new()
                    };
                    let mut mo = ma;
                    mo += &mb;
                    let mut do_ = da;
                    if mode.outward() {
                        // do_ still holds da: dst = widen(da + db + 2u|mo|).
                        par_zip3_masked(&mut do_, &db, &mo, &smask_out, blk, |dst, dbv, mv| {
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
                    state[k + 1] = Some((mo, do_, gerr_out, smask_out));
                }
                TwinOp::Flatten { input } => {
                    let (mi, di, gi, si, _) = take(&mut state, &mut consumers, *input)?;
                    state[k + 1] = Some((mi, di, gi, si));
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
                    let (mut mi, mut di, gerr_in, smask_in, _) =
                        take(&mut state, &mut consumers, *input)?;
                    // Diagonal scaling is per-row elementwise, so the support is
                    // unchanged; the shift only touches the (always live) bias
                    // column.
                    let smask_out: Vec<u64> = if use_mask {
                        smask_in.iter().map(|m| m | bias_bit).collect()
                    } else {
                        Vec::new()
                    };
                    let g = next_up(*scale_rel_err + 4.0 * UNIT);
                    let msl = mi.as_slice_mut().expect("standard layout");
                    let dsl = di.as_slice_mut().expect("standard layout");
                    for (j, (&sj, &tj)) in scale.iter().zip(shift.iter()).enumerate() {
                        let sa = sj.abs();
                        let mrow = &mut msl[j * naug..(j + 1) * naug];
                        let drow = &mut dsl[j * naug..(j + 1) * naug];
                        let m = if use_mask { smask_in[j] } else { u64::MAX };
                        for b in 0..SUPPORT_BLOCKS {
                            if !support_has(m, b) {
                                continue;
                            }
                            let lo = (b * blk).min(naug);
                            let hi = (lo + blk).min(naug);
                            for i in lo..hi {
                                let m2 = sj * mrow[i];
                                if mode.outward() {
                                    drow[i] = next_up(
                                        (sa * drow[i] + g * (m2.abs() + sa * drow[i]))
                                            * (1.0 + 1e-15),
                                    );
                                } else {
                                    drow[i] *= sa;
                                }
                                mrow[i] = m2;
                            }
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
                    state[k + 1] = Some((mi, di, gerr_out, smask_out));
                }
                TwinOp::Relu { input, layer } => {
                    let (mi, di, gerr_in, smask_in, _) = take(&mut state, &mut consumers, *input)?;
                    let n = net.tsize[k + 1];
                    let f32_gerr = if use_f32 {
                        Some(gerr_in.as_slice())
                    } else {
                        None
                    };
                    let (mut l, mut u) = concretize_box(
                        &mi,
                        &di,
                        &mid,
                        &rad,
                        &xabs,
                        mode,
                        net.trunk_relus.len(),
                        f32_gerr,
                        if use_mask { Some(&smask_in) } else { None },
                        blk,
                    );
                    for v in l.iter().chain(u.iter()) {
                        if !v.is_finite() {
                            return Err(NyError::NumericalInstability(
                                "margin_row: non-finite tableau box".into(),
                            ));
                        }
                    }
                    // #backward-interm calibration capture. `clip_rows` slack
                    // MUST be recovered against the bounds THIS layer's own
                    // `m ± d` lines produced (the forward-only ones): after the
                    // shrink-only intersection below, `lo_min - l_published`
                    // under-recovers the line's certified slack (it can go
                    // negative and clamp to 0), and a halfspace paying too
                    // little slack cuts into the true subdomain — a
                    // false-`unsat` generator. Cloned only when both the
                    // tightener and retention are live.
                    let bi_cal =
                        (bi_state.is_some() && clip_rows_enabled()).then(|| (l.clone(), u.clone()));
                    if let Some(bi) = bi_state.as_mut() {
                        // Shrink-only intersection with the lane's own
                        // backward enclosure over the frozen prefix gates
                        // (soundness: module docs of `backward_interm`).
                        bi.tighten_layer(
                            net,
                            mode,
                            &mid,
                            &rad,
                            &xabs,
                            lo,
                            hi,
                            &mut layers,
                            k,
                            &mut l,
                            &mut u,
                        );
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
                    // #clip-rows: retain the INPUT-RELATIVE affine rows for the
                    // unstable neurons. `mi`/`di` are (n, n_in) and are exactly
                    // what `concretize_box` just consumed to produce `l`/`u`, so
                    // the frame is established by construction here rather than
                    // asserted later. Only `unst` rows are kept: a split can only
                    // constrain an unstable neuron, and keeping all of them would
                    // be (n x n_in) f64 per trunk layer.
                    // Calibration frame (#backward-interm): the FORWARD-only
                    // bounds when the tightener ran (see `bi_cal` above),
                    // otherwise the published ones — which then ARE the
                    // forward bounds, so the historical behavior is
                    // bit-identical.
                    let clip_rows = clip_rows_enabled().then(|| {
                        let (l_cal, u_cal) = bi_cal
                            .as_ref()
                            .map_or((l.as_slice(), u.as_slice()), |(lc, uc)| {
                                (lc.as_slice(), uc.as_slice())
                            });
                        retain_clip_rows(&mi, &di, &unst, &mid, &rad, l_cal, u_cal)
                    });
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
                        clip_rows,
                    });
                    debug_assert_eq!(layers.len() - 1, *layer);
                    if k == last_relu {
                        if timing {
                            op_times.push((k, "relu", n, op_t0.elapsed().as_secs_f64()));
                        }
                        break; // downstream tableau unused beyond retention
                    }
                    // Apply gates: L = (M-D)*alpha; U = (M+D)*s + c@bias.
                    // Support is unchanged (per-row elementwise) plus the bias
                    // column that `c[j]` writes into.
                    let smask_out: Vec<u64> = if use_mask {
                        smask_in.iter().map(|m| m | bias_bit).collect()
                    } else {
                        Vec::new()
                    };
                    let (mo, do_) = apply_gates(
                        &mi,
                        &di,
                        &alpha,
                        &s,
                        &c,
                        n_in,
                        mode,
                        if use_mask { Some(&smask_in) } else { None },
                        blk,
                    );
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
                    state[k + 1] = Some((mo, do_, gerr_out, smask_out));
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
        if let Some(bi) = bi_state.as_ref() {
            bi.finish();
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

/// #clip-rows retention body (extracted so the calibration frame is testable
/// in isolation — see the #backward-interm ordering trap below).
///
/// `mi`/`di` are the layer's pre-activation tableau rows (`(n, n_in + 1)`,
/// bias in the last column) exactly as `concretize_box` consumed them, so the
/// frame is established by construction. Only `unst` rows are kept.
///
/// CALIBRATION INVARIANT (load-bearing): `l_cal`/`u_cal` MUST be the bounds
/// these very `m ± d` lines produced through `concretize_box` — the
/// FORWARD-only bounds. `lo_min` is the exact box-minimum of the `m - d` line
/// and `l_cal[i]` is that minus the certified slack, so the difference IS the
/// slack, recovered without re-deriving it. Passing a TIGHTER published bound
/// (e.g. after the #backward-interm intersection) under-recovers the slack
/// (negative, clamped to 0), and every Clip-and-Verify halfspace built from
/// the line would then claim the line bounds `z` pointwise more tightly than
/// certified — cutting into the true subdomain, a false-`unsat` generator.
pub(crate) fn retain_clip_rows(
    mi: &Array2<f64>,
    di: &Array2<f64>,
    unst: &[usize],
    mid: &[f64],
    rad: &[f64],
    l_cal: &[f64],
    u_cal: &[f64],
) -> ClipRows {
    let n_in = mid.len();
    let mut m_sel = Array2::<f64>::zeros((unst.len(), n_in));
    let mut d_sel = Array2::<f64>::zeros((unst.len(), n_in));
    let mut bm = vec![0.0f64; unst.len()];
    let mut bd = vec![0.0f64; unst.len()];
    let mut sl_lo = vec![0.0f64; unst.len()];
    let mut sl_up = vec![0.0f64; unst.len()];
    for (r, &i) in unst.iter().enumerate() {
        for j in 0..n_in {
            m_sel[[r, j]] = mi[[i, j]];
            d_sel[[r, j]] = di[[i, j]];
        }
        // The augmented constant column. `mi` is
        // `(n, n_in + 1)`; column `n_in` is the bias.
        bm[r] = mi[[i, n_in]];
        bd[r] = di[[i, n_in]];
        let (mut lo_min, mut up_max) = (bm[r] - bd[r], bm[r] + bd[r]);
        for j in 0..n_in {
            let (lo_c, up_c) = (m_sel[[r, j]] - d_sel[[r, j]], m_sel[[r, j]] + d_sel[[r, j]]);
            lo_min += lo_c.mul_add(mid[j], -(lo_c.abs() * rad[j]));
            up_max += up_c.mul_add(mid[j], up_c.abs() * rad[j]);
        }
        sl_lo[r] = (lo_min - l_cal[i]).max(0.0);
        sl_up[r] = (u_cal[i] - up_max).max(0.0);
    }
    ClipRows {
        m: m_sel,
        d: d_sel,
        sl_lo,
        sl_up,
        bm,
        bd,
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

/// [`par_zip3`] restricted to the live column blocks of a
/// `#tableau-support-mask` (empty mask ⇒ dense).
///
/// Every writer that uses this maps `(0, 0) -> 0` in exact arithmetic, and the
/// destination arrives zeroed, so a skipped block already holds the exact
/// result. See the module comment for the invariant.
fn par_zip3_masked(
    dst: &mut Array2<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
    smask: &[u64],
    blk: usize,
    f: impl Fn(&mut f64, f64, f64) + Sync,
) {
    if smask.is_empty() {
        par_zip3(dst, a, b, f);
        return;
    }
    let cols = dst.ncols();
    let ds = dst.as_slice_mut().expect("standard layout");
    let asl = a.as_slice().expect("standard layout");
    let bs = b.as_slice().expect("standard layout");
    ds.par_chunks_mut(cols)
        .zip(asl.par_chunks(cols).zip(bs.par_chunks(cols)))
        .zip(smask.par_iter())
        .for_each(|((d, (ar, br)), &m)| {
            for blk_i in 0..SUPPORT_BLOCKS {
                if !support_has(m, blk_i) {
                    continue;
                }
                let lo = (blk_i * blk).min(cols);
                let hi = (lo + blk).min(cols);
                for ((dv, &av), &bv) in d[lo..hi].iter_mut().zip(&ar[lo..hi]).zip(&br[lo..hi]) {
                    f(dv, av, bv);
                }
            }
        });
}

/// Elementwise map over the live column blocks of a `#tableau-support-mask`
/// (empty mask ⇒ dense `par_mapv_inplace`).
///
/// Used for the outward `certify_up` widening of the D lane: outside the mask
/// the value is an exact `0.0` whose certified upper bound is `0.0`, while
/// `certify_up(0.0, _)` would return `next_up(0.0) = 5e-324`.
fn map_masked(
    dst: &mut Array2<f64>,
    smask: &[u64],
    blk: usize,
    f: impl Fn(f64) -> f64 + Sync + Send,
) {
    if smask.is_empty() {
        dst.par_mapv_inplace(f);
        return;
    }
    let cols = dst.ncols();
    let ds = dst.as_slice_mut().expect("standard layout");
    ds.par_chunks_mut(cols)
        .zip(smask.par_iter())
        .for_each(|(row, &m)| {
            for blk_i in 0..SUPPORT_BLOCKS {
                if !support_has(m, blk_i) {
                    continue;
                }
                let lo = (blk_i * blk).min(cols);
                let hi = (lo + blk).min(cols);
                for v in row[lo..hi].iter_mut() {
                    *v = f(*v);
                }
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
#[allow(clippy::too_many_arguments)]
fn conv_f32_gerr(
    c: &ConvOp,
    mi: &Array2<f64>,
    din: &Array2<f64>,
    gerr_in: &[f64],
    xabs: &[f64],
    s_xabs: f64,
    n_in: usize,
    smask: Option<&[u64]>,
    blk: usize,
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
            // #tableau-support-mask: skipped blocks contribute an exact `0.0`.
            let mask = smask.map_or(u64::MAX, |s| s[t]);
            for b in 0..SUPPORT_BLOCKS {
                if !support_has(mask, b) {
                    continue;
                }
                let lo = (b * blk).min(n_in);
                let hi = (lo + blk).min(n_in);
                for i in lo..hi {
                    braw += (mrow[i].abs() + drow[i]) * xabs[i];
                }
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
    smask: Option<&[u64]>,
    blk: usize,
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
            // #tableau-support-mask: a masked-out block has `m == d == 0.0`, so
            // every one of the five accumulations above adds an exact `0.0`
            // there. Skipping it is bit-identical, not an approximation.
            let mask = smask.map_or(u64::MAX, |s| s[j]);
            for b in 0..SUPPORT_BLOCKS {
                if !support_has(mask, b) {
                    continue;
                }
                let lo = (b * blk).min(n_in);
                let hi = (lo + blk).min(n_in);
                for i in lo..hi {
                    let lo_c = mrow[i] - drow[i];
                    let up_c = mrow[i] + drow[i];
                    vl += lo_c * mid[i];
                    rl += lo_c.abs() * rad[i];
                    vu += up_c * mid[i];
                    ru += up_c.abs() * rad[i];
                    tabs += (lo_c.abs() + up_c.abs()) * xabs[i];
                }
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
#[allow(clippy::too_many_arguments)]
fn apply_gates(
    m: &Array2<f64>,
    d: &Array2<f64>,
    alpha: &[f64],
    s: &[f64],
    c: &[f64],
    n_in: usize,
    mode: RoundMode,
    smask: Option<&[u64]>,
    blk: usize,
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
            // #tableau-support-mask: outside the mask `msr[i] == dsr[i] == 0.0`,
            // so `lo_c == up_c == 0.0` and both outputs are EXACTLY zero — which
            // is what the freshly zeroed `mo`/`do_` already hold. (The dense lane
            // instead stores `next_up(0.0) = 5e-324` into `drow[i]`; dropping
            // that subnormal makes `D` equal its exact value, so the tableau
            // stays a valid outward enclosure.) The bias column is always live.
            let mask = smask.map_or(u64::MAX, |s| s[j]);
            for b in 0..SUPPORT_BLOCKS {
                if !support_has(mask, b) {
                    continue;
                }
                let lo = (b * blk).min(naug);
                let hi = (lo + blk).min(naug);
                for i in lo..hi {
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
            }
        });
    (mo, do_)
}

/// #clip-rows FRAME ORACLE.
///
/// The retained rows are the halfspace coefficients Clip-and-Verify will build
/// its constraints from. If they are in the wrong frame the constraints are
/// nonsense in a way that does NOT fail loudly — it produces a tighter-looking
/// bound over the wrong region, which is a false-`unsat` generator. That is
/// exactly how the sequential clip earned its quarantine (`domain/clip.rs`: the
/// old adapter read output-relative rows as input-relative, "wrong in both axes
/// at once").
///
/// So the frame is checked against something already trusted rather than
/// argued: `l`/`u` in `LayerGates` were produced by `concretize_box` from these
/// very rows, so re-concretizing a retained row over the root box MUST reproduce
/// that neuron's stored interval. Anything else means the selection or the
/// orientation is wrong.
///
/// Cheap enough to run on every retained layer, and it runs BEFORE any consumer
/// exists — which is the only useful time to catch this.
#[must_use]
pub fn clip_rows_frame_holds(gates: &RootGates, tol: f64) -> bool {
    for (li, layer) in gates.layers.iter().enumerate() {
        let Some(rows) = layer.clip_rows.as_ref() else {
            continue;
        };
        if rows.m.nrows() != layer.unst.len() || rows.m.ncols() != gates.mid.len() {
            eprintln!(
                "[clip-frame] layer {li}: shape {:?} against unst {} / n_in {}",
                rows.m.shape(),
                layer.unst.len(),
                gates.mid.len()
            );
            return false;
        }
        // The lines are CALIBRATED against the published bound at retention, so
        // exact reproduction is true by construction and would test nothing.
        // What is worth testing is that the calibration is SANE: a row in the
        // wrong frame (transposed, output-relative, bias-dropped) still yields
        // numbers, but its box-minimum misses the published bound by a wild
        // margin, so the recovered slack blows up. Certified slack is a small
        // fraction of the interval; a frame error is multiples of it.
        for (r, &i) in layer.unst.iter().enumerate() {
            let width = (layer.u[i] - layer.l[i]).abs().max(1e-12);
            let (sl, su) = (rows.sl_lo[r], rows.sl_up[r]);
            if !sl.is_finite() || !su.is_finite() || sl < 0.0 || su < 0.0 {
                eprintln!("[clip-frame] layer {li} row {r}: bad slack {sl:.3e}/{su:.3e}");
                return false;
            }
            if sl > tol * width || su > tol * width {
                eprintln!(
                    "[clip-frame] layer {li} row {r} neuron {i}: slack {sl:.3e}/{su:.3e} \
against width {width:.3e} — frame is wrong, not merely rounded"
                );
                return false;
            }
        }
    }
    true
}
