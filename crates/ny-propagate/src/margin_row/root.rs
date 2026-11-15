// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Frozen root gates from a forward DeepPoly (M/D) tableau (#twinwall).
//!
//! Port of the verified Python reference (`core.py::RootTableau`): a forward
//! tableau `A_l = M - D`, `A_u = M + D` over augmented input rows, frozen
//! DeepPoly gates per trunk relu chosen from the concretized boxes. In
//! [`RoundMode::Outward`] every op charges a certified bound on its own
//! rounding (Higham `gamma_n` per conv, per-site bounds elsewhere) into the
//! `D` lane's BIAS column, where the sign is known, so the sandwich encloses
//! the stored network for every input sign (#root-rounding); the boxes are
//! concretized outward, and the upper gate lines are
//! REPAIRED so `s*y + c >= relu(y)` holds in real arithmetic on the certified
//! box (checked at the three kink points; `c` is bumped up on any deficit).
//!
//! The tableau stops after the LAST trunk relu: the backward engine owns
//! everything from there to the margin.
//!
//! #root-box-source (ENGINE_PROGRAM M5b): `margin_row.root_box_source:
//! patch_backward` replaces this tableau as the source of the trunk boxes with
//! a certified IBP reference plus a certified patch-backward pass over the
//! reference-unstable neurons (`root_box.rs`). The declaration default
//! `tableau` is this module's walk, unchanged; the dispatch is the one enum
//! compare at the top of `build_retaining_inner`.

use ndarray::Array2;
use ny_core::{NyError, Result};
use rayon::prelude::*;
use std::time::Instant;

use super::net::{
    conv_apply_forward, conv_apply_forward_prec_masked, conv_forward_f32_native, ConvOp, TwinNet,
    TwinOp,
};
use super::rounding::{
    certify_up, f32_store_slack, gamma_n, gamma_n_f32, next_down, next_up, RoundMode,
    F32_MIN_POSITIVE, SUBNORMAL_F32, UNIT, UNIT_F32,
};

/// Completion authority minted only after this original unsplit builder has
/// successfully constructed its actual gates. Sibling producers may consume
/// the type, but cannot certify an arbitrary supplied root with its constructor.
pub(super) struct CompletedOriginalRootAuthority(());

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
    /// carried error is under-scaled (false-UNSAT risk). The FORWARD f32
    /// `g_err` functional contracts by the same constant ([`gate_gerr_out`]).
    /// Stable and split-baked neurons always keep their exact fixed lines.
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
///
/// THE THIRD GATE. Retention happens ONCE, at root build, and every
/// Clip-and-Verify consumer bails out when it is off
/// (`clip::halfspaces_for_splits`, `clip::tighten_intermediate`,
/// `clip::any_rows_retained`). So a clip measurement taken without this armed
/// is VACUOUS, not negative — the mechanism reads as armed and refuses. It is
/// also why the closure-adaptive policy does NOT list clip among its rungs: a
/// mechanism whose prerequisite cannot be escalated mid-search would have to be
/// bought up front on every row, and the measured effect does not pay for it.
///
/// DELIVERABILITY. This was env-only, and `vnncomp_scripts/run_instance.sh`
/// exports exactly ONE `NY_*`, so on a scored run the third gate could never be
/// opened and the whole Clip-and-Verify family was unreachable BY CONSTRUCTION.
/// `margin_row.clip_rows` is the typed route; env still wins in both directions.
/// Read per call rather than latched: retention happens once per root build, and
/// a `OnceLock` would freeze whatever the FIRST build in the process saw —
/// before the vnncomp preset bridge had run, or before a unit test's env scope.
fn clip_rows_enabled() -> bool {
    crate::preset_gates::margin_row_clip_rows_enabled()
}

/// Exact execution evidence for one completed root-tableau precision walk.
///
/// Every count describes work actually completed before the terminal root was
/// published. Configuration is evidence-only; no bound consumer reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RootPrecisionExecutionEvidence {
    pub configured_root_f32: bool,
    pub configured_tableau_f32: bool,
    pub outward_eligible: bool,
    pub eligible_conv_ops: usize,
    pub root_f32_conv_ops: usize,
    pub root_f64_conv_ops: usize,
    pub eligible_store_ops: usize,
    pub tableau_f32_store_ops: usize,
    pub tableau_f64_store_ops: usize,
    pub tableau_refused: bool,
    pub policy_mismatch: bool,
}

impl RootPrecisionExecutionEvidence {
    /// `pub(super)` for `#root-box-source` (`root_box.rs`), whose
    /// patch-backward walk reports the tableau's precision work it did not run
    /// (zero ops, policy honoured); visibility only.
    pub(super) fn completed(
        policy: super::RootPrecisionRuntimePolicy,
        mode: RoundMode,
        actual_root_f32: bool,
        actual_tableau_f32: bool,
        conv_ops: usize,
        store_ops: usize,
    ) -> Self {
        let outward_eligible = mode.outward();
        let eligible_conv_ops = if outward_eligible { conv_ops } else { 0 };
        let eligible_store_ops = if outward_eligible { store_ops } else { 0 };
        let expected_root_f32 = policy.root_f32 && outward_eligible;
        let expected_tableau_f32 = policy.tableau_f32 && expected_root_f32;
        Self {
            configured_root_f32: policy.root_f32,
            configured_tableau_f32: policy.tableau_f32,
            outward_eligible,
            eligible_conv_ops,
            root_f32_conv_ops: if actual_root_f32 {
                eligible_conv_ops
            } else {
                0
            },
            root_f64_conv_ops: if actual_root_f32 {
                0
            } else {
                eligible_conv_ops
            },
            eligible_store_ops,
            tableau_f32_store_ops: if actual_tableau_f32 {
                eligible_store_ops
            } else {
                0
            },
            tableau_f64_store_ops: if actual_tableau_f32 {
                0
            } else {
                eligible_store_ops
            },
            tableau_refused: policy.tableau_f32 && !actual_tableau_f32,
            policy_mismatch: actual_root_f32 != expected_root_f32
                || actual_tableau_f32 != expected_tableau_f32,
        }
    }
}

#[cfg(test)]
mod root_precision_evidence_tests {
    use super::*;

    #[test]
    fn completed_evidence_partitions_exact_compute_and_storage_work() {
        let policy = super::super::RootPrecisionRuntimePolicy {
            root_f32: true,
            tableau_f32: true,
            err_lane: super::super::ErrLane::PerCol,
            ibp_seed: false,
            box_source: super::super::RootBoxSource::Tableau,
        };
        let armed = RootPrecisionExecutionEvidence::completed(
            policy,
            RoundMode::Outward,
            true,
            true,
            7,
            19,
        );
        assert_eq!(armed.eligible_conv_ops, 7);
        assert_eq!(armed.root_f32_conv_ops, 7);
        assert_eq!(armed.root_f64_conv_ops, 0);
        assert_eq!(armed.eligible_store_ops, 19);
        assert_eq!(armed.tableau_f32_store_ops, 19);
        assert_eq!(armed.tableau_f64_store_ops, 0);
        assert!(!armed.tableau_refused);

        let refused = RootPrecisionExecutionEvidence::completed(
            super::super::RootPrecisionRuntimePolicy {
                root_f32: false,
                tableau_f32: true,
                err_lane: super::super::ErrLane::PerCol,
                ibp_seed: false,
                box_source: super::super::RootBoxSource::Tableau,
            },
            RoundMode::Outward,
            false,
            false,
            7,
            19,
        );
        assert_eq!(refused.root_f64_conv_ops, 7);
        assert_eq!(refused.tableau_f64_store_ops, 19);
        assert!(refused.tableau_refused);

        let parity = RootPrecisionExecutionEvidence::completed(
            policy,
            RoundMode::Parity,
            false,
            false,
            7,
            19,
        );
        assert!(!parity.outward_eligible);
        assert_eq!(parity.eligible_conv_ops, 0);
        assert_eq!(parity.eligible_store_ops, 0);
        assert!(parity.tableau_refused);

        let mismatch = RootPrecisionExecutionEvidence::completed(
            policy,
            RoundMode::Outward,
            true,
            false,
            7,
            19,
        );
        assert!(mismatch.policy_mismatch);
        assert_eq!(mismatch.root_f32_conv_ops, 7);
        assert_eq!(mismatch.tableau_f64_store_ops, 19);
    }
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
    /// Frozen backward-intermediate configuration and binding counters from
    /// this root build.  It is evidence only and never feeds a bound.
    pub(crate) backward_interm_execution: super::backward_interm::BackwardIntermExecutionEvidence,
    /// Immutable root-precision authority carried into every descendant root.
    pub(crate) root_precision_policy: super::RootPrecisionRuntimePolicy,
    /// Frozen precision configuration and completed root-walk counters.
    pub(crate) root_precision_execution: RootPrecisionExecutionEvidence,
    /// #spec-alpha: per-START-NODE (spec) alpha policy, read at EXACTLY one
    /// site (`bab::root_eval_impl`) and installed at exactly one site
    /// (`margin_row/mod.rs`, after `#rebake` has finished with the root).
    ///
    /// `None` is the shipped state and makes the whole mechanism one `Option`
    /// discriminant test per instance. Deliberately NOT `Default`, so the
    /// compiler enumerates every `RootGates` literal -- including the epoch
    /// child root, which must carry `None` so the axis fires once per instance
    /// rather than once per epoch.
    pub spec_alpha_policy: Option<super::spec_alpha::SpecAlphaPolicy>,
    /// #head-alpha: lower slopes for the HEAD relu (the relu on the first head
    /// Gemm's output `y`), one per head neuron, installed by the root alpha
    /// ascent (`alpha_opt`). `None` is the shipped state: the binary area
    /// heuristic of `bounds::head_gates`, bit-identically.
    ///
    /// Consumed ONLY through `bounds::head_gates_with`, and only where the
    /// neuron is unstable in the y-box being evaluated (`ly < 0 < uy`); stable
    /// neurons keep their exact lines. Any slope in `[0, 1]` is a valid lower
    /// line `alpha * y <= relu(y)` for an unstable relu, whatever the box, so
    /// the same vector is sound at the root and on every tree domain.
    ///
    /// Deliberately NOT `Default`, like `spec_alpha_policy`: every literal must
    /// decide whether it carries the parent's head alpha.
    pub head_alpha: Option<Vec<f64>>,
    /// #root-eval-reuse (margin-row throughput plan B1(c)): stored
    /// `bab::root_eval` results for THIS root, keyed on a content fingerprint
    /// of every field `root_eval` reads (the gates generation). Performance
    /// only — a hit is bit-identical to recomputing, a miss recomputes. Every
    /// literal starts it EMPTY: a derived root (epoch child, rebake
    /// candidate, interm scratch) never inherits another root's evaluation.
    pub(crate) root_eval_memo: super::bab::RootEvalMemo,
    /// Native unsplit outward construction provenance for the optional head
    /// pass. Scratch and split frames have no authority; never imported.
    pub(super) head_group_source: Option<super::head_group_source::RootSource>,
}

/// Retention policy for the Tier-0 tableau rows (#epoch-bab).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Unstable rows eligible before the configured per-layer truncation.
    /// Evidence-only: consumers never use this count to compute a bound or
    /// choose a candidate.
    pub eligible: usize,
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
            eligible: 0,
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

// --------------------------------------------------------------------------
//  #tableau-f32: f32 STORAGE for the forward-tableau walk
// --------------------------------------------------------------------------
//
// `root_f32` made the conv COMPUTE f32 but left every tensor STORED f64: the
// conv widened its f32 accumulator into an f64 dst, the next conv re-cast that
// f64 back into a fresh f32 scratch (a full extra read+write per lane per
// layer), and every elementwise pass (`din`, Add, ChannelAffine, gate
// application) streamed 8-byte lanes. On the bandwidth-bound tinyimagenet walk
// that is 2x the traffic the arithmetic needs. This lever stores the M and D
// lanes as `Array2<f32>` END-TO-END across the walk.
//
// SOUNDNESS. Every store falls in exactly one class:
//   (a) EXACT stores — conv M/D lane outputs: the value written today is
//       `f64::from(f32_acc)`; storing the accumulator directly is the same
//       number bit for bit (`conv_forward_f32_native`). Identity `m0`
//       (1.0/0.0) and `d0` zeros are f32-exact. Zero new error.
//   (b) DIRECTED-UP stores — every D-lane (radius-role) elementwise result and
//       `din`: stored via `ny_core::f64_to_f32_up`, so the stored radius only
//       GROWS.
//   (c) NEAREST stores — M-lane (center-role) elementwise results cannot be
//       direction-rounded without bias; the store perturbation bound
//       `f32_store_slack(stored)` is folded into the SAME-SITE D widening
//       term (`TabElem::fold_charge`), maintaining per stored coefficient
//       `d32 - d64 >= |m32 - m64|`: the stored f32 sandwich interval
//       `[m32 - d32, m32 + d32]` CONTAINS the f64-path interval — the pinned
//       per-site containment rule, oracle-tested against the f64 path at HEAD
//       (`tableau_f32_store_per_site_containment_oracle`).
//
// THE ENDPOINT ACCOUNT IS g_err, NOT THE D FOLD. Interval-widening of a
// coefficient is endpoint-monotone only on a nonneg input box; on a
// mixed-sign box (tinyimagenet's IS mixed-sign: mean-normalized pixels, ~34%
// negative lows) a D-up-rounded coefficient can TIGHTEN a concretized
// endpoint by O(UNIT_F32 * magnitude) — measured by the zoo containment gate
// on a symmetric box before this account existed. So ALL storage dust
// ((b) up-round movement + (c) nearest movement + its D fold) is ALSO charged
// into the per-neuron `g_err` functional — `g_err[j] >= sum_i (|dM_ji| +
// |dD_ji|) * xabs_ext_i` vs the f64 path, consumed by `concretize_box` as an
// additive slack on BOTH endpoints, which is sign-independent. The charge is
// FUSED into each elementwise pass (arithmetic only against the L2-resident
// `xabs_ext` vector, no extra memory pass) and is the EXACT per-site
// deviation: each site recomputes the f64 site expression on its own inputs
// (a handful of flops it already has in registers — bit-the-same value the
// f64 lane stores) and accumulates `(|m32 - m64| + |d32 - d64|) * xabs_ext_i`
// per row, widened outward by `1 + 2*gamma_n(naug + 8)` for the
// accumulation's own f64 rounding. (The first cut charged the WORST-CASE
// magnitude bound `4*UNIT_F32*(|m32|+|d32|) + 4*F32_MIN_POSITIVE` per
// coefficient instead — sound, but the aggregate conservatism cost
// knife-edge rows what the floor saving bought and FAILED the 2026-08-20
// ship gate; `rounding::f32_store_gerr_term` survives only as the
// regression-pin yardstick.) Dust that routes through a conv (`din`
// up-store, D-lane `certify_up` re-store) is folded into `conv_f32_gerr`'s
// envelope (`store_dust` arm) against the D-lane-only functional `Bd[t]`.
// Each store site is charged EXACTLY ONCE: elementwise sites in their own
// fused row charge, conv-routed din/do_ stores in `conv_f32_gerr`, conv M/D
// main outputs are class (a) exact (no charge) — audited 2026-08-20, no
// double-charge existed; the defect was worst-case-vs-exact magnitude.
// Reading a stored f32 back as f64 is EXACT, so READS
// need no charge; the conv's `gamma_n_f32(k_fwd + 8)` budget for the deleted
// input cast becomes a sound over-count.
//
// SCOPE. The walk state (`mi`/`di` and the inter-layer scratch) and the
// blocked-conv dst/scratch in `net.rs`. NOT `engine.rs` backward (packs stay
// f64), NOT `gpu_backward`, NOT `clip_rows` (retention happens ONCE: the
// retained rows are built by EXACT f64 widening of the stored f32 rows, so
// `retain_clip_rows`' calibration invariant — slack recovered against the
// bounds these very rows produced through `concretize_box` — holds verbatim
// and automatically includes the storage charges). `build_rebaked` and the
// epoch Tier-2 rebuilds run through the same walk and inherit the lever.

/// Element type of the forward-tableau walk state (`#tableau-f32`).
///
/// `f64` is the default lane — every role method is the identity and each
/// call site's `if Self::F32_STORE` arm folds away at compile time, so the
/// stored values are bit-identical to the historical monomorphic code.
/// `f32` is the storage lane (active only when `root_f32` compute is armed and
/// the mode is `Outward`); its role methods implement the (a)/(b)/(c) store
/// classes documented on the `#tableau-f32` block above.
pub(crate) trait TabElem: Copy + Send + Sync + 'static {
    /// `true` on the f32 storage lane.
    const F32_STORE: bool;
    /// Additive zero (exact in both widths).
    const ZERO: Self;
    /// Identity-tableau one (exact in both widths).
    const ONE: Self;
    /// EXACT widening read (`f64` identity / `f32 -> f64`, no rounding).
    fn wide(self) -> f64;
    /// Class (c) store of a CENTER-role value: nearest store plus the
    /// certified perturbation bound to fold into the same-site D widening
    /// (`0.0` and unused on the f64 lane).
    fn store_center(v: f64) -> (Self, f64);
    /// Class (b) store of a RADIUS-role value: identity on f64, directed-UP
    /// f32 store (`ny_core::f64_to_f32_up`) on the storage lane.
    fn store_radius_up(v: f64) -> Self;
    /// Fold a class-(c) store charge into a same-site D widening expression.
    /// The f64 arm returns `base` UNTOUCHED (not `base + 0.0`) so the default
    /// lane's expressions keep their exact historical shape.
    #[inline]
    fn fold_charge(base: f64, charge: f64) -> f64 {
        if Self::F32_STORE {
            base + charge
        } else {
            base
        }
    }
    /// Forward conv into this element type (class (a): both lanes publish the
    /// identical accumulator value; the f32 lane skips the cast pass).
    fn conv_forward_masked(
        c: &ConvOp,
        src: &Array2<Self>,
        dst: &mut Array2<Self>,
        abs_w: bool,
        use_f32: bool,
        mask: Option<(&[u64], &[u64], usize)>,
    );
}

impl TabElem for f64 {
    const F32_STORE: bool = false;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    #[inline]
    fn wide(self) -> f64 {
        self
    }
    #[inline]
    fn store_center(v: f64) -> (f64, f64) {
        (v, 0.0)
    }
    #[inline]
    fn store_radius_up(v: f64) -> f64 {
        v
    }
    #[inline]
    fn conv_forward_masked(
        c: &ConvOp,
        src: &Array2<f64>,
        dst: &mut Array2<f64>,
        abs_w: bool,
        use_f32: bool,
        mask: Option<(&[u64], &[u64], usize)>,
    ) {
        conv_apply_forward_prec_masked(c, src, dst, abs_w, use_f32, mask);
    }
}

impl TabElem for f32 {
    const F32_STORE: bool = true;
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    #[inline]
    fn wide(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn store_center(v: f64) -> (f32, f64) {
        #[allow(clippy::cast_possible_truncation)]
        let stored = v as f32; // nearest
        (stored, f32_store_slack(stored))
    }
    #[inline]
    fn store_radius_up(v: f64) -> f32 {
        ny_core::f64_to_f32_up(v)
    }
    #[inline]
    fn conv_forward_masked(
        c: &ConvOp,
        src: &Array2<f32>,
        dst: &mut Array2<f32>,
        abs_w: bool,
        use_f32: bool,
        mask: Option<(&[u64], &[u64], usize)>,
    ) {
        debug_assert!(use_f32, "f32 storage lane requires the f32 compute lane");
        conv_forward_f32_native(c, src, dst, abs_w, mask);
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
        Ok(Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            None,
            &[],
            super::RootPrecisionRuntimePolicy {
                root_f32: use_f32,
                tableau_f32: false,
                err_lane: super::ErrLane::PerCol,
                ibp_seed: false,
                // #root-box-source: the precision oracles compare TABLEAU lanes.
                box_source: super::RootBoxSource::Tableau,
            },
            super::BackwardIntermRuntimePolicy::default(),
            None,
        )?
        .0)
    }

    /// [`Self::build_prec`] with an additional EXPLICIT `#tableau-f32` storage
    /// override (bypassing the `NY_MARGIN_ROW_TABLEAU_F32` env gate; used by
    /// the storage-lane oracles). `store_f32` is only effective when `use_f32`
    /// is set and the mode is [`RoundMode::Outward`]; otherwise it is refused
    /// exactly like the env route (loud `[tableau-f32] INERT` line, f64
    /// storage).
    pub fn build_prec_store(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        use_f32: bool,
        store_f32: bool,
    ) -> Result<Self> {
        Ok(Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            None,
            &[],
            super::RootPrecisionRuntimePolicy {
                root_f32: use_f32,
                tableau_f32: store_f32,
                err_lane: super::ErrLane::PerCol,
                ibp_seed: false,
                // #root-box-source: the storage oracles compare TABLEAU lanes.
                box_source: super::RootBoxSource::Tableau,
            },
            super::BackwardIntermRuntimePolicy::default(),
            None,
        )?
        .0)
    }

    /// Test entry with EXPLICIT f32 compute/storage flags AND Tier-0 retention
    /// (`#tableau-f32` oracles), bypassing every env gate so unit tests are
    /// order-independent of process environment.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_retaining_prec_store(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        use_f32: bool,
        store_f32: bool,
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            &[],
            super::RootPrecisionRuntimePolicy {
                root_f32: use_f32,
                tableau_f32: store_f32,
                err_lane: super::ErrLane::PerCol,
                ibp_seed: false,
                // #root-box-source: the storage oracles compare TABLEAU lanes.
                box_source: super::RootBoxSource::Tableau,
            },
            super::BackwardIntermRuntimePolicy::default(),
            None,
        )
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
        Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            None,
            &[],
            super::RootPrecisionRuntimePolicy::default(),
            super::BackwardIntermRuntimePolicy::from_config(bi),
            None,
        )
    }

    /// #mr-ibp-seed: build with an EXPLICIT seed flag, bypassing the preset /
    /// env gate (`margin_row.ibp_seed`), on the default f64 lane with no
    /// backward-interm phase and no retention. The containment oracles' entry:
    /// the unit tests here and the real-ONNX gates in ny-cli, which must arm
    /// and disarm the seed deterministically inside one process. `ibp_seed:
    /// false` is the dark lane, bit-identical to [`Self::build_prec`] with
    /// `use_f32 = false`.
    pub fn build_ibp_seeded(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        ibp_seed: bool,
    ) -> Result<Self> {
        Ok(Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            None,
            &[],
            super::RootPrecisionRuntimePolicy {
                ibp_seed,
                ..super::RootPrecisionRuntimePolicy::default()
            },
            super::BackwardIntermRuntimePolicy::default(),
            None,
        )?
        .0)
    }

    /// Test entry with an EXPLICIT precision policy (the `#mr-ibp-seed` flag
    /// included) and an EXPLICIT backward-interm config, bypassing every env
    /// gate so unit tests are order-independent of process environment.
    #[cfg(test)]
    pub(crate) fn build_with_policy_for_test(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        precision_policy: super::RootPrecisionRuntimePolicy,
        bi: Option<super::backward_interm::BiCfg>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            None,
            None,
            &[],
            precision_policy,
            super::BackwardIntermRuntimePolicy::from_config(bi),
            None,
        )
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
            super::RootPrecisionRuntimePolicy::from_layered_process(),
            super::BackwardIntermRuntimePolicy::from_layered_process(),
            None,
        )
    }

    /// Scored root build with the immutable policy derived from the run's
    /// receipt-owning [`ny_levers::LeverSet`].
    pub(crate) fn build_retaining_with_policy(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        precision_policy: super::RootPrecisionRuntimePolicy,
        backward_interm_policy: super::BackwardIntermRuntimePolicy,
        alpha_seed: Option<&[Vec<f64>]>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            splits,
            precision_policy,
            backward_interm_policy,
            alpha_seed,
        )
    }

    /// #rebake: rebuild the forward tableau with a SEEDED lower-slope vector —
    /// the joint-forward-re-bake round body (`rebake.rs`).
    ///
    /// `alpha_seed[li][j]` (one vector per trunk relu layer, layer width) is
    /// baked in place of the `gates_from_box` heuristic wherever neuron `j`
    /// is UNSTABLE in THIS build; stable and split-baked neurons keep their
    /// exact fixed lines, out-of-range / non-finite / shape-mismatched seed
    /// entries are ignored (fail-open to the heuristic). Any seeded value is
    /// clamped to `[0, 1]`, so every baked lower line is a valid DeepPoly
    /// relaxation and the build's own certificates (`repair_upper_lines`,
    /// `ms = max(alpha, s)`, the fractional-alpha `apply_gates` widening)
    /// apply verbatim — a rebaked build is certified by the SAME construction
    /// as a plain one, for any seed whatsoever.
    ///
    /// `bi` is an EXPLICIT backward-interm config (the caller re-budgets it
    /// per round); `None` skips the phase. `precision_policy` is the initial
    /// root's frozen authority, so a rebake never reopens environment state.
    pub(crate) fn build_rebaked(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        alpha_seed: &[Vec<f64>],
        bi: Option<super::backward_interm::BiCfg>,
        precision_policy: super::RootPrecisionRuntimePolicy,
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::build_retaining_inner(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            &[],
            precision_policy,
            super::BackwardIntermRuntimePolicy::from_config(bi),
            Some(alpha_seed),
        )
    }

    /// Every root build's single entry: the box-source dispatch
    /// (`#root-box-source`, ENGINE_PROGRAM M5b), then the tableau build.
    ///
    /// `precision_policy.box_source` is `Tableau` (the declaration default) on
    /// every build no one armed: the branch below is then one enum compare and
    /// the build IS [`Self::build_tableau`] with the same arguments, bit for
    /// bit. Armed (`PatchBackward`), the certified patch-backward source
    /// (`root_box.rs`) builds the root instead, with two fail-closed exits
    /// that both run the unchanged tableau build with the same arguments:
    ///
    /// * a REFUSAL, before the source runs, when the build requests a tableau
    ///   product the source cannot produce (`#clip-rows` retention, Tier-0
    ///   retained rows, the `root_f32` / `tableau_f32` kernels —
    ///   [`super::root_box::PatchBackwardConflicts`]); it prints `[root-box]
    ///   refused reason=` instead of silently dropping the product;
    /// * a DECLINE (Parity mode, a non-finite reference box, two certified
    ///   enclosures that crossed), which prints why.
    #[allow(clippy::too_many_arguments)]
    fn build_retaining_inner(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        precision_policy: super::RootPrecisionRuntimePolicy,
        bi_policy: super::BackwardIntermRuntimePolicy,
        alpha_seed: Option<&[Vec<f64>]>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        // Qualification precedes either source backend and its worker
        // arithmetic. Attach the private generation only after a completed
        // successful build; the dark policy does no hashing or FP probing.
        // Patch and envelope producers retain separate error arithmetic.
        // They cannot supply the absolute-underflow-qualified source for the
        // joint head functional. A declined source affects only that optional
        // refinement and preserves the existing root build.
        let source = if precision_policy.box_source == super::RootBoxSource::Tableau
            && !matches!(
                precision_policy.err_lane,
                super::errlane::ErrLane::Envelope | super::errlane::ErrLane::PerChannel
            )
            && !(bi_policy.enabled && bi_policy.patch.cfg().is_some())
        {
            super::head_group_source::RootSource::begin(net, lo, hi, mode, splits)
        } else {
            None
        };
        let (mut root, retained) = Self::build_retaining_inner_body(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            splits,
            precision_policy,
            bi_policy,
            alpha_seed,
        )?;
        if let Some(source) = source.as_ref() {
            // New input-row consumers require the successful original gate
            // generation, not merely shape-valid subsequently supplied gates.
            // A failed bounded snapshot leaves those consumers unauthenticated
            // while preserving the existing optional head-source semantics.
            source.complete_input_rows(&root, deadline, CompletedOriginalRootAuthority(()));
        }
        root.head_group_source = source;
        Ok((root, retained))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_retaining_inner_body(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        precision_policy: super::RootPrecisionRuntimePolicy,
        bi_policy: super::BackwardIntermRuntimePolicy,
        alpha_seed: Option<&[Vec<f64>]>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        if precision_policy.box_source == super::RootBoxSource::PatchBackward {
            // Read exactly what the tableau walk below would read: the same
            // `#clip-rows` gate (per call, like the walk), the same `retain`,
            // the same precision policy.
            let conflicts = super::root_box::PatchBackwardConflicts {
                clip_rows: clip_rows_enabled(),
                retain: retain.is_some(),
                root_f32: precision_policy.root_f32,
                tableau_f32: precision_policy.tableau_f32,
            };
            if conflicts.any() {
                let tableau_t0 = Instant::now();
                let out = Self::build_tableau(
                    net,
                    lo,
                    hi,
                    mode,
                    deadline,
                    retain,
                    splits,
                    precision_policy,
                    bi_policy,
                    alpha_seed,
                );
                super::root_box::report_refused(
                    conflicts,
                    tableau_t0.elapsed().as_secs_f64(),
                    out.is_ok(),
                );
                return out;
            }
            let declined = match super::root_box::try_build(
                net,
                lo,
                hi,
                mode,
                deadline,
                retain,
                splits,
                precision_policy,
                bi_policy,
                alpha_seed,
            ) {
                super::root_box::Attempt::Built(out) => return *out,
                super::root_box::Attempt::Declined(declined) => declined,
            };
            let tableau_t0 = Instant::now();
            let out = Self::build_tableau(
                net,
                lo,
                hi,
                mode,
                deadline,
                retain,
                splits,
                precision_policy,
                bi_policy,
                alpha_seed,
            );
            super::root_box::report_declined(
                &declined,
                tableau_t0.elapsed().as_secs_f64(),
                out.is_ok(),
            );
            return out;
        }
        Self::build_tableau(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            splits,
            precision_policy,
            bi_policy,
            alpha_seed,
        )
    }

    /// Full TABLEAU build implementation with an EXPLICIT f32-fast-path flag
    /// (`NY_MARGIN_ROW_ROOT_F32`) and an EXPLICIT `#tableau-f32` storage flag
    /// (`NY_MARGIN_ROW_TABLEAU_F32`). `use_f32` only takes effect in
    /// [`RoundMode::Outward`]; when off, every bound is bit-identical to the
    /// pure-f64 lane (epoch-bab retention/splits included). When on, the two
    /// bandwidth-bound forward-conv lanes (M and D) run in f32 and the
    /// per-tensor certified error accumulator `g_err[j]` is threaded through
    /// each op and consumed as a pure additive concretize slack. `store_f32`
    /// additionally stores the walk state in f32 (`#tableau-f32` block above);
    /// it is REFUSED — loudly — unless `use_f32` is effective.
    #[allow(clippy::too_many_arguments)]
    fn build_tableau(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        precision_policy: super::RootPrecisionRuntimePolicy,
        bi_policy: super::BackwardIntermRuntimePolicy,
        alpha_seed: Option<&[Vec<f64>]>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        // Source-aware compilation may admit a wide patch-only net. Every
        // parity/conflict/reference/crossing fallback still passes THIS guard
        // before any full forward tableau allocation; refusal leaves the
        // complete verifier fallback available with its remaining budget.
        if !net.root_tableau_within_cap() {
            return Err(NyError::InvalidSpec(
                "margin_row: root tableau exceeds original entry cap; patch-only fallback refused"
                    .into(),
            ));
        }
        // f32 lanes + the certified error slack are only meaningful in the
        // certified-outward verdict mode.
        let use_f32 = precision_policy.root_f32 && mode.outward();
        // #tableau-f32: storage narrowing is defined ONLY on the f32 compute
        // lane (its g_err machinery is what the store charges compose with).
        // Requested without it => refuse LOUDLY and run the f64 storage lane,
        // so a mis-armed A/B is readable in its own log (R9: no silent nulls).
        let store_f32 = precision_policy.tableau_f32
            && (use_f32 || {
                eprintln!(
                    "[tableau-f32] INERT: refused — f32 tableau STORAGE needs the f32 \
                     compute lane armed and Outward mode (margin_row.root_f32 preset / \
                     NY_MARGIN_ROW_ROOT_F32=1); storage stays f64"
                );
                false
            });
        if store_f32 {
            Self::build_walk::<f32>(
                net,
                lo,
                hi,
                mode,
                deadline,
                retain,
                splits,
                precision_policy,
                bi_policy,
                alpha_seed,
            )
        } else {
            Self::build_walk_f64(
                net,
                lo,
                hi,
                mode,
                deadline,
                retain,
                splits,
                use_f32,
                precision_policy,
                bi_policy,
                alpha_seed,
            )
        }
    }

    /// f64-stored walk (the default lane; `use_f32` selects the f32 COMPUTE
    /// kernels with f64 storage — the shipped `root_f32` route).
    #[allow(clippy::too_many_arguments)]
    fn build_walk_f64(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        use_f32: bool,
        precision_policy: super::RootPrecisionRuntimePolicy,
        bi_policy: super::BackwardIntermRuntimePolicy,
        alpha_seed: Option<&[Vec<f64>]>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::walk_ops::<f64>(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            splits,
            use_f32,
            precision_policy,
            bi_policy,
            alpha_seed,
        )
    }

    /// f32-stored walk (`#tableau-f32`; implies the f32 compute lane).
    #[allow(clippy::too_many_arguments)]
    fn build_walk<T: TabElem>(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        precision_policy: super::RootPrecisionRuntimePolicy,
        bi_policy: super::BackwardIntermRuntimePolicy,
        alpha_seed: Option<&[Vec<f64>]>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        Self::walk_ops::<T>(
            net,
            lo,
            hi,
            mode,
            deadline,
            retain,
            splits,
            true,
            precision_policy,
            bi_policy,
            alpha_seed,
        )
    }

    /// The forward-tableau walk, generic over the STORED element type
    /// (`#tableau-f32`). For `T = f64` every store helper is the identity and
    /// each `T::F32_STORE` branch folds away, so the emitted arithmetic — and
    /// therefore every published bit — is the historical monomorphic code's.
    #[allow(clippy::too_many_arguments)]
    fn walk_ops<T: TabElem>(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        deadline: Option<Instant>,
        retain: Option<&RetainCfg>,
        splits: &[(usize, usize, i8)],
        use_f32: bool,
        precision_policy: super::RootPrecisionRuntimePolicy,
        bi_policy: super::BackwardIntermRuntimePolicy,
        alpha_seed: Option<&[Vec<f64>]>,
    ) -> Result<(Self, Option<RetainedRows>)> {
        debug_assert!(
            !T::F32_STORE || (use_f32 && mode.outward()),
            "f32 storage lane requires the f32 compute lane in Outward mode"
        );
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
        // Augmented magnitude weights (bias column weight 1.0) for the
        // #tableau-f32 fused storage-dust charges; L2-resident, reused by
        // every elementwise pass on the storage lane.
        let xabs_ext: Vec<f64> = xabs.iter().copied().chain(std::iter::once(1.0)).collect();

        // Sum of the augmented input magnitude weights (xabs on the input
        // columns, 1.0 on the bias column). Constant across the tableau; scales
        // the per-conv f32 FTZ/subnormal absolute floor in the g_err accumulator.
        let s_xabs: f64 = next_up(xabs.iter().sum::<f64>() + 1.0);
        // #root-rounding: gradual-underflow floor of every sign-safe rounding
        // charge, and the TEST-ONLY switch back to the historical
        // coefficient-column charge (always `false` outside `cfg(test)`).
        let round_floor = root_round_floor(&xabs_ext);
        let legacy_round = legacy_rounding_forced();

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
        // #fold-ball: certified per-tensor magnitudes for the sign-safe relative
        // fold charge — `Some` only in Outward mode on a walk that folds a
        // nonzero `weight_rel_err` / `scale_rel_err` (see `FoldMag`); `None`
        // keeps every historical expression and value.
        let mut fold_mag = FoldMag::new(net, last_relu, mode, &consumers, &xabs);
        // #mr-ibp-seed: the certified interval state carried beside the tableau
        // (`ibp_seed.rs`). `None` unless `margin_row.ibp_seed` is armed on this
        // root's frozen precision policy — the dark walk is byte-identical and
        // silent. Armed but declined (residual trunk, Parity, RSS) it prints
        // `declined=<reason>` per layer and intersects nothing.
        let mut ibp = super::ibp_seed::IbpSeed::new(
            net,
            last_relu,
            mode,
            &consumers,
            lo,
            hi,
            precision_policy.ibp_seed,
        );

        // Identity tableau at the input (1.0/0.0 are exact in both widths).
        let mut m0 = Array2::<T>::from_elem((n_in, naug), T::ZERO);
        for i in 0..n_in {
            m0[[i, i]] = T::ONE;
        }
        let d0 = Array2::<T>::from_elem((n_in, naug), T::ZERO);
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
        let mut state: Vec<Option<(Array2<T>, Array2<T>, Vec<f64>, Vec<u64>)>> =
            vec![None; net.ops.len() + 1];
        state[0] = Some((m0, d0, g0, s0));
        let mut layers = Vec::new();
        // Logical operator sites whose precision-sensitive work completed.
        // Flatten aliases existing storage, while the terminal ReLU only
        // concretizes a box and deliberately does not publish another tableau.
        let mut completed_conv_ops = 0usize;
        let mut completed_store_ops = 0usize;
        // #epoch-carry: BI runs on the ROOT build and, when the caller asks for
        // it, inside a Tier-2 epoch rebuild too.
        //
        // The gate used to be `splits.is_empty()`, which excluded EVERY epoch
        // rebuild by construction — and that is why arming `NY_EPOCH_DEPTH`
        // measured as a bound REGRESSION: the epoch re-entered idx_4169 at
        // -0.389998 against its own parent's -0.371373 on a strict SUBSET of the
        // box, which is impossible for a sound tightening, because
        // `build_retaining` discarded 584 `#backward-interm`-tightened columns.
        //
        // The caller now decides, by handing in a policy whose `enabled` it
        // actually means. `try_epoch` passed `BackwardIntermRuntimePolicy::default()`
        // (enabled: false), so lifting this gate alone would have changed nothing
        // — and would have reported `enabled=true` with every counter at zero,
        // which reads as armed while running nothing. Both had to move together.
        // NOTE the disjunction. On the ROOT (`splits.is_empty()`) the state is
        // created even when the policy is DISABLED, because the terminal receipt
        // must be able to prove a kill-switch suppressed an otherwise eligible
        // phase without running a pass — dropping that would destroy the
        // evidence path. Inside an epoch the state is created only when the
        // caller actually armed BI.
        // MEASURED 2026-08-30 (docs/EPOCH_CARRY_2026-08-30.md): re-running BI
        // inside an epoch is UNAFFORDABLE. At depth 4 — the only trigger depth
        // these rows reach — one epoch attempt consumed the whole remaining wall
        // and cost the outer tree ~160 expansions (271 -> 111), with closure
        // still 0.000; at depth >= 8 the epoch never fires at all. The original
        // guard ("Tier-2 epoch rebuilds must not re-spend the budget mid-search")
        // was a COST decision, not an oversight, and it is restored here.
        //
        // The cheap half of #epoch-carry is KEPT: `try_epoch` still seeds the
        // rebuild with the root's optimised alpha, so an epoch no longer
        // re-derives the `gates_from_box` area heuristic.
        let mut bi_state = if splits.is_empty() && mode.outward() {
            Some(
                super::backward_interm::BiState::new(bi_policy, deadline)
                    .with_err_lane(precision_policy.err_lane),
            )
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
                    if T::F32_STORE {
                        // R9: the lever's own observable, even on a truncated build.
                        eprintln!(
                            "[tableau-f32] armed: build DEADLINE-truncated at op {k} \
                             after {:.2}s",
                            build_t0.elapsed().as_secs_f64()
                        );
                    }
                    return Err(NyError::DeadlineExceeded(
                        "margin_row root tableau deadline".into(),
                    ));
                }
            }
            let op_t0 = Instant::now();
            // #mr-ibp-seed: advance the interval state over this op (a ReLU's
            // output is set later, from its FINAL published box).
            if let Some(seed) = ibp.as_mut() {
                seed.forward(net, k, op);
            }
            let take = |st: &mut Vec<Option<(Array2<T>, Array2<T>, Vec<f64>, Vec<u64>)>>,
                        cons: &mut Vec<usize>,
                        id: usize|
             -> Result<(Array2<T>, Array2<T>, Vec<f64>, Vec<u64>, bool)> {
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
                    let mut mo = Array2::<T>::from_elem((n_out, naug), T::ZERO);
                    // M lane (f64 default, f32 fast path stored back exactly, or
                    // #tableau-f32 native f32 store — all publish the same number
                    // for a given compute lane; store class (a), zero new error).
                    T::conv_forward_masked(c, &mi, &mut mo, false, use_f32, cmask);
                    let mut do_ = Array2::<T>::from_elem((n_out, naug), T::ZERO);
                    // g_err accumulation for the f32 fast path (see conv_f32_gerr
                    // doc): g_err_out = conv_abs(g_err_in + gamma_f32 * B) + FTZ
                    // floor, with B[t] = sum_i (|mi_ti| + |din_ti|) * xabs_ext_i
                    // (`din` is the D lane the conv consumes: `di` itself on the
                    // #root-rounding path, the widened one on the legacy path).
                    let mut gerr_out: Vec<f64> = Vec::new();
                    // #fold-ball: when the walk tracks certified magnitudes (some
                    // op carries a RELATIVE fold error), this conv's
                    // `weight_rel_err` is charged SIGN-SAFELY into the D BIAS
                    // column (`FoldMag` docs) and `rel_d` below is 0; otherwise
                    // every relative fold error is 0 (outside the TEST-ONLY
                    // legacy switch, which runs the whole pre-repair walk).
                    let conv_fold = match fold_mag.as_ref() {
                        Some(fm) => Some(conv_fold_ball(c, fm.get(c.input)?, n_out)),
                        None => None,
                    };
                    if mode.outward() && !legacy_round {
                        // #root-rounding: NO coefficient-column widening. The M
                        // and D lanes are the plain products `W M`, `|W| D`, and
                        // their rounding is charged SIGN-SAFELY into the D bias
                        // column (#root-rounding block comment):
                        // `E_o = g sum_t |W_ot| sum_i (|M_ti| + |D_ti|) xabs_ext_i`.
                        // On the f32 compute lane `conv_f32_gerr` already charges
                        // both lanes' error against exact arithmetic into `g_err`
                        // (the same sign-safe account), so `E` is added only on
                        // the f64 lane — or if a relative fold error reached this
                        // conv, which the walk's invariant forbids (`FoldMag` is
                        // `Some` whenever one exists outside the legacy switch)
                        // but which `g` would still charge soundly.
                        let rel_d = if conv_fold.is_some() {
                            0.0
                        } else {
                            c.weight_rel_err
                        };
                        let g =
                            next_up(gamma_n(c.k_fwd + 2) + rel_d + gamma_n(c.k_fwd + 2) * rel_d);
                        if use_f32 {
                            gerr_out = conv_f32_gerr(
                                c,
                                &mi,
                                &di,
                                &gerr_in,
                                &xabs,
                                s_xabs,
                                n_in,
                                if use_mask { Some(&smask_in) } else { None },
                                blk,
                                T::F32_STORE,
                            );
                        }
                        T::conv_forward_masked(c, &di, &mut do_, true, use_f32, cmask);
                        if !use_f32 || rel_d != 0.0 {
                            let e =
                                conv_round_charge(c, &mi, &di, &xabs, n_in, n_out, g, round_floor);
                            charge_bias_column(&mut do_, &e, n_in, &mut gerr_out);
                        }
                    } else if mode.outward() {
                        // HISTORICAL coefficient-column charge (TEST-ONLY legacy
                        // switch, `with_legacy_rounding_charge_for_test`).
                        // D input absorbs the M-product error: Din = D + g(|M| + D).
                        let rel_d = if conv_fold.is_some() {
                            0.0
                        } else {
                            c.weight_rel_err
                        };
                        let g =
                            next_up(gamma_n(c.k_fwd + 2) + rel_d + gamma_n(c.k_fwd + 2) * rel_d);
                        let mut din = Array2::<T>::from_elem(mi.raw_dim(), T::ZERO);
                        // Masked-out columns have `m == d == 0.0`, so the exact
                        // result is `0.0` and the zeroed `din` already holds it.
                        // #tableau-f32: `din` is RADIUS-role — stored directed-UP
                        // (class (b)); the up-rounded value feeds both consumers
                        // in their loosening direction (bigger B[t] in g_err,
                        // bigger |W|-gathered D output).
                        par_zip2_radius_masked(&mut din, &mi, &di, &smask_in, blk, |m, d| {
                            d + g * (m.abs() + d)
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
                                T::F32_STORE,
                            );
                        }
                        T::conv_forward_masked(c, &din, &mut do_, true, use_f32, cmask);
                        let g2 = gamma_n(c.k_fwd + 8);
                        // Same argument: `certify_up` of an exact `0.0` is `0.0`
                        // (its `next_up` would otherwise return 5e-324 dust).
                        // #tableau-f32: radius-role re-store is directed UP.
                        map_radius_masked(&mut do_, &smask_out, blk, |v| certify_up(v, g2));
                    } else {
                        T::conv_forward_masked(c, &di, &mut do_, true, use_f32, cmask);
                    }
                    // Bias into the M bias column (+ certified bias error into D).
                    // #tableau-f32: the bias fold is a CENTER-role nearest store
                    // (class (c)); its perturbation is charged into the same-site
                    // D bias widening (stored UP, class (b)) AND its storage
                    // dust into g_err (bias-column weight 1.0).
                    let p = c.oshape.1 * c.oshape.2;
                    // #fold-ball: the sign-safe relative-fold charge rides in the
                    // per-channel bias error (bias column, weight 1 > 0).
                    let bias_err: &[f64] = conv_fold
                        .as_ref()
                        .map_or(c.bias_err.as_slice(), |f| f.bias_err.as_slice());
                    conv_bias_fold(
                        &mut mo,
                        &mut do_,
                        &c.bias,
                        bias_err,
                        p,
                        n_in,
                        mode,
                        &mut gerr_out,
                    );
                    state[k + 1] = Some((mo, do_, gerr_out, smask_out));
                    if let (Some(fm), Some(f)) = (fold_mag.as_mut(), conv_fold) {
                        fm.set(k + 1, f.mag_out);
                        fm.release(c.input);
                    }
                    completed_conv_ops += 1;
                    completed_store_ops += 1;
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
                    // #tableau-f32: the sum is CENTER-role (nearest store, class
                    // (c), charge folded into the same-site D widening); the D
                    // result is RADIUS-role (directed-UP store, class (b)); the
                    // storage dust is charged into g_err (`gstore`, fused).
                    let (mo, do_, gstore) =
                        add_op(ma, &mb, da, &db, &smask_out, blk, mode, &xabs_ext);
                    // f32 error is additive across the two branches: the elementwise
                    // sum's coefficient errors add (its own f64 add rounding is a
                    // second-order effect on the errors, absorbed by next_up).
                    let gerr_out: Vec<f64> = if use_f32 {
                        if T::F32_STORE {
                            ga.iter()
                                .zip(&gb)
                                .zip(&gstore)
                                .map(|((a, b), gs)| next_up(next_up(a + b) + gs))
                                .collect()
                        } else {
                            // Bit-identical shipped expression.
                            ga.iter().zip(&gb).map(|(a, b)| next_up(a + b)).collect()
                        }
                    } else {
                        Vec::new()
                    };
                    state[k + 1] = Some((mo, do_, gerr_out, smask_out));
                    if let Some(fm) = fold_mag.as_mut() {
                        let sum = fold_mag_add(fm.get(*lhs)?, fm.get(*rhs)?);
                        fm.set(k + 1, sum);
                        fm.release(*lhs);
                        fm.release(*rhs);
                    }
                    completed_store_ops += 1;
                }
                TwinOp::Flatten { input } => {
                    let (mi, di, gi, si, _) = take(&mut state, &mut consumers, *input)?;
                    state[k + 1] = Some((mi, di, gi, si));
                    if let Some(fm) = fold_mag.as_mut() {
                        let same = fm.get(*input)?.to_vec();
                        fm.set(k + 1, same);
                        fm.release(*input);
                    }
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
                    // #fold-ball: with magnitudes tracked, the RELATIVE scale
                    // error moves out of the coefficient charge into the
                    // per-neuron shift error (bias column, sign-safe); without
                    // them every `scale_rel_err` is 0 and nothing changes.
                    let affine_fold = match fold_mag.as_ref() {
                        Some(fm) => Some(affine_fold_ball(
                            scale,
                            shift,
                            *scale_rel_err,
                            shift_err,
                            fm.get(*input)?,
                        )?),
                        None => None,
                    };
                    let (scale_rel_d, shift_err_d): (f64, &[f64]) = match affine_fold.as_ref() {
                        Some(f) => (0.0, f.shift_err.as_slice()),
                        None => (*scale_rel_err, shift_err.as_slice()),
                    };
                    let gstore = chaffine_op(
                        &mut mi,
                        &mut di,
                        scale,
                        shift,
                        scale_rel_d,
                        shift_err_d,
                        n_in,
                        mode,
                        if use_mask { Some(&smask_in) } else { None },
                        blk,
                        &xabs_ext,
                    );
                    // f32 error carries through the diagonal scale: the new
                    // coefficient errors are |dM'| + |dD'| <= |scale_j| * (|dM| +
                    // |dD|) (the shift enters only the bias column with no f32
                    // error; the affine's own f64 rounding on the error-carrying
                    // coefficients is second order, absorbed by next_up + 1e-15).
                    // #tableau-f32: plus this site's storage dust (`gstore`).
                    let gerr_out: Vec<f64> = if use_f32 {
                        if T::F32_STORE {
                            scale
                                .iter()
                                .zip(&gerr_in)
                                .zip(&gstore)
                                .map(|((&sj, &gj), gs)| {
                                    next_up(next_up(sj.abs() * gj * (1.0 + 1e-15)) + gs)
                                })
                                .collect()
                        } else {
                            // Bit-identical shipped expression.
                            scale
                                .iter()
                                .zip(&gerr_in)
                                .map(|(&sj, &gj)| next_up(sj.abs() * gj * (1.0 + 1e-15)))
                                .collect()
                        }
                    } else {
                        Vec::new()
                    };
                    state[k + 1] = Some((mi, di, gerr_out, smask_out));
                    if let (Some(fm), Some(f)) = (fold_mag.as_mut(), affine_fold) {
                        fm.set(k + 1, f.mag_out);
                        fm.release(*input);
                    }
                    completed_store_ops += 1;
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
                    // false-`unsat` generator. Cloned only when a tightener
                    // (#backward-interm or an ACTIVE #mr-ibp-seed) and retention
                    // are both live; dark, the condition is the historical
                    // `bi_policy.enabled && clip_rows_enabled()` verbatim.
                    //
                    // #mr-ibp-seed ORDERING (the reason this capture comes
                    // FIRST): the IBP intersection below is a second shrink-only
                    // tightener with exactly the trap described above, so it
                    // must run AFTER this capture, never before it. The
                    // TEST-ONLY mutation switch `with_bi_cal_after_seed_for_test`
                    // moves the capture after the seed so
                    // `ibp_seed_keeps_bi_cal_on_forward_only_lines` can prove it
                    // detects that defect; it is a constant `false` outside
                    // `cfg(test)`.
                    let seed_live = ibp.as_ref().is_some_and(super::ibp_seed::IbpSeed::active);
                    let cal_wanted = (bi_policy.enabled || seed_live) && clip_rows_enabled();
                    let cal_late = bi_cal_after_seed_forced();
                    let mut bi_cal = (cal_wanted && !cal_late).then(|| (l.clone(), u.clone()));
                    // #mr-ibp-seed: shrink-only intersection with the layer's
                    // certified IBP box (`ibp_seed.rs`), BEFORE #backward-interm.
                    // Why before: `tighten_layer`'s contract is only that the
                    // box handed in is a valid enclosure — it selects the
                    // unstable neurons FROM that box and intersects its own
                    // backward enclosure shrink-only — so seeding first is
                    // sound and makes it walk fewer (IBP-stabilized) neurons,
                    // the throughput miss on eps8 wide rows (3144w built no root
                    // in 30 s). It also turns #backward-interm into a CROSS-CHECK
                    // of the seed: an over-tight seeded box would cross its
                    // certified backward box and show up in its `crossed=`
                    // counter. The prefix gates it reads were re-certified on
                    // the seeded (published) boxes of earlier layers, which are
                    // valid enclosures, so its own enclosure stays valid.
                    if let Some(seed) = ibp.as_mut() {
                        seed.seed_layer(layers.len(), *input, &mut l, &mut u);
                    }
                    if cal_late && cal_wanted {
                        bi_cal = Some((l.clone(), u.clone()));
                    }
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
                    // #fold-ball: the relu output lies in `[0, max(u, 0)]` for
                    // every member of the fold ball, because `u` is this
                    // layer's final certified box (after any shrink-only
                    // intersection above) and holds for the whole ball.
                    if let Some(fm) = fold_mag.as_mut() {
                        fm.set(k + 1, u.iter().map(|&v| v.max(0.0)).collect());
                        fm.release(*input);
                    }
                    // #mr-ibp-seed: the relu output interval `[max(l, 0),
                    // max(u, 0)]` from this layer's FINAL published box (after
                    // both shrink-only intersections), so the next layer's IBP
                    // starts from the tightest predecessor available.
                    if let Some(seed) = ibp.as_mut() {
                        seed.post_relu(k, &l, &u);
                    }
                    let mut gates = gates_from_box(&l, &u);
                    if mode.outward() {
                        repair_upper_lines(&l, &u, &mut gates.1, &mut gates.2);
                    }
                    let (mut alpha, mut s, mut c) = gates;
                    // #rebake seed: bake a previous round's optimized alpha in
                    // place of the area heuristic, UNSTABLE neurons only (a
                    // stable neuron's exact fixed line is tighter than any
                    // relaxation and must stand). Clamped to [0, 1], so every
                    // seeded line is a valid DeepPoly lower relaxation; `ms`
                    // is derived AFTER this from the seeded values, and
                    // `apply_gates` takes its fractional-alpha widening branch
                    // for every non-binary row. Shape/finiteness mismatches
                    // fail open to the heuristic (never an error: a stale
                    // seed costs tightness, not soundness).
                    if let Some(seed) = alpha_seed {
                        if let Some(sa) = seed.get(layers.len()) {
                            if sa.len() == n {
                                for j in 0..n {
                                    if l[j] < 0.0 && u[j] > 0.0 && sa[j].is_finite() {
                                        alpha[j] = sa[j].clamp(0.0, 1.0);
                                    }
                                }
                            }
                        }
                    }
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
                        layer_ret.eligible = unst.len();
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
                                // RANKER-ONLY rows; nearest f32 is fine in both
                                // storage lanes (reads widen exactly first).
                                #[allow(clippy::cast_possible_truncation)]
                                for i in 0..naug {
                                    layer_ret.a_l.push((mrow[i].wide() - drow[i].wide()) as f32);
                                    layer_ret.a_u.push((mrow[i].wide() + drow[i].wide()) as f32);
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
                    // Calibration frame (#backward-interm, #mr-ibp-seed): the
                    // FORWARD-only bounds when a tightener was live (see
                    // `bi_cal` above), otherwise the published ones — which
                    // then ARE the forward bounds, so the historical behavior
                    // is bit-identical.
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
                    let (mo, do_, gstore) = apply_gates(
                        &mi,
                        &di,
                        &alpha,
                        &s,
                        &c,
                        n_in,
                        mode,
                        if use_mask { Some(&smask_in) } else { None },
                        blk,
                        &xabs_ext,
                    );
                    // Propagate the f32 error through the gate transform by the
                    // certified contraction `max(alpha_j, s_j)` — see
                    // [`gate_gerr_out`] for the derivation and for why the
                    // historical `(alpha_j + s_j)` was a strict OVER-charge.
                    // BOTH branches carry the same constant; the storage lane
                    // additionally adds this site's own dust (`gstore`,
                    // #tableau-f32), which is a separate account and unchanged.
                    let gerr_out: Vec<f64> = if use_f32 {
                        if T::F32_STORE {
                            (0..n)
                                .map(|j| {
                                    next_up(gate_gerr_out(alpha[j], s[j], gerr_in[j]) + gstore[j])
                                })
                                .collect()
                        } else {
                            (0..n)
                                .map(|j| gate_gerr_out(alpha[j], s[j], gerr_in[j]))
                                .collect()
                        }
                    } else {
                        Vec::new()
                    };
                    state[k + 1] = Some((mo, do_, gerr_out, smask_out));
                    completed_store_ops += 1;
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
        // #mr-ibp-seed terminal line, `complete=1` (the per-layer lines are
        // printed at each trunk relu). A walk that returns early — the deadline
        // above, any `?` — prints it from `IbpSeed`'s `Drop` with `complete=0`.
        if let Some(seed) = ibp.as_mut() {
            seed.finish();
        }
        if T::F32_STORE {
            // #tableau-f32 R9 observable: one line per root/rebake build. The
            // A/B's primary floor observable stays the pre-existing
            // `[backward-interm] done ... secs=` line, WHICH IS OP-WALK WALL —
            // its `t0` is taken at `BiState::new` above, after root setup but
            // before the forward tableau walk, not at the first BI pass. That
            // is exactly what #tableau-f32 wants (this line proves ARMING and
            // carries the independently timed full-build seconds), and
            // exactly what a reader asking "what does #backward-interm cost"
            // must NOT use: that reader wants `work_secs=` on the same line.
            // The two were conflated for two days and produced a whole dead
            // front on tinyimagenet (backward_interm.rs header has the
            // measurement).
            eprintln!(
                "[tableau-f32] armed: ops={} layers={} build_secs={:.2}",
                last_relu + 1,
                layers.len(),
                build_t0.elapsed().as_secs_f64()
            );
        }
        let backward_interm_execution = bi_state.as_ref().map_or_else(
            || super::backward_interm::BackwardIntermExecutionEvidence::from_policy(bi_policy),
            super::backward_interm::BiState::finish,
        );
        let root_precision_execution = RootPrecisionExecutionEvidence::completed(
            precision_policy,
            mode,
            use_f32,
            T::F32_STORE,
            completed_conv_ops,
            completed_store_ops,
        );
        Ok((
            Self {
                mode,
                mid,
                rad,
                xabs,
                layers,
                lo: lo.to_vec(),
                hi: hi.to_vec(),
                backward_interm_execution,
                root_precision_policy: precision_policy,
                root_precision_execution,
                // #spec-alpha is installed by the root driver AFTER `#rebake`
                // returns, never by the builder.
                spec_alpha_policy: None,
                // #head-alpha is installed by the ascent, never by the builder.
                head_alpha: None,
                // #root-eval-reuse: a fresh build has evaluated nothing.
                root_eval_memo: super::bab::RootEvalMemo::default(),
                head_group_source: None,
            },
            retained,
        ))
    }
}

#[cfg(test)]
thread_local! {
    /// TEST-ONLY #mr-ibp-seed ordering mutation (see
    /// [`with_bi_cal_after_seed_for_test`]).
    static BI_CAL_AFTER_SEED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Is the `bi_cal` capture forced AFTER the #mr-ibp-seed intersection on this
/// thread? Always `false` outside `cfg(test)`, so no production build can
/// reach the defective ordering.
#[cfg(test)]
fn bi_cal_after_seed_forced() -> bool {
    BI_CAL_AFTER_SEED.with(std::cell::Cell::get)
}

#[cfg(not(test))]
const fn bi_cal_after_seed_forced() -> bool {
    false
}

/// TEST-ONLY mutation switch for `ibp_seed_keeps_bi_cal_on_forward_only_lines`:
/// runs `f` with the `#backward-interm` / `#mr-ibp-seed` calibration capture
/// taken AFTER the seed's shrink-only intersection (the defect the capture's
/// position prevents), so the test can prove it tells the two apart.
/// Thread-local and reset on exit (also on unwind).
#[cfg(test)]
pub(crate) fn with_bi_cal_after_seed_for_test<Out>(f: impl FnOnce() -> Out) -> Out {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            BI_CAL_AFTER_SEED.with(|c| c.set(false));
        }
    }
    BI_CAL_AFTER_SEED.with(|c| c.set(true));
    let _reset = Reset;
    f()
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
pub(crate) fn retain_clip_rows<T: TabElem>(
    mi: &Array2<T>,
    di: &Array2<T>,
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
        // #tableau-f32 INTERFACE: retention happens ONCE, here, and the
        // retained rows STAY f64 — on the f32 storage lane the stored rows are
        // widened EXACTLY (f32 -> f64 is lossless), so the retained rows ARE
        // the rows `concretize_box` consumed and the calibration invariant
        // below (slack = line's own box-minimum minus the published bound)
        // holds verbatim, storage charges included automatically.
        for j in 0..n_in {
            m_sel[[r, j]] = mi[[i, j]].wide();
            d_sel[[r, j]] = di[[i, j]].wide();
        }
        // The augmented constant column. `mi` is
        // `(n, n_in + 1)`; column `n_in` is the bias.
        bm[r] = mi[[i, n_in]].wide();
        bd[r] = di[[i, n_in]].wide();
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

/// RADIUS-role masked elementwise writer: `dst = store_radius_up(f(a, b))`
/// over the live column blocks of a `#tableau-support-mask` (empty mask ⇒
/// dense). On the f64 lane the store is the identity, so the written values
/// are bit-identical to the historical `par_zip3_masked` call; on the
/// `#tableau-f32` lane the result is stored directed-UP (class (b)).
///
/// Every writer that uses this maps `(0, 0) -> 0` in exact arithmetic, and the
/// destination arrives zeroed, so a skipped block already holds the exact
/// result. See the module comment for the invariant.
pub(crate) fn par_zip2_radius_masked<T: TabElem>(
    dst: &mut Array2<T>,
    a: &Array2<T>,
    b: &Array2<T>,
    smask: &[u64],
    blk: usize,
    f: impl Fn(f64, f64) -> f64 + Sync,
) {
    let cols = dst.ncols();
    let ds = dst.as_slice_mut().expect("standard layout");
    let asl = a.as_slice().expect("standard layout");
    let bs = b.as_slice().expect("standard layout");
    if smask.is_empty() {
        ds.par_chunks_mut(cols)
            .zip(asl.par_chunks(cols).zip(bs.par_chunks(cols)))
            .for_each(|(d, (ar, br))| {
                for ((dv, &av), &bv) in d.iter_mut().zip(ar).zip(br) {
                    *dv = T::store_radius_up(f(av.wide(), bv.wide()));
                }
            });
        return;
    }
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
                    *dv = T::store_radius_up(f(av.wide(), bv.wide()));
                }
            }
        });
}

/// RADIUS-role elementwise map over the live column blocks of a
/// `#tableau-support-mask` (empty mask ⇒ dense): `dst = store_radius_up(f(dst))`.
///
/// Used for the outward `certify_up` widening of the D lane: outside the mask
/// the value is an exact `0.0` whose certified upper bound is `0.0`, while
/// `certify_up(0.0, _)` would return `next_up(0.0) = 5e-324`. On the f64 lane
/// the store is the identity (bit-identical to the historical `map_masked`).
fn map_radius_masked<T: TabElem>(
    dst: &mut Array2<T>,
    smask: &[u64],
    blk: usize,
    f: impl Fn(f64) -> f64 + Sync + Send,
) {
    let cols = dst.ncols();
    let ds = dst.as_slice_mut().expect("standard layout");
    if smask.is_empty() {
        ds.par_chunks_mut(cols).for_each(|row| {
            for v in row.iter_mut() {
                *v = T::store_radius_up(f(v.wide()));
            }
        });
        return;
    }
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
                    *v = T::store_radius_up(f(v.wide()));
                }
            }
        });
}

/// ChannelAffine (folded-BN) op body: `M' = s ⊙ M` (+ shift in the bias
/// column); `D' = |s| ⊙ D` widened by the certified parameter/rounding
/// envelope in Outward mode. Generic over the stored element (`#tableau-f32`:
/// M' entries are CENTER-role — nearest store, class (c), perturbation bound
/// folded into the same-site D widening; D' entries are RADIUS-role —
/// directed-UP store, class (b)). On the f64 lane every expression is the
/// historical monomorphic code verbatim (`fold_charge` is the identity).
///
/// Extracted from the walk so the per-site containment oracle
/// (`tableau_f32_store_per_site_containment_oracle`) can drive this exact
/// site over an adversarial grid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chaffine_op<T: TabElem>(
    mi: &mut Array2<T>,
    di: &mut Array2<T>,
    scale: &[f64],
    shift: &[f64],
    scale_rel_err: f64,
    shift_err: &[f64],
    n_in: usize,
    mode: RoundMode,
    smask_in: Option<&[u64]>,
    blk: usize,
    xabs_ext: &[f64],
) -> Vec<f64> {
    if mode.outward() && !legacy_rounding_forced() {
        return chaffine_op_signsafe(
            mi,
            di,
            scale,
            shift,
            scale_rel_err,
            shift_err,
            n_in,
            smask_in,
            blk,
            xabs_ext,
        );
    }
    let naug = n_in + 1;
    let g = next_up(scale_rel_err + 4.0 * UNIT);
    // Outward widening for the fused dust accumulation (see `add_op`).
    let acc_widen = 1.0 + 2.0 * gamma_n(naug + 8);
    let msl = mi.as_slice_mut().expect("standard layout");
    let dsl = di.as_slice_mut().expect("standard layout");
    let mut gstore = vec![0.0f64; scale.len()];
    for (j, (&sj, &tj)) in scale.iter().zip(shift.iter()).enumerate() {
        let sa = sj.abs();
        let mrow = &mut msl[j * naug..(j + 1) * naug];
        let drow = &mut dsl[j * naug..(j + 1) * naug];
        let m = smask_in.map_or(u64::MAX, |s| s[j]);
        let mut acc = 0.0f64;
        // f64-lane reference pair for the bias column's composed store (the
        // scale pass below THEN the shift fold): defaults to the pre-op values
        // (a masked-out scale pass leaves the column untouched on both lanes).
        let mut bias_m64 = mrow[n_in].wide();
        let mut bias_d64 = drow[n_in].wide();
        for b in 0..SUPPORT_BLOCKS {
            if !support_has(m, b) {
                continue;
            }
            let lo = (b * blk).min(naug);
            let hi = (lo + blk).min(naug);
            for i in lo..hi {
                // #tableau-f32: M' is CENTER-role (nearest store, charge into
                // the same-site D term); D' is RADIUS-role (directed-UP store);
                // the per-row g_err storage-dust charge is the EXACT deviation
                // `|dM| + |dD|` against the f64 site expression on THIS lane's
                // own inputs (`d64` below is bit-the-same value the f64 lane
                // stores), accumulated fused.
                let m2 = sj * mrow[i].wide();
                let (m2_stored, m2_charge) = T::store_center(m2);
                if mode.outward() {
                    let dv = drow[i].wide();
                    let base = sa * dv + g * (m2.abs() + sa * dv);
                    drow[i] = T::store_radius_up(next_up(
                        T::fold_charge(base, m2_charge) * (1.0 + 1e-15),
                    ));
                    if T::F32_STORE {
                        let d64 = next_up(base * (1.0 + 1e-15));
                        if i == n_in {
                            bias_m64 = m2;
                            bias_d64 = d64;
                        } else {
                            acc += ((m2_stored.wide() - m2).abs() + (drow[i].wide() - d64).abs())
                                * xabs_ext[i];
                        }
                    }
                } else {
                    let dv = drow[i].wide();
                    drow[i] = T::store_radius_up(dv * sa);
                    if T::F32_STORE {
                        let d64 = dv * sa;
                        if i == n_in {
                            bias_m64 = m2;
                            bias_d64 = d64;
                        } else {
                            acc += ((m2_stored.wide() - m2).abs() + (drow[i].wide() - d64).abs())
                                * xabs_ext[i];
                        }
                    }
                }
                mrow[i] = m2_stored;
            }
        }
        let b2 = mrow[n_in].wide() + tj;
        let (b2_stored, b2_charge) = T::store_center(b2);
        if mode.outward() {
            drow[n_in] = T::store_radius_up(next_up(T::fold_charge(
                drow[n_in].wide() + shift_err[j] + 2.0 * UNIT * b2.abs(),
                b2_charge,
            )));
        }
        mrow[n_in] = b2_stored;
        if T::F32_STORE {
            // The bias column stores TWICE on this site (the scale pass above,
            // then the shift fold here). Charge ONE composed term — the final
            // stored pair vs the f64 site (scale THEN shift) on this lane's own
            // inputs — which is exactly the per-site oracle's final-vs-final
            // functional and handles the cancellation case exactly.
            let b2_64 = bias_m64 + tj;
            let d64_f = if mode.outward() {
                next_up(bias_d64 + shift_err[j] + 2.0 * UNIT * b2_64.abs())
            } else {
                bias_d64
            };
            acc += ((b2_stored.wide() - b2_64).abs() + (drow[n_in].wide() - d64_f).abs())
                * xabs_ext[n_in];
            gstore[j] = next_up(acc * acc_widen);
        }
    }
    gstore
}

// --------------------------------------------------------------------------
//  #root-rounding: the sign-safe charge of the tableau's own ROUNDING
// --------------------------------------------------------------------------
//
// THE DEFECT THIS REPAIRS. The tableau is a SANDWICH, not a coefficient interval:
// `(M - D) x~ <= v(x) <= (M + D) x~` for every `x` in the box, with
// `x~ = (x, 1)`. Every Outward op used to charge its own rounding by widening
// the COEFFICIENT columns of D — the conv's `din = d + gamma_{k+2}(|m| + d)`
// plus the `certify_up` of the D lane, the gate's `6u(|mm| + dd)`, the channel
// affine's `4u(|m2| + |s| d)`, the Add's `2u |mo|`. A coefficient widening
// widens the walls only where `x_i >= 0`: at `x_i < 0` the lower wall
// `(m - d) x_i` moves UP by `d |x_i|` (and the upper wall down), so the charge
// TIGHTENS the enclosure on the negative side of the box. The concretize
// envelope `gamma_{n_in + 16 + 8R} (tabs + |bl| + |bu|)` was supposed to
// re-absorb that ("DEPTH-INDEPENDENT"), but it scales with the row's OWN
// magnitudes, which a cancelling row with a near-zero bias makes tiny.
// Exact-rational counterexample (no fold error at all): x in [-1, 0.01],
// z1 = w x + 0.02 with w = (-1, -1 + 1e-6) (both relus stable-active),
// z2 = relu(z1_0) - relu(z1_1) = -1e-6 x. The trunk-1 box's upper end was
// 9.999999971380213e-07, 2.891e-15 BELOW the exact max 1.0000000000287557e-06
// at x = -1 (`root_box_encloses_the_exact_minimal_rounding_leak`); 64 of 96
// swept cancelling nets escaped by up to 3.6e-14.
//
// THE REPAIR. Each op now produces the EXACT products of its stored inputs in
// round-to-nearest (no coefficient widening at all) and charges a certified
// bound on its own rounding as an ADDITIVE term in the D BIAS column, where
// the sign is known (`x~_bias = 1 > 0`). If a site's computed coefficients
// deviate from its exact target by at most `e_i` per coefficient, the walls
// move by at most `sum_i e_i |x~_i| <= sum_i e_i xabs_ext_i` for EVERY `x` in
// the box whatever its sign (`|x_i| <= |mid_i| + rad_i`), so adding that sum
// to `D[bias]` keeps both walls outside the exact ones pointwise. This is the
// form the f32 lane's `g_err` and the #fold-ball charge already use. The
// charged bound per site (`W` = the site's own coefficient magnitudes):
//   * conv (f64 lane): `gamma_{k+2} sum_t |W_ot| (|M_ti| + |D_ti|)` — the
//     Higham bound of the two length-`k` dot products, any order or FMA
//     (`conv_round_charge`, one single-column `|W|` conv); on the f32 compute
//     lane `conv_f32_gerr` charges both lanes against exact arithmetic into
//     `g_err` instead, which is the same account;
//   * gate: `4u (|lo_c| + |up_c| (+ |up0|) on the bias)` — a derivation over
//     the <= 3 roundings of `lo_c`, `up0`, `up_c` and the two of `mm`/`dd`
//     gives `<= 3.0000002u W`, for binary AND fractional alpha alike;
//   * channel affine: `(srel + 4u)(|s m| + |s d|)` — covers both products
//     (`<= 1.0000001u` each) and, soundly, any relative scale error that
//     reaches the site (`|ds v| <= srel |s| |v|`, `|v| <= sum (|m|+|d|)|x~|`);
//     the shift keeps its historical bias-column terms;
//   * Add: `2u (|ma + mb| + |da + db|)` (needs `1.0000001u` per lane).
// `root_round_charge` turns the fused accumulator into a certified upper bound
// (`1 + 2 gamma_{naug+10}` covers the accumulation's own rounding and the
// `(1 + u)` of `xabs`; a `4 MIN_POSITIVE sum(xabs_ext)` floor covers every
// gradual-underflow absolute error, `>= 2^52` products' worth). The bias
// column is processed LAST in every row so its stored value includes the
// row's whole charge, which keeps the `#tableau-f32` per-site containment
// rule `d32 - d64 >= |m32 - m64|` exact for every column
// (`store_radius_charged`).
//
// WHAT THE ENVELOPE STILL DOES. `concretize_box`'s `gam` term now covers only
// the concretization's own rounding (its sums and the `m -+ d` subtractions);
// it is kept unchanged, so it is now a strict over-charge for that job.
//
// Bits: every Outward root box moves (sub-1e-12 relative; see the fix report
// `scratchpad/vbs/diag10/fix_root_rounding.md`). Parity mode is untouched.
// The historical expressions survive only behind the TEST-ONLY switch below.

#[cfg(test)]
thread_local! {
    /// TEST-ONLY #root-rounding mutation switch (see
    /// [`with_legacy_rounding_charge_for_test`]).
    static ROOT_ROUNDING_LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Is the historical coefficient-column ROUNDING charge forced on this thread?
/// Also forced by the #fold-ball switch, which reproduces the whole pre-repair
/// walk (see [`with_legacy_fold_charge_for_test`]). Always `false` outside
/// `cfg(test)`, so no production build can reach the legacy expressions. Read
/// on the calling thread, never inside a rayon closure.
#[cfg(test)]
fn legacy_rounding_forced() -> bool {
    ROOT_ROUNDING_LEGACY.with(std::cell::Cell::get) || legacy_fold_charge_forced()
}

#[cfg(not(test))]
const fn legacy_rounding_forced() -> bool {
    false
}

/// TEST-ONLY mutation switch for the #root-rounding pins
/// (`root_box_encloses_the_exact_minimal_rounding_leak`,
/// `root_box_encloses_the_exact_affine_hull_on_cancelling_stable_nets`,
/// `f32_root_lanes_enclose_the_exact_hull_over_a_signed_d_lane`): runs `f`
/// with every Outward op charging its rounding the HISTORICAL way, in the
/// coefficient columns of D, and with `conv_f32_gerr`'s historical SIGNED
/// D-lane functional (bit for bit the pre-repair expressions on the f64 AND
/// the f32 lanes), so a test can prove its fixture tells the repair from the
/// defect. Thread-local and reset on exit (also on unwind).
#[cfg(test)]
pub(crate) fn with_legacy_rounding_charge_for_test<Out>(f: impl FnOnce() -> Out) -> Out {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ROOT_ROUNDING_LEGACY.with(|c| c.set(false));
        }
    }
    ROOT_ROUNDING_LEGACY.with(|c| c.set(true));
    let _reset = Reset;
    f()
}

/// Per-coefficient rounding factor of one gate site (#root-rounding block).
const GATE_ROUND_REL: f64 = 4.0 * UNIT;
/// Per-coefficient rounding factor of one residual-Add site (both lanes).
const ADD_ROUND_REL: f64 = 2.0 * UNIT;

/// Gradual-underflow floor of a #root-rounding charge: `4 MIN_POSITIVE` per
/// unit of `sum(xabs_ext)`, i.e. `>= 2^52` subnormal-scale absolute errors per
/// coefficient, all rounded up.
fn root_round_floor(xabs_ext: &[f64]) -> f64 {
    next_up(4.0 * f64::MIN_POSITIVE * next_up(xabs_ext.iter().sum::<f64>()))
}

/// Widening of a fused #root-rounding accumulator over `naug` columns.
fn root_round_widen(naug: usize) -> f64 {
    1.0 + 2.0 * gamma_n(naug + 10)
}

/// Certified #root-rounding charge `>= g * sum_i w_i |x~_i|` from the raw f64
/// accumulator `acc = fl(sum_i w_i xabs_ext_i)` of non-negative terms: `acc`
/// under-reports by at most a `1 - gamma` factor and `xabs` by `1 + u`, both
/// inside `widen = 1 + 2 gamma_{naug+10}`; each later op is one correctly
/// rounded op on non-negative upper bounds, dominated by its `next_up`; `floor`
/// covers underflow.
#[inline]
fn root_round_charge(g: f64, acc: f64, widen: f64, floor: f64) -> f64 {
    next_up(next_up(g * next_up(acc * widen)) + floor)
}

/// RADIUS store of a #root-rounding site value `v64` with the same-site
/// CENTER-store `charge`: the f64 lane stores `v64` exactly (it IS the site
/// value); the `#tableau-f32` lane stores `f64_to_f32_up(next_up(v64 +
/// charge))`, which is `>= v64 + charge` exactly — so `d32 - d64 >= |m32 -
/// m64|` holds with no reliance on the up-store adding slack.
#[inline]
fn store_radius_charged<T: TabElem>(v64: f64, charge: f64) -> T {
    if T::F32_STORE {
        T::store_radius_up(next_up(v64 + charge))
    } else {
        T::store_radius_up(v64)
    }
}

/// Block iteration over the COEFFICIENT columns `0..n_in` of one row: masked
/// => the `#tableau-support-mask` blocks, clamped to `n_in` so the bias column
/// is never visited (every #root-rounding body handles it last, explicitly);
/// dense => ONE block covering them all, whatever `blk` the caller passed.
#[inline]
const fn coef_blocks(masked: bool, n_in: usize, blk: usize) -> (usize, usize) {
    if masked {
        (SUPPORT_BLOCKS, blk)
    } else {
        (1, n_in)
    }
}

/// #root-rounding conv charge (f64 compute lane): per output row `o`,
/// `E_o >= g sum_t |W_ot| B_t` with `B_t = sum_i (|M_ti| + |D_ti|) xabs_ext_i`
/// over the conv's SOURCE rows — one length-`naug` reduction per source row
/// and one single-column `|W|` conv, like `conv_f32_gerr`. `g` is the conv's
/// `gamma_{k+2}` (plus any relative fold error, `g = gamma + rel + gamma rel`),
/// which bounds each lane's per-coefficient dot-product error by
/// `g (|W| |M|)` resp. `g (|W| |D|)` in any summation order. Masked-out
/// columns are exact zeros and contribute exactly nothing, so the reduction is
/// dense.
#[allow(clippy::too_many_arguments)]
fn conv_round_charge<T: TabElem>(
    c: &ConvOp,
    mi: &Array2<T>,
    di: &Array2<T>,
    xabs: &[f64],
    n_in: usize,
    n_out: usize,
    g: f64,
    floor: f64,
) -> Vec<f64> {
    let naug = n_in + 1;
    let n_src = mi.nrows();
    let bwiden = root_round_widen(naug);
    let mis = mi.as_slice().expect("standard layout");
    let dis = di.as_slice().expect("standard layout");
    let mut bcol = Array2::<f64>::zeros((n_src, 1));
    bcol.as_slice_mut()
        .expect("standard layout")
        .par_iter_mut()
        .enumerate()
        .for_each(|(t, bt)| {
            let mrow = &mis[t * naug..(t + 1) * naug];
            let drow = &dis[t * naug..(t + 1) * naug];
            let mut braw = 0.0f64;
            for i in 0..n_in {
                braw += (mrow[i].wide().abs() + drow[i].wide().abs()) * xabs[i];
            }
            // Bias column (index n_in): xabs_ext weight 1.
            braw += mrow[n_in].wide().abs() + drow[n_in].wide().abs();
            *bt = next_up(braw * bwiden);
        });
    let mut gcol = Array2::<f64>::zeros((n_out, 1));
    conv_apply_forward(c, &bcol, &mut gcol, true);
    // The single-column conv sums `k_fwd` non-negative products.
    let gwiden = 1.0 + 2.0 * gamma_n(c.k_fwd + 2);
    gcol.iter()
        .map(|&v| next_up(next_up(g * next_up(v * gwiden)) + floor))
        .collect()
}

/// Add a per-row #root-rounding charge `e[j]` to the D BIAS column, rounded
/// up. On the `#tableau-f32` lane the up-store dust of that one store is
/// charged into `gerr` (bias-column weight 1), like every other store site.
fn charge_bias_column<T: TabElem>(d: &mut Array2<T>, e: &[f64], n_in: usize, gerr: &mut [f64]) {
    for (j, &ej) in e.iter().enumerate() {
        let v = next_up(d[[j, n_in]].wide() + ej);
        d[[j, n_in]] = T::store_radius_up(v);
        if T::F32_STORE && !gerr.is_empty() {
            gerr[j] = next_up(gerr[j] + (d[[j, n_in]].wide() - v));
        }
    }
}

/// #root-rounding ChannelAffine body (Outward): `M' = s M`, `D' = |s| D` as
/// plain products, the shift into the bias column with its historical
/// `shift_err + 2u |b2|` terms, and the sign-safe charge
/// `(scale_rel_err + 4u) sum_i (|s m_i| + |s d_i|) xabs_ext_i` added to the D
/// bias column last. Storage classes as in `chaffine_op`.
#[allow(clippy::too_many_arguments)]
fn chaffine_op_signsafe<T: TabElem>(
    mi: &mut Array2<T>,
    di: &mut Array2<T>,
    scale: &[f64],
    shift: &[f64],
    scale_rel_err: f64,
    shift_err: &[f64],
    n_in: usize,
    smask_in: Option<&[u64]>,
    blk: usize,
    xabs_ext: &[f64],
) -> Vec<f64> {
    let naug = n_in + 1;
    let g = next_up(scale_rel_err + 4.0 * UNIT);
    let acc_widen = 1.0 + 2.0 * gamma_n(naug + 8);
    let rwiden = root_round_widen(naug);
    let floor = root_round_floor(xabs_ext);
    let (nblk, bw) = coef_blocks(smask_in.is_some(), n_in, blk);
    let msl = mi.as_slice_mut().expect("standard layout");
    let dsl = di.as_slice_mut().expect("standard layout");
    let mut gstore = vec![0.0f64; scale.len()];
    for (j, (&sj, &tj)) in scale.iter().zip(shift.iter()).enumerate() {
        let sa = sj.abs();
        let mrow = &mut msl[j * naug..(j + 1) * naug];
        let drow = &mut dsl[j * naug..(j + 1) * naug];
        let mask = smask_in.map_or(u64::MAX, |sm| sm[j]);
        let mut racc = 0.0f64;
        let mut dust = 0.0f64;
        for b in 0..nblk {
            if !support_has(mask, b) {
                continue;
            }
            let lo = (b * bw).min(n_in);
            let hi = (lo + bw).min(n_in);
            for i in lo..hi {
                let m2 = sj * mrow[i].wide();
                let d2 = sa * drow[i].wide();
                racc += (m2.abs() + d2.abs()) * xabs_ext[i];
                let (m2_stored, m2_charge) = T::store_center(m2);
                mrow[i] = m2_stored;
                drow[i] = store_radius_charged::<T>(d2, m2_charge);
                if T::F32_STORE {
                    dust +=
                        ((m2_stored.wide() - m2).abs() + (drow[i].wide() - d2).abs()) * xabs_ext[i];
                }
            }
        }
        // Bias column, last: scale, shift, then the whole row's charge.
        let m2b = sj * mrow[n_in].wide();
        let d2b = sa * drow[n_in].wide();
        racc += (m2b.abs() + d2b.abs()) * xabs_ext[n_in];
        let b2 = m2b + tj;
        let e = root_round_charge(g, racc, rwiden, floor);
        let d64 = next_up(next_up(next_up(d2b + shift_err[j]) + 2.0 * UNIT * b2.abs()) + e);
        let (b2_stored, b2_charge) = T::store_center(b2);
        mrow[n_in] = b2_stored;
        drow[n_in] = store_radius_charged::<T>(d64, b2_charge);
        if T::F32_STORE {
            dust +=
                ((b2_stored.wide() - b2).abs() + (drow[n_in].wide() - d64).abs()) * xabs_ext[n_in];
            gstore[j] = next_up(dust * acc_widen);
        }
    }
    gstore
}

/// #root-rounding residual-Add body (Outward): `mo = ma + mb`, `do = da + db`
/// as plain sums, and the sign-safe charge
/// `2u sum_i (|ma_i + mb_i| + |da_i + db_i|) xabs_ext_i` added to the D bias
/// column last. Storage classes as in `add_op`.
fn add_op_signsafe<T: TabElem>(
    mut ma: Array2<T>,
    mb: &Array2<T>,
    mut da: Array2<T>,
    db: &Array2<T>,
    smask_out: &[u64],
    blk: usize,
    xabs_ext: &[f64],
) -> (Array2<T>, Array2<T>, Vec<f64>) {
    let cols = ma.ncols();
    let rows = ma.nrows();
    let mut gstore = vec![0.0f64; rows];
    if cols == 0 {
        return (ma, da, gstore);
    }
    let n_in = cols - 1;
    let acc_widen = 1.0 + 2.0 * gamma_n(cols + 8);
    let rwiden = root_round_widen(cols);
    let floor = root_round_floor(xabs_ext);
    let (nblk, bw) = coef_blocks(!smask_out.is_empty(), n_in, blk);
    {
        let ms = ma.as_slice_mut().expect("standard layout");
        let ds = da.as_slice_mut().expect("standard layout");
        let mbs = mb.as_slice().expect("standard layout");
        let dbs = db.as_slice().expect("standard layout");
        ms.par_chunks_mut(cols)
            .zip(ds.par_chunks_mut(cols))
            .zip(mbs.par_chunks(cols).zip(dbs.par_chunks(cols)))
            .zip(gstore.par_iter_mut())
            .enumerate()
            .for_each(|(j, (((mrow, drow), (mbr, dbr)), gs))| {
                let mask = if smask_out.is_empty() {
                    u64::MAX
                } else {
                    smask_out[j]
                };
                let mut racc = 0.0f64;
                let mut dust = 0.0f64;
                for b in 0..nblk {
                    if !support_has(mask, b) {
                        continue;
                    }
                    let lo = (b * bw).min(n_in);
                    let hi = (lo + bw).min(n_in);
                    for i in lo..hi {
                        let sum = mrow[i].wide() + mbr[i].wide();
                        let dsum = drow[i].wide() + dbr[i].wide();
                        racc += (sum.abs() + dsum.abs()) * xabs_ext[i];
                        let (m_stored, m_charge) = T::store_center(sum);
                        mrow[i] = m_stored;
                        drow[i] = store_radius_charged::<T>(dsum, m_charge);
                        if T::F32_STORE {
                            dust += ((m_stored.wide() - sum).abs() + (drow[i].wide() - dsum).abs())
                                * xabs_ext[i];
                        }
                    }
                }
                // Bias column, last.
                let sum = mrow[n_in].wide() + mbr[n_in].wide();
                let dsum = drow[n_in].wide() + dbr[n_in].wide();
                racc += (sum.abs() + dsum.abs()) * xabs_ext[n_in];
                let d64 = next_up(dsum + root_round_charge(ADD_ROUND_REL, racc, rwiden, floor));
                let (m_stored, m_charge) = T::store_center(sum);
                mrow[n_in] = m_stored;
                drow[n_in] = store_radius_charged::<T>(d64, m_charge);
                if T::F32_STORE {
                    dust += ((m_stored.wide() - sum).abs() + (drow[n_in].wide() - d64).abs())
                        * xabs_ext[n_in];
                    *gs = next_up(dust * acc_widen);
                }
            });
    }
    (ma, da, gstore)
}

/// #root-rounding gate body (Outward): `L = (M - D) alpha`,
/// `U = (M + D) s (+ c on the bias)`, `M' = (L + U)/2`, `D' = (U - L)/2` as
/// plain f64 results, and the sign-safe charge
/// `4u sum_i (|lo_c| + |up_c| (+ |up0|)) xabs_ext_i` added to the D bias
/// column last — ONE formula for binary and fractional alpha (see the
/// #root-rounding block and `apply_gates`' header). Storage classes as in
/// `apply_gates`.
#[allow(clippy::too_many_arguments)]
fn apply_gates_signsafe<T: TabElem>(
    m: &Array2<T>,
    d: &Array2<T>,
    alpha: &[f64],
    s: &[f64],
    c: &[f64],
    n_in: usize,
    smask: Option<&[u64]>,
    blk: usize,
    xabs_ext: &[f64],
) -> (Array2<T>, Array2<T>, Vec<f64>) {
    let naug = n_in + 1;
    let n = m.nrows();
    let mut mo = Array2::<T>::from_elem((n, naug), T::ZERO);
    let mut do_ = Array2::<T>::from_elem((n, naug), T::ZERO);
    let mut gstore = vec![0.0f64; n];
    let acc_widen = 1.0 + 2.0 * gamma_n(naug + 8);
    let rwiden = root_round_widen(naug);
    let floor = root_round_floor(xabs_ext);
    let (nblk, bw) = coef_blocks(smask.is_some(), n_in, blk);
    let msrc = m.as_slice().expect("standard layout");
    let dsrc = d.as_slice().expect("standard layout");
    let mdst = mo.as_slice_mut().expect("standard layout");
    let ddst = do_.as_slice_mut().expect("standard layout");
    mdst.par_chunks_mut(naug)
        .zip(ddst.par_chunks_mut(naug))
        .zip(gstore.par_iter_mut())
        .enumerate()
        .for_each(|(j, ((mrow, drow), gs))| {
            let a_j = alpha[j];
            let s_j = s[j];
            let msr = &msrc[j * naug..(j + 1) * naug];
            let dsr = &dsrc[j * naug..(j + 1) * naug];
            // Outside the mask `m == d == 0.0`, so both outputs are EXACTLY
            // zero, which the freshly zeroed `mo`/`do_` already hold.
            let mask = smask.map_or(u64::MAX, |sm| sm[j]);
            let mut racc = 0.0f64;
            let mut dust = 0.0f64;
            for b in 0..nblk {
                if !support_has(mask, b) {
                    continue;
                }
                let lo = (b * bw).min(n_in);
                let hi = (lo + bw).min(n_in);
                for i in lo..hi {
                    let lo_c = (msr[i].wide() - dsr[i].wide()) * a_j;
                    let up_c = (msr[i].wide() + dsr[i].wide()) * s_j;
                    #[allow(clippy::manual_midpoint)]
                    let mm = (lo_c + up_c) * 0.5;
                    let dd = (up_c - lo_c) * 0.5;
                    racc += (lo_c.abs() + up_c.abs()) * xabs_ext[i];
                    let (mm_stored, mm_charge) = T::store_center(mm);
                    mrow[i] = mm_stored;
                    drow[i] = store_radius_charged::<T>(dd, mm_charge);
                    if T::F32_STORE {
                        dust += ((mm_stored.wide() - mm).abs() + (drow[i].wide() - dd).abs())
                            * xabs_ext[i];
                    }
                }
            }
            // Bias column, last: the intercept, then the whole row's charge.
            let lo_c = (msr[n_in].wide() - dsr[n_in].wide()) * a_j;
            let up0 = (msr[n_in].wide() + dsr[n_in].wide()) * s_j;
            let up_c = up0 + c[j];
            #[allow(clippy::manual_midpoint)]
            let mm = (lo_c + up_c) * 0.5;
            let dd = (up_c - lo_c) * 0.5;
            racc += (lo_c.abs() + up_c.abs() + up0.abs()) * xabs_ext[n_in];
            let d64 = next_up(dd + root_round_charge(GATE_ROUND_REL, racc, rwiden, floor));
            let (mm_stored, mm_charge) = T::store_center(mm);
            mrow[n_in] = mm_stored;
            drow[n_in] = store_radius_charged::<T>(d64, mm_charge);
            if T::F32_STORE {
                dust += ((mm_stored.wide() - mm).abs() + (drow[n_in].wide() - d64).abs())
                    * xabs_ext[n_in];
                *gs = next_up(dust * acc_widen);
            }
        });
    (mo, do_, gstore)
}

// --------------------------------------------------------------------------
//  #fold-ball: the sign-safe charge of a RELATIVE fold error
// --------------------------------------------------------------------------
//
// THE DEFECT THIS REPAIRS. The tableau is a SANDWICH, not a coefficient
// interval: `(M - D) x~ <= v(x) <= (M + D) x~` for every `x` in the box, with
// `x~ = (x, 1)`. A folded op's true parameters lie anywhere in the certified
// ball `|W - W'| <= rel |W'|` (`ConvOp::weight_rel_err`, likewise
// `scale_rel_err`) around the stored `W'`, so the true output carries an extra
// `E v` with `|E v| <= rel |W'| |v|`. That term is not linear in `x`, and
// adding `rel |W'| |M|` to the COEFFICIENT columns of D (the historical
// `g = gamma + rel + gamma rel` in `din = d + g(|m| + d)`) widens the walls
// only where `x_i >= 0`: at `x_i < 0` the lower wall `(m - d) x_i` moves UP by
// `d |x_i|`. Concretized, a positive lower coefficient `m - d` at `lo_i < 0`
// gives `(m - d) lo_i = m lo_i + d |lo_i|`, i.e. a bigger D RAISES the lower
// bound. The concretize headroom `gamma_{n_in + 16 + 8R} * tabs` re-absorbs
// that only while `rel` stays far below it (production folds carry
// `rel = 1e-15` against a headroom of `>= 3e-13` on a 3072-dim input). The
// `conv_bn` fixture (`rel = 1e-13`, `n_in = 12`, headroom `4.9e-15`) escaped
// its trunk-0 box by up to 1.6e-12 for members of its fold ball, and with
// `bias_err = 0` even the STORED network escaped (the `#frozen-dual-state`
// exact-rational containment gate, 2026-09-27; replica in the fix report).
//
// THE REPAIR. Charge the fold error where the sign is known: the bias column
// (`x~_bias = 1 > 0`), as `rel * sum_taps |W'| mag_in`, where `mag_in` bounds
// `|v_in|` over the box for EVERY member of the ball. On a walk with no
// relative fold error [`FoldMag::new`] returns `None` and every relative fold
// error is 0.
//
// CLOSED (#root-rounding): the remaining ROUNDING term `gamma_{k+2}` in the
// coefficient columns had the
// SAME sign structure and leaked (x in [-1, 0.01], z2 = -1e-6 x missed its
// trunk-1 box by 2.9e-15). Every op's rounding is now charged sign-safely in
// the bias column too — see the #root-rounding block.
//
// MAGNITUDES (each bounds `|v_t(x)|` for every `x` in the box and every member
// of the ball; every float op rounds up):
//   * input: `next_up(|mid| + rad)`;
//   * relu output: `max(u, 0)` from the layer's final certified box;
//   * conv: `(1 + rel) cert(sum |W'| mag_in) + |b| + bias_err`;
//   * channel affine: `(1 + rel) |s| mag_in + |shift| + shift_err`;
//   * add: `mag_a + mag_b`; flatten: unchanged.

#[cfg(test)]
thread_local! {
    /// TEST-ONLY #fold-ball mutation switch (see
    /// [`with_legacy_fold_charge_for_test`]).
    static FOLD_BALL_LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Is the TEST-ONLY legacy-charge switch set on this thread? Always `false`
/// outside `cfg(test)`, so no production build can reach the legacy charge.
#[cfg(test)]
fn legacy_fold_charge_forced() -> bool {
    FOLD_BALL_LEGACY.with(std::cell::Cell::get)
}

#[cfg(not(test))]
const fn legacy_fold_charge_forced() -> bool {
    false
}

/// TEST-ONLY mutation switch for the #fold-ball pin
/// (`root_box_encloses_the_exact_fold_ball_hull`): runs `f` with
/// [`FoldMag::new`] forced to `None`, i.e. with the HISTORICAL
/// (pre-#fold-ball) coefficient-column charge of `weight_rel_err` /
/// `scale_rel_err`, so the test can prove its fixture tells the repaired
/// charge from the defect it repairs. It also forces the historical
/// coefficient-column ROUNDING charge ([`legacy_rounding_forced`]), so the
/// walk is the whole pre-#fold-ball one bit for bit: the #root-rounding sites
/// would otherwise charge a coefficient-free `rel` soundly themselves
/// (`g = gamma + rel + gamma rel` is part of their sign-safe charge) and the
/// pin could no longer see the defect. Thread-local and reset on exit (also on
/// unwind): no other test thread and no production path can observe it.
#[cfg(test)]
pub(crate) fn with_legacy_fold_charge_for_test<Out>(f: impl FnOnce() -> Out) -> Out {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FOLD_BALL_LEGACY.with(|c| c.set(false));
        }
    }
    FOLD_BALL_LEGACY.with(|c| c.set(true));
    let _reset = Reset;
    f()
}

/// Per-tensor certified magnitudes for the #fold-ball charge (see the block
/// comment above). A tensor's vector is dropped after its last consumer.
struct FoldMag {
    mag: Vec<Option<Vec<f64>>>,
    left: Vec<usize>,
}

impl FoldMag {
    /// `Some` only in Outward mode on a walk (ops `..=last_relu`) that folds a
    /// nonzero relative weight or scale error; `consumers` are the walk's own
    /// per-tensor consumer counts, `xabs` its `|mid| + rad`.
    fn new(
        net: &TwinNet,
        last_relu: usize,
        mode: RoundMode,
        consumers: &[usize],
        xabs: &[f64],
    ) -> Option<Self> {
        let folds_rel = net.ops[..=last_relu].iter().any(|op| match op {
            TwinOp::Conv(c) => c.weight_rel_err != 0.0,
            TwinOp::ChannelAffine { scale_rel_err, .. } => *scale_rel_err != 0.0,
            _ => false,
        });
        if !(mode.outward() && folds_rel) || legacy_fold_charge_forced() {
            return None;
        }
        let mut mag = vec![None; net.ops.len() + 1];
        // `mid +- rad` covers the box exactly, so `|x| <= |mid| + rad`; the
        // `next_up` covers the rounding of that sum.
        mag[0] = Some(xabs.iter().map(|&v| next_up(v)).collect());
        Some(Self {
            mag,
            left: consumers.to_vec(),
        })
    }

    fn get(&self, id: usize) -> Result<&[f64]> {
        self.mag.get(id).and_then(|m| m.as_deref()).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "margin_row: fold-ball magnitude of tensor {id} is missing"
            ))
        })
    }

    fn set(&mut self, id: usize, v: Vec<f64>) {
        self.mag[id] = Some(v);
    }

    fn release(&mut self, id: usize) {
        if let Some(n) = self.left.get_mut(id) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.mag[id] = None;
            }
        }
    }
}

/// #fold-ball charge and output magnitude of one conv.
struct ConvFold {
    /// Per-channel bias error with the sign-safe relative charge folded in;
    /// exactly `c.bias_err` when `weight_rel_err == 0`.
    bias_err: Vec<f64>,
    /// Certified magnitude of the conv output (every member of the ball).
    mag_out: Vec<f64>,
}

/// The #fold-ball charge of conv `c` given its input magnitude `mag_in`:
/// `bias_err[ch] + max_{j in ch} rel * cert(sum_taps |W'| mag_in)` (the max
/// over the channel's spatial positions keeps the charge per channel, as
/// [`conv_bias_fold`] consumes it) and the output magnitude.
///
/// `cert(v) = next_up(v (1 + 2 gamma_{k+2}) + MIN_POSITIVE)`: the float sum of
/// `k` non-negative products (any order) is `>= exact (1 - gamma_k)`, and
/// `1 / (1 - gamma_k) <= 1 + 2 gamma_k`, with two roundings of the scaling
/// covered by `gamma_{k+2} >= gamma_k + 2u`; `MIN_POSITIVE` covers gradual
/// underflow of up to `2^52` products. Every other op is a single correctly
/// rounded op on non-negative upper bounds, so its `next_up` dominates the
/// exact result.
fn conv_fold_ball(c: &ConvOp, mag_in: &[f64], n_out: usize) -> ConvFold {
    let col = Array2::from_shape_vec((mag_in.len(), 1), mag_in.to_vec())
        .expect("a single column always has a valid shape");
    let mut dst = Array2::<f64>::zeros((n_out, 1));
    conv_apply_forward(c, &col, &mut dst, true);
    let g = gamma_n(c.k_fwd + 2);
    let p = (c.oshape.1 * c.oshape.2).max(1);
    let rel = c.weight_rel_err;
    let grow = next_up(1.0 + rel);
    let mut charge = vec![0.0f64; c.bias.len()];
    let mut mag_out = Vec::with_capacity(n_out);
    for (j, &v) in dst.iter().enumerate() {
        let ch = j / p;
        let wm = next_up(v * (1.0 + 2.0 * g) + f64::MIN_POSITIVE);
        if rel > 0.0 {
            charge[ch] = charge[ch].max(next_up(rel * wm));
        }
        mag_out.push(next_up(
            next_up(wm * grow) + next_up(c.bias[ch].abs() + c.bias_err[ch]),
        ));
    }
    let bias_err = if rel > 0.0 {
        c.bias_err
            .iter()
            .zip(&charge)
            .map(|(e, q)| next_up(e + q))
            .collect()
    } else {
        c.bias_err.clone()
    };
    ConvFold { bias_err, mag_out }
}

/// #fold-ball charge and output magnitude of one channel affine.
struct AffineFold {
    /// Per-neuron shift error with the sign-safe relative charge folded in;
    /// exactly `shift_err` when `scale_rel_err == 0`.
    shift_err: Vec<f64>,
    /// Certified magnitude of the affine output (every member of the ball).
    mag_out: Vec<f64>,
}

/// The #fold-ball charge of a channel affine (per-NEURON `scale` / `shift` /
/// `shift_err`, as [`TwinOp::ChannelAffine`] stores them):
/// `shift_err[j] + rel |s_j| mag_in[j]`, and the output magnitude
/// `(1 + rel) |s_j| mag_in[j] + |shift_j| + shift_err[j]`, all rounded up.
fn affine_fold_ball(
    scale: &[f64],
    shift: &[f64],
    rel: f64,
    shift_err: &[f64],
    mag_in: &[f64],
) -> Result<AffineFold> {
    if mag_in.len() != scale.len() || shift.len() != scale.len() || shift_err.len() != scale.len() {
        return Err(NyError::InvalidSpec(
            "margin_row: channel-affine fold-ball magnitude shape mismatch".into(),
        ));
    }
    let grow = next_up(1.0 + rel);
    let mut out_err = Vec::with_capacity(scale.len());
    let mut mag_out = Vec::with_capacity(scale.len());
    for (((&s, &t), &e), &m) in scale.iter().zip(shift).zip(shift_err).zip(mag_in) {
        let sm = next_up(s.abs() * m);
        out_err.push(if rel > 0.0 {
            next_up(e + next_up(rel * sm))
        } else {
            e
        });
        mag_out.push(next_up(next_up(sm * grow) + next_up(t.abs() + e)));
    }
    Ok(AffineFold {
        shift_err: out_err,
        mag_out,
    })
}

/// #fold-ball magnitude of an elementwise add.
fn fold_mag_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| next_up(x + y)).collect()
}

/// Conv bias fold: bias into the M bias column, certified bias error into the
/// D bias column. Generic over the stored element (`#tableau-f32`: the folded
/// bias is a CENTER-role nearest store, class (c), charged into the same-site
/// D bias widening, itself stored directed-UP, class (b); the storage-dust
/// `g_err` charge for the stored bias pair — weight 1 on the bias column — is
/// added into `gerr` when non-empty). f64 lane: historical expressions
/// verbatim, `gerr` untouched.
///
/// Extracted from the walk for the per-site containment oracle.
pub(crate) fn conv_bias_fold<T: TabElem>(
    mo: &mut Array2<T>,
    do_: &mut Array2<T>,
    bias: &[f64],
    bias_err: &[f64],
    p: usize,
    n_in: usize,
    mode: RoundMode,
    gerr: &mut [f64],
) {
    let n_out = mo.nrows();
    for j in 0..n_out {
        let ch = j / p;
        let b_new = mo[[j, n_in]].wide() + bias[ch];
        let (b_stored, b_charge) = T::store_center(b_new);
        mo[[j, n_in]] = b_stored;
        if mode.outward() {
            let d_old = do_[[j, n_in]].wide();
            let extra = next_up(T::fold_charge(bias_err[ch] + UNIT * b_new.abs(), b_charge));
            do_[[j, n_in]] = T::store_radius_up(next_up(d_old + extra));
            if T::F32_STORE && !gerr.is_empty() {
                // EXACT site dust, bias-column weight 1.0: deviations against
                // the f64 site expression on this lane's own inputs (`d64` is
                // bit-the-same value the f64 lane stores).
                let extra64 = next_up(bias_err[ch] + UNIT * b_new.abs());
                let d64 = next_up(d_old + extra64);
                gerr[j] = next_up(
                    gerr[j]
                        + ((b_stored.wide() - b_new).abs() + (do_[[j, n_in]].wide() - d64).abs()),
                );
            }
        } else if T::F32_STORE && !gerr.is_empty() {
            // Parity keeps D untouched on both lanes: only the center store
            // deviates.
            gerr[j] = next_up(gerr[j] + (b_stored.wide() - b_new).abs());
        }
    }
}

/// Residual-Add op body: `mo = ma + mb`; outward
/// `do = next_up(((da + db) + 2u*|mo|) * (1 + 1e-15))`, generic over the
/// stored element (`#tableau-f32`).
///
/// One fused masked pass computes the CENTER (nearest store, class (c), its
/// perturbation bound folded into the same-site D widening) and the RADIUS
/// (directed-UP store, class (b)), and — storage lane only — accumulates the
/// per-row `g_err` storage-dust charge
/// `sum_i f32_store_gerr_term(m32, d32) * xabs_ext_i` FUSED into the same
/// pass (`xabs_ext` is a 1-D L2-resident vector, so the accumulation adds
/// arithmetic but no memory traffic). Outside the union mask every operand is
/// exactly zero, so the reused buffers already hold the exact result. On the
/// f64 lane the elementwise expressions are the historical
/// `mo += &mb` / `par_zip3_masked` pair verbatim — bit-identical values — and
/// the returned charge vector is all-zero and unused.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_op<T: TabElem>(
    mut ma: Array2<T>,
    mb: &Array2<T>,
    mut da: Array2<T>,
    db: &Array2<T>,
    smask_out: &[u64],
    blk: usize,
    mode: RoundMode,
    xabs_ext: &[f64],
) -> (Array2<T>, Array2<T>, Vec<f64>) {
    if mode.outward() && !legacy_rounding_forced() {
        return add_op_signsafe(ma, mb, da, db, smask_out, blk, xabs_ext);
    }
    let outward = mode.outward();
    let cols = ma.ncols();
    let rows = ma.nrows();
    let mut gstore = vec![0.0f64; rows];
    // Outward widening for the fused dust accumulation (`<= cols` nonneg terms,
    // <= 4 roundings each, any association): dominates both this row-sum's own
    // f64 undershoot and the oracle's reference evaluation of the functional.
    let acc_widen = 1.0 + 2.0 * gamma_n(cols + 8);
    {
        let ms = ma.as_slice_mut().expect("standard layout");
        let ds = da.as_slice_mut().expect("standard layout");
        let mbs = mb.as_slice().expect("standard layout");
        let dbs = db.as_slice().expect("standard layout");
        let body = |mrow: &mut [T], drow: &mut [T], mbr: &[T], dbr: &[T], xr: &[f64]| -> f64 {
            let mut acc = 0.0f64;
            for i in 0..mrow.len() {
                let sum = mrow[i].wide() + mbr[i].wide();
                let (m_stored, m_charge) = T::store_center(sum);
                if outward {
                    let dsum = drow[i].wide() + dbr[i].wide();
                    let base = dsum + 2.0 * UNIT * sum.abs();
                    drow[i] =
                        T::store_radius_up(next_up(T::fold_charge(base, m_charge) * (1.0 + 1e-15)));
                    if T::F32_STORE {
                        // EXACT site dust (see the #tableau-f32 doc block):
                        // `d64` is bit-the-same value the f64 lane stores from
                        // these inputs, so the two deviations below ARE the
                        // per-site oracle's `|dM| + |dD|` terms.
                        let d64 = next_up(base * (1.0 + 1e-15));
                        acc +=
                            ((m_stored.wide() - sum).abs() + (drow[i].wide() - d64).abs()) * xr[i];
                    }
                } else {
                    let dsum = drow[i].wide() + dbr[i].wide();
                    drow[i] = T::store_radius_up(dsum);
                    if T::F32_STORE {
                        acc +=
                            ((m_stored.wide() - sum).abs() + (drow[i].wide() - dsum).abs()) * xr[i];
                    }
                }
                mrow[i] = m_stored;
            }
            acc
        };
        if smask_out.is_empty() {
            ms.par_chunks_mut(cols)
                .zip(ds.par_chunks_mut(cols))
                .zip(mbs.par_chunks(cols).zip(dbs.par_chunks(cols)))
                .zip(gstore.par_iter_mut())
                .for_each(|(((mrow, drow), (mbr, dbr)), gs)| {
                    *gs = next_up(body(mrow, drow, mbr, dbr, xabs_ext) * acc_widen);
                });
        } else {
            ms.par_chunks_mut(cols)
                .zip(ds.par_chunks_mut(cols))
                .zip(mbs.par_chunks(cols).zip(dbs.par_chunks(cols)))
                .zip(smask_out.par_iter().zip(gstore.par_iter_mut()))
                .for_each(|(((mrow, drow), (mbr, dbr)), (&m, gs))| {
                    let mut acc = 0.0f64;
                    for blk_i in 0..SUPPORT_BLOCKS {
                        if !support_has(m, blk_i) {
                            continue;
                        }
                        let lo = (blk_i * blk).min(cols);
                        let hi = (lo + blk).min(cols);
                        acc += body(
                            &mut mrow[lo..hi],
                            &mut drow[lo..hi],
                            &mbr[lo..hi],
                            &dbr[lo..hi],
                            &xabs_ext[lo..hi],
                        );
                    }
                    *gs = next_up(acc * acc_widen);
                });
        }
    }
    (ma, da, gstore)
}

/// #gate-gerr-max: g_err propagation across ONE relu GATE (the forward twin of
/// the backward engine's [`LayerGates::ms`] carry).
///
/// The gate is `L' = alpha_j*(M - D)`, `U' = s_j*(M + D)` (`+ c_j` on the bias
/// column only) — `apply_gates`, whose header carries the enclosure derivation.
/// `g_err` is the SUM-convention functional seeded in `build_prec_store` and
/// consumed by `concretize_box`:
///
/// ```text
/// g_err[j] >= sum_i (|dM_ji| + |dD_ji|) * xabs_ext_i     (bias column weight 1)
/// ```
///
/// Under that convention the certified contraction across the gate is
/// `max(alpha_j, s_j)`, NOT `alpha_j + s_j`. Per coefficient, an input error
/// `(dM, dD)` maps to `dL' = alpha_j*(dM - dD)` and `dU' = s_j*(dM + dD)` (the
/// intercept `c_j` is the same constant on both sides of the comparison and
/// carries no error), and with `dM' = (dL' + dU')/2`, `dD' = (dU' - dL')/2` the
/// identity `|(x+y)/2| + |(y-x)/2| = max(|x|, |y|)` gives
///
/// ```text
/// |dM'| + |dD'| = max(alpha_j*|dM - dD|, s_j*|dM + dD|)
///              <= max(alpha_j, s_j) * max(|dM - dD|, |dM + dD|)
///               = max(alpha_j, s_j) * (|dM| + |dD|)
/// ```
///
/// (the last step is the same identity read the other way). `alpha_j` and `s_j`
/// are uniform over the coefficient index `i`, so the factor pulls straight out
/// of the `sum_i (...) * xabs_ext_i` aggregate: `g_err_out[j] <=
/// max(alpha_j, s_j) * g_err_in[j]`. The bound is ATTAINED (`dD = 0` makes it an
/// equality), so `max` is the best uniform constant, not a convenient one.
///
/// UNIFORMITY IS A CONVENIENCE OF THIS FORM, NOT A HYPOTHESIS OF THE BOUND.
/// The per-coefficient step above never used it — it holds for each `i` with
/// that `i`'s own slopes — and only the aggregation needs one common constant,
/// for which `max_i max(a_ji, s_ji)` serves and stays in `[0, 1]`. Two call
/// sites had hardened the sentence above into "a per-start-node alpha cannot be
/// baked forward"; as a statement about the ERROR CARRY that is false. It says
/// nothing about the relaxation: whether the enclosure stays valid under a
/// non-uniform lower slope is OPEN, and a tableau column is an INPUT
/// coordinate, not a start node. See [`gate_gerr_out_nonuniform`], which
/// carries the generalized derivation, its tests and the corrected consequence.
///
/// The historical `(alpha_j + s_j)` came from bounding `|dM'|` and `|dD'|`
/// separately and re-adding, an OVER-charge by the factor
/// `1 + min(alpha_j, s_j)/max(alpha_j, s_j) in [1, 2]` per trunk relu. It is
/// EXACTLY 1 (this function is then bit-identical to the old expression)
/// wherever either slope is zero — every `alpha = 0` unstable neuron
/// (`gates_from_box` sets `alpha = 0` when `u < -l`), every stable-inactive
/// neuron, and every negative baked split — so the whole saving comes from the
/// `alpha = 1` population, where the factor is `1 + s_j in [1.5, 2]`.
///
/// `alpha, s in [0, 1]` (baked splits are `(1, 1)` or `(0, 0)`), so the
/// contraction is `<= 1`: the functional can no longer GROW across a relu, which
/// the old constant permitted.
///
/// SCOPE. This is the propagation of the INPUT error only. The gate's own f64
/// evaluation rounding is the same coefficient movement the f64 lane makes and
/// belongs to the depth-scaled `gamma_n(n_in + 16 + 8*relu_depth)` envelope in
/// `concretize_box` (see `apply_gates`' header, "the cross-layer leak of the
/// coefficient movement itself is the depth-scaled `concretize_box` envelope's
/// job"), which is untouched here; the storage lane's own dust is `gstore`, also
/// untouched and added by the caller. `next_up` keeps the returned charge an
/// upper bound of the exact product.
#[inline]
pub(crate) fn gate_gerr_out(alpha_j: f64, s_j: f64, gerr_in_j: f64) -> f64 {
    // `alpha_j.max(s_j)` IS `LayerGates::ms[j]` by construction (`build_prec_store`
    // derives `ms` from these very slopes) — the forward site and the backward
    // carry now quote one constant.
    next_up(alpha_j.max(s_j) * gerr_in_j)
}

/// #gate-gerr-max-nonuniform: the SAME gate carry when `alpha` and `s` VARY
/// over the coefficient index `i` — the case [`gate_gerr_out`]'s header says it
/// cannot cover.
///
/// ## What the old note claimed, and what is actually true
///
/// [`gate_gerr_out`]'s derivation ends "`alpha_j` and `s_j` are uniform over
/// the coefficient index `i`, so the factor pulls straight out of the
/// `sum_i (...) * xabs_ext_i` aggregate", and two call-site comments turned
/// that into a prohibition: *"a per-start-node alpha cannot be baked forward"*
/// (`margin_row/mod.rs`, `margin_row/spec_alpha.rs`). That prohibition is
/// STRONGER THAN THE MATHEMATICS. Uniformity is what lets a single factor pull
/// out; it is not what makes the bound true.
///
/// The per-coefficient step of that derivation never used uniformity:
///
/// ```text
/// |dM'_ji| + |dD'_ji| = max(a_ji*|dM_ji - dD_ji|, s_ji*|dM_ji + dD_ji|)
///                    <= max(a_ji, s_ji) * (|dM_ji| + |dD_ji|)
/// ```
///
/// holds for EACH `i` on its own, with that `i`'s own slopes. Only the
/// aggregation step needs a common constant, and any upper bound of the
/// per-`i` factors serves:
///
/// ```text
/// g_err_out[j] =  sum_i (|dM'_ji| + |dD'_ji|) * xabs_ext_i
///              <= sum_i max(a_ji, s_ji) * (|dM_ji| + |dD_ji|) * xabs_ext_i
///              <= A_j * sum_i (|dM_ji| + |dD_ji|) * xabs_ext_i
///               = A_j * g_err_in[j],   A_j := max_i max(a_ji, s_ji)
/// ```
///
/// `A_j` is still in `[0, 1]` because every slope is, so the functional still
/// CANNOT GROW across a relu — the property the `max` form was adopted for.
/// When the slopes happen to be uniform, `A_j` collapses to `max(alpha_j, s_j)`
/// and this is [`gate_gerr_out`] exactly.
///
/// ## Consequence
///
/// This function's own result stands: the ERROR CARRY is valid with slopes that
/// vary over `i`, with `A_j = max_i max(a_ji, s_ji)`. That is a real
/// generalization of [`gate_gerr_out`] and it is what the derivation proves.
///
/// ⚠ **An earlier version of this note went on to claim that "the margin-row
/// tableau carries the spec rows as coefficient COLUMNS", and therefore that
/// varying alpha over `i` DELIVERS abc's per-start-node alpha. THAT CLAIM IS
/// FALSE.** The tableau's columns are the network's INPUT coordinates plus a
/// bias: `naug = net.n_in + 1`, and `build_retaining_inner` validates
/// `lo.len() == n_in` against the input box. The spec/start-node axis is a
/// different axis entirely — it indexes `RootEval::dj`, one entry per
/// adversarial class, and it is produced on the BACKWARD side. So "alpha
/// varying over `i`" is a per-INPUT-COORDINATE alpha, not a per-start-node one.
///
/// **Two separate questions were conflated, and only one is answered here.**
/// This derivation bounds the floating-point error functional; it says nothing
/// about whether the RELAXATION stays a valid enclosure when the lower slope
/// differs per column. That second question is OPEN. A randomized probe of a
/// simplified concretization found apparent violations at 1.56% under
/// per-column alpha — but its negative control FAILED (the uniform, shipped,
/// moat-clean case violated at 0.146% in the same harness), so the probe models
/// the real bound badly and settles nothing in either direction.
///
/// ⛔ **Therefore: do NOT arm a non-uniform per-column alpha on the forward
/// bake.** No production caller uses a non-uniform alpha; only
/// `margin_row/tests.rs` exercises this function, and it must stay that way
/// until the enclosure question is answered on the real concretization, error
/// carry included. The sound route to abc's per-start-node alpha does not need
/// non-uniformity at all: bake a SEPARATE tableau per blocking class, each with
/// its own uniform alpha (not implemented).
///
/// ## What this function does NOT do
///
/// It changes no behaviour. Nothing calls it yet: the forward lane still bakes
/// one slope per neuron, and `#spec-alpha` is still backward-only. This exists
/// so the obstruction is retired with a proof and a test rather than by
/// assertion, and so the forward adoption is a separate, measurable step.
///
/// `slopes_j` and `alphas_j` are neuron `j`'s per-column slopes. Empty input
/// returns `gerr_in_j` unchanged (charge 1, the identity — never 0, which would
/// admit everything).
#[inline]
#[cfg(test)]
pub(crate) fn gate_gerr_out_nonuniform(alphas_j: &[f64], slopes_j: &[f64], gerr_in_j: f64) -> f64 {
    debug_assert_eq!(
        alphas_j.len(),
        slopes_j.len(),
        "per-column alpha and s must be the same width"
    );
    if alphas_j.is_empty() {
        return gerr_in_j;
    }
    let a_max = alphas_j
        .iter()
        .zip(slopes_j.iter())
        .fold(0.0_f64, |acc, (&a, &s)| acc.max(a.max(s)));
    gate_gerr_out(a_max, a_max, gerr_in_j)
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
/// * f32 D-lane error at `(j,i)`: `<= gamma_f32 * (|W| @ |din|)_ji`. The D
///   lane is SIGNED (every gate with `U < L` makes it negative), so the bound
///   needs the MAGNITUDE `|din|`: the historical `B[t]` summed `|mi| + din`,
///   which charges a coefficient with `din = -|mi|` NOTHING while both its
///   f32 products are off by `~u32 |W| |mi|` (#root-rounding follow-up,
///   `conv_f32_gerr_charges_a_signed_d_lane_by_magnitude`).
/// * propagated input error: `<= (|W| @ g_err_in-as-coeffs)`.
///
/// Summing `E_out_ji * xabs_ext_i` over `i` collapses the per-coefficient matrix
/// into a per-neuron VECTOR conv (`|W|` gather, ONE column) over
/// `g_err_in[t] + gamma_f32 * B[t]`, with `B[t] = sum_i (|mi_ti| + |din_ti|) *
/// xabs_ext_i` (bias column weight 1). Cost: one length-`n` reduction + one
/// single-column `conv_abs` — negligible beside the two full-width conv lanes.
/// Everything is rounded outward (upper bound). A per-output additive floor
/// covers f32 subnormal/FTZ rounding. Pure over-estimate -> pure loosening.
/// `store_dust` (`#tableau-f32` storage lane only): additionally budget the
/// walk-storage dust that routes through this conv —
/// * the `din` directed-UP store (`<= 2*UNIT_F32*din32 + F32_MIN_POSITIVE`
///   per coefficient) and the D-output `certify_up` re-store
///   (`<= ~2*UNIT_F32*do32 + F32_MIN_POSITIVE`), both dominated by an extra
///   `8*UNIT_F32 * Bd[t]` inside the `|W|`-gather, with `Bd[t] =
///   sum_i |din_ti| * xabs_ext_i` the D-LANE-ONLY functional (the din dust is
///   `<= 2*UNIT_F32*Bd` outright, and the do-side functional is `<= (1+eps) *
///   (|W| @ [2*UNIT_F32*Bd])_j`) — 2x margin. Charging the full `B[t]`
///   (|M|+D envelope) here was sound but ~|M|/D-fold conservative — D runs
///   orders below |M| in the early layers — and the aggregate loosening
///   tipped knife-edge rows (the 2026-08-20 ship-gate FAIL);
/// * the `F32_MIN_POSITIVE` absolute floors: the din side per SOURCE row
///   (`+ F32_MIN_POSITIVE * s_xabs` inside `ginp`, so `|W|` scales it), the
///   do side per OUTPUT row (`+ 2*F32_MIN_POSITIVE * s_xabs` on the ftz
///   floor).
///
/// The stored `mi` input needs NO charge here: reading a stored f32 back is
/// EXACT, and `mi`'s own store dust was charged at its store site and arrives
/// through `gerr_in` (propagated by the `|W|` gather like every input error).
/// `store_dust = false` keeps the shipped f32-compute lane's expressions
/// bit-identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_f32_gerr<T: TabElem>(
    c: &ConvOp,
    mi: &Array2<T>,
    din: &Array2<T>,
    gerr_in: &[f64],
    xabs: &[f64],
    s_xabs: f64,
    n_in: usize,
    smask: Option<&[u64]>,
    blk: usize,
    store_dust: bool,
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
    // D-lane magnitude in `B`/`Bd`: `|din|` (the D lane is SIGNED). The
    // TEST-ONLY #root-rounding legacy switch restores the historical signed
    // `din` so that switch replays the whole pre-repair walk on the f32 lanes
    // too (`f32_root_lanes_enclose_the_exact_hull_over_a_signed_d_lane`);
    // always the magnitude outside `cfg(test)`. Read here, on the calling
    // thread, never inside the rayon closure below.
    let signed_legacy = legacy_rounding_forced();
    let dmag = move |v: f64| -> f64 {
        if signed_legacy {
            v
        } else {
            v.abs()
        }
    };
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
            // D-lane-only functional `Bd[t] = sum_i |din_ti| * xabs_ext_i`
            // (storage lane only): the din/do_ up-store dust is proportional
            // to the D LANE's own magnitudes, so it is budgeted against Bd,
            // not the full |M|+D envelope B (the old fused-B charge was the
            // heavy conservatism that tipped knife-edge rows — see the
            // #tableau-f32 doc block).
            let mut braw_d = 0.0;
            // #tableau-support-mask: skipped blocks contribute an exact `0.0`.
            let mask = smask.map_or(u64::MAX, |s| s[t]);
            for b in 0..SUPPORT_BLOCKS {
                if !support_has(mask, b) {
                    continue;
                }
                let lo = (b * blk).min(n_in);
                let hi = (lo + blk).min(n_in);
                for i in lo..hi {
                    braw += (mrow[i].wide().abs() + dmag(drow[i].wide())) * xabs[i];
                    if store_dust {
                        braw_d += dmag(drow[i].wide()) * xabs[i];
                    }
                }
            }
            // Bias column (index n_in): xabs_ext weight = 1.
            braw += mrow[n_in].wide().abs() + dmag(drow[n_in].wide());
            let b_up = next_up(braw * (1.0 + bwiden));
            *gt = if store_dust {
                braw_d += dmag(drow[n_in].wide());
                let bd_up = next_up(braw_d * (1.0 + bwiden));
                next_up(
                    gerr_in[t] + gf32 * b_up + 8.0 * UNIT_F32 * bd_up + F32_MIN_POSITIVE * s_xabs,
                )
            } else {
                next_up(gerr_in[t] + gf32 * b_up)
            };
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
    let ftz = if store_dust {
        // Per-output absolute floor of the do_-side up-store (never publishes
        // a subnormal, so the movement floor is F32_MIN_POSITIVE per column).
        next_up(ftz + 2.0 * F32_MIN_POSITIVE * s_xabs)
    } else {
        ftz
    };
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
fn concretize_box<T: TabElem>(
    m: &Array2<T>,
    d: &Array2<T>,
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
    //
    // REFUTED AS STATED and REPAIRED (#root-rounding): the coefficient-column
    // widenings were not re-absorbed by
    // `rl` in general. A positive-coefficient row concretizes at `lo_i < 0` to
    // `m lo_i + d |lo_i|`, so every coefficient widening RAISED the lower end
    // there (and lowered the upper end at a negative-coefficient row), and this
    // envelope covers that only in proportion to `tabs + |bl| + |bu|`, which a
    // cancelling row with a near-zero bias makes tiny (a 1-input, 2-trunk-relu
    // stable-active net missed its trunk-1 box by 2.9e-15 in exact rationals).
    // Every op now charges its own rounding sign-safely in the D BIAS column
    // (#root-rounding block), so the tableau arriving here is an exact
    // sandwich of the stored network over the box, and `gam` only has to cover
    // THIS function's rounding: the `m -+ d` subtractions and the `n_in`-term
    // sums, `<= gamma_{n_in + 3} (tabs + |bl|)` per endpoint. The historical
    // depth-scaled constant is kept, now a strict over-charge for that job.
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
                    // Reads widen EXACTLY (f32 -> f64 is lossless); all
                    // arithmetic below is f64 in both storage lanes.
                    let lo_c = mrow[i].wide() - drow[i].wide();
                    let up_c = mrow[i].wide() + drow[i].wide();
                    vl += lo_c * mid[i];
                    rl += lo_c.abs() * rad[i];
                    vu += up_c * mid[i];
                    ru += up_c.abs() * rad[i];
                    tabs += (lo_c.abs() + up_c.abs()) * xabs[i];
                }
            }
            let bl = mrow[n_in].wide() - drow[n_in].wide();
            let bu = mrow[n_in].wide() + drow[n_in].wide();
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
/// `c = -u*l/(u-l)`, `alpha = [u >= -l]`. If an intermediate makes the
/// legacy slope/intercept nonfinite on finite endpoints, use an equivalent
/// bounded-ratio slope and a certified endpoint intercept instead. Every
/// finite legacy slope/intercept retains its original bits.
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
            if !(s[j].is_finite() && c[j].is_finite()) {
                if let Some((slope, intercept)) = finite_upper_line_from_endpoints(ll, uu, None) {
                    s[j] = slope;
                    c[j] = intercept;
                }
            }
            alpha[j] = if uu >= -ll { 1.0 } else { 0.0 };
        }
    }
    (alpha, s, c)
}

/// Exceptional arithmetic only: avoid both `u-l` and `-u*l` overflow. For
/// any stored slope in [0,1], a certified intercept is
/// `max(-s*l, u-s*u)`. Directed products/subtraction include gradual-underflow
/// dust. The known endpoint bounds `-s*l <= -l` and `u-s*u <= u` cap excess
/// outward nudges, so even MAX endpoints need no infinite intercept.
fn finite_upper_line_from_endpoints(
    l: f64,
    u: f64,
    preferred_slope: Option<f64>,
) -> Option<(f64, f64)> {
    if !(l.is_finite() && u.is_finite() && l < 0.0 && u > 0.0) {
        return None;
    }
    let a = -l;
    let slope = match preferred_slope {
        Some(s) if s.is_finite() && (0.0..=1.0).contains(&s) => s,
        _ if a >= u => {
            let ratio = u / a;
            ratio / (1.0 + ratio)
        }
        _ => {
            let ratio = a / u;
            1.0 / (1.0 + ratio)
        }
    };
    let left = if slope == 0.0 {
        0.0
    } else {
        next_up(slope * a).min(a)
    };
    let right = if slope == 1.0 {
        0.0
    } else {
        let product_lower = if slope == 0.0 {
            0.0
        } else {
            next_down(slope * u).max(0.0)
        };
        next_up(u - product_lower).min(u)
    };
    let intercept = left.max(right).max(0.0);
    (slope.is_finite() && intercept.is_finite()).then_some((slope, intercept))
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
        if s[j] == 0.0 && c[j] == 0.0 && u[j] <= 0.0 {
            continue; // inactive: exact
        }
        if s[j] == 1.0 && c[j] == 0.0 && l[j] >= 0.0 {
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
        let repaired = if bump > 0.0 {
            next_up(next_up(c[j] + bump))
        } else {
            c[j]
        };
        if s[j].is_finite() && repaired.is_finite() {
            c[j] = repaired;
        } else if let Some((slope, intercept)) =
            finite_upper_line_from_endpoints(l[j], u[j], Some(s[j]))
        {
            s[j] = slope;
            c[j] = intercept;
        } else {
            // Invalid/nonfinite endpoint domains remain visible to the
            // existing numerical firewall; they cannot gain gate authority.
            c[j] = repaired;
        }
    }
}

#[cfg(test)]
#[path = "root_extreme_gate_tests.rs"]
mod extreme_gate_tests;

/// Gate application on the tableau (Python parity):
/// `L = (M-D)*alpha; U = (M+D)*s; U_bias += c; M' = (L+U)/2; D' = (U-L)/2`.
///
/// OUTWARD (#root-rounding, the shipped path, `apply_gates_signsafe`). The
/// obligation is FUNCTIONAL, not per coefficient: for every `x~ = (x, 1)` with
/// `|x_i| <= xabs_i`,
///
///   (‡)  `(M' - D') x~ <= L x~`  and  `(M' + D') x~ >= U x~`  in REAL arithmetic,
///
/// with `L = (m-d)*alpha`, `U = (m+d)*s (+ c on the bias)` the exact products
/// of the STORED inputs. `M'`, `D'` are the plain f64 results; per coefficient
/// they miss (‡)'s exact walls by at most `e_i`, where (u = 2^-53)
/// `|lo_c - L| <= gamma_2/(1-gamma_2) |lo_c|` (the sub and, for fractional
/// alpha, the mul), `|up0 - U0| <= gamma_2/(1-gamma_2) |up0|`, the bias add
/// `<= u/(1-u) |up_c|`, and `mm -+ dd = lo_c|up_c + d6 Mh -+ d7 Dh` with
/// `|d6 Mh| + |d7 Dh| <= u max(|lo_c|, |up_c|)`; together
/// `e_i <= 3.0000002u W_i`, `W_i = |lo_c| + |up_c| (+ |up0| on the bias)`,
/// for binary AND fractional alpha (no chord coupling is needed). The charge
/// `4u sum_i W_i xabs_ext_i` (certified by `root_round_charge`) is added to
/// `D'[bias]`, which moves both walls outward by it at EVERY `x`, so (‡) holds
/// whatever the sign of the input box. Pinned in exact rationals by
/// `fractional_alpha_apply_gates_widening_covers_the_extra_rounding`.
///
/// HISTORICAL coefficient-column widening (reachable only through the
/// TEST-ONLY legacy switch `with_legacy_rounding_charge_for_test`; Parity
/// stores `mm`/`dd` unwidened). The per-coefficient obligation was
///
///   (†)  `M' - D' <= L` and `M' + D' >= U`  in REAL arithmetic,
///
/// where `L = (m-d)*alpha` and `U = (m+d)*s (+ c on the bias column)` are the
/// exact products of the STORED inputs — the computed sandwich rows must not
/// overstate the exact ones (the cross-layer leak of the coefficient movement
/// itself is the depth-scaled `concretize_box` envelope's job, unchanged here).
///
/// * BINARY `alpha in {0, 1}` (every build until #rebake): the historical line
///   `next_up((dd + 6u*(|mm| + dd)) * (1 + 1e-15))`. Its budget `B = |mm| + dd
///   = max(up_c, -lo_c)` covers (†) because `gates_from_box` COUPLES the
///   heuristic alpha to the chord: `alpha = 1` only where `u >= -l`, i.e.
///   `s >= 1/2`, which gives `|lo_c| <= 2B`; and `alpha = 0` makes `lo_c`
///   exactly `0`. Worst case sits at ~4uB against the 6uB provided (kept
///   bit-identical; every existing consumer and the parity oracle depend on
///   the exact bits).
/// * FRACTIONAL `alpha in (0, 1)` (#rebake bakes the optimizer's alpha back
///   into the tableau): the coupling is GONE — `alpha -> 1` with `s -> 0`
///   makes `|lo_c| / B` unbounded, so NO constant times `(|mm| + dd)` covers
///   the lower product's rounding, and the `(1+1e-15)` + `next_up` headroom
///   cannot absorb it (the anatomy note's hope is disproved by that regime;
///   the grid test carries the counterexample). The branch below therefore
///   widens by the row magnitudes themselves. Derivation (u = 2^-53, one
///   rounding per op, `*0.5` exact):
///
///   ```text
///   lo_c = L(1+d1)(1+d2) — fractional alpha: the sub AND the mul round;
///                          d2 is the one NEW rounding vs binary
///   up_c = U0(1+d3)(1+d4) [(+c)(1+d5) on bias],  U0 = (m+d)*s
///   mm   = Mh(1+d6), dd = Dh(1+d7),  Mh = (lo_c+up_c)/2, Dh = (up_c-lo_c)/2
///   ```
///
///   (†) reduces to `drow >= Dh + u|Mh| + max(|e_lo|, |e_up|)` with
///   `e_lo = lo_c - L`, `e_up = up_c - U`, because
///   `mm - L = Dh + d6*Mh + e_lo` and `U - mm = Dh - d6*Mh - e_up`. Bounds:
///   `Dh <= dd + 1.1u|dd|`, `|Mh| <= 1.1|mm|`, `|e_lo| <= 2.01u|lo_c|`,
///   `|e_up| <= 2.01u|up_c|` (bias: `<= 2.03u|up0| + 1.01u|up_c|`). With
///   `W = |lo_c| + |up_c| (+ |up0| on bias)` and `|mm|, |dd| <= W/2`, the
///   requirement is `<= dd + u(|mm| + |dd|) + 2.03uW + O(u^2 W)`, and
///   `dd + u*(|mm| + |dd| + 8W)` dominates it with surplus >= 5uW. The fp
///   evaluation of the widening term (<= 7 roundings, all on a nonneg
///   expression) under-shoots by <= 8u * (u * 9W) = O(u^2 W), inside the
///   surplus; the final `next_up(fl(dd + wid)) >= dd + wid` exactly. The
///   `8W` term is deliberately fat (4W would suffice) — headroom over shaving,
///   at a widening cost of ~1e-15 relative. Subnormal-range absolute dust is
///   handled by the same convention as the historical line (below the ulp of
///   every sum it enters; see the support-mask module comment).
///
///   The movement `|drow - dd| <= ~9.5uW` stays first-order-in-u times the
///   row magnitudes — the same class as the historical ~7uB — so the
///   depth-scaled `gamma_n(n_in + 16 + 8*relu_depth)` concretize envelope
///   was supposed to absorb its cross-layer leak (n_in >> 8R on every
///   twin-wall net). That argument was REFUTED (a coefficient widening moves a
///   wall INWARD at negative inputs, and the envelope scales with the row's own
///   magnitudes), which is what #root-rounding repairs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_gates<T: TabElem>(
    m: &Array2<T>,
    d: &Array2<T>,
    alpha: &[f64],
    s: &[f64],
    c: &[f64],
    n_in: usize,
    mode: RoundMode,
    smask: Option<&[u64]>,
    blk: usize,
    xabs_ext: &[f64],
) -> (Array2<T>, Array2<T>, Vec<f64>) {
    if mode.outward() && !legacy_rounding_forced() {
        return apply_gates_signsafe(m, d, alpha, s, c, n_in, smask, blk, xabs_ext);
    }
    let naug = n_in + 1;
    let n = m.nrows();
    let mut mo = Array2::<T>::from_elem((n, naug), T::ZERO);
    let mut do_ = Array2::<T>::from_elem((n, naug), T::ZERO);
    let mut gstore = vec![0.0f64; n];
    // Outward widening for the fused dust accumulation (see `add_op`).
    let acc_widen = 1.0 + 2.0 * gamma_n(naug + 8);
    let msrc = m.as_slice().expect("standard layout");
    let dsrc = d.as_slice().expect("standard layout");
    let mdst = mo.as_slice_mut().expect("standard layout");
    let ddst = do_.as_slice_mut().expect("standard layout");
    mdst.par_chunks_mut(naug)
        .zip(ddst.par_chunks_mut(naug))
        .zip(gstore.par_iter_mut())
        .enumerate()
        .for_each(|(j, ((mrow, drow), gs))| {
            let a_j = alpha[j];
            let s_j = s[j];
            // Fractional lower slope ⇒ the (M-D)*alpha product ROUNDS, which
            // the historical widening's derivation excluded — take the
            // magnitude-scaled branch (header doc). Binary rows keep the
            // historical line bit-identically.
            let frac_alpha = a_j != 0.0 && a_j != 1.0;
            let msr = &msrc[j * naug..(j + 1) * naug];
            let dsr = &dsrc[j * naug..(j + 1) * naug];
            // #tableau-support-mask: outside the mask `msr[i] == dsr[i] == 0.0`,
            // so `lo_c == up_c == 0.0` and both outputs are EXACTLY zero — which
            // is what the freshly zeroed `mo`/`do_` already hold. (The dense lane
            // instead stores `next_up(0.0) = 5e-324` into `drow[i]`; dropping
            // that subnormal makes `D` equal its exact value, so the tableau
            // stays a valid outward enclosure.) The bias column is always live.
            let mask = smask.map_or(u64::MAX, |s| s[j]);
            let mut acc = 0.0f64;
            for b in 0..SUPPORT_BLOCKS {
                if !support_has(mask, b) {
                    continue;
                }
                let lo = (b * blk).min(naug);
                let hi = (lo + blk).min(naug);
                for i in lo..hi {
                    let lo_c = (msr[i].wide() - dsr[i].wide()) * a_j;
                    let up0 = (msr[i].wide() + dsr[i].wide()) * s_j;
                    let mut up_c = up0;
                    if i == n_in {
                        up_c += c[j];
                    }
                    // Bit-identical (a+b)*0.5 anchor: midpoint's overflow-edge branch
                    // would move the produced center on this bound path.
                    #[allow(clippy::manual_midpoint)]
                    let mm = (lo_c + up_c) * 0.5;
                    let dd = (up_c - lo_c) * 0.5;
                    if mode.outward() {
                        // #tableau-f32: M' is CENTER-role (nearest store, class
                        // (c), charge into this same-site D widening); D' is
                        // RADIUS-role (directed-UP store, class (b)); the g_err
                        // charge is the EXACT site deviation `|dM| + |dD|`
                        // against the f64 site expression on this lane's own
                        // inputs (`d64` is bit-the-same value the f64 lane
                        // stores).
                        let (mm_stored, mm_charge) = T::store_center(mm);
                        mrow[i] = mm_stored;
                        if frac_alpha {
                            // Fractional-alpha widening (full derivation in the
                            // header doc): covers (†) standalone, regime-free.
                            let w =
                                lo_c.abs() + up_c.abs() + if i == n_in { up0.abs() } else { 0.0 };
                            let base = dd + UNIT * (mm.abs() + dd.abs() + 8.0 * w);
                            drow[i] = T::store_radius_up(next_up(T::fold_charge(base, mm_charge)));
                            if T::F32_STORE {
                                let d64 = next_up(base);
                                acc += ((mm_stored.wide() - mm).abs()
                                    + (drow[i].wide() - d64).abs())
                                    * xabs_ext[i];
                            }
                        } else {
                            // Historical binary-alpha line — bit-identical on the
                            // f64 lane (`fold_charge` is the untouched identity).
                            let base = dd + 6.0 * UNIT * (mm.abs() + dd);
                            drow[i] = T::store_radius_up(next_up(
                                T::fold_charge(base, mm_charge) * (1.0 + 1e-15),
                            ));
                            if T::F32_STORE {
                                let d64 = next_up(base * (1.0 + 1e-15));
                                acc += ((mm_stored.wide() - mm).abs()
                                    + (drow[i].wide() - d64).abs())
                                    * xabs_ext[i];
                            }
                        }
                    } else {
                        let mm_stored = T::store_center(mm).0;
                        mrow[i] = mm_stored;
                        drow[i] = T::store_radius_up(dd);
                        if T::F32_STORE {
                            acc += ((mm_stored.wide() - mm).abs() + (drow[i].wide() - dd).abs())
                                * xabs_ext[i];
                        }
                    }
                }
            }
            if T::F32_STORE {
                *gs = next_up(acc * acc_widen);
            }
        });
    (mo, do_, gstore)
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
