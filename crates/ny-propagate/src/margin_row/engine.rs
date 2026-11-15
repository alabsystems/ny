// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sparse margin-row CROWN backward over frozen gates (#twinwall).
//!
//! Port of the verified Python reference engine (validated against an
//! exact-rational falsifier harness during development),
//! functionally equivalent in `Parity` mode (selfchecked there vs
//! `core.crown_backward_rows` bit-identically, T1-T5), extended with a
//! CERTIFIED coefficient-error lane in `Outward` mode:
//!
//! * every coefficient matrix `L` is accompanied by an elementwise error
//!   matrix `E` with the invariant that the exact-real backward recursion
//!   applied to the *stored* upstream values stays within `L ± E`;
//! * ReLU sign-splits are exact (`|max(x,0)-max(y,0)| + |min(x,0)-min(y,0)|
//!   = |x-y|`), so `E` contracts by `max(alpha, s) <= 1` per gate and grows
//!   only by fresh rounding (`2u|L|` per elementwise step, Higham `gamma_n`
//!   per accumulation, `RHO_FOLD` per folded conv);
//! * concretization subtracts (lower) / adds (upper) the full error penalty
//!   `E . (|mid|+rad) + eb` plus a `gamma` envelope over the accumulation and
//!   rounds the final endpoint outward twice.
//!
//! The Outward lanes also charge an absolute binary64 underflow budget per
//! nominal multiplication and repair underflowing error products before later
//! source amplification. Grouped envelopes carry monotone upper budgets for
//! every column's lost product. Patch and GPU-row producers have separate
//! arithmetic and are not covered by that correction.
//!
//! Per-row single-neuron gate EXCEPTIONS reproduce the batched child-scoring
//! trick (engine.py T5: one pass scores all candidate children exactly).

use ndarray::Array2;
use ny_core::{NyError, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ops::Range;
use std::time::Instant;

use super::errlane::{self, EnvErr, ErrLane, PassLane};
use super::gpu_conv::{self, ConvLane, GpuConvRoute};
use super::net::{
    axpy, axpy_with_deadline, check_deadline, conv_apply_backward,
    conv_apply_backward_with_deadline, errconv_note_bias_rows, errconv_note_conc_rows,
    errconv_screens_enabled, error_conv_with_deadline, error_conv_without_deadline,
    nominal_product_dust, underflow_product_up, ConvOp, TwinNet, TwinOp,
};
use super::root::{LayerGates, RootGates};
use super::rounding::{certify_up, gamma_n, next_down, next_up, slack16, RoundMode, UNIT};
use super::support_program::{PreparedPrefix, PreparedPrefixFrame};

/// Backward direction of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneDir {
    /// Lower rows: concretize toward -inf.
    Lower,
    /// Upper rows: concretize toward +inf.
    Upper,
}

/// Per-layer dense gate override (piece-fixed trunk splits applied).
#[derive(Clone)]
pub struct GateVecs {
    /// Lower slopes.
    pub alpha: Vec<f64>,
    /// Upper slopes.
    pub s: Vec<f64>,
    /// Upper intercepts.
    pub c: Vec<f64>,
    /// `max(alpha, s)` (error contraction).
    pub ms: Vec<f64>,
}

/// #margin-row-beta: one trunk split's Lagrangian term for a backward pass.
///
/// For a domain whose region is `box ∩ {s_j * z_j(x) >= 0}` (per split), weak
/// duality gives, for ANY `beta_j >= 0`,
///
/// ```text
///   min_{region} f  >=  min_{region} [f - Σ_j beta_j * s_j * z_j]
///                  >=  min_{box}    relax(f - Σ_j beta_j * s_j * z_j)
/// ```
///
/// because `beta_j * s_j * z_j >= 0` on the region (so subtracting it never
/// raises the objective there), and the lane's relaxed backward pass +
/// concretize is a valid lower bound of any seeded functional over any set
/// containing the region. Symmetrically for the Upper lane with `+ Σ beta s z`.
/// A WRONG `beta` therefore costs tightness, never validity — the only
/// obligations are `beta >= 0` and the coefficient/error algebra at the
/// application site (see `apply_beta_terms`).
#[derive(Debug, Clone, Copy)]
pub struct BetaSplit {
    /// Absolute neuron index within the layer.
    pub neuron: usize,
    /// Split sign: `+1` = active branch (`z >= 0`), `-1` = inactive (`z <= 0`).
    pub sign: i8,
    /// Multiplier, must be finite and `> 0`. Callers filter zero terms: the
    /// engine also skips them so that a `beta = 0` entry cannot perturb a
    /// `-0.0` coefficient bit-wise.
    pub beta: f64,
}

/// #margin-row-beta-percol: one COLUMN's split-Lagrangian terms at a layer.
///
/// The per-column extension of [`BetaSplit`]: a domain carries one margin
/// objective per seed column, and each column's Lagrangian is independent —
/// weak duality holds PER COLUMN for any `beta >= 0` vector attached to that
/// column alone. The engine realizes a per-column term with the SAME certified
/// `apply_beta_terms` arithmetic, restricted to `col..col + 1` (exactly the
/// column-range shape the domain-stacked arm already exercises per block).
#[derive(Debug, Clone)]
pub struct PcBetaCol {
    /// Seed column (margin objective) these terms price.
    pub col: usize,
    /// The column's Lagrangian terms at this layer.
    pub terms: Vec<BetaSplit>,
}

/// Domain gate overrides keyed by trunk layer index.
#[derive(Default, Clone)]
pub struct DomainGates {
    /// Overridden layers.
    pub layers: BTreeMap<usize, GateVecs>,
    /// #margin-row-beta (`NY_MARGIN_ROW_BETA=1`): split-Lagrangian terms per
    /// trunk layer. Empty by default, in which case every pass is BIT-IDENTICAL
    /// to the pre-beta engine (the application site is skipped entirely).
    pub beta: BTreeMap<usize, Vec<BetaSplit>>,
    /// #margin-row-beta-percol (`NY_MARGIN_ROW_BETA_PERCOL=1`): PER-COLUMN
    /// split-Lagrangian terms per trunk layer. Applied AFTER `beta` (the two
    /// compose additively: a column's effective multiplier is the shared term
    /// plus its own — a sum of `beta >= 0` is a valid multiplier). Column
    /// indices are the SEED columns of the pass this gate set is built for;
    /// callers must never reuse a `beta_pc`-carrying dom on a pass with a
    /// different column layout (the candidate-score passes strip it). Empty by
    /// default ⇒ bit-identical passes.
    pub beta_pc: BTreeMap<usize, Vec<PcBetaCol>>,
    /// #clip-and-verify, the VERIFY half: this domain's region is EMPTY.
    ///
    /// Set when constrained re-concretization produces CROSSED bounds on any
    /// neuron: `l_new` under-estimates the constrained minimum and `u_new`
    /// over-estimates the constrained maximum (both pay their certified slack
    /// in the safe direction), so `l_new > u_new` forces `min > max` over
    /// `box ∩ halfspaces` — a Farkas certificate that the set is empty. The
    /// domain's true region is a SUBSET of that set (each halfspace is a
    /// necessary condition of its split), so an empty intersection discharges
    /// the domain with no counterexample possible inside it. The certificate
    /// is conservative: slack makes it strictly harder to trigger, never
    /// falsely.
    pub infeasible: bool,
    /// #mr-close-telemetry: when `infeasible`, the certificate's slack — the
    /// largest `l_new - u_new` crossing at the layer that discharged the
    /// domain. Never read by a bound; since #soundness-spine M2c it is the
    /// emptiness leaf's cover-gate EVIDENCE (a slack that is not finite and
    /// `> 0` is refused, which downgrades the tree), besides the
    /// proof-integrity receipt. `0.0` (the default) whenever `infeasible` is
    /// false.
    pub infeasible_slack: f64,
}

/// One domain's exclusive ownership of a contiguous column range in a
/// cross-domain backward pass.
///
/// `exceptions` use DOMAIN-LOCAL row numbers (`0..columns.len()`).  The
/// stacked entry point validates the ordered, gap-free partition and shifts
/// those rows into the combined matrix itself.  Keeping the shift here makes
/// it impossible for the caller to accidentally attach a candidate exception
/// to another domain's gate block.
pub(crate) struct RowDomainGateBlock<'g> {
    /// Exclusive combined-matrix columns owned by this domain.
    pub(crate) columns: Range<usize>,
    /// Piece-fixed trunk gates for exactly these columns.
    pub(crate) gates: &'g DomainGates,
    /// Per-candidate exceptions, numbered relative to `columns.start`.
    pub(crate) exceptions: &'g Exceptions,
}

/// One per-row gate exception (batched child scoring).
#[derive(Debug, Clone, Copy)]
pub struct Exc {
    /// Pass row the exception applies to.
    pub row: usize,
    /// Absolute neuron index within the layer.
    pub neuron: usize,
    /// Replacement lower slope.
    pub a2: f64,
    /// Replacement upper slope.
    pub s2: f64,
    /// Replacement intercept.
    pub c2: f64,
}

/// Exceptions grouped by trunk layer.
///
/// The ENGINE's contract is at most one exception per `(row, neuron, layer)`:
/// each entry snapshots its own `(orig, eorig)` cell before the in-place gate
/// transform and writes back to that same cell, so entries at distinct neurons
/// of the same row are independent. Two entries at the SAME `(row, neuron)`
/// would double-apply from a stale snapshot and charge the bias-error lane
/// twice.
///
/// The candidate-SCORING producers additionally keep at most one entry per
/// `(row, layer)`, and the GPU lane (`MAX_EXC_RECORDS`) and the domain-stacked
/// arm are sized against that narrower habit. `#spec-alpha` is the one producer
/// that uses the full `(row, neuron)` rank, which is why it validates its own
/// set (`spec_alpha::build_exceptions`) rather than relying on the habit — the
/// uniform arm below indexes `cur.l[[x.neuron, x.row]]` with NO validation of
/// its own, so a malformed set is a PANIC inside a verdict pass rather than a
/// wrong answer.
#[derive(Default)]
pub struct Exceptions {
    /// Layer index -> entries.
    pub by_layer: BTreeMap<usize, Vec<Exc>>,
}

/// Seed rows at the head pre-activation `y` (`(n_y, R)`, column r = row r).
pub struct Seed {
    /// Coefficients.
    pub s: Array2<f64>,
    /// Certified elementwise coefficient error (None = exact, e.g. identity).
    pub e: Option<Array2<f64>>,
}

/// What a backward pass collects at trunk relus (#epoch-bab).
#[derive(Default, Clone, Copy)]
pub struct Collect<'c> {
    /// Gather per-unstable |coef| sums (branching shortlist input).
    pub unst_abs: bool,
    /// Capture per-row coefficient vectors at each layer's RETAINED neurons
    /// (Tier-0 trunk-variant ranker input; nearest-mode values, ranker-only).
    pub rows: Option<&'c super::root::RetainedRows>,
    /// #alpha-opt: capture the PRE-transform incoming coefficients on the relu
    /// OUTPUT at every layer's UNSTABLE neurons (`(unst.len(), R)` per layer).
    ///
    /// GRADIENT-ONLY: these values never feed a verdict. They are the `v` of
    /// the gate transform `lv = vp*alpha + vn*s` below, captured so the alpha
    /// optimizer can read `vp = max(v, 0)` — the exact coefficient that
    /// multiplies `alpha_j` in this pass — without re-deriving the walk.
    pub unst_rows: bool,
}

/// Result of one backward pass: rows at the network input.
pub struct PassOut {
    /// Coefficients `(n_in, R)`.
    pub a: Array2<f64>,
    /// Certified elementwise error on `a` (Outward mode only).
    pub e: Option<Array2<f64>>,
    /// Bias per row.
    pub b: Vec<f64>,
    /// Certified bias error per row.
    pub eb: Vec<f64>,
    /// `collect_unst` output: layer -> per-unstable |coef| sums.
    pub coll: Option<BTreeMap<usize, Vec<f64>>>,
    /// Tier-0 capture: layer -> `(n_retained, R)` incoming coefficients on
    /// the relu OUTPUT at the layer's retained neurons (#epoch-bab).
    pub coll_rows: Option<BTreeMap<usize, Array2<f64>>>,
    /// #alpha-opt capture: layer -> `(unst.len(), R)` PRE-transform incoming
    /// coefficients at the layer's unstable neurons (gradient-only; see
    /// [`Collect::unst_rows`]). Row order follows `LayerGates::unst`.
    pub unst_rows: Option<BTreeMap<usize, Array2<f64>>>,
}

/// Producer provenance is issued only after a checked Outward prefix pair
/// completes. A nearest diagnostic PassOut cannot call the sealed converter.
/// This token is neither returned separately nor cloneable.
pub(super) struct DenseInputRowsAuthority(());

/// Arithmetic choice for a second lower-row candidate. Both slopes bound
/// ReLU globally; selecting a candidate never changes its authenticated scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CrossingLowerPolicy {
    Legacy,
    Zero,
}
impl CrossingLowerPolicy {
    pub(super) fn alpha(self, gate: &LayerGates, index: usize, baseline: f64) -> f64 {
        if self == Self::Zero && gate.l[index] < 0.0 && gate.u[index] > 0.0 {
            0.0
        } else {
            baseline
        }
    }
}

/// Arithmetic-only frozen reader. Gate access grants no source or product
/// authority; checked producer scopes independently authenticate publication.
pub(super) trait FrozenGateFrame: Sync {
    fn mode(&self) -> RoundMode;
    fn layer_count(&self) -> usize;
    fn layer(&self, index: usize) -> Result<&LayerGates>;
    fn lo(&self) -> &[f64];
    fn hi(&self) -> &[f64];
    fn mid(&self) -> &[f64];
    fn rad(&self) -> &[f64];
    fn xabs(&self) -> &[f64];
    fn crossing_lower_policy(&self) -> CrossingLowerPolicy {
        CrossingLowerPolicy::Legacy
    }
    fn checkpoint(&self) -> Result<()> {
        Ok(())
    }
    fn cooperative(&self) -> bool {
        false
    }
}

/// Borrows the same exact frame and deadline/cancellation lineage. No gate
/// arrays, proof owner or ledger are copied. Stable gates keep exact slopes;
/// the original ms also remains a conservative contraction for alpha=0.
pub(super) struct ZeroCrossingLowerFrame<'a>(pub(super) &'a dyn FrozenGateFrame);
impl FrozenGateFrame for ZeroCrossingLowerFrame<'_> {
    fn mode(&self) -> RoundMode {
        self.0.mode()
    }
    fn layer_count(&self) -> usize {
        self.0.layer_count()
    }
    fn layer(&self, index: usize) -> Result<&LayerGates> {
        self.0.layer(index)
    }
    fn lo(&self) -> &[f64] {
        self.0.lo()
    }
    fn hi(&self) -> &[f64] {
        self.0.hi()
    }
    fn mid(&self) -> &[f64] {
        self.0.mid()
    }
    fn rad(&self) -> &[f64] {
        self.0.rad()
    }
    fn xabs(&self) -> &[f64] {
        self.0.xabs()
    }
    fn crossing_lower_policy(&self) -> CrossingLowerPolicy {
        CrossingLowerPolicy::Zero
    }
    fn checkpoint(&self) -> Result<()> {
        self.0.checkpoint()
    }
    fn cooperative(&self) -> bool {
        self.0.cooperative()
    }
}
impl FrozenGateFrame for RootGates {
    fn mode(&self) -> RoundMode {
        self.mode
    }
    fn layer_count(&self) -> usize {
        self.layers.len()
    }
    fn layer(&self, index: usize) -> Result<&LayerGates> {
        self.layers
            .get(index)
            .ok_or_else(|| NyError::InvalidSpec("margin_row: missing frozen gate".into()))
    }
    fn lo(&self) -> &[f64] {
        &self.lo
    }
    fn hi(&self) -> &[f64] {
        &self.hi
    }
    fn mid(&self) -> &[f64] {
        &self.mid
    }
    fn rad(&self) -> &[f64] {
        &self.rad
    }
    fn xabs(&self) -> &[f64] {
        &self.xabs
    }
}
impl FrozenGateFrame for PreparedPrefixFrame<'_> {
    fn mode(&self) -> RoundMode {
        self.mode()
    }
    fn layer_count(&self) -> usize {
        self.layers().len()
    }
    fn layer(&self, index: usize) -> Result<&LayerGates> {
        self.gate(index)
            .ok_or_else(|| NyError::InvalidSpec("margin_row: missing prepared gate".into()))
    }
    fn lo(&self) -> &[f64] {
        self.root().lo()
    }
    fn hi(&self) -> &[f64] {
        self.root().hi()
    }
    fn mid(&self) -> &[f64] {
        self.root().mid()
    }
    fn rad(&self) -> &[f64] {
        self.root().rad()
    }
    fn xabs(&self) -> &[f64] {
        self.root().xabs()
    }
    fn checkpoint(&self) -> Result<()> {
        self.checkpoint()
    }
    fn cooperative(&self) -> bool {
        true
    }
}
struct BackwardKernel<'a> {
    net: &'a TwinNet,
    frame: &'a dyn FrozenGateFrame,
    gpu_conv: &'a GpuConvRoute,
}

/// The immutable ordinary projection keeps all numerical premises alive.
/// Full source/provider validation is paid by the producer before and after
/// the walk; arithmetic polling checks only the original ledger and clock.
struct PreparedOrdinaryFrame<'a> {
    adapter: &'a super::support_program::PreparedRootAdapter,
    deadline: Instant,
}
impl FrozenGateFrame for PreparedOrdinaryFrame<'_> {
    fn mode(&self) -> RoundMode {
        self.adapter.gates().mode
    }
    fn layer_count(&self) -> usize {
        self.adapter.gates().layers.len()
    }
    fn layer(&self, index: usize) -> Result<&LayerGates> {
        self.adapter.gates().layers.get(index).ok_or_else(|| {
            NyError::InvalidSpec("margin_row: missing prepared ordinary gate".into())
        })
    }
    fn lo(&self) -> &[f64] {
        &self.adapter.gates().lo
    }
    fn hi(&self) -> &[f64] {
        &self.adapter.gates().hi
    }
    fn mid(&self) -> &[f64] {
        &self.adapter.gates().mid
    }
    fn rad(&self) -> &[f64] {
        &self.adapter.gates().rad
    }
    fn xabs(&self) -> &[f64] {
        &self.adapter.gates().xabs
    }
    fn checkpoint(&self) -> Result<()> {
        check_optional_deadline(Some(self.deadline))?;
        self.adapter
            .source()
            .budget()
            .checkpoint_for(crate::certify::work::LeaseClass::Fallback)
            .map_err(|error| match error {
                crate::certify::work::WorkRefusal::Deadline => NyError::DeadlineExceeded(
                    "margin_row: prepared ordinary resource deadline".into(),
                ),
                _ => NyError::UnsupportedConfiguration(format!(
                    "margin_row: prepared ordinary resource refusal: {error}"
                )),
            })
    }
    fn cooperative(&self) -> bool {
        true
    }
}

/// Actual ordinary y-row numerical pass on an authenticated immutable root.
/// Its producer admits the entire seed/frontier/output overlap first. This
/// arithmetic helper cannot mint rows or publication authority from a seed.
/// Unlike the legacy deadline wrapper's whole-head limit, paid original
/// heads are covered by bounded tiles; column/cell/operator limits remain.
pub(in crate::margin_row) fn run_prepared_ordinary_tile(
    adapter: &super::support_program::PreparedRootAdapter,
    seed: &Seed,
    direction: LaneDir,
    requested: Instant,
) -> Result<PassOut> {
    let deadline = adapter.deadline(Some(requested))?;
    adapter.checkpoint(Some(deadline))?;
    let engine = BackwardEngine::new(adapter.net(), adapter.gates());
    let net = engine.net;
    let columns = seed.s.ncols();
    if engine.err_lane != ErrLane::PerCol
        || !engine.gpu_conv.is_off()
        || columns == 0
        || columns > DEADLINE_MAX_COLS
        || net.ops.len() > DEADLINE_MAX_OPS
        || seed.s.nrows() != net.n_y
        || seed.s.as_slice().is_none()
        || seed
            .e
            .as_ref()
            .is_some_and(|e| e.dim() != seed.s.dim() || e.as_slice().is_none())
        || net.tsize[..=net.i_gemm1]
            .iter()
            .chain(std::iter::once(&net.n_y))
            .any(|&n| {
                n.checked_mul(columns)
                    .is_none_or(|c| c > DEADLINE_MAX_CELLS)
            })
    {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: prepared ordinary pass requires bounded per-column CPU tiles".into(),
        ));
    }
    let frame = PreparedOrdinaryFrame { adapter, deadline };
    let kernel = BackwardKernel {
        net,
        frame: &frame,
        gpu_conv: &engine.gpu_conv,
    };
    ny_core::require_f64_interval_proof_environment()?;
    for (i, op) in net.ops.iter().enumerate() {
        kernel.poll::<true>(Some(deadline), i)?;
        if let TwinOp::Conv(c) = op {
            if gamma_n(c.k_bwd.checked_add(8).ok_or_else(|| {
                NyError::InvalidSpec("margin_row: ordinary proof length overflow".into())
            })?) > 0.25
            {
                return Err(NyError::UnsupportedConfiguration(
                    "margin_row: ordinary convolution proof length exceeds envelope".into(),
                ));
            }
        }
    }
    kernel.finite_with_deadline(seed.s.as_slice().expect("checked"), Some(deadline))?;
    if let Some(errors) = &seed.e {
        for (i, &e) in errors.as_slice().expect("checked").iter().enumerate() {
            kernel.poll::<true>(Some(deadline), i)?;
            if !e.is_finite() || e < 0.0 {
                return Err(NyError::InvalidSpec(
                    "margin_row: ordinary seed errors must be finite and nonnegative".into(),
                ));
            }
        }
    }
    for qualified in rayon::broadcast(|_| {
        frame.checkpoint()?;
        ny_core::require_f64_interval_proof_environment()
    }) {
        qualified?;
    }
    let output = kernel.run_collect_lane::<true>(
        seed,
        None,
        direction,
        None,
        Collect::default(),
        None,
        None,
        PassLane::PerCol,
        None,
        Some(deadline),
    )?;
    adapter.checkpoint(Some(deadline))?;
    ny_core::require_f64_interval_proof_environment()?;
    errlane::note_pass(PassLane::PerCol, columns, None);
    Ok(output)
}

/// Arithmetic-only conditional y-row tile on a sealed original phase domain.
/// The producer admits its seed, changed gates, walk and output overlap. The
/// immutable baseline frame and sparse overrides are each applied once;
/// returned arrays acquire child authority only in that producer's owner.
pub(in crate::margin_row) fn run_prepared_domain_tile(
    frame: &super::support_program::PreparedDomainFrame,
    seed: &Seed,
    direction: LaneDir,
    requested: Instant,
) -> Result<PassOut> {
    let deadline = frame.deadline(Some(requested))?;
    frame.checkpoint(Some(deadline))?;
    let net = frame.domain().adapter().net();
    let columns = seed.s.ncols();
    if columns == 0
        || columns > DEADLINE_MAX_COLS
        || net.ops.len() > DEADLINE_MAX_OPS
        || seed.s.nrows() != net.n_y
        || seed.s.as_slice().is_none()
        || seed
            .e
            .as_ref()
            .is_some_and(|e| e.dim() != seed.s.dim() || e.as_slice().is_none())
        || net.tsize[..=net.i_gemm1]
            .iter()
            .chain(std::iter::once(&net.n_y))
            .any(|&n| {
                n.checked_mul(columns)
                    .is_none_or(|c| c > DEADLINE_MAX_CELLS)
            })
    {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: prepared domain pass requires bounded per-column CPU tiles".into(),
        ));
    }
    let route = GpuConvRoute::Off;
    let kernel = BackwardKernel {
        net,
        frame,
        gpu_conv: &route,
    };
    ny_core::require_f64_interval_proof_environment()?;
    for (i, op) in net.ops.iter().enumerate() {
        kernel.poll::<true>(Some(deadline), i)?;
        if let TwinOp::Conv(c) = op {
            if gamma_n(c.k_bwd.checked_add(8).ok_or_else(|| {
                NyError::InvalidSpec("margin_row: domain proof length overflow".into())
            })?) > 0.25
            {
                return Err(NyError::UnsupportedConfiguration(
                    "margin_row: domain convolution proof length exceeds envelope".into(),
                ));
            }
        }
    }
    kernel.finite_with_deadline(seed.s.as_slice().expect("checked"), Some(deadline))?;
    if let Some(errors) = &seed.e {
        for (i, &e) in errors.as_slice().expect("checked").iter().enumerate() {
            kernel.poll::<true>(Some(deadline), i)?;
            if !e.is_finite() || e < 0.0 {
                return Err(NyError::InvalidSpec(
                    "margin_row: domain seed errors must be finite and nonnegative".into(),
                ));
            }
        }
    }
    for qualified in rayon::broadcast(|_| {
        FrozenGateFrame::checkpoint(frame)?;
        ny_core::require_f64_interval_proof_environment()
    }) {
        qualified?;
    }
    let output = kernel.run_collect_lane::<true>(
        seed,
        Some(frame.overrides()),
        direction,
        None,
        Collect::default(),
        None,
        None,
        PassLane::PerCol,
        None,
        Some(deadline),
    )?;
    frame.checkpoint(Some(deadline))?;
    ny_core::require_f64_interval_proof_environment()?;
    errlane::note_pass(PassLane::PerCol, columns, None);
    Ok(output)
}

/// Arithmetic-only head-seeded tile on a genuinely completed original trunk.
/// The terminal producer admits the full seed/walk/output overlap before it
/// calls this helper. Returned rows are actual kernel output, not a terminal
/// certificate: arbitrary internal seeds cannot mint publication authority.
pub(in crate::margin_row) fn run_prepared_terminal_tile(
    source: &super::support_program::PreparedRootConstruction,
    seed: &Seed,
    direction: LaneDir,
    requested: Instant,
) -> Result<PassOut> {
    let deadline = source.deadline(Some(requested))?;
    let prefix = source.prefix();
    let root = prefix.root();
    let net = root.model().net();
    let columns = seed.s.ncols();
    if root.policy().err_lane != ErrLane::PerCol
        || prefix.layer_count() != net.trunk_relus.len()
        || columns == 0
        || columns > DEADLINE_MAX_COLS
        || net.ops.len() > DEADLINE_MAX_OPS
        || seed.s.nrows() != net.n_y
        || seed.s.as_slice().is_none()
        || seed
            .e
            .as_ref()
            .is_some_and(|e| e.dim() != seed.s.dim() || e.as_slice().is_none())
        || net.tsize[..=net.i_gemm1]
            .iter()
            .chain(std::iter::once(&net.n_y))
            .any(|&n| {
                n.checked_mul(columns)
                    .is_none_or(|cells| cells > DEADLINE_MAX_CELLS)
            })
    {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: prepared terminal requires bounded per-column CPU tiles".into(),
        ));
    }
    let (input, width) = match net.ops.get(net.i_gemm1) {
        Some(TwinOp::Gemm { input, shape, .. }) => (*input, shape.1),
        _ => {
            return Err(NyError::InvalidSpec(
                "margin_row: missing original head".into(),
            ))
        }
    };
    if !source.trunk_output().root().same_root(root)
        || source.trunk_output().tensor() != input
        || source.trunk_output().lo().len() != width
        || source.trunk_output().hi().len() != width
    {
        return Err(NyError::InvalidSpec(
            "margin_row: completed terminal head input changed".into(),
        ));
    }
    ny_core::require_f64_interval_proof_environment()?;
    let frame = prefix.frame(Some(deadline))?;
    let route = GpuConvRoute::Off;
    let kernel = BackwardKernel {
        net,
        frame: &frame,
        gpu_conv: &route,
    };
    kernel.finite_with_deadline(seed.s.as_slice().expect("checked"), Some(deadline))?;
    if let Some(error) = &seed.e {
        for (i, &value) in error.as_slice().expect("checked").iter().enumerate() {
            kernel.poll::<true>(Some(deadline), i)?;
            if !value.is_finite() || value < 0.0 {
                return Err(NyError::InvalidSpec(
                    "margin_row: terminal seed error must be finite and nonnegative".into(),
                ));
            }
        }
    }
    for qualified in rayon::broadcast(|_| {
        frame.checkpoint()?;
        ny_core::require_f64_interval_proof_environment()
    }) {
        qualified?;
    }
    let output = kernel.run_collect_lane::<true>(
        seed,
        None,
        direction,
        None,
        Collect::default(),
        None,
        None,
        PassLane::PerCol,
        None,
        Some(deadline),
    )?;
    frame.checkpoint()?;
    ny_core::require_f64_interval_proof_environment()?;
    errlane::note_pass(PassLane::PerCol, columns, None);
    Ok(output)
}

/// Produce complete input-relative rows for the next original trunk ReLU.
/// The immutable prefix authenticates the actual model/domain/gates; all
/// dense allocations use its existing work ledger and memory authority.
/// Large tiles decline so a separately admitted patch producer can serve them.
pub(super) fn run_prepared_prefix_input_rows(
    prefix: &PreparedPrefix,
    targets: &[usize],
    requested: Option<Instant>,
) -> Result<super::input_rows::InputRowPair<'static>> {
    use super::input_rows::{InputRowPair, InputRowScope, OriginalRowDemand};
    use crate::certify::work::{LeaseClass, WorkRequest};
    let root = prefix.root();
    let budget = root.budget();
    let deadline = prefix.deadline(requested)?.ok_or_else(|| {
        NyError::UnsupportedConfiguration(
            "margin_row: prepared dense rows require inherited finite deadline".into(),
        )
    })?;
    root.deadline(Some(deadline))?;
    ny_core::require_f64_interval_proof_environment()?;
    if root.policy().err_lane != ErrLane::PerCol {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: prepared dense rows require frozen per-column policy".into(),
        ));
    }
    let net = root.model().net();
    let relu_op = *net
        .trunk_relus
        .get(prefix.layer_count())
        .ok_or_else(|| NyError::InvalidSpec("margin_row: no next original trunk ReLU".into()))?;
    let (input, n_pre) = prefix.for_relu(relu_op)?;
    if targets.is_empty() || targets.len() > DEADLINE_MAX_COLS || net.ops.len() > DEADLINE_MAX_OPS {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: prepared dense tile exceeds bounded CPU policy".into(),
        ));
    }
    let r = targets.len();
    let overflow = || {
        NyError::UnsupportedConfiguration("margin_row: prepared dense admission overflow".into())
    };
    // Metadata inspection is itself admitted before scanning an original DAG.
    let header_work = net
        .ops
        .len()
        .checked_mul(256)
        .and_then(|n| n.checked_add(1024))
        .ok_or_else(overflow)?;
    let mut header = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: u64::try_from(header_work).map_err(|_| overflow())?,
                ..WorkRequest::default()
            },
        )
        .map_err(prepared_refusal)?;
    let deadline = header.deadline().map_or(deadline, |d| d.min(deadline));
    let mut request = prepared_dense_request(net, relu_op, n_pre, r, deadline, prefix)?;
    if prefix.layer_count() != 0 {
        // The extra lower pass overlaps the complete first lower output. Its
        // actual A/E/b/eb payload and bounded selection traversal are admitted
        // before either candidate is allocated, on the original same ledger.
        let candidate_bytes = net
            .n_in
            .checked_mul(r)
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| r.checked_mul(16).and_then(|b| n.checked_add(b)))
            .and_then(|n| n.checked_add(size_of::<PassOut>()))
            .ok_or_else(overflow)?;
        request.scratch_bytes = request
            .scratch_bytes
            .checked_add(candidate_bytes)
            .ok_or_else(overflow)?;
        request.work = request
            .work
            .checked_mul(2)
            .and_then(|n| {
                net.n_in
                    .checked_mul(r)
                    .and_then(|c| c.checked_mul(512))
                    .and_then(|c| u64::try_from(c).ok())
                    .and_then(|c| n.checked_add(c))
            })
            .ok_or_else(overflow)?;
    }
    let scope =
        InputRowScope::from_prepared(prefix, OriginalRowDemand { relu_op, targets }, deadline)?;
    let deadline = deadline.min(scope.deadline());
    let frame_bytes = prefix
        .layer_count()
        .checked_mul(size_of::<&LayerGates>())
        .and_then(|n| n.checked_add(size_of::<PreparedPrefixFrame<'_>>()))
        .ok_or_else(overflow)?;
    let required = scope
        .resident_floor(deadline)?
        .checked_add(request.scratch_bytes)
        .and_then(|n| n.checked_add(request.persistent_bytes))
        .and_then(|n| n.checked_add(frame_bytes))
        .ok_or_else(overflow)?;
    if required > super::root_box::ROOT_SOURCE_BYTES_CAP {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes: super::root_box::ROOT_SOURCE_BYTES_CAP,
            site: "margin_row prepared dense source total live",
        });
    }
    let mut lease = budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(prepared_refusal)?;
    let deadline = lease.deadline().map_or(deadline, |d| d.min(deadline));
    let frame = prefix.frame(Some(deadline))?;
    for qualified in rayon::broadcast(|_| {
        frame.checkpoint()?;
        ny_core::require_f64_interval_proof_environment()
    }) {
        qualified?;
    }
    let mut seed = Seed {
        s: zeros_with_checkpoint((n_pre, r), Some(deadline), Some(&|| frame.checkpoint()))?,
        e: None,
    };
    for (column, &target) in scope.targets().iter().enumerate() {
        frame.checkpoint()?;
        seed.s[[target, column]] = 1.0;
    }
    if net.tsize.get(input) != Some(&n_pre) {
        return Err(NyError::InvalidSpec(
            "margin_row: changed prepared seed geometry".into(),
        ));
    }
    lease.mark_materialized().map_err(prepared_refusal)?;
    let route = GpuConvRoute::Off;
    let kernel = BackwardKernel {
        net,
        frame: &frame,
        gpu_conv: &route,
    };
    let mut lower = kernel.run_collect_lane::<true>(
        &seed,
        None,
        LaneDir::Lower,
        None,
        Collect::default(),
        None,
        Some(relu_op),
        PassLane::PerCol,
        None,
        Some(deadline),
    )?;
    errlane::note_pass(PassLane::PerCol, r, None);
    frame.checkpoint()?;
    if prefix.layer_count() != 0 {
        let zero_frame = ZeroCrossingLowerFrame(&frame);
        let alternative_kernel = BackwardKernel {
            net,
            frame: &zero_frame,
            gpu_conv: &route,
        };
        let mut alternative = alternative_kernel.run_collect_lane::<true>(
            &seed,
            None,
            LaneDir::Lower,
            None,
            Collect::default(),
            None,
            Some(relu_op),
            PassLane::PerCol,
            None,
            Some(deadline),
        )?;
        errlane::note_pass(PassLane::PerCol, r, None);
        select_prepared_lower_columns(&frame, &mut lower, &mut alternative, deadline)?;
        // Complete planes move together; the losing payload is released
        // before the unchanged upper pass allocates its output.
        drop(alternative);
        frame.checkpoint()?;
    }
    let upper = kernel.run_collect_lane::<true>(
        &seed,
        None,
        LaneDir::Upper,
        None,
        Collect::default(),
        None,
        Some(relu_op),
        PassLane::PerCol,
        None,
        Some(deadline),
    )?;
    errlane::note_pass(PassLane::PerCol, r, None);
    frame.checkpoint()?;
    drop(seed);
    drop(frame);
    root.deadline(Some(deadline))?;
    header.complete().map_err(prepared_refusal)?;
    drop(header);
    let resident = lease.commit_persistent().map_err(prepared_refusal)?;
    // The private authority is minted only here, after both complete certified
    // CPU passes. The converter checks actual arrays and exact ledger identity.
    InputRowPair::from_prepared_dense(
        scope,
        lower,
        upper,
        DenseInputRowsAuthority(()),
        resident,
        deadline,
    )
}

fn prepared_refusal(error: crate::certify::work::WorkRefusal) -> NyError {
    if error == crate::certify::work::WorkRefusal::Deadline {
        NyError::DeadlineExceeded("margin_row: prepared dense inherited deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!(
            "margin_row: prepared dense resource refusal: {error}"
        ))
    }
}

/// Arithmetic-only directed original-box evaluation for choosing two genuine
/// producer candidates. The caller has already paid the scan and every live
/// payload; this helper allocates nothing and mints no source authority.
pub(super) fn prepared_lower_bound_values(
    frame: &dyn FrozenGateFrame,
    bias: f64,
    bias_error: f64,
    entries: impl Iterator<Item = (usize, f64, f64)>,
    deadline: Instant,
) -> Result<f64> {
    frame.checkpoint()?;
    check_deadline(deadline)?;
    ny_core::require_f64_interval_proof_environment()?;
    if !bias.is_finite() || !bias_error.is_finite() || bias_error < 0.0 {
        return Err(NyError::InvalidSpec(
            "margin_row: invalid prepared lower candidate bias".into(),
        ));
    }
    let mut nominal = bias;
    let mut error = bias_error;
    for (visit, (id, coefficient, coefficient_error)) in entries.enumerate() {
        if visit.is_multiple_of(DEADLINE_POLL_ITEMS) {
            frame.checkpoint()?;
            check_deadline(deadline)?;
        }
        let (&lo, &hi) = frame.lo().get(id).zip(frame.hi().get(id)).ok_or_else(|| {
            NyError::InvalidSpec("margin_row: prepared candidate input mapping".into())
        })?;
        if !lo.is_finite()
            || !hi.is_finite()
            || lo > hi
            || !coefficient.is_finite()
            || !coefficient_error.is_finite()
            || coefficient_error < 0.0
        {
            return Err(NyError::InvalidSpec(
                "margin_row: invalid prepared candidate cell".into(),
            ));
        }
        let endpoint = if coefficient >= 0.0 { lo } else { hi };
        let term = if coefficient == 0.0 || endpoint == 0.0 {
            coefficient * endpoint
        } else {
            next_down(coefficient * endpoint)
        };
        nominal = if nominal == 0.0 {
            term
        } else if term == 0.0 {
            nominal
        } else {
            next_down(nominal + term)
        };
        let magnitude = lo.abs().max(hi.abs());
        let penalty = if coefficient_error == 0.0 || magnitude == 0.0 {
            0.0
        } else {
            next_up(coefficient_error * magnitude)
        };
        if penalty != 0.0 {
            error = next_up(error + penalty);
        }
        if !nominal.is_finite() || !error.is_finite() {
            return Err(NyError::InvalidSpec(
                "margin_row: unrepresentable prepared candidate bound".into(),
            ));
        }
    }
    let bound = if error == 0.0 {
        nominal
    } else {
        next_down(nominal - error)
    };
    frame.checkpoint()?;
    check_deadline(deadline)?;
    if !bound.is_finite() {
        return Err(NyError::InvalidSpec(
            "margin_row: unrepresentable prepared candidate endpoint".into(),
        ));
    }
    Ok(bound)
}

fn select_prepared_lower_columns(
    frame: &dyn FrozenGateFrame,
    baseline: &mut PassOut,
    alternative: &mut PassOut,
    deadline: Instant,
) -> Result<()> {
    let shape = baseline.a.dim();
    if shape != alternative.a.dim()
        || shape.0 != frame.lo().len()
        || baseline.e.as_ref().is_none_or(|e| e.dim() != shape)
        || alternative.e.as_ref().is_none_or(|e| e.dim() != shape)
        || baseline.b.len() != shape.1
        || baseline.eb.len() != shape.1
        || alternative.b.len() != shape.1
        || alternative.eb.len() != shape.1
    {
        return Err(NyError::InvalidSpec(
            "margin_row: prepared lower candidate geometry".into(),
        ));
    }
    for column in 0..shape.1 {
        frame.checkpoint()?;
        check_deadline(deadline)?;
        let bound = |out: &PassOut| {
            prepared_lower_bound_values(
                frame,
                out.b[column],
                out.eb[column],
                (0..shape.0).map(|row| {
                    (
                        row,
                        out.a[[row, column]],
                        out.e.as_ref().expect("checked")[[row, column]],
                    )
                }),
                deadline,
            )
        };
        let better = bound(alternative)? > bound(baseline)?;
        if better {
            let baseline_error = baseline.e.as_mut().expect("checked");
            let alternative_error = alternative.e.as_mut().expect("checked");
            for row in 0..shape.0 {
                if row.is_multiple_of(DEADLINE_POLL_ITEMS) {
                    frame.checkpoint()?;
                    check_deadline(deadline)?;
                }
                std::mem::swap(
                    &mut baseline.a[[row, column]],
                    &mut alternative.a[[row, column]],
                );
                std::mem::swap(
                    &mut baseline_error[[row, column]],
                    &mut alternative_error[[row, column]],
                );
            }
            std::mem::swap(&mut baseline.b[column], &mut alternative.b[column]);
            std::mem::swap(&mut baseline.eb[column], &mut alternative.eb[column]);
        }
    }
    Ok(())
}

fn prepared_dense_request(
    net: &TwinNet,
    relu_op: usize,
    seed_rows: usize,
    columns: usize,
    deadline: Instant,
    prefix: &PreparedPrefix,
) -> Result<crate::certify::work::WorkRequest> {
    let overflow = || {
        NyError::UnsupportedConfiguration(
            "margin_row: prepared dense capacity/work overflow".into(),
        )
    };
    let mut prefix_rows = 0usize;
    for (i, &rows) in net.tsize[..=relu_op].iter().enumerate() {
        if i.is_multiple_of(DEADLINE_POLL_ITEMS) {
            prefix.deadline(Some(deadline))?;
        }
        let cells = rows.checked_mul(columns).ok_or_else(overflow)?;
        if cells > DEADLINE_MAX_CELLS {
            return Err(NyError::UnsupportedConfiguration(
                "margin_row: prepared dense tile exceeds cell policy".into(),
            ));
        }
        prefix_rows = prefix_rows.checked_add(rows).ok_or_else(overflow)?;
    }
    let persistent = net
        .n_in
        .checked_mul(columns)
        .and_then(|n| n.checked_mul(32))
        .and_then(|n| columns.checked_mul(32).and_then(|b| n.checked_add(b)))
        .and_then(|n| n.checked_add(super::input_rows::INPUT_ROWS_OWNER_BYTES))
        .ok_or_else(overflow)?;
    // All possible live residual states plus removed current/destination/error
    // temporaries; the persistent claim separately holds the entire lower pass
    // while upper arithmetic runs. Seed and every Rayon accumulator are paid.
    let mut scratch = prefix_rows
        .checked_mul(columns)
        .and_then(|n| n.checked_mul(48))
        .ok_or_else(overflow)?;
    scratch = seed_rows
        .checked_mul(columns)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| scratch.checked_add(n))
        .ok_or_else(overflow)?;
    scratch = net
        .ops
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(size_of::<Option<LaneMat>>()))
        .and_then(|n| scratch.checked_add(n))
        .ok_or_else(overflow)?;
    scratch = scratch.checked_add(16 * 1024).ok_or_else(overflow)?;
    let mut max_acc = 0usize;
    let mut work = 4096usize;
    for (op, original) in net.ops.iter().enumerate() {
        prefix.deadline(Some(deadline))?;
        let add = |total: &mut usize, n: usize, scale: usize| -> Result<()> {
            *total = n
                .checked_mul(scale)
                .and_then(|n| total.checked_add(n))
                .ok_or_else(overflow)?;
            Ok(())
        };
        add(&mut work, 1, 512)?;
        if let TwinOp::Conv(c) = original {
            add(&mut work, c.back_taps.len(), 64)?;
            if op < relu_op {
                max_acc = max_acc.max(c.kernel.1.checked_mul(columns).ok_or_else(overflow)?);
                add(
                    &mut work,
                    net.tsize[c.input]
                        .checked_mul(columns)
                        .ok_or_else(overflow)?,
                    c.k_bwd
                        .checked_add(1)
                        .and_then(|n| n.checked_mul(64))
                        .ok_or_else(overflow)?,
                )?;
            }
        } else if op < relu_op {
            add(
                &mut work,
                net.tsize[op + 1]
                    .checked_mul(columns)
                    .ok_or_else(overflow)?,
                256,
            )?;
        }
    }
    let workers = rayon::current_num_threads();
    scratch = workers
        .checked_mul(
            max_acc
                .checked_mul(8)
                .and_then(|n| n.checked_add(4096))
                .ok_or_else(overflow)?,
        )
        .and_then(|n| scratch.checked_add(n))
        .ok_or_else(overflow)?;
    work = seed_rows
        .checked_mul(columns)
        .and_then(|n| n.checked_mul(64))
        .and_then(|n| work.checked_add(n))
        .ok_or_else(overflow)?;
    if scratch.checked_add(persistent).ok_or_else(overflow)?
        > super::root_box::ROOT_SOURCE_BYTES_CAP
    {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: prepared dense tile exceeds root-source ceiling".into(),
        ));
    }
    Ok(crate::certify::work::WorkRequest {
        scratch_bytes: scratch,
        persistent_bytes: persistent,
        work: u64::try_from(work).map_err(|_| overflow())?,
        proposals: 0,
    })
}

/// The engine: borrows the compiled net + frozen root gates.
pub struct BackwardEngine<'a> {
    /// Compiled net.
    pub net: &'a TwinNet,
    /// Frozen root gates + box.
    pub root: &'a RootGates,
    /// #mr-errlane (throughput plan B3): which certified coefficient-error
    /// lane this engine's Outward passes carry. Set ONCE at construction from
    /// the root's frozen `RootPrecisionRuntimePolicy::err_lane` (which the
    /// scored route resolves from the receipt-owning LeverSet key
    /// `margin_row.err_envelope`); never read from a global. `PerCol` (the
    /// default) keeps one corrected error bound for every coefficient.
    err_lane: ErrLane,
    /// Pass-width floor for the row envelope ([`errlane::ENV_MIN_COLS`];
    /// lowered only by unit tests so small fixtures can engage it).
    err_min_cols: usize,
    /// #mr-gpu-eval: where this engine's conv backward products run. Copied
    /// at construction from the process-latched typed key
    /// `margin_row.gpu_conv` ([`gpu_conv::lane_route`]); `Off` (the default)
    /// uses the original signed CPU kernel. Underflow-corrected coefficient
    /// error contractions use the CPU helpers independently of this route.
    gpu_conv: GpuConvRoute,
}

/// One tensor's lane state during a backward walk: coefficients, the
/// certified per-column error lane, or the envelope error.
///
/// `pub(crate)` (fields included) for `#bi-patch` (`patch.rs`), whose
/// per-column receptive-field walk carries the SAME state in patch-local rows
/// and merges it with the unchanged `merge_into`; visibility only.
pub(crate) struct LaneMat {
    pub(crate) l: Array2<f64>,
    pub(crate) e: Option<Array2<f64>>,
    /// #mr-errlane: the ENVELOPE error (`Some` only on a pass that publishes
    /// an envelope; then `e` is `None`). See `errlane.rs`.
    pub(crate) env: Option<EnvErr>,
}

impl<'a> BackwardEngine<'a> {
    /// New engine over a net + its root gates. The certified error lane is
    /// the root's frozen policy (`root.root_precision_policy.err_lane`).
    pub fn new(net: &'a TwinNet, root: &'a RootGates) -> Self {
        Self {
            net,
            root,
            err_lane: root.root_precision_policy.err_lane,
            err_min_cols: errlane::ENV_MIN_COLS,
            gpu_conv: gpu_conv::lane_route(),
        }
    }

    /// #mr-gpu-eval: override where this engine's conv backward products run
    /// (tests and research probes inject an explicit engine; `#bi-gpu` hands
    /// the backward-interm prefix pairs its scoped route; every other scored
    /// pass takes the lane route through [`Self::new`]).
    #[must_use]
    pub(crate) fn with_gpu_conv_route(mut self, route: GpuConvRoute) -> Self {
        self.gpu_conv = route;
        self
    }

    /// #mr-gpu-eval: this engine's conv backward route.
    #[must_use]
    pub(crate) fn gpu_conv_route(&self) -> &GpuConvRoute {
        &self.gpu_conv
    }

    /// Shared frozen-route dispatch for ordinary and patch backward products.
    /// Off skips the device seam; every refused device result is overwritten
    /// by the same CPU kernel. Borrowed route/rounding operands come from the
    /// caller's existing frame, without another route lookup or owner clone.
    pub(crate) fn conv_backward(
        route: &GpuConvRoute,
        cv: &ConvOp,
        src: &Array2<f64>,
        dst: &mut Array2<f64>,
        abs_w: bool,
        lane: ConvLane,
        outward: impl FnOnce() -> bool,
    ) {
        if route.is_off() || !gpu_conv::offload(route, cv, src, dst, abs_w, lane, outward()) {
            conv_apply_backward(cv, src, dst, abs_w);
        }
    }

    /// Override the certified error lane (research probes and tests; scored
    /// passes take the root's frozen policy through [`Self::new`]).
    #[must_use]
    pub fn with_err_lane(mut self, lane: ErrLane) -> Self {
        self.err_lane = lane;
        self
    }

    /// The certified error lane this engine carries.
    #[must_use]
    pub fn err_lane(&self) -> ErrLane {
        self.err_lane
    }

    /// Test-only: lower the row-envelope width floor so a small fixture
    /// (`R < ENV_MIN_COLS`) exercises the envelope arithmetic.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_err_min_cols(mut self, min_cols: usize) -> Self {
        self.err_min_cols = min_cols.max(1);
        self
    }

    fn outward(&self) -> bool {
        self.root.mode.outward()
    }

    /// Identity seed (for the 2 x n_y y-row refresh passes).
    pub fn identity_seed(&self) -> Seed {
        let n_y = self.net.n_y;
        let mut s = Array2::<f64>::zeros((n_y, n_y));
        for j in 0..n_y {
            s[[j, j]] = 1.0;
        }
        Seed { s, e: None }
    }

    /// One backward pass. `dom`: piece-fixed gate overrides; `exc`: per-row
    /// single-neuron exceptions; `collect_unst`: gather |coef| mass at
    /// unstable trunk neurons (branching shortlist input).
    pub fn run(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect_unst: bool,
    ) -> Result<PassOut> {
        self.run_collect(
            seed,
            dom,
            dir,
            exc,
            Collect {
                unst_abs: collect_unst,
                rows: None,
                unst_rows: false,
            },
        )
    }

    /// One source-owned CPU pass that discards every partial result on expiry.
    ///
    /// This optional root-certificate entry uses the same arbitrary signed seed
    /// columns and per-column arithmetic as [`Self::run`]. It refuses envelope
    /// publication policies, non-Outward roots, and oversized scratch before
    /// numerical work. Audit shadows, collections and GPU offload are absent.
    /// Cooperative checks bound arithmetic batches, not allocator, scheduler or
    /// Rayon-pool startup latency; the caller must also check its original
    /// instance deadline before accepting a complete certificate.
    pub(super) fn run_with_deadline(
        &self,
        seed: &Seed,
        dir: LaneDir,
        deadline: Instant,
    ) -> Result<PassOut> {
        check_deadline(deadline)?;
        let r = seed.s.ncols();
        if !self.outward()
            || !matches!(
                self.err_lane.publish(r, self.err_min_cols),
                PassLane::PerCol
            )
            || r == 0
            || r > DEADLINE_MAX_COLS
            || self.net.n_y > 512
            || self.net.ops.len() > DEADLINE_MAX_OPS
            || self.net.ops.iter().any(|op| {
                matches!(op, TwinOp::Conv(cv) if cv.k_bwd.checked_add(8)
                    .is_none_or(|n| gamma_n(n) > 0.25))
            })
            || self.net.tsize.iter().any(|&n| {
                n.checked_mul(r)
                    .is_none_or(|cells| cells > DEADLINE_MAX_CELLS)
            })
            || seed.s.nrows() != self.net.n_y
            || seed.s.as_slice().is_none()
            || seed
                .e
                .as_ref()
                .is_some_and(|e| e.raw_dim() != seed.s.raw_dim() || e.as_slice().is_none())
        {
            return Err(NyError::UnsupportedConfiguration(
                "margin_row: deadline pass requires bounded Outward per-column CPU rows".into(),
            ));
        }
        finite_with_deadline(seed.s.as_slice().expect("validated"), Some(deadline))?;
        if let Some(e) = seed.e.as_ref() {
            for (i, &v) in e.as_slice().expect("validated").iter().enumerate() {
                poll_deadline(Some(deadline), i)?;
                if !v.is_finite() || v < 0.0 {
                    return Err(NyError::InvalidSpec(
                        "margin_row: deadline seed error must be finite and nonnegative".into(),
                    ));
                }
            }
        }
        ny_core::require_f64_interval_proof_environment()?;
        for qualified in rayon::broadcast(|_| {
            check_deadline(deadline)?;
            ny_core::require_f64_interval_proof_environment()
        }) {
            qualified?;
        }
        check_deadline(deadline)?;
        let out = self.run_collect_lane::<true>(
            seed,
            None,
            dir,
            None,
            Collect::default(),
            None,
            None,
            PassLane::PerCol,
            None,
            Some(deadline),
        )?;
        check_deadline(deadline)?;
        Ok(out)
    }

    /// One backward pass with the full collection spec (#epoch-bab).
    pub fn run_collect(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect: Collect<'_>,
    ) -> Result<PassOut> {
        self.run_collect_impl(seed, dom, dir, exc, collect, None, None, None)
    }

    /// #backward-interm: one certified backward pass seeded at the INPUT of
    /// trunk relu op `relu_op` (identity-style rows on its pre-activation),
    /// walked through the already-frozen PREFIX gates and concretizable over
    /// the root box with the lane's unchanged `concretize_*`.
    ///
    /// SOUNDNESS: the prefix relus' gates are the same frozen `LayerGates`
    /// every verdict pass uses (valid DeepPoly relaxations on their certified
    /// boxes), the certified error lane starts at zero (identity seed is
    /// exact) and contracts/grows by the unchanged per-op algebra, so the
    /// concretized rows are valid outward bounds on this relu's pre-activation
    /// over the root box — an independent enclosure the caller may intersect
    /// (shrink-only) with the forward tableau's.
    pub(crate) fn run_prefix(&self, seed: &Seed, relu_op: usize, dir: LaneDir) -> Result<PassOut> {
        self.run_collect_impl(
            seed,
            None,
            dir,
            None,
            Collect::default(),
            None,
            Some(relu_op),
            None,
        )
    }

    /// A complete CPU prefix pass under the caller's original absolute
    /// deadline. The source mode and per-column error lane are qualified
    /// before copying its seed or allocating reverse tensors. This creates no
    /// clipping authority by itself: the scoped product entry validates the
    /// original target identity before sealing its returned rows.
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn run_prefix_with_deadline(
        &self,
        seed: &Seed,
        relu_op: usize,
        dir: LaneDir,
        deadline: Instant,
    ) -> Result<PassOut> {
        check_deadline(deadline)?;
        let r = seed.s.ncols();
        let Some(TwinOp::Relu { input, .. }) = self.net.ops.get(relu_op) else {
            return Err(NyError::InvalidSpec(
                "margin_row: deadline prefix seed op is not a relu".into(),
            ));
        };
        let Some(&n_pre) = self.net.tsize.get(*input) else {
            return Err(NyError::InvalidSpec(
                "margin_row: deadline prefix seed tensor is missing".into(),
            ));
        };
        if !self.outward()
            || !self.gpu_conv.is_off()
            || !matches!(
                self.err_lane.publish(r, self.err_min_cols),
                PassLane::PerCol
            )
            || r == 0
            || r > DEADLINE_MAX_COLS
            || relu_op >= self.net.i_gemm1
            || self.net.ops.len() > DEADLINE_MAX_OPS
            || self.net.tsize.len() != self.net.ops.len() + 1
            || self.net.tsize[..=relu_op].iter().any(|&n| {
                n.checked_mul(r)
                    .is_none_or(|cells| cells > DEADLINE_MAX_CELLS)
            })
            || seed.s.nrows() != n_pre
            || seed.s.as_slice().is_none()
            || seed
                .e
                .as_ref()
                .is_some_and(|e| e.raw_dim() != seed.s.raw_dim() || e.as_slice().is_none())
        {
            return Err(NyError::UnsupportedConfiguration(
                "margin_row: deadline prefix requires bounded Outward per-column CPU rows".into(),
            ));
        }
        finite_with_deadline(seed.s.as_slice().expect("validated"), Some(deadline))?;
        if let Some(e) = seed.e.as_ref() {
            for (i, &value) in e.as_slice().expect("validated").iter().enumerate() {
                poll_deadline(Some(deadline), i)?;
                if !value.is_finite() || value < 0.0 {
                    return Err(NyError::InvalidSpec(
                        "margin_row: deadline prefix seed error must be finite and nonnegative"
                            .into(),
                    ));
                }
            }
        }
        ny_core::require_f64_interval_proof_environment()?;
        for qualified in rayon::broadcast(|_| {
            check_deadline(deadline)?;
            ny_core::require_f64_interval_proof_environment()
        }) {
            qualified?;
        }
        check_deadline(deadline)?;
        self.run_collect_impl_cancellable::<true>(
            seed,
            None,
            dir,
            None,
            Collect::default(),
            None,
            Some(relu_op),
            None,
            Some(deadline),
        )
    }

    /// Conservative tile estimate, not a lease or global-route admission.
    /// `resident_bytes` must come from the caller's already-admitted ownership
    /// census, including original model/root, prior tiles, opaque caches and
    /// overlapping workers. The visible owner floor prevents under-reporting
    /// their directly accessible payloads; it cannot price hidden consumers.
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn dense_prefix_input_product_bytes(
        &self,
        seed: &Seed,
        relu_op: usize,
        resident_bytes: usize,
    ) -> Option<usize> {
        self.dense_prefix_input_product_bytes_from_base(
            seed,
            relu_op,
            resident_bytes.max(input_product_resident_floor(self.net, self.root)?),
        )
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    fn dense_prefix_input_product_bytes_from_base(
        &self,
        seed: &Seed,
        relu_op: usize,
        resident_bytes: usize,
    ) -> Option<usize> {
        let r = seed.s.ncols();
        let prefix_rows = self
            .net
            .tsize
            .get(..=relu_op)?
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))?;
        let seed_bytes = seed
            .s
            .len()
            .checked_add(seed.e.as_ref().map_or(0, Array2::len))?
            .checked_mul(size_of::<f64>())?;
        // Price every possible residual frontier plus the removed current
        // tensor, destination and combined-error temporaries. The whole
        // completed lower pass remains resident throughout the upper pass.
        let scratch = prefix_rows.checked_mul(r)?.checked_mul(48)?;
        let pair = self
            .net
            .n_in
            .checked_mul(r)?
            .checked_mul(32)?
            .checked_add(r.checked_mul(32)?)?;
        let metadata = self
            .net
            .ops
            .len()
            .checked_add(1)?
            .checked_mul(size_of::<Option<LaneMat>>())?
            .checked_add(r.checked_mul(16)?)?
            .checked_add(size_of::<super::input_rows::InputRowPair<'_>>())?;
        resident_bytes
            .checked_add(seed_bytes)?
            .checked_add(scratch)?
            .checked_add(pair)?
            .checked_add(metadata)
    }

    /// Return the producer's explicit input-relative lower and upper rows.
    /// The caller must already own a whole-route admission; these integers
    /// only check this conservative tile against the inherited cap and do
    /// not authorize retention, default capability or a new global budget.
    /// Scope and identity seeds are checked before reverse allocation. Both
    /// passes keep the original deadline and move their A/E/b/eb payloads
    /// directly into the sealed row interface, without M/D repacking.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn run_prefix_input_rows<'q>(
        &self,
        seed: &Seed,
        relu_op: usize,
        targets: &'q [usize],
        resident_bytes: usize,
        budget_bytes: usize,
        deadline: Instant,
    ) -> Result<super::input_rows::InputRowPair<'q>>
    where
        'a: 'q,
    {
        use super::input_rows::{InputRowPair, InputRowScope};
        const SITE: &str = "margin_row dense input-row product tile";
        check_deadline(deadline)?;
        let deadline = self
            .root
            .head_group_source
            .as_ref()
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "margin_row: input rows lack completed original builder authority".into(),
                )
            })?
            .input_rows_deadline(deadline)?;
        let scope = InputRowScope::new(self.net, self.root, relu_op, targets, deadline)?;
        let deadline = scope.deadline();
        let r = scope.columns();
        if !self.gpu_conv.is_off()
            || !matches!(
                self.err_lane.publish(r, self.err_min_cols),
                PassLane::PerCol
            )
            || r > DEADLINE_MAX_COLS
            || self.net.ops.len() > DEADLINE_MAX_OPS
            || self.net.tsize.len() != self.net.ops.len() + 1
            || self.net.tsize[..=relu_op].iter().any(|&n| {
                n.checked_mul(r)
                    .is_none_or(|cells| cells > DEADLINE_MAX_CELLS)
            })
        {
            return Err(NyError::UnsupportedConfiguration(
                "margin_row: input-row product requires bounded per-column CPU policy".into(),
            ));
        }
        let input = match self.net.ops.get(relu_op) {
            Some(TwinOp::Relu { input, .. }) => *input,
            _ => return Err(NyError::InvalidSpec("margin_row: changed prefix op".into())),
        };
        let n_pre =
            *self.net.tsize.get(input).ok_or_else(|| {
                NyError::InvalidSpec("margin_row: missing prefix seed shape".into())
            })?;
        if seed.s.dim() != (n_pre, r)
            || !seed.s.is_standard_layout()
            || seed
                .e
                .as_ref()
                .is_some_and(|e| e.dim() != seed.s.dim() || !e.is_standard_layout())
        {
            return Err(NyError::InvalidSpec(
                "margin_row: input-row seed/error shape/layout mismatch".into(),
            ));
        }
        for (i, &value) in seed.s.as_slice().expect("validated").iter().enumerate() {
            poll_deadline(Some(deadline), i)?;
            let expected = if i / r == targets[i % r] { 1.0 } else { 0.0 };
            if value != expected {
                return Err(NyError::InvalidSpec(
                    "margin_row: input-row seed is not the original target identity".into(),
                ));
            }
        }
        if let Some(error) = seed.e.as_ref() {
            for (i, &value) in error.as_slice().expect("validated").iter().enumerate() {
                poll_deadline(Some(deadline), i)?;
                if value != 0.0 {
                    return Err(NyError::InvalidSpec(
                        "margin_row: input-row identity seed must have exact zero error".into(),
                    ));
                }
            }
        }
        let budget_bytes = budget_bytes.min(super::root_box::ROOT_SOURCE_BYTES_CAP);
        let floor = input_product_resident_floor_with_deadline(self.net, self.root, deadline)?;
        let required_bytes = self
            .dense_prefix_input_product_bytes_from_base(seed, relu_op, resident_bytes.max(floor))
            .unwrap_or(usize::MAX);
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: SITE,
            });
        }
        let lower = self.run_prefix_with_deadline(seed, relu_op, LaneDir::Lower, deadline)?;
        let upper = self.run_prefix_with_deadline(seed, relu_op, LaneDir::Upper, deadline)?;
        check_deadline(deadline)?;
        InputRowPair::from_dense(scope, lower, upper, DenseInputRowsAuthority(()), deadline)
    }

    /// #interm-alpha: prefix pass under an explicit per-layer gate override.
    ///
    /// Identical to [`Self::run_prefix`] in every respect except that the
    /// prefix relus resolve their `(alpha, s, c, ms)` through `dom` where a
    /// layer is present. `dom = None` is BIT-IDENTICAL to [`Self::run_prefix`]
    /// (same arguments, same code path), so the pre-#interm-alpha behaviour is
    /// reproduced exactly by passing nothing.
    ///
    /// WHY THIS EXISTS. `backward_interm` computes each trunk relu's input box
    /// with a backward pass over the FROZEN prefix — abc's intermediate-bound
    /// computation — but abc holds a SEPARATE optimizable alpha set per
    /// (relu layer, downstream start node) (`alpha[relu][start_node]`, shape
    /// `[2, spec, batch, *relu_shape]`, verified in
    /// `references/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/relu.py:155`
    /// and `backward_bound.py:927-944`), while ny reuses the ONE shared
    /// `(layer, neuron)` heuristic alpha for every start node and both lanes.
    /// This entry point is what lets a start node choose its own.
    pub(crate) fn run_prefix_dom(
        &self,
        seed: &Seed,
        relu_op: usize,
        dir: LaneDir,
        dom: Option<&DomainGates>,
    ) -> Result<PassOut> {
        self.run_collect_impl(
            seed,
            dom,
            dir,
            None,
            Collect::default(),
            None,
            Some(relu_op),
            None,
        )
    }

    /// Cross-domain row stack used only by the default-off margin-row
    /// candidate-score canary.
    ///
    /// Every block owns one ordered, contiguous, non-empty range and the
    /// ranges must exactly partition all seed columns.  Exceptions are local
    /// to their block and are shifted only after validation.  Any malformed
    /// layout fails closed before entering verdict arithmetic.
    /// Cross-domain stacked pass with the DEFAULT collection spec.
    ///
    /// Preserved verbatim so the four existing call sites and the bit-identity
    /// assert at `bab.rs` are untouched; see
    /// [`Self::run_domain_stacked_collect`] for the general form.
    pub(crate) fn run_domain_stacked(
        &self,
        seed: &Seed,
        blocks: &[RowDomainGateBlock<'_>],
        dir: LaneDir,
    ) -> Result<PassOut> {
        self.run_domain_stacked_collect(
            seed,
            blocks,
            dir,
            Collect {
                unst_abs: false,
                rows: None,
                unst_rows: false,
            },
        )
    }

    /// Cross-domain stacked pass with an EXPLICIT collection spec.
    ///
    /// WHY THIS EXISTS. `run_domain_stacked` hardcoded
    /// `Collect { unst_abs: false, rows: None, unst_rows: false }`, which made
    /// the batched path unable to capture the per-unstable-neuron
    /// pre-transform coefficients — exactly the quantity a PER-DOMAIN alpha
    /// gradient needs. That turned the one affordable route to per-domain alpha
    /// (batching, measured at a constant ~2.4 ms/column, so 10 optimizer steps
    /// cost 24*nf ms/domain instead of 1.28 s) into a dead end by a literal
    /// rather than by a constraint.
    ///
    /// IT IS SAFE TO ASK FOR, and that is a structural fact rather than a
    /// hope: the `unst_rows` capture (see `run_collect_impl`) runs BEFORE the
    /// `domain_stack` branch, reads `cur.l` and emits an `(rec.unst.len(), r)`
    /// matrix where `r` is the TOTAL stacked column count. Every block owns a
    /// contiguous `block.columns` range, so a caller slices its own domain's
    /// columns straight out. The capture is documented gradient-only —
    /// "nothing verdict-bearing reads this" — so widening it cannot move a
    /// published bound; it costs memory and a memcpy.
    ///
    /// The default wrapper above keeps the shipped path byte-identical.
    pub(crate) fn run_domain_stacked_collect(
        &self,
        seed: &Seed,
        blocks: &[RowDomainGateBlock<'_>],
        dir: LaneDir,
        collect: Collect,
    ) -> Result<PassOut> {
        let r = seed.s.ncols();
        if seed
            .e
            .as_ref()
            .is_some_and(|e| e.raw_dim() != seed.s.raw_dim())
        {
            return Err(NyError::InvalidSpec(
                "margin_row: domain-stack seed/error shape mismatch".into(),
            ));
        }
        validate_domain_stack(self.root, blocks, r)?;
        let shifted = shift_block_exceptions(blocks);
        self.run_collect_impl(
            seed,
            None,
            dir,
            Some(&shifted),
            collect,
            Some(blocks),
            None,
            None,
        )
    }

    /// #mr-stack (margin-row throughput plan B5): ONE cross-domain stacked pass,
    /// returned as one [`PassOut`] PER BLOCK, each bit-identical to the pass that
    /// block's domain would have run alone over its own seed columns
    /// (`run_collect(seed_k, Some(gates_k), dir, exc_k, collect)`).
    ///
    /// WHY THIS EXISTS. The tree's main eval pass runs at R = nf = 1-3 columns
    /// on cifar100 (column_retire), where one pass costs far more than its
    /// column share; the B1 probe measured one stacked pass over K=8 domains at
    /// 0.15-0.20x the CPU of K separate `run_collect` calls, bit-identical, on
    /// all three probe nets. [`Self::run_domain_stacked_collect`] could already
    /// run such a pass, but its outputs were only usable column-sliced: the
    /// `unst_abs` branching mass summed across EVERY domain's columns. This
    /// entry point asks the engine for per-block sums and slices every other
    /// output (coefficients, error lane, bias, bias error, Tier-0 and
    /// `unst_rows` captures) back into per-domain passes.
    ///
    /// WHY EACH BLOCK IS BIT-IDENTICAL. Every op of the backward walk is
    /// column-local (elementwise gate transforms and error updates, per-column
    /// `axpy` folds with no FMA, the conv kernels whose column blocking never
    /// reorders a column's fold, per-element `certify_up`); the stacked ReLU arm
    /// resolves each column's gates, intercepts and β terms from its owning block
    /// in the uniform arm's order (see [`apply_domain_stacked_relu`]); the only
    /// cross-column reduction in a pass, the `unst_abs` sum, is taken per block
    /// here; and the captures are copies. Slicing copies values bit-for-bit.
    ///
    /// FAIL-CLOSED. The layout is validated exactly as for the score stack
    /// (ordered, gap-free, non-empty blocks; finite gates; no `beta_pc`), and
    /// the non-finite firewall covers the WHOLE stacked pass. Any `Err` here is
    /// the caller's cue to run the domains separately — which reproduces each
    /// domain's own result, error or not, exactly.
    pub(crate) fn run_domain_stacked_split(
        &self,
        seed: &Seed,
        blocks: &[RowDomainGateBlock<'_>],
        dir: LaneDir,
        collect: Collect<'_>,
    ) -> Result<Vec<PassOut>> {
        let r = seed.s.ncols();
        if seed
            .e
            .as_ref()
            .is_some_and(|e| e.raw_dim() != seed.s.raw_dim())
        {
            return Err(NyError::InvalidSpec(
                "margin_row: domain-stack seed/error shape mismatch".into(),
            ));
        }
        validate_domain_stack(self.root, blocks, r)?;
        let shifted = shift_block_exceptions(blocks);
        let mut per_block: Vec<BTreeMap<usize, Vec<f64>>> =
            (0..blocks.len()).map(|_| BTreeMap::new()).collect();
        let out = self.run_collect_impl(
            seed,
            None,
            dir,
            Some(&shifted),
            collect,
            Some(blocks),
            None,
            if collect.unst_abs {
                Some(&mut per_block)
            } else {
                None
            },
        )?;
        Ok(split_domain_stacked(
            &out,
            blocks,
            collect.unst_abs.then_some(per_block),
        ))
    }

    /// RESEARCH PROBE ONLY (`ny vnncomp-research margin-row col-scaling-v2`,
    /// margin-row throughput plan B1(b)): one cross-domain stacked pass over
    /// `blocks`, each a contiguous column range plus that domain's gates, with
    /// NO candidate exceptions and an explicit collection spec.
    ///
    /// Exactly [`Self::run_domain_stacked_collect`] with empty exception
    /// tables, so the same layout validation fails closed on any malformed
    /// partition. It exists only because `RowDomainGateBlock` is crate-private
    /// and the probe lives in `ny-cli`; no scored path calls it.
    #[doc(hidden)]
    pub fn probe_domain_stacked_collect(
        &self,
        seed: &Seed,
        blocks: &[(Range<usize>, &DomainGates)],
        dir: LaneDir,
        collect: Collect<'_>,
    ) -> Result<PassOut> {
        let empty = Exceptions::default();
        let rows: Vec<RowDomainGateBlock<'_>> = blocks
            .iter()
            .map(|(columns, gates)| RowDomainGateBlock {
                columns: columns.clone(),
                gates,
                exceptions: &empty,
            })
            .collect();
        self.run_domain_stacked_collect(seed, &rows, dir, collect)
    }

    /// Every backward pass enters here. Chooses the certified error lane this
    /// pass PUBLISHES from the engine's frozen [`ErrLane`] and the pass width
    /// (#mr-errlane), runs it, counts it, and — in the `audit` arm only —
    /// shadows the envelope lanes on the same pass. With the lane `PerCol`
    /// this publishes the corrected per-column body and counts the pass.
    ///
    /// `coll_blocks` (#mr-stack, B5): with `domain_stack` and
    /// `collect.unst_abs`, receive each block's own `unst_abs` sums (one map
    /// per block, in block order) instead of one sum across all columns; the
    /// returned `coll` is then an empty map. `None` everywhere else, which is
    /// the established behaviour byte-for-byte. It is threaded to the PUBLISHED
    /// pass only; the audit arm's shadow passes are measurement-only and never
    /// receive it.
    #[allow(clippy::too_many_arguments)]
    fn run_collect_impl(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect: Collect<'_>,
        domain_stack: Option<&[RowDomainGateBlock<'_>]>,
        prefix_relu: Option<usize>,
        coll_blocks: Option<&mut Vec<BTreeMap<usize, Vec<f64>>>>,
    ) -> Result<PassOut> {
        self.run_collect_impl_cancellable::<false>(
            seed,
            dom,
            dir,
            exc,
            collect,
            domain_stack,
            prefix_relu,
            coll_blocks,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_collect_impl_cancellable<const CANCELLABLE: bool>(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect: Collect<'_>,
        domain_stack: Option<&[RowDomainGateBlock<'_>]>,
        prefix_relu: Option<usize>,
        coll_blocks: Option<&mut Vec<BTreeMap<usize, Vec<f64>>>>,
        deadline: Option<Instant>,
    ) -> Result<PassOut> {
        let deadline = if CANCELLABLE { deadline } else { None };
        check_optional_deadline(deadline)?;
        if !self.outward() {
            // Parity mode carries no error lane at all.
            return self.run_collect_lane::<CANCELLABLE>(
                seed,
                dom,
                dir,
                exc,
                collect,
                domain_stack,
                prefix_relu,
                PassLane::PerCol,
                coll_blocks,
                deadline,
            );
        }
        let r = seed.s.ncols();
        let lane = self.err_lane.publish(r, self.err_min_cols);
        let out = self.run_collect_lane::<CANCELLABLE>(
            seed,
            dom,
            dir,
            exc,
            collect,
            domain_stack,
            prefix_relu,
            lane,
            coll_blocks,
            deadline,
        )?;
        let pen = matches!(lane, PassLane::Envelope { .. })
            .then(|| errlane::max_penalty(&out, &self.root.xabs))
            .flatten();
        errlane::note_pass(lane, r, pen);
        if !CANCELLABLE && self.err_lane == ErrLane::Audit {
            self.audit_pass(seed, dom, dir, exc, domain_stack, prefix_relu, &out);
        }
        check_optional_deadline(deadline)?;
        Ok(out)
    }

    /// #mr-errlane `audit` arm: re-run THIS pass with each envelope lane and
    /// record how much looser it is than the published per-column lane. The
    /// shadows are discarded; a shadow failure is counted, never propagated,
    /// so every bound the audit arm publishes is the `off` arm's (its extra
    /// wall can still move time-driven policies; `eel_off_is_bit_identical`
    /// pins the deadline-free tree bit-for-bit).
    #[allow(clippy::too_many_arguments)]
    fn audit_pass(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        domain_stack: Option<&[RowDomainGateBlock<'_>]>,
        prefix_relu: Option<usize>,
        published: &PassOut,
    ) {
        let r = seed.s.ncols();
        let lower = matches!(dir, LaneDir::Lower);
        errlane::note_audited();
        let bc = self.concretize(published, lower);
        let shadows = [
            (errlane::Variant::Row, false, r >= self.err_min_cols.max(1)),
            (errlane::Variant::Channel, true, r >= 1),
        ];
        for (variant, per_channel, runs) in shadows {
            if !runs {
                continue;
            }
            match self.run_collect_lane::<false>(
                seed,
                dom,
                dir,
                exc,
                Collect::default(),
                domain_stack,
                prefix_relu,
                PassLane::Envelope { per_channel },
                None,
                None,
            ) {
                Ok(shadow) => {
                    let be = self.concretize(&shadow, lower);
                    errlane::record_audit(
                        variant,
                        published,
                        &shadow,
                        &bc,
                        &be,
                        lower,
                        &self.root.xabs,
                    );
                }
                Err(_) => errlane::record_audit_error(variant),
            }
        }
    }

    /// One backward pass in an explicit error lane. Every Outward lane repairs
    /// gradual underflow; envelopes dominate the repaired per-column updates.
    /// Nominal coefficients and biases retain their original arithmetic.
    #[allow(clippy::too_many_arguments)]
    fn run_collect_lane<const CANCELLABLE: bool>(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect: Collect<'_>,
        domain_stack: Option<&[RowDomainGateBlock<'_>]>,
        prefix_relu: Option<usize>,
        lane: PassLane,
        coll_blocks: Option<&mut Vec<BTreeMap<usize, Vec<f64>>>>,
        deadline: Option<Instant>,
    ) -> Result<PassOut> {
        BackwardKernel {
            net: self.net,
            frame: self.root,
            gpu_conv: &self.gpu_conv,
        }
        .run_collect_lane::<CANCELLABLE>(
            seed,
            dom,
            dir,
            exc,
            collect,
            domain_stack,
            prefix_relu,
            lane,
            coll_blocks,
            deadline,
        )
    }

    /// The 2 x n_y y-row refresh (lower + upper lanes, identity seeds).
    pub fn y_rows(&self, dom: Option<&DomainGates>) -> Result<(PassOut, PassOut)> {
        let seed = self.identity_seed();
        let al = self.run(&seed, dom, LaneDir::Lower, None, false)?;
        let au = self.run(&seed, dom, LaneDir::Upper, None, false)?;
        Ok((al, au))
    }

    /// One backward pass, GPU-authoritative when the dark seam admits it.
    ///
    /// Semantically IDENTICAL to `run(seed, dom, dir, None, false)`: the seam
    /// either returns a certified [`PassOut`] the caller concretizes with the
    /// lane's own unchanged `concretize_*`, or refuses — in which case this is
    /// exactly the CPU call, bit-for-bit. With `NY_MARGIN_ROW_GPU` unset the
    /// seam refuses before touching anything, so every existing bound is
    /// unchanged by construction.
    pub(crate) fn run_seamed(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        ctx: &super::gpu_seam::SeamCtx<'_>,
    ) -> Result<PassOut> {
        // #margin-row-beta: the seam predates the beta terms and would compute
        // WITHOUT them (a looser but still sound bound). No current caller
        // passes a beta-carrying dom here; refuse defensively so that can
        // never change silently. Per-column terms (`beta_pc`) are refused for
        // the same reason.
        if dom.is_some_and(|d| !d.beta.is_empty() || !d.beta_pc.is_empty()) {
            return self.run(seed, dom, dir, None, false);
        }
        // #margin-row-gpu-eft (`NY_MARGIN_ROW_GPU_BACKWARD=1`, default OFF):
        // the certified EFT/double-single lane. Disabled, it costs one latched
        // env read; armed, it runs the once-per-process M1 device self-check
        // and refuses `ShadowOnly` until the M3 A/B grants verdict authority
        // (engagement telemetry per rule R9) — ANY `Err` falls through to the
        // old seam + CPU path, bit-for-bit. Deadline stays HOST-side in that
        // lane by construction.
        let lane_ctx = super::gpu_backward::LaneCtx {
            y_abs: ctx.y_abs,
            deadline: ctx.deadline,
            // The root direct pass runs ONCE per instance; a residency cache
            // would have nothing to hit. The hot sites (B4/B5) carry the
            // BaB-owned cache through their own `LaneCtx`.
            plan_cache: None,
        };
        if let Ok(pass) = super::gpu_backward::run_pass_certified(self, seed, dom, dir, &lane_ctx) {
            return Ok(pass);
        }
        let out = match super::gpu_seam::run_pass(self, seed, dom, dir, ctx) {
            Ok(pass) => pass,
            Err(_refused) => self.run(seed, dom, dir, None, false)?,
        };
        // #margin-row-gpu-eft M2 (`NY_MARGIN_ROW_GPU_SHADOW=1`, default OFF):
        // shadow the pass we are about to return — the device computes the
        // same pass, is compared (enclosure overlap + probe + radius
        // telemetry), and DISCARDED. `out` is returned unchanged either way,
        // so a shadow run's verdicts are bit-identical to an unshadowed run.
        super::gpu_backward::shadow_compare(self, seed, dom, dir, &lane_ctx, &out);
        Ok(out)
    }

    /// The 2 x n_y y-row refresh, GPU-authoritative when the seam admits it.
    ///
    /// The identity seed is exact in f32 and carries no certified error, so it
    /// needs no `y_abs`; and the device publishes BOTH lanes from ONE walk, so
    /// an admitted refresh costs one dispatch instead of two CPU passes.
    pub(crate) fn y_rows_seamed(
        &self,
        dom: Option<&DomainGates>,
        ctx: &super::gpu_seam::SeamCtx<'_>,
    ) -> Result<(PassOut, PassOut)> {
        let seed = self.identity_seed();
        // #margin-row-beta: see `run_seamed` — a beta-carrying dom never
        // enters the seam (defensive; the y-pack never carries beta today).
        if dom.is_some_and(|d| !d.beta.is_empty() || !d.beta_pc.is_empty()) {
            let al = self.run(&seed, dom, LaneDir::Lower, None, false)?;
            let au = self.run(&seed, dom, LaneDir::Upper, None, false)?;
            return Ok((al, au));
        }
        // B6 (#margin-row-gpu-eft, SITE_Y_REFRESH): the CERTIFIED lane gets
        // the refresh BEFORE the dark old seam. Two transactions rather than
        // one two-walk program: the lane's program shape is single-direction
        // (the gate quads are DIRECTION-BAKED at build time,
        // `transaction::Builder::apply_activation`), so a two-walk program
        // would have to fork that baking — two transactions reuse the
        // walk verbatim and cost one extra submit on a site the design ranks
        // last. Both lanes must admit: a mixed (device lower / CPU upper)
        // pack would pair enclosures from two different arithmetics inside
        // one y-box, which is sound but makes the pack run-dependent — the
        // `Busy`-nondeterminism risk the design names. So a refusal on
        // EITHER lane falls through to the established path, bit-for-bit.
        let lane_ctx = super::gpu_backward::LaneCtx {
            y_abs: None,
            deadline: ctx.deadline,
            // UNFINISHED (integrator, 2 lines): the BaB-owned cache reaches
            // this site only through `bab::build_pack`, and B6's brief is
            // explicitly "no bab.rs change" — so the y-refresh rebuilds its
            // statics per pass today. Verdict-neutral (the site is SHADOW),
            // but it is the site's whole perf story.
            plan_cache: None,
        };
        let certified =
            super::gpu_backward::run_pass_certified_y(self, &seed, dom, LaneDir::Lower, &lane_ctx)
                .and_then(|al| {
                    super::gpu_backward::run_pass_certified_y(
                        self,
                        &seed,
                        dom,
                        LaneDir::Upper,
                        &lane_ctx,
                    )
                    .map(|au| (al, au))
                });
        if let Ok(pair) = certified {
            return Ok(pair);
        }
        let pair = match super::gpu_seam::run_pass_pair(self, &seed, dom, ctx) {
            Ok(pair) => pair,
            Err(_refused) => {
                let al = self.run(&seed, dom, LaneDir::Lower, None, false)?;
                let au = self.run(&seed, dom, LaneDir::Upper, None, false)?;
                (al, au)
            }
        };
        // Per-site shadow: compute, enclosure-compare against the CPU pass
        // AT THIS CALL SITE, discard. `pair` is returned unchanged either
        // way, so a shadowed run's verdicts are bit-identical.
        for (dir, cpu) in [(LaneDir::Lower, &pair.0), (LaneDir::Upper, &pair.1)] {
            super::gpu_backward::shadow_compare_site(
                super::gpu_backward::SITE_Y_REFRESH,
                self,
                &seed,
                dom,
                dir,
                super::gpu_backward::PassShape::default(),
                &lane_ctx,
                cpu,
            );
        }
        Ok(pair)
    }

    /// B5 — the `eval_node` main pass, GPU-authoritative when the certified
    /// lane admits it (#margin-row-gpu-eft, `SITE_EVAL`).
    ///
    /// Semantically IDENTICAL to `run_collect(seed, dom, dir, None, collect)`:
    /// the lane either returns a certified [`PassOut`] carrying the same
    /// `coll` map, or refuses — in which case this is exactly the CPU call,
    /// bit-for-bit.
    ///
    /// ADMISSION (design B5): only the exact heuristic-collect shape
    /// `Collect{unst_abs:true, rows:None, unst_rows:false}`. The tier-0 row
    /// capture (`collect.rows`, gated on `tier0_exact > 0`, default 0) and
    /// the #alpha-opt `unst_rows` capture (beta-armed, env-only) have no
    /// device readback and stay on the CPU — where they are today.
    /// Exceptions are never passed here (`eval_node` passes `exc = None`).
    pub(crate) fn run_collect_seamed(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        collect: Collect<'_>,
        ctx: &super::gpu_backward::LaneCtx<'_>,
    ) -> Result<PassOut> {
        // #margin-row-beta: the lane predates the beta terms and would
        // compute WITHOUT them (looser but sound). Refuse defensively — the
        // same guard `run_seamed` carries.
        let beta = dom.is_some_and(|d| !d.beta.is_empty() || !d.beta_pc.is_empty());
        let admitted = !beta && super::gpu_backward::collect_shape_admits(collect);
        if admitted {
            if let Ok(pass) =
                super::gpu_backward::run_pass_certified_collect(self, seed, dom, dir, collect, ctx)
            {
                return Ok(pass);
            }
        }
        let out = self.run_collect(seed, dom, dir, None, collect)?;
        if admitted {
            super::gpu_backward::shadow_compare_site(
                super::gpu_backward::SITE_EVAL,
                self,
                seed,
                dom,
                dir,
                super::gpu_backward::PassShape {
                    exc: None,
                    collect_unst_abs: true,
                },
                ctx,
                &out,
            );
        }
        Ok(out)
    }

    /// B4 — the batched-children scoring pass, GPU-authoritative when the
    /// certified lane admits it (#margin-row-gpu-eft, `SITE_SCORE`).
    ///
    /// Semantically IDENTICAL to `run(seed, dom, dir, Some(exc), false)`. The
    /// exceptions ride the device FULLY CERTIFIED (FIX A1), because this
    /// pass's concretized rows become child bounds without re-proving
    /// (bab.rs:1612 / :1555).
    pub(crate) fn run_exc_seamed(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: &Exceptions,
        ctx: &super::gpu_backward::LaneCtx<'_>,
    ) -> Result<PassOut> {
        let beta = dom.is_some_and(|d| !d.beta.is_empty() || !d.beta_pc.is_empty());
        if !beta {
            if let Ok(pass) =
                super::gpu_backward::run_pass_certified_exc(self, seed, dom, exc, dir, ctx)
            {
                return Ok(pass);
            }
        }
        let out = self.run(seed, dom, dir, Some(exc), false)?;
        if !beta {
            super::gpu_backward::shadow_compare_site(
                super::gpu_backward::SITE_SCORE,
                self,
                seed,
                dom,
                dir,
                super::gpu_backward::PassShape {
                    exc: Some(exc),
                    collect_unst_abs: false,
                },
                ctx,
                &out,
            );
        }
        Ok(out)
    }

    /// Concretize lower rows over the root box: per row
    /// `A@mid + b - |A|@rad` (Python parity), minus the certified penalty and
    /// rounded toward -inf in Outward mode.
    pub fn concretize_lower(&self, out: &PassOut) -> Vec<f64> {
        self.concretize(out, true)
    }

    /// Concretize upper rows over the root box (toward +inf in Outward mode).
    pub fn concretize_upper(&self, out: &PassOut) -> Vec<f64> {
        self.concretize(out, false)
    }

    fn concretize(&self, out: &PassOut, lower: bool) -> Vec<f64> {
        let root = self.root;
        let n_in = root.mid.len();
        let r = out.a.ncols();
        let asl = out.a.as_slice().expect("standard layout");
        let esl = out.e.as_ref().map(|e| e.as_slice().expect("layout"));
        let mut v = vec![0.0; r];
        let mut rr = vec![0.0; r];
        let mut tabs = vec![0.0; r];
        let mut pen = vec![0.0; r];
        let outward = root.mode.outward();
        let mut dust = if outward { vec![0.0; r] } else { Vec::new() };
        let screen = outward && errconv_screens_enabled();
        let (mut rows_fast, mut rows_slow) = (0u64, 0u64);
        for i in 0..n_in {
            let m = root.mid[i];
            let rd = root.rad[i];
            let xa = root.xabs[i];
            let arow = &asl[i * r..(i + 1) * r];
            // #mr-errconv-screen: a VALID row runs the same repaired terms
            // branch-free.
            if screen {
                let erow = esl.map(|es| &es[i * r..(i + 1) * r]);
                if concretize_row_is_valid(arow, erow, xa) {
                    rows_fast += 1;
                    #[cfg(test)]
                    ERRCONV_TEST_ROWS.with(|n| n.set(add_row(n.get(), 2)));
                    concretize_row_branch_free(
                        &mut v, &mut rr, &mut tabs, &mut dust, &mut pen, arow, erow, m, rd, xa,
                    );
                    continue;
                }
                rows_slow += 1;
            }
            for (ri, av) in arow.iter().enumerate() {
                v[ri] += av * m;
                let aa = av.abs();
                rr[ri] += aa * rd;
                tabs[ri] += error_mul(aa, xa, outward);
                if outward {
                    let dm = nominal_product_dust(*av, m);
                    let dr = nominal_product_dust(aa, rd);
                    if dm != 0.0 {
                        dust[ri] = next_up(dust[ri] + dm);
                    }
                    if dr != 0.0 {
                        dust[ri] = next_up(dust[ri] + dr);
                    }
                }
            }
            if let Some(es) = esl {
                let erow = &es[i * r..(i + 1) * r];
                for (ri, ev) in erow.iter().enumerate() {
                    pen[ri] += error_mul(*ev, xa, outward);
                }
            }
        }
        errconv_note_conc_rows(rows_fast, rows_slow);
        let gam = gamma_n(n_in + 16);
        (0..r)
            .map(|ri| {
                let base = v[ri] + out.b[ri];
                let raw = if lower { base - rr[ri] } else { base + rr[ri] };
                if !root.mode.outward() {
                    return raw;
                }
                let mut slack = next_up(
                    certify_up(pen[ri] + out.eb[ri], 1e-14)
                        + next_up(error_mul(
                            gam,
                            tabs[ri] + out.b[ri].abs() + pen[ri] + out.eb[ri],
                            true,
                        )),
                );
                if dust[ri] != 0.0 {
                    slack = next_up(slack + certify_up(dust[ri], gam));
                }
                if lower {
                    next_down(next_down(raw - slack))
                } else {
                    next_up(next_up(raw + slack))
                }
            })
            .collect()
    }
}

impl BackwardKernel<'_> {
    fn outward(&self) -> bool {
        self.frame.mode().outward()
    }

    fn checkpoint<const CANCELLABLE: bool>(&self, deadline: Option<Instant>) -> Result<()> {
        if CANCELLABLE {
            check_optional_deadline(deadline)?;
            self.frame.checkpoint()?;
        }
        Ok(())
    }

    fn poll<const CANCELLABLE: bool>(&self, deadline: Option<Instant>, index: usize) -> Result<()> {
        if index.is_multiple_of(DEADLINE_POLL_ITEMS) {
            self.checkpoint::<CANCELLABLE>(deadline)?;
        }
        Ok(())
    }

    fn conv_backward(
        &self,
        cv: &ConvOp,
        src: &Array2<f64>,
        dst: &mut Array2<f64>,
        abs_w: bool,
        lane: ConvLane,
    ) {
        BackwardEngine::conv_backward(self.gpu_conv, cv, src, dst, abs_w, lane, || self.outward());
    }

    fn timed_conv(
        &self,
        cv: &ConvOp,
        src: &Array2<f64>,
        dst: &mut Array2<f64>,
        abs_w: bool,
        deadline: Instant,
    ) -> Result<()> {
        if self.frame.cooperative() {
            super::net::conv_apply_backward_with_checkpoint(cv, src, dst, abs_w, deadline, &|| {
                self.frame.checkpoint()
            })
        } else {
            conv_apply_backward_with_deadline(cv, src, dst, abs_w, deadline)
        }
    }

    fn timed_error_conv(
        &self,
        cv: &ConvOp,
        error: &Array2<f64>,
        nominal: &Array2<f64>,
        dst: &mut Array2<f64>,
        deadline: Instant,
    ) -> Result<()> {
        if self.frame.cooperative() {
            super::net::error_conv_with_checkpoint(cv, error, nominal, dst, deadline, &|| {
                self.frame.checkpoint()
            })
        } else {
            error_conv_with_deadline(cv, error, nominal, dst, deadline)
        }
    }

    fn zeros_with_deadline(
        &self,
        shape: (usize, usize),
        deadline: Option<Instant>,
    ) -> Result<Array2<f64>> {
        if !self.frame.cooperative() {
            return zeros_with_deadline(shape, deadline);
        }
        let check = || self.frame.checkpoint();
        zeros_with_checkpoint(shape, deadline, Some(&check))
    }

    fn clone_with_deadline(
        &self,
        src: &Array2<f64>,
        deadline: Option<Instant>,
    ) -> Result<Array2<f64>> {
        if !self.frame.cooperative() {
            return clone_with_deadline(src, deadline);
        }
        let check = || self.frame.checkpoint();
        clone_with_checkpoint(src, deadline, Some(&check))
    }

    fn finite_with_deadline(&self, values: &[f64], deadline: Option<Instant>) -> Result<()> {
        let check = || self.frame.checkpoint();
        finite_with_checkpoint(values, deadline, self.frame.cooperative().then_some(&check))
    }

    fn combined_err_checked(
        &self,
        l: &Array2<f64>,
        e: Option<&Array2<f64>>,
        g: f64,
        deadline: Option<Instant>,
        repair_underflow: bool,
    ) -> Result<Array2<f64>> {
        if !self.frame.cooperative() {
            return combined_err_checked(l, e, g, deadline, repair_underflow);
        }
        let check = || self.frame.checkpoint();
        combined_err_with_checkpoint(l, e, g, deadline, repair_underflow, Some(&check))
    }

    #[allow(clippy::too_many_arguments)]
    fn merge_step_checked(
        &self,
        state: &mut [Option<LaneMat>],
        id: usize,
        contrib: LaneMat,
        outward: bool,
        gtab: Option<&[usize]>,
        op: usize,
        eb: &[f64],
        deadline: Option<Instant>,
    ) -> Result<()> {
        if !self.frame.cooperative() {
            return merge_step_checked(state, id, contrib, outward, gtab, op, eb, deadline);
        }
        let check = || self.frame.checkpoint();
        merge_step_with_checkpoint(
            state,
            id,
            contrib,
            outward,
            gtab,
            op,
            eb,
            deadline,
            Some(&check),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_collect_lane<const CANCELLABLE: bool>(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect: Collect<'_>,
        domain_stack: Option<&[RowDomainGateBlock<'_>]>,
        prefix_relu: Option<usize>,
        lane: PassLane,
        mut coll_blocks: Option<&mut Vec<BTreeMap<usize, Vec<f64>>>>,
        deadline: Option<Instant>,
    ) -> Result<PassOut> {
        // Separate monomorphizations let the ordinary pass erase every
        // optional clock branch while keeping one numerical implementation.
        let deadline = if CANCELLABLE { deadline } else { None };
        self.checkpoint::<CANCELLABLE>(deadline)?;
        let net = self.net;
        let outward = self.outward();
        // Seed evaluation is per column before any grouping. Every Outward
        // source and envelope therefore shares its underflow correction.
        let repair_underflow = outward;
        if repair_underflow {
            let gamma_safe = |n: usize, multiplier: usize, offset: usize| {
                n.checked_mul(multiplier)
                    .and_then(|n| n.checked_add(offset))
                    .is_some_and(|n| gamma_n(n) <= 0.25)
            };
            let mut unsafe_gamma = !gamma_safe(net.n_y, 2, 8);
            for i in 0..self.frame.layer_count() {
                let rec = self.frame.layer(i)?;
                self.poll::<CANCELLABLE>(deadline, i)?;
                unsafe_gamma |= !gamma_safe(rec.n, 4, 16);
            }
            for (k, op) in net.ops.iter().enumerate() {
                self.poll::<CANCELLABLE>(deadline, k)?;
                unsafe_gamma |= match op {
                    TwinOp::Conv(cv) => {
                        let mut max_taps = 0usize;
                        for (i, taps) in cv.back_taps.iter().enumerate() {
                            self.poll::<CANCELLABLE>(deadline, i)?;
                            max_taps = max_taps.max(taps.len());
                        }
                        !gamma_safe(cv.k_bwd, 1, 8)
                            || max_taps
                                .checked_add(16)
                                .is_none_or(|offset| !gamma_safe(cv.k_bwd, 2, offset))
                            || !gamma_safe(net.tsize[k + 1], 8, 16)
                    }
                    TwinOp::ChannelAffine { scale, .. } => !gamma_safe(scale.len(), 8, 16),
                    _ => false,
                };
            }
            unsafe_gamma |= self
                .frame
                .mid()
                .len()
                .checked_add(16)
                .is_none_or(|n| gamma_n(n) > 1e-8);
            if unsafe_gamma {
                return Err(NyError::UnsupportedConfiguration(
                    "margin_row: relative contraction exceeds its certified range".into(),
                ));
            }
        }
        let r = seed.s.ncols();
        let mut state: Vec<Option<LaneMat>> = (0..=net.ops.len()).map(|_| None).collect();
        let mut b = vec![0.0; r];
        let mut eb = vec![0.0; r];
        // #mr-errlane: per-tensor envelope group sizes (`None` = per-column
        // lane, which never consults them).
        let gtab: Option<Vec<usize>> = match lane {
            PassLane::Envelope { .. } if !outward => None,
            PassLane::PerCol => None,
            PassLane::Envelope { per_channel: false } => Some(vec![1; net.tsize.len()]),
            PassLane::Envelope { per_channel: true } => Some(errlane::channel_groups(net)),
        };
        let gtab = gtab.as_deref();
        let limit = if let Some(pk) = prefix_relu {
            // #backward-interm PREFIX entry: seed rows directly at the INPUT
            // tensor of trunk relu op `pk` (its pre-activation `z`), walk only
            // the ops strictly before it. Every relu encountered has a smaller
            // layer index, so its gates are already frozen in the frozen frame —
            // the same certified per-gate algebra as a head-seeded pass, on a
            // shorter walk. The seed here is exact (identity rows), so its
            // error lane starts at zero when the caller supplies none.
            if pk >= net.i_gemm1 {
                return Err(NyError::InvalidSpec(
                    "margin_row: prefix seed op is not in the trunk".into(),
                ));
            }
            let TwinOp::Relu { input, .. } = &net.ops[pk] else {
                return Err(NyError::InvalidSpec(
                    "margin_row: prefix seed op is not a relu".into(),
                ));
            };
            let n_pre = net.tsize[*input];
            if seed.s.nrows() != n_pre {
                return Err(NyError::shape_mismatch(vec![n_pre], vec![seed.s.nrows()]));
            }
            let (e0, env0) = match gtab {
                // #mr-errlane: the envelope of the seed error (an exact
                // identity seed is the zero envelope, built without the
                // `n_pre x R` zeros the per-column lane allocates).
                Some(g) => (
                    None,
                    Some(match seed.e.as_ref() {
                        Some(e) => EnvErr::from_matrix(e, g[*input]),
                        None => EnvErr::zeros(n_pre, g[*input]),
                    }),
                ),
                None => (
                    if outward {
                        Some(match seed.e.as_ref() {
                            Some(error) => self.clone_with_deadline(error, deadline)?,
                            None => self.zeros_with_deadline(seed.s.dim(), deadline)?,
                        })
                    } else {
                        None
                    },
                    None,
                ),
            };
            state[*input] = Some(LaneMat {
                l: self.clone_with_deadline(&seed.s, deadline)?,
                e: e0,
                env: env0,
            });
            #[cfg(test)]
            trace_step(usize::MAX, "seed", state[*input].as_ref(), &eb);
            pk
        } else {
            if seed.s.nrows() != net.n_y {
                return Err(NyError::shape_mismatch(vec![net.n_y], vec![seed.s.nrows()]));
            }
            let (w1, b1, (n_y, n_h)) = net.gemm1();
            // #cct-margin-row-admission: a head Gemm with a FOLDED
            // BatchNormalization carries a certified fold error vs the exact
            // `Gemm -> BN` map: `|W1 - W1'| <= rel |W1'|` entrywise and
            // `|b1 - b1'| <= b1_err` per row. Charged below exactly like the
            // Conv arm charges its own fold (`gamma + rel + gamma*rel` in the
            // coefficient envelope, `(|S| + Es) * bias_err` in the bias lane).
            // Unfolded heads omit these fold-error terms. Their nominal
            // arithmetic and normal-product error arithmetic are unchanged.
            let (b1_err, w1_rel) = net.gemm1_fold_err();
            let folded = w1_rel != 0.0 || b1_err.iter().any(|e| *e != 0.0);

            // --- Seed through gemm1: L0[k] = sum_j W1[j,k] * S[j]; b = S @ b1.
            let mut l0 = self.zeros_with_deadline((n_h, r), deadline)?;
            {
                let ss = seed.s.as_slice().expect("standard layout");
                let dst = l0.as_slice_mut().expect("standard layout");
                dst.par_chunks_mut(r).enumerate().try_for_each(|(k, acc)| {
                    self.checkpoint::<CANCELLABLE>(deadline)?;
                    for j in 0..n_y {
                        self.poll::<CANCELLABLE>(deadline, j)?;
                        let w = w1[j * n_h + k];
                        if w != 0.0 {
                            axpy_checked(acc, w, &ss[j * r..(j + 1) * r], deadline)?;
                        }
                    }
                    Ok::<(), NyError>(())
                })?;
            }
            for j in 0..n_y {
                self.poll::<CANCELLABLE>(deadline, j)?;
                let bj = b1[j];
                if bj != 0.0 {
                    axpy_checked(&mut b, bj, row(&seed.s, j), deadline)?;
                }
            }
            let e0 = if outward {
                // E0 = |W1|^T (Es + g(|S| + Es)), certified; eb likewise via |b1|.
                //
                // Folded head: the exact seed is `W1ex^T Sex` with
                // `|W1ex - W1'| <= rel |W1'|` and `|Sex - S| <= Es`, so
                // `|W1ex^T Sex - fl(W1'^T S)| <= |W1'|^T (Es + (gamma + rel
                // + gamma*rel)(|S| + Es))` — the Conv arm's envelope
                // (`engine.rs` Conv backward) with the head's `rel`.
                let g = if w1_rel == 0.0 {
                    gamma_n(n_y + 2)
                } else {
                    let gm = gamma_n(n_y + 2);
                    next_up(gm + w1_rel + error_mul(gm, w1_rel, repair_underflow))
                };
                let comb = self.combined_err_checked(
                    &seed.s,
                    seed.e.as_ref(),
                    g,
                    deadline,
                    repair_underflow,
                )?;
                let mut e0 = self.zeros_with_deadline((n_h, r), deadline)?;
                {
                    let cs = comb.as_slice().expect("standard layout");
                    let dst = e0.as_slice_mut().expect("standard layout");
                    let g2 = gamma_n(n_y + 8);
                    dst.par_chunks_mut(r).enumerate().try_for_each(|(k, acc)| {
                        self.checkpoint::<CANCELLABLE>(deadline)?;
                        for j in 0..n_y {
                            self.poll::<CANCELLABLE>(deadline, j)?;
                            let w = w1[j * n_h + k].abs();
                            if w != 0.0 {
                                error_axpy_checked(
                                    acc,
                                    w,
                                    &cs[j * r..(j + 1) * r],
                                    row(&seed.s, j),
                                    deadline,
                                    repair_underflow,
                                )?;
                            }
                        }
                        for v in acc.iter_mut() {
                            *v = certify_up(*v, g2);
                        }
                        Ok::<(), NyError>(())
                    })?;
                }
                {
                    let cs = comb.as_slice().expect("standard layout");
                    for j in 0..n_y {
                        self.poll::<CANCELLABLE>(deadline, j)?;
                        let bj = b1[j].abs();
                        if bj != 0.0 {
                            error_axpy_checked(
                                &mut eb,
                                bj,
                                &cs[j * r..(j + 1) * r],
                                row(&seed.s, j),
                                deadline,
                                repair_underflow,
                            )?;
                        }
                    }
                    // Folded head bias: `|b1ex - b1'| <= b1_err` moves the
                    // functional `sum_j Sex[j] b1[j]` by at most
                    // `sum_j b1_err[j] (|S[j]| + Es[j])` (the Conv bias arm's
                    // `(mag + er) * berr`). That doubles the accumulated term
                    // count, so the certify envelope widens with it.
                    let g2 = if folded {
                        let ss = seed.s.as_slice().expect("standard layout");
                        let es = seed.e.as_ref().map(|e| e.as_slice().expect("layout"));
                        for j in 0..n_y {
                            self.poll::<CANCELLABLE>(deadline, j)?;
                            let be = b1_err[j];
                            if be == 0.0 {
                                continue;
                            }
                            let sr = &ss[j * r..(j + 1) * r];
                            for (ri, ebv) in eb.iter_mut().enumerate() {
                                let er = es.map_or(0.0, |es| es[j * r + ri]);
                                *ebv += error_mul(be, sr[ri].abs() + er, repair_underflow);
                            }
                        }
                        gamma_n(2 * n_y + 8)
                    } else {
                        gamma_n(n_y + 8)
                    };
                    for v in &mut eb {
                        *v = certify_up(*v, g2);
                    }
                }
                Some(e0)
            } else {
                None
            };
            let g1_in = match &net.ops[net.i_gemm1] {
                TwinOp::Gemm { input, .. } => *input,
                _ => unreachable!("validated"),
            };
            // #mr-errlane: the seed's gemm1 error is computed PER COLUMN above
            // (it costs one `n_h x n_y x R` product, the size of `l0`) and only
            // then enveloped. Enveloping the SEED instead would sum every
            // class's `|W1|` row where a column touches two: the identity and
            // margin seeds are sparse over classes, and that is the one place
            // an envelope would be loose by a factor of `n_y`.
            let (e0, env0) = match gtab {
                Some(g) => (None, e0.map(|e| EnvErr::from_matrix(&e, g[g1_in]))),
                None => (e0, None),
            };
            state[g1_in] = Some(LaneMat {
                l: l0,
                e: e0,
                env: env0,
            });
            #[cfg(test)]
            trace_step(usize::MAX, "seed", state[g1_in].as_ref(), &eb);
            net.i_gemm1
        };
        let mut coll: Option<BTreeMap<usize, Vec<f64>>> = collect.unst_abs.then(BTreeMap::new);
        let mut coll_rows: Option<BTreeMap<usize, Array2<f64>>> =
            collect.rows.map(|_| BTreeMap::new());
        let mut unst_rows: Option<BTreeMap<usize, Array2<f64>>> =
            collect.unst_rows.then(BTreeMap::new);
        let lower = matches!(dir, LaneDir::Lower);

        for k in (0..limit).rev() {
            self.checkpoint::<CANCELLABLE>(deadline)?;
            if state[k + 1].is_none() {
                continue;
            }
            let cur = state[k + 1].take().expect("checked above");
            match &net.ops[k] {
                TwinOp::Flatten { input } => {
                    self.merge_step_checked(
                        &mut state, *input, cur, outward, gtab, k, &eb, deadline,
                    )?;
                }
                TwinOp::ChannelAffine {
                    input,
                    scale,
                    shift,
                    scale_rel_err,
                    shift_err,
                } => {
                    // Backward through `y = s ⊙ x + t`: bias folds `Σ v_j t_j`
                    // (with certified parameter error), coefficients scale by
                    // `s` with the same error-lane algebra as a fixed gate.
                    let mut cur = cur;
                    {
                        let ls = cur.l.as_slice().expect("standard layout");
                        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
                        let envr = cur.env.as_ref();
                        let n = scale.len();
                        let gam_b = gamma_n(n + 2);
                        for j in 0..n {
                            self.poll::<CANCELLABLE>(deadline, j)?;
                            let tj = shift[j];
                            let te = shift_err[j];
                            if tj == 0.0 && te == 0.0 {
                                continue;
                            }
                            let lr = &ls[j * r..(j + 1) * r];
                            if tj != 0.0 {
                                axpy_checked(&mut b, tj, lr, deadline)?;
                            }
                            if let Some(es) = es {
                                let er = &es[j * r..(j + 1) * r];
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    if repair_underflow {
                                        let charge =
                                            error_mul(
                                                er[ri] + error_mul(gam_b, mag, repair_underflow),
                                                tj.abs(),
                                                repair_underflow,
                                            ) + error_mul(mag + er[ri], te, repair_underflow);
                                        *ebv +=
                                            add_nominal_dust(charge, tj, lr[ri], repair_underflow);
                                    } else {
                                        *ebv +=
                                            (er[ri] + gam_b * mag) * tj.abs() + (mag + er[ri]) * te;
                                    }
                                    // Running-accumulator rounding carry
                                    // (mirrors the conv-bias arm).
                                    if tj != 0.0 {
                                        *ebv +=
                                            error_mul(2.0 * UNIT, b[ri].abs(), repair_underflow);
                                    }
                                }
                            } else if let Some(env) = envr {
                                // #mr-errlane: the same charge with the row's
                                // envelope in place of its per-column entry.
                                let ev = env.at(j);
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    let charge =
                                        error_mul(ev + error_mul(gam_b, mag, true), tj.abs(), true)
                                            + error_mul(mag + ev, te.abs(), true);
                                    *ebv += add_nominal_dust(charge, tj, lr[ri], true);
                                    if tj != 0.0 {
                                        *ebv +=
                                            error_mul(2.0 * UNIT, b[ri].abs(), repair_underflow);
                                    }
                                }
                            }
                        }
                        if outward {
                            let gc = gamma_n(8 * n + 16);
                            for v in &mut eb {
                                *v = certify_up(*v, gc);
                            }
                        }
                    }
                    // #mr-errlane: the envelope's pre-transform magnitude.
                    let lam_v = cur
                        .env
                        .as_ref()
                        .map(|env| errlane::group_absmax(&cur.l, env.grp));
                    {
                        let ls = cur.l.as_slice_mut().expect("standard layout");
                        let rel = *scale_rel_err;
                        match cur.e.as_mut() {
                            Some(em) => {
                                let es = em.as_slice_mut().expect("standard layout");
                                ls.par_chunks_mut(r)
                                    .zip(es.par_chunks_mut(r))
                                    .enumerate()
                                    .try_for_each(|(j, (lrow, erow))| {
                                        self.checkpoint::<CANCELLABLE>(deadline)?;
                                        let sj = scale[j];
                                        let sa = sj.abs();
                                        for (lv, ev) in lrow.iter_mut().zip(erow.iter_mut()) {
                                            let v = *lv;
                                            let lnew = v * sj;
                                            *ev = if repair_underflow {
                                                let charge = error_mul(
                                                    *ev + error_mul(rel, v.abs(), repair_underflow)
                                                        + error_mul(
                                                            4.0 * UNIT,
                                                            v.abs(),
                                                            repair_underflow,
                                                        ),
                                                    sa,
                                                    repair_underflow,
                                                ) + error_mul(
                                                    rel + 4.0 * UNIT,
                                                    lnew.abs(),
                                                    repair_underflow,
                                                );
                                                slack16(add_nominal_dust(
                                                    charge,
                                                    v,
                                                    sj,
                                                    repair_underflow,
                                                ))
                                            } else {
                                                slack16(
                                                    (*ev + rel * v.abs() + 4.0 * UNIT * v.abs())
                                                        * sa
                                                        + (rel + 4.0 * UNIT) * lnew.abs(),
                                                )
                                            };
                                            *lv = lnew;
                                        }
                                        Ok::<(), NyError>(())
                                    })?;
                            }
                            None => {
                                ls.par_chunks_mut(r).enumerate().try_for_each(|(j, lrow)| {
                                    self.checkpoint::<CANCELLABLE>(deadline)?;
                                    let sj = scale[j];
                                    for lv in lrow.iter_mut() {
                                        *lv *= sj;
                                    }
                                    Ok::<(), NyError>(())
                                })?;
                            }
                        }
                    }
                    // #mr-errlane: `|lnew| = fl(|v|·|s_j|) <= fl(Λv·S[g])`, so the
                    // per-column update's every term is dominated group-wise.
                    if let (Some(env), Some(lam_v)) = (cur.env.as_mut(), lam_v) {
                        let rel = scale_rel_err.abs();
                        let smax = errlane::group_vec_absmax(scale, env.grp);
                        for ((ev, &lv), &sa) in env.v.iter_mut().zip(&lam_v).zip(&smax) {
                            let lnew = lv * sa;
                            let charge = error_mul(
                                *ev + error_mul(rel, lv, true) + error_mul(4.0 * UNIT, lv, true),
                                sa,
                                true,
                            ) + error_mul(rel + 4.0 * UNIT, lnew, true);
                            *ev = slack16(errlane::env_nominal_dust_up(charge, lv, sa));
                        }
                    }
                    #[cfg(test)]
                    trace_step(k, "affine", Some(&cur), &eb);
                    self.merge_step_checked(
                        &mut state, *input, cur, outward, gtab, k, &eb, deadline,
                    )?;
                }
                TwinOp::Add { lhs, rhs } => {
                    let copy = LaneMat {
                        l: self.clone_with_deadline(&cur.l, deadline)?,
                        e: cur
                            .e
                            .as_ref()
                            .map(|e| self.clone_with_deadline(e, deadline))
                            .transpose()?,
                        env: cur.env.clone(),
                    };
                    self.merge_step_checked(
                        &mut state, *lhs, cur, outward, gtab, k, &eb, deadline,
                    )?;
                    self.merge_step_checked(
                        &mut state, *rhs, copy, outward, gtab, k, &eb, deadline,
                    )?;
                }
                TwinOp::Relu { input, layer } => {
                    let li = *layer;
                    let rec = self.frame.layer(li)?;
                    let mut cur = cur;
                    if let Some(cmap) = coll.as_mut() {
                        if !rec.unst.is_empty() {
                            let ls = cur.l.as_slice().expect("standard layout");
                            match (coll_blocks.as_deref_mut(), domain_stack) {
                                // #mr-stack (throughput plan B5): a stacked
                                // MAIN-EVAL pass needs each domain's OWN
                                // branching mass, not one sum across every
                                // domain's columns. Each block sums exactly its
                                // own contiguous column slice with the same
                                // expression as the uniform arm below: the same
                                // values, in the same ascending-column order,
                                // from the same `Sum` start, so every block's
                                // sums are bit-identical to that domain's own
                                // pass. Read-only (pre-transform), heuristic
                                // input only — like the uniform sums.
                                (Some(per_block), Some(blocks)) => {
                                    for (slot, block) in per_block.iter_mut().zip(blocks) {
                                        let (c0, c1) = (block.columns.start, block.columns.end);
                                        let sums: Vec<f64> = rec
                                            .unst
                                            .iter()
                                            .map(|&idx| {
                                                ls[idx * r + c0..idx * r + c1]
                                                    .iter()
                                                    .map(|v| v.abs())
                                                    .sum::<f64>()
                                            })
                                            .collect();
                                        slot.insert(li, sums);
                                    }
                                }
                                _ => {
                                    let sums: Vec<f64> = rec
                                        .unst
                                        .iter()
                                        .map(|&idx| {
                                            ls[idx * r..(idx + 1) * r].iter().map(|v| v.abs()).sum()
                                        })
                                        .collect();
                                    cmap.insert(li, sums);
                                }
                            }
                        }
                    }
                    // Tier-0 capture (#epoch-bab): incoming coefficients on
                    // the relu OUTPUT at this layer's retained neurons,
                    // BEFORE the gate transform. Nearest values, ranker-only.
                    if let (Some(rmap), Some(ret)) = (coll_rows.as_mut(), collect.rows) {
                        if let Some(lr) = ret.layers.get(li) {
                            if !lr.idx.is_empty() {
                                let ls = cur.l.as_slice().expect("standard layout");
                                let mut vmat = Array2::<f64>::zeros((lr.idx.len(), r));
                                let vs = vmat.as_slice_mut().expect("standard layout");
                                for (row_i, &j) in lr.idx.iter().enumerate() {
                                    vs[row_i * r..(row_i + 1) * r]
                                        .copy_from_slice(&ls[j * r..(j + 1) * r]);
                                }
                                rmap.insert(li, vmat);
                            }
                        }
                    }
                    // #alpha-opt capture: PRE-transform coefficients at this
                    // layer's unstable neurons. Same values, same placement as
                    // the Tier-0 capture above, but indexed by `rec.unst`
                    // (every unstable neuron, not the retained subset).
                    // Gradient-only: nothing verdict-bearing reads this.
                    if let Some(umap) = unst_rows.as_mut() {
                        if !rec.unst.is_empty() {
                            let ls = cur.l.as_slice().expect("standard layout");
                            let mut vmat = Array2::<f64>::zeros((rec.unst.len(), r));
                            let vs = vmat.as_slice_mut().expect("standard layout");
                            for (row_i, &j) in rec.unst.iter().enumerate() {
                                vs[row_i * r..(row_i + 1) * r]
                                    .copy_from_slice(&ls[j * r..(j + 1) * r]);
                            }
                            umap.insert(li, vmat);
                        }
                    }
                    // The established uniform-domain arm below is kept
                    // byte-for-byte as the default path.  The canary enters
                    // this separate arm only through `run_domain_stacked`.
                    if let Some(blocks) = domain_stack {
                        apply_domain_stacked_relu(
                            &mut cur,
                            &mut b,
                            &mut eb,
                            rec,
                            li,
                            r,
                            lower,
                            outward,
                            blocks,
                            exc,
                            repair_underflow,
                        )?;
                        self.merge_step_checked(
                            &mut state, *input, cur, outward, gtab, k, &eb, deadline,
                        )?;
                        continue;
                    }
                    let over = dom.and_then(|d| d.layers.get(&li));
                    let (alpha, s, c, ms) = match over {
                        Some(gv) => (&gv.alpha[..], &gv.s[..], &gv.c[..], &gv.ms[..]),
                        None => (&rec.alpha[..], &rec.s[..], &rec.c[..], &rec.ms[..]),
                    };
                    let lower_policy = self.frame.crossing_lower_policy();
                    // Snapshot originals for exceptions BEFORE the in-place
                    // transform. #mr-errlane: an envelope pass snapshots the
                    // neuron's envelope before either lane transforms it.
                    let excs = exc.and_then(|e| e.by_layer.get(&li));
                    let snaps: Vec<(f64, f64)> = excs
                        .map(|list| {
                            list.iter()
                                .map(|x| {
                                    let orig = cur.l[[x.neuron, x.row]];
                                    let eorig = match (cur.e.as_ref(), cur.env.as_ref()) {
                                        (Some(em), _) => em[[x.neuron, x.row]],
                                        (None, Some(env)) => env.at(x.neuron),
                                        (None, None) => 0.0,
                                    };
                                    (orig, eorig)
                                })
                                .collect()
                        })
                        .unwrap_or_default();

                    // Bias accumulation (before transform): lower lane uses the
                    // negative part, upper lane the positive part, against c.
                    let gam_b = gamma_n(rec.n + 2);
                    {
                        let ls = cur.l.as_slice().expect("standard layout");
                        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
                        let envr = cur.env.as_ref();
                        for j in 0..rec.n {
                            self.poll::<CANCELLABLE>(deadline, j)?;
                            let cj = c[j];
                            if cj == 0.0 {
                                continue;
                            }
                            let lr = &ls[j * r..(j + 1) * r];
                            for (ri, bv) in b.iter_mut().enumerate() {
                                let v = lr[ri];
                                let part = if lower { v.min(0.0) } else { v.max(0.0) };
                                *bv += part * cj;
                            }
                            if let Some(es) = es {
                                let er = &es[j * r..(j + 1) * r];
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    // SOUNDNESS (running-bias accumulation): the fl() rounding
                                    // of `*bv += part*cj` above is <= u*|b[ri]| on the ACCUMULATOR
                                    // magnitude (dominated by the gemm1 seed / earlier layers),
                                    // NOT the term magnitude gam_b*mag. Carry 2u*|b_running| per
                                    // step (mirrors merge_into's 2u*|result|). Soundness-loosening:
                                    // only widens eb, so the certified dj can only shrink.
                                    if repair_underflow {
                                        let part = if lower {
                                            lr[ri].min(0.0)
                                        } else {
                                            lr[ri].max(0.0)
                                        };
                                        let charge = error_mul(
                                            er[ri] + error_mul(gam_b, mag, repair_underflow),
                                            cj,
                                            repair_underflow,
                                        ) + error_mul(
                                            2.0 * UNIT,
                                            b[ri].abs(),
                                            repair_underflow,
                                        );
                                        *ebv +=
                                            add_nominal_dust(charge, part, cj, repair_underflow);
                                    } else {
                                        *ebv +=
                                            (er[ri] + gam_b * mag) * cj + 2.0 * UNIT * b[ri].abs();
                                    }
                                }
                            } else if let Some(env) = envr {
                                // #mr-errlane: the row's envelope in place of
                                // its per-column entry (monotone: `|cj| >= cj`).
                                let ev = env.at(j);
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    let part = if lower {
                                        lr[ri].min(0.0)
                                    } else {
                                        lr[ri].max(0.0)
                                    };
                                    let charge =
                                        error_mul(ev + error_mul(gam_b, mag, true), cj.abs(), true)
                                            + error_mul(2.0 * UNIT, b[ri].abs(), true);
                                    *ebv += add_nominal_dust(charge, part, cj, true);
                                }
                            }
                        }
                        if outward {
                            let gc = gamma_n(4 * rec.n + 16);
                            for v in &mut eb {
                                *v = certify_up(*v, gc);
                            }
                        }
                    }
                    // #mr-errlane: the envelope's pre-transform magnitude.
                    let lam_v = cur
                        .env
                        .as_ref()
                        .map(|env| errlane::group_absmax(&cur.l, env.grp));
                    // In-place gate transform + error contraction.
                    {
                        let screen = errconv_screens_enabled();
                        let cols = r;
                        let ls = cur.l.as_slice_mut().expect("standard layout");
                        match cur.e.as_mut() {
                            Some(em) => {
                                let es = em.as_slice_mut().expect("standard layout");
                                ls.par_chunks_mut(cols)
                                    .zip(es.par_chunks_mut(cols))
                                    .enumerate()
                                    .try_for_each(|(j, (lrow, erow))| {
                                        self.checkpoint::<CANCELLABLE>(deadline)?;
                                        let (aj, sj, mj) =
                                            (lower_policy.alpha(rec, j, alpha[j]), s[j], ms[j]);
                                        for (lv, ev) in lrow.iter_mut().zip(erow.iter_mut()) {
                                            let v = *lv;
                                            let vp = v.max(0.0);
                                            let vn = v - vp;
                                            *lv = if lower {
                                                vp * aj + vn * sj
                                            } else {
                                                vp * sj + vn * aj
                                            };
                                            *ev = if repair_underflow && screen {
                                                // #mr-errconv-screen: the same
                                                // repaired update, branch-free.
                                                let (sp, sn) =
                                                    if lower { (aj, sj) } else { (sj, aj) };
                                                relu_error_update_branch_free(
                                                    *ev, v, *lv, vp, vn, sp, sn, mj,
                                                )
                                            } else if repair_underflow {
                                                let charge = error_mul(
                                                    *ev + error_mul(
                                                        4.0 * UNIT,
                                                        v.abs(),
                                                        repair_underflow,
                                                    ),
                                                    mj,
                                                    repair_underflow,
                                                ) + error_mul(
                                                    4.0 * UNIT,
                                                    lv.abs(),
                                                    repair_underflow,
                                                );
                                                let (sp, sn) =
                                                    if lower { (aj, sj) } else { (sj, aj) };
                                                let charge = add_nominal_dust(
                                                    charge,
                                                    vp,
                                                    sp,
                                                    repair_underflow,
                                                );
                                                slack16(add_nominal_dust(
                                                    charge,
                                                    vn,
                                                    sn,
                                                    repair_underflow,
                                                ))
                                            } else {
                                                slack16(
                                                    (*ev + 4.0 * UNIT * v.abs()) * mj
                                                        + 4.0 * UNIT * lv.abs(),
                                                )
                                            };
                                        }
                                        Ok::<(), NyError>(())
                                    })?;
                            }
                            None => {
                                ls.par_chunks_mut(cols)
                                    .enumerate()
                                    .try_for_each(|(j, lrow)| {
                                        self.checkpoint::<CANCELLABLE>(deadline)?;
                                        let (aj, sj) = (lower_policy.alpha(rec, j, alpha[j]), s[j]);
                                        for lv in lrow.iter_mut() {
                                            let v = *lv;
                                            let vp = v.max(0.0);
                                            let vn = v - vp;
                                            *lv = if lower {
                                                vp * aj + vn * sj
                                            } else {
                                                vp * sj + vn * aj
                                            };
                                        }
                                        Ok::<(), NyError>(())
                                    })?;
                            }
                        }
                    }
                    // #mr-errlane: envelope gate contraction. `M[g]` bounds
                    // every `ms_j` AND every slope, so `|lv| = fl(|v|·slope)
                    // <= fl(Λv·M)` and no second sweep over `L` is needed.
                    if let (Some(env), Some(lam_v)) = (cur.env.as_mut(), lam_v.as_ref()) {
                        let mut m = vec![0.0f64; env.v.len()];
                        errlane::fold_gate_max(&mut m, alpha, s, ms, env.grp);
                        errlane::env_gate_update(env, lam_v, &m);
                    }
                    #[cfg(test)]
                    trace_step(k, "relu.gate", Some(&cur), &eb);
                    // Exceptions: exact per-(row, neuron) gate replacement.
                    if let Some(list) = excs {
                        // Debug-only: the `(row, neuron)` keys at this layer
                        // must be distinct. A duplicate double-applies the gate
                        // from a STALE pre-transform snapshot. Producers
                        // validate their own sets; this is the backstop that
                        // makes the contract above testable rather than merely
                        // written down.
                        debug_assert!(
                            {
                                let mut keys: Vec<(usize, usize)> =
                                    list.iter().map(|x| (x.row, x.neuron)).collect();
                                keys.sort_unstable();
                                let before = keys.len();
                                keys.dedup();
                                keys.len() == before
                            },
                            "margin_row: duplicate (row, neuron) exception at layer {li}"
                        );
                        for (x, &(orig, eorig)) in list.iter().zip(&snaps) {
                            let vp = orig.max(0.0);
                            let vn = orig - vp;
                            let (lnew, bpart) = if lower {
                                (vp * x.a2 + vn * x.s2, vn * (x.c2 - c[x.neuron]))
                            } else {
                                (vp * x.s2 + vn * x.a2, vp * (x.c2 - c[x.neuron]))
                            };
                            cur.l[[x.neuron, x.row]] = lnew;
                            b[x.row] += bpart;
                            if let Some(em) = cur.e.as_mut() {
                                let m2 = x.a2.abs().max(x.s2.abs());
                                em[[x.neuron, x.row]] = exception_coefficient_error(
                                    eorig,
                                    orig,
                                    vp,
                                    vn,
                                    x.a2,
                                    x.s2,
                                    m2,
                                    lnew,
                                    lower,
                                    repair_underflow,
                                );
                                eb[x.row] = exception_bias_error(
                                    eb[x.row],
                                    b[x.row],
                                    eorig,
                                    orig,
                                    if lower { vn } else { vp },
                                    x.c2 - c[x.neuron],
                                    bpart,
                                    repair_underflow,
                                );
                            } else if let Some(env) = cur.env.as_mut() {
                                // #mr-errlane: `eorig` is the neuron's
                                // PRE-transform envelope; the exception's
                                // per-column value is folded in by max, so
                                // the envelope still bounds the untouched
                                // columns (gate value) and this one.
                                env_exception_update(env, x, orig, eorig, lnew, lower);
                                eb[x.row] = exception_bias_error(
                                    eb[x.row],
                                    b[x.row],
                                    eorig,
                                    orig,
                                    if lower { vn } else { vp },
                                    x.c2 - c[x.neuron],
                                    bpart,
                                    true,
                                );
                            }
                        }
                    }
                    #[cfg(test)]
                    trace_step(k, "relu.exc", Some(&cur), &eb);
                    // #margin-row-beta: split-Lagrangian coefficient shift,
                    // applied LAST (after the gate transform and exceptions,
                    // which never target a neuron this domain already split —
                    // candidate selection excludes used splits). No-op on an
                    // empty term list, hence bit-identical when disarmed.
                    if let Some(terms) = dom.and_then(|d| d.beta.get(&li)) {
                        apply_beta_terms(&mut cur, terms, lower, 0..r);
                    }
                    #[cfg(test)]
                    trace_step(k, "relu.beta", Some(&cur), &eb);
                    // #margin-row-beta-percol: per-COLUMN Lagrangian terms —
                    // the same certified apply, restricted to the owning
                    // column (the domain-stacked arm's column-range anatomy on
                    // the uniform path). Applied after the shared terms; the
                    // two compose additively per entry, each charging its own
                    // rounding. An out-of-range column is skipped: dropping a
                    // valid `beta >= 0` term only loosens, never unsounds.
                    if let Some(pcs) = dom.and_then(|d| d.beta_pc.get(&li)) {
                        for pc in pcs {
                            if pc.col < r {
                                apply_beta_terms(&mut cur, &pc.terms, lower, pc.col..pc.col + 1);
                            }
                        }
                    }
                    #[cfg(test)]
                    trace_step(k, "relu.beta_pc", Some(&cur), &eb);
                    self.merge_step_checked(
                        &mut state, *input, cur, outward, gtab, k, &eb, deadline,
                    )?;
                }
                TwinOp::Conv(cv) => {
                    let n_out = net.tsize[k + 1];
                    debug_assert_eq!(cur.l.nrows(), n_out);
                    let p = cv.oshape.1 * cv.oshape.2;
                    // Bias: b += Lo @ brep (+ certified fold/bias error).
                    {
                        let ls = cur.l.as_slice().expect("standard layout");
                        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
                        let gam_b = gamma_n(n_out + 2);
                        let screen = repair_underflow && errconv_screens_enabled();
                        let (mut rows_fast, mut rows_slow) = (0u64, 0u64);
                        for j in 0..n_out {
                            self.poll::<CANCELLABLE>(deadline, j)?;
                            let ch = j / p;
                            let bias = cv.bias[ch];
                            let berr = cv.bias_err[ch];
                            if bias == 0.0 && berr == 0.0 {
                                continue;
                            }
                            let lr = &ls[j * r..(j + 1) * r];
                            if bias != 0.0 {
                                axpy_checked(&mut b, bias, lr, deadline)?;
                            }
                            if let Some(es) = es {
                                let er = &es[j * r..(j + 1) * r];
                                let ab = bias.abs();
                                // #mr-errconv-screen: a VALID row runs the
                                // same repaired charges branch-free.
                                if screen {
                                    if bias_row_is_valid(lr, er, &b, bias, berr) {
                                        rows_fast += 1;
                                        #[cfg(test)]
                                        ERRCONV_TEST_ROWS.with(|n| n.set(add_row(n.get(), 0)));
                                        bias_row_charges_branch_free(
                                            &mut eb, lr, er, &b, bias, berr, gam_b,
                                        );
                                        continue;
                                    }
                                    rows_slow += 1;
                                }
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    if repair_underflow {
                                        let charge =
                                            error_mul(
                                                er[ri] + error_mul(gam_b, mag, repair_underflow),
                                                ab,
                                                repair_underflow,
                                            ) + error_mul(mag + er[ri], berr, repair_underflow);
                                        *ebv += add_nominal_dust(
                                            charge,
                                            bias,
                                            lr[ri],
                                            repair_underflow,
                                        );
                                    } else {
                                        *ebv += (er[ri] + gam_b * mag) * ab + (mag + er[ri]) * berr;
                                    }
                                    // SOUNDNESS: the `axpy(&mut b, bias, lr)` accumulation above
                                    // rounds to <= u*|b[ri]| on the running accumulator magnitude
                                    // (dominated by earlier layers), not the term magnitude used
                                    // in gam_b*mag. Carry 2u*|b_running| per accumulating step
                                    // (mirrors merge_into). Soundness-loosening: widens eb only.
                                    if bias != 0.0 {
                                        *ebv +=
                                            error_mul(2.0 * UNIT, b[ri].abs(), repair_underflow);
                                    }
                                }
                            } else if let Some(env) = cur.env.as_ref() {
                                // #mr-errlane: the row's envelope in place of
                                // its per-column entry.
                                let ev = env.at(j);
                                let ab = bias.abs();
                                // #mr-errconv-screen: the envelope lane always
                                // repairs (`true` below); a VALID row runs the
                                // same charges branch-free.
                                if errconv_screens_enabled() && repair_underflow {
                                    if bias_row_is_valid_env(lr, ev, &b, bias, berr) {
                                        rows_fast += 1;
                                        #[cfg(test)]
                                        ERRCONV_TEST_ROWS.with(|n| n.set(add_row(n.get(), 1)));
                                        bias_row_charges_branch_free_env(
                                            &mut eb, lr, ev, &b, bias, berr, gam_b,
                                        );
                                        continue;
                                    }
                                    rows_slow += 1;
                                }
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    let charge =
                                        error_mul(ev + error_mul(gam_b, mag, true), ab, true)
                                            + error_mul(mag + ev, berr, true);
                                    *ebv += add_nominal_dust(charge, bias, lr[ri], true);
                                    if bias != 0.0 {
                                        *ebv +=
                                            error_mul(2.0 * UNIT, b[ri].abs(), repair_underflow);
                                    }
                                }
                            }
                        }
                        errconv_note_bias_rows(rows_fast, rows_slow);
                        if outward {
                            let gc = gamma_n(8 * n_out + 16);
                            for v in &mut eb {
                                *v = certify_up(*v, gc);
                            }
                        }
                    }
                    let n_in_t = net.tsize[cv.input];
                    let mut lnew = self.zeros_with_deadline((n_in_t, r), deadline)?;
                    // #mr-gpu-eval: the signed product may run on the sound
                    // f64 device GEMM; the error lane below certifies it for
                    // ANY summation order (`gpu_conv` module docs).
                    match deadline {
                        Some(at) => self.timed_conv(cv, &cur.l, &mut lnew, false, at)?,
                        None => self.conv_backward(cv, &cur.l, &mut lnew, false, ConvLane::Signed),
                    }
                    // #mr-errlane: an envelope pass skips the `n x R` `|W|`
                    // conv entirely (`cur.e` is `None` there, and feeding
                    // `None` to `combined_err` would silently mean `E = 0`, so
                    // the per-column arm is gated on the absence of an
                    // envelope, not merely on `outward`).
                    let enew = if outward && cur.env.is_none() {
                        let g = next_up(
                            gamma_n(cv.k_bwd + 2)
                                + cv.weight_rel_err
                                + error_mul(
                                    gamma_n(cv.k_bwd + 2),
                                    cv.weight_rel_err,
                                    repair_underflow,
                                ),
                        );
                        let comb = self.combined_err_checked(
                            &cur.l,
                            cur.e.as_ref(),
                            g,
                            deadline,
                            repair_underflow,
                        )?;
                        let mut en = self.zeros_with_deadline((n_in_t, r), deadline)?;
                        match deadline {
                            Some(at) => self.timed_error_conv(cv, &comb, &cur.l, &mut en, at)?,
                            None if repair_underflow => {
                                error_conv_without_deadline(cv, &comb, &cur.l, &mut en)?
                            }
                            None => self.conv_backward(cv, &comb, &mut en, true, ConvLane::Abs),
                        }
                        let g2 = gamma_n(cv.k_bwd + 8);
                        if deadline.is_none() {
                            en.par_mapv_inplace(|v| certify_up(v, g2));
                        } else {
                            en.as_slice_mut()
                                .expect("standard layout")
                                .par_chunks_mut(DEADLINE_POLL_ITEMS)
                                .try_for_each(|chunk| {
                                    self.checkpoint::<CANCELLABLE>(deadline)?;
                                    for v in chunk {
                                        *v = certify_up(*v, g2);
                                    }
                                    Ok::<(), NyError>(())
                                })?;
                        }
                        Some(en)
                    } else {
                        None
                    };
                    let envnew = cur
                        .env
                        .as_ref()
                        .map(|env| {
                            let g = next_up(
                                gamma_n(cv.k_bwd + 2)
                                    + cv.weight_rel_err
                                    + error_mul(
                                        gamma_n(cv.k_bwd + 2),
                                        cv.weight_rel_err,
                                        repair_underflow,
                                    ),
                            );
                            let grp_in = gtab.map_or(1, |gt| gt[cv.input]);
                            errlane::env_conv_backward(cv, &cur.l, env, g, grp_in)
                        })
                        .transpose()?;
                    self.merge_step_checked(
                        &mut state,
                        cv.input,
                        LaneMat {
                            l: lnew,
                            e: enew,
                            env: envnew,
                        },
                        outward,
                        gtab,
                        k,
                        &eb,
                        deadline,
                    )?;
                }
                TwinOp::Gemm { .. } => {
                    return Err(NyError::InvalidSpec(
                        "margin_row: unexpected trunk gemm".into(),
                    ))
                }
            }
        }
        let fin = state[0]
            .take()
            .ok_or_else(|| NyError::InvalidSpec("margin_row: no input rows".into()))?;
        // #mr-errlane: an envelope pass publishes its input-tensor envelope
        // BROADCAST to the `n_in x R` matrix every consumer already reads, so
        // concretization, the GPU shadows and the parent lift are unchanged
        // (and the firewall below covers it like any other error lane).
        let e = match (fin.e, fin.env) {
            (Some(e), _) => Some(e),
            (None, Some(env)) => Some(env.broadcast(r)),
            (None, None) => None,
        };
        let out = PassOut {
            a: fin.l,
            e,
            b,
            eb,
            coll,
            coll_rows,
            unst_rows,
        };
        if deadline.is_some() {
            self.finite_with_deadline(out.a.as_slice().expect("standard layout"), deadline)?;
            self.finite_with_deadline(&out.b, deadline)?;
            if let Some(e) = out.e.as_ref() {
                self.finite_with_deadline(e.as_slice().expect("standard layout"), deadline)?;
            }
            self.finite_with_deadline(&out.eb, deadline)?;
            self.checkpoint::<CANCELLABLE>(deadline)?;
            return Ok(out);
        }
        // Fail-closed NaN/Inf firewall: verdict math must never see non-finite.
        if out.a.iter().any(|v| !v.is_finite())
            || out.b.iter().any(|v| !v.is_finite())
            || out
                .e
                .as_ref()
                .is_some_and(|e| e.iter().any(|v| !v.is_finite()))
            || out.eb.iter().any(|v| !v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "margin_row: non-finite backward rows".into(),
            ));
        }
        Ok(out)
    }
}

/// Minimum residency visible to the tile producers. Vec capacity is priced
/// where accessible; ndarray lengths are only a floor. Opaque memo, provenance,
/// alpha-policy/cache allocations and all concurrent owners remain obligations
/// of the caller's admitted base. This function grants no resource authority.
#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
pub(super) fn input_product_resident_floor(net: &TwinNet, root: &RootGates) -> Option<usize> {
    input_product_resident_floor_impl(net, root, None)
}

/// Same minimum with cooperative scans under the inherited tile deadline.
/// Overflow and expired scans refuse before any numerical product allocation.
pub(super) fn input_product_resident_floor_with_deadline(
    net: &TwinNet,
    root: &RootGates,
    deadline: Instant,
) -> Result<usize> {
    check_deadline(deadline)?;
    let bytes = input_product_resident_floor_impl(net, root, Some(deadline));
    check_deadline(deadline)?;
    bytes.ok_or_else(|| {
        NyError::UnsupportedConfiguration(
            "margin_row: visible input-row owner capacity overflow".into(),
        )
    })
}

fn input_product_resident_floor_impl(
    net: &TwinNet,
    root: &RootGates,
    deadline: Option<Instant>,
) -> Option<usize> {
    check_optional_deadline(deadline).ok()?;
    let bytes64 = |v: &Vec<f64>| v.capacity().checked_mul(size_of::<f64>());
    let mut bytes = size_of::<TwinNet>()
        .checked_add(size_of::<RootGates>())?
        .checked_add(net.ops.capacity().checked_mul(size_of::<TwinOp>())?)?
        .checked_add(net.tsize.capacity().checked_mul(size_of::<usize>())?)?
        .checked_add(net.trunk_relus.capacity().checked_mul(size_of::<usize>())?)?;
    for op in &net.ops {
        check_optional_deadline(deadline).ok()?;
        match op {
            TwinOp::Conv(cv) => {
                bytes = bytes.checked_add(size_of::<ConvOp>())?;
                for values in [&cv.wmat, &cv.wt, &cv.bias, &cv.bias_err] {
                    bytes = bytes.checked_add(bytes64(values)?)?;
                }
                bytes = bytes
                    .checked_add(cv.gather.capacity().checked_mul(size_of::<usize>())?)?
                    .checked_add(
                        cv.back_taps
                            .capacity()
                            .checked_mul(size_of::<Vec<(usize, usize, usize)>>())?,
                    )?;
                for (i, taps) in cv.back_taps.iter().enumerate() {
                    poll_deadline(deadline, i).ok()?;
                    bytes = bytes.checked_add(taps.capacity().checked_mul(size_of::<(
                        usize,
                        usize,
                        usize,
                    )>(
                    ))?)?;
                }
            }
            TwinOp::ChannelAffine {
                scale,
                shift,
                shift_err,
                ..
            } => {
                for values in [scale, shift, shift_err] {
                    bytes = bytes.checked_add(bytes64(values)?)?;
                }
            }
            TwinOp::Gemm {
                weight,
                bias,
                bias_err,
                ..
            } => {
                for values in [weight, bias, bias_err] {
                    bytes = bytes.checked_add(bytes64(values)?)?;
                }
            }
            _ => {}
        }
    }
    for values in [&root.lo, &root.hi, &root.mid, &root.rad, &root.xabs] {
        bytes = bytes.checked_add(bytes64(values)?)?;
    }
    if let Some(source) = &root.head_group_source {
        bytes = bytes.checked_add(source.input_rows_resident_bytes())?;
    }
    if let Some(values) = &root.head_alpha {
        bytes = bytes.checked_add(bytes64(values)?)?;
    }
    bytes = bytes.checked_add(
        root.layers
            .capacity()
            .checked_mul(size_of::<LayerGates>())?,
    )?;
    for gate in &root.layers {
        check_optional_deadline(deadline).ok()?;
        for values in [&gate.l, &gate.u, &gate.alpha, &gate.s, &gate.c, &gate.ms] {
            bytes = bytes.checked_add(bytes64(values)?)?;
        }
        bytes = bytes.checked_add(gate.unst.capacity().checked_mul(size_of::<usize>())?)?;
        if let Some(rows) = &gate.clip_rows {
            bytes = bytes.checked_add(
                rows.m
                    .len()
                    .checked_add(rows.d.len())?
                    .checked_mul(size_of::<f64>())?,
            )?;
            for values in [&rows.sl_lo, &rows.sl_up, &rows.bm, &rows.bd] {
                bytes = bytes.checked_add(bytes64(values)?)?;
            }
        }
    }
    check_optional_deadline(deadline).ok()?;
    Some(bytes)
}

/// Shift every block's DOMAIN-LOCAL exception rows into the combined matrix
/// (`row + block.columns.start`). Called only after [`validate_domain_stack`]
/// has proved each local row lies inside its owner's span.
fn shift_block_exceptions(blocks: &[RowDomainGateBlock<'_>]) -> Exceptions {
    let mut shifted = Exceptions::default();
    for block in blocks {
        for (&li, list) in &block.exceptions.by_layer {
            let dst = shifted.by_layer.entry(li).or_default();
            for x in list {
                dst.push(Exc {
                    row: block.columns.start + x.row,
                    neuron: x.neuron,
                    a2: x.a2,
                    s2: x.s2,
                    c2: x.c2,
                });
            }
        }
    }
    shifted
}

/// Copy columns `cols` of `m` into a fresh standard-layout matrix, bit-for-bit.
fn column_block(m: &Array2<f64>, cols: &Range<usize>) -> Array2<f64> {
    let (rows, total) = m.dim();
    let w = cols.end - cols.start;
    let mut out = Array2::<f64>::zeros((rows, w));
    match m.as_slice() {
        Some(src) => {
            let dst = out.as_slice_mut().expect("fresh array is standard layout");
            for i in 0..rows {
                dst[i * w..(i + 1) * w]
                    .copy_from_slice(&src[i * total + cols.start..i * total + cols.end]);
            }
        }
        None => out.assign(&m.slice(ndarray::s![.., cols.clone()])),
    }
    out
}

/// #mr-stack (B5): slice one stacked pass back into per-block passes. `coll`
/// comes from the engine's per-block `unst_abs` sums (`None` when the pass did
/// not collect them, exactly like a per-domain pass without `unst_abs`); the
/// Tier-0 and `unst_rows` captures are sliced to each block's columns.
fn split_domain_stacked(
    out: &PassOut,
    blocks: &[RowDomainGateBlock<'_>],
    coll_blocks: Option<Vec<BTreeMap<usize, Vec<f64>>>>,
) -> Vec<PassOut> {
    let mut coll_iter = coll_blocks.map(Vec::into_iter);
    blocks
        .iter()
        .map(|block| {
            let cols = block.columns.clone();
            let slice_map = |m: &BTreeMap<usize, Array2<f64>>| -> BTreeMap<usize, Array2<f64>> {
                m.iter()
                    .map(|(&li, mat)| (li, column_block(mat, &cols)))
                    .collect()
            };
            PassOut {
                a: column_block(&out.a, &cols),
                e: out.e.as_ref().map(|e| column_block(e, &cols)),
                b: out.b[cols.clone()].to_vec(),
                eb: out.eb[cols.clone()].to_vec(),
                coll: coll_iter.as_mut().and_then(Iterator::next),
                coll_rows: out.coll_rows.as_ref().map(slice_map),
                unst_rows: out.unst_rows.as_ref().map(slice_map),
            }
        })
        .collect()
}

fn validate_domain_stack(
    root: &RootGates,
    blocks: &[RowDomainGateBlock<'_>],
    columns: usize,
) -> Result<()> {
    if columns == 0 || blocks.is_empty() {
        return Err(NyError::InvalidSpec(
            "margin_row: empty domain-stack layout".into(),
        ));
    }
    let mut next = 0usize;
    for (bi, block) in blocks.iter().enumerate() {
        if block.columns.start != next
            || block.columns.start >= block.columns.end
            || block.columns.end > columns
        {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: malformed domain-stack block {bi}: {:?}, expected start {next}, total {columns}",
                block.columns
            )));
        }
        let width = block.columns.end - block.columns.start;
        for (&li, gates) in &block.gates.layers {
            let Some(rec) = root.layers.get(li) else {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} references missing gate layer {li}"
                )));
            };
            if gates.alpha.len() != rec.n
                || gates.s.len() != rec.n
                || gates.c.len() != rec.n
                || gates.ms.len() != rec.n
            {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} gate shape mismatch at layer {li}"
                )));
            }
            if gates
                .alpha
                .iter()
                .chain(&gates.s)
                .chain(&gates.c)
                .chain(&gates.ms)
                .any(|v| !v.is_finite())
            {
                return Err(NyError::NumericalInstability(format!(
                    "margin_row: non-finite domain-stack gate in block {bi}, layer {li}"
                )));
            }
        }
        // #margin-row-beta: a stacked block's Lagrangian terms fail closed on
        // any shape or sign problem BEFORE entering verdict arithmetic (the
        // apply site would skip them silently; a malformed stack should not
        // degrade quietly).
        // #margin-row-beta-percol: a stacked block's column space is
        // BLOCK-LOCAL (candidate rows), while `beta_pc` columns index the
        // margin objectives of the domain's own eval pass. No caller builds
        // such a block (the eval clears `beta_pc` before the dom reaches any
        // scoring path); refuse rather than silently misapply a column.
        if !block.gates.beta_pc.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: domain-stack block {bi} carries per-column beta \
(column spaces differ; unsupported)"
            )));
        }
        for (&li, terms) in &block.gates.beta {
            let Some(rec) = root.layers.get(li) else {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} beta references missing layer {li}"
                )));
            };
            for t in terms {
                if t.neuron >= rec.n || t.sign == 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "margin_row: malformed domain-stack beta in block {bi}, layer {li}"
                    )));
                }
                if !(t.beta.is_finite() && t.beta >= 0.0) {
                    return Err(NyError::NumericalInstability(format!(
                        "margin_row: non-finite/negative domain-stack beta in block {bi}, layer {li}"
                    )));
                }
            }
        }
        for (&li, list) in &block.exceptions.by_layer {
            let Some(rec) = root.layers.get(li) else {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} exception references missing layer {li}"
                )));
            };
            let mut rows = std::collections::BTreeSet::new();
            for x in list {
                if x.row >= width || x.neuron >= rec.n {
                    return Err(NyError::InvalidSpec(format!(
                        "margin_row: domain-stack block {bi} exception escapes its owner"
                    )));
                }
                if !rows.insert(x.row) {
                    return Err(NyError::InvalidSpec(format!(
                        "margin_row: duplicate domain-stack exception in block {bi}, layer {li}, row {}",
                        x.row
                    )));
                }
                if !(x.a2.is_finite() && x.s2.is_finite() && x.c2.is_finite()) {
                    return Err(NyError::NumericalInstability(format!(
                        "margin_row: non-finite domain-stack exception in block {bi}, layer {li}"
                    )));
                }
            }
        }
        next = block.columns.end;
    }
    if next != columns {
        return Err(NyError::InvalidSpec(format!(
            "margin_row: incomplete domain-stack coverage: ended at {next}, total {columns}"
        )));
    }
    Ok(())
}

#[inline]
fn block_gates<'a>(
    block: &'a RowDomainGateBlock<'_>,
    li: usize,
    rec: &'a LayerGates,
) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
    match block.gates.layers.get(&li) {
        Some(gv) => (&gv.alpha, &gv.s, &gv.c, &gv.ms),
        None => (&rec.alpha, &rec.s, &rec.c, &rec.ms),
    }
}

/// Apply one ReLU backward step with per-column domain-gate ownership.
///
/// For each individual column this executes the same neuron order, scalar
/// operations, error updates, and exception correction as the established
/// uniform-domain arm.  Only independent columns from several domains share
/// the surrounding dispatch.
#[allow(clippy::too_many_arguments)]
fn apply_domain_stacked_relu(
    cur: &mut LaneMat,
    b: &mut [f64],
    eb: &mut [f64],
    rec: &LayerGates,
    li: usize,
    r: usize,
    lower: bool,
    outward: bool,
    blocks: &[RowDomainGateBlock<'_>],
    exc: Option<&Exceptions>,
    repair_underflow: bool,
) -> Result<()> {
    let excs = exc.and_then(|e| e.by_layer.get(&li));
    let snaps: Vec<(f64, f64)> = excs
        .map(|list| {
            list.iter()
                .map(|x| {
                    let orig = cur.l[[x.neuron, x.row]];
                    // #mr-errlane: an envelope pass snapshots the neuron's
                    // pre-transform envelope.
                    let eorig = match (cur.e.as_ref(), cur.env.as_ref()) {
                        (Some(em), _) => em[[x.neuron, x.row]],
                        (None, Some(env)) => env.at(x.neuron),
                        (None, None) => 0.0,
                    };
                    (orig, eorig)
                })
                .collect()
        })
        .unwrap_or_default();

    // Keep the established per-column reduction order: neuron j is still the
    // outer loop, and every column accumulates exactly one j term at a time.
    let gam_b = gamma_n(rec.n + 2);
    {
        let ls = cur.l.as_slice().expect("standard layout");
        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
        let envr = cur.env.as_ref();
        for j in 0..rec.n {
            for block in blocks {
                let (_, _, c, _) = block_gates(block, li, rec);
                let cj = c[j];
                if cj == 0.0 {
                    continue;
                }
                let row0 = j * r;
                for ri in block.columns.clone() {
                    let v = ls[row0 + ri];
                    let part = if lower { v.min(0.0) } else { v.max(0.0) };
                    b[ri] += part * cj;
                }
                if let Some(es) = es {
                    for ri in block.columns.clone() {
                        let mag = ls[row0 + ri].abs();
                        if repair_underflow {
                            let part = if lower {
                                ls[row0 + ri].min(0.0)
                            } else {
                                ls[row0 + ri].max(0.0)
                            };
                            let charge =
                                error_mul(es[row0 + ri] + error_mul(gam_b, mag, true), cj, true)
                                    + error_mul(2.0 * UNIT, b[ri].abs(), true);
                            eb[ri] += add_nominal_dust(charge, part, cj, true);
                        } else {
                            eb[ri] += (es[row0 + ri] + gam_b * mag) * cj + 2.0 * UNIT * b[ri].abs();
                        }
                    }
                } else if let Some(env) = envr {
                    let ev = env.at(j);
                    for ri in block.columns.clone() {
                        let mag = ls[row0 + ri].abs();
                        let part = if lower {
                            ls[row0 + ri].min(0.0)
                        } else {
                            ls[row0 + ri].max(0.0)
                        };
                        let charge = error_mul(ev + error_mul(gam_b, mag, true), cj.abs(), true)
                            + error_mul(2.0 * UNIT, b[ri].abs(), true);
                        eb[ri] += add_nominal_dust(charge, part, cj, true);
                    }
                }
            }
        }
        if outward {
            let gc = gamma_n(4 * rec.n + 16);
            for v in eb.iter_mut() {
                *v = certify_up(*v, gc);
            }
        }
    }

    // #mr-errlane: the envelope's pre-transform magnitude.
    let lam_v = cur
        .env
        .as_ref()
        .map(|env| errlane::group_absmax(&cur.l, env.grp));
    // Independent columns use the gates of exactly their owning block.  The
    // parallel grain remains one neuron row, as in the uniform-domain path.
    {
        let cols = r;
        let ls = cur.l.as_slice_mut().expect("standard layout");
        match cur.e.as_mut() {
            Some(em) => {
                let es = em.as_slice_mut().expect("standard layout");
                ls.par_chunks_mut(cols)
                    .zip(es.par_chunks_mut(cols))
                    .enumerate()
                    .for_each(|(j, (lrow, erow))| {
                        for block in blocks {
                            let (alpha, s, _, ms) = block_gates(block, li, rec);
                            let (aj, sj, mj) = (alpha[j], s[j], ms[j]);
                            for ri in block.columns.clone() {
                                let v = lrow[ri];
                                let vp = v.max(0.0);
                                let vn = v - vp;
                                lrow[ri] = if lower {
                                    vp * aj + vn * sj
                                } else {
                                    vp * sj + vn * aj
                                };
                                erow[ri] = if repair_underflow {
                                    let charge = error_mul(
                                        erow[ri] + error_mul(4.0 * UNIT, v.abs(), true),
                                        mj,
                                        true,
                                    ) + error_mul(4.0 * UNIT, lrow[ri].abs(), true);
                                    let (sp, sn) = if lower { (aj, sj) } else { (sj, aj) };
                                    let charge = add_nominal_dust(charge, vp, sp, true);
                                    slack16(add_nominal_dust(charge, vn, sn, true))
                                } else {
                                    slack16(
                                        (erow[ri] + 4.0 * UNIT * v.abs()) * mj
                                            + 4.0 * UNIT * lrow[ri].abs(),
                                    )
                                };
                            }
                        }
                    });
            }
            None => {
                ls.par_chunks_mut(cols).enumerate().for_each(|(j, lrow)| {
                    for block in blocks {
                        let (alpha, s, _, _) = block_gates(block, li, rec);
                        let (aj, sj) = (alpha[j], s[j]);
                        for ri in block.columns.clone() {
                            let v = lrow[ri];
                            let vp = v.max(0.0);
                            let vn = v - vp;
                            lrow[ri] = if lower {
                                vp * aj + vn * sj
                            } else {
                                vp * sj + vn * aj
                            };
                        }
                    }
                });
            }
        }
    }

    // #mr-errlane: the envelope contraction bounds EVERY block's gates at the
    // neuron (a column's own block is one of them).
    if let (Some(env), Some(lam_v)) = (cur.env.as_mut(), lam_v.as_ref()) {
        let mut m = vec![0.0f64; env.v.len()];
        for block in blocks {
            let (alpha, s, _, ms) = block_gates(block, li, rec);
            errlane::fold_gate_max(&mut m, alpha, s, ms, env.grp);
        }
        errlane::env_gate_update(env, lam_v, &m);
    }
    #[cfg(test)]
    trace_step(usize::MAX - 1, "stack.gate", Some(&*cur), eb);

    // Exceptions retain their original owning domain's intercept.  Validation
    // above guarantees every shifted row lies in exactly one block.
    if let Some(list) = excs {
        for (x, &(orig, eorig)) in list.iter().zip(&snaps) {
            let Some(block) = blocks.iter().find(|block| block.columns.contains(&x.row)) else {
                return Err(NyError::InvalidSpec(
                    "margin_row: stacked exception has no domain owner".into(),
                ));
            };
            let (_, _, c, _) = block_gates(block, li, rec);
            let vp = orig.max(0.0);
            let vn = orig - vp;
            let (lnew, bpart) = if lower {
                (vp * x.a2 + vn * x.s2, vn * (x.c2 - c[x.neuron]))
            } else {
                (vp * x.s2 + vn * x.a2, vp * (x.c2 - c[x.neuron]))
            };
            cur.l[[x.neuron, x.row]] = lnew;
            b[x.row] += bpart;
            if let Some(em) = cur.e.as_mut() {
                let m2 = x.a2.abs().max(x.s2.abs());
                em[[x.neuron, x.row]] = exception_coefficient_error(
                    eorig,
                    orig,
                    vp,
                    vn,
                    x.a2,
                    x.s2,
                    m2,
                    lnew,
                    lower,
                    repair_underflow,
                );
                eb[x.row] = exception_bias_error(
                    eb[x.row],
                    b[x.row],
                    eorig,
                    orig,
                    if lower { vn } else { vp },
                    x.c2 - c[x.neuron],
                    bpart,
                    repair_underflow,
                );
            } else if let Some(env) = cur.env.as_mut() {
                env_exception_update(env, x, orig, eorig, lnew, lower);
                eb[x.row] = exception_bias_error(
                    eb[x.row],
                    b[x.row],
                    eorig,
                    orig,
                    if lower { vn } else { vp },
                    x.c2 - c[x.neuron],
                    bpart,
                    true,
                );
            }
        }
    }
    #[cfg(test)]
    trace_step(usize::MAX - 1, "stack.exc", Some(&*cur), eb);
    // #margin-row-beta, stacked arm: each block's Lagrangian terms apply to
    // exactly its own columns — same math and same (post-exception) placement
    // as the uniform arm, so a stacked pass stays bit-identical to the
    // per-domain passes it replaces.
    for block in blocks {
        if let Some(terms) = block.gates.beta.get(&li) {
            apply_beta_terms(cur, terms, lower, block.columns.clone());
        }
    }
    #[cfg(test)]
    trace_step(usize::MAX - 1, "stack.beta", Some(&*cur), eb);
    Ok(())
}

/// #mr-errlane: fold one per-(row, neuron) exception into the neuron's
/// envelope. The per-column value `slack16((e + 4u|orig|)·max(|a2|,|s2|) +
/// 4u|lnew|)` is evaluated at the PRE-transform envelope `eorig >= E[n, row]`
/// and max-ed in, so the group still bounds the exception's column AND the
/// untouched columns (whose value the gate update already covered).
fn env_exception_update(env: &mut EnvErr, x: &Exc, orig: f64, eorig: f64, lnew: f64, lower: bool) {
    let m2 = x.a2.abs().max(x.s2.abs());
    let vp = orig.max(0.0);
    let vn = orig - vp;
    let val = exception_coefficient_error(eorig, orig, vp, vn, x.a2, x.s2, m2, lnew, lower, true);
    let g = x.neuron / env.grp;
    env.v[g] = errlane::nan_max(env.v[g], val);
}

/// Test-only per-op trace hook (`errlane::trace`); a no-op without an open
/// capture, and compiled out of every non-test build.
#[cfg(test)]
fn trace_step(op: usize, tag: &'static str, lm: Option<&LaneMat>, eb: &[f64]) {
    if !errlane::trace::active() {
        return;
    }
    let Some(lm) = lm else {
        return;
    };
    let e = match (lm.e.as_ref(), lm.env.as_ref()) {
        (Some(e), _) => Some(e.clone()),
        (None, Some(env)) => Some(env.broadcast(lm.l.ncols())),
        (None, None) => None,
    };
    errlane::trace::push(errlane::trace::Step {
        op,
        tag,
        e,
        l: lm.l.clone(),
        eb: eb.to_vec(),
    });
}

/// Merge one op's contribution into its destination tensor, then (tests only)
/// record the destination's state for the per-op dominance oracle.
#[allow(clippy::too_many_arguments)]
fn merge_step(
    state: &mut [Option<LaneMat>],
    id: usize,
    contrib: LaneMat,
    outward: bool,
    gtab: Option<&[usize]>,
    op: usize,
    eb: &[f64],
) {
    merge_into(state, id, contrib, outward, gtab);
    #[cfg(test)]
    trace_step(op, "merge", state[id].as_ref(), eb);
    #[cfg(not(test))]
    let _ = (op, eb);
}

/// #margin-row-beta: add each split's Lagrangian term to the coefficient the
/// backward pass now holds on that split neuron's PRE-activation `z`.
///
/// Placement: called at trunk relu `li` AFTER the gate transform, i.e. when
/// `cur.l[[neuron, .]]` is the coefficient multiplying `z_{li,neuron}` in the
/// relaxed functional. Adding `delta` there is exactly seeding the extra term
/// `delta * z_j(x)` through the remaining prefix walk:
///
/// * Lower lane bounds `f - Σ beta_j s_j z_j` from below ⇒ `delta = -s_j*beta_j`;
/// * Upper lane bounds `f + Σ beta_j s_j z_j` from above ⇒ `delta = +s_j*beta_j`.
///
/// See [`BetaSplit`] for the weak-duality argument that makes ANY `beta >= 0`
/// valid on the split region.
///
/// # Error-lane invariant (the load-bearing line)
///
/// The certified contraction `ms = max(alpha, s)` at this relu bounds the
/// Lipschitz constant of the GATE map `v -> vp*alpha + vn*s`, and that update
/// has already run. The beta shift is an ADDITIVE CONSTANT in the output
/// coefficient — `T_beta(v) = T(v) + delta` has the SAME Lipschitz constant as
/// `T` — so the carried upstream error needs NO rescaling (no `alpha_eff`
/// outside `[0,1]` ever enters the contraction). The only fresh error is the
/// one f64 addition below (`delta` itself is exact: `beta * (±1.0)`), bounded
/// by `u * |fl(l + delta)|` and charged at double weight. Loosening-only.
fn apply_beta_terms(cur: &mut LaneMat, terms: &[BetaSplit], lower: bool, cols: Range<usize>) {
    let r = cur.l.ncols();
    let n = cur.l.nrows();
    let ls = cur.l.as_slice_mut().expect("standard layout");
    let mut es = cur.e.as_mut().map(|e| e.as_slice_mut().expect("layout"));
    let mut env = cur.env.as_mut();
    for t in terms {
        // Fail-safe skips: a non-positive or non-finite beta is refused here
        // (beta = 0 must not even perturb a `-0.0` coefficient bit-wise, and
        // a negative beta would flip the duality direction — unsound), as is
        // a signless term (delta would be ±0.0) or an out-of-range neuron.
        if !(t.beta.is_finite() && t.beta > 0.0) || t.sign == 0 || t.neuron >= n {
            continue;
        }
        let s = f64::from(t.sign.signum());
        let delta = if lower { -s * t.beta } else { s * t.beta };
        let row0 = t.neuron * r;
        for ri in cols.clone() {
            let lv = ls[row0 + ri] + delta;
            ls[row0 + ri] = lv;
            if let Some(es) = es.as_deref_mut() {
                es[row0 + ri] = slack16(es[row0 + ri] + underflow_product_up(2.0 * UNIT, lv.abs()));
            }
        }
        // #mr-errlane: the same charge on the neuron's envelope, at the
        // largest shifted magnitude over the charged columns (the other rows
        // of the group are unchanged and stay below the grown envelope).
        if let Some(env) = env.as_deref_mut() {
            let mx = cols
                .clone()
                .fold(0.0f64, |m, ri| errlane::nan_max(m, ls[row0 + ri].abs()));
            let g = t.neuron / env.grp;
            env.v[g] = slack16(env.v[g] + underflow_product_up(2.0 * UNIT, mx));
        }
    }
}

/// Build the per-layer dense gate override for a set of trunk piece-fixes
/// (`(layer, unst_pos, dir)`), mirroring `engine.py::domain_gates`.
pub fn domain_gates(root: &RootGates, trunk_splits: &[(usize, usize, i8)]) -> DomainGates {
    domain_gates_acc(root, trunk_splits, None).0
}

/// [`domain_gates`] threaded with a domain's ACCUMULATED clip state
/// (#clip-stateful).
///
/// Returns the gate overrides plus the set the CHILDREN should inherit. With
/// `acc_in = None` and the stateful gate dark this is byte-identical to
/// [`domain_gates`] and the second element is always `None`.
///
/// # Where the two arms diverge
///
/// The y-pack producers (`bab::rows_for`, the parallel prebuild, the GPU batch
/// prefill) keep calling the STATELESS [`domain_gates`]. They are keyed by trunk
/// alone in a pure LRU whose stated contract is "a hit returns exactly what a
/// miss would have rebuilt", and they hold no `DomainEntry`. Threading state
/// through them would break that contract for no bound: the pack is the
/// identity-seeded y-row pack, while the GATING bound comes from
/// `eval_with_pack`'s certified pass, which is the call site that does get the
/// accumulated gates. A looser pack is sound (it only over-approximates), so the
/// split is a tightness choice, not a correctness one.
pub fn domain_gates_acc(
    root: &RootGates,
    trunk_splits: &[(usize, usize, i8)],
    acc_in: Option<&super::clip::ClipAcc>,
) -> (DomainGates, Option<std::sync::Arc<super::clip::ClipAcc>>) {
    let mut dg = domain_gates_split_only(root, trunk_splits);
    let acc_out = clip_tighten_gates(root, trunk_splits, &mut dg, acc_in);
    (dg, acc_out)
}

/// #clip-and-verify: narrow OTHER unstable neurons using the halfspaces this
/// domain's splits imply, and rebuild their relaxation gates from the tightened
/// box.
///
/// This is what makes one split tighten the WHOLE domain. Tightening the output
/// y-box instead was measured inert (96 of 100 rows moved, trajectory
/// bit-identical) because nothing downstream consumes it; a narrowed
/// INTERMEDIATE `[l, u]` shrinks that neuron's ReLU triangle and so tightens
/// every backward pass through it.
///
/// # Soundness
///
/// `tighten_intermediate` only ever shrinks an interval, and the gates are
/// rebuilt by the SAME certified functions the root uses — `gates_from_box`
/// then `repair_upper_lines` in outward mode. A neuron that becomes stable
/// under the tightened box gets the exact fixed line, which is what the root
/// would have produced had it known the bound.
/// Returns the accumulated set children should inherit (`None` unless
/// #clip-stateful is armed and something was carried).
fn clip_tighten_gates(
    root: &RootGates,
    trunk_splits: &[(usize, usize, i8)],
    dg: &mut DomainGates,
    acc_in: Option<&super::clip::ClipAcc>,
) -> Option<std::sync::Arc<super::clip::ClipAcc>> {
    // #clip-stateful: OFF ⇒ every line below behaves exactly as it did before
    // the accumulator existed (`acc` stays `None`, the stateful merge block is
    // skipped, nothing is returned), so the dark arm is byte-identical.
    let stateful = clip_stateful_enabled();
    if trunk_splits.is_empty() || !clip_interm_enabled() {
        if stateful && !trunk_splits.is_empty() {
            // R9 VACUITY GUARD. `NY_MARGIN_ROW_CLIP_INTERM` is a separate gate
            // and the campaign has already banked one whole session's worth of
            // "measured a lever that was never armed". Say so out loud rather
            // than returning a silent `None` that reads identically to a dark
            // build.
            super::clip::report_acc_unarmed("clip_interm is OFF");
        }
        return None;
    }
    // FAIL-CLOSED FRAME GATE. This consumer is the DANGEROUS one: it can set
    // `dg.infeasible` and close a domain outright, so a halfspace built from a
    // mis-framed row would discharge a region that is not actually empty and no
    // later backward pass would ever disagree. The y-box consumer has always
    // been guarded; until now this one was not.
    //
    // It covers the accumulated path too, and it must: the halfspaces an
    // accumulated entry was derived from came through exactly this frame, and a
    // stored entry then OUTLIVES the call that produced it. A refusal here means
    // no `ClipAcc` is ever built, which is the fail-closed direction.
    if !super::clip::frame_ok(root) {
        if stateful {
            super::clip::report_acc_unarmed("frame check FAILED");
        }
        return None;
    }
    let hs = super::clip::halfspaces_for_splits(root, trunk_splits);
    if hs.is_empty() {
        if stateful {
            // Almost always THE THIRD GATE: `NY_MARGIN_ROW_CLIP_ROWS` /
            // `margin_row.clip_rows` is off, so no layer retained the
            // input-relative rows and every halfspace is unbuildable. A
            // measurement taken here is VACUOUS, not negative.
            super::clip::report_acc_unarmed("no halfspaces (clip_rows retention OFF?)");
        }
        return None;
    }
    let (x0, eps) = (root.mid.as_slice(), root.rad.as_slice());
    let topk = clip_interm_topk();
    // The inherited set, opened for this node's own generation. `None` in the
    // dark arm, which is what keeps the loop below allocation-identical.
    let mut acc = stateful.then(|| {
        let mut a = acc_in.cloned().unwrap_or_default();
        a.open_generation();
        a
    });
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.clip_rows.is_none() {
            continue;
        }
        // #mr-close-telemetry: the `_crossing` variant is the same call plus
        // the largest crossing, recorded below as the emptiness certificate's
        // slack (telemetry only).
        let (tight, crossed, max_crossing) = super::clip::tighten_intermediate_acc_crossing(
            layer,
            li,
            &hs,
            x0,
            eps,
            topk,
            acc.as_ref(),
        );
        if crossed > 0 {
            if let Some(a) = acc.as_mut() {
                // STATEFUL: refuse rather than close. A crossing whose lower
                // endpoint is INHERITED would, if trusted, discharge a
                // non-empty domain — the one failure mode this project cannot
                // absorb. `tighten_intermediate_acc` already dropped the
                // crossed neurons from `tight`, so continuing simply keeps the
                // parent intervals. See `ClipAcc::merge`.
                a.note_refused(crossed);
            } else {
                // VERIFY (stateless arm, unchanged): crossed bounds certify the
                // region empty. Stop tightening — the caller discharges the
                // whole domain.
                super::clip::report_infeasible(li, crossed);
                dg.infeasible = true;
                dg.infeasible_slack = max_crossing;
                return None;
            }
        }
        // Fold this visit's fresh tightening into the accumulated set. Every
        // entry is an intersection of valid enclosures, so the stored interval
        // still encloses the neuron's true range over this domain's region.
        if let Some(a) = acc.as_mut() {
            for t in &tight {
                a.merge(li, t.idx, t.l, t.u);
            }
        }
        // Neurons the ACCUMULATED set narrows at this layer — the union of
        // everything the ancestor chain proved with what this visit added. In
        // the dark arm this is exactly `tight`, in original order.
        let entries: Vec<(usize, f64, f64)> = match acc.as_ref() {
            Some(a) => a.layer_entries(li).collect(),
            None => tight.iter().map(|t| (t.idx, t.l, t.u)).collect(),
        };
        if entries.is_empty() {
            continue;
        }
        let gv = dg.layers.entry(li).or_insert_with(|| GateVecs {
            alpha: layer.alpha.clone(),
            s: layer.s.clone(),
            c: layer.c.clone(),
            ms: layer.ms.clone(),
        });
        // Rebuild only the touched neurons, through the certified path.
        let (mut lv, mut uv) = (layer.l.clone(), layer.u.clone());
        let mut touched = Vec::with_capacity(entries.len());
        for &(idx, tl, tu) in &entries {
            // A neuron this domain SPLIT already carries an exact fixed line;
            // do not overwrite it with a relaxation.
            if trunk_splits
                .iter()
                .any(|&(l2, p2, _)| l2 == li && root.layers[l2].unst.get(p2).copied() == Some(idx))
            {
                continue;
            }
            if idx >= lv.len() {
                continue;
            }
            lv[idx] = tl;
            uv[idx] = tu;
            touched.push(idx);
        }
        if touched.is_empty() {
            continue;
        }
        let (alpha2, mut s2, mut c2) = super::root::gates_from_box(&lv, &uv);
        if root.mode.outward() {
            super::root::repair_upper_lines(&lv, &uv, &mut s2, &mut c2);
        }
        for &i in &touched {
            gv.alpha[i] = alpha2[i];
            gv.s[i] = s2[i];
            gv.c[i] = c2[i];
            // Root derives `ms = max(alpha, s)` (root.rs), and `s = u/(u-l)`
            // MOVES when the bounds tighten — so it must be re-derived, not
            // carried over. This also covers a neuron that became stable:
            // active gives (1,1) -> 1, inactive (0,0) -> 0.
            gv.ms[i] = alpha2[i].max(s2[i]);
        }
        super::clip::report_interm(li, touched.len(), tight.len());
    }
    let acc = acc?;
    // R9: shrink is measured against the ROOT interval, so "accumulated over 30
    // levels" is distinguishable from "re-derived the same 1.6e-4".
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    for (li, layer) in root.layers.iter().enumerate() {
        for (idx, lo, hi) in acc.layer_entries(li) {
            if idx >= layer.l.len() {
                continue;
            }
            let w0 = layer.u[idx] - layer.l[idx];
            if w0 > 0.0 && (hi - lo).is_finite() {
                let r = (1.0 - (hi - lo) / w0).clamp(0.0, 1.0);
                sum += r;
                max = max.max(r);
                n += 1;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = if n == 0 { 0.0 } else { sum / n as f64 };
    super::clip::report_acc(
        trunk_splits.len(),
        acc.generation(),
        acc.len(),
        acc.fresh(),
        acc.refused(),
        mean,
        max,
    );
    Some(std::sync::Arc::new(acc))
}

/// #clip-and-verify intermediate tightening: default OFF.
///
/// #closure-adaptive: this was going to be a rung of the runtime ladder — it is
/// the site that repairs `domain_gates_split_only`'s purely LOCAL edit, so it
/// was the obvious candidate. The measurement removed it. `mean_rel_shrink`
/// reads 8.4e-16 to 1.6e-4 with the mechanism provably armed, the closure
/// numbers come out bit-identical to baseline, and the cause is geometric: one
/// split is one halfspace in 3072 dimensions, which cuts the split neuron 63%
/// and everything near-orthogonal to it by nothing. See `closure.rs`.
/// DELIVERABILITY. This used to be `ny_levers::read`, i.e. env-only, and
/// `vnncomp_scripts/run_instance.sh` exports exactly ONE `NY_*` — so on a scored
/// run the whole Clip-and-Verify half was unreachable BY CONSTRUCTION. The
/// typed key `margin_row.clip_interm` is the only route that can arm it in
/// competition; env still wins in BOTH directions for research A/Bs. No shipped
/// config sets the key, so this is byte-identical today.
///
/// READ PER CALL, never latched — and that is load-bearing, not stylistic.
///
/// `preset_gates::margin_row_clip_interm_enabled` is already per-call
/// (`bool_gate!` expands to `resolve_bool(&decl, &cell)`: env first in both
/// directions, then the typed preset cell, then the declaration default), so a
/// `OnceLock` around it buys nothing and costs correctness: ANY read before the
/// vnncomp bridge calls `set_clip_interm_preset` freezes the dark value for the
/// whole process, and the gate then reads as armed while doing nothing. That is
/// the exact shape that left three other mechanisms silently inert.
///
/// This is not hypothetical. A latched version of this function shipped briefly
/// during the #clip-stateful merge and turned all three of the accumulator's
/// moat tests into `fixture retained no clip rows` — an earlier test in the same
/// binary had already latched the gate dark, so the fixture's arming could not
/// take. The tests caught it; a scored run would not have, because there the
/// first read happens to fall after the bridge. Do not reintroduce the latch.
fn clip_interm_enabled() -> bool {
    crate::preset_gates::margin_row_clip_interm_enabled()
}

/// #clip-stateful: accumulate the Clip-and-Verify intermediate tightening per
/// domain and inherit it parent -> child, instead of recomputing from the root
/// frame and discarding it (`margin_row.clip_stateful`; env wins both ways).
///
/// Default OFF. Inert unless `clip_interm_enabled()` is also armed — the
/// accumulator has nothing to accumulate otherwise.
fn clip_stateful_enabled() -> bool {
    ny_levers::read_over_config(
        &ny_levers::decls::margin_row::MARGIN_ROW_CLIP_STATEFUL,
        Some(ny_levers::LeverValue::Bool(
            super::clip_stateful_preset_armed(),
        )),
    )
    .map(|r| r.value.as_bool())
    .unwrap_or(false)
}

/// How many unstable neurons per layer to re-concretize. Matches
/// alpha-beta-CROWN's `bab.clip.interm_topk` default of 20.
fn clip_interm_topk() -> usize {
    crate::preset_gates::margin_row_clip_topk()
}

fn domain_gates_split_only(root: &RootGates, trunk_splits: &[(usize, usize, i8)]) -> DomainGates {
    let mut dg = DomainGates::default();
    for &(li, pos, d) in trunk_splits {
        let rec = &root.layers[li];
        let idx = rec.unst[pos];
        let gv = dg.layers.entry(li).or_insert_with(|| GateVecs {
            alpha: rec.alpha.clone(),
            s: rec.s.clone(),
            c: rec.c.clone(),
            ms: rec.ms.clone(),
        });
        if d > 0 {
            gv.alpha[idx] = 1.0;
            gv.s[idx] = 1.0;
            gv.c[idx] = 0.0;
            gv.ms[idx] = 1.0;
        } else {
            gv.alpha[idx] = 0.0;
            gv.s[idx] = 0.0;
            gv.c[idx] = 0.0;
            gv.ms[idx] = 0.0;
        }
    }
    dg
}

fn row(m: &Array2<f64>, j: usize) -> &[f64] {
    let r = m.ncols();
    &m.as_slice().expect("standard layout")[j * r..(j + 1) * r]
}

// Cooperative deadline batches are bounded by scalar work, and matrix scratch
// is capped before entering the optional pass. Allocation/reservation and task
// scheduling are not preemptible; no strict wall-time bound is claimed.
const DEADLINE_MAX_COLS: usize = 64;
const DEADLINE_MAX_OPS: usize = 4096;
const DEADLINE_MAX_CELLS: usize = 16 * 1024 * 1024;
const DEADLINE_POLL_ITEMS: usize = 256;

#[inline]
fn check_optional_deadline(deadline: Option<Instant>) -> Result<()> {
    deadline.map_or(Ok(()), check_deadline)
}

fn check_clock_with_checkpoint(
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    check_optional_deadline(deadline)?;
    if let Some(checkpoint) = checkpoint {
        checkpoint()?;
    }
    Ok(())
}

#[inline]
fn poll_deadline(deadline: Option<Instant>, index: usize) -> Result<()> {
    if index.is_multiple_of(DEADLINE_POLL_ITEMS) {
        check_optional_deadline(deadline)?;
    }
    Ok(())
}

#[inline]
fn axpy_checked(acc: &mut [f64], w: f64, src: &[f64], deadline: Option<Instant>) -> Result<()> {
    match deadline {
        // The enclosing Gemm / bias row loop checks at least every 256 rows.
        // At the optional entry's R <= 64 cap this bounds a coefficient batch
        // by 16K multiply-add terms without a clock read per nonzero weight.
        Some(at) if acc.len() > DEADLINE_MAX_COLS => axpy_with_deadline(acc, w, src, at),
        _ => {
            axpy(acc, w, src);
            Ok(())
        }
    }
}

// Relative-error formulas alone are insufficient for arbitrary f64 seeds:
// a nominal multiply can lose up to eta/2 in gradual underflow, and an error
// multiply rounded to zero can be amplified by a later source weight. The
// Outward per-column lane therefore carries eta for EACH underflowing nominal product,
// and widens EACH nonnegative error product in that range before propagation.
// Normal-product error arithmetic and Parity runs remain unchanged.
//
// Positive sums in the gradual-underflow range are exact (integer eta
// lattice); normal sum/multiply errors retain the existing relative charge.
// The final multiplication in certify_up / slack16 already has an immediate
// next_up, which also dominates its <= eta/2 absolute underflow rounding.
// Each nominal dot's eta dust is twice the <= eta/2 lost product, covering
// internal accumulation amplification under the preflight gamma <= 0.25
// premise; the unchanged post-dot certifier prices the error dot itself.
#[inline]
pub(super) fn error_mul(a: f64, b: f64, repair_underflow: bool) -> f64 {
    if repair_underflow {
        underflow_product_up(a, b)
    } else {
        a * b
    }
}

// #mr-errconv-screen for the per-element repaired charges (the conv bias arm
// and concretize). On a VALID row -- every operand finite, every error budget
// `>= 0`, magnitudes bounded so no sum or product of the row overflows -- the
// repaired helpers reduce to the branch-free forms below, bit for bit, so the
// row runs them without the scalar refusal / widening / dust branches (and
// vectorizes). Any other row keeps the scalar repaired loop. The two loops
// compute the same function; the screen only chooses which runs.

const ETA: f64 = f64::from_bits(1);

// review(mrcost): test-only per-thread engagement receipts of the screened
// row sites that run on the pass's calling thread: `[per-column bias rows,
// envelope bias rows, concretize rows]` taken by the branch-free forms. The
// bias arm and concretize loops are sequential, so a whole-engine test can
// prove each site was reached without reading the process-global counters
// other concurrently running tests also move.
#[cfg(test)]
thread_local! {
    pub(super) static ERRCONV_TEST_ROWS: std::cell::Cell<[u64; 3]> =
        const { std::cell::Cell::new([0; 3]) };
}

#[cfg(test)]
fn add_row(mut rows: [u64; 3], site: usize) -> [u64; 3] {
    rows[site] += 1;
    rows
}

/// [`underflow_product_up`] for finite operands that are not negative (`-0.0`
/// allowed): its refusal branch cannot fire, and `p = a * b` is widened to
/// `next_up(p)` exactly when `a != 0 && b != 0 && p < MIN_POSITIVE`; then both
/// operands are positive, so `p` is `+0.0` or positive subnormal and its
/// `next_up` is `bits + 1` (the 0/1 flag added to the bits; `wrapping_add`
/// only keeps the release profile's overflow check out of the loop).
// Non-short-circuit `&` is the point (branch-free); see the block comment above.
#[allow(clippy::needless_bitwise_bool)]
#[inline(always)]
pub(super) fn up_mul_valid(a: f64, b: f64) -> f64 {
    let p = a * b;
    let widen = u64::from((a != 0.0) & (b != 0.0) & (p < f64::MIN_POSITIVE));
    f64::from_bits(p.to_bits().wrapping_add(widen))
}

/// [`underflow_product_up`] for EVERY input, branch-free: the refusal
/// (`+inf` for a non-finite or negative operand) is selected by mask, and on
/// the remaining operands it is [`up_mul_valid`].
// Non-short-circuit `&` is the point (branch-free), and `!(x < 0.0)` mirrors
// `underflow_product_up`'s refusal predicate exactly (it admits `-0.0`).
#[allow(clippy::needless_bitwise_bool, clippy::neg_cmp_op_on_partial_ord)]
#[inline(always)]
pub(super) fn up_mul_any(a: f64, b: f64) -> f64 {
    let bad = !(a.is_finite() & b.is_finite() & !(a < 0.0) & !(b < 0.0));
    select_f64(bad, f64::INFINITY, up_mul_valid(a, b))
}

/// One element of the uniform ReLU gate transform's repaired error update
/// (`run_collect_lane`, PerCol lane): `slack16(dust(dust(error_mul(e +
/// error_mul(4u, |v|), ms) + error_mul(4u, |l'|), vp, sp), vn, sn))`,
/// branch-free and equal to the scalar form for EVERY input: each
/// `error_mul(.., .., true)` is [`up_mul_any`]; the charge is a sum of values
/// in `[+0, +inf]` (the second term is `>= +0` whatever the first), so each
/// dust step's `next_up(charge + eta)` and the final `slack16`'s
/// `next_up(x * (1 + 16u))` see a `+0`, positive or `+inf` argument, where
/// [`next_up`] is [`nextup_pos`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn relu_error_update_branch_free(
    e: f64,
    v: f64,
    lnew: f64,
    vp: f64,
    vn: f64,
    sp: f64,
    sn: f64,
    mj: f64,
) -> f64 {
    let charge =
        up_mul_any(e + up_mul_any(4.0 * UNIT, v.abs()), mj) + up_mul_any(4.0 * UNIT, lnew.abs());
    let charge = select_f64(dust_flag(vp, sp), nextup_pos(charge + ETA), charge);
    let charge = select_f64(dust_flag(vn, sn), nextup_pos(charge + ETA), charge);
    nextup_pos(charge * (1.0 + 16.0 * UNIT))
}

/// [`error_axpy_checked`]'s repaired loop (no deadline), branch-free and equal
/// for EVERY input: `term = underflow_product_up(w, e)` is [`up_mul_any`]
/// (`+0`, positive or `+inf`, never NaN), so the dust step's
/// `next_up(term + eta)` is [`nextup_pos`].
#[inline]
pub(super) fn error_axpy_branch_free(acc: &mut [f64], w: f64, error: &[f64], nominal: &[f64]) {
    for ((a, &e), &v) in acc.iter_mut().zip(error).zip(nominal) {
        let term = up_mul_any(w, e);
        *a += select_f64(dust_flag(w, v), nextup_pos(term + ETA), term);
    }
}

/// Exactly `nominal_product_dust(a, b) != 0` ([`nominal_product_dust`]) for
/// every input (the same predicate, evaluated without short-circuit branches).
// Non-short-circuit `&` is the point (branch-free); see the block comment above.
#[allow(clippy::needless_bitwise_bool)]
#[inline(always)]
pub(super) fn dust_flag(a: f64, b: f64) -> bool {
    (a != 0.0) & (b != 0.0) & ((a * b).abs() < f64::MIN_POSITIVE)
}

/// [`next_up`] for a positive or `+inf` argument: `bits + 1`, or itself at
/// `+inf` (its positive-finite and infinite branches).
#[inline(always)]
pub(super) fn nextup_pos(x: f64) -> f64 {
    f64::from_bits(x.to_bits().wrapping_add(u64::from(x < f64::INFINITY)))
}

/// `if c { a } else { b }` through an all-ones / all-zeros mask (no branch).
#[inline(always)]
pub(super) fn select_f64(c: bool, a: f64, b: f64) -> f64 {
    let m = 0u64.wrapping_sub(u64::from(c));
    f64::from_bits((a.to_bits() & m) | (b.to_bits() & !m))
}

/// Bound under which every sum and product of a screened bias row is finite
/// (`2^400`): `(er + gam_b*|l|) * |bias| + (|l| + er) * berr <= 2^803`.
const SCREEN_MAGNITUDE_CAP: f64 = 2.582_249_878_086_908_6e120;

/// The conv bias arm's row `j` is VALID for [`bias_row_charges_branch_free`]:
/// `|bias|` and `berr` finite, `berr >= 0`, both `<= 2^400`; every `l` finite
/// with `|l| <= 2^400`; every error budget finite, `>= 0` (`-0.0` allowed) and
/// `<= 2^400`; every running bias `b` finite (it is read only when
/// `bias != 0`, after the row's `axpy`). One fused pass.
// Non-short-circuit `&` is the point (branch-free), and `!(x < 0.0)` mirrors
// `underflow_product_up`'s refusal predicate exactly (it admits `-0.0`).
#[allow(clippy::needless_bitwise_bool, clippy::neg_cmp_op_on_partial_ord)]
pub(super) fn bias_row_is_valid(lr: &[f64], er: &[f64], b: &[f64], bias: f64, berr: f64) -> bool {
    let cap = SCREEN_MAGNITUDE_CAP;
    if !(bias.abs() <= cap && berr >= 0.0 && berr <= cap) || lr.len() != er.len() {
        return false;
    }
    let mut ok = true;
    for (&l, &e) in lr.iter().zip(er) {
        ok &= (l.abs() <= cap) & (e <= cap) & !(e < 0.0);
    }
    if bias != 0.0 {
        ok &= b.len() == lr.len();
        for &v in b {
            ok &= v.is_finite();
        }
    }
    ok
}

/// The repaired bias-arm charges of one VALID row, branch-free: per column
/// `charge = U(er + U(gam_b, |l|), |bias|) + U(|l| + er, berr)` (`U` =
/// [`underflow_product_up`]), plus the dust `next_up(charge + eta)` when
/// `bias * l` underflows, then (bias != 0) the carry `U(2u, |b|)` -- exactly
/// the scalar loop's `error_mul` / `add_nominal_dust` sequence, since on a
/// valid row every `U` operand is finite and not negative (`er + t` and
/// `|l| + er` are bounded sums of such), `charge` is finite and not negative
/// so `charge + eta` is positive, and the accumulation order is unchanged.
pub(super) fn bias_row_charges_branch_free(
    eb: &mut [f64],
    lr: &[f64],
    er: &[f64],
    b: &[f64],
    bias: f64,
    berr: f64,
    gam_b: f64,
) {
    let ab = bias.abs();
    for ((ebv, &l), &e) in eb.iter_mut().zip(lr).zip(er) {
        let mag = l.abs();
        let charge = up_mul_valid(e + up_mul_valid(gam_b, mag), ab) + up_mul_valid(mag + e, berr);
        *ebv += select_f64(dust_flag(bias, l), nextup_pos(charge + ETA), charge);
    }
    if bias != 0.0 {
        let carry = 2.0 * UNIT;
        for (ebv, &v) in eb.iter_mut().zip(b) {
            *ebv += up_mul_valid(carry, v.abs());
        }
    }
}

/// [`bias_row_is_valid`] for the envelope lane, whose per-column error entry
/// is the row's envelope value `ev` (finite, `>= 0`, `<= 2^400`).
// Non-short-circuit `&` is the point (branch-free), and `!(x < 0.0)` mirrors
// `underflow_product_up`'s refusal predicate exactly (it admits `-0.0`).
#[allow(clippy::needless_bitwise_bool, clippy::neg_cmp_op_on_partial_ord)]
pub(super) fn bias_row_is_valid_env(lr: &[f64], ev: f64, b: &[f64], bias: f64, berr: f64) -> bool {
    let cap = SCREEN_MAGNITUDE_CAP;
    if !(bias.abs() <= cap && berr >= 0.0 && berr <= cap && ev <= cap && !(ev < 0.0)) {
        return false;
    }
    let mut ok = true;
    for &l in lr {
        ok &= l.abs() <= cap;
    }
    if bias != 0.0 {
        ok &= b.len() == lr.len();
        for &v in b {
            ok &= v.is_finite();
        }
    }
    ok
}

/// [`bias_row_charges_branch_free`] for the envelope lane: the per-column
/// error entry is the row's envelope value `ev`, exactly as in its scalar
/// loop (`error_mul(ev + error_mul(gam_b, |l|), |bias|) + error_mul(|l| + ev,
/// berr)`, the dust, then the carry). Same proof, `ev` in place of `er`.
pub(super) fn bias_row_charges_branch_free_env(
    eb: &mut [f64],
    lr: &[f64],
    ev: f64,
    b: &[f64],
    bias: f64,
    berr: f64,
    gam_b: f64,
) {
    let ab = bias.abs();
    for (ebv, &l) in eb.iter_mut().zip(lr) {
        let mag = l.abs();
        let charge = up_mul_valid(ev + up_mul_valid(gam_b, mag), ab) + up_mul_valid(mag + ev, berr);
        *ebv += select_f64(dust_flag(bias, l), nextup_pos(charge + ETA), charge);
    }
    if bias != 0.0 {
        let carry = 2.0 * UNIT;
        for (ebv, &v) in eb.iter_mut().zip(b) {
            *ebv += up_mul_valid(carry, v.abs());
        }
    }
}

/// Concretize's input row `i` is VALID for [`concretize_row_branch_free`]:
/// `xa` finite and `>= 0`, every coefficient finite, every error budget
/// finite and `>= 0` (`-0.0` allowed). `mid`/`rad` need nothing: the dust
/// predicate is evaluated exactly for any value. One fused pass per lane.
// Non-short-circuit `&` is the point (branch-free), and `!(x < 0.0)` mirrors
// `underflow_product_up`'s refusal predicate exactly (it admits `-0.0`).
#[allow(clippy::needless_bitwise_bool, clippy::neg_cmp_op_on_partial_ord)]
pub(super) fn concretize_row_is_valid(arow: &[f64], erow: Option<&[f64]>, xa: f64) -> bool {
    if !(xa.is_finite() && xa >= 0.0) {
        return false;
    }
    let mut ok = true;
    for &a in arow {
        ok &= a.is_finite();
    }
    if let Some(es) = erow {
        for &e in es {
            ok &= e.is_finite() & !(e < 0.0);
        }
    }
    ok
}

/// Concretize's repaired accumulation of one VALID input row, branch-free:
/// `v += a*mid`, `rr += |a|*rad`, `tabs += U(|a|, xa)`, the two dust steps
/// `dust = next_up(dust + eta)` when `a*mid` / `|a|*rad` underflow, and
/// `pen += U(e, xa)` -- the scalar loop's operations in its order. `dust`
/// starts at `0` and only ever takes such steps, so `dust + eta` is positive.
#[allow(clippy::too_many_arguments)]
pub(super) fn concretize_row_branch_free(
    v: &mut [f64],
    rr: &mut [f64],
    tabs: &mut [f64],
    dust: &mut [f64],
    pen: &mut [f64],
    arow: &[f64],
    erow: Option<&[f64]>,
    m: f64,
    rd: f64,
    xa: f64,
) {
    for ri in 0..arow.len() {
        let av = arow[ri];
        v[ri] += av * m;
        let aa = av.abs();
        rr[ri] += aa * rd;
        tabs[ri] += up_mul_valid(aa, xa);
        let d = dust[ri];
        let d = select_f64(dust_flag(av, m), nextup_pos(d + ETA), d);
        dust[ri] = select_f64(dust_flag(aa, rd), nextup_pos(d + ETA), d);
    }
    if let Some(es) = erow {
        for (p, &e) in pen.iter_mut().zip(es) {
            *p += up_mul_valid(e, xa);
        }
    }
}

#[inline]
fn add_nominal_dust(charge: f64, a: f64, b: f64, repair_underflow: bool) -> f64 {
    if repair_underflow {
        let dust = nominal_product_dust(a, b);
        if dust != 0.0 {
            // UP locally, so this additional addition does not increase the
            // existing relative-error depth of a running error accumulation.
            return next_up(charge + dust);
        }
    }
    charge
}

fn error_axpy_checked(
    acc: &mut [f64],
    w: f64,
    error_src: &[f64],
    nominal_src: &[f64],
    deadline: Option<Instant>,
    repair_underflow: bool,
) -> Result<()> {
    if !repair_underflow {
        axpy(acc, w, error_src);
        return Ok(());
    }
    debug_assert!(w >= 0.0);
    debug_assert_eq!(acc.len(), nominal_src.len());
    if deadline.is_none() && errconv_screens_enabled() {
        // #mr-errconv-screen: the same repaired terms, branch-free.
        error_axpy_branch_free(acc, w, error_src, nominal_src);
        return Ok(());
    }
    for (i, ((a, &e), &v)) in acc.iter_mut().zip(error_src).zip(nominal_src).enumerate() {
        poll_deadline(deadline, i)?;
        let term = underflow_product_up(w, e);
        *a += add_nominal_dust(term, w, v, true);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn exception_coefficient_error(
    eorig: f64,
    orig: f64,
    vp: f64,
    vn: f64,
    a: f64,
    slope: f64,
    max_slope: f64,
    lnew: f64,
    lower: bool,
    repair_underflow: bool,
) -> f64 {
    if !repair_underflow {
        return slack16((eorig + 4.0 * UNIT * orig.abs()) * max_slope + 4.0 * UNIT * lnew.abs());
    }
    let charge = error_mul(
        eorig + error_mul(4.0 * UNIT, orig.abs(), true),
        max_slope,
        true,
    ) + error_mul(4.0 * UNIT, lnew.abs(), true);
    let (sp, sn) = if lower { (a, slope) } else { (slope, a) };
    let charge = add_nominal_dust(charge, vp, sp, true);
    slack16(add_nominal_dust(charge, vn, sn, true))
}

#[allow(clippy::too_many_arguments)]
fn exception_bias_error(
    eb: f64,
    running_b: f64,
    eorig: f64,
    orig: f64,
    part: f64,
    delta_c: f64,
    bpart: f64,
    repair_underflow: bool,
) -> f64 {
    if !repair_underflow {
        return slack16(
            eb + (eorig + 4.0 * UNIT * orig.abs()) * delta_c.abs() + 4.0 * UNIT * bpart.abs(),
        );
    }
    let t1 = error_mul(
        eorig + error_mul(4.0 * UNIT, orig.abs(), true),
        delta_c.abs(),
        true,
    );
    let t2 = error_mul(4.0 * UNIT, bpart.abs(), true);
    // The nominal b += bpart rounds on the RUNNING bias magnitude, which
    // can be much larger than bpart. This charge fixes the old exception
    // helper's missing carry (the Conv/ReLU bias arms already carried it).
    let carry = error_mul(2.0 * UNIT, running_b.abs(), true);
    slack16(eb + add_nominal_dust(t1, part, delta_c, true) + t2 + carry)
}

pub(super) fn zeros_with_deadline(
    shape: (usize, usize),
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    zeros_with_checkpoint(shape, deadline, None)
}

pub(super) fn zeros_with_checkpoint(
    shape: (usize, usize),
    deadline: Option<Instant>,

    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<Array2<f64>> {
    let Some(at) = deadline else {
        return Ok(Array2::<f64>::zeros(shape));
    };
    check_clock_with_checkpoint(Some(at), checkpoint)?;
    let cells = shape.0.checked_mul(shape.1).ok_or_else(|| {
        NyError::UnsupportedConfiguration("margin_row: deadline scratch dimensions overflow".into())
    })?;
    if cells > DEADLINE_MAX_CELLS {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: deadline scratch exceeds the cell cap".into(),
        ));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(cells).map_err(|_| {
        NyError::InternalError("margin_row: deadline scratch reservation failed".into())
    })?;
    check_clock_with_checkpoint(Some(at), checkpoint)?;
    reconcile_timed_scratch_capacity(cells, values.capacity())?;
    while values.len() < cells {
        check_clock_with_checkpoint(Some(at), checkpoint)?;
        values.resize((values.len() + DEADLINE_POLL_ITEMS).min(cells), 0.0);
    }
    check_clock_with_checkpoint(Some(at), checkpoint)?;
    Array2::from_shape_vec(shape, values)
        .map_err(|e| NyError::InternalError(format!("margin_row: deadline scratch shape: {e}")))
}

/// The tile estimate pays exactly the requested scalar cells for each
/// temporary. A reserve that exposes additional Vec capacity cannot silently
/// consume another plane's allowance; refuse and drop it before publication.
/// Physical allocator metadata/RSS are outside this payload-capacity census.
fn reconcile_timed_scratch_capacity(requested: usize, actual: usize) -> Result<()> {
    let payload_bytes = |cells: usize| cells.checked_mul(size_of::<f64>());
    let budget_bytes = payload_bytes(requested).ok_or_else(|| {
        NyError::UnsupportedConfiguration("margin_row: scratch capacity byte overflow".into())
    })?;
    let required_bytes = payload_bytes(actual).unwrap_or(usize::MAX);
    if actual < requested {
        return Err(NyError::InternalError(
            "margin_row: deadline scratch reserve is smaller than requested".into(),
        ));
    }
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "margin_row deadline scratch capacity",
        });
    }
    Ok(())
}

pub(super) fn clone_with_deadline(
    src: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    clone_with_checkpoint(src, deadline, None)
}

pub(super) fn clone_with_checkpoint(
    src: &Array2<f64>,
    deadline: Option<Instant>,

    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<Array2<f64>> {
    let Some(at) = deadline else {
        return Ok(src.clone());
    };
    check_clock_with_checkpoint(Some(at), checkpoint)?;
    let mut out = zeros_with_checkpoint(src.dim(), Some(at), checkpoint)?;
    let dst = out.as_slice_mut().expect("standard layout");
    let src = src.as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(
            "margin_row: deadline copy requires standard layout".into(),
        )
    })?;
    for (d, s) in dst
        .chunks_mut(DEADLINE_POLL_ITEMS)
        .zip(src.chunks(DEADLINE_POLL_ITEMS))
    {
        check_clock_with_checkpoint(Some(at), checkpoint)?;
        d.copy_from_slice(s);
    }
    check_clock_with_checkpoint(Some(at), checkpoint)?;
    Ok(out)
}

pub(super) fn finite_with_deadline(values: &[f64], deadline: Option<Instant>) -> Result<()> {
    finite_with_checkpoint(values, deadline, None)
}

pub(super) fn finite_with_checkpoint(
    values: &[f64],
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    for (i, &v) in values.iter().enumerate() {
        if i.is_multiple_of(DEADLINE_POLL_ITEMS) {
            check_clock_with_checkpoint(deadline, checkpoint)?;
        }
        if !v.is_finite() {
            return Err(NyError::NumericalInstability(
                "margin_row: non-finite backward rows".into(),
            ));
        }
    }
    check_clock_with_checkpoint(deadline, checkpoint)
}

pub(super) fn combined_err_checked(
    l: &Array2<f64>,
    e: Option<&Array2<f64>>,
    g: f64,
    deadline: Option<Instant>,
    repair_underflow: bool,
) -> Result<Array2<f64>> {
    combined_err_with_checkpoint(l, e, g, deadline, repair_underflow, None)
}

pub(super) fn combined_err_with_checkpoint(
    l: &Array2<f64>,
    e: Option<&Array2<f64>>,
    g: f64,
    deadline: Option<Instant>,
    repair_underflow: bool,

    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<Array2<f64>> {
    if deadline.is_none() && !repair_underflow {
        return Ok(combined_err(l, e, g));
    }
    let mut comb = zeros_with_checkpoint(l.dim(), deadline, checkpoint)?;
    let cs = comb.as_slice_mut().expect("standard layout");
    let ls = l.as_slice().expect("standard layout");
    match e {
        Some(em) => {
            let es = em.as_slice().expect("standard layout");
            cs.par_chunks_mut(DEADLINE_POLL_ITEMS)
                .zip(
                    ls.par_chunks(DEADLINE_POLL_ITEMS)
                        .zip(es.par_chunks(DEADLINE_POLL_ITEMS)),
                )
                .try_for_each(|(c, (lr, er))| {
                    check_clock_with_checkpoint(deadline, checkpoint)?;
                    for ((cv, &lv), &ev) in c.iter_mut().zip(lr).zip(er) {
                        *cv = ev + underflow_product_up(g, lv.abs() + ev);
                    }
                    Ok::<(), NyError>(())
                })?;
        }
        None => {
            cs.par_chunks_mut(DEADLINE_POLL_ITEMS)
                .zip(ls.par_chunks(DEADLINE_POLL_ITEMS))
                .try_for_each(|(c, lr)| {
                    check_clock_with_checkpoint(deadline, checkpoint)?;
                    for (cv, &lv) in c.iter_mut().zip(lr) {
                        *cv = underflow_product_up(g, lv.abs());
                    }
                    Ok::<(), NyError>(())
                })?;
        }
    }
    check_clock_with_checkpoint(deadline, checkpoint)?;
    Ok(comb)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn merge_step_checked(
    state: &mut [Option<LaneMat>],
    id: usize,
    contrib: LaneMat,
    outward: bool,
    gtab: Option<&[usize]>,
    op: usize,
    eb: &[f64],
    deadline: Option<Instant>,
) -> Result<()> {
    merge_step_with_checkpoint(state, id, contrib, outward, gtab, op, eb, deadline, None)
}

pub(super) fn merge_step_with_checkpoint(
    state: &mut [Option<LaneMat>],
    id: usize,
    contrib: LaneMat,
    outward: bool,
    gtab: Option<&[usize]>,
    op: usize,
    eb: &[f64],
    deadline: Option<Instant>,

    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    let Some(at) = deadline else {
        merge_step(state, id, contrib, outward, gtab, op, eb);
        return Ok(());
    };
    check_clock_with_checkpoint(Some(at), checkpoint)?;
    // The optional entry refuses envelope publication before numerical work.
    if gtab.is_some() || contrib.env.is_some() {
        return Err(NyError::UnsupportedConfiguration(
            "margin_row: deadline merge requires per-column errors".into(),
        ));
    }
    match state[id].take() {
        None => state[id] = Some(contrib),
        Some(mut acc) => {
            if acc.env.is_some() || acc.l.dim() != contrib.l.dim() {
                return Err(NyError::InvalidSpec(
                    "margin_row: deadline merge shape/lane mismatch".into(),
                ));
            }
            acc.l
                .as_slice_mut()
                .expect("standard layout")
                .par_chunks_mut(DEADLINE_POLL_ITEMS)
                .zip(
                    contrib
                        .l
                        .as_slice()
                        .expect("standard layout")
                        .par_chunks(DEADLINE_POLL_ITEMS),
                )
                .try_for_each(|(a, c)| {
                    check_clock_with_checkpoint(Some(at), checkpoint)?;
                    for (av, &cv) in a.iter_mut().zip(c) {
                        *av += cv;
                    }
                    Ok::<(), NyError>(())
                })?;
            match (acc.e.as_mut(), contrib.e) {
                (Some(ea), Some(ec)) => {
                    ea.as_slice_mut()
                        .expect("standard layout")
                        .par_chunks_mut(DEADLINE_POLL_ITEMS)
                        .zip(
                            ec.as_slice()
                                .expect("standard layout")
                                .par_chunks(DEADLINE_POLL_ITEMS)
                                .zip(
                                    acc.l
                                        .as_slice()
                                        .expect("standard layout")
                                        .par_chunks(DEADLINE_POLL_ITEMS),
                                ),
                        )
                        .try_for_each(|(er, (cr, lr))| {
                            check_clock_with_checkpoint(Some(at), checkpoint)?;
                            for ((ev, &cv), &lv) in er.iter_mut().zip(cr).zip(lr) {
                                *ev =
                                    slack16(*ev + cv + underflow_product_up(2.0 * UNIT, lv.abs()));
                            }
                            Ok::<(), NyError>(())
                        })?;
                }
                (None, None) => {}
                _ => {
                    return Err(NyError::InvalidSpec(
                        "margin_row: deadline merge errors must agree".into(),
                    ))
                }
            }
            state[id] = Some(acc);
        }
    }
    check_clock_with_checkpoint(Some(at), checkpoint)
}

#[cfg(test)]
#[path = "engine_deadline_tests.rs"]
mod deadline_tests;

/// `E + g * (|L| + E)` elementwise (error input to a linear step).
/// `pub(crate)` for `#bi-patch` (the same expression on patch-local rows).
pub(crate) fn combined_err(l: &Array2<f64>, e: Option<&Array2<f64>>, g: f64) -> Array2<f64> {
    let mut comb = Array2::<f64>::zeros(l.raw_dim());
    let cols = l.ncols();
    let cs = comb.as_slice_mut().expect("standard layout");
    let ls = l.as_slice().expect("standard layout");
    match e {
        Some(em) => {
            let es = em.as_slice().expect("standard layout");
            cs.par_chunks_mut(cols)
                .zip(ls.par_chunks(cols).zip(es.par_chunks(cols)))
                .for_each(|(c, (lr, er))| {
                    for ((cv, &lv), &ev) in c.iter_mut().zip(lr).zip(er) {
                        *cv = ev + g * (lv.abs() + ev);
                    }
                });
        }
        None => {
            cs.par_chunks_mut(cols)
                .zip(ls.par_chunks(cols))
                .for_each(|(c, lr)| {
                    for (cv, &lv) in c.iter_mut().zip(lr) {
                        *cv = g * lv.abs();
                    }
                });
        }
    }
    comb
}

/// Accumulate a contribution into a tensor's lane state (elementwise add;
/// certified add-rounding into the error lane when outward).
///
/// #mr-errlane: on an envelope pass (`gtab` is `Some`) the contribution's
/// envelope is first regrouped to the destination tensor's grouping, and two
/// envelopes merge with the per-column formula at the group maxima
/// (`slack16(Ê_a + Ê_b + 2u·max|l_a + l_b|)`). With `gtab = None` (every
/// per-column pass) both envelopes are `None`; both branches repair the
/// fresh positive error product before it can be amplified downstream.
/// `pub(crate)` for `#bi-patch`, which merges its patch-local contributions
/// with this exact arithmetic (visibility only).
pub(crate) fn merge_into(
    state: &mut [Option<LaneMat>],
    id: usize,
    contrib: LaneMat,
    outward: bool,
    gtab: Option<&[usize]>,
) {
    let mut contrib = contrib;
    if let Some(gt) = gtab {
        if let Some(env) = contrib.env.take() {
            contrib.env = Some(env.regroup(gt[id]));
        }
    }
    match state[id].take() {
        None => state[id] = Some(contrib),
        Some(mut acc) => {
            acc.l += &contrib.l;
            match (acc.e.as_mut(), contrib.e) {
                (Some(ea), Some(ecb)) => {
                    let cols = ea.ncols();
                    let es = ea.as_slice_mut().expect("standard layout");
                    let cs = ecb.as_slice().expect("standard layout");
                    let ls = acc.l.as_slice().expect("standard layout");
                    es.par_chunks_mut(cols)
                        .zip(cs.par_chunks(cols).zip(ls.par_chunks(cols)))
                        .for_each(|(er, (cr, lr))| {
                            for ((ev, &cv), &lv) in er.iter_mut().zip(cr).zip(lr) {
                                *ev =
                                    slack16(*ev + cv + underflow_product_up(2.0 * UNIT, lv.abs()));
                            }
                        });
                }
                (None, None) => {}
                _ => debug_assert!(!outward, "error lanes must agree"),
            }
            match (acc.env.as_mut(), contrib.env) {
                (Some(ea), Some(ecb)) => {
                    let ecb = ecb.regroup(ea.grp);
                    let lam = errlane::group_absmax(&acc.l, ea.grp);
                    for ((ev, &cv), &lv) in ea.v.iter_mut().zip(&ecb.v).zip(&lam) {
                        *ev = slack16(*ev + cv + underflow_product_up(2.0 * UNIT, lv));
                    }
                }
                (None, None) => {}
                _ => debug_assert!(!outward, "envelope lanes must agree"),
            }
            state[id] = Some(acc);
        }
    }
}

#[cfg(test)]
#[path = "engine_input_products_tests.rs"]
mod input_products_tests;

#[cfg(test)]
#[path = "engine_prepared_products_tests.rs"]
mod prepared_products_tests;
