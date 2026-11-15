// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bbe — the UNTRUSTED search side's plain data types (ENGINE_PROGRAM §1.3-1.4).
//!
//! The engine program splits every hot loop into an untrusted SEARCH (f32, CPU
//! SIMD or a device, in the `ny-bbe` crate) and a trusted CERTIFY step (CPU
//! f64, Outward/Higham, existing code in `ny-propagate`). This module is the
//! only thing the two sides share, so it is deliberately small and plain:
//!
//! * [`Untrusted`] / [`Hint`] — the type discipline. A search value has no
//!   `Deref`, no conversion to `f64` and no comparison with zero; the only way
//!   out is [`Untrusted::into_hint`] followed by [`Hint::into_ingest`], and the
//!   only callers of the latter are INGEST GATES that clamp every value before
//!   a certified evaluation decides whether it is kept.
//! * [`SearchBackend`] — the injected backend trait. `ny-propagate` sees a
//!   backend only as `&dyn SearchBackend`; it never depends on `ny-bbe` (or on
//!   any CUDA crate), and `ny-bbe` depends on nothing but this crate. `ny-cli`
//!   constructs the backend and registers it (R2 / JE C9: ny-cli ->
//!   {ny-propagate, ny-bbe} -> ny-core).
//! * [`SearchNet`] — a lossless, plain-data copy of a lane's own IR (the
//!   margin-row `TwinNet` today): the op list, f64 weights and conv gather
//!   tables. The backend lowers it into its own search program (f32); the
//!   certifier never reads it (R3: "the certifier reads the lane IR, never
//!   `BoundProgram`"), so a lowering bug costs tightness or a refusal, never
//!   soundness.
//! * V2 — [`GraphSearchNet`], [`GraphRootRequest`], [`GraphAlphaHint`]: the
//!   vit lane's graph search copy and its root α search (module [`graph`]).
//! * M7 — [`GraphBabRequest`], [`GraphDomainHint`]: the graph β-CROWN lane's
//!   BATCHED domain search over the same [`GraphSearchNet`] (the root is the
//!   domain with no split): per domain, ReLU slopes and one Lagrange
//!   multiplier per split literal, proposed as hints for the lane's ingest
//!   gate and ONE certified evaluation each (module [`graph_bab`]).
//! * M4 — [`FdSearchProblem`], [`LevelChunk`], [`LevelOut`]: the frozen-dual
//!   tier's level search. A backend scores every node of a level chunk at four
//!   dual points in its own arithmetic and proposes the best; the lane
//!   re-scores a proposal in CPU f64 (its unchanged certified evaluator) and
//!   prunes only on that, splitting the node otherwise. THE DEVICE NEVER
//!   CERTIFIES.
//! * M6 — [`WaveProblem`], [`WaveBatch`], [`WaveOut`], [`BranchOut`]: the
//!   margin-row WAVE tree's search. A backend bounds a wave of sub-domains
//!   with output-only beta-CROWN (certified boxes frozen, splits pinned) and
//!   ranks their split candidates; the lane certifies a claimed domain with
//!   its own Outward evaluator (hints clamped at ingest) and splits every
//!   other domain and every claim it cannot certify.
//! * M8 — [`FlProblem`], [`FlBatch`], [`FlOut`] (module [`forward_linear`]):
//!   the lowdim lane's FORWARD-LINEAR wave search. A backend bounds a whole
//!   wave of input-split domains (forward-linear intermediates, a CROWN
//!   objective pass) in its own arithmetic; the lane re-bounds only the
//!   domains a hint CLAIMS with its own certified f64 forms and closes on the
//!   unchanged strict `lb > thr`, splitting every other domain into its exact
//!   halves.
//!
//! Every backend method may refuse ([`Refusal`]); a refusal is never a verdict
//! and the lane then runs its legacy phase unchanged.

use std::fmt;
use std::time::Instant;

pub mod fallback;
pub mod forward_linear;
pub mod graph;
pub mod graph_bab;

pub use fallback::{FallbackBackend, WaveRoute, WaveSwitch};
pub use forward_linear::{
    FlBatch, FlFamily, FlHandle, FlLinear, FlOp, FlOpConsts, FlOut, FlOutWriter, FlProblem,
    FlProgram, FlStats, FlView,
};
pub use graph::{
    GraphAlphaHint, GraphRootOutcome, GraphRootRequest, GraphRootStats, GraphSearchNet,
    GraphSearchOp, SparseTerm, TensorBox, UnaryKind,
};
pub use graph_bab::{
    GraphBabOutcome, GraphBabRequest, GraphBabStats, GraphDomainHint, GraphDomainRequest,
    GraphDomainSplit, GraphDomainStats,
};

// ---------------------------------------------------------------------------
//  The type discipline
// ---------------------------------------------------------------------------

/// A value produced by UNTRUSTED search.
///
/// It deliberately implements no `Deref`, no `Into<f64>`, no `PartialOrd`
/// and no arithmetic, so it cannot be compared with zero or folded into a
/// bound by accident. `Display` is provided for telemetry lines only.
#[must_use]
pub struct Untrusted<T>(T);

impl<T> Untrusted<T> {
    /// Wrap a search output (any backend may construct one).
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// Hand the value on as a HINT for an ingest gate.
    pub fn into_hint(self) -> Hint<T> {
        Hint(self.0)
    }
}

impl<T: fmt::Display> fmt::Display for Untrusted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<T: fmt::Debug> fmt::Debug for Untrusted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Untrusted").field(&self.0).finish()
    }
}

/// A search output on its way into an INGEST GATE.
///
/// The only exit is [`Hint::into_ingest`]. Every caller of it must clamp each
/// value into its valid set (for a ReLU lower slope: finite, in `[0, 1]`, only
/// on neurons unstable in the CERTIFIED boxes) and then let an unchanged
/// certified evaluation decide whether anything is kept. Grepping for
/// `into_ingest` lists every ingest gate in the tree.
#[must_use]
pub struct Hint<T>(T);

impl<T> Hint<T> {
    /// Surrender the hinted value to an ingest gate. See the type docs for the
    /// caller's obligations.
    pub fn into_ingest(self) -> T {
        self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for Hint<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Hint").field(&self.0).finish()
    }
}

/// Why a backend declined a request. Never a verdict: the lane runs its
/// legacy phase unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The backend does not implement this method.
    Unsupported(&'static str),
    /// The request or the network failed the backend's structural validation.
    Invalid(String),
    /// Nothing to search (no objective, no unstable neuron, ...).
    Nothing(&'static str),
    /// The deadline passed before the search had a complete result.
    Deadline,
    /// The working set does not fit the admitted memory.
    Memory {
        /// Estimated bytes.
        need: u64,
        /// Admitted bytes.
        cap: u64,
    },
    /// A DEVICE fault (a driver error, or a panic in the device code): the
    /// backend has latched itself off for the process, so every later call
    /// refuses too. Only this refusal (never a host-side validation refusal)
    /// may switch a tree to another backend ([`FallbackBackend`]).
    Fault(String),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::Invalid(why) => write!(f, "invalid: {why}"),
            Self::Nothing(why) => write!(f, "nothing to search: {why}"),
            Self::Deadline => write!(f, "deadline reached before a complete result"),
            Self::Memory { need, cap } => write!(f, "memory: need {need} B > cap {cap} B"),
            Self::Fault(why) => write!(f, "device fault: {why}"),
        }
    }
}

/// Arithmetic a backend searches in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPrecision {
    /// IEEE binary32, round to nearest, no directed rounding.
    F32,
    /// IEEE binary64, round to nearest (test/reference backends).
    F64,
}

/// What a backend is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendCaps {
    /// A host backend.
    Cpu {
        /// Worker threads it will use (0 = the host's available parallelism).
        threads: usize,
        /// Rows evaluated per SIMD block.
        lanes: usize,
        /// Search arithmetic.
        precision: SearchPrecision,
    },
    /// A CUDA device backend (`ny-cuda::search`), in a context of its own.
    /// Zero fields mean "not probed yet": a backend initialises lazily, on its
    /// first upload, and never at construction.
    Cuda {
        /// `major * 10 + minor` compute capability (0 = not probed).
        sm: u32,
        /// The device shares the host's memory (GB10) rather than owning a
        /// discrete pool (A10G).
        unified: bool,
        /// Device memory the backend may lease, in bytes (0 = not probed).
        mem_bytes: u64,
    },
}

/// The injected UNTRUSTED search backend.
///
/// Implementations live in `ny-bbe` (CPU) and `ny-cuda::search` (device);
/// [`FallbackBackend`] composes a device with the CPU search behind it.
/// Nothing a backend returns is trusted: every output is [`Untrusted`], and a
/// caller consumes it only through an ingest gate. Methods default to
/// [`Refusal::Unsupported`], so a backend implements exactly what it serves
/// and each milestone adds at most three methods (ENGINE_PROGRAM §1.4).
pub trait SearchBackend: Send + Sync {
    /// Short stable name for telemetry (`cpu_f32`, ...).
    fn name(&self) -> &'static str;

    /// Root-search capabilities. Individual methods document other untrusted
    /// arithmetic they use; these capabilities do not grant verdict authority.
    fn caps(&self) -> BackendCaps;

    /// M5a: abc's joint-alpha ROOT search (CROWN init, spec prune,
    /// per-start-node alpha with an in-loop refresh of the unstable rows,
    /// early exits) over `net`, returning the final start node's slopes as
    /// a hint for the lane's existing certify-once ingest gate.
    fn root_joint_alpha(
        &self,
        net: &SearchNet,
        request: &RootSearchRequest,
    ) -> Result<RootSearchOutcome, Refusal> {
        let _ = (net, request);
        Err(Refusal::Unsupported("root_joint_alpha"))
    }

    /// Refine a box-support dual by concave breakpoint search. The returned
    /// nonnegative multipliers are hints; the caller must validate them and
    /// independently certify the bound against its own current problem. The
    /// CPU implementation uses untrusted f64 even when root search uses f32.
    fn refine_dual(&self, request: &DualSearchRequest<'_>) -> Result<DualSearchOutcome, Refusal> {
        let _ = request;
        Err(Refusal::Unsupported("refine_dual"))
    }

    /// V2 (#vit-root-alpha): root α search over a graph lane's
    /// [`GraphSearchNet`] with the lane's certified boxes frozen as REFERENCE
    /// boxes: ReLU slopes, the attention products' McCormick interpolation
    /// parameters and the fixed-index softmax's MulBinary planes, jointly,
    /// against the margin rows. Returns the best parameters as a hint for the
    /// lane's ingest gate (clamped, one certified evaluation, kept only when
    /// strictly better). The device never certifies.
    fn graph_root_alpha(
        &self,
        net: &GraphSearchNet,
        request: &GraphRootRequest,
    ) -> Result<GraphRootOutcome, Refusal> {
        let _ = (net, request);
        Err(Refusal::Unsupported("graph_root_alpha"))
    }

    /// M7 (#graph-lane-search): batched α/β search over a batch of graph
    /// β-CROWN domains of one [`GraphSearchNet`] (the root is the domain with
    /// no split), with the lane's certified boxes frozen as REFERENCE boxes
    /// (each domain may replace some of them with its own): per domain, the
    /// ReLU slopes and one Lagrange multiplier per split literal, jointly,
    /// against the objective rows. Returns one hint slot per domain for the
    /// lane's ingest gate (clamped, ONE certified evaluation per domain by the
    /// lane's own evaluator, kept per row only when strictly better). A domain
    /// the backend cannot read gets no hint; the others are unaffected. The
    /// device never certifies.
    fn graph_bab_batch(
        &self,
        net: &GraphSearchNet,
        request: &GraphBabRequest,
    ) -> Result<GraphBabOutcome, Refusal> {
        let _ = (net, request);
        Err(Refusal::Unsupported("graph_bab_batch"))
    }

    /// M4: take a search copy of one frozen-dual problem (the split rows in
    /// f32, see [`FdSearchProblem`]) for the [`Self::fd_level`] calls that
    /// follow. `lease` is the memory the backend may hold for the copy AND
    /// its per-call working set; a backend that needs more refuses with
    /// [`Refusal::Memory`] (the lane then runs its f64 search unchanged).
    fn upload_fd(
        &self,
        problem: &FdSearchProblem,
        lease: &DeviceLease,
    ) -> Result<FdHandle, Refusal> {
        let _ = (problem, lease);
        Err(Refusal::Unsupported("upload_fd"))
    }

    /// M4: search one chunk of one frozen-dual BFS level: every node at the
    /// four dual points (warm, slack-clamped, floor, far) in the backend's
    /// arithmetic, far-first, plus the children's warm start. Writes
    /// `out` (prepared by the backend to the chunk's shape); every value in it
    /// is a HINT — the lane certifies a node only by re-evaluating it in CPU
    /// f64 at a point this call proposed, and splits it otherwise.
    fn fd_level(
        &self,
        handle: &FdHandle,
        chunk: &LevelChunk<'_>,
        out: &mut LevelOut,
    ) -> Result<Untrusted<LevelStats>, Refusal> {
        let _ = (handle, chunk, out);
        Err(Refusal::Unsupported("fd_level"))
    }

    /// M6: take a search copy of one margin-row WAVE tree (the lane IR, the
    /// frozen certified boxes, the slope universe and the objective rows, see
    /// [`WaveProblem`]) for the [`Self::backward_batch`] /
    /// [`Self::branch_scores`] calls that follow. A backend that needs more
    /// than `lease` refuses with [`Refusal::Memory`] (the lane then runs its
    /// legacy tree unchanged).
    fn upload_wave(
        &self,
        problem: &WaveProblem,
        lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        let _ = (problem, lease);
        Err(Refusal::Unsupported("upload_wave"))
    }

    /// M6: output-only beta-CROWN over one WAVE of sub-domains: per domain,
    /// `cfg.iters` projected Adam steps on its slopes (the upload's universe)
    /// and its split multipliers, with the certified boxes FROZEN and the
    /// domain's splits PINNED, keeping the best iterate. Every value written to
    /// `out` is a HINT: the lane re-scores a domain only by its own certified
    /// evaluation (with the hints clamped at ingest) and splits it otherwise.
    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        let _ = (handle, batch, cfg, out);
        Err(Refusal::Unsupported("backward_batch"))
    }

    /// M6: branching for a wave: per domain, up to `k` universe positions the
    /// domain has not split, ranked by a BaBSR-style intercept score from ONE
    /// backward pass at the batch's (hinted) slopes and multipliers. A
    /// candidate is a SCHEDULING hint; the lane validates it against its own
    /// split sites (any valid split of a region is a cover of it).
    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        let _ = (handle, batch, k, out);
        Err(Refusal::Unsupported("branch_scores"))
    }

    /// M8: take a search copy of one lowdim forward-linear problem (the lane's
    /// lowered programs, per-clause constants and objective rows, see
    /// [`FlProblem`]) for the [`Self::forward_linear_batch`] calls that follow.
    /// A backend that needs more than `lease` refuses with
    /// [`Refusal::Memory`] (the lane then runs its legacy bounder unchanged).
    fn upload_forward_linear(
        &self,
        problem: &FlProblem,
        lease: &DeviceLease,
    ) -> Result<FlHandle, Refusal> {
        let _ = (problem, lease);
        Err(Refusal::Unsupported("upload_forward_linear"))
    }

    /// M8: bound one WAVE of input-split domains of one family: per domain,
    /// forward-linear intermediates over its box and one CROWN backward pass
    /// per requested objective row, in the backend's arithmetic. Every value
    /// written to `out` is a HINT: the lane re-bounds a domain only by its own
    /// certified f64 forms (and only when a hint CLAIMS it) and splits it
    /// otherwise.
    fn forward_linear_batch(
        &self,
        handle: &FlHandle,
        batch: &FlBatch<'_>,
        out: &mut FlOut,
    ) -> Result<Untrusted<FlStats>, Refusal> {
        let _ = (handle, batch, out);
        Err(Refusal::Unsupported("forward_linear_batch"))
    }
}

/// One split's two opposite halfspaces in generator coordinates.
#[derive(Debug, Clone, Copy)]
pub struct DualSearchRow<'a> {
    /// Shared generator coefficients of the halfspaces.
    pub a: &'a [f64],
    /// Orientation: exactly `+1` or `-1`.
    pub sign: f64,
    /// Right hand side of the first halfspace.
    pub b0: f64,
    /// Right hand side of the opposite halfspace.
    pub b1: f64,
    /// The opposite halfspace is vacuous; keep its multiplier unchanged.
    pub vacuous: bool,
}

/// Borrowed, untrusted box-support dual search input.
///
/// At a proposed multiplier vector, reduced costs are
/// `rc + sum_i sign_i * ((l0_i - initial_l0_i) -
/// (l1_i - initial_l1_i)) * a_i`. Search maximizes their negative L1 norm
/// minus `sum_i (l0_i * b0_i + l1_i * b1_i)`. The omitted constant cancels
/// from every comparison. All input data remains owned by the certifier.
#[derive(Debug, Clone, Copy)]
pub struct DualSearchRequest<'a> {
    /// Reduced costs at the supplied initial multipliers.
    pub rc: &'a [f64],
    /// Split rows, in multiplier order.
    pub rows: &'a [DualSearchRow<'a>],
    /// Initial nonnegative multipliers for the first halfspaces.
    pub l0: &'a [f64],
    /// Initial nonnegative multipliers for the opposite halfspaces.
    pub l1: &'a [f64],
    /// Maximum number of strictly improving accepted steps.
    pub steps: usize,
    /// Deadline; return the best complete hint if it expires during search.
    pub deadline: Option<Instant>,
    /// Admitted bytes for all search-owned buffers.
    pub mem_cap_bytes: u64,
}

/// A dual search proposal. Only a certified ingest gate may consume it.
#[derive(Debug, Clone, PartialEq)]
pub struct DualMultipliers {
    /// First-halfspace multipliers, in request row order.
    pub l0: Vec<f64>,
    /// Opposite-halfspace multipliers, in request row order.
    pub l1: Vec<f64>,
}

/// Complete dual search hint and verdict-free telemetry.
///
/// All counters are search receipts, never certificate inputs. Backends that
/// do not compile support leave them at their default value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SupportSearchStats {
    /// Requests whose support view was compiled and used.
    pub compiled: u64,
    /// Requests that kept the original representation after bounded admission.
    pub fallback: u64,
    /// Original generator columns in compiled requests.
    pub input_columns: u64,
    /// Representative columns actually searched.
    pub active_columns: u64,
    /// Columns absent from every active row.
    pub untouched_columns: u64,
    /// Additional exactly equal/opposite columns represented by a weight.
    pub grouped_columns: u64,
    /// Nonzero columns folded under a checked full multiplier-box sign guard.
    pub sign_folded_columns: u64,
    /// Row coefficients inspected while compiling the support view.
    pub build_reads: u64,
    /// Row coefficients inspected during proposal search.
    pub search_reads: u64,
    /// Equivalent uncompressed work for completed representative reads. A
    /// grouped read represents its integer weight; folded columns are charged
    /// only after a whole row pass completes. This is a comparison allowance,
    /// not observed physical reads by a simultaneously executed dense solver.
    pub dense_search_reads: u64,
    /// Breakpoint events actually sorted.
    pub events: u64,
    /// Events represented before exact duplicate/opposite grouping.
    pub ungrouped_events: u64,
    /// Conservative peak search-owned allocation, using actual vector capacities.
    pub peak_bytes: u64,
    /// Workspace estimate for the original full-column search.
    pub dense_bytes: u64,
}

impl SupportSearchStats {
    /// Combine independently completed receipts. Byte fields are peaks.
    pub fn merge(&mut self, other: Self) {
        macro_rules! sum {
            ($($field:ident),* $(,)?) => { $(
                self.$field = self.$field.saturating_add(other.$field);
            )* };
        }
        sum!(
            compiled,
            fallback,
            input_columns,
            active_columns,
            untouched_columns,
            grouped_columns,
            sign_folded_columns,
            build_reads,
            search_reads,
            dense_search_reads,
            events,
            ungrouped_events
        );
        self.peak_bytes = self.peak_bytes.max(other.peak_bytes);
        self.dense_bytes = self.dense_bytes.max(other.dense_bytes);
    }
}

/// Complete dual search hint and verdict-free telemetry.
#[derive(Debug)]
pub struct DualSearchOutcome {
    /// Finite nonnegative multipliers, independently recertified by the caller.
    pub multipliers: Untrusted<DualMultipliers>,
    /// Strictly improving accepted steps.
    pub steps: usize,
    /// Improvement of the untrusted objective over the supplied starting point.
    pub gain: Untrusted<f64>,
    /// `steps`, `stationary`, `deadline`, or `nonfinite`.
    pub stop: &'static str,
    /// Verdict-free support compilation and search-work receipt.
    pub support: SupportSearchStats,
}

// ---------------------------------------------------------------------------
//  The lane-IR search copy
// ---------------------------------------------------------------------------

/// Padding sentinel of [`SearchConv::gather`].
pub const GATHER_PAD: u32 = u32::MAX;

/// A convolution in gather form (the margin-row `ConvOp` tables verbatim, with
/// `u32` indices): output `(o, sp)` is
/// `bias[o] + sum_t weight[o * k + t] * input[gather[t * p + sp]]`, padded taps
/// ([`GATHER_PAD`]) skipped. Transposed convolutions compile to the same form.
#[derive(Debug, Clone)]
pub struct SearchConv {
    /// Consumed tensor id.
    pub input: usize,
    /// Output channels.
    pub cout: usize,
    /// Taps per output (`cin * kh * kw`).
    pub k: usize,
    /// Output spatial positions (`oh * ow`).
    pub p: usize,
    /// `[k][p]` flat input indices, [`GATHER_PAD`] = zero padding.
    pub gather: Vec<u32>,
    /// `[cout][k]` weights.
    pub weight: Vec<f64>,
    /// `[cout]` bias.
    pub bias: Vec<f64>,
}

/// One op of a [`SearchNet`]; tensor `k + 1` is the output of op `k`, tensor 0
/// is the input.
#[derive(Debug, Clone)]
pub enum SearchOp {
    /// Convolution (gather form).
    Conv(Box<SearchConv>),
    /// Per-neuron affine `scale * x + shift`.
    Affine {
        /// Consumed tensor id.
        input: usize,
        /// Per-neuron scale.
        scale: Vec<f64>,
        /// Per-neuron shift.
        shift: Vec<f64>,
    },
    /// ReLU. `head` marks the lane's head ReLU (the margin-row `y` layer).
    Relu {
        /// Consumed tensor id.
        input: usize,
        /// The head ReLU (not a trunk layer).
        head: bool,
    },
    /// Elementwise sum.
    Add {
        /// Left tensor id.
        lhs: usize,
        /// Right tensor id.
        rhs: usize,
    },
    /// Flat identity.
    Flatten {
        /// Consumed tensor id.
        input: usize,
    },
    /// Dense layer, row-major `[n_out][n_in]` weights.
    Gemm {
        /// Consumed tensor id.
        input: usize,
        /// Rows.
        n_out: usize,
        /// Columns.
        n_in: usize,
        /// Row-major weights.
        weight: Vec<f64>,
        /// Bias.
        bias: Vec<f64>,
    },
}

/// A lossless plain-data copy of a lane's network, for SEARCH only.
///
/// The certified error budgets of folded layers (BN folds) are deliberately
/// absent: search is untrusted, and every published bound is recomputed by the
/// lane's certified evaluator from the lane's own IR.
#[derive(Debug, Clone)]
pub struct SearchNet {
    /// Flat input size (tensor 0).
    pub n_in: usize,
    /// Ops in execution order.
    pub ops: Vec<SearchOp>,
}

impl SearchNet {
    /// Validate the op list and return every tensor's flat size (index 0 is
    /// the input). Fails on a forward reference, a size mismatch, a gather
    /// index out of range or a non-finite parameter.
    pub fn tensor_sizes(&self) -> Result<Vec<usize>, String> {
        if self.n_in == 0 || self.ops.is_empty() {
            return Err("empty search net".into());
        }
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        let mut sizes = vec![self.n_in];
        for (k, op) in self.ops.iter().enumerate() {
            let size_of = |id: usize| -> Result<usize, String> {
                sizes
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("op {k} references tensor {id} before it exists"))
            };
            let out = match op {
                SearchOp::Conv(c) => {
                    let n_src = size_of(c.input)?;
                    if c.cout == 0
                        || c.k == 0
                        || c.p == 0
                        || c.gather.len() != c.k * c.p
                        || c.weight.len() != c.cout * c.k
                        || c.bias.len() != c.cout
                    {
                        return Err(format!("conv {k}: inconsistent table sizes"));
                    }
                    if c.gather
                        .iter()
                        .any(|&g| g != GATHER_PAD && g as usize >= n_src)
                    {
                        return Err(format!("conv {k}: gather index out of range"));
                    }
                    if !finite(&c.weight) || !finite(&c.bias) {
                        return Err(format!("conv {k}: non-finite parameters"));
                    }
                    c.cout * c.p
                }
                SearchOp::Affine {
                    input,
                    scale,
                    shift,
                } => {
                    let n = size_of(*input)?;
                    if scale.len() != n || shift.len() != n {
                        return Err(format!("affine {k}: size mismatch"));
                    }
                    if !finite(scale) || !finite(shift) {
                        return Err(format!("affine {k}: non-finite parameters"));
                    }
                    n
                }
                SearchOp::Relu { input, .. } | SearchOp::Flatten { input } => size_of(*input)?,
                SearchOp::Add { lhs, rhs } => {
                    let a = size_of(*lhs)?;
                    let b = size_of(*rhs)?;
                    if a != b {
                        return Err(format!("add {k}: size mismatch {a} vs {b}"));
                    }
                    a
                }
                SearchOp::Gemm {
                    input,
                    n_out,
                    n_in,
                    weight,
                    bias,
                } => {
                    let n = size_of(*input)?;
                    if n != *n_in || weight.len() != n_out * n_in || bias.len() != *n_out {
                        return Err(format!("gemm {k}: shape mismatch"));
                    }
                    if !finite(weight) || !finite(bias) {
                        return Err(format!("gemm {k}: non-finite parameters"));
                    }
                    *n_out
                }
            };
            sizes.push(out);
        }
        Ok(sizes)
    }

    /// Plain f64 point evaluation of every tensor (index 0 is `x`), in the
    /// op order and tap order of the gather tables. A REFERENCE for lowering
    /// self-checks (G1) and tests; nothing verdict-bearing reads it.
    pub fn forward_f64(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        let sizes = self.tensor_sizes()?;
        if x.len() != self.n_in {
            return Err(format!(
                "input has {} values, net expects {}",
                x.len(),
                self.n_in
            ));
        }
        let mut vals: Vec<Vec<f64>> = Vec::with_capacity(sizes.len());
        vals.push(x.to_vec());
        for op in &self.ops {
            let out = match op {
                SearchOp::Conv(c) => {
                    let src = &vals[c.input];
                    let mut dst = vec![0.0; c.cout * c.p];
                    for o in 0..c.cout {
                        for sp in 0..c.p {
                            let mut acc = c.bias[o];
                            for t in 0..c.k {
                                let g = c.gather[t * c.p + sp];
                                if g != GATHER_PAD {
                                    acc += c.weight[o * c.k + t] * src[g as usize];
                                }
                            }
                            dst[o * c.p + sp] = acc;
                        }
                    }
                    dst
                }
                SearchOp::Affine {
                    input,
                    scale,
                    shift,
                } => vals[*input]
                    .iter()
                    .zip(scale.iter().zip(shift))
                    .map(|(v, (a, b))| a * v + b)
                    .collect(),
                SearchOp::Relu { input, .. } => vals[*input].iter().map(|v| v.max(0.0)).collect(),
                SearchOp::Add { lhs, rhs } => vals[*lhs]
                    .iter()
                    .zip(&vals[*rhs])
                    .map(|(a, b)| a + b)
                    .collect(),
                SearchOp::Flatten { input } => vals[*input].clone(),
                SearchOp::Gemm {
                    input,
                    n_out,
                    n_in,
                    weight,
                    bias,
                } => {
                    let src = &vals[*input];
                    (0..*n_out)
                        .map(|o| {
                            let mut acc = bias[o];
                            for i in 0..*n_in {
                                acc += weight[o * n_in + i] * src[i];
                            }
                            acc
                        })
                        .collect()
                }
            };
            vals.push(out);
        }
        Ok(vals)
    }

    /// Approximate resident bytes of this copy (admission accounting).
    #[must_use]
    pub fn approx_bytes(&self) -> u64 {
        let f = size_of::<f64>() as u64;
        self.ops
            .iter()
            .map(|op| match op {
                SearchOp::Conv(c) => {
                    (c.gather.len() as u64) * 4 + ((c.weight.len() + c.bias.len()) as u64) * f
                }
                SearchOp::Affine { scale, shift, .. } => ((scale.len() + shift.len()) as u64) * f,
                SearchOp::Gemm { weight, bias, .. } => ((weight.len() + bias.len()) as u64) * f,
                SearchOp::Relu { .. } | SearchOp::Add { .. } | SearchOp::Flatten { .. } => 0,
            })
            .fold(0u64, u64::saturating_add)
    }
}

// ---------------------------------------------------------------------------
//  M5a: the joint-alpha root search request and outcome
// ---------------------------------------------------------------------------

/// A reference box for one ReLU's pre-activation (the lane's CERTIFIED box).
/// The search intersects its own boxes with it; nothing flows back.
#[derive(Debug, Clone)]
pub struct ReluBox {
    /// The ReLU op index in the [`SearchNet`].
    pub op: usize,
    /// Lower sides.
    pub l: Vec<f64>,
    /// Upper sides.
    pub u: Vec<f64>,
}

/// One M5a request: property `y_target - y_j > 0` for every `j` in `adv`, on
/// the LAST tensor of the net, over the box `[lo, hi]`.
#[derive(Debug, Clone)]
pub struct RootSearchRequest {
    /// Input box lower sides.
    pub lo: Vec<f64>,
    /// Input box upper sides.
    pub hi: Vec<f64>,
    /// Certified reference boxes (any subset of the ReLU ops).
    pub ref_boxes: Vec<ReluBox>,
    /// The target class (an index into the last tensor).
    pub target: usize,
    /// The adversarial classes.
    pub adv: Vec<usize>,
    /// Joint iterations after the CROWN initialisation (abc root: 20).
    pub iters: usize,
    /// Adam step on the slopes (abc root: 0.25).
    pub lr: f64,
    /// Wall deadline; the search stops at it and returns its best so far.
    pub deadline: Option<Instant>,
    /// Worker threads (0 = the host's available parallelism).
    pub threads: usize,
    /// Admitted working-set bytes.
    pub mem_cap_bytes: u64,
}

/// The M5a hint: the FINAL start node's lower slopes, one per neuron.
///
/// Neurons the search saw as stable carry their exact limit (1 active, 0
/// inactive). The ingest gate reads only the neurons unstable in the
/// CERTIFIED boxes, clamps to `[0, 1]` and refuses non-finite values.
#[derive(Debug, Clone)]
pub struct RootAlphaHint {
    /// One `(relu op index, slopes)` per NON-head ReLU, in op order.
    pub trunk: Vec<(usize, Vec<f32>)>,
    /// The head ReLU's slopes (empty when the net has no head ReLU).
    pub head: Vec<f32>,
}

/// Telemetry of one M5a search. Every bound here is an UNTRUSTED search
/// number (f32 CROWN, no rounding charged) and is never published.
#[derive(Debug)]
pub struct RootSearchStats {
    /// Joint iterations completed after the CROWN initialisation.
    pub iters: usize,
    /// Start-node passes executed (init included).
    pub passes: usize,
    /// Wall seconds of the search (lowering included).
    pub search_secs: f64,
    /// The CROWN-init root bound (min over every adv class).
    pub crown_lb: Untrusted<f64>,
    /// The search's root bound (min over every adv class) — the S3 metric.
    pub root_lb: Untrusted<f64>,
    /// Adversarial classes.
    pub specs: usize,
    /// Classes the CROWN initialisation already proved (search-side), pruned
    /// from the optimisation (abc `prune_after_crown`).
    pub pruned: usize,
    /// Intermediate start-node rows at the initialisation.
    pub rows: usize,
    /// True when the intermediate slopes were shared per start node (the
    /// per-row alpha did not fit the memory cap).
    pub share_rows: bool,
    /// Worker threads used.
    pub threads: usize,
    /// ReLU pre-activation sides the search tightened against the reference.
    pub tightened: usize,
    /// Reference-unstable neurons the search proved stable.
    pub stabilized: usize,
    /// Why the search stopped: `crown_verified`, `verified`, `patience`,
    /// `iters` or `deadline`.
    pub stop: &'static str,
}

/// What [`SearchBackend::root_joint_alpha`] returns.
#[derive(Debug)]
pub struct RootSearchOutcome {
    /// The slopes, for the ingest gate only.
    pub alpha: Untrusted<RootAlphaHint>,
    /// Telemetry.
    pub stats: RootSearchStats,
}

// ---------------------------------------------------------------------------
//  M4: the frozen-dual level search (`margin_row.frozen_dual_bab_search`)
// ---------------------------------------------------------------------------
//
// The frozen-dual tier (`ny-propagate::margin_row::zono_bab`) walks a
// level-synchronous split tree; each node carries one Lagrange multiplier pair
// per split and is scored at four dual points. The certified score is the f64
// evaluator's; M4 moves the SCORING SEARCH (which point is best, the line
// search that sets the children's warm start) onto a backend and keeps only a
// certification of the proposed point in f64. These are the plain data both
// sides see. Nothing here is trusted: the lane certifies a node by evaluating
// it itself at a proposed point (weak duality makes every non-negative point
// valid) and SPLITS it when that fails.

/// Dual points scored per node.
pub const FD_POINTS: usize = 4;
/// Point index: the warm start inherited from the parent.
pub const FD_WARM: usize = 0;
/// Point index: the warm start with the multipliers of cuts slack at the
/// relaxation's minimiser set to 0.
pub const FD_CLAMPED: usize = 1;
/// Point index: every multiplier 0 (the box-corner floor).
pub const FD_FLOOR: usize = 2;
/// Point index: the clamped far probe along the projected supergradient.
pub const FD_FAR: usize = 3;

/// The frozen-dual search's step tolerance (which multipliers move, whether a
/// node keeps stepping). SEARCH ONLY — it never decides a prune. Equal to the
/// f64 search's `SEARCH_TOL` (a compile-time assertion in ny-propagate's
/// `zono_bab::wide`, plus its test `fd_wide_search_rules_match_the_f64_path`).
pub const FD_SEARCH_TOL: f64 = 1e-9;

/// Log-spaced buckets of the frozen-dual line search (the replica's `Kb`),
/// equal to the f64 search's `LOG_BUCKETS` (same pin).
pub const FD_LOG_BUCKETS: usize = 256;

/// Columns per tile of the frozen-dual level search's column sums.
///
/// The CPU f32 search (ny-bbe) folds each node's `sum_j |rc_j|` and each row's
/// supergradient `sum_j (rc_j < 0 ? a_ij : -a_ij)` as per-tile partial sums
/// (column order inside a tile, then tile order), so this width FIXES the f32
/// summation order. A backend that reproduces the CPU search bit for bit
/// (ny-cuda's device kernels, which are known-answer-tested against it) sums
/// in the same tiles. SEARCH ONLY: no certified value depends on it.
pub const FD_SUM_TILE: usize = 256;

/// The search copy of one frozen-dual problem: the split rows of
/// `FrozenDualProblem` in split order, rounded to f32 (the backend's search
/// arithmetic; the certified problem stays f64 in the lane). Rows are
/// row-major `depth x n`; per-row pairs are `[Off, On]`.
///
/// Rounding the rows to f32 moves the function the search optimises by about
/// one f32 ulp of each term ("store dust"); that costs tightness of the
/// proposals, never soundness, because the lane re-scores every proposal on
/// its own f64 rows.
#[derive(Debug, Clone, Default)]
pub struct FdSearchProblem {
    /// Generator count.
    pub n: usize,
    /// Split rows (the split order's length).
    pub depth: usize,
    /// `depth x n` row coefficients.
    pub a: Vec<f32>,
    /// Each row's own generator column (distinct, `< n`).
    pub enc: Vec<u32>,
    /// Substitution coefficient `rho`, `[Off, On]`.
    pub rho: Vec<[f32; 2]>,
    /// Substituted constant `c0s`, `[Off, On]`.
    pub c0s: Vec<[f32; 2]>,
    /// Sign-cut right-hand side `b0`, `[Off, On]`.
    pub b0: Vec<[f32; 2]>,
    /// Far-cut right-hand side `b1`, `[Off, On]`.
    pub b1: Vec<[f32; 2]>,
    /// Upper bound on each row's `sum_j |a_j|`.
    pub row_abs: Vec<f32>,
    /// Margin generator coefficients (`n`).
    pub d: Vec<f32>,
    /// Margin constant.
    pub c0: f32,
    /// `sum_j |d_j|`.
    pub abs_d: f32,
}

impl FdSearchProblem {
    /// Structural validation: shapes, distinct in-range own columns, every
    /// value finite. A backend refuses an invalid copy ([`Refusal::Invalid`]).
    pub fn validate(&self) -> Result<(), String> {
        let (n, depth) = (self.n, self.depth);
        if n == 0 {
            return Err("fd search copy: no generator".into());
        }
        if self.a.len() != depth.saturating_mul(n)
            || self.enc.len() != depth
            || self.rho.len() != depth
            || self.c0s.len() != depth
            || self.b0.len() != depth
            || self.b1.len() != depth
            || self.row_abs.len() != depth
            || self.d.len() != n
        {
            return Err("fd search copy: inconsistent shapes".into());
        }
        let mut seen = vec![false; n];
        for &e in &self.enc {
            let e = e as usize;
            if e >= n || seen[e] {
                return Err(format!(
                    "fd search copy: own column {e} out of range or repeated"
                ));
            }
            seen[e] = true;
        }
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        let pairs = |v: &[[f32; 2]]| v.iter().all(|p| p[0].is_finite() && p[1].is_finite());
        if !finite(&self.a)
            || !finite(&self.row_abs)
            || !finite(&self.d)
            || !pairs(&self.rho)
            || !pairs(&self.c0s)
            || !pairs(&self.b0)
            || !pairs(&self.b1)
            || !self.c0.is_finite()
            || !self.abs_d.is_finite()
        {
            return Err("fd search copy: non-finite value (f32 overflow?)".into());
        }
        Ok(())
    }

    /// Resident bytes of this copy (admission accounting).
    #[must_use]
    pub fn approx_bytes(&self) -> u64 {
        let f = size_of::<f32>() as u64;
        let rows = (self.a.len() as u64)
            .saturating_add(self.d.len() as u64)
            .saturating_add(self.row_abs.len() as u64)
            .saturating_add(self.enc.len() as u64);
        let pairs = (self.depth as u64).saturating_mul(4 * 2);
        rows.saturating_add(pairs).saturating_mul(f)
    }
}

/// The memory a backend may hold for one upload plus its per-call working set
/// (host bytes for a CPU backend; device bytes for a discrete device; debited
/// from host headroom on a unified one). A backend that needs more refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLease {
    /// Admitted bytes.
    pub bytes: u64,
}

/// A backend's uploaded frozen-dual problem. Opaque: only the backend that
/// made it can read its payload, and any other backend refuses it.
pub struct FdHandle {
    backend: &'static str,
    payload: Box<dyn std::any::Any + Send + Sync>,
}

impl FdHandle {
    /// Wrap a backend's upload.
    pub fn new<T: std::any::Any + Send + Sync>(backend: &'static str, payload: T) -> Self {
        Self {
            backend,
            payload: Box::new(payload),
        }
    }

    /// The backend that made this handle.
    #[must_use]
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// The payload, if it is a `T`.
    #[must_use]
    pub fn payload<T: std::any::Any>(&self) -> Option<&T> {
        self.payload.downcast_ref::<T>()
    }
}

impl fmt::Debug for FdHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FdHandle")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

/// One chunk of one BFS level: `count` nodes of `depth >= 1` splits,
/// row-major (`count x depth`). `sides` are `0` = Off, `1` = On; `l0` / `l1`
/// are the nodes' warm-start multipliers (the search state the lane stores).
#[derive(Debug, Clone, Copy)]
pub struct LevelChunk<'a> {
    /// Splits per node (>= 1).
    pub depth: usize,
    /// Nodes.
    pub count: usize,
    /// `count x depth` sides.
    pub sides: &'a [u8],
    /// `count x depth` sign-cut multipliers.
    pub l0: &'a [f32],
    /// `count x depth` far-cut multipliers.
    pub l1: &'a [f32],
    /// Stop (refusing with [`Refusal::Deadline`]) once this passes.
    pub deadline: Option<Instant>,
}

impl LevelChunk<'_> {
    /// Shape validation (a backend refuses a malformed chunk).
    pub fn validate(&self) -> Result<(), String> {
        let want = self.count.checked_mul(self.depth);
        if self.depth == 0 {
            return Err("fd level chunk: depth 0 (the root is the lane's own)".into());
        }
        if want != Some(self.sides.len())
            || want != Some(self.l0.len())
            || want != Some(self.l1.len())
        {
            return Err("fd level chunk: inconsistent shapes".into());
        }
        if self.sides.iter().any(|&s| s > 1) {
            return Err("fd level chunk: a side is neither 0 nor 1".into());
        }
        Ok(())
    }
}

/// The per-node outputs of [`SearchBackend::fd_level`]: reusable buffers the
/// BACKEND fills (through [`LevelOut::writer`]) and the lane reads only
/// through [`LevelOut::hint`] (an ingest gate: every value is clamped and
/// every proposal re-scored in f64 before anything is kept).
///
/// Per node `k`: the search's value and absolute-term sum `S` at each of the
/// [`FD_POINTS`] points (NaN = not evaluated or refused); the clamped and far
/// multipliers (`[l0 (depth) | l1 (depth)]`); and the children's warm start
/// (same layout).
#[derive(Debug, Clone, Default)]
pub struct LevelOut {
    count: usize,
    depth: usize,
    value: Vec<f32>,
    scale: Vec<f32>,
    clamped: Vec<f32>,
    far: Vec<f32>,
    next: Vec<f32>,
}

/// Mutable view of a [`LevelOut`] for the backend that fills it.
#[derive(Debug)]
pub struct LevelOutWriter<'a> {
    /// Nodes.
    pub count: usize,
    /// Splits per node.
    pub depth: usize,
    /// `count x FD_POINTS` values.
    pub value: &'a mut [f32],
    /// `count x FD_POINTS` absolute-term sums.
    pub scale: &'a mut [f32],
    /// `count x 2 depth` clamped multipliers.
    pub clamped: &'a mut [f32],
    /// `count x 2 depth` far multipliers.
    pub far: &'a mut [f32],
    /// `count x 2 depth` children's warm start.
    pub next: &'a mut [f32],
}

/// Read-only view of a [`LevelOut`], reached only through
/// [`Hint::into_ingest`]. Every accessor returns search values: the caller
/// must sanitize multipliers (finite, `>= 0`) and certify before keeping
/// anything.
#[derive(Debug, Clone, Copy)]
pub struct LevelView<'a> {
    count: usize,
    depth: usize,
    value: &'a [f32],
    scale: &'a [f32],
    clamped: &'a [f32],
    far: &'a [f32],
    next: &'a [f32],
}

impl LevelOut {
    /// Resize to `count` nodes of `depth` splits: values and sums NaN,
    /// multipliers 0.
    pub fn prepare(&mut self, count: usize, depth: usize) {
        let points = count.saturating_mul(FD_POINTS);
        let mults = count.saturating_mul(depth).saturating_mul(2);
        self.count = count;
        self.depth = depth;
        for (buf, len, fill) in [
            (&mut self.value, points, f32::NAN),
            (&mut self.scale, points, f32::NAN),
            (&mut self.clamped, mults, 0.0),
            (&mut self.far, mults, 0.0),
            (&mut self.next, mults, 0.0),
        ] {
            buf.clear();
            buf.resize(len, fill);
        }
    }

    /// The backend's mutable view.
    pub fn writer(&mut self) -> LevelOutWriter<'_> {
        LevelOutWriter {
            count: self.count,
            depth: self.depth,
            value: &mut self.value,
            scale: &mut self.scale,
            clamped: &mut self.clamped,
            far: &mut self.far,
            next: &mut self.next,
        }
    }

    /// The lane's view, as a hint for its ingest gate.
    pub fn hint(&self) -> Hint<LevelView<'_>> {
        Hint(LevelView {
            count: self.count,
            depth: self.depth,
            value: &self.value,
            scale: &self.scale,
            clamped: &self.clamped,
            far: &self.far,
            next: &self.next,
        })
    }

    /// Bytes [`Self::prepare`] allocates for `count` nodes of `depth`.
    #[must_use]
    pub fn bytes_for(count: usize, depth: usize) -> u64 {
        let per_node = (2 * FD_POINTS as u64).saturating_add((depth as u64).saturating_mul(6));
        (count as u64)
            .saturating_mul(per_node)
            .saturating_mul(size_of::<f32>() as u64)
    }
}

impl<'a> LevelView<'a> {
    /// Nodes.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Splits per node.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// True iff every buffer has the shape `count x depth` promises (a
    /// backend could have written a `LevelOut` of another shape).
    #[must_use]
    pub fn well_formed(&self) -> bool {
        let points = self.count.checked_mul(FD_POINTS);
        let mults = self
            .count
            .checked_mul(self.depth)
            .and_then(|v| v.checked_mul(2));
        points == Some(self.value.len())
            && points == Some(self.scale.len())
            && mults == Some(self.clamped.len())
            && mults == Some(self.far.len())
            && mults == Some(self.next.len())
    }

    /// The search's value at point `p` of node `k` (NaN = not evaluated).
    #[must_use]
    pub fn value(&self, k: usize, p: usize) -> f32 {
        self.value[k * FD_POINTS + p]
    }

    /// The search's absolute-term sum `S` at point `p` of node `k`.
    #[must_use]
    pub fn scale(&self, k: usize, p: usize) -> f32 {
        self.scale[k * FD_POINTS + p]
    }

    fn pair(&self, buf: &'a [f32], k: usize) -> (&'a [f32], &'a [f32]) {
        let d = self.depth;
        let at = k * 2 * d;
        (&buf[at..at + d], &buf[at + d..at + 2 * d])
    }

    /// Node `k`'s clamped multipliers `(l0, l1)`.
    #[must_use]
    pub fn clamped(&self, k: usize) -> (&'a [f32], &'a [f32]) {
        self.pair(self.clamped, k)
    }

    /// Node `k`'s far multipliers `(l0, l1)`.
    #[must_use]
    pub fn far(&self, k: usize) -> (&'a [f32], &'a [f32]) {
        self.pair(self.far, k)
    }

    /// Node `k`'s children's warm start `(l0, l1)`.
    #[must_use]
    pub fn next(&self, k: usize) -> (&'a [f32], &'a [f32]) {
        self.pair(self.next, k)
    }
}

/// Telemetry of one [`SearchBackend::fd_level`] call (untrusted).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LevelStats {
    /// Nodes searched.
    pub nodes: u64,
    /// Dual-point evaluations (warm included).
    pub evals: u64,
    /// Nodes the search refused (non-finite value).
    pub refused: u64,
    /// Wall seconds.
    pub secs: f64,
    /// Worker threads used.
    pub threads: usize,
}

// ---------------------------------------------------------------------------
//  M6: the margin-row wave beta-BaB search (`margin_row.tree_mode: wave`)
// ---------------------------------------------------------------------------
//
// The margin-row tree's WAVE engine (`ny-propagate::margin_row::bab::wave`)
// pops hundreds of sub-domains per step. A backend bounds every one of them
// with output-only beta-CROWN (the certified root boxes FROZEN, the domain's
// splits PINNED, Adam on the domain's slopes and split multipliers) and ranks
// its split candidates; the lane CLAIMS a domain whose search bound clears its
// threshold and certifies it on a private CPU pool with its own unchanged
// Outward evaluator (the hints clamped at ingest), splitting every domain it
// does not claim and every claim it cannot certify. THE DEVICE NEVER CERTIFIES:
// every value below is a hint.

/// "No candidate" sentinel of [`BranchOut`].
pub const WAVE_NONE: u32 = u32::MAX;

// The wave search's arithmetic constants, ONE copy for every backend (ny-bbe's
// CPU search, the reference, and ny-cuda's device search, which reproduces it
// bit for bit): a retune here moves both, so they cannot drift apart.

/// Adam's first-moment decay of the wave ascent (abc's BaB).
pub const WAVE_ADAM_B1: f64 = 0.9;
/// Adam's second-moment decay of the wave ascent.
pub const WAVE_ADAM_B2: f64 = 0.999;
/// Adam's denominator floor of the wave ascent.
pub const WAVE_ADAM_EPS: f64 = 1e-8;
/// Per-iteration learning-rate decay of the wave ascent (abc's BaB
/// `lr_decay`): step `it` (0-based) uses `lr * WAVE_LR_DECAY^it`.
pub const WAVE_LR_DECAY: f64 = 0.98;
/// The slope learning rate a non-finite or non-positive configured one falls
/// back to (and [`WaveSearchCfg::default`]'s).
pub const WAVE_DEFAULT_LR_ALPHA: f64 = 0.1;
/// The multiplier learning rate a non-finite or non-positive configured one
/// falls back to (and [`WaveSearchCfg::default`]'s).
pub const WAVE_DEFAULT_LR_BETA: f64 = 0.05;
/// Most candidates [`SearchBackend::branch_scores`] may return per domain.
pub const WAVE_MAX_BRANCH_K: usize = 16;

// The wave search's INGEST rules and Adam schedule, ONE copy for the CPU
// reference (ny-bbe `wave.rs`) and the device search's host side (ny-cuda
// `wave_plan.rs`, which computes the kernels' per-iteration constants on the
// host and mirrors the ingest in its self-test oracle): a change here moves
// both, and ny-cli's bit-identity KATs pin that they still agree.

/// The configured learning rates after ingest: a finite, positive rate is
/// kept; anything else takes [`WAVE_DEFAULT_LR_ALPHA`] /
/// [`WAVE_DEFAULT_LR_BETA`].
#[must_use]
pub fn wave_ingest_rates(lr_alpha: f64, lr_beta: f64) -> (f64, f64) {
    (
        if lr_alpha.is_finite() && lr_alpha > 0.0 {
            lr_alpha
        } else {
            WAVE_DEFAULT_LR_ALPHA
        },
        if lr_beta.is_finite() && lr_beta > 0.0 {
            lr_beta
        } else {
            WAVE_DEFAULT_LR_BETA
        },
    )
}

/// Adam's bias corrections at step `t` (counted from 1):
/// `(1 - WAVE_ADAM_B1^t, 1 - WAVE_ADAM_B2^t)`.
///
/// NEVER inlined, like [`wave_lr_decay`]: `powi` of a CONSTANT exponent may be
/// constant-folded by the compiler with another algorithm than the runtime's
/// repeated squaring (measured: `1 - 0.9^4` folds to `0.3439`, the runtime
/// gives `0.3438999999999999`), so an inlined call with a literal step would
/// disagree with the CPU search's and the device's runtime calls by an ulp.
/// One out-of-line copy gives every caller the runtime bits.
#[must_use]
#[inline(never)]
pub fn wave_adam_bias(t: i32) -> (f64, f64) {
    (1.0 - WAVE_ADAM_B1.powi(t), 1.0 - WAVE_ADAM_B2.powi(t))
}

/// The learning-rate factor of iteration `it` (0-based): `WAVE_LR_DECAY^it`
/// (never inlined: see [`wave_adam_bias`]).
#[must_use]
#[inline(never)]
pub fn wave_lr_decay(it: usize) -> f64 {
    WAVE_LR_DECAY.powi(i32::try_from(it).unwrap_or(i32::MAX))
}

/// The ingest of a warm slope hint: a finite value clamped into `[0, 1]`
/// (in f64), else `None` (the domain keeps the upload's `alpha0`).
#[must_use]
pub fn wave_warm_alpha(a: f32) -> Option<f64> {
    a.is_finite().then(|| f64::from(a).clamp(0.0, 1.0))
}

/// The ingest of a warm multiplier hint: finite and positive, else 0.
#[must_use]
pub fn wave_warm_beta(b: f32) -> f64 {
    if b.is_finite() && b > 0.0 {
        f64::from(b)
    } else {
        0.0
    }
}

/// One ReLU neuron of a [`WaveProblem`]'s net: the ReLU's OP index in the
/// [`SearchNet`] and the neuron's index in that op's input tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveSite {
    /// The ReLU op index.
    pub op: u32,
    /// The neuron.
    pub neuron: u32,
}

/// One split literal of a domain: a position in [`WaveProblem::universe`] and
/// the closed half-space of that neuron's pre-activation the domain lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveSplit {
    /// Universe position.
    pub pos: u32,
    /// `true`: `z >= 0` (active); `false`: `z <= 0` (inactive).
    pub active: bool,
}

/// The search copy of one margin-row wave tree: the lane IR, the lane's
/// CERTIFIED pre-activation box of every ReLU (frozen for the whole tree), the
/// slope / split UNIVERSE and the objective rows `y_target - y_classes[r]` on
/// the net's last tensor.
///
/// The universe is the lane's own list of splittable neurons (every one
/// unstable in its certified box); slopes and splits travel as universe
/// positions, so the lane maps every hint back to its own split sites and
/// certified gates without reading any backend structure (R3).
#[derive(Debug, Clone)]
pub struct WaveProblem {
    /// The lane IR copy.
    pub net: SearchNet,
    /// One certified box per ReLU op of `net` (keyed by op, any order).
    pub boxes: Vec<ReluBox>,
    /// The splittable neurons, in the lane's order.
    pub universe: Vec<WaveSite>,
    /// Warm-start lower slope per universe position. The backend clamps it
    /// into `[0, 1]`; a non-finite value takes CROWN's adaptive slope.
    pub alpha0: Vec<f64>,
    /// Input box lower sides.
    pub lo: Vec<f64>,
    /// Input box upper sides.
    pub hi: Vec<f64>,
    /// The target class (an index into the last tensor).
    pub target: usize,
    /// The objective classes: row `r` is `y_target - y_classes[r]`.
    pub classes: Vec<usize>,
}

impl WaveProblem {
    /// Structural validation; returns every tensor's flat size. Refuses a net
    /// that does not validate, an inconsistent or NaN box, a ReLU without
    /// exactly one box, a universe neuron that is out of range, repeated, or
    /// not unstable in its box, and an objective row out of range or equal to
    /// the target.
    pub fn validate(&self) -> Result<Vec<usize>, String> {
        let sizes = self.net.tensor_sizes()?;
        let n_in = self.net.n_in;
        if self.lo.len() != n_in || self.hi.len() != n_in {
            return Err("wave problem: input box width mismatch".into());
        }
        if self
            .lo
            .iter()
            .zip(&self.hi)
            .any(|(l, h)| !(l.is_finite() && h.is_finite() && l <= h))
        {
            return Err("wave problem: input box not finite and ordered".into());
        }
        let out = *sizes.last().ok_or("wave problem: no tensor")?;
        if self.target >= out {
            return Err("wave problem: target out of range".into());
        }
        if self.classes.is_empty() || self.classes.iter().any(|&c| c >= out || c == self.target) {
            return Err("wave problem: objective class out of range or equal to the target".into());
        }
        let mut relu_width: Vec<Option<usize>> = vec![None; self.net.ops.len()];
        for (k, op) in self.net.ops.iter().enumerate() {
            if let SearchOp::Relu { input, .. } = op {
                relu_width[k] = Some(sizes[*input]);
            }
        }
        let mut boxed = vec![false; self.net.ops.len()];
        for b in &self.boxes {
            let Some(Some(width)) = relu_width.get(b.op).copied() else {
                return Err(format!(
                    "wave problem: box for op {} which is not a ReLU",
                    b.op
                ));
            };
            if boxed[b.op] {
                return Err(format!("wave problem: two boxes for ReLU op {}", b.op));
            }
            boxed[b.op] = true;
            if b.l.len() != width || b.u.len() != width {
                return Err(format!("wave problem: box width mismatch at op {}", b.op));
            }
            if b.l
                .iter()
                .zip(&b.u)
                .any(|(l, u)| l.is_nan() || u.is_nan() || l > u)
            {
                return Err(format!("wave problem: NaN or crossed box at op {}", b.op));
            }
        }
        if relu_width
            .iter()
            .zip(&boxed)
            .any(|(w, done)| w.is_some() && !done)
        {
            return Err("wave problem: a ReLU op has no certified box".into());
        }
        if self.alpha0.len() != self.universe.len() {
            return Err("wave problem: alpha0 is not one slope per universe position".into());
        }
        if !u32::try_from(self.universe.len()).is_ok_and(|n| n != WAVE_NONE) {
            return Err("wave problem: universe too large for u32 positions".into());
        }
        let mut seen = std::collections::HashSet::with_capacity(self.universe.len());
        for site in &self.universe {
            let op = site.op as usize;
            let Some(Some(width)) = relu_width.get(op).copied() else {
                return Err(format!("wave problem: universe site op {op} is not a ReLU"));
            };
            let j = site.neuron as usize;
            if j >= width || !seen.insert(*site) {
                return Err(format!(
                    "wave problem: universe neuron {j} of op {op} out of range or repeated"
                ));
            }
            let b = self
                .boxes
                .iter()
                .find(|b| b.op == op)
                .ok_or("wave problem: universe op without a box")?;
            if !(b.l[j] < 0.0 && b.u[j] > 0.0) {
                return Err(format!(
                    "wave problem: universe neuron {j} of op {op} is stable in its box"
                ));
            }
        }
        Ok(sizes)
    }

    /// Approximate resident bytes of this copy (admission accounting).
    #[must_use]
    pub fn approx_bytes(&self) -> u64 {
        let f = size_of::<f64>() as u64;
        let boxes = self
            .boxes
            .iter()
            .map(|b| ((b.l.len() + b.u.len()) as u64).saturating_mul(f))
            .fold(0u64, u64::saturating_add);
        self.net
            .approx_bytes()
            .saturating_add(boxes)
            .saturating_add((self.universe.len() as u64).saturating_mul(8 + f))
            .saturating_add(((self.lo.len() + self.hi.len()) as u64).saturating_mul(f))
    }
}

/// A backend's uploaded wave problem. Opaque: only the backend that made it
/// can read its payload, and any other backend refuses it.
pub struct WaveHandle {
    backend: &'static str,
    stride: usize,
    payload: Box<dyn std::any::Any + Send + Sync>,
}

impl WaveHandle {
    /// Wrap a backend's upload of a problem with `stride` universe positions.
    pub fn new<T: std::any::Any + Send + Sync>(
        backend: &'static str,
        stride: usize,
        payload: T,
    ) -> Self {
        Self {
            backend,
            stride,
            payload: Box::new(payload),
        }
    }

    /// The backend that made this handle.
    #[must_use]
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Universe positions of the uploaded problem (the slope stride).
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The payload, if it is a `T`.
    #[must_use]
    pub fn payload<T: std::any::Any>(&self) -> Option<&T> {
        self.payload.downcast_ref::<T>()
    }
}

impl fmt::Debug for WaveHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WaveHandle")
            .field("backend", &self.backend)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

/// One wave of sub-domains. Domain `k` optimises objective row `rows[k]`, owns
/// the literals `splits[offsets[k]..offsets[k + 1]]` with their warm
/// multipliers at the same positions of `beta`, and starts from the slopes
/// `alpha[k * stride..(k + 1) * stride]` (an empty `alpha` starts every domain
/// at the upload's `alpha0`).
#[derive(Debug, Clone, Copy)]
pub struct WaveBatch<'a> {
    /// Domains.
    pub count: usize,
    /// Per domain, its objective row (an index into [`WaveProblem::classes`]).
    pub rows: &'a [u32],
    /// `count + 1` non-decreasing offsets into `splits` / `beta`, from 0.
    pub offsets: &'a [u32],
    /// Every domain's literals, concatenated.
    pub splits: &'a [WaveSplit],
    /// Warm split multipliers, aligned with `splits`.
    pub beta: &'a [f32],
    /// Warm slopes, `count x stride`, or empty.
    pub alpha: &'a [f32],
    /// Universe positions per slope row (must equal the handle's stride).
    pub stride: usize,
    /// Stop (refusing with [`Refusal::Deadline`]) once this passes.
    pub deadline: Option<Instant>,
}

impl WaveBatch<'_> {
    /// Shape validation against an upload of `stride` universe positions and
    /// `classes` objective rows. A backend refuses a malformed batch.
    pub fn validate(&self, stride: usize, classes: usize) -> Result<(), String> {
        if self.stride != stride {
            return Err("wave batch: slope stride differs from the upload".into());
        }
        if self.rows.len() != self.count || self.offsets.len() != self.count + 1 {
            return Err("wave batch: rows / offsets do not match the count".into());
        }
        if self.offsets.first() != Some(&0)
            || self.offsets.windows(2).any(|w| w[0] > w[1])
            || self.offsets.last().map(|&o| o as usize) != Some(self.splits.len())
        {
            return Err("wave batch: offsets are not a partition of the splits".into());
        }
        if self.beta.len() != self.splits.len() {
            return Err("wave batch: beta is not one multiplier per split".into());
        }
        if !self.alpha.is_empty() && Some(self.alpha.len()) != self.count.checked_mul(stride) {
            return Err("wave batch: alpha is neither empty nor count x stride".into());
        }
        if self.rows.iter().any(|&r| r as usize >= classes) {
            return Err("wave batch: objective row out of range".into());
        }
        let mut seen: Vec<u32> = Vec::new();
        for w in self.offsets.windows(2) {
            let lits = &self.splits[w[0] as usize..w[1] as usize];
            seen.clear();
            seen.extend(lits.iter().map(|s| s.pos));
            seen.sort_unstable();
            if seen.windows(2).any(|p| p[0] == p[1]) {
                return Err("wave batch: a domain splits one neuron twice".into());
            }
            if seen.last().is_some_and(|&p| p as usize >= stride) {
                return Err("wave batch: split position outside the universe".into());
            }
        }
        Ok(())
    }
}

/// Controls of one [`SearchBackend::backward_batch`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveSearchCfg {
    /// Adam steps per domain (`iters + 1` bound evaluations; 0 scores the
    /// warm start only).
    pub iters: usize,
    /// Adam step on the slopes.
    pub lr_alpha: f64,
    /// Adam step on the split multipliers.
    pub lr_beta: f64,
    /// A domain leaves the optimisation as soon as its search bound exceeds
    /// this (abc's early stop of verified rows).
    pub stop_above: f64,
    /// Worker threads (0 = the backend's own default).
    pub threads: usize,
}

impl Default for WaveSearchCfg {
    fn default() -> Self {
        Self {
            iters: 8,
            lr_alpha: WAVE_DEFAULT_LR_ALPHA,
            lr_beta: WAVE_DEFAULT_LR_BETA,
            stop_above: 0.0,
            threads: 0,
        }
    }
}

/// The per-domain outputs of [`SearchBackend::backward_batch`]: buffers the
/// BACKEND fills (through [`WaveOut::writer`]) and the lane reads only
/// through [`WaveOut::hint`] (its ingest gate: slopes clamped into `[0, 1]`,
/// multipliers kept only when finite and positive, and every claim certified
/// before anything closes).
///
/// Per domain `k`: the best search bound and its absolute-term sum `S`
/// (NaN = none), the bound evaluations spent, the multipliers at the best
/// iterate (aligned with the batch's `splits`) and its slopes (`stride`
/// values; NaN = keep the lane's own).
#[derive(Debug, Clone, Default)]
pub struct WaveOut {
    count: usize,
    stride: usize,
    splits: usize,
    lb: Vec<f32>,
    scale: Vec<f32>,
    evals: Vec<u32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
}

/// Mutable view of a [`WaveOut`] for the backend that fills it.
#[derive(Debug)]
pub struct WaveOutWriter<'a> {
    /// Domains.
    pub count: usize,
    /// Slopes per domain.
    pub stride: usize,
    /// Best search bound per domain.
    pub lb: &'a mut [f32],
    /// Its absolute-term sum per domain.
    pub scale: &'a mut [f32],
    /// Bound evaluations per domain.
    pub evals: &'a mut [u32],
    /// Multipliers, aligned with the batch's `splits`.
    pub beta: &'a mut [f32],
    /// `count x stride` slopes.
    pub alpha: &'a mut [f32],
}

/// Read-only view of a [`WaveOut`], reached only through
/// [`Hint::into_ingest`]. Every accessor returns search values.
#[derive(Debug, Clone, Copy)]
pub struct WaveView<'a> {
    count: usize,
    stride: usize,
    splits: usize,
    lb: &'a [f32],
    scale: &'a [f32],
    evals: &'a [u32],
    beta: &'a [f32],
    alpha: &'a [f32],
}

impl WaveOut {
    /// Resize to `count` domains of `stride` slopes and `splits` literals in
    /// total: bounds and sums NaN, evaluations 0, multipliers 0, slopes NaN.
    pub fn prepare(&mut self, count: usize, stride: usize, splits: usize) {
        self.count = count;
        self.stride = stride;
        self.splits = splits;
        for (buf, len, fill) in [
            (&mut self.lb, count, f32::NAN),
            (&mut self.scale, count, f32::NAN),
            (&mut self.beta, splits, 0.0),
            (&mut self.alpha, count.saturating_mul(stride), f32::NAN),
        ] {
            buf.clear();
            buf.resize(len, fill);
        }
        self.evals.clear();
        self.evals.resize(count, 0);
    }

    /// The backend's mutable view.
    pub fn writer(&mut self) -> WaveOutWriter<'_> {
        WaveOutWriter {
            count: self.count,
            stride: self.stride,
            lb: &mut self.lb,
            scale: &mut self.scale,
            evals: &mut self.evals,
            beta: &mut self.beta,
            alpha: &mut self.alpha,
        }
    }

    /// The lane's view, as a hint for its ingest gate.
    pub fn hint(&self) -> Hint<WaveView<'_>> {
        Hint(WaveView {
            count: self.count,
            stride: self.stride,
            splits: self.splits,
            lb: &self.lb,
            scale: &self.scale,
            evals: &self.evals,
            beta: &self.beta,
            alpha: &self.alpha,
        })
    }

    /// Bytes [`Self::prepare`] allocates.
    #[must_use]
    pub fn bytes_for(count: usize, stride: usize, splits: usize) -> u64 {
        let per_domain = 12u64.saturating_add((stride as u64).saturating_mul(4));
        (count as u64)
            .saturating_mul(per_domain)
            .saturating_add((splits as u64).saturating_mul(4))
    }
}

impl<'a> WaveView<'a> {
    /// Domains.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Slopes per domain.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// True iff every buffer has the shape `count`, `stride` and `splits`
    /// promise (a backend could have written a `WaveOut` of another shape).
    #[must_use]
    pub fn well_formed(&self) -> bool {
        self.lb.len() == self.count
            && self.scale.len() == self.count
            && self.evals.len() == self.count
            && self.beta.len() == self.splits
            && Some(self.alpha.len()) == self.count.checked_mul(self.stride)
    }

    /// Domain `k`'s best search bound (NaN = none).
    #[must_use]
    pub fn lb(&self, k: usize) -> f32 {
        self.lb[k]
    }

    /// Domain `k`'s absolute-term sum at that bound.
    #[must_use]
    pub fn scale(&self, k: usize) -> f32 {
        self.scale[k]
    }

    /// Domain `k`'s bound evaluations.
    #[must_use]
    pub fn evals(&self, k: usize) -> u32 {
        self.evals[k]
    }

    /// Every domain's multipliers, aligned with the batch's `splits`.
    #[must_use]
    pub fn beta(&self) -> &'a [f32] {
        self.beta
    }

    /// Domain `k`'s slopes (`stride` values).
    #[must_use]
    pub fn alpha(&self, k: usize) -> &'a [f32] {
        &self.alpha[k * self.stride..(k + 1) * self.stride]
    }
}

/// Telemetry of one [`SearchBackend::backward_batch`] call (untrusted).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WaveStats {
    /// Domains searched.
    pub domains: u64,
    /// Bound evaluations (backward passes) over all domains.
    pub passes: u64,
    /// Domains that left the optimisation early (bound above `stop_above`).
    pub early_stopped: u64,
    /// Wall seconds.
    pub secs: f64,
    /// Worker threads used.
    pub threads: usize,
}

/// The per-domain outputs of [`SearchBackend::branch_scores`]: up to `k`
/// universe positions per domain, best first ([`WAVE_NONE`] pads), with their
/// scores (NaN pads). Filled through [`BranchOut::writer`], read through
/// [`BranchOut::hint`].
#[derive(Debug, Clone, Default)]
pub struct BranchOut {
    count: usize,
    k: usize,
    pos: Vec<u32>,
    score: Vec<f32>,
}

/// Mutable view of a [`BranchOut`] for the backend that fills it.
#[derive(Debug)]
pub struct BranchOutWriter<'a> {
    /// Domains.
    pub count: usize,
    /// Candidates per domain.
    pub k: usize,
    /// `count x k` universe positions.
    pub pos: &'a mut [u32],
    /// `count x k` scores.
    pub score: &'a mut [f32],
}

/// Read-only view of a [`BranchOut`], reached only through
/// [`Hint::into_ingest`].
#[derive(Debug, Clone, Copy)]
pub struct BranchView<'a> {
    count: usize,
    k: usize,
    pos: &'a [u32],
    score: &'a [f32],
}

impl BranchOut {
    /// Resize to `count` domains of `k` candidates, all padding.
    pub fn prepare(&mut self, count: usize, k: usize) {
        self.count = count;
        self.k = k;
        let len = count.saturating_mul(k);
        self.pos.clear();
        self.pos.resize(len, WAVE_NONE);
        self.score.clear();
        self.score.resize(len, f32::NAN);
    }

    /// The backend's mutable view.
    pub fn writer(&mut self) -> BranchOutWriter<'_> {
        BranchOutWriter {
            count: self.count,
            k: self.k,
            pos: &mut self.pos,
            score: &mut self.score,
        }
    }

    /// The lane's view, as a hint.
    pub fn hint(&self) -> Hint<BranchView<'_>> {
        Hint(BranchView {
            count: self.count,
            k: self.k,
            pos: &self.pos,
            score: &self.score,
        })
    }
}

impl<'a> BranchView<'a> {
    /// Domains.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Candidates per domain.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// True iff both buffers are `count x k`.
    #[must_use]
    pub fn well_formed(&self) -> bool {
        let len = self.count.checked_mul(self.k);
        len == Some(self.pos.len()) && len == Some(self.score.len())
    }

    /// Domain `d`'s candidate positions, best first.
    #[must_use]
    pub fn pos(&self, d: usize) -> &'a [u32] {
        &self.pos[d * self.k..(d + 1) * self.k]
    }

    /// Domain `d`'s candidate scores.
    #[must_use]
    pub fn score(&self, d: usize) -> &'a [f32] {
        &self.score[d * self.k..(d + 1) * self.k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> SearchNet {
        SearchNet {
            n_in: 2,
            ops: vec![
                SearchOp::Gemm {
                    input: 0,
                    n_out: 2,
                    n_in: 2,
                    weight: vec![1.0, -1.0, 0.5, 2.0],
                    bias: vec![0.25, -0.5],
                },
                SearchOp::Relu {
                    input: 1,
                    head: false,
                },
                SearchOp::Gemm {
                    input: 2,
                    n_out: 2,
                    n_in: 2,
                    weight: vec![1.0, 1.0, -1.0, 3.0],
                    bias: vec![0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn search_net_sizes_and_forward_are_exact_on_a_two_layer_net() {
        let net = tiny();
        assert_eq!(net.tensor_sizes().expect("valid"), vec![2, 2, 2, 2]);
        let vals = net.forward_f64(&[1.0, 0.5]).expect("forward");
        // z = [1 - 0.5 + 0.25, 0.5 + 1 - 0.5] = [0.75, 1.0]
        assert_eq!(vals[1], vec![0.75, 1.0]);
        assert_eq!(vals[2], vec![0.75, 1.0]);
        assert_eq!(vals[3], vec![1.75, -0.75 + 3.0]);
    }

    #[test]
    fn search_net_validation_refuses_forward_references_and_bad_gathers() {
        let mut net = tiny();
        net.ops[0] = SearchOp::Relu {
            input: 3,
            head: false,
        };
        assert!(net.tensor_sizes().is_err());
        let conv = SearchNet {
            n_in: 4,
            ops: vec![SearchOp::Conv(Box::new(SearchConv {
                input: 0,
                cout: 1,
                k: 1,
                p: 2,
                gather: vec![0, 9],
                weight: vec![1.0],
                bias: vec![0.0],
            }))],
        };
        assert!(
            conv.tensor_sizes().is_err(),
            "gather index 9 >= 4 must refuse"
        );
    }

    fn fd_copy(n: usize, depth: usize) -> FdSearchProblem {
        FdSearchProblem {
            n,
            depth,
            a: vec![0.5; depth * n],
            enc: (0..depth as u32).collect(),
            rho: vec![[1.0, -1.0]; depth],
            c0s: vec![[-0.1, -0.2]; depth],
            b0: vec![[0.3, 0.4]; depth],
            b1: vec![[0.5, 0.6]; depth],
            row_abs: vec![0.5 * n as f32; depth],
            d: vec![0.25; n],
            c0: 1.0,
            abs_d: 0.25 * n as f32,
        }
    }

    #[test]
    fn fd_search_copy_validation_refuses_bad_shapes_columns_and_overflow() {
        assert!(fd_copy(5, 3).validate().is_ok());
        let mut short = fd_copy(5, 3);
        short.a.pop();
        assert!(short.validate().is_err(), "a short row block must refuse");
        let mut repeated = fd_copy(5, 3);
        repeated.enc[2] = 0;
        assert!(
            repeated.validate().is_err(),
            "a repeated own column must refuse"
        );
        let mut outside = fd_copy(5, 3);
        outside.enc[1] = 5;
        assert!(
            outside.validate().is_err(),
            "an own column >= n must refuse"
        );
        let mut overflow = fd_copy(5, 3);
        overflow.b1[1][0] = f32::INFINITY;
        assert!(overflow.validate().is_err(), "an f32 overflow must refuse");
        assert!(
            FdSearchProblem::default().validate().is_err(),
            "n = 0 must refuse"
        );
        assert!(fd_copy(5, 3).approx_bytes() >= 4 * 15);
    }

    #[test]
    fn fd_level_out_shapes_views_and_handles() {
        let mut out = LevelOut::default();
        out.prepare(3, 2);
        {
            let w = out.writer();
            assert_eq!((w.count, w.depth), (3, 2));
            assert_eq!(w.value.len(), 3 * FD_POINTS);
            assert_eq!(w.next.len(), 3 * 2 * 2);
            w.value[FD_POINTS + FD_FAR] = 0.75;
            w.far[2 * 2 + 1] = 9.0;
            w.next[2 * 2 * 2 + 3] = 4.0;
        }
        let view = out.hint().into_ingest();
        assert!(view.well_formed());
        assert_eq!((view.count(), view.depth()), (3, 2));
        assert_eq!(view.value(1, FD_FAR), 0.75);
        assert!(view.value(0, FD_WARM).is_nan(), "unwritten values are NaN");
        assert_eq!(view.far(1), (&[0.0, 9.0][..], &[0.0, 0.0][..]));
        assert_eq!(view.next(2).1, &[0.0, 4.0][..]);
        assert_eq!(LevelOut::bytes_for(3, 2), 3 * (8 + 12) * 4);
        // A handle only opens for its own payload type.
        let h = FdHandle::new("test", 7u32);
        assert_eq!(h.backend(), "test");
        assert_eq!(h.payload::<u32>(), Some(&7));
        assert!(h.payload::<u64>().is_none());
        // Chunk validation.
        let sides = [0u8, 1, 1, 0];
        let l = [0.0f32; 4];
        let ok = LevelChunk {
            depth: 2,
            count: 2,
            sides: &sides,
            l0: &l,
            l1: &l,
            deadline: None,
        };
        assert!(ok.validate().is_ok());
        assert!(LevelChunk { count: 3, ..ok }.validate().is_err());
        assert!(LevelChunk {
            depth: 0,
            count: 0,
            sides: &[],
            l0: &[],
            l1: &[],
            deadline: None
        }
        .validate()
        .is_err());
        let bad = [0u8, 2, 1, 0];
        assert!(LevelChunk { sides: &bad, ..ok }.validate().is_err());
    }

    #[test]
    fn untrusted_values_leave_only_through_a_hint() {
        let u = Untrusted::new(0.5_f64);
        assert_eq!(format!("{u:.2}"), "0.50");
        let v: f64 = u.into_hint().into_ingest();
        assert_eq!(v, 0.5);
        assert_eq!(
            Refusal::Memory { need: 2, cap: 1 }.to_string(),
            "memory: need 2 B > cap 1 B"
        );
        assert_eq!(
            Refusal::Fault("launch: ILLEGAL_ADDRESS".into()).to_string(),
            "device fault: launch: ILLEGAL_ADDRESS"
        );
    }

    /// The shared ingest rules and Adam schedule (ONE copy for the CPU
    /// reference and the device's host side) are the reference expressions.
    #[test]
    fn wave_ingest_rules_and_adam_schedule_are_the_reference_expressions() {
        assert_eq!(wave_ingest_rates(0.3, 0.2), (0.3, 0.2));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                wave_ingest_rates(bad, bad),
                (WAVE_DEFAULT_LR_ALPHA, WAVE_DEFAULT_LR_BETA)
            );
        }
        let d = WaveSearchCfg::default();
        assert_eq!(
            (d.lr_alpha, d.lr_beta),
            (WAVE_DEFAULT_LR_ALPHA, WAVE_DEFAULT_LR_BETA)
        );
        // The RUNTIME `powi` (a black-boxed exponent): what the CPU search's
        // and the device's host calls compute. A literal exponent may fold to
        // other bits (`1 - 0.9^4`), which is why both functions stay out of
        // line.
        let rt = |x: f64, t: i32| x.powi(std::hint::black_box(t));
        assert_eq!(wave_adam_bias(1), (1.0 - 0.9, 1.0 - 0.999));
        for t in [2, 4, 7, 9] {
            assert_eq!(wave_adam_bias(t), (1.0 - rt(0.9, t), 1.0 - rt(0.999, t)));
        }
        assert_eq!(
            wave_adam_bias(4).0.to_bits(),
            0.343_899_999_999_999_9_f64.to_bits()
        );
        assert_eq!(wave_lr_decay(0), 1.0);
        assert_eq!(wave_lr_decay(5), rt(0.98, 5));
        assert_eq!(wave_lr_decay(usize::MAX), rt(0.98, i32::MAX));
        assert_eq!(wave_warm_alpha(0.25), Some(0.25));
        assert_eq!(wave_warm_alpha(-3.0), Some(0.0));
        assert_eq!(wave_warm_alpha(7.0), Some(1.0));
        assert_eq!(wave_warm_alpha(f32::NAN), None);
        assert_eq!(wave_warm_alpha(f32::INFINITY), None);
        assert_eq!(wave_warm_beta(0.5), 0.5);
        for bad in [0.0, -0.0, -2.0, f32::NAN, f32::INFINITY] {
            assert_eq!(wave_warm_beta(bad).to_bits(), 0.0_f64.to_bits(), "{bad}");
        }
    }

    fn wave_problem() -> WaveProblem {
        WaveProblem {
            net: tiny(),
            boxes: vec![ReluBox {
                op: 1,
                l: vec![-1.0, 0.5],
                u: vec![2.0, 3.0],
            }],
            universe: vec![WaveSite { op: 1, neuron: 0 }],
            alpha0: vec![0.5],
            lo: vec![-1.0, -1.0],
            hi: vec![1.0, 1.0],
            target: 0,
            classes: vec![1],
        }
    }

    #[test]
    fn wave_problem_validation_refuses_bad_boxes_universe_and_rows() {
        let ok = wave_problem();
        assert_eq!(ok.validate().expect("valid"), vec![2, 2, 2, 2]);
        assert!(ok.approx_bytes() > 0);
        let mut m = wave_problem();
        m.boxes.clear();
        assert!(m.validate().is_err(), "a ReLU without a box must refuse");
        let mut m = wave_problem();
        m.boxes.push(m.boxes[0].clone());
        assert!(m.validate().is_err(), "two boxes for one ReLU must refuse");
        let mut m = wave_problem();
        m.boxes[0].op = 0;
        assert!(m.validate().is_err(), "a box on a non-ReLU op must refuse");
        let mut m = wave_problem();
        m.boxes[0].l[1] = f64::NAN;
        assert!(m.validate().is_err(), "a NaN box must refuse");
        let mut m = wave_problem();
        m.boxes[0].l[0] = 5.0;
        assert!(m.validate().is_err(), "a crossed box must refuse");
        let mut m = wave_problem();
        m.universe = vec![WaveSite { op: 1, neuron: 1 }];
        assert!(
            m.validate().is_err(),
            "a universe neuron stable in its box must refuse"
        );
        let mut m = wave_problem();
        m.universe.push(WaveSite { op: 1, neuron: 0 });
        m.alpha0.push(0.5);
        assert!(
            m.validate().is_err(),
            "a repeated universe neuron must refuse"
        );
        let mut m = wave_problem();
        m.universe = vec![WaveSite { op: 1, neuron: 7 }];
        assert!(m.validate().is_err(), "an out-of-range neuron must refuse");
        let mut m = wave_problem();
        m.alpha0.clear();
        assert!(
            m.validate().is_err(),
            "alpha0 must be one slope per position"
        );
        let mut m = wave_problem();
        m.classes = vec![0];
        assert!(
            m.validate().is_err(),
            "an objective row equal to the target"
        );
        let mut m = wave_problem();
        m.classes.clear();
        assert!(m.validate().is_err(), "no objective row");
        let mut m = wave_problem();
        m.target = 2;
        assert!(m.validate().is_err(), "target out of range");
        let mut m = wave_problem();
        m.lo[0] = 2.0;
        assert!(m.validate().is_err(), "a crossed input box");
    }

    #[test]
    fn wave_batch_validation_refuses_bad_offsets_repeats_and_rows() {
        let splits = [
            WaveSplit {
                pos: 0,
                active: true,
            },
            WaveSplit {
                pos: 1,
                active: false,
            },
        ];
        let beta = [0.0f32, 0.5];
        let rows = [0u32, 0];
        let offsets = [0u32, 1, 2];
        let ok = WaveBatch {
            count: 2,
            rows: &rows,
            offsets: &offsets,
            splits: &splits,
            beta: &beta,
            alpha: &[],
            stride: 2,
            deadline: None,
        };
        assert!(ok.validate(2, 1).is_ok());
        assert!(ok.validate(3, 1).is_err(), "stride mismatch");
        assert!(ok.validate(2, 0).is_err(), "row out of range");
        let alpha = [0.5f32; 4];
        assert!(WaveBatch {
            alpha: &alpha,
            ..ok
        }
        .validate(2, 1)
        .is_ok());
        assert!(WaveBatch {
            alpha: &alpha[..3],
            ..ok
        }
        .validate(2, 1)
        .is_err());
        let bad_offsets = [0u32, 2, 1];
        assert!(WaveBatch {
            offsets: &bad_offsets,
            ..ok
        }
        .validate(2, 1)
        .is_err());
        let short_offsets = [0u32, 1, 1];
        assert!(WaveBatch {
            offsets: &short_offsets,
            ..ok
        }
        .validate(2, 1)
        .is_err());
        let repeat = [
            WaveSplit {
                pos: 1,
                active: true,
            },
            WaveSplit {
                pos: 1,
                active: false,
            },
        ];
        let one = [0u32];
        let whole = [0u32, 2];
        assert!(
            WaveBatch {
                count: 1,
                rows: &one,
                offsets: &whole,
                splits: &repeat,
                ..ok
            }
            .validate(2, 1)
            .is_err(),
            "one neuron split twice in a domain must refuse"
        );
        let outside = [
            WaveSplit {
                pos: 0,
                active: true,
            },
            WaveSplit {
                pos: 2,
                active: true,
            },
        ];
        assert!(WaveBatch {
            splits: &outside,
            ..ok
        }
        .validate(2, 1)
        .is_err());
        assert!(WaveBatch {
            beta: &beta[..1],
            ..ok
        }
        .validate(2, 1)
        .is_err());
    }

    #[test]
    fn wave_out_and_branch_out_shapes_views_and_handles() {
        let mut out = WaveOut::default();
        out.prepare(2, 3, 4);
        {
            let w = out.writer();
            assert_eq!((w.count, w.stride), (2, 3));
            assert_eq!(w.alpha.len(), 6);
            assert_eq!(w.beta.len(), 4);
            w.lb[1] = 0.25;
            w.alpha[3] = 0.75;
            w.beta[2] = 1.5;
            w.evals[0] = 9;
        }
        let view = out.hint().into_ingest();
        assert!(view.well_formed());
        assert_eq!((view.count(), view.stride()), (2, 3));
        assert!(view.lb(0).is_nan(), "unwritten bounds are NaN");
        assert_eq!(view.lb(1), 0.25);
        assert!(view.scale(1).is_nan());
        assert_eq!(view.evals(0), 9);
        assert_eq!(view.beta(), &[0.0, 0.0, 1.5, 0.0][..]);
        assert!(view.alpha(0).iter().all(|a| a.is_nan()));
        assert_eq!(view.alpha(1)[0], 0.75);
        assert_eq!(WaveOut::bytes_for(2, 3, 4), 2 * (12 + 12) + 16);

        let mut br = BranchOut::default();
        br.prepare(3, 2);
        {
            let w = br.writer();
            assert_eq!((w.count, w.k), (3, 2));
            w.pos[2] = 7;
            w.score[2] = 0.5;
        }
        let bv = br.hint().into_ingest();
        assert!(bv.well_formed());
        assert_eq!((bv.count(), bv.k()), (3, 2));
        assert_eq!(bv.pos(1), &[7, WAVE_NONE][..]);
        assert_eq!(bv.score(1)[0], 0.5);
        assert!(bv.score(0)[0].is_nan());

        let h = WaveHandle::new("test", 5, 3u8);
        assert_eq!((h.backend(), h.stride()), ("test", 5));
        assert_eq!(h.payload::<u8>(), Some(&3));
        assert!(h.payload::<u16>().is_none());
        assert!(format!("{h:?}").contains("stride: 5"));
        let cfg = WaveSearchCfg::default();
        assert!(cfg.iters > 0 && cfg.lr_alpha > 0.0 && cfg.lr_beta > 0.0);
    }
}
