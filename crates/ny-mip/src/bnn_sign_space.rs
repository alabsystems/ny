// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LP-guided sign-space falsification for binarized (`Sign`) conv suffixes.
//!
//! Method and evidence: `docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md`.
//!
//! # Status: default-off, UNWIRED, SAT-only
//!
//! Nothing outside this module's own `#[cfg(test)]` oracle calls
//! [`falsify_bnn_sign_suffix_unwired`]. It is not reachable from any verifier
//! command or verdict path, mirroring the posture of
//! [`crate::verified_sdp_bound`] and [`crate::certified_box64`].
//!
//! [`SignSpaceOutcome`] **has no `Verified`/`Unsat` variant by construction**,
//! exactly as [`crate::solver::OneSidedSatDecline`] has none: a falsifier can
//! only ever exhibit a witness, so it can never authorize a verified verdict.
//! A [`SignSpaceOutcome::Refused`] is therefore never a soundness event — the
//! caller falls back to its existing path unchanged.
//!
//! A returned [`SignSpaceCandidate`] is a **claim**, not a verdict. Its
//! `input` MUST still be replayed through the ORIGINAL network and property by
//! the caller's existing witness gate before publication. Nothing here may be
//! trusted from the search's internal arithmetic.
//!
//! # The admitted fragment
//!
//! ```text
//!   x in [lo,hi] (H x W x C, NHWC)
//!     -> Conv1 (weights all +/-1, stride 1, VALID, dilation 1, groups 1, no bias)
//!     -> optional MaxPool (VALID, no dilation)
//!     -> optional per-channel affine (folded BatchNorm), scale > 0
//!     -> B(v) = Sign(Sign(v) + c),  0 < c < 1
//!     -> Conv2 (per-output-channel |W| constant and > 0)
//!     -> optional MaxPool -> optional affine -> B
//!     -> zero or more further BINARY stages, each either
//!          Conv (per-channel |W| constant > 0, optional bias, optional
//!                MaxPool, optional affine) -> B
//!        or Dense (per-column |W| constant > 0, optional affine) -> B
//!     -> Dense (weights all +/-1) -> logits
//! ```
//!
//! Anything else is [`SignSpaceRefusal`]. This module takes PLAIN DATA — it
//! has no ONNX dependency and knows nothing about graphs; admission of a real
//! graph into this shape is the caller's job, and the caller's admission is
//! re-checked here against the tensors themselves.
//!
//! # Zero maps to +1 (soundness-critical)
//!
//! ONNX `Sign(0) = 0`, which is not in `{-1,+1}`. The composite
//! `B(v) = Sign(Sign(v) + c)` with `0 < c < 1` maps `v = 0` (and `-0.0`)
//! through `0 + c > 0` to `+1`, so
//!
//! ```text
//!   B(v) = +1  iff  v >= 0        (NON-strict)
//! ```
//!
//! Getting this backwards is a direct correctness bug at exactly the decision
//! boundary. A BARE `Sign` is three-valued and is refused.
//!
//! # Per-channel weight scales, and why they are safe
//!
//! A folded BatchNorm turns a `+/-1` convolution into `W[c,..] = s_c * (+/-1)`.
//! The `Sign` boundary survives EXACTLY WHEN `s_c > 0`: with `n_c` the
//! unit-weight accumulator, `y_c = a_c * (s_c * n_c + b_c) + o_c`, so
//!
//! ```text
//!   B(y_c) = +1  iff  n_c >= t_c,   t_c = -(a_c*b_c + o_c) / (a_c*s_c)
//! ```
//!
//! provided `a_c > 0` and `s_c > 0`. Both are CHECKED per channel — a single
//! non-positive scale inverts a bit and silently corrupts every threshold
//! downstream of it, so it is refused, never absorbed. `|W|` being constant
//! within a channel is checked BITWISE (`f32` magnitudes compared for
//! equality), not within an epsilon band.
//!
//! # MaxPool: the OR identity
//!
//! A `MaxPool` sitting BEFORE its affine and `Sign` is a plain monotone max
//! over the pre-activations, and a positive affine slope preserves order, so
//!
//! ```text
//!   B(affine(max_w z_w)) = [max_w z_w >= t] = OR_w [z_w >= t]
//! ```
//!
//! which is why a pooled unit needs only ONE window member above the threshold
//! to be `+1`, but ALL of them below it to be `-1`. That asymmetry is used in
//! both directions: the realizability LP gives a `+1` unit a single row (the
//! member with the most slack at the incumbent point) and a `-1` unit one row
//! per member.
//!
//! With no pool the window is `1x1` and every formula degenerates to the
//! un-pooled one, so the shallow (`model_30`) path is bit-for-bit unchanged.
//!
//! # Exact FIXED/FREE prepass
//!
//! `z1_k = sum_p w_kp x_p` with `w in {+/-1}` and **each pixel index occurring
//! at most once per patch** (checked, not assumed). A linear form over
//! *distinct* variables on a product box attains its extrema coordinatewise,
//! so interval arithmetic is EXACT here, not a relaxation:
//!
//! ```text
//!   L_k = sum_{w=+1} lo_p - sum_{w=-1} hi_p
//!   U_k = sum_{w=+1} hi_p - sum_{w=-1} lo_p
//!   FIXED +1  iff  L_k >= t_k   (NON-strict, matching B)
//!   FIXED -1  iff  U_k <  t_k   (STRICT)
//!   FREE      otherwise
//! ```
//!
//! Swapping either strictness is a false-`unsat` generator upstream and a
//! wrong-witness generator here; both strictnesses are pinned by mutation
//! tests in `bnn_sign_space_tests.rs`.
//!
//! Under pooling the same two numbers are formed per POOLED unit as
//! `L_k = max_w L_w` and `U_k = max_w U_w`. `U_k` is still EXACT (the max of
//! the members' suprema is attained by driving the argmax member to its own
//! supremum). `L_k` is a sound LOWER BOUND but need not be attained, because
//! the members cannot in general be driven to their infima simultaneously. The
//! consequence is one-directional and safe: a unit that is really always `+1`
//! may be classified FREE, which costs an LP row, never a wrong sign.
//!
//! # Realizability LP (the heart of the method)
//!
//! ```text
//!   max s   s.t.  s1_k * (w_k . x - t_k) >= s   for all FREE k,
//!                 x in [lo,hi],   TOL <= s <= 1
//! ```
//!
//! FIXED units need no row: the exact prepass already proved every `x` in the
//! box realizes their sign. A feasible point means a real pixel vector
//! realizes the WHOLE sign pattern, so the logits are REAL rather than
//! relaxed — and the LP's primal `x` IS the counterexample.
//!
//! **Deviation from the paper form, and why.** The slack column is bounded
//! BELOW by the acceptance tolerance rather than by `0`. The question the
//! search actually asks is "is this pattern realizable at slack `>= TOL`",
//! not "what is the maximum slack", and the tightened bound turns every
//! rejection into an exact Farkas infeasibility. That matters: measured on
//! `model_30_idx_1703`, an `Infeasible` verdict costs ~18ms while an
//! `Optimal` costs ~220ms. The objective is still MAXIMIZE, so an accepted
//! pattern still gets the largest slack available and hence the most
//! `f32`-replay headroom.
//!
//! # Cost model (STEP 0, measured before the search was written)
//!
//! One exact-rational realizability LP on `model_30_idx_1703` (488 rows, ~2.7k
//! columns) costs ~60-220ms — a different cost class from the float LP the
//! original prototype used, and the reason the search pays ONE LP per BLOCK of
//! greedy flips (doubling on success, halving on rejection) instead of one per
//! flip. A full single-flip ranking of all 488 free bits costs ~0.5ms, so
//! ranking is free by comparison and is redone before every single flip.
//!
//! # Everything downstream of `s1` is exact integer arithmetic
//!
//! With `s1 in {+/-1}` and unit-signed weights, every stage accumulator, every
//! stage sign and the logits are integers, held here as `i32`/`i64`. The
//! search therefore has NO floating-point error at all; floats appear only in
//! the box, the `z1` prepass, the per-channel thresholds and the LP.
//! **Post-softmax margins saturate** (41-43 of 43 outputs underflow to `0.0f`
//! on the traffic-sign nets) and are useless; this module works exclusively in
//! PRE-SOFTMAX LOGITS, and softmax is order-preserving so the comparison is
//! invariant.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::ir::{Col, MilpProblem};

// ---------------------------------------------------------------------------
// Hard caps (crate style: `RELU_HARD_MAX_*`, `AY_LP_DUAL_HARD_MAX_*`).
// Exceeded => refuse BEFORE any allocation.
// ---------------------------------------------------------------------------

/// Hard cap on first-layer units (`H1 * W1 * C1`).
pub const SIGN_SPACE_HARD_MAX_UNITS: usize = 1 << 22;
/// Hard cap on the flattened width of any stage (`H * W * C`).
pub const SIGN_SPACE_HARD_MAX_FLAT: usize = 1 << 24;
/// Hard cap on FREE first-layer units (the genuine search dimension).
pub const SIGN_SPACE_HARD_MAX_FREE_UNITS: usize = 1 << 16;
/// Hard cap on realizability-LP columns (pixels + one slack).
pub const SIGN_SPACE_HARD_MAX_LP_COLUMNS: usize = 1 << 18;
/// Hard cap on realizability-LP rows.
///
/// Strictly below `ay_lib::WINDOW_ROWS_GATE` (8192), so this lane can never be
/// reclassified as window-class and can never pick up the window recipe or the
/// `NY_MIP_WINDOW_TIMEOUT_SECS` budget floor.
pub const SIGN_SPACE_HARD_MAX_LP_ROWS: usize = 8191;
/// Hard cap on realizability-LP structural nonzeros.
pub const SIGN_SPACE_HARD_MAX_LP_NONZEROS: usize = 1 << 24;
/// Hard cap on realizability-LP solves in one call.
pub const SIGN_SPACE_HARD_MAX_LP_SOLVES: usize = 1 << 20;
/// Hard cap on wall time for one call.
pub const SIGN_SPACE_HARD_MAX_WALL_TIME: Duration = Duration::from_hours(1);
/// Hard cap on the number of BINARY stages between the first layer and the
/// final dense (`conv2` is stage `0`).
pub const SIGN_SPACE_HARD_MAX_STAGES: usize = 8;
/// Hard cap on a single `MaxPool` window's area.
pub const SIGN_SPACE_HARD_MAX_POOL_AREA: usize = 64;
/// Hard cap on the exact-`f64`-accumulation guard: `taps * max|pixel|`.
///
/// `2^24` is the first integer not representable in `f32`; below it the tap
/// sum is exact in `f64` and in integral `f32` alike.
pub const SIGN_SPACE_EXACT_ACCUMULATION_LIMIT: f64 = 16_777_216.0;

/// How many elementary `f32` roundings a runtime may spend evaluating one
/// folded `affine(scale * n + bias)` decision.
///
/// The real count is at most six (the product, the bias add, the affine
/// multiply, the affine add, and the `f32` storage of the two constants);
/// eight is the conservative constant actually used.
const FOLDED_AFFINE_ROUNDINGS: f64 = 8.0;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Input tensor geometry, NHWC, matching the ONNX input and the row-major
/// flattening the vnnlib parser produces (`index = row*W*C + col*C + chan`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputGeometry {
    /// Height.
    pub height: usize,
    /// Width.
    pub width: usize,
    /// Channels.
    pub channels: usize,
}

/// A `MaxPool` sitting between a convolution and its `Sign`.
///
/// VALID only (no padding), no dilation, `ceil_mode = 0`: the output extent is
/// `(in - kernel) / stride + 1`, so trailing rows/columns that no window
/// covers are simply not read — exactly what ONNX does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSpec {
    /// Window height.
    pub kernel_h: usize,
    /// Window width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
}

impl PoolSpec {
    /// The `1x1` stride-`1` pool, which is the identity.
    pub const IDENTITY: Self = Self {
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
    };

    #[inline]
    const fn area(&self) -> usize {
        self.kernel_h * self.kernel_w
    }
}

/// `(in - kernel) / stride + 1`, or `None` when the window does not fit.
const fn pool_out_dim(in_dim: usize, kernel: usize, stride: usize) -> Option<usize> {
    if kernel == 0 || stride == 0 || in_dim < kernel {
        return None;
    }
    Some((in_dim - kernel) / stride + 1)
}

/// The output positions whose window covers input position `i`.
///
/// Window `o` covers `[o*stride, o*stride + kernel)`. Trailing inputs covered
/// by no window (`in = 11`, `kernel = stride = 2`, `out = 5` covers `0..10`)
/// yield an EMPTY range, which is the whole point of computing this rather
/// than assuming `i / stride`.
fn pool_parents(i: usize, kernel: usize, stride: usize, out: usize) -> std::ops::Range<usize> {
    if out == 0 || kernel == 0 || stride == 0 {
        return 0..0;
    }
    let hi = (i / stride).min(out - 1);
    let lo = (i + 1).saturating_sub(kernel).div_ceil(stride);
    if lo > hi {
        0..0
    } else {
        lo..hi + 1
    }
}

/// One convolution in the admitted fragment.
///
/// `weights` is in ONNX layout `[out_channels, in_channels, kernel_h,
/// kernel_w]`, row-major. Only stride 1, zero padding, dilation 1 and a single
/// group are admitted.
///
/// The FIRST convolution must be entrywise exactly `+/-1.0f32` with no bias;
/// every later convolution may carry a folded BatchNorm, i.e.
/// `W[c,..] = s_c * (+/-1)` with `s_c` finite and strictly positive plus an
/// optional per-channel `bias`.
#[derive(Debug, Clone, Copy)]
pub struct ConvSpec<'a> {
    /// `[out_channels, in_channels, kernel_h, kernel_w]`, row-major.
    pub weights: &'a [f32],
    /// Optional per-output-channel bias (a folded BatchNorm's shift).
    pub bias: Option<&'a [f64]>,
    /// Output channel count.
    pub out_channels: usize,
    /// Input channel count.
    pub in_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// `(stride_h, stride_w)`; only `(1, 1)` is admitted.
    pub stride: (usize, usize),
    /// `(pad_top, pad_left, pad_bottom, pad_right)`; only all-zero is admitted.
    pub padding: (usize, usize, usize, usize),
    /// `(dilation_h, dilation_w)`; only `(1, 1)` is admitted.
    pub dilation: (usize, usize),
    /// Convolution groups; only `1` is admitted.
    pub groups: usize,
}

impl<'a> ConvSpec<'a> {
    /// A stride-1, VALID, dilation-1, single-group, bias-free convolution —
    /// the only geometry this lane admits.
    pub fn valid_unit_stride(
        weights: &'a [f32],
        out_channels: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Self {
        Self {
            weights,
            bias: None,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }

    /// The same convolution with a per-output-channel bias attached.
    #[must_use]
    pub fn with_bias(mut self, bias: &'a [f64]) -> Self {
        self.bias = Some(bias);
        self
    }
}

/// A per-output-channel affine map sitting between a convolution (or its
/// `MaxPool`) and its `Sign`: `y_c = scale_c * z_c + offset_c`.
///
/// This is the FOLDED form of a BatchNorm (`scale = gamma / sqrt(var + eps)`,
/// `offset = beta - scale * mean`). `scale_c` must be finite and STRICTLY
/// POSITIVE: a negative scale inverts the downstream bit and silently corrupts
/// every threshold.
#[derive(Debug, Clone, Copy)]
pub struct SignSpaceAffine<'a> {
    /// Per-output-channel multiplier; must be finite and `> 0`.
    pub scale: &'a [f64],
    /// Per-output-channel additive term; must be finite.
    pub offset: &'a [f64],
}

/// The activation the caller found at a `Sign` site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignSpaceActivation<'a> {
    /// `Sign -> Add(add_constant) -> Sign`. Admitted iff `0 < add_constant < 1`,
    /// which is exactly the condition making `B(v) = +1 iff v >= 0`.
    SignAddSign {
        /// The scalar added between the two `Sign`s (measured `0.1` on the
        /// traffic-sign nets).
        add_constant: f64,
    },
    /// A BARE ONNX `Sign`. Three-valued at `0`, so never admitted.
    BareSign,
    /// A `Relu`. Never admitted.
    Relu,
    /// Anything else the caller saw, carried for the refusal message.
    Other {
        /// The op type as the caller names it.
        op: &'a str,
    },
}

/// One BINARY stage: it consumes a `+/-1` tensor and emits a `+/-1` tensor.
///
/// The `Conv` variant is much larger than `Dense`, which clippy flags. Boxing it
/// is the usual remedy and is the wrong trade here: a network carries a HANDFUL
/// of stages (they arrive as one `&[BinaryStage]`, one entry per layer), so the
/// wasted bytes are bounded by the layer count, while boxing would add an
/// allocation and a pointer hop to a `Copy` descriptor and change a public type.
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::large_enum_variant,
    reason = "a handful of stages per network; boxing costs more than the padding"
)]
pub enum BinaryStage<'a> {
    /// A convolution over the previous stage's signs.
    Conv {
        /// The convolution.
        conv: ConvSpec<'a>,
        /// `MaxPool` between the convolution and its affine/`Sign`.
        pool: Option<PoolSpec>,
        /// Folded per-channel affine (BatchNorm) after the pool.
        affine: Option<SignSpaceAffine<'a>>,
        /// The activation closing the stage.
        activation: SignSpaceActivation<'a>,
    },
    /// A dense layer over the previous stage's signs, `[in_dim, out_dim]`
    /// row-major — the ONNX `MatMul` right-hand operand layout.
    Dense {
        /// `[in_dim, out_dim]`, row-major.
        weights: &'a [f32],
        /// Input width.
        in_dim: usize,
        /// Output width.
        out_dim: usize,
        /// Folded per-column affine (BatchNorm).
        affine: Option<SignSpaceAffine<'a>>,
        /// The activation closing the stage.
        activation: SignSpaceActivation<'a>,
    },
}

/// A reference pre-softmax forward, supplied by the caller (e.g. ONNX Runtime).
///
/// When present, the module's own logit engine is checked against it EXACTLY
/// at the box centre and at sampled box vertices before any search runs — the
/// cheapest guard against a flatten/layout transposition that would otherwise
/// produce arithmetically plausible garbage. Returning `None` for a sample is
/// treated as "oracle unavailable" and skips that sample.
pub type ReferenceForward<'a> = &'a dyn Fn(&[f64]) -> Option<Vec<f64>>;

/// Everything the falsifier needs. Plain data: no I/O, no env reads, no ONNX.
pub struct SignSpaceRequest<'a> {
    /// Input geometry (NHWC).
    pub input: InputGeometry,
    /// First convolution (the only layer touching the real-valued box). Must
    /// be entrywise exactly `+/-1` and bias-free.
    pub conv1: ConvSpec<'a>,
    /// `MaxPool` between `conv1` and its affine/activation.
    pub conv1_pool: Option<PoolSpec>,
    /// Folded affine after `conv1` (and its pool), if any.
    pub conv1_affine: Option<SignSpaceAffine<'a>>,
    /// Activation after `conv1` (+ pool + affine).
    pub activation1: SignSpaceActivation<'a>,
    /// Second convolution, over the `+/-1` first-layer signs.
    pub conv2: ConvSpec<'a>,
    /// `MaxPool` between `conv2` and its affine/activation.
    pub conv2_pool: Option<PoolSpec>,
    /// Folded affine after `conv2` (and its pool), if any.
    pub conv2_affine: Option<SignSpaceAffine<'a>>,
    /// Activation after `conv2` (+ pool + affine).
    pub activation2: SignSpaceActivation<'a>,
    /// Further binary stages between `conv2`'s activation and the final dense.
    /// EMPTY is the shallow (`model_30`) shape.
    pub stages: &'a [BinaryStage<'a>],
    /// Final dense weights `[flat_in, num_classes]`, row-major, entrywise
    /// `+/-1.0f32`. A per-CLASS scale would not preserve the argmax and is
    /// therefore refused rather than folded.
    pub dense: &'a [f32],
    /// Output class count.
    pub num_classes: usize,
    /// Per-pixel lower bounds, flat NHWC. READ FROM THE VNNLIB — never
    /// reconstructed from a centre and an epsilon (box widths are not
    /// uniformly `2` and centres are not always integral).
    pub lo: &'a [f64],
    /// Per-pixel upper bounds, flat NHWC. Same provenance rule as [`Self::lo`].
    pub hi: &'a [f64],
    /// The property's target (true) class.
    pub target_class: usize,
    /// The challenger classes of the argmax-complement disjunction. Must be
    /// exactly every class other than [`Self::target_class`].
    pub challengers: &'a [usize],
    /// Search start, flat NHWC. `None` uses the box midpoint.
    pub reference_input: Option<&'a [f64]>,
    /// Optional independent pre-softmax forward for the layout self-check.
    pub reference_forward: Option<ReferenceForward<'a>>,
}

impl std::fmt::Debug for SignSpaceRequest<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignSpaceRequest")
            .field("input", &self.input)
            .field("stages", &self.stages.len())
            .field("num_classes", &self.num_classes)
            .field("target_class", &self.target_class)
            .field("challengers", &self.challengers.len())
            .field("has_reference_forward", &self.reference_forward.is_some())
            .finish_non_exhaustive()
    }
}

/// Which point the lazy row-generation loop adopts once a realizability LP
/// hands back a primal.
///
/// The LP's primal is a VERTEX of the pixel polytope, and on the deeper nets
/// jumping to it is destructive: the unit the search wanted to flip sits only
/// `20-75` away from its threshold, but the vertex is far enough away to break
/// `40-60` OTHER free units, which the next round then has to chase. Measured
/// active-set growth `4 -> 46 -> 88 -> ... -> 297` with the worst slack never
/// converging (`docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` §10).
///
/// Neither variant can affect SOUNDNESS. Both only choose which in-box point
/// the next round evaluates; acceptance is still decided by evaluating every
/// free unit's true OR-slack at a concrete point, and a candidate is still
/// re-derived from scratch by the witness finalizer and replayed by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SegmentMove {
    /// Adopt the LP primal wholesale. The behaviour every banked measurement
    /// was taken with; kept so the two can be A/B'd on the same row.
    #[default]
    Vertex,
    /// Adopt `x0 + theta*(x_LP - x0)` for the SMALLEST `theta` that carries
    /// every unit currently short of the tolerance past its own threshold.
    ///
    /// `z1` is LINEAR in `x`, so along the segment
    /// `z1_k(theta) = z1_k(x0) + theta*(z1_k(x_LP) - z1_k(x0))` EXACTLY, and
    /// both endpoint arrays are already in hand. Every crossing is therefore a
    /// closed form, not a bisection — and every unit whose `z1` does not cross
    /// its threshold on `[0, theta]` keeps its sign, which is precisely the
    /// incumbent pattern the vertex jump destroys.
    MinimalTheta,
}

/// WHERE the realizability LP is allowed to put the pixel vector.
///
/// The realizability LP maximizes ONE column — the slack `s`, capped at `1` —
/// and the other `n_pixels` columns appear in no objective at all. So the LP is
/// free to place every pixel anywhere in the box, and measurably does: on the
/// deeper traffic-sign nets adopting its primal breaks `40-60` OTHER free units,
/// lazy row generation chases them, and the active set grows
/// `4 -> 46 -> 88 -> ... -> 297` without the worst slack converging
/// (`docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` §10).
///
/// [`SegmentMove::MinimalTheta`] already measured out the FORWARD reading of
/// that (the slack cap means the vertex does not overshoot along the flip
/// direction; the minimal move is a median 97.6% of it). The remaining reading
/// is SIDEWAYS travel in the ~6900 columns the objective never mentions, and
/// this is the knob for it: intersect every pixel column's bounds with an
/// L-infinity ball around the incumbent.
///
/// SOUNDNESS. Every variant only ever INTERSECTS the vnnlib bounds, so the
/// restricted feasible set is a SUBSET of the full-box one:
///
/// * a point returned under a trust region is still in the box, and is still
///   accepted only by evaluating every free unit's true OR-slack on it;
/// * a restricted LP that fails proves NOTHING about the box, so a failure may
///   never be read as "unrealizable". [`Admitted::solve_lp_under_trust_region`]
///   is where that is enforced: it EXPANDS and re-solves, and the last radius
///   it tries is always the full box, whose answer is exactly today's.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TrustRegion {
    /// The full vnnlib box on every pixel column. The shipped behaviour and the
    /// configuration every banked traffic-signs number was measured on.
    #[default]
    FullBox,
    /// Start at `initial_fraction` of the box's widest half-width and DOUBLE on
    /// every LP that fails to hand back a point, up to the full box.
    ///
    /// A fraction rather than an absolute pixel radius because the fragment's
    /// boxes are not one geometry: the traffic-sign rows run `eps = 1..10` in
    /// raw pixel units, and the tiny fixtures in the test module have widths
    /// below `1`.
    Doubling {
        /// Opening radius as a fraction of the widest `(hi - lo) / 2`.
        initial_fraction: f64,
    },
    /// [`Self::Doubling`], plus `refine` bisection steps between the last radius
    /// that failed and the first that worked.
    ///
    /// This is the L-INFINITY PROXIMITY objective — `min ||x - x0||_inf` subject
    /// to the same sign rows, with `s >= tolerance` already a CONSTRAINT rather
    /// than an objective (the slack column's lower bound is the tolerance) —
    /// computed by bisection on the radius instead of by adding a column and
    /// `2 * n_pixels` rows to an exact-rational LP. The optimum is bracketed to
    /// `(hi - lo) / 2^refine` of the doubling step, and it costs ZERO extra
    /// columns and ZERO extra rows.
    Nearest {
        /// Opening radius as a fraction of the widest `(hi - lo) / 2`.
        initial_fraction: f64,
        /// Bisection steps after the first feasible radius.
        refine: usize,
    },
}

impl TrustRegion {
    /// The opening radius as a fraction of the widest half-width, or `None` for
    /// [`Self::FullBox`].
    const fn initial_fraction(self) -> Option<f64> {
        match self {
            Self::FullBox => None,
            Self::Doubling { initial_fraction }
            | Self::Nearest {
                initial_fraction, ..
            } => Some(initial_fraction),
        }
    }

    /// How many bisection steps to spend once a radius works.
    const fn refine(self) -> usize {
        match self {
            Self::FullBox | Self::Doubling { .. } => 0,
            Self::Nearest { refine, .. } => refine,
        }
    }
}

/// Budget and tolerance knobs. Every field is additionally clamped by the
/// `SIGN_SPACE_HARD_MAX_*` caps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignSpaceLimits {
    /// Realizability slack at which a flip is accepted.
    ///
    /// The LP is exact-rational, so this exists SOLELY to survive the `f32`
    /// ONNX Runtime replay. A value below the geometry-derived floor
    /// ([`f32_replay_slack_floor`]) is REFUSED rather than silently accepted;
    /// `0.05` carries roughly `7x` headroom at `3x3x3` and is BELOW the floor
    /// at `5x5x3`, where the caller must raise it.
    pub tolerance: f64,
    /// How many ranked single-flip candidates to spend LP on per round once
    /// the block schedule has collapsed to single flips.
    pub candidate_head: usize,
    /// Largest block of greedy flips tested by a SINGLE realizability LP.
    ///
    /// The block doubles on every realizable block and halves on every
    /// rejection, so this only caps how far the schedule can run ahead; a
    /// rejected block is rolled back bit-for-bit and never accepted.
    pub max_flip_batch: usize,
    /// Cap on FREE units.
    pub max_free_units: usize,
    /// Cap on LP columns.
    pub max_lp_columns: usize,
    /// Cap on LP rows.
    pub max_lp_rows: usize,
    /// Cap on LP structural nonzeros.
    pub max_lp_nonzeros: usize,
    /// Cap on LP solves.
    pub max_lp_solves: usize,
    /// Abandon the search after this many realizability LPs have been solved
    /// without EVER accepting a flip. `0` disables the rule.
    ///
    /// #bnn-lp-stall. This is a BUDGET rule, not a correctness one: the search
    /// can only ever end in `Candidate` or `Exhausted`, and stopping early
    /// reaches `Exhausted` sooner. It exists because the greedy walk is a
    /// PREFIX search — it ranks every unlocked free bit once and then pays one
    /// LP per ranked candidate until one is realizable — so a run that has
    /// accepted nothing has not narrowed anything, and the next LP costs what
    /// the last one did.
    ///
    /// MEASURED (`reports/measured-2026/traffic_signs_recognition_2023_NOTES.md`):
    /// on all nine open 48x48/64x64 rows the walk pays 127-207 LPs over the
    /// full 217 s lane budget and accepts **0** flips, every time; on the three
    /// `model_30` rows the lane owns it accepts its first flip inside the first
    /// handful of LPs and goes on to 78-99 flips. So the two populations are
    /// separated by a wide margin at any threshold in between, and the deeper
    /// nets stop paying ~210 s for a result they were always going to reach.
    ///
    /// CORRECTION, MEASURED 2026-08-17 AT HEAD, both dark levers at their
    /// shipped defaults: the "0 flips on all nine open rows, every time" claim
    /// above is NOT true of `model_48_idx_1703_eps_1`, which accepts **34**
    /// flips over 370 LPs, reaches best margin -82, produces no candidate and
    /// spends its whole 217.52 s lane budget — this rule never fires there,
    /// because `flips == 0` is false from the first accepted flip.
    /// `model_64_idx_1703_eps_1` DID stop here, at 32 LPs / 53.56 s. One of the
    /// nine was measured each way; the other seven are unmeasured. See
    /// [`Self::stall_margin_lp_solves`] for the rule that asks the question
    /// this one cannot.
    pub stall_lp_solves: usize,
    /// Abandon the search after this many realizability LPs have been solved
    /// without the best pattern margin STRICTLY improving. `0` disables it.
    ///
    /// #lane-value-stall. [`Self::stall_lp_solves`] is a NEVER-STARTED test:
    /// it asks whether the walk has accepted a flip, and a walk that accepts
    /// flips forever while the margin stands still disarms it permanently.
    /// MEASURED on `model_48_idx_1703_eps_1`: 370 LPs, 34 accepted flips, best
    /// margin -82, no candidate, the full 217.52 s lane budget spent — the
    /// stall rule never fired because `flips != 0`. This rule asks the
    /// question that actually prices the walk: has the thing the search is
    /// trying to move MOVED?
    ///
    /// Budget-only, exactly like its sibling: the two reachable ends are
    /// `Candidate` and `Exhausted`, and this can only bring `Exhausted`
    /// forward. It can never turn a SAT into anything else.
    ///
    /// DEFAULT 0 (disabled), so the shipped walk is byte-identical. It is
    /// armed only by the marginal-value lane scheduler, behind
    /// `NY_LANE_VALUE_SCHEDULER`, and the threshold it arms is not yet
    /// measured on the nine open rows.
    pub stall_margin_lp_solves: usize,
    /// Wall-clock budget for the whole call.
    pub max_wall_time: Duration,
    /// Per-LP time limit handed to `ay-milp`.
    pub per_lp_time: Duration,
    /// Cap on lazy row-generation rounds inside ONE realizability test.
    ///
    /// Running out of rounds declines the pattern, which only ever loses a
    /// SAT; it can never accept one.
    pub max_row_generation_rounds: usize,
    /// Which point each lazy row-generation round adopts from its LP primal.
    ///
    /// Soundness-neutral by construction — see [`SegmentMove`].
    pub segment_move: SegmentMove,
    /// Where the realizability LP is allowed to put the pixel vector.
    ///
    /// Soundness-neutral by construction — see [`TrustRegion`]: the bounds are
    /// always intersected with the vnnlib box, and a restricted LP that fails
    /// expands rather than concluding.
    pub trust_region: TrustRegion,
    /// Cap on trust-region expansions inside ONE realizability call.
    ///
    /// Purely a runaway guard. The doubling reaches the full box in
    /// `log2(1 / initial_fraction)` steps — 3 at `1/8`, 6 at `1/64` — so this
    /// is not reached by any declared arm; if it ever were, the region is
    /// dropped to the full box, which is today's behaviour and never a verdict.
    pub max_trust_expansions: usize,
    /// Box vertices sampled for the reference-forward layout self-check.
    pub self_check_vertex_samples: usize,
    /// Incremental flips replayed against a from-scratch recompute in the
    /// engine self-check.
    pub self_check_incremental_flips: usize,
}

impl Default for SignSpaceLimits {
    fn default() -> Self {
        Self {
            tolerance: 0.05,
            candidate_head: 8,
            max_flip_batch: 32,
            // The pooled first layer of the deeper traffic-sign nets is
            // 22x22x32 (`model_48`) and 30x30x32 (`model_64`), so a 4096 cap
            // refused them outright on the FREE-unit budget alone. This is
            // still half the hard cap and is not binding for `model_30`
            // (measured 488-999 free units), so nothing about the shallow path
            // changes.
            max_free_units: 32_768,
            max_lp_columns: 1 << 16,
            max_lp_rows: SIGN_SPACE_HARD_MAX_LP_ROWS,
            max_lp_nonzeros: 4_000_000,
            max_lp_solves: 20_000,
            // #bnn-lp-stall. 32 is chosen to sit far above the shallow path's
            // demonstrated need and far below the deep path's demonstrated
            // futility: the three `model_30` rows accept their first flip
            // early enough that this threshold does not change their
            // trajectory AT ALL (same flips, same LP count, same candidate —
            // pinned by the NOTES re-measurement), while the nine open 48/64
            // rows reach it after ~40-60 s instead of burning the full 217 s
            // to reach the same `Exhausted`.
            stall_lp_solves: 32,
            // DISABLED by default: see `stall_margin_lp_solves`. Arming it is
            // the lane scheduler's job and is dark.
            stall_margin_lp_solves: 0,
            // Measured: the three banked `model_30` eps=1 rows are recovered
            // in 58-110s on a 2026 laptop, so a 2-minute default would clip
            // the slowest of them.
            max_wall_time: Duration::from_mins(5),
            // Measured: an active-set LP on those rows solves in ~5-25ms and a
            // 10s cap let a single degenerate solve eat 20x the useful budget.
            // Abandoning at 1s costs a flip; abandoning is always safe, since
            // an unresolved LP is a REJECTED pattern, never an accepted one.
            per_lp_time: Duration::from_secs(1),
            max_row_generation_rounds: 24,
            // DEFAULT UNCHANGED, on purpose. Every banked measurement in
            // `docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` was taken on
            // the vertex jump, and §10's rule is that a lever becomes the
            // default only once it MEASURES better on the same rows.
            segment_move: SegmentMove::Vertex,
            // DEFAULT UNCHANGED, same rule: the shipped path keeps the full
            // vnnlib box on every pixel column, which is what every banked
            // measurement was taken on.
            trust_region: TrustRegion::FullBox,
            max_trust_expansions: 24,
            self_check_vertex_samples: 4,
            self_check_incremental_flips: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// The phase classification of one first-layer unit, from the EXACT interval
/// prepass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitPhase {
    /// `L_k >= t_k`: every `x` in the box gives `+1`.
    FixedPositive,
    /// `U_k < t_k`: every `x` in the box gives `-1`.
    FixedNegative,
    /// Both signs are attainable somewhere in the box.
    Free,
}

/// A CLAIMED counterexample. Never a verdict; the caller must replay
/// [`Self::input`] through the original network and property.
#[derive(Debug, Clone, PartialEq)]
pub struct SignSpaceCandidate {
    /// The counterexample, flat NHWC — the realizability LP's primal `x`,
    /// rounded to `f32` and re-clamped into the box, then re-verified by a
    /// from-scratch `f64` forward on exactly these values.
    pub input: Vec<f64>,
    /// Pre-softmax logits of [`Self::input`], exact integers.
    pub logits: Vec<i64>,
    /// `argmax` of [`Self::logits`], lowest index on a tie.
    pub argmax: usize,
    /// The best challenger, i.e. the one attaining [`Self::logit_margin`].
    pub best_challenger: usize,
    /// `logits[best_challenger] - logits[target]`, STRICTLY POSITIVE.
    pub logit_margin: i64,
    /// Realizability slack of the accepted sign pattern.
    pub lp_slack: f64,
    /// FREE unit count from the exact prepass.
    pub free_units: usize,
    /// Accepted sign flips relative to the reference point's pattern.
    pub flips: usize,
    /// Realizability LPs solved.
    pub lp_solves: usize,
    /// Wall time consumed.
    pub elapsed: Duration,
}

/// Why the lane declined. NEVER interpretable as any verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum SignSpaceRefusal {
    /// The activation is not `Sign -> Add(c) -> Sign` with `0 < c < 1`, so
    /// `B(v) = +1 iff v >= 0` is not established.
    CompositeActivationNotBinary {
        /// `1` for the post-conv1 site, `2` for post-conv2, `3 + i` for the
        /// `i`-th further binary stage.
        site: usize,
        /// What was seen.
        detail: String,
    },
    /// Some weight is not exactly `+1.0f32` / `-1.0f32` (bitwise, not epsilon).
    ///
    /// Only the FIRST convolution and the FINAL dense are held to this: the
    /// first because the `f32` replay bound is derived for unit taps, the
    /// final because a per-class scale does not preserve the argmax.
    NonUnitWeights {
        /// Which tensor.
        tensor: &'static str,
        /// Flat index of the first offender.
        index: usize,
        /// Its value.
        value: f32,
    },
    /// `|W|` is not constant within an output channel, so the tensor is not
    /// `s_c * (+/-1)` and the `Sign` boundary is not a threshold on an integer
    /// accumulator.
    ChannelScaleNotConstant {
        /// Which tensor.
        tensor: &'static str,
        /// The offending output channel.
        channel: usize,
        /// Flat index of the first entry disagreeing with the channel's scale.
        index: usize,
        /// Its value.
        value: f32,
        /// The scale established by the channel's earlier entries.
        scale: f64,
    },
    /// A per-output-channel weight scale is zero, negative or non-finite, so
    /// folding it into a threshold would invert or destroy the bit.
    NonPositiveChannelScale {
        /// Which tensor.
        tensor: &'static str,
        /// The offending output channel.
        channel: usize,
        /// Its scale.
        scale: f64,
    },
    /// A conv1 patch repeats an input index, so interval arithmetic would be a
    /// RELAXATION rather than exact and FIXED/FREE would be unsound.
    PatchIndicesNotDistinct {
        /// Spatial output position (row-major) whose patch repeats an index.
        position: usize,
    },
    /// `taps * max|pixel|` reaches `2^24`, so tap-sum exactness is not proven.
    AccumulationNotExact {
        /// Taps per conv1 unit.
        taps: usize,
        /// Largest `|lo|`/`|hi|` over the box.
        max_abs: f64,
    },
    /// A convolution geometry outside stride-1 / VALID / dilation-1 / groups-1.
    UnsupportedConvGeometry {
        /// Which convolution.
        tensor: &'static str,
        /// What was seen.
        detail: String,
    },
    /// A pooling geometry this lane cannot reason about.
    UnsupportedPoolGeometry {
        /// `1` for post-conv1, `2` for post-conv2, `3 + i` for stage `i`.
        site: usize,
        /// What was seen.
        detail: String,
    },
    /// A folded affine with a non-positive or non-finite scale (it would
    /// invert the downstream bit), or a non-finite offset.
    BatchNormNotFoldable {
        /// `1` for post-conv1, `2` for post-conv2, `3 + i` for stage `i`.
        site: usize,
        /// Offending channel.
        channel: usize,
        /// Its scale.
        scale: f64,
    },
    /// A folded convolution bias is not finite.
    FoldedBiasNotFinite {
        /// `1` for post-conv1, `2` for post-conv2, `3 + i` for stage `i`.
        site: usize,
        /// Offending channel.
        channel: usize,
        /// Its bias.
        bias: f64,
    },
    /// Tensor lengths disagree with the declared geometry.
    ShapeMismatch {
        /// What disagreed.
        detail: String,
    },
    /// The property is not the argmax-complement disjunction over every class.
    PropertyNotArgmaxComplement {
        /// What disagreed.
        detail: String,
    },
    /// A box coordinate is non-finite or inverted.
    DegenerateBox {
        /// Flat pixel index.
        index: usize,
        /// Its lower bound.
        lo: f64,
        /// Its upper bound.
        hi: f64,
    },
    /// The caller's acceptance tolerance is below the `f32`-replay floor this
    /// geometry demands.
    ToleranceBelowFloor {
        /// What the caller asked for.
        requested: f64,
        /// The derived floor.
        floor: f64,
    },
    /// The logit engine disagreed with the caller's reference forward, or the
    /// incremental delta path disagreed with a from-scratch recompute.
    LogitEngineDisagrees {
        /// Which self-check sample.
        sample: usize,
        /// Which output class.
        class: usize,
        /// The module's value.
        engine: f64,
        /// The reference value.
        reference: f64,
    },
    /// A declared or hard limit would be exceeded.
    LimitExceeded {
        /// Which limit.
        limit: &'static str,
        /// What the request needs.
        requested: u128,
        /// The cap.
        cap: u128,
    },
}

/// What one call produced. **There is deliberately no `Verified`/`Unsat`
/// variant**: this lane can only ever exhibit a witness.
#[derive(Debug, Clone, PartialEq)]
pub enum SignSpaceOutcome {
    /// A claimed counterexample, pending the caller's independent replay.
    Candidate(Box<SignSpaceCandidate>),
    /// The search ran out of realizable improving flips, LP budget or time.
    Exhausted {
        /// Best pre-softmax margin reached in pattern space (`<= 0`).
        best_logit_margin: i64,
        /// MARGINAL VALUE, in the lane's own unit: how far the best pattern
        /// margin moved over the whole walk (`best - initial`, `>= 0`).
        ///
        /// This is the number the walk can honestly price itself by, and it is
        /// NOT `flips`: a walk can accept 34 flips and move the margin
        /// nowhere. Denominator: [`Self::Exhausted::lp_solves`].
        margin_gain: i64,
        /// FREE unit count from the exact prepass.
        free_units: usize,
        /// Accepted flips.
        flips: usize,
        /// Realizability LPs solved.
        lp_solves: usize,
        /// Wall time consumed.
        elapsed: Duration,
    },
    /// The request is outside the admitted fragment. Not a soundness event.
    Refused(SignSpaceRefusal),
}

/// A genuine failure of the LP machinery (as opposed to a typed refusal).
#[derive(Debug, thiserror::Error)]
pub enum SignSpaceError {
    /// The realizability LP could not be lowered to an `ay-milp` model.
    #[error("realizability LP lowering failed: {0}")]
    Lowering(String),
    /// `ay-milp` reported an error.
    #[error("realizability LP solver error: {0}")]
    Solver(String),
}

// ---------------------------------------------------------------------------
// Tolerance floor
// ---------------------------------------------------------------------------

/// Half an `f32` ULP at `v` (`0` at `v == 0`), used to bound rounding error.
fn f32_half_ulp(v: f64) -> f64 {
    let v = v.abs();
    if v == 0.0 || !v.is_finite() {
        return 0.0;
    }
    // f32 has a 24-bit significand: for v in [2^e, 2^(e+1)), ulp = 2^(e-23).
    let e = v.log2().floor();
    (2.0f64).powf(e - 24.0)
}

/// Bound on the `f64` evaluation error of one `taps`-term unit-weight dot
/// product, used ONLY to keep lazy row generation from rejecting a pattern the
/// exact-rational LP just certified at exactly the tolerance.
///
/// `taps - 1` additions, each a half ULP at the largest partial-sum magnitude.
/// At `27` taps and pixel scale `255` this is ~`2e-11`, i.e. nine orders of
/// magnitude below the `f32` replay floor that actually gates the witness.
fn f64_slack_verification_epsilon(taps: usize, max_abs: f64) -> f64 {
    if taps == 0 {
        return 0.0;
    }
    (taps - 1) as f64 * f64::EPSILON * 0.5 * (taps as f64 * max_abs)
}

/// The realizability slack a sign pattern must carry to survive the `f32`
/// ONNX Runtime replay of the FIRST convolution.
///
/// Derivation (re-derived here rather than inherited as a constant, because it
/// is GEOMETRY-SPECIFIC): the products `w_kp * x_p` are exact (`w = +/-1`), so
/// the only errors are
///
/// * the `f64 -> f32` rounding of each of the `taps` pixel values:
///   `taps * halfulp(max_abs)`;
/// * `taps - 1` accumulations, each bounded by a half ULP at the largest
///   partial-sum magnitude `taps * max_abs`, regardless of summation order:
///   `(taps - 1) * halfulp(taps * max_abs)`.
///
/// At `3x3x3` with `max_abs = 255`: `27 * 255 = 6885 in [2^12, 2^13)`, so
/// `halfulp = 2^-12`, giving `26 * 2.44e-4 + 27 * 7.63e-6 ~= 6.6e-3` — the
/// default `TOL = 0.05` carries ~7.6x headroom. At `5x5x3`:
/// `75 * 255 = 19125 in [2^14, 2^15)`, `halfulp = 2^-10`, giving `~7.3e-2`, so
/// `0.05` is BELOW the floor there and is refused.
///
/// A `MaxPool` between the convolution and its `Sign` does NOT change this:
/// `max` is exact in any precision, so a pooled unit's decision is still one
/// `taps`-term dot product compared against a threshold.
#[must_use]
pub fn f32_replay_slack_floor(taps: usize, max_abs: f64) -> f64 {
    if taps == 0 {
        return 0.0;
    }
    let accum = (taps - 1) as f64 * f32_half_ulp(taps as f64 * max_abs);
    let inputs = taps as f64 * f32_half_ulp(max_abs);
    accum + inputs
}

/// Accumulator-space slack a folded-affine decision must carry to survive an
/// `f32` replay of `slope * n + offset >= 0`.
///
/// EXACTLY ZERO whenever `offset == 0`. Then the decision is
/// `fl(slope * n) >= 0` with `slope > 0`, and `f32` rounding is
/// SIGN-PRESERVING — it never carries a nonzero across zero and it maps `0` to
/// `0` — so the comparison is exact no matter what `slope` is. That covers the
/// shallow net (`slope == 1`, `offset == 0`, no affine arithmetic at all),
/// which is why widening the fragment cannot move the shallow path.
fn folded_affine_guard(slope: f64, offset: f64, max_abs_accumulator: f64) -> f64 {
    if offset == 0.0 {
        return 0.0;
    }
    let magnitude = slope * max_abs_accumulator + offset.abs();
    FOLDED_AFFINE_ROUNDINGS * f32_half_ulp(magnitude) / slope
}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

/// One binary stage, normalized: unit-signed weights plus per-channel
/// thresholds on the (possibly pooled) INTEGER accumulator.
struct Stage {
    kind: StageKind,
    /// `Sign` fires `+1` iff the pooled accumulator is `>= t[channel]`.
    t: Vec<f64>,
    /// Accumulator-space `f32` replay guard, per channel.
    guard: Vec<f64>,
    /// Channels (the modulus mapping an output index to a threshold).
    channels: usize,
    /// Pre-pool accumulator length.
    z_len: usize,
    /// Post-pool output length.
    out_len: usize,
}

/// Owned mirror of [`BinaryStage`]; same size trade-off, same reasoning.
#[allow(
    clippy::large_enum_variant,
    reason = "a handful of stages per network; boxing costs more than the padding"
)]
enum StageKind {
    Conv {
        /// `[out, in, kh, kw]` signs.
        w: Vec<i8>,
        out_c: usize,
        in_c: usize,
        kh: usize,
        kw: usize,
        in_w: usize,
        conv_h: usize,
        conv_w: usize,
        pool: PoolSpec,
        out_h: usize,
        out_w: usize,
    },
    Dense {
        /// `[in_dim, out_dim]` signs.
        w: Vec<i8>,
        in_dim: usize,
        out_dim: usize,
    },
}

/// The shape flowing between stages.
#[derive(Debug, Clone, Copy)]
enum StageShape {
    Spatial { h: usize, w: usize, c: usize },
    Flat { len: usize },
}

impl StageShape {
    const fn len(self) -> usize {
        match self {
            Self::Spatial { h, w, c } => h * w * c,
            Self::Flat { len } => len,
        }
    }
}

/// The validated problem: everything the search may assume.
struct Admitted<'a> {
    input: InputGeometry,
    /// conv1 signs, `[out, in, kh, kw]`.
    conv1_w: Vec<i8>,
    conv1: ConvSpec<'a>,
    /// conv1 RAW output geometry (before pooling).
    h1c: usize,
    w1c: usize,
    pool1: PoolSpec,
    /// conv1 POOLED output geometry — the search's unit space.
    h1: usize,
    w1: usize,
    c1: usize,
    /// Per-channel `Sign` thresholds on the raw conv1 dot product.
    t1: Vec<f64>,
    /// Per-channel `f32` replay guard added to the geometry floor.
    guard1: Vec<f64>,
    /// Binary stages; `stages[0]` is always `conv2`.
    stages: Vec<Stage>,
    dense: &'a [f32],
    num_classes: usize,
    lo: &'a [f64],
    hi: &'a [f64],
    target_class: usize,
    challengers: Vec<usize>,
    n_pixels: usize,
    /// Raw (pre-pool) conv1 unit count.
    n_raw1: usize,
    /// Pooled conv1 unit count — the search dimension.
    n_units1: usize,
    /// Final dense input width.
    n_flat: usize,
    taps1: usize,
    /// Largest `|lo|`/`|hi|` over the whole box.
    max_abs: f64,
}

impl Admitted<'_> {
    #[inline]
    const fn unit_index(&self, r: usize, c: usize, ch: usize) -> usize {
        (r * self.w1 + c) * self.c1 + ch
    }

    /// `(row, col, channel)` of a POOLED first-layer unit index.
    #[inline]
    const fn unit_coords(&self, k: usize) -> (usize, usize, usize) {
        let ch = k % self.c1;
        let spatial = k / self.c1;
        (spatial / self.w1, spatial % self.w1, ch)
    }

    #[inline]
    const fn raw_index(&self, r: usize, c: usize, ch: usize) -> usize {
        (r * self.w1c + c) * self.c1 + ch
    }

    /// The RAW conv1 positions the pooled unit at `(r, c)` maxes over.
    ///
    /// Every member is in range by construction of the pooled extent, so this
    /// never filters.
    fn pool1_members(&self, r: usize, c: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let pool = self.pool1;
        (0..pool.kernel_h).flat_map(move |a| {
            (0..pool.kernel_w).map(move |b| (r * pool.stride_h + a, c * pool.stride_w + b))
        })
    }

    #[inline]
    fn dense_weight(&self, j: usize, class: usize) -> i64 {
        i64::from(self.dense[j * self.num_classes + class] as i8)
    }

    #[inline]
    fn conv1_weight(&self, oc: usize, ic: usize, kr: usize, kc: usize) -> f64 {
        let k = ((oc * self.conv1.in_channels + ic) * self.conv1.kernel_h + kr)
            * self.conv1.kernel_w
            + kc;
        f64::from(self.conv1_w[k])
    }

    /// The `taps1` flat pixel indices of the conv1 patch at RAW output
    /// `(r, c)`, paired with their `+/-1` weights for output channel `ch`.
    fn conv1_patch(&self, r: usize, c: usize, ch: usize, out: &mut Vec<(usize, f64)>) {
        out.clear();
        let InputGeometry {
            width, channels, ..
        } = self.input;
        for kr in 0..self.conv1.kernel_h {
            for kc in 0..self.conv1.kernel_w {
                for ic in 0..self.conv1.in_channels {
                    let p = ((r + kr) * width + (c + kc)) * channels + ic;
                    out.push((p, self.conv1_weight(ch, ic, kr, kc)));
                }
            }
        }
    }
}

fn check_conv_geometry(conv: &ConvSpec<'_>, tensor: &'static str) -> Result<(), SignSpaceRefusal> {
    if conv.stride != (1, 1) {
        return Err(SignSpaceRefusal::UnsupportedConvGeometry {
            tensor,
            detail: format!("stride {:?} != (1, 1)", conv.stride),
        });
    }
    if conv.padding != (0, 0, 0, 0) {
        return Err(SignSpaceRefusal::UnsupportedConvGeometry {
            tensor,
            detail: format!("padding {:?} != VALID", conv.padding),
        });
    }
    if conv.dilation != (1, 1) {
        return Err(SignSpaceRefusal::UnsupportedConvGeometry {
            tensor,
            detail: format!("dilation {:?} != (1, 1)", conv.dilation),
        });
    }
    if conv.groups != 1 {
        return Err(SignSpaceRefusal::UnsupportedConvGeometry {
            tensor,
            detail: format!("groups {} != 1", conv.groups),
        });
    }
    if conv.kernel_h == 0 || conv.kernel_w == 0 || conv.out_channels == 0 || conv.in_channels == 0 {
        return Err(SignSpaceRefusal::UnsupportedConvGeometry {
            tensor,
            detail: "zero-sized convolution".to_string(),
        });
    }
    let expected = conv.out_channels * conv.in_channels * conv.kernel_h * conv.kernel_w;
    if conv.weights.len() != expected {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "{tensor} has {} entries, geometry needs {expected}",
                conv.weights.len()
            ),
        });
    }
    if let Some(bias) = conv.bias {
        if bias.len() != conv.out_channels {
            return Err(SignSpaceRefusal::ShapeMismatch {
                detail: format!(
                    "{tensor} bias has {} entries, geometry needs {}",
                    bias.len(),
                    conv.out_channels
                ),
            });
        }
    }
    Ok(())
}

fn check_unit_weights(weights: &[f32], tensor: &'static str) -> Result<(), SignSpaceRefusal> {
    for (index, &value) in weights.iter().enumerate() {
        // Bitwise +/-1, not an epsilon band: a 0.999 weight is a different
        // network, not a rounding artifact.
        if value != 1.0f32 && value != -1.0f32 {
            return Err(SignSpaceRefusal::NonUnitWeights {
                tensor,
                index,
                value,
            });
        }
    }
    Ok(())
}

/// Which output channel a flat weight index belongs to.
#[derive(Debug, Clone, Copy)]
enum ChannelLayout {
    /// `[out, ...]`: `channel = index / per_channel`.
    OutMajor { per_channel: usize },
    /// `[in, out]`: `channel = index % out_channels`.
    InMajor { out_channels: usize },
}

/// Split a weight tensor into its unit SIGN pattern and its per-output-channel
/// magnitude, refusing anything that is not `W[c, ..] = s_c * (+/-1)` with
/// `s_c` finite and strictly positive.
///
/// The magnitude test is BITWISE (`f32` absolute values compared for exact
/// equality), matching the way the folded-BatchNorm tensors actually look: a
/// channel is one scale repeated, not a band of nearly-equal numbers.
fn split_channel_signs(
    weights: &[f32],
    out_channels: usize,
    layout: ChannelLayout,
    tensor: &'static str,
) -> Result<(Vec<i8>, Vec<f64>), SignSpaceRefusal> {
    let mut scales = vec![f64::NAN; out_channels];
    let mut signs = vec![0i8; weights.len()];
    for (index, &value) in weights.iter().enumerate() {
        let channel = match layout {
            ChannelLayout::OutMajor { per_channel } => index / per_channel,
            ChannelLayout::InMajor { out_channels } => index % out_channels,
        };
        if channel >= out_channels {
            return Err(SignSpaceRefusal::ShapeMismatch {
                detail: format!("{tensor} index {index} maps outside {out_channels} channels"),
            });
        }
        if !value.is_finite() || value == 0.0f32 {
            return Err(SignSpaceRefusal::NonUnitWeights {
                tensor,
                index,
                value,
            });
        }
        let magnitude = f64::from(value.abs());
        if scales[channel].is_nan() {
            scales[channel] = magnitude;
        } else if scales[channel] != magnitude {
            return Err(SignSpaceRefusal::ChannelScaleNotConstant {
                tensor,
                channel,
                index,
                value,
                scale: scales[channel],
            });
        }
        signs[index] = if value > 0.0f32 { 1 } else { -1 };
    }
    for (channel, &scale) in scales.iter().enumerate() {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(SignSpaceRefusal::NonPositiveChannelScale {
                tensor,
                channel,
                scale,
            });
        }
    }
    Ok((signs, scales))
}

/// Turn one activation site into the composite-binary proof obligation.
fn check_activation(
    activation: &SignSpaceActivation<'_>,
    site: usize,
) -> Result<(), SignSpaceRefusal> {
    match *activation {
        SignSpaceActivation::SignAddSign { add_constant } => {
            // B(v) = Sign(Sign(v) + c). For 0 < c < 1:
            //   v > 0 -> +1 + c > 0 -> +1
            //   v = 0 ->  0 + c > 0 -> +1     <-- ZERO MAPS TO +1
            //   v < 0 -> -1 + c < 0 -> -1
            // Outside (0, 1) the middle or an outer case collapses.
            if add_constant.is_finite() && add_constant > 0.0 && add_constant < 1.0 {
                Ok(())
            } else {
                Err(SignSpaceRefusal::CompositeActivationNotBinary {
                    site,
                    detail: format!("Sign->Add({add_constant})->Sign needs 0 < c < 1"),
                })
            }
        }
        SignSpaceActivation::BareSign => Err(SignSpaceRefusal::CompositeActivationNotBinary {
            site,
            detail: "bare ONNX Sign is three-valued at 0".to_string(),
        }),
        SignSpaceActivation::Relu => Err(SignSpaceRefusal::CompositeActivationNotBinary {
            site,
            detail: "Relu is not a binary activation".to_string(),
        }),
        SignSpaceActivation::Other { op } => Err(SignSpaceRefusal::CompositeActivationNotBinary {
            site,
            detail: format!("unsupported activation `{op}`"),
        }),
    }
}

/// Validate a pool and return it (or the identity when there is none).
fn check_pool(pool: Option<PoolSpec>, site: usize) -> Result<PoolSpec, SignSpaceRefusal> {
    let Some(pool) = pool else {
        return Ok(PoolSpec::IDENTITY);
    };
    if pool.kernel_h == 0 || pool.kernel_w == 0 || pool.stride_h == 0 || pool.stride_w == 0 {
        return Err(SignSpaceRefusal::UnsupportedPoolGeometry {
            site,
            detail: format!("degenerate pool {pool:?}"),
        });
    }
    if pool.area() > SIGN_SPACE_HARD_MAX_POOL_AREA {
        return Err(SignSpaceRefusal::UnsupportedPoolGeometry {
            site,
            detail: format!(
                "window area {} exceeds {SIGN_SPACE_HARD_MAX_POOL_AREA}",
                pool.area()
            ),
        });
    }
    Ok(pool)
}

/// Fold a per-channel weight scale, an optional bias and an optional affine
/// into a threshold on the UNIT-WEIGHT accumulator, plus the accumulator-space
/// `f32` replay guard for that decision.
///
/// `y_c = a_c * (s_c * n_c + b_c) + o_c` with `a_c > 0` and `s_c > 0`, so
/// `B(y_c) = +1  iff  n_c >= t_c` with `t_c = -(a_c*b_c + o_c) / (a_c*s_c)`.
/// A negative `a_c` or `s_c` would REVERSE that equivalence, which is why both
/// are refused rather than absorbed.
fn fold_thresholds(
    scales: &[f64],
    bias: Option<&[f64]>,
    affine: Option<&SignSpaceAffine<'_>>,
    channels: usize,
    site: usize,
    max_abs_accumulator: f64,
) -> Result<(Vec<f64>, Vec<f64>), SignSpaceRefusal> {
    if scales.len() != channels {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "site {site} has {} channel scales, geometry needs {channels}",
                scales.len()
            ),
        });
    }
    if let Some(affine) = affine {
        if affine.scale.len() != channels || affine.offset.len() != channels {
            return Err(SignSpaceRefusal::ShapeMismatch {
                detail: format!(
                    "affine at site {site} has {}/{} entries, geometry needs {channels}",
                    affine.scale.len(),
                    affine.offset.len()
                ),
            });
        }
    }
    if let Some(bias) = bias {
        if bias.len() != channels {
            return Err(SignSpaceRefusal::ShapeMismatch {
                detail: format!(
                    "bias at site {site} has {} entries, geometry needs {channels}",
                    bias.len()
                ),
            });
        }
    }
    let mut thresholds = Vec::with_capacity(channels);
    let mut guards = Vec::with_capacity(channels);
    for channel in 0..channels {
        let weight_scale = scales[channel];
        let (affine_scale, affine_offset) = affine.map_or((1.0, 0.0), |affine| {
            (affine.scale[channel], affine.offset[channel])
        });
        if !affine_scale.is_finite() || affine_scale <= 0.0 || !affine_offset.is_finite() {
            return Err(SignSpaceRefusal::BatchNormNotFoldable {
                site,
                channel,
                scale: affine_scale,
            });
        }
        let bias_c = bias.map_or(0.0, |bias| bias[channel]);
        if !bias_c.is_finite() {
            return Err(SignSpaceRefusal::FoldedBiasNotFinite {
                site,
                channel,
                bias: bias_c,
            });
        }
        let slope = affine_scale * weight_scale;
        let offset = affine_scale.mul_add(bias_c, affine_offset);
        if !slope.is_finite() || slope <= 0.0 || !offset.is_finite() {
            return Err(SignSpaceRefusal::BatchNormNotFoldable {
                site,
                channel,
                scale: slope,
            });
        }
        let threshold = if offset == 0.0 { 0.0 } else { -offset / slope };
        let guard = folded_affine_guard(slope, offset, max_abs_accumulator);
        if !threshold.is_finite() || !guard.is_finite() {
            return Err(SignSpaceRefusal::BatchNormNotFoldable {
                site,
                channel,
                scale: slope,
            });
        }
        thresholds.push(threshold);
        guards.push(guard);
    }
    Ok((thresholds, guards))
}

fn limit(name: &'static str, requested: usize, cap: usize) -> Result<(), SignSpaceRefusal> {
    if requested > cap {
        return Err(SignSpaceRefusal::LimitExceeded {
            limit: name,
            requested: requested as u128,
            cap: cap as u128,
        });
    }
    Ok(())
}

/// Normalize one binary convolution stage. `shape` is the incoming shape.
fn admit_conv_stage(
    conv: &ConvSpec<'_>,
    pool: Option<PoolSpec>,
    affine: Option<&SignSpaceAffine<'_>>,
    shape: StageShape,
    site: usize,
    tensor: &'static str,
) -> Result<(Stage, StageShape), SignSpaceRefusal> {
    check_conv_geometry(conv, tensor)?;
    let StageShape::Spatial { h, w, c } = shape else {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!("{tensor} follows a flattened stage; a convolution needs a grid"),
        });
    };
    if conv.in_channels != c {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "{tensor} takes {} channels, the previous stage produces {c}",
                conv.in_channels
            ),
        });
    }
    if h < conv.kernel_h || w < conv.kernel_w {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!("{tensor} kernel exceeds its {h}x{w} input"),
        });
    }
    let conv_h = h - conv.kernel_h + 1;
    let conv_w = w - conv.kernel_w + 1;
    let pool = check_pool(pool, site)?;
    let (Some(out_h), Some(out_w)) = (
        pool_out_dim(conv_h, pool.kernel_h, pool.stride_h),
        pool_out_dim(conv_w, pool.kernel_w, pool.stride_w),
    ) else {
        return Err(SignSpaceRefusal::UnsupportedPoolGeometry {
            site,
            detail: format!("{pool:?} does not fit a {conv_h}x{conv_w} feature map"),
        });
    };
    let out_c = conv.out_channels;
    let (w_signs, scales) = split_channel_signs(
        conv.weights,
        out_c,
        ChannelLayout::OutMajor {
            per_channel: conv.in_channels * conv.kernel_h * conv.kernel_w,
        },
        tensor,
    )?;
    let taps = conv.in_channels * conv.kernel_h * conv.kernel_w;
    let (t, guard) = fold_thresholds(&scales, conv.bias, affine, out_c, site, taps as f64)?;
    let z_len = conv_h * conv_w * out_c;
    let out_len = out_h * out_w * out_c;
    limit("flat", z_len, SIGN_SPACE_HARD_MAX_FLAT)?;
    limit("flat", out_len, SIGN_SPACE_HARD_MAX_FLAT)?;
    Ok((
        Stage {
            kind: StageKind::Conv {
                w: w_signs,
                out_c,
                in_c: conv.in_channels,
                kh: conv.kernel_h,
                kw: conv.kernel_w,
                in_w: w,
                conv_h,
                conv_w,
                pool,
                out_h,
                out_w,
            },
            t,
            guard,
            channels: out_c,
            z_len,
            out_len,
        },
        StageShape::Spatial {
            h: out_h,
            w: out_w,
            c: out_c,
        },
    ))
}

/// Normalize one binary dense stage.
fn admit_dense_stage(
    weights: &[f32],
    in_dim: usize,
    out_dim: usize,
    affine: Option<&SignSpaceAffine<'_>>,
    shape: StageShape,
    site: usize,
    tensor: &'static str,
) -> Result<(Stage, StageShape), SignSpaceRefusal> {
    if in_dim == 0 || out_dim == 0 {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!("{tensor} is {in_dim}x{out_dim}"),
        });
    }
    if shape.len() != in_dim {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "{tensor} takes {in_dim} inputs, the previous stage produces {}",
                shape.len()
            ),
        });
    }
    if weights.len() != in_dim * out_dim {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "{tensor} has {} entries, geometry needs {}",
                weights.len(),
                in_dim * out_dim
            ),
        });
    }
    let (w_signs, scales) = split_channel_signs(
        weights,
        out_dim,
        ChannelLayout::InMajor {
            out_channels: out_dim,
        },
        tensor,
    )?;
    let (t, guard) = fold_thresholds(&scales, None, affine, out_dim, site, in_dim as f64)?;
    limit("flat", out_dim, SIGN_SPACE_HARD_MAX_FLAT)?;
    Ok((
        Stage {
            kind: StageKind::Dense {
                w: w_signs,
                in_dim,
                out_dim,
            },
            t,
            guard,
            channels: out_dim,
            z_len: out_dim,
            out_len: out_dim,
        },
        StageShape::Flat { len: out_dim },
    ))
}

fn admit<'a>(
    request: &SignSpaceRequest<'a>,
    limits: &SignSpaceLimits,
) -> Result<Admitted<'a>, SignSpaceRefusal> {
    // --- activations first: without B(v) = +1 iff v >= 0 nothing else matters.
    check_activation(&request.activation1, 1)?;
    check_activation(&request.activation2, 2)?;
    limit("stages", request.stages.len(), SIGN_SPACE_HARD_MAX_STAGES)?;
    for (index, stage) in request.stages.iter().enumerate() {
        let activation = match stage {
            BinaryStage::Conv { activation, .. } | BinaryStage::Dense { activation, .. } => {
                activation
            }
        };
        check_activation(activation, 3 + index)?;
    }

    // --- conv1: the ONLY layer touching the real box, and the one the `f32`
    // replay bound is derived for. Exactly +/-1 and bias-free, no exceptions.
    check_conv_geometry(&request.conv1, "conv1")?;
    check_unit_weights(request.conv1.weights, "conv1")?;
    if request.conv1.bias.is_some() {
        return Err(SignSpaceRefusal::UnsupportedConvGeometry {
            tensor: "conv1",
            detail: "a first-layer bias is outside the derived f32 replay bound".to_string(),
        });
    }
    // The final dense must be exactly +/-1: a per-CLASS scale would reorder the
    // logits, so it cannot be folded into anything.
    check_unit_weights(request.dense, "dense")?;

    let InputGeometry {
        height,
        width,
        channels,
    } = request.input;
    if height == 0 || width == 0 || channels == 0 {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!("degenerate input geometry {height}x{width}x{channels}"),
        });
    }
    if request.conv1.in_channels != channels {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "conv1 in_channels {} != input channels {channels}",
                request.conv1.in_channels
            ),
        });
    }
    if height < request.conv1.kernel_h || width < request.conv1.kernel_w {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: "conv1 kernel exceeds the input".to_string(),
        });
    }
    let h1c = height - request.conv1.kernel_h + 1;
    let w1c = width - request.conv1.kernel_w + 1;
    let c1 = request.conv1.out_channels;
    let pool1 = check_pool(request.conv1_pool, 1)?;
    let (Some(h1), Some(w1)) = (
        pool_out_dim(h1c, pool1.kernel_h, pool1.stride_h),
        pool_out_dim(w1c, pool1.kernel_w, pool1.stride_w),
    ) else {
        return Err(SignSpaceRefusal::UnsupportedPoolGeometry {
            site: 1,
            detail: format!("{pool1:?} does not fit a {h1c}x{w1c} feature map"),
        });
    };

    let n_pixels = height * width * channels;
    let n_raw1 = h1c * w1c * c1;
    let n_units1 = h1 * w1 * c1;
    limit("units", n_raw1, SIGN_SPACE_HARD_MAX_UNITS)?;
    limit("units", n_units1, SIGN_SPACE_HARD_MAX_UNITS)?;

    if request.num_classes == 0 {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: "num_classes == 0".to_string(),
        });
    }

    // --- the binary stage chain: conv2 first, then whatever the caller found.
    let mut shape = StageShape::Spatial {
        h: h1,
        w: w1,
        c: c1,
    };
    let mut stages: Vec<Stage> = Vec::with_capacity(1 + request.stages.len());
    let (stage, next) = admit_conv_stage(
        &request.conv2,
        request.conv2_pool,
        request.conv2_affine.as_ref(),
        shape,
        2,
        "conv2",
    )?;
    stages.push(stage);
    shape = next;
    for (index, spec) in request.stages.iter().enumerate() {
        let site = 3 + index;
        let (stage, next) = match *spec {
            BinaryStage::Conv {
                conv,
                pool,
                affine,
                activation: _,
            } => admit_conv_stage(&conv, pool, affine.as_ref(), shape, site, "stage conv")?,
            BinaryStage::Dense {
                weights,
                in_dim,
                out_dim,
                affine,
                activation: _,
            } => admit_dense_stage(
                weights,
                in_dim,
                out_dim,
                affine.as_ref(),
                shape,
                site,
                "stage dense",
            )?,
        };
        stages.push(stage);
        shape = next;
    }
    let n_flat = shape.len();
    limit("flat", n_flat, SIGN_SPACE_HARD_MAX_FLAT)?;
    if request.dense.len() != n_flat * request.num_classes {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "dense has {} entries, geometry needs {}x{}",
                request.dense.len(),
                n_flat,
                request.num_classes
            ),
        });
    }

    // --- property: the 42-singleton argmax complement, checked not assumed.
    if request.target_class >= request.num_classes {
        return Err(SignSpaceRefusal::PropertyNotArgmaxComplement {
            detail: format!(
                "target {} out of range for {} classes",
                request.target_class, request.num_classes
            ),
        });
    }
    let mut seen = vec![false; request.num_classes];
    for &c in request.challengers {
        if c >= request.num_classes {
            return Err(SignSpaceRefusal::PropertyNotArgmaxComplement {
                detail: format!("challenger {c} out of range"),
            });
        }
        if c == request.target_class {
            return Err(SignSpaceRefusal::PropertyNotArgmaxComplement {
                detail: "challenger equals the target".to_string(),
            });
        }
        if std::mem::replace(&mut seen[c], true) {
            return Err(SignSpaceRefusal::PropertyNotArgmaxComplement {
                detail: format!("challenger {c} repeated"),
            });
        }
    }
    if request.challengers.len() + 1 != request.num_classes {
        return Err(SignSpaceRefusal::PropertyNotArgmaxComplement {
            detail: format!(
                "{} challengers is not the complement of 1 target in {} classes",
                request.challengers.len(),
                request.num_classes
            ),
        });
    }

    // --- box: read as given, never reconstructed.
    if request.lo.len() != n_pixels || request.hi.len() != n_pixels {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "box has {}/{} entries, geometry needs {n_pixels}",
                request.lo.len(),
                request.hi.len()
            ),
        });
    }
    let mut max_abs = 0.0f64;
    for index in 0..n_pixels {
        let (lo, hi) = (request.lo[index], request.hi[index]);
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(SignSpaceRefusal::DegenerateBox { index, lo, hi });
        }
        max_abs = max_abs.max(lo.abs()).max(hi.abs());
    }
    if let Some(reference) = request.reference_input {
        if reference.len() != n_pixels {
            return Err(SignSpaceRefusal::ShapeMismatch {
                detail: format!(
                    "reference_input has {} entries, geometry needs {n_pixels}",
                    reference.len()
                ),
            });
        }
    }

    // --- exact-accumulation precondition, asserted rather than assumed.
    let taps1 = request.conv1.kernel_h * request.conv1.kernel_w * request.conv1.in_channels;
    if taps1 as f64 * max_abs >= SIGN_SPACE_EXACT_ACCUMULATION_LIMIT {
        return Err(SignSpaceRefusal::AccumulationNotExact {
            taps: taps1,
            max_abs,
        });
    }

    // --- tolerance floor for THIS geometry.
    let floor = f32_replay_slack_floor(taps1, max_abs);
    if !(limits.tolerance.is_finite() && limits.tolerance > 0.0) || limits.tolerance < floor {
        return Err(SignSpaceRefusal::ToleranceBelowFloor {
            requested: limits.tolerance,
            floor,
        });
    }

    // --- folded first-layer thresholds (scale > 0 proven per channel).
    let (t1, guard1) = fold_thresholds(
        &vec![1.0; c1],
        None,
        request.conv1_affine.as_ref(),
        c1,
        1,
        taps1 as f64 * max_abs,
    )?;

    let conv1_w: Vec<i8> = request
        .conv1
        .weights
        .iter()
        .map(|&v| if v > 0.0f32 { 1i8 } else { -1i8 })
        .collect();

    let admitted = Admitted {
        input: request.input,
        conv1_w,
        conv1: request.conv1,
        h1c,
        w1c,
        pool1,
        h1,
        w1,
        c1,
        t1,
        guard1,
        stages,
        dense: request.dense,
        num_classes: request.num_classes,
        lo: request.lo,
        hi: request.hi,
        target_class: request.target_class,
        challengers: request.challengers.to_vec(),
        n_pixels,
        n_raw1,
        n_units1,
        n_flat,
        taps1,
        max_abs,
    };

    // --- patch-index distinctness: the precondition making the interval
    // prepass EXACT. Recomputed, never inferred from a kernel shape, because a
    // dilated/grouped/negatively-padded conv can break it silently. Every RAW
    // conv1 position is checked, including ones no pooling window reads.
    let mut patch = Vec::with_capacity(taps1);
    let mut seen_indices = HashSet::with_capacity(taps1);
    for r in 0..h1c {
        for c in 0..w1c {
            admitted.conv1_patch(r, c, 0, &mut patch);
            seen_indices.clear();
            for &(p, _) in &patch {
                if p >= n_pixels {
                    return Err(SignSpaceRefusal::ShapeMismatch {
                        detail: format!("conv1 patch index {p} exceeds {n_pixels} pixels"),
                    });
                }
                if !seen_indices.insert(p) {
                    return Err(SignSpaceRefusal::PatchIndicesNotDistinct {
                        position: r * w1c + c,
                    });
                }
            }
        }
    }

    // --- declared limits, before any large allocation.
    limit("max_lp_columns", n_pixels + 1, limits.max_lp_columns)?;
    limit(
        "max_lp_columns",
        n_pixels + 1,
        SIGN_SPACE_HARD_MAX_LP_COLUMNS,
    )?;
    limit(
        "max_lp_rows",
        limits.max_lp_rows,
        SIGN_SPACE_HARD_MAX_LP_ROWS,
    )?;
    limit(
        "max_lp_solves",
        limits.max_lp_solves,
        SIGN_SPACE_HARD_MAX_LP_SOLVES,
    )?;
    limit(
        "max_free_units",
        limits.max_free_units,
        SIGN_SPACE_HARD_MAX_FREE_UNITS,
    )?;
    limit(
        "max_flip_batch",
        limits.max_flip_batch,
        SIGN_SPACE_HARD_MAX_FREE_UNITS,
    )?;
    if limits.max_wall_time > SIGN_SPACE_HARD_MAX_WALL_TIME {
        return Err(SignSpaceRefusal::LimitExceeded {
            limit: "max_wall_time",
            requested: limits.max_wall_time.as_millis(),
            cap: SIGN_SPACE_HARD_MAX_WALL_TIME.as_millis(),
        });
    }
    if limits.candidate_head == 0 {
        return Err(SignSpaceRefusal::LimitExceeded {
            limit: "candidate_head",
            requested: 0,
            cap: 1,
        });
    }

    Ok(admitted)
}

// ---------------------------------------------------------------------------
// STEP 1: the exact FIXED/FREE prepass
// ---------------------------------------------------------------------------

/// Per-unit interval bounds and phase from the EXACT prepass.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitClassification {
    /// `L_k` for each POOLED first-layer unit. Exact with no pool; a sound
    /// lower bound (`max_w L_w`) under pooling.
    pub lower: Vec<f64>,
    /// `U_k` for each POOLED first-layer unit. Exact in both cases.
    pub upper: Vec<f64>,
    /// Phase of each pooled first-layer unit.
    pub phase: Vec<UnitPhase>,
    /// Indices of the FREE units, ascending.
    pub free: Vec<usize>,
}

impl Admitted<'_> {
    /// `(L, U)` per RAW conv1 unit, exact by patch-index distinctness.
    fn raw_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let mut lower = vec![0.0; self.n_raw1];
        let mut upper = vec![0.0; self.n_raw1];
        let mut patch = Vec::with_capacity(self.taps1);
        for r in 0..self.h1c {
            for c in 0..self.w1c {
                for ch in 0..self.c1 {
                    self.conv1_patch(r, c, ch, &mut patch);
                    let mut l = 0.0f64;
                    let mut u = 0.0f64;
                    for &(p, w) in &patch {
                        if w > 0.0 {
                            l += self.lo[p];
                            u += self.hi[p];
                        } else {
                            l -= self.hi[p];
                            u -= self.lo[p];
                        }
                    }
                    let k = self.raw_index(r, c, ch);
                    lower[k] = l;
                    upper[k] = u;
                }
            }
        }
        (lower, upper)
    }

    fn classify(&self) -> UnitClassification {
        let (raw_lower, raw_upper) = self.raw_bounds();
        let mut lower = vec![0.0; self.n_units1];
        let mut upper = vec![0.0; self.n_units1];
        let mut phase = vec![UnitPhase::Free; self.n_units1];
        let mut free = Vec::new();
        for r in 0..self.h1 {
            for c in 0..self.w1 {
                for ch in 0..self.c1 {
                    // The pooled value is `max_w z_w`, so its supremum is
                    // `max_w U_w` (exact) and `max_w L_w` is a sound lower
                    // bound on its infimum.
                    let mut l = f64::NEG_INFINITY;
                    let mut u = f64::NEG_INFINITY;
                    for (mr, mc) in self.pool1_members(r, c) {
                        let m = self.raw_index(mr, mc, ch);
                        l = l.max(raw_lower[m]);
                        u = u.max(raw_upper[m]);
                    }
                    let t = self.t1[ch];
                    let k = self.unit_index(r, c, ch);
                    lower[k] = l;
                    upper[k] = u;
                    // NON-strict for +1 (B fires at exactly zero) and STRICT
                    // for -1. Swapping either is a direct correctness bug.
                    phase[k] = if l >= t {
                        UnitPhase::FixedPositive
                    } else if u < t {
                        UnitPhase::FixedNegative
                    } else {
                        free.push(k);
                        UnitPhase::Free
                    };
                }
            }
        }
        UnitClassification {
            lower,
            upper,
            phase,
            free,
        }
    }
}

// ---------------------------------------------------------------------------
// STEP 2: the incremental logit engine (exact integers throughout)
// ---------------------------------------------------------------------------

/// Downstream state of one first-layer sign pattern.
///
/// Every stage accumulator, every stage sign and the logits are EXACT
/// integers: `s1 in {+/-1}` and every weight is unit-signed, so no
/// floating-point error can enter the search.
struct Engine {
    /// Pooled first-layer signs — the search variables.
    s0: Vec<i8>,
    /// Pre-pool accumulators, per stage.
    z: Vec<Vec<i32>>,
    /// Post-pool accumulators, per stage.
    p: Vec<Vec<i32>>,
    /// Output signs, per stage.
    s: Vec<Vec<i8>>,
    logits: Vec<i64>,
    // Scratch for the incremental delta path; never read across calls.
    zstamp: Vec<Vec<u64>>,
    ostamp: Vec<Vec<u64>>,
    generation: u64,
    touched: Vec<usize>,
    changes: Vec<(u32, i8)>,
    next_changes: Vec<(u32, i8)>,
}

impl Admitted<'_> {
    /// RAW `z1` at a concrete input, in `f64` (exact under the admitted
    /// accumulation bound).
    fn z1_at(&self, x: &[f64]) -> Vec<f64> {
        let mut z1 = vec![0.0; self.n_raw1];
        let mut patch = Vec::with_capacity(self.taps1);
        for r in 0..self.h1c {
            for c in 0..self.w1c {
                for ch in 0..self.c1 {
                    self.conv1_patch(r, c, ch, &mut patch);
                    let mut acc = 0.0;
                    for &(p, w) in &patch {
                        acc += w * x[p];
                    }
                    z1[self.raw_index(r, c, ch)] = acc;
                }
            }
        }
        z1
    }

    /// The pooled value `max_w z_w` of unit `k`, and the member attaining it
    /// (lowest raw index on a tie, so the choice is deterministic).
    fn pooled_at(&self, k: usize, z1: &[f64]) -> (f64, usize) {
        let (r, c, ch) = self.unit_coords(k);
        let mut best = f64::NEG_INFINITY;
        let mut best_member = self.raw_index(r * self.pool1.stride_h, c * self.pool1.stride_w, ch);
        for (mr, mc) in self.pool1_members(r, c) {
            let m = self.raw_index(mr, mc, ch);
            if z1[m] > best {
                best = z1[m];
                best_member = m;
            }
        }
        (best, best_member)
    }

    /// The first-layer sign pattern at a concrete input.
    /// `B(v) = +1 iff max_w z_w >= t` — the OR identity.
    fn s1_at(&self, x: &[f64]) -> Vec<i8> {
        let z1 = self.z1_at(x);
        self.s1_from_z1(&z1)
    }

    fn s1_from_z1(&self, z1: &[f64]) -> Vec<i8> {
        (0..self.n_units1)
            .map(|k| {
                let ch = k % self.c1;
                let (pooled, _) = self.pooled_at(k, z1);
                if pooled >= self.t1[ch] {
                    1i8
                } else {
                    -1i8
                }
            })
            .collect()
    }

    /// A from-scratch downstream forward: the INDEPENDENT recompute the
    /// incremental delta path is checked against, and the only arithmetic
    /// trusted when a candidate is finalized.
    fn forward_from_pattern(&self, s0: &[i8]) -> Engine {
        let mut z: Vec<Vec<i32>> = Vec::with_capacity(self.stages.len());
        let mut p: Vec<Vec<i32>> = Vec::with_capacity(self.stages.len());
        let mut s: Vec<Vec<i8>> = Vec::with_capacity(self.stages.len());
        let mut input: &[i8] = s0;
        for stage in &self.stages {
            let mut zs = vec![0i32; stage.z_len];
            stage.compute_z(input, &mut zs);
            let mut ps = vec![0i32; stage.out_len];
            let mut ss = vec![0i8; stage.out_len];
            stage.pool_and_sign(&zs, &mut ps, &mut ss);
            z.push(zs);
            p.push(ps);
            s.push(ss);
            input = s.last().expect("just pushed");
        }
        let mut logits = vec![0i64; self.num_classes];
        let last = s.last().expect("at least one stage");
        for j in 0..self.n_flat {
            let value = i64::from(last[j]);
            if value == 0 {
                continue;
            }
            for (class, logit) in logits.iter_mut().enumerate() {
                *logit += value * self.dense_weight(j, class);
            }
        }
        let zstamp = self.stages.iter().map(|s| vec![0u64; s.z_len]).collect();
        let ostamp = self.stages.iter().map(|s| vec![0u64; s.out_len]).collect();
        Engine {
            s0: s0.to_vec(),
            z,
            p,
            s,
            logits,
            zstamp,
            ostamp,
            generation: 0,
            touched: Vec::new(),
            changes: Vec::new(),
            next_changes: Vec::new(),
        }
    }

    /// Apply one first-layer flip incrementally, propagating the delta through
    /// every binary stage and into the logits. Applying it twice is the exact
    /// undo: all arithmetic is integer and every pooled value is recomputed
    /// from its window rather than tracked differentially.
    fn flip(&self, engine: &mut Engine, u: usize) {
        engine.s0[u] = -engine.s0[u];
        let mut cur = std::mem::take(&mut engine.changes);
        let mut next = std::mem::take(&mut engine.next_changes);
        let mut touched = std::mem::take(&mut engine.touched);
        cur.clear();
        cur.push((u as u32, 2 * engine.s0[u]));

        for (si, stage) in self.stages.iter().enumerate() {
            if cur.is_empty() {
                break;
            }
            engine.generation += 1;
            let generation = engine.generation;
            touched.clear();
            stage.scatter(
                &cur,
                &mut engine.z[si],
                &mut engine.zstamp[si],
                generation,
                &mut touched,
            );
            next.clear();
            for &zi in &touched {
                stage.refresh(
                    zi,
                    &engine.z[si],
                    &mut engine.p[si],
                    &mut engine.s[si],
                    &mut engine.ostamp[si],
                    generation,
                    &mut next,
                );
            }
            std::mem::swap(&mut cur, &mut next);
        }

        for &(j, delta) in &cur {
            let j = j as usize;
            let delta = i64::from(delta);
            for (class, logit) in engine.logits.iter_mut().enumerate() {
                *logit += delta * self.dense_weight(j, class);
            }
        }

        engine.changes = cur;
        engine.next_changes = next;
        engine.touched = touched;
    }

    /// `max_c (logit[c] - logit[target])` over the challenger set, with the
    /// attaining challenger. PRE-SOFTMAX by construction.
    fn margin(&self, logits: &[i64]) -> (i64, usize) {
        let target = logits[self.target_class];
        let mut best = i64::MIN;
        let mut best_class = self.challengers[0];
        for &c in &self.challengers {
            let m = logits[c] - target;
            if m > best {
                best = m;
                best_class = c;
            }
        }
        (best, best_class)
    }

    fn argmax(&self, logits: &[i64]) -> usize {
        let mut best = 0usize;
        for c in 1..self.num_classes {
            if logits[c] > logits[best] {
                best = c;
            }
        }
        best
    }
}

impl Stage {
    /// From-scratch accumulators over a `+/-1` input.
    fn compute_z(&self, input: &[i8], z: &mut [i32]) {
        match &self.kind {
            StageKind::Conv {
                w,
                out_c,
                in_c,
                kh,
                kw,
                in_w,
                conv_h,
                conv_w,
                ..
            } => {
                for i in 0..*conv_h {
                    for j in 0..*conv_w {
                        for co in 0..*out_c {
                            let mut acc = 0i32;
                            for kr in 0..*kh {
                                for kc in 0..*kw {
                                    for ci in 0..*in_c {
                                        let weight = w[((co * in_c + ci) * kh + kr) * kw + kc];
                                        let idx = ((i + kr) * in_w + (j + kc)) * in_c + ci;
                                        acc += i32::from(weight) * i32::from(input[idx]);
                                    }
                                }
                            }
                            z[(i * conv_w + j) * out_c + co] = acc;
                        }
                    }
                }
            }
            StageKind::Dense { w, in_dim, out_dim } => {
                z[..*out_dim].fill(0);
                for i in 0..*in_dim {
                    let value = i32::from(input[i]);
                    if value == 0 {
                        continue;
                    }
                    let row = &w[i * out_dim..(i + 1) * out_dim];
                    for (o, &weight) in row.iter().enumerate() {
                        z[o] += i32::from(weight) * value;
                    }
                }
            }
        }
    }

    /// From-scratch pooling and thresholding of the accumulators.
    fn pool_and_sign(&self, z: &[i32], p: &mut [i32], s: &mut [i8]) {
        for oi in 0..self.out_len {
            let pooled = self.pooled(oi, z);
            p[oi] = pooled;
            s[oi] = if f64::from(pooled) >= self.t[oi % self.channels] {
                1
            } else {
                -1
            };
        }
    }

    /// `max` over the pooling window feeding output index `oi`.
    fn pooled(&self, oi: usize, z: &[i32]) -> i32 {
        match &self.kind {
            StageKind::Conv {
                out_c,
                conv_w,
                pool,
                out_w,
                ..
            } => {
                let co = oi % out_c;
                let spatial = oi / out_c;
                let or = spatial / out_w;
                let oc = spatial % out_w;
                let mut best = i32::MIN;
                for a in 0..pool.kernel_h {
                    for b in 0..pool.kernel_w {
                        let i = or * pool.stride_h + a;
                        let j = oc * pool.stride_w + b;
                        best = best.max(z[(i * conv_w + j) * out_c + co]);
                    }
                }
                best
            }
            StageKind::Dense { .. } => z[oi],
        }
    }

    /// Add `delta` to every accumulator each changed input index feeds,
    /// recording the touched accumulator indices exactly once.
    fn scatter(
        &self,
        changes: &[(u32, i8)],
        z: &mut [i32],
        stamp: &mut [u64],
        generation: u64,
        touched: &mut Vec<usize>,
    ) {
        match &self.kind {
            StageKind::Conv {
                w,
                out_c,
                in_c,
                kh,
                kw,
                in_w,
                conv_h,
                conv_w,
                ..
            } => {
                for &(idx, delta) in changes {
                    let idx = idx as usize;
                    let delta = i32::from(delta);
                    let ci = idx % in_c;
                    let spatial = idx / in_c;
                    let r = spatial / in_w;
                    let c = spatial % in_w;
                    for kr in 0..*kh {
                        let Some(i) = r.checked_sub(kr) else { continue };
                        if i >= *conv_h {
                            continue;
                        }
                        for kc in 0..*kw {
                            let Some(j) = c.checked_sub(kc) else { continue };
                            if j >= *conv_w {
                                continue;
                            }
                            for co in 0..*out_c {
                                let weight = w[((co * in_c + ci) * kh + kr) * kw + kc];
                                let zi = (i * conv_w + j) * out_c + co;
                                z[zi] += i32::from(weight) * delta;
                                if stamp[zi] != generation {
                                    stamp[zi] = generation;
                                    touched.push(zi);
                                }
                            }
                        }
                    }
                }
            }
            StageKind::Dense { w, out_dim, .. } => {
                for &(idx, delta) in changes {
                    let idx = idx as usize;
                    let delta = i32::from(delta);
                    let row = &w[idx * out_dim..(idx + 1) * out_dim];
                    for (o, &weight) in row.iter().enumerate() {
                        z[o] += i32::from(weight) * delta;
                        if stamp[o] != generation {
                            stamp[o] = generation;
                            touched.push(o);
                        }
                    }
                }
            }
        }
    }

    /// Recompute the pooled value and sign of every output the touched
    /// accumulator `zi` feeds, recording the sign deltas for the next stage.
    fn refresh(
        &self,
        zi: usize,
        z: &[i32],
        p: &mut [i32],
        s: &mut [i8],
        stamp: &mut [u64],
        generation: u64,
        out: &mut Vec<(u32, i8)>,
    ) {
        match &self.kind {
            StageKind::Conv {
                out_c,
                conv_w,
                pool,
                out_h,
                out_w,
                ..
            } => {
                let co = zi % out_c;
                let spatial = zi / out_c;
                let i = spatial / conv_w;
                let j = spatial % conv_w;
                for or in pool_parents(i, pool.kernel_h, pool.stride_h, *out_h) {
                    for oc in pool_parents(j, pool.kernel_w, pool.stride_w, *out_w) {
                        let oi = (or * out_w + oc) * out_c + co;
                        if stamp[oi] == generation {
                            continue;
                        }
                        stamp[oi] = generation;
                        self.settle(oi, z, p, s, out);
                    }
                }
            }
            StageKind::Dense { .. } => {
                if stamp[zi] == generation {
                    return;
                }
                stamp[zi] = generation;
                self.settle(zi, z, p, s, out);
            }
        }
    }

    fn settle(&self, oi: usize, z: &[i32], p: &mut [i32], s: &mut [i8], out: &mut Vec<(u32, i8)>) {
        let pooled = self.pooled(oi, z);
        p[oi] = pooled;
        let after = if f64::from(pooled) >= self.t[oi % self.channels] {
            1i8
        } else {
            -1i8
        };
        let before = s[oi];
        if after != before {
            s[oi] = after;
            out.push((oi as u32, after - before));
        }
    }

    /// Does any output of this stage sit inside ITS OWN channel's folded-affine
    /// rounding bound, i.e. close enough to its threshold that an `f32` runtime
    /// could land on the other side?
    ///
    /// A zero guard means the decision is exact in `f32`, so a zero margin
    /// there is the legitimate `B(0) = +1` boundary and not a hazard.
    fn violates_replay_guard(&self, p: &[i32]) -> bool {
        (0..self.out_len).any(|oi| {
            let channel = oi % self.channels;
            let guard = self.guard[channel];
            guard > 0.0 && (f64::from(p[oi]) - self.t[channel]).abs() <= guard
        })
    }
}

// ---------------------------------------------------------------------------
// STEP 3: the realizability LP + primal extraction
// ---------------------------------------------------------------------------

/// The assembled realizability LP: the IR problem, the pixel columns it
/// actually declares, and the slack column.
struct RealizabilityLp {
    problem: MilpProblem,
    /// `(flat pixel index, column)` for every pixel some active row reads.
    pixels: Vec<(usize, Col)>,
    slack: Col,
}

/// What one realizability LP produced.
enum Realizability {
    /// A concrete in-box `x` realizing the WHOLE pattern with this slack.
    Realizable { slack: f64, x: Vec<f64> },
    /// No `x` realizes the pattern at the required slack, or the solve was
    /// inconclusive. Conservative: only ever loses a SAT.
    Rejected,
}

/// The evolving L-infinity radius of the trust region inside ONE realizability
/// call.
///
/// THE INVARIANT THIS TYPE EXISTS FOR: `radius == None` means the FULL vnnlib
/// box, i.e. exactly the shipped LP, and it is the ONLY state from which an
/// inconclusive LP is allowed to decline the pattern. Every other state must
/// [`Self::expand`] instead, and expansion always terminates at `None` —
/// shrinking the box can only make the LP more constrained, so infeasibility
/// under a region proves nothing about the box.
#[derive(Debug, Clone, Copy)]
struct TrustState {
    /// Current radius; `None` is the full box.
    radius: Option<f64>,
    /// The radius at or above which the region covers the whole box.
    full: f64,
    /// Largest radius already known to yield no point (bisection bracket).
    last_failed: Option<f64>,
    /// Bisection steps to spend once a radius works.
    refine: usize,
    /// Expansions so far, against `limits.max_trust_expansions`.
    expansions: usize,
}

impl TrustState {
    /// Arm the region declared by `limits` around `anchor`, or fall back to the
    /// full box when the declaration is `FullBox`, the geometry is degenerate,
    /// or the anchor is not exactly in the box.
    fn new(admitted: &Admitted<'_>, limits: &SignSpaceLimits, anchor: &[f64]) -> Self {
        let full = admitted.box_half_width();
        let radius = limits
            .trust_region
            .initial_fraction()
            .filter(|f| f.is_finite() && *f > 0.0 && *f < 1.0)
            .filter(|_| full.is_finite() && full > 0.0)
            .filter(|_| admitted.trust_anchor_is_in_box(anchor))
            .map(|f| (full * f).max(f64::MIN_POSITIVE));
        Self {
            radius,
            full,
            last_failed: None,
            refine: limits.trust_region.refine(),
            expansions: 0,
        }
    }

    /// Take the next radius after an inconclusive LP.
    ///
    /// Returns `false` when the region was ALREADY the full box, which is the
    /// one and only state in which the caller may decline. Otherwise the radius
    /// doubles — and any doubling that reaches the box, or exhausts the
    /// expansion budget, becomes the full box itself, so the last LP tried is
    /// always the shipped one.
    fn expand(&mut self, limits: &SignSpaceLimits) -> bool {
        let Some(radius) = self.radius else {
            return false;
        };
        self.last_failed = Some(radius);
        self.expansions += 1;
        let doubled = radius * 2.0;
        self.radius = if doubled.is_finite()
            && doubled < self.full
            && self.expansions < limits.max_trust_expansions
        {
            Some(doubled)
        } else {
            None
        };
        true
    }
}

// ---------------------------------------------------------------------------
// Closed-form segment crossings (the MINIMAL-move lever)
//
// `z1` is LINEAR in `x`, so on the segment `x(theta) = x0 + theta*(x_LP - x0)`
// every RAW conv1 accumulator is the linear form
//
//     z1_m(theta) = a_m + theta * (b_m - a_m),    a_m = z1_m(x0), b_m = z1_m(x_LP)
//
// with BOTH endpoint arrays already computed. So every threshold crossing is a
// division, not a bisection, and the whole set of units that flip on the way is
// known in one pass.
//
// A POOLED unit is `max_w z1_w`, a max of linear functions: piecewise linear
// and CONVEX in theta, NOT linear. The closed form therefore applies per WINDOW
// MEMBER, and the two directions are asymmetric exactly as the OR identity says
// (`B(max_w z_w) = +1 iff SOME member is up`, `= -1 iff EVERY member is down`).
// ---------------------------------------------------------------------------

/// Smallest `theta` in `[0, 1]` from which `a + theta*(b - a) >= c` holds for
/// the whole of `[theta, 1]`, or `None` when no such `theta` exists.
///
/// The form is linear, so "holds at `theta`" and "holds from `theta` on" differ
/// only when it is FALLING — and a falling form that is below `c` at `b` never
/// holds at `1`, which the `b < c` arm rejects first. Closed form throughout.
/// `NY_BNN_SIGN_SPACE_TRACE` — dark, print-only tracing of lazy row generation
/// (declared as `ny_levers::decls::dark_probes::BNN_SIGN_SPACE_TRACE`).
///
/// PRESENCE gate, the same shape as `NY_MIP_TRACE`: ANY present value arms it,
/// including `"0"`, the empty string and non-UTF-8, which is why the lookup
/// goes through `var_os` plus a lossy conversion rather than a `Bool` parse.
/// Read ONCE per realizability call rather than per round — the loop it guards
/// is the hot one, and re-reading the environment inside it would be the
/// measurement perturbing the thing measured.
fn sign_space_trace_armed() -> bool {
    !matches!(
        ny_levers::read_with(
            &ny_levers::decls::dark_probes::BNN_SIGN_SPACE_TRACE,
            |name| { std::env::var_os(name).map(|value| value.to_string_lossy().into_owned()) }
        )
        .value,
        ny_levers::LeverValue::Unset
    )
}

fn segment_cross_up(a: f64, b: f64, c: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return None;
    }
    if b < c {
        // Below the level at the far endpoint: no suffix of the segment works.
        return None;
    }
    if a >= c {
        // Already there, and `b >= c`, so the whole segment is there.
        return Some(0.0);
    }
    // `a < c <= b`, hence `b - a > 0` and the crossing is in `(0, 1]`.
    let theta = (c - a) / (b - a);
    theta.is_finite().then(|| theta.clamp(0.0, 1.0))
}

/// Smallest `theta` in `[0, 1]` from which `a + theta*(b - a) <= c` holds for
/// the whole of `[theta, 1]`, or `None` when no such `theta` exists.
fn segment_cross_down(a: f64, b: f64, c: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return None;
    }
    if b > c {
        return None;
    }
    if a <= c {
        return Some(0.0);
    }
    // `a > c >= b`, hence `b - a < 0` and the crossing is in `(0, 1]`.
    let theta = (c - a) / (b - a);
    theta.is_finite().then(|| theta.clamp(0.0, 1.0))
}

impl Admitted<'_> {
    /// `s1_k * (max_w (w_k . x) - t_k)`: how much slack unit `k` has at `x`,
    /// under the OR identity.
    ///
    /// `+1` needs the max above the threshold (ONE member suffices); `-1`
    /// needs the max below it (which is exactly "all members below").
    fn unit_slack(&self, k: usize, s1: &[i8], z1: &[f64]) -> (f64, usize) {
        let ch = k % self.c1;
        let (pooled, member) = self.pooled_at(k, z1);
        (f64::from(s1[k]) * (pooled - self.t1[ch]), member)
    }

    /// The RAW conv1 rows one FREE unit contributes to the LP.
    ///
    /// * `s1_k = -1`: EVERY window member must fall below the threshold, so
    ///   every member gets a row. This is exact.
    /// * `s1_k = +1`: ONE member above the threshold suffices, and a
    ///   disjunction is not a linear constraint — so the incumbent point's
    ///   best member is pinned instead. That is a RESTRICTION of the true
    ///   condition, which is why an infeasible LP no longer proves the pattern
    ///   unrealizable; it only means "not realizable with these member
    ///   choices". Declining is safe (it loses a SAT, never invents one), and
    ///   with no pool the window is a singleton and the restriction is vacuous.
    fn unit_rows(&self, k: usize, s1: &[i8], z1: &[f64], out: &mut Vec<usize>) {
        let (r, c, ch) = self.unit_coords(k);
        if s1[k] >= 0 {
            let (_, member) = self.pooled_at(k, z1);
            out.push(member);
        } else {
            for (mr, mc) in self.pool1_members(r, c) {
                out.push(self.raw_index(mr, mc, ch));
            }
        }
    }

    /// Smallest `theta` in `[0, 1]` at which unit `k` holds its DESIRED sign
    /// with OR-slack `>= level` for the rest of the segment, or `None` when the
    /// segment never gets there.
    ///
    /// `a` is `z1` at the segment's start and `b` is `z1` at its end; both are
    /// RAW (pre-pool) arrays. The pooled value is a max of linear functions, so:
    ///
    /// * a `+1` unit needs `max_w z_w >= t + level`, i.e. SOME member up — the
    ///   crossing is the **MIN** over the members that actually cross;
    /// * a `-1` unit needs `max_w z_w <= t - level`, i.e. EVERY member down —
    ///   the crossing is the **MAX** over members, and one member that never
    ///   gets there kills the whole unit.
    ///
    /// With no pool the window is a singleton and both arms collapse to the one
    /// linear crossing, so the shallow path is bit-for-bit the un-pooled rule.
    fn unit_crossing_theta(
        &self,
        k: usize,
        sign: i8,
        level: f64,
        a: &[f64],
        b: &[f64],
    ) -> Option<f64> {
        let (r, c, ch) = self.unit_coords(k);
        let t = self.t1[ch];
        if sign >= 0 {
            let mut earliest: Option<f64> = None;
            for (mr, mc) in self.pool1_members(r, c) {
                let m = self.raw_index(mr, mc, ch);
                if let Some(theta) = segment_cross_up(a[m], b[m], t + level) {
                    earliest = Some(earliest.map_or(theta, |best: f64| best.min(theta)));
                }
            }
            earliest
        } else {
            let mut latest = 0.0f64;
            for (mr, mc) in self.pool1_members(r, c) {
                let m = self.raw_index(mr, mc, ch);
                latest = latest.max(segment_cross_down(a[m], b[m], t - level)?);
            }
            Some(latest)
        }
    }

    /// The MINIMAL `theta` carrying every deficient unit past its threshold.
    ///
    /// `deficient` is exactly the set of free units whose OR-slack at the
    /// segment's start is below the acceptance tolerance — the units the LP was
    /// asked to fix. Each one is satisfied from its own crossing onwards, so the
    /// segment's answer is the MAX of the individual crossings; every unit that
    /// is NOT deficient keeps its sign until its own (later) crossing, which is
    /// the whole point of not going all the way to `1`.
    ///
    /// # The nudge, and where its size comes from
    ///
    /// Landing exactly ON a crossing puts the unit at slack `== tolerance`,
    /// which the `f32` rounding of the point (`to_replay_bytes`) could undo:
    /// re-reading `x` as the `f32` bytes a runtime sees perturbs each
    /// `taps1`-term accumulator by up to `f32_replay_slack_floor(taps1,
    /// max_abs)` (the `f32` input roundings plus the `taps1 - 1` accumulations
    /// — the SAME bound the witness gate uses), and a folded first-layer affine
    /// adds its own per-channel `guard1`. So the target level is raised by
    /// exactly that worst-case budget rather than by a round number: the
    /// crossing is solved for `tolerance + floor + guard1[ch]`, which is the
    /// smallest level at which no `f32` replay of the chosen point can push the
    /// unit back under the tolerance.
    ///
    /// The raised level is a PREFERENCE, not a requirement, and it is CLAMPED
    /// to what the segment can actually deliver — the LP only promises
    /// `>= tolerance` at `theta == 1`, and the search is not entitled to more
    /// than the far endpoint has. The clamp is read from the SAME `b` array the
    /// crossing is solved against, less the module's own `f64` re-evaluation
    /// bound, and that detail is load-bearing rather than cosmetic: without it a
    /// level the endpoint attains to the last bit reads back as UNATTAINABLE,
    /// the unit forces `theta = 1`, and the whole lever silently collapses into
    /// the vertex jump it was written to replace. (Measured: it did — the first
    /// armed A/B on `model_48_idx_1703_eps_3` was byte-identical to the
    /// unarmed one, 22 LP solves and `-110` on both arms.)
    fn minimal_segment_theta(
        &self,
        s1: &[i8],
        deficient: &[usize],
        a: &[f64],
        b: &[f64],
        tolerance: f64,
    ) -> f64 {
        let floor = f32_replay_slack_floor(self.taps1, self.max_abs);
        let epsilon = f64_slack_verification_epsilon(self.taps1, self.max_abs);
        let mut theta = 0.0f64;
        for &k in deficient {
            let ch = k % self.c1;
            let wanted = tolerance + floor + self.guard1[ch];
            let attainable = self.unit_slack(k, s1, b).0 - epsilon;
            let crossing = self
                .unit_crossing_theta(k, s1[k], wanted.min(attainable), a, b)
                .unwrap_or(1.0);
            theta = theta.max(crossing);
            if theta >= 1.0 {
                return 1.0;
            }
        }
        theta.clamp(0.0, 1.0)
    }

    /// `x0 + theta*(x_lp - x0)`, ASSERTED into the input box.
    ///
    /// `x0` and `x_lp` are both in `[lo, hi]` and a product of intervals is
    /// convex, so the exact combination is in the box for every
    /// `theta in [0, 1]`. That is a proof about REALS: the combination is
    /// evaluated in `f64`, so membership is CHECKED against the vnnlib `lo`/`hi`
    /// directly — never inherited from the argument and never taken on a
    /// tolerance — and a coordinate that rounded outside is pinned back to the
    /// bound it crossed (a move of at most one ULP, and still a point of the
    /// box). Anything non-finite abandons the move entirely.
    fn blend_into_box(&self, x0: &[f64], x_lp: &[f64], theta: f64) -> Option<Vec<f64>> {
        if !theta.is_finite() || !(0.0..=1.0).contains(&theta) {
            return None;
        }
        if x0.len() != self.n_pixels || x_lp.len() != self.n_pixels {
            return None;
        }
        let mut out = Vec::with_capacity(self.n_pixels);
        for p in 0..self.n_pixels {
            let (lo, hi) = (self.lo[p], self.hi[p]);
            let value = theta.mul_add(x_lp[p] - x0[p], x0[p]);
            if !value.is_finite() {
                return None;
            }
            let value = if value < lo {
                lo
            } else if value > hi {
                hi
            } else {
                value
            };
            // Exact, no tolerance. `admit` already refused a non-finite or
            // inverted bound, so this can only fail on a NaN that slipped
            // through above — in which case the move is abandoned, not shipped.
            if !(value >= lo && value <= hi) {
                return None;
            }
            out.push(value);
        }
        Some(out)
    }

    /// `max s  s.t.  s1_k * (w_m . x - t_k) >= s  for the selected raw rows,
    /// x in [lo,hi], TOL <= s <= 1`.
    ///
    /// The slack floor is the acceptance tolerance, so infeasibility IS
    /// "not realizable at the required slack" and is proven by exact Farkas
    /// rather than computed as an optimum (measured ~12x cheaper).
    ///
    /// FIXED units get NO row: the exact prepass already proved every `x` in
    /// the box realizes their sign, which is the entire point of the prepass.
    /// Only pixels some selected row actually reads get a column.
    ///
    /// `trust` is the optional L-infinity trust region `(anchor, radius)`. It
    /// only ever INTERSECTS the box bounds — `[max(lo, a - r), min(hi, a + r)]`
    /// — so the feasible set can only shrink, and any point the LP returns is
    /// still an in-box point. See [`TrustRegion`].
    fn build_realizability_lp(
        &self,
        s1: &[i8],
        units: &[usize],
        z1: &[f64],
        limits: &SignSpaceLimits,
        trust: Option<(&[f64], f64)>,
    ) -> Result<RealizabilityLp, SignSpaceRefusal> {
        // `(raw conv1 position, +/-1 sign of the pooled unit that owns it)`.
        let mut rows: Vec<(usize, i8)> = Vec::with_capacity(units.len());
        let mut members: Vec<usize> = Vec::with_capacity(self.pool1.area());
        for &k in units {
            members.clear();
            self.unit_rows(k, s1, z1, &mut members);
            for &m in &members {
                rows.push((m, s1[k]));
            }
        }
        limit("max_lp_rows", rows.len(), limits.max_lp_rows)?;
        limit("max_lp_rows", rows.len(), SIGN_SPACE_HARD_MAX_LP_ROWS)?;
        let nonzeros = rows.len() * (self.taps1 + 1);
        limit("max_lp_nonzeros", nonzeros, limits.max_lp_nonzeros)?;
        limit("max_lp_nonzeros", nonzeros, SIGN_SPACE_HARD_MAX_LP_NONZEROS)?;

        let mut column_of: Vec<Option<Col>> = vec![None; self.n_pixels];
        let mut patch = Vec::with_capacity(self.taps1);
        let mut problem = MilpProblem::new();
        let mut pixels: Vec<(usize, Col)> = Vec::new();
        for &(m, _) in &rows {
            let (r, c, ch) = self.raw_coords(m);
            self.conv1_patch(r, c, ch, &mut patch);
            for &(p, _) in &patch {
                if column_of[p].is_none() {
                    // The BOX bounds, intersected with the trust region when one
                    // is armed. `trust_bounds` can only narrow `[lo, hi]`, never
                    // widen it, and never inverts it.
                    let (bound_lo, bound_hi) = self.trust_bounds(p, trust);
                    // `ColSpec::obj` is ignored by the lowering; the objective
                    // is passed at solve time via `LpSession::optimize`.
                    let col = problem.add_col(0.0, bound_lo, bound_hi);
                    column_of[p] = Some(col);
                    pixels.push((p, col));
                }
            }
        }
        limit("max_lp_columns", pixels.len() + 1, limits.max_lp_columns)?;
        limit(
            "max_lp_columns",
            pixels.len() + 1,
            SIGN_SPACE_HARD_MAX_LP_COLUMNS,
        )?;
        let slack = problem.add_col(0.0, limits.tolerance, 1.0);

        let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(self.taps1 + 1);
        for &(m, unit_sign) in &rows {
            let (r, c, ch) = self.raw_coords(m);
            self.conv1_patch(r, c, ch, &mut patch);
            let sign = f64::from(unit_sign);
            coeffs.clear();
            for &(p, w) in &patch {
                let col = column_of[p].expect("every active-row pixel has a column");
                coeffs.push((col, sign * w));
            }
            coeffs.push((slack, -1.0));
            // sign * (w . x) - s >= sign * t
            problem.add_row(sign * self.t1[ch], f64::INFINITY, coeffs.iter().copied());
        }
        Ok(RealizabilityLp {
            problem,
            pixels,
            slack,
        })
    }

    /// One pixel column's bounds under an optional trust region.
    ///
    /// THE ONE PLACE the region touches the LP, and it is a pure INTERSECTION
    /// with the vnnlib bounds:
    ///
    /// ```text
    ///     [max(lo_p, a_p - r),  min(hi_p, a_p + r)]  subset of  [lo_p, hi_p]
    /// ```
    ///
    /// The intersection is non-empty whenever the anchor is in the box, which
    /// the caller establishes once per call ([`Admitted::trust_anchor_is_in_box`])
    /// and which is why an inverted bound cannot be constructed here. Belt and
    /// braces anyway: a degenerate radius (non-finite, negative) or an anchor
    /// coordinate that somehow escaped falls back to the FULL box bounds for
    /// that pixel, which is the shipped behaviour and can only be more
    /// permissive.
    fn trust_bounds(&self, p: usize, trust: Option<(&[f64], f64)>) -> (f64, f64) {
        let (lo, hi) = (self.lo[p], self.hi[p]);
        let Some((anchor, radius)) = trust else {
            return (lo, hi);
        };
        if !(radius.is_finite() && radius >= 0.0) {
            return (lo, hi);
        }
        let Some(&a) = anchor.get(p) else {
            return (lo, hi);
        };
        if !(a.is_finite() && a >= lo && a <= hi) {
            return (lo, hi);
        }
        let bound_lo = lo.max(a - radius);
        let bound_hi = hi.min(a + radius);
        if bound_lo.is_finite() && bound_hi.is_finite() && bound_lo <= bound_hi {
            (bound_lo, bound_hi)
        } else {
            (lo, hi)
        }
    }

    /// Is `anchor` exactly inside the vnnlib box, coordinate by coordinate?
    ///
    /// Exact, no tolerance — the same rule as [`Admitted::blend_into_box`].
    /// A trust region is only armed around an anchor that passes this, so the
    /// intersection in [`Admitted::trust_bounds`] is provably non-empty.
    fn trust_anchor_is_in_box(&self, anchor: &[f64]) -> bool {
        anchor.len() == self.n_pixels
            && (0..self.n_pixels).all(|p| {
                anchor[p].is_finite() && anchor[p] >= self.lo[p] && anchor[p] <= self.hi[p]
            })
    }

    /// The widest `(hi - lo) / 2` over the box: the radius at which an
    /// L-infinity region around ANY in-box anchor already covers the box.
    fn box_half_width(&self) -> f64 {
        (0..self.n_pixels).fold(0.0f64, |acc, p| acc.max(self.hi[p] - self.lo[p])) * 0.5
    }

    /// Solve ONE active-set LP and return the completed primal, or `None` when
    /// the outcome carries no usable point.
    fn solve_active_lp(
        &self,
        s1: &[i8],
        units: &[usize],
        z1: &[f64],
        limits: &SignSpaceLimits,
        completion: &[f64],
        deadline: Instant,
        trust: Option<(&[f64], f64)>,
    ) -> Result<Result<Option<Vec<f64>>, SignSpaceRefusal>, SignSpaceError> {
        let lp = match self.build_realizability_lp(s1, units, z1, limits, trust) {
            Ok(parts) => parts,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let model = crate::ay_lib::to_ay_model(&lp.problem)
            .map_err(|e| SignSpaceError::Lowering(e.to_string()))?;
        let slack_col = model
            .col_at(lp.slack.0)
            .ok_or_else(|| SignSpaceError::Lowering("slack column out of range".to_string()))?;
        let opts = ay_milp::SolveOpts::new()
            .with_time_limit(limits.per_lp_time)
            .with_deadline(deadline);
        let mut session = ay_milp::LpSession::new(&model, &opts)
            .map_err(|e| SignSpaceError::Solver(e.to_string()))?;
        let outcome = session
            .optimize(slack_col, ay_milp::Sense::Maximize)
            .map_err(|e| SignSpaceError::Solver(e.to_string()))?;

        let model_values: Vec<BigRational> = match outcome {
            ay_milp::Outcome::Optimal { model_values, .. }
            | ay_milp::Outcome::Feasible { model_values, .. } => model_values,
            // Infeasible / Unbounded / Bound / Unknown: no realizing point in
            // hand. Never a verdict, just a rejected flip.
            _ => return Ok(Ok(None)),
        };
        if model_values.len() != model.num_cols() {
            return Ok(Ok(None));
        }
        // Repeat AY's own point gate locally, as `ay_lib::map_one_sided_sat_outcome`
        // does: a forged, truncated or stale point can only be REJECTED here,
        // never promoted to a witness.
        if model.check_point(&model_values).is_err() {
            return Ok(Ok(None));
        }
        // STRICT extraction (the twin of `ay_lib::extract_candidate_cols`):
        // never manufacture a missing or non-representable coordinate and leave
        // the concrete replay to discover it.
        let mut x = completion.to_vec();
        for &(pixel, col) in &lp.pixels {
            let Some(value) = model_values
                .get(col.0)
                .and_then(ToPrimitive::to_f64)
                .filter(|v| v.is_finite())
            else {
                return Ok(Ok(None));
            };
            x[pixel] = value;
        }
        Ok(Ok(Some(x)))
    }

    /// Solve one active-set LP under the current trust region, EXPANDING the
    /// region rather than concluding whenever the restricted LP hands back no
    /// point.
    ///
    /// THE SAFETY RULE, in one sentence: a trust region can only make the LP
    /// MORE constrained, so a failure under one proves nothing and must never
    /// reach the caller as a decline. Concretely, `Ok(None)` is returned from
    /// exactly one place here — after [`TrustState::expand`] has reported that
    /// the region was ALREADY the full box — plus the deadline guard, which is
    /// a decline for a reason that has nothing to do with the region.
    ///
    /// The `Nearest` arm then spends `refine` bisection steps between the last
    /// radius that failed and the first that worked, keeping the point from the
    /// smallest radius that still produced one. That is `min ||x - x0||_inf`
    /// subject to the same rows, bracketed rather than solved as an LP — and it
    /// costs no extra columns and no extra rows, which at exact-rational
    /// precision over ~6900 pixel columns is the whole reason it is done this
    /// way.
    fn solve_lp_under_trust_region(
        &self,
        s1: &[i8],
        units: &[usize],
        z1: &[f64],
        limits: &SignSpaceLimits,
        completion: &[f64],
        deadline: Instant,
        trust: &mut TrustState,
        trace: bool,
    ) -> Result<Result<Option<Vec<f64>>, SignSpaceRefusal>, SignSpaceError> {
        loop {
            let bounds = trust.radius.map(|r| (completion, r));
            let solved =
                self.solve_active_lp(s1, units, z1, limits, completion, deadline, bounds)?;
            match solved {
                Err(refusal) => return Ok(Err(refusal)),
                Ok(Some(x)) => {
                    // Only when a region is actually in play: with the shipped
                    // `FullBox` the trace stays exactly the stream §10 recorded.
                    if trace && (trust.radius.is_some() || trust.expansions > 0) {
                        eprintln!(
                            "NY_BNN_SIGN_SPACE_TRACE trust radius={} expansions={} feasible",
                            trust.radius.unwrap_or(f64::INFINITY),
                            trust.expansions,
                        );
                    }
                    return Ok(Ok(Some(self.refine_trust_radius(
                        s1, units, z1, limits, completion, deadline, trust, x, trace,
                    )?)));
                }
                Ok(None) => {
                    // NEVER a conclusion under a region. Expand and re-solve;
                    // `expand` returns false only when the region was already
                    // the full box, which IS today's LP and today's decline.
                    if !trust.expand(limits) {
                        return Ok(Ok(None));
                    }
                    if trace {
                        eprintln!(
                            "NY_BNN_SIGN_SPACE_TRACE trust expand to radius={} expansions={}",
                            trust.radius.unwrap_or(f64::INFINITY),
                            trust.expansions,
                        );
                    }
                    if Instant::now() >= deadline {
                        return Ok(Ok(None));
                    }
                }
            }
        }
    }

    /// Bisect the radius between the last failure and the first success,
    /// keeping the point from the smallest radius that still yields one.
    ///
    /// A no-op on every arm but [`TrustRegion::Nearest`], and a no-op when the
    /// first radius tried already worked with nothing below it to bracket
    /// against. Never returns a point from a radius that did not produce one:
    /// `best` starts as the feasible point the caller already holds.
    #[allow(clippy::too_many_arguments)]
    fn refine_trust_radius(
        &self,
        s1: &[i8],
        units: &[usize],
        z1: &[f64],
        limits: &SignSpaceLimits,
        completion: &[f64],
        deadline: Instant,
        trust: &mut TrustState,
        feasible: Vec<f64>,
        trace: bool,
    ) -> Result<Vec<f64>, SignSpaceError> {
        let mut best = feasible;
        let (Some(mut hi), Some(mut lo)) = (trust.radius, trust.last_failed) else {
            return Ok(best);
        };
        for _ in 0..trust.refine {
            if Instant::now() >= deadline || lo >= hi {
                break;
            }
            let mid = lo.midpoint(hi);
            if !mid.is_finite() || mid <= lo || mid >= hi {
                break;
            }
            match self.solve_active_lp(
                s1,
                units,
                z1,
                limits,
                completion,
                deadline,
                Some((completion, mid)),
            )? {
                Ok(Some(x)) => {
                    best = x;
                    hi = mid;
                }
                // Infeasible or inconclusive at `mid` — the bracket's lower end
                // moves up, and the point already in hand is kept. This can only
                // narrow the region; it can never conclude anything.
                _ => lo = mid,
            }
        }
        if trace && trust.refine > 0 {
            eprintln!("NY_BNN_SIGN_SPACE_TRACE trust refined radius={hi}");
        }
        // Start the next round from the tightened radius; it can expand again.
        trust.radius = Some(hi);
        trust.last_failed = None;
        Ok(best)
    }

    /// Is the WHOLE sign pattern realizable by some in-box `x` at slack
    /// `>= tolerance`, and if so, at which `x`?
    ///
    /// LAZY ROW GENERATION. The full LP carries one row per FREE unit (488-999
    /// on the shallow traffic-sign rows, times the pool area for `-1` units on
    /// the deeper ones) but only a handful are ever tight, and one
    /// exact-rational solve of the full system costs 60-220ms. So this solves
    /// over an ACTIVE SUBSET and re-checks the returned point against EVERY
    /// free unit, adding the violated ones and repeating.
    ///
    /// Feasibility is never taken from the subset: it is established by
    /// evaluating every free unit's TRUE (OR-identity) slack on the concrete
    /// point. Running out of rounds returns `Rejected`, which only ever loses
    /// a SAT.
    fn solve_realizability(
        &self,
        s1: &[i8],
        free: &[usize],
        limits: &SignSpaceLimits,
        completion: &[f64],
        deadline: Instant,
    ) -> Result<Result<Realizability, SignSpaceRefusal>, SignSpaceError> {
        let mut in_active = vec![false; free.len()];
        let mut active: Vec<usize> = Vec::new();
        let mut deficient: Vec<usize> = Vec::new();
        let mut x = completion.to_vec();
        let epsilon = f64_slack_verification_epsilon(self.taps1, self.max_abs);
        let trace = sign_space_trace_armed();
        // The ANCHOR is `completion`, and deliberately so: it is the point the
        // search is standing on, and it is also the point every pixel WITHOUT a
        // column keeps in the assembled primal, so the region is centred on the
        // same vector the LP is completing against. Dark by default — with
        // `TrustRegion::FullBox` this is `radius: None` and every bound below is
        // the vnnlib bound, byte for byte.
        let mut trust = TrustState::new(self, limits, completion);

        for round in 0..=limits.max_row_generation_rounds {
            // Every free unit's true slack at the current point.
            let z1 = self.z1_at(&x);
            let mut worst = f64::INFINITY;
            let mut added = 0usize;
            deficient.clear();
            for (slot, &k) in free.iter().enumerate() {
                let (slack, _) = self.unit_slack(k, s1, &z1);
                worst = worst.min(slack);
                if slack < limits.tolerance {
                    // EVERY unit short of the tolerance, not just the newly
                    // discovered ones: these are exactly the units the LP is
                    // being asked to fix, and the minimal move has to carry all
                    // of them (they are all in the active set by construction,
                    // so they are all constrained at `theta == 1`).
                    deficient.push(k);
                    if !in_active[slot] {
                        in_active[slot] = true;
                        active.push(k);
                        added += 1;
                    }
                }
            }
            if trace {
                eprintln!(
                    "NY_BNN_SIGN_SPACE_TRACE round={round} active={} added={added} \
                     deficient={} worst={worst:.4}",
                    active.len(),
                    deficient.len(),
                );
            }
            if worst >= limits.tolerance - epsilon {
                // The concrete point realizes the WHOLE pattern. The epsilon
                // only absorbs `f64` re-evaluation noise on a slack the
                // exact-rational LP already put at `>= tolerance`; the witness
                // itself is gated far above it by the `f32` replay floor.
                return Ok(Ok(Realizability::Realizable { slack: worst, x }));
            }
            if added == 0 || round == limits.max_row_generation_rounds {
                // No new information to add (or the round budget is spent):
                // decline rather than loop.
                return Ok(Ok(Realizability::Rejected));
            }
            match self.solve_lp_under_trust_region(
                s1, &active, &z1, limits, completion, deadline, &mut trust, trace,
            )? {
                // The active set outgrew an LP size cap. That is a DECLINE of
                // this pattern, not a refusal of the request: under pooling a
                // `-1` unit contributes one row per window member, so the
                // active set can outgrow `max_lp_rows` deep into a search that
                // was admitted perfectly legitimately. Declining loses a SAT
                // and nothing else; aborting the whole call would throw away
                // every flip already banked.
                Err(_) => return Ok(Ok(Realizability::Rejected)),
                Ok(None) => return Ok(Ok(Realizability::Rejected)),
                Ok(Some(next)) => {
                    x = match limits.segment_move {
                        SegmentMove::Vertex => next,
                        SegmentMove::MinimalTheta => {
                            // `z1` is linear in `x`, so the two endpoint arrays
                            // determine EVERY unit's crossing in closed form.
                            // `z1` above is already the start endpoint.
                            let z1_lp = self.z1_at(&next);
                            let theta = self.minimal_segment_theta(
                                s1,
                                &deficient,
                                &z1,
                                &z1_lp,
                                limits.tolerance,
                            );
                            if trace {
                                eprintln!("NY_BNN_SIGN_SPACE_TRACE round={round} theta={theta:.6}");
                            }
                            if theta >= 1.0 {
                                next
                            } else {
                                // On `None` (a non-finite blend) keep the
                                // vertex: it is the point the LP certified and
                                // it is known in-box.
                                self.blend_into_box(&x, &next, theta).unwrap_or(next)
                            }
                        }
                    };
                }
            }
        }
        Ok(Ok(Realizability::Rejected))
    }

    /// `(row, col, channel)` of a RAW conv1 unit index.
    #[inline]
    const fn raw_coords(&self, m: usize) -> (usize, usize, usize) {
        let ch = m % self.c1;
        let spatial = m / self.c1;
        (spatial / self.w1c, spatial % self.w1c, ch)
    }
}

// ---------------------------------------------------------------------------
// Witness finalization: f32 replay preparation + independent recompute
// ---------------------------------------------------------------------------

/// The smallest `f32` whose `f64` value is `>= lo` (`None` if none exists).
fn f32_ceil_in(lo: f64) -> Option<f32> {
    let mut c = lo as f32;
    if !c.is_finite() {
        return None;
    }
    if f64::from(c) < lo {
        c = c.next_up();
    }
    (f64::from(c) >= lo).then_some(c)
}

/// The largest `f32` whose `f64` value is `<= hi` (`None` if none exists).
fn f32_floor_in(hi: f64) -> Option<f32> {
    let mut c = hi as f32;
    if !c.is_finite() {
        return None;
    }
    if f64::from(c) > hi {
        c = c.next_down();
    }
    (f64::from(c) <= hi).then_some(c)
}

impl Admitted<'_> {
    /// Round the LP primal to the bytes ONNX Runtime will actually read, then
    /// re-clamp INTO the box.
    ///
    /// The `f32` cast comes FIRST and the clamp second, matching
    /// `mip_highs::clamp_witness_to_box`, so the returned vector is exactly
    /// the `f32` values the organizer's runtime sees AND is in-box in `f64`
    /// with no tolerance.
    fn to_replay_bytes(&self, x: &[f64]) -> Option<Vec<f64>> {
        let mut out = Vec::with_capacity(x.len());
        for (p, &v) in x.iter().enumerate() {
            let (lo, hi) = (self.lo[p], self.hi[p]);
            let mut v32 = v as f32;
            if !v32.is_finite() {
                return None;
            }
            if f64::from(v32) < lo {
                v32 = f32_ceil_in(lo)?;
            }
            if f64::from(v32) > hi {
                v32 = f32_floor_in(hi)?;
            }
            let value = f64::from(v32);
            if value < lo || value > hi {
                // The box admits no f32 at all; refuse rather than ship a
                // witness the organizer's replay would reject as out-of-box.
                return None;
            }
            out.push(value);
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// The public entry point
// ---------------------------------------------------------------------------

/// LP-guided sign-space falsification of a binarized conv suffix.
///
/// Returns [`SignSpaceOutcome::Candidate`] with the realizability LP's primal
/// as a concrete in-box counterexample, [`SignSpaceOutcome::Exhausted`] when
/// the greedy search runs out of realizable improving flips or budget, or
/// [`SignSpaceOutcome::Refused`] when the request is outside the admitted
/// fragment.
///
/// **There is no verified/unsat outcome by construction.** A candidate is a
/// CLAIM: the caller MUST replay [`SignSpaceCandidate::input`] through the
/// ORIGINAL network and property before publishing anything.
pub fn falsify_bnn_sign_suffix_unwired(
    request: &SignSpaceRequest<'_>,
    limits: &SignSpaceLimits,
) -> Result<SignSpaceOutcome, SignSpaceError> {
    let started = Instant::now();
    let admitted = match admit(request, limits) {
        Ok(admitted) => admitted,
        Err(refusal) => return Ok(SignSpaceOutcome::Refused(refusal)),
    };
    let deadline = started + limits.max_wall_time;

    // --- STEP 1: the exact prepass.
    let classification = admitted.classify();
    let free = classification.free;
    if let Err(refusal) = limit("max_free_units", free.len(), limits.max_free_units) {
        return Ok(SignSpaceOutcome::Refused(refusal));
    }

    // --- the reference point, already in the bytes a replay would read.
    let midpoint: Vec<f64> = (0..admitted.n_pixels)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();
    let start = request.reference_input.unwrap_or(&midpoint);
    let Some(x0) = admitted.to_replay_bytes(start) else {
        return Ok(SignSpaceOutcome::Refused(SignSpaceRefusal::DegenerateBox {
            index: 0,
            lo: admitted.lo[0],
            hi: admitted.hi[0],
        }));
    };

    // --- STEP 2 self-check, BEFORE any search: a transposed flatten is
    // arithmetically plausible and completely wrong.
    let s1_0 = admitted.s1_at(&x0);
    let mut engine = admitted.forward_from_pattern(&s1_0);
    if let Err(refusal) = self_check(&admitted, request, limits, &x0, &engine) {
        return Ok(SignSpaceOutcome::Refused(refusal));
    }

    // --- STEP 4: greedy search over the free bits.
    //
    // Two nested economies, because one exact-rational LP costs ~60ms here
    // while one full single-flip ranking costs ~0.5ms (STEP 0 measurement):
    //
    //   * BATCH mode applies `batch` greedy flips — each chosen by a complete
    //     re-ranking of every unlocked free bit — and pays ONE LP for the
    //     whole block. The doc measures ~250 of 488 flips as SIMULTANEOUSLY
    //     realizable, so blocks usually survive. Success doubles the block,
    //     rejection halves it and rolls the block back exactly.
    //   * At `batch == 1` the block schedule degenerates to the ranked-head
    //     walk: one LP per ranked candidate until one is realizable.
    //
    // Correctness does not depend on any of that: the LP is the ONLY gate that
    // can accept a pattern, and a rejected block is undone bit-for-bit.
    let mut locked = vec![false; free.len()];
    let mut flips = 0usize;
    let mut lp_solves = 0usize;
    let (mut best_margin, _) = admitted.margin(&engine.logits);
    // #lane-value-stall. The walk's own VALUE unit is the movement of this
    // margin; the denominator is `lp_solves`. Both are reported on `Exhausted`
    // so the lane can price itself without the scheduler guessing.
    let initial_margin = best_margin;
    let mut last_gain_lp = 0usize;
    let mut batch = 1usize;
    // The first window of the ranked-head walk; it widens through the tail
    // when the head is exhausted (see the walk below).
    let head = limits.candidate_head.max(1);

    // #bnn-lp-stall. "This walk has paid `stall_lp_solves` LPs and accepted
    // NOTHING." Budget-only: the two reachable ends are `Candidate` and
    // `Exhausted`, and this can only bring `Exhausted` forward, so it cannot
    // turn a SAT into anything else — it can only decline to keep paying for a
    // search that has not moved once. See `SignSpaceLimits::stall_lp_solves`.
    //
    // #lane-value-stall adds the SECOND question, `stall_margin_lp_solves`:
    // "this walk has paid that many LPs since the margin last MOVED". Disabled
    // by default, so the shipped rule is exactly the one above.
    let stalled = |flips: usize, lp_solves: usize, last_gain_lp: usize| {
        (limits.stall_lp_solves > 0 && flips == 0 && lp_solves >= limits.stall_lp_solves)
            || (limits.stall_margin_lp_solves > 0
                && lp_solves.saturating_sub(last_gain_lp) >= limits.stall_margin_lp_solves)
    };

    // The reference point's own pattern is realizable by the reference point,
    // so a violation already present there is a candidate immediately.
    if best_margin > 0 && lp_solves < limits.max_lp_solves && Instant::now() < deadline {
        lp_solves += 1;
        match admitted.solve_realizability(&engine.s0, &free, limits, &x0, deadline)? {
            Err(refusal) => return Ok(SignSpaceOutcome::Refused(refusal)),
            Ok(Realizability::Realizable { slack, x }) => {
                if let Some(candidate) = finalize(
                    &admitted,
                    &x,
                    slack,
                    free.len(),
                    flips,
                    lp_solves,
                    started.elapsed(),
                ) {
                    return Ok(SignSpaceOutcome::Candidate(Box::new(candidate)));
                }
            }
            Ok(Realizability::Rejected) => {}
        }
    }

    loop {
        if Instant::now() >= deadline
            || lp_solves >= limits.max_lp_solves
            || stalled(flips, lp_solves, last_gain_lp)
        {
            break;
        }

        if batch > 1 {
            // Apply a block of greedy flips, re-ranking before each one.
            let mut applied: Vec<usize> = Vec::with_capacity(batch);
            for _ in 0..batch {
                let Some(slot) = best_unlocked_flip(&admitted, &mut engine, &free, &locked) else {
                    break;
                };
                admitted.flip(&mut engine, free[slot]);
                locked[slot] = true;
                applied.push(slot);
            }
            if applied.is_empty() {
                break;
            }
            lp_solves += 1;
            match admitted.solve_realizability(&engine.s0, &free, limits, &x0, deadline)? {
                Err(refusal) => return Ok(SignSpaceOutcome::Refused(refusal)),
                Ok(Realizability::Realizable { slack, x }) => {
                    flips += applied.len();
                    let (margin, _) = admitted.margin(&engine.logits);
                    if margin > best_margin {
                        best_margin = margin;
                        last_gain_lp = lp_solves;
                    }
                    if margin > 0 {
                        if let Some(candidate) = finalize(
                            &admitted,
                            &x,
                            slack,
                            free.len(),
                            flips,
                            lp_solves,
                            started.elapsed(),
                        ) {
                            return Ok(SignSpaceOutcome::Candidate(Box::new(candidate)));
                        }
                    }
                    batch = (batch * 2).min(limits.max_flip_batch.max(1));
                }
                Ok(Realizability::Rejected) => {
                    // Exact rollback, newest flip first.
                    for &slot in applied.iter().rev() {
                        admitted.flip(&mut engine, free[slot]);
                        locked[slot] = false;
                    }
                    batch /= 2;
                }
            }
            continue;
        }

        // batch == 1: the ranked-head walk. Rank EVERY unlocked free bit by its
        // single-flip margin. Deterministic: margin descending, then unit index
        // ascending. No RNG anywhere.
        let mut ranked: Vec<(i64, usize)> = Vec::with_capacity(free.len());
        for (slot, &u) in free.iter().enumerate() {
            if locked[slot] {
                continue;
            }
            admitted.flip(&mut engine, u);
            let (m, _) = admitted.margin(&engine.logits);
            admitted.flip(&mut engine, u);
            ranked.push((m, slot));
        }
        if ranked.is_empty() {
            break;
        }
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| free[a.1].cmp(&free[b.1])));

        // Walk the ranked list, one LP per candidate, starting with the head
        // and widening through the tail rather than declaring exhaustion after
        // the first few. `Exhausted` here therefore means "NO unlocked free bit
        // is individually realizable", which is the honest reading; a shallow
        // head was measured stopping `idx_178` at 25 flips with budget to
        // spare. Nothing is re-tried: the engine is unchanged by a rejection,
        // so the ranking is unchanged too.
        let mut start = 0usize;
        let mut accepted = false;
        let mut window = head;
        while start < ranked.len() && !accepted {
            let stop = (start + window).min(ranked.len());
            for &(candidate_margin, slot) in &ranked[start..stop] {
                if Instant::now() >= deadline
                    || lp_solves >= limits.max_lp_solves
                    || stalled(flips, lp_solves, last_gain_lp)
                {
                    break;
                }
                let u = free[slot];
                admitted.flip(&mut engine, u);
                lp_solves += 1;
                match admitted.solve_realizability(&engine.s0, &free, limits, &x0, deadline)? {
                    Err(refusal) => return Ok(SignSpaceOutcome::Refused(refusal)),
                    Ok(Realizability::Realizable { slack, x }) => {
                        locked[slot] = true;
                        flips += 1;
                        accepted = true;
                        if candidate_margin > best_margin {
                            best_margin = candidate_margin;
                            last_gain_lp = lp_solves;
                        }
                        if candidate_margin > 0 {
                            if let Some(candidate) = finalize(
                                &admitted,
                                &x,
                                slack,
                                free.len(),
                                flips,
                                lp_solves,
                                started.elapsed(),
                            ) {
                                return Ok(SignSpaceOutcome::Candidate(Box::new(candidate)));
                            }
                        }
                        break;
                    }
                    Ok(Realizability::Rejected) => {
                        // Not realizable at the required slack: undo and try
                        // the next ranked candidate.
                        admitted.flip(&mut engine, u);
                    }
                }
            }
            if Instant::now() >= deadline
                || lp_solves >= limits.max_lp_solves
                || stalled(flips, lp_solves, last_gain_lp)
            {
                break;
            }
            start = stop;
            window = window.saturating_mul(4);
        }
        if !accepted {
            break;
        }
        batch = 2.min(limits.max_flip_batch.max(1));
    }

    Ok(SignSpaceOutcome::Exhausted {
        best_logit_margin: best_margin,
        margin_gain: best_margin.saturating_sub(initial_margin).max(0),
        free_units: free.len(),
        flips,
        lp_solves,
        elapsed: started.elapsed(),
    })
}

/// The unlocked FREE bit whose single flip gives the best pre-softmax margin.
///
/// Deterministic: margin descending, then first-layer unit index ascending.
/// The engine is left EXACTLY as it was found — every trial flip is undone.
fn best_unlocked_flip(
    admitted: &Admitted<'_>,
    engine: &mut Engine,
    free: &[usize],
    locked: &[bool],
) -> Option<usize> {
    let mut best: Option<(i64, usize)> = None;
    for (slot, &u) in free.iter().enumerate() {
        if locked[slot] {
            continue;
        }
        admitted.flip(engine, u);
        let (m, _) = admitted.margin(&engine.logits);
        admitted.flip(engine, u);
        match best {
            Some((bm, bs)) if bm > m || (bm == m && free[bs] <= u) => {}
            _ => best = Some((m, slot)),
        }
    }
    best.map(|(_, slot)| slot)
}

/// Turn an LP primal into a candidate, or reject it.
///
/// Everything here is recomputed FROM SCRATCH on the `f32`-rounded, re-clamped
/// input — the search's incremental state is not consulted. A candidate is
/// emitted only when that independent forward yields a STRICTLY POSITIVE
/// pre-softmax margin, which is stronger than the non-strict
/// `Y[c] >= Y[t]` the property asks for and refuses the degenerate tie that
/// softmax underflow would otherwise make indistinguishable.
fn finalize(
    admitted: &Admitted<'_>,
    x: &[f64],
    slack: f64,
    free_units: usize,
    flips: usize,
    lp_solves: usize,
    elapsed: Duration,
) -> Option<SignSpaceCandidate> {
    let input = admitted.to_replay_bytes(x)?;
    for (p, &v) in input.iter().enumerate() {
        if v < admitted.lo[p] || v > admitted.hi[p] {
            return None;
        }
    }
    // REPLAY-STABILITY, order-independently. `finalize` reasons in `f64`; the
    // organizer's runtime reasons in `f32` with an unknown summation order.
    // Requiring every first-layer unit's POOLED value to clear its threshold by
    // more than the geometry's `f32` accumulation bound makes the whole sign
    // pattern — and therefore every integer downstream of it — identical under
    // ANY `f32` ordering, so the two forwards cannot disagree.
    //
    // Under pooling the same single number does both jobs: `max_w z_w > t + e`
    // means SOME member is above by more than `e` (so the `f32` max is above
    // `t`), and `max_w z_w < t - e` means EVERY member is below by more than
    // `e` (so the `f32` max is below `t`).
    //
    // `max` itself is exact in any precision, and every stage accumulator is a
    // sum of unit-signed terms that stays integral and below `2^24`, where
    // `f32` addition is exact — so the only remaining runtime rounding is the
    // folded affine at each site, which `Stage::guard` bounds.
    let floor = f32_replay_slack_floor(admitted.taps1, admitted.max_abs);
    let z1 = admitted.z1_at(&input);
    for k in 0..admitted.n_units1 {
        let channel = k % admitted.c1;
        let (pooled, _) = admitted.pooled_at(k, &z1);
        if (pooled - admitted.t1[channel]).abs() <= floor + admitted.guard1[channel] {
            return None;
        }
    }
    let s1 = admitted.s1_from_z1(&z1);
    let engine = admitted.forward_from_pattern(&s1);
    // The same argument, one stage at a time: a stage decision sitting inside
    // its folded-affine rounding bound is one an `f32` runtime could flip.
    // A ZERO guard means the decision is exact in `f32` (see
    // `folded_affine_guard`), and a zero margin there is the legitimate
    // `B(0) = +1` boundary rather than a rounding hazard — so it must NOT be
    // rejected. Only a positive guard imposes a positive margin.
    for (index, stage) in admitted.stages.iter().enumerate() {
        if stage.violates_replay_guard(&engine.p[index]) {
            return None;
        }
    }
    let (logit_margin, best_challenger) = admitted.margin(&engine.logits);
    if logit_margin <= 0 {
        return None;
    }
    Some(SignSpaceCandidate {
        argmax: admitted.argmax(&engine.logits),
        logits: engine.logits,
        input,
        best_challenger,
        logit_margin,
        lp_slack: slack,
        free_units,
        flips,
        lp_solves,
        elapsed,
    })
}

// ---------------------------------------------------------------------------
// Self-checks (STEP 2 gate)
// ---------------------------------------------------------------------------

fn self_check(
    admitted: &Admitted<'_>,
    request: &SignSpaceRequest<'_>,
    limits: &SignSpaceLimits,
    x0: &[f64],
    engine: &Engine,
) -> Result<(), SignSpaceRefusal> {
    // (a) the incremental delta path must equal a from-scratch recompute.
    if let Some(stride) = admitted
        .n_units1
        .checked_div(limits.self_check_incremental_flips)
        .map(|stride| stride.max(1))
    {
        let sample: Vec<usize> = (0..admitted.n_units1)
            .step_by(stride)
            .take(limits.self_check_incremental_flips)
            .collect();
        let mut incremental = admitted.forward_from_pattern(&engine.s0);
        for &u in &sample {
            admitted.flip(&mut incremental, u);
        }
        let scratch = admitted.forward_from_pattern(&incremental.s0);
        for class in 0..admitted.num_classes {
            if incremental.logits[class] != scratch.logits[class] {
                return Err(SignSpaceRefusal::LogitEngineDisagrees {
                    sample: usize::MAX,
                    class,
                    engine: incremental.logits[class] as f64,
                    reference: scratch.logits[class] as f64,
                });
            }
        }
    }

    // (b) against the caller's independent forward, at the centre and at
    // sampled box vertices. Integers on both sides, so equality is exact.
    let Some(reference) = request.reference_forward else {
        return Ok(());
    };
    let mut samples: Vec<Vec<f64>> = vec![x0.to_vec()];
    for s in 0..limits.self_check_vertex_samples {
        // Deterministic vertex schedule; no RNG.
        let vertex: Vec<f64> = (0..admitted.n_pixels)
            .map(|p| {
                if (p + s) % (s + 2) == 0 {
                    admitted.lo[p]
                } else {
                    admitted.hi[p]
                }
            })
            .collect();
        if let Some(v) = admitted.to_replay_bytes(&vertex) {
            samples.push(v);
        }
    }
    // NOTE: no near-threshold exemption here, deliberately. A unit sitting
    // within the `f32` accumulation bound of its threshold COULD make an
    // honest reference disagree, but exempting those samples would also
    // silently exempt the layout transposition this check exists to catch (a
    // symmetric box puts many units at exactly `z1 = 0`). Refusing is safe —
    // a refusal is never a soundness event — so this fails closed.
    for (index, sample) in samples.iter().enumerate() {
        let Some(expected) = reference(sample) else {
            continue;
        };
        if expected.len() != admitted.num_classes {
            return Err(SignSpaceRefusal::LogitEngineDisagrees {
                sample: index,
                class: usize::MAX,
                engine: admitted.num_classes as f64,
                reference: expected.len() as f64,
            });
        }
        let s1 = admitted.s1_at(sample);
        let ours = admitted.forward_from_pattern(&s1);
        for class in 0..admitted.num_classes {
            let mine = ours.logits[class] as f64;
            // Both sides are integers; anything past half a unit is a real
            // disagreement, not rounding.
            if (mine - expected[class]).abs() > 0.5 {
                return Err(SignSpaceRefusal::LogitEngineDisagrees {
                    sample: index,
                    class,
                    engine: mine,
                    reference: expected[class],
                });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Diagnostic entry points (used by the tests and by an offline harness; never
// on a verdict path).
// ---------------------------------------------------------------------------

/// Run only the admission predicate and the EXACT FIXED/FREE prepass.
///
/// Exposed so an offline harness (and the module's own oracle tests) can pin
/// the measured free counts without paying for a search.
pub fn classify_first_layer_unwired(
    request: &SignSpaceRequest<'_>,
    limits: &SignSpaceLimits,
) -> Result<UnitClassification, SignSpaceRefusal> {
    let admitted = admit(request, limits)?;
    Ok(admitted.classify())
}

/// Pre-softmax logits of a concrete input under the admitted fragment.
///
/// The same from-scratch forward a candidate is finalized with; exposed for
/// oracle tests and for offline cross-checks against ONNX Runtime.
pub fn logits_at_unwired(
    request: &SignSpaceRequest<'_>,
    limits: &SignSpaceLimits,
    x: &[f64],
) -> Result<Vec<i64>, SignSpaceRefusal> {
    let admitted = admit(request, limits)?;
    if x.len() != admitted.n_pixels {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "input has {} entries, geometry needs {}",
                x.len(),
                admitted.n_pixels
            ),
        });
    }
    let s1 = admitted.s1_at(x);
    Ok(admitted.forward_from_pattern(&s1).logits)
}

/// Solve ONE realizability LP for the sign pattern induced by `x`, returning
/// the optimal slack and the LP's primal.
///
/// This is the STEP-0 measurement entry point and the oracle for the
/// "the primal genuinely satisfies every constraint" test.
pub fn realizability_probe_unwired(
    request: &SignSpaceRequest<'_>,
    limits: &SignSpaceLimits,
    x: &[f64],
) -> Result<Option<(f64, Vec<f64>)>, SignSpaceError> {
    let admitted = match admit(request, limits) {
        Ok(admitted) => admitted,
        Err(_) => return Ok(None),
    };
    if x.len() != admitted.n_pixels {
        return Ok(None);
    }
    let classification = admitted.classify();
    let s1 = admitted.s1_at(x);
    let deadline = Instant::now() + limits.max_wall_time;
    let Some(completion) = admitted.to_replay_bytes(x) else {
        return Ok(None);
    };
    match admitted.solve_realizability(&s1, &classification.free, limits, &completion, deadline)? {
        Ok(Realizability::Realizable { slack, x }) => Ok(Some((slack, x))),
        Ok(Realizability::Rejected) | Err(_) => Ok(None),
    }
}

/// The STE-PGD falsification lane over the SAME admitted fragment.
///
/// A child module so it can reuse this module's PRIVATE, already-validated
/// machinery — `admit`, the exact `Admitted` forward, and the `finalize`
/// witness gate — instead of standing up a second extraction of the same net.
#[path = "bnn_ste_pgd.rs"]
mod bnn_ste_pgd;

pub use bnn_ste_pgd::{falsify_bnn_ste_pgd_unwired, StePgdLimits};

#[cfg(test)]
#[path = "bnn_sign_space_tests.rs"]
mod bnn_sign_space_tests;
