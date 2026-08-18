// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified scalar f64 CROWN backward for tiny graph nets — the "f64 tail
//! pass" (docs/LSNC_F64_TAIL_DESIGN.md).
//!
//! # Why (#lsnc-f64-tail)
//!
//! The lsnc_relu input-split lane is f32 end-to-end. Its verdict floor is the
//! (correctly certified) f32 STORAGE noise of intermediate coefficients —
//! a band of ~1e-5..1e-3 per domain that swamps the 1e-6 clearance the
//! Lyapunov-decrease spec rows need (design §1.1). This module re-runs the
//! CROWN backward for ONE domain in a certified-outward f64 carrier: same
//! relaxation quality (the lane's own f32 per-node anchors, exact-widened;
//! the root-frozen MulBinary SPSA alphas), but an f64 envelope of
//! `gamma_n·S ≈ 1e-11` instead of the f32 floor.
//!
//! # Soundness contract (design §5)
//!
//! - Every relaxation plane is VALID-BY-CONSTRUCTION for its exact computed
//!   f64 coefficients: ReLU upper intercepts are directed-rounded up from
//!   both endpoint conditions; every bilinear (MulBinary/McCormick) plane
//!   goes through the corner-certify-and-repair gadget (§5.4) — the bilinear
//!   gap `d` attains its minimum at a box corner, 4 directed-rounded corner
//!   evaluations certify or repair the intercept.
//! - The backward is a telescoped chain: every step's inequality holds
//!   EXACTLY for the stored f64 coefficients (whose own signs drove the
//!   relaxation selection — no sign-flip hazard), and each step's rounding is
//!   discharged as a coefficient-space error weighted by the RANGE of the
//!   node the coefficients land on (the same discipline as the f32 lane's
//!   `fold_coeff_err_over_box_eager`, parity checklist I-C3 — strictly
//!   stronger than the design-§5.5 input-concretized `γ·S` alone, which
//!   under-counts intermediate-node roundings). A parallel ABS shadow walk
//!   provides the per-step Higham base; the total envelope
//!   `gamma_n · (Σ step_abs·node_range + bias_abs + concretize_abs)` is
//!   compensated for its own rounding via [`certify_up`] and finalized with
//!   [`next_down`] (Higham, *Accuracy and Stability of Numerical Algorithms*,
//!   §3.5 — order independent, so no bit-parity claim is made or needed).
//! - Comparison against thresholds happens in f64 (`f32 -> f64` is exact);
//!   the grouped verdict mirrors `disjunctive_domain_verified` including the
//!   non-finite-never-verifies rule.
//! - FAIL-CLOSED everywhere: unsupported op / shape surprise / failed anchor
//!   collection => [`F64TailOutcome::Unsupported`] (the caller keeps the f32
//!   verdict); NaN/Inf on a row => that row poisons to `-inf` (never
//!   verifies). `Verified` is the ONLY verdict-changing outcome.
//!
//! # Gate (design §6.2, parity checklist B)
//!
//! Default OFF. `NY_F64_TAIL=1` enables (cached; test override
//! [`force_f64_tail`]). Guard band `NY_F64_TAIL_BAND` (default `5e-3`)
//! bounds which still-unverified domains the batch seam escalates. With the
//! gate off the hooks return before doing ANY work — the lane is
//! byte-identical (verified by `gate_off_is_byte_identical` in
//! `input_split/f64_tail.rs`).
//!
//! # Alpha-tail escalation (docs/LSNC_ALPHA_TAIL_DESIGN.md, options A+B)
//!
//! `NY_ALPHA_TAIL=1` (default OFF; cached, test override
//! [`force_alpha_tail`]) arms the SAME escalation seam with two additions:
//!
//! 1. **Per-domain MulBinary alpha refresh** ([`f64_tail_verify_refreshed`]):
//!    the root-frozen SPSA alphas are optimized for the wrong box and the
//!    wrong rows at the tail (design §2/§3.A). A short SPSA+Adam refresh
//!    (`NY_ALPHA_TAIL_ITERS`, default 20 — the same recipe as
//!    `input_split/mul_binary.rs`) re-targets them for THE domain's box and
//!    its BLOCKING clause rows, evaluated through the certified f64 row walk
//!    on the cached anchors. Warm-started at the root alphas with
//!    keep-best-seen PER ROW, so the refreshed pass can only meet-or-beat the
//!    frozen-alpha pass. Sound-by-construction: every `r ∈ [0,1]`
//!    parameterizes a convex combination of two valid McCormick facets
//!    (design §2.1), and the corner-certify-and-repair gadget validates every
//!    plane regardless of the bits — the optimizer only SELECTS which sound
//!    relaxation the certified pass evaluates. Mixing alphas across rows is
//!    sound because every row walk is an independent certificate.
//! 2. **Micro-BaB** (hook side, `input_split/f64_tail.rs`): a near-threshold
//!    domain the refreshed single-shot cannot close is midpoint-split
//!    (exact cover) to depth `NY_ALPHA_TAIL_DEPTH` (default 3) and the
//!    CHILDREN are refreshed-f64 re-bounded; ALL children must verify for
//!    the parent to count (grouped semantics), else the whole escalation
//!    declines fail-closed.
//!
//! `NY_ALPHA_TAIL` composes with/supersedes `NY_F64_TAIL`: one escalation
//! path at the same seam, refreshed when alpha-tail is armed. Guard band
//! `NY_ALPHA_TAIL_BAND` (default `5e-3`, same band as the f64 tail);
//! micro-BaB eligibility band `NY_ALPHA_TAIL_MICRO_BAND` (default `5e-4` —
//! beyond that the measured 1.5-2x/level tightening cannot reach at depth
//! ≤ 4, design §3.B).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use ndarray::Array2;
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::margin_row::rounding::{certify_up, gamma_n, next_down, next_up, UNIT};
use crate::network::core::graph::{GraphNetwork, NETWORK_INPUT};

/// Outcome of one f64 tail pass over a single domain (design §5.6).
///
/// `Verified` is the ONLY verdict-changing outcome. It carries the certified
/// per-spec-row f64 lower bounds so the batch-seam hook can raise `obj_bounds`
/// monotonically (design §6.3; rows that did not clear stay `-inf` and are
/// left untouched by the caller).
#[derive(Debug, Clone)]
pub(crate) enum F64TailOutcome {
    /// Certified: every clause has a row with finite `l_cert > threshold`,
    /// established end-to-end in f64 with the outward-rounded envelope.
    Verified {
        /// Certified f64 lower bound per spec row (`-inf` = row not certified).
        row_lowers: Vec<f64>,
    },
    /// The pass ran but could not refute every clause. `min_gap_f64` is the
    /// worst clause's best certified row gap (the f64 mirror of
    /// `disjunctive_domain_priority`) — the fp-vs-relaxation discriminator
    /// telemetry of design §6.5.
    NotVerified { min_gap_f64: f64 },
    /// Op class / shape / anchor-collection miss — fail closed, the f32
    /// verdict stands unchanged.
    Unsupported,
}

// ---------------------------------------------------------------------------
// Gate (house pattern: cached atomic + cfg(test) override, cf.
// `backward_core.rs::input_split_batched_relu_enabled`).
// ---------------------------------------------------------------------------

static F64_TAIL_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the f64 tail pass is enabled. Default OFF; opt-in `NY_F64_TAIL=1`.
///
/// Parity class: the pass is ADDITIVE (only `Verified` changes anything and
/// only by monotonically raising certified lower bounds); with the gate off
/// the hooks are byte-identical no-ops. Parity test:
/// `input_split::f64_tail::tests::gate_off_is_byte_identical`.
pub(crate) fn f64_tail_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match F64_TAIL_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = matches!(std::env::var("NY_F64_TAIL").ok().as_deref(), Some("1"));
            F64_TAIL_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Test-only override for the gate: `Some(true|false)` forces ON/OFF, `None`
/// restores the env-derived default. Mirrors `force_batched_relu`.
#[cfg(test)]
pub(crate) fn force_f64_tail(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    F64_TAIL_MODE.store(v, Ordering::Relaxed);
}

/// Guard band for the batch-seam escalation trigger (design §6.1): a
/// still-unverified domain is eligible iff its grouped f32 gap is
/// `>= -band`. Env `NY_F64_TAIL_BAND`, default `5e-3` (covers the measured
/// −0.004 plateau of `state_35` with margin; caps wasted f64 work).
pub(crate) fn f64_tail_band() -> f32 {
    match std::env::var("NY_F64_TAIL_BAND")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        Some(band) if band.is_finite() && band >= 0.0 => band,
        _ => 5e-3,
    }
}

// ---------------------------------------------------------------------------
// Alpha-tail gates (docs/LSNC_ALPHA_TAIL_DESIGN.md; module doc above).
// ---------------------------------------------------------------------------

static ALPHA_TAIL_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the alpha-tail escalation is armed. Default OFF; opt-in
/// `NY_ALPHA_TAIL=1`. Composes with/supersedes `NY_F64_TAIL` at the same
/// seam (either gate arms the escalation; alpha-tail additionally enables
/// the per-domain refresh + micro-BaB). Parity class: ADDITIVE — only
/// certified `Verified` outcomes change anything, and only via the monotonic
/// merge; with both gates off the hooks are byte-identical no-ops.
pub(crate) fn alpha_tail_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match ALPHA_TAIL_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = matches!(std::env::var("NY_ALPHA_TAIL").ok().as_deref(), Some("1"));
            ALPHA_TAIL_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Test-only override for the alpha-tail gate (mirrors [`force_f64_tail`]).
#[cfg(test)]
pub(crate) fn force_alpha_tail(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    ALPHA_TAIL_MODE.store(v, Ordering::Relaxed);
}

/// Escalation guard band when the alpha tail is armed. Env
/// `NY_ALPHA_TAIL_BAND`, default `5e-3` — the SAME band as the f64 tail
/// (design decision: the refresh targets the same in-band cohort).
pub(crate) fn alpha_tail_band() -> f32 {
    match std::env::var("NY_ALPHA_TAIL_BAND")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        Some(band) if band.is_finite() && band >= 0.0 => band,
        _ => 5e-3,
    }
}

/// Micro-BaB eligibility band: only domains whose POST-REFRESH grouped f64
/// gap is `>= -micro_band` are split (design §3.B: beyond ~5e-4 the measured
/// per-level tightening cannot reach at depth ≤ 4, so splitting is wasted
/// work). Env `NY_ALPHA_TAIL_MICRO_BAND`, default `5e-4`.
pub(crate) fn alpha_tail_micro_band() -> f32 {
    match std::env::var("NY_ALPHA_TAIL_MICRO_BAND")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        Some(band) if band.is_finite() && band >= 0.0 => band,
        _ => 5e-4,
    }
}

/// Micro-BaB depth cap. Env `NY_ALPHA_TAIL_DEPTH`, default 3 (≤ 2^3 = 8
/// leaves best-first with short-circuit on the first failing child), clamped
/// to `0..=6`. `0` disables micro-BaB (refresh only).
pub(crate) fn alpha_tail_depth() -> usize {
    match std::env::var("NY_ALPHA_TAIL_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(depth) => depth.min(6),
        None => 3,
    }
}

/// SPSA refresh iteration count. Env `NY_ALPHA_TAIL_ITERS`, default 20
/// (the root optimizer's own count — ≤ 32 scalars converge well inside it),
/// clamped to `0..=512`. `0` disables the refresh (baseline pass only).
pub(crate) fn alpha_tail_iters() -> usize {
    match std::env::var("NY_ALPHA_TAIL_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(iters) => iters.min(512),
        None => 20,
    }
}

// ---------------------------------------------------------------------------
// Corner-certify-and-repair gadget (design §5.4).
// ---------------------------------------------------------------------------

/// NaN-PROPAGATING min (IEEE `f64::min` absorbs NaN — checklist I-A4 hazard).
#[inline]
fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// Certify-or-repair a candidate LOWER plane `z >= alpha*x + beta*y + nu` of
/// the bilinear `z = x*y` over `[xl,xu] x [yl,yu]`.
///
/// The gap `d(x,y) = x*y - alpha*x - beta*y - nu` is bilinear, so its minimum
/// over the box is attained at one of the 4 corners. Each corner gets a
/// certified (directed-rounded DOWN) lower bound on `d`; if the minimum `m`
/// is negative the intercept is repaired to `nu' = next_down(nu + m)`:
/// `d'(x,y) = d(x,y) + (nu - nu') >= m + (-m) = 0`, so the repaired plane is
/// rigorously valid whatever rounding produced `(alpha, beta, nu)`.
///
/// Returns `None` (caller must poison/decline) when any corner evaluation or
/// the repaired intercept is non-finite.
pub(crate) fn repair_lower_plane(
    alpha: f64,
    beta: f64,
    nu: f64,
    xl: f64,
    xu: f64,
    yl: f64,
    yu: f64,
) -> Option<f64> {
    let mut m = f64::INFINITY;
    for &x in &[xl, xu] {
        for &y in &[yl, yu] {
            // Certified LOWER bound of d = x*y - alpha*x - beta*y - nu:
            // round the positive term down, the subtracted terms up, and every
            // intermediate subtraction down.
            let t1 = next_down(x * y);
            let t2 = next_up(alpha * x);
            let t3 = next_up(beta * y);
            let d = next_down(next_down(next_down(t1 - t2) - t3) - nu);
            m = nan_min(m, d);
        }
    }
    if !m.is_finite() {
        return None;
    }
    let repaired = if m >= 0.0 { nu } else { next_down(nu + m) };
    repaired.is_finite().then_some(repaired)
}

/// Mirror of [`repair_lower_plane`] for an UPPER plane
/// `z <= alpha*x + beta*y + nu`: the gap `e = alpha*x + beta*y + nu - x*y`
/// gets certified corner lower bounds (plane terms rounded down, the product
/// up); a negative minimum `m` repairs `nu' = next_up(nu - m)` so
/// `e' = e + (nu' - nu) >= m + (-m) = 0`.
pub(crate) fn repair_upper_plane(
    alpha: f64,
    beta: f64,
    nu: f64,
    xl: f64,
    xu: f64,
    yl: f64,
    yu: f64,
) -> Option<f64> {
    let mut m = f64::INFINITY;
    for &x in &[xl, xu] {
        for &y in &[yl, yu] {
            let t1 = next_up(x * y);
            let t2 = next_down(alpha * x);
            let t3 = next_down(beta * y);
            let e = next_down(next_down(next_down(t2 + t3) + nu) - t1);
            m = nan_min(m, e);
        }
    }
    if !m.is_finite() {
        return None;
    }
    let repaired = if m >= 0.0 { nu } else { next_up(nu - m) };
    repaired.is_finite().then_some(repaired)
}

// ---------------------------------------------------------------------------
// Op support (KEEP IN SYNC with `RowWalk::backward_node` below — every arm
// that can succeed must be listed here; anything else fails closed).
// ---------------------------------------------------------------------------

fn f64_tail_supports_layer(layer: &Layer) -> bool {
    match layer {
        // #f64-tail-conv: Conv2d is the reason this walker could never run on the
        // ResNet benchmarks (cifar100 / tinyimagenet / yolo) — every one of them
        // is a conv DAG, so the walk declined at the first conv and the certified
        // f64 replay was unreachable exactly where it is worth the most.
        Layer::Conv2d(conv) => conv.dilation == (1, 1) && conv.input_shape.is_some(),
        Layer::Linear(_)
        | Layer::ReLU(_)
        | Layer::MulBinary(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::AddConstant(_)
        | Layer::SubConstant(_)
        | Layer::MulConstant(_)
        | Layer::DivConstant(_)
        | Layer::Concat(_)
        | Layer::Slice(_)
        | Layer::ReduceSum(_)
        | Layer::Flatten(_)
        | Layer::Reshape(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        // #nn4sys-dual arms: certified interval substitution (see the walk).
        | Layer::Sigmoid(_)
        | Layer::Div(_) => true,
        Layer::Gather(gather) => gather.constant_indices().is_some(),
        _ => false,
    }
}

/// Certified lower endpoint of `sigmoid` over `x` (#nn4sys-dual increment 1).
///
/// `sigma(x) = 1/(1+e^{-x})`. Directed per step: `e^{-x}` rounded UP (libm exp
/// is faithful, <= 1 ulp, so one `next_up` certifies), denominator UP, the
/// division rounded DOWN with one extra `next_down` covering the IEEE divide
/// rounding. Clamped into `[0, 1]` (always valid: `0 <= sigma <= 1`).
fn sigmoid_round_down(x: f64) -> f64 {
    let e = next_up((-x).exp());
    let denom = next_up(1.0 + e);
    if !denom.is_finite() || denom <= 0.0 {
        // e^{-x} overflowed (x very negative): sigma is tiny; 0 is a valid
        // lower endpoint.
        return 0.0;
    }
    next_down(next_down(1.0 / denom)).clamp(0.0, 1.0)
}

/// Certified upper endpoint of `sigmoid` over `x` — mirror of
/// [`sigmoid_round_down`], all roundings flipped.
fn sigmoid_round_up(x: f64) -> f64 {
    let e = next_down((-x).exp()).max(0.0);
    let denom = next_down(1.0 + e).max(1.0);
    next_up(next_up(1.0 / denom)).clamp(0.0, 1.0)
}

/// Whether the #nn4sys-dual seam debug env is set (stage telemetry).
pub(crate) fn tail_debug() -> bool {
    std::env::var("NY_DUAL_F64_TAIL_DEBUG").ok().as_deref() == Some("1")
}

/// Debug probe for the seam's UNSUPPORTED census (#nn4sys-dual telemetry).
pub(crate) fn f64_tail_supports_layer_probe(layer: &Layer) -> bool {
    f64_tail_supports_layer(layer)
}

/// Whether every output-ancestor node of `graph` is in the f64-tail op set.
pub(crate) fn graph_supports_f64_tail(graph: &GraphNetwork) -> bool {
    match graph.output_ancestors() {
        Ok(needed) => needed.iter().all(|name| {
            graph
                .node(name)
                .is_some_and(|node| f64_tail_supports_layer(node.layer()))
        }),
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// The certified backward walk.
// ---------------------------------------------------------------------------

/// Internal decline signal: the pass cannot certify this graph/domain (shape
/// or op surprise). The whole pass returns `Unsupported` — never a bound.
struct Decline;

type WalkResult<T> = Result<T, Decline>;

/// Shared per-domain context for the 39 per-row walks.
struct TailCtx<'a> {
    graph: &'a GraphNetwork,
    /// Exec order up to and including the output node.
    exec_prefix: Vec<&'a str>,
    /// Ancestors of the output node (only these are walked).
    needed: HashSet<&'a str>,
    /// f32-sound per-node output boxes, exact-widened to f64, flattened.
    anchors: HashMap<&'a str, (Vec<f64>, Vec<f64>)>,
    /// Per-node output shapes (for index-movement ops / broadcast maps).
    shapes: HashMap<&'a str, Vec<usize>>,
    input_lo: Vec<f64>,
    input_hi: Vec<f64>,
    input_shape: Vec<usize>,
}

impl TailCtx<'_> {
    fn width_of(&self, name: &str) -> WalkResult<usize> {
        if name == NETWORK_INPUT {
            return Ok(self.input_lo.len());
        }
        self.anchors
            .get(name)
            .map(|(lo, _)| lo.len())
            .ok_or(Decline)
    }

    fn shape_of(&self, name: &str) -> WalkResult<&[usize]> {
        if name == NETWORK_INPUT {
            return Ok(&self.input_shape);
        }
        self.shapes.get(name).map(|s| s.as_slice()).ok_or(Decline)
    }

    fn anchor_of(&self, name: &str) -> WalkResult<(&[f64], &[f64])> {
        if name == NETWORK_INPUT {
            return Ok((&self.input_lo, &self.input_hi));
        }
        self.anchors
            .get(name)
            .map(|(lo, hi)| (lo.as_slice(), hi.as_slice()))
            .ok_or(Decline)
    }
}

/// Signed coefficients + ABS shadow for one node's outputs.
struct NodeVec {
    a: Vec<f64>,
    s: Vec<f64>,
}

/// One spec row's backward walk state.
struct RowWalk {
    coeffs: HashMap<String, NodeVec>,
    input_a: Vec<f64>,
    input_s: Vec<f64>,
    bias: f64,
    bias_abs: f64,
    /// Node-range-weighted step-error units (multiplied by `gamma_n` at the
    /// end): every push of coefficients onto a node discharges
    /// `Σ_j |step abs|_j · max|node_j|` here — the telescoped-chain error
    /// accounting (each backward step is exact on the STORED coefficients,
    /// whose sign drove the relaxation choice; the step's own rounding is a
    /// coefficient-space error that must be weighted by the range of the node
    /// the coefficient lives on, mirroring the f32 lane's per-node
    /// `fold_coeff_err_over_box_eager` discipline, checklist I-C3). The
    /// design-§5.5 input-concretized `γ·S` term alone would under-count
    /// intermediate-node roundings; this strengthens it.
    env_units: f64,
    /// Conservative global rounding-depth counter: an upper bound on the
    /// number of RN roundings ANY single step chain accumulates (products,
    /// contraction adds, merge adds, bias adds, shadow arithmetic).
    /// Deliberately over-counted (`+16` slop per node) — at lsnc scales even a
    /// 10x overshoot moves the envelope by < 1e-12 (gamma is ~linear in n).
    n_acc: usize,
    /// Latched when any coefficient/bias/anchor goes non-finite: the row
    /// poisons to `-inf` (fail closed), never to a bound.
    poisoned: bool,
}

impl RowWalk {
    fn new(n_inputs: usize) -> Self {
        Self {
            coeffs: HashMap::new(),
            input_a: vec![0.0; n_inputs],
            input_s: vec![0.0; n_inputs],
            bias: 0.0,
            bias_abs: 0.0,
            env_units: 0.0,
            n_acc: 0,
            poisoned: false,
        }
    }

    /// Accumulate `(add_a, add_s)` onto `dst`'s coefficient/shadow vectors,
    /// discharging the step's rounding budget against `dst`'s value range
    /// (see [`RowWalk::env_units`]). `add_s[j]` upper-bounds the abs-value
    /// computation of this step's coefficient j, so `gamma · add_s[j]` bounds
    /// the step's rounding (products + contraction adds + merge adds), and
    /// weighting by `max|dst_j|` converts it to a sound value-space error.
    fn push(
        &mut self,
        ctx: &TailCtx<'_>,
        dst: &str,
        add_a: Vec<f64>,
        add_s: Vec<f64>,
    ) -> WalkResult<()> {
        debug_assert_eq!(add_a.len(), add_s.len());
        let (dst_lo, dst_hi) = ctx.anchor_of(dst)?;
        if add_a.len() != dst_lo.len() {
            return Err(Decline);
        }
        for (j, &sv) in add_s.iter().enumerate() {
            if sv != 0.0 {
                self.env_units += sv * dst_lo[j].abs().max(dst_hi[j].abs());
            }
        }
        if dst == NETWORK_INPUT {
            for ((t, v), (ts, vs)) in self
                .input_a
                .iter_mut()
                .zip(add_a)
                .zip(self.input_s.iter_mut().zip(add_s))
            {
                *t += v;
                *ts += vs;
            }
            return Ok(());
        }
        let width = dst_lo.len();
        let entry = self
            .coeffs
            .entry(dst.to_string())
            .or_insert_with(|| NodeVec {
                a: vec![0.0; width],
                s: vec![0.0; width],
            });
        for ((t, v), (ts, vs)) in entry
            .a
            .iter_mut()
            .zip(add_a)
            .zip(entry.s.iter_mut().zip(add_s))
        {
            *t += v;
            *ts += vs;
        }
        Ok(())
    }

    /// Add a bias contribution `coeff * value` (RN; covered by the envelope).
    #[inline]
    fn add_bias(&mut self, coeff: f64, shadow: f64, value: f64) {
        self.bias += coeff * value;
        self.bias_abs += shadow * value.abs();
    }
}

/// Certified f64 lower bound for one spec row (design §5), or:
/// - `Ok(-inf)` — row poisoned (non-finite somewhere): sound, never verifies;
/// - `Err(Decline)` — op/shape surprise: the whole pass must return
///   `Unsupported`.
///
/// `mul_binary_alphas` selects WHICH sound MulBinary relaxation the walk
/// certifies (interpolated facets when present, plane-select fallback when
/// absent) — a parameter rather than ctx state so the alpha-tail refresh can
/// evaluate candidate maps against one shared ctx.
fn certified_row_lower(
    ctx: &TailCtx<'_>,
    spec_row: &[f64],
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
) -> WalkResult<f64> {
    let output_name = ctx.graph.output_name();
    let out_width = ctx.width_of(output_name)?;
    if spec_row.len() != out_width {
        return Err(Decline);
    }

    let mut walk = RowWalk::new(ctx.input_lo.len());
    walk.coeffs.insert(
        output_name.to_string(),
        NodeVec {
            a: spec_row.to_vec(),
            s: spec_row.iter().map(|v| v.abs()).collect(),
        },
    );

    for &name in ctx.exec_prefix.iter().rev() {
        if !ctx.needed.contains(name) {
            continue;
        }
        let Some(node_vec) = walk.coeffs.remove(name) else {
            continue;
        };
        if node_vec.a.iter().any(|v| !v.is_finite()) || node_vec.s.iter().any(|v| !v.is_finite()) {
            walk.poisoned = true;
            break;
        }
        let node = ctx.graph.node(name).ok_or(Decline)?;
        backward_node(ctx, &mut walk, name, node, node_vec, mul_binary_alphas)?;
        if walk.poisoned {
            break;
        }
    }

    if walk.poisoned {
        return Ok(f64::NEG_INFINITY);
    }
    // Every non-input node's coefficients must have been consumed; leftovers
    // mean a node outside the walked prefix carried weight — decline.
    if !walk.coeffs.is_empty() {
        return Err(Decline);
    }

    // Concretize over the exact input box (design §5.5) + ABS shadow.
    let mut sum_l = walk.bias;
    let mut s_conc = 0.0_f64;
    for (j, (&a, &s)) in walk.input_a.iter().zip(walk.input_s.iter()).enumerate() {
        let (lo, hi) = (ctx.input_lo[j], ctx.input_hi[j]);
        sum_l += if a >= 0.0 { a * lo } else { a * hi };
        s_conc += s * lo.abs().max(hi.abs());
    }
    walk.n_acc += ctx.input_lo.len() + 16;

    if !sum_l.is_finite() || !s_conc.is_finite() || !walk.env_units.is_finite() {
        return Ok(f64::NEG_INFINITY);
    }

    // Envelope, three sound parts under one over-counted gamma:
    //  - `env_units`: per-push step roundings discharged against the RANGE of
    //    the node the coefficients land on (telescoped-chain accounting; the
    //    relaxation choices are valid for the STORED coefficients' own signs,
    //    so only each step's own rounding needs charging — checklist I-C3);
    //  - `bias_abs`: the bias chain's adds/products are value-space errors,
    //    `gamma · Σ|contributions|` covers every summation order (Higham §3.5);
    //  - `s_conc`: the final concretize dot's own roundings over the input box.
    // The env expressions are themselves RN-computed nonneg sums whose
    // relative under-statement is <= gamma; `certify_up` with rel = 2*gamma
    // dominates the inverse factor plus the final multiply's rounding.
    let gamma = gamma_n(walk.n_acc + 256);
    // NaN-closed saturation guard: gamma_n saturates to 1.0 for absurd n;
    // anything not comfortably < 0.1 poisons the row (fail closed).
    if gamma >= 0.1 || !gamma.is_finite() {
        return Ok(f64::NEG_INFINITY);
    }
    let env_base = walk.env_units + walk.bias_abs + s_conc;
    let rel = (2.0 * gamma).clamp(2.0 * UNIT, 0.25);
    let env = certify_up(gamma * env_base, rel);
    let l_cert = next_down(sum_l - env);
    if l_cert.is_nan() {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(l_cert)
}

/// Row-major flat index of `multi` in `shape`.
#[inline]
fn flat_index(multi: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0usize;
    for (m, s) in multi.iter().zip(shape.iter()) {
        idx = idx * s + m;
    }
    idx
}

/// Decompose flat index `flat` (row-major) into `shape` coordinates.
fn multi_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut multi = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        let s = shape[d].max(1);
        multi[d] = flat % s;
        flat /= s;
    }
    multi
}

/// Per-node backward dispatch: pushes this node's coefficients back to its
/// parents through the op's (certified) linear relaxation. Mirrors the f32
/// lane's relaxation choices (design §5.3).
fn backward_node(
    ctx: &TailCtx<'_>,
    walk: &mut RowWalk,
    name: &str,
    node: &crate::network::core::GraphNode,
    node_vec: NodeVec,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
) -> WalkResult<()> {
    let r = backward_node_inner(ctx, walk, name, node, node_vec, mul_binary_alphas);
    if r.is_err() && tail_debug() {
        eprintln!(
            "[dual-f64-tail-stage] DECLINE at node '{name}' ({})",
            node.layer().layer_type()
        );
    }
    r
}

fn backward_node_inner(
    ctx: &TailCtx<'_>,
    walk: &mut RowWalk,
    name: &str,
    node: &crate::network::core::GraphNode,
    node_vec: NodeVec,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
) -> WalkResult<()> {
    let inputs = node.inputs();
    let a = node_vec.a;
    let s = node_vec.s;
    let width = a.len();
    // Generous global rounding-depth bump: covers this node's per-term
    // product, its contraction adds (<= width), merge adds, and bias adds.
    walk.n_acc += 2 * width + 16;

    let unary = || -> WalkResult<&str> { inputs.first().map(|s| s.as_str()).ok_or(Decline) };

    match node.layer() {
        // ---- exact index movement -------------------------------------------------
        Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
            let parent = unary()?;
            if ctx.width_of(parent)? != width {
                return Err(Decline);
            }
            walk.push(ctx, parent, a, s)
        }
        Layer::Slice(slice) => {
            let parent = unary()?;
            let parent_shape = ctx.shape_of(parent)?.to_vec();
            let (axis, start, end) = slice.resolved_range(&parent_shape).map_err(|_| Decline)?;
            // Output shape: parent with the sliced axis narrowed.
            let mut out_shape = parent_shape.clone();
            out_shape[axis] = end.saturating_sub(start);
            if out_shape.iter().product::<usize>() != width {
                return Err(Decline);
            }
            let parent_width = ctx.width_of(parent)?;
            let mut back_a = vec![0.0f64; parent_width];
            let mut back_s = vec![0.0f64; parent_width];
            for k in 0..width {
                let mut multi = multi_index(k, &out_shape);
                multi[axis] += start;
                let p = flat_index(&multi, &parent_shape);
                if p >= parent_width {
                    return Err(Decline);
                }
                back_a[p] += a[k];
                back_s[p] += s[k];
            }
            walk.push(ctx, parent, back_a, back_s)
        }
        Layer::Gather(gather) => {
            let parent = unary()?;
            let parent_shape = ctx.shape_of(parent)?.to_vec();
            let indices = gather.constant_indices().ok_or(Decline)?;
            let ndim = parent_shape.len();
            let axis_raw = gather.axis_raw();
            let axis = if axis_raw < 0 {
                axis_raw + ndim as i64
            } else {
                axis_raw
            };
            if axis < 0 || axis as usize >= ndim {
                return Err(Decline);
            }
            let axis = axis as usize;
            let axis_len = parent_shape[axis] as i64;
            let mut out_shape: Vec<usize> = parent_shape[..axis].to_vec();
            out_shape.extend_from_slice(indices.shape());
            out_shape.extend_from_slice(&parent_shape[axis + 1..]);
            if out_shape.iter().product::<usize>() != width {
                return Err(Decline);
            }
            let idx_flat: Vec<i64> = indices.iter().copied().collect();
            let parent_width = ctx.width_of(parent)?;
            let mut back_a = vec![0.0f64; parent_width];
            let mut back_s = vec![0.0f64; parent_width];
            let idx_rank = indices.shape().len();
            for k in 0..width {
                let multi = multi_index(k, &out_shape);
                // multi = pre[..axis] ++ idx coords ++ post
                let idx_coords = &multi[axis..axis + idx_rank];
                let idx_pos = flat_index(idx_coords, indices.shape());
                let raw = *idx_flat.get(idx_pos).ok_or(Decline)?;
                let resolved = if raw < 0 { raw + axis_len } else { raw };
                if resolved < 0 || resolved >= axis_len {
                    return Err(Decline);
                }
                let mut parent_multi: Vec<usize> = multi[..axis].to_vec();
                parent_multi.push(resolved as usize);
                parent_multi.extend_from_slice(&multi[axis + idx_rank..]);
                let p = flat_index(&parent_multi, &parent_shape);
                if p >= parent_width {
                    return Err(Decline);
                }
                back_a[p] += a[k];
                back_s[p] += s[k];
            }
            walk.push(ctx, parent, back_a, back_s)
        }
        Layer::Concat(concat) => backward_concat(ctx, walk, name, concat, inputs, &a, &s),
        Layer::ReduceSum(reduce) => {
            let parent = unary()?;
            let parent_shape = ctx.shape_of(parent)?.to_vec();
            let ndim = parent_shape.len();
            let mut axes: Vec<usize> = if reduce.axes.is_empty() {
                (0..ndim).collect()
            } else {
                let mut resolved = Vec::with_capacity(reduce.axes.len());
                for &ax in &reduce.axes {
                    let r = if ax < 0 { ax + ndim as i64 } else { ax };
                    if r < 0 || r as usize >= ndim {
                        return Err(Decline);
                    }
                    resolved.push(r as usize);
                }
                resolved
            };
            axes.sort_unstable();
            axes.dedup();
            let mut out_shape = parent_shape.clone();
            if reduce.keepdims {
                for &ax in &axes {
                    out_shape[ax] = 1;
                }
            } else {
                for &ax in axes.iter().rev() {
                    out_shape.remove(ax);
                }
            }
            if out_shape.iter().product::<usize>().max(1) != width {
                return Err(Decline);
            }
            let parent_width = ctx.width_of(parent)?;
            let mut back_a = vec![0.0f64; parent_width];
            let mut back_s = vec![0.0f64; parent_width];
            for p in 0..parent_width {
                let pm = multi_index(p, &parent_shape);
                let om: Vec<usize> = if reduce.keepdims {
                    pm.iter()
                        .enumerate()
                        .map(|(d, &c)| if axes.contains(&d) { 0 } else { c })
                        .collect()
                } else {
                    pm.iter()
                        .enumerate()
                        .filter(|(d, _)| !axes.contains(d))
                        .map(|(_, &c)| c)
                        .collect()
                };
                let o = flat_index(&om, &out_shape);
                if o >= width {
                    return Err(Decline);
                }
                back_a[p] = a[o];
                back_s[p] = s[o];
            }
            walk.push(ctx, parent, back_a, back_s)
        }

        // ---- exact linear ---------------------------------------------------------
        Layer::Sub(_) => {
            let (p0, p1) = match (inputs.first(), inputs.get(1)) {
                (Some(x), Some(y)) => (x.as_str(), y.as_str()),
                _ => return Err(Decline),
            };
            if ctx.width_of(p0)? != width || ctx.width_of(p1)? != width {
                return Err(Decline);
            }
            let neg: Vec<f64> = a.iter().map(|v| -v).collect();
            walk.push(ctx, p0, a, s.clone())?;
            walk.push(ctx, p1, neg, s)
        }
        Layer::Add(_) => {
            let (p0, p1) = match (inputs.first(), inputs.get(1)) {
                (Some(x), Some(y)) => (x.as_str(), y.as_str()),
                _ => return Err(Decline),
            };
            if ctx.width_of(p0)? != width || ctx.width_of(p1)? != width {
                return Err(Decline);
            }
            walk.push(ctx, p0, a.clone(), s.clone())?;
            walk.push(ctx, p1, a, s)
        }
        Layer::AddConstant(add) => {
            let parent = unary()?;
            if ctx.width_of(parent)? != width {
                return Err(Decline);
            }
            let out_shape = ctx.shape_of(name)?.to_vec();
            let cmap = constant_index_map(&out_shape, add.constant().shape(), width)?;
            let cflat = crate::contiguous_flat_slice(add.constant());
            for j in 0..width {
                let c = f64::from(cflat[cmap[j]]);
                if !c.is_finite() {
                    walk.poisoned = true;
                    return Ok(());
                }
                walk.add_bias(a[j], s[j], c);
            }
            walk.push(ctx, parent, a, s)
        }
        Layer::SubConstant(sub) => {
            let parent = unary()?;
            if ctx.width_of(parent)? != width {
                return Err(Decline);
            }
            let out_shape = ctx.shape_of(name)?.to_vec();
            let cmap = constant_index_map(&out_shape, sub.constant().shape(), width)?;
            let cflat = crate::contiguous_flat_slice(sub.constant());
            if sub.reverse {
                // y = c - x: bias += a·c, coefficients negate (exact).
                for j in 0..width {
                    let c = f64::from(cflat[cmap[j]]);
                    if !c.is_finite() {
                        walk.poisoned = true;
                        return Ok(());
                    }
                    walk.add_bias(a[j], s[j], c);
                }
                let neg: Vec<f64> = a.iter().map(|v| -v).collect();
                walk.push(ctx, parent, neg, s)
            } else {
                // y = x - c: bias += a·(-c).
                for j in 0..width {
                    let c = f64::from(cflat[cmap[j]]);
                    if !c.is_finite() {
                        walk.poisoned = true;
                        return Ok(());
                    }
                    walk.add_bias(a[j], s[j], -c);
                }
                walk.push(ctx, parent, a, s)
            }
        }
        Layer::MulConstant(mul) => {
            let parent = unary()?;
            if ctx.width_of(parent)? != width {
                return Err(Decline);
            }
            let out_shape = ctx.shape_of(name)?.to_vec();
            let cmap = constant_index_map(&out_shape, mul.constant().shape(), width)?;
            let cflat = crate::contiguous_flat_slice(mul.constant());
            let mut back_a = vec![0.0f64; width];
            let mut back_s = vec![0.0f64; width];
            for j in 0..width {
                let c = f64::from(cflat[cmap[j]]);
                if !c.is_finite() {
                    walk.poisoned = true;
                    return Ok(());
                }
                back_a[j] = a[j] * c;
                back_s[j] = s[j] * c.abs();
            }
            walk.push(ctx, parent, back_a, back_s)
        }
        Layer::DivConstant(div) => {
            let parent = unary()?;
            if ctx.width_of(parent)? != width {
                return Err(Decline);
            }
            let out_shape = ctx.shape_of(name)?.to_vec();
            let cmap = constant_index_map(&out_shape, div.constant().shape(), width)?;
            let cflat = crate::contiguous_flat_slice(div.constant());
            let mut back_a = vec![0.0f64; width];
            let mut back_s = vec![0.0f64; width];
            for j in 0..width {
                let c = f64::from(cflat[cmap[j]]);
                if !c.is_finite() || c == 0.0 {
                    return Err(Decline);
                }
                back_a[j] = a[j] / c;
                back_s[j] = s[j] / c.abs();
            }
            walk.push(ctx, parent, back_a, back_s)
        }
        // #f64-tail-conv: one-row conv backward in f64. The signed row `a` is
        // transposed through the kernel and the radius row `s` through |kernel|,
        // exactly as the Linear arm does with `w` / `|w|` — a Conv2d IS a linear
        // map, just with shared weights, so the same certified-envelope accounting
        // applies verbatim.
        //
        // This replays ONE margin row end-to-end in f64, which is what makes it
        // worth having: the patches route re-bounds its certified error at every
        // layer and multiplies it by the kernel norm, so the error compounds
        // geometrically (measured: 93-100% of the emitted error is that carry).
        // A single end-to-end f64 replay has NO carry — its error is measured
        // once, at the end.
        Layer::Conv2d(conv) => {
            let parent = unary()?;
            let p_width = ctx.width_of(parent)?;
            let (in_h, in_w) = conv.input_shape.ok_or(Decline)?;
            if conv.dilation != (1, 1) {
                return Err(Decline);
            }
            let ks = conv.kernel.shape();
            if ks.len() != 4 {
                return Err(Decline);
            }
            let (out_c, in_c_per_group, kh, kw) = (ks[0], ks[1], ks[2], ks[3]);
            let groups = conv.groups;
            if groups == 0 || !out_c.is_multiple_of(groups) {
                return Err(Decline);
            }
            let (sh, sw) = conv.stride;
            let (ph, pw) = conv.padding;
            if sh == 0 || sw == 0 {
                return Err(Decline);
            }
            let out_h = (in_h + 2 * ph).checked_sub(kh).ok_or(Decline)? / sh + 1;
            let out_w = (in_w + 2 * pw).checked_sub(kw).ok_or(Decline)? / sw + 1;
            let in_c = in_c_per_group * groups;
            let ohw = out_h * out_w;
            let ihw = in_h * in_w;
            if width != out_c * ohw || p_width != in_c * ihw {
                return Err(Decline);
            }
            let out_c_per_group = out_c / groups;

            let mut back_a = vec![0.0f64; p_width];
            let mut back_s = vec![0.0f64; p_width];
            for g in 0..groups {
                for oc_local in 0..out_c_per_group {
                    let oc = g * out_c_per_group + oc_local;
                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let o = oc * ohw + oy * out_w + ox;
                            let (ai, si) = (a[o], s[o]);
                            if ai == 0.0 && si == 0.0 {
                                continue;
                            }
                            for ic_local in 0..in_c_per_group {
                                let ic = g * in_c_per_group + ic_local;
                                for ki in 0..kh {
                                    let iy = (oy * sh + ki) as isize - ph as isize;
                                    if iy < 0 || iy >= in_h as isize {
                                        continue;
                                    }
                                    for kj in 0..kw {
                                        let ix = (ox * sw + kj) as isize - pw as isize;
                                        if ix < 0 || ix >= in_w as isize {
                                            continue;
                                        }
                                        let w = f64::from(conv.kernel[[oc, ic_local, ki, kj]]);
                                        if !w.is_finite() {
                                            walk.poisoned = true;
                                            return Ok(());
                                        }
                                        let idx = ic * ihw + iy as usize * in_w + ix as usize;
                                        back_a[idx] += ai * w;
                                        back_s[idx] += si * w.abs();
                                    }
                                }
                            }
                            if let Some(bias) = conv.bias.as_ref() {
                                let bb = f64::from(bias[oc]);
                                if !bb.is_finite() {
                                    walk.poisoned = true;
                                    return Ok(());
                                }
                                walk.add_bias(ai, si, bb);
                            }
                        }
                    }
                }
            }
            walk.push(ctx, parent, back_a, back_s)
        }

        Layer::Linear(lin) => {
            let parent = unary()?;
            let (out_dim, in_dim) = lin.weight.dim();
            let p_width = ctx.width_of(parent)?;
            // Batched matmul lowering (#nn4sys-dual): the mscn set MatMuls
            // apply the weight on the LAST axis over shared leading dims
            // (batch = 1 is the plain Linear). Airtight shape guard: last
            // dims must equal (out_dim, in_dim) and every leading dim must
            // match exactly — anything else declines.
            if out_dim == 0 || in_dim == 0 || !width.is_multiple_of(out_dim) {
                return Err(Decline);
            }
            let batch = width / out_dim;
            if p_width != batch * in_dim {
                return Err(Decline);
            }
            if batch > 1 {
                let out_shape = ctx.shape_of(name)?;
                let p_shape = ctx.shape_of(parent)?;
                let shapes_ok = out_shape.last() == Some(&out_dim)
                    && p_shape.last() == Some(&in_dim)
                    && out_shape.len() == p_shape.len()
                    && out_shape[..out_shape.len() - 1] == p_shape[..p_shape.len() - 1];
                if !shapes_ok {
                    return Err(Decline);
                }
            }
            let mut back_a = vec![0.0f64; p_width];
            let mut back_s = vec![0.0f64; p_width];
            for b in 0..batch {
                for o in 0..out_dim {
                    let (ai, si) = (a[b * out_dim + o], s[b * out_dim + o]);
                    if ai == 0.0 && si == 0.0 {
                        continue;
                    }
                    for j in 0..in_dim {
                        let w = f64::from(lin.weight[[o, j]]);
                        if !w.is_finite() {
                            walk.poisoned = true;
                            return Ok(());
                        }
                        back_a[b * in_dim + j] += ai * w;
                        back_s[b * in_dim + j] += si * w.abs();
                    }
                    if let Some(bias) = lin.bias.as_ref() {
                        let bb = f64::from(bias[o]);
                        if !bb.is_finite() {
                            walk.poisoned = true;
                            return Ok(());
                        }
                        walk.add_bias(ai, si, bb);
                    }
                }
            }
            walk.push(ctx, parent, back_a, back_s)
        }

        // ---- relaxations (valid-by-construction, design §5.3/§5.4) ---------------
        Layer::ReLU(_) => {
            let parent = unary()?;
            let (pl, pu) = ctx.anchor_of(parent)?;
            if pl.len() != width {
                return Err(Decline);
            }
            let mut back_a = vec![0.0f64; width];
            let mut back_s = vec![0.0f64; width];
            for j in 0..width {
                let (aj, sj) = (a[j], s[j]);
                if aj == 0.0 && sj == 0.0 {
                    continue;
                }
                let (l, u) = (pl[j], pu[j]);
                if !l.is_finite() || !u.is_finite() {
                    walk.poisoned = true;
                    return Ok(());
                }
                if l >= 0.0 {
                    back_a[j] = aj;
                    back_s[j] = sj;
                } else if u <= 0.0 {
                    // relu(x) = 0: coefficient dies (exact).
                } else if aj >= 0.0 {
                    // Lower relaxation: slope alpha ∈ {0, 1} by the SAME
                    // adaptive rule as the f32 lane (`u > -l`) — exact.
                    if u > -l {
                        back_a[j] = aj;
                        back_s[j] = sj;
                    }
                } else {
                    // Upper relaxation (chord): slope = RN(u/(u-l)); intercept
                    // certified from BOTH endpoint conditions with directed
                    // rounding (validity over a convex piece needs only the
                    // two endpoint checks):
                    //   at x=l: beta >= -slope*l ; at x=u: beta >= u - slope*u.
                    let slope = u / (u - l);
                    if !slope.is_finite() {
                        walk.poisoned = true;
                        return Ok(());
                    }
                    let c1 = -next_down(slope * l);
                    let c2 = next_up(u - next_down(slope * u));
                    let beta = c1.max(c2);
                    if !beta.is_finite() {
                        walk.poisoned = true;
                        return Ok(());
                    }
                    back_a[j] = aj * slope;
                    back_s[j] = sj * slope.abs();
                    walk.add_bias(aj, sj, beta);
                }
            }
            walk.push(ctx, parent, back_a, back_s)
        }
        Layer::MulBinary(_) => {
            backward_mul_binary(ctx, walk, name, inputs, &a, &s, mul_binary_alphas)
        }

        // ---- #nn4sys-dual arms: certified interval substitution -----------------
        // Increment 1 for the mscn dual DAG (Sigmoid + tensor-Div, both absent
        // from the f32 lane's relaxation set here). Each coefficient concretizes
        // AT this node: the sign-selected certified endpoint of the node's range
        // enters the bias (zero-slope planes — valid for the lower side by
        // monotonicity/corner analysis below), and nothing propagates past the
        // node. Chord/tangent planes are the named upgrade if the A/B against
        // the genuine 0/22 dual baseline shows this substitution is the
        // residual looseness.
        Layer::Sigmoid(_) => {
            let parent = unary()?;
            let (pl, pu) = ctx.anchor_of(parent)?;
            if pl.len() != width {
                return Err(Decline);
            }
            let back_a = vec![0.0f64; width];
            let back_s = vec![0.0f64; width];
            for j in 0..width {
                let (aj, sj) = (a[j], s[j]);
                if aj == 0.0 && sj == 0.0 {
                    continue;
                }
                let (l, u) = (pl[j], pu[j]);
                if !l.is_finite() || !u.is_finite() {
                    walk.poisoned = true;
                    return Ok(());
                }
                // sigma is monotone increasing: over [l, u] the range is
                // [sigma(l), sigma(u)], endpoints certified by directed
                // rounding. aj >= 0 wants the lower endpoint, aj < 0 the upper.
                let c = if aj >= 0.0 {
                    sigmoid_round_down(l)
                } else {
                    sigmoid_round_up(u)
                };
                if !c.is_finite() {
                    walk.poisoned = true;
                    return Ok(());
                }
                walk.add_bias(aj, sj, c);
            }
            walk.push(ctx, parent, back_a, back_s)
        }
        Layer::Div(_) => {
            let (px, pd) = match (inputs.first(), inputs.get(1)) {
                (Some(x), Some(d)) => (x.as_str(), d.as_str()),
                _ => return Err(Decline),
            };
            // Broadcast-aware (the mscn dual divides set features by a
            // broadcast set-count): flat index maps from the output shape
            // onto each parent's shape, same mechanism as
            // `constant_index_map`.
            let out_shape = ctx.shape_of(name)?.to_vec();
            let xmap = constant_index_map(&out_shape, ctx.shape_of(px)?, width)?;
            let dmap = constant_index_map(&out_shape, ctx.shape_of(pd)?, width)?;
            let x_width = ctx.width_of(px)?;
            let d_width = ctx.width_of(pd)?;
            if xmap.iter().any(|&m| m >= x_width) || dmap.iter().any(|&m| m >= d_width) {
                return Err(Decline);
            }
            let (xl, xu) = ctx.anchor_of(px)?;
            let (dl, du) = ctx.anchor_of(pd)?;
            let back_x_a = vec![0.0f64; x_width];
            let back_x_s = vec![0.0f64; x_width];
            let back_d_a = vec![0.0f64; d_width];
            let back_d_s = vec![0.0f64; d_width];
            for j in 0..width {
                let (aj, sj) = (a[j], s[j]);
                if aj == 0.0 && sj == 0.0 {
                    continue;
                }
                let (lx, ux, ld, ud) = (xl[xmap[j]], xu[xmap[j]], dl[dmap[j]], du[dmap[j]]);
                if !lx.is_finite() || !ux.is_finite() || !ld.is_finite() || !ud.is_finite() {
                    walk.poisoned = true;
                    return Ok(());
                }
                if ld <= 0.0 && ud >= 0.0 {
                    // Denominator interval spans zero: x/d is unbounded on the
                    // box — no valid finite plane exists. Decline the pass.
                    return Err(Decline);
                }
                // d is sign-definite on the box, so x/d is monotone in each
                // argument separately and its extrema over the rectangle sit at
                // the four corners; IEEE division is correctly rounded, one
                // directed step certifies each corner.
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for &(cx, cd) in &[(lx, ld), (lx, ud), (ux, ld), (ux, ud)] {
                    let q = cx / cd;
                    if !q.is_finite() {
                        walk.poisoned = true;
                        return Ok(());
                    }
                    lo = lo.min(next_down(q));
                    hi = hi.max(next_up(q));
                }
                let c = if aj >= 0.0 { lo } else { hi };
                walk.add_bias(aj, sj, c);
            }
            walk.push(ctx, px, back_x_a, back_x_s)?;
            walk.push(ctx, pd, back_d_a, back_d_s)
        }

        _ => Err(Decline),
    }
}

/// Broadcast-aware flat index map from an output shape onto a constant's
/// shape (unary constant ops: the output shape equals the input shape).
fn constant_index_map(
    out_shape: &[usize],
    const_shape: &[usize],
    width: usize,
) -> WalkResult<Vec<usize>> {
    let map = crate::shape::broadcast_flat_index_map(out_shape, const_shape);
    if map.len() != width {
        return Err(Decline);
    }
    let const_len: usize = const_shape.iter().product::<usize>().max(1);
    if map.iter().any(|&m| m >= const_len) {
        return Err(Decline);
    }
    Ok(map)
}

/// MulBinary backward (design §5.3): interpolated McCormick facets from the
/// root-frozen SPSA alphas when available (mirroring
/// `propagate_linear_binary_with_alpha`), else the plane-selection McCormick
/// (mirroring `select_mccormick_plane`); EVERY plane used goes through the
/// corner-certify-and-repair gadget before its coefficients enter the walk.
/// Same-shape elementwise only (the lsnc form); anything else declines.
fn backward_mul_binary(
    ctx: &TailCtx<'_>,
    walk: &mut RowWalk,
    name: &str,
    inputs: &[String],
    a: &[f64],
    s: &[f64],
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
) -> WalkResult<()> {
    let (px, py) = match (inputs.first(), inputs.get(1)) {
        (Some(x), Some(y)) => (x.as_str(), y.as_str()),
        _ => return Err(Decline),
    };
    let width = a.len();
    // Broadcast-aware (#nn4sys-dual): flat index maps from the output shape
    // onto each parent (identity when same-shape — the lsnc form). Broadcast
    // fan-in accumulates at the mapped parent index, which preserves plane
    // validity (a sum of valid per-output inequalities).
    let out_shape = ctx.shape_of(name)?.to_vec();
    let xmap = constant_index_map(&out_shape, ctx.shape_of(px)?, width)?;
    let ymap = constant_index_map(&out_shape, ctx.shape_of(py)?, width)?;
    let x_width = ctx.width_of(px)?;
    let y_width = ctx.width_of(py)?;
    if xmap.iter().any(|&m| m >= x_width) || ymap.iter().any(|&m| m >= y_width) {
        return Err(Decline);
    }
    let (xl, xu) = ctx.anchor_of(px)?;
    let (yl, yu) = ctx.anchor_of(py)?;

    let alphas = mul_binary_alphas
        .and_then(|m| m.get(name))
        .filter(|arr| arr.shape() == [2, width]);

    let mut back_x_a = vec![0.0f64; x_width];
    let mut back_x_s = vec![0.0f64; x_width];
    let mut back_y_a = vec![0.0f64; y_width];
    let mut back_y_s = vec![0.0f64; y_width];

    for j in 0..width {
        let (wj, sj) = (a[j], s[j]);
        if wj == 0.0 && sj == 0.0 {
            continue;
        }
        let (lx, ux, ly, uy) = (xl[xmap[j]], xu[xmap[j]], yl[ymap[j]], yu[ymap[j]]);
        if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
            walk.poisoned = true;
            return Ok(());
        }

        // Candidate facet for the needed direction: w >= 0 uses the LOWER
        // plane of z, w < 0 the UPPER plane (single lower-form walk).
        let need_lower = wj >= 0.0;
        let (alpha, beta, nu) = match alphas {
            Some(arr) => {
                // Interpolated facets (auto_LiRPA bivariate.py:40-75), f64 RN;
                // r values clamp to [0,1] exactly as the f32 lane does.
                if need_lower {
                    let r = f64::from(arr[[0, j]].clamp(0.0, 1.0));
                    let alpha_l = (ly - uy) * r + uy;
                    let beta_l = (lx - ux) * r + ux;
                    let ny_l = (uy * ux - ly * lx) * r - uy * ux;
                    (alpha_l, beta_l, ny_l)
                } else {
                    let r = f64::from(arr[[1, j]].clamp(0.0, 1.0));
                    let alpha_u = (uy - ly) * r + ly;
                    let beta_u = (lx - ux) * r + ux;
                    let ny_u = (ly * ux - uy * lx) * r - ly * ux;
                    (alpha_u, beta_u, ny_u)
                }
            }
            None => select_mccormick_plane_f64(lx, ux, ly, uy, need_lower),
        };
        if !alpha.is_finite() || !beta.is_finite() || !nu.is_finite() {
            walk.poisoned = true;
            return Ok(());
        }

        // Corner-certify-and-repair (design §5.4): after this, the plane is
        // rigorously valid whatever rounding produced (alpha, beta, nu).
        let repaired = if need_lower {
            repair_lower_plane(alpha, beta, nu, lx, ux, ly, uy)
        } else {
            repair_upper_plane(alpha, beta, nu, lx, ux, ly, uy)
        };
        let Some(nu) = repaired else {
            walk.poisoned = true;
            return Ok(());
        };

        back_x_a[xmap[j]] += wj * alpha;
        back_x_s[xmap[j]] += sj * alpha.abs();
        back_y_a[ymap[j]] += wj * beta;
        back_y_s[ymap[j]] += sj * beta.abs();
        walk.add_bias(wj, sj, nu);
    }

    walk.push(ctx, px, back_x_a, back_x_s)?;
    walk.push(ctx, py, back_y_a, back_y_s)
}

/// f64 mirror of `select_mccormick_plane` (`layers/binary_ops/mul/mod.rs`)
/// collapsed to the single lower-form walk: `need_lower` selects between the
/// larger lower plane and the smaller upper plane at the box midpoint.
fn select_mccormick_plane_f64(
    lx: f64,
    ux: f64,
    ly: f64,
    uy: f64,
    need_lower: bool,
) -> (f64, f64, f64) {
    // Bit-identical (a+b)*0.5 anchors on the McCormick midpoint plane.
    #[allow(clippy::manual_midpoint)]
    let x0 = (lx + ux) * 0.5;
    #[allow(clippy::manual_midpoint)]
    let y0 = (ly + uy) * 0.5;
    // (coeff_x, coeff_y, const, value at midpoint)
    let l1 = (ly, lx, -lx * ly, lx * y0 + ly * x0 - lx * ly);
    let l2 = (uy, ux, -ux * uy, ux * y0 + uy * x0 - ux * uy);
    let u1 = (uy, lx, -lx * uy, lx * y0 + uy * x0 - lx * uy);
    let u2 = (ly, ux, -ux * ly, ux * y0 + ly * x0 - ux * ly);
    if need_lower {
        if l1.3 >= l2.3 {
            (l1.0, l1.1, l1.2)
        } else {
            (l2.0, l2.1, l2.2)
        }
    } else if u1.3 <= u2.3 {
        (u1.0, u1.1, u1.2)
    } else {
        (u2.0, u2.1, u2.2)
    }
}

/// Concat backward: coefficient slices along the concat axis route to the
/// live inputs (ONNX order, interleaving `constant_inputs` slots exactly like
/// the forward's `node_inputs`); POINT constants contribute `a·c` to the bias.
fn backward_concat(
    ctx: &TailCtx<'_>,
    walk: &mut RowWalk,
    name: &str,
    concat: &crate::layers::ConcatLayer,
    inputs: &[String],
    a: &[f64],
    s: &[f64],
) -> WalkResult<()> {
    let out_shape = ctx.shape_of(name)?.to_vec();
    let rank = out_shape.len();
    let axis = concat.normalize_axis(rank).map_err(|_| Decline)?;
    let width = a.len();
    if out_shape.iter().product::<usize>() != width {
        return Err(Decline);
    }

    // Slot list in ONNX order: Some(constant) or None (next live input).
    enum Slot<'a> {
        Live(&'a str),
        Constant(&'a BoundedTensor),
    }
    let mut slots: Vec<Slot<'_>> = Vec::new();
    match &concat.constant_inputs {
        Some(constant_inputs) => {
            let mut live_iter = inputs.iter();
            for entry in constant_inputs {
                match entry {
                    Some(constant) => slots.push(Slot::Constant(constant)),
                    None => slots.push(Slot::Live(live_iter.next().ok_or(Decline)?.as_str())),
                }
            }
            if live_iter.next().is_some() {
                return Err(Decline);
            }
        }
        None => {
            for input in inputs {
                slots.push(Slot::Live(input.as_str()));
            }
        }
    }
    if slots.is_empty() {
        return Err(Decline);
    }

    // Per-slot shape: must match the output on every non-axis dim.
    let mut slot_shapes: Vec<Vec<usize>> = Vec::with_capacity(slots.len());
    for slot in &slots {
        let shape: Vec<usize> = match slot {
            Slot::Live(n) => ctx.shape_of(n)?.to_vec(),
            Slot::Constant(c) => c.shape().to_vec(),
        };
        if shape.len() != rank {
            return Err(Decline);
        }
        for (d, (&sd, &od)) in shape.iter().zip(out_shape.iter()).enumerate() {
            if d != axis && sd != od {
                return Err(Decline);
            }
        }
        slot_shapes.push(shape);
    }
    let total_axis: usize = slot_shapes.iter().map(|s| s[axis]).sum();
    if total_axis != out_shape[axis] {
        return Err(Decline);
    }

    // Route each output coefficient to its slot.
    let mut slot_backs: Vec<(Vec<f64>, Vec<f64>)> = slot_shapes
        .iter()
        .map(|shape| {
            let w = shape.iter().product::<usize>().max(1);
            (vec![0.0f64; w], vec![0.0f64; w])
        })
        .collect();
    let offsets: Vec<usize> = slot_shapes
        .iter()
        .scan(0usize, |off, shape| {
            let cur = *off;
            *off += shape[axis];
            Some(cur)
        })
        .collect();
    for k in 0..width {
        let multi = multi_index(k, &out_shape);
        let c = multi[axis];
        // Find the slot containing axis coordinate `c`.
        let mut slot_idx = None;
        for (i, (&off, shape)) in offsets.iter().zip(slot_shapes.iter()).enumerate() {
            if c >= off && c < off + shape[axis] {
                slot_idx = Some(i);
                break;
            }
        }
        let i = slot_idx.ok_or(Decline)?;
        let mut local = multi.clone();
        local[axis] = c - offsets[i];
        let p = flat_index(&local, &slot_shapes[i]);
        slot_backs[i].0[p] += a[k];
        slot_backs[i].1[p] += s[k];
    }

    for (slot, (back_a, back_s)) in slots.into_iter().zip(slot_backs) {
        match slot {
            Slot::Live(parent) => walk.push(ctx, parent, back_a, back_s)?,
            Slot::Constant(constant) => {
                // Point constants only: a widened constant would need an
                // interval contribution — decline instead.
                let flat = constant.flatten();
                let (clo, chi) = (flat.lower(), flat.upper());
                if clo.len() != back_a.len() {
                    return Err(Decline);
                }
                for (j, (&ca, &cs)) in back_a.iter().zip(back_s.iter()).enumerate() {
                    if ca == 0.0 && cs == 0.0 {
                        continue;
                    }
                    let (lo, hi) = (clo[[j]], chi[[j]]);
                    if lo != hi || !lo.is_finite() {
                        return Err(Decline);
                    }
                    walk.add_bias(ca, cs, f64::from(lo));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

/// Certified f64 re-verification of one input-split domain (design §5.6).
///
/// Fail-closed: EVERY internal failure (unsupported op, shape surprise,
/// failed anchor collection, expired deadline) returns
/// [`F64TailOutcome::Unsupported`] — never a bound. Only
/// [`F64TailOutcome::Verified`] may change the caller's state, and only by
/// monotonically raising certified lower bounds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn f64_tail_verify(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> F64TailOutcome {
    let total_rows: usize = clause_sizes.iter().sum();
    if total_rows == 0
        || total_rows != spec_matrix.nrows()
        || total_rows != thresholds.len()
        || clause_sizes.contains(&0)
    {
        return F64TailOutcome::Unsupported;
    }
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return F64TailOutcome::Unsupported;
    }
    if !graph_supports_f64_tail(graph) {
        return F64TailOutcome::Unsupported;
    }

    // Anchors: the lane's own f32-sound per-node boxes for THIS domain
    // (design §5.2 — same relaxation base as the f32 lane, exact-widened).
    let collected;
    let node_bounds: &HashMap<String, BoundedTensor> = match node_bounds {
        Some(nb) => nb,
        None => {
            match crate::network::collect_intermediate_bounds(graph, input_bounds, deadline, engine)
            {
                Ok(nb) => {
                    collected = nb;
                    &collected
                }
                Err(_) => return F64TailOutcome::Unsupported,
            }
        }
    };

    let Some(ctx) = build_tail_ctx(graph, input_bounds, node_bounds) else {
        if tail_debug() {
            eprintln!("[dual-f64-tail-stage] build_tail_ctx -> None (anchor/shape/exec miss)");
        }
        return F64TailOutcome::Unsupported;
    };

    let mut row_lowers = Vec::with_capacity(total_rows);
    for row in spec_matrix.rows() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return F64TailOutcome::Unsupported;
        }
        let row_f64: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
        if row_f64.iter().any(|v| !v.is_finite()) {
            row_lowers.push(f64::NEG_INFINITY);
            continue;
        }
        match certified_row_lower(&ctx, &row_f64, mul_binary_alphas) {
            Ok(l) => row_lowers.push(l),
            Err(Decline) => {
                if tail_debug() {
                    eprintln!(
                        "[dual-f64-tail-stage] row {} walk DECLINED",
                        row_lowers.len()
                    );
                }
                return F64TailOutcome::Unsupported;
            }
        }
    }

    let (verified, min_gap) = grouped_verdict_f64(&row_lowers, thresholds, clause_sizes);
    if verified {
        F64TailOutcome::Verified { row_lowers }
    } else {
        F64TailOutcome::NotVerified {
            min_gap_f64: min_gap,
        }
    }
}

/// Build the shared per-domain walk context: f32 -> f64 exact widening of the
/// per-node anchors and the input box. `None` on any shape/anchor miss or a
/// non-finite input box (the caller fails closed with `Unsupported`).
fn build_tail_ctx<'a>(
    graph: &'a GraphNetwork,
    input_bounds: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> Option<TailCtx<'a>> {
    let Ok(exec) = graph.exec_order() else {
        return None;
    };
    let Ok(needed) = graph.output_ancestors() else {
        return None;
    };
    let output_name = graph.output_name();
    let output_pos = exec.iter().position(|n| n.as_str() == output_name)?;
    let exec_prefix: Vec<&str> = exec[..=output_pos].iter().map(|n| n.as_str()).collect();

    let mut anchors: HashMap<&str, (Vec<f64>, Vec<f64>)> = HashMap::new();
    let mut shapes: HashMap<&str, Vec<usize>> = HashMap::new();
    for &name in &exec_prefix {
        if !needed.contains(name) {
            continue;
        }
        let bounds = node_bounds.get(name)?;
        let flat = bounds.flatten();
        let lo: Vec<f64> = flat.lower().iter().map(|&v| f64::from(v)).collect();
        let hi: Vec<f64> = flat.upper().iter().map(|&v| f64::from(v)).collect();
        anchors.insert(name, (lo, hi));
        shapes.insert(name, bounds.shape().to_vec());
    }
    let in_flat = input_bounds.flatten();
    let input_lo: Vec<f64> = in_flat.lower().iter().map(|&v| f64::from(v)).collect();
    let input_hi: Vec<f64> = in_flat.upper().iter().map(|&v| f64::from(v)).collect();
    if input_lo
        .iter()
        .chain(input_hi.iter())
        .any(|v| !v.is_finite())
    {
        return None;
    }

    Some(TailCtx {
        graph,
        exec_prefix,
        needed,
        anchors,
        shapes,
        input_lo,
        input_hi,
        input_shape: input_bounds.shape().to_vec(),
    })
}

/// Grouped verdict in f64 — the mirror of `disjunctive_domain_verified` /
/// `disjunctive_domain_priority` (f32 -> f64 threshold widening is exact;
/// non-finite rows never verify and gap to -inf). Returns
/// `(verified, min_gap)`.
fn grouped_verdict_f64(
    row_lowers: &[f64],
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> (bool, f64) {
    let mut offset = 0usize;
    let mut verified = true;
    let mut min_gap = f64::INFINITY;
    for &size in clause_sizes {
        let mut clause_best = f64::NEG_INFINITY;
        let mut clause_ok = false;
        for k in offset..offset + size {
            let t = f64::from(thresholds[k]);
            let l = row_lowers[k];
            let gap = if l.is_finite() {
                l - t
            } else {
                f64::NEG_INFINITY
            };
            if gap > clause_best {
                clause_best = gap;
            }
            if l.is_finite() && l > t {
                clause_ok = true;
            }
        }
        if !clause_ok {
            verified = false;
        }
        if clause_best < min_gap {
            min_gap = clause_best;
        }
        offset += size;
    }
    (verified, min_gap)
}

// ---------------------------------------------------------------------------
// Alpha-tail: per-domain MulBinary alpha refresh (design option A).
// ---------------------------------------------------------------------------

/// Deterministic 64-bit PRNG (SplitMix64) for the refresh's Bernoulli ±1
/// SPSA perturbations. A local fixed-seed generator (instead of the process
/// RNG) keeps every domain's refresh outcome independent of rayon thread
/// interleaving — the same determinism discipline as the sorted-key draw
/// order in `input_split/mul_binary.rs` (task #36). Soundness never depends
/// on the draws (any alpha in [0,1] is sound); determinism is for
/// reproducible verdict traces only.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_bool(&mut self) -> bool {
        // Top bit of the mixed output.
        self.next_u64() >> 63 == 1
    }
}

/// Result of one refreshed (alpha-tail) pass over a single domain.
///
/// `outcome` follows the [`F64TailOutcome`] one-way contract. `row_lowers`
/// are the KEEP-BEST merged certified per-row lowers (baseline warm-alpha
/// pass max-merged with every refresh candidate evaluated) — each row's
/// value comes from ONE certified evaluation, so the per-row max of
/// certificates is itself a certificate (rows are independent walks).
pub(crate) struct AlphaTailEval {
    pub(crate) outcome: F64TailOutcome,
    /// Grouped f64 gap of the BASELINE (warm-alpha, pre-refresh) pass — the
    /// landed `gap_f64` telemetry channel.
    pub(crate) gap_baseline: f64,
    /// Grouped f64 gap AFTER the refresh (keep-best merged rows) — the
    /// `gap_f64_refreshed` decision artifact of the alpha-tail design §5.
    pub(crate) gap_refreshed: f64,
    /// Keep-best merged certified rows (empty on `Unsupported`). Consumed by
    /// the soundness tests (keep-best + containment assertions); the hook
    /// reads the rows through `outcome` instead.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) row_lowers: Vec<f64>,
    /// Best-seen alpha map by refresh objective (micro-BaB warm start);
    /// `None` when no refresh ran (no alphas / no MulBinary / zero iters).
    pub(crate) refreshed_alphas: Option<HashMap<String, Array2<f32>>>,
}

impl AlphaTailEval {
    fn unsupported() -> Self {
        Self {
            outcome: F64TailOutcome::Unsupported,
            gap_baseline: f64::NEG_INFINITY,
            gap_refreshed: f64::NEG_INFINITY,
            row_lowers: Vec::new(),
            refreshed_alphas: None,
        }
    }
}

/// One refresh-objective evaluation: walks the BLOCKING rows with the
/// candidate alpha map, folds every certified value into the keep-best row
/// merge, and returns the blocking-clause grouped objective
/// `min over blocking clauses of (max over clause rows of l - t)` computed
/// from THIS candidate's values (candidate-coherent for the SPSA gradient;
/// the keep-best merge is what feeds the verdict). `None` aborts the refresh
/// (deadline / walk decline) — accumulated keep-best rows stay valid.
#[allow(clippy::too_many_arguments)]
fn refresh_eval(
    ctx: &TailCtx<'_>,
    rows_f64: &[Option<Vec<f64>>],
    thresholds: &[f32],
    blocking_clauses: &[(usize, usize)],
    alphas: &HashMap<String, Array2<f32>>,
    best_rows: &mut [f64],
    deadline: Option<Instant>,
) -> Option<f64> {
    let mut cand: Vec<(usize, f64)> = Vec::new();
    for &(offset, size) in blocking_clauses {
        for k in offset..offset + size {
            let Some(row) = rows_f64[k].as_ref() else {
                continue; // spec-poisoned row: contributes -inf below
            };
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            let Ok(l) = certified_row_lower(ctx, row, Some(alphas)) else {
                return None;
            };
            if l > best_rows[k] {
                best_rows[k] = l;
            }
            cand.push((k, l));
        }
    }
    let mut obj = f64::INFINITY;
    for &(offset, size) in blocking_clauses {
        let mut clause_best = f64::NEG_INFINITY;
        for k in offset..offset + size {
            let l = cand
                .iter()
                .find(|&&(i, _)| i == k)
                .map_or(f64::NEG_INFINITY, |&(_, l)| l);
            let gap = if l.is_finite() {
                l - f64::from(thresholds[k])
            } else {
                f64::NEG_INFINITY
            };
            if gap > clause_best {
                clause_best = gap;
            }
        }
        if clause_best < obj {
            obj = clause_best;
        }
    }
    Some(obj)
}

/// Certified f64 re-verification of one domain WITH the per-domain MulBinary
/// alpha refresh (alpha-tail design option A; module doc above).
///
/// Baseline = the plain warm-alpha pass (identical to [`f64_tail_verify`]).
/// If it does not verify, an SPSA+Adam loop (`iters` iterations, the
/// `input_split/mul_binary.rs` recipe: Bernoulli ±1, eps 1e-3, Adam lr 0.1)
/// re-targets the alphas for THIS box and the BLOCKING clause rows, each
/// candidate evaluated through the certified row walk on the shared cached
/// anchors. Keep-best PER ROW across the baseline and every candidate means
/// the refreshed pass can only meet-or-beat the frozen-alpha pass.
///
/// Soundness: the optimizer only selects which sound relaxation the
/// certified walk evaluates (any clamped alpha in [0,1] is a convex
/// combination of valid McCormick facets, and corner-repair re-validates
/// every plane); every merged row value is one certified evaluation's
/// output. Fail-closed: validation/anchor/op misses return `Unsupported`;
/// mid-refresh declines keep the (certified) best-so-far merge.
#[allow(clippy::too_many_arguments)]
pub(crate) fn f64_tail_verify_refreshed(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    warm_alphas: Option<&HashMap<String, Array2<f32>>>,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    iters: usize,
    seed: u64,
) -> AlphaTailEval {
    let total_rows: usize = clause_sizes.iter().sum();
    if total_rows == 0
        || total_rows != spec_matrix.nrows()
        || total_rows != thresholds.len()
        || clause_sizes.contains(&0)
    {
        return AlphaTailEval::unsupported();
    }
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return AlphaTailEval::unsupported();
    }
    if !graph_supports_f64_tail(graph) {
        return AlphaTailEval::unsupported();
    }

    let collected;
    let node_bounds: &HashMap<String, BoundedTensor> = match node_bounds {
        Some(nb) => nb,
        None => {
            match crate::network::collect_intermediate_bounds(graph, input_bounds, deadline, engine)
            {
                Ok(nb) => {
                    collected = nb;
                    &collected
                }
                Err(_) => return AlphaTailEval::unsupported(),
            }
        }
    };
    let Some(ctx) = build_tail_ctx(graph, input_bounds, node_bounds) else {
        return AlphaTailEval::unsupported();
    };

    // Widened spec rows (`None` = spec-poisoned row, bound stays -inf).
    let rows_f64: Vec<Option<Vec<f64>>> = spec_matrix
        .rows()
        .into_iter()
        .map(|row| {
            let r: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            r.iter().all(|v| v.is_finite()).then_some(r)
        })
        .collect();

    // Baseline pass with the warm (root-frozen) alphas.
    let mut baseline = Vec::with_capacity(total_rows);
    for row in &rows_f64 {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return AlphaTailEval::unsupported();
        }
        match row {
            None => baseline.push(f64::NEG_INFINITY),
            Some(r) => match certified_row_lower(&ctx, r, warm_alphas) {
                Ok(l) => baseline.push(l),
                Err(Decline) => return AlphaTailEval::unsupported(),
            },
        }
    }
    let (verified0, gap_baseline) = grouped_verdict_f64(&baseline, thresholds, clause_sizes);
    if verified0 {
        return AlphaTailEval {
            outcome: F64TailOutcome::Verified {
                row_lowers: baseline.clone(),
            },
            gap_baseline,
            gap_refreshed: gap_baseline,
            row_lowers: baseline,
            refreshed_alphas: None,
        };
    }

    // Blocking clauses = clauses without a certified row above threshold.
    let mut blocking_clauses: Vec<(usize, usize)> = Vec::new();
    let mut offset = 0usize;
    for &size in clause_sizes {
        let satisfied = (offset..offset + size).any(|k| {
            let l = baseline[k];
            l.is_finite() && l > f64::from(thresholds[k])
        });
        if !satisfied {
            blocking_clauses.push((offset, size));
        }
        offset += size;
    }

    // Refresh eligibility: a warm alpha map, a MulBinary node to steer, and
    // a non-zero iteration budget. Otherwise the baseline IS the answer.
    let graph_has_mul = graph
        .nodes
        .values()
        .any(|node| matches!(node.layer(), Layer::MulBinary(_)));
    let no_refresh = |rows: Vec<f64>, gap: f64| AlphaTailEval {
        outcome: F64TailOutcome::NotVerified { min_gap_f64: gap },
        gap_baseline,
        gap_refreshed: gap,
        row_lowers: rows,
        refreshed_alphas: None,
    };
    let Some(warm) = warm_alphas else {
        return no_refresh(baseline, gap_baseline);
    };
    if iters == 0 || warm.is_empty() || !graph_has_mul || blocking_clauses.is_empty() {
        return no_refresh(baseline, gap_baseline);
    }

    // SPSA+Adam refresh (the `optimize_mul_binary_alphas_spsa` recipe with
    // the per-domain box + blocking-rows certified-f64 objective).
    let eps = 1e-3_f32;
    let lr = 0.1_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let adam_eps = 1e-8_f32;

    let mut cur: HashMap<String, Array2<f32>> = warm.clone();
    let mut adam_m: HashMap<String, Array2<f32>> = cur
        .iter()
        .map(|(k, v)| (k.clone(), Array2::zeros(v.raw_dim())))
        .collect();
    let mut adam_v: HashMap<String, Array2<f32>> = cur
        .iter()
        .map(|(k, v)| (k.clone(), Array2::zeros(v.raw_dim())))
        .collect();
    let mut rng = SplitMix64::new(seed);
    let mut names: Vec<String> = cur.keys().cloned().collect();
    names.sort_unstable();

    let mut best_rows = baseline.clone();
    let mut best_alphas = cur.clone();
    // Baseline objective over the blocking clauses (for best-alpha tracking).
    let mut best_obj = {
        let mut obj = f64::INFINITY;
        for &(off, size) in &blocking_clauses {
            let mut clause_best = f64::NEG_INFINITY;
            for k in off..off + size {
                let l = baseline[k];
                let gap = if l.is_finite() {
                    l - f64::from(thresholds[k])
                } else {
                    f64::NEG_INFINITY
                };
                clause_best = clause_best.max(gap);
            }
            obj = obj.min(clause_best);
        }
        obj
    };

    'spsa: for iter in 0..iters {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        // Bernoulli ±1 perturbations in sorted-key order (determinism).
        let perturbations: Vec<(usize, Array2<f32>)> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let pert = Array2::from_shape_fn(cur[name].raw_dim(), |_| {
                    if rng.next_bool() {
                        1.0_f32
                    } else {
                        -1.0_f32
                    }
                });
                (i, pert)
            })
            .collect();

        let mut alpha_plus = cur.clone();
        let mut alpha_minus = cur.clone();
        for (i, pert) in &perturbations {
            let name = &names[*i];
            if let Some(a) = alpha_plus.get_mut(name) {
                a.zip_mut_with(pert, |v, &p| *v = (*v + eps * p).clamp(0.0, 1.0));
            }
            if let Some(a) = alpha_minus.get_mut(name) {
                a.zip_mut_with(pert, |v, &p| *v = (*v - eps * p).clamp(0.0, 1.0));
            }
        }

        let Some(obj_plus) = refresh_eval(
            &ctx,
            &rows_f64,
            thresholds,
            &blocking_clauses,
            &alpha_plus,
            &mut best_rows,
            deadline,
        ) else {
            break 'spsa;
        };
        if obj_plus > best_obj {
            best_obj = obj_plus;
            best_alphas = alpha_plus.clone();
        }
        let Some(obj_minus) = refresh_eval(
            &ctx,
            &rows_f64,
            thresholds,
            &blocking_clauses,
            &alpha_minus,
            &mut best_rows,
            deadline,
        ) else {
            break 'spsa;
        };
        if obj_minus > best_obj {
            best_obj = obj_minus;
            best_alphas = alpha_minus.clone();
        }

        #[allow(clippy::cast_possible_truncation)]
        let diff = (obj_plus - obj_minus) as f32;
        if !diff.is_finite() {
            continue;
        }
        let t = (iter + 1) as f32;
        let bc1 = (1.0 - beta1.powf(t)).max(f32::EPSILON);
        let bc2 = (1.0 - beta2.powf(t)).max(f32::EPSILON);
        for (i, pert) in &perturbations {
            let name = &names[*i];
            let (Some(alpha), Some(m), Some(v)) = (
                cur.get_mut(name),
                adam_m.get_mut(name),
                adam_v.get_mut(name),
            ) else {
                continue;
            };
            let shape = alpha.raw_dim();
            for row in 0..shape[0] {
                for col in 0..shape[1] {
                    let p = pert[[row, col]];
                    let grad = diff / (2.0 * eps * p);
                    let neg_grad = -grad;
                    m[[row, col]] = beta1 * m[[row, col]] + (1.0 - beta1) * neg_grad;
                    v[[row, col]] = beta2 * v[[row, col]] + (1.0 - beta2) * neg_grad * neg_grad;
                    let m_hat = m[[row, col]] / bc1;
                    let v_hat = v[[row, col]] / bc2;
                    alpha[[row, col]] -= lr * m_hat / (v_hat.sqrt() + adam_eps);
                    alpha[[row, col]] = alpha[[row, col]].clamp(0.0, 1.0);
                    if alpha[[row, col]].is_nan() {
                        alpha[[row, col]] = 0.5;
                        m[[row, col]] = 0.0;
                        v[[row, col]] = 0.0;
                    }
                }
            }
        }
    }

    // Final evaluation at the converged point (keep-best absorbs it).
    if let Some(obj_final) = refresh_eval(
        &ctx,
        &rows_f64,
        thresholds,
        &blocking_clauses,
        &cur,
        &mut best_rows,
        deadline,
    ) {
        if obj_final > best_obj {
            best_alphas = cur;
        }
    }

    let (verified, gap_refreshed) = grouped_verdict_f64(&best_rows, thresholds, clause_sizes);
    let outcome = if verified {
        F64TailOutcome::Verified {
            row_lowers: best_rows.clone(),
        }
    } else {
        F64TailOutcome::NotVerified {
            min_gap_f64: gap_refreshed,
        }
    };
    AlphaTailEval {
        outcome,
        gap_baseline,
        gap_refreshed,
        row_lowers: best_rows,
        refreshed_alphas: Some(best_alphas),
    }
}

#[cfg(test)]
mod tests;
