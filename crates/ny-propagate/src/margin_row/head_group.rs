// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Optional source-owned head-group root margin refinement.
//!
//! Envelopes come only from the lane-private, producer-bound root YPack in
//! the same immutable network invocation. This module emits numeric lower
//! bounds for immediate same-frame use; it mints no verdict or receipt.
//!
//! For a group G, enumerate every CLOSED pattern s in {0,1}^G. If
//! L_i(x)<=y_i<=U_i(x), eta_i*y_i>=0 and eta_i=2*s_i-1, then, for
//! nonnegative l_i,u_i,p_i, a lower bound on the phase margin is the box
//! minimum of
//!   rest(x)+sum(l_i L_i(x)-u_i U_i(x))
//!             +sum((q_i*s_i-l_i+u_i-eta_i*p_i)*y_i).
//! Every coefficient is replayed with intervals and the residual y-box is
//! retained. Proposal optimization is never a proof premise. Minimum over
//! ALL patterns covers the actual ReLU graph, including zero boundaries.

use std::time::{Duration, Instant};

use super::bab::YPack;
use super::engine::{BackwardEngine, LaneDir, PassOut, Seed};
use super::net::TwinOp;
use super::rounding::RoundMode;

const MAX_INPUTS: usize = 4096;
const MAX_HEAD: usize = 512;
const MAX_GROUP: usize = 4;
const MAX_MAG: f64 = 1e100;
const MAX_PROPOSAL_ROUNDS: usize = 4;
const MAX_NATIVE_COLUMNS: usize = 4;
const MAX_NATIVE_BATCHES: usize = 2;
const MAX_ADVICE_FEATURES: usize = 256;

#[path = "head_group_advice.rs"]
mod advice;
#[path = "direct_head.rs"]
pub(super) mod direct;
#[path = "head_group_functional.rs"]
mod functional;
#[path = "input_head.rs"]
pub(super) mod input_direct;
#[path = "head_group_native.rs"]
mod native;
pub use advice::register_head_phase_advice_factory;

/// Checked limits for an optional, source-owned root refinement.
/// This policy carries no proof authority and does not change the base memo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadGroupPolicy {
    max_seconds: f64,
    groups_per_margin: usize,
    proposal_rounds: usize,
    directional_cuts: bool,
    phase_cover: Option<PhaseCoverPolicy>,
}

/// Explicit opt-in for a complete nominal-sign cover of authored input heads.
/// The count is sign-split heads, so seven admits all 128 closed masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseCoverPolicy {
    max_heads: usize,
}

impl PhaseCoverPolicy {
    pub fn new(max_heads: usize) -> Option<Self> {
        (1..=7).contains(&max_heads).then_some(Self { max_heads })
    }

    pub fn max_heads(self) -> usize {
        self.max_heads
    }
}

impl HeadGroupPolicy {
    /// Refuse unbounded work rather than silently changing a typed setting.
    pub fn new(max_seconds: f64, groups_per_margin: usize, proposal_rounds: usize) -> Option<Self> {
        (max_seconds.is_finite()
            && max_seconds > 0.0
            && max_seconds <= 5.0
            && (1..=2).contains(&groups_per_margin)
            && (1..=MAX_PROPOSAL_ROUNDS).contains(&proposal_rounds))
        .then_some(Self {
            max_seconds,
            groups_per_margin,
            proposal_rounds,
            directional_cuts: true,
            phase_cover: None,
        })
    }

    /// Maximum live slice, charged inside the original driver deadline.
    pub fn max_seconds(self) -> f64 {
        self.max_seconds
    }
    /// Source-only portfolios checked for each still-open class margin.
    pub fn groups_per_margin(self) -> usize {
        self.groups_per_margin
    }
    /// Bounded native coordinate proposal rounds; not an optimality claim.
    pub fn proposal_rounds(self) -> usize {
        self.proposal_rounds
    }

    /// Include the Source-certified directional-cut stage in the bounded root pass.
    pub fn with_directional_cuts(mut self, enabled: bool) -> Self {
        self.directional_cuts = enabled;
        self
    }

    pub fn directional_cuts(self) -> bool {
        self.directional_cuts
    }

    /// Keep the old portfolio unchanged unless a separate bounded cover is armed.
    pub fn with_phase_cover(mut self, policy: PhaseCoverPolicy) -> Self {
        self.phase_cover = Some(policy);
        self
    }

    pub fn phase_cover(self) -> Option<PhaseCoverPolicy> {
        self.phase_cover
    }
}

/// Telemetry for one complete source-certified phase cover.
#[derive(Clone, Debug)]
pub struct HeadGroupCandidate {
    /// Original class identity; candidates are never reused for another row.
    pub adversary: usize,
    /// Source head indices, in the phase-bit order used during replay.
    pub group: Vec<usize>,
    /// Certified lower endpoint of this complete cover.
    pub lower: f64,
    /// Every one of the `2^group.len()` closed patterns was checked.
    pub phases: usize,
}

/// Numeric bounds for immediate use in the same immutable root invocation.
/// The caller retains verdict/formula authority and its original deadline.
#[derive(Clone, Debug)]
pub struct HeadGroupReport {
    /// Aligned to the caller's adversary rows; includes their old bounds.
    pub bounds: Vec<f64>,
    /// Complete groups only; an expired partial group is never counted.
    pub groups_checked: usize,
    /// Complete closed phase covers only.
    pub phases_checked: usize,
    /// Number of adversary rows with a strictly stronger numeric bound.
    pub classes_improved: usize,
    /// Completed cancellable whole-functional CPU passes, not proposals.
    pub native_passes: usize,
    /// Completed columns, including alternative proposals for a phase.
    pub native_columns: usize,
    /// Distinct phase bounds strengthened by source-owned native replay.
    pub native_improved_phases: usize,
    /// Completed provider invocations; their hints have no proof authority.
    pub advice_calls: usize,
    /// Well-formed phase multiplier candidates submitted to owned replay.
    pub advice_candidates: usize,
    /// Distinct phase bounds strengthened by replay of returned multipliers.
    pub advice_improved_phases: usize,
    /// V2 global slope proposals; returned duals remain untrusted.
    pub functional_advice_calls: usize,
    pub functional_advice_candidates: usize,
    /// Margins strengthened by the unconditional complete functional replay.
    pub functional_classes_improved: usize,
    /// Total charged construction, proposal and replay wall time.
    pub elapsed_s: f64,
    /// Diagnostic candidates; these are not receipts or reusable proof tokens.
    pub candidates: Vec<HeadGroupCandidate>,
}

#[derive(Clone, Copy, Debug)]
struct I {
    lo: f64,
    hi: f64,
}

fn good(x: f64) -> bool {
    x.is_finite() && x.abs() <= MAX_MAG
}
fn add_down(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        (a + b).next_down()
    }
}
fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        (a + b).next_up()
    }
}
fn mul_down(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else if a == 1.0 {
        b
    } else if b == 1.0 {
        a
    } else if a == -1.0 {
        -b
    } else if b == -1.0 {
        -a
    } else {
        (a * b).next_down()
    }
}
fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else if a == 1.0 {
        b
    } else if b == 1.0 {
        a
    } else if a == -1.0 {
        -b
    } else if b == -1.0 {
        -a
    } else {
        (a * b).next_up()
    }
}

impl I {
    fn p(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn valid(self) -> bool {
        good(self.lo) && good(self.hi) && self.lo <= self.hi
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn add(self, b: Self) -> Option<Self> {
        if !self.valid() || !b.valid() {
            return None;
        }
        let r = Self {
            lo: add_down(self.lo, b.lo),
            hi: add_up(self.hi, b.hi),
        };
        r.valid().then_some(r)
    }
    fn sub(self, b: Self) -> Option<Self> {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Option<Self> {
        if !self.valid() || !b.valid() {
            return None;
        }
        if self.lo == 0.0 && self.hi == 0.0 || b.lo == 0.0 && b.hi == 0.0 {
            return Some(Self::p(0.0));
        }
        if self.lo == 1.0 && self.hi == 1.0 {
            return Some(b);
        }
        if b.lo == 1.0 && b.hi == 1.0 {
            return Some(self);
        }
        if self.lo == -1.0 && self.hi == -1.0 {
            return Some(b.neg());
        }
        if b.lo == -1.0 && b.hi == -1.0 {
            return Some(self.neg());
        }
        if self.lo == self.hi && b.lo == b.hi {
            let r = Self {
                lo: mul_down(self.lo, b.lo),
                hi: mul_up(self.hi, b.hi),
            };
            return r.valid().then_some(r);
        }
        let mut r = Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        };
        for (a, b) in [
            (self.lo, b.lo),
            (self.lo, b.hi),
            (self.hi, b.lo),
            (self.hi, b.hi),
        ] {
            r.lo = r.lo.min(mul_down(a, b));
            r.hi = r.hi.max(mul_up(a, b));
        }
        r.valid().then_some(r)
    }
}

fn live(deadline: Instant) -> bool {
    Instant::now() < deadline
}
fn fp_qualified() -> bool {
    ny_core::require_f64_interval_proof_environment().is_ok()
}

#[derive(Clone)]
struct Affine {
    a: Vec<I>,
    b: I,
}
impl Affine {
    fn zero(n: usize) -> Self {
        Self {
            a: vec![I::p(0.0); n],
            b: I::p(0.0),
        }
    }
    fn valid(&self, n: usize) -> bool {
        self.a.len() == n && self.b.valid() && self.a.iter().all(|v| v.valid())
    }
    fn add_scaled(&mut self, row: &Self, c: f64, deadline: Instant) -> Option<()> {
        if !good(c) || self.a.len() != row.a.len() {
            return None;
        }
        self.b = self.b.add(row.b.mul(I::p(c))?)?;
        for (i, (a, b)) in self.a.iter_mut().zip(&row.a).enumerate() {
            if i & 63 == 0 && !live(deadline) {
                return None;
            }
            *a = a.add(b.mul(I::p(c))?)?;
        }
        live(deadline).then_some(())
    }
    fn lower(&self, box_: &[I], deadline: Instant) -> Option<f64> {
        if !self.valid(box_.len()) {
            return None;
        }
        let mut r = self.b;
        // Aggregate all affine contributions BEFORE taking any box minima.
        for (j, (a, x)) in self.a.iter().zip(box_).enumerate() {
            if j & 63 == 0 && !live(deadline) {
                return None;
            }
            r = r.add(a.mul(*x)?)?;
        }
        (live(deadline) && r.valid()).then_some(r.lo)
    }
    fn upper(&self, box_: &[I], deadline: Instant) -> Option<f64> {
        if !self.valid(box_.len()) {
            return None;
        }
        let mut r = self.b;
        for (j, (a, x)) in self.a.iter().zip(box_).enumerate() {
            if j & 63 == 0 && !live(deadline) {
                return None;
            }
            r = r.add(a.mul(*x)?)?;
        }
        (live(deadline) && r.valid()).then_some(r.hi)
    }
}

// The FP environment alone cannot certify that a frozen chord encloses
// ReLU. Replay its endpoint/kink inequalities before accepting source gates.
// This is a conservative qualifier, not a proof of the producer's y boxes.
pub(super) fn root_lines_qualified(root: &super::root::RootGates) -> bool {
    root.layers.iter().all(|layer| {
        [
            &layer.l,
            &layer.u,
            &layer.alpha,
            &layer.s,
            &layer.c,
            &layer.ms,
        ]
        .iter()
        .all(|v| v.len() == layer.n)
            && (0..layer.n).all(|j| {
                let (l, u, a, s, c, ms) = (
                    layer.l[j],
                    layer.u[j],
                    layer.alpha[j],
                    layer.s[j],
                    layer.c[j],
                    layer.ms[j],
                );
                [l, u, a, s, c, ms].iter().all(|v| good(*v))
                    && l <= u
                    && (0.0..=1.0).contains(&a)
                    && (0.0..=1.0).contains(&s)
                    && c >= 0.0
                    && ms >= a.max(s)
                    && [l, 0.0, u].into_iter().all(|y| {
                        I::p(s)
                            .mul(I::p(y))
                            .and_then(|v| v.add(I::p(c)))
                            .is_some_and(|v| v.lo >= y.max(0.0))
                    })
            })
    })
}

#[derive(Clone)]
struct Envelope {
    lower: Affine,
    upper: Affine,
    y: I,
    // Any [0,1] alpha is valid. This is copied from the lane's source-owned
    // current head gates for the unselected-neuron remainder.
    alpha: f64,
}

#[derive(Clone)]
struct SourceHeadProblem {
    root: Vec<I>,
    head: Vec<Envelope>,
    q: Vec<I>,
    bias: I,
}
impl SourceHeadProblem {
    fn validate(&self) -> Option<()> {
        let n = self.root.len();
        let h = self.head.len();
        if n == 0
            || n > MAX_INPUTS
            || !(2..=MAX_HEAD).contains(&h)
            || self.q.len() != h
            || !self.bias.valid()
            || self.root.iter().any(|a| !a.valid())
            || self.q.iter().any(|a| !a.valid())
            || self.head.iter().any(|a| {
                !a.lower.valid(n)
                    || !a.upper.valid(n)
                    || !a.y.valid()
                    || !good(a.alpha)
                    || !(0.0..=1.0).contains(&a.alpha)
            })
        {
            return None;
        }
        // Shape checks do not mint envelope authority. The only production
        // constructor is from_owned_pack, behind the private producer binding.
        Some(())
    }
}

// Certify q*rho(y) >= v*y+k on the scalar source interval. Endpoints and0
// suffice because every exact q in its interval gives a piecewise affine
// residual. Interval products repair BOTH q and proposed slope roundoff.
fn scalar_intercept(q: I, v: f64, y: I) -> Option<f64> {
    if !q.valid() || !good(v) || !y.valid() {
        return None;
    }
    let mut k = f64::INFINITY;
    for t in [y.lo, y.hi, 0.0]
        .into_iter()
        .filter(|t| *t >= y.lo && *t <= y.hi)
    {
        k = k.min(q.mul(I::p(t.max(0.0)))?.sub(I::p(v).mul(I::p(t))?)?.lo);
    }
    good(k).then_some(k)
}

fn scalar_line(q: I, e: &Envelope) -> Option<(f64, f64)> {
    let nominal = 0.5 * q.lo + 0.5 * q.hi;
    let slope = if e.y.hi <= 0.0 {
        0.0
    } else if e.y.lo >= 0.0 {
        1.0
    } else if nominal >= 0.0 {
        e.alpha
    } else {
        e.y.hi / (e.y.hi - e.y.lo)
    };
    let v = nominal * slope;
    Some((v, scalar_intercept(q, v, e.y)?))
}

fn remainder(problem: &SourceHeadProblem, group: &[usize], deadline: Instant) -> Option<Affine> {
    let mut r = Affine::zero(problem.root.len());
    r.b = problem.bias;
    for (j, e) in problem.head.iter().enumerate() {
        if !live(deadline) {
            return None;
        }
        if group.contains(&j) {
            continue;
        }
        let (v, k) = scalar_line(problem.q[j], e)?;
        r.add_scaled(if v >= 0.0 { &e.lower } else { &e.upper }, v, deadline)?;
        r.b = r.b.add(I::p(k))?;
    }
    Some(r)
}

#[derive(Clone, Debug)]
struct Proposal {
    mask: usize,
    phase: Vec<f64>,
}

#[derive(Clone)]
struct FunctionalProposal {
    mask: usize,
    slopes: Vec<f64>,
    phase: Vec<f64>,
}

// Multipliers are the ONLY proposal data. lambdaL/U are selected to cancel
// the NOMINAL y coefficient; replay prices its ENTIRE interval residual.
fn phase_lower(
    problem: &SourceHeadProblem,
    group: &[usize],
    rest: &Affine,
    p: &Proposal,
    deadline: Instant,
) -> Option<f64> {
    if !live(deadline)
        || !fp_qualified()
        || p.mask >= (1usize << group.len())
        || p.phase.len() != group.len()
        || p.phase.iter().any(|v| !good(*v) || *v < 0.0)
    {
        return None;
    }
    let mut r = rest.clone();
    let mut residual = I::p(0.0);
    for (i, &j) in group.iter().enumerate() {
        if !live(deadline) {
            return None;
        }
        let s = if p.mask & (1 << i) != 0 { 1.0 } else { 0.0 };
        let eta = if s == 1.0 { 1.0 } else { -1.0 };
        let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
        let nominal = q * s - eta * p.phase[i];
        if !good(nominal) {
            return None;
        }
        let lower = nominal.max(0.0);
        let upper = (-nominal).max(0.0);
        r.add_scaled(&problem.head[j].lower, lower, deadline)?;
        r.add_scaled(&problem.head[j].upper, -upper, deadline)?;
        let c = problem.q[j]
            .mul(I::p(s))?
            .sub(I::p(lower))?
            .add(I::p(upper))?
            .sub(I::p(eta).mul(I::p(p.phase[i]))?)?;
        residual = residual.add(c.mul(problem.head[j].y)?)?;
    }
    let lower = I::p(r.lower(&problem.root, deadline)?).add(residual)?.lo;
    (good(lower) && live(deadline)).then_some(lower)
}

// RANKER ONLY. Rounded midpoint composition never enters acceptance. This
// keeps expensive interval replay at one certificate per CLOSED phase.
fn score_phase(
    problem: &SourceHeadProblem,
    group: &[usize],
    rest: &Affine,
    p: &Proposal,
    deadline: Instant,
) -> Option<f64> {
    let midpoint = |a: I| 0.5 * a.lo + 0.5 * a.hi;
    let mut selected = Vec::with_capacity(group.len());
    let mut value = midpoint(rest.b);
    for (i, &j) in group.iter().enumerate() {
        let s = if p.mask & (1 << i) != 0 { 1.0 } else { 0.0 };
        let eta = if s == 1.0 { 1.0 } else { -1.0 };
        let coefficient = midpoint(problem.q[j]) * s - eta * p.phase[i];
        let row = if coefficient >= 0.0 {
            &problem.head[j].lower
        } else {
            &problem.head[j].upper
        };
        value += coefficient * midpoint(row.b);
        selected.push((coefficient, row));
    }
    for (j, x) in problem.root.iter().enumerate() {
        if j & 63 == 0 && !live(deadline) {
            return None;
        }
        let mut coefficient = midpoint(rest.a[j]);
        for (p, row) in &selected {
            coefficient += p * midpoint(row.a[j]);
        }
        value += coefficient * if coefficient >= 0.0 { x.lo } else { x.hi };
    }
    (good(value) && live(deadline)).then_some(value)
}

fn proposal_phase(
    problem: &SourceHeadProblem,
    group: &[usize],
    rest: &Affine,
    mask: usize,
    deadline: Instant,
    rounds: usize,
) -> Option<Proposal> {
    let mut p = Proposal {
        mask,
        phase: vec![0.0; group.len()],
    };
    let mut best = score_phase(problem, group, rest, &p, deadline)?;
    // Bounded first-party coordinate portfolio. These finite values do NOT
    // assert optimality and never prune a phase. Ranking is advice only;
    // every retained candidate is rechecked by phase_lower.
    for _ in 0..rounds.min(MAX_PROPOSAL_ROUNDS) {
        let mut moved = false;
        for (i, &j) in group.iter().enumerate() {
            let old = p.phase[i];
            let mut chosen = old;
            let q = (0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi)
                .abs()
                .max(f64::EPSILON);
            for scale in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
                if !live(deadline) {
                    return None;
                }
                p.phase[i] = q * scale;
                if let Some(b) = score_phase(problem, group, rest, &p, deadline) {
                    if b > best {
                        best = b;
                        chosen = p.phase[i];
                        moved = true
                    }
                }
            }
            p.phase[i] = chosen;
        }
        if !moved {
            break;
        }
    }
    Some(p)
}

struct RootGroupBound<'a> {
    source: &'a SourceHeadProblem,
    lower: f64,
    phases: usize,
    // Every mask already has a complete owned-source certificate. Optional
    // native work can improve any subset without creating a missing phase.
    phase_lower: Vec<f64>,
    proposals: Vec<Proposal>,
    group: Vec<usize>,
}
impl RootGroupBound<'_> {
    // This private bound is used before the owned source problem is changed
    // to another margin. No public importer can rebase it onto another frame.
    fn applies_to(&self, source: &SourceHeadProblem) -> bool {
        std::ptr::eq(self.source, source)
    }
    #[cfg(test)]
    fn retained_with(&self, other: &Self) -> Option<f64> {
        std::ptr::eq(self.source, other.source).then_some(self.lower.max(other.lower))
    }
}

// Reconstruct each directed error penalty from the verdict-grade pass.
// RowDots.err_dot is nearest-summed; tightened scalar y boxes cannot recover
// this penalty. Interpreting the stored coefficients exactly is safe only
// after charging the authored pass's complete e/eb envelope (including BN).
fn owned_row(
    pass: &PassOut,
    root: &[I],
    head_width: usize,
    neuron: usize,
    lower: bool,
    deadline: Instant,
) -> Option<Affine> {
    let e = pass.e.as_ref()?;
    if pass.a.dim() != (root.len(), head_width)
        || e.dim() != pass.a.dim()
        || pass.b.len() != head_width
        || pass.eb.len() != head_width
        || neuron >= head_width
        || !good(pass.b[neuron])
        || !good(pass.eb[neuron])
        || pass.eb[neuron] < 0.0
    {
        return None;
    }
    let mut penalty = I::p(pass.eb[neuron]);
    let mut a = Vec::with_capacity(root.len());
    for (j, x) in root.iter().enumerate() {
        if j & 63 == 0 && !live(deadline) {
            return None;
        }
        let coefficient = pass.a[[j, neuron]];
        let error = e[[j, neuron]];
        if !good(coefficient) || !good(error) || error < 0.0 || !x.valid() {
            return None;
        }
        penalty = penalty.add(I::p(error).mul(I::p(x.lo.abs().max(x.hi.abs())))?)?;
        a.push(I::p(coefficient));
    }
    let b = if lower {
        I::p(pass.b[neuron]).sub(I::p(penalty.hi))?
    } else {
        I::p(pass.b[neuron]).add(I::p(penalty.hi))?
    };
    (live(deadline) && b.valid()).then_some(Affine { a, b })
}

// Lane-private constructor. No imported affine row, caller-supplied box or
// proposal can create a SourceHeadProblem through the production API.
fn from_owned_pack(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    deadline: Instant,
) -> Option<SourceHeadProblem> {
    let net = eng.net;
    if !live(deadline)
        || !fp_qualified()
        || eng.root.mode != RoundMode::Outward
        || net.n_in == 0
        || net.n_in > MAX_INPUTS
        || !(2..=MAX_HEAD).contains(&net.n_y)
        || !(2..=10).contains(&net.n_out)
        || !pack.owned_for_head_groups(eng)
        || eng.root.lo.len() != net.n_in
        || eng.root.hi.len() != net.n_in
        || pack.ly0.len() != net.n_y
        || pack.uy0.len() != net.n_y
    {
        return None;
    }
    // Retain the authored head wiring; a final folded Gemm is deliberately
    // unsupported. The first folded Gemm is covered by the pass e/eb charge.
    let first = net.ops.get(net.i_gemm1)?;
    let head_relu = net.ops.get(net.i_gemm1.checked_add(1)?)?;
    let final_gemm = net.ops.get(net.i_gemm1.checked_add(2)?)?;
    let TwinOp::Gemm {
        bias_err,
        weight_rel_err,
        shape,
        ..
    } = first
    else {
        return None;
    };
    if shape.0 != net.n_y
        || bias_err.len() != net.n_y
        || bias_err.iter().any(|x| !good(*x) || *x < 0.0)
        || !good(*weight_rel_err)
        || *weight_rel_err < 0.0
        || !matches!(head_relu, TwinOp::Relu { input, .. } if *input == net.i_gemm1 + 1)
    {
        return None;
    }
    let TwinOp::Gemm {
        input,
        weight,
        bias,
        bias_err,
        weight_rel_err,
        shape,
    } = final_gemm
    else {
        return None;
    };
    if *input != net.i_gemm1 + 2
        || *shape != (net.n_out, net.n_y)
        || weight.len() != net.n_out.checked_mul(net.n_y)?
        || bias.len() != net.n_out
        || weight.iter().chain(bias).any(|x| !good(*x))
        || bias_err.len() != net.n_out
        || bias_err.iter().any(|x| *x != 0.0)
        || *weight_rel_err != 0.0
    {
        return None;
    }
    let root: Vec<_> = eng
        .root
        .lo
        .iter()
        .zip(&eng.root.hi)
        .map(|(&lo, &hi)| I { lo, hi })
        .collect();
    if root.iter().any(|x| !x.valid()) {
        return None;
    }
    let head_alpha = eng.root.head_alpha.as_deref();
    if head_alpha.is_some_and(|a| {
        a.len() != net.n_y || a.iter().any(|x| !good(*x) || !(0.0..=1.0).contains(x))
    }) {
        return None;
    }
    let mut head = Vec::with_capacity(net.n_y);
    for i in 0..net.n_y {
        let lower = owned_row(&pack.al, &root, net.n_y, i, true, deadline)?;
        let upper = owned_row(&pack.au, &root, net.n_y, i, false, deadline)?;
        // Derive the scalar enclosure from these very same directed rows.
        // No nearest-summed endpoint/error penalty becomes source authority.
        let y = I {
            lo: lower.lower(&root, deadline)?,
            hi: upper.upper(&root, deadline)?,
        };
        if !y.valid() || !live(deadline) {
            return None;
        }
        let alpha = head_alpha.map_or_else(|| f64::from(u8::from(y.hi > -y.lo)), |a| a[i]);
        head.push(Envelope {
            lower,
            upper,
            y,
            alpha,
        });
    }
    let problem = SourceHeadProblem {
        root,
        head,
        q: vec![I::p(0.0); net.n_y],
        bias: I::p(0.0),
    };
    problem.validate()?;
    live(deadline).then_some(problem)
}

// Construct authored margin coefficients with independent interval repair;
// nominal MarginBatch subtraction is a proposal, not a source equation.
fn bind_margin(
    problem: &mut SourceHeadProblem,
    eng: &BackwardEngine<'_>,
    target: usize,
    adversary: usize,
    deadline: Instant,
) -> Option<()> {
    if target >= eng.net.n_out || adversary >= eng.net.n_out || target == adversary {
        return None;
    }
    let (w, b, (_, width)) = eng.net.gemm2();
    for i in 0..width {
        if i & 63 == 0 && !live(deadline) {
            return None;
        }
        problem.q[i] = I::p(w[target * width + i]).sub(I::p(w[adversary * width + i]))?;
    }
    problem.bias = I::p(b[target]).sub(I::p(b[adversary]))?;
    live(deadline).then_some(())
}

// Selection only: rank source instability weighted by the authored margin,
// then favor an opposite-sign neighbor with a correlated root normal. No
// rounded score, tie, correlation or group omission changes replay authority.
fn source_groups(
    problem: &SourceHeadProblem,
    limit: usize,
    deadline: Instant,
) -> Option<Vec<Vec<usize>>> {
    let nominal = |q: I| 0.5 * q.lo + 0.5 * q.hi;
    let priority = |j: usize| {
        let y = problem.head[j].y;
        nominal(problem.q[j]).abs() * (y.hi / (y.hi - y.lo)) * (-y.lo)
    };
    let mut ranked: Vec<_> = (0..problem.head.len())
        .filter(|&j| {
            let y = problem.head[j].y;
            y.lo < 0.0 && y.hi > 0.0 && nominal(problem.q[j]) != 0.0
        })
        .collect();
    ranked.sort_by(|&a, &b| priority(b).total_cmp(&priority(a)).then(a.cmp(&b)));
    if ranked.len() < 2 {
        return Some(Vec::new());
    }
    let seed = ranked[0];
    let max_abs = |e: &Envelope| {
        e.lower
            .a
            .iter()
            .map(|a| a.lo.abs().max(a.hi.abs()))
            .fold(0.0, f64::max)
    };
    let seed_scale = max_abs(&problem.head[seed]);
    let mut partner = ranked[1];
    let mut best = f64::NEG_INFINITY;
    for &j in ranked.iter().skip(1).take(32) {
        if !live(deadline) {
            return None;
        }
        if nominal(problem.q[seed]).is_sign_positive() == nominal(problem.q[j]).is_sign_positive() {
            continue;
        }
        let scale = max_abs(&problem.head[j]);
        let mut dot = 0.0;
        let mut n1 = 0.0;
        let mut n2 = 0.0;
        if scale > 0.0 && seed_scale > 0.0 {
            for (i, (a, b)) in problem.head[seed]
                .lower
                .a
                .iter()
                .zip(&problem.head[j].lower.a)
                .enumerate()
            {
                if i & 63 == 0 && !live(deadline) {
                    return None;
                }
                let x = a.lo / seed_scale;
                let y = b.lo / scale;
                dot += x * y;
                n1 += x * x;
                n2 += y * y;
            }
        }
        let correlation = if n1 > 0.0 && n2 > 0.0 {
            (dot / (n1 * n2).sqrt()).max(0.0)
        } else {
            0.0
        };
        let score = priority(j) * (1.0 + correlation);
        if score.is_finite() && score > best {
            best = score;
            partner = j;
        }
    }
    let pair = vec![seed, partner];
    let mut result = vec![pair.clone()];
    if limit > 1 && ranked.len() >= 4 {
        let mut four = pair;
        four.extend(
            ranked
                .into_iter()
                .filter(|j| *j != seed && *j != partner)
                .take(2),
        );
        result.push(four);
    }
    live(deadline).then_some(result)
}

/// Refine only still-open margins of the producer-bound root pack.
///
/// The driver must call this once on its final immutable OUTER root frame,
/// with no inherited head/trunk restrictions, and apply the result immediately
/// to the same `(target, adversaries)` rows. Policy-off must bypass this call;
/// the base root-evaluation memo contains none of these optional results.
pub(super) fn improve_root_margins(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    target: usize,
    adversaries: &[usize],
    existing: &[f64],
    policy: HeadGroupPolicy,
    deadline: Instant,
) -> Option<HeadGroupReport> {
    improve_root_margins_with_budget(
        eng,
        pack,
        target,
        adversaries,
        existing,
        policy,
        deadline,
        &mut advice::AdviceBudget::new(),
    )
}

fn improve_root_margins_with_budget(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    target: usize,
    adversaries: &[usize],
    existing: &[f64],
    policy: HeadGroupPolicy,
    deadline: Instant,
    advice_budget: &mut advice::AdviceBudget,
) -> Option<HeadGroupReport> {
    let start = Instant::now();
    if !live(deadline)
        || !fp_qualified()
        || existing.len() != adversaries.len()
        || existing.iter().any(|x| !good(*x))
        || target >= eng.net.n_out
        || adversaries
            .iter()
            .enumerate()
            .any(|(i, &a)| a >= eng.net.n_out || a == target || adversaries[..i].contains(&a))
        || HeadGroupPolicy::new(
            policy.max_seconds,
            policy.groups_per_margin,
            policy.proposal_rounds,
        )
        .is_none()
    {
        return None;
    }
    let slice = start
        .checked_add(Duration::from_secs_f64(policy.max_seconds))?
        .min(deadline);
    let mut problem = from_owned_pack(eng, pack, slice)?;
    let mut report = HeadGroupReport {
        bounds: existing.to_vec(),
        groups_checked: 0,
        phases_checked: 0,
        classes_improved: 0,
        native_passes: 0,
        native_columns: 0,
        native_improved_phases: 0,
        advice_calls: 0,
        advice_candidates: 0,
        advice_improved_phases: 0,
        functional_advice_calls: 0,
        functional_advice_candidates: 0,
        functional_classes_improved: 0,
        elapsed_s: 0.0,
        candidates: Vec::new(),
    };
    // Complete the global portfolio across open margins before any one class
    // can spend the whole local slice in selected-group enumeration.
    let global = functional::refine_unconditional_portfolio(
        eng,
        pack,
        &mut problem,
        target,
        adversaries,
        existing,
        advice_budget,
        slice,
    )?;
    report.functional_advice_calls += global.report.calls;
    report.functional_advice_candidates += global.report.candidates;
    report.native_passes += global.report.native_passes;
    report.native_columns += global.report.native_columns;
    for (row, lower) in global.lower.into_iter().enumerate() {
        if let Some(lower) = lower {
            if !live(deadline) {
                return None;
            }
            if lower > report.bounds[row] {
                report.functional_classes_improved += 1;
            }
            report.bounds[row] = report.bounds[row].max(lower);
            report.candidates.push(HeadGroupCandidate {
                adversary: adversaries[row],
                group: Vec::new(),
                lower,
                phases: 1,
            });
            report.groups_checked += 1;
            report.phases_checked += 1;
        }
    }
    'margins: for (row, &adversary) in adversaries.iter().enumerate() {
        if report.bounds[row] > 0.0 {
            continue;
        }
        if !live(slice) {
            break;
        }
        if bind_margin(&mut problem, eng, target, adversary, slice).is_none() {
            if !live(slice) && live(deadline) {
                break 'margins;
            }
            return None;
        }
        let Some(groups) = source_groups(&problem, policy.groups_per_margin, slice) else {
            if !live(slice) && live(deadline) {
                break 'margins;
            }
            return None;
        };
        for group in groups {
            let Some(mut bound) =
                propose_and_check(&problem, &group, slice, policy.proposal_rounds)
            else {
                // A spent local slice discards only this incomplete cover.
                // Completed same-source covers remain valid while the original
                // driver deadline is live. Other refusals decline the family.
                if !live(slice) && live(deadline) {
                    break 'margins;
                }
                return None;
            };
            if !bound.applies_to(&problem) || !good(bound.lower) || !live(deadline) {
                return None;
            }
            let refined =
                native::refine_complete_cover(eng, pack, &problem, &group, &mut bound, slice)?;
            report.native_passes += refined.passes;
            report.native_columns += refined.columns;
            report.native_improved_phases += refined.improved_phases;
            let global_phase = functional::refine_complete_cover(
                eng,
                pack,
                &problem,
                &group,
                &mut bound,
                advice_budget,
                slice,
            )?;
            report.functional_advice_calls += global_phase.calls;
            report.functional_advice_candidates += global_phase.candidates;
            report.native_passes += global_phase.native_passes;
            report.native_columns += global_phase.native_columns;
            report.native_improved_phases += global_phase.improved_phases;
            let advised = advice::refine_complete_cover(
                eng,
                pack,
                &problem,
                &group,
                &mut bound,
                advice_budget,
                slice,
            )?;
            report.advice_calls += advised.calls;
            report.advice_candidates += advised.candidates;
            report.advice_improved_phases += advised.improved_phases;
            report.native_passes += advised.native_passes;
            report.native_columns += advised.native_columns;
            if !live(deadline) {
                return None;
            }
            report.bounds[row] = report.bounds[row].max(bound.lower);
            report.groups_checked += 1;
            report.phases_checked += bound.phases;
            report.candidates.push(HeadGroupCandidate {
                adversary,
                group,
                lower: bound.lower,
                phases: bound.phases,
            });
        }
    }
    report.classes_improved = report
        .bounds
        .iter()
        .zip(existing)
        .filter(|(a, b)| a > b)
        .count();
    report.elapsed_s = start.elapsed().as_secs_f64();
    (fp_qualified() && live(deadline)).then_some(report)
}

#[cfg(test)]
fn check_complete<'a>(
    problem: &'a SourceHeadProblem,
    group: &[usize],
    proposals: &[Proposal],
    deadline: Instant,
) -> Option<RootGroupBound<'a>> {
    problem.validate()?;
    if !(2..=MAX_GROUP).contains(&group.len())
        || group.iter().any(|j| *j >= problem.head.len())
        || group
            .iter()
            .enumerate()
            .any(|(i, j)| group[..i].contains(j))
    {
        return None;
    }
    let rest = remainder(problem, group, deadline)?;
    check_complete_with_remainder(problem, group, &rest, proposals, deadline)
}

// `rest` is created only by our outward remainder() from this exact owned
// problem/group, before the rounded multiplier proposer sees it. It is not
// part of the proposal transcript, and cannot be supplied by an external API.
// Reuse that source certificate rather than rebuilding all unselected rows.
fn check_complete_with_remainder<'a>(
    problem: &'a SourceHeadProblem,
    group: &[usize],
    rest: &Affine,
    proposals: &[Proposal],
    deadline: Instant,
) -> Option<RootGroupBound<'a>> {
    if !live(deadline)
        || !fp_qualified()
        || !rest.valid(problem.root.len())
        || !(2..=MAX_GROUP).contains(&group.len())
        || group.iter().any(|j| *j >= problem.head.len())
        || group
            .iter()
            .enumerate()
            .any(|(i, j)| group[..i].contains(j))
        || proposals.len() != 1usize << group.len()
    {
        return None;
    }
    let mut seen = vec![false; 1 << group.len()];
    let mut mask_lower = vec![f64::NAN; seen.len()];
    let mut by_mask = vec![
        Proposal {
            mask: 0,
            phase: Vec::new()
        };
        seen.len()
    ];
    let mut lower = f64::INFINITY;
    for p in proposals {
        if p.mask >= seen.len() || seen[p.mask] {
            return None;
        }
        seen[p.mask] = true;
        let phase = phase_lower(problem, group, rest, p, deadline)?;
        mask_lower[p.mask] = phase;
        by_mask[p.mask] = p.clone();
        lower = lower.min(phase);
    }
    (live(deadline) && seen.iter().all(|v| *v) && good(lower)).then_some(RootGroupBound {
        source: problem,
        lower,
        phases: proposals.len(),
        phase_lower: mask_lower,
        proposals: by_mask,
        group: group.to_vec(),
    })
}

fn propose_and_check<'a>(
    problem: &'a SourceHeadProblem,
    group: &[usize],
    deadline: Instant,
    rounds: usize,
) -> Option<RootGroupBound<'a>> {
    problem.validate()?;
    if !(2..=MAX_GROUP).contains(&group.len())
        || group.iter().any(|j| *j >= problem.head.len())
        || group
            .iter()
            .enumerate()
            .any(|(i, j)| group[..i].contains(j))
    {
        return None;
    }
    let rest = remainder(problem, group, deadline)?;
    let proposals = (0..1usize << group.len())
        .map(|mask| proposal_phase(problem, group, &rest, mask, deadline, rounds))
        .collect::<Option<Vec<_>>>()?;
    check_complete_with_remainder(problem, group, &rest, &proposals, deadline)
}

#[cfg(test)]
#[path = "head_group_tests.rs"]
mod tests;
