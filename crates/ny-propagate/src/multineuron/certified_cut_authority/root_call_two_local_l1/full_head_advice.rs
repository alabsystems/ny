// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded full-head multiplier ADVICE. Ordinary finite f64 scores are neither
//! outward bounds nor certificates. Both fresh certificates and the subsequent
//! checked full folds retain all proof authority. No source cache is consulted.
#[cfg(test)]
use super::head_source::Bn;
use super::head_source::{resolve, target_slopes, Slopes, Source};
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Quote {
    pub(super) input_dim: usize,
    pub(super) target_dim: usize,
    pub(super) output_dim: usize,
    pub(super) work: u64,
    pub(super) scratch_bytes: usize,
}
#[derive(Debug, Clone, Copy)]
pub(super) struct Decision {
    pub(super) multipliers: [f32; 2],
    pub(super) branch_upper: [f32; 2],
    pub(super) support_planes: usize,
    pub(super) lp_candidates: usize,
    pub(super) original_score: f64,
    pub(super) chosen_score: f64,
    pub(super) score_evaluations: usize,
    pub(super) full_scans: usize,
}

pub(super) fn quote(e: usize, d: usize, o: usize) -> Option<Quote> {
    if !(1..=65536).contains(&e)
        || !(2..=8192).contains(&d)
        || !(1..=8192).contains(&o)
        || d.checked_mul(o)? > 1 << 20
    {
        return None;
    }
    let (ee, dd, oo) = (
        u64::try_from(e).ok()?,
        u64::try_from(d).ok()?,
        u64::try_from(o).ok()?,
    );
    let work = 160u64
        .checked_mul(dd)?
        .checked_mul(ee.checked_add(oo)?.checked_add(1)?)?
        .checked_add(2048u64.checked_mul(ee.checked_add(1)?)?)?
        .checked_add(8192u64.checked_mul(ee.checked_add(dd)?.checked_add(1)?)?)?
        .checked_add(4 << 20)?
        .checked_add(KELLEY_WORK)?;
    let scratch_bytes = 32usize
        .checked_mul(e)?
        .checked_add(80usize.checked_mul(d)?)?
        .checked_add(64 << 10)?;
    Some(Quote {
        input_dim: e,
        target_dim: d,
        output_dim: o,
        work,
        scratch_bytes,
    })
}
fn safe(v: f64) -> Option<f64> {
    ny_core::is_crown_coeff_safe_f64(v).then_some(v)
}
fn scalar(v: f32) -> Option<f64> {
    ny_core::is_crown_coeff_safe(v).then_some(f64::from(v))
}
fn add(a: f64, b: f64) -> Option<f64> {
    safe(a + b)
}
fn mul(a: f64, b: f64) -> Option<f64> {
    safe(a * b)
}
fn poll(guard: &mut Poll<'_>, i: usize) -> Option<()> {
    (!i.is_multiple_of(32) || guard.check()).then_some(())
}
// Four contraction vectors plus one72B row record per target cell. All
// allocations are dropped before either genuine checked fold starts.
#[derive(Clone, Copy, Default)]
struct Row {
    a: f64,
    q0: f64,
    q: [f64; 2],
    original: f64,
    scale: f64,
    radius: f64,
    selected_weight: [f64; 2],
}
struct Proxy<'a> {
    lower: &'a [f32],
    upper: &'a [f32],
    u: Vec<f64>,
    v: [Vec<f64>; 2],
    original: Vec<f64>,
    rows: Vec<Row>,
    beta0: f64,
    beta: [f64; 2],
    original_beta: f64,
    lambda: [f32; 2],
    branch_hi: [f64; 2],
    selected: [[usize; 8]; 2],
    active: [usize; 2],
}
const MAX_PLANES: usize = 16;
// Fifteen LP proposals: sum_m=1..15 [4+4*C(m,2)+C(m,3)]=4,120.
// At most49,732 plane values are evaluated across that full schedule.
const MAX_LP_CANDIDATES: usize = 4120;
const KELLEY_WORK: u64 = 16 << 20;
fn scratch_actual(e: usize, d: usize) -> Option<usize> {
    e.checked_mul(4 * size_of::<f64>())?
        .checked_add(d.checked_mul(size_of::<Row>())?)?
        .checked_add(size_of::<Proxy<'_>>())?
        .checked_add(size_of::<Source<'_>>())?
        .checked_add(size_of::<Slopes<'_>>())?
        .checked_add(size_of::<Envelope>())?
        .checked_add(size_of::<[[f32; 2]; MAX_PLANES]>())?
        // Decisions, scalar reductions, one tiny solve, source borrows and
        // nested frames. The quote retains another32KiB beyond this reserve.
        .checked_add(32 << 10)
}
#[allow(clippy::too_many_arguments)]
fn contract<'a>(
    s: &Source<'a>,
    slopes: Slopes<'_>,
    paid: Quote,
    active: [usize; 2],
    ids: [[usize; 8]; 2],
    weights: [[f32; 8]; 2],
    lambda: [f32; 2],
    bounds: [f32; 2],
    guard: &mut Poll<'_>,
) -> Option<Proxy<'a>> {
    let (e, d, o) = (paid.input_dim, paid.target_dim, paid.output_dim);
    if quote(e, d, o) != Some(paid)
        || scratch_actual(e, d)? > paid.scratch_bytes
        || !valid_selection(active, ids, weights, d)
        || s.matrix.len() != d.checked_mul(e)?
        || s.lower.len() != e
        || s.upper.len() != e
        || s.final_matrix.len() != o.checked_mul(d)?
        || s.objective.len() != o
        || s.bias.is_some_and(|v| v.len() != d)
        || s.final_bias.is_some_and(|v| v.len() != o)
        || s.bn.is_some_and(|v| {
            [
                v.scale.len(),
                v.shift.len(),
                v.scale_error.len(),
                v.bias_error.len(),
                v.lower.len(),
                v.upper.len(),
            ]
            .iter()
            .any(|n| *n != d)
        })
        || !guard.check()
    {
        return None;
    }
    let l = [scalar(lambda[0])?, scalar(lambda[1])?];
    if l.iter().any(|v| *v <= 0.) || bounds.iter().any(|v| !v.is_finite() || *v < 0.) {
        return None;
    }
    let b = [scalar(bounds[0])?, scalar(bounds[1])?];
    let mut constant = 0.;
    for k in 0..o {
        poll(guard, k)?;
        constant = add(
            constant,
            mul(
                scalar(s.objective[k])?,
                s.final_bias.map_or(Some(0.), |v| scalar(v[k]))?,
            )?,
        )?;
    }
    let beta = [-mul(l[0], b[0])?, -mul(l[1], b[1])?];
    let mut p = Proxy {
        lower: s.lower,
        upper: s.upper,
        u: vec![0.; e],
        v: [vec![0.; e], vec![0.; e]],
        original: vec![0.; e],
        rows: Vec::with_capacity(d),
        beta0: constant,
        beta,
        original_beta: add(add(constant, beta[0])?, beta[1])?,
        lambda,
        branch_hi: [1.; 2],
        selected: ids,
        active,
    };
    let actual =
        p.u.capacity()
            .checked_add(p.v[0].capacity())?
            .checked_add(p.v[1].capacity())?
            .checked_add(p.original.capacity())?
            .checked_mul(size_of::<f64>())?
            .checked_add(p.rows.capacity().checked_mul(size_of::<Row>())?)?
            .checked_add(scratch_actual(0, 0)?)?;
    if actual > paid.scratch_bytes {
        return None;
    }
    for k in 0..e {
        poll(guard, k)?;
        if scalar(s.lower[k])? > scalar(s.upper[k])? {
            return None;
        }
    }
    for j in 0..d {
        poll(guard, j)?;
        let mut a = 0.;
        for k in 0..o {
            poll(guard, k)?;
            a = add(
                a,
                mul(scalar(s.objective[k])?, scalar(s.final_matrix[k * d + j])?)?,
            )?;
        }
        let mut w = [0.; 2];
        for g in 0..2 {
            for k in 0..active[g] {
                if ids[g][k] == j {
                    w[g] = scalar(weights[g][k])?;
                }
            }
        }
        let selected = w[0] > 0. || w[1] > 0.;
        if selected && a >= 0. {
            return None;
        }
        let lw = [mul(l[0], w[0])?, mul(l[1], w[1])?];
        let post = add(a, mul(2., add(lw[0], lw[1])?)?)?;
        let (os, ob) = slopes.branch(j, post)?;
        let original_q = add(mul(post, os)?, -add(lw[0], lw[1])?)?;
        // Selected rows use the shared negative-post branch. Untouched rows
        // retain their actual descriptor and full parent cancellation.
        let (slope, intercept) = if selected {
            (
                scalar(slopes.negative[j])?,
                scalar(slopes.negative_bias[j])?,
            )
        } else {
            slopes.branch(j, a)?
        };
        let q0 = mul(a, slope)?;
        let q = [
            mul(lw[0], add(2. * slope, -1.)?)?,
            mul(lw[1], add(2. * slope, -1.)?)?,
        ];
        let i0 = mul(a, intercept)?;
        let i = [mul(2. * lw[0], intercept)?, mul(2. * lw[1], intercept)?];
        for g in 0..2 {
            if w[g] > 0. {
                p.branch_hi[g] = p.branch_hi[g].min(safe(-a / (2. * lw[g]))?);
            }
        }
        let (scale, shift, radius) = if let Some(bn) = s.bn {
            let (lo, hi, se, be) = (
                scalar(bn.lower[j])?,
                scalar(bn.upper[j])?,
                scalar(bn.scale_error[j])?,
                scalar(bn.bias_error[j])?,
            );
            if lo > hi || se < 0. || be < 0. {
                return None;
            }
            (
                scalar(bn.scale[j])?,
                scalar(bn.shift[j])?,
                add(mul(se, lo.abs().max(hi.abs()))?, be)?,
            )
        } else {
            (1., 0., 0.)
        };
        let z_bias = add(
            mul(scale, s.bias.map_or(Some(0.), |v| scalar(v[j]))?)?,
            shift,
        )?;
        p.beta0 = add(p.beta0, add(i0, mul(q0, z_bias)?)?)?;
        for g in 0..2 {
            p.beta[g] = add(p.beta[g], add(i[g], mul(q[g], z_bias)?)?)?;
        }
        p.original_beta = add(
            p.original_beta,
            add(mul(post, ob)?, mul(original_q, z_bias)?)?,
        )?;
        let qs = [
            mul(q0, scale)?,
            mul(q[0], scale)?,
            mul(q[1], scale)?,
            mul(original_q, scale)?,
        ];
        for k in 0..e {
            poll(guard, k)?;
            let weight = scalar(s.matrix[j * e + k])?;
            p.u[k] = add(p.u[k], mul(qs[0], weight)?)?;
            for g in 0..2 {
                p.v[g][k] = add(p.v[g][k], mul(qs[g + 1], weight)?)?;
            }
            p.original[k] = add(p.original[k], mul(qs[3], weight)?)?;
        }
        p.rows.push(Row {
            a,
            q0,
            q,
            original: original_q,
            scale,
            radius,
            selected_weight: w,
        });
    }
    guard.check().then_some(p)
}

#[derive(Clone, Copy, Debug)]
struct Evaluation {
    value: f64,
    gradient: [f64; 2],
}
impl Proxy<'_> {
    fn in_branch(&self, value: [f32; 2]) -> bool {
        (0..2).all(|g| {
            let v = f64::from(value[g]);
            value[g].is_finite()
                && v >= f64::from(self.lambda[g]) / 16.
                && value[g] <= self.lambda[g]
                && value[g] > 0.
                && self.selected[g][..self.active[g]].iter().all(|&j| {
                    let r = &self.rows[j];
                    r.a + 2. * v * r.selected_weight[g] <= 0.
                })
        })
    }
    fn theta(&self, value: [f32; 2]) -> [f64; 2] {
        [
            f64::from(value[0]) / f64::from(self.lambda[0]),
            f64::from(value[1]) / f64::from(self.lambda[1]),
        ]
    }
    fn evaluate(
        &self,
        value: [f32; 2],
        original: bool,
        guard: &mut Poll<'_>,
    ) -> Option<Evaluation> {
        let theta = self.theta(value);
        let mut sum = if original {
            self.original_beta
        } else {
            add(
                add(self.beta0, mul(theta[0], self.beta[0])?)?,
                mul(theta[1], self.beta[1])?,
            )?
        };
        let mut gradient = if original { [0.; 2] } else { self.beta };
        for k in 0..self.u.len() {
            poll(guard, k)?;
            let a = if original {
                self.original[k]
            } else {
                add(
                    add(self.u[k], mul(theta[0], self.v[0][k])?)?,
                    mul(theta[1], self.v[1][k])?,
                )?
            };
            // A single active endpoint contributes BOTH components. At zero
            // choose lower for both, never two unrelated one-sided partials.
            let endpoint = f64::from(if a < 0. { self.upper[k] } else { self.lower[k] });
            sum = add(sum, mul(a, endpoint)?)?;
            if !original {
                for g in 0..2 {
                    gradient[g] = add(gradient[g], mul(self.v[g][k], endpoint)?)?;
                }
            }
        }
        for (j, r) in self.rows.iter().enumerate() {
            poll(guard, j)?;
            let q = if original {
                r.original
            } else {
                add(add(r.q0, mul(theta[0], r.q[0])?)?, mul(theta[1], r.q[1])?)?
            };
            sum = add(sum, -mul(r.radius, q.abs())?)?;
            if !original {
                // At a zero residual choose ONE sign(+1) for the entire vector.
                let factor = if q < 0. { r.radius } else { -r.radius };
                for g in 0..2 {
                    gradient[g] = add(gradient[g], mul(factor, r.q[g])?)?;
                }
            }
        }
        guard.check().then_some(Evaluation {
            value: sum,
            gradient,
        })
    }
    fn score(&self, value: [f32; 2], original: bool, guard: &mut Poll<'_>) -> Option<f64> {
        Some(self.evaluate(value, original, guard)?.value)
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Plane {
    slope: [f64; 2],
    intercept: f64,
}
impl Plane {
    fn sample(point: [f64; 2], evaluation: Evaluation) -> Option<Self> {
        Some(Self {
            slope: evaluation.gradient,
            intercept: add(
                add(evaluation.value, -mul(point[0], evaluation.gradient[0])?)?,
                -mul(point[1], evaluation.gradient[1])?,
            )?,
        })
    }
    fn value(self, point: [f64; 2]) -> Option<f64> {
        add(
            add(self.intercept, mul(self.slope[0], point[0])?)?,
            mul(self.slope[1], point[1])?,
        )
    }
}
#[derive(Clone, Copy)]
struct Rectangle {
    low: [f64; 2],
    high: [f64; 2],
}
impl Rectangle {
    fn contains(self, p: [f64; 2]) -> bool {
        (0..2).all(|g| p[g].is_finite() && p[g] >= self.low[g] && p[g] <= self.high[g])
    }
}
struct Envelope {
    planes: [Plane; MAX_PLANES],
    len: usize,
}
impl Envelope {
    fn new() -> Self {
        Self {
            planes: [Plane::default(); MAX_PLANES],
            len: 0,
        }
    }
    fn push(&mut self, p: Plane) -> Option<()> {
        if self.len == MAX_PLANES {
            return None;
        }
        self.planes[self.len] = p;
        self.len += 1;
        Some(())
    }
    fn min_value(&self, p: [f64; 2]) -> Option<f64> {
        let mut result = f64::INFINITY;
        for plane in &self.planes[..self.len] {
            result = result.min(plane.value(p)?);
        }
        safe(result)
    }
    fn propose(
        &self,
        rect: Rectangle,
        visits: &mut usize,
        guard: &mut Poll<'_>,
    ) -> Option<[f64; 2]> {
        if self.len == 0 || self.len >= MAX_PLANES || !guard.check() {
            return None;
        }
        let mut best: Option<([f64; 2], f64)> = None;
        let mut visit = |p: [f64; 2]| -> Option<()> {
            *visits = visits.checked_add(1)?;
            if *visits > MAX_LP_CANDIDATES {
                return None;
            }
            poll(guard, *visits)?;
            if rect.contains(p) {
                let score = self.min_value(p)?;
                if best.is_none_or(|(old, v)| {
                    score > v || (score == v && (p[0], p[1]) < (old[0], old[1]))
                }) {
                    best = Some((p, score));
                }
            }
            Some(())
        };
        // Vertical rectangle constraints + one support plane: all corners.
        for x in [rect.low[0], rect.high[0]] {
            for y in [rect.low[1], rect.high[1]] {
                visit([x, y])?;
            }
        }
        for i in 0..self.len {
            for j in i + 1..self.len {
                let equation = plane_difference(self.planes[i], self.planes[j]);
                // Two support planes + one rectangle edge. Always charge/poll
                // all four slots, including parallel/degenerate skipped equations.
                for axis in 0..2 {
                    for side in 0..2 {
                        let mut p = [f64::NAN; 2];
                        if let Some(eq) = equation {
                            let fixed = if side == 0 {
                                rect.low[axis]
                            } else {
                                rect.high[axis]
                            };
                            let free = 1 - axis;
                            if eq[free].abs() > 64. * f64::EPSILON {
                                p[axis] = fixed;
                                p[free] = (eq[2] - eq[axis] * fixed) / eq[free];
                            }
                        }
                        visit(p)?;
                    }
                }
                // Three support planes: normalized2x2 differences. Near-parallel
                // or nonfinite geometry is skipped, never used as a bound/result.
                for k in j + 1..self.len {
                    let p = equation
                        .and_then(|a| {
                            plane_difference(self.planes[i], self.planes[k])
                                .and_then(|b| intersection(a, b))
                        })
                        .unwrap_or([f64::NAN; 2]);
                    visit(p)?;
                }
            }
        }
        guard.check().then_some(best?.0)
    }
}
fn plane_difference(a: Plane, b: Plane) -> Option<[f64; 3]> {
    let row = [
        a.slope[0] - b.slope[0],
        a.slope[1] - b.slope[1],
        b.intercept - a.intercept,
    ];
    let norm = row[0].abs().max(row[1].abs());
    if !norm.is_finite() || norm == 0. {
        return None;
    }
    let normalized = [row[0] / norm, row[1] / norm, row[2] / norm];
    normalized
        .iter()
        .all(|x| x.is_finite())
        .then_some(normalized)
}
fn intersection(a: [f64; 3], b: [f64; 3]) -> Option<[f64; 2]> {
    let det = a[0] * b[1] - a[1] * b[0];
    if !det.is_finite() || det.abs() <= 64. * f64::EPSILON {
        return None;
    }
    let p = [
        (a[2] * b[1] - a[1] * b[2]) / det,
        (a[0] * b[2] - a[2] * b[0]) / det,
    ];
    p.iter().all(|x| x.is_finite()).then_some(p)
}
fn same_pair(a: [f32; 2], b: [f32; 2]) -> bool {
    (0..2).all(|g| a[g].to_bits() == b[g].to_bits())
}
fn search(p: &Proxy<'_>, guard: &mut Poll<'_>) -> Option<Decision> {
    let original_score = p.score(p.lambda, true, guard)?;
    let mut out = Decision {
        multipliers: p.lambda,
        branch_upper: p.lambda,
        original_score,
        chosen_score: original_score,
        score_evaluations: 1,
        full_scans: 1,
        support_planes: 0,
        lp_candidates: 0,
    };
    let mut low = [0.; 2];
    let mut high = [0.; 2];
    for g in 0..2 {
        let lo = f64::from(p.lambda[g]) / 16.;
        let hi = f64::from(p.lambda[g]) * p.branch_hi[g].min(1.);
        low[g] = lo as f32;
        if f64::from(low[g]) < lo {
            low[g] = next_up_f32(low[g]);
        }
        high[g] = hi as f32;
        if f64::from(high[g]) > hi {
            high[g] = ny_tensor::next_down_f32(high[g]);
        }
    }
    // At most one representable downward guard correction, same as the old
    // seam. Empty guarded domain deliberately retains separately scored original.
    for g in 0..2 {
        let valid = |v: f32| v.is_finite() && v > 0. && self_branch_group(p, g, v);
        if high[g].is_finite() && high[g] > 0. && !valid(high[g]) {
            high[g] = ny_tensor::next_down_f32(high[g]);
        }
    }
    out.branch_upper = high;
    if !p.in_branch(low) || !p.in_branch(high) || (0..2).any(|g| low[g] > high[g]) {
        return guard.check().then_some(out);
    }
    let rect = Rectangle {
        low: p.theta(low),
        high: p.theta(high),
    };
    let mut envelope = Envelope::new();
    let mut visited = [[0.; 2]; MAX_PLANES];
    let mut value = high; // Both coordinates start INSIDE their guarded branches.
    for iteration in 0..MAX_PLANES {
        if !guard.check() || !p.in_branch(value) {
            return None;
        }
        visited[iteration] = value;
        let evaluation = p.evaluate(value, false, guard)?;
        out.score_evaluations += 1;
        out.full_scans += 1;
        if evaluation.value > out.chosen_score {
            out.multipliers = value;
            out.chosen_score = evaluation.value;
        }
        envelope.push(Plane::sample(p.theta(value), evaluation)?)?;
        out.support_planes = envelope.len;
        if iteration + 1 == MAX_PLANES {
            break;
        }
        let point = envelope.propose(rect, &mut out.lp_candidates, guard)?;
        let mut next = [0.; 2];
        for g in 0..2 {
            next[g] = ((point[g] * f64::from(p.lambda[g])) as f32).clamp(low[g], high[g]);
        }
        if !p.in_branch(next) {
            return None;
        }
        // Rounded repeats end the declared search. Never refund its work or
        // replace an entered source/deadline failure with the old candidate.
        if visited[..=iteration].iter().any(|v| same_pair(*v, next)) {
            break;
        }
        value = next;
    }
    (guard.check()
        && out.score_evaluations <= 17
        && out.full_scans <= 17
        && out.support_planes <= MAX_PLANES
        && out.lp_candidates <= MAX_LP_CANDIDATES)
        .then_some(out)
}
fn self_branch_group(p: &Proxy<'_>, g: usize, value: f32) -> bool {
    p.selected[g][..p.active[g]].iter().all(|&j| {
        let r = &p.rows[j];
        r.a + 2. * f64::from(value) * r.selected_weight[g] <= 0.
    })
}

pub(super) fn choose(
    context: &TwoLocalContext<'_, '_>,
    fresh: &FreshTwoLocalSupport<'_, '_>,
    paid: Quote,
    guard: &mut Poll<'_>,
) -> Option<Decision> {
    let q = context.quote;
    if q.full_head_advice != Some(paid)
        || q.source_shape != [paid.input_dim, paid.target_dim, paid.output_dim]
        || !std::ptr::eq(fresh.seal, &raw const q.seal)
        || !guard.check()
    {
        return None;
    }
    let source = resolve(
        &q.base,
        [paid.input_dim, paid.target_dim, paid.output_dim],
        guard,
    )?;
    let slopes = target_slopes(context, paid.target_dim, guard)?;
    let proxy = contract(
        &source,
        slopes,
        paid,
        q.active,
        q.neurons,
        q.weights,
        q.multipliers,
        [fresh.arithmetic[0].bound(), fresh.arithmetic[1].bound()],
        guard,
    )?;
    search(&proxy, guard)
}

#[cfg(test)]
#[path = "full_head_advice_tests.rs"]
mod tests;
