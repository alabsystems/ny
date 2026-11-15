// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Source-box affine/ReLU upper supports, with numerical proposals separated
//! from authority.  For exact authored `r = max(w*x + b, 0)` this certifies
//! `a*x - r >= k`.  `a` contains arbitrary exact-f64 proposals. The residual is exactly the
//! minimum of two affine functions. Both affine box minima are replayed with
//! outward arithmetic, so no numerical proposal can supply an authoritative
//! right-hand side.
//!
//! This is the multivariate-input single-neuron hull studied by Anderson et
//! al. and Tjandraatmadja et al.; it is not a claim of a new convex-hull theorem.
//! The source-only portfolio below uses box-vertex chain marginals (Lovasz
//! upper facets).  An optional point can propose a separating chain, but does
//! not change the certifier or grant primal/witness authority.

use super::lp_certificate::suitable_fp_environment;
use super::lp_certificate::LinearRow;
use super::{add_down, add_up, mul_down, mul_up, Enclosure, PairedChain, Stage};
use std::time::Instant;

const MAX_WIDTH: usize = 512;
const MAX_PORTFOLIO: usize = 64;
const MAX_MAGNITUDE: f64 = 1e100;

#[derive(Clone, Debug)]
pub(super) struct AffineReluCut {
    pub(super) slopes: Vec<f64>,
    /// Exact-f64 GEQ right-hand side, already repaired outward.
    pub(super) intercept: f64,
}

fn admissible(x: f64) -> bool {
    x.is_finite() && x.abs() <= MAX_MAGNITUDE
}

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}

fn valid(weights: &[f64], bias: f64, bounds: &[(f64, f64)]) -> bool {
    !weights.is_empty()
        && weights.len() <= MAX_WIDTH
        && bounds.len() == weights.len()
        && admissible(bias)
        && weights.iter().copied().all(admissible)
        && bounds
            .iter()
            .all(|&(l, u)| admissible(l) && admissible(u) && l <= u)
}

/// Outward lower endpoint of the product of two enclosed scalars.
fn product_lower(al: f64, au: f64, bl: f64, bu: f64) -> Option<f64> {
    let products = [
        mul_down(al, bl),
        mul_down(al, bu),
        mul_down(au, bl),
        mul_down(au, bu),
    ];
    products
        .iter()
        .all(|x| x.is_finite())
        .then(|| products.into_iter().fold(f64::INFINITY, f64::min))
}

fn product_interval(al: f64, au: f64, value: f64) -> Option<(f64, f64)> {
    let lower = if value >= 0.0 {
        mul_down(al, value)
    } else {
        mul_down(au, value)
    };
    let upper = if value >= 0.0 {
        mul_up(au, value)
    } else {
        mul_up(al, value)
    };
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some((lower, upper))
}

/// Certify an arbitrary proposed upper plane against the exact authored
/// affine/ReLU graph over a certified SOURCE predecessor box.
pub(super) fn certified_affine_relu_cut(
    weights: &[f64],
    bias: f64,
    bounds: &[(f64, f64)],
    slopes: &[f64],
    deadline: Option<Instant>,
) -> Option<f64> {
    if !valid(weights, bias, bounds)
        || slopes.len() != weights.len()
        || !slopes.iter().copied().all(admissible)
        || !suitable_fp_environment()
        || !alive(deadline)
    {
        return None;
    }
    // a*x - max(w*x+b,0) = min(a*x, (a-w)*x-b).
    // Minima commute, so the global box minimum needs no phase solver.
    let mut zero = 0.0;
    let mut linear = -bias;
    for ((&a, &w), &(l, u)) in slopes.iter().zip(weights).zip(bounds) {
        if !alive(deadline) {
            return None;
        }
        zero = add_down(zero, product_lower(a, a, l, u)?);
        // The authored w and proposed a are exact f64 values, but their
        // subtraction need not be. Retain its enclosing interval explicitly.
        let cl = add_down(a, -w);
        let cu = add_up(a, -w);
        linear = add_down(linear, product_lower(cl, cu, l, u)?);
    }
    let result = zero.min(linear);
    (admissible(result) && alive(deadline)).then_some(result)
}

/// A chain of box vertices yields a candidate upper facet for the hinge of
/// a positive weighted sum. These coefficients are deliberately uncertified;
/// all rounding is repaired by `certified_affine_relu_cut` afterwards.
pub(super) fn chain_slopes(
    weights: &[f64],
    bias: f64,
    bounds: &[(f64, f64)],
    order: &[usize],
) -> Option<Vec<f64>> {
    let mut pre = bias;
    let mut widths = Vec::with_capacity(weights.len());
    for (&w, &(l, u)) in weights.iter().zip(bounds) {
        pre += w * if w >= 0.0 { l } else { u };
        widths.push(w.abs() * (u - l));
    }
    if !pre.is_finite() || widths.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut slopes = vec![0.0; weights.len()];
    for &i in order {
        let d = widths[i];
        if d > 0.0 {
            // One crossing coordinate has a fractional marginal; the other
            // marginals are 0 or d. Clamp only affects proposal quality.
            let fraction = ((pre + d).max(0.0) - pre.max(0.0)) / d;
            slopes[i] = weights[i] * fraction.clamp(0.0, 1.0);
            pre += d;
        }
    }
    slopes.iter().copied().all(admissible).then_some(slopes)
}

fn append_unique(result: &mut Vec<Vec<f64>>, candidate: Vec<f64>, limit: usize) {
    if result.len() < limit && !result.iter().any(|old| old == &candidate) {
        result.push(candidate);
    }
}

/// Deterministic SOURCE-only upper-facet portfolio. `limit` is a resource cap,
/// not a completeness promise. It introduces no variables or binary phases.
pub(super) fn candidate_slopes(
    weights: &[f64],
    bias: f64,
    bounds: &[(f64, f64)],
    limit: usize,
) -> Option<Vec<Vec<f64>>> {
    if !valid(weights, bias, bounds) || limit == 0 || limit > MAX_PORTFOLIO {
        return None;
    }
    let n = weights.len();
    let mut result = Vec::new();
    let mut order: Vec<_> = (0..n).collect();
    append_unique(
        &mut result,
        chain_slopes(weights, bias, bounds, &order)?,
        limit,
    );
    order.reverse();
    append_unique(
        &mut result,
        chain_slopes(weights, bias, bounds, &order)?,
        limit,
    );
    order.sort_by(|&i, &j| {
        let di = weights[i].abs() * (bounds[i].1 - bounds[i].0);
        let dj = weights[j].abs() * (bounds[j].1 - bounds[j].0);
        dj.total_cmp(&di).then(i.cmp(&j))
    });
    append_unique(
        &mut result,
        chain_slopes(weights, bias, bounds, &order)?,
        limit,
    );
    order.reverse();
    append_unique(
        &mut result,
        chain_slopes(weights, bias, bounds, &order)?,
        limit,
    );
    // Negative-weight coordinates tend to have high sign-flipped normalized
    // values for sparse nonnegative predecessor activations. This source-only
    // order is particularly useful near zero without using a diagnostic point.
    order.sort_by(|&i, &j| {
        (weights[i] >= 0.0)
            .cmp(&(weights[j] >= 0.0))
            .then(i.cmp(&j))
    });
    append_unique(
        &mut result,
        chain_slopes(weights, bias, bounds, &order)?,
        limit,
    );
    order.reverse();
    append_unique(
        &mut result,
        chain_slopes(weights, bias, bounds, &order)?,
        limit,
    );
    // Coordinate rotations prioritize different marginal crossing variables.
    for offset in 0..n.min(limit.saturating_mul(2)) {
        if result.len() >= limit {
            break;
        }
        order = (0..n).map(|i| (i + offset) % n).collect();
        append_unique(
            &mut result,
            chain_slopes(weights, bias, bounds, &order)?,
            limit,
        );
        order.reverse();
        append_unique(
            &mut result,
            chain_slopes(weights, bias, bounds, &order)?,
            limit,
        );
    }
    Some(result)
}

/// Numerical separation proposal at any vector (even an infeasible one).
/// Sorting normalized box coordinates supplies the most relevant chain.
/// The point is never interpreted as a source-network or SAT witness.
pub(super) fn separating_slopes(
    weights: &[f64],
    bias: f64,
    bounds: &[(f64, f64)],
    point: &[f64],
) -> Option<Vec<f64>> {
    if !valid(weights, bias, bounds)
        || point.len() != weights.len()
        || !point.iter().copied().all(admissible)
    {
        return None;
    }
    let normalized = |i: usize| {
        let (l, u) = bounds[i];
        if l == u || weights[i] == 0.0 {
            0.0
        } else if weights[i] > 0.0 {
            (point[i] - l) / (u - l)
        } else {
            (u - point[i]) / (u - l)
        }
    };
    let mut order: Vec<_> = (0..weights.len()).collect();
    order.sort_by(|&i, &j| normalized(j).total_cmp(&normalized(i)).then(i.cmp(&j)));
    chain_slopes(weights, bias, bounds, &order)
}

pub(super) fn cuts(
    weights: &[f64],
    bias: f64,
    bounds: &[(f64, f64)],
    limit: usize,
    deadline: Option<Instant>,
) -> Option<Vec<AffineReluCut>> {
    let mut result = Vec::new();
    for slopes in candidate_slopes(weights, bias, bounds, limit)? {
        let intercept = certified_affine_relu_cut(weights, bias, bounds, &slopes, deadline)?;
        result.push(AffineReluCut { slopes, intercept });
    }
    alive(deadline).then_some(result)
}

#[derive(Clone)]
pub(super) struct ComposedAffine {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) weights: Vec<f64>,
    pub(super) bias: Vec<f64>,
    pub(super) weight_lo: Vec<f64>,
    pub(super) weight_hi: Vec<f64>,
    pub(super) bias_lo: Vec<f64>,
    pub(super) bias_hi: Vec<f64>,
}

/// Compose a consecutive affine run ONLY as a numerical proposal. Alongside
/// every rounded coefficient, propagate an enclosure of the exact authored
/// composition. The original source LP keeps all authored equations intact.
pub(super) fn compose(
    chain: &PairedChain,
    start: usize,
    end: usize,
    width: usize,
    g: bool,
    deadline: Option<Instant>,
) -> Option<ComposedAffine> {
    if width == 0 || width > MAX_WIDTH || start >= end || end > chain.stages.len() {
        return None;
    }
    let mut previous = ComposedAffine {
        rows: width,
        cols: width,
        weights: vec![0.0; width.checked_mul(width)?],
        bias: vec![0.0; width],
        weight_lo: vec![0.0; width * width],
        weight_hi: vec![0.0; width * width],
        bias_lo: vec![0.0; width],
        bias_hi: vec![0.0; width],
    };
    for j in 0..width {
        previous.weights[j * width + j] = 1.0;
        previous.weight_lo[j * width + j] = 1.0;
        previous.weight_hi[j * width + j] = 1.0;
    }
    for stage in &chain.stages[start..end] {
        if !alive(deadline) {
            return None;
        }
        let Stage::Affine {
            rows,
            cols,
            wf,
            wg,
            bf,
            bg,
        } = stage
        else {
            return None;
        };
        if *cols != previous.rows || *rows == 0 || *rows > MAX_WIDTH {
            return None;
        }
        let (weights, bias) = if g { (wg, bg) } else { (wf, bf) };
        if weights.len() != rows.checked_mul(*cols)? || bias.len() != *rows {
            return None;
        }
        let count = rows.checked_mul(width)?;
        let mut next = ComposedAffine {
            rows: *rows,
            cols: width,
            weights: vec![0.0; count],
            bias: bias.clone(),
            weight_lo: vec![0.0; count],
            weight_hi: vec![0.0; count],
            bias_lo: bias.clone(),
            bias_hi: bias.clone(),
        };
        for i in 0..*rows {
            if !alive(deadline) {
                return None;
            }
            for h in 0..*cols {
                let w = weights[i * cols + h];
                if !admissible(w) {
                    return None;
                }
                if w == 0.0 {
                    continue;
                }
                next.bias[i] += w * previous.bias[h];
                let (bl, bu) = product_interval(previous.bias_lo[h], previous.bias_hi[h], w)?;
                next.bias_lo[i] = add_down(next.bias_lo[i], bl);
                next.bias_hi[i] = add_up(next.bias_hi[i], bu);
                for j in 0..width {
                    let k = i * width + j;
                    let p = h * width + j;
                    next.weights[k] += w * previous.weights[p];
                    let (wl, wu) =
                        product_interval(previous.weight_lo[p], previous.weight_hi[p], w)?;
                    next.weight_lo[k] = add_down(next.weight_lo[k], wl);
                    next.weight_hi[k] = add_up(next.weight_hi[k], wu);
                }
            }
        }
        if next
            .weights
            .iter()
            .chain(&next.bias)
            .chain(&next.weight_lo)
            .chain(&next.weight_hi)
            .chain(&next.bias_lo)
            .chain(&next.bias_hi)
            .any(|&v| !admissible(v))
        {
            return None;
        }
        previous = next;
    }
    Some(previous)
}

/// Absolute source-box bound on authored affine value minus its rounded
/// numerical composition. Coefficient errors are multiplied by the full
/// predecessor source boxes, including sign and cancellation.
pub(super) fn composition_error(
    affine: &ComposedAffine,
    row: usize,
    bounds: &[(f64, f64)],
    deadline: Option<Instant>,
) -> Option<f64> {
    if row >= affine.rows || bounds.len() != affine.cols {
        return None;
    }
    let mut lower = add_down(affine.bias_lo[row], -affine.bias[row]);
    let mut upper = add_up(affine.bias_hi[row], -affine.bias[row]);
    for (j, &(l, u)) in bounds.iter().enumerate() {
        if !alive(deadline) {
            return None;
        }
        let k = row * affine.cols + j;
        let cl = add_down(affine.weight_lo[k], -affine.weights[k]);
        let cu = add_up(affine.weight_hi[k], -affine.weights[k]);
        lower = add_down(lower, product_lower(cl, cu, l, u)?);
        // max interval product = -min product of its negation.
        upper = add_up(upper, -product_lower(-cu, -cl, l, u)?);
    }
    let error = lower.abs().max(upper.abs());
    (lower <= upper && admissible(error) && alive(deadline)).then_some(error)
}

/// Append source-owned cuts at an authored ReLU stage, peeling ALL consecutive
/// preceding affine operators back to the last ReLU/input state. Bindings are
/// private source-builder state maps: state 0 is the shared root, state s is
/// the input of stage s, and state s+1 is its output. No variables are added.
///
/// ReLU is 1-Lipschitz, so an exact composition-error enclosure repairs every
/// proposed composed-affine cut by subtracting that error from its RHS.
pub(super) fn rows(
    chain: &PairedChain,
    states: &[Enclosure],
    stage: usize,
    bindings: &[(Vec<usize>, Vec<usize>)],
    deadline: Option<Instant>,
) -> Option<Vec<LinearRow>> {
    const CUTS_PER_NEURON: usize = 8;
    let Stage::Relu { width } = chain.stages.get(stage)? else {
        return None;
    };
    if states.len() <= stage + 1 || bindings.len() <= stage + 1 || !alive(deadline) {
        return None;
    }
    let mut start = stage;
    while start > 0 && matches!(chain.stages[start - 1], Stage::Affine { .. }) {
        start -= 1;
    }
    if start == stage {
        // Consecutive ReLUs gain nothing from this affine-input family.
        return Some(Vec::new());
    }
    let mut result = Vec::new();
    for g in [false, true] {
        let source = &states[start];
        let pre = &states[stage];
        let (lo, hi, pre_lo, pre_hi, inputs, outputs) = if g {
            (
                &source.g_lo,
                &source.g_hi,
                &pre.g_lo,
                &pre.g_hi,
                &bindings[start].1,
                &bindings[stage + 1].1,
            )
        } else {
            (
                &source.f_lo,
                &source.f_hi,
                &pre.f_lo,
                &pre.f_hi,
                &bindings[start].0,
                &bindings[stage + 1].0,
            )
        };
        if lo.len() != hi.len()
            || lo.len() != inputs.len()
            || outputs.len() != *width
            || pre_lo.len() != *width
            || pre_hi.len() != *width
        {
            return None;
        }
        let bounds: Vec<_> = lo.iter().copied().zip(hi.iter().copied()).collect();
        let affine = compose(chain, start, stage, lo.len(), g, deadline)?;
        if affine.rows != *width {
            return None;
        }
        for i in 0..*width {
            if !alive(deadline) {
                return None;
            }
            if pre_lo[i] >= 0.0 || pre_hi[i] <= 0.0 {
                continue;
            }
            let weights = &affine.weights[i * affine.cols..(i + 1) * affine.cols];
            let error = composition_error(&affine, i, &bounds, deadline)?;
            for cut in cuts(weights, affine.bias[i], &bounds, CUTS_PER_NEURON, deadline)? {
                let rhs = add_down(cut.intercept, -error);
                if !admissible(rhs) {
                    return None;
                }
                let mut terms: Vec<_> = inputs
                    .iter()
                    .copied()
                    .zip(cut.slopes)
                    .filter(|&(_, a)| a != 0.0)
                    .collect();
                terms.push((outputs[i], -1.0));
                result.push(LinearRow { terms, rhs });
            }
        }
    }
    alive(deadline).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn multivariate_facet_excludes_scalar_triangle_false_point() {
        let w = [1.0, 1.0];
        let bounds = [(0.0, 1.0), (0.0, 1.0)];
        let a = separating_slopes(&w, -1.0, &bounds, &[1.0, 0.0]).unwrap();
        assert_eq!(a, [0.0, 1.0]);
        let k = certified_affine_relu_cut(&w, -1.0, &bounds, &a, None).unwrap();
        // At preactivation 0 the scalar [-1,1] triangle allows r=1/2.
        // This new source-box facet instead certifies r <= x_2 + tiny repair.
        assert!(k > -1e-14);
        assert!(a[0] * 1.0 + a[1] * 0.0 - 0.5 < k);
    }

    #[test]
    fn arbitrary_signed_slopes_enclose_exact_dyadic_relu_graph() {
        let w = [1.5, -0.75, 0.5];
        let bounds = [(-1.0, 1.0), (-0.5, 1.5), (0.0, 1.0)];
        for a in [[-0.25, 2.0, 0.125], [2.0, -1.5, 0.25], [0.0, 0.0, 0.0]] {
            let k = certified_affine_relu_cut(&w, -0.125, &bounds, &a, None).unwrap();
            for i in 0..=16 {
                for j in 0..=16 {
                    for q in 0..=8 {
                        let x = [-1.0 + i as f64 / 8.0, -0.5 + j as f64 / 8.0, q as f64 / 8.0];
                        let z = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] - 0.125;
                        let lhs = a[0] * x[0] + a[1] * x[1] + a[2] * x[2] - z.max(0.0);
                        assert!(lhs >= k, "{a:?}, {x:?}: {lhs} < {k}");
                    }
                }
            }
        }
    }

    #[test]
    fn arbitrary_planes_retain_exact_coefficient_cancellation_error() {
        let w = [1.0 + 2.0f64.powi(-52)];
        let a = [1.0];
        let k = certified_affine_relu_cut(&w, 0.0, &[(1.0, 1.0)], &a, None).unwrap();
        assert!(k <= -2.0f64.powi(-52));
        assert!(k > -1e-14);
    }

    #[test]
    fn all_source_portfolio_cuts_enclose_both_closed_phases() {
        let w = [-1.0, 2.0, -0.5];
        let bounds = [(-1.0, 1.0), (0.0, 1.0), (-0.5, 0.5)];
        let portfolio = cuts(&w, -0.25, &bounds, 16, None).unwrap();
        assert!(portfolio.len() >= 2);
        for cut in &portfolio {
            for i in 0..=8 {
                for j in 0..=4 {
                    for q in 0..=4 {
                        let x = [-1.0 + i as f64 / 4.0, j as f64 / 4.0, -0.5 + q as f64 / 4.0];
                        let z = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] - 0.25;
                        let lhs =
                            cut.slopes[0] * x[0] + cut.slopes[1] * x[1] + cut.slopes[2] * x[2]
                                - z.max(0.0);
                        assert!(lhs >= cut.intercept);
                    }
                }
            }
        }
        assert_eq!(
            candidate_slopes(&w, -0.25, &bounds, 16),
            candidate_slopes(&w, -0.25, &bounds, 16)
        );
    }

    #[test]
    fn source_and_slope_tampering_is_repaired_or_rejected() {
        let w = [1.0, 1.0];
        let bounds = [(0.0, 1.0), (0.0, 1.0)];
        let a = [0.0, 1.0];
        let k = certified_affine_relu_cut(&w, -1.0, &bounds, &a, None).unwrap();
        let changed = certified_affine_relu_cut(&w, 0.0, &bounds, &a, None).unwrap();
        assert!(changed < k - 0.9);
        assert!(certified_affine_relu_cut(&w, -1.0, &bounds, &[f64::NAN, 0.0], None).is_none());
        assert!(certified_affine_relu_cut(&w, -1.0, &bounds[..1], &a, None).is_none());
        assert!(candidate_slopes(&w, -1.0, &bounds, MAX_PORTFOLIO + 1).is_none());
        let invalid = [(1.0, 0.0), (0.0, 1.0)];
        assert!(certified_affine_relu_cut(&w, -1.0, &invalid, &a, None).is_none());
    }

    #[test]
    fn expired_deadline_never_returns_support_authority() {
        let past = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        assert!(
            certified_affine_relu_cut(&[1.0], 0.0, &[(-1.0, 1.0)], &[0.5], Some(past)).is_none()
        );
        assert!(cuts(&[1.0], 0.0, &[(-1.0, 1.0)], 2, Some(past)).is_none());
    }

    #[test]
    fn source_rows_peel_separate_matmul_and_add_without_new_variables() {
        let chain = PairedChain {
            input_dim: 2,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, 1.0],
                    wg: vec![1.0, 1.0],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![1.0],
                    bf: vec![-1.0],
                    bg: vec![-1.0],
                },
                Stage::Relu { width: 1 },
            ],
        };
        let domain = super::super::BoxDomain {
            lo: vec![0.0, 0.0],
            hi: vec![1.0, 1.0],
        };
        let states = super::super::forward::states(&chain, &domain, None, true).unwrap();
        let bindings = vec![
            (vec![0, 1], vec![0, 1]),
            (vec![2], vec![3]),
            (vec![4], vec![5]),
            (vec![6], vec![7]),
        ];
        let source_rows = rows(&chain, &states, 2, &bindings, None).unwrap();
        assert!(!source_rows.is_empty());
        assert!(source_rows
            .iter()
            .all(|r| r.terms.iter().all(|&(i, _)| [0, 1, 6, 7].contains(&i))));
        let inflated = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        assert!(source_rows
            .iter()
            .any(|r| { r.terms.iter().map(|&(j, a)| a * inflated[j]).sum::<f64>() < r.rhs }));
        for i in 0..=8 {
            for j in 0..=8 {
                let x = [i as f64 / 8.0, j as f64 / 8.0];
                let y = (x[0] + x[1] - 1.0).max(0.0);
                let point = [
                    x[0],
                    x[1],
                    x[0] + x[1],
                    x[0] + x[1],
                    x[0] + x[1] - 1.0,
                    x[0] + x[1] - 1.0,
                    y,
                    y,
                ];
                for row in &source_rows {
                    let lhs = row.terms.iter().map(|&(j, a)| a * point[j]).sum::<f64>();
                    assert!(lhs >= row.rhs);
                }
            }
        }
    }

    #[test]
    fn rounded_affine_composition_has_an_explicit_source_error_charge() {
        let factor = 1.0 + 2.0f64.powi(-27);
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![factor],
                    wg: vec![factor],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![factor],
                    wg: vec![factor],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
            ],
        };
        let affine = compose(&chain, 0, 2, 1, false, None).unwrap();
        // Exact factor^2 includes +2^-54, which this stored proposal loses.
        assert_eq!(affine.weights[0], 1.0 + 2.0f64.powi(-26));
        let error = composition_error(&affine, 0, &[(1.0, 1.0)], None).unwrap();
        assert!(error >= 2.0f64.powi(-54));
        assert!(error < 1e-12);
    }
}
