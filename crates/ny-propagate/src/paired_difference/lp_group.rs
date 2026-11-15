// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded source-certified two-neuron ReLU group supports.
//!
//! A source prefix backward pass certifies L <= z_i - t*z_j <= U over
//! the WHOLE original input box, using only states before this ReLU. The
//! prepared paired geometry then covers every one of the four activation
//! phases of (z_i,t*z_j), including zero boundaries, and certifies
//!
//! cf*relu(z_i) + cg*relu(t*z_j) >= af*z_i + ag*t*z_j + k.
//!
//! For positive t, relu(t*z_j)=t*relu(z_j). For negative t, it equals
//! (-t)*(relu(z_j)-z_j). Both substitutions preserve the exact graph.
//! Every rounded scaled row coefficient is repaired against rigorous source
//! boxes. In particular, negative t keeps separate duplicate pre-column
//! terms instead of treating a rounded sum as an authored equality.
//!
//! Source affine features and objective weights choose a small portfolio;
//! they are numerical advice with no proof authority. Source correlations,
//! geometric intercept replay, and coefficient repair certify every row.
//! No conditional phase row or proposed LP point enters the root model.
//! This is a group relaxation in the established two-neuron sense and makes
//! no priority or competition-score claim.

use super::lp_certificate::{suitable_fp_environment, LinearRow};
use super::*;

const LAST_RELU_STAGES: usize = 2;
const MAX_TWIN_PAIRS: usize = 8;
const MAX_GROUP_PAIRS: usize = 8;
const MAX_PLANES_PER_PAIR: usize = 16;
const MAX_ROWS_PER_STAGE: usize = (MAX_TWIN_PAIRS + MAX_GROUP_PAIRS) * MAX_PLANES_PER_PAIR;
const MAX_FEATURE_OPS: usize = 4_000_000;
const MAX_MAGNITUDE: f64 = 1e100;

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Endpoint {
    g: bool,
    neuron: usize,
}

impl Endpoint {
    fn id(self, width: usize) -> usize {
        usize::from(self.g) * width + self.neuron
    }
}

#[derive(Clone)]
struct Feature {
    // max(abs(unit)) == 1; log_scale retains the removed numerical scale.
    unit: Vec<f64>,
    log_scale: f64,
}

#[derive(Clone, Copy)]
struct Pair {
    left: Endpoint,
    right: Endpoint,
    scale: f64,
}

fn range(source: &Enclosure, e: Endpoint) -> (f64, f64) {
    if e.g {
        (source.g_lo[e.neuron], source.g_hi[e.neuron])
    } else {
        (source.f_lo[e.neuron], source.f_hi[e.neuron])
    }
}

fn valid_interval(l: f64, u: f64) -> bool {
    l.is_finite() && u.is_finite() && l <= u && magnitude(l, u) <= MAX_MAGNITUDE
}

// Numerical feature composition stops at the previous ReLU, so a bias-only
// identity affine immediately before activation cannot hide real correlation.
// Renormalization avoids overflowing long contiguous affine source blocks.
fn affine_features(
    chain: &PairedChain,
    before: usize,
    g: bool,
    deadline: Option<Instant>,
) -> Option<Vec<Feature>> {
    let Stage::Affine { rows, .. } = chain.stages.get(before.checked_sub(1)?)? else {
        return None;
    };
    let mut features = Vec::with_capacity(*rows);
    let mut operations = 0usize;
    for neuron in 0..*rows {
        if !alive(deadline) {
            return None;
        }
        let mut unit = vec![0.0; *rows];
        unit[neuron] = 1.0;
        let mut log_scale = 0.0;
        for s in (0..before).rev() {
            let Stage::Affine {
                rows, cols, wf, wg, ..
            } = &chain.stages[s]
            else {
                break;
            };
            if unit.len() != *rows {
                return None;
            }
            operations = operations.checked_add(rows.checked_mul(*cols)?)?;
            if operations > MAX_FEATURE_OPS || !alive(deadline) {
                return None;
            }
            let weights = if g { wg } else { wf };
            let mut next = vec![0.0; *cols];
            for i in 0..*rows {
                if i & 0x1f == 0 && !alive(deadline) {
                    return None;
                }
                for j in 0..*cols {
                    next[j] += unit[i] * weights[i * cols + j];
                }
            }
            let scale = next.iter().map(|v| v.abs()).fold(0.0, f64::max);
            if !scale.is_finite() {
                return None;
            }
            if scale == 0.0 {
                unit = next;
            } else {
                for value in &mut next {
                    *value /= scale;
                }
                log_scale += scale.log2();
                unit = next;
            }
        }
        if !log_scale.is_finite() || unit.iter().any(|v| !v.is_finite()) {
            return None;
        }
        features.push(Feature { unit, log_scale });
    }
    Some(features)
}

fn correlation(a: &Feature, b: &Feature) -> Option<(f64, f64)> {
    if a.unit.len() != b.unit.len() {
        return None;
    }
    let mut aa = 0.0;
    let mut bb = 0.0;
    let mut ab = 0.0;
    for (&x, &y) in a.unit.iter().zip(&b.unit) {
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    if aa <= 0.0 || bb <= 0.0 || !ab.is_finite() {
        return None;
    }
    let cosine = ab / (aa * bb).sqrt();
    if !cosine.is_finite() || cosine.abs() < 0.75 {
        return None;
    }
    let exponent = (a.log_scale - b.log_scale + 0.5 * (aa / bb).log2())
        .round()
        .clamp(-8.0, 8.0) as i32;
    let scale = 2.0_f64.powi(exponent).copysign(cosine);
    Some((cosine.abs().min(1.0), scale))
}

fn outgoing_features(
    chain: &PairedChain,
    stage: usize,
    g: bool,
    deadline: Option<Instant>,
) -> Option<Vec<Feature>> {
    let mut end = stage.checked_add(1)?;
    while matches!(chain.stages.get(end), Some(Stage::Affine { .. })) {
        end += 1;
    }
    (end > stage + 1).then_some(())?;
    affine_features(chain, end, g, deadline)
}

fn gap(l: f64, u: f64) -> f64 {
    if l < 0.0 && u > 0.0 {
        u * (-l / (u - l))
    } else {
        0.0
    }
}

fn priority(source: &Enclosure, e: Endpoint, outgoing: Option<&[Feature]>) -> f64 {
    let (l, u) = range(source, e);
    let weight = outgoing
        .map(|rows| {
            rows.iter()
                .filter_map(|r| {
                    r.unit
                        .get(e.neuron)
                        .map(|v| v.abs() * 2.0_f64.powf(r.log_scale.clamp(-256.0, 256.0)))
                })
                .fold(0.0, f64::max)
        })
        .unwrap_or(1.0);
    let score = gap(l, u) * weight;
    if score.is_finite() {
        score
    } else {
        gap(l, u)
    }
}

fn selected_pairs(
    chain: &PairedChain,
    stage: usize,
    source: &Enclosure,
    outgoing_f: Option<&[Feature]>,
    outgoing_g: Option<&[Feature]>,
    deadline: Option<Instant>,
) -> Option<(Vec<Pair>, Vec<Pair>)> {
    let width = source.f_lo.len();
    let mut ranking: Vec<(usize, f64)> = (0..width)
        .filter_map(|j| {
            let f = Endpoint {
                g: false,
                neuron: j,
            };
            let g = Endpoint { g: true, neuron: j };
            let score = priority(source, f, outgoing_f).max(priority(source, g, outgoing_g));
            (gap(source.f_lo[j], source.f_hi[j]) > 0.0 || gap(source.g_lo[j], source.g_hi[j]) > 0.0)
                .then_some((j, score))
        })
        .collect();
    ranking.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let twins: Vec<_> = ranking
        .iter()
        .take(MAX_TWIN_PAIRS)
        .map(|&(j, _)| Pair {
            left: Endpoint {
                g: false,
                neuron: j,
            },
            right: Endpoint { g: true, neuron: j },
            scale: 1.0,
        })
        .collect();

    let ff = affine_features(chain, stage, false, deadline);
    let gf = affine_features(chain, stage, true, deadline);
    if !alive(deadline) {
        return None;
    }
    let (Some(ff), Some(gf)) = (ff, gf) else {
        return Some((twins, Vec::new()));
    };
    if ff.len() != width || gf.len() != width {
        return None;
    }
    // Two objective-leading anchors plus two source-order extremes give
    // bounded diversity; neither tower monopolizes the group budget.
    let mut anchors: Vec<_> = ranking.iter().take(2).map(|&(j, _)| j).collect();
    if let Some(first) = ranking.iter().map(|&(j, _)| j).min() {
        if !anchors.contains(&first) {
            anchors.push(first);
        }
    }
    if let Some(last) = ranking.iter().map(|&(j, _)| j).max() {
        if !anchors.contains(&last) {
            anchors.push(last);
        }
    }
    let mut groups = Vec::new();
    for j in anchors {
        for is_g in [false, true] {
            if !alive(deadline) {
                return None;
            }
            let left = Endpoint { g: is_g, neuron: j };
            let features = if is_g { &gf } else { &ff };
            let mut best: Option<(Endpoint, f64, f64)> = None;
            // Same-tower groups add information absent from twin-only hulls.
            for k in 0..width {
                if k == j {
                    continue;
                }
                if k & 0x1f == 0 && !alive(deadline) {
                    return None;
                }
                let right = Endpoint { g: is_g, neuron: k };
                let Some((cosine, scale)) = correlation(&features[j], &features[k]) else {
                    continue;
                };
                let outgoing = if is_g { outgoing_g } else { outgoing_f };
                let score = cosine
                    * cosine
                    * (1.0
                        + priority(source, right, outgoing)
                            / (1.0 + priority(source, left, outgoing)));
                if score.is_finite()
                    && best.as_ref().is_none_or(|&(e, _, old)| {
                        score > old || (score == old && right.id(width) < e.id(width))
                    })
                {
                    best = Some((right, scale, score));
                }
            }
            if let Some((right, scale, _)) = best {
                if !groups.iter().any(|p: &Pair| {
                    (p.left == left && p.right == right) || (p.left == right && p.right == left)
                }) {
                    groups.push(Pair { left, right, scale });
                }
            }
            if groups.len() == MAX_GROUP_PAIRS {
                break;
            }
        }
        if groups.len() == MAX_GROUP_PAIRS {
            break;
        }
    }
    Some((twins, groups))
}

fn center_values(
    chain: &PairedChain,
    domain: &BoxDomain,
    before: usize,
    deadline: Option<Instant>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let center: Vec<_> = domain
        .lo
        .iter()
        .zip(&domain.hi)
        .map(|(&l, &u)| 0.5 * l + 0.5 * u)
        .collect();
    let (mut f, mut g) = (center.clone(), center);
    for stage in chain.stages.iter().take(before) {
        if !alive(deadline) {
            return None;
        }
        match stage {
            Stage::Affine {
                rows,
                cols,
                wf,
                wg,
                bf,
                bg,
            } => {
                let mut nf = bf.clone();
                let mut ng = bg.clone();
                for i in 0..*rows {
                    if i & 0x1f == 0 && !alive(deadline) {
                        return None;
                    }
                    for j in 0..*cols {
                        nf[i] += wf[i * cols + j] * f[j];
                        ng[i] += wg[i * cols + j] * g[j];
                    }
                }
                f = nf;
                g = ng;
            }
            Stage::Relu { .. } => {
                for v in f.iter_mut().chain(&mut g) {
                    *v = v.max(0.0);
                }
            }
        }
        if f.iter().chain(&g).any(|v| !v.is_finite()) {
            return None;
        }
    }
    Some((f, g))
}

fn scaled_bounds(l: f64, u: f64, t: f64) -> Option<(f64, f64)> {
    let (l, u) = if t > 0.0 {
        (mul_down(t, l), mul_up(t, u))
    } else {
        (mul_down(t, u), mul_up(t, l))
    };
    valid_interval(l, u).then_some((l, u))
}

fn strip(
    chain: &PairedChain,
    domain: &BoxDomain,
    states: &[Enclosure],
    stage: usize,
    pair: Pair,
    deadline: Option<Instant>,
) -> Option<(f64, f64)> {
    let width = states[stage].f_lo.len();
    let mut cf = vec![0.0; width];
    let mut cg = vec![0.0; width];
    if pair.left.g {
        cg[pair.left.neuron] = 1.0;
    } else {
        cf[pair.left.neuron] = 1.0;
    }
    if pair.right.g {
        cg[pair.right.neuron] -= pair.scale;
    } else {
        cf[pair.right.neuron] -= pair.scale;
    }
    let lower = chain
        .back(
            stage,
            cf.clone(),
            cg.clone(),
            0.0,
            states,
            domain,
            true,
            deadline,
        )?
        .0;
    for v in cf.iter_mut().chain(&mut cg) {
        *v = -*v;
    }
    let upper = -chain
        .back(stage, cf, cg, 0.0, states, domain, true, deadline)?
        .0;
    (alive(deadline) && valid_interval(lower, upper)).then_some((lower, upper))
}

fn normal_portfolio(
    pair: Pair,
    outgoing_f: Option<&[Feature]>,
    outgoing_g: Option<&[Feature]>,
) -> Vec<(f64, f64)> {
    let mut result = vec![
        (1.0, -1.0),
        (-1.0, 1.0),
        (1.0, -0.5),
        (-1.0, 0.5),
        (1.0, -2.0),
        (-1.0, 2.0),
        (-1.0, -1.0),
        (0.5, -1.0),
    ];
    let left = if pair.left.g { outgoing_g } else { outgoing_f };
    let right = if pair.right.g { outgoing_g } else { outgoing_f };
    if let (Some(left), Some(right)) = (left, right) {
        let mut scored = Vec::new();
        for (a, b) in left.iter().zip(right) {
            let Some(&x) = a.unit.get(pair.left.neuron) else {
                continue;
            };
            let Some(&y) = b.unit.get(pair.right.neuron) else {
                continue;
            };
            let xf = x
                * 2.0_f64.powf(a.log_scale.clamp(-256.0, 256.0))
                * if pair.left.g { -1.0 } else { 1.0 };
            let yg = y
                * 2.0_f64.powf(b.log_scale.clamp(-256.0, 256.0))
                * if pair.right.g { -1.0 } else { 1.0 }
                / pair.scale.abs();
            let scale = xf.abs().max(yg.abs());
            if scale > 0.0 && scale.is_finite() {
                let normal = (xf / scale, yg / scale);
                if normal.0.is_finite() && normal.1.is_finite() {
                    scored.push((normal, scale));
                }
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        for ((a, b), _) in scored.into_iter().take(2) {
            for n in [(a, b), (-a, -b)] {
                if !result.contains(&n) {
                    result.push(n);
                }
            }
        }
    }
    result
}

// Repair arbitrary rounded products even though t is currently a bounded
// signed power of two. This explicitly covers subnormal products as well.
fn scaled_coefficient(a: f64, t: f64, bound: f64) -> Option<(f64, f64)> {
    let p = a * t;
    if !p.is_finite() || p.abs() > MAX_MAGNITUDE || !bound.is_finite() {
        return None;
    }
    let charge = if a == 0.0 || t == 1.0 || t == -1.0 || a == 1.0 || a == -1.0 {
        0.0
    } else {
        let error = (p - p.next_down()).max(p.next_up() - p).next_up();
        mul_up(error, bound)
    };
    charge.is_finite().then_some((p, charge))
}

fn scaled_row(
    pair: Pair,
    pre_left: usize,
    pre_right: usize,
    post_left: usize,
    post_right: usize,
    source: &Enclosure,
    cf: f64,
    cg: f64,
    af: f64,
    ag: f64,
    k: f64,
) -> Option<LinearRow> {
    let (l, u) = range(source, pair.right);
    let pre_bound = magnitude(l, u);
    let post_bound = u.max(0.0);
    let (post, cp) = scaled_coefficient(cg, pair.scale.abs(), post_bound)?;
    let (pre, ca) = scaled_coefficient(-ag, pair.scale, pre_bound)?;
    let mut terms = vec![
        (post_left, cf),
        (post_right, post),
        (pre_left, -af),
        (pre_right, pre),
    ];
    let mut charge = add_up(cp, ca);
    if pair.scale < 0.0 {
        let (extra, ce) = scaled_coefficient(cg, pair.scale, pre_bound)?;
        terms.push((pre_right, extra));
        charge = add_up(charge, ce);
    }
    let rhs = if charge == 0.0 {
        k
    } else {
        add_down(k, -charge)
    };
    if !rhs.is_finite()
        || rhs.abs() > MAX_MAGNITUDE
        || terms
            .iter()
            .any(|(_, a)| !a.is_finite() || a.abs() > MAX_MAGNITUDE)
    {
        return None;
    }
    Some(LinearRow { terms, rhs })
}

fn planes(
    pair: Pair,
    ranges: PairedReluRanges,
    source: &Enclosure,
    indices: (usize, usize, usize, usize),
    center: (f64, f64),
    normals: &[(f64, f64)],
    deadline: Option<Instant>,
) -> Option<Vec<LinearRow>> {
    let hull = ranges.prepare()?;
    let mut by_normal = Vec::new();
    for &(cf, cg) in normals {
        if !alive(deadline) {
            return None;
        }
        let mut candidates = Vec::new();
        for (af, ag) in ranges.candidate_slopes(cf, cg) {
            if !alive(deadline) {
                return None;
            }
            let k = hull.certified_intercept(cf, cg, af, ag)?;
            let score = af * center.0 + ag * center.1 + k;
            if !score.is_finite() {
                continue;
            }
            let row = scaled_row(
                pair, indices.0, indices.1, indices.2, indices.3, source, cf, cg, af, ag, k,
            )?;
            candidates.push((row, score));
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        by_normal.push(candidates);
    }
    let mut result = Vec::new();
    let mut remaining = Vec::new();
    // Keep one best support for every normal before filling second choices;
    // incomparable normal magnitudes cannot erase a direction from the set.
    for mut candidates in by_normal {
        if !candidates.is_empty() {
            let (row, _) = candidates.remove(0);
            result.push(row);
        }
        remaining.extend(candidates);
        if result.len() == MAX_PLANES_PER_PAIR {
            break;
        }
    }
    remaining.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (row, _) in remaining {
        if result.len() == MAX_PLANES_PER_PAIR {
            break;
        }
        if !result
            .iter()
            .any(|old| old.terms == row.terms && old.rhs == row.rhs)
        {
            result.push(row);
        }
    }
    Some(result)
}

/// The source LP builder supplies the private source variable bindings.
/// No proposal file may supply a row, state, variable index or source bound.
pub(super) fn rows(
    chain: &PairedChain,
    domain: &BoxDomain,
    states: &[Enclosure],
    stage: usize,
    pre_f: &[usize],
    pre_g: &[usize],
    post_f: &[usize],
    post_g: &[usize],
    deadline: Option<Instant>,
) -> Option<Vec<LinearRow>> {
    if !alive(deadline) || !suitable_fp_environment() {
        return None;
    }
    chain.validate().ok()?;
    chain.input_state(domain)?;
    let Stage::Relu { width } = chain.stages.get(stage)? else {
        return None;
    };
    let source = states.get(stage)?;
    if states.len() != chain.stages.len() + 1
        || [
            pre_f.len(),
            pre_g.len(),
            post_f.len(),
            post_g.len(),
            source.f_lo.len(),
            source.f_hi.len(),
            source.g_lo.len(),
            source.g_hi.len(),
            source.d_lo.len(),
            source.d_hi.len(),
        ]
        .iter()
        .any(|&n| n != *width)
    {
        return None;
    }
    for j in 0..*width {
        if !valid_interval(source.f_lo[j], source.f_hi[j])
            || !valid_interval(source.g_lo[j], source.g_hi[j])
            || !valid_interval(source.d_lo[j], source.d_hi[j])
        {
            return None;
        }
    }
    if chain.stages[stage + 1..]
        .iter()
        .filter(|s| matches!(s, Stage::Relu { .. }))
        .count()
        >= LAST_RELU_STAGES
    {
        return Some(Vec::new());
    }
    let outgoing_f = outgoing_features(chain, stage, false, deadline);
    let outgoing_g = outgoing_features(chain, stage, true, deadline);
    let (twins, groups) = selected_pairs(
        chain,
        stage,
        source,
        outgoing_f.as_deref(),
        outgoing_g.as_deref(),
        deadline,
    )?;
    let center = center_values(chain, domain, stage, deadline);
    if !alive(deadline) {
        return None;
    }
    let index = |e: Endpoint, post: bool| -> usize {
        match (e.g, post) {
            (false, false) => pre_f[e.neuron],
            (true, false) => pre_g[e.neuron],
            (false, true) => post_f[e.neuron],
            (true, true) => post_g[e.neuron],
        }
    };
    let mut result = Vec::new();
    for (is_twin, pair) in twins
        .into_iter()
        .map(|p| (true, p))
        .chain(groups.into_iter().map(|p| (false, p)))
    {
        if !alive(deadline) {
            return None;
        }
        let (lf, uf) = range(source, pair.left);
        let (lg, ug) = range(source, pair.right);
        let (lg, ug) = scaled_bounds(lg, ug, pair.scale)?;
        let (dl, du) = if is_twin {
            (source.d_lo[pair.left.neuron], source.d_hi[pair.left.neuron])
        } else {
            strip(chain, domain, states, stage, pair, deadline)?
        };
        let dl = dl.max(add_down(lf, -ug));
        let du = du.min(add_up(uf, -lg));
        if !valid_interval(dl, du) {
            return None;
        }
        let ranges = PairedReluRanges {
            lf,
            uf,
            lg,
            ug,
            dl,
            du,
        };
        let coordinates = if let Some((f, g)) = &center {
            let value = |e: Endpoint| if e.g { g[e.neuron] } else { f[e.neuron] };
            (value(pair.left), pair.scale * value(pair.right))
        } else {
            (0.5 * lf + 0.5 * uf, 0.5 * lg + 0.5 * ug)
        };
        let normals = normal_portfolio(pair, outgoing_f.as_deref(), outgoing_g.as_deref());
        let more = planes(
            pair,
            ranges,
            source,
            (
                index(pair.left, false),
                index(pair.right, false),
                index(pair.left, true),
                index(pair.right, true),
            ),
            coordinates,
            &normals,
            deadline,
        )?;
        if result.len().checked_add(more.len())? > MAX_ROWS_PER_STAGE {
            return None;
        }
        result.extend(more);
    }
    alive(deadline).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(t: f64) -> PairedChain {
        PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 2,
                    cols: 1,
                    wf: vec![1.0, t],
                    wg: vec![1.0, t],
                    bf: vec![0.0; 2],
                    bg: vec![0.0; 2],
                },
                // Exercise the bias-only identity block seen in admitted graphs.
                Stage::Affine {
                    rows: 2,
                    cols: 2,
                    wf: vec![1.0, 0.0, 0.0, 1.0],
                    wg: vec![1.0, 0.0, 0.0, 1.0],
                    bf: vec![0.0; 2],
                    bg: vec![0.0; 2],
                },
                Stage::Relu { width: 2 },
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, -1.0],
                    wg: vec![1.0, -1.0],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
            ],
        }
    }
    fn domain() -> BoxDomain {
        BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        }
    }
    fn generated(chain: &PairedChain) -> Vec<LinearRow> {
        let d = domain();
        let states = refine::states(chain, &d, None, true).unwrap();
        rows(
            chain,
            &d,
            &states,
            2,
            &[0, 1],
            &[2, 3],
            &[4, 5],
            &[6, 7],
            None,
        )
        .unwrap()
    }
    fn lhs(row: &LinearRow, values: &[f64]) -> f64 {
        row.terms.iter().map(|&(j, a)| a * values[j]).sum()
    }

    #[test]
    fn cross_neuron_rows_remove_a_spurious_single_neuron_triangle_point() {
        let chain = chain(1.0);
        let rows = generated(&chain);
        // Independent triangles at z0=z1=0 permit r0=.5,r1=0.
        // Exact equal source affine rows require their ReLUs to coincide.
        let spurious = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0];
        assert!(rows.iter().any(|r| lhs(r, &spurious) < r.rhs - 0.1));
        assert!(rows
            .iter()
            .any(|r| r.terms.iter().any(|&(j, a)| j == 4 && a != 0.0)
                && r.terms.iter().any(|&(j, a)| j == 5 && a != 0.0)));
    }

    #[test]
    fn positive_and_negative_scaled_group_rows_cover_exact_dyadic_traces() {
        for t in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0] {
            let chain = chain(t);
            let rows = generated(&chain);
            assert!(!rows.is_empty());
            for n in -32..=32 {
                let x = n as f64 / 32.0;
                let y = t * x;
                let values = [x, y, x, y, x.max(0.0), y.max(0.0), x.max(0.0), y.max(0.0)];
                for row in &rows {
                    // This fixture's norms/slopes and grid are dyadic with
                    // sufficiently small integer products for exact f64 dots.
                    assert!(lhs(row, &values) >= row.rhs, "t={t} x={x} row={row:?}");
                }
            }
        }
    }

    #[test]
    fn negative_scale_keeps_exact_duplicate_pre_terms() {
        let chain = chain(-1.0);
        let rows = generated(&chain);
        assert!(rows.iter().any(|r| r
            .terms
            .iter()
            .enumerate()
            .any(|(i, &(j, a))| a != 0.0
                && r.terms[i + 1..].iter().any(|&(k, b)| j == k && b != 0.0))));
        let spurious = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0];
        assert!(rows.iter().any(|r| lhs(r, &spurious) < r.rhs - 0.1));
    }

    #[test]
    fn feature_composition_sees_through_identity_bias_stages() {
        let chain = chain(-2.0);
        let f = affine_features(&chain, 2, false, None).unwrap();
        let (cosine, t) = correlation(&f[0], &f[1]).unwrap();
        assert_eq!(cosine, 1.0);
        assert_eq!(t, -0.5);
    }

    #[test]
    fn source_binding_shapes_nan_and_deadline_fail_closed() {
        let chain = chain(1.0);
        let d = domain();
        let states = refine::states(&chain, &d, None, true).unwrap();
        let call = |s: &[Enclosure], p: &[usize], deadline| {
            rows(&chain, &d, s, 2, p, &[2, 3], &[4, 5], &[6, 7], deadline)
        };
        assert!(call(&states, &[0], None).is_none());
        assert!(call(&states, &[0, 1], Some(Instant::now())).is_none());
        assert!(call(&states[..2], &[0, 1], None).is_none());
        let mut bad = states;
        bad[2].f_lo[0] = f64::NAN;
        assert!(call(&bad, &[0, 1], None).is_none());
    }

    #[test]
    fn stage_row_budget_and_deterministic_portfolio_are_explicit() {
        let chain = chain(1.0);
        let a = generated(&chain);
        let b = generated(&chain);
        assert!(a.len() <= MAX_ROWS_PER_STAGE);
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.terms, b.terms);
            assert_eq!(a.rhs, b.rhs);
        }
    }

    #[test]
    fn subnormal_scaled_coefficient_rounding_is_repaired() {
        let tiny = f64::from_bits(1);
        let (rounded, charge) = scaled_coefficient(tiny, 0.5, 1.0).unwrap();
        assert_eq!(rounded, 0.0);
        assert!(charge >= tiny);
        assert!(scaled_coefficient(f64::MAX, 2.0, 1.0).is_none());
    }
}
