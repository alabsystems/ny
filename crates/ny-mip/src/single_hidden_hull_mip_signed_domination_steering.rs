// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded geometry advice only. Every final row is freshly checked from Source.
use super::*;
#[derive(Clone, Copy)]
struct Feature {
    head: usize,
    positive: bool,
    shape: [f64; 2],
    scale: f64,
    priority: f64,
}
fn center(l: f64, u: f64) -> Option<f64> {
    finite(0.5 * l + 0.5 * u)
}
fn feature(p: &HullMarginProblem, head: usize) -> Option<Feature> {
    let mut bias = p.b1[head];
    let (mut radius, mut shape) = (0., [0.; 2]);
    for i in 0..p.input_dim {
        let w = p.w_row(head)[i];
        let r = finite(0.5 * p.upper[i] - 0.5 * p.lower[i])?;
        bias = finite(bias + w * center(p.lower[i], p.upper[i])?)?;
        let x = finite(w * r)?;
        radius = finite(radius + x.abs())?;
        shape[i & 1] = finite(shape[i & 1] + x)?;
    }
    let scale = finite(radius + bias.abs())?;
    if scale <= 0. {
        return None;
    }
    shape[1] = finite(shape[1] + bias)?;
    for v in &mut shape {
        *v = finite(*v / scale)?;
    }
    let positive = p.c[head] > 0.;
    let unstable = bias - radius < 0. && bias + radius > 0.;
    Some(Feature {
        head,
        positive,
        shape,
        scale,
        priority: if !positive && unstable {
            finite(-p.c[head] * scale)?.max(0.)
        } else {
            0.
        },
    })
}
fn geometry(
    p: &HullMarginProblem,
    count: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(Vec<Feature>, Vec<usize>)> {
    let mut all = Vec::with_capacity(p.hidden_dim);
    for head in 0..p.hidden_dim {
        if !live(end) || !poll() {
            return None;
        }
        if p.c[head] != 0. {
            if let Some(f) = feature(p, head) {
                all.push(f);
            }
        }
    }
    // Only bounded indices move; no H*D pair scan or N*P buffer is created.
    let mut targets: Vec<usize> = Vec::with_capacity(count);
    for (index, f) in all.iter().enumerate() {
        if f.positive || f.priority <= 0. {
            continue;
        }
        let slot = targets
            .iter()
            .position(|&j| f.priority > all[j].priority)
            .unwrap_or(targets.len());
        if slot < count {
            targets.insert(slot, index);
            if targets.len() > count {
                targets.pop();
            }
        }
    }
    Some((all, targets))
}
fn pool(all: &[Feature], target: Feature) -> Vec<usize> {
    let mut best: Vec<(f64, usize)> = Vec::with_capacity(8);
    for f in all.iter().filter(|f| f.positive) {
        let distance = (f.shape[0] - target.shape[0]).abs() + (f.shape[1] - target.shape[1]).abs();
        if best.len() < 8 {
            best.push((distance, f.head));
            continue;
        }
        let worst = (0..best.len())
            .max_by(|&a, &b| best[a].0.total_cmp(&best[b].0))
            .unwrap();
        if distance < best[worst].0 {
            best[worst] = (distance, f.head);
        }
    }
    best.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    best.into_iter().map(|(_, j)| j).collect()
}
fn ratio(p: &HullMarginProblem, a: &kernel::Proposal, slot: usize) -> Option<f64> {
    let j = *a.positive.get(slot)?;
    let (mut numerator, mut denominator) = (0., 0.);
    let mut rb = p.b1[a.head];
    let mut pb = p.b1[j];
    for (&k, &lambda) in a.positive.iter().zip(&a.lambda) {
        if k != j {
            rb = finite(rb - lambda * p.b1[k])?;
        }
    }
    for i in 0..p.input_dim {
        let mut r = p.w_row(a.head)[i];
        for (&k, &lambda) in a.positive.iter().zip(&a.lambda) {
            if k != j {
                r = finite(r - lambda * p.w_row(k)[i])?;
            }
        }
        let c = center(p.lower[i], p.upper[i])?;
        rb = finite(rb + r * c)?;
        pb = finite(pb + p.w_row(j)[i] * c)?;
        let width = finite(0.5 * p.upper[i] - 0.5 * p.lower[i])?;
        let (r, v) = (finite(r * width)?, finite(p.w_row(j)[i] * width)?);
        numerator = finite(numerator + r * v)?;
        denominator = finite(denominator + v * v)?;
    }
    numerator = finite(numerator + rb * pb)?;
    denominator = finite(denominator + pb * pb)?;
    if denominator <= 0. {
        return Some(0.);
    }
    finite(numerator / denominator).map(|x| x.clamp(0., 256.))
}
/// This point residual only chooses a direction and fitting score. It grants
/// neither upper-bound nor original graph authority, even after a good fit.
fn nominal(p: &HullMarginProblem, a: &kernel::Proposal) -> Option<(f64, Vec<f64>)> {
    let mut r = p.w_row(a.head).to_vec();
    let mut bias = p.b1[a.head];
    for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
        bias = finite(bias - lambda * p.b1[j])?;
        for (i, v) in r.iter_mut().enumerate() {
            *v = finite(*v - lambda * p.w_row(j)[i])?;
        }
    }
    let (mut c, mut width) = (bias, 0.);
    for (i, &v) in r.iter().enumerate() {
        c = finite(c + v * center(p.lower[i], p.upper[i])?)?;
        width = finite(width + v.abs() * finite(0.5 * p.upper[i] - 0.5 * p.lower[i])?)?;
    }
    let lo = finite(c - width)?;
    let hi = finite(c + width)?;
    let rho = if lo >= 0. {
        1.
    } else if hi <= 0. {
        0.
    } else {
        finite(hi / finite(2. * width)?)?.clamp(0., 1.)
    };
    let v = r
        .into_iter()
        .map(|v| finite(rho * v))
        .collect::<Option<Vec<_>>>()?;
    Some((finite(c.abs() + width)?, v))
}
fn score(
    row: &SignedMajorant,
    p: &HullMarginProblem,
    mut a: kernel::Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, kernel::Proposal)> {
    if !live(end) || !poll() {
        return None;
    }
    let (value, v) = nominal(p, &a)?;
    if !live(end) || !poll() {
        return None;
    }
    a.v = v;
    row.budget
        .state
        .lock()
        .ok()?
        .component_rows
        .domination
        .nominal_scores += 1;
    Some((value, a))
}
fn pair(
    row: &SignedMajorant,
    p: &HullMarginProblem,
    target: Feature,
    positive: &[usize],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, kernel::Proposal)> {
    let mut best = None;
    for &j in positive.iter().take(2) {
        let mut a = kernel::Proposal {
            head: target.head,
            positive: vec![j],
            lambda: vec![1.],
            v: vec![],
        };
        let fitted = ratio(p, &a, 0).unwrap_or(1.);
        for lambda in [1., fitted] {
            a.lambda[0] = lambda;
            let Some(candidate) = score(row, p, a.clone(), end, poll) else {
                continue;
            };
            if lambda > 0.
                && best
                    .as_ref()
                    .is_none_or(|b: &(f64, kernel::Proposal)| candidate.0 < b.0)
            {
                best = Some(candidate);
            }
        }
    }
    best
}
fn sparse(
    row: &SignedMajorant,
    p: &HullMarginProblem,
    mut best: (f64, kernel::Proposal),
    pool: &[usize],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> (f64, kernel::Proposal) {
    for &j in pool {
        if best.1.positive.len() == 4 {
            break;
        }
        if !best.1.positive.contains(&j) {
            best.1.positive.push(j);
            best.1.lambda.push(0.);
        }
    }
    for slot in 0..best.1.positive.len() {
        let fitted = ratio(p, &best.1, slot).unwrap_or(best.1.lambda[slot]);
        for lambda in [best.1.lambda[slot], 0., 1., fitted] {
            let mut a = best.1.clone();
            a.lambda[slot] = lambda;
            if let Some(candidate) = score(row, p, a, end, poll) {
                if candidate.0 < best.0 {
                    best = candidate;
                }
            }
        }
    }
    best
}
pub(super) fn proposals(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    t: &resources::Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<kernel::Proposal>> {
    resources::begin_advice(row, t)?;
    let p = cuts.problem;
    let (all, targets) = geometry(p, t.targets, t.end, poll)?;
    let mut out = Vec::with_capacity(targets.len());
    let mut fitted = 0;
    for index in targets {
        if !live(t.end) || !poll() {
            return None;
        }
        let target = all[index];
        let pool = pool(&all, target);
        let Some(mut best) = pair(row, p, target, &pool, t.end, poll) else {
            continue;
        };
        if fitted < t.sparse && pool.len() > 1 {
            fitted += 1;
            best = sparse(row, p, best, &pool, t.end, poll);
            row.budget
                .state
                .lock()
                .ok()?
                .component_rows
                .domination
                .sparse_fits += 1;
        }
        if best.0 < target.scale && best.1.lambda.iter().any(|v| *v > 0.) {
            out.push(best.1);
        }
    }
    Some(out)
}
