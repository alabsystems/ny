// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Fixed-size advice only. The chosen fresh prefix/full fold grants authority.
use super::*;
pub(super) const FAMILY_SCRATCH: usize = 8192;
const LOCAL_FIXED_WORK: u64 = (1 << 24) + 8192;
const LOCAL_WORK_PER_INPUT: u64 = 1024;
const MAX_LOCAL_INPUT: usize = 65536;
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum FamilyChoice {
    Joint {
        pairs: [[usize; 2]; 2],
        scales: [f32; 2],
    },
    Triplet {
        neurons: [usize; 3],
        weights: [f32; 3],
        multiplier: f32,
    },
}
#[derive(Clone, Copy)]
struct Candidate {
    choice: FamilyChoice,
    first: usize,
    len: usize,
    baseline: f64,
    score: f64,
    preferred: bool,
}
#[derive(Clone, Copy)]
struct Form {
    row: [f64; 4],
    lower: f64,
    upper: f64,
}
const EMPTY_FORM: Form = Form {
    row: [0.; 4],
    lower: 0.,
    upper: 0.,
};
struct Screen {
    forms: [Form; 28],
    candidates: [Option<Candidate>; 7],
    forms_len: usize,
    len: usize,
}
impl Screen {
    fn new() -> Self {
        Self {
            forms: [EMPTY_FORM; 28],
            candidates: [None; 7],
            forms_len: 0,
            len: 0,
        }
    }
    fn add(
        &mut self,
        choice: FamilyChoice,
        rows: &[[f64; 4]],
        baseline: f64,
        preferred: bool,
    ) -> Option<()> {
        if self.len == 7 || self.forms_len.checked_add(rows.len())? > 28 || !baseline.is_finite() {
            return None;
        }
        self.candidates[self.len] = Some(Candidate {
            choice,
            first: self.forms_len,
            len: rows.len(),
            baseline,
            score: 0.,
            preferred,
        });
        self.len += 1;
        for row in rows {
            self.forms[self.forms_len] = Form {
                row: *row,
                ..EMPTY_FORM
            };
            self.forms_len += 1;
        }
        Some(())
    }
}
fn q8<const K: usize>(magnitudes: [f64; K]) -> Option<([f32; K], f32)> {
    if magnitudes.iter().any(|x| !x.is_finite() || *x <= 0.) {
        return None;
    }
    let largest = magnitudes.iter().copied().fold(0f64, f64::max);
    let multiplier = (0.5 * largest) as f32;
    if !ny_core::is_crown_coeff_safe(multiplier) || multiplier <= 0. {
        return None;
    }
    let mut w = [0.; K];
    for i in 0..K {
        let ratio = magnitudes[i] / largest;
        if !(1. / 256. ..=1.).contains(&ratio) {
            return None;
        }
        let n = (256. * ratio).round();
        w[i] = (n / 256.) as f32;
        if !(1. ..=256.).contains(&n)
            || f64::from(w[i]) != n / 256.
            || f64::from(2. * w[i]) != 2. * f64::from(w[i])
        {
            return None;
        }
    }
    Some((w, multiplier))
}
fn add_matching(
    s: &mut Screen,
    slots: [[usize; 2]; 2],
    best: &[(usize, f64, f64); 4],
    abs: &[f64; 4],
    preferred: bool,
) -> Option<()> {
    let mut pairs = [[0; 2]; 2];
    let mut scales = [0.; 2];
    let mut rows = [[0.; 4]; 4];
    let mut baseline = 0.;
    for p in 0..2 {
        let [i, j] = slots[p];
        let (w, lambda) = q8([best[i].2, best[j].2])?;
        pairs[p] = [best[i].0, best[j].0];
        pairs[p].sort_unstable();
        scales[p] = lambda;
        // Each pair's support is converted to the original objective units.
        let wi = f64::from(w[0]) * f64::from(lambda);
        let wj = f64::from(w[1]) * f64::from(lambda);
        rows[2 * p][i] = wi;
        rows[2 * p][j] = wj;
        rows[2 * p + 1][i] = wi;
        rows[2 * p + 1][j] = -wj;
        baseline += wi * abs[i] + wj * abs[j];
    }
    if pairs[1] < pairs[0] {
        pairs.swap(0, 1);
        scales.swap(0, 1);
    }
    s.add(
        FamilyChoice::Joint { pairs, scales },
        &rows,
        baseline,
        preferred,
    )
}
fn add_triplet(
    s: &mut Screen,
    slots: [usize; 3],
    best: &[(usize, f64, f64); 4],
    abs: &[f64; 4],
) -> Option<()> {
    let (w, multiplier) = q8(slots.map(|i| best[i].2))?;
    let mut zipped = std::array::from_fn::<_, 3, _>(|i| (best[slots[i]].0, w[i]));
    zipped.sort_unstable_by_key(|x| x.0);
    let neurons = zipped.map(|x| x.0);
    let weights = zipped.map(|x| x.1);
    let signs = [[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [-1., 1., 1.]];
    let mut rows = [[0.; 4]; 4];
    let mut baseline = 0.;
    for i in 0..3 {
        let scaled = f64::from(w[i]) * f64::from(multiplier);
        baseline += scaled * abs[slots[i]];
        for r in 0..4 {
            rows[r][slots[i]] = signs[r][i] * scaled;
        }
    }
    s.add(
        FamilyChoice::Triplet {
            neurons,
            weights,
            multiplier,
        },
        &rows,
        baseline,
        false,
    )
}
fn check(lease: &WorkLease, end: Instant) -> Option<()> {
    lease.checkpoint().ok()?;
    (Instant::now() < end).then_some(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn family_target(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    target: &str,
    end: Instant,
    allow_joint: bool,
) -> Option<FamilyChoice> {
    let (best, count) = ranked_target(graph, input, bounds, lease, objective, target, end)?;
    // This is ADDITIONAL to the ranking walk, and precedes every new lookup.
    lease.reserve_more_work(LOCAL_FIXED_WORK).ok()?;
    check(lease, end)?;
    if size_of::<Screen>() + 4096 > FAMILY_SCRATCH || target.len() > 1024 {
        return None;
    }
    let relu = graph.node(target)?;
    let pre_name = relu.require_unary_input().ok()?;
    if pre_name.len() > 1024 {
        return None;
    }
    let pre = bounds.get(pre_name)?;
    if pre.shape().len() > 4 {
        return None;
    }
    let lo = pre.lower().as_slice()?;
    let hi = pre.upper().as_slice()?;
    let pre_node = graph.node(pre_name)?;
    // Only original Linear or Linear -> raw per-channel BN is scored. No
    // folded attack matrix, unrelated cache, residual or spatial remapping.
    let (linear, linear_parent, bn) = match pre_node.layer() {
        Layer::Linear(l) => (l, pre_node.require_unary_input().ok()?, None),
        Layer::BatchNorm(bn) => {
            let parent = pre_node.require_unary_input().ok()?;
            if parent.len() > 1024 {
                return None;
            }
            let node = graph.node(parent)?;
            let Layer::Linear(l) = node.layer() else {
                return None;
            };
            let bb = bounds.get(parent)?;
            if !matches!(bb.shape(),[n] if *n==pre.len())
                && !matches!(bb.shape(),[1,n] if *n==pre.len())
            {
                return None;
            }
            if bn.num_channels != pre.len()
                || bn.scale.shape() != [pre.len()]
                || bn.bias.shape() != [pre.len()]
            {
                return None;
            }
            (l, node.require_unary_input().ok()?, Some(bn))
        }
        _ => return None,
    };
    if linear_parent.len() > 1024 || linear.out_features() != pre.len() {
        return None;
    }
    let x = if linear_parent == crate::network::NETWORK_INPUT {
        input
    } else {
        bounds.get(linear_parent)?
    };
    let e = linear.in_features();
    if e == 0 || e > MAX_LOCAL_INPUT || x.len() != e || x.shape().len() > 4 {
        return None;
    }
    let w = linear.weight.as_slice()?;
    let bias = if let Some(v) = linear.bias.as_ref() {
        Some(v.as_slice()?)
    } else {
        None
    };
    if w.len() != linear.out_features().checked_mul(e)?
        || bias.is_some_and(|v| v.len() != pre.len())
    {
        return None;
    }
    let xl = x.lower().as_slice()?;
    let xu = x.upper().as_slice()?;
    if xl.len() != e || xu.len() != e || lo.len() != pre.len() || hi.len() != pre.len() {
        return None;
    }
    // Four selected rows, at most28 forms, two extrema/form: no dense copies.
    lease
        .reserve_more_work(LOCAL_WORK_PER_INPUT.checked_mul(u64::try_from(e).ok()?)?)
        .ok()?;
    check(lease, end)?;
    let mut scales = [1f64; 4];
    let mut centers = [0f64; 4];
    let mut abs = [0f64; 4];
    let bn_arrays = if let Some(bn) = bn {
        Some((bn.scale.as_slice()?, bn.bias.as_slice()?))
    } else {
        None
    };
    for k in 0..count {
        let i = best[k].0;
        if i >= pre.len()
            || !ny_core::is_crown_coeff_safe(lo[i])
            || !ny_core::is_crown_coeff_safe(hi[i])
            || lo[i] > hi[i]
        {
            return None;
        }
        abs[k] = f64::from(lo[i]).abs().max(f64::from(hi[i]).abs());
        let bi = bias.map_or(0., |v| v[i]);
        if !ny_core::is_crown_coeff_safe(bi) {
            return None;
        }
        if let Some((s, b)) = bn_arrays {
            if !ny_core::is_crown_coeff_safe(s[i]) || !ny_core::is_crown_coeff_safe(b[i]) {
                return None;
            }
            scales[k] = f64::from(s[i]);
            centers[k] = scales[k] * f64::from(bi) + f64::from(b[i]);
        } else {
            centers[k] = f64::from(bi);
        }
    }
    let mut screen = Screen::new();
    if count == 4 && allow_joint {
        for (ordinal, pairs) in [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]]
            .into_iter()
            .enumerate()
        {
            // Invalid Q8 ratio drops only that advice, never a proof result.
            let _ = add_matching(&mut screen, pairs, &best, &abs, ordinal == 0);
        }
    }
    if count == 3 {
        let _ = add_triplet(&mut screen, [0, 1, 2], &best, &abs);
    } else {
        for t in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let _ = add_triplet(&mut screen, t, &best, &abs);
        }
    }
    if screen.len == 0 {
        return None;
    }
    for form in &mut screen.forms[..screen.forms_len] {
        let c = (0..count).map(|k| form.row[k] * centers[k]).sum::<f64>();
        form.lower = c;
        form.upper = c;
    }
    for j in 0..e {
        if j % 128 == 0 {
            check(lease, end)?;
        }
        if !ny_core::is_crown_coeff_safe(xl[j])
            || !ny_core::is_crown_coeff_safe(xu[j])
            || xl[j] > xu[j]
        {
            return None;
        }
        let mut a = [0f64; 4];
        for k in 0..count {
            let v = w[best[k].0 * e + j];
            if !ny_core::is_crown_coeff_safe(v) {
                return None;
            }
            a[k] = scales[k] * f64::from(v);
        }
        for form in &mut screen.forms[..screen.forms_len] {
            let v = (0..count).map(|k| form.row[k] * a[k]).sum::<f64>();
            form.lower += v * f64::from(if v >= 0. { xl[j] } else { xu[j] });
            form.upper += v * f64::from(if v >= 0. { xu[j] } else { xl[j] });
        }
    }
    let mut chosen = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_key = None;
    for candidate in screen.candidates[..screen.len].iter_mut().flatten() {
        let f = &screen.forms[candidate.first..candidate.first + candidate.len];
        if f.iter()
            .any(|v| !v.lower.is_finite() || !v.upper.is_finite() || v.lower > v.upper)
        {
            return None;
        }
        let support = match candidate.choice {
            FamilyChoice::Joint { .. } => {
                f[..2]
                    .iter()
                    .map(|v| v.upper.max(-v.lower))
                    .fold(0f64, f64::max)
                    + f[2..]
                        .iter()
                        .map(|v| v.upper.max(-v.lower))
                        .fold(0f64, f64::max)
            }
            FamilyChoice::Triplet { .. } => {
                f.iter().map(|v| v.upper.max(-v.lower)).fold(0f64, f64::max)
            }
        };
        candidate.score = (candidate.baseline - support).max(0.);
        if !candidate.score.is_finite() {
            return None;
        }
        // Same objective units. Preserve the original matching on ties, then
        // use canonical neuron order rather than incidental rank-slot order.
        let key = (
            if candidate.preferred { 0 } else { 1 },
            match candidate.choice {
                FamilyChoice::Joint { pairs, .. } => {
                    [0, pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]]
                }
                FamilyChoice::Triplet { neurons, .. } => [1, neurons[0], neurons[1], neurons[2], 0],
            },
        );
        if candidate.score > best_score
            || (candidate.score == best_score && best_key.is_some_and(|k| key < k))
        {
            best_score = candidate.score;
            chosen = Some(candidate.choice);
            best_key = Some(key);
        }
    }
    check(lease, end)?;
    chosen
}
