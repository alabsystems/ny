// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private unconditional transport covers of the owned direct head source.
//! ReLU(a)-ReLU(b) >= min(0,a-b). Each signed difference is contracted over
//! EVERY Gemm1 feature with independent fold/bias errors before box support.
//! Matching is advice; every residual Q coefficient and the original B remain
//! in the complete certificate. No imported rows, cells or solver authority.

use super::*;
use std::cmp::Ordering;

const MAX_PAIRS: usize = 16;
const MAX_NEIGHBORS: usize = 4;
const MAX_CLASSES: usize = 9;
const MAX_FEATURES: usize = 65_536;
const MAX_HYPEREDGE_POSITIVES: usize = 3;
const MAX_JENSEN_RANK_TRIALS: usize = 512;

#[derive(Default)]
pub(super) struct Report {
    pub(super) pairs_checked: usize,
    pub(super) candidates: usize,
    pub(super) classes_improved: usize,
    pub(super) seed_checks: usize,
    cache_hits: usize,
    fold_terms: usize,
    geometry_terms: usize,
    fractional_candidates: usize,
    fractional_edges: usize,
    fractional_shared_endpoints: usize,
    fractional_seed_checks: usize,
    fractional_cache_hits: usize,
    fractional_fold_terms: usize,
    fractional_classes_improved: usize,
    jensen_candidates: usize,
    jensen_hyperedges: usize,
    jensen_seed_checks: usize,
    jensen_cache_hits: usize,
    jensen_fold_terms: usize,
    jensen_classes_improved: usize,
    jensen_rank_trials: usize,
    slice_expired: bool,
    authority_expired: bool,
}

impl Report {
    pub(super) fn receipt(&self) {
        eprintln!("[direct-head-transport] pairs={} candidates={} classes_improved={} seed_checks={} cache_hits={} fold_terms={} geometry_terms={} slice_expired={} authority_expired={}",
            self.pairs_checked, self.candidates, self.classes_improved, self.seed_checks,
            self.cache_hits, self.fold_terms, self.geometry_terms,
            u8::from(self.slice_expired), u8::from(self.authority_expired));
        eprintln!("[direct-head-fractional-transport] candidates={} edges={} shared_endpoints={} seed_checks={} cache_hits={} fold_terms={} classes_improved={} slice_expired={} authority_expired={}",
            self.fractional_candidates, self.fractional_edges, self.fractional_shared_endpoints,
            self.fractional_seed_checks, self.fractional_cache_hits, self.fractional_fold_terms,
            self.fractional_classes_improved, u8::from(self.slice_expired), u8::from(self.authority_expired));
        eprintln!("[direct-head-jensen-transport] candidates={} hyperedges={} seed_checks={} cache_hits={} fold_terms={} classes_improved={} rank_trials={} slice_expired={} authority_expired={}",
            self.jensen_candidates, self.jensen_hyperedges, self.jensen_seed_checks,
            self.jensen_cache_hits, self.jensen_fold_terms, self.jensen_classes_improved,
            self.jensen_rank_trials, u8::from(self.slice_expired), u8::from(self.authority_expired));
    }
}

#[derive(Clone, Copy)]
struct Pair {
    positive: usize,
    negative: usize,
    tau: f64,
}

// Closed-form supports are certificate authority; no nominal vertex or LP
// geometry is used. These are conservative special cases of NY's existing
// paired_relu_hull / paired_difference transported-ReLU support primitive.
fn difference_floor(positive: I, negative: I, delta_lo: f64) -> Option<f64> {
    if !positive.valid() || !negative.valid() || !good(delta_lo) {
        return None;
    }
    let independent = I::p(positive.lo.max(0.0))
        .sub(I::p(negative.hi.max(0.0)))?
        .lo;
    // If b>=0, relu(a)-relu(b)>=a-b. If b<0, the difference is relu(a).
    // Thus min(delta_lo, relu(a)'s marginal lower) is always valid.
    let mut floor = delta_lo.min(positive.lo.max(0.0)).max(independent);
    if negative.lo >= 0.0 {
        floor = floor.max(delta_lo);
    }
    good(floor).then_some(floor)
}

/// Full recipient replay. Arbitrary finite tau>=0 is valid: capacities only
/// guide matching. Subtract/add tau DIRECTED and retain the entire remainder,
/// even if a residual changes sign or nominal cancellation looks exact.
fn candidate(
    cache: &mut adaptive::Contractions<'_, '_>,
    source: &DirectScalarHeadSource,
    q: &[I],
    b: I,
    pairs: &[Pair],
    deadline: Instant,
    work: &mut adaptive::Report,
) -> Option<f64> {
    #[allow(unused_mut)]
    let mut deadline = deadline;
    if pairs.is_empty()
        || pairs.len() > MAX_PAIRS
        || q.len() != source.y.len()
        || q.iter().any(|q| !q.valid())
        || !b.valid()
        || !live(deadline)
        || !fp_qualified()
        || !cache.belongs_to(source)
    {
        return None;
    }
    // Distinct directed edges may share either endpoint. Exact duplicate
    // edges are refused as bounded admission, not as a capacity premise.
    // Validate the entire proposal before contractions or any partial result.
    for (i, pair) in pairs.iter().enumerate() {
        if pair.positive >= q.len()
            || pair.negative >= q.len()
            || pair.positive == pair.negative
            || pairs[..i]
                .iter()
                .any(|other| other.positive == pair.positive && other.negative == pair.negative)
            || !good(pair.tau)
            || pair.tau < 0.0
        {
            return None;
        }
    }
    let mut residual = q.to_vec();
    let mut offset = b;
    for pair in pairs {
        if !live(deadline) {
            return None;
        }
        // A zero mass is exactly the zero functional. It does not need a
        // difference certificate, but still retains all original Q below.
        if pair.tau == 0.0 {
            continue;
        }
        let mut seed = vec![0.0; q.len()];
        seed[pair.positive] = 1.0;
        seed[pair.negative] = -1.0;
        #[cfg(test)]
        let seeds_before = work.contraction_counts().0;
        let delta = cache.lower(&seed, deadline, work)?;
        let floor = difference_floor(source.y[pair.positive], source.y[pair.negative], delta)?;
        offset = offset.add(I::p(pair.tau).mul(I::p(floor))?)?;
        residual[pair.positive] = residual[pair.positive].sub(I::p(pair.tau))?;
        residual[pair.negative] = residual[pair.negative].add(I::p(pair.tau))?;
        #[cfg(test)]
        if expire_after_pair()
            || (work.contraction_counts().0 > seeds_before
                && STOP_AFTER_NEW_PAIR.with(|stop| stop.replace(false)))
        {
            deadline = Instant::now();
        }
    }
    // Full Q and Y, including unpaired, straddling, zero/subnormal and
    // overdrawn coefficients; no scalar term is inferred from its nominal.
    for (j, (&q, &y)) in residual.iter().zip(&source.y).enumerate() {
        if j & 63 == 0 && !live(deadline) {
            return None;
        }
        offset = offset.add(I::p(scalar_intercept(q, 0.0, y)?))?;
    }
    (live(deadline) && fp_qualified() && offset.valid()).then_some(offset.lo)
}

// lambda and tau are represented point proposals. The mathematical masses
// are tau*lambda, which need not be representable and are NEVER RN-coalesced.
#[derive(Clone)]
struct Hyperedge {
    positive: Vec<(usize, f64)>,
    negative: usize,
    tau: f64,
}

/// Jensen: sum lambda_i*rho(y_i) >= rho(sum lambda_i*y_i), lambda_i>=0.
/// Thus the normalized difference support is valid; tau>=0 scales it. A
/// source-global seed(lambda,-1) is reusable across recipient scales. Every
/// tau*lambda product and full Q remainder is nevertheless repriced directed.
fn jensen_candidate(
    cache: &mut adaptive::Contractions<'_, '_>,
    source: &DirectScalarHeadSource,
    q: &[I],
    b: I,
    hyperedges: &[Hyperedge],
    deadline: Instant,
    work: &mut adaptive::Report,
) -> Option<f64> {
    #[allow(unused_mut)]
    let mut deadline = deadline;
    if hyperedges.is_empty()
        || hyperedges.len() > MAX_PAIRS
        || q.len() != source.y.len()
        || q.iter().any(|q| !q.valid())
        || !b.valid()
        || !live(deadline)
        || !fp_qualified()
        || !cache.belongs_to(source)
    {
        return None;
    }
    // Entire candidate admission precedes the first contraction. No capacity
    // or normalization equality is a proof premise, including across edges.
    for edge in hyperedges {
        if edge.negative >= q.len()
            || edge.positive.is_empty()
            || edge.positive.len() > MAX_HYPEREDGE_POSITIVES
            || !good(edge.tau)
            || edge.tau < 0.0
        {
            return None;
        }
        for (at, &(head, lambda)) in edge.positive.iter().enumerate() {
            if head >= q.len()
                || head == edge.negative
                || !good(lambda)
                || lambda < 0.0
                || edge.positive[..at].iter().any(|&(other, _)| other == head)
            {
                return None;
            }
        }
    }
    let mut residual = q.to_vec();
    let mut offset = b;
    for edge in hyperedges {
        if !live(deadline) {
            return None;
        }
        if edge.tau == 0.0 {
            continue;
        }
        let mut seed = vec![0.0; q.len()];
        for &(head, lambda) in &edge.positive {
            seed[head] = lambda;
        }
        seed[edge.negative] = -1.0;
        #[cfg(test)]
        let old_seeds = work.contraction_counts().0;
        let delta = cache.lower(&seed, deadline, work)?;
        #[cfg(test)]
        if work.contraction_counts().0 > old_seeds
            && STOP_AFTER_NEW_HYPEREDGE.with(|stop| stop.replace(false))
        {
            deadline = Instant::now();
        }
        if !live(deadline) {
            return None;
        }
        let mut aggregate = I::p(0.0);
        let mut independent = I::p(0.0);
        for &(head, lambda) in &edge.positive {
            aggregate = aggregate.add(I::p(lambda).mul(source.y[head])?)?;
            independent = independent.add(I::p(lambda).mul(I::p(source.y[head].lo.max(0.0)))?)?;
            // Directed product BEFORE residual multiplication by potentially
            // large Y. A subnormal mathematical mass cannot disappear.
            let mass = I::p(edge.tau).mul(I::p(lambda))?;
            residual[head] = residual[head].sub(mass)?;
        }
        independent = independent.sub(I::p(source.y[edge.negative].hi.max(0.0)))?;
        let floor =
            difference_floor(aggregate, source.y[edge.negative], delta)?.max(independent.lo);
        offset = offset.add(I::p(edge.tau).mul(I::p(floor))?)?;
        residual[edge.negative] = residual[edge.negative].add(I::p(edge.tau))?;
    }
    for (j, (&q, &y)) in residual.iter().zip(&source.y).enumerate() {
        if j & 63 == 0 && !live(deadline) {
            return None;
        }
        offset = offset.add(I::p(scalar_intercept(q, 0.0, y)?))?;
    }
    (live(deadline) && fp_qualified() && offset.valid()).then_some(offset.lo)
}

// O(heads*128) projection and a sparse nearest-key shortlist. This geometry
// can miss good pairs or rank bad ones; it NEVER certifies their difference.
struct Geometry {
    rows: Vec<Vec<f64>>,
    bias: Vec<f64>,
    center: Vec<f64>,
    radius: Vec<f64>,
}

impl Geometry {
    fn build(
        eng: &BackwardEngine<'_>,
        source: &DirectScalarHeadSource,
        deadline: Instant,
        report: &mut Report,
    ) -> Option<Self> {
        let (w, b, (heads, width)) = eng.net.gemm1();
        if heads != source.y.len()
            || width != source.features.len()
            || width == 0
            || width > MAX_FEATURES
            || w.len() != heads.checked_mul(width)?
            || b.len() != heads
        {
            return None;
        }
        let n = width.min(FUNCTIONAL_MAX_FEATURES);
        let indices: Vec<_> = (0..n)
            .map(|k| if n == 1 { 0 } else { k * (width - 1) / (n - 1) })
            .collect();
        let mut center = Vec::with_capacity(n);
        let mut radius = Vec::with_capacity(n);
        for &k in &indices {
            if !live(deadline) {
                return None;
            }
            let x = source.features[k];
            center.push(0.5 * x.lo + 0.5 * x.hi);
            radius.push((0.5 * x.hi - 0.5 * x.lo) * (width as f64 / n as f64));
        }
        let mut rows = Vec::with_capacity(heads);
        for j in 0..heads {
            let mut row = Vec::with_capacity(n);
            for (i, &k) in indices.iter().enumerate() {
                if i & 63 == 0 && !live(deadline) {
                    return None;
                }
                row.push(w[j * width + k]);
                report.geometry_terms += 1;
            }
            rows.push(row);
        }
        Some(Self {
            rows,
            bias: b.to_vec(),
            center,
            radius,
        })
    }

    fn compare(&self, a: usize, b: usize) -> Ordering {
        // Stable shifted features make small weight perturbations change
        // authored bias too. Put coefficient keys first, so a small bias
        // change does not separate every near twin from its row neighborhood.
        self.rows[a]
            .iter()
            .zip(self.rows[b].iter())
            .map(|(a, b)| a.total_cmp(b))
            .find(|c| *c != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
            .then(self.bias[a].total_cmp(&self.bias[b]))
    }

    fn rank(
        &self,
        source: &DirectScalarHeadSource,
        pair: Pair,
        deadline: Instant,
        report: &mut Report,
    ) -> Option<f64> {
        let mut difference = self.bias[pair.positive] - self.bias[pair.negative];
        for k in 0..self.center.len() {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            let d = self.rows[pair.positive][k] - self.rows[pair.negative][k];
            difference += d * self.center[k] - d.abs() * self.radius[k];
            report.geometry_terms += 1;
        }
        let independent = source.y[pair.positive].lo.max(0.0) - source.y[pair.negative].hi.max(0.0);
        let rank = pair.tau * (difference.min(0.0) - independent);
        good(rank).then_some(rank)
    }

    fn edges(
        &self,
        source: &DirectScalarHeadSource,
        q: &[I],
        deadline: Instant,
        report: &mut Report,
    ) -> Option<Vec<(f64, Pair)>> {
        if q.len() != self.rows.len() || q.iter().any(|q| !q.valid()) {
            return None;
        }
        let mut positive: Vec<_> = (0..q.len()).filter(|&j| q[j].lo > 0.0).collect();
        positive.sort_by(|&a, &b| self.compare(a, b).then(a.cmp(&b)));
        let mut edges = Vec::new();
        for negative in (0..q.len()).filter(|&j| q[j].hi < 0.0) {
            if !live(deadline) {
                return None;
            }
            let at = positive.partition_point(|&p| self.compare(p, negative) == Ordering::Less);
            for &p in &positive
                [at.saturating_sub(MAX_NEIGHBORS)..(at + MAX_NEIGHBORS).min(positive.len())]
            {
                let pair = Pair {
                    positive: p,
                    negative,
                    tau: q[p].lo.min(-q[negative].hi),
                };
                if let Some(rank) = self.rank(source, pair, deadline, report) {
                    edges.push((rank, pair));
                }
            }
        }
        edges.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then(a.1.positive.cmp(&b.1.positive))
                .then(a.1.negative.cmp(&b.1.negative))
        });
        Some(edges)
    }

    fn disjoint(edges: &[(f64, Pair)], heads: usize, deadline: Instant) -> Option<Vec<Pair>> {
        let mut used = vec![false; heads];
        let mut pairs = Vec::new();
        for &(_, pair) in edges {
            if !live(deadline) {
                return None;
            }
            if !used[pair.positive] && !used[pair.negative] {
                used[pair.positive] = true;
                used[pair.negative] = true;
                pairs.push(pair);
                if pairs.len() == MAX_PAIRS {
                    break;
                }
            }
        }
        Some(pairs)
    }

    // Retained for the disjoint reference and existing tests. Production
    // computes each sparse shortlist once, then reuses it in both passes.
    #[cfg(test)]
    fn propose(
        &self,
        source: &DirectScalarHeadSource,
        q: &[I],
        deadline: Instant,
        report: &mut Report,
    ) -> Option<Vec<Pair>> {
        Self::disjoint(&self.edges(source, q, deadline, report)?, q.len(), deadline)
    }

    fn fractional(edges: &[(f64, Pair)], q: &[I], deadline: Instant) -> Option<(Vec<Pair>, usize)> {
        // Floating remaining capacities only choose represented point masses.
        // The checker recomputes every full interval residual, including any
        // overdraw caused by advice rounding. No sum of masses has authority.
        let mut positive: Vec<_> = q.iter().map(|q| q.lo.max(0.0)).collect();
        let mut negative: Vec<_> = q.iter().map(|q| (-q.hi).max(0.0)).collect();
        let mut degree = vec![0usize; q.len()];
        let mut pairs = Vec::new();
        for &(_, edge) in edges {
            if !live(deadline) {
                return None;
            }
            let tau = positive[edge.positive].min(negative[edge.negative]);
            if !good(tau) || tau <= 0.0 {
                continue;
            }
            pairs.push(Pair { tau, ..edge });
            positive[edge.positive] = (positive[edge.positive] - tau).max(0.0);
            negative[edge.negative] = (negative[edge.negative] - tau).max(0.0);
            degree[edge.positive] += 1;
            degree[edge.negative] += 1;
            if pairs.len() == MAX_PAIRS {
                break;
            }
        }
        let shared = degree.iter().filter(|&&n| n > 1).count();
        Some((pairs, shared))
    }

    // Sampled nominal geometry is search ONLY. Even omitted coefficients,
    // nearest arithmetic, normalization drift and capacities have no authority.
    fn combination_score(
        &self,
        positive: &[(usize, f64)],
        negative: usize,
        deadline: Instant,
        report: &mut Report,
    ) -> Option<f64> {
        let mut difference = -self.bias[negative];
        for &(head, lambda) in positive {
            difference += lambda * self.bias[head];
        }
        for k in 0..self.center.len() {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            let mut d = -self.rows[negative][k];
            for &(head, lambda) in positive {
                d += lambda * self.rows[head][k];
            }
            difference += d * self.center[k] - d.abs() * self.radius[k];
            report.geometry_terms += 1;
        }
        good(difference).then_some(difference.min(0.0))
    }

    fn jensen(
        &self,
        source: &DirectScalarHeadSource,
        q: &[I],
        edges: &[(f64, Pair)],
        deadline: Instant,
        report: &mut Report,
    ) -> Option<Vec<Hyperedge>> {
        let mut capacity: Vec<_> = q.iter().map(|q| q.lo.max(0.0)).collect();
        let mut hyperedges = Vec::new();
        for negative in (0..q.len()).filter(|&j| q[j].hi < 0.0) {
            if !live(deadline) {
                return None;
            }
            let mut positive: Vec<_> = edges
                .iter()
                .filter(|(_, p)| p.negative == negative)
                .map(|(_, p)| p.positive)
                .collect();
            positive.sort_unstable();
            positive.dedup();
            positive.truncate(2 * MAX_NEIGHBORS);
            positive.retain(|&head| capacity[head] > 0.0);
            if positive.len() < 2 {
                continue;
            }
            let mut single = f64::NEG_INFINITY;
            for &head in &positive {
                single = single.max(self.combination_score(
                    &[(head, 1.0)],
                    negative,
                    deadline,
                    report,
                )?);
            }
            let mut best: Option<(f64, Hyperedge)> = None;
            'trials: for a in 0..positive.len() {
                for c in a + 1..positive.len() {
                    for lambda in [0.25, 0.5, 0.75] {
                        if report.jensen_rank_trials >= MAX_JENSEN_RANK_TRIALS {
                            break 'trials;
                        }
                        report.jensen_rank_trials += 1;
                        let proposal = [(positive[a], lambda), (positive[c], 1.0 - lambda)];
                        let score =
                            self.combination_score(&proposal, negative, deadline, report)?;
                        // Conservative opportunity gate: exact pairs with score0
                        // cannot spend a new combination seed. Ranking never certifies.
                        if score <= single {
                            continue;
                        }
                        let tau = (-q[negative].hi)
                            .min(capacity[positive[a]] / lambda)
                            .min(capacity[positive[c]] / (1.0 - lambda));
                        let independent = lambda * source.y[positive[a]].lo.max(0.0)
                            + (1.0 - lambda) * source.y[positive[c]].lo.max(0.0)
                            - source.y[negative].hi.max(0.0);
                        let rank = tau * (score - independent);
                        if !good(tau) || tau <= 0.0 || !good(rank) || rank <= 0.0 {
                            continue;
                        }
                        if best.as_ref().is_none_or(|(old, _)| rank > *old) {
                            best = Some((
                                rank,
                                Hyperedge {
                                    positive: proposal.to_vec(),
                                    negative,
                                    tau,
                                },
                            ));
                        }
                    }
                }
            }
            if let Some((_, edge)) = best {
                for &(head, lambda) in &edge.positive {
                    capacity[head] = (capacity[head] - edge.tau * lambda).max(0.0);
                }
                hyperedges.push(edge);
                if hyperedges.len() == MAX_PAIRS {
                    break;
                }
            }
            if report.jensen_rank_trials >= MAX_JENSEN_RANK_TRIALS {
                break;
            }
        }
        Some(hyperedges)
    }
}

/// Existing disjoint candidates for ALL original margins run first. Then a
/// second complete shared-mass candidate may improve each still-open margin. A
/// shared cache admits at most16 full point seeds across all three passes.
/// Jensen combinations are ranked only after all old recipients finish.
/// Local expiry retains completed candidates; original expiry/generation/FP
/// refusal restores ALL incoming floors. The outer authorizer rechecks again.
pub(super) fn refine(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    margins: &[(Vec<I>, I)],
    bounds: &mut [f64],
    deadline: Instant,
    authority: Instant,
) -> Report {
    refine_portfolio(eng, source, margins, bounds, deadline, authority, true)
}

fn refine_portfolio(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    margins: &[(Vec<I>, I)],
    bounds: &mut [f64],
    deadline: Instant,
    authority: Instant,
    jensen: bool,
) -> Report {
    let mut report = Report::default();
    let before = bounds.to_vec();
    let valid = || {
        live(authority)
            && fp_qualified()
            && source
                .seal
                .same_frame(eng, source.target, &source.adversaries)
    };
    if bounds.len() != margins.len()
        || bounds.len() != source.adversaries.len()
        || bounds.is_empty()
        || bounds.len() > MAX_CLASSES
        || bounds.iter().any(|v| !good(*v))
        || margins
            .iter()
            .any(|(q, b)| q.len() != source.y.len() || !b.valid() || q.iter().any(|v| !v.valid()))
        || !live(deadline)
        || !valid()
    {
        report.slice_expired = !live(deadline);
        report.authority_expired = !live(authority);
        return report;
    }
    if !bounds.iter().zip(margins).any(|(&lower, (q, _))| {
        lower <= 0.0 && q.iter().any(|q| q.lo > 0.0) && q.iter().any(|q| q.hi < 0.0)
    }) {
        return report;
    }
    let Some(geometry) = Geometry::build(eng, source, deadline, &mut report) else {
        report.slice_expired = !live(deadline);
        return report;
    };
    let mut cache = adaptive::Contractions::new(eng, source);
    let mut work = adaptive::Report::default();
    let mut shortlists = vec![None; bounds.len()];
    // Preserve the entire old portfolio before spending a single new seed on
    // shared mass. Later original classes retain their old cache/window chance.
    for row in 0..bounds.len() {
        if !live(deadline) || !live(authority) {
            break;
        }
        if bounds[row] > 0.0 {
            continue;
        }
        let (q, b) = &margins[row];
        let Some(edges) = geometry.edges(source, q, deadline, &mut report) else {
            break;
        };
        let Some(pairs) = Geometry::disjoint(&edges, q.len(), deadline) else {
            break;
        };
        if let Some(lower) = candidate(&mut cache, source, q, *b, &pairs, deadline, &mut work) {
            bounds[row] = bounds[row].max(lower);
            report.pairs_checked += pairs.len();
            report.candidates += 1;
            shortlists[row] = Some(edges);
        }
    }
    let after_disjoint = bounds.to_vec();
    let old_counts = work.contraction_counts();
    #[cfg(test)]
    let (deadline, authority) = match STOP_AFTER_DISJOINT.with(|stop| stop.replace(0)) {
        1 => (Instant::now(), authority),
        2 => (deadline, Instant::now()),
        3 => {
            STOP_AFTER_NEW_PAIR.with(|stop| stop.set(true));
            (deadline, authority)
        }
        _ => (deadline, authority),
    };
    for row in 0..bounds.len() {
        if !live(deadline) || !live(authority) {
            break;
        }
        if bounds[row] > 0.0 {
            continue;
        }
        let Some(edges) = &shortlists[row] else {
            // A refused/partial disjoint candidate is not a completed floor.
            // Preserve its refusal rather than retrying with a fresh budget.
            continue;
        };
        let (q, b) = &margins[row];
        let Some((pairs, shared)) = Geometry::fractional(edges, q, deadline) else {
            break;
        };
        if shared == 0 {
            continue;
        }
        if let Some(lower) = candidate(&mut cache, source, q, *b, &pairs, deadline, &mut work) {
            bounds[row] = bounds[row].max(lower);
            report.pairs_checked += pairs.len();
            report.candidates += 1;
            report.fractional_candidates += 1;
            report.fractional_edges += pairs.len();
            report.fractional_shared_endpoints += shared;
        }
    }
    let after_fractional = bounds.to_vec();
    let fractional_counts = work.contraction_counts();
    report.fractional_seed_checks = fractional_counts.0 - old_counts.0;
    report.fractional_cache_hits = fractional_counts.1 - old_counts.1;
    report.fractional_fold_terms = fractional_counts.2 - old_counts.2;
    #[cfg(test)]
    let (deadline, authority) = match STOP_AFTER_FRACTIONAL.with(|stop| stop.replace(0)) {
        1 => (Instant::now(), authority),
        2 => (deadline, Instant::now()),
        3 => {
            STOP_AFTER_NEW_HYPEREDGE.with(|stop| stop.set(true));
            (deadline, authority)
        }
        _ => (deadline, authority),
    };
    // ALL old disjoint/shared recipients finish before the first new seed.
    // A whole finite proposal is ranked before ANY full contraction. Cache
    // misses (including refused work) share the original global16 limit.
    if jensen && fractional_counts.0 < MAX_PAIRS {
        for row in 0..bounds.len() {
            if !live(deadline) || !live(authority) {
                break;
            }
            if bounds[row] > 0.0 {
                continue;
            }
            let Some(edges) = &shortlists[row] else {
                continue;
            };
            let (q, b) = &margins[row];
            let Some(hyperedges) = geometry.jensen(source, q, edges, deadline, &mut report) else {
                break;
            };
            if let Some(lower) =
                jensen_candidate(&mut cache, source, q, *b, &hyperedges, deadline, &mut work)
            {
                bounds[row] = bounds[row].max(lower);
                report.jensen_candidates += 1;
                report.jensen_hyperedges += hyperedges.len();
            }
        }
    }
    (report.seed_checks, report.cache_hits, report.fold_terms) = work.contraction_counts();
    report.jensen_seed_checks = report.seed_checks - fractional_counts.0;
    report.jensen_cache_hits = report.cache_hits - fractional_counts.1;
    report.jensen_fold_terms = report.fold_terms - fractional_counts.2;
    report.slice_expired = !live(deadline);
    report.authority_expired = !live(authority);
    if !valid() || !live(authority) {
        bounds.copy_from_slice(&before);
        report.classes_improved = 0;
        report.fractional_classes_improved = 0;
        report.jensen_classes_improved = 0;
    } else {
        report.classes_improved = before
            .iter()
            .zip(bounds.iter())
            .filter(|(a, b)| b > a)
            .count();
        report.fractional_classes_improved = after_disjoint
            .iter()
            .zip(after_fractional.iter())
            .filter(|(a, b)| b > a)
            .count();
        report.jensen_classes_improved = after_fractional
            .iter()
            .zip(bounds.iter())
            .filter(|(a, b)| b > a)
            .count();
    }
    report
}

#[cfg(test)]
thread_local! {
    static STOP_AFTER_PAIR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static STOP_AFTER_NEW_PAIR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static STOP_AFTER_DISJOINT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static STOP_AFTER_FRACTIONAL: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static STOP_AFTER_NEW_HYPEREDGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
fn expire_after_pair() -> bool {
    STOP_AFTER_PAIR.with(|stop| stop.replace(false))
}

#[cfg(test)]
#[path = "transport_direct_head_tests.rs"]
mod tests;
