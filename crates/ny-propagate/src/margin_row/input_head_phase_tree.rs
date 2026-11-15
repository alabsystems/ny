// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Source-owned complete nominal-phase frontier. This is a separate complete
//! stage under the original instance authority, never a renewed root slice.
use super::*;
use ny_core::head_functional_advice::{HeadFunctionalLpHint, FUNCTIONAL_MAX_ROWS};

const RESERVE: Duration = Duration::from_millis(75);
type Row = triangle::OwnedRow;

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub(super) nodes: usize,
    pub(super) frontier: usize,
    pub(super) depth: usize,
    pub(super) calls: usize,
    /// Explicit weighted work units, not an instruction or elapsed-time bound.
    pub(super) work: usize,
}

impl Limits {
    fn valid(self) -> bool {
        (1..=8191).contains(&self.nodes)
            && (1..=4096).contains(&self.frontier)
            && (1..=128).contains(&self.depth)
            && self.calls <= 8192
            && (1..=1_000_000_000).contains(&self.work)
    }
}

#[derive(Default)]
pub(super) struct Refinement {
    pub(super) lower: f64,
    pub(super) complete: bool,
    pub(super) nodes: usize,
    pub(super) splits: usize,
    pub(super) open: usize,
    pub(super) closed: usize,
    pub(super) empty: usize,
    pub(super) depth: usize,
    pub(super) calls: usize,
    pub(super) native_replayed: usize,
    pub(super) ay_replayed: usize,
    pub(super) work: usize,
    pub(super) elapsed_seconds: f64,
}

#[derive(Clone)]
struct Node {
    path: Vec<(usize, bool)>,
    floor: f64,
}

struct PathRows<'a> {
    source: &'a Source,
    rows: Vec<Row>,
}

struct Model<'a> {
    source: &'a Source,
    boxes: Vec<I>,
    rows: Vec<Row>,
    roles: Vec<Option<usize>>,
    rank: Vec<usize>,
    work_hint: usize,
}

fn midpoint(a: I) -> Option<f64> {
    let v = 0.5 * a.lo + 0.5 * a.hi;
    (a.valid() && good(v)).then_some(v)
}

impl<'a> Model<'a> {
    fn build(
        source: &'a Source,
        prefix: Option<triangle::PhasePrefix<'a>>,
        limits: Limits,
        end: Instant,
    ) -> Option<Self> {
        let room = FUNCTIONAL_MAX_ROWS.checked_sub(2usize.checked_mul(limits.depth)?)?;
        let parts = prefix.and_then(|p| p.into_phase_parts(source, room, end));
        let (boxes, rows) = match parts {
            Some(parts) => parts,
            None => triangle::Model::build_phase_base(source, end)?.into_phase_parts(),
        };
        if rows.len() > room || boxes.len() != source.spec.inputs + source.spec.heads {
            return None;
        }
        let mut roles = Vec::with_capacity(source.spec.heads);
        let mut at = 0usize;
        let mut rank = Vec::new();
        for (head, (&q, &y)) in source.q.iter().zip(&source.y).enumerate() {
            if !live(end) {
                return None;
            }
            roles.push(Some(at));
            at = at.checked_add(1 + usize::from(y.hi > 0.0))?;
            if q.lo < 0.0 && y.lo < 0.0 && y.hi > 0.0 {
                let score = (-q.lo) * (-y.lo).min(y.hi);
                if good(score) {
                    rank.push((head, score));
                }
            }
        }
        if at > rows.len() {
            return None;
        }
        rank.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let work_hint = rows
            .iter()
            .try_fold(boxes.len(), |n, r| n.checked_add(r.coefficients.len()))?;
        Some(Self {
            source,
            boxes,
            rows,
            roles,
            rank: rank.into_iter().map(|v| v.0).collect(),
            work_hint,
        })
    }

    fn path_rows(&self, path: &[(usize, bool)], end: Instant) -> Option<PathRows<'a>> {
        let s = self.source;
        let mut seen = vec![false; s.spec.heads];
        let mut rows = Vec::with_capacity(path.len().checked_mul(2)?);
        for &(head, active) in path {
            if !live(end) || head >= seen.len() || std::mem::replace(&mut seen[head], true) {
                return None;
            }
            let b = I::p(f64::from(s.spec.first_bias[head]));
            let e = I::p(s.first_error[head]);
            let mut sign = Vec::new();
            let mut cap = Vec::new();
            for input in 0..s.spec.inputs {
                if input & 63 == 0 && !live(end) {
                    return None;
                }
                let w = f64::from(s.spec.first[input * s.spec.heads + head]);
                if w != 0.0 {
                    sign.push((input, I::p(if active { w } else { -w })));
                    if active {
                        cap.push((input, I::p(w)));
                    }
                }
            }
            rows.push(Row {
                coefficients: sign,
                lower: if active { b.neg() } else { b },
            });
            cap.push((s.spec.inputs + head, I::p(-1.0)));
            rows.push(Row {
                coefficients: cap,
                lower: if active { b.neg().sub(e)? } else { e.neg() },
            });
        }
        rows.iter()
            .all(|r| r.valid(self.boxes.len()))
            .then_some(PathRows { source: s, rows })
    }

    fn request(&self, path: &PathRows<'_>, end: Instant) -> Option<HeadFunctionalLpRequest> {
        if !std::ptr::eq(self.source, path.source) {
            return None;
        }
        let mut rows = Vec::with_capacity(self.rows.len().checked_add(path.rows.len())?);
        for row in self.rows.iter().chain(&path.rows) {
            if !live(end) || !row.valid(self.boxes.len()) {
                return None;
            }
            let coefficients = row
                .coefficients
                .iter()
                .map(|&(j, a)| midpoint(a).map(|v| (j, v)))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .filter(|v| v.1 != 0.0)
                .collect();
            rows.push(HeadFunctionalLpRow {
                coefficients,
                lower: row.lower.lo,
            });
        }
        let mut objective = vec![0.0; self.source.spec.inputs];
        objective.extend(
            self.source
                .q
                .iter()
                .copied()
                .map(midpoint)
                .collect::<Option<Vec<_>>>()?,
        );
        let request = HeadFunctionalLpRequest {
            protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
            feature_count: self.source.spec.inputs,
            bounds: self.boxes.iter().map(|b| [b.lo, b.hi]).collect(),
            objective,
            rows,
            slope_rows: self.roles.clone(),
            // Optional V2 role tags are not the Source's branch identity.
            // The complete path lives privately in Node and these GEQ rows.
            phase_rows: vec![],
        };
        (live(end) && request.validate()).then_some(request)
    }

    fn replay(
        &self,
        source: &Source,
        path: &PathRows<'_>,
        duals: &[f64],
        sign: f64,
        zero: bool,
        end: Instant,
    ) -> Option<f64> {
        if !std::ptr::eq(self.source, source)
            || !Arc::ptr_eq(&self.source.spec, &source.spec)
            || !std::ptr::eq(self.source, path.source)
            || !fp_qualified()
            || !live(end)
            || duals.len() != self.rows.len().checked_add(path.rows.len())?
            || (sign != 1.0 && sign != -1.0)
        {
            return None;
        }
        let mut residual = vec![I::p(0.0); source.spec.inputs];
        residual.extend(source.q.iter().map(|&q| if zero { I::p(0.0) } else { q }));
        let mut value = if zero {
            I::p(0.0)
        } else {
            source.bias.sub(I::p(source.output_error))?
        };
        for (row, &raw) in self.rows.iter().chain(&path.rows).zip(duals) {
            if !live(end) || !good(raw) || !row.valid(residual.len()) {
                return None;
            }
            let lambda = I::p((sign * raw).max(0.0));
            if lambda.lo == 0.0 {
                continue;
            }
            value = value.add(lambda.mul(row.lower)?)?;
            for (at, &(j, a)) in row.coefficients.iter().enumerate() {
                if at & 63 == 0 && !live(end) {
                    return None;
                }
                residual[j] = residual[j].sub(lambda.mul(a)?)?;
            }
        }
        for (j, (&a, &b)) in residual.iter().zip(&self.boxes).enumerate() {
            if j & 63 == 0 && !live(end) {
                return None;
            }
            value = value.add(a.mul(b)?)?;
        }
        (live(end) && fp_qualified() && value.valid()).then_some(value.lo)
    }

    fn native(&self, node: &Node) -> Option<Vec<f64>> {
        let s = self.source;
        let mut dual = vec![
            0.0;
            self.rows
                .len()
                .checked_add(node.path.len().checked_mul(2)?)?
        ];
        for head in 0..s.spec.heads {
            let q = midpoint(s.q[head])?;
            let at = self.roles[head]?;
            if q >= 0.0 {
                dual[at] = q;
                continue;
            }
            if let Some(bit) = node.path.iter().position(|v| v.0 == head) {
                dual[self.rows.len() + 2 * bit + 1] = -q;
            } else if s.y[head].hi > 0.0 {
                let r = self.rows.get(at + 1)?.coefficients.last()?.1;
                let scale = -midpoint(r)?;
                if scale > 0.0 {
                    dual[at + 1] = -q / scale;
                }
            }
        }
        dual.iter().all(|&v| good(v)).then_some(dual)
    }
}

/// A bounded phase-I proposal only. Its delta is never Source proof state.
/// Source later replays just the original rows with the zero objective.
fn phase_one(request: &HeadFunctionalLpRequest, end: Instant) -> Option<HeadFunctionalLpRequest> {
    if request.feature_count >= 128 {
        return None;
    }
    let d = request.feature_count;
    let mut upper = I::p(1.0);
    for row in &request.rows {
        if !live(end) {
            return None;
        }
        let mut least = I::p(0.0);
        for &(j, a) in &row.coefficients {
            least = least.add(I::p(a).mul(I {
                lo: request.bounds[j][0],
                hi: request.bounds[j][1],
            })?)?;
        }
        upper.hi = upper.hi.max(I::p(row.lower).sub(least)?.hi + 1.0);
    }
    if !good(upper.hi) || upper.hi <= 0.0 {
        return None;
    }
    let mut trial = request.clone();
    trial.feature_count += 1;
    trial.bounds.insert(d, [0.0, upper.hi]);
    trial.objective = vec![0.0; trial.bounds.len()];
    trial.objective[d] = 1.0;
    for row in &mut trial.rows {
        for (j, _) in &mut row.coefficients {
            if *j >= d {
                *j += 1;
            }
        }
        let at = row.coefficients.partition_point(|v| v.0 < d);
        row.coefficients.insert(at, (d, 1.0));
    }
    (live(end) && trial.validate()).then_some(trial)
}

fn propose<F>(
    request: &HeadFunctionalLpRequest,
    end: Instant,
    calls: &mut usize,
    limits: Limits,
    provider: &F,
) -> Option<HeadFunctionalLpHint>
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    if !live(end) || *calls >= limits.calls {
        return None;
    }
    let provider = std::panic::catch_unwind(std::panic::AssertUnwindSafe(provider))
        .ok()
        .flatten()?;
    if !std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        provider.supports_functional_advice()
    }))
    .ok()?
    {
        return None;
    }
    *calls += 1;
    let hint = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        provider.propose_functional(request, end)
    }))
    .ok()
    .flatten()?;
    (live(end) && hint.validate(request)).then_some(hint)
}

fn snapshot(frontier: &[Node], closed: f64, before: f64, authority: Instant) -> f64 {
    let floor = frontier.iter().map(|n| n.floor).fold(closed, f64::min);
    if live(authority) && fp_qualified() && good(floor) {
        before.max(floor)
    } else {
        before
    }
}

pub(super) fn refine_with_prefix<'a, F>(
    source: &'a Source,
    before: f64,
    prefix: Option<triangle::PhasePrefix<'a>>,
    search_deadline: Instant,
    authority: Instant,
    limits: Limits,
    provider: &F,
) -> Refinement
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let started = Instant::now();
    let mut report = Refinement {
        lower: before,
        ..Refinement::default()
    };
    let _ = (|| {
        let end = search_deadline.min(authority);
        let proof_end = end.checked_sub(RESERVE)?;
        if !limits.valid() || !good(before) || !live(proof_end) || !fp_qualified() {
            return None;
        }
        let setup = prefix.as_ref().and_then(|p| p.root_work_hint()).unwrap_or(
            source
                .spec
                .heads
                .checked_mul(2usize.checked_mul(source.spec.inputs)?.checked_add(4)?)?
                .checked_add(source.spec.inputs)?,
        );
        let setup = setup.checked_mul(8)?;
        if setup > limits.work {
            return None;
        }
        report.work = setup;
        let model = Model::build(source, prefix, limits, proof_end)?;
        let mut frontier = vec![Node {
            path: vec![],
            floor: before,
        }];
        let mut closed = f64::INFINITY;
        report.nodes = 1;
        report.open = 1;
        while !frontier.is_empty() && live(proof_end) && fp_qualified() {
            let index = frontier
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.floor.total_cmp(&b.1.floor))
                .map(|v| v.0)?;
            let path_terms = frontier[index]
                .path
                .len()
                .checked_mul(source.spec.inputs.checked_mul(2)?.checked_add(1)?)?;
            let comparisons = source
                .spec
                .heads
                .checked_mul(frontier[index].path.len().checked_add(1)?)?
                .checked_mul(2)?;
            let terms = model
                .work_hint
                .checked_add(path_terms)?
                .checked_add(frontier.len())?
                .checked_add(comparisons)?;
            let charge = terms.checked_mul(16)?;
            if report.work.checked_add(charge)? > limits.work {
                break;
            }
            report.work += charge;
            let path = model.path_rows(&frontier[index].path, proof_end)?;
            let native = model.native(&frontier[index])?;
            if let Some(lower) = model.replay(source, &path, &native, 1.0, false, end) {
                frontier[index].floor = frontier[index].floor.max(lower);
                report.native_replayed += 1;
                report.lower = snapshot(&frontier, closed, report.lower, authority);
            }
            let mut empty = false;
            if frontier[index].floor <= 0.0 {
                let request = model.request(&path, proof_end)?;
                let ordinary = propose(&request, proof_end, &mut report.calls, limits, provider);
                if let Some(hint) = ordinary.as_ref() {
                    for sign in [1.0, -1.0] {
                        if let Some(lower) =
                            model.replay(source, &path, &hint.row_duals, sign, false, end)
                        {
                            frontier[index].floor = frontier[index].floor.max(lower);
                            report.ay_replayed += 1;
                            report.lower = snapshot(&frontier, closed, report.lower, authority);
                        }
                        empty |= model
                            .replay(source, &path, &hint.row_duals, sign, true, end)
                            .is_some_and(|v| v > 0.0);
                    }
                } else if let Some(trial) = phase_one(&request, proof_end) {
                    if let Some(hint) =
                        propose(&trial, proof_end, &mut report.calls, limits, provider)
                    {
                        for sign in [1.0, -1.0] {
                            empty |= model
                                .replay(source, &path, &hint.row_duals, sign, true, end)
                                .is_some_and(|v| v > 0.0);
                        }
                    }
                }
            }
            if empty || frontier[index].floor > 0.0 {
                let node = frontier.swap_remove(index);
                if empty {
                    report.empty += 1;
                } else {
                    closed = closed.min(node.floor);
                    report.closed += 1;
                }
                report.open = frontier.len();
                report.lower = snapshot(&frontier, closed, report.lower, authority);
                continue;
            }
            let node = &frontier[index];
            let split = model
                .rank
                .iter()
                .copied()
                .find(|head| !node.path.iter().any(|v| v.0 == *head));
            if node.path.len() >= limits.depth
                || report.nodes.checked_add(2)? > limits.nodes
                || frontier.len().checked_add(1)? > limits.frontier
                || split.is_none()
            {
                break;
            }
            let head = split?;
            let mut low = node.clone();
            low.path.push((head, false));
            let mut high = node.clone();
            high.path.push((head, true));
            // Both children exist before replacing their still-covered parent.
            frontier.swap_remove(index);
            frontier.push(low);
            frontier.push(high);
            report.nodes += 2;
            report.splits += 1;
            report.open = frontier.len();
            report.depth = report.depth.max(frontier.last()?.path.len());
        }
        report.open = frontier.len();
        report.complete = frontier.is_empty()
            && report.closed > 0
            && good(closed)
            && live(end)
            && live(authority)
            && fp_qualified();
        report.lower = snapshot(&frontier, closed, report.lower, authority);
        Some(())
    })();
    report.elapsed_seconds = started.elapsed().as_secs_f64();
    report
}

#[cfg(test)]
#[path = "input_head_phase_tree_tests.rs"]
mod tests;
