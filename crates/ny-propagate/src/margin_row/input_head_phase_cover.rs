// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Complete closed nominal-sign covers, owned by one immutable input Source.
//!
//! For t=Wx+b and |z-t|<=ROOT E1, the inactive nominal half t<=0
//! implies ReLU(z)<=E1; the active half t>=0 implies ReLU(z)<=t+E1.
//! These are conditional rows, never unconditional continuation recipes.
//! All masks retain every authored head, all error debt, and an inherited
//! unconditional floor. Only the minimum over the entire cover can improve it.

use super::*;
use ny_core::head_phase_cover_advice::{
    PhaseCoverLeaf, PhaseCoverLpHint, PhaseCoverLpRequest, PHASE_COVER_MAX_ROWS,
    PHASE_COVER_PROTOCOL_VERSION,
};

const LOCAL_SLICE: Duration = Duration::from_millis(750);
const REPLAY_RESERVE: Duration = Duration::from_millis(75);

type OwnedRow = triangle::OwnedRow;

pub(super) struct Model<'a> {
    source: &'a Source,
    boxes: Vec<I>,
    rows: Vec<OwnedRow>,
    selected: Vec<usize>,
    leaves: Vec<Vec<OwnedRow>>,
    request: PhaseCoverLpRequest,
}

#[derive(Default)]
pub(super) struct Refinement {
    pub(super) lower: f64,
    pub(super) selected_heads: usize,
    pub(super) leaves: usize,
    pub(super) calls: usize,
    pub(super) replayed: usize,
    pub(super) native_replayed: usize,
    pub(super) ay_replayed: usize,
    pub(super) elapsed_seconds: f64,
}

fn midpoint(value: I) -> Option<f64> {
    let point = 0.5 * value.lo + 0.5 * value.hi;
    (value.valid() && good(point)).then_some(point)
}

fn export_row(row: &OwnedRow, deadline: Instant) -> Option<HeadFunctionalLpRow> {
    if !live(deadline) || !fp_qualified() {
        return None;
    }
    let mut coefficients = Vec::with_capacity(row.coefficients.len());
    for (at, &(index, value)) in row.coefficients.iter().enumerate() {
        if at & 63 == 0 && !live(deadline) {
            return None;
        }
        let point = midpoint(value)?;
        if point != 0.0 {
            coefficients.push((index, point));
        }
    }
    Some(HeadFunctionalLpRow {
        coefficients,
        lower: row.lower.lo,
    })
}

impl<'a> Model<'a> {
    pub(super) fn build(
        source: &'a Source,
        policy: PhaseCoverPolicy,
        deadline: Instant,
    ) -> Option<Self> {
        Self::build_with_prefix(source, policy, None, deadline)
    }

    pub(super) fn build_with_prefix(
        source: &'a Source,
        policy: PhaseCoverPolicy,
        prefix: Option<triangle::PhasePrefix<'a>>,
        deadline: Instant,
    ) -> Option<Self> {
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        let d = source.spec.inputs;
        let h = source.spec.heads;
        let n = d.checked_add(h)?;
        if source.y.len() != h || source.q.len() != h {
            return None;
        }
        let mut ranked = Vec::new();
        for (head, (&y, &q)) in source.y.iter().zip(&source.q).enumerate() {
            if !live(deadline) {
                return None;
            }
            if y.lo < 0.0 && y.hi > 0.0 {
                let score = midpoint(q)?.abs() * (-y.lo).min(y.hi);
                if good(score) && score > 0.0 {
                    ranked.push((head, score));
                }
            }
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let selected: Vec<_> = ranked
            .into_iter()
            .take(policy.max_heads())
            .map(|(head, _)| head)
            .collect();
        if selected.is_empty() {
            return None;
        }
        let maximum_rows = PHASE_COVER_MAX_ROWS.checked_sub(selected.len().checked_mul(2)?)?;
        let cached =
            prefix.and_then(|prefix| prefix.into_phase_parts(source, maximum_rows, deadline));
        let (boxes, rows) = match cached {
            Some(parts) => parts,
            None => triangle::Model::build_phase_base(source, deadline)?.into_phase_parts(),
        };
        if boxes.len() != n || boxes.iter().any(|value| !value.valid()) || rows.len() > maximum_rows
        {
            return None;
        }
        let count = 1usize.checked_shl(selected.len() as u32)?;
        let mut leaves = Vec::with_capacity(count);
        let mut exported_leaves = Vec::with_capacity(count);
        for mask in 0..count {
            if !live(deadline) || !fp_qualified() {
                return None;
            }
            let mut leaf = Vec::with_capacity(selected.len().checked_mul(2)?);
            for (bit, &head) in selected.iter().enumerate() {
                let active = mask & (1usize << bit) != 0;
                let b = I::p(f64::from(source.spec.first_bias[head]));
                let error = I::p(source.first_error[head]);
                let mut sign = Vec::with_capacity(d);
                let mut cap = Vec::with_capacity(d + 1);
                for input in 0..d {
                    if input & 63 == 0 && !live(deadline) {
                        return None;
                    }
                    let weight = f64::from(source.spec.first[input * h + head]);
                    if weight != 0.0 {
                        sign.push((input, I::p(if active { weight } else { -weight })));
                        if active {
                            cap.push((input, I::p(weight)));
                        }
                    }
                }
                // These rows mean t>=0 or t<=0, with CLOSED zero in both.
                leaf.push(OwnedRow {
                    coefficients: sign,
                    lower: if active { b.neg() } else { b },
                });
                cap.push((d + head, I::p(-1.0)));
                // Every branch keeps ROOT cast/dot/bias E1, including the
                // inactive nominal branch whose actual FP ReLU can be >0.
                leaf.push(OwnedRow {
                    coefficients: cap,
                    lower: if active {
                        b.neg().sub(error)?
                    } else {
                        error.neg()
                    },
                });
            }
            if leaf.iter().any(|row| !row.valid(n)) {
                return None;
            }
            exported_leaves.push(PhaseCoverLeaf {
                mask,
                rows: leaf
                    .iter()
                    .map(|row| export_row(row, deadline))
                    .collect::<Option<Vec<_>>>()?,
            });
            leaves.push(leaf);
        }
        let mut objective = vec![0.0; d];
        objective.extend(
            source
                .q
                .iter()
                .copied()
                .map(midpoint)
                .collect::<Option<Vec<_>>>()?,
        );
        let request = PhaseCoverLpRequest {
            protocol_version: PHASE_COVER_PROTOCOL_VERSION,
            selected_heads: selected.len(),
            bounds: boxes.iter().map(|value| [value.lo, value.hi]).collect(),
            objective,
            rows: rows
                .iter()
                .map(|row| export_row(row, deadline))
                .collect::<Option<Vec<_>>>()?,
            leaves: exported_leaves,
        };
        if !live(deadline) || !fp_qualified() || !request.validate() {
            return None;
        }
        Some(Self {
            source,
            boxes,
            rows,
            selected,
            leaves,
            request,
        })
    }

    pub(super) fn request(&self) -> &PhaseCoverLpRequest {
        &self.request
    }

    fn applies(&self, source: &Source, deadline: Instant) -> bool {
        std::ptr::eq(self.source, source)
            && Arc::ptr_eq(&self.source.spec, &source.spec)
            && self.boxes.len() == source.spec.inputs + source.spec.heads
            && self.leaves.len() == 1usize << self.selected.len()
            && live(deadline)
            && fp_qualified()
    }

    /// Any finite interpreted nonnegative multipliers give a bound, including
    /// entirely wrong advice. Every x/r residual is priced against its box.
    fn replay_leaf(
        &self,
        source: &Source,
        mask: usize,
        duals: &[f64],
        sign: f64,
        deadline: Instant,
    ) -> Option<f64> {
        if !self.applies(source, deadline)
            || (sign != 1.0 && sign != -1.0)
            || mask >= self.leaves.len()
            || duals.len() != self.rows.len().checked_add(self.leaves[mask].len())?
        {
            return None;
        }
        let mut residual = vec![I::p(0.0); source.spec.inputs];
        residual.extend(source.q.iter().copied());
        let mut constant = source.bias.sub(I::p(source.output_error))?;
        for (row, &raw) in self.rows.iter().chain(&self.leaves[mask]).zip(duals) {
            if !live(deadline) || !good(raw) || !row.valid(residual.len()) {
                return None;
            }
            let multiplier = I::p((sign * raw).max(0.0));
            if multiplier.lo == 0.0 {
                continue;
            }
            constant = constant.add(multiplier.mul(row.lower)?)?;
            for (at, &(index, coefficient)) in row.coefficients.iter().enumerate() {
                if at & 63 == 0 && !live(deadline) {
                    return None;
                }
                residual[index] = residual[index].sub(multiplier.mul(coefficient)?)?;
            }
        }
        for (index, (&coefficient, &box_)) in residual.iter().zip(&self.boxes).enumerate() {
            if index & 63 == 0 && !live(deadline) {
                return None;
            }
            constant = constant.add(coefficient.mul(box_)?)?;
        }
        (live(deadline) && fp_qualified() && constant.valid()).then_some(constant.lo)
    }

    fn native_duals(&self, mask: usize) -> Option<Vec<f64>> {
        let mut duals = vec![0.0; self.rows.len().checked_add(self.leaves.get(mask)?.len())?];
        for (bit, &head) in self.selected.iter().enumerate() {
            // A Source-owned proposal, not a special-case verdict. All q
            // enclosure and other-head residuals remain in replay_leaf.
            duals[self.rows.len() + 2 * bit + 1] = (-midpoint(self.source.q[head])?).max(0.0);
        }
        Some(duals)
    }

    fn replay_hint(
        &self,
        source: &Source,
        hint: &PhaseCoverLpHint,
        floors: &mut [f64],
        deadline: Instant,
    ) -> Option<Vec<usize>> {
        if !self.applies(source, deadline)
            || !hint.validate(&self.request)
            || floors.len() != self.leaves.len()
            || floors.iter().any(|&floor| !good(floor))
        {
            return None;
        }
        let mut replayed = Vec::with_capacity(hint.leaves.len());
        for leaf in &hint.leaves {
            let mut accepted = false;
            for sign in [1.0, -1.0] {
                if let Some(bound) =
                    self.replay_leaf(source, leaf.mask, &leaf.row_duals, sign, deadline)
                {
                    floors[leaf.mask] = floors[leaf.mask].max(bound);
                    accepted = true;
                }
            }
            if accepted {
                replayed.push(leaf.mask);
            }
        }
        (self.applies(source, deadline)).then_some(replayed)
    }
}

fn cover_lower(floors: &[f64]) -> Option<f64> {
    (!floors.is_empty() && floors.iter().all(|&floor| good(floor)))
        .then(|| floors.iter().copied().fold(f64::INFINITY, f64::min))
}

pub(super) fn refine<F>(
    source: &Source,
    before: f64,
    policy: PhaseCoverPolicy,
    parent_slice: Instant,
    authority: Instant,
    provider: &F,
) -> Refinement
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    refine_with_prefix(
        source,
        before,
        policy,
        None,
        parent_slice,
        authority,
        provider,
    )
}

pub(super) fn refine_with_prefix<'a, F>(
    source: &'a Source,
    before: f64,
    policy: PhaseCoverPolicy,
    prefix: Option<triangle::PhasePrefix<'a>>,
    parent_slice: Instant,
    authority: Instant,
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
        if !good(before) || !live(parent_slice) || !live(authority) || !fp_qualified() {
            return None;
        }
        let deadline = started
            .checked_add(LOCAL_SLICE)?
            .min(parent_slice)
            .min(authority);
        let call_deadline = deadline.checked_sub(REPLAY_RESERVE)?;
        let model = match prefix {
            Some(prefix) => Model::build_with_prefix(source, policy, Some(prefix), call_deadline),
            None => Model::build(source, policy, call_deadline),
        }?;
        report.selected_heads = model.selected.len();
        report.leaves = model.leaves.len();
        // The entire frontier exists before a single conditional certificate
        // is replayed. An unresolved or missing sibling retains this floor.
        let mut floors = vec![before; model.leaves.len()];
        let mut covered = vec![false; model.leaves.len()];
        for mask in 0..model.leaves.len() {
            if !model.applies(source, deadline) {
                return None;
            }
            let duals = model.native_duals(mask)?;
            if let Some(bound) = model.replay_leaf(source, mask, &duals, 1.0, deadline) {
                floors[mask] = floors[mask].max(bound);
                covered[mask] = true;
                report.native_replayed += 1;
            }
        }
        let native_lower = cover_lower(&floors)?;
        if native_lower <= 0.0 && live(call_deadline) {
            let proposed = (|| {
                let provider = std::panic::catch_unwind(std::panic::AssertUnwindSafe(provider))
                    .ok()
                    .flatten()?;
                let supports = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    provider.supports_phase_cover_advice()
                }))
                .ok()?;
                if !supports || !live(call_deadline) {
                    return None;
                }
                report.calls += 1;
                let hint = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    provider.propose_phase_cover(model.request(), call_deadline)
                }))
                .ok()
                .flatten()?;
                (live(call_deadline) && hint.validate(model.request())).then_some(hint)
            })();
            if let Some(hint) = proposed {
                // Stage the whole batch: malformed/expired arithmetic cannot
                // leave a partially committed new frontier behind.
                let mut proposed_floors = floors.clone();
                if let Some(replayed) =
                    model.replay_hint(source, &hint, &mut proposed_floors, deadline)
                {
                    floors = proposed_floors;
                    report.ay_replayed = replayed.len();
                    for mask in replayed {
                        covered[mask] = true;
                    }
                }
            }
        }
        let proposed = cover_lower(&floors)?;
        report.replayed = covered.into_iter().filter(|&value| value).count();
        // This scalar is unconditional. No leaf row/form/recipe is stored in
        // the generic input-box continuation or exposed as foreign authority.
        report.lower = commit_triangle_best(before, proposed, deadline, authority);
        Some(())
    })();
    report.elapsed_seconds = started.elapsed().as_secs_f64();
    report
}

#[cfg(test)]
#[path = "input_head_phase_cover_tests.rs"]
mod tests;
