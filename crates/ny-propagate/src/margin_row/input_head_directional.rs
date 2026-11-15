// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Independent directional support certificates, intersected in one root LP.
//!
//! For exact dyadic tau>=0, sum tau*ReLU(Wx+b+e) is the maximum of
//! ALL algebraic subset affine forms. Exact finite eta additionally supports
//! input-affine rows eta*x-sum(tau*r), independently of the margin objective. Each form is contracted against every
//! original input and E1 ball. These are algebraic masks, not nominal-sign
//! branches. Neither the output bias nor Eout belongs in this geometry proof.
//! Certification cannot read the LP prefix or any other candidate: circular
//! dependencies are excluded by construction. Only the final original-margin
//! replay uses the intersection, bias, signed q enclosures and Eout.

use super::*;
use ny_core::head_functional_advice::HeadFunctionalLpHint;
type OwnedRow = triangle::OwnedRow;

pub(super) const MAX_CUTS: usize = 4;
const MAX_POOL: usize = 32;
const MAX_SUPPORT: usize = 8;
const MIN_SUPPORT: usize = 6;
const MAX_CANDIDATES: usize = 32;
const MAX_MASKS: usize = 8192;
const MAX_ROUNDS: usize = 3;
const MAX_ATTEMPTS: usize = 32;
const MAX_CELLS: usize = 2_000_000;
const REPLAY_RESERVE: Duration = Duration::from_millis(75);

#[derive(Clone, Debug)]
struct Direction {
    heads: Vec<(usize, f64)>,
}

impl Direction {
    fn valid(&self, source: &Source) -> bool {
        (MIN_SUPPORT..=MAX_SUPPORT).contains(&self.heads.len())
            && self.heads.iter().enumerate().all(|(at, &(head, tau))| {
                head < source.spec.heads
                    && [0.25, 0.5, 1.0].contains(&tau)
                    && (at == 0 || self.heads[at - 1].0 < head)
            })
    }
}

/// Only the independent complete-mask producer can construct this object. It carries no helper endpoint.
pub(super) struct CertifiedDirection<'a> {
    source: &'a Source,
    direction: Direction,
    eta: Vec<(usize, f64)>,
    lower: f64,
    deadline: Instant,
    geometry: GeometryKind,
}

enum GeometryKind {
    NegativeOnly,
    Signed(signed::PositiveProof),
}

impl CertifiedDirection<'_> {
    pub(super) fn owned_row(&self, source: &Source, deadline: Instant) -> Option<OwnedRow> {
        if !std::ptr::eq(self.source, source)
            || !Arc::ptr_eq(&self.source.spec, &source.spec)
            || !live(self.deadline)
            || !live(deadline)
            || !fp_qualified()
            || !good(self.lower)
            || !match &self.geometry {
                GeometryKind::NegativeOnly => self.direction.valid(source),
                GeometryKind::Signed(proof) => proof.valid(source, &self.direction, &self.eta),
            }
            || !valid_eta(source, &self.eta)
        {
            return None;
        }
        let mut coefficients: Vec<_> = self.eta.iter().map(|&(j, a)| (j, I::p(a))).collect();
        if let GeometryKind::Signed(proof) = &self.geometry {
            coefficients.extend(
                proof
                    .terms()
                    .iter()
                    .map(|&(head, alpha)| (source.spec.inputs + head, I::p(alpha))),
            );
        }
        coefficients.extend(
            self.direction
                .heads
                .iter()
                .map(|&(head, tau)| (source.spec.inputs + head, I::p(-tau))),
        );
        if matches!(&self.geometry, GeometryKind::Signed(_)) {
            coefficients.sort_by_key(|&(j, _)| j);
        }
        let row = OwnedRow {
            coefficients,
            lower: I::p(self.lower),
        };
        row.valid(source.spec.inputs.checked_add(source.spec.heads)?)
            .then_some(row)
    }

    fn work_terms(&self, inputs: usize) -> Option<usize> {
        let positive = match &self.geometry {
            GeometryKind::NegativeOnly => return inputs.checked_add(MAX_SUPPORT),
            GeometryKind::Signed(proof) => proof.terms().len(),
        };
        inputs
            .checked_add(self.direction.heads.len())?
            .checked_add(positive)
    }
}

#[derive(Default)]
pub(super) struct Refinement {
    pub(super) lower: f64,
    pub(super) candidates: usize,
    pub(super) certified: usize,
    pub(super) cuts: usize,
    pub(super) masks: usize,
    pub(super) work_cells: usize,
    pub(super) calls: usize,
    pub(super) native_replayed: usize,
    pub(super) ay_replayed: usize,
    pub(super) elapsed_seconds: f64,
    pub(super) attempted: usize,
    pub(super) affine_cuts: usize,
    pub(super) rounds: Vec<RoundReport>,
}

#[derive(Default, Debug)]
pub(super) struct RoundReport {
    pub(super) round: usize,
    pub(super) candidates: usize,
    pub(super) attempted: usize,
    pub(super) certified: usize,
    pub(super) cuts: usize,
    pub(super) masks: usize,
    pub(super) work_cells: usize,
    pub(super) calls: usize,
    pub(super) native_replayed: usize,
    pub(super) ay_replayed: usize,
    pub(super) before: f64,
    pub(super) after: f64,
    pub(super) elapsed_seconds: f64,
}

struct Budget {
    deadline: Instant,
    cells: usize,
    masks: usize,
}

/// Born once from the unchanged root-stage authority, never from a later clock.
#[derive(Clone, Copy)]
struct PublicationAuthority {
    deadline: Instant,
}

impl Budget {
    fn publication_charge(&mut self, cells: usize, authority: PublicationAuthority) -> Option<()> {
        if !live(authority.deadline) || !fp_qualified() {
            return None;
        }
        let next = self.cells.checked_add(cells)?;
        if next > MAX_CELLS {
            return None;
        }
        self.cells = next;
        Some(())
    }
    fn charge(&mut self, cells: usize) -> Option<()> {
        if !live(self.deadline) || !fp_qualified() {
            return None;
        }
        let next = self.cells.checked_add(cells)?;
        if next > MAX_CELLS {
            return None;
        }
        self.cells = next;
        Some(())
    }

    fn complete_mask(&mut self) -> Option<()> {
        self.masks = self.masks.checked_add(1)?;
        #[cfg(test)]
        if EXPIRE_AFTER_MASK.with(|fault| fault.get() == Some(self.masks)) {
            self.deadline = Instant::now().checked_sub(Duration::from_millis(1))?;
        }
        self.charge(0)
    }
}

/// The proof has no prefix/other-cut argument. Every candidate depends ONLY
/// on this immutable Source, its original box, and a complete algebraic cover.
#[cfg(test)]
fn certify<'a>(
    source: &'a Source,
    direction: Direction,
    budget: &mut Budget,
) -> Option<CertifiedDirection<'a>> {
    certify_affine(source, direction, vec![], budget)
}

fn valid_eta(source: &Source, eta: &[(usize, f64)]) -> bool {
    eta.iter().enumerate().all(|(at, &(j, a))| {
        j < source.spec.inputs && good(a) && a != 0.0 && (at == 0 || eta[at - 1].0 < j)
    })
}

fn certify_affine<'a>(
    source: &'a Source,
    direction: Direction,
    eta: Vec<(usize, f64)>,
    budget: &mut Budget,
) -> Option<CertifiedDirection<'a>> {
    budget.charge(0)?;
    let d = source.spec.inputs;
    let h = source.spec.heads;
    if !direction.valid(source)
        || !valid_eta(source, &eta)
        || source.lo.len() != d
        || source.hi.len() != d
        || source.spec.first.len() != d.checked_mul(h)?
        || source.spec.first_bias.len() != h
        || source.first_error.len() != h
        || source.y.len() != h
    {
        return None;
    }
    let count = 1usize.checked_shl(direction.heads.len() as u32)?;
    if budget.masks.checked_add(count)? > MAX_MASKS {
        return None;
    }
    let boxes: Vec<_> = source
        .lo
        .iter()
        .zip(&source.hi)
        .map(|(&lo, &hi)| I { lo, hi })
        .collect();
    // eta may have positive minimum on this box. Zero is NOT a valid
    // initializer for the exact minimum over the complete mask family.
    let mut lower = f64::INFINITY;
    for mask in 0..count {
        budget.charge(d)?;
        let mut form = Affine::zero(d);
        for &(j, a) in &eta {
            budget.charge(1)?;
            form.a[j] = I::p(a);
        }
        for (bit, &(head, tau)) in direction.heads.iter().enumerate() {
            if mask & (1usize << bit) == 0 {
                continue;
            }
            let error = source.first_error[head];
            if !good(error) || error < 0.0 {
                return None;
            }
            budget.charge(2)?;
            let negative_tau = I::p(-tau);
            // Negative seed lower support equals the negative upper support.
            // -tau*e >= -tau*E1; no nominal cancellation of error is allowed.
            form.b = form
                .b
                .add(negative_tau.mul(I::p(f64::from(source.spec.first_bias[head])))?)?
                .add(negative_tau.mul(I::p(error))?)?;
            for input in 0..d {
                if input & 31 == 0 {
                    budget.charge(0)?;
                }
                budget.charge(1)?;
                form.a[input] = form.a[input]
                    .add(negative_tau.mul(I::p(f64::from(source.spec.first[input * h + head])))?)?;
            }
        }
        budget.charge(d)?;
        lower = lower.min(form.lower(&boxes, budget.deadline)?);
        budget.complete_mask()?;
    }
    budget.charge(0)?;
    let mut box_floor = I::p(0.0);
    for &(j, a) in &eta {
        budget.charge(1)?;
        box_floor = box_floor.add(I::p(a).mul(boxes[j])?)?;
    }
    for &(head, tau) in &direction.heads {
        budget.charge(1)?;
        box_floor = box_floor.add(I::p(-tau).mul(I::p(source.y[head].hi.max(0.0)))?)?;
    }
    // A direction has its OWN box floor; a scalar margin floor, output bias,
    // or another direction's endpoint is never an initializer for this cut.
    if lower <= box_floor.lo {
        return None;
    }
    good(lower).then_some(CertifiedDirection {
        source,
        direction,
        eta,
        lower,
        deadline: budget.deadline,
        geometry: GeometryKind::NegativeOnly,
    })
}

#[cfg(test)]
fn native_hint(
    model: &triangle::Model<'_>,
    directions: &[CertifiedDirection<'_>],
) -> Option<HeadFunctionalLpHint> {
    let request = model.request();
    let first_cut = request.rows.len().checked_sub(directions.len())?;
    let mut residual = request.objective.clone();
    let mut row_duals = vec![0.0; request.rows.len()];
    for (at, direction) in directions.iter().enumerate() {
        let multiplier = direction
            .direction
            .heads
            .iter()
            .map(|&(head, tau)| (-residual[request.feature_count + head]).max(0.0) / tau)
            .fold(f64::INFINITY, f64::min);
        if !good(multiplier) {
            return None;
        }
        row_duals[first_cut + at] = multiplier;
        for &(head, tau) in &direction.direction.heads {
            residual[request.feature_count + head] += multiplier * tau;
        }
    }
    let hint = HeadFunctionalLpHint { row_duals };
    hint.validate(request).then_some(hint)
}

#[cfg(test)]
use feedback::geometry_debt;
pub(super) use feedback::refine_with_prefix;
#[path = "input_head_directional_allocated.rs"]
mod allocated;
#[path = "input_head_directional_feedback.rs"]
mod feedback;
#[path = "input_head_directional_native.rs"]
mod native;
#[path = "input_head_directional_search.rs"]
mod search;
#[path = "input_head_directional_signed.rs"]
mod signed;
#[path = "input_head_directional_signed_search.rs"]
mod signed_search;

#[cfg(test)]
fn proposals(source: &Source, budget: &mut Budget) -> Option<Vec<Direction>> {
    search::proposals(source, budget)
}

#[cfg(test)]
thread_local! {
    static EXPIRE_AFTER_MASK: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static EXPIRE_AFTER_ROUND: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[path = "input_head_directional_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "input_head_directional_affine_tests.rs"]
mod affine_tests;
#[cfg(test)]
#[path = "input_head_directional_allocated_retention_tests.rs"]
mod allocated_retention_tests;
#[cfg(test)]
#[path = "input_head_directional_allocated_tests.rs"]
mod allocated_tests;
#[cfg(test)]
#[path = "input_head_directional_feedback_tests.rs"]
mod feedback_tests;
#[cfg(test)]
#[path = "input_head_directional_gray_tests.rs"]
mod gray_tests;
