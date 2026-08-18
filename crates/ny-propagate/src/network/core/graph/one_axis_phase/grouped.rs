// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verdict-neutral grouped reuse for exact one-axis phase covers.
//!
//! A context is the exact `(input_shape, free_axis, fixed coordinates)` tuple;
//! the value stored at `fixed_inputs[free_axis]` is deliberately ignored.
//! Members of one context share a single phase cover over the convex hull of
//! their one-axis intervals.  Constraint peeling and observation construction
//! remain member-local.
//!
//! Distinct contexts are visited by a deterministic nearest-Hamming walk.  The
//! axis-static and linear-static caches are reset at every context boundary.
//! Only the exact input/output cache for dense linear nodes survives, allowing
//! [`linear_tensor_from_phase_delta`] to apply sparse exact cross-context
//! updates.  Replay starts with independent empty caches and reconstructs the
//! same traversal and every phase cell.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::*;

pub const ONE_AXIS_GROUPED_PHASE_CERTIFICATE_VERSION: &str = "ny.one-axis-grouped-phase.m3.v1";

/// Limits that bound the complete grouped transaction.
///
/// `phase.max_exact_operations` is a global budget for the whole transaction,
/// while `phase.max_phase_cells` remains a per-context cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneAxisGroupedPhaseLimits {
    pub phase: OneAxisPhaseLimits,
    pub max_problems: usize,
    pub max_contexts: usize,
    pub max_total_constraints: usize,
    pub max_total_phase_cells: usize,
    pub max_hamming_coordinate_comparisons: usize,
}

impl Default for OneAxisGroupedPhaseLimits {
    fn default() -> Self {
        Self {
            phase: OneAxisPhaseLimits::default(),
            max_problems: 16_384,
            max_contexts: 512,
            max_total_constraints: 65_536,
            max_total_phase_cells: 262_144,
            max_hamming_coordinate_comparisons: 100_000_000,
        }
    }
}

/// One exact fixed-input context in deterministic traversal order.
#[derive(Clone, Debug, PartialEq)]
pub struct OneAxisGroupedContextCertificate {
    /// Domain-separated digest of the exact context.  Exact equality, rather
    /// than digest equality, is used while grouping.
    pub context_digest: [u8; 32],
    /// Indices into the caller-supplied problem slice.
    pub member_indices: Vec<usize>,
    /// Convex hull of every member interval in this exact context.
    pub lower: OneAxisRational,
    pub upper: OneAxisRational,
    pub cells: Vec<OneAxisPhaseCellCertificate>,
    pub wrapper: OneAxisWrapperEnclosure,
}

/// Constraint-local material derived from a grouped context cover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisGroupedMemberCertificate {
    /// Index into `OneAxisGroupedPhaseCertificate::contexts`.
    pub context: usize,
    pub peeled_constraints: Vec<OneAxisPeeledConstraint>,
    pub observation: OneAxisPhaseObservation,
}

/// Atomic, verdict-neutral certificate for a supplied problem list.
#[derive(Clone, Debug, PartialEq)]
pub struct OneAxisGroupedPhaseCertificate {
    pub version: &'static str,
    pub verdict_authority: bool,
    pub grouped_problem_digest: [u8; 32],
    pub graph_digest: [u8; 32],
    pub contexts: Vec<OneAxisGroupedContextCertificate>,
    /// Aligned one-for-one with the caller-supplied problem slice.
    pub members: Vec<OneAxisGroupedMemberCertificate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OneAxisGroupedPhaseAttempt {
    pub certificate: Option<OneAxisGroupedPhaseCertificate>,
    pub decline: Option<OneAxisPhaseDecline>,
    /// Fully completed contexts before success or fail-closed refusal.
    pub contexts_examined: usize,
    /// Phase cells belonging to fully completed contexts.
    pub phase_cells_examined: usize,
    pub exact_operations: usize,
    /// Exact dense updates whose retained input came from an earlier context.
    pub cross_context_sparse_linear_updates: usize,
    pub hamming_coordinate_comparisons: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisGroupedReplayResult {
    pub accepted: bool,
    pub observations: Option<Vec<OneAxisPhaseObservation>>,
    pub decline: Option<OneAxisPhaseDecline>,
    pub contexts_replayed: usize,
    pub phase_cells_replayed: usize,
    pub exact_operations: usize,
    pub cross_context_sparse_linear_updates: usize,
    pub hamming_coordinate_comparisons: usize,
}

#[derive(Clone, Debug)]
struct ContextPlan {
    representative: usize,
    members: Vec<usize>,
    lower: BigRational,
    upper: BigRational,
    digest: [u8; 32],
}

struct GroupPlan {
    contexts: Vec<ContextPlan>,
    traversal: Vec<usize>,
    grouped_problem_digest: [u8; 32],
}

struct PlanningWork {
    deadline: Instant,
    work_items: usize,
    hamming_coordinate_comparisons: usize,
}

impl PlanningWork {
    fn new(deadline: Instant) -> Self {
        Self {
            deadline,
            work_items: 0,
            hamming_coordinate_comparisons: 0,
        }
    }

    fn poll(&mut self) -> Result<(), OneAxisPhaseDeclineReason> {
        self.work_items = self.work_items.wrapping_add(1);
        if self.work_items.is_multiple_of(1024) && Instant::now() >= self.deadline {
            Err(OneAxisPhaseDeclineReason::Deadline)
        } else {
            Ok(())
        }
    }

    fn charge_hamming(
        &mut self,
        limits: &OneAxisGroupedPhaseLimits,
    ) -> Result<(), OneAxisPhaseDeclineReason> {
        self.poll()?;
        self.hamming_coordinate_comparisons = self
            .hamming_coordinate_comparisons
            .checked_add(1)
            .ok_or(OneAxisPhaseDeclineReason::HammingTraversalLimit)?;
        if self.hamming_coordinate_comparisons > limits.max_hamming_coordinate_comparisons {
            return Err(OneAxisPhaseDeclineReason::HammingTraversalLimit);
        }
        Ok(())
    }

    fn check_deadline(&self) -> Result<(), OneAxisPhaseDeclineReason> {
        if Instant::now() >= self.deadline {
            Err(OneAxisPhaseDeclineReason::Deadline)
        } else {
            Ok(())
        }
    }
}

fn context_digest(
    problem: &OneAxisExactProblem,
    work: &mut PlanningWork,
) -> Result<[u8; 32], OneAxisPhaseDeclineReason> {
    work.check_deadline()?;
    let mut hasher = Sha256::new();
    hasher.update(b"ny.one-axis-grouped-context.m3.v1");
    hash_usize(&mut hasher, problem.input_shape.len());
    for &dimension in &problem.input_shape {
        work.poll()?;
        hash_usize(&mut hasher, dimension);
    }
    hash_usize(&mut hasher, problem.free_axis);
    hash_usize(&mut hasher, problem.fixed_inputs.len());
    for (index, value) in problem.fixed_inputs.iter().enumerate() {
        work.poll()?;
        if index != problem.free_axis {
            hash_usize(&mut hasher, index);
            hash_rational(&mut hasher, &value.0);
        }
    }
    Ok(hasher.finalize().into())
}

fn same_context(
    left: &OneAxisExactProblem,
    right: &OneAxisExactProblem,
    work: &mut PlanningWork,
) -> Result<bool, OneAxisPhaseDeclineReason> {
    if left.input_shape != right.input_shape || left.free_axis != right.free_axis {
        return Ok(false);
    }
    for index in 0..left.fixed_inputs.len() {
        work.poll()?;
        if index != left.free_axis && left.fixed_inputs[index] != right.fixed_inputs[index] {
            return Ok(false);
        }
    }
    Ok(true)
}

fn hamming_distance(
    left: &OneAxisExactProblem,
    right: &OneAxisExactProblem,
    limits: &OneAxisGroupedPhaseLimits,
    work: &mut PlanningWork,
) -> Result<usize, OneAxisPhaseDeclineReason> {
    if left.input_shape != right.input_shape || left.fixed_inputs.len() != right.fixed_inputs.len()
    {
        return Err(OneAxisPhaseDeclineReason::InvalidProblem);
    }
    let mut distance = 0usize;
    for index in 0..left.fixed_inputs.len() {
        work.charge_hamming(limits)?;
        let differs = match (index == left.free_axis, index == right.free_axis) {
            (true, true) => false,
            (true, false) | (false, true) => true,
            (false, false) => left.fixed_inputs[index] != right.fixed_inputs[index],
        };
        distance = distance
            .checked_add(usize::from(differs))
            .ok_or(OneAxisPhaseDeclineReason::HammingTraversalLimit)?;
    }
    Ok(distance)
}

fn context_tie_cmp(left_index: usize, right_index: usize, contexts: &[ContextPlan]) -> Ordering {
    contexts[left_index]
        .digest
        .cmp(&contexts[right_index].digest)
        .then_with(|| {
            contexts[left_index]
                .representative
                .cmp(&contexts[right_index].representative)
        })
}

fn deterministic_hamming_traversal(
    contexts: &[ContextPlan],
    problems: &[OneAxisExactProblem],
    limits: &OneAxisGroupedPhaseLimits,
    work: &mut PlanningWork,
) -> Result<Vec<usize>, OneAxisPhaseDeclineReason> {
    if contexts.is_empty() {
        return Err(OneAxisPhaseDeclineReason::InvalidProblem);
    }
    let first = (0..contexts.len())
        .min_by(|&left, &right| context_tie_cmp(left, right, contexts))
        .ok_or(OneAxisPhaseDeclineReason::InvalidProblem)?;
    let mut visited = vec![false; contexts.len()];
    visited[first] = true;
    let mut traversal = Vec::with_capacity(contexts.len());
    traversal.push(first);
    while traversal.len() < contexts.len() {
        work.check_deadline()?;
        let current = *traversal
            .last()
            .ok_or(OneAxisPhaseDeclineReason::InvalidProblem)?;
        let current_problem = &problems[contexts[current].representative];
        let mut best = None::<(usize, usize)>;
        for candidate in 0..contexts.len() {
            if visited[candidate] {
                continue;
            }
            let distance = hamming_distance(
                current_problem,
                &problems[contexts[candidate].representative],
                limits,
                work,
            )?;
            let replace = match best {
                None => true,
                Some((best_distance, best_index)) => {
                    distance < best_distance
                        || (distance == best_distance
                            && context_tie_cmp(candidate, best_index, contexts) == Ordering::Less)
                }
            };
            if replace {
                best = Some((distance, candidate));
            }
        }
        let next = best
            .map(|(_, index)| index)
            .ok_or(OneAxisPhaseDeclineReason::InvalidProblem)?;
        visited[next] = true;
        traversal.push(next);
    }
    Ok(traversal)
}

fn grouped_problem_digest(
    problems: &[OneAxisExactProblem],
    contexts: &[ContextPlan],
    problem_contexts: &[usize],
    work: &mut PlanningWork,
) -> Result<[u8; 32], OneAxisPhaseDeclineReason> {
    work.check_deadline()?;
    let mut hasher = Sha256::new();
    hasher.update(ONE_AXIS_GROUPED_PHASE_CERTIFICATE_VERSION.as_bytes());
    hash_usize(&mut hasher, problems.len());
    for (index, problem) in problems.iter().enumerate() {
        work.poll()?;
        let context = problem_contexts
            .get(index)
            .and_then(|&context| contexts.get(context))
            .ok_or(OneAxisPhaseDeclineReason::InvalidProblem)?;
        hasher.update(context.digest);
        hash_rational(&mut hasher, &problem.lower.0);
        hash_rational(&mut hasher, &problem.upper.0);
        hash_usize(&mut hasher, problem.constraints.len());
        for constraint in &problem.constraints {
            work.poll()?;
            hasher.update([match constraint.relation {
                OneAxisConstraintRelation::LessEqual => 0,
                OneAxisConstraintRelation::GreaterEqual => 1,
            }]);
            hash_rational(&mut hasher, &constraint.bound.0);
        }
    }
    Ok(hasher.finalize().into())
}

fn build_group_plan(
    problems: &[OneAxisExactProblem],
    limits: &OneAxisGroupedPhaseLimits,
    work: &mut PlanningWork,
) -> Result<GroupPlan, OneAxisPhaseDeclineReason> {
    work.check_deadline()?;
    if problems.is_empty() {
        return Err(OneAxisPhaseDeclineReason::InvalidProblem);
    }
    if problems.len() > limits.max_problems {
        return Err(OneAxisPhaseDeclineReason::ProblemLimit);
    }
    let input_shape = &problems[0].input_shape;
    let mut total_constraints = 0usize;
    for problem in problems {
        validate_problem(problem, &limits.phase, work.deadline)?;
        if &problem.input_shape != input_shape {
            return Err(OneAxisPhaseDeclineReason::InvalidProblem);
        }
        total_constraints = total_constraints
            .checked_add(problem.constraints.len())
            .ok_or(OneAxisPhaseDeclineReason::TotalConstraintLimit)?;
        if total_constraints > limits.max_total_constraints {
            return Err(OneAxisPhaseDeclineReason::TotalConstraintLimit);
        }
    }

    let mut contexts = Vec::<ContextPlan>::new();
    let mut buckets = HashMap::<[u8; 32], Vec<usize>>::new();
    let mut problem_contexts = Vec::with_capacity(problems.len());
    for (problem_index, problem) in problems.iter().enumerate() {
        let digest = context_digest(problem, work)?;
        let mut matching = None;
        if let Some(bucket) = buckets.get(&digest) {
            for &candidate in bucket {
                if same_context(problem, &problems[contexts[candidate].representative], work)? {
                    matching = Some(candidate);
                    break;
                }
            }
        }
        let context_index = match matching {
            Some(context_index) => context_index,
            None => {
                if contexts.len() >= limits.max_contexts {
                    return Err(OneAxisPhaseDeclineReason::ContextLimit);
                }
                let context_index = contexts.len();
                contexts.push(ContextPlan {
                    representative: problem_index,
                    members: Vec::new(),
                    lower: problem.lower.0.clone(),
                    upper: problem.upper.0.clone(),
                    digest,
                });
                buckets.entry(digest).or_default().push(context_index);
                context_index
            }
        };
        let context = &mut contexts[context_index];
        context.members.push(problem_index);
        if problem.lower.0 < context.lower {
            context.lower = problem.lower.0.clone();
        }
        if problem.upper.0 > context.upper {
            context.upper = problem.upper.0.clone();
        }
        problem_contexts.push(context_index);
    }

    let traversal = deterministic_hamming_traversal(&contexts, problems, limits, work)?;
    let grouped_problem_digest =
        grouped_problem_digest(problems, &contexts, &problem_contexts, work)?;
    Ok(GroupPlan {
        contexts,
        traversal,
        grouped_problem_digest,
    })
}

fn union_problem(plan: &ContextPlan, problems: &[OneAxisExactProblem]) -> OneAxisExactProblem {
    let mut problem = problems[plan.representative].clone();
    problem.lower = OneAxisRational(plan.lower.clone());
    problem.upper = OneAxisRational(plan.upper.clone());
    // This slot has no semantics, but canonicalizing it avoids retaining a
    // representative-member artifact in the context-local evaluation input.
    problem.fixed_inputs[problem.free_axis] = problem.lower.clone();
    problem
}

fn grouped_attempt_decline(
    reason: OneAxisPhaseDecline,
    contexts_examined: usize,
    phase_cells_examined: usize,
    exact_operations: usize,
    cross_context_sparse_linear_updates: usize,
    hamming_coordinate_comparisons: usize,
) -> OneAxisGroupedPhaseAttempt {
    OneAxisGroupedPhaseAttempt {
        certificate: None,
        decline: Some(reason),
        contexts_examined,
        phase_cells_examined,
        exact_operations,
        cross_context_sparse_linear_updates,
        hamming_coordinate_comparisons,
    }
}

fn generate_context_cover(
    graph: &GraphNetwork,
    problem: &OneAxisExactProblem,
    limits: &OneAxisGroupedPhaseLimits,
    completed_cells: usize,
    linear_phase_cache: &mut HashMap<String, LinearPhaseCache>,
    budget: &mut ExactBudget<'_>,
) -> Result<(Vec<OneAxisPhaseCellCertificate>, OneAxisWrapperEnclosure), OneAxisPhaseDecline> {
    let mut cursor = problem.lower.0.clone();
    let mut cells = Vec::new();
    let mut global_wrapper = None;
    // These values depend on the exact fixed context and must never cross its
    // boundary.  Only the exact input/output linear cache above is shared.
    let mut static_cache = HashMap::new();
    let mut linear_static_cache = HashMap::new();

    loop {
        if cells.len() >= limits.phase.max_phase_cells {
            return Err(decline(OneAxisPhaseDeclineReason::PhaseCellLimit, None));
        }
        if completed_cells
            .checked_add(cells.len())
            .is_none_or(|count| count >= limits.max_total_phase_cells)
        {
            return Err(decline(
                OneAxisPhaseDeclineReason::TotalPhaseCellLimit,
                None,
            ));
        }
        let evaluation = evaluate_phase(
            graph,
            problem,
            &cursor,
            &mut static_cache,
            &mut linear_static_cache,
            linear_phase_cache,
            budget,
        )?;
        let current_wrapper = wrapper_enclosure(&evaluation.wrapper);
        match global_wrapper {
            Some(expected) if !same_wrapper(expected, current_wrapper) => {
                return Err(decline(
                    OneAxisPhaseDeclineReason::ReplayMismatch,
                    Some(graph.output_name()),
                ));
            }
            None => global_wrapper = Some(current_wrapper),
            _ => {}
        }
        let endpoint = evaluation.endpoint.clone();
        cells.push(certificate_cell(&cursor, &evaluation));
        if cursor == problem.upper.0 || endpoint == problem.upper.0 {
            break;
        }
        cursor = endpoint;
    }
    let wrapper = global_wrapper
        .ok_or_else(|| decline(OneAxisPhaseDeclineReason::CertificateMalformed, None))?;
    Ok((cells, wrapper))
}

fn observation_from_context_cells(
    cells: &[OneAxisPhaseCellCertificate],
    peeled: &[OneAxisPeeledConstraint],
    lower: &BigRational,
    upper: &BigRational,
    budget: &mut ExactBudget<'_>,
) -> Option<OneAxisPhaseObservation> {
    let mut necessary_nonempty = false;
    let mut witness = None;
    let mut covered = false;
    for cell in cells {
        if !budget.poll_work() {
            return None;
        }
        let clipped_lower = if cell.lower.0 < *lower {
            lower.clone()
        } else {
            cell.lower.0.clone()
        };
        let clipped_upper = if cell.upper.0 > *upper {
            upper.clone()
        } else {
            cell.upper.0.clone()
        };
        if clipped_lower > clipped_upper {
            continue;
        }
        covered = true;
        let affine = ExactAffine {
            slope: cell.core.slope.0.clone(),
            bias: cell.core.bias.0.clone(),
            depends: true,
        };
        let initial = ExactRegion {
            lower: clipped_lower,
            upper: clipped_upper,
        };
        let mut necessary = initial.clone();
        let mut necessary_ok = true;
        for constraint in peeled {
            if !necessary.apply(&affine, &constraint.necessary, budget)? {
                necessary_ok = false;
                break;
            }
        }
        necessary_nonempty |= necessary_ok;

        if witness.is_none() {
            let mut sufficient = initial;
            let mut sufficient_ok = true;
            for constraint in peeled {
                if !sufficient.apply(&affine, &constraint.sufficient, budget)? {
                    sufficient_ok = false;
                    break;
                }
            }
            if sufficient_ok {
                witness = Some(OneAxisRational(sufficient.lower));
            }
        }
    }
    covered.then_some(match witness {
        Some(free_value) => OneAxisPhaseObservation::ExactWitness { free_value },
        None if !necessary_nonempty => OneAxisPhaseObservation::CertifiedEmpty,
        None => OneAxisPhaseObservation::Inconclusive,
    })
}

fn derive_member(
    problem: &OneAxisExactProblem,
    context_index: usize,
    cells: &[OneAxisPhaseCellCertificate],
    wrapper: OneAxisWrapperEnclosure,
    limits: &OneAxisGroupedPhaseLimits,
    budget: &mut ExactBudget<'_>,
) -> Result<OneAxisGroupedMemberCertificate, OneAxisPhaseDecline> {
    let wrapper_value = WrapperValue {
        offset: DirectedInterval {
            lower: wrapper.offset_lower,
            upper: wrapper.offset_upper,
        },
        sign: wrapper.sigmoid_sign,
        core: ExactAffine::constant(BigRational::zero()),
    };
    let peeled_constraints = peel_constraints(problem, &wrapper_value, budget.deadline)
        .ok_or_else(|| {
            decline(
                if Instant::now() >= budget.deadline {
                    OneAxisPhaseDeclineReason::Deadline
                } else {
                    OneAxisPhaseDeclineReason::DirectedArithmetic
                },
                None,
            )
        })?;
    if peeled_constraints.iter().any(|constraint| {
        !guard_within_limit(&constraint.necessary, &limits.phase)
            || !guard_within_limit(&constraint.sufficient, &limits.phase)
    }) {
        return Err(decline(OneAxisPhaseDeclineReason::RationalBitLimit, None));
    }
    let observation = observation_from_context_cells(
        cells,
        &peeled_constraints,
        &problem.lower.0,
        &problem.upper.0,
        budget,
    )
    .ok_or_else(|| {
        decline(
            budget
                .failure
                .unwrap_or(OneAxisPhaseDeclineReason::ReplayMismatch),
            None,
        )
    })?;
    Ok(OneAxisGroupedMemberCertificate {
        context: context_index,
        peeled_constraints,
        observation,
    })
}

fn grouped_certificate_within_limits(
    certificate: &OneAxisGroupedPhaseCertificate,
    limits: &OneAxisGroupedPhaseLimits,
    deadline: Instant,
) -> bool {
    if Instant::now() >= deadline
        || certificate.contexts.is_empty()
        || certificate.contexts.len() > limits.max_contexts
        || certificate.members.is_empty()
        || certificate.members.len() > limits.max_problems
    {
        return false;
    }
    let mut total_cells = 0usize;
    let mut total_memberships = 0usize;
    for (context_index, context) in certificate.contexts.iter().enumerate() {
        if context_index.is_multiple_of(64) && Instant::now() >= deadline {
            return false;
        }
        if context.member_indices.is_empty()
            || context.member_indices.len() > limits.max_problems
            || context.cells.is_empty()
            || context.cells.len() > limits.phase.max_phase_cells
            || !public_rational_within_limit(&context.lower, &limits.phase)
            || !public_rational_within_limit(&context.upper, &limits.phase)
            || context.lower.0 > context.upper.0
            || !context.wrapper.offset_lower.is_finite()
            || !context.wrapper.offset_upper.is_finite()
            || context.wrapper.offset_lower > context.wrapper.offset_upper
            || !matches!(context.wrapper.sigmoid_sign, -1 | 1)
        {
            return false;
        }
        let Some(updated_memberships) = total_memberships.checked_add(context.member_indices.len())
        else {
            return false;
        };
        total_memberships = updated_memberships;
        if total_memberships > limits.max_problems {
            return false;
        }
        let Some(updated_total) = total_cells.checked_add(context.cells.len()) else {
            return false;
        };
        total_cells = updated_total;
        if total_cells > limits.max_total_phase_cells {
            return false;
        }
        for (cell_index, cell) in context.cells.iter().enumerate() {
            if cell_index.is_multiple_of(64) && Instant::now() >= deadline {
                return false;
            }
            if cell.relu_scalars > limits.phase.max_relu_scalars_per_phase
                || !public_rational_within_limit(&cell.lower, &limits.phase)
                || !public_rational_within_limit(&cell.upper, &limits.phase)
                || !public_rational_within_limit(&cell.core.slope, &limits.phase)
                || !public_rational_within_limit(&cell.core.bias, &limits.phase)
                || cell.lower.0 > cell.upper.0
            {
                return false;
            }
        }
    }
    let mut total_constraints = 0usize;
    for (member_index, member) in certificate.members.iter().enumerate() {
        if member_index.is_multiple_of(64) && Instant::now() >= deadline {
            return false;
        }
        if member.context >= certificate.contexts.len()
            || member.peeled_constraints.len() > limits.phase.max_constraints
        {
            return false;
        }
        let Some(updated_total) = total_constraints.checked_add(member.peeled_constraints.len())
        else {
            return false;
        };
        total_constraints = updated_total;
        if total_constraints > limits.max_total_constraints {
            return false;
        }
        if member.peeled_constraints.iter().any(|constraint| {
            !guard_within_limit(&constraint.necessary, &limits.phase)
                || !guard_within_limit(&constraint.sufficient, &limits.phase)
        }) {
            return false;
        }
        if let OneAxisPhaseObservation::ExactWitness { free_value } = &member.observation {
            if !public_rational_within_limit(free_value, &limits.phase) {
                return false;
            }
        }
    }
    true
}

fn grouped_replay_reject(
    reason: OneAxisPhaseDecline,
    contexts_replayed: usize,
    phase_cells_replayed: usize,
    exact_operations: usize,
    cross_context_sparse_linear_updates: usize,
    hamming_coordinate_comparisons: usize,
) -> OneAxisGroupedReplayResult {
    OneAxisGroupedReplayResult {
        accepted: false,
        observations: None,
        decline: Some(reason),
        contexts_replayed,
        phase_cells_replayed,
        exact_operations,
        cross_context_sparse_linear_updates,
        hamming_coordinate_comparisons,
    }
}

impl GraphNetwork {
    /// Generate one union-interval phase cover per exact fixed-input context.
    ///
    /// This API is source-only and verdict-neutral.  It has no production
    /// verifier caller, and every successful certificate explicitly records
    /// `verdict_authority=false`.
    pub fn exact_grouped_one_axis_phase_certificate_until(
        &self,
        problems: &[OneAxisExactProblem],
        limits: OneAxisGroupedPhaseLimits,
        deadline: Instant,
    ) -> OneAxisGroupedPhaseAttempt {
        let mut planning = PlanningWork::new(deadline);
        let plan = match build_group_plan(problems, &limits, &mut planning) {
            Ok(plan) => plan,
            Err(reason) => {
                return grouped_attempt_decline(
                    decline(reason, None),
                    0,
                    0,
                    0,
                    0,
                    planning.hamming_coordinate_comparisons,
                )
            }
        };
        let Some(admitted_graph_digest) = graph_digest(self, &problems[0], deadline) else {
            return grouped_attempt_decline(
                decline(
                    if Instant::now() >= deadline {
                        OneAxisPhaseDeclineReason::Deadline
                    } else {
                        OneAxisPhaseDeclineReason::StructuralRefusal
                    },
                    None,
                ),
                0,
                0,
                0,
                0,
                planning.hamming_coordinate_comparisons,
            );
        };

        let mut budget = ExactBudget::new(&limits.phase, deadline);
        let mut linear_phase_cache = HashMap::new();
        let mut preflight_axes = HashSet::new();
        let mut contexts = Vec::with_capacity(plan.contexts.len());
        let mut members = vec![None; problems.len()];
        let mut completed_contexts = 0usize;
        let mut completed_cells = 0usize;

        for (traversal_index, &planned_index) in plan.traversal.iter().enumerate() {
            let planned = &plan.contexts[planned_index];
            let union = union_problem(planned, problems);
            if preflight_axes.insert(union.free_axis) {
                if let Err(reason) = preflight_graph(self, &union, &limits.phase, deadline) {
                    return grouped_attempt_decline(
                        reason,
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    );
                }
            }
            budget.begin_context(traversal_index + 1);
            let (cells, wrapper) = match generate_context_cover(
                self,
                &union,
                &limits,
                completed_cells,
                &mut linear_phase_cache,
                &mut budget,
            ) {
                Ok(cover) => cover,
                Err(reason) => {
                    return grouped_attempt_decline(
                        reason,
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    )
                }
            };
            for &member_index in &planned.members {
                let member = match derive_member(
                    &problems[member_index],
                    traversal_index,
                    &cells,
                    wrapper,
                    &limits,
                    &mut budget,
                ) {
                    Ok(member) => member,
                    Err(reason) => {
                        return grouped_attempt_decline(
                            reason,
                            completed_contexts,
                            completed_cells,
                            budget.operations,
                            budget.cross_context_sparse_linear_updates,
                            planning.hamming_coordinate_comparisons,
                        )
                    }
                };
                members[member_index] = Some(member);
            }
            completed_cells = match completed_cells.checked_add(cells.len()) {
                Some(count) => count,
                None => {
                    return grouped_attempt_decline(
                        decline(OneAxisPhaseDeclineReason::TotalPhaseCellLimit, None),
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    )
                }
            };
            contexts.push(OneAxisGroupedContextCertificate {
                context_digest: planned.digest,
                member_indices: planned.members.clone(),
                lower: OneAxisRational(planned.lower.clone()),
                upper: OneAxisRational(planned.upper.clone()),
                cells,
                wrapper,
            });
            completed_contexts += 1;
        }

        let Some(members) = members.into_iter().collect::<Option<Vec<_>>>() else {
            return grouped_attempt_decline(
                decline(OneAxisPhaseDeclineReason::CertificateMalformed, None),
                completed_contexts,
                completed_cells,
                budget.operations,
                budget.cross_context_sparse_linear_updates,
                planning.hamming_coordinate_comparisons,
            );
        };
        let exact_operations = budget.operations;
        let cross_context_sparse_linear_updates = budget.cross_context_sparse_linear_updates;
        OneAxisGroupedPhaseAttempt {
            certificate: Some(OneAxisGroupedPhaseCertificate {
                version: ONE_AXIS_GROUPED_PHASE_CERTIFICATE_VERSION,
                verdict_authority: false,
                grouped_problem_digest: plan.grouped_problem_digest,
                graph_digest: admitted_graph_digest,
                contexts,
                members,
            }),
            decline: None,
            contexts_examined: completed_contexts,
            phase_cells_examined: completed_cells,
            exact_operations,
            cross_context_sparse_linear_updates,
            hamming_coordinate_comparisons: planning.hamming_coordinate_comparisons,
        }
    }

    /// Independently replay an untrusted grouped certificate.
    ///
    /// Replay reconstructs exact grouping, the Hamming traversal, every
    /// union-interval phase cell, every peeled guard, and every observation
    /// from separate empty caches.
    pub fn replay_exact_grouped_one_axis_phase_certificate_until(
        &self,
        problems: &[OneAxisExactProblem],
        certificate: &OneAxisGroupedPhaseCertificate,
        limits: OneAxisGroupedPhaseLimits,
        deadline: Instant,
    ) -> OneAxisGroupedReplayResult {
        if Instant::now() >= deadline {
            return grouped_replay_reject(
                decline(OneAxisPhaseDeclineReason::Deadline, None),
                0,
                0,
                0,
                0,
                0,
            );
        }
        if certificate.version != ONE_AXIS_GROUPED_PHASE_CERTIFICATE_VERSION
            || certificate.verdict_authority
            || !grouped_certificate_within_limits(certificate, &limits, deadline)
        {
            return grouped_replay_reject(
                decline(OneAxisPhaseDeclineReason::CertificateMalformed, None),
                0,
                0,
                0,
                0,
                0,
            );
        }

        let mut planning = PlanningWork::new(deadline);
        let plan = match build_group_plan(problems, &limits, &mut planning) {
            Ok(plan) => plan,
            Err(reason) => {
                return grouped_replay_reject(
                    decline(reason, None),
                    0,
                    0,
                    0,
                    0,
                    planning.hamming_coordinate_comparisons,
                )
            }
        };
        if certificate.grouped_problem_digest != plan.grouped_problem_digest
            || certificate.contexts.len() != plan.contexts.len()
            || certificate.members.len() != problems.len()
        {
            return grouped_replay_reject(
                decline(OneAxisPhaseDeclineReason::ProblemDigestMismatch, None),
                0,
                0,
                0,
                0,
                planning.hamming_coordinate_comparisons,
            );
        }
        let Some(admitted_graph_digest) = graph_digest(self, &problems[0], deadline) else {
            return grouped_replay_reject(
                decline(
                    if Instant::now() >= deadline {
                        OneAxisPhaseDeclineReason::Deadline
                    } else {
                        OneAxisPhaseDeclineReason::StructuralRefusal
                    },
                    None,
                ),
                0,
                0,
                0,
                0,
                planning.hamming_coordinate_comparisons,
            );
        };
        if certificate.graph_digest != admitted_graph_digest {
            return grouped_replay_reject(
                decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                0,
                0,
                0,
                0,
                planning.hamming_coordinate_comparisons,
            );
        }

        let mut budget = ExactBudget::new(&limits.phase, deadline);
        let mut linear_phase_cache = HashMap::new();
        let mut preflight_axes = HashSet::new();
        let mut observations = vec![None; problems.len()];
        let mut completed_contexts = 0usize;
        let mut completed_cells = 0usize;

        for (traversal_index, (&planned_index, supplied)) in
            plan.traversal.iter().zip(&certificate.contexts).enumerate()
        {
            let planned = &plan.contexts[planned_index];
            if supplied.context_digest != planned.digest
                || supplied.member_indices != planned.members
                || supplied.lower.0 != planned.lower
                || supplied.upper.0 != planned.upper
            {
                return grouped_replay_reject(
                    decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                    completed_contexts,
                    completed_cells,
                    budget.operations,
                    budget.cross_context_sparse_linear_updates,
                    planning.hamming_coordinate_comparisons,
                );
            }
            let union = union_problem(planned, problems);
            if preflight_axes.insert(union.free_axis) {
                if let Err(reason) = preflight_graph(self, &union, &limits.phase, deadline) {
                    return grouped_replay_reject(
                        reason,
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    );
                }
            }
            budget.begin_context(traversal_index + 1);
            let mut static_cache = HashMap::new();
            let mut linear_static_cache = HashMap::new();
            let mut cursor = union.lower.0.clone();
            let mut rebuilt_cells = Vec::with_capacity(supplied.cells.len());
            let mut rebuilt_wrapper = None;
            for supplied_cell in &supplied.cells {
                if supplied_cell.lower.0 != cursor {
                    return grouped_replay_reject(
                        decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    );
                }
                let evaluation = match evaluate_phase(
                    self,
                    &union,
                    &cursor,
                    &mut static_cache,
                    &mut linear_static_cache,
                    &mut linear_phase_cache,
                    &mut budget,
                ) {
                    Ok(evaluation) => evaluation,
                    Err(reason) => {
                        return grouped_replay_reject(
                            reason,
                            completed_contexts,
                            completed_cells,
                            budget.operations,
                            budget.cross_context_sparse_linear_updates,
                            planning.hamming_coordinate_comparisons,
                        )
                    }
                };
                let expected = certificate_cell(&cursor, &evaluation);
                if supplied_cell != &expected {
                    return grouped_replay_reject(
                        decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    );
                }
                let current_wrapper = wrapper_enclosure(&evaluation.wrapper);
                match rebuilt_wrapper {
                    Some(wrapper) if !same_wrapper(wrapper, current_wrapper) => {
                        return grouped_replay_reject(
                            decline(
                                OneAxisPhaseDeclineReason::ReplayMismatch,
                                Some(self.output_name()),
                            ),
                            completed_contexts,
                            completed_cells,
                            budget.operations,
                            budget.cross_context_sparse_linear_updates,
                            planning.hamming_coordinate_comparisons,
                        )
                    }
                    None => rebuilt_wrapper = Some(current_wrapper),
                    _ => {}
                }
                cursor = evaluation.endpoint;
                rebuilt_cells.push(expected);
                if cursor == union.upper.0 {
                    break;
                }
            }
            if cursor != union.upper.0 || rebuilt_cells.len() != supplied.cells.len() {
                return grouped_replay_reject(
                    decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                    completed_contexts,
                    completed_cells,
                    budget.operations,
                    budget.cross_context_sparse_linear_updates,
                    planning.hamming_coordinate_comparisons,
                );
            }
            let Some(wrapper) = rebuilt_wrapper else {
                return grouped_replay_reject(
                    decline(OneAxisPhaseDeclineReason::CertificateMalformed, None),
                    completed_contexts,
                    completed_cells,
                    budget.operations,
                    budget.cross_context_sparse_linear_updates,
                    planning.hamming_coordinate_comparisons,
                );
            };
            if !same_wrapper(wrapper, supplied.wrapper) {
                return grouped_replay_reject(
                    decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                    completed_contexts,
                    completed_cells,
                    budget.operations,
                    budget.cross_context_sparse_linear_updates,
                    planning.hamming_coordinate_comparisons,
                );
            }
            for &member_index in &planned.members {
                let expected = match derive_member(
                    &problems[member_index],
                    traversal_index,
                    &rebuilt_cells,
                    wrapper,
                    &limits,
                    &mut budget,
                ) {
                    Ok(member) => member,
                    Err(reason) => {
                        return grouped_replay_reject(
                            reason,
                            completed_contexts,
                            completed_cells,
                            budget.operations,
                            budget.cross_context_sparse_linear_updates,
                            planning.hamming_coordinate_comparisons,
                        )
                    }
                };
                if certificate.members[member_index] != expected {
                    return grouped_replay_reject(
                        decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    );
                }
                observations[member_index] = Some(expected.observation);
            }
            completed_cells = match completed_cells.checked_add(rebuilt_cells.len()) {
                Some(count) if count <= limits.max_total_phase_cells => count,
                _ => {
                    return grouped_replay_reject(
                        decline(OneAxisPhaseDeclineReason::TotalPhaseCellLimit, None),
                        completed_contexts,
                        completed_cells,
                        budget.operations,
                        budget.cross_context_sparse_linear_updates,
                        planning.hamming_coordinate_comparisons,
                    )
                }
            };
            completed_contexts += 1;
        }
        let Some(observations) = observations.into_iter().collect::<Option<Vec<_>>>() else {
            return grouped_replay_reject(
                decline(OneAxisPhaseDeclineReason::ReplayMismatch, None),
                completed_contexts,
                completed_cells,
                budget.operations,
                budget.cross_context_sparse_linear_updates,
                planning.hamming_coordinate_comparisons,
            );
        };
        OneAxisGroupedReplayResult {
            accepted: true,
            observations: Some(observations),
            decline: None,
            contexts_replayed: completed_contexts,
            phase_cells_replayed: completed_cells,
            exact_operations: budget.operations,
            cross_context_sparse_linear_updates: budget.cross_context_sparse_linear_updates,
            hamming_coordinate_comparisons: planning.hamming_coordinate_comparisons,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ndarray::{arr1, arr2};

    use super::*;
    use crate::layers::{LinearLayer, ReLULayer, SigmoidLayer};
    use crate::network::GraphNode;

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(10)
    }

    fn exact(value: i64) -> OneAxisRational {
        OneAxisRational::from_integer(value)
    }

    fn exact_decimal(value: &str) -> OneAxisRational {
        OneAxisRational::parse_decimal(value).expect("exact decimal")
    }

    fn grouped_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.set_declared_shape(NETWORK_INPUT, vec![4]);
        graph.add_node(GraphNode::from_input(
            "linear",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32, 1.0, 1.0, 1.0]]), Some(arr1(&[0.0_f32])))
                    .expect("valid linear"),
            ),
        ));
        graph.set_declared_shape("linear", vec![1]);
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear".to_string()],
        ));
        graph.set_declared_shape("relu", vec![1]);
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["relu".to_string()],
        ));
        graph.set_declared_shape("sigmoid", vec![1]);
        graph.set_output("sigmoid");
        graph
    }

    fn problem(
        fixed: [i64; 4],
        lower: i64,
        upper: i64,
        relation: OneAxisConstraintRelation,
        bound: &str,
    ) -> OneAxisExactProblem {
        OneAxisExactProblem {
            input_shape: vec![4],
            fixed_inputs: fixed.into_iter().map(exact).collect(),
            free_axis: 0,
            lower: exact(lower),
            upper: exact(upper),
            constraints: vec![OneAxisOutputConstraint {
                relation,
                bound: exact_decimal(bound),
            }],
        }
    }

    #[test]
    fn paired_guards_share_one_union_cover_and_match_serial_observations() {
        let graph = grouped_graph();
        let problems = vec![
            problem(
                [-2, 0, 0, 0],
                -2,
                2,
                OneAxisConstraintRelation::LessEqual,
                "0.5",
            ),
            // The ignored free-axis placeholder differs deliberately.
            problem(
                [99, 0, 0, 0],
                -2,
                2,
                OneAxisConstraintRelation::GreaterEqual,
                "0.5",
            ),
        ];
        let attempt = graph.exact_grouped_one_axis_phase_certificate_until(
            &problems,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        let certificate = attempt.certificate.expect("grouped certificate");
        assert_eq!(certificate.contexts.len(), 1);
        assert_eq!(certificate.contexts[0].member_indices, vec![0, 1]);
        assert_eq!(certificate.contexts[0].lower, exact(-2));
        assert_eq!(certificate.contexts[0].upper, exact(2));
        assert_eq!(certificate.contexts[0].cells.len(), 2);
        assert!(!certificate.verdict_authority);

        let mut serial_cells = 0usize;
        for (index, problem) in problems.iter().enumerate() {
            let serial = graph.exact_one_axis_phase_certificate_until(
                problem,
                OneAxisPhaseLimits::default(),
                deadline(),
            );
            serial_cells += serial.phase_cells_examined;
            assert_eq!(
                certificate.members[index].observation,
                serial.certificate.expect("serial certificate").observation
            );
        }
        assert!(
            attempt.phase_cells_examined < serial_cells,
            "one shared cover should evaluate fewer cells: grouped={} serial={serial_cells}",
            attempt.phase_cells_examined
        );

        let replay = graph.replay_exact_grouped_one_axis_phase_certificate_until(
            &problems,
            &certificate,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        assert!(replay.accepted, "{replay:?}");
        assert_eq!(
            replay.observations,
            Some(
                certificate
                    .members
                    .iter()
                    .map(|member| member.observation.clone())
                    .collect()
            )
        );
    }

    #[test]
    fn disjoint_member_intervals_use_their_clipped_regions() {
        let graph = grouped_graph();
        let problems = vec![
            problem(
                [-2, 0, 0, 0],
                -2,
                -1,
                OneAxisConstraintRelation::LessEqual,
                "0.5",
            ),
            problem(
                [1, 0, 0, 0],
                1,
                2,
                OneAxisConstraintRelation::GreaterEqual,
                "0.5",
            ),
        ];
        let attempt = graph.exact_grouped_one_axis_phase_certificate_until(
            &problems,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        let certificate = attempt
            .certificate
            .clone()
            .unwrap_or_else(|| panic!("grouped certificate: {attempt:?}"));
        assert_eq!(certificate.contexts.len(), 1);
        assert_eq!(certificate.contexts[0].lower, exact(-2));
        assert_eq!(certificate.contexts[0].upper, exact(2));
        for (index, problem) in problems.iter().enumerate() {
            let serial = graph
                .exact_one_axis_phase_certificate_until(
                    problem,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .certificate
                .expect("serial certificate");
            assert_eq!(certificate.members[index].observation, serial.observation);
        }
    }

    #[test]
    fn hamming_walk_is_permutation_stable_and_uses_exact_cross_context_deltas() {
        let graph = grouped_graph();
        let a = problem(
            [0, 0, 0, 0],
            -2,
            2,
            OneAxisConstraintRelation::GreaterEqual,
            "0.5",
        );
        let b = problem(
            [0, 1, 0, 0],
            -2,
            2,
            OneAxisConstraintRelation::GreaterEqual,
            "0.5",
        );
        let c = problem(
            [0, 1, 1, 0],
            -2,
            2,
            OneAxisConstraintRelation::GreaterEqual,
            "0.5",
        );
        let first_problems = vec![c.clone(), a.clone(), b.clone()];
        let first = graph.exact_grouped_one_axis_phase_certificate_until(
            &first_problems,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        let second = graph.exact_grouped_one_axis_phase_certificate_until(
            &[b, c, a],
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        let first_certificate = first.certificate.expect("first grouped certificate");
        let second_certificate = second.certificate.expect("second grouped certificate");
        assert_eq!(
            first_certificate
                .contexts
                .iter()
                .map(|context| context.context_digest)
                .collect::<Vec<_>>(),
            second_certificate
                .contexts
                .iter()
                .map(|context| context.context_digest)
                .collect::<Vec<_>>()
        );
        assert!(first.hamming_coordinate_comparisons > 0);
        assert!(
            first.cross_context_sparse_linear_updates > 0,
            "the one-coordinate Hamming edge should take an exact dense delta"
        );
        for context in &first_certificate.contexts {
            let [member_index] = context.member_indices.as_slice() else {
                panic!("each test context has one member");
            };
            let serial = graph
                .exact_one_axis_phase_certificate_until(
                    &first_problems[*member_index],
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .certificate
                .expect("independent serial certificate");
            assert_eq!(context.cells, serial.cells);
            assert_eq!(context.wrapper, serial.wrapper);
            assert_eq!(
                first_certificate.members[*member_index].observation,
                serial.observation
            );
        }
        let replay = graph.replay_exact_grouped_one_axis_phase_certificate_until(
            &first_problems,
            &first_certificate,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        assert!(replay.accepted, "{replay:?}");
        assert!(replay.cross_context_sparse_linear_updates > 0);
    }

    #[test]
    fn differing_free_axes_are_distinct_and_match_independent_serial_covers() {
        let graph = grouped_graph();
        let first = problem(
            [0, 0, 0, 0],
            -2,
            2,
            OneAxisConstraintRelation::GreaterEqual,
            "0.5",
        );
        let mut second = first.clone();
        second.free_axis = 1;
        let problems = vec![first, second];
        let attempt = graph.exact_grouped_one_axis_phase_certificate_until(
            &problems,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        let certificate = attempt.certificate.expect("two-axis grouped certificate");
        assert_eq!(certificate.contexts.len(), 2);
        for context in &certificate.contexts {
            let [member_index] = context.member_indices.as_slice() else {
                panic!("each test context has one member");
            };
            let serial = graph
                .exact_one_axis_phase_certificate_until(
                    &problems[*member_index],
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .certificate
                .expect("independent serial certificate");
            assert_eq!(context.cells, serial.cells);
        }
        let replay = graph.replay_exact_grouped_one_axis_phase_certificate_until(
            &problems,
            &certificate,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        assert!(replay.accepted, "{replay:?}");
    }

    #[test]
    fn grouped_caps_and_deadline_decline_atomically() {
        let graph = grouped_graph();
        let problems = vec![
            problem(
                [0, 0, 0, 0],
                -2,
                2,
                OneAxisConstraintRelation::GreaterEqual,
                "0.5",
            ),
            problem(
                [0, 1, 0, 0],
                -2,
                2,
                OneAxisConstraintRelation::GreaterEqual,
                "0.5",
            ),
        ];
        let context_capped = graph.exact_grouped_one_axis_phase_certificate_until(
            &problems,
            OneAxisGroupedPhaseLimits {
                max_contexts: 1,
                ..OneAxisGroupedPhaseLimits::default()
            },
            deadline(),
        );
        assert!(context_capped.certificate.is_none());
        assert_eq!(
            context_capped.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::ContextLimit)
        );

        let hamming_capped = graph.exact_grouped_one_axis_phase_certificate_until(
            &problems,
            OneAxisGroupedPhaseLimits {
                max_hamming_coordinate_comparisons: 0,
                ..OneAxisGroupedPhaseLimits::default()
            },
            deadline(),
        );
        assert!(hamming_capped.certificate.is_none());
        assert_eq!(
            hamming_capped.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::HammingTraversalLimit)
        );

        let phase_capped = graph.exact_grouped_one_axis_phase_certificate_until(
            &[problems[0].clone()],
            OneAxisGroupedPhaseLimits {
                max_total_phase_cells: 1,
                ..OneAxisGroupedPhaseLimits::default()
            },
            deadline(),
        );
        assert!(phase_capped.certificate.is_none());
        assert_eq!(
            phase_capped.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::TotalPhaseCellLimit)
        );

        let expired = graph.exact_grouped_one_axis_phase_certificate_until(
            &problems,
            OneAxisGroupedPhaseLimits::default(),
            Instant::now(),
        );
        assert!(expired.certificate.is_none());
        assert_eq!(
            expired.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::Deadline)
        );
    }

    #[test]
    fn grouped_replay_rejects_authority_cover_membership_and_observation_tampering() {
        let graph = grouped_graph();
        let problems = vec![
            problem(
                [-2, 0, 0, 0],
                -2,
                2,
                OneAxisConstraintRelation::LessEqual,
                "0.5",
            ),
            problem(
                [2, 0, 0, 0],
                -2,
                2,
                OneAxisConstraintRelation::GreaterEqual,
                "0.5",
            ),
        ];
        let attempt = graph.exact_grouped_one_axis_phase_certificate_until(
            &problems,
            OneAxisGroupedPhaseLimits::default(),
            deadline(),
        );
        let certificate = attempt
            .certificate
            .clone()
            .unwrap_or_else(|| panic!("grouped certificate: {attempt:?}"));
        let reject = |candidate: &OneAxisGroupedPhaseCertificate| {
            assert!(
                !graph
                    .replay_exact_grouped_one_axis_phase_certificate_until(
                        &problems,
                        candidate,
                        OneAxisGroupedPhaseLimits::default(),
                        deadline(),
                    )
                    .accepted
            );
        };

        let expired = graph.replay_exact_grouped_one_axis_phase_certificate_until(
            &problems,
            &certificate,
            OneAxisGroupedPhaseLimits::default(),
            Instant::now(),
        );
        assert!(!expired.accepted);
        assert_eq!(
            expired.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::Deadline)
        );

        let mut authority = certificate.clone();
        authority.verdict_authority = true;
        reject(&authority);

        let mut core = certificate.clone();
        core.contexts[0].cells[0].core.bias = exact(7);
        reject(&core);

        let mut membership = certificate.clone();
        membership.contexts[0].member_indices.reverse();
        reject(&membership);

        let mut guard = certificate.clone();
        guard.members[0].peeled_constraints[0].necessary = if matches!(
            &guard.members[0].peeled_constraints[0].necessary,
            OneAxisCoreGuard::Always
        ) {
            OneAxisCoreGuard::Impossible
        } else {
            OneAxisCoreGuard::Always
        };
        reject(&guard);

        let mut observation = certificate.clone();
        observation.members[0].observation = match observation.members[0].observation {
            OneAxisPhaseObservation::Inconclusive => OneAxisPhaseObservation::CertifiedEmpty,
            _ => OneAxisPhaseObservation::Inconclusive,
        };
        reject(&observation);

        let mut truncated = certificate.clone();
        truncated.contexts[0].cells.pop();
        reject(&truncated);

        let mut changed_problems = problems;
        changed_problems[0].constraints[0].bound = exact_decimal("0.6");
        assert!(
            !graph
                .replay_exact_grouped_one_axis_phase_certificate_until(
                    &changed_problems,
                    &certificate,
                    OneAxisGroupedPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );
    }
}
