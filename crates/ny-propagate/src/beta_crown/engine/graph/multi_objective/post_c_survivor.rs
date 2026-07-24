// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-off Stage-B Patch-CROWN for a small post-C-matrix survivor set.
//!
//! The existing multi-objective root request remains Stage A and runs first.
//! When its bounds-only C-matrix fast path leaves at most 16 disjunctive rows
//! unresolved, this module may run one compact, generic full-DAG coefficient
//! backward. Publication is atomic: every row, index, cache shape, deadline,
//! and intersection is validated before either the bounds or cache are returned.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;

const POST_C_SURVIVOR_MAX_ROWS: usize = 16;
const POST_C_SURVIVOR_MIN_RESERVE: Duration = Duration::from_secs(12);
const POST_C_SURVIVOR_MAX_RUNTIME: Duration = Duration::from_secs(8);
const POST_C_SURVIVOR_MAX_WORKSPACE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Four coefficient copies conservatively cover the live lower/upper backward
/// pair plus the lower/upper cache retained for child warm-start. Patches are
/// normally much smaller; this admission estimate deliberately treats every
/// cached node as dense so the experimental lane declines before allocation on
/// an obviously oversized graph.
const POST_C_SURVIVOR_COEFFICIENT_COPIES: u64 = 4;

#[derive(Debug)]
pub(super) struct PostCSurvivorPlan {
    stage_a_bounds: Vec<(f32, f32)>,
    pub(super) active_indices: Vec<usize>,
    pub(super) compact_spec_matrix: Array2<f32>,
    pub(super) deadline: Instant,
    pub(super) estimated_workspace_bytes: u64,
}

#[must_use]
pub(super) struct PostCSurvivorAccepted {
    pub(super) merged_bounds: Vec<(f32, f32)>,
    pub(super) compact_cache: CachedLinearBounds,
    pub(super) active_indices: Vec<usize>,
}

fn finite_ordered(lower: f32, upper: f32) -> bool {
    lower.is_finite() && upper.is_finite() && lower <= upper
}

fn bounded_post_c_deadline(now: Instant, global_deadline: Option<Instant>) -> Option<Instant> {
    match global_deadline {
        Some(deadline)
            if deadline.saturating_duration_since(now) >= POST_C_SURVIVOR_MIN_RESERVE =>
        {
            Some((now + POST_C_SURVIVOR_MAX_RUNTIME).min(deadline))
        }
        Some(_) => None,
        // No global deadline denotes unbounded caller headroom. Stage B remains
        // locally bounded to eight seconds.
        None => Some(now + POST_C_SURVIVOR_MAX_RUNTIME),
    }
}

fn estimated_post_c_workspace_bytes(
    rows: usize,
    spec_cols: usize,
    input_elements: usize,
    node_lengths: impl IntoIterator<Item = usize>,
) -> Option<u64> {
    let rows = u64::try_from(rows).ok()?;
    let spec_cols = u64::try_from(spec_cols).ok()?;
    let mut coefficient_elements = u64::try_from(input_elements).ok()?;
    let mut node_count = 0_u64;
    for length in node_lengths {
        coefficient_elements = coefficient_elements.checked_add(u64::try_from(length).ok()?)?;
        node_count = node_count.checked_add(1)?;
    }

    let coefficient_bytes = rows
        .checked_mul(coefficient_elements)?
        .checked_mul(size_of::<f32>() as u64)?
        .checked_mul(POST_C_SURVIVOR_COEFFICIENT_COPIES)?;
    let bias_bytes = rows
        .checked_mul(node_count.checked_add(1)?)?
        .checked_mul(size_of::<f32>() as u64)?
        .checked_mul(POST_C_SURVIVOR_COEFFICIENT_COPIES)?;
    let compact_spec_bytes = rows
        .checked_mul(spec_cols)?
        .checked_mul(size_of::<f32>() as u64)?;
    coefficient_bytes
        .checked_add(bias_bytes)?
        .checked_add(compact_spec_bytes)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_post_c_survivor_plan(
    enabled: bool,
    conjunctive: bool,
    stage_a: &BoundedTensor,
    full_spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    now: Instant,
    global_deadline: Option<Instant>,
) -> Option<PostCSurvivorPlan> {
    build_post_c_survivor_plan_with_headroom(
        enabled,
        conjunctive,
        stage_a,
        full_spec_matrix,
        thresholds,
        input,
        node_bounds,
        now,
        global_deadline,
        crate::network::crown_memory::process_memory_headroom_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_post_c_survivor_plan_with_headroom(
    enabled: bool,
    conjunctive: bool,
    stage_a: &BoundedTensor,
    full_spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    now: Instant,
    global_deadline: Option<Instant>,
    process_headroom_bytes: Option<u64>,
) -> Option<PostCSurvivorPlan> {
    if !enabled || conjunctive || thresholds.is_empty() {
        return None;
    }
    let row_count = thresholds.len();
    if stage_a.shape() != [row_count]
        || full_spec_matrix.nrows() != row_count
        || full_spec_matrix.ncols() == 0
        || full_spec_matrix.iter().any(|value| !value.is_finite())
        || thresholds.iter().any(|threshold| !threshold.is_finite())
    {
        return None;
    }

    let stage_a_bounds: Vec<_> = stage_a
        .lower()
        .iter()
        .copied()
        .zip(stage_a.upper().iter().copied())
        .collect();
    if stage_a_bounds.len() != row_count
        || stage_a_bounds
            .iter()
            .any(|&(lower, upper)| !finite_ordered(lower, upper))
    {
        return None;
    }

    // Disjunctive root authority requires every row. A row is removed only by
    // the verifier's exact strict certificate (`lower > threshold`) and only
    // after its enclosure and threshold passed the finite/ordered checks above.
    let active_indices: Vec<_> = stage_a_bounds
        .iter()
        .zip(thresholds)
        .enumerate()
        .filter_map(|(index, (&(lower, _upper), &threshold))| (lower <= threshold).then_some(index))
        .collect();
    if active_indices.is_empty() || active_indices.len() > POST_C_SURVIVOR_MAX_ROWS {
        return None;
    }

    let deadline = bounded_post_c_deadline(now, global_deadline)?;
    let estimated_workspace_bytes = estimated_post_c_workspace_bytes(
        active_indices.len(),
        full_spec_matrix.ncols(),
        input.len(),
        node_bounds.values().map(BoundedTensor::len),
    )?;
    if estimated_workspace_bytes > POST_C_SURVIVOR_MAX_WORKSPACE_BYTES
        || process_headroom_bytes.is_some_and(|headroom| estimated_workspace_bytes > headroom)
    {
        return None;
    }

    let mut compact_spec_matrix = Array2::zeros((active_indices.len(), full_spec_matrix.ncols()));
    for (compact_row, &original_row) in active_indices.iter().enumerate() {
        compact_spec_matrix
            .row_mut(compact_row)
            .assign(&full_spec_matrix.row(original_row));
    }

    Some(PostCSurvivorPlan {
        stage_a_bounds,
        active_indices,
        compact_spec_matrix,
        deadline,
        estimated_workspace_bytes,
    })
}

fn cache_has_exact_rows(cache: &CachedLinearBounds, expected_rows: usize) -> bool {
    if expected_rows == 0
        || cache.lower_a.is_empty()
        || cache.lower_a.len() != cache.upper_a.len()
        || cache.lower_a.len() != cache.lower_b.len()
        || cache.lower_a.len() != cache.upper_b.len()
    {
        return false;
    }
    cache.lower_a.iter().all(|(name, lower_a)| {
        let Some(upper_a) = cache.upper_a.get(name) else {
            return false;
        };
        let Some(lower_b) = cache.lower_b.get(name) else {
            return false;
        };
        let Some(upper_b) = cache.upper_b.get(name) else {
            return false;
        };
        lower_a.nrows() == expected_rows
            && upper_a.nrows() == expected_rows
            && lower_a.ncols() == upper_a.ncols()
            && lower_b.len() == expected_rows
            && upper_b.len() == expected_rows
            && lower_a.iter().all(|value| value.is_finite())
            && upper_a.iter().all(|value| value.is_finite())
            && lower_b.iter().all(|value| value.is_finite())
            && upper_b.iter().all(|value| value.is_finite())
    })
}

fn merge_compact_post_c_bounds(
    stage_a_bounds: &[(f32, f32)],
    active_indices: &[usize],
    compact_bounds: &BoundedTensor,
) -> Option<Vec<(f32, f32)>> {
    if active_indices.is_empty()
        || compact_bounds.shape() != [active_indices.len()]
        || active_indices.windows(2).any(|pair| pair[0] >= pair[1])
        || active_indices
            .iter()
            .any(|&index| index >= stage_a_bounds.len())
        || stage_a_bounds
            .iter()
            .any(|&(lower, upper)| !finite_ordered(lower, upper))
    {
        return None;
    }

    let compact_rows: Vec<_> = compact_bounds
        .lower()
        .iter()
        .copied()
        .zip(compact_bounds.upper().iter().copied())
        .collect();
    if compact_rows.len() != active_indices.len()
        || compact_rows
            .iter()
            .any(|&(lower, upper)| !finite_ordered(lower, upper))
    {
        return None;
    }

    // Build into a private copy. A single malformed or disjoint row returns
    // `None`, so no earlier row can leak partial authority to the caller.
    let mut merged = stage_a_bounds.to_vec();
    for (&original_row, &(compact_lower, compact_upper)) in active_indices.iter().zip(&compact_rows)
    {
        let (stage_a_lower, stage_a_upper) = merged[original_row];
        let intersection = (
            stage_a_lower.max(compact_lower),
            stage_a_upper.min(compact_upper),
        );
        if !finite_ordered(intersection.0, intersection.1) {
            return None;
        }
        merged[original_row] = intersection;
    }
    Some(merged)
}

fn accept_post_c_survivor_result(
    plan: PostCSurvivorPlan,
    compact_bounds: BoundedTensor,
    compact_cache: Option<CachedLinearBounds>,
    completed_at: Instant,
) -> Option<PostCSurvivorAccepted> {
    if completed_at > plan.deadline {
        return None;
    }
    let compact_cache = compact_cache?;
    if !cache_has_exact_rows(&compact_cache, plan.active_indices.len()) {
        return None;
    }
    let merged_bounds =
        merge_compact_post_c_bounds(&plan.stage_a_bounds, &plan.active_indices, &compact_bounds)?;
    Some(PostCSurvivorAccepted {
        merged_bounds,
        compact_cache,
        active_indices: plan.active_indices,
    })
}

/// Execute one compact generic full-DAG backward. There is intentionally no
/// alpha-state parameter: M34 grants Stage B fixed/adaptive Patch-CROWN
/// authority only, never inherited or optimized alpha authority.
pub(super) fn run_post_c_survivor_candidate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    engine: Option<&dyn GemmEngine>,
    plan: PostCSurvivorPlan,
) -> Option<PostCSurvivorAccepted> {
    if Instant::now() >= plan.deadline {
        return None;
    }
    let result = SpecCrownRequest::new(graph, input, &plan.compact_spec_matrix, engine)
        .node_bounds(node_bounds)
        .deadline_opt(Some(plan.deadline))
        // `run_with_backward_cache` sets the typed input-linear request bit,
        // bypassing every bounds-only C-matrix/GPU/forward-map fast return.
        // No truncation is supplied: this is the generic full-DAG backward.
        .run_with_backward_cache()
        .ok()?;
    accept_post_c_survivor_result(plan, result.0, result.1, Instant::now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::GraphNode;
    use ndarray::{arr1, arr2};

    fn bounded(values: &[(f32, f32)]) -> BoundedTensor {
        BoundedTensor::new(
            arr1(&values.iter().map(|&(lower, _)| lower).collect::<Vec<_>>()).into_dyn(),
            arr1(&values.iter().map(|&(_, upper)| upper).collect::<Vec<_>>()).into_dyn(),
        )
        .unwrap()
    }

    fn cache(rows: usize) -> CachedLinearBounds {
        let mut cache = CachedLinearBounds::default();
        cache.lower_a.insert(
            "relu".to_string(),
            Array2::from_shape_fn((rows, 2), |(row, col)| (row + col) as f32),
        );
        cache.upper_a.insert(
            "relu".to_string(),
            Array2::from_shape_fn((rows, 2), |(row, col)| (row + col + 1) as f32),
        );
        cache
            .lower_b
            .insert("relu".to_string(), ndarray::Array1::zeros(rows));
        cache
            .upper_b
            .insert("relu".to_string(), ndarray::Array1::ones(rows));
        cache
    }

    fn plan(
        stage_a_bounds: Vec<(f32, f32)>,
        active_indices: Vec<usize>,
        deadline: Instant,
    ) -> PostCSurvivorPlan {
        PostCSurvivorPlan {
            compact_spec_matrix: Array2::zeros((active_indices.len(), 2)),
            stage_a_bounds,
            active_indices,
            deadline,
            estimated_workspace_bytes: 0,
        }
    }

    #[test]
    fn disabled_and_conjunctive_plans_are_exact_no_ops() {
        let stage_a = bounded(&[(2.0, 3.0), (-1.0, 4.0)]);
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0_f32, 0.0];
        let input = bounded(&[(-1.0, 1.0), (-1.0, 1.0)]);
        let nodes = HashMap::new();
        let now = Instant::now();
        for (enabled, conjunctive) in [(false, false), (true, true)] {
            assert!(
                build_post_c_survivor_plan_with_headroom(
                    enabled,
                    conjunctive,
                    &stage_a,
                    &spec,
                    &thresholds,
                    &input,
                    &nodes,
                    now,
                    None,
                    Some(u64::MAX),
                )
                .is_none(),
                "disabled/conjunctive execution must leave Stage A untouched"
            );
        }
    }

    #[test]
    fn plan_keeps_exact_survivor_rows_in_original_order() {
        let stage_a = bounded(&[(2.0, 3.0), (-1.0, 4.0), (5.0, 6.0), (0.0, 2.0)]);
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.25, -0.75]]);
        let thresholds = [0.0_f32, 0.0, 4.0, 0.5];
        let input = bounded(&[(-1.0, 1.0), (-1.0, 1.0)]);
        let now = Instant::now();
        let plan = build_post_c_survivor_plan_with_headroom(
            true,
            false,
            &stage_a,
            &spec,
            &thresholds,
            &input,
            &HashMap::new(),
            now,
            Some(now + POST_C_SURVIVOR_MIN_RESERVE),
            Some(u64::MAX),
        )
        .expect("two survivors fit the bounded policy");
        assert_eq!(plan.active_indices, vec![1, 3]);
        assert_eq!(plan.compact_spec_matrix.row(0), spec.row(1));
        assert_eq!(plan.compact_spec_matrix.row(1), spec.row(3));
        assert_eq!(plan.deadline, now + POST_C_SURVIVOR_MAX_RUNTIME);
    }

    #[test]
    fn resource_policy_rejects_short_reserve_too_many_rows_and_four_gib_overflow() {
        let now = Instant::now();
        assert!(bounded_post_c_deadline(
            now,
            Some(now + POST_C_SURVIVOR_MIN_RESERVE.saturating_sub(Duration::from_nanos(1)),)
        )
        .is_none());
        assert_eq!(
            bounded_post_c_deadline(now, Some(now + POST_C_SURVIVOR_MIN_RESERVE)),
            Some(now + POST_C_SURVIVOR_MAX_RUNTIME)
        );

        let rows = POST_C_SURVIVOR_MAX_ROWS + 1;
        let stage_a = bounded(&vec![(-1.0, 1.0); rows]);
        let spec = Array2::ones((rows, 1));
        let thresholds = vec![0.0; rows];
        let input = bounded(&[(-1.0, 1.0)]);
        assert!(build_post_c_survivor_plan_with_headroom(
            true,
            false,
            &stage_a,
            &spec,
            &thresholds,
            &input,
            &HashMap::new(),
            now,
            None,
            Some(u64::MAX),
        )
        .is_none());

        let one_stage_a = bounded(&[(-1.0, 1.0)]);
        let one_spec = Array2::ones((1, 1));
        assert!(
            build_post_c_survivor_plan_with_headroom(
                true,
                false,
                &one_stage_a,
                &one_spec,
                &[0.0],
                &input,
                &HashMap::new(),
                now,
                None,
                Some(0),
            )
            .is_none(),
            "observed process headroom below the estimate must refuse Stage B"
        );

        let over = estimated_post_c_workspace_bytes(16, 1, 1, [20_000_000]).unwrap();
        assert!(over > POST_C_SURVIVOR_MAX_WORKSPACE_BYTES);
        assert!(
            estimated_post_c_workspace_bytes(usize::MAX, usize::MAX, usize::MAX, [usize::MAX])
                .is_none()
        );
    }

    #[test]
    fn successful_candidate_intersects_only_survivors_atomically() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let accepted = accept_post_c_survivor_result(
            plan(
                vec![(3.0, 4.0), (-2.0, 5.0), (7.0, 8.0), (-4.0, 6.0)],
                vec![1, 3],
                deadline,
            ),
            bounded(&[(-1.0, 4.0), (-3.0, 2.0)]),
            Some(cache(2)),
            Instant::now(),
        )
        .expect("fully validated compact result gains authority");
        assert_eq!(
            accepted.merged_bounds,
            vec![(3.0, 4.0), (-1.0, 4.0), (7.0, 8.0), (-3.0, 2.0)]
        );
        assert_eq!(accepted.active_indices, vec![1, 3]);
    }

    #[test]
    fn every_fault_rejects_the_entire_candidate() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let stage_a = vec![(3.0, 4.0), (-2.0, 5.0), (7.0, 8.0), (-4.0, 6.0)];

        assert!(accept_post_c_survivor_result(
            plan(stage_a.clone(), vec![1, 3], deadline),
            bounded(&[(-1.0, 4.0)]),
            Some(cache(2)),
            Instant::now(),
        )
        .is_none());
        assert!(accept_post_c_survivor_result(
            plan(stage_a.clone(), vec![1, 1], deadline),
            bounded(&[(-1.0, 4.0), (-3.0, 2.0)]),
            Some(cache(2)),
            Instant::now(),
        )
        .is_none());
        assert!(accept_post_c_survivor_result(
            plan(stage_a.clone(), vec![1, 3], deadline),
            bounded(&[(6.0, 7.0), (-3.0, 2.0)]),
            Some(cache(2)),
            Instant::now(),
        )
        .is_none());
        assert!(accept_post_c_survivor_result(
            plan(stage_a.clone(), vec![1, 3], deadline),
            bounded(&[(-1.0, 4.0), (-3.0, 2.0)]),
            Some(cache(1)),
            Instant::now(),
        )
        .is_none());
        let mut non_finite_cache = cache(2);
        non_finite_cache.lower_a.get_mut("relu").unwrap()[[0, 0]] = f32::NAN;
        assert!(accept_post_c_survivor_result(
            plan(stage_a.clone(), vec![1, 3], deadline),
            bounded(&[(-1.0, 4.0), (-3.0, 2.0)]),
            Some(non_finite_cache),
            Instant::now(),
        )
        .is_none());
        assert!(accept_post_c_survivor_result(
            plan(stage_a.clone(), vec![1, 3], deadline),
            bounded(&[(-1.0, 4.0), (-3.0, 2.0)]),
            None,
            Instant::now(),
        )
        .is_none());
        assert!(accept_post_c_survivor_result(
            plan(stage_a, vec![1, 3], deadline),
            bounded(&[(-1.0, 4.0), (-3.0, 2.0)]),
            Some(cache(2)),
            deadline + Duration::from_nanos(1),
        )
        .is_none());
    }

    #[test]
    fn tiny_generic_full_dag_candidate_returns_valid_bounds_and_cache() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "hidden",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), Some(arr1(&[0.2, 0.1]))).unwrap(),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["hidden".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, -0.5], [-0.25, 1.0]]), None).unwrap()),
            vec!["relu".to_string()],
        ));
        graph.set_output("output");
        let input = bounded(&[(-1.0, 1.0)]);
        let node_bounds = graph.collect_node_bounds(&input).unwrap();
        let stage_a = bounded(&[(-10.0, 10.0), (-10.0, 10.0)]);
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let now = Instant::now();
        let plan = build_post_c_survivor_plan_with_headroom(
            true,
            false,
            &stage_a,
            &spec,
            &[0.0, 0.0],
            &input,
            &node_bounds,
            now,
            None,
            Some(u64::MAX),
        )
        .unwrap();
        let accepted = run_post_c_survivor_candidate(&graph, &input, &node_bounds, None, plan)
            .expect("tiny generic backward should complete with a matching cache");
        assert_eq!(accepted.merged_bounds.len(), 2);
        assert!(accepted
            .merged_bounds
            .iter()
            .all(|&(lower, upper)| finite_ordered(lower, upper)));
        assert!(cache_has_exact_rows(&accepted.compact_cache, 2));
    }
}
