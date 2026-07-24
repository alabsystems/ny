// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Discriminating oracles for warm-α consumption in the REORDER deferred
//! rebound (cgan step-2C).
//!
//! Contract under test (`bound_deferred_domains_batch_with_metrics` /
//! `bound_deferred_dense_spec_domains_batch`):
//! - `input_split_alpha_iteration == 0` (default): the rebound never reads
//!   `inherited_alpha_state` — bounds byte-identical to a domain without one,
//!   and the inherited Arc passes through untouched.
//! - `input_split_alpha_iteration > 0`: domains carrying parent slopes run the
//!   per-domain SPSA refinement as an overlay that INTERSECTS with the frozen
//!   batch result (tighter-or-equal by construction) and store the refined α
//!   as a NEW Arc for their children to warm-start from.

use std::sync::Arc;

use super::super::batching::{
    bound_deferred_disjunctive_domains_batch, bound_deferred_domains_batch,
    bound_deferred_multi_obj_domains_batch, force_warm_refine_parallel,
    WARM_REFINE_PARALLEL_BATCHES,
};
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::propagation::batched::SPEC_GATE_TEST_LOCK;
use crate::bounds::GraphAlphaState;

use super::*;

/// Root fixture shared by the oracles: the 2-ReLU DAG from #3870 plus a
/// root-optimized α seed built the same way the warmup path does.
fn warm_rebound_fixture() -> (
    GraphNetwork,
    HashMap<String, BoundedTensor>,
    Arc<GraphAlphaState>,
    BoundedTensor,
) {
    let graph = build_reference_bounds_graph_3870();
    let root_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("valid root input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");

    let mut alpha_state = GraphAlphaState::new();
    for (relu, pre) in [("relu1", "linear1"), ("relu2", "linear2")] {
        let pre_act = root_node_bounds
            .get(pre)
            .expect("pre-activation bounds present");
        alpha_state
            .add_relu_node(relu, pre_act, false)
            .expect("alpha node registration");
    }
    let child_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.2]).into_dyn(),
        arr1(&[0.4_f32, 1.0]).into_dyn(),
    )
    .expect("valid child input");

    (graph, root_node_bounds, Arc::new(alpha_state), child_input)
}

fn warm_config(alpha_iteration: usize) -> BetaCrownConfig {
    BetaCrownConfig {
        use_alpha_crown: true,
        input_split_alpha_iteration: alpha_iteration,
        ..BetaCrownConfig::default()
    }
}

fn deferred_multi_obj_domain(
    input_bounds: &BoundedTensor,
    num_specs: usize,
    inherited_alpha_state: Option<Arc<GraphAlphaState>>,
) -> MultiObjInputDomain {
    MultiObjInputDomain {
        input_bounds: Arc::new(input_bounds.clone()),
        obj_bounds: vec![(f32::NEG_INFINITY, f32::INFINITY); num_specs],
        linear_bounds: None,
        depth: 1,
        priority: 0.0,
        needs_bounding: true,
        node_bounds_override: None,
        inherited_alpha_state,
    }
}

fn deferred_single_obj_domain(
    input_bounds: &BoundedTensor,
    inherited_alpha_state: Option<Arc<GraphAlphaState>>,
) -> GraphInputDomain {
    GraphInputDomain {
        input_bounds: Arc::new(input_bounds.clone()),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: 0.0,
        linear_bounds: None,
        needs_bounding: true,
        node_bounds_override: None,
        inherited_alpha_state,
    }
}

fn assert_refined_state_is_queue_sized(state: &GraphAlphaState) {
    assert!(state.velocity.is_empty());
    assert!(state.adam_m.is_empty());
    assert!(state.adam_v.is_empty());
    assert!(state.velocity_upper.is_empty());
    assert!(state.adam_m_upper.is_empty());
    assert!(state.adam_v_upper.is_empty());
}

fn assert_relu_alpha_states_are_bit_identical(
    serial: &GraphAlphaState,
    parallel: &GraphAlphaState,
) {
    assert_eq!(serial.alphas.len(), parallel.alphas.len());
    assert_eq!(serial.alphas_upper.len(), parallel.alphas_upper.len());
    for (name, serial_alpha) in &serial.alphas {
        let parallel_alpha = parallel
            .alphas
            .get(name)
            .unwrap_or_else(|| panic!("parallel state missing lower alpha {name}"));
        assert_eq!(
            f32_bits(serial_alpha.iter()),
            f32_bits(parallel_alpha.iter()),
            "lower alpha {name} changed under parallel scheduling"
        );
    }
    for (name, serial_alpha) in &serial.alphas_upper {
        let parallel_alpha = parallel
            .alphas_upper
            .get(name)
            .unwrap_or_else(|| panic!("parallel state missing upper alpha {name}"));
        assert_eq!(
            f32_bits(serial_alpha.iter()),
            f32_bits(parallel_alpha.iter()),
            "upper alpha {name} changed under parallel scheduling"
        );
    }
}

fn assert_domain_encloses_concrete_grid(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    domain: &MultiObjInputDomain,
) {
    let flat = input.flatten();
    let lower = flat.lower();
    let upper = flat.upper();
    for i in 0..=4 {
        for j in 0..=4 {
            let x0 = lower[[0]] + (upper[[0]] - lower[[0]]) * (i as f32 / 4.0);
            let x1 = lower[[1]] + (upper[[1]] - lower[[1]]) * (j as f32 / 4.0);
            let point =
                BoundedTensor::concrete(arr1(&[x0, x1]).into_dyn()).expect("valid grid point");
            let output = graph
                .propagate_concrete_point(&point, None, None)
                .expect("concrete graph evaluation should succeed")
                .flatten();
            for (row, coeffs) in spec_matrix.rows().into_iter().enumerate() {
                let value = coeffs
                    .iter()
                    .zip(output.lower().iter())
                    .map(|(&coefficient, &output)| coefficient * output)
                    .sum::<f32>();
                let (bound_lower, bound_upper) = domain.obj_bounds[row];
                assert!(
                    value >= bound_lower - 1.0e-4,
                    "sample ({i},{j}) row {row}: value {value} below parallel warm lower {bound_lower}"
                );
                assert!(
                    value <= bound_upper + 1.0e-4,
                    "sample ({i},{j}) row {row}: value {value} above parallel warm upper {bound_upper}"
                );
            }
        }
    }
}

fn f32_bits<'a>(values: impl IntoIterator<Item = &'a f32>) -> Vec<u32> {
    values.into_iter().map(|value| value.to_bits()).collect()
}

fn assert_rebound_fields_are_bit_identical(
    with_seed: &MultiObjInputDomain,
    without_seed: &MultiObjInputDomain,
) {
    assert_eq!(
        with_seed
            .obj_bounds
            .iter()
            .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect::<Vec<_>>(),
        without_seed
            .obj_bounds
            .iter()
            .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        with_seed.priority.to_bits(),
        without_seed.priority.to_bits()
    );
    assert_eq!(with_seed.depth, without_seed.depth);
    assert_eq!(with_seed.needs_bounding, without_seed.needs_bounding);
    assert_eq!(
        with_seed.node_bounds_override.is_some(),
        without_seed.node_bounds_override.is_some()
    );

    let with_input = with_seed.input_bounds.flatten();
    let without_input = without_seed.input_bounds.flatten();
    assert_eq!(
        f32_bits(with_input.lower().iter()),
        f32_bits(without_input.lower().iter())
    );
    assert_eq!(
        f32_bits(with_input.upper().iter()),
        f32_bits(without_input.upper().iter())
    );

    match (&with_seed.linear_bounds, &without_seed.linear_bounds) {
        (Some(with_linear), Some(without_linear)) => {
            assert_eq!(with_linear.lower_a().dim(), without_linear.lower_a().dim());
            assert_eq!(with_linear.upper_a().dim(), without_linear.upper_a().dim());
            assert_eq!(
                f32_bits(with_linear.lower_a().iter()),
                f32_bits(without_linear.lower_a().iter())
            );
            assert_eq!(
                f32_bits(with_linear.upper_a().iter()),
                f32_bits(without_linear.upper_a().iter())
            );
            assert_eq!(
                f32_bits(with_linear.lower_b().iter()),
                f32_bits(without_linear.lower_b().iter())
            );
            assert_eq!(
                f32_bits(with_linear.upper_b().iter()),
                f32_bits(without_linear.upper_b().iter())
            );
        }
        (None, None) => {}
        _ => panic!("gate-off inherited alpha changed linear-bound presence"),
    }
}

/// alpha_iteration == 0 gate: the deferred rebound must be BYTE-IDENTICAL
/// whether or not a domain carries inherited α, and the inherited Arc must
/// pass through untouched (no refinement ran).
#[test]
fn test_deferred_rebound_alpha_iteration_zero_is_byte_identical() {
    let (graph, root_node_bounds, seed_alpha, child_input) = warm_rebound_fixture();
    let spec_matrix = arr2(&[[1.0_f32, -0.5], [-0.3, 1.0]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let config = warm_config(0);

    let mut domains = vec![
        deferred_multi_obj_domain(&child_input, 2, Some(Arc::clone(&seed_alpha))),
        deferred_multi_obj_domain(&child_input, 2, None),
    ];
    bound_deferred_multi_obj_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        &thresholds,
        None,
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        &config,
        None,
        0,
    )
    .expect("frozen deferred rebound should succeed");

    assert_eq!(
        domains[0].obj_bounds, domains[1].obj_bounds,
        "with alpha_iteration == 0 the rebound must never read inherited α: \
         bounds must be byte-identical with and without it"
    );
    assert_eq!(domains[0].priority, domains[1].priority);
    let carried = domains[0]
        .inherited_alpha_state
        .as_ref()
        .expect("inherited α must be preserved for children");
    assert!(
        Arc::ptr_eq(carried, &seed_alpha),
        "with alpha_iteration == 0 no refinement may run: the inherited Arc must pass through untouched"
    );
    assert!(domains.iter().all(|d| !d.needs_bounding));
}

/// Exact cGAN route oracle: either warm gate being disabled must make inherited
/// per-domain alpha observationally inert for every rebound field.
#[test]
fn test_deferred_disjunctive_rebound_alpha_gate_off_is_bit_identical_f8() {
    let (graph, root_node_bounds, seed_alpha, child_input) = warm_rebound_fixture();
    let spec_matrix = arr2(&[[1.0_f32, -0.5], [-0.3, 1.0]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize];
    let mut alpha_disabled = warm_config(3);
    alpha_disabled.use_alpha_crown = false;

    for config in [warm_config(0), alpha_disabled] {
        let mut domains = vec![
            deferred_multi_obj_domain(&child_input, 2, Some(Arc::clone(&seed_alpha))),
            deferred_multi_obj_domain(&child_input, 2, None),
        ];
        bound_deferred_disjunctive_domains_batch(
            &mut domains,
            &graph,
            &spec_matrix,
            &thresholds,
            &clause_sizes,
            None,
            Some(&root_node_bounds),
            Some(seed_alpha.as_ref()),
            None,
            None,
            None,
            &config,
            None,
            0,
        )
        .expect("gate-off disjunctive deferred rebound should succeed");

        assert_rebound_fields_are_bit_identical(&domains[0], &domains[1]);
        let carried = domains[0]
            .inherited_alpha_state
            .as_ref()
            .expect("gate-off rebound must preserve inherited alpha for descendants");
        assert!(Arc::ptr_eq(carried, &seed_alpha));
        assert!(domains.iter().all(|domain| !domain.needs_bounding));
    }
}

/// alpha_iteration > 0 oracle (multi-objective lane): the deferred rebound must
/// produce per-spec bounds tighter-or-equal to the frozen-only rebound and
/// store a NEW refined α Arc on the domain.
#[test]
fn test_deferred_multi_obj_rebound_warm_alpha_tightens_and_stores_refined_arc() {
    let (graph, root_node_bounds, seed_alpha, child_input) = warm_rebound_fixture();
    let spec_matrix = arr2(&[[1.0_f32, -0.5], [-0.3, 1.0]]);
    let thresholds = [0.0_f32, 0.0_f32];

    // Frozen baseline: identical domain, alpha_iteration == 0.
    let mut frozen_domains = vec![deferred_multi_obj_domain(
        &child_input,
        2,
        Some(Arc::clone(&seed_alpha)),
    )];
    bound_deferred_multi_obj_domains_batch(
        &mut frozen_domains,
        &graph,
        &spec_matrix,
        &thresholds,
        None,
        Some(&root_node_bounds),
        Some(seed_alpha.as_ref()),
        None,
        None,
        None,
        &warm_config(0),
        None,
        0,
    )
    .expect("frozen deferred rebound should succeed");

    // Warm lane: same domain, alpha_iteration > 0.
    let mut warm_domains = vec![deferred_multi_obj_domain(
        &child_input,
        2,
        Some(Arc::clone(&seed_alpha)),
    )];
    bound_deferred_multi_obj_domains_batch(
        &mut warm_domains,
        &graph,
        &spec_matrix,
        &thresholds,
        None,
        Some(&root_node_bounds),
        Some(seed_alpha.as_ref()),
        None,
        None,
        None,
        &warm_config(3),
        None,
        0,
    )
    .expect("warm deferred rebound should succeed");

    let frozen = &frozen_domains[0];
    let warm = &warm_domains[0];
    assert!(!warm.needs_bounding);
    for (idx, ((warm_l, warm_u), (frozen_l, frozen_u))) in warm
        .obj_bounds
        .iter()
        .zip(frozen.obj_bounds.iter())
        .enumerate()
    {
        assert!(
            warm_l >= frozen_l,
            "spec {idx}: warm lower bound must be tighter-or-equal to frozen: warm={warm_l}, frozen={frozen_l}"
        );
        assert!(
            warm_u <= frozen_u,
            "spec {idx}: warm upper bound must be tighter-or-equal to frozen: warm={warm_u}, frozen={frozen_u}"
        );
        assert!(
            warm_l.is_finite() && warm_u.is_finite(),
            "spec {idx}: warm rebound must keep finite bounds on this fixture"
        );
    }
    let refined = warm
        .inherited_alpha_state
        .as_ref()
        .expect("warm rebound must store refined α for children");
    assert!(
        !Arc::ptr_eq(refined, &seed_alpha),
        "warm rebound must store a NEW refined α Arc (proof the refinement lane ran)"
    );
    assert_refined_state_is_queue_sized(refined);
}

/// End-to-end cGAN route oracle: the deferred disjunctive lane must consume a
/// warm candidate, retain the frozen enclosure by intersection, and enclose
/// concrete evaluations when warm refinement uses fixed IBP references.
#[test]
fn test_deferred_disjunctive_rebound_warm_candidate_and_fixed_reference_soundness_f8() {
    let (graph, root_node_bounds, seed_alpha, child_input) = warm_rebound_fixture();
    let spec_matrix = arr2(&[[1.0_f32, -0.5], [-0.3, 1.0]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize];

    let mut frozen_domains = vec![deferred_multi_obj_domain(
        &child_input,
        2,
        Some(Arc::clone(&seed_alpha)),
    )];
    bound_deferred_disjunctive_domains_batch(
        &mut frozen_domains,
        &graph,
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        None,
        Some(&root_node_bounds),
        Some(seed_alpha.as_ref()),
        None,
        None,
        None,
        &warm_config(0),
        None,
        0,
    )
    .expect("frozen disjunctive deferred rebound should succeed");

    let mut warm_config = warm_config(3);
    warm_config.alpha_config.fix_interm_bounds = true;
    let mut warm_domains = vec![deferred_multi_obj_domain(
        &child_input,
        2,
        Some(Arc::clone(&seed_alpha)),
    )];
    bound_deferred_disjunctive_domains_batch(
        &mut warm_domains,
        &graph,
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        None,
        Some(&root_node_bounds),
        Some(seed_alpha.as_ref()),
        None,
        None,
        None,
        &warm_config,
        None,
        0,
    )
    .expect("warm disjunctive deferred rebound should succeed");

    let frozen = &frozen_domains[0];
    let warm = &warm_domains[0];
    assert!(!warm.needs_bounding);
    for (row, ((warm_lower, warm_upper), (frozen_lower, frozen_upper))) in
        warm.obj_bounds.iter().zip(&frozen.obj_bounds).enumerate()
    {
        assert!(
            warm_lower >= frozen_lower,
            "row {row}: warm lower must retain the frozen sound floor"
        );
        assert!(
            warm_upper <= frozen_upper,
            "row {row}: warm upper must retain the frozen sound ceiling"
        );
        assert!(warm_lower.is_finite() && warm_upper.is_finite());
    }
    let refined = warm
        .inherited_alpha_state
        .as_ref()
        .expect("warm disjunctive rebound must store refined alpha");
    assert!(
        !Arc::ptr_eq(refined, &seed_alpha),
        "a new Arc proves that an eligible warm candidate actually engaged"
    );
    assert_refined_state_is_queue_sized(refined);

    let child_flat = child_input.flatten();
    let lower = child_flat.lower();
    let upper = child_flat.upper();
    for i in 0..=8 {
        for j in 0..=8 {
            let x0 = lower[[0]] + (upper[[0]] - lower[[0]]) * (i as f32 / 8.0);
            let x1 = lower[[1]] + (upper[[1]] - lower[[1]]) * (j as f32 / 8.0);
            let point =
                BoundedTensor::concrete(arr1(&[x0, x1]).into_dyn()).expect("valid concrete sample");
            let output = graph
                .propagate_concrete_point(&point, None, None)
                .expect("concrete graph evaluation should succeed")
                .flatten();
            let values = output.lower();
            for (row, coeffs) in spec_matrix.rows().into_iter().enumerate() {
                let value = coeffs
                    .iter()
                    .zip(values.iter())
                    .map(|(&coefficient, &output)| coefficient * output)
                    .sum::<f32>();
                let (bound_lower, bound_upper) = warm.obj_bounds[row];
                assert!(
                    value >= bound_lower - 1.0e-4,
                    "sample ({i},{j}) row {row}: value {value} below warm lower {bound_lower}"
                );
                assert!(
                    value <= bound_upper + 1.0e-4,
                    "sample ({i},{j}) row {row}: value {value} above warm upper {bound_upper}"
                );
            }
        }
    }
}

/// alpha_iteration > 0 oracle (single-objective lane): scalar bounds
/// tighter-or-equal to frozen, refined Arc stored, priority recomputed.
#[test]
fn test_deferred_single_obj_rebound_warm_alpha_tightens_and_stores_refined_arc() {
    let (graph, root_node_bounds, seed_alpha, child_input) = warm_rebound_fixture();
    let spec_matrix = arr2(&[[1.0_f32, -0.5]]);

    let mut frozen_domains = vec![deferred_single_obj_domain(
        &child_input,
        Some(Arc::clone(&seed_alpha)),
    )];
    bound_deferred_domains_batch(
        &mut frozen_domains,
        &graph,
        &spec_matrix,
        None,
        Some(&root_node_bounds),
        Some(seed_alpha.as_ref()),
        None,
        None,
        None,
        &warm_config(0),
    )
    .expect("frozen deferred rebound should succeed");

    let mut warm_domains = vec![deferred_single_obj_domain(
        &child_input,
        Some(Arc::clone(&seed_alpha)),
    )];
    let config = warm_config(3);
    bound_deferred_domains_batch(
        &mut warm_domains,
        &graph,
        &spec_matrix,
        None,
        Some(&root_node_bounds),
        Some(seed_alpha.as_ref()),
        None,
        None,
        None,
        &config,
    )
    .expect("warm deferred rebound should succeed");

    let frozen = &frozen_domains[0];
    let warm = &warm_domains[0];
    assert!(!warm.needs_bounding);
    assert!(
        warm.lower_bound >= frozen.lower_bound,
        "warm lower bound must be tighter-or-equal to frozen: warm={}, frozen={}",
        warm.lower_bound,
        frozen.lower_bound
    );
    assert!(
        warm.upper_bound <= frozen.upper_bound,
        "warm upper bound must be tighter-or-equal to frozen: warm={}, frozen={}",
        warm.upper_bound,
        frozen.upper_bound
    );
    assert!(warm.lower_bound.is_finite() && warm.upper_bound.is_finite());
    assert_eq!(
        warm.priority,
        config
            .domain_priority(warm.lower_bound, warm.upper_bound)
            .expect("priority recomputation should succeed"),
        "warm rebound must recompute priority from the overlaid bounds"
    );
    let refined = warm
        .inherited_alpha_state
        .as_ref()
        .expect("warm rebound must store refined α for children");
    assert!(
        !Arc::ptr_eq(refined, &seed_alpha),
        "warm rebound must store a NEW refined α Arc (proof the refinement lane ran)"
    );
    assert_refined_state_is_queue_sized(refined);
}

/// #cgan-warm-par oracle: a two-candidate grouped-disjunctive rebound must
/// actually take the parallel scheduler, preserve candidate/application order,
/// match the serial reference bit-for-bit on deterministic CPU arithmetic, keep
/// the frozen enclosure intersection, and enclose concrete samples.
#[test]
fn test_deferred_disjunctive_parallel_warm_refine_matches_serial_and_is_sound() {
    let _gate_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (graph, root_node_bounds, seed_alpha, child_a) = warm_rebound_fixture();
    let child_b = BoundedTensor::new(
        arr1(&[-0.6_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 0.7]).into_dyn(),
    )
    .expect("valid second child input");
    let spec_matrix = arr2(&[[1.0_f32, -0.5], [-0.3, 1.0]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize];
    let mut config = warm_config(3);
    config.input_split_warm_parallel = true;
    let make_domains = || {
        vec![
            deferred_multi_obj_domain(&child_a, 2, Some(Arc::clone(&seed_alpha))),
            deferred_multi_obj_domain(&child_b, 2, Some(Arc::clone(&seed_alpha))),
        ]
    };
    let run = |domains: &mut [MultiObjInputDomain], config: &BetaCrownConfig| {
        bound_deferred_disjunctive_domains_batch(
            domains,
            &graph,
            &spec_matrix,
            &thresholds,
            &clause_sizes,
            None,
            Some(&root_node_bounds),
            Some(seed_alpha.as_ref()),
            None,
            None,
            None,
            config,
            None,
            0,
        )
    };

    let mut frozen = make_domains();
    let frozen_result = run(&mut frozen, &warm_config(0));

    // A global selector must not arm a category whose preset/config leaves the
    // scoped activation false.
    WARM_REFINE_PARALLEL_BATCHES.store(0, std::sync::atomic::Ordering::Relaxed);
    force_warm_refine_parallel(Some(true));
    let mut scoped_off = make_domains();
    let scoped_off_result = run(&mut scoped_off, &warm_config(3));
    let scoped_off_parallel_batches =
        WARM_REFINE_PARALLEL_BATCHES.load(std::sync::atomic::Ordering::Relaxed);

    force_warm_refine_parallel(Some(false));
    let mut serial = make_domains();
    let serial_result = run(&mut serial, &config);

    WARM_REFINE_PARALLEL_BATCHES.store(0, std::sync::atomic::Ordering::Relaxed);
    force_warm_refine_parallel(Some(true));
    let mut parallel = make_domains();
    let parallel_result = run(&mut parallel, &config);
    let parallel_batches = WARM_REFINE_PARALLEL_BATCHES.load(std::sync::atomic::Ordering::Relaxed);
    force_warm_refine_parallel(None);

    frozen_result.expect("frozen rebound should succeed");
    scoped_off_result.expect("preset-scoped-off warm rebound should succeed serially");
    serial_result.expect("serial warm rebound should succeed");
    parallel_result.expect("parallel warm rebound should succeed");
    assert_eq!(
        scoped_off_parallel_batches, 0,
        "environment selector must not bypass the default-false preset scope"
    );
    assert_eq!(
        parallel_batches, 1,
        "two eligible candidates must exercise exactly one parallel warm batch"
    );

    for (idx, ((frozen_domain, serial_domain), parallel_domain)) in
        frozen.iter().zip(&serial).zip(&parallel).enumerate()
    {
        assert_rebound_fields_are_bit_identical(&scoped_off[idx], serial_domain);
        assert_rebound_fields_are_bit_identical(serial_domain, parallel_domain);
        for (row, ((parallel_lower, parallel_upper), (frozen_lower, frozen_upper))) in
            parallel_domain
                .obj_bounds
                .iter()
                .zip(&frozen_domain.obj_bounds)
                .enumerate()
        {
            assert!(
                parallel_lower >= frozen_lower,
                "domain {idx} row {row}: parallel warm lower regressed below frozen"
            );
            assert!(
                parallel_upper <= frozen_upper,
                "domain {idx} row {row}: parallel warm upper regressed above frozen"
            );
        }

        let serial_alpha = serial_domain
            .inherited_alpha_state
            .as_ref()
            .expect("serial warm rebound must store refined alpha");
        let parallel_alpha = parallel_domain
            .inherited_alpha_state
            .as_ref()
            .expect("parallel warm rebound must store refined alpha");
        assert!(!Arc::ptr_eq(parallel_alpha, &seed_alpha));
        assert_refined_state_is_queue_sized(parallel_alpha);
        assert_relu_alpha_states_are_bit_identical(serial_alpha, parallel_alpha);
    }

    assert_domain_encloses_concrete_grid(&graph, &child_a, &spec_matrix, &parallel[0]);
    assert_domain_encloses_concrete_grid(&graph, &child_b, &spec_matrix, &parallel[1]);
}
