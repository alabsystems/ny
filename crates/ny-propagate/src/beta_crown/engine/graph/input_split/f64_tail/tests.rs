// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Hook-level tests for the f64 tail pass (design §6.6 + parity checklist B):
//! gate-off byte-identity, gate-on verification of an fp-blocked domain
//! through BOTH call sites, and hook-level fail-closed on an unsupported net.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::*;
use crate::layers::{Layer, LinearLayer, SigmoidLayer};
use crate::network::graph_crown_f64_tail::{f64_tail_verify, force_f64_tail, F64TailOutcome};
use crate::{GraphNetwork, GraphNode};

/// The gate is a process-global atomic; serialize every test that forces it.
static GATE_LOCK: Mutex<()> = Mutex::new(());

/// Restore both process-global test overrides even when an assertion unwinds.
/// The mutex serializes the override window; this guard prevents one failing
/// canary from poisoning every later gate test in the same test process.
struct GateOverrideReset;

impl Drop for GateOverrideReset {
    fn drop(&mut self) {
        force_f64_tail(None);
        force_alpha_tail(None);
    }
}

fn bt(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lo.len()]), lo.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[hi.len()]), hi.to_vec()).unwrap(),
    )
    .expect("finite bounds")
}

/// The design-§1.1 storage-tax geometry in miniature — lsnc's Gemm->ReLU->Gemm
/// shape: the spec row's backward through l2 produces INTERMEDIATE
/// coefficients `a1 = row·W2 ≈ [1000, -1000]`; the STABLY-ACTIVE ReLU node
/// boundary materializes them in the lane's f32 carrier (absolute cast error
/// ~u·1000 ≈ 3e-5 per entry, carried as certified coefficient error and
/// discharged over the [1,5] pre-activation box => ~1e-4 penalty), while
/// `a1·W1` then cancels down to O(1) composite coefficients — so the penalty
/// dwarfs the O(1)-magnitude bound's f32 ulp (~2e-7) and the f64 tail bound
/// is materially tighter than the f32 floor, with the win surviving the
/// directed f64 -> f32 downcast. The ReLU is stable-active (pre-activation
/// [1, 5]), so BOTH lanes use the exact identity relaxation: the diff is pure
/// floating point, not relaxation quality.
fn build_cancellation_net() -> GraphNetwork {
    let w1 = arr2(&[[1.0007_f32, 0.9993], [0.9993, 0.99895]]);
    // +3 biases keep the ReLU pre-activation in [1, 5]: stably active.
    let b1 = arr1(&[3.0_f32, 3.0]);
    // row [1,1] gives a1 = [1000.35, -1000.15]: large stored intermediates.
    let w2 = arr2(&[[500.2_f32, -500.1], [500.15, -500.05]]);
    let b2 = arr1(&[0.0_f32, 0.0]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(crate::layers::ReLULayer),
        vec!["l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("l2")),
        vec!["relu".to_string()],
    ));
    graph.set_output("l2");
    graph
}

fn make_domain(
    input: &BoundedTensor,
    obj_bounds: Vec<(f32, f32)>,
    priority: f32,
) -> MultiObjInputDomain {
    MultiObjInputDomain {
        input_bounds: Arc::new(input.clone()),
        obj_bounds,
        linear_bounds: None,
        depth: 7,
        priority,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

/// Compute (f32 lane lower, certified f64 lower) for row `[1, 1]` of the
/// cancellation net, plus a threshold that sits BETWEEN them within the
/// default guard band — an "fp-blocked" fixture: the f32 verdict fails, the
/// f64 verdict must succeed.
fn fp_blocked_fixture() -> (GraphNetwork, BoundedTensor, Array2<f32>, f32, f32) {
    let graph = build_cancellation_net();
    let input = bt(&[-1.0, -1.0], &[1.0, 1.0]);
    let spec = arr2(&[[1.0_f32, 1.0]]);
    let node_bounds =
        crate::network::collect_intermediate_bounds(&graph, &input, None, None).expect("anchors");
    let (f32_bounds, _) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &input,
            &spec,
            None,
            &node_bounds,
            None,
        )
        .expect("f32 backward");
    let f32_lower = f32_bounds.flatten().lower()[[0]];
    let l_cert = match f64_tail_verify(
        &graph,
        &input,
        &spec,
        &[-1e9_f32],
        &[1],
        None,
        Some(&node_bounds),
        None,
        None,
    ) {
        F64TailOutcome::Verified { row_lowers } => row_lowers[0],
        other => panic!("fixture pass must verify against -1e9, got {other:?}"),
    };
    let diff = l_cert - f64::from(f32_lower);
    assert!(
        diff > 1e-6,
        "fixture must exhibit a material f32 storage tax, got diff={diff:e}"
    );
    // Threshold strictly between f32_lower and l_cert, within the 5e-3 band.
    #[allow(clippy::cast_possible_truncation)]
    let threshold = (f64::from(f32_lower) + diff.min(2e-3) / 2.0) as f32;
    assert!(
        threshold >= f32_lower,
        "f32 verdict must FAIL at the fixture threshold"
    );
    assert!(
        f64::from(threshold) < l_cert,
        "f64 verdict must clear the fixture threshold"
    );
    // The win must survive the directed f64 -> f32 downcast (fixture
    // geometry: bound at O(1) magnitude, so an f32 ulp is ~1e-7 << diff).
    #[allow(clippy::cast_possible_truncation)]
    let cast = next_down_f32(l_cert as f32);
    assert!(
        cast > threshold,
        "downcast certified lower {cast} must still clear threshold {threshold}"
    );
    (graph, input, spec, threshold, f32_lower)
}

#[test]
fn gate_off_is_byte_identical_and_gate_on_verifies() {
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    let (graph, input, spec, threshold, f32_lower) = fp_blocked_fixture();
    let thresholds = [threshold];
    let clause_sizes = [1usize];
    let obj_bounds = vec![(f32_lower, f32::INFINITY)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);
    assert!(
        priority.is_finite() && (-5e-3..=0.0).contains(&priority),
        "fixture domain must sit inside the guard band, got {priority}"
    );

    // --- Gate OFF: escalate must be a byte-identical no-op. ---
    force_f64_tail(Some(false));
    let mut domains_off = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains_off,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
    );
    assert_eq!(
        domains_off[0].obj_bounds[0].0.to_bits(),
        obj_bounds[0].0.to_bits(),
        "gate off: lower bound bits must be untouched"
    );
    assert_eq!(
        domains_off[0].obj_bounds[0].1.to_bits(),
        obj_bounds[0].1.to_bits(),
        "gate off: upper bound bits must be untouched"
    );
    assert_eq!(domains_off[0].priority.to_bits(), priority.to_bits());
    assert!(!disjunctive_domain_verified(
        &domains_off[0].obj_bounds,
        &thresholds,
        &clause_sizes
    ));
    let deadline = Instant::now() + Duration::from_secs(30);
    assert!(
        !f64_tail_last_chance(
            &graph,
            &domains_off[0],
            &spec,
            &thresholds,
            &clause_sizes,
            None,
            None,
            deadline,
        ),
        "gate off: last chance must decline without work"
    );

    // --- Gate ON: the fp-blocked domain must verify through call site 1. ---
    force_f64_tail(Some(true));
    let mut domains_on = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains_on,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
    );
    assert!(
        disjunctive_domain_verified(&domains_on[0].obj_bounds, &thresholds, &clause_sizes),
        "gate on: the raised f32 bounds must pass the UNCHANGED f32 verdict funnel"
    );
    assert!(
        domains_on[0].obj_bounds[0].0 > threshold,
        "certified lower {} must clear threshold {}",
        domains_on[0].obj_bounds[0].0,
        threshold
    );
    // Monotone: never below the f32 floor it started from.
    assert!(domains_on[0].obj_bounds[0].0 >= f32_lower);

    // --- Gate ON: call site 2 (pop-side last chance) on a fresh domain. ---
    let domain = make_domain(&input, obj_bounds, priority);
    assert!(
        f64_tail_last_chance(
            &graph,
            &domain,
            &spec,
            &thresholds,
            &clause_sizes,
            None,
            None,
            deadline,
        ),
        "gate on: last chance must certify the fp-blocked domain"
    );

    force_f64_tail(None);
}

#[test]
fn out_of_band_domain_is_not_escalated() {
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    let (graph, input, spec, _threshold, f32_lower) = fp_blocked_fixture();
    // Threshold far above the bound: gap way outside the band -> ineligible.
    let thresholds = [f32_lower + 1.0];
    let clause_sizes = [1usize];
    let obj_bounds = vec![(f32_lower, f32::INFINITY)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);
    assert!(priority < -5e-3);

    force_f64_tail(Some(true));
    let mut domains = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
    );
    force_f64_tail(None);
    assert_eq!(
        domains[0].obj_bounds[0].0.to_bits(),
        obj_bounds[0].0.to_bits(),
        "out-of-band domain must not be touched"
    );
    assert_eq!(domains[0].priority.to_bits(), priority.to_bits());
}

#[test]
fn unsupported_net_declines_and_leaves_domain_untouched() {
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    // Sigmoid head: outside the f64-tail op class.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32], [2.0]]), None).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer),
        vec!["l1".to_string()],
    ));
    graph.set_output("sig");

    let input = bt(&[-1.0], &[1.0]);
    let spec = arr2(&[[1.0_f32, 0.0]]);
    let thresholds = [0.5_f32];
    let clause_sizes = [1usize];
    // In-band, unverified domain.
    let obj_bounds = vec![(0.4999_f32, 1.0)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);
    assert!(priority.is_finite() && priority >= -5e-3);

    force_f64_tail(Some(true));
    let mut domains = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
    );
    let last = f64_tail_last_chance(
        &graph,
        &domains[0],
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        Instant::now() + Duration::from_secs(5),
    );
    force_f64_tail(None);

    assert!(!last, "unsupported net must decline the last chance");
    assert_eq!(
        domains[0].obj_bounds[0].0.to_bits(),
        obj_bounds[0].0.to_bits(),
        "declined escalation must leave the domain byte-identical"
    );
    assert_eq!(
        domains[0].obj_bounds[0].1.to_bits(),
        obj_bounds[0].1.to_bits()
    );
    assert_eq!(domains[0].priority.to_bits(), priority.to_bits());
    assert!(!disjunctive_domain_verified(
        &domains[0].obj_bounds,
        &thresholds,
        &clause_sizes
    ));
}

#[test]
fn mul_binary_alphas_are_threaded_through() {
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    // Smoke: passing a (well-formed) alpha map through both hooks must not
    // panic or alter behavior on a net without MulBinary nodes.
    let (graph, input, spec, threshold, f32_lower) = fp_blocked_fixture();
    let thresholds = [threshold];
    let clause_sizes = [1usize];
    let mut alphas: HashMap<String, Array2<f32>> = HashMap::new();
    alphas.insert(
        "absent_node".to_string(),
        Array2::from_elem((2, 4), 0.5_f32),
    );
    let obj_bounds = vec![(f32_lower, f32::INFINITY)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);

    force_f64_tail(Some(true));
    let mut domains = vec![make_domain(&input, obj_bounds, priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        Some(&alphas),
        None,
        None,
    );
    force_f64_tail(None);
    assert!(disjunctive_domain_verified(
        &domains[0].obj_bounds,
        &thresholds,
        &clause_sizes
    ));
}

// ---------------------------------------------------------------------------
// Alpha-tail escalation (docs/LSNC_ALPHA_TAIL_DESIGN.md options A+B).
//
// Every test here forces process-global gates, so it takes BOTH the
// spec-gate parity mutex (`batched::SPEC_GATE_TEST_LOCK`, registered per the
// house discipline for gate-forcing parity tests) and the local GATE_LOCK,
// in that fixed order.
// ---------------------------------------------------------------------------

use crate::beta_crown::engine::graph::propagation::batched::SPEC_GATE_TEST_LOCK;
use crate::network::graph_crown_f64_tail::{f64_tail_verify_refreshed, force_alpha_tail};

/// The x^2 relaxation-gap fixture: `mul = id * id` over `[-d, d]`. The
/// interpolated-McCormick family is capped at `-d^2` on the parent box (no
/// alpha can beat it), while each half box contains an EXACT facet that the
/// per-child refresh must find (r -> 0 on the left half, r -> 1 on the
/// right). With `t` between the cap and the exact-child bounds, the domain
/// closes ONLY through refresh + micro-BaB — options A and B jointly.
fn build_square_net() -> (GraphNetwork, HashMap<String, Array2<f32>>) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "id",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("id")),
    ));
    graph.add_node(GraphNode::new(
        "mul",
        Layer::MulBinary(crate::layers::MulBinaryLayer),
        vec!["id".to_string(), "id".to_string()],
    ));
    graph.set_output("mul");
    let mut alphas: HashMap<String, Array2<f32>> = HashMap::new();
    alphas.insert("mul".to_string(), Array2::from_elem((2, 1), 0.5_f32));
    (graph, alphas)
}

#[test]
fn alpha_gate_off_is_byte_identical() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    let (graph, alphas) = build_square_net();
    let d = 0.02_f32;
    let input = bt(&[-d], &[d]);
    let spec = arr2(&[[1.0_f32]]);
    let thresholds = [-1.2e-4_f32];
    let clause_sizes = [1usize];
    let obj_bounds = vec![(-3.0e-4_f32, f32::INFINITY)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);
    assert!(priority.is_finite() && (-5e-3..0.0).contains(&priority));

    // BOTH gates off: escalate + last chance are byte-identical no-ops.
    force_f64_tail(Some(false));
    force_alpha_tail(Some(false));
    let mut domains = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        Some(&alphas),
        None,
        None,
    );
    assert_eq!(
        domains[0].obj_bounds[0].0.to_bits(),
        obj_bounds[0].0.to_bits(),
        "both gates off: lower bound bits must be untouched"
    );
    assert_eq!(domains[0].priority.to_bits(), priority.to_bits());
    assert!(
        !f64_tail_last_chance(
            &graph,
            &domains[0],
            &spec,
            &thresholds,
            &clause_sizes,
            Some(&alphas),
            None,
            Instant::now() + Duration::from_secs(5),
        ),
        "both gates off: last chance must decline without work"
    );
    force_f64_tail(None);
    force_alpha_tail(None);
}

#[test]
fn alpha_tail_micro_bab_closes_relaxation_gap() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    let (graph, alphas) = build_square_net();
    let d = 0.02_f32;
    let input = bt(&[-d], &[d]);
    let spec = arr2(&[[1.0_f32]]);
    // Between the parent family cap (-d^2 = -4e-4) and the exact-facet child
    // bounds (~0): refresh alone CANNOT close, refresh + micro-BaB MUST.
    let thresholds = [-1.2e-4_f32];
    let clause_sizes = [1usize];
    let obj_bounds = vec![(-3.0e-4_f32, f32::INFINITY)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);
    assert!(priority.is_finite() && (-5e-3..0.0).contains(&priority));

    // Unit precondition: the refreshed SINGLE-SHOT honestly fails on the
    // parent box (the close below is genuinely micro-BaB's).
    let single = f64_tail_verify_refreshed(
        &graph,
        &input,
        &spec,
        &thresholds,
        &clause_sizes,
        Some(&alphas),
        None,
        None,
        None,
        20,
        0xA1FA_7A11,
    );
    assert!(
        matches!(single.outcome, F64TailOutcome::NotVerified { .. }),
        "single-shot refresh must NOT close the parent (family cap)"
    );

    // NY_ALPHA_TAIL supersedes NY_F64_TAIL: arm ONLY the alpha gate.
    force_f64_tail(Some(false));
    force_alpha_tail(Some(true));
    let mut domains = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        Some(&alphas),
        None,
        None,
    );
    force_f64_tail(None);
    force_alpha_tail(None);
    assert!(
        disjunctive_domain_verified(&domains[0].obj_bounds, &thresholds, &clause_sizes),
        "micro-BaB (all children certified) must verify the parent through the f32 funnel"
    );
    assert!(domains[0].obj_bounds[0].0 > thresholds[0]);
    // Monotone: never below the f32 floor it started from.
    assert!(domains[0].obj_bounds[0].0 >= obj_bounds[0].0);
}

#[test]
fn alpha_tail_micro_bab_fails_closed_when_any_child_cannot_verify() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    let (graph, alphas) = build_square_net();
    let d = 0.02_f32;
    let input = bt(&[-d], &[d]);
    let spec = arr2(&[[1.0_f32]]);
    // Threshold ABOVE the true min (x^2 >= 0, t = 1e-6): any child containing
    // x = 0 can never certify, so the grouped ALL-children contract must
    // decline the whole escalation and leave the domain byte-identical.
    let thresholds = [1.0e-6_f32];
    let clause_sizes = [1usize];
    let obj_bounds = vec![(-9.9e-5_f32, f32::INFINITY)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);
    assert!(priority.is_finite() && (-5e-3..0.0).contains(&priority));

    force_f64_tail(Some(false));
    force_alpha_tail(Some(true));
    let mut domains = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        Some(&alphas),
        None,
        None,
    );
    force_f64_tail(None);
    force_alpha_tail(None);
    assert!(
        !disjunctive_domain_verified(&domains[0].obj_bounds, &thresholds, &clause_sizes),
        "a truly-violated threshold must never verify"
    );
    assert_eq!(
        domains[0].obj_bounds[0].0.to_bits(),
        obj_bounds[0].0.to_bits(),
        "declined micro-BaB must leave the domain byte-identical"
    );
    assert_eq!(domains[0].priority.to_bits(), priority.to_bits());
}

#[test]
fn alpha_tail_alone_arms_the_seam_for_fp_blocked_domains() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    // The landed fp-blocked LINEAR fixture (no MulBinary): the refreshed pass
    // degenerates to the plain certified baseline, and NY_ALPHA_TAIL alone
    // (NY_F64_TAIL off) must still arm the seam and verify it.
    let (graph, input, spec, threshold, f32_lower) = fp_blocked_fixture();
    let thresholds = [threshold];
    let clause_sizes = [1usize];
    let obj_bounds = vec![(f32_lower, f32::INFINITY)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);

    force_f64_tail(Some(false));
    force_alpha_tail(Some(true));
    let mut domains = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
    );
    assert!(
        disjunctive_domain_verified(&domains[0].obj_bounds, &thresholds, &clause_sizes),
        "alpha gate alone must arm the seam (supersedes NY_F64_TAIL)"
    );
    // Pop-side last chance, same arming.
    let domain = make_domain(&input, obj_bounds, priority);
    assert!(f64_tail_last_chance(
        &graph,
        &domain,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        Instant::now() + Duration::from_secs(30),
    ));
    force_f64_tail(None);
    force_alpha_tail(None);
}

#[test]
fn alpha_tail_declines_unsupported_net_and_leaves_domain_untouched() {
    let _spec_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _gate_reset = GateOverrideReset;
    // Sigmoid head: outside the f64-tail op class — the alpha path must fail
    // closed exactly like the landed path.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32], [2.0]]), None).expect("l1")),
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer),
        vec!["l1".to_string()],
    ));
    graph.set_output("sig");

    let input = bt(&[-1.0], &[1.0]);
    let spec = arr2(&[[1.0_f32, 0.0]]);
    let thresholds = [0.5_f32];
    let clause_sizes = [1usize];
    let obj_bounds = vec![(0.4999_f32, 1.0)];
    let priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);

    force_f64_tail(Some(false));
    force_alpha_tail(Some(true));
    let mut domains = vec![make_domain(&input, obj_bounds.clone(), priority)];
    f64_tail_escalate_batch(
        &mut domains,
        &graph,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
    );
    let last = f64_tail_last_chance(
        &graph,
        &domains[0],
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        None,
        Instant::now() + Duration::from_secs(5),
    );
    force_f64_tail(None);
    force_alpha_tail(None);

    assert!(!last);
    assert_eq!(
        domains[0].obj_bounds[0].0.to_bits(),
        obj_bounds[0].0.to_bits(),
        "declined alpha escalation must leave the domain byte-identical"
    );
    assert_eq!(domains[0].priority.to_bits(), priority.to_bits());
}
