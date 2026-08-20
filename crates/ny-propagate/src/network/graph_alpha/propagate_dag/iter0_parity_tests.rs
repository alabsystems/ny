// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #iter0-alpha-parity CI invariant (docs/ITER0_ALPHA_PARITY_LOCALIZED_2026-07-30.md).
//!
//! The α loop's iteration-0 fold must reproduce the pre-loop CROWN baseline on
//! the SAME state (same intermediate node-bounds map, CROWN-initialized α).
//! WHY: on 2026-07-29 the cifar100 root α loop evaluated its iteration-0 bound
//! at −2.15e23 against a −1989.90 baseline — every iterate garbage, the whole
//! warmup budget consumed (docs/ROOT_ALPHA_STEP_EXPLODES_AND_STALLS_2026-07-29.md).
//! At HEAD the parity holds exactly on the repro row; this test freezes that
//! equality on a synthetic conv+residual net so the defect class can never
//! ship silently again.
//!
//! The equality is STRUCTURAL, not coincidental: `init_alpha_from_bounds`
//! seeds every unstable neuron's α with the identical adaptive rule the
//! fixed-slope CROWN backward uses (`u > -l ⇒ 1 else 0`,
//! `relu_crown_relaxation` in `network/relu_relax.rs`), and the default
//! `full_conv_alpha=true` keeps α per-neuron, so the iteration-0 relaxation is
//! the CROWN relaxation. Any divergence means one of the two folds evaluated a
//! different representation — exactly the 1e23 defect class.

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;

use super::init::DagAlphaInitResult;
use crate::bounds::AlphaCrownConfig;
use crate::layers::{FlattenLayer, Layer, LinearLayer};
use crate::network::core::{GraphNetwork, GraphNode};
use crate::network::graph_alpha::resnet_skeleton::test_support::{add, box_input, conv, lcg, relu};

/// Deterministic conv resnet with the cifar100 mechanism in miniature:
///
/// ```text
/// input[2,6,6] → conv0 → relu0
///   → { b1c1 → b1r1 → b1c2 } + relu0 = add1          (identity skip)
///   → b2r1 → { b2c1 } ; { p2c1 } from add1; add2      (projection skip)
///   → relu_out → flatten → margins (Gemm, 10 rows)
/// ```
///
/// Ten output rows give the single shared per-ReLU α vector real multi-
/// objective sharing pressure (the cifar100 row folds 99 margin rows through
/// one α per ReLU); two residual Add junctions exercise the merge path the
/// per-node trace flagged (Conv_29→Add_28). All weights come from the
/// fixed-seed LCG — no RNG, byte-stable across runs.
fn margin_resnet_fixture() -> GraphNetwork {
    let mut rng = lcg(0x17E2_0A1F_A171_7E57);
    let mut g = GraphNetwork::new();
    g.add_node(conv(
        &mut rng,
        "conv0",
        crate::NETWORK_INPUT,
        (2, 4),
        3,
        1,
        1,
        true,
    ));
    g.add_node(relu("relu0", "conv0"));
    g.add_node(conv(&mut rng, "b1c1", "relu0", (4, 4), 3, 1, 1, false));
    g.add_node(relu("b1r1", "b1c1"));
    g.add_node(conv(&mut rng, "b1c2", "b1r1", (4, 4), 3, 1, 1, true));
    g.add_node(add("add1", "b1c2", "relu0"));
    g.add_node(relu("b2r1", "add1"));
    g.add_node(conv(&mut rng, "b2c1", "b2r1", (4, 8), 3, 1, 1, true));
    g.add_node(conv(&mut rng, "p2c1", "add1", (4, 8), 1, 1, 0, false));
    g.add_node(add("add2", "b2c1", "p2c1"));
    g.add_node(relu("relu_out", "add2"));
    g.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["relu_out".to_string()],
    ));
    // 8*6*6 = 288 features → 10 margin rows (≥8 objectives).
    let in_features = 8 * 6 * 6;
    let out_rows = 10;
    let w = Array2::from_shape_fn((out_rows, in_features), |_| rng() * 0.2);
    let b = Array1::from_shape_fn(out_rows, |_| rng() * 0.1);
    g.add_node(GraphNode::new(
        "margins",
        Layer::Linear(LinearLayer::new(w, Some(b)).expect("valid margin gemm")),
        vec!["flatten".to_string()],
    ));
    g.set_output("margins");
    g
}

/// Both sides of the invariant, computed on the SAME state, mirroring the
/// loop's CPU arms exactly:
/// - baseline: `propagate_crown_with_engine_and_deadline_and_node_bounds`
///   fed the init map (the loop's reuse arm — `crown_bounds` in
///   `dag_alpha_optimize_loop`, which seeds `best_lower_sum`);
/// - iter-0: `dag_alpha_backward_pass_with_engine` with the CROWN-initialized
///   α and zeroed gradient buffers (the loop's `need_grad` iteration-0 fold).
///
/// `alpha_override` is the trip-wire seam: `Some(v)` overwrites every ReLU's
/// lower/upper α with `v` AFTER init — a stand-in for any bug that makes the
/// loop fold evaluate through a different representation than the baseline.
fn run_iter0_pair(alpha_override: Option<f32>) -> (BoundedTensor, BoundedTensor) {
    let graph = margin_resnet_fixture();
    let input = box_input(&[2, 6, 6], -0.6, 0.6);
    let config = AlphaCrownConfig::default();

    let mut init = match graph
        .init_dag_alpha_state(&input, &config, None, None)
        .expect("init succeeds")
    {
        DagAlphaInitResult::Ready(state) => *state,
        DagAlphaInitResult::EarlyReturn { .. } => {
            panic!("#iter0-alpha-parity fixture must have unstable ReLU neurons")
        }
    };
    // Non-vacuity guard: parity on a stable-only net would hold even with a
    // broken α fold (α is only read at unstable neurons). MEASURED at
    // authoring: all 720 ReLU neurons are unstable under this box (IBP
    // intermediates straddle zero through the whole conv trunk) — maximal α
    // pressure. Keep a loose floor so a legitimate collector change cannot
    // silently drain the test's teeth.
    assert!(
        init.runtime.graph().num_unstable() >= 100,
        "#iter0-alpha-parity fixture must keep a substantial unstable population, got {}",
        init.runtime.graph().num_unstable()
    );
    assert_eq!(
        init.output_dim, 10,
        "#iter0-alpha-parity fixture must fold >=8 margin objectives through one shared alpha"
    );

    if let Some(v) = alpha_override {
        let names: Vec<String> = init.relu_nodes.iter().map(|(n, _)| n.clone()).collect();
        for name in &names {
            let (lo, up) = init
                .runtime
                .graph_mut()
                .relu_alpha_pair_mut(name)
                .expect("fixture relu has alpha");
            lo.fill(v);
            up.fill(v);
        }
    }

    let node_bounds: &HashMap<String, BoundedTensor> = &init.node_bounds;

    let baseline = graph
        .propagate_crown_with_engine_and_deadline_and_node_bounds(
            &input,
            None,
            None,
            Some(node_bounds),
        )
        .expect("baseline CROWN fold succeeds")
        .bounds;

    // Zeroed per-ReLU gradient buffers, sized exactly as the loop sizes them.
    let mut gradients: Vec<Array1<f32>> = init
        .relu_nodes
        .iter()
        .map(|(name, _)| {
            let pre = graph
                .relu_preactivation_bounds(name, &input, node_bounds, "iter0-parity-test")
                .expect("fixture relu pre bounds");
            Array1::zeros(pre.len())
        })
        .collect();
    let mut gradients_upper: Vec<Array1<f32>> =
        gradients.iter().map(|g| Array1::zeros(g.len())).collect();

    let iter0 = graph
        .dag_alpha_backward_pass_with_engine(
            &input,
            node_bounds,
            &init.exec_order,
            init.output_dim,
            init.input_dim,
            init.runtime.relu_name_to_idx(),
            init.runtime.graph(),
            init.runtime.invprop(),
            &mut gradients,
            &mut gradients_upper,
            None,
            None,
            None,
            None,
        )
        .expect("iteration-0 alpha fold succeeds");

    (baseline, iter0)
}

/// The parity check itself, shared by the invariant test and the trip-wire.
/// Returns `Err` naming #iter0-alpha-parity when any output element diverges.
///
/// Tolerance: BIT-EXACT (`f32::to_bits` equality), zero epsilon. WHY the code
/// path warrants it: the α init writes the identical adaptive lower slope the
/// fixed-slope CROWN backward computes (`u > -l ⇒ 1 else 0` in both
/// `init_alpha_from_bounds` and `relu_crown_relaxation`), the default
/// `full_conv_alpha=true` keeps α per-neuron (no channel reduction), and both
/// folds route linear layers through the shared backward dispatch — so
/// iteration 0 performs the same arithmetic as the baseline walk, and
/// MEASURED on this fixture every one of the 10 lower + 10 upper elements is
/// bit-identical (the cifar100 repro row likewise printed exact equality to
/// all 7 displayed sig figs). An epsilon here would shrink the trip-wire's
/// detection margin for no benefit. If a deliberate arithmetic-order refactor
/// ever breaks bit-exactness, re-derive an outward-rounding tolerance from
/// that refactor's actual error bound instead of loosening this blindly.
fn check_iter0_parity(baseline: &BoundedTensor, iter0: &BoundedTensor) -> Result<(), String> {
    // Element-count equality, not shape equality: the loop itself is
    // layout-agnostic here (#2076, #2087 — `update_elementwise_best_bounds`
    // accepts a shape-only ndim mismatch), and the two folds legitimately
    // disagree on a leading unit axis ([1, 10] vs [10]).
    if baseline.lower().len() != iter0.lower().len() {
        return Err(format!(
            "#iter0-alpha-parity VIOLATED: element count {:?} vs {:?}",
            baseline.shape(),
            iter0.shape()
        ));
    }
    let mut worst: Option<(usize, &str, f32, f32)> = None;
    for (side, b, i) in [
        ("lower", baseline.lower(), iter0.lower()),
        ("upper", baseline.upper(), iter0.upper()),
    ] {
        for (idx, (bv, iv)) in b.iter().zip(i.iter()).enumerate() {
            if bv.to_bits() != iv.to_bits() {
                let diff = (bv - iv).abs();
                if worst.is_none_or(|(_, _, wb, wi)| diff > (wb - wi).abs()) {
                    worst = Some((idx, side, *bv, *iv));
                }
            }
        }
    }
    match worst {
        None => Ok(()),
        Some((idx, side, bv, iv)) => Err(format!(
            "#iter0-alpha-parity VIOLATED: the alpha loop's iteration-0 fold diverged from \
             the pre-loop CROWN baseline on the same state — {side}[{idx}] baseline={bv:.9e} \
             iter0={iv:.9e} (the 2026-07-29 1e23 defect class; \
             docs/ITER0_ALPHA_PARITY_LOCALIZED_2026-07-30.md)"
        )),
    }
}

#[test]
fn iteration_zero_alpha_fold_reproduces_the_preloop_crown_baseline() {
    let (baseline, iter0) = run_iter0_pair(None);
    // Bitwise-equal NaNs would satisfy the parity check while both folds are
    // garbage — require finite bounds first so parity can only certify a pair
    // of real numbers.
    for (name, b) in [("baseline", &baseline), ("iter0", &iter0)] {
        assert!(
            b.lower()
                .iter()
                .chain(b.upper().iter())
                .all(|v| v.is_finite()),
            "#iter0-alpha-parity: {name} bounds must be finite"
        );
    }
    if let Err(msg) = check_iter0_parity(&baseline, &iter0) {
        panic!("{msg}");
    }
}

#[test]
fn iter0_parity_harness_detects_an_alpha_representation_divergence() {
    // 0.5 is a valid α (any value in [0,1] is a sound lower slope) that no
    // adaptive init can produce, so an unstable neuron folded through it MUST
    // move the bound — a stand-in for the loop evaluating a different
    // representation than the baseline.
    let (baseline, iter0) = run_iter0_pair(Some(0.5));
    let err = check_iter0_parity(&baseline, &iter0)
        .expect_err("harness must detect a diverged iteration-0 fold");
    assert!(
        err.contains("#iter0-alpha-parity"),
        "detection message must name the invariant: {err}"
    );
}

/// Run ONLY the dag-α walk (no baseline pair) with a state mutation applied
/// after init — the #spec-axis-alpha walk harness. Returns the fold bounds.
fn run_walk_with(mutate: impl FnOnce(&mut crate::bounds::GraphAlphaState)) -> BoundedTensor {
    let graph = margin_resnet_fixture();
    let input = box_input(&[2, 6, 6], -0.6, 0.6);
    let config = AlphaCrownConfig::default();
    let mut init = match graph
        .init_dag_alpha_state(&input, &config, None, None)
        .expect("init succeeds")
    {
        DagAlphaInitResult::Ready(state) => *state,
        DagAlphaInitResult::EarlyReturn { .. } => panic!("fixture must be unstable"),
    };
    mutate(init.runtime.graph_mut());
    let node_bounds: &HashMap<String, BoundedTensor> = &init.node_bounds;
    let mut gradients: Vec<Array1<f32>> = init
        .relu_nodes
        .iter()
        .map(|(name, _)| {
            let pre = graph
                .relu_preactivation_bounds(name, &input, node_bounds, "spec-axis-test")
                .expect("fixture relu pre bounds");
            Array1::zeros(pre.len())
        })
        .collect();
    let mut gradients_upper: Vec<Array1<f32>> =
        gradients.iter().map(|g| Array1::zeros(g.len())).collect();
    graph
        .dag_alpha_backward_pass_with_engine(
            &input,
            node_bounds,
            &init.exec_order,
            init.output_dim,
            init.input_dim,
            init.runtime.relu_name_to_idx(),
            init.runtime.graph(),
            init.runtime.invprop(),
            &mut gradients,
            &mut gradients_upper,
            None,
            None,
            None,
            None,
        )
        .expect("spec-axis walk succeeds")
}

/// Install δ rows for every ReLU node: `slots` maps spec rows to a constant
/// δ value per slot (0.0 for the parity arm).
fn install_deltas(state: &mut crate::bounds::GraphAlphaState, slots: &[(usize, f32)]) {
    state.spec_slot_rows = slots.iter().map(|&(row, _)| row).collect();
    let names: Vec<String> = state.alphas.keys().cloned().collect();
    for name in names {
        let width = state.alphas[&name].len();
        let mut deltas = Array2::<f32>::zeros((slots.len(), width));
        for (slot, &(_, value)) in slots.iter().enumerate() {
            deltas.row_mut(slot).fill(value);
        }
        state.spec_deltas.insert(name, deltas);
    }
}

/// #spec-axis-alpha design §5.2: ACTIVE slots with δ = 0 must leave the whole
/// dense walk bit-identical — the parity anchor now proven through the full
/// fold, not just the accessor.
#[test]
fn spec_axis_zero_delta_active_slots_keep_the_whole_walk_bit_identical() {
    let _env = crate::tests::lock_env_shared();
    let shared = run_walk_with(|_| {});
    let with_slots = run_walk_with(|state| install_deltas(state, &[(2, 0.0), (7, 0.0)]));
    for (index, (a, b)) in shared
        .lower()
        .iter()
        .chain(shared.upper().iter())
        .zip(with_slots.lower().iter().chain(with_slots.upper().iter()))
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "δ=0 active slots must be bit-identical through the walk at element {index} \
             (#spec-axis-alpha / #iter0-alpha-parity)"
        );
    }
}

/// A nonzero lower-δ on one spec row must (a) leave every OTHER row's lower
/// bound bit-identical, (b) leave the ENTIRE upper path bit-identical
/// (upper-isolation: lower δ must never leak through the single-alpha
/// `map_or` fallback), and (c) actually move its own row — non-vacuity.
#[test]
fn spec_axis_delta_moves_only_its_row_and_never_the_upper_path() {
    const TARGET_ROW: usize = 3;
    let shared = run_walk_with(|_| {});
    let steered = run_walk_with(|state| install_deltas(state, &[(TARGET_ROW, 0.35)]));

    let shared_lower: Vec<f32> = shared.lower().iter().copied().collect();
    let steered_lower: Vec<f32> = steered.lower().iter().copied().collect();
    let shared_upper: Vec<f32> = shared.upper().iter().copied().collect();
    let steered_upper: Vec<f32> = steered.upper().iter().copied().collect();

    for (row, (a, b)) in shared_upper.iter().zip(steered_upper.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "upper path must be untouched by lower δ (row {row}, #spec-axis-alpha upper-isolation)"
        );
    }
    for (row, (a, b)) in shared_lower.iter().zip(steered_lower.iter()).enumerate() {
        if row == TARGET_ROW {
            continue;
        }
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {row} owns no slot and must be bit-identical (#spec-axis-alpha row isolation)"
        );
    }
    assert_ne!(
        shared_lower[TARGET_ROW].to_bits(),
        steered_lower[TARGET_ROW].to_bits(),
        "δ=0.35 on an all-unstable fixture must move its own row — a no-op here means the \
         spec table never reached the compose (#spec-axis-alpha non-vacuity)"
    );
    assert!(
        steered_lower[TARGET_ROW].is_finite(),
        "steered row must remain a real bound"
    );
}
