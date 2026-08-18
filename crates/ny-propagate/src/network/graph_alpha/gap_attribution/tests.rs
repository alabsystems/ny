// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #gap-attribution tests.
//!
//! Two tiers:
//!  1. UNIT — the trichotomy classifier, the depth estimate and the identity
//!     check as pure logic, including determinism of every reported ordering.
//!  2. IDENTITY — Theorem 1 end to end on the conv+residual margin fixture
//!     (the `#binding-row-replay` / iter0-parity idiom): run the real certified
//!     DAG alpha fold, attribute one seed row, and assert the decomposition
//!     reproduces `f(x*) - B(x*)`. This is the tier that makes the whole
//!     construction trustworthy — if it fails, nothing downstream may be used.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD};
use ny_tensor::BoundedTensor;

use super::*;
use crate::bounds::GraphAlphaState;
use crate::layers::{FlattenLayer, Layer, LinearLayer};
use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};
use crate::network::graph_alpha::resnet_skeleton::test_support::{add, box_input, conv, lcg, relu};

// ===========================================================================
// Tier 1 — unit
// ===========================================================================

fn attribution(per_node: Vec<(&str, Vec<f64>)>, f: f64, b: f64) -> GapAttribution {
    let mut map = HashMap::new();
    let mut sum = 0.0;
    for (name, g) in per_node {
        let live = g.iter().filter(|v| **v > 0.0).count();
        sum += g.iter().sum::<f64>();
        map.insert(
            name.to_string(),
            NodeGap {
                g: Array1::from(g),
                unstable: 0,
                live,
            },
        );
    }
    GapAttribution {
        row: 0,
        x_star: ArrayD::zeros(ndarray::IxDyn(&[1])),
        f_x_star: f,
        bound_at_x_star: b,
        sum_g: sum,
        per_node: map,
        residual: f - b - sum,
    }
}

#[test]
fn identity_holds_when_gaps_sum_to_the_difference() {
    let a = attribution(vec![("relu1", vec![0.25, 0.75])], 3.0, 2.0);
    assert!(a.residual.abs() < 1e-12);
    a.verify_identity(1e-9).expect("identity should hold");
}

#[test]
fn identity_refuses_when_gaps_do_not_account_for_the_difference() {
    // Theorem 1 violated: the difference is 1.0 but only 0.4 is attributed.
    // This is the shape a non-affine non-ReLU node would produce.
    let a = attribution(vec![("relu1", vec![0.2, 0.2])], 3.0, 2.0);
    let err = a.verify_identity(1e-9).expect_err("must refuse");
    assert!(format!("{err}").contains("Theorem 1 violated"), "{err}");
}

#[test]
fn identity_refuses_a_negative_attributed_gap() {
    // Sums correctly, but a negative g_j means the selected relaxation did not
    // match the coefficient's sign — non-negativity is part of Theorem 1.
    let a = attribution(vec![("relu1", vec![1.5, -0.5])], 3.0, 2.0);
    let err = a.verify_identity(1e-9).expect_err("must refuse");
    assert!(
        format!("{err}").contains("negative attributed gap"),
        "{err}"
    );
}

#[test]
fn negative_report_is_deterministic_across_node_order() {
    // HashMap iteration order varies run to run; the diagnostic must not.
    for _ in 0..8 {
        let a = attribution(
            vec![
                ("zzz", vec![-0.5]),
                ("aaa", vec![-0.25]),
                ("mmm", vec![-1.0]),
            ],
            0.0,
            1.75,
        );
        let err = format!("{}", a.verify_identity(1e-9).unwrap_err());
        assert!(err.contains("'aaa'[0]"), "{err}");
    }
}

#[test]
fn classify_falsified_takes_precedence() {
    // f(x*) below threshold is a counterexample regardless of the gap size.
    let a = attribution(vec![("relu1", vec![5.0])], -1.0, -6.0);
    match a.classify(-6.0, 0.0) {
        GapVerdict::Falsified { true_value, .. } => assert_eq!(true_value, -1.0),
        other => panic!("expected Falsified, got {other:?}"),
    }
}

#[test]
fn classify_arithmetic_limited_when_relaxation_gap_cannot_reach() {
    // f(x*) = 0.5 clears the threshold, so the domain is NOT falsified, but
    // only 0.5 of relaxation slack exists against a 2.0 shortfall. The residual
    // 1.5 is certified-arithmetic conservatism: branching is provably futile.
    let a = attribution(vec![("relu1", vec![0.5])], 0.5, 0.0);
    match a.classify(-2.0, 0.0) {
        GapVerdict::ArithmeticLimited {
            relaxation_gap,
            needed,
        } => {
            assert!((relaxation_gap - 0.5).abs() < 1e-12);
            assert!((needed - 2.0).abs() < 1e-12);
        }
        other => panic!("expected ArithmeticLimited, got {other:?}"),
    }
    // And that shortfall is exactly E = B(x*) - lb_sound.
    assert!((a.certified_error(-2.0) - 2.0).abs() < 1e-12);
}

#[test]
fn classify_relaxation_limited_when_there_is_room() {
    let a = attribution(vec![("relu1", vec![3.0])], 3.0, 0.0);
    match a.classify(-2.0, 0.0) {
        GapVerdict::RelaxationLimited { .. } => {}
        other => panic!("expected RelaxationLimited, got {other:?}"),
    }
}

#[test]
fn attribution_depth_counts_the_largest_gaps_first() {
    let a = attribution(vec![("r", vec![0.5, 0.3, 0.1, 0.05])], 0.95, 0.0);
    assert_eq!(a.attribution_depth(0.0), Some(0));
    assert_eq!(a.attribution_depth(0.4), Some(1)); // 0.5 alone clears it
    assert_eq!(a.attribution_depth(0.7), Some(2)); // 0.5 + 0.3
    assert_eq!(a.attribution_depth(0.9), Some(3)); // + 0.1
                                                   // Deduction 3's decisive case: the whole attribution cannot reach it, so no
                                                   // amount of branching verifies this domain.
    assert_eq!(a.attribution_depth(2.0), None);
}

#[test]
fn attribution_depth_ignores_inert_neurons() {
    // A thousand zero-gap neurons must not inflate the estimate; only the two
    // live ones can carry any improvement (Deduction 1).
    let mut g = vec![0.0; 1000];
    g[400] = 0.6;
    g[900] = 0.6;
    let a = attribution(vec![("r", g)], 1.2, 0.0);
    assert_eq!(a.live_neurons(), 2);
    assert_eq!(a.attribution_depth(1.0), Some(2));
}

#[test]
fn top_neurons_ranks_by_attributed_gap_and_is_deterministic() {
    let a = attribution(vec![("b", vec![0.1, 0.9]), ("a", vec![0.4, 0.0])], 1.4, 0.0);
    let top = a.top_neurons(3);
    assert_eq!(top[0], ("b".to_string(), 1, 0.9));
    assert_eq!(top[1], ("a".to_string(), 0, 0.4));
    assert_eq!(top[2], ("b".to_string(), 0, 0.1));
    // Zero-gap neurons are inert and never ranked.
    assert_eq!(a.top_neurons(99).len(), 3);
}

#[test]
fn ties_break_deterministically_on_node_then_index() {
    for _ in 0..8 {
        let a = attribution(vec![("b", vec![0.5]), ("a", vec![0.5])], 1.0, 0.0);
        let top = a.top_neurons(2);
        assert_eq!(top[0].0, "a");
        assert_eq!(top[1].0, "b");
    }
}

#[test]
fn non_finite_residual_is_refused() {
    let mut a = attribution(vec![("r", vec![1.0])], 1.0, 0.0);
    a.residual = f64::NAN;
    assert!(a.verify_identity(1e-9).is_err());
}

// ===========================================================================
// Tier 2 — Theorem 1 end to end on a real certified fold
// ===========================================================================

/// The iter0-parity conv+residual fixture verbatim (identity skip + projection
/// skip + 10 margin rows through one shared alpha per ReLU) — the same
/// topology class as `CIFAR100_resnet_medium`, small enough to fold in a test.
fn margin_resnet_fixture() -> GraphNetwork {
    let mut rng = lcg(0x17E2_0A1F_A171_7E57);
    let mut g = GraphNetwork::new();
    g.add_node(conv(
        &mut rng,
        "conv0",
        NETWORK_INPUT,
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

const FIXTURE_RELUS: [&str; 4] = ["relu0", "b1r1", "b2r1", "relu_out"];
const OUTPUT_DIM: usize = 10;

struct Harness {
    graph: GraphNetwork,
    input: BoundedTensor,
    node_bounds: HashMap<String, BoundedTensor>,
    exec_order: Vec<String>,
    relu_name_to_idx: HashMap<String, usize>,
}

impl Harness {
    fn new() -> Self {
        let graph = margin_resnet_fixture();
        let input = box_input(&[2, 6, 6], -0.6, 0.6);
        let node_bounds = graph.collect_node_bounds(&input).expect("IBP forward");
        let exec_order = graph.exec_order().expect("exec order").to_vec();
        let relu_name_to_idx: HashMap<String, usize> = FIXTURE_RELUS
            .iter()
            .enumerate()
            .map(|(i, n)| (n.to_string(), i))
            .collect();
        Self {
            graph,
            input,
            node_bounds,
            exec_order,
            relu_name_to_idx,
        }
    }

    /// Alpha state with interior values so no neuron sits on a relaxation
    /// endpoint (which would make the identity trivially exact there).
    fn interior_alpha_state(&self, channel_shared: bool) -> GraphAlphaState {
        let mut st = GraphAlphaState::new();
        let mut rng = lcg(0xB1D1_0A11_5EED);
        for name in FIXTURE_RELUS {
            let pre = self
                .graph
                .relu_preactivation_bounds(name, &self.input, &self.node_bounds, "gap-attr-test")
                .expect("pre bounds");
            st.add_relu_node(name, pre, channel_shared)
                .expect("alpha init");
            let alpha = st.alphas.get_mut(name).expect("lower alpha");
            for a in alpha.iter_mut() {
                *a = 0.5 + 0.3 * rng(); // in (0.5, 0.8)
            }
        }
        st
    }

    fn fold_with_intermediates(
        &self,
        st: &GraphAlphaState,
    ) -> (BoundedTensor, GraphAlphaCrownIntermediate) {
        let g: Vec<Array1<f32>> = FIXTURE_RELUS
            .iter()
            .map(|name| {
                let pre = self
                    .graph
                    .relu_preactivation_bounds(
                        name,
                        &self.input,
                        &self.node_bounds,
                        "gap-attr-test",
                    )
                    .expect("pre bounds");
                Array1::zeros(pre.len())
            })
            .collect();
        let mut gu: Vec<Array1<f32>> = g.iter().map(|v| Array1::zeros(v.len())).collect();
        let mut g = g;
        self.graph
            .dag_alpha_backward_pass_with_intermediates(
                &self.input,
                &self.node_bounds,
                &self.exec_order,
                OUTPUT_DIM,
                self.input.flatten().len(),
                &self.relu_name_to_idx,
                st,
                None,
                &mut g,
                &mut gu,
                None,
                None,
                None,
                None,
            )
            .expect("alpha fold with intermediates")
    }
}

/// Unit seed row `e_row` — the fixture's fold is identity-seeded, so seed row
/// `r` is output row `r` and `f(x*) = output_r(x*)`.
fn unit_objective(row: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; OUTPUT_DIM];
    c[row] = 1.0;
    c
}

/// THE test. Theorem 1 must reproduce `f(x*) - B(x*)` from the per-neuron
/// decomposition, on every seed row of a real certified conv+residual fold.
#[test]
fn theorem_1_identity_holds_on_the_conv_residual_fold() {
    let h = Harness::new();
    let st = h.interior_alpha_state(false);
    let (_bounds, inter) = h.fold_with_intermediates(&st);

    let mut worst_rel = 0.0f64;
    for row in 0..OUTPUT_DIM {
        let attr = h
            .graph
            .attribute_row_gap(&h.input, &st, &inter, row, &unit_objective(row))
            .expect("attribution");

        let diff = attr.f_x_star - attr.bound_at_x_star;
        // The decomposition sums thousands of f32-sourced terms, so judge it
        // relative to the magnitudes involved rather than absolutely.
        let scale = attr
            .f_x_star
            .abs()
            .max(attr.bound_at_x_star.abs())
            .max(attr.sum_g.abs())
            .max(1.0);
        let rel = attr.residual.abs() / scale;
        worst_rel = worst_rel.max(rel);

        // Corollary 2: the gap is non-negative — the true value at x* can
        // never be below the bound the pass certified there.
        assert!(
            diff >= -1e-3 * scale,
            "row {row}: f(x*)={} is below B(x*)={}, which contradicts soundness",
            attr.f_x_star,
            attr.bound_at_x_star
        );
        assert!(
            attr.sum_g >= -1e-9,
            "row {row}: sum_j g_j = {} must be non-negative",
            attr.sum_g
        );
        // Measured worst relative residual on this fixture is 6.13e-7 (f32
        // round-off). 1e-5 is a real regression guard, not a rubber stamp.
        attr.verify_identity(1e-5 * scale).unwrap_or_else(|e| {
            panic!("row {row}: {e}\n  f(x*)={:.9e}\n  B(x*)={:.9e}\n  diff={:.9e}\n  sum_g={:.9e}\n  residual={:.9e}",
                attr.f_x_star, attr.bound_at_x_star, diff, attr.sum_g, attr.residual)
        });
    }
    eprintln!("[theorem-1] worst relative residual across {OUTPUT_DIM} rows: {worst_rel:.3e}");
}

/// The same identity must survive a channel-shared alpha layout
/// (#channel-alpha-grad), because that is what `full_conv_alpha: false` ships.
#[test]
fn theorem_1_identity_holds_under_channel_shared_alpha() {
    let h = Harness::new();
    let st = h.interior_alpha_state(true);
    let (_bounds, inter) = h.fold_with_intermediates(&st);
    for row in [0usize, 4, 9] {
        let attr = h
            .graph
            .attribute_row_gap(&h.input, &st, &inter, row, &unit_objective(row))
            .expect("attribution");
        let scale = attr
            .f_x_star
            .abs()
            .max(attr.bound_at_x_star.abs())
            .max(attr.sum_g.abs())
            .max(1.0);
        attr.verify_identity(1e-5 * scale)
            .unwrap_or_else(|e| panic!("row {row}: {e}"));
    }
}

/// `x*` must actually be the argmin corner: `B(x*)` should agree with the
/// fold's certified row lower bound, up to the fold's outward rounding, and
/// never be BELOW it by more than that. This is what licenses calling
/// `bound_at_x_star` "lb_nominal" in Corollary 4.
#[test]
fn x_star_reproduces_the_folds_certified_lower_bound() {
    let h = Harness::new();
    let st = h.interior_alpha_state(false);
    let (bounds, inter) = h.fold_with_intermediates(&st);
    for row in 0..OUTPUT_DIM {
        let attr = h
            .graph
            .attribute_row_gap(&h.input, &st, &inter, row, &unit_objective(row))
            .expect("attribution");
        let lb_sound = bounds.lower()[[row]] as f64;
        let e = attr.certified_error(lb_sound);
        let scale = lb_sound.abs().max(1.0);
        // E >= 0: the sound fold is never tighter than the nominal corner
        // evaluation. A negative E would mean the fold reported a bound it
        // cannot support at x*.
        assert!(
            e >= -1e-3 * scale,
            "row {row}: certified error E = {e} is negative \
             (B(x*)={}, lb_sound={lb_sound}) — the fold claims more than x* allows",
            attr.bound_at_x_star
        );
    }
}

#[test]
fn malformed_upper_pre_relu_width_returns_shape_error_instead_of_panicking() {
    let h = Harness::new();
    let st = h.interior_alpha_state(false);
    let (_bounds, mut inter) = h.fold_with_intermediates(&st);
    let (pre_l, pre_u) = inter
        .pre_relu_bounds
        .get("relu0")
        .expect("relu0 pre-activation bounds")
        .clone();
    assert!(pre_u.len() > 1);
    let short_upper = Array1::from_iter(pre_u.iter().take(pre_u.len() - 1).copied());
    inter
        .pre_relu_bounds
        .insert("relu0".to_string(), (pre_l, short_upper));

    let err = h
        .graph
        .attribute_row_gap(&h.input, &st, &inter, 0, &unit_objective(0))
        .expect_err("malformed intermediate widths must fail closed");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

/// Corollary 2 as a usable ceiling, and the §3.3 concentration statistic that
/// decides whether Deductions 1-2 or Deduction 3 is the load-bearing claim.
#[test]
fn attribution_reports_a_usable_ceiling_and_concentration() {
    let h = Harness::new();
    let st = h.interior_alpha_state(false);
    let (bounds, inter) = h.fold_with_intermediates(&st);
    let row = 0usize;
    let attr = h
        .graph
        .attribute_row_gap(&h.input, &st, &inter, row, &unit_objective(row))
        .expect("attribution");
    let lb_sound = bounds.lower()[[row]] as f64;

    // Every live neuron is unstable: a stable neuron's relaxation is exact, so
    // it can carry no attributed gap. This is Theorem 1's last clause.
    assert!(
        attr.live_neurons() <= attr.unstable_neurons(),
        "live {} exceeds unstable {} — a stable neuron was attributed gap",
        attr.live_neurons(),
        attr.unstable_neurons()
    );

    // The ceiling is finite and the depth estimate is consistent with it:
    // asking for exactly the available gap must be reachable, asking for more
    // must not be.
    assert!(attr.sum_g.is_finite());
    if attr.sum_g > 0.0 {
        assert!(attr.attribution_depth(attr.sum_g * 0.5).is_some());
        assert_eq!(attr.attribution_depth(attr.sum_g * 2.0), None);
    }

    // The decisive statistic for the theory doc's Sec 3.3: how many neurons
    // must be split before half / ninety percent of the available gap is
    // covered. A small number supports attribution-directed branching
    // (Deductions 1-2); a number near `live` supports budget redirection
    // (Deduction 3) instead.
    eprintln!(
        "[concentration] row {row}: neurons for 50% of gap = {:?}, for 90% = {:?}, live = {}",
        attr.attribution_depth(attr.sum_g * 0.5),
        attr.attribution_depth(attr.sum_g * 0.9),
        attr.live_neurons(),
    );
    eprintln!(
        "[ceiling] row {row}: lb_sound={:.6} B(x*)={:.6} f(x*)={:.6} sum_g={:.6} E={:.6} \
         live={} unstable={} verdict={:?}",
        lb_sound,
        attr.bound_at_x_star,
        attr.f_x_star,
        attr.sum_g,
        attr.certified_error(lb_sound),
        attr.live_neurons(),
        attr.unstable_neurons(),
        attr.classify(lb_sound, 0.0),
    );
}

// ===========================================================================
// Tier 3 — compact exact margin seed
// ===========================================================================

/// A small conjunctive spec over the fixture's 10 outputs: `y_0 - y_i >= 0`.
/// Same shape as cifar100's 99 margin rows, three orders smaller.
fn margin_spec() -> crate::bounds::AlphaSpecAscent {
    let rows: Vec<crate::bounds::AlphaSpecEarlyExit> = (1..5)
        .map(|i| {
            let mut c = vec![0.0f32; OUTPUT_DIM];
            c[0] = 1.0;
            c[i] = -1.0;
            crate::bounds::AlphaSpecEarlyExit {
                objective: c,
                threshold: 0.0,
                verify_upper_bound: false,
            }
        })
        .collect();
    crate::bounds::AlphaSpecAscent::new(rows).expect("valid margin spec")
}

/// Direct C-matrix seeding is exactly affine-head-equivalent and must preserve
/// Theorem 1 without cloning the graph.
#[test]
fn theorem_1_identity_holds_through_the_exact_margin_seed() {
    let h = Harness::new();
    let st = h.interior_alpha_state(false);
    let spec = margin_spec();
    // Compact carriers must preserve arbitrary original-row identity/order.
    let selected = [3usize, 0, 2];
    let seed = margin_probe_matrix(&spec, &selected, OUTPUT_DIM, None).expect("margin seed");

    // The custom seed must leave graph identity and bound storage untouched.
    let graph_order_before = h.graph.node_order.clone();
    let graph_output_before = h.graph.output_node.clone();
    let bound_storage_before: HashMap<&str, (*const f32, *const f32)> = h
        .node_bounds
        .iter()
        .map(|(name, bounds)| {
            (
                name.as_str(),
                (bounds.lower().as_ptr(), bounds.upper().as_ptr()),
            )
        })
        .collect();
    let deadline = Instant::now() + std::time::Duration::from_secs(30);
    let (bounds, inter) = h
        .graph
        .dag_alpha_backward_pass_with_intermediates_and_exact_seed(
            &h.input,
            &h.node_bounds,
            &h.exec_order,
            OUTPUT_DIM,
            h.input.flatten().len(),
            &h.relu_name_to_idx,
            &st,
            None,
            deadline,
            &seed,
        )
        .expect("margin-seeded fold");
    assert_eq!(h.graph.node_order, graph_order_before);
    assert_eq!(h.graph.output_node, graph_output_before);
    for (name, bounds) in &h.node_bounds {
        assert_eq!(
            bound_storage_before[name.as_str()],
            (bounds.lower().as_ptr(), bounds.upper().as_ptr()),
            "exact seed must not clone or replace bound storage for '{name}'"
        );
    }

    let lower: Vec<f32> = bounds.lower().iter().copied().collect();
    assert_eq!(
        lower.len(),
        selected.len(),
        "fold width is the margin width"
    );

    for row in 0..selected.len() {
        let objective: Vec<f32> = seed.row(row).iter().copied().collect();
        let attr = h
            .graph
            .attribute_row_gap(&h.input, &st, &inter, row, &objective)
            .unwrap_or_else(|e| panic!("row {row}: {e}"));
        let scale = attr
            .f_x_star
            .abs()
            .max(attr.bound_at_x_star.abs())
            .max(attr.sum_g.abs())
            .max(1.0);
        attr.verify_identity(1e-5 * scale)
            .unwrap_or_else(|e| panic!("margin row {row}: {e}"));

        // Corollary 2 at a real threshold: the ceiling must be consistent with
        // the classification. Whichever branch fires, it must agree with the
        // f(x*) vs threshold comparison.
        let lb_sound = f64::from(lower[row]);
        let t = f64::from(spec.rows[selected[row]].threshold);
        match attr.classify(lb_sound, t) {
            GapVerdict::Falsified { true_value, .. } => {
                assert!(true_value < t, "Falsified must mean f(x*) < threshold")
            }
            GapVerdict::ArithmeticLimited { .. } | GapVerdict::RelaxationLimited { .. } => {
                assert!(
                    attr.f_x_star >= t,
                    "a non-falsified verdict must mean f(x*) >= threshold"
                )
            }
        }
    }
}

/// The exact seed must refuse rather than guess when the spec does not match
/// the network's output width.
#[test]
fn margin_seed_refuses_a_width_mismatch() {
    let bad = crate::bounds::AlphaSpecAscent::new(vec![crate::bounds::AlphaSpecEarlyExit {
        objective: vec![1.0; OUTPUT_DIM + 3],
        threshold: 0.0,
        verify_upper_bound: false,
    }])
    .expect("constructible");
    assert!(margin_probe_matrix(&bad, &[0], OUTPUT_DIM, None).is_err());
}

#[test]
fn probe_row_count_is_hard_capped_before_dense_capture() {
    assert_eq!(probe_row_count(99, 99), GAP_PROBE_MAX_ROWS);
    assert_eq!(probe_row_count(2, 99), 2);
    assert_eq!(probe_row_count(0, 99), 1);
    assert_eq!(probe_row_count(3, 2), 2);
    assert_eq!(probe_row_count(3, 0), 0);
}

#[test]
fn exact_seed_face_refuses_oversized_or_delta_bearing_requests_before_fold() {
    let h = Harness::new();
    let clean = h.interior_alpha_state(false);
    let deadline = Instant::now() + std::time::Duration::from_secs(30);
    let oversized = Array2::zeros((GAP_PROBE_MAX_ROWS + 1, OUTPUT_DIM));
    assert!(h
        .graph
        .dag_alpha_backward_pass_with_intermediates_and_exact_seed(
            &h.input,
            &h.node_bounds,
            &h.exec_order,
            OUTPUT_DIM,
            h.input.flatten().len(),
            &h.relu_name_to_idx,
            &clean,
            None,
            deadline,
            &oversized,
        )
        .is_err());

    let mut delta = clean;
    let node = FIXTURE_RELUS[0];
    let width = delta.alpha(node).expect("fixture alpha").len();
    delta.spec_slot_rows = vec![0];
    delta
        .spec_deltas
        .insert(node.to_string(), Array2::zeros((1, width)));
    let one = Array2::zeros((1, OUTPUT_DIM));
    assert!(h
        .graph
        .dag_alpha_backward_pass_with_intermediates_and_exact_seed(
            &h.input,
            &h.node_bounds,
            &h.exec_order,
            OUTPUT_DIM,
            h.input.flatten().len(),
            &h.relu_name_to_idx,
            &delta,
            None,
            deadline,
            &one,
        )
        .is_err());
}

#[test]
fn probe_deadline_min_composes_private_and_global_authority() {
    let now = Instant::now();
    let private = std::time::Duration::from_secs(5);
    let early_global = now + std::time::Duration::from_secs(2);
    assert_eq!(
        compose_probe_deadline(now, private, Some(early_global), true),
        Some(early_global)
    );
    assert_eq!(
        compose_probe_deadline(now, private, None, true),
        None,
        "scored attribution must not mint time without caller authority"
    );
    assert_eq!(
        compose_probe_deadline(now, private, None, false),
        Some(now + private),
        "standalone diagnostics remain privately bounded"
    );
    let expired = Some(
        now.checked_sub(std::time::Duration::from_millis(1))
            .expect("system uptime exceeds one millisecond"),
    );
    let point = vec![0.0; OUTPUT_DIM];
    assert!(rank_margin_probe_rows(&margin_spec(), &point, &point, 3, expired).is_err());
    assert!(margin_probe_matrix(&margin_spec(), &[0], OUTPUT_DIM, expired).is_err());
}

#[test]
fn expired_attribution_deadline_refuses_before_substantive_work() {
    let h = Harness::new();
    let state = h.interior_alpha_state(false);
    // Deliberately malformed state/row proves deadline authority is checked
    // before shape inspection, witness allocation, or concrete dispatch.
    let empty = GraphAlphaCrownIntermediate::new();
    let expired = Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    let result =
        h.graph
            .attribute_row_gap_until(&h.input, &state, &empty, usize::MAX, &[], expired);
    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
}

#[test]
fn attribution_uses_an_exact_honestly_static_shared_checkpoint() {
    let h = Harness::new();
    let mut state = h.interior_alpha_state(false);
    let node = FIXTURE_RELUS[0];
    let base = state.alpha(node).expect("base alpha").clone();
    state.spec_slot_rows = vec![0];
    state
        .spec_deltas
        .insert(node.to_string(), Array2::from_elem((1, base.len()), 0.125));
    let upper = Array1::from_elem(base.len(), 0.875);
    state.alphas_upper.insert(node.to_string(), upper.clone());
    assert!(
        state.has_spec_deltas(),
        "fixture must exercise the delta arm"
    );
    assert_ne!(
        state.alpha_for_row(node, 0).expect("delta row").as_array(),
        base.view(),
        "nonzero delta must make the original per-row view differ"
    );

    let shared =
        settled_shared_probe_alpha(&state, false).expect("delta state needs a base-only view");
    assert!(!shared.has_spec_deltas());
    assert_eq!(shared.alpha(node).expect("shared base"), &base);
    assert_eq!(
        shared
            .alpha_for_row(node, 0)
            .expect("shared row")
            .as_array(),
        base.view(),
        "the static probe must decompose its settled shared checkpoint exactly"
    );

    let upper_shared =
        settled_shared_probe_alpha(&state, true).expect("upper path needs an oriented view");
    assert!(!upper_shared.has_spec_deltas());
    assert_eq!(
        upper_shared.alpha(node).expect("oriented upper base"),
        &upper,
        "negated upper objectives must use the checkpoint's upper alpha, not lower alpha"
    );
    assert_ne!(upper, base, "fixture must be asymmetric across paths");
}

#[test]
fn margin_probe_selects_the_worst_property_rows_with_stable_ties() {
    let spec = margin_spec();
    let mut point = vec![0.0f32; OUTPUT_DIM];
    // Rows are y0-y1 .. y0-y4. Their slacks are -3, -1, -4, -2.
    point[1] = 3.0;
    point[2] = 1.0;
    point[3] = 4.0;
    point[4] = 2.0;
    assert_eq!(
        rank_margin_probe_rows(&spec, &point, &point, 3, None).expect("rank rows"),
        vec![2, 0, 3]
    );

    // A malformed row box refuses the whole advisory selection.
    assert!(rank_margin_probe_rows(&spec, &point[..OUTPUT_DIM - 1], &point, 3, None).is_err());
}

#[test]
fn margin_seed_orients_upper_properties_as_exact_lower_properties() {
    let mut objective = vec![0.0f32; OUTPUT_DIM];
    objective[0] = 1.0;
    objective[1] = -1.0;
    let upper = crate::bounds::AlphaSpecAscent::new(vec![crate::bounds::AlphaSpecEarlyExit {
        objective: objective.clone(),
        threshold: 0.25,
        verify_upper_bound: true,
    }])
    .expect("upper spec");
    let actual = margin_probe_matrix(&upper, &[0], OUTPUT_DIM, None).expect("margin seed");
    let expected = Array2::from_shape_vec(
        (1, OUTPUT_DIM),
        objective.into_iter().map(|value| -value).collect(),
    )
    .expect("weight shape");
    assert_eq!(actual, expected);
}

// ===========================================================================
// Tier 4 — the branching prior (#attr-branch)
// ===========================================================================

#[test]
fn prior_keeps_each_row_separate_rather_than_blending() {
    // Blending was the first design and it was WRONG: aggregating six binding
    // rows on cifar100 put 1181 of 1366 neurons above zero, washing out the
    // per-row concentration (d50 = 8) the prior exists to exploit. Each row
    // must keep its own profile, filed under its own spec-row index.
    let a = attribution(vec![("n", vec![900.0, 100.0])], 1000.0, 0.0);
    let b = attribution(vec![("n", vec![0.0, 1.0])], 1.0, 0.0);
    let prior = build_attribution_prior(&[a, b], &[48, 8]);

    assert_eq!(prior.len(), 2, "one entry per spec row");
    // Row 48 keeps its own shape, normalised by its own sum_g.
    assert!(
        (prior[&48].score("n", 0).unwrap() - 0.9).abs() < 1e-12,
        "{:?}",
        prior[&48]
    );
    assert!((prior[&48].score("n", 1).unwrap() - 0.1).abs() < 1e-12);
    // Row 8's inert neuron stays inert HERE even though it is the top neuron
    // of row 48 -- which is exactly the signal blending destroyed.
    assert!(
        (prior[&8].score("n", 0).unwrap() - 0.0).abs() < 1e-12,
        "{:?}",
        prior[&8]
    );
    assert!((prior[&8].score("n", 1).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn prior_keeps_complete_zero_gap_rows_as_all_zero_ties() {
    let zero = attribution(vec![("n", vec![0.0, 0.0])], 0.0, 0.0);
    let good = attribution(vec![("n", vec![1.0, 3.0])], 4.0, 0.0);
    let prior = build_attribution_prior(&[zero, good], &[1, 2]);
    assert_eq!(prior[&1].score("n", 0), Some(0.0));
    assert_eq!(prior[&1].score("n", 1), Some(0.0));
    assert!((prior[&2].score("n", 0).unwrap() - 0.25).abs() < 1e-12);
    assert!((prior[&2].score("n", 1).unwrap() - 0.75).abs() < 1e-12);
}

/// Publish/lookup exercises a PROCESS-GLOBAL, so it lives in one test rather
/// than several that would race under the parallel test harness.
#[test]
fn prior_owner_lives_through_consumers_while_diagnostic_scope_is_independent() {
    clear_attribution_prior();
    let stale = attribution(vec![("stale", vec![1.0])], 1.0, 0.0);
    publish_attribution_prior(build_attribution_prior(&[stale], &[1]));
    let deadline = Instant::now() + std::time::Duration::from_secs(1);
    let guard = attribution_run_guard_if_until(true, true, Some(deadline))
        .expect("guard acquisition")
        .expect("armed run guard");
    assert_eq!(attribution_run_deadline(), Some(deadline));
    assert!(
        !attribution_prior_published(),
        "run acquisition must clear stale publication"
    );
    assert!(!attribution_prior_published());
    assert!(!attribution_prior_has_row(48));
    assert_eq!(attribution_prior_score(48, "relu0", 0), None);

    let attr = attribution(vec![("relu0", vec![0.0, 2.5, 1.5])], 4.0, 0.0);
    publish_attribution_prior(build_attribution_prior(&[attr], &[48]));

    let contender = std::thread::spawn(|| {
        let deadline = Instant::now() + std::time::Duration::from_millis(5);
        attribution_run_guard_if_until(true, true, Some(deadline)).is_err()
    });
    assert!(
        contender.join().expect("contender thread"),
        "a competing armed run must expire rather than proceed unguarded"
    );

    let diagnostic = std::thread::spawn(|| {
        // Diagnostic-only scope must ignore the held prior mutex, including an
        // already-expired boundary: it carries authority to the contained
        // probe, which refuses internally without changing the verifier API.
        let expired = Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .expect("system uptime exceeds one millisecond");
        let scope = attribution_run_guard_if_until(true, false, Some(expired))
            .expect("diagnostic scope cannot fail on contention/deadline")
            .expect("armed diagnostic scope");
        assert_eq!(attribution_run_deadline(), Some(expired));
        drop(scope);
        assert_eq!(attribution_run_deadline(), None);
    });
    diagnostic.join().expect("diagnostic thread");
    assert!(
        attribution_prior_has_row(48),
        "diagnostic-only scope must not clear the prior owner's publication"
    );
    assert_eq!(
        attribution_run_deadline(),
        Some(deadline),
        "diagnostic thread must not replace the prior owner's deadline scope"
    );

    // This models evaluate_root returning Continue: the publication must stay
    // live because ownership belongs to the still-active OUTER verifier.
    assert!(attribution_prior_published());
    assert!(attribution_prior_has_row(48));
    assert!(
        !attribution_prior_has_row(7),
        "an unattributed row has no prior"
    );

    assert_eq!(attribution_prior_score(48, "relu0", 1), Some(0.625));
    // A covered neuron with zero attributed gap reports 0.0 -- that IS an
    // opinion (Deduction 1: it carries none of this row's gap).
    assert_eq!(attribution_prior_score(48, "relu0", 0), Some(0.0));
    // Unknown row / neuron / node all report None -- "no opinion", which
    // callers must not conflate with zero.
    assert_eq!(attribution_prior_score(7, "relu0", 1), None);
    assert_eq!(attribution_prior_score(48, "relu0", 99), None);
    assert_eq!(attribution_prior_score(48, "nope", 0), None);

    // The outer verification/BaB scope ends here. Every return path drops this
    // owner and clears both the publication and its deadline authority.
    drop(guard);
    assert!(!attribution_prior_published());
    assert_eq!(attribution_run_deadline(), None);
}
