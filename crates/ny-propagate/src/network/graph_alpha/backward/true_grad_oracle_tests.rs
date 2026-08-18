// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference oracle for the TRUE per-neuron alpha gradient
//! (#cifar100 task 11 — the evidence-driven pivot after the LOCAL gradient
//! rule was measured to DEGRADE bounds in the wide-alpha subdomain ascent).
//!
//! Settles, against NY's actual CPU CROWN backward
//! (`dag_alpha_backward_pass_*`), which analytic formula equals dlb/dalpha_i
//! for the lower bound of one spec row:
//!
//! LOCAL rule — what NY's default analytic paths compute
//! (`propagate_linear_with_alpha` relu/mod.rs:631, AnalyticChain
//! backward/gradients.rs:105-110, GPU warmup `..._sound_grad`
//! ny-cuda/sound_crown.rs:1051-1057, refuted wide-alpha batched.rs):
//!
//! ```text
//!   g_i = pre_lower_i * sum_j max(A_at_relu[j,i], 0)
//! ```
//!
//! TRUE chain rule (closed form of autograd-through-the-backward, what
//! alpha-beta-CROWN effectively computes):
//!
//! ```text
//!   g_i = nu_i * hhat_i(x*)    if nu_i > 0, else 0
//! ```
//!
//! where `nu = A_at_relu` (the post-activation coefficients when the backward
//! reaches the ReLU — relaxation branch is selected per its sign,
//! relu/mod.rs:620-641), `x*` is the concretization argmin corner of the
//! FINAL row (`x*_j = xl_j` if `final_A[j] > 0` else `xu_j`,
//! concretize.rs:194-195), and `hhat_i(x*)` is the RELAXED-linear forward
//! evaluation of neuron i's pre-activation at `x*`: each EARLIER ReLU applies
//! the same per-neuron affine relaxation the backward selected
//! (`nu_m[i'] > 0` -> slope alpha, no intercept; `nu_m[i'] < 0` -> chord
//! slope u/(u-l) with its intercept; stable neurons identity/zero) — NOT the
//! concrete ReLU.
//!
//! The local rule replaces `hhat_i(x*)` by `pre_lower_i`, which has the wrong
//! SIGN whenever the relaxed pre-activation at the argmin is positive —
//! exactly why the wide-alpha ascent degraded in both lr signs.

use super::*;
use crate::layers::activations::relu_crossing_upper_chord;
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::core::GraphNode;
use ndarray::{arr1, arr2, Array1, Array2};
use std::collections::HashMap;

const W1: [[f32; 3]; 4] = [
    [0.7, -0.5, 0.4],
    [-0.6, 0.9, 0.3],
    [0.5, 0.8, -0.7],
    [-0.4, -0.6, 0.8],
];
const B1: [f32; 4] = [0.05, -0.10, 0.15, -0.05];
const W2: [[f32; 4]; 4] = [
    [0.6, -0.7, 0.5, 0.3],
    [-0.5, 0.4, -0.6, 0.7],
    [0.8, 0.3, -0.4, -0.5],
    [0.3, -0.8, 0.6, 0.4],
];
const B2: [f32; 4] = [-0.05, 0.10, -0.15, 0.20];
const W3: [[f32; 4]; 1] = [[0.9, -0.6, 0.7, -0.8]];
const B3: [f32; 1] = [0.1];

/// Input(3) -> linear1(4x3) -> relu1 -> linear2(4x4) -> relu2 -> linear3(1x4).
/// The 1-row linear3 IS the spec row, so the network lower bound is the
/// spec-row lower bound the alpha optimizers ascend.
fn build_net() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&W1), Some(arr1(&B1))).expect("linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&W2), Some(arr1(&B2))).expect("linear2")),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(arr2(&W3), Some(arr1(&B3))).expect("linear3")),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input = BoundedTensor::new(
        arr1(&[-1.0f32, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0, 1.0]).into_dyn(),
    )
    .expect("input box");
    (graph, input)
}

struct OracleHarness {
    graph: GraphNetwork,
    input: BoundedTensor,
    ibp: HashMap<String, BoundedTensor>,
    exec_order: Vec<String>,
    relu_name_to_idx: HashMap<String, usize>,
}

impl OracleHarness {
    fn new() -> Self {
        let (graph, input) = build_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP forward");
        let exec_order = graph.exec_order().expect("exec_order").to_vec();
        let relu_name_to_idx: HashMap<String, usize> =
            [("relu1".to_string(), 0), ("relu2".to_string(), 1)]
                .into_iter()
                .collect();
        Self {
            graph,
            input,
            ibp,
            exec_order,
            relu_name_to_idx,
        }
    }

    fn make_alpha_state(&self, a1: &[f32; 4], a2: &[f32; 4]) -> GraphAlphaState {
        let mut st = GraphAlphaState::new();
        for name in ["relu1", "relu2"] {
            let pre = self
                .graph
                .relu_preactivation_bounds(name, &self.input, &self.ibp, "oracle")
                .expect("pre bounds");
            st.add_relu_node(name, pre, false).expect("alpha init");
        }
        *st.alphas.get_mut("relu1").expect("relu1 alpha") = arr1(a1);
        *st.alphas.get_mut("relu2").expect("relu2 alpha") = arr1(a2);
        st
    }

    /// The spec-row lower bound through NY's ACTUAL alpha backward.
    fn lower_bound(&self, st: &GraphAlphaState) -> f32 {
        let mut gradients = vec![Array1::<f32>::zeros(4), Array1::<f32>::zeros(4)];
        let mut gradients_upper = vec![Array1::<f32>::zeros(4), Array1::<f32>::zeros(4)];
        let bounds = self
            .graph
            .dag_alpha_backward_pass_with_engine(
                &self.input,
                &self.ibp,
                &self.exec_order,
                1,
                3,
                &self.relu_name_to_idx,
                st,
                None,
                &mut gradients,
                &mut gradients_upper,
                None,
                None,
                None,
                None,
            )
            .expect("alpha backward");
        bounds.lower()[[0]]
    }

    /// Central finite difference dlb/dalpha through the actual backward.
    fn fd_grad(&self, st: &GraphAlphaState, relu: &str, i: usize, h: f32) -> f32 {
        let base = st.alphas.get(relu).expect("relu alphas")[i];
        assert!(
            base - h > 0.0 && base + h < 1.0,
            "FD probe must stay interior to [0,1]"
        );
        let mut plus = st.clone();
        plus.alphas.get_mut(relu).expect("relu alphas")[i] = base + h;
        let mut minus = st.clone();
        minus.alphas.get_mut(relu).expect("relu alphas")[i] = base - h;
        (self.lower_bound(&plus) - self.lower_bound(&minus)) / (2.0 * h)
    }
}

/// Per-neuron relaxed multiplier/intercept selected by the backward for the
/// LOWER row: `nu > 0` -> (alpha, 0); `nu < 0` -> chord; stable -> identity/0.
fn relaxed_slope_intercept(l: f32, u: f32, nu: f32, alpha: f32) -> (f32, f32) {
    if l >= 0.0 {
        (1.0, 0.0)
    } else if u <= 0.0 {
        (0.0, 0.0)
    } else if nu > 0.0 {
        (alpha, 0.0)
    } else {
        // nu <= 0: the exact chord the backward used (activations/mod.rs).
        relu_crossing_upper_chord(l, u, None)
    }
}

fn matvec(w: &[[f32; 4]], h: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(h.iter()).map(|(&a, &b)| a as f64 * b).sum())
        .collect()
}

/// Both analytic candidates for every neuron of both ReLU layers, computed
/// from the SAME intermediates NY's backward stored.
struct Candidates {
    local: [Array1<f32>; 2],
    truth: [Array1<f32>; 2],
    /// Relaxed pre-activation value at x* per layer (diagnostics + flip check).
    hhat: [Vec<f64>; 2],
    /// In-place gradients filled by `propagate_linear_with_alpha` (production
    /// local rule) — must match `local` exactly.
    inplace: Vec<Array1<f32>>,
}

fn compute_candidates(h: &OracleHarness, st: &GraphAlphaState) -> Candidates {
    let mut gradients = vec![Array1::<f32>::zeros(4), Array1::<f32>::zeros(4)];
    let mut gradients_upper = vec![Array1::<f32>::zeros(4), Array1::<f32>::zeros(4)];
    let (_bounds, inter) = h
        .graph
        .dag_alpha_backward_pass_with_intermediates(
            &h.input,
            &h.ibp,
            &h.exec_order,
            1,
            3,
            &h.relu_name_to_idx,
            st,
            None,
            &mut gradients,
            &mut gradients_upper,
            None,
            None,
            None,
            None,
        )
        .expect("alpha backward with intermediates");

    // nu at each ReLU: the stored lower A row (1 spec row -> row 0).
    let nu1 = inter
        .a_at_relu("relu1")
        .expect("A at relu1")
        .row(0)
        .to_owned();
    let nu2 = inter
        .a_at_relu("relu2")
        .expect("A at relu2")
        .row(0)
        .to_owned();
    let (pre1_l, pre1_u) = inter.pre_relu_bounds("relu1").expect("pre1").clone();
    let (pre2_l, pre2_u) = inter.pre_relu_bounds("relu2").expect("pre2").clone();

    // x*: concretization argmin corner of the FINAL row (concretize.rs:194-195:
    // lb += A>0 ? A*xl : A*xu).
    let final_a: Array2<f32> = inter.final_bounds.lower_a().clone();
    assert_eq!(final_a.nrows(), 1, "single spec row");
    let xl = h.input.lower();
    let xu = h.input.upper();
    let x_star: Vec<f64> = (0..3)
        .map(|j| {
            let a = final_a[[0, j]];
            assert!(a != 0.0, "test net must avoid zero final coefficients");
            if a > 0.0 {
                xl[[j]] as f64
            } else {
                xu[[j]] as f64
            }
        })
        .collect();

    // Relaxed-linear forward at x* with the backward's selected relaxations.
    let a1 = st.alphas.get("relu1").expect("relu1 alphas");
    let z1: Vec<f64> = matvec(
        &W1.map(|r| {
            // widen 3-col rows to the matvec helper's 4-col shape
            [r[0], r[1], r[2], 0.0]
        }),
        &[x_star[0], x_star[1], x_star[2], 0.0],
    )
    .iter()
    .zip(B1.iter())
    .map(|(&s, &b)| s + b as f64)
    .collect();
    let h1: Vec<f64> = (0..4)
        .map(|i| {
            let (s, t) = relaxed_slope_intercept(pre1_l[i], pre1_u[i], nu1[i], a1[i]);
            s as f64 * z1[i] + t as f64
        })
        .collect();
    let z2: Vec<f64> = matvec(&W2, &h1)
        .iter()
        .zip(B2.iter())
        .map(|(&s, &b)| s + b as f64)
        .collect();

    let mk = |nu: &Array1<f32>,
              pre_l: &Array1<f32>,
              pre_u: &Array1<f32>,
              z: &[f64]|
     -> (Array1<f32>, Array1<f32>) {
        let mut local = Array1::<f32>::zeros(4);
        let mut truth = Array1::<f32>::zeros(4);
        for i in 0..4 {
            if !(pre_l[i] < 0.0 && pre_u[i] > 0.0) {
                continue; // stable: no alpha, both formulas zero
            }
            local[i] = pre_l[i] * nu[i].max(0.0);
            truth[i] = if nu[i] > 0.0 {
                (nu[i] as f64 * z[i]) as f32
            } else {
                0.0
            };
        }
        (local, truth)
    };
    let (local1, truth1) = mk(&nu1, &pre1_l, &pre1_u, &z1);
    let (local2, truth2) = mk(&nu2, &pre2_l, &pre2_u, &z2);

    Candidates {
        local: [local1, local2],
        truth: [truth1, truth2],
        hhat: [z1, z2],
        inplace: gradients,
    }
}

fn run_oracle(a1: [f32; 4], a2: [f32; 4]) -> (f32, f32, bool) {
    let h = OracleHarness::new();
    let st = h.make_alpha_state(&a1, &a2);
    let cand = compute_candidates(&h, &st);

    // Sanity: every neuron of both layers is unstable in this net, so alpha
    // is live everywhere (the FD probe below assumes it).
    for name in ["relu1", "relu2"] {
        let mask = st.unstable_mask.get(name).expect("mask");
        assert!(mask.iter().all(|&m| m), "{name}: all neurons unstable");
    }

    // The production in-place gradient IS the local rule (same numbers).
    for (k, name) in ["relu1", "relu2"].iter().enumerate() {
        for i in 0..4 {
            assert!(
                (cand.inplace[k][i] - cand.local[k][i]).abs() <= 1e-6,
                "{name}[{i}]: in-place grad {} != local-rule {} — the two \
                 'analytic' implementations diverged",
                cand.inplace[k][i],
                cand.local[k][i]
            );
        }
    }

    let fd_h = 5e-3f32;
    let mut max_err_true = 0.0f32;
    let mut max_err_local = 0.0f32;
    let mut sign_flip_seen = false;
    for (k, name) in ["relu1", "relu2"].iter().enumerate() {
        for i in 0..4 {
            let fd = h.fd_grad(&st, name, i, fd_h);
            let t = cand.truth[k][i];
            let l = cand.local[k][i];
            let tol = 5e-3 + 0.02 * fd.abs();
            eprintln!(
                "[oracle] {name}[{i}] fd={fd:+.5} true={t:+.5} local={l:+.5} hhat={:+.5}",
                cand.hhat[k][i]
            );
            assert!(
                (fd - t).abs() <= tol,
                "{name}[{i}]: TRUE formula {t} disagrees with finite difference {fd} (tol {tol})"
            );
            max_err_true = max_err_true.max((fd - t).abs());
            max_err_local = max_err_local.max((fd - l).abs());
            // The failure mode that refuted the local rule: relaxed
            // pre-activation at x* positive while pre_lower < 0 flips the sign.
            if t > 1e-3 && l < -1e-3 {
                sign_flip_seen = true;
            }
        }
    }
    (max_err_true, max_err_local, sign_flip_seen)
}

/// The gated point-forward surrogate, held to the finite-difference probe that
/// refuted the rule it replaces. `#envelope-grad` uses the concrete activation
/// at the concretization argmin as a proxy for `ĥ_i(x*)`, instead of the
/// constant `pre_lower[i]`; it is exact at the first ReLU and heuristic deeper.
///
/// TWO claims, and deliberately not a third:
///
/// 1. The worst-case FD error strictly improves in BOTH α settings.
/// 2. THE DECISIVE ONE — the local rule is sign-definite `<= 0`
///    (`local_rule_nonpos`, machine-checked in Lean), so wherever the true
///    derivative is POSITIVE it points exactly the wrong way and Adam walks
///    downhill. The envelope rule recovers that sign. This is the property that
///    makes the ascent able to move at all, and no step size can substitute for
///    it.
///
/// NOT claimed: per-neuron dominance. On deeper layers the degenerate-box
/// forward uses the concrete rather than the relaxed ReLU, and a neuron can come
/// out further from FD than the local rule happened to be (setting B,
/// `relu2[2]`: envelope 9.7e-1 vs local 1.5e-1). The aggregate and the sign both
/// still improve. Overstating this to "uniformly better" is the kind of claim
/// that survives review and then quietly fails in measurement.
#[ntest::timeout(60000)]
#[test]
fn fd_oracle_envelope_grad_beats_the_local_rule() {
    for (label, a1, a2) in [
        (
            "A",
            [0.30f32, 0.55, 0.45, 0.70],
            [0.35f32, 0.60, 0.50, 0.65],
        ),
        (
            "B",
            [0.62f32, 0.28, 0.71, 0.44],
            [0.57f32, 0.39, 0.66, 0.52],
        ),
    ] {
        let h = OracleHarness::new();
        let st = h.make_alpha_state(&a1, &a2);
        let cand = compute_candidates(&h, &st);

        let mut g = vec![Array1::<f32>::zeros(4), Array1::<f32>::zeros(4)];
        let mut gu = vec![Array1::<f32>::zeros(4), Array1::<f32>::zeros(4)];
        let (_b, inter) = h
            .graph
            .dag_alpha_backward_pass_with_intermediates(
                &h.input,
                &h.ibp,
                &h.exec_order,
                1,
                3,
                &h.relu_name_to_idx,
                &st,
                None,
                &mut g,
                &mut gu,
                None,
                None,
                None,
                None,
            )
            .expect("alpha backward with intermediates");

        let names = vec!["relu1".to_string(), "relu2".to_string()];
        let env = h
            .graph
            .compute_graph_chain_rule_gradients_envelope_for_test(
                &h.input, &names, &inter, &st, None,
            );
        assert_eq!(env.len(), 2, "one gradient array per ReLU");

        let (mut err_env, mut err_local) = (0.0f32, 0.0f32);
        let mut sign_recovered = false;
        for (k, name) in ["relu1", "relu2"].iter().enumerate() {
            for i in 0..4 {
                let fd = h.fd_grad(&st, name, i, 5e-3);
                let (e, l) = (env[k][i], cand.local[k][i]);
                eprintln!(
                    "[envelope-{label}] {name}[{i}] fd={fd:+.5} envelope={e:+.5} local={l:+.5}"
                );
                assert!(
                    l <= 1e-6,
                    "{name}[{i}]: the local rule is sign-definite <= 0 by construction \
                     (local_rule_nonpos); a positive {l} means the rule under test changed"
                );
                if fd > 1e-3 && l < -1e-3 && e > 1e-3 {
                    sign_recovered = true;
                }
                err_env = err_env.max((fd - e).abs());
                err_local = err_local.max((fd - l).abs());
            }
        }
        eprintln!(
            "[envelope-{label}] max|fd-envelope|={err_env:.2e} max|fd-local|={err_local:.2e}"
        );
        assert!(
            err_env < err_local,
            "setting {label}: the envelope rule must beat the local rule on \
             worst-case FD error (envelope {err_env:.2e}, local {err_local:.2e})"
        );
        assert!(
            sign_recovered,
            "setting {label}: expected at least one neuron where the true \
             derivative is positive, the local rule is negative (walks downhill) \
             and the envelope rule recovers the positive sign — that recovery is \
             the entire point of the change"
        );
    }
}

/// THE decisive oracle: central finite differences of NY's actual CPU CROWN
/// lower bound match the TRUE chain-rule formula at every unstable neuron of
/// BOTH ReLU layers, and materially disagree with the local rule (with at
/// least one SIGN flip — ascending the local gradient would walk downhill).
#[ntest::timeout(60000)]
#[test]
fn fd_oracle_true_alpha_gradient_matches_local_rule_refuted() {
    let (max_err_true, max_err_local, sign_flip) =
        run_oracle([0.30, 0.55, 0.45, 0.70], [0.35, 0.60, 0.50, 0.65]);
    eprintln!(
        "[oracle] setting A: max|fd-true|={max_err_true:.2e} max|fd-local|={max_err_local:.2e} sign_flip={sign_flip}"
    );
    assert!(
        max_err_local > 10.0 * (max_err_true + 1e-4),
        "expected the local rule to be materially wrong where the true \
         formula is right (true err {max_err_true:.2e}, local err {max_err_local:.2e})"
    );
    assert!(
        sign_flip,
        "expected at least one sign-flipped neuron (relaxed pre-activation \
         at x* positive while pre_lower negative) — the refutation mechanism"
    );

    // Second alpha setting: guard against a coincidental match.
    let (max_err_true_b, max_err_local_b, _) =
        run_oracle([0.62, 0.28, 0.71, 0.44], [0.57, 0.39, 0.66, 0.52]);
    eprintln!(
        "[oracle] setting B: max|fd-true|={max_err_true_b:.2e} max|fd-local|={max_err_local_b:.2e}"
    );
    assert!(
        max_err_local_b > 10.0 * (max_err_true_b + 1e-4),
        "setting B: local rule should stay refuted (true err \
         {max_err_true_b:.2e}, local err {max_err_local_b:.2e})"
    );
}
