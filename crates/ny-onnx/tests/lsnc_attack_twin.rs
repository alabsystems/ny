// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-attack-twin — the point-folded ATTACK twin on the SHIPPED
//! `lsnc_relu` model (2026 corpus, `1.0`).
//!
//! The unit tests in `ny_onnx::attack_twin` pin the mechanism on a synthetic
//! Lyapunov cone; this file pins the facts only the real graph can:
//!
//! 1. the real graph, loaded exactly as `ny-cli` loads it, is OUTSIDE the exact
//!    point-VJP fragment even with the exact-reductions widening (its constant
//!    `MatMul` cones fail closed) — the motivating refusal;
//! 2. the twin folds exactly the four COMPUTED frontier tensors (the two
//!    equilibrium-network outputs, the Lyapunov matrix `P`, and `0 @ P`) and is
//!    INSIDE the widened fragment;
//! 3. the twin's forward is the authored network's ONNX Runtime forward (fold
//!    fidelity), and its point gradient matches a kink-gated central difference
//!    of that ORT forward on `Y_7` (the field counterexamples' clause) and on
//!    `V = Y_1` (the band).
//!
//! Registered with `required-features = ["external-vnncomp"]`; a missing corpus
//! is a hard failure, never a skip.

use std::path::PathBuf;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::attack_twin::PointFoldedAttackTwin;
use ny_onnx::diff::{read_input_shape_maybe_gzip, OrtForward};
use ny_onnx::{
    load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig,
    ShapeInferBackend,
};

mod common;
use common::{benchmark_root, BenchYear};

/// The official 2026 input box of every `lsnc_relu` row.
const BOX_LO: [f32; 6] = [-0.7, -0.7, -1.0, -3.0, -3.0, -2.0];
const BOX_HI: [f32; 6] = [0.7, 0.7, 1.0, 3.0, 3.0, 2.0];

fn model_path() -> PathBuf {
    let path = benchmark_root(BenchYear::Vnncomp2026)
        .join("lsnc_relu/1.0/onnx/relu_quadrotor2d_state.onnx");
    assert!(
        path.is_file(),
        "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
        path.display()
    );
    path
}

fn load_config() -> OnnxLoadConfig {
    OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::InProcess)
}

/// Deterministic xorshift64* points in the official box.
struct Points(u64);

impl Points {
    fn next(&mut self) -> Vec<f32> {
        (0..6)
            .map(|d| {
                self.0 ^= self.0 >> 12;
                self.0 ^= self.0 << 25;
                self.0 ^= self.0 >> 27;
                let unit =
                    (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64;
                BOX_LO[d] + (BOX_HI[d] - BOX_LO[d]) * unit as f32
            })
            .collect()
    }
}

#[test]
fn lsnc_real_graph_is_outside_the_widened_point_vjp_fragment() {
    let model = load_onnx_with_config(model_path(), &load_config()).expect("lsnc model loads");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        })
        .expect("lsnc model lowers to a graph");
    assert!(!graph.supports_attack_point_gradient());
    assert!(
        !graph.supports_attack_point_gradient_with_exact_reductions(),
        "the real graph must stay outside even the widened fragment (first refusal: {:?}); \
         otherwise the twin is not what makes the lane live",
        graph.first_attack_point_gradient_refusal_with_exact_reductions()
    );
}

#[test]
fn lsnc_attack_twin_folds_the_constant_cones_and_matches_ort() {
    let path = model_path();
    let twin = PointFoldedAttackTwin::build(&path, 6, &load_config())
        .unwrap_or_else(|error| panic!("the lsnc twin must build: {error}"));
    assert_eq!(
        twin.folded_tensors(),
        4,
        "the two equilibrium-network outputs, P, and 0 @ P: {twin:?}"
    );
    assert!(
        twin.supports_gradient(),
        "the lsnc twin must be inside the widened point-VJP fragment (first refusal: {:?})",
        twin.first_gradient_refusal()
    );

    let (_, declared) = read_input_shape_maybe_gzip(&path, 6).expect("declared input shape");
    let shape = twin.input_shape(&declared);
    let mut ort = OrtForward::from_path(&path, 6).expect("the authored model runs under ORT");
    let mut points = Points(0x6C73_6E63_7477_696E);

    // Fold fidelity: the twin's forward IS the authored network's ORT forward.
    for _ in 0..64 {
        let x = points.next();
        let point = ArrayD::from_shape_vec(IxDyn(&shape), x.clone()).expect("point");
        let twin_y = twin.point_forward(&point, None).expect("twin forward");
        let ort_y = ort.run(&x).expect("ORT forward");
        assert_eq!(twin_y.len(), 8);
        assert_eq!(ort_y.len(), 8);
        for (j, (&t, &o)) in twin_y.iter().zip(&ort_y).enumerate() {
            let tol = 1e-4 * (1.0 + f64::from(o).abs());
            assert!(
                (f64::from(t) - f64::from(o)).abs() <= tol,
                "Y_{j} at {x:?}: twin {t} vs authored ORT {o}"
            );
        }
    }

    // Gradient: the twin's point-VJP against a kink-gated central difference of
    // the AUTHORED network's ORT forward, on Y_7 and on V = Y_1.
    let objective = |ort: &mut OrtForward, x: &[f32], out: usize| -> f64 {
        f64::from(ort.run(x).expect("ORT forward")[out])
    };
    let mut compared = 0usize;
    let mut attempted = 0usize;
    for _ in 0..16 {
        let x = points.next();
        let point = ArrayD::from_shape_vec(IxDyn(&shape), x.clone()).expect("point");
        for out in [7usize, 1] {
            let mut spec = vec![0.0_f32; 8];
            spec[out] = 1.0;
            let row = Array2::from_shape_vec((1, 8), spec).expect("spec row");
            let grad: Vec<f32> = twin
                .point_gradient(&point, &row, None)
                .expect("the twin's walk must not error")
                .expect("an admitted twin yields a gradient")
                .iter()
                .copied()
                .collect();
            assert_eq!(grad.len(), 6);
            // A central difference is EXACT on a quadratic (V) and on a linear
            // piece (Y_7 between ReLU kinks), so two step sizes agree unless a
            // step crossed a kink; that disagreement is the kink gate.
            let mut central_at = |h: f32, d: usize| -> f64 {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[d] += h;
                xm[d] -= h;
                let taken = f64::from(xp[d]) - f64::from(xm[d]);
                (objective(&mut ort, &xp, out) - objective(&mut ort, &xm, out)) / taken
            };
            for d in 0..6 {
                attempted += 1;
                let h = 1e-3 * (BOX_HI[d] - BOX_LO[d]);
                let coarse = central_at(h, d);
                let fine = central_at(0.5 * h, d);
                if (coarse - fine).abs() > 1e-2 * (1.0 + fine.abs()) {
                    continue; // a step crossed a ReLU kink
                }
                let g = f64::from(grad[d]);
                let tol = 1e-2 * (1.0 + g.abs().max(fine.abs()));
                assert!(
                    (g - fine).abs() <= tol,
                    "d Y_{out} / d X_{d} at {x:?}: twin point-VJP {g} vs ORT central \
                     difference {fine}"
                );
                compared += 1;
            }
        }
    }
    assert!(
        compared * 2 >= attempted,
        "the kink gate must leave most coordinates comparable ({compared}/{attempted})"
    );
}
