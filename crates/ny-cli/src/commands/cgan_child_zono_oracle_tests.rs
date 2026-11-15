// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Named V2 test 3 (`child_zono_matches_cz_oracle_small_box`): the dense
//! binary64 zonotope core against ny-mip's exact-dyadic constrained-zonotope
//! transformers, which with zero predicate rows ARE a plain certified
//! zonotope computed independently (exact rational arithmetic, sparse
//! storage, their own rounding into the box remainder).
//!
//! Both implement the same abstract transformer (DeepZ with
//! `lambda = u / (u - l)`), so after every step:
//!
//! * the dense box must ENCLOSE the oracle's box (it carries strictly more
//!   rounding slack), up to a relative `1e-9` for the slope differences a
//!   rounding-wider pre-activation box induces; and
//! * the two boxes must AGREE to a relative `1e-6` — a transformer defect
//!   (wrong offset, wrong slope, dropped generator) is far larger.
//!
//! Compiled only with the `mip` feature, like the other constrained-zonotope
//! modules: the merge gate must run `-p ny-cli --bin ny --features mip` for
//! this test to exist, and must require its own `... ok` line.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_mip::{
    constrained_zonotope_affine_unwired, transform_relu_unwired,
    unconstrained_zonotope_box_unwired, CertifiedBox64Limits, ConstrainedZonotope64,
    ConstrainedZonotopeAffineLimits, ReluTransformLimits,
};
use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
use ny_propagate::zono_core::{
    concretize, RowPar, StepOp, ZonoForwardOptions, ZonoNodeView, ZonoProgram,
};
use ny_propagate::{GraphNetwork, GraphNode};
use ny_tensor::BoundedTensor;

fn affine_limits() -> ConstrainedZonotopeAffineLimits {
    ConstrainedZonotopeAffineLimits {
        max_input_value_count: 10_000,
        max_output_value_count: 10_000,
        max_alpha_dim: 1_000,
        max_generator_nonzeros: 100_000,
        max_weight_elements: 1_000_000,
        max_matrix_visits: 1_000_000,
        max_interval_products: 2_000_000,
        max_constraint_count: 100_000,
        max_constraint_elements: 100_000,
    }
}

fn box_limits() -> CertifiedBox64Limits {
    CertifiedBox64Limits {
        max_values: 10_000,
        max_stored_f64: 20_000,
        max_weight_elements: 1_000_000,
        max_work_items: 10_000_000,
        max_scalar_products: 10_000_000,
    }
}

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        #[allow(clippy::cast_precision_loss)]
        let value = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        value
    }
    fn sym(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }
}

#[allow(clippy::cast_possible_truncation)]
fn layer_weights(rng: &mut Rng, rows: usize, cols: usize) -> (Array2<f32>, Array1<f32>) {
    let w = Array2::from_shape_fn((rows, cols), |_| (rng.sym() * 0.8) as f32);
    let b = Array1::from_shape_fn(rows, |_| (rng.sym() * 0.2) as f32);
    (w, b)
}

#[test]
fn child_zono_matches_cz_oracle_small_box() {
    let mut rng = Rng(0x5DEE_CE66_D1CE_4E5B);
    let dims = [4usize, 8, 6, 1];
    let mut layers = Vec::new();
    for window in dims.windows(2) {
        layers.push(layer_weights(&mut rng, window[1], window[0]));
    }
    let mut graph = GraphNetwork::new();
    let mut prev: Option<String> = None;
    for (index, (w, b)) in layers.iter().enumerate() {
        let name = format!("fc{index}");
        let layer = Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).expect("fc"));
        match prev {
            None => graph.add_node(GraphNode::from_input(name.clone(), layer)),
            Some(ref input) => {
                graph.add_node(GraphNode::new(name.clone(), layer, vec![input.clone()]));
            }
        }
        prev = Some(name.clone());
        if index + 1 < layers.len() {
            let relu = format!("relu{index}");
            graph.add_node(GraphNode::new(
                relu.clone(),
                Layer::ReLU(ReLULayer),
                vec![name],
            ));
            prev = Some(relu);
        }
    }
    graph.set_output(prev.expect("output"));

    let opts = ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: false,
        row_par: RowPar::Seq,
    };
    let mut compared = 0usize;
    let mut unstable_seen = 0usize;
    for trial in 0..24 {
        #[allow(clippy::cast_possible_truncation)]
        let (lo, hi): (Vec<f32>, Vec<f32>) = (0..dims[0])
            .map(|_| {
                let center = rng.sym();
                let half = 0.02 + 0.3 * rng.unit();
                ((center - half) as f32, (center + half) as f32)
            })
            .unzip();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[dims[0]]), lo.clone()).expect("lo"),
            ArrayD::from_shape_vec(IxDyn(&[dims[0]]), hi.clone()).expect("hi"),
        )
        .expect("box");
        let program = ZonoProgram::compile_with_ibp_shapes(&graph, &input).expect("compile");

        // Dense core: the concretised box after every step.
        let mut dense_boxes: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        let mut observer = |view: &ZonoNodeView<'_>| {
            if view.step.is_some() {
                dense_boxes.push(concretize(view.state, view.abs_g));
            }
        };
        let outcome = program
            .forward(&input, &opts, &[], None, Some(&mut observer))
            .expect("dense forward");
        unstable_seen += outcome.relus.iter().map(|r| r.unstable).sum::<usize>();

        // Oracle: the same chain through the exact-dyadic transformers.
        let lo64: Vec<f64> = lo.iter().map(|&v| f64::from(v)).collect();
        let hi64: Vec<f64> = hi.iter().map(|&v| f64::from(v)).collect();
        let mut cz =
            ConstrainedZonotope64::from_certified_bounds(&lo64, &hi64, &vec![false; dims[0]])
                .expect("oracle box");
        let mut layer_index = 0usize;
        for (step_index, step) in program.steps().iter().enumerate() {
            cz = match &step.op {
                StepOp::Affine(_) => {
                    let (w, b) = &layers[layer_index];
                    layer_index += 1;
                    let w64 = w.mapv(f64::from);
                    let b64: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
                    constrained_zonotope_affine_unwired(&cz, w64.view(), &b64, affine_limits())
                        .expect("oracle affine")
                        .0
                }
                StepOp::Relu => transform_relu_unwired(&cz, ReluTransformLimits::default())
                    .expect("oracle relu"),
                other => panic!("unexpected step {other:?} in the oracle MLP"),
            };
            let (oracle_box, _) =
                unconstrained_zonotope_box_unwired(&cz, box_limits()).expect("oracle box hull");
            let (dense_lo, dense_hi) = &dense_boxes[step_index];
            for (j, (&o_lo, &o_hi)) in oracle_box
                .lower()
                .iter()
                .zip(oracle_box.upper())
                .enumerate()
            {
                let scale = 1.0 + o_lo.abs().max(o_hi.abs());
                assert!(
                    dense_lo[j] <= o_lo + 1e-9 * scale && dense_hi[j] >= o_hi - 1e-9 * scale,
                    "trial {trial} step {step_index}[{j}]: dense [{}, {}] does not enclose \
                     the oracle [{o_lo}, {o_hi}]",
                    dense_lo[j],
                    dense_hi[j]
                );
                assert!(
                    (dense_lo[j] - o_lo).abs() <= 1e-6 * scale
                        && (dense_hi[j] - o_hi).abs() <= 1e-6 * scale,
                    "trial {trial} step {step_index}[{j}]: dense [{}, {}] disagrees with the \
                     oracle [{o_lo}, {o_hi}]",
                    dense_lo[j],
                    dense_hi[j]
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 0, "nothing compared: VOID");
    assert!(
        unstable_seen > 0,
        "no unstable ReLU in any trial: the ReLU oracle is VOID"
    );
}
