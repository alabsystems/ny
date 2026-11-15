// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #ml4acopf-vjp — the exact point-VJP must be REACHABLE on the shipped
//! ml4acopf_2024 models, not only on hand-built fixtures.
//!
//! Before #ml4acopf-vjp every one of these nets was refused by the static gate
//! (`Attack lanes: …118_ieee_ml4acopf-linear-nonresidual.onnx is outside the
//! exact-gradient fragment`), so the post-BaB lane printed `Post-BaB attack:
//! DECLINED — no exact-gradient support on this net` and left ~38% of a 600 s
//! budget unspent on rows where eight other verifiers find counterexamples.
//! The ny-propagate unit tests (`point_vjp_ml4acopf_tests.rs`) pin EXACTNESS
//! per admitted variant on small graphs; this file pins the two facts only the
//! real graphs can: that the loader's actual lowering of every model lands
//! entirely inside the fragment (a single stray `OpaqueSkip`, broadcast shape
//! or builder-inserted node would keep the lane dead), and that
//! `attack_point_gradient` then really RUNS on the production input shape and
//! returns a finite, non-zero gradient for the outputs the property constrains.
//!
//! The graph is loaded exactly as `ny-cli`'s `load_graph_network` loads it
//! (`CompoundNodePolicy::DecomposeNormalization`, in-process shape inference as
//! in the CLI's own test configuration), and the input tensor takes the shape
//! the attack lanes give it: `read_input_shape_maybe_gzip`'s declared `[1, n]`
//! mapped through `GraphNetwork::attack_point_gradient_input_shape`.
//!
//! Cargo registers this integration target with `required-features =
//! ["external-vnncomp"]`; a missing corpus is a hard failure, never a skip.

use std::path::Path;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::diff::read_input_shape_maybe_gzip;
use ny_onnx::vnnlib::{load_vnnlib, OutputConstraint};
use ny_onnx::{
    load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig,
    ShapeInferBackend,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

mod common;
use common::{benchmark_root, BenchYear};

/// Every model of the 2026 ml4acopf_2024 corpus, each with one of its own
/// scored properties (for the input box and the constrained outputs).
const MODELS: [(&str, &str); 9] = [
    ("14_ieee_ml4acopf.onnx", "14_ieee_prop1.vnnlib"),
    (
        "14_ieee_ml4acopf-linear-nonresidual.onnx",
        "14_ieee_prop1.vnnlib",
    ),
    (
        "14_ieee_ml4acopf-linear-residual.onnx",
        "14_ieee_prop1.vnnlib",
    ),
    ("118_ieee_ml4acopf.onnx", "118_ieee_prop3.vnnlib"),
    (
        "118_ieee_ml4acopf-linear-nonresidual.onnx",
        "118_ieee_prop3.vnnlib",
    ),
    (
        "118_ieee_ml4acopf-linear-residual.onnx",
        "118_ieee_prop3.vnnlib",
    ),
    ("300_ieee_ml4acopf.onnx", "300_ieee_prop2.vnnlib"),
    (
        "300_ieee_ml4acopf-linear-nonresidual.onnx",
        "300_ieee_prop2.vnnlib",
    ),
    (
        "300_ieee_ml4acopf-linear-residual.onnx",
        "300_ieee_prop2.vnnlib",
    ),
];

/// At most this many constrained outputs are differentiated per model (the
/// 118-bus disjunctions reference hundreds; a handful proves the lane runs).
const MAX_ROWS_PER_MODEL: usize = 6;

fn corpus_dir() -> std::path::PathBuf {
    benchmark_root(BenchYear::Vnncomp2026).join("ml4acopf_2024/1.0")
}

/// `ny-cli`'s `load_graph_network`, verbatim in its options.
fn load_like_the_cli(path: &Path) -> GraphNetwork {
    assert!(
        path.is_file(),
        "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
        path.display()
    );
    let config = OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::InProcess);
    let model = load_onnx_with_config(path, &config)
        .unwrap_or_else(|error| panic!("{} must load: {error}", path.display()));
    let options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    model
        .to_graph_network_with_options(options)
        .unwrap_or_else(|error| panic!("{} must lower to a graph: {error}", path.display()))
}

/// The output index each atom of the property constrains, first-seen order.
/// Reads both the flat conjunction and every disjunctive clause, since a
/// top-level `(or ...)` property keeps its atoms in the clause list.
fn constrained_outputs<'a>(constraints: impl Iterator<Item = &'a OutputConstraint>) -> Vec<usize> {
    let mut outputs: Vec<usize> = Vec::new();
    for constraint in constraints {
        let index = match *constraint {
            OutputConstraint::LessEq(i, _)
            | OutputConstraint::GreaterEq(i, _)
            | OutputConstraint::LessThan(i, _)
            | OutputConstraint::GreaterThan(i, _)
            | OutputConstraint::LessEqConst(i, _)
            | OutputConstraint::GreaterEqConst(i, _)
            | OutputConstraint::LessThanConst(i, _)
            | OutputConstraint::GreaterThanConst(i, _) => i,
            // `OutputConstraint` is `#[non_exhaustive]`: a variant added later
            // must fail this test loudly, never be skipped silently.
            ref other => panic!("unhandled OutputConstraint variant: {other:?}"),
        };
        if !outputs.contains(&index) {
            outputs.push(index);
        }
    }
    outputs
}

/// Static admission of every shipped model. Reports EVERY refused model at
/// once, each with the node that blocks it, so one run is a full census.
#[test]
fn ml4acopf_shipped_models_are_inside_the_exact_point_vjp_fragment() {
    let onnx_dir = corpus_dir().join("onnx");
    let mut refused: Vec<String> = Vec::new();
    for (model, _) in MODELS {
        let graph = load_like_the_cli(&onnx_dir.join(model));
        if !graph.supports_attack_point_gradient() {
            refused.push(format!(
                "{model}: first refused node {:?}",
                graph.first_attack_point_gradient_refusal()
            ));
        }
    }
    assert!(
        refused.is_empty(),
        "these shipped ml4acopf models are still OUTSIDE the exact point-VJP fragment, so \
         every gradient attack lane stays statically dead on them:\n{}",
        refused.join("\n")
    );
}

/// `attack_point_gradient` RUNS on every shipped model at the centre of a real
/// property box, with the attack lanes' own input shape, and returns a finite
/// gradient of the right shape for each constrained output — non-zero for at
/// least one of them (a structurally dead path, e.g. a gradient lost at a
/// Floor or a zero-slope relaxation, would return all zeros).
#[test]
fn ml4acopf_point_vjp_runs_on_the_production_input_shape() {
    let root = corpus_dir();
    let mut failures: Vec<String> = Vec::new();
    for (model, property) in MODELS {
        let onnx_path = root.join("onnx").join(model);
        let vnnlib_path = root.join("vnnlib").join(property);
        let graph = load_like_the_cli(&onnx_path);
        let spec = load_vnnlib(&vnnlib_path)
            .unwrap_or_else(|error| panic!("{property} must parse: {error}"));
        let dim = spec.num_inputs;
        assert_eq!(
            spec.input_bounds.len(),
            dim,
            "{property}: one box per input"
        );

        let (_, declared_shape) = read_input_shape_maybe_gzip(&onnx_path, dim)
            .unwrap_or_else(|error| panic!("{model}: input shape unreadable: {error}"));
        // Exactly what `gradient_guided_falsify` / `equality_seek_falsify` hand
        // the graph: the declared ORT shape, mapped to the graph's own input
        // convention.
        let input_shape = graph.attack_point_gradient_input_shape(&declared_shape);
        assert_eq!(
            input_shape.iter().product::<usize>(),
            dim,
            "{model}: the graph-side shape {input_shape:?} (declared {declared_shape:?}) must \
             keep the flat input length"
        );
        let centre: Vec<f32> = spec
            .input_bounds
            .iter()
            .map(|&(lo, hi)| f64::midpoint(lo, hi) as f32)
            .collect();
        let x = ArrayD::from_shape_vec(IxDyn(&input_shape), centre).expect("input shape");

        let output = graph
            .propagate_concrete_point(
                &BoundedTensor::concrete(x.clone()).expect("concrete input"),
                None,
                None,
            )
            .unwrap_or_else(|error| panic!("{model}: concrete forward failed: {error}"));
        let num_outputs = output.len();
        assert_eq!(
            num_outputs, spec.num_outputs,
            "{model}: forward width must match the property's output count"
        );

        let rows = constrained_outputs(
            spec.output_constraints
                .iter()
                .chain(spec.output_constraint_clauses.iter().flatten()),
        );
        assert!(!rows.is_empty(), "{property}: no constrained output");
        let mut any_nonzero = false;
        for &k in rows.iter().take(MAX_ROWS_PER_MODEL) {
            let mut row = Array2::<f32>::zeros((1, num_outputs));
            row[[0, k]] = 1.0;
            match graph.attack_point_gradient(&x, &row, None, None) {
                Ok(Some(gradient)) => {
                    if gradient.shape() != input_shape.as_slice() {
                        failures.push(format!(
                            "{model} Y_{k}: gradient shape {:?} != input shape {input_shape:?}",
                            gradient.shape()
                        ));
                    } else if gradient.iter().any(|g| !g.is_finite()) {
                        failures.push(format!("{model} Y_{k}: non-finite gradient"));
                    } else {
                        any_nonzero |= gradient.iter().any(|&g| g != 0.0);
                    }
                }
                Ok(None) => failures.push(format!(
                    "{model} Y_{k}: attack_point_gradient declined (Ok(None)); first refused \
                     node {:?}",
                    graph.first_attack_point_gradient_refusal()
                )),
                Err(error) => failures.push(format!("{model} Y_{k}: error {error}")),
            }
        }
        if !any_nonzero {
            failures.push(format!(
                "{model}: every constrained output had an identically-zero gradient"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "the point-VJP does not run cleanly on the shipped ml4acopf models:\n{}",
        failures.join("\n")
    );
}
