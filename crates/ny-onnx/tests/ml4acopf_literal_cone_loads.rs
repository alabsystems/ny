// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression pin for the ml4acopf_2024 load gate.
//!
//! Commit 819b0554 introduced two rules that, together, made the three plain
//! `{14,118,300}_ieee_ml4acopf` models unloadable:
//!
//!   1. `reduction_schema` requires FLOAT32 comparison operands, so
//!      `standard ONNX Equal node '/Equal' has comparison operands dtype 7`
//!      refused the model before folding ran;
//!   2. `const_fold::ops::constants` restricted a `ConstantOfShape` fill value
//!      to `data_type == 1`, so even with the schema relaxed the INT64 cone
//!      could no longer fold away.
//!
//! Both are collateral on the same authored subgraph. `/Equal` is not a runtime
//! comparison at all: its complete transitive input cone is `Constant` nodes,
//! it contains no `Shape` node, and it does not touch the single graph input.
//! It is PyTorch's `Tensor.expand(1, -1)` lowering, whose exact value is
//!
//!   ConstantOfShape([2], value=INT64 1) = [1, 1]
//!   Mul([1, 1], -1)                     = [-1, -1]
//!   Equal([1, -1], [-1, -1])            = [false, true]
//!   Where([false, true], [1, 1], [1, -1]) = [1, 1]
//!   Expand(<20 float constants>, [1, 1]) = a (1, 20) float constant
//!
//! for every input in the scored box, because nothing in that chain reads an
//! input. The `Expand` result is consumed as a constant bias by `Sub`.
//!
//! This test pins the property that actually matters — the models LOAD and the
//! literal cone is GONE — on the shipped files. A unit fixture cannot do that:
//! the regression was a interaction between the raw schema gate, the folder,
//! and the INT64 control-path audit, and only the real graph exercises all
//! three.
//!
//! Cargo registers this integration target with `required-features =
//! ["external-vnncomp"]`. The default suite is hermetic; selecting the VNN-COMP
//! lane runs this test and makes a missing corpus a hard failure.

use ny_core::LayerType;
use ny_onnx::load_onnx;
mod common;
use common::{benchmark_root, BenchYear};

const MODELS: [&str; 3] = [
    "14_ieee_ml4acopf.onnx",
    "118_ieee_ml4acopf.onnx",
    "300_ieee_ml4acopf.onnx",
];

#[test]
fn ml4acopf_models_load_and_erase_their_literal_expand_cone() {
    let root = benchmark_root(BenchYear::Vnncomp2025).join("ml4acopf_2024/onnx");
    for model in MODELS {
        let path = root.join(model);
        assert!(
            path.is_file(),
            "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
            path.display()
        );

        let loaded = load_onnx(&path)
            .unwrap_or_else(|error| panic!("{model} must load, not fail closed: {error}"));
        let layers = &loaded.network.layers;

        // The whole point of the admission is that these never become layers.
        // A load that "succeeds" while keeping a Compare/Where/Expand layer over
        // reinterpreted INT64 constants is exactly what the raw gate exists to
        // prevent, so assert the erasure, not just the load.
        for forbidden in [
            LayerType::Compare,
            LayerType::Where,
            LayerType::Expand,
            LayerType::Cast,
        ] {
            let survivors = layers
                .iter()
                .filter(|layer| layer.layer_type == forbidden)
                .map(|layer| layer.name.clone())
                .collect::<Vec<_>>();
            assert!(
                survivors.is_empty(),
                "{model}: the literal `Tensor.expand` cone must be constant-folded away, \
                 but {forbidden:?} layers survived: {survivors:?}"
            );
        }

        // Sanity: the model really did load a network, not an empty shell.
        assert!(
            layers.len() > 20,
            "{model}: expected the full ACOPF graph, got {} layers",
            layers.len()
        );
    }
}
