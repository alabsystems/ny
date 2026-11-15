// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! #cctsdb-readout-continuity pin: every layer input of the admitted patch-1
//! load must have a producer.
//!
//! The loader drops exact-identity casts (FLOAT32 always; BOOL on an
//! already-BOOL operand) and erases certified expand-identity `Where` cones.
//! A drop that does not FORWARD the erased output name leaves a live consumer
//! reading a tensor nobody produces, which the graph builder reports as
//! `Skipped op '<op>': activation input '<name>' not found in tensor_to_node`
//! before degrading the op — silently losing the row's proof lane.  patch-1
//! reads four such names (`Cast_199/201/246/248`) plus the `Where` cone's, so
//! this assertion is the cheap standing guard for both canonicalizations.

mod common;

use ny_onnx::load_onnx;
use std::collections::HashSet;

use common::{benchmark_root, BenchYear};

#[test]
fn cctsdb_patch1_admitted_load_has_no_dangling_layer_inputs() {
    let path = benchmark_root(BenchYear::Vnncomp2025)
        .join("cctsdb_yolo_2023/onnx")
        .join("patch-1.onnx");
    let loaded = load_onnx(&path).expect("patch-1 loads");
    let layers = &loaded.network.layers;

    let mut produced: HashSet<String> = HashSet::new();
    for layer in layers {
        for out in &layer.outputs {
            produced.insert(out.clone());
        }
    }
    for input in &loaded.network.inputs {
        produced.insert(input.name.clone());
    }
    for name in loaded.weights.keys() {
        produced.insert(name.to_string());
    }

    let mut dangling: Vec<String> = Vec::new();
    for layer in layers {
        for input in layer.inputs.iter().filter(|i| !i.is_empty()) {
            if !produced.contains(input.as_str()) {
                dangling.push(format!(
                    "layer '{}' ({:?}) reads '{}'",
                    layer.name, layer.layer_type, input
                ));
            }
        }
    }
    assert!(
        dangling.is_empty(),
        "every layer input needs a producer after the identity-cast and \
         expand-identity canonicalizations; dangling: {dangling:?}"
    );
}
