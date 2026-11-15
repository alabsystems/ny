// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic census (#cgan-f64-tail B0): which LOADED-graph layers block the
//! certified f64 tail on cgan row 2. The raw onnx op census (ConvTranspose /
//! Conv / BatchNormalization / Gemm / Relu / Reshape / Constant) is all inside
//! the walker's op set, yet the armed probe printed
//! `[f64-tail] batch-seam graph_supported=false` — so the gap is introduced by
//! the LOAD pipeline (BN folds, normalization decomposition, shape baking),
//! and only a census of the loaded `GraphNetwork` can name it.
//!
//! AN EXAMPLE, NOT A TEST. It asserts nothing — it prints a census — and it
//! needs the `benchmarks/` checkout. As a `#[test]` it could only exist
//! `#[ignore]`d, which the source policy prohibits for the reason this file
//! illustrates: a diagnostic that never runs is indistinguishable from a
//! contract that never runs, and the suite reports the same green either way.
//! Run it deliberately:
//!
//! ```sh
//! targo --unverified run -p ny-cli --example f64_tail_census
//! ```

use std::path::Path;

fn census(path: &Path, decompose_norm: bool) {
    let model = ny_onnx::load_onnx_with_config(path, &ny_onnx::OnnxLoadConfig::default())
        .expect("load onnx");
    let options = if decompose_norm {
        ny_onnx::GraphNetworkOptions {
            compound_node_policy: ny_onnx::CompoundNodePolicy::DecomposeNormalization,
            ..ny_onnx::GraphNetworkOptions::default()
        }
    } else {
        ny_onnx::GraphNetworkOptions::default()
    };
    let graph = model
        .to_graph_network_with_options(options)
        .expect("to graph network");
    let unsupported = ny_propagate::network::f64_tail_unsupported_census(&graph);
    println!(
        "census model={} decompose_norm={} nodes={} unsupported={}",
        path.file_name().unwrap().to_string_lossy(),
        decompose_norm,
        graph.node_names().len(),
        unsupported.len()
    );
    for (name, layer_type) in &unsupported {
        println!("  UNSUPPORTED {name}: {layer_type}");
    }
}

fn main() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let onnx =
        repo_root.join("benchmarks/vnncomp2025/benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_1.onnx");
    assert!(onnx.is_file(), "missing {}", onnx.display());
    census(&onnx, false);
    census(&onnx, true);
}
