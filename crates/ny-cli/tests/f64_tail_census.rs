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
//! It needs the benchmarks/ checkout, so this target is in the explicit
//! `external-vnncomp` lane (`required-features` in Cargo.toml): absent from the
//! default lane, a hard failure when selected without the corpus. Run it with
//! `targo --unverified test -p ny-cli --features external-vnncomp --test
//! f64_tail_census -- --nocapture`. `examples/f64_tail_census.rs` prints the
//! same census as an example (d13e1b66f converted this file into it on
//! 2026-08-25; b1e402055 re-added this copy on 2026-09-13).

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

#[test]
fn cgan_row2_loaded_graph_census() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let onnx =
        repo_root.join("benchmarks/vnncomp2025/benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_1.onnx");
    assert!(onnx.is_file(), "missing {}", onnx.display());
    census(&onnx, false);
    census(&onnx, true);
}
