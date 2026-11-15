// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// What the cctsdb_yolo_2023 load gate actually decides.
//
// HISTORY. Commit 25dee0c5 replaced the Cast lowering with `target != 1 =>
// ModelLoad error`, which rejected cctsdb's Cast->INT64 position gates and its
// Equal->Cast(BOOL) class mask, turning 39 banked rows into `unknown` in ~0.7s.
// That gate was right about f16/bf16/DOUBLE and wrong about integers and BOOL;
// 2e187bf0 restored the load.  819b0554 then hardened the raw comparison and
// INT64 structural schemas ("has comparison operands dtype 7"), refusing BOTH
// models again, and this file pinned that refusal as "cctsdb is a
// data-dependent shape program".
//
// THAT DIAGNOSIS WAS HALF RIGHT, and this file now pins each half separately.
//
// PATCH-1 LOADS (#cctsdb-expand-identity / #cctsdb-trunc-gated-index).  Its
// four dtype-7 `Equal`s are PyTorch's `Tensor.expand(-1,-1,-1,-1)` keep-dims
// idiom over a runtime shape: `Equal(Shape(index_grid), -1 * ones)` is
// CONSTANT all-false by ONNX semantics (dims are never negative), and each
// paired `Where` passes through the very `Shape` output it compares, so the
// loader certifies the cone, canonicalizes `Where -> Identity`, and deletes
// the constant-false comparison.  The remaining INT64 machinery is the
// trunc-gated runtime index chain: patch coordinates read from the FLOAT input
// via `Gather -> Cast(->INT64)` (a guarded `LayerType::Trunc`), exact integer
// index arithmetic, Slice bounds, ScatterND indices, and the
// ArgMax/BOOL-mask readout casts.  Consumer positions are still bounded by the
// INT64 control audit; the value-range obligations (f32 mirror < 2^24, Slice
// bounds in range) are discharged at solve time by the Trunc gating, the
// bounded-index ScatterND path, the taint-gated IBP degrade, and the per-cell
// enumeration driver.  Every mask shape in patch-1 folds statically to
// [1,3,64,64] because its base Slice bounds are authored constants.
//
// PATCH-3 STAYS REFUSED, and for the reason the old comment gave — but at the
// node where it is actually true.  Its zero-patch values tensor
// `ConstantOfShape_39` takes its shape from `Shape(Slice(..., starts/ends
// read from the runtime input))`; ONNX Slice CLAMPS, so the tensor's EXTENT is
// runtime data ny's fixed-shape graph cannot represent, and the constant
// folder correctly cannot erase it.  `unknown` remains the sound answer for
// its 3 banked-unsat rows until that tensor gets a real certificate; its sat
// rows are recovered by the graph-free falsification lane, which needs no
// load.  Cost recorded in configs/model_load_waivers.yaml.
//
// Cargo registers this integration target with `required-features =
// ["external-vnncomp"]`. The default suite therefore remains hermetic; the
// explicitly selected VNN-COMP lane runs this test and treats missing corpus
// files as a hard failure.

use ny_core::LayerType;
use ny_onnx::load_onnx;
mod common;
use common::{benchmark_root, BenchYear};

#[test]
fn cctsdb_patch1_loads_with_certified_expand_identity_and_trunc_gated_index() {
    let root = benchmark_root(BenchYear::Vnncomp2025).join("cctsdb_yolo_2023/onnx");
    let path = root.join("patch-1.onnx");
    assert!(
        path.is_file(),
        "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
        path.display()
    );

    let loaded = load_onnx(&path)
        .unwrap_or_else(|error| panic!("patch-1.onnx must load, not fail closed: {error}"));
    let layers = &loaded.network.layers;

    // The certified expand-identity cones must be GONE: a load that keeps a
    // Where layer over reinterpreted INT64 shape values is exactly what the
    // raw gate exists to prevent.
    let surviving_where: Vec<_> = layers
        .iter()
        .filter(|layer| layer.layer_type == LayerType::Where)
        .map(|layer| layer.name.clone())
        .collect();
    assert!(
        surviving_where.is_empty(),
        "the certified expand-identity Where cones must be canonicalized away, \
         but Where layers survived: {surviving_where:?}"
    );
    // No fail-closed Cast layer may survive either; the runtime coordinate
    // casts owe guarded Trunc layers instead.
    assert!(
        !layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Cast),
        "no fail-closed Cast layer may survive the admission"
    );
    assert!(
        layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Trunc),
        "the runtime patch-coordinate casts must lower to guarded Trunc layers"
    );
    // Sanity: the full YOLO backbone, not an empty shell.
    assert!(
        layers.len() > 100,
        "expected the full patch-1 graph, got {} layers",
        layers.len()
    );
}

#[test]
fn cctsdb_patch3_zero_patch_constant_of_shape_stays_refused() {
    let root = benchmark_root(BenchYear::Vnncomp2025).join("cctsdb_yolo_2023/onnx");
    let path = root.join("patch-3.onnx");
    assert!(
        path.is_file(),
        "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
        path.display()
    );

    let error = load_onnx(&path).expect_err(
        "patch-3's zero-patch ConstantOfShape has a runtime-extent shape (clamped Slice); \
         admitting it would mean fabricating its extent. Fail closed instead.",
    );
    let message = error.to_string();
    assert!(
        message.contains("survived constant folding on a live data path"),
        "the refusal must name the unfoldable constant-family node, got: {message}"
    );
}
