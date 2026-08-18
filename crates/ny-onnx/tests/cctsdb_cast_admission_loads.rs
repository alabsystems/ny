// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// What the cctsdb_yolo_2023 load gate actually decides.
//
// HISTORY. Commit 25dee0c5 replaced the Cast lowering with `target != 1 =>
// ModelLoad error`, which rejected cctsdb's Cast->INT64 position gates and its
// Equal->Cast(BOOL) class mask, turning 39 banked rows into `unknown` in ~0.7s.
// That gate was right about f16/bf16/DOUBLE (a const-fold would erase their
// rounding into a plain FLOAT constant) and wrong about integers and BOOL. This
// file was then written to assert that BOTH cctsdb models LOAD.
//
// WHY THAT ASSERTION IS WRONG. cctsdb is a DATA-DEPENDENT SHAPE PROGRAM, and no
// dtype policy can fix that. Traced in the shipped patch-1.onnx: four Slice
// nodes (Slice_34/38/59/70) take their `starts` and `ends` from
//
//     Gather(GRAPH INPUT '0') -> Cast(->INT64) -> Unsqueeze / Add
//
// i.e. the patch coordinates are read out of the runtime input. ONNX Slice
// CLAMPS out-of-range bounds, so the sliced length -- and with it Shape_87's
// value, Expand_94's target, and the ScatterND output shape -- depends on the
// runtime value. ny's tensor graph is fixed-shape, so it can only represent this
// model by ASSUMING the coordinates are in range, and the loader cannot see the
// input box to discharge that. This is the same class the loader already refuses
// categorically for NonZero ("data-dependent output shape that ny cannot
// represent soundly"). `819b0554`'s gates are that refusal generalized, not a
// drafting slip.
//
// So `unknown` here is the SOUND answer, and this test now pins the refusal.
// COST, recorded rather than hidden: cctsdb_yolo_2023 is worth 100.0 banked
// normalized on the extended track, ~39 rows. Regaining it needs a real
// certificate -- a load- or build-time obligation that every runtime INT64 lane
// value is finite, in range for its destination dtype, and below 2^24 in the f32
// mirror (two distinct int64s above that share one f32, which would make `Equal`
// return TRUE for unequal inputs), plus a proof that the Slice bounds are in
// range so the lowered shapes are the authored ones. That is a design change,
// not a gate relaxation.
//
// The half this file was written to protect is NOT lost: that INT32/INT64/BOOL
// Casts lower to a guarded `LayerType::Trunc` (and that no fail-closed
// `LayerType::Cast` survives) is covered at unit level by
// `loader::convert::tests::cast_to_int_lowers_to_trunc` and its runtime-INT64
// companion, which do not need the benchmark tree.
//
// Cargo registers this integration target with `required-features =
// ["external-vnncomp"]`. The default suite therefore remains hermetic; the
// explicitly selected VNN-COMP lane runs this test and treats missing corpus
// files as a hard failure.

use ny_onnx::load_onnx;
mod common;
use common::{benchmark_root, BenchYear};

const MODELS: [&str; 2] = ["patch-1.onnx", "patch-3.onnx"];

#[test]
fn cctsdb_models_are_refused_as_data_dependent_shape_programs() {
    let root = benchmark_root(BenchYear::Vnncomp2025).join("cctsdb_yolo_2023/onnx");
    for model in MODELS {
        let path = root.join(model);
        assert!(
            path.is_file(),
            "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
            path.display()
        );

        let error = load_onnx(&path).expect_err(
            "cctsdb is a data-dependent shape program; loading it would require assuming \
             the patch coordinates are in range. Fail closed instead.",
        );
        let message = error.to_string();
        assert!(
            message.contains("dtype")
                || message.contains("INT64")
                || message.contains("int64")
                || message.contains("structural"),
            "{model}: the refusal must name the integer/structural policy, got: {message}"
        );
    }
}
