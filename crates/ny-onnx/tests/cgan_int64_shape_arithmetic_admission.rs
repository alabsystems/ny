// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #cgan-int64-shape-arithmetic: what the INT64 structural-control gate admits,
// and — the half that matters — what it still refuses.
//
// WHAT CHANGED. `819b0554` required every INT64 value reaching a structural
// consumer to come from an "exact static shape-control producer", and its
// producer list had no arithmetic in it. cgan's small_transformer computes
//
//     Div_59 = Gather(Shape(input.47), Const) / Const
//
// with `input.47` fixed at [1, 256, 8, 8] — a compile-time constant, far below
// the 2^24 point where distinct i64s share one f32 mirror — and was refused
// with "'onnx::Cast_363' is produced by 'Div_59' (Div), which is not an exact
// static shape-control producer". `configs/model_load_waivers.yaml` carried
// that as REGULAR-track debt worth 9.5 banked normalized from 2026-08-04.
//
// WHY THE ADMISSION IS SAFE, AND WHY IT IS NOT AN OPERATOR ALLOWLIST. Adding
// Add/Sub/Mul/Div to `require_static_source` does not assert those nodes are
// static — it recurses into their inputs, and the recursion bottoms out ONLY at
// an initializer, a `Constant`, or a `Shape`. A chain rooted anywhere else meets
// an unlisted producer and is refused exactly as before. That is the whole
// safety argument, and the cctsdb half of this file is what tests it rather
// than asserting it.
//
// AND IT IS STILL FAIL-CLOSED AT THE ARITHMETIC PASS.
// `validate_arithmetic_float32_data_path` does not ADMIT a structurally
// authenticated node; it DEFERS its FLOAT32 refusal into the same
// `LiteralExemptions` ratchet the literal cone uses, so the node is admitted
// only if constant folding actually erases it and the original message
// re-raises verbatim if it survives. Admitted-if-folded, never
// admitted-because-declared.
//
// Cargo registers this integration target with `required-features =
// ["external-vnncomp"]`, matching its cctsdb sibling: the default suite stays
// hermetic, and the explicitly selected VNN-COMP lane treats a missing corpus
// file as a hard failure rather than a skip.

use ny_onnx::load_onnx;
mod common;
use common::{benchmark_root, BenchYear};

/// The model both retired waiver entries named, under both benchmark roots.
const SMALL_TRANSFORMER: &str = "cGAN_imgSz32_nCh_3_small_transformer.onnx";

#[test]
fn cgan_small_transformer_loads_under_both_benchmark_roots() {
    let candidates = [
        benchmark_root(BenchYear::Vnncomp2025).join(format!("cgan_2023/onnx/{SMALL_TRANSFORMER}")),
        benchmark_root(BenchYear::Vnncomp2026)
            .join(format!("cgan2026/1.0/onnx/{SMALL_TRANSFORMER}")),
    ];
    for path in candidates {
        assert!(
            path.is_file(),
            "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
            path.display()
        );
        if let Err(error) = load_onnx(&path) {
            panic!(
                "{} must load: its shape arithmetic is a compile-time constant over a fixed \
                 [1, 256, 8, 8] tensor. Got: {error}",
                path.display()
            );
        }
    }
}

/// The negative control, and the reason the admission is not a widening.
///
/// Deliberately a SEPARATE assertion from the sibling
/// `cctsdb_cast_admission_loads.rs`: that file pins the refusal against the
/// Cast/dtype policy, this one pins it against THIS change. If a future edit
/// opens the structural recursion far enough to admit a runtime-rooted chain,
/// the capability test above keeps passing and only this one fails — which is
/// the signal that would otherwise be missing.
#[test]
fn a_runtime_rooted_int64_chain_is_still_refused() {
    let root = benchmark_root(BenchYear::Vnncomp2025).join("cctsdb_yolo_2023/onnx");
    for model in ["patch-1.onnx", "patch-3.onnx"] {
        let path = root.join(model);
        assert!(
            path.is_file(),
            "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
            path.display()
        );
        let error = load_onnx(&path).expect_err(
            "cctsdb's Slice bounds are Gather(GRAPH INPUT) -> Cast(->INT64): the structural \
             recursion must meet a graph input, not an initializer, and refuse.",
        );
        let message = error.to_string();
        assert!(
            message.contains("dtype")
                || message.contains("INT64")
                || message.contains("int64")
                || message.contains("structural"),
            "{model}: the refusal must still name the integer/structural policy, got: {message}"
        );
    }
}
