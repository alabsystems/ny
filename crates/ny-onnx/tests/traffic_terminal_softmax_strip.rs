// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end guard test for the dark `NY_STRIP_TERMINAL_SOFTMAX` transform,
// against the REAL staged traffic_signs_recognition_2023 corpus.
//
// WHY THIS FILE EXISTS. The unit tests in `optimization.rs` build synthetic
// fixtures, so they can only prove the guard is internally consistent. They
// cannot prove two things that actually matter:
//
//   1. that the strict argmax-complement predicate still ADMITS the real 45
//      instances (a guard that refuses everything is trivially "sound" and
//      completely worthless), and
//   2. that the shape the parser really produces for these files is the shape
//      the predicate was written against.
//
// Both are checked here against the shipped bytes.
//
// THE FLOAT32 FACT THIS GUARD EXISTS FOR, measured with ORT 1.19.2 over 301
// points of the `model_30_idx_1703_eps_1.00000` input box on
// `3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx`: logit magnitudes reach
// 2590, and 42 of the 43 softmax outputs are EXACTLY `0.0f` at every single
// sampled point. So a bare pairwise atom between two non-argmax classes, e.g.
// `(>= Y[0,0] Y[0,42])`, is SAT on the original ONNX (`p_0 == p_42 == 0.0f` at
// 301/301) and UNSAT on the peeled logits (`z_0 >= z_42` at 0/301; at the box
// midpoint `z_0 = 296`, `z_42 = 460`). Stripping there manufactures a false
// `unsat`. Only the argmax-complement shape closes that gap, which is why the
// spec-side predicate is not optional.

use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{
    load_and_peel_terminal_softmax_single_group, load_onnx, read_onnx_bytes_maybe_gzip,
    strip_terminal_softmax, OnnxLoadConfig, STRIP_TERMINAL_SOFTMAX_ENV,
};
use std::path::{Path, PathBuf};
mod common;
use common::{benchmark_root, BenchYear};

const MODEL: &str = "3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx";

fn traffic_root() -> PathBuf {
    benchmark_root(BenchYear::Vnncomp2026).join("traffic_signs_recognition_2023/2.0")
}

fn traffic_model_path() -> PathBuf {
    let plain = traffic_root().join("onnx").join(MODEL);
    if plain.is_file() {
        plain
    } else {
        Path::new(&format!("{}.gz", plain.display())).to_path_buf()
    }
}

fn traffic_spec_path(name: &str) -> PathBuf {
    let plain = traffic_root().join("vnnlib").join(name);
    if plain.is_file() {
        plain
    } else {
        Path::new(&format!("{}.gz", plain.display())).to_path_buf()
    }
}

/// The real corpus parses into exactly the shape the strict predicate expects:
/// 43 outputs, a top-level disjunction of 42 SINGLETON clauses, every atom a
/// non-strict `GreaterEq(i, t)` sharing one right-hand true label `t`, the
/// challenger set exactly `{0..42} \ {t}`, no per-clause input boxes, no dual
/// network, and a flat list that is exactly the clause concatenation.
#[test]
fn every_staged_traffic_spec_is_an_argmax_complement_disjunction() {
    let vnnlib_dir = traffic_root().join("vnnlib");
    assert!(
        vnnlib_dir.is_dir(),
        "Benchmark directory missing: {}. Run benchmarks/download_benchmarks.sh first.",
        vnnlib_dir.display()
    );

    let mut checked = 0usize;
    for entry in std::fs::read_dir(&vnnlib_dir).expect("read vnnlib dir") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().unwrap().to_string_lossy();
        if !(name.ends_with(".vnnlib") || name.ends_with(".vnnlib.gz")) {
            continue;
        }
        // Count each spec ONCE. `download_benchmarks.sh` decompresses the corpus
        // but keeps the `.gz` originals beside the results, so the directory
        // holds both `x.vnnlib` and `x.vnnlib.gz` for every instance — 90
        // entries for 45 specs. Prefer the plain file exactly as
        // `traffic_spec_path` does, and skip the archive when it is present.
        if name.ends_with(".vnnlib.gz") && path.with_extension("").is_file() {
            continue;
        }
        let spec = load_vnnlib(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let name = path.file_name().unwrap().to_string_lossy().into_owned();

        assert_eq!(spec.num_outputs, 43, "{name}: output count");
        assert!(spec.is_disjunction, "{name}: not a top-level disjunction");
        assert!(
            spec.dual_network.is_none(),
            "{name}: unexpected dual network"
        );
        assert!(
            spec.per_clause_input_bounds.iter().all(|m| m.is_empty()),
            "{name}: unexpected per-clause input bounds"
        );
        assert_eq!(
            spec.output_constraint_clauses.len(),
            42,
            "{name}: clause count"
        );

        let mut true_label = None;
        let mut challengers = [false; 43];
        for clause in &spec.output_constraint_clauses {
            assert_eq!(clause.len(), 1, "{name}: clause is not a singleton");
            let ny_onnx::vnnlib::OutputConstraint::GreaterEq(challenger, label) = clause[0] else {
                panic!("{name}: atom is not a non-strict output-vs-output GreaterEq: {clause:?}");
            };
            assert_ne!(challenger, label, "{name}: self-comparison");
            match true_label {
                None => true_label = Some(label),
                Some(existing) => assert_eq!(existing, label, "{name}: mixed true labels"),
            }
            assert!(!challengers[challenger], "{name}: repeated challenger");
            challengers[challenger] = true;
        }
        let true_label = true_label.expect("at least one clause");
        for (index, hit) in challengers.iter().enumerate() {
            assert_eq!(
                *hit,
                index != true_label,
                "{name}: challenger set is not the complement of the true label"
            );
        }

        let concatenated: Vec<_> = spec
            .output_constraint_clauses
            .iter()
            .flatten()
            .cloned()
            .collect();
        assert_eq!(
            spec.output_constraints, concatenated,
            "{name}: flat list diverges from the clauses"
        );
        checked += 1;
    }
    assert_eq!(checked, 45, "expected the full staged corpus of 45 specs");
}

/// The dark path is byte-identical on the REAL model + REAL spec: with the gate
/// unset, the graph keeps its Softmax, the graph output keeps its name, and the
/// spec keeps every atom.
#[test]
fn real_traffic_instance_is_untouched_with_the_gate_unset() {
    let model_path = traffic_model_path();
    let spec_path = traffic_spec_path("model_30_idx_1703_eps_1.00000.vnnlib");
    assert!(
        model_path.is_file() && spec_path.is_file(),
        "Benchmark files missing under {}. Run benchmarks/download_benchmarks.sh first.",
        traffic_root().display()
    );

    let mut model = load_onnx(&model_path).expect("load traffic model");
    let mut spec = load_vnnlib(&spec_path).expect("load spec");

    let layers_before = model.network.layers.len();
    let output_before = model.network.outputs[0].name.clone();
    let flat_before = spec.output_constraints.clone();
    let clauses_before = spec.output_constraint_clauses.clone();

    let report =
        ny_test_utils::env::with_serialized_env_vars_removed(&[STRIP_TERMINAL_SOFTMAX_ENV], || {
            strip_terminal_softmax(&mut model, &mut spec)
        });

    assert!(!report.peeled, "dark path stripped the real instance");
    assert_eq!(
        report.reason.as_deref(),
        Some("NY_STRIP_TERMINAL_SOFTMAX is not exactly \"1\" (dark by default)")
    );
    assert_eq!(model.network.layers.len(), layers_before);
    assert_eq!(model.network.outputs[0].name, output_before);
    assert_eq!(spec.output_constraints, flat_before);
    assert_eq!(spec.output_constraint_clauses, clauses_before);
}

/// The authenticated typed route fires on the exact real model bytes. The
/// logical-byte reader makes `.onnx` and `.onnx.gz` spellings authenticate the
/// same protobuf, and that same retained slice is parsed and hashed.
#[test]
fn real_traffic_instance_strips_only_under_exact_lattice_certificate() {
    let model_path = traffic_model_path();
    let spec_path = traffic_spec_path("model_30_idx_1703_eps_1.00000.vnnlib");
    assert!(
        model_path.is_file() && spec_path.is_file(),
        "Benchmark files missing under {}. Run benchmarks/download_benchmarks.sh first.",
        traffic_root().display()
    );

    let model_bytes = read_onnx_bytes_maybe_gzip(&model_path).expect("read logical model bytes");
    let mut spec = load_vnnlib(&spec_path).expect("load spec");
    let flat_before = spec.output_constraints.clone();
    let clauses_before = spec.output_constraint_clauses.clone();
    let (model, report) = load_and_peel_terminal_softmax_single_group(
        MODEL,
        &model_bytes,
        &OnnxLoadConfig::default(),
        &mut spec,
    )
    .expect("load and qualify exact traffic bytes");

    assert!(
        report.peeled,
        "armed strip declined the real instance: {:?}",
        report.reason
    );
    assert_eq!(report.layer_type, Some(ny_core::LayerType::Softmax));
    assert_eq!(
        model
            .network
            .layers
            .iter()
            .filter(|l| l.layer_type == ny_core::LayerType::Softmax)
            .count(),
        0,
        "the terminal Softmax must be gone"
    );
    assert!(model.network.outputs[0].name.contains("MatMul"));
    assert_eq!(
        spec.output_constraints, flat_before,
        "atoms are the identity under the strip"
    );
    assert_eq!(
        spec.output_constraint_clauses, clauses_before,
        "clause structure must be preserved"
    );
}
