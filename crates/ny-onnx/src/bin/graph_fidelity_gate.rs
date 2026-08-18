// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//! Graph-fidelity gate: bit-diff NY's post-load weights against the authored
//! ONNX initializers.
//!
//! DIAGNOSTIC ONLY. It loads, compares, and prints. Nothing on a verdict path
//! calls it and it changes no bound, no verdict, and no loader behaviour.
//!
//! Why it exists (W0.2, `docs/W02_POINT_BOX_ENCLOSURE_MEASURED_2026-07-26.md`
//! §3): NY's loader folds Conv/Gemm + `BatchNormalization` at load time
//! entirely in f32, so on `cifar100_2024` and `tinyimagenet_2024` the certified
//! point-box enclosure is about an f32-folded surrogate, and the f64 forward of
//! the *unmodified* ONNX falls outside it. A certificate must be able to state
//! which function it is about. This tool answers that per node.
//!
//! Two references are reported per node:
//!
//! * `auth_*` — deviation from the authored ONNX initializer. Nonzero means the
//!   loaded network is not the benchmark network.
//! * `fold_*` — deviation from the same fold expression re-evaluated in f64
//!   from the authored initializers. This is the precision the rewrite lost,
//!   i.e. the part NY's enclosure does not account for. `-` means the rewrite
//!   is not attributable to a Conv/Gemm/ConvTranspose + BN fold, which is a
//!   louder signal than a large `fold_*`.
//!
//! The `verdict` column has three values. `authored` — every authored payload
//! survives load unchanged. `authored-f32` — every FLOAT coefficient survives,
//! but the constant-folder added integer shape tensors; the certified function
//! is still the benchmark network. `rewritten` — a FLOAT coefficient changed,
//! so it is not. Only `rewritten` fails `--gate`.
//!
//! ```text
//! target/release/graph_fidelity_gate benchmarks/vnncomp2025/benchmarks/cifar100_2024/onnx/*.onnx
//! target/release/graph_fidelity_gate --json model.onnx > fidelity.json
//! find benchmarks/vnncomp2025 -name '*.onnx' | sort | xargs target/release/graph_fidelity_gate --summary
//! ```
//!
//! Flags: `--json` (machine-readable, one object per model), `--all` (every
//! node and tensor, not just the non-faithful ones), `--summary` (one line per
//! model), `--gate` (exit 1 when any model is not the authored graph; opt-in,
//! so ordinary runs always exit 0 unless a model failed to load).

use ny_onnx::fidelity::{format_report, graph_fidelity_report, GraphFidelityReport};

fn main() {
    let mut json = false;
    let mut all = false;
    let mut summary = false;
    let mut gate = false;
    let mut paths: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--json" => json = true,
            "--all" => all = true,
            "--summary" => summary = true,
            "--gate" => gate = true,
            "-h" | "--help" => {
                println!(
                    "usage: graph_fidelity_gate [--json] [--all] [--summary] [--gate] <model.onnx>..."
                );
                return;
            }
            other => paths.push(other.to_string()),
        }
    }
    if paths.is_empty() {
        eprintln!(
            "usage: graph_fidelity_gate [--json] [--all] [--summary] [--gate] <model.onnx>..."
        );
        std::process::exit(2);
    }

    let mut failed_to_load = 0usize;
    let mut unfaithful = 0usize;
    if summary && !json {
        println!("model\tverdict\tidentical\trewritten\tdropped\tsynthesized\tundetermined\tauth_max_rel\tfold_max_rel\tfold_max_ulp\tunexplained");
    }
    for path in &paths {
        match graph_fidelity_report(path) {
            Ok(report) => {
                // The gate fails on a rewrite of the FLOAT coefficients only.
                // Constant-folded integer shape tensors change the graph's
                // representation, not the function being certified.
                if !report.float_weights_are_authored() {
                    unfaithful += 1;
                }
                if json {
                    match serde_json::to_string(&report) {
                        Ok(line) => println!("{line}"),
                        Err(err) => {
                            eprintln!("{path}: failed to serialize report: {err}");
                            failed_to_load += 1;
                        }
                    }
                } else if summary {
                    println!("{}", summary_line(&report));
                } else {
                    print!("{}", format_report(&report, all));
                }
            }
            Err(err) => {
                failed_to_load += 1;
                if json {
                    println!(
                        "{}",
                        serde_json::json!({ "model": path, "error": err.to_string() })
                    );
                } else {
                    println!("graph-fidelity gate: {path}\n  ERROR: {err}");
                }
            }
        }
    }

    if failed_to_load > 0 {
        eprintln!("{failed_to_load}/{} model(s) failed to load", paths.len());
        std::process::exit(2);
    }
    if gate && unfaithful > 0 {
        eprintln!(
            "{unfaithful}/{} model(s) are NOT the authored graph",
            paths.len()
        );
        std::process::exit(1);
    }
}

fn summary_line(report: &GraphFidelityReport) -> String {
    use ny_onnx::fidelity::TensorStatus;
    let authored = report.worst_vs_authored();
    let fold = report.worst_vs_fold_reference();
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6e}\t{:.6e}\t{}\t{}",
        report.model,
        if report.is_authored_graph() {
            "authored"
        } else if report.float_weights_are_authored() {
            "authored-f32"
        } else {
            "rewritten"
        },
        report.count(TensorStatus::Identical),
        report.count(TensorStatus::Rewritten),
        report.count(TensorStatus::Dropped),
        report.count(TensorStatus::Synthesized),
        report.count(TensorStatus::Undetermined),
        authored.max_rel,
        fold.max_rel,
        fold.max_ulp,
        report.unexplained_rewrites().len(),
    )
}
