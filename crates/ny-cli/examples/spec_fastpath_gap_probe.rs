// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cross-category audit probe for the spec-CROWN ROOT fast path
//! (#w4-root-margin / #w4-root-gpu, see 4726b45b frac-head finding):
//! computes the SAME spec bound through the plain
//! `propagate_crown_with_specs_and_engine` entry (root-candidate fast paths
//! eligible) and the `_with_linear` request (full spec-CROWN backward), and
//! reports the per-row gaps.
//!
//! This is an explicit external-corpus measurement lane, not a test. Run one
//! probe (or all of them, sequentially) with:
//! `cargo run -p ny-cli --release --example spec_fastpath_gap_probe -- <name>`.

use ndarray::Array2;
use ny_onnx::vnnlib::{load_vnnlib, OutputConstraint};
use ny_onnx::{load_onnx_with_config, GraphNetworkOptions, OnnxLoadConfig};
use ny_tensor::BoundedTensor;
use std::path::Path;

const BENCH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/vnncomp2025/benchmarks"
);

/// Build one spec row per (clause, constraint): diff rows Y_i - Y_j for
/// relational constraints, +Y_i for const-threshold constraints. The exact
/// verdict polarity does not matter for the gap audit — both routes bound the
/// identical rows.
fn spec_rows(clauses: &[Vec<OutputConstraint>], num_outputs: usize) -> Array2<f32> {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for clause in clauses {
        for c in clause {
            let mut row = vec![0.0f32; num_outputs];
            match *c {
                OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                    row[i] += 1.0;
                    row[j] -= 1.0;
                }
                OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                    row[j] += 1.0;
                    row[i] -= 1.0;
                }
                OutputConstraint::LessEqConst(i, _)
                | OutputConstraint::GreaterEqConst(i, _)
                | OutputConstraint::LessThanConst(i, _)
                | OutputConstraint::GreaterThanConst(i, _) => {
                    row[i] = 1.0;
                }
                _ => continue,
            }
            rows.push(row);
        }
    }
    let n = rows.len();
    Array2::from_shape_vec((n, num_outputs), rows.into_iter().flatten().collect())
        .expect("spec matrix")
}

fn run_probe(label: &str, onnx_rel: &str, vnnlib_rel: &str) -> anyhow::Result<()> {
    let onnx_path = Path::new(BENCH).join(onnx_rel);
    let vnnlib_path = Path::new(BENCH).join(vnnlib_rel);
    anyhow::ensure!(
        onnx_path.is_file(),
        "{label}: missing ONNX prerequisite {}",
        onnx_path.display()
    );
    anyhow::ensure!(
        vnnlib_path.is_file(),
        "{label}: missing VNN-LIB prerequisite {}",
        vnnlib_path.display()
    );
    let model = load_onnx_with_config(&onnx_path, &OnnxLoadConfig::default())?;
    let input_shape = model
        .network
        .inputs
        .first()
        .map(|input| ny_onnx::resolve_dynamic_shape(&input.shape, 1))
        .unwrap_or_else(|| vec![1]);
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .map_err(|error| anyhow::anyhow!("{label}: build graph: {error}"))?;
    let vnnlib = load_vnnlib(&vnnlib_path)?;

    let n_inputs: usize = input_shape.iter().product();
    assert_eq!(
        vnnlib.input_bounds.len(),
        n_inputs,
        "{label}: vnnlib inputs vs model input shape {input_shape:?}"
    );

    // Input box: global bounds, overridden by clause 0's per-clause bounds
    // when present (mscn cardinality style).
    let mut lo: Vec<f32> = vnnlib.input_bounds.iter().map(|&(l, _)| l as f32).collect();
    let mut hi: Vec<f32> = vnnlib.input_bounds.iter().map(|&(_, u)| u as f32).collect();
    if let Some(clause_bounds) = vnnlib.per_clause_input_bounds.first() {
        for (&idx, &(l, u)) in clause_bounds {
            lo[idx] = l as f32;
            hi[idx] = u as f32;
        }
    }
    let input = BoundedTensor::new(
        ndarray::Array1::from(lo)
            .into_shape_with_order(ndarray::IxDyn(&input_shape))
            .expect("shape lo"),
        ndarray::Array1::from(hi)
            .into_shape_with_order(ndarray::IxDyn(&input_shape))
            .expect("shape hi"),
    )
    .expect("input box");

    let spec = spec_rows(&vnnlib.output_constraint_clauses, vnnlib.num_outputs);
    println!(
        "\n=== {label}: {} spec rows, {} outputs, input {:?} ===",
        spec.nrows(),
        vnnlib.num_outputs,
        input_shape
    );

    let t0 = std::time::Instant::now();
    let fast = graph
        .propagate_crown_with_specs_and_engine(&input, &spec, None)
        .expect("fast path");
    let t_fast = t0.elapsed();
    let t0 = std::time::Instant::now();
    let (full, _) = graph
        .propagate_crown_with_specs_and_engine_with_linear(&input, &spec, None)
        .expect("full backward");
    let t_full = t0.elapsed();

    println!(
        "plain entry {:.3}s, _with_linear {:.3}s",
        t_fast.as_secs_f64(),
        t_full.as_secs_f64()
    );
    let mut max_gap = 0f32;
    for r in 0..spec.nrows() {
        let (fl, fu) = (fast.lower()[r], fast.upper()[r]);
        let (bl, bu) = (full.lower()[r], full.upper()[r]);
        let gap_lo = bl - fl; // >0: fast path looser on the lower bound
        let gap_hi = fu - bu; // >0: fast path looser on the upper bound
        max_gap = max_gap.max(gap_lo.max(gap_hi));
        println!(
            "row {r:>3}: fast [{fl:>12.6}, {fu:>12.6}]  full [{bl:>12.6}, {bu:>12.6}]  gap(lo {gap_lo:+.6}, hi {gap_hi:+.6})"
        );
    }
    println!("{label}: max fast-path looseness across rows = {max_gap:+.6}");
    Ok(())
}

fn cgan_prop1_root_gap() -> anyhow::Result<()> {
    run_probe(
        "cgan cGAN_imgSz32_nCh_1 prop_1",
        "cgan_2023/onnx/cGAN_imgSz32_nCh_1.onnx",
        "cgan_2023/vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib",
    )
}

fn metaroom_6cnn_spec_gap() -> anyhow::Result<()> {
    run_probe(
        "metaroom 6cnn_ry_30_5 spec_idx_114",
        "metaroom_2023/onnx/6cnn_ry_30_5_no_custom_OP.onnx",
        "metaroom_2023/vnnlib/spec_idx_114_eps_0.00000436.vnnlib",
    )
}

fn mscn_128d_band_clause_gap() -> anyhow::Result<()> {
    run_probe(
        "mscn_128d cardinality_0_1_128",
        "nn4sys/onnx/mscn_128d.onnx",
        "nn4sys/vnnlib/cardinality_0_1_128.vnnlib",
    )
}

fn acasxu_prop2_root_gap() -> anyhow::Result<()> {
    run_probe(
        "acasxu 1_1 prop_2",
        "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
        "acasxu_2023/vnnlib/prop_2.vnnlib",
    )
}

fn run_named(name: &str) -> anyhow::Result<()> {
    match name {
        "cgan" => cgan_prop1_root_gap(),
        "metaroom" => metaroom_6cnn_spec_gap(),
        "mscn" => mscn_128d_band_clause_gap(),
        "acasxu" => acasxu_prop2_root_gap(),
        "all" => {
            for probe in ["cgan", "metaroom", "mscn", "acasxu"] {
                run_named(probe)?;
            }
            Ok(())
        }
        _ => anyhow::bail!("unknown probe {name:?}; expected cgan, metaroom, mscn, acasxu, or all"),
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing probe name: cgan|metaroom|mscn|acasxu|all"))?;
    anyhow::ensure!(args.next().is_none(), "expected exactly one probe name");
    run_named(&name)
}
