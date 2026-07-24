// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//! M2 throughput gate microbench: batched dense-spec CROWN backward leaves/s on
//! the mscn_2048d_dual net.
//!
//! Loads the real ONNX net + a dual clause box, builds a batch of sub-boxes of
//! one clause box, runs ONE shared-root batched dense-spec CROWN backward over
//! them (2 output-direction spec rows [[+1],[-1]]), and reports domains/second.
//!
//! Run:
//!   cargo run -p ny-cli --release --example microbench_mscn_backward
//! Optional args: <onnx> <vnnlib> <batch_sizes csv> <shared_root 0|1>

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::load_vnnlib;
use ny_propagate::bench_batched::bench_batched_dense_spec_backward;
use ny_tensor::BoundedTensor;

const DEFAULT_ONNX: &str = "benchmarks/vnncomp2025/benchmarks/nn4sys/onnx/mscn_2048d_dual.onnx";
const DEFAULT_VNNLIB: &str =
    "benchmarks/vnncomp2025/benchmarks/nn4sys/vnnlib/cardinality_1_240_2048_dual.vnnlib";

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let onnx = args.first().map(String::as_str).unwrap_or(DEFAULT_ONNX);
    let vnnlib_path = args.get(1).map(String::as_str).unwrap_or(DEFAULT_VNNLIB);
    let batch_sizes: Vec<usize> = args
        .get(2)
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![64, 128, 256]);
    let use_shared_root = args.get(3).map(|s| s == "1").unwrap_or(true);

    eprintln!("Loading ONNX: {onnx}");
    let model = ny_onnx::load_onnx(onnx)?;
    let graph = model.to_graph_network()?;
    eprintln!("Graph built: {} nodes", graph.num_nodes());

    eprintln!("Loading VNNLIB: {vnnlib_path}");
    let spec = load_vnnlib(vnnlib_path)?;
    eprintln!(
        "spec: {} inputs, {} outputs, {} clauses, disjunction={}",
        spec.num_inputs,
        spec.num_outputs,
        spec.output_constraint_clauses.len(),
        spec.is_disjunction
    );

    // Base input box from global bounds, shaped like the graph input with the
    // batch dim stripped (graph models: [batch, 22, 14] -> [22, 14]).
    let (base_lo, base_hi) = spec.split_input_bounds_f32();
    let n_in = base_lo.len();
    // mscn input is 22x14 = 308.
    let shape: Vec<usize> = if n_in == 308 {
        vec![22, 14]
    } else {
        vec![n_in]
    };

    // Choose the first clause with >= 1 ranged axis in its clause box.
    let mut chosen: Option<(usize, Vec<f32>, Vec<f32>, Vec<usize>)> = None;
    for (ci, per_clause) in spec.per_clause_input_bounds.iter().enumerate() {
        if per_clause.is_empty() {
            continue;
        }
        let mut lo = base_lo.clone();
        let mut hi = base_hi.clone();
        for (&idx, &(l, u)) in per_clause.iter() {
            if idx < lo.len() {
                lo[idx] = l as f32;
                hi[idx] = u as f32;
            }
        }
        let ranged: Vec<usize> = (0..lo.len()).filter(|&i| hi[i] > lo[i]).collect();
        if !ranged.is_empty() {
            chosen = Some((ci, lo, hi, ranged));
            break;
        }
    }
    let (ci, clause_lo, clause_hi, ranged) =
        chosen.ok_or_else(|| anyhow::anyhow!("no clause with a ranged axis found"))?;
    // widest ranged axis
    let split_axis = *ranged
        .iter()
        .max_by(|&&a, &&b| {
            (clause_hi[a] - clause_lo[a])
                .partial_cmp(&(clause_hi[b] - clause_lo[b]))
                .unwrap()
        })
        .unwrap();
    eprintln!(
        "chosen clause #{ci}: {} ranged axes {:?}; split axis {} width {}",
        ranged.len(),
        ranged,
        split_axis,
        clause_hi[split_axis] - clause_lo[split_axis]
    );

    // Optional shared-root reference bounds computed once on the parent clause box.
    let clause_box = make_box(&clause_lo, &clause_hi, &shape)?;
    let shared_root: Option<HashMap<String, BoundedTensor>> = if use_shared_root {
        let t0 = Instant::now();
        let m = graph.collect_forward_linear_bounds_dag_with_engine(&clause_box, None)?;
        eprintln!(
            "shared-root forward-linear reference bounds: {} nodes in {:.3}s",
            m.len(),
            t0.elapsed().as_secs_f64()
        );
        Some(m)
    } else {
        None
    };

    // 2 output-direction spec rows: [[+1], [-1]] over the single output Y0.
    let spec_matrix = Array2::<f32>::from_shape_vec((2, spec.num_outputs), {
        let mut v = vec![0.0f32; 2 * spec.num_outputs];
        v[0] = 1.0; // row 0: +Y0
        v[spec.num_outputs] = -1.0; // row 1: -Y0
        v
    })?;

    println!("\n==== M2 MICROBENCH: batched dense-spec CROWN backward, mscn_2048d_dual ====");
    println!(
        "shared_root={use_shared_root}  spec_rows={}",
        spec_matrix.nrows()
    );
    println!(
        "{:>7} | {:>10} | {:>12} | {:>10} | {:>10} | {:>6} | sample Y0 [lb,ub]",
        "batch", "total_s", "leaves/s", "fwd_s", "bwd_s", "fast?"
    );

    for &b in &batch_sizes {
        let boxes = build_subboxes(&clause_lo, &clause_hi, split_axis, b, &shape)?;
        // 1 warmup + 3 measured
        let _ = bench_batched_dense_spec_backward(
            &graph,
            &boxes,
            &spec_matrix,
            None,
            shared_root.as_ref(),
        )?;
        let mut times = Vec::new();
        let mut last = None;
        for _ in 0..3 {
            let r = bench_batched_dense_spec_backward(
                &graph,
                &boxes,
                &spec_matrix,
                None,
                shared_root.as_ref(),
            )?;
            times.push(r.total_elapsed_s);
            last = Some(r);
        }
        let r = last.unwrap();
        times.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let median = times[times.len() / 2];
        let leaves_per_s = (r.n_domains as f64) / median;
        println!(
            "{:>7} | {:>10.4} | {:>12.1} | {:>10} | {:>10} | {:>6} | [{:.6}, {:.6}]",
            r.n_domains,
            median,
            leaves_per_s,
            r.forward_elapsed_s
                .map(|s| format!("{s:.4}"))
                .unwrap_or_else(|| "-".into()),
            r.backward_elapsed_s
                .map(|s| format!("{s:.4}"))
                .unwrap_or_else(|| "-".into()),
            r.batched_fast_path,
            r.sample_row0_lb,
            r.sample_row0_ub,
        );
    }

    Ok(())
}

fn make_box(lo: &[f32], hi: &[f32], shape: &[usize]) -> anyhow::Result<BoundedTensor> {
    let lower = ArrayD::from_shape_vec(IxDyn(shape), lo.to_vec())?;
    let upper = ArrayD::from_shape_vec(IxDyn(shape), hi.to_vec())?;
    Ok(BoundedTensor::new(lower, upper)?)
}

/// Split the parent clause box along `axis` into `n` equal-width sub-boxes.
fn build_subboxes(
    lo: &[f32],
    hi: &[f32],
    axis: usize,
    n: usize,
    shape: &[usize],
) -> anyhow::Result<Vec<BoundedTensor>> {
    let a = lo[axis] as f64;
    let b = hi[axis] as f64;
    let step = (b - a) / (n as f64);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut sl = lo.to_vec();
        let mut su = hi.to_vec();
        sl[axis] = (a + step * (k as f64)) as f32;
        su[axis] = (a + step * ((k + 1) as f64)) as f32;
        out.push(make_box(&sl, &su, shape)?);
    }
    Ok(out)
}
