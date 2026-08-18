// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "external-vnncomp")]
//
// Opt-in diagnostic (task #83): localise vit_2023 pgd_2_3_16's ~2.4e4 BOX-INDEPENDENT
// bound floor by node.
//
// A sound relaxation of a continuous network is EXACT on a degenerate (zero
// width) box: the reachable set is a point, so any residual output width is
// pure abstraction slack. This diagnostic propagates a zero-width box (lower ==
// upper == the vnnlib box centre) and prints every node's output width in
// execution order. The FIRST node whose width is non-trivial at zero input
// width IS the defect.
//
// Run with:
//   cargo run -p ny-onnx --release --features external-vnncomp \
//       --example vit-zero-width-node-bisect -- scan

use ny_core::Bound;
use ny_onnx::{load_onnx_with_config, GraphNetworkOptions, OnnxLoadConfig};
use ny_propagate::Verifier;

const BENCH_DEFAULT: &str = "../../benchmarks/vnncomp2025/benchmarks/vit_2023";

fn bench_dir() -> String {
    std::env::var("NY_VIT_BENCH").unwrap_or_else(|_| BENCH_DEFAULT.to_string())
}

fn read_vnnlib_box(path: &str) -> (Vec<f32>, Vec<f32>) {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read VNN-LIB fixture {path}: {error}"));
    let mut lo: Vec<(usize, f32)> = Vec::new();
    let mut hi: Vec<(usize, f32)> = Vec::new();
    for line in txt.lines() {
        let t = line.trim();
        // (assert (<= X_0 1.23))  /  (assert (>= X_0 1.20))
        let Some(rest) = t.strip_prefix("(assert (") else {
            continue;
        };
        let op_le = rest.starts_with("<= X_");
        let op_ge = rest.starts_with(">= X_");
        if !op_le && !op_ge {
            continue;
        }
        let body = &rest[5..];
        let Some(sp) = body.find(' ') else { continue };
        let idx: usize = body[..sp].parse().expect("index");
        let val: f32 = body[sp + 1..]
            .trim_end_matches([')', ' '])
            .parse()
            .expect("value");
        if op_le {
            hi.push((idx, val));
        } else {
            lo.push((idx, val));
        }
    }
    let n = lo
        .iter()
        .chain(&hi)
        .map(|(index, _)| *index)
        .max()
        .unwrap_or_else(|| panic!("{path}: no direct input bounds found"))
        + 1;
    let mut l = vec![f32::NEG_INFINITY; n];
    let mut u = vec![f32::INFINITY; n];
    for (i, v) in lo {
        l[i] = l[i].max(v);
    }
    for (i, v) in hi {
        u[i] = u[i].min(v);
    }
    assert!(
        l.iter()
            .zip(&u)
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper),
        "{path}: every input must have finite, ordered lower and upper bounds"
    );
    (l, u)
}

fn scan(model_file: &str, vnnlib_file: &str, scale: f32) {
    let bench = bench_dir();
    let model_path = format!("{bench}/onnx/{model_file}");
    let vnnlib_path = format!("{bench}/vnnlib/{vnnlib_file}");
    assert!(
        std::path::Path::new(&model_path).is_file(),
        "missing {model_path}"
    );
    assert!(
        std::path::Path::new(&vnnlib_path).is_file(),
        "missing {vnnlib_path}"
    );
    let (l, u) = read_vnnlib_box(&vnnlib_path);
    let n = l.len();
    let degenerate: Vec<Bound> = (0..n)
        .map(|i| {
            let c = 0.5 * (l[i] + u[i]);
            let r = 0.5 * (u[i] - l[i]) * scale;
            Bound::new(c - r, c + r)
        })
        .collect();

    let model = load_onnx_with_config(&model_path, &OnnxLoadConfig::default()).expect("load");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("convert");
    let input = Verifier::bounds_to_tensor(&degenerate, Some(&[3, 32, 32])).expect("input");
    let detailed = graph.propagate_ibp_detailed(&input, 0.0).expect("ibp");

    println!("=== {model_file} / {vnnlib_file} / box-scale {scale:e} ===");
    println!(
        "{:>4}  {:<18} {:>12} {:>12} {:>9} {:>12} {:>12}  node",
        "idx", "layer_type", "in_width", "out_width", "sens", "min", "max"
    );
    let mut first_nonzero: Option<usize> = None;
    for (i, node) in detailed.nodes.iter().enumerate() {
        if first_nonzero.is_none() && node.output_width > 0.0 {
            first_nonzero = Some(i);
        }
        println!(
            "{:>4}  {:<18} {:>12.4e} {:>12.4e} {:>9.3} {:>12.4} {:>12.4}  {}",
            i,
            node.layer_type,
            node.input_width,
            node.output_width,
            node.sensitivity,
            node.min_bound,
            node.max_bound,
            node.name
        );
    }
    let last = detailed.nodes.last().expect("nodes");
    println!(
        "FINAL: nodes={} width={:.6} min={:.6} max={:.6}",
        detailed.nodes.len(),
        last.output_width,
        last.min_bound,
        last.max_bound
    );
    match first_nonzero {
        Some(f) => {
            let nd = &detailed.nodes[f];
            println!(
                "FIRST-NONTRIVIAL: idx={} type={} in_width={:.3e} out_width={:.3e} name={}",
                f, nd.layer_type, nd.input_width, nd.output_width, nd.name
            );
        }
        None => println!("FIRST-NONTRIVIAL: none (every node width <= 1e-6)"),
    }

    // Per layer-type width summary over the whole graph.
    use std::collections::BTreeMap;
    let mut by_type: BTreeMap<&str, (usize, f32)> = BTreeMap::new();
    for node in &detailed.nodes {
        let e = by_type.entry(node.layer_type.as_str()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 = e.1.max(node.output_width);
    }
    println!("--- per-layer-type max output width ---");
    for (t, (c, w)) in by_type {
        println!("  {t:<20} count={c:<4} max_out_width={w:.6e}");
    }
    println!();
}

/// Same graph, two box scales, per-node RATIO. If the ratio is roughly constant
/// down the graph, the whole zero-width floor is ONE amplified seed. If the
/// ratio COLLAPSES at some node, that node injects a box-independent constant
/// and IS the defect.
fn compare(model_file: &str, vnnlib_file: &str, s_lo: f32, s_hi: f32) {
    let bench = bench_dir();
    let model_path = format!("{bench}/onnx/{model_file}");
    let vnnlib_path = format!("{bench}/vnnlib/{vnnlib_file}");
    let (l, u) = read_vnnlib_box(&vnnlib_path);
    let n = l.len();
    let model = load_onnx_with_config(&model_path, &OnnxLoadConfig::default()).expect("load");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("convert");
    let run = |scale: f32| {
        let degenerate: Vec<Bound> = (0..n)
            .map(|i| {
                let c = 0.5 * (l[i] + u[i]);
                let r = 0.5 * (u[i] - l[i]) * scale;
                Bound::new(c - r, c + r)
            })
            .collect();
        let input = Verifier::bounds_to_tensor(&degenerate, Some(&[3, 32, 32])).expect("input");
        graph.propagate_ibp_detailed(&input, 0.0).expect("ibp")
    };
    let a = run(s_lo);
    let b = run(s_hi);
    assert!(
        !a.nodes.is_empty() && a.nodes.len() == b.nodes.len(),
        "ratio scan requires the same non-empty graph at both scales"
    );
    println!("=== RATIO SCAN {model_file} / {vnnlib_file}: scale {s_hi:e} vs {s_lo:e} ===");
    println!(
        "{:>4}  {:<18} {:>12} {:>12} {:>10}  node",
        "idx", "layer_type", "w(lo)", "w(hi)", "hi/lo"
    );
    for (i, (na, nb)) in a.nodes.iter().zip(b.nodes.iter()).enumerate() {
        let r = if na.output_width > 0.0 {
            nb.output_width / na.output_width
        } else {
            f32::INFINITY
        };
        println!(
            "{:>4}  {:<18} {:>12.4e} {:>12.4e} {:>10.4}  {}",
            i, na.layer_type, na.output_width, nb.output_width, r, na.name
        );
    }
    println!();
}

/// CROWN PREFIX BISECT. Point `NY_VIT_PREFIX_DIR` at a directory of ONNX
/// prefix models `pNNN.onnx` (produced by onnx.utils.extract_model, one per
/// node output) and this reports, for each prefix, the IBP and CROWN output
/// width at ZERO input width. The first prefix whose CROWN width jumps by
/// orders of magnitude over its IBP width localises the CROWN floor to that
/// node.
fn vit_crown_prefix_bisect() {
    let dir = std::env::var("NY_VIT_PREFIX_DIR")
        .expect("NY_VIT_PREFIX_DIR must name the generated ONNX prefix directory");
    let row = std::env::var("NY_BISECT_ROW").unwrap_or_else(|_| "pgd_2_3_16_2446".to_string());
    let bench = bench_dir();
    let (l, u) = read_vnnlib_box(&format!("{bench}/vnnlib/{row}.vnnlib"));
    let n = l.len();
    let scale: f32 = std::env::var("NY_BISECT_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let degenerate: Vec<Bound> = (0..n)
        .map(|i| {
            let c = 0.5 * (l[i] + u[i]);
            let r = 0.5 * (u[i] - l[i]) * scale;
            Bound::new(c - r, c + r)
        })
        .collect();
    let input = Verifier::bounds_to_tensor(&degenerate, Some(&[3, 32, 32])).expect("input");

    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("prefix dir") {
        let path = entry
            .unwrap_or_else(|error| panic!("failed to read an entry in {dir}: {error}"))
            .path();
        if path
            .extension()
            .is_some_and(|extension| extension == "onnx")
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('p'))
        {
            files.push(path);
        }
    }
    files.sort();
    assert!(
        !files.is_empty(),
        "NY_VIT_PREFIX_DIR={dir} contains no p*.onnx prefix models"
    );
    println!(
        "=== CROWN PREFIX BISECT (scale {scale:e}), {} prefixes ===",
        files.len()
    );
    println!(
        "{:>6} {:>14} {:>14} {:>10}  last_node",
        "prefix", "ibp_width", "crown_width", "crown/ibp"
    );
    for p in files {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let path = p
            .to_str()
            .unwrap_or_else(|| panic!("prefix path is not UTF-8: {}", p.display()));
        let model = load_onnx_with_config(path, &OnnxLoadConfig::default())
            .unwrap_or_else(|error| panic!("{name}: load failed: {error}"));
        let graph = model
            .to_graph_network_with_options(GraphNetworkOptions::default())
            .unwrap_or_else(|error| panic!("{name}: graph conversion failed: {error}"));
        let detailed = graph
            .propagate_ibp_detailed(&input, 0.0)
            .unwrap_or_else(|error| panic!("{name}: detailed IBP failed: {error}"));
        let last_node = detailed
            .nodes
            .last()
            .unwrap_or_else(|| panic!("{name}: converted prefix graph contains no nodes"));
        let last = format!("{} [{}]", last_node.name, last_node.layer_type);
        let ibp = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|error| panic!("{name}: IBP failed: {error}"))
            .max_width();
        let crown = graph
            .propagate_crown(&input)
            .unwrap_or_else(|error| panic!("{name}: CROWN failed: {error}"))
            .max_width();
        assert!(
            ibp.is_finite() && crown.is_finite(),
            "{name}: non-finite widths IBP={ibp}, CROWN={crown}"
        );
        println!(
            "{name:>6} {ibp:>14.4e} {crown:>14.4e} {:>10.3e}  {last}",
            crown / ibp.max(f32::MIN_POSITIVE)
        );
    }
}

/// ORACLE-INTERMEDIATE ABLATION on ny's OWN CROWN.
///
/// At zero input width every node's true value is a POINT, so
/// `collect_node_activations_pointwise` is the EXACT intermediate bound. Feeding
/// it to CROWN removes ALL intermediate-bound looseness and leaves only the
/// relaxations' own backward composition. Re-admitting the IBP interval for ONE
/// layer type at a time then prices that type's contribution.
///
/// UNSOUND BY CONSTRUCTION (the oracle is only valid at zero width) —
/// attribution only, never a verdict.
fn vit_crown_oracle_intermediate_ablation() {
    let row = std::env::var("NY_BISECT_ROW").unwrap_or_else(|_| "pgd_2_3_16_2446".to_string());
    let model_file = if row.starts_with("pgd") {
        "pgd_2_3_16.onnx"
    } else {
        "ibp_3_3_8.onnx"
    };
    let bench = bench_dir();
    let (l, u) = read_vnnlib_box(&format!("{bench}/vnnlib/{row}.vnnlib"));
    let n = l.len();
    let model = load_onnx_with_config(
        &format!("{bench}/onnx/{model_file}"),
        &OnnxLoadConfig::default(),
    )
    .expect("load");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("convert");
    let degenerate: Vec<Bound> = (0..n)
        .map(|i| {
            let c = 0.5 * (l[i] + u[i]);
            Bound::new(c, c)
        })
        .collect();
    let input = Verifier::bounds_to_tensor(&degenerate, Some(&[3, 32, 32])).expect("input");

    let detailed = graph.propagate_ibp_detailed(&input, 0.0).expect("detailed");
    let ty: std::collections::HashMap<String, String> = detailed
        .nodes
        .iter()
        .map(|nd| (nd.name.clone(), nd.layer_type.clone()))
        .collect();
    let nb_ibp = graph.collect_node_bounds(&input).expect("nb ibp");
    let nb_pt = graph
        .collect_node_activations_pointwise(&input, None)
        .expect("nb pointwise");

    let ident = ndarray::Array2::<f32>::eye(10);
    let arm = |nb: &std::collections::HashMap<String, ny_tensor::BoundedTensor>, label: &str| {
        let (b, _) = graph
            .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &ident, None, nb)
            .expect("crown");
        println!("  {label:<44} crown_width = {:.6e}", b.max_width());
    };

    println!("=== ORACLE INTERMEDIATE ABLATION (zero width): {model_file} / {row} ===");
    println!(
        "  {:<44} max intermediate width = {:.4e}",
        "IBP intermediates",
        nb_ibp
            .values()
            .map(|b| b.max_width())
            .fold(0.0f32, f32::max)
    );
    println!(
        "  {:<44} max intermediate width = {:.4e}",
        "POINTWISE (oracle) intermediates",
        nb_pt.values().map(|b| b.max_width()).fold(0.0f32, f32::max)
    );
    arm(&nb_ibp, "A: all-IBP intermediates (shipped)");
    arm(&nb_pt, "B: all-ORACLE intermediates");
    for t in ["Softmax", "BilinearCrown", "ReLU", "BatchNorm", "Linear"] {
        let mut mixed = nb_pt.clone();
        for (name, b) in &nb_ibp {
            if ty.get(name).map(|x| x == t).unwrap_or(false) {
                mixed.insert(name.clone(), b.clone());
            }
        }
        arm(&mixed, &format!("C[{t}]: ORACLE except {t} (IBP)"));
    }
}

/// IBP vs CROWN vs CROWN-with-IBP-intermediates at several box scales.
/// A sound method is EXACT at scale 0; whatever residual width each shows there
/// is that method's own box-independent floor.
fn vit_ibp_vs_crown_floor() {
    let row = std::env::var("NY_BISECT_ROW").unwrap_or_else(|_| "pgd_2_3_16_2446".to_string());
    let model_file = if row.starts_with("pgd") {
        "pgd_2_3_16.onnx"
    } else {
        "ibp_3_3_8.onnx"
    };
    let bench = bench_dir();
    let model_path = format!("{bench}/onnx/{model_file}");
    let vnnlib_path = format!("{bench}/vnnlib/{row}.vnnlib");
    let (l, u) = read_vnnlib_box(&vnnlib_path);
    let n = l.len();
    let model = load_onnx_with_config(&model_path, &OnnxLoadConfig::default()).expect("load");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("convert");

    println!("=== IBP vs CROWN floor: {model_file} / {row} ===");
    println!(
        "{:>10} {:>16} {:>16} {:>16}",
        "scale", "ibp_width", "crown_width", "crown_ibp_nb_width"
    );
    for scale in [0.0f32, 1e-3, 1e-2, 1e-1, 1.0] {
        let degenerate: Vec<Bound> = (0..n)
            .map(|i| {
                let c = 0.5 * (l[i] + u[i]);
                let r = 0.5 * (u[i] - l[i]) * scale;
                Bound::new(c - r, c + r)
            })
            .collect();
        let input = Verifier::bounds_to_tensor(&degenerate, Some(&[3, 32, 32])).expect("input");
        let ibp = graph.propagate_ibp(&input).expect("ibp");
        let crown = graph.propagate_crown(&input).expect("crown");
        let nb = graph.collect_node_bounds(&input).expect("nb");
        let ident = ndarray::Array2::<f32>::eye(10);
        let (crown_nb, _) = graph
            .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &ident, None, &nb)
            .expect("crown+nb");
        println!(
            "{:>10.1e} {:>16.6e} {:>16.6e} {:>16.6e}",
            scale,
            ibp.max_width(),
            crown.max_width(),
            crown_nb.max_width()
        );
    }
}

fn vit_pgd_zero_vs_full_ratio_scan() {
    let row = std::env::var("NY_BISECT_ROW").unwrap_or_else(|_| "pgd_2_3_16_2446".to_string());
    let model = if row.starts_with("pgd") {
        "pgd_2_3_16.onnx"
    } else {
        "ibp_3_3_8.onnx"
    };
    compare(model, &format!("{row}.vnnlib"), 0.0, 1.0);
}

fn vit_pgd_zero_width_per_node_bisect() {
    let row = std::env::var("NY_BISECT_ROW").unwrap_or_else(|_| "pgd_2_3_16_2446".to_string());
    let model = if row.starts_with("pgd") {
        "pgd_2_3_16.onnx"
    } else {
        "ibp_3_3_8.onnx"
    };
    let scales: Vec<f32> = std::env::var("NY_BISECT_SCALES")
        .unwrap_or_else(|_| "0".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("scale"))
        .collect();
    for s in scales {
        scan(model, &format!("{row}.vnnlib"), s);
    }
}

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "scan".to_string());
    match mode.as_str() {
        "scan" => vit_pgd_zero_width_per_node_bisect(),
        "ratio" => vit_pgd_zero_vs_full_ratio_scan(),
        "prefix" => vit_crown_prefix_bisect(),
        "oracle" => vit_crown_oracle_intermediate_ablation(),
        "floor" => vit_ibp_vs_crown_floor(),
        other => panic!(
            "unknown diagnostic {other:?}; expected one of scan, ratio, prefix, oracle, floor"
        ),
    }
}
