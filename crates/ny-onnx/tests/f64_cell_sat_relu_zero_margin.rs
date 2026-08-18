// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "external-vnncomp")]
//
// #sat-relu-zero-margin acceptance: the f64 cell walk must certify sat_relu's
// EXACTLY-zero-margin counterexamples.
//
// W0.2 (`docs/W02_POINT_BOX_ENCLOSURE_MEASURED_2026-07-26.md` §4) recorded
// sat_relu as a "permanent structural NO" for enclosure-based CE certification:
// the unsafe region is `Y_0 >= 1.0 AND Y_1 <= 0.0` — BOTH NON-STRICT — and the
// network attains exactly 1.0 and 0.0, so no positive-margin counterexample
// exists at all. Charging the Higham `gamma_n * Sum|terms|` relative term there
// straddled the boundary (measured `Y_0 in [1 - 9.77e-15, 1 + 1.00e-14]`,
// `margin_worst = -9.769963e-15`) and nothing could be certified.
//
// The class is compiled from k-SAT as `Gemm -> ReLU -> Gemm` with weights in
// {-1, +1, +2} and integer biases, evaluated at boolean corners: every product
// and partial sum is an integer far below 2^53, so the rounding contribution is
// not small, it is provably ZERO (`integer_exact_linear_reduction` in
// ny-propagate's `graph_ibp_f64_cell`). This test pins that the enclosure
// collapses onto the exact integer value — width 0 — on the ORGANIZER-VALIDATED
// alpha-beta-CROWN witnesses for the whole locally-available SAT class, so the
// worst-case margins are exactly 0.0 and a certified side that admits `m == 0`
// on non-strict constraints (ny-cli `property_violation_certain_f64`) certifies.

use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::OutputConstraint as OC;
use ny_onnx::{load_onnx_with_config, GraphNetworkOptions, OnnxLoadConfig};
use ny_propagate::Interval64;

const BENCH: &str = "../../benchmarks/vnncomp2025/benchmarks/sat_relu";
const ABC_CE: &str = "../../external_tools/vnncomp2025_results/alpha_beta_crown/2025_sat_relu";

/// Parse an official `.counterexample` file's `(X_i  v)` / `(Y_j  v)` pairs.
fn parse_counterexample(text: &str) -> (Vec<f64>, Vec<f64>) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for raw in text.split('(') {
        let body = raw.trim_end_matches([')', '\n', ' ']);
        let mut parts = body.split_whitespace();
        let Some(name) = parts.next() else { continue };
        let Some(value) = parts.next().and_then(|v| v.parse::<f64>().ok()) else {
            continue;
        };
        match name.split_once('_') {
            Some(("X", _)) => xs.push(value),
            Some(("Y", _)) => ys.push(value),
            _ => {}
        }
    }
    (xs, ys)
}

/// Every output constraint of the spec, whichever container the parser used.
fn all_constraints(spec: &ny_onnx::vnnlib::VnnLibSpec) -> Vec<OC> {
    if spec.output_constraint_clauses.is_empty() {
        spec.output_constraints.clone()
    } else {
        spec.output_constraint_clauses.concat()
    }
}

#[test]
fn f64_cell_certifies_sat_relu_zero_margin_counterexamples() {
    let (bench, ce_dir) = (Path::new(BENCH), Path::new(ABC_CE));
    assert!(
        bench.join("onnx").is_dir() && ce_dir.is_dir(),
        "external sat_relu benchmark and pinned alpha-beta-CROWN counterexamples are required; \
         run benchmarks/download_benchmarks.sh and stage the result corpus"
    );

    let mut onnx_files: Vec<PathBuf> = std::fs::read_dir(bench.join("onnx"))
        .expect("onnx dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "onnx")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("sat_"))
        })
        .collect();
    onnx_files.sort();
    assert!(
        !onnx_files.is_empty(),
        "no uncompressed sat_relu SAT models found"
    );

    // Per-instance (stem, Y_0 lower, Y_1 upper, max width, m0, m1). Collected
    // rather than asserted inline so that a run with the kill switch
    // `NY_F64_EXACT_INTEGER=0` still PRINTS the baseline margins before failing.
    let mut rows: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    for onnx in &onnx_files {
        let stem = onnx.file_stem().unwrap().to_str().unwrap().to_string();
        let ce_path = ce_dir.join(format!("{stem}_{stem}.counterexample.gz"));
        assert!(
            ce_path.is_file(),
            "{stem}: pinned counterexample missing at {}",
            ce_path.display()
        );
        let vnnlib = [
            bench.join(format!("vnnlib/{stem}.vnnlib")),
            bench.join(format!("vnnlib/{stem}.vnnlib.gz")),
        ]
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("{stem}: no vnnlib"));

        let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib).expect("vnnlib parses");
        let constraints = all_constraints(&spec);
        // The two asserts must land in ONE clause (a CONJUNCTION). Were they
        // split across two clauses of a disjunction, both the acceptance gate
        // and the certified side would accept a witness meeting only one of
        // them — so pin the shape, not just the constraint multiset.
        assert!(
            spec.output_constraint_clauses.len() <= 1,
            "{stem}: sat_relu's unsafe region is a conjunction, got {} clauses \
             (is_disjunction={})",
            spec.output_constraint_clauses.len(),
            spec.is_disjunction
        );
        // The spec shape the whole finding is about: two NON-STRICT constraints
        // pinned to the network's exactly-attainable extremes.
        assert_eq!(constraints.len(), 2, "{stem}: expected 2 constraints");
        assert!(
            constraints.iter().all(|c| !c.is_strict()),
            "{stem}: sat_relu constraints must both be non-strict"
        );
        assert!(
            constraints.contains(&OC::GreaterEqConst(0, 1.0))
                && constraints.contains(&OC::LessEqConst(1, 0.0)),
            "{stem}: expected (>= Y_0 1.0) and (<= Y_1 0.0), got {constraints:?}"
        );

        let ce_text = ny_load::io::read_string_maybe_gzip(&ce_path).expect("ce file");
        let (x, declared_y) = parse_counterexample(&ce_text);
        assert_eq!(
            x.len(),
            spec.input_bounds.len(),
            "{stem}: counterexample arity"
        );
        // The organizer's own witnesses are boolean corners.
        assert!(
            x.iter().all(|v| *v == 0.0 || *v == 1.0),
            "{stem}: abc witness is not a boolean corner"
        );

        let model = load_onnx_with_config(onnx, &OnnxLoadConfig::default()).expect("model loads");
        let graph = model
            .to_graph_network_with_options(GraphNetworkOptions::default())
            .expect("graph network");
        assert!(
            graph.supports_ibp_f64_cell(),
            "{stem}: f64 cell unsupported"
        );

        let point = ArrayD::from_shape_vec(IxDyn(&[x.len()]), x.clone()).expect("input tensor");
        let out = graph
            .propagate_ibp_f64_cell(&Interval64::point(point))
            .expect("f64 cell forward");
        let lo: Vec<f64> = out.lower.iter().copied().collect();
        let hi: Vec<f64> = out.upper.iter().copied().collect();
        assert_eq!(lo.len(), 2, "{stem}: sat_relu emits Y_0, Y_1");

        // SOUNDNESS cross-check that holds in BOTH modes: the enclosure must
        // contain the organizer-published output values.
        if declared_y.len() == 2 {
            for j in 0..2 {
                assert!(
                    lo[j] <= declared_y[j] && declared_y[j] <= hi[j],
                    "{stem} Y_{j}: abc's published {} outside enclosure [{}, {}]",
                    declared_y[j],
                    lo[j],
                    hi[j]
                );
            }
        }

        // Worst-case (unfavorable-endpoint) margins for the two constraints.
        let m0 = lo[0] - 1.0; // (>= Y_0 1.0)
        let m1 = 0.0 - hi[1]; // (<= Y_1 0.0)
        rows.push((
            stem,
            lo[0],
            hi[1],
            (hi[0] - lo[0]).max(hi[1] - lo[1]),
            m0,
            m1,
        ));
    }

    let certified = rows
        .iter()
        .filter(|(_, _, _, w, m0, m1)| *w == 0.0 && *m0 == 0.0 && *m1 == 0.0)
        .count();
    let worst_margin = rows
        .iter()
        .map(|(_, _, _, _, m0, m1)| m0.min(*m1))
        .fold(f64::INFINITY, f64::min);
    let worst_width = rows
        .iter()
        .map(|(_, _, _, w, _, _)| *w)
        .fold(0.0f64, f64::max);
    for (stem, y0, y1, w, m0, m1) in &rows {
        eprintln!(
            "  {stem:<16} Y_0_lo={y0:.17e} Y_1_hi={y1:.17e} width={w:.3e} \
             m0={m0:+.6e} m1={m1:+.6e}"
        );
    }
    eprintln!(
        "sat_relu exact-enclosure certification: {certified}/{} instances certified at margin \
         exactly 0.0 (max width {worst_width:.3e}, min margin {worst_margin:+.6e}); \
         {} complete local uncompressed SAT models",
        rows.len(),
        onnx_files.len()
    );
    assert!(
        certified >= 30 && certified == rows.len(),
        "expected every checked sat_relu SAT row to certify at margin exactly 0, \
         got {certified}/{} (min margin {worst_margin:+.6e}, max width {worst_width:.3e})",
        rows.len()
    );
}

/// NEGATIVE control on the real benchmark net: the exactness path is a property
/// of the OPERANDS. A fractional in-box input must keep the Higham widening, so
/// no zero-margin certification can be manufactured by feeding arbitrary points.
#[test]
fn sat_relu_fractional_input_keeps_the_higham_widening() {
    let onnx = Path::new(BENCH).join("onnx/sat_v14_c57.onnx");
    assert!(
        onnx.is_file(),
        "external sat_relu fixture missing at {}; run benchmarks/download_benchmarks.sh",
        onnx.display()
    );
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("model loads");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph network");
    let x = vec![0.5f64; 14];
    let point = ArrayD::from_shape_vec(IxDyn(&[14]), x).expect("input tensor");
    let out = graph
        .propagate_ibp_f64_cell(&Interval64::point(point))
        .expect("f64 cell forward");
    let widths: Vec<f64> = out
        .lower
        .iter()
        .zip(out.upper.iter())
        .map(|(l, h)| h - l)
        .collect();
    assert!(
        widths.iter().any(|w| *w > 0.0),
        "a fractional input must not take the exactness path (widths {widths:?})"
    );
    eprintln!("fractional-input enclosure widths: {widths:?}");
}
