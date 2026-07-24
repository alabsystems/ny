// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic probe for the cgan_2023 bound-quality arc (#cgan-conv-f64-gemm).
//!
//! Prints per-node IBP vs alpha-CROWN intermediate bound widths and the
//! spec-guided CROWN output bounds for cGAN_imgSz32_nCh_1 prop_1's input box.
//! `#[ignore]`d: needs the local VNN-COMP 2025 benchmark checkout; run with
//! `cargo test -p ny-cli --release --test cgan_bound_probe -- --ignored --nocapture`.

use ndarray::Array2;
use ny_onnx::{load_onnx_with_config, GraphNetworkOptions, OnnxLoadConfig};

use ny_tensor::BoundedTensor;
use std::path::Path;

const ONNX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/vnncomp2025/benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_1.onnx",
);

// Input box of cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib.
// Literals kept verbatim from the vnnlib spec (greppable provenance); f32
// parsing rounds them to the same values clippy would suggest.
#[allow(clippy::excessive_precision)]
const LOWER: [f32; 5] = [
    0.30481159687042236,
    0.4375501275062561,
    0.5820457339286804,
    -1.0,
    0.6903666853904724,
];
// Verbatim vnnlib literals (see LOWER); f32 parse is bit-identical either way.
#[allow(clippy::excessive_precision)]
const UPPER: [f32; 5] = [
    0.3248116075992584,
    0.45755013823509216,
    0.6020457148551941,
    -0.9800000190734863,
    0.7103666663169861,
];

fn width_stats(b: &BoundedTensor) -> (f32, f32) {
    let mut max_w = 0.0f32;
    let mut sum_w = 0.0f64;
    for (l, u) in b.lower().iter().zip(b.upper().iter()) {
        let w = u - l;
        if w > max_w {
            max_w = w;
        }
        sum_w += w as f64;
    }
    (max_w, (sum_w / b.lower().len() as f64) as f32)
}

#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_bound_probe() {
    if std::env::var("NY_PROBE_LOG").is_ok() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(std::env::var("NY_PROBE_LOG").unwrap())
            .with_test_writer()
            .try_init();
    }
    let onnx = Path::new(ONNX).to_path_buf();
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("load onnx");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph");

    let input = BoundedTensor::new(
        ndarray::arr1(&LOWER).into_dyn(),
        ndarray::arr1(&UPPER).into_dyn(),
    )
    .expect("input box");

    // 1. Pure IBP node bounds.
    let t0 = std::time::Instant::now();
    let ibp = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("ibp");
    println!("IBP collection: {:.2}s", t0.elapsed().as_secs_f64());

    // 2. CROWN-IBP node bounds with provenance (what the alpha warmup's
    //    reference-bound step runs under deep_seq).
    let t0 = std::time::Instant::now();
    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .expect("crown-ibp");
    println!(
        "CROWN-IBP collection: {:.2}s ({} node sets, {} fallback events)",
        t0.elapsed().as_secs_f64(),
        crown_ibp.bounds.len(),
        crown_ibp.fallback_events.len(),
    );
    for ev in &crown_ibp.fallback_events {
        println!(
            "  fallback: layer {} ({}) reason {:?}: {}",
            ev.layer_index, ev.layer_type, ev.reason, ev.details
        );
    }
    let alpha_bounds = crown_ibp.bounds;
    let provenance = crown_ibp.provenance;

    let order = graph.exec_order().expect("order");
    println!(
        "\n{:<24} {:>14} {:>14} {:>14} {:>14}  provenance",
        "node", "ibp_max_w", "ibp_mean_w", "crown_max_w", "crown_mean_w"
    );
    for name in order {
        let i = ibp.get(name);
        let a = alpha_bounds.get(name);
        let (im, iw) = i.map(width_stats).unwrap_or((f32::NAN, f32::NAN));
        let (am, aw) = a.map(width_stats).unwrap_or((f32::NAN, f32::NAN));
        println!(
            "{:<24} {:>14.4e} {:>14.4e} {:>14.4e} {:>14.4e}  {:?}",
            name,
            im,
            iw,
            am,
            aw,
            provenance.get(name)
        );
    }

    // Optional dump of the collected per-node bounds for offline comparison
    // against the numpy oracle (tools/cgan_crown_reference). Set
    // NY_PROBE_DUMP=/path/prefix to write `<prefix>.csv` with
    // `node,idx,lower,upper` rows for every collected node.
    if let Ok(prefix) = std::env::var("NY_PROBE_DUMP") {
        use std::io::Write;
        let mut f = std::fs::File::create(format!("{prefix}.csv")).expect("dump file");
        for (node, b) in &alpha_bounds {
            let flat = b.flatten();
            for (i, (l, u)) in flat.lower().iter().zip(flat.upper().iter()).enumerate() {
                writeln!(f, "{node},{i},{l},{u}").expect("dump write");
            }
        }
        println!("dumped bounds to {prefix}.csv");
    }

    // 3. Spec-guided CROWN with the alpha node bounds (2-row disjunctive spec).
    let spec = Array2::from_shape_vec((2, 1), vec![1.0f32, -1.0f32]).expect("spec");
    let t0 = std::time::Instant::now();
    let out = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(&input, &spec, None, &alpha_bounds)
        .expect("spec crown");
    println!(
        "\nspec CROWN (alpha node bounds): {:.3}s -> row0 [{:.6}, {:.6}] row1 [{:.6}, {:.6}]",
        t0.elapsed().as_secs_f64(),
        out.lower()[0],
        out.upper()[0],
        out.lower()[1],
        out.upper()[1],
    );

    // 4. Spec-guided CROWN with IBP node bounds (pre-fix effective behavior).
    let t0 = std::time::Instant::now();
    let out_ibp = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(&input, &spec, None, &ibp)
        .expect("spec crown ibp");
    println!(
        "spec CROWN (IBP node bounds):   {:.3}s -> row0 [{:.6}, {:.6}] row1 [{:.6}, {:.6}]",
        t0.elapsed().as_secs_f64(),
        out_ibp.lower()[0],
        out_ibp.upper()[0],
        out_ibp.lower()[1],
        out_ibp.upper()[1],
    );
}

const ONNX_NCH3: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/vnncomp2025/benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3.onnx",
);

// Input box of cGAN_imgSz32_nCh_3_prop_1_input_eps_0.015_output_eps_0.020.vnnlib.
// Literals kept verbatim from the vnnlib spec (greppable provenance); f32
// parsing rounds them to the same values clippy would suggest.
#[allow(clippy::excessive_precision)]
const LOWER3: [f32; 5] = [
    0.43712833523750305,
    0.9700000286102295,
    -0.09279977530241013,
    0.7007557153701782,
    -1.0,
];
// Verbatim vnnlib literals (see LOWER3); f32 parse is bit-identical either way.
#[allow(clippy::excessive_precision)]
const UPPER3: [f32; 5] = [
    0.46712833642959595,
    1.0,
    -0.06279977411031723,
    0.7007557153701782,
    -0.9700000286102295,
];

/// Validated numpy textbook-CROWN reference widths (per-node max width) from
/// `tools/cgan_crown_reference/crown_probe.out` (same adaptive-slope relaxation,
/// same sequential CROWN∩IBP anchor cascade, f32 coefficients; 302-point
/// sampling showed zero soundness violations; BN_5 matches the exact f64
/// affine width).
const ORACLE_NCH1: [(&str, f64); 8] = [
    ("BatchNormalization_5", 0.1811),
    ("BatchNormalization_8", 0.2278),
    ("BatchNormalization_11", 0.1798),
    ("Conv_14", 0.1335),
    ("Conv_16", 0.1046),
    ("Conv_19", 0.06469),
    ("Conv_22", 0.07632),
    ("Gemm_27", 0.02200),
];
const ORACLE_NCH3: [(&str, f64); 8] = [
    ("BatchNormalization_5", 0.3212),
    ("BatchNormalization_8", 0.3205),
    ("BatchNormalization_11", 0.2171),
    ("Conv_14", 0.1104),
    ("Conv_16", 0.1496),
    ("Conv_19", 0.02581),
    ("Conv_22", 0.02981),
    ("Gemm_27", 0.03357),
];

/// Oracle-parity assertion for the per-node CROWN-IBP collection
/// (#cgan-conv-err-compose): every pre-ReLU node's max width must be within
/// `K = 1.05` of the validated numpy textbook-CROWN reference, and no
/// suspiciously tighter than `0.99×` (a sudden sub-oracle width would signal a
/// soundness bug, since the oracle already matches the exact affine widths on
/// the relaxation-free prefix). Measured post-fix (2026-07-10): worst ratio
/// 1.004 (Conv_22, nCh_1).
fn assert_oracle_parity(
    onnx_path: &str,
    lower: &[f32],
    upper: &[f32],
    oracle: &[(&str, f64)],
    tag: &str,
) {
    let onnx = Path::new(onnx_path).to_path_buf();
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("load onnx");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph");
    let input = BoundedTensor::new(
        ndarray::Array1::from_vec(lower.to_vec()).into_dyn(),
        ndarray::Array1::from_vec(upper.to_vec()).into_dyn(),
    )
    .expect("input box");
    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .expect("crown-ibp");
    let mut failures = Vec::new();
    for (node, oracle_w) in oracle {
        let b = crown_ibp
            .bounds
            .get(*node)
            .unwrap_or_else(|| panic!("{tag}: no bounds for {node}"));
        let (max_w, _) = width_stats(b);
        let ratio = max_w as f64 / oracle_w;
        println!("{tag} {node}: ny {max_w:.5e} oracle {oracle_w:.5e} ratio {ratio:.4}");
        // Band: intermediate pre-ReLU nodes must match the oracle within 5%
        // (measured worst 1.004). The SCALAR classifier head (Gemm_27) is
        // discretely sensitive to adaptive-slope flips of near-threshold
        // (u ≈ −l) neurons at Relu_23 — a 0.4% anchor difference at Conv_22
        // flips a handful of 512 slopes and moves the single output row by a
        // few percent (the oracle's own slope ablation measured ±20%), so it
        // gets a wider [0.95, 1.15] band (measured: 1.013 nCh_1, 1.073 nCh_3).
        let (lo_band, hi_band) = if *node == "Gemm_27" {
            (0.95, 1.15)
        } else {
            (0.99, 1.05)
        };
        if !(lo_band..=hi_band).contains(&ratio) {
            failures.push(format!(
                "{node}: ratio {ratio:.4} outside [{lo_band}, {hi_band}]"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{tag}: per-node width vs oracle out of band:\n{}",
        failures.join("\n")
    );
}

/// Oracle parity, nCh_1 prop_1 (#cgan-conv-err-compose).
#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_oracle_parity_nch1() {
    assert_oracle_parity(ONNX, &LOWER, &UPPER, &ORACLE_NCH1, "nCh_1");
}

/// Oracle parity, nCh_3 prop_1 (#cgan-conv-err-compose).
#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_oracle_parity_nch3() {
    assert_oracle_parity(ONNX_NCH3, &LOWER3, &UPPER3, &ORACLE_NCH3, "nCh_3");
}

/// Pure-affine exactness (#cgan-conv-err-compose bug A): the prefix
/// Gemm_0 -> Reshape_2 -> BN_3 -> ConvTranspose_4 -> BN_5 contains no ReLU, no
/// anchors, no relaxation choices, so the CROWN width at BatchNormalization_5
/// must equal the exact affine width `Σ_k |W_eff[:,k]|·(u_k − l_k)` within f32
/// forward-difference noise. Pre-fix this was 2.05× loose (the certified
/// coefficient error's `row_max·‖kernel‖₁` over-bound discharged at BN_3 over
/// its output box); post-fix it matches to < 1e-4 relative.
///
/// The exact width is computed by finite differences through the network's own
/// f32 forward (affine ⇒ unit basis steps are exact in real arithmetic; the
/// f32 evaluation noise is ~1e-6 relative, well under the 1e-4 tolerance,
/// which itself is 50× smaller than the smallest observed regression, the
/// ~0.5%-of-width residual BN_3 fold).
#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_bn5_pure_affine_exactness() {
    let onnx = Path::new(ONNX).to_path_buf();
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("load onnx");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph");
    let input = BoundedTensor::new(
        ndarray::arr1(&LOWER).into_dyn(),
        ndarray::arr1(&UPPER).into_dyn(),
    )
    .expect("input box");
    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .expect("crown-ibp");
    let bn5 = crown_ibp.bounds.get("BatchNormalization_5").expect("BN_5");

    // Exact affine widths via f64 accumulation of f32 forward differences at
    // BN_5. Forward evaluation: point-box IBP; the sound Higham widening is
    // symmetric, so the midpoint is the f32 forward value.
    let forward_mid = |x: &[f32]| -> Vec<f64> {
        let point = BoundedTensor::new(
            ndarray::Array1::from_vec(x.to_vec()).into_dyn(),
            ndarray::Array1::from_vec(x.to_vec()).into_dyn(),
        )
        .expect("point box");
        let bounds = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point ibp");
        let b = bounds.get("BatchNormalization_5").expect("BN_5 point");
        b.lower()
            .iter()
            .zip(b.upper().iter())
            .map(|(l, u)| f64::midpoint(*l as f64, *u as f64))
            .collect()
    };
    let center: Vec<f32> = LOWER
        .iter()
        .zip(UPPER.iter())
        .map(|(l, u)| (l + u) / 2.0)
        .collect();
    let y0 = forward_mid(&center);
    let mut exact_w = vec![0.0f64; y0.len()];
    for k in 0..LOWER.len() {
        let mut xk = center.clone();
        xk[k] += 1.0;
        let yk = forward_mid(&xk);
        let dxk = (UPPER[k] - LOWER[k]) as f64;
        for i in 0..y0.len() {
            exact_w[i] += (yk[i] - y0[i]).abs() * dxk;
        }
    }
    let exact_max = exact_w.iter().cloned().fold(0.0f64, f64::max);
    let (crown_max, _) = width_stats(bn5);
    let rel = (crown_max as f64 - exact_max).abs() / exact_max;
    println!("BN_5 pure-affine: crown {crown_max:.6e} exact {exact_max:.6e} rel diff {rel:.3e}");
    assert!(
        rel <= 1e-4,
        "BN_5 CROWN width must equal the exact affine width on the pure-affine \
         prefix (crown {crown_max:.6e}, exact {exact_max:.6e}, rel {rel:.3e} > 1e-4)"
    );
}

/// Sampling soundness (#cgan-conv-err-compose): 302 concrete points inside the
/// input box (32 corners + 270 seeded-random interior points) must stay INSIDE
/// every claimed per-node bound of the collection. The concrete value is only
/// known up to the sound point-IBP interval, so a violation is flagged only
/// when that interval lies strictly outside the claimed bound (definite
/// violation). Mirrors tools/cgan_crown_reference/sampling_check.out.
#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_sampling_soundness_nch1() {
    let onnx = Path::new(ONNX).to_path_buf();
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("load onnx");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph");
    let input = BoundedTensor::new(
        ndarray::arr1(&LOWER).into_dyn(),
        ndarray::arr1(&UPPER).into_dyn(),
    )
    .expect("input box");
    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .expect("crown-ibp");

    // 32 corners + 270 LCG interior points.
    let mut samples: Vec<[f32; 5]> = Vec::new();
    for mask in 0..32u32 {
        let mut p = [0.0f32; 5];
        for (k, v) in p.iter_mut().enumerate() {
            *v = if mask & (1 << k) != 0 {
                UPPER[k]
            } else {
                LOWER[k]
            };
        }
        samples.push(p);
    }
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) as f32
    };
    for _ in 0..270 {
        let mut p = [0.0f32; 5];
        for (k, v) in p.iter_mut().enumerate() {
            let t = next01();
            *v = LOWER[k] + t * (UPPER[k] - LOWER[k]);
        }
        samples.push(p);
    }

    let mut violations = 0usize;
    let mut checked = 0usize;
    for (si, p) in samples.iter().enumerate() {
        let point = BoundedTensor::new(
            ndarray::Array1::from_vec(p.to_vec()).into_dyn(),
            ndarray::Array1::from_vec(p.to_vec()).into_dyn(),
        )
        .expect("point box");
        let point_bounds = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point ibp");
        for (node, claimed) in &crown_ibp.bounds {
            let Some(pb) = point_bounds.get(node) else {
                continue;
            };
            if pb.lower().len() != claimed.lower().len() {
                continue;
            }
            for (i, ((pl, pu), (cl, cu))) in pb
                .lower()
                .iter()
                .zip(pb.upper().iter())
                .zip(claimed.lower().iter().zip(claimed.upper().iter()))
                .enumerate()
            {
                checked += 1;
                // Definite violation: the sound point interval lies strictly
                // outside the claimed enclosure.
                if pu < cl || pl > cu {
                    violations += 1;
                    if violations <= 10 {
                        println!(
                            "VIOLATION sample {si} node {node} idx {i}: point [{pl}, {pu}] \
                             outside claimed [{cl}, {cu}]"
                        );
                    }
                }
            }
        }
    }
    println!("sampling soundness: {checked} point-element checks, {violations} violations");
    assert_eq!(
        violations, 0,
        "claimed bounds excluded concrete network values"
    );
}

/// Replicates the disjunctive precheck's per-output fallback call
/// (`crown_precheck_per_output` -> `propagate_crown_with_engine_and_deadline`)
/// to see the Y bound the OFFICIAL pipeline decides with (#cgan-conv-err-compose).
#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_per_output_precheck_probe() {
    let onnx = Path::new(ONNX).to_path_buf();
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("load onnx");
    let graph_options = if std::env::var("NY_PROBE_DECOMPOSE_NORM").as_deref() == Ok("1") {
        println!("probe: CompoundNodePolicy::DecomposeNormalization (BaB pipeline setting)");
        GraphNetworkOptions {
            compound_node_policy: ny_onnx::CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        }
    } else {
        GraphNetworkOptions::default()
    };
    let graph = model
        .to_graph_network_with_options(graph_options)
        .expect("graph");
    let input = BoundedTensor::new(
        ndarray::arr1(&LOWER).into_dyn(),
        ndarray::arr1(&UPPER).into_dyn(),
    )
    .expect("input box");
    let mut graph = graph;
    if std::env::var("NY_PROBE_MATRIX_MODE").as_deref() == Ok("1") {
        graph.set_use_patches_mode(false);
        println!("probe: use_patches_mode = false (matrix mode, BaB pipeline setting)");
    }
    // NY_PROBE_PIPELINE_BOX=1: use the exact box the vnncomp pipeline parses
    // (outward-rounded vnnlib constants) instead of the round-to-nearest one.
    let input = if std::env::var("NY_PROBE_PIPELINE_BOX").as_deref() == Ok("1") {
        println!("probe: pipeline (outward-rounded) input box");
        BoundedTensor::new(
            ndarray::arr1(&[0.30481157f32, 0.4375501, 0.5820457, -1.0000001, 0.6903666]).into_dyn(),
            ndarray::arr1(&[0.32481164f32, 0.45755017, 0.6020458, -0.97999996, 0.7103667])
                .into_dyn(),
        )
        .expect("pipeline box")
    } else {
        input
    };
    // NY_PROBE_DEADLINE_SECS: replicate the pipeline's precheck deadline.
    let deadline = std::env::var("NY_PROBE_DEADLINE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));
    if deadline.is_some() {
        println!("probe: deadline {deadline:?}");
    }
    let t0 = std::time::Instant::now();
    let out = graph
        .propagate_crown_with_engine_and_deadline(&input, None, deadline)
        .expect("crown");
    println!(
        "per-output precheck path: {:.1}s -> Y [{:.6}, {:.6}] (band (0.29981, 0.32981))",
        t0.elapsed().as_secs_f64(),
        out.bounds.lower()[0],
        out.bounds.upper()[0],
    );
}

/// Minimal repro: affine chain Linear -> Reshape -> ConvTranspose2d -> ReLU.
/// The CROWN-IBP collection's bound at the pre-ReLU ConvTranspose node should
/// match the exact affine composition (all layers linear), NOT the IBP
/// interval composition.
#[test]
#[ignore = "diagnostic probe; run explicitly with --ignored --nocapture"]
fn cgan_minimal_affine_chain_probe() {
    use ny_propagate::layers::{ConvTranspose2dLayer, LinearLayer, ReLULayer, ReshapeLayer};
    use ny_propagate::{BoundPropagation, GraphNetwork, GraphNode, Layer};

    let in_dim = 5usize;
    // Linear 5 -> 8 with sign-mixed weights.
    let w = Array2::from_shape_fn((8, in_dim), |(i, j)| {
        (((i * 7 + j * 3) % 11) as f32 * 0.21 - 1.0) * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
    });
    let lin = LinearLayer::new(w.clone(), None).expect("lin");
    // Reshape 8 -> (2,2,2)
    let reshape = ReshapeLayer {
        target_shape: vec![2, 2, 2],
    };
    // ConvTranspose 2->2, k2, s2 -> (2,4,4)
    let kernel = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 2, 2, 2]), |d| {
        (((d[0] * 5 + d[1] * 3 + d[2] * 2 + d[3]) % 7) as f32 * 0.33 - 1.0)
            * if (d[0] + d[1] + d[2] + d[3]) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
    });
    let convt =
        ConvTranspose2dLayer::new_full(kernel.clone(), None, (2, 2), (0, 0), (1, 1), (0, 0))
            .expect("convt");

    // BatchNorm layers mirroring the real cGAN chain (BN between Reshape and
    // ConvT, and BN as the pre-ReLU target).
    use ny_propagate::layers::BatchNormLayer;
    let mk_bn = |ch: usize, seed: usize| -> BatchNormLayer {
        let g = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            0.5 + ((d[0] * 3 + seed) % 5) as f32 * 0.3
        });
        let b = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            ((d[0] + seed) % 3) as f32 * 0.1 - 0.1
        });
        let m = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            ((d[0] * 2 + seed) % 4) as f32 * 0.2 - 0.3
        });
        let v = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            0.5 + ((d[0] + seed) % 3) as f32 * 0.4
        });
        BatchNormLayer::new(&g, &b, &m, &v, 1e-5).expect("bn")
    };
    let bn_a = mk_bn(2, 1);
    let bn_b = mk_bn(2, 2);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(lin)));
    graph.add_node(GraphNode::new(
        "reshape",
        Layer::Reshape(reshape),
        vec!["lin".into()],
    ));
    graph.add_node(GraphNode::new(
        "bn_a",
        Layer::BatchNorm(bn_a.clone()),
        vec!["reshape".into()],
    ));
    graph.add_node(GraphNode::new(
        "convt",
        Layer::ConvTranspose2d(convt),
        vec!["bn_a".into()],
    ));
    graph.add_node(GraphNode::new(
        "bn_b",
        Layer::BatchNorm(bn_b.clone()),
        vec!["convt".into()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn_b".into()],
    ));
    graph.set_output("relu");

    let lower = ndarray::Array1::from_elem(in_dim, 0.3f32) - 0.01f32;
    let upper = ndarray::Array1::from_elem(in_dim, 0.3f32) + 0.01f32;
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).expect("box");

    let ibp = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("ibp");
    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .expect("crown ibp");
    for ev in &crown_ibp.fallback_events {
        println!(
            "fallback: layer {} ({}) reason {:?}: {}",
            ev.layer_index, ev.layer_type, ev.reason, ev.details
        );
    }

    // Exact affine width at convt via basis propagation in f64.
    // y = convT(reshape(W x)) ; A[:, k] = y(c + e_k) - y(c).
    let forward = |x: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
        let h = w.dot(x);
        let h3 = h.into_shape_with_order(ndarray::IxDyn(&[2, 2, 2])).unwrap();
        let cv =
            ConvTranspose2dLayer::new_full(kernel.clone(), None, (2, 2), (0, 0), (1, 1), (0, 0))
                .unwrap();
        let bt = BoundedTensor::concrete(h3).unwrap();
        let bt = bn_a.propagate_ibp(&bt).unwrap();
        let out = cv.propagate_ibp(&bt).unwrap();
        let out = bn_b.propagate_ibp(&out).unwrap();
        ndarray::Array1::from_iter(out.lower().iter().cloned())
    };
    let c = ndarray::Array1::from_elem(in_dim, 0.3f32);
    let y0 = forward(&c);
    let mut exact_w = ndarray::Array1::<f64>::zeros(y0.len());
    for k in 0..in_dim {
        let mut e = c.clone();
        e[k] += 1.0;
        let yk = forward(&e);
        for i in 0..y0.len() {
            exact_w[i] += 2.0 * 0.01 * ((yk[i] - y0[i]).abs() as f64);
        }
    }

    let (ibp_max, ibp_mean) = width_stats(ibp.get("bn_b").expect("ibp bn_b"));
    let (cr_max, cr_mean) = width_stats(crown_ibp.bounds.get("bn_b").expect("crown bn_b"));
    let ex_max = exact_w.iter().cloned().fold(0.0f64, f64::max);
    let ex_mean = exact_w.mean().unwrap();
    println!("convt widths: ibp max {ibp_max:.4e} mean {ibp_mean:.4e}");
    println!("convt widths: crown-ibp max {cr_max:.4e} mean {cr_mean:.4e}");
    println!("convt widths: exact affine max {ex_max:.4e} mean {ex_mean:.4e}");
    println!("provenance: {:?}", crown_ibp.provenance.get("bn_b"));
    assert!(
        (cr_max as f64) <= ex_max * 1.05 + 1e-4,
        "CROWN-IBP bound at affine pre-ReLU node must match the exact affine width \
         (got {cr_max:.4e}, exact {ex_max:.4e}, ibp {ibp_max:.4e})"
    );
}

/// Bisect: which BN position degrades the affine-chain CROWN bound to IBP?
#[test]
#[ignore = "diagnostic probe; run explicitly with --ignored --nocapture"]
fn cgan_affine_chain_bn_bisect() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_test_writer()
        .try_init();
    use ny_propagate::layers::{
        BatchNormLayer, ConvTranspose2dLayer, LinearLayer, ReLULayer, ReshapeLayer,
    };
    use ny_propagate::{GraphNetwork, GraphNode, Layer};

    let in_dim = 5usize;
    let w = Array2::from_shape_fn((8, in_dim), |(i, j)| {
        (((i * 7 + j * 3) % 11) as f32 * 0.21 - 1.0) * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
    });
    let kernel = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 2, 2, 2]), |d| {
        (((d[0] * 5 + d[1] * 3 + d[2] * 2 + d[3]) % 7) as f32 * 0.33 - 1.0)
            * if (d[0] + d[1] + d[2] + d[3]) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
    });
    let mk_bn = |ch: usize, seed: usize| -> BatchNormLayer {
        let g = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            0.5 + ((d[0] * 3 + seed) % 5) as f32 * 0.3
        });
        let b = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            ((d[0] + seed) % 3) as f32 * 0.1 - 0.1
        });
        let m = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            ((d[0] * 2 + seed) % 4) as f32 * 0.2 - 0.3
        });
        let v = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            0.5 + ((d[0] + seed) % 3) as f32 * 0.4
        });
        BatchNormLayer::new(&g, &b, &m, &v, 1e-5).expect("bn")
    };

    for variant in ["bn_before_convt", "bn_after_convt"] {
        let lin = LinearLayer::new(w.clone(), None).expect("lin");
        let reshape = ReshapeLayer {
            target_shape: vec![2, 2, 2],
        };
        let convt =
            ConvTranspose2dLayer::new_full(kernel.clone(), None, (2, 2), (0, 0), (1, 1), (0, 0))
                .expect("convt");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("lin", Layer::Linear(lin)));
        graph.add_node(GraphNode::new(
            "reshape",
            Layer::Reshape(reshape),
            vec!["lin".into()],
        ));
        let (target, prev) = if variant == "bn_before_convt" {
            graph.add_node(GraphNode::new(
                "bn_a",
                Layer::BatchNorm(mk_bn(2, 1)),
                vec!["reshape".into()],
            ));
            graph.add_node(GraphNode::new(
                "convt",
                Layer::ConvTranspose2d(convt),
                vec!["bn_a".into()],
            ));
            ("convt", "convt")
        } else {
            graph.add_node(GraphNode::new(
                "convt",
                Layer::ConvTranspose2d(convt),
                vec!["reshape".into()],
            ));
            graph.add_node(GraphNode::new(
                "bn_b",
                Layer::BatchNorm(mk_bn(2, 2)),
                vec!["convt".into()],
            ));
            ("bn_b", "bn_b")
        };
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec![prev.to_string()],
        ));
        graph.set_output("relu");

        let lower = ndarray::Array1::from_elem(in_dim, 0.3f32) - 0.01f32;
        let upper = ndarray::Array1::from_elem(in_dim, 0.3f32) + 0.01f32;
        let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).expect("box");

        let ibp = graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("ibp");
        let crown_ibp = graph
            .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
            .expect("crown ibp");
        let (ibp_max, ibp_mean) = width_stats(ibp.get(target).unwrap());
        let (cr_max, cr_mean) = width_stats(crown_ibp.bounds.get(target).unwrap());
        println!(
            "{variant}: target {target} ibp {ibp_max:.6e}/{ibp_mean:.6e} crown {cr_max:.6e}/{cr_mean:.6e} provenance {:?} fallbacks {}",
            crown_ibp.provenance.get(target),
            crown_ibp.fallback_events.len()
        );
    }
}

/// Step the variant-(a) backward manually to find the non-finite step.
#[test]
#[ignore = "diagnostic probe; run explicitly with --ignored --nocapture"]
fn cgan_affine_chain_manual_steps() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();
    use ny_propagate::layers::{BatchNormLayer, ConvTranspose2dLayer, LinearLayer};
    use ny_propagate::LinearBounds;

    let kernel = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 2, 2, 2]), |d| {
        (((d[0] * 5 + d[1] * 3 + d[2] * 2 + d[3]) % 7) as f32 * 0.33 - 1.0)
            * if (d[0] + d[1] + d[2] + d[3]) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
    });
    let mut convt =
        ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0)).expect("ct");
    convt.set_input_shape(2, 2);

    let seed = LinearBounds::identity(32);
    let after_convt = convt
        .propagate_linear_with_engine(&seed, None)
        .expect("convt backward");
    let nonfinite = |lb: &LinearBounds| -> (usize, usize) {
        let a_bad = lb
            .lower_a()
            .iter()
            .chain(lb.upper_a().iter())
            .filter(|v| !v.is_finite())
            .count();
        let b_bad = lb
            .lower_b()
            .iter()
            .chain(lb.upper_b().iter())
            .filter(|v| !v.is_finite())
            .count();
        (a_bad, b_bad)
    };
    println!(
        "after convt backward: nonfinite {:?}",
        nonfinite(&after_convt)
    );

    let mk_bn = |ch: usize, seed: usize| -> BatchNormLayer {
        let g = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            0.5 + ((d[0] * 3 + seed) % 5) as f32 * 0.3
        });
        let b = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            ((d[0] + seed) % 3) as f32 * 0.1 - 0.1
        });
        let m = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            ((d[0] * 2 + seed) % 4) as f32 * 0.2 - 0.3
        });
        let v = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[ch]), |d| {
            0.5 + ((d[0] + seed) % 3) as f32 * 0.4
        });
        BatchNormLayer::new(&g, &b, &m, &v, 1e-5).expect("bn")
    };
    let bn = mk_bn(2, 1);
    // Pre-activation of bn_a: some finite (2,2,2) box.
    let pre = BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 2, 2]), -1.0f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 2, 2]), 1.0f32),
    )
    .expect("pre");
    let after_bn = bn
        .propagate_linear_with_bounds(&after_convt, &pre)
        .expect("bn backward");
    println!("after bn backward: nonfinite {:?}", nonfinite(&after_bn));
    // Concretize each over a small box at the bn input / convt input dims.
    let box8 = BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[8]), -0.1f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[8]), 0.1f32),
    )
    .expect("box8");
    let c1 = after_convt.concretize_sound(&box8);
    let c2 = after_bn.concretize_sound(&box8);
    let count_inf = |b: &BoundedTensor| {
        b.lower()
            .iter()
            .chain(b.upper().iter())
            .filter(|v| !v.is_finite())
            .count()
    };
    println!("concretize(after_convt) nonfinite: {}", count_inf(&c1));
    println!("concretize(after_bn) nonfinite: {}", count_inf(&c2));

    // Continue the exact collection chain: reshape passthrough, then Linear
    // backward, then concretize at the 5-d input box.
    let in_dim = 5usize;
    let w = Array2::from_shape_fn((8, in_dim), |(i, j)| {
        (((i * 7 + j * 3) % 11) as f32 * 0.21 - 1.0) * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
    });
    let lin = LinearLayer::new(w, None).expect("lin");
    use ny_propagate::BoundPropagation;
    let after_lin = lin
        .propagate_linear(&after_bn)
        .expect("lin backward")
        .into_owned();
    println!("after lin backward: nonfinite {:?}", nonfinite(&after_lin));
    let input5 = BoundedTensor::new(
        (ndarray::Array1::from_elem(in_dim, 0.3f32) - 0.01f32).into_dyn(),
        (ndarray::Array1::from_elem(in_dim, 0.3f32) + 0.01f32).into_dyn(),
    )
    .expect("input5");
    let cfinal = after_lin.concretize_sound(&input5);
    println!("final concretize nonfinite: {}", count_inf(&cfinal));
    let mut wmax = 0.0f32;
    for (l, u) in cfinal.lower().iter().zip(cfinal.upper().iter()) {
        wmax = wmax.max(u - l);
    }
    println!("final concretize max width: {wmax:.6e}");
}

/// Alpha-init probe (#cgan lever 1, alpha-on-tight-refs): replicate the DAG
/// alpha-CROWN init's per-ReLU unstable-neuron count from the CROWN-IBP
/// collection (the same reference map the BaB alpha warmup consumes). If the
/// total is 0 the warmup early-returns without a single alpha iteration
/// ("No optimizable activation state") even though the root spec bound is
/// hundreds of units wide.
#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_alpha_unstable_probe() {
    use ny_propagate::Layer;

    let onnx = Path::new(ONNX).to_path_buf();
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("load onnx");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph");

    let input = BoundedTensor::new(
        ndarray::arr1(&LOWER).into_dyn(),
        ndarray::arr1(&UPPER).into_dyn(),
    )
    .expect("input box");

    let t0 = std::time::Instant::now();
    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .expect("crown-ibp");
    println!(
        "CROWN-IBP collection: {:.2}s ({} node sets)",
        t0.elapsed().as_secs_f64(),
        crown_ibp.bounds.len(),
    );
    let bounds = crown_ibp.bounds;

    let order = graph.exec_order().expect("order");
    let mut total_unstable = 0usize;
    println!(
        "\n{:<16} {:<24} {:>8} {:>10} {:>10} {:>10}",
        "relu", "producer", "neurons", "unstable", "stable_pos", "stable_neg"
    );
    for name in order {
        let node = graph.node(name).expect("node");
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let producer = node.inputs().first().expect("relu producer");
        let Some(pre) = bounds.get(producer) else {
            println!("{name:<16} {producer:<24} MISSING PRE-ACTIVATION BOUNDS");
            continue;
        };
        let mut unstable = 0usize;
        let mut stable_pos = 0usize;
        let mut stable_neg = 0usize;
        for (l, u) in pre.lower().iter().zip(pre.upper().iter()) {
            if *l < 0.0 && *u > 0.0 {
                unstable += 1;
            } else if *l >= 0.0 {
                stable_pos += 1;
            } else {
                stable_neg += 1;
            }
        }
        total_unstable += unstable;
        println!(
            "{:<16} {:<24} {:>8} {:>10} {:>10} {:>10}",
            name,
            producer,
            pre.lower().len(),
            unstable,
            stable_pos,
            stable_neg
        );
    }
    println!("\ntotal unstable: {total_unstable}");
}

/// Warmup-replica probe (#cgan lever 1): call the exact BaB-lane alpha warmup
/// entry (`collect_alpha_crown_bounds_dag_with_engine`) on the real prop_1
/// graph/box with debug tracing, to explain why the in-loop DAG alpha backward
/// produces lower_sum ~-7.7e7 at iteration 0 while the plain per-node CROWN
/// collection realizes an output bound of ~[-128.7, 139.3] (iteration-0 must
/// equal plain CROWN by the documented invariant).
#[test]
#[ignore = "needs the local VNN-COMP 2025 benchmark checkout; run with --ignored"]
fn cgan_alpha_warmup_replica_probe() {
    use ny_propagate::AlphaCrownConfig;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("ny_propagate=debug")
        .with_test_writer()
        .try_init();

    let onnx = Path::new(ONNX).to_path_buf();
    let model = load_onnx_with_config(&onnx, &OnnxLoadConfig::default()).expect("load onnx");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph");

    let input = BoundedTensor::new(
        ndarray::arr1(&LOWER).into_dyn(),
        ndarray::arr1(&UPPER).into_dyn(),
    )
    .expect("input box");

    // NY_PROBE_ALPHA_ITERS / NY_PROBE_ALPHA_PATIENCE override the loop length
    // so the raw per-iteration trajectory can be measured past the default
    // early stop (the merged-best baseline kills it at patience=10 otherwise).
    let iters: usize = std::env::var("NY_PROBE_ALPHA_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let patience: usize = std::env::var("NY_PROBE_ALPHA_PATIENCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let config = AlphaCrownConfig {
        iterations: iters,
        learning_rate: 0.2,
        early_stop_patience: patience,
        ..AlphaCrownConfig::default()
    };
    // Patches-mode discriminator: NY_PROBE_NO_PATCHES=1 disables Dense->Patches
    // re-entry in the backward, isolating whether the ~-7.7e7 iter-0 lower_sum
    // comes from the patches segment (compose or ensure_dense materialization).
    let mut graph = graph;
    if std::env::var("NY_PROBE_NO_PATCHES").as_deref() == Ok("1") {
        graph.set_use_patches_mode(false);
        println!("probe: use_patches_mode = false");
    }
    let t0 = std::time::Instant::now();
    let (bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag_with_engine(&input, &config, None)
        .expect("alpha collect");
    println!(
        "warmup replica: {:.2}s, {} bound sets, {} unstable",
        t0.elapsed().as_secs_f64(),
        bounds.len(),
        alpha_state.num_unstable()
    );
    let order = graph.exec_order().expect("order");
    let out = order.last().expect("output node");
    if let Some(b) = bounds.get(out) {
        println!(
            "output node '{}' bound: [{:.6}, {:.6}]",
            out,
            b.lower()[0],
            b.upper()[0]
        );
    }
}
