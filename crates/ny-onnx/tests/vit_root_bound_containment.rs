// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-bank-gate (3): root bound CONTAINMENT on the REAL vit_2023 ONNX,
//! including the CLS-token `Concat` and the positional-embedding consumer
//! after it.
//!
//! # Why
//!
//! On 2026-08-25 an N-ary `Concat` pairing defect published finite,
//! plausible, NON-containing bounds (`±1.18e-38` for a true `[-1, 1]`) and
//! minted 21 false vit_2023 unsats (`531b658e1`, retracted `d63a62511`). The
//! repair added a loader-accurate FIXTURE; `vit_forward_matches_ort.rs` checks
//! the loaded function against ORT but never a bound. Nothing asserted that
//! ny's bounds on the real vit graph CONTAIN the network. This does:
//!
//! 1. sample deterministic points of one real property's input box
//!    (`ibp_3_3_8_9175`, a row of the 2026-08-30 alarm);
//! 2. evaluate every node at each point through the SOUND f64 fence walk
//!    (`GraphNetwork::propagate_f64_fence_nodes`), which shares no code with
//!    IBP/CROWN/alpha and is the evaluator the lane fence itself uses — so this
//!    test ALSO proves the lane fence can evaluate the whole vit graph
//!    (Softmax, ReduceMean) instead of disarming as inconclusive;
//! 3. assert that ny's per-node IBP bounds, and the multi-objective lane's own
//!    root bootstrap (`compute_initial_graph_bounds`, the alarm's
//!    `alpha iterations: 1` and plain CROWN), contain every point value
//!    beyond the fence's f32 band, node by node — with the CLS `Concat` and its
//!    consumer REQUIRED to be among the compared nodes;
//! 4. (#vit-crown-interm, #softmax-fixed-index) do the same for the typed
//!    `fix_interm_bounds: false` CROWN-IBP reference route (`alpha1_fi`,
//!    `alpha1_fi_cap`) and for the fixed-index softmax REWRITE, whose bounds
//!    are checked against the fence walk of the UNREWRITTEN graph;
//! 5. (#vit-margin-alpha-complex) check that the certified margin ascent
//!    running THROUGH the fixed-index softmax (MulBinary / Reciprocal α
//!    frozen) publishes margins below the UNREWRITTEN graph's sampled values.
//!
//! # When the fixture is absent
//!
//! The corpus under `benchmarks/vnncomp2026` is gitignored and absent from
//! worktrees. Each test then prints a `[vit-containment] SKIP` line and
//! returns. A SKIP IS VOID, NOT A PASS: a gate must require the
//! `[vit-containment] test=... PASS` line (and the test's own `... ok`).
//!
//! Two traps, both of which turn this gate VOID:
//!
//! * `benchmarks/` itself is a TRACKED directory (17 files), so
//!   `ln -s <main>/benchmarks <worktree>/benchmarks` creates
//!   `benchmarks/benchmarks` and the test still skips. Link the corpus
//!   directory instead (it is gitignored, so `git status` stays clean):
//!   `ln -s <home>/ny/benchmarks/vnncomp2026 <worktree>/benchmarks/vnncomp2026`.
//! * libtest CAPTURES a passing test's stderr, so the `PASS`/`SKIP` lines
//!   are only visible with `-- --nocapture` (or `--show-output`). Without it a
//!   green run and a skipped run print the same `... ok`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig};
use ny_propagate::bounds::{AlphaSpecAscent, AlphaSpecEarlyExit};
use ny_propagate::containment_fence::{
    domain_fence_sample_points, first_band_separation, DOMAIN_FENCE_REL_BAND,
};
use ny_propagate::{
    AlphaCrownConfig, BetaCrownConfig, BetaCrownVerifier, GraphNetwork, Interval64, Layer,
};
use ny_tensor::BoundedTensor;

const MODEL: &str = "benchmarks/vnncomp2026/benchmarks/vit_2023/1.0/onnx/ibp_3_3_8.onnx";
const PROPERTY: &str =
    "benchmarks/vnncomp2026/benchmarks/vit_2023/1.0/vnnlib/ibp_3_3_8_9175.vnnlib";
const INPUT_SHAPE: [usize; 3] = [3, 32, 32];
const SAMPLES: usize = 16;

struct Fixture {
    graph: GraphNetwork,
    input: BoundedTensor,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Load the real model and property exactly the way the CLI's graph lane
/// does (`DecomposeNormalization`), or print a loud SKIP.
fn load_fixture(test: &str) -> Option<Fixture> {
    let root = repo_root();
    let model = root.join(MODEL);
    let property = root.join(PROPERTY);
    if !model.is_file() || !property.is_file() {
        eprintln!(
            "[vit-containment] SKIP test={test}: fixture absent (model {} present={}, property \
             {} present={}). Worktrees carry no benchmarks/vnncomp2026 — symlink the main \
             tree's benchmarks/vnncomp2026 into this tree's (tracked) benchmarks/ and re-run \
             with -- --nocapture. A SKIP is VOID, not a pass.",
            model.display(),
            model.is_file(),
            property.display(),
            property.is_file()
        );
        return None;
    }
    let onnx = load_onnx_with_config(&model, &OnnxLoadConfig::default()).expect("vit ONNX loads");
    let graph = onnx
        .to_graph_network_with_options(GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        })
        .expect("vit converts to a GraphNetwork");
    let spec = load_vnnlib(&property).expect("vit property parses");
    let n: usize = INPUT_SHAPE.iter().product();
    assert_eq!(spec.num_inputs, n, "vit input is 3x32x32");
    assert_eq!(spec.input_bounds.len(), n);
    // Nearest-rounded f32 is monotone, so lower <= upper survives the cast.
    let lower32: Vec<f32> = spec.input_bounds.iter().map(|(l, _)| *l as f32).collect();
    let upper32: Vec<f32> = spec.input_bounds.iter().map(|(_, u)| *u as f32).collect();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&INPUT_SHAPE), lower32.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&INPUT_SHAPE), upper32.clone()).unwrap(),
    )
    .expect("the property box is a valid BoundedTensor");
    Some(Fixture {
        graph,
        input,
        lower: lower32.iter().map(|v| f64::from(*v)).collect(),
        upper: upper32.iter().map(|v| f64::from(*v)).collect(),
    })
}

/// Same row-major layout once leading singleton axes are ignored.
fn same_layout(a: &[usize], b: &[usize]) -> bool {
    let strip = |shape: &[usize]| -> Vec<usize> {
        shape.iter().copied().skip_while(|&dim| dim == 1).collect()
    };
    strip(a) == strip(b)
}

/// The CLS-token `Concat` nodes (a constant slot beside a dynamic edge — the
/// 2026-08-25 shape) and every node that consumes one of them (the
/// positional-embedding add).
fn cls_concat_and_consumers(graph: &GraphNetwork) -> (Vec<String>, Vec<String>) {
    let mut concats = Vec::new();
    for name in graph.node_names() {
        if let Some(node) = graph.node(name) {
            if let Layer::Concat(concat) = node.layer() {
                if concat.constant_inputs.is_some() {
                    concats.push(name.clone());
                }
            }
        }
    }
    let mut consumers = Vec::new();
    for name in graph.node_names() {
        if let Some(node) = graph.node(name) {
            if node.inputs().iter().any(|input| concats.contains(input)) {
                consumers.push(name.clone());
            }
        }
    }
    (concats, consumers)
}

/// One sound f64 walk per sample point, every node.
fn sample_walks(fixture: &Fixture) -> Vec<HashMap<String, Interval64>> {
    let points = domain_fence_sample_points(&fixture.lower, &fixture.upper, SAMPLES);
    assert_eq!(
        points.len(),
        SAMPLES,
        "a 3072-dim box yields the full sample set"
    );
    points
        .into_iter()
        .map(|point| {
            let array = ArrayD::from_shape_vec(IxDyn(&INPUT_SHAPE), point).unwrap();
            fixture
                .graph
                .propagate_f64_fence_nodes(&Interval64::point(array))
                .expect(
                    "the fence walk must evaluate EVERY vit node (Softmax and ReduceMean \
                     included); otherwise the lane fence is inconclusive on vit",
                )
        })
        .collect()
}

/// What a containment sweep compared.
struct Sweep {
    nodes_compared: Vec<String>,
    elements: usize,
    skipped_layout: usize,
    min_slack: f64,
}

/// Compare certified per-node bounds against every walk; panic on the first
/// separation beyond the band, naming node, coordinate and sample.
fn assert_contained(
    label: &str,
    certified: &HashMap<String, BoundedTensor>,
    walks: &[HashMap<String, Interval64>],
) -> Sweep {
    let mut sweep = Sweep {
        nodes_compared: Vec::new(),
        elements: 0,
        skipped_layout: 0,
        min_slack: f64::INFINITY,
    };
    let mut names: Vec<&String> = certified.keys().collect();
    names.sort();
    for name in names {
        let bounds = &certified[name];
        let cert_lower: Vec<f64> = bounds.lower().iter().map(|v| f64::from(*v)).collect();
        let cert_upper: Vec<f64> = bounds.upper().iter().map(|v| f64::from(*v)).collect();
        let mut compared = false;
        for (sample, walk) in walks.iter().enumerate() {
            let Some(observed) = walk.get(name.as_str()) else {
                continue;
            };
            if observed.lower.len() != cert_lower.len()
                || !same_layout(bounds.shape(), observed.lower.shape())
            {
                if sample == 0 {
                    sweep.skipped_layout += 1;
                }
                continue;
            }
            let obs_lower: Vec<f64> = observed.lower.iter().copied().collect();
            let obs_upper: Vec<f64> = observed.upper.iter().copied().collect();
            if let Some(separation) = first_band_separation(
                &cert_lower,
                &cert_upper,
                &obs_lower,
                &obs_upper,
                DOMAIN_FENCE_REL_BAND,
                sample,
            ) {
                panic!(
                    "[vit-containment] {label}: node '{name}' ({:?}) does NOT contain the \
                     network: {} — a certified vit bound that excludes the network's own \
                     value is a false-proof path (the 2026-08-25 class). SOUNDNESS EMERGENCY.",
                    bounds.shape(),
                    separation.describe()
                );
            }
            for i in 0..cert_lower.len() {
                let slack = (obs_lower[i] - cert_lower[i]).min(cert_upper[i] - obs_upper[i]);
                if slack.is_finite() && slack < sweep.min_slack {
                    sweep.min_slack = slack;
                }
            }
            sweep.elements += cert_lower.len();
            compared = true;
        }
        if compared {
            sweep.nodes_compared.push(name.clone());
        }
    }
    sweep
}

/// Per-node IBP bounds of the real vit graph contain every sampled point,
/// and the CLS `Concat` plus its consumer are among the compared nodes.
#[test]
fn vit_bank_gate_real_onnx_ibp_bounds_contain_samples_through_cls_concat() {
    let test = "ibp";
    let Some(fixture) = load_fixture(test) else {
        return;
    };
    let (concats, consumers) = cls_concat_and_consumers(&fixture.graph);
    assert!(
        !concats.is_empty(),
        "the loaded vit graph must carry the CLS-token Concat (a constant slot beside a \
         dynamic edge); if the loader now represents it differently, re-aim this test"
    );
    let ibp = fixture
        .graph
        .collect_node_bounds(&fixture.input)
        .expect("per-node IBP on the real vit box");
    let walks = sample_walks(&fixture);
    let sweep = assert_contained(test, &ibp, &walks);
    for name in concats.iter().chain(consumers.iter()) {
        assert!(
            sweep.nodes_compared.contains(name),
            "CLS path node '{name}' was not compared (absent from the IBP map or a layout \
             mismatch) — the test would not see the 2026-08-25 shape"
        );
    }
    eprintln!(
        "[vit-containment] test={test} PASS samples={} nodes_compared={} elements={} \
         skipped_layout={} cls_concat={:?} cls_consumers={:?} min_slack={:e}",
        walks.len(),
        sweep.nodes_compared.len(),
        sweep.elements,
        sweep.skipped_layout,
        concats,
        consumers,
        sweep.min_slack
    );
}

/// The multi-objective lane's own root bootstrap — the node map every BaB
/// child is built from — contains every sampled point, under plain CROWN and
/// under the 2026-08-30 alarm's `alpha iterations: 1`, and so does its root
/// output enclosure.
///
/// #vit-crown-interm adds the typed `fix_interm_bounds: false` route twice:
/// `alpha1_fi` (the uncapped CROWN-IBP reference map, then the post-loop
/// all-node alpha pass) and `alpha1_fi_cap` (the same route under
/// `crown_ibp_reference_cap_secs`, which publishes the optimizer's own map and
/// caps the post-loop pass). A gate must require BOTH
/// `[vit-containment] test=bootstrap/alpha1_fi PASS` and
/// `[vit-containment] test=bootstrap/alpha1_fi_cap PASS`.
#[test]
fn vit_bank_gate_real_onnx_root_bootstrap_bounds_contain_samples() {
    let test = "bootstrap";
    let Some(fixture) = load_fixture(test) else {
        return;
    };
    let walks = sample_walks(&fixture);
    let (concats, _) = cls_concat_and_consumers(&fixture.graph);
    let output_name = fixture.graph.output_name().to_string();

    for (arm, use_alpha, fix_interm_bounds, reference_cap) in [
        ("crown", false, true, None),
        ("alpha1", true, true, None),
        ("alpha1_fi", true, false, None),
        ("alpha1_fi_cap", true, false, Some(10)),
    ] {
        if cfg!(debug_assertions) && reference_cap.is_none() && !fix_interm_bounds {
            // The UNCAPPED CROWN-IBP reference route overran even 1500 s in the unoptimized test
            // profile (int/wave3 v4 and int/wave4 gates, 2026-09-25/26). A timeout is not a
            // containment result; `alpha1_fi_cap` exercises the same typed route and must PASS.
            // Printed on the raw stderr so a gate sees the SKIP without --nocapture.
            use std::io::Write as _;
            let _ = writeln!(
                std::io::stderr(),
                "[vit-containment] test={test}/{arm} SKIP debug-profile (uncapped route; covered by {test}/alpha1_fi_cap)"
            );
            continue;
        }
        let base = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            use_alpha_crown: use_alpha,
            alpha_config: AlphaCrownConfig {
                iterations: 1,
                fix_interm_bounds,
                crown_ibp_reference_cap_secs: reference_cap,
                ..base.alpha_config.clone()
            },
            ..base
        };
        let verifier = BetaCrownVerifier::new(config);
        let started = Instant::now();
        let (node_bounds, output) = verifier
            .compute_initial_graph_bounds(
                &fixture.graph,
                &fixture.input,
                // The UNCAPPED CROWN-IBP reference route runs its per-node collection and the
                // post-loop all-node alpha pass to completion; in the unoptimized test profile that
                // overran 300 s (int/wave3 v4 gate, 2026-09-25: 89 deadline hits after 166 s of the
                // alpha pass, then DeadlineExceeded). A timeout is not a containment result, so this
                // arm gets room to finish; the capped arm keeps the 300 s bound.
                Some(
                    Instant::now()
                        + Duration::from_secs(if reference_cap.is_none() && !fix_interm_bounds {
                            1_500
                        } else {
                            300
                        }),
                ),
            )
            .expect("the lane's root bootstrap runs on the real vit box");
        let label = format!("{test}/{arm}");
        let sweep = assert_contained(&label, &node_bounds, &walks);
        assert!(
            !sweep.nodes_compared.is_empty(),
            "{label}: no bootstrap node was comparable — the sweep checked nothing"
        );

        let mut outputs = HashMap::new();
        outputs.insert(output_name.clone(), output);
        let output_sweep = assert_contained(&format!("{label}/output"), &outputs, &walks);
        assert_eq!(
            output_sweep.nodes_compared.len(),
            1,
            "{label}: the root output enclosure must be compared"
        );
        let concat_compared = concats
            .iter()
            .filter(|name| sweep.nodes_compared.contains(name))
            .count();
        eprintln!(
            "[vit-containment] test={label} PASS samples={} nodes_compared={} elements={} \
             skipped_layout={} cls_concat_compared={}/{} min_slack={:e} output_min_slack={:e} \
             bootstrap_secs={:.2}",
            walks.len(),
            sweep.nodes_compared.len(),
            sweep.elements,
            sweep.skipped_layout,
            concat_compared,
            concats.len(),
            sweep.min_slack,
            output_sweep.min_slack,
            started.elapsed().as_secs_f64()
        );
    }
}

/// #softmax-fixed-index: the REWRITTEN vit graph
/// (`softmax: complex_fixed_index`) contains the ORIGINAL network.
///
/// The ground truth is the f64 fence walk of the UNREWRITTEN graph, so this
/// compares the rewrite against the function it replaces, not against itself:
/// every node the two graphs share by name — including each softmax output
/// (now the closing `Clip(0, 1)`), everything downstream of it, the CLS
/// `Concat` and the logits — must contain every sampled point, under plain
/// IBP and under the lane's root bootstrap (`alpha1`, and `alpha1_fi_cap`: the
/// typed CROWN-IBP reference route the rewrite is built to feed). Every
/// softmax must decompose and every softmax output must be compared.
///
/// A gate must require `[vit-containment] test=softmax_fixed_index/ibp PASS`,
/// `.../alpha1 PASS` and `.../alpha1_fi_cap PASS`; a SKIP is VOID.
#[test]
fn vit_bank_gate_real_onnx_softmax_fixed_index_bounds_contain_samples() {
    let test = "softmax_fixed_index";
    let Some(original) = load_fixture(test) else {
        return;
    };
    // Ground truth from the graph BEFORE the rewrite.
    let walks = sample_walks(&original);
    let Some(mut fixture) = load_fixture(test) else {
        return;
    };
    let softmax_nodes: Vec<String> = fixture
        .graph
        .node_names()
        .iter()
        .filter(|name| {
            matches!(
                fixture.graph.node(name.as_str()).map(|node| node.layer()),
                Some(Layer::Softmax(_))
            )
        })
        .cloned()
        .collect();
    assert!(
        !softmax_nodes.is_empty(),
        "the loaded vit graph must carry Softmax nodes; if the loader now fuses them, \
         re-aim this test"
    );
    let report = fixture
        .graph
        .decompose_softmax_complex_fixed_index(&fixture.input)
        .expect("the fixed-index rewrite runs on the real vit box");
    assert_eq!(
        report.decomposed.len(),
        softmax_nodes.len(),
        "every vit softmax must take the fixed-index rewrite; skipped: {:?}",
        report.skipped
    );
    for name in &softmax_nodes {
        assert!(
            matches!(
                fixture.graph.node(name.as_str()).map(|node| node.layer()),
                Some(Layer::Clip(_))
            ),
            "softmax '{name}' must close as Clip(0, 1) under its original name"
        );
    }
    let require_softmax_compared = |label: &str, sweep: &Sweep| {
        for name in &softmax_nodes {
            assert!(
                sweep.nodes_compared.contains(name),
                "{label}: softmax output '{name}' was not compared (absent or a layout \
                 mismatch) — the test would not see the rewrite"
            );
        }
    };

    let ibp = fixture
        .graph
        .collect_node_bounds(&fixture.input)
        .expect("per-node IBP on the rewritten vit graph");
    let label = format!("{test}/ibp");
    let sweep = assert_contained(&label, &ibp, &walks);
    require_softmax_compared(&label, &sweep);
    eprintln!(
        "[vit-containment] test={label} PASS samples={} nodes_compared={} elements={} \
         skipped_layout={} softmax_decomposed={}/{} wide_ibp={} min_slack={:e}",
        walks.len(),
        sweep.nodes_compared.len(),
        sweep.elements,
        sweep.skipped_layout,
        report.decomposed.len(),
        softmax_nodes.len(),
        report.wide_ibp.len(),
        sweep.min_slack
    );

    let output_name = fixture.graph.output_name().to_string();
    for (arm, fix_interm_bounds, reference_cap) in
        [("alpha1", true, None), ("alpha1_fi_cap", false, Some(15))]
    {
        let base = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            use_alpha_crown: true,
            alpha_config: AlphaCrownConfig {
                iterations: 1,
                fix_interm_bounds,
                crown_ibp_reference_cap_secs: reference_cap,
                ..base.alpha_config.clone()
            },
            ..base
        };
        let verifier = BetaCrownVerifier::new(config);
        let started = Instant::now();
        let (node_bounds, output) = verifier
            .compute_initial_graph_bounds(
                &fixture.graph,
                &fixture.input,
                Some(Instant::now() + Duration::from_secs(300)),
            )
            .expect("the lane's root bootstrap runs on the rewritten vit graph");
        let label = format!("{test}/{arm}");
        let sweep = assert_contained(&label, &node_bounds, &walks);
        require_softmax_compared(&label, &sweep);
        let mut outputs = HashMap::new();
        outputs.insert(output_name.clone(), output);
        let output_sweep = assert_contained(&format!("{label}/output"), &outputs, &walks);
        assert_eq!(
            output_sweep.nodes_compared.len(),
            1,
            "{label}: the root output enclosure must be compared"
        );
        eprintln!(
            "[vit-containment] test={label} PASS samples={} nodes_compared={} elements={} \
             skipped_layout={} min_slack={:e} output_min_slack={:e} bootstrap_secs={:.2}",
            walks.len(),
            sweep.nodes_compared.len(),
            sweep.elements,
            sweep.skipped_layout,
            sweep.min_slack,
            output_sweep.min_slack,
            started.elapsed().as_secs_f64()
        );
    }
}

/// #vit-margin-alpha: the certified C-row margin ascent's CERTIFY-ONCE
/// publication on the REAL vit graph never exceeds the network's own margin
/// on any sampled point of the box.
///
/// The rows are the vit property shape `Y_t − Y_i` (i ≠ t, threshold 0) with
/// `t` the class the f64 fence walk assigns at the box centre — every
/// vit_2023 row is truth-unsat, so `t` is the property's label; containment
/// itself does not depend on which rows are used. The network evaluator is the
/// sound f64 fence walk (`propagate_f64_fence_nodes`), which
/// `vit_forward_matches_ort.rs` pins to ONNX Runtime; the comparison allows the
/// same f32 band as every other arm in this file.
///
/// The lane runs with EVERY part armed — shared ReLU α, one δ slot per row,
/// and the attention McCormick parameters (`matmul_alpha`) — and the test
/// REQUIRES a publication: a lane that did not engage is a failure here, not a
/// pass. Required line: `[vit-containment] test=margin_alpha PASS`.
#[test]
fn vit_margin_alpha_root_bound_below_ort_min() {
    let test = "margin_alpha";
    let Some(fixture) = load_fixture(test) else {
        return;
    };
    let output_name = fixture.graph.output_name().to_string();

    // Target class at the box centre.
    let centre: Vec<f64> = fixture
        .lower
        .iter()
        .zip(&fixture.upper)
        .map(|(lo, hi)| 0.5 * (lo + hi))
        .collect();
    let centre_walk = fixture
        .graph
        .propagate_f64_fence_nodes(&Interval64::point(
            ArrayD::from_shape_vec(IxDyn(&INPUT_SHAPE), centre).unwrap(),
        ))
        .expect("the fence walk evaluates the vit centre");
    let centre_out = centre_walk
        .get(output_name.as_str())
        .expect("the output node is evaluated");
    let logits: Vec<f64> = centre_out
        .lower
        .iter()
        .zip(centre_out.upper.iter())
        .map(|(lo, hi)| 0.5 * (lo + hi))
        .collect();
    let classes = logits.len();
    assert_eq!(classes, 10, "vit_2023 has ten classes");
    let target = (0..classes)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .expect("non-empty logits");
    let rows: Vec<AlphaSpecEarlyExit> = (0..classes)
        .filter(|&i| i != target)
        .map(|i| {
            let mut objective = vec![0.0_f32; classes];
            objective[target] = 1.0;
            objective[i] = -1.0;
            AlphaSpecEarlyExit {
                objective,
                threshold: 0.0,
                verify_upper_bound: false,
            }
        })
        .collect();

    let config = AlphaCrownConfig {
        iterations: 8,
        learning_rate: 0.5,
        early_stop_patience: 1000,
        adaptive_skip: false,
        root_alpha_margin: true,
        root_alpha_margin_gradient: true,
        matmul_alpha: true,
        alpha_spec_slots: rows.len(),
        spec_ascent: AlphaSpecAscent::new(rows.clone()),
        deadline: Some(Instant::now() + Duration::from_secs(300)),
        ..AlphaCrownConfig::default()
    };
    let started = Instant::now();
    let (_, alpha_state) = fixture
        .graph
        .collect_alpha_crown_bounds_dag_with_engine(&fixture.input, &config, None)
        .expect("the DAG alpha collection runs on the real vit box");
    let publication = alpha_state.margin_alpha_publication().unwrap_or_else(|| {
        panic!(
            "[vit-containment] {test}: the certified C-row margin ascent did not publish on \
             the real vit graph — the lane never engaged (check the `[alpha-margin] armed=` \
             line); a missing publication is a FAILURE of this arm, not a pass"
        )
    });
    assert_eq!(publication.lower().len(), rows.len());
    assert!(
        publication.matmul_alpha_nodes() > 0,
        "vit's attention products must carry interpolation parameters in the lane"
    );

    let cert_lower: Vec<f64> = publication.lower().iter().map(|v| f64::from(*v)).collect();
    let cert_upper: Vec<f64> = publication.upper().iter().map(|v| f64::from(*v)).collect();
    let walks = sample_walks(&fixture);
    let mut min_observed = vec![f64::INFINITY; rows.len()];
    for (sample, walk) in walks.iter().enumerate() {
        let out = walk
            .get(output_name.as_str())
            .expect("the output node is evaluated at every sample");
        let y_lo: Vec<f64> = out.lower.iter().copied().collect();
        let y_hi: Vec<f64> = out.upper.iter().copied().collect();
        // Interval value of each margin row at this point.
        let (mut obs_lower, mut obs_upper) = (Vec::new(), Vec::new());
        for row in &rows {
            let (mut lo, mut hi) = (0.0_f64, 0.0_f64);
            for (c, &coefficient) in row.objective.iter().enumerate() {
                let c64 = f64::from(coefficient);
                if c64 >= 0.0 {
                    lo += c64 * y_lo[c];
                    hi += c64 * y_hi[c];
                } else {
                    lo += c64 * y_hi[c];
                    hi += c64 * y_lo[c];
                }
            }
            obs_lower.push(lo);
            obs_upper.push(hi);
        }
        if let Some(separation) = first_band_separation(
            &cert_lower,
            &cert_upper,
            &obs_lower,
            &obs_upper,
            DOMAIN_FENCE_REL_BAND,
            sample,
        ) {
            panic!(
                "[vit-containment] {test}: the certify-once margin publication does NOT \
                 contain the network: {} — a published margin above the network's own value \
                 is a false-proof path. SOUNDNESS EMERGENCY.",
                separation.describe()
            );
        }
        for (slot, &value) in min_observed.iter_mut().zip(&obs_lower) {
            *slot = slot.min(value);
        }
    }
    let worst_gap = cert_lower
        .iter()
        .zip(&min_observed)
        .map(|(cert, observed)| observed - cert)
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "[vit-containment] test={test} PASS rows={} target={target} samples={} \
         committed={} spec_slots={} matmul_alpha_nodes={} min_published_lower={:e} \
         min_gap_to_sampled_min={:e} secs={:.2}",
        rows.len(),
        walks.len(),
        publication.committed_iteration(),
        publication.spec_slots(),
        publication.matmul_alpha_nodes(),
        cert_lower.iter().copied().fold(f64::INFINITY, f64::min),
        worst_gap,
        started.elapsed().as_secs_f64()
    );
}

/// #vit-margin-alpha-complex (P6): the certified C-row margin ascent running
/// WITH the fixed-index softmax (MulBinary / Reciprocal α frozen at their
/// DAG-init values under `margin_alpha_freeze_mul_reciprocal`) on the REAL vit
/// graph never publishes a margin above the network's own value at any
/// sampled point of the box.
///
/// Ground truth is the sound f64 fence walk of the UNREWRITTEN graph (pinned to
/// ONNX Runtime by `vit_forward_matches_ort.rs`), so a defect in the rewrite,
/// in the frozen relaxations or in the lane's certify-once pass all show up as
/// a separation. The lane must ENGAGE: a missing publication is a failure of
/// this arm, not a pass. Required line:
/// `[vit-containment] test=margin_alpha_frozen_complex PASS`; a SKIP is VOID.
#[test]
fn vit_margin_alpha_frozen_complex_softmax_root_bound_below_sampled_min() {
    let test = "margin_alpha_frozen_complex";
    let Some(original) = load_fixture(test) else {
        return;
    };
    // Ground truth from the graph BEFORE the rewrite.
    let walks = sample_walks(&original);
    let output_name = original.graph.output_name().to_string();
    let Some(mut fixture) = load_fixture(test) else {
        return;
    };
    let report = fixture
        .graph
        .decompose_softmax_complex_fixed_index(&fixture.input)
        .expect("the fixed-index rewrite runs on the real vit box");
    assert!(
        !report.decomposed.is_empty(),
        "the vit softmaxes must take the fixed-index rewrite; skipped: {:?}",
        report.skipped
    );

    // Target class at the box centre, from the UNREWRITTEN graph.
    let centre: Vec<f64> = fixture
        .lower
        .iter()
        .zip(&fixture.upper)
        .map(|(lo, hi)| 0.5 * (lo + hi))
        .collect();
    let centre_walk = original
        .graph
        .propagate_f64_fence_nodes(&Interval64::point(
            ArrayD::from_shape_vec(IxDyn(&INPUT_SHAPE), centre).unwrap(),
        ))
        .expect("the fence walk evaluates the vit centre");
    let centre_out = centre_walk
        .get(output_name.as_str())
        .expect("the output node is evaluated");
    let logits: Vec<f64> = centre_out
        .lower
        .iter()
        .zip(centre_out.upper.iter())
        .map(|(lo, hi)| 0.5 * (lo + hi))
        .collect();
    let classes = logits.len();
    assert_eq!(classes, 10, "vit_2023 has ten classes");
    let target = (0..classes)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .expect("non-empty logits");
    let rows: Vec<AlphaSpecEarlyExit> = (0..classes)
        .filter(|&i| i != target)
        .map(|i| {
            let mut objective = vec![0.0_f32; classes];
            objective[target] = 1.0;
            objective[i] = -1.0;
            AlphaSpecEarlyExit {
                objective,
                threshold: 0.0,
                verify_upper_bound: false,
            }
        })
        .collect();

    let config = AlphaCrownConfig {
        iterations: 8,
        learning_rate: 0.5,
        early_stop_patience: 1000,
        adaptive_skip: false,
        root_alpha_margin: true,
        root_alpha_margin_gradient: true,
        matmul_alpha: true,
        margin_alpha_freeze_mul_reciprocal: true,
        alpha_spec_slots: rows.len(),
        spec_ascent: AlphaSpecAscent::new(rows.clone()),
        deadline: Some(Instant::now() + Duration::from_secs(300)),
        ..AlphaCrownConfig::default()
    };
    let started = Instant::now();
    let (_, alpha_state) = fixture
        .graph
        .collect_alpha_crown_bounds_dag_with_engine(&fixture.input, &config, None)
        .expect("the DAG alpha collection runs on the real rewritten vit box");
    let publication = alpha_state.margin_alpha_publication().unwrap_or_else(|| {
        panic!(
            "[vit-containment] {test}: the margin ascent did not publish on the fixed-index \
             vit graph — the lane never engaged (check `[alpha-margin-frozen] … reason=` and \
             `[alpha-margin] armed=`); a missing publication is a FAILURE of this arm"
        )
    });
    assert_eq!(publication.lower().len(), rows.len());

    let cert_lower: Vec<f64> = publication.lower().iter().map(|v| f64::from(*v)).collect();
    let cert_upper: Vec<f64> = publication.upper().iter().map(|v| f64::from(*v)).collect();
    let mut min_observed = vec![f64::INFINITY; rows.len()];
    for (sample, walk) in walks.iter().enumerate() {
        let out = walk
            .get(output_name.as_str())
            .expect("the output node is evaluated at every sample");
        let y_lo: Vec<f64> = out.lower.iter().copied().collect();
        let y_hi: Vec<f64> = out.upper.iter().copied().collect();
        let (mut obs_lower, mut obs_upper) = (Vec::new(), Vec::new());
        for row in &rows {
            let (mut lo, mut hi) = (0.0_f64, 0.0_f64);
            for (c, &coefficient) in row.objective.iter().enumerate() {
                let c64 = f64::from(coefficient);
                if c64 >= 0.0 {
                    lo += c64 * y_lo[c];
                    hi += c64 * y_hi[c];
                } else {
                    lo += c64 * y_hi[c];
                    hi += c64 * y_lo[c];
                }
            }
            obs_lower.push(lo);
            obs_upper.push(hi);
        }
        if let Some(separation) = first_band_separation(
            &cert_lower,
            &cert_upper,
            &obs_lower,
            &obs_upper,
            DOMAIN_FENCE_REL_BAND,
            sample,
        ) {
            panic!(
                "[vit-containment] {test}: the frozen-softmax margin publication does NOT \
                 contain the network: {} — a published margin above the network's own value is \
                 a false-proof path. SOUNDNESS EMERGENCY.",
                separation.describe()
            );
        }
        for (slot, &value) in min_observed.iter_mut().zip(&obs_lower) {
            *slot = slot.min(value);
        }
    }
    let worst_gap = cert_lower
        .iter()
        .zip(&min_observed)
        .map(|(cert, observed)| observed - cert)
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "[vit-containment] test={test} PASS rows={} target={target} samples={} committed={} \
         spec_slots={} matmul_alpha_nodes={} decomposed={} min_published_lower={:e} \
         min_gap_to_sampled_min={:e} secs={:.2}",
        rows.len(),
        walks.len(),
        publication.committed_iteration(),
        publication.spec_slots(),
        publication.matmul_alpha_nodes(),
        report.decomposed.len(),
        cert_lower.iter().copied().fold(f64::INFINITY, f64::min),
        worst_gap,
        started.elapsed().as_secs_f64()
    );
}

/// #vit-zono-root (VIBECHECK I1): the certified forward-zonotope ROOT map on
/// the REAL vit graph contains the network, plain AND under the fixed-index
/// softmax rewrite (ground truth: the f64 fence walk of the UNREWRITTEN
/// graph), and so does the lane's own root bootstrap with the key armed (the
/// map seeding the typed `fix_interm_bounds: false` CROWN-IBP route and merged
/// shrink-only into what the root CROWN / alpha pass reads).
///
/// The map must be ADMITTED on the real model (a decline makes the whole lane
/// VOID on vit, so it fails here, loudly, with its reason). A gate must
/// require `[vit-containment] test=vit_zono_root/plain PASS`,
/// `.../fixed_index PASS`, `.../bootstrap_plain PASS` and
/// `.../bootstrap_fixed_index PASS`; a SKIP is VOID.
#[test]
fn vit_zono_root_real_onnx_boxes_contain_samples() {
    let test = "vit_zono_root";
    let Some(original) = load_fixture(test) else {
        return;
    };
    let walks = sample_walks(&original);
    let output_name = original.graph.output_name().to_string();
    let armed = BetaCrownConfig {
        vit_zono_root: true,
        vit_zono_root_max_secs: 1_500.0,
        ..BetaCrownConfig::default()
    };
    let plan = ny_propagate::zono_core::vit_zono_root_plan(&armed).expect("armed plan");

    for (arm, rewrite) in [("plain", false), ("fixed_index", true)] {
        let Some(mut fixture) = load_fixture(test) else {
            return;
        };
        if rewrite {
            let report = fixture
                .graph
                .decompose_softmax_complex_fixed_index(&fixture.input)
                .expect("the fixed-index rewrite runs on the real vit box");
            assert!(
                !report.decomposed.is_empty(),
                "the vit softmaxes must take the fixed-index rewrite; skipped: {:?}",
                report.skipped
            );
        }
        let (concats, consumers) = cls_concat_and_consumers(&fixture.graph);

        // (1) The map itself.
        let started = Instant::now();
        let map = ny_propagate::zono_core::compute_vit_zono_root_map(
            &fixture.graph,
            &fixture.input,
            &plan,
            Some(Instant::now() + Duration::from_mins(25)),
        )
        .unwrap_or_else(|decline| {
            panic!(
                "[vit-containment] test={test}/{arm}: the root map DECLINED on the real vit \
                 model ({}; census {}) — the lane is VOID on vit",
                decline.reason, decline.census
            )
        });
        let label = format!("{test}/{arm}");
        let sweep = assert_contained(&label, &map.bounds, &walks);
        for name in concats.iter().chain(consumers.iter()) {
            assert!(
                sweep.nodes_compared.contains(name),
                "{label}: CLS path node '{name}' was not compared"
            );
        }
        assert!(
            sweep.nodes_compared.contains(&output_name),
            "{label}: the logits were not compared"
        );
        let ibp = fixture
            .graph
            .collect_node_bounds(&fixture.input)
            .expect("per-node IBP on the real vit box");
        let width = |bounds: &BoundedTensor| -> f64 {
            bounds
                .lower()
                .iter()
                .zip(bounds.upper().iter())
                .map(|(l, u)| f64::from(*u) - f64::from(*l))
                .sum()
        };
        let zono_out = map.bounds.get(&output_name).map_or(f64::NAN, &width);
        let ibp_out = ibp.get(&output_name).map_or(f64::NAN, width);
        eprintln!(
            "[vit-containment] test={label} PASS samples={} nodes_compared={} elements={} \
             skipped_layout={} min_slack={:e} census={} symbols={} fresh_bilinear={} \
             fresh_softmax={} fresh_band={} tightened_elems={} self_check_elems={} \
             output_width_zono={zono_out:e} output_width_ibp={ibp_out:e} secs={:.2}",
            walks.len(),
            sweep.nodes_compared.len(),
            sweep.elements,
            sweep.skipped_layout,
            sweep.min_slack,
            map.census,
            map.stats.symbols,
            map.stats.fresh_bilinear,
            map.stats.fresh_softmax,
            map.stats.fresh_band,
            map.tightened_elems,
            map.self_check_elems,
            started.elapsed().as_secs_f64()
        );

        // (2) The lane's own root bootstrap with the key armed: the typed
        // CROWN-IBP reference route seeded by the map, then the alpha pass.
        let base = BetaCrownConfig {
            vit_zono_root: true,
            vit_zono_root_max_secs: 1_500.0,
            ..BetaCrownConfig::default()
        };
        let config = BetaCrownConfig {
            use_alpha_crown: true,
            alpha_config: AlphaCrownConfig {
                iterations: 1,
                fix_interm_bounds: false,
                crown_ibp_reference_cap_secs: Some(15),
                ..base.alpha_config.clone()
            },
            ..base
        };
        let verifier = BetaCrownVerifier::new(config);
        let started = Instant::now();
        let (node_bounds, output) = verifier
            .compute_initial_graph_bounds(
                &fixture.graph,
                &fixture.input,
                Some(Instant::now() + Duration::from_mins(30)),
            )
            .expect("the armed root bootstrap runs on the real vit box");
        let label = format!("{test}/bootstrap_{arm}");
        let sweep = assert_contained(&label, &node_bounds, &walks);
        assert!(
            !sweep.nodes_compared.is_empty(),
            "{label}: no bootstrap node was comparable — the sweep checked nothing"
        );
        let mut outputs = HashMap::new();
        outputs.insert(output_name.clone(), output);
        let output_sweep = assert_contained(&format!("{label}/output"), &outputs, &walks);
        assert_eq!(
            output_sweep.nodes_compared.len(),
            1,
            "{label}: the root output enclosure must be compared"
        );
        // The lane's own observable: the armed bootstrap must have COMPUTED an
        // ADMITTED root map for this graph identity and box, which leaves it
        // in the exact-box cache (the verifier's configured clone keeps the
        // graph's semantic scope, and `base` carries the same two ceilings as
        // `plan`). Without this, a bootstrap that never reached the reference
        // hook, or whose map declined, would still print PASS above:
        // containment of the dark route, i.e. VOID for this lane.
        let (served, hit) = ny_propagate::zono_core::vit_zono_root_map(
            &fixture.graph,
            &fixture.input,
            &plan,
            Some(Instant::now() + Duration::from_mins(25)),
        );
        assert!(
            hit && served.is_some(),
            "{label}: the armed bootstrap left no admitted root map for this graph and box \
             (cache_hit={hit}, admitted={}): the lane never engaged or declined (read its \
             `[vit-zono-root] armed=1 admitted=` line); the bootstrap PASS would be VOID",
            served.is_some()
        );
        eprintln!(
            "[vit-containment] test={label} PASS samples={} nodes_compared={} elements={} \
             skipped_layout={} min_slack={:e} output_min_slack={:e} bootstrap_secs={:.2}",
            walks.len(),
            sweep.nodes_compared.len(),
            sweep.elements,
            sweep.skipped_layout,
            sweep.min_slack,
            output_sweep.min_slack,
            started.elapsed().as_secs_f64()
        );
    }
}
