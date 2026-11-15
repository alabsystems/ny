// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane stage 3 — THE THROUGHPUT MICRO-BENCHMARK. This number decides
//! the lsnc build.
//!
//! The numpy decision gate (`scratchpad/lsncproto`, 2026-09-24) needs
//! 2.0e5-1.4e6 domains per `lsnc_relu` unsat row with the full `allmax` stack
//! (row 23: 1.96e5, row 79: 2.68e5, row 2: 8.93e5, row 10: 1.40e6), and a row
//! has ~10-14 s of lane time inside its official 25 s. So the lane must
//! sustain roughly 2e4 (low-band rows) to 1.5e5 (high-band rows) domains/s on
//! this net. This target measures it, factor by factor, on row 2's clause 0
//! (the `dV >= 1e-6` clause inside row 2's `V` band — the clause that carries
//! almost the whole tree), 2026 `1.0` corpus only.
//!
//! Registered ONLY with `--features external-vnncomp,benchmarks`: it is a
//! wall-clock measurement on the competition corpus, so it is absent from the
//! default lane (TEST_CONFORMANCE.md forbids `ignore`) and a missing corpus is
//! a HARD failure. It refuses a debug build (VOID). Run it on a quiet box:
//!
//! ```text
//! NY_BENCH_ROOT_2026=<home>/ny/benchmarks/vnncomp2026/benchmarks \
//!   rustup run trust targo --unverified test --locked --offline --release \
//!   -p ny-onnx --features external-vnncomp,benchmarks --test lsnc_lane_bench \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! One line per configuration, each run for [`BENCH_SECS`]:
//!
//! ```text
//! [lowdim-bench] config=allmax row=2 clause=0 domains=N secs=S rate=R threads=T
//!   us_per_domain_thread=U verdict=unproved stop=deadline closed=.. clip_closed=..
//!   clip_shrunk=.. joint_hits=.. depth=.. split=sbwr clip=1 joint=1 parallel=full
//! ```
//!
//! then one `[lowdim-bench] projection ...` line per gate row (the seconds the
//! `allmax` rate needs for the gate's domain estimate — an estimate that
//! INCLUDED the prototype's child clip with its parent's planes, which this
//! lane deliberately omits, so the lane's own counts may be higher), and a
//! final `[lowdim-bench] DONE configs=K`. A configuration whose armed lever
//! left no trace (`joint_hits=0` with the joint armed, `clip_closed=0
//! clip_shrunk=0` with the clip armed) fails the target: VOID, not slow.

use std::path::Path;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::{
    load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig,
    ShapeInferBackend,
};
use ny_propagate::lowdim::{
    admit, lane_threads, prove_conjunction_admitted, LaneAdmission, LowDimConfig, LowDimOutcome,
    LowDimParallel, LowDimSplit,
};
use ny_tensor::BoundedTensor;

mod common;
use common::{benchmark_root, BenchYear};

/// Wall seconds per configuration.
const BENCH_SECS: f64 = 10.0;

/// The numpy gate's `allmax` domain estimates, 2026 `1.0` corpus
/// (`scratchpad/lsncproto/runs_gate_band/gate.log`), as `(row, domains)`.
const GATE_ALLMAX: [(usize, f64); 4] = [(23, 1.963e5), (79, 2.682e5), (2, 8.928e5), (10, 1.395e6)];

/// The official input box of every `lsnc_relu` row.
const BOX_LO: [f32; 6] = [-0.7, -0.7, -1.0, -3.0, -3.0, -2.0];
const BOX_HI: [f32; 6] = [0.7, 0.7, 1.0, 3.0, 3.0, 2.0];

fn load_config() -> OnnxLoadConfig {
    OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::InProcess)
}

fn graph_options() -> GraphNetworkOptions {
    GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    }
}

/// The real graph with its lane twin attached, admitted over the shared box
/// (the same route `ny-cli` takes: load, build the twin, attach, admit).
fn admitted(dir: &Path) -> LaneAdmission {
    let path = dir.join("onnx/relu_quadrotor2d_state.onnx");
    let model = load_onnx_with_config(&path, &load_config()).expect("the lsnc model loads");
    let mut graph = model
        .to_graph_network_with_options(graph_options())
        .expect("the lsnc model lowers to a graph");
    let twin = ny_onnx::lowdim_twin::build_lane_twin(&path, &load_config(), graph_options())
        .unwrap_or_else(|reason| panic!("the lsnc lane twin must build: {reason}"));
    graph.attach_lowdim_twin(twin);
    let lo = ArrayD::from_shape_vec(IxDyn(&[1, 6]), BOX_LO.to_vec()).expect("shape");
    let hi = ArrayD::from_shape_vec(IxDyn(&[1, 6]), BOX_HI.to_vec()).expect("shape");
    let root = BoundedTensor::new(lo, hi).expect("box");
    admit(&graph, &root).unwrap_or_else(|reason| panic!("the twin must be admitted: {reason}"))
}

/// The number after `prefix` up to the next `)`.
fn number_after(text: &str, prefix: &str, what: &str) -> f64 {
    let at = text
        .find(prefix)
        .unwrap_or_else(|| panic!("{what}: `{prefix}` not in the vnnlib"))
        + prefix.len();
    let rest = &text[at..];
    let end = rest
        .find(')')
        .unwrap_or_else(|| panic!("{what}: unterminated"));
    rest[..end]
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{what}: {e}"))
}

/// The smallest f32 at or above `t` (a proof of `o . y > t32 >= t` refutes
/// the conjunct `o . y <= t`).
fn f32_up(t: f64) -> f32 {
    let f = t as f32;
    if f64::from(f) < t {
        f.next_up()
    } else {
        f
    }
}

/// Clause 0 of a row as lane conjuncts (a counterexample satisfies every
/// `objective . y <= threshold`): `V = Y_1` in the row's band (the top-level
/// `Y_1` asserts, which the spec parser distributes into every disjunct) and
/// the first disjunct `Y_0 >= thr`. Read from the row's own file.
fn clause0(dir: &Path, row: usize) -> (Vec<Vec<f32>>, Vec<f32>, (f64, f64)) {
    let path = dir.join(format!("vnnlib/quadrotor2d_state_{row}.vnnlib"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("row {row}: {}: {e}", path.display()));
    let band_hi = number_after(&text, "(assert (<= Y_1 ", "band upper");
    let band_lo = number_after(&text, "(assert (>= Y_1 ", "band lower");
    let dv_thr = number_after(&text, "(and (>= Y_0 ", "clause 0");
    let e = |j: usize, s: f32| {
        let mut o = vec![0.0f32; 8];
        o[j] = s;
        o
    };
    (
        vec![e(1, -1.0), e(1, 1.0), e(0, -1.0)],
        vec![f32_up(-band_lo), f32_up(band_hi), f32_up(-dv_thr)],
        (band_lo, band_hi),
    )
}

/// The `allmax` stack in a given wave mode.
fn allmax(parallel: LowDimParallel) -> LowDimConfig {
    LowDimConfig {
        joint_quadratic: true,
        refine_max_width: 8,
        clip: true,
        split: LowDimSplit::SbWr,
        parallel,
        ..LowDimConfig::default()
    }
}

#[test]
#[expect(
    clippy::assertions_on_constants,
    reason = "Reject debug execution at runtime while permitting debug typechecking"
)]
fn lsnc_lane_throughput_row2_clause0() {
    assert!(
        !cfg!(debug_assertions),
        "VOID: the lsnc throughput bench must run in --release (a debug build's rate is \
         meaningless)"
    );
    let dir = benchmark_root(BenchYear::Vnncomp2026).join("lsnc_relu/1.0");
    let adm = admitted(&dir);
    assert_eq!(
        adm.joint_sites(),
        2,
        "V and dV must be the lane's two joint sites"
    );
    let refine = LowDimConfig {
        refine_max_width: 8,
        parallel: LowDimParallel::Full,
        ..LowDimConfig::default()
    };
    let configs: Vec<(&str, usize, LowDimConfig)> = vec![
        ("refine8", 2, refine.clone()),
        (
            "joint",
            2,
            LowDimConfig {
                joint_quadratic: true,
                ..refine.clone()
            },
        ),
        (
            "joint+clip",
            2,
            LowDimConfig {
                joint_quadratic: true,
                clip: true,
                ..refine
            },
        ),
        ("allmax", 2, allmax(LowDimParallel::Full)),
        (
            "allmax-sb",
            2,
            LowDimConfig {
                split: LowDimSplit::SbSum,
                ..allmax(LowDimParallel::Full)
            },
        ),
        ("allmax-bound", 2, allmax(LowDimParallel::Bound)),
        ("allmax-serial", 2, allmax(LowDimParallel::Serial)),
        ("allmax-row23", 23, allmax(LowDimParallel::Full)),
    ];
    let mut allmax_rate = None;
    for (name, row, cfg) in &configs {
        let (objectives, thresholds, band) = clause0(&dir, *row);
        let mut witness = None;
        let out = prove_conjunction_admitted(
            &adm,
            &objectives,
            &thresholds,
            Instant::now() + Duration::from_secs_f64(BENCH_SECS),
            cfg,
            &mut witness,
        );
        let (stats, verdict, stop) = match &out {
            LowDimOutcome::Verified(s) => (*s, "verified", "-".to_string()),
            LowDimOutcome::Unproved(s, stop) => (*s, "unproved", format!("{stop:?}")),
            LowDimOutcome::Declined(reason) => panic!("{name}: the lane DECLINED: {reason}"),
        };
        let threads = lane_threads(cfg);
        let rate = stats.domains as f64 / stats.secs.max(1e-9);
        let us = stats.secs * f64::from(threads) / (stats.domains.max(1) as f64) * 1e6;
        eprintln!(
            "[lowdim-bench] config={name} row={row} clause=0 band=[{:.6},{:.6}] domains={} \
             secs={:.3} rate={rate:.0} threads={threads} us_per_domain_thread={us:.1} \
             verdict={verdict} stop={stop} closed={} clip_closed={} clip_shrunk={} joint_hits={} \
             depth={} split={} clip={} joint={} parallel={}",
            band.0,
            band.1,
            stats.domains,
            stats.secs,
            stats.closed,
            stats.clip_closed,
            stats.clip_shrunk,
            stats.joint_hits,
            stats.max_depth,
            cfg.split.token(),
            u8::from(cfg.clip),
            u8::from(cfg.joint_quadratic),
            cfg.parallel.token()
        );
        assert!(stats.domains > 0, "{name}: no domain bounded: VOID");
        if cfg.joint_quadratic {
            assert!(
                stats.joint_hits > 0,
                "{name}: the joint is armed but never bounded a domain: VOID"
            );
        }
        if cfg.clip {
            assert!(
                stats.clip_closed + stats.clip_shrunk > 0,
                "{name}: the clip is armed but never shrank or closed a box: VOID"
            );
        }
        if *name == "allmax" {
            allmax_rate = Some(rate);
        }
    }
    let rate = allmax_rate.expect("the allmax configuration ran");
    for (row, est) in GATE_ALLMAX {
        eprintln!(
            "[lowdim-bench] projection config=allmax rate={rate:.0} row={row} \
             gate_allmax_domains={est:.3e} lane_secs_needed={:.1}",
            est / rate.max(1e-9)
        );
    }
    eprintln!("[lowdim-bench] DONE configs={}", configs.len());
}
