// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #iso-edge-cap step 5 (WAVE6_PLAN §7.C6) — THE LOWDIM BENCH on the stitched
//! isomorphic difference network. This number decides phase 2
//! (`feat/iso-relational-lowdim`: wiring the `#lowdim-prover` lane into the
//! relational route before its BaB).
//!
//! TEST-ONLY. No production path reads anything here. The route mirrors what
//! phase 2 would wire: load both towers of a row (`f`, `g`), stitch them with
//! `build_difference_network` (`h = f - g`, one shared input), admit the
//! stitched graph to the lane over the row's input box, then prove each of
//! the 10 band clauses (`h_i >= -eps` and `-h_i >= -eps` for the 5 outputs)
//! with `prove_conjunction_admitted`.
//!
//! Registered ONLY with `--features external-vnncomp,benchmarks`: it reads the
//! competition corpus and measures wall-clock, so it is absent from the
//! default lane and a missing corpus is a HARD failure. The two timing
//! targets refuse a debug build (VOID); the two soundness targets run in any
//! profile. Run it on a quiet box:
//!
//! ```text
//! NY_BENCH_ROOT_2026=<home>/ny/benchmarks/vnncomp2026/benchmarks \
//!   rustup run trust targo --unverified test --locked --offline --release \
//!   -p ny-onnx --features external-vnncomp,benchmarks --test iso_lane_bench \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! Lines (one per clause, one per row, one gate verdict):
//!
//! ```text
//! [lowdim-bench] config=r50-sbwr row=4 clause=3 domains=N secs=S rate=R threads=T
//!   verdict=verified|unproved stop=.. closed=.. clip_closed=.. clip_shrunk=.. depth=..
//! [lowdim-bench] row-summary config=r50-sbwr row=4 clauses_closed=K/10 domains=N secs=S rate=R
//! [lowdim-bench] GATE config=r50-sbwr rate_min=R rows_all_closed=K/4 need_rate=10000
//!   need_rows=2 pass=0|1
//! ```
//!
//! THE PHASE-2 GATE (WAVE6_PLAN §7.C6): at least 10 k domains/s at refine
//! width 50, AND all 10 clauses closed within a 60 s total on at least 2 of
//! the 4 gap rows (`i4`, `i12`, `i22`, `i26`). The gate is PRINTED, never
//! asserted: it is a decision number, not a correctness property.
//!
//! SOUNDNESS (asserted): on the four rows with ORT-validated violations
//! (`i14`, `i21`, `i33`, `i48`, banked `sat`) some band atom is genuinely
//! feasible, so the lane must fail to prove at least one clause on each.
//! Proving all 10 on any of them is a SOUNDNESS FAILURE. And the lane's
//! rigorous per-box lower bounds must contain every sampled output, both of
//! its own lowered program and of the stitched f32 graph.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::{
    load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig,
    ShapeInferBackend,
};
use ny_propagate::lowdim::{
    admit, bound_boxes, lane_threads, lowered_point_values, prove_conjunction_admitted,
    LaneAdmission, LowDimConfig, LowDimOutcome, LowDimParallel, LowDimSplit,
};
use ny_propagate::{build_difference_network, GraphNetwork};
use ny_tensor::BoundedTensor;

mod common;
use common::{benchmark_root, BenchYear};

/// The four gap rows the plan names (truth `unsat`, ny banks `timeout`).
const GAP_ROWS: [usize; 4] = [4, 12, 22, 26];
/// The four rows with ORT-validated violations (banked `sat`).
const SAT_ROWS: [usize; 4] = [14, 21, 33, 48];
/// Wall seconds per (row, configuration) in the throughput arms.
const ARM_ROW_SECS: f64 = 20.0;
/// Wall seconds per row for the phase-2 gate.
const GATE_ROW_SECS: f64 = 60.0;
/// The gate's throughput floor, domains per second at refine width 50.
const GATE_RATE: f64 = 10_000.0;
/// The gate's closure floor: rows (of 4) on which all 10 clauses close.
const GATE_ROWS: usize = 2;

fn category_dir() -> PathBuf {
    benchmark_root(BenchYear::Vnncomp2026).join("isomorphic_acasxu_2026/2.0")
}

fn load_config() -> OnnxLoadConfig {
    OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::InProcess)
}

fn graph_options() -> GraphNetworkOptions {
    GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    }
}

/// One row of `instances.csv`: `"[('f', '<onnx>'), ('g', '<onnx>')]",./vnnlib/instance_N.vnnlib,<secs>`.
struct Instance {
    f: PathBuf,
    g: PathBuf,
    vnnlib: PathBuf,
}

/// Quote-aware read of `instances.csv` for row `row` (the file has no header;
/// the row is matched by its vnnlib FILE NAME, never by line position).
fn instance(dir: &Path, row: usize) -> Instance {
    let csv_path = dir.join("instances.csv");
    let csv = std::fs::read_to_string(&csv_path)
        .unwrap_or_else(|e| panic!("{}: {e}", csv_path.display()));
    let want = format!("instance_{row}.vnnlib");
    for line in csv.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (networks, rest) = if let Some(stripped) = line.strip_prefix('"') {
            let end = stripped
                .find('"')
                .unwrap_or_else(|| panic!("unterminated quoted field: {line}"));
            (
                &stripped[..end],
                stripped[end + 1..].trim_start_matches(','),
            )
        } else {
            let comma = line.find(',').expect("csv field");
            (&line[..comma], &line[comma + 1..])
        };
        let vnnlib_rel = rest.split(',').next().expect("vnnlib field").trim();
        if Path::new(vnnlib_rel).file_name().and_then(|n| n.to_str()) != Some(want.as_str()) {
            continue;
        }
        // The single-quoted strings, in order: f, <path>, g, <path>.
        let quoted: Vec<&str> = networks.split('\'').skip(1).step_by(2).collect();
        let path_of = |name: &str| -> PathBuf {
            let at = quoted
                .iter()
                .position(|q| *q == name)
                .unwrap_or_else(|| panic!("row {row}: network `{name}` missing in {networks}"));
            dir.join(quoted.get(at + 1).expect("network path"))
        };
        return Instance {
            f: path_of("f"),
            g: path_of("g"),
            vnnlib: dir.join(vnnlib_rel),
        };
    }
    panic!("row {row}: {want} is not in {}", csv_path.display());
}

fn load_graph(path: &Path) -> GraphNetwork {
    let model = load_onnx_with_config(path, &load_config())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    model
        .to_graph_network_with_options(graph_options())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The smallest f32 at or above `t`.
fn f32_up(t: f64) -> f32 {
    let f = t as f32;
    if f64::from(f) < t {
        f.next_up()
    } else {
        f
    }
}

/// The largest f32 at or below `t`.
fn f32_down(t: f64) -> f32 {
    let f = t as f32;
    if f64::from(f) > t {
        f.next_down()
    } else {
        f
    }
}

/// One row, stitched and admitted.
struct Row {
    row: usize,
    diff: GraphNetwork,
    adm: LaneAdmission,
    /// The admitted input shape (the stitched graph accepted `[d]` or the
    /// declared `[1, 1, 1, d]`).
    shape: Vec<usize>,
    /// The admitted root box (the declared box rounded outward to f32), as
    /// f64 — exactly what the lane holds, so `bound_boxes` accepts sub-boxes.
    root_lo: Vec<f64>,
    root_hi: Vec<f64>,
    /// The 10 band clauses as `(objective over h, threshold)`: a proof of
    /// `objective . h > threshold` refutes the conjunct `objective . h <= threshold`,
    /// and `threshold >= -eps`, so it refutes the band atom.
    clauses: Vec<(Vec<f32>, f32)>,
}

fn stitched_row(row: usize) -> Row {
    let dir = category_dir();
    let inst = instance(&dir, row);
    let spec =
        ny_onnx::vnnlib::load_vnnlib(&inst.vnnlib).unwrap_or_else(|e| panic!("row {row}: {e}"));
    let dual = spec
        .dual_network
        .as_ref()
        .unwrap_or_else(|| panic!("row {row}: not a dual-network vnnlib"));
    let eps = dual
        .validation
        .isomorphic_output_atoms
        .iter()
        .map(|atom| atom.constant.abs())
        .fold(f64::INFINITY, f64::min);
    assert!(
        eps.is_finite() && eps > 0.0,
        "row {row}: no isomorphic band atoms (eps={eps})"
    );
    let f = load_graph(&inst.f);
    let g = load_graph(&inst.g);
    let diff = build_difference_network(&f, &g)
        .unwrap_or_else(|e| panic!("row {row}: difference network: {e}"));
    // Round the box OUTWARD: the lane then proves over a superset of the
    // declared box, which is the safe direction for a measurement.
    let lo: Vec<f32> = dual
        .f_input_bounds
        .iter()
        .map(|(l, _)| f32_down(*l))
        .collect();
    let hi: Vec<f32> = dual
        .f_input_bounds
        .iter()
        .map(|(_, h)| f32_up(*h))
        .collect();
    let d = lo.len();
    let mut reasons = Vec::new();
    for shape in [vec![d], vec![1, 1, 1, d]] {
        let root = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), lo.clone()).expect("shape"),
            ArrayD::from_shape_vec(IxDyn(&shape), hi.clone()).expect("shape"),
        )
        .expect("box");
        match admit(&diff, &root) {
            Ok(adm) => {
                let width = adm.output_width();
                let threshold = f32_up(-eps);
                let mut clauses = Vec::with_capacity(2 * width);
                for i in 0..width {
                    for sign in [1.0_f32, -1.0_f32] {
                        let mut objective = vec![0.0_f32; width];
                        objective[i] = sign;
                        clauses.push((objective, threshold));
                    }
                }
                eprintln!(
                    "[lowdim-bench] admitted row={row} shape={shape:?} d={} width={width} \
                     eps={eps} clauses={} f={} g={}",
                    adm.input_dim(),
                    clauses.len(),
                    inst.f.display(),
                    inst.g.display()
                );
                return Row {
                    row,
                    diff,
                    adm,
                    shape,
                    root_lo: lo.iter().map(|&v| f64::from(v)).collect(),
                    root_hi: hi.iter().map(|&v| f64::from(v)).collect(),
                    clauses,
                };
            }
            Err(reason) => reasons.push(format!("{shape:?}: {reason}")),
        }
    }
    panic!("row {row}: the lane DECLINED the stitched net: {reasons:?}");
}

/// The lane stack under test: clip on, parallel full, split and refine width
/// as named.
fn lane(refine: usize, split: LowDimSplit) -> LowDimConfig {
    LowDimConfig {
        refine_max_width: refine,
        split,
        clip: true,
        parallel: LowDimParallel::Full,
        ..LowDimConfig::default()
    }
}

fn arms() -> Vec<(&'static str, LowDimConfig)> {
    vec![
        ("r0-widest", lane(0, LowDimSplit::Widest)),
        ("r0-sb", lane(0, LowDimSplit::SbSum)),
        ("r0-sbwr", lane(0, LowDimSplit::SbWr)),
        ("r50-widest", lane(50, LowDimSplit::Widest)),
        ("r50-sb", lane(50, LowDimSplit::SbSum)),
        ("r50-sbwr", lane(50, LowDimSplit::SbWr)),
    ]
}

/// What one (row, configuration) run did.
#[derive(Debug, Default, Clone, Copy)]
struct RowRun {
    clauses_closed: usize,
    clauses_run: usize,
    domains: u64,
    secs: f64,
    clip_engaged: u64,
}

impl RowRun {
    fn rate(self) -> f64 {
        self.domains as f64 / self.secs.max(1e-9)
    }
}

/// Run every band clause of `row` in order under ONE shared deadline of
/// `row_secs`; a clause reached after the deadline is not run (and is not
/// closed).
fn run_row(name: &str, row: &Row, cfg: &LowDimConfig, row_secs: f64) -> RowRun {
    let deadline = Instant::now() + Duration::from_secs_f64(row_secs);
    let threads = lane_threads(cfg);
    let mut run = RowRun::default();
    for (index, (objective, threshold)) in row.clauses.iter().enumerate() {
        if Instant::now() >= deadline {
            break;
        }
        let mut witness = None;
        let out = prove_conjunction_admitted(
            &row.adm,
            std::slice::from_ref(objective),
            std::slice::from_ref(threshold),
            deadline,
            cfg,
            &mut witness,
        );
        let (stats, verdict, stop) = match &out {
            LowDimOutcome::Verified(s) => (*s, "verified", "-".to_string()),
            LowDimOutcome::Unproved(s, stop) => (*s, "unproved", format!("{stop:?}")),
            LowDimOutcome::Declined(reason) => {
                panic!(
                    "{name}: row {} clause {index}: the lane DECLINED: {reason}",
                    row.row
                )
            }
        };
        run.clauses_run += 1;
        run.domains += stats.domains;
        run.secs += stats.secs;
        run.clip_engaged += stats.clip_closed + stats.clip_shrunk;
        if matches!(out, LowDimOutcome::Verified(_)) {
            run.clauses_closed += 1;
        }
        let rate = stats.domains as f64 / stats.secs.max(1e-9);
        eprintln!(
            "[lowdim-bench] config={name} row={} clause={index} domains={} secs={:.3} \
             rate={rate:.0} threads={threads} verdict={verdict} stop={stop} closed={} \
             clip_closed={} clip_shrunk={} depth={} split={} refine_width={} clip={} parallel={}",
            row.row,
            stats.domains,
            stats.secs,
            stats.closed,
            stats.clip_closed,
            stats.clip_shrunk,
            stats.max_depth,
            cfg.split.token(),
            cfg.refine_max_width,
            u8::from(cfg.clip),
            cfg.parallel.token(),
        );
        assert!(
            stats.domains > 0,
            "{name}: row {} clause {index}: no domain bounded: VOID",
            row.row
        );
    }
    eprintln!(
        "[lowdim-bench] row-summary config={name} row={} clauses_closed={}/{} clauses_run={} \
         domains={} secs={:.3} rate={:.0}",
        row.row,
        run.clauses_closed,
        row.clauses.len(),
        run.clauses_run,
        run.domains,
        run.secs,
        run.rate()
    );
    run
}

/// THROUGHPUT ARMS: refine width {0, 50} x split {widest, sb, sbwr}, clip on,
/// parallel full, 20 s per row, on the four gap rows.
#[test]
#[expect(
    clippy::assertions_on_constants,
    reason = "Reject debug execution at runtime while permitting debug typechecking"
)]
fn iso_lane_throughput_arms() {
    assert!(
        !cfg!(debug_assertions),
        "VOID: the iso lane throughput bench must run in --release"
    );
    let rows: Vec<Row> = GAP_ROWS.iter().map(|&r| stitched_row(r)).collect();
    let arms = arms();
    for (name, cfg) in &arms {
        let mut clip_engaged = 0u64;
        let mut domains = 0u64;
        let mut secs = 0.0_f64;
        for row in &rows {
            let run = run_row(name, row, cfg, ARM_ROW_SECS);
            clip_engaged += run.clip_engaged;
            domains += run.domains;
            secs += run.secs;
        }
        eprintln!(
            "[lowdim-bench] arm-summary config={name} rows={} domains={domains} secs={secs:.3} \
             rate={:.0}",
            rows.len(),
            domains as f64 / secs.max(1e-9)
        );
        assert!(
            clip_engaged > 0,
            "{name}: the clip is armed but never shrank or closed a box on any row: VOID"
        );
    }
    eprintln!("[lowdim-bench] DONE arms={}", arms.len());
}

/// THE PHASE-2 GATE, at refine width 50 with the `sbwr` split: printed, not
/// asserted.
#[test]
#[expect(
    clippy::assertions_on_constants,
    reason = "Reject debug execution at runtime while permitting debug typechecking"
)]
fn iso_lane_phase2_gate() {
    assert!(
        !cfg!(debug_assertions),
        "VOID: the phase-2 gate is a wall-clock number and must run in --release"
    );
    let name = "r50-sbwr";
    let cfg = lane(50, LowDimSplit::SbWr);
    let mut rate_min = f64::INFINITY;
    let mut rows_all_closed = 0usize;
    for &r in &GAP_ROWS {
        let row = stitched_row(r);
        let run = run_row(name, &row, &cfg, GATE_ROW_SECS);
        rate_min = rate_min.min(run.rate());
        if run.clauses_closed == row.clauses.len() {
            rows_all_closed += 1;
        }
    }
    let pass = rate_min >= GATE_RATE && rows_all_closed >= GATE_ROWS;
    eprintln!(
        "[lowdim-bench] GATE config={name} rate_min={rate_min:.0} rows_all_closed={rows_all_closed}/{} \
         need_rate={GATE_RATE:.0} need_rows={GATE_ROWS} pass={}",
        GAP_ROWS.len(),
        u8::from(pass)
    );
}

/// NEGATIVE CONTROL (soundness): on every row with a validated violation the
/// lane must fail to prove at least one band clause, under the gate's stack
/// and under the plain one.
#[test]
fn iso_lane_negative_control_sat_rows() {
    for (name, cfg) in [
        ("r50-sbwr", lane(50, LowDimSplit::SbWr)),
        ("r0-widest", lane(0, LowDimSplit::Widest)),
    ] {
        for &r in &SAT_ROWS {
            let row = stitched_row(r);
            let run = run_row(name, &row, &cfg, ARM_ROW_SECS);
            assert!(
                run.clauses_closed < row.clauses.len(),
                "SOUNDNESS FAILURE: {name} proved all {} band clauses on row {r}, which has an \
                 ORT-validated violation",
                row.clauses.len()
            );
        }
    }
}

/// A tiny deterministic generator (no external RNG crate needed).
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// SAMPLED CONTAINMENT (soundness): the lane's rigorous lower bound of every
/// band objective, on the root box and on random sub-boxes, sits at or below
/// the objective's value at every sampled point of that box — for the lane's
/// own lowered program (f64) and for the stitched f32 graph.
#[test]
fn iso_lane_bound_containment_sampled() {
    for r in [4usize, 14] {
        let row = stitched_row(r);
        let d = row.adm.input_dim();
        let (root_lo, root_hi) = (row.root_lo.clone(), row.root_hi.clone());
        assert_eq!(
            root_lo.len(),
            d,
            "row {r}: box and program dimensions differ"
        );
        let objectives: Vec<Vec<f64>> = row
            .clauses
            .iter()
            .map(|(o, _)| o.iter().map(|&v| f64::from(v)).collect())
            .collect();
        let mut rng = Lcg(0x5eed_0000_0000_0000 ^ r as u64);
        for refine in [0usize, 50] {
            let cfg = lane(refine, LowDimSplit::Widest);
            // The root box plus 16 random sub-boxes.
            let mut boxes: Vec<(Vec<f64>, Vec<f64>)> = vec![(root_lo.clone(), root_hi.clone())];
            for _ in 0..16 {
                let mut lo = Vec::with_capacity(d);
                let mut hi = Vec::with_capacity(d);
                for k in 0..d {
                    let w = root_hi[k] - root_lo[k];
                    // Clamped: `lo + w * u` may round an ulp past the root.
                    let a = (root_lo[k] + w * rng.unit()).clamp(root_lo[k], root_hi[k]);
                    let b = (root_lo[k] + w * rng.unit()).clamp(root_lo[k], root_hi[k]);
                    lo.push(a.min(b));
                    hi.push(a.max(b));
                }
                boxes.push((lo, hi));
            }
            let flat_lo: Vec<f64> = boxes.iter().flat_map(|(l, _)| l.clone()).collect();
            let flat_hi: Vec<f64> = boxes.iter().flat_map(|(_, h)| h.clone()).collect();
            let lower = bound_boxes(&row.adm, &flat_lo, &flat_hi, &objectives, &cfg)
                .unwrap_or_else(|e| panic!("row {r}: bound_boxes: {e}"));
            assert_eq!(lower.len(), boxes.len());
            let mut checked = 0usize;
            for ((lo, hi), lbs) in boxes.iter().zip(&lower) {
                for _ in 0..64 {
                    let x: Vec<f64> = (0..d)
                        .map(|k| (lo[k] + (hi[k] - lo[k]) * rng.unit()).clamp(lo[k], hi[k]))
                        .collect();
                    let program = lowered_point_values(&row.adm, &x)
                        .unwrap_or_else(|e| panic!("row {r}: lowered point: {e}"));
                    let graph = graph_point_values(&row, &x);
                    for (j, objective) in objectives.iter().enumerate() {
                        let value: f64 = objective.iter().zip(&program).map(|(a, b)| a * b).sum();
                        assert!(
                            lbs[j] <= value + 1e-9 * (1.0 + value.abs()),
                            "SOUNDNESS FAILURE: row {r} refine {refine} objective {j}: lane lower \
                             {} above the lowered program's value {value} at {x:?}",
                            lbs[j]
                        );
                        let value32: f64 = objective
                            .iter()
                            .zip(&graph)
                            .map(|(a, b)| a * f64::from(*b))
                            .sum();
                        assert!(
                            lbs[j] <= value32 + 1e-3,
                            "SOUNDNESS FAILURE: row {r} refine {refine} objective {j}: lane lower \
                             {} above the stitched f32 graph's value {value32} at {x:?}",
                            lbs[j]
                        );
                        checked += 1;
                    }
                }
            }
            eprintln!(
                "[lowdim-bench] containment row={r} refine_width={refine} boxes={} checks={checked}",
                boxes.len()
            );
            assert!(checked > 0, "no containment check ran: VOID");
        }
    }
}

/// The stitched graph's f32 output at `x` (an f32 forward; fidelity, not
/// rigor — hence the looser tolerance above).
fn graph_point_values(row: &Row, x: &[f64]) -> Vec<f32> {
    let point: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let array = ArrayD::from_shape_vec(IxDyn(&row.shape), point).expect("point shape");
    let tensor = BoundedTensor::new(array.clone(), array).expect("point");
    let out = row
        .diff
        .propagate_concrete_point(&tensor, None, None)
        .unwrap_or_else(|e| panic!("row {}: forward: {e}", row.row));
    out.lower().iter().copied().collect()
}
