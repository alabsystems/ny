// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #fwd-root-screen (V4): the clause verdicts and the union on synthetic
//! boxes, and — on the REAL ml4acopf models — ORT/pointwise containment on
//! every model in both years, the true-sat stop-rule controls, the C1 union
//! and the F1 replica parity (`scratchpad/vbs/vibecheck/gates/F1`).
//!
//! The real-model tests need the ml4acopf corpus. From a worktree they
//! resolve the main checkout through the worktree's `.git` file; a missing
//! corpus or an unavailable ONNX Runtime is VOID and FAILS (never a pass by
//! skipping).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::zono_core::form_check::{check_point, summarize, ContainmentReport};
use ny_propagate::zono_core::{
    run_forward_root_screen, DagForwardOptions, DagNodeView, DagProgram, ZonoNodeView,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

use super::{
    clause_provably_unsat, constraint_refuted_f64, screen_clause_verdicts, union_refutations,
};

const BIG_BUDGET: u64 = 16 << 30;

fn tensor(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("upper"),
    )
    .expect("valid tensor")
}

// ---------------------------------------------------------------------------
// Pure verdict / union semantics (no corpus)
// ---------------------------------------------------------------------------

/// The f64 clause check mirrors the f32 precheck's semantics exactly:
/// non-strict constraints refute on a strict inequality, strict ones on a
/// non-strict one, relational constraints compare opposite ends, non-finite
/// data refutes nothing, an empty clause is never refuted, and a clause needs
/// only ONE provably false constraint.
#[test]
fn fwd_root_screen_clause_verdicts_match_box_semantics() {
    let lo = [0.5, -2.0, 1.0, f64::NEG_INFINITY];
    let hi = [0.7, -1.0, 1.0, 3.0];
    let refuted = |c: OutputConstraint| constraint_refuted_f64(&lo, &hi, &c).0;
    assert!(refuted(OutputConstraint::LessEqConst(0, 0.4)));
    assert!(!refuted(OutputConstraint::LessEqConst(0, 0.5)));
    assert!(refuted(OutputConstraint::LessThanConst(0, 0.5)));
    assert!(refuted(OutputConstraint::GreaterEqConst(1, -0.5)));
    assert!(!refuted(OutputConstraint::GreaterEqConst(1, -1.0)));
    assert!(refuted(OutputConstraint::GreaterThanConst(1, -1.0)));
    assert!(refuted(OutputConstraint::LessEq(0, 1)));
    assert!(!refuted(OutputConstraint::LessEq(1, 0)));
    assert!(refuted(OutputConstraint::GreaterEq(1, 2)));
    assert!(refuted(OutputConstraint::GreaterThan(2, 2)));
    assert!(!refuted(OutputConstraint::GreaterEq(2, 2)));
    assert!(
        !refuted(OutputConstraint::LessEqConst(3, -10.0)),
        "infinite end"
    );
    assert!(
        !refuted(OutputConstraint::LessEqConst(9, 0.0)),
        "out of range"
    );
    assert!(
        !refuted(OutputConstraint::LessEqConst(0, f64::NAN)),
        "NaN constant"
    );

    let clauses = vec![
        vec![
            OutputConstraint::GreaterEqConst(0, 0.0),
            OutputConstraint::LessEqConst(0, 0.4),
        ],
        vec![OutputConstraint::GreaterEqConst(0, 0.0)],
        vec![],
    ];
    let (verdicts, margins) = screen_clause_verdicts(&lo, &hi, &clauses);
    assert_eq!(verdicts, vec![true, false, false]);
    assert!((margins[0] - 0.1).abs() < 1e-12, "best constraint margin");
    assert!(margins[1] < 0.0);
    assert_eq!(margins[2], f64::NEG_INFINITY);
}

/// The union is a pure OR: it adds the screen's refutations, never clears a
/// precheck refutation, and ignores a missing or mis-sized screen result.
#[test]
fn fwd_root_screen_union_is_a_pure_or() {
    let precheck = [true, false, false, true];
    let screen = [false, true, false, true];
    assert_eq!(
        union_refutations(&precheck, Some(&screen)),
        vec![true, true, false, true]
    );
    assert_eq!(union_refutations(&precheck, None), precheck.to_vec());
    assert_eq!(
        union_refutations(&precheck, Some(&[true, true])),
        precheck.to_vec(),
        "a mis-sized screen result changes nothing"
    );
    let all_false = [false; 4];
    assert_eq!(
        union_refutations(&precheck, Some(&all_false)),
        precheck.to_vec()
    );
}

// ---------------------------------------------------------------------------
// Real corpus plumbing
// ---------------------------------------------------------------------------

const CORPORA: [&str; 3] = [
    "benchmarks/vnncomp2025/benchmarks/ml4acopf_2024",
    "benchmarks/vnncomp2026/benchmarks/ml4acopf_2024/1.0",
    "benchmarks/vnncomp2026/benchmarks/ml4acopf_2024/2.0",
];
const CORPUS_2026: &str = "benchmarks/vnncomp2026/benchmarks/ml4acopf_2024/1.0";

/// Resolve a repo-relative corpus path in this checkout or, from a worktree,
/// in the main checkout (via the worktree's `.git` file).
fn corpus_path(relative: &str) -> PathBuf {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let direct = workspace.join(relative);
    if direct.exists() {
        return direct;
    }
    if let Ok(text) = std::fs::read_to_string(workspace.join(".git")) {
        if let Some(gitdir) = text.trim().strip_prefix("gitdir:") {
            let gitdir = PathBuf::from(gitdir.trim());
            if let Some(main_git) = gitdir
                .ancestors()
                .find(|path| path.file_name().is_some_and(|name| name == ".git"))
            {
                if let Some(main) = main_git.parent() {
                    let candidate = main.join(relative);
                    if candidate.exists() {
                        return candidate;
                    }
                }
            }
        }
    }
    panic!(
        "VOID (not a pass): {relative} is missing under {} and under the main checkout",
        workspace.display()
    );
}

fn load_graph(onnx: &Path) -> GraphNetwork {
    let model = ny_onnx::load_onnx_with_config(onnx, &ny_onnx::OnnxLoadConfig::default())
        .unwrap_or_else(|error| panic!("load {}: {error:?}", onnx.display()));
    model
        .to_graph_network_with_options(ny_onnx::GraphNetworkOptions {
            compound_node_policy: ny_onnx::CompoundNodePolicy::DecomposeNormalization,
            ..ny_onnx::GraphNetworkOptions::default()
        })
        .unwrap_or_else(|error| panic!("graph {}: {error:?}", onnx.display()))
}

/// `(graph, lower, upper, clauses)` of one row, with C1 armed on the graph (the
/// shipped ml4acopf preset arms `bab.pwl_lookup_exact_range`).
fn load_row(
    corpus: &str,
    model: &str,
    prop: &str,
) -> (
    GraphNetwork,
    PathBuf,
    Vec<f32>,
    Vec<f32>,
    Vec<Vec<OutputConstraint>>,
) {
    let onnx = corpus_path(&format!("{corpus}/onnx/{model}.onnx"));
    let vnnlib = corpus_path(&format!("{corpus}/vnnlib/{prop}.vnnlib"));
    let mut graph = load_graph(&onnx);
    graph.set_pwl_lookup_exact_range(true);
    let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib)
        .unwrap_or_else(|error| panic!("vnnlib {}: {error:?}", vnnlib.display()));
    let (lo, hi) = spec.split_input_bounds_f32();
    let clauses = if spec.output_constraint_clauses.is_empty() {
        vec![spec.output_constraints]
    } else {
        spec.output_constraint_clauses
    };
    (graph, onnx, lo, hi, clauses)
}

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn between(&mut self, lo: f32, hi: f32) -> f32 {
        let v = f64::from(lo) + (f64::from(hi) - f64::from(lo)) * self.unit();
        (v as f32).clamp(lo, hi)
    }
}

/// `count` f32 points of the box (the midpoint, then alternately uniform and
/// vertex points), each exactly representable in both f32 and f64.
fn box_points(lo: &[f32], hi: &[f32], count: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut points = vec![lo
        .iter()
        .zip(hi)
        .map(|(&l, &h)| ((0.5 * f64::from(l) + 0.5 * f64::from(h)) as f32).clamp(l, h))
        .collect::<Vec<f32>>()];
    for k in 0..count {
        points.push(
            lo.iter()
                .zip(hi)
                .map(|(&l, &h)| {
                    if k % 2 == 0 {
                        rng.between(l, h)
                    } else if rng.unit() < 0.5 {
                        l
                    } else {
                        h
                    }
                })
                .collect(),
        );
    }
    points
}

/// Relative slack for the program's OWN binary64 evaluation error when its
/// point value is compared with ny's rigorous point enclosure. The enclosure
/// contains the EXACT value; the program's f64 value differs from that by its
/// own rounding only (~1e-12 absolute even on the 300-bus balance rows), never
/// by f32 noise, so this slack is orders of magnitude above what it absorbs.
const POINT_EVAL_SLACK: f64 = 1e-9;

/// The program's own f64 point semantics agree with ny's graph at every node
/// the program computes (the wiring check: a layout or index error in the
/// zonotope would corrupt the sampled "truth" and the enclosure alike).
///
/// The reference is ny's SOUND graph IBP of the degenerate box `[x, x]`
/// (`collect_node_bounds`, C1 armed exactly as in production): every node box
/// encloses the EXACT value of ny's layer semantics at `x`, so a correctly
/// wired program's binary64 value lies inside it, up to
/// `POINT_EVAL_SLACK`. The enclosure carries its own width: it is one or a
/// few f32 ulps wherever f32 rounding is benign, and it widens honestly where
/// it is not (a Floor whose argument straddles an integer at `x` encloses both
/// branches).
///
/// This replaces a comparison against ny's COLLAPSED f32 point forward
/// (`collect_node_activations_pointwise`) under a fixed tolerance of
/// `1e-3 * max(1, max |node|)`. That tolerance is not a bound on the collapsed
/// reference's own error, and on the 300-bus linear-residual nets the
/// reference is wrong by more than it: a near-zero NEGATIVE angle difference
/// is range-reduced in f32 as `theta + 2*pi` and folded through the Relu
/// `|.|` chain at magnitudes up to ~9, which leaves ~7e-7 absolute in the
/// sine surrogate; branch coefficients up to 1500..2120 carry that into the
/// flows, and the power-balance rows (values ~3e-2) end ~1.3e-3 away from the
/// exact value — with a correct program. Membership in the rigorous enclosure
/// has no such false alarm and is at least as strict wherever the enclosure is
/// narrow, which the non-vacuity floor below requires of most values.
fn cross_check_semantics(
    model: &str,
    graph: &GraphNetwork,
    program: &DagProgram,
    points: &[Vec<f32>],
) {
    let mut compared = 0usize;
    let mut narrow = 0usize;
    for x32 in points.iter().take(3) {
        let degenerate = tensor(x32, x32);
        let enclosure = graph
            .collect_node_bounds(&degenerate)
            .expect("ny point enclosure (sound graph IBP of the degenerate box)");
        let exact: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();
        program
            .evaluate_points(&exact, 1, |_, name, values| {
                let Some(bounds) = enclosure.get(name) else {
                    return;
                };
                assert_eq!(bounds.len(), values.len(), "{model}: `{name}` length");
                for (j, ((&mine, &lo), &hi)) in values
                    .iter()
                    .zip(bounds.lower().iter())
                    .zip(bounds.upper().iter())
                    .enumerate()
                {
                    let (lo, hi) = (f64::from(lo), f64::from(hi));
                    let slack = POINT_EVAL_SLACK * (1.0 + mine.abs());
                    assert!(
                        lo - slack <= mine && mine <= hi + slack,
                        "{model}: `{name}`[{j}]: zonotope program {mine:e} is outside ny's \
                         point enclosure [{lo:e}, {hi:e}]"
                    );
                    if hi - lo <= 1e-3 * (1.0 + mine.abs()) {
                        narrow += 1;
                    }
                    compared += 1;
                }
            })
            .expect("program evaluation");
    }
    assert!(
        compared > 0,
        "{model}: semantic cross-check compared nothing: VOID"
    );
    // Non-vacuity: a membership check against wide enclosures would pass a
    // mis-wired program. Measured (python replay of ny's directed IBP on these
    // points): 96..100% of values sit in an enclosure narrower than
    // 1e-3 (1 + |v|) on every ml4acopf model.
    assert!(
        4 * narrow >= 3 * compared,
        "{model}: semantic cross-check is VACUOUS: only {narrow}/{compared} point \
         enclosures are narrower than 1e-3 (1 + |v|)"
    );
}

/// Box and pointwise-form containment at every node for the given points.
#[derive(Debug, Default)]
struct Containment {
    form: ContainmentReport,
    box_checked: usize,
    box_violations: usize,
    worst_box_excess: f64,
}

fn contain(
    program: &DagProgram,
    input: &BoundedTensor,
    ibp: &HashMap<String, BoundedTensor>,
    points: &[Vec<f32>],
) -> (Containment, Vec<f64>, Vec<f64>) {
    let exact: Vec<Vec<f64>> = points
        .iter()
        .map(|p| p.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let flat: Vec<f64> = exact.iter().flatten().copied().collect();
    let mut values: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    program
        .evaluate_points(&flat, exact.len(), |_, name, v| {
            values.entry(name.to_string()).or_default().push(v.to_vec());
        })
        .expect("program evaluation");
    let mut report = Containment::default();
    let mut observer = |view: &DagNodeView<'_>| {
        let Some(node_values) = values.get(view.node) else {
            return;
        };
        let summary = summarize(&ZonoNodeView {
            step: None,
            node: view.node,
            shape: view.shape,
            state: view.state,
            abs_g: view.abs_g,
            input_symbols: view.input_symbols,
        });
        for (point, v) in exact.iter().zip(node_values) {
            check_point(&summary, point, v, &mut report.form);
            for (j, &value) in v.iter().enumerate() {
                report.box_checked += 1;
                let excess = (view.lo[j] - value).max(value - view.hi[j]);
                if excess > 1e-10 * (1.0 + value.abs()) {
                    report.box_violations += 1;
                    report.worst_box_excess = report.worst_box_excess.max(excess);
                }
            }
        }
    };
    let opts = DagForwardOptions {
        intersect_ibp: true,
        byte_budget: BIG_BUDGET,
    };
    let outcome = program
        .forward(input, &opts, ibp, None, Some(&mut observer))
        .expect("certified forward pass");
    (report, outcome.output_lo, outcome.output_hi)
}

// ---------------------------------------------------------------------------
// Named V4 tests on the real models
// ---------------------------------------------------------------------------

/// Named V4 test 1: every ml4acopf ONNX in both years (all 27 files), the
/// prop4 box and two random sub-boxes: the program compiles (admitted), its
/// f64 point values lie in ny's rigorous point enclosure at every node
/// (`cross_check_semantics`), every node's box AND pointwise form contain
/// the sampled values, and the certified output box contains ONNX Runtime's
/// output (f32 tolerance `1e-5 (1 + |y|)`).
#[test]
fn fwd_root_screen_contains_ort_samples_ml4acopf() {
    let buses = ["14", "118", "300"];
    let variants = ["-linear-residual", "-linear-nonresidual", ""];
    let mut rng = Rng(0x4657_4452_4F4F_5431);
    let mut files = 0usize;
    let mut total = Containment::default();
    let mut ort_points = 0usize;
    let mut max_ort_excursion = 0.0f64;
    for corpus in CORPORA {
        for bus in buses {
            for variant in variants {
                let model = format!("{bus}_ieee_ml4acopf{variant}");
                let (graph, onnx, box_lo, box_hi, _) =
                    load_row(corpus, &model, &format!("{bus}_ieee_prop4"));
                let n = box_lo.len();
                let mut ort = ny_onnx::diff::OrtForward::from_path(&onnx, n).unwrap_or_else(|e| {
                    panic!("VOID (not a pass): ONNX Runtime unavailable for {model}: {e:?}")
                });
                files += 1;
                for sub in 0..3 {
                    let (lo, hi): (Vec<f32>, Vec<f32>) = if sub == 0 {
                        (box_lo.clone(), box_hi.clone())
                    } else {
                        box_lo
                            .iter()
                            .zip(&box_hi)
                            .map(|(&l, &h)| {
                                let (a, b) = (rng.between(l, h), rng.between(l, h));
                                (a.min(b), a.max(b))
                            })
                            .unzip()
                    };
                    let input = tensor(&lo, &hi);
                    let (program, ibp) = DagProgram::compile_with_ibp(&graph, &input, None)
                        .unwrap_or_else(|e| panic!("{model}: the screen must admit it: {e}"));
                    let points = box_points(&lo, &hi, 12, &mut rng);
                    if sub == 0 {
                        cross_check_semantics(&model, &graph, &program, &points);
                        let census = program.census();
                        if variant.is_empty() {
                            for kind in ["sigmoid", "sin", "cos", "mul", "square"] {
                                assert!(census.contains(kind), "{model}: census {census}");
                            }
                        } else {
                            assert!(census.contains("pwl:3"), "{model}: census {census}");
                        }
                    }
                    let (report, out_lo, out_hi) = contain(&program, &input, &ibp, &points);
                    for x in &points {
                        let y = ort
                            .run(x)
                            .unwrap_or_else(|e| panic!("ORT run {model}: {e:?}"));
                        assert_eq!(y.len(), out_lo.len(), "{model}: output width");
                        for (j, &value) in y.iter().enumerate() {
                            let value = f64::from(value);
                            let excursion = (out_lo[j] - value).max(value - out_hi[j]);
                            max_ort_excursion = max_ort_excursion.max(excursion);
                            assert!(
                                excursion <= 1e-5 * (1.0 + value.abs()),
                                "{model} sub {sub}: ORT Y_{j} = {value:e} escapes the certified \
                                 box [{:e}, {:e}]",
                                out_lo[j],
                                out_hi[j]
                            );
                        }
                        ort_points += 1;
                    }
                    assert_eq!(
                        report.box_violations, 0,
                        "{model} sub {sub}: box containment violated: {report:?}"
                    );
                    assert_eq!(
                        report.form.form_violations, 0,
                        "{model} sub {sub}: pointwise form containment violated: {report:?}"
                    );
                    total.form.absorb(&report.form);
                    total.box_checked += report.box_checked;
                }
            }
        }
    }
    assert_eq!(
        files, 27,
        "every ml4acopf model in both years must be exercised"
    );
    assert!(
        total.form.checked > 100_000,
        "checked too little: {total:?}"
    );
    eprintln!(
        "[fwd-root-test] files={files} box_checked={} form_checked={} ort_points={ort_points} \
         max_ort_excursion={max_ort_excursion:e}",
        total.box_checked, total.form.checked
    );
}

/// Named V4 test 5 (the stop-rule control): on the TRUE-SAT rows 14 lr/lnr p1,
/// 118 lnr p5 and 300 base/lr p2, the production screen must NOT refute every
/// clause, and no clause it does refute may be satisfied by any sampled ORT
/// output.
#[test]
fn fwd_root_screen_true_sat_rows_do_not_verify() {
    let rows = [
        ("14_ieee_ml4acopf-linear-residual", "14_ieee_prop1"),
        ("14_ieee_ml4acopf-linear-nonresidual", "14_ieee_prop1"),
        ("118_ieee_ml4acopf-linear-nonresidual", "118_ieee_prop5"),
        ("300_ieee_ml4acopf", "300_ieee_prop2"),
        ("300_ieee_ml4acopf-linear-residual", "300_ieee_prop2"),
    ];
    let mut rng = Rng(0x5341_5452_4F57_5331);
    for (model, prop) in rows {
        let (graph, onnx, lo, hi, clauses) = load_row(CORPUS_2026, model, prop);
        let input = tensor(&lo, &hi);
        let screen = run_forward_root_screen(&graph, &input, BIG_BUDGET, None, true)
            .unwrap_or_else(|d| panic!("{model}: the screen must admit it: {}", d.reason));
        let (refuted, margins) =
            screen_clause_verdicts(&screen.output_lo, &screen.output_hi, &clauses);
        let count = refuted.iter().filter(|&&r| r).count();
        eprintln!(
            "[fwd-root-test] true-sat {model} {prop}: refuted={count}/{} min_margin={:e}",
            clauses.len(),
            margins.iter().copied().fold(f64::INFINITY, f64::min)
        );
        assert!(
            count < clauses.len(),
            "SOUNDNESS: {model} {prop} is truth-SAT and the screen refuted every clause"
        );
        let mut ort = ny_onnx::diff::OrtForward::from_path(&onnx, lo.len())
            .unwrap_or_else(|e| panic!("VOID (not a pass): ONNX Runtime unavailable: {e:?}"));
        for x in box_points(&lo, &hi, 32, &mut rng) {
            let y = ort.run(&x).expect("ORT run");
            let y_lo: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
            for (clause, &was_refuted) in clauses.iter().zip(&refuted) {
                if !was_refuted {
                    continue;
                }
                // A sampled point SATISFIES the clause when no constraint is
                // provably false on the degenerate box [y, y].
                let satisfied = clause
                    .iter()
                    .all(|c| !constraint_refuted_f64(&y_lo, &y_lo, c).0);
                assert!(
                    !satisfied,
                    "SOUNDNESS: {model} {prop}: a refuted clause is satisfied at a sample"
                );
            }
        }
    }
}

/// Named V4 test 6: the rows C1 banks (14 lr p3/p4) and the IBP-banked native
/// rows (118 base p3, 300 base p3) stay refuted under the UNION of ny's IBP
/// route (C1 armed) and the screen — the screen never replaces IBP/C1, so
/// arming it cannot lose them.
#[test]
fn fwd_root_screen_union_keeps_c1_rows() {
    let rows = [
        ("14_ieee_ml4acopf-linear-residual", "14_ieee_prop3"),
        ("14_ieee_ml4acopf-linear-residual", "14_ieee_prop4"),
        ("118_ieee_ml4acopf", "118_ieee_prop3"),
        ("300_ieee_ml4acopf", "300_ieee_prop3"),
    ];
    for (model, prop) in rows {
        let (graph, _, lo, hi, clauses) = load_row(CORPUS_2026, model, prop);
        let input = tensor(&lo, &hi);
        let ibp = graph.propagate_ibp(&input).expect("IBP forward (C1 armed)");
        let ibp_refuted: Vec<bool> = clauses
            .iter()
            .map(|clause| clause_provably_unsat(&ibp, clause))
            .collect();
        let screen = run_forward_root_screen(&graph, &input, BIG_BUDGET, None, true)
            .unwrap_or_else(|d| panic!("{model}: the screen must admit it: {}", d.reason));
        let (screen_refuted, _) =
            screen_clause_verdicts(&screen.output_lo, &screen.output_hi, &clauses);
        let merged = union_refutations(&ibp_refuted, Some(screen_refuted.as_slice()));
        let count = |v: &[bool]| v.iter().filter(|&&r| r).count();
        eprintln!(
            "[fwd-root-test] union {model} {prop}: ibp={} screen={} union={}/{}",
            count(ibp_refuted.as_slice()),
            count(screen_refuted.as_slice()),
            count(merged.as_slice()),
            clauses.len()
        );
        assert!(
            merged.iter().zip(&ibp_refuted).all(|(&m, &i)| m || !i),
            "the union must keep every IBP refutation"
        );
        assert_eq!(
            count(merged.as_slice()),
            clauses.len(),
            "{model} {prop}: the union must keep the banked row refuted"
        );
    }
}

/// Named V4 test 7: replica parity. On the four F1 rows the PURE zonotope pass
/// (no IBP intersection — the F1 replica's ZONO mode) reproduces F1's worst
/// clause margins within `1e-6`, both against ZONO_NODIAG (this build's
/// product policy) and ZONO; and the PRODUCTION pass (IBP ∩ C1 intersection)
/// refutes every clause of all four rows.
#[test]
fn fwd_root_screen_replica_parity() {
    // (model, prop, F1 ZONO_NODIAG worst margin, F1 ZONO worst margin), from
    // gates/F1/r_target.json.
    let rows = [
        (
            "14_ieee_ml4acopf",
            "14_ieee_prop1",
            5.747_702_100_2e-5,
            5.747_702_637_4e-5,
        ),
        (
            "118_ieee_ml4acopf-linear-residual",
            "118_ieee_prop5",
            6.087_968_801_6e-3,
            6.087_970_128_8e-3,
        ),
        (
            "300_ieee_ml4acopf-linear-nonresidual",
            "300_ieee_prop105",
            8.804_303_323_9e-3,
            8.804_303_967_3e-3,
        ),
        (
            "300_ieee_ml4acopf-linear-residual",
            "300_ieee_prop105",
            8.080_707_714_0e-3,
            8.080_708_436_2e-3,
        ),
    ];
    for (model, prop, nodiag, zono) in rows {
        let (graph, _, lo, hi, clauses) = load_row(CORPUS_2026, model, prop);
        let input = tensor(&lo, &hi);
        let pure = run_forward_root_screen(&graph, &input, BIG_BUDGET, None, false)
            .unwrap_or_else(|d| panic!("{model}: pure pass declined: {}", d.reason));
        let (_, margins) = screen_clause_verdicts(&pure.output_lo, &pure.output_hi, &clauses);
        let worst = margins.iter().copied().fold(f64::INFINITY, f64::min);
        eprintln!(
            "[fwd-root-test] parity {model} {prop}: worst={worst:e} f1_nodiag={nodiag:e} \
             f1_zono={zono:e} symbols={} census={}",
            pure.stats.symbols, pure.census
        );
        assert!(
            (worst - nodiag).abs() <= 1e-6 && (worst - zono).abs() <= 1e-6,
            "{model} {prop}: pure-zonotope worst margin {worst:e} vs F1 {nodiag:e}/{zono:e}"
        );

        let production = run_forward_root_screen(&graph, &input, BIG_BUDGET, None, true)
            .unwrap_or_else(|d| panic!("{model}: production pass declined: {}", d.reason));
        let (refuted, prod_margins) =
            screen_clause_verdicts(&production.output_lo, &production.output_hi, &clauses);
        eprintln!(
            "[fwd-root-test] production {model} {prop}: refuted={}/{} worst={:e}",
            refuted.iter().filter(|&&r| r).count(),
            clauses.len(),
            prod_margins.iter().copied().fold(f64::INFINITY, f64::min)
        );
        assert!(
            refuted.iter().all(|&r| r),
            "{model} {prop}: the production screen must refute every clause"
        );
    }
}
