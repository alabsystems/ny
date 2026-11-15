// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #pwl-lookup: the precheck's row ∩ IBP intersection, and real-ONNX
//! containment of the armed IBP on every ml4acopf lookup-table net.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::layers::{
    AddConstantLayer, LinearLayer, ReLULayer, SqueezeLayer, SubConstantLayer, UnsqueezeLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

use super::{
    clause_provably_unsat, crown_precheck_clauses, ibp_spec_row_bounds,
    intersect_output_enclosures, intersect_row_bounds, is_clause_unsatisfiable,
    precheck_output_with_ibp, BetaCrownModel,
};

fn tensor(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("upper"),
    )
    .expect("valid tensor")
}

fn named(name: &str, layer: Layer, input: &str) -> GraphNode {
    GraphNode::new(name, layer, vec![input.to_string()])
}

/// The "hat" lookup table `f(x) = ReLU(x+1) − 2·ReLU(x) + ReLU(x−1)`, whose
/// exact range over any interval is inside `[0, 1]`, followed by a per-output
/// offset. With three UNSTABLE hinges on `[-3, 3]`, hinge-by-hinge IBP gives
/// `[-6, 4]` and triangle-relaxed CROWN an upper bound near 6 — only the exact
/// 1-D range sees `f <= 1`.
fn hat_graph(offsets: &[f32]) -> GraphNetwork {
    let n = offsets.len();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        Layer::Linear(LinearLayer::new(Array2::eye(n), None).expect("identity")),
    ));
    graph.add_node(named("col", Layer::Unsqueeze(UnsqueezeLayer::new(-1)), "z"));
    graph.add_node(named(
        "shift",
        Layer::SubConstant(SubConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).expect("b"),
        )),
        "col",
    ));
    graph.add_node(named("hinge", Layer::ReLU(ReLULayer), "shift"));
    graph.add_node(named(
        "table",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 3), vec![1.0, -2.0, 1.0]).expect("w"),
                None::<Array1<f32>>,
            )
            .expect("table"),
        ),
        "hinge",
    ));
    graph.add_node(named(
        "flat",
        Layer::Squeeze(SqueezeLayer::new(-1)),
        "table",
    ));
    graph.add_node(named(
        "out",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), offsets.to_vec()).expect("offsets"),
        )),
        "flat",
    ));
    graph.set_output("out");
    graph
}

fn precheck(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    deadline: Option<Instant>,
) -> Vec<bool> {
    let model = BetaCrownModel::Graph(Box::new(graph.clone()));
    crown_precheck_clauses(&model, input, clauses, &[], None, deadline)
}

/// The precheck decides each row on `CROWN ∩ IBP` when (and only when) the
/// graph is armed: the pure helpers intersect shrink-only and fail closed on
/// disjoint/NaN rows, the IBP spec row is directed-rounded, and end to end a
/// clause only the exact lookup range can refute is verified armed and NOT
/// verified unarmed — on both the per-output and the spec-guided routes.
#[test]
fn precheck_row_bound_is_crown_intersect_ibp() {
    // --- Pure helpers -----------------------------------------------------
    assert_eq!(
        intersect_row_bounds((0.0, 5.0), (-1.0, 1.2)),
        Some((0.0, 1.2))
    );
    assert_eq!(
        intersect_row_bounds((f32::NEG_INFINITY, f32::INFINITY), (1.0, 2.0)),
        Some((1.0, 2.0))
    );
    assert_eq!(
        intersect_row_bounds((0.0, 1.0), (2.0, 3.0)),
        None,
        "disjoint"
    );
    assert_eq!(
        intersect_row_bounds((f32::NAN, 1.0), (0.0, 3.0)),
        None,
        "nan"
    );

    let crown = tensor(&[0.0, 0.0], &[5.0, 5.0]);
    let merged = intersect_output_enclosures(&crown, &tensor(&[-1.0, 3.0], &[1.2, 4.0]))
        .expect("overlapping enclosures intersect");
    assert_eq!(
        merged.lower().iter().copied().collect::<Vec<_>>(),
        [0.0, 3.0]
    );
    assert_eq!(
        merged.upper().iter().copied().collect::<Vec<_>>(),
        [1.2, 4.0]
    );
    assert!(
        intersect_output_enclosures(&crown, &tensor(&[-1.0, 6.0], &[1.2, 7.0])).is_err(),
        "one disjoint element must refuse the WHOLE intersection"
    );
    assert!(intersect_output_enclosures(&crown, &tensor(&[0.0], &[1.0])).is_err());

    let ibp = tensor(&[1.0, 0.5], &[2.0, 0.75]);
    let c =
        Array2::from_shape_vec((4, 2), vec![1.0, -1.0, -1.0, 1.0, 2.0, -1.0, 1.0, 0.0]).expect("c");
    let rows = ibp_spec_row_bounds(&ibp, &c);
    assert_eq!(rows[0], Some((0.25, 1.5)), "Y0 − Y1");
    assert_eq!(rows[1], Some((-1.5, -0.25)), "Y1 − Y0");
    assert_eq!(rows[2], None, "a non-unit coefficient keeps the CROWN row");
    assert_eq!(rows[3], None, "a single-output row keeps the CROWN row");
    // Directed rounding: 0.1 − 0.2 is not exact in f32; the row must still
    // enclose the real difference of the two f32 endpoints.
    let inexact = tensor(&[0.1, 0.2], &[0.1, 0.2]);
    let (lo, up) = ibp_spec_row_bounds(
        &inexact,
        &Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).expect("c"),
    )[0]
    .expect("finite row");
    let exact = f64::from(0.1f32) - f64::from(0.2f32);
    assert!(f64::from(lo) <= exact && exact <= f64::from(up));

    // --- End to end: per-output route (constant threshold) ---------------
    let graph = hat_graph(&[0.0]);
    let input = tensor(&[-3.0], &[3.0]);
    let clause = vec![vec![OutputConstraint::GreaterEqConst(0, 1.2)]];
    let deadline = Some(Instant::now() + Duration::from_secs(120));
    assert_eq!(
        precheck(&graph, &input, &clause, deadline),
        vec![false],
        "unarmed: hinge-by-hinge bounds cannot refute Y0 >= 1.2"
    );
    let mut armed = graph.clone();
    armed.set_pwl_lookup_exact_range(true);
    // The helper itself, on a deliberately wide CROWN box: the armed IBP box
    // (exact lookup range) is what refutes the clause.
    let wide = tensor(&[-10.0], &[10.0]);
    let merged = precheck_output_with_ibp(&armed, &input, &clause, wide, deadline);
    assert!(
        merged.upper()[[0]] <= 1.0 + 1e-6 && merged.lower()[[0]] >= -1e-6,
        "armed: CROWN [-10, 10] ∩ exact IBP must be ~[0, 1], got [{}, {}]",
        merged.lower()[[0]],
        merged.upper()[[0]]
    );
    assert!(is_clause_unsatisfiable(
        &clause[0],
        merged.lower(),
        merged.upper()
    ));
    let crown_ok_with_deadline = graph
        .propagate_crown_with_engine_and_deadline(&input, None, deadline)
        .is_ok();
    let armed_with_deadline = precheck(&armed, &input, &clause, deadline);
    if crown_ok_with_deadline {
        assert_eq!(
            armed_with_deadline,
            vec![true],
            "armed: the exact range f <= 1 refutes Y0 >= 1.2"
        );
    }
    // No deadline: the graph CROWN backward runs, and the armed precheck
    // must add the IBP box on top of it whenever that backward succeeds.
    let unarmed_none = precheck(&graph, &input, &clause, None);
    let armed_none = precheck(&armed, &input, &clause, None);
    assert!(
        armed_none[0] || !unarmed_none[0],
        "arming may only ADD verified clauses"
    );
    if graph
        .propagate_crown_with_engine_and_deadline(&input, None, None)
        .is_ok()
    {
        assert!(
            armed_none[0],
            "armed: CROWN ∩ IBP must refute Y0 >= 1.2 when the CROWN backward succeeds"
        );
    }

    // --- End to end: spec-guided route (relational) -----------------------
    // Y0 = hat(x0) + 0 over x0 ∈ [-3, 3] (exact [0, 1]); Y1 = hat(x1) + 1.5
    // over x1 ∈ [2, 3], where every hinge is stable (Y1 ≡ 1.5 exactly).
    let graph = hat_graph(&[0.0, 1.5]);
    let input = tensor(&[-3.0, 2.0], &[3.0, 3.0]);
    let clause = vec![vec![OutputConstraint::GreaterEq(0, 1)]];
    for deadline in [deadline, None] {
        // Sufficient for a decision: if the per-output CROWN returns, either
        // the spec route (row ∩ IBP row) or its per-output fallback (box ∩
        // IBP box) sees the exact lookup range.
        let crown_backward_ok = graph
            .propagate_crown_with_engine_and_deadline(&input, None, deadline)
            .is_ok();
        let unarmed = precheck(&graph, &input, &clause, deadline);
        let mut armed = graph.clone();
        armed.set_pwl_lookup_exact_range(true);
        let armed_result = precheck(&armed, &input, &clause, deadline);
        assert!(
            armed_result[0] || !unarmed[0],
            "arming may only ADD verified clauses (deadline {deadline:?})"
        );
        if crown_backward_ok {
            assert!(
                armed_result[0],
                "armed: Y0 <= 1 < 1.5 = Y1 refutes Y0 >= Y1 (deadline {deadline:?})"
            );
        }
    }
    let unarmed = precheck(
        &graph,
        &input,
        &clause,
        Some(Instant::now() + Duration::from_secs(120)),
    );
    assert_eq!(
        unarmed,
        vec![false],
        "unarmed: neither IBP [-6, 4] nor CROWN (~6) can refute Y0 >= Y1 = 1.5"
    );
}

// ---------------------------------------------------------------------------
// Real ONNX: every ml4acopf lookup-table net, both years, ORT-sampled
// ---------------------------------------------------------------------------

/// xorshift64* (no `rand` in the fixture).
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

/// Every ml4acopf net in both corpora (2025 and both 2026 copies): the six
/// `-linear-*` nets must census exactly 3 lookup tables (the sigmoid and the
/// Cos/Sin corrections) and the three native nets none. On every linear net,
/// over the full prop4 input box and random sub-boxes, BOTH armed IBP entry
/// points are (a) inside the unarmed box, (b) strictly tighter somewhere, and
/// (c) contain ONNX Runtime's output at random points of the box.
///
/// (c) carries a float tolerance `1e-5·(1 + |y|)`: ny encloses the REAL
/// function and ORT evaluates it in f32 (a 54-term MatMul accumulation), so an
/// ulp-level ORT excursion outside an exact-range box is ORT's rounding, not a
/// defect. The ulp-level certificate is the exact-rational synthetic suite in
/// `ny-propagate` (`pwl_lookup_range_*`); this test guards the WIRING (pattern
/// match, broadcast orientation, element order) on the real graphs, where a
/// defect shows up as an O(1e-3..1) miss.
///
/// Needs the corpus. A missing fixture or an unavailable ORT is VOID and FAILS
/// (it never passes by skipping): in a worktree, symlink `benchmarks/` from the
/// main tree first.
#[test]
fn pwl_lookup_real_onnx_contains_ort_samples() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let corpora = [
        "benchmarks/vnncomp2025/benchmarks/ml4acopf_2024",
        "benchmarks/vnncomp2026/benchmarks/ml4acopf_2024/1.0",
        "benchmarks/vnncomp2026/benchmarks/ml4acopf_2024/2.0",
    ];
    let buses = ["14", "118", "300"];
    let variants = ["-linear-residual", "-linear-nonresidual", ""];

    let mut missing: Vec<PathBuf> = Vec::new();
    let mut jobs = Vec::new();
    for corpus in corpora {
        for bus in buses {
            for variant in variants {
                let onnx = root.join(format!("{corpus}/onnx/{bus}_ieee_ml4acopf{variant}.onnx"));
                let vnnlib = root.join(format!("{corpus}/vnnlib/{bus}_ieee_prop4.vnnlib"));
                for path in [&onnx, &vnnlib] {
                    if !path.is_file() {
                        missing.push(path.clone());
                    }
                }
                jobs.push((onnx, vnnlib, !variant.is_empty()));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "VOID (not a pass): the ml4acopf corpus is missing {} file(s), e.g. {}. Symlink \
         benchmarks/ from the main tree into this worktree and re-run.",
        missing.len(),
        missing[0].display()
    );

    let mut rng = Rng(0x5057_4C4F_4E4E_5831);
    let mut pattern_files = 0usize;
    let mut ort_points = 0usize;
    let mut max_excursion = 0.0f64;
    for (onnx, vnnlib, is_linear) in jobs {
        let mut graph = load_graph(&onnx);
        let census = graph.pwl_lookup_census();
        if !is_linear {
            assert!(
                census.is_empty(),
                "{}: a native-Sigmoid net must carry no lookup table, found {census:?}",
                onnx.display()
            );
            continue;
        }
        assert_eq!(
            census.len(),
            3,
            "{}: expected the sigmoid + Cos + Sin tables, found {census:?}",
            onnx.display()
        );
        assert!(census.iter().all(|(_, hinges)| *hinges >= 2));
        pattern_files += 1;

        let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib)
            .unwrap_or_else(|error| panic!("vnnlib {}: {error:?}", vnnlib.display()));
        let (box_lo, box_hi) = spec.split_input_bounds_f32();
        let n = box_lo.len();
        let mut ort = ny_onnx::diff::OrtForward::from_path(&onnx, n).unwrap_or_else(|error| {
            panic!(
                "VOID (not a pass): ONNX Runtime unavailable for {}: {error:?}",
                onnx.display()
            )
        });

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

            graph.set_pwl_lookup_exact_range(false);
            let plain_forward = graph.propagate_ibp(&input).expect("plain forward");
            let plain_collect = graph
                .collect_node_bounds(&input)
                .expect("plain collect")
                .remove(graph.output_name())
                .expect("output bounds");
            graph.set_pwl_lookup_exact_range(true);
            let armed_forward = graph.propagate_ibp(&input).expect("armed forward");
            let armed_collect = graph
                .collect_node_bounds(&input)
                .expect("armed collect")
                .remove(graph.output_name())
                .expect("output bounds");
            graph.set_pwl_lookup_exact_range(false);

            let points: Vec<Vec<f32>> = (0..48)
                .map(|_| {
                    lo.iter()
                        .zip(&hi)
                        .map(|(&l, &h)| rng.between(l, h))
                        .collect()
                })
                .collect();
            let outputs: Vec<Vec<f32>> = points
                .iter()
                .map(|x| {
                    ort.run(x)
                        .unwrap_or_else(|error| panic!("ORT run {}: {error:?}", onnx.display()))
                })
                .collect();

            for (armed, plain, route) in [
                (&armed_forward, &plain_forward, "propagate_ibp"),
                (&armed_collect, &plain_collect, "collect_node_bounds"),
            ] {
                let al: Vec<f32> = armed.lower().iter().copied().collect();
                let au: Vec<f32> = armed.upper().iter().copied().collect();
                let pl: Vec<f32> = plain.lower().iter().copied().collect();
                let pu: Vec<f32> = plain.upper().iter().copied().collect();
                assert_eq!(al.len(), spec.num_outputs, "{route}: output width");
                let mut tighter = 0usize;
                for j in 0..al.len() {
                    assert!(
                        pl[j] <= al[j] && au[j] <= pu[j],
                        "{} {route}: armed [{}, {}] escapes plain [{}, {}] at Y_{j}",
                        onnx.display(),
                        al[j],
                        au[j],
                        pl[j],
                        pu[j]
                    );
                    tighter += usize::from(pl[j] < al[j] || au[j] < pu[j]);
                }
                assert!(
                    tighter > 0,
                    "{} {route}: the armed range tightened nothing on a lookup-table net",
                    onnx.display()
                );
                for y in &outputs {
                    assert_eq!(y.len(), al.len(), "ORT output width");
                    for j in 0..al.len() {
                        let value = f64::from(y[j]);
                        let tolerance = 1e-5 * (1.0 + value.abs());
                        let excursion = (f64::from(al[j]) - value).max(value - f64::from(au[j]));
                        max_excursion = max_excursion.max(excursion);
                        assert!(
                            excursion <= tolerance,
                            "{} {route}: ORT Y_{j} = {value:e} escapes the armed box \
                             [{:e}, {:e}] by {excursion:e}",
                            onnx.display(),
                            al[j],
                            au[j]
                        );
                    }
                    ort_points += 1;
                }
                if sub == 0 && route == "propagate_ibp" {
                    let refuted = |bounds: &BoundedTensor| {
                        spec.output_constraint_clauses
                            .iter()
                            .filter(|clause| clause_provably_unsat(bounds, clause))
                            .count()
                    };
                    eprintln!(
                        "[pwl-lookup-test] {} prop4 full box: clauses={} refuted armed={} \
                         plain={}",
                        onnx.display(),
                        spec.output_constraint_clauses.len(),
                        refuted(armed),
                        refuted(plain)
                    );
                }
            }
        }
    }
    assert_eq!(
        pattern_files, 18,
        "all 18 lookup-table files must be exercised"
    );
    eprintln!(
        "[pwl-lookup-test] files={pattern_files} ort_points={ort_points} \
         max_ort_excursion={max_excursion:e}"
    );
}
