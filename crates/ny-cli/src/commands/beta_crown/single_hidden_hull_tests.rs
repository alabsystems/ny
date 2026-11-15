// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #hull-root adapter tests.
//!
//! Hermetic tests (always on under `mip`) pin the gate, the fixed-schema
//! observable, dark-by-default behavior, and end-to-end verification of a
//! gadget whose ideal-hull minimum is known in closed form — including the
//! SAT negative control.
//!
//! The real-corpus soundness audits are `external-vnncomp` (they need the
//! gitignored `benchmarks/vnncomp2026` tree — symlink it into a worktree — and
//! run ~145 LPs, so run them in release):
//!
//! ```text
//! targo --unverified test --release -p ny-cli --features mip,external-vnncomp --bin ny hull_
//! ```

use super::*;
use ndarray::{arr1, arr2};
use ny_mip::single_hidden_hull::hull_root_bound;
use ny_onnx::vnnlib::parse_vnnlib;
use ny_propagate::layers::{LinearLayer, ReLULayer};

const GADGET_SPEC: &str = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 Y_1))
"#;

/// `Y_0 - Y_1 = constant - relu(x0 + x1 - 1) + 0.9 relu(x0) + 0.1 relu(x1)`
/// on `[0,1]^2`. The concave envelope of `relu(x0 + x1 - 1)` there is
/// `min(x0, x1)`, so the ideal-hull minimum is `constant` (up to the f32
/// rounding of 0.9/0.1, which only moves it by ~2e-8) and the true minimum is
/// `constant` too, attained at `(0,0)`: `constant > 0` is UNSAT, `< 0` is SAT.
fn gadget_network(constant: f32) -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0_f32, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            Some(arr1(&[-1.0_f32, 0.0, 0.0])),
        )
        .expect("first affine layer"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[-1.0_f32, 0.9, 0.1], [0.0, 0.0, 0.0]]),
            Some(arr1(&[constant, 0.0])),
        )
        .expect("second affine layer"),
    ));
    network
}

fn unit_box() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite input box")
}

fn spec(text: &str) -> VnnLibSpec {
    parse_vnnlib(text).expect("test spec parses")
}

fn far_deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn hull_root_verifies_the_gadget_row_through_the_lane_adapter() {
    let network = gadget_network(0.05);
    let problem = admit(&network, &unit_box(), &spec(GADGET_SPEC)).expect("gadget is admitted");
    let report = hull_root_bound(&problem, far_deadline());
    assert!(report.verified(), "{report:?}");
    let certified = report.certified.expect("certified");
    assert!(
        certified > 0.049 && certified <= 0.050_000_001,
        "certified {certified} is not the ideal-hull value"
    );

    let result = try_single_hidden_hull_root(
        SingleHiddenHullMode::Root,
        &network,
        &unit_box(),
        &spec(GADGET_SPEC),
        far_deadline(),
        Instant::now(),
    )
    .expect("armed hull root verifies the gadget");
    assert!(matches!(result.result, BabVerificationStatus::Verified));
}

#[test]
fn hull_does_not_verify_a_sat_row() {
    // Y_0 - Y_1 = -0.05 at (0,0): a genuine counterexample exists.
    let network = gadget_network(-0.05);
    let problem = admit(&network, &unit_box(), &spec(GADGET_SPEC)).expect("gadget is admitted");
    let report = hull_root_bound(&problem, far_deadline());
    assert!(!report.verified(), "{report:?}");
    assert!(
        report.certified.expect("certified") <= -0.049,
        "certified value exceeds the true minimum"
    );
    assert!(try_single_hidden_hull_root(
        SingleHiddenHullMode::Root,
        &network,
        &unit_box(),
        &spec(GADGET_SPEC),
        far_deadline(),
        Instant::now(),
    )
    .is_none());
}

#[test]
fn hull_root_is_inert_when_off() {
    // Off is the default and the shipped state: even a row the hull WOULD
    // prove is left to the historical lane.
    assert_eq!(SingleHiddenHullMode::default(), SingleHiddenHullMode::Off);
    assert!(try_single_hidden_hull_root(
        SingleHiddenHullMode::Off,
        &gadget_network(0.05),
        &unit_box(),
        &spec(GADGET_SPEC),
        far_deadline(),
        Instant::now(),
    )
    .is_none());
}

#[test]
fn hull_declines_on_non_single_hidden() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 1.0], [1.0, -1.0]]), None).expect("layer 1"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 0.5], [0.5, 1.0]]), None).expect("layer 2"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, -1.0], [0.0, 1.0]]), None).expect("layer 3"),
    ));
    assert_eq!(
        admit(&network, &unit_box(), &spec(GADGET_SPEC)).err(),
        Some("not_single_hidden")
    );
    assert!(try_single_hidden_hull_root(
        SingleHiddenHullMode::Root,
        &network,
        &unit_box(),
        &spec(GADGET_SPEC),
        far_deadline(),
        Instant::now(),
    )
    .is_none());
}

#[test]
fn hull_declines_multi_row_constant_and_disjunctive_specs() {
    let network = gadget_network(0.05);
    let header = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n\
                  (declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
                  (assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n\
                  (assert (>= X_1 0.0))\n(assert (<= X_1 1.0))\n";
    let two_rows = format!("{header}(assert (<= Y_0 Y_1))\n(assert (<= Y_0 0.5))\n");
    assert_eq!(
        admit(&network, &unit_box(), &spec(&two_rows)).err(),
        Some("not_one_relational_row")
    );
    let constant = format!("{header}(assert (<= Y_0 0.5))\n");
    assert_eq!(
        admit(&network, &unit_box(), &spec(&constant)).err(),
        Some("not_one_relational_row")
    );
    let disjunction = format!("{header}(assert (or (and (<= Y_0 Y_1)) (and (>= Y_0 Y_1))))\n");
    assert!(admit(&network, &unit_box(), &spec(&disjunction)).is_err());
}

#[test]
fn single_relational_margin_signs_match_the_unsafe_row() {
    // Unsafe Y_0 <= Y_1 is excluded by Y_0 - Y_1 > 0.
    assert_eq!(
        single_relational_margin(&spec(GADGET_SPEC)),
        Some([(0, 1.0), (1, -1.0)])
    );
    // Unsafe Y_0 >= Y_1 is excluded by Y_1 - Y_0 > 0.
    let flipped = GADGET_SPEC.replace("(<= Y_0 Y_1)", "(>= Y_0 Y_1)");
    assert_eq!(
        single_relational_margin(&spec(&flipped)),
        Some([(1, 1.0), (0, -1.0)])
    );
    // A self-comparison is never admitted. The parser itself refuses
    // `(<= Y_0 Y_0)` (`Y_0 - Y_0` has no variable left: "Constraint must
    // reference at least one variable"), so feeding that text through
    // `spec()` would panic before this gate is ever reached. Build the
    // degenerate row by hand so the adapter's own guard is what is tested.
    let mut degenerate = spec(GADGET_SPEC);
    degenerate.output_constraints = vec![OutputConstraint::LessEq(0, 0)];
    degenerate.output_constraint_clauses = vec![vec![OutputConstraint::LessEq(0, 0)]];
    assert_eq!(single_relational_margin(&degenerate), None);
}

#[test]
fn hull_root_deadline_is_a_bounded_slice() {
    let now = Instant::now();
    let cap = Duration::from_secs_f64(HULL_ROOT_MAX_SECS);
    assert_eq!(hull_root_deadline(now, None), now + cap);
    assert_eq!(
        hull_root_deadline(now, Some(Duration::from_secs(100))),
        now + cap
    );
    assert_eq!(
        hull_root_deadline(now, Some(Duration::from_secs(8))),
        now + Duration::from_secs(2)
    );
    assert_eq!(hull_root_deadline(now, Some(Duration::ZERO)), now);
}

#[test]
fn hull_observable_has_a_fixed_schema_on_every_armed_path() {
    const KEYS: [&str; 14] = [
        "armed",
        "mode=",
        "engaged=",
        "verdict=",
        "reason=",
        "unstable=",
        "neg=",
        "pos=",
        "lp_lb=",
        "cert_lb=",
        "cert_route=",
        "lp_status=",
        "sg_iters=",
        "t_ms=",
    ];
    let declined = observable_payload(
        SingleHiddenHullMode::Root,
        None,
        "not_single_hidden",
        Duration::from_millis(3),
    );
    assert!(declined.starts_with("armed mode=root engaged=0 verdict=declined"));
    assert!(declined.contains("reason=not_single_hidden"));
    assert!(declined.contains("cert_lb=na"));

    let problem =
        admit(&gadget_network(0.05), &unit_box(), &spec(GADGET_SPEC)).expect("gadget is admitted");
    let report = hull_root_bound(&problem, far_deadline());
    let engaged = observable_payload(
        SingleHiddenHullMode::Root,
        Some(&report),
        "ok",
        Duration::from_millis(5),
    );
    assert!(engaged.starts_with("armed mode=root engaged=1 verdict=verified"));
    for payload in [&declined, &engaged] {
        for key in KEYS {
            assert!(payload.contains(key), "missing `{key}` in `{payload}`");
        }
    }
}

// ---------------------------------------------------------------------------
// #hull-bab (phase 2): the ReLU-split tree through the lane adapter.
// ---------------------------------------------------------------------------

const ABS_SPEC: &str = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 Y_1))
"#;

/// `Y_0 - Y_1 = C - relu(x) + relu(2x) - relu(-x) + relu(-2x) = C + |x|` on
/// `[-1, 1]`. The two concave neurons' ideal hulls are chords, so the ROOT
/// hull value is `C - 1`; splitting both concave neurons makes every leaf
/// exact with minimum `C` at `x = 0`. `C = 0.25` is UNSAT and only the tree
/// proves it; `C < 0` is SAT at `x = 0`. All constants are exact in f32.
fn abs_network(constant: f32) -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0_f32], [2.0], [-1.0], [-2.0]]),
            Some(arr1(&[0.0_f32, 0.0, 0.0, 0.0])),
        )
        .expect("first affine layer"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[-1.0_f32, 1.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
            Some(arr1(&[constant, 0.0])),
        )
        .expect("second affine layer"),
    ));
    network
}

fn symmetric_box() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite input box")
}

#[test]
fn hull_bab_verifies_a_row_the_root_cannot_through_the_lane_adapter() {
    let network = abs_network(0.25);
    let problem =
        admit(&network, &symmetric_box(), &spec(ABS_SPEC)).expect("abs gadget is admitted");
    let root = hull_root_bound(&problem, far_deadline());
    assert!(
        !root.verified(),
        "the root alone must not prove it: {root:?}"
    );
    assert!(try_single_hidden_hull_root(
        SingleHiddenHullMode::Root,
        &network,
        &symmetric_box(),
        &spec(ABS_SPEC),
        far_deadline(),
        Instant::now(),
    )
    .is_none());

    let result = try_single_hidden_hull_bab(
        &network,
        &symmetric_box(),
        &spec(ABS_SPEC),
        far_deadline(),
        Instant::now(),
        2,
    )
    .expect("the hull tree verifies the abs gadget");
    assert!(matches!(result.result, BabVerificationStatus::Verified));
    assert!(result.domains_explored >= 3, "a proof needs a split");
    assert!(result.domains_verified >= 2 && result.max_depth_reached >= 1);
}

#[test]
fn hull_bab_does_not_verify_a_sat_row() {
    // abs gadget with C < 0: Y_0 - Y_1 = C at x = 0.
    for constant in [-0.25_f32, -0.001] {
        let network = abs_network(constant);
        assert!(
            try_single_hidden_hull_bab(
                &network,
                &symmetric_box(),
                &spec(ABS_SPEC),
                far_deadline(),
                Instant::now(),
                2,
            )
            .is_none(),
            "C={constant}: the tree verified a SAT row"
        );
        let problem =
            admit(&network, &symmetric_box(), &spec(ABS_SPEC)).expect("abs gadget is admitted");
        let report = hull_bab(
            &problem,
            far_deadline(),
            HullBabOptions {
                workers: 1,
                ..HullBabOptions::default()
            },
        );
        assert!(!report.verified(), "{report:?}");
        assert!(
            report.worst.is_some_and(|w| w <= f64::from(constant)),
            "worst open bound exceeds the true minimum: {report:?}"
        );
    }
    // The and-gadget SAT row of phase 1 as well.
    assert!(try_single_hidden_hull_bab(
        &gadget_network(-0.05),
        &unit_box(),
        &spec(GADGET_SPEC),
        far_deadline(),
        Instant::now(),
        2,
    )
    .is_none());
}

#[test]
fn hull_mode_dispatch_is_inert_when_off_and_root_never_branches() {
    let network = abs_network(0.25);
    let remaining = Some(Duration::from_secs(20));
    // Off: no work at all (and no line); the historical lane keeps the row.
    assert!(try_single_hidden_hull(
        SingleHiddenHullMode::Off,
        &network,
        &symmetric_box(),
        &spec(ABS_SPEC),
        remaining,
        Instant::now(),
    )
    .is_none());
    // Root: phase 1 only — this row needs a split, so root mode must decline.
    assert!(try_single_hidden_hull(
        SingleHiddenHullMode::Root,
        &network,
        &symmetric_box(),
        &spec(ABS_SPEC),
        remaining,
        Instant::now(),
    )
    .is_none());
    // Bab: the tree proves it.
    let result = try_single_hidden_hull(
        SingleHiddenHullMode::Bab,
        &network,
        &symmetric_box(),
        &spec(ABS_SPEC),
        remaining,
        Instant::now(),
    )
    .expect("bab mode verifies the abs gadget");
    assert!(matches!(result.result, BabVerificationStatus::Verified));
}

#[test]
fn hull_bab_declines_on_non_single_hidden() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 1.0], [1.0, -1.0]]), None).expect("layer 1"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 0.5], [0.5, 1.0]]), None).expect("layer 2"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, -1.0], [0.0, 1.0]]), None).expect("layer 3"),
    ));
    assert!(try_single_hidden_hull_bab(
        &network,
        &unit_box(),
        &spec(GADGET_SPEC),
        far_deadline(),
        Instant::now(),
        2,
    )
    .is_none());
}

#[test]
fn hull_bab_deadline_is_a_bounded_slice() {
    let within = |a: Instant, b: Instant| {
        let d = if a > b { a - b } else { b - a };
        d <= Duration::from_millis(1)
    };
    let now = Instant::now();
    let cap = Duration::from_secs_f64(HULL_BAB_MAX_SECS);
    assert!(within(hull_bab_deadline(now, None), now + cap));
    assert!(within(
        hull_bab_deadline(now, Some(Duration::from_secs(1_000))),
        now + cap
    ));
    assert!(within(
        hull_bab_deadline(now, Some(Duration::from_secs(10))),
        now + Duration::from_secs(6)
    ));
    assert!(within(hull_bab_deadline(now, Some(Duration::ZERO)), now));
    // The tree's slice is strictly larger than the root's on a scored budget.
    let remaining = Some(Duration::from_secs(18));
    assert!(hull_bab_deadline(now, remaining) > hull_root_deadline(now, remaining));
}

#[test]
fn hull_bab_observable_has_a_fixed_schema_on_every_armed_path() {
    const KEYS: [&str; 21] = [
        "armed",
        "mode=bab",
        "engaged=",
        "verdict=",
        "reason=",
        "stop=",
        "unstable=",
        "neg=",
        "pos=",
        "root_lb=",
        "root_route=",
        "root_lp_lb=",
        "root_lp_status=",
        "nodes=",
        "closed=",
        "open=",
        "worst=",
        "depth=",
        "lp_fail=",
        "workers=",
        "t_ms=",
    ];
    let declined = bab_observable_payload(None, "not_single_hidden", 4, Duration::from_millis(2));
    assert!(declined.starts_with("armed mode=bab engaged=0 verdict=declined"));
    assert!(declined.contains("reason=not_single_hidden"));
    assert!(declined.contains("stop=na") && declined.contains("nodes=na"));
    assert!(declined.contains("worst=na") && declined.contains("workers=4"));

    let problem = admit(&abs_network(0.25), &symmetric_box(), &spec(ABS_SPEC)).expect("admitted");
    let proved = hull_bab(&problem, far_deadline(), HullBabOptions::default());
    let engaged = bab_observable_payload(Some(&proved), "ok", 1, Duration::from_millis(9));
    assert!(engaged.starts_with("armed mode=bab engaged=1 verdict=verified"));
    assert!(engaged.contains("stop=proved") && engaged.contains("open=0"));

    let sat_problem =
        admit(&abs_network(-0.25), &symmetric_box(), &spec(ABS_SPEC)).expect("admitted");
    let open = hull_bab(&sat_problem, far_deadline(), HullBabOptions::default());
    let undecided = bab_observable_payload(Some(&open), "ok", 1, Duration::from_millis(9));
    assert!(undecided.starts_with("armed mode=bab engaged=1 verdict=undecided"));
    assert!(undecided.contains("stop=exhausted"));
    for payload in [&declined, &engaged, &undecided] {
        for key in KEYS {
            assert!(payload.contains(key), "missing `{key}` in `{payload}`");
        }
        assert!(!payload.contains('\n'), "the observable must be ONE line");
    }
}

// ---------------------------------------------------------------------------
// Real-corpus soundness audits (external-vnncomp).
// ---------------------------------------------------------------------------

#[cfg(feature = "external-vnncomp")]
mod corpus {
    use super::super::{admit, try_single_hidden_hull_bab, try_single_hidden_hull_root};
    use ny_mip::single_hidden_hull::bab::{hull_bab, HullBabOptions};
    use ny_mip::single_hidden_hull::{hull_root_bound, HullLpStatus};
    use ny_onnx::vnnlib::VnnLibSpec;
    use ny_propagate::{Network, SingleHiddenHullMode};
    use ny_tensor::BoundedTensor;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    fn workspace() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn corpus_root() -> PathBuf {
        let root = workspace().join("benchmarks/vnncomp2026/benchmarks/safenlp_2024/1.0");
        assert!(
            root.join("onnx/ruarobot/perturbations_0.onnx").is_file(),
            "safenlp corpus missing at {} — symlink benchmarks/vnncomp2026 from the main tree",
            root.display()
        );
        root
    }

    /// Load one row the way production does (`to_propagate_network`, the
    /// lane's outward-rounded f32 box) and run the adapter's own gate.
    fn row(onnx: &str, vnnlib: &str) -> (Network, BoundedTensor, VnnLibSpec) {
        let root = corpus_root();
        let network = ny_onnx::load_onnx(root.join(onnx))
            .expect("load safenlp ONNX")
            .to_propagate_network()
            .expect("sequential safenlp network");
        let spec = ny_onnx::vnnlib::load_vnnlib(root.join(vnnlib)).expect("load safenlp VNNLIB");
        let (lower, upper) = spec.split_input_bounds_f32();
        let n = lower.len();
        let input = BoundedTensor::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), lower).expect("lower"),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), upper).expect("upper"),
        )
        .expect("finite box");
        (network, input, spec)
    }

    fn load_records(file: &str) -> Vec<serde_json::Value> {
        let path = workspace()
            .join("reports/measured-2026/safenlp_undecided_2026-08-30")
            .join(file);
        let text = std::fs::read_to_string(&path).expect("tracked MILP report");
        let json: serde_json::Value = serde_json::from_str(&text).expect("report JSON");
        json["records"].as_array().expect("records array").clone()
    }

    /// cert <= the exact MILP optimum (or incumbent) of `min Y_0 - Y_1` on
    /// every row of both tracked oracle reports (112 closed optima).
    #[test]
    fn hull_cert_never_exceeds_exact_milp_optimum() {
        let mut checked = 0usize;
        let mut optimal = 0usize;
        let mut lp_solved = 0usize;
        let mut root_verified = 0usize;
        for file in ["bound_gap_milp_hardness.json", "capped_rows_long_milp.json"] {
            for record in load_records(file) {
                let onnx = record["onnx"].as_str().expect("onnx");
                let vnnlib = record["vnnlib"].as_str().expect("vnnlib");
                let objective = record["milp_obj"].as_f64().expect("milp_obj");
                let status = record["milp_status"].as_i64().expect("milp_status");
                let (network, input, spec) = row(onnx, vnnlib);
                let problem = admit(&network, &input, &spec)
                    .unwrap_or_else(|reason| panic!("{vnnlib}: declined ({reason})"));
                let report = hull_root_bound(&problem, Instant::now() + Duration::from_secs(60));
                let certified = report.certified.expect("certified");
                assert!(
                    certified <= objective + 1e-6 * (1.0 + objective.abs()),
                    "{vnnlib}: certified {certified} EXCEEDS the MILP value {objective} \
                     (status {status}) — SOUNDNESS"
                );
                checked += 1;
                optimal += usize::from(status == 0);
                lp_solved += usize::from(report.lp_status == HullLpStatus::Solved);
                root_verified += usize::from(report.verified());
            }
        }
        eprintln!(
            "HULL_MILP_AUDIT checked={checked} optimal={optimal} lp_solved={lp_solved} \
             root_verified={root_verified}"
        );
        assert!(
            optimal >= 112,
            "only {optimal} closed MILP optima were audited"
        );
        // Fail closed: a triangle-only run would make this audit near-vacuous.
        assert!(
            lp_solved * 10 >= checked * 9,
            "LP proposer engaged on only {lp_solved}/{checked} rows — the audit is VOID"
        );
    }

    /// Appendix A of `diag6/safenlp.md`: the 30 rows whose offline float
    /// hull-LP root is `> 0`, with that value (4 decimals).
    const ROOT_ROWS: [(&str, f64); 30] = [
        ("ruarobot/hyperrectangle_4888", 0.7083),
        ("ruarobot/hyperrectangle_5258", 0.5078),
        ("ruarobot/hyperrectangle_1113", 0.4368),
        ("ruarobot/hyperrectangle_4175", 0.3895),
        ("ruarobot/hyperrectangle_1184", 0.3810),
        ("ruarobot/hyperrectangle_4990", 0.3597),
        ("medical/hyperrectangle_2883", 0.3353),
        ("medical/hyperrectangle_2707", 0.3075),
        ("ruarobot/hyperrectangle_5606", 0.2763),
        ("ruarobot/hyperrectangle_1598", 0.2691),
        ("ruarobot/hyperrectangle_3514", 0.2564),
        ("ruarobot/hyperrectangle_4743", 0.2155),
        ("medical/hyperrectangle_2027", 0.2027),
        ("ruarobot/hyperrectangle_3241", 0.1706),
        ("ruarobot/hyperrectangle_5409", 0.1687),
        ("medical/hyperrectangle_2068", 0.1649),
        ("ruarobot/hyperrectangle_4420", 0.1605),
        ("ruarobot/hyperrectangle_1207", 0.1496),
        ("medical/hyperrectangle_2780", 0.1472),
        ("ruarobot/hyperrectangle_1157", 0.1455),
        ("ruarobot/hyperrectangle_2791", 0.1130),
        ("ruarobot/hyperrectangle_7233", 0.0762),
        ("medical/hyperrectangle_57", 0.0349),
        ("ruarobot/hyperrectangle_5493", 0.0303),
        ("ruarobot/hyperrectangle_3063", 0.0302),
        ("medical/hyperrectangle_1104", 0.0282),
        ("medical/hyperrectangle_2158", 0.0246),
        ("medical/hyperrectangle_178", 0.0127),
        ("ruarobot/hyperrectangle_5249", 0.0074),
        ("ruarobot/hyperrectangle_455", 0.0004),
    ];

    /// cert <= ny's own LP value, and both sit on the offline hull value.
    /// `lp - cert <= 1e-5` is the dual-mapping / rounding QUALITY check (a
    /// wrong mapping shows up as a gap here long before it shows up as a lost
    /// row); `cert <= offline + 1e-3` would flag a relaxation TIGHTER than the
    /// ideal hull, which cannot be right.
    #[test]
    fn hull_cert_close_to_float_lp_on_root_rows() {
        let mut verified = 0usize;
        for (stem, offline) in ROOT_ROWS {
            let (family, name) = stem.split_once('/').expect("family/name");
            let (network, input, spec) = row(
                &format!("onnx/{family}/perturbations_0.onnx"),
                &format!("vnnlib/{family}/{name}.vnnlib"),
            );
            let problem = admit(&network, &input, &spec).expect("admitted");
            let report = hull_root_bound(&problem, Instant::now() + Duration::from_secs(60));
            assert_eq!(report.lp_status, HullLpStatus::Solved, "{stem}: {report:?}");
            let certified = report.certified.expect("certified");
            let lp = report.lp_value.expect("LP telemetry value");
            assert!(
                lp - certified <= 1e-5 * (1.0 + lp.abs()),
                "{stem}: lp {lp} vs certified {certified} (quality)"
            );
            assert!(
                certified <= offline + 1e-3,
                "{stem}: certified {certified} is TIGHTER than the offline hull {offline}"
            );
            assert!(
                (certified - offline).abs() <= 1e-3,
                "{stem}: certified {certified} drifted from the offline hull {offline}"
            );
            verified += usize::from(report.verified());
        }
        eprintln!("HULL_ROOT_ROWS verified={verified}/30 (expected >= 28)");
    }

    /// The one field-SAT row in the cluster: true minimum -0.0259.
    #[test]
    fn hull_does_not_verify_field_sat_row_1800() {
        let (network, input, spec) = row(
            "onnx/ruarobot/perturbations_0.onnx",
            "vnnlib/ruarobot/hyperrectangle_1800.vnnlib",
        );
        let problem = admit(&network, &input, &spec).expect("admitted");
        let report = hull_root_bound(&problem, Instant::now() + Duration::from_secs(60));
        assert!(!report.verified(), "{report:?}");
        assert!(report.certified.expect("certified") <= -0.025);
        assert!(try_single_hidden_hull_root(
            SingleHiddenHullMode::Root,
            &network,
            &input,
            &spec,
            Instant::now() + Duration::from_secs(60),
            Instant::now(),
        )
        .is_none());
    }

    /// #hull-bab negative control: the one field-SAT row of the gap cluster
    /// (true minimum -0.0259) must never be verified by the tree.
    #[test]
    fn hull_bab_does_not_verify_field_sat_row_1800() {
        let (network, input, spec) = row(
            "onnx/ruarobot/perturbations_0.onnx",
            "vnnlib/ruarobot/hyperrectangle_1800.vnnlib",
        );
        let problem = admit(&network, &input, &spec).expect("admitted");
        let report = hull_bab(
            &problem,
            Instant::now() + Duration::from_secs(10),
            HullBabOptions {
                workers: 4,
                ..HullBabOptions::default()
            },
        );
        eprintln!("HULL_BAB_ROW_1800 {report:?}");
        assert!(!report.verified(), "{report:?}");
        assert!(try_single_hidden_hull_bab(
            &network,
            &input,
            &spec,
            Instant::now() + Duration::from_secs(10),
            Instant::now(),
            4,
        )
        .is_none());
    }

    /// #hull-bab negative control on BANKED SAT rows: each has a validated
    /// counterexample, so the tree must never verify one. The first 40 sat
    /// rows of the tracked bank, in file order.
    #[test]
    fn hull_bab_never_verifies_a_banked_sat_row() {
        let bank = workspace().join("reports/measured-2026/safenlp_2024.csv");
        let text = std::fs::read_to_string(&bank).expect("tracked safenlp bank");
        let rows: Vec<(String, String)> = text
            .lines()
            .filter_map(|line| {
                let fields: Vec<&str> = line.split(',').collect();
                (fields.len() >= 5 && fields[4] == "sat")
                    .then(|| (fields[1].to_string(), fields[2].to_string()))
            })
            .take(40)
            .collect();
        assert_eq!(rows.len(), 40, "expected at least 40 banked sat rows");
        let mut lp_engaged = 0usize;
        for (onnx, vnnlib) in &rows {
            let (network, input, spec) = row(onnx, vnnlib);
            let problem = admit(&network, &input, &spec)
                .unwrap_or_else(|reason| panic!("{vnnlib}: declined ({reason})"));
            let report = hull_bab(
                &problem,
                Instant::now() + Duration::from_secs(3),
                HullBabOptions {
                    workers: 4,
                    ..HullBabOptions::default()
                },
            );
            assert!(
                !report.verified(),
                "{vnnlib}: the hull tree VERIFIED a banked SAT row — SOUNDNESS: {report:?}"
            );
            // `report.nodes` is always >= 1 (the root is counted), so it is no
            // engagement signal; the root LP settling is.
            lp_engaged += usize::from(report.root_lp_status == HullLpStatus::Solved);
        }
        eprintln!(
            "HULL_BAB_SAT_CONTROL rows={} lp_engaged={lp_engaged}",
            rows.len()
        );
        // Fail closed: a control whose LP proposer never settled ran on the
        // triangle bound alone, the weakest bound the tree has — VOID, not a
        // pass.
        assert!(
            lp_engaged * 2 >= rows.len(),
            "the root LP settled on only {lp_engaged}/{} SAT-control rows — the control is VOID",
            rows.len()
        );
    }

    /// #hull-bab box-free capability probe on the 145 audited rows (all
    /// truth-UNSAT with a known MILP value): how many the tree proves in a
    /// short budget, with the soundness invariant that a verified row's MILP
    /// value is positive. Prints `HULL_BAB_MILP_AUDIT ...`.
    #[test]
    fn hull_bab_milp_audit_rows_short_budget() {
        let (mut rows, mut verified, mut nodes) = (0usize, 0usize, 0usize);
        for file in ["bound_gap_milp_hardness.json", "capped_rows_long_milp.json"] {
            for record in load_records(file) {
                let onnx = record["onnx"].as_str().expect("onnx");
                let vnnlib = record["vnnlib"].as_str().expect("vnnlib");
                let objective = record["milp_obj"].as_f64().expect("milp_obj");
                let (network, input, spec) = row(onnx, vnnlib);
                let problem = admit(&network, &input, &spec)
                    .unwrap_or_else(|reason| panic!("{vnnlib}: declined ({reason})"));
                let report = hull_bab(
                    &problem,
                    Instant::now() + Duration::from_secs(4),
                    HullBabOptions {
                        workers: 4,
                        ..HullBabOptions::default()
                    },
                );
                if report.verified() {
                    assert!(
                        objective > 0.0,
                        "{vnnlib}: verified but the MILP value is {objective} — SOUNDNESS"
                    );
                }
                // Every audited row has `milp_obj > 0`, so the check above can
                // never fire. The real invariant: `milp_obj` is scipy's
                // incumbent (`res.fun`), an UPPER bound on the exact minimum
                // over the vnnlib box, which the lane's outward-rounded box
                // contains. The root certificate, and the lowest open bound
                // (open and closed leaves cover the box, and every bound is
                // `<=` its region's minimum), must therefore sit at or below
                // it on EVERY row, verified or not. Same tolerance as the
                // phase-1 audit above.
                let tol = 1e-6 * (1.0 + objective.abs());
                if let Some(root) = report.root_certified {
                    assert!(
                        root <= objective + tol,
                        "{vnnlib}: root certificate {root} EXCEEDS the MILP value {objective} \
                         — SOUNDNESS"
                    );
                }
                if let Some(worst) = report.worst {
                    assert!(
                        worst <= objective + tol,
                        "{vnnlib}: lowest open bound {worst} EXCEEDS the MILP value {objective} \
                         — SOUNDNESS: {report:?}"
                    );
                }
                rows += 1;
                verified += usize::from(report.verified());
                nodes += report.nodes;
            }
        }
        eprintln!("HULL_BAB_MILP_AUDIT rows={rows} verified={verified} nodes={nodes}");
        assert!(rows >= 140, "only {rows} audited rows");
    }
}
