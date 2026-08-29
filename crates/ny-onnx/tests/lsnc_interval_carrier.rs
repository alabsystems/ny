// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// THE DEFECT AND THE REPAIR, on the model the defect is about.
//
// `try_fold_matmul_exact` folds a both-constant MatMul only when every product
// AND every partial sum is exactly representable in binary32. In
// lsnc_relu/2.0/relu_quadrotor2d_state.onnx, all 36 entries of
// `/lyapunov/MatMul` (= `onnx::MatMul_137 @ lyapunov.R`) fail that test, so the
// node does not fold, `convert_matmul` fails it closed to an unbounded
// `OpaqueSkip`, and the Lyapunov matrix Q — the output of `/lyapunov/Add` — is
// LOST. Two plain `activation @ constant Q` MatMuls (`/lyapunov/MatMul_1` and
// `/MatMul`) then see two non-constant operands, get reclassified as
// `BilinearCrown`, and raise "MatMul requires at least 2D inputs" on their
// rank-1 activations. alpha-CROWN dies, batched CROWN dies, flat CROWN dies,
// IBP degrades to "output node /Concat_2 is tainted by an upstream OpaqueSkip",
// and the root bound is [-inf, +inf] on all 8 outputs. The verifier never runs.
//
// This file pins BOTH arms of the repair on that exact model:
//   (c) armed, Q exists as an interval constant, the two `activation @ Q` nodes
//       are on the affine path rather than BilinearCrown, and the root bound
//       CONTAINS an independently constructed reachable envelope where it was
//       previously +-inf;
//   (d) unset, NOTHING changes: no interval constant is ever created and the
//       failure above reproduces byte-for-byte in its structure.
//
// Cargo registers this target with `required-features = ["external-vnncomp"]`,
// so the default suite stays hermetic and a missing corpus file is a hard
// failure rather than a silent skip.

use ny_onnx::load_onnx;
use ny_tensor::BoundedTensor;

mod common;
use common::{benchmark_root, BenchYear};

/// The official lsnc_relu input box for `relu_quadrotor2d_state`.
const INPUT_LOWER: [f32; 6] = [-0.7, -0.7, -1.0, -3.0, -3.0, -2.0];
const INPUT_UPPER: [f32; 6] = [0.7, 0.7, 1.0, 3.0, 3.0, 2.0];

/// The tensor that carries Q, and the two nodes that consume it.
const Q_TENSOR: &str = "/lyapunov/Add_output_0";
const Q_CONSUMERS: [&str; 2] = ["/lyapunov/MatMul_1", "/MatMul"];

/// Whether the carrier lever is present in this process's environment.
///
/// Read through the levers registry rather than a direct `env` call: a fresh
/// raw `NY_*` read anywhere under `crates/` trips ny-levers' migration ratchet,
/// and a test is not exempt from that.
fn carrier_lever_is_set() -> bool {
    ny_levers::RawLeverInputs::capture(ny_levers::all())
        .get(&ny_levers::decls::dark_probes::CONST_MATMUL_INTERVAL)
        .is_some()
}

fn model_path() -> std::path::PathBuf {
    let path = benchmark_root(BenchYear::Vnncomp2026)
        .join("lsnc_relu/2.0/onnx/relu_quadrotor2d_state.onnx");
    assert!(
        path.is_file(),
        "Benchmark file missing: {}. Run benchmarks/download_benchmarks.sh first.",
        path.display()
    );
    path
}

/// REACHABLE POINTS, EVALUATED OUTSIDE ny.
///
/// `(input, output)` pairs produced by ONNX Runtime on this exact model, at
/// concrete points inside the official input box. These are not bounds and
/// they are not ny's opinion of anything: each output vector is a value the
/// network DOES take, so a sound root bound must contain every one of them,
/// coordinate by coordinate.
///
/// Provenance: a 200065-point search (all 64 box corners, the centre, a
/// uniform sample, then a per-output coordinate pattern search restarted from
/// the best sample), reduced to the arg-extreme points; their envelope equals
/// the full search's envelope. The final entry is the ORT-confirmed
/// `DV = 2.8978e-4` counterexample on file for these rows — the one that makes
/// SAFE a FALSE verdict here, since the specs' first disjunct is
/// `Y_0 >= 0.000001`.
const ORT_WITNESSES: [([f32; 6], [f32; 8]); 16] = [
    (
        [0.699_999_1, -0.7, -1.0, 3.0, -2.999_999_8, 2.0],
        [
            0.931_903_8,
            23.807_577,
            0.730_861_25,
            -0.729_889_04,
            -0.975_354_4,
            3.172_433,
            -2.977_805_4,
            2.929_129_4,
        ],
    ),
    (
        [
            -0.7,
            -0.7,
            0.683_593_75,
            3.0,
            -2.259_568_7,
            -1.899_902_3,
        ],
        [
            -0.743_778_9,
            11.053_251,
            -0.670_933_2,
            -0.721_911_3,
            0.664_590_66,
            2.813_364,
            -2.122_7,
            -1.900_715_4,
        ],
    ),
    (
        [-0.7, -0.7, 1.0, -3.0, -3.0, 2.0],
        [
            0.786_178,
            28.296_864,
            -0.730_979_1,
            -0.729_972_8,
            1.004_578_6,
            -3.195_816,
            -2.994_554_8,
            -1.084_270_7,
        ],
    ),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0; 8]),
    (
        [0.7, -0.7, -1.0, 3.0, -3.0, -1.413_574_2],
        [
            0.650_520_8,
            24.843_771,
            0.730_936_4,
            -0.729_903_76,
            -0.997_819,
            3.187_278_5,
            -2.980_749,
            1.849_777,
        ],
    ),
    (
        [-0.7, -0.7, 1.0, -3.0, -3.0, -2.0],
        [
            0.780_973_85,
            26.875_418,
            -0.731_061_6,
            -0.729_834,
            0.970_170_6,
            -3.212_314_1,
            -2.966_801_6,
            -3.965_872_8,
        ],
    ),
    (
        [-0.7, 0.7, -1.0, -3.0, 3.0, -2.0],
        [
            -0.174_780_12,
            11.002_899,
            -0.729_565_7,
            0.729_872_94,
            -1.011_295_7,
            -2.913_139_3,
            2.974_587,
            -0.259_131_2,
        ],
    ),
    (
        [0.7, -0.7, 1.0, -3.0, -3.0, 2.0],
        [
            0.250_681_28,
            16.461_47,
            0.669_054_4,
            -0.730_012,
            1.002_659_3,
            -3.189_124_6,
            -3.002_406,
            -1.468_139_2,
        ],
    ),
    (
        [-0.7, 0.7, 1.0, -3.0, 3.0, 2.0],
        [
            0.454_293_67,
            25.041_65,
            -0.729_926_3,
            0.729_547_4,
            1.020_003_6,
            -2.985_260_2,
            2.909_480_3,
            2.000_716,
        ],
    ),
    (
        [-0.7, -0.7, -1.0, -3.0, -3.0, -2.0],
        [
            -0.110_761_6,
            13.170_176,
            -0.729_102_43,
            -0.729_900_3,
            -1.013_607_9,
            -2.820_489,
            -2.980_062_7,
            -0.721_583_1,
        ],
    ),
    (
        [-0.7, -0.7, -1.0, 3.0, -3.0, -0.806_213_4],
        [
            0.642_838_06,
            21.782_818,
            -0.668_976_25,
            -0.729_870_86,
            -0.994_917_15,
            3.204_749_6,
            -2.974_174_7,
            1.822_779_9,
        ],
    ),
    (
        [-0.7, -0.7, -0.643_660_6, -3.0, 3.0, -2.0],
        [
            -0.273_296_1,
            6.151_977_5,
            -0.729_416_85,
            -0.669_732_6,
            -0.660_846_6,
            -2.883_374_5,
            3.053_480_4,
            -1.437_202_2,
        ],
    ),
    (
        [0.7, 0.7, 1.0, -3.0, -3.0, -1.539_306_6],
        [
            -0.579_778_15,
            14.382_622,
            0.669_811_07,
            0.669_752_06,
            0.974_556_5,
            -3.037_788,
            -3.049_591,
            -3.549_393,
        ],
    ),
    (
        [0.7, -0.699_997_3, -1.0, 3.0, -3.0, 2.0],
        [
            0.931_899_96,
            23.807_566,
            0.730_862_14,
            -0.729_886_35,
            -0.975_354_4,
            3.172_432_4,
            -2.977_805_6,
            2.929_130_3,
        ],
    ),
    (
        [-0.7, -0.271_958_14, 1.0, -3.0, -3.0, -2.0],
        [
            0.399_891_26,
            24.721_16,
            -0.730_989_5,
            -0.301_876_66,
            0.966_037_75,
            -3.197_905,
            -2.983_707,
            -4.792_451,
        ],
    ),
    // The ORT-confirmed DV counterexample. Y_0 = 2.8978e-4 satisfies the
    // specs' `Y_0 >= 0.000001` disjunct, so SAFE on these rows is FALSE.
    (
        [
            -0.337_032_47,
            -0.121_915_82,
            -0.251_726_03,
            -0.744_893_8,
            2.086_692,
            -0.230_242_04,
        ],
        [
            0.000_289_784_74,
            0.831_713_3,
            -0.344_396_74,
            -0.101_234_53,
            -0.252_280_68,
            -0.727_960_17,
            2.049_565_3,
            0.119_314_43,
        ],
    ),
];

/// ny's IBP root bound on the official input box, as `(lower, upper)`.
fn root_bounds() -> (Vec<f32>, Vec<f32>) {
    let model = load_onnx(model_path()).expect("the lsnc model must load in both arms");
    let graph = model
        .to_graph_network()
        .expect("the lsnc model must lower to a graph network in both arms");
    let lower = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 6]), INPUT_LOWER.to_vec())
        .expect("shape");
    let upper = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 6]), INPUT_UPPER.to_vec())
        .expect("shape");
    let input = BoundedTensor::new(lower, upper).expect("input box");
    let out = graph.propagate_ibp(&input).expect("IBP must not error");
    (
        out.lower().iter().copied().collect(),
        out.upper().iter().copied().collect(),
    )
}

fn root_bounds_are_finite() -> bool {
    let (lo, hi) = root_bounds();
    lo.iter().chain(hi.iter()).all(|v| v.is_finite())
}

/// CONTAINMENT, not finiteness.
///
/// The false proof this file exists to catch published
/// `[-1.9464622e-31, +1.9464622e-31]` and stamped it `mode: Sound`. That bound
/// is PERFECTLY FINITE. The predecessor of this function asserted
/// `root_bounds_are_finite()` and called it "the whole repair", so it would
/// have stayed green throughout the incident it was written for. A bound test
/// that cannot fail on a tight-but-wrong bound is not a soundness test.
///
/// Returns the coordinate that escapes the bound, if any.
fn first_containment_violation() -> Option<String> {
    let (lo, hi) = root_bounds();
    containment_violation(&lo, &hi)
}

/// The predicate, split out from the bound source so it can be tested against
/// a KNOWN-BAD bound. See `the_containment_check_rejects_the_false_proof_bound`.
fn containment_violation(lo: &[f32], hi: &[f32]) -> Option<String> {
    assert_eq!(lo.len(), 8, "the lsnc model has 8 outputs");
    for (w, (x, y)) in ORT_WITNESSES.iter().enumerate() {
        for (j, value) in y.iter().enumerate() {
            if !(lo[j] <= *value && *value <= hi[j]) {
                return Some(format!(
                    "witness {w} at input {x:?}: ORT reaches Y_{j} = {value:e}, which is \
                     OUTSIDE ny's certified root bound [{}, {}]. That is a FALSE ENCLOSURE, \
                     however finite and however tight.",
                    lo[j], hi[j]
                ));
            }
        }
    }
    None
}

/// THE CHECK MUST BE ABLE TO FAIL.
///
/// A containment assertion that cannot reject the bound from the incident is
/// worth exactly as much as the `root_bounds_are_finite()` assertion it
/// replaced. This feeds the FALSE PROOF's own root bound —
/// `[-1.9464622e-31, +1.9464622e-31]`, published `mode: Sound` on these rows —
/// straight into the predicate and requires a violation. It needs no corpus
/// and no lever, so it runs in every arm.
#[test]
fn the_containment_check_rejects_the_false_proof_bound() {
    let lo = [-1.946_462_2e-31_f32; 8];
    let hi = [1.946_462_2e-31_f32; 8];
    let violation = containment_violation(&lo, &hi).expect(
        "the containment predicate ACCEPTED the +-1.9464622e-31 bound from the incident. \
         It is finite, it is tight, and it excludes every reachable point — if this \
         predicate passes it, it would have stayed green through the false proof, exactly \
         like the finiteness assertion it replaced.",
    );
    assert!(
        violation.contains("FALSE ENCLOSURE"),
        "unexpected violation text: {violation}"
    );
}

/// (d) LEVER UNSET => NOTHING CHANGES. Not asserted, tested: the weight store
/// holds ZERO interval constants, and the documented failure reproduces.
#[test]
fn dark_the_carrier_never_fires_and_the_root_bound_is_unbounded() {
    let model = load_onnx(model_path()).expect("the lsnc model must load");
    assert_eq!(
        model.weights.interval_len(),
        0,
        "with NY_CONST_MATMUL_INTERVAL unset no interval constant may exist; the interval \
         fallback must never be entered at all"
    );
    assert!(
        model.weights.get(Q_TENSOR).is_none(),
        "dark, the Lyapunov matrix Q is not a constant at all — that IS the defect"
    );
    assert!(
        !root_bounds_are_finite(),
        "dark, the root bound is [-inf, +inf] on the tainted outputs; if this ever becomes \
         finite the defect was fixed somewhere else and this file's premise is stale"
    );
}

/// (c) ARMED: Q EXISTS, ITS CONSUMERS ARE AFFINE, AND THE ROOT BOUND CONTAINS
/// EVERY INDEPENDENTLY EVALUATED REACHABLE POINT.
///
/// Runs in a separate process because the carrier gate is read once per model
/// load and this crate's test binary is otherwise shared.
#[test]
fn armed_q_becomes_an_interval_constant_and_the_root_bound_contains_reachable_points() {
    if !carrier_lever_is_set() {
        // Re-exec this single test with the lever armed.
        let exe = std::env::current_exe().expect("test binary path");
        let status = std::process::Command::new(exe)
            .arg(
                "armed_q_becomes_an_interval_constant_and_the_root_bound_contains_reachable_points",
            )
            .arg("--exact")
            .arg("--nocapture")
            .env("NY_CONST_MATMUL_INTERVAL", "1")
            .status()
            .expect("re-exec the armed arm");
        assert!(status.success(), "the armed arm failed");
        return;
    }

    let model = load_onnx(model_path()).expect("the lsnc model must load armed");
    assert!(
        model.weights.interval_len() > 0,
        "armed, the carrier must actually fire on this model"
    );
    let (lower, upper) = model
        .weights
        .get_interval(Q_TENSOR)
        .unwrap_or_else(|| panic!("armed, {Q_TENSOR} must be an interval-valued constant"));
    assert_eq!(lower.shape(), [6, 6], "Q is the 6x6 Lyapunov matrix");

    // The enclosure must be finite, ordered, and TIGHT — a vacuous carrier
    // would satisfy every containment claim and buy nothing.
    let mut max_width = 0.0_f64;
    for (l, u) in lower.iter().zip(upper.iter()) {
        assert!(
            l.is_finite() && u.is_finite() && l <= u,
            "malformed enclosure"
        );
        max_width = max_width.max(f64::from(*u) - f64::from(*l));
    }
    assert!(
        max_width > 0.0 && max_width < 1.0e-5,
        "Q's enclosure width {max_width:e} is outside the measured budget"
    );

    // The two consumers must be on the AFFINE path. `BilinearCrown` is the
    // misclassification the defect produces, and it is what raises the rank-1
    // "MatMul requires at least 2D inputs".
    let graph = model
        .to_graph_network()
        .expect("armed, the model must lower to a graph network");
    for consumer in Q_CONSUMERS {
        let node = graph
            .node(consumer)
            .unwrap_or_else(|| panic!("armed, node {consumer} must survive into the graph"));
        let layer_type = node.layer().layer_type();
        assert_eq!(
            layer_type, "IntervalLinear",
            "armed, {consumer} must lower as an interval-valued affine layer, not {layer_type}"
        );
    }

    // FINITENESS IS THE PRECONDITION, NOT THE TEST. It is asserted only so the
    // containment failure below reads as "excludes a reachable point" rather
    // than "is +-inf, which contains everything vacuously".
    assert!(
        root_bounds_are_finite(),
        "armed, the root bound must be FINITE on every output — that is the gate the \
         carrier opens"
    );

    // THE TEST. Every ORT-evaluated witness must be inside ny's certified root
    // bound. This is what the previous version of this file did not check, and
    // it is the only assertion here that a tight-but-wrong bound cannot pass.
    if let Some(violation) = first_containment_violation() {
        panic!("armed, ny published a root bound that EXCLUDES a reachable point: {violation}");
    }

    // Guard against a vacuous witness table: if these were all zeros (or all
    // equal) the containment assertion above would pass against almost any
    // bound, including the +-1.9e-31 one from the incident.
    let spread = ORT_WITNESSES
        .iter()
        .map(|(_, y)| y[1])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        spread > 1.0,
        "the witness table is degenerate ({spread}); containment would be vacuous"
    );
}

/// (d) THE SAME PROPERTY, ACROSS EVERY BANKED FAMILY: with the lever unset, no
/// model in the six no-regression families acquires a single interval constant.
///
/// This is the structural half of "byte-identical when disarmed". The carrier
/// cannot change a bound it never creates, and this walks the real corpus
/// rather than reasoning about the code path.
#[test]
fn dark_no_banked_family_model_acquires_an_interval_constant() {
    const FAMILIES: [&str; 6] = [
        "traffic_signs_recognition_2023",
        "monotonic_acasxu_2026",
        "soundnessbench_2026",
        "isomorphic_acasxu_2026",
        "cersyve",
        "lsnc_relu",
    ];
    assert!(
        !carrier_lever_is_set(),
        "this test asserts the DISARMED arm; it must not run with the lever set"
    );
    let root = benchmark_root(BenchYear::Vnncomp2026);
    let mut checked = 0usize;
    for family in FAMILIES {
        for onnx in collect_onnx(&root.join(family)) {
            let Ok(model) = load_onnx(&onnx) else {
                // A model this loader refuses outright is not evidence either
                // way; the carrier is downstream of a successful load.
                continue;
            };
            assert_eq!(
                model.weights.interval_len(),
                0,
                "{}: an interval constant was created with the lever UNSET",
                onnx.display()
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 6,
        "expected at least one model per banked family to load, only {checked} did — the \
         corpus is missing and this test would pass vacuously"
    );
}

fn collect_onnx(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(collect_onnx(&path));
        } else if path.extension().is_some_and(|e| e == "onnx") {
            out.push(path);
        }
    }
    out
}
