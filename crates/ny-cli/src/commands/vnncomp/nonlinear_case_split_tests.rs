// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #acc-strict-intervals — the typed `nonlinear_vnnlib.case_split` key.
//!
//! * DARK BY DEFAULT: absent everywhere, resolved `false` from every snapshot
//!   shape, armed by no shipped yaml, and a dark lane prints nothing new.
//! * DELIVERY: the key survives the real `deny_unknown_fields` schema and the
//!   frozen `BetaCrownPresetSnapshot` the router reads.
//! * ENGAGEMENT: on a synthetic knife-edge spec the dark lane cannot close,
//!   the armed lane proves `unsat` and its receipt counts the refutations.
//! * THE fp32-REPLAY GUARD: the armed lane refuses a refutation whose output
//!   margin is thinner than the replay band — the mechanism that keeps the
//!   case split from "proving" field-SAT ACC rows in real arithmetic.
//!
//! The real ACC rows 15/38 and the 50-row moat are MEASUREMENTS, not unit
//! tests; the ny-onnx tests pin the evaluator on the rows' own formula.

use std::fs;
use std::path::{Path, PathBuf};

use super::{
    nonlinear_case_split_from_snapshot, nonlinear_fp32_replay_guard_input,
    nonlinear_fp32_replay_guard_output, try_nonlinear_unsat, NonlinearCaseSplitStats,
    NONLINEAR_FP32_REPLAY_MARGIN_REL,
};
use crate::commands::beta_crown::BetaCrownPresetSnapshot;
use crate::preset::PresetConfig;

/// Every configuration year that ships.
const YEARS: &[&str] = &["vnncomp24", "vnncomp25", "vnncomp26"];

/// A 1-in/1-out ONNX `Gemm` (`Y = weight * X + bias`, f32) plus a VNN-LIB 2.0
/// spec with the caller's assertions, in a fresh temp dir.
pub(super) fn write_scalar_gemm_and_spec(
    weight: f32,
    bias: f32,
    asserts: &str,
) -> (tempfile::TempDir, PathBuf, PathBuf) {
    use ny_onnx::onnx_proto::{
        tensor_shape_proto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
        TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use prost::Message as _;

    let value_info = |name: &str, dims: &[i64]| ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1, // FLOAT
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                        })
                        .collect(),
                }),
            }),
        }),
    };
    let f32_tensor = |name: &str, dims: &[i64], values: Vec<f32>| TensorProto {
        dims: dims.to_vec(),
        data_type: 1, // FLOAT
        name: name.to_string(),
        float_data: values,
        ..TensorProto::default()
    };
    let model = ModelProto {
        ir_version: 7,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            node: vec![NodeProto {
                input: vec!["x".to_string(), "w".to_string(), "b".to_string()],
                output: vec!["y".to_string()],
                name: "Gemm_y".to_string(),
                op_type: "Gemm".to_string(),
                domain: String::new(),
                attribute: vec![],
            }],
            name: "scalar_gemm".to_string(),
            initializer: vec![
                f32_tensor("w", &[1, 1], vec![weight]),
                f32_tensor("b", &[1], vec![bias]),
            ],
            input: vec![value_info("x", &[1, 1])],
            output: vec![value_info("y", &[1, 1])],
            ..GraphProto::default()
        }),
        ..ModelProto::default()
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let onnx = dir.path().join("scalar_gemm.onnx");
    fs::write(&onnx, model.encode_to_vec()).expect("write onnx");
    let vnnlib = dir.path().join("spec.vnnlib");
    let spec_text = format!(
        "(vnnlib-version <2.0>)\n\
         (declare-network f\n\
         (declare-input X real [1,1])\n\
         (declare-output Y real [1,1])\n\
         )\n\
         {asserts}\n"
    );
    fs::write(&vnnlib, spec_text).expect("write vnnlib");
    (dir, onnx, vnnlib)
}

/// A miniature ACC: `Y = X - 1` on `X in [0, 2]` with the strict `(> X 0)`.
/// The property is false at every real point — `Y > 0` forces `X*Y > 0`,
/// `Y < 0` forces `X*Y < 0`, and `Y = 0` needs `X <= 0` — but only through
/// strict inequalities: the falsity margin is zero along Y = 0 (X = 1) and at
/// the corner X -> 0+. Closed interval logic can never discharge those cells.
const KNIFE_EDGE_SPEC: &str = "\
(assert (and (>= X[0,0] 0.0) (<= X[0,0] 2.0)))
(assert (> X[0,0] 0.0))
(assert (>= (* X[0,0] X[0,0]) 0.0))
(assert (or (and (> Y[0,0] 0.0) (<= (* X[0,0] Y[0,0]) 0.0)) \
(and (< Y[0,0] 0.0) (>= (* X[0,0] Y[0,0]) 0.0)) \
(and (== Y[0,0] 0.0) (<= X[0,0] 0.0))))";

#[test]
fn nonlinear_case_split_key_is_dark_by_default() {
    assert_eq!(PresetConfig::default().nonlinear_vnnlib.case_split, None);
    assert!(!nonlinear_case_split_from_snapshot(None));
    let loaded = BetaCrownPresetSnapshot::Loaded(Box::default());
    assert!(!nonlinear_case_split_from_snapshot(Some(&loaded)));
    let invalid = BetaCrownPresetSnapshot::Invalid("semantically invalid".to_string());
    assert!(!nonlinear_case_split_from_snapshot(Some(&invalid)));
    let mut explicit_off = PresetConfig::default();
    explicit_off.nonlinear_vnnlib.case_split = Some(false);
    let explicit_off = BetaCrownPresetSnapshot::Loaded(Box::new(explicit_off));
    assert!(!nonlinear_case_split_from_snapshot(Some(&explicit_off)));
    // A dark lane prints no receipt at all, so its stderr is unchanged.
    assert_eq!(NonlinearCaseSplitStats::new(false).render(), None);
}

#[test]
fn nonlinear_case_split_key_round_trips_through_the_real_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let armed = dir.path().join("armed.yaml");
    fs::write(&armed, "nonlinear_vnnlib:\n  case_split: true\n").expect("write yaml");
    let preset = crate::preset::load_preset(&armed).expect("the key is in the schema");
    assert_eq!(preset.nonlinear_vnnlib.case_split, Some(true));
    let snapshot = BetaCrownPresetSnapshot::load(&armed);
    assert!(
        snapshot.invalid_error().is_none(),
        "an armed-only preset must pass semantic validation: {:?}",
        snapshot.invalid_error()
    );
    assert!(nonlinear_case_split_from_snapshot(Some(&snapshot)));

    // A misspelled key must fail loudly at load, never silently stay dark.
    let typo = dir.path().join("typo.yaml");
    fs::write(&typo, "nonlinear_vnnlib:\n  case_splt: true\n").expect("write yaml");
    assert!(crate::preset::load_preset(&typo).is_err());
}

/// PROMOTION IS NOT ARMING: no shipped yaml sets the key, and no shipped yaml
/// exists for the ACC category at all. The brief is explicit that an ACC yaml
/// would move all 50 rows off AUTO DEFAULTS, which is verdict-affecting beyond
/// this lever; whichever arming route the measurement commit takes (a yaml or
/// a flipped code default) must edit this pin in the same commit.
#[test]
fn no_shipped_preset_arms_the_nonlinear_case_split() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut armed = Vec::new();
    for year in YEARS {
        let acc = root
            .join(year)
            .join("adaptive_cruise_control_non_linear_2026.yaml");
        assert!(
            !acc.exists(),
            "{} exists: an ACC preset moves all 50 rows off AUTO DEFAULTS and must \
             arrive with its own measurement",
            acc.display()
        );
        let Ok(entries) = fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            scanned += 1;
            if preset.nonlinear_vnnlib.case_split.is_some() {
                armed.push(path.display().to_string());
            }
        }
    }
    assert!(
        scanned >= 20,
        "expected the shipped preset corpus, found {scanned} files — a silent read failure \
         would make this pin vacuous"
    );
    assert!(
        armed.is_empty(),
        "nonlinear_vnnlib.case_split is SET in a shipped preset: {armed:?}"
    );
}

#[test]
fn nonlinear_case_split_receipt_has_a_fixed_schema() {
    let stats = NonlinearCaseSplitStats {
        armed: true,
        attempted: 7,
        refuted: 5,
        undecided: 1,
        declined: 1,
        pieces: 40,
        pivots: Some("x0@{0,2} y0@{0}".to_string()),
    };
    assert_eq!(
        stats.render().as_deref(),
        Some(
            "#nonlinear-vnnlib: case-split refuted 5 cell(s) (pivots x0@{0,2} y0@{0}) armed=1 \
             attempted=7 undecided=1 declined=1 pieces=40 fp32_guard=1"
        )
    );
    // Armed but never reached a parse: still one line, with the null visible.
    assert_eq!(
        NonlinearCaseSplitStats::new(true).render().as_deref(),
        Some(
            "#nonlinear-vnnlib: case-split refuted 0 cell(s) (pivots unparsed) armed=1 \
             attempted=0 undecided=0 declined=0 pieces=0 fp32_guard=1"
        )
    );
}

/// ENGAGEMENT, end to end through the real lane: the dark lane cannot close
/// the knife-edge spec (it bisects into the f64 grid and gives up), while the
/// armed lane refutes the whole declared box and records it.
#[test]
fn armed_case_split_proves_a_knife_edge_spec_the_dark_lane_cannot() {
    let (_dir, onnx, vnnlib) = write_scalar_gemm_and_spec(1.0, -1.0, KNIFE_EDGE_SPEC);

    let mut dark = NonlinearCaseSplitStats::new(false);
    let dark_verdict = try_nonlinear_unsat(&onnx, &vnnlib, 10, None, &mut dark);
    assert_ne!(
        dark_verdict,
        Some(true),
        "the dark lane must not close a zero-margin knife edge; if it now does, this \
         fixture no longer demonstrates the case split"
    );
    assert_eq!((dark.attempted, dark.refuted, dark.pieces), (0, 0, 0));
    assert_eq!(dark.render(), None);

    let mut armed = NonlinearCaseSplitStats::new(true);
    let armed_verdict = try_nonlinear_unsat(&onnx, &vnnlib, 10, None, &mut armed);
    assert_eq!(
        armed_verdict,
        Some(true),
        "the armed lane must refute every cell of the knife-edge spec: {armed:?}"
    );
    assert!(armed.refuted >= 1, "{armed:?}");
    assert!(armed.attempted >= armed.refuted, "{armed:?}");
    assert_eq!(armed.pivots.as_deref(), Some("x0@{0,2} y0@{0}"));
    let line = armed.render().expect("armed receipt");
    assert!(
        line.starts_with("#nonlinear-vnnlib: case-split refuted "),
        "{line}"
    );
}

/// THE fp32-REPLAY GUARD. `Y = f32(100.001) = 100.0009994506836` everywhere,
/// so `(> Y 100.001)` is false in real arithmetic by 5.5e-7 — exactly the ACC
/// clamp margin that the scored fp32 replay crosses on the 16 field-SAT rows.
/// The dark lane (today's behaviour) proves this spec `unsat` in real
/// arithmetic; the armed lane must refuse, because a refutation that thin is
/// not robust to the replay the scorer actually runs.
#[test]
fn armed_lane_refuses_an_output_margin_thinner_than_the_fp32_replay_band() {
    let spec = "\
(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))
(assert (>= (* X[0,0] X[0,0]) 0.0))
(assert (> Y[0,0] 100.001))";
    let (_dir, onnx, vnnlib) = write_scalar_gemm_and_spec(0.0, 100.001_f32, spec);

    let mut dark = NonlinearCaseSplitStats::new(false);
    assert_eq!(
        try_nonlinear_unsat(&onnx, &vnnlib, 10, None, &mut dark),
        Some(true),
        "fixture: the real-arithmetic lane refutes the 5.5e-7 margin today"
    );

    let mut armed = NonlinearCaseSplitStats::new(true);
    assert_ne!(
        try_nonlinear_unsat(&onnx, &vnnlib, 2, None, &mut armed),
        Some(true),
        "the armed lane must not rest a refutation on an fp32-thin output margin"
    );
    assert!(
        armed.attempted >= 1,
        "the split must have been tried: {armed:?}"
    );
    assert_eq!(armed.refuted, 0, "{armed:?}");
}

#[test]
fn fp32_replay_guard_input_hull_contains_every_f32_cast_of_the_cell() {
    let cells = [
        (0.0, 2.0),
        (0.1, 0.30000000000000004),
        (-1e-300, 1e-300),
        (28.78, 28.78_f64.next_up()),
        (-40.0, -39.99999999),
        (1.0 / 3.0, 2.0 / 3.0),
    ];
    let lower: Vec<f64> = cells.iter().map(|&(lo, _)| lo).collect();
    let upper: Vec<f64> = cells.iter().map(|&(_, hi)| hi).collect();
    let (hull_lo, hull_hi) =
        nonlinear_fp32_replay_guard_input(&lower, &upper).expect("finite cells");
    for (index, &(lo, hi)) in cells.iter().enumerate() {
        assert!(
            hull_lo[index] <= lo && hi <= hull_hi[index],
            "hull must contain the cell"
        );
        assert_eq!(
            f64::from(hull_lo[index] as f32),
            hull_lo[index],
            "f32-representable"
        );
        assert_eq!(
            f64::from(hull_hi[index] as f32),
            hull_hi[index],
            "f32-representable"
        );
        for step in 0..=64_u32 {
            let x = lo + (hi - lo) * f64::from(step) / 64.0;
            let x = x.clamp(lo, hi);
            let cast = f64::from(x as f32);
            assert!(
                hull_lo[index] <= cast && cast <= hull_hi[index],
                "f32({x}) = {cast} escapes the hull [{}, {}]",
                hull_lo[index],
                hull_hi[index]
            );
        }
    }
    assert!(nonlinear_fp32_replay_guard_input(&[f64::NAN], &[1.0]).is_none());
}

#[test]
fn fp32_replay_guard_output_only_widens_by_at_least_the_band() {
    let mut lower = vec![-1.0, 100.0009994506836, -99.997, 0.0];
    let mut upper = vec![1.0, 100.0009994506836, -99.997, 0.0];
    let (before_lo, before_hi) = (lower.clone(), upper.clone());
    nonlinear_fp32_replay_guard_output(&mut lower, &mut upper).expect("finite");
    for index in 0..lower.len() {
        let scale = before_lo[index].abs().max(before_hi[index].abs()).max(1.0);
        let band = scale * NONLINEAR_FP32_REPLAY_MARGIN_REL;
        assert!(
            lower[index] <= before_lo[index] - band,
            "lower side {index}"
        );
        assert!(
            upper[index] >= before_hi[index] + band,
            "upper side {index}"
        );
    }
    // The band must swallow the ACC clamp margin and keep the ACC unsat margin.
    assert!(
        upper[1] > 100.001,
        "the 5.5e-7 clamp margin must not survive the band"
    );
    assert!(
        upper[2] < -99.997 + 0.003 && lower[2] > -100.0,
        "the 3e-3 margin must survive"
    );
    let (mut bad_lo, mut bad_hi) = (vec![f64::INFINITY], vec![f64::INFINITY]);
    assert!(nonlinear_fp32_replay_guard_output(&mut bad_lo, &mut bad_hi).is_none());
}
