// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-attack-twin — the search core on analytic oracles (no model, no ORT),
//! and the lever's arrival through the scored-run projection.

use std::time::{Duration, Instant};

use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};

use super::super::preset_delivered_config_value;
use super::*;

/// `y0 = x2 + 0.1 * x0²`, `y1 = x0 + x1` over `[-1, 1]³`. The trusted forward
/// and the "twin" gradient are the same analytic function here; the tests pin
/// the SEARCH, not the twin.
struct Analytic {
    /// When set, the direction oracle lies: it returns the NEGATED gradient.
    adversarial_direction: bool,
}

impl TwinPgdOracle for Analytic {
    fn trusted_outputs(&mut self, x: &[f32]) -> Option<Vec<f64>> {
        let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        Some(vec![x[2] + 0.1 * x[0] * x[0], x[0] + x[1]])
    }

    fn twin_gradient(&mut self, x: &[f32], row: &[f32]) -> Option<Vec<f32>> {
        // Jacobian rows: dy0 = [0.2 x0, 0, 1], dy1 = [1, 1, 0].
        let grad = vec![row[0] * 0.2 * x[0] + row[1], row[1], row[0]];
        Some(if self.adversarial_direction {
            grad.into_iter().map(|g| -g).collect()
        } else {
            grad
        })
    }
}

/// The lsnc shape in miniature: a band on `y1` distributed into every
/// disjunct, plus one distinctive target per disjunct. Clause 0 is feasible
/// only ON the `x2 = +1` face region (`y0 >= 1.05` needs `x2 >= 0.95`) with
/// `x0 >= 0.71`; clause 1 is infeasible.
fn banded_spec(target: f64) -> VnnLibSpec {
    let band = [
        OutputConstraint::LessEqConst(1, 0.6),
        OutputConstraint::GreaterEqConst(1, 0.5),
    ];
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 3;
    spec.num_outputs = 2;
    spec.input_bounds = vec![(-1.0, 1.0); 3];
    spec.is_disjunction = true;
    spec.output_constraint_clauses = vec![
        vec![
            OutputConstraint::GreaterEqConst(0, target),
            band[0].clone(),
            band[1].clone(),
        ],
        vec![
            OutputConstraint::LessEqConst(0, -5.0),
            band[0].clone(),
            band[1].clone(),
        ],
    ];
    spec
}

fn search_box() -> (Vec<f32>, Vec<f32>, Vec<Option<f64>>) {
    (vec![-1.0; 3], vec![1.0; 3], vec![None; 3])
}

#[test]
fn twin_pgd_search_finds_a_planted_face_counterexample() {
    let spec = banded_spec(1.05);
    let (lo, hi, pin) = search_box();
    let mut oracle = Analytic {
        adversarial_direction: false,
    };
    let deadline = Instant::now() + Duration::from_secs(20);
    let (found, stats) = twin_pgd_search(
        &mut oracle,
        &spec,
        &lo,
        &hi,
        &pin,
        deadline,
        0x1234,
        0,
        None,
    );
    let x = found.unwrap_or_else(|| panic!("the planted counterexample must be found: {stats:?}"));
    let y = oracle.trusted_outputs(&x).expect("analytic forward");
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    assert!(
        property_violated_f64(&spec, &x64, &y),
        "a returned point must violate the property on the TRUSTED outputs: x={x:?} y={y:?}"
    );
    assert_eq!(stats.found_clause, Some(0), "only clause 0 is satisfiable");
    assert_eq!(stats.stop, "found");
    assert!(
        x[2] >= 0.95,
        "the witness sits on the x2 face region: {x:?}"
    );
}

/// SOUNDNESS OF THE SEARCH CONTRACT: whatever the direction oracle says, the
/// search returns nothing unless the TRUSTED outputs violate the property. On
/// an infeasible property it runs (the counters prove it was not VOID) and
/// returns `None`, even with a lying direction oracle.
#[test]
fn twin_pgd_search_returns_nothing_on_an_infeasible_property() {
    // y0 <= 1.1 on the whole box, so `y0 >= 10` is unsatisfiable.
    let spec = banded_spec(10.0);
    let (lo, hi, pin) = search_box();
    for adversarial_direction in [false, true] {
        let mut oracle = Analytic {
            adversarial_direction,
        };
        let deadline = Instant::now() + Duration::from_millis(300);
        let (found, stats) =
            twin_pgd_search(&mut oracle, &spec, &lo, &hi, &pin, deadline, 7, 1, None);
        assert!(
            found.is_none(),
            "an infeasible property can never yield a point (adversarial={adversarial_direction})"
        );
        assert!(stats.steps > 0, "the search must actually run: {stats:?}");
        assert!(stats.ort_evals >= stats.steps);
        assert_eq!(stats.found_clause, None);
    }
}

#[test]
fn twin_pgd_search_spends_nothing_past_its_deadline() {
    let spec = banded_spec(1.05);
    let (lo, hi, pin) = search_box();
    let mut oracle = Analytic {
        adversarial_direction: false,
    };
    let (found, stats) = twin_pgd_search(
        &mut oracle,
        &spec,
        &lo,
        &hi,
        &pin,
        Instant::now(),
        3,
        0,
        None,
    );
    assert!(found.is_none());
    assert_eq!(stats.restarts, 0);
    assert_eq!(stats.ort_evals, 0);
}

/// A raised stop flag (another worker already confirmed a violation) ends the
/// search before its next restart, and a finding worker raises it.
#[test]
fn twin_pgd_search_honours_and_raises_the_stop_flag() {
    let spec = banded_spec(1.05);
    let (lo, hi, pin) = search_box();
    let mut oracle = Analytic {
        adversarial_direction: false,
    };
    let raised = AtomicBool::new(true);
    let deadline = Instant::now() + Duration::from_secs(20);
    let (found, stats) = twin_pgd_search(
        &mut oracle,
        &spec,
        &lo,
        &hi,
        &pin,
        deadline,
        5,
        0,
        Some(&raised),
    );
    assert!(found.is_none());
    assert_eq!(stats.restarts, 0);
    assert_eq!(stats.stop, "stopped");

    let lowered = AtomicBool::new(false);
    let (found, _) = twin_pgd_search(
        &mut oracle,
        &spec,
        &lo,
        &hi,
        &pin,
        deadline,
        5,
        2,
        Some(&lowered),
    );
    assert!(found.is_some(), "the planted counterexample must be found");
    assert!(
        lowered.load(Ordering::Relaxed),
        "a finding worker must raise the flag for the others"
    );
}

/// The distributed band is shared by every disjunct, so it is not what a
/// face-biased start should follow: only the disjunct's own target is.
#[test]
fn search_clauses_treat_a_distributed_conjunct_as_shared() {
    let spec = banded_spec(1.05);
    let (lo, hi, _) = search_box();
    let clauses = search_clauses(&spec, &lo, &hi);
    assert_eq!(clauses.len(), 2);
    assert_eq!(
        clauses[0].distinctive,
        vec![OutputConstraint::GreaterEqConst(0, 1.05)]
    );
    assert_eq!(
        clauses[1].distinctive,
        vec![OutputConstraint::LessEqConst(0, -5.0)]
    );
    assert_eq!(clauses[0].constraints.len(), 3);
}

/// A face-biased start snaps the coordinate the distinctive target is most
/// sensitive to (`dy0/dx2 = 1` over width 2 beats `0.2 |x0|` over width 2) onto
/// the face its gradient points at — the lsnc `X_5 = +2` move.
#[test]
fn face_restart_snaps_the_distinctive_targets_dominant_coordinate() {
    let spec = banded_spec(1.05);
    let (lo, hi, _) = search_box();
    let clauses = search_clauses(&spec, &lo, &hi);
    let mut oracle = Analytic {
        adversarial_direction: false,
    };
    let mut rng = SimpleRng::new(99);
    let mut stats = TwinPgdStats::default();
    let x = restart_start(&mut oracle, &clauses[0], 2, true, &mut rng, &mut stats);
    assert_eq!(x[2], 1.0, "x2 must be snapped onto its upper face: {x:?}");
    assert_eq!(stats.face_restarts, 1);
    let mut rng = SimpleRng::new(99);
    let uniform = restart_start(&mut oracle, &clauses[0], 2, false, &mut rng, &mut stats);
    assert!(uniform.iter().all(|v| (-1.0..=1.0).contains(v)));
    assert_eq!(stats.face_restarts, 1, "a uniform start snaps nothing");
}

fn projected_policy(preset: &crate::preset::PresetConfig) -> PointFoldedTwinPolicy {
    let serialized = serde_json::to_value(preset).expect("preset serializes");
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    let levers = ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        preset_delivered_config_value(&serialized, decl)
    })
    .expect("the projected preset must match its declarations");
    PointFoldedTwinPolicy::from_levers(&levers).expect("the twin policy must resolve")
}

/// Absent the key, the lane is dark on the scored projection.
#[test]
fn point_folded_twin_policy_is_dark_without_the_key() {
    let policy = projected_policy(&crate::preset::PresetConfig::default());
    assert!(!policy.armed);
    assert_eq!(policy.source, ny_levers::Source::Default);
}

/// REACHABILITY: the typed key arrives through the SAME projection the scored
/// run uses (`preset_delivered_config_value` over the serialized preset, with
/// NO environment), with `Source::Config` — never through a legacy env read.
#[test]
fn point_folded_twin_typed_key_arms_through_the_scored_projection() {
    let mut preset = crate::preset::PresetConfig::default();
    preset.attack.point_folded_twin = Some(true);
    let policy = projected_policy(&preset);
    assert!(policy.armed);
    assert_eq!(policy.source, ny_levers::Source::Config);
    assert_ne!(policy.source, ny_levers::Source::LegacyEnv);
}

fn shipped_configs() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

/// REACHABILITY ON THE SHIPPED FILE: the real `lsnc_relu` preset, loaded and
/// projected exactly as a scored run projects it (no environment at all), arms
/// the lane with `Source::Config`. No `vnncomp26` shadow exists to override it.
#[test]
fn the_shipped_lsnc_preset_arms_the_twin_through_the_scored_projection() {
    let path = shipped_configs().join("vnncomp25/lsnc_relu.yaml");
    let preset = crate::preset::load_preset(&path)
        .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
    assert_eq!(preset.attack.point_folded_twin, Some(true));
    let policy = projected_policy(&preset);
    assert!(
        policy.armed,
        "the shipped lsnc_relu preset must arm the twin lane"
    );
    assert_eq!(policy.source, ny_levers::Source::Config);
    assert!(
        !shipped_configs().join("vnncomp26/lsnc_relu.yaml").exists(),
        "a vnncomp26 shadow would decide the 2026 rows instead of this preset"
    );
}

/// SCOPE: the key is armed for exactly the category whose rows motivate it.
/// Another family arming it needs its own measurement first.
#[test]
fn only_the_lsnc_relu_preset_arms_the_point_folded_twin() {
    let mut armed = Vec::new();
    for year in std::fs::read_dir(shipped_configs()).expect("list configs/") {
        let year = year.expect("configs entry").path();
        let Some(year_name) = year.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !year_name.starts_with("vnncomp") || !year.is_dir() {
            continue;
        }
        let year_name = year_name.to_string();
        for entry in std::fs::read_dir(&year).expect("list preset dir") {
            let path = entry.expect("preset entry").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
                continue;
            }
            let preset = crate::preset::load_preset(&path)
                .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
            if preset.attack.point_folded_twin.is_some() {
                let file = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .expect("preset file name");
                armed.push(format!("{year_name}/{file}"));
            }
        }
    }
    armed.sort();
    assert_eq!(armed, ["vnncomp25/lsnc_relu.yaml"]);
}

/// OBSERVABILITY: the terminal `stop=` token never flattens a VOID worker into
/// `deadline`. A lane whose workers all lost the gradient reports that reason,
/// disagreeing workers report `mixed`, and only a found point reports `found`.
#[test]
fn lane_stop_does_not_flatten_a_void_worker_into_deadline() {
    assert_eq!(lane_stop(true, &["deadline", "stopped"]), "found");
    assert_eq!(lane_stop(false, &["deadline", "deadline"]), "deadline");
    assert_eq!(
        lane_stop(false, &["gradient-unavailable", "gradient-unavailable"]),
        "gradient-unavailable"
    );
    assert_eq!(
        lane_stop(false, &["deadline", "worker-setup-failed"]),
        "mixed"
    );
    assert_eq!(lane_stop(false, &[]), "no-worker");
}
