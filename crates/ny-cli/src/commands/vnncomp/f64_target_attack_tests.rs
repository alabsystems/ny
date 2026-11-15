// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #sbench-f64-attack tests: the dark-by-default pins (policy, caps, shipped
//! presets, preset validation), the numeric helpers, and the soundness
//! contract of the float64 target-margin repair — a candidate leaves the
//! repair ONLY when the trusted forward confirms it at zero tolerance AND its
//! exact-f64 margin is non-negative, whatever the trusted forward's noise.

use std::path::Path;
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array1, Array2};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer, PointVjpF64Plan};

use super::super::{
    constraint_margin, ort_active_set_phase_budget, ort_active_set_phase_budget_with_caps,
    ort_active_set_seed_budget, ort_active_set_seed_budget_with_caps, property_violated_f64,
    OrtActiveSetCaps, ORT_ACTIVE_SET_PER_SEED_BUDGET, ORT_ACTIVE_SET_TOTAL_BUDGET,
};
use super::lp_maximin::f64_lp_maximin_lift;
use super::{
    build_plan, f32_ulp_at, f64_lift_lineage, f64_target_margin_repair, lift_rng_seed,
    margin_output_row, repair_conjuncts, target_delta_for, F64AttackTelemetry,
    PostBabActiveSetPolicy,
};
use crate::preset::{AttackPreset, F64LiftStep, PresetConfig};

fn preset_with(attack: AttackPreset) -> PresetConfig {
    PresetConfig {
        attack,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Dark by default.
// ---------------------------------------------------------------------------

/// No preset, or a preset that sets neither key, is the historical tail: the
/// constant caps, no f64 stage, and a telemetry value that prints nothing.
#[test]
fn postbab_active_set_policy_is_dark_by_default() {
    for policy in [
        PostBabActiveSetPolicy::from_preset(None),
        PostBabActiveSetPolicy::from_preset(Some(&PresetConfig::default())),
    ] {
        assert_eq!(policy, PostBabActiveSetPolicy::HISTORICAL);
        assert_eq!(policy.caps(), OrtActiveSetCaps::HISTORICAL);
        assert_eq!(policy.f64_target_ulps(), None);
        assert!(!policy.is_configured());
        assert!(
            !F64AttackTelemetry::new(policy).armed,
            "a disarmed policy must never print the engagement line"
        );
    }
    assert_eq!(
        OrtActiveSetCaps::HISTORICAL,
        OrtActiveSetCaps {
            per_seed: ORT_ACTIVE_SET_PER_SEED_BUDGET,
            total: ORT_ACTIVE_SET_TOTAL_BUDGET,
            iteration_scale: 1,
        }
    );
}

/// The scored path now calls the `_with_caps` forms; under the historical
/// caps they must be bit-identical to the pinned constant forms on a dense
/// grid, and a configured 6 s total must reproduce the constants exactly.
#[test]
fn historical_caps_reproduce_the_active_set_budgets_bit_for_bit() {
    assert_eq!(
        OrtActiveSetCaps::from_total(Duration::from_secs(6)),
        OrtActiveSetCaps::HISTORICAL
    );
    for millis in (0..=20_000u64).step_by(125) {
        let remaining = Duration::from_millis(millis);
        for seed_count in 0..=4usize {
            assert_eq!(
                ort_active_set_phase_budget_with_caps(
                    remaining,
                    seed_count,
                    OrtActiveSetCaps::HISTORICAL
                ),
                ort_active_set_phase_budget(remaining, seed_count),
                "phase remaining={millis}ms seeds={seed_count}"
            );
            for seed_index in 0..=seed_count {
                assert_eq!(
                    ort_active_set_seed_budget_with_caps(
                        remaining,
                        seed_index,
                        seed_count,
                        OrtActiveSetCaps::HISTORICAL
                    ),
                    ort_active_set_seed_budget(remaining, seed_index, seed_count),
                    "seed remaining={millis}ms index={seed_index} seeds={seed_count}"
                );
            }
        }
    }
}

/// A configured total halves into the per-lineage cap and scales the Newton
/// iteration caps by `ceil(total / 6 s)`; the phase rule itself is unchanged.
#[test]
fn configured_active_set_repair_secs_scale_the_caps() {
    let twelve = PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
        active_set_repair_secs: Some(12.0),
        ..Default::default()
    })));
    assert!(twelve.is_configured());
    assert_eq!(
        twelve.f64_target_ulps(),
        None,
        "the secs key alone never arms f64"
    );
    let caps = twelve.caps();
    assert_eq!(caps.per_seed, Duration::from_secs(6));
    assert_eq!(caps.total, Duration::from_secs(12));
    assert_eq!(caps.iteration_scale, 2);
    // B=20 s, three lineages: first keeps min(20, 6); extra = min(20-6-3, 12-6).
    assert_eq!(
        ort_active_set_phase_budget_with_caps(Duration::from_secs(20), 3, caps),
        Duration::from_secs(12)
    );
    let seven = OrtActiveSetCaps::from_total(Duration::from_secs(7));
    assert_eq!(seven.per_seed, Duration::from_millis(3500));
    assert_eq!(seven.iteration_scale, 2);
}

/// Values the preset validator refuses are ignored by the resolver as well
/// (defence in depth for hand-built presets).
#[test]
fn out_of_range_values_resolve_to_the_historical_policy() {
    for (secs, ulps) in [
        (Some(f64::NAN), None),
        (Some(0.25), None),
        (Some(1.0e9), None),
        (None, Some(0)),
        (None, Some(crate::preset::F64_TARGET_MARGIN_ULPS_MAX + 1)),
    ] {
        let policy = PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
            active_set_repair_secs: secs,
            f64_target_margin_ulps: ulps,
            ..Default::default()
        })));
        assert_eq!(
            policy,
            PostBabActiveSetPolicy::HISTORICAL,
            "secs={secs:?} ulps={ulps:?}"
        );
    }
}

/// Both keys parse from yaml, validate in range, and are REFUSED out of range
/// (an invalid snapshot is a sound `unknown`, never a silently clamped run).
#[test]
fn preset_validation_accepts_in_range_and_refuses_out_of_range_f64_attack_keys() {
    let armed: PresetConfig = serde_yaml::from_str(
        "attack:\n  f64_target_margin_ulps: 8\n  active_set_repair_secs: 12.0\n",
    )
    .expect("the armed keys parse");
    assert_eq!(armed.attack.f64_target_margin_ulps, Some(8));
    assert_eq!(armed.attack.active_set_repair_secs, Some(12.0));
    crate::preset::validate_preset(&armed).expect("in-range keys validate");
    let policy = PostBabActiveSetPolicy::from_preset(Some(&armed));
    assert_eq!(policy.f64_target_ulps(), Some(8));
    assert!(F64AttackTelemetry::new(policy).armed);

    for (yaml, key) in [
        (
            "attack:\n  f64_target_margin_ulps: 0\n",
            "attack.f64_target_margin_ulps",
        ),
        (
            "attack:\n  f64_target_margin_ulps: 70000\n",
            "attack.f64_target_margin_ulps",
        ),
        (
            "attack:\n  active_set_repair_secs: 0.5\n",
            "attack.active_set_repair_secs",
        ),
        (
            "attack:\n  active_set_repair_secs: 601.0\n",
            "attack.active_set_repair_secs",
        ),
        (
            "attack:\n  active_set_repair_secs: .nan\n",
            "attack.active_set_repair_secs",
        ),
    ] {
        let preset: PresetConfig = serde_yaml::from_str(yaml).expect("parses");
        let err = crate::preset::validate_preset(&preset)
            .expect_err("an out-of-range f64-attack key must be refused");
        // Non-vacuity: the refusal must be THIS key's, not an unrelated one.
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(key),
            "the refusal of {yaml:?} must name {key}; got {rendered}"
        );
    }
}

/// DARK: no shipped preset arms either key. Arming happens only in the commit
/// that carries its measurement.
#[test]
fn no_shipped_preset_arms_the_f64_attack_keys() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut armed = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let Ok(entries) = std::fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            scanned += 1;
            if preset.attack.f64_target_margin_ulps.is_some()
                || preset.attack.active_set_repair_secs.is_some()
            {
                armed.push(path.display().to_string());
            }
        }
    }
    assert!(
        scanned > 10,
        "the scan must actually read the shipped presets"
    );
    assert!(
        armed.is_empty(),
        "shipped presets arm #sbench-f64-attack: {armed:?}"
    );
}

/// The engagement line carries the fixed schema the brief names.
#[test]
fn f64_attack_telemetry_line_has_the_fixed_schema() {
    let policy = PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
        f64_target_margin_ulps: Some(4),
        ..Default::default()
    })));
    let mut telemetry = F64AttackTelemetry::new(policy);
    let line = telemetry.line();
    for field in [
        "[f64-attack] armed=1 engaged=0 reached=0 target_ulps=4",
        " best_margin_f64=",
        " best_margin_ort=",
        " target_delta=",
        " reason=not_reached",
    ] {
        assert!(line.contains(field), "missing {field:?} in {line}");
    }
    telemetry.reach(3);
    telemetry.decline("plan_unavailable");
    let line = telemetry.line();
    assert!(line.contains("reached=1") && line.contains("seeds=3"));
    assert!(line.contains("reason=plan_unavailable"));
    // Silence the drop-time print in the test output.
    telemetry.armed = false;
}

// ---------------------------------------------------------------------------
// Numeric helpers.
// ---------------------------------------------------------------------------

#[test]
fn f32_ulp_at_is_the_f32_spacing_at_the_output_magnitude() {
    assert_eq!(f32_ulp_at(1.0), 2f64.powi(-23));
    assert_eq!(f32_ulp_at(-3.0), 2f64.powi(-22));
    // soundnessbench property_054's largest threshold magnitude.
    assert_eq!(f32_ulp_at(8.228478475), 2f64.powi(-20));
    assert_eq!(f32_ulp_at(0.0), 2f64.powi(-149));
    assert_eq!(f32_ulp_at(f64::NAN), 2f64.powi(104));
    assert_eq!(f32_ulp_at(1e300), 2f64.powi(104));
    for v in [0.3_f64, 1.5, 7.99, 123.456] {
        let v32 = v as f32;
        let next = f32::from_bits(v32.to_bits() + 1);
        assert_eq!(f32_ulp_at(v), f64::from(next) - f64::from(v32), "v={v}");
    }
}

/// `margin_output_row` is exactly `d constraint_margin / d Y` for every shape.
#[test]
fn margin_output_row_is_the_gradient_of_constraint_margin() {
    let outputs = [0.25_f64, -1.5, 3.0];
    let constraints = [
        OutputConstraint::LessEq(0, 1),
        OutputConstraint::LessThan(2, 0),
        OutputConstraint::GreaterEq(1, 2),
        OutputConstraint::GreaterThan(0, 2),
        OutputConstraint::LessEqConst(1, 0.5),
        OutputConstraint::LessThanConst(2, -4.0),
        OutputConstraint::GreaterEqConst(0, 1.0),
        OutputConstraint::GreaterThanConst(2, 2.5),
    ];
    for c in &constraints {
        let row = margin_output_row(c, outputs.len()).expect("row");
        let base = constraint_margin(c, &outputs);
        for j in 0..outputs.len() {
            let mut bumped = outputs;
            bumped[j] += 1.0;
            assert_eq!(constraint_margin(c, &bumped) - base, row[j], "{c:?} j={j}");
        }
    }
    assert!(margin_output_row(&OutputConstraint::LessEq(0, 7), 3).is_none());
    let rows = vec![margin_output_row(&OutputConstraint::GreaterEqConst(1, 0.0), 3).unwrap()];
    assert_eq!(target_delta_for(3, &outputs, &rows), 3.0 * 2f64.powi(-23));
}

// ---------------------------------------------------------------------------
// Soundness of the repair on a planted thin counterexample slab.
// ---------------------------------------------------------------------------

/// `Y = ReLU([x0 + x1, 1 - x0 - x1])` on `x in [0, 0.5]^2`: the property
/// `Y_0 >= 0.5 - p AND Y_1 >= 0.5 - p` holds exactly on the slab
/// `|x0 + x1 - 0.5| <= p`, whose peak margin is `p` — thinner than a few f32
/// ULPs for the `p` used below, like SoundnessBench's planted regions.
fn slab_network() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "hidden",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 1.0], [-1.0, -1.0]]),
                Some(arr1(&[0.0_f32, 1.0])),
            )
            .expect("hidden linear"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["hidden".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]),
                Some(arr1(&[0.0_f32, 0.0])),
            )
            .expect("output linear"),
        ),
        vec!["relu".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn slab_spec(threshold: &str) -> VnnLibSpec {
    ny_onnx::vnnlib::parse_vnnlib(&format!(
        "(declare-const X_0 Real)\n(declare-const X_1 Real)\n\
         (declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
         (assert (>= X_0 0.0))\n(assert (<= X_0 0.5))\n\
         (assert (>= X_1 0.0))\n(assert (<= X_1 0.5))\n\
         (assert (or (and (>= Y_0 {threshold}) (>= Y_1 {threshold}))))\n"
    ))
    .expect("slab property parses")
}

const BOX_LO: [f32; 2] = [0.0, 0.0];
const BOX_HI: [f32; 2] = [0.5, 0.5];
const NO_PIN: [Option<f64>; 2] = [None, None];

fn slab_plan(graph: &GraphNetwork) -> PointVjpF64Plan {
    let plan = build_plan(
        graph,
        &[2],
        &BOX_LO,
        &BOX_HI,
        Instant::now() + Duration::from_secs(30),
    )
    .expect("the slab net has a chain twin");
    assert_eq!(plan.kind(), "chain");
    plan
}

fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// Deterministic per-point, per-output +-amplitude "rounding noise": the
/// trusted forward's disagreement with exact arithmetic, as ORT's f32
/// accumulation produces it.
fn noise(x: &[f32], output: usize, amplitude: f64) -> f64 {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15 ^ (output as u64);
    for v in x {
        h ^= u64::from(v.to_bits());
        h = h.wrapping_mul(0x0100_0000_01B3);
        h ^= h >> 29;
    }
    if h & 1 == 0 {
        amplitude
    } else {
        -amplitude
    }
}

fn noisy_trusted(plan: &PointVjpF64Plan, x: &[f32], amplitude: f64) -> Option<Vec<f64>> {
    let (y, _) = plan.forward(&widen(x))?;
    Some(
        y.iter()
            .enumerate()
            .map(|(j, &v)| v + noise(x, j, amplitude))
            .collect(),
    )
}

fn exact_min_margin(plan: &PointVjpF64Plan, spec: &VnnLibSpec, x: &[f32]) -> f64 {
    let (y, _) = plan.forward(&widen(x)).expect("exact forward");
    repair_conjuncts(spec)
        .expect("the slab property is one conjunction")
        .iter()
        .map(|c| constraint_margin(c, &y))
        .fold(f64::INFINITY, f64::min)
}

/// With trusted noise of 3 f32 ULPs at |Y| ~ 0.5 and a 2e-6 slab, a target of
/// 8 ULPs lands a point the NOISY trusted forward confirms — and that point
/// satisfies both acceptance predicates.
#[test]
fn f64_target_repair_crosses_a_slab_thinner_than_the_trusted_noise_floor() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.499998");
    let amplitude = 3.0 * f32_ulp_at(0.5);
    let outcome = f64_target_margin_repair(
        &plan,
        |x: &[f32]| noisy_trusted(&plan, x, amplitude),
        &spec,
        &BOX_LO,
        &BOX_HI,
        &NO_PIN,
        &[0.245, 0.245],
        8,
        Instant::now() + Duration::from_secs(20),
        64,
    )
    .expect("the near seed is in scope");
    assert!(outcome.initial_margin_f64 < 0.0);
    assert!(
        outcome.target_delta > amplitude,
        "delta must clear the noise floor"
    );
    let found = outcome
        .violation
        .clone()
        .unwrap_or_else(|| panic!("no candidate: {outcome:?}"));
    let trusted = noisy_trusted(&plan, &found, amplitude).expect("trusted outputs");
    assert!(property_violated_f64(&spec, &widen(&found), &trusted));
    assert!(exact_min_margin(&plan, &spec, &found) >= 0.0);
    assert!(outcome.best_margin_ort >= 0.0);
}

/// SOUNDNESS, sampled: over many seeds, noise levels (including noise larger
/// than the slab itself) and targets, EVERY emitted candidate is confirmed by
/// the trusted forward at zero tolerance and has a non-negative exact-f64
/// margin. The repair never hands on a point either predicate rejects.
#[test]
fn f64_target_repair_never_emits_a_point_either_predicate_rejects() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.499998");
    let mut emitted = 0usize;
    let mut state: u64 = 0x5EED_F64A_77AC_0001;
    let mut next_unit = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for trial in 0..60 {
        let sum = 0.48 + 0.04 * next_unit();
        let split = next_unit();
        let seed = [
            (sum * split).min(0.5) as f32,
            (sum * (1.0 - split)).min(0.5) as f32,
        ];
        let amplitude = [0.0, 1.0, 3.0, 40.0][trial % 4] * f32_ulp_at(0.5);
        let ulps = [1u32, 4, 16][trial % 3];
        let Some(outcome) = f64_target_margin_repair(
            &plan,
            |x: &[f32]| noisy_trusted(&plan, x, amplitude),
            &spec,
            &BOX_LO,
            &BOX_HI,
            &NO_PIN,
            &seed,
            ulps,
            Instant::now() + Duration::from_secs(5),
            64,
        ) else {
            continue;
        };
        if let Some(found) = outcome.violation {
            emitted += 1;
            let trusted = noisy_trusted(&plan, &found, amplitude).expect("trusted outputs");
            assert!(
                property_violated_f64(&spec, &widen(&found), &trusted),
                "trial {trial}: emitted a point the trusted forward rejects"
            );
            assert!(
                exact_min_margin(&plan, &spec, &found) >= 0.0,
                "trial {trial}: emitted a point exact f64 refutes"
            );
            assert!(found.iter().zip(&BOX_LO).all(|(v, lo)| v >= lo));
            assert!(found.iter().zip(&BOX_HI).all(|(v, hi)| v <= hi));
        }
    }
    assert!(
        emitted > 0,
        "the sampled trials must exercise the emitting path"
    );
}

/// SOUNDNESS: on an exactly-UNSAT slab (peak margin -5e-7) the repair emits
/// nothing and never even consults a trusted forward that LIES "violated"
/// everywhere — it only asks where the exact margin is already >= 0.
#[test]
fn f64_target_repair_never_consults_or_emits_on_an_exactly_unsat_slab() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.5000005");
    let mut consulted = 0usize;
    let outcome = f64_target_margin_repair(
        &plan,
        |_x: &[f32]| {
            consulted += 1;
            // Adversarial: always report a gross violation.
            Some(vec![10.0, 10.0])
        },
        &spec,
        &BOX_LO,
        &BOX_HI,
        &NO_PIN,
        &[0.245, 0.245],
        4,
        Instant::now() + Duration::from_secs(5),
        64,
    )
    .expect("the near seed is in scope");
    assert!(outcome.violation.is_none(), "{outcome:?}");
    assert!(outcome.best_margin_f64 < 0.0, "the slab is exactly empty");
    assert_eq!(outcome.trusted_evals, 0);
    assert_eq!(consulted, 0);
}

/// A trusted forward that never confirms yields no candidate, however far the
/// exact margin clears the target (the target escalates, then stops).
#[test]
fn f64_target_repair_emits_nothing_the_trusted_forward_never_confirms() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.49");
    let outcome = f64_target_margin_repair(
        &plan,
        |_x: &[f32]| Some(vec![-10.0, -10.0]),
        &spec,
        &BOX_LO,
        &BOX_HI,
        &NO_PIN,
        &[0.24, 0.24],
        4,
        Instant::now() + Duration::from_secs(5),
        64,
    )
    .expect("the near seed is in scope");
    assert!(outcome.violation.is_none(), "{outcome:?}");
    assert!(
        outcome.trusted_evals >= 1,
        "the exact margin must have cleared zero"
    );
    assert!(outcome.best_margin_f64 >= 0.0);
}

/// Out of scope shapes decline instead of guessing: a far seed, a wrong
/// arity, and a zero target.
#[test]
fn f64_target_repair_declines_out_of_scope_lineages() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.499998");
    let run = |seed: &[f32], ulps: u32| {
        f64_target_margin_repair(
            &plan,
            |x: &[f32]| noisy_trusted(&plan, x, 0.0),
            &spec,
            &BOX_LO,
            &BOX_HI,
            &NO_PIN,
            seed,
            ulps,
            Instant::now() + Duration::from_secs(5),
            64,
        )
    };
    assert!(
        run(&[0.0, 0.0], 4).is_none(),
        "margin -0.5 is not a near miss"
    );
    assert!(run(&[0.245], 4).is_none(), "arity mismatch");
    assert!(run(&[0.245, 0.245], 0).is_none(), "zero target");
}

// ---------------------------------------------------------------------------
// #sbench-f64-maximin: dark by default.
// ---------------------------------------------------------------------------

fn lp_maximin_policy(hops: usize, secs: Option<f64>) -> PostBabActiveSetPolicy {
    PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
        f64_target_margin_ulps: Some(8),
        f64_lift_step: Some(F64LiftStep::LpMaximin),
        f64_basin_hops: Some(hops),
        active_set_repair_secs: secs,
        ..Default::default()
    })))
}

/// DEFAULT = C8, BYTE FOR BYTE: with only C8's keys (or an explicit
/// `min_norm`), the policy, the ORT phase caps and the telemetry are C8's, and
/// the lineage dispatch returns exactly C8's outcome (field for field, via
/// its Debug rendering) with no lift record.
#[test]
fn f64_lift_default_is_byte_identical_to_c8() {
    let c8_only = PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
        f64_target_margin_ulps: Some(8),
        active_set_repair_secs: Some(45.0),
        ..Default::default()
    })));
    let explicit = PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
        f64_target_margin_ulps: Some(8),
        active_set_repair_secs: Some(45.0),
        f64_lift_step: Some(F64LiftStep::MinNorm),
        f64_basin_hops: Some(0),
        ..Default::default()
    })));
    assert_eq!(c8_only, explicit);
    for policy in [c8_only, PostBabActiveSetPolicy::HISTORICAL] {
        assert_eq!(policy.lift_step(), F64LiftStep::MinNorm);
        assert_eq!(policy.basin_hops(), 0);
        assert_eq!(policy.ort_phase_caps(), policy.caps());
        let mut telemetry = F64AttackTelemetry::new(policy);
        assert!(!telemetry.lift_armed, "min_norm never prints [f64-lift]");
        telemetry.armed = false;
    }
    assert_eq!(
        PostBabActiveSetPolicy::HISTORICAL.ort_phase_caps(),
        OrtActiveSetCaps::HISTORICAL
    );

    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.499998");
    let amplitude = 3.0 * f32_ulp_at(0.5);
    for (seed, ulps) in [
        ([0.245f32, 0.245], 8u32),
        ([0.26, 0.23], 4),
        ([0.24, 0.25], 16),
    ] {
        let direct = f64_target_margin_repair(
            &plan,
            |x: &[f32]| noisy_trusted(&plan, x, amplitude),
            &spec,
            &BOX_LO,
            &BOX_HI,
            &NO_PIN,
            &seed,
            ulps,
            Instant::now() + Duration::from_secs(20),
            64,
        );
        let dispatched = f64_lift_lineage(
            c8_only,
            &plan,
            |x: &[f32]| noisy_trusted(&plan, x, amplitude),
            &spec,
            &BOX_LO,
            &BOX_HI,
            &NO_PIN,
            &seed,
            ulps,
            Instant::now() + Duration::from_secs(20),
            64,
        );
        let (outcome, lift) = dispatched.expect("in scope");
        assert!(lift.is_none(), "min_norm has no lift record");
        assert_eq!(
            format!("{:?}", direct.expect("in scope")),
            format!("{outcome:?}"),
            "seed {seed:?}: the default dispatch must be C8's call"
        );
    }
}

/// `lp_maximin` moves the configured total onto the f64 lift: the ORT phase
/// keeps the historical caps, the f64 phase gets `active_set_repair_secs`,
/// and the hops come through; the telemetry arms the `[f64-lift]` line.
#[test]
fn lp_maximin_ort_phase_keeps_historical_caps() {
    let policy = lp_maximin_policy(60, Some(45.0));
    assert_eq!(policy.lift_step(), F64LiftStep::LpMaximin);
    assert_eq!(policy.basin_hops(), 60);
    assert_eq!(policy.ort_phase_caps(), OrtActiveSetCaps::HISTORICAL);
    assert_eq!(
        policy.caps(),
        OrtActiveSetCaps::from_total(Duration::from_secs(45))
    );
    assert_eq!(policy.f64_budget_source(), "attack.active_set_repair_secs");
    let historical_budget = lp_maximin_policy(0, None);
    assert_eq!(historical_budget.caps(), OrtActiveSetCaps::HISTORICAL);
    assert_eq!(historical_budget.f64_budget_source(), "historical_caps");
    let mut telemetry = F64AttackTelemetry::new(policy);
    assert!(telemetry.armed && telemetry.lift_armed);
    telemetry.armed = false;
    telemetry.lift_armed = false;
}

/// Both keys parse from yaml and validate together with the stage key; the
/// validator REFUSES a step key without the stage, hops without
/// `lp_maximin`, and hops past the cap; an unknown step fails to parse; and
/// the resolver ignores the step keys when the stage is unarmed (defence in
/// depth for hand-built presets).
#[test]
fn f64_lift_keys_parse_validate_and_refuse() {
    let armed: PresetConfig = serde_yaml::from_str(
        "attack:\n  f64_target_margin_ulps: 16\n  active_set_repair_secs: 50.0\n  \
         f64_lift_step: lp_maximin\n  f64_basin_hops: 60\n",
    )
    .expect("the armed keys parse");
    assert_eq!(armed.attack.f64_lift_step, Some(F64LiftStep::LpMaximin));
    assert_eq!(armed.attack.f64_basin_hops, Some(60));
    crate::preset::validate_preset(&armed).expect("a coherent arming validates");
    let policy = PostBabActiveSetPolicy::from_preset(Some(&armed));
    assert_eq!(policy.lift_step(), F64LiftStep::LpMaximin);
    assert_eq!(policy.basin_hops(), 60);

    let min_norm: PresetConfig = serde_yaml::from_str(
        "attack:\n  f64_target_margin_ulps: 16\n  f64_lift_step: min_norm\n  f64_basin_hops: 0\n",
    )
    .expect("parses");
    crate::preset::validate_preset(&min_norm).expect("an explicit default validates");

    for (yaml, key) in [
        ("attack:\n  f64_lift_step: lp_maximin\n", "attack.f64_lift_step"),
        ("attack:\n  f64_lift_step: min_norm\n", "attack.f64_lift_step"),
        (
            "attack:\n  f64_target_margin_ulps: 16\n  f64_basin_hops: 5\n",
            "attack.f64_basin_hops",
        ),
        (
            "attack:\n  f64_target_margin_ulps: 16\n  f64_lift_step: min_norm\n  f64_basin_hops: 5\n",
            "attack.f64_basin_hops",
        ),
        (
            "attack:\n  f64_target_margin_ulps: 16\n  f64_lift_step: lp_maximin\n  \
             f64_basin_hops: 100001\n",
            "attack.f64_basin_hops",
        ),
    ] {
        let preset: PresetConfig = serde_yaml::from_str(yaml).expect("parses");
        let err = crate::preset::validate_preset(&preset)
            .expect_err("an incoherent f64-lift arming must be refused");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(key),
            "the refusal of {yaml:?} must name {key}; got {rendered}"
        );
    }
    assert!(
        serde_yaml::from_str::<PresetConfig>("attack:\n  f64_lift_step: newton\n").is_err(),
        "an unknown step must not parse"
    );

    // The resolver never arms a step without the stage.
    let orphan = PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
        f64_lift_step: Some(F64LiftStep::LpMaximin),
        f64_basin_hops: Some(60),
        ..Default::default()
    })));
    assert_eq!(orphan, PostBabActiveSetPolicy::HISTORICAL);
    // Hops without lp_maximin resolve to none; an over-cap hop count is
    // treated as absent.
    let hops_without_step = PostBabActiveSetPolicy::from_preset(Some(&preset_with(AttackPreset {
        f64_target_margin_ulps: Some(8),
        f64_basin_hops: Some(60),
        ..Default::default()
    })));
    assert_eq!(hops_without_step.basin_hops(), 0);
    assert_eq!(
        lp_maximin_policy(crate::preset::F64_BASIN_HOPS_MAX + 1, None).basin_hops(),
        0
    );
}

/// DARK: no shipped preset sets either step key.
#[test]
fn no_shipped_preset_arms_the_f64_lift_keys() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut armed = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let Ok(entries) = std::fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            scanned += 1;
            if preset.attack.f64_lift_step.is_some() || preset.attack.f64_basin_hops.is_some() {
                armed.push(path.display().to_string());
            }
        }
    }
    assert!(
        scanned > 10,
        "the scan must actually read the shipped presets"
    );
    assert!(
        armed.is_empty(),
        "shipped presets arm #sbench-f64-maximin: {armed:?}"
    );
}

/// The `[f64-lift]` line carries the brief's prefix and a fixed schema.
#[test]
fn f64_lift_telemetry_line_has_the_fixed_schema() {
    let mut telemetry = F64AttackTelemetry::new(lp_maximin_policy(12, Some(30.0)));
    let line = telemetry.lift_line();
    assert!(
        line.starts_with("[f64-lift] step=lp_maximin iters=0 hops=0 best_f64="),
        "{line}"
    );
    for field in [
        " best_ort=",
        " rho_final=",
        " armed=1 reached=0 seeds_run=0 accepted=0 lp_solves=0 lp_failures=0",
        " hops_cap=12 hops_improved=0",
        " budget_source=attack.active_set_repair_secs",
        " stop=none reason=not_reached",
    ] {
        assert!(line.contains(field), "missing {field:?} in {line}");
    }
    telemetry.reach(2);
    telemetry.absorb_lift(&super::F64LiftStats {
        slp_iters: 40,
        accepted: 9,
        lp_solves: 40,
        lp_failures: 1,
        lp_pivots: 900,
        lp_cut_short: 0,
        hops: 3,
        hops_improved: 2,
        rho_final: 1e-6,
        stop: "hops exhausted",
    });
    telemetry.note_lift_budget(Duration::from_millis(12_500));
    let line = telemetry.lift_line();
    for field in [
        "iters=40 hops=3 ",
        " reached=1 ",
        " accepted=9 lp_solves=40 lp_failures=1 lp_pivots=900 lp_cut_short=0",
        " hops_improved=2 ",
        " phase_budget_s=12.500 ",
        " stop=hops exhausted ",
    ] {
        assert!(line.contains(field), "missing {field:?} in {line}");
    }
    telemetry.armed = false;
    telemetry.lift_armed = false;
}

/// The per-lineage hop seed is a pure function of the seed point.
#[test]
fn lift_rng_seed_is_a_pure_function_of_the_seed_point() {
    assert_eq!(lift_rng_seed(&[0.25, 0.5]), lift_rng_seed(&[0.25, 0.5]));
    assert_ne!(lift_rng_seed(&[0.25, 0.5]), lift_rng_seed(&[0.5, 0.25]));
    assert_ne!(lift_rng_seed(&[0.0]), lift_rng_seed(&[-0.0]));
}

// ---------------------------------------------------------------------------
// #sbench-f64-maximin: soundness of the LP-maximin lift on the slab.
// ---------------------------------------------------------------------------

fn run_lp_maximin(
    plan: &PointVjpF64Plan,
    spec: &VnnLibSpec,
    seed: &[f32],
    ulps: u32,
    hops: usize,
    amplitude: f64,
) -> Option<super::F64RepairOutcome> {
    f64_lp_maximin_lift(
        plan,
        |x: &[f32]| noisy_trusted(plan, x, amplitude),
        spec,
        &BOX_LO,
        &BOX_HI,
        &NO_PIN,
        seed,
        ulps,
        hops,
        lift_rng_seed(seed),
        Instant::now() + Duration::from_secs(20),
        64,
    )
    .map(|(outcome, _)| outcome)
}

/// The LP-maximin lift crosses the C8 slab (peak 2e-6) under 3-ULP trusted
/// noise with an 8-ULP target, through the lineage dispatch, and the point
/// satisfies both acceptance predicates.
#[test]
fn lp_maximin_crosses_a_slab_thinner_than_the_trusted_noise_floor() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.499998");
    let amplitude = 3.0 * f32_ulp_at(0.5);
    let (outcome, lift) = f64_lift_lineage(
        lp_maximin_policy(4, None),
        &plan,
        |x: &[f32]| noisy_trusted(&plan, x, amplitude),
        &spec,
        &BOX_LO,
        &BOX_HI,
        &NO_PIN,
        &[0.245, 0.245],
        8,
        Instant::now() + Duration::from_secs(20),
        64,
    )
    .expect("the near seed is in scope");
    let lift = lift.expect("lp_maximin returns its lift record");
    assert!(lift.lp_solves >= 1 && lift.accepted >= 1, "{lift:?}");
    assert!(outcome.initial_margin_f64 < 0.0);
    let found = outcome
        .violation
        .clone()
        .unwrap_or_else(|| panic!("no candidate: {outcome:?}"));
    let trusted = noisy_trusted(&plan, &found, amplitude).expect("trusted outputs");
    assert!(property_violated_f64(&spec, &widen(&found), &trusted));
    assert!(exact_min_margin(&plan, &spec, &found) >= 0.0);
    assert_eq!(outcome.stop, "target reached");
    // The `[f64-lift]` line's `rho_final=` on the SUCCESS exit is this run's
    // radius, not the `F64LiftStats::default()` zero (review fix).
    assert!(
        lift.rho_final > 0.0,
        "a consult-driven stop must record its run's trust radius: {lift:?}"
    );
}

/// SOUNDNESS, sampled: over many seeds, noise levels (including noise larger
/// than the slab), targets and hop counts, EVERY emitted point is confirmed
/// by the trusted forward at zero tolerance, has a non-negative exact-f64
/// margin, and lies in the box.
#[test]
fn lp_maximin_never_emits_a_point_either_predicate_rejects() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.499998");
    let mut emitted = 0usize;
    let mut state: u64 = 0x5EED_F64A_77AC_0002;
    let mut next_unit = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for trial in 0..60 {
        let sum = 0.48 + 0.04 * next_unit();
        let split = next_unit();
        let seed = [
            (sum * split).min(0.5) as f32,
            (sum * (1.0 - split)).min(0.5) as f32,
        ];
        let amplitude = [0.0, 1.0, 3.0, 40.0][trial % 4] * f32_ulp_at(0.5);
        let ulps = [1u32, 4, 16][trial % 3];
        let hops = [0usize, 8][trial % 2];
        let Some(outcome) = run_lp_maximin(&plan, &spec, &seed, ulps, hops, amplitude) else {
            continue;
        };
        if let Some(found) = outcome.violation {
            emitted += 1;
            let trusted = noisy_trusted(&plan, &found, amplitude).expect("trusted outputs");
            assert!(
                property_violated_f64(&spec, &widen(&found), &trusted),
                "trial {trial}: emitted a point the trusted forward rejects"
            );
            assert!(
                exact_min_margin(&plan, &spec, &found) >= 0.0,
                "trial {trial}: emitted a point exact f64 refutes"
            );
            assert!(found.iter().zip(&BOX_LO).all(|(v, lo)| v >= lo));
            assert!(found.iter().zip(&BOX_HI).all(|(v, hi)| v <= hi));
        }
    }
    assert!(
        emitted > 0,
        "the sampled trials must exercise the emitting path"
    );
}

/// SOUNDNESS: on an exactly-UNSAT slab (peak margin -5e-7) the lift, hops
/// included, emits nothing and never even consults a trusted forward that
/// LIES "violated" everywhere.
#[test]
fn lp_maximin_never_consults_or_emits_on_an_exactly_unsat_slab() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.5000005");
    let mut consulted = 0usize;
    let (outcome, lift) = f64_lp_maximin_lift(
        &plan,
        |_x: &[f32]| {
            consulted += 1;
            Some(vec![10.0, 10.0])
        },
        &spec,
        &BOX_LO,
        &BOX_HI,
        &NO_PIN,
        &[0.245, 0.245],
        4,
        16,
        lift_rng_seed(&[0.245, 0.245]),
        Instant::now() + Duration::from_secs(10),
        64,
    )
    .expect("the near seed is in scope");
    assert!(outcome.violation.is_none(), "{outcome:?}");
    assert!(outcome.best_margin_f64 < 0.0, "the slab is exactly empty");
    assert_eq!(outcome.trusted_evals, 0);
    assert_eq!(consulted, 0);
    assert_eq!(lift.hops, 16, "{lift:?}");
}

/// A trusted forward that never confirms yields no candidate, however far
/// the exact margin clears the target (the target escalates, then stops).
#[test]
fn lp_maximin_emits_nothing_the_trusted_forward_never_confirms() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.49");
    let (outcome, lift) = f64_lp_maximin_lift(
        &plan,
        |_x: &[f32]| Some(vec![-10.0, -10.0]),
        &spec,
        &BOX_LO,
        &BOX_HI,
        &NO_PIN,
        &[0.24, 0.24],
        4,
        8,
        1,
        Instant::now() + Duration::from_secs(10),
        64,
    )
    .expect("the near seed is in scope");
    assert!(outcome.violation.is_none(), "{outcome:?}");
    assert!(outcome.trusted_evals >= 1);
    assert!(outcome.best_margin_f64 >= 0.0);
    assert_eq!(lift.stop, "target escalations exhausted");
}

/// Out-of-scope lineages decline exactly as C8's do.
#[test]
fn lp_maximin_declines_out_of_scope_lineages() {
    let graph = slab_network();
    let plan = slab_plan(&graph);
    let spec = slab_spec("0.499998");
    assert!(
        run_lp_maximin(&plan, &spec, &[0.0, 0.0], 4, 8, 0.0).is_none(),
        "margin -0.5 is not a near miss"
    );
    assert!(
        run_lp_maximin(&plan, &spec, &[0.245], 4, 8, 0.0).is_none(),
        "arity mismatch"
    );
    assert!(
        run_lp_maximin(&plan, &spec, &[0.245, 0.245], 0, 8, 0.0).is_none(),
        "zero target"
    );
}

// ---------------------------------------------------------------------------
// #sbench-f64-maximin: the planted thin region behind a trap basin.
// ---------------------------------------------------------------------------

/// Trap and peak positions on the `u = X_0` axis, the trap height, and the
/// planted peak margin.
const PLANT_P1: f32 = 0.25;
const PLANT_P2: f32 = 0.25 + 1.0 / 512.0; // 0.25 + 2^-9, exact in f32
const PLANT_TRAP: f32 = -0.000_488_281_25; // -2^-11, exact in f32
const PLANT_PEAK: f64 = 2e-6;

fn plant_ystar(j: usize) -> f32 {
    0.5 + (j as f32 + 1.0) / 64.0
}

/// `g(u) = max(TRAP - |u - P1|, -|u - P2|)` from ReLUs, and twelve outputs
/// `Y_(2j) = g + y_j`, `Y_(2j+1) = g - y_j` over `y = X_1..X_6` (`X_7` is
/// free). With thresholds `Y_(2j) >= y*_j - PEAK`, `Y_(2j+1) >= -y*_j - PEAK`
/// every margin is `g(u) +- (y_j - y*_j) + PEAK`: a 12-atom region of peak
/// width `PEAK` at `(P2, y*)`, and a strict local maximum of the minimum
/// margin at `(P1, y*)` of value `TRAP + PEAK < 0`, whose basin ends
/// `(P2 - P1 + TRAP) / 2` to the right of `P1`.
fn planted_network() -> GraphNetwork {
    let mut w1 = vec![0.0f32; 11 * 8];
    let mut b1 = vec![0.0f32; 11];
    let set = |w: &mut Vec<f32>, cols: usize, r: usize, c: usize, v: f32| w[r * cols + c] = v;
    set(&mut w1, 8, 0, 0, 1.0);
    b1[0] = -PLANT_P1;
    set(&mut w1, 8, 1, 0, -1.0);
    b1[1] = PLANT_P1;
    set(&mut w1, 8, 2, 0, 1.0);
    b1[2] = -PLANT_P2;
    set(&mut w1, 8, 3, 0, -1.0);
    b1[3] = PLANT_P2;
    for i in 0..7 {
        set(&mut w1, 8, 4 + i, 1 + i, 1.0);
    }
    let mut w2 = vec![0.0f32; 10 * 11];
    let mut b2 = vec![0.0f32; 10];
    set(&mut w2, 11, 0, 0, -1.0);
    set(&mut w2, 11, 0, 1, -1.0);
    set(&mut w2, 11, 0, 2, 1.0);
    set(&mut w2, 11, 0, 3, 1.0);
    b2[0] = PLANT_TRAP;
    set(&mut w2, 11, 1, 2, 1.0);
    set(&mut w2, 11, 2, 3, 1.0);
    for i in 0..7 {
        set(&mut w2, 11, 3 + i, 4 + i, 1.0);
    }
    let mut w3 = vec![0.0f32; 12 * 10];
    for j in 0..6 {
        for (sign, k) in [(1.0f32, 2 * j), (-1.0, 2 * j + 1)] {
            set(&mut w3, 10, k, 0, 1.0);
            set(&mut w3, 10, k, 1, -1.0);
            set(&mut w3, 10, k, 2, -1.0);
            set(&mut w3, 10, k, 3 + j, sign);
        }
    }
    let linear = |w: Vec<f32>, rows: usize, cols: usize, b: Vec<f32>| {
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((rows, cols), w).expect("weight shape"),
                Some(Array1::from_vec(b)),
            )
            .expect("linear layer"),
        )
    };
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("l1", linear(w1, 11, 8, b1)));
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer),
        vec!["l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l2",
        linear(w2, 10, 11, b2),
        vec!["r1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "r2",
        Layer::ReLU(ReLULayer),
        vec!["l2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        linear(w3, 12, 10, vec![0.0; 12]),
        vec!["r2".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn planted_spec() -> VnnLibSpec {
    let mut text = String::new();
    for i in 0..8 {
        text.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for k in 0..12 {
        text.push_str(&format!("(declare-const Y_{k} Real)\n"));
    }
    for i in 0..8 {
        text.push_str(&format!(
            "(assert (>= X_{i} 0.0))\n(assert (<= X_{i} 1.0))\n"
        ));
    }
    text.push_str("(assert (or (and");
    for j in 0..6 {
        let ystar = f64::from(plant_ystar(j));
        text.push_str(&format!(
            " (>= Y_{} {}) (>= Y_{} {})",
            2 * j,
            ystar - PLANT_PEAK,
            2 * j + 1,
            -ystar - PLANT_PEAK
        ));
    }
    text.push_str(")))\n");
    ny_onnx::vnnlib::parse_vnnlib(&text).expect("planted property parses")
}

/// THE REGION NO LOCAL LIFT REACHES: from a seed on the far side of the trap
/// (`u = P1 + 3e-4`, `y` off `y*` by `2e-4`), C8's min-norm lift and the
/// LP-maximin lift WITHOUT hops both climb to the trap's local maximum
/// (exact margin `TRAP + PEAK = -4.86e-4`) and stall; with basin hops the
/// LP-maximin lift lands in the planted 12-atom region (peak `2e-6`, i.e.
/// ~34 f32 ULPs at |Y| ~ 0.6), and its point satisfies both acceptance
/// predicates under 3-ULP trusted noise.
#[test]
fn lp_maximin_reaches_a_planted_thin_region() {
    let graph = planted_network();
    let lo = [0.0f32; 8];
    let hi = [1.0f32; 8];
    let pin: [Option<f64>; 8] = [None; 8];
    let plan = build_plan(
        &graph,
        &[8],
        &lo,
        &hi,
        Instant::now() + Duration::from_secs(30),
    )
    .expect("the planted net has a chain twin");
    assert_eq!(plan.kind(), "chain");
    let spec = planted_spec();
    let amplitude = 3.0 * f32_ulp_at(0.5);
    // Offsets are applied in f64 and rounded once to f32.
    let mut seed = [0.5f32; 8];
    seed[0] = (f64::from(PLANT_P1) + 3e-4) as f32;
    for j in 0..6 {
        let offset = if j % 2 == 0 { 2e-4 } else { -2e-4 };
        seed[1 + j] = (f64::from(plant_ystar(j)) + offset) as f32;
    }
    // The planted optimum really is a counterexample, thinner than the noise
    // floor it is judged under times twelve atoms.
    let mut optimum = [0.5f32; 8];
    optimum[0] = PLANT_P2;
    for j in 0..6 {
        optimum[1 + j] = plant_ystar(j);
    }
    let peak = exact_min_margin(&plan, &spec, &optimum);
    assert!(
        (peak - PLANT_PEAK).abs() < 1e-12,
        "planted peak {peak:e} != {PLANT_PEAK:e}"
    );
    let seed_margin = exact_min_margin(&plan, &spec, &seed);
    assert!(
        seed_margin < -4e-4 && seed_margin > -2e-2,
        "{seed_margin:e}"
    );

    let min_norm = f64_target_margin_repair(
        &plan,
        |x: &[f32]| noisy_trusted(&plan, x, amplitude),
        &spec,
        &lo,
        &hi,
        &pin,
        &seed,
        8,
        Instant::now() + Duration::from_secs(20),
        64,
    )
    .expect("the seed is in scope");
    assert!(
        min_norm.violation.is_none() && min_norm.best_margin_f64 < -4e-4,
        "C8's local lift must stall in the trap: {min_norm:?}"
    );

    let lift = |hops: usize| {
        f64_lp_maximin_lift(
            &plan,
            |x: &[f32]| noisy_trusted(&plan, x, amplitude),
            &spec,
            &lo,
            &hi,
            &pin,
            &seed,
            8,
            hops,
            lift_rng_seed(&seed),
            Instant::now() + Duration::from_secs(60),
            64,
        )
        .expect("the seed is in scope")
    };
    let (stalled, stalled_stats) = lift(0);
    assert!(
        stalled.violation.is_none() && stalled.best_margin_f64 < -4e-4,
        "without hops the maximin lift must stall in the trap too: {stalled:?} {stalled_stats:?}"
    );

    let (hopped, hop_stats) = lift(64);
    let found = hopped
        .violation
        .clone()
        .unwrap_or_else(|| panic!("hops must reach the planted region: {hopped:?} {hop_stats:?}"));
    assert!(hop_stats.hops >= 1, "{hop_stats:?}");
    let trusted = noisy_trusted(&plan, &found, amplitude).expect("trusted outputs");
    assert!(property_violated_f64(&spec, &widen(&found), &trusted));
    let exact = exact_min_margin(&plan, &spec, &found);
    assert!((0.0..=PLANT_PEAK + 1e-12).contains(&exact), "{exact:e}");
    assert!(found.iter().all(|v| (0.0..=1.0).contains(v)));
}
