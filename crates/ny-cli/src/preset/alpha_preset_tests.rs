// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::apply::apply_preset;
use super::*;
use ny_propagate::BetaCrownConfig;

/// Verify that solver.alpha-crown.full_conv_alpha flows through to
/// AlphaCrownConfig. The reference cifar100 config sets this to false
/// to enable channel-shared alpha (63x fewer parameters). #4404.
#[test]
fn apply_preset_maps_full_conv_alpha_into_alpha_config_4404() {
    let preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                full_conv_alpha: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    assert!(
        config.alpha_config.full_conv_alpha,
        "default should be true (per-neuron alpha)"
    );
    apply_preset(&mut config, &preset).expect("full_conv_alpha preset should apply");
    assert!(
        !config.alpha_config.full_conv_alpha,
        "full_conv_alpha should be false after preset application"
    );
}

#[test]
fn apply_preset_maps_aggregate_reference_refresh_budget_and_rejects_bad_fractions() {
    let mut default_config = BetaCrownConfig::default();
    apply_preset(&mut default_config, &PresetConfig::default())
        .expect("empty preset must preserve refresh defaults");
    assert_eq!(
        default_config.alpha_config.reference_refresh_fraction,
        ny_propagate::AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
    );
    assert_eq!(default_config.alpha_config.reference_refresh_max_secs, None);

    let preset: PresetConfig = serde_yaml::from_str(
        r#"
solver:
  alpha_crown:
    reference_refresh_fraction: 0.125
    reference_refresh_max_secs: 9
bab:
  alpha_crown:
    reference_refresh_fraction: 0.2
    reference_refresh_max_secs: 12
"#,
    )
    .expect("typed aggregate refresh keys must parse from production YAML");
    assert_eq!(
        preset.solver.alpha_crown.reference_refresh_fraction,
        Some(0.125)
    );
    assert_eq!(
        preset.solver.alpha_crown.reference_refresh_max_secs,
        Some(9)
    );
    assert_eq!(preset.bab.alpha_crown.reference_refresh_fraction, Some(0.2));
    assert_eq!(preset.bab.alpha_crown.reference_refresh_max_secs, Some(12));
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("valid solver and BaB refresh budgets must apply");
    assert_eq!(config.alpha_config.reference_refresh_fraction, 0.2);
    assert_eq!(config.alpha_config.reference_refresh_max_secs, Some(12));

    for invalid in [f32::NAN, f32::NEG_INFINITY, 0.0, 0.009, 1.001] {
        let invalid_preset = PresetConfig {
            solver: SolverPreset {
                alpha_crown: AlphaCrownPreset {
                    lr_alpha: Some(0.75),
                    reference_refresh_fraction: Some(invalid),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut unchanged = BetaCrownConfig::default();
        let original_lr = unchanged.alpha_config.learning_rate;
        let error = apply_preset(&mut unchanged, &invalid_preset)
            .expect_err("invalid refresh fractions must fail preset application");
        assert!(
            error
                .to_string()
                .contains("reference_refresh_fraction must be finite and in [0.01, 1.0]"),
            "unexpected error for {invalid:?}: {error:#}"
        );
        assert_eq!(
            unchanged.alpha_config.learning_rate, original_lr,
            "validation must happen before any preset mutation"
        );
    }

    let disable_preset = PresetConfig {
        bab: BabPreset {
            alpha_crown: AlphaCrownPreset {
                reference_refresh_max_secs: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut disabled = BetaCrownConfig::default();
    apply_preset(&mut disabled, &disable_preset).expect("zero is an explicit scheduling disable");
    assert_eq!(disabled.alpha_config.reference_refresh_max_secs, Some(0));
}

#[test]
fn apply_preset_maps_forward_linear_deadline_fallback_with_default_off() {
    let mut absent = BetaCrownConfig::default();
    apply_preset(&mut absent, &PresetConfig::default()).expect("empty preset should apply");
    assert!(
        !absent.alpha_config.forward_linear_deadline_fallback_to_ibp,
        "absent key must preserve the historical CROWN-IBP fallback"
    );

    let preset: PresetConfig = serde_yaml::from_str(
        r#"
solver:
  alpha_crown:
    forward_linear_deadline_fallback_to_ibp: true
bab:
  alpha_crown:
    forward_linear_deadline_fallback_to_ibp: false
"#,
    )
    .expect("typed deadline-fallback keys must parse");
    assert_eq!(
        preset
            .solver
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true)
    );
    assert_eq!(
        preset
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(false)
    );

    let mut overridden = BetaCrownConfig::default();
    apply_preset(&mut overridden, &preset).expect("typed fallback policy should apply");
    assert!(
        !overridden
            .alpha_config
            .forward_linear_deadline_fallback_to_ibp,
        "bab.alpha_crown must retain the existing override precedence"
    );

    let bab_enabled = PresetConfig {
        bab: BabPreset {
            alpha_crown: AlphaCrownPreset {
                forward_linear_deadline_fallback_to_ibp: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut enabled = BetaCrownConfig::default();
    apply_preset(&mut enabled, &bab_enabled).expect("explicit opt-in should apply");
    assert!(enabled.alpha_config.forward_linear_deadline_fallback_to_ibp);
}

/// `alpha_crown.fix_interm_bounds` reaches `AlphaCrownConfig::fix_interm_bounds`
/// from BOTH the `solver:` and the `bab:` section (the two locations
/// `apply_preset` feeds through `apply_alpha_preset`), and an ABSENT key leaves
/// the built-in default alone so every preset that does not name it stays
/// byte-identical. #ml4acopf-interm-bounds.
#[test]
fn apply_preset_maps_fix_interm_bounds_into_alpha_config() {
    // Absent ⇒ untouched default (the cheap IBP-intermediate mode).
    let mut config = BetaCrownConfig::default();
    assert!(
        config.alpha_config.fix_interm_bounds,
        "default should be true (IBP intermediates)"
    );
    apply_preset(&mut config, &PresetConfig::default()).expect("empty preset should apply");
    assert!(
        config.alpha_config.fix_interm_bounds,
        "an absent fix_interm_bounds key must not change the default"
    );

    // solver.alpha_crown location.
    let solver_preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                fix_interm_bounds: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &solver_preset).expect("preset should apply");
    assert!(
        !config.alpha_config.fix_interm_bounds,
        "solver.alpha_crown.fix_interm_bounds: false must select CROWN-IBP intermediates"
    );

    // bab.alpha_crown location (where the ml4acopf preset declares it).
    let bab_preset = PresetConfig {
        bab: BabPreset {
            alpha_crown: AlphaCrownPreset {
                fix_interm_bounds: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &bab_preset).expect("preset should apply");
    assert!(
        !config.alpha_config.fix_interm_bounds,
        "bab.alpha_crown.fix_interm_bounds: false must select CROWN-IBP intermediates"
    );

    // Explicit `true` is honoured too, so a category can opt back in.
    let mut config = BetaCrownConfig {
        alpha_config: ny_propagate::AlphaCrownConfig {
            fix_interm_bounds: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let true_preset = PresetConfig {
        bab: BabPreset {
            alpha_crown: AlphaCrownPreset {
                fix_interm_bounds: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    apply_preset(&mut config, &true_preset).expect("preset should apply");
    assert!(config.alpha_config.fix_interm_bounds);
}

#[test]
fn cgan_sparse_target_complete_root_is_typed_and_default_dark() {
    let mut default_config = BetaCrownConfig::default();
    assert!(!default_config.alpha_config.cgan_sparse_target_complete_root);
    apply_preset(&mut default_config, &PresetConfig::default())
        .expect("empty preset should preserve the dark default");
    assert!(!default_config.alpha_config.cgan_sparse_target_complete_root);

    let preset: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    fix_interm_bounds: false
    cgan_sparse_target_complete_root: true
"#,
    )
    .expect("typed cGAN root policy");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("typed cGAN root policy should apply");
    assert!(!config.alpha_config.fix_interm_bounds);
    assert!(config.alpha_config.cgan_sparse_target_complete_root);
}

#[test]
fn cgan_complete_crown_ibp_root_is_typed_and_default_dark() {
    let mut default_config = BetaCrownConfig::default();
    assert!(!default_config.alpha_config.cgan_complete_crown_ibp_root);
    apply_preset(&mut default_config, &PresetConfig::default())
        .expect("empty preset should preserve the dark default");
    assert!(!default_config.alpha_config.cgan_complete_crown_ibp_root);

    let preset: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  alpha_crown:
    cgan_complete_crown_ibp_root: true
"#,
    )
    .expect("typed complete cGAN root policy");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("typed complete cGAN root policy should apply");
    assert!(config.alpha_config.fix_interm_bounds);
    assert!(config.alpha_config.cgan_complete_crown_ibp_root);
    assert!(!config.alpha_config.cgan_sparse_target_complete_root);
}

/// `alpha_crown.alpha_zero_yield_frac` reaches `AlphaCrownConfig::alpha_zero_yield_frac`,
/// an absent key leaves the default (`None`, byte-identical) alone, and an
/// out-of-range value is REJECTED at validation rather than silently dropped
/// at read time. #alpha-zero-yield delivery (measured_gate_delivery.rs).
#[test]
fn apply_preset_maps_alpha_zero_yield_frac_into_alpha_config() {
    // Absent ⇒ untouched None default.
    let mut config = BetaCrownConfig::default();
    assert_eq!(config.alpha_config.alpha_zero_yield_frac, None);
    apply_preset(&mut config, &PresetConfig::default()).expect("empty preset should apply");
    assert_eq!(
        config.alpha_config.alpha_zero_yield_frac, None,
        "an absent alpha_zero_yield_frac key must stay byte-identical"
    );

    // Armed ⇒ delivered.
    let preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                alpha_zero_yield_frac: Some(0.25),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("valid fraction should apply");
    assert_eq!(config.alpha_config.alpha_zero_yield_frac, Some(0.25));
    assert_eq!(
        effective_alpha_zero_yield_frac(&preset),
        Some(0.25),
        "the receipt adapter must see the same solver-layer value"
    );

    // BaB applies after solver and therefore wins when both layers name the
    // field. Pin the adapter to the same precedence as the operative config.
    let layered = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                alpha_zero_yield_frac: Some(0.25),
                ..Default::default()
            },
            ..Default::default()
        },
        bab: BabPreset {
            alpha_crown: AlphaCrownPreset {
                alpha_zero_yield_frac: Some(0.5),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &layered).expect("both valid fractions should apply");
    assert_eq!(config.alpha_config.alpha_zero_yield_frac, Some(0.5));
    assert_eq!(effective_alpha_zero_yield_frac(&layered), Some(0.5));

    // Out of range ⇒ rejected loudly at validation, not dropped at read time.
    for bad in [0.0, 0.9, 1.5, -0.1, f64::NAN] {
        let preset = PresetConfig {
            solver: SolverPreset {
                alpha_crown: AlphaCrownPreset {
                    alpha_zero_yield_frac: Some(bad),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig::default();
        let err = apply_preset(&mut config, &preset);
        assert!(
            err.is_err(),
            "alpha_zero_yield_frac={bad} must be rejected at validation"
        );
    }
}

#[test]
fn shipped_alpha_zero_yield_bindings_require_promotion_grade_evidence() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut bindings: Vec<(String, &'static str, u64)> = Vec::new();
    let mut competition_dirs: Vec<_> = fs::read_dir(&root)
        .expect("readable configs root")
        .map(|entry| entry.expect("configs entry"))
        .filter(|entry| entry.file_type().expect("config entry type").is_dir())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("vnncomp"))
        .collect();
    competition_dirs.sort_by_key(|entry| entry.file_name());
    assert!(
        !competition_dirs.is_empty(),
        "at least one shipped VNN-COMP config directory must be audited"
    );
    for competition_dir in competition_dirs {
        let year = competition_dir.file_name().to_string_lossy().into_owned();
        let entries = fs::read_dir(competition_dir.path()).expect("readable competition configs");
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
                continue;
            }
            let raw = fs::read_to_string(&path).expect("readable preset");
            let preset: PresetConfig = serde_yaml::from_str(&raw)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            let relative = format!(
                "{year}/{}",
                path.file_name().expect("preset filename").to_string_lossy()
            );
            if let Some(value) = preset.solver.alpha_crown.alpha_zero_yield_frac {
                bindings.push((relative.clone(), "solver", value.to_bits()));
            }
            if let Some(value) = preset.bab.alpha_crown.alpha_zero_yield_frac {
                bindings.push((relative, "bab", value.to_bits()));
            }
        }
    }

    assert!(
        bindings.is_empty(),
        "shipped bindings require a retained, promotion-grade A/B covering every row \
         they can reach: {bindings:?}"
    );

    let preset = load_preset(&root.join("vnncomp25/cifar100_2024.yaml"))
        .expect("the shipped CIFAR100 preset loads");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("the shipped CIFAR100 preset applies");
    assert_eq!(
        config.alpha_config.alpha_zero_yield_frac, None,
        "the sampled, incompletely retained evidence does not arm the 200-row category"
    );
}

/// `alpha_crown.early_stop_patience` reaches `AlphaCrownConfig::early_stop_patience`
/// from BOTH the `solver:` and the `bab:` section, an ABSENT key leaves the
/// reference default (10) alone, and it does NOT disturb the unrelated
/// BaB-level `BetaCrownConfig::early_stop_patience`. #alpha-patience-preset.
#[test]
fn apply_preset_maps_early_stop_patience_into_alpha_config() {
    // Absent ⇒ untouched reference default.
    let mut config = BetaCrownConfig::default();
    assert_eq!(
        config.alpha_config.early_stop_patience, 10,
        "default should be the α,β-CROWN reference value"
    );
    let bab_default = config.early_stop_patience;
    apply_preset(&mut config, &PresetConfig::default()).expect("empty preset should apply");
    assert_eq!(
        config.alpha_config.early_stop_patience, 10,
        "an absent early_stop_patience key must not change the default"
    );
    assert_eq!(config.early_stop_patience, bab_default);

    // solver.alpha_crown location. 20 iterations need >10 patience, or the
    // counter (which starts accumulating at iteration 0, where the bound is
    // still the CROWN init and improvement is zero) breaks the loop at iter 9.
    let solver_preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                iterations: Some(20),
                early_stop_patience: Some(20),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &solver_preset).expect("preset should apply");
    assert_eq!(config.alpha_config.iterations, 20);
    assert_eq!(config.alpha_config.early_stop_patience, 20);
    assert_eq!(
        config.early_stop_patience, bab_default,
        "the alpha knob must not touch the BaB-level early_stop_patience"
    );

    // bab.alpha_crown location.
    let bab_preset = PresetConfig {
        bab: BabPreset {
            alpha_crown: AlphaCrownPreset {
                early_stop_patience: Some(4),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &bab_preset).expect("preset should apply");
    assert_eq!(config.alpha_config.early_stop_patience, 4);
    assert_eq!(config.early_stop_patience, bab_default);

    // 0 is a real reference value ("stop at the first non-improving iteration"),
    // not a sentinel for unset, so it must pass through.
    let zero_preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                early_stop_patience: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &zero_preset).expect("preset should apply");
    assert_eq!(config.alpha_config.early_stop_patience, 0);
}

/// Landing the KEY is the deliverable. No shipped preset names it in this
/// change, so every shipped category keeps the reference patience of 10 and is
/// byte-identical. #alpha-patience-preset.
#[test]
fn no_shipped_preset_sets_alpha_early_stop_patience() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut set: Vec<String> = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|e| panic!("preset {} must parse: {e}", path.display()));
            if preset
                .solver
                .alpha_crown
                .early_stop_patience
                .or(preset.bab.alpha_crown.early_stop_patience)
                .is_some()
            {
                set.push(format!("{year}/{}", path.display()));
            }
        }
    }
    assert!(
        set.is_empty(),
        "no shipped preset may set alpha early_stop_patience in this change: {set:?}"
    );
}

/// The YAML key is spelled `early_stop_patience` under `alpha_crown`, and a
/// malformed value must FAIL TO PARSE rather than silently arming a default —
/// an absent key is the only thing that keeps the built-in 10.
#[test]
fn early_stop_patience_yaml_key_parses_and_rejects_malformed_values() {
    let preset: PresetConfig =
        serde_yaml::from_str("solver:\n  alpha_crown:\n    early_stop_patience: 20\n")
            .expect("valid key must parse");
    assert_eq!(preset.solver.alpha_crown.early_stop_patience, Some(20));

    // Absent key ⇒ None ⇒ byte-identical.
    let bare: PresetConfig = serde_yaml::from_str("solver:\n  alpha_crown: {}\n").expect("parses");
    assert_eq!(bare.solver.alpha_crown.early_stop_patience, None);

    for malformed in ["\"twenty\"", "-1", "1.5", "true"] {
        let yaml = format!("solver:\n  alpha_crown:\n    early_stop_patience: {malformed}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "malformed early_stop_patience {malformed} must not parse"
        );
    }
}

/// ml4acopf_2024 is the ONLY shipped VNN-COMP 2025 preset that enables CROWN-IBP
/// intermediate tightening, and it may only do so while it ALSO carries the
/// `bab.max_queue_bytes` companion cap.
///
/// The key was held until 2026-07-26 because it drove
/// 118_ieee_ml4acopf-linear-residual / 118_ieee_prop2 to 116.3 GiB RSS and a
/// SIGKILL. The blowup was NOT the O(N^2) CROWN-IBP sweep (measured at 2.86 GiB,
/// 2.4% of peak) but an UNBOUNDED BaB domain queue; this key is the enabler, not
/// the allocator, because it makes the root bound finite and lets a hopeless BaB
/// actually run. `bab.max_queue_bytes` bounds that queue, so the pair is safe and
/// the pairing is what this test pins: enabling the tightening WITHOUT the cap is
/// the configuration that OOMs a 121 GiB box.
/// #ml4acopf-interm-bounds #ml4acopf-bab-queue-mem.
#[test]
fn crown_ibp_intermediates_only_ml4acopf_and_only_with_a_queue_cap() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/vnncomp25");

    let mut opted_in: Vec<String> = Vec::new();
    for entry in fs::read_dir(&dir).expect("readable configs dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        if path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| stem.ends_with("_canary"))
        {
            continue;
        }
        let text = fs::read_to_string(&path).expect("readable preset");
        let preset: PresetConfig = serde_yaml::from_str(&text)
            .unwrap_or_else(|e| panic!("preset {} must parse: {e}", path.display()));
        let requested = preset
            .solver
            .alpha_crown
            .fix_interm_bounds
            .or(preset.bab.alpha_crown.fix_interm_bounds);
        if requested == Some(false) {
            opted_in.push(
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string(),
            );
        }
    }
    opted_in.sort();
    assert_eq!(
        opted_in,
        vec!["ml4acopf_2024".to_string()],
        "exactly one category may enable CROWN-IBP intermediates today"
    );

    // The pairing invariant: the tightening is only safe alongside the queue cap.
    let ml4 =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/vnncomp25/ml4acopf_2024.yaml");
    let preset: PresetConfig =
        serde_yaml::from_str(&fs::read_to_string(&ml4).expect("readable ml4acopf preset"))
            .expect("ml4acopf preset must parse");
    assert!(
        preset.bab.max_queue_bytes.is_some_and(|b| b > 0),
        "ml4acopf enables CROWN-IBP intermediates, so it MUST also set a positive \
         bab.max_queue_bytes — without the cap that combination OOM-kills a 121 GiB box \
         (measured 116.3 GiB, rc=137)"
    );
}

// ---------------------------------------------------------------------------
// attack.pgd_order: after — deferred attack placement (#pgd-order-after)
// ---------------------------------------------------------------------------

/// `after` without the compat marker now RESOLVES instead of erroring, and moves the
/// attack's budget from before BaB to after it.
#[test]
fn pgd_order_after_defers_the_attack_budget() {
    use crate::preset::{resolve_initial_pgd_schedule, AttackPreset, ResolvedInitialPgdSchedule};

    let preset = PresetConfig {
        attack: AttackPreset {
            pgd_order: Some("after".to_string()),
            ..Default::default()
        },
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                upfront_pgd_fraction: Some(0.40),
                post_bab_pgd_fraction: Some(0.10),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        resolve_initial_pgd_schedule(&preset).expect("resolves"),
        Some(ResolvedInitialPgdSchedule::Deferred),
        "'after' must no longer be an error"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("preset applies");
    assert!(
        config.phase_budget.upfront_pgd_fraction.abs() < 1e-9,
        "the upfront stage must be emptied, got {}",
        config.phase_budget.upfront_pgd_fraction
    );
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.40).abs() < 1e-9,
        "the attack keeps its 0.40 slice, spent AFTER BaB; got {}",
        config.phase_budget.post_bab_pgd_fraction
    );
}

/// The compat marker still pins the historical upfront placement, so a preset that has not
/// been re-measured is untouched.
#[test]
fn pgd_order_after_with_compat_marker_stays_upfront() {
    use crate::preset::{
        resolve_initial_pgd_schedule, AttackPreset, NyPgdOrderCompat, ResolvedInitialPgdSchedule,
    };

    let preset = PresetConfig {
        attack: AttackPreset {
            pgd_order: Some("after".to_string()),
            ny_pgd_order_compat: Some(NyPgdOrderCompat::Upfront),
            ..Default::default()
        },
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                upfront_pgd_fraction: Some(0.40),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        resolve_initial_pgd_schedule(&preset).expect("resolves"),
        Some(ResolvedInitialPgdSchedule::Upfront)
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("applies");
    assert!(
        (config.phase_budget.upfront_pgd_fraction - 0.40).abs() < 1e-9,
        "compat must preserve the upfront slice"
    );
}

/// EVERY shipped preset that asks for `after` still carries the compat marker, so this change
/// is inert on the competition configs until one is deliberately re-measured.
#[test]
fn no_shipped_preset_silently_switches_to_deferred_attack() {
    use crate::preset::{resolve_initial_pgd_schedule, ResolvedInitialPgdSchedule};

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/vnncomp25");
    let mut deferred: Vec<String> = Vec::new();
    for entry in fs::read_dir(&dir).expect("readable configs dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let raw = fs::read_to_string(&path).expect("read preset");
        let Ok(preset) = serde_yaml::from_str::<PresetConfig>(&raw) else {
            continue;
        };
        if matches!(
            resolve_initial_pgd_schedule(&preset),
            Ok(Some(ResolvedInitialPgdSchedule::Deferred))
        ) {
            deferred.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        deferred.is_empty(),
        "these presets would change scheduling without a measured A/B: {deferred:?}"
    );
}

/// `middle` is still unimplemented and must say so rather than silently deferring.
#[test]
fn pgd_order_middle_still_errors_without_compat() {
    use crate::preset::{resolve_initial_pgd_schedule, AttackPreset};

    let preset = PresetConfig {
        attack: AttackPreset {
            pgd_order: Some("middle".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let err = resolve_initial_pgd_schedule(&preset).expect_err("middle is not implemented");
    let msg = format!("{err}");
    assert!(
        msg.contains("middle"),
        "error should name the schedule: {msg}"
    );
}

// ---------------------------------------------------------------------------
// #root-alpha-margin delivery: typed key + env override (see
// crates/ny-cli/tests/measured_gate_delivery.rs for why a typed key is required)
// ---------------------------------------------------------------------------

#[test]
fn root_alpha_margin_preset_key_reaches_the_alpha_config() {
    let preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                root_alpha_margin: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    assert!(
        !config.alpha_config.root_alpha_margin,
        "default must stay off so an unnamed key is byte-identical"
    );
    apply_preset(&mut config, &preset).expect("preset applies");
    assert!(
        config.alpha_config.root_alpha_margin,
        "the typed key is the only way this experimental lever reaches a scored run"
    );
}

#[test]
fn root_alpha_margin_absent_key_leaves_the_default_untouched() {
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &PresetConfig::default()).expect("empty preset applies");
    assert!(!config.alpha_config.root_alpha_margin);
}

#[test]
fn no_shipped_preset_arms_root_alpha_margin_without_a_current_positive_ab() {
    // Same discipline as the pgd_order guard: an experimental lever may be DELIVERABLE without
    // being ARMED. Arming one changes scored behaviour and requires a current sound-path
    // positive A/B.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed: Vec<String> = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let raw = fs::read_to_string(&path).expect("readable preset");
            let preset: PresetConfig = serde_yaml::from_str(&raw)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.solver.alpha_crown.root_alpha_margin == Some(true)
                || preset.bab.alpha_crown.root_alpha_margin == Some(true)
            {
                armed.push(format!(
                    "{year}/{}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }
    assert!(
        armed.is_empty(),
        "these presets arm #root-alpha-margin without a current sound-path positive A/B: \
         {armed:?}"
    );
}

/// #envelope-grad DELIVERY. The rule is measured to move the root census where
/// every other alpha lever left it bit-identical, so it has to reach a SCORED
/// run — and `vnncomp_scripts/run_instance.sh` exports exactly one `NY_*`, so an
/// env-only lever never can. This pins the preset path that does.
#[test]
fn apply_preset_maps_alpha_envelope_grad_into_alpha_config() {
    // Absent => byte-identical to the shipped local rule.
    let mut config = BetaCrownConfig::default();
    assert!(
        !config.alpha_config.alpha_envelope_grad,
        "the envelope rule must default OFF; the shipped local rule is the baseline"
    );
    apply_preset(&mut config, &PresetConfig::default()).expect("empty preset should apply");
    assert!(
        !config.alpha_config.alpha_envelope_grad,
        "an absent alpha_envelope_grad key must leave the local rule in place"
    );

    // Armed via solver.alpha_crown.
    let preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                alpha_envelope_grad: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("envelope grad should apply");
    assert!(
        config.alpha_config.alpha_envelope_grad,
        "solver.alpha_crown.alpha_envelope_grad must reach AlphaCrownConfig"
    );

    // Explicit false must be honoured, not treated as absent — a preset that
    // deliberately disables the rule has to win over any future default flip.
    let preset_off = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                alpha_envelope_grad: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    config.alpha_config.alpha_envelope_grad = true;
    apply_preset(&mut config, &preset_off).expect("explicit false should apply");
    assert!(
        !config.alpha_config.alpha_envelope_grad,
        "an explicit `false` must disarm, not be indistinguishable from absent"
    );
}
