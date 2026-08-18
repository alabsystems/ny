// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Phase budget preset tests (#2206 Packet E).

use super::apply::apply_preset;
use super::load_preset;
use super::{BabPreset, PhaseBudgetPreset, PresetConfig};
use ny_propagate::BetaCrownConfig;
use std::path::Path;

fn assert_phase_budget_pair(name: &str, path: &str, initial_bounds_fraction: f32) {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join(path)).unwrap();
    assert_eq!(
        preset.bab.phase_budget.initial_bounds_fraction,
        Some(initial_bounds_fraction),
        "{name} should set initial_bounds_fraction to {initial_bounds_fraction}"
    );
    assert_eq!(
        preset.bab.phase_budget.post_bab_pgd_fraction,
        Some(0.10),
        "{name} should set post_bab_pgd_fraction to 0.10"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - initial_bounds_fraction).abs() < 1e-6,
        "{name} initial_bounds_fraction should propagate to config as {initial_bounds_fraction}"
    );
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
        "{name} post_bab_pgd_fraction should propagate to config"
    );
}

/// #2206 Packet E: phase_budget preset overrides flow to BetaCrownConfig.
#[test]
fn apply_preset_propagates_phase_budget_overrides_2206() {
    let preset = PresetConfig {
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                initial_bounds_fraction: Some(0.15),
                upfront_pgd_fraction: Some(0.10),
                mip_min_secs: Some(10),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "initial_bounds_fraction should be overridden to 0.15"
    );
    assert!(
        (config.phase_budget.upfront_pgd_fraction - 0.10).abs() < 1e-6,
        "upfront_pgd_fraction should be overridden to 0.10"
    );
    assert_eq!(
        config.phase_budget.mip_min_secs, 10,
        "mip_min_secs should be overridden to 10"
    );
    // Unset fields stay at defaults.
    assert!(
        (config.phase_budget.reduced_verification_fraction - 0.40).abs() < 1e-6,
        "unset reduced_verification_fraction should remain at default 0.40"
    );
    assert!(
        (config.phase_budget.disjunctive_pgd_fraction - 0.50).abs() < 1e-6,
        "unset disjunctive_pgd_fraction should remain at default 0.50"
    );
}

/// #2206 Packet E: phase_budget YAML section deserializes correctly.
#[test]
fn phase_budget_yaml_deserialization_2206() {
    let yaml = r#"
bab:
  timeout: 100
  phase_budget:
    initial_bounds_fraction: 0.15
    upfront_pgd_fraction: 0.10
    reduced_verification_fraction: 0.30
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();

    assert_eq!(preset.bab.phase_budget.initial_bounds_fraction, Some(0.15));
    assert_eq!(preset.bab.phase_budget.upfront_pgd_fraction, Some(0.10));
    assert_eq!(
        preset.bab.phase_budget.reduced_verification_fraction,
        Some(0.30)
    );
    // Omitted fields remain None (default).
    assert!(preset.bab.phase_budget.disjunctive_pgd_fraction.is_none());

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "YAML initial_bounds_fraction should flow to config"
    );
    assert!(
        (config.phase_budget.reduced_verification_fraction - 0.30).abs() < 1e-6,
        "YAML reduced_verification_fraction should flow to config"
    );
}

/// #2206 Packet E: soundnessbench and cifar100_2024 presets cap initial_bounds_fraction.
#[test]
fn soundnessbench_and_cifar100_presets_cap_initial_bounds_fraction_2206() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let soundnessbench =
        load_preset(&repo_root.join("configs/vnncomp25/soundnessbench.yaml")).unwrap();
    assert_eq!(
        soundnessbench.bab.phase_budget.initial_bounds_fraction,
        Some(0.15),
        "soundnessbench should cap initial_bounds_fraction to 0.15"
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &soundnessbench).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "soundnessbench initial_bounds_fraction should propagate to config"
    );

    let cifar100 = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(
        cifar100.bab.phase_budget.initial_bounds_fraction,
        Some(0.15),
        "cifar100_2024 should cap initial_bounds_fraction to 0.15"
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar100).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "cifar100_2024 initial_bounds_fraction should propagate to config"
    );
    assert_eq!(
        cifar100.bab.phase_budget.mip_min_fraction,
        Some(0.0),
        "cifar100_2024 must not reserve BaB time for its ineligible 44M-NNZ Graph-MIP"
    );
    assert_eq!(
        cifar100.bab.phase_budget.mip_min_secs,
        Some(0),
        "cifar100_2024 must not impose a hidden minimum MIP reservation"
    );
    assert_eq!(
        config.phase_budget.mip_min_fraction, 0.0,
        "the zero MIP fraction should propagate to the runtime ledger"
    );
    assert_eq!(
        config.phase_budget.mip_min_secs, 0,
        "the zero MIP floor should propagate to the runtime ledger"
    );
    assert!(
        !config.phase_budget.requests_mip_reservation(),
        "CIFAR's zero policy must decline both the time reserve and root-bounds stash"
    );
}

/// #graph-mip-admission-refund: tinyimagenet_2024 mirrors cifar100_2024's zero
/// MIP reservation. Its single shipped model has the SAME op multiset as
/// CIFAR100_resnet_medium at 3.06x the spatial extent, so it is inside the
/// encoder's layer set (no static #deadlane disarm) yet ~56x over the 5M encode-
/// nnz admission cap — the reservation is unreachable dead time on all 200 rows.
#[test]
fn tinyimagenet_preset_declines_the_unreachable_graph_mip_reservation() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();
    assert_eq!(
        preset.bab.phase_budget.mip_min_fraction,
        Some(0.0),
        "tinyimagenet_2024 must not reserve BaB time for its ineligible whole-net Graph-MIP"
    );
    assert_eq!(
        preset.bab.phase_budget.mip_min_secs,
        Some(0),
        "tinyimagenet_2024 must not impose a hidden minimum MIP reservation"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.phase_budget.mip_min_fraction, 0.0,
        "the zero MIP fraction should propagate to the runtime ledger"
    );
    assert_eq!(
        config.phase_budget.mip_min_secs, 0,
        "the zero MIP floor should propagate to the runtime ledger"
    );
    assert!(
        !config.phase_budget.requests_mip_reservation(),
        "tinyimagenet's zero policy must decline the time reserve, the AUTO whole-net \
         escalation, and the root-bounds stash"
    );

    // Nothing else in the stanza moved: the two keys are the ONLY delta.
    assert_eq!(
        preset.bab.phase_budget.initial_bounds_fraction,
        Some(0.15),
        "tinyimagenet_2024 keeps its warmup cap"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_fraction,
        Some(0.40),
        "tinyimagenet_2024 keeps the measured TinyImageNet falsification fraction; \
         the cifar100 0.05 transfer regressed the official-budget sample"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_max_secs,
        Some(30),
        "tinyimagenet_2024 keeps its absolute PGD ceiling"
    );
    assert_eq!(
        preset.bab.phase_budget.post_bab_pgd_fraction,
        Some(0.10),
        "tinyimagenet_2024 keeps its post-BaB falsification reserve"
    );
    assert_eq!(
        preset.bab.phase_budget.attack_extension_fraction,
        Some(0.0),
        "tinyimagenet_2024 keeps the attack extension disabled"
    );
}

/// No OTHER shipped category may drift to a zero reservation as a side effect of
/// the tinyimagenet stanza: cifar100 and tinyimagenet are the only two presets
/// that decline, and every preset carrying an explicit nonzero MIP policy keeps
/// it. Presets that mention no MIP key at all inherit the batteries-included
/// default (asserted separately in `PhaseBudgetConfig::default`'s own test).
#[test]
fn only_cifar100_and_tinyimagenet_decline_the_mip_reservation() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let declining = ["cifar100_2024.yaml", "tinyimagenet_2024.yaml"];
    let mut seen_declines = Vec::new();

    for dir in ["configs/vnncomp25", "configs/vnncomp26"] {
        let entries = std::fs::read_dir(repo_root.join(dir)).unwrap();
        for entry in entries {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap()
                .to_string();
            let preset = load_preset(&path).unwrap();
            let mut config = BetaCrownConfig::default();
            apply_preset(&mut config, &preset).unwrap();
            if config.phase_budget.requests_mip_reservation() {
                continue;
            }
            assert!(
                declining.contains(&file_name.as_str()),
                "{dir}/{file_name} unexpectedly declines the MIP reservation"
            );
            seen_declines.push(file_name);
        }
    }

    seen_declines.sort();
    assert_eq!(
        seen_declines,
        vec![
            "cifar100_2024.yaml".to_string(),
            "tinyimagenet_2024.yaml".to_string()
        ],
        "exactly the two ResNet categories with an over-cap whole-net encode decline"
    );
}

/// The preset key is the gate here (there is no env var), so the malformed-value
/// case is a malformed YAML scalar: it must FAIL the load rather than silently
/// arming the refund. A stanza that is absent leaves the batteries-included
/// default untouched.
#[test]
fn malformed_mip_reservation_keys_do_not_arm_the_refund() {
    let dir = tempfile::tempdir().expect("tempdir");

    for bad in [
        "bab:\n  phase_budget:\n    mip_min_fraction: not_a_number\n",
        "bab:\n  phase_budget:\n    mip_min_secs: \"0\"\n",
        "bab:\n  phase_budget:\n    mip_min_secs: -1\n",
        "bab:\n  phase_budget:\n    mip_min_fraction: [0.0]\n",
    ] {
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, bad).expect("write");
        assert!(
            load_preset(&path).is_err(),
            "malformed preset must be rejected, not silently applied: {bad:?}"
        );
    }

    // Absent stanza => the default reservation survives untouched.
    let path = dir.path().join("silent.yaml");
    std::fs::write(&path, "bab:\n  batch_size: 128\n").expect("write");
    let preset = load_preset(&path).expect("valid preset");
    assert!(preset.bab.phase_budget.mip_min_fraction.is_none());
    assert!(preset.bab.phase_budget.mip_min_secs.is_none());
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.phase_budget.requests_mip_reservation(),
        "a preset without the stanza must keep the default whole-net MIP reservation"
    );
}

#[test]
fn nn4sys_and_safenlp_retain_nonzero_mip_reservation_admission() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (name, path) in [
        ("nn4sys/MSCN", "configs/vnncomp25/nn4sys.yaml"),
        ("safenlp", "configs/vnncomp26/safenlp_2024.yaml"),
    ] {
        let preset = load_preset(&repo_root.join(path)).unwrap();
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert!(
            config.phase_budget.requests_mip_reservation(),
            "{name} must retain its existing whole-net MIP reservation and stash"
        );
    }
}

/// #2206 Packet E: post_bab_pgd_fraction flows through preset system.
#[test]
fn post_bab_pgd_fraction_preset_override_2206() {
    let preset = PresetConfig {
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                post_bab_pgd_fraction: Some(0.25),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    // Default is 0.10 (matches reference competition configs)
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
        "default post_bab_pgd_fraction should be 0.10"
    );

    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.25).abs() < 1e-6,
        "post_bab_pgd_fraction should be overridden to 0.25"
    );
}

/// #2206 Packet E: post_bab_pgd_fraction deserializes from YAML.
#[test]
fn post_bab_pgd_fraction_yaml_deserialization_2206() {
    let yaml = r#"
bab:
  phase_budget:
    initial_bounds_fraction: 0.15
    post_bab_pgd_fraction: 0.10
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(preset.bab.phase_budget.post_bab_pgd_fraction, Some(0.10));
    assert_eq!(preset.bab.phase_budget.initial_bounds_fraction, Some(0.15));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
        "YAML post_bab_pgd_fraction should flow to config"
    );
}

#[test]
fn invalid_post_bab_fraction_is_rejected_before_preset_mutation() {
    for value in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let preset = PresetConfig {
            bab: BabPreset {
                batch_size: Some(999),
                phase_budget: PhaseBudgetPreset {
                    post_bab_pgd_fraction: Some(value),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig::default();
        let original_batch_size = config.batch_size;
        let original_fraction = config.phase_budget.post_bab_pgd_fraction;
        let error = apply_preset(&mut config, &preset)
            .expect_err("invalid duration fraction must fail preset application");
        assert!(
            error.to_string().contains("post_bab_pgd_fraction"),
            "field-specific diagnostic expected for {value}, got {error}"
        );
        assert_eq!(
            config.batch_size, original_batch_size,
            "phase validation must run before unrelated preset mutation"
        );
        assert_eq!(
            config.phase_budget.post_bab_pgd_fraction, original_fraction,
            "invalid phase policy must not be partially installed"
        );
    }
}

#[test]
fn finite_post_bab_fraction_preserves_local_clamp_compatibility() {
    for value in [-0.01, 0.51] {
        let preset = PresetConfig {
            bab: BabPreset {
                phase_budget: PhaseBudgetPreset {
                    post_bab_pgd_fraction: Some(value),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset)
            .expect("finite legacy values must retain engine-local clamp semantics");
        assert_eq!(
            config.phase_budget.post_bab_pgd_fraction, value,
            "preset application preserves the raw finite value for existing consumers"
        );
    }
}

/// #2206 Packet E: VNN-COMP presets set post_bab_pgd_fraction for PGD reservation.
#[test]
fn vnncomp_presets_set_post_bab_pgd_fraction_2206() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    for (name, path) in [
        ("soundnessbench", "configs/vnncomp25/soundnessbench.yaml"),
        ("cifar100_2024", "configs/vnncomp25/cifar100_2024.yaml"),
        (
            "tinyimagenet_2024",
            "configs/vnncomp25/tinyimagenet_2024.yaml",
        ),
    ] {
        let preset = load_preset(&repo_root.join(path)).unwrap();
        assert_eq!(
            preset.bab.phase_budget.post_bab_pgd_fraction,
            Some(0.10),
            "{name} should set post_bab_pgd_fraction to 0.10"
        );

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert!(
            (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
            "{name} post_bab_pgd_fraction should propagate to config"
        );
    }
}

/// #2206 Packet E: tinyimagenet_2024 preset caps initial_bounds_fraction.
#[test]
fn tinyimagenet_preset_caps_initial_bounds_fraction_2206() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();
    assert_eq!(
        preset.bab.phase_budget.initial_bounds_fraction,
        Some(0.15),
        "tinyimagenet_2024 should cap initial_bounds_fraction to 0.15"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "tinyimagenet_2024 initial_bounds_fraction should propagate to config"
    );
}

/// Sealed row-12 evidence promotes the canonical cersyve preset to a proof-only
/// tail while leaving its optional GPU-BaB sidecar unchanged.
#[test]
fn cersyve_main_uses_measured_proof_tail_while_gpu_sidecar_stays_unchanged() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    assert_eq!(preset.bab.phase_budget.initial_bounds_fraction, Some(0.15));
    assert_eq!(preset.bab.phase_budget.post_bab_pgd_fraction, Some(0.0));
    assert_eq!(
        preset.bab.phase_budget.vnncomp_post_bab_attack,
        Some(false),
        "canonical cersyve must disable the independent outer tail attack explicitly"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!((config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6);
    assert_eq!(config.phase_budget.post_bab_pgd_fraction, 0.0);

    let sidecar_path = "configs/vnncomp25/cersyve_gpu_bab.yaml";
    assert_phase_budget_pair("cersyve_gpu_bab", sidecar_path, 0.15);
    let sidecar = load_preset(&repo_root.join(sidecar_path)).unwrap();
    assert_eq!(
        sidecar.bab.phase_budget.vnncomp_post_bab_attack, None,
        "the unmeasured sidecar must retain its existing implicit wrapper policy"
    );
}

/// #4283 regression: lsnc_relu disables the alpha-CROWN warmup cap in both the
/// canonical preset and the standalone GPU-BaB sidecar.
#[test]
fn lsnc_relu_presets_disable_warmup_cap_4283() {
    for (name, path) in [
        ("lsnc_relu", "configs/vnncomp25/lsnc_relu.yaml"),
        (
            "lsnc_relu_gpu_bab",
            "configs/vnncomp25/lsnc_relu_gpu_bab.yaml",
        ),
    ] {
        assert_phase_budget_pair(name, path, 1.0);
    }
}

/// #2206: default PhaseBudgetPreset (all None) does not alter config defaults.
#[test]
fn empty_phase_budget_preset_preserves_defaults_2206() {
    let preset = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    let before = config.phase_budget.clone();
    apply_preset(&mut config, &preset).unwrap();

    assert!(
        (config.phase_budget.initial_bounds_fraction - before.initial_bounds_fraction).abs() < 1e-6,
        "empty preset should not alter initial_bounds_fraction"
    );
    assert!(
        (config.phase_budget.upfront_pgd_fraction - before.upfront_pgd_fraction).abs() < 1e-6,
        "empty preset should not alter upfront_pgd_fraction"
    );
    assert_eq!(
        config.phase_budget.mip_min_secs, before.mip_min_secs,
        "empty preset should not alter mip_min_secs"
    );
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - before.post_bab_pgd_fraction).abs() < 1e-6,
        "empty preset should not alter post_bab_pgd_fraction"
    );
}

/// #attack-floor: the lsnc_relu preset must carry the disjunctive-attack floor
/// all the way into the runtime ledger, and no other category may acquire it by
/// accident.
///
/// MEASURED (official 25s instance budget => 20s internal, preset confirmed in
/// the log): with the tiny-budget cap alone the disjunctive attack gets 3.0s and
/// `quadrotor2d_state_34` returns `unknown` 3/3; with the 5s floor it returns
/// an ORT-confirmed `sat` at ~4.9s wall 3/3. BaB still keeps 15s of the 20s.
#[test]
fn lsnc_preset_carries_the_disjunctive_attack_floor() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_min_secs,
        Some(8),
        "lsnc_relu must floor its disjunctive attack at 8s: the 15% tiny-budget \
         cap gives 3.0s and the state_34 counterexample lands at ~4.54s (measured \
         cliff: 4s => unknown, 5s => sat; 8s is the headroom choice)"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.phase_budget.disjunctive_pgd_min_secs,
        Some(8),
        "the floor must reach the runtime phase-budget ledger"
    );

    // Opt-in only: the default policy is untouched for every other category.
    assert_eq!(
        ny_propagate::PhaseBudgetConfig::default().disjunctive_pgd_min_secs,
        None,
        "the attack floor must stay off by default"
    );
}

/// #attack-anchor: the falsification slice can be anchored at the PHASE start
/// instead of the ledger start, the knob is plumbed end to end, and it is
/// default-OFF so no unmeasured category can move.
///
/// The defect it repairs, MEASURED on `cifar100_2024` `CIFAR100_resnet_large`
/// at the official 100 s budget with the shipped `disjunctive_pgd_fraction:
/// 0.05` — the `[pgd-vjp-disj]` diagnostic reports
/// `deadline (0.1s): wave_steps=0`: the batched exact-VJP falsifier received
/// **0.1 s of its 5 s slice and took ZERO steps**, because the ledger-start
/// anchor had already charged ~4.9 s of model load / graph build / VNN-LIB
/// parse against the attack's own budget. Same defect class as #four-walls'
/// CROWN-precheck repair, one phase over.
#[test]
fn attack_anchor_is_plumbed_and_default_off() {
    let preset = PresetConfig {
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                disjunctive_pgd_from_phase_start: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.phase_budget.disjunctive_pgd_from_phase_start,
        "the phase-start anchor must reach the runtime phase-budget ledger"
    );

    assert!(
        !ny_propagate::PhaseBudgetConfig::default().disjunctive_pgd_from_phase_start,
        "the falsification slice must stay ledger-anchored by default"
    );
}

/// #attack-stall (design S4): the adaptive attack cutoff is plumbed end to end
/// but ships INERT, and no shipped preset arms it.
///
/// It stays off because both errors are measured and only one has a measured
/// gain — which is zero. Reclaiming the slice converted NOTHING on the GT-unsat
/// oval21 rows it was designed for (`docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md`
/// S4: `unknown 57 s` in both arms on all three rows; `relusplitter.yaml`'s
/// #reclaim-pgd note repeats it). Cutting too eagerly costs rows: on
/// tinyimagenet this same disjunctive-PGD lane is what FINDS the sat rows, at
/// 12.35 s / 17.67 s / 20.39 s of a 100 s budget (b61b5f10, 8/15 -> 3/15 when a
/// sibling's allocation was ported in). Arming it for a category therefore
/// needs an A/B for THAT category covering its sat rows — which is what
/// `NY_ATTACK_STALL_WINDOW` exists for.
#[test]
fn attack_stall_cutoff_is_plumbed_but_armed_nowhere() {
    // The knob reaches the runtime ledger when a preset does set it.
    let preset = PresetConfig {
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                disjunctive_pgd_stall_window_fraction: Some(0.5),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.phase_budget.disjunctive_pgd_stall_window_fraction,
        Some(0.5),
        "the stall window fraction must reach the runtime phase-budget ledger"
    );

    // Sealed default: inert.
    assert_eq!(
        ny_propagate::PhaseBudgetConfig::default().disjunctive_pgd_stall_window_fraction,
        None,
        "the adaptive attack cutoff must stay off by default"
    );

    // And no shipped preset arms it — the A/B that would license one has not
    // been run for any category.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for dir in ["configs/vnncomp25", "configs/vnncomp26"] {
        let entries = std::fs::read_dir(repo_root.join(dir)).unwrap();
        for entry in entries {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let preset = load_preset(&path).unwrap();
            assert_eq!(
                preset
                    .bab
                    .phase_budget
                    .disjunctive_pgd_stall_window_fraction,
                None,
                "{} must not arm the adaptive attack cutoff: no category has an A/B \
                 for it, and the lane it cuts is what finds tinyimagenet's sat rows",
                path.display()
            );
        }
    }
}
