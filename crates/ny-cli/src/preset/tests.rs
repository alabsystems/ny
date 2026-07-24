// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::apply::{apply_clip_preset, apply_preset, parse_reduce_op};
use super::branching::parse_branching_method;
use super::*;
use ny_propagate::{BetaCrownConfig, BranchingHeuristic, InputClipType, KfsbReduceOp};
use std::path::Path;
use tempfile::tempdir;

/// Determine if PGD attack should be enabled based on preset.
///
/// Returns `Some(false)` for "skip"/"none"/"disabled", `Some(true)` for "before"/"middle"/"after".
fn should_enable_pgd(preset: &PresetConfig) -> Option<bool> {
    preset
        .attack
        .pgd_order
        .as_ref()
        .map(|order| !matches!(order.to_lowercase().as_str(), "skip" | "none" | "disabled"))
}

#[test]
fn parse_branching_methods() {
    assert!(matches!(
        parse_branching_method("width").unwrap(),
        BranchingHeuristic::LargestBoundWidth
    ));
    assert!(matches!(
        parse_branching_method("kfsb").unwrap(),
        BranchingHeuristic::Kfsb
    ));
    assert!(matches!(
        parse_branching_method("kfsb-intercept-only").unwrap(),
        BranchingHeuristic::KfsbInterceptOnly
    ));
    assert!(matches!(
        parse_branching_method("fsb").unwrap(),
        BranchingHeuristic::FilteredSmartBranching
    ));
    assert!(matches!(
        parse_branching_method("input").unwrap(),
        BranchingHeuristic::InputSplit
    ));
}

#[test]
fn resolve_branching_uses_input_split_from_preset() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("input".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let branching = resolve_branching(&preset)
        .unwrap()
        .expect("preset branching should resolve");
    assert!(matches!(
        branching.heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        !branching.use_relu_split,
        "input branching should not enable relu split"
    );
}

#[test]
fn resolve_branching_keeps_fsb_and_kfsb_distinct_2370() {
    for (method, expected) in [
        ("fsb", BranchingHeuristic::FilteredSmartBranching),
        ("kfsb", BranchingHeuristic::Kfsb),
        ("kfsb-intercept-only", BranchingHeuristic::KfsbInterceptOnly),
    ] {
        let preset = PresetConfig {
            bab: BabPreset {
                branching: BranchingPreset {
                    method: Some(method.to_string()),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };

        let branching = resolve_branching(&preset)
            .unwrap()
            .expect("preset branching should resolve");
        assert_eq!(
            branching.heuristic, expected,
            "preset method '{method}' should resolve to the documented heuristic"
        );
        // kFSB, FSB, and BoundImpact are ReLU-splitting strategies: they select
        // which ReLU neuron to pin active/inactive. They must route through the
        // graph ReLU-split path. (#4300)
        assert!(
            branching.use_relu_split,
            "preset method '{method}' should set use_relu_split=true (ReLU-splitting strategy)"
        );
    }
}

#[test]
fn resolve_branching_uses_relu_split_from_preset() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("relu".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let branching = resolve_branching(&preset)
        .unwrap()
        .expect("preset branching should resolve");
    assert!(matches!(
        branching.heuristic,
        BranchingHeuristic::LargestBoundWidth
    ));
    assert!(
        branching.use_relu_split,
        "relu branching should enable relu split"
    );
}

#[test]
fn apply_preset_sets_solver_build_batch_size_4354() {
    let preset = PresetConfig {
        solver: SolverPreset {
            build_batch_size: Some(128),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();

    apply_preset(&mut config, &preset).expect("solver build_batch_size should apply cleanly");

    assert_eq!(
        config.build_batch_size,
        Some(128),
        "solver.build_batch_size should map onto BetaCrownConfig::build_batch_size"
    );
}

#[test]
fn resolve_branching_uses_input_split_when_enabled_without_method() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    enable: Some(true),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let branching = resolve_branching(&preset)
        .unwrap()
        .expect("input_split.enable should resolve preset branching");
    assert!(matches!(
        branching.heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        !branching.use_relu_split,
        "input split presets should not enable relu split"
    );
}

#[test]
fn resolve_branching_maps_alpha_beta_crown_sb_input_split() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("sb".to_string()),
                input_split: InputSplitPreset {
                    enable: Some(true),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let branching = resolve_branching(&preset)
        .unwrap()
        .expect("sb + input_split.enable should resolve");
    assert!(matches!(
        branching.heuristic,
        BranchingHeuristic::InputSplit
    ));
}

#[test]
fn parse_branching_method_rejects_unknown() {
    let err = parse_branching_method("ksfb").unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("ksfb"), "error should mention the typo: {msg}");
}

#[test]
fn parse_reduce_ops() {
    assert!(matches!(parse_reduce_op("min").unwrap(), KfsbReduceOp::Min));
    assert!(matches!(parse_reduce_op("max").unwrap(), KfsbReduceOp::Max));
    assert!(matches!(
        parse_reduce_op("mean").unwrap(),
        KfsbReduceOp::Mean
    ));
}

#[test]
fn parse_reduce_op_rejects_unknown() {
    let err = parse_reduce_op("average").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("average"),
        "error should mention the typo: {msg}"
    );
}

#[test]
fn load_preset_yaml() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("test.yaml");
    fs::write(
        &config_path,
        r#"
general:
  root_path: ./data
bab:
  batch_size: 256
  branching:
    method: kfsb
    candidates: 7
"#,
    )
    .unwrap();

    let preset = load_preset(&config_path).unwrap();
    assert_eq!(preset.bab.batch_size, Some(256));
    assert_eq!(preset.bab.branching.method, Some("kfsb".to_string()));
    assert_eq!(preset.bab.branching.candidates, Some(7));
}

#[test]
fn model_onnx_optimization_flag_parses_single_string() {
    let preset: PresetConfig = serde_yaml::from_str(
        r#"
model:
  onnx_optimization_flags: merge_linear
"#,
    )
    .expect("single string flag should parse");

    assert_eq!(preset.model.onnx_optimization_flags, vec!["merge_linear"]);

    let flags = resolve_onnx_optimization_flags(&preset).expect("flag should resolve");
    assert_eq!(flags, vec![ny_onnx::OnnxOptimizationFlag::MergeLinear]);
}

#[test]
fn model_onnx_optimization_flag_parses_yaml_sequence() {
    let preset: PresetConfig = serde_yaml::from_str(
        r#"
model:
  onnx_optimization_flags:
    - merge_linear
"#,
    )
    .expect("sequence flag should parse");

    assert_eq!(preset.model.onnx_optimization_flags, vec!["merge_linear"]);
}

#[test]
fn build_onnx_load_config_enables_merge_linear() {
    let preset: PresetConfig = serde_yaml::from_str(
        r#"
model:
  onnx_optimization_flags: merge_linear
"#,
    )
    .expect("preset should parse");

    let config = build_onnx_load_config(&preset).expect("config should build");
    assert!(
        config.has_optimization_flag(ny_onnx::OnnxOptimizationFlag::MergeLinear),
        "merge_linear should be enabled on the loader config"
    );
}

#[test]
fn unsupported_onnx_optimization_flag_is_rejected() {
    let preset: PresetConfig = serde_yaml::from_str(
        r#"
model:
  onnx_optimization_flags: cache_onnx_conversion
"#,
    )
    .expect("preset should parse");

    let err = resolve_onnx_optimization_flags(&preset).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cache_onnx_conversion"),
        "error should name the unsupported flag: {msg}"
    );
}

#[test]
fn vnncomp25_cersyve_and_lsnc_relu_presets_cap_pgd_restarts() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    assert_eq!(cersyve.attack.pgd_order.as_deref(), Some("before"));
    assert_eq!(cersyve.attack.pgd_restarts, Some(100));

    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    assert_eq!(lsnc_relu.attack.pgd_order.as_deref(), Some("before"));
    assert_eq!(lsnc_relu.attack.pgd_restarts, Some(1000));
}

#[test]
fn relusplitter_rsplitter_gpu_bab_sidecar_stays_isolated_3862() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let main = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml")).unwrap();
    let sidecar =
        load_preset(&repo_root.join("configs/vnncomp25/relusplitter_rsplitter_gpu_bab.yaml"))
            .unwrap();

    assert_eq!(main.bab.branching.method.as_deref(), Some("kfsb"));
    assert_eq!(sidecar.bab.branching.method.as_deref(), Some("babsr"));
    assert_eq!(sidecar.attack.pgd_order.as_deref(), Some("middle"));
    assert_eq!(sidecar.bab.batch_size, Some(4));

    let resolved = resolve_branching(&sidecar)
        .unwrap()
        .expect("sidecar preset should resolve a branching mode");
    assert!(matches!(
        resolved.heuristic,
        BranchingHeuristic::BoundImpact
    ));
    // BaBSR (BoundImpact) is a ReLU-splitting strategy and now correctly
    // sets use_relu_split=true (#4300). With gpu_bab=true the dispatch
    // short-circuits to verify_graph_gpu_domain_list regardless.
    assert!(
        resolved.use_relu_split,
        "BaBSR should set use_relu_split=true (ReLU-splitting strategy, #4300)"
    );
}

/// Regression: the #3813 cut-aware WGPU multi-objective sidecar must explicitly
/// request a wgpu device, keep kfsb branching, and enable cuts. Without the
/// explicit device field the benchmark runner silently falls back to CPU.
#[test]
fn relusplitter_multiobjective_wgpu_sidecar_requests_wgpu_device_3813() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let sidecar = load_preset(
        &repo_root.join("configs/vnncomp25/relusplitter_rsplitter_multiobjective_wgpu.yaml"),
    )
    .unwrap();

    // Must explicitly request wgpu so the benchmark runner cannot silently fall
    // back to CPU (the root cause of the ambiguous #3862 negative).
    assert_eq!(
        sidecar.general.device.as_deref(),
        Some("wgpu"),
        "sidecar must explicitly set device: wgpu"
    );

    // Must keep the shared multi-objective kfsb branching (the only
    // measured-positive relusplitter lane) — not babsr.
    assert_eq!(
        sidecar.bab.branching.method.as_deref(),
        Some("kfsb"),
        "sidecar must keep kfsb branching for shared multi-objective path"
    );

    // Must enable cuts (the cuts-enabled lane is the only one that moved beyond
    // domains=0).
    assert_eq!(
        sidecar.bab.cuts.enabled,
        Some(true),
        "sidecar must enable cuts"
    );

    // conv_mode: auto routes through matrix path when cuts enabled.
    assert_eq!(
        sidecar.general.conv_mode,
        Some(ny_propagate::ConvMode::Auto),
        "sidecar must use conv_mode: auto"
    );
}

#[test]
fn apply_preset_to_config() {
    let preset = PresetConfig {
        bab: BabPreset {
            batch_size: Some(128),
            branching: BranchingPreset {
                method: Some("kfsb".to_string()),
                candidates: Some(10),
                reduceop: Some("max".to_string()),
                ..Default::default()
            },
            beta_crown: BetaCrownPreset {
                lr_beta: Some(0.2),
                iterations: Some(15),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert_eq!(config.batch_size, 128);
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::Kfsb
    ));
    assert_eq!(config.fsb_candidates, 10);
    assert!(matches!(config.kfsb_reduce_op, KfsbReduceOp::Max));
    assert_eq!(config.beta_lr, 0.2);
    assert_eq!(config.beta_iterations, 15);
}

/// #kfsb-multi: `bab.branching.kfsb_multi` arms `use_kfsb_multi_branching`, and
/// a preset that omits it leaves the field at its default (false). Guards the
/// cifar100-scoped opt-in so no other preset can silently arm the wave lane.
#[test]
fn apply_preset_propagates_kfsb_multi_arming() {
    // Opt-in preset arms the field.
    let armed = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("kfsb".to_string()),
                candidates: Some(7),
                reduceop: Some("max".to_string()),
                kfsb_multi: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).unwrap();
    assert!(
        config.use_kfsb_multi_branching,
        "bab.branching.kfsb_multi: true must arm use_kfsb_multi_branching"
    );

    // A preset that omits kfsb_multi must leave the default (false) untouched.
    let unset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("kfsb".to_string()),
                candidates: Some(7),
                reduceop: Some("max".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &unset).unwrap();
    assert!(
        !config.use_kfsb_multi_branching,
        "a preset without kfsb_multi must leave use_kfsb_multi_branching at its default (false)"
    );
}

/// #cifar-head-crown: all three typed preset values must reach the verifier
/// config, while omission leaves the pass off with bounded inert defaults.
#[test]
fn apply_preset_propagates_root_crown_interm_dense_head_policy() {
    let preset = PresetConfig {
        bab: BabPreset {
            root_crown_interm_dense_head: Some(true),
            root_crown_interm_max_secs: Some(4),
            root_crown_interm_max_dim: Some(777),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(config.root_crown_interm_dense_head);
    assert_eq!(config.root_crown_interm_max_secs, 4);
    assert_eq!(config.root_crown_interm_max_dim, 777);

    let mut omitted = BetaCrownConfig::default();
    apply_preset(&mut omitted, &PresetConfig::default()).unwrap();
    assert!(!omitted.root_crown_interm_dense_head);
    assert_eq!(omitted.root_crown_interm_max_secs, 2);
    assert_eq!(omitted.root_crown_interm_max_dim, 512);
}

#[test]
fn apply_preset_propagates_root_sparse_interm_crown_policy() {
    let preset = PresetConfig {
        bab: BabPreset {
            root_sparse_interm_crown: Some(true),
            root_sparse_interm_crown_max_secs: Some(3),
            root_sparse_interm_crown_max_dim: Some(4_096),
            root_sparse_interm_crown_max_rows: Some(96),
            root_sparse_interm_crown_max_targets: Some(2),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(config.root_sparse_interm_crown);
    assert_eq!(config.root_sparse_interm_crown_max_secs, 3);
    assert_eq!(config.root_sparse_interm_crown_max_dim, 4_096);
    assert_eq!(config.root_sparse_interm_crown_max_rows, 96);
    assert_eq!(config.root_sparse_interm_crown_max_targets, 2);

    let mut omitted = BetaCrownConfig::default();
    apply_preset(&mut omitted, &PresetConfig::default()).unwrap();
    assert!(!omitted.root_sparse_interm_crown);
    assert_eq!(omitted.root_sparse_interm_crown_max_secs, 2);
    assert_eq!(omitted.root_sparse_interm_crown_max_dim, 8_192);
    assert_eq!(omitted.root_sparse_interm_crown_max_rows, 512);
    assert_eq!(omitted.root_sparse_interm_crown_max_targets, 4);
}

/// #kfsb-multi: the SHIPPED cifar100 presets arm the wave-batched selector,
/// while a representative non-cifar preset (acasxu) leaves it off. This is the
/// cifar100-scoping guarantee: only these two presets flip the field.
#[test]
fn cifar100_presets_arm_kfsb_multi_others_do_not() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    for rel in [
        "configs/vnncomp25/cifar100_2024.yaml",
        "configs/vnncomp24/cifar100.yaml",
    ] {
        let preset = load_preset(&repo_root.join(rel)).unwrap();
        assert_eq!(
            preset.bab.branching.kfsb_multi,
            Some(true),
            "{rel} must set bab.branching.kfsb_multi: true"
        );
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert!(
            config.use_kfsb_multi_branching,
            "{rel} must arm use_kfsb_multi_branching"
        );
    }

    // Non-cifar preset (acasxu) must NOT arm the wave lane.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(
        acasxu.bab.branching.kfsb_multi, None,
        "acasxu preset must not set kfsb_multi"
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &acasxu).unwrap();
    assert!(
        !config.use_kfsb_multi_branching,
        "acasxu preset must leave use_kfsb_multi_branching off (cifar100-scoped)"
    );
}

#[test]
fn apply_preset_propagates_input_split_sb_tuning() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("input".to_string()),
                input_split: InputSplitPreset {
                    sb_coeff_thresh: Some(1.0e-2),
                    touch_zero_score: Some(0.1),
                    sb_margin_weight: Some(0.75),
                    sb_sum: Some(true),
                    sb_primary_spec: Some(2),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert_eq!(config.input_split_coeff_thresh, 1.0e-2);
    assert_eq!(config.input_split_touch_zero_score, 0.1);
    assert_eq!(config.input_split_sb_margin_weight, 0.75);
    assert!(config.input_split_sb_sum);
    assert_eq!(config.input_split_sb_primary_spec, Some(2));
}

#[test]
fn apply_preset_propagates_input_split_alpha_iteration() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("input".to_string()),
                input_split: InputSplitPreset {
                    alpha_iteration: Some(5),
                    lr_alpha: Some(0.07),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert_eq!(config.input_split_alpha_iteration, 5);
    assert_eq!(config.input_split_lr_alpha, 0.07);
}

#[test]
fn input_split_alpha_iteration_serde_alias_parses() {
    // Field name `alpha_iteration` plus the alpha-beta-CROWN alias
    // `input_split_alpha_iteration` must both deserialize.
    let yaml = r#"
bab:
  branching:
    method: input
    input_split:
      input_split_alpha_iteration: 5
      input_split_lr_alpha: 0.07
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(
        preset.bab.branching.input_split.alpha_iteration,
        Some(5),
        "input_split_alpha_iteration alias should map to alpha_iteration"
    );
    assert_eq!(
        preset.bab.branching.input_split.lr_alpha,
        Some(0.07),
        "input_split_lr_alpha alias should map to lr_alpha"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_alpha_iteration, 5);
    assert_eq!(config.input_split_lr_alpha, 0.07);
}

#[test]
fn input_split_alpha_iteration_defaults_to_zero_when_absent() {
    // Absent preset field => config retains the default (0 = off).
    let yaml = r#"
bab:
  branching:
    method: input
    input_split:
      depth: 2
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(preset.bab.branching.input_split.alpha_iteration, None);
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_alpha_iteration, 0);
}

#[test]
fn per_disjunct_alpha_nested_preset_reaches_beta_crown_config() {
    let preset: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  beta_crown:
    optimize_disjuncts_separately: true
"#,
    )
    .expect("nested per-disjunct alpha preset should parse");

    assert_eq!(
        preset.bab.beta_crown.optimize_disjuncts_separately,
        Some(true)
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("nested per-disjunct alpha preset should apply");
    assert!(
        config.optimize_disjuncts_separately,
        "bab.beta_crown.optimize_disjuncts_separately must arm the live verifier config"
    );
}

#[test]
fn misplaced_per_disjunct_alpha_preset_is_rejected_narrowly() {
    let error = serde_yaml::from_str::<PresetConfig>(
        r#"
bab:
  optimize_disjuncts_separately: true
"#,
    )
    .expect_err("the misplaced sibling key must not be silently ignored");
    let message = error.to_string();
    assert!(
        message.contains("bab.optimize_disjuncts_separately")
            && message.contains("bab.beta_crown.optimize_disjuncts_separately"),
        "error should identify both the misplaced and correct paths: {message}"
    );
}

#[test]
fn input_split_warm_parallel_is_preset_scoped_and_defaults_off() {
    let absent: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  branching:
    method: input
    input_split:
      reorder_bab: true
"#,
    )
    .unwrap();
    let mut absent_config = BetaCrownConfig::default();
    apply_preset(&mut absent_config, &absent).unwrap();
    assert_eq!(absent.bab.branching.input_split.warm_parallel, None);
    assert!(!absent_config.input_split_warm_parallel);

    let enabled: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  branching:
    method: input
    input_split:
      reorder_bab: true
      warm_parallel: true
"#,
    )
    .unwrap();
    let mut enabled_config = BetaCrownConfig::default();
    apply_preset(&mut enabled_config, &enabled).unwrap();
    assert_eq!(enabled.bab.branching.input_split.warm_parallel, Some(true));
    assert!(enabled_config.reorder_bab);
    assert!(enabled_config.input_split_warm_parallel);
}

#[test]
fn pgd_order_parsing() {
    let preset_with_skip = PresetConfig {
        attack: AttackPreset {
            pgd_order: Some("skip".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(should_enable_pgd(&preset_with_skip), Some(false));

    let preset_with_before = PresetConfig {
        attack: AttackPreset {
            pgd_order: Some("before".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(should_enable_pgd(&preset_with_before), Some(true));
}

/// attack.pgd_order decodes to enablement only: the skip family disables,
/// before/input_bab/middle/after enable (middle/after warn — reference
/// alpha-beta-CROWN scheduling is not implemented), and unknown values are
/// rejected like attack_mode instead of silently enabling PGD.
#[test]
fn apply_preset_decodes_pgd_order_enablement() {
    for (order, expected) in [
        ("skip", false),
        ("none", false),
        ("disabled", false),
        ("before", true),
        ("input_bab", true),
        ("middle", true),
        ("after", true),
    ] {
        let preset = PresetConfig {
            attack: AttackPreset {
                pgd_order: Some(order.to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert_eq!(
            config.enable_pgd_attack, expected,
            "pgd_order '{order}' should decode to enable_pgd_attack={expected}"
        );
    }

    let typo = PresetConfig {
        attack: AttackPreset {
            pgd_order: Some("skpi".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    assert!(
        apply_preset(&mut config, &typo).is_err(),
        "unknown pgd_order values must be rejected, not silently treated as enabled"
    );
}

/// Test that solver section values are merged into config,
/// and bab section overrides solver values.
///
/// This is critical for alpha-beta-CROWN compatibility where
/// batch_size and crown settings are under solver: section.
#[test]
fn solver_section_merging() {
    // Preset with both solver: and bab: sections having overlapping values
    let preset = PresetConfig {
        solver: SolverPreset {
            batch_size: Some(64), // Should be overridden by bab.batch_size
            alpha_crown: AlphaCrownPreset {
                lr_alpha: Some(0.1),
                iterations: Some(10),
                ..Default::default()
            },
            beta_crown: BetaCrownPreset {
                lr_alpha: Some(0.05), // Should be overridden by bab.beta_crown
                lr_beta: Some(0.15),
                iterations: Some(8),
                ..Default::default()
            },
            ..Default::default()
        },
        bab: BabPreset {
            batch_size: Some(128), // Should override solver.batch_size
            alpha_crown: AlphaCrownPreset {
                lr_alpha: Some(0.25), // Should override solver.alpha_crown
                ..Default::default()  // iterations stays from solver
            },
            beta_crown: BetaCrownPreset {
                lr_alpha: Some(0.12), // Should override solver.beta_crown
                ..Default::default()  // lr_beta and iterations stay from solver
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    // batch_size: bab overrides solver
    assert_eq!(
        config.batch_size, 128,
        "bab.batch_size should override solver.batch_size"
    );

    // alpha_crown.lr_alpha: bab overrides solver
    assert_eq!(
        config.alpha_config.learning_rate, 0.25,
        "bab.alpha_crown.lr_alpha should override solver.alpha_crown.lr_alpha"
    );

    // alpha_crown.iterations: solver applied (bab didn't set it)
    assert_eq!(
        config.alpha_config.iterations, 10,
        "solver.alpha_crown.iterations should apply (bab.alpha_crown.iterations is None)"
    );

    // beta_crown.lr_alpha: bab overrides solver
    assert_eq!(
        config.alpha_lr, 0.12,
        "bab.beta_crown.lr_alpha should override solver.beta_crown.lr_alpha"
    );

    // beta_crown.lr_beta: solver applied (bab didn't set it)
    assert_eq!(
        config.beta_lr, 0.15,
        "solver.beta_crown.lr_beta should apply (bab.beta_crown.lr_beta is None)"
    );

    // beta_crown.iterations: solver applied (bab didn't set it)
    assert_eq!(
        config.beta_iterations, 8,
        "solver.beta_crown.iterations should apply (bab.beta_crown.iterations is None)"
    );
}

/// Test solver-only preset (alpha-beta-CROWN style, no bab section).
#[test]
fn solver_only_preset() {
    let preset = PresetConfig {
        solver: SolverPreset {
            batch_size: Some(256),
            alpha_crown: AlphaCrownPreset {
                lr_alpha: Some(0.3),
                iterations: Some(25),
                ..Default::default()
            },
            beta_crown: BetaCrownPreset {
                lr_beta: Some(0.2),
                iterations: Some(12),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert_eq!(config.batch_size, 256, "solver.batch_size should apply");
    assert_eq!(
        config.alpha_config.learning_rate, 0.3,
        "solver.alpha_crown.lr_alpha should apply"
    );
    assert_eq!(
        config.alpha_config.iterations, 25,
        "solver.alpha_crown.iterations should apply"
    );
    assert_eq!(
        config.beta_lr, 0.2,
        "solver.beta_crown.lr_beta should apply"
    );
    assert_eq!(
        config.beta_iterations, 12,
        "solver.beta_crown.iterations should apply"
    );
}

/// Test loading YAML with solver section (alpha-beta-CROWN format).
#[test]
fn load_alpha_beta_crown_yaml_format() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("abcrown.yaml");
    fs::write(
        &config_path,
        r#"
solver:
  batch_size: 512
  alpha-crown:
    lr_alpha: 0.25
    iteration: 20
    start_save_best: 0.25
  beta-crown:
    lr_alpha: 0.1
    lr_beta: 0.2
    iteration: 10
bab:
  branching:
    method: kfsb
"#,
    )
    .unwrap();

    let preset = load_preset(&config_path).unwrap();

    // Check solver section parsed correctly
    assert_eq!(preset.solver.batch_size, Some(512));
    assert_eq!(preset.solver.alpha_crown.lr_alpha, Some(0.25));
    assert_eq!(preset.solver.alpha_crown.iterations, Some(20)); // "iteration" alias
    assert_eq!(preset.solver.alpha_crown.start_save_best, Some(0.25));
    assert_eq!(preset.solver.beta_crown.lr_alpha, Some(0.1));
    assert_eq!(preset.solver.beta_crown.lr_beta, Some(0.2));
    assert_eq!(preset.solver.beta_crown.iterations, Some(10)); // "iteration" alias

    // Apply and verify
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert_eq!(config.batch_size, 512);
    assert_eq!(config.alpha_config.learning_rate, 0.25);
    assert_eq!(config.alpha_config.iterations, 20);
    assert_eq!(config.alpha_config.start_save_best, 0.25);
    assert_eq!(config.alpha_lr, 0.1);
    assert_eq!(config.beta_lr, 0.2);
    assert_eq!(config.beta_iterations, 10);
}

#[test]
fn load_alpha_beta_crown_input_split_yaml_format() {
    let yaml = r#"
bab:
  branching:
    method: sb
    input_split:
      enable: true
      sb_coeff_thresh: 1.0e-2
      sb_sum: true
      touch_zero_score: 0.1
      sb_margin_weight: 0.5
      sb_primary_spec: 1
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert_eq!(config.input_split_coeff_thresh, 1.0e-2);
    assert!(config.input_split_sb_sum);
    assert_eq!(config.input_split_touch_zero_score, 0.1);
    assert_eq!(config.input_split_sb_margin_weight, 0.5);
    assert_eq!(config.input_split_sb_primary_spec, Some(1));
}

#[test]
fn apply_preset_rejects_unknown_branching_method() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("ksfb".to_string()), // typo
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    let err = apply_preset(&mut config, &preset).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("ksfb"), "error should mention the typo: {msg}");
}

#[test]
fn apply_preset_rejects_unknown_reduce_op() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                reduceop: Some("average".to_string()), // not valid
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    let err = apply_preset(&mut config, &preset).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("average"),
        "error should mention the typo: {msg}"
    );
}

#[test]
fn apply_preset_rejects_alpha_beta_crown_sb_without_input_split_enable() {
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("sb".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    let err = apply_preset(&mut config, &preset).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("input_split.enable"),
        "error should explain the missing input_split.enable gate: {msg}"
    );
}

#[test]
fn clip_type_complete_deserialized_from_yaml() {
    let yaml = r#"
bab:
  clip:
    relaxed: true
    clip_type: complete
    neuron_selection_ratio: 0.5
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();
    let clip = &preset.bab.clip;
    assert_eq!(clip.clip_type.as_deref(), Some("complete"));
    assert_eq!(clip.neuron_selection_ratio, Some(0.5));

    let mut config = BetaCrownConfig::default();
    apply_clip_preset(&mut config, clip);
    assert!(config.enable_relaxed_clip);
    assert_eq!(config.input_clip_type, InputClipType::Complete);
    assert_eq!(config.clip_neuron_selection_ratio, 0.5);
}

#[test]
fn clip_type_relaxed_is_default_when_omitted() {
    let yaml = r#"
bab:
  clip:
    relaxed: true
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_clip_preset(&mut config, &preset.bab.clip);
    assert_eq!(config.input_clip_type, InputClipType::Relaxed);
}

// VNN-COMP benchmark preset loading tests moved to vnncomp_preset_tests.rs

/// Test that acasxu_2023 preset loads and produces correct config.
/// Mirrors BetaCrownConfig::acas_xu() settings.
#[test]
fn acasxu_2023_preset_matches_hardcoded_config() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();

    // Verify YAML fields parsed correctly
    assert_eq!(
        preset.solver.bound_prop_method.as_deref(),
        Some("crown"),
        "acasxu preset should use plain CROWN (not alpha-CROWN)"
    );
    assert_eq!(preset.solver.batch_size, Some(16384));
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("after"));
    assert_eq!(preset.attack.pgd_restarts, Some(10000));
    assert_eq!(preset.bab.branching.method, Some("input".to_string()));
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(preset.bab.clip.relaxed, Some(true));

    // Apply and verify config matches BetaCrownConfig::acas_xu()
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    let reference = BetaCrownConfig::acas_xu();
    assert_eq!(
        config.use_alpha_crown, reference.use_alpha_crown,
        "preset should disable alpha-CROWN like acas_xu() config"
    );
    assert_eq!(
        config.batch_size, reference.batch_size,
        "preset batch_size should match acas_xu() config"
    );
    assert_eq!(
        config.branching_heuristic, reference.branching_heuristic,
        "preset branching should match acas_xu() config"
    );
    assert_eq!(
        config.reorder_bab, reference.reorder_bab,
        "preset input-split reordering should match acas_xu() config"
    );
    assert_eq!(
        config.input_split_depth, reference.input_split_depth,
        "preset multi-dim input_split_depth should match acas_xu() config"
    );
    assert_eq!(
        config.enable_relaxed_clip, reference.enable_relaxed_clip,
        "preset relaxed_clip should match acas_xu() config"
    );
    assert_eq!(
        config.enable_pgd_attack, reference.enable_pgd_attack,
        "preset pgd_attack should match acas_xu() config"
    );
    assert_eq!(
        config.pgd_restarts, reference.pgd_restarts,
        "preset pgd_restarts should match acas_xu() config"
    );
}

// Phase budget preset tests moved to phase_budget_tests.rs (#2206 Packet E)

/// #4303: auto_enlarge_batch_size from bab: preset flows to BetaCrownConfig.
#[test]
fn apply_preset_propagates_auto_enlarge_batch_size_4303() {
    let preset = PresetConfig {
        bab: BabPreset {
            auto_enlarge_batch_size: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply::apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.auto_enlarge_batch_size,
        "auto_enlarge_batch_size=true from bab preset must propagate to config"
    );
}

/// #4303: auto_enlarge_batch_size from solver: preset also flows.
#[test]
fn apply_preset_propagates_auto_enlarge_from_solver_4303() {
    let preset = PresetConfig {
        solver: SolverPreset {
            auto_enlarge_batch_size: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply::apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.auto_enlarge_batch_size,
        "auto_enlarge_batch_size=true from solver preset must propagate to config"
    );
}

/// #4303: bab overrides solver for auto_enlarge_batch_size.
#[test]
fn apply_preset_bab_overrides_solver_auto_enlarge_4303() {
    let preset = PresetConfig {
        solver: SolverPreset {
            auto_enlarge_batch_size: Some(true),
            ..Default::default()
        },
        bab: BabPreset {
            auto_enlarge_batch_size: Some(false),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply::apply_preset(&mut config, &preset).unwrap();
    assert!(
        !config.auto_enlarge_batch_size,
        "bab auto_enlarge_batch_size=false should override solver=true"
    );
}

// #1449 attack_mode tests moved to attack_mode_tests.rs

/// #2517: early_stop_patience from bab preset flows to BetaCrownConfig.
#[test]
fn apply_preset_propagates_early_stop_patience_2517() {
    let preset = PresetConfig {
        bab: BabPreset {
            early_stop_patience: Some(5),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("early_stop_patience should apply");
    assert_eq!(
        config.early_stop_patience, 5,
        "bab.early_stop_patience should propagate to BetaCrownConfig (used by beta-CROWN optimize loop)"
    );
}

/// bab.pruning_in_iteration parses for reference-config compatibility but has
/// no engine consumer: apply_preset must leave the config untouched (it warns
/// instead) rather than copy the value into a field nothing reads.
#[test]
fn apply_preset_ignores_unimplemented_pruning_in_iteration() {
    let preset = PresetConfig {
        bab: BabPreset {
            pruning_in_iteration: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("preset should still apply cleanly");
    assert!(
        !config.alpha_config.pruning_in_iteration,
        "unimplemented bab.pruning_in_iteration must not be copied into AlphaCrownConfig"
    );
}

/// #2517: beta lr_decay from bab.beta-crown preset flows to AlphaCrownConfig.
#[test]
fn apply_preset_propagates_beta_lr_decay_2517() {
    let preset = PresetConfig {
        bab: BabPreset {
            beta_crown: BetaCrownPreset {
                lr_decay: Some(0.95),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("beta lr_decay should apply");
    assert_eq!(
        config.alpha_config.lr_decay, 0.95,
        "bab.beta-crown.lr_decay should propagate to alpha_config.lr_decay"
    );
}

/// #2517: beta lr_decay overrides alpha lr_decay when both are set.
#[test]
fn apply_preset_beta_lr_decay_overrides_alpha_2517() {
    let preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                lr_decay: Some(0.99),
                ..Default::default()
            },
            ..Default::default()
        },
        bab: BabPreset {
            beta_crown: BetaCrownPreset {
                lr_decay: Some(0.90),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("lr_decay override should apply");
    assert_eq!(
        config.alpha_config.lr_decay, 0.90,
        "beta lr_decay should override alpha lr_decay (beta applied last)"
    );
}

/// soundnessbench-pgd: `attack.pgd_lr_decay` parses from YAML and propagates to
/// `BetaCrownConfig::pgd_lr_decay` (→ `AdamClippingParams::lr_decay`).
/// Pure attack tuning: PGD counterexamples are re-validated before emission,
/// so this cannot affect soundness.
#[test]
fn apply_preset_propagates_pgd_lr_decay() {
    let yaml = "\
attack:
  pgd_lr_decay: 0.997
";
    let preset: PresetConfig =
        serde_yaml::from_str(yaml).expect("attack.pgd_lr_decay should parse from YAML");
    assert_eq!(preset.attack.pgd_lr_decay, Some(0.997));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("pgd_lr_decay should apply");
    assert_eq!(
        config.pgd_lr_decay, 0.997,
        "attack.pgd_lr_decay should propagate to BetaCrownConfig.pgd_lr_decay"
    );
    // Confirm it threads into the runtime PGD/Adam config.
    let pgd = config.pgd_attack_config(1, 1, None);
    assert_eq!(
        pgd.adam.lr_decay, 0.997,
        "pgd_lr_decay should override AdamClippingParams.lr_decay"
    );
}

/// soundnessbench-pgd: the shipped soundnessbench.yaml wires the reference PGD
/// attack knobs (pure falsification; cannot affect soundness).
#[test]
fn soundnessbench_preset_wires_reference_pgd_knobs() {
    use ny_propagate::PgdAlphaMode;

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/soundnessbench.yaml"))
        .expect("soundnessbench.yaml should load");

    assert_eq!(preset.attack.pgd_restarts, Some(250));
    assert_eq!(preset.attack.pgd_steps, Some(1000));
    assert_eq!(preset.attack.pgd_alpha.as_deref(), Some("0.005"));
    assert_eq!(preset.attack.pgd_lr_decay, Some(0.997));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("soundnessbench preset should apply");
    assert_eq!(config.pgd_restarts, 250);
    assert_eq!(config.pgd_steps, 1000);
    assert_eq!(config.pgd_lr_decay, 0.997);
    assert!(matches!(config.pgd_alpha_mode, PgdAlphaMode::Scalar(a) if a == 0.005));
}
