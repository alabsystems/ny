// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::apply::{apply_clip_preset, apply_preset, parse_reduce_op};
use super::branching::parse_branching_method;
use super::*;
use ny_propagate::{
    BetaCrownConfig, BranchingHeuristic, DepthTwoBranchLookaheadConfig,
    DepthTwoBranchLookaheadMode, InputClipType, KfsbReduceOp,
};
use std::path::Path;
use tempfile::tempdir;

/// Determine if PGD attack should be enabled based on preset.
///
/// Returns the enablement of a valid executable initial schedule.
fn should_enable_pgd(preset: &PresetConfig) -> Option<bool> {
    resolve_initial_pgd_schedule(preset)
        .ok()
        .flatten()
        .map(|schedule| !matches!(schedule, ResolvedInitialPgdSchedule::Disabled))
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
fn resolve_branching_nonlinear_method_selects_genbab_ml4acopf() {
    // `method: nonlinear` (alpha-beta-CROWN's GenBaB token; alias `genbab`)
    // must resolve to GenBaB on the graph ReLU-split route (#ml4acopf-genbab).
    for method in ["nonlinear", "genbab", "NONLINEAR"] {
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
        assert!(
            matches!(branching.heuristic, BranchingHeuristic::GenBaB(_)),
            "preset method '{method}' should resolve to GenBaB"
        );
        assert!(
            branching.use_relu_split,
            "GenBaB runs in the graph engine: use_relu_split must be true"
        );
    }
}

#[test]
fn resolve_branching_explicit_method_wins_over_nonlinear_split_ml4acopf() {
    // Documented precedence: an explicit `method` always wins over an implicit
    // `nonlinear_split` section. `method: input` + populated nonlinear_split
    // keeps input splitting — GenBaB must be named via `method: nonlinear`.
    // (This was the ml4acopf silent-never-GenBaB trap: the preset carried both.)
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                method: Some("input".to_string()),
                nonlinear_split: NonlinearSplitPreset {
                    filter: Some(true),
                    filter_beta: Some(true),
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
        .expect("preset branching should resolve");
    assert!(
        matches!(branching.heuristic, BranchingHeuristic::InputSplit),
        "explicit method: input must keep input splitting"
    );
}

#[test]
fn vnncomp25_ml4acopf_preset_selects_genbab() {
    // The shipped ml4acopf_2024 preset must actually route to GenBaB — the
    // 2026-07 regression was a preset that requested GenBaB via nonlinear_split
    // but pinned `method: input`, so GenBaB was silently never selected.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/ml4acopf_2024.yaml")).unwrap();
    let branching = resolve_branching(&preset)
        .unwrap()
        .expect("ml4acopf preset branching should resolve");
    assert!(
        matches!(branching.heuristic, BranchingHeuristic::GenBaB(_)),
        "ml4acopf_2024.yaml must select GenBaB branching"
    );
    assert!(branching.use_relu_split);
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
fn solver_dark_probe_keys_are_typed_and_default_silent() {
    let absent: PresetConfig =
        serde_yaml::from_str("solver: {}\n").expect("empty solver preset should parse");
    assert_eq!(absent.solver.const_matmul_interval, None);
    assert_eq!(absent.solver.joint_bilinear_quad, None);

    let configured: PresetConfig = serde_yaml::from_str(
        "solver:\n  const_matmul_interval: true\n  joint_bilinear_quad: false\n",
    )
    .expect("typed solver research gates should parse");
    assert_eq!(configured.solver.const_matmul_interval, Some(true));
    assert_eq!(configured.solver.joint_bilinear_quad, Some(false));
}

/// #ml4acopf-bab-queue-mem: `bab.max_queue_bytes` arms the graph ReLU-split
/// queue's byte budget, and a preset that omits it leaves the queue unlimited
/// (0), i.e. byte-identical to today for every category that does not opt in.
#[test]
fn apply_preset_propagates_max_queue_bytes_and_defaults_to_unlimited() {
    let armed = PresetConfig {
        bab: BabPreset {
            max_queue_bytes: Some(2 * 1024 * 1024 * 1024),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).expect("bab.max_queue_bytes should apply cleanly");
    assert_eq!(config.max_queue_bytes, 2 * 1024 * 1024 * 1024);

    let omitted = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &omitted).expect("omitted key should apply cleanly");
    assert_eq!(
        config.max_queue_bytes, 0,
        "presets that omit bab.max_queue_bytes must keep the unlimited queue"
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
fn kfsb_preset_fields_keep_ordinary_and_multi_objective_policies_isolated() {
    let generic_only = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                candidates: Some(7),
                reduceop: Some("max".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut generic_config = BetaCrownConfig::default();
    apply_preset(&mut generic_config, &generic_only).expect("generic kFSB policy should apply");
    assert_eq!(generic_config.fsb_candidates, 7);
    assert_eq!(generic_config.kfsb_reduce_op, KfsbReduceOp::Max);
    assert_eq!(
        generic_config.kfsb_multi_candidates, None,
        "ordinary candidate tuning must not populate the multi-objective override"
    );
    assert_eq!(
        generic_config.kfsb_multi_reduce_op, None,
        "ordinary reduce-op tuning must not populate the multi-objective override"
    );

    let dedicated_only = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                kfsb_multi_candidates: Some(2),
                kfsb_multi_reduceop: Some("max".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let generic_defaults = BetaCrownConfig::default();
    let mut dedicated_config = BetaCrownConfig::default();
    apply_preset(&mut dedicated_config, &dedicated_only)
        .expect("multi-objective kFSB policy should apply");
    assert_eq!(
        dedicated_config.fsb_candidates, generic_defaults.fsb_candidates,
        "multi-objective candidate tuning must not retune ordinary FSB/kFSB"
    );
    assert_eq!(
        dedicated_config.kfsb_reduce_op, generic_defaults.kfsb_reduce_op,
        "multi-objective reduction tuning must not retune ordinary kFSB"
    );
    assert_eq!(dedicated_config.kfsb_multi_candidates, Some(2));
    assert_eq!(
        dedicated_config.kfsb_multi_reduce_op,
        Some(KfsbReduceOp::Max)
    );

    let both = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                candidates: Some(10),
                reduceop: Some("max".to_string()),
                kfsb_multi_candidates: Some(5),
                kfsb_multi_reduceop: Some("min".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut both_config = BetaCrownConfig::default();
    apply_preset(&mut both_config, &both).expect("independent kFSB policies should apply");
    assert_eq!(both_config.fsb_candidates, 10);
    assert_eq!(both_config.kfsb_reduce_op, KfsbReduceOp::Max);
    assert_eq!(both_config.kfsb_multi_candidates, Some(5));
    assert_eq!(both_config.kfsb_multi_reduce_op, Some(KfsbReduceOp::Min));
}

#[test]
fn shipped_wave_kfsb_presets_mirror_generic_policy_without_changing_it() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let configs = repo_root.join("configs/vnncomp25");
    // Seven presets explicitly select kFSB; four high-dimensional image/DAG
    // presets deliberately rely on model-class auto-selection of kFSB. Other
    // branching routes (input, impact, nonlinear, relu, sb, babsr) must not be
    // given selector-scoped values merely because they carry generic fields.
    let expected = [
        ("cctsdb_yolo_2023.yaml", 7),
        ("cifar100_2024.yaml", 7),
        ("collins_rul_cnn_2022.yaml", 8),
        ("cora_2024.yaml", 10),
        ("malbeware.yaml", 5),
        ("relusplitter.yaml", 10),
        ("relusplitter_rsplitter_multiobjective_wgpu.yaml", 10),
        ("soundnessbench.yaml", 5),
        ("tinyimagenet_2024.yaml", 10),
        ("traffic_signs_recognition_2023.yaml", 5),
        ("yolo_2023.yaml", 10),
    ];

    let expected_names = expected
        .iter()
        .map(|(name, _)| name.to_string())
        .collect::<std::collections::BTreeSet<_>>();
    let mut mirrored_names = std::collections::BTreeSet::new();
    for entry in fs::read_dir(&configs).expect("VNN-COMP 2025 config directory should exist") {
        let path = entry
            .expect("config directory entry should be readable")
            .path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
            continue;
        }
        let preset = load_preset(&path).expect("shipped VNN-COMP 2025 preset should load");
        let branching = &preset.bab.branching;
        if branching.kfsb_multi_candidates.is_some() || branching.kfsb_multi_reduceop.is_some() {
            mirrored_names.insert(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .expect("config filename should be UTF-8")
                    .to_owned(),
            );
        }
    }
    assert_eq!(
        mirrored_names, expected_names,
        "only wave-kFSB-eligible presets should carry selector-scoped mirrors"
    );

    for (name, candidates) in expected {
        let preset = load_preset(&configs.join(name)).expect("wave-kFSB preset should load");
        let branching = &preset.bab.branching;
        assert_eq!(branching.candidates, Some(candidates), "preset={name}");
        assert_eq!(
            branching.kfsb_multi_candidates,
            Some(candidates),
            "preset={name}"
        );
        assert_eq!(branching.reduceop.as_deref(), Some("max"), "preset={name}");
        assert_eq!(
            branching.kfsb_multi_reduceop.as_deref(),
            Some("max"),
            "preset={name}"
        );

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).expect("wave-kFSB preset should apply");
        assert_eq!(config.fsb_candidates, candidates, "preset={name}");
        assert_eq!(
            config.kfsb_multi_candidates,
            Some(candidates),
            "preset={name}"
        );
        assert_eq!(config.kfsb_reduce_op, KfsbReduceOp::Max, "preset={name}");
        assert_eq!(
            config.kfsb_multi_reduce_op,
            Some(KfsbReduceOp::Max),
            "preset={name}"
        );
    }
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
fn preset_yaml_rejects_unknown_keys_at_every_mapping_level() {
    let cases = [
        ("top level", "generall: {}\n", "generall"),
        ("general", "general:\n  root_pat: ./data\n", "root_pat"),
        (
            "model",
            "model:\n  onnx_optimisation_flags: merge_linear\n",
            "onnx_optimisation_flags",
        ),
        ("attack", "attack:\n  pgd_orderr: before\n", "pgd_orderr"),
        (
            "margin row",
            "margin_row:\n  reserve_second: 5\n",
            "reserve_second",
        ),
        ("solver", "solver:\n  batch_sizes: 16\n", "batch_sizes"),
        (
            "solver mip",
            "solver:\n  mip:\n    mip_sovler: ay\n",
            "mip_sovler",
        ),
        ("bab", "bab:\n  max_domain: 10\n", "max_domain"),
        (
            "phase budget",
            "bab:\n  phase_budget:\n    upfront_pgd_fracton: 0.2\n",
            "upfront_pgd_fracton",
        ),
        (
            "branching",
            "bab:\n  branching:\n    canddiates: 5\n",
            "canddiates",
        ),
        (
            "depth-two lookahead",
            "bab:\n  branching:\n    depth2_lookahead:\n      canddiates: 5\n",
            "canddiates",
        ),
        (
            "input split",
            "bab:\n  branching:\n    input_split:\n      reorder_bba: true\n",
            "reorder_bba",
        ),
        (
            "nonlinear split",
            "bab:\n  branching:\n    nonlinear_split:\n      filter_betta: true\n",
            "filter_betta",
        ),
        (
            "alpha crown",
            "solver:\n  alpha_crown:\n    lr_alhpa: 0.1\n",
            "lr_alhpa",
        ),
        (
            "beta crown",
            "solver:\n  beta_crown:\n    lr_betta: 0.1\n",
            "lr_betta",
        ),
        ("cuts", "bab:\n  cuts:\n    max_cut: 10\n", "max_cut"),
        (
            "invprop",
            "bab:\n  invprop:\n    share_gamma: true\n",
            "share_gamma",
        ),
        (
            "clip",
            "bab:\n  clip:\n    interm_top_k: 4\n",
            "interm_top_k",
        ),
    ];

    for (location, yaml, misspelled_key) in cases {
        let error = match serde_yaml::from_str::<PresetConfig>(yaml) {
            Ok(_) => panic!("{location} typo {misspelled_key:?} must be rejected"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains(misspelled_key),
            "{location} error must identify {misspelled_key:?}: {message}"
        );
    }
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
fn forward_linear_spec_alpha_requires_typed_authored_graph_policy() {
    let unsafe_preset: PresetConfig = serde_yaml::from_str(
        r#"
model:
  forward_linear_spec_alpha: true
"#,
    )
    .expect("typed candidate key should parse");
    let error = match build_onnx_load_config(&unsafe_preset) {
        Ok(_) => panic!("candidate authority over the folded graph must be refused"),
        Err(error) => error,
    };
    assert!(
        format!("{error:#}").contains("require_authored_float32_initializers"),
        "admission error must prescribe both typed authored-graph guards: {error:#}"
    );

    let admitted: PresetConfig = serde_yaml::from_str(
        r#"
model:
  batch_norm_folding: preserve_raw
  require_authored_float32_initializers: true
  forward_linear_spec_alpha: true
"#,
    )
    .expect("typed raw candidate preset should parse");
    let config = build_onnx_load_config(&admitted).expect("raw candidate should be admitted");
    assert_eq!(
        config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::PreserveRaw
    );
    assert!(config.require_authored_float32_initializers());
    assert!(config.raw_float32_initializer_provenance_enabled());
    assert_eq!(admitted.model.forward_linear_spec_alpha, Some(true));
}

#[test]
fn forward_linear_spec_alpha_is_default_off_and_legacy_loader_stays_unchanged() {
    let preset = PresetConfig::default();
    let config = build_onnx_load_config(&preset).expect("default loader config");
    assert_eq!(preset.model.forward_linear_spec_alpha, None);
    assert_eq!(
        config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::LegacyEnvironment
    );
    assert!(!config.require_authored_float32_initializers());
}

#[test]
fn legacy_cgan_alpha_surrogate_key_is_an_exact_alias() {
    let generic: PresetConfig = serde_yaml::from_str(
        r#"
model:
  batch_norm_folding: preserve_raw
  require_authored_float32_initializers: true
  forward_linear_spec_alpha: true
"#,
    )
    .expect("generic key should parse");
    let legacy: PresetConfig = serde_yaml::from_str(
        r#"
model:
  batch_norm_folding: preserve_raw
  require_authored_float32_initializers: true
  cgan_forward_alpha_surrogate: true
"#,
    )
    .expect("legacy key should remain accepted");

    assert_eq!(generic.model.forward_linear_spec_alpha, Some(true));
    assert_eq!(legacy.model.forward_linear_spec_alpha, Some(true));
    build_onnx_load_config(&generic).expect("generic admission should pass loader guards");
    build_onnx_load_config(&legacy).expect("legacy alias should pass the same loader guards");
    assert_eq!(
        serde_json::to_value(&generic).unwrap(),
        serde_json::to_value(&legacy).unwrap(),
        "the legacy spelling must not create a second policy bit"
    );
}

#[test]
fn forward_linear_spec_alpha_malformed_values_fail_closed() {
    for malformed in [
        r#"model: { forward_linear_spec_alpha: "1" }"#,
        r#"model: { forward_linear_spec_alpha: enabled }"#,
        r#"model: { cgan_forward_alpha_surrogate: "true" }"#,
    ] {
        let error = serde_yaml::from_str::<PresetConfig>(malformed)
            .expect_err("non-boolean admission values must be rejected");
        assert!(
            format!("{error:#}").contains("boolean"),
            "malformed admission should report its type error: {error:#}"
        );
    }

    let duplicate = serde_yaml::from_str::<PresetConfig>(
        r#"
model:
  forward_linear_spec_alpha: true
  cgan_forward_alpha_surrogate: true
"#,
    )
    .expect_err("generic and legacy spellings together must fail closed");
    assert!(
        format!("{duplicate:#}").contains("duplicate field"),
        "two spellings must not create ambiguous precedence: {duplicate:#}"
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

/// Regression: the #3813 WGPU multi-objective sidecar must explicitly request a
/// wgpu device, keep kfsb branching, preserve matrix Conv2d, and keep
/// quarantined cut authority disabled. Without the explicit device field the
/// benchmark runner silently falls back to CPU.
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

    // The legacy cut consumer adds a scalar after CROWN concretization, so it
    // cannot carry proof authority.
    assert_eq!(
        sidecar.bab.cuts.enabled,
        Some(false),
        "sidecar must keep quarantined cuts disabled"
    );
    assert_eq!(sidecar.bab.cuts.near_miss, Some(false));
    assert_eq!(sidecar.bab.cuts.proactive, Some(false));

    // Preserve the measured matrix path explicitly now that cuts are disabled.
    assert_eq!(
        sidecar.general.conv_mode,
        Some(ny_propagate::ConvMode::Matrix),
        "sidecar must explicitly use conv_mode: matrix"
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
/// benchmark-scoped opt-in so no unrelated preset can silently arm the wave
/// lane.
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

#[test]
fn apply_preset_propagates_kfsb_cert_reuse_policy() {
    let armed = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                kfsb_cert_reuse: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).unwrap();
    assert!(config.kfsb_cert_reuse);

    let disarmed = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                kfsb_cert_reuse: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig {
        kfsb_cert_reuse: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &disarmed).unwrap();
    assert!(!config.kfsb_cert_reuse);

    let mut omitted = BetaCrownConfig {
        kfsb_cert_reuse: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut omitted, &PresetConfig::default()).unwrap();
    assert!(
        omitted.kfsb_cert_reuse,
        "an omitted preset key must not overwrite an already-resolved policy"
    );
}

/// Root-alpha phase checkpoints are preset-owned scheduling policy: an
/// explicit value propagates in either direction, while omission preserves the
/// default-dark verifier configuration.
#[test]
fn apply_preset_propagates_root_alpha_phase_checkpoint_policy() {
    let armed = PresetConfig {
        bab: BabPreset {
            root_alpha_phase_checkpoint: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &armed).unwrap();
    assert!(config.root_alpha_phase_checkpoint);

    let disarmed = PresetConfig {
        bab: BabPreset {
            root_alpha_phase_checkpoint: Some(false),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig {
        root_alpha_phase_checkpoint: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut config, &disarmed).unwrap();
    assert!(!config.root_alpha_phase_checkpoint);

    let mut omitted = BetaCrownConfig {
        root_alpha_phase_checkpoint: true,
        ..BetaCrownConfig::default()
    };
    apply_preset(&mut omitted, &PresetConfig::default()).unwrap();
    assert!(
        omitted.root_alpha_phase_checkpoint,
        "an omitted preset key must not overwrite an already-resolved policy"
    );
}

#[test]
fn apply_preset_propagates_typed_depth_two_lookahead_without_arming_on_omission() {
    let preset: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  branching:
    depth2_lookahead:
      mode: select
      candidates: 15
      top_rounds: 5
      discount: 0.5
"#,
    )
    .expect("typed depth-2 preset parses");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("typed depth-2 preset applies");
    assert_eq!(
        config.depth_two_branch_lookahead,
        DepthTwoBranchLookaheadConfig {
            mode: DepthTwoBranchLookaheadMode::Select,
            candidates: 15,
            top_rounds: 5,
            discount: 0.5,
        }
    );
    config.validate().expect("typed experiment validates");

    let omitted: PresetConfig =
        serde_yaml::from_str("bab: {}").expect("omitted depth-2 preset parses");
    let mut default = BetaCrownConfig::default();
    apply_preset(&mut default, &omitted).expect("omitted depth-2 preset applies");
    assert_eq!(
        default.depth_two_branch_lookahead.mode,
        DepthTwoBranchLookaheadMode::Off
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
fn apply_preset_propagates_bounded_atomic_root_c_margin_iterations() {
    let preset: PresetConfig = serde_yaml::from_str("bab:\n  atomic_root_c_margin_iterations: 8\n")
        .expect("typed exact-C iteration preset parses");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("the hard cap applies");
    assert_eq!(config.atomic_root_c_margin_iterations, 8);
    config.validate().expect("applied hard cap validates");

    let mut omitted = BetaCrownConfig {
        atomic_root_c_margin_iterations: 3,
        ..Default::default()
    };
    apply_preset(&mut omitted, &PresetConfig::default()).expect("omission is inert");
    assert_eq!(
        omitted.atomic_root_c_margin_iterations, 3,
        "an omitted typed key must not overwrite an existing value"
    );

    let over: PresetConfig = serde_yaml::from_str("bab:\n  atomic_root_c_margin_iterations: 9\n")
        .expect("the typed parser preserves the invalid value for a precise apply error");
    let error = apply_preset(&mut BetaCrownConfig::default(), &over)
        .expect_err("work above the hard cap must fail closed");
    assert!(error
        .to_string()
        .contains("bab.atomic_root_c_margin_iterations"));
}

#[test]
fn apply_preset_propagates_root_interm_cuda_factory_policy() {
    for (preset_value, expected) in [(true, true), (false, false)] {
        let preset = PresetConfig {
            bab: BabPreset {
                root_interm_cuda_factory: Some(preset_value),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig {
            root_interm_cuda_factory: !preset_value,
            ..Default::default()
        };
        apply_preset(&mut config, &preset).expect("typed factory policy applies");
        assert_eq!(config.root_interm_cuda_factory, expected);
    }

    let mut omitted = BetaCrownConfig {
        root_interm_cuda_factory: true,
        ..Default::default()
    };
    apply_preset(&mut omitted, &PresetConfig::default()).expect("omitted policy applies");
    assert!(
        omitted.root_interm_cuda_factory,
        "an omitted Option must not overwrite an existing typed value"
    );

    let parsed: PresetConfig = serde_yaml::from_str("bab:\n  root_interm_cuda_factory: true\n")
        .expect("factory preset key is accepted");
    assert_eq!(parsed.bab.root_interm_cuda_factory, Some(true));
}

#[test]
fn apply_preset_propagates_mo_cuda_factory_engine_handoff_policy() {
    for (preset_value, expected) in [(true, true), (false, false)] {
        let preset = PresetConfig {
            bab: BabPreset {
                mo_cuda_factory_engine_handoff: Some(preset_value),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig {
            mo_cuda_factory_engine_handoff: !preset_value,
            ..Default::default()
        };
        apply_preset(&mut config, &preset).expect("typed post-root handoff policy applies");
        assert_eq!(config.mo_cuda_factory_engine_handoff, expected);
    }

    let mut omitted = BetaCrownConfig {
        mo_cuda_factory_engine_handoff: true,
        ..Default::default()
    };
    apply_preset(&mut omitted, &PresetConfig::default()).expect("omitted policy applies");
    assert!(
        omitted.mo_cuda_factory_engine_handoff,
        "an omitted Option must not overwrite an existing typed value"
    );

    let parsed: PresetConfig =
        serde_yaml::from_str("bab:\n  mo_cuda_factory_engine_handoff: true\n")
            .expect("post-root handoff preset key is accepted");
    assert_eq!(parsed.bab.mo_cuda_factory_engine_handoff, Some(true));
}

#[test]
fn apply_preset_propagates_mo_cuda_bounded_shared_executor_policy() {
    for (preset_value, expected) in [(true, true), (false, false)] {
        let preset = PresetConfig {
            bab: BabPreset {
                mo_cuda_bounded_shared_executor: Some(preset_value),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig {
            mo_cuda_bounded_shared_executor: !preset_value,
            ..Default::default()
        };
        apply_preset(&mut config, &preset).expect("typed bounded shared-executor policy applies");
        assert_eq!(config.mo_cuda_bounded_shared_executor, expected);
    }

    let mut omitted = BetaCrownConfig {
        mo_cuda_bounded_shared_executor: true,
        ..Default::default()
    };
    apply_preset(&mut omitted, &PresetConfig::default()).expect("omitted policy applies");
    assert!(
        omitted.mo_cuda_bounded_shared_executor,
        "an omitted Option must not overwrite an existing typed value"
    );

    let parsed: PresetConfig =
        serde_yaml::from_str("bab:\n  mo_cuda_bounded_shared_executor: true\n")
            .expect("bounded shared-executor preset key is accepted");
    assert_eq!(parsed.bab.mo_cuda_bounded_shared_executor, Some(true));
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

/// #kfsb-multi: the shipped CIFAR-100 presets arm the wave-batched selector,
/// while a representative unrelated preset leaves it off.
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
        "acasxu preset must leave use_kfsb_multi_branching off"
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
fn input_split_override_parallel_is_preset_scoped_and_defaults_off() {
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
    assert_eq!(absent.bab.branching.input_split.override_parallel, None);
    assert!(!absent_config.input_split_override_parallel);

    let enabled: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  branching:
    method: input
    input_split:
      reorder_bab: true
      override_parallel: true
"#,
    )
    .unwrap();
    let mut enabled_config = BetaCrownConfig::default();
    apply_preset(&mut enabled_config, &enabled).unwrap();
    assert_eq!(
        enabled.bab.branching.input_split.override_parallel,
        Some(true)
    );
    assert!(enabled_config.reorder_bab);
    assert!(enabled_config.input_split_override_parallel);
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

/// Implemented orders decode exactly. Reference middle/after require an
/// explicit NY compatibility contract before they may use upfront placement.
#[test]
fn apply_preset_decodes_pgd_order_enablement() {
    for (order, expected) in [
        ("skip", false),
        ("none", false),
        ("disabled", false),
        ("before", true),
        ("input_bab", true),
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

    // `after` is IMPLEMENTED now (#pgd-order-after): it resolves to the deferred schedule,
    // which empties the upfront slice and hands it to the post-BaB stage. Only `middle`
    // (attack interleaved with BaB) remains unimplemented and must still fail closed.
    {
        let deferred = PresetConfig {
            attack: AttackPreset {
                pgd_order: Some("after".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &deferred).expect("'after' is implemented");
        assert!(
            config.phase_budget.upfront_pgd_fraction.abs() < 1e-9,
            "deferring must empty the upfront slice"
        );
    }

    {
        let order = "middle";
        let missing_contract = PresetConfig {
            attack: AttackPreset {
                pgd_order: Some(order.to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut config = BetaCrownConfig::default();
        let before = serde_json::to_value(&config).unwrap();
        let error = apply_preset(&mut config, &missing_contract)
            .expect_err("unimplemented order without compatibility must fail closed");
        assert!(error.to_string().contains("ny_pgd_order_compat"));
        assert_eq!(
            serde_json::to_value(&config).unwrap(),
            before,
            "schedule validation must happen before mutating verifier config"
        );

        let explicit_compatibility = PresetConfig {
            attack: AttackPreset {
                pgd_order: Some(order.to_string()),
                ny_pgd_order_compat: Some(NyPgdOrderCompat::Upfront),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_preset(&mut config, &explicit_compatibility).unwrap();
        assert!(config.enable_pgd_attack);
        assert_eq!(
            resolve_initial_pgd_schedule(&explicit_compatibility).unwrap(),
            Some(ResolvedInitialPgdSchedule::Upfront)
        );
    }

    for invalid in ["skpi", " skip ", ""] {
        let preset = PresetConfig {
            attack: AttackPreset {
                pgd_order: Some(invalid.to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            apply_preset(&mut BetaCrownConfig::default(), &preset).is_err(),
            "unknown/near-miss pgd_order {invalid:?} must not silently enable PGD"
        );
    }

    for invalid_order in [None, Some("before"), Some("skip"), Some("input_bab")] {
        let preset = PresetConfig {
            attack: AttackPreset {
                pgd_order: invalid_order.map(str::to_string),
                ny_pgd_order_compat: Some(NyPgdOrderCompat::Upfront),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            apply_preset(&mut BetaCrownConfig::default(), &preset).is_err(),
            "compatibility field must be rejected outside middle/after: {invalid_order:?}"
        );
    }
}

#[test]
fn shipped_middle_after_presets_declare_upfront_compatibility() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut scheduled = 0usize;

    for directory in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let directory = repo_root.join("configs").join(directory);
        for entry in fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
                continue;
            }
            let preset = load_preset(&path).unwrap();
            if preset.attack.pgd_order.as_deref().is_some_and(|order| {
                order.eq_ignore_ascii_case("middle") || order.eq_ignore_ascii_case("after")
            }) {
                scheduled += 1;
                assert_eq!(
                    preset.attack.ny_pgd_order_compat,
                    Some(NyPgdOrderCompat::Upfront),
                    "{} must explicitly preserve NY's initial/upfront compatibility behavior",
                    path.display()
                );
                validate_preset(&preset).unwrap_or_else(|error| {
                    panic!(
                        "{} must remain semantically valid under the frozen-preset gate: {error:#}",
                        path.display()
                    )
                });
            }
        }
    }

    assert_eq!(
        scheduled, 22,
        "review every shipped middle/after preset when the compatibility set changes"
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

#[test]
fn input_split_fresh_domain_clip_has_typed_default_dark_preset_ingress() {
    let preset: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  branching:
    method: input
    input_split:
      enable: true
      reorder_bab: true
      ibp_enhancement: true
  clip:
    relaxed: true
    relaxed_iterations: 20
    input_split_fresh_domain_clip: true
"#,
    )
    .expect("typed fresh-domain clip preset parses");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("typed fresh-domain clip preset applies");
    assert!(config.input_split_fresh_domain_clip);
    config
        .validate()
        .expect("the production-shaped fresh-domain clip composition validates");

    let mut omitted = BetaCrownConfig {
        input_split_fresh_domain_clip: true,
        ..Default::default()
    };
    apply_clip_preset(&mut omitted, &ClipPreset::default());
    assert!(
        omitted.input_split_fresh_domain_clip,
        "omission must not overwrite an existing typed routing decision"
    );

    let disabled: PresetConfig =
        serde_yaml::from_str("bab:\n  clip:\n    input_split_fresh_domain_clip: false\n")
            .expect("an explicit false parses");
    apply_clip_preset(&mut omitted, &disabled.bab.clip);
    assert!(!omitted.input_split_fresh_domain_clip);
}

#[test]
fn input_split_conic_objective_has_typed_default_dark_preset_ingress() {
    let default = BetaCrownConfig::default();
    assert!(!default.input_split_conic_objective);

    let preset: PresetConfig = serde_yaml::from_str(
        r#"
bab:
  branching:
    method: input
    input_split:
      conic_objective: true
      conic_queue_refresh_batch_size: 1024
"#,
    )
    .expect("typed conic-objective preset parses");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("typed conic-objective preset applies");
    assert!(config.input_split_conic_objective);
    assert_eq!(config.input_split_conic_queue_refresh_batch_size, 1024);

    let mut omitted = BetaCrownConfig {
        input_split_conic_objective: true,
        input_split_conic_queue_refresh_batch_size: 2048,
        ..Default::default()
    };
    apply_preset(&mut omitted, &PresetConfig::default())
        .expect("an omitted conic-objective key applies as a no-op");
    assert!(
        omitted.input_split_conic_objective,
        "omission must not overwrite an existing typed routing decision"
    );
    assert_eq!(omitted.input_split_conic_queue_refresh_batch_size, 2048);

    let disabled: PresetConfig = serde_yaml::from_str(
        "bab:\n  branching:\n    input_split:\n      conic_objective: false\n",
    )
    .expect("an explicit false parses");
    apply_preset(&mut omitted, &disabled).expect("an explicit false applies");
    assert!(!omitted.input_split_conic_objective);
    assert_eq!(omitted.input_split_conic_queue_refresh_batch_size, 2048);
}

#[test]
fn shipped_cersyve_preset_does_not_arm_conic_objective_yet() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml"))
        .expect("shipped Cersyve preset loads");
    assert_eq!(preset.bab.branching.input_split.conic_objective, None);
    assert_eq!(
        preset
            .bab
            .branching
            .input_split
            .conic_queue_refresh_batch_size,
        None
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("shipped Cersyve preset applies");
    assert!(!config.input_split_conic_objective);
    assert_eq!(config.input_split_conic_queue_refresh_batch_size, 512);
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

/// #preset-strict: every preset shipped in configs/vnncomp*/ must load with
/// ZERO unrecognized keys. `load_preset` errors on unknown paths, so this test
/// is what guarantees an in-repo preset can only drift from the schema by
/// failing CI — a typo'd key was previously dropped in silence, which is the
/// structural enabler of the "configured but never fires" bug class.
#[test]
fn every_shipped_vnncomp_preset_loads_with_no_unknown_keys() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let configs_root = repo_root.join("configs");
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for entry in fs::read_dir(&configs_root).expect("list configs/") {
        let dir = entry.expect("read configs/ entry").path();
        let is_vnncomp_dir = dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("vnncomp"));
        if !is_vnncomp_dir || !dir.is_dir() {
            continue;
        }
        for file in fs::read_dir(&dir).expect("list preset dir") {
            let path = file.expect("read preset entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            checked += 1;
            if let Err(error) = load_preset(&path) {
                failures.push(format!("{}: {error:#}", path.display()));
            }
        }
    }
    assert!(
        checked > 20,
        "expected to check every shipped vnncomp preset, found only {checked} — \
         did the configs layout move?"
    );
    assert!(
        failures.is_empty(),
        "presets with unknown or invalid keys:\n{}",
        failures.join("\n")
    );
}

/// #dist-shift-5291: the typed nested-deadline authority key
/// (`bab.branching.input_split.nested_deadline`) must map onto
/// `BetaCrownConfig::input_split_nested_deadline`, and its absence must leave
/// the shipped deadline-carrying default untouched.
#[test]
fn input_split_nested_deadline_preset_maps_to_config() {
    // Absent: shipped default (None = keep the nested deadline).
    let preset = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_nested_deadline, None);

    // Explicit relaxation: the dist_shift ship shape.
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    nested_deadline: Some(false),
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
    assert_eq!(config.input_split_nested_deadline, Some(false));

    // Explicit keep round-trips too (a preset may pin the shipped arm).
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    nested_deadline: Some(true),
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
    assert_eq!(config.input_split_nested_deadline, Some(true));
}

/// #dist-shift-5291 pinning test: the shipped dist_shift preset does NOT yet
/// arm the nested-deadline relaxation. The ship gate
/// (docs/DIST_SHIFT_NESTED_DEADLINE_2026-08-20.md) is 3/arm on
/// index5291_delta0.13 plus the category sentinels; flipping the yaml without
/// updating this assertion is the drift this test exists to catch.
#[test]
fn dist_shift_2023_nested_deadline_is_not_yet_armed() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/dist_shift_2023.yaml")).unwrap();
    assert_eq!(preset.bab.branching.input_split.nested_deadline, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_nested_deadline, None);
}

/// #cgan-dark-tail FIX-1: the typed per-child bound cap
/// (`bab.branching.input_split.child_bound_max_secs`) must map onto
/// `BetaCrownConfig::input_split_child_bound_max_secs`, and its ABSENCE must
/// leave the shipped no-cap default (`None`) untouched — the key ships DARK.
#[test]
fn input_split_child_bound_max_secs_preset_maps_to_config() {
    // Absent: shipped default = NO CAP = today's behaviour byte-for-byte.
    let preset = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.input_split_child_bound_max_secs, None,
        "the per-child cap must ship dark (None = no cap)"
    );

    // Named: the measurement shape.
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    child_bound_max_secs: Some(20.0),
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
    assert_eq!(config.input_split_child_bound_max_secs, Some(20.0));
}

/// #cgan-dark-tail FIX-1 pinning test: the shipped cgan preset does NOT yet
/// name the per-child cap. FIX-1 lands dark on purpose — the R9 gate is all
/// three of `frozen_fallback_completions` 0-1 -> >>1, the last trajectory `t=`
/// 565-606 -> >=880, and `explored=` roughly doubling, PLUS the lever's own
/// `[child-bound-cap] final ... capped_calls=` > 0 (armed is not engaged).
///
/// The arming diff is pre-staged and COMMENTED OUT in
/// `configs/vnncomp25/cgan_2023.yaml` together with the measurement that
/// motivates it; uncommenting it is a pure `# ` removal, and it must land in
/// the same commit as the updated assertion here. Flipping the yaml without
/// updating this assertion is exactly the drift this test catches.
#[test]
fn cgan_2023_child_bound_max_secs_is_not_yet_armed() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    assert_eq!(preset.bab.branching.input_split.child_bound_max_secs, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_child_bound_max_secs, None);
}

/// #cost-observer: the typed self-calibrating per-child allowance key
/// (`bab.branching.input_split.child_bound_calibration`) must map onto
/// `BetaCrownConfig::input_split_child_bound_calibration`, and its ABSENCE must
/// leave `None` — i.e. the `child_bound_max_secs` constant stays in charge and
/// the route is byte-for-byte the shipped one.
#[test]
fn input_split_child_bound_calibration_preset_maps_to_config() {
    // Absent: the constant governs. This is the byte-identity leg.
    let preset = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.input_split_child_bound_calibration, None,
        "calibration must ship dark; the constant stays in charge"
    );

    // Present but EMPTY: arming needs no numbers at all, because every number
    // it would need is measured. This is the intended production shape.
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    child_bound_calibration: Some(
                        InputSplitChildBoundCalibrationPreset::default(),
                    ),
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
    let armed = config
        .input_split_child_bound_calibration
        .expect("an empty block still arms — presence is the switch");
    assert_eq!(armed.headroom, None, "unnamed fields stay unnamed");
    let policy = armed.to_policy();
    let default_policy = ny_core::cost_observer::SlicePolicy::default();
    assert_eq!(
        policy, default_policy,
        "an empty block must resolve to the shipped SlicePolicy default"
    );

    // Named: each dimensionless field overrides its own default and nothing else.
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    child_bound_calibration: Some(InputSplitChildBoundCalibrationPreset {
                        headroom: Some(64.0),
                        max_share_of_remaining: Some(0.4),
                        ..Default::default()
                    }),
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
    let policy = config
        .input_split_child_bound_calibration
        .expect("named block arms")
        .to_policy();
    assert_eq!(policy.headroom, 64.0);
    assert_eq!(policy.max_share_of_remaining, 0.4);
    assert_eq!(
        policy.probe_share, default_policy.probe_share,
        "an unnamed field must keep its default, not be zeroed"
    );
    assert_eq!(policy.window, default_policy.window);
}

/// #cost-observer: the key must be reachable through the REAL yaml route, from
/// a file on disk, with no environment variable involved anywhere.
///
/// `vnncomp_scripts/run_instance.sh` exports ZERO `NY_*` variables, so a
/// mechanism reachable only from an env var is dead capability in competition.
/// This key has no env route at all by design — the assertion that matters is
/// that the yaml one carries every field end-to-end.
#[test]
fn child_bound_calibration_is_reachable_from_a_yaml_file_on_disk() {
    let yaml = r"
bab:
  branching:
    input_split:
      child_bound_calibration:
        headroom: 96.0
        max_share_of_remaining: 0.3
        probe_share: 0.02
        min_samples: 4
        window: 64
        growth: 3.0
";
    let dir = tempdir().unwrap();
    let path = dir.path().join("armed.yaml");
    fs::write(&path, yaml).unwrap();
    let preset = load_preset(&path).expect("armed yaml must parse");

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    let policy = config
        .input_split_child_bound_calibration
        .expect("the yaml route must arm the calibration")
        .to_policy();
    assert_eq!(policy.headroom, 96.0);
    assert_eq!(policy.max_share_of_remaining, 0.3);
    assert_eq!(policy.probe_share, 0.02);
    assert_eq!(policy.min_samples, 4);
    assert_eq!(policy.window, 64);
    assert_eq!(policy.growth, 3.0);
}

/// #cost-observer pinning test: the shipped cgan preset does NOT yet name the
/// calibration block, so the mechanism lands dark exactly like FIX-1 did.
///
/// Arming it is a pure `# ` removal in `configs/vnncomp25/cgan_2023.yaml`,
/// pre-staged there together with the measurement that motivates it, and it
/// must land in the same commit as the updated assertion here.
#[test]
fn cgan_2023_child_bound_calibration_is_not_yet_armed() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    assert!(preset
        .bab
        .branching
        .input_split
        .child_bound_calibration
        .is_none());

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_child_bound_calibration, None);
}

/// #cgan-f64-tail (design B step B5): the typed f64-tail key
/// (`bab.branching.input_split.f64_tail.{enabled, band, max_escalations}`)
/// must map onto `BetaCrownConfig::input_split_f64_tail_*`, and its absence
/// must leave every field `None` (dark, byte-identical).
#[test]
fn input_split_f64_tail_preset_maps_to_config() {
    // Absent: dark default.
    let preset = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_f64_tail_enabled, None);
    assert_eq!(config.input_split_f64_tail_band, None);
    assert_eq!(config.input_split_f64_tail_max_escalations, None);

    // The cgan ship shape: enabled, 5e-4 band, cap 8.
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    f64_tail: Some(InputSplitF64TailPreset {
                        enabled: Some(true),
                        band: Some(5e-4),
                        max_escalations: Some(8),
                    }),
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
    assert_eq!(config.input_split_f64_tail_enabled, Some(true));
    assert_eq!(config.input_split_f64_tail_band, Some(5e-4));
    assert_eq!(config.input_split_f64_tail_max_escalations, Some(8));

    // Explicit disarm round-trips (a preset may pin the mechanism off).
    let preset = PresetConfig {
        bab: BabPreset {
            branching: BranchingPreset {
                input_split: InputSplitPreset {
                    f64_tail: Some(InputSplitF64TailPreset {
                        enabled: Some(false),
                        band: None,
                        max_escalations: None,
                    }),
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
    assert_eq!(config.input_split_f64_tail_enabled, Some(false));
    assert_eq!(config.input_split_f64_tail_band, None);
    assert_eq!(config.input_split_f64_tail_max_escalations, None);
}

/// #cgan-f64-tail pinning test: the shipped cgan preset does NOT yet arm the
/// f64 tail. The ship gate is the design-B measurement protocol (A/B 3/arm on
/// cgan row 2 at 900 s with `anchor_source=warm-refined` engagement lines,
/// then the cgan moat: 12 sat rows + test_nano unsat + the lsnc regression
/// rows). Flipping cgan_2023.yaml without updating this assertion is the
/// drift this test exists to catch.
///
/// Why it stays dark despite a very strong measurement. At band 5e-4 the
/// certified tail closes 484/484 escalations with ZERO residuals over 3 runs
/// — but cgan's frontier sits at gap ~ -8e-4, OUTSIDE that band, so the
/// measured conversion count is ZERO: arming at 5e-4 is verdict-NEUTRAL, not
/// strictly better. Band 2e-3 (87.3% recovery, 0.1235 s mean, throughput
/// -positive) is the conversion PROBE and has not been run. The arming diff
/// for it is pre-staged and commented out in `cgan_2023.yaml`.
#[test]
fn cgan_2023_f64_tail_is_not_yet_armed() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    assert!(preset.bab.branching.input_split.f64_tail.is_none());

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.input_split_f64_tail_enabled, None);
    assert_eq!(config.input_split_f64_tail_band, None);
    assert_eq!(config.input_split_f64_tail_max_escalations, None);
}

#[test]
fn margin_row_declined_schedule_fractions_preserve_historical_fallback() {
    for value in [0.0_f32, 1.0, -0.25, f32::NAN, f32::INFINITY] {
        let mut max_fraction = PresetConfig::default();
        max_fraction.margin_row.reserve_max_frac = Some(value);
        apply_preset(&mut BetaCrownConfig::default(), &max_fraction)
            .expect("a declined optional reserve ceiling restores absence");

        let mut release_fraction = PresetConfig::default();
        release_fraction.margin_row.release_frac = Some(value);
        apply_preset(&mut BetaCrownConfig::default(), &release_fraction)
            .expect("a declined adaptive threshold restores the 0.40 default");
    }

    let mut admitted = PresetConfig::default();
    admitted.margin_row.reserve_max_frac = Some(0.35_f32);
    admitted.margin_row.release_frac = Some(0.40_f32);
    apply_preset(&mut BetaCrownConfig::default(), &admitted)
        .expect("finite open-unit scheduler fractions remain admissible");
}

#[test]
fn margin_row_schedule_yaml_preserves_exact_typed_scalar_boundaries() {
    for malformed in [
        "margin_row:\n  reserve_secs: \"0\"\n",
        "margin_row:\n  reserve_secs: false\n",
        "margin_row:\n  adaptive_reserve: 1\n",
        "margin_row:\n  reserve_max_frac: \"0.25\"\n",
        "margin_row:\n  release_frac: []\n",
        "margin_row:\n  early_exit: 1\n",
        "margin_row:\n  verifier_threads: \"0\"\n",
        "margin_row:\n  verifier_threads: -1\n",
        "margin_row:\n  concurrent_threads: false\n",
        "margin_row:\n  concurrent_threads: []\n",
        "margin_row: []\n",
    ] {
        assert!(
            serde_yaml::from_str::<PresetConfig>(malformed).is_err(),
            "wrong-typed scheduler YAML must fail before runtime projection: {malformed:?}"
        );
    }

    let explicit_zero = serde_yaml::from_str::<PresetConfig>(
        "margin_row:\n  early_exit: false\n  verifier_threads: 0\n  concurrent_threads: 0\n",
    )
    .expect("canonical scheduler resource scalars must deserialize");
    assert_eq!(explicit_zero.margin_row.early_exit, Some(false));
    assert_eq!(explicit_zero.margin_row.verifier_threads, Some(0));
    assert_eq!(explicit_zero.margin_row.concurrent_threads, Some(0));

    let positive = serde_yaml::from_str::<PresetConfig>(
        "margin_row:\n  verifier_threads: 7\n  concurrent_threads: 11\n",
    )
    .expect("positive scheduler widths must deserialize");
    assert_eq!(positive.margin_row.verifier_threads, Some(7));
    assert_eq!(positive.margin_row.concurrent_threads, Some(11));
}
