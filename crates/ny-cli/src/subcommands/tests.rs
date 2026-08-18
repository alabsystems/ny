// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for CLI subcommand parsing.

use super::{BackendArg, Cli, Commands, LayerNormModeArg, LogFormat, MipSolverArg};
use clap::{Command, CommandFactory, Parser};

#[test]
fn cli_beta_crown_default_mip_solver_is_preset_driven() {
    // No `--mip-solver` parses as None: the handler then applies the preset's
    // `solver.mip.mip_solver`, with ay as the only selectable solver.
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            assert_eq!(args.mip_solver, None);
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_beta_crown_cut_authority_defaults_off() {
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            assert!(
                !args.enable_cuts,
                "standalone beta-crown must not request quarantined cut authority by default"
            );
            assert!(!args.enable_near_miss_cuts);
            assert!(!args.proactive_cuts);
        }
        _ => panic!("expected beta-crown command"),
    }

    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx", "--enable-cuts"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => assert!(
            args.enable_cuts,
            "an explicit research request should parse before validation rejects it"
        ),
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_beta_crown_parses_queue_memory_cap() {
    let cli = Cli::parse_from([
        "ny",
        "beta-crown",
        "model.onnx",
        "--max-queue-bytes",
        "2147483648",
    ]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            assert_eq!(args.max_queue_bytes, Some(2 * 1024 * 1024 * 1024));
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_beta_crown_explicit_mip_solver_wins() {
    // An explicit `--mip-solver ay` must parse as Some(AY): it is the
    // one-flag rollback that overrides a preset's `solver.mip.mip_solver: scip`.
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx", "--mip-solver", "ay"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            assert_eq!(args.mip_solver, Some(MipSolverArg::AY));
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_into_parts_matches_parsed_defaults() {
    let cli = Cli::parse_from(["ny", "verify"]);
    let (verbose, log_format, command) = cli.into_parts();

    assert_eq!(verbose, 0);
    assert_eq!(log_format, LogFormat::Text);
    assert!(matches!(command, Commands::Verify(_)));
}

#[test]
fn cli_parses_inspect_cost_flag() {
    let cli = Cli::parse_from(["ny", "inspect", "model.onnx", "--cost", "--json"]);
    let (_, _, command) = cli.into_parts();

    assert!(matches!(
        command,
        Commands::Inspect {
            cost: true,
            json: true,
            ..
        }
    ));
}

#[test]
fn cli_parses_inspect_timing_profile_flag() {
    let cli = Cli::parse_from([
        "ny",
        "inspect",
        "model.onnx",
        "--cost",
        "--timing-profile",
        "profile.json",
    ]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::Inspect { timing_profile, .. } => {
            assert_eq!(timing_profile, Some("profile.json".into()));
        }
        _ => panic!("expected inspect command"),
    }
}

#[test]
fn cli_rejects_compare_backend_and_legacy_gpu_together() {
    let result = Cli::try_parse_from([
        "ny",
        "compare",
        "reference.onnx",
        "target.onnx",
        "--backend",
        "wgpu",
        "--gpu",
    ]);
    assert!(
        result.is_err(),
        "the legacy --gpu flag must not override an explicit --backend"
    );
}

fn subcommand_arg_help(command: &Command, arg_id: &str) -> String {
    command
        .get_arguments()
        .find(|arg| arg.get_id().as_str() == arg_id)
        .unwrap_or_else(|| panic!("expected {arg_id} argument"))
        .get_help()
        .unwrap_or_else(|| panic!("expected help for {arg_id}"))
        .to_string()
}

#[test]
fn cli_help_keeps_compare_and_diff_onnx_contracts() {
    let cmd = Cli::command();

    let compare = cmd
        .find_subcommand("compare")
        .expect("compare subcommand should exist");
    let compare_about = compare
        .get_about()
        .expect("compare about should be set")
        .to_string();
    assert!(
        compare_about.contains("single-input ONNX"),
        "compare about should state the single-input ONNX contract: {compare_about}"
    );

    let reference_help = subcommand_arg_help(compare, "reference");
    assert!(
        reference_help.contains("ONNX"),
        "compare reference help should mention ONNX: {reference_help}"
    );
    let target_help = subcommand_arg_help(compare, "target");
    assert!(
        target_help.contains("ONNX"),
        "compare target help should mention ONNX: {target_help}"
    );

    let diff = cmd
        .find_subcommand("diff")
        .expect("diff subcommand should exist");
    let diff_about = diff
        .get_about()
        .expect("diff about should be set")
        .to_string();
    assert!(
        diff_about.contains("ONNX"),
        "diff about should mention ONNX: {diff_about}"
    );
}

#[test]
fn cli_rejects_bench_backend_and_legacy_gpu_together() {
    let result = Cli::try_parse_from([
        "ny",
        "bench",
        "--benchmark",
        "full",
        "--backend",
        "wgpu",
        "--gpu",
    ]);
    assert!(
        result.is_err(),
        "the legacy --gpu flag must not override an explicit --backend"
    );
}

#[test]
fn cli_parses_verify_layernorm_ibp_validated_mode() {
    let cli = Cli::parse_from(["ny", "verify", "--layernorm-mode", "ibp-validated"]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::Verify(args) => {
            assert_eq!(args.layernorm_mode, LayerNormModeArg::IbpValidated);
        }
        _ => panic!("expected verify command"),
    }
}

#[test]
fn cli_parses_beta_crown_heuristic_softmax_flags() {
    let cli = Cli::parse_from([
        "ny",
        "beta-crown",
        "model.onnx",
        "--allow-heuristic-logsoftmax",
        "--allow-heuristic-softmax",
    ]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::BetaCrown(args) => {
            assert!(args.allow_heuristic_logsoftmax);
            assert!(args.allow_heuristic_softmax);
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_parses_beta_crown_backend_omitted_is_none() {
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx"]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::BetaCrown(args) => {
            assert!(
                args.backend.is_none(),
                "omitted --backend should parse as None"
            );
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_parses_beta_crown_backend_explicit_wgpu() {
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx", "--backend", "wgpu"]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::BetaCrown(args) => {
            assert_eq!(args.backend, Some(BackendArg::Wgpu));
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_parses_beta_crown_backend_explicit_cpu_is_some() {
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx", "--backend", "cpu"]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::BetaCrown(args) => {
            assert_eq!(
                args.backend,
                Some(BackendArg::Cpu),
                "explicit --backend cpu should be Some(Cpu), not None"
            );
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_beta_crown_pgd_attack_defaults_on() {
    // PGD falsification is now default-on so run_instance.sh needs no --pgd-attack.
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            assert!(args.pgd_attack, "pgd_attack must default to true");
            assert!(!args.no_pgd_attack, "no_pgd_attack must default to false");
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_beta_crown_no_pgd_attack_disables() {
    // `--no-pgd-attack` flips the effective value off (combined in main.rs as
    // `pgd_attack && !no_pgd_attack`).
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx", "--no-pgd-attack"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            // The raw `pgd_attack` is still its default (true); the disable comes
            // from `no_pgd_attack`. main.rs computes `pgd_attack && !no_pgd_attack`,
            // so the EFFECTIVE value must be off here.
            assert!(args.no_pgd_attack, "--no-pgd-attack must set no_pgd_attack");
            let effective_pgd = args.pgd_attack && !args.no_pgd_attack;
            assert!(
                !effective_pgd,
                "effective PGD must be off with --no-pgd-attack"
            );
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_beta_crown_pgd_attack_explicit_false_disables() {
    // `--pgd-attack=false` disables PGD without the separate negation flag.
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx", "--pgd-attack=false"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            assert!(!args.pgd_attack, "--pgd-attack=false must parse as false");
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_beta_crown_pgd_attack_bare_flag_stays_true() {
    // A bare `--pgd-attack` (no value) keeps it enabled.
    let cli = Cli::parse_from(["ny", "beta-crown", "model.onnx", "--pgd-attack"]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::BetaCrown(args) => {
            assert!(args.pgd_attack, "bare --pgd-attack must parse as true");
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_parses_beta_crown_input_split_metrics_jsonl_flag() {
    let cli = Cli::parse_from([
        "ny",
        "beta-crown",
        "model.onnx",
        "--input-split-metrics-jsonl",
        "metrics.jsonl",
    ]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::BetaCrown(args) => {
            assert_eq!(args.input_split_metrics_jsonl, Some("metrics.jsonl".into()));
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_parses_beta_crown_domain_batch_metrics_jsonl_flag() {
    let cli = Cli::parse_from([
        "ny",
        "beta-crown",
        "model.onnx",
        "--domain-batch-metrics-jsonl",
        "domain_metrics.jsonl",
    ]);
    let (_, _, command) = cli.into_parts();

    match command {
        Commands::BetaCrown(args) => {
            assert_eq!(
                args.domain_batch_metrics_jsonl,
                Some("domain_metrics.jsonl".into())
            );
        }
        _ => panic!("expected beta-crown command"),
    }
}

#[test]
fn cli_top_level_about_matches_repo_scope() {
    let cmd = Cli::command();
    let about = cmd
        .get_about()
        .expect("top-level about should be set")
        .to_string();

    assert!(
        !about.contains("Whisper scale"),
        "top-level about should not reference stale Whisper-scale scope: {about}"
    );
    assert!(
        about.contains("verification"),
        "top-level about should mention verification: {about}"
    );
}

#[test]
fn cli_version_comes_from_package_metadata() {
    let cmd = Cli::command();
    assert_eq!(
        cmd.get_version(),
        Some(env!("CARGO_PKG_VERSION")),
        "Clap version must track the package version"
    );
}

#[test]
fn cli_rejects_conflicting_certificate_flags() {
    let result = Cli::try_parse_from([
        "ny",
        "beta-crown",
        "model.onnx",
        "--no-certificate",
        "--emit-certificate",
        "model.cert.json",
    ]);
    assert!(
        result.is_err(),
        "contradictory certificate flags must not depend on argument precedence"
    );
}

#[test]
fn cli_vnncomp_v1_protocol_argv_is_unchanged() {
    // PROTOCOL PIN: run_instance.sh and vnncomp_sweep.rs both spell exactly
    // `ny vnncomp v1 CATEGORY ONNX VNNLIB RESULTS TIMEOUT [--configs-dir D]`.
    // Converting `v1` from a positional to a subcommand must be argv-invisible.
    let cli = Cli::parse_from([
        "ny",
        "vnncomp",
        "v1",
        "cifar100_2024",
        "model.onnx",
        "prop.vnnlib",
        "out/results.txt",
        "210.0",
        "--configs-dir",
        "configs",
    ]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::Vnncomp {
            action:
                super::VnncompAction::V1 {
                    category,
                    timeout_secs,
                    configs_dir,
                    ..
                },
        } => {
            assert_eq!(category, "cifar100_2024");
            assert_eq!(timeout_secs, 210, "fractional budget must floor to 210");
            assert_eq!(
                configs_dir.as_deref(),
                Some(std::path::Path::new("configs"))
            );
        }
        _ => panic!("expected vnncomp v1 command"),
    }
}

#[test]
fn cli_vnncomp_v1_help_marks_cgan_depth_two_production_disabled() {
    let command = Cli::command();
    let vnncomp = command
        .find_subcommand("vnncomp")
        .expect("vnncomp subcommand should exist");
    let v1 = vnncomp
        .find_subcommand("v1")
        .expect("vnncomp v1 subcommand should exist");
    let help = v1
        .get_long_about()
        .or_else(|| v1.get_about())
        .expect("vnncomp v1 help should exist")
        .to_string();
    assert!(help.contains("NY_CGAN_INPUT_LEAF=1"));
    assert!(help.contains("requires an `mip` build"));
    assert!(help.contains("depth-two replay is production-disabled"));
    assert!(help.contains("disabled_not_requested"));
}

#[test]
fn cli_vnncomp_plan_parses_instance_shape() {
    // I2: the plan printer takes the same instance quadruple as the scored
    // path, minus RESULTS_FILE, plus --json.
    let cli = Cli::parse_from([
        "ny",
        "vnncomp",
        "plan",
        "cifar100_2024",
        "model.onnx",
        "prop.vnnlib",
        "100",
        "--json",
    ]);
    let (_, _, command) = cli.into_parts();
    match command {
        Commands::Vnncomp {
            action:
                super::VnncompAction::Plan {
                    category,
                    budget_secs,
                    configs_dir,
                    json,
                    ..
                },
        } => {
            assert_eq!(category, "cifar100_2024");
            assert_eq!(budget_secs, 100);
            assert_eq!(configs_dir, None);
            assert!(json);
        }
        _ => panic!("expected vnncomp plan command"),
    }
}
