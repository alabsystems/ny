// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cargo-owned inventory of the remaining pytest suites.
//!
//! This is migration control, not a claim that listing a Python test makes it
//! hermetic. New Python correctness tests must not silently enlarge the second
//! test framework: add the contract as a Rust/Cargo test instead. Existing
//! entries leave this manifest when their assertions migrate.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
enum Disposition {
    /// A repository-tool contract whose assertions must move to a Rust test.
    RustCargoMigration,
    /// Python-surface conformance, selected explicitly and required to fail
    /// fast if its interpreter, package, or built extension is unavailable.
    ExternalPythonConformance,
}

struct MigrationSuite {
    root: &'static str,
    disposition: Disposition,
    files: &'static [&'static str],
}

const TOOLING_TESTS: &[&str] = &[
    "test_abcrown_transfer_baseline.py",
    "test_abcrown_transfer_dispositions.py",
    "test_active_script_portability.py",
    "test_archive_vnncomp_sat_result.py",
    "test_audit_reachability_overtake.py",
    "test_audit_unsat_by_falsification.py",
    "test_benchmark_runner_exit_codes.py",
    "test_benchmark_shared.py",
    "test_benchmark_vnncomp_all_script.py",
    "test_benchmark_vnncomp_preset_bounded.py",
    "test_benchmark_vnncomp_script.py",
    "test_bounds_doc_citations.py",
    "test_canonicalize_vnncomp_ce_y.py",
    "test_ce_falsified_watchlist.py",
    "test_cifar100_attack_fast_kernels_qualify.py",
    "test_cifar100_bound_parity.py",
    "test_constrained_zonotope_dual_oracle.py",
    "test_constrained_zonotope_remainder_oracle.py",
    "test_download_benchmarks_script.py",
    "test_git_dep_pin_tooling.py",
    "test_gpu_crown_backward_refresh_policy.py",
    "test_gpu_crown_backward_regression.py",
    "test_gpu_crown_backward_regression_cli.py",
    "test_gpu_crown_backward_regression_duplicates.py",
    "test_gpu_crown_backward_regression_source_artifact.py",
    "test_install_tool_prebuilt.py",
    "test_legacy_helper_safety.py",
    "test_m7_cuda_validate_safety.py",
    "test_main16_gap_audit.py",
    "test_materialize_vnncomp2025_large_models.py",
    "test_measure_ny_scorecard_provenance.py",
    "test_ny_measurement_provenance.py",
    "test_ny_retroactive_scorecard.py",
    "test_ny_safe_gpu_run.py",
    "test_plan_biccos_mts_factorial.py",
    "test_plan_compact_tail_envelope.py",
    "test_profile_vnncomp_row.py",
    "test_promote_regular_bank.py",
    "test_python_api_script.py",
    "test_render_backend_benchmark_report.py",
    "test_render_backend_benchmark_report_provenance.py",
    "test_render_nvidia_vulkan_validation_report.py",
    "test_replay_ny_counterexamples.py",
    "test_replay_vnncomp2025_counterexample.py",
    "test_run_abcrown_transfer_factorials.py",
    "test_run_acasxu_benchmark_report.py",
    "test_run_instance_preset_resolution.py",
    "test_run_nvidia_vulkan_validation.py",
    "test_strict_ce_portability.py",
    "test_sync_vnncomp_reference_results.py",
    "test_system_health_check_syspolicyd.py",
    "test_validate_extended_bank.py",
    "test_validate_vnncomp_results.py",
    "test_verify_prebuilt.py",
    "test_vnncomp_ay_pin_coherence.py",
    "test_vnncomp_competitive_score.py",
    "test_vnncomp_dashboard.py",
    "test_vnncomp_dry_run.py",
    "test_vnncomp_monoculture.py",
    "test_vnncomp_prepare_instance_script.py",
    "test_vnncomp_publication.py",
    "test_vnncomp_trust_linux_build.py",
];

const PYTHON_BINDING_TESTS: &[&str] = &["test_ny.py", "test_types.py"];

const PYTEST_PLUGIN_TESTS: &[&str] = &["test_assertions.py", "test_config.py", "test_version.py"];

/// Exact functions deliberately absent from ordinary pytest collection. Each
/// file is named in TEST_CONFORMANCE.md with an explicit `external_*` selector.
const EXTERNAL_TOOLING_TESTS: &[&str] = &[
    "test_ce_falsified_watchlist.py::external_committed_watchlist_matches_the_official_artifacts",
    "test_m7_cuda_validate_safety.py::external_m7_overrides_inherited_degraded_build_inside_real_guard",
    "test_measure_ny_scorecard_provenance.py::external_completion_write_failure_propagates_nonzero_from_trap",
    "test_measure_ny_scorecard_provenance.py::external_configs_dir_is_passed_and_content_addressed",
    "test_measure_ny_scorecard_provenance.py::external_configs_dir_rejects_relative_or_missing_path",
    "test_measure_ny_scorecard_provenance.py::external_default_scratch_is_unique_and_keyed_by_run_id",
    "test_measure_ny_scorecard_provenance.py::external_exact_instance_selector_is_provenanced_and_runs_only_that_row",
    "test_measure_ny_scorecard_provenance.py::external_explicit_noncuda_debug_measurement_skips_cuda_selfcheck",
    "test_measure_ny_scorecard_provenance.py::external_explicit_solver_binary_is_allowed_captured_and_sealed",
    "test_measure_ny_scorecard_provenance.py::external_explicit_twenty_cpu_lane_is_attested_and_provenanced",
    "test_measure_ny_scorecard_provenance.py::external_integrity_failure_propagates_nonzero_from_completion_trap",
    "test_measure_ny_scorecard_provenance.py::external_isolated_output_reruns_legacy_row_and_honors_row_cap",
    "test_measure_ny_scorecard_provenance.py::external_missing_input_fails_without_emitting_unarchived_csv_row",
    "test_measure_ny_scorecard_provenance.py::external_multiple_vnnlib_versions_require_explicit_selection",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_ignores_ignored_sibling_python_bytecode",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_ignores_path_prepended_env_and_timeout_shims",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_reexecs_through_gpu_guard_before_measurement",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_refuses_measurement_without_gpu_guard",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_forged_gpu_guard_marker",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_invalid_containment_profile_before_guard",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_invalid_expected_cpu_selector_before_guard",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_loader_injection_before_any_guard_child",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_preexisting_isolation_paths",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_receiptless_automatic_binary_before_results",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_relative_scratch_before_solver",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_selected_cpu_policy_mismatch",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_self_erasing_bash_env_from_initial_snapshot",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_shell_function_attestation_spoofing",
    "test_measure_ny_scorecard_provenance.py::external_scorecard_rejects_symlinked_scratch_root",
    "test_measure_ny_scorecard_provenance.py::external_sealed_cuda_runtime_tamper_is_rejected_before_first_row",
    "test_measure_ny_scorecard_provenance.py::external_solver_and_identity_probes_receive_only_manifest_environment",
    "test_measure_ny_scorecard_provenance.py::external_source_cuda_symlink_retarget_after_start_cannot_change_measured_runtime",
    "test_measure_ny_scorecard_provenance.py::external_sweep_binds_rows_and_sat_artifact_to_completed_run",
    "test_measure_ny_scorecard_provenance.py::external_sweep_ignores_pythonpath_sitecustomize_before_rejecting_it",
    "test_measure_ny_scorecard_provenance.py::external_sweep_refuses_failed_sealed_cuda_selfcheck",
    "test_measure_ny_scorecard_provenance.py::external_sweep_rejects_ld_library_path_with_non_cuda_override",
    "test_measure_ny_scorecard_provenance.py::external_sweep_rejects_unreviewed_solver_runtime_environment",
    "test_measure_ny_scorecard_provenance.py::external_top_level_instances_list_precedes_nested_payload_list",
    "test_ny_measurement_provenance.py::external_non_utf8_byte_paths_are_deterministic_and_not_normalized",
    "test_ny_safe_gpu_run.py::external_guard_preserves_environment_status_and_attests_limits",
    "test_ny_safe_gpu_run.py::external_guard_preserves_literal_argument_boundaries",
    "test_ny_safe_gpu_run.py::external_sigterm_kills_the_complete_guarded_process_tree",
    "test_replay_ny_counterexamples.py::external_patched_relational_assignment_format_is_strict_official_witness",
    "test_replay_ny_counterexamples.py::external_real_pinned_official_v1_checker_smoke",
    "test_replay_ny_counterexamples.py::external_real_pinned_official_v2_checker_and_malformed_ny_syntax_smoke",
    "test_replay_vnncomp2025_counterexample.py::external_consumer_safe_bound_replay_executes_retained_worker",
    "test_replay_vnncomp2025_counterexample.py::external_official_checker_sources_match_commit_and_retained_copy",
    "test_replay_vnncomp2025_counterexample.py::external_retained_runtime_matches_all_exact_pins",
];

const MIGRATION_SUITES: &[MigrationSuite] = &[
    MigrationSuite {
        root: "tests",
        disposition: Disposition::RustCargoMigration,
        files: TOOLING_TESTS,
    },
    MigrationSuite {
        root: "crates/ny-python/tests",
        disposition: Disposition::ExternalPythonConformance,
        files: PYTHON_BINDING_TESTS,
    },
    MigrationSuite {
        root: "crates/ny-python/ny_pytest/tests",
        disposition: Disposition::ExternalPythonConformance,
        files: PYTEST_PLUGIN_TESTS,
    },
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|candidate| {
            fs::read_to_string(candidate.join("Cargo.toml"))
                .is_ok_and(|manifest| manifest.lines().any(|line| line.trim() == "[workspace]"))
        })
        .expect("ny-test-utils must live below the workspace manifest")
        .to_path_buf()
}

fn collect_python_tests(root: &Path, directory: &Path, tests: &mut Vec<String>) {
    let entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to inventory {}: {error}", directory.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "failed to inspect an entry below {}: {error}",
                directory.display()
            )
        });
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "failed to inspect file type for {}: {error}",
                path.display()
            )
        });
        if file_type.is_dir() {
            if entry.file_name() != "__pycache__" {
                collect_python_tests(root, &path, tests);
            }
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let is_pytest_module = file_name.starts_with("test_")
            || Path::new(file_name)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.ends_with("_test"));
        if file_type.is_file()
            && is_pytest_module
            && Path::new(file_name)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("py"))
        {
            tests.push(
                path.strip_prefix(root)
                    .expect("inventoried path must remain below its suite root")
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

fn collect_external_python_contracts(root: &Path, directory: &Path, tests: &mut Vec<String>) {
    let entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to inventory {}: {error}", directory.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "failed to inspect an entry below {}: {error}",
                directory.display()
            )
        });
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "failed to inspect file type for {}: {error}",
                path.display()
            )
        });
        if file_type.is_dir() {
            if entry.file_name() != "__pycache__" {
                collect_external_python_contracts(root, &path, tests);
            }
            continue;
        }
        if !file_type.is_file() || path.extension().is_none_or(|extension| extension != "py") {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .expect("external contract path must remain below tests/")
            .to_string_lossy()
            .replace('\\', "/");
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        for line in source.lines() {
            if let Some((name, _)) = line
                .strip_prefix("def external_")
                .and_then(|tail| tail.split_once('('))
            {
                tests.push(format!("{relative}::external_{name}"));
            }
        }
    }
}

#[test]
fn every_collected_python_test_has_an_explicit_migration_disposition() {
    let workspace = workspace_root();
    for suite in MIGRATION_SUITES {
        let root = workspace.join(suite.root);
        let mut actual = Vec::new();
        collect_python_tests(&root, &root, &mut actual);
        actual.sort_unstable();

        let mut expected: Vec<_> = suite.files.iter().map(|path| (*path).to_owned()).collect();
        expected.sort_unstable();
        let declared_entries = expected.len();
        expected.dedup();
        assert_eq!(
            expected.len(),
            declared_entries,
            "duplicate Python test in the migration manifest for {}",
            suite.root
        );
        assert_eq!(
            actual, expected,
            "Python test inventory drifted below {} ({:?}). Add new correctness contracts under \
             Cargo; remove migrated entries from this manifest. Python-surface conformance must \
             remain explicit and fail fast.",
            suite.root, suite.disposition
        );
    }
}

#[test]
fn every_decollected_python_contract_has_an_explicit_external_lane() {
    let workspace = workspace_root();
    let conformance = fs::read_to_string(workspace.join("TEST_CONFORMANCE.md"))
        .expect("TEST_CONFORMANCE.md must remain readable");

    let root = workspace.join("tests");
    let mut actual = Vec::new();
    collect_external_python_contracts(&root, &root, &mut actual);
    actual.sort_unstable();
    let actual_count = actual.len();
    actual.dedup();
    assert_eq!(
        actual.len(),
        actual_count,
        "duplicate external Python contract"
    );

    let mut expected: Vec<_> = EXTERNAL_TOOLING_TESTS
        .iter()
        .map(|contract| (*contract).to_owned())
        .collect();
    expected.sort_unstable();
    assert_eq!(
        actual, expected,
        "external Python contract inventory drifted; document every decollected contract"
    );

    let mut documented_files: Vec<_> = expected
        .iter()
        .map(|contract| {
            contract
                .split_once("::")
                .expect("valid contract identity")
                .0
        })
        .collect();
    documented_files.sort_unstable();
    documented_files.dedup();
    for file in documented_files {
        assert!(
            conformance.contains(file),
            "TEST_CONFORMANCE.md must document the external lane containing {file}"
        );
    }
}
