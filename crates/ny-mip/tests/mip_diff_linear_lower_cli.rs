// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_mip::{dump, MilpProblem};
use std::path::PathBuf;
use std::process::Command;

fn temp_model_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ny-mip-diff-linear-lower-{}-{label}.milp",
        std::process::id()
    ))
}

fn run_replay_with_args(
    problem: &MilpProblem,
    label: &str,
    replay_args: &[&str],
) -> std::process::Output {
    let path = temp_model_path(label);
    std::fs::write(&path, dump::to_milp_text(problem)).expect("write replay fixture");
    let mut command = Command::new(env!("CARGO_BIN_EXE_mip-diff"));
    command.env_clear().args(replay_args).args([
        "--timeout",
        "5",
        path.to_str().expect("UTF-8 temp path"),
    ]);
    let output = command.output().expect("run mip-diff");
    let _ = std::fs::remove_file(path);
    output
}

fn run_replay(problem: &MilpProblem, label: &str) -> std::process::Output {
    run_replay_with_args(problem, label, &["--linear-lower-replay"])
}

fn add_canonical_relu(
    problem: &mut MilpProblem,
    lower: f64,
    upper: f64,
) -> (ny_mip::ir::Col, ny_mip::ir::Col) {
    let input = problem.add_col(0.0, lower, upper);
    let output = problem.add_col(0.0, 0.0, upper);
    let binary = problem.add_integer_col(0.0, 0.0, 1.0);
    problem.add_row(0.0, f64::INFINITY, [(output, 1.0), (input, -1.0)]);
    problem.add_row(
        f64::NEG_INFINITY,
        -lower,
        [(output, 1.0), (input, -1.0), (binary, -lower)],
    );
    problem.add_row(f64::NEG_INFINITY, 0.0, [(output, 1.0), (binary, -upper)]);
    (output, binary)
}

fn two_relu_decision() -> MilpProblem {
    let mut decision = MilpProblem::new();
    let (first_output, _) = add_canonical_relu(&mut decision, -1.0, 2.0);
    let (second_output, _) = add_canonical_relu(&mut decision, -1.0, 4.0);
    decision.add_row(
        f64::NEG_INFINITY,
        f64::from(-100.0_f32),
        [(first_output, -1.0), (second_output, -1.0)],
    );
    decision
}

fn three_relu_decision() -> MilpProblem {
    let mut decision = MilpProblem::new();
    let mut objective = Vec::new();
    for upper in [2.0, 4.0, 6.0] {
        let (output, _) = add_canonical_relu(&mut decision, -1.0, upper);
        objective.push((output, -1.0));
    }
    decision.add_row(f64::NEG_INFINITY, f64::from(-100.0_f32), objective);
    decision
}

fn four_relu_decision() -> MilpProblem {
    let mut decision = MilpProblem::new();
    let mut objective = Vec::new();
    for upper in [2.0, 4.0, 6.0, 8.0] {
        let (output, _) = add_canonical_relu(&mut decision, -1.0, upper);
        objective.push((output, -1.0));
    }
    decision.add_row(f64::NEG_INFINITY, f64::from(-100.0_f32), objective);
    decision
}

#[test]
fn linear_lower_cli_reports_certified_route_and_replay_counts() {
    let mut decision = MilpProblem::new();
    let x = decision.add_col(0.0, 1.0, 2.0);
    decision.add_row(f64::NEG_INFINITY, f64::from(0.99_f32), [(x, 1.0)]);

    let output = run_replay(&decision, "certified");
    assert!(
        output.status.success(),
        "mip-diff failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("certified"), "{stdout}");
    assert!(stdout.contains("relaxation-entailment"), "{stdout}");
    assert!(stdout.contains("1 certified, 0 inconclusive"), "{stdout}");
}

#[test]
fn linear_lower_cli_rejects_non_f32_threshold() {
    let mut decision = MilpProblem::new();
    let x = decision.add_col(0.0, -1.0, 1.0);
    decision.add_row(f64::NEG_INFINITY, 0.1_f64, [(x, 1.0)]);

    let output = run_replay(&decision, "invalid-threshold");
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("validation error"), "{stdout}");
    assert!(
        stdout.contains("exact widening of a finite f32"),
        "{stdout}"
    );
}

#[test]
fn split_replay_stays_single_column_and_rank_selects_the_control() {
    let decision = two_relu_decision();
    let rank_zero = run_replay_with_args(
        &decision,
        "split-rank-zero",
        &["--linear-lower-split-replay"],
    );
    assert!(
        rank_zero.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&rank_zero.stderr)
    );
    let stdout = String::from_utf8_lossy(&rank_zero.stdout);
    assert!(stdout.contains("split advice"), "{stdout}");
    assert!(stdout.contains("selected=1 ids=[c5]"), "{stdout}");

    let rank_one = run_replay_with_args(
        &decision,
        "split-rank-one",
        &[
            "--linear-lower-split-replay",
            "--linear-lower-split-rank",
            "1",
        ],
    );
    assert!(
        rank_one.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&rank_one.stderr)
    );
    let stdout = String::from_utf8_lossy(&rank_one.stdout);
    assert!(stdout.contains("selected=1 ids=[c2]"), "{stdout}");
}

#[test]
fn tree_replay_prints_exactly_two_ids_and_labels_selection_source() {
    let decision = two_relu_decision();
    let intercept =
        run_replay_with_args(&decision, "tree-intercept", &["--linear-lower-tree-replay"]);
    assert!(
        intercept.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&intercept.stderr)
    );
    let stdout = String::from_utf8_lossy(&intercept.stdout);
    assert!(stdout.contains("tree advice"), "{stdout}");
    assert!(stdout.contains("source=intercept"), "{stdout}");
    assert!(stdout.contains("selected=2 ids=[c5,c2]"), "{stdout}");

    let full = run_replay_with_args(
        &decision,
        "tree-full-babsr",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-full-babsr",
        ],
    );
    assert!(
        full.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&full.stderr)
    );
    let stdout = String::from_utf8_lossy(&full.stdout);
    assert!(stdout.contains("source=full-babsr-top2"), "{stdout}");
    assert!(stdout.contains("selected=2 ids=["), "{stdout}");
}

#[test]
fn full_babsr_selection_requires_tree_replay_mode() {
    let output = run_replay_with_args(
        &two_relu_decision(),
        "tree-mode-required",
        &["--linear-lower-tree-full-babsr"],
    );
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-full-babsr requires --linear-lower-tree-replay"),
        "{stderr}"
    );
}

#[test]
fn fixed_tree_replay_preserves_explicit_nonprefix_rank_order() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-ranks",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-fixed-ranks",
            "1,3,0,2",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("source=fixed-ranked-indices"), "{stdout}");
    assert!(stdout.contains("ranks=[1,3,0,2]"), "{stdout}");
    assert!(stdout.contains("selected=4 ids=[c8,c2,c11,c5]"), "{stdout}");
}

#[test]
fn fixed_tree_rank_flag_is_strictly_typed_and_scoped() {
    let mode_required = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-ranks-mode-required",
        &["--linear-lower-tree-fixed-ranks", "1,3,0,2"],
    );
    assert_eq!(mode_required.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&mode_required.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-fixed-ranks requires --linear-lower-tree-replay"),
        "{stderr}"
    );

    for (label, ranks, detail) in [
        ("tree-fixed-ranks-duplicate", "1,3,1", "rank 1 is repeated"),
        (
            "tree-fixed-ranks-too-many",
            "0,1,2,3,4",
            "expected one through four ranks",
        ),
        (
            "tree-fixed-ranks-out-of-range",
            "0,8",
            "rank 8 is outside 0..7",
        ),
    ] {
        let output = run_replay_with_args(
            &four_relu_decision(),
            label,
            &[
                "--linear-lower-tree-replay",
                "--linear-lower-tree-fixed-ranks",
                ranks,
            ],
        );
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(detail), "{stderr}");
    }

    let conflict = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-ranks-conflict",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-fixed-ranks",
            "1,3,0,2",
            "--linear-lower-tree-target-fsb",
        ],
    );
    assert_eq!(conflict.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&conflict.stderr);
    assert!(
        stderr.contains(
            "--linear-lower-tree-fixed-ranks is mutually exclusive with every other tree \
             selection mode"
        ),
        "{stderr}"
    );
}

#[test]
fn fixed_tree_col_replay_preserves_explicit_raw_order_without_ranking() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-cols",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-fixed-cols",
            "2,11,5,8",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("source=fixed-column-ids"), "{stdout}");
    assert!(stdout.contains("cols=[2,11,5,8]"), "{stdout}");
    assert!(stdout.contains("selected=4 ids=[c2,c11,c5,c8]"), "{stdout}");
}

#[test]
fn fixed_tree_col_flag_is_strictly_typed_and_scoped() {
    let mode_required = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-cols-mode-required",
        &["--linear-lower-tree-fixed-cols", "2,11,5,8"],
    );
    assert_eq!(mode_required.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&mode_required.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-fixed-cols requires --linear-lower-tree-replay"),
        "{stderr}"
    );

    for (label, cols, detail) in [
        (
            "tree-fixed-cols-duplicate",
            "2,11,2",
            "column ID 2 is repeated",
        ),
        (
            "tree-fixed-cols-empty",
            "2,,5",
            "column IDs may not contain an empty item",
        ),
        (
            "tree-fixed-cols-malformed",
            "2,nope",
            "`nope` is not a nonnegative integer column ID",
        ),
        (
            "tree-fixed-cols-too-many",
            "2,5,8,11,14",
            "expected one through four column IDs",
        ),
    ] {
        let output = run_replay_with_args(
            &four_relu_decision(),
            label,
            &[
                "--linear-lower-tree-replay",
                "--linear-lower-tree-fixed-cols",
                cols,
            ],
        );
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(detail), "{stderr}");
    }

    let repeated = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-cols-repeated",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-fixed-cols",
            "2,11",
            "--linear-lower-tree-fixed-cols",
            "5,8",
        ],
    );
    assert_eq!(repeated.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&repeated.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-fixed-cols may be supplied only once"),
        "{stderr}"
    );
}

#[test]
fn fixed_tree_col_flag_conflicts_with_ranked_tree_selection() {
    let fixed_rank_conflict = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-cols-rank-conflict",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-fixed-cols",
            "2,11,5,8",
            "--linear-lower-tree-fixed-ranks",
            "0,1,2,3",
        ],
    );
    assert_eq!(fixed_rank_conflict.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&fixed_rank_conflict.stderr);
    assert!(
        stderr.contains(
            "--linear-lower-tree-fixed-ranks and --linear-lower-tree-fixed-cols are mutually \
             exclusive"
        ),
        "{stderr}"
    );

    let target_fsb_conflict = run_replay_with_args(
        &four_relu_decision(),
        "tree-fixed-cols-target-fsb-conflict",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-fixed-cols",
            "2,11,5,8",
            "--linear-lower-tree-target-fsb",
        ],
    );
    assert_eq!(target_fsb_conflict.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&target_fsb_conflict.stderr);
    assert!(
        stderr.contains(
            "--linear-lower-tree-fixed-cols is mutually exclusive with every other tree \
             selection mode"
        ),
        "{stderr}"
    );
}

#[test]
fn fixed_tree_col_replay_reuses_strict_model_validation() {
    for (label, cols, detail) in [
        (
            "tree-fixed-cols-out-of-range",
            "99",
            "split 0 references column 99, but the model has 12 columns",
        ),
        (
            "tree-fixed-cols-continuous",
            "0",
            "split 0 column 0 is not an unfixed integer [0, 1] column",
        ),
    ] {
        let output = run_replay_with_args(
            &four_relu_decision(),
            label,
            &[
                "--linear-lower-tree-replay",
                "--linear-lower-tree-fixed-cols",
                cols,
            ],
        );
        assert_eq!(output.status.code(), Some(1));
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(detail), "{stdout}");
    }
}

#[test]
fn target_fsb_replay_labels_and_retains_the_dynamic_pool() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-target-fsb",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-target-fsb",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("source=target-fsb-full4-intercept4"),
        "{stdout}"
    );
    assert!(
        stdout.contains("target-FSB probe config: pivots_per_call=25 shared_ms=1500"),
        "{stdout}"
    );
    assert!(stdout.contains("selected=4 ids=[c11,c8,c5,c2]"), "{stdout}");
}

#[test]
fn target_fsb_replay_prints_effective_probe_limit_overrides() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-target-fsb-probe-limits",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-target-fsb",
            "--linear-lower-tree-target-fsb-probe-pivots",
            "7",
            "--linear-lower-tree-target-fsb-probe-ms",
            "123",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("target-FSB probe config: pivots_per_call=7 shared_ms=123"),
        "{stdout}"
    );
}

#[test]
fn adaptive_fsb_replay_uses_ranked_pool_and_prints_resolved_defaults() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-fsb-defaults",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-fsb",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("source=adaptive-fsb-full4-intercept4"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "adaptive-FSB config (measurement-only): root_rank=1 hard_value=1 \
             pivots_per_call=25 shared_ms=1500"
        ),
        "{stdout}"
    );
    assert!(stdout.contains("selected=4 ids=[c11,c8,c5,c2]"), "{stdout}");
    assert!(
        stdout.contains("root_rank=1 root=c8 hard_value=1"),
        "{stdout}"
    );
}

#[test]
fn adaptive_fsb_replay_honors_explicit_root_and_hard_side() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-fsb-explicit",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-fsb",
            "--linear-lower-tree-adaptive-fsb-root-rank",
            "0",
            "--linear-lower-tree-adaptive-fsb-hard-value",
            "0",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("root_rank=0 hard_value=0 pivots_per_call=25 shared_ms=1500"),
        "{stdout}"
    );
    assert!(
        stdout.contains("root_rank=0 root=c11 hard_value=0"),
        "{stdout}"
    );
}

#[test]
fn adaptive_fsb_mode_and_configuration_require_tree_replay() {
    let mode_only = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-fsb-mode-required",
        &["--linear-lower-tree-adaptive-fsb"],
    );
    assert_eq!(mode_only.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&mode_only.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-adaptive-fsb requires --linear-lower-tree-replay"),
        "{stderr}"
    );

    let config_only = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-fsb-config-required",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-fsb-root-rank",
            "1",
        ],
    );
    assert_eq!(config_only.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&config_only.stderr);
    assert!(
        stderr.contains(
            "adaptive-FSB root/hard flags require --linear-lower-tree-replay and \
             --linear-lower-tree-adaptive-fsb"
        ),
        "{stderr}"
    );
}

#[test]
fn adaptive_fsb_conflicts_with_complete_tree_selection_modes() {
    for (label, conflicting) in [
        ("adaptive-with-full-babsr", "--linear-lower-tree-full-babsr"),
        ("adaptive-with-target-fsb", "--linear-lower-tree-target-fsb"),
    ] {
        let output = run_replay_with_args(
            &four_relu_decision(),
            label,
            &[
                "--linear-lower-tree-replay",
                "--linear-lower-tree-adaptive-fsb",
                conflicting,
            ],
        );
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(
                "--linear-lower-tree-adaptive-fsb is mutually exclusive with \
                 --linear-lower-tree-full-babsr and --linear-lower-tree-target-fsb"
            ),
            "{stderr}"
        );
    }
}

#[test]
fn adaptive_fsb_rejects_invalid_hard_value_and_post_dedup_root_rank() {
    let invalid_hard = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-fsb-invalid-hard",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-fsb",
            "--linear-lower-tree-adaptive-fsb-hard-value",
            "2",
        ],
    );
    assert_eq!(invalid_hard.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&invalid_hard.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-adaptive-fsb-hard-value requires 0 or 1"),
        "{stderr}"
    );

    let invalid_root = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-fsb-invalid-root",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-fsb",
            "--linear-lower-tree-adaptive-fsb-root-rank",
            "4",
        ],
    );
    assert_eq!(invalid_root.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&invalid_root.stdout);
    assert!(
        stdout.contains("adaptive-FSB root rank 4 is outside 4 post-dedup candidates"),
        "{stdout}"
    );
}

#[test]
fn adaptive_comb_fsb_replay_uses_ranked_pool_and_prints_resolved_defaults() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-comb-fsb-defaults",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-comb-fsb",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("source=adaptive-comb-fsb-full4-intercept4"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "adaptive-comb-FSB config (measurement-only): root_rank=1 root_hard_value=1 \
             pivots_per_call=25 shared_ms=1500"
        ),
        "{stdout}"
    );
    assert!(stdout.contains("selected=4 ids=[c11,c8,c5,c2]"), "{stdout}");
    assert!(
        stdout.contains("root_rank=1 root=c8 root_hard_value=1"),
        "{stdout}"
    );
    assert!(
        stdout.contains("certified") || stdout.contains("inconclusive"),
        "{stdout}"
    );
}

#[test]
fn adaptive_comb_fsb_replay_honors_explicit_root_and_hard_side() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-comb-fsb-explicit",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-comb-fsb",
            "--linear-lower-tree-adaptive-comb-fsb-root-rank",
            "0",
            "--linear-lower-tree-adaptive-comb-fsb-root-hard-value",
            "0",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("root_rank=0 root_hard_value=0 pivots_per_call=25 shared_ms=1500"),
        "{stdout}"
    );
    assert!(
        stdout.contains("root_rank=0 root=c11 root_hard_value=0"),
        "{stdout}"
    );
}

#[test]
fn adaptive_comb_fsb_mode_and_configuration_require_tree_replay() {
    let mode_only = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-comb-fsb-mode-required",
        &["--linear-lower-tree-adaptive-comb-fsb"],
    );
    assert_eq!(mode_only.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&mode_only.stderr);
    assert!(
        stderr
            .contains("--linear-lower-tree-adaptive-comb-fsb requires --linear-lower-tree-replay"),
        "{stderr}"
    );

    let config_only = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-comb-fsb-config-required",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-comb-fsb-root-rank",
            "1",
        ],
    );
    assert_eq!(config_only.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&config_only.stderr);
    assert!(
        stderr.contains(
            "adaptive-comb-FSB root flags require --linear-lower-tree-replay and \
             --linear-lower-tree-adaptive-comb-fsb"
        ),
        "{stderr}"
    );
}

#[test]
fn adaptive_comb_fsb_conflicts_with_every_other_tree_selection_mode() {
    for (label, conflicting) in [
        (
            "adaptive-comb-with-full-babsr",
            "--linear-lower-tree-full-babsr",
        ),
        (
            "adaptive-comb-with-target-fsb",
            "--linear-lower-tree-target-fsb",
        ),
        (
            "adaptive-comb-with-adaptive-fsb",
            "--linear-lower-tree-adaptive-fsb",
        ),
        (
            "adaptive-comb-with-adaptive-five-comb",
            "--linear-lower-tree-adaptive-five-comb-fsb",
        ),
    ] {
        let output = run_replay_with_args(
            &four_relu_decision(),
            label,
            &[
                "--linear-lower-tree-replay",
                "--linear-lower-tree-adaptive-comb-fsb",
                conflicting,
            ],
        );
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(
                "--linear-lower-tree-adaptive-comb-fsb is mutually exclusive with \
                 --linear-lower-tree-full-babsr, --linear-lower-tree-target-fsb, and \
                 --linear-lower-tree-adaptive-fsb, and \
                 --linear-lower-tree-adaptive-five-comb-fsb"
            ),
            "{stderr}"
        );
    }
}

#[test]
fn adaptive_comb_fsb_rejects_invalid_hard_value_rank_and_candidate_count() {
    let invalid_hard = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-comb-fsb-invalid-hard",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-comb-fsb",
            "--linear-lower-tree-adaptive-comb-fsb-root-hard-value",
            "2",
        ],
    );
    assert_eq!(invalid_hard.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&invalid_hard.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-adaptive-comb-fsb-root-hard-value requires 0 or 1"),
        "{stderr}"
    );

    let invalid_root = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-comb-fsb-invalid-root",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-comb-fsb",
            "--linear-lower-tree-adaptive-comb-fsb-root-rank",
            "4",
        ],
    );
    assert_eq!(invalid_root.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&invalid_root.stdout);
    assert!(
        stdout.contains("adaptive-comb-FSB root rank 4 is outside 4 post-dedup candidates"),
        "{stdout}"
    );

    let too_few = run_replay_with_args(
        &two_relu_decision(),
        "tree-adaptive-comb-fsb-too-few",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-comb-fsb",
        ],
    );
    assert_eq!(too_few.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&too_few.stdout);
    assert!(
        stdout.contains(
            "adaptive-comb-FSB tree replay requires three to eight ranked binary candidates"
        ),
        "{stdout}"
    );
}

#[test]
fn adaptive_five_comb_fsb_replay_uses_ranked_pool_and_prints_resolved_defaults() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-five-comb-fsb-defaults",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-five-comb-fsb",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("source=adaptive-five-comb-fsb-full4-intercept4"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "adaptive-five-comb-FSB config (measurement-only): root_rank=1 \
             root_hard_value=1 pivots_per_call=25 shared_ms=1500"
        ),
        "{stdout}"
    );
    assert!(stdout.contains("selected=4 ids=[c11,c8,c5,c2]"), "{stdout}");
    assert!(
        stdout.contains("root_rank=1 root=c8 root_hard_value=1"),
        "{stdout}"
    );
    assert!(
        stdout.contains("certified") || stdout.contains("inconclusive"),
        "{stdout}"
    );
}

#[test]
fn adaptive_five_comb_fsb_cli_dispatches_to_exact_five_leaf_authority() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-five-comb-fsb-proof",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-five-comb-fsb",
            "--linear-lower-tree-adaptive-five-comb-fsb-root-rank",
            "0",
            "--linear-lower-tree-adaptive-five-comb-fsb-root-hard-value",
            "0",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = stdout
        .lines()
        .find(|line| line.contains("five-comb-fsb-proof.milp") && line.contains("certified"))
        .unwrap_or_else(|| panic!("missing certified result row:\n{stdout}"));
    let fields: Vec<&str> = result.split_whitespace().collect();
    assert_eq!(
        &fields[2..6],
        ["certified", "tree-farkas", "5", "5"],
        "{stdout}"
    );
}

#[test]
fn adaptive_five_comb_fsb_replay_honors_explicit_root_and_hard_side() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-five-comb-fsb-explicit",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-five-comb-fsb",
            "--linear-lower-tree-adaptive-five-comb-fsb-root-rank",
            "0",
            "--linear-lower-tree-adaptive-five-comb-fsb-root-hard-value",
            "0",
        ],
    );
    assert!(
        output.status.success(),
        "mip-diff failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("root_rank=0 root_hard_value=0 pivots_per_call=25 shared_ms=1500"),
        "{stdout}"
    );
    assert!(
        stdout.contains("root_rank=0 root=c11 root_hard_value=0"),
        "{stdout}"
    );
}

#[test]
fn adaptive_five_comb_fsb_mode_configuration_and_conflicts_fail_closed() {
    let mode_only = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-five-comb-fsb-mode-required",
        &["--linear-lower-tree-adaptive-five-comb-fsb"],
    );
    assert_eq!(mode_only.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&mode_only.stderr);
    assert!(
        stderr.contains(
            "--linear-lower-tree-adaptive-five-comb-fsb requires \
             --linear-lower-tree-replay"
        ),
        "{stderr}"
    );

    let config_only = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-five-comb-fsb-config-required",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-five-comb-fsb-root-rank",
            "1",
        ],
    );
    assert_eq!(config_only.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&config_only.stderr);
    assert!(
        stderr.contains(
            "adaptive-five-comb-FSB root flags require --linear-lower-tree-replay and \
             --linear-lower-tree-adaptive-five-comb-fsb"
        ),
        "{stderr}"
    );

    for (label, conflicting) in [
        (
            "adaptive-five-comb-with-full-babsr",
            "--linear-lower-tree-full-babsr",
        ),
        (
            "adaptive-five-comb-with-target-fsb",
            "--linear-lower-tree-target-fsb",
        ),
        (
            "adaptive-five-comb-with-adaptive-fsb",
            "--linear-lower-tree-adaptive-fsb",
        ),
        (
            "adaptive-five-comb-with-adaptive-comb",
            "--linear-lower-tree-adaptive-comb-fsb",
        ),
    ] {
        let output = run_replay_with_args(
            &four_relu_decision(),
            label,
            &[
                "--linear-lower-tree-replay",
                "--linear-lower-tree-adaptive-five-comb-fsb",
                conflicting,
            ],
        );
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("mutually exclusive"), "{stderr}");
    }
}

#[test]
fn adaptive_five_comb_fsb_rejects_invalid_hard_value_rank_and_candidate_count() {
    let invalid_hard = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-five-comb-fsb-invalid-hard",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-five-comb-fsb",
            "--linear-lower-tree-adaptive-five-comb-fsb-root-hard-value",
            "2",
        ],
    );
    assert_eq!(invalid_hard.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&invalid_hard.stderr);
    assert!(
        stderr
            .contains("--linear-lower-tree-adaptive-five-comb-fsb-root-hard-value requires 0 or 1"),
        "{stderr}"
    );

    let invalid_root = run_replay_with_args(
        &four_relu_decision(),
        "tree-adaptive-five-comb-fsb-invalid-root",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-five-comb-fsb",
            "--linear-lower-tree-adaptive-five-comb-fsb-root-rank",
            "4",
        ],
    );
    assert_eq!(invalid_root.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&invalid_root.stdout);
    assert!(
        stdout.contains("adaptive-five-comb-FSB root rank 4 is outside 4 post-dedup candidates"),
        "{stdout}"
    );

    let too_few = run_replay_with_args(
        &three_relu_decision(),
        "tree-adaptive-five-comb-fsb-too-few",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-adaptive-five-comb-fsb",
        ],
    );
    assert_eq!(too_few.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&too_few.stdout);
    assert!(
        stdout.contains(
            "adaptive-five-comb-FSB tree replay requires four to eight ranked binary candidates"
        ),
        "{stdout}"
    );
}

#[test]
fn target_fsb_selection_requires_tree_replay_mode() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-target-fsb-mode-required",
        &["--linear-lower-tree-target-fsb"],
    );
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--linear-lower-tree-target-fsb requires --linear-lower-tree-replay"),
        "{stderr}"
    );
}

#[test]
fn target_fsb_and_static_full_babsr_are_mutually_exclusive() {
    let output = run_replay_with_args(
        &four_relu_decision(),
        "tree-target-fsb-mutual-exclusion",
        &[
            "--linear-lower-tree-replay",
            "--linear-lower-tree-full-babsr",
            "--linear-lower-tree-target-fsb",
        ],
    );
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(
            "--linear-lower-tree-full-babsr and --linear-lower-tree-target-fsb are mutually exclusive"
        ),
        "{stderr}"
    );
}

#[test]
fn target_fsb_probe_limit_flags_require_target_fsb_tree_replay() {
    for (label, args) in [
        (
            "probe-pivots-mode-required",
            vec!["--linear-lower-tree-target-fsb-probe-pivots", "7"],
        ),
        (
            "probe-ms-mode-required",
            vec![
                "--linear-lower-tree-replay",
                "--linear-lower-tree-target-fsb-probe-ms",
                "123",
            ],
        ),
    ] {
        let output = run_replay_with_args(&four_relu_decision(), label, &args);
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(
                "target-FSB probe-limit flags require --linear-lower-tree-replay and \
                 --linear-lower-tree-target-fsb"
            ),
            "{stderr}"
        );
    }
}

#[test]
fn target_fsb_probe_limit_flags_reject_zero() {
    for (label, flag) in [
        (
            "probe-pivots-zero",
            "--linear-lower-tree-target-fsb-probe-pivots",
        ),
        ("probe-ms-zero", "--linear-lower-tree-target-fsb-probe-ms"),
    ] {
        let output = run_replay_with_args(
            &four_relu_decision(),
            label,
            &[
                "--linear-lower-tree-replay",
                "--linear-lower-tree-target-fsb",
                flag,
                "0",
            ],
        );
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("mip-diff: {flag} must be nonzero")),
            "{stderr}"
        );
    }
}
