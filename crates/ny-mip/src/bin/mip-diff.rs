// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Backend differential + certificate harness over a dumped `.milp` corpus
// (gates G0/LG0/LG3).
//
// Usage:
//   NY_MIP_DUMP=corpus/ ny beta-crown ...       # capture production instances
//   mip-diff corpus/*.milp                      # default: ay (lib) vs ay-proc
//   mip-diff --timeout 60 corpus/               # a directory expands to *.milp
//   mip-diff --certify corpus/                  # LG3: ay vs its own certificates
//   mip-diff --linear-lower-replay decision.milp
//                                                # exact shared-tail proof replay
//   mip-diff --linear-lower-split-replay decision.milp
//                                                # opt-in BaBSR root-advice replay

#![forbid(unsafe_code)]

//   mip-diff --linear-lower-split-replay --linear-lower-split-rank 3 decision.milp
//                                                # probe another top-eight split
//   mip-diff --linear-lower-tree-replay decision.milp
//                                                # intercept-ranked depth-two replay
//   mip-diff --linear-lower-tree-replay --linear-lower-tree-full-babsr decision.milp
//                                                # winner-style pair selection
//   mip-diff --linear-lower-tree-replay --linear-lower-tree-target-fsb decision.milp
//                                                # dynamic target-FSB over a 4+4 pool
//   mip-diff --linear-lower-tree-replay --linear-lower-tree-target-fsb \
//     --linear-lower-tree-target-fsb-probe-pivots 64 \
//     --linear-lower-tree-target-fsb-probe-ms 3000 decision.milp
//                                                # measurement-only probe limits
//   mip-diff --linear-lower-tree-replay \
//     --linear-lower-tree-fixed-ranks 1,4,6,5 decision.milp
//                                                # fixed ordered 16-leaf replay
//   mip-diff --linear-lower-tree-replay \
//     --linear-lower-tree-fixed-cols 7719,7718,7717,7716 decision.milp
//                                                # fixed raw-column 16-leaf replay
//   mip-diff --linear-lower-tree-replay \
//     --linear-lower-tree-fixed-cols 7719,7718,7717,7716 \
//     --linear-lower-tree-parallel-workers 8 decision.milp
//                                                # bounded-parallel 16-leaf replay
//   mip-diff --linear-lower-tree-replay --linear-lower-tree-adaptive-fsb \
//     --linear-lower-tree-adaptive-fsb-root-rank 1 \
//     --linear-lower-tree-adaptive-fsb-hard-value 1 decision.milp
//                                                # adaptive three-leaf measurement
//   mip-diff --linear-lower-tree-replay --linear-lower-tree-adaptive-comb-fsb \
//     --linear-lower-tree-adaptive-comb-fsb-root-rank 1 \
//     --linear-lower-tree-adaptive-comb-fsb-root-hard-value 1 decision.milp
//                                                # adaptive four-leaf comb measurement
//   mip-diff --linear-lower-tree-replay --linear-lower-tree-adaptive-five-comb-fsb \
//     --linear-lower-tree-adaptive-five-comb-fsb-root-rank 1 \
//     --linear-lower-tree-adaptive-five-comb-fsb-root-hard-value 1 decision.milp
//                                                # adaptive five-leaf comb measurement
//
// Diff mode: solves with two backends and compares verdicts. Sat-vs-Unsat
// between backends is a DISAGREEMENT (exit 1) — one of the solvers is
// wrong about a bit-identical problem. Timeout/Error on either side is
// recorded but is not a disagreement. Wall time per backend feeds the
// baseline ledger (docs/AY_MIP_P0.md).
//
// Certify mode (LG3, replaces the deleted HiGHS oracle): solves with the
// production ay backend and holds every verdict to its own evidence — an
// UNSAT must carry a VERIFIED exact certificate (Farkas or case-split,
// checked at the seam), a SAT witness is re-checked downstream anyway.
// Reports certification coverage; exits 1 on any hard failure: a
// certificate that FAILED verification (the seam surfaces it as an error,
// never as a bare UNSAT), a solver error, or an unloadable file.
// Certificate ABSENCE is reported, not fatal: some exact-but-
// uncertifiable trees exist until P4 completes the factory.
//
// Backends: `ay` (in-process ay-milp library, the production default),
// `ay-proc` (frozen P0 subprocess lane, needs the `ay` binary via
// $NY_AY/$PATH).

use ny_mip::{
    certify_linear_lower_bound_at_with_ay,
    certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_unwired,
    certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired,
    certify_linear_lower_bound_at_with_ay_adaptive_three_leaf_target_fsb_unwired,
    certify_linear_lower_bound_at_with_ay_branch_advice,
    certify_linear_lower_bound_at_with_ay_branch_advice_with_target_fsb_probe_limits_unwired,
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_unwired,
    certify_linear_lower_bound_at_with_ay_parallel_selector_tree_unwired, dump,
    ir::Col,
    solver::{
        rank_canonical_relu_binaries_for_lower_form,
        rank_canonical_relu_binaries_for_lower_form_full_babsr_union,
    },
    CertifiedLinearLowerDecisionConfig, CertifiedLinearLowerProofRoute,
    CertifiedLinearLowerTargetFsbProbeLimits, MilpProblem, MipBackend, MipConfig, MipResult,
    MipSolver, CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn usage() -> ! {
    eprintln!(
        "usage: mip-diff [--timeout <secs>] [--backends <a>,<b>] \
         [--certify | --linear-lower-replay | --linear-lower-split-replay | \
         --linear-lower-tree-replay] \
         [--linear-lower-split-rank <0..7>] \
         [--linear-lower-tree-full-babsr] \
         [--linear-lower-tree-target-fsb] \
         [--linear-lower-tree-target-fsb-probe-pivots <positive integer>] \
         [--linear-lower-tree-target-fsb-probe-ms <positive integer>] \
         [--linear-lower-tree-fixed-ranks <rank[,rank...]>] \
         [--linear-lower-tree-fixed-cols <col[,col...]>] \
         [--linear-lower-tree-parallel-workers <1..16>] \
         [--linear-lower-tree-adaptive-fsb] \
         [--linear-lower-tree-adaptive-fsb-root-rank <zero-based rank>] \
         [--linear-lower-tree-adaptive-fsb-hard-value <0|1>] \
         [--linear-lower-tree-adaptive-comb-fsb] \
         [--linear-lower-tree-adaptive-comb-fsb-root-rank <zero-based rank>] \
         [--linear-lower-tree-adaptive-comb-fsb-root-hard-value <0|1>] \
         [--linear-lower-tree-adaptive-five-comb-fsb] \
         [--linear-lower-tree-adaptive-five-comb-fsb-root-rank <zero-based rank>] \
         [--linear-lower-tree-adaptive-five-comb-fsb-root-hard-value <0|1>] \
         <file.milp | dir>...\n\
         backends: ay | ay-proc (default: ay,ay-proc)"
    );
    std::process::exit(2)
}

fn parse_backend(name: &str) -> MipBackend {
    match name {
        "ay" => MipBackend::Ay,
        "ay-proc" => MipBackend::AyProc,
        _ => usage(),
    }
}

fn backend_name(b: MipBackend) -> &'static str {
    match b {
        MipBackend::Ay => "ay",
        MipBackend::AyProc => "ay-proc",
    }
}

fn main() {
    let mut timeout_secs = 300.0_f64;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut pair = (MipBackend::Ay, MipBackend::AyProc);
    let mut certify = false;
    let mut linear_lower_replay = false;
    let mut linear_lower_split_replay = false;
    let mut linear_lower_tree_replay = false;
    let mut linear_lower_tree_full_babsr = false;
    let mut linear_lower_tree_target_fsb = false;
    let mut linear_lower_tree_adaptive_fsb = false;
    let mut linear_lower_tree_adaptive_comb_fsb = false;
    let mut linear_lower_tree_adaptive_five_comb_fsb = false;
    let mut linear_lower_tree_target_fsb_probe_pivots = None;
    let mut linear_lower_tree_target_fsb_probe_ms = None;
    let mut linear_lower_tree_fixed_ranks = None;
    let mut linear_lower_tree_fixed_cols = None;
    let mut linear_lower_tree_parallel_workers = None;
    let mut linear_lower_tree_adaptive_fsb_root_rank = 1usize;
    let mut linear_lower_tree_adaptive_fsb_root_rank_set = false;
    let mut linear_lower_tree_adaptive_fsb_hard_value = true;
    let mut linear_lower_tree_adaptive_fsb_hard_value_set = false;
    let mut linear_lower_tree_adaptive_comb_fsb_root_rank = 1usize;
    let mut linear_lower_tree_adaptive_comb_fsb_root_rank_set = false;
    let mut linear_lower_tree_adaptive_comb_fsb_root_hard_value = true;
    let mut linear_lower_tree_adaptive_comb_fsb_root_hard_value_set = false;
    let mut linear_lower_tree_adaptive_five_comb_fsb_root_rank = 1usize;
    let mut linear_lower_tree_adaptive_five_comb_fsb_root_rank_set = false;
    let mut linear_lower_tree_adaptive_five_comb_fsb_root_hard_value = true;
    let mut linear_lower_tree_adaptive_five_comb_fsb_root_hard_value_set = false;
    let mut linear_lower_split_rank = 0usize;
    let mut linear_lower_split_rank_set = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--timeout" => {
                let Some(v) = args.next().and_then(|s| s.parse().ok()) else {
                    usage()
                };
                timeout_secs = v;
            }
            "--certify" => certify = true,
            "--linear-lower-replay" => linear_lower_replay = true,
            "--linear-lower-split-replay" => linear_lower_split_replay = true,
            "--linear-lower-tree-replay" => linear_lower_tree_replay = true,
            "--linear-lower-tree-full-babsr" => linear_lower_tree_full_babsr = true,
            "--linear-lower-tree-target-fsb" => linear_lower_tree_target_fsb = true,
            "--linear-lower-tree-adaptive-fsb" => linear_lower_tree_adaptive_fsb = true,
            "--linear-lower-tree-adaptive-comb-fsb" => linear_lower_tree_adaptive_comb_fsb = true,
            "--linear-lower-tree-adaptive-five-comb-fsb" => {
                linear_lower_tree_adaptive_five_comb_fsb = true
            }
            "--linear-lower-tree-target-fsb-probe-pivots" => {
                let Some(value) = args.next().and_then(|value| value.parse::<u64>().ok()) else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-target-fsb-probe-pivots requires a \
                         positive integer"
                    );
                    usage()
                };
                if value == 0 {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-target-fsb-probe-pivots must be nonzero"
                    );
                    usage();
                }
                linear_lower_tree_target_fsb_probe_pivots = Some(value);
            }
            "--linear-lower-tree-target-fsb-probe-ms" => {
                let Some(value) = args.next().and_then(|value| value.parse::<u64>().ok()) else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-target-fsb-probe-ms requires a \
                         positive integer"
                    );
                    usage()
                };
                if value == 0 {
                    eprintln!("mip-diff: --linear-lower-tree-target-fsb-probe-ms must be nonzero");
                    usage();
                }
                linear_lower_tree_target_fsb_probe_ms = Some(value);
            }
            "--linear-lower-tree-fixed-ranks" => {
                let Some(value) = args.next() else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-fixed-ranks requires one through four \
                         comma-separated ranks in 0..7"
                    );
                    usage()
                };
                if linear_lower_tree_fixed_ranks.is_some() {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-fixed-ranks may be supplied only once"
                    );
                    usage();
                }
                linear_lower_tree_fixed_ranks = Some(
                    LinearLowerFixedTreeRanks::parse(&value).unwrap_or_else(|error| {
                        eprintln!("mip-diff: invalid --linear-lower-tree-fixed-ranks: {error}");
                        usage()
                    }),
                );
            }
            "--linear-lower-tree-fixed-cols" => {
                let Some(value) = args.next() else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-fixed-cols requires one through four \
                         comma-separated column IDs"
                    );
                    usage()
                };
                if linear_lower_tree_fixed_cols.is_some() {
                    eprintln!("mip-diff: --linear-lower-tree-fixed-cols may be supplied only once");
                    usage();
                }
                linear_lower_tree_fixed_cols = Some(
                    LinearLowerFixedTreeCols::parse(&value).unwrap_or_else(|error| {
                        eprintln!("mip-diff: invalid --linear-lower-tree-fixed-cols: {error}");
                        usage()
                    }),
                );
            }
            "--linear-lower-tree-parallel-workers" => {
                let Some(value) = args.next() else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-parallel-workers requires an integer in \
                         1..=16"
                    );
                    usage()
                };
                if linear_lower_tree_parallel_workers.is_some() {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-parallel-workers may be supplied only once"
                    );
                    usage();
                }
                linear_lower_tree_parallel_workers = Some(
                    parse_parallel_selector_workers(&value).unwrap_or_else(|error| {
                        eprintln!(
                            "mip-diff: invalid --linear-lower-tree-parallel-workers: {error}"
                        );
                        usage()
                    }),
                );
            }
            "--linear-lower-tree-adaptive-fsb-root-rank" => {
                let Some(value) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-adaptive-fsb-root-rank requires a \
                         zero-based nonnegative integer"
                    );
                    usage()
                };
                linear_lower_tree_adaptive_fsb_root_rank = value;
                linear_lower_tree_adaptive_fsb_root_rank_set = true;
            }
            "--linear-lower-tree-adaptive-fsb-hard-value" => {
                let Some(value) = args.next() else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-adaptive-fsb-hard-value requires 0 or 1"
                    );
                    usage()
                };
                linear_lower_tree_adaptive_fsb_hard_value = match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => {
                        eprintln!(
                            "mip-diff: --linear-lower-tree-adaptive-fsb-hard-value requires 0 or 1"
                        );
                        usage()
                    }
                };
                linear_lower_tree_adaptive_fsb_hard_value_set = true;
            }
            "--linear-lower-tree-adaptive-comb-fsb-root-rank" => {
                let Some(value) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-adaptive-comb-fsb-root-rank requires a \
                         zero-based nonnegative integer"
                    );
                    usage()
                };
                linear_lower_tree_adaptive_comb_fsb_root_rank = value;
                linear_lower_tree_adaptive_comb_fsb_root_rank_set = true;
            }
            "--linear-lower-tree-adaptive-comb-fsb-root-hard-value" => {
                let Some(value) = args.next() else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-adaptive-comb-fsb-root-hard-value \
                         requires 0 or 1"
                    );
                    usage()
                };
                linear_lower_tree_adaptive_comb_fsb_root_hard_value = match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => {
                        eprintln!(
                            "mip-diff: --linear-lower-tree-adaptive-comb-fsb-root-hard-value \
                             requires 0 or 1"
                        );
                        usage()
                    }
                };
                linear_lower_tree_adaptive_comb_fsb_root_hard_value_set = true;
            }
            "--linear-lower-tree-adaptive-five-comb-fsb-root-rank" => {
                let Some(value) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-adaptive-five-comb-fsb-root-rank requires \
                         a zero-based nonnegative integer"
                    );
                    usage()
                };
                linear_lower_tree_adaptive_five_comb_fsb_root_rank = value;
                linear_lower_tree_adaptive_five_comb_fsb_root_rank_set = true;
            }
            "--linear-lower-tree-adaptive-five-comb-fsb-root-hard-value" => {
                let Some(value) = args.next() else {
                    eprintln!(
                        "mip-diff: --linear-lower-tree-adaptive-five-comb-fsb-root-hard-value \
                         requires 0 or 1"
                    );
                    usage()
                };
                linear_lower_tree_adaptive_five_comb_fsb_root_hard_value = match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => {
                        eprintln!(
                            "mip-diff: --linear-lower-tree-adaptive-five-comb-fsb-root-hard-value \
                             requires 0 or 1"
                        );
                        usage()
                    }
                };
                linear_lower_tree_adaptive_five_comb_fsb_root_hard_value_set = true;
            }
            "--linear-lower-split-rank" => {
                let Some(rank) = args.next().and_then(|value| value.parse().ok()) else {
                    usage()
                };
                linear_lower_split_rank = rank;
                linear_lower_split_rank_set = true;
            }
            "--backends" => {
                let Some(v) = args.next() else { usage() };
                let mut parts = v.splitn(2, ',');
                let (Some(a), Some(b)) = (parts.next(), parts.next()) else {
                    usage()
                };
                pair = (parse_backend(a), parse_backend(b));
            }
            "--help" | "-h" => usage(),
            _ => inputs.push(PathBuf::from(arg)),
        }
    }
    let replay_mode_count = usize::from(certify)
        + usize::from(linear_lower_replay)
        + usize::from(linear_lower_split_replay)
        + usize::from(linear_lower_tree_replay);
    if replay_mode_count > 1 {
        eprintln!(
            "mip-diff: --certify, --linear-lower-replay, \
             --linear-lower-split-replay, and --linear-lower-tree-replay are mutually exclusive"
        );
        usage();
    }
    if !linear_lower_split_replay && linear_lower_split_rank_set {
        eprintln!("mip-diff: --linear-lower-split-rank requires --linear-lower-split-replay");
        usage();
    }
    if (linear_lower_tree_adaptive_fsb_root_rank_set
        || linear_lower_tree_adaptive_fsb_hard_value_set)
        && !(linear_lower_tree_replay && linear_lower_tree_adaptive_fsb)
    {
        eprintln!(
            "mip-diff: adaptive-FSB root/hard flags require \
             --linear-lower-tree-replay and --linear-lower-tree-adaptive-fsb"
        );
        usage();
    }
    if (linear_lower_tree_adaptive_comb_fsb_root_rank_set
        || linear_lower_tree_adaptive_comb_fsb_root_hard_value_set)
        && !(linear_lower_tree_replay && linear_lower_tree_adaptive_comb_fsb)
    {
        eprintln!(
            "mip-diff: adaptive-comb-FSB root flags require \
             --linear-lower-tree-replay and --linear-lower-tree-adaptive-comb-fsb"
        );
        usage();
    }
    if (linear_lower_tree_adaptive_five_comb_fsb_root_rank_set
        || linear_lower_tree_adaptive_five_comb_fsb_root_hard_value_set)
        && !(linear_lower_tree_replay && linear_lower_tree_adaptive_five_comb_fsb)
    {
        eprintln!(
            "mip-diff: adaptive-five-comb-FSB root flags require \
             --linear-lower-tree-replay and --linear-lower-tree-adaptive-five-comb-fsb"
        );
        usage();
    }
    if (linear_lower_tree_target_fsb_probe_pivots.is_some()
        || linear_lower_tree_target_fsb_probe_ms.is_some())
        && !(linear_lower_tree_replay && linear_lower_tree_target_fsb)
    {
        eprintln!(
            "mip-diff: target-FSB probe-limit flags require \
             --linear-lower-tree-replay and --linear-lower-tree-target-fsb"
        );
        usage();
    }
    if linear_lower_tree_fixed_ranks.is_some() && !linear_lower_tree_replay {
        eprintln!("mip-diff: --linear-lower-tree-fixed-ranks requires --linear-lower-tree-replay");
        usage();
    }
    if linear_lower_tree_fixed_cols.is_some() && !linear_lower_tree_replay {
        eprintln!("mip-diff: --linear-lower-tree-fixed-cols requires --linear-lower-tree-replay");
        usage();
    }
    if linear_lower_tree_fixed_ranks.is_some() && linear_lower_tree_fixed_cols.is_some() {
        eprintln!(
            "mip-diff: --linear-lower-tree-fixed-ranks and \
             --linear-lower-tree-fixed-cols are mutually exclusive"
        );
        usage();
    }
    if let Err(error) = validate_parallel_selector_cli(
        linear_lower_tree_replay,
        linear_lower_tree_fixed_ranks,
        linear_lower_tree_fixed_cols,
        linear_lower_tree_parallel_workers,
    ) {
        eprintln!("mip-diff: {error}");
        usage();
    }
    if linear_lower_tree_fixed_ranks.is_some()
        && (linear_lower_tree_full_babsr
            || linear_lower_tree_target_fsb
            || linear_lower_tree_adaptive_fsb
            || linear_lower_tree_adaptive_comb_fsb
            || linear_lower_tree_adaptive_five_comb_fsb)
    {
        eprintln!(
            "mip-diff: --linear-lower-tree-fixed-ranks is mutually exclusive with every other \
             tree selection mode"
        );
        usage();
    }
    if linear_lower_tree_fixed_cols.is_some()
        && (linear_lower_tree_full_babsr
            || linear_lower_tree_target_fsb
            || linear_lower_tree_adaptive_fsb
            || linear_lower_tree_adaptive_comb_fsb
            || linear_lower_tree_adaptive_five_comb_fsb)
    {
        eprintln!(
            "mip-diff: --linear-lower-tree-fixed-cols is mutually exclusive with every other \
             tree selection mode"
        );
        usage();
    }
    if linear_lower_tree_full_babsr && linear_lower_tree_target_fsb {
        eprintln!(
            "mip-diff: --linear-lower-tree-full-babsr and \
             --linear-lower-tree-target-fsb are mutually exclusive"
        );
        usage();
    }
    if linear_lower_tree_adaptive_fsb
        && (linear_lower_tree_full_babsr || linear_lower_tree_target_fsb)
    {
        eprintln!(
            "mip-diff: --linear-lower-tree-adaptive-fsb is mutually exclusive with \
             --linear-lower-tree-full-babsr and --linear-lower-tree-target-fsb"
        );
        usage();
    }
    if linear_lower_tree_adaptive_comb_fsb
        && (linear_lower_tree_full_babsr
            || linear_lower_tree_target_fsb
            || linear_lower_tree_adaptive_fsb
            || linear_lower_tree_adaptive_five_comb_fsb)
    {
        eprintln!(
            "mip-diff: --linear-lower-tree-adaptive-comb-fsb is mutually exclusive with \
             --linear-lower-tree-full-babsr, --linear-lower-tree-target-fsb, and \
             --linear-lower-tree-adaptive-fsb, and \
             --linear-lower-tree-adaptive-five-comb-fsb"
        );
        usage();
    }
    if linear_lower_tree_adaptive_five_comb_fsb
        && (linear_lower_tree_full_babsr
            || linear_lower_tree_target_fsb
            || linear_lower_tree_adaptive_fsb
            || linear_lower_tree_adaptive_comb_fsb)
    {
        eprintln!(
            "mip-diff: --linear-lower-tree-adaptive-five-comb-fsb is mutually exclusive with \
             --linear-lower-tree-full-babsr, --linear-lower-tree-target-fsb, \
             --linear-lower-tree-adaptive-fsb, and --linear-lower-tree-adaptive-comb-fsb"
        );
        usage();
    }
    if !linear_lower_tree_replay && linear_lower_tree_full_babsr {
        eprintln!("mip-diff: --linear-lower-tree-full-babsr requires --linear-lower-tree-replay");
        usage();
    }
    if !linear_lower_tree_replay && linear_lower_tree_target_fsb {
        eprintln!("mip-diff: --linear-lower-tree-target-fsb requires --linear-lower-tree-replay");
        usage();
    }
    if !linear_lower_tree_replay && linear_lower_tree_adaptive_fsb {
        eprintln!("mip-diff: --linear-lower-tree-adaptive-fsb requires --linear-lower-tree-replay");
        usage();
    }
    if !linear_lower_tree_replay && linear_lower_tree_adaptive_comb_fsb {
        eprintln!(
            "mip-diff: --linear-lower-tree-adaptive-comb-fsb requires \
             --linear-lower-tree-replay"
        );
        usage();
    }
    if !linear_lower_tree_replay && linear_lower_tree_adaptive_five_comb_fsb {
        eprintln!(
            "mip-diff: --linear-lower-tree-adaptive-five-comb-fsb requires \
             --linear-lower-tree-replay"
        );
        usage();
    }
    if inputs.is_empty() {
        usage();
    }

    let production_probe_limits = CertifiedLinearLowerTargetFsbProbeLimits::production();
    let target_fsb_probe_limits = CertifiedLinearLowerTargetFsbProbeLimits::new(
        linear_lower_tree_target_fsb_probe_pivots
            .unwrap_or(production_probe_limits.max_probe_pivots_per_call()),
        linear_lower_tree_target_fsb_probe_ms.map_or_else(
            || production_probe_limits.probe_time_limit(),
            Duration::from_millis,
        ),
    )
    .unwrap_or_else(|error| {
        eprintln!("mip-diff: invalid target-FSB probe limits: {error}");
        usage()
    });

    let files = collect_files(&inputs);
    if files.is_empty() {
        eprintln!("mip-diff: no .milp files found");
        std::process::exit(2);
    }

    if certify {
        run_certify(&files, timeout_secs);
    }
    if linear_lower_replay {
        run_linear_lower_replay(&files, timeout_secs);
    }
    if linear_lower_split_replay {
        run_linear_lower_split_replay(&files, timeout_secs, linear_lower_split_rank);
    }
    if linear_lower_tree_replay {
        run_linear_lower_tree_replay(
            &files,
            timeout_secs,
            linear_lower_tree_full_babsr,
            linear_lower_tree_target_fsb,
            linear_lower_tree_adaptive_fsb,
            linear_lower_tree_adaptive_fsb_root_rank,
            linear_lower_tree_adaptive_fsb_hard_value,
            linear_lower_tree_adaptive_comb_fsb,
            linear_lower_tree_adaptive_comb_fsb_root_rank,
            linear_lower_tree_adaptive_comb_fsb_root_hard_value,
            linear_lower_tree_adaptive_five_comb_fsb,
            linear_lower_tree_adaptive_five_comb_fsb_root_rank,
            linear_lower_tree_adaptive_five_comb_fsb_root_hard_value,
            linear_lower_tree_fixed_ranks,
            linear_lower_tree_fixed_cols,
            linear_lower_tree_parallel_workers,
            target_fsb_probe_limits,
        );
    }

    let (left, right) = pair;
    println!(
        "{:<40} {:>6} {:>6}  {:>9} {:>9}  {:>9} {:>9}",
        "instance",
        "cols",
        "rows",
        backend_name(left),
        "t(s)",
        backend_name(right),
        "t(s)"
    );
    let mut disagreements = 0usize;
    for path in &files {
        match run_one(path, timeout_secs, pair) {
            Ok(disagreed) => disagreements += usize::from(disagreed),
            Err(e) => {
                println!("{:<40} load error: {e}", short_name(path));
                disagreements += 1; // an unloadable corpus file fails the gate
            }
        }
    }
    println!(
        "\n{} instance(s), {} disagreement(s)",
        files.len(),
        disagreements
    );
    if disagreements > 0 {
        std::process::exit(1);
    }
}

struct LinearLowerReplayRequest {
    base: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
}

/// Recover the exact request that produced a certified-linear-lower fallback
/// dump. That seam appends one canonical `objective <= f64::from(q)` row to an
/// unmarked base problem, where `q` is a finite binary32 threshold.
fn extract_linear_lower_request(
    decision: &MilpProblem,
) -> Result<LinearLowerReplayRequest, String> {
    if let Some(row) = decision.margin_row() {
        return Err(format!(
            "captured decision model has unexpected margin marker on row {}",
            row.0
        ));
    }
    let decision_row = decision
        .rows()
        .last()
        .ok_or_else(|| "captured decision model has no final decision row".to_owned())?;
    if decision_row.lb.to_bits() != f64::NEG_INFINITY.to_bits() {
        return Err("final decision row lower bound is not exactly -infinity".to_owned());
    }
    if !decision_row.ub.is_finite() {
        return Err("final decision row upper bound is not finite".to_owned());
    }
    let requested_lower = decision_row.ub as f32;
    if !requested_lower.is_finite()
        || f64::from(requested_lower).to_bits() != decision_row.ub.to_bits()
    {
        return Err(
            "final decision row upper bound is not the exact widening of a finite f32".to_owned(),
        );
    }
    if decision_row.coeffs.is_empty() {
        return Err("final decision row has an empty objective".to_owned());
    }

    let mut objective = Vec::with_capacity(decision_row.coeffs.len());
    let mut previous_col = None;
    for (term, &(col, coefficient)) in decision_row.coeffs.iter().enumerate() {
        if col >= decision.num_cols() {
            return Err(format!(
                "final decision objective term {term} references column {col}, \
                 but the model has {} columns",
                decision.num_cols()
            ));
        }
        if !coefficient.is_finite() {
            return Err(format!(
                "final decision objective term {term} has a non-finite coefficient"
            ));
        }
        if coefficient == 0.0 {
            return Err(format!(
                "final decision objective term {term} has a zero coefficient"
            ));
        }
        if previous_col.is_some_and(|previous| col <= previous) {
            return Err(
                "final decision objective columns are not in strictly increasing canonical order"
                    .to_owned(),
            );
        }
        previous_col = Some(col);
        objective.push((Col(col), coefficient));
    }

    let mut base = MilpProblem::new();
    for column in decision.cols() {
        if column.integer {
            base.add_integer_col(column.obj, column.lb, column.ub);
        } else {
            base.add_col(column.obj, column.lb, column.ub);
        }
    }
    for row in &decision.rows()[..decision.num_rows() - 1] {
        base.add_row(
            row.lb,
            row.ub,
            row.coeffs
                .iter()
                .map(|&(col, coefficient)| (Col(col), coefficient)),
        );
    }

    Ok(LinearLowerReplayRequest {
        base,
        objective,
        requested_lower,
    })
}

fn proof_route_name(route: CertifiedLinearLowerProofRoute) -> &'static str {
    match route {
        CertifiedLinearLowerProofRoute::RelaxationEntailment => "relaxation-entailment",
        CertifiedLinearLowerProofRoute::RootFarkas => "root-farkas",
        CertifiedLinearLowerProofRoute::TreeFarkas => "tree-farkas",
    }
}

fn print_linear_lower_result(
    path: &Path,
    threshold: Option<f32>,
    verdict: &str,
    route: &str,
    leaves: Option<usize>,
    replays: Option<usize>,
    elapsed_secs: f64,
    detail: Option<&str>,
) {
    let threshold = threshold
        .map(|value| format!("0x{:08x}", value.to_bits()))
        .unwrap_or_else(|| "-".to_owned());
    let leaves = leaves
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_owned());
    let replays = replays
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_owned());
    println!(
        "{:<40} {:>10} {:>13} {:>22} {:>8} {:>8} {:>9.3}{}",
        short_name(path),
        threshold,
        verdict,
        route,
        leaves,
        replays,
        elapsed_secs,
        detail.map(|text| format!("  {text}")).unwrap_or_default()
    );
}

/// Replay the complete production decision-only authority path: reconstruct
/// the pre-decision model, retry the relaxation entailment, then (if needed)
/// solve and independently replay the stamped MILP proof with the production
/// 4,096-leaf ceiling.
fn run_linear_lower_replay(files: &[PathBuf], timeout_secs: f64) -> ! {
    let mut certified = 0usize;
    let mut inconclusive = 0usize;
    let mut failures = 0usize;
    println!(
        "{:<40} {:>10} {:>13} {:>22} {:>8} {:>8} {:>9}",
        "instance", "threshold", "verdict", "route", "leaves", "replays", "t(s)"
    );
    for path in files {
        let request = std::fs::read_to_string(path)
            .map_err(|error| format!("load error: {error}"))
            .and_then(|text| {
                dump::from_milp_text(&text).map_err(|error| format!("load error: {error}"))
            })
            .and_then(|decision| {
                extract_linear_lower_request(&decision)
                    .map_err(|error| format!("validation error: {error}"))
            });
        let request = match request {
            Ok(request) => request,
            Err(error) => {
                failures += 1;
                print_linear_lower_result(path, None, "error", "-", None, None, 0.0, Some(&error));
                continue;
            }
        };

        let started = Instant::now();
        let result = certify_linear_lower_bound_at_with_ay(
            &request.base,
            &request.objective,
            request.requested_lower,
            CertifiedLinearLowerDecisionConfig {
                proof_timeout_secs: timeout_secs,
                max_tree_leaves: CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
            },
        );
        let elapsed_secs = started.elapsed().as_secs_f64();
        match result {
            Ok(Some(proof)) => {
                if proof.lower.to_bits() != request.requested_lower.to_bits() {
                    failures += 1;
                    print_linear_lower_result(
                        path,
                        Some(request.requested_lower),
                        "error",
                        "-",
                        None,
                        None,
                        elapsed_secs,
                        Some("certified result returned a different threshold"),
                    );
                    continue;
                }
                certified += 1;
                print_linear_lower_result(
                    path,
                    Some(request.requested_lower),
                    "certified",
                    proof_route_name(proof.proof_route),
                    Some(proof.ay_tree_leaves),
                    Some(proof.ny_cert_farkas_replays),
                    elapsed_secs,
                    None,
                );
            }
            Ok(None) => {
                inconclusive += 1;
                print_linear_lower_result(
                    path,
                    Some(request.requested_lower),
                    "inconclusive",
                    "-",
                    None,
                    None,
                    elapsed_secs,
                    None,
                );
            }
            Err(error) => {
                failures += 1;
                let detail = error.to_string();
                print_linear_lower_result(
                    path,
                    Some(request.requested_lower),
                    "error",
                    "-",
                    None,
                    None,
                    elapsed_secs,
                    Some(&detail),
                );
            }
        }
    }
    println!(
        "\n{} instance(s): {certified} certified, {inconclusive} inconclusive, \
         {failures} failure(s)",
        files.len()
    );
    std::process::exit(if failures > 0 { 1 } else { 0 })
}

const LINEAR_LOWER_SPLIT_REPLAY_ADVICE_CAP: usize = 8;
const LINEAR_LOWER_TREE_REPLAY_DEPTH: usize = 2;
const LINEAR_LOWER_TARGET_FSB_CANDIDATES_PER_SCORE: usize = 4;
const LINEAR_LOWER_TARGET_FSB_MIN_CANDIDATES: usize = 3;
const LINEAR_LOWER_ADAPTIVE_FIVE_COMB_MIN_CANDIDATES: usize = 4;
const LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES: usize =
    2 * LINEAR_LOWER_TARGET_FSB_CANDIDATES_PER_SCORE;
const LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH: usize = 4;
const LINEAR_LOWER_PARALLEL_SELECTOR_MAX_WORKERS: usize = 1 << LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH;

/// Ordered indices into the diagnostic full-BaBSR/intercept candidate pool.
///
/// A fixed-size carrier keeps the CLI mode typed and copyable while preserving
/// the user's exact order. Validation occurs before any model is loaded, so a
/// duplicate or malformed rank can never silently retarget a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearLowerFixedTreeRanks {
    ranks: [usize; 4],
    len: usize,
}

impl LinearLowerFixedTreeRanks {
    fn parse(value: &str) -> Result<Self, String> {
        let mut ranks = [0usize; 4];
        let mut len = 0usize;
        for token in value.split(',') {
            if token.is_empty() {
                return Err("ranks may not contain an empty item".to_owned());
            }
            if len == ranks.len() {
                return Err("expected one through four ranks".to_owned());
            }
            let rank = token
                .parse::<usize>()
                .map_err(|_| format!("`{token}` is not a nonnegative integer"))?;
            if rank >= LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES {
                return Err(format!(
                    "rank {rank} is outside 0..{}",
                    LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES - 1
                ));
            }
            if ranks[..len].contains(&rank) {
                return Err(format!("rank {rank} is repeated"));
            }
            ranks[len] = rank;
            len += 1;
        }
        if len == 0 {
            return Err("expected one through four ranks".to_owned());
        }
        Ok(Self { ranks, len })
    }

    fn as_slice(&self) -> &[usize] {
        &self.ranks[..self.len]
    }
}

/// Ordered raw column IDs for a diagnostic fixed-assignment tree.
///
/// Unlike ranked controls, these IDs are never scored or retargeted. Model-
/// dependent validation remains centralized in the fixed-assignment proof API,
/// which rejects stale, nonbinary, fixed, or duplicate columns without
/// changing caller order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearLowerFixedTreeCols {
    cols: [usize; 4],
    len: usize,
}

impl LinearLowerFixedTreeCols {
    fn parse(value: &str) -> Result<Self, String> {
        let mut cols = [0usize; 4];
        let mut len = 0usize;
        for token in value.split(',') {
            if token.is_empty() {
                return Err("column IDs may not contain an empty item".to_owned());
            }
            if len == cols.len() {
                return Err("expected one through four column IDs".to_owned());
            }
            let col = token
                .parse::<usize>()
                .map_err(|_| format!("`{token}` is not a nonnegative integer column ID"))?;
            if cols[..len].contains(&col) {
                return Err(format!("column ID {col} is repeated"));
            }
            cols[len] = col;
            len += 1;
        }
        if len == 0 {
            return Err("expected one through four column IDs".to_owned());
        }
        Ok(Self { cols, len })
    }

    fn as_slice(&self) -> &[usize] {
        &self.cols[..self.len]
    }
}

fn parse_parallel_selector_workers(value: &str) -> Result<usize, String> {
    let workers = value
        .parse::<usize>()
        .map_err(|_| format!("`{value}` is not a nonnegative integer"))?;
    if !(1..=LINEAR_LOWER_PARALLEL_SELECTOR_MAX_WORKERS).contains(&workers) {
        return Err(format!(
            "worker count {workers} is outside 1..={LINEAR_LOWER_PARALLEL_SELECTOR_MAX_WORKERS}"
        ));
    }
    Ok(workers)
}

fn validate_parallel_selector_cli(
    tree_replay: bool,
    fixed_ranks: Option<LinearLowerFixedTreeRanks>,
    fixed_cols: Option<LinearLowerFixedTreeCols>,
    max_workers: Option<usize>,
) -> Result<(), String> {
    let Some(max_workers) = max_workers else {
        return Ok(());
    };
    if !tree_replay {
        return Err(
            "--linear-lower-tree-parallel-workers requires --linear-lower-tree-replay".to_owned(),
        );
    }
    if !(1..=LINEAR_LOWER_PARALLEL_SELECTOR_MAX_WORKERS).contains(&max_workers) {
        return Err(format!(
            "--linear-lower-tree-parallel-workers must be in \
             1..={LINEAR_LOWER_PARALLEL_SELECTOR_MAX_WORKERS}"
        ));
    }
    let selector_count = match (fixed_ranks, fixed_cols) {
        (Some(ranks), None) => ranks.as_slice().len(),
        (None, Some(cols)) => cols.as_slice().len(),
        (None, None) => {
            return Err(
                "--linear-lower-tree-parallel-workers requires exactly four selectors from \
                 --linear-lower-tree-fixed-ranks or --linear-lower-tree-fixed-cols"
                    .to_owned(),
            );
        }
        (Some(_), Some(_)) => {
            return Err(
                "--linear-lower-tree-parallel-workers accepts only one fixed selector source"
                    .to_owned(),
            );
        }
    };
    if selector_count != LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH {
        return Err(format!(
            "--linear-lower-tree-parallel-workers requires exactly \
             {LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH} fixed selectors, got {selector_count}"
        ));
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct LinearLowerReplayAdvice {
    valid_binary_count: usize,
    ranked_binary_count: usize,
    selected: Vec<Col>,
}

fn valid_binary_cols(problem: &MilpProblem) -> Vec<Col> {
    problem
        .cols()
        .iter()
        .enumerate()
        .filter_map(|(index, column)| {
            (column.integer
                && column.lb.to_bits() == 0.0_f64.to_bits()
                && column.ub.to_bits() == 1.0_f64.to_bits())
            .then_some(Col(index))
        })
        .collect()
}

fn linear_lower_split_replay_advice(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    primary_rank: usize,
) -> LinearLowerReplayAdvice {
    let valid_binaries = valid_binary_cols(problem);
    let ranked = rank_canonical_relu_binaries_for_lower_form(problem, objective, &valid_binaries);
    let ranked_binary_count = ranked.len();
    let selected = ranked
        .get(primary_rank)
        .copied()
        .filter(|_| primary_rank < LINEAR_LOWER_SPLIT_REPLAY_ADVICE_CAP)
        .into_iter()
        .collect();
    LinearLowerReplayAdvice {
        valid_binary_count: valid_binaries.len(),
        ranked_binary_count,
        selected,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearLowerTreeReplaySelection {
    Intercept,
    FullBabsr,
    TargetFsb,
    AdaptiveFsb,
    AdaptiveCombFsb,
    AdaptiveFiveCombFsb,
}

impl LinearLowerTreeReplaySelection {
    fn name(self) -> &'static str {
        match self {
            Self::Intercept => "intercept",
            Self::FullBabsr => "full-babsr-top2",
            Self::TargetFsb => "target-fsb-full4-intercept4",
            Self::AdaptiveFsb => "adaptive-fsb-full4-intercept4",
            Self::AdaptiveCombFsb => "adaptive-comb-fsb-full4-intercept4",
            Self::AdaptiveFiveCombFsb => "adaptive-five-comb-fsb-full4-intercept4",
        }
    }
}

fn linear_lower_tree_replay_advice(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    selection: LinearLowerTreeReplaySelection,
) -> LinearLowerReplayAdvice {
    let valid_binaries = valid_binary_cols(problem);
    let mut ranked = match selection {
        LinearLowerTreeReplaySelection::Intercept => {
            rank_canonical_relu_binaries_for_lower_form(problem, objective, &valid_binaries)
        }
        LinearLowerTreeReplaySelection::FullBabsr => {
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                problem,
                objective,
                &valid_binaries,
                LINEAR_LOWER_SPLIT_REPLAY_ADVICE_CAP,
            )
        }
        LinearLowerTreeReplaySelection::TargetFsb
        | LinearLowerTreeReplaySelection::AdaptiveFsb
        | LinearLowerTreeReplaySelection::AdaptiveCombFsb
        | LinearLowerTreeReplaySelection::AdaptiveFiveCombFsb => {
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                problem,
                objective,
                &valid_binaries,
                LINEAR_LOWER_TARGET_FSB_CANDIDATES_PER_SCORE,
            )
        }
    };
    let ranked_binary_count = ranked.len();
    match selection {
        LinearLowerTreeReplaySelection::Intercept | LinearLowerTreeReplaySelection::FullBabsr => {
            ranked.truncate(LINEAR_LOWER_TREE_REPLAY_DEPTH);
        }
        LinearLowerTreeReplaySelection::TargetFsb
        | LinearLowerTreeReplaySelection::AdaptiveFsb
        | LinearLowerTreeReplaySelection::AdaptiveCombFsb
        | LinearLowerTreeReplaySelection::AdaptiveFiveCombFsb => {
            ranked.truncate(LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES);
        }
    }
    LinearLowerReplayAdvice {
        valid_binary_count: valid_binaries.len(),
        ranked_binary_count,
        selected: ranked,
    }
}

fn linear_lower_fixed_tree_replay_advice(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    ranks: LinearLowerFixedTreeRanks,
) -> LinearLowerReplayAdvice {
    let valid_binaries = valid_binary_cols(problem);
    let mut ranked = rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
        problem,
        objective,
        &valid_binaries,
        LINEAR_LOWER_TARGET_FSB_CANDIDATES_PER_SCORE,
    );
    let ranked_binary_count = ranked.len();
    ranked.truncate(LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES);
    let selected = ranks
        .as_slice()
        .iter()
        .filter_map(|&rank| ranked.get(rank).copied())
        .collect();
    LinearLowerReplayAdvice {
        valid_binary_count: valid_binaries.len(),
        ranked_binary_count,
        selected,
    }
}

fn linear_lower_fixed_col_tree_replay_advice(
    problem: &MilpProblem,
    cols: LinearLowerFixedTreeCols,
) -> LinearLowerReplayAdvice {
    LinearLowerReplayAdvice {
        valid_binary_count: valid_binary_cols(problem).len(),
        ranked_binary_count: 0,
        selected: cols.as_slice().iter().copied().map(Col).collect(),
    }
}

fn format_advice_ids(cols: &[Col]) -> String {
    let ids: Vec<String> = cols
        .iter()
        .map(|col| {
            let index = col.0;
            format!("c{index}")
        })
        .collect();
    format!("[{}]", ids.join(","))
}

fn print_split_replay_advice(path: &Path, advice: &LinearLowerReplayAdvice) {
    println!(
        "mip-diff: split advice {}: valid_binaries={} ranked={} selected={} ids={}",
        short_name(path),
        advice.valid_binary_count,
        advice.ranked_binary_count,
        advice.selected.len(),
        format_advice_ids(&advice.selected),
    );
}

fn print_tree_replay_advice(
    path: &Path,
    selection: LinearLowerTreeReplaySelection,
    advice: &LinearLowerReplayAdvice,
) {
    println!(
        "mip-diff: tree advice {}: source={} valid_binaries={} ranked={} selected={} ids={}",
        short_name(path),
        selection.name(),
        advice.valid_binary_count,
        advice.ranked_binary_count,
        advice.selected.len(),
        format_advice_ids(&advice.selected),
    );
}

fn print_fixed_tree_replay_advice(
    path: &Path,
    ranks: LinearLowerFixedTreeRanks,
    advice: &LinearLowerReplayAdvice,
) {
    let ranks = ranks
        .as_slice()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "mip-diff: tree advice {}: source=fixed-ranked-indices ranks=[{}] valid_binaries={} \
         ranked={} selected={} ids={}",
        short_name(path),
        ranks,
        advice.valid_binary_count,
        advice.ranked_binary_count,
        advice.selected.len(),
        format_advice_ids(&advice.selected),
    );
}

fn print_fixed_col_tree_replay_advice(
    path: &Path,
    cols: LinearLowerFixedTreeCols,
    advice: &LinearLowerReplayAdvice,
) {
    let cols = cols
        .as_slice()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "mip-diff: tree advice {}: source=fixed-column-ids cols=[{}] valid_binaries={} \
         selected={} ids={}",
        short_name(path),
        cols,
        advice.valid_binary_count,
        advice.selected.len(),
        format_advice_ids(&advice.selected),
    );
}

/// Opt-in comparison lane for target-guided root branching. Request
/// reconstruction and proof reporting intentionally mirror the production
/// control replay above; only the explicitly printed advice and authority call
/// differ.
fn run_linear_lower_split_replay(files: &[PathBuf], timeout_secs: f64, primary_rank: usize) -> ! {
    run_linear_lower_advised_replay(
        files,
        timeout_secs,
        LinearLowerAdvisedReplay::Split { primary_rank },
        CertifiedLinearLowerTargetFsbProbeLimits::production(),
    )
}

/// Opt-in tree comparison lane. The ordinary scorer's top two are the stable
/// default, while the winner-style full-BaBSR top two remain an explicit static
/// control. Complete and adaptive target-FSB are separate measurement modes
/// that pass the same deduplicated full-score-four plus intercept-score-four
/// pool to their respective proof authorities.
fn run_linear_lower_tree_replay(
    files: &[PathBuf],
    timeout_secs: f64,
    full_babsr: bool,
    target_fsb: bool,
    adaptive_fsb: bool,
    adaptive_fsb_root_rank: usize,
    adaptive_fsb_hard_value: bool,
    adaptive_comb_fsb: bool,
    adaptive_comb_fsb_root_rank: usize,
    adaptive_comb_fsb_root_hard_value: bool,
    adaptive_five_comb_fsb: bool,
    adaptive_five_comb_fsb_root_rank: usize,
    adaptive_five_comb_fsb_root_hard_value: bool,
    fixed_ranks: Option<LinearLowerFixedTreeRanks>,
    fixed_cols: Option<LinearLowerFixedTreeCols>,
    parallel_workers: Option<usize>,
    target_fsb_probe_limits: CertifiedLinearLowerTargetFsbProbeLimits,
) -> ! {
    let selection = if adaptive_five_comb_fsb {
        LinearLowerTreeReplaySelection::AdaptiveFiveCombFsb
    } else if adaptive_comb_fsb {
        LinearLowerTreeReplaySelection::AdaptiveCombFsb
    } else if adaptive_fsb {
        LinearLowerTreeReplaySelection::AdaptiveFsb
    } else if target_fsb {
        LinearLowerTreeReplaySelection::TargetFsb
    } else if full_babsr {
        LinearLowerTreeReplaySelection::FullBabsr
    } else {
        LinearLowerTreeReplaySelection::Intercept
    };
    if target_fsb {
        println!(
            "mip-diff: target-FSB probe config: pivots_per_call={} shared_ms={}",
            target_fsb_probe_limits.max_probe_pivots_per_call(),
            target_fsb_probe_limits.probe_time_limit().as_millis(),
        );
    }
    if adaptive_fsb {
        let fixed_limits = CertifiedLinearLowerTargetFsbProbeLimits::production();
        println!(
            "mip-diff: adaptive-FSB config (measurement-only): root_rank={} hard_value={} \
             pivots_per_call={} shared_ms={}",
            adaptive_fsb_root_rank,
            u8::from(adaptive_fsb_hard_value),
            fixed_limits.max_probe_pivots_per_call(),
            fixed_limits.probe_time_limit().as_millis(),
        );
    }
    if adaptive_comb_fsb {
        let fixed_limits = CertifiedLinearLowerTargetFsbProbeLimits::production();
        println!(
            "mip-diff: adaptive-comb-FSB config (measurement-only): root_rank={} \
             root_hard_value={} pivots_per_call={} shared_ms={}",
            adaptive_comb_fsb_root_rank,
            u8::from(adaptive_comb_fsb_root_hard_value),
            fixed_limits.max_probe_pivots_per_call(),
            fixed_limits.probe_time_limit().as_millis(),
        );
    }
    if adaptive_five_comb_fsb {
        let fixed_limits = CertifiedLinearLowerTargetFsbProbeLimits::production();
        println!(
            "mip-diff: adaptive-five-comb-FSB config (measurement-only): root_rank={} \
             root_hard_value={} pivots_per_call={} shared_ms={}",
            adaptive_five_comb_fsb_root_rank,
            u8::from(adaptive_five_comb_fsb_root_hard_value),
            fixed_limits.max_probe_pivots_per_call(),
            fixed_limits.probe_time_limit().as_millis(),
        );
    }
    if let Some(max_workers) = parallel_workers {
        println!(
            "mip-diff: parallel selector-tree config (measurement-only): \
             max_workers={max_workers} shared_timeout_secs={timeout_secs:.3}"
        );
    }
    let mode = if let Some(mode) = fixed_tree_replay_mode(fixed_ranks, fixed_cols, parallel_workers)
    {
        mode
    } else if adaptive_five_comb_fsb {
        LinearLowerAdvisedReplay::AdaptiveFiveCombTree {
            root_rank: adaptive_five_comb_fsb_root_rank,
            root_hard_value: adaptive_five_comb_fsb_root_hard_value,
        }
    } else if adaptive_comb_fsb {
        LinearLowerAdvisedReplay::AdaptiveCombTree {
            root_rank: adaptive_comb_fsb_root_rank,
            root_hard_value: adaptive_comb_fsb_root_hard_value,
        }
    } else if adaptive_fsb {
        LinearLowerAdvisedReplay::AdaptiveTree {
            root_rank: adaptive_fsb_root_rank,
            hard_value: adaptive_fsb_hard_value,
        }
    } else {
        LinearLowerAdvisedReplay::Tree { selection }
    };
    run_linear_lower_advised_replay(files, timeout_secs, mode, target_fsb_probe_limits)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearLowerAdvisedReplay {
    Split {
        primary_rank: usize,
    },
    Tree {
        selection: LinearLowerTreeReplaySelection,
    },
    FixedAssignmentTree {
        ranks: LinearLowerFixedTreeRanks,
    },
    FixedColumnAssignmentTree {
        cols: LinearLowerFixedTreeCols,
    },
    ParallelFixedAssignmentTree {
        ranks: LinearLowerFixedTreeRanks,
        max_workers: usize,
    },
    ParallelFixedColumnAssignmentTree {
        cols: LinearLowerFixedTreeCols,
        max_workers: usize,
    },
    AdaptiveTree {
        root_rank: usize,
        hard_value: bool,
    },
    AdaptiveCombTree {
        root_rank: usize,
        root_hard_value: bool,
    },
    AdaptiveFiveCombTree {
        root_rank: usize,
        root_hard_value: bool,
    },
}

fn fixed_tree_replay_mode(
    fixed_ranks: Option<LinearLowerFixedTreeRanks>,
    fixed_cols: Option<LinearLowerFixedTreeCols>,
    parallel_workers: Option<usize>,
) -> Option<LinearLowerAdvisedReplay> {
    if let Some(cols) = fixed_cols {
        return Some(parallel_workers.map_or(
            LinearLowerAdvisedReplay::FixedColumnAssignmentTree { cols },
            |max_workers| LinearLowerAdvisedReplay::ParallelFixedColumnAssignmentTree {
                cols,
                max_workers,
            },
        ));
    }
    fixed_ranks.map(|ranks| {
        parallel_workers.map_or(
            LinearLowerAdvisedReplay::FixedAssignmentTree { ranks },
            |max_workers| LinearLowerAdvisedReplay::ParallelFixedAssignmentTree {
                ranks,
                max_workers,
            },
        )
    })
}

fn run_linear_lower_advised_replay(
    files: &[PathBuf],
    timeout_secs: f64,
    mode: LinearLowerAdvisedReplay,
    target_fsb_probe_limits: CertifiedLinearLowerTargetFsbProbeLimits,
) -> ! {
    let mut certified = 0usize;
    let mut inconclusive = 0usize;
    let mut failures = 0usize;
    println!(
        "{:<40} {:>10} {:>13} {:>22} {:>8} {:>8} {:>9}",
        "instance", "threshold", "verdict", "route", "leaves", "replays", "t(s)"
    );
    for path in files {
        let request = std::fs::read_to_string(path)
            .map_err(|error| format!("load error: {error}"))
            .and_then(|text| {
                dump::from_milp_text(&text).map_err(|error| format!("load error: {error}"))
            })
            .and_then(|decision| {
                extract_linear_lower_request(&decision)
                    .map_err(|error| format!("validation error: {error}"))
            });
        let request = match request {
            Ok(request) => request,
            Err(error) => {
                failures += 1;
                print_linear_lower_result(path, None, "error", "-", None, None, 0.0, Some(&error));
                continue;
            }
        };

        let advice = match mode {
            LinearLowerAdvisedReplay::Split { primary_rank } => {
                linear_lower_split_replay_advice(&request.base, &request.objective, primary_rank)
            }
            LinearLowerAdvisedReplay::Tree { selection } => {
                linear_lower_tree_replay_advice(&request.base, &request.objective, selection)
            }
            LinearLowerAdvisedReplay::FixedAssignmentTree { ranks }
            | LinearLowerAdvisedReplay::ParallelFixedAssignmentTree { ranks, .. } => {
                linear_lower_fixed_tree_replay_advice(&request.base, &request.objective, ranks)
            }
            LinearLowerAdvisedReplay::FixedColumnAssignmentTree { cols }
            | LinearLowerAdvisedReplay::ParallelFixedColumnAssignmentTree { cols, .. } => {
                linear_lower_fixed_col_tree_replay_advice(&request.base, cols)
            }
            LinearLowerAdvisedReplay::AdaptiveTree { .. } => linear_lower_tree_replay_advice(
                &request.base,
                &request.objective,
                LinearLowerTreeReplaySelection::AdaptiveFsb,
            ),
            LinearLowerAdvisedReplay::AdaptiveCombTree { .. } => linear_lower_tree_replay_advice(
                &request.base,
                &request.objective,
                LinearLowerTreeReplaySelection::AdaptiveCombFsb,
            ),
            LinearLowerAdvisedReplay::AdaptiveFiveCombTree { .. } => {
                linear_lower_tree_replay_advice(
                    &request.base,
                    &request.objective,
                    LinearLowerTreeReplaySelection::AdaptiveFiveCombFsb,
                )
            }
        };
        match mode {
            LinearLowerAdvisedReplay::Split { .. } => {
                print_split_replay_advice(path, &advice);
            }
            LinearLowerAdvisedReplay::Tree { selection } => {
                print_tree_replay_advice(path, selection, &advice);
            }
            LinearLowerAdvisedReplay::FixedAssignmentTree { ranks }
            | LinearLowerAdvisedReplay::ParallelFixedAssignmentTree { ranks, .. } => {
                print_fixed_tree_replay_advice(path, ranks, &advice);
            }
            LinearLowerAdvisedReplay::FixedColumnAssignmentTree { cols }
            | LinearLowerAdvisedReplay::ParallelFixedColumnAssignmentTree { cols, .. } => {
                print_fixed_col_tree_replay_advice(path, cols, &advice);
            }
            LinearLowerAdvisedReplay::AdaptiveTree { .. } => {
                print_tree_replay_advice(
                    path,
                    LinearLowerTreeReplaySelection::AdaptiveFsb,
                    &advice,
                );
            }
            LinearLowerAdvisedReplay::AdaptiveCombTree { .. } => {
                print_tree_replay_advice(
                    path,
                    LinearLowerTreeReplaySelection::AdaptiveCombFsb,
                    &advice,
                );
            }
            LinearLowerAdvisedReplay::AdaptiveFiveCombTree { .. } => {
                print_tree_replay_advice(
                    path,
                    LinearLowerTreeReplaySelection::AdaptiveFiveCombFsb,
                    &advice,
                );
            }
        }
        let selection_error = match mode {
            LinearLowerAdvisedReplay::Split { .. } if advice.selected.len() != 1 => {
                Some("requested split rank is outside the ranked top-eight candidates".to_owned())
            }
            LinearLowerAdvisedReplay::Tree {
                selection: LinearLowerTreeReplaySelection::TargetFsb,
            } if !(LINEAR_LOWER_TARGET_FSB_MIN_CANDIDATES
                ..=LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES)
                .contains(&advice.selected.len()) =>
            {
                Some(
                    "target-FSB tree replay requires three to eight ranked binary candidates"
                        .to_owned(),
                )
            }
            LinearLowerAdvisedReplay::Tree {
                selection:
                    LinearLowerTreeReplaySelection::Intercept
                    | LinearLowerTreeReplaySelection::FullBabsr,
            } if advice.selected.len() != LINEAR_LOWER_TREE_REPLAY_DEPTH => {
                Some("tree replay requires exactly two ranked binary candidates".to_owned())
            }
            LinearLowerAdvisedReplay::FixedAssignmentTree { ranks }
            | LinearLowerAdvisedReplay::ParallelFixedAssignmentTree { ranks, .. }
                if advice.selected.len() != ranks.as_slice().len() =>
            {
                Some(format!(
                    "fixed assignment-tree rank list [{}] is outside the {} post-dedup ranked \
                     candidates",
                    ranks
                        .as_slice()
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    advice.ranked_binary_count,
                ))
            }
            LinearLowerAdvisedReplay::ParallelFixedColumnAssignmentTree { .. }
                if advice.selected.len() != LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH =>
            {
                Some(
                    "parallel selector-tree replay requires exactly four fixed column IDs"
                        .to_owned(),
                )
            }
            LinearLowerAdvisedReplay::AdaptiveTree { .. }
                if !(2..=LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES)
                    .contains(&advice.selected.len()) =>
            {
                Some(
                    "adaptive-FSB tree replay requires two to eight ranked binary candidates"
                        .to_owned(),
                )
            }
            LinearLowerAdvisedReplay::AdaptiveTree { root_rank, .. }
                if root_rank >= advice.selected.len() =>
            {
                Some(format!(
                    "adaptive-FSB root rank {root_rank} is outside {} post-dedup candidates",
                    advice.selected.len()
                ))
            }
            LinearLowerAdvisedReplay::AdaptiveCombTree { .. }
                if !(LINEAR_LOWER_TARGET_FSB_MIN_CANDIDATES
                    ..=LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES)
                    .contains(&advice.selected.len()) =>
            {
                Some(
                    "adaptive-comb-FSB tree replay requires three to eight ranked binary \
                     candidates"
                        .to_owned(),
                )
            }
            LinearLowerAdvisedReplay::AdaptiveCombTree { root_rank, .. }
                if root_rank >= advice.selected.len() =>
            {
                Some(format!(
                    "adaptive-comb-FSB root rank {root_rank} is outside {} post-dedup candidates",
                    advice.selected.len()
                ))
            }
            LinearLowerAdvisedReplay::AdaptiveFiveCombTree { .. }
                if !(LINEAR_LOWER_ADAPTIVE_FIVE_COMB_MIN_CANDIDATES
                    ..=LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES)
                    .contains(&advice.selected.len()) =>
            {
                Some(
                    "adaptive-five-comb-FSB tree replay requires four to eight ranked binary \
                     candidates"
                        .to_owned(),
                )
            }
            LinearLowerAdvisedReplay::AdaptiveFiveCombTree { root_rank, .. }
                if root_rank >= advice.selected.len() =>
            {
                Some(format!(
                    "adaptive-five-comb-FSB root rank {root_rank} is outside {} post-dedup \
                     candidates",
                    advice.selected.len()
                ))
            }
            _ => None,
        };
        if let Some(detail) = selection_error {
            failures += 1;
            print_linear_lower_result(
                path,
                Some(request.requested_lower),
                "error",
                "-",
                None,
                None,
                0.0,
                Some(&detail),
            );
            continue;
        }
        if let LinearLowerAdvisedReplay::AdaptiveTree {
            root_rank,
            hard_value,
        } = mode
        {
            let root = advice.selected[root_rank];
            println!(
                "mip-diff: adaptive-FSB resolved root {}: root_rank={} root=c{} hard_value={}",
                short_name(path),
                root_rank,
                root.0,
                u8::from(hard_value),
            );
        }
        if let LinearLowerAdvisedReplay::AdaptiveCombTree {
            root_rank,
            root_hard_value,
        } = mode
        {
            let root = advice.selected[root_rank];
            println!(
                "mip-diff: adaptive-comb-FSB resolved root {}: root_rank={} root=c{} \
                 root_hard_value={}",
                short_name(path),
                root_rank,
                root.0,
                u8::from(root_hard_value),
            );
        }
        if let LinearLowerAdvisedReplay::AdaptiveFiveCombTree {
            root_rank,
            root_hard_value,
        } = mode
        {
            let root = advice.selected[root_rank];
            println!(
                "mip-diff: adaptive-five-comb-FSB resolved root {}: root_rank={} root=c{} \
                 root_hard_value={}",
                short_name(path),
                root_rank,
                root.0,
                u8::from(root_hard_value),
            );
        }
        let started = Instant::now();
        let max_tree_leaves = match mode {
            LinearLowerAdvisedReplay::FixedColumnAssignmentTree { cols } => {
                1usize << cols.as_slice().len()
            }
            LinearLowerAdvisedReplay::ParallelFixedAssignmentTree { .. }
            | LinearLowerAdvisedReplay::ParallelFixedColumnAssignmentTree { .. } => {
                1usize << LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH
            }
            _ => CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
        };
        let config = CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: timeout_secs,
            max_tree_leaves,
        };
        let result = match mode {
            LinearLowerAdvisedReplay::ParallelFixedAssignmentTree { max_workers, .. }
            | LinearLowerAdvisedReplay::ParallelFixedColumnAssignmentTree { max_workers, .. } => {
                certify_linear_lower_bound_at_with_ay_parallel_selector_tree_unwired(
                    &request.base,
                    &request.objective,
                    request.requested_lower,
                    &advice.selected,
                    max_workers,
                    config,
                )
            }
            LinearLowerAdvisedReplay::FixedAssignmentTree { .. }
            | LinearLowerAdvisedReplay::FixedColumnAssignmentTree { .. } => {
                certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_unwired(
                    &request.base,
                    &request.objective,
                    request.requested_lower,
                    &advice.selected,
                    config,
                )
            }
            LinearLowerAdvisedReplay::AdaptiveFiveCombTree {
                root_rank,
                root_hard_value,
            } => {
                certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_unwired(
                    &request.base,
                    &request.objective,
                    request.requested_lower,
                    &advice.selected,
                    root_rank,
                    root_hard_value,
                    config,
                )
            }
            LinearLowerAdvisedReplay::AdaptiveCombTree {
                root_rank,
                root_hard_value,
            } => {
                certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
                    &request.base,
                    &request.objective,
                    request.requested_lower,
                    &advice.selected,
                    root_rank,
                    root_hard_value,
                    config,
                )
            }
            LinearLowerAdvisedReplay::AdaptiveTree {
                root_rank,
                hard_value,
            } => {
                certify_linear_lower_bound_at_with_ay_adaptive_three_leaf_target_fsb_unwired(
                    &request.base,
                    &request.objective,
                    request.requested_lower,
                    &advice.selected,
                    root_rank,
                    hard_value,
                    config,
                )
            }
            LinearLowerAdvisedReplay::Tree {
                selection: LinearLowerTreeReplaySelection::TargetFsb,
            } => {
                certify_linear_lower_bound_at_with_ay_branch_advice_with_target_fsb_probe_limits_unwired(
                    &request.base,
                    &request.objective,
                    request.requested_lower,
                    &advice.selected,
                    config,
                    target_fsb_probe_limits,
                )
            }
            _ => certify_linear_lower_bound_at_with_ay_branch_advice(
                &request.base,
                &request.objective,
                request.requested_lower,
                &advice.selected,
                config,
            ),
        };
        let elapsed_secs = started.elapsed().as_secs_f64();
        match result {
            Ok(Some(proof)) => {
                if proof.lower.to_bits() != request.requested_lower.to_bits() {
                    failures += 1;
                    print_linear_lower_result(
                        path,
                        Some(request.requested_lower),
                        "error",
                        "-",
                        None,
                        None,
                        elapsed_secs,
                        Some("certified result returned a different threshold"),
                    );
                    continue;
                }
                certified += 1;
                print_linear_lower_result(
                    path,
                    Some(request.requested_lower),
                    "certified",
                    proof_route_name(proof.proof_route),
                    Some(proof.ay_tree_leaves),
                    Some(proof.ny_cert_farkas_replays),
                    elapsed_secs,
                    None,
                );
            }
            Ok(None) => {
                inconclusive += 1;
                print_linear_lower_result(
                    path,
                    Some(request.requested_lower),
                    "inconclusive",
                    "-",
                    None,
                    None,
                    elapsed_secs,
                    None,
                );
            }
            Err(error) => {
                failures += 1;
                let detail = error.to_string();
                print_linear_lower_result(
                    path,
                    Some(request.requested_lower),
                    "error",
                    "-",
                    None,
                    None,
                    elapsed_secs,
                    Some(&detail),
                );
            }
        }
    }
    println!(
        "\n{} instance(s): {certified} certified, {inconclusive} inconclusive, \
         {failures} failure(s)",
        files.len()
    );
    std::process::exit(if failures > 0 { 1 } else { 0 })
}

/// LG3 certify mode: every ay UNSAT must carry verified evidence.
fn run_certify(files: &[PathBuf], timeout_secs: f64) -> ! {
    let mut sat = 0usize;
    let mut unsat_certified = 0usize;
    let mut unsat_bare = 0usize;
    let mut inconclusive = 0usize;
    let mut failures = 0usize;
    println!("{:<40} {:>10}", "instance", "verdict");
    for path in files {
        let (verdict, detail) = match std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| dump::from_milp_text(&text).map_err(|e| e.to_string()))
        {
            Ok(problem) => {
                let (result, _) = solve(&problem, MipBackend::Ay, timeout_secs);
                match result {
                    MipResult::Sat { .. } => {
                        sat += 1;
                        ("sat", None)
                    }
                    MipResult::Unsat { certified: true } => {
                        unsat_certified += 1;
                        ("unsat+cert", None)
                    }
                    MipResult::Unsat { certified: false } => {
                        unsat_bare += 1;
                        ("unsat", None)
                    }
                    MipResult::Timeout => {
                        inconclusive += 1;
                        ("timeout", None)
                    }
                    MipResult::Error(e) => {
                        failures += 1;
                        ("error", Some(e))
                    }
                }
            }
            Err(e) => {
                failures += 1;
                ("load-error", Some(e))
            }
        };
        match detail {
            Some(detail) => println!("{:<40} {verdict:>10}  {detail}", short_name(path)),
            None => println!("{:<40} {verdict:>10}", short_name(path)),
        }
    }
    println!(
        "\n{} instance(s): {sat} sat, {unsat_certified} unsat certified, \
         {unsat_bare} unsat bare, {inconclusive} inconclusive, {failures} failures",
        files.len()
    );
    // A certificate that fails verification arrives from the seam as
    // `MipResult::Error` (ay_lib::map_outcome), so it lands in `failures`
    // with the other hard errors; only certificate ABSENCE degrades to
    // `unsat bare`.
    std::process::exit(if failures > 0 { 1 } else { 0 })
}

fn run_one(
    path: &Path,
    timeout_secs: f64,
    (left, right): (MipBackend, MipBackend),
) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let problem = dump::from_milp_text(&text).map_err(|e| e.to_string())?;
    let (cols, rows) = (problem.num_cols(), problem.num_rows());

    let (lres, t_left) = solve(&problem, left, timeout_secs);
    let (rres, t_right) = solve(&problem, right, timeout_secs);

    let disagreed = matches!(
        (&lres, &rres),
        (MipResult::Sat { .. }, MipResult::Unsat { .. })
            | (MipResult::Unsat { .. }, MipResult::Sat { .. })
    );
    println!(
        "{:<40} {cols:>6} {rows:>6}  {:>9} {t_left:>9.3}  {:>9} {t_right:>9.3}{}",
        short_name(path),
        verdict(&lres),
        verdict(&rres),
        if disagreed { "  <-- DISAGREEMENT" } else { "" }
    );
    Ok(disagreed)
}

fn solve(problem: &MilpProblem, backend: MipBackend, timeout_secs: f64) -> (MipResult, f64) {
    let parts = ny_mip::MipParts {
        problem: problem.clone(),
        input_vars: vec![],
        output_vars: vec![],
        binary_vars: vec![],
        binary_widths: vec![],
        num_cols: problem.num_cols(),
    };
    let config = MipConfig {
        backend,
        parallel_split: 1, // serial: compare raw solver strength, not racing
        timeout_secs,
        ..MipConfig::default()
    };
    let start = Instant::now();
    let result = MipSolver::new(parts, config)
        .check_feasibility()
        .unwrap_or_else(|e| MipResult::Error(e.to_string()));
    (result, start.elapsed().as_secs_f64())
}

fn verdict(r: &MipResult) -> &'static str {
    match r {
        MipResult::Sat { .. } => "sat",
        MipResult::Unsat { certified: true } => "unsat+cert",
        MipResult::Unsat { certified: false } => "unsat",
        MipResult::Timeout => "timeout",
        MipResult::Error(_) => "error",
    }
}

fn collect_files(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let Ok(entries) = std::fs::read_dir(input) else {
                continue;
            };
            let mut batch: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "milp"))
                .collect();
            batch.sort();
            files.extend(batch);
        } else {
            files.push(input.clone());
        }
    }
    files
}

fn short_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both authority tests acquire ny-mip's process-wide bounded worker lease.
    // Running them concurrently can make one correctly decline admission, which
    // is production behavior but not the contract either isolated unit test is
    // exercising.
    static LINEAR_LOWER_AUTHORITY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn one_col_problem() -> (MilpProblem, Col) {
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.25, -2.0, 3.0);
        (problem, x)
    }

    fn extraction_error(problem: &MilpProblem) -> String {
        extract_linear_lower_request(problem)
            .err()
            .expect("model must be rejected")
    }

    fn add_canonical_relu(problem: &mut MilpProblem, lower: f64, upper: f64) -> (Col, Col) {
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

    #[test]
    fn advised_replays_keep_controls_fixed_and_target_fsb_pool_dynamic() {
        let mut problem = MilpProblem::new();
        let mut objective = Vec::new();
        let mut binaries = Vec::new();
        for upper in 1..=9 {
            let (output, binary) = add_canonical_relu(&mut problem, -1.0, f64::from(upper));
            objective.push((output, -1.0));
            binaries.push(binary);
        }
        // These are not exact, unfixed [0,1] integer columns and must never
        // enter the ranking candidate list.
        problem.add_col(0.0, 0.0, 1.0);
        problem.add_integer_col(0.0, 0.0, 2.0);
        problem.add_integer_col(0.0, 1.0, 1.0);
        problem.add_integer_col(0.0, -0.0, 1.0);

        let advice = linear_lower_split_replay_advice(&problem, &objective, 0);
        let expected: Vec<Col> = binaries.iter().rev().copied().take(8).collect();
        assert_eq!(advice.valid_binary_count, 9);
        assert_eq!(advice.ranked_binary_count, 9);
        assert_eq!(advice.selected, expected[..1]);

        assert_eq!(
            format_advice_ids(&advice.selected),
            format!("[c{}]", expected[0].0)
        );
        assert_eq!(format_advice_ids(&[]), "[]");

        let alternate = linear_lower_split_replay_advice(&problem, &objective, 3);
        assert_eq!(alternate.selected, [expected[3]]);
        assert!(
            linear_lower_split_replay_advice(&problem, &objective, 8)
                .selected
                .is_empty(),
            "a rank outside the top-eight cap must fail closed"
        );

        let tree = linear_lower_tree_replay_advice(
            &problem,
            &objective,
            LinearLowerTreeReplaySelection::Intercept,
        );
        assert_eq!(tree.valid_binary_count, 9);
        assert_eq!(tree.ranked_binary_count, 9);
        assert_eq!(tree.selected, expected[..2]);

        let full_ranked = rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
            &problem,
            &objective,
            &binaries,
            LINEAR_LOWER_SPLIT_REPLAY_ADVICE_CAP,
        );
        assert!(full_ranked.len() >= LINEAR_LOWER_TREE_REPLAY_DEPTH);
        let full_tree = linear_lower_tree_replay_advice(
            &problem,
            &objective,
            LinearLowerTreeReplaySelection::FullBabsr,
        );
        assert_eq!(
            full_tree.selected,
            full_ranked[..LINEAR_LOWER_TREE_REPLAY_DEPTH]
        );

        let target_fsb = linear_lower_tree_replay_advice(
            &problem,
            &objective,
            LinearLowerTreeReplaySelection::TargetFsb,
        );
        let target_ranked = rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
            &problem,
            &objective,
            &binaries,
            LINEAR_LOWER_TARGET_FSB_CANDIDATES_PER_SCORE,
        );
        assert_eq!(target_fsb.ranked_binary_count, target_ranked.len());
        assert_eq!(target_fsb.selected, target_ranked);
        assert!(
            (LINEAR_LOWER_TARGET_FSB_MIN_CANDIDATES..=LINEAR_LOWER_TARGET_FSB_MAX_CANDIDATES)
                .contains(&target_fsb.selected.len()),
            "target-FSB must retain a dynamic candidate pool"
        );
        assert_eq!(
            LinearLowerTreeReplaySelection::TargetFsb.name(),
            "target-fsb-full4-intercept4"
        );

        let adaptive_fsb = linear_lower_tree_replay_advice(
            &problem,
            &objective,
            LinearLowerTreeReplaySelection::AdaptiveFsb,
        );
        assert_eq!(adaptive_fsb.ranked_binary_count, target_ranked.len());
        assert_eq!(adaptive_fsb.selected, target_ranked);
        assert_eq!(
            LinearLowerTreeReplaySelection::AdaptiveFsb.name(),
            "adaptive-fsb-full4-intercept4"
        );

        let adaptive_comb_fsb = linear_lower_tree_replay_advice(
            &problem,
            &objective,
            LinearLowerTreeReplaySelection::AdaptiveCombFsb,
        );
        assert_eq!(adaptive_comb_fsb.ranked_binary_count, target_ranked.len());
        assert_eq!(adaptive_comb_fsb.selected, target_ranked);
        assert_eq!(
            LinearLowerTreeReplaySelection::AdaptiveCombFsb.name(),
            "adaptive-comb-fsb-full4-intercept4"
        );

        let adaptive_five_comb_fsb = linear_lower_tree_replay_advice(
            &problem,
            &objective,
            LinearLowerTreeReplaySelection::AdaptiveFiveCombFsb,
        );
        assert_eq!(
            adaptive_five_comb_fsb.ranked_binary_count,
            target_ranked.len()
        );
        assert_eq!(adaptive_five_comb_fsb.selected, target_ranked);
        assert_eq!(
            LinearLowerTreeReplaySelection::AdaptiveFiveCombFsb.name(),
            "adaptive-five-comb-fsb-full4-intercept4"
        );
    }

    #[test]
    fn reconstructs_base_objective_and_signed_zero_threshold_bit_exactly() {
        let mut base = MilpProblem::new();
        let x = base.add_col(-0.0, -2.5, 4.0);
        let z = base.add_integer_col(1.25, 0.0, 1.0);
        base.add_row(-1.0, 2.0, [(z, -3.5), (x, 0.125)]);
        let base_text = dump::to_milp_text(&base);

        let mut decision = base.clone();
        decision.add_row(
            f64::NEG_INFINITY,
            f64::from(-0.0_f32),
            [(x, -2.0), (z, 0.75)],
        );
        let parsed =
            dump::from_milp_text(&dump::to_milp_text(&decision)).expect("decision roundtrip");
        let request = extract_linear_lower_request(&parsed).expect("valid captured request");

        assert_eq!(dump::to_milp_text(&request.base), base_text);
        assert_eq!(request.requested_lower.to_bits(), (-0.0_f32).to_bits());
        assert_eq!(request.objective.len(), 2);
        assert_eq!(request.objective[0].0, x);
        assert_eq!(request.objective[0].1.to_bits(), (-2.0_f64).to_bits());
        assert_eq!(request.objective[1].0, z);
        assert_eq!(request.objective[1].1.to_bits(), 0.75_f64.to_bits());
    }

    #[test]
    fn rejects_missing_or_marked_decision_row() {
        let (empty_rows, _) = one_col_problem();
        assert!(extraction_error(&empty_rows).contains("no final decision row"));

        let (mut marked, x) = one_col_problem();
        marked
            .add_margin_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)])
            .expect("valid marker");
        assert!(extraction_error(&marked).contains("unexpected margin marker"));
    }

    #[test]
    fn rejects_noncanonical_decision_bounds() {
        let (mut finite_lower, x) = one_col_problem();
        finite_lower.add_row(0.0, 1.0, [(x, 1.0)]);
        assert!(extraction_error(&finite_lower).contains("not exactly -infinity"));

        let (mut infinite_upper, x) = one_col_problem();
        infinite_upper.add_row(f64::NEG_INFINITY, f64::INFINITY, [(x, 1.0)]);
        assert!(extraction_error(&infinite_upper).contains("upper bound is not finite"));

        let (mut non_f32_upper, x) = one_col_problem();
        non_f32_upper.add_row(f64::NEG_INFINITY, 0.1_f64, [(x, 1.0)]);
        assert!(extraction_error(&non_f32_upper).contains("exact widening of a finite f32"));
    }

    #[test]
    fn rejects_noncanonical_decision_objective() {
        let (mut empty, _) = one_col_problem();
        empty.add_row(f64::NEG_INFINITY, 0.0, []);
        assert!(extraction_error(&empty).contains("empty objective"));

        let (mut zero, x) = one_col_problem();
        zero.add_row(f64::NEG_INFINITY, 0.0, [(x, -0.0)]);
        assert!(extraction_error(&zero).contains("zero coefficient"));

        let (mut non_finite, x) = one_col_problem();
        non_finite.add_row(f64::NEG_INFINITY, 0.0, [(x, f64::NAN)]);
        assert!(extraction_error(&non_finite).contains("non-finite coefficient"));

        let mut unsorted = MilpProblem::new();
        let x = unsorted.add_col(0.0, 0.0, 1.0);
        let y = unsorted.add_col(0.0, 0.0, 1.0);
        unsorted.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (x, 1.0)]);
        assert!(extraction_error(&unsorted).contains("strictly increasing canonical order"));

        let (mut duplicate, x) = one_col_problem();
        duplicate.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0), (x, -1.0)]);
        assert!(extraction_error(&duplicate).contains("strictly increasing canonical order"));
    }

    #[test]
    fn parallel_selector_worker_cli_is_bounded_and_requires_four_fixed_selectors() {
        for workers in [1, 4, 8, 16] {
            assert_eq!(
                parse_parallel_selector_workers(&workers.to_string()),
                Ok(workers)
            );
        }
        for invalid in ["0", "17", "many", "1,4"] {
            assert!(
                parse_parallel_selector_workers(invalid).is_err(),
                "{invalid} must be rejected"
            );
        }

        let four_cols = LinearLowerFixedTreeCols::parse("9,8,7,6").unwrap();
        let three_cols = LinearLowerFixedTreeCols::parse("9,8,7").unwrap();
        let four_ranks = LinearLowerFixedTreeRanks::parse("0,1,2,3").unwrap();
        assert!(
            validate_parallel_selector_cli(false, None, None, None).is_ok(),
            "the default-off path must impose no new CLI constraint"
        );
        assert!(validate_parallel_selector_cli(false, None, Some(four_cols), Some(4)).is_err());
        assert!(validate_parallel_selector_cli(true, None, None, Some(4)).is_err());
        assert!(validate_parallel_selector_cli(true, None, Some(three_cols), Some(4)).is_err());
        assert!(
            validate_parallel_selector_cli(true, Some(four_ranks), Some(four_cols), Some(4))
                .is_err()
        );
        assert!(validate_parallel_selector_cli(true, None, Some(four_cols), Some(16)).is_ok());
        assert!(validate_parallel_selector_cli(true, Some(four_ranks), None, Some(1)).is_ok());
    }

    #[test]
    fn parallel_selector_flag_routes_only_fixed_trees_to_parallel_authority() {
        let cols = LinearLowerFixedTreeCols::parse("9,8,7,6").unwrap();
        let ranks = LinearLowerFixedTreeRanks::parse("0,1,2,3").unwrap();

        assert_eq!(
            fixed_tree_replay_mode(None, Some(cols), None),
            Some(LinearLowerAdvisedReplay::FixedColumnAssignmentTree { cols })
        );
        assert_eq!(
            fixed_tree_replay_mode(None, Some(cols), Some(8)),
            Some(
                LinearLowerAdvisedReplay::ParallelFixedColumnAssignmentTree {
                    cols,
                    max_workers: 8,
                }
            )
        );
        assert_eq!(
            fixed_tree_replay_mode(Some(ranks), None, Some(4)),
            Some(LinearLowerAdvisedReplay::ParallelFixedAssignmentTree {
                ranks,
                max_workers: 4,
            })
        );
        assert_eq!(fixed_tree_replay_mode(None, None, Some(8)), None);
    }

    #[test]
    fn replay_mode_calls_production_linear_lower_authority() {
        let _authority_guard = LINEAR_LOWER_AUTHORITY_TEST_LOCK
            .lock()
            .expect("linear-lower authority test lock");
        let mut base = MilpProblem::new();
        let x = base.add_col(0.0, 1.0, 2.0);
        let requested_lower = 0.99_f32;
        let mut decision = base;
        decision.add_row(f64::NEG_INFINITY, f64::from(requested_lower), [(x, 1.0)]);
        let request = extract_linear_lower_request(&decision).expect("valid request");
        let proof = certify_linear_lower_bound_at_with_ay(
            &request.base,
            &request.objective,
            request.requested_lower,
            CertifiedLinearLowerDecisionConfig {
                proof_timeout_secs: 5.0,
                max_tree_leaves: CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
            },
        )
        .expect("authority call")
        .expect("relaxation proves the threshold");

        assert_eq!(proof.lower.to_bits(), requested_lower.to_bits());
        assert_eq!(
            proof.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(proof.ay_tree_leaves, 0);
        assert_eq!(proof.ny_cert_farkas_replays, 1);
    }

    #[test]
    fn extracted_dump_request_routes_to_verified_parallel_selector_tree() {
        let _authority_guard = LINEAR_LOWER_AUTHORITY_TEST_LOCK
            .lock()
            .expect("linear-lower authority test lock");
        let mut base = MilpProblem::new();
        let x = base.add_col(0.0, 1.0, 2.0);
        let selectors = (0..LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH)
            .map(|_| base.add_integer_col(0.0, 0.0, 1.0))
            .collect::<Vec<_>>();
        let requested_lower = 0.99_f32;
        let mut decision = base;
        decision.add_row(f64::NEG_INFINITY, f64::from(requested_lower), [(x, 1.0)]);
        let parsed =
            dump::from_milp_text(&dump::to_milp_text(&decision)).expect("decision roundtrip");
        let request = extract_linear_lower_request(&parsed).expect("valid dumped request");
        let proof = certify_linear_lower_bound_at_with_ay_parallel_selector_tree_unwired(
            &request.base,
            &request.objective,
            request.requested_lower,
            &selectors,
            4,
            CertifiedLinearLowerDecisionConfig {
                proof_timeout_secs: 5.0,
                max_tree_leaves: 1 << LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH,
            },
        )
        .expect("parallel authority call")
        .expect("all sixteen fixed selector leaves prove the threshold");

        assert_eq!(proof.lower.to_bits(), requested_lower.to_bits());
        assert_eq!(
            proof.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(
            proof.ay_tree_leaves,
            1 << LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH
        );
        assert_eq!(
            proof.ny_cert_farkas_replays,
            1 << LINEAR_LOWER_PARALLEL_SELECTOR_DEPTH
        );
    }
}
