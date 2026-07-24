// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use ny_propagate::beta_crown::NonlinearBranchingConfig;
use ny_propagate::BranchingHeuristic;

use super::PresetConfig;
use crate::commands::beta_crown::branching::{parse_branching_heuristic, RELU_TOKEN};

/// Branching selection resolved from a preset file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedBranching {
    pub(crate) heuristic: BranchingHeuristic,
    pub(crate) use_relu_split: bool,
}

/// Resolve the preset's requested branching mode for CLI routing decisions.
///
/// CLI flags still take precedence over this result; callers should only use it
/// when the user did not pass an explicit `--branching`.
pub(crate) fn resolve_branching(preset: &PresetConfig) -> Result<Option<ResolvedBranching>> {
    let input_split_enabled = preset.bab.branching.input_split.enable.unwrap_or(false);
    let Some(method) = preset.bab.branching.method.as_deref() else {
        // No explicit `method`. A configured `nonlinear_split` section selects the
        // GenBaB branching path (general nonlinearities: bounded Mul/MatMul,
        // Sigmoid, Sin/Cos, …). GenBaB runs in the GRAPH engine, so it is routed
        // like a ReLU split (`use_relu_split: true`) — input splitting cannot touch
        // the nonlinear product frontier these nets are bottlenecked on.
        // SOUND: GenBaB is search-only; every split is a complete case partition and
        // the per-child McCormick relaxation stays a valid outer bound, so this never
        // changes a verdict.
        if preset.bab.branching.nonlinear_split.requests_genbab() {
            return Ok(Some(ResolvedBranching {
                heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig::default()),
                use_relu_split: true,
            }));
        }
        return Ok(input_split_enabled.then_some(ResolvedBranching {
            heuristic: BranchingHeuristic::InputSplit,
            use_relu_split: false,
        }));
    };
    parse_resolved_branching(method, input_split_enabled).map(Some)
}

fn parse_resolved_branching(method: &str, input_split_enabled: bool) -> Result<ResolvedBranching> {
    let normalized = method.to_ascii_lowercase();
    if normalized == RELU_TOKEN {
        return Ok(ResolvedBranching {
            heuristic: BranchingHeuristic::LargestBoundWidth,
            use_relu_split: true,
        });
    }
    if normalized == "sb" {
        if input_split_enabled {
            return Ok(ResolvedBranching {
                heuristic: BranchingHeuristic::InputSplit,
                use_relu_split: false,
            });
        }
        anyhow::bail!(
            "preset branching method 'sb' requires bab.branching.input_split.enable: true"
        );
    }

    let heuristic = parse_branching_method(method)?;
    // kFSB and kFSB-intercept-only are ReLU-splitting strategies: they select
    // which ReLU neuron to pin active/inactive. Route them through the graph
    // ReLU-split path, not input splitting. (#4300)
    let use_relu_split = matches!(
        heuristic,
        BranchingHeuristic::Kfsb
            | BranchingHeuristic::KfsbInterceptOnly
            | BranchingHeuristic::FilteredSmartBranching
            | BranchingHeuristic::BoundImpact
    );
    Ok(ResolvedBranching {
        heuristic,
        use_relu_split,
    })
}

/// Parse branching method string to enum.
///
/// Returns an error for unknown methods instead of silently defaulting.
pub(super) fn parse_branching_method(method: &str) -> Result<BranchingHeuristic> {
    let normalized = method.to_ascii_lowercase();
    parse_branching_heuristic(&normalized)
        .with_context(|| format!("in preset branching method '{method}'"))
}
