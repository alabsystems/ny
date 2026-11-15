// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use ny_propagate::beta_crown::NonlinearBranchingConfig;
use ny_propagate::BranchingHeuristic;

use super::PresetConfig;
use crate::commands::beta_crown::branching::{parse_branching_heuristic, RELU_TOKEN};

/// #genbab-candidates: build the GenBaB config from the preset instead of
/// taking `NonlinearBranchingConfig::default()` wholesale.
///
/// WHY THIS EXISTS. `NonlinearBranchingConfig::num_candidates` defaults to **1**
/// and `selector.rs` ends `candidates.truncate(self.config.num_candidates)`
/// after sorting by score — so GenBaB branching considers exactly ONE candidate
/// per split. No preset key could raise it: both GenBaB construction sites took
/// `::default()`.
///
/// Meanwhile `configs/vnncomp25/ml4acopf_2024.yaml` writes `candidates: 100`
/// under `branching:`, which `preset::apply` sends to `config.fsb_candidates` —
/// a field read only by the sequential selector, the GPU kFSB lane and
/// kFSB-multi (`engine/branching/sequential.rs`, `gpu_bab/kfsb.rs`,
/// `batched/kfsb_multi.rs`). ml4acopf runs none of them. So the 100 it asks for
/// has never reached the selector it meant, and the selector it meant has been
/// running at 1.
///
/// ONLY `num_candidates` is delivered here, deliberately. The sibling
/// `filter`/`filter_beta` fields are declared on `NonlinearBranchingConfig` and
/// read by NOTHING outside their own unit tests (checked across the crate), so
/// wiring them would be a fake fix that silences a contract entry without
/// changing behaviour. `preset/contract.rs` continues to report them unhonoured,
/// which is correct.
///
/// `configs/vnncomp25/ml4acopf_2024.yaml` sets `nonlinear_split.num_candidates:
/// 100` (#genbab-selector-convention); every other preset keeps the `Default`
/// of 1. VALUE-NEUTRAL today: both GenBaB consumers branch on `decisions[0]`
/// and the sort is stable, so the chosen split does not depend on the budget
/// until a consumer (the BBPS filter) reads `decisions[1..]`.
fn genbab_config(preset: &PresetConfig) -> NonlinearBranchingConfig {
    let mut config = NonlinearBranchingConfig::default();
    if let Some(candidates) = preset.bab.branching.nonlinear_split.num_candidates {
        config.num_candidates = candidates;
    }
    config
}

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
                heuristic: BranchingHeuristic::GenBaB(genbab_config(preset)),
                use_relu_split: true,
            }));
        }
        return Ok(input_split_enabled.then_some(ResolvedBranching {
            heuristic: BranchingHeuristic::InputSplit,
            use_relu_split: false,
        }));
    };
    parse_resolved_branching(method, input_split_enabled, genbab_config(preset)).map(Some)
}

fn parse_resolved_branching(
    method: &str,
    input_split_enabled: bool,
    genbab: NonlinearBranchingConfig,
) -> Result<ResolvedBranching> {
    let normalized = method.to_ascii_lowercase();
    if normalized == RELU_TOKEN {
        return Ok(ResolvedBranching {
            heuristic: BranchingHeuristic::LargestBoundWidth,
            use_relu_split: true,
        });
    }
    // `nonlinear` (alpha-beta-CROWN's GenBaB method token; `genbab` accepted as
    // the ny-native alias) selects GenBaB branching explicitly. Same routing as
    // the method-absent `nonlinear_split` arm in `resolve_branching`: GenBaB
    // runs in the GRAPH engine, so it uses the ReLU-split route (#ml4acopf-genbab).
    // An explicit `method` always wins over an implicit `nonlinear_split`
    // section, so presets that pin `method: input` alongside nonlinear-split
    // tuning keep input splitting — selecting GenBaB requires naming it here.
    if normalized == "nonlinear" || normalized == "genbab" {
        return Ok(ResolvedBranching {
            heuristic: BranchingHeuristic::GenBaB(genbab),
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
    // ReLU-split path, not input splitting.
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
