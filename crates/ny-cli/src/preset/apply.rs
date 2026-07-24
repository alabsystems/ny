// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::{bail, Result};
use ny_propagate::{
    AlphaCrownConfig, BetaCrownConfig, InputClipType, KfsbReduceOp, PgdAlphaMode,
    PgdInitialization, PgdOptimizer,
};
use tracing::warn;

use super::{
    resolve_branching, AlphaCrownPreset, BetaCrownPreset, ClipPreset, CutsPreset,
    PhaseBudgetPreset, PresetConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedBoundPropMode {
    pub(crate) use_alpha_crown: bool,
    pub(crate) use_forward_bounds: bool,
}

/// Apply preset configuration to a BetaCrownConfig.
///
/// CLI flags take precedence over preset values (apply preset first, then CLI).
/// Supports both alpha-beta-CROWN structure (solver: + bab:) and ny (bab: only).
///
/// Returns an error if preset contains an unsupported bound propagation method,
/// unknown branching method, or unknown reduce operation string.
pub(crate) fn apply_preset(config: &mut BetaCrownConfig, preset: &PresetConfig) -> Result<()> {
    apply_solver_and_bab_settings(config, preset)?;
    apply_branching_preset(config, preset)?;

    apply_alpha_preset(&mut config.alpha_config, &preset.solver.alpha_crown);
    apply_alpha_preset(&mut config.alpha_config, &preset.bab.alpha_crown);

    apply_beta_preset(config, &preset.solver.beta_crown);
    apply_beta_preset(config, &preset.bab.beta_crown);

    apply_cuts_preset(config, &preset.bab.cuts);
    apply_clip_preset(config, &preset.bab.clip);
    apply_phase_budget_preset(config, &preset.bab.phase_budget);
    apply_attack_preset(config, preset)?;
    // General preset: conv_mode. Applied after cuts so auto-mode resolves correctly.
    if let Some(conv_mode) = preset.general.conv_mode {
        config.conv_mode = conv_mode;
    }
    warn_unimplemented_fields(preset);
    Ok(())
}

/// Emit warnings for preset fields that are parsed but have no engine counterpart.
fn warn_unimplemented_fields(preset: &PresetConfig) {
    if preset.general.loss_reduction_func.is_some() {
        warn!("preset field general.loss_reduction_func is not yet supported — ignored");
    }
    if preset.attack.attack_tolerance.is_some() {
        warn!("preset field attack.attack_tolerance is not yet supported — ignored");
    }
    // pruning_in_iteration: `AlphaCrownConfig` declares the field but no engine
    // code reads it, so copying the value would silently no-op. Running without
    // in-iteration pruning is conservative (more work, never a looser bound).
    if preset.bab.pruning_in_iteration.is_some() {
        warn!("preset field bab.pruning_in_iteration is not yet supported — ignored");
    }
    // share_alphas: accepted as a no-op — ny's graph alpha state ALREADY
    // implements the shared semantics alpha-beta-CROWN opts into with
    // share_alphas=True: GraphAlphaState keys one Array1<f32> of alphas per
    // node (bounds/alpha/graph.rs:33-72) and bilinear alphas are [4,m,n,k]
    // per MatMul node (graph_alpha/propagate_dag/init.rs:248-266) — there is
    // no per-spec-row alpha dimension anywhere to share. Warning removed so
    // winner-parity presets (vit) load clean.
    let nonlinear = &preset.bab.branching.nonlinear_split;
    if nonlinear.filter.is_some() {
        warn!("preset field bab.branching.nonlinear_split.filter is not yet supported — ignored");
    }
    if nonlinear.filter_beta.is_some() {
        warn!(
            "preset field bab.branching.nonlinear_split.filter_beta is not yet supported — ignored"
        );
    }
    if preset.bab.invprop.apply_output_constraints_to.is_some() {
        warn!(
            "preset field bab.invprop.apply_output_constraints_to is not yet supported — ignored"
        );
    }
    if preset.bab.invprop.share_gammas.is_some() {
        warn!("preset field bab.invprop.share_gammas is not yet supported — ignored");
    }
}

pub(crate) fn resolve_use_alpha_from_bound_prop_method(
    method: Option<&str>,
) -> Result<Option<bool>> {
    Ok(resolve_bound_prop_mode(method)?.map(|mode| mode.use_alpha_crown))
}

pub(crate) fn resolve_bound_prop_mode(
    method: Option<&str>,
) -> Result<Option<ResolvedBoundPropMode>> {
    let Some(method) = method.map(str::trim).filter(|method| !method.is_empty()) else {
        return Ok(None);
    };

    match method.to_ascii_lowercase().as_str() {
        "crown" => Ok(Some(ResolvedBoundPropMode {
            use_alpha_crown: false,
            use_forward_bounds: false,
        })),
        "alpha-crown" | "alpha_crown" => Ok(Some(ResolvedBoundPropMode {
            use_alpha_crown: true,
            use_forward_bounds: false,
        })),
        "forward+backward" | "forward+crown" => Ok(Some(ResolvedBoundPropMode {
            use_alpha_crown: false,
            use_forward_bounds: true,
        })),
        _ => bail!(
            "unsupported solver.bound_prop_method '{method}': ny currently supports only 'crown', 'alpha-crown', 'forward+backward', and 'forward+crown' preset modes"
        ),
    }
}

fn apply_solver_and_bab_settings(
    config: &mut BetaCrownConfig,
    preset: &PresetConfig,
) -> Result<()> {
    let bab = &preset.bab;
    let solver = &preset.solver;

    if let Some(mode) = resolve_bound_prop_mode(solver.bound_prop_method.as_deref())? {
        config.use_alpha_crown = mode.use_alpha_crown;
        config.use_forward_bounds = mode.use_forward_bounds;
    }

    if let Some(batch_size) = solver.batch_size {
        config.batch_size = batch_size;
    }
    if let Some(build_batch_size) = solver.build_batch_size {
        config.build_batch_size = Some(build_batch_size);
    }
    if let Some(batch_size) = bab.batch_size {
        config.batch_size = batch_size;
    }
    if let Some(crown_backward_layers) = bab.crown_backward_layers {
        config.crown_backward_layers = Some(crown_backward_layers);
    }
    if let Some(timeout) = bab.timeout {
        config.timeout = Duration::from_secs(timeout);
    }
    if let Some(max_domains) = bab.max_domains {
        config.max_domains = max_domains;
    }
    if let Some(max_depth) = bab.max_depth {
        config.max_depth = max_depth;
    }
    if let Some(interm_transfer) = bab.interm_transfer {
        config.enable_interm_transfer = interm_transfer;
    }
    if let Some(enabled) = bab.root_crown_interm_dense_head {
        config.root_crown_interm_dense_head = enabled;
    }
    if let Some(max_secs) = bab.root_crown_interm_max_secs {
        config.root_crown_interm_max_secs = max_secs;
    }
    if let Some(max_dim) = bab.root_crown_interm_max_dim {
        config.root_crown_interm_max_dim = max_dim;
    }
    if let Some(enabled) = bab.root_sparse_interm_crown {
        config.root_sparse_interm_crown = enabled;
    }
    if let Some(max_secs) = bab.root_sparse_interm_crown_max_secs {
        config.root_sparse_interm_crown_max_secs = max_secs;
    }
    if let Some(max_dim) = bab.root_sparse_interm_crown_max_dim {
        config.root_sparse_interm_crown_max_dim = max_dim;
    }
    if let Some(max_rows) = bab.root_sparse_interm_crown_max_rows {
        config.root_sparse_interm_crown_max_rows = max_rows;
    }
    if let Some(max_targets) = bab.root_sparse_interm_crown_max_targets {
        config.root_sparse_interm_crown_max_targets = max_targets;
    }
    if let Some(beta_graft) = bab.beta_graft {
        config.mo_beta_graft = beta_graft; // #mo-beta-graft
    }
    if let Some(ratio) = solver.min_batch_size_ratio {
        config.min_batch_fill_ratio = ratio;
    }
    if let Some(ratio) = bab.min_batch_size_ratio {
        config.min_batch_fill_ratio = ratio;
    }
    // auto_enlarge_batch_size: solver takes precedence, bab overrides (#4303).
    if let Some(auto_enlarge) = solver.auto_enlarge_batch_size {
        config.auto_enlarge_batch_size = auto_enlarge;
    }
    if let Some(auto_enlarge) = bab.auto_enlarge_batch_size {
        config.auto_enlarge_batch_size = auto_enlarge;
    }
    if let Some(patience) = bab.early_stop_patience {
        config.early_stop_patience = patience;
    }
    // Per-node CROWN-IBP time-budget overrides (#4413, #cgan-bn11-budget).
    // Only explicitly-set preset values override; unset keeps the built-in
    // constants (2.0 s floor / 12.0 s cap) byte-identically.
    if let Some(floor_secs) = bab.crown_ibp_per_node_floor_secs {
        config.crown_ibp_per_node_floor_secs = Some(floor_secs);
    }
    if let Some(cap_secs) = bab.crown_ibp_per_node_cap_secs {
        config.crown_ibp_per_node_cap_secs = Some(cap_secs);
    }

    Ok(())
}

fn apply_branching_preset(config: &mut BetaCrownConfig, preset: &PresetConfig) -> Result<()> {
    let branching = &preset.bab.branching;

    if let Some(branching) = resolve_branching(preset)? {
        config.branching_heuristic = branching.heuristic;
    }
    if let Some(candidates) = branching.candidates {
        config.fsb_candidates = candidates;
    }
    if let Some(ref reduceop) = branching.reduceop {
        config.kfsb_reduce_op = parse_reduce_op(reduceop)?;
    }
    if let Some(kfsb_multi) = branching.kfsb_multi {
        // #kfsb-multi: cifar100-scoped opt-in for the wave-batched selector.
        config.use_kfsb_multi_branching = kfsb_multi;
    }
    if let Some(coeff_thresh) = branching.input_split.sb_coeff_thresh {
        config.input_split_coeff_thresh = coeff_thresh;
    }
    if let Some(touch_zero_score) = branching.input_split.touch_zero_score {
        config.input_split_touch_zero_score = touch_zero_score;
    }
    if let Some(sb_margin_weight) = branching.input_split.sb_margin_weight {
        config.input_split_sb_margin_weight = sb_margin_weight;
    }
    if let Some(sb_sum) = branching.input_split.sb_sum {
        config.input_split_sb_sum = sb_sum;
    }
    if let Some(sb_primary_spec) = branching.input_split.sb_primary_spec {
        config.input_split_sb_primary_spec = Some(sb_primary_spec);
    }
    if let Some(ibp_enhancement) = branching.input_split.ibp_enhancement {
        config.input_split_ibp_enhancement = ibp_enhancement;
    }
    if let Some(stacked_rebound) = branching.input_split.stacked_rebound {
        config.input_split_stacked_rebound = stacked_rebound;
    }
    if let Some(warm_parallel) = branching.input_split.warm_parallel {
        config.input_split_warm_parallel = warm_parallel;
    }
    if let Some(reorder_bab) = branching.input_split.reorder_bab {
        config.reorder_bab = reorder_bab;
    }
    if let Some(adv_check) = branching.input_split.adv_check {
        config.adv_check = adv_check;
    }
    if let Some(depth) = branching.input_split.depth {
        config.input_split_depth = depth;
    }
    if let Some(alpha_iteration) = branching.input_split.alpha_iteration {
        config.input_split_alpha_iteration = alpha_iteration;
    }
    if let Some(lr_alpha) = branching.input_split.lr_alpha {
        config.input_split_lr_alpha = lr_alpha;
    }
    Ok(())
}

fn apply_attack_preset(config: &mut BetaCrownConfig, preset: &PresetConfig) -> Result<()> {
    if let Some(pgd_restarts) = preset.attack.pgd_restarts {
        config.pgd_restarts = pgd_restarts;
    }
    if let Some(pgd_steps) = preset.attack.pgd_steps {
        config.pgd_steps = pgd_steps;
    }
    if let Some(ref order) = preset.attack.pgd_order {
        // Only enablement (plus the "input_bab" upfront-suppression
        // discriminator, resolved by the beta-crown handler) is implemented;
        // reference alpha-beta-CROWN's before/middle/after ordering is not.
        // "before" matches the upfront schedule that actually runs, so only
        // "middle"/"after" warn. Unknown values are rejected like attack_mode.
        match order.to_lowercase().as_str() {
            "skip" | "none" | "disabled" => config.enable_pgd_attack = false,
            "before" | "input_bab" => config.enable_pgd_attack = true,
            sched @ ("middle" | "after") => {
                config.enable_pgd_attack = true;
                warn!(
                    "preset field attack.pgd_order '{sched}' scheduling is not implemented — \
                     PGD runs on the default upfront schedule"
                );
            }
            other => bail!(
                "unknown attack.pgd_order '{other}': supported values are 'before', 'middle', \
                 'after' (PGD enabled, upfront schedule), 'input_bab' (PGD enabled, no upfront \
                 stage), and 'skip'/'none'/'disabled' (PGD disabled)"
            ),
        }
    }
    if let Some(restart_when_stuck) = preset.attack.pgd_restart_when_stuck {
        config.pgd_restart_when_stuck = restart_when_stuck;
    }
    if let Some(ref attack_mode) = preset.attack.attack_mode {
        match attack_mode.to_lowercase().as_str() {
            "pgd" => {
                config.pgd_initialization = PgdInitialization::Uniform;
            }
            "diversed_pgd" => {
                config.pgd_initialization = PgdInitialization::Osi;
            }
            "diversed_gama_pgd" => {
                // OSI initialization + the GAMA guidance loss (#1449).
                // Reference: alpha-beta-CROWN `attack_mode: diversed_GAMA_PGD`
                // → `initialization="osi", GAMA_loss=True`
                // (`attack_interface.py:29-35`). Attack-only: candidates are
                // re-validated before any `sat`, never affects soundness.
                config.pgd_initialization = PgdInitialization::Osi;
                config.pgd_gama = true;
            }
            "boundary" => {
                bail!("attack_mode 'boundary' is not supported in ny");
            }
            other => {
                bail!(
                    "unknown attack_mode '{}': supported modes are 'PGD', \
                     'diversed_PGD', and 'diversed_GAMA_PGD'",
                    other
                );
            }
        }
    }
    if let Some(osi_steps) = preset.attack.osi_steps {
        config.pgd_osi_steps = osi_steps;
    }
    if let Some(pgd_lr_decay) = preset.attack.pgd_lr_decay {
        config.pgd_lr_decay = pgd_lr_decay;
    }
    // STE surrogate for Sign layers during attack gradients (#surrogate-sign)
    // and the dense low-effective-dimension sweep pre-phase (#dense-sweep).
    // Both are attack-only: candidates are re-validated before any `sat`.
    if let Some(surrogate_sign_gradient) = preset.attack.surrogate_sign_gradient {
        config.pgd_surrogate_sign_gradient = surrogate_sign_gradient;
    }
    if let Some(dense_low_dim_sweep) = preset.attack.dense_low_dim_sweep {
        config.pgd_dense_low_dim_sweep = dense_low_dim_sweep;
    }
    if let Some(dense_sweep_max_dims) = preset.attack.dense_sweep_max_dims {
        config.pgd_dense_sweep_max_dims = dense_sweep_max_dims;
    }
    if let Some(dense_sweep_points) = preset.attack.dense_sweep_points {
        config.pgd_dense_sweep_points = dense_sweep_points;
    }
    if preset.attack.pgd_alpha_scale.unwrap_or(false) {
        let alpha = preset
            .attack
            .pgd_alpha
            .as_deref()
            .unwrap_or("0.01")
            .parse::<f32>()
            .map_err(|_| {
                anyhow::anyhow!(
                    "attack.pgd_alpha_scale=true requires a numeric attack.pgd_alpha value"
                )
            })?;
        config.pgd_optimizer = PgdOptimizer::SignedGradient;
        config.pgd_alpha_mode = PgdAlphaMode::InputRangeScaled(alpha);
    } else if let Some(alpha) = preset.attack.pgd_alpha.as_deref() {
        if alpha.eq_ignore_ascii_case("auto") {
            config.pgd_alpha_mode = PgdAlphaMode::Auto;
        } else {
            config.pgd_alpha_mode = PgdAlphaMode::Scalar(alpha.parse::<f32>().map_err(|_| {
                anyhow::anyhow!("attack.pgd_alpha must be a numeric value or 'auto', got '{alpha}'")
            })?);
        }
    }
    Ok(())
}

fn apply_alpha_preset(config: &mut AlphaCrownConfig, preset: &AlphaCrownPreset) {
    if let Some(lr_alpha) = preset.lr_alpha {
        config.learning_rate = lr_alpha;
    }
    if let Some(iterations) = preset.iterations {
        config.iterations = iterations;
    }
    if let Some(lr_decay) = preset.lr_decay {
        config.lr_decay = lr_decay;
    }
    if let Some(start_save_best) = preset.start_save_best {
        config.start_save_best = start_save_best;
    }
    if let Some(full_conv_alpha) = preset.full_conv_alpha {
        config.full_conv_alpha = full_conv_alpha;
    }
}

fn apply_beta_preset(config: &mut BetaCrownConfig, preset: &BetaCrownPreset) {
    if let Some(lr_alpha) = preset.lr_alpha {
        config.alpha_lr = lr_alpha;
    }
    if let Some(lr_beta) = preset.lr_beta {
        config.beta_lr = lr_beta;
    }
    if let Some(iterations) = preset.iterations {
        config.beta_iterations = iterations;
    }
    if let Some(max_depth) = preset.max_depth {
        config.beta_max_depth = max_depth;
    }
    if let Some(optimize_disjuncts_separately) = preset.optimize_disjuncts_separately {
        config.optimize_disjuncts_separately = optimize_disjuncts_separately;
    }
    if let Some(lr_decay) = preset.lr_decay {
        // Beta lr_decay shares AlphaCrownConfig::lr_decay.
        // In alpha-beta-CROWN, solver.beta-crown.lr_decay overrides solver.alpha-crown.lr_decay
        // when both are set. Apply beta last so it takes precedence.
        config.alpha_config.lr_decay = lr_decay;
    }
}

fn apply_cuts_preset(config: &mut BetaCrownConfig, preset: &CutsPreset) {
    if let Some(enabled) = preset.enabled {
        config.enable_cuts = enabled;
    }
    if let Some(max_cuts) = preset.max_cuts {
        config.max_cuts = max_cuts;
    }
    if let Some(min_cut_depth) = preset.min_cut_depth {
        config.min_cut_depth = min_cut_depth;
    }
    if let Some(near_miss) = preset.near_miss {
        config.enable_near_miss_cuts = near_miss;
    }
    if let Some(near_miss_margin) = preset.near_miss_margin {
        config.near_miss_margin = near_miss_margin;
    }
    if let Some(proactive) = preset.proactive {
        config.enable_proactive_cuts = proactive;
    }
    if let Some(max_proactive) = preset.max_proactive {
        config.max_proactive_cuts = max_proactive;
    }
}

pub(crate) fn apply_clip_preset(config: &mut BetaCrownConfig, preset: &ClipPreset) {
    if let Some(relaxed) = preset.relaxed {
        config.enable_relaxed_clip = relaxed;
    }
    if let Some(relaxed_iterations) = preset.relaxed_iterations {
        config.relaxed_clip_iterations = relaxed_iterations;
    }
    if let Some(ref clip_type) = preset.clip_type {
        config.input_clip_type = match clip_type.to_lowercase().as_str() {
            "complete" => InputClipType::Complete,
            _ => InputClipType::Relaxed,
        };
    }
    if let Some(ratio) = preset.neuron_selection_ratio {
        config.clip_neuron_selection_ratio = ratio;
    }
    if let Some(interm_domain) = preset.interm_domain {
        config.enable_clip_interm_domain = interm_domain;
    }
    if let Some(interm_topk) = preset.interm_topk {
        config.clip_interm_topk = interm_topk;
    }
    if let Some(in_alpha_crown) = preset.in_alpha_crown {
        config.clip_in_alpha_crown = in_alpha_crown;
    }
    if let Some(prune) = preset.prune {
        config.clip_interm_prune = prune;
    }
    if let Some(use_final_layer) = preset.use_final_layer {
        config.clip_interm_use_final_layer = use_final_layer;
    }
}

fn apply_phase_budget_preset(config: &mut BetaCrownConfig, preset: &PhaseBudgetPreset) {
    let pb = &mut config.phase_budget;
    if let Some(v) = preset.initial_bounds_fraction {
        pb.initial_bounds_fraction = v;
    }
    if let Some(v) = preset.upfront_pgd_fraction {
        pb.upfront_pgd_fraction = v;
    }
    if let Some(v) = preset.reduced_verification_fraction {
        pb.reduced_verification_fraction = v;
    }
    if let Some(v) = preset.disjunctive_pgd_fraction {
        pb.disjunctive_pgd_fraction = v;
    }
    if let Some(v) = preset.disjunctive_precheck_fraction {
        pb.disjunctive_precheck_fraction = v;
    }
    if let Some(v) = preset.mip_min_fraction {
        pb.mip_min_fraction = v;
    }
    if let Some(v) = preset.mip_min_secs {
        pb.mip_min_secs = v;
    }
    if let Some(v) = preset.mip_max_secs {
        pb.mip_max_secs = v;
    }
    if let Some(v) = preset.post_bab_pgd_fraction {
        pb.post_bab_pgd_fraction = v;
    }
    if let Some(v) = preset.attack_extension_fraction {
        pb.attack_extension_fraction = v;
    }
    if preset.disjunctive_pgd_max_secs.is_some() {
        pb.disjunctive_pgd_max_secs = preset.disjunctive_pgd_max_secs;
    }
}

pub(crate) fn parse_reduce_op(op: &str) -> Result<KfsbReduceOp> {
    match op.to_lowercase().as_str() {
        "min" => Ok(KfsbReduceOp::Min),
        "max" => Ok(KfsbReduceOp::Max),
        "mean" => Ok(KfsbReduceOp::Mean),
        _ => anyhow::bail!("Unknown kFSB reduce operation: '{op}'. Use: min, max, or mean"),
    }
}
