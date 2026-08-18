// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default value helpers for `BetaCrownConfig` serde deserialization.
//!
//! Each function returns the default value for the corresponding config field.

pub(crate) fn default_max_domains() -> usize {
    100_000
}

pub(crate) fn default_timeout() -> std::time::Duration {
    std::time::Duration::from_mins(5)
}

pub(crate) fn default_use_alpha_crown() -> bool {
    true
}

pub(crate) fn default_max_depth() -> usize {
    100
}

pub(crate) fn default_enable_la_warm_start() -> bool {
    true
}

pub(crate) fn default_max_cuts() -> usize {
    1000
}

pub(crate) fn default_input_split_depth() -> usize {
    1
}

pub(crate) fn default_input_split_coeff_thresh() -> f32 {
    1e-3
}

/// Per-sub-domain α refinement iterations in input-split BaB.
///
/// Default 0 = ny's historical behavior: a single frozen-alpha CROWN pass per
/// domain (no per-domain α optimization). Set > 0 to enable warm-started α
/// re-optimization for each sub-domain (alpha-beta-CROWN
/// `input_split_alpha_iteration`, reference default 5).
pub(crate) fn default_input_split_alpha_iteration() -> usize {
    0
}

/// Learning rate for per-sub-domain α refinement in input-split BaB.
///
/// Matches alpha-beta-CROWN `input_split_lr_alpha` (default 0.05). Only used
/// when `input_split_alpha_iteration > 0`.
pub(crate) fn default_input_split_lr_alpha() -> f32 {
    0.05
}

pub(crate) fn default_input_split_touch_zero_score() -> f32 {
    0.0
}

/// Maximum deferred-rebound tranche for the authenticated affine-conic lane.
///
/// A small tranche returns freshly bounded domains to the priority heap before
/// the next pop and also bounds the live per-domain affine matrices.
pub(crate) fn default_input_split_conic_queue_refresh_batch_size() -> usize {
    512
}

pub(crate) fn default_input_split_sb_margin_weight() -> f32 {
    1.0
}

pub(crate) fn default_min_cut_depth() -> usize {
    2
}

pub(crate) fn default_near_miss_margin() -> f32 {
    0.1
}

pub(crate) fn default_max_proactive_cuts() -> usize {
    100
}

pub(crate) fn default_biccos_drop_ratio() -> f32 {
    0.5
}

pub(crate) fn default_cut_stale_iters() -> usize {
    200
}

pub(crate) fn default_cut_hard_stale_iters() -> usize {
    1000
}

pub(crate) fn default_cut_lambda_min() -> f32 {
    1e-3
}

pub(crate) fn default_cut_proactive_fraction() -> f32 {
    0.2
}

pub(crate) fn default_biccos_min_verified() -> usize {
    5
}

pub(crate) fn default_biccos_min_verified_rate() -> f32 {
    0.05
}

pub(crate) fn default_biccos_verified_rate_window() -> usize {
    20
}

pub(crate) fn default_biccos_min_cuts() -> usize {
    3
}

pub(crate) fn default_biccos_min_bound_gain() -> f32 {
    1e-4
}

pub(crate) fn default_biccos_bound_gain_window() -> usize {
    20
}

pub(crate) fn default_biccos_cold_max_iters() -> usize {
    40
}

pub(crate) fn default_biccos_cut_window() -> usize {
    40
}

pub(crate) fn default_biccos_min_cut_yield() -> f32 {
    0.05
}

pub(crate) fn default_biccos_cut_yield_window() -> usize {
    20
}

pub(crate) fn default_biccos_cut_yield_patience() -> usize {
    2
}

pub(crate) fn default_fsb_candidates() -> usize {
    8
}

pub(crate) fn default_pgd_restarts() -> usize {
    100
}

pub(crate) fn default_pgd_steps() -> usize {
    50
}

pub(crate) fn default_pgd_osi_steps() -> usize {
    20
}

/// Per-step exponential decay applied to the PGD/Adam learning rate.
///
/// Matches `AdamClippingParams::lr_decay` (0.99) and alpha-beta-CROWN
/// `attack_pgd.py:255` (`ExponentialLR(opt, lr_decay)`).
pub(crate) fn default_pgd_lr_decay() -> f32 {
    0.99
}

/// Effective-dimension gate for the dense low-dim sweep (#dense-sweep).
pub(crate) fn default_pgd_dense_sweep_max_dims() -> usize {
    3
}

/// Forward-evaluation budget for the dense low-dim sweep (#dense-sweep):
/// a 128×128 initial grid plus refinement for the 2-dim case.
pub(crate) fn default_pgd_dense_sweep_points() -> usize {
    32_768
}

pub(crate) fn default_relaxed_clip_iterations() -> usize {
    1 // Single iteration matches Clip-and-Verify baseline
}

pub(crate) fn default_clip_neuron_selection_ratio() -> f32 {
    -1.0 // Disabled: apply to all neurons (matches alpha-beta-CROWN default)
}

pub(crate) fn default_clip_interm_topk() -> usize {
    3 // Preserve NY's public/API default; scored presets opt into larger values explicitly.
}

/// Root dense-head CROWN intermediate pass wall-clock cap.
///
/// The measured cifar100_2024 head backward takes about 0.5 s on the GB10;
/// 2 s leaves thermal/runtime headroom while keeping the pass a small bounded
/// pre-BaB investment.
pub(crate) fn default_root_crown_interm_max_secs() -> u64 {
    2
}

/// Maximum dense-fed ReLU pre-activation width admitted to the root CROWN pass.
///
/// cifar100_2024's measured target is 100-wide. The 512 cap admits comparable
/// classifier heads without accidentally seeding a very wide dense hidden layer.
pub(crate) fn default_root_crown_interm_max_dim() -> usize {
    512
}

/// Root sparse-row CROWN intermediate pass wall-clock cap.
///
/// The measured cifar100_2024 crossing-row fold over all structurally eligible
/// convolutional targets takes about 1.0-1.5 s on the GB10. Two seconds bounds
/// the one-time investment while leaving the root objective and BaB dominant.
pub(crate) fn default_root_sparse_interm_crown_max_secs() -> u64 {
    2
}

/// Largest ReLU pre-activation admitted to the sparse-row root CROWN pass.
///
/// Unlike the dense identity pass, cost is capped by selected crossing rows;
/// 8192 admits CIFAR residual tensors without admitting unbounded activations.
pub(crate) fn default_root_sparse_interm_crown_max_dim() -> usize {
    8_192
}

/// Maximum crossing rows seeded for any one sparse intermediate target.
pub(crate) fn default_root_sparse_interm_crown_max_rows() -> usize {
    512
}

/// Maximum number of convolutional targets processed deepest-first at root.
pub(crate) fn default_root_sparse_interm_crown_max_targets() -> usize {
    4
}

/// One row window: byte-identical to the historical single comprehensive sweep.
/// Raising it trades wall clock for root coverage; every extra window is an
/// independent atomic shrink-only sweep, so stopping early is always safe.
pub(crate) fn default_root_comprehensive_gpu_interm_chunks() -> usize {
    1
}

pub(crate) fn default_use_analytical_beta_gradients() -> bool {
    true
}

pub(crate) fn default_root_beta_iterations() -> usize {
    20 // Run 20 β optimization iterations on root before BaB
}

pub(crate) fn default_early_stop_patience() -> usize {
    10 // Matches alpha-beta-CROWN optimized_bounds.py early_stop_patience
}

pub(crate) fn default_beta_max_depth() -> usize {
    3 // Optimize β for domains at depth 0, 1, 2, 3
}

pub(crate) fn default_lambda_opt_interval() -> usize {
    20 // Optimize cut lambdas every 20 domains
}

pub(crate) fn default_lambda_lr() -> f32 {
    0.05 // Learning rate for lambda Adam optimization
}

pub(crate) fn default_max_relu_split_depth() -> usize {
    1 // Single-neuron split (current behavior, backward compatible)
}

pub(crate) fn default_min_batch_fill_ratio() -> f32 {
    0.5 // Multi-depth when queue < 50% of batch_size
}

pub(crate) fn default_max_queue_size() -> usize {
    500_000 // Cap on stored domains before low-priority eviction
}

/// #relational-bab edge escalation: distance-to-threshold gate (see
/// `BetaCrownConfig::input_split_edge_milp_gap`).
pub(crate) fn default_edge_milp_gap() -> f32 {
    0.01
}

/// #relational-bab edge escalation: minimum BaB depth gate.
pub(crate) fn default_edge_milp_depth() -> usize {
    20
}

/// #relational-bab option B: per-wave cap on α edge passes.
pub(crate) fn default_edge_alpha_top() -> usize {
    256
}

/// #relational-bab option B: α iterations per edge pass.
pub(crate) fn default_edge_alpha_iters() -> usize {
    25
}
