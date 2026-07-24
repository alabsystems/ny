// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Diagnostics and telemetry for DAG α-CROWN propagation.
//!
//! All functions in this file are behind `tracing::enabled!(tracing::Level::DEBUG)` guards
//! at the call site. They produce debug-level logging only and have no effect on bounds
//! computation.

use ny_core::Result;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::super::runtime_state::DagAlphaRuntimeState;
use crate::network::core::GraphNetwork;

/// Compute bound width statistics: (mean_width, max_width, min_width, invalid_count).
fn bound_width_summary(bounds: &BoundedTensor) -> (f32, f32, f32, usize) {
    let mut min_w = f32::INFINITY;
    let mut max_w = 0.0f32;
    let mut sum_w = 0.0f32;
    let mut count = 0usize;
    let mut invalid = 0usize;

    for (&l, &u) in bounds.lower().iter().zip(bounds.upper().iter()) {
        let w = u - l;
        if !w.is_finite() {
            invalid += 1;
            continue;
        }
        min_w = min_w.min(w);
        max_w = max_w.max(w);
        sum_w += w;
        count += 1;
    }

    let mean_w = if count > 0 {
        sum_w / (count as f32)
    } else {
        f32::NAN
    };
    (mean_w, max_w, min_w, invalid)
}

/// Compute unstable neuron statistics: (unstable_count, total_count, mean_width, max_width).
fn unstable_summary(bounds: &BoundedTensor) -> (usize, usize, f32, f32) {
    let mut unstable = 0usize;
    let mut total = 0usize;
    let mut max_w = 0.0f32;
    let mut mean_w_sum = 0.0f32;
    let mut mean_w_count = 0usize;

    for (&l, &u) in bounds.lower().iter().zip(bounds.upper().iter()) {
        total += 1;
        if l < 0.0 && u > 0.0 {
            unstable += 1;
        }
        let w = u - l;
        if w.is_finite() {
            max_w = max_w.max(w);
            mean_w_sum += w;
            mean_w_count += 1;
        }
    }

    let mean_w = if mean_w_count > 0 {
        mean_w_sum / (mean_w_count as f32)
    } else {
        f32::NAN
    };
    (unstable, total, mean_w, max_w)
}

impl GraphNetwork {
    /// Log pre-loop diagnostics: per-node bound widths and per-ReLU unstable neuron stats.
    ///
    /// Called once before the optimization loop when debug tracing is enabled.
    pub(super) fn log_pre_loop_diagnostics(
        &self,
        exec_order: &[String],
        node_bounds: &HashMap<String, BoundedTensor>,
        relu_nodes: &[(String, usize)],
        input: &BoundedTensor,
    ) -> Result<()> {
        let mut per_node: Vec<(f32, f32, String, String, usize)> = Vec::new();
        for name in exec_order {
            if let Some(bounds) = node_bounds.get(name) {
                let (mean_w, max_w, _min_w, invalid) = bound_width_summary(bounds);
                let layer_type = self
                    .nodes
                    .get(name)
                    .map(|n| n.layer.layer_type().to_string())
                    .unwrap_or_else(|| "<missing>".to_string());
                per_node.push((max_w, mean_w, name.clone(), layer_type, invalid));
            }
        }
        per_node.sort_by(|a, b| crate::cmp_utils::nan_last_descending_cmp(&a.0, &b.0));

        if let (Some((max_w, _, max_name, max_ty, _)), Some((_, mean_w, _, _, _))) = (
            per_node.first(),
            per_node.get(per_node.len().saturating_sub(1)),
        ) {
            debug!(
                "DAG α-CROWN: IBP width stats across {} nodes: widest={} ({} max_w={:.3e}), narrowest_mean_w≈{:.3e}",
                per_node.len(),
                max_name,
                max_ty,
                max_w,
                mean_w
            );
        }

        for (rank, (max_w, mean_w, name, ty, invalid)) in per_node.iter().take(20).enumerate() {
            debug!(
                "DAG α-CROWN: widest#{} node='{}' type={} mean_w={:.3e} max_w={:.3e} invalid={} ",
                rank, name, ty, mean_w, max_w, invalid
            );
        }

        for (rank, (name, _)) in relu_nodes.iter().take(20).enumerate() {
            let pre = self.relu_preactivation_bounds(
                name,
                input,
                node_bounds,
                "dag-alpha-debug-summary",
            )?;
            let (unstable, total, mean_w, max_w) = unstable_summary(pre);
            let ratio = if total > 0 {
                (unstable as f32) / (total as f32)
            } else {
                0.0
            };
            debug!(
                "DAG α-CROWN: ReLU preact#{} node='{}' unstable={}/{} ({:.1}%) mean_w={:.3e} max_w={:.3e}",
                rank,
                name,
                unstable,
                total,
                ratio * 100.0,
                mean_w,
                max_w
            );
        }

        Ok(())
    }
}

/// Log per-iteration alpha telemetry: alpha statistics and velocity for unstable ReLU neurons.
///
/// Called every 5 iterations within the optimization loop when debug tracing is enabled.
pub(super) fn log_iteration_telemetry(
    runtime: &DagAlphaRuntimeState,
    iter: usize,
    best_lower_sum: f32,
    prev_best_lower_sum: f32,
    lower_sum: f32,
    lr: f32,
) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let mut alpha_min = f32::INFINITY;
        let mut alpha_max = f32::NEG_INFINITY;
        let mut alpha_sum = 0.0f32;
        let mut alpha_count = 0usize;
        let mut vel_abs_sum = 0.0f32;
        let mut vel_abs_max = 0.0f32;
        let mut vel_count = 0usize;

        for node_name in runtime.relu_nodes() {
            let Some(alpha) = runtime.graph().alpha(node_name) else {
                continue;
            };
            let Some(mask) = runtime.graph().relu_unstable_mask(node_name) else {
                continue;
            };
            let Some(vel) = runtime.graph().relu_velocity(node_name) else {
                continue;
            };
            for i in 0..alpha.len() {
                if mask[i] {
                    let a = alpha[i];
                    if a.is_finite() {
                        alpha_min = alpha_min.min(a);
                        alpha_max = alpha_max.max(a);
                        alpha_sum += a;
                        alpha_count += 1;
                    }
                    let v = vel[i].abs();
                    if v.is_finite() {
                        vel_abs_sum += v;
                        vel_abs_max = vel_abs_max.max(v);
                        vel_count += 1;
                    }
                }
            }
        }

        let alpha_mean = if alpha_count > 0 {
            alpha_sum / (alpha_count as f32)
        } else {
            f32::NAN
        };
        let vel_abs_mean = if vel_count > 0 {
            vel_abs_sum / (vel_count as f32)
        } else {
            f32::NAN
        };

        debug!(
            "DAG α-CROWN iter {}: best_impr={:.3e} alpha_unstable_mean={:.3e} [{:.3e},{:.3e}] vel_abs_mean={:.3e} vel_abs_max={:.3e}",
            iter,
            best_lower_sum - prev_best_lower_sum,
            alpha_mean,
            alpha_min,
            alpha_max,
            vel_abs_mean,
            vel_abs_max
        );
    }
    // info-level so a `-v` probe can measure per-iteration alpha progress
    // (called every 5 iterations — bounded log volume).
    info!(
        "DAG α-CROWN iter {}: lower_sum = {:.6}, best_impr = {:.3e}, lr = {:.6}",
        iter,
        lower_sum,
        best_lower_sum - prev_best_lower_sum,
        lr
    );
}

/// Log summary of gradient skips due to non-finite values.
///
/// Called after the optimization loop completes.
pub(super) fn log_gradient_skip_summary(
    total_gradient_skips: usize,
    iterations: usize,
    num_relu_nodes: usize,
) {
    if total_gradient_skips > 0 {
        let total_updates = iterations * num_relu_nodes;
        warn!(
            "DAG α-CROWN: skipped {total_gradient_skips}/{total_updates} gradient updates \
             due to non-finite values"
        );
    }
}
