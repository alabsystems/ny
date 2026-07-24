// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Saturation-Escape Branching (SEB) input-split scorer — M1.
//!
//! See `docs/SATURATION_ESCAPE_BRANCHING_DESIGN.md`.
//!
//! On nets with smooth saturating activations (sigmoid / tanh) the binding
//! output margin sits behind a saturated activation, so `∂output/∂x ≈ 0` for
//! every input dimension and the baseline SB scorer degenerates to width-only —
//! blind splitting on a ~300-dim input. SEB instead scores each candidate split
//! dimension by how much it shrinks the *saturated width* of the binding
//! pre-activation (logit): the portion of `[l_z, u_z]` with `|z| > τ`, which
//! carries zero usable bound signal. Splitting the dims that pull the logit out
//! of saturation is what makes the box close (probe: 57 leaves vs 697 blind).
//!
//! Soundness: this module is **advisory only**. It returns a ranked list of
//! *which* input dimensions to midpoint-split; the split partition itself
//! (`multi_dim_split_boxes`) is exact and unchanged, so the union cover stays
//! complete regardless of the ranking. The saturation widths are computed from
//! the sound IBP node bounds and consumed for the argmax ONLY — never to raise
//! or lower any verdict bound. Gated off by default via `NY_SAT_ESCAPE_BRANCH`.

use std::collections::HashMap;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::GraphNetwork;

/// Env gate for Saturation-Escape Branching (M1). Default OFF ⇒ the shipped
/// path is byte-identical. Mirrors the `imb::enabled()` convention.
pub(crate) fn enabled() -> bool {
    matches!(
        std::env::var("NY_SAT_ESCAPE_BRANCH").ok().as_deref(),
        Some("1")
    )
}

/// Cap on the number of positive-width candidate dimensions the finite-diff
/// scorer probes, bounding the per-split IBP cost. When a domain has more ranged
/// dims than this, the widest `MAX_CANDIDATES` are probed (still advisory).
const MAX_CANDIDATES: usize = 96;

/// A saturating activation node and the knee `τ` of its activation.
struct SaturatingLogit {
    /// Name of the node producing the pre-activation (logit) — the sigmoid/tanh
    /// node's first input.
    logit_node: String,
    /// Saturation knee: `|z| > τ ⇒ activation derivative ≈ 0`.
    tau: f32,
}

/// Discover every sigmoid/tanh node's logit (pre-activation) producer.
fn find_saturating_logits(graph: &GraphNetwork) -> Vec<SaturatingLogit> {
    let mut out = Vec::new();
    for name in graph.node_names() {
        let Some(node) = graph.node(name) else {
            continue;
        };
        let tau = match node.layer() {
            // σ'(4) ≈ 0.018; the logit is effectively saturated beyond ~4.
            Layer::Sigmoid(_) => 4.0_f32,
            // tanh'(2.5) ≈ 0.028; knee is tighter than sigmoid.
            Layer::Tanh(_) => 2.5_f32,
            _ => continue,
        };
        if let Some(logit) = node.inputs().first() {
            out.push(SaturatingLogit {
                logit_node: logit.clone(),
                tau,
            });
        }
    }
    out
}

/// Width of `[l, u] ∩ { |z| > τ }` — the saturated (bound-signal-free) portion
/// of a scalar logit interval.
#[inline]
fn saturated_width(l: f32, u: f32, tau: f32) -> f32 {
    // width of [l,u] ∩ (τ, ∞)
    let upper = (u - tau).max(0.0) - (l - tau).max(0.0);
    // width of [l,u] ∩ (-∞, -τ)
    let lower = (-tau - l).max(0.0) - (-tau - u).max(0.0);
    upper + lower
}

/// Total saturated width over every saturating logit node, summed across each
/// node's output elements. Returns `None` if any logit's bounds are missing
/// (the scorer then declines and the caller falls back to the SB heuristic).
fn total_saturated_width(
    logits: &[SaturatingLogit],
    node_bounds: &HashMap<String, BoundedTensor>,
) -> Option<f32> {
    let mut total = 0.0_f32;
    for logit in logits {
        let bounds = node_bounds.get(&logit.logit_node)?;
        let flat = bounds.flatten();
        let lo = flat.lower();
        let up = flat.upper();
        for idx in 0..flat.len() {
            let l = lo[[idx]];
            let u = up[[idx]];
            if l.is_finite() && u.is_finite() {
                total += saturated_width(l, u, logit.tau);
            }
        }
    }
    Some(total)
}

/// Select the top-`depth` input dimensions by saturation-escape score, or `None`
/// to defer to the baseline SB scorer.
///
/// Score(i) = `W_sat(base) − E[W_sat | bisect dim i]`, where the expectation
/// averages the saturated width of the two midpoint children, each recomputed
/// from a sound IBP pass over the tightened box. A dimension that pulls the
/// binding logit out of saturation scores highest.
///
/// Returns `None` (⇒ fall back to SB) when: SEB is unsupported for this graph
/// (no sigmoid/tanh node), the base logit is not saturated (`W_sat ≤ 0`, so the
/// smooth node already carries usable signal and SB's objective coefficient is
/// meaningful), the IBP pass fails, or no dimension yields a positive reduction.
pub(crate) fn select_seb_dims(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    depth: usize,
) -> Option<Vec<usize>> {
    if depth == 0 {
        return None;
    }
    let logits = find_saturating_logits(graph);
    if logits.is_empty() {
        return None;
    }

    let base_bounds = graph
        .collect_node_bounds_with_engine(input_bounds, engine)
        .ok()?;
    let w_base = total_saturated_width(&logits, &base_bounds)?;
    // Nothing is saturated: the smooth node already carries a usable bound, so
    // the objective coefficient the SB scorer reads is meaningful — defer.
    // (NaN also defers: partial_cmp is None then, which is != Greater.)
    if w_base.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }

    let flat = input_bounds.flatten();
    let len = flat.len();
    let lo = flat.lower();
    let up = flat.upper();

    // Candidate ranged dims (positive finite width). Cap at MAX_CANDIDATES,
    // keeping the widest, to bound the finite-diff IBP cost.
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for dim in 0..len {
        let l = lo[[dim]];
        let u = up[[dim]];
        let width = u - l;
        if width.is_finite() && width > 0.0 {
            candidates.push((dim, width));
        }
    }
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() > MAX_CANDIDATES {
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(MAX_CANDIDATES);
    }

    let shape = input_bounds.lower().shape().to_vec();
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    for (dim, _width) in candidates {
        let l = lo[[dim]];
        let u = up[[dim]];
        let mid = l + (u - l) / 2.0;
        let Some(w_left) =
            child_saturated_width(graph, &flat, &shape, dim, l, mid, &logits, engine)
        else {
            continue;
        };
        let Some(w_right) =
            child_saturated_width(graph, &flat, &shape, dim, mid, u, &logits, engine)
        else {
            continue;
        };
        let expected = f32::midpoint(w_left, w_right);
        let reduction = w_base - expected;
        if reduction.is_finite() && reduction > 0.0 {
            scores.push((dim, reduction));
        }
    }

    if scores.is_empty() {
        return None;
    }
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scores.truncate(depth);
    Some(scores.into_iter().map(|(dim, _)| dim).collect())
}

/// Saturated width of the binding logit after tightening `dim` to `[new_l,
/// new_u]` and rerunning IBP. `None` if the pass or bounds are unavailable.
#[allow(clippy::too_many_arguments)]
fn child_saturated_width(
    graph: &GraphNetwork,
    flat: &BoundedTensor,
    shape: &[usize],
    dim: usize,
    new_l: f32,
    new_u: f32,
    logits: &[SaturatingLogit],
    engine: Option<&dyn GemmEngine>,
) -> Option<f32> {
    let mut child_lower = flat.lower().clone();
    let mut child_upper = flat.upper().clone();
    child_lower[[dim]] = new_l;
    child_upper[[dim]] = new_u;
    let child_lower = child_lower.into_shape_clone(ndarray::IxDyn(shape)).ok()?;
    let child_upper = child_upper.into_shape_clone(ndarray::IxDyn(shape)).ok()?;
    let child = BoundedTensor::new(child_lower, child_upper).ok()?;
    let node_bounds = graph.collect_node_bounds_with_engine(&child, engine).ok()?;
    total_saturated_width(logits, &node_bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_width_fully_inside_saturation() {
        // [5, 10] with τ=4: entirely in the z>τ region → width 5.
        assert_eq!(saturated_width(5.0, 10.0, 4.0), 5.0);
    }

    #[test]
    fn saturated_width_straddles_upper_knee() {
        // [2, 10], τ=4: saturated part is (4, 10] → width 6.
        assert_eq!(saturated_width(2.0, 10.0, 4.0), 6.0);
    }

    #[test]
    fn saturated_width_inside_linear_region_is_zero() {
        // [-3, 3], τ=4: no |z|>4 portion.
        assert_eq!(saturated_width(-3.0, 3.0, 4.0), 0.0);
    }

    #[test]
    fn saturated_width_two_sided() {
        // [-10, 10], τ=4: (-10,-4] width 6 + (4,10] width 6 = 12.
        assert_eq!(saturated_width(-10.0, 10.0, 4.0), 12.0);
    }

    #[test]
    fn enabled_defaults_off() {
        // Not asserting on the process env (test isolation), just that the
        // helper reads the documented variable name without panicking.
        let _ = enabled();
    }
}
