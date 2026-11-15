// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Paid advice only. All local support is recomputed from original source under
//! the full execution lease; no scalar or descriptor here grants authority.
use super::*;
const LOCAL_SELECTION_FIXED_WORK: u64 = (1 << 24) + 8192;
const LOCAL_SELECTION_WORK_PER_CELL: u64 = 512;
fn ranked_local_target(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    target: &str,
    end: Instant,
) -> Option<([(usize, f64, f64); 8], usize)> {
    lease.reserve_more_work(LOCAL_SELECTION_FIXED_WORK).ok()?;
    lease.checkpoint().ok()?;
    if Instant::now() >= end
        || target.len() > 1024
        || graph.output_name().len() > 1024
        || objective.is_empty()
        || objective.len() > 8192
    {
        return None;
    }
    let relu = graph.node(target)?;
    if !matches!(relu.layer(), Layer::ReLU(_)) {
        return None;
    }
    let parent = relu.require_unary_input().ok()?;
    if parent.len() > 1024 {
        return None;
    }
    let pre = if parent == crate::network::NETWORK_INPUT {
        input
    } else {
        bounds.get(parent)?
    };
    let width = pre.len();
    if !(5..=8192).contains(&width) || pre.shape().len() > 4 {
        return None;
    }
    let last = graph.node(graph.output_name())?;
    let Layer::Linear(linear) = last.layer() else {
        return None;
    };
    if linear.in_features() != width
        || linear.out_features() != objective.len()
        || width.checked_mul(objective.len())? > (1 << 20)
    {
        return None;
    }
    let mut current = last.require_unary_input().ok()?;
    for _ in 0..MAX_NODES {
        lease.checkpoint().ok()?;
        if Instant::now() >= end || current.len() > 1024 {
            return None;
        }
        if current == target {
            break;
        }
        let shape = graph.node(current)?;
        if !matches!(shape.layer(), Layer::Flatten(_) | Layer::Reshape(_)) {
            return None;
        }
        let parent = shape.require_unary_input().ok()?;
        for name in [current, parent] {
            if name.len() > 1024 {
                return None;
            }
            let b = bounds.get(name)?;
            if b.shape().len() > 4
                || b.len() != width
                || b.lower().as_slice().is_none()
                || b.upper().as_slice().is_none()
            {
                return None;
            }
        }
        current = parent;
    }
    if current != target {
        return None;
    }
    let lower = pre.lower().as_slice()?;
    let upper = pre.upper().as_slice()?;
    let weights = linear.weight.as_slice()?;
    if lower.len() != width
        || upper.len() != width
        || weights.len() != width.checked_mul(objective.len())?
    {
        return None;
    }
    // Pay the full D*O scan even when only crossing cells use those weights.
    // The +1 column covers every bound/gap/rank check, including stable cells.
    let scan = LOCAL_SELECTION_WORK_PER_CELL
        .checked_mul(u64::try_from(width).ok()?)?
        .checked_mul(u64::try_from(objective.len()).ok()?.checked_add(1)?)?;
    lease.reserve_more_work(scan).ok()?;
    let mut best = [(0usize, 0f64, 0f64); 8]; // index, score, negative magnitude
    let mut count = 0;
    for i in 0..width {
        if i % 128 == 0 {
            lease.checkpoint().ok()?;
            if Instant::now() >= end {
                return None;
            }
        }
        let (lo, hi) = (lower[i], upper[i]);
        if !ny_core::is_crown_coeff_safe(lo) || !ny_core::is_crown_coeff_safe(hi) || lo > hi {
            return None;
        }
        if lo >= 0. || hi <= 0. {
            continue;
        }
        let mut coefficient = 0f64;
        for (row, &c) in objective.iter().enumerate() {
            if row % 128 == 0 {
                lease.checkpoint().ok()?;
                if Instant::now() >= end {
                    return None;
                }
            }
            let w = weights[row * width + i];
            if !ny_core::is_crown_coeff_safe(c) || !ny_core::is_crown_coeff_safe(w) {
                return None;
            }
            coefficient += f64::from(c) * f64::from(w);
        }
        if !coefficient.is_finite() {
            return None;
        }
        if coefficient >= 0. {
            continue;
        }
        let gap = -f64::from(lo) * f64::from(hi) / (f64::from(hi) - f64::from(lo));
        let score = -coefficient * gap;
        if !score.is_finite() || score <= 0. {
            return None;
        }
        // Stable insertion order breaks equal scores by increasing coordinate.
        let position = (0..count).find(|&p| score > best[p].1).unwrap_or(count);
        if position < 8 {
            for p in ((position + 1)..=count.min(7)).rev() {
                best[p] = best[p - 1];
            }
            best[position] = (i, score, -coefficient);
            count = (count + 1).min(8);
        }
    }
    lease.checkpoint().ok()?;
    (count >= 5 && Instant::now() < end).then_some((best, count))
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LocalChoice {
    pub(super) active: usize,
    pub(super) neurons: [usize; 8],
    pub(super) weights: [f32; 8],
    pub(super) multiplier: f32,
}
pub(super) fn local_target(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    target: &str,
    end: Instant,
) -> Option<LocalChoice> {
    let (mut best, count) =
        ranked_local_target(graph, input, bounds, lease, objective, target, end)?;
    // The fixed advice reservation also covers this bounded original source
    // walk. No selected weight values are scanned here.
    let pre = graph.node(target)?.require_unary_input().ok()?;
    if pre.len() > 1024 {
        return None;
    }
    let node = graph.node(pre)?;
    let linear = match node.layer() {
        Layer::Linear(l) => l,
        Layer::BatchNorm(_) => {
            let name = node.require_unary_input().ok()?;
            if name.len() > 1024 {
                return None;
            }
            let Layer::Linear(l) = graph.node(name)?.layer() else {
                return None;
            };
            l
        }
        _ => return None,
    };
    if linear.in_features() == 0 || linear.in_features() > 65536 || linear.out_features() > 8192 {
        return None;
    }
    let magnitude = best[..count].iter().map(|v| v.2).fold(0f64, f64::max);
    let multiplier = (0.5 * magnitude) as f32;
    if !ny_core::is_crown_coeff_safe(multiplier) || multiplier <= 0. {
        return None;
    }
    best[..count].sort_unstable_by_key(|v| v.0);
    let mut neurons = [0; 8];
    let mut weights = [0f32; 8];
    for i in 0..count {
        let quantized = (256. * best[i].2 / magnitude).round();
        if !quantized.is_finite() || !(1. ..=256.).contains(&quantized) {
            return None;
        }
        neurons[i] = best[i].0;
        weights[i] = (quantized as f32) / 256.;
    }
    lease.checkpoint().ok()?;
    (Instant::now() < end).then_some(LocalChoice {
        active: count,
        neurons,
        weights,
        multiplier,
    })
}
