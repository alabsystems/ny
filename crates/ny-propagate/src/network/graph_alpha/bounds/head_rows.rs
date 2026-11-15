// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One original fixed-reference head opportunity, before broad CROWN walks.
//! Ranking is advice. Only same-map identity-row CROWN can commit endpoints.

use super::budget_policy;
use super::row_refine::{RowRefineOutcome, RowRefineRequest};
use crate::layers::Layer;
use crate::network::core::GraphNetwork;
use crate::types::CrownIbpPerNodeTimeBudget;
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Only the validated original root-fixed-reference entry sets this option.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OriginalRootHeadRows;

const MAX_HEAD_NODES: usize = 128;
const MAX_PREFIX_NODES: usize = 128;
const MAX_WIDTH: usize = 8192;
const MAX_WEIGHT_CELLS: usize = 1 << 20;
const MAX_ROWS: usize = 8;

mod bounded_gpu;

struct Plan {
    target: String,
    ranked: [usize; MAX_ROWS],
    selected: usize,
    eligible: usize,
    total: usize,
}

/// Private collector-owned commit. It cannot be constructed from caller scalars
/// or reused for a different graph/map/alpha request.
pub(super) struct HeadRowsTransaction {
    pub(super) target: String,
    pub(super) rows: [usize; MAX_ROWS],
    pub(super) selected: usize,
    pub(super) eligible: usize,
    pub(super) total: usize,
    pub(super) entered: bool,
    pub(super) stage: &'static str,
    committed: Option<Box<BoundedTensor>>,
    pub(super) seed_bytes: usize,
    pub(super) estimated_macs: u128,
    pub(super) elapsed_secs: f64,
    pub(super) grant_secs: f64,
    pub(super) rate_macs_per_sec: u64,
    pub(super) gpu: bounded_gpu::Observation,
}

fn live(end: Instant) -> bool {
    Instant::now() < end
}

fn plan(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
    demand: &HashSet<String>,
    end: Instant,
) -> Option<Plan> {
    // Both cost estimators traverse ancestors without an internal deadline.
    // Bound their entire possible graph, edge and name domains before entry.
    if graph.num_nodes() > MAX_PREFIX_NODES || graph.node_names().len() > MAX_PREFIX_NODES {
        return None;
    }
    for name in graph.node_names() {
        if !live(end) || name.len() > 1024 {
            return None;
        }
        let node = graph.node(name)?;
        if node.inputs().len() > 2 || node.inputs().iter().any(|s| s.len() > 1024) {
            return None;
        }
    }
    if !live(end) || graph.output_name().len() > 1024 {
        return None;
    }
    let output = graph.node(graph.output_name())?;
    let Layer::Linear(final_linear) = output.layer() else {
        return None;
    };
    let width = final_linear.in_features();
    let outputs = final_linear.out_features();
    let cells = width.checked_mul(outputs)?;
    if width == 0
        || width > MAX_WIDTH
        || outputs == 0
        || outputs > MAX_WIDTH
        || cells > MAX_WEIGHT_CELLS
    {
        return None;
    }
    let weights = final_linear.weight.as_slice()?;
    if weights.len() != cells {
        return None;
    }
    let mut name = output.require_unary_input().ok()?;
    let mut relu_name = None;
    for _ in 0..MAX_HEAD_NODES {
        if !live(end) || name.len() > 1024 {
            return None;
        }
        let node = graph.node(name)?;
        let boxed = bounds.get(name)?;
        if boxed.len() != width || boxed.shape().len() > 2 {
            return None;
        }
        if matches!(node.layer(), Layer::ReLU(_)) {
            relu_name = Some(name);
            break;
        }
        if !matches!(node.layer(), Layer::Flatten(_) | Layer::Reshape(_)) {
            return None;
        }
        name = node.require_unary_input().ok()?;
    }
    let target = graph.node(relu_name?)?.require_unary_input().ok()?;
    if target.len() > 1024 || !demand.contains(target) {
        return None;
    }
    let target_node = graph.node(target)?;
    let producer = match target_node.layer() {
        Layer::Linear(linear) => linear,
        Layer::BatchNorm(_) => {
            let parent = target_node.require_unary_input().ok()?;
            if parent.len() > 1024 {
                return None;
            }
            let Layer::Linear(linear) = graph.node(parent)?.layer() else {
                return None;
            };
            if bounds.get(parent)?.len() != width {
                return None;
            }
            linear
        }
        _ => return None,
    };
    if producer.out_features() != width || producer.in_features() == 0 {
        return None;
    }
    let base = bounds.get(target)?;
    if base.len() != width || base.shape().len() > 2 {
        return None;
    }
    let lower = base.lower().as_slice()?;
    let upper = base.upper().as_slice()?;
    let mut best = [(usize::MAX, 0.0_f64); MAX_ROWS];
    let mut selected = 0;
    let mut eligible = 0;
    for i in 0..width {
        if i % 32 == 0 && !live(end) {
            return None;
        }
        let (lo, hi) = (lower[i], upper[i]);
        if !ny_core::is_crown_coeff_safe(lo) || !ny_core::is_crown_coeff_safe(hi) || lo > hi {
            return None;
        }
        let mut magnitude = 0.0_f64;
        for o in 0..outputs {
            if o % 128 == 0 && !live(end) {
                return None;
            }
            let weight = weights[o * width + i];
            if !ny_core::is_crown_coeff_safe(weight) {
                return None;
            }
            magnitude = magnitude.max(f64::from(weight).abs());
        }
        if lo >= 0.0 || hi <= 0.0 || magnitude == 0.0 {
            continue;
        }
        let score = magnitude * (-f64::from(lo)) * f64::from(hi) / (f64::from(hi) - f64::from(lo));
        if !score.is_finite() || score <= 0.0 {
            return None;
        }
        eligible += 1;
        let position = (0..selected)
            .find(|&j| score > best[j].1 || (score == best[j].1 && i < best[j].0))
            .unwrap_or(selected);
        if position < MAX_ROWS {
            let next_len = (selected + 1).min(MAX_ROWS);
            for j in (position + 1..next_len).rev() {
                best[j] = best[j - 1];
            }
            best[position] = (i, score);
            selected = next_len;
        }
    }
    if selected == 0 || !live(end) {
        return None;
    }
    let mut ranked = [0; MAX_ROWS];
    for i in 0..selected {
        ranked[i] = best[i].0;
    }
    Some(Plan {
        target: target.into(),
        ranked,
        selected,
        eligible,
        total: width,
    })
}

fn reservation(
    now: Instant,
    end: Instant,
    total: usize,
    policy: &CrownIbpPerNodeTimeBudget,
) -> Option<Instant> {
    let remaining = end.checked_duration_since(now)?.as_secs_f64();
    let cap = budget_policy::weighted_per_node_budget_secs_unfloored(
        remaining,
        1.0,
        1.0,
        total as f64,
        policy,
    )?;
    let grant = 2.0_f64.min(remaining / 4.0).min(cap);
    let (floor, _) = budget_policy::effective_floor_secs_for_dims(policy, total as f64);
    if !grant.is_finite() || grant <= 0.0 || grant < floor {
        return None;
    }
    now.checked_add(Duration::from_secs_f64(grant))
        .map(|local| local.min(end))
}

fn local_bytes(rows: usize, total: usize) -> Option<(usize, usize)> {
    let seed = rows.checked_mul(total)?.checked_mul(2 * size_of::<f32>())?;
    // Original publication base plus scattered/intersected/committed targets can
    // coexist. Prefix/cache/error/backend workspace remains independently gated.
    let publication = total.checked_mul(8 * size_of::<f32>())?;
    Some((
        seed,
        seed.checked_add(publication)?
            .checked_add(size_of::<HeadRowsTransaction>())?
            .checked_add(4096)?,
    ))
}

/// One invocation: zero or one subset walk, never a full-width retry.
pub(super) fn run(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    demand: &HashSet<String>,
    engine: Option<&dyn GemmEngine>,
    end: Instant,
    dense_budget: usize,
    prefix_cost_enabled: bool,
) -> Option<HeadRowsTransaction> {
    let started = Instant::now();
    // Planning shares the original tightening clock; it does not get an extra
    // two seconds when rank/cost inspection finishes.
    let plan = plan(graph, bounds, demand, end)?;
    let mut result = HeadRowsTransaction {
        target: plan.target,
        rows: plan.ranked,
        selected: plan.selected,
        eligible: plan.eligible,
        total: plan.total,
        entered: false,
        stage: "budget",
        committed: None,
        seed_bytes: 0,
        estimated_macs: 0,
        elapsed_secs: 0.0,
        grant_secs: 0.0,
        rate_macs_per_sec: 0,
        gpu: bounded_gpu::Observation::default(),
    };
    let Some(local_end) = reservation(
        started,
        end,
        result.total,
        &graph.crown_ibp_per_node_time_budget,
    ) else {
        return Some(result);
    };
    result.grant_secs = local_end.saturating_duration_since(started).as_secs_f64();
    let base = bounds.get(&result.target)?;
    let rate = crate::network::core::graph::forward_linear_cached_rate();
    result.rate_macs_per_sec = rate.macs_per_sec;
    let model = budget_policy::WalkCostModel::new(rate.macs_per_sec as f64);
    let prefix = budget_policy::PrefixCostAdmissionModel::new(prefix_cost_enabled);
    let mut admitted = None;
    let mut previous = 0;
    for ceiling in [8, 4, 2, 1] {
        let count = result.selected.min(ceiling);
        if count == previous {
            continue;
        }
        previous = count;
        if !live(local_end) {
            result.stage = "admission-deadline";
            break;
        }
        let Some((seed, memory)) = local_bytes(count, result.total) else {
            continue;
        };
        if memory > dense_budget {
            result.stage = "memory";
            continue;
        }
        let Some(macs) = graph.collector_walk_macs_with_shapes(&result.target, count, Some(bounds))
        else {
            result.stage = "cost-unavailable";
            continue;
        };
        let Some(estimate) = model
            .estimate_secs(macs)
            .filter(|v| v.is_finite() && *v > 0.0)
        else {
            result.stage = "cost-unavailable";
            continue;
        };
        let remaining = local_end
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if !matches!(
            budget_policy::admit_walk_with_estimate(estimate, remaining, remaining, false),
            budget_policy::WalkAdmissionDecision::Admit
        ) {
            result.stage = "walk-cost";
            continue;
        }
        let units =
            super::crown_tighten::dense_crown_prefix_work_units(graph, &result.target, count);
        if matches!(
            prefix.admit(
                units,
                Some(local_end.saturating_duration_since(Instant::now()))
            ),
            budget_policy::PrefixCostAdmission::RetainIbp { .. }
        ) {
            result.stage = "prefix-cost";
            continue;
        }
        if !live(local_end) {
            result.stage = "admission-deadline";
            break;
        }
        admitted = Some((count, seed, macs));
        break;
    }
    let Some((count, seed, macs)) = admitted else {
        result.elapsed_secs = started.elapsed().as_secs_f64();
        return Some(result);
    };
    result.selected = count;
    let ranked = result.rows;
    result.rows[..count].sort_unstable();
    if result.rows[..count].windows(2).any(|p| p[0] >= p[1])
        || result.rows[..count].iter().any(|&i| i >= result.total)
    {
        result.stage = "row-identity";
        return Some(result);
    }
    result.seed_bytes = seed;
    result.estimated_macs = macs;
    result.entered = true;
    let finalized_ancestors = HashMap::new();
    let request = RowRefineRequest {
        input,
        target: &result.target,
        crown_bounds: &finalized_ancestors,
        ibp_bounds: bounds,
        base,
        engine,
        rows: &result.rows[..count],
        per_node_deadline: Some(local_end),
        deadline_is_hard: true,
        collector_patches_override: false,
        label: "root selective head rows",
        cut_ctx: None,
        dense_budget_bytes: dense_budget,
        enforce_dense_budget: true,
        patches_seed: false,
        shrink_only: true,
    };
    // The ordinary suffix needs three seconds of unpollable host headroom;
    // this opportunity has at most two. Use the separately bounded small-row
    // contract with charged, deadline-aware extraction instead. A pre-dispatch
    // refusal retains the original host route; a dispatched attempt owns the
    // opportunity, including errors, and is never retried.
    let outcome = match bounded_gpu::try_refine(graph, &request, &ranked[..count], &mut result.gpu)
    {
        Some(attempt) => {
            result.selected = attempt.selected;
            result.rows = attempt.rows;
            result.seed_bytes = local_bytes(attempt.selected, result.total).map_or(0, |v| v.0);
            // Row-linear work estimate, descriptive only; admission above paid
            // for the original, possibly larger batch.
            result.estimated_macs = macs / count as u128 * attempt.selected as u128;
            attempt.outcome
        }
        None => graph.refine_rows_over_box(request),
    };
    result.finish(base, outcome, local_end);
    result.elapsed_secs = started.elapsed().as_secs_f64();
    Some(result)
}

fn selected_intersection(
    base: &BoundedTensor,
    candidate: &BoundedTensor,
    rows: &[usize],
) -> Option<BoundedTensor> {
    if base.shape() != candidate.shape()
        || rows.is_empty()
        || rows.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let lo = candidate.lower().as_slice()?;
    let hi = candidate.upper().as_slice()?;
    let mut lower = base.lower().clone();
    let mut upper = base.upper().clone();
    let lows = lower.as_slice_mut()?;
    let highs = upper.as_slice_mut()?;
    for &i in rows {
        if i >= lo.len()
            || !ny_core::is_crown_coeff_safe(lo[i])
            || !ny_core::is_crown_coeff_safe(hi[i])
            || lo[i] > hi[i]
            || !ny_core::is_crown_coeff_safe(lows[i])
            || !ny_core::is_crown_coeff_safe(highs[i])
            || lows[i] > highs[i]
        {
            return None;
        }
        if lo[i] > lows[i] {
            lows[i] = lo[i];
        }
        if hi[i] < highs[i] {
            highs[i] = hi[i];
        }
        if lows[i] > highs[i] {
            return None;
        }
    }
    BoundedTensor::new(lower, upper).ok()
}

impl HeadRowsTransaction {
    /// Fixed-size observation only; these values never enter admission/proof.
    pub(super) fn selected_endpoints(&self, bound: &BoundedTensor) -> Option<[[f32; 8]; 2]> {
        let lo = bound.lower().as_slice()?;
        let hi = bound.upper().as_slice()?;
        let mut endpoints = [[0.0; 8]; 2];
        for j in 0..self.selected {
            let i = self.rows[j];
            endpoints[0][j] = *lo.get(i)?;
            endpoints[1][j] = *hi.get(i)?;
        }
        Some(endpoints)
    }

    fn finish(&mut self, base: &BoundedTensor, outcome: RowRefineOutcome, local_end: Instant) {
        self.stage = match outcome {
            RowRefineOutcome::Refined {
                bounds: candidate,
                k,
                ..
            } if k == self.selected => {
                // This selected-only merge also checks exact shape/finite endpoints.
                // Initial publication is atomic and must finish before local_end.
                match selected_intersection(base, &candidate, &self.rows[..self.selected]) {
                    Some(committed) if live(local_end) => {
                        self.committed = Some(Box::new(committed));
                        "committed"
                    }
                    Some(_) => "publication-deadline",
                    None => "publication-refused",
                }
            }
            RowRefineOutcome::Refined { .. } => "row-count",
            RowRefineOutcome::Declined(_) => "walk-declined",
            RowRefineOutcome::Disjoint { .. } => "disjoint",
            RowRefineOutcome::Failed(ny_core::NyError::DeadlineExceeded(_)) => "walk-deadline",
            RowRefineOutcome::Failed(_) => "walk-failed",
        };
    }

    /// Consume the on-time private commit at its topological visit. The original
    /// baseline needs no new intersection; a composed baseline needs a fresh,
    /// selected-only intersection under the remaining original collector clock.
    pub(super) fn publish(
        &mut self,
        base: &BoundedTensor,
        original: &BoundedTensor,
        collection_end: Option<Instant>,
    ) -> (BoundedTensor, usize) {
        let Some(committed) = self.committed.take() else {
            return (base.clone(), 0);
        };
        if std::ptr::eq(base, original) {
            return (*committed, self.selected);
        }
        if collection_end.is_none_or(|end| !live(end)) {
            self.stage = "composed-publication-deadline";
            return (base.clone(), 0);
        }
        match selected_intersection(base, &committed, &self.rows[..self.selected]) {
            Some(merged) if collection_end.is_some_and(live) => (merged, self.selected),
            _ => {
                self.stage = "composed-publication-refused";
                (base.clone(), 0)
            }
        }
    }
}

#[cfg(test)]
#[path = "head_rows_tests.rs"]
mod tests;
