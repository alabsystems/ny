// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One original-domain intermediate transaction and one complete objective replay.
//! Cached coefficients rank proposals only. Every published scalar comes from
//! newly extracted checked arithmetic over the original input and objective.
use super::*;
use crate::batched_domain::CachedLinearBounds;
use crate::network::raw_root_cut::{
    bounded_fold_bytes, bounded_fold_work, extract_charged_raw_bn_chain,
    quote_charged_raw_bn_chain, ExtractionLimits, ExtractionQuote,
};
use ndarray::{ArrayD, IxDyn};
use ny_core::{GpuCrownBackward, GpuCrownSeed};

pub(super) const FIXED_BYTES: usize = 16 << 10;
const PLAN_WORK: u64 = (MAX_NODES * MAX_NODES * (8 * 1024 + 4096)) as u64;
const MAX_SELECTED: usize = 64;
const BATCH: usize = ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS;
const MAX_TARGET: usize = 1 << 20; // The existing raw-source dimension limit.
const MAP_BYTES: usize = 2 << 20;

#[derive(Clone, Copy)]
pub(crate) struct Advice<'a> {
    pub cache: &'a CachedLinearBounds,
    pub active_indices: &'a [usize],
}

pub(super) fn capable(engine: &dyn GemmEngine) -> bool {
    engine.as_gpu_crown_backward().is_some_and(|g| {
        let n = g.deadline_bounded_resnet_sound_max_rows();
        g.provides_sound_gpu_crown() && n > 0 && n <= BATCH
    })
}

pub(super) fn ready(local: Option<&dyn GemmEngine>) -> Option<&dyn GemmEngine> {
    if crate::sound_f64_gemm::cpu_only_f64_active() {
        return None;
    }
    local.filter(|e| capable(*e)).or_else(|| {
        let ordinary = crate::sound_f64_gemm::preinitialized_root_cut_engine();
        let [resident, wide] = crate::sound_gpu_gate::preinitialized_root_cut_engines();
        [ordinary, resident, wide]
            .into_iter()
            .flatten()
            .find(|e| capable(*e))
    })
}

struct Poll<'a> {
    end: Instant,
    callback: &'a mut dyn FnMut() -> bool,
    alive: bool,
}
impl Poll<'_> {
    fn check(&mut self) -> bool {
        self.alive &= Instant::now() < self.end && (self.callback)();
        self.alive
    }
}

pub(super) struct Quote<'a> {
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    rows: [usize; MAX_SELECTED],
    selected: usize,
    batch: usize,
    prefix: ExtractionQuote,
    full: ExtractionQuote,
    limits: ExtractionLimits,
    end: Instant,
    work: u64,
    bytes: usize,
    score: f64,
}
impl Quote<'_> {
    pub(super) fn work(&self) -> u64 {
        self.work
    }
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(super) fn calls(&self) -> usize {
        self.selected.div_ceil(self.batch) + 1
    }
}

/// Reject a pure affine input prefix: its forward box has no upstream
/// activation correlation for this method to recover. This is selection only.
fn has_upstream_relu(graph: &GraphNetwork, target: &str) -> Option<bool> {
    let mut current = target;
    let mut found = false;
    for _ in 0..MAX_NODES {
        if current == crate::network::NETWORK_INPUT {
            return Some(found);
        }
        if current.len() > 1024 {
            return None;
        }
        let node = graph.node(current)?;
        found |= matches!(node.layer(), Layer::ReLU(_));
        current = node.require_unary_input().ok()?;
    }
    None
}

fn spatial_source(graph: &GraphNetwork, target: &str) -> Option<()> {
    let node = graph.node(target)?;
    match node.layer() {
        Layer::Conv2d(_) => Some(()),
        Layer::BatchNorm(_) => {
            let parent = node.require_unary_input().ok()?;
            if parent.len() > 1024 {
                return None;
            }
            matches!(graph.node(parent)?.layer(), Layer::Conv2d(_)).then_some(())
        }
        _ => None,
    }
}

fn costs(
    prefix: ExtractionQuote,
    full: ExtractionQuote,
    width: usize,
    objective: usize,
    selected: usize,
    batch: usize,
) -> Option<(u64, usize)> {
    let mut work = bounded_fold_work(full, 1, 1)?;
    let mut first = 0;
    while first < selected {
        let n = batch.min(selected - first);
        // Charge complete per-call validation/extraction even though the one
        // immutable prefix transcript is reused. No batch borrows a new clock.
        work = work.checked_add(bounded_fold_work(prefix, n as u64, 1)?)?;
        first += n;
    }
    work = work
        .checked_add(512u64.checked_mul(width as u64)?)?
        .checked_add(4096 * MAX_NODES as u64)?
        .checked_add(4096 * objective as u64)?;
    // Prefix and full descriptors never coexist. The copied target endpoints,
    // borrowed overlay, immutable seed Arc copy peaks and all result/metadata
    // places are additional to the shared fold's conservative workspace.
    let fold = bounded_fold_bytes(prefix, batch.min(selected))?.max(bounded_fold_bytes(full, 1)?);
    let bytes = fold
        .checked_add(32usize.checked_mul(width)?)?
        .checked_add(16usize.checked_mul(batch)?.checked_mul(width)?)?
        .checked_add(MAP_BYTES)?;
    Some((work, bytes))
}

#[allow(clippy::too_many_arguments)]
fn quote_rows<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    ranked: &[(usize, f64)],
    batch: usize,
    max_work: u64,
    max_bytes: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Quote<'a>> {
    if !poll()
        || Instant::now() >= end
        || ranked.is_empty()
        || ranked.len() > MAX_SELECTED
        || batch == 0
        || batch > BATCH
        || target.len() > 1024
        || graph.output_name().len() > 1024
        || bounds.len() > MAX_NODES
        || objective.is_empty()
        || objective.len() > 8192
        || objective.iter().any(|v| !ny_core::is_crown_coeff_safe(*v))
        || !ny_core::is_crown_coeff_safe(old.0)
        || !ny_core::is_crown_coeff_safe(old.1)
        || old.0 > old.1
    {
        return None;
    }
    spatial_source(graph, target)?;
    let base = bounds.get(target)?;
    let shape = base.shape();
    if !matches!(shape, [_, _, _] | [1, _, _, _])
        || base.is_empty()
        || base.len() > MAX_TARGET
        || ranked
            .iter()
            .any(|v| v.0 >= base.len() || !v.1.is_finite() || v.1 <= 0.)
    {
        return None;
    }
    for (i, v) in ranked.iter().enumerate() {
        if ranked[..i].iter().any(|x| x.0 == v.0) {
            return None;
        }
    }
    let limits = ExtractionLimits {
        max_nodes: MAX_NODES,
        max_caller_bytes: max_bytes.min(256 << 20),
    };
    let prefix = quote_charged_raw_bn_chain(
        graph, input, target, bounds, bounds, alpha, limits, end, poll,
    )?;
    let full = quote_charged_raw_bn_chain(
        graph,
        input,
        graph.output_name(),
        bounds,
        bounds,
        alpha,
        limits,
        end,
        poll,
    )?;
    if prefix.output_dim != base.len()
        || prefix.input_dim != input.len()
        || full.output_dim != objective.len()
        || full.input_dim != input.len()
    {
        return None;
    }
    for ceiling in [64, 32, 16, 8, 4, 2, 1] {
        let selected = ranked.len().min(ceiling);
        let (work, bytes) = costs(prefix, full, base.len(), objective.len(), selected, batch)?;
        if work > max_work || bytes > max_bytes {
            continue;
        }
        let mut rows = [0; MAX_SELECTED];
        for (i, v) in ranked[..selected].iter().enumerate() {
            rows[i] = v.0;
        }
        rows[..selected].sort_unstable();
        let score = ranked[..selected].iter().map(|v| v.1).sum::<f64>();
        if !poll() || Instant::now() >= end || !score.is_finite() {
            return None;
        }
        return Some(Quote {
            graph,
            input,
            bounds,
            alpha,
            objective,
            old,
            target,
            rows,
            selected,
            batch,
            prefix,
            full,
            limits,
            end,
            work,
            bytes,
            score,
        });
    }
    None
}

/// Paid bounded ranking. Cache values and scores are never used in a proof.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    row: usize,
    objective_count: usize,
    advice: Advice<'_>,
    engine: &dyn GemmEngine,
    lease: &mut WorkLease,
    max_work: u64,
    max_bytes: usize,
    end: Instant,
) -> Option<Quote<'a>> {
    lease.reserve_more_work(PLAN_WORK).ok()?;
    let live = |lease: &WorkLease| Instant::now() < end && lease.checkpoint().is_ok();
    if !live(lease)
        || graph.nodes.len() > MAX_NODES
        || bounds.len() > MAX_NODES
        || advice.active_indices.len() > MAX_ROWS
    {
        return None;
    }
    for (i, &index) in advice.active_indices.iter().enumerate() {
        if i % 32 == 0 && !live(lease) {
            return None;
        }
        if index >= objective_count || advice.active_indices[..i].contains(&index) {
            return None;
        }
    }
    let cache_row = advice.active_indices.iter().position(|i| *i == row)?;
    let batch = engine
        .as_gpu_crown_backward()?
        .deadline_bounded_resnet_sound_max_rows()
        .min(BATCH);
    if batch == 0 {
        return None;
    }
    let mut names = [None; MAX_NODES];
    for (i, name) in graph.nodes.keys().enumerate() {
        names[i] = Some(name);
    }
    names[..graph.nodes.len()].sort_unstable();
    // Shape/reference metadata only. Price ALL eligible coefficient scans
    // before reading one cache coefficient or comparing affordable quotes.
    // Consequently a later candidate cannot silently invalidate an earlier
    // plan by spending its remaining work allowance.
    let mut candidates: [Option<(&'a str, &'a BoundedTensor, &ndarray::Array2<f32>)>; MAX_NODES] =
        [None; MAX_NODES];
    let mut count = 0;
    let mut scan_work = 0u64;
    for name in names[..graph.nodes.len()].iter().flatten() {
        if !live(lease) || name.len() > 1024 {
            return None;
        }
        let node = graph.node(name)?;
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let target = node.require_unary_input().ok()?;
        if target.len() > 1024 || spatial_source(graph, target).is_none() {
            continue;
        }
        if !has_upstream_relu(graph, target)? {
            continue;
        }
        let base = bounds.get(target)?;
        if !matches!(base.shape(), [_, _, _] | [1, _, _, _]) || base.len() > MAX_TARGET {
            continue;
        }
        let Some(coefficients) = advice.cache.lower_a.get(name.as_str()) else {
            continue;
        };
        if coefficients.nrows() != advice.active_indices.len() || coefficients.ncols() != base.len()
        {
            continue;
        }
        scan_work = scan_work.checked_add(256u64.checked_mul(base.len() as u64)?)?;
        candidates[count] = Some((target, base, coefficients));
        count += 1;
    }
    let remaining = max_work.checked_sub(PLAN_WORK)?.checked_sub(scan_work)?;
    lease.reserve_more_work(scan_work).ok()?;
    let mut best: Option<Quote<'a>> = None;
    for &(target, base, coefficients) in candidates[..count].iter().flatten() {
        if !live(lease) {
            return None;
        }
        let lo = base.lower().as_slice()?;
        let hi = base.upper().as_slice()?;
        let coefficients = coefficients.row(cache_row);
        let coefficients = coefficients.as_slice()?;
        let mut ranked = [(usize::MAX, 0f64); MAX_SELECTED];
        let mut selected = 0;
        for i in 0..base.len() {
            if i % 128 == 0 && !live(lease) {
                return None;
            }
            let (l, u, c) = (lo[i], hi[i], coefficients[i]);
            if !ny_core::is_crown_coeff_safe(l)
                || !ny_core::is_crown_coeff_safe(u)
                || l > u
                || !c.is_finite()
            {
                return None;
            }
            if l >= 0. || u <= 0. || c == 0. {
                continue;
            }
            let score =
                f64::from(c).abs() * (-f64::from(l)) * f64::from(u) / (f64::from(u) - f64::from(l));
            if !score.is_finite() || score <= 0. {
                return None;
            }
            let position = (0..selected)
                .find(|&p| score > ranked[p].1)
                .unwrap_or(selected);
            if position < MAX_SELECTED {
                for j in (position + 1..=selected.min(MAX_SELECTED - 1)).rev() {
                    ranked[j] = ranked[j - 1];
                }
                ranked[position] = (i, score);
                selected = (selected + 1).min(MAX_SELECTED);
            }
        }
        if selected == 0 {
            continue;
        }
        if let Some(candidate) = quote_rows(
            graph,
            input,
            bounds,
            alpha,
            objective,
            old,
            target,
            &ranked[..selected],
            batch,
            remaining,
            max_bytes,
            end,
            &mut || live(lease),
        ) {
            if best.as_ref().is_none_or(|old| {
                candidate.score > old.score
                    || (candidate.score == old.score && candidate.target < old.target)
            }) {
                best = Some(candidate);
            }
        }
    }
    if !live(lease) {
        return None;
    }
    best
}

// Fixed labels and counters only; included in the quoted metadata/work slack.
// This describes reached execution, never a certificate or device-kernel count.
struct ExecutionObservation<'a> {
    target: &'a str,
    selected: usize,
    end: Instant,
    phase: &'static str,
    error_class: &'static str,
    prefix_entered: usize,
    prefix_completed: usize,
    replay_entered: usize,
    replay_completed: usize,
    complete: bool,
}
impl ExecutionObservation<'_> {
    fn result<T>(&mut self, result: ny_core::Result<T>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                self.error_class = if error.is_deadline_exceeded() {
                    "deadline"
                } else if error.is_cpu_memory_exceeded() {
                    "memory"
                } else {
                    "backend"
                };
                None
            }
        }
    }
}
impl Drop for ExecutionObservation<'_> {
    fn drop(&mut self) {
        tracing::info!(
            target = self.target,
            selected_rows = self.selected,
            phase = self.phase,
            error_class = self.error_class,
            prefix_api_entered = self.prefix_entered,
            prefix_api_completed = self.prefix_completed,
            original_replay_entered = self.replay_entered,
            original_replay_completed = self.replay_completed,
            deadline_expired_at_return = Instant::now() >= self.end,
            complete = self.complete,
            "Certified spatial intermediate execution detail"
        );
    }
}

pub(super) struct Completed {
    pub(super) lower: f32,
    pub(super) rows: usize,
    pub(super) prefix_calls: usize,
}

fn seed(rows: &[usize], width: usize, poll: &mut Poll<'_>) -> Option<GpuCrownSeed> {
    let mut a = Vec::new();
    a.try_reserve_exact(rows.len().checked_mul(width)?).ok()?;
    for &coordinate in rows {
        let start = a.len();
        while a.len() < start + width {
            if !poll.check() {
                return None;
            }
            a.resize((a.len() + 4096).min(start + width), 0.);
        }
        a[start + coordinate] = 1.;
    }
    let a = Arc::<[f32]>::from(a);
    let b = Arc::<[f32]>::from(vec![0.; rows.len()]);
    Some(GpuCrownSeed {
        lower_a: a.clone(),
        upper_a: a,
        lower_b: b.clone(),
        upper_b: b,
        num_specs: rows.len(),
        current_dim: width,
    })
}

fn copy_values(values: &[f32], poll: &mut Poll<'_>) -> Option<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(values.len()).ok()?;
    for chunk in values.chunks(4096) {
        if !poll.check() {
            return None;
        }
        out.extend_from_slice(chunk);
    }
    Some(out)
}

/// All refined maps remain private, including on a required replay refusal.
/// No scalar from a prefix, cache, score or incomplete batch can be returned.
pub(super) fn evaluate(
    q: &Quote<'_>,
    gpu: &dyn GpuCrownBackward,
    callback: &mut dyn FnMut() -> bool,
) -> Option<Completed> {
    let mut observation = ExecutionObservation {
        target: q.target,
        selected: q.selected,
        end: q.end,
        phase: "admission",
        error_class: "none",
        prefix_entered: 0,
        prefix_completed: 0,
        replay_entered: 0,
        replay_completed: 0,
        complete: false,
    };
    let mut poll = Poll {
        end: q.end,
        callback,
        alive: true,
    };
    if !poll.check()
        || !gpu.provides_sound_gpu_crown()
        || gpu.deadline_bounded_resnet_sound_max_rows() < q.batch
        || gpu.deadline_bounded_resnet_sound_max_rows() > BATCH
    {
        return None;
    }
    observation.phase = "proof-environment";
    ny_core::require_f64_interval_proof_environment().ok()?;
    let base = q.bounds.get(q.target)?;
    let width = base.len();
    let input_lo = q.input.lower().as_slice()?;
    let input_hi = q.input.upper().as_slice()?;
    let mut lower = [0f32; MAX_SELECTED];
    let mut upper = lower;
    let mut prefix_calls = 0;
    {
        observation.phase = "prefix-source";
        let chain = extract_charged_raw_bn_chain(
            q.graph,
            q.input,
            q.target,
            q.bounds,
            q.bounds,
            q.alpha,
            q.limits,
            q.end,
            &mut || poll.check(),
        )?;
        if chain.quote != q.prefix || chain.output_dim != width || !poll.check() {
            return None;
        }
        for first in (0..q.selected).step_by(q.batch) {
            let last = (first + q.batch).min(q.selected);
            let rows = &q.rows[first..last];
            observation.phase = "prefix-seed";
            let s = seed(rows, width, &mut poll)?;
            if !poll.check() {
                return None;
            }
            observation.phase = "prefix-api";
            observation.prefix_entered += 1;
            let result = observation.result(
                gpu.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                    &chain.segments,
                    &s,
                    input_lo,
                    input_hi,
                    &chain.frontier_abs,
                    &chain.node_abs,
                    q.end,
                ),
            )?;
            if !poll.check()
                || !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&result, rows.len())
            {
                return None;
            }
            observation.phase = "prefix-shrink";
            for i in 0..rows.len() {
                let coordinate = rows[i];
                let l = base.lower().as_slice()?[coordinate].max(result.lower_bounds[i]);
                let u = base.upper().as_slice()?[coordinate].min(result.upper_bounds[i]);
                if l > u {
                    return None;
                }
                lower[first + i] = l;
                upper[first + i] = u;
            }
            prefix_calls += 1;
            observation.prefix_completed += 1;
        }
    } // Drop every prefix descriptor before materializing the full replay.
    observation.phase = "private-target-map";
    let mut lo = copy_values(base.lower().as_slice()?, &mut poll)?;
    let mut hi = copy_values(base.upper().as_slice()?, &mut poll)?;
    for i in 0..q.selected {
        lo[q.rows[i]] = lower[i];
        hi[q.rows[i]] = upper[i];
    }
    if !poll.check() {
        return None;
    }
    let refined = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(base.shape()), lo).ok()?,
        ArrayD::from_shape_vec(IxDyn(base.shape()), hi).ok()?,
    )
    .ok()?;
    if !poll.check() {
        return None;
    }
    let mut overlay = HashMap::new();
    overlay.try_reserve(q.bounds.len()).ok()?;
    for (name, bound) in q.bounds {
        if !poll.check() || name.len() > 1024 {
            return None;
        }
        overlay.insert(
            name.clone(),
            if name == q.target { &refined } else { bound },
        );
    }
    observation.phase = "replay-source";
    let full = extract_charged_raw_bn_chain(
        q.graph,
        q.input,
        q.graph.output_name(),
        &overlay,
        &overlay,
        q.alpha,
        q.limits,
        q.end,
        &mut || poll.check(),
    )?;
    if full.quote != q.full || !poll.check() {
        return None;
    }
    let a = Arc::<[f32]>::from(copy_values(q.objective, &mut poll)?);
    let b = Arc::<[f32]>::from(vec![0.]);
    let seed = GpuCrownSeed {
        lower_a: a.clone(),
        upper_a: a,
        lower_b: b.clone(),
        upper_b: b,
        num_specs: 1,
        current_dim: q.objective.len(),
    };
    observation.phase = "original-replay-api";
    observation.replay_entered += 1;
    let result = observation.result(
        gpu.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
            &full.segments,
            &seed,
            input_lo,
            input_hi,
            &full.frontier_abs,
            &full.node_abs,
            q.end,
        ),
    )?;
    if !poll.check() || !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&result, 1) {
        return None;
    }
    ny_core::require_f64_interval_proof_environment().ok()?;
    observation.phase = "replay-merge";
    let lower = q.old.0.max(result.lower_bounds[0]);
    if !ny_core::is_crown_coeff_safe(lower) || lower > q.old.1 {
        return None;
    }
    observation.replay_completed = 1;
    observation.complete = true;
    observation.phase = "complete";
    Some(Completed {
        lower,
        rows: q.selected,
        prefix_calls,
    })
}

#[cfg(test)]
#[path = "certified_root_spatial_tests.rs"]
mod tests;
