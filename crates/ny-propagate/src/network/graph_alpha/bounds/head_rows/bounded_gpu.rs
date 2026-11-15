// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Same-source selected identity rows through the call-local bounded GPU fold.
//! No ordinary GPU method, cold factory, wider deadline or full-width retry.

use super::{local_bytes, GraphNetwork, RowRefineOutcome, RowRefineRequest, MAX_ROWS};
use crate::network::crown_memory::{process_memory_envelope, ProcessMemoryEnvelope};
use crate::network::raw_root_cut::{
    bounded_fold_bytes, bounded_fold_work, extract_charged_raw_bn_chain,
    quote_charged_raw_bn_chain, ExtractionLimits,
};
use crate::output_margin_seed::scatter_margin_rows_over_bounds;
use ny_core::{GemmEngine, GpuCrownBackward, GpuCrownSeed, NyError};
use std::time::Instant;

// Reuse the bounded root fold's conservative ceilings. Backend dispatch sizes
// and device/transport allocations remain governed by its call-local API.
const MAX_BYTES: usize = 2 << 30;
const MAX_WORK: u64 = 16_000_000_000;

pub(in crate::network::graph_alpha::bounds) struct Observation {
    pub(in crate::network::graph_alpha::bounds) stage: &'static str,
    pub(in crate::network::graph_alpha::bounds) workspace_bytes: usize,
    pub(in crate::network::graph_alpha::bounds) work: u64,
}

impl Default for Observation {
    fn default() -> Self {
        Self {
            stage: "not-requested",
            workspace_bytes: 0,
            work: 0,
        }
    }
}

fn backend(engine: &dyn GemmEngine, rows: usize) -> Option<&dyn GpuCrownBackward> {
    let gpu = engine.as_gpu_crown_backward()?;
    let capacity = gpu.deadline_bounded_resnet_sound_max_rows();
    (gpu.provides_sound_gpu_crown()
        && capacity <= ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
        && rows <= capacity)
        .then_some(gpu)
}

fn fits_memory(bytes: usize, cap: usize, envelope: ProcessMemoryEnvelope) -> bool {
    bytes <= cap.min(MAX_BYTES)
        && match envelope {
            ProcessMemoryEnvelope::Bounded { headroom_bytes } => {
                // Leave half the observed headroom to existing process owners.
                (bytes as u128) * 2 <= u128::from(headroom_bytes)
            }
            ProcessMemoryEnvelope::Unbounded => true,
            ProcessMemoryEnvelope::Unavailable => false,
        }
}

pub(super) struct Attempt {
    pub(super) outcome: RowRefineOutcome,
    pub(super) selected: usize,
    pub(super) rows: [usize; MAX_ROWS],
}

/// `None` means no dispatch: the original host opportunity remains available.
/// `Some` owns this attempt, even when the backend errors or returns bad data.
pub(super) fn try_refine(
    graph: &GraphNetwork,
    request: &RowRefineRequest<'_>,
    ranked: &[usize],
    observation: &mut Observation,
) -> Option<Attempt> {
    if crate::sound_f64_gemm::cpu_only_f64_active() {
        observation.stage = "cpu-only";
        return None;
    }
    let deadline = request.per_node_deadline?;
    let rows = request.rows.len();
    if !request.deadline_is_hard
        || !request.shrink_only
        || request.cut_ctx.is_some()
        || !(1..=MAX_ROWS).contains(&rows)
        || request.base.len() > super::MAX_WIDTH
        || request.rows.windows(2).any(|p| p[0] >= p[1])
        || request.rows.iter().any(|&i| i >= request.base.len())
        || Instant::now() >= deadline
    {
        observation.stage = "request";
        return None;
    }
    let mut identities = [0; MAX_ROWS];
    if ranked.len() != rows {
        return None;
    }
    identities[..rows].copy_from_slice(ranked);
    identities[..rows].sort_unstable();
    if &identities[..rows] != request.rows {
        return None;
    }
    observation.stage = "capability";
    // Observe only ready engines. In particular, do not start/wait for CUDA
    // initialization from a two-second numerical grant.
    let gpu = request
        .engine
        .and_then(|engine| backend(engine, 1))
        .or_else(|| {
            // Default CUDA GEMM need not opt into ordinary CUDA CROWN. The
            // original-root cut route also borrows this ready f64 engine;
            // looking only at resident/wide CROWN slots would keep the new
            // path dark under the shipped default policy.
            let ordinary = crate::sound_f64_gemm::preinitialized_root_cut_engine();
            let [resident, wide] = crate::sound_gpu_gate::preinitialized_root_cut_engines();
            [ordinary, resident, wide]
                .into_iter()
                .flatten()
                .find_map(|engine| backend(engine, 1))
        })?;
    try_with_backend(graph, request, ranked, gpu, observation, deadline)
}

fn try_with_backend(
    graph: &GraphNetwork,
    request: &RowRefineRequest<'_>,
    ranked: &[usize],
    gpu: &dyn GpuCrownBackward,
    observation: &mut Observation,
    deadline: Instant,
) -> Option<Attempt> {
    let started = Instant::now();
    let requested = request.rows.len();
    let width = request.base.len();
    let limits = ExtractionLimits {
        max_nodes: super::MAX_PREFIX_NODES,
        max_caller_bytes: request.dense_budget_bytes.min(256 << 20),
    };
    observation.stage = "quote";
    let mut live = || Instant::now() < deadline;
    let quote = quote_charged_raw_bn_chain(
        graph,
        request.input,
        request.target,
        request.crown_bounds,
        request.ibp_bounds,
        None,
        limits,
        deadline,
        &mut live,
    )?;
    if quote.output_dim != width || quote.input_dim != request.input.len() {
        return None;
    }
    let envelope = process_memory_envelope();
    observation.stage = "resources";
    // Shape-only admission can shrink the batch before any materialization or
    // dispatch. Keep the highest-ranked rows, then restore canonical ordering.
    // A refused eight-row quote must not hide an affordable four-row GPU fold.
    let rows = [8, 4, 2, 1].into_iter().find_map(|ceiling| {
        let count = requested
            .min(ceiling)
            .min(gpu.deadline_bounded_resnet_sound_max_rows());
        let (_, local) = local_bytes(count, width)?;
        let bytes = bounded_fold_bytes(quote, count)?.checked_add(local)?;
        let work = bounded_fold_work(quote, count as u64, 1)?;
        observation.workspace_bytes = bytes;
        observation.work = work;
        (count > 0
            && work <= MAX_WORK
            && fits_memory(bytes, request.dense_budget_bytes, envelope)
            && live())
        .then_some(count)
    })?;
    let mut identities = [0; MAX_ROWS];
    identities[..rows].copy_from_slice(&ranked[..rows]);
    identities[..rows].sort_unstable();
    let attempted = |outcome| {
        Some(Attempt {
            outcome,
            selected: rows,
            rows: identities,
        })
    };
    observation.stage = "source";
    let chain = extract_charged_raw_bn_chain(
        graph,
        request.input,
        request.target,
        request.crown_bounds,
        request.ibp_bounds,
        None,
        limits,
        deadline,
        &mut live,
    )?;
    if chain.quote != quote || !live() {
        return None;
    }
    // Exact one-hot rows in the target's original flattened coordinates. The
    // cap is 8 x 8192; the two sides can share immutable seed storage.
    let mut a = Vec::new();
    a.try_reserve_exact(rows.checked_mul(width)?).ok()?;
    for &coordinate in &identities[..rows] {
        if !live() {
            return None;
        }
        let start = a.len();
        a.resize(start + width, 0.0);
        a[start + coordinate] = 1.0;
    }
    let a = std::sync::Arc::<[f32]>::from(a);
    let bias = std::sync::Arc::<[f32]>::from(vec![0.0; rows]);
    let seed = GpuCrownSeed {
        lower_a: a.clone(),
        upper_a: a,
        lower_b: bias.clone(),
        upper_b: bias,
        num_specs: rows,
        current_dim: width,
    };
    // The extractor already authenticated contiguous finite input endpoints.
    // Borrow them directly: no flattened input or bound-map copy is needed.
    let lower = request.input.lower().as_slice()?;
    let upper = request.input.upper().as_slice()?;
    if !live() {
        return None;
    }
    observation.stage = "dispatched";
    let result = gpu.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &chain.segments,
        &seed,
        lower,
        upper,
        &chain.frontier_abs,
        &chain.node_abs,
        deadline,
    );
    // This check also rejects a backend that returns success after its grant.
    // Once dispatched there is no host retry, partial salvage or new clock.
    if !live() {
        observation.stage = "deadline";
        return attempted(RowRefineOutcome::Failed(NyError::DeadlineExceeded(
            "selected head bounded GPU fold exceeded its original grant".into(),
        )));
    }
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            observation.stage = "backend-error";
            return attempted(RowRefineOutcome::Failed(error));
        }
    };
    if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&result, rows) {
        observation.stage = "payload";
        return attempted(RowRefineOutcome::Declined("bounded-gpu-payload"));
    }
    let scattered = match scatter_margin_rows_over_bounds(
        request.base,
        &identities[..rows],
        &result.lower_bounds,
        &result.upper_bounds,
    ) {
        Ok(bound) => bound,
        Err(error) => return attempted(RowRefineOutcome::Failed(error)),
    };
    // Exactly the host kernel's shrink-only contract; the transaction still
    // performs its selected-only intersection and on-time publication barrier.
    let outcome = match request.base.intersection_per_element(&scattered) {
        Some((bounds, 0)) => RowRefineOutcome::Refined {
            bounds: Box::new(bounds),
            k: rows,
            secs: started.elapsed().as_secs_f64(),
        },
        Some((_, count)) => RowRefineOutcome::Disjoint { count },
        None => RowRefineOutcome::Declined("intersection-unavailable"),
    };
    observation.stage = "returned";
    attempted(outcome)
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "gpu-tests-cuda"))]
mod device_tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Duration;

    #[test]
    fn selected_head_bounded_gpu_cuda_original_rational_bn_source() {
        let engine = ny_cuda::CudaGemmEngine::new().expect("explicit CUDA conformance needs CUDA");
        for rows in [1, 8] {
            let (graph, input, bounds) = tests::source(rows);
            let before = ny_cuda::cuda_deadline_f64_gemm_stats();
            let end = Instant::now() + Duration::from_secs(30);
            let mut transaction = super::super::run(
                &graph,
                &input,
                &bounds,
                &HashSet::from(["bn".into()]),
                Some(&engine),
                end,
                32 << 20,
                true,
            )
            .expect("original selected opportunity");
            assert_eq!(transaction.gpu.stage, "returned");
            assert_eq!(transaction.stage, "committed");
            let base = &bounds["bn"];
            let (published, count) = transaction.publish(base, base, Some(end));
            assert_eq!(count, rows);
            for i in 0..rows {
                let lo = f64::from(published.lower()[[i]]);
                let hi = f64::from(published.upper()[[i]]);
                // Exact real source (|x|-3/2)/3. Both extrema are attained.
                assert!(lo <= -0.5 && lo > -0.501, "row {i}: {lo}");
                assert!((-1.0 / 6.0..-0.166).contains(&hi), "row {i}: {hi}");
            }
            assert_eq!(
                published.lower()[[rows]].to_bits(),
                base.lower()[[rows]].to_bits()
            );
            assert_eq!(
                published.upper()[[rows]].to_bits(),
                base.upper()[[rows]].to_bits()
            );
            let after = ny_cuda::cuda_deadline_f64_gemm_stats();
            assert!(after.calls > before.calls && after.dispatches > before.dispatches);
            eprintln!("selected-head-cuda rows={rows} grant={} seconds={} lower={} upper={} calls={} dispatches={}",
                transaction.grant_secs, transaction.elapsed_secs,
                published.lower()[[0]], published.upper()[[0]],
                after.calls - before.calls, after.dispatches - before.dispatches);
        }
    }
}
