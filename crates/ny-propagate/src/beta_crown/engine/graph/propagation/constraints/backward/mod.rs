// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared backward CROWN core for constrained graph propagation.
//!
//! Extracted from the two near-duplicate backward passes in `constraints/mod.rs`:
//! - `propagate_crown_with_graph_constraints` (standard path)
//! - `propagate_crown_with_graph_constraints_storing_intermediates` (gradient path)
//!
//! Both share ~90% of their match arms. This module unifies them behind a single
//! traversal function parameterized by `BackwardMode`, eliminating drift risk.
//!
//! Part of #1813 (wave 2 dedup).
//! Split into directory module by #4293.

mod dispatch;
mod finalize;
mod linear;
mod relu;
mod setup;

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::domain::GraphCrownContext;
use crate::bounds::GraphAlphaCrownIntermediate;
use crate::network::CrownMergeAccumulator;
use crate::{GraphNetwork, LinearBounds, NETWORK_INPUT};

use super::super::super::super::BetaCrownVerifier;
use super::lookups::ConstraintLookups;
use super::patches::ConstrainedPatchesPolicy;

/// Controls whether the backward pass stores intermediate A-matrices for gradient computation.
pub(in crate::beta_crown::engine::graph::propagation) enum BackwardMode {
    /// Standard backward CROWN: no intermediate storage, includes debug diagnostics.
    Standard,
    /// Stores A matrices at constrained ReLU nodes for analytical gradient computation.
    /// Requires constraint lookups to identify which ReLU nodes to capture.
    StoringIntermediates { lookups: Box<ConstraintLookups> },
}

/// Groups the immutable graph parameters for constrained backward CROWN.
pub(in crate::beta_crown::engine::graph::propagation) struct BackwardParams<'a> {
    pub(in crate::beta_crown::engine::graph::propagation) graph: &'a GraphNetwork,
    pub(in crate::beta_crown::engine::graph::propagation) constrained_input: &'a BoundedTensor,
    pub(in crate::beta_crown::engine::graph::propagation) exec_order: &'a [String],
    pub(in crate::beta_crown::engine::graph::propagation) context: &'a GraphCrownContext<'a>,
    pub(in crate::beta_crown::engine::graph::propagation) beta_state:
        Option<&'a crate::beta_crown::state::GraphBetaState>,
    pub(in crate::beta_crown::engine::graph::propagation) objective: Option<&'a [f32]>,
    /// Multi-row spec matrix for batched spec-guided CROWN (#4306).
    /// When set, seeds the backward pass with an (N, D) matrix instead of
    /// identity or a single-row objective. Takes precedence over `objective`.
    pub(in crate::beta_crown::engine::graph::propagation) spec_matrix:
        Option<&'a ndarray::Array2<f32>>,
    pub(in crate::beta_crown::engine::graph::propagation) seed_cache:
        Option<&'a CachedLinearBounds>,
    pub(in crate::beta_crown::engine::graph::propagation) capture_linear_bounds: bool,
    /// Per-node deadline for intra-kernel timeout enforcement (#3795).
    /// When set, the DispatchContext carries this deadline so expensive
    /// backward kernels (e.g., Conv2d) can bail early.
    pub(in crate::beta_crown::engine::graph::propagation) deadline: Option<std::time::Instant>,
    pub(in crate::beta_crown::engine::graph::propagation) patches_policy: ConstrainedPatchesPolicy,
}

/// Full result of the constrained backward CROWN pass including concretization.
pub(in crate::beta_crown::engine::graph::propagation) struct BackwardCrownResult {
    /// Concretized output bounds.
    pub(in crate::beta_crown::engine::graph::propagation) output_bounds: BoundedTensor,
    /// Intermediate storage (populated only in `StoringIntermediates` mode).
    pub(in crate::beta_crown::engine::graph::propagation) intermediate:
        Option<GraphAlphaCrownIntermediate>,
    /// Full cached lA coefficients captured during the backward pass.
    pub(in crate::beta_crown::engine::graph::propagation) captured_la: Option<CachedLinearBounds>,
}

struct ConstrainedBackwardState {
    node_crown_bounds: CrownMergeAccumulator,
    intermediate: Option<GraphAlphaCrownIntermediate>,
    captured_linear_bounds: Option<HashMap<String, LinearBounds>>,
    input_accumulated: bool,
}

struct ConstrainedBackwardSetup<'graph, 'mode> {
    output_node: &'graph str,
    output_shape: Vec<usize>,
    output_dim: usize,
    input_dim: usize,
    mode_lookups: Option<&'mode ConstraintLookups>,
    state: ConstrainedBackwardState,
}

fn resolve_pre_activation<'a>(
    first_input: &str,
    constrained_input: &'a BoundedTensor,
    bounds_cache: &'a HashMap<String, Arc<BoundedTensor>>,
) -> Result<&'a BoundedTensor> {
    if first_input == NETWORK_INPUT {
        Ok(constrained_input)
    } else {
        bounds_cache
            .get(first_input)
            .map(|a| a.as_ref())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for {} not found",
                    first_input
                ))
            })
    }
}

impl BetaCrownVerifier {
    /// Shared backward CROWN pass for constrained graph propagation.
    ///
    /// Traverses the graph in reverse topological order, propagating linear bounds
    /// backward through each layer, then concretizes and applies cut planes.
    /// The `mode` parameter controls whether intermediate A-matrices are stored
    /// at constrained ReLU nodes for gradient computation.
    ///
    /// Both `propagate_crown_with_graph_constraints` and
    /// `propagate_crown_with_graph_constraints_storing_intermediates` delegate to this.
    pub(in crate::beta_crown::engine::graph::propagation) fn backward_crown_constrained(
        &self,
        params: &BackwardParams<'_>,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        mode: BackwardMode,
    ) -> Result<BackwardCrownResult> {
        // #unsat-keystone step 4: GPU beta-capable resnet fast-path. This is the SHARED
        // per-domain backward funnel — the BaB root bound, child bounds, and multi-
        // objective spec-guided passes all reach here — so one injection covers them all,
        // batched over the spec_matrix rows. Default ON (opt out NY_RESNET_BETA_GPU=0),
        // sound (β≥0 is a valid Lagrangian dual + the GPU resnet backward is a sound
        // enclosure), CPU fallback on any miss → the 0-wrong moat holds. Standard mode
        // only (no intermediate/lA capture on this path).
        if matches!(mode, BackwardMode::Standard) {
            if let Some(output_bounds) =
                self.try_gpu_beta_constrained_backward(params, bounds_cache_mut)
            {
                return Ok(BackwardCrownResult {
                    output_bounds,
                    intermediate: None,
                    captured_la: None,
                });
            }
        }
        let is_standard = matches!(mode, BackwardMode::Standard);
        let mut setup = self.initialize_constrained_backward(params, &mode, &*bounds_cache_mut)?;

        for node_name in params.exec_order.iter().rev() {
            // Per-node deadline check: a single constrained backward pass through a
            // deep residual conv net (e.g. TinyImageNet ResNet) can be expensive, and
            // the per-domain check in the BaB loop only fires between domains. Without
            // this, one slow domain's backward overruns the wall-clock budget badly
            // (observed 23 min on a 90s budget once binary/residual child propagation
            // was enabled). Bailing here yields a sound Timeout/unresolved domain.
            if params
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                return Err(NyError::DeadlineExceeded(
                    "constrained backward CROWN: deadline exceeded".to_string(),
                ));
            }
            if let Some(result) = self.process_constrained_backward_node(
                params,
                is_standard,
                node_name,
                bounds_cache_mut,
                &mut setup,
            )? {
                return Ok(result);
            }
        }

        self.finalize_constrained_backward(params, is_standard, bounds_cache_mut, setup)
    }

    /// GPU beta-capable resnet backward for the constrained (per-domain) bound
    /// (#unsat-keystone step 4). Seeds from the `spec_matrix` (batched over its N rows)
    /// or a single `objective`, decomposes the output suffix (alpha=None — default ReLU
    /// slopes from the CONSTRAINED `bounds_cache` already reflect the splits), folds the
    /// per-ReLU β-CROWN dual from `beta_state`, and runs the sound GPU resnet backward.
    /// Returns `Some(output_bounds)` (shape `[N]`) when applicable, else `None` → caller
    /// runs the proven CPU constrained backward. Gated + sound (β≥0 valid dual + sound
    /// GPU enclosure); default ON, opt out `NY_RESNET_BETA_GPU=0`.
    fn try_gpu_beta_constrained_backward(
        &self,
        params: &BackwardParams<'_>,
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    ) -> Option<BoundedTensor> {
        if !crate::network::resnet_beta_gpu_enabled() {
            return None;
        }
        let gpu = params
            .context
            .engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())?;
        // ReLU-only splits (the additive ±β term is the ReLU split_point=0 dual).
        if !params.context.history.genbab_constraints.is_empty() {
            return None;
        }
        let graph = params.graph;
        if !graph
            .nodes
            .values()
            .any(|n| matches!(n.layer, crate::layers::Layer::Conv2d(_)))
        {
            return None;
        }
        // Seed: spec_matrix (N×D, batched) preferred; else a single objective (1×D).
        // Skip the identity-seed (intermediate-node) case — left to CPU.
        let (seed_vec, n_specs, od): (Vec<f32>, usize, usize) = if let Some(m) = params.spec_matrix
        {
            (m.iter().copied().collect(), m.nrows(), m.ncols())
        } else {
            let o = params.objective?;
            (o.to_vec(), 1, o.len())
        };
        if od == 0 || n_specs == 0 || n_specs > 512 {
            return None;
        }
        let probe = std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1");
        let (segments, relu_names, frontier_abs, node_abs) =
            crate::network::extract_gpu_resnet_segments_with_relu_names(
                graph,
                params.constrained_input,
                &graph.output_node,
                bounds_cache,
                bounds_cache,
                None,
            )?;
        let mut beta_signed: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
        for name in &relu_names {
            let nn = bounds_cache.get(name)?.lower().len();
            let mut bs = vec![0.0f32; nn];
            if let Some(beta) = params.beta_state {
                for entry in beta.entries_for_node(name) {
                    if entry.split_point().abs() < 1e-6 {
                        let idx = entry.neuron_idx();
                        if idx < nn {
                            bs[idx] = entry.signed_value();
                        }
                    }
                }
            }
            beta_signed.push(bs);
        }
        let seed = ny_core::GpuCrownSeed {
            lower_a: seed_vec.clone().into(),
            upper_a: seed_vec.into(),
            lower_b: vec![0.0f32; n_specs].into(),
            upper_b: vec![0.0f32; n_specs].into(),
            num_specs: n_specs,
            current_dim: od,
        };
        let in_lo: Vec<f32> = params.constrained_input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = params.constrained_input.upper().iter().copied().collect();
        let result = gpu
            .crown_backward_gpu_resnet_sound_beta(
                &segments,
                &seed,
                &in_lo,
                &in_hi,
                &beta_signed,
                &frontier_abs,
                &node_abs,
            )
            .ok()?;
        if result.lower_bounds.len() != n_specs
            || result.upper_bounds.len() != n_specs
            || result
                .lower_bounds
                .iter()
                .chain(result.upper_bounds.iter())
                .any(|v| v.is_nan())
        {
            return None;
        }
        if probe {
            eprintln!(
                "[beta-gpu-funnel] SUCCESS n_specs={n_specs} relus={} od={od}",
                relu_names.len()
            );
        }
        let lower =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n_specs]), result.lower_bounds)
                .ok()?;
        let upper =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n_specs]), result.upper_bounds)
                .ok()?;
        BoundedTensor::new_repaired(lower, upper, ny_tensor::RepairStrategy::Widen).ok()
    }
}
