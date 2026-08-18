// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared domain types, priority ordering, and helper functions for
//! both single-objective and multi-objective input-split verifiers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use super::bounds_eval::{
    build_input_split_reference_bounds, compute_plain_crown_or_ibp_bounds_with_node_bounds,
};
#[cfg(test)]
use crate::beta_crown::engine::{
    graph::domain_batch::{DenseSpecBatchRequest, GraphDomainBatchExecutor},
    tensor_ext::BoundedTensorExt,
};
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::GraphNetwork;

pub(crate) use super::bounds_eval::{
    graph_crown_error_should_fallback, graph_ibp_prescreen_error_should_skip,
    graph_output_bounds_are_finite, graph_spec_crown_with_mul_binary_and_truncation,
    graph_spec_ibp_fallback, graph_spec_ibp_root_screen_with_deadline,
    try_graph_spec_ibp_prescreen_bounds,
};

#[derive(Debug, Clone)]
pub(super) struct GraphInputDomain {
    pub(super) input_bounds: Arc<BoundedTensor>,
    pub(super) lower_bound: f32,
    pub(super) upper_bound: f32,
    pub(super) depth: usize,
    pub(super) priority: f32,
    pub(super) linear_bounds: Option<LinearBounds>,
    /// True if this domain needs CROWN bounding before it can be split.
    /// Used in reorder_bab mode: children are enqueued without bounds and
    /// bounded when popped. Reference: alpha-beta-CROWN reorder_bab.
    pub(super) needs_bounding: bool,
    /// Child-local node-bounds override carried by complete clipping until the
    /// deferred CROWN pass consumes it.
    pub(super) node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
    /// Per-sub-domain refined α slopes inherited from the parent (warm-start).
    ///
    /// Only populated when `input_split_alpha_iteration > 0` (the lightweight
    /// per-sub-domain α-refinement path, mirroring alpha-beta-CROWN's input-split
    /// BaB). When set, a child warm-starts its α optimization from these slopes
    /// instead of the frozen root α state. `None` = ny's historical behavior
    /// (frozen root α threaded into every per-domain pass). Wrapped in `Arc` so
    /// attaching the same parent slopes to multiple children is cheap.
    ///
    /// SOUND: any α in [0,1] yields a valid CROWN over-approximation, and a
    /// parent's slopes are a feasible start for a child whose box is a SUBSET of
    /// the parent's; `warm_start` re-clamps to [0,1] regardless.
    pub(super) inherited_alpha_state: Option<Arc<GraphAlphaState>>,
}

pub(super) fn cmp_input_domain_priority(lhs: f32, rhs: f32) -> std::cmp::Ordering {
    match (lhs.is_nan(), rhs.is_nan()) {
        // Surface invalid domains immediately instead of treating them as arbitrary equals.
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (true, true) => std::cmp::Ordering::Equal,
        (false, false) => lhs.total_cmp(&rhs),
    }
}

impl PartialEq for GraphInputDomain {
    fn eq(&self, other: &Self) -> bool {
        cmp_input_domain_priority(self.priority, other.priority) == std::cmp::Ordering::Equal
    }
}

impl Eq for GraphInputDomain {}

impl PartialOrd for GraphInputDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphInputDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority = pop first
        cmp_input_domain_priority(self.priority, other.priority)
    }
}

/// Batched scalar objective bounds for single-objective input-split paths.
///
/// This carrier intentionally stores one scalar lower/upper pair per domain.
/// Multi-row spec matrices need a different result shape because collapsing a
/// dense spec-guided tensor through `lower_scalar()`/`upper_scalar()` loses the
/// per-objective semantics.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct BatchedScalarBounds {
    pub(crate) lower_bounds: Vec<f32>,
    pub(crate) upper_bounds: Vec<f32>,
    pub(crate) linear_bounds: Vec<Option<LinearBounds>>,
}

#[cfg(test)]
#[path = "shared_tests.rs"]
mod tests;

/// Compute spec-guided CROWN bounds with IBP fallback for a sub-domain.
///
/// Tries CROWN with alpha/MulBinary alpha state; falls back to IBP on recoverable
/// errors. Used by both single-objective and multi-objective verifier loops.
///
/// When `ibp_enhancement` is true, runs a cheap IBP forward pass on the subdomain
/// first, builds a reference-bounds map from the IBP/inherited bounds, and then
/// still runs a fresh spec-guided CROWN pass that tightens against those
/// references. This matches alpha-beta-CROWN's `reference_bounds` contract
/// instead of freezing IBP as the active intermediate-bounds map. Part of #3870.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input_bounds,
        spec_matrix,
        engine,
        alpha_node_bounds,
        None,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        ibp_enhancement,
    )
}

/// Like `compute_crown_or_ibp_bounds`, but allows a child-local node-bounds
/// override to replace the shared root alpha/IBP cache.
///
/// When `ibp_enhancement` is true, builds a reference-bounds map from the
/// inherited alpha/clip bounds plus current-domain IBP, then runs fresh CROWN
/// with that reference map so nonlinear relaxations see the tighter of
/// `{fresh CROWN, reference}` at each node. Part of #3870.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds_with_node_bounds(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    child_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    let fixed_node_bounds = child_node_bounds.or(alpha_node_bounds);
    if !ibp_enhancement {
        return compute_plain_crown_or_ibp_bounds_with_node_bounds(
            graph,
            input_bounds,
            spec_matrix,
            engine,
            fixed_node_bounds,
            alpha_state,
            mul_binary_alphas,
            deadline,
            crown_backward_layers,
        );
    }

    let cached_fallback_bounds = match build_input_split_reference_bounds(
        alpha_node_bounds,
        child_node_bounds,
        None,
    ) {
        Ok(bounds) => bounds,
        Err(err) => {
            tracing::debug!(
                    "IBP-enhanced reference-bounds merge failed before fresh IBP; retrying plain input-split bounds: {err}"
                );
            return compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            );
        }
    };
    let ibp_reference_bounds =
        // Keep the IBP reference-bounds pass under the caller deadline budget;
        // if it expires, avoid routing through an uncapped collection first (#4207, #4208).
        match graph.collect_node_bounds_with_engine_and_deadline(input_bounds, engine, deadline) {
            Ok(bounds) => Some(bounds),
            Err(NyError::DeadlineExceeded(_)) => {
                if let Some(bounds) = cached_fallback_bounds.as_ref() {
                    tracing::debug!(
                        "IBP-enhanced reference-bounds exceeded deadline, reusing cached node bounds for plain IBP fallback"
                    );
                    return graph_spec_ibp_fallback(
                        graph,
                        input_bounds,
                        spec_matrix,
                        engine,
                        Some(bounds),
                    );
                }
                tracing::debug!(
                    "IBP-enhanced reference-bounds exceeded deadline with no cached node bounds; returning conservative spec bounds"
                );
                return Ok((
                    BoundedTensor::new_conservative(&[spec_matrix.nrows()]),
                    None,
                ));
            }
            Err(err) => {
                tracing::debug!(
                    "IBP-enhanced reference-bounds setup failed; retrying plain input-split bounds: {err}"
                );
                return compute_plain_crown_or_ibp_bounds_with_node_bounds(
                    graph,
                    input_bounds,
                    spec_matrix,
                    engine,
                    fixed_node_bounds,
                    alpha_state,
                    mul_binary_alphas,
                    deadline,
                    crown_backward_layers,
                );
            }
        };
    let reference_node_bounds = match build_input_split_reference_bounds(
        alpha_node_bounds,
        child_node_bounds,
        ibp_reference_bounds.as_ref(),
    ) {
        Ok(bounds) => bounds,
        Err(err) => {
            tracing::debug!(
                "IBP-enhanced reference-bounds merge with fresh IBP failed; retrying plain input-split bounds: {err}"
            );
            return compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            );
        }
    };

    let crown_result = graph_spec_crown_with_mul_binary_and_truncation(
        graph,
        input_bounds,
        spec_matrix,
        engine,
        None,
        reference_node_bounds.as_ref(),
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
    );
    match crown_result {
        Ok(result) if graph_output_bounds_are_finite(&result.0) => Ok(result),
        Ok((_bounds, _linear)) => {
            tracing::debug!(
                "IBP-enhanced spec-guided CROWN produced non-finite bounds; retrying plain input-split bounds"
            );
            compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            )
        }
        Err(NyError::DeadlineExceeded(_)) => {
            tracing::debug!(
                "IBP-enhanced spec-guided CROWN exceeded deadline, falling back to IBP"
            );
            let ibp_result = graph_spec_ibp_fallback(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                reference_node_bounds.as_ref(),
            )?;
            if graph_output_bounds_are_finite(&ibp_result.0) {
                Ok(ibp_result)
            } else {
                tracing::debug!(
                    "IBP-enhanced deadline fallback produced non-finite bounds; retrying plain input-split bounds"
                );
                compute_plain_crown_or_ibp_bounds_with_node_bounds(
                    graph,
                    input_bounds,
                    spec_matrix,
                    engine,
                    fixed_node_bounds,
                    alpha_state,
                    mul_binary_alphas,
                    deadline,
                    crown_backward_layers,
                )
            }
        }
        Err(err) => {
            tracing::debug!(
                "IBP-enhanced spec-guided CROWN failed; retrying plain input-split bounds: {err}"
            );
            compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            )
        }
    }
}

/// Lightweight per-sub-domain α refinement for the single-objective input-split
/// BaB loop, mirroring alpha-beta-CROWN's `input_split/bounding.py:90-179`.
///
/// Each sub-domain warm-starts from its parent's optimized alphas and re-optimizes
/// them for `config.input_split_alpha_iteration` SPSA iterations at
/// `lr = config.input_split_lr_alpha`, with `fix_interm_bounds = true` (skipping
/// the O(N²) intermediate CROWN pass). Only a handful of iterations are needed
/// because the parent slopes are already near-optimal for the child's tighter box.
///
/// Returns the spec-guided objective bounds + linear bounds (computed with the
/// REFINED alphas and refined intermediate bounds) AND the refined `GraphAlphaState`
/// so it can be saved onto BOTH child sub-domains as their warm-start seed.
///
/// SOUND: `collect_alpha_crown_bounds_dag_warm_with_engine` keeps every α in [0,1]
/// (so every per-domain bound is a valid CROWN over-approximation), and the final
/// spec-guided pass is the same sound CROWN-with-IBP-fallback used by the frozen
/// path — only the α values and the intermediate-bound map differ. This helper
/// propagates optimization or bound failures; each caller decides whether to
/// retry its historical frozen-alpha computation or fail closed. Callers must
/// only invoke this when `input_split_alpha_iteration > 0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_warm_start_crown_bounds_with_refined_alpha(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    child_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    parent_alpha: &GraphAlphaState,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    config: &crate::beta_crown::config::BetaCrownConfig,
) -> Result<(BoundedTensor, Option<LinearBounds>, GraphAlphaState)> {
    let alpha_config = warm_start_alpha_config(config, deadline);

    // Step 1: warm-start + re-optimize alphas for this sub-domain's tighter box.
    let (refined_node_bounds, refined_alpha) = graph
        .collect_alpha_crown_bounds_dag_warm_with_engine(
            input_bounds,
            &alpha_config,
            parent_alpha,
            engine,
        )?;

    // Step 2: spec-guided objective pass with the REFINED alphas + refined
    // intermediate bounds. A child-local node-bounds override (e.g. from complete
    // clipping) still takes precedence over the refined map, exactly as in the
    // frozen path. Reuses the same sound CROWN-with-IBP-fallback dispatch.
    let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input_bounds,
        spec_matrix,
        engine,
        Some(&refined_node_bounds),
        child_node_bounds,
        Some(&refined_alpha),
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        config.input_split_ibp_enhancement,
    )?;

    Ok((bounds, linear, refined_alpha))
}

/// Derive the deliberately lightweight child-domain alpha configuration.
///
/// Keeping this as one pure helper makes the root/child routing boundary
/// executable in tests: a root may opt into the cGAN target-complete collector,
/// but no inherited runtime flag may make a BaB child pay that root-only cost.
fn warm_start_alpha_config(
    config: &crate::beta_crown::config::BetaCrownConfig,
    deadline: Option<Instant>,
) -> crate::AlphaCrownConfig {
    let mut alpha_config = config.alpha_config.clone();
    alpha_config.iterations = config.input_split_alpha_iteration;
    alpha_config.learning_rate = config.input_split_lr_alpha;
    // Skip the O(N²) post-optimization intermediate CROWN pass: the warm-start
    // path returns the (sound) IBP intermediate bounds directly. Matches the
    // reference `fix_interm_bounds=True` for input-split refinement.
    alpha_config.fix_interm_bounds = true;
    alpha_config.cgan_sparse_target_complete_root = false;
    alpha_config.cgan_complete_crown_ibp_root = false;
    alpha_config.deadline = deadline;
    alpha_config
}

/// Compute CROWN/IBP bounds for a single-objective batch.
/// Enforces `spec_matrix.nrows() == 1` and extracts scalar lower/upper pairs.
/// Part of #3870 Phase 2. Refactored in #4116 Packet A to delegate to dense-spec helper.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds_batched(
    graph: &GraphNetwork,
    input_bounds_batch: &[&BoundedTensor],
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
) -> Result<BatchedScalarBounds> {
    if spec_matrix.nrows() != 1 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "compute_crown_or_ibp_bounds_batched only supports single-row spec matrices; got {} rows",
            spec_matrix.nrows()
        )));
    }

    let spec_result = GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
        graph,
        input_bounds_batch,
        spec_matrix,
        engine,
        alpha_node_bounds,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        ibp_enhancement,
        stacked_rebound: false,
    })?;

    let n = spec_result.bounds.len();
    let mut lower_bounds = Vec::with_capacity(n);
    let mut upper_bounds = Vec::with_capacity(n);

    for bounds in &spec_result.bounds {
        lower_bounds.push(bounds.lower_scalar());
        upper_bounds.push(bounds.upper_scalar());
    }

    Ok(BatchedScalarBounds {
        lower_bounds,
        upper_bounds,
        linear_bounds: spec_result.linear_bounds,
    })
}

/// Conjunctive domain check: verified if ANY objective has `lower > threshold`.
/// The proof-relevant lower endpoint and threshold must be finite, while a
/// `+inf` upper endpoint remains a valid one-sided enclosure. NaN and inverted
/// intervals never acquire proof authority. Part of #3646.
pub(super) fn multi_obj_domain_verified(obj_bounds: &[(f32, f32)], thresholds: &[f32]) -> bool {
    if obj_bounds.is_empty() || obj_bounds.len() != thresholds.len() {
        return false;
    }
    obj_bounds
        .iter()
        .zip(thresholds.iter())
        .any(|(&(lower, upper), &threshold)| {
            super::grouped_semantics::objective_interval_verified(lower, upper, threshold)
        })
}

/// Conjunctive domain priority: use the closest-to-verified objective.
/// NaN bounds yield NEG_INFINITY priority. Part of #3646.
pub(super) fn multi_obj_domain_priority(obj_bounds: &[(f32, f32)], thresholds: &[f32]) -> f32 {
    if obj_bounds.is_empty() || obj_bounds.len() != thresholds.len() {
        return f32::NEG_INFINITY;
    }
    obj_bounds
        .iter()
        .zip(thresholds.iter())
        .map(|(&(l, u), &t)| {
            if !l.is_finite() || !u.is_finite() || !t.is_finite() || l > u {
                return f32::NEG_INFINITY;
            }
            let gap = l - t;
            if gap.is_finite() {
                gap
            } else {
                f32::NEG_INFINITY
            }
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Extract per-objective bounds from a multi-row BoundedTensor.
///
/// Returns an error unless the tensor contains exactly one scalar interval per
/// objective row. Exact layout prevents trailing or non-contiguous storage
/// from being silently ignored by verdict code.
pub(crate) fn extract_obj_bounds(
    bounds: &BoundedTensor,
    num_specs: usize,
) -> Result<Vec<(f32, f32)>> {
    let lower = bounds.lower();
    let upper = bounds.upper();
    let n_lower = lower.len();
    let n_upper = upper.len();
    if num_specs == 0 || num_specs != n_lower || num_specs != n_upper {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_specs],
            got: vec![n_lower.max(n_upper)],
        });
    }
    let lower_flat = lower.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("objective lower bounds must be contiguous".to_string())
    })?;
    let upper_flat = upper.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("objective upper bounds must be contiguous".to_string())
    })?;
    let mut extracted = Vec::with_capacity(num_specs);
    for i in 0..num_specs {
        let (lower, upper) = (lower_flat[i], upper_flat[i]);
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(NyError::NumericalInstability(format!(
                "objective row {i} is malformed: lower={lower}, upper={upper}"
            )));
        }
        extracted.push((lower, upper));
    }
    Ok(extracted)
}

/// Build a child BoundedTensor from flat lower/upper arrays reshaped to `shape`.
pub(super) fn build_child_input(
    child_lower: &ndarray::ArrayD<f32>,
    child_upper: &ndarray::ArrayD<f32>,
    shape: &[usize],
) -> Result<BoundedTensor> {
    let lower_arr = child_lower
        .clone()
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower: {}", e)))?;
    let upper_arr = child_upper
        .clone()
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper: {}", e)))?;
    BoundedTensor::new(lower_arr, upper_arr)
}

/// Build a child BoundedTensor by consuming owned flat lower/upper arrays.
///
/// Identical result to `build_child_input`, but takes the arrays by value so a
/// caller that already owns the freshly-split boxes (and does not reuse them)
/// avoids two full-box clones per child. `into_shape_clone` only reallocates
/// when the array is not contiguous in the requested order; the input-split
/// boxes originate from `flatten()` (standard layout) so this is the same
/// bitwise reshape with no extra copy. The element values are untouched, so the
/// resulting bounds are bit-for-bit identical to the borrowing path.
pub(super) fn build_child_input_owned(
    child_lower: ndarray::ArrayD<f32>,
    child_upper: ndarray::ArrayD<f32>,
    shape: &[usize],
) -> Result<BoundedTensor> {
    let lower_arr = child_lower
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower: {}", e)))?;
    let upper_arr = child_upper
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper: {}", e)))?;
    BoundedTensor::new(lower_arr, upper_arr)
}

pub(super) fn scalar_output_bounds(lower: f32, upper: f32) -> Result<BoundedTensor> {
    BoundedTensor::new(
        Array1::from_vec(vec![lower]).into_dyn(),
        Array1::from_vec(vec![upper]).into_dyn(),
    )
}

/// Multi-dimensional midpoint input split (mirrors `create_input_split_children`):
/// midpoint-splits each dim in `split_dims`, producing up to 2^split_dims.len() child
/// boxes that EXACTLY COVER the parent box — every parent point lies in some child, so
/// BaB completeness (hence soundness of the verdict) is preserved. A dim that is out of
/// range, non-finite, or non-positive-width is left unsplit (the box passes through
/// whole). Boxes are flat, matching `build_child_input`'s expectation.
///
/// Shared by the single-objective and multi-objective conjunctive input-split loops
/// (moved from `single_objective/process_batch.rs` so both honor `input_split_depth`).
pub(super) fn multi_dim_split_boxes(
    flat_lower: ndarray::ArrayD<f32>,
    flat_upper: ndarray::ArrayD<f32>,
    split_dims: &[usize],
) -> Vec<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> {
    // The caller hands us the freshly-flattened root box by value, so it becomes
    // the initial box directly — no extra clone of the parent's lower/upper.
    let len = flat_lower.len();
    let mut boxes = vec![(flat_lower, flat_upper)];
    for &dim in split_dims {
        if dim >= len {
            continue;
        }
        let mut next = Vec::with_capacity(boxes.len() * 2);
        for (lo, hi) in boxes {
            let l = lo[[dim]];
            let u = hi[[dim]];
            if !l.is_finite() || !u.is_finite() || u <= l {
                next.push((lo, hi));
                continue;
            }
            let mid = l + (u - l) / 2.0;
            let mut left_hi = hi.clone();
            left_hi[[dim]] = mid;
            next.push((lo.clone(), left_hi));
            let mut right_lo = lo;
            right_lo[[dim]] = mid;
            next.push((right_lo, hi));
        }
        boxes = next;
    }
    boxes
}

pub(super) use super::multi_obj_domain::{MultiObjBounds, MultiObjInputDomain};
