// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP (Interval Bound Propagation) methods for GraphNetwork.
//!
//! This module contains IBP propagation methods extracted from the GraphNetwork impl.
//! Extraction is incremental - methods are moved here as they are refactored.

use crate::layers::Layer;
use crate::network::core::graph::ibp::dispatch::{
    check_nan_firewall, check_nan_firewall_with_poll, dispatch_ibp_resolved,
    intersect_zonotope_ibp, intersect_zonotope_ibp_with_poll, resolve_node_inputs, ResolvedInputs,
};
use crate::network::core::graph::ibp::gpu_plan::try_lower_graph_dag;

use ndarray::IxDyn;
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::debug;

use super::core::GraphNetwork;

/// Extension trait for IBP propagation on graph networks.
///
/// Contains `propagate_ibp_impl` and `propagate_ibp_sound_impl`.
/// Additional methods will be migrated incrementally per
/// designs/2026-01-28-oversized-file-splits-remaining.md.
pub(crate) trait GraphNetworkIbpExt {
    /// Propagate bounds through the graph using IBP.
    fn propagate_ibp_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Propagate bounds through the graph using IBP with an optional GEMM engine.
    fn propagate_ibp_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Propagate bounds through the graph using IBP with an optional GEMM engine,
    /// aborting with `DeadlineExceeded` between nodes and cooperatively within
    /// deadline-aware convolution nodes once the deadline passes.
    fn propagate_ibp_with_engine_and_deadline_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor>;

    /// Propagate bounds through the graph using IBP while preserving a leading axis.
    fn propagate_ibp_with_engine_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Propagate bounds through the graph using IBP with a certified per-node
    /// widening, so floating-point rounding cannot shrink a node box below the
    /// true range.
    ///
    /// The widening applied depends on how the node accumulates:
    /// - Linear (CERTIFIED): `in_features + 2` ULPs (Higham Thm 3.1 over the two
    ///   matmuls, the combine and the bias).
    /// - Conv1d / ConvTranspose1d (CERTIFIED): the node's ordinary result hulled
    ///   with a bit-decoded, outward-directed f64 interval contraction. This
    ///   covers cancellation and binary32 product underflow independently of
    ///   FTZ/DAZ.
    /// - Conv2d / ConvTranspose2d (CERTIFIED): the node's own
    ///   `propagate_ibp_sound_with_engine`, which folds in its certified
    ///   coefficient-abssum/Higham error for the window sum.
    /// - AveragePool (CERTIFIED): `AveragePoolLayer::propagate_ibp_sound`, which
    ///   folds in the certified `γ⁶⁴_{k+1}·S/d` Higham term for its `k`-term f64
    ///   window sum (uniform `+1/k` weights). The plain forward's outward 1-ULP
    ///   store covers only the f64→f32 cast; the f64 accumulation residual can
    ///   exceed 1 f32 ULP of the result under ≥2^29 cancellation.
    /// - Every other node (ASSUMPTION, not a certificate): 1 ULP on top of its own
    ///   IBP. Valid for ops exact in f32 (ReLU/clip/abs/neg, Flatten/Reshape/
    ///   Concat/transpose, MaxPool's exact max), for single-rounding ops whose one
    ///   nearest-rounding is ≤ half-ULP (binary Add/Sub/Mul/Div, constant
    ///   arithmetic), for ops that already round their endpoints outward themselves
    ///   (e.g. BatchNorm), and for pointwise transcendentals (Exp, Log, Sigmoid,
    ///   Tanh, GELU, Softplus, Erf, ...) ONLY IF the platform libm is faithfully
    ///   rounded (≤ 1 ULP) — an assumption, not a proof. Ops on this arm that
    ///   ACCUMULATE across many terms (Softmax/LogSumExp denominators, LayerNorm/
    ///   RMSNorm/GroupNorm statistics, ReduceSum/ReduceMean, CumSum, binary
    ///   MatMul, Bilinear) are NOT certified: their rounding residual can exceed
    ///   1 ULP under the same cancellation. Certifying them is an open item
    ///   (#sound-ibp-generic-arm); AveragePool was the last accumulator with a
    ///   trivial (uniform-coefficient) certificate.
    fn propagate_ibp_sound_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Propagate SOUND (directed-rounding) bounds with an optional GEMM engine.
    ///
    /// Same certified per-node widening as [`Self::propagate_ibp_sound_impl`], but
    /// threads an `engine` so a DAG-lowerable graph can take the GPU-resident SOUND
    /// DAG plan when the soundness gate is engaged and the engine advertises one
    /// (`provides_sound_gpu_dag_ibp`). Every other case — gate off, no engine, an
    /// un-lowerable graph, or any GPU error — falls through to the proven-sound CPU
    /// graph loop, so a verdict is never decided by a failed or unsound GPU op.
    fn propagate_ibp_sound_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// True concrete (point) forward; collapses each node output to its center.
    fn propagate_concrete_point_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor>;

    /// Point forward (as above) preserving a prepended leading restart axis.
    fn propagate_concrete_point_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeadingAxisMode {
    Plain,
    PreserveLeadingAxis,
}

const GRAPH_IBP_POLL_ELEMENTS: usize = 4_096;

#[inline]
fn check_graph_ibp_deadline(deadline: Instant, node_name: &str, stage: &str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "Graph IBP forward: deadline exceeded {stage} for node '{node_name}'"
        )));
    }
    Ok(())
}

fn summarize_bounds(bounds: &BoundedTensor) -> (f32, f32, bool, bool, bool) {
    const MAX_BOUND: f32 = f32::MAX / 2.0;
    let mut max_width = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut saturated = false;
    let mut has_nan = false;
    let mut has_non_finite = false;

    for (&lower, &upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        if lower.is_nan() || upper.is_nan() {
            has_nan = true;
        }
        if !lower.is_finite() || !upper.is_finite() {
            has_non_finite = true;
        }
        let width = upper - lower;
        if width.is_finite() {
            max_width = max_width.max(width);
        } else {
            has_non_finite = true;
        }
        max_abs = max_abs.max(lower.abs()).max(upper.abs());
        if lower <= -0.999 * MAX_BOUND || upper >= 0.999 * MAX_BOUND {
            saturated = true;
        }
    }

    (max_width, max_abs, saturated, has_nan, has_non_finite)
}

fn summarize_bounds_with_poll<F>(
    bounds: &BoundedTensor,
    mut poll: F,
) -> Result<(f32, f32, bool, bool, bool)>
where
    F: FnMut() -> Result<()>,
{
    const MAX_BOUND: f32 = f32::MAX / 2.0;
    let mut max_width = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut saturated = false;
    let mut has_nan = false;
    let mut has_non_finite = false;

    poll()?;
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        if index.is_multiple_of(GRAPH_IBP_POLL_ELEMENTS) {
            poll()?;
        }
        if lower.is_nan() || upper.is_nan() {
            has_nan = true;
        }
        if !lower.is_finite() || !upper.is_finite() {
            has_non_finite = true;
        }
        let width = upper - lower;
        if width.is_finite() {
            max_width = max_width.max(width);
        } else {
            has_non_finite = true;
        }
        max_abs = max_abs.max(lower.abs()).max(upper.abs());
        if lower <= -0.999 * MAX_BOUND || upper >= 0.999 * MAX_BOUND {
            saturated = true;
        }
    }
    poll()?;

    Ok((max_width, max_abs, saturated, has_nan, has_non_finite))
}

fn concrete_center_with_deadline(
    bounds: &BoundedTensor,
    deadline: Option<Instant>,
    node_name: &str,
    stage: &str,
) -> Result<BoundedTensor> {
    if let Some(deadline) = deadline {
        let center =
            bounds.center_with_poll(|| check_graph_ibp_deadline(deadline, node_name, stage))?;
        BoundedTensor::concrete_with_poll(center, || {
            check_graph_ibp_deadline(deadline, node_name, stage)
        })
    } else {
        BoundedTensor::concrete(bounds.center())
    }
}

impl GraphNetworkIbpExt for GraphNetwork {
    #[inline]
    fn propagate_ibp_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        propagate_ibp_core(self, input, false, None, LeadingAxisMode::Plain, None)
    }

    #[inline]
    fn propagate_ibp_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        propagate_ibp_core(self, input, false, engine, LeadingAxisMode::Plain, None)
    }

    #[inline]
    fn propagate_ibp_with_engine_and_deadline_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        propagate_ibp_core(self, input, false, engine, LeadingAxisMode::Plain, deadline)
    }

    #[inline]
    fn propagate_ibp_with_engine_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        propagate_ibp_core(
            self,
            input,
            false,
            engine,
            LeadingAxisMode::PreserveLeadingAxis,
            None,
        )
    }

    #[inline]
    fn propagate_ibp_sound_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        propagate_ibp_core(self, input, true, None, LeadingAxisMode::Plain, None)
    }

    #[inline]
    fn propagate_ibp_sound_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        propagate_ibp_core(self, input, true, engine, LeadingAxisMode::Plain, None)
    }

    #[inline]
    fn propagate_concrete_point_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        propagate_concrete_point_core(self, input, engine, LeadingAxisMode::Plain, deadline)
    }

    #[inline]
    fn propagate_concrete_point_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        propagate_concrete_point_core(
            self,
            input,
            engine,
            LeadingAxisMode::PreserveLeadingAxis,
            None,
        )
    }
}

/// Shared IBP propagation for graph networks.
///
/// When `sound` is true, applies each node's certified widening (see
/// `propagate_ibp_sound_impl` for the per-node rule) so floating-point
/// errors cannot cause unsound bounds.
fn propagate_ibp_core(
    network: &GraphNetwork,
    input: &BoundedTensor,
    sound: bool,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    propagate_ibp_core_inner(
        network,
        input,
        sound,
        engine,
        leading_axis_mode,
        deadline,
        false,
    )
}

/// True concrete (point) forward through the DAG: collapse every node's output to
/// its interval center before caching, so per-node soundness widening (esp.
/// BatchNorm) cannot be amplified by the rest of the graph (#cgan-eval). NON-
/// soundness-critical — for sat-finding / witness evaluation only.
fn propagate_concrete_point_core(
    network: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    // Start from the box center (callers pass a degenerate box; a point forward is
    // only defined at the center).
    let point = concrete_center_with_deadline(
        input,
        deadline,
        "<network input>",
        "while centering the concrete input",
    )?;

    // DAG-lowerable fast path — keep the cached-plan / GPU-resident hot path (#4276).
    // `try_lower_graph_dag` accepts ONLY affine + ReLU ops (Linear, Conv2d, Add,
    // Flatten, Reshape, AveragePool, ReLU) and fail-closes on every other op —
    // crucially on the widening ops this routine exists to tame (BatchNorm, and the
    // directed-rounding sound path). So a DAG-lowerable graph applies NO soundness
    // widening in the non-sound forward: a point input stays degenerate
    // (lower == upper) through every node, and the box forward's center IS the true
    // network value (to ~ULP, matching ORT). We can therefore reuse the regular box
    // forward — which dispatches to the GPU-resident DAG cached plan when the engine
    // supports it (and the naive per-node CPU loop otherwise) — and take the
    // (degenerate) center, instead of forcing the slower per-node center-collapse
    // loop that bypasses the cached plan and regresses the adv_check / BaB hot path.
    // Widening graphs (BatchNorm — the cgan_2023 generators) are NOT DAG-lowerable
    // and fall through to the per-node collapse below, where re-centering after each
    // node is genuinely required to stop the deep DAG from amplifying the widening
    // (#cgan-eval).
    if leading_axis_mode == LeadingAxisMode::Plain
        && try_lower_graph_dag(network, point.shape()).is_some()
    {
        let out = propagate_ibp_core_inner(
            network,
            &point,
            false,
            engine,
            leading_axis_mode,
            deadline,
            false,
        )?;
        return concrete_center_with_deadline(
            &out,
            deadline,
            "<network output>",
            "while centering the concrete output",
        );
    }

    propagate_ibp_core_inner(
        network,
        &point,
        false,
        engine,
        leading_axis_mode,
        deadline,
        true,
    )
}

/// Execute a prepared GPU-resident DAG IBP plan and wrap its output as a
/// `BoundedTensor`. Returns `Ok(None)` when the input cannot be presented as
/// contiguous flat slices of the expected length (the caller then falls back to the
/// CPU loop). A GPU execution error surfaces as `Err` from the plan; the caller
/// treats it as a fall-through, never a verdict.
fn execute_dag_plan_to_tensor(
    cached_plan: &dyn ny_core::GpuDagIbpModelPlan,
    input: &BoundedTensor,
) -> Result<Option<BoundedTensor>> {
    let input_elements = checked_shape_product(input.shape()).unwrap_or(0);
    let (Some(lower_slice), Some(upper_slice)) =
        (input.lower().as_slice(), input.upper().as_slice())
    else {
        return Ok(None);
    };
    if lower_slice.len() != input_elements || upper_slice.len() != input_elements {
        return Ok(None);
    }
    let result = cached_plan.dag_ibp_forward_cached(lower_slice, upper_slice, input.shape())?;
    let lower_arr =
        ndarray::ArrayD::from_shape_vec(IxDyn(&result.output_shape), result.lower_bounds)
            .map_err(|e| NyError::InvalidSpec(format!("sound dag_ibp lower shape: {e}")))?;
    let upper_arr =
        ndarray::ArrayD::from_shape_vec(IxDyn(&result.output_shape), result.upper_bounds)
            .map_err(|e| NyError::InvalidSpec(format!("sound dag_ibp upper shape: {e}")))?;
    Ok(Some(BoundedTensor::new(lower_arr, upper_arr)?))
}

#[allow(clippy::too_many_arguments)]
fn propagate_ibp_core_inner(
    network: &GraphNetwork,
    input: &BoundedTensor,
    sound: bool,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
    deadline: Option<Instant>,
    collapse_to_center: bool,
) -> Result<BoundedTensor> {
    if network.nodes.is_empty() {
        return Ok(input.clone());
    }

    // SOUND GPU-resident DAG fast path (verdict-legal; `docs/SOUND_GPU_IBP_PLAN.md`
    // T1.0). Mirrors the sequential `propagate_ibp_sound_with_engine`: only the
    // CERTIFIED sound DAG plan may serve a `sound` request, and only when the
    // soundness gate is engaged (`gpu_dag_ibp_forward_route` → `use_sound`). The
    // fast (unsound) plan is NEVER routed into a sound bound. Any miss — gate off,
    // no engine, an un-lowerable graph, a plan `None`, or a GPU error — falls
    // through to the proven-sound CPU graph loop below, so the 0-wrong moat holds.
    // Each sound op emits `[low − r_lo, high + r_hi] ⊇` its predecessors' truth, so
    // by induction over topological order the returned box ⊇ both the true range and
    // the CPU sound bound.
    if deadline.is_none()
        && !collapse_to_center
        && sound
        && leading_axis_mode == LeadingAxisMode::Plain
    {
        if let Some((ext, use_sound)) = crate::sound_gpu_gate::gpu_dag_ibp_forward_route(engine) {
            if use_sound {
                if let Some(plan_desc) = try_lower_graph_dag(network, input.shape()) {
                    match ext.prepare_sound_dag_model_plan(&plan_desc) {
                        Ok(Some(cached_plan)) => {
                            match execute_dag_plan_to_tensor(cached_plan.as_ref(), input) {
                                Ok(Some(tensor)) => {
                                    debug!(
                                        "GraphNetwork IBP: GPU SOUND DAG resident path succeeded \
                                         (output shape {:?})",
                                        tensor.shape()
                                    );
                                    return Ok(tensor);
                                }
                                Ok(None) => {
                                    debug!(
                                        "GraphNetwork IBP: sound DAG input not contiguous, \
                                         CPU sound fallback"
                                    );
                                }
                                Err(e) => {
                                    debug!(
                                        "GraphNetwork IBP: sound DAG execution failed, \
                                         CPU sound fallback: {e}"
                                    );
                                }
                            }
                        }
                        Ok(None) => {
                            debug!(
                                "GraphNetwork IBP: sound DAG plan returned None, CPU sound fallback"
                            );
                        }
                        Err(e) => {
                            debug!(
                                "GraphNetwork IBP: sound DAG plan preparation failed, \
                                 CPU sound fallback: {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    // Attempt GPU-resident DAG fast path (#4319).
    // Only for non-sound, plain leading axis mode (the common hot path).
    // Falls back silently to the CPU graph loop on None or any error.
    // Skipped entirely when collapsing to center: the fast path returns a single
    // whole-box result with no per-node hook to re-center.
    if deadline.is_none()
        && !collapse_to_center
        && !sound
        && leading_axis_mode == LeadingAxisMode::Plain
    {
        if let Some(engine) = engine {
            if let Some(ext) = engine.as_gpu_dag_ibp_forward_ext() {
                if let Some(plan_desc) = try_lower_graph_dag(network, input.shape()) {
                    match ext.prepare_dag_model_plan(&plan_desc) {
                        Ok(Some(cached_plan)) => {
                            let input_elements = checked_shape_product(input.shape()).unwrap_or(0);
                            if let (Some(lower_slice), Some(upper_slice)) =
                                (input.lower().as_slice(), input.upper().as_slice())
                            {
                                if lower_slice.len() == input_elements
                                    && upper_slice.len() == input_elements
                                {
                                    match cached_plan.dag_ibp_forward_cached(
                                        lower_slice,
                                        upper_slice,
                                        input.shape(),
                                    ) {
                                        Ok(result) => {
                                            debug!(
                                                "GraphNetwork IBP: GPU DAG resident path \
                                                 succeeded (output shape {:?})",
                                                result.output_shape
                                            );
                                            let lower_arr = ndarray::ArrayD::from_shape_vec(
                                                IxDyn(&result.output_shape),
                                                result.lower_bounds,
                                            )
                                            .map_err(|e| {
                                                NyError::InvalidSpec(format!(
                                                    "dag_ibp lower shape: {e}"
                                                ))
                                            })?;
                                            let upper_arr = ndarray::ArrayD::from_shape_vec(
                                                IxDyn(&result.output_shape),
                                                result.upper_bounds,
                                            )
                                            .map_err(|e| {
                                                NyError::InvalidSpec(format!(
                                                    "dag_ibp upper shape: {e}"
                                                ))
                                            })?;
                                            return BoundedTensor::new(lower_arr, upper_arr);
                                        }
                                        Err(e) => {
                                            debug!(
                                                "GraphNetwork IBP: GPU DAG execution failed, \
                                                 falling back to CPU: {e}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        Ok(None) => {
                            debug!("GraphNetwork IBP: GPU DAG plan returned None, CPU fallback");
                        }
                        Err(e) => {
                            debug!(
                                "GraphNetwork IBP: GPU DAG plan preparation failed, \
                                 CPU fallback: {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    // Get execution order
    let exec_order = network.exec_order()?;

    // Taint-gated degrade (#cctsdb never-hard-error): nodes forward-reachable
    // from an OpaqueSkip already carry no information beyond [-inf, +inf], so
    // when bound computation at such a node fails structurally (shape
    // mismatch from a skipped op's unknown output shape, etc.) we substitute
    // conservative unbounded bounds of the declared shape instead of aborting
    // the whole pass. Soundness: [-inf, +inf] over-approximates any op output.
    // Errors at UNtainted nodes still abort (they indicate real bugs), as does
    // DeadlineExceeded. NY_STRICT_IBP=1 restores the abort-on-error behavior
    // (degrade is default-on per the batteries-included principle).
    let strict_ibp = std::env::var_os("NY_STRICT_IBP").is_some_and(|v| v == "1");
    let tainted: std::collections::HashSet<&str> = if !strict_ibp
        && network
            .nodes
            .values()
            .any(|n| matches!(n.layer, Layer::OpaqueSkip(_)))
    {
        let mut tainted: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for node_name in exec_order {
            let Some(node) = network.nodes.get(node_name) else {
                continue;
            };
            if matches!(node.layer, Layer::OpaqueSkip(_))
                || node
                    .inputs
                    .iter()
                    .any(|input_name| tainted.contains(input_name.as_str()))
            {
                tainted.insert(node_name.as_str());
            }
        }
        tainted
    } else {
        std::collections::HashSet::new()
    };

    /// Conservative unbounded bounds of the given shape.
    fn unbounded_of_shape(shape: &[usize]) -> Result<BoundedTensor> {
        let lower = ndarray::ArrayD::from_elem(IxDyn(shape), f32::NEG_INFINITY);
        let upper = ndarray::ArrayD::from_elem(IxDyn(shape), f32::INFINITY);
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    /// Preserve the historical allocation/validation path without a deadline;
    /// finite-deadline callers initialize each endpoint in bounded chunks.
    fn unbounded_of_shape_with_deadline(
        shape: &[usize],
        deadline: Option<Instant>,
        node_name: &str,
        stage: &str,
    ) -> Result<BoundedTensor> {
        if let Some(deadline) = deadline {
            BoundedTensor::new_conservative_with_poll(shape, || {
                check_graph_ibp_deadline(deadline, node_name, stage)
            })
        } else {
            unbounded_of_shape(shape)
        }
    }

    /// Errors that the tainted-node degrade path may recover from.
    /// DeadlineExceeded and numerical errors at untainted nodes must abort.
    fn is_degradable_error(error: &NyError) -> bool {
        matches!(
            error,
            NyError::ShapeMismatch { .. } | NyError::InvalidSpec(_) | NyError::UnsupportedOp(_)
        )
    }

    // Store bounds for each node's output
    let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
        std::collections::HashMap::new();

    // Process nodes in topological order
    for node_name in exec_order {
        // Wall-clock deadline enforcement between nodes (#4321), supplemented by
        // cooperative polling inside deadline-aware convolution nodes. Aborting
        // here lets the caller surface a graceful Timeout/Unknown. Sound: only
        // ever short-circuits to an error.
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "Graph IBP forward: deadline exceeded before node '{}'",
                node_name
            )));
        }
        let node = network
            .nodes
            .get(node_name)
            .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

        debug!(
            "GraphNetwork IBP{}: processing node {} ({})",
            if sound { " (sound)" } else { "" },
            node_name,
            node.layer.layer_type()
        );

        // OpaqueSkip shape mismatch logging (unique to this dispatch site).
        if matches!(&node.layer, Layer::OpaqueSkip(_)) && node.inputs.len() > 1 {
            // This branch only runs for multi-input OpaqueSkip, so `require_unary_input`
            // (which rejects >1 input) would always error here. Use the first declared
            // input directly for the diagnostic shape comparison (#2666 regression).
            let first_input_name = node.inputs[0].as_str();
            let first_shape = network
                .bounds_ref(first_input_name, input, &bounds_cache)?
                .shape()
                .to_vec();
            let has_mismatch = node.inputs[1..].iter().any(|input_name| {
                network
                    .bounds_ref(input_name, input, &bounds_cache)
                    .map(|b| b.shape() != first_shape.as_slice())
                    .unwrap_or(false)
            });
            if has_mismatch {
                debug!(
                    "GraphNetwork IBP: OpaqueSkip {} has mismatched input shapes; using first input shape",
                    node_name
                );
            }
        }

        let resolved = resolve_node_inputs(node, node_name, &mut |name| {
            Ok(network.bounds_ref(name, input, &bounds_cache)?.clone())
        })?;
        if std::env::var_os("NY_TRACE_IBP_SHAPES").is_some() {
            let input_shapes: Vec<Vec<usize>> = match &resolved {
                ResolvedInputs::Unary(bounds) => vec![bounds.shape().to_vec()],
                ResolvedInputs::Binary(a, b) => vec![a.shape().to_vec(), b.shape().to_vec()],
                ResolvedInputs::Ternary(a, b, c) => {
                    vec![a.shape().to_vec(), b.shape().to_vec(), c.shape().to_vec()]
                }
                ResolvedInputs::Nary(inputs) => inputs
                    .iter()
                    .map(|bounds| bounds.shape().to_vec())
                    .collect(),
            };
            eprintln!(
                "IBP node '{}' ({}) inputs {:?}",
                node_name,
                node.layer.layer_type(),
                input_shapes
            );
        }

        let node_is_tainted = tainted.contains(node_name.as_str());

        // Shape-as-value hazard (#cctsdb A2): index-producing ops read their
        // input's SHAPE as part of the output VALUE (e.g. ArgMax index range
        // = [0, axis_len-1]). Under taint the input's shape may itself be a
        // conservative substitute (wrong axis length), so a computed FINITE
        // index interval could be unsound. Degrade these to unbounded
        // outright when tainted; [-inf, +inf] encloses every index.
        let force_shape_value_degrade = node_is_tainted
            && matches!(
                node.layer,
                Layer::ArgMax(_)
                    | Layer::ArgMin(_)
                    | Layer::ArgSort(_)
                    | Layer::Topk(_)
                    | Layer::NonZero(_)
            );

        let computed: Result<BoundedTensor> = if force_shape_value_degrade
            && network.declared_shape(node_name).is_some()
        {
            tracing::warn!(
                "GraphNetwork IBP: node '{}' ({}) is tainted by an upstream OpaqueSkip and \
                 reads input shape as value; substituting declared-shape unbounded bounds",
                node_name,
                node.layer.layer_type()
            );
            unbounded_of_shape_with_deadline(
                network
                    .declared_shape(node_name)
                    .expect("checked is_some above"),
                deadline,
                node_name,
                "while creating a tainted shape-value fallback",
            )
        } else {
            let computed = (|| -> Result<BoundedTensor> {
                let output_bounds = match resolved {
                    ResolvedInputs::Unary(input_bounds) => match (&node.layer, leading_axis_mode) {
                        (Layer::Linear(linear), _) => linear
                            .propagate_ibp_with_engine_and_deadline(
                                &input_bounds,
                                engine,
                                deadline,
                            )?,
                        // Conv family sound path: the f32 window sum needs each
                        // layer's own certificate, NOT the generic 1-ULP
                        // widening below. The 1D variants use a directed-f64
                        // interval contraction; Conv2d uses the certified f64
                        // dual-accumulator kernel under a finite deadline
                        // (#cgan-conv-ibp-magnitude-floor) and its
                        // coefficient-abssum construction otherwise;
                        // ConvTranspose2d keeps its abssum construction.
                        (Layer::Conv1d(conv), _) => {
                            if sound || deadline.is_some() {
                                conv.propagate_ibp_sound_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            } else {
                                conv.propagate_ibp_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            }
                        }
                        (Layer::Conv2d(conv), _) => {
                            if sound || deadline.is_some() {
                                // A finite-deadline graph IBP result can still
                                // feed verifier root decisions. Plain f32
                                // Conv2d under-encloses under cancellation, so
                                // the deadline route must carry a certificate
                                // as well as avoiding opaque engine/faer work.
                                // Under a finite deadline that certificate is
                                // the f64 dual-accumulator kernel — strictly
                                // tighter than the old gamma*S widening whose
                                // magnitude-scaled floor stopped cgan BaB trees
                                // from closing (#cgan-conv-ibp-magnitude-floor).
                                // `deadline=None && !sound` retains the
                                // historical fast path.
                                conv.propagate_ibp_sound_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            } else {
                                conv.propagate_ibp_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            }
                        }
                        (Layer::ConvTranspose1d(conv), _) => {
                            if sound || deadline.is_some() {
                                conv.propagate_ibp_sound_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            } else {
                                conv.propagate_ibp_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            }
                        }
                        (Layer::ConvTranspose2d(conv), _) => {
                            if sound || deadline.is_some() {
                                // A finite-deadline graph result can feed root
                                // decisions, so use the certified directed-f64
                                // route and never enter the unpolled legacy
                                // transpose forward under that authority.
                                conv.propagate_ibp_sound_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            } else {
                                conv.propagate_ibp_with_engine_and_deadline(
                                    &input_bounds,
                                    engine,
                                    deadline,
                                )?
                            }
                        }
                        // AveragePool sound path: the plain forward accumulates each
                        // window sum in f64 and directed-rounds only the final f64→f32
                        // store, which certifies the cast but NOT the f64 accumulation
                        // residual — under ≥2^29 cancellation that residual exceeds the
                        // generic 1-ULP widening below. The sound forward folds the
                        // certified γ⁶⁴_{k+1}·S/d Higham term outward; see
                        // `AveragePoolLayer::propagate_ibp_sound` (#avgpool-1ulp-arm).
                        // Non-sound requests fall through to the generic dispatch.
                        (Layer::AveragePool(pool), _) if sound => {
                            pool.propagate_ibp_sound(&input_bounds)?
                        }
                        (Layer::Reshape(layer), LeadingAxisMode::PreserveLeadingAxis) => {
                            layer.propagate_ibp_preserve_leading_axis(&input_bounds)?
                        }
                        (Layer::Flatten(layer), LeadingAxisMode::PreserveLeadingAxis) => {
                            layer.propagate_ibp_preserve_leading_axis(&input_bounds)?
                        }
                        (Layer::Softmax(layer), LeadingAxisMode::PreserveLeadingAxis) => {
                            layer.propagate_ibp_preserve_leading_axis(&input_bounds)?
                        }
                        (Layer::LogSoftmax(layer), LeadingAxisMode::PreserveLeadingAxis) => {
                            layer.propagate_ibp_preserve_leading_axis(&input_bounds)?
                        }
                        _ => dispatch_ibp_resolved(
                            node,
                            node_name,
                            ResolvedInputs::Unary(input_bounds),
                        )?,
                    },
                    ResolvedInputs::Binary(ref input_a, ref input_b) => {
                        // Zonotope tightening for attention MatMul and SwiGLU MulBinary.
                        // Matches detailed.rs and graph_alpha/bounds/ibp.rs (#2706).
                        match &node.layer {
                            Layer::MatMul(matmul) if matmul.transpose_b => {
                                if let Some(tighter) = network
                                    .try_attention_matmul_bounds_zonotope(
                                        node,
                                        input,
                                        &bounds_cache,
                                    )?
                                {
                                    tighter
                                } else {
                                    node.layer.propagate_ibp_binary(input_a, input_b)?
                                }
                            }
                            Layer::MulBinary(_) => {
                                let ibp = node.layer.propagate_ibp_binary(input_a, input_b)?;
                                match network.try_ffn_swiglu_bounds_zonotope(
                                    node,
                                    input,
                                    &bounds_cache,
                                )? {
                                    // The zonotope and plain-IBP results are BOTH sound
                                    // over-approximations; keep the per-element tighter
                                    // of the two (intersection). On some kernels (large
                                    // RMSNorm-induced base width) the zonotope's coarse
                                    // scale-normalized quadratic mul is looser than plain
                                    // IBP, so intersecting prevents a regression while
                                    // still capturing the correlation gains where the
                                    // zonotope wins. Soundness: intersection of two valid
                                    // bounds is valid (#swiglu-intersect).
                                    Some(zono) => {
                                        if let Some(deadline) = deadline {
                                            intersect_zonotope_ibp_with_poll(zono, ibp, || {
                                                check_graph_ibp_deadline(
                                                    deadline,
                                                    node_name,
                                                    "while intersecting zonotope bounds",
                                                )
                                            })?
                                        } else {
                                            intersect_zonotope_ibp(zono, ibp)
                                        }
                                    }
                                    None => ibp,
                                }
                            }
                            _ => node.layer.propagate_ibp_binary(input_a, input_b)?,
                        }
                    }
                    other => dispatch_ibp_resolved(node, node_name, other)?,
                };

                // DFL / expectation-decode tightening: when this Linear/MatMul node
                // contracts a row-stochastic Softmax output against constant weights
                // (along the softmax axis), each output element is a convex combination
                // of the contracted constants and provably lies in their [min, max]
                // range. Intersect the term-wise IBP interval (which drops the simplex
                // constraint and over-counts) with that envelope. Intersection only
                // tightens — never widens — so this is sound; it is a no-op (`None`)
                // whenever the producer/structure is not a constant-weighted softmax
                // contraction. See `ibp::dfl_envelope`.
                let output_bounds = if matches!(&node.layer, Layer::Linear(_) | Layer::MatMul(_)) {
                    if let Some(deadline) = deadline {
                        // The optional DFL recognizer/envelope currently scans,
                        // allocates, and (for the perturbed-MatMul case) sorts
                        // without a cooperative polling contract. Refuse that
                        // tightening under finite authority; the untightened
                        // IBP box remains sound. Keep the `None` arm byte-for-byte
                        // on the historical helper path below.
                        check_graph_ibp_deadline(
                            deadline,
                            node_name,
                            "before refusing unpolled DFL simplex-envelope postprocessing",
                        )?;
                        debug!(
                            "GraphNetwork IBP: node '{}' finite deadline; skipping optional \
                             unpolled DFL simplex-envelope postprocessing",
                            node_name
                        );
                        output_bounds
                    } else {
                        match network.try_dfl_simplex_envelope(
                            node,
                            &output_bounds,
                            input,
                            &bounds_cache,
                        )? {
                            Some(tightened) => tightened,
                            None => output_bounds,
                        }
                    }
                } else {
                    output_bounds
                };
                Ok(output_bounds)
            })();
            if force_shape_value_degrade {
                // No declared shape: keep the computed shape but drop the
                // (potentially unsound under taint) finite index values.
                tracing::warn!(
                    "GraphNetwork IBP: node '{}' ({}) is tainted and reads input shape as \
                     value; dropping computed values for unbounded bounds",
                    node_name,
                    node.layer.layer_type()
                );
                computed.and_then(|bounds| {
                    unbounded_of_shape_with_deadline(
                        bounds.shape(),
                        deadline,
                        node_name,
                        "while creating a tainted shape-value fallback",
                    )
                })
            } else {
                computed
            }
        };

        let output_bounds = match computed {
            Ok(bounds) => bounds,
            Err(e) if node_is_tainted && is_degradable_error(&e) => {
                // Taint-gated degrade (#cctsdb A2): this node is downstream of
                // an OpaqueSkip, so the failure stems from a conservative
                // substitution (unknown skipped-op output shape), not a real
                // propagation bug. Substitute declared-shape unbounded bounds
                // — a sound over-approximation of any op output — and keep
                // going so the pass reaches the network output.
                tracing::warn!(
                    "GraphNetwork IBP: degrading tainted node '{}' ({}) to unbounded bounds \
                     after error: {} (downstream of OpaqueSkip; set NY_STRICT_IBP=1 to abort)",
                    node_name,
                    node.layer.layer_type(),
                    e
                );
                let shape: Vec<usize> = match network.declared_shape(node_name) {
                    Some(shape) => shape.to_vec(),
                    None => match node.inputs.first() {
                        Some(first) => network
                            .bounds_ref(first, input, &bounds_cache)?
                            .shape()
                            .to_vec(),
                        // No declared shape and no inputs: cannot shape a
                        // substitute; propagate the original error.
                        None => return Err(e),
                    },
                };
                unbounded_of_shape_with_deadline(
                    &shape,
                    deadline,
                    node_name,
                    "while creating a tainted error fallback",
                )?
            }
            Err(e) => return Err(e),
        };

        // Apply directed rounding for soundness: lower bounds down, upper bounds up.
        // Linear layers use n-ULP rounding proportional to dot product size; all other
        // layers use 1-ULP rounding. The conv family and AveragePool already carry
        // their own certified rounding enclosures from the sound forward above, so
        // their extra ULP here is redundant but harmless (widening only ever loosens).
        let mut output_bounds = output_bounds;
        if sound {
            match &node.layer {
                Layer::Linear(linear) => {
                    let rounding_ulps = u32::try_from(linear.in_features())
                        .unwrap_or(u32::MAX)
                        .saturating_add(2);
                    if let Some(deadline) = deadline {
                        output_bounds.round_for_soundness_n_ulps_inplace_with_poll(
                            rounding_ulps,
                            || {
                                check_graph_ibp_deadline(
                                    deadline,
                                    node_name,
                                    "while applying soundness rounding",
                                )
                            },
                        )?;
                    } else {
                        output_bounds.round_for_soundness_n_ulps_inplace(rounding_ulps);
                    }
                }
                _ => {
                    if let Some(deadline) = deadline {
                        output_bounds.round_for_soundness_inplace_with_poll(|| {
                            check_graph_ibp_deadline(
                                deadline,
                                node_name,
                                "while applying soundness rounding",
                            )
                        })?;
                    } else {
                        output_bounds.round_for_soundness_inplace();
                    }
                }
            }
        }

        let (max_width, max_abs, saturated, has_nan, has_non_finite) =
            if let Some(deadline) = deadline {
                summarize_bounds_with_poll(&output_bounds, || {
                    check_graph_ibp_deadline(deadline, node_name, "while summarizing output bounds")
                })?
            } else {
                summarize_bounds(&output_bounds)
            };
        debug!(
            "GraphNetwork IBP: {} ({}) shape {:?} max_width {:.2e} max_abs {:.2e} saturated={} nan={} non_finite={}",
            node_name,
            node.layer.layer_type(),
            output_bounds.shape(),
            max_width,
            max_abs,
            saturated,
            has_nan,
            has_non_finite
        );
        if saturated || has_nan || has_non_finite {
            debug!(
                "GraphNetwork IBP: WARNING: bounds degraded at {} ({})",
                node_name,
                node.layer.layer_type()
            );
        }
        // Tainted nodes legitimately produce NaN from [-inf, +inf] interval
        // arithmetic (e.g. 0 * inf in MulBinary). Substitute the sound
        // unbounded interval instead of tripping the NaN firewall; NaN at an
        // UNtainted node still aborts below (real numerical bug).
        let output_bounds = if has_nan && node_is_tainted {
            tracing::warn!(
                "GraphNetwork IBP: tainted node '{}' ({}) produced NaN bounds from \
                 unbounded-interval arithmetic; substituting unbounded bounds",
                node_name,
                node.layer.layer_type()
            );
            let shape = network
                .declared_shape(node_name)
                .unwrap_or_else(|| output_bounds.shape());
            unbounded_of_shape_with_deadline(
                shape,
                deadline,
                node_name,
                "while creating a tainted NaN fallback",
            )?
        } else {
            output_bounds
        };

        // NaN firewall (#2563, #2706). summarize_bounds already detected NaN;
        // use shared helper to enforce consistent error policy.
        let firewall_context = if sound {
            "GraphNetwork IBP (sound)"
        } else {
            "GraphNetwork IBP"
        };
        if let Some(deadline) = deadline {
            check_nan_firewall_with_poll(
                &output_bounds,
                firewall_context,
                node_name,
                node.layer.layer_type(),
                || check_graph_ibp_deadline(deadline, node_name, "while checking the NaN firewall"),
            )?;
        } else {
            check_nan_firewall(
                &output_bounds,
                firewall_context,
                node_name,
                node.layer.layer_type(),
            )?;
        }

        // Collapse to the interval center so a point input stays a point and the
        // per-node soundness widening cannot be amplified by downstream nodes
        // (#cgan-eval). Only for the non-soundness-critical point forward.
        let output_bounds = if collapse_to_center {
            concrete_center_with_deadline(
                &output_bounds,
                deadline,
                node_name,
                "while centering the node output",
            )?
        } else {
            output_bounds
        };

        if let Some(deadline) = deadline {
            check_graph_ibp_deadline(deadline, node_name, "before caching the node output")?;
        }
        bounds_cache.insert(node_name.clone(), output_bounds);
    }

    // Return the output node's bounds
    let output_name = network.output_name();
    let effective_output: &str = if output_name.is_empty() {
        // Use the last node in exec order as output
        exec_order
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
    } else {
        output_name
    };
    let result = bounds_cache.remove(effective_output).ok_or_else(|| {
        NyError::InvalidSpec(format!("Output bounds not found for {}", effective_output))
    })?;

    // Fail-closed guard (#cctsdb A2): a tainted OUTPUT node may carry finite
    // bounds whose element alignment silently depends on conservative shape
    // substitutions made upstream (a wrong-shaped substitute can misalign
    // downstream gathers/broadcasts). Do not trust them: return fully
    // unbounded bounds of the output's shape ("sound unknown") so no verdict
    // can ever be derived from a tainted output.
    if tainted.contains(effective_output) {
        tracing::warn!(
            "GraphNetwork IBP: output node '{}' is tainted by an upstream OpaqueSkip; \
             returning unbounded output bounds (sound unknown). Set NY_STRICT_IBP=1 to \
             restore strict behavior.",
            effective_output
        );
        let shape = network
            .declared_shape(effective_output)
            .unwrap_or_else(|| result.shape());
        return unbounded_of_shape_with_deadline(
            shape,
            deadline,
            effective_output,
            "while creating a tainted output fallback",
        );
    }
    Ok(result)
}

#[cfg(test)]
mod postprocessing_tests {
    use super::{summarize_bounds, summarize_bounds_with_poll};
    use crate::network::core::graph::ibp::dispatch::{
        check_nan_firewall, check_nan_firewall_with_poll,
    };
    use ndarray::{ArrayD, IxDyn};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    #[test]
    fn pollable_summary_matches_and_cancels_before_publication_8193_elements() {
        let bounds = BoundedTensor::new(
            ArrayD::from_shape_fn(IxDyn(&[8_193]), |index| -3.0_f32 + index[0] as f32 * 0.0001),
            ArrayD::from_shape_fn(IxDyn(&[8_193]), |index| 5.0_f32 + index[0] as f32 * 0.0002),
        )
        .expect("bounds");
        let expected = summarize_bounds(&bounds);
        let mut polls = 0usize;
        let actual = summarize_bounds_with_poll(&bounds, || {
            polls += 1;
            Ok(())
        })
        .expect("pollable summary");
        assert_eq!(actual, expected);
        assert_eq!(polls, 5);

        let mut injected_polls = 0usize;
        let error = summarize_bounds_with_poll(&bounds, || {
            injected_polls += 1;
            if injected_polls == 5 {
                Err(NyError::DeadlineExceeded(
                    "injected summary publication poll".to_string(),
                ))
            } else {
                Ok(())
            }
        })
        .expect_err("final poll must prevent publication");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(injected_polls, 5);
    }

    #[test]
    fn pollable_nan_firewall_matches_and_polls_8193_elements() {
        let bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[8_193]), -1.0),
            ArrayD::from_elem(IxDyn(&[8_193]), 1.0),
        )
        .expect("bounds");
        check_nan_firewall(&bounds, "test", "node", "Identity").expect("ordinary firewall");
        let mut polls = 0usize;
        check_nan_firewall_with_poll(&bounds, "test", "node", "Identity", || {
            polls += 1;
            Ok(())
        })
        .expect("pollable firewall");
        assert_eq!(polls, 7);

        let mut injected_polls = 0usize;
        let error = check_nan_firewall_with_poll(&bounds, "test", "node", "Identity", || {
            injected_polls += 1;
            if injected_polls == 7 {
                Err(NyError::DeadlineExceeded(
                    "injected firewall publication poll".to_string(),
                ))
            } else {
                Ok(())
            }
        })
        .expect_err("final poll must prevent publication");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(injected_polls, 7);
    }

    #[test]
    fn pollable_nan_firewall_preserves_nan_error() {
        let lower = ArrayD::from_elem(IxDyn(&[8_193]), -1.0);
        let mut upper = ArrayD::from_elem(IxDyn(&[8_193]), 1.0);
        upper[[8_192]] = f32::NAN;
        let bounds = BoundedTensor::new_unchecked(lower, upper).expect("test-only NaN bounds");
        let expected = check_nan_firewall(&bounds, "test", "node", "Identity")
            .expect_err("ordinary firewall must reject NaN");
        let actual = check_nan_firewall_with_poll(&bounds, "test", "node", "Identity", || Ok(()))
            .expect_err("pollable firewall must reject NaN");
        assert_eq!(actual.to_string(), expected.to_string());
    }
}
