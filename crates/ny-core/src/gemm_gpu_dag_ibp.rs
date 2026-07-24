// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-DAG resident IBP plan types (#4276, #4318).
//!
//! Separate from `gemm_gpu_ibp.rs` (sequential resident IBP) to avoid
//! cross-worker file ownership conflicts.

use std::sync::Arc;

use crate::Result;

use super::gpu_ibp::GpuIbpResult;

/// Per-op descriptor for graph-DAG GPU-resident IBP forward pass.
///
/// Unlike sequential [`super::GpuIbpLayer`], each op references its input(s) by
/// index within the plan's op list. Index `usize::MAX` is the sentinel for
/// the network input tensor.
///
/// Reference: designs/2026-03-21-issue-4276-dag-ibp-child-packets.md §Packet A
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum GpuDagIbpOp {
    /// Linear (dense) layer reading from one prior op.
    Linear {
        weight: Arc<[f32]>,
        bias: Option<Arc<[f32]>>,
        out_features: usize,
        in_features: usize,
        /// Index of the input op in the plan, or `NETWORK_INPUT_IDX`.
        input_idx: usize,
    },
    /// Conv2d (groups=1) reading from one prior op.
    Conv2d {
        weight: Arc<[f32]>,
        bias: Option<Arc<[f32]>>,
        out_channels: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        groups: usize,
        input_h: usize,
        input_w: usize,
        /// Index of the input op in the plan, or `NETWORK_INPUT_IDX`.
        input_idx: usize,
    },
    /// Element-wise ReLU.
    ReLU {
        num_elements: usize,
        /// Index of the input op in the plan, or `NETWORK_INPUT_IDX`.
        input_idx: usize,
    },
    /// Element-wise addition of two intermediate results (residual connection).
    Add {
        num_elements: usize,
        /// Index of the first input op.
        input_a_idx: usize,
        /// Index of the second input op.
        input_b_idx: usize,
    },
    /// Metadata-only reshape (Flatten / Reshape). No buffer change.
    View {
        output_shape: Arc<[usize]>,
        /// Index of the input op in the plan, or `NETWORK_INPUT_IDX`.
        input_idx: usize,
    },
    /// Average pooling (global or windowed).
    ///
    /// AveragePool is linear so IBP bounds are exact: pool(lower), pool(upper).
    AveragePool {
        channels: usize,
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        count_include_pad: bool,
        /// Whether this is global average pooling (kernel covers full spatial).
        is_global: bool,
        /// Total output elements (batch * channels * output_h * output_w).
        num_elements: usize,
        /// Index of the input op in the plan, or `NETWORK_INPUT_IDX`.
        input_idx: usize,
    },
}

/// Sentinel index meaning "the network input tensor" in [`GpuDagIbpOp`] index fields.
pub const NETWORK_INPUT_IDX: usize = usize::MAX;

/// Complete graph-DAG resident IBP plan descriptor.
///
/// Ops are in topological (execution) order. Each op's `input_idx` fields
/// reference earlier entries or [`NETWORK_INPUT_IDX`].
#[derive(Clone, Debug)]
pub struct GpuDagIbpPlanDesc {
    /// Ops in topological order.
    pub ops: Vec<GpuDagIbpOp>,
    /// Network input shape.
    pub input_shape: Vec<usize>,
    /// Index of the op whose output is the network output.
    pub output_op_idx: usize,
}

/// Cached graph-DAG GPU execution plan for resident IBP forward passes.
pub trait GpuDagIbpModelPlan: Sync + Send {
    /// Run one resident graph-DAG IBP forward pass using cached static buffers.
    fn dag_ibp_forward_cached(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult>;
}

/// Optional cached-plan preparation for graph-DAG GPU-resident IBP backends.
///
/// Analogous to [`super::GpuIbpForwardExt`] but for DAG topologies with residual
/// connections. Callers should fall back to the CPU graph IBP loop when this
/// returns `Ok(None)`.
pub trait GpuDagIbpForwardExt: Sync + Send {
    /// Prepare a reusable graph-DAG resident-IBP model plan (FAST, unsound — the
    /// f32 reductions carry no certified rounding-error term, so the bound can be
    /// *tighter* than the true range; never legal for a verdict).
    fn prepare_dag_model_plan(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> Result<Option<Box<dyn GpuDagIbpModelPlan>>>;

    /// Whether this backend can produce a SOUND (verdict-legal) graph-DAG IBP
    /// plan (`docs/SOUND_GPU_IBP_PLAN.md`, T1.0). Backends without a certified
    /// sound DAG path leave this `false`, so the caller keeps the proven-sound
    /// CPU graph loop for verdicts.
    fn provides_sound_gpu_dag_ibp(&self) -> bool {
        false
    }

    /// Prepare a reusable SOUND graph-DAG resident-IBP model plan whose every
    /// emitted interval is a CERTIFIED enclosure of both the true forward range
    /// and the CPU `propagate_ibp_sound` bound (each node applies the directed
    /// `3·γ_k·S + 4·N·u·|endpoint| + flush` widening; Metal FTZ-safe). Returns
    /// `Ok(None)` when this backend has no sound DAG path (the default), so the
    /// caller falls back to the CPU graph IBP loop.
    fn prepare_sound_dag_model_plan(
        &self,
        _plan: &GpuDagIbpPlanDesc,
    ) -> Result<Option<Box<dyn GpuDagIbpModelPlan>>> {
        Ok(None)
    }
}
