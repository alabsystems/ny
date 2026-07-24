// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Types for backward CROWN dispatch.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use ndarray::Array1;

use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::MulBinaryRelaxationMode;

/// Borrowed view over a node-bounds map whose values are either owned
/// `BoundedTensor`s or `Arc`-shared ones (#cone-delta increment 2: the
/// constrained/batched BaB caches share out-of-cone entries with the parent
/// domain via `Arc`). One dispatch surface serves both worlds without
/// materializing a converted map.
#[derive(Clone, Copy)]
pub(crate) enum NodeBoundsView<'a> {
    /// Map of owned tensors (graph-CROWN / graph-alpha / sequential callers).
    Plain(&'a HashMap<String, BoundedTensor>),
    /// Map of `Arc`-shared tensors (constrained / batched BaB callers).
    Shared(&'a HashMap<String, Arc<BoundedTensor>>),
}

impl<'a> NodeBoundsView<'a> {
    /// Look up a node's bounds by name.
    pub(crate) fn get(&self, name: &str) -> Option<&'a BoundedTensor> {
        match self {
            Self::Plain(m) => m.get(name),
            Self::Shared(m) => m.get(name).map(|a| a.as_ref()),
        }
    }
}

impl<'a> From<&'a HashMap<String, BoundedTensor>> for NodeBoundsView<'a> {
    fn from(m: &'a HashMap<String, BoundedTensor>) -> Self {
        Self::Plain(m)
    }
}

impl<'a> From<&'a HashMap<String, Arc<BoundedTensor>>> for NodeBoundsView<'a> {
    fn from(m: &'a HashMap<String, Arc<BoundedTensor>>) -> Self {
        Self::Shared(m)
    }
}

/// Context needed for backward dispatch through a single node.
pub(crate) struct DispatchContext<'a> {
    /// Name of the node being processed (for error messages).
    pub node_name: &'a str,
    /// The layer to dispatch through.
    pub layer: &'a Layer,
    /// Input node names for this node.
    pub inputs: &'a [String],
    /// Pre-activation bounds for the first input.
    pub pre_activation: &'a BoundedTensor,
    /// Network input bounds (for resolving `_input` references).
    pub network_input: &'a BoundedTensor,
    /// Cached node bounds (for resolving multi-input node references).
    pub node_bounds: NodeBoundsView<'a>,
    /// Optional GEMM engine for accelerated linear layer propagation.
    pub engine: Option<&'a dyn GemmEngine>,
    /// Optional wall-clock deadline for deadline-aware backward kernels.
    pub deadline: Option<Instant>,
    /// Optional bilinear alpha parameters for McCormick interpolation (#3287).
    /// Maps node name → [4, m, n, k] alpha array. When present and a
    /// BilinearCrown node is found in the map, uses `interpolated_mccormick`
    /// with optimizable r_l/r_u instead of the fixed midpoint heuristic.
    pub bilinear_alphas: Option<&'a HashMap<String, ndarray::Array4<f32>>>,
    /// Relaxation mode for MulBinary CROWN backward (#3439).
    /// Controls which McCormick envelope selection strategy to use for
    /// element-wise multiplication layers. Used when no alpha is available.
    pub mul_binary_relaxation: MulBinaryRelaxationMode,
    /// Optional MulBinary alpha parameters for interpolated McCormick (#3439).
    /// Maps node name → [2, n] alpha array where row 0 = r_l, row 1 = r_u.
    /// When present and a MulBinary node is found in the map, uses
    /// `propagate_linear_binary_with_alpha` with optimizable interpolation.
    pub mul_binary_alphas: Option<&'a HashMap<String, ndarray::Array2<f32>>>,
    /// Optional per-(node, group) `inv_rms` range override for GenBaB norm
    /// branching (#norm-genbab). Maps a `Layer::RmsNorm` node name → a vector of
    /// per-normalization-group windows (indexed by batch row); `None` leaves a
    /// group unconstrained. When present for the node being dispatched, the
    /// decomposed RmsNorm CROWN backward INTERSECTS each group's IBP-derived
    /// `inv_rms` interval with its window (never widening — see `InvRmsOverride`),
    /// tightening the reciprocal/sqrt relaxation so it survives the fused-IBP
    /// validation on this subdomain. Per-group (not per-node) windows are
    /// required for soundness; see `InvRmsOverride`.
    pub norm_inv_rms_override: Option<&'a HashMap<String, Vec<Option<(f32, f32)>>>>,
}

/// Result of dispatching a single node's backward CROWN propagation.
///
/// Each variant tells the caller how to distribute the propagated linear
/// bounds to the node's input(s). The caller handles accumulation and
/// fallback logic.
#[derive(Debug)]
pub(crate) enum BackwardDispatchResult {
    /// Propagated bounds to accumulate to `node.inputs[0]`.
    Single(Box<LinearBounds>),
    /// Two sets of A-matrix bounds for `node.inputs[0]` and `node.inputs[1]`,
    /// plus a single bias pair accumulated exactly once by the engine.
    ///
    /// # Separate bias channel (#2617, #2530)
    ///
    /// Each `LinearBounds` in `bounds_a`/`bounds_b` MUST have `lower_b = upper_b = 0`.
    /// All bias goes through `bias_lower`/`bias_upper`. The engine accumulates
    /// bias exactly once, eliminating the class of bias-splitting bugs (#2520,
    /// #2527, #2529, #2530).
    ///
    /// Reference: alpha-beta-CROWN `backward_bound.py:253-296` — engine
    /// accumulates bias once via `lb = lb + lower_b`.
    Binary {
        bounds_a: Box<LinearBounds>,
        bounds_b: Box<LinearBounds>,
        bias_lower: Array1<f32>,
        bias_upper: Array1<f32>,
    },
    /// N sets of A-matrix bounds (`None` = constant/skipped input),
    /// plus a single bias pair accumulated exactly once by the engine.
    ///
    /// # Separate bias channel (#2617, #2530)
    ///
    /// Each `LinearBounds` MUST have `lower_b = upper_b = 0`.
    /// All bias goes through `bias_lower`/`bias_upper`.
    Nary {
        bounds: Vec<Option<LinearBounds>>,
        bias_lower: Array1<f32>,
        bias_upper: Array1<f32>,
    },
    /// Forward `node_lb` directly to `node.inputs[0]` unchanged (SkipMerge).
    PassThrough,
    /// Layer doesn't support CROWN backward; caller decides fallback.
    Unsupported(String),
}
