// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dispatch helpers for special-case batched CROWN graph nodes.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::BatchedLinearBounds;
use crate::layers::Layer;
use crate::network::core::graph::backward_helpers::where_constant_mask;
use crate::network::core::graph::batched_accumulator::BatchedCrownAccumulator;
use crate::network::GraphNode;
use crate::types::{BoundsProvenance, CrownBackwardResult};
use crate::MulBinaryRelaxationMode;

use super::binary_ops::{resolve_binary_input_bounds, AttentionCompositionRuntime};
use super::GraphNetwork;

/// Result of the Conv2d Patches fast-path attempt in the batched CROWN backward loop.
///
/// The backward loop tries Patches-mode propagation for Conv2d nodes before
/// falling through to the standard Dense dispatch.
pub(super) enum PatchesDispatchResult {
    /// Patches backward handled this node. Caller should `continue` the loop.
    Handled,
    /// Partial CROWN fallback was triggered. Caller should return this result.
    PartialFallback(Box<CrownBackwardResult>),
    /// Patches not applicable or already converted to Dense. Proceed with standard dispatch.
    DenseBounds(Box<BatchedLinearBounds>),
}

/// Result of dispatching a special-case binary or graph-structured batched operator.
#[derive(Debug)]
pub(super) enum SpecialBatchedDispatchResult {
    NotHandled,
    Handled,
    PartialFallback(Box<CrownBackwardResult>),
}

/// Whether a batched CROWN backward error should trigger partial IBP fallback.
///
/// Unsupported operations/configurations and explicit numerical-instability
/// guards degrade to IBP because the fallback remains sound. Shape mismatch is
/// intentionally excluded for the base path because silently widening previously
/// supported layers can mask regressions.
pub(super) fn is_fallback_eligible_error(error: &NyError) -> bool {
    matches!(
        error,
        NyError::UnsupportedOp(_)
            | NyError::UnsupportedConfiguration(_)
            | NyError::NumericalInstability(_)
    )
}

/// Bilinear batched fallback extends the base fallback set with shape mismatch.
///
/// Bilinear alpha-parameterized relaxation can hit legitimate shape-mismatch
/// fallback cases during partial composition, so the dispatch path keeps that
/// variant in the eligible set.
pub(super) fn is_bilinear_fallback_eligible_error(error: &NyError) -> bool {
    is_fallback_eligible_error(error) || matches!(error, NyError::ShapeMismatch { .. })
}

/// Whether `shape` is exactly the tensor layout that `bounds`'s A-matrix
/// columns index — i.e. `A` has shape `[batch.., out_dim, in_dim]` and `shape`
/// is `[batch.., in_dim]`.
///
/// Every arm below that reasons *positionally* about A's last axis (the Concat
/// column split, the constant-mask Where column split, the embedded-constant
/// Where bias fold) is only exact when the operand tensor lays out as
/// `[batch.., in_dim]` against those same batch dimensions. Anything else —
/// a broadcast operand, a rank mismatch, a re-batched carrier — is refused, and
/// the caller keeps the existing sound fallback.
fn batched_operand_layout_matches(bounds: &BatchedLinearBounds, shape: &[usize]) -> bool {
    let a_shape = bounds.lower_a().shape();
    if a_shape.len() < 2 {
        return false;
    }
    let batch = &a_shape[..a_shape.len() - 2];
    let in_dim = a_shape[a_shape.len() - 1];
    shape.len() == batch.len() + 1 && shape[..batch.len()] == *batch && shape[batch.len()] == in_dim
}

fn global_operand_layout_matches(bounds: &BatchedLinearBounds, shape: &[usize]) -> bool {
    let a = bounds.lower_a().shape();
    a.len() >= 2
        && a.len() <= 10
        && shape.len() <= 8
        && checked_shape_product(&a[..a.len() - 2]) == Some(1)
        && checked_shape_product(shape) == Some(a[a.len() - 1])
        && bounds.input_shape() == shape
}

/// Recover global property rows through an ORIGINAL Flatten that collapsed
/// every tensor cell into one feature row. Singleton carrier prefixes are
/// storage metadata, not separate Softmax groups. Keep the full original
/// column order and all certified error cells, without allocating new arrays.
/// Local/batched carriers retain the ordinary dispatcher unchanged.
pub(crate) fn remap_global_flatten_batched(
    layer: &Layer,
    bounds: BatchedLinearBounds,
    pre_activation: &BoundedTensor,
) -> Result<BatchedLinearBounds> {
    let Layer::Flatten(flatten) = layer else {
        return Ok(bounds);
    };
    bounds.validate_internal_shapes()?;
    let a_shape = bounds.lower_a().shape();
    let rank = a_shape.len();
    let original_shape = pre_activation.shape();
    if rank > 10 || original_shape.len() > 8 || bounds.input_shape().len() > 8 {
        return Ok(bounds);
    }
    let columns = a_shape[rank - 1];
    let rows = a_shape[rank - 2];
    // The authored Flatten must really produce [1, ALL original cells].
    // Equal products alone do not authenticate arbitrary reshape semantics.
    let flattened_shape = flatten.compute_output_shape(original_shape)?;
    if rows == 0
        || columns == 0
        || flattened_shape.as_slice() != [1, columns]
        || bounds.input_shape() != flattened_shape.as_slice()
        || checked_shape_product(original_shape) != Some(columns)
        || checked_shape_product(&a_shape[..rank - 2]) != Some(1)
        || bounds.lower_b().len() != rows
        || !bounds.lower_a().is_standard_layout()
        || !bounds.upper_a().is_standard_layout()
        || !bounds.lower_b().is_standard_layout()
        || !bounds.upper_b().is_standard_layout()
        || bounds
            .lower_a_err
            .iter()
            .chain(bounds.upper_a_err.iter())
            .any(|error| !error.is_standard_layout())
    {
        return Ok(bounds);
    }
    let BatchedLinearBounds {
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        output_shape,
        lower_a_err,
        upper_a_err,
        ..
    } = bounds;
    let matrix = |array: ArrayD<f32>| {
        array
            .into_shape_with_order(IxDyn(&[rows, columns]))
            .map_err(|error| NyError::InvalidSpec(format!("global Flatten carrier: {error}")))
    };
    let bias = |array: ArrayD<f32>| {
        array
            .into_shape_with_order(IxDyn(&[rows]))
            .map_err(|error| NyError::InvalidSpec(format!("global Flatten bias: {error}")))
    };
    let remapped = BatchedLinearBounds {
        lower_a: matrix(lower_a)?,
        lower_b: bias(lower_b)?,
        upper_a: matrix(upper_a)?,
        upper_b: bias(upper_b)?,
        lower_a_err: lower_a_err.map(matrix).transpose()?,
        upper_a_err: upper_a_err.map(matrix).transpose()?,
        input_shape: original_shape.to_vec(),
        output_shape,
    };
    remapped.validate_internal_shapes()?;
    Ok(remapped)
}

/// Run an EXACT-linear binary split under the certified coefficient-error
/// contract (#vnncomp-aw-soundness), mirroring the carrier path in
/// [`Layer::propagate_crown_backward_batched`] and the dense
/// `dispatch_exact_linear_with_err`.
///
/// The batched `Linear` backward attaches a fresh `γ_n·S` per-coefficient error
/// to every carrier it produces, so in any real graph the bounds arriving at a
/// `Sub` / `Concat` / constant-mask `Where` node CARRY an error. Constructing
/// the split with `new_or_conservative` alone would silently drop it — a bound
/// that is tighter than the arithmetic justifies. Instead, run the SAME split
/// twice: once on the error-free coefficients (the real result) and once on the
/// non-negative [`coeff_err_carrier`], then attach the carried result. Because
/// the split is a fixed column transform with no sign cancellation, the carried
/// coefficient magnitudes bound `T_abs(err_in)` exactly — the same argument the
/// carrier path already relies on for `Add`/`Slice`/`Transpose`.
///
/// `attach_err_from_carried` takes coefficient MAGNITUDES, so a split that
/// negates a side (`Sub`) is handled without a sign fix-up.
fn split_exact_linear_with_coeff_err(
    node_lb: &BatchedLinearBounds,
    split: impl Fn(&BatchedLinearBounds) -> Result<(BatchedLinearBounds, BatchedLinearBounds)>,
) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
    let Some(carrier) = node_lb.coeff_err_carrier() else {
        return split(node_lb);
    };
    let mut plain = node_lb.clone();
    plain.lower_a_err = None;
    plain.upper_a_err = None;
    let (mut real_a, mut real_b) = split(&plain)?;
    let (carried_a, carried_b) = split(&carrier)?;
    real_a.attach_err_from_carried(&carried_a);
    real_b.attach_err_from_carried(&carried_b);
    Ok((real_a, real_b))
}

/// Exact batched CROWN backward for `Sub` (C = A - B).
///
/// Mirrors the DENSE `SubLayer::propagate_linear_binary` term for term:
/// * A's coefficients pass through unchanged,
/// * B's coefficients are NEGATED with **no lower/upper swap** — CROWN composes
///   by substitution, so from `obj >= lower_a·C + lower_b` and `C = A - B` we get
///   `obj >= lower_a·A + (-lower_a)·B + lower_b`. B's LOWER coefficient is
///   `-lower_a`, not `-upper_a`. (Swapping is the rule for negating a bounded
///   *quantity*, not for negating the coefficients of a linear *relation*, and
///   it is NOT conservative — see the proof note in `sub.rs`, which records a
///   reproduced false bound from the swapped form.)
/// * the incoming bias is split in half with DIRECTED rounding so the two
///   accumulated halves re-sum conservatively, exactly as `Add` does.
///
/// `SubLayer` has no batched backward of its own, so the split lives here beside
/// the dispatcher that needs it; the transform is exact (no relaxation), so
/// there is nothing to prove beyond the substitution above.
fn batched_sub_backward(
    bounds: &BatchedLinearBounds,
) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
    let lower_b_half = bounds.lower_b().mapv(|v| next_down_f32(v * 0.5));
    let upper_b_half = bounds.upper_b().mapv(|v| next_up_f32(v * 0.5));

    let bounds_a = BatchedLinearBounds::new_or_conservative(
        bounds.lower_a().clone(),
        lower_b_half.clone(),
        bounds.upper_a().clone(),
        upper_b_half.clone(),
        bounds.input_shape().to_vec(),
        bounds.output_shape().to_vec(),
    )?;
    let bounds_b = BatchedLinearBounds::new_or_conservative(
        bounds.lower_a().mapv(|v| -v),
        lower_b_half,
        bounds.upper_a().mapv(|v| -v),
        upper_b_half,
        bounds.input_shape().to_vec(),
        bounds.output_shape().to_vec(),
    )?;
    Ok((bounds_a, bounds_b))
}

/// Batched analogue of `mask_linear_bounds_columns` (`backward_helpers.rs`):
/// build the exact backward bounds for ONE branch of a constant-condition
/// `Where` by zeroing the A-columns that belong to the other branch.
///
/// A has shape `[batch.., out_dim, in_dim]` and the condition tensor has shape
/// `[batch.., in_dim]`, so element `(p, i, j)` of A multiplies output position
/// `(p, j)` of the `Where`, whose branch is `mask[p * in_dim + j]`. With a fixed
/// mask `M`, `y = M∘t + (1-M)∘f` EXACTLY, so
/// `Σ_j A[p,i,j]·y[p,j] = Σ_{j∈M} A[p,i,j]·t[p,j] + Σ_{j∉M} A[p,i,j]·f[p,j]`
/// — a pure linear column split, no relaxation, hence sound and exactly as
/// tight as the dense arm it mirrors.
///
/// The two branch contributions are accumulated separately, so the incoming bias
/// is carried on the TRUE branch only and zeroed on the FALSE branch (same
/// convention as the dense helper) — never halved, never doubled.
fn mask_batched_bounds_columns(
    node_lb: &BatchedLinearBounds,
    mask: &[bool],
    keep_true: bool,
) -> Result<BatchedLinearBounds> {
    let a_shape = node_lb.lower_a().shape().to_vec();
    let ndim = a_shape.len();
    if ndim < 2 {
        return Err(NyError::InvalidSpec(
            "batched Where constant-mask split requires ndim >= 2 coefficients".to_string(),
        ));
    }
    let in_dim = a_shape[ndim - 1];
    let out_dim = a_shape[ndim - 2];
    let total_batch = checked_shape_product(&a_shape[..ndim - 2])
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "batched Where constant-mask split: batch dims {:?} overflow usize",
                &a_shape[..ndim - 2]
            ))
        })?
        .max(1);
    if mask.len() != total_batch.saturating_mul(in_dim) {
        return Err(NyError::shape_mismatch(
            vec![total_batch * in_dim],
            vec![mask.len()],
        ));
    }

    // Rebuild in guaranteed standard (row-major) layout so the flat chunking
    // below is the logical [batch.., out, in] order regardless of the incoming
    // array's strides.
    let masked = |src: &ArrayD<f32>| -> Result<ArrayD<f32>> {
        let mut values: Vec<f32> = src.iter().copied().collect();
        for (row_index, row) in values.chunks_mut(in_dim.max(1)).enumerate() {
            let base = (row_index / out_dim.max(1)) * in_dim;
            for (column, value) in row.iter_mut().enumerate() {
                if mask[base + column] != keep_true {
                    *value = 0.0;
                }
            }
        }
        ArrayD::from_shape_vec(IxDyn(&a_shape), values).map_err(|e| {
            NyError::InvalidSpec(format!(
                "batched Where constant-mask split: rebuild of {a_shape:?} failed: {e}"
            ))
        })
    };

    let (lower_b, upper_b) = if keep_true {
        (node_lb.lower_b().clone(), node_lb.upper_b().clone())
    } else {
        (
            ArrayD::zeros(node_lb.lower_b().raw_dim()),
            ArrayD::zeros(node_lb.upper_b().raw_dim()),
        )
    };

    BatchedLinearBounds::new_or_conservative(
        masked(node_lb.lower_a())?,
        lower_b,
        masked(node_lb.upper_a())?,
        upper_b,
        node_lb.input_shape().to_vec(),
        node_lb.output_shape().to_vec(),
    )
}

/// Test hook for the constant-mask column split (see
/// `where_constant_mask_columns_partition_the_carrier`).
#[cfg(test)]
pub(super) fn mask_batched_bounds_columns_for_test(
    node_lb: &BatchedLinearBounds,
    mask: &[bool],
    keep_true: bool,
) -> Result<BatchedLinearBounds> {
    mask_batched_bounds_columns(node_lb, mask, keep_true)
}

impl GraphNetwork {
    /// Handle the binary and graph-structured batched operators that do not route
    /// through the standard unary `Layer::propagate_crown_backward_batched` path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_special_batched_operator(
        &self,
        node: &GraphNode,
        node_name: &str,
        node_lb: &BatchedLinearBounds,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        mul_binary_relaxation: MulBinaryRelaxationMode,
        bilinear_alphas: Option<&HashMap<String, ndarray::Array4<f32>>>,
        attention_runtime: &mut AttentionCompositionRuntime<'_>,
        node_linear_bounds: &mut BatchedCrownAccumulator,
    ) -> Result<SpecialBatchedDispatchResult> {
        // The immutable original PV/Softmax/scale/QK region is a proof-only
        // three-source replacement for this contribution. It does not alter
        // the existing MatMul identity-retry policy or skip pending internals.
        if let Some(regions) = &mut attention_runtime.regions {
            if regions.try_global_unary(node, node_lb, input, node_bounds, node_linear_bounds)? {
                return Ok(SpecialBatchedDispatchResult::Handled);
            }
            if regions.try_batched(
                self,
                node,
                node_lb,
                input,
                node_bounds,
                node_linear_bounds,
                None,
            )? {
                return Ok(SpecialBatchedDispatchResult::Handled);
            }
        }
        match &node.layer {
            Layer::PointLinear(_) => Err(crate::network::proof_eligibility::diagnostic_only()),
            Layer::ExpandLikeLastAxis(expand) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for ExpandLikeLastAxis source",
                    "IBP bounds for ExpandLikeLastAxis reference",
                )?;
                let (lb_a, lb_b) = expand
                    .propagate_linear_batched_binary(
                        node_lb,
                        binary_inputs.input_a_bounds,
                        binary_inputs.input_b_bounds,
                    )
                    .map_err(|e| {
                        NyError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (ExpandLikeLastAxis): {}",
                            node_name, e
                        ))
                    })?;
                Self::accumulate_binary_bounds(
                    binary_inputs.input_a_name,
                    lb_a,
                    binary_inputs.input_b_name,
                    lb_b,
                    node_linear_bounds,
                )?;
                Ok(SpecialBatchedDispatchResult::Handled)
            }
            Layer::Add(add) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node, input, node_bounds, "IBP bounds for Add input A", "IBP bounds for Add input B",
                )?;
                node_lb.validate_internal_shapes()?;
                node_lb.validate_no_nan()?;
                let shape = binary_inputs.input_a_bounds.shape();
                if shape != binary_inputs.input_b_bounds.shape()
                    || node_lb.input_shape() != shape
                    || !(batched_operand_layout_matches(node_lb, shape)
                        || global_operand_layout_matches(node_lb, shape))
                {
                    // A broadcast would require summing original columns, not
                    // duplicating them. Preserve the complete ordinary fallback.
                    return Ok(SpecialBatchedDispatchResult::NotHandled);
                }
                // Add duplicates each coefficient onto both original operands;
                // its certified error must follow BOTH copies too. The direct
                // layer split only constructs the nominal arrays/bias halves.
                let (lb_a, lb_b) = split_exact_linear_with_coeff_err(node_lb, |bounds| {
                    add.propagate_linear_batched_binary(bounds)
                }).map_err(|e| {
                    NyError::InvalidSpec(format!(
                        "Batched CROWN failed at node '{}' (Add): {}",
                        node_name, e
                    ))
                })?;
                Self::accumulate_binary_bounds(binary_inputs.input_a_name, lb_a, binary_inputs.input_b_name, lb_b, node_linear_bounds)?;
                Ok(SpecialBatchedDispatchResult::Handled)
            }
            // === Sub: exact binary split, B negated (dense-dispatch mirror) ===
            // The DENSE dispatcher already had a `Layer::Sub` arm; the
            // batched dispatcher had none, so every `Sub` node fell through to the
            // generic unary dispatch, whose `UnsupportedOp` either concretized the
            // node through its IBP forward (partial CROWN — the condition-blind
            // interval difference) or, with any other pending path, aborted the whole
            // batched pass. nn emits `Sub` in every fusion difference-graph, so that
            // was the difference bound being thrown away at the one node that carries
            // the correlation between the two branches.
            Layer::Sub(_) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for Sub input A",
                    "IBP bounds for Sub input B",
                )?;
                // The identity / negated-identity Jacobian is exact ONLY when both
                // operands have the node's own shape. `SubLayer`'s IBP broadcasts;
                // a broadcast operand would need the backward to SUM coefficients
                // over the broadcast axes, which this split does not do. Refuse.
                let operand_shape = binary_inputs.input_a_bounds.shape();
                if operand_shape != binary_inputs.input_b_bounds.shape()
                    || node_lb.input_shape() != operand_shape
                    || !batched_operand_layout_matches(node_lb, operand_shape)
                {
                    debug!(
                        "GraphNetwork batched CROWN: Sub '{}' operand layout not provably \
                         elementwise (a {:?}, b {:?}, bounds input {:?}) — keeping IBP fallback",
                        node_name,
                        operand_shape,
                        binary_inputs.input_b_bounds.shape(),
                        node_lb.input_shape(),
                    );
                    return Ok(SpecialBatchedDispatchResult::NotHandled);
                }
                let (lb_a, lb_b) = split_exact_linear_with_coeff_err(node_lb, batched_sub_backward)
                    .map_err(|e| {
                        NyError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (Sub): {}",
                            node_name, e
                        ))
                    })?;
                Self::accumulate_binary_bounds(
                    binary_inputs.input_a_name,
                    lb_a,
                    binary_inputs.input_b_name,
                    lb_b,
                    node_linear_bounds,
                )?;
                Ok(SpecialBatchedDispatchResult::Handled)
            }
            // === Concat: last-axis column split (dense-dispatch mirror) ===
            // `ConcatLayer::propagate_linear_batched_binary` is implemented and
            // tested but had ZERO production callers — no dispatcher ever routed a
            // Concat node to it, so nn's Concat-heavy tensor graphs concretized at
            // every join.
            //
            // That backward is a prefix/suffix split of A's LAST axis sized by
            // `product(input_shape)`. It is the exact backward only when the join
            // really is a last-axis split of the same column space, so the guards
            // below refuse everything else:
            //   * exactly two dynamic operands, no embedded constants (the dense
            //     N-ary arm folds `W_i·c_i` into a separate bias channel that the
            //     batched accumulator does not have);
            //   * equal rank, and the normalized axis is the LAST axis (an axis-0
            //     join of `[1, n]` tensors lives in different BATCH positions of the
            //     batched carrier, not in different columns — no column split can
            //     express it);
            //   * `product(shape) == shape.last()` for both operands, because the
            //     layer sizes the split by the shape PRODUCT while the split itself
            //     is along the last axis; the two agree only when the leading dims
            //     are all 1, and disagreeing would silently cut the matrix in the
            //     wrong place;
            //   * the incoming carrier's column space is exactly the join output.
            Layer::Concat(concat) => {
                let has_embedded_constants = concat
                    .constant_inputs
                    .as_ref()
                    .is_some_and(|ci| ci.iter().any(|c| c.is_some()));
                if has_embedded_constants || node.inputs.len() != 2 {
                    return Ok(SpecialBatchedDispatchResult::NotHandled);
                }
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for Concat input A",
                    "IBP bounds for Concat input B",
                )?;
                let a_shape = binary_inputs.input_a_bounds.shape().to_vec();
                let b_shape = binary_inputs.input_b_bounds.shape().to_vec();
                let ndim = a_shape.len();
                let split_is_exact = ndim >= 1
                    && b_shape.len() == ndim
                    && concat
                        .normalize_axis(ndim)
                        .is_ok_and(|axis| axis == ndim - 1)
                    && checked_shape_product(&a_shape) == Some(a_shape[ndim - 1])
                    && checked_shape_product(&b_shape) == Some(b_shape[ndim - 1]);
                if !split_is_exact {
                    debug!(
                        "GraphNetwork batched CROWN: Concat '{}' join (axis {}, a {:?}, b {:?}) \
                         is not a provable last-axis column split — keeping IBP fallback",
                        node_name, concat.axis, a_shape, b_shape
                    );
                    return Ok(SpecialBatchedDispatchResult::NotHandled);
                }
                let joined = a_shape[ndim - 1].checked_add(b_shape[ndim - 1]);
                let mut expected_output = a_shape[..ndim - 1].to_vec();
                match joined {
                    Some(joined) => expected_output.push(joined),
                    None => return Ok(SpecialBatchedDispatchResult::NotHandled),
                }
                if node_lb.input_shape() != expected_output.as_slice()
                    || !batched_operand_layout_matches(node_lb, &expected_output)
                {
                    debug!(
                        "GraphNetwork batched CROWN: Concat '{}' carrier column space {:?} \
                         != join output {:?} — keeping IBP fallback",
                        node_name,
                        node_lb.input_shape(),
                        expected_output
                    );
                    return Ok(SpecialBatchedDispatchResult::NotHandled);
                }
                match split_exact_linear_with_coeff_err(node_lb, |bounds| {
                    concat.propagate_linear_batched_binary(bounds, &a_shape, &b_shape)
                }) {
                    Ok((lb_a, lb_b)) => {
                        Self::accumulate_binary_bounds(
                            binary_inputs.input_a_name,
                            lb_a,
                            binary_inputs.input_b_name,
                            lb_b,
                            node_linear_bounds,
                        )?;
                        Ok(SpecialBatchedDispatchResult::Handled)
                    }
                    Err(e) => {
                        // Guarded above, so this is a defence-in-depth path: keep the
                        // established sound IBP fallback rather than publishing a
                        // partially-split carrier.
                        debug!(
                            "GraphNetwork batched CROWN: Concat '{}' batched split failed ({}) \
                             — keeping IBP fallback",
                            node_name, e
                        );
                        Ok(SpecialBatchedDispatchResult::NotHandled)
                    }
                }
            }
            Layer::MatMul(matmul) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for MatMul input A",
                    "IBP bounds for MatMul input B",
                )?;
                match self.handle_matmul_batched(
                    matmul,
                    node_name,
                    node_lb,
                    node_bounds,
                    output_shape,
                    binary_inputs,
                    attention_runtime,
                    node_linear_bounds,
                )? {
                    Some(fallback) => Ok(SpecialBatchedDispatchResult::PartialFallback(Box::new(
                        fallback,
                    ))),
                    None => Ok(SpecialBatchedDispatchResult::Handled),
                }
            }
            Layer::MulBinary(mul) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for MulBinary input A",
                    "IBP bounds for MulBinary input B",
                )?;
                match mul.propagate_linear_batched_binary(
                    node_lb,
                    binary_inputs.input_a_bounds,
                    binary_inputs.input_b_bounds,
                    mul_binary_relaxation,
                ) {
                    Ok((lb_a, lb_b)) => {
                        let label = format!("MulBinary {:?}", mul_binary_relaxation);
                        if let Some(fallback) = Self::validate_binary_bounds_and_accumulate(
                            lb_a,
                            lb_b,
                            binary_inputs.input_a_name,
                            binary_inputs.input_b_name,
                            node_name,
                            &label,
                            node_lb,
                            node_bounds,
                            output_shape,
                            node_linear_bounds,
                        )? {
                            return Ok(SpecialBatchedDispatchResult::PartialFallback(Box::new(
                                fallback,
                            )));
                        }
                        Ok(SpecialBatchedDispatchResult::Handled)
                    }
                    Err(e) if is_fallback_eligible_error(&e) => {
                        debug!(
                            "GraphNetwork batched CROWN: MulBinary '{}' {:?} failed ({}), using partial CROWN",
                            node_name,
                            mul_binary_relaxation,
                            e
                        );
                        Self::partial_crown_ibp_fallback(
                            node_lb,
                            node_name,
                            node_bounds,
                            output_shape,
                        )
                        .map(Box::new)
                        .map(SpecialBatchedDispatchResult::PartialFallback)
                    }
                    Err(e) => Err(e),
                }
            }
            Layer::BilinearCrown(bilinear) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for BilinearCrown input A",
                    "IBP bounds for BilinearCrown input B",
                )?;
                let node_alpha = bilinear_alphas.and_then(|m| m.get(node_name));
                match bilinear.propagate_linear_batched_binary_with_alpha(
                    node_lb,
                    binary_inputs.input_a_bounds,
                    binary_inputs.input_b_bounds,
                    node_alpha,
                ) {
                    Ok((lb_a, lb_b)) => {
                        if let Some(fallback) = Self::validate_binary_bounds_and_accumulate(
                            lb_a,
                            lb_b,
                            binary_inputs.input_a_name,
                            binary_inputs.input_b_name,
                            node_name,
                            "BilinearCrown",
                            node_lb,
                            node_bounds,
                            output_shape,
                            node_linear_bounds,
                        )? {
                            return Ok(SpecialBatchedDispatchResult::PartialFallback(Box::new(
                                fallback,
                            )));
                        }
                        Ok(SpecialBatchedDispatchResult::Handled)
                    }
                    Err(e) if is_bilinear_fallback_eligible_error(&e) => {
                        debug!(
                            "BilinearCrown '{}' failed ({}), using partial CROWN",
                            node_name, e
                        );
                        Self::partial_crown_ibp_fallback(
                            node_lb,
                            node_name,
                            node_bounds,
                            output_shape,
                        )
                        .map(Box::new)
                        .map(SpecialBatchedDispatchResult::PartialFallback)
                    }
                    Err(e) => Err(e),
                }
            }
            // === Where: condition-tier routing (DAG-CROWN mirror) ===
            //
            // The dense arm (graph_crown/propagation.rs) has four tiers; this is
            // the batched port of three of them, in the same order:
            //   1. EMBEDDED-CONSTANT `Where` — both branches are frozen constants,
            //      so the output carries no linear dependence on `cond`: fold the
            //      exact per-element select into the bias and route ZERO
            //      coefficients to `cond`.
            //   2/3. SIGN-PROVABLE condition — one branch is taken over the ENTIRE
            //      input box, so `Where` IS that branch: route the carrier to it
            //      unchanged.
            //   4. CONSTANT MASK — the condition is bound-independent per element,
            //      so `Where` is an exact fixed 0/1 column select: split the
            //      carrier's columns between the branches.
            // The fifth dense tier (data-dependent condition -> concretize the node
            // and accumulate a pure bias to the NETWORK INPUT) is deliberately NOT
            // ported: the batched accumulator has no bias-to-network-input channel,
            // and inventing one would risk dropping the bias on a path that never
            // reaches `_input`. That case stays `NotHandled`, i.e. the existing
            // sound IBP-union fallback.
            Layer::Where(where_layer) => {
                // --- Tier 1: embedded-constant Where -------------------------
                // CORRECTION (was: "the generic fallback concretizes through its
                // forward bound, which already performs the exact constant
                // select"). It does NOT. `WhereLayer::propagate_ibp` routes an
                // embedded-constant node to `propagate_ibp_with_condition`, which
                // broadcasts both constants and hands them to
                // `propagate_ibp_ternary` — the CONDITION-BLIND union
                // `[min(t,f), max(t,f)]`. The exact select lives only in
                // `embedded_constant_select_output`, which the IBP forward never
                // calls. So the old bail concretized through the union and was
                // strictly looser than the dense arm on every constant condition
                // (and, with any other pending backward path, made the whole
                // batched pass abort instead of falling back).
                if where_layer.has_embedded_constants() {
                    let Ok(cond_input) = node.require_unary_input() else {
                        return Ok(SpecialBatchedDispatchResult::NotHandled);
                    };
                    let cond_bounds = super::resolve_node_bounds(
                        cond_input,
                        input,
                        node_bounds,
                        "IBP bounds for embedded-constant Where condition",
                    )?;
                    // Both constants are broadcast to the condition's shape, so the
                    // node's output shape IS the condition's shape; require the
                    // carrier's column space to agree before folding positionally.
                    if node_lb.input_shape() != cond_bounds.shape()
                        || !batched_operand_layout_matches(node_lb, cond_bounds.shape())
                    {
                        return Ok(SpecialBatchedDispatchResult::NotHandled);
                    }
                    let select = match where_layer.embedded_constant_select_output(cond_bounds) {
                        Ok(select) => select,
                        Err(e) => {
                            debug!(
                                "GraphNetwork batched CROWN: Where '{}' embedded-constant select \
                                 unavailable ({}) — keeping IBP fallback",
                                node_name, e
                            );
                            return Ok(SpecialBatchedDispatchResult::NotHandled);
                        }
                    };
                    // `select` is exact where the condition is determined and the
                    // sound union elsewhere, so concretizing the carrier over it is
                    // an enclosure of every reachable objective value — and it is
                    // never wider than the IBP-union concretization it replaces.
                    let concrete = match node_lb.concretize_sound(&select) {
                        Ok(concrete) => concrete,
                        Err(e) => {
                            debug!(
                                "GraphNetwork batched CROWN: Where '{}' embedded-constant \
                                 concretization failed ({}) — keeping IBP fallback",
                                node_name, e
                            );
                            return Ok(SpecialBatchedDispatchResult::NotHandled);
                        }
                    };
                    let (folded_lower, folded_upper) = concrete.into_parts();
                    if folded_lower.shape() != node_lb.lower_b().shape() {
                        return Ok(SpecialBatchedDispatchResult::NotHandled);
                    }
                    // Zero coefficients: the output is a constant vector w.r.t.
                    // whatever produced `cond`. The bias rides the `cond` edge, which
                    // (like every graph edge) roots at the network input, so it is
                    // carried to the terminal concretization rather than dropped.
                    let zero_a = ArrayD::zeros(node_lb.lower_a().raw_dim());
                    let folded = BatchedLinearBounds::new_or_conservative(
                        zero_a.clone(),
                        folded_lower,
                        zero_a,
                        folded_upper,
                        cond_bounds.shape().to_vec(),
                        node_lb.output_shape().to_vec(),
                    )?;
                    node_linear_bounds.accumulate_dense(cond_input, folded)?;
                    return Ok(SpecialBatchedDispatchResult::Handled);
                }
                let (cond_input, true_input, false_input) = node.require_ternary_inputs()?;
                let cond_bounds = super::resolve_node_bounds(
                    cond_input,
                    input,
                    node_bounds,
                    "IBP bounds for Where condition",
                )?;
                // --- Tiers 2/3: sign-provable condition ----------------------
                let cond_all_true = cond_bounds.lower().iter().all(|&v| v >= 0.5);
                let cond_all_false = cond_bounds.upper().iter().all(|&v| v <= 0.5);
                if cond_all_true {
                    node_linear_bounds.accumulate_dense(true_input, node_lb.clone())?;
                    return Ok(SpecialBatchedDispatchResult::Handled);
                }
                if cond_all_false {
                    node_linear_bounds.accumulate_dense(false_input, node_lb.clone())?;
                    return Ok(SpecialBatchedDispatchResult::Handled);
                }
                // --- Tier 4: exact constant-mask column split ----------------
                // A causal / padding attention mask is exactly this shape: a fixed
                // 0/1 tensor with BOTH branches live, which tiers 2/3 cannot prove
                // and which the union fallback blurs into `[min(t,f), max(t,f)]` at
                // every position. The split below is an exact linear transform.
                if batched_operand_layout_matches(node_lb, cond_bounds.shape()) {
                    if let Some(mask) = where_constant_mask(cond_bounds) {
                        // The column split is exact-linear, so any certified
                        // coefficient error rides the carrier through it.
                        match split_exact_linear_with_coeff_err(node_lb, |bounds| {
                            Ok((
                                mask_batched_bounds_columns(bounds, &mask, true)?,
                                mask_batched_bounds_columns(bounds, &mask, false)?,
                            ))
                        }) {
                            Ok((true_lb, false_lb)) => {
                                node_linear_bounds.accumulate_dense(true_input, true_lb)?;
                                node_linear_bounds.accumulate_dense(false_input, false_lb)?;
                                return Ok(SpecialBatchedDispatchResult::Handled);
                            }
                            Err(e) => {
                                debug!(
                                    "GraphNetwork batched CROWN: Where '{}' constant-mask split \
                                     failed ({}) — keeping IBP fallback",
                                    node_name, e
                                );
                            }
                        }
                    }
                }
                Ok(SpecialBatchedDispatchResult::NotHandled)
            }
            // === Deliberately still refused (audited, NOT an oversight) ======
            //
            // These layers DO have dense CROWN relaxations, but no BATCHED one,
            // and there is no sound way to reach the dense relaxation from here:
            // the batched carrier is `[batch.., out_dim, in_dim]` (per-position,
            // last-axis-only), while every dense relaxation below is written
            // against a flat `[out, in]` `LinearBounds` whose columns index the
            // WHOLE tensor. Re-deriving a batched envelope in this dispatcher —
            // rather than porting the proven per-layer relaxation next to its
            // dense twin and its soundness tests — is exactly the "false proof"
            // this file must not contain. Each stays `NotHandled`, i.e. the
            // existing sound IBP/partial-CROWN fallback:
            //
            // * `Div`  — the dense path does not use the canonical dispatcher at
            //   all: it has a SITE-SPECIFIC handler in the DAG-CROWN coordinator
            //   (`backward_div_to_numerator_with_deadline`) that either scales the
            //   carrier by a positive-denominator reciprocal envelope or
            //   concretizes the node. Its "concretize" branch needs the same
            //   bias-to-network-input channel the batched accumulator lacks.
            // * `MinBinary` / `MaxBinary` — dense `propagate_linear_binary`
            //   builds a piecewise-linear concave/convex envelope over the joint
            //   (x, y) box and splits it across two flat operands.
            // * `Atan2` — dense `propagate_linear_binary` builds a C¹ envelope
            //   that refuses near the origin and across the branch cut.
            // * `Gather`, `AveragePool`, `MaxPool2d` — these are UNARY, so they
            //   never reach this binary dispatcher; they route to
            //   `dispatch_batched_unary`, and the gap is at the layer level:
            //   `Layer::propagate_crown_backward_batched` lists all three in its
            //   explicit unsupported arm. Fixing them means adding
            //   `propagate_linear_batched*` to those layers (they are gather /
            //   scatter / spatial ops whose backward is an index map over the
            //   FLAT tensor, not over a last axis), not adding an arm here.
            //
            // EVERY REMAINING VARIANT IS LISTED EXPLICITLY — there is NO
            // catch-all. This is the compile-time half of the dispatch-coverage
            // guard (`network::dispatch_matrix`): a `_ =>` arm here is exactly
            // how `Where`, `Concat` and `Sub` each spent months silently taking
            // the generic fallback while the DENSE dispatcher handled them.
            // Adding a `Layer` variant is now a compile error until someone
            // decides, in this file, whether the batched graph path handles it.
            // Nothing below changes behaviour: every arm returns the same
            // `NotHandled` the catch-all did.
            Layer::Div(_)
            | Layer::MinBinary(_)
            | Layer::MaxBinary(_)
            | Layer::Atan2(_)
            // Unary and shape/constant ops: they never reach this binary
            // dispatcher at all — the driver routes them to
            // `dispatch_batched_unary`, whose coverage is the exhaustive match
            // in `Layer::propagate_crown_backward_batched`.
            | Layer::Linear(_)
            | Layer::IntervalConst(_)
            | Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::Conv3d(_)
            | Layer::ConvTranspose1d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::AveragePool(_)
            | Layer::MaxPool2d(_)
            | Layer::ReLU(_)
            | Layer::LeakyReLU(_)
            | Layer::Clip(_)
            | Layer::Elu(_)
            | Layer::Selu(_)
            | Layer::PRelu(_)
            | Layer::HardSigmoid(_)
            | Layer::HardSwish(_)
            | Layer::SiLU(_)
            | Layer::Exp(_)
            | Layer::Log(_)
            | Layer::Celu(_)
            | Layer::Mish(_)
            | Layer::LogSoftmax(_)
            | Layer::LogSumExp(_)
            | Layer::ThresholdedRelu(_)
            | Layer::Shrink(_)
            | Layer::Softsign(_)
            | Layer::Snake(_)
            | Layer::Softmax(_)
            | Layer::CausalSoftmax(_)
            | Layer::SelfAttention(_)
            | Layer::GELU(_)
            | Layer::LayerNorm(_)
            | Layer::RmsNorm(_)
            | Layer::InstanceNorm1d(_)
            | Layer::GroupNorm(_)
            | Layer::AdaIN1d(_)
            | Layer::BatchNorm(_)
            | Layer::AddConstant(_)
            | Layer::Transpose(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::Pad(_)
            | Layer::MulConstant(_)
            | Layer::Abs(_)
            | Layer::Sqrt(_)
            | Layer::DivConstant(_)
            | Layer::SubConstant(_)
            | Layer::PowConstant(_)
            | Layer::ReduceMean(_)
            | Layer::ReduceSum(_)
            | Layer::ReduceMax(_)
            | Layer::ReduceMin(_)
            | Layer::Topk(_)
            | Layer::ArgMax(_)
            | Layer::ArgMin(_)
            | Layer::ArgSort(_)
            | Layer::CumSum(_)
            | Layer::Tanh(_)
            | Layer::Sigmoid(_)
            | Layer::Erf(_)
            | Layer::Softplus(_)
            | Layer::Sin(_)
            | Layer::Cos(_)
            | Layer::Tan(_)
            | Layer::Arctan(_)
            | Layer::RoPE(_)
            | Layer::Tile(_)
            | Layer::Gather(_)
            | Layer::ScatterAdd(_)
            | Layer::IndexAdd(_)
            | Layer::ScatterNd(_)
            | Layer::Slice(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::Resize(_)
            | Layer::Compare(_)
            | Layer::CompareTensor(_)
            | Layer::NonZero(_)
            | Layer::Floor(_)
            | Layer::Ceil(_)
            | Layer::Round(_)
            | Layer::Trunc(_)
            | Layer::Sign(_)
            | Layer::Reciprocal(_)
            | Layer::SkipMerge(_)
            | Layer::OpaqueSkip(_)
            | Layer::QdqPerturbation(_)
            | Layer::EpsilonExpand(_) => Ok(SpecialBatchedDispatchResult::NotHandled),
        }
    }

    /// Dispatch a unary layer through the standard batched CROWN backward path.
    ///
    /// Returns `Ok(None)` on successful accumulation (continue the loop).
    /// Returns `Ok(Some(bt))` when partial CROWN fallback is triggered (caller
    /// should return this from the main method).
    ///
    /// Extracted to avoid duplication between the main dispatch and the
    /// Conv2d Patches fallback path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_batched_unary(
        layer: &Layer,
        node_name: &str,
        node_lb: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        first_input: &str,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        node_linear_bounds: &mut BatchedCrownAccumulator,
    ) -> Result<Option<BoundedTensor>> {
        if layer.is_diagnostic_point_only() {
            return Err(crate::network::proof_eligibility::diagnostic_only());
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "batched CROWN: deadline exceeded before unary dispatch at '{node_name}'"
            )));
        }
        let propagated =
            layer.propagate_crown_backward_batched(node_lb, Some(pre_activation), engine);
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "batched CROWN: deadline exceeded after unary dispatch at '{node_name}'"
            )));
        }
        match propagated {
            Ok(new_lb) => {
                node_linear_bounds.accumulate_dense(first_input, new_lb)?;
                Ok(None)
            }
            // Catch UnsupportedOp and UnsupportedConfiguration for partial CROWN
            // fallback. NumericalInstability also triggers fallback — non-finite
            // pre-activation bounds should degrade to IBP, not abort verification.
            // ShapeMismatch also triggers fallback — defense-in-depth for layers
            // that still emit ShapeMismatch from batched CROWN (e.g., Resize, Pad).
            Err(
                e @ NyError::UnsupportedOp(_)
                | e @ NyError::UnsupportedConfiguration(_)
                | e @ NyError::ShapeMismatch { .. }
                | e @ NyError::NumericalInstability(_),
            ) => {
                // Check for other pending paths (from upstream binary op splits).
                // If other intermediate nodes or network-input bounds have been
                // accumulated from a different path, partial CROWN fallback at this
                // node would only capture this one path's contribution — producing
                // unsound bounds. Propagate the error so the caller can fall back
                // to IBP entirely.
                let has_other_pending = !node_linear_bounds.is_empty();
                if has_other_pending {
                    debug!(
                        "GraphNetwork batched CROWN: unsupported layer '{}' ({}: {}) \
                         with other pending paths — propagating error",
                        node_name,
                        layer.layer_type(),
                        e
                    );
                    return Err(e);
                }
                debug!(
                    "GraphNetwork batched CROWN: unsupported layer '{}' ({}: {}), using partial CROWN",
                    node_name,
                    layer.layer_type(),
                    e
                );
                let node_ibp_bounds = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!("IBP bounds for node '{}' not found", node_name))
                })?;
                let fallback =
                    Self::partial_crown_fallback(node_lb, node_ibp_bounds, output_shape)?;
                Ok(Some(fallback))
            }
            // Preserve error type — don't wrap in InvalidSpec.
            Err(e) => Err(e),
        }
    }

    /// Try Conv2d Patches fast-path for a single-input Conv2d node, or convert to Dense.
    ///
    /// Phase 4: For single-input Conv2d nodes in Patches mode, uses the unbatched
    /// Patches backward to compose strides/padding without materializing the full dense
    /// A-matrix. This provides the same 222x memory reduction as the unbatched CROWN path.
    ///
    /// Returns `Handled` if Patches succeeded, `PartialFallback` if Dense fallback
    /// triggered partial CROWN, or `DenseBounds` with the converted linear bounds
    /// for standard binary/unary dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_conv2d_patches_or_dense(
        node_bcb: BatchedCrownBounds,
        layer: &Layer,
        num_inputs: usize,
        node_name: &str,
        pre_activation: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        first_input: &str,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        node_linear_bounds: &mut BatchedCrownAccumulator,
        peak_memory_bytes: &mut usize,
    ) -> Result<PatchesDispatchResult> {
        let is_patches = matches!(&node_bcb, BatchedCrownBounds::Patches(_));
        if is_patches && num_inputs == 1 {
            if let Layer::Conv2d(conv) = layer {
                if let BatchedCrownBounds::Patches(pb) = node_bcb {
                    let input_shape_dims = pre_activation.shape();
                    if input_shape_dims.len() >= 3 {
                        let (in_h, in_w) = (
                            input_shape_dims[input_shape_dims.len() - 2],
                            input_shape_dims[input_shape_dims.len() - 1],
                        );
                        let mut conv_with_shape = conv.clone();
                        conv_with_shape.set_input_shape(in_h, in_w);
                        match conv_with_shape
                            .propagate_patches_engine_and_deadline(&pb, engine, deadline)
                        {
                            Ok(result) => {
                                let new_bcb = if deadline.is_some() {
                                    BatchedCrownBounds::from_crown_bounds_with_deadline(
                                        result, deadline,
                                    )?
                                } else {
                                    BatchedCrownBounds::from_crown_bounds(result)?
                                };
                                let node_mem = new_bcb.memory_bytes();
                                *peak_memory_bytes = (*peak_memory_bytes).max(node_mem);
                                debug!(
                                    "GraphNetwork batched CROWN: Conv2d '{}' Patches backward [{}] ({:.1} MB)",
                                    node_name,
                                    if new_bcb.is_patches() { "Patches" } else { "Dense" },
                                    node_mem as f64 / 1_048_576.0
                                );
                                node_linear_bounds.accumulate(first_input, new_bcb)?;
                                return Ok(PatchesDispatchResult::Handled);
                            }
                            Err(e) if e.is_deadline_exceeded() => return Err(e),
                            Err(e) => {
                                debug!(
                                    "GraphNetwork batched CROWN: Conv2d '{}' Patches failed ({}), falling back to Dense",
                                    node_name, e
                                );
                            }
                        }
                    }
                    let fallback_label = if input_shape_dims.len() >= 3 {
                        "crown_batched:conv2d_patches_fallback"
                    } else {
                        "crown_batched:conv2d_small_input"
                    };
                    let node_lb = BatchedCrownBounds::Patches(pb)
                        .into_batched_dense_checked_with_deadline(fallback_label, deadline)?;
                    if let Some(fallback) = Self::dispatch_batched_unary(
                        layer,
                        node_name,
                        &node_lb,
                        pre_activation,
                        engine,
                        deadline,
                        first_input,
                        node_bounds,
                        output_shape,
                        node_linear_bounds,
                    )? {
                        return Ok(PatchesDispatchResult::PartialFallback(Box::new(
                            CrownBackwardResult {
                                bounds: fallback,
                                provenance: BoundsProvenance::Crown,
                            },
                        )));
                    }
                    return Ok(PatchesDispatchResult::Handled);
                }
            }
        }
        Ok(PatchesDispatchResult::DenseBounds(Box::new(
            node_bcb.into_batched_dense_checked_with_deadline(
                "crown_batched:dense_dispatch",
                deadline,
            )?,
        )))
    }
}

#[cfg(test)]
mod global_attention_row_tests {
    use super::*;
    use crate::layers::{AddLayer, FlattenLayer, MulConstantLayer};
    use crate::network::core::graph::dispatch_plan::CrownDispatchPlan;
    use crate::network::NETWORK_INPUT;
    use ndarray::Dimension;
    use num_rational::BigRational;
    use num_traits::Signed;

    fn rational(value: f32) -> BigRational {
        BigRational::from_float(f64::from(value)).unwrap()
    }

    fn assert_bits(left: &ArrayD<f32>, right: &ArrayD<f32>) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    fn global_carrier(shape: &[usize], rows: usize) -> BatchedLinearBounds {
        let columns = checked_shape_product(shape).unwrap();
        let a_shape = [1, 1, 1, rows, columns];
        let lower = ArrayD::from_shape_fn(IxDyn(&a_shape), |i| {
            ((i[3] * 7 + i[4] * 3) % 13) as f32 * 0.125 - 0.75
        });
        let upper = lower.mapv(|value| -value * 0.5);
        let bias_shape = [1, 1, 1, rows];
        let mut result = BatchedLinearBounds::new(
            lower,
            ArrayD::from_shape_fn(IxDyn(&bias_shape), |i| i[3] as f32 * 0.03125 - 0.25),
            upper,
            ArrayD::from_shape_fn(IxDyn(&bias_shape), |i| 0.5 - i[3] as f32 * 0.015625),
            vec![1, columns],
            vec![1, rows],
        )
        .unwrap();
        result.lower_a_err = Some(ArrayD::from_shape_fn(IxDyn(&a_shape), |i| {
            ((i[3] + i[4] * 2) % 5) as f32 * 0.015625
        }));
        result.upper_a_err = Some(ArrayD::from_shape_fn(IxDyn(&a_shape), |i| {
            ((i[3] * 3 + i[4]) % 7) as f32 * 0.0078125
        }));
        result
    }

    #[test]
    fn attention_region_global_flatten_keeps_all_rank_axes_rows_and_signed_error_bits() {
        for shape in [vec![2, 3, 2], vec![2, 2, 3, 2], vec![1, 2, 3, 2, 2]] {
            let input = BoundedTensor::new(
                ArrayD::from_shape_fn(IxDyn(&shape), |i| {
                    i.slice().iter().sum::<usize>() as f32 * 0.125 - 0.5
                }),
                ArrayD::from_shape_fn(IxDyn(&shape), |i| {
                    i.slice().iter().sum::<usize>() as f32 * 0.125 - 0.25
                }),
            )
            .unwrap();
            let original = global_carrier(&shape, 9);
            let saved = original.clone();
            let pointers = [
                original.lower_a.as_ptr(),
                original.upper_a.as_ptr(),
                original.lower_b.as_ptr(),
                original.upper_b.as_ptr(),
                original.lower_a_err.as_ref().unwrap().as_ptr(),
                original.upper_a_err.as_ref().unwrap().as_ptr(),
            ];
            let result = remap_global_flatten_batched(
                &Layer::Flatten(FlattenLayer::flatten_all()),
                original,
                &input,
            )
            .unwrap();
            assert_eq!(result.input_shape(), shape.as_slice());
            assert_eq!(result.lower_a.shape(), [9, input.len()]);
            assert_eq!(result.lower_b.shape(), [9]);
            assert_eq!(result.output_shape(), saved.output_shape());
            for (actual, expected) in [
                (&result.lower_a, &saved.lower_a),
                (&result.upper_a, &saved.upper_a),
                (&result.lower_b, &saved.lower_b),
                (&result.upper_b, &saved.upper_b),
                (
                    result.lower_a_err.as_ref().unwrap(),
                    saved.lower_a_err.as_ref().unwrap(),
                ),
                (
                    result.upper_a_err.as_ref().unwrap(),
                    saved.upper_a_err.as_ref().unwrap(),
                ),
            ] {
                assert_bits(actual, expected);
            }
            assert_eq!(
                pointers,
                [
                    result.lower_a.as_ptr(),
                    result.upper_a.as_ptr(),
                    result.lower_b.as_ptr(),
                    result.upper_b.as_ptr(),
                    result.lower_a_err.as_ref().unwrap().as_ptr(),
                    result.upper_a_err.as_ref().unwrap().as_ptr(),
                ],
                "the exact metadata remap must allocate no coefficient/bias/error copies"
            );
            let concrete = result.concretize_sound(&input).unwrap();
            // Independent exact affine/error arithmetic at original tensor
            // endpoints, with nonzero signed rows and distinct error planes.
            for take_upper in [false, true] {
                let point = if take_upper {
                    input.upper()
                } else {
                    input.lower()
                };
                for row in 0..9 {
                    let mut lower = rational(result.lower_b[[row]]);
                    let mut upper = rational(result.upper_b[[row]]);
                    for (column, &x) in point.iter().enumerate() {
                        let x = rational(x);
                        lower += rational(result.lower_a[[row, column]]) * &x
                            - rational(result.lower_a_err.as_ref().unwrap()[[row, column]])
                                * x.abs();
                        upper += rational(result.upper_a[[row, column]]) * &x
                            + rational(result.upper_a_err.as_ref().unwrap()[[row, column]])
                                * x.abs();
                    }
                    assert!(rational(concrete.lower().as_slice().unwrap()[row]) <= lower);
                    assert!(rational(concrete.upper().as_slice().unwrap()[row]) >= upper);
                }
            }
        }
    }

    #[test]
    fn attention_region_global_flatten_declines_local_rows_and_malformed_extents() {
        let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[2, 3, 2]))).unwrap();
        let flatten = Layer::Flatten(FlattenLayer::flatten_all());
        let local = BatchedLinearBounds::identity(&[2, 6]).unwrap();
        let expected = local.clone();
        let declined = remap_global_flatten_batched(&flatten, local, &input).unwrap();
        assert_eq!(declined.lower_a.shape(), expected.lower_a.shape());
        assert_eq!(declined.input_shape(), expected.input_shape());
        assert_bits(&declined.lower_a, &expected.lower_a);
        let original = global_carrier(input.shape(), 9);
        let authored_local = remap_global_flatten_batched(
            &Layer::Flatten(FlattenLayer::new(1)),
            original.clone(),
            &input,
        )
        .unwrap();
        assert_eq!(authored_local.lower_a.shape(), original.lower_a.shape());
        let wrong_extent = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[2, 2, 2]))).unwrap();
        let declined =
            remap_global_flatten_batched(&flatten, original.clone(), &wrong_extent).unwrap();
        assert_eq!(declined.lower_a.shape(), original.lower_a.shape());
        let mut malformed = original;
        malformed.upper_a_err = Some(ArrayD::zeros(IxDyn(&[1, 1, 1, 9, 11])));
        assert!(remap_global_flatten_batched(&flatten, malformed, &input).is_err());
        // Zero coefficients still preserve a negative lower and positive upper
        // bound, so this metadata adapter cannot create a closing margin.
        let zero = BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(&[1, 3, 12])),
            ArrayD::from_elem(IxDyn(&[1, 3]), -0.25),
            ArrayD::zeros(IxDyn(&[1, 3, 12])),
            ArrayD::from_elem(IxDyn(&[1, 3]), 0.5),
            vec![1, 12],
            vec![1, 3],
        )
        .unwrap();
        let zero = remap_global_flatten_batched(&flatten, zero, &input).unwrap();
        let concrete = zero.concretize_sound(&input).unwrap();
        assert!(concrete.lower().iter().all(|&value| value <= -0.25));
        assert!(concrete.upper().iter().all(|&value| value >= 0.5));
    }

    fn add_dispatch(incoming: &BatchedLinearBounds) -> Result<[BatchedLinearBounds; 2]> {
        let mut graph = GraphNetwork::new();
        for name in ["left", "right"] {
            graph.add_node(GraphNode::from_input(
                name,
                Layer::MulConstant(MulConstantLayer::scalar(1.0)),
            ));
        }
        graph.add_node(GraphNode::binary(
            "sum",
            Layer::Add(AddLayer),
            "left",
            "right",
        ));
        graph.set_output("sum");
        let plan = CrownDispatchPlan::build(&graph)?;
        let mut accumulator = BatchedCrownAccumulator::new(&plan);
        let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 3])))?;
        let boxes = HashMap::from([
            ("left".into(), input.clone()),
            ("right".into(), input.clone()),
        ]);
        let mut runtime = AttentionCompositionRuntime::production();
        let result = graph.dispatch_special_batched_operator(
            graph.nodes.get("sum").unwrap(),
            "sum",
            incoming,
            &input,
            &boxes,
            &[1, 2],
            MulBinaryRelaxationMode::default(),
            None,
            &mut runtime,
            &mut accumulator,
        )?;
        assert!(matches!(result, SpecialBatchedDispatchResult::Handled));
        assert!(accumulator.take(NETWORK_INPUT).is_none());
        Ok([
            accumulator
                .take("left")
                .unwrap()
                .into_batched_dense_checked("Add:left")?,
            accumulator
                .take("right")
                .unwrap()
                .into_batched_dense_checked("Add:right")?,
        ])
    }

    #[test]
    fn attention_region_add_preserves_signed_and_error_only_original_rows() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.0, 0.0, 0.0, -0.5, 0.25, 1.0])
            .unwrap();
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.0, 0.0, 0.0, 0.5, -0.25, -1.0])
                .unwrap();
        let mut incoming = BatchedLinearBounds::new(
            lower,
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, -0.125]).unwrap(),
            upper,
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.25]).unwrap(),
            vec![1, 3],
            vec![1, 2],
        )
        .unwrap();
        incoming.lower_a_err = Some(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.125, 0.25, 0.0, 0.0, 0.125, 0.25])
                .unwrap(),
        );
        incoming.upper_a_err = Some(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.0, 0.25, 0.125, 0.25, 0.0, 0.125])
                .unwrap(),
        );
        let children = add_dispatch(&incoming).unwrap();
        for child in &children {
            assert!(child.has_coeff_err());
            assert!(child.lower_a_err.as_ref().unwrap()[[0, 0, 0]] >= 0.125);
            assert!(child.upper_a_err.as_ref().unwrap()[[0, 0, 2]] >= 0.125);
        }
        let lo = [-0.75, 0.25, -0.5, -0.25, -0.5, 0.125];
        let hi = [0.5, 0.75, 0.25, 0.75, 0.25, 0.625];
        for mask in 0..64 {
            let point: [f32; 6] =
                std::array::from_fn(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] });
            for row in 0..2 {
                let mut original_lower = rational(incoming.lower_b[[0, row]]);
                let mut original_upper = rational(incoming.upper_b[[0, row]]);
                let mut actual_lower = BigRational::from_integer(0.into());
                let mut actual_upper = actual_lower.clone();
                for column in 0..3 {
                    let z = rational(point[column]) + rational(point[3 + column]);
                    original_lower += rational(incoming.lower_a[[0, row, column]]) * &z
                        - rational(incoming.lower_a_err.as_ref().unwrap()[[0, row, column]])
                            * z.abs();
                    original_upper += rational(incoming.upper_a[[0, row, column]]) * &z
                        + rational(incoming.upper_a_err.as_ref().unwrap()[[0, row, column]])
                            * z.abs();
                }
                for (side, child) in children.iter().enumerate() {
                    actual_lower += rational(child.lower_b[[0, row]]);
                    actual_upper += rational(child.upper_b[[0, row]]);
                    for column in 0..3 {
                        let x = rational(point[side * 3 + column]);
                        actual_lower += rational(child.lower_a[[0, row, column]]) * &x
                            - rational(child.lower_a_err.as_ref().unwrap()[[0, row, column]])
                                * x.abs();
                        actual_upper += rational(child.upper_a[[0, row, column]]) * &x
                            + rational(child.upper_a_err.as_ref().unwrap()[[0, row, column]])
                                * x.abs();
                    }
                }
                assert!(
                    actual_lower <= original_lower,
                    "Add lower mask={mask} row={row}"
                );
                assert!(
                    actual_upper >= original_upper,
                    "Add upper mask={mask} row={row}"
                );
            }
        }
    }

    #[test]
    fn attention_region_add_zero_and_malformed_error_controls_fail_closed() {
        let zero = BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(&[1, 2, 3])),
            ArrayD::from_elem(IxDyn(&[1, 2]), -0.5),
            ArrayD::zeros(IxDyn(&[1, 2, 3])),
            ArrayD::from_elem(IxDyn(&[1, 2]), 0.25),
            vec![1, 3],
            vec![1, 2],
        )
        .unwrap();
        let children = add_dispatch(&zero).unwrap();
        assert!(children.iter().all(|child| !child.has_coeff_err()));
        for row in 0..2 {
            assert!(
                rational(children[0].lower_b[[0, row]]) + rational(children[1].lower_b[[0, row]])
                    <= rational(-0.5)
            );
            assert!(
                rational(children[0].upper_b[[0, row]]) + rational(children[1].upper_b[[0, row]])
                    >= rational(0.25)
            );
        }
        let mut malformed = zero.clone();
        malformed.lower_a_err = Some(ArrayD::zeros(IxDyn(&[1, 2, 2])));
        assert!(add_dispatch(&malformed).is_err());
        for value in [f32::NAN, -0.125] {
            let mut invalid = zero.clone();
            invalid.upper_a_err = Some(ArrayD::from_elem(IxDyn(&[1, 2, 3]), value));
            assert!(add_dispatch(&invalid).is_err());
        }
    }

    #[test]
    fn attention_region_add_broadcast_original_operand_declines_without_publication() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "left",
            Layer::MulConstant(MulConstantLayer::scalar(1.0)),
        ));
        graph.add_node(GraphNode::from_input(
            "right",
            Layer::ReduceMean(crate::layers::ReduceMeanLayer::last_axis()),
        ));
        graph.add_node(GraphNode::binary(
            "sum",
            Layer::Add(AddLayer),
            "left",
            "right",
        ));
        graph.set_output("sum");
        let plan = CrownDispatchPlan::build(&graph).unwrap();
        let mut accumulator = BatchedCrownAccumulator::new(&plan);
        let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 3]))).unwrap();
        let right = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 1]))).unwrap();
        let boxes = HashMap::from([("left".into(), input.clone()), ("right".into(), right)]);
        let bounds = BatchedLinearBounds::identity(&[1, 3]).unwrap();
        let mut runtime = AttentionCompositionRuntime::production();
        let result = graph
            .dispatch_special_batched_operator(
                graph.nodes.get("sum").unwrap(),
                "sum",
                &bounds,
                &input,
                &boxes,
                &[1, 3],
                MulBinaryRelaxationMode::default(),
                None,
                &mut runtime,
                &mut accumulator,
            )
            .unwrap();
        assert!(matches!(result, SpecialBatchedDispatchResult::NotHandled));
        assert!(
            accumulator.is_empty(),
            "broadcast columns cannot be copied onto a shorter operand"
        );
        // The default coordinator must follow its checked partial fallback,
        // rather than reaching the old Add-always-handled unreachable arm.
        let published = graph.propagate_crown_batched(&input).unwrap();
        assert!(published.lower().iter().all(|&v| v <= 0.0));
        assert!(published.upper().iter().all(|&v| v >= 0.0));
    }
}

#[cfg(test)]
mod tests {
    use ny_core::NyError;

    use super::{is_bilinear_fallback_eligible_error, is_fallback_eligible_error};

    #[test]
    fn test_unsupported_op_is_fallback_eligible() {
        assert!(is_fallback_eligible_error(&NyError::UnsupportedOp(
            "test".to_string()
        )));
    }

    #[test]
    fn test_shape_mismatch_is_not_base_fallback_eligible() {
        assert!(!is_fallback_eligible_error(&NyError::ShapeMismatch {
            expected: vec![1, 2],
            got: vec![2, 1],
        }));
    }

    #[test]
    fn test_shape_mismatch_is_bilinear_fallback_eligible() {
        assert!(is_bilinear_fallback_eligible_error(
            &NyError::ShapeMismatch {
                expected: vec![1, 2],
                got: vec![2, 1],
            }
        ));
    }

    #[test]
    fn test_invalid_spec_is_not_fallback_eligible() {
        assert!(!is_fallback_eligible_error(&NyError::InvalidSpec(
            "bad node".to_string()
        )));
        assert!(!is_bilinear_fallback_eligible_error(&NyError::InvalidSpec(
            "bad node".to_string()
        )));
    }

    #[test]
    fn test_internal_error_is_not_fallback_eligible() {
        assert!(!is_fallback_eligible_error(&NyError::InternalError(
            "broken invariant".to_string()
        )));
        assert!(!is_bilinear_fallback_eligible_error(
            &NyError::InternalError("broken invariant".to_string())
        ));
    }
}
