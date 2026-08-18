// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Forward-map alpha optimizer (#w4-root-alpha-opt): choose per-neuron ReLU
//! lower slopes that tighten the forward-linear C-margin ROOT bound.
//!
//! # Why this exists
//!
//! W4-7 built the alpha-FED forward map (any per-neuron `α ∈ [0, 1]` composes
//! soundly with intercept 0 on crossing neurons) but proved the alpha WARMUP's
//! slopes are optimized for the GPU-backward relaxation and are ~8-10x LOOSER
//! for the forward map. The forward map needs alphas optimized against its OWN
//! objective: the C-margin lower bound of the unverified (straggler) spec rows.
//!
//! # The surrogate (why this is cheap)
//!
//! A full certified map rebuild is ~20s at cifar100 release scale — the conv
//! coefficient composition (im2col + f64 GEMM over every network-input column)
//! dominates and depends on every upstream ReLU diagonal, so nothing useful is
//! cacheable across alpha changes. Instead of rebuilding per candidate, this
//! module optimizes on a POINT-EVALUATION surrogate:
//!
//! * For the margin row `r`, the fixed-slope composed map gives the
//!   concretization vertex `x*_r` (per input coordinate: the box end selected
//!   by the composed lower-coefficient sign). At `x*_r` the composed lower
//!   bound EQUALS the composed affine lower function evaluated at `x*_r`.
//! * Evaluating the composed lower/upper functions at a point never needs the
//!   O(input_dim) coefficient matrices: carry two VECTORS per node (the lower
//!   and upper affine field values `L_v(x), U_v(x)`) through the same
//!   recurrences the certified pass uses (center-radius conv/dense, diagonal
//!   ReLU, Add). One pass costs a handful of direct convolutions — milliseconds
//!   instead of seconds.
//! * With relaxation slopes and vertices held fixed, the margin value is
//!   MULTILINEAR in the alphas (linear in each coordinate separately), so the
//!   per-coordinate optimum is a vertex of `[0, 1]` and the exact gradient is
//!   one adjoint (backward sensitivity) pass: `∂g/∂α_i = λ_i · L_pred,i(x*)`
//!   — the margin-row sensitivity at the ReLU output times the pre-activation
//!   lower field. Coordinate moves toward the preferred vertex with a halving
//!   step (interior points reachable) plus surrogate re-evaluation give a
//!   guarded ascent that never returns worse-than-adaptive alphas.
//!
//! # Soundness
//!
//! The optimizer is a HEURISTIC that only CHOOSES alphas. Every claimed bound
//! comes from the certified alpha-fed rebuild
//! (`collect_forward_linear_state_cached_with_alphas`) which is sound for ANY
//! `α ∈ [0, 1]`, and the caller intersects element-wise with the fixed-slope
//! candidates. Nothing computed here reaches the verdict directly.

use std::collections::{BTreeMap, HashMap};
use std::mem::size_of;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::bounds::LinearBounds;
use crate::layers::activations::relu::relu_linear_relaxation;
use crate::layers::Layer;

use super::image::{
    resolve_conv_geometry, resolve_conv_transpose_geometry, ConvGeometry, ConvTransposeGeometry,
};
use super::{GraphNetwork, NETWORK_INPUT};

/// Cap on the number of straggler rows carried in the surrogate objective
/// (each row adds one forward-field row to every batched pass).
const MAX_ROWS: usize = 12;
/// Maximum accepted coordinate sweeps.
const MAX_SWEEPS: usize = 6;
/// Smallest step toward the preferred vertex before a sweep gives up.
const MIN_STEP: f64 = 0.12;
/// Hard cap for the surrogate's persistent parameters plus its conservatively
/// estimated peak field working set. This optimizer is optional: refusing a
/// graph leaves the already-certified fixed-slope candidates unchanged.
pub(super) const MAX_SURROGATE_BYTES: usize = 256 * 1024 * 1024;
/// Hard cap on the estimated multiply/add work of one complete forward or
/// adjoint pass. The optimizer deadline bounds the number of such passes.
pub(super) const MAX_SURROGATE_PASS_MACS: u64 = 1u64 << 32;
/// Long kernels poll the deadline at least this often. A partially-filled
/// scratch array is always dropped on expiry and never reaches an objective.
pub(super) const DEADLINE_POLL_WORK: u64 = 4096;
/// Row-independent optimizer vectors per ReLU at the high-water mark:
/// adaptive/current/candidate alphas and the adjoint gradient. The saved
/// pre-field is row-dependent and accounted separately.
const RELU_VECTOR_COPIES: usize = 4;
/// Center/radius, operator outputs, and DAG clones at the largest node width.
const OPERATOR_SCRATCH_COPIES: usize = 8;
/// Deliberately conservative charge for every retained surrogate node. Besides
/// `SurrNode` itself this covers hash buckets, Vec/String headers, forward and
/// adjoint Option slots, allocator rounding, and one recursion-free execution
/// entry. The optimizer declines before allocating when this structural charge
/// alone reaches the cap.
const STRUCTURAL_BYTES_PER_NODE: usize = 64 * 1024;
/// Additional conservative charge per predecessor edge.
const STRUCTURAL_BYTES_PER_EDGE: usize = 4 * 1024;
#[cfg(test)]
static LAST_IN_KERNEL_DEADLINE_WORK: AtomicU64 = AtomicU64::new(0);

fn deadline_error(context: &str) -> NyError {
    NyError::DeadlineExceeded(format!("alpha-opt: deadline exceeded in {context}"))
}

#[inline]
fn check_deadline(deadline: Option<Instant>, context: &str) -> Result<()> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        Err(deadline_error(context))
    } else {
        Ok(())
    }
}

#[inline]
fn poll_parallel_deadline(
    work: &mut u64,
    cancelled: &AtomicBool,
    deadline: Option<Instant>,
) -> bool {
    *work = work.saturating_add(1);
    if cancelled.load(Ordering::Relaxed) {
        return true;
    }
    if work.is_multiple_of(DEADLINE_POLL_WORK) && deadline.is_some_and(|d| Instant::now() >= d) {
        #[cfg(test)]
        LAST_IN_KERNEL_DEADLINE_WORK.store(*work, Ordering::Relaxed);
        cancelled.store(true, Ordering::Relaxed);
        return true;
    }
    false
}

fn checked_product_usize(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
}

fn checked_product_u64(parts: &[usize]) -> Option<u64> {
    parts.iter().try_fold(1u64, |acc, &value| {
        acc.checked_mul(u64::try_from(value).ok()?)
    })
}

fn reserve_bytes(used: &mut usize, bytes: usize) -> bool {
    match used.checked_add(bytes) {
        Some(next) if next <= MAX_SURROGATE_BYTES => {
            *used = next;
            true
        }
        _ => false,
    }
}

fn reserve_f64_vectors(used: &mut usize, elements: usize, copies: usize) -> bool {
    elements
        .checked_mul(copies)
        .and_then(|count| count.checked_mul(size_of::<f64>()))
        .is_some_and(|bytes| reserve_bytes(used, bytes))
}

fn checked_bytes(elements: usize, copies: usize, element_bytes: usize) -> Option<usize> {
    elements
        .checked_mul(copies)
        .and_then(|count| count.checked_mul(element_bytes))
}

fn checked_add_allocation(
    total: &mut usize,
    elements: usize,
    copies: usize,
    element_bytes: usize,
) -> Option<()> {
    let bytes = checked_bytes(elements, copies, element_bytes)?;
    *total = total.checked_add(bytes)?;
    Some(())
}

fn try_zeroed_f64(len: usize) -> Option<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, 0.0);
    Some(values)
}

fn try_zeroed_bool(len: usize) -> Option<Vec<bool>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, false);
    Some(values)
}

fn try_zeros_array2_f64(rows: usize, cols: usize) -> Option<Array2<f64>> {
    let len = rows.checked_mul(cols)?;
    let values = try_zeroed_f64(len)?;
    Array2::from_shape_vec((rows, cols), values).ok()
}

fn try_zeros_array2_f32(rows: usize, cols: usize) -> Option<Array2<f32>> {
    let len = rows.checked_mul(cols)?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, 0.0f32);
    Array2::from_shape_vec((rows, cols), values).ok()
}

/// Fixed-capacity row selection, so deciding which at-most-12 clauses merit
/// composition performs no heap allocation before the complete resource plan.
#[derive(Debug, Clone, Copy)]
struct SelectedRows {
    len: usize,
    indices: [usize; MAX_ROWS],
    current_lower: [f64; MAX_ROWS],
}

impl SelectedRows {
    fn empty() -> Self {
        Self {
            len: 0,
            indices: [0; MAX_ROWS],
            current_lower: [f64::NEG_INFINITY; MAX_ROWS],
        }
    }

    /// Insert one row in ascending lower-bound order, retaining only MAX_ROWS.
    fn insert(&mut self, row: usize, lower: f64) {
        if self.len == MAX_ROWS && lower >= self.current_lower[MAX_ROWS - 1] {
            return;
        }
        let mut pos = self.len.min(MAX_ROWS - 1);
        if self.len < MAX_ROWS {
            self.len += 1;
        }
        while pos > 0 && lower < self.current_lower[pos - 1] {
            if pos < MAX_ROWS {
                self.indices[pos] = self.indices[pos - 1];
                self.current_lower[pos] = self.current_lower[pos - 1];
            }
            pos -= 1;
        }
        self.indices[pos] = row;
        self.current_lower[pos] = lower;
    }
}

fn select_rows_before_composition(
    spec_rows: usize,
    current_lower: Option<&BoundedTensor>,
    deadline: Option<Instant>,
) -> Result<SelectedRows> {
    let mut selected = SelectedRows::empty();
    if let Some(bounds) = current_lower.filter(|bounds| bounds.len() == spec_rows) {
        for (row, &lower) in bounds.lower().iter().enumerate() {
            if row.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "pre-composition row selection")?;
            }
            let lower = f64::from(lower);
            if lower < 0.0 {
                selected.insert(row, lower);
            }
        }
    } else {
        // Without a certified current candidate there is no cheap ranking
        // signal. Deterministically inspect at most the first MAX_ROWS rather
        // than composing an unbounded number of clauses.
        for row in 0..spec_rows.min(MAX_ROWS) {
            selected.insert(row, f64::NEG_INFINITY);
        }
    }
    check_deadline(deadline, "pre-composition row selection")?;
    Ok(selected)
}

#[derive(Debug, Clone, Copy, Default)]
struct SurrogateScan {
    parameter_bytes: usize,
    construction_bytes: usize,
    structural_bytes: usize,
    sum_node_dims: usize,
    sum_relu_dims: usize,
    max_node_dim: usize,
    pass_work_per_row: u64,
    node_count: usize,
    edge_count: usize,
    relu_count: usize,
    crossing_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct SurrogateResourcePlan {
    scan: SurrogateScan,
    planned_rows: usize,
    planned_bytes: usize,
    composition_work: u64,
}

/// Complete conservative optimizer allocation plan. This deliberately sums
/// allocations from disjoint phases instead of attempting liveness reuse:
/// over-counting is acceptable for an optional heuristic; under-counting is
/// not. No heap allocation occurs before this function accepts the plan.
fn finalize_resource_plan(
    scan: SurrogateScan,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    retained_request_bytes: usize,
) -> Option<SurrogateResourcePlan> {
    if rows == 0 || rows > MAX_ROWS || input_dim == 0 || output_dim == 0 {
        return None;
    }
    let mut bytes = retained_request_bytes
        .checked_add(scan.parameter_bytes)?
        .checked_add(scan.construction_bytes)?
        .checked_add(scan.structural_bytes)?;

    // Forward/adjoint numeric state:
    // - retained lower+upper fields (or adjoints) at every node,
    // - one saved lower pre-field at every ReLU,
    // - bounded max-width center/radius/operator scratch,
    // - row-independent alpha/current/candidate/gradient vectors.
    let per_row_elements = scan
        .sum_node_dims
        .checked_mul(2)?
        .checked_add(scan.sum_relu_dims)?
        .checked_add(scan.max_node_dim.checked_mul(OPERATOR_SCRATCH_COPIES)?)?;
    checked_add_allocation(&mut bytes, per_row_elements, rows, size_of::<f64>())?;
    checked_add_allocation(
        &mut bytes,
        scan.sum_relu_dims,
        RELU_VECTOR_COPIES,
        size_of::<f64>(),
    )?;

    // Pre-composition row copy and the heuristic lower composition.
    checked_add_allocation(
        &mut bytes,
        rows.checked_mul(output_dim)?,
        1,
        size_of::<f32>(),
    )?;
    checked_add_allocation(
        &mut bytes,
        rows.checked_mul(input_dim)?,
        1,
        size_of::<f64>(),
    )?;
    // Flattened input lower/upper vectors plus the point batch.
    checked_add_allocation(&mut bytes, input_dim, 2, size_of::<f64>())?;
    checked_add_allocation(
        &mut bytes,
        rows.checked_mul(input_dim)?,
        1,
        size_of::<f64>(),
    )?;
    // C+/C- and their two adjoint seed clones.
    checked_add_allocation(
        &mut bytes,
        rows.checked_mul(output_dim)?,
        4,
        size_of::<f64>(),
    )?;
    // Baseline/candidate values, row weights, and composition lower values.
    checked_add_allocation(&mut bytes, rows, 4, size_of::<f64>())?;
    // Returned per-ReLU f32 alpha arrays (conservative: every ReLU coordinate).
    checked_add_allocation(&mut bytes, scan.sum_relu_dims, 1, size_of::<f32>())?;

    if bytes > MAX_SURROGATE_BYTES {
        return None;
    }
    let composition_work = checked_product_u64(&[rows, output_dim, input_dim])?
        .checked_add(checked_product_u64(&[rows, output_dim])?)?
        .checked_add(checked_product_u64(&[rows, input_dim])?)?;
    let pass_work = u64::try_from(rows)
        .ok()?
        .checked_mul(scan.pass_work_per_row)?;
    if composition_work > MAX_SURROGATE_PASS_MACS
        || pass_work > MAX_SURROGATE_PASS_MACS
        || composition_work.checked_add(pass_work)? > MAX_SURROGATE_PASS_MACS
    {
        return None;
    }
    Some(SurrogateResourcePlan {
        scan,
        planned_rows: rows,
        planned_bytes: bytes,
        composition_work,
    })
}

/// Summary of one optimizer run (kept `Copy` so the caller can memoize it).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AlphaOptStats {
    /// Surrogate min-row objective at the adaptive starting alphas.
    pub(crate) baseline_min: f64,
    /// Surrogate min-row objective at the returned alphas (>= baseline_min).
    pub(crate) predicted_min: f64,
    /// Accepted sweeps.
    pub(crate) sweeps: usize,
    /// Crossing coordinates moved off their adaptive value.
    pub(crate) moved: usize,
    /// Crossing coordinates strictly inside (0, 1) in the result.
    pub(crate) interior: usize,
    /// Straggler rows in the surrogate objective.
    pub(crate) rows: usize,
}

enum SurrOp {
    Conv {
        /// Kernel, (oc, ic, kh, kw) C-order flat, f64.
        w: Vec<f64>,
        /// |Kernel|, same layout.
        wabs: Vec<f64>,
        /// Per-output-channel bias (empty = no bias).
        bias: Vec<f64>,
        /// Boxed: keeps the variant near the others' size (clippy `large_enum_variant`).
        geo: Box<ConvGeometry>,
    },
    ConvTranspose {
        /// Kernel repacked as contiguous (ic, kh, kw, oc), f64.
        w: Vec<f64>,
        /// |Kernel|, same layout.
        wabs: Vec<f64>,
        /// Per-output-channel bias (empty = no bias).
        bias: Vec<f64>,
        geo: Box<ConvTransposeGeometry>,
    },
    Dense {
        /// Weight, (m, k) row-major flat, f64.
        w: Vec<f64>,
        wabs: Vec<f64>,
        /// Per-row bias (empty = no bias).
        bias: Vec<f64>,
        m: usize,
        k: usize,
    },
    Relu {
        relu_idx: usize,
    },
    BatchNorm {
        /// Shape-expanded nominal diagonal scale and bias. Certified BatchNorm
        /// parameter errors remain solely in the authoritative rebuild.
        scale: Vec<f64>,
        bias: Vec<f64>,
    },
    Add,
    Pass,
}

fn surrogate_op_work_per_row(op: &SurrOp, dim: usize) -> Option<u64> {
    let elementwise = u64::try_from(dim).ok()?;
    match op {
        SurrOp::Conv { geo, .. } => {
            checked_product_u64(&[geo.out_c, geo.out_h, geo.out_w, geo.in_c, geo.kh, geo.kw])?
                .checked_mul(2)?
                .checked_add(elementwise.checked_mul(12)?)
        }
        SurrOp::ConvTranspose { geo, .. } => {
            checked_product_u64(&[geo.in_c, geo.in_h, geo.in_w, geo.out_c, geo.kh, geo.kw])?
                .checked_mul(2)?
                .checked_add(elementwise.checked_mul(12)?)
        }
        SurrOp::Dense { m, k, .. } => checked_product_u64(&[*m, *k])?
            .checked_mul(2)?
            .checked_add(elementwise.checked_mul(12)?),
        SurrOp::Relu { .. } | SurrOp::BatchNorm { .. } => elementwise.checked_mul(4),
        SurrOp::Add => elementwise.checked_mul(2),
        SurrOp::Pass => Some(elementwise),
    }
}

/// Allocation-free surface/resource scan. It intentionally borrows
/// `node_order` instead of materializing a topological-sort Vec; the builder
/// later declines if that insertion order is not topological. Such a decline
/// affects only this heuristic and leaves the certified fixed candidate intact.
fn scan_surrogate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    deadline: Option<Instant>,
) -> Result<Option<SurrogateScan>> {
    check_deadline(deadline, "surrogate resource scan")?;
    if graph.node_order.len() != graph.nodes.len() || graph.node_order.is_empty() {
        return Ok(None);
    }
    let mut scan = SurrogateScan {
        node_count: graph.node_order.len(),
        max_node_dim: input.len(),
        ..SurrogateScan::default()
    };
    scan.structural_bytes = match scan.node_count.checked_mul(STRUCTURAL_BYTES_PER_NODE) {
        Some(bytes) => bytes,
        None => return Ok(None),
    };

    for (i, name) in graph.node_order.iter().enumerate() {
        if i.is_multiple_of(DEADLINE_POLL_WORK as usize) {
            check_deadline(deadline, "surrogate resource scan")?;
        }
        let Some(node) = graph.nodes.get(name) else {
            return Ok(None);
        };
        let Some(bounds) = node_bounds.get(name) else {
            return Ok(None);
        };
        let dim = bounds.len();
        scan.structural_bytes =
            match scan
                .structural_bytes
                .checked_add(name.len())
                .and_then(|bytes| {
                    node.inputs
                        .len()
                        .checked_mul(STRUCTURAL_BYTES_PER_EDGE)
                        .and_then(|edge_bytes| bytes.checked_add(edge_bytes))
                }) {
                Some(bytes) => bytes,
                None => return Ok(None),
            };
        scan.edge_count = match scan.edge_count.checked_add(node.inputs.len()) {
            Some(edges) => edges,
            None => return Ok(None),
        };

        let pred_bounds = |slot: usize| -> Option<&BoundedTensor> {
            match node.inputs.get(slot).map(String::as_str) {
                Some(NETWORK_INPUT) => Some(input),
                Some(pred) => node_bounds.get(pred),
                None => None,
            }
        };
        let pred_dim = |slot: usize| pred_bounds(slot).map_or(0, BoundedTensor::len);

        let node_work = match &node.layer {
            Layer::Conv2d(conv) => {
                if node.inputs.len() != 1 {
                    return Ok(None);
                }
                let Some(pre) = pred_bounds(0) else {
                    return Ok(None);
                };
                let geo = match resolve_conv_geometry(name, conv, pre.shape(), pre.len(), dim) {
                    Ok(geo) => geo,
                    Err(NyError::UnsupportedConfiguration(_) | NyError::ShapeMismatch { .. }) => {
                        return Ok(None)
                    }
                    Err(error) => return Err(error),
                };
                let Some(weight_elements) =
                    checked_product_usize(&[geo.out_c, geo.in_c, geo.kh, geo.kw])
                else {
                    return Ok(None);
                };
                if conv.kernel.len() != weight_elements
                    || conv
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.len() != geo.out_c)
                    || checked_add_allocation(
                        &mut scan.parameter_bytes,
                        weight_elements,
                        2,
                        size_of::<f64>(),
                    )
                    .is_none()
                    || checked_add_allocation(
                        &mut scan.parameter_bytes,
                        conv.bias.as_ref().map_or(0, |bias| bias.len()),
                        1,
                        size_of::<f64>(),
                    )
                    .is_none()
                {
                    return Ok(None);
                }
                checked_product_u64(&[geo.out_c, geo.out_h, geo.out_w, geo.in_c, geo.kh, geo.kw])
                    .and_then(|work| work.checked_mul(2))
                    .and_then(|work| {
                        u64::try_from(dim)
                            .ok()
                            .and_then(|elements| elements.checked_mul(12))
                            .and_then(|elements| work.checked_add(elements))
                    })
            }
            Layer::ConvTranspose2d(conv) => {
                if node.inputs.len() != 1 {
                    return Ok(None);
                }
                let Some(pre) = pred_bounds(0) else {
                    return Ok(None);
                };
                let geo = match resolve_conv_transpose_geometry(
                    name,
                    conv,
                    pre.shape(),
                    pre.len(),
                    dim,
                ) {
                    Ok(geo) => geo,
                    Err(NyError::UnsupportedConfiguration(_) | NyError::ShapeMismatch { .. }) => {
                        return Ok(None)
                    }
                    Err(error) => return Err(error),
                };
                let Some(weight_elements) =
                    checked_product_usize(&[geo.in_c, geo.kh, geo.kw, geo.out_c])
                else {
                    return Ok(None);
                };
                if conv.kernel.len() != weight_elements
                    || conv
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.len() != geo.out_c)
                    || checked_add_allocation(
                        &mut scan.parameter_bytes,
                        weight_elements,
                        2,
                        size_of::<f64>(),
                    )
                    .is_none()
                    || checked_add_allocation(
                        &mut scan.parameter_bytes,
                        conv.bias.as_ref().map_or(0, |bias| bias.len()),
                        1,
                        size_of::<f64>(),
                    )
                    .is_none()
                {
                    return Ok(None);
                }
                checked_product_u64(&[geo.in_c, geo.in_h, geo.in_w, geo.out_c, geo.kh, geo.kw])
                    .and_then(|work| work.checked_mul(2))
                    .and_then(|work| {
                        u64::try_from(dim)
                            .ok()
                            .and_then(|elements| elements.checked_mul(12))
                            .and_then(|elements| work.checked_add(elements))
                    })
            }
            Layer::Linear(linear) => {
                if node.inputs.len() != 1 {
                    return Ok(None);
                }
                let m = linear.weight.nrows();
                let k = linear.weight.ncols();
                let Some(weight_elements) = m.checked_mul(k) else {
                    return Ok(None);
                };
                if m != dim
                    || k != pred_dim(0)
                    || linear.weight.len() != weight_elements
                    || linear.bias.as_ref().is_some_and(|bias| bias.len() != m)
                    || checked_add_allocation(
                        &mut scan.parameter_bytes,
                        weight_elements,
                        2,
                        size_of::<f64>(),
                    )
                    .is_none()
                    || checked_add_allocation(
                        &mut scan.parameter_bytes,
                        linear.bias.as_ref().map_or(0, |bias| bias.len()),
                        1,
                        size_of::<f64>(),
                    )
                    .is_none()
                {
                    return Ok(None);
                }
                checked_product_u64(&[m, k])
                    .and_then(|work| work.checked_mul(2))
                    .and_then(|work| {
                        u64::try_from(dim)
                            .ok()
                            .and_then(|elements| elements.checked_mul(12))
                            .and_then(|elements| work.checked_add(elements))
                    })
            }
            Layer::BatchNorm(_) => {
                if node.inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                if checked_add_allocation(&mut scan.parameter_bytes, dim, 2, size_of::<f64>())
                    .is_none()
                    || checked_add_allocation(
                        &mut scan.construction_bytes,
                        dim,
                        4,
                        size_of::<f32>(),
                    )
                    .is_none()
                {
                    return Ok(None);
                }
                u64::try_from(dim)
                    .ok()
                    .and_then(|elements| elements.checked_mul(4))
            }
            Layer::ReLU(_) => {
                if node.inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                if checked_add_allocation(&mut scan.parameter_bytes, dim, 4, size_of::<f64>())
                    .is_none()
                    || checked_add_allocation(&mut scan.parameter_bytes, dim, 1, size_of::<bool>())
                        .is_none()
                {
                    return Ok(None);
                }
                let Some(pre) = pred_bounds(0) else {
                    return Ok(None);
                };
                for (j, (&lower, &upper)) in pre.lower().iter().zip(pre.upper().iter()).enumerate()
                {
                    if j.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                        check_deadline(deadline, "surrogate ReLU resource scan")?;
                    }
                    let relaxation = relu_linear_relaxation(lower, upper);
                    if [
                        relaxation.lower_slope,
                        relaxation.lower_intercept,
                        relaxation.upper_slope,
                        relaxation.upper_intercept,
                    ]
                    .iter()
                    .any(|value| !value.is_finite())
                    {
                        return Ok(None);
                    }
                    if lower < 0.0 && upper > 0.0 && lower.is_finite() && upper.is_finite() {
                        scan.crossing_count = match scan.crossing_count.checked_add(1) {
                            Some(count) => count,
                            None => return Ok(None),
                        };
                    }
                }
                scan.relu_count = match scan.relu_count.checked_add(1) {
                    Some(count) => count,
                    None => return Ok(None),
                };
                scan.sum_relu_dims = match scan.sum_relu_dims.checked_add(dim) {
                    Some(total) => total,
                    None => return Ok(None),
                };
                u64::try_from(dim)
                    .ok()
                    .and_then(|elements| elements.checked_mul(4))
            }
            Layer::Add(_) => {
                if node.inputs.len() != 2 || pred_dim(0) != dim || pred_dim(1) != dim {
                    return Ok(None);
                }
                u64::try_from(dim)
                    .ok()
                    .and_then(|elements| elements.checked_mul(2))
            }
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
                if node.inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                u64::try_from(dim).ok()
            }
            _ => return Ok(None),
        };
        let Some(node_work) = node_work else {
            return Ok(None);
        };
        scan.sum_node_dims = match scan.sum_node_dims.checked_add(dim) {
            Some(total) => total,
            None => return Ok(None),
        };
        scan.max_node_dim = scan.max_node_dim.max(dim);
        scan.pass_work_per_row = match scan.pass_work_per_row.checked_add(node_work) {
            Some(total) => total,
            None => return Ok(None),
        };
    }
    if scan.crossing_count == 0 {
        return Ok(None);
    }
    let output_name = if graph.output_node.is_empty() {
        graph.node_order.last().map(String::as_str)
    } else {
        Some(graph.output_node.as_str())
    };
    if output_name.is_none_or(|name| !graph.nodes.contains_key(name)) {
        return Ok(None);
    }
    check_deadline(deadline, "surrogate resource scan")?;
    Ok(Some(scan))
}

struct SurrNode {
    /// Predecessors as indices into the exec-ordered node list
    /// (`None` = the network input).
    inputs: Vec<Option<usize>>,
    dim: usize,
    op: SurrOp,
}

/// Fixed relaxation snapshot of one ReLU node (taken from the cached
/// fixed-slope pass's running pre-activation bounds — the same source the
/// certified pass consumed).
struct SurrRelu {
    name: String,
    /// Adaptive lower slope (the α ∈ {0,1} the fixed pass used on crossing
    /// neurons; exact 0/1 slope on stable neurons).
    dl_adaptive: Vec<f64>,
    /// Lower intercept (0 for crossing/stable; kept general).
    cl: Vec<f64>,
    /// Chord upper slope / intercept (never touched by alpha).
    du: Vec<f64>,
    cu: Vec<f64>,
    crossing: Vec<bool>,
}

struct Surrogate {
    nodes: Vec<SurrNode>,
    relus: Vec<SurrRelu>,
    output_idx: usize,
    input_dim: usize,
    /// Accepted before any optimizer-owned heap allocation. Runtime passes may
    /// use fewer rows but never exceed this plan.
    resource_plan: SurrogateResourcePlan,
}

/// Defense-in-depth check that every runtime pass stays within the already
/// accepted complete plan. Returning `false` is a heuristic refusal.
fn surrogate_resources_fit(s: &Surrogate, rows: usize) -> bool {
    rows > 0
        && rows <= s.resource_plan.planned_rows
        && s.resource_plan.planned_bytes <= MAX_SURROGATE_BYTES
        && s.resource_plan.composition_work <= MAX_SURROGATE_PASS_MACS
        && u64::try_from(rows)
            .ok()
            .and_then(|value| value.checked_mul(s.resource_plan.scan.pass_work_per_row))
            .is_some_and(|work| work <= MAX_SURROGATE_PASS_MACS)
}

/// Build the fixed-relaxation surrogate net from the cached fixed-slope pass
/// state. Returns `Ok(None)` (fail open — the fixed candidates stand) when the
/// graph leaves the certified image op surface or any relaxation datum is
/// non-finite.
fn build_surrogate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    resource_plan: SurrogateResourcePlan,
    deadline: Option<Instant>,
) -> Result<Option<Surrogate>> {
    check_deadline(deadline, "surrogate construction")?;
    let exec = &graph.node_order;
    if exec.len() != resource_plan.scan.node_count {
        return Ok(None);
    }
    let mut idx_of: HashMap<&str, usize> = HashMap::new();
    let mut nodes: Vec<SurrNode> = Vec::new();
    let mut relus: Vec<SurrRelu> = Vec::new();
    if idx_of.try_reserve(resource_plan.scan.node_count).is_err()
        || nodes
            .try_reserve_exact(resource_plan.scan.node_count)
            .is_err()
        || relus
            .try_reserve_exact(resource_plan.scan.relu_count)
            .is_err()
    {
        return Ok(None);
    }
    let mut parameter_bytes = 0usize;
    let mut sum_node_dims = 0usize;
    let mut sum_relu_dims = 0usize;
    // Input rows remain live throughout optimization and may be wider than
    // every graph node (for example, an immediately dimension-reducing Linear).
    let mut max_node_dim = input.len();
    let mut pass_work_per_row = 0u64;

    for (i, name) in exec.iter().enumerate() {
        check_deadline(deadline, "surrogate construction")?;
        let node = graph.nodes.get(name).ok_or_else(|| {
            NyError::InvalidSpec(format!("alpha-opt surrogate: unknown node '{name}'"))
        })?;
        let dim = match node_bounds.get(name) {
            Some(b) => b.len(),
            None => return Ok(None),
        };
        let mut inputs = Vec::new();
        if inputs.try_reserve_exact(node.inputs.len()).is_err() {
            return Ok(None);
        }
        for pred in &node.inputs {
            if pred == NETWORK_INPUT {
                inputs.push(None);
            } else {
                match idx_of.get(pred.as_str()) {
                    Some(&p) => inputs.push(Some(p)),
                    None => return Ok(None),
                }
            }
        }
        let pred_bounds = |slot: usize| -> Option<&BoundedTensor> {
            match inputs.get(slot) {
                Some(None) => Some(input),
                Some(Some(p)) => node_bounds.get(&exec[*p]),
                None => None,
            }
        };
        let pred_dim = |slot: usize| -> usize {
            match inputs.get(slot) {
                Some(None) => input.len(),
                Some(Some(p)) => nodes[*p].dim,
                None => 0,
            }
        };

        let op = match &node.layer {
            Layer::Conv2d(conv) => {
                if inputs.len() != 1 {
                    return Ok(None);
                }
                let pred_shape = match &inputs[0] {
                    None => input.shape(),
                    Some(p) => match node_bounds.get(&exec[*p]) {
                        Some(b) => b.shape(),
                        None => return Ok(None),
                    },
                };
                let geo = match resolve_conv_geometry(name, conv, pred_shape, pred_dim(0), dim) {
                    Ok(geo) => geo,
                    Err(NyError::UnsupportedConfiguration(_) | NyError::ShapeMismatch { .. }) => {
                        return Ok(None)
                    }
                    Err(e) => return Err(e),
                };
                let Some(weight_elements) =
                    checked_product_usize(&[geo.out_c, geo.in_c, geo.kh, geo.kw])
                else {
                    return Ok(None);
                };
                if conv.kernel.len() != weight_elements
                    || conv
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.len() != geo.out_c)
                    || !reserve_f64_vectors(&mut parameter_bytes, weight_elements, 2)
                    || !reserve_f64_vectors(
                        &mut parameter_bytes,
                        conv.bias.as_ref().map_or(0, |bias| bias.len()),
                        1,
                    )
                {
                    return Ok(None);
                }
                let (Some(mut w), Some(mut wabs)) = (
                    try_zeroed_f64(weight_elements),
                    try_zeroed_f64(weight_elements),
                ) else {
                    return Ok(None);
                };
                let mut pack_work = 0u64;
                for oc in 0..geo.out_c {
                    for ic in 0..geo.in_c {
                        for ki in 0..geo.kh {
                            for kj in 0..geo.kw {
                                if pack_work.is_multiple_of(DEADLINE_POLL_WORK) {
                                    check_deadline(deadline, "surrogate Conv2d packing")?;
                                }
                                pack_work = pack_work.saturating_add(1);
                                let value = f64::from(conv.kernel[[oc, ic, ki, kj]]);
                                if !value.is_finite() {
                                    return Ok(None);
                                }
                                let index = ((oc * geo.in_c + ic) * geo.kh + ki) * geo.kw + kj;
                                w[index] = value;
                                wabs[index] = value.abs();
                            }
                        }
                    }
                }
                let mut bias = match conv.bias.as_ref() {
                    Some(values) => match try_zeroed_f64(values.len()) {
                        Some(bias) => bias,
                        None => return Ok(None),
                    },
                    None => Vec::new(),
                };
                for (index, (&value, slot)) in conv
                    .bias
                    .iter()
                    .flat_map(|values| values.iter())
                    .zip(bias.iter_mut())
                    .enumerate()
                {
                    if index.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                        check_deadline(deadline, "surrogate Conv2d bias packing")?;
                    }
                    *slot = f64::from(value);
                    if !slot.is_finite() {
                        return Ok(None);
                    }
                }
                SurrOp::Conv {
                    w,
                    wabs,
                    bias,
                    geo: Box::new(geo),
                }
            }
            Layer::ConvTranspose2d(conv) => {
                if inputs.len() != 1 {
                    return Ok(None);
                }
                let pred_shape = match &inputs[0] {
                    None => input.shape(),
                    Some(p) => match node_bounds.get(&exec[*p]) {
                        Some(b) => b.shape(),
                        None => return Ok(None),
                    },
                };
                let geo =
                    match resolve_conv_transpose_geometry(name, conv, pred_shape, pred_dim(0), dim)
                    {
                        Ok(geo) => geo,
                        Err(
                            NyError::UnsupportedConfiguration(_) | NyError::ShapeMismatch { .. },
                        ) => return Ok(None),
                        Err(e) => return Err(e),
                    };
                let Some(weight_elements) =
                    checked_product_usize(&[geo.in_c, geo.kh, geo.kw, geo.out_c])
                else {
                    return Ok(None);
                };
                if conv.kernel.len() != weight_elements
                    || conv
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.len() != geo.out_c)
                    || !reserve_f64_vectors(&mut parameter_bytes, weight_elements, 2)
                    || !reserve_f64_vectors(
                        &mut parameter_bytes,
                        conv.bias.as_ref().map_or(0, |bias| bias.len()),
                        1,
                    )
                {
                    return Ok(None);
                }
                // The runtime ConvTranspose2d type has no groups field; the
                // ONNX converter rejects group != 1 before constructing it.
                let (Some(mut w), Some(mut wabs)) = (
                    try_zeroed_f64(weight_elements),
                    try_zeroed_f64(weight_elements),
                ) else {
                    return Ok(None);
                };
                let mut pack_work = 0u64;
                for ic in 0..geo.in_c {
                    for ki in 0..geo.kh {
                        for kj in 0..geo.kw {
                            let base = ((ic * geo.kh + ki) * geo.kw + kj) * geo.out_c;
                            for oc in 0..geo.out_c {
                                if pack_work.is_multiple_of(DEADLINE_POLL_WORK) {
                                    check_deadline(deadline, "surrogate ConvTranspose2d packing")?;
                                }
                                pack_work = pack_work.saturating_add(1);
                                let value = f64::from(conv.kernel[[ic, oc, ki, kj]]);
                                if !value.is_finite() {
                                    return Ok(None);
                                }
                                w[base + oc] = value;
                                wabs[base + oc] = value.abs();
                            }
                        }
                    }
                }
                let mut bias = match conv.bias.as_ref() {
                    Some(values) => match try_zeroed_f64(values.len()) {
                        Some(bias) => bias,
                        None => return Ok(None),
                    },
                    None => Vec::new(),
                };
                for (index, (&value, slot)) in conv
                    .bias
                    .iter()
                    .flat_map(|values| values.iter())
                    .zip(bias.iter_mut())
                    .enumerate()
                {
                    if index.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                        check_deadline(deadline, "surrogate ConvTranspose2d bias packing")?;
                    }
                    *slot = f64::from(value);
                    if !slot.is_finite() {
                        return Ok(None);
                    }
                }
                SurrOp::ConvTranspose {
                    w,
                    wabs,
                    bias,
                    geo: Box::new(geo),
                }
            }
            Layer::Linear(linear) => {
                if inputs.len() != 1 {
                    return Ok(None);
                }
                let m = linear.weight.nrows();
                let k = linear.weight.ncols();
                if m != dim || k != pred_dim(0) {
                    return Ok(None);
                }
                let Some(weight_elements) = m.checked_mul(k) else {
                    return Ok(None);
                };
                if linear.weight.len() != weight_elements
                    || linear.bias.as_ref().is_some_and(|bias| bias.len() != m)
                    || !reserve_f64_vectors(&mut parameter_bytes, weight_elements, 2)
                    || !reserve_f64_vectors(
                        &mut parameter_bytes,
                        linear.bias.as_ref().map_or(0, |bias| bias.len()),
                        1,
                    )
                {
                    return Ok(None);
                }
                let (Some(mut w), Some(mut wabs)) = (
                    try_zeroed_f64(weight_elements),
                    try_zeroed_f64(weight_elements),
                ) else {
                    return Ok(None);
                };
                let mut pack_work = 0u64;
                for r in 0..m {
                    for c in 0..k {
                        if pack_work.is_multiple_of(DEADLINE_POLL_WORK) {
                            check_deadline(deadline, "surrogate Linear packing")?;
                        }
                        pack_work = pack_work.saturating_add(1);
                        let value = f64::from(linear.weight[[r, c]]);
                        if !value.is_finite() {
                            return Ok(None);
                        }
                        w[r * k + c] = value;
                        wabs[r * k + c] = value.abs();
                    }
                }
                let mut bias = match linear.bias.as_ref() {
                    Some(values) => match try_zeroed_f64(values.len()) {
                        Some(bias) => bias,
                        None => return Ok(None),
                    },
                    None => Vec::new(),
                };
                for (index, (&value, slot)) in linear
                    .bias
                    .iter()
                    .flat_map(|values| values.iter())
                    .zip(bias.iter_mut())
                    .enumerate()
                {
                    if index.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                        check_deadline(deadline, "surrogate Linear bias packing")?;
                    }
                    *slot = f64::from(value);
                    if !slot.is_finite() {
                        return Ok(None);
                    }
                }
                SurrOp::Dense {
                    w,
                    wabs,
                    bias,
                    m,
                    k,
                }
            }
            Layer::BatchNorm(batch_norm) => {
                if inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                let Some(pre) = pred_bounds(0) else {
                    return Ok(None);
                };
                // Account for both retained f64 arrays and the four temporary
                // f32 arrays produced by shape-aware parameter expansion.
                let Some(expanded_temp_bytes) = dim
                    .checked_mul(4)
                    .and_then(|value| value.checked_mul(size_of::<f32>()))
                else {
                    return Ok(None);
                };
                let mut construction_peak = parameter_bytes;
                if !reserve_f64_vectors(&mut construction_peak, dim, 2)
                    || !reserve_bytes(&mut construction_peak, expanded_temp_bytes)
                    || !reserve_f64_vectors(&mut parameter_bytes, dim, 2)
                {
                    return Ok(None);
                }
                let (scale, bias, scale_err, bias_err) =
                    match batch_norm.expanded_affine_parameters(pre.shape(), dim) {
                        Ok(parameters) => parameters,
                        Err(
                            NyError::UnsupportedConfiguration(_)
                            | NyError::ShapeMismatch { .. }
                            | NyError::InvalidSpec(_),
                        ) => return Ok(None),
                        Err(error) => return Err(error),
                    };
                if scale.len() != dim
                    || bias.len() != dim
                    || scale_err.len() != dim
                    || bias_err.len() != dim
                    || scale
                        .iter()
                        .chain(bias.iter())
                        .any(|value| !value.is_finite())
                    || scale_err
                        .iter()
                        .chain(bias_err.iter())
                        .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Ok(None);
                }
                check_deadline(deadline, "surrogate BatchNorm expansion")?;
                let (Some(mut scale64), Some(mut bias64)) =
                    (try_zeroed_f64(dim), try_zeroed_f64(dim))
                else {
                    return Ok(None);
                };
                for j in 0..dim {
                    if j.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                        check_deadline(deadline, "surrogate BatchNorm packing")?;
                    }
                    scale64[j] = f64::from(scale[j]);
                    bias64[j] = f64::from(bias[j]);
                }
                SurrOp::BatchNorm {
                    scale: scale64,
                    bias: bias64,
                }
            }
            Layer::ReLU(_) => {
                if inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                let Some(relu_bytes) = dim
                    .checked_mul(4)
                    .and_then(|value| value.checked_mul(size_of::<f64>()))
                    .and_then(|value| {
                        dim.checked_mul(size_of::<bool>())
                            .and_then(|flags| value.checked_add(flags))
                    })
                else {
                    return Ok(None);
                };
                if !reserve_bytes(&mut parameter_bytes, relu_bytes) {
                    return Ok(None);
                }
                let pre = match pred_bounds(0) {
                    Some(b) => b,
                    None => return Ok(None),
                };
                let (Some(mut dl), Some(mut cl), Some(mut du), Some(mut cu), Some(mut crossing)) = (
                    try_zeroed_f64(dim),
                    try_zeroed_f64(dim),
                    try_zeroed_f64(dim),
                    try_zeroed_f64(dim),
                    try_zeroed_bool(dim),
                ) else {
                    return Ok(None);
                };
                for (j, (&l, &u)) in pre.lower().iter().zip(pre.upper().iter()).enumerate() {
                    if j.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                        check_deadline(deadline, "surrogate ReLU construction")?;
                    }
                    let relax = relu_linear_relaxation(l, u);
                    let vals = [
                        relax.lower_slope,
                        relax.lower_intercept,
                        relax.upper_slope,
                        relax.upper_intercept,
                    ];
                    if vals.iter().any(|v| !v.is_finite()) {
                        return Ok(None);
                    }
                    dl[j] = f64::from(relax.lower_slope);
                    cl[j] = f64::from(relax.lower_intercept);
                    du[j] = f64::from(relax.upper_slope);
                    cu[j] = f64::from(relax.upper_intercept);
                    crossing[j] = l < 0.0 && u > 0.0 && l.is_finite() && u.is_finite();
                }
                relus.push(SurrRelu {
                    name: name.clone(),
                    dl_adaptive: dl,
                    cl,
                    du,
                    cu,
                    crossing,
                });
                SurrOp::Relu {
                    relu_idx: relus.len() - 1,
                }
            }
            Layer::Add(_) => {
                if inputs.len() != 2 || pred_dim(0) != dim || pred_dim(1) != dim {
                    return Ok(None);
                }
                SurrOp::Add
            }
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
                if inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                SurrOp::Pass
            }
            _ => return Ok(None),
        };

        let Some(node_work) = surrogate_op_work_per_row(&op, dim) else {
            return Ok(None);
        };
        let Some(next_sum_dims) = sum_node_dims.checked_add(dim) else {
            return Ok(None);
        };
        let Some(next_pass_work) = pass_work_per_row.checked_add(node_work) else {
            return Ok(None);
        };
        sum_node_dims = next_sum_dims;
        pass_work_per_row = next_pass_work;
        max_node_dim = max_node_dim.max(dim);
        if matches!(&op, SurrOp::Relu { .. }) {
            let Some(next_relu_dims) = sum_relu_dims.checked_add(dim) else {
                return Ok(None);
            };
            sum_relu_dims = next_relu_dims;
        }
        idx_of.insert(exec[i].as_str(), i);
        nodes.push(SurrNode { inputs, dim, op });
    }

    let output_name = if graph.output_node.is_empty() {
        match exec.last() {
            Some(last) => last.as_str(),
            None => return Ok(None),
        }
    } else {
        graph.output_node.as_str()
    };
    let Some(&output_idx) = idx_of.get(output_name) else {
        return Ok(None);
    };
    if parameter_bytes != resource_plan.scan.parameter_bytes
        || sum_node_dims != resource_plan.scan.sum_node_dims
        || sum_relu_dims != resource_plan.scan.sum_relu_dims
        || max_node_dim != resource_plan.scan.max_node_dim
        || pass_work_per_row != resource_plan.scan.pass_work_per_row
        || relus.len() != resource_plan.scan.relu_count
    {
        return Ok(None);
    }

    Ok(Some(Surrogate {
        nodes,
        relus,
        output_idx,
        input_dim: input.len(),
        resource_plan,
    }))
}

/// Direct batched conv: `out[r] = conv(x[r])` for each row of `xs`
/// (`(rows, conv_in)` → `(rows, conv_out)`), plain f64, no bias.
fn conv_apply_batch(
    w: &[f64],
    geo: &ConvGeometry,
    xs: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    check_deadline(deadline, "Conv2d field application")?;
    let rows = xs.nrows();
    let conv_out = checked_product_usize(&[geo.out_c, geo.out_h, geo.out_w]).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d output size overflow".into())
    })?;
    let in_spatial = geo.in_h.checked_mul(geo.in_w).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d input spatial size overflow".into())
    })?;
    let out_spatial = geo.out_h.checked_mul(geo.out_w).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d output spatial size overflow".into())
    })?;
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let xs_flat = xs.as_slice().expect("row-major xs");
    let conv_in = geo.in_c.checked_mul(in_spatial).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d input size overflow".into())
    })?;
    let expected_w = geo.out_c.checked_mul(geo.contraction).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d kernel size overflow".into())
    })?;
    if xs.ncols() != conv_in || w.len() != expected_w {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, conv_in],
            got: xs.shape().to_vec(),
        });
    }
    let mut out = Array2::<f64>::zeros((rows, conv_out));
    if rows == 0 || conv_out == 0 {
        return Ok(out);
    }
    let cancelled = AtomicBool::new(false);
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(conv_out)
        .enumerate()
        .for_each(|(r, orow)| {
            let mut work = 0u64;
            if deadline.is_some_and(|d| Instant::now() >= d) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            let x = &xs_flat[r * conv_in..(r + 1) * conv_in];
            for oc in 0..geo.out_c {
                if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                    return;
                }
                let w_oc = &w[oc * geo.contraction..(oc + 1) * geo.contraction];
                for oh in 0..geo.out_h {
                    if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                        return;
                    }
                    for ow in 0..geo.out_w {
                        if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                            return;
                        }
                        let mut acc = 0.0f64;
                        for ic in 0..geo.in_c {
                            if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                return;
                            }
                            let x_ic = &x[ic * in_spatial..(ic + 1) * in_spatial];
                            let w_ic = &w_oc[ic * geo.kh * geo.kw..(ic + 1) * geo.kh * geo.kw];
                            for ki in 0..geo.kh {
                                if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                    return;
                                }
                                let ih = (oh * sh + ki * dh) as isize - ph as isize;
                                if ih < 0 || ih >= geo.in_h as isize {
                                    continue;
                                }
                                let x_row = &x_ic[ih as usize * geo.in_w..];
                                let w_row = &w_ic[ki * geo.kw..(ki + 1) * geo.kw];
                                for kj in 0..geo.kw {
                                    if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                        return;
                                    }
                                    let iw = (ow * sw + kj * dw) as isize - pw as isize;
                                    if iw < 0 || iw >= geo.in_w as isize {
                                        continue;
                                    }
                                    acc += w_row[kj] * x_row[iw as usize];
                                }
                            }
                        }
                        orow[oc * out_spatial + oh * geo.out_w + ow] = acc;
                    }
                }
            }
        });
    if cancelled.load(Ordering::Relaxed) {
        Err(deadline_error("Conv2d field application"))
    } else {
        check_deadline(deadline, "Conv2d field application")?;
        Ok(out)
    }
}

/// Transposed batched conv (adjoint of [`conv_apply_batch`]):
/// `(rows, conv_out)` sensitivities → `(rows, conv_in)`.
fn conv_apply_batch_t(
    w: &[f64],
    geo: &ConvGeometry,
    lams: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    check_deadline(deadline, "Conv2d adjoint")?;
    let rows = lams.nrows();
    let in_spatial = geo.in_h.checked_mul(geo.in_w).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d input spatial size overflow".into())
    })?;
    let out_spatial = geo.out_h.checked_mul(geo.out_w).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d output spatial size overflow".into())
    })?;
    let conv_in = geo.in_c.checked_mul(in_spatial).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d input size overflow".into())
    })?;
    let conv_out = geo.out_c.checked_mul(out_spatial).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d output size overflow".into())
    })?;
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let lam_flat = lams.as_slice().expect("row-major lams");
    let kernel_spatial = geo.kh.checked_mul(geo.kw).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d kernel spatial size overflow".into())
    })?;
    let expected_w = geo.out_c.checked_mul(geo.contraction).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Conv2d kernel size overflow".into())
    })?;
    if lams.ncols() != conv_out || w.len() != expected_w {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, conv_out],
            got: lams.shape().to_vec(),
        });
    }
    let mut out = Array2::<f64>::zeros((rows, conv_in));
    if rows == 0 || conv_in == 0 {
        return Ok(out);
    }
    let cancelled = AtomicBool::new(false);
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(conv_in)
        .enumerate()
        .for_each(|(r, orow)| {
            let mut work = 0u64;
            if deadline.is_some_and(|d| Instant::now() >= d) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            let lam = &lam_flat[r * conv_out..(r + 1) * conv_out];
            for ic in 0..geo.in_c {
                for ih in 0..geo.in_h {
                    for iw in 0..geo.in_w {
                        if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                            return;
                        }
                        let mut acc = 0.0f64;
                        for ki in 0..geo.kh {
                            if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                return;
                            }
                            let oh_num = ih as isize + ph as isize - (ki * dh) as isize;
                            if oh_num < 0 {
                                continue;
                            }
                            let oh_num = oh_num as usize;
                            if !oh_num.is_multiple_of(sh) {
                                continue;
                            }
                            let oh = oh_num / sh;
                            if oh >= geo.out_h {
                                continue;
                            }
                            for kj in 0..geo.kw {
                                if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                    return;
                                }
                                let ow_num = iw as isize + pw as isize - (kj * dw) as isize;
                                if ow_num < 0 {
                                    continue;
                                }
                                let ow_num = ow_num as usize;
                                if !ow_num.is_multiple_of(sw) {
                                    continue;
                                }
                                let ow = ow_num / sw;
                                if ow >= geo.out_w {
                                    continue;
                                }
                                let k_idx = (ic * geo.kh + ki) * geo.kw + kj;
                                let mut w_idx = k_idx;
                                let mut l_idx = oh * geo.out_w + ow;
                                for _oc in 0..geo.out_c {
                                    if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                        return;
                                    }
                                    acc += w[w_idx] * lam[l_idx];
                                    w_idx += geo.in_c * kernel_spatial;
                                    l_idx += out_spatial;
                                }
                            }
                        }
                        orow[ic * in_spatial + ih * geo.in_w + iw] = acc;
                    }
                }
            }
        });
    if cancelled.load(Ordering::Relaxed) {
        Err(deadline_error("Conv2d adjoint"))
    } else {
        check_deadline(deadline, "Conv2d adjoint")?;
        Ok(out)
    }
}

/// Batched ConvTranspose2d scatter. `w` is packed `(ic, kh, kw, oc)`.
fn conv_transpose_apply_batch(
    w: &[f64],
    geo: &ConvTransposeGeometry,
    xs: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    check_deadline(deadline, "ConvTranspose2d field application")?;
    let rows = xs.nrows();
    let in_size = checked_product_usize(&[geo.in_c, geo.in_h, geo.in_w]).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt ConvTranspose2d input overflow".into())
    })?;
    let out_size = checked_product_usize(&[geo.out_c, geo.out_h, geo.out_w]).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt ConvTranspose2d output overflow".into())
    })?;
    let Some(expected_w) = checked_product_usize(&[geo.in_c, geo.kh, geo.kw, geo.out_c]) else {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-opt ConvTranspose2d kernel size overflow".into(),
        ));
    };
    if xs.ncols() != in_size || w.len() != expected_w {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, in_size],
            got: xs.shape().to_vec(),
        });
    }
    let xs_flat = xs.as_slice().expect("row-major xs");
    let mut out = Array2::<f64>::zeros((rows, out_size));
    if rows == 0 || out_size == 0 {
        return Ok(out);
    }
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let out_spatial = geo.out_h.checked_mul(geo.out_w).ok_or_else(|| {
        NyError::UnsupportedConfiguration(
            "alpha-opt ConvTranspose2d output spatial size overflow".into(),
        )
    })?;
    let cancelled = AtomicBool::new(false);
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(out_size)
        .enumerate()
        .for_each(|(r, orow)| {
            let mut work = 0u64;
            if deadline.is_some_and(|d| Instant::now() >= d) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            let x = &xs_flat[r * in_size..(r + 1) * in_size];
            for ic in 0..geo.in_c {
                for ih in 0..geo.in_h {
                    for iw in 0..geo.in_w {
                        if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                            return;
                        }
                        let value = x[(ic * geo.in_h + ih) * geo.in_w + iw];
                        if value == 0.0 {
                            continue;
                        }
                        for ki in 0..geo.kh {
                            if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                return;
                            }
                            let oh = (ih * sh + ki * dh) as isize - ph as isize;
                            if oh < 0 || oh >= geo.out_h as isize {
                                continue;
                            }
                            for kj in 0..geo.kw {
                                if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                    return;
                                }
                                let ow = (iw * sw + kj * dw) as isize - pw as isize;
                                if ow < 0 || ow >= geo.out_w as isize {
                                    continue;
                                }
                                let cell = oh as usize * geo.out_w + ow as usize;
                                let wbase = ((ic * geo.kh + ki) * geo.kw + kj) * geo.out_c;
                                for oc in 0..geo.out_c {
                                    if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                        return;
                                    }
                                    orow[oc * out_spatial + cell] += w[wbase + oc] * value;
                                }
                            }
                        }
                    }
                }
            }
        });
    if cancelled.load(Ordering::Relaxed) {
        Err(deadline_error("ConvTranspose2d field application"))
    } else {
        check_deadline(deadline, "ConvTranspose2d field application")?;
        Ok(out)
    }
}

/// Exact adjoint of [`conv_transpose_apply_batch`]. Output-padding-only cells
/// are never visited by the shared scatter coordinates and therefore
/// contribute exactly zero upstream.
fn conv_transpose_apply_batch_t(
    w: &[f64],
    geo: &ConvTransposeGeometry,
    lams: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    check_deadline(deadline, "ConvTranspose2d adjoint")?;
    let rows = lams.nrows();
    let in_size = checked_product_usize(&[geo.in_c, geo.in_h, geo.in_w]).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt ConvTranspose2d input overflow".into())
    })?;
    let out_size = checked_product_usize(&[geo.out_c, geo.out_h, geo.out_w]).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt ConvTranspose2d output overflow".into())
    })?;
    let Some(expected_w) = checked_product_usize(&[geo.in_c, geo.kh, geo.kw, geo.out_c]) else {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-opt ConvTranspose2d kernel size overflow".into(),
        ));
    };
    if lams.ncols() != out_size || w.len() != expected_w {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, out_size],
            got: lams.shape().to_vec(),
        });
    }
    let lam_flat = lams.as_slice().expect("row-major lams");
    let mut out = Array2::<f64>::zeros((rows, in_size));
    if rows == 0 || in_size == 0 {
        return Ok(out);
    }
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let out_spatial = geo.out_h.checked_mul(geo.out_w).ok_or_else(|| {
        NyError::UnsupportedConfiguration(
            "alpha-opt ConvTranspose2d output spatial size overflow".into(),
        )
    })?;
    let cancelled = AtomicBool::new(false);
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(in_size)
        .enumerate()
        .for_each(|(r, orow)| {
            let mut work = 0u64;
            if deadline.is_some_and(|d| Instant::now() >= d) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            let lam = &lam_flat[r * out_size..(r + 1) * out_size];
            for ic in 0..geo.in_c {
                for ih in 0..geo.in_h {
                    for iw in 0..geo.in_w {
                        if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                            return;
                        }
                        let mut acc = 0.0f64;
                        for ki in 0..geo.kh {
                            if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                return;
                            }
                            let oh = (ih * sh + ki * dh) as isize - ph as isize;
                            if oh < 0 || oh >= geo.out_h as isize {
                                continue;
                            }
                            for kj in 0..geo.kw {
                                if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                    return;
                                }
                                let ow = (iw * sw + kj * dw) as isize - pw as isize;
                                if ow < 0 || ow >= geo.out_w as isize {
                                    continue;
                                }
                                let cell = oh as usize * geo.out_w + ow as usize;
                                let wbase = ((ic * geo.kh + ki) * geo.kw + kj) * geo.out_c;
                                for oc in 0..geo.out_c {
                                    if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                                        return;
                                    }
                                    acc += w[wbase + oc] * lam[oc * out_spatial + cell];
                                }
                            }
                        }
                        orow[(ic * geo.in_h + ih) * geo.in_w + iw] = acc;
                    }
                }
            }
        });
    if cancelled.load(Ordering::Relaxed) {
        Err(deadline_error("ConvTranspose2d adjoint"))
    } else {
        check_deadline(deadline, "ConvTranspose2d adjoint")?;
        Ok(out)
    }
}

fn dense_apply_batch(
    w: &[f64],
    m: usize,
    k: usize,
    xs: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    check_deadline(deadline, "Linear field application")?;
    let rows = xs.nrows();
    let expected_w = m.checked_mul(k).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Linear weight size overflow".into())
    })?;
    if xs.ncols() != k || w.len() != expected_w {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, k],
            got: xs.shape().to_vec(),
        });
    }
    let xs_flat = xs.as_slice().expect("row-major xs");
    let mut out = Array2::<f64>::zeros((rows, m));
    if rows == 0 || m == 0 {
        return Ok(out);
    }
    let cancelled = AtomicBool::new(false);
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(r, orow)| {
            let mut work = 0u64;
            if deadline.is_some_and(|d| Instant::now() >= d) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            let x = &xs_flat[r * k..(r + 1) * k];
            for (i, o) in orow.iter_mut().enumerate() {
                let wr = &w[i * k..(i + 1) * k];
                let mut acc = 0.0;
                for (&weight, &value) in wr.iter().zip(x.iter()) {
                    if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                        return;
                    }
                    acc += weight * value;
                }
                *o = acc;
            }
        });
    if cancelled.load(Ordering::Relaxed) {
        Err(deadline_error("Linear field application"))
    } else {
        check_deadline(deadline, "Linear field application")?;
        Ok(out)
    }
}

fn dense_apply_batch_t(
    w: &[f64],
    m: usize,
    k: usize,
    lams: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    check_deadline(deadline, "Linear adjoint")?;
    let rows = lams.nrows();
    let expected_w = m.checked_mul(k).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt Linear weight size overflow".into())
    })?;
    if lams.ncols() != m || w.len() != expected_w {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, m],
            got: lams.shape().to_vec(),
        });
    }
    let lam_flat = lams.as_slice().expect("row-major lams");
    let mut out = Array2::<f64>::zeros((rows, k));
    if rows == 0 || k == 0 {
        return Ok(out);
    }
    let cancelled = AtomicBool::new(false);
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(r, orow)| {
            let mut work = 0u64;
            if deadline.is_some_and(|d| Instant::now() >= d) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            let lam = &lam_flat[r * m..(r + 1) * m];
            for (i, &l) in lam.iter().enumerate() {
                if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                    return;
                }
                if l == 0.0 {
                    continue;
                }
                let wr = &w[i * k..(i + 1) * k];
                for (o, &wv) in orow.iter_mut().zip(wr.iter()) {
                    if poll_parallel_deadline(&mut work, &cancelled, deadline) {
                        return;
                    }
                    *o += l * wv;
                }
            }
        });
    if cancelled.load(Ordering::Relaxed) {
        Err(deadline_error("Linear adjoint"))
    } else {
        check_deadline(deadline, "Linear adjoint")?;
        Ok(out)
    }
}

/// Center-radius affine field step shared by Conv/ConvTranspose/Dense:
/// `L' = A·c − |A|·r + b`, `U' = A·c + |A|·r + b` with `c = (L+U)/2`,
/// `r = (U−L)/2` — the same algebra the certified compositions use.
fn affine_fields(
    apply: impl Fn(&Array2<f64>, bool) -> Result<Array2<f64>>,
    l: &Array2<f64>,
    u: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    check_deadline(deadline, "affine center-radius preparation")?;
    let c = (l + u) * 0.5;
    let r = (u - l) * 0.5;
    check_deadline(deadline, "affine center-radius preparation")?;
    let gc = apply(&c, false)?;
    let mut radius_zero = true;
    for (index, value) in r.iter().enumerate() {
        if index.is_multiple_of(DEADLINE_POLL_WORK as usize) {
            check_deadline(deadline, "affine radius scan")?;
        }
        if *value != 0.0 {
            radius_zero = false;
            break;
        }
    }
    if radius_zero {
        Ok((gc.clone(), gc))
    } else {
        let gr = apply(&r, true)?;
        let result = (&gc - &gr, &gc + &gr);
        check_deadline(deadline, "affine center-radius combination")?;
        Ok(result)
    }
}

fn add_spatial_bias(
    fields: &mut Array2<f64>,
    out_c: usize,
    spatial: usize,
    bias: &[f64],
    deadline: Option<Instant>,
) -> Result<()> {
    if bias.is_empty() {
        return Ok(());
    }
    let expected_fields = out_c.checked_mul(spatial).ok_or_else(|| {
        NyError::UnsupportedConfiguration("alpha-opt spatial bias size overflow".into())
    })?;
    if bias.len() != out_c || fields.ncols() != expected_fields {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c, spatial],
            got: vec![bias.len(), fields.ncols()],
        });
    }
    let mut work = 0usize;
    for mut row in fields.rows_mut() {
        for oc in 0..out_c {
            let b = bias[oc];
            for s in 0..spatial {
                if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                    check_deadline(deadline, "spatial bias application")?;
                }
                work = work.saturating_add(1);
                row[oc * spatial + s] += b;
            }
        }
    }
    Ok(())
}

fn add_dense_bias(fields: &mut Array2<f64>, bias: &[f64], deadline: Option<Instant>) -> Result<()> {
    if bias.is_empty() {
        return Ok(());
    }
    if fields.ncols() != bias.len() {
        return Err(NyError::ShapeMismatch {
            expected: vec![fields.ncols()],
            got: vec![bias.len()],
        });
    }
    let mut work = 0usize;
    for mut row in fields.rows_mut() {
        for (v, b) in row.iter_mut().zip(bias.iter()) {
            if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "dense bias application")?;
            }
            work = work.saturating_add(1);
            *v += b;
        }
    }
    Ok(())
}

fn batch_norm_fields(
    scale: &[f64],
    bias: &[f64],
    l: &Array2<f64>,
    u: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    if l.raw_dim() != u.raw_dim() || l.ncols() != scale.len() || scale.len() != bias.len() {
        return Err(NyError::ShapeMismatch {
            expected: l.shape().to_vec(),
            got: u.shape().to_vec(),
        });
    }
    let mut nl = Array2::<f64>::zeros(l.raw_dim());
    let mut nu = Array2::<f64>::zeros(u.raw_dim());
    let mut work = 0usize;
    for r in 0..l.nrows() {
        for j in 0..l.ncols() {
            if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "BatchNorm field application")?;
            }
            work = work.saturating_add(1);
            let s = scale[j];
            let b = bias[j];
            if s >= 0.0 {
                nl[[r, j]] = s * l[[r, j]] + b;
                nu[[r, j]] = s * u[[r, j]] + b;
            } else {
                // Negative diagonal scale reverses the affine sides.
                nl[[r, j]] = s * u[[r, j]] + b;
                nu[[r, j]] = s * l[[r, j]] + b;
            }
        }
    }
    check_deadline(deadline, "BatchNorm field application")?;
    Ok((nl, nu))
}

fn batch_norm_adjoint(
    scale: &[f64],
    lam_l: &Array2<f64>,
    lam_u: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    if lam_l.raw_dim() != lam_u.raw_dim() || lam_l.ncols() != scale.len() {
        return Err(NyError::ShapeMismatch {
            expected: lam_l.shape().to_vec(),
            got: lam_u.shape().to_vec(),
        });
    }
    let mut nl = Array2::<f64>::zeros(lam_l.raw_dim());
    let mut nu = Array2::<f64>::zeros(lam_u.raw_dim());
    let mut work = 0usize;
    for r in 0..lam_l.nrows() {
        for j in 0..lam_l.ncols() {
            if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "BatchNorm adjoint")?;
            }
            work = work.saturating_add(1);
            let s = scale[j];
            if s >= 0.0 {
                nl[[r, j]] = s * lam_l[[r, j]];
                nu[[r, j]] = s * lam_u[[r, j]];
            } else {
                // L_out=s*U_in+b and U_out=s*L_in+b.
                nl[[r, j]] = s * lam_u[[r, j]];
                nu[[r, j]] = s * lam_l[[r, j]];
            }
        }
    }
    check_deadline(deadline, "BatchNorm adjoint")?;
    Ok((nl, nu))
}

struct ForwardFields {
    out_l: Array2<f64>,
    out_u: Array2<f64>,
    /// Per-ReLU pre-activation LOWER fields (only when requested).
    relu_pre_l: Option<Vec<Array2<f64>>>,
}

/// Evaluate the lower/upper affine fields at the batch of points `xs`
/// (one row per straggler-row vertex), under the given alphas.
fn forward_fields(
    s: &Surrogate,
    alphas: &[Vec<f64>],
    xs: &Array2<f64>,
    want_pre: bool,
    deadline: Option<Instant>,
) -> Result<ForwardFields> {
    let rows = xs.nrows();
    if xs.ncols() != s.input_dim
        || alphas.len() != s.relus.len()
        || !surrogate_resources_fit(s, rows)
    {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-opt surrogate resource/shape preflight refused field pass".into(),
        ));
    }
    check_deadline(deadline, "forward field pass")?;
    let mut fields: Vec<Option<(Array2<f64>, Array2<f64>)>> = vec![None; s.nodes.len()];
    let mut relu_pre_l = want_pre.then(|| Vec::with_capacity(s.relus.len()));

    let fetch = |fields: &Vec<Option<(Array2<f64>, Array2<f64>)>>,
                 slot: &Option<usize>|
     -> Result<(Array2<f64>, Array2<f64>)> {
        match slot {
            None => Ok((xs.clone(), xs.clone())),
            Some(p) => fields[*p]
                .clone()
                .ok_or_else(|| NyError::InternalError("alpha-opt: missing upstream field".into())),
        }
    };

    for (i, node) in s.nodes.iter().enumerate() {
        check_deadline(deadline, "forward field pass")?;
        let out = match &node.op {
            SurrOp::Conv { w, wabs, bias, geo } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                let (mut nl, mut nu) = affine_fields(
                    |m, abs| conv_apply_batch(if abs { wabs } else { w }, geo, m, deadline),
                    &l,
                    &u,
                    deadline,
                )?;
                let spatial = geo.out_h.checked_mul(geo.out_w).ok_or_else(|| {
                    NyError::UnsupportedConfiguration(
                        "alpha-opt Conv2d output spatial size overflow".into(),
                    )
                })?;
                add_spatial_bias(&mut nl, geo.out_c, spatial, bias, deadline)?;
                add_spatial_bias(&mut nu, geo.out_c, spatial, bias, deadline)?;
                (nl, nu)
            }
            SurrOp::ConvTranspose { w, wabs, bias, geo } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                let (mut nl, mut nu) = affine_fields(
                    |m, abs| {
                        conv_transpose_apply_batch(if abs { wabs } else { w }, geo, m, deadline)
                    },
                    &l,
                    &u,
                    deadline,
                )?;
                let spatial = geo.out_h.checked_mul(geo.out_w).ok_or_else(|| {
                    NyError::UnsupportedConfiguration(
                        "alpha-opt ConvTranspose2d output spatial size overflow".into(),
                    )
                })?;
                add_spatial_bias(&mut nl, geo.out_c, spatial, bias, deadline)?;
                add_spatial_bias(&mut nu, geo.out_c, spatial, bias, deadline)?;
                (nl, nu)
            }
            SurrOp::Dense {
                w,
                wabs,
                bias,
                m,
                k,
            } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                let (mut nl, mut nu) = affine_fields(
                    |x, abs| dense_apply_batch(if abs { wabs } else { w }, *m, *k, x, deadline),
                    &l,
                    &u,
                    deadline,
                )?;
                add_dense_bias(&mut nl, bias, deadline)?;
                add_dense_bias(&mut nu, bias, deadline)?;
                (nl, nu)
            }
            SurrOp::BatchNorm { scale, bias } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                batch_norm_fields(scale, bias, &l, &u, deadline)?
            }
            SurrOp::Relu { relu_idx } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                let relu = &s.relus[*relu_idx];
                let alpha = &alphas[*relu_idx];
                if alpha.len() != node.dim {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![node.dim],
                        got: vec![alpha.len()],
                    });
                }
                if let Some(pre) = relu_pre_l.as_mut() {
                    pre.push(l.clone());
                }
                let mut nl = l;
                let mut nu = u;
                let mut work = 0usize;
                for r in 0..rows {
                    for j in 0..node.dim {
                        if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                            check_deadline(deadline, "ReLU field application")?;
                        }
                        work = work.saturating_add(1);
                        let dl = if relu.crossing[j] {
                            alpha[j]
                        } else {
                            relu.dl_adaptive[j]
                        };
                        nl[[r, j]] = dl * nl[[r, j]] + relu.cl[j];
                        nu[[r, j]] = relu.du[j] * nu[[r, j]] + relu.cu[j];
                    }
                }
                (nl, nu)
            }
            SurrOp::Add => {
                let (la, ua) = fetch(&fields, &node.inputs[0])?;
                let (lb, ub) = fetch(&fields, &node.inputs[1])?;
                check_deadline(deadline, "Add field application")?;
                let out = (&la + &lb, &ua + &ub);
                check_deadline(deadline, "Add field application")?;
                out
            }
            SurrOp::Pass => fetch(&fields, &node.inputs[0])?,
        };
        if out
            .0
            .iter()
            .chain(out.1.iter())
            .any(|value| !value.is_finite())
        {
            return Err(NyError::UnsupportedConfiguration(format!(
                "alpha-opt surrogate node {i} produced a non-finite field"
            )));
        }
        fields[i] = Some(out);
    }

    let (out_l, out_u) = fields[s.output_idx]
        .take()
        .ok_or_else(|| NyError::InternalError("alpha-opt: output field missing".into()))?;
    Ok(ForwardFields {
        out_l,
        out_u,
        relu_pre_l,
    })
}

/// Per-row surrogate margin values: `g_r = Σ_k C⁺_rk·L_out[r,k] + C⁻_rk·U_out[r,k]`.
fn margin_values(
    out_l: &Array2<f64>,
    out_u: &Array2<f64>,
    cpos: &Array2<f64>,
    cneg: &Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    if out_l.raw_dim() != out_u.raw_dim()
        || out_l.raw_dim() != cpos.raw_dim()
        || out_l.raw_dim() != cneg.raw_dim()
    {
        return Err(NyError::ShapeMismatch {
            expected: out_l.shape().to_vec(),
            got: cpos.shape().to_vec(),
        });
    }
    let Some(mut values) = try_zeroed_f64(out_l.nrows()) else {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: out_l.nrows().saturating_mul(size_of::<f64>()),
            budget_bytes: MAX_SURROGATE_BYTES,
            site: "forward alpha surrogate margin values",
        });
    };
    let mut work = 0usize;
    for r in 0..out_l.nrows() {
        let mut acc = 0.0f64;
        for k in 0..out_l.ncols() {
            if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "surrogate margin evaluation")?;
            }
            work = work.saturating_add(1);
            acc += cpos[[r, k]] * out_l[[r, k]] + cneg[[r, k]] * out_u[[r, k]];
        }
        if !acc.is_finite() {
            return Err(NyError::UnsupportedConfiguration(
                "alpha-opt surrogate produced a non-finite margin".into(),
            ));
        }
        values[r] = acc;
    }
    check_deadline(deadline, "surrogate margin evaluation")?;
    Ok(values)
}

/// One adjoint pass: sensitivities of the weighted objective w.r.t. each
/// crossing alpha. Seeds carry the row weights (`λ_L = w_r·C_r⁺`,
/// `λ_U = w_r·C_r⁻`). Returns per-ReLU gradient vectors.
fn alpha_gradients(
    s: &Surrogate,
    alphas: &[Vec<f64>],
    relu_pre_l: &[Array2<f64>],
    seed_l: Array2<f64>,
    seed_u: Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<f64>>> {
    let rows = seed_l.nrows();
    let output_dim = s.nodes[s.output_idx].dim;
    if seed_l.raw_dim() != seed_u.raw_dim()
        || seed_l.ncols() != output_dim
        || alphas.len() != s.relus.len()
        || relu_pre_l.len() != s.relus.len()
        || !surrogate_resources_fit(s, rows)
    {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-opt surrogate resource/shape preflight refused adjoint pass".into(),
        ));
    }
    check_deadline(deadline, "adjoint pass")?;
    let mut lams: Vec<Option<(Array2<f64>, Array2<f64>)>> = vec![None; s.nodes.len()];
    lams[s.output_idx] = Some((seed_l, seed_u));
    let mut grads: Vec<Vec<f64>> = s
        .relus
        .iter()
        .map(|r| vec![0.0f64; r.dl_adaptive.len()])
        .collect();

    let accumulate = |lams: &mut Vec<Option<(Array2<f64>, Array2<f64>)>>,
                      slot: &Option<usize>,
                      contrib: (Array2<f64>, Array2<f64>)|
     -> Result<()> {
        let Some(p) = slot else { return Ok(()) };
        match &mut lams[*p] {
            Some((l, u)) => {
                check_deadline(deadline, "DAG adjoint accumulation")?;
                *l += &contrib.0;
                *u += &contrib.1;
                check_deadline(deadline, "DAG adjoint accumulation")?;
            }
            entry @ None => *entry = Some(contrib),
        }
        Ok(())
    };

    for i in (0..s.nodes.len()).rev() {
        check_deadline(deadline, "adjoint pass")?;
        let Some((lam_l, lam_u)) = lams[i].take() else {
            continue;
        };
        let node = &s.nodes[i];
        match &node.op {
            SurrOp::Conv { w, wabs, geo, .. } => {
                // Adjoint of the center-radius step: with s = λL+λU, d = λU−λL,
                // λL_up = (Aᵀs − |A|ᵀd)/2, λU_up = (Aᵀs + |A|ᵀd)/2.
                let ssum = &lam_l + &lam_u;
                let diff = &lam_u - &lam_l;
                check_deadline(deadline, "Conv2d adjoint preparation")?;
                let tc = conv_apply_batch_t(w, geo, &ssum, deadline)?;
                let tr = conv_apply_batch_t(wabs, geo, &diff, deadline)?;
                accumulate(
                    &mut lams,
                    &node.inputs[0],
                    ((&tc - &tr) * 0.5, (&tc + &tr) * 0.5),
                )?;
            }
            SurrOp::ConvTranspose { w, wabs, geo, .. } => {
                let ssum = &lam_l + &lam_u;
                let diff = &lam_u - &lam_l;
                check_deadline(deadline, "ConvTranspose2d adjoint preparation")?;
                let tc = conv_transpose_apply_batch_t(w, geo, &ssum, deadline)?;
                let tr = conv_transpose_apply_batch_t(wabs, geo, &diff, deadline)?;
                accumulate(
                    &mut lams,
                    &node.inputs[0],
                    ((&tc - &tr) * 0.5, (&tc + &tr) * 0.5),
                )?;
            }
            SurrOp::Dense { w, wabs, m, k, .. } => {
                let ssum = &lam_l + &lam_u;
                let diff = &lam_u - &lam_l;
                check_deadline(deadline, "Linear adjoint preparation")?;
                let tc = dense_apply_batch_t(w, *m, *k, &ssum, deadline)?;
                let tr = dense_apply_batch_t(wabs, *m, *k, &diff, deadline)?;
                accumulate(
                    &mut lams,
                    &node.inputs[0],
                    ((&tc - &tr) * 0.5, (&tc + &tr) * 0.5),
                )?;
            }
            SurrOp::BatchNorm { scale, .. } => {
                let upstream = batch_norm_adjoint(scale, &lam_l, &lam_u, deadline)?;
                accumulate(&mut lams, &node.inputs[0], upstream)?;
            }
            SurrOp::Relu { relu_idx } => {
                let relu = &s.relus[*relu_idx];
                let alpha = &alphas[*relu_idx];
                let pre_l = &relu_pre_l[*relu_idx];
                let grad = &mut grads[*relu_idx];
                if alpha.len() != node.dim || pre_l.nrows() != rows || pre_l.ncols() != node.dim {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![rows, node.dim],
                        got: pre_l.shape().to_vec(),
                    });
                }
                let mut nl = lam_l;
                let mut nu = lam_u;
                let mut work = 0usize;
                for r in 0..rows {
                    for j in 0..node.dim {
                        if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                            check_deadline(deadline, "ReLU adjoint")?;
                        }
                        work = work.saturating_add(1);
                        if relu.crossing[j] {
                            grad[j] += nl[[r, j]] * pre_l[[r, j]];
                        }
                        let dl = if relu.crossing[j] {
                            alpha[j]
                        } else {
                            relu.dl_adaptive[j]
                        };
                        nl[[r, j]] *= dl;
                        nu[[r, j]] *= relu.du[j];
                    }
                }
                accumulate(&mut lams, &node.inputs[0], (nl, nu))?;
            }
            SurrOp::Add => {
                check_deadline(deadline, "Add adjoint")?;
                accumulate(&mut lams, &node.inputs[0], (lam_l.clone(), lam_u.clone()))?;
                accumulate(&mut lams, &node.inputs[1], (lam_l, lam_u))?;
            }
            SurrOp::Pass => {
                accumulate(&mut lams, &node.inputs[0], (lam_l, lam_u))?;
            }
        }
    }
    if grads
        .iter()
        .flat_map(|gradient| gradient.iter())
        .any(|value| !value.is_finite())
    {
        return Err(NyError::UnsupportedConfiguration(
            "alpha-opt surrogate produced a non-finite gradient".into(),
        ));
    }
    Ok(grads)
}

/// Softmin weights over the row values (concentrating on the worst rows).
fn softmin_weights(vals: &[f64]) -> Vec<f64> {
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let temp = ((max - min) / 3.0).max(1e-3);
    let mut w: Vec<f64> = vals.iter().map(|v| (-(v - min) / temp).exp()).collect();
    let sum: f64 = w.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for v in &mut w {
            *v /= sum;
        }
    } else {
        let uniform = 1.0 / w.len().max(1) as f64;
        w.fill(uniform);
    }
    w
}

struct HeuristicLowerComposition {
    coefficients: Array2<f64>,
    lower: Vec<f64>,
}

/// Compose only the selected C rows with the cached output map's LOWER affine
/// side. This is deliberately an f64 heuristic, not a certificate: it chooses
/// concretization vertices and decides whether a fresh authoritative rebuild is
/// worth attempting. The rebuilt/intersected `LinearBounds` remain the only
/// verdict authority.
///
/// Restricting to at most MAX_ROWS makes the composition allocation/work
/// exactly preflightable. Every coefficient product, bias product, and vertex
/// product participates in deadline polling; expiry drops the local scratch and
/// returns no partial composition.
fn compose_selected_lower_heuristic(
    selected_spec: &Array2<f32>,
    output_lb: &LinearBounds,
    input_lower: &[f64],
    input_upper: &[f64],
    deadline: Option<Instant>,
) -> Result<Option<HeuristicLowerComposition>> {
    let rows = selected_spec.nrows();
    let output_dim = selected_spec.ncols();
    let input_dim = input_lower.len();
    if rows == 0
        || rows > MAX_ROWS
        || input_upper.len() != input_dim
        || output_lb.num_outputs() != output_dim
        || output_lb.num_inputs() != input_dim
    {
        return Ok(None);
    }
    check_deadline(deadline, "selected-row lower composition")?;
    let Some(mut coefficients) = try_zeros_array2_f64(rows, input_dim) else {
        return Ok(None);
    };
    let Some(mut lower) = try_zeroed_f64(rows) else {
        return Ok(None);
    };
    let mut work = 0usize;
    for r in 0..rows {
        let mut bias = 0.0f64;
        for k in 0..output_dim {
            if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "selected-row lower bias composition")?;
            }
            work = work.saturating_add(1);
            let c = f64::from(selected_spec[[r, k]]);
            if !c.is_finite() {
                return Ok(None);
            }
            let source = if c >= 0.0 {
                output_lb.lower_b()[k]
            } else {
                output_lb.upper_b()[k]
            };
            bias += c * f64::from(source);
        }
        if !bias.is_finite() {
            return Ok(None);
        }
        let mut vertex = bias;
        for j in 0..input_dim {
            let mut coefficient = 0.0f64;
            for k in 0..output_dim {
                if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                    check_deadline(deadline, "selected-row lower coefficient composition")?;
                }
                work = work.saturating_add(1);
                let c = f64::from(selected_spec[[r, k]]);
                let source = if c >= 0.0 {
                    output_lb.lower_a()[[k, j]]
                } else {
                    output_lb.upper_a()[[k, j]]
                };
                coefficient += c * f64::from(source);
            }
            if !coefficient.is_finite() {
                return Ok(None);
            }
            coefficients[[r, j]] = coefficient;
            if work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "selected-row lower concretization")?;
            }
            work = work.saturating_add(1);
            vertex += coefficient
                * if coefficient >= 0.0 {
                    input_lower[j]
                } else {
                    input_upper[j]
                };
        }
        if !vertex.is_finite() {
            return Ok(None);
        }
        lower[r] = vertex;
    }
    check_deadline(deadline, "selected-row lower composition")?;
    Ok(Some(HeuristicLowerComposition {
        coefficients,
        lower,
    }))
}

/// Optimize per-neuron forward-map lower slopes against the C-margin
/// objective of the unverified (straggler) spec rows. Pure heuristic — see
/// module docs. Returns `Ok(None)` when the surrogate refuses, no row is
/// unverified, or no improvement over the adaptive start is found.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_margin_alphas(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    current_lower: Option<&BoundedTensor>,
    node_bounds: &HashMap<String, BoundedTensor>,
    output_lb: &LinearBounds,
    _engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    retained_request_bytes: usize,
) -> Result<Option<(BTreeMap<String, Array1<f32>>, AlphaOptStats)>> {
    let out_dim = spec_matrix.ncols();
    if spec_matrix.nrows() == 0
        || out_dim != output_lb.num_outputs()
        || input.len() != output_lb.num_inputs()
    {
        return Ok(None);
    }

    // Select at most MAX_ROWS on the stack, then perform the complete
    // allocation/work plan before copying one spec value or building one
    // surrogate node.
    let selected = select_rows_before_composition(spec_matrix.nrows(), current_lower, deadline)?;
    if selected.len == 0 {
        return Ok(None);
    }
    let Some(scan) = scan_surrogate(graph, input, node_bounds, deadline)? else {
        return Ok(None);
    };
    let Some(resource_plan) = finalize_resource_plan(
        scan,
        selected.len,
        input.len(),
        out_dim,
        retained_request_bytes,
    ) else {
        tracing::info!(
            rows = selected.len,
            nodes = scan.node_count,
            parameter_bytes = scan.parameter_bytes,
            structural_bytes = scan.structural_bytes,
            pass_work_per_row = scan.pass_work_per_row,
            "forward-map alpha optimizer: complete pre-composition resource plan refused surrogate"
        );
        return Ok(None);
    };

    let Some(mut selected_spec) = try_zeros_array2_f32(selected.len, out_dim) else {
        return Ok(None);
    };
    let mut copy_work = 0usize;
    for r in 0..selected.len {
        for k in 0..out_dim {
            if copy_work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "selected-row spec copy")?;
            }
            copy_work = copy_work.saturating_add(1);
            selected_spec[[r, k]] = spec_matrix[[selected.indices[r], k]];
        }
    }

    let (Some(mut in_lo), Some(mut in_hi)) =
        (try_zeroed_f64(input.len()), try_zeroed_f64(input.len()))
    else {
        return Ok(None);
    };
    for (j, (&lower, &upper)) in input.lower().iter().zip(input.upper().iter()).enumerate() {
        if j.is_multiple_of(DEADLINE_POLL_WORK as usize) {
            check_deadline(deadline, "surrogate input endpoint copy")?;
        }
        in_lo[j] = f64::from(lower);
        in_hi[j] = f64::from(upper);
        if !in_lo[j].is_finite() || !in_hi[j].is_finite() {
            return Ok(None);
        }
    }
    let Some(composed) =
        compose_selected_lower_heuristic(&selected_spec, output_lb, &in_lo, &in_hi, deadline)?
    else {
        return Ok(None);
    };

    // The current certified candidate and selected-row heuristic composition
    // jointly determine which copied rows remain stragglers. Keep the fixed
    // stack ordering worst-first; no post-composition row Vec is allocated.
    let mut active = SelectedRows::empty();
    for selected_row in 0..selected.len {
        let lower = selected.current_lower[selected_row].max(composed.lower[selected_row]);
        if lower < 0.0 {
            active.insert(selected_row, lower);
        }
    }
    if active.len == 0 {
        return Ok(None);
    }
    let k_rows = active.len;

    let Some(surrogate) = build_surrogate(graph, input, node_bounds, resource_plan, deadline)?
    else {
        return Ok(None);
    };
    if !surrogate_resources_fit(&surrogate, k_rows) {
        return Ok(None);
    }

    // Per-row concretization vertices and C+/C- objective seeds.
    let Some(mut xs) = try_zeros_array2_f64(k_rows, surrogate.input_dim) else {
        return Ok(None);
    };
    let (Some(mut cpos), Some(mut cneg)) = (
        try_zeros_array2_f64(k_rows, out_dim),
        try_zeros_array2_f64(k_rows, out_dim),
    ) else {
        return Ok(None);
    };
    let mut seed_work = 0usize;
    for ri in 0..k_rows {
        let selected_row = active.indices[ri];
        for j in 0..surrogate.input_dim {
            if seed_work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "surrogate vertex construction")?;
            }
            seed_work = seed_work.saturating_add(1);
            xs[[ri, j]] = if composed.coefficients[[selected_row, j]] >= 0.0 {
                in_lo[j]
            } else {
                in_hi[j]
            };
        }
        for k in 0..out_dim {
            if seed_work.is_multiple_of(DEADLINE_POLL_WORK as usize) {
                check_deadline(deadline, "surrogate objective seed construction")?;
            }
            seed_work = seed_work.saturating_add(1);
            let c = f64::from(selected_spec[[selected_row, k]]);
            cpos[[ri, k]] = c.max(0.0);
            cneg[[ri, k]] = c.min(0.0);
        }
    }

    // Adaptive start (exactly the slopes the fixed pass used).
    let alpha0: Vec<Vec<f64>> = surrogate
        .relus
        .iter()
        .map(|r| r.dl_adaptive.clone())
        .collect();

    let eval = |alphas: &[Vec<f64>], want_pre: bool| -> Result<(Vec<f64>, ForwardFields)> {
        let fields = forward_fields(&surrogate, alphas, &xs, want_pre, deadline)?;
        let vals = margin_values(&fields.out_l, &fields.out_u, &cpos, &cneg, deadline)?;
        Ok((vals, fields))
    };

    let (mut best_vals, _) = eval(&alpha0, false)?;
    let baseline_min = best_vals.iter().copied().fold(f64::INFINITY, f64::min);
    if !baseline_min.is_finite() {
        return Ok(None);
    }
    let mut best_alpha = alpha0.clone();
    let mut best_min = baseline_min;
    let mut eta = 1.0f64;
    let mut sweeps = 0usize;

    for _ in 0..MAX_SWEEPS {
        // Gradient of the softmin-weighted objective at the current best.
        let weights = softmin_weights(&best_vals);
        let mut best_g: f64 = best_vals
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum();
        let mut seed_l = cpos.clone();
        let mut seed_u = cneg.clone();
        for (ri, &w) in weights.iter().enumerate() {
            for k in 0..out_dim {
                seed_l[[ri, k]] *= w;
                seed_u[[ri, k]] *= w;
            }
        }
        let (grads, pre_fields) = {
            let (_, fields) = match eval(&best_alpha, true) {
                Ok(res) => res,
                Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
                Err(e) => return Err(e),
            };
            let pre = fields
                .relu_pre_l
                .ok_or_else(|| NyError::InternalError("alpha-opt: pre fields missing".into()))?;
            let grads =
                match alpha_gradients(&surrogate, &best_alpha, &pre, seed_l, seed_u, deadline) {
                    Ok(g) => g,
                    Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
                    Err(e) => return Err(e),
                };
            (grads, pre)
        };
        drop(pre_fields);
        if grads.iter().zip(surrogate.relus.iter()).all(|(g, r)| {
            g.iter()
                .zip(r.crossing.iter())
                .all(|(v, &c)| !c || *v == 0.0)
        }) {
            break;
        }

        // Guarded coordinate move toward the per-coordinate preferred vertex.
        let mut accepted = false;
        let mut step = eta;
        // Geometric line search: `step` halves each rejection, so the loop
        // terminates (0.12 = MIN_STEP > 0; a NaN step exits immediately).
        #[allow(clippy::while_float)]
        while step >= MIN_STEP {
            let mut cand = best_alpha.clone();
            let mut moved_any = false;
            for (ri, relu) in surrogate.relus.iter().enumerate() {
                for j in 0..relu.crossing.len() {
                    if !relu.crossing[j] || grads[ri][j] == 0.0 {
                        continue;
                    }
                    let next = (cand[ri][j] + step * grads[ri][j].signum()).clamp(0.0, 1.0);
                    if next != cand[ri][j] {
                        cand[ri][j] = next;
                        moved_any = true;
                    }
                }
            }
            if !moved_any {
                break;
            }
            let (vals, _) = match eval(&cand, false) {
                Ok(res) => res,
                Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
                Err(e) => return Err(e),
            };
            let g: f64 = vals.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();
            let vmin = vals.iter().copied().fold(f64::INFINITY, f64::min);
            if g.is_finite() && g > best_g + 1e-9 * (1.0 + best_g.abs()) {
                best_alpha = cand;
                best_vals = vals;
                best_g = g;
                best_min = vmin;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
        sweeps += 1;
        eta = (step * 1.5).min(1.0);
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(deadline_error("optimizer sweep completion"));
        }
    }

    // Only worth the ~20s certified rebuild when the surrogate predicts a
    // real improvement of the worst straggler row...
    let improve_tol = 1e-6 * (1.0 + baseline_min.abs());
    // Negated form is deliberate: a NaN best_min/baseline_min makes the `>` false,
    // so we skip the rebuild (fail-closed); `<=` would proceed on NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if sweeps == 0 || !(best_min > baseline_min + improve_tol) {
        return Ok(None);
    }
    // ...AND the optimized map can actually move the INTERSECTION: the
    // caller element-wise intersects with the current root candidate, so the
    // rebuild only helps if some selected row's alpha-map bound exceeds the
    // candidate's. The surrogate value is an OPTIMISTIC (fixed-vertex,
    // fixed-slope) estimate of the true rebuilt bound, so this is a
    // necessary condition — skipping on its failure never discards a win.
    // MEASURED (release, cifar100 @95s, 4 instances): the post-W4-4 GPU
    // per-entry backward dominates the root candidate at −4.3..−9.4 while
    // the optimized forward margin converges at −17..−27; without this gate
    // the rebuild burned ~25s of BaB budget per instance for an intersection
    // no-op.
    let beats_candidate = best_vals.iter().enumerate().any(|(row, &value)| {
        let candidate = active.current_lower[row];
        value > candidate + 1e-6 * (1.0 + candidate.abs())
    });
    if !beats_candidate {
        tracing::info!(
            rows = k_rows,
            sweeps,
            surrogate_baseline_min = baseline_min,
            surrogate_predicted_min = best_min,
            candidate_min = active.current_lower[..k_rows]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            "forward-map alpha optimizer: optimized surrogate cannot beat the current root candidate on any straggler row — skipping the certified rebuild (#w4-root-alpha-opt)"
        );
        return Ok(None);
    }

    let mut moved = 0usize;
    let mut interior = 0usize;
    let mut alphas_out: BTreeMap<String, Array1<f32>> = BTreeMap::new();
    for (ri, relu) in surrogate.relus.iter().enumerate() {
        if !relu.crossing.iter().any(|&c| c) {
            continue;
        }
        let mut arr = Array1::<f32>::from_elem(relu.crossing.len(), f32::NAN);
        for j in 0..relu.crossing.len() {
            if relu.crossing[j] {
                let v = best_alpha[ri][j];
                arr[j] = v as f32;
                if (v - alpha0[ri][j]).abs() > 1e-6 {
                    moved += 1;
                }
                if v > 0.0 && v < 1.0 {
                    interior += 1;
                }
            }
        }
        alphas_out.insert(relu.name.clone(), arr);
    }
    if moved == 0 {
        return Ok(None);
    }

    Ok(Some((
        alphas_out,
        AlphaOptStats {
            baseline_min,
            predicted_min: best_min,
            sweeps,
            moved,
            interior,
            rows: k_rows,
        },
    )))
}

#[cfg(test)]
mod grad_tests {
    use super::*;
    use crate::layers::convolution::conv2d::conv2d_transpose_forward;
    use crate::layers::{ConvTranspose2dLayer, ReLULayer};
    use crate::network::GraphNode;
    use ndarray::{ArrayD, IxDyn};
    use std::time::Duration;

    fn build_surrogate_for_test(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        map: &HashMap<String, BoundedTensor>,
        rows: usize,
    ) -> Surrogate {
        let scan = scan_surrogate(graph, input, map, None)
            .expect("scan must not error")
            .expect("graph must be on surrogate surface");
        let output_name = if graph.output_node.is_empty() {
            graph.node_order.last().expect("output node")
        } else {
            &graph.output_node
        };
        let output_dim = map.get(output_name).expect("output bounds").len();
        let plan = finalize_resource_plan(scan, rows, input.len(), output_dim, 0)
            .expect("test graph must fit complete resource plan");
        build_surrogate(graph, input, map, plan, None)
            .expect("build must not error")
            .expect("planned graph must build")
    }

    fn assert_alpha_gradient_matches_finite_difference(
        graph: GraphNetwork,
        input: BoundedTensor,
        label: &str,
    ) {
        let map = graph
            .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
            .expect("fixed pass");
        let surrogate = build_surrogate_for_test(&graph, &input, &map, 2);
        assert!(
            surrogate
                .relus
                .iter()
                .any(|r| r.crossing.iter().any(|&c| c)),
            "{label} fixture must have crossing neurons"
        );
        assert!(surrogate_resources_fit(&surrogate, 2));

        // Two evaluation points inside the box (deterministic).
        let input_flat = input.flatten();
        let n = input_flat.len();
        let mut xs = Array2::<f64>::zeros((2, n));
        for j in 0..n {
            let l = f64::from(input_flat.lower()[j]);
            let u = f64::from(input_flat.upper()[j]);
            xs[[0, j]] = l + (u - l) * 0.25;
            xs[[1, j]] = l + (u - l) * 0.75;
        }
        // Deterministic mixed-sign objective over every output coordinate.
        let out_dim = surrogate.nodes[surrogate.output_idx].dim;
        let weights = [0.7f64, 0.3];
        let mut cpos = Array2::<f64>::zeros((2, out_dim));
        let mut cneg = Array2::<f64>::zeros((2, out_dim));
        for r in 0..2 {
            for k in 0..out_dim {
                let magnitude = 0.25 + ((k % 5) as f64) * 0.17;
                let coefficient = if (r + k).is_multiple_of(2) {
                    magnitude
                } else {
                    -magnitude
                };
                cpos[[r, k]] = coefficient.max(0.0);
                cneg[[r, k]] = coefficient.min(0.0);
            }
        }

        // Mixed starting alphas (interior, so both vertices are reachable).
        let alphas: Vec<Vec<f64>> = surrogate
            .relus
            .iter()
            .map(|r| {
                r.crossing
                    .iter()
                    .enumerate()
                    .map(|(j, _)| 0.3 + 0.4 * ((j % 3) as f64) / 2.0)
                    .collect()
            })
            .collect();

        let objective = |a: &[Vec<f64>]| -> f64 {
            let fields = forward_fields(&surrogate, a, &xs, false, None).expect("fields");
            let vals =
                margin_values(&fields.out_l, &fields.out_u, &cpos, &cneg, None).expect("margins");
            vals.iter().zip(weights.iter()).map(|(v, w)| v * w).sum()
        };

        let fields = forward_fields(&surrogate, &alphas, &xs, true, None).expect("fields");
        let pre = fields.relu_pre_l.expect("pre fields");
        let mut seed_l = cpos.clone();
        let mut seed_u = cneg.clone();
        for (r, &w) in weights.iter().enumerate() {
            for k in 0..out_dim {
                seed_l[[r, k]] *= w;
                seed_u[[r, k]] *= w;
            }
        }
        let grads =
            alpha_gradients(&surrogate, &alphas, &pre, seed_l, seed_u, None).expect("adjoint");

        let h = 0.05f64;
        let mut checked = 0usize;
        for (ri, relu) in surrogate.relus.iter().enumerate() {
            for j in 0..relu.crossing.len() {
                if !relu.crossing[j] {
                    continue;
                }
                let mut plus = alphas.clone();
                plus[ri][j] += h;
                let mut minus = alphas.clone();
                minus[ri][j] -= h;
                let fd = (objective(&plus) - objective(&minus)) / (2.0 * h);
                let ad = grads[ri][j];
                let tol = 1e-7 * (1.0 + fd.abs().max(ad.abs()));
                assert!(
                    (fd - ad).abs() <= tol,
                    "{label} grad mismatch at relu {ri} ('{}') coord {j}: fd={fd}, adjoint={ad}",
                    relu.name
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 3,
            "{label} needs >=3 crossing coordinates, got {checked}"
        );

        // Degenerate-box sanity: evaluating the fields at a point equals the
        // concretized composed bound at that point for a zero-radius box is
        // implicitly covered by the MC containment suite; here assert the
        // objective is finite.
        assert!(objective(&alphas).is_finite());
    }

    /// The surrogate objective is exactly linear in each alpha coordinate
    /// (multilinear overall), so the adjoint gradient must match a central
    /// finite difference. This pins Conv/Dense/ReLU/Add.
    #[test]
    fn test_alpha_gradient_matches_finite_difference() {
        let (graph, input) = super::super::tests_image::build_residual_dag_for_grad_test();
        assert_alpha_gradient_matches_finite_difference(graph, input, "Conv2d residual");
    }

    /// Same gradient oracle on the cGAN operator surface, including an
    /// asymmetric ConvTranspose, output_padding, and a negative BatchNorm scale.
    #[test]
    fn test_convt_signed_bn_alpha_gradient_matches_finite_difference() {
        let (graph, input) = super::super::tests_image::build_convt_signed_bn_for_grad_test();
        assert_alpha_gradient_matches_finite_difference(graph, input, "ConvTranspose+signed-BN");
    }

    #[test]
    fn test_conv_transpose_forward_adjoint_and_output_padding_cells() {
        // stride=2, kernel=1, output_padding=1 creates a 4x4 grid from 2x2
        // input. Only even/even coordinates receive a linear contribution.
        let layer = ConvTranspose2dLayer::new_full(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![-2.0]).unwrap(),
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        .unwrap();
        let geo = resolve_conv_transpose_geometry("test", &layer, &[1, 2, 2], 4, 16).unwrap();
        let w = vec![-2.0f64];
        let wabs = vec![2.0f64];
        let xs =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0]).unwrap();
        let tx = conv_transpose_apply_batch(&w, &geo, &xs, None).unwrap();
        for r in 0..2 {
            for oh in 0..4 {
                for ow in 0..4 {
                    let got = tx[[r, oh * 4 + ow]];
                    let expected = if oh.is_multiple_of(2) && ow.is_multiple_of(2) {
                        -2.0 * xs[[r, (oh / 2) * 2 + ow / 2]]
                    } else {
                        0.0
                    };
                    assert_eq!(got, expected, "row {r}, output ({oh},{ow})");
                }
            }
        }

        // Dot-product identity independently pins T^T for W and |W|.
        let ys = Array2::from_shape_fn((2, 16), |(r, j)| (r as f64 + 1.0) * (j as f64 - 5.0) / 7.0);
        for kernel in [&w, &wabs] {
            let lhs_fields = conv_transpose_apply_batch(kernel, &geo, &xs, None).unwrap();
            let rhs_fields = conv_transpose_apply_batch_t(kernel, &geo, &ys, None).unwrap();
            let lhs: f64 = lhs_fields.iter().zip(ys.iter()).map(|(a, b)| a * b).sum();
            let rhs: f64 = xs.iter().zip(rhs_fields.iter()).map(|(a, b)| a * b).sum();
            assert!((lhs - rhs).abs() <= 1e-12 * (1.0 + lhs.abs().max(rhs.abs())));
        }

        // The bottom-right output-padding-only cell has no upstream edge.
        let mut padding_lambda = Array2::<f64>::zeros((1, 16));
        padding_lambda[[0, 15]] = 1.0;
        let upstream = conv_transpose_apply_batch_t(&w, &geo, &padding_lambda, None).unwrap();
        assert!(upstream.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn test_asymmetric_conv_transpose_matches_independent_runtime_and_integer_oracle() {
        let input_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 1.0, 2.0, 3.0, -2.0, 4.0];
        let input = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), input_values.clone()).expect("input");
        // ONNX layout [in_c, out_c, kh, kw], listed one [kh,kw] plane
        // for each (in_c,out_c).
        let kernel_values = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // ic0/oc0
            -1.0, 0.0, 2.0, 3.0, -2.0, 1.0, // ic0/oc1
            2.0, -1.0, 1.0, 0.0, 3.0, -2.0, // ic1/oc0
            1.0, 4.0, -3.0, -1.0, 2.0, 5.0, // ic1/oc1
        ];
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 3]), kernel_values).expect("kernel");
        let stride = (2, 3);
        let padding = (1, 2);
        let dilation = (2, 1);
        let output_padding = (1, 2);
        let expected: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // oc0 row0
            23.0, 9.0, 25.0, 23.0, 26.0, 29.0, 36.0, // row1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row2
            18.0, 20.0, 19.0, 34.0, 24.0, 42.0, 28.0, // row3
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // oc1 row0
            -5.0, -2.0, -10.0, 23.0, 5.0, 14.0, 13.0, // row1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row2
            19.0, 17.0, -14.0, -5.0, 14.0, -4.0, 26.0, // row3
        ];

        // Independent production runtime implementation (GEMM + col2im).
        let runtime =
            conv2d_transpose_forward(&input, &kernel, stride, padding, dilation, output_padding)
                .expect("runtime ConvTranspose");
        assert_eq!(runtime.shape(), &[2, 4, 7]);
        assert_eq!(
            runtime
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
            expected
        );

        let layer = ConvTranspose2dLayer::new_full(
            kernel.clone(),
            None,
            stride,
            padding,
            dilation,
            output_padding,
        )
        .expect("layer");
        let geo = resolve_conv_transpose_geometry("oracle", &layer, &[2, 2, 3], 12, 56)
            .expect("geometry");
        let mut packed = vec![0.0f64; 24];
        for ic in 0..geo.in_c {
            for ki in 0..geo.kh {
                for kj in 0..geo.kw {
                    let base = ((ic * geo.kh + ki) * geo.kw + kj) * geo.out_c;
                    for oc in 0..geo.out_c {
                        packed[base + oc] = f64::from(kernel[[ic, oc, ki, kj]]);
                    }
                }
            }
        }
        let xs = Array2::from_shape_vec(
            (1, input_values.len()),
            input_values.into_iter().map(f64::from).collect(),
        )
        .expect("row");
        let surrogate =
            conv_transpose_apply_batch(&packed, &geo, &xs, None).expect("surrogate scatter");
        assert_eq!(surrogate.as_slice().expect("contiguous"), expected);
    }

    #[test]
    fn test_signed_batch_norm_forward_and_adjoint_swap() {
        let l = Array2::from_shape_vec((1, 3), vec![1.0, -2.0, 0.0]).unwrap();
        let u = Array2::from_shape_vec((1, 3), vec![3.0, 4.0, 2.0]).unwrap();
        let scale = vec![2.0, -3.0, -0.5];
        let bias = vec![0.5, 1.0, -1.0];
        let (nl, nu) = batch_norm_fields(&scale, &bias, &l, &u, None).unwrap();
        assert_eq!(nl.as_slice().unwrap(), &[2.5, -11.0, -2.0]);
        assert_eq!(nu.as_slice().unwrap(), &[6.5, 7.0, -1.0]);

        let lam_l = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let lam_u = Array2::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();
        let (in_l, in_u) = batch_norm_adjoint(&scale, &lam_l, &lam_u, None).unwrap();
        assert_eq!(in_l.as_slice().unwrap(), &[2.0, -15.0, -3.0]);
        assert_eq!(in_u.as_slice().unwrap(), &[8.0, -6.0, -1.5]);
    }

    #[test]
    fn test_resource_plan_checked_overflow_and_excessive_composition_dims_refuse() {
        let overflow = SurrogateScan {
            parameter_bytes: usize::MAX,
            crossing_count: 1,
            max_node_dim: 1,
            ..SurrogateScan::default()
        };
        assert!(finalize_resource_plan(overflow, 1, 1, 1, 0).is_none());

        let composition_overflow = SurrogateScan {
            structural_bytes: 1,
            sum_node_dims: 1,
            sum_relu_dims: 1,
            max_node_dim: 1,
            pass_work_per_row: 1,
            node_count: 1,
            relu_count: 1,
            crossing_count: 1,
            ..SurrogateScan::default()
        };
        assert!(
            finalize_resource_plan(composition_overflow, MAX_ROWS, usize::MAX / 2, 3, 0).is_none()
        );
        assert!(
            finalize_resource_plan(composition_overflow, MAX_ROWS, 4_000_000, 4_000_000, 0)
                .is_none()
        );
        assert!(finalize_resource_plan(composition_overflow, 1, 1, 1, 0).is_some());
        assert!(
            finalize_resource_plan(composition_overflow, 1, 1, 1, MAX_SURROGATE_BYTES,).is_none(),
            "retained memo bytes must share, not sit outside, the surrogate cap"
        );
    }

    #[test]
    fn test_deep_narrow_graph_is_refused_by_structural_metadata_plan() {
        const NODES: usize = 3_900;
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), -1.0),
            ArrayD::from_elem(IxDyn(&[1]), 1.0),
        )
        .expect("input");
        let one_bound = input.clone();
        let mut graph = GraphNetwork::new();
        let mut bounds = HashMap::new();
        let mut previous = String::new();
        for i in 0..NODES {
            let name = format!("deep_relu_{i:04}");
            let node = if i == 0 {
                GraphNode::from_input(name.as_str(), Layer::ReLU(ReLULayer))
            } else {
                GraphNode::new(
                    name.as_str(),
                    Layer::ReLU(ReLULayer),
                    vec![previous.clone()],
                )
            };
            graph.add_node(node);
            bounds.insert(name.clone(), one_bound.clone());
            previous = name;
        }
        let scan = scan_surrogate(&graph, &input, &bounds, None)
            .expect("scan")
            .expect("surface");
        assert_eq!(scan.node_count, NODES);
        assert!(
            scan.structural_bytes > MAX_SURROGATE_BYTES,
            "deep graph structural charge must exceed cap: {}",
            scan.structural_bytes
        );
        assert!(finalize_resource_plan(scan, 1, 1, 1, 0).is_none());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_live_deadline_cancels_mostly_clipped_convt_forward_and_adjoint_in_kernel() {
        // Warm the Rayon worker path so the live deadline cannot expire merely
        // while the first worker is being spawned.
        let warm_geo = ConvTransposeGeometry {
            in_c: 1,
            in_h: 1,
            in_w: 1,
            out_c: 1,
            out_h: 1,
            out_w: 1,
            kh: 1,
            kw: 1,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        };
        let point = Array2::from_elem((1, 1), 1.0);
        conv_transpose_apply_batch(&[1.0], &warm_geo, &point, None).expect("warm forward");
        conv_transpose_apply_batch_t(&[1.0], &warm_geo, &point, None).expect("warm adjoint");

        // Exactly one of 16M width taps lands in the 1-cell output. Before
        // rejected taps consumed poll work, both kernels traversed the entire
        // loop without an in-kernel deadline observation.
        const TAPS: usize = 16_000_001;
        let clipped_geo = ConvTransposeGeometry {
            in_c: 1,
            in_h: 1,
            in_w: 1,
            out_c: 1,
            out_h: 1,
            out_w: 1,
            kh: 1,
            kw: TAPS,
            stride: (1, 1),
            padding: (0, TAPS / 2),
            dilation: (1, 1),
        };
        let weights = try_zeroed_f64(TAPS).expect("bounded test weights");

        LAST_IN_KERNEL_DEADLINE_WORK.store(0, Ordering::Relaxed);
        let started = Instant::now();
        let forward = conv_transpose_apply_batch(
            &weights,
            &clipped_geo,
            &point,
            Some(started + Duration::from_millis(10)),
        );
        let forward_elapsed = started.elapsed();
        assert!(matches!(forward, Err(NyError::DeadlineExceeded(_))));
        let forward_work = LAST_IN_KERNEL_DEADLINE_WORK.load(Ordering::Relaxed);
        assert!(
            forward_work > 0 && forward_work.is_multiple_of(DEADLINE_POLL_WORK),
            "deadline must be observed by a rejected-tap poll, work={forward_work}"
        );
        assert!(
            forward_elapsed < Duration::from_secs(1),
            "forward cancellation latency was {forward_elapsed:?}"
        );

        LAST_IN_KERNEL_DEADLINE_WORK.store(0, Ordering::Relaxed);
        let started = Instant::now();
        let adjoint = conv_transpose_apply_batch_t(
            &weights,
            &clipped_geo,
            &point,
            Some(started + Duration::from_millis(10)),
        );
        let adjoint_elapsed = started.elapsed();
        assert!(matches!(adjoint, Err(NyError::DeadlineExceeded(_))));
        let adjoint_work = LAST_IN_KERNEL_DEADLINE_WORK.load(Ordering::Relaxed);
        assert!(
            adjoint_work > 0 && adjoint_work.is_multiple_of(DEADLINE_POLL_WORK),
            "adjoint deadline must be observed by a rejected-tap poll, work={adjoint_work}"
        );
        assert!(
            adjoint_elapsed < Duration::from_secs(1),
            "adjoint cancellation latency was {adjoint_elapsed:?}"
        );
    }

    #[test]
    fn test_surrogate_caps_and_deadline_refuse_without_partial_output() {
        let (graph, input) = super::super::tests_image::build_convt_signed_bn_for_grad_test();
        let map = graph
            .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
            .unwrap();
        let mut surrogate = build_surrogate_for_test(&graph, &input, &map, 2);
        assert!(surrogate_resources_fit(&surrogate, 2));
        let planned_bytes = surrogate.resource_plan.planned_bytes;
        surrogate.resource_plan.planned_bytes = MAX_SURROGATE_BYTES + 1;
        assert!(!surrogate_resources_fit(&surrogate, 1));
        surrogate.resource_plan.planned_bytes = planned_bytes;
        surrogate.resource_plan.scan.pass_work_per_row = MAX_SURROGATE_PASS_MACS + 1;
        assert!(!surrogate_resources_fit(&surrogate, 1));
    }
}
