// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Additive indexed accumulation layers: ScatterAdd and IndexAdd.
//!
//! # Rounding contract (#vnncomp-aw-soundness)
//!
//! Over the REALS `+` is associative and commutative, so a destination cell hit
//! by several index entries has one well-defined value and needs no union — that
//! is what separates this family from the overwriting `ScatterND`.
//!
//! In f32 it is neither. `a + b` is round-to-NEAREST, so an accumulation both
//! rounds and depends on visit order, and a plain `+=` used for BOTH endpoints
//! moves them the same way: the interval is TRANSLATED off the true value rather
//! than widened around it. Every accumulation in this module therefore rounds
//! OUTWARD for the endpoint it is building:
//!
//! * static indices — `scatter_add_tensor` / `index_add_tensor` take a directed
//!   combiner (`add_down_f32` for the lower output, `add_up_f32` for the upper);
//! * dynamic indices — the additive hull folds in f64 with directed adds and
//!   lands with one directed cast ([`additive_mass_down`] / [`additive_mass_up`]),
//!   then shifts `data` with `add_down_f32` / `add_up_f32`;
//! * CROWN — the constant shift is an INTERVAL when the constant operand had to
//!   accumulate, and the `A @ shift` bias fold is directed end to end (see
//!   [`additive_scatter_backward`]).
//!
//! The directed helpers are exact when the arithmetic is exact, so the common
//! duplicate-free case costs nothing.

use ndarray::{Array2, ArrayD, Dimension, IxDyn};
use ny_core::dd::{next_down_f64, next_up_f64, two_sum};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{
    add_down_f32, add_up_f32, cast_f64_to_f32_down, cast_f64_to_f32_up, BoundedTensor,
};
use std::borrow::Cow;

use super::super::common::BoundPropagation;
use crate::LinearBounds;

/// `a + b` in f64 rounded toward -inf: the largest f64 that is `<= a + b`.
///
/// The f64 mirror of [`ny_tensor::add_down_f32`], built on Knuth's `TwoSum`.
/// The residual `lo` is EXACTLY what the round-to-nearest add discarded, so
/// this steps only when the add was inexact — an exact add costs nothing.
///
/// Needed because a certified accumulation cannot simply "widen to f64 and
/// hope": f64 round-to-nearest still moves a running sum INWARD, and the
/// f64 -> f32 store at the end does not necessarily step (it does not when the
/// f64 value happens to be exactly f32-representable), so nothing downstream
/// would absorb the drift.
#[inline]
fn add_down_f64(a: f64, b: f64) -> f64 {
    let (hi, lo) = two_sum(a, b);
    // OVERFLOW first: `TwoSum`'s residual is meaningless once `hi` is not
    // finite, and `next_down_f64` preserves infinities, so the generic path
    // would hand back `+inf` as a LOWER bound on a finite sum.
    if !hi.is_finite() {
        return if hi == f64::INFINITY && a.is_finite() && b.is_finite() {
            f64::MAX
        } else {
            hi
        };
    }
    if lo < 0.0 {
        next_down_f64(hi)
    } else {
        hi
    }
}

/// `a + b` in f64 rounded toward +inf. The mirror of [`add_down_f64`].
#[inline]
fn add_up_f64(a: f64, b: f64) -> f64 {
    let (hi, lo) = two_sum(a, b);
    if !hi.is_finite() {
        return if hi == f64::NEG_INFINITY && a.is_finite() && b.is_finite() {
            f64::MIN
        } else {
            hi
        };
    }
    if lo > 0.0 {
        next_up_f64(hi)
    } else {
        hi
    }
}

/// Which of the (data, src) operands is the bounded/variable activation input,
/// for the constant-index CROWN backward of an additive scatter/index-add layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariableOperand {
    /// `y = data_var + scatter(src_const)`. Jacobian w.r.t. data is the identity;
    /// the scattered constant only shifts the bias.
    Data,
    /// `y = data_const + S @ src_var`, where `S` is the 0/1 accumulation matrix.
    /// Jacobian w.r.t. src is `S`; the constant data only shifts the bias.
    Src,
}

/// Apply an exact additive-scatter CROWN backward step.
///
/// `node_lb` is linear in this layer's output (flattened size `output_size`).
/// The result is linear in the single variable operand.
///
/// * `output_size` — flat size of the layer output (== data tensor size).
/// * `var_size` — flat size of the variable operand.
/// * `const_shift_lo[i]` / `const_shift_up[i]` — an ENCLOSURE of the constant
///   contribution added to output position `i` by the constant operand(s).
///   The two slices coincide when that contribution is exactly representable
///   (the common case: no duplicate destinations, or a constant `data` operand
///   that is simply copied); they separate only when the constant scatter had
///   to accumulate several f32 values into one cell, where the accumulation
///   itself rounds. Folds into the bias.
/// * `var_to_out[k]` — for [`VariableOperand::Src`], the list of output flat
///   positions that variable element `k` flows into (accumulation may map one
///   src element to one output element; identity for Data is handled by caller).
///
/// Math: `new_A[:, k] = sum_{i in var_to_out[k]} A[:, i]` and
/// `new_b[:] += A @ const_shift`.
fn additive_scatter_backward(
    node_lb: &LinearBounds,
    output_size: usize,
    var_size: usize,
    const_shift_lo: &[f32],
    const_shift_up: &[f32],
    var_to_out: &VarToOut,
) -> Result<LinearBounds> {
    if node_lb.num_inputs() != output_size {
        return Err(NyError::ShapeMismatch {
            expected: vec![output_size],
            got: vec![node_lb.num_inputs()],
        });
    }
    // The bias fold indexes `A[:, col]` for every shift column, so both shift
    // slices must span exactly the output. Checked rather than debug-asserted:
    // a mismatch here would index out of bounds in a release build.
    if const_shift_lo.len() != output_size || const_shift_up.len() != output_size {
        return Err(NyError::ShapeMismatch {
            expected: vec![output_size, output_size],
            got: vec![const_shift_lo.len(), const_shift_up.len()],
        });
    }
    let num_outputs = node_lb.num_outputs();
    let lower_a = node_lb.lower_a();
    let upper_a = node_lb.upper_a();

    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, var_size));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, var_size));

    match var_to_out {
        // Data-variable: y = data + const, Jacobian = identity, so var_size ==
        // output_size and column k maps 1:1 to output column k.
        VarToOut::Identity => {
            debug_assert_eq!(var_size, output_size);
            new_lower_a.assign(lower_a);
            new_upper_a.assign(upper_a);
        }
        // Src-variable: accumulate each output column into the src column(s) that
        // feed it. `map[k]` lists every output position fed by src element k.
        VarToOut::Map(map) => {
            for (src_k, out_positions) in map.iter().enumerate() {
                for &out_i in out_positions {
                    for row in 0..num_outputs {
                        new_lower_a[[row, src_k]] += lower_a[[row, out_i]];
                        new_upper_a[[row, src_k]] += upper_a[[row, out_i]];
                    }
                }
            }
        }
    }

    // Fold the constant operand contribution into the bias: b += A @ const_shift.
    //
    // SOUNDNESS (#vnncomp-aw-soundness): this dot product is part of a CERTIFIED
    // bound, so every step must round OUTWARD.
    //
    // 1. Each PRODUCT is exact: both factors are f32, and an f32×f32 product needs
    //    at most 48 < 53 significand bits, so widening to f64 loses nothing (the
    //    exponent range is nowhere near f64's, even for subnormal f32 factors).
    // 2. Each ADDITION is directed (`add_down_f64` / `add_up_f64`). Plain f64 `+=`
    //    is round-to-NEAREST and moves the running sum INWARD; the old code leaned
    //    on the final `next_down_f32`/`next_up_f32` step to absorb that, which is a
    //    heuristic, not a bound — under cancellation the f64 drift can exceed one
    //    f32 ULP, and the f64→f32 cast does not always step at all.
    // 3. The STORE is a directed cast (`cast_f64_to_f32_down` / `..._up`), which is
    //    the tightest f32 on the correct side of the f64 value AND handles the
    //    overflow case `next_down_f32(+inf) == +inf` would have got wrong (that
    //    would have published `+inf` as a LOWER bias).
    // 4. The const shift is itself an INTERVAL (see `const_shift_lo/up`): the
    //    endpoint that drives the bound outward depends on the coefficient's sign,
    //    so pick `lo` where the coefficient is >= 0 and `up` where it is negative
    //    for the lower bias, and the mirror for the upper.
    let mut new_lower_b = node_lb.lower_b().clone();
    let mut new_upper_b = node_lb.upper_b().clone();
    let const_is_zero =
        const_shift_lo.iter().all(|&v| v == 0.0) && const_shift_up.iter().all(|&v| v == 0.0);
    if !const_is_zero {
        for row in 0..num_outputs {
            let mut lo_acc = node_lb.lower_b()[row] as f64;
            let mut up_acc = node_lb.upper_b()[row] as f64;
            for col in 0..output_size {
                let (c_lo, c_up) = (const_shift_lo[col] as f64, const_shift_up[col] as f64);
                let al = lower_a[[row, col]] as f64;
                let au = upper_a[[row, col]] as f64;
                // Minimize `al * c` over c in [c_lo, c_up] for the lower bias,
                // maximize `au * c` for the upper.
                lo_acc = add_down_f64(lo_acc, al * if al >= 0.0 { c_lo } else { c_up });
                up_acc = add_up_f64(up_acc, au * if au >= 0.0 { c_up } else { c_lo });
            }
            new_lower_b[row] = cast_f64_to_f32_down(lo_acc);
            new_upper_b[row] = cast_f64_to_f32_up(up_acc);
        }
    }

    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Mapping from each variable-operand flat index to the output flat index/indices
/// it contributes to under the additive scatter.
enum VarToOut {
    /// Variable operand is `data`: identity (output position == data position).
    Identity,
    /// Variable operand is `src`: `map[k]` = output positions fed by src elem `k`.
    Map(Vec<Vec<usize>>),
}

/// Flatten an array to row-major (C-order) Vec, copying if non-contiguous.
fn flatten_row_major(arr: &ArrayD<f32>) -> Vec<f32> {
    match arr.as_slice() {
        Some(slice) => slice.to_vec(),
        None => arr.iter().copied().collect(),
    }
}

/// Row-major strides for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Build the per-src-element → output-position map for ScatterAdd
/// (`src.shape() == indices.shape()`, `axis` is the scatter axis).
///
/// For each src multi-index `coord`, the output coordinate is `coord` with
/// `coord[axis]` replaced by `indices[coord]` (normalized). The mapping is
/// 1-element-per-src (additive scatter is injective per src element), but we
/// store it as a Vec for uniformity with the accumulation backend.
fn build_scatter_add_src_map(
    data_shape: &[usize],
    src_shape: &[usize],
    axis: usize,
    indices: &ArrayD<i64>,
) -> Result<Vec<Vec<usize>>> {
    let axis_len = data_shape[axis];
    let data_strides = row_major_strides(data_shape);
    let src_size = checked_shape_product(src_shape)
        .ok_or_else(|| NyError::InvalidSpec("ScatterAdd src shape overflow".to_string()))?;
    let mut map: Vec<Vec<usize>> = vec![Vec::new(); src_size];

    for (src_k, coord) in ndarray::indices(src_shape).into_iter().enumerate() {
        let coord_vec: Vec<usize> = (0..src_shape.len()).map(|d| coord[d]).collect();
        let scatter_idx = normalize_index(indices[IxDyn(&coord_vec)], axis_len, "ScatterAdd")?;
        let mut out_coord = coord_vec.clone();
        out_coord[axis] = scatter_idx;
        let out_flat: usize = out_coord
            .iter()
            .zip(&data_strides)
            .map(|(c, s)| c * s)
            .sum();
        map[src_k].push(out_flat);
    }
    Ok(map)
}

/// Build the per-src-element → output-position map for IndexAdd
/// (1-D `indices` of length `src.shape()[axis]`; src matches data on all other
/// axes). For each src multi-index `coord`, output coordinate is `coord` with
/// `coord[axis]` replaced by `indices[coord[axis]]`.
fn build_index_add_src_map(
    data_shape: &[usize],
    src_shape: &[usize],
    axis: usize,
    indices: &ArrayD<i64>,
) -> Result<Vec<Vec<usize>>> {
    let axis_len = data_shape[axis];
    let data_strides = row_major_strides(data_shape);
    let src_size = checked_shape_product(src_shape)
        .ok_or_else(|| NyError::InvalidSpec("IndexAdd src shape overflow".to_string()))?;
    let mut map: Vec<Vec<usize>> = vec![Vec::new(); src_size];

    for (src_k, coord) in ndarray::indices(src_shape).into_iter().enumerate() {
        let coord_vec: Vec<usize> = (0..src_shape.len()).map(|d| coord[d]).collect();
        let scatter_idx =
            normalize_index(indices[IxDyn(&[coord_vec[axis]])], axis_len, "IndexAdd")?;
        let mut out_coord = coord_vec.clone();
        out_coord[axis] = scatter_idx;
        let out_flat: usize = out_coord
            .iter()
            .zip(&data_strides)
            .map(|(c, s)| c * s)
            .sum();
        map[src_k].push(out_flat);
    }
    Ok(map)
}

#[derive(Debug, Clone)]
enum ResolvedIndices {
    Static(Box<ArrayD<i64>>),
    Dynamic,
}

#[derive(Debug, Clone)]
struct ResolvedInputs {
    data: BoundedTensor,
    indices: ResolvedIndices,
    src: BoundedTensor,
}

fn finish_bounds(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<BoundedTensor> {
    if lower.iter().all(|v| v.is_finite()) && upper.iter().all(|v| v.is_finite()) {
        BoundedTensor::new(lower, upper)
    } else {
        BoundedTensor::new_allow_infinite(lower, upper)
    }
}

fn has_non_finite_bounds(bounds: &BoundedTensor) -> bool {
    bounds.lower().iter().any(|v| !v.is_finite()) || bounds.upper().iter().any(|v| !v.is_finite())
}

fn broadcast_infinite_bounds(shape: &[usize]) -> Result<BoundedTensor> {
    BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(shape), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(shape), f32::INFINITY),
    )
}

/// The most NEGATIVE additive mass `src` can contribute to any one destination
/// cell, rounded DOWN (toward -inf).
///
/// Under a dynamic (runtime) index every src element routes to exactly one
/// destination, so a cell's accumulated contribution is `sum over the routed
/// subset`. That subset is unknown, but each omitted term of `min(lower, 0)` is
/// `<= 0`, so summing the negative parts of ALL elements is a valid lower bound
/// on every subset sum.
///
/// SOUNDNESS: the old code folded this with a plain f32 `+`, which is
/// round-to-NEAREST and unbounded in length — a 1001-element fold of
/// `[-1e8, -1.0 x 1000]` returned exactly `-1e8`, losing all 1000 small terms to
/// absorption and certifying a lower bound 1000 above the reachable value. The
/// fold now runs in f64 with DIRECTED addition and lands with ONE directed cast.
///
/// Why no `gamma_n` allowance is needed on top: [`add_down_f64`] returns a value
/// `<= a + b` EXACTLY (Knuth's `TwoSum` reports the discarded residual, so the
/// step is taken exactly when the add was inexact). By induction the running
/// `acc` is `<=` the exact real sum of the terms consumed so far, for any length,
/// with no assumption about magnitudes or cancellation. `cast_f64_to_f32_down`
/// then returns the largest f32 `<= acc`. A `gamma_n * sum|x|` term would be a
/// second, looser way to bound the same drift and is redundant here.
fn additive_mass_down(lower: &ArrayD<f32>) -> f32 {
    let mut acc = 0.0_f64;
    for &value in lower.iter() {
        if value < 0.0 {
            acc = add_down_f64(acc, f64::from(value));
        }
    }
    cast_f64_to_f32_down(acc)
}

/// The most POSITIVE additive mass `src` can contribute to any one destination
/// cell, rounded UP. The mirror of [`additive_mass_down`].
fn additive_mass_up(upper: &ArrayD<f32>) -> f32 {
    let mut acc = 0.0_f64;
    for &value in upper.iter() {
        if value > 0.0 {
            acc = add_up_f64(acc, f64::from(value));
        }
    }
    cast_f64_to_f32_up(acc)
}

/// Widen `data` by the dynamic-index additive hull, rounding each endpoint
/// OUTWARD (`add_down_f32` below, `add_up_f32` above) — a plain f32 `+` here
/// would translate both endpoints by the same round-to-nearest error instead of
/// widening the interval.
fn shift_by_additive_hull(data: &BoundedTensor, src: &BoundedTensor) -> Result<BoundedTensor> {
    let additive_lower = additive_mass_down(src.lower());
    let additive_upper = additive_mass_up(src.upper());
    finish_bounds(
        data.lower()
            .mapv(|value| add_down_f32(value, additive_lower)),
        data.upper().mapv(|value| add_up_f32(value, additive_upper)),
    )
}

fn normalize_index(raw: i64, axis_len: usize, layer: &str) -> Result<usize> {
    let normalized = if raw < 0 { axis_len as i64 + raw } else { raw };
    if normalized < 0 || normalized >= axis_len as i64 {
        return Err(NyError::InvalidSpec(format!(
            "{layer}: index {raw} out of bounds for axis length {axis_len}"
        )));
    }
    Ok(normalized as usize)
}

#[derive(Debug, Clone)]
pub struct ScatterAddLayer {
    axis: i64,
    data_constant: Option<ArrayD<f32>>,
    indices: Option<ArrayD<i64>>,
    src_constant: Option<ArrayD<f32>>,
}

impl ScatterAddLayer {
    pub fn new(
        axis: i64,
        data_constant: Option<ArrayD<f32>>,
        indices: Option<ArrayD<i64>>,
        src_constant: Option<ArrayD<f32>>,
    ) -> Self {
        Self {
            axis,
            data_constant,
            indices,
            src_constant,
        }
    }

    pub fn activation_input_count(&self) -> usize {
        usize::from(self.data_constant.is_none())
            + usize::from(self.indices.is_none())
            + usize::from(self.src_constant.is_none())
    }

    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        super::super::common::resolve_axis(self.axis, ndim, "ScatterAdd")
    }

    fn resolve_inputs(&self, activation_inputs: &[&BoundedTensor]) -> Result<ResolvedInputs> {
        let expected = self.activation_input_count();
        if activation_inputs.len() != expected {
            return Err(NyError::InvalidSpec(format!(
                "ScatterAdd expects {} activation input(s), got {}",
                expected,
                activation_inputs.len()
            )));
        }

        let mut next_input = 0usize;

        let data = if let Some(data) = &self.data_constant {
            BoundedTensor::concrete(data.clone())?
        } else {
            let input = activation_inputs
                .get(next_input)
                .ok_or_else(|| NyError::InvalidSpec("ScatterAdd missing data input".to_string()))?;
            next_input += 1;
            (*input).clone()
        };

        let indices = if let Some(indices) = &self.indices {
            ResolvedIndices::Static(Box::new(indices.clone()))
        } else {
            let _input = activation_inputs.get(next_input).ok_or_else(|| {
                NyError::InvalidSpec("ScatterAdd missing index input".to_string())
            })?;
            next_input += 1;
            ResolvedIndices::Dynamic
        };

        let src = if let Some(src) = &self.src_constant {
            BoundedTensor::concrete(src.clone())?
        } else {
            let input = activation_inputs
                .get(next_input)
                .ok_or_else(|| NyError::InvalidSpec("ScatterAdd missing src input".to_string()))?;
            (*input).clone()
        };

        Ok(ResolvedInputs { data, indices, src })
    }

    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input_a, input_b])
    }

    pub fn propagate_ibp_ternary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        input_c: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input_a, input_b, input_c])
    }

    fn propagate_ibp_with_inputs(
        &self,
        activation_inputs: &[&BoundedTensor],
    ) -> Result<BoundedTensor> {
        let resolved = self.resolve_inputs(activation_inputs)?;
        if has_non_finite_bounds(&resolved.data) || has_non_finite_bounds(&resolved.src) {
            return broadcast_infinite_bounds(resolved.data.shape());
        }

        match resolved.indices {
            // The accumulation is DIRECTED per endpoint: `add_down_f32` builds
            // the lower output, `add_up_f32` the upper. A plain `+=` here is
            // round-to-nearest and drifts BOTH endpoints the same way, which
            // translates the interval instead of widening it.
            ResolvedIndices::Static(indices) => finish_bounds(
                self.scatter_add_tensor(
                    resolved.data.lower(),
                    &indices,
                    resolved.src.lower(),
                    add_down_f32,
                )?,
                self.scatter_add_tensor(
                    resolved.data.upper(),
                    &indices,
                    resolved.src.upper(),
                    add_up_f32,
                )?,
            ),
            ResolvedIndices::Dynamic => {
                self.propagate_dynamic_indices(&resolved.data, &resolved.src)
            }
        }
    }

    fn propagate_dynamic_indices(
        &self,
        data: &BoundedTensor,
        src: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if src.lower().is_empty() {
            return Ok(data.clone());
        }
        shift_by_additive_hull(data, src)
    }

    /// Accumulate `src` into a copy of `data` at the (static) scatter indices.
    ///
    /// `accumulate` is the DIRECTED combiner for the endpoint being built —
    /// `add_down_f32` for a lower bound, `add_up_f32` for an upper. Mirrors
    /// `ScatterNdLayer::scatter_tensor`'s `combine_duplicate` parameter. It is
    /// applied to EVERY src element, not only to duplicate destinations: even a
    /// single `data + src` add is round-to-nearest and can land on the wrong
    /// side of the true sum.
    fn scatter_add_tensor(
        &self,
        data: &ArrayD<f32>,
        indices: &ArrayD<i64>,
        src: &ArrayD<f32>,
        accumulate: fn(f32, f32) -> f32,
    ) -> Result<ArrayD<f32>> {
        let axis = self.resolve_axis(data.ndim())?;
        if indices.ndim() != src.ndim() {
            return Err(NyError::InvalidSpec(format!(
                "ScatterAdd index rank {} must match src rank {}",
                indices.ndim(),
                src.ndim()
            )));
        }
        if indices.shape() != src.shape() {
            return Err(NyError::ShapeMismatch {
                expected: src.shape().to_vec(),
                got: indices.shape().to_vec(),
            });
        }
        if data.ndim() != src.ndim() {
            return Err(NyError::InvalidSpec(format!(
                "ScatterAdd data rank {} must match src rank {}",
                data.ndim(),
                src.ndim()
            )));
        }
        for dim in 0..data.ndim() {
            if dim != axis && src.shape()[dim] > data.shape()[dim] {
                return Err(NyError::InvalidSpec(format!(
                    "ScatterAdd src dim {} size {} exceeds data dim size {}",
                    dim,
                    src.shape()[dim],
                    data.shape()[dim]
                )));
            }
        }

        let mut output = data.clone();
        let axis_len = data.shape()[axis];
        for (coord, &value) in src.indexed_iter() {
            let scatter_idx = normalize_index(indices[coord.clone()], axis_len, "ScatterAdd")?;
            let mut dst_coord = coord.slice().to_vec();
            dst_coord[axis] = scatter_idx;
            let slot = &mut output[IxDyn(&dst_coord)];
            *slot = accumulate(*slot, value);
        }
        Ok(output)
    }
}

impl ScatterAddLayer {
    /// Identify the single variable operand for constant-index CROWN backward.
    ///
    /// Returns `UnsupportedOp` (so the caller falls back to IBP) when indices are
    /// dynamic, or when zero / more than one of (data, src) is variable.
    fn crown_variable_operand(&self) -> Result<VariableOperand> {
        if self.indices.is_none() {
            return Err(NyError::UnsupportedOp(
                "ScatterAdd CROWN backward requires static (constant) indices".to_string(),
            ));
        }
        match (self.data_constant.is_none(), self.src_constant.is_none()) {
            (true, false) => Ok(VariableOperand::Data),
            (false, true) => Ok(VariableOperand::Src),
            _ => Err(NyError::UnsupportedOp(
                "ScatterAdd CROWN backward requires exactly one variable operand".to_string(),
            )),
        }
    }

    /// Exact CROWN backward for ScatterAdd with constant indices and a single
    /// variable operand. See [`additive_scatter_backward`] for the math.
    pub fn crown_backward(&self, node_lb: &LinearBounds) -> Result<LinearBounds> {
        let variable = self.crown_variable_operand()?;
        let indices = self.indices.as_ref().expect("static indices checked above");

        // The output (== data) shape: known from data_constant for the
        // Src-variable case; for the Data-variable case the data shape equals the
        // flattened output size carried by node_lb (data and output coincide).
        let output_size = node_lb.num_inputs();

        match variable {
            VariableOperand::Data => {
                // y = data + scatter(src_const). Need src_const + a data shape to
                // build the additive constant vector. We only have the flat output
                // size, but scatter requires multi-dim coords. Reconstruct via the
                // data shape inferred from src/indices is not possible in general,
                // so require src_constant and recompute the scatter using a zero
                // data tensor of the output (== data) shape. The data shape is not
                // directly available, so fall back when it is ambiguous.
                let src_const = self.src_constant.as_ref().ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "ScatterAdd CROWN (data-variable) requires constant src".to_string(),
                    )
                })?;
                let data_shape = self.infer_data_shape_for_data_variable(output_size)?;
                let zeros = ArrayD::<f32>::zeros(IxDyn(&data_shape));
                // The constant scatter itself accumulates in f32, so it is an
                // INTERVAL, not a number, whenever two src elements land on one
                // cell. Build both sides and let the bias fold pick per sign.
                // With no duplicate destinations `0 + v` is exact and the two
                // agree bit-for-bit, so the common case is not widened.
                let shift_lo = self.scatter_add_tensor(&zeros, indices, src_const, add_down_f32)?;
                let shift_up = self.scatter_add_tensor(&zeros, indices, src_const, add_up_f32)?;
                additive_scatter_backward(
                    node_lb,
                    output_size,
                    output_size,
                    &flatten_row_major(&shift_lo),
                    &flatten_row_major(&shift_up),
                    &VarToOut::Identity,
                )
            }
            VariableOperand::Src => {
                let data_const = self.data_constant.as_ref().ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "ScatterAdd CROWN (src-variable) requires constant data".to_string(),
                    )
                })?;
                if data_const.len() != output_size {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![output_size],
                        got: vec![data_const.len()],
                    });
                }
                let data_shape = data_const.shape().to_vec();
                let axis = self.resolve_axis(data_shape.len())?;
                let src_shape = indices.shape().to_vec();
                let var_size = checked_shape_product(&src_shape).ok_or_else(|| {
                    NyError::InvalidSpec("ScatterAdd: src shape overflow".to_string())
                })?;
                let map = build_scatter_add_src_map(&data_shape, &src_shape, axis, indices)?;
                // `data_const` is copied, never accumulated, so its constant
                // contribution is exact: the enclosure is the degenerate point.
                let shift_flat = flatten_row_major(data_const);
                additive_scatter_backward(
                    node_lb,
                    output_size,
                    var_size,
                    &shift_flat,
                    &shift_flat,
                    &VarToOut::Map(map),
                )
            }
        }
    }

    /// Infer the data tensor shape for the data-variable case. ScatterAdd
    /// preserves the data shape, but only the flat size is known from `node_lb`.
    /// When `src_constant`/`indices` share the data rank (the common 1-D and
    /// matching-rank cases), prefer the data-shaped reconstruction; otherwise
    /// treat the data as 1-D of `output_size` which is exact for the additive
    /// constant when indices index a 1-D axis.
    fn infer_data_shape_for_data_variable(&self, output_size: usize) -> Result<Vec<usize>> {
        // src/indices share the data rank in valid ScatterAdd; if src rank == 1
        // and its single dim divides output, fall back to 1-D. We keep it simple
        // and sound: use a 1-D data shape only when indices are 1-D; reject
        // higher-rank ambiguity so the caller falls back to IBP.
        let indices = self.indices.as_ref().expect("static indices");
        if indices.ndim() <= 1 {
            return Ok(vec![output_size]);
        }
        Err(NyError::UnsupportedOp(
            "ScatterAdd CROWN (data-variable) supports 1-D indices only without explicit \
             data shape"
                .to_string(),
        ))
    }
}

impl BoundPropagation for ScatterAddLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input])
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.crown_backward(bounds).map(Cow::Owned)
    }
}

#[derive(Debug, Clone)]
pub struct IndexAddLayer {
    axis: i64,
    data_constant: Option<ArrayD<f32>>,
    indices: Option<ArrayD<i64>>,
    src_constant: Option<ArrayD<f32>>,
}

impl IndexAddLayer {
    pub fn new(
        axis: i64,
        data_constant: Option<ArrayD<f32>>,
        indices: Option<ArrayD<i64>>,
        src_constant: Option<ArrayD<f32>>,
    ) -> Self {
        Self {
            axis,
            data_constant,
            indices,
            src_constant,
        }
    }

    pub fn activation_input_count(&self) -> usize {
        usize::from(self.data_constant.is_none())
            + usize::from(self.indices.is_none())
            + usize::from(self.src_constant.is_none())
    }

    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        super::super::common::resolve_axis(self.axis, ndim, "IndexAdd")
    }

    fn resolve_inputs(&self, activation_inputs: &[&BoundedTensor]) -> Result<ResolvedInputs> {
        let expected = self.activation_input_count();
        if activation_inputs.len() != expected {
            return Err(NyError::InvalidSpec(format!(
                "IndexAdd expects {} activation input(s), got {}",
                expected,
                activation_inputs.len()
            )));
        }

        let mut next_input = 0usize;

        let data = if let Some(data) = &self.data_constant {
            BoundedTensor::concrete(data.clone())?
        } else {
            let input = activation_inputs
                .get(next_input)
                .ok_or_else(|| NyError::InvalidSpec("IndexAdd missing data input".to_string()))?;
            next_input += 1;
            (*input).clone()
        };

        let indices = if let Some(indices) = &self.indices {
            ResolvedIndices::Static(Box::new(indices.clone()))
        } else {
            let _input = activation_inputs
                .get(next_input)
                .ok_or_else(|| NyError::InvalidSpec("IndexAdd missing index input".to_string()))?;
            next_input += 1;
            ResolvedIndices::Dynamic
        };

        let src = if let Some(src) = &self.src_constant {
            BoundedTensor::concrete(src.clone())?
        } else {
            let input = activation_inputs
                .get(next_input)
                .ok_or_else(|| NyError::InvalidSpec("IndexAdd missing src input".to_string()))?;
            (*input).clone()
        };

        Ok(ResolvedInputs { data, indices, src })
    }

    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input_a, input_b])
    }

    pub fn propagate_ibp_ternary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        input_c: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input_a, input_b, input_c])
    }

    fn propagate_ibp_with_inputs(
        &self,
        activation_inputs: &[&BoundedTensor],
    ) -> Result<BoundedTensor> {
        let resolved = self.resolve_inputs(activation_inputs)?;
        if has_non_finite_bounds(&resolved.data) || has_non_finite_bounds(&resolved.src) {
            return broadcast_infinite_bounds(resolved.data.shape());
        }

        match resolved.indices {
            // Directed per endpoint — see the matching comment in
            // `ScatterAddLayer::propagate_ibp_with_inputs`.
            ResolvedIndices::Static(indices) => finish_bounds(
                self.index_add_tensor(
                    resolved.data.lower(),
                    &indices,
                    resolved.src.lower(),
                    add_down_f32,
                )?,
                self.index_add_tensor(
                    resolved.data.upper(),
                    &indices,
                    resolved.src.upper(),
                    add_up_f32,
                )?,
            ),
            ResolvedIndices::Dynamic => {
                self.propagate_dynamic_indices(&resolved.data, &resolved.src)
            }
        }
    }

    fn propagate_dynamic_indices(
        &self,
        data: &BoundedTensor,
        src: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if src.lower().is_empty() {
            return Ok(data.clone());
        }
        shift_by_additive_hull(data, src)
    }

    /// Accumulate `src` into a copy of `data` at the (static) 1-D index map.
    /// `accumulate` is the DIRECTED combiner for the endpoint being built —
    /// see [`ScatterAddLayer::scatter_add_tensor`].
    fn index_add_tensor(
        &self,
        data: &ArrayD<f32>,
        indices: &ArrayD<i64>,
        src: &ArrayD<f32>,
        accumulate: fn(f32, f32) -> f32,
    ) -> Result<ArrayD<f32>> {
        let axis = self.resolve_axis(data.ndim())?;
        if indices.ndim() != 1 {
            return Err(NyError::InvalidSpec(format!(
                "IndexAdd index rank must be 1, got {}",
                indices.ndim()
            )));
        }
        if src.ndim() != data.ndim() {
            return Err(NyError::InvalidSpec(format!(
                "IndexAdd src rank {} must match data rank {}",
                src.ndim(),
                data.ndim()
            )));
        }
        if indices.shape()[0] != src.shape()[axis] {
            return Err(NyError::InvalidSpec(format!(
                "IndexAdd index length {} must match src axis {} length {}",
                indices.shape()[0],
                axis,
                src.shape()[axis]
            )));
        }
        for dim in 0..data.ndim() {
            if dim != axis && src.shape()[dim] != data.shape()[dim] {
                return Err(NyError::ShapeMismatch {
                    expected: data.shape().to_vec(),
                    got: src.shape().to_vec(),
                });
            }
        }

        let mut output = data.clone();
        let axis_len = data.shape()[axis];
        for (coord, &value) in src.indexed_iter() {
            let dst_index = normalize_index(indices[IxDyn(&[coord[axis]])], axis_len, "IndexAdd")?;
            let mut dst_coord = coord.slice().to_vec();
            dst_coord[axis] = dst_index;
            let slot = &mut output[IxDyn(&dst_coord)];
            *slot = accumulate(*slot, value);
        }
        Ok(output)
    }
}

impl IndexAddLayer {
    /// Identify the single variable operand for constant-index CROWN backward.
    fn crown_variable_operand(&self) -> Result<VariableOperand> {
        if self.indices.is_none() {
            return Err(NyError::UnsupportedOp(
                "IndexAdd CROWN backward requires static (constant) indices".to_string(),
            ));
        }
        match (self.data_constant.is_none(), self.src_constant.is_none()) {
            (true, false) => Ok(VariableOperand::Data),
            (false, true) => Ok(VariableOperand::Src),
            _ => Err(NyError::UnsupportedOp(
                "IndexAdd CROWN backward requires exactly one variable operand".to_string(),
            )),
        }
    }

    /// Exact CROWN backward for IndexAdd with constant indices and a single
    /// variable operand.
    pub fn crown_backward(&self, node_lb: &LinearBounds) -> Result<LinearBounds> {
        let variable = self.crown_variable_operand()?;
        let indices = self.indices.as_ref().expect("static indices checked above");
        let output_size = node_lb.num_inputs();

        match variable {
            VariableOperand::Data => {
                let src_const = self.src_constant.as_ref().ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "IndexAdd CROWN (data-variable) requires constant src".to_string(),
                    )
                })?;
                // y = data + index_add(src_const). data shape == output shape.
                // src must match data on all axes except `axis`, so data shape ==
                // src shape with axis replaced by `output_size / (other dims)`. We
                // reconstruct it from src_const shape and output_size.
                let data_shape = infer_index_add_data_shape(
                    src_const.shape(),
                    output_size,
                    self.axis,
                    indices.len(),
                )?;
                let zeros = ArrayD::<f32>::zeros(IxDyn(&data_shape));
                // Interval constant shift — see the ScatterAdd counterpart.
                let shift_lo = self.index_add_tensor(&zeros, indices, src_const, add_down_f32)?;
                let shift_up = self.index_add_tensor(&zeros, indices, src_const, add_up_f32)?;
                additive_scatter_backward(
                    node_lb,
                    output_size,
                    output_size,
                    &flatten_row_major(&shift_lo),
                    &flatten_row_major(&shift_up),
                    &VarToOut::Identity,
                )
            }
            VariableOperand::Src => {
                let data_const = self.data_constant.as_ref().ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "IndexAdd CROWN (src-variable) requires constant data".to_string(),
                    )
                })?;
                if data_const.len() != output_size {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![output_size],
                        got: vec![data_const.len()],
                    });
                }
                let data_shape = data_const.shape().to_vec();
                let axis = self.resolve_axis(data_shape.len())?;
                // src shape == data shape with axis length replaced by indices.len().
                let mut src_shape = data_shape.clone();
                src_shape[axis] = indices.len();
                let var_size = checked_shape_product(&src_shape).ok_or_else(|| {
                    NyError::InvalidSpec("IndexAdd: src shape overflow".to_string())
                })?;
                let map = build_index_add_src_map(&data_shape, &src_shape, axis, indices)?;
                // `data_const` is copied, never accumulated: exact.
                let shift_flat = flatten_row_major(data_const);
                additive_scatter_backward(
                    node_lb,
                    output_size,
                    var_size,
                    &shift_flat,
                    &shift_flat,
                    &VarToOut::Map(map),
                )
            }
        }
    }
}

impl BoundPropagation for IndexAddLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input])
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.crown_backward(bounds).map(Cow::Owned)
    }
}

/// Infer the IndexAdd data (== output) shape for the data-variable case from the
/// src constant shape and the flat output size. data and src match on all axes
/// except `axis`, where data has the full axis length and src has `indices.len()`.
fn infer_index_add_data_shape(
    src_shape: &[usize],
    output_size: usize,
    axis: i64,
    _index_len: usize,
) -> Result<Vec<usize>> {
    let ndim = src_shape.len();
    let axis = super::super::common::resolve_axis(axis, ndim, "IndexAdd")?;
    // Product of all src dims except `axis`.
    let mut other: usize = 1;
    for (d, &s) in src_shape.iter().enumerate() {
        if d != axis {
            other = other
                .checked_mul(s)
                .ok_or_else(|| NyError::InvalidSpec("IndexAdd: src shape overflow".to_string()))?;
        }
    }
    if other == 0 || !output_size.is_multiple_of(other) {
        return Err(NyError::UnsupportedOp(
            "IndexAdd CROWN (data-variable): cannot infer data shape from src".to_string(),
        ));
    }
    let axis_len = output_size / other;
    let mut data_shape = src_shape.to_vec();
    data_shape[axis] = axis_len;
    Ok(data_shape)
}

#[cfg(test)]
#[path = "accumulate_tests.rs"]
mod accumulate_tests;
