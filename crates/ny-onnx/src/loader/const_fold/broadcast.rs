// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};

/// ONNX `Where` over an f32 condition view and any cloneable value payload.
///
/// The condition stays f32 because that is how folded BOOL results are
/// published; the selected values are generic so an exact INT64 payload can be
/// selected without a detour through f32.
pub(crate) fn broadcast_where<T: Clone>(
    cond: &ArrayD<f32>,
    true_val: &ArrayD<T>,
    false_val: &ArrayD<T>,
) -> Option<ArrayD<T>> {
    if cond.shape() == true_val.shape() && cond.shape() == false_val.shape() {
        return Some(
            ndarray::Zip::from(cond)
                .and(true_val)
                .and(false_val)
                .map_collect(|&c, t, f| if c != 0.0 { t.clone() } else { f.clone() }),
        );
    }

    let out_ndim = cond.ndim().max(true_val.ndim()).max(false_val.ndim());
    let mut out_shape = vec![1usize; out_ndim];
    let shapes = [cond.shape(), true_val.shape(), false_val.shape()];
    for (i, dim) in out_shape.iter_mut().enumerate().rev() {
        let mut resolved_dim = 1usize;
        for shape in &shapes {
            let shape_dim = if i >= out_ndim - shape.len() {
                shape[i - (out_ndim - shape.len())]
            } else {
                1
            };
            if shape_dim == resolved_dim || shape_dim == 1 {
                continue;
            }
            if resolved_dim == 1 {
                resolved_dim = shape_dim;
            } else {
                return None;
            }
        }
        *dim = resolved_dim;
    }

    const MAX_BROADCAST_ELEMENTS: usize = 10_000_000;
    let total = ny_core::checked_shape_product(&out_shape)?;
    if total > MAX_BROADCAST_ELEMENTS {
        return None;
    }

    let cond_view = cond.broadcast(IxDyn(&out_shape))?;
    let true_view = true_val.broadcast(IxDyn(&out_shape))?;
    let false_view = false_val.broadcast(IxDyn(&out_shape))?;
    Some(
        ndarray::Zip::from(&cond_view)
            .and(&true_view)
            .and(&false_view)
            .map_collect(|&c, t, f| if c != 0.0 { t.clone() } else { f.clone() }),
    )
}

pub(crate) fn broadcast_binop<T: Copy>(
    a: &ArrayD<T>,
    b: &ArrayD<T>,
    op: fn(T, T) -> T,
) -> Option<ArrayD<T>> {
    if a.shape() == b.shape() {
        return Some(ndarray::Zip::from(a).and(b).map_collect(|&x, &y| op(x, y)));
    }
    let out_ndim = a.ndim().max(b.ndim());
    let mut out_shape = vec![1usize; out_ndim];
    for (i, dim) in out_shape.iter_mut().enumerate().rev() {
        let a_dim = if i >= out_ndim - a.ndim() {
            a.shape()[i - (out_ndim - a.ndim())]
        } else {
            1
        };
        let b_dim = if i >= out_ndim - b.ndim() {
            b.shape()[i - (out_ndim - b.ndim())]
        } else {
            1
        };
        if a_dim == b_dim {
            *dim = a_dim;
        } else if a_dim == 1 {
            *dim = b_dim;
        } else if b_dim == 1 {
            *dim = a_dim;
        } else {
            return None;
        }
    }

    const MAX_BROADCAST_ELEMENTS: usize = 10_000_000;
    let total = ny_core::checked_shape_product(&out_shape)?;
    if total > MAX_BROADCAST_ELEMENTS {
        return None;
    }

    let a_view = a.broadcast(IxDyn(&out_shape))?;
    let b_view = b.broadcast(IxDyn(&out_shape))?;
    Some(
        ndarray::Zip::from(&a_view)
            .and(&b_view)
            .map_collect(|&x, &y| op(x, y)),
    )
}

/// Checked variant used by exact integer constant folding.  Unlike the f32
/// evaluator, integer operations must be able to decline a single element on
/// division-by-zero or overflow without evaluating the remaining broadcast.
pub(crate) fn broadcast_binop_checked<T: Copy>(
    a: &ArrayD<T>,
    b: &ArrayD<T>,
    op: fn(T, T) -> Option<T>,
) -> Option<ArrayD<T>> {
    let out_ndim = a.ndim().max(b.ndim());
    let mut out_shape = vec![1usize; out_ndim];
    for (i, dim) in out_shape.iter_mut().enumerate().rev() {
        let a_dim = if i >= out_ndim - a.ndim() {
            a.shape()[i - (out_ndim - a.ndim())]
        } else {
            1
        };
        let b_dim = if i >= out_ndim - b.ndim() {
            b.shape()[i - (out_ndim - b.ndim())]
        } else {
            1
        };
        if a_dim == b_dim {
            *dim = a_dim;
        } else if a_dim == 1 {
            *dim = b_dim;
        } else if b_dim == 1 {
            *dim = a_dim;
        } else {
            return None;
        }
    }

    const MAX_BROADCAST_ELEMENTS: usize = 10_000_000;
    let total = ny_core::checked_shape_product(&out_shape)?;
    if total > MAX_BROADCAST_ELEMENTS {
        return None;
    }

    let a_view = a.broadcast(IxDyn(&out_shape))?;
    let b_view = b.broadcast(IxDyn(&out_shape))?;
    let mut values = Vec::new();
    values.try_reserve_exact(total).ok()?;
    for (&x, &y) in a_view.iter().zip(b_view.iter()) {
        values.push(op(x, y)?);
    }
    ArrayD::from_shape_vec(IxDyn(&out_shape), values).ok()
}
