// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};

pub(crate) fn broadcast_where(
    cond: &ArrayD<f32>,
    true_val: &ArrayD<f32>,
    false_val: &ArrayD<f32>,
) -> Option<ArrayD<f32>> {
    if cond.shape() == true_val.shape() && cond.shape() == false_val.shape() {
        return Some(
            ndarray::Zip::from(cond)
                .and(true_val)
                .and(false_val)
                .map_collect(|&c, &t, &f| if c != 0.0 { t } else { f }),
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
            .map_collect(|&c, &t, &f| if c != 0.0 { t } else { f }),
    )
}

pub(crate) fn broadcast_binop(
    a: &ArrayD<f32>,
    b: &ArrayD<f32>,
    op: fn(f32, f32) -> f32,
) -> Option<ArrayD<f32>> {
    if a.shape() == b.shape() {
        return Some(ndarray::Zip::from(a).and(b).map_collect(|&x, &y| op(x, y)));
    }
    if a.len() == 1 {
        let scalar = a.iter().next().copied().unwrap_or(0.0);
        return Some(b.mapv(|v| op(scalar, v)));
    }
    if b.len() == 1 {
        let scalar = b.iter().next().copied().unwrap_or(0.0);
        return Some(a.mapv(|v| op(v, scalar)));
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
