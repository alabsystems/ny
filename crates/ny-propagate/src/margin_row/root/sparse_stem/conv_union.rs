// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Compact Conv outputs in the original input frame.
//! This kernel does not add bias/fold/error charges or publish root state.
//! Its caller must retain the existing shared Conv error and bias/fold path.

use super::*;

use ny_core::packed_root_conv::{Request as PackedRequest, Run as ConvRun};

// Source coordinates are a valid prefix, so source offsets are the cumulative
// lengths of preceding runs. Each bucket starts a new prefix at zero.
fn merge_runs(source: &[usize], union: &[usize], mut emit: impl FnMut(ConvRun)) {
    let mut dest = 0;
    let mut pending: Option<ConvRun> = None;
    for &i in source {
        if i == usize::MAX {
            break;
        }
        while union[dest] < i {
            dest += 1;
        }
        debug_assert_eq!(union[dest], i);
        match &mut pending {
            Some(run) if run.dest + run.len == dest => run.len += 1,
            _ => {
                if let Some(run) = pending.take() {
                    emit(run);
                }
                pending = Some(ConvRun { dest, len: 1 });
            }
        }
    }
    if let Some(run) = pending {
        emit(run);
    }
}

// Same nonfused operations and per-update conversions as the scalar reference.
// Contiguous source/destination slices permit vectorization without regrouping
// the products belonging to any individual coefficient.
#[inline]
fn axpy_stored<T: TabElem>(dst: &mut [T], w: f32, src: &[T]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d = T::from_acc(d.wide() as f32 + w * s.wide() as f32);
    }
}

struct ConvUnionPlan {
    // A GPU-only plan must never enter the compact CPU fallback on refusal.
    cpu_eligible: bool,
    coordinates: Array2<usize>,
    // Run ranges per [output spatial, spatial tap], shared across all channels.
    offsets: Vec<usize>,
    runs: Vec<ConvRun>,
    spatial_taps: usize,
    weights: Vec<f32>,
}

fn allocation_bytes(elements: usize, width: usize) -> Option<usize> {
    elements
        .checked_mul(width)
        .filter(|&n| n <= isize::MAX as usize)
}

impl<T: TabElem> PackedRows<T> {
    fn plan_conv(&self, c: &ConvOp, deadline: Option<Instant>) -> Result<Option<ConvUnionPlan>> {
        check_deadline(deadline)?;
        let unsupported = || Ok::<_, NyError>(None);
        if !T::F32_STORE || self.p == 0 || self.n_in == 0 {
            return unsupported();
        }
        let (co, ci, kh, kw) = c.kernel;
        let Some(spatial_taps) = kh.checked_mul(kw).filter(|&n| n > 0) else {
            return unsupported();
        };
        let Some(k) = ci.checked_mul(spatial_taps) else {
            return unsupported();
        };
        let Some(p_out) = c.oshape.1.checked_mul(c.oshape.2).filter(|&n| n > 0) else {
            return unsupported();
        };
        let Some(n_out) = co.checked_mul(p_out) else {
            return unsupported();
        };
        let Some(n_aug) = self.n_in.checked_add(1) else {
            return unsupported();
        };
        let slots = self.bias_column();
        if co == 0
            || ci == 0
            || c.k_fwd != k
            || c.ishape.0 != ci
            || c.oshape.0 != co
            || c.ishape.1.checked_mul(c.ishape.2) != Some(self.p)
            || ci.checked_mul(self.p) != Some(self.m.nrows())
            || self.m.dim() != self.d.dim()
            || slots.checked_add(1) != Some(self.m.ncols())
            || self.coordinates.nrows() != self.p
            || k.checked_mul(p_out) != Some(c.gather.len())
            || co.checked_mul(k) != Some(c.wmat.len())
        {
            return unsupported();
        }
        // A plan is local to this call, so it cannot be reused with different
        // coefficients, geometry, or captured weights after admission.
        let mut weights = Vec::with_capacity(c.wmat.len());
        for &w in &c.wmat {
            let captured = w as f32;
            // Bitwise original zero classification avoids f64 DAZ erasure.
            if w.to_bits() & 0x7fff_ffff_ffff_ffff != 0 && !captured.is_normal() {
                return unsupported();
            }
            weights.push(captured);
        }
        for source_sp in 0..self.p {
            check_deadline(deadline)?;
            let mut previous = None;
            let mut padding = false;
            for &i in self.coordinates.row(source_sp).iter() {
                if i == usize::MAX {
                    padding = true;
                } else {
                    if padding || i >= self.n_in || previous.is_some_and(|prev| prev >= i) {
                        return unsupported();
                    }
                    previous = Some(i);
                }
            }
        }
        // Prove sharing from actual gathers, including padding. Geometry
        // metadata alone is insufficient, particularly for transposed Conv.
        for sp in 0..p_out {
            check_deadline(deadline)?;
            for tap in 0..spatial_taps {
                let first = c.gather[tap * p_out + sp];
                if first != usize::MAX && first >= self.p {
                    return unsupported();
                }
                for ic in 0..ci {
                    let expected = if first == usize::MAX {
                        usize::MAX
                    } else {
                        ic * self.p + first
                    };
                    if c.gather[(ic * spatial_taps + tap) * p_out + sp] != expected {
                        return unsupported();
                    }
                }
            }
        }
        let Some(dense_bytes) = n_out
            .checked_mul(n_aug)
            .and_then(|n| allocation_bytes(n, 2 * size_of::<T>()))
        else {
            return unsupported();
        };
        let Some(buckets) = p_out.checked_mul(spatial_taps) else {
            return unsupported();
        };
        let Some(offset_len) = buckets.checked_add(1) else {
            return unsupported();
        };
        let Some(offset_bytes) = allocation_bytes(offset_len, size_of::<usize>()) else {
            return unsupported();
        };
        let Some(old_map_bytes) = buckets
            .checked_mul(slots)
            .and_then(|n| allocation_bytes(n, size_of::<usize>()))
        else {
            return unsupported();
        };
        if offset_bytes >= dense_bytes || allocation_bytes(self.n_in, size_of::<usize>()).is_none()
        {
            return unsupported();
        }
        let mut seen = vec![usize::MAX; self.n_in];
        let mut unions = Vec::with_capacity(p_out);
        let mut out_slots = 0;
        for sp in 0..p_out {
            check_deadline(deadline)?;
            let mut union = Vec::new();
            for tap in 0..spatial_taps {
                let gi = c.gather[tap * p_out + sp];
                if gi == usize::MAX {
                    continue;
                }
                for &i in self.coordinates.row(gi).iter() {
                    if i == usize::MAX {
                        break;
                    }
                    if seen[i] != sp {
                        seen[i] = sp;
                        union.push(i);
                    }
                }
            }
            union.sort_unstable();
            out_slots = out_slots.max(union.len());
            if out_slots >= self.n_in {
                return unsupported();
            }
            unions.push(union);
        }
        let Some(coordinate_len) = p_out.checked_mul(out_slots) else {
            return unsupported();
        };
        let Some(coordinate_bytes) = allocation_bytes(coordinate_len, size_of::<usize>()) else {
            return unsupported();
        };
        let Some(payload_bytes) = out_slots
            .checked_add(1)
            .and_then(|w| n_out.checked_mul(w))
            .and_then(|n| allocation_bytes(n, 2 * size_of::<T>()))
        else {
            return unsupported();
        };
        let Some(fixed_bytes) = payload_bytes
            .checked_add(coordinate_bytes)
            .and_then(|n| n.checked_add(offset_bytes))
        else {
            return unsupported();
        };
        if fixed_bytes >= dense_bytes {
            return unsupported();
        }
        // Preserve the original CPU admission boundary. A qualified GPU may
        // examine a cheaper compressed plan, subject to its own live limits.
        let cpu_eligible = payload_bytes
            .checked_add(coordinate_bytes)
            .and_then(|n| n.checked_add(old_map_bytes))
            .is_some_and(|n| n < dense_bytes);
        if !cpu_eligible && !crate::margin_row::packed_root_engine::ready() {
            return unsupported();
        }
        // Count before allocating. Fragmented maps can cost more than scalar
        // indices; actual descriptors and offsets are part of admission.
        let max_runs = (dense_bytes - fixed_bytes - 1) / size_of::<ConvRun>();
        let mut run_count = 0usize;
        for (sp, union) in unions.iter().enumerate() {
            check_deadline(deadline)?;
            for tap in 0..spatial_taps {
                let gi = c.gather[tap * p_out + sp];
                if gi != usize::MAX {
                    merge_runs(self.coordinates.row(gi).as_slice().unwrap(), union, |_| {
                        run_count = run_count.saturating_add(1);
                    });
                    if run_count > max_runs {
                        return unsupported();
                    }
                }
            }
        }
        if allocation_bytes(run_count, size_of::<ConvRun>()).is_none() {
            return unsupported();
        }
        // This bounds result payload + coordinates + run metadata, not peak
        // allocation: seen/unions/captured weights remain planning scratch.
        let mut coordinates = Array2::from_elem((p_out, out_slots), usize::MAX);
        let mut offsets = Vec::with_capacity(offset_len);
        let mut runs = Vec::with_capacity(run_count);
        for (sp, union) in unions.iter().enumerate() {
            check_deadline(deadline)?;
            for (slot, &i) in union.iter().enumerate() {
                coordinates[[sp, slot]] = i;
            }
            for tap in 0..spatial_taps {
                offsets.push(runs.len());
                let gi = c.gather[tap * p_out + sp];
                if gi == usize::MAX {
                    continue;
                }
                merge_runs(self.coordinates.row(gi).as_slice().unwrap(), union, |run| {
                    runs.push(run)
                });
            }
        }
        offsets.push(runs.len());
        debug_assert_eq!(offsets.len(), offset_len);
        debug_assert_eq!(runs.len(), run_count);
        Ok(Some(ConvUnionPlan {
            cpu_eligible,
            coordinates,
            offsets,
            runs,
            spatial_taps,
            weights,
        }))
    }

    /// Preserve every original tap and f32 accumulation operation. Return None
    /// before publication when the representation/weights/size are unsupported.
    /// Bias, inherited error and fold charges are deliberately caller-owned.
    pub(in crate::margin_row::root) fn contract_packed(
        &self,
        c: &ConvOp,
        deadline: Option<Instant>,
    ) -> Result<Option<Self>> {
        let Some(plan) = self.plan_conv(c, deadline)? else {
            return Ok(None);
        };
        if self.m.iter().chain(&self.d).any(|v| !v.wide().is_finite()) {
            return Err(NyError::NumericalInstability(
                "margin_row nonfinite packed Conv source".into(),
            ));
        }
        let p_out = plan.coordinates.nrows();
        let bias = plan.coordinates.ncols();
        let stride = bias + 1;
        let source_bias = self.bias_column();
        let source_stride = source_bias + 1;
        if let (Some(ms), Some(ds)) = (
            T::as_f32_slice(self.m.as_slice().expect("packed standard layout")),
            T::as_f32_slice(self.d.as_slice().expect("packed standard layout")),
        ) {
            let request = PackedRequest {
                m: ms,
                d: ds,
                weights: &plan.weights,
                gather: &c.gather,
                offsets: &plan.offsets,
                runs: &plan.runs,
                source_stride,
                output_stride: stride,
                output_spatial: p_out,
                output_channels: c.oshape.0,
                spatial_taps: plan.spatial_taps,
                taps: c.k_fwd,
            };
            if let Some(output) =
                crate::margin_row::packed_root_engine::contract(&request, deadline)?
            {
                check_deadline(deadline)?;
                let expected = c
                    .oshape
                    .0
                    .checked_mul(p_out)
                    .and_then(|n| n.checked_mul(stride))
                    .ok_or_else(|| {
                        NyError::InternalError("packed accelerator output shape overflow".into())
                    })?;
                if output.m.len() != expected || output.d.len() != expected {
                    return Err(NyError::InternalError(
                        "packed accelerator output length mismatch".into(),
                    ));
                }
                for chunk in output.m.chunks(16384).chain(output.d.chunks(16384)) {
                    check_deadline(deadline)?;
                    if chunk
                        .iter()
                        .any(|v| v.to_bits() & 0x7f80_0000 == 0x7f80_0000)
                    {
                        return Err(NyError::NumericalInstability(
                            "nonfinite packed accelerator result".into(),
                        ));
                    }
                }
                let convert = |values| {
                    T::from_f32_vec(values).ok_or_else(|| {
                        NyError::InternalError("packed accelerator storage mismatch".into())
                    })
                };
                let shape = (c.oshape.0 * p_out, stride);
                let m = Array2::from_shape_vec(shape, convert(output.m)?)
                    .map_err(|_| NyError::InternalError("packed M output shape mismatch".into()))?;
                let d = Array2::from_shape_vec(shape, convert(output.d)?)
                    .map_err(|_| NyError::InternalError("packed D output shape mismatch".into()))?;
                check_deadline(deadline)?;
                return Ok(Some(Self {
                    n_in: self.n_in,
                    p: p_out,
                    coordinates: plan.coordinates,
                    m,
                    d,
                }));
            }
        }
        check_deadline(deadline)?;
        if !plan.cpu_eligible {
            return Ok(None);
        }
        let mut m = Array2::from_elem((c.oshape.0 * p_out, stride), T::ZERO);
        let mut d = Array2::from_elem(m.raw_dim(), T::ZERO);
        let ms = self.m.as_slice().expect("packed standard layout");
        let ds = self.d.as_slice().expect("packed standard layout");
        let co = c.oshape.0;
        // Window coordinates as well as channels: whole rows eventually exceed
        // L1 even for one channel. Each worker's paired accumulator is at most
        // 16 channels * 256 columns * 2 lanes * sizeof(f32) = 32KiB.
        // Windows contain independent coefficients; each coefficient still
        // receives exactly the original ascending-tap sequence.
        let channel_tile = co.min(16);
        let column_tile = stride.min(256);
        let mut m3 = m
            .view_mut()
            .into_shape_with_order((co, p_out, stride))
            .unwrap();
        let mut d3 = d
            .view_mut()
            .into_shape_with_order((co, p_out, stride))
            .unwrap();
        m3.axis_iter_mut(ndarray::Axis(1))
            .into_par_iter()
            .zip(d3.axis_iter_mut(ndarray::Axis(1)).into_par_iter())
            .enumerate()
            .try_for_each_init(
                || {
                    (
                        vec![T::ZERO; channel_tile * column_tile],
                        vec![T::ZERO; channel_tile * column_tile],
                    )
                },
                |(am, ad), (sp, (mut mout, mut dout))| {
                    check_deadline(deadline)?;
                    for first_oc in (0..co).step_by(channel_tile) {
                        let channels = (co - first_oc).min(channel_tile);
                        for first_col in (0..stride).step_by(column_tile) {
                            check_deadline(deadline)?;
                            let columns = (stride - first_col).min(column_tile);
                            let col_end = first_col + columns;
                            am[..channels * columns].fill(T::ZERO);
                            ad[..channels * columns].fill(T::ZERO);
                            for t in 0..c.k_fwd {
                                if t % 64 == 0 {
                                    check_deadline(deadline)?;
                                }
                                let gi = c.gather[t * p_out + sp];
                                if gi == usize::MAX {
                                    continue;
                                }
                                let base = gi * source_stride;
                                let sm = &ms[base..base + source_stride];
                                let sd = &ds[base..base + source_stride];
                                let bucket = sp * plan.spatial_taps + t % plan.spatial_taps;
                                let runs =
                                    &plan.runs[plan.offsets[bucket]..plan.offsets[bucket + 1]];
                                let mut weights = [0.0f32; 16];
                                let mut abs_weights = [0.0f32; 16];
                                for local_oc in 0..channels {
                                    let w = plan.weights[(first_oc + local_oc) * c.k_fwd + t];
                                    weights[local_oc] = w;
                                    abs_weights[local_oc] = w.abs();
                                }
                                // Clip each shared run once, then reuse the same source
                                // segment and mapping across the output-channel tile.
                                let mut src_start = 0;
                                for run in runs {
                                    if run.dest >= col_end {
                                        break;
                                    }
                                    let start = run.dest.max(first_col);
                                    let end = (run.dest + run.len).min(col_end);
                                    if start < end {
                                        let source = src_start + start - run.dest;
                                        let dest = start - first_col;
                                        let len = end - start;
                                        for local_oc in 0..channels {
                                            let w = weights[local_oc];
                                            if w == 0.0 {
                                                continue;
                                            }
                                            let offset = local_oc * columns + dest;
                                            axpy_stored(
                                                &mut am[offset..offset + len],
                                                w,
                                                &sm[source..source + len],
                                            );
                                            axpy_stored(
                                                &mut ad[offset..offset + len],
                                                abs_weights[local_oc],
                                                &sd[source..source + len],
                                            );
                                        }
                                    }
                                    src_start += run.len;
                                }
                                // Bias belongs to exactly the final coordinate window.
                                if first_col <= bias && bias < col_end {
                                    for local_oc in 0..channels {
                                        let w = weights[local_oc];
                                        if w == 0.0 {
                                            continue;
                                        }
                                        let b = local_oc * columns + bias - first_col;
                                        am[b] = T::from_acc(
                                            am[b].wide() as f32 + w * sm[source_bias].wide() as f32,
                                        );
                                        ad[b] = T::from_acc(
                                            ad[b].wide() as f32
                                                + abs_weights[local_oc]
                                                    * sd[source_bias].wide() as f32,
                                        );
                                    }
                                }
                            }
                            for local_oc in 0..channels {
                                let oc = first_oc + local_oc;
                                mout.row_mut(oc).as_slice_mut().unwrap()[first_col..col_end]
                                    .copy_from_slice(
                                        &am[local_oc * columns..(local_oc + 1) * columns],
                                    );
                                dout.row_mut(oc).as_slice_mut().unwrap()[first_col..col_end]
                                    .copy_from_slice(
                                        &ad[local_oc * columns..(local_oc + 1) * columns],
                                    );
                            }
                        }
                    }
                    Ok::<_, NyError>(())
                },
            )?;
        if m.iter().chain(&d).any(|v| !v.wide().is_finite()) {
            return Err(NyError::NumericalInstability(
                "margin_row nonfinite packed Conv result".into(),
            ));
        }
        Ok(Some(Self {
            n_in: self.n_in,
            p: p_out,
            coordinates: plan.coordinates,
            m,
            d,
        }))
    }
}

#[cfg(test)]
mod tests;
