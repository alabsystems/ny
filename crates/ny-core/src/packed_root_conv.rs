// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Ordered paired-f32 contraction for an already admitted compact root plan.
//! This interface transfers arithmetic, not proof authority: the caller still
//! owns the coordinate union, scalar bias, error envelope and root publication.
//! Each coefficient starts at +0 and uses ascending taps with separate RN-f32
//! multiplication/addition. D is signed; only the weight is made absolute.

use crate::{NyError, Result};
use std::time::Instant;

/// One consecutive destination interval. Source positions are the cumulative
/// lengths of preceding runs within the same spatial/tap bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    /// First destination coordinate (the physical bias is excluded).
    pub dest: usize,
    /// Number of consecutive coordinates.
    pub len: usize,
}

/// Borrowed, call-local arithmetic inputs. No engine may retain these borrows.
pub struct Request<'a> {
    /// Channel-major source rows, including their last physical bias column.
    pub m: &'a [f32],
    /// Signed source D coefficients, with the same layout as M.
    pub d: &'a [f32],
    /// Output-channel-major captured f32 weights.
    pub weights: &'a [f32],
    /// Tap-major gathered source rows; usize::MAX denotes padding.
    pub gather: &'a [usize],
    /// Run offsets per [output spatial, spatial tap], followed by one sentinel.
    pub offsets: &'a [usize],
    /// Ordered, nonoverlapping runs, sharing geometry across input channels.
    pub runs: &'a [Run],
    /// Source row width including physical bias.
    pub source_stride: usize,
    /// Result row width including physical bias.
    pub output_stride: usize,
    /// Number of output spatial sites.
    pub output_spatial: usize,
    /// Number of output channels.
    pub output_channels: usize,
    /// Spatial taps per input channel.
    pub spatial_taps: usize,
    /// Full ascending contraction length (input channels times spatial taps).
    pub taps: usize,
}

/// Explicit incremental payload limits, in addition to live-device admission.
/// These do not claim to bound whole-process RSS or external device allocations.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Absolute caller deadline; initialization must happen before it exists.
    pub deadline: Option<Instant>,
    /// New host payload: paired output, converted gathers and expanded inverse.
    pub max_new_host_bytes: usize,
    /// All seven device buffers, including borrowed inputs' device copies.
    pub max_device_bytes: usize,
    /// Maximum paired multiply/add steps per launch (two lanes per tap).
    pub max_dispatch_steps: usize,
}

/// Owned complete paired output. A partial transaction may never return this.
pub struct Output {
    /// Channel-major M coefficients.
    pub m: Vec<f32>,
    /// Channel-major signed D coefficients.
    pub d: Vec<f32>,
}

/// Optional accelerator. None means a safe pre-dispatch refusal; errors,
/// deadline expiry and arithmetic failures must remain errors at the caller.
/// Implementations must drain every queued access before returning or unwinding.
pub trait Engine: Send + Sync {
    /// Contract an admitted plan, preserving its exact per-coefficient order.
    fn contract(&self, request: &Request<'_>, limits: Limits) -> Result<Option<Output>>;
}

/// A checked shape and byte account; constructing it never allocates buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Elements in each source lane.
    pub source_len: usize,
    /// Elements in each result lane, also the number of independent owners.
    pub output_len: usize,
    /// u32 entries in the expanded inverse map. A bias-only plan has zero
    /// entries and must use no allocation (a null device pointer is unread).
    pub inverse_len: usize,
    /// u32 entries in the device gather.
    pub gather_len: usize,
    /// Incremental host payload including both result lanes.
    pub new_host_bytes: usize,
    /// Complete simultaneous device payload.
    pub device_bytes: usize,
    /// Work-limited owners per dispatch, each retaining its full contraction.
    /// Device adapters must additionally clamp this to legal launch dimensions.
    pub owners_per_dispatch: usize,
}

/// Poll at bounded host-work boundaries, including before any allocation.
pub fn check_deadline(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|end| Instant::now() >= end) {
        Err(NyError::DeadlineExceeded("packed root convolution".into()))
    } else {
        Ok(())
    }
}

impl Request<'_> {
    /// Validate shape, gather sharing, run ranges, finite sources and admitted
    /// weights. None declines representational/byte/work limits before upload.
    pub fn layout(&self, limits: Limits) -> Result<Option<Layout>> {
        check_deadline(limits.deadline)?;
        let malformed = || NyError::InvalidSpec("malformed packed root convolution".into());
        let (ss, os, p, co, st, taps) = (
            self.source_stride,
            self.output_stride,
            self.output_spatial,
            self.output_channels,
            self.spatial_taps,
            self.taps,
        );
        if [ss, os, p, co, st, taps].contains(&0) || taps % st != 0 {
            return Err(malformed());
        }
        if [ss, os, p, co, st, taps]
            .iter()
            .any(|&n| n >= u32::MAX as usize)
        {
            return Ok(None);
        }
        let checked_product =
            |a: usize, b: usize| a.checked_mul(b).filter(|&n| n <= isize::MAX as usize);
        let Some(out_len) = checked_product(co, p).and_then(|v| checked_product(v, os)) else {
            return Ok(None);
        };
        let Some(gather_len) = checked_product(taps, p) else {
            return Ok(None);
        };
        let Some(weight_len) = checked_product(co, taps) else {
            return Ok(None);
        };
        let Some(buckets) = checked_product(p, st) else {
            return Ok(None);
        };
        let Some(inverse_len) = checked_product(buckets, os - 1) else {
            return Ok(None);
        };
        if self.m.len() != self.d.len()
            || self.m.is_empty()
            || self.m.len() % ss != 0
            || self.gather.len() != gather_len
            || self.weights.len() != weight_len
            || self.offsets.len() != buckets + 1
            || self.offsets.first() != Some(&0)
            || self.offsets.last() != Some(&self.runs.len())
        {
            return Err(malformed());
        }
        let rows = self.m.len() / ss;
        let ci = taps / st;
        if rows % ci != 0 {
            return Err(malformed());
        }
        if rows >= u32::MAX as usize {
            return Ok(None);
        }
        let input_spatial = rows / ci;
        let Some(host_words) = out_len
            .checked_mul(2)
            .and_then(|v| v.checked_add(gather_len))
            .and_then(|v| v.checked_add(inverse_len))
        else {
            return Ok(None);
        };
        let Some(host_bytes) = checked_product(host_words, 4) else {
            return Ok(None);
        };
        let Some(device_words) = self
            .m
            .len()
            .checked_mul(2)
            .and_then(|v| v.checked_add(weight_len))
            .and_then(|v| v.checked_add(host_words))
        else {
            return Ok(None);
        };
        let Some(device_bytes) = checked_product(device_words, 4) else {
            return Ok(None);
        };
        let Some(paired_taps) = taps.checked_mul(2) else {
            return Ok(None);
        };
        let owners_per_dispatch = limits.max_dispatch_steps / paired_taps;
        if host_bytes > limits.max_new_host_bytes
            || device_bytes > limits.max_device_bytes
            || owners_per_dispatch == 0
        {
            return Ok(None);
        }
        for sp in 0..p {
            check_deadline(limits.deadline)?;
            for tap in 0..st {
                let first = self.gather[tap * p + sp];
                if first != usize::MAX && first >= input_spatial {
                    return Err(malformed());
                }
                for ic in 0..ci {
                    if ic % 256 == 0 {
                        check_deadline(limits.deadline)?;
                    }
                    let expected = if first == usize::MAX {
                        first
                    } else {
                        ic * input_spatial + first
                    };
                    if self.gather[(ic * st + tap) * p + sp] != expected {
                        return Err(malformed());
                    }
                }
                let bucket = sp * st + tap;
                let start = self.offsets[bucket];
                let end = self.offsets[bucket + 1];
                if start > end || end > self.runs.len() || (first == usize::MAX && start != end) {
                    return Err(malformed());
                }
                let mut previous_end = 0;
                let mut source_end = 0usize;
                for (i, run) in self.runs[start..end].iter().enumerate() {
                    if i % 256 == 0 {
                        check_deadline(limits.deadline)?;
                    }
                    let Some(dest_end) = run.dest.checked_add(run.len) else {
                        return Err(malformed());
                    };
                    let Some(next_source) = source_end.checked_add(run.len) else {
                        return Err(malformed());
                    };
                    if run.len == 0
                        || run.dest < previous_end
                        || dest_end >= os
                        || next_source >= ss
                    {
                        return Err(malformed());
                    }
                    previous_end = dest_end;
                    source_end = next_source;
                }
            }
        }
        for chunk in self.m.chunks(16384).chain(self.d.chunks(16384)) {
            check_deadline(limits.deadline)?;
            if chunk
                .iter()
                .any(|v| v.to_bits() & 0x7f80_0000 == 0x7f80_0000)
            {
                return Err(NyError::NumericalInstability(
                    "nonfinite packed root source".into(),
                ));
            }
        }
        for chunk in self.weights.chunks(16384) {
            check_deadline(limits.deadline)?;
            if chunk
                .iter()
                .any(|v| v.to_bits() & 0x7fff_ffff != 0 && !v.is_normal())
            {
                return Err(NyError::InvalidSpec(
                    "packed root requires zero or normal f32 weights".into(),
                ));
            }
        }
        Ok(Some(Layout {
            source_len: self.m.len(),
            output_len: out_len,
            inverse_len,
            gather_len,
            new_host_bytes: host_bytes,
            device_bytes,
            owners_per_dispatch,
        }))
    }
}

#[cfg(test)]
mod tests;
