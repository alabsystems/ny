// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Owned, word-carrying device frontier for an Add/Sub intermediate sweep.
//!
//! This is intentionally a different type from the legacy seg-resident
//! `ResidentCoeffBufs`. That older transport has eight f32 buffers and cannot
//! carry the four per-entry taint twins or the per-row accumulator; its merge is
//! therefore verdict-dead. A [`DeviceSweepCarrier`] always owns all thirteen
//! buffers and is deliberately not `Clone`: duplicating a `wgpu::Buffer` handle
//! would alias one allocation while making memory accounting appear to own two.
//!
//! The first implementation admits only full IEEE/DenormPreserve authority.
//! Charged/flush authority must cleanly decline until the elementwise merge's
//! DAZ/FTZ charge has its own oracle and route-table audit.

use ny_core::{NyError, Result};

/// Arithmetic authority currently implemented by the DAG carrier kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SweepDagAuthority {
    FullIeee,
}

impl SweepDagAuthority {
    /// Select the presently audited route. A charged-only device returns
    /// `None`, which the caller must translate to a pre-dispatch clean decline.
    pub(super) fn select(full_ieee: bool, _charged_flush: bool) -> Option<Self> {
        full_ieee.then_some(Self::FullIeee)
    }
}

/// Exact logical sizes of the thirteen carrier buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SweepCarrierLayout {
    pub(super) rows: usize,
    pub(super) dim: usize,
    pub(super) matrix_elements: usize,
    /// Size of each coefficient/error value or word buffer.
    pub(super) matrix_bytes: u64,
    /// Size of each bias value/error or row-word buffer.
    pub(super) row_bytes: u64,
    /// Eight matrix buffers plus five row buffers.
    pub(super) logical_bytes: usize,
}

impl SweepCarrierLayout {
    pub(super) fn new(rows: usize, dim: usize) -> Result<Self> {
        if rows == 0 || dim == 0 {
            return Err(invalid("sweep carrier rows and dim must be nonzero"));
        }
        let matrix_elements = rows
            .checked_mul(dim)
            .ok_or_else(|| invalid("sweep carrier matrix element count overflow"))?;
        let matrix_bytes_usize = matrix_elements
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| invalid("sweep carrier matrix byte count overflow"))?;
        let row_bytes_usize = rows
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| invalid("sweep carrier row byte count overflow"))?;
        let logical_bytes = matrix_bytes_usize
            .checked_mul(8)
            .and_then(|bytes| {
                row_bytes_usize
                    .checked_mul(5)
                    .and_then(|rows| bytes.checked_add(rows))
            })
            .ok_or_else(|| invalid("sweep carrier logical byte count overflow"))?;
        Ok(Self {
            rows,
            dim,
            matrix_elements,
            matrix_bytes: u64::try_from(matrix_bytes_usize)
                .map_err(|_| invalid("sweep carrier matrix bytes do not fit u64"))?,
            row_bytes: u64::try_from(row_bytes_usize)
                .map_err(|_| invalid("sweep carrier row bytes do not fit u64"))?,
            logical_bytes,
        })
    }

    /// Check the two limits relevant to every carrier buffer before allocation.
    pub(super) fn validate_device_limits(
        self,
        max_buffer_size: u64,
        max_storage_binding_size: u64,
    ) -> Result<Self> {
        let capacity = max_buffer_size.min(max_storage_binding_size);
        if self.matrix_bytes > capacity {
            return Err(NyError::GpuBatchCapacityExceeded {
                requested: usize::try_from(self.matrix_bytes).unwrap_or(usize::MAX),
                capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
                unit: "bytes per worded sweep carrier buffer",
                site: "WGPU intermediate DAG carrier preflight",
            });
        }
        if self.row_bytes > capacity {
            return Err(NyError::GpuBatchCapacityExceeded {
                requested: usize::try_from(self.row_bytes).unwrap_or(usize::MAX),
                capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
                unit: "bytes per worded sweep carrier row buffer",
                site: "WGPU intermediate DAG carrier preflight",
            });
        }
        Ok(self)
    }
}

/// Four f32 coefficient/error lanes and their four u32 taint twins.
pub(super) struct SweepMatrixBuffers {
    pub(super) lower_center: wgpu::Buffer,
    pub(super) upper_center: wgpu::Buffer,
    pub(super) lower_radius: wgpu::Buffer,
    pub(super) upper_radius: wgpu::Buffer,
    pub(super) lower_center_word: wgpu::Buffer,
    pub(super) upper_center_word: wgpu::Buffer,
    pub(super) lower_radius_word: wgpu::Buffer,
    pub(super) upper_radius_word: wgpu::Buffer,
}

impl SweepMatrixBuffers {
    fn as_array(&self) -> [&wgpu::Buffer; 8] {
        [
            &self.lower_center,
            &self.upper_center,
            &self.lower_radius,
            &self.upper_radius,
            &self.lower_center_word,
            &self.upper_center_word,
            &self.lower_radius_word,
            &self.upper_radius_word,
        ]
    }
}

/// Four f32 bias/error lanes and the u32 per-spec-row taint accumulator.
pub(super) struct SweepRowBuffers {
    pub(super) lower_bias: wgpu::Buffer,
    pub(super) upper_bias: wgpu::Buffer,
    pub(super) lower_bias_radius: wgpu::Buffer,
    pub(super) upper_bias_radius: wgpu::Buffer,
    pub(super) taint_rows: wgpu::Buffer,
}

impl SweepRowBuffers {
    fn as_array(&self) -> [&wgpu::Buffer; 5] {
        [
            &self.lower_bias,
            &self.upper_bias,
            &self.lower_bias_radius,
            &self.upper_bias_radius,
            &self.taint_rows,
        ]
    }

    fn bias_array(&self) -> [&wgpu::Buffer; 4] {
        [
            &self.lower_bias,
            &self.upper_bias,
            &self.lower_bias_radius,
            &self.upper_bias_radius,
        ]
    }
}

/// One independently owned device allocation set for a live DAG frontier.
///
/// No `Clone` implementation is provided. Moving this value moves ownership;
/// a branch fork must allocate a distinct carrier and encode explicit copies.
pub(super) struct DeviceSweepCarrier {
    pub(super) layout: SweepCarrierLayout,
    pub(super) matrix: SweepMatrixBuffers,
    pub(super) row: SweepRowBuffers,
}

impl DeviceSweepCarrier {
    /// Allocate all thirteen exact-sized buffers. WebGPU initializes newly
    /// created buffers to zero before shader-visible use; callers reusing a
    /// carrier must call [`Self::encode_clear_all`] instead.
    pub(super) fn allocate_zero_initialized(
        device: &wgpu::Device,
        layout: SweepCarrierLayout,
        label: &str,
    ) -> Result<Self> {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let allocate = |suffix: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}_{suffix}")),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let matrix = SweepMatrixBuffers {
            lower_center: allocate("lower_center", layout.matrix_bytes),
            upper_center: allocate("upper_center", layout.matrix_bytes),
            lower_radius: allocate("lower_radius", layout.matrix_bytes),
            upper_radius: allocate("upper_radius", layout.matrix_bytes),
            lower_center_word: allocate("lower_center_word", layout.matrix_bytes),
            upper_center_word: allocate("upper_center_word", layout.matrix_bytes),
            lower_radius_word: allocate("lower_radius_word", layout.matrix_bytes),
            upper_radius_word: allocate("upper_radius_word", layout.matrix_bytes),
        };
        let row = SweepRowBuffers {
            lower_bias: allocate("lower_bias", layout.row_bytes),
            upper_bias: allocate("upper_bias", layout.row_bytes),
            lower_bias_radius: allocate("lower_bias_radius", layout.row_bytes),
            upper_bias_radius: allocate("upper_bias_radius", layout.row_bytes),
            taint_rows: allocate("taint_rows", layout.row_bytes),
        };
        let carrier = Self {
            layout,
            matrix,
            row,
        };
        carrier.validate_owned_sizes()?;
        Ok(carrier)
    }

    /// Sum actual WGPU buffer sizes exactly once. Since the carrier cannot be
    /// cloned, this is also the allocation ownership count used by a receipt.
    pub(super) fn actual_owned_bytes(&self) -> Result<usize> {
        self.all_buffers()
            .into_iter()
            .try_fold(0usize, |total, buffer| {
                let bytes = usize::try_from(buffer.size())
                    .map_err(|_| invalid("sweep carrier buffer size does not fit usize"))?;
                total
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("sweep carrier actual byte count overflow"))
            })
    }

    pub(super) fn validate_owned_sizes(&self) -> Result<()> {
        for (index, buffer) in self.matrix.as_array().into_iter().enumerate() {
            if buffer.size() != self.layout.matrix_bytes {
                return Err(invalid(format!(
                    "sweep carrier matrix buffer {index} size {} != {}",
                    buffer.size(),
                    self.layout.matrix_bytes
                )));
            }
        }
        for (index, buffer) in self.row.as_array().into_iter().enumerate() {
            if buffer.size() != self.layout.row_bytes {
                return Err(invalid(format!(
                    "sweep carrier row buffer {index} size {} != {}",
                    buffer.size(),
                    self.layout.row_bytes
                )));
            }
        }
        let actual = self.actual_owned_bytes()?;
        if actual != self.layout.logical_bytes {
            return Err(invalid(format!(
                "sweep carrier actual bytes {actual} != logical bytes {}",
                self.layout.logical_bytes
            )));
        }
        Ok(())
    }

    /// Encode the Add/Sub rhs fork invariant: copy coefficient centres, radii,
    /// every coefficient/error word, and the sticky row accumulator; clear all
    /// four bias lanes so the incoming affine bias follows lhs exactly once.
    pub(super) fn encode_fork_biasless_rhs(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rhs: &Self,
    ) -> Result<()> {
        self.require_same_layout(rhs)?;
        for (source, target) in self
            .matrix
            .as_array()
            .into_iter()
            .zip(rhs.matrix.as_array())
        {
            encoder.copy_buffer_to_buffer(source, 0, target, 0, self.layout.matrix_bytes);
        }
        encoder.copy_buffer_to_buffer(
            &self.row.taint_rows,
            0,
            &rhs.row.taint_rows,
            0,
            self.layout.row_bytes,
        );
        rhs.encode_clear_bias(encoder);
        Ok(())
    }

    pub(super) fn encode_clear_bias(&self, encoder: &mut wgpu::CommandEncoder) {
        for buffer in self.row.bias_array() {
            encoder.clear_buffer(buffer, 0, None);
        }
    }

    fn all_buffers(&self) -> [&wgpu::Buffer; 13] {
        let matrix = self.matrix.as_array();
        let row = self.row.as_array();
        [
            matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5], matrix[6], matrix[7],
            row[0], row[1], row[2], row[3], row[4],
        ]
    }

    fn require_same_layout(&self, other: &Self) -> Result<()> {
        if self.layout != other.layout {
            return Err(NyError::shape_mismatch(
                vec![self.layout.rows, self.layout.dim],
                vec![other.layout.rows, other.layout.dim],
            ));
        }
        self.validate_owned_sizes()?;
        other.validate_owned_sizes()
    }
}

fn invalid(message: impl Into<String>) -> NyError {
    NyError::InvalidSpec(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_counts_every_owned_lane_once() {
        let layout = SweepCarrierLayout::new(288, 8192).unwrap();
        assert_eq!(layout.matrix_elements, 2_359_296);
        assert_eq!(layout.matrix_bytes, 9_437_184);
        assert_eq!(layout.row_bytes, 1_152);
        assert_eq!(layout.logical_bytes, 75_503_232);
        assert_eq!(
            layout.logical_bytes,
            usize::try_from(layout.matrix_bytes * 8 + layout.row_bytes * 5).unwrap()
        );
    }

    #[test]
    fn layout_capacity_is_checked_per_real_buffer() {
        let layout = SweepCarrierLayout::new(32, 2048).unwrap();
        assert!(layout.validate_device_limits(u64::MAX, u64::MAX).is_ok());
        assert!(matches!(
            layout.validate_device_limits(layout.matrix_bytes - 1, u64::MAX),
            Err(NyError::GpuBatchCapacityExceeded { .. })
        ));
        assert!(matches!(
            layout.validate_device_limits(u64::MAX, layout.matrix_bytes - 1),
            Err(NyError::GpuBatchCapacityExceeded { .. })
        ));
    }

    #[test]
    fn charged_only_authority_declines() {
        assert_eq!(
            SweepDagAuthority::select(true, false),
            Some(SweepDagAuthority::FullIeee)
        );
        assert_eq!(
            SweepDagAuthority::select(true, true),
            Some(SweepDagAuthority::FullIeee)
        );
        assert_eq!(SweepDagAuthority::select(false, true), None);
        assert_eq!(SweepDagAuthority::select(false, false), None);
    }
}
