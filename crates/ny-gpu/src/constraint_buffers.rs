// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU buffers for batched constraint processing in Clip-and-Verify.
//!
//! This module provides GPU buffer management for linear constraints
//! collected during Branch-and-Bound, enabling GPU-parallel constraint
//! processing across multiple BaB domains.
//!
//! # Design
//!
//! The `GpuConstraintBuffers` struct holds wgpu buffers containing:
//! - Concatenated constraint headers from all domains
//! - Concatenated coefficients and indices
//! - Domain offset array for O(1) per-domain constraint lookup
//!
//! # Sources
//!
//! - Design doc: `designs/2026-01-29-gpu-constraint-buffer-layout.md`
//! - Issue: #226, #256

use ny_propagate::BatchedConstraintBuffer;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, Device};

/// Convert a `usize` to `u32` for GPU shader interfaces, returning an error on overflow.
fn checked_u32(value: usize, field: &str) -> ny_core::Result<u32> {
    u32::try_from(value).map_err(|_| {
        ny_core::NyError::InvalidSpec(format!(
            "GpuConstraintBuffers {field} value {value} exceeds u32::MAX"
        ))
    })
}

/// Validate the CPU representation before any adapter-dependent allocation.
/// Keeping this boundary explicit lets malformed-input coverage remain
/// hermetic on hosts without WGPU while `from_cpu_buffer` uses the exact same
/// admission check.
fn validate_cpu_buffer(cpu_buffer: &BatchedConstraintBuffer) -> ny_core::Result<()> {
    cpu_buffer.validate_for_gpu()
}

fn logical_memory_bytes(batch_size: u32, total_constraints: u32, total_terms: u32) -> usize {
    (total_constraints as usize) * 16
        + (total_terms as usize) * 4
        + (total_terms as usize) * 4
        + ((batch_size + 1) as usize) * 4
}

/// GPU buffers for batched constraint processing.
///
/// Holds wgpu storage buffers containing packed constraint data
/// from multiple BaB domains for GPU-parallel Clip-and-Verify passes.
///
/// # Buffer Layout
///
/// ```text
/// headers:       [H0, H1, ..., Hn] where each H is 16 bytes
/// coeffs:        [c0, c1, ..., cm] where each c is f32
/// indices:       [i0, i1, ..., im] where each i is u32
/// domain_offsets: [o0, o1, ..., ok+1] where k = batch_size
/// ```
///
/// Domain `d` has constraints at indices `domain_offsets[d]..domain_offsets[d+1]`.
pub struct GpuConstraintBuffers {
    /// Header buffer: `[total_constraints]` ConstraintHeader (16 bytes each).
    ///
    /// Use binding `@group(0) @binding(0)` in WGSL shaders.
    pub headers: Buffer,

    /// Coefficient buffer: `[total_terms]` f32.
    ///
    /// Use binding `@group(0) @binding(1)` in WGSL shaders.
    pub coeffs: Buffer,

    /// Index buffer: `[total_terms]` u32.
    ///
    /// Use binding `@group(0) @binding(2)` in WGSL shaders.
    pub indices: Buffer,

    /// Domain offset buffer: `[batch_size + 1]` u32.
    ///
    /// Use binding `@group(0) @binding(3)` in WGSL shaders.
    /// `domain_offsets[i]..domain_offsets[i+1]` gives header range for domain `i`.
    pub domain_offsets: Buffer,

    /// Number of domains in this batch.
    pub batch_size: u32,

    /// Total number of constraints across all domains.
    pub total_constraints: u32,

    /// Total number of terms (coefficient/index pairs) across all domains.
    pub total_terms: u32,
}

impl GpuConstraintBuffers {
    /// Create GPU buffers from a CPU-side batched constraint buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device for buffer allocation
    /// * `cpu_buffer` - CPU-side batched constraint buffer to upload
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ny_gpu::{GpuConstraintBuffers, WgpuDevice};
    /// use ny_propagate::BatchedConstraintBuffer;
    ///
    /// let wgpu_device = WgpuDevice::new().unwrap();
    /// let cpu_buffer = BatchedConstraintBuffer::empty();
    /// let gpu_buffers = GpuConstraintBuffers::from_cpu_buffer(
    ///     wgpu_device.device(),
    ///     &cpu_buffer,
    /// );
    /// ```
    pub fn from_cpu_buffer(
        device: &Device,
        cpu_buffer: &BatchedConstraintBuffer,
    ) -> ny_core::Result<Self> {
        validate_cpu_buffer(cpu_buffer)?;
        // Handle empty case: wgpu requires non-zero buffer sizes
        // Create minimum 4-byte buffers for empty data
        let header_bytes: &[u8] = if cpu_buffer.headers.is_empty() {
            &[0u8; 16] // Minimum 16 bytes (one header placeholder)
        } else {
            bytemuck::cast_slice(&cpu_buffer.headers)
        };
        let headers = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("constraint_headers"),
            contents: header_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let coeff_bytes: &[u8] = if cpu_buffer.coeffs.is_empty() {
            &[0u8; 4] // Minimum 4 bytes (one f32)
        } else {
            bytemuck::cast_slice(&cpu_buffer.coeffs)
        };
        let coeffs = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("constraint_coeffs"),
            contents: coeff_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let index_bytes: &[u8] = if cpu_buffer.indices.is_empty() {
            &[0u8; 4] // Minimum 4 bytes (one u32)
        } else {
            bytemuck::cast_slice(&cpu_buffer.indices)
        };
        let indices = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("constraint_indices"),
            contents: index_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Domain offsets: convert usize to u32 for GPU compatibility
        let offsets_u32: Vec<u32> = cpu_buffer
            .domain_header_offsets
            .iter()
            .map(|&x| checked_u32(x, "domain_header_offsets"))
            .collect::<ny_core::Result<Vec<u32>>>()?;
        let offset_bytes: &[u8] = if offsets_u32.is_empty() {
            // Minimum: [0] for empty buffer
            &[0u8; 4]
        } else {
            bytemuck::cast_slice(&offsets_u32)
        };
        let domain_offsets = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("constraint_domain_offsets"),
            contents: offset_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        Ok(Self {
            headers,
            coeffs,
            indices,
            domain_offsets,
            batch_size: checked_u32(cpu_buffer.batch_size, "batch_size")?,
            total_constraints: checked_u32(cpu_buffer.total_constraints, "total_constraints")?,
            total_terms: checked_u32(cpu_buffer.total_terms, "total_terms")?,
        })
    }

    /// Check if the buffers contain any constraints.
    pub fn is_empty(&self) -> bool {
        self.total_constraints == 0
    }

    /// Get the total GPU memory usage in bytes (approximate).
    ///
    /// Note: This is the logical size, not the actual allocated size
    /// which may be larger due to alignment requirements.
    pub fn memory_bytes(&self) -> usize {
        // 16 bytes per header + 4 bytes per coeff + 4 bytes per index + 4 bytes per offset
        logical_memory_bytes(self.batch_size, self.total_constraints, self.total_terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::beta_crown::constraint_store::ConstraintHeader;
    #[cfg(feature = "gpu-tests")]
    use ny_propagate::beta_crown::constraint_store::{
        ConstraintOrigin, ConstraintSense, DomainConstraintStore,
    };
    use std::mem::size_of;

    #[test]
    fn test_constraint_header_size() {
        assert_eq!(
            size_of::<ConstraintHeader>(),
            16,
            "ConstraintHeader must be 16 bytes for GPU buffer layout"
        );
    }

    #[test]
    fn test_checked_u32_accepts_u32_max() {
        assert_eq!(
            checked_u32(u32::MAX as usize, "test_field").unwrap(),
            u32::MAX
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_checked_u32_returns_error_on_overflow() {
        let overflow = (u32::MAX as usize).checked_add(1).unwrap();
        let err = checked_u32(overflow, "test_field").unwrap_err();
        assert!(err.to_string().contains("exceeds u32::MAX"));
    }

    #[cfg(feature = "gpu-tests")]
    fn create_test_device() -> Device {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to find adapter");

            let (device, _queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("Failed to create device");

            device
        })
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_empty_buffer() {
        let device = create_test_device();
        let cpu_buffer = BatchedConstraintBuffer::empty();
        let gpu_buffers = GpuConstraintBuffers::from_cpu_buffer(&device, &cpu_buffer).unwrap();

        assert!(gpu_buffers.is_empty());
        assert_eq!(gpu_buffers.batch_size, 0);
        assert_eq!(gpu_buffers.total_constraints, 0);
        assert_eq!(gpu_buffers.total_terms, 0);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_single_domain() {
        let device = create_test_device();

        let mut store = DomainConstraintStore::new();
        store
            .delta_mut()
            .add_constraint(
                &[0, 1, 2],
                &[1.0, -1.0, 0.5],
                0.5,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();

        let stores = vec![&store];
        let cpu_buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();
        let gpu_buffers = GpuConstraintBuffers::from_cpu_buffer(&device, &cpu_buffer).unwrap();

        assert!(!gpu_buffers.is_empty());
        assert_eq!(gpu_buffers.batch_size, 1);
        assert_eq!(gpu_buffers.total_constraints, 1);
        assert_eq!(gpu_buffers.total_terms, 3);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_multiple_domains() {
        let device = create_test_device();

        // Domain 0: 1 constraint
        let mut store0 = DomainConstraintStore::new();
        store0
            .delta_mut()
            .add_constraint(
                &[0, 1],
                &[1.0, -1.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();

        // Domain 1: 2 constraints
        let mut store1 = DomainConstraintStore::new();
        store1
            .delta_mut()
            .add_constraint(
                &[0],
                &[2.0],
                1.0,
                ConstraintSense::Ge,
                ConstraintOrigin::Output,
            )
            .unwrap();
        store1
            .delta_mut()
            .add_constraint(
                &[1, 2],
                &[0.5, 0.5],
                0.5,
                ConstraintSense::Le,
                ConstraintOrigin::BoundProp,
            )
            .unwrap();

        let stores = vec![&store0, &store1];
        let cpu_buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();
        let gpu_buffers = GpuConstraintBuffers::from_cpu_buffer(&device, &cpu_buffer).unwrap();

        assert_eq!(gpu_buffers.batch_size, 2);
        assert_eq!(gpu_buffers.total_constraints, 3);
        assert_eq!(gpu_buffers.total_terms, 5); // 2 + 1 + 2
    }

    #[test]
    fn test_memory_bytes() {
        // 1 header * 16 + 2 coeffs * 4 + 2 indices * 4 + 2 offsets * 4
        // = 16 + 8 + 8 + 8 = 40 bytes
        assert_eq!(logical_memory_bytes(1, 1, 2), 40);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_bind_group_creation() {
        // Smoke test: verify GPU buffers can be bound to the expected layout
        use crate::constraint_shaders::CONSTRAINT_BUFFER_LAYOUT_ENTRIES;

        let device = create_test_device();

        let mut store = DomainConstraintStore::new();
        store
            .delta_mut()
            .add_constraint(
                &[0, 1],
                &[1.0, -1.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();

        let stores = vec![&store];
        let cpu_buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();
        let gpu_buffers = GpuConstraintBuffers::from_cpu_buffer(&device, &cpu_buffer).unwrap();

        // Create bind group layout matching WGSL shader expectations
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("constraint_buffer_test_layout"),
            entries: &CONSTRAINT_BUFFER_LAYOUT_ENTRIES,
        });

        // Create bind group with the GPU buffers - validates compatibility
        let _bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constraint_buffer_test_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffers.headers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_buffers.coeffs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_buffers.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_buffers.domain_offsets.as_entire_binding(),
                },
            ],
        });

        // If we reach here without panic, buffers are compatible with shader layout
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_upload_readback_round_trip() {
        // Smoke test: upload data to GPU and read it back to verify integrity.
        // This validates that:
        // 1. Buffer creation uploads correct data
        // 2. Data layout matches between CPU and GPU
        // 3. Readback produces identical values

        let (device, queue) = create_test_device_and_queue();

        // Create constraint store with known values
        let mut store = DomainConstraintStore::new();
        store
            .delta_mut()
            .add_constraint(
                &[5, 10],
                &[1.5, -2.5],
                3.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();
        store
            .delta_mut()
            .add_constraint(
                &[0],
                &[7.0],
                -1.0,
                ConstraintSense::Ge,
                ConstraintOrigin::Output,
            )
            .unwrap();

        let stores = vec![&store];
        let cpu_buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();
        let gpu_buffers = GpuConstraintBuffers::from_cpu_buffer(&device, &cpu_buffer).unwrap();

        // Read back coefficients
        let coeff_bytes = (cpu_buffer.coeffs.len() * 4) as u64;
        let staging_coeffs = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("staging_coeffs"),
            contents: &vec![0u8; coeff_bytes as usize],
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&gpu_buffers.coeffs, 0, &staging_coeffs, 0, coeff_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read back
        let buffer_slice = staging_coeffs.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let readback_coeffs: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_coeffs.unmap();

        // Verify coefficients match
        assert_eq!(
            readback_coeffs.len(),
            cpu_buffer.coeffs.len(),
            "Coefficient count mismatch"
        );
        for (i, (expected, actual)) in cpu_buffer
            .coeffs
            .iter()
            .zip(readback_coeffs.iter())
            .enumerate()
        {
            assert!(
                (expected - actual).abs() < 1e-6,
                "Coefficient {} mismatch: expected {}, got {}",
                i,
                expected,
                actual
            );
        }

        // Read back indices
        let index_bytes = (cpu_buffer.indices.len() * 4) as u64;
        let staging_indices = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("staging_indices"),
            contents: &vec![0u8; index_bytes as usize],
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder_indices"),
        });
        encoder.copy_buffer_to_buffer(&gpu_buffers.indices, 0, &staging_indices, 0, index_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_indices.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let readback_indices: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_indices.unmap();

        // Verify indices match
        assert_eq!(
            readback_indices.len(),
            cpu_buffer.indices.len(),
            "Index count mismatch"
        );
        for (i, (expected, actual)) in cpu_buffer
            .indices
            .iter()
            .zip(readback_indices.iter())
            .enumerate()
        {
            assert_eq!(
                *expected, *actual,
                "Index {} mismatch: expected {}, got {}",
                i, expected, actual
            );
        }
    }
    #[cfg(feature = "gpu-tests")]
    fn create_test_device_and_queue() -> (Device, wgpu::Queue) {
        // Device and queue must be created together in wgpu
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to find adapter");
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("Failed to create device")
        })
    }
}
#[cfg(test)]
mod validation_tests;
