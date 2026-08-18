// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed accelerated decoder verification compatibility APIs.

use crate::GpuCompositionalDetails;
use ny_core::Result;
use ny_gpu::ComputeDevice;

use super::{verify::decoder_verification_unavailable, DecoderModel};

impl DecoderModel {
    /// Unavailable accelerated decoder-block verification surface.
    ///
    /// The previous implementation reconstructed attention from inferred
    /// conventions rather than executing a proven-equivalent graph. This
    /// method therefore fails closed for every device and returns no bounds.
    pub fn verify_block_compositional_gpu(
        &self,
        _block_index: usize,
        _input: &ny_tensor::BoundedTensor,
        _gpu_device: Option<&ComputeDevice>,
    ) -> Result<(ny_tensor::BoundedTensor, GpuCompositionalDetails)> {
        Err(decoder_verification_unavailable())
    }

    /// Unavailable sequential accelerated decoder verification surface.
    ///
    /// Fails closed without returning bounds or per-block details.
    pub fn verify_sequential_gpu(
        &self,
        _input: &ny_tensor::BoundedTensor,
        _start_block: usize,
        _end_block: usize,
        _gpu_device: Option<&ComputeDevice>,
    ) -> Result<(ny_tensor::BoundedTensor, Vec<GpuCompositionalDetails>)> {
        Err(decoder_verification_unavailable())
    }
}
