// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed decoder verification compatibility APIs.

use ny_core::{NyError, Result};

use super::{DecoderModel, DecoderVerificationDetails};

pub(super) fn decoder_verification_unavailable() -> NyError {
    NyError::UnsupportedConfiguration(
        "decoder verification is unavailable: the current decoder analysis code has not proven \
         that its extracted attention topology, projection conventions, masks, normalization \
         attributes, and cross-attention composition are equivalent to the loaded ONNX graph; \
         no bounds or verification details were produced"
            .to_string(),
    )
}

impl DecoderModel {
    /// Unavailable compositional decoder-block verification surface.
    ///
    /// Decoder model loading and subgraph inspection remain available. This
    /// method fails closed until the extracted block is proven equivalent to
    /// the loaded graph and covered by independent concrete-point tests.
    pub fn verify_block_compositional(
        &self,
        _block_index: usize,
        _input: &ny_tensor::BoundedTensor,
    ) -> Result<(ny_tensor::BoundedTensor, DecoderVerificationDetails)> {
        Err(decoder_verification_unavailable())
    }

    /// Unavailable sequential decoder verification surface.
    ///
    /// Fails closed without returning bounds or per-block details.
    pub fn verify_sequential(
        &self,
        _input: &ny_tensor::BoundedTensor,
        _start_block: usize,
        _end_block: usize,
    ) -> Result<(ny_tensor::BoundedTensor, Vec<DecoderVerificationDetails>)> {
        Err(decoder_verification_unavailable())
    }
}
