// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::BatchedConstraintBuffer;
use ny_core::{NyError, Result};

fn validate_totals(buffer: &BatchedConstraintBuffer) -> Result<()> {
    if buffer.total_constraints != buffer.headers.len() {
        return Err(NyError::InvalidSpec(format!(
            "BatchedConstraintBuffer invalid GPU metadata: total_constraints {} does not match headers length {}",
            buffer.total_constraints,
            buffer.headers.len()
        )));
    }
    if buffer.total_terms != buffer.coeffs.len() {
        return Err(NyError::InvalidSpec(format!(
            "BatchedConstraintBuffer invalid GPU metadata: total_terms {} does not match coeffs length {}",
            buffer.total_terms,
            buffer.coeffs.len()
        )));
    }
    if buffer.total_terms != buffer.indices.len() {
        return Err(NyError::InvalidSpec(format!(
            "BatchedConstraintBuffer invalid GPU metadata: total_terms {} does not match indices length {}",
            buffer.total_terms,
            buffer.indices.len()
        )));
    }
    if buffer.coeffs.len() != buffer.indices.len() {
        return Err(NyError::InvalidSpec(format!(
            "BatchedConstraintBuffer invalid GPU metadata: coeffs length {} does not match indices length {}",
            buffer.coeffs.len(),
            buffer.indices.len()
        )));
    }
    Ok(())
}

fn validate_offsets(buffer: &BatchedConstraintBuffer) -> Result<()> {
    let expected_offset_len = buffer.batch_size.checked_add(1).ok_or_else(|| {
        NyError::InvalidSpec(
            "BatchedConstraintBuffer batch_size overflow while validating offsets".to_string(),
        )
    })?;
    if buffer.domain_header_offsets.len() != expected_offset_len {
        return Err(NyError::InvalidSpec(format!(
            "BatchedConstraintBuffer invalid GPU metadata: domain_header_offsets length {} does not match batch_size + 1 ({expected_offset_len})",
            buffer.domain_header_offsets.len()
        )));
    }
    if buffer.domain_header_offsets.first().copied() != Some(0) {
        return Err(NyError::InvalidSpec(format!(
            "BatchedConstraintBuffer invalid GPU metadata: first domain_header_offsets entry must be 0, got {:?}",
            buffer.domain_header_offsets.first()
        )));
    }
    for (domain_idx, offsets) in buffer.domain_header_offsets.windows(2).enumerate() {
        if offsets[0] > offsets[1] {
            return Err(NyError::InvalidSpec(format!(
                "BatchedConstraintBuffer invalid GPU metadata: domain_header_offsets not monotonic at domain {domain_idx} ({} > {})",
                offsets[0],
                offsets[1]
            )));
        }
    }
    if buffer.domain_header_offsets.last().copied() != Some(buffer.headers.len()) {
        return Err(NyError::InvalidSpec(format!(
            "BatchedConstraintBuffer invalid GPU metadata: last domain_header_offsets entry {:?} does not match headers length {}",
            buffer.domain_header_offsets.last(),
            buffer.headers.len()
        )));
    }
    Ok(())
}

fn validate_headers(buffer: &BatchedConstraintBuffer) -> Result<()> {
    for (header_idx, header) in buffer.headers.iter().enumerate() {
        if !(0..=2).contains(&header.origin) {
            return Err(NyError::InvalidSpec(format!(
                "BatchedConstraintBuffer invalid GPU metadata: header {header_idx} has invalid origin byte {}",
                header.origin
            )));
        }
        if !(0..=1).contains(&header.sense) {
            return Err(NyError::InvalidSpec(format!(
                "BatchedConstraintBuffer invalid GPU metadata: header {header_idx} has invalid sense byte {}",
                header.sense
            )));
        }

        let data_start = header.data_start as usize;
        let data_len = header.data_len as usize;
        let data_end = data_start.checked_add(data_len).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BatchedConstraintBuffer invalid GPU metadata: header {header_idx} term range overflow ({data_start} + {data_len})"
            ))
        })?;
        if data_end > buffer.coeffs.len() {
            return Err(NyError::InvalidSpec(format!(
                "BatchedConstraintBuffer invalid GPU metadata: header {header_idx} term range {data_start}..{data_end} exceeds coeffs length {}",
                buffer.coeffs.len()
            )));
        }
        if data_end > buffer.indices.len() {
            return Err(NyError::InvalidSpec(format!(
                "BatchedConstraintBuffer invalid GPU metadata: header {header_idx} term range {data_start}..{data_end} exceeds indices length {}",
                buffer.indices.len()
            )));
        }
    }
    Ok(())
}

impl BatchedConstraintBuffer {
    /// Validate that the packed metadata is internally consistent before GPU upload.
    ///
    /// This protects the WGSL kernels' domain/header indexing contract from
    /// malformed or manually-constructed buffers.
    pub fn validate_for_gpu(&self) -> Result<()> {
        validate_totals(self)?;
        validate_offsets(self)?;
        validate_headers(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::BatchedConstraintBuffer;
    use crate::beta_crown::constraint_store::{
        ConstraintHeader, ConstraintOrigin, ConstraintSense, DomainConstraintStore,
    };
    use ny_core::NyError;

    fn make_valid_buffer() -> BatchedConstraintBuffer {
        let mut store = DomainConstraintStore::new();
        store
            .delta_mut()
            .add_constraint(
                &[0, 1],
                &[1.0, -1.0],
                0.5,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();
        let stores = vec![&store];
        BatchedConstraintBuffer::from_domain_stores(&stores).unwrap()
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_validate_for_gpu_accepts_valid_buffer() {
        let buffer = make_valid_buffer();
        buffer.validate_for_gpu().unwrap();
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_validate_for_gpu_rejects_invalid_offsets() {
        let mut buffer = make_valid_buffer();
        buffer.domain_header_offsets = vec![0, 2];

        let err = buffer.validate_for_gpu().unwrap_err();
        assert!(matches!(
            err,
            NyError::InvalidSpec(ref msg)
                if msg.contains("last domain_header_offsets entry")
        ));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_validate_for_gpu_rejects_invalid_header_range() {
        let mut buffer = make_valid_buffer();
        buffer.headers[0] = ConstraintHeader::new(
            1,
            2,
            buffer.headers[0].bias,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        );

        let err = buffer.validate_for_gpu().unwrap_err();
        assert!(matches!(
            err,
            NyError::InvalidSpec(ref msg)
                if msg.contains("header 0 term range 1..3 exceeds coeffs length 2")
        ));
    }
}
