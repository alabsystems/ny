// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-ready batched constraint buffer for Clip-and-Verify.
//!
//! Packs constraints from multiple BaB domains into contiguous buffers
//! for efficient CPU→GPU transfer and parallel processing.
//!
//! # Design
//!
//! Uses concatenated arena layout (not zero-padded dense matrices) to minimize
//! memory waste when domain constraint counts vary significantly.
//!
//! Layout:
//! ```text
//! headers:  [d0_h0, d0_h1, ..., d1_h0, d1_h1, ..., dN_hM]
//! coeffs:   [d0_c*, d1_c*, ..., dN_c*] concatenated
//! indices:  [d0_i*, d1_i*, ..., dN_i*] concatenated
//! offsets:  [0, d0_count, d0+d1_count, ..., total_count]
//! ```
//!
//! where `offsets[i]` gives the starting header index for domain `i`.
//!
//! # Sources
//!
//! - Design doc: `designs/2026-01-29-gpu-constraint-buffer-layout.md`
//! - Issue: #226
use crate::beta_crown::constraint_store::{ConstraintHeader, DomainConstraintStore};
use ny_core::{NyError, Result};
use std::mem::size_of;
mod validation;
/// GPU-ready batched constraint buffer packing multiple domain stores.
///
/// # Example
///
/// ```rust,no_run
/// use ny_propagate::beta_crown::constraint_store::DomainConstraintStore;
/// use ny_propagate::BatchedConstraintBuffer;
///
/// let stores: Vec<DomainConstraintStore> = vec![]; // domains would provide these
/// let store_refs: Vec<&DomainConstraintStore> = stores.iter().collect();
/// let batched = BatchedConstraintBuffer::from_domain_stores(&store_refs).unwrap();
/// ```
///
/// # Semantics
///
/// - `is_empty()` returns true when batch_size is 0 (no domains)
/// - Use `has_constraints()` to check if any constraints exist
#[derive(Debug, Clone)]
pub struct BatchedConstraintBuffer {
    /// Number of domains in this batch.
    pub batch_size: usize,

    /// Concatenated constraint headers from all domains.
    /// Each header's `data_start` is relative to the concatenated arenas.
    pub headers: Vec<ConstraintHeader>,

    /// Concatenated coefficient arena from all domains.
    pub coeffs: Vec<f32>,

    /// Concatenated index arena from all domains.
    pub indices: Vec<u32>,

    /// Prefix-sum offsets mapping domain index → header range.
    /// `offsets[i]..offsets[i+1]` gives the header indices for domain `i`.
    /// Length: `batch_size + 1`.
    pub domain_header_offsets: Vec<usize>,

    /// Total constraint count across all domains.
    pub total_constraints: usize,

    /// Total terms (coefficient/index pairs) across all domains.
    pub total_terms: usize,
}

impl BatchedConstraintBuffer {
    /// Pack constraint stores from multiple domains into a single batched buffer.
    ///
    /// # Arguments
    /// * `stores` - Per-domain constraint stores to batch
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if the total data exceeds u32::MAX
    /// (GPU buffer format limitation).
    pub fn from_domain_stores(stores: &[&DomainConstraintStore]) -> Result<Self> {
        let batch_size = stores.len();

        if batch_size == 0 {
            return Ok(Self::empty());
        }

        // Pre-calculate capacities
        let total_constraints: usize = stores.iter().map(|s| s.len()).sum();
        let total_terms: usize = stores
            .iter()
            .map(|s| s.base().total_terms() + s.delta().total_terms())
            .sum();

        let mut headers = Vec::with_capacity(total_constraints);
        let mut coeffs = Vec::with_capacity(total_terms);
        let mut indices = Vec::with_capacity(total_terms);
        let mut domain_header_offsets = Vec::with_capacity(batch_size + 1);

        domain_header_offsets.push(0);
        // Use usize for internal tracking to avoid u32 overflow
        let mut term_offset: usize = 0;

        // Early check: if total_terms exceeds u32::MAX, fail fast
        if total_terms > u32::MAX as usize {
            return Err(NyError::InvalidSpec(format!(
                "Total terms {} exceeds u32::MAX, GPU buffer format cannot represent this",
                total_terms
            )));
        }

        for store in stores {
            // Process base constraints
            let base = store.base();
            for header in base.headers() {
                let mut adjusted = *header;
                // Combine original data_start with term_offset, with checked conversion
                let new_offset = adjusted.data_start as usize + term_offset;
                adjusted.data_start = u32::try_from(new_offset).map_err(|_| {
                    NyError::InvalidSpec(format!(
                        "Constraint data_start {} exceeds u32::MAX, GPU buffer format limitation",
                        new_offset
                    ))
                })?;
                headers.push(adjusted);
            }
            coeffs.extend_from_slice(base.coeffs());
            indices.extend_from_slice(base.indices());
            term_offset = term_offset.checked_add(base.total_terms()).ok_or_else(|| {
                NyError::InvalidSpec(
                    "Term offset overflow during base constraint processing".to_string(),
                )
            })?;

            // Process delta constraints
            let delta = store.delta();
            for header in delta.headers() {
                let mut adjusted = *header;
                let new_offset = adjusted.data_start as usize + term_offset;
                adjusted.data_start = u32::try_from(new_offset).map_err(|_| {
                    NyError::InvalidSpec(format!(
                        "Constraint data_start {} exceeds u32::MAX, GPU buffer format limitation",
                        new_offset
                    ))
                })?;
                headers.push(adjusted);
            }
            coeffs.extend_from_slice(delta.coeffs());
            indices.extend_from_slice(delta.indices());
            term_offset = term_offset
                .checked_add(delta.total_terms())
                .ok_or_else(|| {
                    NyError::InvalidSpec(
                        "Term offset overflow during delta constraint processing".to_string(),
                    )
                })?;

            domain_header_offsets.push(headers.len());
        }

        Ok(Self {
            batch_size,
            headers,
            coeffs,
            indices,
            domain_header_offsets,
            total_constraints,
            total_terms,
        })
    }

    /// Create an empty buffer.
    pub fn empty() -> Self {
        Self {
            batch_size: 0,
            headers: Vec::new(),
            coeffs: Vec::new(),
            indices: Vec::new(),
            domain_header_offsets: vec![0],
            total_constraints: 0,
            total_terms: 0,
        }
    }

    /// Check if the buffer has no domains (batch_size == 0).
    ///
    /// Note: A buffer with domains but no constraints is not considered empty.
    /// Use `has_constraints()` to check for actual constraint content.
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    /// Check if the buffer contains any constraints.
    pub fn has_constraints(&self) -> bool {
        self.total_constraints > 0
    }

    /// The constraint count for a specific domain.
    ///
    /// Returns `None` if `domain_idx >= batch_size`.
    pub fn domain_constraint_count(&self, domain_idx: usize) -> Option<usize> {
        if domain_idx >= self.batch_size {
            return None;
        }
        Some(self.domain_header_offsets[domain_idx + 1] - self.domain_header_offsets[domain_idx])
    }

    /// The header range for a specific domain.
    ///
    /// Returns `None` if `domain_idx >= batch_size`.
    pub fn domain_header_range(&self, domain_idx: usize) -> Option<std::ops::Range<usize>> {
        if domain_idx >= self.batch_size {
            return None;
        }
        Some(self.domain_header_offsets[domain_idx]..self.domain_header_offsets[domain_idx + 1])
    }

    /// Headers for a specific domain.
    ///
    /// Returns `None` if `domain_idx >= batch_size`.
    pub fn domain_headers(&self, domain_idx: usize) -> Option<&[ConstraintHeader]> {
        let range = self.domain_header_range(domain_idx)?;
        Some(&self.headers[range])
    }

    /// Memory size in bytes (CPU-side).
    pub fn memory_bytes(&self) -> usize {
        self.headers.len() * size_of::<ConstraintHeader>()
            + self.coeffs.len() * size_of::<f32>()
            + self.indices.len() * size_of::<u32>()
            + self.domain_header_offsets.len() * size_of::<usize>()
    }

    /// Average constraints per domain.
    pub fn avg_constraints_per_domain(&self) -> f32 {
        if self.batch_size == 0 {
            0.0
        } else {
            self.total_constraints as f32 / self.batch_size as f32
        }
    }

    /// Average terms per constraint.
    pub fn avg_terms_per_constraint(&self) -> f32 {
        if self.total_constraints == 0 {
            0.0
        } else {
            self.total_terms as f32 / self.total_constraints as f32
        }
    }
}

impl Default for BatchedConstraintBuffer {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::constraint_store::{
        ArenaConstraintStore, ConstraintOrigin, ConstraintSense,
    };

    #[ntest::timeout(10000)]
    #[test]
    fn test_empty_buffer() {
        let buffer = BatchedConstraintBuffer::empty();
        assert_eq!(buffer.batch_size, 0);
        assert!(buffer.is_empty());
        assert_eq!(buffer.total_constraints, 0);
        assert_eq!(buffer.total_terms, 0);
        assert_eq!(buffer.memory_bytes(), size_of::<usize>()); // just the offset vec[0]
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_from_empty_stores() {
        let stores: Vec<&DomainConstraintStore> = vec![];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();
        assert!(buffer.is_empty());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_single_domain_single_constraint() {
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
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();

        assert_eq!(buffer.batch_size, 1);
        assert_eq!(buffer.total_constraints, 1);
        assert_eq!(buffer.total_terms, 2);
        assert_eq!(buffer.domain_constraint_count(0), Some(1));
        assert_eq!(buffer.headers.len(), 1);
        assert_eq!(buffer.coeffs, vec![1.0, -1.0]);
        assert_eq!(buffer.indices, vec![0, 1]);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_multiple_domains() {
        // Domain 0: 1 constraint with 2 terms
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

        // Domain 1: 2 constraints with 1 and 3 terms
        let mut store1 = DomainConstraintStore::new();
        store1
            .delta_mut()
            .add_constraint(
                &[2],
                &[2.0],
                1.0,
                ConstraintSense::Ge,
                ConstraintOrigin::Output,
            )
            .unwrap();
        store1
            .delta_mut()
            .add_constraint(
                &[0, 1, 2],
                &[0.5, 0.5, 0.5],
                0.5,
                ConstraintSense::Le,
                ConstraintOrigin::BoundProp,
            )
            .unwrap();

        let stores = vec![&store0, &store1];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();

        assert_eq!(buffer.batch_size, 2);
        assert_eq!(buffer.total_constraints, 3);
        assert_eq!(buffer.total_terms, 6); // 2 + 1 + 3

        // Check domain offsets
        assert_eq!(buffer.domain_header_offsets, vec![0, 1, 3]);
        assert_eq!(buffer.domain_constraint_count(0), Some(1));
        assert_eq!(buffer.domain_constraint_count(1), Some(2));

        // Check concatenated coefficients
        assert_eq!(buffer.coeffs, vec![1.0, -1.0, 2.0, 0.5, 0.5, 0.5]);
        assert_eq!(buffer.indices, vec![0, 1, 2, 0, 1, 2]);

        // Check header data_start adjustments
        // Domain 0, constraint 0: starts at 0
        assert_eq!(buffer.headers[0].data_start, 0);
        // Domain 1, constraint 0: starts at 2 (after domain 0's 2 terms)
        assert_eq!(buffer.headers[1].data_start, 2);
        // Domain 1, constraint 1: starts at 3 (after domain 0's 2 + domain 1 first's 1)
        assert_eq!(buffer.headers[2].data_start, 3);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_base_and_delta_combined() {
        // Create a store with base (inherited) constraints
        let mut base = ArenaConstraintStore::new();
        base.add_constraint(
            &[0],
            &[1.0],
            0.0,
            ConstraintSense::Le,
            ConstraintOrigin::Output,
        )
        .unwrap();

        let mut store = DomainConstraintStore::with_base(base);
        store
            .delta_mut()
            .add_constraint(
                &[1],
                &[2.0],
                1.0,
                ConstraintSense::Ge,
                ConstraintOrigin::Split,
            )
            .unwrap();

        let stores = vec![&store];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();

        assert_eq!(buffer.batch_size, 1);
        assert_eq!(buffer.total_constraints, 2); // 1 base + 1 delta
        assert_eq!(buffer.total_terms, 2);
        assert_eq!(buffer.headers.len(), 2);

        // First header is from base (data_start=0)
        assert_eq!(buffer.headers[0].data_start, 0);
        // Second header is from delta (data_start=1, after base's 1 term)
        assert_eq!(buffer.headers[1].data_start, 1);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_domain_headers_accessor() {
        let mut store0 = DomainConstraintStore::new();
        store0
            .delta_mut()
            .add_constraint(
                &[0],
                &[1.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();

        let mut store1 = DomainConstraintStore::new();
        store1
            .delta_mut()
            .add_constraint(
                &[1],
                &[2.0],
                0.0,
                ConstraintSense::Ge,
                ConstraintOrigin::Split,
            )
            .unwrap();
        store1
            .delta_mut()
            .add_constraint(
                &[2],
                &[3.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();

        let stores = vec![&store0, &store1];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();

        assert_eq!(buffer.domain_headers(0).unwrap().len(), 1);
        assert_eq!(buffer.domain_headers(1).unwrap().len(), 2);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_statistics() {
        let mut store = DomainConstraintStore::new();
        store
            .delta_mut()
            .add_constraint(
                &[0, 1, 2],
                &[1.0, 2.0, 3.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();

        let stores = vec![&store];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();

        assert_eq!(buffer.avg_constraints_per_domain(), 1.0);
        assert_eq!(buffer.avg_terms_per_constraint(), 3.0);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_has_constraints() {
        // Empty buffer has no constraints
        let empty = BatchedConstraintBuffer::empty();
        assert!(!empty.has_constraints());
        assert!(empty.is_empty());

        // Buffer with constraint
        let mut store = DomainConstraintStore::new();
        store
            .delta_mut()
            .add_constraint(
                &[0],
                &[1.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();
        let stores = vec![&store];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();
        assert!(buffer.has_constraints());
        assert!(!buffer.is_empty());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_get_methods_bounds_checking() {
        let mut store = DomainConstraintStore::new();
        store
            .delta_mut()
            .add_constraint(
                &[0],
                &[1.0],
                0.0,
                ConstraintSense::Le,
                ConstraintOrigin::Split,
            )
            .unwrap();
        let stores = vec![&store];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();

        // Valid index
        assert!(buffer.domain_constraint_count(0).is_some());
        assert!(buffer.domain_header_range(0).is_some());
        assert!(buffer.domain_headers(0).is_some());

        // Invalid index
        assert!(buffer.domain_constraint_count(1).is_none());
        assert!(buffer.domain_header_range(1).is_none());
        assert!(buffer.domain_headers(1).is_none());

        // Empty buffer
        let empty = BatchedConstraintBuffer::empty();
        assert!(empty.domain_constraint_count(0).is_none());
    }

    /// Regression guard for #274: keep overflow handling on the Result path.
    ///
    /// We cannot create 4GB of actual data in a unit test, but we verify:
    /// 1. The function returns Result (API contract)
    /// 2. Normal usage returns Ok
    #[ntest::timeout(10000)]
    #[test]
    fn test_overflow_returns_result_not_panic() {
        // Verify normal case returns Ok
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

        let result = BatchedConstraintBuffer::from_domain_stores(&stores);
        assert!(result.is_ok(), "Normal case should return Ok");

        // Verify the error type is correct by checking the function signature
        // returns ny_core::Result<Self> (compile-time verification)
        let _: Result<BatchedConstraintBuffer> = result;
    }

    /// Integration test: verify offsets remain consistent with concatenation order.
    ///
    /// This test exercises the offset tracking paths; actual overflow testing
    /// requires mocking or dedicated integration tests with large allocations.
    /// See designs/2026-01-29-gpu-constraint-buffer-layout.md for overflow scenarios.
    #[ntest::timeout(10000)]
    #[test]
    fn test_overflow_protection_code_paths_exist() {
        // The actual overflow would require ~4GB of constraint data which is impractical for unit tests.

        // Create a constraint store and verify the offset is tracked correctly
        let mut store0 = DomainConstraintStore::new();
        for _ in 0..100 {
            store0
                .delta_mut()
                .add_constraint(
                    &[0, 1, 2, 3, 4], // 5 terms each
                    &[1.0; 5],
                    0.0,
                    ConstraintSense::Le,
                    ConstraintOrigin::Split,
                )
                .unwrap();
        }

        let stores = vec![&store0];
        let buffer = BatchedConstraintBuffer::from_domain_stores(&stores).unwrap();

        // Verify offsets are correctly tracked (500 terms total)
        assert_eq!(buffer.total_terms, 500);
        assert_eq!(buffer.total_constraints, 100);

        // Last header should have data_start near end of buffer
        let last_header = buffer.headers.last().unwrap();
        assert_eq!(last_header.data_start, 495); // 99 * 5 = 495
    }
}
