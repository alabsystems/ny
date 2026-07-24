// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core arena-based constraint store for BaB domains.
//!
//! Uses arena allocation following ay's ClauseDB pattern:
//! - All coefficients in contiguous `Vec<f32>`
//! - All indices in contiguous `Vec<u32>`
//! - Compact headers with offsets into arenas
//! - Scope markers for O(1) backtracking
//!
//! # Sources
//!
//! - ay ClauseDB: `crates/ay-sat/src/clause_db.rs`
//! - Design doc: `designs/2026-01-29-linear-constraint-store.md`
//! - Issue: #234

use super::types::{ConstraintHeader, ConstraintOrigin, ConstraintSense, LinearConstraintRef};
use ny_core::{NyError, Result};
use std::mem::size_of;
use tracing::warn;

/// Arena-based constraint store for BaB domains.
///
/// Uses arena allocation following ay's ClauseDB pattern:
/// - All coefficients in contiguous `Vec<f32>`
/// - All indices in contiguous `Vec<u32>`
/// - Compact headers with offsets into arenas
/// - Scope markers for O(1) backtracking
///
/// # Example
///
/// ```
/// use ny_propagate::beta_crown::constraint_store::{
///     ArenaConstraintStore, ConstraintSense, ConstraintOrigin,
/// };
///
/// let mut store = ArenaConstraintStore::new();
///
/// // Add a constraint: x[0] - x[1] <= 0.5
/// store.add_constraint(
///     &[0, 1],
///     &[1.0, -1.0],
///     0.5,
///     ConstraintSense::Le,
///     ConstraintOrigin::Split,
/// );
///
/// // Save state for backtracking
/// store.push_scope();
///
/// // Add another constraint
/// store.add_constraint(&[2], &[1.0], 0.0, ConstraintSense::Le, ConstraintOrigin::Split);
///
/// assert_eq!(store.len(), 2);
///
/// // Backtrack
/// store.pop_scope();
/// assert_eq!(store.len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ArenaConstraintStore {
    /// Constraint headers.
    headers: Vec<ConstraintHeader>,
    /// Arena for coefficients.
    coeffs: Vec<f32>,
    /// Arena for variable indices.
    indices: Vec<u32>,
    /// Scope markers for push/pop: (headers_len, coeffs_len, indices_len).
    scope_markers: Vec<(usize, usize, usize)>,
}

impl ArenaConstraintStore {
    /// Create an empty constraint store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a constraint store with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `constraints` - Expected number of constraints
    /// * `avg_terms` - Average terms per constraint
    pub fn with_capacity(constraints: usize, avg_terms: usize) -> Self {
        let total_terms = constraints * avg_terms;
        Self {
            headers: Vec::with_capacity(constraints),
            coeffs: Vec::with_capacity(total_terms),
            indices: Vec::with_capacity(total_terms),
            scope_markers: Vec::with_capacity(16),
        }
    }

    /// Number of constraints in the store.
    pub fn len(&self) -> usize {
        self.headers.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    /// Total number of terms (for memory statistics).
    pub fn total_terms(&self) -> usize {
        self.coeffs.len()
    }

    /// Add a constraint to the store.
    ///
    /// # Arguments
    /// * `indices` - Variable indices (must match length of coeffs)
    /// * `coeffs` - Coefficients for each variable
    /// * `bias` - Right-hand side value
    /// * `sense` - Constraint sense (Le or Ge)
    /// * `origin` - Where this constraint came from
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if:
    /// - `indices.len() != coeffs.len()`
    /// - `indices.len() > u16::MAX` (constraint too large)
    ///
    /// Returns `NyError::NumericalInstability` if:
    /// - `bias` is NaN or infinite (#2259)
    /// - Any coefficient is NaN or infinite (#2259)
    pub fn add_constraint(
        &mut self,
        indices: &[u32],
        coeffs: &[f32],
        bias: f32,
        sense: ConstraintSense,
        origin: ConstraintOrigin,
    ) -> Result<()> {
        if indices.len() != coeffs.len() {
            return Err(NyError::InvalidSpec(format!(
                "indices and coeffs must have same length ({} vs {})",
                indices.len(),
                coeffs.len()
            )));
        }
        if indices.len() > u16::MAX as usize {
            return Err(NyError::InvalidSpec(format!(
                "constraint too large ({} terms, max {})",
                indices.len(),
                u16::MAX
            )));
        }
        // NaN/Inf validation (#2259): NaN bias makes the constraint meaningless
        // (IEEE 754: NaN comparisons are always false), and Inf coefficients make
        // constraints trivially satisfied or violated depending on sign.
        // Must follow length equality check so indices[pos] is safe.
        if !bias.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "constraint bias is {} — constraint would be meaningless under \
                 IEEE 754 semantics (#2259)",
                bias
            )));
        }
        if let Some(pos) = coeffs.iter().position(|c| !c.is_finite()) {
            return Err(NyError::NumericalInstability(format!(
                "constraint coefficient[{}] is {} (variable index {}) — constraint \
                 would be trivially satisfied or violated (#2259)",
                pos, coeffs[pos], indices[pos]
            )));
        }

        let data_start = u32::try_from(self.coeffs.len()).map_err(|_| {
            NyError::InvalidSpec(format!(
                "arena overflow: {} coefficients exceeds u32::MAX — \
                 all subsequent headers would point to wrong arena locations (#2261)",
                self.coeffs.len()
            ))
        })?;
        let data_len = indices.len() as u16;

        self.coeffs.extend_from_slice(coeffs);
        self.indices.extend_from_slice(indices);

        let header = ConstraintHeader::new(data_start, data_len, bias, sense, origin);
        self.headers.push(header);
        Ok(())
    }

    /// Push a scope marker for backtracking.
    ///
    /// On BaB split: save current state before adding child constraints.
    pub fn push_scope(&mut self) {
        self.scope_markers
            .push((self.headers.len(), self.coeffs.len(), self.indices.len()));
    }

    /// Pop to the last scope marker.
    ///
    /// On BaB backtrack: restore state to before the last `push_scope()`.
    /// Returns false if no scope markers exist.
    pub fn pop_scope(&mut self) -> bool {
        if let Some((h_len, c_len, i_len)) = self.scope_markers.pop() {
            self.headers.truncate(h_len);
            self.coeffs.truncate(c_len);
            self.indices.truncate(i_len);
            true
        } else {
            false
        }
    }

    /// Current scope depth (number of push_scope calls without pop).
    pub fn scope_depth(&self) -> usize {
        self.scope_markers.len()
    }

    /// Get a constraint by index.
    ///
    /// Returns `None` if the index is out of bounds or the header's data range
    /// exceeds the arena (possible GPU buffer corruption — see #2261).
    /// Logs a warning when a constraint is dropped due to corruption (#2981).
    pub fn get(&self, idx: usize) -> Option<LinearConstraintRef<'_>> {
        let header = self.headers.get(idx)?;
        let range = header.data_range();

        // Bounds-check: corrupted data_start/data_len could index beyond arena.
        if range.end > self.coeffs.len() || range.end > self.indices.len() {
            warn!(
                "Constraint at index {idx}: data range {range:?} exceeds arena \
                 (coeffs={}, indices={}) — constraint dropped. \
                 See #2261 for GPU buffer corruption investigation.",
                self.coeffs.len(),
                self.indices.len()
            );
            return None;
        }

        // Decode sense/origin with warning on corruption (#2981 Slice 2).
        // Before this fix, `.ok()?` silently dropped constraints with corrupted
        // headers, weakening bounds without any diagnostic trace.
        let sense = match header.sense() {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    "Constraint header decode failure at index {idx}: {e} — \
                     constraint dropped. See #2261 for GPU buffer corruption investigation."
                );
                return None;
            }
        };
        let origin = match header.origin() {
            Ok(o) => o,
            Err(e) => {
                warn!(
                    "Constraint header decode failure at index {idx}: {e} — \
                     constraint dropped. See #2261 for GPU buffer corruption investigation."
                );
                return None;
            }
        };

        Some(LinearConstraintRef {
            indices: &self.indices[range.clone()],
            coeffs: &self.coeffs[range],
            bias: header.bias,
            sense,
            origin,
        })
    }

    /// Iterate over all constraints.
    ///
    /// Skips constraints whose data range exceeds the arena or whose header
    /// bytes are corrupted (possible GPU buffer corruption — see #2261).
    /// Logs a warning for each skipped constraint (#2981).
    pub fn iter(&self) -> impl Iterator<Item = LinearConstraintRef<'_>> {
        let coeffs_len = self.coeffs.len();
        let indices_len = self.indices.len();
        self.headers
            .iter()
            .enumerate()
            .filter_map(move |(idx, header)| {
                let range = header.data_range();

                // Bounds-check: corrupted data_start/data_len could index beyond arena.
                if range.end > coeffs_len || range.end > indices_len {
                    warn!(
                        "Constraint at index {idx}: data range {range:?} exceeds arena \
                         (coeffs={coeffs_len}, indices={indices_len}) — constraint dropped. \
                         See #2261 for GPU buffer corruption investigation."
                    );
                    return None;
                }

                // Decode sense/origin with warning on corruption (#2981 Slice 2).
                let sense = match header.sense() {
                    Ok(s) => s,
                    Err(e) => {
                        warn!(
                            "Constraint header decode failure at index {idx}: {e} — \
                             constraint dropped. See #2261."
                        );
                        return None;
                    }
                };
                let origin = match header.origin() {
                    Ok(o) => o,
                    Err(e) => {
                        warn!(
                            "Constraint header decode failure at index {idx}: {e} — \
                             constraint dropped. See #2261."
                        );
                        return None;
                    }
                };

                Some(LinearConstraintRef {
                    indices: &self.indices[range.clone()],
                    coeffs: &self.coeffs[range],
                    bias: header.bias,
                    sense,
                    origin,
                })
            })
    }

    /// Get raw headers for GPU serialization.
    pub fn headers(&self) -> &[ConstraintHeader] {
        &self.headers
    }

    /// Get raw coefficients arena for GPU serialization.
    pub fn coeffs(&self) -> &[f32] {
        &self.coeffs
    }

    /// Get raw indices arena for GPU serialization.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.headers.len() * size_of::<ConstraintHeader>()
            + self.coeffs.len() * size_of::<f32>()
            + self.indices.len() * size_of::<u32>()
            + self.scope_markers.len() * size_of::<(usize, usize, usize)>()
    }

    /// Corrupt a header's data range for testing bounds-checking guards.
    ///
    /// This replaces the `unsafe` raw-pointer mutation that was previously used
    /// in tests, eliminating UB from casting `*const` to `*mut` (#2754).
    #[cfg(test)]
    pub(crate) fn corrupt_header_data_range(&mut self, idx: usize, data_start: u32, data_len: u16) {
        self.headers[idx].data_start = data_start;
        self.headers[idx].data_len = data_len;
    }

    /// Corrupt a header's sense byte for testing decode-failure logging (#2981).
    #[cfg(test)]
    pub(crate) fn corrupt_header_sense(&mut self, idx: usize, sense: u8) {
        self.headers[idx].sense = sense;
    }

    /// Corrupt a header's origin byte for testing decode-failure logging (#2981).
    #[cfg(test)]
    pub(crate) fn corrupt_header_origin(&mut self, idx: usize, origin: u8) {
        self.headers[idx].origin = origin;
    }
}
