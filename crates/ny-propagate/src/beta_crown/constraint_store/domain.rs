// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain-specific constraint store with base + delta pattern.
//!
//! Used for BaB domains where child domains inherit parent constraints
//! and add their own split constraints.

use super::arena::ArenaConstraintStore;
use super::types::LinearConstraintRef;
use ny_core::Result;

/// Domain-specific constraint store with base + delta pattern.
///
/// Used for BaB domains where child domains inherit parent constraints
/// and add their own split constraints.
///
/// # Pattern
///
/// - `base`: Constraints inherited from parent domain (immutable, shared)
/// - `delta`: Constraints added by this domain (owned)
///
/// On domain split:
/// 1. Child copies parent's delta into its base
/// 2. Child adds new split constraint to its delta
/// 3. Parent's base remains unchanged
#[derive(Debug, Clone, Default)]
pub struct DomainConstraintStore {
    /// Constraints inherited from parent domain.
    base: ArenaConstraintStore,
    /// Constraints added by this domain.
    delta: ArenaConstraintStore,
}

impl DomainConstraintStore {
    /// Create an empty domain constraint store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a base constraint store.
    pub fn with_base(base: ArenaConstraintStore) -> Self {
        Self {
            base,
            delta: ArenaConstraintStore::new(),
        }
    }

    /// Total number of constraints (base + delta).
    pub fn len(&self) -> usize {
        self.base.len() + self.delta.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.base.is_empty() && self.delta.is_empty()
    }

    /// Number of constraints in base.
    pub fn base_len(&self) -> usize {
        self.base.len()
    }

    /// Number of constraints in delta.
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Get the delta (mutable) for adding constraints.
    pub fn delta_mut(&mut self) -> &mut ArenaConstraintStore {
        &mut self.delta
    }

    /// Get the delta (immutable).
    pub fn delta(&self) -> &ArenaConstraintStore {
        &self.delta
    }

    /// Get the base (immutable).
    pub fn base(&self) -> &ArenaConstraintStore {
        &self.base
    }

    /// Iterate over all constraints (base then delta).
    pub fn iter(&self) -> impl Iterator<Item = LinearConstraintRef<'_>> {
        self.base.iter().chain(self.delta.iter())
    }

    /// Create a child store that inherits this store's constraints as base.
    ///
    /// The child has:
    /// - base: this store's base + delta merged
    /// - delta: empty (ready for new split constraints)
    pub fn create_child(&self) -> Result<Self> {
        let mut child_base = self.base.clone();

        // Merge delta into child's base
        // SAFETY: Constraints were already validated when added to delta
        for constraint in self.delta.iter() {
            child_base.add_constraint(
                constraint.indices,
                constraint.coeffs,
                constraint.bias,
                constraint.sense,
                constraint.origin,
            )?;
        }

        Ok(Self {
            base: child_base,
            delta: ArenaConstraintStore::new(),
        })
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.base.memory_bytes() + self.delta.memory_bytes()
    }
}
