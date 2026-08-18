// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared logical-byte census for resident `DomainList` rows.
//!
//! The census intentionally measures live queue payload, not allocator
//! capacity or process RSS. It is used by both queue eviction and adaptive
//! microbatch sizing so those two memory controls cannot drift.

use std::mem::size_of;

use super::types::DomainMetadata;
use super::DomainList;

impl DomainList {
    /// Fixed tensor payload carried by every scalar-objective queue row.
    pub(super) fn estimated_tensor_bytes_per_domain(&self) -> usize {
        let tensor_elements = self
            .config
            .layer_names
            .iter()
            .filter_map(|name| self.config.layer_shapes.get(name))
            .map(|shape| shape.iter().copied().fold(1usize, usize::saturating_mul))
            .sum::<usize>()
            .saturating_add(
                self.config
                    .input_shape
                    .iter()
                    .copied()
                    .fold(1usize, usize::saturating_mul),
            )
            .saturating_add(1);
        tensor_elements.saturating_mul(2usize.saturating_mul(size_of::<f32>()))
    }

    /// Estimated bytes for one specific resident row.
    pub(super) fn estimated_row_bytes(&self, metadata: &DomainMetadata) -> usize {
        self.estimated_tensor_bytes_per_domain()
            .saturating_add(metadata.estimated_owned_bytes())
            .max(1)
    }

    /// Estimated total live payload in the scalar-objective resident frontier.
    pub(crate) fn estimated_resident_bytes(&self) -> usize {
        let tensor_bytes = self
            .estimated_tensor_bytes_per_domain()
            .saturating_mul(self.metadata.len());
        self.metadata
            .iter()
            .map(DomainMetadata::estimated_owned_bytes)
            .fold(tensor_bytes, usize::saturating_add)
    }

    /// Estimated average row footprint before pick-out.
    pub(crate) fn estimated_bytes_per_domain(&self) -> usize {
        if self.metadata.is_empty() {
            return self
                .estimated_tensor_bytes_per_domain()
                .saturating_add(size_of::<DomainMetadata>())
                .max(1);
        }
        self.estimated_resident_bytes()
            .div_ceil(self.metadata.len())
            .max(1)
    }
}
