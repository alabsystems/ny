// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Read-only multi-objective node bounds with allocation provenance.

use std::collections::HashMap;
use std::fmt;
use std::iter::FusedIterator;
use std::mem::size_of;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use ny_tensor::{
    BoundedTensor, BoundedTensorHostAllocationInvalidV1, BoundedTensorHostAllocationProvenanceV1,
    BoundedTensorHostAllocationUnsupportedV1,
};

use crate::network::{
    TrackedStringMap, TrackedStringMapAllocationFactV1, TRACKED_STRING_MAP_ALLOCATION_MODEL_V1,
};

/// Allocation model for [`NodeBoundsMap`].
///
/// V1 composes the pinned `hashbrown 0.16.1` table observation with exact
/// current `String` capacities, the allocation layout requested for each
/// `Arc<BoundedTensor>` by the pinned Rust 1.95.0 implementation, and every
/// tensor's ndarray-0.17.2-qualified host-allocation receipt.
///
/// Arc sharing is charged conservatively per map reference. Two keys that
/// point to the same allocation are each charged one complete Arc allocation
/// request and one complete tensor receipt. External aliases do not reduce the
/// charge. This is intentionally an attribution envelope, not a unique-live-
/// allocation census.
///
/// Only the table component is a historical high-water. Key capacities, Arc
/// requests, and tensor receipts describe the entries live at observation
/// time; V1 does not retain peaks for nested owners that have been removed.
/// Consequently the table high-water may exceed current live table bytes while
/// the composed total may still fall after whole entries are removed.
///
/// V1 excludes allocator metadata and size-class slack, process RSS, unrelated
/// domain fields, and any authority to select or open a retained runtime.
pub const NODE_BOUNDS_HOST_ALLOCATION_MODEL_V1: u32 = 1;

/// Opaque, read-only node-bound carrier.
///
/// Ordinary lookup and iteration remain available, but the concrete table and
/// every table-changing operation are private. In particular this type exposes
/// no `DerefMut`, raw mutable map, `entry`, `insert`, `remove`, `reserve`, or
/// shrink API. Domain code replaces a complete map through audited crate-local
/// constructors, so the table high-water observation cannot be bypassed.
///
/// ```compile_fail
/// use std::sync::Arc;
/// use ny_propagate::beta_crown::NodeBoundsMap;
/// use ny_tensor::BoundedTensor;
///
/// fn cannot_insert(map: &mut NodeBoundsMap, value: Arc<BoundedTensor>) {
///     map.insert("node".to_owned(), value);
/// }
/// ```
///
/// ```compile_fail
/// use ny_propagate::beta_crown::NodeBoundsMap;
///
/// fn cannot_open_entry(map: &mut NodeBoundsMap) {
///     let _ = map.entry("node".to_owned());
/// }
/// ```
pub struct NodeBoundsMap {
    inner: TrackedStringMap<Arc<BoundedTensor>>,
}

impl NodeBoundsMap {
    pub(crate) fn new() -> Self {
        Self {
            inner: TrackedStringMap::new(),
        }
    }

    pub(crate) fn from_shared_hash_map(source: HashMap<String, Arc<BoundedTensor>>) -> Self {
        Self {
            inner: TrackedStringMap::from_entries(source),
        }
    }

    /// Number of live node-bound entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether no node-bound entries are present.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrow one shared tensor by node name.
    #[inline]
    pub fn get(&self, name: &str) -> Option<&Arc<BoundedTensor>> {
        self.inner.get(name)
    }

    /// Whether a node name is present.
    #[inline]
    pub fn contains_key(&self, name: &str) -> bool {
        self.inner.contains_key(name)
    }

    /// Iterate over node names and shared tensors in the table's ordinary
    /// unordered iteration order.
    #[inline]
    pub fn iter(&self) -> NodeBoundsMapIter<'_> {
        NodeBoundsMapIter {
            inner: self.inner.iter(),
        }
    }

    /// Iterate over live node names in ordinary unordered table order.
    #[inline]
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &String> + FusedIterator + '_ {
        self.iter().map(|(name, _)| name)
    }

    /// Iterate over shared tensors in ordinary unordered table order.
    #[inline]
    pub fn values(
        &self,
    ) -> impl ExactSizeIterator<Item = &Arc<BoundedTensor>> + FusedIterator + '_ {
        self.iter().map(|(_, bounds)| bounds)
    }

    /// Make an ordinary map copy for a legacy seam that still requires the
    /// concrete standard-library type. Tensor buffers remain Arc-shared.
    pub(crate) fn to_shared_hash_map(&self) -> HashMap<String, Arc<BoundedTensor>> {
        self.iter()
            .map(|(name, bounds)| (name.clone(), Arc::clone(bounds)))
            .collect()
    }

    pub(crate) fn allocation_fact_v1(&self) -> TrackedStringMapAllocationFactV1 {
        self.inner.allocation_fact_v1()
    }

    /// Observe this exact map's narrow host-allocation model.
    ///
    /// The accounted receipt retains a real source borrow, is neither `Clone`
    /// nor `Copy`, and has no runtime-selection authority. Any unsupported or
    /// invalid tensor makes the whole map observation non-accounted; V1 never
    /// returns a partial byte total.
    pub fn host_allocation_observation_v1(&self) -> NodeBoundsHostAllocationObservationV1<'_> {
        let table = self.allocation_fact_v1();
        if TRACKED_STRING_MAP_ALLOCATION_MODEL_V1 != 1 || table.model_version() != 1 {
            return NodeBoundsHostAllocationObservationV1::Invalid {
                source: self,
                node_name: None,
                reason: NodeBoundsHostAllocationInvalidV1::TrackedStringMapModelMismatch,
            };
        }
        let mut accumulator = NodeBoundsChargeAccumulatorV1::default();
        for (name, bounds) in self.iter() {
            let tensor = match bounds.host_allocation_provenance_v1() {
                BoundedTensorHostAllocationProvenanceV1::Accounted(receipt) => receipt,
                BoundedTensorHostAllocationProvenanceV1::Unsupported(reason) => {
                    return NodeBoundsHostAllocationObservationV1::Unsupported {
                        source: self,
                        node_name: name,
                        reason: NodeBoundsHostAllocationUnsupportedV1::BoundedTensor(reason),
                    };
                }
                BoundedTensorHostAllocationProvenanceV1::Invalid(reason) => {
                    return NodeBoundsHostAllocationObservationV1::Invalid {
                        source: self,
                        node_name: Some(name),
                        reason: NodeBoundsHostAllocationInvalidV1::BoundedTensor(reason),
                    };
                }
                _ => {
                    return NodeBoundsHostAllocationObservationV1::Invalid {
                        source: self,
                        node_name: Some(name),
                        reason: NodeBoundsHostAllocationInvalidV1::BoundedTensorModelMismatch,
                    };
                }
            };

            if let Err(reason) = accumulator.add_reference(
                name.capacity(),
                rust_1_95_arc_allocation_request_bytes(),
                tensor.accountable_charged_bytes(),
            ) {
                return NodeBoundsHostAllocationObservationV1::Invalid {
                    source: self,
                    node_name: Some(name),
                    reason,
                };
            }
        }

        let (
            conservative_accounting_charge_bytes_excluding_inline,
            conservative_accounting_charge_bytes_including_inline,
        ) = match compose_charge_v1(
            table.conservative_table_high_water_charge_bytes(),
            accumulator.key_capacity_bytes,
            accumulator.arc_allocation_request_bytes,
            accumulator.tensor_accounting_charge_bytes,
            size_of::<Self>(),
        ) {
            Ok(charges) => charges,
            Err(reason) => {
                return NodeBoundsHostAllocationObservationV1::Invalid {
                    source: self,
                    node_name: None,
                    reason,
                };
            }
        };

        NodeBoundsHostAllocationObservationV1::Accounted(NodeBoundsHostAllocationReceiptV1 {
            source: self,
            table,
            referenced_tensor_count: accumulator.referenced_tensor_count,
            exact_key_capacity_bytes: accumulator.key_capacity_bytes,
            arc_allocation_request_bytes_per_reference: rust_1_95_arc_allocation_request_bytes(),
            conservative_per_reference_arc_allocation_request_bytes: accumulator
                .arc_allocation_request_bytes,
            bounded_tensor_accounting_charge_bytes: accumulator.tensor_accounting_charge_bytes,
            conservative_accounting_charge_bytes_excluding_inline,
            conservative_accounting_charge_bytes_including_inline,
        })
    }

    // Hostile module tests exercise the complete tracked mutation surface.
    // Production node-bound code intentionally replaces whole maps instead.
    #[cfg(test)]
    pub(crate) fn insert(
        &mut self,
        name: String,
        bounds: Arc<BoundedTensor>,
    ) -> Option<Arc<BoundedTensor>> {
        self.inner.test_insert(name, bounds)
    }

    #[cfg(test)]
    fn remove(&mut self, name: &str) -> Option<Arc<BoundedTensor>> {
        self.inner.test_remove(name)
    }

    #[cfg(test)]
    fn clear(&mut self) {
        self.inner.test_clear();
    }

    #[cfg(test)]
    fn try_reserve(&mut self, additional: usize) -> Result<(), hashbrown::TryReserveError> {
        self.inner.test_try_reserve(additional)
    }

    #[cfg(test)]
    fn shrink_to_fit(&mut self) {
        self.inner.test_shrink_to_fit();
    }
}

impl Clone for NodeBoundsMap {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
    }
}

impl fmt::Debug for NodeBoundsMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

/// Iterator returned by [`NodeBoundsMap::iter`] and `&NodeBoundsMap`.
pub struct NodeBoundsMapIter<'a> {
    inner: hashbrown::hash_map::Iter<'a, String, Arc<BoundedTensor>>,
}

impl<'a> Iterator for NodeBoundsMapIter<'a> {
    type Item = (&'a String, &'a Arc<BoundedTensor>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for NodeBoundsMapIter<'_> {}
impl FusedIterator for NodeBoundsMapIter<'_> {}

impl<'a> IntoIterator for &'a NodeBoundsMap {
    type Item = (&'a String, &'a Arc<BoundedTensor>);
    type IntoIter = NodeBoundsMapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Copyable read-only view over either the legacy standard map or the tracked
/// node-bounds carrier.
///
/// This compatibility view lets shared propagation code consume both owners
/// without exposing either concrete table for mutation. It carries no
/// allocation or verdict authority.
///
/// This is the intentional public compatibility correction for
/// `GraphCrownContext::base_bounds`: callers retain lookup and unordered
/// iteration, while the concrete backing table is no longer observable.
///
/// ```compile_fail
/// use ny_propagate::beta_crown::NodeBoundsView;
///
/// fn cannot_mutate(mut view: NodeBoundsView<'_>) {
///     view.clear();
/// }
/// ```
#[derive(Clone, Copy)]
pub struct NodeBoundsView<'a> {
    inner: NodeBoundsViewInner<'a>,
}

#[derive(Clone, Copy)]
enum NodeBoundsViewInner<'a> {
    Standard(&'a HashMap<String, Arc<BoundedTensor>>),
    Tracked(&'a NodeBoundsMap),
}

impl<'a> NodeBoundsView<'a> {
    /// Borrow a legacy standard-library node-bound map read-only.
    #[inline]
    pub fn from_hash_map(source: &'a HashMap<String, Arc<BoundedTensor>>) -> Self {
        Self {
            inner: NodeBoundsViewInner::Standard(source),
        }
    }

    /// Borrow a provenance-tracked node-bound map read-only.
    #[inline]
    pub fn from_node_bounds_map(source: &'a NodeBoundsMap) -> Self {
        Self {
            inner: NodeBoundsViewInner::Tracked(source),
        }
    }

    /// Number of live entries in the borrowed owner.
    #[inline]
    pub fn len(self) -> usize {
        match self.inner {
            NodeBoundsViewInner::Standard(source) => source.len(),
            NodeBoundsViewInner::Tracked(source) => source.len(),
        }
    }

    /// Whether the borrowed owner has no live entries.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Borrow one shared tensor by node name.
    #[inline]
    pub fn get(self, name: &str) -> Option<&'a Arc<BoundedTensor>> {
        match self.inner {
            NodeBoundsViewInner::Standard(source) => source.get(name),
            NodeBoundsViewInner::Tracked(source) => source.get(name),
        }
    }

    #[inline]
    pub(crate) fn contains_key(self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Iterate over names and tensors in the owner's ordinary unordered order.
    #[inline]
    pub fn iter(self) -> NodeBoundsViewIter<'a> {
        NodeBoundsViewIter {
            inner: match self.inner {
                NodeBoundsViewInner::Standard(source) => {
                    NodeBoundsViewIterInner::Standard(source.iter())
                }
                NodeBoundsViewInner::Tracked(source) => {
                    NodeBoundsViewIterInner::Tracked(source.iter())
                }
            },
        }
    }

    /// Iterate over names in the owner's ordinary unordered order.
    #[inline]
    pub fn keys(self) -> impl ExactSizeIterator<Item = &'a String> + FusedIterator {
        self.iter().map(|(name, _)| name)
    }

    #[inline]
    pub(crate) fn values(
        self,
    ) -> impl ExactSizeIterator<Item = &'a Arc<BoundedTensor>> + FusedIterator {
        self.iter().map(|(_, bounds)| bounds)
    }

    pub(crate) fn to_shared_hash_map(self) -> HashMap<String, Arc<BoundedTensor>> {
        self.iter()
            .map(|(name, bounds)| (name.clone(), Arc::clone(bounds)))
            .collect()
    }
}

impl<'a> From<&'a HashMap<String, Arc<BoundedTensor>>> for NodeBoundsView<'a> {
    fn from(source: &'a HashMap<String, Arc<BoundedTensor>>) -> Self {
        Self::from_hash_map(source)
    }
}

impl<'a> From<&'a NodeBoundsMap> for NodeBoundsView<'a> {
    fn from(source: &'a NodeBoundsMap) -> Self {
        Self::from_node_bounds_map(source)
    }
}

/// Iterator returned by [`NodeBoundsView::iter`].
pub struct NodeBoundsViewIter<'a> {
    inner: NodeBoundsViewIterInner<'a>,
}

enum NodeBoundsViewIterInner<'a> {
    Standard(std::collections::hash_map::Iter<'a, String, Arc<BoundedTensor>>),
    Tracked(NodeBoundsMapIter<'a>),
}

impl<'a> Iterator for NodeBoundsViewIter<'a> {
    type Item = (&'a String, &'a Arc<BoundedTensor>);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            NodeBoundsViewIterInner::Standard(iter) => iter.next(),
            NodeBoundsViewIterInner::Tracked(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            NodeBoundsViewIterInner::Standard(iter) => iter.size_hint(),
            NodeBoundsViewIterInner::Tracked(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for NodeBoundsViewIter<'_> {}
impl FusedIterator for NodeBoundsViewIter<'_> {}

/// Clean V1 capability miss for one map entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NodeBoundsHostAllocationUnsupportedV1 {
    BoundedTensor(BoundedTensorHostAllocationUnsupportedV1),
}

/// Hard V1 provenance failure. No byte count is authoritative in this state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NodeBoundsHostAllocationInvalidV1 {
    TrackedStringMapModelMismatch,
    BoundedTensor(BoundedTensorHostAllocationInvalidV1),
    BoundedTensorModelMismatch,
    ReferencedTensorCountOverflow,
    KeyCapacityBytesOverflow,
    ArcAllocationRequestBytesOverflow,
    BoundedTensorAccountingChargeBytesOverflow,
    TotalAccountingChargeOverflow,
}

/// Borrow-bound, non-authorizing node-bounds allocation observation.
///
/// Unsupported/invalid results retain the same exact source borrow and name
/// the first failing entry. `Accounted` is a conservative per-reference model,
/// not a unique-allocation or process-memory census.
#[must_use]
#[derive(Debug)]
#[non_exhaustive]
pub enum NodeBoundsHostAllocationObservationV1<'a> {
    Accounted(NodeBoundsHostAllocationReceiptV1<'a>),
    Unsupported {
        source: &'a NodeBoundsMap,
        node_name: &'a str,
        reason: NodeBoundsHostAllocationUnsupportedV1,
    },
    Invalid {
        source: &'a NodeBoundsMap,
        node_name: Option<&'a str>,
        reason: NodeBoundsHostAllocationInvalidV1,
    },
}

/// Accounted V1 facts for one exact [`NodeBoundsMap`].
///
/// This receipt is deliberately not `Clone` or `Copy`. Its source borrow
/// freezes safe replacement of the map while the facts are live, but grants no
/// provider, plan, admission, or runtime authority.
///
/// ```compile_fail
/// use ny_propagate::beta_crown::NodeBoundsHostAllocationReceiptV1;
///
/// fn cannot_clone(receipt: NodeBoundsHostAllocationReceiptV1<'_>) {
///     let _ = receipt.clone();
/// }
/// ```
#[must_use]
pub struct NodeBoundsHostAllocationReceiptV1<'a> {
    source: &'a NodeBoundsMap,
    table: TrackedStringMapAllocationFactV1,
    referenced_tensor_count: usize,
    exact_key_capacity_bytes: usize,
    arc_allocation_request_bytes_per_reference: usize,
    conservative_per_reference_arc_allocation_request_bytes: usize,
    bounded_tensor_accounting_charge_bytes: usize,
    conservative_accounting_charge_bytes_excluding_inline: usize,
    conservative_accounting_charge_bytes_including_inline: usize,
}

impl fmt::Debug for NodeBoundsHostAllocationReceiptV1<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NodeBoundsHostAllocationReceiptV1")
            .field("model_version", &self.model_version())
            .field("inline_owner_bytes", &self.inline_owner_bytes())
            .field(
                "current_table_allocation_bytes",
                &self.current_table_allocation_bytes(),
            )
            .field(
                "conservative_table_high_water_charge_bytes",
                &self.conservative_table_high_water_charge_bytes(),
            )
            .field("referenced_tensor_count", &self.referenced_tensor_count)
            .field("exact_key_capacity_bytes", &self.exact_key_capacity_bytes)
            .field(
                "arc_allocation_request_bytes_per_reference",
                &self.arc_allocation_request_bytes_per_reference,
            )
            .field(
                "conservative_per_reference_arc_allocation_request_bytes",
                &self.conservative_per_reference_arc_allocation_request_bytes,
            )
            .field(
                "bounded_tensor_accounting_charge_bytes",
                &self.bounded_tensor_accounting_charge_bytes,
            )
            .field(
                "conservative_accounting_charge_bytes_excluding_inline",
                &self.conservative_accounting_charge_bytes_excluding_inline,
            )
            .field(
                "conservative_accounting_charge_bytes_including_inline",
                &self.conservative_accounting_charge_bytes_including_inline,
            )
            .finish_non_exhaustive()
    }
}

impl NodeBoundsHostAllocationReceiptV1<'_> {
    /// Whether this receipt retains the exact supplied source map.
    #[inline]
    pub fn matches_source(&self, source: &NodeBoundsMap) -> bool {
        std::ptr::eq(self.source, source)
    }

    /// Version of the composed node-bounds accounting model.
    #[inline]
    pub fn model_version(&self) -> u32 {
        debug_assert_eq!(
            self.table.model_version(),
            TRACKED_STRING_MAP_ALLOCATION_MODEL_V1
        );
        NODE_BOUNDS_HOST_ALLOCATION_MODEL_V1
    }

    /// Inline bytes of the opaque node-bounds owner itself.
    #[inline]
    pub fn inline_owner_bytes(&self) -> usize {
        size_of::<NodeBoundsMap>()
    }

    /// Pinned hashbrown's current table-allocation layout observation.
    #[inline]
    pub fn current_table_allocation_bytes(&self) -> usize {
        self.table.current_table_allocation_bytes()
    }

    /// Greatest table-allocation observation over this owner's history.
    ///
    /// This historical charge can exceed the table bytes still live now.
    #[inline]
    pub fn conservative_table_high_water_charge_bytes(&self) -> usize {
        self.table.conservative_table_high_water_charge_bytes()
    }

    /// Number of live map references whose nested ownership was charged.
    #[inline]
    pub fn referenced_tensor_count(&self) -> usize {
        self.referenced_tensor_count
    }

    /// Exact sum of every live `String` key's byte capacity.
    #[inline]
    pub fn exact_key_capacity_bytes(&self) -> usize {
        self.exact_key_capacity_bytes
    }

    /// Allocation-layout bytes requested by Rust 1.95.0 for one
    /// `Arc<BoundedTensor>`, including its two atomic counters and inline
    /// tensor owner but excluding tensor-owned buffers and allocator metadata.
    #[inline]
    pub fn arc_allocation_request_bytes_per_reference(&self) -> usize {
        self.arc_allocation_request_bytes_per_reference
    }

    /// Per-reference Arc request charge. Duplicate keys and external aliases
    /// deliberately do not deduplicate this total.
    #[inline]
    pub fn conservative_per_reference_arc_allocation_request_bytes(&self) -> usize {
        self.conservative_per_reference_arc_allocation_request_bytes
    }

    /// Sum of every live referenced tensor's ndarray-qualified host charge.
    #[inline]
    pub fn bounded_tensor_accounting_charge_bytes(&self) -> usize {
        self.bounded_tensor_accounting_charge_bytes
    }

    /// Table high-water plus live key capacities, per-reference Arc requests,
    /// and every tensor receipt, excluding the map wrapper's inline bytes.
    #[inline]
    pub fn conservative_accounting_charge_bytes_excluding_inline(&self) -> usize {
        self.conservative_accounting_charge_bytes_excluding_inline
    }

    /// Complete V1 charge including the map wrapper's inline owner.
    #[inline]
    pub fn conservative_accounting_charge_bytes_including_inline(&self) -> usize {
        self.conservative_accounting_charge_bytes_including_inline
    }
}

/// Private mirror of Rust 1.95.0's `alloc::sync::ArcInner<T>` layout.
///
/// Source-qualified at:
/// <https://github.com/rust-lang/rust/blob/1.95.0/library/alloc/src/sync.rs>
/// (`ArcInner<T>` and `Arc::new`). The workspace pins exactly Rust 1.95.0;
/// changing that pin requires a new model or a source/layout re-audit.
#[repr(C)]
struct Rust195ArcInnerLayout<T> {
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: T,
}

#[inline]
fn rust_1_95_arc_allocation_request_bytes() -> usize {
    size_of::<Rust195ArcInnerLayout<BoundedTensor>>()
}

#[derive(Default)]
struct NodeBoundsChargeAccumulatorV1 {
    referenced_tensor_count: usize,
    key_capacity_bytes: usize,
    arc_allocation_request_bytes: usize,
    tensor_accounting_charge_bytes: usize,
}

impl NodeBoundsChargeAccumulatorV1 {
    fn add_reference(
        &mut self,
        key_capacity_bytes: usize,
        arc_allocation_request_bytes: usize,
        tensor_accounting_charge_bytes: usize,
    ) -> Result<(), NodeBoundsHostAllocationInvalidV1> {
        self.referenced_tensor_count = self
            .referenced_tensor_count
            .checked_add(1)
            .ok_or(NodeBoundsHostAllocationInvalidV1::ReferencedTensorCountOverflow)?;
        self.key_capacity_bytes = self
            .key_capacity_bytes
            .checked_add(key_capacity_bytes)
            .ok_or(NodeBoundsHostAllocationInvalidV1::KeyCapacityBytesOverflow)?;
        self.arc_allocation_request_bytes = self
            .arc_allocation_request_bytes
            .checked_add(arc_allocation_request_bytes)
            .ok_or(NodeBoundsHostAllocationInvalidV1::ArcAllocationRequestBytesOverflow)?;
        self.tensor_accounting_charge_bytes = self
            .tensor_accounting_charge_bytes
            .checked_add(tensor_accounting_charge_bytes)
            .ok_or(NodeBoundsHostAllocationInvalidV1::BoundedTensorAccountingChargeBytesOverflow)?;
        Ok(())
    }
}

fn compose_charge_v1(
    table_high_water_bytes: usize,
    key_capacity_bytes: usize,
    arc_allocation_request_bytes: usize,
    tensor_accounting_charge_bytes: usize,
    inline_owner_bytes: usize,
) -> Result<(usize, usize), NodeBoundsHostAllocationInvalidV1> {
    let excluding_inline = table_high_water_bytes
        .checked_add(key_capacity_bytes)
        .and_then(|bytes| bytes.checked_add(arc_allocation_request_bytes))
        .and_then(|bytes| bytes.checked_add(tensor_accounting_charge_bytes))
        .ok_or(NodeBoundsHostAllocationInvalidV1::TotalAccountingChargeOverflow)?;
    let including_inline = inline_owner_bytes
        .checked_add(excluding_inline)
        .ok_or(NodeBoundsHostAllocationInvalidV1::TotalAccountingChargeOverflow)?;
    Ok((excluding_inline, including_inline))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> Arc<BoundedTensor> {
        Arc::new(BoundedTensor::new_conservative(&[1]))
    }

    fn accounted(map: &NodeBoundsMap) -> NodeBoundsHostAllocationReceiptV1<'_> {
        match map.host_allocation_observation_v1() {
            NodeBoundsHostAllocationObservationV1::Accounted(receipt) => receipt,
            other => panic!("expected accounted node bounds, got {other:?}"),
        }
    }

    #[test]
    fn empty_map_is_source_bound_and_charges_only_inline_owner() {
        let map = NodeBoundsMap::new();
        let receipt = accounted(&map);

        assert!(receipt.matches_source(&map));
        assert_eq!(receipt.model_version(), 1);
        assert_eq!(receipt.inline_owner_bytes(), size_of::<NodeBoundsMap>());
        assert_eq!(receipt.current_table_allocation_bytes(), 0);
        assert_eq!(receipt.conservative_table_high_water_charge_bytes(), 0);
        assert_eq!(receipt.referenced_tensor_count(), 0);
        assert_eq!(
            receipt.conservative_accounting_charge_bytes_including_inline(),
            size_of::<NodeBoundsMap>()
        );
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[test]
    fn rust_1_95_arc_and_hashbrown_0_16_1_layout_kat() {
        assert_eq!(size_of::<BoundedTensor>(), 264);
        assert_eq!(rust_1_95_arc_allocation_request_bytes(), 280);
        assert_eq!(size_of::<NodeBoundsMap>(), 56);

        let mut map = NodeBoundsMap::new();
        map.try_reserve(64).unwrap();
        let reserved = accounted(&map);
        assert_eq!(reserved.current_table_allocation_bytes(), 4_240);
        assert_eq!(reserved.conservative_table_high_water_charge_bytes(), 4_240);
    }

    #[test]
    fn key_spare_capacity_and_duplicate_arc_references_are_each_charged() {
        let shared = bounds();
        let external_alias = Arc::clone(&shared);
        let tensor_charge = match shared.host_allocation_provenance_v1() {
            BoundedTensorHostAllocationProvenanceV1::Accounted(receipt) => {
                receipt.accountable_charged_bytes()
            }
            other => panic!("expected accounted tensor, got {other:?}"),
        };
        let mut first = String::with_capacity(64);
        first.push_str("first");
        let first_capacity = first.capacity();
        let mut second = String::with_capacity(96);
        second.push_str("second");
        let second_capacity = second.capacity();

        let mut map = NodeBoundsMap::new();
        map.insert(first, Arc::clone(&shared));
        map.insert(second, shared);
        let receipt = accounted(&map);

        assert_eq!(receipt.referenced_tensor_count(), 2);
        assert_eq!(
            receipt.exact_key_capacity_bytes(),
            first_capacity + second_capacity
        );
        assert_eq!(
            receipt.conservative_per_reference_arc_allocation_request_bytes(),
            2 * rust_1_95_arc_allocation_request_bytes()
        );
        assert_eq!(
            receipt.bounded_tensor_accounting_charge_bytes(),
            2 * tensor_charge
        );
        let expected_excluding_inline = receipt.conservative_table_high_water_charge_bytes()
            + first_capacity
            + second_capacity
            + 2 * rust_1_95_arc_allocation_request_bytes()
            + 2 * tensor_charge;
        assert_eq!(
            receipt.conservative_accounting_charge_bytes_excluding_inline(),
            expected_excluding_inline
        );
        assert_eq!(
            receipt.conservative_accounting_charge_bytes_including_inline(),
            size_of::<NodeBoundsMap>() + expected_excluding_inline
        );
        assert_eq!(Arc::strong_count(&external_alias), 3);
    }

    #[test]
    fn tombstones_clear_and_shrink_cannot_erase_table_high_water() {
        let mut map = NodeBoundsMap::new();
        map.try_reserve(64).unwrap();
        for index in 0..32 {
            map.insert(format!("node-{index}"), bounds());
        }
        let peak = accounted(&map).conservative_table_high_water_charge_bytes();

        for index in 0..16 {
            assert!(map.remove(&format!("node-{index}")).is_some());
        }
        assert_eq!(
            accounted(&map).conservative_table_high_water_charge_bytes(),
            peak
        );
        map.clear();
        map.shrink_to_fit();
        let shrunk = accounted(&map);
        assert_eq!(shrunk.current_table_allocation_bytes(), 0);
        assert_eq!(shrunk.conservative_table_high_water_charge_bytes(), peak);
    }

    #[test]
    fn clone_recaptures_the_new_table_instead_of_inheriting_a_dropped_peak() {
        let mut source = NodeBoundsMap::new();
        source.try_reserve(128).unwrap();
        source.insert("live".to_owned(), bounds());
        let source_peak = accounted(&source).conservative_table_high_water_charge_bytes();
        source.shrink_to_fit();

        let cloned = source.clone();
        let cloned_receipt = accounted(&cloned);
        assert_eq!(
            cloned_receipt.conservative_table_high_water_charge_bytes(),
            cloned_receipt.current_table_allocation_bytes()
        );
        assert!(cloned_receipt.conservative_table_high_water_charge_bytes() < source_peak);
        assert!(Arc::ptr_eq(
            source.get("live").unwrap(),
            cloned.get("live").unwrap()
        ));

        let mut clone_from_target = NodeBoundsMap::new();
        clone_from_target.try_reserve(256).unwrap();
        clone_from_target.clone_from(&source);
        let clone_from_receipt = accounted(&clone_from_target);
        assert_eq!(
            clone_from_receipt.conservative_table_high_water_charge_bytes(),
            clone_from_receipt.current_table_allocation_bytes()
        );
        assert!(clone_from_receipt.conservative_table_high_water_charge_bytes() < source_peak);
        assert!(Arc::ptr_eq(
            source.get("live").unwrap(),
            clone_from_target.get("live").unwrap()
        ));
    }

    #[test]
    fn conversion_and_views_preserve_entries_and_arc_identity() {
        let first = bounds();
        let second = bounds();
        let retained_first = Arc::clone(&first);
        let retained_second = Arc::clone(&second);
        let standard = HashMap::from([("first".to_owned(), first), ("second".to_owned(), second)]);

        let tracked = NodeBoundsMap::from_shared_hash_map(standard.clone());
        assert_eq!(tracked.len(), standard.len());
        assert!(Arc::ptr_eq(tracked.get("first").unwrap(), &retained_first));
        assert!(Arc::ptr_eq(
            tracked.get("second").unwrap(),
            &retained_second
        ));

        let mut standard_keys: Vec<_> = NodeBoundsView::from_hash_map(&standard)
            .keys()
            .map(String::as_str)
            .collect();
        let mut tracked_keys: Vec<_> = NodeBoundsView::from_node_bounds_map(&tracked)
            .keys()
            .map(String::as_str)
            .collect();
        standard_keys.sort_unstable();
        tracked_keys.sort_unstable();
        assert_eq!(tracked_keys, standard_keys);
    }

    #[test]
    fn one_unsupported_tensor_refuses_the_whole_map_without_partial_bytes() {
        use ndarray::{ArrayD, IxDyn};
        use ny_tensor::L2Constraint;

        let l2 = L2Constraint::new(
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::from_elem(IxDyn(&[]), 1.0),
            0,
            &[1],
        )
        .expect("valid test L2 constraint");
        let unsupported = Arc::new(BoundedTensor::new_conservative(&[1]).with_l2_constraint(l2));
        let mut map = NodeBoundsMap::new();
        map.insert("accounted".to_owned(), bounds());
        map.insert("unsupported".to_owned(), unsupported);

        match map.host_allocation_observation_v1() {
            NodeBoundsHostAllocationObservationV1::Unsupported {
                source,
                node_name,
                reason:
                    NodeBoundsHostAllocationUnsupportedV1::BoundedTensor(
                        BoundedTensorHostAllocationUnsupportedV1::L2ConstraintPresent,
                    ),
            } => {
                assert!(std::ptr::eq(source, &raw const map));
                assert_eq!(node_name, "unsupported");
            }
            other => panic!("expected whole-map unsupported result, got {other:?}"),
        }
    }

    #[test]
    fn checked_accumulator_rejects_every_overflow_without_a_partial_total() {
        let mut count = NodeBoundsChargeAccumulatorV1 {
            referenced_tensor_count: usize::MAX,
            ..Default::default()
        };
        assert_eq!(
            count.add_reference(0, 0, 0),
            Err(NodeBoundsHostAllocationInvalidV1::ReferencedTensorCountOverflow)
        );

        let mut keys = NodeBoundsChargeAccumulatorV1 {
            key_capacity_bytes: usize::MAX,
            ..Default::default()
        };
        assert_eq!(
            keys.add_reference(1, 0, 0),
            Err(NodeBoundsHostAllocationInvalidV1::KeyCapacityBytesOverflow)
        );

        let mut arcs = NodeBoundsChargeAccumulatorV1 {
            arc_allocation_request_bytes: usize::MAX,
            ..Default::default()
        };
        assert_eq!(
            arcs.add_reference(0, 1, 0),
            Err(NodeBoundsHostAllocationInvalidV1::ArcAllocationRequestBytesOverflow)
        );

        let mut tensors = NodeBoundsChargeAccumulatorV1 {
            tensor_accounting_charge_bytes: usize::MAX,
            ..Default::default()
        };
        assert_eq!(
            tensors.add_reference(0, 0, 1),
            Err(NodeBoundsHostAllocationInvalidV1::BoundedTensorAccountingChargeBytesOverflow)
        );

        assert_eq!(
            compose_charge_v1(usize::MAX, 1, 0, 0, 0),
            Err(NodeBoundsHostAllocationInvalidV1::TotalAccountingChargeOverflow)
        );
        assert_eq!(
            compose_charge_v1(0, 0, 0, usize::MAX, 1),
            Err(NodeBoundsHostAllocationInvalidV1::TotalAccountingChargeOverflow)
        );
    }
}
