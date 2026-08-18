// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared allocation-provenance carrier for String-keyed maps.

// Graph structural maps and the multi-objective node-bounds carrier share this
// implementation. Every table-changing surface is pinned by hostile tests.
#![cfg_attr(not(test), allow(dead_code))]

use std::{collections::hash_map::RandomState, fmt, mem::size_of, ops::Deref};

use hashbrown::{HashMap, TryReserveError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Version of the tracked String-map table-allocation model.
///
/// V1 is tied to exactly `hashbrown 0.16.1`. Its public
/// [`HashMap::allocation_size`] observation covers the table allocation used
/// for buckets, control bytes, and their layout padding. The conservative
/// charge is the greatest such observation over this owner's mutation
/// history, plus this wrapper's inline bytes.
///
/// This is not allocator bookkeeping, usable-size slack, RSS, or a charge for
/// allocations owned inside `String` keys or values. Those nested owners need
/// separate provenance. The exact dependency pin and known-answer tests are
/// part of this model; changing either requires a new version or a proof that
/// the V1 observations are unchanged.
pub(crate) const TRACKED_STRING_MAP_ALLOCATION_MODEL_V1: u32 = 1;

type InnerMap<V> = HashMap<String, V, RandomState>;

/// One tracked String map's table-allocation facts.
///
/// `current_table_allocation_bytes` is the pinned implementation's exact table
/// layout observation. `conservative_table_high_water_charge_bytes` is a
/// historical accounting envelope: after shrinking, it deliberately need not
/// describe bytes still allocated by the current table. Callers that already
/// include the wrapper through a containing object's inline `size_of` must not
/// add `inline_owner_bytes` a second time.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TrackedStringMapAllocationFactV1 {
    model_version: u32,
    inline_owner_bytes: usize,
    current_table_allocation_bytes: usize,
    conservative_table_high_water_charge_bytes: usize,
}

impl TrackedStringMapAllocationFactV1 {
    pub(crate) fn model_version(&self) -> u32 {
        self.model_version
    }

    pub(crate) fn inline_owner_bytes(&self) -> usize {
        self.inline_owner_bytes
    }

    pub(crate) fn current_table_allocation_bytes(&self) -> usize {
        self.current_table_allocation_bytes
    }

    pub(crate) fn conservative_table_high_water_charge_bytes(&self) -> usize {
        self.conservative_table_high_water_charge_bytes
    }

    pub(crate) fn conservative_accounting_charge_bytes_including_inline(&self) -> Option<usize> {
        self.inline_owner_bytes
            .checked_add(self.conservative_table_high_water_charge_bytes)
    }
}

/// String-keyed hash map that keeps table-allocation provenance complete.
///
/// `Deref` is deliberately read-only. Structural mutation is limited to the
/// methods below, each of which refreshes the high-water observation after the
/// underlying operation. In particular there is no `DerefMut`, raw mutable-map
/// accessor, or entry API that could insert without refreshing provenance.
pub(crate) struct TrackedStringMap<V> {
    inner: InnerMap<V>,
    table_allocation_high_water_bytes: usize,
}

impl<V> TrackedStringMap<V> {
    pub(crate) fn new() -> Self {
        Self::from_inner(InnerMap::with_hasher(RandomState::new()))
    }

    fn from_inner(inner: InnerMap<V>) -> Self {
        let table_allocation_high_water_bytes = inner.allocation_size();
        Self {
            inner,
            table_allocation_high_water_bytes,
        }
    }

    /// Build a new owner from semantic entries and observe its actual table.
    ///
    /// This deliberately does not infer allocation from the iterator's size
    /// hint or from a source map's public capacity. The pinned map builds its
    /// table normally, then V1 observes that table's allocation layout.
    pub(crate) fn from_entries(entries: impl IntoIterator<Item = (String, V)>) -> Self {
        Self::from_inner(entries.into_iter().collect())
    }

    fn refresh_table_allocation_high_water(&mut self) {
        self.table_allocation_high_water_bytes = self
            .table_allocation_high_water_bytes
            .max(self.inner.allocation_size());
    }

    pub(crate) fn allocation_fact_v1(&self) -> TrackedStringMapAllocationFactV1 {
        let current_table_allocation_bytes = self.inner.allocation_size();
        debug_assert!(
            self.table_allocation_high_water_bytes >= current_table_allocation_bytes,
            "tracked String map current allocation exceeds its accounting high-water"
        );
        TrackedStringMapAllocationFactV1 {
            model_version: TRACKED_STRING_MAP_ALLOCATION_MODEL_V1,
            inline_owner_bytes: size_of::<Self>(),
            current_table_allocation_bytes,
            conservative_table_high_water_charge_bytes: self.table_allocation_high_water_bytes,
        }
    }

    pub(super) fn insert(&mut self, key: String, value: V) -> Option<V> {
        let replaced = self.inner.insert(key, value);
        self.refresh_table_allocation_high_water();
        replaced
    }

    /// A value-only mutable borrow cannot change the hash table allocation.
    pub(crate) fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.inner.get_mut(key)
    }

    /// Iterating mutably over values cannot change the hash table allocation.
    pub(super) fn values_mut(&mut self) -> hashbrown::hash_map::ValuesMut<'_, String, V> {
        self.inner.values_mut()
    }

    pub(super) fn remove(&mut self, key: &str) -> Option<V> {
        let removed = self.inner.remove(key);
        self.refresh_table_allocation_high_water();
        removed
    }

    pub(super) fn remove_entry(&mut self, key: &str) -> Option<(String, V)> {
        let removed = self.inner.remove_entry(key);
        self.refresh_table_allocation_high_water();
        removed
    }

    pub(super) fn clear(&mut self) {
        self.inner.clear();
        self.refresh_table_allocation_high_water();
    }

    pub(super) fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
        self.refresh_table_allocation_high_water();
    }

    pub(super) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let result = self.inner.try_reserve(additional);
        self.refresh_table_allocation_high_water();
        result
    }

    pub(super) fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
        self.refresh_table_allocation_high_water();
    }

    pub(super) fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity);
        self.refresh_table_allocation_high_water();
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&String, &mut V) -> bool) {
        self.inner.retain(|key, value| keep(key, value));
        self.refresh_table_allocation_high_water();
    }
}

#[cfg(test)]
impl<V> TrackedStringMap<V> {
    pub(crate) fn test_insert(&mut self, key: String, value: V) -> Option<V> {
        self.insert(key, value)
    }

    pub(crate) fn test_remove(&mut self, key: &str) -> Option<V> {
        self.remove(key)
    }

    pub(crate) fn test_clear(&mut self) {
        self.clear();
    }

    pub(crate) fn test_try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve(additional)
    }

    pub(crate) fn test_shrink_to_fit(&mut self) {
        self.shrink_to_fit();
    }
}

impl<V> Default for TrackedStringMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone> Clone for TrackedStringMap<V> {
    fn clone(&self) -> Self {
        // A clone owns a distinct allocation. Start its high-water from the
        // actual cloned table rather than copying historical source peaks.
        Self::from_inner(self.inner.clone())
    }

    fn clone_from(&mut self, source: &Self) {
        // Build the replacement first. If cloning a value panics, `self`
        // remains untouched and its provenance remains valid. The successful
        // replacement starts from the distinct cloned table's actual bytes.
        *self = source.clone();
    }
}

impl<V: fmt::Debug> fmt::Debug for TrackedStringMap<V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Provenance is metadata, not part of the semantic map representation.
        self.inner.fmt(formatter)
    }
}
impl<V> Deref for TrackedStringMap<V> {
    type Target = InnerMap<V>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, V> IntoIterator for &'a TrackedStringMap<V> {
    type Item = (&'a String, &'a V);
    type IntoIter = hashbrown::hash_map::Iter<'a, String, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<V: Serialize> Serialize for TrackedStringMap<V> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Provenance is intentionally not accepted as serialized authority.
        // Only semantic entries cross the wire.
        self.inner.serialize(serializer)
    }
}

impl<'de, V: Deserialize<'de>> Deserialize<'de> for TrackedStringMap<V> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = InnerMap::<V>::deserialize(deserializer)?;
        // Re-observe the actual allocation created by this decoder; never
        // trust or reconstruct provenance from logical entry counts.
        Ok(Self::from_inner(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation<V>(map: &TrackedStringMap<V>) -> TrackedStringMapAllocationFactV1 {
        map.allocation_fact_v1()
    }

    #[test]
    fn empty_map_has_only_inline_owner_charge() {
        let map = TrackedStringMap::<u64>::new();
        let observed = observation(&map);

        assert_eq!(observed.model_version(), 1);
        assert_eq!(
            observed.inline_owner_bytes(),
            size_of::<TrackedStringMap<u64>>()
        );
        assert_eq!(observed.current_table_allocation_bytes(), 0);
        assert_eq!(observed.conservative_table_high_water_charge_bytes(), 0);
        assert_eq!(
            observed.conservative_accounting_charge_bytes_including_inline(),
            Some(size_of::<TrackedStringMap<u64>>())
        );
    }

    /// Literal V1 KAT for the reviewed x86_64 table layout. These values bind
    /// the dependency pin to bucket storage, control bytes, layout padding,
    /// and the wrapper's inline high-water field. They are deliberately not
    /// inferred from logical capacity.
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[test]
    fn pinned_hashbrown_0_16_1_allocation_layout_kat() {
        let mut map = TrackedStringMap::<u64>::new();
        assert_eq!(size_of::<TrackedStringMap<u64>>(), 56);

        map.try_reserve(64).unwrap();
        let reserved = observation(&map);
        assert_eq!(reserved.current_table_allocation_bytes(), 4_240);
        assert_eq!(reserved.conservative_table_high_water_charge_bytes(), 4_240);
        assert_eq!(
            reserved.conservative_accounting_charge_bytes_including_inline(),
            Some(4_296)
        );

        map.insert("one".to_owned(), 1);
        map.shrink_to_fit();
        let shrunk = observation(&map);
        assert_eq!(shrunk.current_table_allocation_bytes(), 148);
        assert_eq!(shrunk.conservative_table_high_water_charge_bytes(), 4_240);
    }

    #[test]
    fn growth_removal_clear_and_shrink_never_erase_high_water() {
        let mut map = TrackedStringMap::new();
        map.try_reserve(64).unwrap();
        for index in 0_u64..64 {
            map.insert(format!("node-{index}"), index);
        }
        let peak = observation(&map);
        assert!(peak.current_table_allocation_bytes() > 0);
        assert_eq!(
            peak.conservative_table_high_water_charge_bytes(),
            peak.current_table_allocation_bytes()
        );

        for index in 0_u64..48 {
            assert_eq!(map.remove(&format!("node-{index}")), Some(index));
        }
        let tombstoned = observation(&map);
        assert_eq!(
            tombstoned.conservative_table_high_water_charge_bytes(),
            peak.conservative_table_high_water_charge_bytes()
        );

        map.clear();
        let cleared = observation(&map);
        assert_eq!(map.len(), 0);
        assert_eq!(
            cleared.conservative_table_high_water_charge_bytes(),
            peak.conservative_table_high_water_charge_bytes()
        );

        map.shrink_to_fit();
        let shrunk = observation(&map);
        assert_eq!(shrunk.current_table_allocation_bytes(), 0);
        assert_eq!(
            shrunk.conservative_table_high_water_charge_bytes(),
            peak.conservative_table_high_water_charge_bytes()
        );
    }

    #[test]
    fn spare_capacity_is_charged_even_when_logical_length_is_small() {
        let mut map = TrackedStringMap::new();
        map.try_reserve(128).unwrap();
        map.insert("only".to_owned(), 7_u64);
        let observed = observation(&map);

        assert_eq!(map.len(), 1);
        assert!(map.capacity() >= 128);
        assert!(observed.current_table_allocation_bytes() > size_of::<(String, u64)>());
        assert_eq!(
            observed.current_table_allocation_bytes(),
            observed.conservative_table_high_water_charge_bytes()
        );
    }

    #[test]
    fn clone_and_clone_from_recapture_each_new_owner_allocation() {
        let mut source = TrackedStringMap::new();
        source.try_reserve(128).unwrap();
        source.insert("source".to_owned(), 1_u64);
        let source_peak = observation(&source).conservative_table_high_water_charge_bytes();
        source.shrink_to_fit();

        let cloned = source.clone();
        assert_eq!(cloned.get("source"), Some(&1));
        assert_eq!(
            observation(&cloned).conservative_table_high_water_charge_bytes(),
            observation(&cloned).current_table_allocation_bytes()
        );
        assert!(observation(&cloned).conservative_table_high_water_charge_bytes() < source_peak);

        let mut target = TrackedStringMap::new();
        target.try_reserve(256).unwrap();
        let target_peak = observation(&target).conservative_table_high_water_charge_bytes();
        target.clone_from(&source);
        assert_eq!(target.get("source"), Some(&1));
        assert_eq!(
            observation(&target).conservative_table_high_water_charge_bytes(),
            observation(&target).current_table_allocation_bytes()
        );
        assert!(observation(&target).conservative_table_high_water_charge_bytes() < target_peak);
    }

    #[test]
    fn serde_reobserves_decoder_allocation_and_does_not_serialize_provenance() {
        let mut source = TrackedStringMap::new();
        source.try_reserve(256).unwrap();
        for index in 0_u64..8 {
            source.insert(format!("node-{index}"), index);
        }
        let source_peak = observation(&source).conservative_table_high_water_charge_bytes();

        let encoded = serde_json::to_string(&source).unwrap();
        assert!(!encoded.contains("high_water"));
        let decoded: TrackedStringMap<u64> = serde_json::from_str(&encoded).unwrap();
        let decoded_observation = observation(&decoded);

        assert_eq!(decoded.len(), source.len());
        assert_eq!(
            decoded_observation.current_table_allocation_bytes(),
            decoded_observation.conservative_table_high_water_charge_bytes()
        );
        assert!(decoded_observation.current_table_allocation_bytes() > 0);
        assert!(decoded_observation.conservative_table_high_water_charge_bytes() < source_peak);
    }

    #[test]
    fn every_exposed_structural_mutator_keeps_high_water_sound() {
        let mut map = TrackedStringMap::new();
        map.reserve(4);
        map.insert("a".to_owned(), 1_u64);
        map.insert("b".to_owned(), 2_u64);
        let before = observation(&map).conservative_table_high_water_charge_bytes();

        assert_eq!(map.remove_entry("a"), Some(("a".to_owned(), 1)));
        map.retain(|key, _| key == "b");
        map.shrink_to(1);
        let after = observation(&map);

        assert_eq!(map.get_mut("b").map(|value| *value += 1), Some(()));
        assert_eq!(map.get("b"), Some(&3));
        assert!(after.conservative_table_high_water_charge_bytes() >= before);
        assert!(
            after.conservative_table_high_water_charge_bytes()
                >= after.current_table_allocation_bytes()
        );
    }

    #[test]
    fn failed_try_reserve_reobserves_without_losing_prior_charge() {
        let mut map = TrackedStringMap::<u64>::new();
        map.insert("live".to_owned(), 1);
        let before = observation(&map).conservative_table_high_water_charge_bytes();

        assert!(map.try_reserve(usize::MAX).is_err());
        let after = observation(&map);
        assert_eq!(map.get("live"), Some(&1));
        assert_eq!(
            after.conservative_table_high_water_charge_bytes(),
            before.max(after.current_table_allocation_bytes())
        );
    }

    #[test]
    fn std_and_pinned_hashbrown_preserve_iteration_parity_for_graph_operations() {
        type StdMap = std::collections::HashMap<String, u64, RandomState>;

        fn assert_parity(tracked: &TrackedStringMap<u64>, standard: &StdMap) {
            let tracked_entries: Vec<_> = tracked
                .iter()
                .map(|(key, value)| (key.as_str(), *value))
                .collect();
            let standard_entries: Vec<_> = standard
                .iter()
                .map(|(key, value)| (key.as_str(), *value))
                .collect();
            assert_eq!(tracked_entries, standard_entries);
        }

        // Production construction grows from a genuinely empty map. Check
        // every automatic growth boundary and duplicate-key replacement, not
        // just the explicit-reserve path below.
        {
            let automatic_hasher = RandomState::new();
            let mut automatic_tracked =
                TrackedStringMap::from_inner(InnerMap::with_hasher(automatic_hasher.clone()));
            let mut automatic_standard = StdMap::with_hasher(automatic_hasher);
            for index in 0_u64..97 {
                let key = format!("automatic-{index:03}");
                assert_eq!(
                    automatic_tracked.insert(key.clone(), index),
                    automatic_standard.insert(key, index)
                );
                assert_parity(&automatic_tracked, &automatic_standard);
            }
            for index in (0_u64..97).rev().step_by(4) {
                let key = format!("automatic-{index:03}");
                let replacement = index + 1_000;
                assert_eq!(
                    automatic_tracked.insert(key.clone(), replacement),
                    automatic_standard.insert(key, replacement)
                );
                assert_parity(&automatic_tracked, &automatic_standard);
            }
        }

        let hasher = RandomState::new();
        let mut tracked = TrackedStringMap::from_inner(InnerMap::with_hasher(hasher.clone()));
        let mut standard = StdMap::with_hasher(hasher);

        tracked.try_reserve(31).unwrap();
        standard.try_reserve(31).unwrap();
        for index in 0_u64..31 {
            let key = format!("node-{index:02}");
            tracked.insert(key.clone(), index);
            standard.insert(key, index);
        }
        assert_parity(&tracked, &standard);

        for index in (0_u64..31).step_by(3) {
            let key = format!("node-{index:02}");
            assert_eq!(tracked.remove(&key), standard.remove(&key));
        }
        assert_parity(&tracked, &standard);

        tracked.reserve(97);
        standard.reserve(97);
        assert_parity(&tracked, &standard);
        tracked.shrink_to(8);
        standard.shrink_to(8);
        assert_parity(&tracked, &standard);

        let tracked_clone = tracked.clone();
        let standard_clone = standard.clone();
        assert_parity(&tracked_clone, &standard_clone);

        tracked.clear();
        standard.clear();
        assert_parity(&tracked, &standard);
    }
}
