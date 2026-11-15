// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One immutable dispatch name index shared by its plan and backward users.
//! Cloning this handle never clones the hash table or any String key. The
//! private Arc has no production mutation or extraction API, so its capacity
//! facts remain valid for every holder until the last handle is dropped.

use super::TrackedStringMap;
#[cfg(test)]
use std::{alloc::Layout, sync::atomic::AtomicUsize};
use std::{fmt, ops::Deref, sync::Arc};

pub(crate) struct SharedNameIndex {
    inner: Arc<TrackedStringMap<usize>>,
}

impl SharedNameIndex {
    pub(crate) fn from_map(map: TrackedStringMap<usize>) -> Self {
        Self {
            inner: Arc::new(map),
        }
    }
    /// Process-local identity of the immutable shared map body. The table and
    /// String payloads have the same lifetime as this Arc allocation.
    #[cfg(test)]
    pub(crate) fn owner_identity(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
    /// Pinned Rust Arc allocation request: two counters followed by the tracked
    /// map's inline body. This excludes its separately quoted table/key payloads,
    /// allocator bookkeeping and RSS; the handle itself is inline in its caller.
    #[cfg(test)]
    pub(crate) fn owner_allocation_request_bytes(&self) -> Option<usize> {
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<TrackedStringMap<usize>>())
            .ok()?;
        Some(layout.pad_to_align().size())
    }
    #[cfg(test)]
    pub(crate) fn unique_map_mut_for_test(&mut self) -> &mut TrackedStringMap<usize> {
        Arc::get_mut(&mut self.inner).expect("test mutation needs the sole name-index owner")
    }
}
impl Clone for SharedNameIndex {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
    }
}
impl Deref for SharedNameIndex {
    type Target = TrackedStringMap<usize>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl fmt::Debug for SharedNameIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}
impl<'a> IntoIterator for &'a SharedNameIndex {
    type Item = (&'a String, &'a usize);
    type IntoIter = hashbrown::hash_map::Iter<'a, String, usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

#[cfg(test)]
#[path = "shared_name_index_tests.rs"]
mod tests;
