// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn index() -> SharedNameIndex {
    let mut map = TrackedStringMap::with_capacity(4096);
    for (name, value) in [("first", 0), ("second", 1), ("_input", 2)] {
        let mut key = String::with_capacity(1024 + name.len());
        key.push_str(name);
        map.insert(key, value);
    }
    SharedNameIndex::from_map(map)
}

#[test]
fn shared_index_256_clones_keep_the_exact_table_keys_and_allocation_facts() {
    let source = index();
    let clones: Vec<_> = (0..256).map(|_| source.clone()).collect();
    assert_eq!(Arc::strong_count(&source.inner), 257);
    for clone in &clones {
        assert_eq!(clone.owner_identity(), source.owner_identity());
        assert_eq!(
            clone.inner.allocation_size(),
            source.inner.allocation_size()
        );
        assert_eq!(
            clone.key_payload_capacity_bytes_v2(),
            source.key_payload_capacity_bytes_v2()
        );
        for (key, &value) in &source {
            let (same, &actual) = clone.get_key_value(key.as_str()).unwrap();
            assert_eq!(actual, value);
            assert_eq!(same.as_ptr(), key.as_ptr());
            assert_eq!(same.capacity(), key.capacity());
        }
    }
}

#[test]
fn shared_index_survives_source_drop_and_releases_the_one_owner_after_the_last_handle() {
    let source = index();
    let first = source.clone();
    let last = source.clone();
    let weak = Arc::downgrade(&source.inner);
    let identity = source.owner_identity();
    drop(source);
    drop(first);
    assert_eq!(last.owner_identity(), identity);
    assert_eq!(last.get("first"), Some(&0));
    assert_eq!(last.get("_input"), Some(&2));
    assert!(!last.contains_key("missing"));
    assert_eq!(weak.strong_count(), 1);
    drop(last);
    assert!(weak.upgrade().is_none());
}

#[test]
fn shared_index_clone_from_reuses_source_and_releases_the_replaced_owner() {
    let source = index();
    let mut target =
        SharedNameIndex::from_map(TrackedStringMap::from_entries([("old".into(), 99)]));
    let weak = Arc::downgrade(&target.inner);
    target.clone_from(&source);
    assert!(weak.upgrade().is_none());
    assert_eq!(target.owner_identity(), source.owner_identity());
    assert!(!target.contains_key("old"));
    assert_eq!(target.get("second"), Some(&1));
}

#[test]
fn shared_index_owner_quote_includes_the_map_body_and_two_arc_counters_once() {
    let source = index();
    let bytes = source.owner_allocation_request_bytes().unwrap();
    let (layout, offset) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<TrackedStringMap<usize>>())
        .unwrap();
    assert_eq!(bytes, layout.pad_to_align().size());
    assert!(bytes >= 2 * size_of::<usize>() + size_of::<TrackedStringMap<usize>>());
    assert_eq!(
        size_of::<SharedNameIndex>(),
        size_of::<Arc<TrackedStringMap<usize>>>()
    );
    assert_eq!(offset % align_of::<TrackedStringMap<usize>>(), 0);
    assert_eq!(
        source.key_payload_capacity_bytes_v2(),
        Some(source.keys().map(String::capacity).sum())
    );
}
