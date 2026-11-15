// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::amortise::NoPoll;
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};
use ndarray::Array2;
use std::collections::HashMap as ReferenceMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn key(value: &str, capacity: usize) -> String {
    let mut key = String::with_capacity(capacity);
    key.push_str(value);
    key
}
fn actual_keys<V>(map: &TrackedStringMap<V>) -> usize {
    map.keys().map(String::capacity).sum()
}
fn exact<V>(map: &TrackedStringMap<V>) {
    assert_eq!(map.key_payload_capacity_bytes_v2(), Some(actual_keys(map)));
    let table = map.allocation_fact_v1();
    assert_eq!(
        table.current_table_allocation_bytes(),
        map.allocation_size()
    );
    assert!(table.conservative_table_high_water_charge_bytes() >= map.allocation_size());
}
fn graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "affine",
        Layer::Linear(LinearLayer::new(Array2::eye(2), None).unwrap()),
    ));
    graph.set_output("affine");
    graph
}
fn plan() -> CrownDispatchPlan {
    CrownDispatchPlan::build(&graph()).unwrap()
}

#[test]
fn dispatch_key_capacity_retains_actual_occupied_key_and_matches_original_lookup_semantics() {
    let mut actual = TrackedStringMap::with_capacity(32);
    let mut reference = ReferenceMap::with_capacity(32);
    for (text, capacity, value) in [
        ("same", 128, 1),
        ("same", 8, 2),
        ("other", 256, 3),
        ("same", 512, 4),
    ] {
        let k = key(text, capacity);
        let retained = actual
            .keys()
            .find(|k| k.as_str() == text)
            .map(String::capacity);
        assert_eq!(
            actual.insert(k, value),
            reference.insert(text.to_owned(), value)
        );
        exact(&actual);
        if let Some(retained) = retained {
            assert_eq!(actual.get_key_value(text).unwrap().0.capacity(), retained);
        }
        for (k, &v) in &reference {
            assert_eq!(actual.get(k), Some(&v));
        }
    }
    assert_eq!(actual.len(), 2);
    let retained = actual.get_key_value("same").unwrap().0.capacity();
    assert_eq!(actual.remove("same"), reference.remove("same"));
    exact(&actual);
    assert!(retained > "same".len());
}

#[test]
fn dispatch_key_capacity_survives_every_map_growth_removal_value_update_and_shrink_surface() {
    let mut map = TrackedStringMap::new();
    exact(&map);
    map.reserve(128);
    map.try_reserve(64).unwrap();
    exact(&map);
    for i in 0..64 {
        map.insert(key(&format!("key{i}"), 128 + i), i);
        exact(&map);
    }
    for v in map.values_mut() {
        *v += 100;
    }
    *map.get_mut("key1").unwrap() = 9;
    exact(&map);
    for i in 0..16 {
        assert!(map.remove(&format!("key{i}")).is_some());
        exact(&map);
    }
    for i in 16..32 {
        let (removed, _) = map.remove_entry(&format!("key{i}")).unwrap();
        assert_ne!(removed, "");
        exact(&map);
    }
    map.retain(|k, _| k.ends_with('2') || k.ends_with('3'));
    exact(&map);
    map.shrink_to(4);
    map.shrink_to_fit();
    exact(&map);
    map.clear();
    exact(&map);
    assert_eq!(map.key_payload_capacity_bytes_v2(), Some(0));
    map.shrink_to_fit();
    exact(&map);
}

#[test]
fn dispatch_key_capacity_clone_clone_from_and_deserialize_observe_each_new_owner() {
    let source = TrackedStringMap::from_entries([(key("one", 1024), 1u64), (key("two", 2048), 2)]);
    exact(&source);
    let cloned = source.clone();
    exact(&cloned);
    assert_eq!(cloned.get("one"), source.get("one"));
    assert_ne!(
        cloned.get_key_value("one").unwrap().0.as_ptr(),
        source.get_key_value("one").unwrap().0.as_ptr()
    );
    let mut target = TrackedStringMap::with_capacity(512);
    target.insert(key("discard", 8192), 3);
    target.clone_from(&source);
    exact(&target);
    assert!(!target.contains_key("discard"));
    let encoded = serde_json::to_string(&source).unwrap();
    let decoded: TrackedStringMap<u64> = serde_json::from_str(&encoded).unwrap();
    exact(&decoded);
    assert!(!encoded.contains("key_payload_capacity"));
    for (k, &v) in &source {
        assert_eq!(decoded.get(k), Some(&v));
    }
}

#[test]
fn dispatch_key_capacity_unwinding_mutation_closes_receipt_until_owner_is_reobserved() {
    let mut plan = plan();
    let mut calls = 0;
    let result = catch_unwind(AssertUnwindSafe(|| {
        plan.name_to_idx.unique_map_mut_for_test().retain(|_, _| {
            calls += 1;
            assert!(calls != 2, "retained key census unwind");
            false
        })
    }));
    assert!(result.is_err());
    assert_eq!(plan.name_to_idx.key_payload_capacity_bytes_v2(), None);
    assert!(plan.allocation_receipt_v2(&mut NoPoll).is_err());
    let table = plan.name_to_idx.allocation_fact_v1();
    assert!(
        table.conservative_table_high_water_charge_bytes()
            >= table.current_table_allocation_bytes()
    );
    let replacement = TrackedStringMap::from_entries([(key("fresh", 128), 0usize)]);
    plan.name_to_idx = SharedNameIndex::from_map(replacement);
    exact(&plan.name_to_idx);
    assert!(plan.allocation_receipt_v2(&mut NoPoll).is_ok());
}

#[test]
fn dispatch_plan_receipt_quotes_all_original_spare_vector_string_and_table_capacities() {
    let mut plan = plan();
    plan.exec_order.reserve(101);
    plan.reverse_order.reserve(103);
    plan.idx_to_name.reserve(107);
    plan.routes.reserve(109);
    plan.name_to_idx.unique_map_mut_for_test().reserve(257);
    let name = &mut plan.idx_to_name[0];
    name.reserve(1024);
    let table = plan.name_to_idx.allocation_size();
    let keys = actual_keys(&plan.name_to_idx);
    let vecs = plan.exec_order.capacity() * size_of::<usize>()
        + plan.reverse_order.capacity() * size_of::<usize>()
        + plan.idx_to_name.capacity() * size_of::<String>()
        + plan.routes.capacity() * size_of::<DispatchRoute>();
    let strings: usize = plan.idx_to_name.iter().map(String::capacity).sum();
    let r = plan.allocation_receipt_v2(&mut NoPoll).unwrap();
    assert!(r.matches_source(&plan));
    assert_eq!(r.source_inline_bytes(), size_of::<CrownDispatchPlan>());
    assert_eq!(r.name_index_table().current_table_allocation_bytes(), table);
    assert_eq!(r.index_key_capacity_bytes(), keys);
    assert_eq!(r.vector_capacity_bytes(), vecs);
    assert_eq!(r.index_name_capacity_bytes(), strings);
    assert_eq!(
        r.heap_charge_bytes_excluding_inline(),
        table + keys + vecs + strings + plan.name_to_idx.owner_allocation_request_bytes().unwrap()
    );
    assert!(vecs > (plan.exec_order.len() + plan.reverse_order.len()) * size_of::<usize>());
}

struct Work {
    calls: usize,
    stop: usize,
    units: usize,
}
impl CooperativeWork for Work {
    fn checkpoint(&mut self) -> Result<()> {
        self.calls += 1;
        if self.calls == self.stop {
            Err(NyError::DeadlineExceeded(
                "dispatch artifact observer".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn before(&mut self, units: usize) -> Result<()> {
        assert!(units <= 4096);
        self.checkpoint()?;
        self.units += units;
        Ok(())
    }
}

#[test]
fn dispatch_plan_receipt_sparse_table_has_fixed_metadata_work_and_no_hidden_bucket_walk() {
    let mut plan = plan();
    plan.name_to_idx.unique_map_mut_for_test().reserve(1 << 16);
    plan.name_to_idx.unique_map_mut_for_test().clear();
    let mut work = Work {
        calls: 0,
        stop: 0,
        units: 0,
    };
    let r = plan.allocation_receipt_v2(&mut work).unwrap();
    assert_eq!(r.index_key_capacity_bytes(), 0);
    assert_eq!(work.units, 80);
    assert_eq!(work.calls, 5);
    assert!(r.name_index_table().current_table_allocation_bytes() > 1 << 16);
}

#[test]
fn dispatch_plan_receipt_every_refusal_retains_exact_original_artifact() {
    let plan = plan();
    let original = plan.clone();
    let before = plan
        .allocation_receipt_v2(&mut NoPoll)
        .unwrap()
        .heap_charge_bytes_excluding_inline();
    let mut baseline = Work {
        calls: 0,
        stop: 0,
        units: 0,
    };
    let repeated = plan.allocation_receipt_v2(&mut baseline).unwrap();
    assert_eq!(repeated.heap_charge_bytes_excluding_inline(), before);
    assert_eq!(baseline.units, 80);
    for stop in 1..=baseline.calls {
        let mut work = Work {
            calls: 0,
            stop,
            units: 0,
        };
        assert!(matches!(plan.allocation_receipt_v2(&mut work),Err(e)if e.is_deadline_exceeded()));
        assert_eq!(work.calls, stop);
        assert_eq!(plan.exec_order, original.exec_order);
        assert_eq!(plan.reverse_order, original.reverse_order);
        assert_eq!(plan.idx_to_name, original.idx_to_name);
        assert_eq!(plan.routes, original.routes);
        assert_eq!(
            plan.allocation_receipt_v2(&mut NoPoll)
                .unwrap()
                .heap_charge_bytes_excluding_inline(),
            before
        );
    }
}

#[test]
fn dispatch_plan_receipt_binds_exact_plan_and_clone_has_distinct_owner_facts() {
    let plan = plan();
    let clone = plan.clone();
    let r = plan.allocation_receipt_v2(&mut NoPoll).unwrap();
    assert!(r.matches_source(&plan));
    assert!(!r.matches_source(&clone));
    let other = clone.allocation_receipt_v2(&mut NoPoll).unwrap();
    assert!(other.matches_source(&clone));
    assert_eq!(
        r.name_index_owner_identity(),
        other.name_index_owner_identity()
    );
    assert_eq!(
        r.name_index_owner_allocation_bytes(),
        plan.name_to_idx.owner_allocation_request_bytes().unwrap()
    );
    assert_eq!(r.index_key_capacity_bytes(), actual_keys(&plan.name_to_idx));
    assert_eq!(
        other.index_key_capacity_bytes(),
        actual_keys(&clone.name_to_idx)
    );
    for (name, &index) in &plan.name_to_idx {
        assert_eq!(clone.name_to_idx.get(name), Some(&index));
    }
}
