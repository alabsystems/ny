// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::beta_crown::state::{
    AlphaNeuronState, GraphAlphaStateRepresentation, GraphDomainAlphaState,
};

const PACKED_QUEUE_ENV: &str = "NY_PACKED_GRAPH_ALPHA_QUEUE";

fn alpha_state(alpha: f32) -> GraphDomainAlphaState {
    let lower = AlphaNeuronState {
        alpha,
        grad: -0.125,
        velocity: 0.25,
        adam_m: -0.375,
        adam_v: 0.5,
        adam_v_max: 0.625,
    };
    let upper = AlphaNeuronState {
        alpha: 1.0 - alpha,
        grad: 0.75,
        velocity: -0.875,
        adam_m: 0.0625,
        adam_v: 0.125,
        adam_v_max: 0.25,
    };
    let mut state = GraphDomainAlphaState::empty();
    state
        .neurons_mut()
        .entry("relu1".to_string())
        .or_default()
        .insert(1, lower);
    state
        .upper_neurons_mut()
        .entry("relu1".to_string())
        .or_default()
        .insert(1, upper);
    state
}

fn processed(lower_bound: f32, alpha: f32) -> ProcessedDomains {
    let mut processed = ProcessedDomains::valid_single_domain();
    processed.global_lbs[0] = lower_bound;
    processed.global_ubs[0] = lower_bound + 2.0;
    processed.metadata[0]
        .update_bounds(lower_bound, lower_bound + 2.0)
        .unwrap();
    processed.metadata[0].set_alpha_state(Some(alpha_state(alpha)));
    processed
}

fn assert_all_field_bits(state: &GraphDomainAlphaState, expected_alpha: f32) {
    let lower = state.neuron("relu1", 1).unwrap();
    let upper = state.upper_neurons()["relu1"].get(&1).unwrap();
    let expected_lower = alpha_state(expected_alpha);
    let expected_lower = expected_lower.neuron("relu1", 1).unwrap();
    let expected_upper_state = alpha_state(expected_alpha);
    let expected_upper = expected_upper_state.upper_neurons()["relu1"]
        .get(&1)
        .unwrap();
    for (actual, expected) in [
        (lower.alpha(), expected_lower.alpha()),
        (lower.grad(), expected_lower.grad()),
        (lower.velocity(), expected_lower.velocity()),
        (lower.adam_m(), expected_lower.adam_m()),
        (lower.adam_v(), expected_lower.adam_v()),
        (lower.adam_v_max(), expected_lower.adam_v_max()),
        (upper.alpha(), expected_upper.alpha()),
        (upper.grad(), expected_upper.grad()),
        (upper.velocity(), expected_upper.velocity()),
        (upper.adam_m(), expected_upper.adam_m()),
        (upper.adam_v(), expected_upper.adam_v()),
        (upper.adam_v_max(), expected_upper.adam_v_max()),
    ] {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn gate_off_keeps_runtime_state_unchanged() {
    crate::tests::with_serialized_env_vars_removed(&[PACKED_QUEUE_ENV], || {
        let mut list = DomainList::new(create_test_config()).unwrap();
        let processed = processed(-0.75, 0.375);
        let census_before = processed.metadata[0].alpha_state_byte_census();
        list.add(processed).unwrap();

        assert_eq!(
            list.metadata[0].alpha_state_representation(),
            Some(GraphAlphaStateRepresentation::Runtime)
        );
        assert_eq!(
            list.metadata[0].alpha_state_byte_census(),
            census_before,
            "gate-off enqueue must move, not rebuild, the runtime hash maps"
        );
        assert_all_field_bits(list.metadata[0].alpha_state().unwrap(), 0.375);

        let picked = list.pick_out(1).unwrap();
        assert_eq!(
            picked.metadata[0].alpha_state_representation(),
            Some(GraphAlphaStateRepresentation::Runtime)
        );
        assert_all_field_bits(picked.metadata[0].alpha_state().unwrap(), 0.375);
    });
}

#[test]
fn packed_clone_sort_pick_and_add_keep_all_rows_aligned() {
    crate::tests::with_serialized_env_vars(&[(PACKED_QUEUE_ENV, "1")], || {
        let mut list = DomainList::new(create_test_config()).unwrap();
        for (lower_bound, alpha) in [(-0.25, 0.25), (-0.75, 0.75), (-0.5, 0.5)] {
            list.add(processed(lower_bound, alpha)).unwrap();
        }

        assert!(list.metadata.iter().all(|metadata| {
            metadata.alpha_state_representation() == Some(GraphAlphaStateRepresentation::Packed)
        }));
        let cloned = list.metadata[1].clone();
        assert_eq!(
            cloned.alpha_state_representation(),
            Some(GraphAlphaStateRepresentation::Packed)
        );
        assert_eq!(
            cloned.alpha_state_byte_census(),
            list.metadata[1].alpha_state_byte_census()
        );

        list.sort_by_domain_priority(false).unwrap();
        let picked = list.pick_out(3).unwrap();
        for (idx, metadata) in picked.metadata.iter().enumerate() {
            assert_eq!(
                picked.global_lbs[idx].to_bits(),
                metadata.lower_bound().to_bits()
            );
            let expected_alpha = -metadata.lower_bound();
            assert_eq!(
                metadata.alpha_state_representation(),
                Some(GraphAlphaStateRepresentation::Runtime)
            );
            assert_all_field_bits(metadata.alpha_state().unwrap(), expected_alpha);
        }
    });
}

#[test]
fn corrupted_packed_metadata_is_refused_and_restored_to_queue() {
    crate::tests::with_serialized_env_vars(&[(PACKED_QUEUE_ENV, "1")], || {
        let mut list = DomainList::new(create_test_config()).unwrap();
        list.add(processed(-0.4, 0.4)).unwrap();
        assert_eq!(
            list.metadata[0].alpha_state_representation(),
            Some(GraphAlphaStateRepresentation::Packed)
        );
        list.metadata[0].corrupt_packed_alpha_layout_for_test();

        let error = list.pick_out(1).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("packed graph alpha queue state invalid"),
            "unexpected error: {error}"
        );
        assert_eq!(list.len(), 1, "failed dequeue must restore every queue row");
        assert_eq!(
            list.metadata[0].alpha_state_representation(),
            Some(GraphAlphaStateRepresentation::Packed)
        );
    });
}

#[test]
fn packed_metadata_from_another_graph_local_queue_is_refused_and_restored() {
    crate::tests::with_serialized_env_vars(&[(PACKED_QUEUE_ENV, "1")], || {
        let mut source = DomainList::new(create_test_config()).unwrap();
        let mut destination = DomainList::new(create_test_config()).unwrap();
        assert_ne!(
            source.alpha_queue_identity, destination.alpha_queue_identity,
            "every DomainList must receive a distinct graph-local identity"
        );

        source.add(processed(-0.2, 0.2)).unwrap();
        destination.add(processed(-0.8, 0.8)).unwrap();
        let destination_alpha = destination.metadata[0].alpha_state.clone();
        destination.metadata[0].alpha_state = source.metadata[0].alpha_state.clone();

        let error = destination.pick_out(1).unwrap_err();
        assert!(
            error.to_string().contains("queue identity"),
            "unexpected cross-queue error: {error}"
        );
        assert_eq!(
            destination.len(),
            1,
            "failed cross-queue dequeue must restore the row"
        );
        assert_eq!(
            destination.metadata[0].alpha_state_representation(),
            Some(GraphAlphaStateRepresentation::Packed)
        );

        // Put back the destination-owned packed alpha and prove the failed
        // dequeue restored every tensor/metadata row in alignment.
        destination.metadata[0].alpha_state = destination_alpha;
        let picked = destination.pick_out(1).unwrap();
        assert_eq!(picked.global_lbs[0].to_bits(), (-0.8f32).to_bits());
        assert_eq!(
            picked.metadata[0].lower_bound().to_bits(),
            (-0.8f32).to_bits()
        );
        assert_all_field_bits(picked.metadata[0].alpha_state().unwrap(), 0.8);
    });
}
