// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;

#[test]
fn decoder_n9_original_block_observation_captures_every_stage_once_and_matches_full_graph() {
    let options = DecoderOptions::default();
    let model = model(false, &[0, 1, 2]);
    let input = model.embedded_tokens(&[1, 3, 4], &options).unwrap();
    let raw0 = model
        .graph_for("block.0", &options)
        .unwrap()
        .point(input.values(), &options)
        .unwrap();
    let ordinary = model
        .graph_for("block.1", &options)
        .unwrap()
        .point(input.values(), &options)
        .unwrap();
    let observer = model.observe_block(1, &options).unwrap();
    assert!(!observer
        .graph()
        .node_names()
        .iter()
        .any(|n| n.starts_with("decoder.block.0") || n == "decoder.logits"));
    assert_eq!(
        observer.graph().num_nodes(),
        observer.boundary().outputs.len()
    );
    assert!(!observer.boundary().certificate_authority);
    assert_eq!(
        observer.boundary().original_lowering_digest,
        model.manifest().lowering_digest
    );
    assert_eq!(observer.boundary().input_tensor, model.block_outputs()[0]);
    let budget = model.budget();
    let before = budget.receipt().unwrap();
    let observation = observer.point(raw0.values(), &options).unwrap();
    let after = budget.receipt().unwrap();
    assert_eq!(
        after.admitted - before.admitted,
        1,
        "one execution, not one per alias"
    );
    assert_eq!(after.completed - before.completed, 1);
    assert_eq!(after.proposals_reserved - before.proposals_reserved, 1);
    assert_eq!(
        after.work_reserved - before.work_reserved,
        observer.point_work_bound()
    );
    assert_eq!(
        after.live_persistent_bytes - before.live_persistent_bytes,
        observation.reserved_bytes()
    );
    assert_eq!(
        observation.tensors().len(),
        observer.boundary().outputs.len()
    );
    assert_eq!(
        observation.execution_trace().steps().len(),
        observer.graph().num_nodes()
    );
    for (expected, actual) in observer
        .boundary()
        .outputs
        .iter()
        .zip(observation.tensors())
    {
        assert_eq!(expected.node, actual.identity().node);
        assert_eq!(expected.tensor, actual.identity().tensor);
        assert!(actual.values().iter().all(|v| v.is_finite()));
        assert_eq!(
            observation
                .execution_trace()
                .steps()
                .iter()
                .filter(|s| s.node == expected.node)
                .count(),
            1
        );
    }
    assert_eq!(
        observation.tensors().last().unwrap().values(),
        ordinary.values()
    );
    drop(observation);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
}

#[test]
fn decoder_n9_observation_identity_binds_original_subset_positions_and_same_input_bits() {
    let options = DecoderOptions::default();
    let model = model(false, &[0, 1]);
    let input = model.embedded_tokens(&[1, 3], &options).unwrap();
    let observer = model.observe_block(0, &options).unwrap();
    let point = observer.point(input.values(), &options).unwrap();
    let repeated = observer.point(input.values(), &options).unwrap();
    assert_eq!(point.execution_digest(), repeated.execution_digest());
    let mut changed = input.values().clone();
    changed[[0, 0]] = f32::from_bits(changed[[0, 0]].to_bits() ^ 1);
    let changed = observer.point(&changed, &options).unwrap();
    assert_ne!(point.input_digest(), changed.input_digest());
    assert_ne!(point.execution_digest(), changed.execution_digest());
    let next = model.observe_block(1, &options).unwrap();
    assert_ne!(
        observer.boundary().subset_digest,
        next.boundary().subset_digest
    );
    let moved = super::model(false, &[4, 5])
        .observe_block(0, &options)
        .unwrap();
    assert_ne!(
        observer.boundary().subset_digest,
        moved.boundary().subset_digest
    );
}

#[test]
fn decoder_n9_observation_refuses_shape_nonfinite_caps_deadline_and_changed_original_specs() {
    let options = DecoderOptions::default();
    let model = model(false, &[0, 1]);
    assert!(model.observe_block(2, &options).is_err());
    let observer = model.observe_block(0, &options).unwrap();
    let wrong = ArrayD::zeros(IxDyn(&[2, 7]));
    assert!(matches!(
        observer.point(&wrong, &options),
        Err(DecoderContractRefusal::Tensor(_))
    ));
    let input = model.embedded_tokens(&[1, 3], &options).unwrap();
    let mut poison = input.values().clone();
    poison[[0, 0]] = f32::NAN;
    let before = model.budget().receipt().unwrap();
    assert!(observer.point(&poison, &options).is_err());
    let after = model.budget().receipt().unwrap();
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    let small = DecoderOptions {
        max_scratch_bytes: 0,
        ..options.clone()
    };
    assert!(matches!(
        observer.point(input.values(), &small),
        Err(DecoderContractRefusal::Resource(_))
    ));
    let expired = DecoderOptions {
        deadline: Some(Instant::now()),
        ..options.clone()
    };
    assert!(matches!(
        observer.point(input.values(), &expired),
        Err(DecoderContractRefusal::Deadline)
    ));
    let mut changed = super::model(false, &[0, 1]);
    changed.graph_model.network.layers[0]
        .name
        .push_str("changed");
    assert!(changed.observe_block(0, &options).is_err());
}
