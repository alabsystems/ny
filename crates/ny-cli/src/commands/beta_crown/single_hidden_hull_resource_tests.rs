// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_onnx::vnnlib::parse_vnnlib;
use ny_propagate::certify::residency::{CancellationToken, MemoryAuthority};
use ny_propagate::certify::work::{WorkLimits, WorkReserve};
use ny_propagate::layers::{AddConstantLayer, AddLayer, LinearLayer, ReLULayer};

#[cfg(feature = "external-vnncomp")]
#[path = "single_hidden_hull_resource_original_probe.rs"]
mod original_probe;

fn budget(
    memory: &MemoryAuthority,
    cancel: &CancellationToken,
    deadline: Option<Instant>,
    proposals: u64,
) -> WorkBudget {
    WorkBudget::new_in(
        memory,
        cancel,
        WorkLimits {
            deadline,
            scratch_bytes: 1 << 20,
            persistent_bytes: 1 << 20,
            max_work: 1 << 30,
            max_proposals: proposals,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap()
}
fn fixture(constant: f32) -> (Network, BoundedTensor, VnnLibSpec) {
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap(),
    ));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0_f32], [0.0]]), Some(arr1(&[constant, 0.0]))).unwrap(),
    ));
    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let spec=parse_vnnlib("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n(assert (>= X_0 -1))\n(assert (<= X_0 1))\n(assert (<= Y_0 Y_1))").unwrap();
    (net, input, spec)
}

fn five_node_fixture(first_bias: ArrayD<f32>, last_bias: ArrayD<f32>) -> Network {
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), None).unwrap(),
    ));
    net.add_layer(Layer::AddConstant(AddConstantLayer::new(first_bias)));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0_f32, -0.5], [0.0, -0.0]]), None).unwrap(),
    ));
    net.add_layer(Layer::AddConstant(AddConstantLayer::new(last_bias)));
    net
}

#[test]
fn hull_resource_cli_original_five_node_signed_biases_are_borrowed_and_replayed() {
    let net = five_node_fixture(
        arr1(&[0.25_f32, -0.125]).into_dyn(),
        arr2(&[[3.0_f32, -0.25]]).into_dyn(),
    );
    let (_, _, spec) = fixture(2.0);
    let input = BoundedTensor::new(
        arr2(&[[-1.0_f32]]).into_dyn(),
        arr2(&[[1.0_f32]]).into_dyn(),
    )
    .unwrap();
    let objective = single_relational_margin(&spec).unwrap();
    let originals = data(&net, &input, &spec, &objective).unwrap();
    let [_, Layer::AddConstant(first), _, _, Layer::AddConstant(last)] = net.layers() else {
        unreachable!()
    };
    let HullBiasData::Binary32(b1) = originals.b1 else {
        panic!("original F32 first bias required")
    };
    let HullBiasData::Binary32(b2) = originals.b2 else {
        panic!("original F32 last bias required")
    };
    assert_eq!(b1.as_ptr(), first.constant().as_ptr());
    assert_eq!(b2.as_ptr(), last.constant().as_ptr());
    assert_eq!(
        b1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        vec![0.25_f32.to_bits(), (-0.125_f32).to_bits()]
    );
    assert_eq!(
        b2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        vec![3.0_f32.to_bits(), (-0.25_f32).to_bits()]
    );
    let memory = MemoryAuthority::new(1 << 20);
    let ledger = budget(&memory, &CancellationToken::default(), None, 2);
    let context = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
    let checked = try_single_hidden_hull_root_in(
        SingleHiddenHullMode::Root,
        &net,
        &input,
        &spec,
        &context,
        Instant::now(),
    )
    .unwrap();
    assert_eq!(checked.proof.origin(), HullProofOrigin::OriginalReplay);
    assert!(checked.proof.lower_bound() > 0.0);
    // The original piecewise linear function is minimized at one of these
    // endpoints or exact authored ReLU breakpoints. All terms are dyadic.
    for x in [-1.0_f64, -0.25, -0.125, 1.0] {
        let value = 3.25 - (x + 0.25).max(0.0) - 0.5 * (-x - 0.125).max(0.0);
        assert!(checked.proof.lower_bound() <= value);
    }
    let mut equivalent = Network::new();
    equivalent.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), Some(arr1(&[0.25, -0.125]))).unwrap(),
    ));
    equivalent.add_layer(Layer::ReLU(ReLULayer));
    equivalent.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[-1.0_f32, -0.5], [0.0, -0.0]]),
            Some(arr1(&[3.0, -0.25])),
        )
        .unwrap(),
    ));
    let equivalent_owner = import_hull_original(
        data(&equivalent, &input, &spec, &objective).unwrap(),
        &context,
    )
    .unwrap();
    assert_eq!(checked.proof.scope(), equivalent_owner.scope());
    drop(equivalent_owner);
    let clone = checked.proof.clone();
    drop(checked);
    assert!(ledger.receipt().unwrap().live_persistent_bytes > 0);
    drop(clone);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn hull_resource_cli_original_five_node_broadcast_and_profile_refusals_precede_import() {
    let (_, input, spec) = fixture(2.0);
    let objective = single_relational_margin(&spec).unwrap();
    for first_bias in [
        ArrayD::from_elem(IxDyn(&[]), 0.25_f32),
        arr1(&[0.25_f32]).into_dyn(),
        arr2(&[[0.25_f32, -0.125], [0.25, -0.125]]).into_dyn(),
        arr2(&[[0.25_f32, -0.125]]).into_dyn(),
    ] {
        let net = five_node_fixture(first_bias, arr1(&[3.0_f32, -0.25]).into_dyn());
        assert_eq!(
            data(&net, &input, &spec, &objective).err(),
            Some("not_one_original_bias_vector")
        );
    }
    let valid = five_node_fixture(
        arr1(&[0.25_f32, -0.125]).into_dyn(),
        arr1(&[3.0_f32, -0.25]).into_dyn(),
    );
    let mut biased = Network::new();
    biased.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    for layer in &valid.layers()[1..] {
        biased.add_layer(layer.clone());
    }
    assert_eq!(
        data(&biased, &input, &spec, &objective).err(),
        Some("original_bias_already_present")
    );
    let mut dynamic = Network::new();
    for (i, layer) in valid.layers().iter().enumerate() {
        dynamic.add_layer(if i == 1 {
            Layer::Add(AddLayer)
        } else {
            layer.clone()
        });
    }
    assert_eq!(
        data(&dynamic, &input, &spec, &objective).err(),
        Some("unpriced_canonicalization")
    );
    let batched = BoundedTensor::new(
        arr2(&[[-1.0_f32], [-1.0]]).into_dyn(),
        arr2(&[[1.0_f32], [1.0]]).into_dyn(),
    )
    .unwrap();
    assert_eq!(
        data(&valid, &batched, &spec, &objective).err(),
        Some("not_one_original_input_vector")
    );
}

#[test]
fn hull_resource_cli_original_five_node_cancel_deadline_and_memory_refuse_without_copy() {
    let net = five_node_fixture(
        arr1(&[0.25_f32, -0.125]).into_dyn(),
        arr1(&[3.0_f32, -0.25]).into_dyn(),
    );
    let (_, input, spec) = fixture(2.0);
    for (bytes, expired, stopped) in [
        (1_usize, false, false),
        (1 << 20, true, false),
        (1 << 20, false, true),
    ] {
        let memory = MemoryAuthority::new(bytes);
        let cancel = CancellationToken::default();
        if stopped {
            cancel.cancel();
        }
        let ledger = budget(
            &memory,
            &cancel,
            expired.then(|| {
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("test clock supports an expired deadline")
            }),
            1,
        );
        let context = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
        let result = try_single_hidden_hull_root_in(
            SingleHiddenHullMode::Root,
            &net,
            &input,
            &spec,
            &context,
            Instant::now(),
        );
        assert_eq!(
            result.err(),
            Some(if bytes == 1 {
                "original_import_refused"
            } else {
                "resource_refused"
            })
        );
        let receipt = ledger.receipt().unwrap();
        assert_eq!(receipt.work_reserved, 0);
        assert_eq!(receipt.live_persistent_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    }
}

#[test]
fn hull_resource_cli_existing_ledger_replays_original_and_retains_until_last_clone() {
    let memory = MemoryAuthority::new(1 << 20);
    let cancel = CancellationToken::default();
    let ledger = budget(
        &memory,
        &cancel,
        Some(Instant::now() + Duration::from_secs(10)),
        1,
    );
    let context = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
    let (net, input, spec) = fixture(2.0);
    let checked = try_single_hidden_hull_root_in(
        SingleHiddenHullMode::Root,
        &net,
        &input,
        &spec,
        &context,
        Instant::now(),
    )
    .unwrap();
    assert!(matches!(
        checked.result.result,
        BabVerificationStatus::Verified
    ));
    assert_eq!(checked.proof.origin(), HullProofOrigin::OriginalReplay);
    assert!(checked.proof.lower_bound() > 0.0 && checked.proof.lower_bound() <= 1.0);
    let receipt = ledger.receipt().unwrap();
    assert_eq!(
        (receipt.attempted, receipt.admitted, receipt.completed),
        (3, 3, 3)
    );
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert!(receipt.live_persistent_bytes > 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        receipt.live_persistent_bytes
    );
    let clone = checked.proof.clone();
    drop(checked);
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        receipt.live_persistent_bytes
    );
    drop(clone);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(
        ledger.receipt().unwrap().work_reserved,
        receipt.work_reserved
    );
}

#[test]
fn hull_resource_cli_off_bab_tie_sat_decline_without_unpriced_lp_or_false_verified() {
    let memory = MemoryAuthority::new(1 << 20);
    let cancel = CancellationToken::default();
    let ledger = budget(&memory, &cancel, None, 4);
    let context = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
    let (net, input, spec) = fixture(2.0);
    for mode in [SingleHiddenHullMode::Off, SingleHiddenHullMode::Bab] {
        assert!(matches!(
            try_single_hidden_hull_root_in(mode, &net, &input, &spec, &context, Instant::now()),
            Err("unsupported_budgeted_mode")
        ));
    }
    assert_eq!(ledger.receipt().unwrap().attempted, 0);
    for constant in [1.0_f32, 0.5] {
        let (net, input, spec) = fixture(constant);
        assert!(matches!(
            try_single_hidden_hull_root_in(
                SingleHiddenHullMode::Root,
                &net,
                &input,
                &spec,
                &context,
                Instant::now()
            ),
            Err("unpriced_lp_or_bab")
        ));
    }
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.attempted, 4);
    assert_eq!(receipt.proposals_reserved, 2);
    assert_eq!(
        (receipt.live_scratch_bytes, receipt.live_persistent_bytes),
        (0, 0)
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise cloned query contexts sharing the original budget rather than replacing its owner."
)]
fn hull_resource_cli_foreign_authority_and_cloned_query_cannot_refill_original_budget() {
    let memory = MemoryAuthority::new(1 << 20);
    let cancel = CancellationToken::default();
    let ledger = budget(&memory, &cancel, None, 1);
    let context = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
    let (net, input, spec) = fixture(2.0);
    let objective = single_relational_margin(&spec).unwrap();
    let owner =
        import_hull_original(data(&net, &input, &spec, &objective).unwrap(), &context).unwrap();
    let certificate = checked_hull_triangle(&owner, &context.clone())
        .unwrap()
        .certificate;
    let foreign_ledger = budget(&memory, &cancel, None, 1);
    let foreign =
        hull_context_for_budget(&foreign_ledger, hull_root_deadline(Instant::now(), None));
    assert!(matches!(
        replay_hull_original(&certificate, &foreign),
        Err(HullQueryRefusal::ForeignAuthority)
    ));
    assert_eq!(foreign_ledger.receipt().unwrap().attempted, 0);
    assert!(matches!(
        checked_hull_triangle(&owner, &context.clone()),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Work))
    ));
    let receipt = ledger.receipt().unwrap();
    assert_eq!(receipt.proposals_reserved, 1);
    assert_eq!(receipt.refused, 1);
}

#[test]
fn hull_resource_cli_shape_spec_memory_cancel_and_expired_deadline_refuse_before_copy() {
    let memory = MemoryAuthority::new(1);
    let cancel = CancellationToken::default();
    let ledger = budget(&memory, &cancel, None, 4);
    let context = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
    let (net, input, mut spec) = fixture(2.0);
    spec.output_constraints
        .push(OutputConstraint::GreaterEq(0, 1));
    assert!(matches!(
        try_single_hidden_hull_root_in(
            SingleHiddenHullMode::Root,
            &net,
            &input,
            &spec,
            &context,
            Instant::now()
        ),
        Err("not_one_relational_row")
    ));
    assert_eq!(ledger.receipt().unwrap().attempted, 0);
    let (net, input, spec) = fixture(2.0);
    assert!(matches!(
        try_single_hidden_hull_root_in(
            SingleHiddenHullMode::Root,
            &net,
            &input,
            &spec,
            &context,
            Instant::now()
        ),
        Err("original_import_refused")
    ));
    assert_eq!(
        (
            ledger.receipt().unwrap().work_reserved,
            ledger.receipt().unwrap().live_persistent_bytes
        ),
        (0, 0)
    );
    cancel.cancel();
    assert!(matches!(
        try_single_hidden_hull_root_in(
            SingleHiddenHullMode::Root,
            &net,
            &input,
            &spec,
            &context,
            Instant::now()
        ),
        Err("resource_refused")
    ));
    assert_eq!(ledger.receipt().unwrap().attempted, 1);
    let memory = MemoryAuthority::new(1 << 20);
    let ledger = budget(
        &memory,
        &CancellationToken::default(),
        Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline"),
        ),
        4,
    );
    let expired = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
    assert!(matches!(
        try_single_hidden_hull_root_in(
            SingleHiddenHullMode::Root,
            &net,
            &input,
            &spec,
            &expired,
            Instant::now()
        ),
        Err("resource_refused")
    ));
    assert_eq!(ledger.receipt().unwrap().attempted, 0);
}

#[test]
fn hull_resource_cli_cancellation_between_query_and_replay_does_not_publish_or_refresh() {
    let memory = MemoryAuthority::new(1 << 20);
    let cancel = CancellationToken::default();
    let ledger = budget(&memory, &cancel, None, 1);
    let context = hull_context_for_budget(&ledger, hull_root_deadline(Instant::now(), None));
    let (net, input, spec) = fixture(2.0);
    let objective = single_relational_margin(&spec).unwrap();
    let owner =
        import_hull_original(data(&net, &input, &spec, &objective).unwrap(), &context).unwrap();
    let certificate = checked_hull_triangle(&owner, &context).unwrap().certificate;
    let before = ledger.receipt().unwrap();
    cancel.cancel();
    assert!(matches!(
        replay_hull_original(&certificate, &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Cancelled))
    ));
    assert_eq!(ledger.receipt().unwrap(), before);
    assert_eq!(certificate.origin(), HullProofOrigin::CheckedQuery);
    drop(owner);
    drop(certificate);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(
        ledger.receipt().unwrap().work_reserved,
        before.work_reserved
    );
}
