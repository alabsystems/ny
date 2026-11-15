// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete thread-local allocator requests for an owned original chain report.
//! Fixed external ledger allocation is outside the measurement. Any ledger
//! capacity surviving report drop is recorded separately, never called freed.
#[path = "support/certificate_allocations.rs"]
mod allocations;
use allocations::{measure, snapshot};
use ndarray::{array, Array1, Array2};
use ny_propagate::certify::scope::Digest;
use ny_propagate::certify::work::{LeaseClass, WorkBudget};
use ny_propagate::layers::{
    GELULayer, GeluApproximation, Layer, PointLinearBuildPolicy, PointLinearLayer, ReLULayer,
    SiLULayer,
};
use ny_propagate::network::original_affine_chain::{
    OriginalChainNode, PreparedOriginalAffineChain,
};
use ny_propagate::network::original_affine_proof::{OriginalAffineIds, OriginalAffineOptions};
use ny_tensor::BoundedTensor;
use std::time::Duration;

fn options(budget: &WorkBudget) -> OriginalAffineOptions<'_> {
    OriginalAffineOptions {
        budget,
        class: LeaseClass::Optional,
        deadline: None,
    }
}
fn owner(w: &Array2<f32>, b: &Array1<f32>, budget: &WorkBudget) -> Layer {
    Layer::PointLinear(
        PointLinearLayer::from_original_view(
            w.view(),
            false,
            Some(b.view().into_dyn()),
            PointLinearBuildPolicy {
                budget,
                deadline: None,
                max_resident_bytes: 1 << 26,
                max_scratch_bytes: 1 << 24,
                max_work: 1 << 32,
            },
        )
        .unwrap(),
    )
}
fn ids(index: usize) -> OriginalAffineIds {
    let digest = |prefix: &[u8], i: usize| Digest::of_bytes(prefix, &i.to_le_bytes());
    OriginalAffineIds {
        lowering: Digest::of_bytes(b"test-chain-lowering", b"ordered original chain"),
        node: digest(b"test-chain-node", index),
        input: digest(b"test-chain-edge", index),
        output: digest(b"test-chain-edge", index + 1),
    }
}
#[test]
fn original_chain_complete_retained_allocations_clone_and_last_alias_observation() {
    for kind in ["affine", "relu", "gelu", "silu", "tied"] {
        let budget =
            WorkBudget::new(None, 1 << 24, 1 << 26, 1 << 32, 4096, Duration::ZERO).unwrap();
        let (row, final_counts) = measure(|| {
            let report = {
                let mut layers = vec![
                    owner(
                        &array![[0.5, -1.0], [0.1, 0.3]],
                        &array![0.125, -0.25],
                        &budget,
                    ),
                    Layer::ReLU(ReLULayer::new()),
                    owner(
                        &array![[1.5, -0.5], [0.1, 0.3], [0.1, 0.3]],
                        &array![-0.125, 0.25, 0.25],
                        &budget,
                    ),
                ];
                match kind {
                    "affine" => {
                        layers.remove(1);
                    }
                    "gelu" => {
                        layers[1] = Layer::GELU(GELULayer::new(GeluApproximation::Erf));
                    }
                    "silu" => {
                        layers[1] = Layer::SiLU(SiLULayer::new());
                    }
                    "tied" => {
                        layers[2] = layers[0].clone();
                    }
                    _ => {}
                }
                let nodes: Vec<_> = layers
                    .iter()
                    .enumerate()
                    .map(|(index, layer)| OriginalChainNode {
                        ids: ids(index),
                        layer,
                    })
                    .collect();
                let context =
                    PreparedOriginalAffineChain::prepare(&nodes, options(&budget)).unwrap();
                let input = BoundedTensor::new(
                    array![[-1.0, -0.5], [-0.25, 0.25]].into_dyn(),
                    array![[1.0, 0.75], [0.5, 1.0]].into_dyn(),
                )
                .unwrap();
                context.forward(&input, options(&budget)).unwrap()
            };
            // Caller arrays, nodes, context, input and model handles are gone.
            let held = snapshot().unwrap();
            assert!(!held.overflowed);
            let scope = std::ptr::from_ref(report.scope());
            let chain_owner_request = held.allocation_containing(scope.addr()).unwrap().bytes;
            let digest = report.binding().output;
            let model = report.scope().model;
            let nodes = report.num_nodes();
            let report_bytes = size_of_val(&report);
            let expected = report
                .reserved_bytes()
                .checked_add(report.original_owner_bytes())
                .unwrap();
            let held_receipt = budget.receipt().unwrap();
            assert_eq!(held_receipt.live_persistent_bytes, expected);
            assert_eq!(held_receipt.live_scratch_bytes, 0);
            let alias = report.clone();
            assert_eq!(snapshot().unwrap(), held, "clone must not allocate");
            assert!(std::ptr::eq(alias.scope(), scope));
            assert_eq!(
                alias.bounds().lower().as_ptr(),
                report.bounds().lower().as_ptr()
            );
            drop(report);
            assert_eq!(
                snapshot().unwrap(),
                held,
                "first alias drop cannot free owner data"
            );
            assert_eq!(budget.receipt().unwrap().live_persistent_bytes, expected);
            drop(alias);
            let released = snapshot().unwrap();
            let final_receipt = budget.receipt().unwrap();
            assert!(!released.overflowed);
            assert_eq!(
                released.allocations, held.allocations,
                "release must not allocate"
            );
            assert_eq!(released.reallocations, held.reallocations);
            assert_eq!(released.requested_bytes, held.requested_bytes);
            assert_eq!(final_receipt.live_persistent_bytes, 0);
            assert_eq!(final_receipt.live_scratch_bytes, 0);
            assert_eq!(final_receipt.work_reserved, held_receipt.work_reserved);
            assert_eq!(
                final_receipt.proposals_reserved,
                held_receipt.proposals_reserved
            );
            (
                held,
                released,
                held_receipt,
                final_receipt,
                chain_owner_request,
                report_bytes,
                digest,
                model,
                nodes,
            )
        })
        .unwrap();
        let (
            held,
            released,
            receipt,
            final_receipt,
            chain_owner_request,
            report_bytes,
            output,
            model,
            nodes,
        ) = row;
        assert_eq!(final_counts, released);
        let held_live = held.live_requested_bytes().unwrap();
        let residual_live = released.live_requested_bytes().unwrap();
        let report_graph_live = held_live.checked_sub(residual_live).unwrap();
        println!(
            "{{\"phase\":\"original_chain_{kind}\",\"nodes\":{nodes},\"report_bytes\":{report_bytes},\"chain_owner_request_bytes\":{chain_owner_request},\"allocations\":{},\"reallocations\":{},\"held_deallocations\":{},\"final_deallocations\":{},\"requested_bytes\":{},\"held_live_allocations\":{},\"residual_live_allocations\":{},\"held_live_requested_bytes\":{held_live},\"residual_live_requested_bytes\":{residual_live},\"report_graph_live_requested_bytes\":{report_graph_live},\"work\":{},\"proposals\":{},\"held_ledger_bytes\":{},\"peak_leased_bytes\":{},\"released_ledger_bytes\":{},\"model\":{:?},\"output\":{:?}}}",
            held.allocations, held.reallocations, held.deallocations, released.deallocations,
            held.requested_bytes, held.live_allocations(), released.live_allocations(),
            receipt.work_reserved, receipt.proposals_reserved, receipt.live_persistent_bytes,
            receipt.peak_leased_bytes, final_receipt.live_persistent_bytes, model.0, output.0,
        );
    }
}
