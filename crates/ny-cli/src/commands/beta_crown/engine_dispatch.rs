// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared engine/backend dispatch for graph verification.
//!
//! Provides a single dispatch function for the three graph BaB execution paths
//! (GPU BaB, ReLU split, input split) that both `beta_crown verify` and
//! `bench_acasxu` can call with the appropriate engine. This eliminates
//! the hardcoded `engine: None` in the benchmark path.
//!
//! Part of #1881: CLI verification semantics unification.

use anyhow::Result;
use ny_core::GemmEngine;
use ny_propagate::{BetaCrownResult, BetaCrownVerifier, GraphNetwork, GraphPrecomputedBounds};
use ny_tensor::BoundedTensor;

/// Dispatch a single-objective graph constraint verification through the
/// appropriate BaB execution path.
///
/// Dispatch order:
/// 1. `gpu_bab` → `verify_graph_gpu_domain_list` (GPU-accelerated BaB via DomainList)
///    Applies to ALL branching modes (ReLU split and input split). The DomainList
///    engine handles branching-mode dispatch internally via `is_input_split_mode`.
/// 2. `use_relu_split` with precomputed bounds → `verify_graph_relu_split_with_bounds_with_engine`
/// 3. `use_relu_split` without precomputed bounds → `verify_graph_relu_split_with_engine_gpu`
/// 4. `!use_relu_split` → `verify_graph_input_split_with_engine`
///
/// Both `verify.rs` and `bench_acasxu.rs` should call this instead of
/// directly calling the verifier methods, ensuring consistent engine wiring.
///
/// Part of #3870: the `gpu_bab` check was previously nested under `use_relu_split`,
/// which blocked `InputSplit + gpu_bab` from reaching the DomainList engine.
// Justification: This dispatch function forwards the exact parameter set needed
// by the underlying verifier methods — model, bounds, spec, execution mode,
// and engine are all independent inputs that cannot be grouped without
// adding indirection that obscures the verification semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_graph_constraint(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_coeffs: &[f32],
    threshold: f32,
    use_relu_split: bool,
    gpu_bab: bool,
    precomputed_bounds: Option<&GraphPrecomputedBounds<'_>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
) -> Result<BetaCrownResult> {
    // gpu_bab takes priority: route ALL branching modes through DomainList.
    // The DomainList BaB loop checks `is_input_split_mode` internally and
    // dispatches to `process_input_split_batch` for InputSplit branching.
    if gpu_bab {
        return Ok(verifier.verify_graph_gpu_domain_list(
            graph,
            input,
            spec_coeffs,
            threshold,
            engine,
            deadline,
        )?);
    }

    if use_relu_split {
        if let Some(bounds) = precomputed_bounds {
            Ok(verifier.verify_graph_relu_split_with_bounds_with_engine(
                graph,
                input,
                spec_coeffs,
                threshold,
                bounds,
                engine,
                deadline,
            )?)
        } else {
            Ok(verifier.verify_graph_relu_split_with_engine_gpu(
                graph,
                input,
                spec_coeffs,
                threshold,
                engine,
                deadline,
            )?)
        }
    } else {
        Ok(verifier.verify_graph_input_split_with_engine(
            graph,
            input,
            spec_coeffs,
            threshold,
            engine,
            deadline,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::dispatch_graph_constraint;
    use ndarray::{arr1, arr2};
    use ny_propagate::{
        beta_crown::{BetaCrownConfig, BranchingHeuristic},
        layers::LinearLayer,
        BetaCrownVerifier, GraphNetwork, Layer, Network,
    };
    use ny_tensor::BoundedTensor;

    fn build_single_output_graph_3870() -> (GraphNetwork, BoundedTensor) {
        let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap();
        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));

        let graph = GraphNetwork::from_sequential(&network).unwrap();
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        (graph, input)
    }

    #[test]
    fn gpu_bab_input_split_dispatch_uses_domain_list_path_3870() {
        let (graph, input) = build_single_output_graph_3870();
        let config = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            max_domains: 0,
            timeout: Duration::from_secs(1),
            use_alpha_crown: false,
            use_crown_ibp: true,
            enable_cuts: false,
            batch_size: 1,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);

        let cpu_only_err = verifier
            .verify_graph_input_split_with_engine(&graph, &input, &[1.0_f32], 0.0, None, None)
            .expect_err("graph input-split path should still reject --crown-ibp");
        assert!(
            cpu_only_err
                .to_string()
                .contains("does not support --crown-ibp"),
            "expected CPU graph input-split rejection, got: {cpu_only_err}"
        );

        let direct_gpu = verifier
            .verify_graph_gpu_domain_list(&graph, &input, &[1.0_f32], 0.0, None, None)
            .expect("direct GPU DomainList path should accept the same config");

        let dispatched = dispatch_graph_constraint(
            &verifier,
            &graph,
            &input,
            &[1.0_f32],
            0.0,
            false,
            true,
            None,
            None,
            None,
        )
        .expect("gpu_bab dispatch should route input split through DomainList");

        assert_eq!(dispatched.result, direct_gpu.result);
        assert_eq!(dispatched.domains_explored, direct_gpu.domains_explored);
        assert_eq!(dispatched.domains_verified, direct_gpu.domains_verified);
        assert_eq!(dispatched.max_depth_reached, direct_gpu.max_depth_reached);
    }
}
