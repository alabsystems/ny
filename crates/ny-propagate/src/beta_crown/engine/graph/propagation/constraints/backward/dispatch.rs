// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Node-step dispatch coordinator for constrained backward CROWN.
//!
//! Contains:
//! - `ConstrainedNodeContext` — per-node context carrier
//! - `capture_constrained_linear_bounds` — linear bounds capture for lA caching
//! - `apply_constrained_backward_dispatch_result` — n-ary dispatch accumulation
//! - `try_finish_constrained_gpu_suffix` — seeded GPU suffix fast-exit
//! - `process_constrained_backward_node` — per-node router (Linear/ReLU/generic)
//! - `process_shared_dispatch_node` — generic operator backward dispatch
//!
//! Part of #4293 (directory-module split from former backward.rs monolith).

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use ny_core::{GpuCrownSeed, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::patches::{CrownBounds, PatchesMaterializationPurpose};
use crate::network::backward_dispatch::{
    dispatch_backward_layer, dispatch_backward_layer_finite_boundary, BackwardDispatchResult,
    DispatchContext,
};
use crate::network::{try_extract_single_gpu_layer, GraphNode};
use crate::{GraphNetwork, Layer, LinearBounds, MulBinaryRelaxationMode, NETWORK_INPUT};

use super::super::super::super::super::BetaCrownVerifier;
use super::super::patches::try_patches_step;
use super::{
    resolve_pre_activation, BackwardCrownResult, BackwardParams, ConstrainedBackwardSetup,
};

pub(super) struct ConstrainedNodeContext<'a> {
    pub node_name: &'a str,
    pub node: &'a GraphNode,
    pub first_input: &'a str,
}

fn capture_constrained_linear_bounds(
    current: &ConstrainedNodeContext<'_>,
    node_cb: &CrownBounds,
    captured_linear_bounds: &mut Option<HashMap<String, LinearBounds>>,
    deadline: Option<std::time::Instant>,
) -> Result<()> {
    let Some(linear_bounds_map) = captured_linear_bounds.as_mut() else {
        return Ok(());
    };
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained linear-bound capture",
    )?;
    let mut retained_capture_bytes = 0usize;
    for (index, bounds) in linear_bounds_map.values().enumerate() {
        retained_capture_bytes = retained_capture_bytes.saturating_add(bounds.memory_bytes());
        if index % 4096 == 4095 {
            super::super::ensure_constrained_propagation_deadline(
                deadline,
                "during constrained capture retained-memory scan",
            )?;
        }
    }
    let captured_lb = match node_cb {
        CrownBounds::Dense(lb) => lb.try_clone_with_deadline(deadline, retained_capture_bytes)?,
        CrownBounds::Patches(bounds) => bounds.to_dense_with_deadline_and_resident_for_purpose(
            deadline,
            retained_capture_bytes,
            PatchesMaterializationPurpose::Other,
        )?,
    };
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained linear-bound capture publication",
    )?;
    linear_bounds_map.insert(current.node_name.to_string(), captured_lb);
    Ok(())
}

fn apply_constrained_backward_dispatch_result(
    params: &BackwardParams<'_>,
    current: &ConstrainedNodeContext<'_>,
    pass_through_bounds: LinearBounds,
    result: BackwardDispatchResult,
    setup: &mut ConstrainedBackwardSetup<'_, '_>,
) -> Result<()> {
    super::super::ensure_constrained_propagation_deadline(
        params.deadline,
        "before constrained dispatch-result publication",
    )?;
    let applied = match result {
        BackwardDispatchResult::Single(new_lb) => {
            params.graph.accumulate_dense_bounds_to_input_with_deadline(
                current.first_input,
                *new_lb,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
                params.deadline,
            )
        }
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            let (input_a_name, input_b_name) = current.node.require_binary_inputs()?;
            verify_constrained_split_path_bias_zero_with_deadline(
                &bounds_a,
                "Constrained dispatch binary lhs split path",
                params.deadline,
            )?;
            verify_constrained_split_path_bias_zero_with_deadline(
                &bounds_b,
                "Constrained dispatch binary rhs split path",
                params.deadline,
            )?;
            GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                &bias_lower,
                &bias_upper,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
                params.deadline,
            )?;
            params
                .graph
                .accumulate_dense_bounds_to_input_with_deadline(
                    input_a_name,
                    *bounds_a,
                    &mut setup.state.node_crown_bounds,
                    setup.output_dim,
                    setup.input_dim,
                    &mut setup.state.input_accumulated,
                    params.deadline,
                )?;
            params.graph.accumulate_dense_bounds_to_input_with_deadline(
                input_b_name,
                *bounds_b,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
                params.deadline,
            )
        }
        BackwardDispatchResult::Nary {
            bounds,
            bias_lower,
            bias_upper,
        } => {
            // Authenticate every split path before the first publication. In
            // particular, a deadline during an O(N) bias scan cannot leave the
            // separate NETWORK_INPUT bias installed on its own.
            for bound in bounds.iter().flatten() {
                verify_constrained_split_path_bias_zero_with_deadline(
                    bound,
                    "Constrained dispatch n-ary split path",
                    params.deadline,
                )?;
            }
            // The zero-A bias wrapper is accumulated under NETWORK_INPUT, so
            // its column count MUST be the network-input width. The shared
            // deadline-aware helper constructs that carrier fallibly and polls
            // its fill/copy loops.
            GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                &bias_lower,
                &bias_upper,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
                params.deadline,
            )?;
            for (input_name, bound_opt) in current.node.inputs.iter().zip(bounds) {
                if let Some(bound) = bound_opt {
                    params
                        .graph
                        .accumulate_dense_bounds_to_input_with_deadline(
                            input_name,
                            bound,
                            &mut setup.state.node_crown_bounds,
                            setup.output_dim,
                            setup.input_dim,
                            &mut setup.state.input_accumulated,
                            params.deadline,
                        )?;
                }
            }
            Ok(())
        }
        // The dispatch result denotes the exact incoming carrier. It is owned
        // by this coordinator, so publication can move it without the shared
        // helper's historical deep clone.
        BackwardDispatchResult::PassThrough => {
            params.graph.accumulate_dense_bounds_to_input_with_deadline(
                current.first_input,
                pass_through_bounds,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
                params.deadline,
            )
        }
        BackwardDispatchResult::Unsupported(reason) => Err(NyError::UnsupportedOp(reason)),
    };

    applied.map_err(|error| match error {
        NyError::UnsupportedOp(reason) => NyError::UnsupportedOp(format!(
            "Constrained CROWN: layer '{}' ({}) unsupported: {}",
            current.node_name,
            current.node.layer.layer_type(),
            reason
        )),
        other => other,
    })
}

fn verify_constrained_split_path_bias_zero_with_deadline(
    bounds: &LinearBounds,
    context: &str,
    deadline: Option<std::time::Instant>,
) -> Result<()> {
    const TOLERANCE: f32 = 1e-30;
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained split-path bias validation",
    )?;
    for (label, bias) in [("lower_b", bounds.lower_b()), ("upper_b", bounds.upper_b())] {
        let mut max_abs = 0.0f32;
        for (index, &value) in bias.iter().enumerate() {
            if value.is_nan() {
                return Err(NyError::InvalidSpec(format!(
                    "{context} produced NaN in {label} split-path bounds \
                     (NaN corruption in dispatch layer)"
                )));
            }
            max_abs = max_abs.max(value.abs());
            if index % 4096 == 4095 {
                super::super::ensure_constrained_propagation_deadline(
                    deadline,
                    "during constrained split-path bias validation",
                )?;
            }
        }
        if max_abs >= TOLERANCE {
            return Err(NyError::InvalidSpec(format!(
                "{context} produced non-zero {label} in split-path bounds \
                 (max |v| = {max_abs:.2e})"
            )));
        }
    }
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "after constrained split-path bias validation",
    )?;
    Ok(())
}

impl BetaCrownVerifier {
    pub(super) fn try_finish_constrained_gpu_suffix(
        &self,
        params: &BackwardParams<'_>,
        node_name: &str,
        node_lb: &LinearBounds,
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
        output_shape: &[usize],
        runtime_refused: &mut bool,
    ) -> Result<Option<BoundedTensor>> {
        // The seeded GPU facade has no cooperative finite implementation: its
        // coefficient scans, layer extraction, and host-vector staging are all
        // opaque O(N) work. An expired finite request remains terminal, while a
        // live finite request declines this optimization before doing any of
        // that work and continues on the audited CPU path.
        if finite_constrained_gpu_suffix_is_declined(params.deadline)? {
            return Ok(None);
        }
        if *runtime_refused {
            return Ok(None);
        }
        // The bounded shared-executor facade grants only the audited
        // constrained 2..=8-row beta transaction. It must never be used as a
        // springboard to the process-global broad seeded GPU suffix.
        if params
            .context
            .engine
            .is_some_and(|engine| engine.forbids_unbounded_cpu_fallback())
        {
            return Ok(None);
        }
        // Soundness gate (#vnncomp-gpu-crown-soundness): masked when soundness is
        // required, forcing the CPU sound constrained-suffix path. See
        // `sound_gpu_gate`.
        let Some((gpu, use_sound)) = crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline(
            params.context.engine,
            params.deadline,
        ) else {
            return Ok(None);
        };

        if node_lb.lower_a().iter().any(|value| !value.is_finite())
            || node_lb.upper_a().iter().any(|value| !value.is_finite())
            || node_lb.lower_b().iter().any(|value| !value.is_finite())
            || node_lb.upper_b().iter().any(|value| !value.is_finite())
        {
            debug!(
                node_name,
                "Constrained GPU suffix skipped: non-finite seed coefficients or bias"
            );
            return Ok(None);
        }

        let mut gpu_layers = Vec::new();
        let mut current_name = node_name;
        loop {
            let node =
                params.graph.nodes.get(current_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!("Node not found: {}", current_name))
                })?;
            if node.inputs().len() != 1 {
                return Ok(None);
            }

            let input_name = node.require_unary_input()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                params.constrained_input
            } else {
                bounds_cache.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        input_name
                    ))
                })?
            };

            if try_extract_single_gpu_layer(&node.layer, pre_activation, &mut gpu_layers).is_none()
            {
                return Ok(None);
            }

            if input_name == NETWORK_INPUT {
                break;
            }
            current_name = input_name;
        }

        let seed = GpuCrownSeed {
            lower_a: node_lb.lower_a().iter().copied().collect::<Vec<_>>().into(),
            upper_a: node_lb.upper_a().iter().copied().collect::<Vec<_>>().into(),
            lower_b: node_lb.lower_b().iter().copied().collect::<Vec<_>>().into(),
            upper_b: node_lb.upper_b().iter().copied().collect::<Vec<_>>().into(),
            num_specs: node_lb.num_outputs(),
            current_dim: node_lb.num_inputs(),
        };
        let input_lower: Vec<f32> = params.constrained_input.lower().iter().copied().collect();
        let input_upper: Vec<f32> = params.constrained_input.upper().iter().copied().collect();

        let seeded = if use_sound {
            gpu.crown_backward_gpu_seeded_sound(&gpu_layers, &seed, &input_lower, &input_upper)
        } else {
            gpu.crown_backward_gpu_seeded(&gpu_layers, &seed, &input_lower, &input_upper)
        };
        let gpu_result = match seeded {
            Ok(result) => result,
            Err(error) => {
                *runtime_refused = true;
                debug!(
                    node_name,
                    error = %error,
                    "Constrained GPU suffix failed; falling back to CPU backward"
                );
                return Ok(None);
            }
        };
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "after constrained GPU suffix",
        )?;

        let expected_rows = output_shape
            .iter()
            .try_fold(1usize, |size, &axis| size.checked_mul(axis));
        if expected_rows != Some(seed.num_specs)
            || !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&gpu_result, seed.num_specs)
        {
            *runtime_refused = true;
            debug!(
                node_name,
                "Constrained GPU suffix produced malformed bounds; falling back to CPU backward"
            );
            return Ok(None);
        }

        let (Ok(lower), Ok(upper)) = (
            ArrayD::from_shape_vec(IxDyn(output_shape), gpu_result.lower_bounds),
            ArrayD::from_shape_vec(IxDyn(output_shape), gpu_result.upper_bounds),
        ) else {
            return Ok(None);
        };
        let output = BoundedTensor::new(lower, upper).ok();
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "before constrained GPU suffix publication",
        )?;
        Ok(output)
    }

    pub(super) fn process_constrained_backward_node(
        &self,
        params: &BackwardParams<'_>,
        is_standard: bool,
        node_name: &str,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        setup: &mut ConstrainedBackwardSetup<'_, '_>,
    ) -> Result<Option<BackwardCrownResult>> {
        let node = params
            .graph
            .nodes
            .get(node_name)
            .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;
        let first_input = if node.layer.is_binary() || node.layer.is_ternary() {
            node.inputs.first().map(String::as_str).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Node '{}' ({}) has no inputs",
                    node_name,
                    node.layer.layer_type()
                ))
            })?
        } else {
            node.require_unary_input()?
        };
        let current = ConstrainedNodeContext {
            node_name,
            node,
            first_input,
        };
        let node_cb = match setup.state.node_crown_bounds.take(node_name)? {
            Some(bounds) => bounds,
            None => return Ok(None),
        };

        capture_constrained_linear_bounds(
            &current,
            &node_cb,
            &mut setup.state.captured_linear_bounds,
            params.deadline,
        )?;

        if is_standard
            && params.deadline.is_none()
            && params.context.history.constraints.len() >= 12
        {
            if let CrownBounds::Dense(lb) = &node_cb {
                debug!(
                    "[#1817 node] {} lower_a={:?} upper_a={:?}",
                    current.node_name,
                    lb.lower_a()
                        .as_slice()
                        .map(|slice| &slice[..slice.len().min(8)]),
                    lb.upper_a()
                        .as_slice()
                        .map(|slice| &slice[..slice.len().min(8)]),
                );
            } else {
                debug!(
                    "[#1817 node] {} carrier=patches memory_bytes={}",
                    current.node_name,
                    node_cb.memory_bytes()
                );
            }
        }

        if matches!(&current.node.layer, Layer::Linear(_)) {
            self.process_linear_override(
                params,
                &current,
                node_cb,
                &*bounds_cache_mut,
                setup,
                is_standard,
            )?;
            return Ok(None);
        }

        if matches!(&current.node.layer, Layer::ReLU(_)) {
            let pre_activation = resolve_pre_activation(
                current.first_input,
                params.constrained_input,
                &*bounds_cache_mut,
            )?;
            self.process_relu_override(
                params,
                &current,
                pre_activation,
                node_cb,
                setup,
                is_standard,
            )?;
            return Ok(None);
        }

        self.process_shared_dispatch_node(
            params,
            &current,
            node_cb,
            bounds_cache_mut,
            setup,
            is_standard,
        )
    }

    fn process_shared_dispatch_node(
        &self,
        params: &BackwardParams<'_>,
        current: &ConstrainedNodeContext<'_>,
        mut node_cb: CrownBounds,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        setup: &mut ConstrainedBackwardSetup<'_, '_>,
        is_standard: bool,
    ) -> Result<Option<BackwardCrownResult>> {
        let pre_activation = resolve_pre_activation(
            current.first_input,
            params.constrained_input,
            &*bounds_cache_mut,
        )?;
        // A finite generic-dispatch closure protects a structured Patches
        // transaction that has crossed into a legacy Dense-only operator. An
        // ordinary beta-CROWN verifier also carries its overall timeout here,
        // but a relation that was Dense on entry is not part of that Patches
        // authority and retains its established constrained route.
        let finite_structured_boundary =
            params.deadline.is_some() && matches!(node_cb, CrownBounds::Patches(_));
        // GenBaB inv_rms norm branching (#norm-genbab): collect per-node inv_rms
        // window overrides from this domain's history, so the RmsNorm CROWN
        // backward narrows its certified inv_rms interval to the requesting
        // child subdomain.
        // Building the override map clones node names and grows nested Vecs
        // without a cooperative seam. Structured finite Patches work declines
        // that crossing; ordinary Dense work keeps the legacy preparation,
        // bracketed by the verifier deadline.
        let has_norm_inv_rms_constraints = params.context.history.has_norm_inv_rms_constraints();
        if has_norm_inv_rms_constraints {
            super::super::ensure_constrained_propagation_deadline(
                params.deadline,
                "before constrained norm-override preparation",
            )?;
        }
        let norm_inv_rms_map = if finite_structured_boundary && has_norm_inv_rms_constraints {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Constrained CROWN: cooperative finite norm override dispatch is unavailable at '{}'",
                current.node_name
            )));
        } else {
            params.context.history.norm_inv_rms_overrides()
        };
        if has_norm_inv_rms_constraints {
            super::super::ensure_constrained_propagation_deadline(
                params.deadline,
                "after constrained norm-override preparation",
            )?;
        }
        let effective_engine = if setup.gpu_suffix_runtime_refused {
            None
        } else {
            params.context.engine
        };
        let mut ctx = DispatchContext {
            node_name: current.node_name,
            layer: &current.node.layer,
            inputs: &current.node.inputs,
            pre_activation,
            network_input: params.constrained_input,
            node_bounds: (&*bounds_cache_mut).into(),
            engine: effective_engine,
            deadline: params.deadline,
            bilinear_alphas: None,
            mul_binary_relaxation: MulBinaryRelaxationMode::default(),
            mul_binary_alphas: None,
            norm_inv_rms_override: norm_inv_rms_map.as_ref(),
        };

        if try_patches_step(
            params.graph,
            "Constrained CROWN",
            current.node_name,
            current.node,
            &mut node_cb,
            current.first_input,
            pre_activation,
            &*bounds_cache_mut,
            &mut setup.state.node_crown_bounds,
            &mut setup.state.input_accumulated,
            effective_engine,
            params.deadline,
            params.patches_policy,
        )? {
            return Ok(None);
        }

        let node_lb = node_cb.into_dense_with_deadline_for_purpose(
            params.deadline,
            PatchesMaterializationPurpose::Other,
        )?;
        if is_standard && !params.capture_linear_bounds && setup.state.node_crown_bounds.is_empty()
        {
            if let Some(output_bounds) = self.try_finish_constrained_gpu_suffix(
                params,
                current.node_name,
                &node_lb,
                &*bounds_cache_mut,
                &setup.output_shape,
                &mut setup.gpu_suffix_runtime_refused,
            )? {
                return Ok(Some(BackwardCrownResult {
                    output_bounds,
                    intermediate: None,
                    captured_la: None,
                }));
            }
            if setup.gpu_suffix_runtime_refused {
                ctx.engine = None;
            }
        }

        // Name the failing node before the error bubbles up as an anonymous
        // "child propagation failed" (#ml4acopf-genbab was diagnosed blind
        // because the ShapeMismatch carried no node context).
        let result = if finite_structured_boundary {
            dispatch_backward_layer_finite_boundary(&ctx, &node_lb)
        } else {
            dispatch_backward_layer(&ctx, &node_lb)
        }
        .map_err(|e| {
            debug!(
                node = current.node_name,
                layer = current.node.layer.layer_type(),
                lb_rows = node_lb.num_outputs(),
                lb_cols = node_lb.num_inputs(),
                error = %e,
                "constrained backward dispatch failed"
            );
            e
        })?;
        apply_constrained_backward_dispatch_result(params, current, node_lb, result, setup)
            .map_err(|e| {
                debug!(
                    node = current.node_name,
                    layer = current.node.layer.layer_type(),
                    error = %e,
                    "constrained backward accumulate failed"
                );
                e
            })?;
        Ok(None)
    }
}

fn finite_constrained_gpu_suffix_is_declined(deadline: Option<std::time::Instant>) -> Result<bool> {
    if deadline.is_none() {
        return Ok(false);
    }
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained finite GPU suffix refusal",
    )?;
    Ok(true)
}

#[cfg(test)]
mod capture_atomic_tests {
    use super::*;
    use crate::bounds::patches::{PatchGeometry, PatchesData, PatchesLinearBounds};
    use crate::layers::ReLULayer;
    use ndarray::{Array1, ArrayD, IxDyn};
    use std::time::Duration;

    #[test]
    fn finite_gpu_suffix_declines_before_opaque_host_staging() {
        let live = std::time::Instant::now() + Duration::from_secs(30);
        assert!(finite_constrained_gpu_suffix_is_declined(Some(live))
            .expect("a live finite request should decline the optional suffix"));
        assert!(!finite_constrained_gpu_suffix_is_declined(None)
            .expect("the legacy no-deadline suffix remains eligible"));

        let expired = std::time::Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second deadline subtraction");
        let error = finite_constrained_gpu_suffix_is_declined(Some(expired))
            .expect_err("an expired finite request must remain terminal");
        assert!(error.is_deadline_exceeded());
    }

    #[test]
    fn constrained_linear_capture_budget_refusal_preserves_source_and_map() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "0");

            let geometry = PatchGeometry::anchored(vec![0, 1], vec![0, 1])
                .expect("fixture axes are non-empty");
            let data = PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(IxDyn(&[1, 2, 2, 1, 1, 1]), 0.5)),
                geometry,
                identity: false,
                output_shape: (1, 2, 2),
                input_shape: (1, 2, 2),
                unstable_idx: None,
            };
            let patches = PatchesLinearBounds {
                row_count: 4,
                lower_a: data.clone(),
                lower_b: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                upper_a: data,
                upper_b: Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]),
            };
            let expected = patches.clone();
            let carrier = CrownBounds::Patches(Box::new(patches));
            let node = GraphNode::from_input("relu_capture", Layer::ReLU(ReLULayer::new()));
            let current = ConstrainedNodeContext {
                node_name: "relu_capture",
                node: &node,
                first_input: NETWORK_INPUT,
            };
            let mut captures = Some(HashMap::from([(
                "sentinel".to_string(),
                LinearBounds::identity(1),
            )]));

            let error = capture_constrained_linear_bounds(&current, &carrier, &mut captures, None)
                .expect_err("zero budget must refuse constrained linear capture");
            assert!(
                matches!(error, NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );
            let CrownBounds::Patches(actual) = &carrier else {
                panic!("borrowed capture changed the source carrier")
            };
            assert_eq!(actual.row_count, expected.row_count);
            assert_eq!(actual.lower_a.geometry, expected.lower_a.geometry);
            assert_eq!(actual.lower_a.patches, expected.lower_a.patches);
            assert_eq!(actual.lower_b, expected.lower_b);
            assert_eq!(actual.upper_a.geometry, expected.upper_a.geometry);
            assert_eq!(actual.upper_a.patches, expected.upper_a.patches);
            assert_eq!(actual.upper_b, expected.upper_b);
            let captures = captures.expect("capture map remains present");
            assert_eq!(captures.len(), 1);
            assert!(captures.contains_key("sentinel"));
            assert!(!captures.contains_key("relu_capture"));
        });
    }

    #[test]
    fn constrained_linear_capture_deadline_refusal_preserves_source_and_map() {
        let patches = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
        let expected_memory = patches.memory_bytes();
        let expected_lower_geometry = patches.lower_a.geometry.clone();
        let expected_upper_geometry = patches.upper_a.geometry.clone();
        let carrier = CrownBounds::Patches(Box::new(patches));
        let node = GraphNode::from_input("relu_capture", Layer::ReLU(ReLULayer::new()));
        let current = ConstrainedNodeContext {
            node_name: "relu_capture",
            node: &node,
            first_input: NETWORK_INPUT,
        };
        let mut captures = Some(HashMap::from([(
            "sentinel".to_string(),
            LinearBounds::identity(1),
        )]));
        let expired = std::time::Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second deadline subtraction");

        let error =
            capture_constrained_linear_bounds(&current, &carrier, &mut captures, Some(expired))
                .expect_err("expired capture must be terminal");

        assert!(error.is_deadline_exceeded());
        let CrownBounds::Patches(actual) = &carrier else {
            panic!("deadline refusal replaced source Patches")
        };
        assert_eq!(actual.memory_bytes(), expected_memory);
        assert_eq!(actual.lower_a.geometry, expected_lower_geometry);
        assert_eq!(actual.upper_a.geometry, expected_upper_geometry);
        let captures = captures.expect("capture map remains present");
        assert_eq!(captures.len(), 1);
        assert!(captures.contains_key("sentinel"));
    }
}
