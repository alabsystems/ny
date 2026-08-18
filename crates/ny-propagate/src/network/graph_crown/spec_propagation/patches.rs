// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches fast-path flow control for spec-guided CROWN backward.
//!
//! This module handles the Patches-mode backward dispatch attempt and the
//! deadline-aware `ensure_dense` downgrade when patches dispatch fails. It is pure flow
//! control around the already-shared patches helper surface — no operator math
//! belongs here. Split from `core.rs` as part of #3960.

use crate::bounds::patches::{CrownBounds, PatchesMaterializationPurpose};
use crate::layers::Layer;
use crate::network::core::{
    crown_backward_step_patches_spec_crown, CrownStepResult, SpecPatchesStepError,
};
use crate::types::CrownIbpFallbackReason;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::debug;

/// Resource failures from any pollable Patches operator or materializer are
/// node authorities, not invitations to retry the unchanged relation through
/// Dense. Semantic/numerical refusals retain the historical dense retry.
fn patches_resource_fallback(error: &NyError) -> Option<CrownIbpFallbackReason> {
    match error {
        NyError::DeadlineExceeded(_) => Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded),
        NyError::CpuMemoryExceeded { .. } => Some(CrownIbpFallbackReason::MemoryBudgetExceeded),
        _ => None,
    }
}

#[inline]
fn patches_dense_retry_is_authorized(error: &NyError) -> bool {
    matches!(
        error,
        NyError::UnsupportedOp(_) | NyError::UnsupportedConfiguration(_)
    )
}

/// Result of the patches fast-path dispatch attempt.
pub(super) enum PatchesDispatchOutcome {
    /// Patches dispatch succeeded; caller should accumulate `node_cb` to input
    /// and continue to the next node.
    AccumulateToInput,
    /// Full IBP fallback needed for the given reason.
    IbpFallback(CrownIbpFallbackReason),
    /// Patches did not apply or failed; `node_cb` has been ensured dense.
    /// Caller should proceed with dense dispatch.
    FallThroughDense,
}

/// Attempt patches-mode backward dispatch with a deadline-aware dense downgrade.
///
/// If the node's `CrownBounds` are in Patches mode, attempts the patches
/// backward step. On success, returns `AccumulateToInput`. On recoverable
/// failure (patches dispatch error), downgrades to Dense mode and returns
/// `FallThroughDense`. On irrecoverable failure (`ensure_dense` fails),
/// returns `IbpFallback`.
pub(super) fn dispatch_patches_or_fallback(
    node_cb: &mut CrownBounds,
    layer: &Layer,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    node_deadline: Option<Instant>,
    node_name: &str,
    layer_type: &str,
) -> Result<PatchesDispatchOutcome> {
    match crown_backward_step_patches_spec_crown(
        layer,
        node_cb,
        pre_activation,
        engine,
        0,
        "SPEC-CROWN",
        node_deadline,
    ) {
        Ok(CrownStepResult::Continue) => return Ok(PatchesDispatchOutcome::AccumulateToInput),
        Ok(CrownStepResult::IbpFallback(fallback)) => {
            return Ok(PatchesDispatchOutcome::IbpFallback(fallback.reason))
        }
        Err(SpecPatchesStepError::ReluDeadlineExceeded) => {
            // A node-local deadline is verifier authority, not a recoverable
            // operator error. In particular, do not materialize the unchanged
            // giant incoming Patches tensor as Dense after a cooperative ReLU
            // worker expires: that would extend the timeout and risk an OOM.
            return Ok(PatchesDispatchOutcome::IbpFallback(
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
            ));
        }
        Err(SpecPatchesStepError::Ordinary(err)) => {
            if let Some(reason) = patches_resource_fallback(&err) {
                debug!(
                    "Spec-guided CROWN: Patches resource authority at {} ({}): {}; keeping Patches atomic",
                    node_name, layer_type, err
                );
                return Ok(PatchesDispatchOutcome::IbpFallback(reason));
            }
            if !patches_dense_retry_is_authorized(&err) {
                return Err(err);
            }
            debug!(
                "Spec-guided CROWN: Patches dispatch is unsupported at {} ({}): {}, falling back to Dense dispatch",
                node_name, layer_type, err
            );
        }
    }
    if matches!(node_cb, CrownBounds::Patches(_)) {
        match node_cb.ensure_dense_with_deadline_for_purpose(
            node_deadline,
            PatchesMaterializationPurpose::Other,
        ) {
            Ok(_) => {}
            Err(err) => {
                debug!(
                    "Spec-guided CROWN: ensure_dense failed at {}: {}, falling back to IBP",
                    node_name, err
                );
                return match patches_resource_fallback(&err) {
                    Some(reason) => Ok(PatchesDispatchOutcome::IbpFallback(reason)),
                    None => Err(err),
                };
            }
        }
    }
    Ok(PatchesDispatchOutcome::FallThroughDense)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::patches::{
        patches_to_dense_call_sites, reset_patches_to_dense_call_count, PatchGeometry, PatchesData,
        PatchesLinearBounds,
    };
    use crate::layers::{Conv2dLayer, ConvTranspose2dLayer, ReLULayer};
    use ndarray::{Array1, ArrayD, IxDyn};
    use std::time::Duration;

    #[test]
    fn expired_relu_transaction_returns_typed_fallback_without_dense_publication() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_PATCHES_DEADLINE_RELU", "1");
            let make_side = |value| PatchesData {
                coeff_err: Some(Array1::from_vec(vec![1.0e-4])),
                patches: Some(ArrayD::from_elem(IxDyn(&[1, 1, 1, 1, 1, 1, 1]), value)),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape: (1, 1, 1),
                input_shape: (1, 1, 1),
                unstable_idx: None,
            };
            let mut bounds = CrownBounds::Patches(Box::new(PatchesLinearBounds {
                row_count: 1,
                lower_a: make_side(-0.75),
                lower_b: Array1::from_vec(vec![0.25]),
                upper_a: make_side(1.25),
                upper_b: Array1::from_vec(vec![-0.5]),
            }));
            let pre_activation = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0),
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), 2.0),
            )
            .unwrap();
            let expired = Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("one millisecond fits before the current instant");

            let outcome = dispatch_patches_or_fallback(
                &mut bounds,
                &Layer::ReLU(ReLULayer::new()),
                &pre_activation,
                None,
                Some(expired),
                "Relu_51",
                "ReLU",
            )
            .expect("deadline authority is a typed fallback policy");
            assert!(matches!(
                outcome,
                PatchesDispatchOutcome::IbpFallback(
                    CrownIbpFallbackReason::PerNodeDeadlineExceeded
                )
            ));
            assert!(
                matches!(bounds, CrownBounds::Patches(_)),
                "deadline fallback must not publish partial or dense CrownBounds"
            );
        });
    }

    #[test]
    fn expired_generic_dense_retry_honors_node_authority_atomically() {
        crate::tests::with_env_edits(|env| {
            env.remove("NY_PATCHES_DEADLINE_RELU");
            let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0f32])
                .expect("valid Conv2d kernel");
            let layer = Layer::Conv2d(
                Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 1, 1)
                    .expect("valid Conv2d layer"),
            );
            let mut bounds = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
                (1, 2, 2),
                (1, 2, 2),
            )));
            let pre_activation = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0),
                ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0),
            )
            .expect("valid Conv2d pre-activation bounds");
            let expired = Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("one millisecond fits before the current instant");

            let outcome = dispatch_patches_or_fallback(
                &mut bounds,
                &layer,
                &pre_activation,
                None,
                Some(expired),
                "Conv_7",
                "Conv2d",
            )
            .expect("deadline authority is a typed fallback policy");

            assert!(matches!(
                outcome,
                PatchesDispatchOutcome::IbpFallback(
                    CrownIbpFallbackReason::PerNodeDeadlineExceeded
                )
            ));
            assert!(
                matches!(bounds, CrownBounds::Patches(_)),
                "expired generic fallback must preserve the Patches carrier"
            );
        });
    }

    #[test]
    fn resource_and_dense_retry_classifiers_preserve_error_authority() {
        assert_eq!(
            patches_resource_fallback(&NyError::DeadlineExceeded("test".into())),
            Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
        );
        assert_eq!(
            patches_resource_fallback(&NyError::CpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
                site: "test",
            }),
            Some(CrownIbpFallbackReason::MemoryBudgetExceeded)
        );
        assert_eq!(
            patches_resource_fallback(&NyError::UnsupportedConfiguration(
                "dense retry remains valid".into()
            )),
            None
        );
        assert!(patches_dense_retry_is_authorized(
            &NyError::UnsupportedConfiguration("dense retry remains valid".into())
        ));
        assert!(patches_dense_retry_is_authorized(&NyError::UnsupportedOp(
            "dense retry remains valid".into()
        )));
        assert!(!patches_dense_retry_is_authorized(&NyError::InvalidSpec(
            "must remain terminal".into()
        )));
        assert!(!patches_dense_retry_is_authorized(
            &NyError::NumericalInstability("must remain terminal".into())
        ));
    }

    fn stride_two_convtranspose_fixture() -> (Layer, CrownBounds, BoundedTensor) {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0f32])
            .expect("valid ConvTranspose2d kernel");
        let layer = Layer::ConvTranspose2d(
            ConvTranspose2dLayer::with_input_shape(
                kernel,
                Some(Array1::from_vec(vec![0.25])),
                (2, 2),
                (0, 0),
                2,
                2,
            )
            .expect("valid ConvTranspose2d layer"),
        );
        let bounds = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
            (1, 3, 3),
            (1, 3, 3),
        )));
        let pre_activation = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0),
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0),
        )
        .expect("valid ConvTranspose2d pre-activation bounds");
        (layer, bounds, pre_activation)
    }

    #[test]
    fn expired_convtranspose_planner_keeps_patches_and_returns_typed_fallback() {
        let (layer, mut bounds, pre_activation) = stride_two_convtranspose_fixture();
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before the current instant");

        let outcome = dispatch_patches_or_fallback(
            &mut bounds,
            &layer,
            &pre_activation,
            None,
            Some(expired),
            "ConvTranspose_7",
            "ConvTranspose2d",
        )
        .expect("deadline authority is a typed fallback policy");

        assert!(matches!(
            outcome,
            PatchesDispatchOutcome::IbpFallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
        ));
        assert!(
            matches!(bounds, CrownBounds::Patches(_)),
            "deadline authority must not densify the unchanged identity"
        );
    }

    #[test]
    fn live_convtranspose_deadline_continues_anchored_without_dense_retry() {
        let (layer, mut bounds, pre_activation) = stride_two_convtranspose_fixture();

        reset_patches_to_dense_call_count();
        let outcome = dispatch_patches_or_fallback(
            &mut bounds,
            &layer,
            &pre_activation,
            None,
            Some(Instant::now() + Duration::from_secs(30)),
            "ConvTranspose_7",
            "ConvTranspose2d",
        )
        .expect("live finite ConvTranspose Patches dispatch must succeed");

        assert!(matches!(outcome, PatchesDispatchOutcome::AccumulateToInput));
        let CrownBounds::Patches(bounds) = bounds else {
            panic!("supported finite ConvTranspose route must remain Patches");
        };
        assert!(matches!(
            bounds.lower_a.geometry,
            PatchGeometry::Anchored(_)
        ));
        assert!(matches!(
            bounds.upper_a.geometry,
            PatchGeometry::Anchored(_)
        ));
        assert!(
            patches_to_dense_call_sites().is_empty(),
            "supported finite ConvTranspose route must not materialize Dense"
        );
    }

    #[test]
    fn memory_limited_convtranspose_planner_keeps_patches_and_returns_typed_fallback() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "0");
            let (layer, mut bounds, pre_activation) = stride_two_convtranspose_fixture();

            let outcome = dispatch_patches_or_fallback(
                &mut bounds,
                &layer,
                &pre_activation,
                None,
                None,
                "ConvTranspose_7",
                "ConvTranspose2d",
            )
            .expect("memory authority is a typed fallback policy");

            assert!(matches!(
                outcome,
                PatchesDispatchOutcome::IbpFallback(CrownIbpFallbackReason::MemoryBudgetExceeded)
            ));
            assert!(
                matches!(bounds, CrownBounds::Patches(_)),
                "memory authority must not densify the unchanged identity"
            );
        });
    }
}
