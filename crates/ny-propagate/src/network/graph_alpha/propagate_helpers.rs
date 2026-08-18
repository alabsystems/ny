// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Helper utilities for α-CROWN propagation.
//!
//! Contains:
//! - [`clamp_inverted_best_bounds`]: Fix cross-iteration elementwise merge inversions
//! - [`GraphNetwork::relu_preactivation_bounds`]: Typed pre-activation lookup (no input fallback)

use std::collections::HashMap;

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::{repair_inverted_bounds_nd, BoundedTensor, InversionRepair};
use tracing::{error, warn};

use crate::network::core::{GraphNetwork, NETWORK_INPUT};

/// Widen inverted intervals to `[-inf, +inf]` in elementwise best-bound arrays.
///
/// After merging per-iteration bounds via `max(lowers)` / `min(uppers)`,
/// some elements can invert (`best_lower[i] > best_upper[i]`) when different
/// iterations produce incompatible per-element optima. This function detects
/// inversions, logs a warning with the count, and widens to `[-inf, +inf]`
/// at each inverted element.
///
/// Widening is sound because `[-inf, +inf]` is always a valid overapproximation.
/// The previous behavior (clamping `lower` to `upper`) created degenerate
/// `[upper, upper]` intervals which could exclude the true value (#2655).
/// This matches `LinearBounds::concretize_sound` (`linear.rs:236-244`) which
/// also widens inversions to `[-inf, +inf]`.
///
/// Returns the number of elements that were widened.
///
/// Reference: alpha-beta-CROWN `optimized_bounds.py:943-947` detects inversions
/// with a print warning but does not correct them. We must correct because
/// `BoundedTensor::new` rejects inverted intervals.
pub(crate) fn clamp_inverted_best_bounds(
    best_lower: &mut ArrayD<f32>,
    best_upper: &mut ArrayD<f32>,
    context: &str,
) -> usize {
    let total = best_lower.len();
    let widened = repair_inverted_bounds_nd(best_lower, best_upper, InversionRepair::WidenToInf);
    if widened > 0 {
        warn!(
            widened_count = widened,
            total = total,
            context = context,
            "α-CROWN elementwise best-bound merge produced inverted intervals; widened to [-inf, +inf]",
        );
    }
    widened
}

/// Merge per-iteration concrete bounds into running element-wise best bounds.
///
/// Uses flat iteration (`iter`/`iter_mut`) so it works for non-standard-layout
/// arrays and across different ndim representations with matching element count.
pub(crate) fn update_elementwise_best_bounds(
    best_lower: &mut ArrayD<f32>,
    best_upper: &mut ArrayD<f32>,
    concrete_bounds: &BoundedTensor,
    iter: usize,
) -> Result<()> {
    if best_lower.len() != concrete_bounds.lower().len() {
        return Err(NyError::InternalError(format!(
            "best_lower length {} != concrete_bounds.lower() length {} during alpha-CROWN iteration {}",
            best_lower.len(),
            concrete_bounds.lower().len(),
            iter,
        )));
    }
    for (best, &curr) in best_lower.iter_mut().zip(concrete_bounds.lower().iter()) {
        // NaN-safe max: if best is NaN (stuck from iteration 0), any finite curr
        // replaces it. IEEE 754: `curr > NaN` is false, so without this guard a
        // NaN best value is permanently stuck (#3093).
        if curr > *best || best.is_nan() {
            *best = curr;
        }
    }

    if best_upper.len() != concrete_bounds.upper().len() {
        return Err(NyError::InternalError(format!(
            "best_upper length {} != concrete_bounds.upper() length {} during alpha-CROWN iteration {}",
            best_upper.len(),
            concrete_bounds.upper().len(),
            iter,
        )));
    }
    for (best, &curr) in best_upper.iter_mut().zip(concrete_bounds.upper().iter()) {
        // NaN-safe min: symmetric to the lower-bound guard above (#3093).
        if curr < *best || best.is_nan() {
            *best = curr;
        }
    }

    Ok(())
}

impl GraphNetwork {
    /// Resolve pre-activation bounds for a ReLU consumer in graph α-CROWN paths.
    ///
    /// The first input edge is treated as the pre-activation producer. If that
    /// producer is missing from `node_bounds`, this is an invariant violation and
    /// returns `InvalidSpec` with producer/consumer context (no input fallback).
    pub(super) fn relu_preactivation_bounds<'a>(
        &self,
        relu_name: &str,
        input: &'a BoundedTensor,
        node_bounds: &'a HashMap<String, BoundedTensor>,
        context: &str,
    ) -> Result<&'a BoundedTensor> {
        let relu_node = self.nodes.get(relu_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ReLU node '{}' not found while resolving pre-activation bounds ({})",
                relu_name, context
            ))
        })?;

        // #2098: Reject nodes with empty inputs rather than fabricating NETWORK_INPUT.
        // A ReLU with no inputs is a graph construction bug; using network input
        // bounds as pre-activation would produce unsound relaxations.
        let producer_name = relu_node
            .inputs
            .first()
            .map(String::as_str)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "ReLU node '{}' has no inputs — cannot determine pre-activation ({})",
                    relu_name, context
                ))
            })?;

        if producer_name == NETWORK_INPUT {
            return Ok(input);
        }

        node_bounds.get(producer_name).ok_or_else(|| {
            error!(
                producer = producer_name,
                consumer_relu = relu_name,
                context = context,
                "Graph α-CROWN missing pre-activation bounds for ReLU",
            );
            NyError::InvalidSpec(format!(
                "Missing pre-activation bounds for producer '{}' consumed by ReLU '{}' ({})",
                producer_name, relu_name, context
            ))
        })
    }

    // `is_sequential_graph()` moved to `network/core/graph/convert.rs` (#4097)
    // to share with non-alpha consumers (resident PGD, etc.).
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, ReLULayer};
    use crate::network::core::GraphNode;
    use ndarray::{arr1, array, Array, IxDyn, ShapeBuilder};

    #[test]
    fn test_relu_preactivation_bounds_missing_entry_returns_invalid_spec() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["missing_pre".to_string()],
        ));
        graph.set_output("relu");

        let input =
            BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()).unwrap();
        let node_bounds = HashMap::new();

        let err = graph
            .relu_preactivation_bounds("relu", &input, &node_bounds, "test-missing")
            .expect_err("missing pre-activation should return InvalidSpec");
        let msg = match err {
            NyError::InvalidSpec(msg) => msg,
            other => unreachable!("expected InvalidSpec, got {other:?}"),
        };

        assert!(msg.contains("missing_pre"), "missing producer not included");
        assert!(msg.contains("relu"), "consumer ReLU not included");
        assert!(msg.contains("test-missing"), "lookup context not included");
    }

    /// #2991: Empty-input ReLU is now caught at construction time by
    /// GraphNode::try_new() arity validation (#2481, #2686).
    #[test]
    fn test_relu_preactivation_bounds_empty_inputs_returns_invalid_spec_2098() {
        let err = GraphNode::try_new("relu", Layer::ReLU(ReLULayer), vec![])
            .expect_err("empty-input ReLU should return InvalidSpec at construction");
        let msg = match err {
            NyError::InvalidSpec(msg) => msg,
            other => unreachable!("expected InvalidSpec, got {other:?}"),
        };

        assert!(msg.contains("relu"), "consumer ReLU not included");
        assert!(msg.contains("1 input"), "arity requirement not included");
    }

    #[test]
    fn test_clamp_inverted_best_bounds_widens_inversions_2655() {
        // Simulate cross-iteration elementwise merge producing inverted intervals.
        // Iteration 1 gives lower=[1.0, -0.5, 0.3], upper=[2.0, 0.5, 1.0]
        // Iteration 2 gives lower=[0.8, 0.6, 0.2], upper=[1.5, 0.4, 0.9]
        // After merge: best_lower=max=[1.0, 0.6, 0.3], best_upper=min=[1.5, 0.4, 0.9]
        // Element [1] is inverted: 0.6 > 0.4
        let mut best_lower = array![1.0_f32, 0.6, 0.3].into_dyn();
        let mut best_upper = array![1.5_f32, 0.4, 0.9].into_dyn();

        let widened = clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, "test");

        assert_eq!(widened, 1, "exactly one element should be widened");
        // Inverted element widened to [-inf, +inf] (#2655: sound overapproximation)
        assert_eq!(best_lower[[1]], f32::NEG_INFINITY);
        assert_eq!(best_upper[[1]], f32::INFINITY);
        // Non-inverted elements unchanged
        assert_eq!(best_lower[[0]], 1.0);
        assert_eq!(best_lower[[2]], 0.3);
        assert_eq!(best_upper[[0]], 1.5);
        assert_eq!(best_upper[[2]], 0.9);

        // After widening, BoundedTensor::new_allow_infinite should succeed
        BoundedTensor::new_allow_infinite(best_lower, best_upper)
            .expect("bounds valid after widening");
    }

    #[test]
    fn test_clamp_inverted_best_bounds_noop_when_valid() {
        let mut best_lower = array![0.0_f32, -1.0, 0.5].into_dyn();
        let mut best_upper = array![1.0_f32, 0.0, 1.5].into_dyn();

        let widened = clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, "test-noop");

        assert_eq!(widened, 0, "no elements should be widened");
        // All elements unchanged
        assert_eq!(best_lower[[0]], 0.0);
        assert_eq!(best_lower[[1]], -1.0);
        assert_eq!(best_lower[[2]], 0.5);
    }

    #[test]
    fn test_update_elementwise_best_bounds_handles_non_standard_layout_2076() {
        let curr_lower =
            Array::from_shape_vec(IxDyn(&[2, 3]).f(), vec![5.0_f32, 1.0, 3.0, 4.0, 6.0, 2.0])
                .unwrap();
        let curr_upper =
            Array::from_shape_vec(IxDyn(&[2, 3]).f(), vec![8.0_f32, 4.0, 7.0, 6.0, 9.0, 5.0])
                .unwrap();
        assert!(
            curr_lower.as_slice().is_none() && curr_upper.as_slice().is_none(),
            "test setup: Fortran-order arrays must be non-standard layout"
        );

        let concrete_bounds = BoundedTensor::new(curr_lower, curr_upper).unwrap();
        let mut best_lower = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0_f32);
        let mut best_upper = ArrayD::from_elem(IxDyn(&[2, 3]), f32::INFINITY);

        update_elementwise_best_bounds(&mut best_lower, &mut best_upper, &concrete_bounds, 2)
            .unwrap();

        assert_eq!(best_lower, concrete_bounds.lower().to_owned());
        assert_eq!(best_upper, concrete_bounds.upper().to_owned());
    }

    #[test]
    fn test_update_elementwise_best_bounds_length_mismatch_returns_error_2076() {
        let concrete_bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, -2.0]).into_dyn(),
            arr1(&[1.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();
        let mut best_lower = ArrayD::from_elem(IxDyn(&[3]), 0.0_f32);
        let mut best_upper = ArrayD::from_elem(IxDyn(&[2]), f32::INFINITY);

        let err =
            update_elementwise_best_bounds(&mut best_lower, &mut best_upper, &concrete_bounds, 7)
                .unwrap_err();

        match err {
            NyError::InternalError(msg) => {
                assert!(
                    msg.contains("best_lower length 3 != concrete_bounds.lower() length 2"),
                    "unexpected error message: {msg}"
                );
                assert!(
                    msg.contains("iteration 7"),
                    "expected iteration context in error message: {msg}"
                );
            }
            other => unreachable!("expected InternalError, got {other}"),
        }
    }

    /// #3093: NaN in best_lower is replaced by subsequent finite value.
    /// Before fix, `curr > NaN` was always false (IEEE 754), so NaN was permanently stuck.
    #[test]
    fn test_update_elementwise_best_bounds_nan_lower_replaced_3093() {
        let mut best_lower = array![f32::NAN, 1.0_f32, f32::NAN].into_dyn();
        let mut best_upper = array![10.0_f32, 10.0, 10.0].into_dyn();

        let concrete_bounds = BoundedTensor::new(
            array![3.0_f32, 0.5, -1.0].into_dyn(),
            array![8.0_f32, 8.0, 8.0].into_dyn(),
        )
        .unwrap();

        update_elementwise_best_bounds(&mut best_lower, &mut best_upper, &concrete_bounds, 1)
            .unwrap();

        // NaN positions [0] and [2] should now have finite values from concrete_bounds
        assert_eq!(
            best_lower[[0]],
            3.0,
            "NaN best_lower[0] should be replaced by 3.0"
        );
        assert_eq!(
            best_lower[[1]],
            1.0,
            "non-NaN best_lower[1] stays at max(1.0, 0.5) = 1.0"
        );
        assert_eq!(
            best_lower[[2]],
            -1.0,
            "NaN best_lower[2] should be replaced by -1.0"
        );
    }

    /// #3093: NaN in best_upper is replaced by subsequent finite value.
    #[test]
    fn test_update_elementwise_best_bounds_nan_upper_replaced_3093() {
        let mut best_lower = array![-10.0_f32, -10.0, -10.0].into_dyn();
        let mut best_upper = array![f32::NAN, 5.0_f32, f32::NAN].into_dyn();

        let concrete_bounds = BoundedTensor::new(
            array![-8.0_f32, -8.0, -8.0].into_dyn(),
            array![2.0_f32, 3.0, 7.0].into_dyn(),
        )
        .unwrap();

        update_elementwise_best_bounds(&mut best_lower, &mut best_upper, &concrete_bounds, 1)
            .unwrap();

        // NaN positions [0] and [2] should now have finite values from concrete_bounds
        assert_eq!(
            best_upper[[0]],
            2.0,
            "NaN best_upper[0] should be replaced by 2.0"
        );
        assert_eq!(
            best_upper[[1]],
            3.0,
            "non-NaN best_upper[1] stays at min(5.0, 3.0) = 3.0"
        );
        assert_eq!(
            best_upper[[2]],
            7.0,
            "NaN best_upper[2] should be replaced by 7.0"
        );
    }

    /// #3093: clamp_inverted_best_bounds widens NaN elements to [-inf, +inf].
    #[test]
    fn test_clamp_inverted_best_bounds_widens_nan_3093() {
        let mut best_lower = array![f32::NAN, 1.0_f32, 0.0].into_dyn();
        let mut best_upper = array![5.0_f32, f32::NAN, 1.0].into_dyn();

        let widened = clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, "test-nan");

        assert_eq!(widened, 2, "two NaN-containing elements should be widened");
        // NaN element [0]: lower was NaN → widened
        assert_eq!(best_lower[[0]], f32::NEG_INFINITY);
        assert_eq!(best_upper[[0]], f32::INFINITY);
        // NaN element [1]: upper was NaN → widened
        assert_eq!(best_lower[[1]], f32::NEG_INFINITY);
        assert_eq!(best_upper[[1]], f32::INFINITY);
        // Valid element [2]: unchanged
        assert_eq!(best_lower[[2]], 0.0);
        assert_eq!(best_upper[[2]], 1.0);
    }

    #[test]
    fn test_relu_preactivation_bounds_uses_input_for_root_relu() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.set_output("relu");

        let input =
            BoundedTensor::new(array![-2.0_f32].into_dyn(), array![2.0_f32].into_dyn()).unwrap();
        let node_bounds = HashMap::new();

        let pre = graph
            .relu_preactivation_bounds("relu", &input, &node_bounds, "test-root")
            .expect("root ReLU should resolve to input bounds");

        assert_eq!(pre.lower(), input.lower());
        assert_eq!(pre.upper(), input.upper());
    }
}
