// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential network extraction from graph for SDP-CROWN.
//!
//! Both `try_to_sequential_network` (SDP-CROWN policy) and
//! `try_collect_crown_ibp_bounds_via_sequential_network` delegate to
//! [`GraphNetwork::try_unary_chain`] for the structural walk,
//! then apply their own layer-policy filter on the result.

use super::demand::nodes_requiring_crown_tightening;
use crate::layers::Layer;
use crate::network::core::graph::CrownIbpRuntimePolicy;
use crate::network::core::{GraphNetwork, Network};
use crate::network::ibp::NetworkIbpExt;
use crate::types::{
    BoundsProvenance, CrownIbpBoundsResult, CrownIbpFallbackEvent, CrownIbpFallbackReason,
    GraphCrownIbpBoundsResult,
};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// A sequential shortcut borrows the graph's tightening clock. Its expired
/// deadline can decline the shortcut, but no other error can authorize fallback.
fn finish_sequential_tightening<T>(
    result: Result<T>,
    deadline: Option<Instant>,
    now: Instant,
) -> Result<Option<T>> {
    match result {
        Ok(result) => Ok(Some(result)),
        Err(NyError::DeadlineExceeded(_)) if deadline.is_some_and(|limit| now >= limit) => Ok(None),
        Err(error) => Err(error),
    }
}

impl GraphNetwork {
    /// Try to convert this graph network to a sequential Network for SDP-CROWN.
    ///
    /// Returns `Some(Network)` only when the graph is sequential and contains
    /// only Linear/ReLU layers; otherwise returns `None`.
    ///
    /// Delegates to [`Self::try_unary_chain`] for the structural walk, then
    /// applies the SDP-CROWN Linear/ReLU policy filter.
    pub fn try_to_sequential_network(&self) -> Option<Network> {
        let exec_order = self.exec_order().ok()?;
        let chain = self.try_unary_chain(exec_order).ok()??;

        // SDP-CROWN policy: only Linear and ReLU layers are accepted
        for layer in &chain.network.layers {
            match layer {
                Layer::Linear(_) | Layer::ReLU(_) => {}
                _ => return None,
            }
        }

        Some(chain.network)
    }

    /// Reuse the sequential `Network` CROWN-IBP collector for linear-chain graphs.
    ///
    /// This lets `GraphNetwork` benefit from the existing sequential GPU partial
    /// fast path in `network/ibp.rs` for graph-loaded ONNX models whose topology
    /// is still a simple chain (for example metaroom/soundnessbench benchmark
    /// cases converted through `GraphNetwork::from_sequential`).
    ///
    /// Delegates to [`Self::try_unary_chain`] for the structural walk.
    pub(in crate::network::graph_alpha) fn try_collect_crown_ibp_bounds_via_sequential_network(
        &self,
        exec_order: &[String],
        input: &BoundedTensor,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        runtime_policy: CrownIbpRuntimePolicy,
    ) -> Result<Option<GraphCrownIbpBoundsResult>> {
        let Some(chain) = self.try_unary_chain(exec_order)? else {
            return Ok(None);
        };

        // An already-expired graph deadline historically governs only CROWN
        // tightening: the graph-native collector publishes the complete IBP
        // map that its caller already owns.  Do not enter the sequential
        // status collector, whose finite API correctly refuses to allocate and
        // clone a new result after expiry.  Hard full-phase callers are stopped
        // at the post-IBP checkpoint in the options entry point.
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Ok(None);
        }

        let precomputed_ibp = chain
            .node_names
            .iter()
            .map(|node_name| {
                ibp_bounds.get(node_name).cloned().ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Sequential CROWN-IBP graph fast path missing IBP bounds for node '{node_name}'"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let sequential = chain
            .network
            .collect_crown_ibp_bounds_with_precomputed_ibp_and_budget_and_flat_bias_policy_impl(
                input,
                precomputed_ibp,
                engine,
                deadline,
                // Preset-scoped per-node budget survives the sequential
                // delegation (#cgan-bn11-budget); default = old constants.
                &self.crown_ibp_per_node_time_budget,
                runtime_policy.honest_provenance_enabled,
                runtime_policy.patches_deadline_flat_bias_enabled,
            );
        // This clock can expire during the shortcut as well as before entry.
        // The graph still owns its complete certified IBP map. Resume its
        // existing deadline fallback without restarting forward collection.
        let Some(sequential) = finish_sequential_tightening(sequential, deadline, Instant::now())?
        else {
            return Ok(None);
        };
        let demand_set = nodes_requiring_crown_tightening(self, exec_order, ibp_bounds);

        Ok(Some(remap_sequential_crown_ibp_result(
            &chain.node_names,
            &demand_set,
            ibp_bounds,
            sequential,
        )))
    }
}

fn remap_sequential_crown_ibp_result(
    layer_names: &[String],
    demand_set: &HashSet<String>,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    sequential: CrownIbpBoundsResult,
) -> GraphCrownIbpBoundsResult {
    let CrownIbpBoundsResult {
        bounds,
        provenance,
        fallback_events,
    } = sequential;

    let bounds = layer_names
        .iter()
        .cloned()
        .zip(bounds)
        .map(|(node_name, bound)| {
            let bound = if demand_set.contains(&node_name) {
                bound
            } else {
                // Match the DAG collector's demand-driven contract: skipped nodes keep
                // their original forward IBP bounds, not forward-propagated
                // tightenings from the sequential shortcut path.
                ibp_bounds.get(&node_name).cloned().unwrap_or(bound)
            };
            (node_name, bound)
        })
        .collect::<HashMap<_, _>>();
    let provenance = layer_names
        .iter()
        .cloned()
        .zip(provenance)
        .map(|(node_name, provenance)| {
            let provenance = if demand_set.contains(&node_name) {
                provenance
            } else {
                // Match the DAG collector's demand-driven policy: nodes outside the
                // selected producer set keep their forward IBP bound and do
                // not claim a real CROWN tightening provenance.
                BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DemandDrivenSkip)
            };
            (node_name, provenance)
        })
        .collect::<HashMap<_, _>>();
    let fallback_events = fallback_events
        .into_iter()
        .filter_map(|event| {
            let Some(node_name) = layer_names.get(event.layer_index) else {
                return Some(event);
            };
            if !demand_set.contains(node_name) {
                return None;
            }
            let details = format!("node '{node_name}': {}", event.details);
            Some(CrownIbpFallbackEvent { details, ..event })
        })
        .collect();

    GraphCrownIbpBoundsResult {
        bounds,
        provenance,
        fallback_events,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn sequential_expiry_requires_the_borrowed_clock_to_have_expired() {
        let now = Instant::now();
        for deadline in [None, Some(now + Duration::from_secs(1))] {
            assert!(matches!(
                finish_sequential_tightening::<()>(
                    Err(NyError::DeadlineExceeded("inner clock".into())),
                    deadline,
                    now,
                ),
                Err(NyError::DeadlineExceeded(_))
            ));
        }
        for deadline in [now, now.checked_sub(Duration::from_secs(1)).unwrap()] {
            assert!(finish_sequential_tightening::<()>(
                Err(NyError::DeadlineExceeded("tightening clock".into())),
                Some(deadline),
                now,
            )
            .unwrap()
            .is_none());
        }
    }

    #[test]
    fn sequential_expiry_never_masks_numerical_source_or_wrapped_errors() {
        let now = Instant::now();
        for error in [
            NyError::NumericalInstability("nonfinite".into()),
            NyError::SoundnessRefusal("source authority".into()),
            NyError::ShapeMismatch {
                expected: vec![2],
                got: vec![3],
            },
            NyError::InvalidSpec("source".into()),
            NyError::CpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
                site: "test",
            },
            NyError::LayerError {
                layer_index: 0,
                layer_type: "wrapped".into(),
                source: Box::new(NyError::DeadlineExceeded("nested".into())),
            },
        ] {
            let expected = error.to_string();
            let actual = finish_sequential_tightening::<()>(Err(error), Some(now), now)
                .expect_err("only a direct expired tightening deadline may fall back");
            assert_eq!(actual.to_string(), expected);
        }
    }

    #[test]
    fn sequential_completed_result_is_moved_unchanged_at_the_time_boundary() {
        let now = Instant::now();
        let value = Box::new([1.0_f32, -0.0]);
        let allocation = value.as_ptr();
        let retained = finish_sequential_tightening(Ok(value), Some(now), now)
            .unwrap()
            .unwrap();
        assert_eq!(allocation, retained.as_ptr());
        assert_eq!(retained[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(retained[1].to_bits(), (-0.0_f32).to_bits());
    }
}
