// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Initial bound computation helpers.

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::beta_crown::config::BetaCrownConfig;
use crate::network::ibp::helpers::crown_ibp_partial_node_count;
use crate::Network;

use super::tensor_ext::BoundedTensorExt;
use super::BetaCrownVerifier;

pub(in crate::beta_crown::engine) struct InitialBoundsComputation {
    pub output_bounds: BoundedTensor,
    pub root_layer_bounds: Option<Vec<BoundedTensor>>,
}

pub(in crate::beta_crown::engine) fn crown_ibp_budget_exceeded(
    config: &BetaCrownConfig,
    network: &Network,
) -> bool {
    let Some(max_nodes) = config.max_crown_ibp_nodes else {
        return false;
    };

    let partial_nodes = crown_ibp_partial_node_count(network);
    let exceeded = partial_nodes > max_nodes;
    if exceeded {
        debug!(
            "Sequential CROWN-IBP node budget exceeded: partial_nodes={}, max_crown_ibp_nodes={}",
            partial_nodes, max_nodes
        );
    }
    exceeded
}

impl BetaCrownVerifier {
    /// Intersect the sequential α-CROWN root bound with the `GraphNetwork`
    /// α-CROWN bound for the equivalent network, returning the tighter (sound)
    /// bound. See call site in `compute_initial_bounds_and_layer_bounds_engine`
    /// for the soundness rationale (#1817).
    ///
    /// Falls back to `seq_bounds` unchanged if conversion or graph propagation
    /// fails, or if shapes/finiteness don't line up — never widens the bound.
    fn tighten_root_with_graph_alpha_crown(
        &self,
        network: &Network,
        input: &BoundedTensor,
        alpha_config: &crate::bounds::AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        seq_bounds: BoundedTensor,
    ) -> BoundedTensor {
        // Restrict to networks without patch-based (Conv/ConvTranspose/MaxPool)
        // layers. For those, the sequential α-CROWN already falls back to plain
        // CROWN (see `propagate_alpha_crown_with_config_and_engine_impl`), and
        // running the graph engine's patch-based backward here would add
        // significant work for marginal benefit. Keeping the graph tightening
        // scoped to dense MLP-style networks — where the sequential vs graph
        // α-CROWN tightness gap actually manifests (#1817) — also avoids
        // perturbing global patch-densification instrumentation used by tests.
        let has_patch_layers = network.layers().iter().any(|l| {
            matches!(
                l,
                crate::layers::Layer::Conv2d(_)
                    | crate::layers::Layer::ConvTranspose2d(_)
                    | crate::layers::Layer::MaxPool2d(_)
            )
        });
        if has_patch_layers {
            return seq_bounds;
        }

        let graph = match crate::GraphNetwork::from_sequential(network) {
            Ok(g) => g,
            Err(_) => return seq_bounds,
        };
        let graph_bounds =
            match graph.propagate_alpha_crown_with_config_and_engine(input, alpha_config, engine) {
                Ok(b) => b,
                Err(_) => return seq_bounds,
            };

        let (sl, su) = seq_bounds.lower_upper();
        let (gl, gu) = graph_bounds.lower_upper();
        if sl.shape() != gl.shape() || su.shape() != gu.shape() {
            return seq_bounds;
        }

        // Element-wise sound intersection: max of lowers, min of uppers. A
        // non-finite value from either engine must never win, so skip it.
        let merge = |a: f32, b: f32, take_max: bool| -> f32 {
            match (a.is_finite(), b.is_finite()) {
                (true, true) => {
                    if take_max {
                        a.max(b)
                    } else {
                        a.min(b)
                    }
                }
                (true, false) => a,
                (false, true) => b,
                (false, false) => a,
            }
        };

        let lower = ndarray::Zip::from(sl)
            .and(gl)
            .map_collect(|&a, &b| merge(a, b, true));
        let upper = ndarray::Zip::from(su)
            .and(gu)
            .map_collect(|&a, &b| merge(a, b, false));

        // Guard against an inverted interval (shouldn't happen for valid bounds,
        // but if it did we keep the original sound sequential bound).
        if ndarray::Zip::from(&lower).and(&upper).any(|&l, &u| l > u) {
            return seq_bounds;
        }

        match BoundedTensor::new(lower, upper) {
            Ok(b) => b,
            Err(_) => seq_bounds,
        }
    }

    /// Compute initial bounds with optional early termination and GPU acceleration.
    ///
    /// `deadline`: If set, alpha-CROWN optimization will bail early when this
    /// wall-clock deadline is exceeded (#2698).
    pub(in crate::beta_crown::engine) fn compute_initial_bounds_and_layer_bounds_engine(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold_check: Option<(f32, bool)>, // (threshold, verify_upper_bound)
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<InitialBoundsComputation> {
        let mut cached_ibp_bounds = None;
        let budget_exceeded = crown_ibp_budget_exceeded(&self.config, network);
        let use_crown_ibp_layer_bounds = self.config.use_crown_ibp && !budget_exceeded;

        if let Some((threshold, verify_upper)) = threshold_check {
            match network.collect_ibp_bounds_with_deadline(input, deadline) {
                Ok(ibp_layer_bounds) => {
                    let ibp_output = ibp_layer_bounds
                        .last()
                        .cloned()
                        .unwrap_or_else(|| input.clone());
                    let ibp_lower = ibp_output.lower_scalar();
                    let ibp_upper = ibp_output.upper_scalar();
                    let verified = BetaCrownConfig::domain_is_verified_for_mode(
                        verify_upper,
                        ibp_lower,
                        ibp_upper,
                        threshold,
                    );
                    if verified {
                        debug!(
                            "IBP fast-path: bounds [{:.2}, {:.2}] verify threshold {}, skipping CROWN",
                            ibp_lower, ibp_upper, threshold
                        );
                        return Ok(InitialBoundsComputation {
                            output_bounds: ibp_output,
                            root_layer_bounds: None,
                        });
                    }
                    debug!(
                        "IBP fast-path: bounds [{:.2}, {:.2}] don't verify threshold {}, proceeding to CROWN",
                        ibp_lower, ibp_upper, threshold
                    );
                    cached_ibp_bounds = Some(ibp_layer_bounds);
                }
                Err(err) => {
                    debug!("IBP fast-path failed: {}, proceeding to CROWN", err);
                }
            }
        }

        let mut cached_root_layer_bounds = match (use_crown_ibp_layer_bounds, cached_ibp_bounds) {
            (true, Some(ibp_bounds)) => {
                Some(network.collect_crown_ibp_bounds_with_precomputed_ibp(
                    input, ibp_bounds, engine, deadline,
                )?)
            }
            (_, Some(ibp_bounds)) => Some(ibp_bounds),
            (_, None) => None,
        };

        if self.config.use_alpha_crown {
            if let Some((threshold, verify_upper)) = threshold_check {
                let fast_bounds = if let Some(layer_bounds) = cached_root_layer_bounds.as_ref() {
                    if use_crown_ibp_layer_bounds {
                        network
                            .propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
                                input,
                                layer_bounds,
                                engine,
                                deadline,
                                self.config.crown_backward_layers,
                            )?
                    } else {
                        network.propagate_crown_with_precomputed_ibp_and_limits(
                            input,
                            layer_bounds.clone(),
                            engine,
                            deadline,
                            self.config.crown_backward_layers,
                        )?
                    }
                } else {
                    network.propagate_crown_with_engine_and_deadline_and_limits(
                        input,
                        engine,
                        deadline,
                        self.config.crown_backward_layers,
                    )?
                };

                let fast_lower = fast_bounds.lower_scalar();
                let fast_upper = fast_bounds.upper_scalar();
                let verified = BetaCrownConfig::domain_is_verified_for_mode(
                    verify_upper,
                    fast_lower,
                    fast_upper,
                    threshold,
                );

                if verified {
                    debug!(
                        "Early termination: fast bounds [{:.2}, {:.2}] already verify threshold {}",
                        fast_lower, fast_upper, threshold
                    );
                    return Ok(InitialBoundsComputation {
                        output_bounds: fast_bounds,
                        root_layer_bounds: cached_root_layer_bounds,
                    });
                }

                debug!(
                    "Fast bounds [{:.2}, {:.2}] don't verify threshold {}, running α-CROWN",
                    fast_lower, fast_upper, threshold
                );
            }

            let mut alpha_config = self.config.alpha_config.clone();
            alpha_config.deadline = deadline;
            let output_bounds = network.propagate_alpha_crown_with_config_and_engine(
                input,
                &alpha_config,
                engine,
            )?;

            // #1817: The sequential `Network` α-CROWN engine produces a looser
            // root bound than the `GraphNetwork` α-CROWN engine for the *same*
            // network and *same* α config (e.g. upper 0.549 vs 0.484 on the
            // three-way comparison net), so the sequential BaB path fails to
            // verify properties the graph path verifies at the root. Both engines
            // compute sound bounds on the same function, so intersecting them
            // (max lower, min upper) is sound and yields a bound at least as tight
            // as either. We adopt the graph engine's tighter root bound here so the
            // sequential path benefits from the same root α-CROWN tightening.
            //
            // Only do this when a threshold is supplied and the sequential bound
            // does NOT already verify: the second α-CROWN pass is only worth its
            // cost when it could flip the root verdict (and otherwise pollutes
            // unrelated work). The result is used solely as a tighter sound bound.
            let needs_graph_tightening =
                threshold_check.is_some_and(|(threshold, verify_upper)| {
                    !BetaCrownConfig::domain_is_verified_for_mode(
                        verify_upper,
                        output_bounds.lower_scalar(),
                        output_bounds.upper_scalar(),
                        threshold,
                    )
                });
            let output_bounds = if needs_graph_tightening {
                self.tighten_root_with_graph_alpha_crown(
                    network,
                    input,
                    &alpha_config,
                    engine,
                    output_bounds,
                )
            } else {
                output_bounds
            };

            return Ok(InitialBoundsComputation {
                output_bounds,
                root_layer_bounds: cached_root_layer_bounds,
            });
        }

        let root_layer_bounds = if let Some(layer_bounds) = cached_root_layer_bounds.take() {
            layer_bounds
        } else if use_crown_ibp_layer_bounds {
            network.collect_crown_ibp_bounds_with_engine_and_deadline(input, engine, deadline)?
        } else {
            network.collect_ibp_bounds_with_deadline(input, deadline)?
        };

        let output_bounds = if use_crown_ibp_layer_bounds {
            network.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
                input,
                &root_layer_bounds,
                engine,
                deadline,
                self.config.crown_backward_layers,
            )?
        } else {
            network.propagate_crown_with_precomputed_ibp_and_limits(
                input,
                root_layer_bounds.clone(),
                engine,
                deadline,
                self.config.crown_backward_layers,
            )?
        };

        Ok(InitialBoundsComputation {
            output_bounds,
            root_layer_bounds: Some(root_layer_bounds),
        })
    }

    /// Compute initial bounds with optional early termination and GPU acceleration.
    ///
    /// `deadline`: If set, alpha-CROWN optimization will bail early when this
    /// wall-clock deadline is exceeded (#2698).
    #[cfg(test)]
    pub(in crate::beta_crown::engine) fn compute_initial_bounds_with_early_exit_engine(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold_check: Option<(f32, bool)>, // (threshold, verify_upper_bound)
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<BoundedTensor> {
        Ok(self
            .compute_initial_bounds_and_layer_bounds_engine(
                network,
                input,
                threshold_check,
                engine,
                deadline,
            )?
            .output_bounds)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ndarray::{arr1, arr2};

    use crate::bounds::LinearBounds;
    use crate::network::GraphNetwork;
    use crate::NETWORK_INPUT;

    fn accumulate_crown_bounds_for_test(
        input_name: &str,
        new_bounds: LinearBounds,
        node_linear_bounds: &mut HashMap<String, LinearBounds>,
        input_accumulated: &mut bool,
    ) {
        if input_name == NETWORK_INPUT {
            if *input_accumulated {
                if let Some(existing) = node_linear_bounds.get_mut(NETWORK_INPUT) {
                    let new_la =
                        GraphNetwork::safe_add(existing.lower_a(), new_bounds.lower_a(), true);
                    let new_lb =
                        GraphNetwork::safe_add(existing.lower_b(), new_bounds.lower_b(), true);
                    let new_ua =
                        GraphNetwork::safe_add(existing.upper_a(), new_bounds.upper_a(), false);
                    let new_ub =
                        GraphNetwork::safe_add(existing.upper_b(), new_bounds.upper_b(), false);
                    *existing.lower_a_mut() = new_la;
                    *existing.lower_b_mut() = new_lb;
                    *existing.upper_a_mut() = new_ua;
                    *existing.upper_b_mut() = new_ub;
                }
            } else {
                node_linear_bounds.insert(NETWORK_INPUT.to_string(), new_bounds);
                *input_accumulated = true;
            }
        } else if let Some(existing) = node_linear_bounds.get_mut(input_name) {
            let new_la = GraphNetwork::safe_add(existing.lower_a(), new_bounds.lower_a(), true);
            let new_lb = GraphNetwork::safe_add(existing.lower_b(), new_bounds.lower_b(), true);
            let new_ua = GraphNetwork::safe_add(existing.upper_a(), new_bounds.upper_a(), false);
            let new_ub = GraphNetwork::safe_add(existing.upper_b(), new_bounds.upper_b(), false);
            *existing.lower_a_mut() = new_la;
            *existing.lower_b_mut() = new_lb;
            *existing.upper_a_mut() = new_ua;
            *existing.upper_b_mut() = new_ub;
        } else {
            node_linear_bounds.insert(input_name.to_string(), new_bounds);
        }
    }

    /// Test for #2102: INF-cancellation during accumulate_crown_bounds must produce
    /// sound conservative bounds (NEG_INFINITY/INFINITY), not NaN.
    ///
    /// Analogous to test_accumulate_crown_ibp_bounds_nan_safe_2093 in graph_alpha,
    /// but exercises the beta_crown engine's accumulate_crown_bounds path.
    #[test]
    fn test_accumulate_crown_bounds_inf_cancellation_safe_2102() {
        // Simulate INF + (-INF) cancellation during bound accumulation.
        // Without NaN-safe addition, this produces NaN that corrupts downstream bounds.
        let existing = LinearBounds {
            lower_a: arr2(&[[f32::NEG_INFINITY, 1.0]]),
            lower_b: arr1(&[f32::NEG_INFINITY]),
            upper_a: arr2(&[[f32::INFINITY, 2.0]]),
            upper_b: arr1(&[f32::INFINITY]),
            lower_a_err: None,
            upper_a_err: None,
        };
        let new_bounds = LinearBounds {
            lower_a: arr2(&[[f32::INFINITY, 3.0]]),
            lower_b: arr1(&[f32::INFINITY]),
            upper_a: arr2(&[[f32::NEG_INFINITY, 4.0]]),
            upper_b: arr1(&[f32::NEG_INFINITY]),
            lower_a_err: None,
            upper_a_err: None,
        };

        // Seed the map with existing bounds, then accumulate (input_accumulated=true)
        let mut node_linear_bounds = HashMap::new();
        node_linear_bounds.insert(NETWORK_INPUT.to_string(), existing);
        let mut input_accumulated = true;

        accumulate_crown_bounds_for_test(
            NETWORK_INPUT,
            new_bounds,
            &mut node_linear_bounds,
            &mut input_accumulated,
        );

        let result = node_linear_bounds
            .get(NETWORK_INPUT)
            .expect("_input key must exist after accumulation");

        // INF + (-INF) = NaN under IEEE 754. NaN-safe addition should recover:
        assert_eq!(
            result.lower_a[[0, 0]],
            f32::NEG_INFINITY,
            "lower_a INF-cancellation should recover to NEG_INFINITY, not NaN"
        );
        assert_eq!(
            result.lower_b[0],
            f32::NEG_INFINITY,
            "lower_b INF-cancellation should recover to NEG_INFINITY, not NaN"
        );
        assert_eq!(
            result.upper_a[[0, 0]],
            f32::INFINITY,
            "upper_a INF-cancellation should recover to INFINITY, not NaN"
        );
        assert_eq!(
            result.upper_b[0],
            f32::INFINITY,
            "upper_b INF-cancellation should recover to INFINITY, not NaN"
        );

        // Normal additions should still work correctly
        assert!(
            (result.lower_a[[0, 1]] - 4.0).abs() < 1e-6,
            "Normal lower_a addition should produce 1.0 + 3.0 = 4.0, got {}",
            result.lower_a[[0, 1]]
        );
        assert!(
            (result.upper_a[[0, 1]] - 6.0).abs() < 1e-6,
            "Normal upper_a addition should produce 2.0 + 4.0 = 6.0, got {}",
            result.upper_a[[0, 1]]
        );
    }

    /// Test for #2102: NaN from upstream must be replaced with conservative infinity.
    ///
    /// safe_add_* replaces all NaN (whether from upstream or from INF cancellation)
    /// with conservative bounds: NEG_INFINITY for lower, INFINITY for upper.
    /// This is sound because NaN is not comparable and would corrupt downstream
    /// min/max operations. See safe_add in graph_crown/utils.rs.
    #[test]
    fn test_accumulate_crown_bounds_nan_input_preserved_2102() {
        // NaN in existing bounds should become conservative infinity after accumulation.
        let existing = LinearBounds {
            lower_a: arr2(&[[f32::NAN]]),
            lower_b: arr1(&[1.0]),
            upper_a: arr2(&[[2.0]]),
            upper_b: arr1(&[f32::NAN]),
            lower_a_err: None,
            upper_a_err: None,
        };
        let new_bounds = LinearBounds {
            lower_a: arr2(&[[5.0]]),
            lower_b: arr1(&[3.0]),
            upper_a: arr2(&[[4.0]]),
            upper_b: arr1(&[6.0]),
            lower_a_err: None,
            upper_a_err: None,
        };

        // Use intermediate node name (not NETWORK_INPUT) to exercise that branch
        let mut node_linear_bounds = HashMap::new();
        node_linear_bounds.insert("node1".to_string(), existing);
        let mut input_accumulated = false;

        accumulate_crown_bounds_for_test(
            "node1",
            new_bounds,
            &mut node_linear_bounds,
            &mut input_accumulated,
        );

        let result = node_linear_bounds
            .get("node1")
            .expect("node1 key must exist after accumulation");

        // NaN input is replaced with conservative infinity by safe_add_*
        // (NEG_INFINITY for lower bounds, INFINITY for upper bounds).
        // This is the sound behavior: NaN is not a valid bound, so we widen
        // to the most conservative value. See safe_add in
        // network/graph_crown/utils.rs.
        assert_eq!(
            result.lower_a[[0, 0]],
            f32::NEG_INFINITY,
            "NaN in lower_a should become NEG_INFINITY (conservative lower bound)"
        );
        assert_eq!(
            result.upper_b[0],
            f32::INFINITY,
            "NaN in upper_b should become INFINITY (conservative upper bound)"
        );
        // Non-NaN additions should still work
        assert!(
            (result.lower_b[0] - 4.0).abs() < 1e-6,
            "Normal lower_b: 1.0 + 3.0 = 4.0, got {}",
            result.lower_b[0]
        );
        assert!(
            (result.upper_a[[0, 0]] - 6.0).abs() < 1e-6,
            "Normal upper_a: 2.0 + 4.0 = 6.0, got {}",
            result.upper_a[[0, 0]]
        );
    }
}
