// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph BaBSR score computation.
//!
//! Extracted from `graph.rs` to keep file sizes under 500 lines.
//! These methods keep the signed lA columns alive long enough to compute
//! bias-aware BaBSR score parts at each graph activation site.

use super::*;

impl BetaCrownVerifier {
    /// Compute objective-directed BaBSR scores from the exact `lA` captured by
    /// the domain's preceding CROWN pass.
    ///
    /// Alpha-beta-CROWN and NeuralSAT both pre-rank kFSB candidates from the
    /// per-domain `lA` that produced the current objective bound.  The graph
    /// lane historically discarded that signal and re-ran a fixed-upper-slope
    /// coefficient proxy instead.  A `MultiObjectiveGraphBabDomain` already
    /// stores the winner-compatible signal in its per-objective
    /// `CachedLinearBounds`; this helper turns the cached lower-A row at each
    /// splittable activation into the existing, shared BaBSR score formula.
    ///
    /// Missing/malformed cache entries are omitted.  The dark-gated caller
    /// fills every omission from the historical proxy, so this helper can only
    /// affect advisory ranking where a complete finite cached column exists.
    pub(in crate::beta_crown::engine) fn compute_graph_babsr_scores_from_cached_la<'a>(
        &self,
        graph: &GraphNetwork,
        node_bounds: impl Into<NodeBoundsView<'a>>,
        input_bounds: &BoundedTensor,
        cached_la: &crate::batched_domain::CachedLinearBounds,
        reduce_op: KfsbReduceOp,
        only_nodes: &std::collections::HashSet<String>,
    ) -> std::collections::HashMap<(String, usize), BabsrScoreParts> {
        let node_bounds = node_bounds.into();
        let mut scores = std::collections::HashMap::new();

        for node_name in only_nodes {
            let Some(node) = graph.node(node_name) else {
                continue;
            };
            if !is_zero_threshold_binary_activation(node.layer()) {
                continue;
            }
            let Some(pre_name) = node.inputs().first() else {
                continue;
            };
            let pre_bounds = if pre_name == NETWORK_INPUT {
                Some(input_bounds)
            } else {
                node_bounds.get(pre_name).map(AsRef::as_ref)
            };
            let Some(pre_bounds) = pre_bounds else {
                continue;
            };
            let Some(coeffs) = cached_la.lower_a.get(node_name) else {
                continue;
            };
            let flat = pre_bounds.flatten();
            // A multi-objective domain cache is split into one row per
            // objective before it reaches this selector.  Fail closed to the
            // historical proxy when that invariant or the activation width
            // does not match, rather than averaging an unrelated row or
            // silently truncating a malformed cache.
            if coeffs.nrows() != 1 || coeffs.ncols() != flat.len() {
                continue;
            }

            let bias_flat = self
                .graph_preact_bias(graph, pre_name, pre_bounds.shape())
                .and_then(|bias| bias.into_shape_with_order((flat.len(),)).ok());
            for neuron_idx in 0..flat.len() {
                let coeff_column = coeffs.column(neuron_idx);
                if coeff_column.iter().any(|coeff| !coeff.is_finite()) {
                    continue;
                }
                let lower = flat.lower()[[neuron_idx]];
                let upper = flat.upper()[[neuron_idx]];
                let bias = bias_flat
                    .as_ref()
                    .map(|flat_bias| flat_bias[neuron_idx])
                    .unwrap_or(0.0);
                scores.insert(
                    (node_name.clone(), neuron_idx),
                    compute_babsr_score_parts(coeff_column, lower, upper, bias, reduce_op),
                );
            }
        }

        scores
    }

    /// Compute BaBSR score parts for graph branching while the signed lA columns
    /// are still available.
    pub(in crate::beta_crown::engine) fn compute_graph_babsr_scores_from_bounds<'a>(
        &self,
        graph: &GraphNetwork,
        node_bounds: impl Into<NodeBoundsView<'a>>,
        input_bounds: &BoundedTensor,
        reduce_op: KfsbReduceOp,
        // #branching-la INC1: when `Some(c)`, seed the coefficient backward with the
        // worst-straggler OBJECTIVE margin row `c` (length output_dim) instead of the
        // identity — so scores measure each neuron's signed influence on the RELEVANT
        // objective (objective-directed BaBSR), not a 100-way-diluted average. `None`
        // preserves the legacy eye seed (all existing callers). Advisory-only (score
        // ranks candidates; never read by any verdict) so this is soundness-free.
        objective_seed: Option<&[f32]>,
        // #branching-la stop-early: `Some(nodes)` = the unstable ReLU node set; STOP the
        // coefficient backward once all are scored — the remaining (input-side) layers have
        // no candidate to score, so their (often largest-spatial) conv adjoints are skipped.
        // `None` = full backward (legacy). Advisory-only ⇒ soundness-free.
        only_nodes: Option<&std::collections::HashSet<String>>,
    ) -> Result<std::collections::HashMap<(String, usize), BabsrScoreParts>> {
        let node_bounds = node_bounds.into();
        self.compute_graph_babsr_scores_from_bounds_impl(
            graph,
            node_bounds,
            input_bounds,
            reduce_op,
            objective_seed,
            only_nodes,
            None,
            None,
        )
    }

    /// #joint-interm-grad: the objective adjoint at every ReLU's PRE-ACTIVATION
    /// producer, harvested from the same walk that computes BaBSR scores.
    ///
    /// This is the `df/d(pre-activation)` half of the indirect alpha-gradient
    /// term. The walk already builds this matrix in order to score with it; the
    /// only change is that a sink keeps it instead of dropping it, so there is no
    /// extra propagation and no new kernel. Seeding with the objective rows makes
    /// the recorded matrix the adjoint OF THE OBJECTIVE, which is what the
    /// sensitivity weights require.
    ///
    /// Advisory-only, exactly like the scores it rides along with: the result
    /// steers which alpha the ascent lands on and is never read by a bound or a
    /// verdict.
    pub(in crate::beta_crown::engine) fn objective_adjoints_at_preactivations<'a>(
        &self,
        graph: &GraphNetwork,
        node_bounds: impl Into<NodeBoundsView<'a>>,
        input_bounds: &BoundedTensor,
        reduce_op: KfsbReduceOp,
        objective_seed: Option<&[f32]>,
        deadline: std::time::Instant,
    ) -> Result<std::collections::HashMap<String, Array2<f32>>> {
        let node_bounds = node_bounds.into();
        let mut sink = std::collections::HashMap::new();
        self.compute_graph_babsr_scores_from_bounds_impl(
            graph,
            node_bounds,
            input_bounds,
            reduce_op,
            objective_seed,
            None,
            Some(deadline),
            Some(&mut sink),
        )?;
        Ok(sink)
    }

    /// Deadline-aware form used only by bounded research observers.
    ///
    /// The historical branching path calls
    /// [`Self::compute_graph_babsr_scores_from_bounds`] and therefore retains
    /// its deadline-free behavior.  Shadow callers pass their own private
    /// deadline here; expiry before, between, or immediately after graph-node
    /// coefficient steps fails closed with no partial score map.
    pub(in crate::beta_crown::engine) fn compute_graph_babsr_scores_from_bounds_until<'a>(
        &self,
        graph: &GraphNetwork,
        node_bounds: impl Into<NodeBoundsView<'a>>,
        input_bounds: &BoundedTensor,
        reduce_op: KfsbReduceOp,
        objective_seed: Option<&[f32]>,
        only_nodes: Option<&std::collections::HashSet<String>>,
        deadline: std::time::Instant,
    ) -> Result<std::collections::HashMap<(String, usize), BabsrScoreParts>> {
        let node_bounds = node_bounds.into();
        self.compute_graph_babsr_scores_from_bounds_impl(
            graph,
            node_bounds,
            input_bounds,
            reduce_op,
            objective_seed,
            only_nodes,
            Some(deadline),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_graph_babsr_scores_from_bounds_impl(
        &self,
        graph: &GraphNetwork,
        node_bounds: NodeBoundsView<'_>,
        input_bounds: &BoundedTensor,
        reduce_op: KfsbReduceOp,
        objective_seed: Option<&[f32]>,
        only_nodes: Option<&std::collections::HashSet<String>>,
        deadline: Option<std::time::Instant>,
        // #joint-interm-grad: when `Some`, record the objective adjoint at each
        // ReLU's PRE-ACTIVATION producer. The walk already materialises exactly
        // this matrix to score with; the sink just stops it being discarded.
        // `None` is byte-identical to the historical walk.
        mut adjoint_sink: Option<&mut std::collections::HashMap<String, Array2<f32>>>,
    ) -> Result<std::collections::HashMap<(String, usize), BabsrScoreParts>> {
        let check_deadline = || {
            if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                Err(ny_core::NyError::DeadlineExceeded(
                    "branch-specific BaBSR shadow scoring exceeded its private deadline"
                        .to_string(),
                ))
            } else {
                Ok(())
            }
        };
        check_deadline()?;
        let mut scores = std::collections::HashMap::new();
        let mut remaining: Option<std::collections::HashSet<String>> = only_nodes.cloned();
        let node_names = graph.exec_order()?;
        if node_names.is_empty() {
            return Ok(scores);
        }

        let output_name = graph.output_name();
        let output_dim = node_bounds
            .get(output_name)
            .map(|bounds| bounds.len())
            .ok_or_else(|| {
                ny_core::NyError::InternalError(format!(
                    "compute_graph_babsr_scores_from_bounds: missing output bounds for node '{}'",
                    output_name
                ))
            })?;

        let mut node_coeffs: std::collections::HashMap<String, Array2<f32>> =
            std::collections::HashMap::new();
        // #branching-la INC1: seed with the objective margin row `c` (1 x output_dim) when
        // provided, else the identity (legacy). A mismatched `c` falls back to identity.
        let seed = match objective_seed {
            Some(c) if c.len() == output_dim => Array2::from_shape_vec((1, output_dim), c.to_vec())
                .unwrap_or_else(|_| Array2::<f32>::eye(output_dim)),
            _ => Array2::<f32>::eye(output_dim),
        };
        node_coeffs.insert(output_name.to_string(), seed);

        for node_name in node_names.iter().rev() {
            check_deadline()?;
            let current = match node_coeffs.get(node_name) {
                Some(coeffs) => coeffs.clone(),
                None => continue,
            };

            let node = match graph.node(node_name) {
                Some(node) => node,
                None => continue,
            };

            match &node.layer {
                Layer::Linear(linear) => {
                    let new_coeffs = current.dot(linear.weight());
                    if let Some(input_name) = node.inputs.first() {
                        if input_name != NETWORK_INPUT {
                            let entry =
                                node_coeffs.entry(input_name.clone()).or_insert_with(|| {
                                    Array2::zeros((new_coeffs.nrows(), new_coeffs.ncols()))
                                });
                            if entry.shape() == new_coeffs.shape() {
                                *entry = &*entry + &new_coeffs;
                            } else {
                                *entry = new_coeffs;
                            }
                        }
                    }
                }
                Layer::ReLU(_) | Layer::Sign(_) => {
                    let pre_name = match node.inputs.first() {
                        Some(name) => name.as_str(),
                        None => {
                            tracing::warn!(
                                node = %node_name,
                                "ReLU/Sign node has empty inputs — skipping score computation"
                            );
                            continue;
                        }
                    };
                    let pre_bounds: Option<&BoundedTensor> = if pre_name == NETWORK_INPUT {
                        Some(input_bounds)
                    } else {
                        node_bounds.get(pre_name).map(|bounds| bounds.as_ref())
                    };

                    if let Some(sink) = adjoint_sink.as_deref_mut() {
                        sink.insert(pre_name.to_string(), current.clone());
                    }
                    if let Some(bounds) = pre_bounds {
                        let flat = bounds.flatten();
                        let num_neurons = current.ncols().min(flat.len());
                        let bias_flat = self
                            .graph_preact_bias(graph, pre_name, bounds.shape())
                            .and_then(|bias| bias.into_shape_with_order((flat.len(),)).ok());
                        if bias_flat.is_none() {
                            tracing::debug!(
                                relu = %node_name,
                                producer = %pre_name,
                                "BaBSR graph: unrecoverable producer bias, using 0.0 fallback"
                            );
                        }

                        for neuron_idx in 0..num_neurons {
                            let lower = flat.lower()[[neuron_idx]];
                            let upper = flat.upper()[[neuron_idx]];
                            let bias = bias_flat
                                .as_ref()
                                .map(|flat_bias| flat_bias[neuron_idx])
                                .unwrap_or(0.0);
                            scores.insert(
                                (node_name.clone(), neuron_idx),
                                compute_babsr_score_parts(
                                    current.column(neuron_idx),
                                    lower,
                                    upper,
                                    bias,
                                    reduce_op,
                                ),
                            );
                        }

                        let is_sign = matches!(&node.layer, Layer::Sign(_));
                        let mut new_coeffs = Array2::<f32>::zeros((current.nrows(), num_neurons));
                        for neuron_idx in 0..num_neurons {
                            let l = flat.lower()[[neuron_idx]];
                            let u = flat.upper()[[neuron_idx]];
                            let slope = if is_sign {
                                sign_fixed_crown_proxy_slope(l, u)
                            } else {
                                relu_upper_slope(l, u)
                            };
                            for row in 0..current.nrows() {
                                new_coeffs[[row, neuron_idx]] = current[[row, neuron_idx]] * slope;
                            }
                        }

                        if let Some(input_name) = node.inputs.first() {
                            if input_name != NETWORK_INPUT {
                                let entry =
                                    node_coeffs.entry(input_name.clone()).or_insert_with(|| {
                                        Array2::zeros((new_coeffs.nrows(), new_coeffs.ncols()))
                                    });
                                if entry.shape() == new_coeffs.shape() {
                                    *entry = &*entry + &new_coeffs;
                                } else {
                                    *entry = new_coeffs;
                                }
                            }
                        }
                    }
                }
                // #branching-la INC2: real transpose-conv adjoint so coefficients reach
                // post-conv ReLUs with correct shape/magnitude (the legacy `_` arm treats
                // Conv2d as identity => misaligned columns => garbage lA behind every
                // conv). Reuses the tested conv coeff backward; groups!=1 or missing
                // input_shape => the sound passthrough fallback. Advisory-only (score
                // ranks candidates; never read by any verdict) so this is soundness-free.
                Layer::Conv2d(conv) if conv.input_shape.is_some() => {
                    // The Conv2d layer's coefficient backward is a TRANSPOSE convolution.
                    // The historical lane uses the f64 GEMM coefficient path; a bounded
                    // observer uses the pollable scalar-f64 sibling so no opaque
                    // faer/engine chunk can monopolize the authority deadline. Compute the conv
                    // OUTPUT spatial dims from the input dims + geometry.
                    let (in_h, in_w) = conv.input_shape.expect("is_some checked");
                    let ksh = conv.kernel.shape();
                    let (out_c, kh, kw) = (ksh[0], ksh[2], ksh[3]);
                    let eff_kh = conv.dilation.0 * (kh - 1) + 1;
                    let eff_kw = conv.dilation.1 * (kw - 1) + 1;
                    let out_h =
                        (in_h + 2 * conv.padding.0).saturating_sub(eff_kh) / conv.stride.0 + 1;
                    let out_w =
                        (in_w + 2 * conv.padding.1).saturating_sub(eff_kw) / conv.stride.1 + 1;
                    let propagated = if let Some(deadline) = deadline {
                        match crate::layers::convolution::conv2d::conv2d_transpose_backward_coeff_f64_with_deadline(
                            &current,
                            &conv.kernel,
                            conv.stride,
                            conv.padding,
                            conv.dilation,
                            (in_h, in_w),
                            (out_h, out_w),
                            out_c,
                            conv.groups,
                            1, // advisory propagation retains one result
                            Some(deadline),
                        ) {
                            Ok(propagated) => propagated.mapv(|value| value as f32),
                            Err(error) if error.is_deadline_exceeded() => return Err(error),
                            Err(_) => current.clone(), // shape/overflow => passthrough
                        }
                    } else {
                        crate::layers::convolution::conv2d::conv2d_transpose_backward_coeff_f64(
                            &current,
                            &conv.kernel,
                            conv.stride,
                            conv.padding,
                            conv.dilation,
                            (in_h, in_w),
                            (out_h, out_w),
                            out_c,
                            conv.groups,
                            1,
                        )
                        .map(|back_f64| back_f64.mapv(|v| v as f32))
                        .unwrap_or_else(|_| current.clone()) // shape/overflow => passthrough
                    };
                    if let Some(input_name) = node.inputs.first() {
                        if input_name != NETWORK_INPUT {
                            let entry =
                                node_coeffs.entry(input_name.clone()).or_insert_with(|| {
                                    Array2::zeros((propagated.nrows(), propagated.ncols()))
                                });
                            if entry.shape() == propagated.shape() {
                                *entry = &*entry + &propagated;
                            } else {
                                *entry = propagated;
                            }
                        }
                    }
                }
                _ => {
                    for input_name in &node.inputs {
                        if input_name != NETWORK_INPUT {
                            let entry =
                                node_coeffs.entry(input_name.clone()).or_insert_with(|| {
                                    Array2::zeros((current.nrows(), current.ncols()))
                                });
                            if entry.shape() == current.shape() {
                                *entry = &*entry + &current;
                            } else {
                                *entry = current.clone();
                            }
                        }
                    }
                }
            }
            check_deadline()?;
            // #branching-la stop-early: once every requested unstable ReLU has been scored,
            // the input-side layers below have no candidate — skip their conv adjoints.
            if let Some(rem) = remaining.as_mut() {
                rem.remove(node_name);
                if rem.is_empty() {
                    break;
                }
            }
        }

        check_deadline()?;
        Ok(scores)
    }

    pub(in crate::beta_crown::engine) fn compute_graph_babsr_scores(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        reduce_op: KfsbReduceOp,
    ) -> Result<std::collections::HashMap<(String, usize), BabsrScoreParts>> {
        self.compute_graph_babsr_scores_from_bounds(
            graph,
            &domain.node_bounds,
            &domain.input_bounds,
            reduce_op,
            None, // single-objective wrapper: legacy identity seed
            None, // full backward (no stop-early)
        )
    }

    pub(in crate::beta_crown::engine) fn compute_graph_babsr_intercept_only_scores(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
    ) -> Result<std::collections::HashMap<(String, usize), f32>> {
        let mut scores = std::collections::HashMap::new();
        let node_names = graph.exec_order()?;
        if node_names.is_empty() {
            return Ok(scores);
        }

        let output_name = graph.output_name();
        let output_dim = domain
            .node_bounds
            .get(output_name)
            .map(|bounds| bounds.len())
            .ok_or_else(|| {
                ny_core::NyError::InternalError(format!(
                    "compute_graph_babsr_intercept_only_scores: missing output bounds for node '{}'",
                    output_name
                ))
            })?;

        let mut node_coeffs: std::collections::HashMap<String, Array2<f32>> =
            std::collections::HashMap::new();
        node_coeffs.insert(output_name.to_string(), Array2::<f32>::eye(output_dim));

        for node_name in node_names.iter().rev() {
            let current = match node_coeffs.get(node_name) {
                Some(coeffs) => coeffs.clone(),
                None => continue,
            };

            let node = match graph.node(node_name) {
                Some(node) => node,
                None => continue,
            };

            match &node.layer {
                Layer::Linear(linear) => {
                    let new_coeffs = current.dot(linear.weight());
                    if let Some(input_name) = node.inputs.first() {
                        if input_name != NETWORK_INPUT {
                            let entry =
                                node_coeffs.entry(input_name.clone()).or_insert_with(|| {
                                    Array2::zeros((new_coeffs.nrows(), new_coeffs.ncols()))
                                });
                            if entry.shape() == new_coeffs.shape() {
                                *entry = &*entry + &new_coeffs;
                            } else {
                                *entry = new_coeffs;
                            }
                        }
                    }
                }
                Layer::ReLU(_) | Layer::Sign(_) => {
                    let pre_name = match node.inputs.first() {
                        Some(name) => name.as_str(),
                        None => {
                            tracing::warn!(
                                node = %node_name,
                                "ReLU/Sign node has empty inputs — skipping intercept-only score computation"
                            );
                            continue;
                        }
                    };
                    let pre_bounds: Option<&BoundedTensor> = if pre_name == NETWORK_INPUT {
                        Some(domain.input_bounds.as_ref())
                    } else {
                        domain
                            .node_bounds
                            .get(pre_name)
                            .map(|bounds| bounds.as_ref())
                    };

                    if let Some(bounds) = pre_bounds {
                        let flat = bounds.flatten();
                        let num_neurons = current.ncols().min(flat.len());

                        for neuron_idx in 0..num_neurons {
                            scores.insert(
                                (node_name.clone(), neuron_idx),
                                compute_babsr_intercept_only_score(
                                    current.column(neuron_idx),
                                    flat.lower()[[neuron_idx]],
                                    flat.upper()[[neuron_idx]],
                                ),
                            );
                        }

                        let is_sign = matches!(&node.layer, Layer::Sign(_));
                        let mut new_coeffs = Array2::<f32>::zeros((current.nrows(), num_neurons));
                        for neuron_idx in 0..num_neurons {
                            let l = flat.lower()[[neuron_idx]];
                            let u = flat.upper()[[neuron_idx]];
                            let slope = if is_sign {
                                sign_fixed_crown_proxy_slope(l, u)
                            } else {
                                relu_upper_slope(l, u)
                            };
                            for row in 0..current.nrows() {
                                new_coeffs[[row, neuron_idx]] = current[[row, neuron_idx]] * slope;
                            }
                        }

                        if let Some(input_name) = node.inputs.first() {
                            if input_name != NETWORK_INPUT {
                                let entry =
                                    node_coeffs.entry(input_name.clone()).or_insert_with(|| {
                                        Array2::zeros((new_coeffs.nrows(), new_coeffs.ncols()))
                                    });
                                if entry.shape() == new_coeffs.shape() {
                                    *entry = &*entry + &new_coeffs;
                                } else {
                                    *entry = new_coeffs;
                                }
                            }
                        }
                    }
                }
                _ => {
                    for input_name in &node.inputs {
                        if input_name != NETWORK_INPUT {
                            let entry =
                                node_coeffs.entry(input_name.clone()).or_insert_with(|| {
                                    Array2::zeros((current.nrows(), current.ncols()))
                                });
                            if entry.shape() == current.shape() {
                                *entry = &*entry + &current;
                            } else {
                                *entry = current.clone();
                            }
                        }
                    }
                }
            }
        }

        Ok(scores)
    }
}
