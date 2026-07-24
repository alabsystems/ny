// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Preset-gated softmax "complex" decomposition (vit_2023, W2 Phase B).
//!
//! Rewrites each `Layer::Softmax` node into the primitive subgraph
//!
//! ```text
//!   x ── SubConstant(c) ── Exp ── ReduceSum(axis, keepdims) ── Reciprocal ─┐
//!               │                                                          │
//!               └────────────────── Exp ────────────── MulBinary(exp, recip)
//! ```
//!
//! i.e. `softmax(x) = exp(x - c) * (1 / Σ_axis exp(x - c))`, which holds
//! EXACTLY for any finite constant `c` (the shift cancels between numerator
//! and denominator). The rewrite converts the softmax bound from the fixed
//! center-point LSE affine relaxation (`layers/softmax/linear/sound.rs`, no
//! learnable parameters) into a composition of primitives that all carry
//! sound CROWN relaxations and, critically, optimizable alpha state in the
//! DAG alpha loop: Reciprocal tangent points (`ReciprocalAlpha`) and
//! MulBinary alphas (#3439). This is the analog of alpha-beta-CROWN's
//! `bound_opts={'softmax': 'complex'}` (vnncomp23 vit winner recipe).
//!
//! # The shift constant
//!
//! `c` is the rowwise (along the softmax axis) maximum of the INTERVAL
//! CENTER of the softmax input, computed from one plain IBP forward pass at
//! rewrite time (after the instance's input box is known, before
//! verification starts). A load-time constant computed without input bounds
//! would be arbitrary; computing it from the root box recenters the Exp
//! domain so `upper(x - c) <= max_i radius_i`, keeping f32 `exp` away from
//! overflow for exactly the instances where decomposed bounds can help.
//! The constant is then FROZEN into the graph as a `SubConstant` node:
//! every subsequent propagation (alpha iterations, BaB subdomains — whose
//! boxes only shrink) sees the same fixed function, so forward/backward
//! passes are always mutually consistent. Soundness does not depend on the
//! VALUE of `c`, only on it being a finite constant.
//!
//! # Fallback policy (soundness first)
//!
//! - Rewrite-time: a node is SKIPPED (kept as direct-LSE `Softmax`) when its
//!   input bounds are unavailable/non-finite, or when the shifted upper bound
//!   exceeds [`SOFTMAX_COMPLEX_SHIFT_GUARD`] (margin below the f32 `exp`
//!   overflow threshold of 88): `ExpLayer::propagate_ibp` ERRORS above that
//!   threshold, which would abort the whole IBP forward pass instead of
//!   degrading. Skipping preserves today's behavior exactly for that node.
//! - Run-time: the decomposed pattern is already recognized by the CROWN
//!   backward (`is_softmax_decomposition_mul`); any non-finite coefficient at
//!   the MulBinary triggers `MulBinaryDispatchResult::SoftmaxNonFinite` →
//!   sound IBP fallback, and Exp/Reciprocal CROWN domain-guard errors degrade
//!   to sound per-node IBP concretization (#3596). The NaN firewall remains
//!   the backstop but is not relied upon.
//!
//! Rewrites are opt-in per preset (`solver.alpha-crown.softmax: complex`) and
//! can be force-disabled with `NY_NO_SOFTMAX_COMPLEX=1` (disable-flag
//! principle).

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use crate::layers::{
    ExpLayer, Layer, MulBinaryLayer, ReciprocalLayer, ReduceSumLayer, SubConstantLayer,
};

use super::{GraphNetwork, GraphNode, NETWORK_INPUT};

/// Maximum allowed post-shift upper bound at the Exp input.
///
/// `ExpLayer::propagate_ibp` rejects upper bounds above 88.0 (f32 exp
/// overflow) with a hard `NumericalInstability` error, which would abort the
/// forward IBP pass for the whole graph. 80.0 leaves an 8-unit margin so
/// rounding/reference-bound drift can never cross the hard threshold.
/// With `c` = rowwise max of the interval center, the post-shift upper bound
/// equals at most `max_i radius_i` over the row, so this is effectively a
/// gate on the input-interval radius.
pub const SOFTMAX_COMPLEX_SHIFT_GUARD: f64 = 80.0;

/// Outcome of [`GraphNetwork::decompose_softmax_complex`].
#[derive(Debug, Default)]
pub struct SoftmaxComplexReport {
    /// Softmax nodes replaced by the primitive subgraph.
    pub decomposed: Vec<String>,
    /// Softmax nodes left untouched (direct-LSE path), with the reason.
    pub skipped: Vec<(String, String)>,
}

/// Per-node rewrite plan computed before any graph mutation.
struct NodePlan {
    node_name: String,
    input_name: String,
    /// Resolved non-negative softmax axis.
    axis: usize,
    /// Full-shape shift constant (already broadcast along the softmax axis).
    shift: ArrayD<f32>,
    /// Shape of the softmax input (= output) in the propagation convention.
    full_shape: Vec<usize>,
}

impl GraphNetwork {
    /// Rewrite every eligible `Softmax` node into the alpha-optimizable
    /// primitive subgraph `SubConstant → Exp → ReduceSum → Reciprocal →
    /// MulBinary` (see module docs).
    ///
    /// `input` is the verification instance's input box; one plain IBP
    /// forward pass over the CURRENT graph supplies the softmax-input
    /// interval centers for the shift constants. Nodes whose bounds are
    /// unavailable, non-finite, or too wide for f32 `exp` are skipped and
    /// keep the existing direct-LSE softmax relaxation. Errors from the
    /// bounds pass are absorbed into "skip all" (the graph is then returned
    /// unchanged) so enabling the rewrite can never make an instance fail
    /// that would have run without it.
    pub fn decompose_softmax_complex(
        &mut self,
        input: &BoundedTensor,
    ) -> Result<SoftmaxComplexReport> {
        let mut report = SoftmaxComplexReport::default();

        let candidates: Vec<String> = self
            .node_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .is_some_and(|node| matches!(node.layer, Layer::Softmax(_)))
            })
            .cloned()
            .collect();
        if candidates.is_empty() {
            return Ok(report);
        }

        // One plain IBP forward pass for interval centers. On failure keep the
        // graph unchanged: the rewrite must never introduce a new failure mode.
        let node_bounds = match self.collect_node_bounds_with_engine_and_deadline(input, None, None)
        {
            Ok(bounds) => bounds,
            Err(err) => {
                debug!(
                    "softmax-complex: IBP bounds pass failed ({}); keeping all {} Softmax nodes on the direct-LSE path",
                    err,
                    candidates.len()
                );
                for name in candidates {
                    report
                        .skipped
                        .push((name, format!("IBP bounds unavailable: {err}")));
                }
                return Ok(report);
            }
        };

        let mut plans: Vec<NodePlan> = Vec::new();
        for name in candidates {
            match self.plan_softmax_complex_node(&name, input, &node_bounds) {
                Ok(plan) => plans.push(plan),
                Err(reason) => {
                    debug!(
                        "softmax-complex: keeping '{}' on the direct-LSE path: {}",
                        name, reason
                    );
                    report.skipped.push((name, reason));
                }
            }
        }

        for plan in plans {
            self.apply_softmax_complex_plan(&plan)?;
            info!(
                "softmax-complex: decomposed '{}' into SubConstant→Exp→ReduceSum→Reciprocal→MulBinary (axis {}, shape {:?})",
                plan.node_name, plan.axis, plan.full_shape
            );
            report.decomposed.push(plan.node_name);
        }
        Ok(report)
    }

    /// Validate one Softmax node and compute its rewrite plan.
    ///
    /// Returns `Err(reason)` (a human-readable skip reason, NOT a hard error)
    /// when the node must stay on the direct-LSE path.
    fn plan_softmax_complex_node(
        &self,
        name: &str,
        input: &BoundedTensor,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    ) -> std::result::Result<NodePlan, String> {
        let node = self
            .nodes
            .get(name)
            .ok_or_else(|| "node disappeared during planning".to_string())?;
        let Layer::Softmax(softmax) = &node.layer else {
            return Err("not a Softmax node".to_string());
        };
        let input_name = node
            .require_unary_input()
            .map_err(|e| format!("malformed inputs: {e}"))?
            .to_string();

        let in_bounds = if input_name == NETWORK_INPUT {
            input
        } else {
            node_bounds
                .get(&input_name)
                .ok_or_else(|| format!("no IBP bounds for input '{input_name}'"))?
        };

        let shape = in_bounds.shape().to_vec();
        let rank = shape.len();
        if rank == 0 {
            return Err("scalar softmax input".to_string());
        }
        let axis = if softmax.axis < 0 {
            softmax.axis + rank as i32
        } else {
            softmax.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(format!(
                "axis {} out of range for rank {}",
                softmax.axis, rank
            ));
        }
        let axis = axis as usize;

        let lower = in_bounds.lower();
        let upper = in_bounds.upper();
        if lower.iter().chain(upper.iter()).any(|v| !v.is_finite()) {
            return Err("non-finite softmax input bounds".to_string());
        }

        // c = rowwise max (along the softmax axis) of the interval center,
        // computed in f64 and materialized at FULL input shape so the
        // SubConstant CROWN backward takes the exact `c_flat.len() ==
        // layer_dim` dot-product path (no broadcast-expansion fallback).
        // Bit-identical to `0.5 * (l + u)`: finite f32-cast operands stay on
        // f64::midpoint's non-overflow `(a + b) * 0.5` path.
        let center = ndarray::Zip::from(lower)
            .and(upper)
            .map_collect(|&l, &u| f64::midpoint(l as f64, u as f64));
        let row_max =
            center.fold_axis(ndarray::Axis(axis), f64::NEG_INFINITY, |acc, &v| acc.max(v));
        // Broadcast the reduced row-max back to full shape via keepdims-style
        // insert + broadcast, then cast to f32 (round-to-nearest is fine: any
        // finite constant is exact for soundness; only conditioning matters).
        let row_max_keep = row_max.insert_axis(ndarray::Axis(axis));
        let shift_f64 = row_max_keep
            .broadcast(shape.as_slice())
            .ok_or_else(|| "row-max broadcast failed".to_string())?
            .to_owned();
        let shift: ArrayD<f32> = shift_f64.mapv(|v| v as f32);
        if shift.iter().any(|v| !v.is_finite()) {
            return Err("non-finite shift constant".to_string());
        }

        // Exp-overflow gate against the ACTUAL f32 constant that will be
        // subtracted: max_i (u_i - c_i) must stay clear of the f32 exp
        // overflow threshold or ExpLayer::propagate_ibp hard-errors and
        // aborts the whole forward pass.
        let max_shifted_upper = upper
            .iter()
            .zip(shift.iter())
            .map(|(&u, &c)| u as f64 - c as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        if max_shifted_upper > SOFTMAX_COMPLEX_SHIFT_GUARD {
            return Err(format!(
                "shifted upper bound {:.2} exceeds exp guard {:.1} (input too wide for decomposed bounds)",
                max_shifted_upper, SOFTMAX_COMPLEX_SHIFT_GUARD
            ));
        }

        for suffix in ["sm_shift", "sm_exp", "sm_sum", "sm_recip"] {
            let candidate = format!("{name}/{suffix}");
            if self.nodes.contains_key(&candidate) {
                return Err(format!("helper node name '{candidate}' already exists"));
            }
        }

        Ok(NodePlan {
            node_name: name.to_string(),
            input_name,
            axis,
            shift,
            full_shape: shape,
        })
    }

    /// Apply one validated rewrite plan: insert the four helper nodes before
    /// the softmax's position and turn the softmax node itself into the
    /// closing `MulBinary` (keeping its name so downstream references and the
    /// graph output stay valid).
    fn apply_softmax_complex_plan(&mut self, plan: &NodePlan) -> Result<()> {
        let shift_name = format!("{}/sm_shift", plan.node_name);
        let exp_name = format!("{}/sm_exp", plan.node_name);
        let sum_name = format!("{}/sm_sum", plan.node_name);
        let recip_name = format!("{}/sm_recip", plan.node_name);

        let position = self
            .node_order
            .iter()
            .position(|n| n == &plan.node_name)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "softmax-complex: node '{}' missing from node_order",
                    plan.node_name
                ))
            })?;

        let shift_layer = SubConstantLayer::try_new(plan.shift.clone())?;
        let mut reduced_shape = plan.full_shape.clone();
        reduced_shape[plan.axis] = 1;

        let new_nodes = [
            GraphNode::new(
                &shift_name,
                Layer::SubConstant(shift_layer),
                vec![plan.input_name.clone()],
            ),
            GraphNode::new(
                &exp_name,
                Layer::Exp(ExpLayer::new()),
                vec![shift_name.clone()],
            ),
            GraphNode::new(
                &sum_name,
                Layer::ReduceSum(ReduceSumLayer::new(vec![plan.axis as i64], true)),
                vec![exp_name.clone()],
            ),
            GraphNode::new(
                &recip_name,
                Layer::Reciprocal(ReciprocalLayer::new()),
                vec![sum_name.clone()],
            ),
        ];
        for (offset, node) in new_nodes.into_iter().enumerate() {
            let node_name = node.name.clone();
            if self.nodes.insert(node_name.clone(), node).is_some() {
                return Err(NyError::InternalError(format!(
                    "softmax-complex: helper node '{node_name}' already existed at apply time"
                )));
            }
            self.node_order.insert(position + offset, node_name);
        }

        let softmax_node = self.nodes.get_mut(&plan.node_name).ok_or_else(|| {
            NyError::InternalError(format!(
                "softmax-complex: node '{}' missing at apply time",
                plan.node_name
            ))
        })?;
        softmax_node.layer = Layer::MulBinary(MulBinaryLayer);
        softmax_node.inputs = vec![exp_name.clone(), recip_name.clone()];

        // Declared-shape metadata for the taint-degrade path.
        self.declared_shapes
            .insert(shift_name, plan.full_shape.clone());
        self.declared_shapes
            .insert(exp_name, plan.full_shape.clone());
        self.declared_shapes.insert(sum_name, reduced_shape.clone());
        self.declared_shapes.insert(recip_name, reduced_shape);

        // Structural mutation: clear exec-order / ancestor / dispatch caches.
        self.invalidate_exec_order_cache();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::SoftmaxLayer;
    use ndarray::{ArrayD, IxDyn};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Enclosure slack for sampled-point checks.
    ///
    /// The decomposition's IBP/CROWN pipeline rounds outward at SubConstant,
    /// Exp, ReduceSum and Reciprocal, but `BoundedTensor::mul` (the MulBinary
    /// IBP endpoint products) and the CROWN concretization matvec still use
    /// round-to-nearest f32 products, so a point evaluation can sit up to a
    /// few ulps outside the reported interval. Softmax outputs live in [0,1],
    /// so 1e-5 absolute covers that while still catching any real relaxation
    /// bug (which produces errors orders of magnitude larger).
    const ENCLOSURE_TOL: f64 = 1e-5;

    fn box_tensor(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap(),
        )
        .unwrap()
    }

    fn softmax_graph(axis: i32) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "sm",
            Layer::Softmax(SoftmaxLayer::new(axis)),
        ));
        graph.set_output("sm");
        graph
    }

    /// True softmax along `axis` computed in f64 (shift-invariant reference).
    fn true_softmax_f64(x: &ArrayD<f32>, axis: usize) -> ArrayD<f64> {
        let x64 = x.mapv(|v| v as f64);
        let row_max = x64.fold_axis(ndarray::Axis(axis), f64::NEG_INFINITY, |acc, &v| acc.max(v));
        let shifted = &x64 - &row_max.insert_axis(ndarray::Axis(axis));
        let exp = shifted.mapv(f64::exp);
        let sum = exp.sum_axis(ndarray::Axis(axis));
        &exp / &sum.insert_axis(ndarray::Axis(axis))
    }

    fn assert_encloses(bounds: &BoundedTensor, truth: &ArrayD<f64>, label: &str) {
        for ((l, u), t) in bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .zip(truth.iter())
        {
            assert!(
                (*l as f64) <= t + ENCLOSURE_TOL && t - ENCLOSURE_TOL <= (*u as f64),
                "{label}: true softmax {t} outside [{l}, {u}]"
            );
        }
    }

    /// Requirement (structure): the rewrite emits the exact primitive chain
    /// recognized by `is_softmax_decomposition_mul`, and the shift constant is
    /// the rowwise max of the input INTERVAL CENTER.
    #[test]
    fn rewrite_emits_primitive_chain_with_center_rowmax_shift() {
        let mut graph = softmax_graph(-1);
        // 2 rows x 3 cols; centers row0 = [0, 2, -1] (max 2), row1 = [5, -3, 1] (max 5).
        let input = box_tensor(
            &[2, 3],
            vec![-1.0, 1.0, -2.0, 4.0, -4.0, 0.0],
            vec![1.0, 3.0, 0.0, 6.0, -2.0, 2.0],
        );
        let report = graph.decompose_softmax_complex(&input).unwrap();
        assert_eq!(report.decomposed, vec!["sm".to_string()]);
        assert!(report.skipped.is_empty());
        assert_eq!(graph.num_nodes(), 5);

        // Chain shape: sm = MulBinary(sm/sm_exp, sm/sm_recip), recip <- sum <- exp <- shift.
        let mul = graph.node("sm").unwrap();
        assert!(matches!(mul.layer(), Layer::MulBinary(_)));
        assert_eq!(mul.inputs(), ["sm/sm_exp", "sm/sm_recip"]);
        assert!(matches!(
            graph.node("sm/sm_exp").unwrap().layer(),
            Layer::Exp(_)
        ));
        assert!(matches!(
            graph.node("sm/sm_recip").unwrap().layer(),
            Layer::Reciprocal(_)
        ));
        assert!(matches!(
            graph.node("sm/sm_sum").unwrap().layer(),
            Layer::ReduceSum(_)
        ));
        let Layer::SubConstant(shift) = graph.node("sm/sm_shift").unwrap().layer() else {
            panic!("expected SubConstant shift node");
        };
        let expected = [2.0f32, 2.0, 2.0, 5.0, 5.0, 5.0];
        for (got, want) in shift.constant().iter().zip(expected.iter()) {
            assert_eq!(got, want, "shift constant != rowwise center max");
        }

        // Idempotence: no Softmax nodes remain, so a second call is a no-op.
        let again = graph.decompose_softmax_complex(&input).unwrap();
        assert!(again.decomposed.is_empty() && again.skipped.is_empty());
    }

    /// Requirement 3(a): sampled forward agreement — softmax(x) vs the
    /// decomposed subgraph at 10^4 random points within instance-scale boxes.
    #[test]
    fn decomposed_forward_agrees_with_softmax_at_10k_points() {
        let shape = [3usize, 5usize];
        let n: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(0x5eed_50f7);

        // Instance-scale box: attention-logit-like centers and radii.
        let centers: Vec<f32> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
        let radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.1..4.0)).collect();
        let lower: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c - r).collect();
        let upper: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c + r).collect();
        let input = box_tensor(&shape, lower.clone(), upper.clone());

        let mut graph = softmax_graph(-1);
        let report = graph.decompose_softmax_complex(&input).unwrap();
        assert_eq!(report.decomposed.len(), 1, "softmax must be decomposed");

        for trial in 0..10_000 {
            let point: Vec<f32> = lower
                .iter()
                .zip(&upper)
                .map(|(&l, &u)| rng.random_range(l..=u))
                .collect();
            let x = ArrayD::from_shape_vec(IxDyn(&shape), point).unwrap();
            let point_box = BoundedTensor::new(x.clone(), x.clone()).unwrap();
            let out = graph.propagate_ibp(&point_box).unwrap();
            let truth = true_softmax_f64(&x, 1);
            assert_encloses(&out, &truth, &format!("forward agreement trial {trial}"));
            // Agreement, not just enclosure: the decomposed point interval
            // must be tight around softmax(x).
            for (l, u) in out.lower().iter().zip(out.upper().iter()) {
                assert!(
                    (*u as f64 - *l as f64) < 1e-4,
                    "point interval unexpectedly wide: [{l}, {u}]"
                );
            }
        }
    }

    /// Requirement 3(b): decomposed CROWN bounds enclose true softmax outputs
    /// on randomized interval inputs (mirrors layers/softmax/linear/tests.rs
    /// vertex+center sampling).
    #[test]
    fn decomposed_crown_bounds_enclose_true_softmax_on_random_boxes() {
        let shape = [2usize, 4usize];
        let n: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(0xc0ff_ee11);

        for round in 0..20 {
            let centers: Vec<f32> = (0..n).map(|_| rng.random_range(-4.0..4.0)).collect();
            let radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.05..2.5)).collect();
            let lower: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c - r).collect();
            let upper: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c + r).collect();
            let input = box_tensor(&shape, lower.clone(), upper.clone());

            let mut graph = softmax_graph(-1);
            let report = graph.decompose_softmax_complex(&input).unwrap();
            assert_eq!(report.decomposed.len(), 1);

            let crown = graph
                .propagate_crown_with_engine_and_deadline(&input, None, None)
                .unwrap()
                .bounds;
            assert!(
                crown
                    .lower()
                    .iter()
                    .chain(crown.upper().iter())
                    .all(|v| v.is_finite()),
                "round {round}: decomposed CROWN bounds must be finite"
            );

            // Sample: box center, 64 random points, and axis-aligned vertices.
            let mut samples: Vec<Vec<f32>> = vec![centers.clone()];
            for _ in 0..64 {
                samples.push(
                    lower
                        .iter()
                        .zip(&upper)
                        .map(|(&l, &u)| rng.random_range(l..=u))
                        .collect(),
                );
            }
            for mask in 0..(1usize << n.min(8)) {
                samples.push(
                    (0..n)
                        .map(|i| {
                            if (mask >> i) & 1 == 1 {
                                upper[i]
                            } else {
                                lower[i]
                            }
                        })
                        .collect(),
                );
            }

            for sample in &samples {
                let x = ArrayD::from_shape_vec(IxDyn(&shape), sample.clone()).unwrap();
                let truth = true_softmax_f64(&x, 1);
                assert_encloses(&crown, &truth, &format!("CROWN enclosure round {round}"));
            }
        }
    }

    /// Requirement 3(c) + soundness-first gate: inputs too wide for f32 exp
    /// must SKIP the rewrite (graph unchanged, direct-LSE path intact) instead
    /// of planting a node that hard-errors during forward IBP.
    #[test]
    fn wide_input_skips_rewrite_and_keeps_direct_lse_path() {
        let mut graph = softmax_graph(-1);
        let input = box_tensor(&[1, 3], vec![-200.0; 3], vec![200.0; 3]);
        let report = graph.decompose_softmax_complex(&input).unwrap();
        assert!(report.decomposed.is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert!(
            report.skipped[0].1.contains("exp guard"),
            "skip reason should cite the exp guard: {}",
            report.skipped[0].1
        );
        assert_eq!(graph.num_nodes(), 1);
        assert!(matches!(
            graph.node("sm").unwrap().layer(),
            Layer::Softmax(_)
        ));

        // The untouched direct path still produces sound [0,1] softmax bounds.
        let out = graph.propagate_ibp(&input).unwrap();
        for (l, u) in out.lower().iter().zip(out.upper().iter()) {
            assert!(*l >= -1e-6 && *u <= 1.0 + 1e-6, "LSE path bounds [{l},{u}]");
        }
    }

    /// Alpha-loop integration: the DAG alpha-CROWN loop must run over the
    /// decomposed subgraph (MulBinary + Reciprocal alphas registered), return
    /// finite sound bounds, and stay at least as tight as plain IBP. The
    /// output node sums softmax rows (`ReduceSum` over everything), so the
    /// true value is exactly the row count — a fixed enclosure oracle.
    #[test]
    fn alpha_crown_dag_over_decomposed_softmax_is_sound_and_no_looser_than_ibp() {
        let shape = [2usize, 3usize];
        let lower = vec![-1.0f32, 0.5, -0.5, 1.0, -2.0, 0.0];
        let upper = vec![1.0f32, 2.0, 0.5, 3.0, 0.0, 1.5];
        let input = box_tensor(&shape, lower, upper);

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "sm",
            Layer::Softmax(SoftmaxLayer::new(-1)),
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1], false)),
            vec!["sm".to_string()],
        ));
        graph.set_output("out");

        let report = graph.decompose_softmax_complex(&input).unwrap();
        assert_eq!(report.decomposed.len(), 1);

        let ibp = graph.propagate_ibp(&input).unwrap();
        let (ibp_l, ibp_u) = (ibp.flatten().lower()[0], ibp.flatten().upper()[0]);

        let config = crate::bounds::AlphaCrownConfig {
            iterations: 8,
            gradient_method: crate::bounds::GradientMethod::AnalyticChain,
            fix_interm_bounds: false,
            adaptive_skip: false,
            adaptive_skip_pilot: false,
            ..crate::bounds::AlphaCrownConfig::default()
        };
        let (alpha_bounds, _alpha_state) = graph
            .collect_alpha_crown_bounds_dag(&input, &config)
            .expect("alpha-CROWN DAG over decomposed softmax should succeed");
        let out = alpha_bounds
            .get(graph.output_name())
            .expect("output node bounds present");
        let (al, au) = (out.flatten().lower()[0], out.flatten().upper()[0]);

        assert!(al.is_finite() && au.is_finite(), "bounds [{al}, {au}]");
        // Softmax rows each sum to exactly 1 → true output is exactly 2.
        assert!(
            al <= 2.0 + 1e-4 && 2.0 - 1e-4 <= au,
            "true output 2.0 outside alpha bounds [{al}, {au}]"
        );
        // Alpha-CROWN must not be looser than the plain IBP of the same graph.
        let tol = 1e-4;
        assert!(
            al >= ibp_l - tol,
            "alpha lower {al} looser than IBP {ibp_l}"
        );
        assert!(
            au <= ibp_u + tol,
            "alpha upper {au} looser than IBP {ibp_u}"
        );
    }

    /// Non-last-axis softmax: the shift/ReduceSum/broadcast chain must follow
    /// the layer's axis, not assume axis=-1.
    #[test]
    fn axis_zero_softmax_decomposes_and_encloses() {
        let shape = [4usize, 3usize];
        let n: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(0xa715_0000);

        let centers: Vec<f32> = (0..n).map(|_| rng.random_range(-3.0..3.0)).collect();
        let lower: Vec<f32> = centers.iter().map(|c| c - 1.5).collect();
        let upper: Vec<f32> = centers.iter().map(|c| c + 1.5).collect();
        let input = box_tensor(&shape, lower.clone(), upper.clone());

        let mut graph = softmax_graph(0);
        let report = graph.decompose_softmax_complex(&input).unwrap();
        assert_eq!(report.decomposed.len(), 1);

        for _ in 0..500 {
            let point: Vec<f32> = lower
                .iter()
                .zip(&upper)
                .map(|(&l, &u)| rng.random_range(l..=u))
                .collect();
            let x = ArrayD::from_shape_vec(IxDyn(&shape), point).unwrap();
            let point_box = BoundedTensor::new(x.clone(), x.clone()).unwrap();
            let out = graph.propagate_ibp(&point_box).unwrap();
            let truth = true_softmax_f64(&x, 0);
            assert_encloses(&out, &truth, "axis-0 forward agreement");
        }
    }
}
