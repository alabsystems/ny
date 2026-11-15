// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Point-folded ATTACK-ONLY twin of an ONNX graph (#lsnc-attack-twin).
//!
//! # Why this exists
//!
//! ny's loader folds a both-constant node only when the result is EXACT: a
//! rounded binary32 dot product is not the authored real-valued product, so
//! `try_fold_matmul_exact` declines it and `convert_matmul` fails the node closed
//! to an `OpaqueSkip`. That is the right call for every BOUND path. It also makes
//! a whole family unattackable by gradient: on `lsnc_relu` four constant-only
//! `MatMul` cones (the Lyapunov matrix `P = eps*I + W^T W`, `0 @ P`, and the two
//! equilibrium-point network evaluations) are skipped, so the graph falls outside
//! the exact point-VJP fragment, every gradient lane is STATICALLY dead, and the
//! family's three hardest sat rows (field counterexamples at the `X_5 = +2` face
//! with margins ~1.5e-4) are out of reach of ny's derivative-free search.
//!
//! This module builds the graph an attack would want instead: every constant
//! cone is evaluated ONCE by ONNX Runtime — the SAME kernels, in the same f32
//! arithmetic, that the organizer's reference forward runs — and baked into the
//! graph as initializers ([`crate::tail_surrogate::constant_frontier`] +
//! [`crate::tail_surrogate::bake_frontier`], the machinery the flat-objective
//! surrogate already ships). The folded model then loads without a skip.
//!
//! # Why it is a DIFFERENT TYPE, and must stay one
//!
//! A point fold is UNSOUND AS A BOUND. The baked constants are rounded values,
//! not enclosures, so any interval or linear bound computed on the twin describes
//! a network that differs from the authored one by an uncharged rounding
//! residual (the exact failure class the loader's exact-fold gate exists to keep
//! out). The twin is therefore a distinct type whose graph is a PRIVATE field:
//! it exposes a point GRADIENT (a search direction) and a point forward (a
//! steering value), and nothing else. It does not deref to, borrow as, convert
//! into, or hand out a [`GraphNetwork`] — the type every bound API takes — which
//! the unit test `attack_twin_is_type_segregated_from_every_bound_api` and the
//! `compile_fail` examples on [`PointFoldedAttackTwin`] pin.
//!
//! Nothing the twin computes is ever a verdict: the consumer accepts a candidate
//! only after the trusted ONNX Runtime forward of the ORIGINAL model confirms the
//! violation, and the scored path re-runs `gate_sat_with_trusted_oracle` on it.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use ndarray::{Array2, ArrayD};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use prost::Message;

use crate::onnx_proto::ModelProto;
use crate::tail_surrogate::{bake_frontier, constant_frontier, FrozenInput};
use crate::{CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig};

/// ONNX op type of a literal constant node. A frontier tensor produced by one of
/// these is already a value, not a computation: it is left in place.
const LITERAL_CONSTANT_OP: &str = "Constant";

/// A graph whose constant cones were point-folded in binary32 by ONNX Runtime,
/// usable ONLY as an attack oracle.
///
/// The twin cannot reach a bound API: there is no accessor for its graph and no
/// conversion into one.
///
/// ```compile_fail
/// // A bound API takes `&GraphNetwork`; the twin does not coerce to one.
/// fn bound_api(_: &ny_propagate::GraphNetwork) {}
/// fn attack(twin: &ny_onnx::attack_twin::PointFoldedAttackTwin) {
///     bound_api(twin);
/// }
/// ```
///
/// ```compile_fail
/// // The folded graph is a private field.
/// fn attack(
///     twin: &ny_onnx::attack_twin::PointFoldedAttackTwin,
/// ) -> &ny_propagate::GraphNetwork {
///     &twin.graph
/// }
/// ```
///
/// The positive control for the two refusals above: the same paths resolve, so
/// they fail for the segregation reason and not for a typo.
///
/// ```
/// fn bound_api(_: &ny_propagate::GraphNetwork) {}
/// fn attack(twin: &ny_onnx::attack_twin::PointFoldedAttackTwin) -> bool {
///     twin.supports_gradient()
/// }
/// ```
pub struct PointFoldedAttackTwin {
    graph: GraphNetwork,
    folded_tensors: usize,
    frontier_tensors: usize,
    excised_constant_nodes: usize,
}

impl std::fmt::Debug for PointFoldedAttackTwin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointFoldedAttackTwin")
            .field("folded_tensors", &self.folded_tensors)
            .field("frontier_tensors", &self.frontier_tensors)
            .field("excised_constant_nodes", &self.excised_constant_nodes)
            .field("nodes", &self.graph.node_names().len())
            .finish_non_exhaustive()
    }
}

impl PointFoldedAttackTwin {
    /// Build the twin of the model at `model_path` (plain or gzip-wrapped).
    ///
    /// `flat_input_len` is the concatenated element count of the declared
    /// activation inputs (the spec's `X_i` count); `load_config` is the SAME
    /// loader configuration the caller uses for the real graph, so the twin
    /// differs from it only in the folded constants.
    ///
    /// # Errors
    ///
    /// A human-readable reason when there is nothing to fold (no computed
    /// constant frontier), when ONNX Runtime cannot evaluate the frontier (for
    /// example a build without the `ort` feature), when the bake declines, or
    /// when the folded model does not load. The caller treats every error as
    /// "no twin" and keeps its historical behaviour.
    pub fn build(
        model_path: &Path,
        flat_input_len: usize,
        load_config: &OnnxLoadConfig,
    ) -> Result<Self, String> {
        let bytes = crate::read_onnx_bytes_maybe_gzip(model_path)
            .map_err(|error| format!("reading {}: {error}", model_path.display()))?;
        Self::build_from_bytes(&bytes, flat_input_len, load_config)
    }

    /// [`Self::build`] from already-decoded model bytes.
    ///
    /// # Errors
    ///
    /// As [`Self::build`].
    pub fn build_from_bytes(
        model_bytes: &[u8],
        flat_input_len: usize,
        load_config: &OnnxLoadConfig,
    ) -> Result<Self, String> {
        let frontier = constant_frontier(model_bytes)?.ok_or_else(|| {
            "the graph has no constant frontier: nothing to point-fold".to_string()
        })?;
        let computed = computed_frontier(model_bytes, &frontier.frontier)?;
        if computed.is_empty() {
            return Err(
                "every constant-frontier tensor is a literal Constant: nothing to point-fold"
                    .to_string(),
            );
        }
        // One ONNX Runtime forward over a copy whose outputs ARE the frontier.
        // The frontier depends on no activation input, so the input value is
        // immaterial; zeros are always in range for the session to accept.
        let mut capture = crate::diff::OrtForward::from_model_bytes(
            &frontier.exposed_model,
            flat_input_len,
            false,
        )
        .map_err(|error| format!("building the frontier-capture session: {error}"))?;
        let captured = capture
            .run_all_outputs_f32_lossy(&vec![0.0_f32; flat_input_len])
            .map_err(|error| format!("capturing the constant frontier: {error}"))?;
        // Bake only COMPUTED binary32 tensors. A non-f32 computed tensor (shape
        // arithmetic) stays in the graph for the loader's exact integer fold.
        let baked: Vec<FrozenInput> = captured
            .into_iter()
            .filter(|(name, _)| computed.contains(name.as_str()))
            .filter_map(|(name, value)| {
                value.map(|(shape, values)| FrozenInput {
                    name,
                    shape,
                    values,
                })
            })
            .collect();
        if baked.is_empty() {
            return Err(format!(
                "none of the {} computed frontier tensor(s) is binary32: nothing to point-fold",
                computed.len()
            ));
        }
        let folded_bytes = bake_frontier(model_bytes, &baked)?.ok_or_else(|| {
            "the frontier bake declined (a folded producer also feeds an unfolded tensor)"
                .to_string()
        })?;
        let model = crate::load_onnx_bytes_with_config(
            "point-folded-attack-twin",
            &folded_bytes,
            load_config,
        )
        .map_err(|error| format!("loading the point-folded twin: {error}"))?;
        let options = GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        };
        let graph = model
            .to_graph_network_with_options(options)
            .map_err(|error| format!("lowering the point-folded twin: {error}"))?;
        Ok(Self {
            graph,
            folded_tensors: baked.len(),
            frontier_tensors: frontier.frontier.len(),
            excised_constant_nodes: frontier.constant_node_count,
        })
    }

    /// How many computed constant tensors were point-folded into initializers.
    pub fn folded_tensors(&self) -> usize {
        self.folded_tensors
    }

    /// How many tensors the constant frontier had (folded or literal).
    pub fn frontier_tensors(&self) -> usize {
        self.frontier_tensors
    }

    /// How many nodes of the original graph computed constants only.
    pub fn excised_constant_nodes(&self) -> usize {
        self.excised_constant_nodes
    }

    /// Node count of the folded graph (diagnostics only).
    pub fn node_count(&self) -> usize {
        self.graph.node_names().len()
    }

    /// Whether the folded graph is inside the exact point-VJP fragment widened
    /// by the exact linear reductions. `false` means [`Self::point_gradient`]
    /// returns `Ok(None)` at every point.
    pub fn supports_gradient(&self) -> bool {
        self.graph
            .supports_attack_point_gradient_with_exact_reductions()
    }

    /// The first folded-graph node outside that fragment, as
    /// `(node name, layer type)` (diagnostics only).
    pub fn first_gradient_refusal(&self) -> Option<(String, &'static str)> {
        self.graph
            .first_attack_point_gradient_refusal_with_exact_reductions()
    }

    /// The input tensor shape the twin's point evaluations take for a model
    /// whose ONNX-declared input shape is `declared` (see
    /// `GraphNetwork::attack_point_gradient_input_shape`).
    pub fn input_shape(&self, declared: &[usize]) -> Vec<usize> {
        self.graph.attack_point_gradient_input_shape(declared)
    }

    /// `d(spec_row · y(x)) / dx` on the folded graph at the point `x`.
    ///
    /// ATTACK DIRECTION ONLY — the exact point-Jacobian of the TWIN, which
    /// matches the authored network up to the folded constants' rounding.
    ///
    /// # Errors
    ///
    /// An internal or structural failure of the walk, as text. `Ok(None)` means
    /// the point is not differentiable here (outside the fragment, a declined
    /// dispatch, or the deadline).
    pub fn point_gradient(
        &self,
        x: &ArrayD<f32>,
        spec_row: &Array2<f32>,
        deadline: Option<Instant>,
    ) -> Result<Option<ArrayD<f32>>, String> {
        self.graph
            .attack_point_gradient_with_exact_reductions(x, spec_row, None, deadline)
            .map_err(|error| format!("{error:#}"))
    }

    /// The folded graph's own f32 forward at the point `x`, flattened.
    ///
    /// STEERING / DIAGNOSTIC VALUE ONLY: it describes the twin, not the authored
    /// network, and it is never compared against a threshold to decide
    /// anything. Acceptance always re-runs the ORIGINAL model.
    ///
    /// # Errors
    ///
    /// The forward's own failure, as text.
    pub fn point_forward(
        &self,
        x: &ArrayD<f32>,
        deadline: Option<Instant>,
    ) -> Result<Vec<f32>, String> {
        let input = BoundedTensor::concrete(x.clone()).map_err(|error| error.to_string())?;
        let out = self
            .graph
            .propagate_concrete_point(&input, None, deadline)
            .map_err(|error| error.to_string())?;
        Ok(out.lower().iter().copied().collect())
    }
}

/// The frontier tensors produced by a COMPUTING node (anything but a literal
/// `Constant`), i.e. the ones a point fold actually evaluates.
fn computed_frontier(model_bytes: &[u8], frontier: &[String]) -> Result<HashSet<String>, String> {
    let model = ModelProto::decode(model_bytes)
        .map_err(|error| format!("failed to parse ONNX protobuf: {error}"))?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| "ONNX model has no graph".to_string())?;
    let producer_op: HashMap<&str, &str> = graph
        .node
        .iter()
        .flat_map(|node| {
            node.output
                .iter()
                .map(move |output| (output.as_str(), node.op_type.as_str()))
        })
        .collect();
    Ok(frontier
        .iter()
        .filter(|name| {
            producer_op
                .get(name.as_str())
                .is_some_and(|op| *op != LITERAL_CONSTANT_OP)
        })
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests;
