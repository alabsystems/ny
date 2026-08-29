// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete verification via MIP (ay).
//!
//! Two layers live here.
//!
//! **The curated `ny-mip` surface.** Encode a network + property as a MILP and
//! check feasibility (SAT = a point in the negated-property region, UNSAT =
//! verified), plus the four lanes an embedded consumer needs around that core:
//!
//! * **LP / OBBT tightening** — [`LpTightener`] (per-neuron pre-activation
//!   tightening) and [`obbt_relaxation_bounds`] (coupled optimization-based
//!   bound tightening over the continuous relaxation).
//! * **SMT-LIB export** — [`to_smtlib_decision`] / [`to_smtlib_minimize`] emit
//!   the exact-rational QF_LRA byte stream the ay backend solves, for corpus
//!   capture and independent replay.
//! * **Star search** — [`star_verify`] / [`star_search_unsafe_conjunction`]
//!   over [`Star`] sets. Deliberately *candidate-only*: see
//!   [`StarUnsafeCandidateVerdict`].
//! * **Farkas infeasibility evidence** — [`prove_infeasible_with_row_farkas`]
//!   is the PRODUCER of exact-rational infeasibility multipliers for a
//!   [`MilpProblem`]; the matching checker lives in
//!   [`ny_api::cert`](crate::cert) (`check_farkas`).
//!
//! **The graph terminal.** [`verify_graph_complete_mip`] runs a real complete
//! MIP verification of a [`GraphNetwork`] against a [`VerificationSpec`], and
//! [`reduce_graph_to_fc_relu`] is the fail-closed reduction it is built on.
//!
//! # Soundness contract of the graph terminal
//!
//! The MIP models the network's REAL-arithmetic semantics (f32 parameters
//! widened exactly to f64), the same semantics every propagation method in ny
//! bounds. Folding the deployed-f32 rounding gap onto a verdict is the separate,
//! opt-in job of [`ny_api::cert::deployed_fp_margin`](crate::cert).
//!
//! Within that model the encoding is an OUTER approximation:
//!
//! * every `Linear` layer becomes an exact affine equality row;
//! * every `ReLU` becomes the standard Big-M formulation, which is EXACT for
//!   ReLU restricted to the supplied pre-activation box;
//! * the pre-activation boxes come from ny's own sound CROWN-IBP node-bound
//!   collection and are then inflated OUTWARD
//!   ([`CompleteMipOptions::preactivation_inflation`]), which can only enlarge
//!   the feasible set.
//!
//! Consequently `UNSAT` of a negated-property disjunct over that superset
//! implies the disjunct is unreachable for the real network — a sound proof.
//! `SAT` is NOT reported as a counterexample: the encoded disjunct is the
//! non-strict relaxation `y_i <= lo_i` of the strict violation `y_i < lo_i`, so
//! a feasible point may sit exactly on the property boundary. Every non-UNSAT
//! outcome is therefore reported as *not proved*, never as `Violated`.

use std::collections::HashSet;

use ny_core::{Bound, NyError, Result, VerificationSpec};

use crate::graph::{GraphNetwork, NETWORK_INPUT};
use crate::layers::Layer;
use crate::verify::Verifier;

/// Typed MIP errors. [`MipError`] converts into
/// [`NyError`](ny_core::NyError) via `From`, so `?` works in an
/// [`ny_core::Result`] function.
pub use ny_mip::MipError;
/// Network -> MILP encoding: the incremental encoder, its extracted parts, and
/// the one-call feed-forward FC+ReLU convenience entry point.
pub use ny_mip::{encode_feedforward, MipEncoder, MipParts};
/// Solver configuration and its two field types.
///
/// [`MipBackend`] selects which solver the solver-neutral IR is lowered to and
/// [`MipFeasibilityIngress`] selects how the final feasibility session must
/// enter that backend. Both are [`MipConfig`] fields, so without them the
/// `backend` and `feasibility_ingress` knobs are unsettable through the facade.
pub use ny_mip::{MipBackend, MipConfig, MipFeasibilityIngress};
/// Solve surface: the solver, its verdict type, the optimization sense, the
/// phase-split UNSAT cache, and the witness-only one-sided SAT probe.
///
/// The probe lane carries no verdict authority by construction — even
/// [`OneSidedSatDecline::InfeasibleIgnored`] is a decline, never an `Unsat`.
pub use ny_mip::{
    MipResult, MipSolver, OneSidedSatDecline, OneSidedSatProbe, OneSidedSatWitness, Sense,
    SplitUnsatCache,
};

/// Solver-neutral MILP problem IR: the value every other lane in this module
/// consumes or produces ([`to_smtlib_decision`], [`obbt_relaxation_bounds`],
/// [`prove_infeasible_with_row_farkas`], [`to_milp_text`]).
pub use ny_mip::ir::{Col, ColSpec, MarginRowError, MilpProblem, Row, RowSpec};

/// LP bound tightening.
///
/// [`LpTightener`] solves per-neuron LP relaxations (triangle relaxation, no
/// binary indicators) to shrink pre-activation boxes before MIP encoding.
/// [`obbt_relaxation_bounds`] instead tightens a whole target set as ONE coupled
/// problem over the continuous relaxation, reaching a fixpoint the independent
/// per-neuron pass cannot. Both only ever produce bounds at least as tight,
/// never looser, and every committed bound is rigorous and outward-rounded.
pub use ny_mip::{obbt_relaxation_bounds, LpTightener, RelaxationObbt};

/// Exact-rational QF_LRA SMT-LIB serialization of a [`MilpProblem`]: the
/// decision (`check-sat`) form and the optimization (`minimize`) form. These are
/// the exact byte streams the ay backend solves, so an external SMT solver can
/// replay a feasibility question independently.
pub use ny_mip::{to_smtlib_decision, to_smtlib_minimize};

/// Round-trippable plain-text dump of a [`MilpProblem`], for capturing and
/// replaying a failing encoding outside the process that produced it.
pub use ny_mip::dump::{from_milp_text, to_milp_text};

/// Farkas infeasibility-evidence PRODUCER.
///
/// [`prove_infeasible_with_row_farkas`] solves the LP lane of a
/// [`MilpProblem`] and, when it is infeasible, returns the exact-rational Farkas
/// multipliers keyed by `(row index, [`RowSide`], multiplier)`. It is
/// fail-closed in three separate ways: it declines any problem with integer
/// columns, it independently re-verifies the backend's certificate against the
/// model before returning it, and it declines any multiplier that is not a row
/// bound. The matching CHECKER is
/// [`ny_api::cert::check_farkas`](crate::cert::check_farkas).
pub use ny_mip::{prove_infeasible_with_row_farkas, RowSide};

/// Structurally exact star-set search.
///
/// [`star_verify`] proves a conjunctive spec and [`star_search_unsafe_conjunction`]
/// searches one conjunctive unsafe region. Both are CANDIDATE-ONLY: the star
/// algebra stores affine coefficients in f32 without a certified roundoff
/// enclosure, so their verdicts are search evidence and must not authorize a
/// verification verdict. [`Star`] is the input set type.
pub use ny_mip::star_verify::{
    star_search_unsafe_conjunction, star_verify, StarBudget, StarLayer, StarSpec, StarStats,
    StarUnsafeCandidateVerdict, StarUnsafeConjunction, StarUnsafeRow, StarVerdict,
};
/// Input set type for the star lane.
pub use ny_tensor::zonotope::Star;

/// Default outward inflation applied to every intermediate pre-activation box
/// before it becomes a Big-M range.
///
/// Matches the `DELTA` the production graph-MIP lowering uses. Widening a
/// pre-activation box can only ENLARGE the encoded feasible set (the Big-M
/// formulation is exact for ReLU on whatever box it is given), so this is a
/// pure safety margin: it can turn a would-be `Unsat` into a non-answer, never
/// a non-answer into a wrong `Unsat`.
pub const DEFAULT_PREACTIVATION_INFLATION: f32 = 1e-4;

/// Options for [`verify_graph_complete_mip`].
#[derive(Debug, Clone, Copy)]
pub struct CompleteMipOptions {
    /// Solver configuration, including the per-solve `timeout_secs` budget.
    pub mip: MipConfig,
    /// Outward inflation applied to every intermediate pre-activation box.
    ///
    /// Must be finite and non-negative. See
    /// [`DEFAULT_PREACTIVATION_INFLATION`].
    pub preactivation_inflation: f32,
}

impl Default for CompleteMipOptions {
    fn default() -> Self {
        Self {
            mip: MipConfig::default(),
            preactivation_inflation: DEFAULT_PREACTIVATION_INFLATION,
        }
    }
}

/// Outcome of a complete MIP verification run over a graph.
///
/// There is deliberately no `Violated` outcome: see the module-level soundness
/// contract.
#[derive(Debug, Clone)]
pub struct CompleteMipReport {
    /// Whether EVERY negated-property disjunct was proven infeasible.
    ///
    /// `true` is a complete proof of the spec's output box for the encoded
    /// fragment, not a relaxation.
    pub verified: bool,
    /// Whether every refuting solve carried independently verified exact
    /// certificate evidence.
    ///
    /// Purely informational: [`Self::verified`] does not depend on it.
    pub certified: bool,
    /// Number of `(output index, side)` disjuncts the run had to refute.
    ///
    /// Infinite spec sides are vacuously true and are not counted.
    pub disjuncts: usize,
    /// Number of disjuncts actually proven infeasible before the run stopped.
    pub disjuncts_refuted: usize,
    /// Why the property was not proved, or `None` when [`Self::verified`].
    pub inconclusive: Option<String>,
}

/// A [`GraphNetwork`] reduced to the sequential FC+ReLU fragment
/// [`encode_feedforward`] accepts.
///
/// Produced only by [`reduce_graph_to_fc_relu`], which fails closed rather than
/// approximate anything.
#[derive(Debug, Clone)]
pub struct FcReluChain {
    /// Node name of each `Linear` layer, input -> output order.
    linear_nodes: Vec<String>,
    /// Row-major `[out x in]` weights per linear layer, widened exactly to f64.
    weights: Vec<Vec<f64>>,
    /// Bias per linear layer (zeros when the layer carries none).
    biases: Vec<Vec<f64>>,
    /// `[input_dim, hidden_1, ..., output_dim]`.
    layer_dims: Vec<usize>,
}

impl FcReluChain {
    /// `[input_dim, hidden_1, ..., output_dim]`.
    #[must_use]
    pub fn layer_dims(&self) -> &[usize] {
        &self.layer_dims
    }

    /// Number of `Linear` layers in the chain (ReLU count is one less).
    #[must_use]
    pub fn num_linear_layers(&self) -> usize {
        self.weights.len()
    }

    /// Graph node names of the `Linear` layers, input -> output order.
    #[must_use]
    pub fn linear_node_names(&self) -> &[String] {
        &self.linear_nodes
    }

    /// Network input dimension.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.layer_dims[0]
    }

    /// Network output dimension.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.layer_dims[self.layer_dims.len() - 1]
    }
}

/// Refusal message shared by every non-fragment rejection, so the caller always
/// learns what would be needed to encode the graph it actually has.
fn refuse(detail: &str) -> NyError {
    NyError::UnsupportedOp(format!(
        "complete MIP terminal: {detail}. The exported MIP encoder \
         (`ny_mip::encode_feedforward`) covers only a sequential \
         `Linear (ReLU Linear)*` chain; the full graph lowering (Conv2d / \
         BatchNorm / residual Add / MaxPool / general DAG topology) is not \
         exported from `ny-mip`, so this graph cannot be encoded soundly. \
         Refusing rather than returning a verdict-shaped non-answer"
    ))
}

/// Reduce a [`GraphNetwork`] to the sequential FC+ReLU chain the exported MIP
/// encoder accepts, or fail closed explaining exactly what blocked it.
///
/// A graph qualifies when ALL of the following hold:
///
/// 1. it has an output node;
/// 2. every node on the walk from the output back toward the network input
///    consumes exactly one input;
/// 3. the walk terminates at [`NETWORK_INPUT`];
/// 4. the walk visits EVERY node the graph holds — no side branch, no orphan,
///    no node consumed twice;
/// 5. every layer is `Linear` or `ReLU`, alternating strictly as
///    `Linear (ReLU Linear)*` — the encoding [`encode_feedforward`] builds; and
/// 6. consecutive linear layers agree dimensionally.
///
/// # Errors
///
/// [`NyError::UnsupportedOp`] naming the offending node and the violated
/// condition. This is the ONLY outcome for a non-fragment graph: the reduction
/// never approximates and never degrades.
pub fn reduce_graph_to_fc_relu(net: &GraphNetwork) -> Result<FcReluChain> {
    let output = net.output_name();
    if output.is_empty() {
        return Err(refuse("the graph has no output node set"));
    }

    // Walk output -> input, collecting (name, layer) in reverse order.
    let mut reversed: Vec<(String, &Layer)> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut current = output.to_string();
    loop {
        if !visited.insert(current.clone()) {
            return Err(refuse(&format!(
                "node '{current}' is revisited — the graph is cyclic, not a chain"
            )));
        }
        let node = net
            .node(&current)
            .ok_or_else(|| refuse(&format!("dangling node reference '{current}'")))?;
        let inputs = node.inputs();
        if inputs.len() != 1 {
            return Err(refuse(&format!(
                "node '{current}' ({}) has {} inputs — the encodable fragment is a chain in \
                 which every node consumes exactly one input",
                node.layer().layer_type(),
                inputs.len()
            )));
        }
        reversed.push((current.clone(), node.layer()));
        let predecessor = inputs[0].clone();
        if predecessor == NETWORK_INPUT {
            break;
        }
        current = predecessor;
    }

    if visited.len() != net.num_nodes() {
        return Err(refuse(&format!(
            "the walk from output '{output}' to the network input covers {} of the graph's {} \
             node(s) — the encodable fragment has no side branches or orphan nodes",
            visited.len(),
            net.num_nodes()
        )));
    }

    reversed.reverse();
    let forward = reversed;

    // Strict `Linear (ReLU Linear)*` alternation.
    if forward.len().is_multiple_of(2) {
        let (last_name, last_layer) = &forward[forward.len() - 1];
        return Err(refuse(&format!(
            "the chain has {} layers and ends at node '{last_name}' ({}); an encodable chain \
             alternates `Linear (ReLU Linear)*`, so it always has an odd number of layers and \
             ends in Linear",
            forward.len(),
            last_layer.layer_type()
        )));
    }

    let mut linear_nodes = Vec::new();
    let mut weights: Vec<Vec<f64>> = Vec::new();
    let mut biases: Vec<Vec<f64>> = Vec::new();
    let mut layer_dims: Vec<usize> = Vec::new();

    for (position, (name, layer)) in forward.iter().enumerate() {
        if position.is_multiple_of(2) {
            let Layer::Linear(linear) = layer else {
                return Err(refuse(&format!(
                    "node '{name}' is '{}' where the encodable chain requires Linear (chain \
                     position {position})",
                    layer.layer_type()
                )));
            };
            let weight = linear.weight();
            let (out_features, in_features) = weight.dim();
            if layer_dims.is_empty() {
                layer_dims.push(in_features);
            } else if layer_dims[layer_dims.len() - 1] != in_features {
                return Err(refuse(&format!(
                    "node '{name}' expects {in_features} inputs but the preceding layer emits \
                     {}",
                    layer_dims[layer_dims.len() - 1]
                )));
            }

            let mut row_major = Vec::with_capacity(out_features * in_features);
            for row in 0..out_features {
                for column in 0..in_features {
                    let value = weight[[row, column]];
                    if !value.is_finite() {
                        return Err(refuse(&format!(
                            "node '{name}' weight[{row}][{column}] is not finite ({value})"
                        )));
                    }
                    row_major.push(f64::from(value));
                }
            }

            let mut bias = vec![0.0_f64; out_features];
            if let Some(values) = linear.bias() {
                if values.len() != out_features {
                    return Err(refuse(&format!(
                        "node '{name}' has {} bias entries for {out_features} outputs",
                        values.len()
                    )));
                }
                for (index, value) in values.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(refuse(&format!(
                            "node '{name}' bias[{index}] is not finite ({value})"
                        )));
                    }
                    bias[index] = f64::from(*value);
                }
            }

            linear_nodes.push(name.clone());
            weights.push(row_major);
            biases.push(bias);
            layer_dims.push(out_features);
        } else if !matches!(layer, Layer::ReLU(_)) {
            return Err(refuse(&format!(
                "node '{name}' is '{}' where the encodable chain requires ReLU (chain position \
                 {position}); only ReLU nonlinearities have an exact Big-M encoding here",
                layer.layer_type()
            )));
        }
    }

    Ok(FcReluChain {
        linear_nodes,
        weights,
        biases,
        layer_dims,
    })
}

/// Sound per-layer pre-activation boxes for the hidden ReLUs of `chain`.
///
/// Bounds come from ny's own CROWN-IBP node-bound collection (a sound enclosure
/// of each node's output over the input box) and are then inflated OUTWARD by
/// `inflation`, which can only enlarge the encoded feasible set.
fn hidden_preactivation_bounds(
    net: &GraphNetwork,
    chain: &FcReluChain,
    input: &ny_tensor::BoundedTensor,
    inflation: f32,
) -> Result<Vec<Vec<Bound>>> {
    let node_bounds = net.collect_crown_ibp_bounds_dag(input)?;
    let hidden = chain.num_linear_layers().saturating_sub(1);
    let mut out = Vec::with_capacity(hidden);

    for index in 0..hidden {
        let name = &chain.linear_nodes[index];
        let expected = chain.layer_dims[index + 1];
        let tensor = node_bounds.get(name).ok_or_else(|| {
            NyError::InternalError(format!(
                "complete MIP terminal: node-bound collection returned no bounds for linear node \
                 '{name}'; refusing to encode a Big-M range without a sound pre-activation box"
            ))
        })?;
        if tensor.len() != expected {
            return Err(NyError::InternalError(format!(
                "complete MIP terminal: node '{name}' produced {} bound elements but its layer \
                 emits {expected}",
                tensor.len()
            )));
        }

        let mut layer = Vec::with_capacity(expected);
        for (lower, upper) in tensor.lower().iter().zip(tensor.upper().iter()) {
            // Round-to-nearest never moves `lower - inflation` above `lower`
            // nor `upper + inflation` below `upper`, so this only widens.
            let lower = lower - inflation;
            let upper = upper + inflation;
            if !lower.is_finite() || !upper.is_finite() {
                return Err(NyError::UnsupportedOp(format!(
                    "complete MIP terminal: node '{name}' has a non-finite pre-activation bound \
                     [{lower}, {upper}]; a Big-M range needs finite endpoints, refusing rather \
                     than encoding an infinite coefficient"
                )));
            }
            // `Bound::new` panics on an inverted interval; a verification path
            // must fail closed with a typed error instead of unwinding.
            if lower > upper {
                return Err(NyError::NumericalInstability(format!(
                    "complete MIP terminal: node '{name}' produced the inverted pre-activation \
                     bound [{lower}, {upper}]; refusing to encode a Big-M range from it"
                )));
            }
            layer.push(Bound::new(lower, upper));
        }
        out.push(layer);
    }

    Ok(out)
}

/// Build a fresh encoder for `chain` with the property left unconstrained.
fn encode_chain(
    chain: &FcReluChain,
    input_bounds: &[Bound],
    intermediate: &[Vec<Bound>],
) -> Result<MipEncoder> {
    Ok(encode_feedforward(
        &chain.weights,
        &chain.biases,
        &chain.layer_dims,
        input_bounds,
        intermediate,
    )?)
}

/// Which side of an output bound a disjunct negates.
#[derive(Debug, Clone, Copy)]
enum Side {
    /// Negation of `y_i >= lo`, relaxed to `y_i <= lo`.
    Below,
    /// Negation of `y_i <= hi`, relaxed to `y_i >= hi`.
    Above,
}

/// Run a complete MIP verification of `net` against `spec`.
///
/// The property `for all x in input box: y in spec output box` is refuted
/// disjunct by disjunct: for each finite output-bound side, the encoder is
/// rebuilt with the (relaxed) negation of that side and asked for feasibility.
/// `Unsat` on every disjunct is a complete proof; anything else is reported as
/// not proved.
///
/// # ENSURES
/// - [`CompleteMipReport::verified`] is `true` only when every counted disjunct
///   returned `MipResult::Unsat`.
/// - No outcome is ever reported as a counterexample (see the module docs).
///
/// # Errors
///
/// [`NyError::UnsupportedOp`] when `net` is outside the encodable fragment (see
/// [`reduce_graph_to_fc_relu`]) or carries non-finite bounds,
/// [`NyError::InvalidSpec`] when `spec` does not match the reduced chain, and
/// [`NyError::InvalidConfig`] for an invalid
/// [`CompleteMipOptions::preactivation_inflation`]. Propagation errors from the
/// pre-activation bound collection are propagated unchanged.
pub fn verify_graph_complete_mip(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    options: &CompleteMipOptions,
) -> Result<CompleteMipReport> {
    if !options.preactivation_inflation.is_finite() || options.preactivation_inflation < 0.0 {
        return Err(NyError::InvalidConfig(format!(
            "complete MIP terminal: preactivation_inflation must be finite and non-negative, got \
             {}",
            options.preactivation_inflation
        )));
    }

    let chain = reduce_graph_to_fc_relu(net)?;

    let input_bounds = spec.input_bounds();
    if input_bounds.len() != chain.input_dim() {
        return Err(NyError::InvalidSpec(format!(
            "complete MIP terminal: spec has {} input bound(s) but the network takes {}",
            input_bounds.len(),
            chain.input_dim()
        )));
    }
    for (index, bound) in input_bounds.iter().enumerate() {
        if !bound.lower().is_finite() || !bound.upper().is_finite() {
            return Err(NyError::UnsupportedOp(format!(
                "complete MIP terminal: input bound {index} is [{}, {}]; the MIP input box must \
                 be finite",
                bound.lower(),
                bound.upper()
            )));
        }
    }

    let output_bounds = spec.output_bounds();
    if output_bounds.is_empty() {
        return Err(NyError::InvalidSpec(
            "complete MIP terminal: empty output_bounds in specification — nothing to verify"
                .to_string(),
        ));
    }
    if output_bounds.len() != chain.output_dim() {
        return Err(NyError::InvalidSpec(format!(
            "complete MIP terminal: spec has {} output bound(s) but the network emits {}",
            output_bounds.len(),
            chain.output_dim()
        )));
    }

    let input = Verifier::bounds_to_tensor(input_bounds, spec.input_shape())?;
    let intermediate =
        hidden_preactivation_bounds(net, &chain, &input, options.preactivation_inflation)?;

    // Every finite spec side is one disjunct of the negated property. An
    // infinite side is vacuously satisfied and contributes nothing to prove.
    let mut disjuncts: Vec<(usize, Side, f64)> = Vec::new();
    for (index, bound) in output_bounds.iter().enumerate() {
        if bound.lower().is_finite() {
            disjuncts.push((index, Side::Below, f64::from(bound.lower())));
        }
        if bound.upper().is_finite() {
            disjuncts.push((index, Side::Above, f64::from(bound.upper())));
        }
    }

    let total = disjuncts.len();
    let mut certified = true;
    let mut refuted = 0_usize;

    for (index, side, threshold) in disjuncts {
        let mut encoder = encode_chain(&chain, input_bounds, &intermediate)?;
        match side {
            Side::Below => encoder.constrain_output_leq_const(index, threshold)?,
            Side::Above => encoder.constrain_output_geq_const(index, threshold)?,
        }
        let solver = MipSolver::new(encoder.into_parts(), options.mip);
        let relation = match side {
            Side::Below => "<=",
            Side::Above => ">=",
        };
        match solver.check_feasibility()? {
            MipResult::Unsat {
                certified: this_certified,
            } => {
                certified &= this_certified;
                refuted += 1;
            }
            MipResult::Sat { .. } => {
                return Ok(CompleteMipReport {
                    verified: false,
                    certified: false,
                    disjuncts: total,
                    disjuncts_refuted: refuted,
                    inconclusive: Some(format!(
                        "the relaxed negation `y[{index}] {relation} {threshold}` is feasible; \
                         this is NOT a counterexample (the encoded disjunct is non-strict, so a \
                         feasible point may lie exactly on the property boundary), so the \
                         property is left unproved"
                    )),
                });
            }
            MipResult::Timeout => {
                return Ok(CompleteMipReport {
                    verified: false,
                    certified: false,
                    disjuncts: total,
                    disjuncts_refuted: refuted,
                    inconclusive: Some(format!(
                        "the solve for `y[{index}] {relation} {threshold}` hit the \
                         {:.3}s budget",
                        options.mip.timeout_secs
                    )),
                });
            }
            MipResult::Error(message) => {
                return Ok(CompleteMipReport {
                    verified: false,
                    certified: false,
                    disjuncts: total,
                    disjuncts_refuted: refuted,
                    inconclusive: Some(format!(
                        "the solve for `y[{index}] {relation} {threshold}` failed: {message}"
                    )),
                });
            }
        }
    }

    Ok(CompleteMipReport {
        verified: true,
        certified,
        disjuncts: total,
        disjuncts_refuted: refuted,
        inconclusive: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphNode;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use ndarray::{Array1, Array2};

    fn linear(weight: Vec<f32>, rows: usize, cols: usize, bias: Vec<f32>) -> LinearLayer {
        LinearLayer::new(
            Array2::from_shape_vec((rows, cols), weight).expect("weight shape"),
            Some(Array1::from_vec(bias)),
        )
        .expect("valid linear layer")
    }

    /// `y = relu(|x| - 0.5) - relu(|x| - 0.5) == 0` written as an
    /// `Linear -> ReLU -> Linear -> ReLU -> Linear` chain.
    ///
    /// Both second-layer neurons carry the SAME pre-activation, so the true
    /// output is exactly `0`, but per-neuron CROWN relaxes them independently
    /// and cannot see the cancellation.
    fn cancelling_chain() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(linear(vec![1.0, -1.0], 2, 1, vec![0.0, 0.0])),
        ));
        graph.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer::new()),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l2",
            Layer::Linear(linear(vec![1.0, 1.0, 1.0, 1.0], 2, 2, vec![-0.5, -0.5])),
            vec!["r1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "r2",
            Layer::ReLU(ReLULayer::new()),
            vec!["l2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l3",
            Layer::Linear(linear(vec![1.0, -1.0], 1, 2, vec![0.0])),
            vec!["r2".to_string()],
        ));
        graph.set_output("l3");
        graph
    }

    /// A residual `Add` join: outside the encodable fragment.
    fn residual_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(linear(vec![1.0, -1.0], 2, 1, vec![0.0, 0.0])),
        ));
        graph.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer::new()),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "l1".to_string(),
            "r1".to_string(),
        ));
        graph.set_output("add");
        graph
    }

    #[test]
    fn chain_reduction_reports_dims_and_nodes() {
        let chain = reduce_graph_to_fc_relu(&cancelling_chain()).expect("fragment graph reduces");
        assert_eq!(chain.layer_dims(), &[1, 2, 2, 1]);
        assert_eq!(chain.num_linear_layers(), 3);
        assert_eq!(chain.linear_node_names(), &["l1", "l2", "l3"]);
        assert_eq!(chain.input_dim(), 1);
        assert_eq!(chain.output_dim(), 1);
    }

    #[test]
    fn residual_join_is_refused_not_approximated() {
        let error = reduce_graph_to_fc_relu(&residual_graph())
            .expect_err("a residual Add join is outside the encodable fragment");
        assert!(
            matches!(error, NyError::UnsupportedOp(_)),
            "refusal must be UnsupportedOp, got {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("'add'") && message.contains("2 inputs"),
            "the refusal must name the blocking node: {message}"
        );
        assert!(
            message.contains("not exported from `ny-mip`"),
            "the refusal must say what is missing: {message}"
        );
    }

    #[test]
    fn trailing_relu_is_refused() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(linear(vec![1.0], 1, 1, vec![0.0])),
        ));
        graph.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer::new()),
            vec!["l1".to_string()],
        ));
        graph.set_output("r1");

        let error = reduce_graph_to_fc_relu(&graph).expect_err("a chain ending in ReLU is refused");
        let message = error.to_string();
        assert!(
            message.contains("ends in Linear"),
            "unexpected refusal: {message}"
        );
    }

    #[test]
    fn unsupported_activation_is_refused_by_name() {
        use crate::layers::TanhLayer;
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(linear(vec![1.0], 1, 1, vec![0.0])),
        ));
        graph.add_node(GraphNode::new(
            "t1",
            Layer::Tanh(TanhLayer::new()),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l2",
            Layer::Linear(linear(vec![1.0], 1, 1, vec![0.0])),
            vec!["t1".to_string()],
        ));
        graph.set_output("l2");

        let message = reduce_graph_to_fc_relu(&graph)
            .expect_err("Tanh has no exact Big-M encoding")
            .to_string();
        assert!(
            message.contains("'t1'") && message.contains("requires ReLU"),
            "unexpected refusal: {message}"
        );
    }

    #[test]
    fn complete_mip_proves_what_propagation_cannot() {
        let graph = cancelling_chain();
        // True output is exactly 0 for every x in [-1, 1]; every propagation
        // rung stops at about +-0.5 (see the ladder's terminal test).
        let spec =
            VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-0.01, 0.01)])
                .expect("valid spec");
        let options = CompleteMipOptions {
            mip: MipConfig {
                timeout_secs: 30.0,
                ..MipConfig::default()
            },
            ..CompleteMipOptions::default()
        };

        let report = verify_graph_complete_mip(&graph, &spec, &options).expect("encodable graph");
        assert!(
            report.verified,
            "the MIP encoding sees the cancellation: {report:?}"
        );
        assert_eq!(report.disjuncts, 2, "both finite output sides are checked");
        assert_eq!(report.disjuncts_refuted, 2);
        assert!(report.inconclusive.is_none());
    }

    #[test]
    fn complete_mip_does_not_claim_a_false_property() {
        let graph = cancelling_chain();
        // `y == 0`, so `y >= 0.5` is false and must NOT be reported verified.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0)],
            vec![Bound::new_allow_infinite(0.5, f32::INFINITY)],
        )
        .expect("valid spec");
        let options = CompleteMipOptions {
            mip: MipConfig {
                timeout_secs: 30.0,
                ..MipConfig::default()
            },
            ..CompleteMipOptions::default()
        };

        let report = verify_graph_complete_mip(&graph, &spec, &options).expect("encodable graph");
        assert!(!report.verified, "a false property must not verify");
        assert_eq!(
            report.disjuncts, 1,
            "the infinite upper side is vacuous and is not counted"
        );
        let reason = report.inconclusive.expect("an honest reason");
        assert!(
            reason.contains("NOT a counterexample"),
            "a feasible relaxed negation must not be dressed up as a refutation: {reason}"
        );
    }

    #[test]
    fn inflation_must_be_finite_and_non_negative() {
        let graph = cancelling_chain();
        let spec = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-1.0, 1.0)])
            .expect("valid spec");
        for inflation in [-1.0_f32, f32::NAN, f32::INFINITY] {
            let options = CompleteMipOptions {
                preactivation_inflation: inflation,
                ..CompleteMipOptions::default()
            };
            assert!(
                matches!(
                    verify_graph_complete_mip(&graph, &spec, &options),
                    Err(NyError::InvalidConfig(_))
                ),
                "{inflation} must be rejected"
            );
        }
    }

    #[test]
    fn spec_shape_mismatches_are_rejected() {
        let graph = cancelling_chain();
        let options = CompleteMipOptions::default();

        let wrong_inputs = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            vec![Bound::new(-1.0, 1.0)],
        )
        .expect("valid spec");
        assert!(matches!(
            verify_graph_complete_mip(&graph, &wrong_inputs, &options),
            Err(NyError::InvalidSpec(_))
        ));

        let wrong_outputs = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0)],
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        )
        .expect("valid spec");
        assert!(matches!(
            verify_graph_complete_mip(&graph, &wrong_outputs, &options),
            Err(NyError::InvalidSpec(_))
        ));
    }

    /// `y = relu(x) + relu(-x) = |x|` as `Linear -> ReLU -> Linear`.
    ///
    /// Unlike [`cancelling_chain`] the output actually varies with the input, so
    /// it can be sampled.
    fn abs_chain() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(linear(vec![1.0, -1.0], 2, 1, vec![0.0, 0.0])),
        ));
        graph.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer::new()),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l2",
            Layer::Linear(linear(vec![1.0, 1.0], 1, 2, vec![0.0])),
            vec!["r1".to_string()],
        ));
        graph.set_output("l2");
        graph
    }

    fn options_with_budget() -> CompleteMipOptions {
        CompleteMipOptions {
            mip: MipConfig {
                timeout_secs: 30.0,
                ..MipConfig::default()
            },
            ..CompleteMipOptions::default()
        }
    }

    /// SOUNDNESS: the encoded feasible set must ENCLOSE the network's real
    /// behaviour, so no reachable output may be refuted.
    ///
    /// For each sampled input the network's exact output `y_s` is taken from the
    /// facade's own propagation over the degenerate box `[x_s, x_s]`, then the
    /// terminal is asked to prove a property that EXCLUDES `y_s`. A correct
    /// encoding finds the true point and refuses; an encoding that had dropped
    /// reachable behaviour would prove the excluding property and hand back a
    /// FALSE `Verified`. This is the direction that matters: the opposite
    /// direction (a too-loose encoding) can only cost tightness.
    #[test]
    fn the_encoding_encloses_sampled_concrete_evaluations() {
        use ndarray::{ArrayD, IxDyn};
        use ny_tensor::BoundedTensor;

        let graph = abs_chain();
        let options = options_with_budget();

        for sample in [-1.0_f32, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0] {
            // The network's exact output at the sample: a degenerate box through
            // Linear/ReLU propagates exactly, which the assertion below pins.
            let point = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1]), sample),
                ArrayD::from_elem(IxDyn(&[1]), sample),
            )
            .expect("degenerate box");
            let concrete = graph.propagate_ibp(&point).expect("point propagation");
            let (lower, upper) = (concrete.lower()[0], concrete.upper()[0]);
            let truth = sample.abs();
            // ny rounds every propagated endpoint OUTWARD, so a degenerate box
            // comes back as a hair-wide interval rather than a point. It must
            // still be tight, and it must enclose the true value.
            assert!(
                upper - lower < 1e-4,
                "a degenerate box through Linear/ReLU must stay tight at x = {sample}: \
                 [{lower}, {upper}]"
            );
            assert!(
                lower <= truth && truth <= upper,
                "the propagated interval must enclose |{sample}| = {truth}: [{lower}, {upper}]"
            );

            let spec = VerificationSpec::new(
                vec![Bound::new(sample, sample)],
                // Excludes the true output by a margin far above any rounding.
                vec![Bound::new_allow_infinite(truth + 0.25, f32::INFINITY)],
            )
            .expect("valid spec");
            let report =
                verify_graph_complete_mip(&graph, &spec, &options).expect("encodable graph");
            assert!(
                !report.verified,
                "the encoding dropped the reachable output {truth} at x = {sample}, which would \
                 make a false property provable: {report:?}"
            );

            // And the same point INSIDE a property is provable, so the refusal
            // above is not simply a terminal that can never decide anything.
            let containing = VerificationSpec::new(
                vec![Bound::new(sample, sample)],
                vec![Bound::new(truth - 0.25, truth + 0.25)],
            )
            .expect("valid spec");
            let report =
                verify_graph_complete_mip(&graph, &containing, &options).expect("encodable graph");
            assert!(
                report.verified,
                "a property containing the true output at x = {sample} must be provable: \
                 {report:?}"
            );
        }
    }

    /// The same enclosure question over a non-degenerate box: the true range of
    /// `|x|` on `[-1, 1]` is `[0, 1]`, so a property must be provable exactly
    /// when it contains that range.
    #[test]
    fn the_encoding_neither_over_nor_under_approximates_a_real_range() {
        let graph = abs_chain();
        let options = options_with_budget();
        let input = vec![Bound::new(-1.0, 1.0)];

        let containing =
            VerificationSpec::new(input.clone(), vec![Bound::new(-0.001, 1.001)]).expect("spec");
        assert!(
            verify_graph_complete_mip(&graph, &containing, &options)
                .expect("encodable")
                .verified,
            "[0, 1] is the true range, so [-0.001, 1.001] must be provable"
        );

        // Cutting the top off the true range must not be provable.
        let cuts_the_top =
            VerificationSpec::new(input.clone(), vec![Bound::new(-0.001, 0.5)]).expect("spec");
        assert!(
            !verify_graph_complete_mip(&graph, &cuts_the_top, &options)
                .expect("encodable")
                .verified,
            "|x| reaches 1 on [-1, 1], so an upper bound of 0.5 must not be provable"
        );

        // Cutting the bottom off must not be provable either.
        let cuts_the_bottom =
            VerificationSpec::new(input, vec![Bound::new(0.5, 1.001)]).expect("spec");
        assert!(
            !verify_graph_complete_mip(&graph, &cuts_the_bottom, &options)
                .expect("encodable")
                .verified,
            "|x| reaches 0 at x = 0, so a lower bound of 0.5 must not be provable"
        );
    }

    /// The terminal's own finite-input-box guard is defence in depth: pin the
    /// upstream invariant it backs up, so the guard is not quietly the only
    /// thing standing between an unbounded box and a Big-M range.
    #[test]
    fn verification_spec_already_rejects_an_infinite_input_box() {
        assert!(
            VerificationSpec::new(
                vec![Bound::new_allow_infinite(f32::NEG_INFINITY, 1.0)],
                vec![Bound::new(-1.0, 1.0)],
            )
            .is_err(),
            "an unbounded input box has no Big-M encoding and must not reach the terminal"
        );
    }
}
