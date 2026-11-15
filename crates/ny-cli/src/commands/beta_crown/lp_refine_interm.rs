// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #relusplitter-lp-refine (wave-6 C7) — the rigorous LP ROOT
//! intermediate-bound refiner (`bab.mip_refine_intermediates`).
//!
//! Diagnosis and pricing: `scratchpad/vbs/diag10/build_relusplitter-lp-refine.md`
//! (offline, zero box). On relusplitter_2026 `model_2_2` the LP-refined
//! pre-activation layers 1-3 move the triangle-LP root from -11.14 (CROWN
//! intermediates) to -3.30, against ny's printed root of -20.15; an idealised
//! LP branch-and-bound over the refined boxes closes 8 of 9 clauses in <= 57
//! domains each, where the same tree over CROWN boxes does not close clause 0
//! in 400.
//!
//! # What it does
//!
//! Attached to the verifier by `dispatch.rs` ONLY when the typed key is armed,
//! and consulted ONCE by the multi-objective graph BaB root, after the
//! bootstrap has stored its per-node boxes:
//!
//! 1. [`extract_relu_chain`] accepts only a pure `Linear -> ReLU -> ... ->
//!    Linear` chain from the network input (conv-free MLPs; anything else
//!    declines with a stable reason). Weights and biases are the graph's own
//!    f32 parameters, taken exactly into f64.
//! 2. The stored boxes of the chain's pre-activation nodes are the LP's
//!    starting boxes (sound by the root's own construction).
//! 3. Pre-activation layers `1..=layers` are refined layer-progressively by
//!    [`LpTightener::tighten_layer_rigorous_prioritized`]: one layer LP per
//!    layer (outward-rounded triangle rows over the ALREADY-refined earlier
//!    layers), loosest triangle first, every accepted value a RIGOROUS bound
//!    (Neumaier–Shcherbina or AY's exact rim; a float optimum is never used),
//!    rounded outward to f32 and never wider than its input. Layer 0 is affine
//!    in the input box — its interval is already exact — so it is never
//!    re-solved.
//! 4. The refined boxes (and the matching ReLU output boxes
//!    `[max(l,0), max(u,0)]`) are returned as candidates; the ROOT intersects
//!    them shrink-only.
//!
//! # Soundness
//!
//! Every candidate is a sound outer enclosure of the node over the root input
//! box: the LP's feasible set contains the true ReLU graph over sound boxes,
//! each bound is a rigorous bound on that LP, and the f64 -> f32 casts round
//! outward. The ROOT intersects, never replaces. Nothing here can emit a
//! verdict; the lever can only turn `unknown` into `unsat`.
//!
//! # Budget and memory
//!
//! The refiner spends at most `budget_frac` of the root window that remains
//! when it starts (absolute deadline, checked between neurons and installed
//! on every AY solve; each solve is additionally capped). Before building any
//! LP it charges `LP_BYTES_PER_NNZ` per constraint nonzero against a quarter
//! of the process memory headroom (MemAvailable-derived when there is no
//! cgroup — competition runs uncapped) and declines rather than risk a global
//! OOM on this unified-memory host.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ny_core::Bound;
use ny_mip::{LpTightener, MipBackend, MipConfig};
use ny_propagate::beta_crown::root_interm_refine::{
    RootIntermRefineOutcome, RootIntermRefineRequest, RootIntermRefineTelemetry, RootIntermRefiner,
};
use ny_propagate::{GraphNetwork, Layer, MipRefineIntermediates, NETWORK_INPUT};
use ny_tensor::BoundedTensor;

/// Below this remaining slice the refiner declines instead of starting.
const MIN_SLICE: Duration = Duration::from_millis(250);
/// Ceiling on one rigorous solve (float lane + NS, or the exact rim).
const PER_SOLVE_CAP_SECS: f64 = 5.0;
/// Bytes charged per LP constraint nonzero (float matrix + LU fill + the exact
/// rim's rational side store, generously).
const LP_BYTES_PER_NNZ: u64 = 512;
/// Admission ceiling when no memory headroom is observable.
const UNOBSERVED_HEADROOM_CAP: u64 = 2 << 30;

/// A pure fully-connected ReLU chain taken exactly from the graph.
#[derive(Debug, Clone)]
pub(crate) struct ReluChain {
    /// Row-major `[out x in]` weights per affine layer (exact f32 values).
    pub(crate) weights: Vec<Vec<f64>>,
    /// Biases per affine layer (exact f32 values; zero when absent).
    pub(crate) biases: Vec<Vec<f64>>,
    /// `[input, hidden..., output]`.
    pub(crate) layer_dims: Vec<usize>,
    /// Graph node whose OUTPUT is hidden layer `h`'s pre-activation.
    pub(crate) pre_nodes: Vec<String>,
    /// The ReLU node consuming `pre_nodes[h]`.
    pub(crate) relu_nodes: Vec<String>,
}

impl ReluChain {
    /// Number of hidden (ReLU) layers.
    pub(crate) fn hidden(&self) -> usize {
        self.pre_nodes.len()
    }

    /// LP constraint nonzeros for a layer LP up to `target` (inclusive), plus
    /// the triangle rows of the earlier layers.
    fn lp_nonzeros(&self, target: usize) -> u64 {
        let mut nnz = 0u64;
        for layer in 0..=target {
            let fan_in = self.layer_dims[layer] as u64;
            let fan_out = self.layer_dims[layer + 1] as u64;
            // Dense affine rows plus the output column of each row.
            nnz = nnz.saturating_add(fan_out.saturating_mul(fan_in.saturating_add(1)));
            if layer < target {
                // Two triangle rows of two entries per hidden neuron.
                nnz = nnz.saturating_add(fan_out.saturating_mul(4));
            }
        }
        nnz
    }
}

/// Accept only `input -> Linear -> ReLU -> Linear -> ... -> ReLU -> Linear`
/// with exactly one input per node, every node on the chain, and the chain's
/// last Linear as the graph output. Declines with a stable reason otherwise.
pub(crate) fn extract_relu_chain(graph: &GraphNetwork) -> Result<ReluChain, &'static str> {
    let order = graph.exec_order().map_err(|_| "no_exec_order")?;
    if order.len() < 3 || order.len() % 2 == 0 {
        return Err("not_relu_chain");
    }
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    let mut layer_dims = Vec::new();
    let mut pre_nodes = Vec::new();
    let mut relu_nodes = Vec::new();
    let mut previous = NETWORK_INPUT.to_string();
    for (position, name) in order.iter().enumerate() {
        let node = graph.node(name).ok_or("missing_node")?;
        if node.inputs().len() != 1 || node.inputs()[0] != previous {
            return Err("not_relu_chain");
        }
        let expect_linear = position % 2 == 0;
        match (expect_linear, node.layer()) {
            (true, Layer::Linear(linear)) => {
                let w = linear.weight();
                let (fan_out, fan_in) = w.dim();
                if fan_out == 0 || fan_in == 0 {
                    return Err("empty_layer");
                }
                let previous_out = layer_dims.last().copied();
                match previous_out {
                    None => layer_dims.push(fan_in),
                    Some(out) if out == fan_in => {}
                    Some(_) => return Err("shape_mismatch"),
                }
                layer_dims.push(fan_out);
                let mut row_major = Vec::with_capacity(fan_out * fan_in);
                for i in 0..fan_out {
                    for j in 0..fan_in {
                        let value = w[[i, j]];
                        if !value.is_finite() {
                            return Err("nonfinite_parameter");
                        }
                        row_major.push(f64::from(value));
                    }
                }
                weights.push(row_major);
                let bias = match linear.bias() {
                    Some(b) => {
                        if b.len() != fan_out || b.iter().any(|v| !v.is_finite()) {
                            return Err("bad_bias");
                        }
                        b.iter().map(|&v| f64::from(v)).collect()
                    }
                    None => vec![0.0; fan_out],
                };
                biases.push(bias);
            }
            (false, Layer::ReLU(_)) => {
                pre_nodes.push(previous.clone());
                relu_nodes.push(name.clone());
            }
            _ => return Err("not_relu_chain"),
        }
        previous.clone_from(name);
    }
    let output = graph.output_name();
    if !output.is_empty() && output != previous {
        return Err("not_relu_chain");
    }
    if pre_nodes.is_empty() {
        return Err("no_hidden_layer");
    }
    Ok(ReluChain {
        weights,
        biases,
        layer_dims,
        pre_nodes,
        relu_nodes,
    })
}

/// Flatten a box into `Bound`s of the expected length; `None` on a length
/// mismatch or any non-finite / inverted entry.
fn tensor_bounds(tensor: &BoundedTensor, expected: usize) -> Option<Vec<Bound>> {
    let lower = tensor.lower();
    let upper = tensor.upper();
    if lower.len() != expected || upper.len() != expected {
        return None;
    }
    lower
        .iter()
        .zip(upper.iter())
        .map(|(&l, &u)| Bound::try_new(l, u).ok())
        .collect()
}

/// The rigorous LP root refiner.
pub(crate) struct LpRootIntermRefiner {
    config: MipRefineIntermediates,
}

impl LpRootIntermRefiner {
    pub(crate) const fn new(config: MipRefineIntermediates) -> Self {
        Self { config }
    }
}

fn declined(reason: &'static str, started: Instant) -> RootIntermRefineOutcome {
    RootIntermRefineOutcome {
        candidates: HashMap::new(),
        telemetry: RootIntermRefineTelemetry {
            reason,
            secs: started.elapsed().as_secs_f64(),
            ..RootIntermRefineTelemetry::default()
        },
    }
}

impl RootIntermRefiner for LpRootIntermRefiner {
    fn name(&self) -> &'static str {
        "lp"
    }

    fn refine(&self, req: &RootIntermRefineRequest<'_>) -> RootIntermRefineOutcome {
        let started = Instant::now();
        if self.config.validate().is_err() {
            return declined("invalid_config", started);
        }
        let chain = match extract_relu_chain(req.graph) {
            Ok(chain) => chain,
            Err(reason) => return declined(reason, started),
        };
        // Layer 0 is exact already; refine 1..=top.
        if chain.hidden() < 2 {
            return declined("no_refinable_layer", started);
        }
        let top = self.config.layers.min(chain.hidden() - 1);
        let Some(input_bounds) = tensor_bounds(req.input_bounds, chain.layer_dims[0]) else {
            return declined("bad_input_box", started);
        };
        let mut interm: Vec<Vec<Bound>> = Vec::with_capacity(top + 1);
        for (h, name) in chain.pre_nodes.iter().enumerate().take(top + 1) {
            let Some(stored) = req.node_bounds.get(name) else {
                return declined("missing_node_bounds", started);
            };
            let Some(bounds) = tensor_bounds(stored, chain.layer_dims[h + 1]) else {
                return declined("bad_node_bounds", started);
            };
            interm.push(bounds);
        }

        // Memory admission, before any LP is built.
        let need = chain.lp_nonzeros(top).saturating_mul(LP_BYTES_PER_NNZ);
        let cap = ny_propagate::network::crown_memory::process_memory_headroom_bytes()
            .map_or(UNOBSERVED_HEADROOM_CAP, |headroom| headroom / 4);
        if need > cap {
            return declined("memory_admission", started);
        }

        // Budget: a share of what remains of the root window, never unbounded.
        let Some(root_deadline) = req.deadline else {
            return declined("no_deadline", started);
        };
        let now = Instant::now();
        let slice = root_deadline
            .saturating_duration_since(now)
            .mul_f64(self.config.budget_frac);
        if slice < MIN_SLICE {
            return declined("no_budget", started);
        }
        let refine_deadline = now + slice;
        let per_solve = (slice.as_secs_f64() / 4.0).min(PER_SOLVE_CAP_SECS);
        let mip_config = MipConfig {
            backend: MipBackend::Ay,
            // `LpTightener` slices each solve at timeout_secs / 10.
            timeout_secs: per_solve * 10.0,
            max_tighten_per_layer: 0,
            lp_tighten: false,
            obbt_rounds: 0,
            ay_hard_deadline: Some(refine_deadline),
            ..MipConfig::default()
        };

        let mut telemetry = RootIntermRefineTelemetry {
            engaged: true,
            reason: "ok",
            ..RootIntermRefineTelemetry::default()
        };
        // Only the prefix the LPs can reference (layers 0..=top).
        let weights = chain.weights[..=top].to_vec();
        let biases = chain.biases[..=top].to_vec();
        let layer_dims = chain.layer_dims[..=top + 1].to_vec();
        let mut refined_through = 0usize;
        for layer in 1..=top {
            if Instant::now() >= refine_deadline {
                telemetry.deadline_hit = true;
                break;
            }
            let tightener = LpTightener::new(
                weights.clone(),
                biases.clone(),
                layer_dims.clone(),
                input_bounds.clone(),
                interm.clone(),
                mip_config,
            );
            match tightener.tighten_layer_rigorous_prioritized(layer, &interm[layer]) {
                Ok((bounds, report)) => {
                    telemetry.layers += 1;
                    telemetry.unstable_before += report.unstable_before;
                    telemetry.unstable_after += report.unstable_after;
                    telemetry.lp_solves += report.lp_solves;
                    telemetry.rigorous += report.rigorous;
                    telemetry.tightened_neurons += report.tightened;
                    telemetry.deadline_hit |= report.deadline_hit;
                    interm[layer] = bounds;
                    refined_through = layer;
                    if report.deadline_hit {
                        break;
                    }
                }
                Err(_) => {
                    // Earlier layers were refined from sound inputs and stay
                    // valid; this layer keeps its stored box.
                    telemetry.reason = "lp_error";
                    break;
                }
            }
        }

        let mut candidates = HashMap::new();
        for layer in 1..=refined_through {
            let lower: Vec<f64> = interm[layer].iter().map(|b| f64::from(b.lower())).collect();
            let upper: Vec<f64> = interm[layer].iter().map(|b| f64::from(b.upper())).collect();
            // relu is monotone: relu(z) in [max(l,0), max(u,0)] exactly.
            let relu_lower: Vec<f64> = lower.iter().map(|v| v.max(0.0)).collect();
            let relu_upper: Vec<f64> = upper.iter().map(|v| v.max(0.0)).collect();
            candidates.insert(chain.pre_nodes[layer].clone(), (lower, upper));
            candidates.insert(chain.relu_nodes[layer].clone(), (relu_lower, relu_upper));
        }
        telemetry.secs = started.elapsed().as_secs_f64();
        RootIntermRefineOutcome {
            candidates,
            telemetry,
        }
    }
}

#[cfg(test)]
#[path = "lp_refine_interm_tests.rs"]
mod tests;
