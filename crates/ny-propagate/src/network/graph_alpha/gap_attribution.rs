// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #gap-attribution — exact per-neuron decomposition of a CROWN bound's
//! looseness. Default-off diagnostics may publish a static, theorem-checked
//! root prior for advisory branching; it never participates in a certified
//! bound or verdict.
//!
//! Implements Theorem 1 of
//! `docs/THEORY_EXACT_GAP_ATTRIBUTION_AND_MAXMIN_ALPHA_2026-08-03.md`.
//!
//! For any concrete `x` in the domain, the difference between the true
//! objective and the CROWN linear bound telescopes exactly over neurons:
//!
//! ```text
//!   f(x) - B(x)  =  SUM_k SUM_j  nu_kj * [ relu(zhat_kj(x)) - r_kj(zhat_kj(x)) ]
//!                =  SUM_j g_j(x)          with every g_j(x) >= 0
//! ```
//!
//! where `nu_kj` is the backward coefficient the pass carried on ReLU k's
//! OUTPUT for this seed row (`intermediate.a_at_relu`), and `r_kj` is the
//! affine substitution the pass SELECTED at that neuron — the lower relaxation
//! when `nu >= 0`, the upper chord when `nu < 0` (`lower_row_slope_intercept`,
//! shared verbatim with `#binding-row-replay` so the two cannot drift).
//!
//! Evaluated at `x*`, the concretization corner that achieves the reported
//! bound, this yields `SUM_j g_j = f(x*) - lb_nominal` (Corollary 1) and hence
//! a SOUND CEILING on the improvement any method can obtain at this domain
//! (Corollary 2), a counterexample whenever `f(x*)` misses the threshold
//! (Corollary 3), and the relaxation/arithmetic split of Corollary 4.
//!
//! ## Why this is not `#binding-row-replay`
//!
//! That module replays the RELAXED forward — it reconstructs `B(x*)`, which is
//! what an alpha gradient needs. Theorem 1 needs the TRUE activations, so this
//! module runs a CONCRETE forward at the same `x*`. The two forwards differ at
//! exactly the ReLU nodes, and their per-neuron difference IS `g_j`.
//!
//! ## Self-checking by construction
//!
//! `residual = f(x*) - B(x*) - SUM_j g_j` must be zero in exact arithmetic.
//! It is retained and `verify_identity` refuses above tolerance. An
//! implementation that cannot reproduce Theorem 1 numerically is broken and
//! must not be trusted; in particular a non-affine, non-ReLU node in the graph
//! (which contributes relaxation this decomposition does not model) shows up
//! here as a non-zero residual rather than as a silently wrong attribution.

// DARK module: the exact producer and attribution-directed KFSB ranking remain
// default-off. Theorem 1 is checked before any prior publication, and every
// refusal preserves the historical ranking.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use ndarray::{Array1, ArrayD};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use super::binding_row_replay::{lower_row_slope_intercept, midpoint, point_box};
use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState};
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::{GraphNetwork, NETWORK_INPUT};

/// Poll every 256 scalar elements in the gated concrete/attribution loops.
/// This keeps deadline overshoot bounded without taxing ordinary propagation.
const GAP_ATTRIBUTION_DEADLINE_CHUNK: usize = 256;

fn check_gap_attribution_deadline(deadline: Option<Instant>, phase: &str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "#gap-attribution: deadline expired {phase}"
        )));
    }
    Ok(())
}

fn poll_gap_attribution_deadline(
    deadline: Option<Instant>,
    index: usize,
    phase: &str,
) -> Result<()> {
    if index.is_multiple_of(GAP_ATTRIBUTION_DEADLINE_CHUNK) {
        check_gap_attribution_deadline(deadline, phase)?;
    }
    Ok(())
}

/// Per-ReLU-node attribution of the bound's looseness.
#[derive(Debug, Clone)]
pub(crate) struct NodeGap {
    /// `g_j` at this node's full neuron width, in f64. Non-negative by
    /// Theorem 1; a negative entry is a bug and `verify_identity` will say so.
    pub(crate) g: Array1<f64>,
    /// Neurons strictly crossing (`l < 0 < u`) at this node.
    pub(crate) unstable: usize,
    /// Neurons carrying strictly positive attributed gap.
    pub(crate) live: usize,
}

/// Corollary 4's trichotomy for one row at one domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GapVerdict {
    /// `f(x*) < threshold`: `x*` is a concrete counterexample in the domain.
    /// Guidance only here — a verdict-bearing caller must re-validate it with
    /// an independent concrete forward, exactly as the PGD path does.
    Falsified { true_value: f64, threshold: f64 },
    /// The remaining gap is certified-arithmetic conservatism, not relaxation
    /// slack. Branching CANNOT close it at any depth.
    ArithmeticLimited { relaxation_gap: f64, needed: f64 },
    /// There is genuinely closable relaxation slack.
    RelaxationLimited { relaxation_gap: f64, needed: f64 },
}

/// Result of attributing one seed row's bound gap at one domain.
#[derive(Debug, Clone)]
pub(crate) struct GapAttribution {
    /// The seed row this attributes.
    pub(crate) row: usize,
    /// The row's concretization argmin corner.
    pub(crate) x_star: ArrayD<f32>,
    /// TRUE objective value at `x*` (concrete forward, no relaxation).
    pub(crate) f_x_star: f64,
    /// `B(x*)` = the row's linear bound evaluated at `x*`. For the argmin
    /// corner this is `lb_nominal` — the bound BEFORE the fold's directed
    /// rounding and certified-error debit.
    pub(crate) bound_at_x_star: f64,
    /// `SUM_j g_j` — the sound ceiling on improvement (Corollary 2).
    pub(crate) sum_g: f64,
    /// Per-node attribution.
    pub(crate) per_node: HashMap<String, NodeGap>,
    /// `f(x*) - B(x*) - SUM_j g_j`. Theorem 1 says zero.
    pub(crate) residual: f64,
}

impl GapAttribution {
    /// Theorem 1 as an executable check. `tol` is absolute; callers should
    /// scale it by the magnitudes involved (`f_x_star`, `bound_at_x_star`)
    /// because the decomposition accumulates over every neuron in the graph.
    pub(crate) fn verify_identity(&self, tol: f64) -> Result<()> {
        if !self.residual.is_finite() {
            return Err(NyError::InvalidSpec(format!(
                "#gap-attribution: non-finite residual for row {} (f={}, B={}, sum_g={})",
                self.row, self.f_x_star, self.bound_at_x_star, self.sum_g
            )));
        }
        if self.residual.abs() > tol {
            return Err(NyError::InvalidSpec(format!(
                "#gap-attribution: Theorem 1 violated for row {}: \
                 f(x*)={:.9e} - B(x*)={:.9e} = {:.9e}, but sum_j g_j={:.9e} \
                 (residual {:.9e} > tol {:.9e}). Either the attribution is wrong \
                 or the graph contains a non-affine non-ReLU node whose relaxation \
                 this decomposition does not model.",
                self.row,
                self.f_x_star,
                self.bound_at_x_star,
                self.f_x_star - self.bound_at_x_star,
                self.sum_g,
                self.residual,
                tol
            )));
        }
        if let Some((name, idx, v)) = self.first_negative() {
            return Err(NyError::InvalidSpec(format!(
                "#gap-attribution: negative attributed gap {v:.9e} at '{name}'[{idx}] \
                 — every g_j is non-negative by Theorem 1, so the selected relaxation \
                 does not match the coefficient's sign."
            )));
        }
        Ok(())
    }

    fn first_negative(&self) -> Option<(&str, usize, f64)> {
        // Deterministic order so the diagnostic is reproducible across runs
        // (HashMap iteration order is not).
        let mut names: Vec<&String> = self.per_node.keys().collect();
        names.sort_unstable();
        for name in names {
            let ng = &self.per_node[name];
            for (i, &v) in ng.g.iter().enumerate() {
                if v < 0.0 {
                    return Some((name.as_str(), i, v));
                }
            }
        }
        None
    }

    /// Corollary 4. `lb_sound` is the bound the fold actually reported for
    /// this row (after directed rounding and the certified-error debit), and
    /// `threshold` the value it must clear.
    pub(crate) fn classify(&self, lb_sound: f64, threshold: f64) -> GapVerdict {
        if self.f_x_star < threshold {
            return GapVerdict::Falsified {
                true_value: self.f_x_star,
                threshold,
            };
        }
        let needed = threshold - lb_sound;
        if self.sum_g < needed {
            GapVerdict::ArithmeticLimited {
                relaxation_gap: self.sum_g,
                needed,
            }
        } else {
            GapVerdict::RelaxationLimited {
                relaxation_gap: self.sum_g,
                needed,
            }
        }
    }

    /// The certified-arithmetic component `E = B(x*) - lb_sound` (Corollary 4).
    /// Non-negative whenever the fold's rounding is genuinely outward.
    pub(crate) fn certified_error(&self, lb_sound: f64) -> f64 {
        self.bound_at_x_star - lb_sound
    }

    /// Deduction 3: the smallest number of neurons whose attributed gaps sum to
    /// at least `needed`, or `None` if the whole attribution cannot reach it.
    ///
    /// HEURISTIC — this assumes splitting neuron `j` removes at most `g_j` from
    /// the gap. It is an exactly-accounted estimate, not a theorem; see the
    /// stated failure mode in the theory doc (a child re-optimises alpha and
    /// may redistribute attribution).
    pub(crate) fn attribution_depth(&self, needed: f64) -> Option<usize> {
        if needed <= 0.0 {
            return Some(0);
        }
        let mut gaps: Vec<f64> = self
            .per_node
            .values()
            .flat_map(|ng| ng.g.iter().copied())
            .filter(|v| *v > 0.0)
            .collect();
        gaps.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut acc = 0.0;
        for (k, v) in gaps.iter().enumerate() {
            acc += v;
            if acc >= needed {
                return Some(k + 1);
            }
        }
        None
    }

    /// Deduction 2: the `k` neurons carrying the most attributed gap, as
    /// `(node, index, g_j)`, descending. The branching score the theory
    /// proposes in place of the triangle-intercept heuristic.
    pub(crate) fn top_neurons(&self, k: usize) -> Vec<(String, usize, f64)> {
        let mut all: Vec<(String, usize, f64)> = self
            .per_node
            .iter()
            .flat_map(|(name, ng)| {
                ng.g.iter()
                    .enumerate()
                    .filter(|(_, v)| **v > 0.0)
                    .map(move |(i, v)| (name.clone(), i, *v))
            })
            .collect();
        all.sort_unstable_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Tie-break on (node, index) so the ranking is deterministic.
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        all.truncate(k);
        all
    }

    /// Total neurons carrying strictly positive attributed gap. The
    /// concentration statistic §3.3's prediction is about: a small number here
    /// supports attribution-directed branching, a large one supports Deduction
    /// 3's budget redirection instead.
    pub(crate) fn live_neurons(&self) -> usize {
        self.per_node.values().map(|ng| ng.live).sum()
    }

    /// Total strictly-crossing neurons — the split-candidate pool size.
    pub(crate) fn unstable_neurons(&self) -> usize {
        self.per_node.values().map(|ng| ng.unstable).sum()
    }
}

impl GraphNetwork {
    /// Attribute one seed row's CROWN bound gap per neuron (Theorem 1).
    ///
    /// `intermediate` and `alpha_state` must come from the SAME fold whose
    /// bound is being attributed — the `dag_alpha_backward_pass_with_intermediates`
    /// output at the current alpha iterate — exactly as `#binding-row-replay`
    /// requires. `row` indexes that fold's SEED row space, and `objective` must
    /// be the seed row's coefficient vector over the network output, so that
    /// `f(x*) = objective . output(x*)`. A mismatch between the two shows up as
    /// a non-zero `residual` rather than as a plausible wrong answer.
    pub(crate) fn attribute_row_gap(
        &self,
        input: &BoundedTensor,
        alpha_state: &GraphAlphaState,
        intermediate: &GraphAlphaCrownIntermediate,
        row: usize,
        objective: &[f32],
    ) -> Result<GapAttribution> {
        self.attribute_row_gap_impl(input, alpha_state, intermediate, row, objective, None)
    }

    /// Deadline-authoritative attribution used by the gated root producer.
    /// Polling lives only on this optional path; ordinary bound propagation is
    /// unchanged. The first check precedes shape reads and all allocations.
    pub(crate) fn attribute_row_gap_until(
        &self,
        input: &BoundedTensor,
        alpha_state: &GraphAlphaState,
        intermediate: &GraphAlphaCrownIntermediate,
        row: usize,
        objective: &[f32],
        deadline: Instant,
    ) -> Result<GapAttribution> {
        self.attribute_row_gap_impl(
            input,
            alpha_state,
            intermediate,
            row,
            objective,
            Some(deadline),
        )
    }

    fn attribute_row_gap_impl(
        &self,
        input: &BoundedTensor,
        alpha_state: &GraphAlphaState,
        intermediate: &GraphAlphaCrownIntermediate,
        row: usize,
        objective: &[f32],
        deadline: Option<Instant>,
    ) -> Result<GapAttribution> {
        check_gap_attribution_deadline(deadline, "before attribution")?;
        // === 1. x*: sign readout of the row's final input-affine coefficients ===
        //
        // Convention is `concretize_scalar_f64`'s and matches
        // #binding-row-replay verbatim: `a > 0` pays the lower corner, `a < 0`
        // the upper, `a == 0` the midpoint (contributes nothing either way).
        let final_a = intermediate.final_bounds.lower_a();
        let final_b = intermediate.final_bounds.lower_b();
        if row >= final_a.nrows() {
            return Err(NyError::InvalidSpec(format!(
                "#gap-attribution: row {} out of range ({} seed rows)",
                row,
                final_a.nrows()
            )));
        }
        check_gap_attribution_deadline(deadline, "before flattening the input box")?;
        let input_flat = input.flatten();
        check_gap_attribution_deadline(deadline, "after flattening the input box")?;
        let n_in = input_flat.len();
        if final_a.ncols() != n_in {
            return Err(NyError::ShapeMismatch {
                expected: vec![n_in],
                got: vec![final_a.ncols()],
            });
        }
        let xl = input_flat.lower();
        let xu = input_flat.upper();
        check_gap_attribution_deadline(deadline, "before allocating the input witness")?;
        let mut xs: Vec<f32> = Vec::with_capacity(n_in);
        for j in 0..n_in {
            poll_gap_attribution_deadline(deadline, j, "while constructing the input witness")?;
            let a = final_a[[row, j]];
            xs.push(if a > 0.0 {
                xl[[j]]
            } else if a < 0.0 {
                xu[[j]]
            } else {
                let l = xl[[j]] as f64;
                (l + 0.5 * (xu[[j]] as f64 - l)) as f32
            });
        }
        check_gap_attribution_deadline(deadline, "after constructing the input witness")?;

        // B(x*) in f64. For the argmin corner this equals lb_nominal.
        let mut bound_at_x_star = final_b[row] as f64;
        for j in 0..n_in {
            poll_gap_attribution_deadline(deadline, j, "while evaluating the affine bound")?;
            bound_at_x_star += final_a[[row, j]] as f64 * xs[j] as f64;
        }
        check_gap_attribution_deadline(deadline, "after evaluating the affine bound")?;
        let x_star = ArrayD::from_shape_vec(input.lower().raw_dim(), xs)
            .map_err(|e| NyError::InvalidSpec(format!("#gap-attribution: x* reshape: {e}")))?;

        // === 2. CONCRETE forward at x*, recording the true pre-activations ===
        //
        // This is the one thing #binding-row-replay does not do: it applies the
        // SELECTED relaxation at each ReLU (reconstructing B(x*)), whereas
        // Theorem 1 needs relu() itself. Non-ReLU nodes replay through their
        // existing IBP dispatch on a degenerate box, collapsed to the midpoint
        // after each node so per-node outward rounding cannot compound — the
        // same idiom, so the two forwards differ at ReLU nodes and nowhere else.
        check_gap_attribution_deadline(deadline, "before resolving graph execution order")?;
        let exec_order = self.exec_order()?;
        check_gap_attribution_deadline(deadline, "after resolving graph execution order")?;
        check_gap_attribution_deadline(deadline, "before allocating concrete-forward state")?;
        let mut values: HashMap<&str, ArrayD<f32>> = HashMap::with_capacity(exec_order.len());
        // True pre-activation at x*, per ReLU node.
        let mut zhat: HashMap<String, Array1<f64>> = HashMap::new();

        let value_of = |values: &HashMap<&str, ArrayD<f32>>,
                        name: &str,
                        x_star: &ArrayD<f32>|
         -> Result<ArrayD<f32>> {
            if name == NETWORK_INPUT {
                return Ok(x_star.clone());
            }
            values.get(name).cloned().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "#gap-attribution: value for input node '{name}' not computed yet"
                ))
            })
        };

        for node_name in exec_order {
            check_gap_attribution_deadline(deadline, "before a concrete graph-node dispatch")?;
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {node_name}")))?;

            let out: ArrayD<f32> = if matches!(node.layer, Layer::ReLU(_)) {
                let src = node.inputs.first().map(String::as_str).ok_or_else(|| {
                    NyError::InvalidSpec(format!("ReLU node '{node_name}' has no input"))
                })?;
                let mut z = value_of(&values, src, &x_star)?;
                check_gap_attribution_deadline(
                    deadline,
                    "before allocating a concrete ReLU pre-activation record",
                )?;
                let mut z_record = Vec::with_capacity(z.len());
                for (index, &value) in z.iter().enumerate() {
                    poll_gap_attribution_deadline(
                        deadline,
                        index,
                        "while recording concrete ReLU pre-activations",
                    )?;
                    z_record.push(f64::from(value));
                }
                zhat.insert(node_name.clone(), Array1::from(z_record));
                // The TRUE activation — this is the whole difference from the
                // relaxed replay. Mutate the owned input clone in place to
                // avoid allocating a second activation-sized tensor.
                for (index, value) in z.iter_mut().enumerate() {
                    poll_gap_attribution_deadline(
                        deadline,
                        index,
                        "while applying the concrete ReLU",
                    )?;
                    *value = if *value > 0.0 { *value } else { 0.0 };
                }
                z
            } else {
                let bt = match node.inputs.len() {
                    1 => {
                        let a = value_of(&values, &node.inputs[0], &x_star)?;
                        node.layer.propagate_ibp(&point_box(&a)?)?
                    }
                    2 => {
                        let a = value_of(&values, &node.inputs[0], &x_star)?;
                        let b = value_of(&values, &node.inputs[1], &x_star)?;
                        node.layer
                            .propagate_ibp_binary(&point_box(&a)?, &point_box(&b)?)?
                    }
                    3 => {
                        let a = value_of(&values, &node.inputs[0], &x_star)?;
                        let b = value_of(&values, &node.inputs[1], &x_star)?;
                        let c = value_of(&values, &node.inputs[2], &x_star)?;
                        node.layer.propagate_ibp_ternary(
                            &point_box(&a)?,
                            &point_box(&b)?,
                            &point_box(&c)?,
                        )?
                    }
                    k => {
                        return Err(NyError::UnsupportedOp(format!(
                            "#gap-attribution: {k}-ary node '{node_name}' ({}) unsupported",
                            node.layer.layer_type()
                        )));
                    }
                };
                midpoint(&bt)
            };
            check_gap_attribution_deadline(deadline, "after a concrete graph-node dispatch")?;
            values.insert(node_name.as_str(), out);
            check_gap_attribution_deadline(
                deadline,
                "after recording a concrete graph-node output",
            )?;
        }

        // === 3. f(x*) = objective . output(x*), concretely ===
        let output_name = if self.output_node.is_empty() {
            exec_order.last().map(String::as_str).unwrap_or_default()
        } else {
            self.output_node.as_str()
        };
        let out_val = values.get(output_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "#gap-attribution: output node '{output_name}' produced no value"
            ))
        })?;
        if objective.len() != out_val.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_val.len()],
                got: vec![objective.len()],
            });
        }
        let mut f_x_star = 0.0f64;
        for (index, (&coefficient, &value)) in objective.iter().zip(out_val.iter()).enumerate() {
            poll_gap_attribution_deadline(
                deadline,
                index,
                "while evaluating the concrete objective",
            )?;
            f_x_star += f64::from(coefficient) * f64::from(value);
        }
        check_gap_attribution_deadline(deadline, "after evaluating the concrete objective")?;

        // === 4. g_j = nu_j * [ relu(zhat_j) - r_j(zhat_j) ] ===
        //
        // r_j is the relaxation the BACKWARD selected, so it is read with the
        // same `lower_row_slope_intercept` the replay uses — sharing the
        // function is what keeps this honest: if the walk's arms change, both
        // move together and the identity check stays meaningful.
        check_gap_attribution_deadline(deadline, "before allocating per-ReLU attribution")?;
        let mut per_node: HashMap<String, NodeGap> =
            HashMap::with_capacity(intermediate.pre_relu_bounds.len());
        let mut sum_g = 0.0f64;
        for (name, (pre_l, pre_u)) in &intermediate.pre_relu_bounds {
            check_gap_attribution_deadline(deadline, "before attributing a ReLU node")?;
            let n = pre_l.len();
            let Some(a_mat) = intermediate.a_at_relu(name) else {
                return Err(NyError::InvalidSpec(format!(
                    "#gap-attribution: no DENSE a_at_relu for '{name}' (beta-sparse or \
                     truncated capture) — attribution refuses partial state"
                )));
            };
            let Some(z) = zhat.get(name) else {
                return Err(NyError::InvalidSpec(format!(
                    "#gap-attribution: ReLU node '{name}' has captured bounds but was not \
                     reached by the concrete forward — the intermediate does not match \
                     this graph"
                )));
            };
            if pre_u.len() != n || a_mat.ncols() != n || z.len() != n || row >= a_mat.nrows() {
                return Err(NyError::ShapeMismatch {
                    expected: vec![n, n, n, n, row + 1],
                    got: vec![
                        pre_l.len(),
                        pre_u.len(),
                        a_mat.ncols(),
                        z.len(),
                        a_mat.nrows(),
                    ],
                });
            }

            // Alpha readout mirrors the replay's measured-shape stride rule
            // (#channel-alpha-grad): per-neuron, or a channel-shared alpha
            // broadcast over the node's recorded conv geometry. Anything else
            // refuses rather than guessing a layout.
            let row_alpha = alpha_state.alpha_for_row(name, row);
            let alpha_arr = row_alpha.as_ref().map(|r| r.as_array());
            let alpha_stride: usize = match alpha_arr {
                None => 1,
                Some(a) if a.len() == n => 1,
                Some(a) => {
                    let Some((_channels, spatial)) =
                        alpha_state.channel_reduction_geometry(name, a.len(), n)
                    else {
                        return Err(NyError::UnsupportedConfiguration(format!(
                            "#gap-attribution: alpha width {} at '{name}' is neither per-neuron \
                             (width {}) nor channel-shared over the recorded geometry",
                            a.len(),
                            n
                        )));
                    };
                    spatial
                }
            };

            check_gap_attribution_deadline(deadline, "before allocating a ReLU gap row")?;
            let mut g = Array1::<f64>::zeros(n);
            let mut unstable = 0usize;
            let mut live = 0usize;
            for i in 0..n {
                poll_gap_attribution_deadline(deadline, i, "while attributing ReLU neurons")?;
                let l = pre_l[i];
                let u = pre_u[i];
                if l < 0.0 && u > 0.0 {
                    unstable += 1;
                }
                let nu = a_mat[[row, i]];
                if nu == 0.0 || !nu.is_finite() {
                    continue;
                }
                let (s, t) =
                    lower_row_slope_intercept(l, u, nu, alpha_arr.map(|a| a[i / alpha_stride]))?;
                if !s.is_finite() || !t.is_finite() {
                    // The walk's non-finite arms carry an infinite intercept;
                    // the bound is vacuous there and no finite attribution
                    // exists. Leave the entry at zero and let the residual
                    // report the discrepancy rather than fabricating a number.
                    continue;
                }
                let zh = z[i];
                let relu_v = if zh > 0.0 { zh } else { 0.0 };
                let relax_v = s as f64 * zh + t as f64;
                let gj = nu as f64 * (relu_v - relax_v);
                g[i] = gj;
                sum_g += gj;
                if gj > 0.0 {
                    live += 1;
                }
            }
            per_node.insert(name.clone(), NodeGap { g, unstable, live });
            check_gap_attribution_deadline(deadline, "after attributing a ReLU node")?;
        }

        check_gap_attribution_deadline(deadline, "after attribution")?;
        let residual = f_x_star - bound_at_x_star - sum_g;

        Ok(GapAttribution {
            row,
            x_star,
            f_x_star,
            bound_at_x_star,
            sum_g,
            per_node,
            residual,
        })
    }
}

// ===========================================================================
// #gap-attribution root probe (build-plan step 3)
// ===========================================================================

/// Exact `"1"` diagnostic gate, matching every other dark gate in this tree.
/// The branching and ranking-diagnostic gates also arm their required producer
/// so either experiment is complete with one switch; all three unset leaves the
/// probe off and the run byte-identical.
const GAP_PROBE_ENV: &str = "NY_GAP_ATTRIBUTION";
/// How many seed rows to attribute. Each row costs one concrete forward plus a
/// walk of every ReLU, so this is deliberately small by default.
const GAP_PROBE_ROWS_ENV: &str = "NY_GAP_ATTRIBUTION_ROWS";
/// Hard resident-row ceiling for the attribution fold.
///
/// The intermediate stores one dense coefficient row across every ReLU.  The
/// old implementation first captured every property row (99 on CIFAR100) and
/// only then attributed the requested three.  Keeping the cap here, at seed
/// construction, bounds the dense capture itself rather than merely its
/// post-processing.
const GAP_PROBE_MAX_ROWS: usize = 3;
/// Wall budget for the probe's OWN fold, in seconds.
///
/// This private cap is intersected with the exact outer BaB deadline. It never
/// mints time; an exhausted outer authority refuses before seed construction.
const GAP_PROBE_BUDGET_ENV: &str = "NY_GAP_ATTRIBUTION_BUDGET_SECS";

pub(crate) fn root_gap_probe_enabled() -> bool {
    std::env::var(GAP_PROBE_ENV).is_ok_and(|v| v == "1")
        || attribution_branching_enabled()
        || attribution_branch_diag_enabled()
}

fn probe_budget() -> std::time::Duration {
    // A branching experiment runs on the scored path, so a malformed/degraded
    // fold must not inherit the old ten-minute diagnostic allowance. The
    // measured full 100-row CIFAR100 probe was 2.96 s; the <=3-row fold should
    // fit comfortably, and an explicit env override remains available for
    // slower diagnostic hardware.
    let default_secs = if attribution_branching_enabled() || attribution_branch_diag_enabled() {
        5
    } else {
        600
    };
    let secs = std::env::var(GAP_PROBE_BUDGET_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_secs);
    std::time::Duration::from_secs(secs.clamp(1, 3600))
}

fn probe_rows(output_dim: usize) -> usize {
    let requested = std::env::var(GAP_PROBE_ROWS_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(GAP_PROBE_MAX_ROWS);
    probe_row_count(requested, output_dim)
}

fn probe_row_count(requested: usize, available: usize) -> usize {
    if available == 0 {
        0
    } else {
        requested.max(1).min(available).min(GAP_PROBE_MAX_ROWS)
    }
}

fn compose_probe_deadline(
    now: Instant,
    private_budget: std::time::Duration,
    global_deadline: Option<Instant>,
    require_global: bool,
) -> Option<Instant> {
    let private = now.checked_add(private_budget)?;
    if require_global {
        global_deadline.map(|global| private.min(global))
    } else {
        Some(global_deadline.map_or(private, |global| private.min(global)))
    }
}

// ===========================================================================
// #attr-branch: the root attribution as a branching prior (Deduction 2)
// ===========================================================================

/// Arms the branching prior. Exact `"1"`; default off.
const ATTR_BRANCH_ENV: &str = "NY_ATTR_BRANCH";

/// Per-neuron branching priority derived from the root attribution.
///
/// Measured motivation (§6c of the theory doc): at the cifar100 root, **8 of
/// 1,366 unstable neurons carry half** the binding margin row's gap and ~50
/// carry ninety percent, while kFSB examines 7 candidates ranked by a triangle
/// intercept that discards the `Λⱼ ≥ 0` branch entirely.
///
/// This is a **root-only prior**, not a per-domain score. Attribution shifts as
/// BaB splits and re-optimises α, so the consumer refuses this publication for
/// every domain with non-zero split depth. Computing a fresh score per domain
/// would need a witness forward per node; until such a bounded producer exists,
/// descendants retain historical kFSB rather than treating stale root `gⱼ` as
/// current evidence.
///
/// A process-global rather than a thread-local: the consumer
/// (`kfsb_multi_prepare_domain`) runs under `par_iter`, so a thread-local
/// published on the main thread would be invisible exactly where it is needed.
/// Keyed by SPEC ROW, then node, then neuron.
///
/// Per-row rather than aggregated, and that is a measured correction rather
/// than a preference. Aggregating the first six binding rows gave "1181 of 1366
/// neurons with positive priority" — the union over rows washes out exactly the
/// concentration the prior exists to exploit (`d50 = 8` *per row*). kFSB already
/// collapses the 99 rows to ONE straggler per domain
/// (`kfsb_multi.rs:4172-4183`), so the selector can and should ask for that
/// row's attribution specifically.
#[derive(Debug)]
struct SparseNodePrior {
    /// Exact source width. An in-range omitted index is a covered zero; an
    /// out-of-range index is incomplete/stale evidence and returns `None`.
    width: usize,
    /// Strictly-positive normalised scores, sorted by neuron index.
    positive: Vec<(usize, f64)>,
}

impl SparseNodePrior {
    fn score(&self, idx: usize) -> Option<f64> {
        if idx >= self.width {
            return None;
        }
        Some(
            self.positive
                .binary_search_by_key(&idx, |(index, _)| *index)
                .ok()
                .map_or(0.0, |position| self.positive[position].1),
        )
    }
}

#[derive(Debug)]
struct RowPrior {
    per_node: HashMap<String, SparseNodePrior>,
    live: usize,
}

impl RowPrior {
    fn score(&self, node: &str, idx: usize) -> Option<f64> {
        self.per_node.get(node)?.score(idx)
    }
}

static ATTR_PRIOR: std::sync::RwLock<Option<std::sync::Arc<HashMap<usize, RowPrior>>>> =
    std::sync::RwLock::new(None);
static ATTR_RUN_SERIAL: Mutex<()> = Mutex::new(());
thread_local! {
    static ATTR_RUN_DEADLINE: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
}

/// Whole-verification lifetime for attribution deadline/state.
///
/// Every armed probe installs the caller's deadline in thread-local scope.
/// Only modes that publish a KFSB prior additionally hold the serial lock
/// across root evaluation and every consumer, clearing on acquisition/drop.
/// A diagnostic-only `NY_GAP_ATTRIBUTION=1` scope never touches that mutex or
/// publication, so it cannot contend with or perturb a scored prior owner.
pub(crate) struct AttributionRunGuard {
    serial: Option<MutexGuard<'static, ()>>,
    previous_deadline: Option<Instant>,
}

impl Drop for AttributionRunGuard {
    fn drop(&mut self) {
        if self.serial.is_some() {
            // The optional guard remains live until after this Drop body, so
            // publication clearing is still serialized with the owner.
            clear_attribution_prior();
        }
        ATTR_RUN_DEADLINE.with(|slot| slot.set(self.previous_deadline));
    }
}

/// Acquire the whole-run owner before its authoritative boundary.
///
/// Prior-producing runs serialize because the publication is process-global.
/// Their contention is charged to the existing BaB budget: acquisition polls
/// briefly and returns `DeadlineExceeded` at equality. Diagnostic-only probes
/// install deadline authority but are deliberately non-failing and mutex-free.
/// Default-off runs return immediately without touching either state.
pub(crate) fn attribution_run_guard(
    deadline: Option<Instant>,
) -> Result<Option<AttributionRunGuard>> {
    let owns_prior = attribution_branching_enabled() || attribution_branch_diag_enabled();
    attribution_run_guard_if_until(root_gap_probe_enabled(), owns_prior, deadline)
}

fn attribution_run_guard_if_until(
    armed: bool,
    owns_prior: bool,
    deadline: Option<Instant>,
) -> Result<Option<AttributionRunGuard>> {
    if !armed {
        return Ok(None);
    }
    if !owns_prior {
        let previous_deadline = ATTR_RUN_DEADLINE.with(|slot| slot.replace(deadline));
        return Ok(Some(AttributionRunGuard {
            serial: None,
            previous_deadline,
        }));
    }
    let limit = deadline.ok_or_else(|| {
        NyError::DeadlineExceeded(
            "#gap-attribution: prior-producing run has no caller/global deadline".to_string(),
        )
    })?;
    let serial = loop {
        match ATTR_RUN_SERIAL.try_lock() {
            Ok(serial) => break serial,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                break poisoned.into_inner();
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                let now = Instant::now();
                if now >= limit {
                    return Err(NyError::DeadlineExceeded(
                        "#gap-attribution: deadline expired waiting for another armed run"
                            .to_string(),
                    ));
                }
                std::thread::sleep(
                    limit
                        .saturating_duration_since(now)
                        .min(std::time::Duration::from_millis(1)),
                );
            }
        }
    };
    clear_attribution_prior();
    let previous_deadline = ATTR_RUN_DEADLINE.with(|slot| slot.replace(deadline));
    let guard = AttributionRunGuard {
        serial: Some(serial),
        previous_deadline,
    };
    if Instant::now() >= limit {
        drop(guard);
        return Err(NyError::DeadlineExceeded(
            "#gap-attribution: acquisition crossed the exact deadline".to_string(),
        ));
    }
    Ok(Some(guard))
}

pub(crate) fn attribution_run_deadline() -> Option<Instant> {
    ATTR_RUN_DEADLINE.with(|slot| slot.get())
}

/// Immutable view of one identity-verified spec row's complete prior.
///
/// Acquiring this once per domain avoids taking the process-global lock from a
/// sort comparator.  More importantly, callers can preflight every candidate
/// against one publication and decline the optional re-ranking atomically if
/// any node/index is absent.  Partial or stale evidence must never silently
/// push an unknown candidate behind a known-zero candidate.
pub(crate) struct AttributionRowPrior {
    publication: std::sync::Arc<HashMap<usize, RowPrior>>,
    row: usize,
}

impl AttributionRowPrior {
    pub(crate) fn score(&self, node: &str, idx: usize) -> Option<f64> {
        self.publication
            .get(&self.row)?
            .score(node, idx)
            .filter(|score| score.is_finite() && *score >= 0.0)
    }
}

pub(crate) fn attribution_branching_enabled() -> bool {
    std::env::var(ATTR_BRANCH_ENV).is_ok_and(|v| v == "1")
}

/// The per-decision ranking-quality diagnostic (`NY_ATTR_BRANCH_DIAG=1`).
///
/// Separate from [`attribution_branching_enabled`] on purpose: the honest
/// measurement needs the prior PUBLISHED while the branching hook is OFF, so
/// the candidate set is still chosen by the incumbent `main_score` and the
/// comparison is not set-selection-biased in the prior's favour.
pub(crate) fn attribution_branch_diag_enabled() -> bool {
    std::env::var("NY_ATTR_BRANCH_DIAG").is_ok_and(|v| v == "1")
}

/// Publish per-row, per-node, per-neuron priorities. Replaces any previous.
fn publish_attribution_prior(prior: HashMap<usize, RowPrior>) {
    if let Ok(mut slot) = ATTR_PRIOR.write() {
        *slot = Some(std::sync::Arc::new(prior));
    }
}

/// Priority for one candidate under one spec row, or `None` when no prior is
/// published, that row was not attributed, or the neuron is not covered.
///
/// `None` must be treated as "no opinion" by callers, never as zero — an
/// unattributed neuron is not a neuron known to be inert. `Some(0.0)` is the
/// genuinely different statement that it carries none of the row's gap.
pub(crate) fn attribution_prior_score(row: usize, node: &str, idx: usize) -> Option<f64> {
    attribution_prior_for_row(row)?.score(node, idx)
}

/// Snapshot a single row from the current publication.  `None` means the
/// producer did not publish complete, identity-verified evidence for this row.
pub(crate) fn attribution_prior_for_row(row: usize) -> Option<AttributionRowPrior> {
    let publication = ATTR_PRIOR.read().ok()?.as_ref()?.clone();
    publication
        .contains_key(&row)
        .then_some(AttributionRowPrior { publication, row })
}

/// Whether a prior exists for this specific spec row.
pub(crate) fn attribution_prior_has_row(row: usize) -> bool {
    ATTR_PRIOR
        .read()
        .ok()
        .and_then(|s| s.as_ref().map(|p| p.contains_key(&row)))
        .unwrap_or(false)
}

pub(crate) fn attribution_prior_published() -> bool {
    ATTR_PRIOR.read().is_ok_and(|s| s.is_some())
}

pub(crate) fn clear_attribution_prior() {
    if let Ok(mut slot) = ATTR_PRIOR.write() {
        *slot = None;
    }
}

/// Build a per-row priority map from attributed rows.
///
/// Each row keeps its OWN normalised profile — deliberately not blended across
/// rows. `attrs[i]` is filed under `rows[i]`, the spec-row index the selector
/// will look it up by. Normalising by the row's own `Σgⱼ` makes priorities
/// comparable across rows without letting a larger-gap row dominate.
fn build_attribution_prior(attrs: &[GapAttribution], rows: &[usize]) -> HashMap<usize, RowPrior> {
    let mut out: HashMap<usize, RowPrior> = HashMap::new();
    for (attr, &row) in attrs.iter().zip(rows) {
        if let Some(prior) = build_row_prior(attr) {
            out.insert(row, prior);
        }
    }
    out
}

fn build_row_prior(attr: &GapAttribution) -> Option<RowPrior> {
    if attr.sum_g < 0.0 || !attr.sum_g.is_finite() {
        return None;
    }
    // A zero-gap row is complete evidence: every covered neuron has exact
    // priority zero, so KFSB should preserve main-score order rather than call
    // the row "missing". Positive rows retain their normalised exact scores.
    let inv = if attr.sum_g > 0.0 {
        1.0 / attr.sum_g
    } else {
        0.0
    };
    let mut live = 0usize;
    let mut per_node = HashMap::with_capacity(attr.per_node.len());
    for (name, ng) in &attr.per_node {
        let positive: Vec<(usize, f64)> = ng
            .g
            .iter()
            .enumerate()
            .filter_map(|(idx, gap)| (gap.is_finite() && *gap > 0.0).then_some((idx, gap * inv)))
            .collect();
        live += positive.len();
        per_node.insert(
            name.clone(),
            SparseNodePrior {
                width: ng.g.len(),
                positive,
            },
        );
    }
    Some(RowPrior { per_node, live })
}

fn select_margin_probe_rows(
    ascent: &crate::bounds::AlphaSpecAscent,
    settled_lower: &ArrayD<f32>,
    settled_upper: &ArrayD<f32>,
    deadline: Option<Instant>,
) -> Result<Vec<usize>> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "#gap-attribution: deadline expired before row selection".to_string(),
        ));
    }
    let lower: Vec<f32> = settled_lower.iter().copied().collect();
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "#gap-attribution: deadline expired while copying settled lower output".to_string(),
        ));
    }
    let upper: Vec<f32> = settled_upper.iter().copied().collect();
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "#gap-attribution: deadline expired while copying settled upper output".to_string(),
        ));
    }
    rank_margin_probe_rows(
        ascent,
        &lower,
        &upper,
        probe_rows(ascent.rows.len()),
        deadline,
    )
}

/// Build the exact shared-alpha checkpoint attributed by this static prior.
///
/// This is intentionally NOT claimed to be the final per-domain BaB alpha:
/// later root assembly may select a different atomic/critical state, discard a
/// warm start, or optimize per-disjunct alpha. The theorem remains exact for
/// this settled root checkpoint and KFSB consumes its scores only as advisory
/// root geometry. Per-spec deltas are removed because a compact selected-row
/// seed has no exact mapping to the raw-output carrier slots. Upper properties
/// orient the checkpoint's upper map into the equivalent lower objective.
fn settled_shared_probe_alpha(
    alpha_state: &GraphAlphaState,
    verify_upper_bound: bool,
) -> Option<GraphAlphaState> {
    (alpha_state.has_spec_deltas() || verify_upper_bound).then(|| {
        let mut shared = alpha_state.clone_for_backward();
        shared.spec_deltas.clear();
        shared.spec_slot_rows.clear();
        if verify_upper_bound {
            for (node, upper) in &alpha_state.alphas_upper {
                shared.alphas.insert(node.clone(), upper.clone());
            }
        }
        shared
    })
}

fn rank_margin_probe_rows(
    ascent: &crate::bounds::AlphaSpecAscent,
    lower: &[f32],
    upper: &[f32],
    row_count: usize,
    deadline: Option<Instant>,
) -> Result<Vec<usize>> {
    if lower.len() != ascent.output_len() || upper.len() != ascent.output_len() {
        return Err(NyError::ShapeMismatch {
            expected: vec![ascent.output_len()],
            got: vec![lower.len().max(upper.len())],
        });
    }

    // The settled raw-output box is already resident.  Its interval projection
    // is only a row-selection heuristic; the selected rows are then recomputed
    // by the exact margin-seeded CROWN fold below.  Refuse the whole selection
    // if any row is malformed rather than silently changing which property
    // rows are eligible for the prior.
    let mut ranked = Vec::with_capacity(ascent.rows.len());
    for (row, objective) in ascent.rows.iter().enumerate() {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "#gap-attribution: deadline expired while ranking spec row {row}"
            )));
        }
        let slack = objective.margin_slack(lower, upper).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "#gap-attribution: cannot rank malformed spec row {row}"
            ))
        })?;
        ranked.push((row, slack));
    }
    ranked.sort_unstable_by(|(row_a, slack_a), (row_b, slack_b)| {
        slack_a.total_cmp(slack_b).then_with(|| row_a.cmp(row_b))
    });
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "#gap-attribution: deadline expired while sorting property rows".to_string(),
        ));
    }
    ranked.truncate(row_count.min(GAP_PROBE_MAX_ROWS));
    Ok(ranked.into_iter().map(|(row, _)| row).collect())
}

/// Construct the compact exact property matrix used as the backward seed.
///
/// The root warmup is identity-seeded
/// (`identity_warmup_seed`), so an attribution taken off it decomposes the gap
/// of a raw logit, not of the property. The relaxation the backward selects
/// depends on the sign of the seed, so a margin row's attribution is NOT
/// recoverable from two logit rows' attributions — the fold has to be seeded
/// with the margin.
///
/// Seeding the original graph directly with this affine matrix is exactly
/// equivalent to appending a Linear head, but allocates only `k * output_dim`
/// coefficients and never clones model layers, weights, or caches.
fn margin_probe_matrix(
    ascent: &crate::bounds::AlphaSpecAscent,
    selected_rows: &[usize],
    output_dim: usize,
    deadline: Option<Instant>,
) -> Result<ndarray::Array2<f32>> {
    use ndarray::Array2;

    let rows = selected_rows.len();
    let width = ascent.output_len();
    if rows == 0 || rows > GAP_PROBE_MAX_ROWS || width == 0 || width != output_dim {
        return Err(NyError::InvalidSpec(format!(
            "#gap-attribution: exact seed needs 1..={GAP_PROBE_MAX_ROWS} rows and \
             output width {output_dim}, got {rows}x{width}"
        )));
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "#gap-attribution: deadline expired before exact seed allocation".to_string(),
        ));
    }
    for (offset, &row) in selected_rows.iter().enumerate() {
        if row >= ascent.rows.len() || selected_rows[..offset].contains(&row) {
            return Err(NyError::InvalidSpec(format!(
                "#gap-attribution: invalid or duplicate selected spec row {row}"
            )));
        }
        if ascent.rows[row]
            .objective
            .iter()
            .any(|coefficient| !coefficient.is_finite())
        {
            return Err(NyError::InvalidSpec(format!(
                "#gap-attribution: non-finite coefficient in selected spec row {row}"
            )));
        }
    }

    // Row r is the selected spec row in lower-bound orientation.
    // An upper-bound property `c.y < t` is exactly `(-c).y > -t`; orienting it
    // here lets the same lower-CROWN attribution theorem rank both modes.
    let mut seed = Array2::zeros((rows, width));
    for (carrier_row, &spec_row) in selected_rows.iter().enumerate() {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "#gap-attribution: deadline expired while filling seed row {carrier_row}"
            )));
        }
        let spec = &ascent.rows[spec_row];
        for column in 0..width {
            seed[[carrier_row, column]] = if spec.verify_upper_bound {
                -spec.objective[column]
            } else {
                spec.objective[column]
            };
        }
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "#gap-attribution: deadline expired after exact seed fill".to_string(),
        ));
    }
    Ok(seed)
}

/// Attribute the root fold's bound gap and report it, once, after the alpha
/// ascent has settled.
///
/// This answers build-plan step 3 — Corollary 4's trichotomy — on the real
/// network rather than on a test fixture. It runs its OWN intermediates fold at
/// the settled alpha rather than reusing the loop's, because the loop's normal
/// path does not capture per-node coefficients; that costs one extra backward
/// pass, which is why it is gated.
///
/// Never fails the run: every error is reported and swallowed. The probe is a
/// diagnostic and must not be able to change a verdict, including by aborting.
///
/// COST NOTE. The intermediates fold necessarily captures a dense
/// `a_at_relu`, but the seed is hard-capped to three selected property rows.
/// On `CIFAR100_resnet_medium` this is at most `3 x 55,460 x 4 B ~= 0.64 MiB`
/// across the ReLU widths, versus ~21 MiB for the old 100-row capture.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_root_gap_probe(
    net: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    exec_order: &[String],
    output_dim: usize,
    input_dim: usize,
    relu_name_to_idx: &HashMap<String, usize>,
    alpha_state: &GraphAlphaState,
    engine: Option<&dyn GemmEngine>,
    global_deadline: Option<Instant>,
    // Settled raw-output box used only to choose the 1--3 worst spec rows. The
    // exact fold below recomputes those rows with their margin coefficients.
    settled_output_lower: &ArrayD<f32>,
    settled_output_upper: &ArrayD<f32>,
    // Required: the fold is re-seeded with these MARGIN rows instead of the
    // warmup's raw-logit identity, which is what makes Corollary 3 fire and the
    // attributed g_j the property's rather than a logit's. Missing metadata is
    // refused before any dense fold rather than falling back to raw logits.
    spec_ascent: Option<&crate::bounds::AlphaSpecAscent>,
) {
    let started = Instant::now();
    let prior_requested = attribution_branching_enabled() || attribution_branch_diag_enabled();
    // A failed or panicked producer later in this owned verification must not
    // leave an earlier root fold's prior visible. An unowned/direct propagate
    // call has no global deadline and must not mutate another verifier's state.
    if prior_requested && global_deadline.is_some() {
        clear_attribution_prior();
    }
    let probe_deadline = compose_probe_deadline(
        Instant::now(),
        probe_budget(),
        global_deadline,
        prior_requested,
    );
    // A diagnostic must not be able to change a verdict, INCLUDING by
    // aborting. The Result path covers typed refusals; catch_unwind covers
    // the rest (an out-of-bounds index in this probe took a whole cifar100
    // run down before this was added). AssertUnwindSafe is justified: the
    // closure only reads shared state and writes to stderr.
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        let deadline = probe_deadline.ok_or_else(|| {
            NyError::DeadlineExceeded(
                "#gap-attribution: scored prior has no caller/global deadline".to_string(),
            )
        })?;
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "#gap-attribution: deadline expired before exact seed construction".to_string(),
            ));
        }

        // Exact attribution requires the property's margin seed because ReLU
        // relaxation selection depends on coefficient sign. A raw-logit
        // fallback would either allocate the full output identity before the
        // three-row cap or rank the wrong objective, so safely refuse it.
        let ascent = spec_ascent.ok_or_else(|| {
            NyError::InvalidSpec(
                "#gap-attribution: exact margin seed unavailable; raw fallback refused".to_string(),
            )
        })?;
        let selected_spec_rows = select_margin_probe_rows(
            ascent,
            settled_output_lower,
            settled_output_upper,
            Some(deadline),
        )?;
        let verify_upper_bound = selected_spec_rows
            .first()
            .and_then(|&row| ascent.rows.get(row))
            .map(|row| row.verify_upper_bound)
            .ok_or_else(|| {
                NyError::InvalidSpec("#gap-attribution: no selected margin rows".to_string())
            })?;
        if selected_spec_rows.iter().any(|&row| {
            ascent
                .rows
                .get(row)
                .is_none_or(|spec| spec.verify_upper_bound != verify_upper_bound)
        }) {
            return Err(NyError::InvalidSpec(
                "#gap-attribution: mixed lower/upper selected rows are unsupported".to_string(),
            ));
        }
        let thresholds: Vec<f32> = selected_spec_rows
            .iter()
            .map(|&row| {
                let spec = &ascent.rows[row];
                if verify_upper_bound {
                    -spec.threshold
                } else {
                    spec.threshold
                }
            })
            .collect();
        let margin_seed =
            margin_probe_matrix(ascent, &selected_spec_rows, output_dim, Some(deadline))?;
        let seed_rows = margin_seed.nrows();
        eprintln!(
            "[gap-attr] seeding: {seed_rows} exact MARGIN row(s) {:?}",
            selected_spec_rows
        );

        // Attribute the settled shared root checkpoint exactly. This is a
        // static advisory prior: later root assembly may transport, replace,
        // or discard alpha without invalidating this theorem-checked geometry.
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "#gap-attribution: deadline expired before shared-alpha snapshot".to_string(),
            ));
        }
        let shared_alpha = settled_shared_probe_alpha(alpha_state, verify_upper_bound);
        let probe_alpha = shared_alpha.as_ref().unwrap_or(alpha_state);

        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "#gap-attribution: deadline expired after shared-alpha snapshot".to_string(),
            ));
        }
        let (bounds, inter) = net.dag_alpha_backward_pass_with_intermediates_and_exact_seed(
            input,
            node_bounds,
            exec_order,
            output_dim,
            input_dim,
            relu_name_to_idx,
            probe_alpha,
            engine,
            deadline,
            &margin_seed,
        )?;
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "#gap-attribution: deadline expired during exact margin fold".to_string(),
            ));
        }

        // Flatten before indexing. The fold's output tensor is NOT reliably
        // 1-D — on `CIFAR100_resnet_medium` it carries the network's output
        // shape — so a `[[row]]` index panics. (It did: that is what this
        // comment is paying for.)
        let lower_flat: Vec<f32> = bounds.lower().iter().copied().collect();
        if lower_flat.len() < seed_rows {
            return Err(NyError::ShapeMismatch {
                expected: vec![seed_rows],
                got: vec![lower_flat.len()],
            });
        }

        let n_seed = inter.final_bounds.lower_a().nrows();
        let n_cols = inter.final_bounds.lower_a().ncols();
        if n_cols != input_dim {
            return Err(NyError::InvalidSpec(format!(
                "#gap-attribution: the fold did not populate intermediates \
                 (final_bounds is {n_seed}x{n_cols}, expected k x {input_dim}). \
                 This is what a degraded/IBP-fallback backward looks like — the \
                 CROWN path that assigns `final_bounds` was never reached. Give \
                 the probe more budget via {GAP_PROBE_BUDGET_ENV}."
            )));
        }
        if n_seed != seed_rows || lower_flat.len() != seed_rows {
            return Err(NyError::ShapeMismatch {
                expected: vec![seed_rows],
                got: vec![n_seed.max(lower_flat.len())],
            });
        }
        eprintln!("[gap-attr] fold seeded {n_seed}/{seed_rows} compact exact margin rows");

        let mut collected_prior: HashMap<usize, RowPrior> = HashMap::new();
        for row in 0..n_seed {
            if Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(format!(
                    "#gap-attribution: deadline expired before row {row}"
                )));
            }
            let spec_row = selected_spec_rows[row];
            // The exact seed row is also the concrete-forward objective. This
            // is affine-head equivalence without constructing or cloning a
            // synthetic graph.
            let objective: Vec<f32> = margin_seed.row(row).iter().copied().collect();

            let attr =
                net.attribute_row_gap_until(input, probe_alpha, &inter, row, &objective, deadline)?;
            if Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(format!(
                    "#gap-attribution: deadline expired while attributing row {row}"
                )));
            }
            let lb_sound = lower_flat[row] as f64;
            let scale = attr
                .f_x_star
                .abs()
                .max(attr.bound_at_x_star.abs())
                .max(attr.sum_g.abs())
                .max(1.0);
            // Theorem 1 is the trust anchor: if it does not hold, say so loudly
            // and do NOT report numbers that would be believed.
            attr.verify_identity(1e-5 * scale)?;
            let prior = build_row_prior(&attr).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "#gap-attribution: row {spec_row} produced malformed gap scores"
                ))
            })?;
            collected_prior.insert(spec_row, prior);
            eprintln!(
                "[gap-attr] seed_row={row} spec_row={spec_row} \
                 identity=ok residual={:.3e} \
                 lb_sound={:.6e} B_x_star={:.6e} f_x_star={:.6e} \
                 sum_g={:.6e} E={:.6e} E_frac={:.3e} \
                 live={} unstable={} d50={:?} d90={:?}",
                attr.residual,
                lb_sound,
                attr.bound_at_x_star,
                attr.f_x_star,
                attr.sum_g,
                attr.certified_error(lb_sound),
                // The number step 3 exists to produce: what fraction of the
                // total gap is certified arithmetic rather than relaxation.
                // Near 1 means branching is futile here (arithmetic-limited).
                attr.certified_error(lb_sound) / (attr.sum_g + attr.certified_error(lb_sound)),
                attr.live_neurons(),
                attr.unstable_neurons(),
                attr.attribution_depth(attr.sum_g * 0.5),
                attr.attribution_depth(attr.sum_g * 0.9),
            );
            // Corollary 4's trichotomy is only meaningful against a real
            // threshold, i.e. on the margin-seeded path.
            let t = thresholds[row];
            eprintln!(
                "[gap-attr] seed_row={row} spec_row={spec_row} threshold={t:.6e} verdict={:?} \
                 attribution_depth_to_verify={:?}",
                attr.classify(lb_sound, f64::from(t)),
                attr.attribution_depth(f64::from(t) - lb_sound),
            );
        }
        if collected_prior.len() != selected_spec_rows.len() {
            return Err(NyError::InvalidSpec(
                "#gap-attribution: incomplete row transaction".to_string(),
            ));
        }

        // Publication is the final operation and occurs only after every exact
        // row, identity check, and diagnostic line completed inside both the
        // private and caller/global deadlines.
        if prior_requested {
            let mut rows: Vec<&usize> = collected_prior.keys().collect();
            rows.sort_unstable();
            let summary = rows
                .iter()
                .map(|r| format!("row{r}:{}", collected_prior[r].live))
                .collect::<Vec<_>>()
                .join(" ");
            if Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(
                    "#gap-attribution: deadline expired before transactional publication"
                        .to_string(),
                ));
            }
            eprintln!(
                "[gap-attr] publishing PER-ROW branching prior; live neurons per row: {summary}"
            );
            publish_attribution_prior(collected_prior);
            if Instant::now() >= deadline {
                // KFSB cannot run until this synchronous root fold returns, so
                // clearing here makes a lock/write that crossed equality fully
                // transactional from every consumer's perspective.
                clear_attribution_prior();
                return Err(NyError::DeadlineExceeded(
                    "#gap-attribution: publication crossed the exact deadline".to_string(),
                ));
            }
        }
        Ok(())
    }));
    match caught {
        Ok(Ok(())) => {}
        Ok(Err(e)) => eprintln!("[gap-attr] probe refused (run unaffected): {e}"),
        Err(_) => eprintln!("[gap-attr] probe PANICKED and was contained (run unaffected)"),
    }
    eprintln!(
        "[gap-attr] probe took {:.3}s",
        started.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests;
