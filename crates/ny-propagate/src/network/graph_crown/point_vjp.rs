// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fast f32 point back-propagation (reverse-mode autodiff at a single concrete
//! input point) for the PGD attack.
//!
//! # What this computes
//!
//! [`GraphNetwork::attack_point_gradient`] returns
//! `d(spec_row · network_output) / d(input)` evaluated **at** the concrete point
//! `x`, reshaped to `x.shape()`. This is an ATTACK gradient: it is the exact
//! local point-Jacobian, and it does **not** need certified soundness (any
//! counterexample the attack finds is re-checked concretely elsewhere).
//!
//! # Why the CROWN backward machinery gives the exact point-Jacobian here
//!
//! At a concrete point the input "box" is degenerate (`[x, x]`), so every node's
//! forward interval is degenerate too. In particular every ReLU's pre-activation
//! interval is `[v, v]`, so the CROWN ReLU relaxation collapses to the EXACT mask
//! (slope `1` if `v > 0`, `0` if `v < 0`, zero intercept) — no crossing /
//! relaxation case ever fires. Consequently a single-row cotangent (the
//! `spec_row`) propagated backward through the network with `err = None` stays
//! exact (`lower_a == upper_a` up to floating-point roundoff), and the row that
//! lands on the network input IS the exact gradient.
//!
//! # Implementation
//!
//! This mirrors the reverse loop in [`super::propagation`]
//! (`crown_backward_with_relaxation_and_deadline_and_truncation`) but:
//! - seeds the OUTPUT node with the `spec_row` (a `1 × num_outputs`
//!   [`LinearBounds`] with `lower_a == upper_a == spec_row`, zero bias,
//!   `*_err = None`) instead of `identity(output_dim)`, so we track a single row;
//! - stays entirely in Dense mode (no Patches);
//! - reuses the exact same per-node backward dispatch
//!   ([`dispatch_backward_layer`] for the linear/graph ops,
//!   [`dispatch_relu_backward`] for ReLU) and the exact same accumulation
//!   frontier ([`CrownMergeAccumulator`] +
//!   [`apply_dense_backward_dispatch_result`]) so residual fan-in summation and
//!   the bias-to-input routing are byte-for-byte the same as the certified path.
//!
//! # Correctness-first note (PERF markers below)
//!
//! Both the Linear backward (`aw_f64_with_abssum`, invoked inside
//! `dispatch_backward_layer`'s `Layer::Linear` arm) and the sound conv backward
//! (`Conv2d` arm) compute in f64 for soundness. For an ATTACK gradient that is
//! unnecessary — see the `// PERF:` markers for exactly where a plain-f32 GEMM
//! (`crate::fast_f32_gemm::with_engine`) and a direct `conv2d_transpose` could be
//! swapped in once this passes its gradient-check. Correctness against the exact
//! oracle comes first; the f32 swap is a follow-up.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use std::cell::Cell;
use std::time::Instant;

use crate::bounds::patches::CrownBounds;
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::network::core::{apply_dense_backward_dispatch_result, GraphNetwork, NETWORK_INPUT};
use crate::network::CrownMergeAccumulator;
use crate::MulBinaryRelaxationMode;

use super::backward_node_dispatch::{dispatch_relu_backward, NodeDispatchResult};

// ---------------------------------------------------------------------------
// ATTACK-only soft-sign surrogate sharpness (β) — thread-local ramp control
// ---------------------------------------------------------------------------

/// Default soft-sign surrogate sharpness β for the ATTACK point-gradient's
/// [`Layer::Sign`] arm. Historically a hard-coded constant; kept as the default
/// so every caller that does NOT opt into the ramp behaves exactly as before
/// (byte-identical). Matches `pgd_attack`'s `SMOOTH_SIGN_BETA`.
pub const DEFAULT_ATTACK_SIGN_BETA: f32 = 10.0;

/// Minimum surrogate β. The surrogate slope stays in `(0, β]`, so β must remain
/// a sane positive sharpness; [`set_attack_sign_beta`] clamps into `[MIN, MAX]`.
const ATTACK_SIGN_BETA_MIN: f32 = 2.0;
/// Maximum surrogate β (see [`ATTACK_SIGN_BETA_MIN`]).
const ATTACK_SIGN_BETA_MAX: f32 = 20.0;

thread_local! {
    // Thread-local so an attack loop can VARY β without threading a new
    // parameter through `attack_point_gradient`'s signature (which would force a
    // change to the certified-path caller `graph_pgd_exact.rs`). ATTACK
    // DIRECTION ONLY — read solely by the non-certified `Layer::Sign` surrogate.
    static ATTACK_SIGN_BETA: Cell<f32> = const { Cell::new(DEFAULT_ATTACK_SIGN_BETA) };
}

/// Current thread-local soft-sign β read by
/// [`GraphNetwork::attack_point_gradient`]'s `Layer::Sign` surrogate. Defaults
/// to [`DEFAULT_ATTACK_SIGN_BETA`].
pub fn attack_sign_beta() -> f32 {
    ATTACK_SIGN_BETA.with(|b| b.get())
}

/// Set the thread-local soft-sign β (clamped to `[2, 20]`).
///
/// ATTACK DIRECTION ONLY: β scales the *non-certified* Sign surrogate slope, so
/// it can only make the search sharper/smoother — it can NEVER change a verdict.
/// Every candidate the attack proposes is still concretely re-checked by the
/// unchanged trusted-oracle gate.
pub fn set_attack_sign_beta(beta: f32) {
    let clamped = beta.clamp(ATTACK_SIGN_BETA_MIN, ATTACK_SIGN_BETA_MAX);
    ATTACK_SIGN_BETA.with(|b| b.set(clamped));
}

/// RAII guard that restores the thread-local β to its prior value on drop, so a
/// ramp installed inside an attack loop cannot leak into later work on the same
/// thread. Construct with the value to install; the previous value is captured
/// and reinstated when the guard is dropped.
pub struct AttackSignBetaGuard {
    prev: f32,
}

impl AttackSignBetaGuard {
    /// Install `beta` (clamped) and remember the previous thread-local value.
    pub fn new(beta: f32) -> Self {
        let prev = attack_sign_beta();
        set_attack_sign_beta(beta);
        Self { prev }
    }
}

impl Drop for AttackSignBetaGuard {
    fn drop(&mut self) {
        // Restore the captured value verbatim (it was already valid/clamped).
        ATTACK_SIGN_BETA.with(|b| b.set(self.prev));
    }
}

impl GraphNetwork {
    /// Whitelist gate: the point back-prop only runs when EVERY node is in the
    /// exact-gradient fragment `{Conv2d, Linear, ReLU, Add, AveragePool, Flatten,
    /// Reshape}` plus the affine constant-arithmetic ops below. Mirrors
    /// `layer_supports_exact_gradient` in `ny-cli graph_pgd_exact.rs`. Any other
    /// layer → the caller gets `Ok(None)` and falls back to its slower gradient
    /// (certified CROWN / SPSA).
    fn point_vjp_supported_fragment(&self) -> bool {
        self.node_names().iter().all(|name| {
            self.node(name).is_some_and(|node| {
                matches!(
                    node.layer(),
                    Layer::Conv2d(_)
                        | Layer::Linear(_)
                        | Layer::ReLU(_)
                        | Layer::Add(_)
                        | Layer::AveragePool(_)
                        // MaxPool2d: EXACT at a point. The degenerate box [x, x]
                        // makes every pooling window have a definite winner (the
                        // argmax input, l == u), so dispatch_backward_layer's
                        // generic unary arm routes propagate_crown_backward →
                        // MaxPool2dLayer::propagate_linear_with_bounds, which
                        // routes the gradient through that winner (exact
                        // route-to-max — see max.rs "If one input definitely
                        // dominates … route gradient through it"). Needed for the
                        // deeper traffic_signs BNNs (net-2/net-3) which stack
                        // MaxPool + BatchNorm between the Sign conv blocks.
                        | Layer::MaxPool2d(_)
                        | Layer::Flatten(_)
                        | Layer::Reshape(_)
                        // Shape/linear plumbing exact at a point: Transpose is a
                        // pure permutation and MatMul(live, const-weight) is affine
                        // (the dense head of the traffic_signs BNNs).
                        // dispatch_backward_layer routes both through the exact
                        // linear transpose.
                        | Layer::Transpose(_)
                        | Layer::MatMul(_)
                        // GAN/deconv fragment (cgan): ConvTranspose + BatchNorm.
                        // dispatch_backward_layer handles all three; attack-only.
                        | Layer::ConvTranspose1d(_)
                        | Layer::ConvTranspose2d(_)
                        | Layer::BatchNorm(_)
                        // Affine constant arithmetic (cora_2024 MLP fragment:
                        // unfused Gemm = MatMul + AddConstant bias, and the
                        // mnist Div-by-constant normalization; d/dx (x/c) = 1/c).
                        // All exact at a point — dispatch_backward_layer routes
                        // them through the plain unary CROWN backward, which is
                        // the exact affine transpose for these layers.
                        | Layer::AddConstant(_)
                        | Layer::SubConstant(_)
                        | Layer::MulConstant(_)
                        | Layer::DivConstant(_)
                        // Sign / binarized nets (traffic_signs_recognition BNNs)
                        // and the trailing Softmax classifier head.
                        // ATTACK-ONLY, NON-EXACT: unlike every other layer above
                        // (each exact at a point), Sign's true point-Jacobian is
                        // 0 a.e. and Softmax's certified relaxation carries a
                        // vanishing gradient. The backward loop special-cases BOTH
                        // — Sign with a soft-sign surrogate slope, Softmax as a
                        // monotone identity pass-through (the logit-margin
                        // direction, matching the external bnn_falsifier prototype)
                        // — so PGD can descend. Each direction is a heuristic, never
                        // a certified bound: every candidate it yields is still
                        // concretely re-checked by the unchanged trusted-oracle
                        // gate. See the Sign / Softmax arms in attack_point_gradient.
                        | Layer::Sign(_)
                        | Layer::Softmax(_)
                )
            })
        })
    }

    /// Fast f32 point back-prop: `d(spec_row · network_output) / d(input)` at the
    /// concrete point `x`, reshaped to `x.shape()`.
    ///
    /// - `spec_row` must have shape `(1, num_outputs)` where `num_outputs` is the
    ///   flattened element count of the network output node.
    /// - `engine` is an optional GEMM engine (e.g. GPU); `None` uses CPU.
    /// - `deadline` aborts the pass (returning `Ok(None)`) if exceeded.
    ///
    /// Returns:
    /// - `Ok(Some(grad))` with `grad.shape() == x.shape()` on success,
    /// - `Ok(None)` when the graph is outside the supported fragment, the graph
    ///   is empty, a layer reports `Unsupported`, or the deadline is hit,
    /// - `Err(..)` only on an internal/structural failure.
    ///
    /// This is an ATTACK gradient — it is the exact point-Jacobian but carries no
    /// certified error interval (see the module docs for why it is exact at a
    /// point, and why that is sufficient for an attack).
    pub fn attack_point_gradient(
        &self,
        x: &ArrayD<f32>,
        spec_row: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> anyhow::Result<Option<ArrayD<f32>>> {
        // --- Gate: supported fragment + non-empty graph. -------------------
        if self.nodes.is_empty() {
            return Ok(None);
        }
        if !self.point_vjp_supported_fragment() {
            return Ok(None);
        }

        // --- Degenerate input box [x, x]: this is what makes every ReLU mask
        //     exact (see module docs). -----------------------------------------
        let input_bounds = BoundedTensor::concrete(x.clone())?;
        let input_dim = input_bounds.len();

        // --- Node values / pre-activations. At a point every entry is
        //     degenerate (lower == upper == node value); this is BOTH the ReLU
        //     pre-activations AND the concrete node values. --------------------
        // PERF: this reuses the certified (Higham-sound, f64-widened) forward IBP
        // collection for correctness. A plain-f32 forward evaluation at the point
        // would be faster and is a valid later optimization; the sound forward is
        // used here only so the pre-activations/masks match the certified path.
        let node_bounds =
            self.collect_node_bounds_with_engine_and_deadline(&input_bounds, engine, deadline)?;

        // --- Output node + its flattened dimension. -------------------------
        let output_node_name = self.output_name().to_string();
        let output_bounds = node_bounds.get(&output_node_name).ok_or_else(|| {
            anyhow::anyhow!("attack_point_gradient: output node '{output_node_name}' not found")
        })?;
        let output_dim = output_bounds.len();

        // Spec row must be a single row selecting a linear combination of outputs.
        if spec_row.nrows() != 1 || spec_row.ncols() != output_dim {
            return Err(anyhow::anyhow!(
                "attack_point_gradient: spec_row must be (1, {output_dim}), got {:?}",
                spec_row.shape()
            ));
        }

        // We track exactly ONE cotangent row (the spec row) through the whole
        // backward pass, so the "output dimension" for every accumulation helper
        // (used only to size the zero bias-coefficient matrices routed to the
        // network input) is 1, NOT the network output dimension.
        const SPEC_ROWS: usize = 1;

        // --- Accumulator frontier (indexed, Dense-only). --------------------
        // `new_indexed` keys every exec-order node name + NETWORK_INPUT; the
        // string-keyed API (insert / take / merge via accumulate_*) uses the
        // O(1) indexed storage underneath.
        let exec_order: Vec<String> = self.exec_order()?.to_vec();
        let mut acc = CrownMergeAccumulator::new_indexed(&exec_order);

        // Seed the OUTPUT node with the spec row: lower_a == upper_a == spec_row,
        // zero biases, err = None (default). err = None means the coeff-error
        // carrier second pass inside `dispatch_backward_layer` is skipped.
        let seed = LinearBounds::new(
            spec_row.clone(),
            Array1::zeros(SPEC_ROWS),
            spec_row.clone(),
            Array1::zeros(SPEC_ROWS),
        )?;
        acc.insert(output_node_name.clone(), CrownBounds::Dense(seed));

        let mut input_accumulated = false;

        // --- Reverse (reverse-topological) walk. Every consumer of a node is
        //     processed before the node itself, so all cotangent contributions
        //     to a node have already landed when we take it. This is what makes
        //     residual Add fan-in summation correct (see the note below). ------
        for node_name in exec_order.iter().rev() {
            let node_name: &str = node_name.as_str();
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }

            // Move out this node's accumulated cotangent. `None` = the node has
            // no consumers on the spec's cone (its gradient is zero) — skip it.
            let node_cb = match acc.take(node_name)? {
                Some(cb) => cb,
                None => continue,
            };

            let node = self.nodes.get(node_name).ok_or_else(|| {
                anyhow::anyhow!("attack_point_gradient: node '{node_name}' not found")
            })?;

            // Dense throughout: convert the (possibly f64-merged) carrier down to
            // a single f32 LinearBounds. We then DROP any accumulated coefficient
            // error: this is an attack gradient, so the small f64-merge roundoff
            // carrier is irrelevant and dropping it keeps `dispatch_backward_layer`
            // on its plain (single-pass) path.
            let mut node_lb = node_cb.into_dense()?;
            node_lb.lower_a_err = None;
            node_lb.upper_a_err = None;

            // First input drives the shared pre-activation lookup (matches
            // propagation.rs). For network-input-fed nodes, the pre-activation is
            // the concrete input box.
            let first_input = node
                .inputs
                .first()
                .map(String::as_str)
                .unwrap_or(NETWORK_INPUT);
            let pre_activation: &BoundedTensor = if first_input == NETWORK_INPUT {
                &input_bounds
            } else {
                node_bounds.get(first_input).ok_or_else(|| {
                    anyhow::anyhow!(
                        "attack_point_gradient: pre-activation for '{first_input}' not found"
                    )
                })?
            };

            // === ReLU: site-specific (handled by dispatch_relu_backward, NOT by
            //     dispatch_backward_layer). At a point the pre-activation box is
            //     degenerate, so `propagate_crown_backward` produces the EXACT
            //     0/1 mask with zero intercept. ===
            if matches!(&node.layer, Layer::ReLU(_)) {
                match dispatch_relu_backward(
                    self.cut_fold_scope(),
                    node,
                    &node_lb,
                    pre_activation,
                    node_name,
                    "attack_point_gradient",
                    None, // alpha_lower: heuristic (exact-at-point) mask
                    None, // alpha_upper
                )? {
                    NodeDispatchResult::SingleDense(bounds) => {
                        self.accumulate_dense_bounds_to_input(
                            first_input,
                            *bounds,
                            &mut acc,
                            SPEC_ROWS,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                    }
                    // A ReLU that cannot dispatch exactly at the point (should not
                    // happen for the whitelist) → bail to the caller's fallback.
                    NodeDispatchResult::IbpFallback(_) => return Ok(None),
                }
                continue;
            }

            // === Sign: ATTACK-ONLY soft-sign surrogate (NON-EXACT). ===
            // Unlike ReLU and every other whitelist layer — which are EXACT at a
            // point — Sign's true point-Jacobian is 0 almost everywhere (the
            // certified relaxation `propagate_crown_backward` gives slope 0), so
            // it carries NO gradient signal and PGD cannot descend on a BNN. To
            // crack binarized nets we instead apply the smooth surrogate
            // `tanh(β·z)`'s local diagonal Jacobian at the pre-activation point z:
            // `slope_j = β·(1 − tanh²(β·z_j))` with β = the thread-local
            // `attack_sign_beta()` (default DEFAULT_ATTACK_SIGN_BETA = 10.0).
            // Sign is element-wise, so input-dim == output-dim and this is a pure
            // diagonal column-scale of the incoming cotangent — the same
            // SingleDense accumulation as the ReLU arm.
            //
            // SOUNDNESS: this is a surrogate ATTACK direction, NOT a certified
            // bound. Every candidate the attack proposes is still validated by
            // the UNCHANGED trusted-oracle gate (real ORT + sound f64 + zero-tol
            // in-box), so a wrong/surrogate direction can only make the attack
            // weaker or stronger — never a wrong verdict. Do NOT route Sign
            // through `propagate_crown_backward` (that is the certified slope-0
            // relaxation, not this attack surrogate).
            if matches!(&node.layer, Layer::Sign(_)) {
                // β = thread-local soft-sign sharpness (default
                // DEFAULT_ATTACK_SIGN_BETA = 10.0, matching pgd_attack's
                // SMOOTH_SIGN_BETA). An attack loop MAY ramp this across
                // restarts/steps (smooth/exploratory ≈2 → sharp/decisive ≈20) to
                // crack tight boxes a fixed β gets stuck on; when unset it is the
                // historical constant, so every non-ramping caller is
                // byte-identical. ATTACK-ONLY, soundness-neutral (see below).
                let beta = attack_sign_beta();
                // Pre-activation point values z_j (degenerate box: lower == z).
                let z: Vec<f32> = pre_activation.flatten().lower().iter().copied().collect();
                // Element-wise: the incoming cotangent's columns index the Sign
                // outputs, which are 1:1 with its inputs, so z must have exactly
                // one entry per coefficient column.
                if z.len() != node_lb.num_inputs() {
                    return Ok(None);
                }
                // ATTACK-ONLY depth normalisation (net-2/net-3): the deeper BNNs
                // stack 3–4 Sign layers, so |z| GROWS with depth. A raw Sign layer
                // that operates on a large-magnitude pre-activation (measured here:
                // rms up to ~370 on net-1's dense head) has β·z ≫ 1 for almost
                // every unit, so tanh(β·z) saturates and the surrogate slope
                // β·(1 − tanh²(β·z)) underflows to ~0 — the gradient dies before it
                // reaches the input. Normalise the tanh ARGUMENT ONLY by the
                // layer's RMS `rms = sqrt(mean_j(z_j²))`, FLOORED AT 1.0:
                //   slope_j = β·(1 − tanh²(β·z_j / max(rms, 1))).
                //
                // WHY THE FLOOR IS 1.0 (not 1e-6): the floor makes the
                // normalisation one-sided — it only ever DE-saturates large-|z|
                // layers (rms > 1, argument shrunk) and NEVER sharpens small-|z|
                // layers (rms < 1 clamped to 1). Dividing a small-|z| layer's
                // argument by rms < 1 would OVER-saturate it, killing the few
                // near-boundary units that carry the whole attack signal — a
                // 1e-6 floor measurably REGRESSES net-1 (and net-2) to a timeout.
                // With the 1.0 floor the fix is strictly additive: it leaves the
                // shallow-net behaviour intact AND revives the deep-net gradient,
                // and in practice STRENGTHENS net-1 (cracks idx_7573/11379/12375
                // from the witness seed in 1–2 steps, where the un-normalised
                // slope needs 400+ steps or times out).
                //
                // CRITICAL: keep the LEADING β — do NOT fold 1/rms into it. The
                // slope magnitude must stay in (0, β]; only the tanh argument is
                // normalised. Using (β/rms)·(…) shrinks the slope and regresses
                // net-1 — ny's attack is sensitive to gradient magnitude. This is
                // a surrogate ascent DIRECTION only; every candidate is still
                // concretely re-checked by the unchanged trusted-oracle gate.
                let rms = {
                    let mean_sq =
                        z.iter().map(|&zj| (zj as f64) * (zj as f64)).sum::<f64>() / z.len() as f64;
                    (mean_sq.sqrt() as f32).max(1.0)
                };
                let slopes: Vec<f32> = z
                    .iter()
                    .map(|&zj| {
                        let t = (beta * zj / rms).tanh();
                        beta * (1.0 - t * t) // β·sech²(β·z/rms) ∈ (0, β]
                    })
                    .collect();
                // Scale coefficient column j by the surrogate slope_j (diagonal
                // Jacobian). Biases carry through unchanged: they route to the
                // network input as a constant channel and do not affect the
                // extracted gradient row. Error carriers are already None here.
                let mut surrogate = node_lb.clone();
                surrogate
                    .lower_a_mut()
                    .indexed_iter_mut()
                    .for_each(|((_, j), v)| *v *= slopes[j]);
                surrogate
                    .upper_a_mut()
                    .indexed_iter_mut()
                    .for_each(|((_, j), v)| *v *= slopes[j]);
                self.accumulate_dense_bounds_to_input(
                    first_input,
                    surrogate,
                    &mut acc,
                    SPEC_ROWS,
                    input_dim,
                    &mut input_accumulated,
                )?;
                continue;
            }

            // === Softmax: ATTACK-ONLY monotone identity pass-through. ===
            // The traffic_signs BNNs end in Softmax, but the property is an
            // argmax/margin over the outputs and softmax is strictly monotone, so
            // the logit-space margin gradient is a valid (and non-vanishing) ascent
            // direction. Routing softmax's certified relaxation here would instead
            // hand back a saturated/near-zero cotangent. We therefore skip the
            // nonlinearity for the ATTACK and pass the incoming cotangent straight
            // to the logits input — exactly what the external bnn_falsifier
            // prototype does (it seeds the backward on the logits and ignores the
            // softmax). Softmax is element-wise-shaped (in-dim == out-dim), so the
            // cotangent flows to the input unchanged. Attack-only: never a bound;
            // every candidate is still concretely re-checked by the trusted gate.
            if matches!(&node.layer, Layer::Softmax(_)) {
                self.accumulate_dense_bounds_to_input(
                    first_input,
                    node_lb,
                    &mut acc,
                    SPEC_ROWS,
                    input_dim,
                    &mut input_accumulated,
                )?;
                continue;
            }

            // === All other whitelist layers: shared canonical dispatch. ===
            // PERF: for `Layer::Linear` this enters `aw_f64_with_abssum` (f64 A·W
            // with abssum error) and for `Layer::Conv2d` the sound conv-transpose
            // backward — both inside `dispatch_backward_layer`. For an ATTACK
            // gradient these could be swapped for `fast_f32_gemm::with_engine(..)`
            // (Linear) and a direct `conv2d_transpose` (Conv2d) to get the f32
            // speedup once this pass is validated against the exact oracle. The
            // dispatch result shape (Single / Binary / PassThrough) is unchanged.
            let ctx = DispatchContext {
                node_name,
                layer: &node.layer,
                inputs: &node.inputs,
                pre_activation,
                network_input: &input_bounds,
                node_bounds: (&node_bounds).into(),
                engine,
                deadline,
                bilinear_alphas: None,
                mul_binary_relaxation: MulBinaryRelaxationMode::default(),
                mul_binary_alphas: None,
                norm_inv_rms_override: None,
            };

            let result = dispatch_backward_layer(&ctx, &node_lb)?;
            if let BackwardDispatchResult::Unsupported(_reason) = &result {
                // Outside the exact-linear fragment at dispatch time → fall back.
                return Ok(None);
            }

            // Distribute the result to the node's input(s). For a binary `Add`
            // (residual), `bounds_a` is routed to inputs[0] and `bounds_b` to
            // inputs[1]; the separate bias channel is folded onto the network
            // input. When two distinct paths reach the SAME node (the classic
            // residual, where one tensor feeds both a conv and the skip Add), the
            // SECOND `accumulate_dense_bounds_to_input` on that node name is a
            // MERGE: `CrownMergeAccumulator::merge_dense` SUMS the two cotangent
            // matrices (in f64) instead of overwriting. That summation is the
            // reverse-mode "fan-in adds" rule, and it is why reverse-topological
            // order (all consumers before the node) is required.
            apply_dense_backward_dispatch_result(
                self,
                node,
                first_input,
                &node_lb,
                result,
                &mut acc,
                SPEC_ROWS,
                input_dim,
                &mut input_accumulated,
                "attack_point_gradient",
            )?;
        }

        // --- Extract the gradient row at the network input. -----------------
        let final_cb = match acc.take(NETWORK_INPUT)? {
            Some(cb) => cb,
            None => return Ok(None), // no path reached the input (zero gradient region)
        };
        let final_lb = final_cb.into_dense()?;
        if final_lb.num_outputs() != SPEC_ROWS || final_lb.num_inputs() != input_dim {
            return Ok(None);
        }

        // At a point lower_a == upper_a in exact arithmetic; the only asymmetry is
        // the ±1 ULP directed-rounding gap introduced by the f64 merge downcast
        // (next_down for lower, next_up for upper). Averaging recovers the
        // symmetric gradient — mirrors graph_pgd_exact.rs:159.
        let grad_row = (final_lb.lower_a() + final_lb.upper_a()) * 0.5;
        let grad_flat: Vec<f32> = grad_row.row(0).to_vec();
        if grad_flat.len() != x.len() {
            return Ok(None);
        }
        let grad = ArrayD::from_shape_vec(IxDyn(x.shape()), grad_flat)?;
        Ok(Some(grad))
    }
}

#[cfg(test)]
#[path = "point_vjp_tests.rs"]
mod tests;
