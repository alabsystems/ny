// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState};
use crate::network::core::{GraphNetwork, NETWORK_INPUT};

use ndarray::{Array1, Array2};
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

/// #envelope-grad gate (dark, default OFF ⇒ byte-identical): replace the
/// local rule's `pre_lower[i]` factor with a point-forward value at the current
/// binding row's concretization argmin.
///
/// The shipped local rule is `Σ_j [A[j,i]]_+ · l_i`. Every factor is
/// sign-definite — `l_i < 0` holds by the unstable guard and only `A[j,i] > 0`
/// terms are admitted — so the "gradient" is `≤ 0` for every neuron, every
/// objective, and every iteration. Adam can then only ever DECREASE α, monotonically
/// to the `0` clamp; it carries no direction, so no step size can help. That is
/// machine-checked in `crates/ny-cert/proofs/lean/NyProof/AlphaGradientDefect.lean`
/// (`local_rule_nonpos`, `clamped_step_nonincreasing`) and corroborated by the
/// measured lr sweep (0.25 / 0.05 / 0.01 leave the bound BIT-IDENTICAL).
///
/// The derivative target is `Σ_j [A[j,i]]_+ · ĥ_i(x*)`, where `ĥ_i(x*)`
/// is the RELAXED-linear forward value of neuron `i`'s pre-activation at the
/// row argmin. The implementation below deliberately uses the CONCRETE
/// point-forward value instead: it is exact for the first ReLU layer but is a
/// heuristic surrogate deeper in the graph, where exact and relaxed upstream
/// ReLUs can differ. It is not sign-definite; the finite-difference fixtures in
/// `backward/true_grad_oracle_tests.rs` show better aggregate error and sign
/// recovery than the local rule, not an exact or uniformly better gradient.
/// The gate remains dark, and no verdict conversion has been measured.
pub(crate) fn envelope_grad_enabled() -> bool {
    envelope_grad_enabled_with(None)
}

/// [`envelope_grad_enabled`] with the typed preset answer layered in.
///
/// DELIVERY, and the reason this function exists: `vnncomp_scripts/run_instance.sh`
/// exports exactly ONE `NY_*` variable, so an env-only lever cannot fire in
/// competition however well it measures (`ny-cli/tests/measured_gate_delivery.rs`
/// guards that). The preset key `bab.alpha_crown.alpha_envelope_grad` is the
/// path that actually reaches a scored run.
///
/// Layering is `read_over_config`'s: an env value wins in BOTH directions so an
/// A/B can still force the rule on or off over a preset that sets it; config
/// second; the declaration default (`false`) last. A malformed env token is a
/// REJECTION, not an arming — `"true"`, `" 1"`, `"01"` all leave the rule dark,
/// which is deliberate and pinned by the lever's own tests.
///
/// Fails closed to the shipped local rule if resolution errors.
pub(crate) fn envelope_grad_enabled_with(config: Option<bool>) -> bool {
    ny_levers::read_over_config(
        &ny_levers::decls::root_alpha::ALPHA_ENVELOPE_GRAD,
        config.map(ny_levers::LeverValue::Bool),
    )
    .map(|resolved| resolved.value.as_bool())
    .unwrap_or(false)
}

/// The dark `x*` envelope diagnostic, shared by this CPU path and the DAG path
/// in `propagate_dag/gradients`. Both must arm together or a diff of the two
/// `[xstar-*]` lines — the whole point of the probe — is impossible.
pub(crate) fn envelope_xstar_probe_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::dark_probes::ENVELOPE_XSTAR_PROBE)
        .value
        .as_bool()
}

/// The dark envelope-RESCALE diagnostic, likewise shared with the DAG path.
pub(crate) fn envelope_rescale_probe_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::dark_probes::ENVELOPE_RESCALE_PROBE)
        .value
        .as_bool()
}

impl GraphNetwork {
    /// Compute chain-rule gradients for GraphNetwork DAG α-CROWN.
    ///
    /// For each unstable neuron i in ReLU node k:
    /// ∂(output_lower_sum)/∂α_k[i] = Σ_j A_to_relu[j,i] × input_contribution[i]
    ///
    /// Where:
    /// - A_to_relu[j,i] is the coefficient from output j to neuron i (before ReLU k)
    /// - input_contribution captures how the neuron value affects downstream computation
    ///
    /// This chains gradients through all DOWNSTREAM layers in the DAG (the A
    /// matrix at the ReLU is exact), but it is still the LOCAL approximation
    /// UPSTREAM: it substitutes `pre_lower[i]` for the relaxed-linear factor at
    /// the final row's concretization argmin x*. The finite-difference oracle
    /// (`backward/true_grad_oracle_tests.rs`) shows that this local rule can have
    /// the WRONG SIGN (it degraded the post-split wide-α ascent in both lr signs
    /// — #cifar100 task 11). It remains a useful warmup heuristic at the root,
    /// where it empirically converges; gradients are non-soundness-critical.
    #[allow(dead_code)] // Legacy face is exercised by direct gradient oracle tests.
    pub(in crate::network::graph_alpha) fn compute_graph_chain_rule_gradients(
        &self,
        input: &BoundedTensor,
        relu_nodes: &[String],
        intermediate: &GraphAlphaCrownIntermediate,
    ) -> Vec<Array1<f32>> {
        self.compute_graph_chain_rule_gradients_with_binding(
            input,
            relu_nodes,
            intermediate,
            None,
            None,
        )
    }

    /// [`Self::compute_graph_chain_rule_gradients`] with the inputs the
    /// #envelope-grad rule needs. `alpha_state`/`engine` are `None` on the legacy
    /// face, which reproduces the local rule byte-for-byte.
    pub(in crate::network::graph_alpha) fn compute_graph_chain_rule_gradients_with_binding(
        &self,
        input: &BoundedTensor,
        relu_nodes: &[String],
        intermediate: &GraphAlphaCrownIntermediate,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn GemmEngine>,
    ) -> Vec<Array1<f32>> {
        self.chain_rule_gradients_inner(
            input,
            relu_nodes,
            intermediate,
            alpha_state,
            engine,
            // The legacy face has no AlphaCrownConfig in scope; env-only here.
            // The DAG lane (which is what cifar100 takes) threads the preset
            // answer at its own call sites.
            envelope_grad_enabled(),
        )
    }

    /// Force the #envelope-grad rule without touching process-global env (cargo
    /// tests run concurrently, so `set_var` on the gate would race).
    #[cfg(test)]
    pub(in crate::network::graph_alpha) fn compute_graph_chain_rule_gradients_envelope_for_test(
        &self,
        input: &BoundedTensor,
        relu_nodes: &[String],
        intermediate: &GraphAlphaCrownIntermediate,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Vec<Array1<f32>> {
        self.chain_rule_gradients_inner(
            input,
            relu_nodes,
            intermediate,
            Some(alpha_state),
            engine,
            true,
        )
    }

    fn chain_rule_gradients_inner(
        &self,
        input: &BoundedTensor,
        relu_nodes: &[String],
        intermediate: &GraphAlphaCrownIntermediate,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn GemmEngine>,
        use_envelope: bool,
    ) -> Vec<Array1<f32>> {
        // #envelope-grad: a concrete point-forward surrogate for `ĥ_i(x*)`
        // per ReLU node, or `None` ⇒ the local rule's `pre_lower[i]`. Computed
        // ONCE for the whole layer sweep.
        let binding: Option<std::collections::HashMap<String, Array1<f32>>> =
            match (use_envelope, alpha_state) {
                (true, Some(state)) => {
                    self.envelope_binding_points(input, relu_nodes, intermediate, state, engine)
                }
                _ => None,
            };
        // NEVER SILENT. This repo has repeatedly lost sessions to a gate that was
        // armed in the environment but inert in the code; an envelope path that
        // fails closed and looks identical to the local rule is exactly that
        // failure mode. Say which rule actually ran.
        if std::env::var("NY_ALPHA_GRAD_PROBE").ok().as_deref() == Some("1") {
            match (use_envelope, &binding) {
                (false, _) => eprintln!("[envelope-grad] OFF -> local rule"),
                (true, None) => eprintln!(
                    "[envelope-grad] ARMED but UNAVAILABLE (alpha_state={}) -> local rule",
                    alpha_state.is_some()
                ),
                (true, Some(m)) => {
                    let hits = relu_nodes.iter().filter(|n| m.contains_key(*n)).count();
                    eprintln!(
                        "[envelope-grad] ACTIVE nodes={}/{} relus",
                        hits,
                        relu_nodes.len()
                    );
                }
            }
        }
        let mut gradients: Vec<Array1<f32>> = Vec::with_capacity(relu_nodes.len());

        for relu_name in relu_nodes {
            // Get A matrix at this ReLU (before ReLU applied)
            let a_at_relu = match intermediate.a_at_relu(relu_name) {
                Some(a) => a,
                None => {
                    // No intermediate stored for this ReLU — use pre-ReLU bounds
                    // to determine correct gradient length (#1937). A length-1
                    // fallback would panic in alpha update when the ReLU has >1 neuron.
                    let n = intermediate
                        .pre_relu_bounds(relu_name)
                        .map(|(lower, _)| lower.len())
                        .unwrap_or_else(|| {
                            tracing::warn!(
                                "AnalyticChain: missing both A matrix and pre-ReLU bounds for '{}' (#1937)",
                                relu_name
                            );
                            0
                        });
                    gradients.push(Array1::zeros(n));
                    continue;
                }
            };

            // Get pre-ReLU bounds
            let (pre_lower, pre_upper) = match intermediate.pre_relu_bounds(relu_name) {
                Some(b) => b,
                None => {
                    gradients.push(Array1::zeros(a_at_relu.ncols()));
                    continue;
                }
            };

            let n_neurons = pre_lower.len();
            let num_outputs = a_at_relu.nrows();
            let mut grad = Array1::<f32>::zeros(n_neurons);

            // #envelope-grad: the per-neuron factor this layer evaluates
            // `∂bound/∂α` at. `None` ⇒ the local rule's `pre_lower[i]`.
            let hhat: Option<&Array1<f32>> = binding
                .as_ref()
                .and_then(|m| m.get(relu_name))
                .filter(|h| h.len() == n_neurons);

            // #envelope-grad-gpu DIAGNOSTIC: the CPU-lane mirror of
            // `[envelope-rescale]` in propagate_dag/gradients/mod.rs. The two
            // lanes name `x*` differently — this one from
            // `intermediate.final_bounds`, the GPU one from the resident fold's
            // own `lower_bounds` via `binding_row_argmin_corner` — and the GPU
            // rescale is MEASURED not to reproduce this lane's behaviour despite
            // exact algebra. Diffing the `h` columns of the two probe lines on
            // the same row is what tells you whether `x*` is the reason.
            if gradients.is_empty() && envelope_rescale_probe_enabled() {
                let mut sample = String::new();
                let mut unstable = 0usize;
                for i in 0..n_neurons {
                    let (l, u) = (pre_lower[i], pre_upper[i]);
                    if !(l.is_finite() && u.is_finite()) || l >= 0.0 || u <= 0.0 {
                        continue;
                    }
                    unstable += 1;
                    if sample.len() < 220 {
                        let h = hhat.map_or(f32::NAN, |h| h[i]);
                        let f = if h.is_finite() { h.clamp(l, u) } else { l };
                        sample.push_str(&format!("(l={l:.3e} u={u:.3e} h={h:.3e} f={f:.3e}) "));
                    }
                }
                // S-STATISTICS, the GPU probe's mirror. `S = sum_j max(a_ji,0)`
                // is the one input still unaccounted for after x* was shown
                // bit-identical across the lanes.
                let (mut s_zero, mut s_sum) = (0usize, 0.0f64);
                for i in 0..n_neurons {
                    let (l, u) = (pre_lower[i], pre_upper[i]);
                    if !(l.is_finite() && u.is_finite()) || l >= 0.0 || u <= 0.0 {
                        continue;
                    }
                    let mut s = 0.0f32;
                    for j in 0..num_outputs {
                        let a = a_at_relu[[j, i]];
                        if a.is_finite() && a > 0.0 {
                            s += a;
                        }
                    }
                    if s == 0.0 {
                        s_zero += 1;
                    }
                    s_sum += f64::from(s);
                }
                eprintln!(
                    "[envelope-cpu] relu0 unstable={unstable} hhat={} rows={num_outputs} \
                     S_zero={s_zero} S_mean={:.4e}\n  {sample}",
                    if hhat.is_some() { "present" } else { "ABSENT" },
                    s_sum / f64::from(u32::try_from(unstable.max(1)).unwrap_or(1))
                );
            }

            // For each neuron in this ReLU layer
            for i in 0..n_neurons {
                let l = pre_lower[i];
                let u = pre_upper[i];

                // Guard: non-finite pre-ReLU bounds cannot produce meaningful gradients.
                // IEEE-754: NaN comparisons return false, so `l >= 0.0 || u <= 0.0`
                // would fail for NaN bounds, treating them as "unstable" and flowing
                // NaN into gradient arithmetic. Explicitly skip non-finite.
                // Mirrors sequential path guard in helpers.rs (#2809).
                if !l.is_finite() || !u.is_finite() {
                    continue;
                }

                // Only unstable neurons (l < 0 < u) have non-zero gradient
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Compute gradient contribution from all output dimensions
                // For lower relaxation y >= α*x with x ∈ [l, u] where l < 0 < u:
                // - Contribution to lower bound = A[j,i] * α * min(x) = A[j,i] * α * l
                // - Gradient ∂bound/∂α = A[j,i] * l
                // Note: l < 0 for unstable neurons, so gradient is typically negative
                // when A[j,i] > 0, meaning increasing α decreases the lower bound.
                let mut grad_i = 0.0f32;

                // #envelope-grad: use the concrete point-forward surrogate at the
                // binding row's argmin, clamped into this neuron's certified
                // interval. The exact target is the relaxed pre-activation there;
                // the concrete value matches it at the first ReLU and is heuristic
                // deeper. The clamp defends against a shape/order mismatch, and a
                // non-finite value falls back to `l`. Unlike `l`, this factor CAN
                // BE POSITIVE, so the resulting steering field is not sign-definite.
                let factor = match hhat {
                    Some(h) if h[i].is_finite() => h[i].clamp(l, u),
                    _ => l,
                };

                for j in 0..num_outputs {
                    let a_ji = a_at_relu[[j, i]];

                    // Guard: non-finite A coefficient cannot produce meaningful
                    // gradient contributions. Mirrors sequential path guard (#2809).
                    if !a_ji.is_finite() {
                        continue;
                    }

                    // When A >= 0, the lower relaxation y >= α*x is the branch
                    // that depends on α. The legacy local rule evaluates it at
                    // `l`; #envelope-grad substitutes the point-forward surrogate.
                    if a_ji > 0.0 {
                        // Lower relaxation active: y >= α*x
                        // Contribution to lower bound: A[j,i] * α * x|binding
                        // Gradient w.r.t. α: A[j,i] * x|binding
                        // `factor` is `l` under the legacy local rule and the
                        // concrete point-forward surrogate for `ĥ_i(x*)` under
                        // #envelope-grad.
                        grad_i += a_ji * factor;
                    }
                    // When A < 0 the upper chord `y <= d*x + b` is used, with
                    // `d = u/(u-l)` and `b = -l*d`.
                    //
                    // Under a FROZEN intermediate map — `joint_interm_alpha_every
                    // == 0`, which is the shipped default in every yaml today —
                    // `d` and `b` are constants with respect to alpha, so this
                    // gradient really is exactly 0 and the line below is correct.
                    //
                    // Under #joint-interm-grad the map is rebuilt at the current
                    // alpha, so `l` and `u` (hence `d` and `b`) become functions
                    // of alpha and this term is NOT zero. It is not computed here:
                    // this function returns d(bound)/d(alpha) at FIXED bounds, and
                    // the missing piece needs d(l,u)/d(alpha), which is a separate
                    // adjoint. See `interm_sensitivity_weights` below for the
                    // host-side `df/d(l,u)` half.
                }

                grad[i] = grad_i;
            }

            gradients.push(grad);
        }

        gradients
    }

    /// #joint-interm-grad: per-neuron sensitivity of the folded lower bound to a
    /// ReLU node's own INTERMEDIATE bounds — the `df/d(l,u)` half of the term the
    /// loop above deliberately drops.
    ///
    /// # Why this exists
    ///
    /// Under a frozen intermediate map the upper chord is constant in alpha and
    /// its gradient is exactly zero. Once the map is rebuilt at the current alpha
    /// (`joint_interm_alpha_every > 0`) that is false, and the full derivative is
    ///
    /// ```text
    ///   df/dalpha_k  =  [direct, already computed above]
    ///                +  SUM_m SUM_i  df/dl_m[i] * dl_m[i]/dalpha_k
    /// ```
    ///
    /// This returns the first factor. The second is a separate adjoint.
    ///
    /// # The tied form, and why the obvious version is WRONG
    ///
    /// The chord is `d = u/D`, `D = u - l`, with intercept `b = -l*d`. Both move
    /// when `l`/`u` move, and they are TIED: `db/dl = -d^2`, `db/du = (1-d)^2`.
    /// Implementing only the `d` channel is not merely incomplete — on the `l`
    /// axis the `b` channel is roughly 5x larger AND OPPOSITE IN SIGN (worked
    /// example: d-part -0.0347, b-part +0.1736, net +0.1389), so the d-only form
    /// gets the DIRECTION wrong. The expressions below are the tied totals.
    ///
    /// # Sign
    ///
    /// `A_neg <= 0` and `hhat` is clamped into `[l, u]`, so `w_l >= 0` and
    /// `w_u <= 0`: raising `l` or lowering `u` — i.e. TIGHTENING — raises the
    /// bound. That is the direction a correct joint ascent must move, and it is
    /// also what lets the downstream adjoint be seeded with these weights
    /// directly (the device harvest is positive-homogeneous per seed row, so a
    /// non-negative weight vector may be substituted for the unit diagonal).
    ///
    /// # Soundness
    ///
    /// None. This is steering data: it selects a search direction and is never
    /// read by a bound or a verdict. Any alpha in `[0,1]` is a certified-sound
    /// relaxation regardless of how it was chosen (`alpha_sound_regardless`).
    pub(crate) fn interm_sensitivity_weights(
        a_at_relu: &Array2<f32>,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
        hhat: Option<&Array1<f32>>,
    ) -> (Array1<f32>, Array1<f32>) {
        let n = pre_lower.len().min(pre_upper.len()).min(a_at_relu.ncols());
        let mut w_l = Array1::<f32>::zeros(pre_lower.len());
        let mut w_u = Array1::<f32>::zeros(pre_lower.len());
        let rows = a_at_relu.nrows();

        for i in 0..n {
            let l = pre_lower[i];
            let u = pre_upper[i];
            // Same guards as the direct loop: non-finite first (NaN compares
            // false, so the stability test alone would misclassify it), then
            // stable neurons, which have no chord to differentiate.
            if !l.is_finite() || !u.is_finite() {
                continue;
            }
            if l >= 0.0 || u <= 0.0 {
                continue;
            }
            let d_span = f64::from(u) - f64::from(l);
            if d_span <= 0.0 || !d_span.is_finite() {
                continue;
            }

            // The channel the direct loop drops: only the NEGATIVE adjoint
            // entries route through the upper chord.
            let mut a_neg = 0.0f64;
            for j in 0..rows {
                let v = a_at_relu[[j, i]];
                if v.is_finite() && v < 0.0 {
                    a_neg += f64::from(v);
                }
            }
            if a_neg == 0.0 {
                continue;
            }

            // Same surrogate and clamp rule as the direct loop (#envelope-grad):
            // the relaxed pre-activation at the binding row's argmin, clamped
            // into this neuron's certified interval; fall back to `l`.
            let factor = f64::from(match hhat {
                Some(h) if h.len() > i && h[i].is_finite() => h[i].clamp(l, u),
                _ => l,
            });
            let d = f64::from(u) / d_span;

            let wl = a_neg * d * (factor - f64::from(u)) / d_span;
            let wu = a_neg * (1.0 - d) * (factor - f64::from(l)) / d_span;
            if wl.is_finite() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    w_l[i] = wl as f32;
                }
            }
            if wu.is_finite() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    w_u[i] = wu as f32;
                }
            }
        }
        (w_l, w_u)
    }

    /// Concrete point-forward value of each ReLU node's pre-activation at the
    /// binding row's concretization argmin `x*`. This is a non-sign-definite
    /// surrogate for the relaxed-linear `ĥ_i(x*)` factor: exact at the first
    /// ReLU, heuristic at deeper ReLUs. `None` ⇒ caller keeps the local rule
    /// (fail-closed; gradients affect bound quality, not relaxation validity).
    ///
    /// `x*` is the argmin CORNER of the binding output row over the input box.
    /// `concretize` accumulates `lb += A>0 ? A·xl : A·xu`, so `x*_k = lo_k` when
    /// `A[r,k] > 0` and `hi_k` otherwise. Evaluating the network at the
    /// DEGENERATE box `[x*, x*]` with center collapse produces the concrete
    /// activation at each node; it does not reconstruct the relaxed-linear
    /// forward value through upstream unstable ReLUs.
    ///
    /// TWO APPROXIMATIONS, both deliberate and both stated so nobody has to
    /// rediscover them:
    ///
    /// 1. ONE `x*` serves all output rows — the row with the smallest current
    ///    concretized lower bound, which this heuristic prioritizes. The
    ///    derivative of `Σ_j lower_j` would use each row's own argmin, at one
    ///    forward pass per row (`output_dim = 100` on cifar100).
    /// 2. The point forward applies the exact ReLU, while the derivative target
    ///    follows the current lower relaxation (slope or chord) through upstream
    ///    unstable ReLUs. They coincide before the first ReLU and wherever the
    ///    concrete and relaxed upstream paths agree, but not in general deeper.
    ///
    /// Neither can make a bound unsound — α only ever selects a valid relaxation
    /// (`alpha_sound_regardless`, machine-checked). They affect STEP DIRECTION
    /// and bound quality only; the replacement is a measured heuristic, not a
    /// proof that every proposed step is an ascent direction. The local field it
    /// replaces is provably a constant sign.
    fn envelope_binding_points(
        &self,
        input: &BoundedTensor,
        relu_nodes: &[String],
        intermediate: &GraphAlphaCrownIntermediate,
        // The current point-forward surrogate does not consume alpha state.
        // Kept in the signature so the caller's availability gate and the
        // `NY_ALPHA_GRAD_PROBE` line are unchanged; do not infer an alpha-aware
        // relaxed forward from this parameter.
        _alpha_state: &GraphAlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Option<std::collections::HashMap<String, Array1<f32>>> {
        let final_a = intermediate.final_bounds.lower_a();
        let final_b = intermediate.final_bounds.lower_b();
        let in_lo = input.lower();
        let in_hi = input.upper();
        let lo = in_lo.as_slice()?;
        let hi = in_hi.as_slice()?;
        if final_a.nrows() == 0 || final_a.ncols() != lo.len() || lo.len() != hi.len() {
            return None;
        }

        // Binding row = smallest concretized lower bound (`concretize.rs`).
        let mut binding_row = 0usize;
        let mut binding_val = f32::INFINITY;
        for r in 0..final_a.nrows() {
            let mut acc = f64::from(final_b.get(r).copied().unwrap_or(0.0));
            for (k, (&l, &h)) in lo.iter().zip(hi.iter()).enumerate() {
                let a = final_a[[r, k]];
                acc += f64::from(a) * f64::from(if a > 0.0 { l } else { h });
            }
            let acc = acc as f32;
            if acc.is_finite() && acc < binding_val {
                binding_val = acc;
                binding_row = r;
            }
        }
        if !binding_val.is_finite() {
            return None;
        }

        // x*: the binding row's argmin corner.
        let x_star: Vec<f32> = (0..lo.len())
            .map(|k| {
                if final_a[[binding_row, k]] > 0.0 {
                    lo[k]
                } else {
                    hi[k]
                }
            })
            .collect();
        if x_star.iter().any(|v| !v.is_finite()) {
            return None;
        }

        if envelope_xstar_probe_enabled() {
            eprintln!("[xstar-cpu] {}", summarize_x_star(binding_row, &x_star));
        }
        self.envelope_points_at(input, relu_nodes, &x_star, engine)
    }

    /// The point-forward half of [`Self::envelope_binding_points`], with `x*`
    /// SUPPLIED rather than derived.
    ///
    /// Split out because the GPU-resident warmup lane can NAME the same `x*` —
    /// the argmin corner of the binding row of the fold it already ran — but has
    /// no [`GraphAlphaCrownIntermediate`] and must never build one: that is
    /// `dag_alpha_backward_pass_with_intermediates`, the ~27 s/iteration CPU
    /// AnalyticChain backward the GPU lane exists to avoid.
    ///
    /// Both lanes MUST route through here. Given the same `x*` the factor is
    /// then bit-identical and the only remaining difference between the lanes is
    /// how `x*` was named. Every approximation documented on the parent applies
    /// verbatim.
    pub(in crate::network::graph_alpha) fn envelope_points_at(
        &self,
        input: &BoundedTensor,
        relu_nodes: &[String],
        x_star: &[f32],
        engine: Option<&dyn GemmEngine>,
    ) -> Option<std::collections::HashMap<String, Array1<f32>>> {
        let in_lo = input.lower();
        if x_star.len() != in_lo.len() || x_star.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let x_star = x_star.to_vec();

        // Forward evaluation AT x*, via a degenerate box `[x*, x*]`.
        //
        // What this computes, precisely, because two other candidates were
        // measured against the finite-difference oracle and both were worse:
        //
        // * At a degenerate box every node's pre-activation is a POINT, so the
        //   pass classifies every neuron stable and applies the EXACT ReLU. The
        //   result is therefore the CONCRETE forward at x*, not the α-relaxed
        //   one. (Supplying per-neuron slopes here is inert for exactly this
        //   reason — measured bit-identical.)
        // * Evaluating the real box's forward-linear LOWER function at x*
        //   instead gives a lower BOUND on the relaxed value, not the value;
        //   composed over layers it under-estimates badly and clamps to `l`.
        //   Measured max|fd-grad| 1.62e0 vs 1.10e-1 for this path.
        //
        // Consequence, stated so nobody re-derives it: for the FIRST ReLU layer
        // the pre-activation is affine in the input, so `ĥ` here is EXACT
        // (measured `fd=+0.11275` vs `envelope=+0.11276`). Deeper layers use the
        // concrete rather than the relaxed forward and are approximate. The
        // finite-difference fixtures show an aggregate/sign improvement, not
        // per-neuron dominance. Unlike the constant `l`, the surrogate is not
        // sign-definite.
        let point = ndarray::ArrayD::from_shape_vec(in_lo.raw_dim(), x_star.clone()).ok()?;
        let degenerate = BoundedTensor::new(point.clone(), point).ok()?;
        // COST/APPROXIMATION TRADEOFF: compute the concrete point forward used by
        // this heuristic directly.
        //
        // What was here before: `collect_forward_linear_bounds_dag_with_alphas`,
        // a dense forward-linear AFFINE composition carrying a
        // `(node_width, n_inputs)` coefficient pair through every node. At
        // cifar100's `n_inputs = 3072` that is ~6144x the MAC count of a point
        // forward, it takes ~15 s, and it is uninterruptible — which cut the
        // ascent from 6/20 iterations to 1/20 (measured 2026-08-12: interior
        // alphas 132 vs the local rule's 59). Replacing that relaxed-linear
        // composition with a concrete point forward changes the factor at deeper
        // ReLUs; it is not a semantics-preserving reduction.
        //
        // THE CENTER-COLLAPSE IS REQUIRED, not a refinement. The obvious cheaper
        // call, `collect_node_bounds_with_engine`, is the SAME sweep without it —
        // and a point input does NOT stay a point through a plain box IBP: per-node
        // soundness widening (BatchNorm worst) is amplified multiplicatively by a
        // deep conv stack, so `.lower()` drifts far below the true activation.
        // That exact read is a production incident in this repo — it fabricated
        // false counterexamples on cgan_2023 (`pgd_attack/attacker/eval.rs`,
        // #cgan-eval). Using it here would re-bias the deep layers back toward
        // negative `h-hat`, i.e. back toward the sign-definite defect this gate
        // exists to escape, while LOOKING like a pure cost fix.
        //
        // These point values are not themselves verdict-bearing bounds. They do
        // steer alpha and can therefore influence a later certified bound
        // indirectly. That remains sound because `clamp_alpha_to_envelope_domain`
        // keeps alpha in [0,1], where every selected relaxation is valid; the
        // point values must never be published as a domain enclosure.
        let node_bounds = self
            .collect_node_activations_pointwise(&degenerate, engine)
            .ok()?;

        let mut out = std::collections::HashMap::with_capacity(relu_nodes.len());
        for name in relu_nodes {
            let Some(node) = self.nodes.get(name) else {
                continue;
            };
            let Some(pre_name) = node.inputs.first() else {
                continue;
            };
            // The ReLU's PRE-activation is its input node's value at x*.
            let values = if pre_name == NETWORK_INPUT {
                Array1::from_vec(x_star.clone())
            } else {
                match node_bounds.get(pre_name).and_then(|b| {
                    let lo = b.lower();
                    lo.as_slice().map(|v| Array1::from_vec(v.to_vec()))
                }) {
                    Some(v) => v,
                    None => continue,
                }
            };
            out.insert(name.clone(), values);
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

/// #envelope-grad-gpu: a comparable one-line summary of a lane's `x*`.
///
/// The CPU lane and the GPU warmup lane name `x*` by different routes (the
/// former from `intermediate.final_bounds`, the latter from the resident fold's
/// own `lower_bounds` via `binding_row_argmin_corner`) and CANNOT run in the
/// same process, so the only way to compare them is to print a canonical digest
/// from each and diff it offline. Corner-valued `x*` means every entry is one of
/// the two box endpoints, so the sign pattern is the whole content: `lo_count`
/// plus the checksum pins it far more tightly than the leading values alone.
pub(in crate::network::graph_alpha) fn summarize_x_star(
    binding_row: usize,
    x_star: &[f32],
) -> String {
    let lo_bits: u64 = x_star
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if *v < 0.0 {
                (i as u64).wrapping_mul(2_654_435_761)
            } else {
                0
            }
        })
        .fold(0u64, |a, b| a ^ b);
    let neg = x_star.iter().filter(|v| **v < 0.0).count();
    let head: Vec<String> = x_star.iter().take(6).map(|v| format!("{v:.4e}")).collect();
    format!(
        "binding_row={binding_row} n={} negatives={neg} sign_digest={lo_bits:016x} head=[{}]",
        x_star.len(),
        head.join(" ")
    )
}

/// #joint-interm-grad: finite-difference oracle for the `df/d(l,u)` weights.
///
/// A sign error here never crashes, never yields an unsound bound, and never
/// fails an existing test — it just walks the ascent AWAY from a proof. That is
/// exactly the failure already catalogued in the direct term
/// (`AlphaGradientDefect.lean`: the shipped local rule is sign-definite, so Adam
/// can only drive alpha to the 0 clamp). The only cheap defence is to difference
/// the real thing, so these tests re-implement the chord independently rather
/// than sharing code with the implementation.
#[cfg(test)]
mod joint_interm_grad_oracle {
    use super::GraphNetwork;
    use ndarray::{Array1, Array2};

    /// The quantity being differentiated: one ReLU's UPPER-chord contribution to
    /// the folded lower bound. Only NEGATIVE adjoint entries route through it,
    /// and the chord is `d*x + b` with `d = u/(u-l)`, `b = -l*d`.
    fn upper_chord_contribution(a_neg: f64, l: f64, u: f64, factor: f64) -> f64 {
        let d = u / (u - l);
        let b = -l * d;
        a_neg * d.mul_add(factor, b)
    }

    fn fd_weights(a_neg: f64, l: f64, u: f64, factor: f64, h: f64) -> (f64, f64) {
        let dl = (upper_chord_contribution(a_neg, l + h, u, factor)
            - upper_chord_contribution(a_neg, l - h, u, factor))
            / (2.0 * h);
        let du = (upper_chord_contribution(a_neg, l, u + h, factor)
            - upper_chord_contribution(a_neg, l, u - h, factor))
            / (2.0 * h);
        (dl, du)
    }

    fn analytic(a_neg: f32, l: f32, u: f32, factor: f32) -> (f32, f32) {
        let a = Array2::from_shape_vec((1, 1), vec![a_neg]).expect("1x1 adjoint");
        let lower = Array1::from_vec(vec![l]);
        let upper = Array1::from_vec(vec![u]);
        let hhat = Array1::from_vec(vec![factor]);
        let (w_l, w_u) = GraphNetwork::interm_sensitivity_weights(&a, &lower, &upper, Some(&hhat));
        (w_l[0], w_u[0])
    }

    #[test]
    fn analytic_weights_match_finite_differences() {
        let cases: &[(f32, f32, f32, f32)] = &[
            (-1.0, -1.0, 1.0, 0.0),
            (-0.5, -2.0, 0.5, -0.5),
            (-2.0, -0.5, 3.0, 1.0),
            (-0.25, -10.0, 2.0, -3.0),
            (-3.0, -0.1, 0.2, 0.05),
        ];
        for &(a_neg, l, u, factor) in cases {
            let (wl, wu) = analytic(a_neg, l, u, factor);
            let (fl, fu) = fd_weights(
                f64::from(a_neg),
                f64::from(l),
                f64::from(u),
                f64::from(factor),
                1e-4,
            );
            let close = |a: f32, b: f64| {
                let scale = f64::from(a).abs().max(b.abs()).max(1e-3);
                (f64::from(a) - b).abs() <= 2e-2 * scale
            };
            assert!(
                close(wl, fl),
                "df/dl mismatch at (a_neg={a_neg}, l={l}, u={u}, factor={factor}): \
                 analytic={wl:.6} fd={fl:.6}. Most likely cause: implementing only \
                 the slope channel `d` and dropping the tied intercept `b = -l*d`. \
                 On the l axis those carry OPPOSITE signs and the intercept is the \
                 larger, so a d-only form points the ascent the wrong way."
            );
            assert!(
                close(wu, fu),
                "df/du mismatch at (a_neg={a_neg}, l={l}, u={u}): analytic={wu:.6} fd={fu:.6}"
            );
        }
    }

    #[test]
    fn tightening_raises_the_bound() {
        // The property the joint program rests on. If this sign is wrong, a
        // "working" gradient steers away from the proof.
        for &(a_neg, l, u, factor) in &[
            (-1.0f32, -1.0f32, 1.0f32, 0.0f32),
            (-0.5, -2.0, 0.5, -0.5),
            (-2.0, -0.5, 3.0, 1.0),
        ] {
            let (wl, wu) = analytic(a_neg, l, u, factor);
            assert!(wl >= 0.0, "df/dl must be >= 0, got {wl} at (l={l}, u={u})");
            assert!(wu <= 0.0, "df/du must be <= 0, got {wu} at (l={l}, u={u})");
        }
    }

    #[test]
    fn stable_and_non_finite_neurons_yield_zero() {
        // Guard parity with the direct loop: a stable neuron has no chord, and a
        // non-finite bound must not flow NaN into steering data, where it would
        // poison an Adam moment silently rather than fail loudly.
        for &(l, u) in &[
            (0.5f32, 2.0f32),
            (-2.0, -0.5),
            (f32::NAN, 1.0),
            (-1.0, f32::INFINITY),
        ] {
            let (wl, wu) = analytic(-1.0, l, u, 0.0);
            assert!(
                wl == 0.0 && wu == 0.0,
                "expected zero weights at (l={l}, u={u}), got ({wl}, {wu})"
            );
        }
    }

    #[test]
    fn positive_adjoint_entries_contribute_nothing() {
        // Positive entries route through the LOWER relaxation, which the direct
        // term already owns. Counting them here would double-count it.
        let a = Array2::from_shape_vec((2, 1), vec![1.5f32, 0.25]).expect("2x1 adjoint");
        let lower = Array1::from_vec(vec![-1.0f32]);
        let upper = Array1::from_vec(vec![1.0f32]);
        let (w_l, w_u) = GraphNetwork::interm_sensitivity_weights(&a, &lower, &upper, None);
        assert!(
            w_l[0] == 0.0 && w_u[0] == 0.0,
            "only negative adjoint entries may contribute to the upper-chord term"
        );
    }
}
