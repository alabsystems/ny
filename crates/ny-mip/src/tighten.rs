// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Per-neuron LP bound tightening via the ay backend (SOLVER POLICY:
// docs/SOLVER_POLICY.md — all solving happens on ay).
//
// Encodes network constraints as a solver-neutral LP IR (triangle relaxation
// for ReLU, no binary indicators) and solves min/max for each unstable
// neuron to get tighter pre-activation bounds.
//
// Reference: alpha-beta-CROWN complete_verifier/lp_mip_solver/bounds_core.py:37-92
// Reference: designs/2026-03-04-highs-mip-solver-integration.md (Phase 2)
// Part of #1763.

use crate::ay::{ObjSense, ObjectiveSpec};
use crate::config::MipConfig;
use crate::error::MipError;
use crate::ir::{Col, MilpProblem};
use crate::solver::MipResult;
use ny_core::{Bound, BoundTightener};
use ny_tensor::{next_down_f32, next_up_f32};
use tracing::debug;

type Result<T> = std::result::Result<T, MipError>;

/// LP bound tightener for FC+ReLU networks.
///
/// Stores the network structure (weights, biases, dimensions, bounds) and
/// solves per-neuron LP relaxations to tighten intermediate bounds. For each
/// unstable neuron, the tightener:
/// 1. Builds an LP encoding the network up to that layer using triangle
///    relaxation for ReLU (no binary indicators)
/// 2. Minimizes the neuron's pre-activation value → new lower bound
/// 3. Maximizes the neuron's pre-activation value → new upper bound
/// 4. Intersects with existing bounds
///
/// The LP relaxation uses the convex hull of ReLU for unstable neurons:
///   y >= 0,  y >= x,  y <= u*(x-l)/(u-l)
/// which is tighter than IBP but weaker than full MIP.
///
/// Reference: alpha-beta-CROWN `build_the_model_mip_refine()` in
/// `complete_verifier/lp_mip_solver/refine_core.py:39-451`
pub struct LpTightener {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    layer_dims: Vec<usize>,
    input_bounds: Vec<Bound>,
    intermediate_bounds: Vec<Vec<Bound>>,
    config: MipConfig,
}

/// Result of tightening a single neuron.
#[derive(Debug)]
pub(crate) struct NeuronTightenResult {
    /// New lower bound (may be same as original if LP didn't improve).
    pub(crate) lower: f32,
    /// New upper bound (may be same as original if LP didn't improve).
    pub(crate) upper: f32,
    /// Whether the neuron's stability status changed.
    pub(crate) became_stable: bool,
}

impl LpTightener {
    fn deadline_expired(&self) -> bool {
        self.config
            .ay_hard_deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
    }

    fn live_per_neuron_timeout_secs(&self) -> Option<f64> {
        let nominal = if self.config.ay_hard_deadline.is_some() {
            (self.config.timeout_secs / 10.0).clamp(0.001, 30.0)
        } else {
            (self.config.timeout_secs / 10.0).clamp(1.0, 30.0)
        };
        match self.config.ay_hard_deadline {
            Some(deadline) => {
                let remaining = deadline
                    .checked_duration_since(std::time::Instant::now())?
                    .as_secs_f64();
                let live = nominal.min(remaining);
                (live >= 0.001).then_some(live)
            }
            None => Some(nominal),
        }
    }

    /// Create a new LP tightener from network parameters.
    ///
    /// # Arguments
    ///
    /// * `weights` — Weight matrices per layer, row-major `[out × in]`
    /// * `biases` — Bias vectors per layer
    /// * `layer_dims` — `[input_dim, hidden1, ..., output_dim]`
    /// * `input_bounds` — Bounds on network inputs
    /// * `intermediate_bounds` — Pre-activation bounds per hidden layer
    /// * `config` — Solver configuration (timeout, threads)
    pub fn new(
        weights: Vec<Vec<f64>>,
        biases: Vec<Vec<f64>>,
        layer_dims: Vec<usize>,
        input_bounds: Vec<Bound>,
        intermediate_bounds: Vec<Vec<Bound>>,
        config: MipConfig,
    ) -> Self {
        Self {
            weights,
            biases,
            layer_dims,
            input_bounds,
            intermediate_bounds,
            config,
        }
    }

    /// Tighten pre-activation bounds for all neurons in a target layer.
    ///
    /// Only unstable neurons (lower < 0 < upper) are optimized. Stable
    /// neurons are returned unchanged.
    ///
    /// Returns the number of neurons that became stable due to tightening.
    pub fn tighten_layer(
        &self,
        target_layer: usize,
        current_bounds: &[Bound],
    ) -> Result<(Vec<Bound>, usize)> {
        // Validate target_layer is within network bounds
        if target_layer >= self.weights.len() {
            return Err(MipError::Encoding(format!(
                "target_layer {} out of range (network has {} layers)",
                target_layer,
                self.weights.len()
            )));
        }
        let expected_dim = self.layer_dims[target_layer + 1];
        if current_bounds.len() != expected_dim {
            return Err(MipError::Encoding(format!(
                "current_bounds length {} doesn't match layer {} dimension {}",
                current_bounds.len(),
                target_layer,
                expected_dim
            )));
        }
        if self.deadline_expired() {
            return Ok((current_bounds.to_vec(), 0));
        }

        let max_per_layer = self.config.max_tighten_per_layer;
        let mut tightened = current_bounds.to_vec();
        let mut newly_stable = 0usize;
        let mut tightened_count = 0usize;

        // One LP per LAYER, one warm session, two re-solves per neuron
        // (in-process backend). The constraints are identical for every
        // neuron in the layer — only the objective column changes — so the
        // per-neuron LP rebuild of the P0 lane is pure waste. AyProc keeps
        // the per-neuron subprocess path.
        let mut session = match self.config.backend {
            crate::config::MipBackend::AyProc => None,
            _ => {
                let (problem, targets) = self.build_layer_lp(target_layer)?;
                if self.deadline_expired() {
                    return Ok((current_bounds.to_vec(), 0));
                }
                let model = crate::ay_lib::to_ay_model(&problem)?;
                // Column order is preserved by the lowering, so the IR
                // target columns map to ay-milp columns at the same index.
                let targets: Vec<ay_milp::Col> = targets
                    .iter()
                    .map(|c| {
                        model.col_at(c.0).ok_or_else(|| {
                            MipError::Encoding(format!("target column {} out of range", c.0))
                        })
                    })
                    .collect::<Result<_>>()?;
                let Some(per_neuron_timeout) = self.live_per_neuron_timeout_secs() else {
                    return Ok((current_bounds.to_vec(), 0));
                };
                let mut opts = ay_milp::SolveOpts::new()
                    .with_time_limit(std::time::Duration::from_secs_f64(per_neuron_timeout));
                if let Some(deadline) = self.config.ay_hard_deadline {
                    opts = opts.with_deadline(deadline);
                }
                let session = ay_milp::LpSession::new(&model, &opts)
                    .map_err(|e| MipError::Solver(e.to_string()))?;
                Some((session, targets))
            }
        };

        // OBBT path (P3): tighten the layer's unstable pre-activation columns
        // as ONE coupled set on the warm session, so a proven bound on one
        // neuron can tighten a sibling sharing the previous layer's variables
        // — a fixpoint the independent per-neuron pass below cannot reach.
        // Only the in-process session backend; opt-in via `obbt_rounds`.
        if self.config.obbt_rounds > 0 {
            if let Some((session, targets)) = &mut session {
                let newly_stable = Self::tighten_layer_via_obbt(
                    session,
                    targets,
                    current_bounds,
                    max_per_layer,
                    self.config.obbt_rounds,
                    &mut tightened,
                )?;
                debug!(
                    "LP+OBBT tightened layer {target_layer} ({} rounds): {newly_stable} became stable",
                    self.config.obbt_rounds
                );
                return Ok((tightened, newly_stable));
            }
        }

        for (i, bound) in current_bounds.iter().enumerate() {
            if self.deadline_expired() {
                break;
            }
            // Skip stable neurons
            if bound.lower() >= 0.0 || bound.upper() <= 0.0 {
                continue;
            }

            // Respect per-layer limit
            if max_per_layer > 0 && tightened_count >= max_per_layer {
                break;
            }

            let result = match &mut session {
                Some((session, targets)) => {
                    self.tighten_neuron_in_session(session, targets[i], *bound)?
                }
                None => self.tighten_neuron(target_layer, i, *bound)?,
            };
            tightened[i] = Bound::new(result.lower, result.upper);

            if result.became_stable {
                newly_stable += 1;
            }
            tightened_count += 1;
        }

        debug!(
            "LP tightened layer {target_layer}: {tightened_count} neurons optimized, \
             {newly_stable} became stable"
        );

        Ok((tightened, newly_stable))
    }

    /// Tighten one neuron on a warm per-layer `LpSession` (in-process
    /// backend): two objective-only re-solves on a persistent basis.
    /// Same rounding contract as [`Self::tighten_neuron`].
    fn tighten_neuron_in_session(
        &self,
        session: &mut ay_milp::LpSession,
        target: ay_milp::Col,
        current: Bound,
    ) -> Result<NeuronTightenResult> {
        let lb = current.lower();
        let ub = current.upper();

        let new_lb = match Self::session_optimum(session, target, ay_milp::Sense::Minimize)? {
            // Soundness: f64→f32 rounds DOWN for lower bounds; never widen.
            Some(opt_val) => next_down_f32(opt_val as f32).max(lb),
            None => lb,
        };

        // Early exit: if lower bound is now >= 0, neuron is always active.
        if new_lb >= 0.0 {
            return Ok(NeuronTightenResult {
                lower: new_lb,
                upper: ub,
                became_stable: true,
            });
        }

        let new_ub = match Self::session_optimum(session, target, ay_milp::Sense::Maximize)? {
            // Soundness: f64→f32 rounds UP for upper bounds; never widen.
            Some(opt_val) => next_up_f32(opt_val as f32).min(ub),
            None => ub,
        };

        // Defensive guard: rounding could theoretically invert bounds.
        let (safe_lb, safe_ub) = if new_lb > new_ub {
            (lb, ub)
        } else {
            (new_lb, new_ub)
        };
        let became_stable = safe_ub <= 0.0;

        Ok(NeuronTightenResult {
            lower: safe_lb,
            upper: safe_ub,
            became_stable,
        })
    }

    /// Tighten a layer's unstable neurons as a coupled OBBT set (P3).
    ///
    /// Selects the unstable neurons (respecting `max_per_layer`), runs the
    /// `LpSession` OBBT fixpoint over their pre-activation columns, then reads
    /// each tightened box back under the SAME rounding contract as the
    /// per-neuron path: f64→f32 rounds DOWN for lower bounds and UP for upper
    /// bounds (never widen), with the defensive inversion guard. OBBT commits
    /// only rigorous bounds, so the result is always at least as tight as the
    /// independent pass and always sound. Returns the newly-stable count.
    fn tighten_layer_via_obbt(
        session: &mut ay_milp::LpSession,
        targets: &[ay_milp::Col],
        current_bounds: &[Bound],
        max_per_layer: usize,
        rounds: usize,
        tightened: &mut [Bound],
    ) -> Result<usize> {
        let mut selected: Vec<usize> = Vec::new();
        for (i, bound) in current_bounds.iter().enumerate() {
            if bound.lower() >= 0.0 || bound.upper() <= 0.0 {
                continue; // stable: nothing to tighten
            }
            if max_per_layer > 0 && selected.len() >= max_per_layer {
                break;
            }
            selected.push(i);
        }
        if selected.is_empty() {
            return Ok(0);
        }
        let sel_cols: Vec<ay_milp::Col> = selected.iter().map(|&i| targets[i]).collect();
        let opts = ay_milp::ObbtOpts {
            max_rounds: rounds,
            ..ay_milp::ObbtOpts::default()
        };
        let report = session
            .obbt(&sel_cols, &opts)
            .map_err(|e| MipError::Solver(e.to_string()))?;
        // An infeasible layer LP yields meaningless per-column boxes: keep the
        // original (sound) bounds, tighten nothing.
        if report.infeasible {
            return Ok(0);
        }
        let mut newly_stable = 0usize;
        for (k, &i) in selected.iter().enumerate() {
            let (lb_f64, ub_f64) = report.bounds[k];
            let lb = current_bounds[i].lower();
            let ub = current_bounds[i].upper();
            // Soundness: f64→f32 rounds DOWN for lower, UP for upper; clamp so
            // the LP result never widens the original bound.
            let new_lb = next_down_f32(lb_f64 as f32).max(lb);
            let new_ub = next_up_f32(ub_f64 as f32).min(ub);
            let (safe_lb, safe_ub) = if new_lb > new_ub {
                (lb, ub)
            } else {
                (new_lb, new_ub)
            };
            tightened[i] = Bound::new(safe_lb, safe_ub);
            if safe_lb >= 0.0 || safe_ub <= 0.0 {
                newly_stable += 1;
            }
        }
        Ok(newly_stable)
    }

    /// One warm re-solve on the rigorous W2 surface: a float basis plus a
    /// Neumaier-Shcherbina directed-rounding correction — every returned
    /// value is a RIGOROUS bound on the true optimum (exact-certified
    /// fallback included). `None` for anything inconclusive (the caller
    /// keeps the original sound bound).
    fn session_optimum(
        session: &mut ay_milp::LpSession,
        target: ay_milp::Col,
        sense: ay_milp::Sense,
    ) -> Result<Option<f64>> {
        use num_traits::ToPrimitive;
        match session
            .rigorous_bound(target, sense)
            .map_err(|e| MipError::Solver(e.to_string()))?
        {
            ay_milp::Outcome::Bound {
                dual_bound,
                rigorous: true,
            } => Ok(dual_bound.to_f64()),
            ay_milp::Outcome::Optimal { value, .. } => Ok(value.to_f64().map_or_else(
                || {
                    tracing::warn!("LP tighten optimum does not fit f64; keeping original bound");
                    None
                },
                Some,
            )),
            _ => Ok(None),
        }
    }

    /// Build the layer LP once: identical constraints for every neuron in
    /// the layer, plus the full vector of target pre-activation columns.
    fn build_layer_lp(&self, target_layer: usize) -> Result<(MilpProblem, Vec<Col>)> {
        let mut problem = MilpProblem::new();

        let mut current_vars = Vec::with_capacity(self.input_bounds.len());
        for b in &self.input_bounds {
            let col = problem.add_col(0.0, b.lower() as f64, b.upper() as f64);
            current_vars.push(col);
        }

        let num_layers = target_layer + 1;
        let mut targets = Vec::new();
        for layer_idx in 0..num_layers {
            let in_features = current_vars.len();
            let out_features = self.layer_dims[layer_idx + 1];
            let is_target = layer_idx == target_layer;

            let mut linear_out = Vec::with_capacity(out_features);
            for i in 0..out_features {
                let y_var = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);

                let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(in_features + 1);
                for (j, &x_var) in current_vars.iter().enumerate() {
                    let w = self.weights[layer_idx][i * in_features + j];
                    if w != 0.0 {
                        coeffs.push((x_var, w));
                    }
                }
                coeffs.push((y_var, -1.0));
                let neg_b = -self.biases[layer_idx][i];
                problem.add_row(neg_b, neg_b, coeffs);

                linear_out.push(y_var);
            }

            if is_target {
                targets = linear_out.clone();
            }
            current_vars = linear_out;

            if !is_target && layer_idx < self.intermediate_bounds.len() {
                current_vars = Self::encode_relu_triangle(
                    &mut problem,
                    &current_vars,
                    &self.intermediate_bounds[layer_idx],
                );
            }
        }

        Ok((problem, targets))
    }

    /// Tighten bounds for a single neuron.
    fn tighten_neuron(
        &self,
        layer_idx: usize,
        neuron_idx: usize,
        current: Bound,
    ) -> Result<NeuronTightenResult> {
        let lb = current.lower();
        let ub = current.upper();

        // Tighten lower bound (minimize).
        // Soundness: f64→f32 must round DOWN for lower bounds.
        // Round-to-nearest (bare `as f32`) can round UP by 1 ULP, excluding
        // reachable states from the bound.
        // Reference: IEEE 754-2019 §4.3.1; ny-tensor rounding utilities.
        let new_lb = match self.optimize_neuron(layer_idx, neuron_idx, ObjSense::Minimize)? {
            Some(opt_val) => {
                let candidate = next_down_f32(opt_val as f32);
                // LP result must not widen bounds
                candidate.max(lb)
            }
            None => lb,
        };

        // Early exit: if lower bound is now >= 0, neuron is always active
        if new_lb >= 0.0 {
            return Ok(NeuronTightenResult {
                lower: new_lb,
                upper: ub,
                became_stable: true,
            });
        }

        // Tighten upper bound (maximize).
        // Soundness: f64→f32 must round UP for upper bounds.
        // Reference: IEEE 754-2019 §4.3.1; ny-tensor rounding utilities.
        let new_ub = match self.optimize_neuron(layer_idx, neuron_idx, ObjSense::Maximize)? {
            Some(opt_val) => {
                let candidate = next_up_f32(opt_val as f32);
                // LP result must not widen bounds
                candidate.min(ub)
            }
            None => ub,
        };

        // Defensive guard: f64->f32 rounding could theoretically invert bounds.
        // Clamp to maintain invariant lower <= upper.
        let (safe_lb, safe_ub) = if new_lb > new_ub {
            (lb, ub) // fall back to original bounds on rounding artifact
        } else {
            (new_lb, new_ub)
        };

        // After the early exit above, new_lb < 0.0 is guaranteed.
        // So became_stable only if upper bound proves always-inactive.
        let became_stable = safe_ub <= 0.0;

        Ok(NeuronTightenResult {
            lower: safe_lb,
            upper: safe_ub,
            became_stable,
        })
    }

    /// Optimize a single neuron's pre-activation value on the ay backend.
    ///
    /// Returns the optimal value, or None if the LP is infeasible or times
    /// out. A bound from a timed-out/inconclusive solve is never used — the
    /// caller keeps the original (sound) bound.
    fn optimize_neuron(
        &self,
        layer_idx: usize,
        neuron_idx: usize,
        sense: ObjSense,
    ) -> Result<Option<f64>> {
        let (problem, target) = self.build_lp(layer_idx, neuron_idx)?;
        // AyProc has no typed absolute-deadline surface, so recompute a live
        // relative slice after model construction and poll again between
        // neurons. The in-process path uses SolveOpts::with_deadline above.
        let Some(per_neuron_timeout) = self.live_per_neuron_timeout_secs() else {
            return Ok(None);
        };
        let result = crate::ay::optimize_col(
            &problem,
            per_neuron_timeout,
            ObjectiveSpec { col: target, sense },
            &[],
            &[],
        )?;
        match result {
            // SOUNDNESS INVARIANT (AyProc-only path — the in-process backend
            // uses `session_optimum`'s rigorous bounds instead): the frozen P0
            // subprocess lane reports Sat only for a COMPLETED optimization
            // (interrupted solves parse as `unknown` -> Timeout), so
            // `objective` here is always the true optimum — never an
            // incumbent-only value, which would be on the WRONG side for
            // bound tightening.
            MipResult::Sat { objective, .. } => Ok(Some(objective)),
            MipResult::Unsat { .. } | MipResult::Timeout => Ok(None),
            MipResult::Error(e) => Err(MipError::Solver(format!(
                "LP tighten neuron [{layer_idx}][{neuron_idx}]: {e}"
            ))),
        }
    }

    /// Build an LP (solver-neutral IR) encoding the network up to
    /// `target_layer`, returning the problem and the target neuron's
    /// pre-activation column (the optimization objective).
    ///
    /// Uses triangle relaxation for ReLU (no binary indicators):
    ///   y >= 0,  y >= x,  y <= u*(x-l)/(u-l)
    fn build_lp(&self, target_layer: usize, target_neuron: usize) -> Result<(MilpProblem, Col)> {
        let mut problem = MilpProblem::new();

        // Input variables (bounded by input region)
        let mut current_vars = Vec::with_capacity(self.input_bounds.len());
        for b in &self.input_bounds {
            let col = problem.add_col(0.0, b.lower() as f64, b.upper() as f64);
            current_vars.push(col);
        }

        // Encode layers 0..=target_layer
        let num_layers = target_layer + 1;
        let mut target_col = None;
        for layer_idx in 0..num_layers {
            let in_features = current_vars.len();
            let out_features = self.layer_dims[layer_idx + 1];
            let is_target = layer_idx == target_layer;

            // Encode linear layer: y = Wx + b
            let mut linear_out = Vec::with_capacity(out_features);
            for i in 0..out_features {
                let y_var = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
                if is_target && i == target_neuron {
                    target_col = Some(y_var);
                }

                // Equality constraint: sum(w_ij * x_j) - y_i = -b_i
                let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(in_features + 1);
                for (j, &x_var) in current_vars.iter().enumerate() {
                    let w = self.weights[layer_idx][i * in_features + j];
                    if w != 0.0 {
                        coeffs.push((x_var, w));
                    }
                }
                coeffs.push((y_var, -1.0));
                let neg_b = -self.biases[layer_idx][i];
                problem.add_row(neg_b, neg_b, coeffs);

                linear_out.push(y_var);
            }

            current_vars = linear_out;

            // Apply triangle relaxation for ReLU on non-final, non-target layers.
            // For the target layer we want pre-activation values.
            if !is_target && layer_idx < self.intermediate_bounds.len() {
                current_vars = Self::encode_relu_triangle(
                    &mut problem,
                    &current_vars,
                    &self.intermediate_bounds[layer_idx],
                );
            }
        }

        let target_col = target_col.ok_or_else(|| {
            MipError::Encoding(format!(
                "target neuron [{target_layer}][{target_neuron}] out of range"
            ))
        })?;
        Ok((problem, target_col))
    }

    /// Encode ReLU using triangle (LP) relaxation.
    ///
    /// For each neuron with pre-activation bounds [l, u]:
    /// - l >= 0: y = x (always active, passthrough)
    /// - u <= 0: y = 0 (always inactive, fixed to zero)
    /// - Otherwise (unstable): triangle relaxation
    ///   - y >= 0
    ///   - y >= x
    ///   - y <= u*(x - l)/(u - l)  (upper envelope line connecting (l,0) to (u,u))
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py:763-768
    fn encode_relu_triangle(
        problem: &mut MilpProblem,
        pre_act_vars: &[Col],
        bounds: &[Bound],
    ) -> Vec<Col> {
        let mut post_act_vars = Vec::with_capacity(pre_act_vars.len());

        for (i, &x_var) in pre_act_vars.iter().enumerate() {
            let lb = bounds[i].lower() as f64;
            let ub = bounds[i].upper() as f64;

            if lb >= 0.0 {
                // Always active: y = x (no new variable needed)
                post_act_vars.push(x_var);
            } else if ub <= 0.0 {
                // Always inactive: y = 0
                let y_var = problem.add_col(0.0, 0.0, 0.0);
                post_act_vars.push(y_var);
            } else {
                // Unstable: triangle relaxation
                // y bounded in [0, ub] (tighter than unbounded)
                let y_var = problem.add_col(0.0, 0.0, ub);

                // y >= x → y - x >= 0
                problem.add_row(0.0, f64::INFINITY, [(y_var, 1.0), (x_var, -1.0)]);

                // Upper envelope: y <= u*(x - l)/(u - l)
                // Let slope = u/(u-l). Then: y <= slope*(x - l) = slope*x - slope*l
                // Rearranged: y - slope*x <= -slope*l
                let slope = ub / (ub - lb);
                problem.add_row(
                    f64::NEG_INFINITY,
                    -slope * lb,
                    [(y_var, 1.0), (x_var, -slope)],
                );

                post_act_vars.push(y_var);
            }
        }

        post_act_vars
    }
}

impl BoundTightener for LpTightener {
    type Error = MipError;

    fn tighten(
        &self,
        layer_idx: usize,
        current_bounds: &[Bound],
    ) -> std::result::Result<Vec<Bound>, MipError> {
        let (tightened, _newly_stable) = self.tighten_layer(layer_idx, current_bounds)?;
        Ok(tightened)
    }
}

/// Outcome of an OBBT pass over an LP relaxation ([`obbt_relaxation_bounds`]).
#[derive(Debug, Clone)]
pub struct RelaxationObbt {
    /// Final `(lb, ub)` per target column, in the order they were given. These
    /// are rigorous outward-rounded f64 bounds; the caller intersects them into
    /// its own (typically f32) boxes with the never-widen rounding contract.
    pub bounds: Vec<(f64, f64)>,
    /// A rigorous solve proved the whole relaxation infeasible. When set, the
    /// per-column `bounds` are not meaningful — the caller must NOT treat this
    /// as a certified verdict (this is only the relaxation, not the MILP), it
    /// simply keeps its incoming boxes.
    pub infeasible: bool,
    /// Fixpoint rounds actually run.
    pub rounds: usize,
    /// How many target columns had their box shrink at least once.
    pub tightened: usize,
}

/// Run OBBT over the CONTINUOUS LP relaxation of `problem` (integer/ReLU-binary
/// columns relaxed to their continuous box), tightening `targets` as ONE coupled
/// set on a single warm [`ay_milp::LpSession`]. A proven bound on one column
/// tightens a coupled sibling — the full-LP coupling a per-neuron pass cannot
/// reach — and each committed bound persists in the session, so later chunks see
/// the earlier ones' tightenings.
///
/// DEADLINE-BOUNDED: the absolute deadline is installed on the persistent
/// session and also checked between `chunk`-sized OBBT batches. Whatever was
/// tightened before expiry is read back from the session model, so a deadline
/// hit yields a valid partial result rather than granting a fresh relative
/// slice to every target.
///
/// # Soundness
/// The LP relaxation's feasible set CONTAINS the MILP's (relaxing integrality
/// only adds points), and `obbt` commits only RIGOROUS (dual, outward-rounded)
/// bounds, so every returned `(lb, ub)` is a valid OUTER bound on the column's
/// reachable value — never too tight, never a wrong verdict. Fail-closed: any
/// solver error propagates as `Err` and an infeasible relaxation returns
/// `infeasible=true`; either way the caller keeps its original sound boxes.
///
/// # Errors
/// Propagates lowering / session / solve errors from the ay backend.
pub fn obbt_relaxation_bounds(
    problem: &MilpProblem,
    targets: &[Col],
    max_rounds: usize,
    per_solve_time_limit: std::time::Duration,
    deadline: std::time::Instant,
    chunk: usize,
) -> Result<RelaxationObbt> {
    let model = crate::ay_lib::to_ay_model_relaxed(problem)?;
    let ay_targets: Vec<ay_milp::Col> = targets
        .iter()
        .map(|c| {
            model.col_at(c.0).ok_or_else(|| {
                MipError::Encoding(format!("OBBT target column {} out of range", c.0))
            })
        })
        .collect::<Result<_>>()?;
    let opts = ay_milp::SolveOpts::new()
        .with_time_limit(per_solve_time_limit)
        .with_deadline(deadline);
    let mut session =
        ay_milp::LpSession::new(&model, &opts).map_err(|e| MipError::Solver(e.to_string()))?;
    let obbt_opts = ay_milp::ObbtOpts {
        max_rounds: max_rounds.max(1),
        ..ay_milp::ObbtOpts::default()
    };
    let batch = chunk.max(1);
    let mut infeasible = false;
    let mut rounds = 0usize;
    let mut tightened = 0usize;
    for group in ay_targets.chunks(batch) {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let report = session
            .obbt(group, &obbt_opts)
            .map_err(|e| MipError::Solver(e.to_string()))?;
        rounds = rounds.max(report.rounds);
        tightened += report.tightened;
        if report.infeasible {
            infeasible = true;
            break;
        }
    }
    // Read final bounds for EVERY target from the persistent session model, so
    // partial (deadline-cut) runs still return each column's current best box.
    let bounds = ay_targets.iter().map(|&c| session.col_bounds(c)).collect();
    Ok(RelaxationObbt {
        bounds,
        infeasible,
        rounds,
        tightened,
    })
}

#[cfg(test)]
mod tests;
