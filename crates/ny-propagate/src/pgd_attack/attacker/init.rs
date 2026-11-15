// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD initialization strategies: Uniform and OSI, plus box-corner
//! seeds for the first restarts (#attack-vertex-geometry).
//!
//! OSI (Output Specification Initialization) pushes each restart seed toward a
//! diverse output-space region before the real PGD attack loop starts.
//!
//! Reference: alpha-beta-CROWN `OSI_init_C` in `attack_utils.py:328-362`.
//! ny uses SPSA gradient estimation (2 evals per step) instead of
//! autograd, so each OSI step costs 2 forward passes.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::RngExt;

use crate::pgd_attack::config::PgdInitialization;
use crate::pgd_attack::PgdStepState;
use crate::Network;

use super::PgdAttacker;

/// The box-corner seed for corner slot `slot` (#attack-vertex-geometry).
///
/// - slot 0: the lower corner `lower`;
/// - slot 1: the upper corner `upper`;
/// - slot `k >= 2`: a box vertex, dimension `d` at `lower[d]` when `coin()`
///   returns `true` and at `upper[d]` otherwise (one draw per dimension).
///
/// Every coordinate is copied from `lower` or `upper` verbatim, so the point
/// is a member of the box `[lower, upper]` exactly, with no rounding. Slots 0
/// and 1 draw nothing from `coin`.
///
/// WHY. On safenlp `ruarobot/hyperrectangle_1800` (field `sat`) the box
/// centre is a strict local minimum of the violation margin and uniform
/// interior seeds reach the counterexample 7/20, while descents from the two
/// extreme corners reach it deterministically
/// (`reports/measured-2026/safenlp_undecided_2026-08-30/NOTES.md` §5).
///
/// Attack-only: a seed is a proposal, and a candidate becomes `sat` only by
/// passing the unchanged trusted re-validation downstream.
pub fn corner_seed_point(
    slot: usize,
    lower: &[f32],
    upper: &[f32],
    mut coin: impl FnMut() -> bool,
) -> Vec<f32> {
    match slot {
        0 => lower.to_vec(),
        1 => upper.to_vec(),
        _ => lower
            .iter()
            .zip(upper)
            .map(|(&lo, &hi)| if coin() { lo } else { hi })
            .collect(),
    }
}

/// Stable receipt token for [`corner_seed_point`]'s slot.
pub fn corner_seed_kind(slot: usize) -> &'static str {
    match slot {
        0 => "lo",
        1 => "hi",
        _ => "vertex",
    }
}

/// The corner slot of the restart whose RNG seed is `seed`, or `None` when
/// that restart keeps its historical seed.
///
/// Every restart loop in this module seeds restart `k` with
/// `base_seed.wrapping_add(k)`, so `seed.wrapping_sub(base_seed)` recovers `k`
/// exactly. `corner_seeds == 0` never yields a slot.
pub(in crate::pgd_attack) fn corner_slot_for_seed(
    base_seed: u64,
    seed: u64,
    corner_seeds: usize,
) -> Option<usize> {
    let restart = seed.wrapping_sub(base_seed);
    (restart < corner_seeds as u64).then_some(restart as usize)
}

impl PgdAttacker<'_> {
    /// Initialize the restart seeded with `seed` (#attack-vertex-geometry).
    ///
    /// When `config.corner_seed_restarts > 0` and this is restart
    /// `k < corner_seed_restarts` (its seed is `config.seed + k`), it starts
    /// at `corner_seed_point(k, ..)` ([`corner_seed_point`]), drawing vertex coins from this
    /// restart's own RNG, and prints one `[pgd-init]` engagement line.
    /// Otherwise this is exactly [`Self::initialize_restart`]: with the
    /// default `corner_seed_restarts == 0` no draw, branch or line differs
    /// from the historical seeding.
    pub(in crate::pgd_attack) fn initialize_restart_for_seed(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        rng: &mut StdRng,
        seed: u64,
    ) -> Result<ArrayD<f32>> {
        self.checkpoint("before restart seed initialization")?;
        let corner_seeds = self.config.corner_seed_restarts;
        let Some(slot) = corner_slot_for_seed(self.config.seed, seed, corner_seeds) else {
            return self.initialize_restart(network, input_bounds, rng);
        };
        let lower: Vec<f32> = input_bounds.lower().iter().copied().collect();
        let upper: Vec<f32> = input_bounds.upper().iter().copied().collect();
        let point = corner_seed_point(slot, &lower, &upper, || rng.random::<bool>());
        // #attack-vertex-geometry ENGAGEMENT OBSERVABLE: one line per
        // corner-seeded restart, printed unconditionally whenever the key is
        // armed and this restart schedule is reached. Hit attribution is not
        // available at this seam (`hit=na`); the ny-cli lanes carry it.
        eprintln!(
            "[pgd-init] schema=ny-propagate/pgd-init/v1 lane=pgd_attacker corner_seeds={} \
             slot={} kind={} seeded=1 hit=na",
            corner_seeds,
            slot,
            corner_seed_kind(slot),
        );
        ArrayD::from_shape_vec(IxDyn(input_bounds.shape()), point).map_err(|error| {
            NyError::InternalError(format!(
                "corner seed shape mismatch for input bounds {:?}: {error}",
                input_bounds.shape()
            ))
        })
    }

    /// Initialize a restart point according to the configured strategy.
    ///
    /// - `Uniform`: standard random sample from `[l, u]` (existing behavior).
    /// - `Osi`: run `osi_steps` of SPSA-based signed gradient ascent on a
    ///   random output-space scalarization `<w, model(x)>` before returning
    ///   the seed. This pushes each restart toward a different output extremum.
    pub(in crate::pgd_attack) fn initialize_restart(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        rng: &mut StdRng,
    ) -> Result<ArrayD<f32>> {
        match self.config.initialization {
            PgdInitialization::Uniform => self.sample_uniform_until(input_bounds, rng),
            PgdInitialization::Osi => self.osi_init(network, input_bounds, rng),
        }
    }

    /// OSI initialization: SPSA-based signed gradient ascent on a random
    /// output-space scalarization.
    ///
    /// Reference: `OSI_init_C` in `attack_utils.py:328-362`.
    ///
    /// ```text
    /// draw w in [-1, 1]^output_dim
    /// draw x_0 uniformly from [l, u]
    /// for t in 0..osi_steps:
    ///     Delta = Rademacher({+1,-1}^n)
    ///     f_w(x) = <w, model(x)>
    ///     g_t = Delta * (f_w(x + sigma*Delta) - f_w(x - sigma*Delta)) / (2*sigma)
    ///     x_{t+1} = project(x_t + alpha * sign(g_t), [l, u])
    /// return x_osi
    /// ```
    fn osi_init(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        rng: &mut StdRng,
    ) -> Result<ArrayD<f32>> {
        let mut x = self.sample_uniform_until(input_bounds, rng)?;

        // Deadline guard: OSI costs 1 + 2*osi_steps forward passes per restart,
        // and the batched attack initializes EVERY restart before its step loop
        // runs (which is where the next deadline check lives). Without this
        // check a short attack budget can be overshot by minutes of pure
        // initialization. Degrading to the plain uniform seed is safe: PGD is a
        // falsification heuristic, so a weaker seed can only miss
        // counterexamples, never produce a wrong verdict.
        if self.past_deadline() {
            return Ok(x);
        }

        // Probe forward pass to discover output dimension.
        let probe_output = self.evaluate(network, &x)?;
        let output_dim = probe_output.len();

        // Random output-space direction w in [-1, 1]^output_dim.
        let w: Vec<f32> = (0..output_dim)
            .map(|_| rng.random_range(-1.0_f32..=1.0_f32))
            .collect();

        let sigma = self
            .config
            .suggested_spsa_delta(input_bounds)
            .max(self.config.spsa_delta);
        let mut step_state = PgdStepState::new_signed_gradient(
            self.config.alpha_mode,
            self.config.step_size,
            input_bounds,
        );

        for _step in 0..self.config.osi_steps {
            // Deadline guard (see above): stop refining this seed once the
            // attack budget is spent; the partially-diversified point is
            // still a valid in-box seed.
            if self.past_deadline() {
                break;
            }
            // Rademacher perturbation vector.
            let delta: ArrayD<f32> = ArrayD::from_shape_fn(IxDyn(x.shape()), |_| {
                if rng.random::<bool>() {
                    1.0
                } else {
                    -1.0
                }
            });

            // Perturbed points.
            let x_plus = self.project(&(&x + &(&delta * sigma)), input_bounds);
            let x_minus = self.project(&(&x - &(&delta * sigma)), input_bounds);

            // Forward pass on both perturbed points.
            let out_plus = self.evaluate(network, &x_plus)?;
            let out_minus = self.evaluate(network, &x_minus)?;

            // Scalarize: f_w(x) = <w, model(x)>.
            let f_plus = scalarize(&w, &out_plus);
            let f_minus = scalarize(&w, &out_minus);

            // NaN guard: skip this step if either forward pass produced NaN.
            if !f_plus.is_finite() || !f_minus.is_finite() {
                continue;
            }

            let diff = f_plus - f_minus;
            // SPSA gradient estimate: g = delta * diff / (2 * sigma).
            // We only need sign(g) = sign(delta * diff) = sign(delta) * sign(diff)
            // = delta * sign(diff) (since delta ∈ {±1}).
            // So: step = step_size * sign(g) = step_size * delta * sign(diff).
            let sign_diff = if diff > 0.0 {
                1.0_f32
            } else if diff < 0.0 {
                -1.0_f32
            } else {
                0.0_f32
            };
            let pseudo_gradient = &delta * sign_diff;
            x = step_state.step(&pseudo_gradient, &x, input_bounds, true);
        }

        Ok(x)
    }
}

/// Compute the dot product <w, output>, handling shape differences.
fn scalarize(w: &[f32], output: &ArrayD<f32>) -> f32 {
    output.iter().zip(w.iter()).map(|(&o, &wi)| o * wi).sum()
}
