// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD initialization strategies: Uniform and OSI (#1449).
//!
//! OSI (Output Specification Initialization) pushes each restart seed toward a
//! diverse output-space region before the real PGD attack loop starts.
//!
//! Reference: alpha-beta-CROWN `OSI_init_C` in `attack_utils.py:328-362`.
//! ny uses SPSA gradient estimation (2 evals per step) instead of
//! autograd, so each OSI step costs 2 forward passes.

use ndarray::{ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::RngExt;

use crate::pgd_attack::config::PgdInitialization;
use crate::pgd_attack::PgdStepState;
use crate::Network;

use super::PgdAttacker;

impl PgdAttacker<'_> {
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
            PgdInitialization::Uniform => Ok(self.sample_uniform(input_bounds, rng)),
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
        let mut x = self.sample_uniform(input_bounds, rng);

        // Deadline guard: OSI costs 1 + 2*osi_steps forward passes per restart,
        // and the batched attack initializes EVERY restart before its step loop
        // runs (which is where the next deadline check lives). Without this
        // check a short attack budget can be overshot by minutes of pure
        // initialization. Degrading to the plain uniform seed is safe: PGD is a
        // falsification heuristic, so a weaker seed can only miss
        // counterexamples, never produce a wrong verdict.
        if self.config.past_deadline() {
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
            if self.config.past_deadline() {
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
