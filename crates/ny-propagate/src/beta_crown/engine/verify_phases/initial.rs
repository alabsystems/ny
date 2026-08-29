// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::result::{
    record_global_timeout_deadline_source, BabVerificationStatus, BetaCrownResult,
};
use crate::Network;

use super::super::cut_gate::CutGateState;
use super::super::tensor_ext::BoundedTensorExt;
use super::super::BetaCrownVerifier;

pub(in crate::beta_crown::engine) enum InitialPhaseOutcome {
    Early(BetaCrownResult),
    Proceed {
        initial_bounds: BoundedTensor,
        layer_bounds: Vec<BoundedTensor>,
        base_layer_bounds: Vec<Arc<BoundedTensor>>,
        bab_timeout: Duration,
        pgd_deadline: Option<Instant>,
    },
}

impl BetaCrownVerifier {
    // The initial phase consumes distinct timing, cut-gate, and verification
    // inputs assembled by the caller from separate engine resources.
    #[allow(clippy::too_many_arguments)]
    /// `overall_deadline`: If `Some`, derive phase budgets from remaining
    /// wall-clock time instead of `self.config.timeout` (#4321).
    pub(in crate::beta_crown::engine) fn evaluate_initial_phase(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        start_time: Instant,
        cut_gate: &CutGateState,
        cut_pool: &mut CutPool,
        overall_deadline: Option<Instant>,
    ) -> Result<InitialPhaseOutcome> {
        let crown_deadline = match overall_deadline {
            Some(deadline) => Some(deadline),
            None => Some(start_time.checked_add(self.config.timeout).ok_or_else(|| {
                NyError::InvalidConfig(format!(
                    "effective timeout {:?} is too large for the platform monotonic clock",
                    self.config.timeout
                ))
            })?),
        };
        let effective_total = crown_deadline
            .expect("the engine always has a finite configured or caller deadline")
            .saturating_duration_since(start_time);
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        let bab_timeout = effective_total.mul_f32(1.0 - pgd_frac);
        let bab_deadline = start_time.checked_add(bab_timeout).ok_or_else(|| {
            NyError::InvalidConfig(
                "BaB timeout is too large for the platform monotonic clock".to_string(),
            )
        })?;
        let pgd_deadline = crown_deadline;
        let initial_deadline = {
            let frac = self
                .config
                .phase_budget
                .initial_bounds_fraction
                .clamp(0.0, 1.0);
            Some(
                start_time
                    .checked_add(bab_timeout.mul_f32(frac))
                    .ok_or_else(|| {
                        NyError::InvalidConfig(
                            "initial-bounds timeout is too large for the platform monotonic clock"
                                .to_string(),
                        )
                    })?,
            )
        };

        let initial_computation = match self.compute_initial_bounds_and_layer_bounds_engine(
            network,
            input,
            Some((threshold, self.config.verify_upper_bound)),
            engine,
            initial_deadline,
        ) {
            Ok(computation) => computation,
            Err(NyError::DeadlineExceeded(_)) => {
                // A finite bound pass has no authority to start a fresh IBP
                // sweep after expiry merely to populate `output_bounds`.
                // Deadline exhaustion is nevertheless a normal verifier
                // outcome, so translate the typed propagation refusal at this
                // public BaB boundary into a sound, allocation-free Timeout.
                return Ok(InitialPhaseOutcome::Early(BetaCrownResult {
                    result: BabVerificationStatus::Timeout,
                    domains_explored: 0,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: None,
                    cuts_generated: 0,
                    domains_verified: 0,
                }));
            }
            Err(error) => return Err(error),
        };
        let initial_bounds = initial_computation.output_bounds;
        let initial_lower = initial_bounds.lower_scalar();
        let initial_upper = initial_bounds.upper_scalar();

        info!(
            "β-CROWN initial bounds: [{}, {}], threshold: {}, verify: {}",
            initial_lower,
            initial_upper,
            threshold,
            self.config.verification_direction_str()
        );

        if self
            .config
            .domain_is_verified(initial_lower, initial_upper, threshold)
        {
            return Ok(InitialPhaseOutcome::Early(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 1,
                time_elapsed: start_time.elapsed(),
                max_depth_reached: 0,
                output_bounds: Some(initial_bounds),
                cuts_generated: 0,
                domains_verified: 1,
            }));
        }
        if self
            .config
            .domain_is_violation(initial_lower, initial_upper, threshold)
        {
            return Ok(InitialPhaseOutcome::Early(BetaCrownResult {
                result: BabVerificationStatus::potential_violation(),
                domains_explored: 1,
                time_elapsed: start_time.elapsed(),
                max_depth_reached: 0,
                output_bounds: Some(initial_bounds),
                cuts_generated: 0,
                domains_verified: 0,
            }));
        }

        let bab_prof =
            ny_levers::read_raw(&ny_levers::decls::bab_diagnostics::ACASXU_PROF).is_some();
        let layer_bounds = if let Some(layer_bounds) = initial_computation.root_layer_bounds {
            if bab_prof {
                eprintln!("[NY_ACASXU_PROF] root layer_bounds arm: root_layer_bounds (from initial computation)");
            }
            layer_bounds
        } else if self.config.use_crown_ibp {
            if bab_prof {
                eprintln!(
                    "[NY_ACASXU_PROF] root layer_bounds arm: sequential CROWN-IBP collection"
                );
            }
            self.collect_sequential_crown_ibp_bounds_with_status(
                network,
                input,
                None,
                engine,
                initial_deadline,
            )?
            .bounds
        } else {
            if bab_prof {
                eprintln!("[NY_ACASXU_PROF] root layer_bounds arm: RAW IBP");
            }
            network.collect_ibp_bounds_with_deadline(input, initial_deadline)?
        };
        if bab_prof {
            let total_width: f64 = layer_bounds
                .iter()
                .map(|b| {
                    let f = b.flatten();
                    (0..f.len())
                        .map(|i| (f.upper()[[i]] - f.lower()[[i]]) as f64)
                        .sum::<f64>()
                })
                .sum();
            eprintln!("[NY_ACASXU_PROF] root layer_bounds total width: {total_width:.1}");
        }
        let base_layer_bounds: Vec<Arc<BoundedTensor>> =
            layer_bounds.iter().cloned().map(Arc::new).collect();

        if self.config.enable_proactive_cuts && self.config.enable_cuts && !cut_gate.is_cold_start()
        {
            let proactive_count = cut_pool.generate_proactive_cuts(
                network,
                &layer_bounds,
                self.config.max_proactive_cuts,
            )?;
            if proactive_count > 0 {
                info!(
                    "Generated {} proactive cuts for sequential network",
                    proactive_count
                );
            }
        } else if self.config.enable_proactive_cuts && cut_gate.is_cold_start() {
            info!("Skipping proactive cuts during BICCOS cold-start gating");
        }

        if start_time.elapsed() > bab_timeout {
            let timeout_result = BetaCrownResult {
                result: BabVerificationStatus::Timeout,
                domains_explored: 0,
                time_elapsed: start_time.elapsed(),
                max_depth_reached: 0,
                output_bounds: Some(initial_bounds),
                cuts_generated: 0,
                domains_verified: 0,
            };
            record_global_timeout_deadline_source(Some(bab_deadline));
            let result = self.try_pgd_attack_with_deadline(
                network,
                input,
                threshold,
                timeout_result,
                pgd_deadline,
            )?;
            return Ok(InitialPhaseOutcome::Early(result));
        }

        Ok(InitialPhaseOutcome::Proceed {
            initial_bounds,
            layer_bounds,
            base_layer_bounds,
            bab_timeout,
            pgd_deadline,
        })
    }
}
