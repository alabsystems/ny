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
    record_global_timeout_deadline_source, record_sequential_initial_schedule_receipt,
    BabVerificationStatus, BetaCrownResult, SequentialEarlyQuitDisposition,
    SequentialInitialScheduleReceipt,
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
    /// The carrier owns its numerical Source, complete original objective and
    /// exact evaluation root. This helper admits only the actual singleton
    /// lower input-split network; unsupported transformations stay independent.
    /// There is no per-node advice call or new time grant.
    pub(in crate::beta_crown::engine) fn input_head_support(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        deadline: Option<Instant>,
    ) -> Option<f32> {
        if self.config.verify_upper_bound
            || threshold != 0.0
            || !matches!(
                self.config.branching_heuristic,
                crate::beta_crown::branching::BranchingHeuristic::InputSplit
            )
        {
            return None;
        }
        let deadline = deadline?;
        let source = self.input_head_continuation()?;
        if !source.matches_scalar_network(network)
            || Instant::now() >= deadline
            || ny_core::require_f64_interval_proof_environment().is_err()
        {
            return None;
        }
        let lower = source.try_support_box(
            input.lower().as_slice()?,
            input.upper().as_slice()?,
            deadline,
        )?;
        (lower.is_finite()
            && Instant::now() < deadline
            && ny_core::require_f64_interval_proof_environment().is_ok())
        .then_some(lower)
    }

    pub(in crate::beta_crown::engine) fn mark_input_head_dependency(&self) {
        if let Some(source) = self.input_head_continuation() {
            source.mark_dependent_use();
        }
    }

    /// Classify ONLY a completed late scalar MAX at the two input-split
    /// sites below. On the authenticated singleton lower/zero route, a finite
    /// nonpositive floor cannot satisfy the actual strict closure predicate.
    /// It never enters generic layer bounds, planes, constraints or child
    /// CROWN recomputation. Its remaining consumers only rank/schedule work.
    /// Positive pruning and any other route keep the conservative sticky flag.
    /// This does not clear earlier dependence or qualify a future cache path.
    pub(in crate::beta_crown::engine) fn mark_input_head_scalar_max_dependency(
        &self,
        network: &Network,
        threshold: f32,
        consumed_lower: f32,
        scalar_count: usize,
        deadline: Option<Instant>,
    ) {
        let neutral = !self.config.verify_upper_bound
            && threshold == 0.0
            && consumed_lower.is_finite()
            && consumed_lower <= 0.0
            && scalar_count == 1
            && !self
                .config
                .domain_is_verified(consumed_lower, consumed_lower, threshold)
            && matches!(
                self.config.branching_heuristic,
                crate::beta_crown::branching::BranchingHeuristic::InputSplit
            )
            && self.input_head_continuation().is_some_and(|source| {
                source.matches_scalar_network(network)
                    && deadline.is_some_and(|end| source.is_live(end))
            });
        if !neutral {
            self.mark_input_head_dependency();
        }
    }

    pub(in crate::beta_crown::engine) fn input_head_live(&self, deadline: Option<Instant>) -> bool {
        match (self.input_head_continuation(), deadline) {
            (Some(source), Some(deadline)) => source.is_live(deadline),
            _ => false,
        }
    }
    // The initial phase consumes distinct timing, cut-gate, and verification
    // inputs assembled by the caller from separate engine resources.
    #[allow(clippy::too_many_arguments)]
    /// `overall_deadline`: If `Some`, derive phase budgets from remaining
    /// wall-clock time instead of `self.config.timeout`.
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
        // #phase-cost: see `ny_core::phase_cost` (observability only).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::RootBounds);
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
        // #seq-single-reserve: one derivation for the whole phase schedule,
        // shared verbatim with the tests that pin it. An explicit
        // `overall_deadline` marks `effective_total` as the caller
        // ledger's ALREADY-RESERVED BaB boundary, so the warmup cap derives
        // from that boundary instead of re-subtracting the post-BaB PGD
        // reserve (the graph lanes' "do not reserve post-BaB time a second
        // time" treatment). `bab_timeout` — the BaB search boundary —
        // deliberately keeps the engine's own single reserve in EVERY
        // disposition: the window `[bab_timeout, pgd_deadline]` is the
        // engine-internal PGD fallback's only home (see `engine/pgd.rs`).
        let schedule = self
            .config
            .phase_budget
            .sequential_initial_schedule(effective_total, overall_deadline.is_some());
        let bab_timeout = schedule.bab_timeout;
        let bab_deadline = start_time.checked_add(bab_timeout).ok_or_else(|| {
            NyError::InvalidConfig(
                "BaB timeout is too large for the platform monotonic clock".to_string(),
            )
        })?;
        let pgd_deadline = crown_deadline;
        let initial_deadline =
            Some(start_time.checked_add(schedule.warmup_cap).ok_or_else(|| {
                NyError::InvalidConfig(
                    "initial-bounds timeout is too large for the platform monotonic clock"
                        .to_string(),
                )
            })?);
        // The lever's own observable: the CLI takes this receipt and emits a
        // `seq_initial_schedule` flight note in both dispositions. The
        // `info!` line below is dark under `RUST_LOG=error`, so the receipt
        // is the load-bearing channel.
        let schedule_receipt = SequentialInitialScheduleReceipt {
            effective_total_secs: effective_total.as_secs_f64(),
            bab_timeout_secs: bab_timeout.as_secs_f64(),
            warmup_cap_secs: schedule.warmup_cap.as_secs_f64(),
            warmup_single_reserve_applied: schedule.warmup_single_reserve_applied,
            caller_deadline: overall_deadline.is_some(),
            early_quit: SequentialEarlyQuitDisposition::None,
        };
        record_sequential_initial_schedule_receipt(schedule_receipt);
        info!(
            "[seq-budget] effective_total={:.3}s bab_timeout={:.3}s warmup_cap={:.3}s \
             warmup_single_reserve_applied={} (#seq-single-reserve: post-BaB reserve \
             not double-taken into the warmup cap)",
            schedule_receipt.effective_total_secs,
            schedule_receipt.bab_timeout_secs,
            schedule_receipt.warmup_cap_secs,
            schedule_receipt.warmup_single_reserve_applied,
        );

        let source_floor = self.input_head_support(network, input, threshold, initial_deadline);
        if source_floor.is_some_and(|lower| lower > 0.0) && self.input_head_live(initial_deadline) {
            self.mark_input_head_dependency();
            return Ok(InitialPhaseOutcome::Early(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 1,
                time_elapsed: start_time.elapsed(),
                max_depth_reached: 0,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 1,
            }));
        }
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
                let timeout_result = BetaCrownResult {
                    result: BabVerificationStatus::Timeout,
                    domains_explored: 0,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: None,
                    cuts_generated: 0,
                    domains_verified: 0,
                };
                // #seq-early-attack: this Early return previously escaped the
                // engine BEFORE `core.rs` constructed its `pgd_fallback`
                // closure, forfeiting the post-BaB attack lane on exactly the
                // rows that had already lost their bounds budget. Feed the
                // SAME result through the SAME fallback the in-phase timeout
                // path below uses. `try_pgd_attack_with_deadline` returns the
                // original Timeout unless an independently re-validated
                // counterexample is found, so a row the attack cannot falsify
                // is byte-identical. No global timeout source is recorded
                // here: the expired deadline is the PHASE-LOCAL warmup cap,
                // which has no authority over the global deadline receipt.
                let (result, disposition) = if self.config.phase_budget.sequential_early_quit_attack
                {
                    (
                        self.try_pgd_attack_with_deadline(
                            network,
                            input,
                            threshold,
                            timeout_result,
                            pgd_deadline,
                        )?,
                        SequentialEarlyQuitDisposition::Attacked,
                    )
                } else {
                    (
                        timeout_result,
                        SequentialEarlyQuitDisposition::ForfeitedByKillSwitch,
                    )
                };
                record_sequential_initial_schedule_receipt(SequentialInitialScheduleReceipt {
                    early_quit: disposition,
                    ..schedule_receipt
                });
                return Ok(InitialPhaseOutcome::Early(result));
            }
            Err(error) => return Err(error),
        };
        let mut initial_bounds = initial_computation.output_bounds;
        let mut initial_lower = initial_bounds.lower_scalar();
        let initial_upper = initial_bounds.upper_scalar();
        if let Some(lower) = source_floor {
            if initial_bounds.lower().len() == 1
                && lower > initial_lower
                && lower <= initial_upper
                && self.input_head_live(initial_deadline)
            {
                initial_bounds = BoundedTensor::new(
                    initial_bounds.lower().mapv(|_| lower),
                    initial_bounds.upper().clone(),
                )?;
                initial_lower = lower;
                self.mark_input_head_scalar_max_dependency(
                    network,
                    threshold,
                    lower,
                    initial_bounds.lower().len(),
                    initial_deadline,
                );
            }
        }

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
