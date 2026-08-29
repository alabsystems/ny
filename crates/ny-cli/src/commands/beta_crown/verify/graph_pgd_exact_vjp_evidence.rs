// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Versioned execution evidence for the governed batched exact-VJP attacks.
//!
//! This is deliberately per invocation. It carries only the frozen typed
//! [`ny_propagate::ExactVjpPgdConfig`] and monotone counters owned by that
//! invocation; no receipt field is reconstructed from process-global state.

use std::cell::Cell;

use ny_propagate::ExactVjpPgdConfig;

use crate::flight::FlightStatus;

const METHOD: &str = "exact_vjp_execution";
const SCHEMA: &str = "ny-cli/exact-vjp-execution/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExactVjpRoute {
    Conjunctive,
    Disjunctive,
    ConjunctiveGeneric,
}

impl ExactVjpRoute {
    const fn token(self) -> &'static str {
        match self {
            Self::Conjunctive => "conjunctive",
            Self::Disjunctive => "disjunctive",
            Self::ConjunctiveGeneric => "conjunctive_generic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExactVjpOutcome {
    Candidate,
    Exhausted,
    Deadline,
    Stalled,
    BatchDisabled,
    GenericCompleted,
    FallbackSequential,
    NotStarted,
    InternalError,
}

impl ExactVjpOutcome {
    const fn token(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Exhausted => "exhausted",
            Self::Deadline => "deadline",
            Self::Stalled => "stalled",
            Self::BatchDisabled => "batch_disabled",
            Self::GenericCompleted => "generic_completed",
            Self::FallbackSequential => "fallback_sequential",
            Self::NotStarted => "not_started",
            Self::InternalError => "internal_error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExactVjpInactiveReason {
    None,
    DeadlineBeforeStart,
    BatchDisabled,
    GpuUnavailable,
    EmptyObjective,
    UnsupportedPlan,
    ForwardFailure,
    VjpFailure,
    GradientCountMismatch,
    NoAttackTarget,
    StructuralIneligible,
    InsufficientRestarts,
    ExperimentalLaneActive,
    GenericFallback,
    InternalError,
}

impl ExactVjpInactiveReason {
    const fn token(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DeadlineBeforeStart => "deadline_before_start",
            Self::BatchDisabled => "batch_disabled",
            Self::GpuUnavailable => "gpu_unavailable",
            Self::EmptyObjective => "empty_objective",
            Self::UnsupportedPlan => "unsupported_plan",
            Self::ForwardFailure => "forward_failure",
            Self::VjpFailure => "vjp_failure",
            Self::GradientCountMismatch => "gradient_count_mismatch",
            Self::NoAttackTarget => "no_attack_target",
            Self::StructuralIneligible => "structural_ineligible",
            Self::InsufficientRestarts => "insufficient_restarts",
            Self::ExperimentalLaneActive => "experimental_lane_active",
            Self::GenericFallback => "generic_fallback",
            Self::InternalError => "internal_error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WidthBinding {
    NotRun,
    Requested,
    RestartCap,
    AdapterHalved,
}

impl WidthBinding {
    const fn token(self) -> &'static str {
        match self {
            Self::NotRun => "not_run",
            Self::Requested => "requested",
            Self::RestartCap => "restart_cap",
            Self::AdapterHalved => "adapter_halved",
        }
    }
}

/// Per-invocation guard. A terminal record is emitted by `Drop`, including on
/// every `?` propagation path. Explicit `finish` calls replace the conservative
/// internal-error sentinel once the caller knows the terminal disposition.
pub(super) struct ExactVjpExecution {
    route: ExactVjpRoute,
    policy: ExactVjpPgdConfig,
    width_default: usize,
    width_initial: usize,
    total_restarts: usize,
    wave_steps_default: usize,
    wave_steps_effective: usize,
    work_started: Cell<bool>,
    width_final: Cell<usize>,
    width_max_active: Cell<usize>,
    width_binding: Cell<WidthBinding>,
    width_retries: Cell<usize>,
    restarts_started: Cell<usize>,
    waves: Cell<usize>,
    vjp_steps: Cell<usize>,
    odi_vjp_steps: Cell<usize>,
    odi_completed_phase_count: Cell<usize>,
    odi_move_count: Cell<usize>,
    legacy_lanes_used: Cell<usize>,
    recycle_count: Cell<usize>,
    exploit_opportunity_count: Cell<usize>,
    exploit_count: Cell<usize>,
    hinge_extra_count: Cell<usize>,
    first_vjp_reported: Cell<bool>,
    second_wave_reported: Cell<bool>,
    recycle_reported: Cell<bool>,
    exploit_reported: Cell<bool>,
    exploit_opportunity_reported: Cell<bool>,
    hinge_reported: Cell<bool>,
    terminal: Cell<Option<(ExactVjpOutcome, ExactVjpInactiveReason, FlightStatus)>>,
}

impl ExactVjpExecution {
    pub(super) fn conjunctive(
        policy: ExactVjpPgdConfig,
        total_restarts: usize,
        pgd_steps: usize,
    ) -> Self {
        Self::new(
            ExactVjpRoute::Conjunctive,
            policy,
            64,
            total_restarts,
            pgd_steps,
            policy.effective_wave_steps(pgd_steps),
        )
    }

    pub(super) fn disjunctive(
        policy: ExactVjpPgdConfig,
        total_restarts: usize,
        pgd_steps: usize,
    ) -> Self {
        Self::new(
            ExactVjpRoute::Disjunctive,
            policy,
            32,
            total_restarts,
            pgd_steps,
            pgd_steps,
        )
    }

    pub(super) fn conjunctive_generic(
        policy: ExactVjpPgdConfig,
        total_restarts: usize,
        pgd_steps: usize,
    ) -> Self {
        Self::new(
            ExactVjpRoute::ConjunctiveGeneric,
            policy,
            64,
            total_restarts,
            pgd_steps,
            pgd_steps,
        )
    }

    fn new(
        route: ExactVjpRoute,
        policy: ExactVjpPgdConfig,
        width_default: usize,
        total_restarts: usize,
        wave_steps_default: usize,
        wave_steps_effective: usize,
    ) -> Self {
        let width_initial = policy.batch_width.unwrap_or(width_default);
        Self {
            route,
            policy,
            width_default,
            width_initial,
            total_restarts,
            wave_steps_default,
            wave_steps_effective,
            work_started: Cell::new(false),
            width_final: Cell::new(width_initial),
            width_max_active: Cell::new(0),
            width_binding: Cell::new(WidthBinding::NotRun),
            width_retries: Cell::new(0),
            restarts_started: Cell::new(0),
            waves: Cell::new(0),
            vjp_steps: Cell::new(0),
            odi_vjp_steps: Cell::new(0),
            odi_completed_phase_count: Cell::new(0),
            odi_move_count: Cell::new(0),
            legacy_lanes_used: Cell::new(0),
            recycle_count: Cell::new(0),
            exploit_opportunity_count: Cell::new(0),
            exploit_count: Cell::new(0),
            hinge_extra_count: Cell::new(0),
            first_vjp_reported: Cell::new(false),
            second_wave_reported: Cell::new(false),
            recycle_reported: Cell::new(false),
            exploit_reported: Cell::new(false),
            exploit_opportunity_reported: Cell::new(false),
            hinge_reported: Cell::new(false),
            terminal: Cell::new(None),
        }
    }

    pub(super) fn mark_work_started(&self) {
        self.work_started.set(true);
    }

    pub(super) fn start_wave(&self, restart_base: usize, active_width: usize, legacy_lanes: usize) {
        self.work_started.set(true);
        // Unique restart indices admitted so adapter retries cannot make this
        // exceed `total_restarts`. `waves` separately counts wave attempts.
        self.restarts_started.set(
            self.restarts_started.get().max(
                restart_base
                    .saturating_add(active_width)
                    .min(self.total_restarts),
            ),
        );
        self.waves.set(self.waves.get().saturating_add(1));
        self.width_final.set(active_width);
        self.width_max_active
            .set(self.width_max_active.get().max(active_width));
        if self.width_binding.get() != WidthBinding::AdapterHalved {
            self.width_binding
                .set(if active_width < self.width_initial {
                    WidthBinding::RestartCap
                } else {
                    WidthBinding::Requested
                });
        }
        self.legacy_lanes_used
            .set(self.legacy_lanes_used.get().saturating_add(legacy_lanes));
        if self.waves.get() >= 2 && !self.second_wave_reported.replace(true) {
            self.emit_progress();
        }
    }

    pub(super) fn adapter_retry(&self, retry_width: usize) {
        self.work_started.set(true);
        self.width_final.set(retry_width);
        self.width_binding.set(WidthBinding::AdapterHalved);
        self.width_retries
            .set(self.width_retries.get().saturating_add(1));
        self.emit_progress();
    }

    pub(super) fn successful_vjp(&self, odi: bool) {
        self.work_started.set(true);
        if odi {
            self.odi_vjp_steps
                .set(self.odi_vjp_steps.get().saturating_add(1));
        } else {
            self.vjp_steps.set(self.vjp_steps.get().saturating_add(1));
        }
        if !self.first_vjp_reported.replace(true) {
            self.emit_progress();
        }
    }

    /// Record coordinates whose stored f32 bits changed after an ODI
    /// eta-scaled update and box projection. The consumer calls this only after
    /// every lane update for the successful VJP step has completed.
    pub(super) fn odi_moved(&self, count: usize) {
        self.odi_move_count
            .set(self.odi_move_count.get().saturating_add(count));
    }

    /// Record one wave whose complete configured ODI phase finished. Partial
    /// phases never reach this method, even when their successful VJP steps
    /// remain visible in the cumulative `odi_vjp_steps` counter.
    pub(super) fn odi_completed_phase(&self) {
        self.odi_completed_phase_count
            .set(self.odi_completed_phase_count.get().saturating_add(1));
    }

    pub(super) fn recycled(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.recycle_count
            .set(self.recycle_count.get().saturating_add(count));
        if !self.recycle_reported.replace(true) {
            self.emit_progress();
        }
    }

    pub(super) fn exploited(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.exploit_count
            .set(self.exploit_count.get().saturating_add(count));
        if !self.exploit_reported.replace(true) {
            self.emit_progress();
        }
    }

    pub(super) fn exploit_opportunity(&self) {
        self.exploit_opportunity_count
            .set(self.exploit_opportunity_count.get().saturating_add(1));
        if !self.exploit_opportunity_reported.replace(true) {
            self.emit_progress();
        }
    }

    pub(super) fn hinge_extra(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.hinge_extra_count
            .set(self.hinge_extra_count.get().saturating_add(count));
        if !self.hinge_reported.replace(true) {
            self.emit_progress();
        }
    }

    pub(super) fn finish(&self, outcome: ExactVjpOutcome, inactive_reason: ExactVjpInactiveReason) {
        let status = terminal_status(outcome, inactive_reason, self.work_started.get());
        self.terminal.set(Some((outcome, inactive_reason, status)));
    }

    fn emit_progress(&self) {
        crate::flight::note(
            METHOD,
            FlightStatus::Ran,
            Some(self.render(
                "progress",
                ExactVjpOutcome::InternalError,
                ExactVjpInactiveReason::None,
            )),
        );
    }

    fn render(
        &self,
        phase: &str,
        outcome: ExactVjpOutcome,
        inactive_reason: ExactVjpInactiveReason,
    ) -> String {
        let configured_width = self
            .policy
            .batch_width
            .map_or_else(|| "none".to_string(), |width| width.to_string());
        let configured_wave_steps = self
            .policy
            .wave_steps
            .map_or_else(|| "none".to_string(), |steps| steps.to_string());
        let progress_outcome = if phase == "progress" {
            "in_progress"
        } else {
            outcome.token()
        };
        format!(
            "schema={SCHEMA} phase={phase} route={} outcome={progress_outcome} inactive_reason={} prefer_exact_batched={} batch_enabled={} work_started={} width_source={} width_configured={configured_width} width_default={} width_initial={} width_final={} width_max_active={} width_binding={} width_retries={} total_restarts={} restarts_started={} waves={} wave_steps_configured={configured_wave_steps} wave_steps_default={} wave_steps_effective={} vjp_steps={} odi_steps_configured={} odi_eta={} odi_vjp_steps={} odi_completed_phase_count={} odi_move_count={} legacy_lanes_configured={} legacy_lanes_used={} lane_window={} recycle_count={} exploit_enabled={} exploit_opportunity_count={} exploit_count={} hinge_tau={} hinge_extra_count={}",
            self.route.token(),
            inactive_reason.token(),
            self.policy.prefer_exact_batched,
            self.policy.batch_enabled,
            self.work_started.get(),
            if self.policy.batch_width.is_some() {
                "configured"
            } else {
                "contextual"
            },
            self.width_default,
            self.width_initial,
            self.width_final.get(),
            self.width_max_active.get(),
            self.width_binding.get().token(),
            self.width_retries.get(),
            self.total_restarts,
            self.restarts_started.get(),
            self.waves.get(),
            self.wave_steps_default,
            self.wave_steps_effective,
            self.vjp_steps.get(),
            self.policy.odi_steps,
            self.policy.odi_eta,
            self.odi_vjp_steps.get(),
            self.odi_completed_phase_count.get(),
            self.odi_move_count.get(),
            self.policy.legacy_lanes,
            self.legacy_lanes_used.get(),
            self.policy.lane_window,
            self.recycle_count.get(),
            self.policy.exploit,
            self.exploit_opportunity_count.get(),
            self.exploit_count.get(),
            self.policy.hinge_tau,
            self.hinge_extra_count.get(),
        )
    }
}

impl Drop for ExactVjpExecution {
    fn drop(&mut self) {
        let (outcome, inactive_reason, status) = self.terminal.get().unwrap_or_else(|| {
            (
                ExactVjpOutcome::InternalError,
                ExactVjpInactiveReason::InternalError,
                if self.work_started.get() {
                    FlightStatus::Ran
                } else {
                    FlightStatus::Skipped
                },
            )
        });
        crate::flight::note(
            METHOD,
            status,
            Some(self.render("terminal", outcome, inactive_reason)),
        );
    }
}

fn terminal_status(
    outcome: ExactVjpOutcome,
    inactive_reason: ExactVjpInactiveReason,
    work_started: bool,
) -> FlightStatus {
    use ExactVjpInactiveReason as Reason;
    use ExactVjpOutcome as Outcome;

    match (outcome, inactive_reason) {
        (
            Outcome::Candidate
            | Outcome::Exhausted
            | Outcome::Deadline
            | Outcome::Stalled
            | Outcome::GenericCompleted,
            Reason::None,
        )
        | (Outcome::BatchDisabled, Reason::BatchDisabled) => FlightStatus::Ran,
        (Outcome::FallbackSequential, _) | (Outcome::InternalError, _) if work_started => {
            FlightStatus::Ran
        }
        (Outcome::NotStarted, _)
        | (Outcome::FallbackSequential, _)
        | (Outcome::InternalError, _) => FlightStatus::Skipped,
        _ => FlightStatus::Skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_reason_is_complete_and_counters_are_monotone() {
        let policy = ExactVjpPgdConfig {
            batch_width: Some(52),
            legacy_lanes: 20,
            odi_steps: 4,
            odi_eta: 0.5,
            wave_steps: Some(250),
            lane_window: 50,
            exploit: true,
            hinge_tau: 0.001,
            ..ExactVjpPgdConfig::default()
        };
        let evidence = ExactVjpExecution::conjunctive(policy, 100, 1_000);
        evidence.start_wave(0, 52, 20);
        evidence.odi_moved(11);
        for _ in 0..policy.odi_steps {
            evidence.successful_vjp(true);
        }
        evidence.odi_completed_phase();
        evidence.successful_vjp(false);
        evidence.recycled(3);
        evidence.exploit_opportunity();
        evidence.exploited(2);
        evidence.hinge_extra(7);
        evidence.adapter_retry(26);
        evidence.start_wave(52, 26, 0);
        let reason = evidence.render(
            "terminal",
            ExactVjpOutcome::Exhausted,
            ExactVjpInactiveReason::None,
        );
        for token in [
            "schema=ny-cli/exact-vjp-execution/v1",
            "phase=terminal",
            "route=conjunctive",
            "outcome=exhausted",
            "inactive_reason=none",
            "work_started=true",
            "width_source=configured",
            "width_configured=52",
            "width_initial=52",
            "width_final=26",
            "width_max_active=52",
            "width_binding=adapter_halved",
            "width_retries=1",
            "total_restarts=100",
            "restarts_started=78",
            "waves=2",
            "wave_steps_configured=250",
            "wave_steps_default=1000",
            "wave_steps_effective=250",
            "vjp_steps=1",
            "odi_steps_configured=4",
            "odi_eta=0.5",
            "odi_vjp_steps=4",
            "odi_completed_phase_count=1",
            "odi_move_count=11",
            "legacy_lanes_configured=20",
            "legacy_lanes_used=20",
            "lane_window=50",
            "recycle_count=3",
            "exploit_enabled=true",
            "exploit_opportunity_count=1",
            "exploit_count=2",
            "hinge_tau=0.001",
            "hinge_extra_count=7",
        ] {
            assert!(reason.contains(token), "missing {token:?} from {reason}");
        }
        evidence.finish(ExactVjpOutcome::Exhausted, ExactVjpInactiveReason::None);
    }

    #[test]
    fn terminal_status_follows_the_v1_phase_contract() {
        assert_eq!(
            terminal_status(
                ExactVjpOutcome::BatchDisabled,
                ExactVjpInactiveReason::BatchDisabled,
                false
            ),
            FlightStatus::Ran
        );
        assert_eq!(
            terminal_status(
                ExactVjpOutcome::NotStarted,
                ExactVjpInactiveReason::GpuUnavailable,
                false
            ),
            FlightStatus::Skipped
        );
        assert_eq!(
            terminal_status(
                ExactVjpOutcome::FallbackSequential,
                ExactVjpInactiveReason::VjpFailure,
                true
            ),
            FlightStatus::Ran
        );
    }
}
