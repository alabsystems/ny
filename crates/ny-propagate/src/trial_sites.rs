// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 — the lanes' side of the optimizer's O-TRIAL rules
//! (ROUTER_PROGRAM §5.1, §5.3, §10 R6).
//!
//! A lane that owns a trial knob asks this module for its site's pure
//! machine (`ny_core::router::trial`) when it starts: `None` unless the
//! acting optimizer armed that site in `ny_core::trial_authority`, which is
//! every run without `optimizer.mode: on`. The lane then feeds the machine
//! its own counters and its own clock at its own unit boundaries (its
//! declared safe points) through the `observe_*` helpers and reads the value
//! for its next unit. Unarmed, nothing here is constructed and the lane's
//! value is today's, bit for bit.
//!
//! FAIL OPEN. Every machine call runs inside `router::guard::fail_open`; a
//! panic is contained, disables the router (which disarms the authority) and
//! the helper DROPS the machine, so the lane returns to today's value from
//! its next unit. A machine is also dropped as soon as the authority is no
//! longer armed (a fault elsewhere, or the session's close).

use std::time::Instant;

use ny_core::router::guard;
use ny_core::router::trial::{
    AimdTrial, Iterate, LrBackoff, SplitObs, SplitRestart, TrialDecision, TrialDir, TrialEvent,
    WidthObs, WidthRange,
};
use ny_core::trial_authority::{self, TrialAuthority, TrialSite};

/// The process's trial authority (every production lane reads this one; a
/// unit test passes its own instance so it never arms the global).
pub(crate) fn global() -> &'static TrialAuthority {
    trial_authority::global()
}

/// Record and print trial events (a no-op unless `authority` is armed).
pub(crate) fn emit(authority: &TrialAuthority, events: Vec<TrialEvent>) {
    if !events.is_empty() {
        authority.emit(events);
    }
}

/// The lane-relative clock: seconds since `start`, and seconds left before
/// `deadline` (`None` without a deadline: a trial then refuses to start).
pub(crate) fn lane_clock(start: Instant, deadline: Option<Instant>) -> (f64, Option<f64>) {
    let now = Instant::now();
    let elapsed = now.saturating_duration_since(start).as_secs_f64();
    let remaining = deadline.map(|d| d.saturating_duration_since(now).as_secs_f64());
    (elapsed, remaining)
}

/// The declared span `[min, max]` of an integer-grid axis in the lever
/// space (`None` when the lever has no grid axis: the site then trials
/// nothing, fail closed).
pub(crate) fn declared_grid_span(lever: &str) -> Option<(u64, u64)> {
    let axis = ny_levers::space::axes()
        .iter()
        .chain(ny_levers::space::unsafe_axes())
        .find(|axis| axis.name == lever)?;
    match axis.domain {
        ny_levers::space::Domain::Grid(grid) => {
            let min = grid.iter().copied().min()?;
            let max = grid.iter().copied().max()?;
            Some((min, max))
        }
        _ => None,
    }
}

fn blocked(site: TrialSite, base: u64, why: &str) -> TrialEvent {
    TrialEvent {
        rule: site.rule().to_owned(),
        site: site.token().to_owned(),
        knob: site.knob().to_owned(),
        dir: TrialDir::Up,
        decision: TrialDecision::Blocked,
        from: base as f64,
        to: base as f64,
        metric: 0.0,
        baseline: 0.0,
        unit: 0,
        window: 0,
        why: why.to_owned(),
    }
}

/// `rule.trial_width` at a width site: an AIMD machine starting at today's
/// value `base`, moving only inside `[base / f, base x f]` intersected with
/// the site's own `[hard_min, hard_max]` (its reader clamp, declared span,
/// memory cap). `None` unless armed; armed but with no admissible range, a
/// `blocked` event says why and the lane keeps `base`.
pub(crate) fn width_trial(
    authority: &TrialAuthority,
    site: TrialSite,
    base: u64,
    hard_min: u64,
    hard_max: u64,
) -> Option<AimdTrial> {
    let rule = authority.width(site)?;
    let built = guard::fail_open("trial_width", || {
        WidthRange::around(base, rule.range_factor, hard_min, hard_max).and_then(|range| {
            AimdTrial::new(rule, site.rule(), site.token(), site.knob(), base, range)
                .map(|m| m.with_prior_trials(authority.started(site)))
        })
    })
    .ok()
    .flatten();
    if built.is_none() {
        emit(
            authority,
            vec![blocked(site, base, "base_outside_declared_range")],
        );
    }
    built
}

/// `rule.trial_width`: a DORMANT site — the knob exists but the lane's loop
/// does not read it on this route (today's value is fixed). Records one
/// `blocked` event when the site is armed, so a measurement can tell a
/// dormant knob from an inert rule; builds nothing.
pub(crate) fn width_dormant(authority: &TrialAuthority, site: TrialSite, live: u64, why: &str) {
    if authority.width(site).is_some() {
        emit(authority, vec![blocked(site, live, why)]);
    }
}

/// Feed a width machine; drop it (today's value from the next unit) when
/// the authority disarmed or the call faulted.
pub(crate) fn observe_width(
    authority: &TrialAuthority,
    trial: &mut Option<AimdTrial>,
    obs: WidthObs,
) {
    let Some(machine) = trial.as_mut() else {
        return;
    };
    if !authority.is_armed() {
        *trial = None;
        return;
    }
    match guard::fail_open("trial_width", || machine.observe(obs)) {
        Ok(events) => emit(authority, events),
        Err(_) => *trial = None,
    }
}

/// The value a width site uses for its next unit: the machine's, else
/// today's `base`.
pub(crate) fn width_value(trial: Option<&AimdTrial>, base: usize) -> usize {
    trial.map_or(base, |t| usize::try_from(t.value()).unwrap_or(base))
}

/// `rule.trial_lr_backoff` at the root alpha ascent (`None` unless armed).
pub(crate) fn lr_backoff(authority: &TrialAuthority) -> Option<LrBackoff> {
    let rule = authority.lr_backoff()?;
    let site = TrialSite::AlphaLr;
    guard::fail_open("trial_lr_backoff", || {
        LrBackoff::new(rule, site.rule(), site.token(), site.knob())
            .map(|m| m.with_prior_trials(authority.started(site)))
    })
    .ok()
    .flatten()
}

/// Feed the lr machine one iterate; drop it (multiplier 1, today's step) when
/// the authority disarmed or the call faulted.
pub(crate) fn observe_lr(authority: &TrialAuthority, trial: &mut Option<LrBackoff>, it: Iterate) {
    let Some(machine) = trial.as_mut() else {
        return;
    };
    if !authority.is_armed() {
        *trial = None;
        return;
    }
    match guard::fail_open("trial_lr_backoff", || machine.iterate(it)) {
        Ok(events) => emit(authority, events),
        Err(_) => *trial = None,
    }
}

/// `rule.trial_split_restart` at the lowdim lane: a machine that may switch
/// `from` to `to` once (`None` unless armed).
pub(crate) fn split_restart(
    authority: &TrialAuthority,
    from: &str,
    to: &str,
) -> Option<SplitRestart> {
    let rule = authority.split_restart()?;
    let site = TrialSite::LowdimSplit;
    // At most ONE restart per ROW: a later run of the lane after a restart
    // builds no machine.
    if authority.started(site) > 0 {
        return None;
    }
    guard::fail_open("trial_split_restart", || {
        SplitRestart::new(rule, site.rule(), site.token(), site.knob(), from, to)
    })
    .ok()
    .flatten()
}

/// Feed the restart machine; `true` means RESTART NOW (the event is already
/// recorded). Drops the machine when the authority disarmed or the call
/// faulted.
pub(crate) fn observe_split(
    authority: &TrialAuthority,
    trial: &mut Option<SplitRestart>,
    obs: SplitObs,
) -> bool {
    let Some(machine) = trial.as_mut() else {
        return false;
    };
    if !authority.is_armed() {
        *trial = None;
        return false;
    }
    match guard::fail_open("trial_split_restart", || machine.observe(obs)) {
        Ok(Some(event)) => {
            let restart = event.decision == TrialDecision::Started;
            emit(authority, vec![event]);
            restart
        }
        Ok(None) => false,
        Err(_) => {
            *trial = None;
            false
        }
    }
}
