// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 — the `ny-cli` half of the optimizer's O-TRIAL rules
//! (ROUTER_PROGRAM §5, §10 R6).
//!
//! `ny_core::router::otrial::commit` decides, purely, which trial rules act
//! on a row and at which sites; `ny_core::trial_authority` holds the arming
//! and the ledger; the pure machines (`ny_core::router::trial`) live inside
//! the lanes that own the knobs (`ny-propagate`: the wave tree's width, the
//! reordered input-split batch, the lowdim wave width and split heuristic,
//! the root alpha ascent's step). This module is the session glue and the
//! one site that lives in this crate, the upfront attack's step pool:
//!
//! | site | helper | what |
//! |---|---|---|
//! | `router::start` (scored entry, after the lever freeze) | [`pinned_keys`] | the trial knobs the row pinned (its file under every path, or a non-default lever source) |
//! | `router::start` (optimizer acting) | [`arm`] | `[optimizer] act rule=` per rule, `[router] trial-site` per site, then arm the authority |
//! | the upfront DLR-APGD lane (APGD geometry) | [`upfront_step_pool`] / [`pool_next`] / [`pool_done`] | successive halving over the step pool (fail open: dropped on a fault or a disarm) |
//! | `finish_after_publication` | [`finish`] | snapshot, `[router] trial done`, DISARM |
//!
//! Every helper is today's value in, today's value out unless the acting
//! optimizer armed its site, and every entry runs inside `guard::fail_open`.

use std::sync::{Mutex, PoisonError};

use ny_core::router::guard;
use ny_core::router::otrial::{TrialCommitments, RULE_TRIAL_ATTACK_HALVING};
use ny_core::router::trial::Halving;
use ny_core::trial_authority::{self, TrialAuthority, TrialCounts, TrialSite, TrialSnapshot};
use serde::{Deserialize, Serialize};

/// The O-TRIAL half of the flight `router` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct TrialRecord {
    /// Whether the trial authority armed.
    pub(crate) armed: bool,
    /// Why it refused to arm, when it did.
    pub(crate) refused: Option<String>,
    /// The authority at the end of the row (every trial event).
    pub(crate) snapshot: Option<TrialSnapshot>,
}

static STATE: Mutex<Option<TrialRecord>> = Mutex::new(None);

fn state() -> std::sync::MutexGuard<'static, Option<TrialRecord>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The trial knobs the row PINNED: a site's knob set by the row's category
/// file under any path it may be written at, or resolved by the frozen lever
/// set from a non-default source. Sorted, deduplicated. A file that cannot
/// be serialized pins every knob (fail closed).
pub(crate) fn pinned_keys(
    file: Option<&crate::preset::PresetConfig>,
    levers: Option<&ny_levers::LeverSet>,
) -> Vec<String> {
    let serialized = file.map(serde_json::to_value);
    let mut out = Vec::new();
    for site in TrialSite::ALL {
        for path in site.pin_paths() {
            let by_file = match &serialized {
                None => false,
                Some(Err(_)) => true,
                Some(Ok(value)) => super::route::file_sets(value, path),
            };
            let by_lever = levers.is_some_and(|levers| {
                super::route::lever_for_key(path)
                    .and_then(|lever| levers.resolved(lever))
                    .is_some_and(|resolved| resolved.source != ny_levers::Source::Default)
            });
            if by_file || by_lever {
                out.push((*path).to_owned());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `[optimizer] act rule=<id> acts=<0|1> [blocked_by=<why>] what=<...>` per
/// rule, then `[router] trial-site site=<cap> knob=<key> rule=<id>
/// armed=<0|1> blocked_by=<why|->` per site.
pub(crate) fn act_lines(commitments: &TrialCommitments) -> Vec<String> {
    let mut lines: Vec<String> = commitments
        .rules
        .iter()
        .map(|act| {
            format!(
                "[optimizer] act rule={} acts={}{} what={}",
                act.rule,
                u8::from(act.acts),
                act.blocked_by
                    .as_deref()
                    .map_or_else(String::new, |why| format!(" blocked_by={why}")),
                act.what.replace(char::is_whitespace, "_"),
            )
        })
        .collect();
    lines.extend(commitments.sites.iter().map(|s| {
        format!(
            "[router] trial-site site={} knob={} rule={} armed={} blocked_by={}",
            s.cap,
            s.knob,
            s.rule,
            u8::from(s.armed),
            s.blocked_by.as_deref().unwrap_or("-"),
        )
    }));
    lines
}

fn armed_line(commitments: &TrialCommitments) -> String {
    let d = commitments.directives;
    let sites: Vec<String> = d.armed_sites().into_iter().map(TrialSite::label).collect();
    format!(
        "[router] trial authority armed sha={} trial_width={} trial_lr_backoff={} \
         trial_split_restart={} trial_attack_halving={} sites={}",
        commitments.sha8(),
        u8::from(d.width.is_some()),
        u8::from(d.lr_backoff.is_some()),
        u8::from(d.split_restart.is_some()),
        u8::from(d.attack_halving.is_some()),
        if sites.is_empty() {
            "-".to_owned()
        } else {
            sites.join(",")
        },
    )
}

/// Arm the process's trial authority from the row's O-TRIAL commitments.
/// Called by `router::start` (inside `fail_open`) only when the optimizer
/// ACTS. Returns the lines to print.
pub(crate) fn arm(commitments: &TrialCommitments) -> Vec<String> {
    let (record, lines) = arm_with(trial_authority::global(), commitments);
    *state() = Some(record);
    lines
}

pub(crate) fn arm_with(
    authority: &TrialAuthority,
    commitments: &TrialCommitments,
) -> (TrialRecord, Vec<String>) {
    let mut lines = act_lines(commitments);
    let mut record = TrialRecord {
        armed: false,
        refused: None,
        snapshot: None,
    };
    if commitments.directives.any() {
        match authority.arm(commitments.directives) {
            Ok(()) => {
                record.armed = authority.is_armed();
                lines.push(armed_line(commitments));
            }
            Err(refusal) => {
                record.refused = Some(format!("{refusal:?}"));
                lines.push(format!(
                    "[router] trial authority refused reason={refusal:?}"
                ));
            }
        }
    } else {
        lines.push("[router] trial authority idle reason=no_site_armed".to_owned());
    }
    (record, lines)
}

/// Snapshot the trial authority and DISARM it. `None` when the optimizer did
/// not act on this row.
pub(crate) fn finish() -> Option<TrialRecord> {
    let record = state().take()?;
    Some(finish_with(trial_authority::global(), record))
}

pub(crate) fn finish_with(authority: &TrialAuthority, mut record: TrialRecord) -> TrialRecord {
    if record.armed {
        record.snapshot = Some(authority.snapshot());
    }
    authority.disarm();
    record
}

/// `[router] trial done armed= events= started= kept= reverted= tabu=
/// frozen= guard= blocked= restarts= dropped=`.
pub(crate) fn done_line(record: &TrialRecord) -> String {
    let (events, counts, dropped, restarts) = record.snapshot.as_ref().map_or_else(
        || (0, TrialCounts::default(), 0, 0),
        |s| {
            (
                s.events.len(),
                s.counts,
                s.dropped,
                s.events
                    .iter()
                    .filter(|e| {
                        e.rule == "rule.trial_split_restart"
                            && e.decision == ny_core::router::trial::TrialDecision::Started
                    })
                    .count(),
            )
        },
    );
    format!(
        "[router] trial done armed={} events={events} started={} kept={} reverted={} tabu={} \
         frozen={} guard={} blocked={} restarts={restarts} dropped={dropped}",
        u8::from(record.armed),
        counts.started,
        counts.kept,
        counts.reverted,
        counts.tabu,
        counts.frozen,
        counts.guard,
        counts.blocked,
    )
}

/// `rule.trial_attack_halving` at a restart's start: the pool's arm and step,
/// or `None` (today's step) when no pool is held.
///
/// Review fix (router-r6): FAIL OPEN like every lane-side machine
/// (`ny_propagate::trial_sites`). The machine call runs inside
/// `guard::fail_open`, and the pool is DROPPED — today's step from this
/// restart on — when the call faults, the router is disabled, or the trial
/// authority is no longer armed (a fault elsewhere, or the session's close).
/// The lane used to call the machine bare and keep its pool after a disarm.
pub(crate) fn pool_next(pool: &mut Option<Halving>) -> Option<(usize, f64)> {
    pool_next_with(trial_authority::global(), pool)
}

pub(crate) fn pool_next_with(
    authority: &TrialAuthority,
    pool: &mut Option<Halving>,
) -> Option<(usize, f64)> {
    let machine = pool.as_mut()?;
    if !authority.is_armed() {
        *pool = None;
        return None;
    }
    let next = guard::fail_open("trial_halving", || machine.next_arm()).ok();
    if next.is_none() {
        *pool = None;
    }
    next
}

/// `rule.trial_attack_halving` at a restart's end (the lane's safe point):
/// feed the pool the restart's improvement and record its events. Fail open
/// exactly as [`pool_next`].
pub(crate) fn pool_done(
    pool: &mut Option<Halving>,
    arm: usize,
    improvement: f64,
    secs: f64,
    done_frac: f64,
) {
    pool_done_with(
        trial_authority::global(),
        pool,
        arm,
        improvement,
        secs,
        done_frac,
    );
}

pub(crate) fn pool_done_with(
    authority: &TrialAuthority,
    pool: &mut Option<Halving>,
    arm: usize,
    improvement: f64,
    secs: f64,
    done_frac: f64,
) {
    let Some(machine) = pool.as_mut() else {
        return;
    };
    if !authority.is_armed() {
        *pool = None;
        return;
    }
    if let Err(fault) = guard::fail_open("trial_halving", || {
        let events = machine.done_restart(arm, improvement, secs, done_frac);
        if !events.is_empty() {
            authority.emit(events);
        }
    }) {
        if !fault.skipped {
            eprintln!("{}", super::telemetry::fault_line(&fault));
        }
        *pool = None;
    }
}

/// `rule.trial_attack_halving`: the upfront DLR-APGD lane's step pool
/// (`None` unless the acting optimizer armed the site). The caller asks it
/// only on the APGD geometry (the sign-PGD and global-step geometries keep
/// their own step rule).
pub(crate) fn upfront_step_pool() -> Option<Halving> {
    let rule = trial_authority::global().attack_halving()?;
    let site = TrialSite::AttackStep;
    guard::fail_open("trial_halving", || {
        Halving::new(
            rule,
            RULE_TRIAL_ATTACK_HALVING.as_str(),
            site.token(),
            site.knob(),
        )
    })
    .ok()
    .flatten()
}
