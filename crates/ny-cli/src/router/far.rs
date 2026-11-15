// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 CR-F — the `ny-cli` half of the FAR hand-on (ROUTER_PROGRAM
//! §4.2 CR-F, §4.3, §10 R4).
//!
//! `ny_core::router::r4::commit` decides, purely, whether CR-F arms and with
//! which routes (source -> STRICTLY sealed recipient, both owning one deadline
//! cell CR-A governs). `ny_core::far_authority` is consulted by the source
//! lanes themselves at their own safe points (lowdim's wave boundary,
//! `ny-propagate`), so this module only opens and closes the session:
//!
//! | site | helper |
//! |---|---|
//! | `router::start` (acting, CR-F eligible) | [`arm`] — the plan, the routes, the row's anchor |
//! | `finish_after_publication` | [`finish`] — snapshot, one `[router] far` line per live decision, DISARM |
//!
//! With no route — every R4 row, since no registry entry is `Sealed` — the
//! authority arms in OBSERVE mode: it classifies at c2/c3 and records the
//! decision (`act=0 blocked_by=no_sealed_recipient`) and hands nothing on.

use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use ny_core::far_authority::{self, FarAuthority, FarSnapshot};
use ny_core::router::plan::Plan;
use ny_core::router::r4::FarCommit;
use serde::{Deserialize, Serialize};

/// The CR-F half of the flight `router` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FarRecord {
    /// Whether the authority armed.
    pub(crate) armed: bool,
    /// The authority at the end of the row.
    pub(crate) snapshot: Option<FarSnapshot>,
}

static STATE: Mutex<Option<FarRecord>> = Mutex::new(None);

fn state() -> std::sync::MutexGuard<'static, Option<FarRecord>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `[router] far armed ...`.
fn armed_line(commit: &FarCommit) -> String {
    let routes = commit
        .routes
        .iter()
        .map(|r| {
            format!(
                "{}({})>{}({})",
                r.from,
                r.from_slot.token(),
                r.to,
                r.to_slot.token()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "[router] far armed rule=CR-F mode={} routes=[{}] ckpt=c2,c3 window=work_units",
        if commit.observe_only {
            "observe"
        } else {
            "hand_on"
        },
        if routes.is_empty() { "-" } else { &routes },
    )
}

/// CR-F: arm the process's FAR authority. Called by `router::start` (inside
/// `fail_open`) only when the router ACTS and CR-F is eligible (its key, CR-1,
/// not a short row). Returns the lines to print.
pub(crate) fn arm(plan: &Plan, commit: Option<&FarCommit>, anchor: Instant) -> Vec<String> {
    let (record, lines) = arm_with(far_authority::global(), plan, commit, anchor);
    *state() = Some(record);
    lines
}

pub(crate) fn arm_with(
    authority: &FarAuthority,
    plan: &Plan,
    commit: Option<&FarCommit>,
    anchor: Instant,
) -> (FarRecord, Vec<String>) {
    match commit {
        None => (
            FarRecord {
                armed: false,
                snapshot: None,
            },
            vec!["[router] far idle rule=CR-F reason=not_eligible".to_owned()],
        ),
        Some(commit) => {
            authority.arm(plan.clone(), commit.clone(), anchor);
            (
                FarRecord {
                    armed: true,
                    snapshot: None,
                },
                vec![armed_line(commit)],
            )
        }
    }
}

/// Close the row's FAR session: snapshot, DISARM, and the lines to print
/// (one per live decision, every refused suspension, the summary). `None`
/// when CR-F did not arm here.
pub(crate) fn finish() -> Option<(FarRecord, Vec<String>)> {
    let mut record = state().take()?;
    let authority = far_authority::global();
    let mut lines = Vec::new();
    if record.armed {
        let snapshot = authority.snapshot();
        lines.extend(event_lines(&snapshot));
        record.snapshot = Some(snapshot);
    }
    authority.disarm();
    lines.push(done_line(&record));
    Some((record, lines))
}

/// One line per live decision and per refused suspension.
pub(crate) fn event_lines(snapshot: &FarSnapshot) -> Vec<String> {
    let mut lines: Vec<String> = snapshot
        .events
        .iter()
        .map(|e| {
            format!(
                "[router] far cap={} ckpt={} at_ms={} unit={} class={} act={} to={} \
                 blocked_by={} handed={} unspent_ms={} credited_ms={} samples={}",
                e.decision.cap,
                e.decision.at.token(),
                e.at_ms,
                e.unit,
                far_authority::class_token(e.decision.class),
                u8::from(e.decision.act),
                e.decision.to.as_deref().unwrap_or("-"),
                e.decision
                    .blocked_by
                    .map(|b| b.token().to_owned())
                    .or_else(|| e.not_handed.clone())
                    .unwrap_or_else(|| "-".to_owned()),
                u8::from(e.handed),
                e.unspent_ms,
                e.credited_ms,
                e.samples.len(),
            )
        })
        .collect();
    lines.extend(snapshot.refusals.iter().map(|r| {
        format!(
            "[router] safepoint refused cap={} at={} reason={} rule=CR-F",
            r.cap,
            r.at.token(),
            r.reason
        )
    }));
    lines
}

/// The `[router] far done ...` summary line.
pub(crate) fn done_line(record: &FarRecord) -> String {
    let s = record.snapshot.as_ref();
    format!(
        "[router] far done armed={} observe_only={} calls={} decisions={} far={} handed={} \
         refused={} dropped={}",
        u8::from(record.armed),
        u8::from(
            s.and_then(|s| s.commit.as_ref())
                .is_some_and(|c| c.observe_only)
        ),
        s.map_or(0, |s| s.calls),
        s.map_or(0, |s| s.events.len()),
        s.map_or(0, |s| s
            .events
            .iter()
            .filter(|e| e.decision.class == ny_core::router::project::FarClass::Far)
            .count()),
        s.map_or(0, |s| s.handed),
        s.map_or(0, |s| s.refusals.len()),
        s.map_or(0, |s| s.dropped),
    )
}
