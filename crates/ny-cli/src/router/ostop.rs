// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r3 — the `ny-cli` half of ACTING SET 2, the optimizer's O-STOP
//! rules (ROUTER_PROGRAM §5, §10 R3).
//!
//! `ny_core::router::ostop::commit` decides, purely, which of the four stop
//! rules act on a row; `ny_core::stop_authority` holds the arming and the
//! ledger. The margin-row and graph-engine phases read the authority directly
//! (`ny-propagate`: the alpha ascent's `rule.governor`, the interm-alpha
//! candidates' `rule.ia_yield` and the ascent's contract, the tree's hand-on,
//! the root phases' `rule.root_admission`). This module is the glue for the
//! session (arm at `router::start`, close at `finish_after_publication`) and
//! for the one rule that lives in this crate, CR-ATK:
//!
//! | site | helper | what |
//! |---|---|---|
//! | `router::start` (after R2's arming) | [`arm`] | arm the stop authority from the row's commitments |
//! | before the upfront attack lane | [`enter_upfront_scope`] | CR-ATK may defer only when the late sat reserve is admissible |
//! | the upfront lane's slice is fixed | [`upfront_slice`] | the slice the slope window is a fraction of |
//! | each upfront restart boundary (its safe point) | [`upfront_restart_boundary`] | slope-flat over half the slice and >= 5 s the lane would still spend (its deadline, or its restart cap at the shortest restart seen) ⇒ defer |
//! | the upfront lane returns, BEFORE the trusted-oracle gate | [`UpfrontDeferScope::close`] | the scope closes, so the gate's ORT repair loop is never read as the upfront lane |
//! | after the upfront lane | `acting::release_lane_to_reserve` | the unspent slice goes to the late reserve, not the terminal |
//! | before the verifier | [`deferred_reserve_secs`] / [`note_defer_held`] | hold the deferred slice out of the verifier (which keeps today's window) |
//! | the post-verifier attack's start | [`note_late_attack_start`] | the late attack spent it (`realized=`) |
//! | `finish_after_publication` | [`finish`] | close every open defer (moot / unrealized), snapshot, DISARM |
//!
//! Every helper is `today` in, `today` out unless the acting optimizer armed
//! its rule, and every entry runs inside `guard::fail_open`.

use std::cell::RefCell;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use ny_core::deadline_authority::{self, LaneSlot};
use ny_core::router::guard;
use ny_core::router::ostop::{attack_flat, SlopeVerdict, StopCommitments};
use ny_core::stop_authority::{self, DeferEvent, DeferOutcome, DeferRule, StopSnapshot};
use serde::{Deserialize, Serialize};

/// Restart-boundary samples one upfront lane keeps (oldest dropped past it).
const MAX_SAMPLES: usize = 1_024;

/// The O-STOP half of the flight `router` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StopRecord {
    /// Whether the stop authority armed.
    pub(crate) armed: bool,
    /// Why it refused to arm, when it did.
    pub(crate) refused: Option<String>,
    /// The authority at the end of the row (yields, hand-ons, defers, root
    /// admissions).
    pub(crate) snapshot: Option<StopSnapshot>,
}

static STATE: Mutex<Option<StopRecord>> = Mutex::new(None);

fn state() -> std::sync::MutexGuard<'static, Option<StopRecord>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Run `f` fail-open; on a fault (or a disabled router) return `fallback`.
fn open_or<T>(site: &'static str, fallback: T, f: impl FnOnce() -> T) -> T {
    match guard::fail_open(site, f) {
        Ok(value) => value,
        Err(fault) => {
            if !fault.skipped {
                eprintln!("{}", super::telemetry::fault_line(&fault));
            }
            fallback
        }
    }
}

/// `[optimizer] act rule=<id> acts=<0|1> [blocked_by=<why>] what=<...>`.
pub(crate) fn act_lines(commitments: &StopCommitments) -> Vec<String> {
    commitments
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
        .collect()
}

/// `[optimizer] authority armed ...`: the directives the phases will read.
fn armed_line(commitments: &StopCommitments) -> String {
    let d = commitments.directives;
    let ratio = |rule: Option<stop_authority::RatioRule>| {
        rule.map_or_else(
            || "-".to_owned(),
            |r| format!("{}/{}", r.ratio, r.min_samples),
        )
    };
    format!(
        "[optimizer] authority armed sha={} governor={} ia_yield={} attack_defer={} \
         root_admission={}",
        commitments.sha8(),
        ratio(d.governor),
        ratio(d.ia_yield),
        d.attack_defer.map_or_else(
            || "-".to_owned(),
            |r| format!("{}/{}ms", r.window_frac, r.min_defer_ms)
        ),
        u8::from(d.root_admission),
    )
}

/// Arm the process's stop authority from the row's O-STOP commitments.
/// Called by `router::start` (inside `fail_open`) only when the optimizer
/// ACTS. Returns the lines to print.
pub(crate) fn arm(commitments: &StopCommitments) -> Vec<String> {
    arm_with(stop_authority::global(), commitments)
}

pub(crate) fn arm_with(
    authority: &stop_authority::StopAuthority,
    commitments: &StopCommitments,
) -> Vec<String> {
    let mut lines = act_lines(commitments);
    let mut record = StopRecord {
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
                lines.push(format!("[optimizer] authority refused reason={refusal:?}"));
            }
        }
    } else {
        lines.push("[optimizer] authority idle reason=no_rule_acts".to_owned());
    }
    *state() = Some(record);
    lines
}

/// Whether the row's verdict token decides it (a decided row needs no late
/// attack: a deferred slice it never spent is MOOT, not lost).
fn decided(verdict: &str) -> bool {
    matches!(verdict, "sat" | "unsat")
}

/// Close every open defer, snapshot the stop authority and DISARM it. `None`
/// when the optimizer did not act on this row.
pub(crate) fn finish(verdict: &str) -> Option<StopRecord> {
    let record = state().take()?;
    Some(finish_with(stop_authority::global(), record, verdict))
}

pub(crate) fn finish_with(
    authority: &stop_authority::StopAuthority,
    mut record: StopRecord,
    verdict: &str,
) -> StopRecord {
    if record.armed {
        let moot = decided(verdict);
        if let Some(defer) = authority.take_pending_defer() {
            authority.record_defer_outcome(DeferOutcome {
                deferred_ms: defer.unspent_ms,
                held_ms: 0,
                late_budget_ms: None,
                realized: false,
                moot,
                reason: if moot { "row_decided" } else { "never_held" }.to_owned(),
            });
        }
        if let Some((deferred, held)) = authority.held_reserve() {
            authority.record_defer_outcome(DeferOutcome {
                deferred_ms: ms(deferred),
                held_ms: ms(held),
                late_budget_ms: None,
                realized: false,
                moot,
                reason: if moot {
                    "row_decided"
                } else {
                    "late_attack_not_started"
                }
                .to_owned(),
            });
        }
        record.snapshot = Some(authority.snapshot());
    }
    authority.disarm();
    record
}

/// The `[optimizer] done ...` summary line.
pub(crate) fn done_line(record: &StopRecord) -> String {
    let empty = StopSnapshot {
        armed: false,
        directives: None,
        yields: Vec::new(),
        hand_ons: Vec::new(),
        ascent_credit_ms: 0,
        defers: Vec::new(),
        defer_outcomes: Vec::new(),
        root_admissions: Vec::new(),
        dropped: 0,
    };
    let s = record.snapshot.as_ref().unwrap_or(&empty);
    let freed: u64 = s
        .hand_ons
        .iter()
        .fold(0u64, |sum, h| sum.saturating_add(h.freed_ms));
    let credited: u64 = s
        .hand_ons
        .iter()
        .fold(0u64, |sum, h| sum.saturating_add(h.credited_ms));
    let deferred: u64 = s
        .defers
        .iter()
        .fold(0u64, |sum, d| sum.saturating_add(d.unspent_ms));
    format!(
        "[optimizer] done armed={} yields={} hand_ons={} freed_ms={freed} credited_ms={credited} \
         unrealized={} ascent_credit_ms={} defers={} deferred_ms={deferred} late_realized={} \
         late_unrealized={} late_moot={} root_admissions={} root_declined={} dropped={}",
        u8::from(record.armed),
        s.yields.len(),
        s.hand_ons.len(),
        s.hand_ons.iter().filter(|h| !h.realized).count(),
        s.ascent_credit_ms,
        s.defers.len(),
        s.defer_outcomes.iter().filter(|o| o.realized).count(),
        s.defer_outcomes
            .iter()
            .filter(|o| !o.realized && !o.moot)
            .count(),
        s.defer_outcomes.iter().filter(|o| o.moot).count(),
        s.root_admissions.len(),
        s.root_admissions.iter().filter(|r| !r.admitted).count(),
        s.dropped,
    )
}

// ---------------------------------------------------------------------------
// CR-ATK (`rule.attack_defer`): the upfront attack's slope-plateau DEFER.
// ---------------------------------------------------------------------------

struct UpfrontScope {
    rule: DeferRule,
    start: Option<Instant>,
    deadline: Option<Instant>,
    samples: Vec<(f64, f64)>,
    deferred: Option<DeferEvent>,
    /// #router-r3 review: the previous restart boundary's offset (seconds),
    /// and the SHORTEST restart observed between two boundaries — the
    /// projection of what a restart-capped lane would still have spent.
    last_boundary_s: Option<f64>,
    min_restart_s: Option<f64>,
}

thread_local! {
    /// The upfront lane runs on the scored entry point's thread; its restart
    /// loop reads this scope. Set only by [`enter_upfront_scope`] when CR-ATK
    /// is armed; every other attack consultation (and every unarmed row) sees
    /// `None`.
    static UPFRONT: RefCell<Option<UpfrontScope>> = const { RefCell::new(None) };
}

/// RAII scope of one upfront lane under CR-ATK. Dropping it clears the scope.
pub(crate) struct UpfrontDeferScope {
    _private: (),
}

impl Drop for UpfrontDeferScope {
    fn drop(&mut self) {
        UPFRONT.with(|cell| {
            if let Ok(mut scope) = cell.try_borrow_mut() {
                *scope = None;
            }
        });
    }
}

impl UpfrontDeferScope {
    /// The defer this lane made, if it did.
    pub(crate) fn deferred(&self) -> Option<DeferEvent> {
        UPFRONT.with(|cell| {
            cell.try_borrow()
                .ok()
                .and_then(|scope| scope.as_ref().and_then(|s| s.deferred.clone()))
        })
    }

    /// #router-r3 review: CLOSE the upfront lane's scope the moment the lane
    /// returns and report whether it deferred. The caller must close it
    /// BEFORE anything else on this thread can run an attack restart loop:
    /// the trusted-oracle gate's ORT repair (`refine_witness_with_ort` ->
    /// `gradient_guided_falsify`) is such a loop, and with the scope still
    /// open it would read as the upfront lane — it could DEFER the repair
    /// (losing the sat it exists to recover) and book a defer the upfront lane
    /// never made. `None` (every unarmed row) is `false`.
    pub(crate) fn close(scope: Option<Self>) -> bool {
        let deferred = scope
            .as_ref()
            .and_then(UpfrontDeferScope::deferred)
            .is_some();
        drop(scope);
        deferred
    }
}

/// CR-ATK: open the upfront lane's defer scope. `None` (no scope: the lane
/// runs exactly today's restart loop) unless the acting optimizer armed
/// `rule.attack_defer`, the router governs the upfront slot, and the late sat
/// reserve is admissible on this row (`late_reserve_admissible`, evaluated
/// only when armed: the
/// post-verifier attack is enabled, memory allows it, no concurrent margin-row
/// worker will hold the tail, no proof tail is reserved). Defer, never cut:
/// where the late reserve cannot be honoured the lane is never stopped.
pub(crate) fn enter_upfront_scope(
    late_reserve_admissible: impl FnOnce() -> bool,
) -> Option<UpfrontDeferScope> {
    enter_upfront_scope_with(
        stop_authority::global(),
        deadline_authority::global(),
        late_reserve_admissible,
    )
}

pub(crate) fn enter_upfront_scope_with(
    stop: &stop_authority::StopAuthority,
    cells: &deadline_authority::Authority,
    late_reserve_admissible: impl FnOnce() -> bool,
) -> Option<UpfrontDeferScope> {
    // Unarmed: return before the admission predicate (a live memory probe) is
    // even evaluated.
    let rule = stop.attack_defer()?;
    open_or("defer_scope", None, || {
        let governed = cells.governs(LaneSlot::AttackUpfront);
        if !governed || !late_reserve_admissible() {
            eprintln!(
                "[optimizer] defer rule=rule.attack_defer armed=0 reason={}",
                if governed {
                    "late_reserve_unavailable"
                } else {
                    "attack_upfront_ungoverned"
                }
            );
            return None;
        }
        UPFRONT.with(|cell| {
            *cell.borrow_mut() = Some(UpfrontScope {
                rule,
                start: None,
                deadline: None,
                samples: Vec::new(),
                deferred: None,
                last_boundary_s: None,
                min_restart_s: None,
            });
        });
        eprintln!(
            "[optimizer] defer rule=rule.attack_defer armed=1 window_frac={} min_defer_ms={}",
            rule.window_frac, rule.min_defer_ms
        );
        Some(UpfrontDeferScope { _private: () })
    })
}

/// CR-ATK: the upfront lane fixed its slice `[start, deadline)`. A no-op
/// without an open scope.
pub(crate) fn upfront_slice(start: Instant, deadline: Instant) {
    UPFRONT.with(|cell| {
        if let Ok(mut scope) = cell.try_borrow_mut() {
            if let Some(scope) = scope.as_mut() {
                scope.start = Some(start);
                scope.deadline = Some(deadline);
            }
        }
    });
}

/// Whether the current thread's upfront lane runs under CR-ATK (one TLS
/// read; the lane gates every extra computation on it).
pub(crate) fn upfront_scope_armed() -> bool {
    UPFRONT.with(|cell| cell.try_borrow().is_ok_and(|scope| scope.is_some()))
}

/// The confirmation path's noise floor at an output whose largest magnitude
/// is `max_abs` (`max(1e-4, 1e-5 |y|max)`, the measured ny-vs-ORT forward
/// disagreement; the same floor the disjunctive attack's stall rule uses).
pub(crate) fn noise_floor(max_abs: f64) -> f64 {
    let max_abs = if max_abs.is_finite() {
        max_abs.abs()
    } else {
        0.0
    };
    (1e-5 * max_abs).max(1e-4)
}

/// CR-ATK at a restart boundary (the lane's safe point): record the lane's
/// running best margin and decide. `true` = DEFER NOW (the lane stops and
/// returns no candidate; its unspent slice goes to the late reserve). Defers
/// only when the slope over the trailing `>= 0.5` of the slice is flat
/// (`router::ostop::attack_flat`) AND at least the rule's minimum is left to
/// defer; otherwise the lane runs on. Sticky once deferred. `false` without
/// an open scope.
///
/// `restarts_left` is the lane's own restart cap still ahead of it (`None`:
/// the lane draws restarts until its deadline). #router-r3 review: a
/// restart-capped lane (the upfront lane keeps `GRAD_REFINE_MAX_RESTARTS`)
/// can end BEFORE its deadline today, so `deadline - now` overstates what it
/// would still have spent, and holding that much out of the verifier would
/// take verifier time today's route gives it. The deferred amount is
/// therefore `min(deadline - now, restarts_left x the SHORTEST restart
/// observed)` — what the lane would still have spent, projected — and a
/// projection under the rule's minimum (or no observed restart yet) is no
/// defer at all.
pub(crate) fn upfront_restart_boundary(
    restart: usize,
    restarts_left: Option<usize>,
    best_overall: f64,
    noise: f64,
) -> bool {
    if !upfront_scope_armed() {
        return false;
    }
    upfront_restart_boundary_at(
        stop_authority::global(),
        Instant::now(),
        restart,
        restarts_left,
        best_overall,
        noise,
    )
}

pub(crate) fn upfront_restart_boundary_at(
    stop: &stop_authority::StopAuthority,
    now: Instant,
    restart: usize,
    restarts_left: Option<usize>,
    best_overall: f64,
    noise: f64,
) -> bool {
    open_or("defer", false, || {
        UPFRONT.with(|cell| {
            let Ok(mut slot) = cell.try_borrow_mut() else {
                return false;
            };
            let Some(scope) = slot.as_mut() else {
                return false;
            };
            if scope.deferred.is_some() {
                return true;
            }
            let (Some(start), Some(deadline)) = (scope.start, scope.deadline) else {
                return false;
            };
            let t = now.saturating_duration_since(start).as_secs_f64();
            // The restart that just ended ran from the previous boundary to
            // this one (the first boundary also carries the lane's setup, so
            // it is never a restart sample).
            if let Some(previous) = scope.last_boundary_s {
                let restart_s = t - previous;
                if restart_s.is_finite() && restart_s > 0.0 {
                    scope.min_restart_s =
                        Some(scope.min_restart_s.map_or(restart_s, |m| m.min(restart_s)));
                }
            }
            scope.last_boundary_s = Some(t);
            if scope.samples.len() >= MAX_SAMPLES {
                scope.samples.drain(..MAX_SAMPLES / 2);
            }
            scope.samples.push((t, best_overall));
            let slice = deadline.saturating_duration_since(start);
            let SlopeVerdict::Flat {
                rise,
                eps,
                window_s,
            } = attack_flat(
                &scope.samples,
                slice.as_secs_f64(),
                scope.rule.window_frac,
                noise,
            )
            else {
                return false;
            };
            let to_deadline = deadline.saturating_duration_since(now);
            // What the lane would still have spent TODAY: up to its deadline,
            // or — for a restart-capped lane — its remaining restarts at the
            // shortest restart observed, whichever ends first. Only that much
            // is deferred (and later held out of the verifier), so the
            // verifier never loses time today's lane would have left it.
            let projected = match restarts_left {
                None => None,
                Some(left) => {
                    let Some(per_restart) = scope.min_restart_s else {
                        // No restart observed between two boundaries: nothing
                        // to project from, so no defer.
                        return false;
                    };
                    let secs = per_restart * left as f64;
                    Some(Duration::from_secs_f64(if secs.is_finite() {
                        secs.clamp(0.0, 1e9)
                    } else {
                        0.0
                    }))
                }
            };
            let unspent = projected.map_or(to_deadline, |p| to_deadline.min(p));
            if unspent < Duration::from_millis(scope.rule.min_defer_ms) {
                // Too little left for the late attack to start (or the lane's
                // own restart cap ends it first): deferring would be a cut,
                // so the lane runs on to today's cap.
                return false;
            }
            let event = DeferEvent {
                restart: restart as u64,
                slice_ms: ms(slice),
                elapsed_ms: ms(now.saturating_duration_since(start)),
                unspent_ms: ms(unspent),
                best_margin: best_overall,
                rise,
                eps,
                window_ms: ms(Duration::from_secs_f64(window_s.clamp(0.0, 1e9))),
            };
            eprintln!(
                "[optimizer] defer rule=rule.attack_defer site=attack.upfront_apgd restart={} \
                 elapsed_ms={} unspent_ms={} to_deadline_ms={} restarts_left={} \
                 projected_ms={} slice_ms={} window_ms={} best_margin={:.6e} rise={:.3e} \
                 eps={:.3e} to=late_sat_reserve",
                event.restart,
                event.elapsed_ms,
                event.unspent_ms,
                ms(to_deadline),
                restarts_left.map_or_else(|| "uncapped".to_owned(), |n| n.to_string()),
                projected.map_or_else(|| "-".to_owned(), |p| ms(p).to_string()),
                event.slice_ms,
                event.window_ms,
                event.best_margin,
                event.rise,
                event.eps,
            );
            stop.record_defer(event.clone());
            scope.deferred = Some(event);
            true
        })
    })
}

/// CR-ATK before the verifier: how many whole seconds of a pending defer to
/// hold out of the verifier as the late sat reserve. Zero when nothing was
/// deferred. When the late consumer is no longer admissible
/// (`late_consumer_available == false`: the memory envelope or the tail
/// changed since the lane deferred) the slice cannot be held: the outcome is
/// recorded as unrealized and the time falls to the verifier through its own
/// live clock. Whole seconds round DOWN, so the verifier is never given less
/// than today's window by the hold.
pub(crate) fn deferred_reserve_secs(late_consumer_available: bool) -> u64 {
    deferred_reserve_secs_with(stop_authority::global(), late_consumer_available)
}

pub(crate) fn deferred_reserve_secs_with(
    authority: &stop_authority::StopAuthority,
    late_consumer_available: bool,
) -> u64 {
    if authority.attack_defer().is_none() {
        return 0;
    }
    open_or("defer_hold", 0, || {
        let Some(defer) = authority.take_pending_defer() else {
            return 0;
        };
        if !late_consumer_available {
            eprintln!(
                "[optimizer] defer-hold rule=rule.attack_defer deferred_ms={} held_s=0 \
                 reason=late_consumer_unavailable_at_hold",
                defer.unspent_ms
            );
            authority.record_defer_outcome(DeferOutcome {
                deferred_ms: defer.unspent_ms,
                held_ms: 0,
                late_budget_ms: None,
                realized: false,
                moot: false,
                reason: "late_consumer_unavailable_at_hold".to_owned(),
            });
            return 0;
        }
        // Remember the exact deferred time until the hold is recorded.
        authority.note_defer_held(Duration::from_millis(defer.unspent_ms), Duration::ZERO);
        defer.unspent_ms / 1_000
    })
}

/// CR-ATK: the verifier's window was cut by `held_secs` (of `requested_secs`)
/// for the late reserve; the verifier keeps `verifier_secs`.
pub(crate) fn note_defer_held(requested_secs: u64, held_secs: u64, verifier_secs: u64) {
    note_defer_held_with(
        stop_authority::global(),
        requested_secs,
        held_secs,
        verifier_secs,
    );
}

pub(crate) fn note_defer_held_with(
    authority: &stop_authority::StopAuthority,
    requested_secs: u64,
    held_secs: u64,
    verifier_secs: u64,
) {
    let Some((deferred, _)) = authority.held_reserve() else {
        return;
    };
    open_or("defer_held", (), || {
        authority.note_defer_held(deferred, Duration::from_secs(held_secs));
        eprintln!(
            "[optimizer] defer-hold rule=rule.attack_defer deferred_ms={} requested_s={requested_secs} \
             held_s={held_secs} verifier_s={verifier_secs} to=late_sat_reserve",
            ms(deferred)
        );
    });
}

/// CR-ATK: the post-verifier attack started with `budget` of attack time. A
/// held deferred slice is REALIZED when that budget covers it (one second of
/// whole-second rounding tolerated). A no-op without a held reserve.
pub(crate) fn note_late_attack_start(budget: Duration) {
    note_late_attack_start_with(stop_authority::global(), budget);
}

pub(crate) fn note_late_attack_start_with(
    authority: &stop_authority::StopAuthority,
    budget: Duration,
) {
    let Some((deferred, held)) = authority.held_reserve() else {
        return;
    };
    open_or("defer_late", (), || {
        let realized = budget.saturating_add(Duration::from_secs(1)) >= deferred;
        eprintln!(
            "[optimizer] defer-late rule=rule.attack_defer deferred_ms={} held_ms={} \
             late_budget_ms={} realized={}",
            ms(deferred),
            ms(held),
            ms(budget),
            u8::from(realized)
        );
        authority.record_defer_outcome(DeferOutcome {
            deferred_ms: ms(deferred),
            held_ms: ms(held),
            late_budget_ms: Some(ms(budget)),
            realized,
            moot: false,
            reason: "late_attack_started".to_owned(),
        });
    });
}
