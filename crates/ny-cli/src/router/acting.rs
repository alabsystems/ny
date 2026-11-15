// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r2 — the `ny-cli` half of ACTING SET 1 (ROUTER_PROGRAM §10 R2).
//!
//! `ny_core::router::acting::commit` decides, purely, what the seven rules do
//! on a row; `ny_core::deadline_authority` holds the monotone deadline cells.
//! This module is the glue between them and the lanes that read caps and
//! deadlines on the scored path — every function below is called AT a lane's
//! own read site, and every one of them returns the caller's own "today"
//! value unchanged whenever the authority does not govern that lane (router
//! off or shadow, rule ablated, CR-A left the lane ungoverned, or a fault
//! disarmed it). That is the whole bit-identity argument, and it is local:
//! each helper is `today` in, `today` out, unless the authority is armed.
//!
//! | site | helper | rule |
//! |---|---|---|
//! | `router::start` (after the lever freeze) | [`arm`] | CR-1, CR-A, CR-T, CR-R (arming) |
//! | attack-slice frame admission | [`governs_attack_slice`] | CR-C (the `#lane-budget-frame` identity plan, admitted by the router) |
//! | after the frame is committed | [`commit_attack_contracts`] | CR-C |
//! | each attack lane's call | [`claim_attack_window`] | CR-1/CR-C (the lane is handed `cell - now`) |
//! | after each attack lane returns without a sat | [`release_lane`] | CR-H (unspent contract to the terminal) |
//! | the internal verifier's start / return | [`claim_terminal`] / [`release_lane`] | CR-T |
//! | the margin-row lanes' deadlines | [`claim_margin_row`] | CR-1 (recorded, never moved: no safe points) |
//! | the inline lane's external cap, the concurrent lane's final join | [`collector_grace`], [`final_join_margin`] | CR-R |
//! | the verifier's configuration boundary | [`apply_verifier_directives`] | CR-K, CR-H (phase hand-on) |
//!
//! Every entry runs inside `guard::fail_open`: a panic is contained, counted,
//! disarms the authority, and the lane gets its own value back.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use ny_core::deadline_authority::{self, AuthoritySnapshot, Frame, LaneSlot, SLOTS};
use ny_core::router::acting::{Commitments, DerivedPlan};
use ny_core::router::derive::{self, DepthGate};
use ny_core::router::guard;
use ny_core::router::safepoint;
use serde::{Deserialize, Serialize};

use crate::commands::lane_allocation::{
    AllocatedLane, AllocationAuthority, AttackSliceAllocation, LaneWindow,
};

/// Lane events one row records (bounded).
const MAX_EVENTS: usize = 64;

/// Slack a collector waits past a lane's publish deadline for the completed
/// result to cross the channel (a struct move, measured in microseconds).
pub(crate) const COLLECT_SLACK: Duration = Duration::from_millis(250);

/// One lane event, as recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct LaneEvent {
    /// The slot.
    pub(crate) slot: String,
    /// `commit`, `claim`, `release`, `grace`.
    pub(crate) event: String,
    /// The rendered detail (the same text as the `[router] lane` line).
    pub(crate) detail: String,
}

/// One derived configuration value (CR-K, CR-H), as applied or blocked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DerivedRecord {
    /// The rule.
    pub(crate) rule: String,
    /// The typed key.
    pub(crate) key: String,
    /// The value before.
    pub(crate) from: String,
    /// The value after (equal to `from` when blocked).
    pub(crate) to: String,
    /// `None` when applied; `pin`, or the depth gate's token, when blocked.
    pub(crate) blocked_by: Option<String>,
}

/// The acting half of the flight `router` record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ActingRecord {
    /// Whether the authority armed (`false` with `refused`).
    pub(crate) armed: bool,
    /// Why the authority refused to arm, when it did.
    pub(crate) refused: Option<String>,
    /// Lane events, in order (bounded).
    pub(crate) events: Vec<LaneEvent>,
    /// Events past the bound.
    pub(crate) events_dropped: u64,
    /// Derived configuration values.
    pub(crate) derived: Vec<DerivedRecord>,
    /// The authority at the end of the row (cells, contracts, hand-ons).
    pub(crate) authority: Option<AuthoritySnapshot>,
}

struct ActingState {
    commitments: Commitments,
    record: ActingRecord,
}

static STATE: Mutex<Option<ActingState>> = Mutex::new(None);

fn state() -> std::sync::MutexGuard<'static, Option<ActingState>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn authority() -> &'static deadline_authority::Authority {
    deadline_authority::global()
}

/// Record and print one lane event.
fn event(slot: LaneSlot, kind: &str, detail: String) {
    eprintln!("[router] lane slot={} event={kind} {detail}", slot.token());
    if let Some(state) = state().as_mut() {
        if state.record.events.len() < MAX_EVENTS {
            state.record.events.push(LaneEvent {
                slot: slot.token().to_owned(),
                event: kind.to_owned(),
                detail,
            });
        } else {
            state.record.events_dropped = state.record.events_dropped.saturating_add(1);
        }
    }
}

/// `[router] act rule=<id> acts=<0|1> [blocked_by=<why>] what=<...>`.
pub(crate) fn act_lines(commitments: &Commitments) -> Vec<String> {
    rule_lines(&commitments.rules)
}

/// #router-r4: the same `[router] act` lines for acting set 3 (CR-L, CR-F).
pub(crate) fn act_lines_r4(commitments: &ny_core::router::r4::R4Commitments) -> Vec<String> {
    rule_lines(&commitments.rules)
}

/// #router-r5: the same `[router] act` lines for acting set 4 (CR-RS, CR-G).
pub(crate) fn act_lines_r5(commitments: &ny_core::router::route::R5Commitments) -> Vec<String> {
    rule_lines(&commitments.rules)
}

fn rule_lines(rules: &[ny_core::router::acting::RuleAct]) -> Vec<String> {
    rules
        .iter()
        .map(|act| {
            format!(
                "[router] act rule={} acts={}{} what={}",
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

/// `[router] authority armed ...` — the frame the lanes will read.
fn armed_line(commitments: &Commitments, frame: &Frame) -> String {
    let caps = LaneSlot::ALL
        .into_iter()
        .map(|slot| format!("{}:{}ms", slot.token(), frame.cap_ns(slot) / 1_000_000))
        .collect::<Vec<_>>()
        .join(",");
    let governed = LaneSlot::ALL
        .into_iter()
        .filter(|slot| commitments.governed(*slot))
        .map(LaneSlot::token)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "[router] authority armed sha={} hard_ms={} join_ms={} grace_ms={} terminal={} \
         governed={} caps=[{caps}]",
        commitments.sha8(),
        frame.hard_ns() / 1_000_000,
        ms(frame.join()),
        commitments
            .grace_ms
            .map_or_else(|| "-".to_owned(), |g| g.to_string()),
        commitments.terminal.map_or("-", LaneSlot::token),
        if governed.is_empty() { "-" } else { &governed },
    )
}

/// CR-1: arm the one deadline authority from the row's commitments. Called by
/// `router::start` (inside `fail_open`) only when the router ACTS. Returns the
/// lines to print. The frame's scored deadline is the instance deadline the
/// entry point already published (`ny_core::instance_budget`); with none
/// published the authority refuses and every lane keeps today's value.
pub(crate) fn arm(commitments: &Commitments, join: Duration) -> Vec<String> {
    let mut lines = act_lines(commitments);
    let mut record = ActingRecord {
        armed: false,
        refused: None,
        events: Vec::new(),
        events_dropped: 0,
        derived: Vec::new(),
        authority: None,
    };
    if commitments.arm_authority {
        match ny_core::instance_budget::deadline() {
            None => {
                record.refused = Some("no_published_deadline".to_owned());
                lines.push("[router] authority refused reason=no_published_deadline".to_owned());
            }
            Some(hard) => {
                let anchor = Instant::now();
                let backoff = commitments.backoffs_ms().map(Duration::from_millis);
                let frame = Frame::new(anchor, hard, backoff, join);
                match authority().arm(
                    frame,
                    commitments.governed_flags(),
                    commitments.terminal,
                    commitments.grace_ms.map(Duration::from_millis),
                ) {
                    Ok(()) => {
                        record.armed = true;
                        lines.push(armed_line(commitments, &frame));
                    }
                    Err(refusal) => {
                        record.refused = Some(format!("{refusal:?}"));
                        lines.push(format!("[router] authority refused reason={refusal:?}"));
                    }
                }
            }
        }
    }
    *state() = Some(ActingState {
        commitments: commitments.clone(),
        record,
    });
    lines
}

/// The row's acting record, closing the session: snapshot the authority,
/// DISARM it, and hand the record to the flight writer. `None` when the
/// router did not act.
pub(crate) fn finish() -> Option<ActingRecord> {
    let taken = state().take()?;
    let mut record = taken.record;
    if record.armed {
        record.authority = Some(authority().snapshot());
    }
    authority().disarm();
    Some(record)
}

/// The `[router] acting done ...` summary line.
pub(crate) fn done_line(record: &ActingRecord) -> String {
    let (hand_ons, unspent, credited, unrealized, deferred) =
        record.authority.as_ref().map_or((0, 0, 0, 0, 0), |a| {
            (
                a.hand_ons.len(),
                a.unspent_ms,
                a.credited_ms,
                a.unrealized,
                a.deferred_ms,
            )
        });
    format!(
        "[router] acting done armed={} events={} derived_applied={} derived_blocked={} \
         hand_ons={hand_ons} unspent_ms={unspent} credited_ms={credited} unrealized={unrealized} \
         deferred_ms={deferred}",
        u8::from(record.armed),
        record.events.len(),
        record
            .derived
            .iter()
            .filter(|d| d.blocked_by.is_none())
            .count(),
        record
            .derived
            .iter()
            .filter(|d| d.blocked_by.is_some())
            .count(),
    )
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

const fn attack_slot(lane: AllocatedLane) -> LaneSlot {
    match lane {
        AllocatedLane::SignSpace => LaneSlot::AttackSignSpace,
        AllocatedLane::StePgd => LaneSlot::AttackStePgd,
        AllocatedLane::UpfrontAttack => LaneSlot::AttackUpfront,
    }
}

fn commits_attack_slice() -> bool {
    !guard::disabled()
        && state()
            .as_ref()
            .is_some_and(|s| s.commitments.attack_contracts)
}

// Every helper below has a `*_with(&Authority, ...)` core, so the unit tests
// drive a LOCAL authority and never arm the process-global one that other
// in-process tests' lanes would read. The scored path calls the thin
// wrappers, which pass `deadline_authority::global()`.

/// CR-C: whether the router commits the attack slice (it then admits the
/// `#lane-budget-frame` identity plan without that lever). `false` unless the
/// authority is armed and governs an attack lane.
pub(crate) fn governs_attack_slice() -> bool {
    governs_attack_slice_with(authority(), commits_attack_slice())
}

pub(crate) fn governs_attack_slice_with(
    authority: &deadline_authority::Authority,
    attack_contracts: bool,
) -> bool {
    authority.is_armed()
        && attack_contracts
        && LaneSlot::ALL
            .into_iter()
            .any(|slot| slot.in_attack_slice() && authority.governs(slot))
}

/// CR-C: record the committed frame's windows as the attack lanes' contracts.
/// Only a FRAME plan is committed (the knapsack and the in-flight scheduler
/// are stronger cap authorities and stand the router down, exactly as they
/// stand the frame down).
pub(crate) fn commit_attack_contracts(allocation: Option<&AttackSliceAllocation>) {
    commit_attack_contracts_with(authority(), commits_attack_slice(), allocation);
}

pub(crate) fn commit_attack_contracts_with(
    authority: &deadline_authority::Authority,
    attack_contracts: bool,
    allocation: Option<&AttackSliceAllocation>,
) {
    let Some(allocation) = allocation else {
        return;
    };
    if allocation.authority() != AllocationAuthority::Frame
        || !governs_attack_slice_with(authority, attack_contracts)
    {
        return;
    }
    open_or("commit", (), || {
        let now = Instant::now();
        for lane in [
            AllocatedLane::SignSpace,
            AllocatedLane::StePgd,
            AllocatedLane::UpfrontAttack,
        ] {
            let slot = attack_slot(lane);
            let asked = match allocation.window(lane) {
                LaneWindow::Cap(cap) => cap,
                LaneWindow::Skip => Duration::ZERO,
                LaneWindow::Private => continue,
            };
            if let Some(held) = authority.commit_contract(slot, asked, now) {
                event(
                    slot,
                    "commit",
                    format!(
                        "rule=CR-C contract_ms={} asked_ms={} trimmed_ms={}",
                        ms(held),
                        ms(asked),
                        ms(asked.saturating_sub(held))
                    ),
                );
            }
        }
    });
}

/// CR-1/CR-C: the window an attack lane is handed at its own start. The lane
/// claims `now + contract` from its cell and is handed `cell - now`; `Skip`
/// for a zero window. `today` unchanged when the lane is not governed or holds
/// no contract.
pub(crate) fn claim_attack_window(slot: LaneSlot, today: LaneWindow) -> LaneWindow {
    claim_attack_window_with(authority(), slot, today)
}

pub(crate) fn claim_attack_window_with(
    authority: &deadline_authority::Authority,
    slot: LaneSlot,
    today: LaneWindow,
) -> LaneWindow {
    if !authority.governs(slot) || authority.contract(slot).is_none() {
        return today;
    }
    open_or("claim", today, || {
        let now = Instant::now();
        let Some(window) = authority.claim_contract(slot, now) else {
            return today;
        };
        // #router-r4: safe points are enforced. A lane without safe points
        // (the STE-PGD lane's fixed anneal) is never handed a window SHORTER
        // than today's; a raise is never a shortening.
        let declared = super::registry::slot_safe_points(slot);
        let today_ns = match today {
            LaneWindow::Cap(cap) => u64::try_from(cap.as_nanos()).unwrap_or(u64::MAX),
            LaneWindow::Skip | LaneWindow::Private => 0,
        };
        let window_ns = u64::try_from(window.as_nanos()).unwrap_or(u64::MAX);
        let ruling = safepoint::may_shorten(declared, safepoint::At::Start, today_ns, window_ns);
        if let safepoint::Ruling::Refused(why) = ruling {
            event(
                slot,
                "claim",
                format!(
                    "rule=CR-1 window_ms={} today_ms={} shorten=refused reason={why} \
                     safe_points=none",
                    ms(window),
                    today_ns / 1_000_000
                ),
            );
            eprintln!(
                "[router] safepoint refused slot={} at=start reason={why} rule=CR-1",
                slot.token()
            );
            return today;
        }
        let cell_ms = authority
            .deadline(slot)
            .and_then(|d| authority.frame().map(|f| f.offset(d) / 1_000_000));
        event(
            slot,
            "claim",
            format!(
                "rule=CR-1 window_ms={} cell_ms={} today={}",
                ms(window),
                cell_ms.map_or_else(|| "-".to_owned(), |c| c.to_string()),
                match today {
                    LaneWindow::Private => "private".to_owned(),
                    LaneWindow::Skip => "skip".to_owned(),
                    LaneWindow::Cap(cap) => format!("{}ms", ms(cap)),
                }
            ),
        );
        if window.is_zero() {
            LaneWindow::Skip
        } else {
            LaneWindow::Cap(window)
        }
    })
}

/// CR-H: `slot` returned; `decided` when its outcome decides the row. An
/// undecided lane's unspent time is handed to the terminal (or returned to
/// the pool). A no-op when the lane was not governed or never claimed.
pub(crate) fn release_lane(slot: LaneSlot, decided: bool) {
    let _ = release_lane_with(authority(), slot, decided);
}

pub(crate) fn release_lane_with(
    authority: &deadline_authority::Authority,
    slot: LaneSlot,
    decided: bool,
) -> Option<deadline_authority::HandOn> {
    if !authority.governs(slot) {
        return None;
    }
    open_or("release", None, || {
        let hand_on = authority.release(slot, Instant::now(), decided)?;
        event(
            slot,
            "release",
            format!(
                "rule=CR-H decided={} unspent_ms={} credited_ms={} to={}",
                u8::from(decided),
                hand_on.unspent_ns / 1_000_000,
                hand_on.credited_ns / 1_000_000,
                hand_on.to.map_or("pool", LaneSlot::token),
            ),
        );
        Some(hand_on)
    })
}

/// #router-r3 CR-ATK: `slot` stopped at a safe point because its own slope
/// rule judged it flat (`router::ostop::upfront_restart_boundary`); its
/// unspent contract is DEFERRED to the late sat reserve instead of credited to
/// the terminal (the caller holds that reserve out of the verifier's window).
/// A no-op when the lane was not governed or never claimed.
pub(crate) fn release_lane_to_reserve(slot: LaneSlot) {
    let _ = release_lane_to_reserve_with(authority(), slot);
}

pub(crate) fn release_lane_to_reserve_with(
    authority: &deadline_authority::Authority,
    slot: LaneSlot,
) -> Option<deadline_authority::HandOn> {
    if !authority.governs(slot) {
        return None;
    }
    open_or("release_reserve", None, || {
        let hand_on = authority.release_to_reserve(slot, Instant::now())?;
        event(
            slot,
            "release",
            format!(
                "rule=CR-ATK deferred=1 unspent_ms={} credited_ms=0 to=late_sat_reserve",
                hand_on.unspent_ns / 1_000_000,
            ),
        );
        Some(hand_on)
    })
}

/// CR-T: the terminal (internal verifier) claims its cell at its start with
/// the timeout its own rule derived. Returns `timeout_secs` UNCHANGED — the
/// terminal takes the remainder and is never shortened — and records whether
/// every hand-on credit landed inside its window.
pub(crate) fn claim_terminal(timeout_secs: u64) -> u64 {
    claim_terminal_with(authority(), timeout_secs)
}

pub(crate) fn claim_terminal_with(
    authority: &deadline_authority::Authority,
    timeout_secs: u64,
) -> u64 {
    if !authority.governs(LaneSlot::Verifier) {
        return timeout_secs;
    }
    open_or("terminal", timeout_secs, || {
        let now = Instant::now();
        let today = now
            .checked_add(Duration::from_secs(timeout_secs))
            .unwrap_or(now);
        if let Some(claim) = authority.claim_deadline(LaneSlot::Verifier, today, now) {
            event(
                LaneSlot::Verifier,
                "claim",
                format!(
                    "rule=CR-T terminal=1 timeout_s={timeout_secs} credit_ms={} realized={} \
                     repeated={}",
                    ms(claim.credit),
                    u8::from(claim.realized),
                    u8::from(claim.repeated)
                ),
            );
        }
        timeout_secs
    })
}

/// CR-1 for a lane without safe points (the margin-row workers): record the
/// deadline its own rule derived; return it UNCHANGED. It is never extended
/// after its start (ROUTER_PROGRAM §6.4).
pub(crate) fn claim_margin_row(slot: LaneSlot, today: Instant) -> Instant {
    claim_margin_row_with(authority(), slot, today)
}

pub(crate) fn claim_margin_row_with(
    authority: &deadline_authority::Authority,
    slot: LaneSlot,
    today: Instant,
) -> Instant {
    if !authority.governs(slot) {
        return today;
    }
    open_or("margin_row", today, || {
        let now = Instant::now();
        if let Some(claim) = authority.claim_deadline(slot, today, now) {
            event(
                slot,
                "claim",
                format!(
                    "rule=CR-1 safe_points=none remaining_ms={} grace_ms={} repeated={}",
                    ms(today.saturating_duration_since(now)),
                    authority.publish_grace().map_or(0, ms),
                    u8::from(claim.repeated)
                ),
            );
        }
        today
    })
}

/// CR-R: how much longer a collector of a governed margin-row lane waits past
/// that lane's search deadline: the publish grace plus [`COLLECT_SLACK`].
/// `Duration::ZERO` (today) when no grace is armed or the lane is ungoverned.
pub(crate) fn collector_grace(slot: LaneSlot) -> Duration {
    collector_grace_with(authority(), slot)
}

pub(crate) fn collector_grace_with(
    authority: &deadline_authority::Authority,
    slot: LaneSlot,
) -> Duration {
    if !authority.governs(slot) {
        return Duration::ZERO;
    }
    authority
        .publish_grace()
        .map_or(Duration::ZERO, |grace| grace + COLLECT_SLACK)
}

/// CR-R: the concurrent lane's final-join margin below the scored deadline.
/// Today's `margin` unchanged unless a grace is armed; then the join waits
/// until the lane's own publish deadline (`hard - 3 s + g`) plus the collect
/// slack, never waiting LESS than today and always leaving at least the
/// write reserve.
pub(crate) fn final_join_margin(margin: Duration) -> Duration {
    final_join_margin_with(authority(), margin)
}

pub(crate) fn final_join_margin_with(
    authority: &deadline_authority::Authority,
    margin: Duration,
) -> Duration {
    let extra = collector_grace_with(authority, LaneSlot::MarginRowConcurrent);
    if extra.is_zero() {
        return margin;
    }
    let tail = Duration::from_millis(derive::MARGIN_ROW_TAIL_MS);
    let wanted = tail.saturating_sub(extra);
    let floor = Duration::from_millis(derive::WRITE_RESERVE_MS);
    let joined = wanted.max(floor).min(margin);
    eprintln!(
        "[router] lane slot={} event=grace rule=CR-R join_margin_ms={} today_ms={}",
        LaneSlot::MarginRowConcurrent.token(),
        ms(joined),
        ms(margin)
    );
    joined
}

fn directives() -> Option<DerivedPlan> {
    // A contained fault disables the router: fail open to today's values.
    if guard::disabled() {
        return None;
    }
    state().as_ref().map(|s| s.commitments.derived)
}

fn record_derived(entry: DerivedRecord) {
    eprintln!(
        "[router] derived rule={} key={} from={} to={} source=router{}",
        entry.rule,
        entry.key,
        entry.from,
        entry.to,
        entry
            .blocked_by
            .as_deref()
            .map_or_else(String::new, |why| format!(" blocked_by={why}")),
    );
    if let Some(state) = state().as_mut() {
        state.record.derived.push(entry);
    }
}

/// The pins the row's own configuration carries for the derived keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct VerifierPins {
    /// `bab.max_depth` set by the row's configuration.
    pub(crate) max_depth: bool,
    /// `bab.disjunctive_clause_retry_passes` set by the row's configuration.
    pub(crate) clause_passes: bool,
    /// `bab.initial_bounds_warmup_fallback` set by the row's configuration.
    pub(crate) warmup_fallback: bool,
}

impl VerifierPins {
    /// The pins a loaded configuration carries.
    pub(crate) fn of(preset: Option<&crate::preset::PresetConfig>) -> Self {
        preset.map_or_else(Self::default, |p| Self {
            max_depth: p.bab.max_depth.is_some(),
            clause_passes: p.bab.disjunctive_clause_retry_passes.is_some(),
            warmup_fallback: p.bab.initial_bounds_warmup_fallback.is_some(),
        })
    }
}

/// CR-K / CR-H at the verifier's configuration boundary: apply the derived
/// values where the row's own configuration did not pin the key, never below
/// the value already in force. A no-op unless the router acts.
///
/// * `initial_bounds_warmup_fallback` (CR-H phase hand-on): an iterative root
///   warmup that blows its phase cap while the search deadline is live hands
///   its time to the search with the bootstrap enclosure already in hand,
///   instead of returning `unknown`. SOUND BY MONOTONICITY (the fallback
///   enclosure is wider; `BetaCrownConfig::initial_bounds_warmup_fallback`).
/// * `disjunctive_clause_retry_passes` (CR-K): the loop stops when a pass
///   decides nothing and never eats the Graph-MIP-reserving deadline; a retry
///   can never erase a recorded potential violation (`ClauseVerdicts`).
/// * `max_depth` (CR-K): lifted only while a resident-queue byte budget in
///   force fits half the host's static lease ([`derive::depth_lift_gate`]);
///   raising a depth cap cannot change a run that never reached it, and a run
///   that ends `unsat` closed its tree before any cap by definition.
pub(crate) fn apply_verifier_directives(
    config: &mut ny_propagate::BetaCrownConfig,
    pins: VerifierPins,
) {
    let Some(derived) = directives() else {
        return;
    };
    let applied = open_or("derived", Vec::new(), || {
        apply_derived(config, pins, derived)
    });
    for entry in applied {
        record_derived(entry);
    }
}

/// The pure core of [`apply_verifier_directives`]: returns what it applied
/// or blocked, in order.
pub(crate) fn apply_derived(
    config: &mut ny_propagate::BetaCrownConfig,
    pins: VerifierPins,
    derived: DerivedPlan,
) -> Vec<DerivedRecord> {
    let mut out = Vec::new();
    if derived.warmup_hand_on {
        let from = config.initial_bounds_warmup_fallback;
        let blocked = pins.warmup_fallback.then(|| "pin".to_owned());
        if blocked.is_none() {
            config.initial_bounds_warmup_fallback = true;
        }
        out.push(DerivedRecord {
            rule: "CR-H".to_owned(),
            key: "bab.initial_bounds_warmup_fallback".to_owned(),
            from: from.to_string(),
            to: config.initial_bounds_warmup_fallback.to_string(),
            blocked_by: blocked,
        });
    }
    if let Some(passes) = derived.clause_passes {
        let from = config.disjunctive_clause_retry_passes;
        let blocked = pins.clause_passes.then(|| "pin".to_owned());
        if blocked.is_none() {
            let passes = usize::try_from(passes).unwrap_or(usize::MAX);
            config.disjunctive_clause_retry_passes = from.max(passes);
        }
        out.push(DerivedRecord {
            rule: "CR-K".to_owned(),
            key: "bab.disjunctive_clause_retry_passes".to_owned(),
            from: from.to_string(),
            to: config.disjunctive_clause_retry_passes.to_string(),
            blocked_by: blocked,
        });
    }
    if let Some(depth) = derived.max_depth {
        let from = config.max_depth;
        let gate = derive::depth_lift_gate(
            u64::try_from(config.max_queue_bytes).unwrap_or(u64::MAX),
            derived.lease_bytes,
        );
        let blocked = if pins.max_depth {
            Some("pin".to_owned())
        } else if gate != DepthGate::Admit {
            Some(gate.token().to_owned())
        } else {
            None
        };
        if blocked.is_none() {
            let depth = usize::try_from(depth).unwrap_or(usize::MAX);
            config.max_depth = from.max(depth);
        }
        out.push(DerivedRecord {
            rule: "CR-K".to_owned(),
            key: "bab.max_depth".to_owned(),
            from: from.to_string(),
            to: config.max_depth.to_string(),
            blocked_by: blocked,
        });
    }
    out
}

/// Slot count sanity (the authority's arrays and the commitments agree).
const _: () = assert!(SLOTS == LaneSlot::ALL.len());
