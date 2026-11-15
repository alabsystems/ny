// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 CR-L — the `ny-cli` half of the MEMORY LEASE (ROUTER_PROGRAM
//! §10 R4, F6).
//!
//! `ny_core::router::r4::commit` decides, purely, whether CR-L acts and with
//! which [`LeasePlan`] (the declared host's static envelope);
//! `ny_core::memory_lease` holds the ledger. This module is the glue between
//! them and the optional-heavy memory gates of `ny vnncomp`:
//!
//! | site | helper | holder |
//! |---|---|---|
//! | `router::start` (acting) | [`arm`] | the terminal's base lease |
//! | the concurrent margin-row spawn decision | [`admit`] / [`release`] (not spawned, finished) | `margin_row_concurrent` |
//! | the internal verifier returns | [`release`] | `terminal` |
//! | the inline margin-row lane | [`admit`] / [`release`] (finished) | `margin_row_inline` |
//! | CR-ATK's late-reserve predicate, the held-reserve recheck | [`peek`] | `post_verifier_tail` |
//! | every post-verifier graph-heavy stage, the f64 oracle | [`admit`] | `post_verifier_tail` |
//! | the post-verifier attack returns | [`release`] | `post_verifier_tail` |
//! | the batched ORT VJP refine wave | [`admit`] / [`ReleaseOnDrop`] | `ort_refine_wave` |
//! | the kFSB single-shot fan-out (`ny-propagate` `gpu_bab::kfsb_fanout`, directly on the authority: asked once per armed graph lane, narrowable, returned when the lane ends) | — | `kfsb_fanout` |
//! | the concurrent upfront-attack lane's admission (`beta_crown::attack_concurrent::admit_site`, only when `attack.concurrent` is armed; floor = the live optional-heavy gate) | [`admit`] / [`release`] (join, or a pool that failed to build) | `attack_concurrent` |
//! | `finish_after_publication` | [`finish`] | snapshot + DISARM |
//!
//! # The floor, and bit-identity
//!
//! Every gate passes TODAY's live decision in as `floor` (the existing
//! `optional_heavy_memory_allowed_now`: cgroup/RLIMIT headroom >= 16 GiB, the
//! operator override, `Unavailable` fails closed). Unarmed — every run
//! without `router.mode: on` — the helper returns `floor()` exactly, evaluated
//! exactly once: today's gate, bit for bit. Armed, it returns
//! `floor() && lease`: the kernel-enforced envelope stays a FLOOR the lease
//! can never widen, and the lease replaces the one arm of today's gate that
//! decided nothing — `Unbounded => true`, i.e. every competition run, which
//! has no cgroup — with an aggregate reservation against the declared host.
//!
//! Every entry runs inside `guard::fail_open`: a panic is contained, disarms
//! the authority, and the gate returns `floor`.

use std::sync::{Mutex, PoisonError};

use ny_core::memory_lease::{self, LeaseAuthority, LeaseSnapshot};
use ny_core::router::guard;
use ny_core::router::lease::{LeaseDecision, LeaseHolder, LeasePlan};
use serde::{Deserialize, Serialize};

/// The lease half of the flight `router` record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct LeaseRecord {
    /// Whether the authority armed.
    pub(crate) armed: bool,
    /// Why it refused to arm, when it did.
    pub(crate) refused: Option<String>,
    /// The authority at the end of the row.
    pub(crate) snapshot: Option<LeaseSnapshot>,
}

static STATE: Mutex<Option<LeaseRecord>> = Mutex::new(None);

fn state() -> std::sync::MutexGuard<'static, Option<LeaseRecord>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

const GIB: f64 = (1u64 << 30) as f64;

fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / GIB)
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

/// `[router] lease armed ...`.
fn armed_line(plan: &LeasePlan, terminal: LeaseDecision) -> String {
    format!(
        "[router] lease armed rule=CR-L envelope_gib={} host={} threads={} dev={} declared={} \
         terminal={}:{}gib heavy_lane_gib={} floor=live_envelope",
        gib(plan.envelope_bytes),
        plan.host.token(),
        plan.threads,
        plan.device.token(),
        u8::from(plan.declared),
        terminal.token(),
        gib(terminal.bytes()),
        gib(ny_core::router::lease::OPTIONAL_HEAVY_LEASE_BYTES),
    )
}

/// CR-L: arm the process's lease authority from the row's plan. Called by
/// `router::start` (inside `fail_open`) only when the router ACTS and CR-L
/// acts. Returns the lines to print.
pub(crate) fn arm(plan: Option<&LeasePlan>) -> Vec<String> {
    let (record, lines) = arm_with(memory_lease::global(), plan);
    *state() = Some(record);
    lines
}

pub(crate) fn arm_with(
    authority: &LeaseAuthority,
    plan: Option<&LeasePlan>,
) -> (LeaseRecord, Vec<String>) {
    let mut record = LeaseRecord {
        armed: false,
        refused: None,
        snapshot: None,
    };
    let mut lines = Vec::new();
    match plan {
        None => lines.push("[router] lease idle rule=CR-L reason=not_acting".to_owned()),
        Some(plan) => match authority.arm(*plan) {
            Ok(terminal) => {
                record.armed = true;
                lines.push(armed_line(plan, terminal));
            }
            Err(refusal) => {
                record.refused = Some(format!("{refusal:?}"));
                lines.push(format!(
                    "[router] lease refused rule=CR-L reason={refusal:?}"
                ));
            }
        },
    }
    (record, lines)
}

/// Close the row's lease session: snapshot the authority, DISARM it, and
/// hand the record to the flight writer. `None` when CR-L did not arm here.
pub(crate) fn finish() -> Option<LeaseRecord> {
    let mut record = state().take()?;
    let authority = memory_lease::global();
    if record.armed {
        record.snapshot = Some(authority.snapshot());
    }
    authority.disarm();
    Some(record)
}

/// The `[router] lease done ...` summary line.
pub(crate) fn done_line(record: &LeaseRecord) -> String {
    let s = record.snapshot.as_ref();
    format!(
        "[router] lease done armed={} granted={} narrowed={} denied={} released={} \
         outstanding_gib={} peak_gib={} envelope_gib={} dropped={}",
        u8::from(record.armed),
        s.map_or(0, |s| s.granted),
        s.map_or(0, |s| s.narrowed),
        s.map_or(0, |s| s.denied),
        s.map_or(0, |s| s.released),
        gib(s.map_or(0, |s| s.outstanding_bytes)),
        gib(s.map_or(0, |s| s.peak_bytes)),
        gib(s.and_then(|s| s.plan).map_or(0, |p| p.envelope_bytes)),
        s.map_or(0, |s| s.dropped),
    )
}

fn decision_line(holder: LeaseHolder, op: &str, decision: LeaseDecision) -> String {
    let (available, want) = match decision {
        LeaseDecision::Denied { available, want } => (Some(available), want),
        LeaseDecision::Narrowed { want, .. } => (None, want),
        LeaseDecision::Granted { bytes } => (None, bytes),
    };
    format!(
        "[router] lease holder={} op={op} decision={} bytes_gib={} want_gib={}{} rule=CR-L",
        holder.token(),
        decision.token(),
        gib(decision.bytes()),
        gib(want),
        available.map_or_else(String::new, |a| format!(" available_gib={}", gib(a))),
    )
}

/// The optional-heavy gate for `holder`: today's live decision (`floor`)
/// unchanged when no lease is armed; `floor() && lease` when one is. A newly
/// decided lease prints its line; a holder re-checking a lease it already
/// holds (a lane's later stages) prints nothing and is never debited twice.
pub(crate) fn admit(holder: LeaseHolder, floor: impl FnOnce() -> bool) -> bool {
    admit_with(memory_lease::global(), holder, floor)
}

pub(crate) fn admit_with(
    authority: &LeaseAuthority,
    holder: LeaseHolder,
    floor: impl FnOnce() -> bool,
) -> bool {
    if !authority.is_armed() {
        return floor();
    }
    let live = floor();
    if !live {
        // The kernel-enforced envelope refuses: the lease is never asked, so
        // nothing is reserved for a lane that will not run.
        return false;
    }
    open_or("lease", live, || {
        let fresh = authority.held(holder) == 0;
        match authority.acquire(holder) {
            None => live,
            Some(decision) => {
                if fresh {
                    eprintln!("{}", decision_line(holder, "acquire", decision));
                }
                decision.admits()
            }
        }
    })
}

/// Would `holder`'s lease be granted now? Reserves nothing (a site that only
/// decides whether to HOLD TIME for a later heavy stage). Today's `floor()`
/// unchanged when no lease is armed.
pub(crate) fn peek(holder: LeaseHolder, floor: impl FnOnce() -> bool) -> bool {
    peek_with(memory_lease::global(), holder, floor)
}

pub(crate) fn peek_with(
    authority: &LeaseAuthority,
    holder: LeaseHolder,
    floor: impl FnOnce() -> bool,
) -> bool {
    if !authority.is_armed() {
        return floor();
    }
    let live = floor();
    if !live {
        return false;
    }
    open_or("lease", live, || {
        authority
            .peek(holder)
            .map_or(live, |decision| decision.admits())
    })
}

/// `holder`'s working set is gone: return its lease (a no-op when no lease
/// is armed or none is held).
pub(crate) fn release(holder: LeaseHolder) {
    release_with(memory_lease::global(), holder);
}

pub(crate) fn release_with(authority: &LeaseAuthority, holder: LeaseHolder) -> Option<u64> {
    if !authority.is_armed() {
        return None;
    }
    open_or("lease", None, || {
        let returned = authority.release(holder)?;
        eprintln!(
            "[router] lease holder={} op=release bytes_gib={} rule=CR-L",
            holder.token(),
            gib(returned)
        );
        Some(returned)
    })
}

/// Releases `holder`'s lease when dropped: for a lane with many return
/// paths (the ORT VJP refine wave). A no-op when nothing is held.
pub(crate) struct ReleaseOnDrop(pub(crate) LeaseHolder);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        release(self.0);
    }
}
