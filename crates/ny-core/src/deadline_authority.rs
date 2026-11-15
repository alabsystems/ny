// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r2 CR-1 — THE ONE DEADLINE AUTHORITY (ROUTER_PROGRAM §2.3, §6.4).
//!
//! # What this is
//!
//! Before R2 every scored lane sized itself from the LIVE clock at the moment
//! it started (`post_beta_deadline`, `instance_deadline`, `remaining - 45 s`,
//! `8 %` of what is left ...). That is how a lane that declined handed its
//! seconds to whichever lane read the clock next, in proportion to that lane's
//! own coefficient (measured on traffic: 327.4 s freed, the upfront attack grew
//! itself 71.0 -> 234.7 s, only ~150 s reached branch-and-bound;
//! `lane_allocation.rs` `SizingRule`). `#lane-budget-frame` fixed the attack
//! slice alone by deciding its caps once, up front.
//!
//! This module generalises that frame into ONE authority over every lane the
//! acting router governs:
//!
//! * a [`Frame`]: the scored deadline `hard`, the anchor, and a per-slot CAP —
//!   the latest instant that slot's cell may ever hold;
//! * one [`LaneDeadline`] cell per [`LaneSlot`]: a MONOTONE atomic. A write
//!   saturates into `[current, cap]`: a cell can be extended toward its cap,
//!   never past it, and never shortened. A contracted lane reads its window
//!   from its cell at its own start; it no longer sizes itself from the live
//!   clock while the authority governs it. A lane that keeps its OWN deadline
//!   (the terminal, and the margin-row lanes, which have no safe points) is
//!   recorded exactly as its rule derived it — the authority never shortens
//!   a lane's own rule — clamped only to the scored deadline;
//! * CONTRACTS (CR-C): a non-terminal lane's committed duration, decided once;
//!   the lane claims its cell at its own start (`now + contract`, capped). The
//!   sum of committed contracts can never exceed the slot's window at commit
//!   time (trimmed, and the trim recorded) — conservation by construction;
//! * the TERMINAL (CR-T): the lane that takes the remainder. It is never
//!   contracted and never shortened: its claim records the deadline its own
//!   rule already derives from the live clock, which is exactly where every
//!   unspent contract lands;
//! * HAND-ON (CR-H): a lane that ends undecided with contract left RELEASES
//!   it; the unspent time is credited to the terminal when the terminal has
//!   not started (it then reaches the terminal through the live clock the
//!   terminal's own rule reads, and the claim checks that it did: `realized`),
//!   and returned to the pool otherwise. A lane with no safe points is never
//!   extended after it started. The ledger records every hand-on, so
//!   `credited <= unspent` and "nothing handed on was lost" are checkable
//!   facts, not comments;
//! * the PUBLISH GRACE (CR-R): a bounded extension, `0 < g < 3 s` and below the
//!   join margin, that a COMPLETED certificate may use to publish after its
//!   lane's search deadline. Search never uses it.
//!
//! # Dark by default, fail open
//!
//! Nothing here is armed unless the router is ACTING (`router.mode: on`, whose
//! declaration default is `off`). Unarmed, every read returns `None` and every
//! caller keeps exactly the value it computes today, so an unarmed process is
//! byte-identical to a binary without this module. A router fault disarms it
//! (`router::guard::fail_open`), which is the same `None` everywhere.
//!
//! # Soundness
//!
//! Scheduling only. A cell is an `Instant`; it never touches a bound, a
//! certificate, a verdict or the publisher, and no module here can name one
//! (the router ratchets). A wrong cell can cost a proof (a lane stopped
//! earlier than it could have been) — never mint one. The publish grace lets
//! an ALREADY-COMPLETE certificate be collected; it does not extend search,
//! and it is bounded below the join margin so publication still lands inside
//! the scored budget.
//!
//! # Never panics
//!
//! Every index is `get`, every arithmetic step saturates, every lock tolerates
//! poison. It is read from lane threads and from `ny-propagate`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Number of governed lane slots.
pub const SLOTS: usize = 6;

/// The publish grace must stay strictly below this (the margin-row lane's own
/// 3 s wrap-up reserve; `margin_row::PUBLISH_GRACE_SECS` doc).
pub const PUBLISH_GRACE_CEILING: Duration = Duration::from_secs(3);

/// Hand-on events one row may record.
pub const MAX_LEDGER: usize = 64;

/// A lane the authority can govern. A FIXED set, so every cell is a static
/// atomic and a lane thread never allocates to read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneSlot {
    /// The LP sign-space attack lane (attack slice, contracted).
    AttackSignSpace,
    /// The STE-PGD attack lane (attack slice, contracted).
    AttackStePgd,
    /// The upfront DLR-APGD attack lane (attack slice, contracted).
    AttackUpfront,
    /// The internal verifier: the TERMINAL, takes the remainder.
    Verifier,
    /// The concurrent margin-row worker (no safe points after spawn).
    MarginRowConcurrent,
    /// The post-verifier inline margin-row lane (no safe points after start).
    MarginRowInline,
}

impl LaneSlot {
    /// Every slot, in index order.
    pub const ALL: [Self; SLOTS] = [
        Self::AttackSignSpace,
        Self::AttackStePgd,
        Self::AttackUpfront,
        Self::Verifier,
        Self::MarginRowConcurrent,
        Self::MarginRowInline,
    ];

    /// The slot's index into the cell arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable token for logs and records.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::AttackSignSpace => "attack_sign_space",
            Self::AttackStePgd => "attack_ste_pgd",
            Self::AttackUpfront => "attack_upfront",
            Self::Verifier => "verifier",
            Self::MarginRowConcurrent => "margin_row_concurrent",
            Self::MarginRowInline => "margin_row_inline",
        }
    }

    /// Parse a token.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|slot| slot.token() == token)
    }

    /// Whether the lane has no safe points once started (the detached
    /// margin-row workers): it keeps the deadline it started with and is
    /// NEVER extended after its claim (ROUTER_PROGRAM §6.4).
    #[must_use]
    pub const fn uncancellable(self) -> bool {
        matches!(self, Self::MarginRowConcurrent | Self::MarginRowInline)
    }

    /// Whether the lane belongs to the contracted attack slice.
    #[must_use]
    pub const fn in_attack_slice(self) -> bool {
        matches!(
            self,
            Self::AttackSignSpace | Self::AttackStePgd | Self::AttackUpfront
        )
    }
}

/// Nanoseconds in a duration, saturating.
fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// One MONOTONE deadline cell: nanoseconds after the frame anchor, stored
/// plus one so that zero means "never written".
///
/// The only write is [`LaneDeadline::raise`], a compare-and-swap loop that
/// stores `max(current, min(requested, cap))`. Concurrent readers therefore
/// observe a non-decreasing sequence bounded by the cap, whatever the
/// interleaving (`lane_deadline_cell_is_monotone_under_concurrent_readers`).
#[derive(Debug, Default)]
pub struct LaneDeadline {
    raw: AtomicU64,
}

impl LaneDeadline {
    /// An unwritten cell.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            raw: AtomicU64::new(0),
        }
    }

    /// The held offset, or `None` when never written.
    #[must_use]
    pub fn offset_ns(&self) -> Option<u64> {
        match self.raw.load(Ordering::Acquire) {
            0 => None,
            raw => Some(raw - 1),
        }
    }

    /// Raise the cell toward `requested_ns`, never past `cap_ns` and never
    /// below its current value. Returns the offset the cell now holds.
    pub fn raise(&self, requested_ns: u64, cap_ns: u64) -> u64 {
        let target = requested_ns.min(cap_ns).saturating_add(1);
        let mut current = self.raw.load(Ordering::Acquire);
        loop {
            if current >= target {
                return current.saturating_sub(1);
            }
            match self.raw.compare_exchange_weak(
                current,
                target,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return target - 1,
                Err(actual) => current = actual,
            }
        }
    }

    /// Forget the cell (arming a new frame).
    pub fn reset(&self) {
        self.raw.store(0, Ordering::Release);
    }
}

/// The frame every cell lives in: one anchor, the scored deadline, and each
/// slot's cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    anchor: Instant,
    hard: Instant,
    caps_ns: [u64; SLOTS],
    join: Duration,
}

impl Frame {
    /// A frame anchored at `anchor` for the scored deadline `hard`. Each
    /// slot's cap is `hard - backoff[slot]` (never before the anchor, never
    /// after `hard`); `join` is the concurrent-lane join margin, the
    /// publication window the publish grace must stay below.
    #[must_use]
    pub fn new(anchor: Instant, hard: Instant, backoff: [Duration; SLOTS], join: Duration) -> Self {
        let hard_ns = nanos(hard.saturating_duration_since(anchor));
        let mut caps_ns = [0u64; SLOTS];
        for (cap, back) in caps_ns.iter_mut().zip(backoff) {
            *cap = hard_ns.saturating_sub(nanos(back));
        }
        Self {
            anchor,
            hard,
            caps_ns,
            join,
        }
    }

    /// The anchor.
    #[must_use]
    pub const fn anchor(&self) -> Instant {
        self.anchor
    }

    /// The scored deadline.
    #[must_use]
    pub const fn hard(&self) -> Instant {
        self.hard
    }

    /// The join margin.
    #[must_use]
    pub const fn join(&self) -> Duration {
        self.join
    }

    /// The scored deadline as an offset.
    #[must_use]
    pub fn hard_ns(&self) -> u64 {
        nanos(self.hard.saturating_duration_since(self.anchor))
    }

    /// `slot`'s cap as an offset.
    #[must_use]
    pub fn cap_ns(&self, slot: LaneSlot) -> u64 {
        self.caps_ns.get(slot.index()).copied().unwrap_or(0)
    }

    /// `at` as an offset (saturating at zero before the anchor).
    #[must_use]
    pub fn offset(&self, at: Instant) -> u64 {
        nanos(at.saturating_duration_since(self.anchor))
    }

    /// The instant at `offset_ns` (the scored deadline if unrepresentable).
    #[must_use]
    pub fn at(&self, offset_ns: u64) -> Instant {
        self.anchor
            .checked_add(Duration::from_nanos(offset_ns))
            .unwrap_or(self.hard)
    }
}

/// Why a frame was refused (the authority then stays unarmed: today's route).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmRefusal {
    /// The publish grace is not strictly below [`PUBLISH_GRACE_CEILING`].
    GraceNotBelowCeiling,
    /// The publish grace is not strictly below the join margin.
    GraceNotBelowJoin,
    /// A slot's cap lies after the scored deadline.
    CapPastHard,
}

/// One hand-on (CR-H).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandOn {
    /// The lane that ended undecided.
    pub from: LaneSlot,
    /// The recipient (`None`: returned to the pool the live clock carries).
    pub to: Option<LaneSlot>,
    /// Its unspent contract, nanoseconds.
    pub unspent_ns: u64,
    /// What the recipient was credited, nanoseconds (`<= unspent_ns`).
    pub credited_ns: u64,
    /// When, as an offset.
    pub at_ns: u64,
    /// #router-r3 CR-ATK: the unspent time was DEFERRED to the late sat
    /// reserve (held out of the terminal's window and spent by the late
    /// attack) instead of credited to the terminal. `credited_ns` is then 0.
    #[serde(default)]
    pub deferred: bool,
    /// #router-r4 CR-F: the lane stopped at a safe point because its progress
    /// stream read FAR, and its unspent contract was handed to an explicit
    /// sealed recipient slot (`to`) rather than to the terminal.
    #[serde(default)]
    pub far: bool,
}

/// The authority's state for one slot, as recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotSnapshot {
    /// The slot.
    pub slot: LaneSlot,
    /// Whether the authority governs it.
    pub governed: bool,
    /// Its cap, milliseconds after the anchor.
    pub cap_ms: u64,
    /// Its committed contract, milliseconds (`None`: none committed).
    pub contract_ms: Option<u64>,
    /// Milliseconds trimmed off a contract to keep the sum inside the window.
    pub trimmed_ms: u64,
    /// When it claimed its cell, milliseconds after the anchor.
    pub claimed_ms: Option<u64>,
    /// Its cell, milliseconds after the anchor.
    pub cell_ms: Option<u64>,
    /// When it released, milliseconds after the anchor.
    pub released_ms: Option<u64>,
    /// Hand-on credit it received, milliseconds.
    pub credit_ms: u64,
}

/// The whole authority, as recorded in the flight `router` record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthoritySnapshot {
    /// Whether it was armed when the snapshot was taken.
    pub armed: bool,
    /// The scored deadline, milliseconds after the anchor.
    pub hard_ms: u64,
    /// The join margin, milliseconds.
    pub join_ms: u64,
    /// The publish grace, milliseconds (`None`: not armed).
    pub grace_ms: Option<u64>,
    /// The terminal slot.
    pub terminal: Option<LaneSlot>,
    /// Every slot.
    pub slots: Vec<SlotSnapshot>,
    /// Every hand-on, in order.
    pub hand_ons: Vec<HandOn>,
    /// Sum of unspent contracts released.
    pub unspent_ms: u64,
    /// Sum credited to a recipient.
    pub credited_ms: u64,
    /// #router-r3: sum DEFERRED to the late sat reserve (CR-ATK).
    #[serde(default)]
    pub deferred_ms: u64,
    /// Hand-ons whose credit the terminal's claim did NOT find in its window.
    pub unrealized: u64,
    /// Hand-ons past [`MAX_LEDGER`] (counted, not recorded).
    pub ledger_dropped: u64,
}

/// A claim for the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalClaim {
    /// The deadline the terminal keeps (exactly the one it asked for).
    pub deadline: Instant,
    /// Hand-on credit banked for it.
    pub credit: Duration,
    /// Whether that credit is inside its window (`deadline - now >= credit`).
    pub realized: bool,
    /// Whether the slot had already claimed (the first claim's cell stands).
    pub repeated: bool,
}

const fn cell_array() -> [LaneDeadline; SLOTS] {
    [const { LaneDeadline::new() }; SLOTS]
}

const fn u64_array() -> [AtomicU64; SLOTS] {
    [const { AtomicU64::new(0) }; SLOTS]
}

const fn bool_array() -> [AtomicBool; SLOTS] {
    [const { AtomicBool::new(false) }; SLOTS]
}

#[derive(Debug, Default)]
struct Ledger {
    events: Vec<HandOn>,
    dropped: u64,
}

/// The deadline authority. One per process ([`global`]); instantiable for
/// tests.
#[derive(Debug)]
pub struct Authority {
    armed: AtomicBool,
    frame: RwLock<Option<Frame>>,
    cells: [LaneDeadline; SLOTS],
    /// Committed contract, nanoseconds + 1 (0 = none).
    contracts: [AtomicU64; SLOTS],
    trimmed: [AtomicU64; SLOTS],
    governed: [AtomicBool; SLOTS],
    /// Claim instant, offset + 1 (0 = not claimed).
    claimed: [AtomicU64; SLOTS],
    /// Release instant, offset + 1 (0 = not released).
    released: [AtomicU64; SLOTS],
    /// Hand-on credit received, nanoseconds.
    credit: [AtomicU64; SLOTS],
    /// Sum of committed contracts, nanoseconds.
    committed_total: AtomicU64,
    /// Terminal slot index + 1 (0 = none).
    terminal: AtomicU64,
    grace_ns: AtomicU64,
    unrealized: AtomicU64,
    ledger: Mutex<Ledger>,
}

impl Default for Authority {
    fn default() -> Self {
        Self::new()
    }
}

impl Authority {
    /// An unarmed authority.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            frame: RwLock::new(None),
            cells: cell_array(),
            contracts: u64_array(),
            trimmed: u64_array(),
            governed: bool_array(),
            claimed: u64_array(),
            released: u64_array(),
            credit: u64_array(),
            committed_total: AtomicU64::new(0),
            terminal: AtomicU64::new(0),
            grace_ns: AtomicU64::new(0),
            unrealized: AtomicU64::new(0),
            ledger: Mutex::new(Ledger {
                events: Vec::new(),
                dropped: 0,
            }),
        }
    }

    fn slot_u64(array: &[AtomicU64; SLOTS], slot: LaneSlot) -> u64 {
        array
            .get(slot.index())
            .map_or(0, |cell| cell.load(Ordering::Acquire))
    }

    fn store_u64(array: &[AtomicU64; SLOTS], slot: LaneSlot, value: u64) {
        if let Some(cell) = array.get(slot.index()) {
            cell.store(value, Ordering::Release);
        }
    }

    /// Arm a frame. Clears every cell, contract, claim and the ledger first.
    /// `governed[slot]` says which lanes consult the authority (CR-A decides
    /// it); `terminal` is the lane that takes the remainder (CR-T); `grace`
    /// is the publish grace (CR-R), which must be strictly below both
    /// [`PUBLISH_GRACE_CEILING`] and the frame's join margin.
    pub fn arm(
        &self,
        frame: Frame,
        governed: [bool; SLOTS],
        terminal: Option<LaneSlot>,
        grace: Option<Duration>,
    ) -> Result<(), ArmRefusal> {
        if let Some(grace) = grace {
            if grace >= PUBLISH_GRACE_CEILING {
                return Err(ArmRefusal::GraceNotBelowCeiling);
            }
            if grace >= frame.join() {
                return Err(ArmRefusal::GraceNotBelowJoin);
            }
        }
        let hard_ns = frame.hard_ns();
        if LaneSlot::ALL
            .into_iter()
            .any(|slot| frame.cap_ns(slot) > hard_ns)
        {
            return Err(ArmRefusal::CapPastHard);
        }
        self.disarm();
        for slot in LaneSlot::ALL {
            if let Some(cell) = self.cells.get(slot.index()) {
                cell.reset();
            }
            Self::store_u64(&self.contracts, slot, 0);
            Self::store_u64(&self.trimmed, slot, 0);
            Self::store_u64(&self.claimed, slot, 0);
            Self::store_u64(&self.released, slot, 0);
            Self::store_u64(&self.credit, slot, 0);
            if let Some(flag) = self.governed.get(slot.index()) {
                flag.store(
                    governed.get(slot.index()).copied().unwrap_or(false),
                    Ordering::Release,
                );
            }
        }
        self.committed_total.store(0, Ordering::Release);
        self.unrealized.store(0, Ordering::Release);
        self.terminal.store(
            terminal.map_or(0, |slot| slot.index() as u64 + 1),
            Ordering::Release,
        );
        self.grace_ns
            .store(grace.map_or(0, nanos), Ordering::Release);
        {
            let mut ledger = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
            ledger.events.clear();
            ledger.dropped = 0;
        }
        *self.frame.write().unwrap_or_else(PoisonError::into_inner) = Some(frame);
        self.armed.store(true, Ordering::Release);
        Ok(())
    }

    /// Disarm: every read returns `None` from here on (today's values).
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
        self.grace_ns.store(0, Ordering::Release);
    }

    /// Whether a frame is armed.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// The armed frame.
    #[must_use]
    pub fn frame(&self) -> Option<Frame> {
        if !self.is_armed() {
            return None;
        }
        self.frame.read().ok().and_then(|frame| *frame)
    }

    /// Whether the authority is armed and governs `slot`.
    #[must_use]
    pub fn governs(&self, slot: LaneSlot) -> bool {
        self.is_armed()
            && self
                .governed
                .get(slot.index())
                .is_some_and(|flag| flag.load(Ordering::Acquire))
    }

    /// The terminal slot, when armed.
    #[must_use]
    pub fn terminal(&self) -> Option<LaneSlot> {
        if !self.is_armed() {
            return None;
        }
        match self.terminal.load(Ordering::Acquire) {
            0 => None,
            index => LaneSlot::ALL.get((index - 1) as usize).copied(),
        }
    }

    /// CR-C: commit `slot`'s contract, decided once at `now`. The sum of
    /// committed contracts never exceeds the slot's window at commit
    /// (`cap - now`): a contract that would overflow it is TRIMMED to fit and
    /// the trim recorded. A second commit may only raise (monotone), inside
    /// the same bound. Returns the contract held, or `None` when the slot is
    /// not governed (the caller keeps today's window).
    pub fn commit_contract(
        &self,
        slot: LaneSlot,
        contract: Duration,
        now: Instant,
    ) -> Option<Duration> {
        if !self.governs(slot) || self.terminal() == Some(slot) {
            return None;
        }
        let frame = self.frame()?;
        let window = frame.cap_ns(slot).saturating_sub(frame.offset(now));
        let held = Self::slot_u64(&self.contracts, slot).saturating_sub(1);
        let had = Self::slot_u64(&self.contracts, slot) != 0;
        let asked = nanos(contract);
        if had && asked <= held {
            return Some(Duration::from_nanos(held));
        }
        let others = self
            .committed_total
            .load(Ordering::Acquire)
            .saturating_sub(if had { held } else { 0 });
        let room = window.saturating_sub(others);
        let granted = asked.min(room);
        let trimmed = asked.saturating_sub(granted);
        Self::store_u64(&self.contracts, slot, granted.saturating_add(1));
        Self::store_u64(&self.trimmed, slot, trimmed);
        self.committed_total
            .store(others.saturating_add(granted), Ordering::Release);
        Some(Duration::from_nanos(granted))
    }

    /// `slot`'s committed contract.
    #[must_use]
    pub fn contract(&self, slot: LaneSlot) -> Option<Duration> {
        if !self.governs(slot) {
            return None;
        }
        match Self::slot_u64(&self.contracts, slot) {
            0 => None,
            raw => Some(Duration::from_nanos(raw - 1)),
        }
    }

    /// CR-C/CR-1: a contracted lane claims its cell at its own start `now`:
    /// `cell = min(now + contract + credit, cap)` (raised, never lowered).
    /// Returns the window the lane is handed (`cell - now`; zero means SKIP),
    /// or `None` when the slot is not governed or holds no contract.
    pub fn claim_contract(&self, slot: LaneSlot, now: Instant) -> Option<Duration> {
        let contract = self.contract(slot)?;
        let frame = self.frame()?;
        let start = frame.offset(now);
        let credit = Self::slot_u64(&self.credit, slot);
        let requested = start.saturating_add(nanos(contract)).saturating_add(credit);
        let cell = self.cells.get(slot.index())?;
        let held = cell.raise(requested, frame.cap_ns(slot));
        Self::store_u64(&self.claimed, slot, start.saturating_add(1));
        Some(Duration::from_nanos(held.saturating_sub(start)))
    }

    /// CR-T / uncancellable lanes: record the deadline `today` the lane's own
    /// rule derives. The authority NEVER shortens it and never extends it:
    /// the lane is handed `today` back unchanged (bit-identical), and the cell
    /// records it (clamped only to the scored deadline, which is the cell's
    /// hard ceiling). The FIRST claim wins: a lane without safe points keeps
    /// the deadline it started with, so a repeated claim neither raises nor
    /// lowers the recorded cell. Returns `None` when the slot is not governed
    /// (the caller keeps `today`).
    pub fn claim_deadline(
        &self,
        slot: LaneSlot,
        today: Instant,
        now: Instant,
    ) -> Option<TerminalClaim> {
        if !self.governs(slot) {
            return None;
        }
        let frame = self.frame()?;
        let start = frame.offset(now);
        let wanted = frame.offset(today).min(frame.hard_ns());
        let credit = Self::slot_u64(&self.credit, slot);
        let first = self
            .claimed
            .get(slot.index())?
            .compare_exchange(
                0,
                start.saturating_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        if !first {
            return Some(TerminalClaim {
                deadline: today,
                credit: Duration::from_nanos(credit),
                realized: true,
                repeated: true,
            });
        }
        let cell = self.cells.get(slot.index())?;
        let held = cell.raise(wanted, frame.cap_ns(slot).max(wanted));
        let window = held.saturating_sub(start);
        let realized = window >= credit;
        if !realized {
            self.unrealized.fetch_add(1, Ordering::AcqRel);
        }
        Some(TerminalClaim {
            deadline: today,
            credit: Duration::from_nanos(credit),
            realized,
            repeated: false,
        })
    }

    /// CR-H: `slot` ended at `now`. When it ended UNDECIDED with contract
    /// left, hand the unspent time on: credit it to the terminal if the
    /// terminal has not started (a started terminal without safe points is
    /// never extended), else return it to the pool. A decided lane ends the
    /// row, so nothing is handed on. Returns the recorded hand-on.
    pub fn release(&self, slot: LaneSlot, now: Instant, decided: bool) -> Option<HandOn> {
        if !self.governs(slot) {
            return None;
        }
        let frame = self.frame()?;
        let at = frame.offset(now);
        let released = self.released.get(slot.index())?;
        if released
            .compare_exchange(0, at.saturating_add(1), Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None; // released once already
        }
        if decided || Self::slot_u64(&self.claimed, slot) == 0 {
            return None;
        }
        let cell = self.cells.get(slot.index())?.offset_ns()?;
        let unspent = cell.saturating_sub(at);
        if unspent == 0 {
            return None;
        }
        let recipient = self.terminal().filter(|terminal| {
            *terminal != slot
                && self.governs(*terminal)
                && Self::slot_u64(&self.claimed, *terminal) == 0
                && Self::slot_u64(&self.released, *terminal) == 0
        });
        let credited = recipient.map_or(0, |terminal| {
            // The credit can never carry the recipient past its own cap:
            // what fits between now and the cap is the most it could use.
            let room = frame.cap_ns(terminal).saturating_sub(at);
            let before = Self::slot_u64(&self.credit, terminal);
            let credited = unspent.min(room.saturating_sub(before));
            Self::store_u64(&self.credit, terminal, before.saturating_add(credited));
            credited
        });
        let event = HandOn {
            from: slot,
            to: recipient.filter(|_| credited > 0),
            unspent_ns: unspent,
            credited_ns: credited,
            at_ns: at,
            deferred: false,
            far: false,
        };
        let mut ledger = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
        if ledger.events.len() < MAX_LEDGER {
            ledger.events.push(event);
        } else {
            ledger.dropped = ledger.dropped.saturating_add(1);
        }
        Some(event)
    }

    /// #router-r3 CR-ATK: `slot` stopped UNDECIDED at a safe point because
    /// its own slope rule judged it flat, and its unspent contract is
    /// DEFERRED to the late sat reserve rather than credited to the terminal:
    /// the reserve is held out of the terminal's window by the caller and
    /// spent by the late attack, so the terminal keeps exactly today's window
    /// and the attack keeps today's total cap. Recorded in the ledger with
    /// `deferred = true`, `credited_ns = 0`. Returns the hand-on (`None` when
    /// the slot is not governed, never claimed, already released, or had
    /// nothing left).
    pub fn release_to_reserve(&self, slot: LaneSlot, now: Instant) -> Option<HandOn> {
        if !self.governs(slot) {
            return None;
        }
        let frame = self.frame()?;
        let at = frame.offset(now);
        let released = self.released.get(slot.index())?;
        if released
            .compare_exchange(0, at.saturating_add(1), Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }
        if Self::slot_u64(&self.claimed, slot) == 0 {
            return None;
        }
        let cell = self.cells.get(slot.index())?.offset_ns()?;
        let unspent = cell.saturating_sub(at);
        if unspent == 0 {
            return None;
        }
        let event = HandOn {
            from: slot,
            to: None,
            unspent_ns: unspent,
            credited_ns: 0,
            at_ns: at,
            deferred: true,
            far: false,
        };
        let mut ledger = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
        if ledger.events.len() < MAX_LEDGER {
            ledger.events.push(event);
        } else {
            ledger.dropped = ledger.dropped.saturating_add(1);
        }
        Some(event)
    }

    /// #router-r4 CR-F: `slot` stopped UNDECIDED at one of its safe points
    /// because its progress stream read FAR; its unspent contract
    /// (`cell - now`) is handed to the explicit `recipient` slot. The
    /// recipient must be governed, distinct, and must not have claimed or
    /// released yet: a started lane without safe points is never extended, and
    /// the credit then reaches it through the live clock its own rule reads at
    /// its start (its claim checks that it did: `realized`). The credit never
    /// carries the recipient past its own cap (`credited <= unspent`).
    /// Returns the hand-on (`None` when `slot` is not governed, never claimed,
    /// already released, or had nothing left, or when the recipient cannot
    /// receive — then nothing is released and `slot` keeps its cell).
    pub fn release_to(&self, slot: LaneSlot, recipient: LaneSlot, now: Instant) -> Option<HandOn> {
        if !self.governs(slot) || recipient == slot || !self.governs(recipient) {
            return None;
        }
        if Self::slot_u64(&self.claimed, recipient) != 0
            || Self::slot_u64(&self.released, recipient) != 0
            || Self::slot_u64(&self.claimed, slot) == 0
        {
            return None;
        }
        let frame = self.frame()?;
        let at = frame.offset(now);
        let cell = self.cells.get(slot.index())?.offset_ns()?;
        let unspent = cell.saturating_sub(at);
        if unspent == 0 {
            return None;
        }
        let released = self.released.get(slot.index())?;
        if released
            .compare_exchange(0, at.saturating_add(1), Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }
        let room = frame.cap_ns(recipient).saturating_sub(at);
        let before = Self::slot_u64(&self.credit, recipient);
        let credited = unspent.min(room.saturating_sub(before));
        Self::store_u64(&self.credit, recipient, before.saturating_add(credited));
        let event = HandOn {
            from: slot,
            to: (credited > 0).then_some(recipient),
            unspent_ns: unspent,
            credited_ns: credited,
            at_ns: at,
            deferred: false,
            far: true,
        };
        let mut ledger = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
        if ledger.events.len() < MAX_LEDGER {
            ledger.events.push(event);
        } else {
            ledger.dropped = ledger.dropped.saturating_add(1);
        }
        Some(event)
    }

    /// `slot`'s cell as an instant.
    #[must_use]
    pub fn deadline(&self, slot: LaneSlot) -> Option<Instant> {
        if !self.governs(slot) {
            return None;
        }
        let frame = self.frame()?;
        let offset = self.cells.get(slot.index())?.offset_ns()?;
        Some(frame.at(offset))
    }

    /// Time left in `slot`'s cell at `now`.
    #[must_use]
    pub fn remaining(&self, slot: LaneSlot, now: Instant) -> Option<Duration> {
        self.deadline(slot)
            .map(|deadline| deadline.saturating_duration_since(now))
    }

    /// CR-R: the publish grace for a completed certificate, when armed.
    #[must_use]
    pub fn publish_grace(&self) -> Option<Duration> {
        if !self.is_armed() {
            return None;
        }
        match self.grace_ns.load(Ordering::Acquire) {
            0 => None,
            ns => Some(Duration::from_nanos(ns)).filter(|g| *g < PUBLISH_GRACE_CEILING),
        }
    }

    /// Every recorded hand-on.
    #[must_use]
    pub fn hand_ons(&self) -> Vec<HandOn> {
        self.ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .events
            .clone()
    }

    /// The authority as recorded (whether or not armed now: a disarmed
    /// authority still reports what it held).
    #[must_use]
    pub fn snapshot(&self) -> AuthoritySnapshot {
        let armed = self.is_armed();
        let frame = self.frame.read().ok().and_then(|frame| *frame);
        let ms = |ns: u64| ns / 1_000_000;
        let decode = |raw: u64| (raw != 0).then(|| ms(raw - 1));
        let slots = LaneSlot::ALL
            .into_iter()
            .map(|slot| SlotSnapshot {
                slot,
                governed: self
                    .governed
                    .get(slot.index())
                    .is_some_and(|flag| flag.load(Ordering::Acquire)),
                cap_ms: frame.map_or(0, |f| ms(f.cap_ns(slot))),
                contract_ms: decode(Self::slot_u64(&self.contracts, slot)),
                trimmed_ms: ms(Self::slot_u64(&self.trimmed, slot)),
                claimed_ms: decode(Self::slot_u64(&self.claimed, slot)),
                cell_ms: self
                    .cells
                    .get(slot.index())
                    .and_then(LaneDeadline::offset_ns)
                    .map(ms),
                released_ms: decode(Self::slot_u64(&self.released, slot)),
                credit_ms: ms(Self::slot_u64(&self.credit, slot)),
            })
            .collect();
        let (hand_ons, ledger_dropped) = {
            let ledger = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
            (ledger.events.clone(), ledger.dropped)
        };
        let unspent = hand_ons
            .iter()
            .fold(0u64, |sum, h| sum.saturating_add(h.unspent_ns));
        let credited = hand_ons
            .iter()
            .fold(0u64, |sum, h| sum.saturating_add(h.credited_ns));
        let deferred = hand_ons
            .iter()
            .filter(|h| h.deferred)
            .fold(0u64, |sum, h| sum.saturating_add(h.unspent_ns));
        let terminal = match self.terminal.load(Ordering::Acquire) {
            0 => None,
            index => LaneSlot::ALL.get((index - 1) as usize).copied(),
        };
        AuthoritySnapshot {
            armed,
            hard_ms: frame.map_or(0, |f| ms(f.hard_ns())),
            join_ms: frame.map_or(0, |f| ms(nanos(f.join()))),
            grace_ms: match self.grace_ns.load(Ordering::Acquire) {
                0 => None,
                ns => Some(ms(ns)),
            },
            terminal,
            slots,
            hand_ons,
            unspent_ms: ms(unspent),
            credited_ms: ms(credited),
            deferred_ms: ms(deferred),
            unrealized: self.unrealized.load(Ordering::Acquire),
            ledger_dropped,
        }
    }
}

static GLOBAL: Authority = Authority::new();

/// The process's one deadline authority. `ny vnncomp` runs one row per
/// process, so there is exactly one frame at any moment.
#[must_use]
pub fn global() -> &'static Authority {
    &GLOBAL
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn frame(anchor: Instant, hard_secs: u64) -> Frame {
        Frame::new(
            anchor,
            anchor + Duration::from_secs(hard_secs),
            [
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(3),
                Duration::from_secs(3),
            ],
            Duration::from_secs(2),
        )
    }

    const ALL_GOVERNED: [bool; SLOTS] = [true; SLOTS];

    /// ROUTER_PROGRAM §6.4: the cell is monotone and capped, whatever the
    /// interleaving of writers, and every concurrent reader observes a
    /// non-decreasing sequence below the cap.
    #[test]
    fn lane_deadline_cell_is_monotone_under_concurrent_readers() {
        let cell = Arc::new(LaneDeadline::new());
        let cap = 1_000_000u64;
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let cell = Arc::clone(&cell);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut last = 0u64;
                    let mut reads = 0u64;
                    while !stop.load(Ordering::Acquire) || reads < 1_000 {
                        if let Some(now) = cell.offset_ns() {
                            assert!(now >= last, "a reader saw the cell go backwards");
                            assert!(now <= cap, "a reader saw the cell past its cap");
                            last = now;
                        }
                        reads += 1;
                    }
                    last
                })
            })
            .collect();
        let writers: Vec<_> = (0..4u64)
            .map(|w| {
                let cell = Arc::clone(&cell);
                std::thread::spawn(move || {
                    let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ w;
                    for _ in 0..20_000 {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        // Requests land both inside and far past the cap.
                        let _ = cell.raise(x % (3 * cap), cap);
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().expect("writer");
        }
        stop.store(true, Ordering::Release);
        for reader in readers {
            let last = reader.join().expect("reader");
            assert!(last <= cap);
        }
        let held = cell.offset_ns().expect("written");
        assert_eq!(
            held, cap,
            "20k requests past the cap must saturate AT the cap"
        );
        // A lower request never lowers it.
        assert_eq!(cell.raise(1, cap), cap);
    }

    #[test]
    fn unarmed_authority_answers_nothing() {
        let authority = Authority::new();
        let now = Instant::now();
        for slot in LaneSlot::ALL {
            assert!(!authority.governs(slot));
            assert_eq!(
                authority.commit_contract(slot, Duration::from_secs(9), now),
                None
            );
            assert_eq!(authority.claim_contract(slot, now), None);
            assert_eq!(authority.claim_deadline(slot, now, now), None);
            assert_eq!(authority.release(slot, now, false), None);
            assert_eq!(authority.deadline(slot), None);
        }
        assert_eq!(authority.publish_grace(), None);
    }

    #[test]
    fn arm_refuses_a_grace_at_or_past_its_bounds() {
        let authority = Authority::new();
        let anchor = Instant::now();
        let f = frame(anchor, 100);
        assert_eq!(
            authority.arm(
                f,
                ALL_GOVERNED,
                Some(LaneSlot::Verifier),
                Some(Duration::from_secs(3))
            ),
            Err(ArmRefusal::GraceNotBelowCeiling)
        );
        assert_eq!(
            authority.arm(
                f,
                ALL_GOVERNED,
                Some(LaneSlot::Verifier),
                Some(Duration::from_secs(2))
            ),
            Err(ArmRefusal::GraceNotBelowJoin)
        );
        assert!(!authority.is_armed(), "a refused frame arms nothing");
        authority
            .arm(
                f,
                ALL_GOVERNED,
                Some(LaneSlot::Verifier),
                Some(Duration::from_millis(1_500)),
            )
            .expect("inside both bounds");
        assert_eq!(
            authority.publish_grace(),
            Some(Duration::from_millis(1_500))
        );
        authority.disarm();
        assert_eq!(authority.publish_grace(), None);
    }

    /// Hand-on conservation: over random contract/spend schedules the sum of
    /// committed contracts never exceeds the window minus reserves, every
    /// credit is at most the unspent time it came from, and the terminal's
    /// claim finds every credit inside its window.
    #[test]
    fn hand_on_conserves_time_and_never_exceeds_the_window() {
        let mut x = 0x5eed_2002u64;
        let mut next = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if n == 0 {
                0
            } else {
                x % n
            }
        };
        for _ in 0..500 {
            let authority = Authority::new();
            let anchor = Instant::now();
            let hard = 20 + next(600);
            let f = frame(anchor, hard);
            authority
                .arm(f, ALL_GOVERNED, Some(LaneSlot::Verifier), None)
                .expect("arm");
            let window_ns = f.cap_ns(LaneSlot::AttackUpfront);
            let mut clock = Duration::ZERO;
            let mut committed = 0u64;
            let mut spent_total = 0u64;
            let lanes = [
                LaneSlot::AttackSignSpace,
                LaneSlot::AttackStePgd,
                LaneSlot::AttackUpfront,
            ];
            for slot in lanes {
                let asked = Duration::from_millis(next(hard * 700));
                let held = authority
                    .commit_contract(slot, asked, anchor)
                    .expect("governed");
                assert!(
                    held <= asked,
                    "a contract is never raised past what was asked"
                );
                committed += nanos(held);
            }
            assert!(
                committed <= window_ns,
                "sum of contracts {committed} exceeds the window {window_ns}"
            );
            let mut unspent_total = 0u64;
            let mut credited_total = 0u64;
            for slot in lanes {
                let start = anchor + clock;
                let window = authority.claim_contract(slot, start).expect("contracted");
                let spend = Duration::from_nanos(next(nanos(window).saturating_add(1)));
                clock += spend;
                spent_total += nanos(spend);
                let decided = next(10) == 0;
                if let Some(hand_on) = authority.release(slot, anchor + clock, decided) {
                    assert!(!decided);
                    assert!(hand_on.credited_ns <= hand_on.unspent_ns);
                    assert_eq!(hand_on.to.is_some(), hand_on.credited_ns > 0);
                    unspent_total += hand_on.unspent_ns;
                    credited_total += hand_on.credited_ns;
                }
            }
            assert!(credited_total <= unspent_total);
            assert!(spent_total + unspent_total <= committed + 1_000);
            // The terminal takes the remainder: its own rule reads the live
            // clock, so the credit is inside its window.
            let terminal_start = anchor + clock;
            let today = f.at(f.cap_ns(LaneSlot::Verifier));
            let claim = authority
                .claim_deadline(LaneSlot::Verifier, today, terminal_start)
                .expect("governed terminal");
            assert_eq!(claim.deadline, today, "the terminal keeps its own deadline");
            assert_eq!(nanos(claim.credit), credited_total);
            assert!(
                claim.realized,
                "every credit lands inside the terminal's window"
            );
            let snapshot = authority.snapshot();
            assert_eq!(snapshot.unrealized, 0);
            assert_eq!(snapshot.credited_ms, credited_total / 1_000_000);
        }
    }

    #[test]
    fn a_claimed_terminal_or_uncancellable_lane_is_never_extended() {
        let authority = Authority::new();
        let anchor = Instant::now();
        let f = frame(anchor, 100);
        authority
            .arm(f, ALL_GOVERNED, Some(LaneSlot::Verifier), None)
            .expect("arm");
        let terminal_deadline = anchor + Duration::from_secs(90);
        let claim = authority
            .claim_deadline(LaneSlot::Verifier, terminal_deadline, anchor)
            .expect("claim");
        assert_eq!(claim.deadline, terminal_deadline);
        // A contracted lane ending undecided AFTER the terminal started
        // returns its time to the pool instead of extending the terminal.
        authority
            .commit_contract(LaneSlot::AttackUpfront, Duration::from_secs(8), anchor)
            .expect("contract");
        authority
            .claim_contract(LaneSlot::AttackUpfront, anchor)
            .expect("claim");
        let hand_on = authority
            .release(
                LaneSlot::AttackUpfront,
                anchor + Duration::from_secs(1),
                false,
            )
            .expect("unspent");
        assert_eq!(hand_on.to, None);
        assert_eq!(hand_on.credited_ns, 0);
        assert_eq!(
            authority.deadline(LaneSlot::Verifier),
            Some(terminal_deadline)
        );
        // A margin-row lane keeps exactly the deadline its own rule derived,
        // even when that lies above the slot's cap, and a second claim with a
        // later deadline cannot push it past the scored deadline.
        let today = anchor + Duration::from_secs(97);
        let mr = authority
            .claim_deadline(LaneSlot::MarginRowInline, today, anchor)
            .expect("claim");
        assert_eq!(mr.deadline, today);
        let later = authority
            .claim_deadline(
                LaneSlot::MarginRowInline,
                anchor + Duration::from_secs(500),
                anchor,
            )
            .expect("claim");
        assert!(
            later.repeated,
            "the first claim of a lane without safe points stands"
        );
        assert_eq!(
            authority.deadline(LaneSlot::MarginRowInline),
            Some(today),
            "a repeated claim neither extends nor shortens the recorded cell"
        );
        // A deadline past the scored deadline is handed back unchanged (the
        // authority never shortens a lane's own rule) but recorded at `hard`.
        let past = anchor + Duration::from_secs(150);
        let concurrent = authority
            .claim_deadline(LaneSlot::MarginRowConcurrent, past, anchor)
            .expect("claim");
        assert_eq!(concurrent.deadline, past);
        assert_eq!(
            authority.deadline(LaneSlot::MarginRowConcurrent),
            Some(f.hard())
        );
    }

    #[test]
    fn an_ungoverned_slot_keeps_todays_value() {
        let authority = Authority::new();
        let anchor = Instant::now();
        let mut governed = ALL_GOVERNED;
        governed[LaneSlot::AttackStePgd.index()] = false;
        authority
            .arm(frame(anchor, 100), governed, Some(LaneSlot::Verifier), None)
            .expect("arm");
        assert!(!authority.governs(LaneSlot::AttackStePgd));
        assert_eq!(
            authority.commit_contract(LaneSlot::AttackStePgd, Duration::from_secs(4), anchor),
            None
        );
        assert!(authority
            .commit_contract(LaneSlot::AttackUpfront, Duration::from_secs(4), anchor)
            .is_some());
        // The terminal is never contracted.
        assert_eq!(
            authority.commit_contract(LaneSlot::Verifier, Duration::from_secs(4), anchor),
            None
        );
    }

    #[test]
    fn extreme_values_never_panic() {
        let authority = Authority::new();
        let anchor = Instant::now();
        let f = Frame::new(
            anchor,
            anchor,
            [Duration::MAX; SLOTS],
            Duration::from_nanos(1),
        );
        authority
            .arm(f, ALL_GOVERNED, Some(LaneSlot::Verifier), None)
            .expect("a degenerate frame still arms");
        for slot in LaneSlot::ALL {
            let _ = authority.commit_contract(slot, Duration::MAX, anchor);
            let _ = authority.claim_contract(slot, anchor + Duration::from_secs(1_000_000));
            let _ = authority.claim_deadline(slot, anchor, anchor);
            let _ = authority.release(slot, anchor, false);
            let _ = authority.remaining(slot, anchor);
        }
        let _ = authority.snapshot();
        let cell = LaneDeadline::new();
        assert_eq!(cell.raise(u64::MAX, u64::MAX), u64::MAX - 1);
    }

    /// #router-r3 CR-ATK: a deferred release hands NOTHING to the terminal
    /// (its window stays today's), records the unspent time as deferred, and
    /// conserves time: `credited + deferred <= unspent` over the ledger.
    #[test]
    fn a_deferred_release_credits_nothing_and_conserves_time() {
        let authority = Authority::new();
        let anchor = Instant::now();
        let f = frame(anchor, 100);
        authority
            .arm(f, ALL_GOVERNED, Some(LaneSlot::Verifier), None)
            .expect("arm");
        authority
            .commit_contract(LaneSlot::AttackUpfront, Duration::from_secs(30), anchor)
            .expect("contract");
        authority
            .claim_contract(LaneSlot::AttackUpfront, anchor)
            .expect("claim");
        let deferred = authority
            .release_to_reserve(LaneSlot::AttackUpfront, anchor + Duration::from_secs(16))
            .expect("unspent");
        assert!(deferred.deferred);
        assert_eq!(deferred.credited_ns, 0);
        assert_eq!(deferred.to, None);
        assert_eq!(deferred.unspent_ns, 14_000_000_000);
        // Released once: a later ordinary release is a no-op.
        assert_eq!(
            authority.release(
                LaneSlot::AttackUpfront,
                anchor + Duration::from_secs(17),
                false
            ),
            None
        );
        // The terminal finds no credit: its window is exactly its own rule's.
        let claim = authority
            .claim_deadline(
                LaneSlot::Verifier,
                anchor + Duration::from_secs(90),
                anchor + Duration::from_secs(16),
            )
            .expect("terminal");
        assert_eq!(claim.credit, Duration::ZERO);
        let snap = authority.snapshot();
        assert_eq!(snap.deferred_ms, 14_000);
        assert_eq!(snap.credited_ms, 0);
        assert!(snap.credited_ms + snap.deferred_ms <= snap.unspent_ms);
        // Unarmed or ungoverned: nothing.
        authority.disarm();
        assert_eq!(
            authority.release_to_reserve(LaneSlot::AttackUpfront, anchor),
            None
        );
    }
}
