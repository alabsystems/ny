// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-close-telemetry: PROOF-INTEGRITY telemetry for the margin-row lane.
//!
//! # Why this exists
//!
//! A margin-row `unsat` is a leaf cover: every leaf of the tree closed at a
//! certified bound `b > 0` (`bab::closed`, STRICT, no floor), every class the
//! root closed has `dj > 0`, and a handful of leaves closed by an emptiness
//! certificate instead of a bound. The root-tableau rounding leak audited in
//! `scratchpad/vbs/diag10/root_leak_bank_audit.md` can over-estimate a
//! certified margin by an amount the audit prices at up to ~1e-8, so a proof
//! whose thinnest certificate sits below that is not established by the proof
//! alone. Until this module, NO artifact recorded the thinnest certificate of
//! any tree-closed row: `BabStats` carried counts, the flight receipts carried
//! the root bound only, and the stderr samples were rate-limited and printed
//! with `{:.6}`, which cannot resolve anything below 5e-7. 272 of 319 banked
//! margin-row `unsat` row-verdicts were therefore unauditable.
//!
//! # What is recorded
//!
//! Every certificate the published verdict rests on, keyed by [`CertKind`]:
//! its count, its minimum value (exact `f64`, so the receipt can print both the
//! bits and a round-trip decimal) and how many fall strictly below `1e-6`,
//! `1e-8` and `1e-10`. Recorded at exactly the sites that call the tree's
//! Kraft `ledger.leaf` (plus the root and the retirement sites), in the
//! deterministic serial merge, so the tally is a pure function of the merged
//! tree — the same on every run that merges the same tree.
//!
//! * `root` — outer-frame classes closed at the root: `dj > 0`.
//! * `eval` / `beta` / `dalpha` — a tree domain closed by its own refreshed
//!   bound, split by the eval STAGE that first made the bound close: the main
//!   certified pass, the β ascent (shared or per-column), or the per-domain
//!   alpha ascent. The value is the bound the `closed()` test read.
//! * `raw` / `lift` — a child closed by its own scored bound, or only through
//!   the parent lift; the value is the child's closing bound `max(raw, b, lift)`.
//! * `queued` — a popped entry whose queued bound already closed (in the
//!   serial loop the whole drained heap, each entry at its own bound).
//! * `infeasible` — a domain proven EMPTY at eval; the value is the certified
//!   slack of the emptiness certificate: `max_i (ly_i - uy_i)` for a crossed
//!   y-box, or the largest `l_new - u_new` crossing of a clip-and-verify
//!   re-concretization.
//! * `head_infeasible` — a head-split child slot whose interval crossed; the
//!   value is `ly_i - uy_i`.
//! * `retire` — a margin column retired (#margin-row-col-retire) at a domain
//!   that stayed OPEN: every descendant leaf certifies only the surviving
//!   columns, so the retired column's certified composite at retirement is its
//!   whole certificate on that subtree. (Columns retired at a domain that
//!   closed at eval are dominated by that leaf's own `eval` value and are not
//!   counted twice.)
//! * `epoch_root` — classes closed at the root of a Tier-2 epoch subtree that
//!   closed (a nested run's `root`). The nested run's other leaves are recorded
//!   under their own kinds; `epoch_closes` counts the epoch closes themselves.
//!
//! `frozen_dual_closes` counts classes the frozen-dual tier closed; that tier's
//! certificate is not exposed to this lane, so it carries a count only.
//! `nonpositive` counts recorded values that are NOT strictly positive (or are
//! NaN). In production (outward rounding, strict closure) it must read 0 —
//! anything else means a close site and its recorded value disagree.
//!
//! # The two outputs
//!
//! * `BabStats::close_integrity` — the authoritative per-invocation tally,
//!   rendered into the flight record by the scored command (the flight events
//!   are what survive `ny benchmarks run`, which nulls stdout and keeps only an
//!   error tail of stderr).
//! * ONE `[mr-close] ...` stderr line on every lane exit, printed beside the
//!   `[mr-census]` line from the census stop receipts ([`stop_line`]): the
//!   final tally on a returned lane, the last merged state on a SNAPSHOT (a
//!   lane abandoned at its slice cap or still running at process exit).
//!
//! # Soundness
//!
//! Telemetry only. Nothing here is read by any bound, gate, closure decision,
//! search order or verdict: every value is copied out of a decision that was
//! already made. Its cost is O(1) per close (one comparison and four
//! increments) plus one small copy under an uncontended mutex per merged
//! frontier batch. `margin_row/tests.rs`'s
//! `close_telemetry_trajectory_hash_unchanged` runs one tree with and without
//! the recording (on the parallel, domain-stacked, serial and classwise
//! routes) and requires the trajectory hash, the verdict and every other
//! statistic to be bit-identical.

use std::sync::{Mutex, MutexGuard, PoisonError};

use super::bab::MarginRowOutcome;

/// `[mr-close]` / flight-field schema version (bump on any field change).
pub const CLOSE_INTEGRITY_SCHEMA: u32 = 1;

/// Thresholds of the `closes_lt_*` counts. A value counts when it is strictly
/// below the threshold or NaN (a NaN certificate counts everywhere:
/// conservative).
pub const LT_THRESHOLDS: [f64; 3] = [1e-6, 1e-8, 1e-10];

/// Field-name suffixes of [`LT_THRESHOLDS`], in order.
const LT_NAMES: [&str; 3] = ["1e-6", "1e-8", "1e-10"];

/// What certified a close (index = discriminant). See the module docs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CertKind {
    /// Outer-frame class closed at the root (`dj > 0`).
    Root = 0,
    /// Tree domain closed by its main certified pass.
    Eval = 1,
    /// Tree domain closed only after the β ascent.
    Beta = 2,
    /// Tree domain closed only after the per-domain alpha ascent.
    DomainAlpha = 3,
    /// Child closed by its own scored bound.
    Raw = 4,
    /// Child closed only through the parent lift.
    Lift = 5,
    /// Popped entry whose queued bound already closed.
    Queued = 6,
    /// Domain proven empty at eval (certified crossing slack).
    Infeasible = 7,
    /// Head-split child slot whose interval crossed (crossing slack).
    HeadInfeasible = 8,
    /// Column retired at a domain that stayed open.
    Retire = 9,
    /// Class closed at the root of a closed Tier-2 epoch subtree.
    EpochRoot = 10,
}

/// Number of [`CertKind`]s.
pub const NKIND: usize = 11;

/// Every kind, in discriminant (= print) order.
pub const ALL_KINDS: [CertKind; NKIND] = [
    CertKind::Root,
    CertKind::Eval,
    CertKind::Beta,
    CertKind::DomainAlpha,
    CertKind::Raw,
    CertKind::Lift,
    CertKind::Queued,
    CertKind::Infeasible,
    CertKind::HeadInfeasible,
    CertKind::Retire,
    CertKind::EpochRoot,
];

impl CertKind {
    /// Stable field token.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Eval => "eval",
            Self::Beta => "beta",
            Self::DomainAlpha => "dalpha",
            Self::Raw => "raw",
            Self::Lift => "lift",
            Self::Queued => "queued",
            Self::Infeasible => "infeasible",
            Self::HeadInfeasible => "head_infeasible",
            Self::Retire => "retire",
            Self::EpochRoot => "epoch_root",
        }
    }

    const fn idx(self) -> usize {
        self as usize
    }
}

/// One close's certificate: its kind and the certified value the close read.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Cert {
    pub(crate) kind: CertKind,
    pub(crate) value: f64,
}

/// Count / minimum / below-threshold counts of one family of certificates.
#[derive(Copy, Clone, Debug)]
pub struct CertTally {
    /// Certificates recorded.
    pub n: u64,
    /// Smallest recorded value (`+inf` when none; a NaN never lowers it).
    pub min: f64,
    /// Values strictly below (or NaN) each of [`LT_THRESHOLDS`].
    pub lt: [u64; 3],
    /// Values not strictly positive (NaN included).
    pub nonpositive: u64,
}

impl CertTally {
    /// The empty tally.
    pub const EMPTY: Self = Self {
        n: 0,
        min: f64::INFINITY,
        lt: [0; 3],
        nonpositive: 0,
    };

    /// Record one certificate. O(1).
    #[inline]
    pub fn note(&mut self, value: f64) {
        self.n = self.n.saturating_add(1);
        if value < self.min {
            self.min = value;
        }
        if value.is_nan() || value <= 0.0 {
            self.nonpositive = self.nonpositive.saturating_add(1);
        }
        for (count, threshold) in self.lt.iter_mut().zip(LT_THRESHOLDS) {
            if value.is_nan() || value < threshold {
                *count = count.saturating_add(1);
            }
        }
    }

    /// Merge another tally of the same family.
    pub fn absorb(&mut self, other: &Self) {
        self.n = self.n.saturating_add(other.n);
        if other.min < self.min {
            self.min = other.min;
        }
        for (a, b) in self.lt.iter_mut().zip(other.lt) {
            *a = a.saturating_add(b);
        }
        self.nonpositive = self.nonpositive.saturating_add(other.nonpositive);
    }

    /// The smallest recorded value, `None` when nothing (finite-ordered) was
    /// recorded.
    #[must_use]
    pub fn min_value(&self) -> Option<f64> {
        (self.n > 0 && self.min < f64::INFINITY).then_some(self.min)
    }
}

impl Default for CertTally {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// The proof-integrity tally of one lane invocation (see the module docs).
#[derive(Copy, Clone, Debug)]
pub struct CloseIntegrityEvidence {
    kinds: [CertTally; NKIND],
    /// Tier-2 epoch closes (each one's nested certificates are recorded under
    /// their own kinds, its nested root under `epoch_root`).
    pub epoch_closes: u64,
    /// Classes the frozen-dual tier closed (certificate not exposed here).
    pub frozen_dual_closes: u64,
}

impl CloseIntegrityEvidence {
    /// Nothing recorded.
    pub const EMPTY: Self = Self {
        kinds: [CertTally::EMPTY; NKIND],
        epoch_closes: 0,
        frozen_dual_closes: 0,
    };

    /// Record one certificate of `kind`. O(1).
    #[inline]
    pub fn note(&mut self, kind: CertKind, value: f64) {
        self.kinds[kind.idx()].note(value);
    }

    /// Record a pre-accumulated tally of `kind` (a worker-built retirement
    /// tally). O(1).
    #[inline]
    pub fn note_tally(&mut self, kind: CertKind, tally: &CertTally) {
        self.kinds[kind.idx()].absorb(tally);
    }

    /// Absorb a CLOSED nested epoch run's tally: its root closes become
    /// `epoch_root`, every other kind keeps its kind.
    pub fn absorb_nested(&mut self, nested: &Self) {
        for kind in ALL_KINDS {
            let dst = if kind == CertKind::Root {
                CertKind::EpochRoot
            } else {
                kind
            };
            self.kinds[dst.idx()].absorb(&nested.kinds[kind.idx()]);
        }
        self.epoch_closes = self.epoch_closes.saturating_add(nested.epoch_closes);
        self.frozen_dual_closes = self
            .frozen_dual_closes
            .saturating_add(nested.frozen_dual_closes);
    }

    /// The tally of one kind.
    #[must_use]
    pub fn kind(&self, kind: CertKind) -> &CertTally {
        &self.kinds[kind.idx()]
    }

    /// Certificates recorded over every kind.
    #[must_use]
    pub fn closes(&self) -> u64 {
        self.kinds
            .iter()
            .fold(0u64, |acc, t| acc.saturating_add(t.n))
    }

    /// Below-threshold counts over every kind, in [`LT_THRESHOLDS`] order.
    #[must_use]
    pub fn lt_totals(&self) -> [u64; 3] {
        let mut out = [0u64; 3];
        for t in &self.kinds {
            for (a, b) in out.iter_mut().zip(t.lt) {
                *a = a.saturating_add(b);
            }
        }
        out
    }

    /// Not-strictly-positive values over every kind (must be 0 in production).
    #[must_use]
    pub fn nonpositive(&self) -> u64 {
        self.kinds
            .iter()
            .fold(0u64, |acc, t| acc.saturating_add(t.nonpositive))
    }

    /// The thinnest certificate over every kind, and its kind (ties resolve
    /// to the first kind in print order).
    #[must_use]
    pub fn min_close(&self) -> Option<(CertKind, f64)> {
        let mut best: Option<(CertKind, f64)> = None;
        for kind in ALL_KINDS {
            if let Some(v) = self.kinds[kind.idx()].min_value() {
                if best.is_none_or(|(_, b)| v < b) {
                    best = Some((kind, v));
                }
            }
        }
        best
    }

    /// The smallest root `dj` over the classes closed at the root.
    #[must_use]
    pub fn min_root_dj(&self) -> Option<f64> {
        self.kinds[CertKind::Root.idx()].min_value()
    }

    /// The flight-record fields (`key=value`, space separated, fixed schema:
    /// every key is present on every receipt, `none`/`na` when empty).
    #[must_use]
    pub fn fields(&self) -> String {
        let lt = self.lt_totals();
        let min = self.min_close();
        let root = self.kind(CertKind::Root);
        let mut out = format!(
            "close_schema={CLOSE_INTEGRITY_SCHEMA} closes={} min_close_bits={} min_close={} \
             min_close_kind={} closes_lt_{}={} closes_lt_{}={} closes_lt_{}={} \
             min_root_dj_bits={} min_root_dj={} root_closed={} epoch_closes={} \
             frozen_dual_closes={} nonpositive={}",
            self.closes(),
            bits(min.map(|(_, v)| v)),
            dec(min.map(|(_, v)| v)),
            min.map_or("none", |(k, _)| k.name()),
            LT_NAMES[0],
            lt[0],
            LT_NAMES[1],
            lt[1],
            LT_NAMES[2],
            lt[2],
            bits(root.min_value()),
            dec(root.min_value()),
            root.n,
            self.epoch_closes,
            self.frozen_dual_closes,
            self.nonpositive(),
        );
        for kind in ALL_KINDS {
            let t = self.kind(kind);
            let name = kind.name();
            out.push_str(&format!(
                " {name}_n={} {name}_min_bits={} {name}_min={} {name}_lt_{}={} \
                 {name}_lt_{}={} {name}_lt_{}={}",
                t.n,
                bits(t.min_value()),
                dec(t.min_value()),
                LT_NAMES[0],
                t.lt[0],
                LT_NAMES[1],
                t.lt[1],
                LT_NAMES[2],
                t.lt[2],
            ));
        }
        out
    }

    /// The one-line `[mr-close]` stderr receipt. `source` says where the tally
    /// came from (see [`PublishSource::token`]).
    #[must_use]
    pub fn receipt_line(&self, source: &str) -> String {
        let lt = self.lt_totals();
        let min = self.min_close();
        let by_type = ALL_KINDS
            .iter()
            .map(|&k| {
                let t = self.kind(k);
                format!("{}:{}:{}", k.name(), t.n, dec(t.min_value()))
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "[mr-close] min_close={} closes_lt_{}={} closes_lt_{}={} closes_lt_{}={} \
             by_type={by_type} min_close_bits={} min_close_kind={} closes={} \
             min_root_dj={} min_root_dj_bits={} epoch_closes={} frozen_dual_closes={} \
             nonpositive={} source={source} v={CLOSE_INTEGRITY_SCHEMA}",
            dec(min.map(|(_, v)| v)),
            LT_NAMES[0],
            lt[0],
            LT_NAMES[1],
            lt[1],
            LT_NAMES[2],
            lt[2],
            bits(min.map(|(_, v)| v)),
            min.map_or("none", |(k, _)| k.name()),
            self.closes(),
            dec(self.min_root_dj()),
            bits(self.min_root_dj()),
            self.epoch_closes,
            self.frozen_dual_closes,
            self.nonpositive(),
        )
    }
}

impl Default for CloseIntegrityEvidence {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Exact bits of a value, `none` when absent.
fn bits(value: Option<f64>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_bits().to_string())
}

/// Shortest round-trip decimal (`{:e}`), `na` when absent.
fn dec(value: Option<f64>) -> String {
    value.map_or_else(|| "na".to_owned(), |v| format!("{v:e}"))
}

// --------------------------------------------------------------------------
//  The process-global published tally (the stderr receipt's source)
// --------------------------------------------------------------------------

/// Where the published tally came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PublishSource {
    /// Nothing published since the lane began (it has not reached its root
    /// verdict, or this binary's lane never ran).
    Unpublished,
    /// The last state an OUTERMOST tree (or the outer root) published while
    /// the lane was still running.
    Live,
    /// The returned lane's own `BabStats::close_integrity` (exact).
    Final,
    /// The lane returned WITHOUT stats (a pre-tree stop or a fail-closed
    /// error); the tally is the last live state, if any.
    FinalNoStats,
}

impl PublishSource {
    /// Stable token printed as `source=`.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Unpublished => "none",
            Self::Live => "live",
            Self::Final => "final",
            Self::FinalNoStats => "final_no_stats",
        }
    }
}

struct Published {
    source: PublishSource,
    evidence: CloseIntegrityEvidence,
}

static PUBLISHED: Mutex<Published> = Mutex::new(Published {
    source: PublishSource::Unpublished,
    evidence: CloseIntegrityEvidence::EMPTY,
});

fn published_lock() -> MutexGuard<'static, Published> {
    // Telemetry must never turn someone else's panic into a second one.
    PUBLISHED.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Clear the published tally (lane start; `census::reset`).
pub(crate) fn reset() {
    let mut guard = published_lock();
    guard.source = PublishSource::Unpublished;
    guard.evidence = CloseIntegrityEvidence::EMPTY;
}

/// Publish a running lane's current tally (outer root, then once per merged
/// frontier step of an OUTERMOST tree). O(1): one small copy.
pub(crate) fn publish_live(evidence: &CloseIntegrityEvidence) {
    let mut guard = published_lock();
    guard.source = PublishSource::Live;
    guard.evidence = *evidence;
}

/// Publish a returned lane's exact tally (the lane wrapper, before its census
/// guard prints the stop receipts). A stats-less outcome keeps the last live
/// tally and says so.
pub(crate) fn publish_outcome(out: &MarginRowOutcome) {
    let stats = match out {
        MarginRowOutcome::Unsat(stats, _) => Some(stats),
        MarginRowOutcome::Unknown { stats, .. } => stats.as_ref(),
    };
    let mut guard = published_lock();
    match stats {
        Some(stats) => {
            guard.source = PublishSource::Final;
            guard.evidence = stats.close_integrity;
        }
        None => guard.source = PublishSource::FinalNoStats,
    }
}

/// The published tally and its source. Safe from any thread.
#[must_use]
pub fn published() -> (PublishSource, CloseIntegrityEvidence) {
    let guard = published_lock();
    (guard.source, guard.evidence)
}

/// The published tally when a margin-row lane is still running (the scored
/// command's exit path records it in the flight record as a SNAPSHOT); `None`
/// when no lane is live.
#[must_use]
pub fn snapshot_if_live() -> Option<(PublishSource, CloseIntegrityEvidence)> {
    super::census::lane_live().then(published)
}

/// The `[mr-close]` stop receipt (one line), printed after `[mr-census]` at
/// every lane stop and on every census SNAPSHOT. Present on every stop of a
/// binary that carries this telemetry, zero or not, so its absence
/// identifies a binary without it.
#[must_use]
pub fn stop_line() -> String {
    let (source, evidence) = published();
    evidence.receipt_line(source.token())
}

#[cfg(test)]
mod close_integrity_tests {
    use super::*;

    /// `mr_close_tally_counts_minimum_and_thresholds`: the per-kind tally's
    /// minimum, strict-below-threshold counts and non-positive alarm, with the
    /// NaN convention (a NaN counts as below every threshold and non-positive,
    /// but never becomes the minimum).
    #[test]
    fn mr_close_tally_counts_minimum_and_thresholds() {
        let mut t = CertTally::default();
        assert_eq!(t.min_value(), None);
        for v in [0.5, 2e-7, 1e-6, 5e-9, 1e-10, 3e-11] {
            t.note(v);
        }
        assert_eq!(t.n, 6);
        assert_eq!(t.min_value().map(f64::to_bits), Some(3e-11f64.to_bits()));
        // < 1e-6: 2e-7, 5e-9, 1e-10, 3e-11 (1e-6 itself is NOT below).
        // < 1e-8: 5e-9, 1e-10, 3e-11.  < 1e-10: 3e-11 only.
        assert_eq!(t.lt, [4, 3, 1]);
        assert_eq!(t.nonpositive, 0);
        t.note(0.0);
        t.note(-1e-3);
        t.note(f64::NAN);
        assert_eq!(t.n, 9);
        assert_eq!(t.nonpositive, 3, "0, negative and NaN are all alarms");
        assert_eq!(t.lt, [7, 6, 4], "each alarm counts below every threshold");
        assert_eq!(t.min_value().map(f64::to_bits), Some((-1e-3f64).to_bits()));

        let mut only_nan = CertTally::default();
        only_nan.note(f64::NAN);
        assert_eq!(
            only_nan.min_value(),
            None,
            "a NaN never becomes the minimum"
        );
        assert_eq!(only_nan.nonpositive, 1);
    }

    /// `mr_close_evidence_min_and_nested_absorb`: the global minimum names its
    /// kind; a nested epoch run's root closes land in `epoch_root` and every
    /// other kind keeps its kind; counts and thresholds add.
    #[test]
    fn mr_close_evidence_min_and_nested_absorb() {
        let mut outer = CloseIntegrityEvidence::default();
        assert_eq!(outer.min_close().map(|(k, _)| k), None);
        outer.note(CertKind::Root, 2.5e-3);
        outer.note(CertKind::Eval, 4e-4);
        outer.note(CertKind::Raw, 7e-9);
        assert_eq!(outer.min_close().map(|(k, _)| k), Some(CertKind::Raw));
        assert_eq!(
            outer.min_root_dj().map(f64::to_bits),
            Some(2.5e-3f64.to_bits())
        );

        let mut nested = CloseIntegrityEvidence::default();
        nested.note(CertKind::Root, 1e-12);
        nested.note(CertKind::Lift, 3e-5);
        nested.epoch_closes = 1;
        outer.absorb_nested(&nested);
        outer.epoch_closes += 1;
        assert_eq!(
            outer.kind(CertKind::Root).n,
            1,
            "the OUTER root is unchanged"
        );
        assert_eq!(outer.kind(CertKind::EpochRoot).n, 1);
        assert_eq!(outer.kind(CertKind::Lift).n, 1);
        assert_eq!(outer.epoch_closes, 2);
        assert_eq!(outer.closes(), 5);
        assert_eq!(outer.min_close().map(|(k, _)| k), Some(CertKind::EpochRoot));
        assert_eq!(
            outer.min_root_dj().map(f64::to_bits),
            Some(2.5e-3f64.to_bits())
        );
        // 7e-9 and 1e-12 are below 1e-6 and 1e-8; only 1e-12 is below 1e-10.
        assert_eq!(outer.lt_totals(), [2, 2, 1]);

        let mut retire = CertTally::default();
        retire.note(0.2);
        retire.note(9e-7);
        outer.note_tally(CertKind::Retire, &retire);
        assert_eq!(outer.kind(CertKind::Retire).n, 2);
        assert_eq!(outer.lt_totals(), [3, 2, 1]);
    }

    /// `mr_close_receipts_have_fixed_schema_and_exact_bits`: the stderr line
    /// starts with the contracted prefix, carries every kind, and prints the
    /// minimum as exact bits AND a decimal that round-trips to the same bits;
    /// the flight fields carry every per-kind key on an EMPTY tally too (a
    /// fixed schema is what lets a reader tell "zero" from "absent").
    #[test]
    fn mr_close_receipts_have_fixed_schema_and_exact_bits() {
        let empty = CloseIntegrityEvidence::default();
        let line = empty.receipt_line("none");
        assert!(
            line.starts_with(
                "[mr-close] min_close=na closes_lt_1e-6=0 closes_lt_1e-8=0 \
                 closes_lt_1e-10=0 by_type="
            ),
            "{line}"
        );
        assert!(line.contains(" source=none "), "{line}");
        assert!(!line.contains('\n'), "the receipt must be ONE line");
        let fields = empty.fields();
        for kind in ALL_KINDS {
            let name = kind.name();
            assert!(line.contains(&format!("{name}:0:na")), "{name}: {line}");
            for key in [
                format!(" {name}_n=0 "),
                format!(" {name}_min_bits=none "),
                format!(" {name}_min=na "),
                format!(" {name}_lt_1e-6=0 "),
                format!(" {name}_lt_1e-8=0 "),
                format!(" {name}_lt_1e-10=0"),
            ] {
                assert!(fields.contains(&key), "missing {key:?}: {fields}");
            }
        }

        let thin = 1.234_567_890_123_4e-9_f64;
        let mut ev = CloseIntegrityEvidence::default();
        ev.note(CertKind::Beta, thin);
        ev.note(CertKind::Root, 0.25);
        let line = ev.receipt_line("final");
        let decimal = format!("{thin:e}");
        assert_eq!(decimal.parse::<f64>().map(f64::to_bits), Ok(thin.to_bits()));
        assert!(
            line.starts_with(&format!("[mr-close] min_close={decimal} ")),
            "{line}"
        );
        assert!(line.contains(" closes_lt_1e-6=1 closes_lt_1e-8=1 closes_lt_1e-10=0 "));
        assert!(
            line.contains(&format!(" min_close_bits={} ", thin.to_bits())),
            "{line}"
        );
        assert!(line.contains(" min_close_kind=beta "), "{line}");
        assert!(line.contains(&format!("beta:1:{decimal}")), "{line}");
        assert!(line.contains(" min_root_dj=2.5e-1 "), "{line}");
        let fields = ev.fields();
        assert!(fields.contains(&format!(" beta_min_bits={} ", thin.to_bits())));
        assert!(fields.contains(" min_close_kind=beta "));
        assert!(fields.contains(" root_closed=1 "));
        assert!(fields.contains(&format!(" min_root_dj_bits={} ", 0.25f64.to_bits())));
    }

    /// `mr_close_stop_line_is_one_line`: the census stop receipt is always
    /// present and one line (the published state itself is process-global and
    /// shared with concurrently running lanes, so only the shape is asserted).
    #[test]
    fn mr_close_stop_line_is_one_line() {
        let line = stop_line();
        assert!(line.starts_with("[mr-close] min_close="), "{line}");
        assert!(line.contains(" source="), "{line}");
        assert!(!line.contains('\n'));
        assert!(
            super::super::census::stop_lines()
                .iter()
                .any(|l| l.starts_with("[mr-close] ")),
            "the census stop receipts must carry the [mr-close] line"
        );
    }
}
