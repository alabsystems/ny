// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PRESET-OVERRIDE CELLS — the storage half of a PROMOTED lever.
//!
//! # The defect this exists to close
//!
//! `vnncomp_scripts/run_instance.sh` — the SCORED entry point — exports no
//! `NY_*` variables. Every environment-only gate therefore reads "unset" on a scored
//! run, so a lever that only an environment variable can arm is DEAD CODE in
//! competition however well it measured by hand. `tests/reachability_ratchet.rs`
//! counts them; this module is the plumbing that lets one be fixed.
//!
//! # What a promoted lever looks like
//!
//! Five parts, and all five are required (the template is
//! `margin_row.backward_interm`):
//!
//! 1. a typed field on the preset struct in `crates/ny-cli/src/preset/mod.rs`;
//! 2. a reader that resolves that field from the loaded preset;
//! 3. a BRIDGE LINE in `crates/ny-cli/src/commands/vnncomp.rs` that calls the
//!    setter BEFORE the consuming lane can start;
//! 4. a setter + reader pair in the owning module — THIS module supplies the
//!    cell and the resolution rule for that pair;
//! 5. `deliver: Deliver::PresetKey("…")` on the `space.rs` axis.
//!
//! # The layering rule, and why it is not re-implemented here
//!
//! [`crate::read_over_config`] already states it, is unit-tested, and is the same code
//! path the typed root-alpha seam uses:
//!
//! * an ADMISSIBLE environment value owns the effective layer according to its
//!   declaration. Ordinary Boolean overrides accept both `"1"` and `"0"`;
//!   one-way rollback kinds accept only their explicit kill token;
//! * for ordinary kinds, a PRESENT but inadmissible value (`"true"`, `" 1"`,
//!   `""`) suppresses the preset and resolves to the DECLARATION DEFAULT. The
//!   explicitly override-style kinds instead record an ignored token and keep
//!   the lower authority, as documented by [`crate::read_over_config`];
//! * absence falls through to the cell, then to the declaration default.
//!
//! So these resolvers are thin: a cell read, then `read_over_config`. Writing a
//! second precedence rule here is exactly how two halves of one gate end up
//! disagreeing, which is the hazard `env.rs` documents at length.
//!
//! # An UNSET cell is byte-identical
//!
//! Every cell starts unset, and unset means `config = None`, which is the
//! argument `read` passes today. A build that ships no preset key resolves
//! every one of these levers exactly as it did before promotion. Promotion
//! makes a lever DELIVERABLE; it does not arm it.
//!
//! # Write-once-then-read, deliberately
//!
//! The CLI writes each cell once, from the resolved preset, before verification
//! starts; everything after that only reads. That ordering matters for the
//! readers that latch their answer in a `OnceLock`: a cell written after the
//! latch has already been taken cannot be seen. The bridge block is therefore
//! placed ahead of every lane it feeds, and this module deliberately offers no
//! "reset" outside tests, so a mid-run rewrite cannot be spelled by accident.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
    RwLock,
};

use crate::decl::LeverDecl;
use crate::env::{read, read_over_config, LeverValue};

/// Cell is empty: the preset said nothing, so the declaration default (or the
/// environment) decides. This is the shipped state of every promoted lever.
const UNSET: u8 = 0;
/// The preset explicitly asked for `false`.
const OFF: u8 = 1;
/// The preset explicitly asked for `true`.
const ON: u8 = 2;

/// Tri-state override cell for a `LeverKind::Bool` (or `Presence`) lever.
///
/// Tri-state and not `bool`, because "the preset did not mention this key" and
/// "the preset explicitly disarmed this key" are different answers: only the
/// second may override a declaration whose default is `true`.
#[derive(Debug, Default)]
pub struct BoolCell(AtomicU8);

impl BoolCell {
    /// A new, EMPTY cell. `const` so it can be a `static` in the owning module.
    #[must_use]
    pub const fn new() -> Self {
        Self(AtomicU8::new(UNSET))
    }

    /// Record the typed preset's answer for this lever.
    pub fn set(&self, on: bool) {
        self.0.store(if on { ON } else { OFF }, Ordering::Relaxed);
    }

    /// The typed preset's answer, or `None` when the key was absent.
    #[must_use]
    pub fn get(&self) -> Option<bool> {
        match self.0.load(Ordering::Relaxed) {
            ON => Some(true),
            OFF => Some(false),
            _ => None,
        }
    }

    /// Return the cell to its shipped, empty state. Tests only: production
    /// writes each cell once, before any reader runs.
    #[cfg(test)]
    pub fn clear(&self) {
        self.0.store(UNSET, Ordering::Relaxed);
    }
}

/// Override cell for an integer lever (`U64`, `U64Trimmed`, `UsizeTrimmed`).
#[derive(Debug, Default)]
pub struct U64Cell {
    /// Whether [`Self::value`] has been written. Published with `Release` and
    /// read with `Acquire`, so a reader that observes the flag also observes
    /// the value that was stored before it.
    set: AtomicBool,
    /// The typed preset's answer. Meaningless while `set` is false.
    value: AtomicU64,
}

impl U64Cell {
    /// A new, EMPTY cell.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            set: AtomicBool::new(false),
            value: AtomicU64::new(0),
        }
    }

    /// Record the typed preset's answer for this lever.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
        self.set.store(true, Ordering::Release);
    }

    /// The typed preset's answer, or `None` when the key was absent.
    #[must_use]
    pub fn get(&self) -> Option<u64> {
        self.set
            .load(Ordering::Acquire)
            .then(|| self.value.load(Ordering::Relaxed))
    }

    /// Return the cell to its shipped, empty state. Tests only.
    #[cfg(test)]
    pub fn clear(&self) {
        self.set.store(false, Ordering::Release);
    }
}

/// Override cell for a floating-point lever (`F64Open`, `F64ClosedTrimmed`,
/// `Secs`). The value is held as `f64::to_bits`, so no precision is lost.
#[derive(Debug, Default)]
pub struct F64Cell {
    /// See [`U64Cell::set`] for the ordering contract.
    set: AtomicBool,
    /// `f64::to_bits` of the typed preset's answer.
    bits: AtomicU64,
}

/// Override cell for textual and enumerated levers.
///
/// Unlike the fixed-width cells, a string cannot live in an atomic. Reads are
/// cold (the consuming route normally keeps its historical `OnceLock`), so a
/// small lock is preferable to leaking strings or inventing numeric tags that
/// duplicate each declaration's enum domain.
#[derive(Debug, Default)]
pub struct TextCell(RwLock<Option<String>>);

impl TextCell {
    /// A new, EMPTY cell.
    #[must_use]
    pub const fn new() -> Self {
        Self(RwLock::new(None))
    }

    /// Record the typed preset's answer for this lever.
    pub fn set(&self, value: impl Into<String>) {
        if let Ok(mut slot) = self.0.write() {
            *slot = Some(value.into());
        }
    }

    /// The typed preset's answer, or `None` when the key was absent or the
    /// lock was poisoned. Poisoning fails closed to the declaration default.
    #[must_use]
    pub fn get(&self) -> Option<String> {
        self.0.read().ok().and_then(|slot| slot.clone())
    }

    /// Return the cell to its shipped, empty state. Tests only.
    #[cfg(test)]
    pub fn clear(&self) {
        if let Ok(mut slot) = self.0.write() {
            *slot = None;
        }
    }
}

impl F64Cell {
    /// A new, EMPTY cell.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            set: AtomicBool::new(false),
            bits: AtomicU64::new(0),
        }
    }

    /// Record the typed preset's answer for this lever.
    pub fn set(&self, value: f64) {
        self.bits.store(value.to_bits(), Ordering::Relaxed);
        self.set.store(true, Ordering::Release);
    }

    /// The typed preset's answer, or `None` when the key was absent.
    #[must_use]
    pub fn get(&self) -> Option<f64> {
        self.set
            .load(Ordering::Acquire)
            .then(|| f64::from_bits(self.bits.load(Ordering::Relaxed)))
    }

    /// Return the cell to its shipped, empty state. Tests only.
    #[cfg(test)]
    pub fn clear(&self) {
        self.set.store(false, Ordering::Release);
    }
}

/// Announce ONCE that a typed preset value was REFUSED by its own declaration.
///
/// A refusal means the yaml carried a value the lever's declared kind or range
/// cannot represent — an out-of-range fraction, a `usize` overflow. The
/// also an INVISIBLE one: a category would run the baseline while its preset
/// claimed a treatment, and the campaign has already banked vacuous nulls of
/// exactly that shape. So say so on stderr, once per lever per process.
fn announce_refusal(decl: &'static LeverDecl) {
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<BTreeSet<&'static str>>> = Mutex::new(None);
    let Ok(mut guard) = SEEN.lock() else {
        return;
    };
    if guard.get_or_insert_with(BTreeSet::new).insert(decl.name) {
        eprintln!(
            "[lever-delivery] preset value for {} is INADMISSIBLE for its declaration \
             and was REFUSED; the shipped resolution stands. Fix the yaml — a refused \
             key measures the baseline while claiming a treatment.",
            decl.name
        );
    }
}

/// Resolve a boolean lever over its typed preset cell.
///
/// An admissible environment token takes the authority defined by the
/// declaration's parser; otherwise an override-style kind may deliberately
/// retain the cell. With no effective environment override the cell decides,
/// and with neither layer the declaration default decides.
#[must_use]
pub fn resolve_bool(decl: &'static LeverDecl, cell: &BoolCell) -> bool {
    match read_over_config(decl, cell.get().map(LeverValue::Bool)) {
        Ok(resolved) => resolved.value.as_bool(),
        Err(_) => {
            announce_refusal(decl);
            read(decl).value.as_bool()
        }
    }
}

/// Resolve a lever whose ARMING RULE is a hand-written predicate over the raw
/// environment token — the [`read_raw`] reader shape.
///
/// Several gates deliberately keep their own unit-tested pure predicate in the
/// production path (`fn gate_on(raw: Option<&str>) -> bool`) because that
/// predicate IS the spec of the arming rule: `NY_SEQ_WEIGHTED_BUDGET`'s legacy
/// OPT-OUT (`!= "0"`), `NY_CGAN_STACKED_BACKWARD`'s exact-`"1"`. Routing those
/// through [`resolve_bool`] would silently swap the declaration's parser in for
/// the predicate, which for an opt-out lever inverts the shipped default. So the
/// predicate stays, and this only supplies the layer beneath it:
///
/// * the environment token PRESENT — the predicate decides, exactly as before,
///   in both directions. A present non-UTF-8 token is passed as `None`, matching
///   the legacy `env::var(...).ok().as_deref()` predicate input; crucially, it
///   still suppresses the typed cell;
/// * ABSENT — the typed preset cell decides;
/// * absent from both — `shipped_default`, which MUST be what the predicate
///   returns for `None`. `a_raw_gate_falls_back_to_its_shipped_default` pins
///   that for each caller.
pub fn resolve_raw_predicate<F>(
    decl: &'static LeverDecl,
    cell: &BoolCell,
    shipped_default: bool,
    predicate: F,
) -> bool
where
    F: FnOnce(Option<&str>) -> bool,
{
    match crate::env::read_raw_os(decl) {
        Some(token) => predicate(token.to_str()),
        None => cell.get().unwrap_or(shipped_default),
    }
}

/// Resolve an integer lever over its typed preset cell.
///
/// `None` means the lever resolved to `LeverValue::Unset` — the shape every
/// `DefaultSpec::Unset` integer declaration uses for "no override, keep the
/// reader's own contextual default".
#[must_use]
pub fn resolve_u64(decl: &'static LeverDecl, cell: &U64Cell) -> Option<u64> {
    match read_over_config(decl, cell.get().map(LeverValue::U64)) {
        Ok(resolved) => resolved.value.as_u64(),
        Err(_) => {
            announce_refusal(decl);
            read(decl).value.as_u64()
        }
    }
}

/// Resolve a floating-point lever over its typed preset cell.
///
/// `kind` decides which `LeverValue` the declaration will accept, so `Secs`
/// declarations get `Secs` and the two `F64*` kinds get `F64`; handing the
/// wrong one would be refused as inadmissible and silently keep the default.
#[must_use]
pub fn resolve_f64(decl: &'static LeverDecl, cell: &F64Cell) -> Option<f64> {
    let config = cell.get().map(|value| {
        if matches!(decl.kind, crate::decl::LeverKind::Secs) {
            LeverValue::Secs(value)
        } else {
            LeverValue::F64(value)
        }
    });
    match read_over_config(decl, config) {
        Ok(resolved) => resolved.value.as_f64(),
        Err(_) => {
            announce_refusal(decl);
            read(decl).value.as_f64()
        }
    }
}

/// Resolve a textual or enumerated lever over its typed preset cell.
///
/// `None` represents a declaration whose effective value is `Unset`; this is
/// distinct from an explicitly configured empty string for a `Text` lever.
#[must_use]
pub fn resolve_text(decl: &'static LeverDecl, cell: &TextCell) -> Option<String> {
    match read_over_config(decl, cell.get().map(LeverValue::Text)) {
        Ok(resolved) => resolved.value.as_str().map(str::to_owned),
        Err(_) => {
            announce_refusal(decl);
            read(decl).value.as_str().map(str::to_owned)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::{Bucket, DefaultSpec, LeverKind, MoatRisk, Provenance, Scope};

    const SCOPE: Scope = Scope {
        package: "ny-levers",
        subsystem: "gate-selftest",
    };

    static DARK_GATE: LeverDecl = LeverDecl {
        name: "NY_LEVERS_GATE_SELFTEST_DARK",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "self-test only",
        provenance: Provenance::Unmeasured {
            why_ok: "self-test only",
        },
        owner: SCOPE,
        readers: &[],
    };

    static ARMED_GATE: LeverDecl = LeverDecl {
        name: "NY_LEVERS_GATE_SELFTEST_ARMED",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "self-test only",
        provenance: Provenance::Unmeasured {
            why_ok: "self-test only",
        },
        owner: SCOPE,
        readers: &[],
    };

    static ROWS: LeverDecl = LeverDecl {
        name: "NY_LEVERS_GATE_SELFTEST_ROWS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "self-test only",
        provenance: Provenance::Unmeasured {
            why_ok: "self-test only",
        },
        owner: SCOPE,
        readers: &[],
    };

    static SHARE: LeverDecl = LeverDecl {
        name: "NY_LEVERS_GATE_SELFTEST_SHARE",
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: 1.0 },
        default: DefaultSpec::F64(0.35),
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "self-test only",
        provenance: Provenance::Unmeasured {
            why_ok: "self-test only",
        },
        owner: SCOPE,
        readers: &[],
    };

    static ROUTE: LeverDecl = LeverDecl {
        name: "NY_LEVERS_GATE_SELFTEST_ROUTE",
        kind: LeverKind::Enum(&["parallel", "sequential"]),
        default: DefaultSpec::Text("parallel"),
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "self-test only",
        provenance: Provenance::Unmeasured {
            why_ok: "self-test only",
        },
        owner: SCOPE,
        readers: &[],
    };

    /// The property the whole ratchet rests on: with no key set, a promoted
    /// lever resolves exactly as it did before promotion.
    #[test]
    fn an_empty_cell_is_the_shipped_resolution() {
        let dark = BoolCell::new();
        let armed = BoolCell::new();
        let rows = U64Cell::new();
        let share = F64Cell::new();
        let route = TextCell::new();

        assert!(!resolve_bool(&DARK_GATE, &dark));
        assert!(resolve_bool(&ARMED_GATE, &armed));
        assert_eq!(resolve_u64(&ROWS, &rows), None);
        assert_eq!(resolve_f64(&SHARE, &share), Some(0.35));
        assert_eq!(resolve_text(&ROUTE, &route).as_deref(), Some("parallel"));
    }

    #[test]
    fn a_written_cell_decides_when_the_environment_is_silent() {
        let dark = BoolCell::new();
        dark.set(true);
        assert!(resolve_bool(&DARK_GATE, &dark));

        // A default-ON declaration: the cell is the typed KILL SWITCH.
        let armed = BoolCell::new();
        armed.set(false);
        assert!(!resolve_bool(&ARMED_GATE, &armed));

        let rows = U64Cell::new();
        rows.set(256);
        assert_eq!(resolve_u64(&ROWS, &rows), Some(256));

        let share = F64Cell::new();
        share.set(0.6);
        assert_eq!(resolve_f64(&SHARE, &share), Some(0.6));

        let route = TextCell::new();
        route.set("sequential");
        assert_eq!(resolve_text(&ROUTE, &route).as_deref(), Some("sequential"));
    }

    /// An out-of-range preset value must not be reinterpreted. It is refused,
    /// the shipped resolution stands, and the refusal is announced.
    #[test]
    fn an_inadmissible_cell_value_is_refused_and_the_default_stands() {
        let share = F64Cell::new();
        share.set(4.0);
        assert_eq!(resolve_f64(&SHARE, &share), Some(0.35));

        let route = TextCell::new();
        route.set("typo");
        assert_eq!(resolve_text(&ROUTE, &route).as_deref(), Some("parallel"));
    }

    /// The `read_raw` shape: the caller's own predicate keeps deciding whenever
    /// the environment speaks, and the cell only fills the silence. The
    /// OPT-OUT case is the one that matters — a lever whose shipped state is
    /// ARMED must stay armed with an empty cell.
    #[test]
    fn a_raw_gate_falls_back_to_its_shipped_default() {
        let optout = BoolCell::new();
        // `NY_SEQ_WEIGHTED_BUDGET`'s rule, verbatim.
        let rule = |raw: Option<&str>| raw != Some("0");

        assert!(
            resolve_raw_predicate(&DARK_GATE, &optout, true, rule),
            "an empty cell must keep the shipped ARMED state of an opt-out lever"
        );
        optout.set(false);
        assert!(!resolve_raw_predicate(&DARK_GATE, &optout, true, rule));
        optout.set(true);
        assert!(resolve_raw_predicate(&DARK_GATE, &optout, true, rule));

        // And the exact-"1" shape, whose shipped state is dark.
        let dark = BoolCell::new();
        let exact_one = |raw: Option<&str>| raw == Some("1");
        assert!(!resolve_raw_predicate(&DARK_GATE, &dark, false, exact_one));
        dark.set(true);
        assert!(resolve_raw_predicate(&DARK_GATE, &dark, false, exact_one));
    }

    /// A non-UTF-8 value is PRESENT even though the legacy `env::var` reader
    /// handed `None` to its predicate. Presence must therefore suppress the
    /// typed cell without changing what the predicate sees.
    #[cfg(unix)]
    #[test]
    fn a_non_utf8_raw_gate_suppresses_the_typed_cell() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        // Routed through the blessed serialized helper rather than calling
        // `std::env::set_var` directly. clippy.toml's env wall is explicit that
        // "every mutation must route through the owning crate's blessed
        // serialized helper ... only those helper internals carry
        // #[allow(clippy::disallowed_methods)]", and the reason is real: this
        // mutates process-global state while the rest of the suite runs in
        // parallel. The previous hand-rolled `RemoveOnDrop` restored the key but
        // took no lock, so it raced; and its `#[allow(env_mutation)]` named a
        // Trust-only lint that does not suppress `clippy::disallowed_methods`
        // anyway, which is why this surfaced the moment the lint sweep first ran.
        ny_test_utils::env::with_env_edits(|env| {
            env.set_os(DARK_GATE.name, &OsString::from_vec(vec![0xff]));

            let armed_cell = BoolCell::new();
            armed_cell.set(true);
            assert!(
                !resolve_raw_predicate(&DARK_GATE, &armed_cell, false, |raw| raw == Some("1")),
                "present non-UTF-8 must not fall through to a true typed cell"
            );

            let disarmed_cell = BoolCell::new();
            disarmed_cell.set(false);
            assert!(
                resolve_raw_predicate(&DARK_GATE, &disarmed_cell, true, |raw| raw != Some("0")),
                "the legacy opt-out predicate must still receive None for non-UTF-8"
            );
        });
    }

    #[test]
    fn cells_start_empty_and_clear_back_to_empty() {
        let cell = BoolCell::new();
        assert_eq!(cell.get(), None);
        cell.set(true);
        assert_eq!(cell.get(), Some(true));
        cell.clear();
        assert_eq!(cell.get(), None);

        let rows = U64Cell::new();
        assert_eq!(rows.get(), None);
        rows.set(7);
        assert_eq!(rows.get(), Some(7));
        rows.clear();
        assert_eq!(rows.get(), None);

        let share = F64Cell::new();
        assert_eq!(share.get(), None);
        share.set(0.5);
        assert_eq!(share.get(), Some(0.5));
        share.clear();
        assert_eq!(share.get(), None);

        let route = TextCell::new();
        assert_eq!(route.get(), None);
        route.set("sequential");
        assert_eq!(route.get().as_deref(), Some("sequential"));
        route.clear();
        assert_eq!(route.get(), None);
    }
}
