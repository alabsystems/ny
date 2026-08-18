// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The declaration vocabulary: [`LeverDecl`] and the enums it is built from.
//!
//! A declaration is the ONLY admissible description of a lever. It records
//! what the lever is, who may choose its value ([`Bucket`]), what it can cost
//! if wrong ([`MoatRisk`]), why its shipped default is defensible
//! ([`Provenance`]), who owns it ([`Scope`]) and every site that reads it
//! ([`ReaderSite`]).

use std::fmt;

/// A Cargo package plus a stable subsystem name inside it.
///
/// Used both for the declaring owner and for each governed read site, so a
/// cross-crate lever is visible as such from the declaration alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scope {
    /// Cargo package name, e.g. `"ny-propagate"`.
    pub package: &'static str,
    /// Stable registry grouping inside that package, e.g. `"root-alpha"`.
    pub subsystem: &'static str,
}

/// One governed consumer of a lever.
///
/// `site` is a path plus either a reviewed line or a stable symbol naming the
/// read. It is documentation, not a compile-time guarantee — line numbers
/// drift, while symbols can move during refactors. Phase 2 replaces the raw
/// read with a `LeverSet` lookup and the pointer becomes stable by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReaderSite {
    /// Package + subsystem doing the reading.
    pub scope: Scope,
    /// What this consumer uses the value FOR (its semantic role).
    pub role: &'static str,
    /// `path:line` or `path:symbol` locator of the read at last review.
    pub site: &'static str,
}

/// The value shape a lever parses to, including its admissible range.
///
/// The range lives in the kind (rather than in a separate policy field)
/// because the legacy readers fold parse and range-filter into one expression;
/// splitting them would invite a migration that silently widens a range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeverKind {
    /// A gate armed by the exact string `"1"` and nothing else.
    ///
    /// See [`crate::read`] for why the chokepoint refuses `"true"`/`"yes"`.
    Bool,
    /// A gate armed by the variable being SET, whatever its value.
    ///
    /// `NY_X=0`, `NY_X=`, and `NY_X=no` all arm a presence lever — that is the
    /// point, and it is why this cannot be folded into [`Self::Bool`]. Several
    /// in-tree readers are `env::var_os("NY_…").is_some()`, and declaring one of
    /// them as `Bool` would put a value in the flight receipt that DISAGREES
    /// with the decision the reader actually made: the receipt would record
    /// `false` for `NY_X=0` while the reader treated it as armed. A receipt that
    /// contradicts the run is worse than no receipt.
    ///
    /// Presence is also the one kind whose parser cannot reject anything, so it
    /// never produces [`crate::Source::LegacyEnvRejected`]. Read it through
    /// [`crate::read_presence`], which preserves the `var_os` lookup — a
    /// non-UTF-8 value is still PRESENT, and `env::var` would have dropped it.
    Presence,
    /// Unsigned integer; a malformed value falls back to the default.
    U64,
    /// Unsigned integer after trimming surrounding whitespace.
    ///
    /// This preserves legacy readers that use `trim().parse::<u64>()`. It is
    /// distinct from both [`Self::U64`] (which rejects whitespace) and
    /// [`Self::UsizeTrimmed`] (whose upper bound is platform-dependent).
    U64Trimmed,
    /// Platform-sized unsigned integer after trimming surrounding whitespace.
    ///
    /// This preserves the legacy `trim().parse::<usize>()` contract used by
    /// dimension and row-count overrides. It is separate from [`Self::U64`]
    /// because accepting whitespace is part of a lever's public parser, not a
    /// harmless convenience that every integer reader may silently acquire.
    UsizeTrimmed,
    /// Finite `f64` restricted to the OPEN interval `(min, max)`.
    ///
    /// Open on both ends because that is what the in-tree fraction readers
    /// actually do — e.g. `alpha_zero_yield_frac` filters
    /// `is_finite() && (0.0..0.9).contains(v) && v > 0.0`, which is exactly
    /// `F64Open { min: 0.0, max: 0.9 }`.
    F64Open {
        /// Exclusive lower bound.
        min: f64,
        /// Exclusive upper bound.
        max: f64,
    },
    /// Finite `f64` in the CLOSED interval `[min, max]`, after trimming.
    ///
    /// Distinct from [`Self::F64Open`] because for some fractions an endpoint is
    /// a meaningful setting rather than an out-of-range value — `NY_KFSB_SIM_SHARE`
    /// filters `(0.0..=1.0).contains(v)` and documents `0` as the kill switch that
    /// restores unbounded ranking. Declaring that as `F64Open { 0.0, 1.0 }` would
    /// reject the kill switch and silently resolve it to the 0.35 default, which
    /// is precisely the parser disagreement the registry exists to prevent.
    F64ClosedTrimmed {
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },
    /// Finite, non-negative seconds.
    Secs,
    /// One of a fixed set of strings; anything else falls back to the default.
    Enum(&'static [&'static str]),
    /// Free-form text (paths, comma lists) taken verbatim.
    Text,
}

/// The shipped default, used whenever the environment does not supply an
/// admissible value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultSpec {
    /// Boolean default (`false` = dark).
    Bool(bool),
    /// Integer default.
    U64(u64),
    /// Floating-point default.
    F64(f64),
    /// Seconds default.
    Secs(f64),
    /// Text default.
    Text(&'static str),
    /// No default value at all: absent means the feature is simply off, and
    /// the resolved value is [`crate::LeverValue::Unset`].
    Unset,
}

/// Who is allowed to choose a lever's value.
///
/// There is deliberately **no `Dead` variant**. A dead lever cannot be
/// declared; it must be deleted — read site, surrounding plumbing and all.
/// That is the whole enforcement mechanism for the deletion phase of the
/// lever-debt plan: if the only way to describe a lever is to say who may set
/// it, then "nobody, it does nothing" has no encoding and the author is forced
/// to delete the code instead of parking it in the registry forever.
///
/// A lever that is unwired *by design* (staged work, like the CPU arm of
/// `NY_EFT_ERR`) is NOT dead: it has a reader that is deliberately not yet
/// reached, and it gets [`Provenance::Unmeasured`] plus a reader-site entry
/// recording that fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    /// Ships armed. Requires evidence — see [`Provenance`].
    DefaultOn,
    /// The value is chosen by an automated policy from instance features.
    Auto,
    /// A user-facing choice; must become a real CLI flag.
    Cli,
    /// Diagnostic / staged / A-B-only. Never advertised, never automatic.
    Debug,
}

impl Bucket {
    /// Stable lowercase name for receipts and `ny levers` output.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DefaultOn => "default_on",
            Self::Auto => "auto",
            Self::Cli => "cli",
            Self::Debug => "debug",
        }
    }
}

/// How badly a wrong value can hurt: `High` means it can move a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoatRisk {
    /// Cannot change any published bound or verdict.
    None,
    /// Performance or diagnostics only, but touches a verdict-carrying path.
    Low,
    /// Can change a published bound or verdict on the authoritative route.
    High,
}

impl MoatRisk {
    /// Stable lowercase name for receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::High => "high",
        }
    }
}

/// Why the shipped default is defensible.
///
/// This is the moat rule in type form: an armed default must be
/// [`Provenance::ValueNeutral`], [`Provenance::Measured`] or
/// [`Provenance::Guard`]. [`Provenance::Unmeasured`] can never be
/// [`Bucket::DefaultOn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Value-neutral by construction: the OFF arm is bit-identical, or the ON
    /// arm removes provably-unused work. Requires a named parity test.
    ValueNeutral {
        /// Test path or `module::test_name` proving the parity claim.
        parity_test: &'static str,
    },
    /// A discriminating measurement on the current sound path.
    Measured {
        /// Commit the measurement was taken at.
        commit: &'static str,
        /// ISO date of the measurement.
        date: &'static str,
        /// Where the numbers live (doc, ledger row, receipt JSON).
        artifact: &'static str,
        /// The measured delta, stated so it can be contradicted.
        delta: &'static str,
    },
    /// Runs, but its effect is not measured on the current sound path.
    /// Cannot be [`Bucket::DefaultOn`].
    Unmeasured {
        /// Why shipping it in its current bucket is nonetheless safe.
        why_ok: &'static str,
    },
    /// A safety guard whose OFF arm removes a check. Cannot be
    /// [`Bucket::Auto`] — a policy may never disarm a guard.
    Guard {
        /// What the guard protects.
        protects: &'static str,
    },
}

impl Provenance {
    /// Stable tag for receipts and grouping.
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::ValueNeutral { .. } => "value_neutral",
            Self::Measured { .. } => "measured",
            Self::Unmeasured { .. } => "unmeasured",
            Self::Guard { .. } => "guard",
        }
    }
}

/// One centrally owned declaration of one `NY_*` lever.
///
/// Construct these only through [`crate::declare_levers`], which also files
/// them in a module registry slice; a `LeverDecl` that no registry lists is
/// invisible to [`crate::collect`] and therefore to the receipt.
#[derive(Debug)]
pub struct LeverDecl {
    /// The environment variable name, e.g. `"NY_INTERM_REFINE"`.
    pub name: &'static str,
    /// Value shape and admissible range.
    pub kind: LeverKind,
    /// The shipped default.
    pub default: DefaultSpec,
    /// Who may choose the value.
    pub bucket: Bucket,
    /// Blast radius if wrong.
    pub moat: MoatRisk,
    /// What it does when set. Prose, written for someone who has never seen
    /// the read site.
    pub doc: &'static str,
    /// Why the shipped default is defensible.
    pub provenance: Provenance,
    /// The package + subsystem that owns the declaration.
    pub owner: Scope,
    /// Every governed read site.
    ///
    /// A lever read from more than one package MUST list them all: that is
    /// what makes a shared gate a *declared* shared gate rather than a
    /// discovered collision, and it is what lets [`crate::collect`] accept the
    /// same declaration being exported by two module registries.
    pub readers: &'static [ReaderSite],
}

impl LeverDecl {
    /// Distinct packages among the declared read sites.
    ///
    /// Cheap linear scan over a slice that is never longer than a handful.
    pub fn reader_packages(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();
        for r in self.readers {
            if !out.contains(&r.scope.package) {
                out.push(r.scope.package);
            }
        }
        out
    }

    /// True when the declaration documents two or more read sites.
    ///
    /// This is the predicate that distinguishes a deliberate multi-reader
    /// lever (`NY_EFT_ERR`) from a declaration that was accidentally exported
    /// twice.
    pub const fn is_multi_reader(&self) -> bool {
        self.readers.len() > 1
    }
}

impl fmt::Display for LeverDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}/{}] owned by {}::{}",
            self.name,
            self.bucket.as_str(),
            self.moat.as_str(),
            self.owner.package,
            self.owner.subsystem
        )
    }
}
