// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ProofProfile`: which artifact sinks run, and at what audit rates
//! (ENGINE_PROGRAM §1.7, milestone M2a).
//!
//! # What it is
//!
//! A two-valued, process-wide choice made ONCE by the entry point:
//!
//! * `ny vnncomp v1` (what `run_instance.sh` runs) resolves to
//!   [`ProofProfile::Competition`];
//! * every other subcommand resolves to [`ProofProfile::Proof`];
//! * `--proof-profile {proof,competition}` overrides either, for banking and
//!   moat runs, and is recorded as [`ProfileSource::Cli`].
//!
//! It is deliberately NOT a preset key and NOT an environment variable. That is
//! the same contract the artifact authority already carried ("deliberately not
//! configurable through presets"), and it means `run_instance.sh`, which exports
//! no lever variables, cannot be surprised by it.
//!
//! # What it may select — exactly two things
//!
//! 1. which proof SINK is wired in (certificate sidecars, proof streams): the
//!    `Proof` profile writes them, `Competition` wires a null sink;
//! 2. the [`AuditRates`] of the downgrade-only guards (how many containment
//!    points, how often an optimised certified kernel is cross-checked).
//!
//! # What it may NOT do
//!
//! Reach certification. No certifier module may name this type (ratcheted in
//! `ny-propagate`'s `certify` tests), every certificate, the cover gate, the sat
//! gate and the cheap guards run identically in both profiles, and every guard
//! rate is strictly positive in both (a profile may sample a guard less often;
//! it may never switch one off).
//!
//! # The safety argument, once
//!
//! Every item the profile drops is write-only (its output flows into a sink no
//! verdict path reads) or downgrade-only (it can map a verdict to `unknown`,
//! never construct one). Removing either kind leaves the verdict-bearing path
//! bit-identical, so a correct verdict can only arrive EARLIER, never differ.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// The two profiles. `Proof` is the default: an unresolved caller (a unit
/// test, a library consumer) gets the complete audit rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ProofProfile {
    /// Full proof artifacts and full audit rates (interactive runs).
    #[default]
    Proof,
    /// Null proof sink and sampled audits (the scored VNN-COMP entry point).
    Competition,
}

impl ProofProfile {
    /// Stable token for observables (`[proof-profile] mode=<token>`).
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Proof => "proof",
            Self::Competition => "competition",
        }
    }

    /// Parse a CLI spelling. Exact tokens only: a typo must be an error at the
    /// flag, never a silent fallback to the other profile.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "proof" => Some(Self::Proof),
            "competition" => Some(Self::Competition),
            _ => None,
        }
    }

    /// Whether proof ARTIFACTS (certificate sidecars, proof streams) are
    /// written by default. Write-only by construction: no verdict path reads
    /// an artifact during the run.
    #[must_use]
    pub const fn writes_artifacts(self) -> bool {
        matches!(self, Self::Proof)
    }

    /// The audit rates this profile selects.
    #[must_use]
    pub const fn audit_rates(self) -> AuditRates {
        match self {
            Self::Proof => AuditRates::PROOF,
            Self::Competition => AuditRates::COMPETITION,
        }
    }
}

impl fmt::Display for ProofProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.token())
    }
}

/// Where the resolved profile came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ProfileSource {
    /// Resolved from the entry point alone.
    #[default]
    Entry,
    /// An explicit `--proof-profile` override.
    Cli,
}

impl ProfileSource {
    /// Stable token for observables (`source=<token>`).
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Entry => "entry",
            Self::Cli => "cli",
        }
    }
}

/// The entry point that resolves the profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryPoint {
    /// `ny vnncomp v1`, the scored protocol.
    VnncompV1,
    /// Every other subcommand.
    Interactive,
}

/// A profile together with its provenance. This is the value that is frozen,
/// printed and written to the flight sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ResolvedProofProfile {
    /// The profile.
    pub mode: ProofProfile,
    /// How it was chosen.
    pub source: ProfileSource,
}

impl ResolvedProofProfile {
    /// Resolve the profile for `entry`, with an optional CLI override.
    #[must_use]
    pub const fn resolve(entry: EntryPoint, cli: Option<ProofProfile>) -> Self {
        match cli {
            Some(mode) => Self {
                mode,
                source: ProfileSource::Cli,
            },
            None => Self {
                mode: match entry {
                    EntryPoint::VnncompV1 => ProofProfile::Competition,
                    EntryPoint::Interactive => ProofProfile::Proof,
                },
                source: ProfileSource::Entry,
            },
        }
    }

    /// The one observable line: `[proof-profile] mode=<m> source=<entry|cli>`.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "[proof-profile] mode={} source={}",
            self.mode.token(),
            self.source.token()
        )
    }

    /// The audit rates of the resolved profile.
    #[must_use]
    pub const fn audit_rates(&self) -> AuditRates {
        self.mode.audit_rates()
    }
}

/// Sampling rates of the downgrade-only guards (ENGINE_PROGRAM §1.7 (a′)).
///
/// Every field is strictly positive in BOTH profiles: the profile chooses how
/// often a guard samples, never whether it runs. `audit_rates_never_disable_a_guard`
/// pins that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuditRates {
    /// G4: re-check one in `crosscheck_every` outputs of an optimised certified
    /// kernel against its scalar reference (plus every one at shallow depth;
    /// the consumer decides that part). `1` = every output.
    pub crosscheck_every: u32,
    /// G2/G3: points sampled per certified frame or state by the containment
    /// and spec-agreement guards (an upper bound; a guard may take fewer when
    /// its own cost admission says the remaining budget cannot pay for more).
    pub containment_points: u32,
    /// B6: verbose per-phase telemetry beyond the one summary line per phase.
    pub verbose_telemetry: bool,
}

impl AuditRates {
    /// Proof profile: full rates.
    pub const PROOF: Self = Self {
        crosscheck_every: 1,
        containment_points: 32,
        verbose_telemetry: true,
    };
    /// Competition profile: sampled rates. Still strictly positive.
    pub const COMPETITION: Self = Self {
        crosscheck_every: 256,
        containment_points: 16,
        verbose_telemetry: false,
    };
}

impl Default for AuditRates {
    fn default() -> Self {
        Self::PROOF
    }
}

static FROZEN: OnceLock<ResolvedProofProfile> = OnceLock::new();

/// Freeze the process profile. The FIRST call wins and later calls return the
/// already-frozen value: an entry point resolves once, and nothing downstream
/// may re-resolve it.
pub fn freeze(resolved: ResolvedProofProfile) -> ResolvedProofProfile {
    *FROZEN.get_or_init(|| resolved)
}

/// The frozen profile, if an entry point has resolved one.
#[must_use]
pub fn frozen() -> Option<ResolvedProofProfile> {
    FROZEN.get().copied()
}

/// The audit rates in force: the frozen profile's, or the complete `Proof`
/// rates when nothing was frozen (tests and library callers).
#[must_use]
pub fn audit_rates() -> AuditRates {
    frozen().unwrap_or_default().audit_rates()
}

static ANNOUNCED: AtomicBool = AtomicBool::new(false);

/// Print `resolved`'s `[proof-profile] mode=<m> source=<entry|cli>` line to
/// stderr ONCE per process: the first entry point to announce prints it, every
/// later call (a second entry point, a nested handler) prints nothing. Returns
/// the line when this call printed it.
pub fn announce(resolved: ResolvedProofProfile) -> Option<String> {
    let line = announce_with(&ANNOUNCED, resolved)?;
    eprintln!("{line}");
    Some(line)
}

/// [`announce`]'s once-per-flag rule without the process global (tests): the
/// observable line the first time `flag` is claimed, `None` afterwards.
#[must_use]
pub fn announce_with(flag: &AtomicBool, resolved: ResolvedProofProfile) -> Option<String> {
    (!flag.swap(true, Ordering::AcqRel)).then(|| resolved.line())
}

#[cfg(test)]
mod tests {
    use super::{AuditRates, EntryPoint, ProfileSource, ProofProfile, ResolvedProofProfile};

    #[test]
    fn entry_points_resolve_without_any_override() {
        let scored = ResolvedProofProfile::resolve(EntryPoint::VnncompV1, None);
        assert_eq!(scored.mode, ProofProfile::Competition);
        assert_eq!(scored.source, ProfileSource::Entry);
        let interactive = ResolvedProofProfile::resolve(EntryPoint::Interactive, None);
        assert_eq!(interactive.mode, ProofProfile::Proof);
        assert_eq!(interactive.source, ProfileSource::Entry);
    }

    #[test]
    fn cli_override_wins_in_both_directions_and_is_recorded() {
        for entry in [EntryPoint::VnncompV1, EntryPoint::Interactive] {
            for mode in [ProofProfile::Proof, ProofProfile::Competition] {
                let r = ResolvedProofProfile::resolve(entry, Some(mode));
                assert_eq!(r.mode, mode);
                assert_eq!(r.source, ProfileSource::Cli);
            }
        }
    }

    #[test]
    fn profile_line_is_the_documented_observable() {
        let r = ResolvedProofProfile::resolve(EntryPoint::VnncompV1, None);
        assert_eq!(r.line(), "[proof-profile] mode=competition source=entry");
        let r = ResolvedProofProfile::resolve(EntryPoint::VnncompV1, Some(ProofProfile::Proof));
        assert_eq!(r.line(), "[proof-profile] mode=proof source=cli");
    }

    #[test]
    fn parse_is_exact_and_round_trips() {
        for mode in [ProofProfile::Proof, ProofProfile::Competition] {
            assert_eq!(ProofProfile::parse(mode.token()), Some(mode));
        }
        for bad in ["", "Proof", "COMPETITION", "comp", "fast", "1"] {
            assert_eq!(ProofProfile::parse(bad), None, "{bad:?} must not parse");
        }
    }

    #[test]
    fn audit_rates_never_disable_a_guard() {
        for rates in [AuditRates::PROOF, AuditRates::COMPETITION] {
            assert!(rates.crosscheck_every >= 1);
            assert!(rates.containment_points >= 1);
        }
        // Proof samples at least as densely as Competition on every axis.
        const {
            assert!(AuditRates::PROOF.crosscheck_every <= AuditRates::COMPETITION.crosscheck_every);
            assert!(
                AuditRates::PROOF.containment_points >= AuditRates::COMPETITION.containment_points
            );
        };
        assert_eq!(ProofProfile::Proof.audit_rates(), AuditRates::PROOF);
        assert_eq!(
            ProofProfile::Competition.audit_rates(),
            AuditRates::COMPETITION
        );
        assert_eq!(AuditRates::default(), AuditRates::PROOF);
    }

    #[test]
    fn profile_line_is_announced_once_per_process() {
        use std::sync::atomic::AtomicBool;
        let flag = AtomicBool::new(false);
        let scored = ResolvedProofProfile::resolve(EntryPoint::VnncompV1, None);
        assert_eq!(
            super::announce_with(&flag, scored).as_deref(),
            Some("[proof-profile] mode=competition source=entry")
        );
        // A second entry point (or a nested handler) prints nothing, whatever
        // it resolved.
        for later in [
            scored,
            ResolvedProofProfile::resolve(EntryPoint::Interactive, Some(ProofProfile::Proof)),
        ] {
            assert_eq!(super::announce_with(&flag, later), None);
        }
    }

    #[test]
    fn only_the_proof_profile_writes_artifacts() {
        assert!(ProofProfile::Proof.writes_artifacts());
        assert!(!ProofProfile::Competition.writes_artifacts());
    }
}
