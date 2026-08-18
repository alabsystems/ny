// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-dark controls for the verdict-neutral exact-star measurement lane.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const STAR_MEASUREMENT_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "beta-crown-star-measurement",
};

const READER: ReaderSite = ReaderSite {
    scope: STAR_MEASUREMENT_SCOPE,
    role: "configure the verdict-neutral exact-star measurement probe",
    site: "crates/ny-cli/src/commands/beta_crown/star_candidate.rs:DarkStarProbe::from_env",
};

declare_levers! {
    registry STAR_LEVERS;

    /// `NY_STAR_DARK_SECONDS` — positive wall-clock budget and master arm.
    pub STAR_DARK_SECONDS = LeverDecl {
        name: "NY_STAR_DARK_SECONDS",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Sets the exact-star measurement budget in whole seconds. Zero or absence keeps \
the lane dark; a positive trimmed `u64` arms it. Malformed, negative, \
non-Unicode, or overflowing input is rejected to zero. The probe cannot emit \
a verdict, but it runs synchronously on the scored thread and can consume time \
that the authoritative verifier needed, so arming it can change a timeout \
verdict and remains a High-risk Debug choice.",
        provenance: Provenance::Unmeasured {
            why_ok: "default zero is dark; no retained timing experiment qualifies synchronous star measurement for automatic use",
        },
        owner: STAR_MEASUREMENT_SCOPE,
        readers: &[READER],
    };

    /// `NY_STAR_DARK_MAX_STARS` — runaway work cap.
    pub STAR_DARK_MAX_STARS = LeverDecl {
        name: "NY_STAR_DARK_MAX_STARS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(50_000_000),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Caps the number of stars explored by an armed exact-star measurement. The \
default is 50,000,000, with the wall-clock budget intended to bind first. \
Surrounding whitespace is accepted before parsing as `usize`; malformed, \
negative, non-Unicode, or platform-overflowing input falls back to the default. \
This knob cannot arm the lane by itself, but on an armed scored run it changes \
work and therefore can change whether the authoritative verifier times out.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while the default-dark master budget is zero; armed timing effects remain measurement-only",
        },
        owner: STAR_MEASUREMENT_SCOPE,
        readers: &[READER],
    };

    /// `NY_STAR_DARK_MAX_DEPTH` — exact-star search depth cap.
    pub STAR_DARK_MAX_DEPTH = LeverDecl {
        name: "NY_STAR_DARK_MAX_DEPTH",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(512),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Sets the maximum exact-star search depth for an armed measurement, default \
512. Surrounding whitespace is accepted before parsing as `usize`; invalid \
input falls back to 512. It is inert while the master budget is zero, but on \
an armed scored run it changes synchronous work and can move a timeout verdict.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while the default-dark master budget is zero; armed timing effects remain measurement-only",
        },
        owner: STAR_MEASUREMENT_SCOPE,
        readers: &[READER],
    };

    /// `NY_STAR_DARK_DUAL_ITERS` — dual-relaxation iteration count.
    pub STAR_DARK_DUAL_ITERS = LeverDecl {
        name: "NY_STAR_DARK_DUAL_ITERS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(32),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Sets the dual-relaxation iteration count for an armed exact-star measurement, \
default 32. Surrounding whitespace is accepted before parsing as `usize`; \
invalid input falls back to 32. It is inert while the master budget is zero, \
but on an armed scored run it changes synchronous work and can move a timeout \
verdict.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while the default-dark master budget is zero; armed timing effects remain measurement-only",
        },
        owner: STAR_MEASUREMENT_SCOPE,
        readers: &[READER],
    };

    /// `NY_STAR_DARK_INPUT_SPLIT` — input-vs-activation branching preference.
    pub STAR_DARK_INPUT_SPLIT = LeverDecl {
        name: "NY_STAR_DARK_INPUT_SPLIT",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the armed measurement's branching preference: zero chooses activation \
splits and every other valid `usize` chooses input splits, preserving the \
legacy numeric parser. Absence or invalid input uses one (input splitting). It \
cannot arm the lane, but changes synchronous search work and can move a scored \
timeout when the master budget is positive.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while the default-dark master budget is zero; armed timing effects remain measurement-only",
        },
        owner: STAR_MEASUREMENT_SCOPE,
        readers: &[READER],
    };

    /// `NY_STAR_DARK_EXACT_BELOW` — exact-LP unstable-neuron threshold.
    pub STAR_DARK_EXACT_BELOW = LeverDecl {
        name: "NY_STAR_DARK_EXACT_BELOW",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Sets the unstable-neuron threshold below which an armed exact-star measurement \
may invoke exact LP, default zero. Surrounding whitespace is accepted before \
parsing as `usize`; invalid input falls back to zero. It cannot arm the lane, \
but changes synchronous search work and can move a scored timeout when the \
master budget is positive.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while the default-dark master budget is zero; armed timing effects remain measurement-only",
        },
        owner: STAR_MEASUREMENT_SCOPE,
        readers: &[READER],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    fn resolve(decl: &'static LeverDecl, raw: Option<&str>) -> (LeverValue, Source) {
        let resolved = read_with(decl, |_| raw.map(str::to_owned));
        (resolved.value, resolved.source)
    }

    #[test]
    fn seconds_preserves_trimmed_positive_master_arm() {
        for (raw, value, source) in [
            (None, 0, Source::Default),
            (Some("0"), 0, Source::LegacyEnv),
            (Some(" 7 "), 7, Source::LegacyEnv),
            (Some("-1"), 0, Source::LegacyEnvRejected),
            (Some("nope"), 0, Source::LegacyEnvRejected),
        ] {
            assert_eq!(
                resolve(&STAR_DARK_SECONDS, raw),
                (LeverValue::U64(value), source),
                "{raw:?}"
            );
        }
    }

    #[test]
    fn search_knob_defaults_and_numeric_branching_match_legacy() {
        for (decl, default) in [
            (&STAR_DARK_MAX_STARS, 50_000_000),
            (&STAR_DARK_MAX_DEPTH, 512),
            (&STAR_DARK_DUAL_ITERS, 32),
            (&STAR_DARK_INPUT_SPLIT, 1),
            (&STAR_DARK_EXACT_BELOW, 0),
        ] {
            assert_eq!(resolve(decl, None).0, LeverValue::U64(default));
            assert_eq!(resolve(decl, Some(" 9 ")).0, LeverValue::U64(9));
            assert_eq!(resolve(decl, Some("bad")).0, LeverValue::U64(default));
        }
        assert_eq!(
            resolve(&STAR_DARK_INPUT_SPLIT, Some("0")).0,
            LeverValue::U64(0)
        );
        assert_eq!(
            resolve(&STAR_DARK_INPUT_SPLIT, Some("2")).0,
            LeverValue::U64(2)
        );
    }
}
