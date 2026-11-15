// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Frozen policy for the VNN-COMP wrapper's upfront falsification lane.
//!
//! These controls used to be reread independently after the flight recorder
//! had already captured its raw-input snapshot. A run could therefore execute
//! a different treatment from the one in its receipt, and automated search
//! could not deliver the numeric controls through a category preset. The
//! scored command now resolves the master plus five budget controls here once
//! and carries that exact
//! [`crate::LeverSet`] into the runtime lane.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const VNNCOMP_UPFRONT: Scope = Scope {
    package: "ny-cli",
    subsystem: "vnncomp-upfront-attack",
};

const VNNCOMP_POSTBAB: Scope = Scope {
    package: "ny-cli",
    subsystem: "vnncomp-postbab-attack",
};

const MASTER_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "project attack.upfront_attack into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "resolve exact environment force/kill over typed preset and contextual automatic admission",
        site: "crates/ny-cli/src/commands/vnncomp.rs:UpfrontAttackPolicy::from_levers",
    },
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "select forced, disabled, automatic-disjunction, or typed relational wrapper admission",
        site: "crates/ny-cli/src/commands/vnncomp.rs:upfront_wrapper_route_from_snapshot",
    },
];

const BUDGET_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "project the typed attack budget field into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "resolve one immutable upfront budget policy from the same LeverSet as the receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:UpfrontAttackPolicy::from_levers",
    },
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "size both the planned attack-slice mirror and the authoritative upfront lane",
        site: "crates/ny-cli/src/commands/vnncomp.rs:upfront_attack_budget and try_upfront_falsify",
    },
];

const BNN_BUDGET_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "project the typed Sign-network attack budget into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "resolve one immutable Sign-network attack budget from the same LeverSet as the receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:UpfrontAttackPolicy::from_levers",
    },
    ReaderSite {
        scope: VNNCOMP_UPFRONT,
        role: "size both the planned attack-slice mirror and the authoritative Sign-network upfront lane",
        site: "crates/ny-cli/src/commands/vnncomp.rs:upfront_effective_budget_decision",
    },
];

const POSTBAB_RESERVE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "project bab.phase_budget.post_bab_reserve_secs into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "resolve one immutable post-BaB reserve from the same LeverSet as the flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:PostBabReservePolicy::from_levers",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "reduce the internal verifier deadline and preserve the requested tail for the outer falsifier",
        site: "crates/ny-cli/src/commands/vnncomp.rs:run_and_translate",
    },
];

const POSTBAB_RESTART_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "project attack.post_bab_restarts into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "resolve one immutable restart-diversity count from the same LeverSet as the flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:PostBabRestartPolicy::from_levers",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "re-enter the converged stage-0 alternation from deterministically diversified basins",
        site: "crates/ny-cli/src/commands/vnncomp.rs:try_postbab_falsify",
    },
];

const POSTBAB_ATTACK_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "project bab.phase_budget.vnncomp_post_bab_attack into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "resolve the source-sensitive wrapper kill policy from the same LeverSet as the flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:PostBabWrapperPolicy::from_levers",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "admit the deferred-PGD consumer and assign the unused scalable tail to proof or falsification",
        site: "crates/ny-cli/src/commands/vnncomp.rs:postbab_wrapper_route_from_snapshot",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "publish the effective and lower authorities plus the resulting route in a versioned execution event",
        site: "crates/ny-cli/src/commands/vnncomp.rs:PostBabWrapperPolicy::receipt_details",
    },
];

const POINT_FOLDED_TWIN_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "project attack.point_folded_twin into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "resolve one immutable twin-lane admission from the same LeverSet as the flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp/attack_twin_lane.rs:PointFoldedTwinPolicy::from_levers",
    },
    ReaderSite {
        scope: VNNCOMP_POSTBAB,
        role: "run exact-gradient PGD on the point-folded attack twin in the post-BaB tail when the real net is outside the exact-gradient fragment",
        site: "crates/ny-cli/src/commands/vnncomp.rs:try_postbab_falsify",
    },
];

declare_levers! {
    registry VNNCOMP_ENTRY_LEVERS;

    /// Exact environment override above contextual wrapper admission.
    pub UPFRONT_ATTACK = LeverDecl {
        name: "NY_UPFRONT_ATTACK",
        kind: LeverKind::BoolOverride,
        // Absence is a real third state: contextual automatic admission. Do
        // not encode it as false, which means an explicit kill at Config/env.
        default: DefaultSpec::Unset,
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Exact override for the VNN-COMP upfront exact-gradient falsification lane. An \
environment token of `1` forces the lane on every structurally supported row; \
`0` disables it. Any other byte sequence is recorded as ignored and leaves the \
typed `attack.upfront_attack` value, or the contextual automatic disjunction \
policy when that key is absent, authoritative. This differs deliberately from \
an ordinary exact Boolean whose malformed token suppresses config. Admission \
changes which counterexamples and proofs retain wall time, so it is verdict-\
affecting even though every published SAT still passes the trusted-oracle gate.",
        provenance: Provenance::Measured {
            commit: "a233053dd",
            date: "2026-07-29",
            artifact: "configs/vnncomp25/safenlp_2024.yaml and the commit's safenlp medical/hyperrectangle_2096 measurement",
            delta: "forced upfront attack converted the measured 20-second safenlp row from timeout to trusted-gated sat; automatic admission remains the absence baseline",
        },
        owner: VNNCOMP_UPFRONT,
        readers: MASTER_READERS,
    };

    /// Ordinary force/relational-lane wall cap in whole seconds.
    pub UPFRONT_ATTACK_CAP = LeverDecl {
        name: "NY_UPFRONT_ATTACK_CAP",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(8),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Whole-second wall cap for the ordinary forced or typed-relational upfront \
attack. The historical parser trims whitespace and accepts every u64, including \
zero. An admissible environment value overrides \
`attack.upfront_attack_cap_secs`; absence uses that typed value or the measured \
8-second default. The cap reallocates scored wall between falsification and \
branch-and-bound, so it changes which verdict is reachable before the deadline.",
        provenance: Provenance::Measured {
            commit: "4913e2479",
            date: "2026-07-14",
            artifact: "commit 4913e2479 upfront-apgd budget regression comparison",
            delta: "25 seconds starved soundnessbench; the retained 8-second cap restored its branch-and-bound budget while keeping fast robustness SATs",
        },
        owner: VNNCOMP_UPFRONT,
        readers: BUDGET_READERS,
    };

    /// Tighter automatic-disjunction wall cap in whole seconds.
    pub UPFRONT_ATTACK_AUTO_CAP = LeverDecl {
        name: "NY_UPFRONT_ATTACK_AUTO_CAP",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(4),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Whole-second clamp applied only to automatic multi-clause-disjunction \
admission: the contextual route, and multi-clause disjunctions under the typed \
`attack.upfront_attack: true` force, which keeps them on their automatic \
admission. The historical parser trims whitespace and accepts every \
u64. An admissible environment value overrides \
`attack.upfront_attack_auto_cap_secs`; absence uses that typed value or four \
seconds. Forced admissions (environment force, or a single-clause row under the \
typed force) and typed-relational routes deliberately retain the ordinary \
cap. This is a scored budget allocation and can change deadline reachability.",
        provenance: Provenance::Measured {
            commit: "14fe1182a",
            date: "2026-07-15",
            artifact: "commit 14fe1182a structural upfront-disjunction admission measurements",
            delta: "the automatic route retained nearby robustness counterexamples while a tighter four-second steal protected proof time on shared categories",
        },
        owner: VNNCOMP_UPFRONT,
        readers: BUDGET_READERS,
    };

    /// Fraction of live scored wall offered to the ordinary upfront lane.
    pub UPFRONT_ATTACK_FRAC = LeverDecl {
        name: "NY_UPFRONT_ATTACK_FRAC",
        // Historical contract: trim, finite, and 0 < value <= 1. The least
        // positive subnormal is the inclusive representation of strict > 0.
        kind: LeverKind::F64ClosedTrimmed {
            min: f64::from_bits(1),
            max: 1.0,
        },
        default: DefaultSpec::F64(0.08),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Fraction of live scored wall, after the trusted-publication reserve, offered to \
the ordinary upfront lane before the wall cap is applied. The parser trims, \
requires a finite value strictly above zero and at most one, and preserves \
subnormal positive inputs exactly as the historical reader did. An admissible \
environment value overrides `attack.upfront_attack_fraction`; absence uses that \
typed value or the measured 0.08 default. Fraction and cap jointly decide how \
much budget remains for branch-and-bound and are therefore verdict-affecting.",
        provenance: Provenance::Measured {
            commit: "4913e2479",
            date: "2026-07-14",
            artifact: "commit 4913e2479 upfront-apgd budget regression comparison",
            delta: "25 percent starved soundnessbench; the retained 8-percent share preserved proof time while keeping the fast attack wins",
        },
        owner: VNNCOMP_UPFRONT,
        readers: BUDGET_READERS,
    };

    /// Fraction of live scored wall offered to the upfront lane on Sign nets.
    pub BNN_ATTACK_FRAC = LeverDecl {
        name: "NY_BNN_ATTACK_FRAC",
        // Historical contract is identical to the ordinary fraction: trim,
        // finite, and 0 < value <= 1.
        kind: LeverKind::F64ClosedTrimmed {
            min: f64::from_bits(1),
            max: 1.0,
        },
        default: DefaultSpec::F64(0.5),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Fraction of live scored wall, after the trusted-publication reserve, offered to \
the upfront surrogate-gradient attack on a binarized Sign network before its \
Sign-specific wall cap is applied. The parser trims, requires a finite value \
strictly above zero and at most one, and preserves the historical 0.5 default. \
An admissible environment value overrides `attack.bnn_attack_fraction`; absence \
uses that typed value or the declaration default. The resulting budget is \
raised to the ordinary upfront budget, so exact execution evidence is required \
before a treatment can receive measurement credit.",
        provenance: Provenance::Measured {
            commit: "5d74508a9",
            date: "2026-07-27",
            artifact: "docs/TRAFFIC_BNN_FALSIFIER.md",
            delta: "the 0.5 share, paired with the 240-second cap, converted traffic_signs from 40 to 41/45; the artifact also records the later 0.9/460-second negative hard-tail sweep",
        },
        owner: VNNCOMP_UPFRONT,
        readers: BNN_BUDGET_READERS,
    };

    /// Whole-second wall cap for the upfront lane on Sign nets.
    pub BNN_ATTACK_CAP = LeverDecl {
        name: "NY_BNN_ATTACK_CAP",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(240),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Whole-second wall cap for the upfront surrogate-gradient attack on a binarized \
Sign network. The historical parser trims whitespace and accepts every u64, \
including zero. An admissible environment value overrides \
`attack.bnn_attack_cap_secs`; absence uses that typed value or the measured \
240-second default. The cap composes with the Sign-specific live-wall fraction \
and the ordinary upfront floor, reallocating scored time between falsification \
and branch-and-bound.",
        provenance: Provenance::Measured {
            commit: "5d74508a9",
            date: "2026-07-27",
            artifact: "docs/TRAFFIC_BNN_FALSIFIER.md",
            delta: "the 240-second cap, paired with the 0.5 share, converted traffic_signs from 40 to 41/45; the artifact also records the later 460-second/0.9 negative hard-tail sweep",
        },
        owner: VNNCOMP_UPFRONT,
        readers: BNN_BUDGET_READERS,
    };

    /// Fixed wall reserve for the outer post-BaB falsifier.
    pub POSTBAB_RESERVE_SECS = LeverDecl {
        name: "NY_POSTBAB_RESERVE_SECS",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Whole seconds held back from the internal verifier for the VNN-COMP wrapper's \
post-BaB falsifier. The historical parser trims whitespace and accepts every \
u64; zero disables the reserve. An admissible environment value overrides \
`bab.phase_budget.post_bab_reserve_secs`, and absence keeps the historical zero. \
The reserve never reduces a verifier window from ten or more seconds to below \
ten, and never extends an already-shorter window; oversized requests saturate \
rather than underflow. This is a \
real scored-budget reallocation: it can recover trusted-ORT SAT witnesses but \
can also remove proof time, so automated measurements require the versioned \
post-BaB execution receipt before they receive engagement credit.",
        provenance: Provenance::Measured {
            commit: "e5ac7dce6",
            date: "2026-07-19",
            artifact: "scratchpad/invprop_ab/acasxu_trio.csv and docs/PATH_TO_1_2026-07-18.md",
            delta: "a 100-second reserve produced trusted-ORT SAT candidates on ACAS Xu 5_3 and 1_6 (plus 1_5 in the host-local campaign); the retained artifacts are hypothesis-grade rather than current bank-grade, so no shipped preset is armed",
        },
        owner: VNNCOMP_POSTBAB,
        readers: POSTBAB_RESERVE_READERS,
    };

    /// Diversified-restart count for the converged post-BaB attack stack.
    pub POSTBAB_RESTARTS = LeverDecl {
        name: "NY_POSTBAB_RESTARTS",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Maximum number of diversified restarts of the post-BaB stage-0 alternation \
(ULP-jitter + Newton-feasibility) after the seeded descent self-terminates \
with `no improvement; stopping` while attack window remains. Zero (the \
default) preserves the historical single-descent stack byte-for-byte. A \
positive count re-enters the alternation from deterministically seeded \
basins — uniform, corner, and orthogonal-jitter-away-from-the-converged-point \
starts, cycled per restart, with the RNG seed derived from the row's model \
and property file names so a verdict is a pure function of the instance and \
never of ambient Date/random state. The best ORT margin is kept across \
restarts, every candidate still passes the unchanged in-attack zero-tolerance \
acceptance, and the caller still routes any witness through the unchanged \
`gate_sat_with_trusted_oracle`, so the lever can only ADD `sat` verdicts \
through the existing gate. Restarts spend only the post-BaB attack window: \
arming reserves two seconds per requested restart (at most half the window) \
out of the seeded DESCENT's share of that same window — the standard window \
is oversubscribed, so a purely-leftover restart lane would receive ~0 s at \
exactly the window that competes — and each restart then takes an equal \
share of the reserve (two-second floor). The verifier grant and every \
deadline outside the attack lane are untouched, so proof time is never \
reduced.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-zero is bit-identical to the shipped stack; the arming motivation is the 2026-08-31 window probe (reports/measured-2026/soundnessbench_window_probe_2026-08-31/README.md), which measured the stage-0 stack CONVERGENT — bit-identical fixed points at 11 s and 49.6 s windows — so diversity, not window length, is the open axis; armed effect is not yet measured, and any sat it produces still passes the unchanged trusted-ORT gate",
        },
        owner: VNNCOMP_POSTBAB,
        readers: POSTBAB_RESTART_READERS,
    };

    /// Independent enable/kill policy for the outer post-BaB falsifier.
    pub POSTBAB_ATTACK = LeverDecl {
        name: "NY_POSTBAB_ATTACK",
        // The historical environment contract is an exact kill override:
        // `0` disables, while `1`, malformed Unicode, and non-Unicode input do
        // not erase the independent typed-preset decision.
        kind: LeverKind::BoolKillOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "\
Independent admission policy for the VNN-COMP wrapper's trusted-ORT-gated \
post-BaB falsifier. The typed key is \
`bab.phase_budget.vnncomp_post_bab_attack`; false disables the wrapper, while \
omission/true preserves the historical enabled lane. The legacy environment \
is an exact kill switch: `0` disables, while `1` and every near-miss spelling \
leave the independent preset decision unchanged. A typed wrapper opt-out plus \
an explicitly zero engine `post_bab_pgd_fraction` assigns the scalable tail to \
proof; an environment-only kill never does. This source-sensitive distinction \
changes which consumer owns scored wall, so runtime must use the exact frozen \
resolution and publish its otherwise-hidden lower authority in the versioned \
flight policy event.",
        provenance: Provenance::Measured {
            commit: "e6acb4c7d",
            date: "2026-07-15",
            artifact: "commit e6acb4c7d post-BaB leftover-budget DLR-APGD measurement",
            delta: "the default-enabled trusted-ORT-gated lane converted soundnessbench model_5 from unknown to sat at 127 seconds; exact environment zero remains the rollback",
        },
        owner: VNNCOMP_POSTBAB,
        readers: POSTBAB_ATTACK_READERS,
    };

    /// Exact-gradient falsification on a point-folded ATTACK twin
    /// (#lsnc-attack-twin).
    pub ATTACK_POINT_FOLDED_TWIN = LeverDecl {
        name: "NY_ATTACK_POINT_FOLDED_TWIN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Admits the post-BaB point-folded-twin falsification lane. When the real graph \
is STATICALLY outside the exact point-VJP fragment because the exact loader \
failed a constant-only cone closed (a rounded binary32 product it will not \
fold), the lane builds an ATTACK-ONLY twin whose constant cones ONNX Runtime \
evaluated once and baked in (ny_onnx::attack_twin::PointFoldedAttackTwin, a \
distinct type with no route to any bound API), and runs projected Adam PGD \
with face-biased restarts on it, ORT-checking every iterate against the \
ORIGINAL model. A point is returned only when the trusted ONNX Runtime forward \
of the original model violates the property at zero tolerance, and the caller \
re-runs the unchanged gate_sat_with_trusted_oracle on it, so the lane can only \
ADD sat verdicts. It runs in the post-BaB tail — after the verdict is already \
non-sat — and takes the whole window only when the ordinary tail has no \
seeded stage to run (at most half of it otherwise), so it never takes proof \
time from branch-and-bound. Exact \"1\" arms it and exact \"0\" disarms it; \
absence defers to the typed key attack.point_folded_twin, whose absence is \
false. MoatRisk::High because arming changes which lane owns the post-BaB \
tail and therefore which counterexamples are reachable inside the budget.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-false; armed by exactly one shipped preset \
                     (configs/vnncomp25/lsnc_relu.yaml, pinned by \
                     only_the_lsnc_relu_preset_arms_the_point_folded_twin) AHEAD of its \
                     measurement, which is still owed (rows 14/17/58 x 3 reps at 25 s plus the \
                     lsnc_relu moat) before this becomes Measured; the motivating numpy prototype (exact-f64 twin gradient, Adam, alpha 0.03 x width, \
                     face-snapped restarts) reached the field's X_5 = +2 counterexample clause on \
                     lsnc_relu 2026 rows 14/17/58 in 6-9% of 100-step restarts; every candidate \
                     still passes the unchanged trusted-ORT gate and the lane has no \
                     verdict-bearing return variant, so it cannot produce unsat",
        },
        owner: VNNCOMP_POSTBAB,
        readers: POINT_FOLDED_TWIN_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, LeverValue, Source};

    #[test]
    fn numeric_parsers_match_the_trimmed_historical_contracts() {
        for (decl, default) in [
            (&UPFRONT_ATTACK_CAP, LeverValue::U64(8)),
            (&UPFRONT_ATTACK_AUTO_CAP, LeverValue::U64(4)),
            (&BNN_ATTACK_CAP, LeverValue::U64(240)),
            (&POSTBAB_RESERVE_SECS, LeverValue::U64(0)),
        ] {
            let accepted =
                read_over_config_with(decl, |_| Some(" 12 ".to_owned()), Some(LeverValue::U64(6)))
                    .expect("typed integer is admissible");
            assert_eq!(accepted.value, LeverValue::U64(12));
            assert_eq!(accepted.source, Source::LegacyEnv);

            let rejected =
                read_over_config_with(decl, |_| Some("nope".to_owned()), Some(LeverValue::U64(6)))
                    .expect("invalid environment input resolves to the declaration default");
            assert_eq!(rejected.value, default);
            assert_eq!(rejected.source, Source::LegacyEnvRejected);
        }

        for (decl, default) in [
            (&UPFRONT_ATTACK_FRAC, LeverValue::F64(0.08)),
            (&BNN_ATTACK_FRAC, LeverValue::F64(0.5)),
        ] {
            for raw in ["5e-324", "0.08", " 1 "] {
                let accepted = read_over_config_with(
                    decl,
                    |_| Some(raw.to_owned()),
                    Some(LeverValue::F64(0.2)),
                )
                .expect("typed fraction is admissible");
                assert_eq!(accepted.source, Source::LegacyEnv, "raw={raw:?}");
            }
            for raw in ["0", "-0", "-1", "NaN", "inf", "1.0001", ""] {
                let rejected = read_over_config_with(
                    decl,
                    |_| Some(raw.to_owned()),
                    Some(LeverValue::F64(0.2)),
                )
                .expect("invalid environment input resolves to the declaration default");
                assert_eq!(rejected.value, default, "{} raw={raw:?}", decl.name);
                assert_eq!(
                    rejected.source,
                    Source::LegacyEnvRejected,
                    "{} raw={raw:?}",
                    decl.name
                );
            }
        }
    }

    #[test]
    fn postbab_reserve_preserves_the_full_historical_u64_grammar() {
        for raw in ["+100", "0100", "\t100\n", "18446744073709551615"] {
            let resolved = read_over_config_with(
                &POSTBAB_RESERVE_SECS,
                |_| Some(raw.to_owned()),
                Some(LeverValue::U64(60)),
            )
            .expect("every parser-valid u64 reserve is admissible");
            let expected = if raw.contains("18446744073709551615") {
                u64::MAX
            } else {
                100
            };
            assert_eq!(resolved.value, LeverValue::U64(expected), "raw={raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "raw={raw:?}");
        }

        for raw in ["-1", "18446744073709551616", "100.0", ""] {
            let resolved = read_over_config_with(
                &POSTBAB_RESERVE_SECS,
                |_| Some(raw.to_owned()),
                Some(LeverValue::U64(60)),
            )
            .expect("invalid environment input resolves to the declaration default");
            assert_eq!(resolved.value, LeverValue::U64(0), "raw={raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "raw={raw:?}");
        }
    }

    #[test]
    fn postbab_attack_preserves_the_exact_kill_override_layer() {
        for (raw, expected, source) in [
            ("0", false, Source::LegacyEnv),
            ("1", false, Source::Config),
            ("false", false, Source::Config),
            ("0 ", false, Source::Config),
        ] {
            let resolved = read_over_config_with(
                &POSTBAB_ATTACK,
                |_| Some(raw.to_owned()),
                Some(LeverValue::Bool(false)),
            )
            .expect("typed post-BaB policy is admissible");
            assert_eq!(resolved.value.as_bool(), expected, "raw={raw:?}");
            assert_eq!(resolved.source, source, "raw={raw:?}");
            if matches!(source, Source::LegacyEnv) {
                assert_eq!(resolved.lower_value, LeverValue::Bool(false));
                assert_eq!(resolved.lower_source, Source::Config);
            } else {
                assert_eq!(resolved.ignored_raw.as_deref(), Some(raw));
            }
        }

        let default = read_over_config_with(&POSTBAB_ATTACK, |_| None, None)
            .expect("absence resolves to the historical enabled default");
        assert_eq!(default.value, LeverValue::Bool(true));
        assert_eq!(default.source, Source::Default);
    }

    /// #lsnc-attack-twin: absent everything the lane is dark; the typed preset
    /// value arrives with `Source::Config` (the route a scored run takes, since
    /// the harness exports no `NY_*`); an exact `0` still kills it.
    #[test]
    fn point_folded_twin_is_dark_by_default_and_the_typed_key_arms_it() {
        let absent = read_over_config_with(&ATTACK_POINT_FOLDED_TWIN, |_| None, None)
            .expect("absence is admissible");
        assert_eq!(absent.value, LeverValue::Bool(false));
        assert_eq!(absent.source, Source::Default);

        let typed = read_over_config_with(
            &ATTACK_POINT_FOLDED_TWIN,
            |_| None,
            Some(LeverValue::Bool(true)),
        )
        .expect("the typed key is admissible");
        assert_eq!(typed.value, LeverValue::Bool(true));
        assert_eq!(typed.source, Source::Config);
        assert_ne!(typed.source, Source::LegacyEnv);

        let killed = read_over_config_with(
            &ATTACK_POINT_FOLDED_TWIN,
            |_| Some("0".to_owned()),
            Some(LeverValue::Bool(true)),
        )
        .expect("an exact 0 is an admissible disarm");
        assert_eq!(killed.value, LeverValue::Bool(false));
        assert_eq!(killed.source, Source::LegacyEnv);
    }

    #[cfg(unix)]
    #[test]
    fn postbab_attack_non_unicode_input_leaves_typed_policy_authoritative() {
        use std::os::unix::ffi::OsStringExt;

        let raw = crate::RawLeverInputs::capture_with(crate::all(), |name| {
            (name == POSTBAB_ATTACK.name).then(|| std::ffi::OsString::from_vec(vec![0xff, b'0']))
        });
        let levers = crate::LeverSet::resolve_layered(crate::all(), &raw, |decl| {
            std::ptr::eq(decl, &raw const POSTBAB_ATTACK).then_some(LeverValue::Bool(false))
        })
        .expect("typed post-BaB policy is admissible");
        let resolved = levers
            .resolved(&POSTBAB_ATTACK)
            .expect("post-BaB attack is registered");
        assert_eq!(resolved.value, LeverValue::Bool(false));
        assert_eq!(resolved.source, Source::Config);
        assert_eq!(resolved.env_utf8, Some(false));
        assert!(resolved.ignored_raw.is_some());
    }
}
