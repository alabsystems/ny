// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The stability threshold of the sequential CROWN-IBP partial walk's Dense
//! unstable-rows seed (#dense-sparse-seed).
//!
//! `network/ibp/crown_partial.rs` seeds a Dense-carrier target's backward walk
//! with one row per IBP-UNSTABLE output neuron instead of the full identity
//! only when at least `min_stable` of the target's neurons are IBP-stable. The
//! historical threshold is the literal `0.9` (#3599 Phase 2). alpha-beta-CROWN
//! takes the sparse path whenever at most 90% of the neurons are unstable
//! (`minimum_sparsity: 0.9` in `bound_general.py`, i.e. `min_stable = 0.1` in
//! ny's convention). On challenging_certified_training_2026 `cifar10_eps2_cnn7`
//! the FC target (Gemm 32768->512 folded with its BatchNormalization) is
//! ~86.7% stable, so the 0.9 threshold walks all 512 rows back through five
//! convolutions (6.2 s on the Mac DIAG log) where ~68 would do.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const CROWN_PARTIAL_SEED_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "crown-ibp-partial-dense-seed",
};

const CROWN_PARTIAL_DENSE_MIN_STABLE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CROWN_PARTIAL_SEED_SCOPE,
    role: "the stability fraction at or above which a Dense-seeded CROWN-IBP \
           partial walk whose successor layer is a ReLU seeds only its \
           IBP-unstable rows; every other walk keeps the historical 0.9",
    site: "crates/ny-propagate/src/network/ibp/crown_partial.rs \
           (dense_seed_min_stable_resolved, dense_seed_threshold)",
}];

declare_levers! {
    registry CROWN_PARTIAL_SEED_LEVERS;

    /// `NY_CROWN_PARTIAL_DENSE_MIN_STABLE` — minimum IBP-stable fraction for
    /// the Dense unstable-rows seed on ReLU-successor targets. Default 0.9,
    /// the historical literal, so an unset key is byte-identical.
    pub CROWN_PARTIAL_DENSE_MIN_STABLE = LeverDecl {
        name: "NY_CROWN_PARTIAL_DENSE_MIN_STABLE",
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: 1.0 },
        default: DefaultSpec::F64(0.9),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Minimum fraction of a Dense-carrier CROWN-IBP partial target's output neurons \
that must be IBP-stable (lower >= 0 or upper <= 0 under the CERTIFIED per-layer \
forward) before the walk seeds ONLY the IBP-unstable rows instead of the full \
identity. Applies only to a target whose successor layer is a ReLU; every other \
target (a final output layer, a Sigmoid/Tanh successor, the Patches conv \
branch) keeps the historical 0.9. At 0.1 this is alpha-beta-CROWN's own \
sparse-intermediate rule (`minimum_sparsity: 0.9`).

Soundness: the tracked set is the certified-IBP-unstable set, a SUPERSET of the \
truly unstable neurons, and every untracked neuron publishes the certified IBP \
box verbatim (`sparse_merge.rs` `scatter_sparse_crown_into_ibp`), the same \
publication the historical 0.9 path already makes on every target that clears \
it. Tracked (IBP-unstable) rows keep the full seed's CROWN value. Untracked \
rows change operand: the full seed publishes raw CROWN, which the collector \
intersects with its INITIAL forward IBP box, while the sparse seed publishes \
the certified IBP taken over the PUBLISHED, tightened prior. Neither operand \
contains the other, so an untracked neuron's magnitude can move in EITHER \
direction. A neuron the full seed would leave unstable can also become stable, \
which means the successor ReLU's relaxation is equal or tighter, never looser \
(up to GEMM rounding on the tracked rows). Later targets inherit those \
magnitudes through their IBP operands, so their bounds and relaxations can \
move either way. Every published value is a \
certified enclosure either way, so a wrong threshold costs tightness or time, \
never soundness. High because it moves published intermediate bounds, and so \
verdicts, in both directions.",
        provenance: Provenance::Unmeasured {
            why_ok: "default 0.9 is the historical literal, so absent it is \
                     byte-identical. Armed, it selects an EXISTING production \
                     path (the Dense unstable-rows seed + certified-IBP scatter) \
                     on more targets; that path publishes only certified \
                     enclosures. No armed-vs-unarmed A/B has run at official \
                     budgets",
        },
        owner: CROWN_PARTIAL_SEED_SCOPE,
        readers: CROWN_PARTIAL_DENSE_MIN_STABLE_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, read_with, LeverValue, Source};

    #[test]
    fn dense_min_stable_default_is_the_historical_literal() {
        // The reader keeps `0.9_f32` for every non-ReLU-successor target and
        // casts this f64 for the rest; both must be the same f32 or an unset
        // key would stop being byte-identical.
        assert_eq!(
            CROWN_PARTIAL_DENSE_MIN_STABLE.default,
            DefaultSpec::F64(0.9)
        );
        let resolved = read_with(&CROWN_PARTIAL_DENSE_MIN_STABLE, |_| None);
        assert_eq!(resolved.source, Source::Default);
        let value = resolved
            .value
            .as_f64()
            .expect("the declaration ships a default");
        assert_eq!((value as f32).to_bits(), 0.9_f32.to_bits());
    }

    #[test]
    fn dense_min_stable_parser_accepts_the_closed_unit_interval_only() {
        for (raw, expected) in [("0.1", 0.1), (" 0.5 ", 0.5), ("0", 0.0), ("1", 1.0)] {
            let resolved = read_with(&CROWN_PARTIAL_DENSE_MIN_STABLE, |_| Some(raw.to_owned()));
            assert_eq!(resolved.source, Source::LegacyEnv, "{raw:?}");
            assert_eq!(resolved.value.as_f64(), Some(expected), "{raw:?}");
        }
        for raw in ["-0.1", "1.5", "nan", "inf", "", "ten"] {
            let resolved = read_with(&CROWN_PARTIAL_DENSE_MIN_STABLE, |_| Some(raw.to_owned()));
            assert!(resolved.rejected_raw.is_some(), "{raw:?} must be rejected");
            assert_eq!(
                resolved.value.as_f64(),
                Some(0.9),
                "a rejected {raw:?} must resolve to the historical default"
            );
        }
    }

    #[test]
    fn dense_min_stable_layers_env_over_preset_over_default() {
        let preset = Some(LeverValue::F64(0.1));
        let from_preset =
            read_over_config_with(&CROWN_PARTIAL_DENSE_MIN_STABLE, |_| None, preset.clone())
                .expect("0.1 is admissible typed config");
        assert_eq!(from_preset.source, Source::Config);
        assert_eq!(from_preset.value.as_f64(), Some(0.1));

        let env_wins = read_over_config_with(
            &CROWN_PARTIAL_DENSE_MIN_STABLE,
            |_| Some("0.9".to_owned()),
            preset.clone(),
        )
        .expect("admissible env over admissible config");
        assert_eq!(env_wins.source, Source::LegacyEnv);
        assert_eq!(env_wins.value.as_f64(), Some(0.9));

        let kill = read_over_config_with(
            &CROWN_PARTIAL_DENSE_MIN_STABLE,
            |_| Some("sparse".to_owned()),
            preset,
        )
        .expect("a malformed env value is a kill switch, not an error");
        assert_eq!(kill.source, Source::LegacyEnvRejected);
        assert_eq!(kill.value.as_f64(), Some(0.9));

        assert!(
            read_over_config_with(
                &CROWN_PARTIAL_DENSE_MIN_STABLE,
                |_| None,
                Some(LeverValue::F64(1.5)),
            )
            .is_err(),
            "an out-of-range typed value must be refused, never clamped into a treatment"
        );
    }
}
