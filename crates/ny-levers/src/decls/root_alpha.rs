// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The root alpha (ascent) phase.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const ROOT_ALPHA: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-alpha",
};

const NY_CLI_PRESET: Scope = Scope {
    package: "ny-cli",
    subsystem: "preset",
};

const ALPHA_ZERO_YIELD_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "validate both typed alpha-crown preset locations before they can reach a run",
        site: "crates/ny-cli/src/preset/apply.rs:610 (validate_alpha_preset)",
    },
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "deliver the typed preset value into AlphaCrownConfig",
        site: "crates/ny-cli/src/preset/apply.rs:597 (apply_alpha_preset)",
    },
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the same solver-then-BaB typed value into the flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:2840-2849 (flight receipt projection)",
    },
    ReaderSite {
        scope: ROOT_ALPHA,
        role: "capture the legacy environment override without losing non-UTF-8 presence",
        site: "crates/ny-propagate/src/network/graph_alpha/propagate_dag/mod.rs:211 (alpha_zero_yield_env_raw)",
    },
    ReaderSite {
        scope: ROOT_ALPHA,
        role: "resolve present environment override above the typed preset default",
        site: "crates/ny-propagate/src/network/graph_alpha/propagate_dag/mod.rs:231 (alpha_zero_yield_frac)",
    },
    ReaderSite {
        scope: ROOT_ALPHA,
        role: "retire the root alpha ascent after the resolved fraction of its window",
        site: "crates/ny-propagate/src/network/graph_alpha/propagate_dag/mod.rs:1878-1900",
    },
];

const ALPHA_ENVELOPE_GRAD_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ROOT_ALPHA,
    role: "select the experimental concretization-argmin alpha-gradient rule",
    site: "crates/ny-propagate/src/network/graph_alpha/backward/gradients.rs:envelope_grad_enabled",
}];

declare_levers! {
    registry ROOT_ALPHA_LEVERS;

    /// `NY_ALPHA_ZERO_YIELD_FRAC` — stop paying for zero in the root ascent.
    pub ALPHA_ZERO_YIELD_FRAC = LeverDecl {
        name: "NY_ALPHA_ZERO_YIELD_FRAC",
        kind: LeverKind::F64Open { min: 0.0, max: 0.9 },
        // The registry-wide fallback and every shipped preset stay unset.
        // Treating the sampled 0.25 candidate as a global default would arm
        // unrelated categories and break the explicit-env kill switch.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Retires the root alpha ascent once this FRACTION of the ascent's own window \
has passed without improvement. The typed preset seam can supply a scoped \
default; no shipped preset currently does. A present valid environment value \
replaces the typed preset and any present invalid value kills it. A fraction \
of the window, never a fixed number of seconds or iterations — that is \
invariant I1 of \
docs/DESIGN_MARGINAL_VALUE_SCHEDULER_2026-08-08.md, and it is why this is the \
template for the whole budget-returning class.

WHY IT EXISTS: `early_stop_patience` already implements \"stop paying for \
zero\", but it counts ITERATIONS against a TIME window. On \
cifar100_resnet_medium an iteration costs ~4-5 s against a 40 s window, so a \
patience of 10 cannot fire before the window closes; the measured ascent runs \
7-10 iterations at `best_impr = 0.000e0` and is then cut off by the deadline \
having improved on its own initialiser exactly zero times. The alpha warmup \
therefore spends 40 % of the official 100 s budget before BaB starts.

SOUNDNESS: pure early exit on the SAME `should_save_best` \
route as ordinary convergence \
(`crates/ny-propagate/src/network/graph_alpha/propagate_dag/mod.rs:1878-1900`) \
— the loop returns the elementwise best bounds it has ALREADY certified. \
Stopping sooner can return a looser certified enclosure; it cannot manufacture \
an invalid bound. Note the \
measurement establishes recovered time and an unchanged internal aggregate, \
not bit identity of the published bound: `best_lower_sum` is nonbinding and \
root-verified moved 0/99 -> 1/99 in the table below. That observable \
bound/verdict-pipeline effect is why this is MoatRisk::High even though both \
arms are sound.

Bucket::Debug records that the typed seam remains available for controlled \
experiments. The attempted category-wide CIFAR100 promotion was retracted: \
16 medium and 16 large rows sampled both model populations, but the run did \
not retain a complete machine-readable artifact and did not cover all 200 \
rows the preset can reach. `DefaultSpec::Unset` is therefore both the \
registry-wide fallback and the shipped state. A custom preset is still \
projected as a `config` source in the flight receipt. Admissible range is the \
OPEN interval (0.0, 0.9), matching \
both preset validation and the live reader's \
`is_finite() && (0.0..0.9).contains(v) && v > 0.0` filter.",
        provenance: Provenance::Measured {
            commit: "a5bc1e73",
            date: "2026-08-11",
            artifact: "docs/LEVER_CENSUS_AND_ROOT_ALPHA_REMEASURE_2026-08-11.md \
                       section 8 (16-row cifar100_medium sample, official 100 s \
                       budget, baseline and treated arms back-to-back per row)",
            delta: "frac=0.25 fired on 15/15 timeout rows, returned 8.4-14.8 s \
                    of root time (mean ~10.1 s), and improved root-verified \
                    objectives by +15/+1/+1 on three rows: zero regressions, \
                    zero verdict changes, no row conversions; the sat row's \
                    counterexample was byte-identical. This is useful \
                    candidate evidence, not a retained row-complete artifact \
                    authorizing the shared 200-row preset.",
        },
        owner: ROOT_ALPHA,
        readers: ALPHA_ZERO_YIELD_READERS,
    };

    /// `NY_ALPHA_ENVELOPE_GRAD` — experimental alpha-gradient direction.
    pub ALPHA_ENVELOPE_GRAD = LeverDecl {
        name: "NY_ALPHA_ENVELOPE_GRAD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the experimental concretization-argmin alpha-gradient rule instead of \
the shipped local rule. The exact environment token `1` arms it; absence, `0`, \
malformed text, and non-Unicode input leave it dark. This remains an A/B-only \
debug treatment: it changes alpha update direction and can therefore change a \
published bound or verdict. Alpha remains clamped to the sound envelope domain, \
so the risk is treatment quality rather than admitting an invalid relaxation, \
but no retained current-path evidence qualifies the treatment for promotion.",
        provenance: Provenance::Unmeasured {
            why_ok: "default OFF preserves the shipped local-gradient route; explicit opt-in is limited to controlled experiments",
        },
        owner: ROOT_ALPHA,
        readers: ALPHA_ENVELOPE_GRAD_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    #[test]
    fn alpha_envelope_grad_is_exact_one_and_default_dark() {
        let absent = read_with(&ALPHA_ENVELOPE_GRAD, |_| None);
        assert_eq!(absent.value, LeverValue::Bool(false));
        assert_eq!(absent.source, Source::Default);

        let armed = read_with(&ALPHA_ENVELOPE_GRAD, |_| Some("1".to_owned()));
        assert_eq!(armed.value, LeverValue::Bool(true));
        assert_eq!(armed.source, Source::LegacyEnv);

        let disarmed = read_with(&ALPHA_ENVELOPE_GRAD, |_| Some("0".to_owned()));
        assert_eq!(disarmed.value, LeverValue::Bool(false));
        assert_eq!(disarmed.source, Source::LegacyEnv);

        for malformed in ["true", "TRUE", "yes", "01", " 1", "1 ", "", "2", "-1"] {
            let resolved = read_with(&ALPHA_ENVELOPE_GRAD, |_| Some(malformed.to_owned()));
            assert_eq!(resolved.value, LeverValue::Bool(false), "{malformed:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{malformed:?}");
            assert_eq!(
                resolved.rejected_raw.as_deref(),
                Some(malformed),
                "{malformed:?}"
            );
        }
    }
}
