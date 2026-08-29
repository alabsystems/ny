// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Governance for the multi-objective graph-domain alpha/beta optimizer.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const MO_OPTIMIZER: Scope = Scope {
    package: "ny-propagate",
    subsystem: "multi-objective-alpha-beta-optimizer",
};

const NY_CLI_PRESET: Scope = Scope {
    package: "ny-cli",
    subsystem: "preset-delivery",
};

const BETA_SPSA_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "deliver bab.mo_beta_spsa to the propagation preset cell before any lane runs",
        site: "crates/ny-cli/src/preset_gates.rs:apply",
    },
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project bab.mo_beta_spsa into the early flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: MO_OPTIMIZER,
        role: "layer the exact environment rollback over typed preset and measured default",
        site: "crates/ny-propagate/src/preset_gates.rs:mo_beta_spsa_enabled",
    },
    ReaderSite {
        scope: MO_OPTIMIZER,
        role: "select the measured two-sided SPSA beta gradient or analytical rollback",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/objectives.rs:\
               optimize_graph_beta_analytical_multi_objective_with_cache",
    },
];

const ALPHA_STEP_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "deliver bab.mo_alpha_step to the propagation preset cell before any lane runs",
        site: "crates/ny-cli/src/preset_gates.rs:apply",
    },
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project bab.mo_alpha_step into the early flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: MO_OPTIMIZER,
        role: "resolve the exact environment-over-preset master at the immutable optimizer boundary",
        site: "crates/ny-propagate/src/preset_gates.rs:mo_alpha_step_enabled",
    },
    ReaderSite {
        scope: MO_OPTIMIZER,
        role: "freeze, admit and receipt fixed every-fourth alpha-only SPSA rounds on relu-split children",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/objectives.rs:\
               optimize_graph_beta_analytical_multi_objective_with_cache",
    },
];

declare_levers! {
    registry MO_OPTIMIZER_LEVERS;

    /// Measured two-sided SPSA gradient for the multi-objective beta optimizer.
    pub MO_BETA_SPSA = LeverDecl {
        name: "NY_MO_BETA_SPSA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "\
Selects the measured two-sided SPSA estimate for multi-objective graph beta \
optimization. The analytical rollback differentiates an A-matrix coefficient \
rather than the split neuron's pre-activation value at the concretization \
corner, so its projected update can leave beta pinned at zero. SPSA perturbs \
beta candidates deterministically and every candidate is re-evaluated through \
the ordinary sound bound pass; the gradient can change search quality and \
deadline-bounded verdicts, but cannot make an invalid bound valid.

The historical reader shipped armed for absence and for every token except \
exact `\"0\"`. The governed Bool parser preserves that execution contract: \
absence uses this armed declaration default, exact `\"0\"` is the rollback, \
and a rejected present token suppresses the typed preset but falls back to the \
same armed default. Exact `\"1\"` explicitly arms. When the environment is \
absent, `bab.mo_beta_spsa` supplies a typed per-category arm or kill switch.",
        provenance: Provenance::Measured {
            commit: "034f974e9",
            date: "2026-08-22",
            artifact: "docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md, section \
                       `beta optimizer: wrong-quantity gradient found; measured SPSA landed`",
            delta: "on the recorded FC-UNSAT probe row, the first joint-lane \
                    beta optimization gain moved the depth-1 child bound from \
                    -8.2152 to -7.4544; each SPSA candidate remained subject \
                    to the sound spec-guided bound pass",
        },
        owner: MO_OPTIMIZER,
        readers: BETA_SPSA_READERS,
    };

    /// Scheduled SPSA stepping of stored multi-objective graph alpha.
    pub MO_ALPHA_STEP = LeverDecl {
        name: "NY_MO_ALPHA_STEP",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Clones the current graph-domain alpha state into one relu-split \
multi-objective beta evaluation and replaces nominal optimizer rounds 4, 8, \
12, ... with alpha-only two-sided SPSA rounds. Beta perturbation and Adam are \
skipped on those rounds, so a 15-iteration evaluation performs twelve beta \
rounds and three alpha rounds inside the same fixed iteration/wall budget. \
Every candidate bound is recomputed by the sound pass and every alpha remains \
in `[0, 1]`. The mechanism is nevertheless verdict-affecting because its bound \
quality changes which domains close before the deadline.

The first joint measurement was negative because hundreds of alpha directions \
dominated the shared beta slope. A later alternating alpha-only schedule also \
regressed because it displaced half of the fixed fifteen beta rounds. The \
fixed every-fourth schedule is the bounded follow-up: it preserves twelve beta \
rounds while giving current-box delta alpha three isolated updates. The \
declaration stays dark and no shipped preset sets `bab.mo_alpha_step`. Exact \
environment `\"1\"` arms and `\"0\"` disarms over that typed key; rejected presence \
suppresses the typed arm and fails dark. Admission additionally requires beta \
SPSA, the explicit relu-split route, split constraints, and a nonempty alpha \
state.",
        provenance: Provenance::Measured {
            commit: "885aa8ad2",
            date: "2026-08-24",
            artifact: "docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md, section \
                       `alpha stepping v1: machinery landed, joint perturbation measured OUT`",
            delta: "on the recorded FC-UNSAT probe row, joint alpha/beta SPSA \
                    regressed the depth-1 bound from about -7.45 to -8.22; \
                    restoring the dark arm restored the -7.4544 beta-only result",
        },
        owner: MO_OPTIMIZER,
        readers: ALPHA_STEP_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, LeverValue, Source};

    fn resolve(decl: &'static LeverDecl, raw: Option<&str>, typed: bool) -> crate::Resolved {
        let raw = raw.map(str::to_owned);
        read_over_config_with(decl, move |_| raw, Some(LeverValue::Bool(typed)))
            .expect("Bool typed value must satisfy a Bool declaration")
    }

    #[test]
    fn beta_spsa_preserves_default_on_and_exact_environment_precedence() {
        assert!(resolve(&MO_BETA_SPSA, None, true).value.as_bool());
        assert!(!resolve(&MO_BETA_SPSA, None, false).value.as_bool());
        assert!(!resolve(&MO_BETA_SPSA, Some("0"), true).value.as_bool());
        assert!(resolve(&MO_BETA_SPSA, Some("1"), false).value.as_bool());

        let rejected = resolve(&MO_BETA_SPSA, Some("true"), false);
        assert!(rejected.value.as_bool());
        assert_eq!(rejected.source, Source::LegacyEnvRejected);
        assert_eq!(rejected.rejected_raw.as_deref(), Some("true"));
    }

    #[test]
    fn alpha_step_is_dark_and_exact_environment_overrides_typed_config() {
        assert!(resolve(&MO_ALPHA_STEP, None, true).value.as_bool());
        assert!(!resolve(&MO_ALPHA_STEP, None, false).value.as_bool());
        assert!(!resolve(&MO_ALPHA_STEP, Some("0"), true).value.as_bool());
        assert!(resolve(&MO_ALPHA_STEP, Some("1"), false).value.as_bool());

        let rejected = resolve(&MO_ALPHA_STEP, Some("true"), true);
        assert!(!rejected.value.as_bool());
        assert_eq!(rejected.source, Source::LegacyEnvRejected);
        assert_eq!(rejected.rejected_raw.as_deref(), Some("true"));
    }
}
