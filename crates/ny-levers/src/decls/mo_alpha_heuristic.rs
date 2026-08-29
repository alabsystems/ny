// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Governance for refreshing stale multi-objective graph-domain alpha values.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const MO_GRAPH_ALPHA: Scope = Scope {
    package: "ny-propagate",
    subsystem: "multi-objective-graph-alpha",
};

const NY_CLI_PRESET: Scope = Scope {
    package: "ny-cli",
    subsystem: "preset-delivery",
};

const READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "deliver bab.mo_alpha_heuristic to the propagation preset cell before any lane runs",
        site: "crates/ny-cli/src/preset_gates.rs:apply",
    },
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the typed bab.mo_alpha_heuristic value into the early flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: MO_GRAPH_ALPHA,
        role: "layer an exact environment override over the typed preset and declaration default",
        site: "crates/ny-propagate/src/preset_gates.rs:mo_alpha_heuristic_enabled",
    },
    ReaderSite {
        scope: MO_GRAPH_ALPHA,
        role: "select stored ancestor alpha or the current-box heuristic for unstable graph neurons",
        site: "crates/ny-propagate/src/beta_crown/state/alpha/graph_runtime.rs:build_alpha_array_from_node_map",
    },
];

declare_levers! {
    registry MO_ALPHA_HEURISTIC_LEVERS;

    /// Refresh stale multi-objective graph-domain alpha from the current box.
    pub MO_ALPHA_HEURISTIC = LeverDecl {
        name: "NY_MO_ALPHA_HEURISTIC",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the CURRENT-box ReLU lower-slope heuristic instead of a stored graph \
alpha for unstable neurons in multi-objective domain propagation. Graph-domain \
alpha is cloned down the tree but is not re-stepped in these lanes; after split \
clamps reshape a triangle, the ancestor value can be materially worse than the \
sound 0/1 heuristic for the child's box. Any alpha in [0,1] defines a sound ReLU \
lower relaxation, so this changes bound quality and search order rather than the \
proof rule. It is nevertheless High-risk because those bounds decide which \
certified domains are closed before the competition deadline.

Exact environment `1` arms and exact `0` disarms, overriding \
`bab.mo_alpha_heuristic` in both directions. A present malformed or non-UTF-8 \
value is recorded as a rejection and suppresses the typed arm to this dark \
default. Absence defers to the typed preset. No production preset is armed: the \
single measured trace proves a useful bound movement, not row-level conversion \
or category-wide non-regression. The delivery key exists so a sealed VNN-COMP \
search can measure those questions without an environment-only result becoming \
unshippable.",
        provenance: Provenance::Measured {
            commit: "9e695ba92",
            date: "2026-08-24",
            artifact: "docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md, section \
                       `Stale alpha confirmed as the bound lever (2026-08-24)`",
            delta: "on the recorded FC-UNSAT joint-lane trace, the current-box \
                    heuristic moved the depth-1 lower bound from -7.4544 to \
                    -7.1403 and reached -7.1162 at depth 16, the first deep \
                    child to beat depth 1; no row-level verdict conversion or \
                    category-wide non-regression was established",
        },
        owner: MO_GRAPH_ALPHA,
        readers: READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, LeverValue, Source};

    #[test]
    fn exact_environment_values_override_typed_config_and_malformed_fails_dark() {
        let resolve = |raw: Option<&str>, typed: bool| {
            let raw = raw.map(str::to_owned);
            read_over_config_with(
                &MO_ALPHA_HEURISTIC,
                move |_| raw,
                Some(LeverValue::Bool(typed)),
            )
            .expect("Bool typed value")
        };

        assert!(resolve(None, true).value.as_bool());
        assert!(!resolve(None, false).value.as_bool());
        assert!(!resolve(Some("0"), true).value.as_bool());
        assert!(resolve(Some("1"), false).value.as_bool());

        let rejected = resolve(Some("true"), true);
        assert!(!rejected.value.as_bool());
        assert_eq!(rejected.source, Source::LegacyEnvRejected);
        assert_eq!(rejected.rejected_raw.as_deref(), Some("true"));
    }
}
