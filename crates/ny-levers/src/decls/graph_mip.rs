// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-MIP leaf publication controls.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const GRAPH_MIP_LEAF_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "graph-mip-leaf",
};

declare_levers! {
    registry GRAPH_MIP_LEVERS;

    /// `NY_GRAPH_MIP_LEAF_SAT` — exact-zero rollback for leaf SAT publication.
    pub GRAPH_MIP_LEAF_SAT = LeverDecl {
        name: "NY_GRAPH_MIP_LEAF_SAT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Controls whether a Graph-MIP leaf witness that passes the leaf oracle's layout \
authority and concrete all-row check may become a run-level violation candidate. \
Exact `0` restores the earlier advisory-only behavior; absent, exact `1`, empty, \
malformed, and non-UTF-8 values retain the shipped ON behavior. Final publication \
is still fail-closed behind organizer-exact VNN-LIB, input-box, strictness, and \
forward-runtime validation. LEGACY-ARMED-UNQUALIFIED: this default was introduced \
without a retained discriminating current-path A/B. Phase 0 has no DefaultStatus \
field yet, so Bucket::Debug classifies the rollback surface while this exact marker \
keeps the armed evidence debt visible. The choice can move a verdict or consume a \
different amount of the absolute deadline, so its moat is High.",
        provenance: Provenance::Unmeasured {
            why_ok: "the armed path is independently exact-sealed before publication, \
                     but no retained current-path A/B qualifies the default as measured",
        },
        owner: GRAPH_MIP_LEAF_SCOPE,
        readers: &[ReaderSite {
            scope: GRAPH_MIP_LEAF_SCOPE,
            role: "gate promotion of an authorized, all-row leaf witness to a run candidate",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   queue.rs:leaf_sat_return_enabled",
        }],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    #[test]
    fn leaf_sat_rollback_preserves_exact_zero_kill_switch() {
        for (raw, enabled, source) in [
            (None, true, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some(""), true, Source::LegacyEnvRejected),
            (Some("false"), true, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&GRAPH_MIP_LEAF_SAT, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value, LeverValue::Bool(enabled), "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }
}
