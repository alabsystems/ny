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

/// The ny-cli-side MIP schedule governors (#metaroom-mip-governor): the leaf
/// oracle's consecutive-undecided cut-out and the escalation's certified-clause
/// skip. Both are SCHEDULE choices over lanes whose every outcome is already
/// advisory, which is why both carry `MoatRisk::Low`.
const MIP_GOVERNOR_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "mip-governor",
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

    /// `NY_MIP_LEAF_UNDECIDED_STREAK_CAP` — consecutive-`Undecided` cut-out for
    /// the Graph-MIP leaf oracle (#metaroom-mip-governor).
    pub MIP_LEAF_UNDECIDED_STREAK_CAP = LeverDecl {
        name: "NY_MIP_LEAF_UNDECIDED_STREAK_CAP",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
How many CONSECUTIVE `Undecided` leaf consults the Graph-MIP leaf oracle may \
spend before it declines every further consult for the rest of the run. Zero \
(the default, and the shipped behaviour) means NO CAP: only a confirmed SAT \
witness (`sat_latch`) or the cumulative wall fraction \
(`NY_GRAPH_MIP_LEAF_TOTAL_FRAC`, default 0.5 — HALF the BaB budget) can stop \
the oracle today.

MEASURED, metaroom_2023 4cnn_ry_57_9 spec_idx_28 at its official 210 s budget: \
six consults x 10.0 s = 60.39 s, every one `undecided`, i.e. 29% of the \
instance budget bought nothing while the branch-and-bound search that DOES \
make progress on that row was starved.

MoatRisk::Low. Declining a consult leaves the leaf in branch-and-bound, which \
is EXACTLY what `Undecided` already does — the oracle can only ever UPGRADE a \
leaf to certified-UNSAT or to a confirmed SAT witness, never downgrade one. So \
a cap can cost a proof the oracle would have found and can never manufacture \
one; it cannot change a verdict from `unsat` to `sat` or the reverse.",
        provenance: Provenance::Unmeasured {
            why_ok: "default 0 is the no-cap shipped behaviour, so an absent key and an \
                     absent declaration resolve identically; the armed arm can only \
                     shorten a lane whose every outcome is already advisory",
        },
        owner: MIP_GOVERNOR_SCOPE,
        readers: &[ReaderSite {
            scope: MIP_GOVERNOR_SCOPE,
            role: "stop consulting the exact leaf solver after N consecutive undecided \
                   replies, returning the remaining budget to branch-and-bound",
            site: "crates/ny-cli/src/commands/beta_crown/graph_mip_leaf.rs (admit_slice)",
        }],
    };

    /// `NY_MIP_ESCALATE_SKIP_CERTIFIED` — let the whole-net MIP escalation skip
    /// clauses a sound root bound has already certified infeasible.
    pub MIP_ESCALATE_SKIP_CERTIFIED = LeverDecl {
        name: "NY_MIP_ESCALATE_SKIP_CERTIFIED",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Seed the disjunctive MIP escalation's pending set from the UNCERTIFIED clauses \
only. The escalation's per-clause slice is `remaining / pending`, so on a \
19-clause row where a sound CROWN bound already refutes 18 clauses the single \
open clause receives 1/19th of the budget instead of all of it.

MEASURED, metaroom_2023 4cnn_ry_57_9 spec_idx_28: 19 clauses, 50 s escalation, \
2.279 s per clause, every clause abandoned at its slice with \
`certified_unsat=0`. Skipping the refuted clauses gives the one open clause \
~50 s, a 22x increase.

SOUNDNESS. A disjunctive `Unsat` requires EVERY clause to be infeasible. A \
clause whose unsafe row `a.y >= b` has a certified upper bound `sup(a.y) < b` \
over a SUPERSET of its own input box IS infeasible, and that is the same \
certificate ny publishes `unsat` on everywhere else. The prescreen is \
FAIL-CLOSED at every step: a malformed constraint, a non-finite bound, an \
input-box override that is not contained in the global box, a bound-collection \
error, or an expired slice all fall back to the historical all-clauses set, and \
the observable says which set was used. A skipped clause is recorded as \
certified-unsat, never as timed-out, so it cannot mask an undecided clause.

MoatRisk::Low for that reason: the arm changes only WHICH clauses the solver \
spends seconds on, and every clause it stops spending them on already carries \
an independent infeasibility certificate.",
        provenance: Provenance::Unmeasured {
            why_ok: "default false is the shipped all-clauses schedule, byte-identical to \
                     the undeclared tree; the armed arm only removes clauses that a \
                     sound bound has already refuted",
        },
        owner: MIP_GOVERNOR_SCOPE,
        readers: &[ReaderSite {
            scope: MIP_GOVERNOR_SCOPE,
            role: "seed the disjunctive escalation's pending clause set from the \
                   uncertified clauses only",
            site: "crates/ny-cli/src/commands/beta_crown/mip_highs.rs (solve_disjunctive)",
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
