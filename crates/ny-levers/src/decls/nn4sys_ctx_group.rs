// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The nn4sys context-grouped disjunctive refinement gate (#ctx-group).

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const NN4SYS_CTX_GROUP_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "nn4sys-disjunctive-refine",
};

const NN4SYS_CTX_GROUP_READERS: &[ReaderSite] = &[ReaderSite {
    scope: NN4SYS_CTX_GROUP_SCOPE,
    role: "the single gate deciding whether disjunctive clause refinement \
           builds context-grouped shared trees over clause-box hulls or the \
           legacy per-clause trees",
    site: "crates/ny-cli/src/commands/beta_crown/verify/disjunctive_box_refine.rs \
           (ctx_group_enabled)",
}];

declare_levers! {
    registry NN4SYS_CTX_GROUP_LEVERS;

    /// `NY_NN4SYS_CTX_GROUP` — exact-`"1"` gate for context-grouped
    /// disjunctive refinement covers on nn4sys. Ships DARK.
    pub NN4SYS_CTX_GROUP = LeverDecl {
        name: "NY_NN4SYS_CTX_GROUP",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms context-grouped refinement covers in the production nn4sys disjunctive \
screen — the M3 exact-lane structural insight \
(docs/NN4SYS_GROUPED_EXACT_CONTEXT_M3_2026-07-26.md: 56,186 one-axis nn4sys \
clauses collapse to 122 `(free_axis, fixed_inputs)` contexts) ported to the \
scored path. Clauses whose tightened boxes are bit-identical on every axis \
except a common set of wide free axes describe the same 1-D/few-D function \
family; ONE shared refinement tree over the hull of their boxes bounds that \
family once instead of one tree per bit-identical box.

Soundness posture (nn4sys is a blind category — a wrong `unsat` is \
uncontradictable, hence High): grouping changes WHICH boxes get bounded, \
never how. Every bound pass on a hull node is the unchanged certified \
machinery; a clause joins a shared tree only with an EXPLICIT cover proof \
(`box_covers`, fail-closed to the per-clause path on any doubt), and on a \
split its obligation moves to exactly the children not provably disjoint \
from its own box (`boxes_provably_disjoint`, fail-closed: doubt keeps the \
obligation). Exact `1` arms; with the gate unset the per-clause path is \
byte-identical.",
        provenance: Provenance::Unmeasured {
            why_ok: "fleet-added and default-dark: unset is byte-identical to the \
                     historical per-clause path, and the armed arm admits a clause \
                     to a shared tree only through the fail-closed box_covers / \
                     boxes_provably_disjoint proofs, so doubt always falls back to \
                     the legacy trees. No armed-vs-unarmed A/B has run",
        },
        owner: NN4SYS_CTX_GROUP_SCOPE,
        readers: NN4SYS_CTX_GROUP_READERS,
    };
}
