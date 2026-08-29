// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The SafeNLP full-cycle measurement master arm (#safenlp-cycle).

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const SAFENLP_CYCLE_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "safenlp-mip-cycle",
};

const SAFENLP_CYCLE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: SAFENLP_CYCLE_SCOPE,
    role: "the single resolver every cycle consumer calls: the MIP-handoff budget \
           arming in dispatch, the conditional-handoff floor in the phase ledger, \
           the serial parallel_split default and the AY margin-reframe arming in \
           mip_highs",
    site: "crates/ny-cli/src/commands/beta_crown/dispatch.rs \
           (safenlp_mip_cycle_enabled)",
}];

declare_levers! {
    registry SAFENLP_CYCLE_LEVERS;

    /// `NY_SAFENLP_MIP_CYCLE` — exact-`"1"` master arm for the SafeNLP
    /// full-cycle A/B (docs/SAFENLP_CYCLE_2026-08-20.md). Ships DARK.
    pub SAFENLP_MIP_CYCLE = LeverDecl {
        name: "NY_SAFENLP_MIP_CYCLE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the SafeNLP FULL-CYCLE set (docs/SAFENLP_CYCLE_2026-08-20.md) as one \
composed measurement lever over four existing pieces, none of which fires on \
its own default: (1) the MIP-handoff budget treatment arms (the same schedule \
the shared-prefix experiment variable arms), so the sequential reduced- \
verification lane receives a real BaB deadline and the planned MIP handoff is \
actually made; (2) the handoff is CONDITIONAL — the phase ledger keeps BaB's \
historical half-total floor instead of the enforce_mip_handoff floor-drop that \
cut BaB to ~4.5 s on every row, and the MIP fires only in the remaining \
reserved wall where the historical schedule ran nothing; (3) the AY margin \
reframe arms (the unique unsafe row is marked, so the pinned ay-milp solves \
the margin objective whose threshold-stop banks UNSAT on a strict bound \
crossing with original-model proof replay — the only handoff shape whose dual \
bound moves on this class, docs/SAFENLP_AY_ARC_2026-08-20.md section 5.1); \
(4) the sequential MIP defaults to the SERIAL one-model solve.

Verdict authority is unchanged on every arm: the reframe maps back to the \
original feasibility model and passes the existing witness/Farkas replay \
gate, and the budget arms only move deadlines inside the scored wall. High \
risk because converting safenlp timeouts is its stated purpose and the \
handoff changes how every safenlp row's wall clock is spent.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark exact-one gate; the pre-registered A/B \
                     (docs/SAFENLP_CYCLE_2026-08-20.md: 3 failing rows x 20 s official, \
                     3 runs/arm, moat 6 banked unsats + 3 sats) has not run yet. \
                     Engagement observables are pre-declared: the [safenlp-cycle] \
                     armed line, safenlp_mip_cycle= in [budget] telemetry, and the \
                     marked-margin session markers",
        },
        owner: SAFENLP_CYCLE_SCOPE,
        readers: SAFENLP_CYCLE_READERS,
    };
}
