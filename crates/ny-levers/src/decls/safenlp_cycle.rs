// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SafeNLP cycle policy: global false, category-selected true (#safenlp-cycle).

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

    /// `NY_SAFENLP_MIP_CYCLE` — global-off SafeNLP cycle policy.
    /// The SafeNLP2024 category selects true through its typed preset key.
    pub SAFENLP_MIP_CYCLE = LeverDecl {
        name: "NY_SAFENLP_MIP_CYCLE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the retained SafeNLP cycle policies through the typed \
`bab.safenlp_mip_cycle` key or the existing exact-one environment override. \
The declaration remains false globally; the owner-authorized SafeNLP2024 \
category selects true and separately selects `general.complete_verifier:mip`. \
The cycle flag alone is not a complete-verifier selector.

The retained pieces are (1) MIP-handoff budget arming where a BaB phase is \
used; (2) its conditional historical BaB floor; (3) marked-margin AY reframing \
with original-model witness/certificate replay; and (4) the serial one-model \
sequential MIP default. The August design history remains in \
docs/SAFENLP_CYCLE_2026-08-20.md. Direct MIP and independently checked native \
root proofs retain their distinct verdict ownership and original-model gates.

All budgets remain inside the original instance clock. The category policy \
is owner-authorized, not evidence of matched performance, broad nonregression \
or points. High risk because changing the selected prover and schedule can \
change which rows finish within their budget.",
        provenance: Provenance::Unmeasured {
            why_ok: "global false with a scoped owner-authorized SafeNLP2024 true \
                     preset; every verdict still requires its unchanged original \
                     witness/certificate or native-proof publication guard. Route \
                     markers record engagement, not matched benefit or authority. \
                     No performance or broad nonregression claim is made here",
        },
        owner: SAFENLP_CYCLE_SCOPE,
        readers: SAFENLP_CYCLE_READERS,
    };
}
