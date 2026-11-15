// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-v3-bankable: the multi-objective cover-certification policy of the
//! graph β-CROWN multi-objective BaB (the lane vit_2023 takes).

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const MO_COVER_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "graph-multi-objective-cover",
};

const MO_COVER_CERTIFY_READERS: &[ReaderSite] = &[ReaderSite {
    scope: MO_COVER_SCOPE,
    role: "resolve the per-call cover-certification policy over the typed \
           `bab.mo_cover_certify` config value before the driver arms its \
           per-row frame",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/mo_cover.rs:resolve_mo_cover_policy",
}];

declare_levers! {
    registry MO_COVER_LEVERS;

    /// `NY_MO_COVER_CERTIFY` — the multi-objective cover certification
    /// policy (`off | ledger | replay`), delivered by the typed preset key
    /// `bab.mo_cover_certify`. Ships DARK (`off`).
    pub MO_COVER_CERTIFY = LeverDecl {
        name: "NY_MO_COVER_CERTIFY",
        kind: LeverKind::Enum(&["off", "ledger", "replay"]),
        default: DefaultSpec::Text("off"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#vit-v3-bankable. The multi-objective COVER CERTIFICATION system of the \
disjunctive graph multi-objective BaB (vit_2023's lane). `off` (default) is \
byte-identical to the historical lane: no frame, no replay, no line, and its \
`Verified` publishes through the counted `BetaCrownVerdict` legacy door.

* `ledger`: a per-ROW path completeness ledger (`certify::graph`) records \
every close with the split literals of the region it closes — a bound close \
per row it proves, an emptiness close only with a certified positive slack. \
A `Verified` whose rows do not cover the root is WITHDRAWN (`Unknown`, \
`[cover-gate] REFUSED scope=invocation lane=graph_mo`); a proved frame \
deposits one cover per row, so the instance door mints an `UnsatToken` \
(`[cover-gate] authorized scope=instance lane=graph`); an unbuildable frame \
keeps the legacy door.
* `replay`: `ledger` plus a CERTIFY-ONCE replay of every witness leaf below \
the root with no reused lane state (no lA warm start, no inherited node \
bounds, no parent flooring, no device batch; the lane's final beta only as a \
clamped hint), on the CPU. `Verified` stands only when the cover is proved \
AND every witness re-certifies (`[mo-cover-replay] ... outcome=confirmed`).

Downgrade-only in both armed modes. Observable: `[mo-cover] armed=1 mode=` \
on every armed call.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark Debug selector; both armed modes can only \
                     withdraw a Verified (never create or tighten one), and \
                     `off` is bit-identical by the pinned default test \
                     (mo_cover_off_is_bit_identical_and_ledger_proves_every_verified_run); \
                     no scored-row A/B or moat has run",
        },
        owner: MO_COVER_SCOPE,
        readers: MO_COVER_CERTIFY_READERS,
    };
}
