// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The sequential CROWN-IBP cost-weighted budget gate (#seq-weighted-budget).

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const SEQ_WEIGHTED_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "ibp-crown-sequential-budget",
};

const SEQ_WEIGHTED_READERS: &[ReaderSite] = &[ReaderSite {
    scope: SEQ_WEIGHTED_SCOPE,
    role: "the OPT-OUT gate choosing cost-weighted vs count-based per-candidate \
           grants for the sequential CROWN-IBP collection budget; the raw string \
           is read once and the decision is the site's `!= \"0\"` predicate",
    site: "crates/ny-propagate/src/network/ibp/crown_ibp.rs \
           (seq_weighted_budget)",
}];

declare_levers! {
    registry SEQ_WEIGHTED_BUDGET_LEVERS;

    /// `NY_SEQ_WEIGHTED_BUDGET` — OPT-OUT of cost-weighted sequential grants.
    pub SEQ_WEIGHTED_BUDGET = LeverDecl {
        name: "NY_SEQ_WEIGHTED_BUDGET",
        kind: LeverKind::Text,
        // The raw predicate is opt-out: absence executes the same policy as
        // token "1". Record that effective shipped state so receipts and the
        // search T0 filter do not mislabel an explicit =1 no-op as a treatment.
        default: DefaultSpec::Text("1"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
OPT-OUT, and the polarity is the point: the reader tests `!= Some(\"0\")`, so \
absence and every token other than exact `0` leave COST-WEIGHTED grants \
ENGAGED. Exact `0` restores the historical count-based shares byte-for-byte. \
Declared as Text with effective default `\"1\"` rather than rounded to a Boolean \
because the shipped predicate accepts every non-`0` token. The effective default \
also makes the receipt and automated search agree that explicit `1` is the \
shipped no-op while preserving the exact raw token for present values.

What it selects: how the sequential CROWN-IBP collection divides its global \
budget across per-layer candidates. Weighted shares use `objective rows x walk \
length` (`len x (k+1)`) — both factors the walk cost actually scales with — \
through the SAME weighted policy (cap + floor) the graph-alpha lane uses. The \
equal-share arm it replaces starved early heavy candidates: on \
`cifar_bias_field_46` at 60 s the layer-6 walk (8192 objective rows, 7-layer \
walk) was granted 6.39 s against a 6.53 s need and DEGRADED TO IBP, poisoning \
every later bound (output min_lo -37 -> -1.4e7), while the collection finished \
with more than 30 s unused because cheap wide layers banked their surplus too \
late.

MoatRisk::High because a candidate that loses its grant degrades to IBP, and \
an IBP-degraded intermediate bound is published into every downstream bound on \
the authoritative sequential route — the failure this gate exists to prevent \
is itself a bound-quality collapse, so the choice is verdict-affecting in both \
directions.",
        provenance: Provenance::Unmeasured {
            why_ok: "the ARMED arm ships (absence engages weighting) on the strength of the \
                     single diagnosed instance recorded at the reader site; no scored \
                     A/B over a retained row set exists yet, and the exact-\"0\" arm is \
                     the byte-for-byte historical fallback if one is needed",
        },
        owner: SEQ_WEIGHTED_SCOPE,
        readers: SEQ_WEIGHTED_READERS,
    };
}
