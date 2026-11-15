// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The cell-enumeration authority kill switch (#no-cell-enum).

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const CELL_ENUM_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "cell-enumeration-authority",
};

const CELL_ENUM_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: CELL_ENUM_SCOPE,
        role: "skip the cell-enumeration verdict path entirely, returning `None` so the \
               caller continues with the normal pipeline unchanged",
        site: "crates/ny-cli/src/commands/beta_crown/cell_enum.rs:try_cell_enumeration",
    },
    ReaderSite {
        scope: CELL_ENUM_SCOPE,
        role: "withhold the exact-f64 point-safety claim, returning `None` (no claim) \
               rather than `Some(true)`/`Some(false)`",
        site: "crates/ny-cli/src/commands/beta_crown/cell_enum.rs:f64_point_definitely_safe",
    },
    ReaderSite {
        scope: CELL_ENUM_SCOPE,
        role: "the same kill switch on the cctsdb sub-box driver, so one variable turns \
               the whole cell-enumeration authority off in one place",
        site: "crates/ny-cli/src/commands/cctsdb_subbox.rs:run",
    },
];

declare_levers! {
    registry NO_CELL_ENUM_LEVERS;

    /// `NY_NO_CELL_ENUM` — exact-`"1"` kill switch for the cell-enumeration authority.
    pub NO_CELL_ENUM = LeverDecl {
        name: "NY_NO_CELL_ENUM",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Exact-`\"1\"` kill switch that withdraws the cell-enumeration authority. Absent \
(the shipped state) leaves the authority ENGAGED; `\"1\"` turns it off at all \
three sites at once; explicit `\"0\"` is an admissible re-arm; any other token is \
a recorded rejection that leaves the default in place.

MIGRATION NOTE. The three readers were byte-identical raw \
`env::var_os(\"NY_NO_CELL_ENUM\").is_some_and(|v| v == \"1\")` calls. \
`LeverKind::Bool` reproduces that predicate EXACTLY on every input — absent and \
malformed both fall to the `false` default, `\"1\"` arms, `\"0\"` disarms — so this \
declaration is behaviour-preserving rather than a parser widening. Non-UTF-8 \
values landed on `false` under `var_os` byte comparison and land on `false` here \
as a rejected token, which is the same answer by a different route.

MoatRisk::Low, and the DIRECTION is why. Arming this can only REMOVE authority: \
`try_cell_enumeration` returns `None` so the caller runs the ordinary pipeline \
unchanged, and `f64_point_definitely_safe` returns `None`, i.e. NO CLAIM, rather \
than a safety answer. It can therefore lose a verdict that cell enumeration \
would have produced, but it cannot manufacture one — there is no arm of this \
switch that publishes a bound or a verdict the unarmed path would not.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default (absence leaves the authority engaged), and the armed arm \
                     only withdraws a verdict path — no scored A/B exists because arming it \
                     can only lose proofs, never mint them",
        },
        owner: CELL_ENUM_SCOPE,
        readers: CELL_ENUM_READERS,
    };
}
