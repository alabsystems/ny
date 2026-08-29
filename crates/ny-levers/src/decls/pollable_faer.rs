// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The pollable chunked-faer bounded batched-CROWN gate (#pollable-faer).

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const POLLABLE_FAER_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "linear-crown-batched",
};

const POLLABLE_FAER_READERS: &[ReaderSite] = &[ReaderSite {
    scope: POLLABLE_FAER_SCOPE,
    role: "the OnceLock-latched gate arming the pollable chunked-faer host \
           implementation for bounded (deadline-facade) batched Linear CROWN; \
           the raw string is read once through read_raw and the decision is \
           derived by the site's unit-tested truthy predicate",
    site: "crates/ny-propagate/src/layers/linear/crown_batched.rs \
           (pollable_faer_enabled)",
}];

declare_levers! {
    registry POLLABLE_FAER_LEVERS;

    /// `NY_POLLABLE_FAER` — legacy-truthy gate for the pollable chunked-faer
    /// bounded batched Linear CROWN host arm. Ships DARK.
    pub POLLABLE_FAER = LeverDecl {
        name: "NY_POLLABLE_FAER",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Arms the pollable chunked-faer host implementation for BOUNDED \
(deadline-facade) batched Linear CROWN. The arm changes NO bound math: it \
runs the SAME certified f64 `A*W`/`|A|*|W|` pair with the SAME \
`gamma_n*S + cast_gap` publication as the engine-absent host path, chunked \
on the output-row axis only with the facade's deadline polled between \
blocks — bit-identical, pinned by \
`pollable_faer_batched_bit_identical_to_cpu_and_scalar`. It is a pure \
throughput/coverage restoration for deadline-carrying CROWN: gate-off, \
every bounded caller is refused here and degrades (partial-CROWN fallback \
or the facade's scalar per-MAC GEMM — the 6f49a660 deadline-cliff class \
that cost dist_shift index5291 and cgan row 7 their banked unsats). Fail \
closed in both directions: a facade without the pollable host-GEMM \
capability keeps the historical refusal even when armed, and no partial \
result is ever published past a failed poll.

PARSER — declared Text, not Bool, to preserve it: the reader latches the \
raw string once in a OnceLock (hot path) and arms on any value that trims \
non-empty and is neither exact `0` nor case-insensitive `false`. \
`NY_POLLABLE_FAER=true` and `=2` ARM it — tokens an exact-\"1\" Bool \
declaration would silently reject, flipping a screening arm to baseline. \
Absence, the empty string, `0`, and `false`/`FALSE` are dark; a non-UTF-8 \
value stays dark (the reader is `env::var`-shaped via `read_raw`).",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark; gate-off is byte-identical to the historical \
                     refusal, and the armed arm's bound math is bit-identical by \
                     the pinned parity test \
                     (pollable_faer_batched_bit_identical_to_cpu_and_scalar), so \
                     only scheduling/coverage can move. Screening runs only — no \
                     banked A/B has run",
        },
        owner: POLLABLE_FAER_SCOPE,
        readers: POLLABLE_FAER_READERS,
    };
}
