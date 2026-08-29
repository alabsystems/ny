// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The blocked-pollable finite-deadline certified Conv2d IBP kernel selector
//! (#pollable-conv-ibp).

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const CONV_IBP_BLOCKED_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "conv2d-ibp-forward",
};

const CONV_IBP_BLOCKED_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CONV_IBP_BLOCKED_SCOPE,
    role: "the OnceLock-latched selector choosing which kernel runs the \
           finite-deadline CERTIFIED f64 Conv2d IBP forward: the blocked \
           tap-hoisted twin (parallel or sequential) or the historical per-MAC \
           scalar oracle; the raw string is read once through read_raw and the \
           decision is derived by the site's unit-tested three-way predicate \
           `conv_ibp_blocked_route_from_token`",
    site: "crates/ny-propagate/src/layers/convolution/conv2d/ops_ibp_blocked.rs \
           (conv_ibp_blocked_route)",
}];

declare_levers! {
    registry CONV_IBP_BLOCKED_LEVERS;

    /// `NY_CONV_IBP_BLOCKED` — kernel selector for the finite-deadline
    /// certified f64 Conv2d IBP forward. SHIPS ARMED (engaged by absence);
    /// exact `0` (or `false`/`off`/`no`) is the byte-identical kill switch.
    pub CONV_IBP_BLOCKED = LeverDecl {
        name: "NY_CONV_IBP_BLOCKED",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::None,
        doc: "\
Selects the KERNEL, never the arithmetic, for the finite-deadline CERTIFIED \
f64 Conv2d IBP forward — the route every Conv2d node of a deadline-carrying \
graph IBP collection takes (`graph_alpha/bounds/ibp.rs` -> \
`Conv2dLayer::propagate_ibp_sound_with_engine_and_deadline`).

THREE ARMS. Absent or any unrecognised token: the blocked kernel, Rayon-\
parallel within each block (the shipped arm). `seq`/`sequential`: the same \
blocked kernel, single-threaded, so a measurement can separate `blocking` \
from `threads`. Exact `0`, `false`, `off`, `no` (any case, trimmed): the \
historical per-MAC pollable scalar contraction \
`conv2d_ibp_forward_grouped_certified_f64_with_deadline`, retained as the \
bit-identity oracle and the kill switch.

DECLARED Text, not Bool, for two reasons: the arming rule is three-valued, \
and the polarity is engaged-by-absence (the `#seq-weighted-budget` and \
`#pollable-faer` precedent), so the receipt must record which token was \
actually seen rather than a Boolean the reader did not compute.

WHY IT SHIPS ARMED. All three arms fold every output over the SAME live taps \
in the SAME order — `conv2d_tap_axis` is the single shared tap predicate and \
it is separable, so the blocked arm's per-axis tables enumerate exactly the \
scalar arm's taps ascending — and publish through the SAME \
`certified_f64_gamma_and_inflate` / `certified_err_term` certificate and the \
SAME directed f32 casts. Block and thread boundaries fall strictly BETWEEN \
whole output elements, so no reduction is ever split. The published box is \
therefore byte-identical on every arm; only wall clock and the deadline-poll \
granularity move (per 4096 MACs -> per MAC-budgeted output block). That is \
why MoatRisk is None and why the armed default needs no verdict A/B.

WHAT IT BUYS. The scalar arm paid three `IxDyn` dynamic-stride index \
computations, a full `conv2d_tap_source` checked resolution and a poll \
counter PER MAC, single-threaded, with a data-dependent branch in the \
innermost statement. Measured consequence of that shape on the same call: \
8.5 s/step on CIFAR100_resnet_large with a finite layer deadline against \
~93 ms/step without one (`graph_crown/point_vjp_batched_resnet.rs:9-12`) — \
a ~91x tax that had been routed around (by WITHHOLDING the layer deadline on \
the attack lane) rather than paid down.

NOT A GEMM RESTORATION, deliberately. Routing this arm back onto the \
`deadline: None` im2col + `gemm_f32` route would re-enter the \
`gamma_{K+2}^{f32}*S + 2u*|y|` magnitude-scaled widening that \
#cgan-conv-ibp-magnitude-floor removed (cGAN_imgSz32_nCh_3_prop_2: \
unsat@640s before 6f49a660, timeout@1800s after). An f32 engine cannot carry \
the f64 certificate, so no arm of this lever enters one.",
        provenance: Provenance::ValueNeutral {
            parity_test: "ny_propagate::layers::convolution::conv2d::tests::\
                          conv2d_blocked_certified_f64_bit_identical_to_scalar_route",
        },
        owner: CONV_IBP_BLOCKED_SCOPE,
        readers: CONV_IBP_BLOCKED_READERS,
    };
}
