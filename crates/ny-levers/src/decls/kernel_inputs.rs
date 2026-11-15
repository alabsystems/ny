// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Governed raw inputs for existing kernel policies and cost calibration.
//!
//! These Text declarations describe INPUTS, not one shared Boolean decision.
//! The ConvTranspose policy has distinct legacy and finite-deadline defaults;
//! the original pure predicates remain authoritative at their existing sites.
//! Compatibility reads are preparatory governance, not a claim that all kernel
//! consumers now use one per-run frozen LeverSet. No shipped default changes.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const CONV: Scope = Scope {
    package: "ny-propagate",
    subsystem: "convolution-kernel-policy",
};
const FL_RATE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "forward-linear-rate",
};
const CONFORMANCE: Scope = Scope {
    package: "ny-cli",
    subsystem: "convtranspose-cuda-conformance",
};
const COLLECTOR: Scope = Scope {
    package: "ny-propagate",
    subsystem: "crown-policy-observation",
};

declare_levers! {
    registry KERNEL_INPUT_LEVERS;

    /// Existing mixed-default ConvTranspose recomputation input.
    pub CONVTRANSPOSE_SOUND_F64_GPU = LeverDecl {
        name: "NY_CONVTRANSPOSE_SOUND_F64_GPU",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Raw OS input for existing ConvTranspose f64 recomputation. The legacy \
              route arms only on exact 1 and defaults off. The explicit finite-deadline \
              route accepts absence or exact 1 and defaults on; all other OS values \
              decline it. Preserve those separate pure predicates, source enclosures, \
              engine/deadline gates and publication barriers. The conformance tool \
              captures exact OS bytes once and retains its existing strict assertions. \
              This declaration changes no policy or backend capability.",
        provenance: Provenance::Unmeasured { why_ok: "governs inherited raw input only; defaults and existing proof/capability predicates are unchanged; no new performance or authority qualification" },
        owner: CONV,
        readers: &[
            ReaderSite { scope: CONV, role: "legacy exact-one recomputation selection", site: "crates/ny-propagate/src/layers/convolution/conv2d/ops_gemm.rs:conv2d_forward_backward_coeff_f64_with_deadline" },
            ReaderSite { scope: CONV, role: "legacy exact-one paired recomputation selection", site: "crates/ny-propagate/src/layers/convolution/conv2d/ops_gemm.rs:conv2d_forward_backward_coeff_f64_pair_with_deadline" },
            ReaderSite { scope: CONV, role: "finite-only absence-or-exact-one selection", site: "crates/ny-propagate/src/layers/convolution/conv2d/ops_gemm_bounded.rs:try_pair" },
            ReaderSite { scope: COLLECTOR, role: "preserve raw OS input in the collector policy observation", site: "crates/ny-propagate/src/network/graph_alpha/bounds/crown.rs" },
            ReaderSite { scope: CONFORMANCE, role: "freeze raw OS input before checking the requested conformance arm", site: "crates/ny-cli/examples/convtranspose_cuda_conformance.rs:main" },
        ],
    };

    /// Existing literal-zero rollback for the dead f32 coefficient pair.
    pub CONV_SKIP_DEAD_F32 = LeverDecl {
        name: "NY_CONV_SKIP_DEAD_F32",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Existing rollback input: exact Unicode 0 restores the old f32 work; \
              absence, non-Unicode and every other token retain the inherited skip. \
              Keep the original predicate and its certified recomputation/fallback \
              behavior. Raw collector observation and conformance input retain OS \
              bytes. Text/Unset describes the input without asserting a second parser \
              or changing the default-on production policy.",
        provenance: Provenance::Unmeasured { why_ok: "declaration and compatibility routing only; inherited default-on behavior and all original arithmetic are unchanged" },
        owner: CONV,
        readers: &[
            ReaderSite { scope: CONV, role: "preserve the exact not-zero legacy predicate", site: "crates/ny-propagate/src/layers/convolution/crown_helpers.rs:conv_skip_dead_f32_enabled" },
            ReaderSite { scope: COLLECTOR, role: "preserve raw OS input in the collector policy observation", site: "crates/ny-propagate/src/network/graph_alpha/bounds/crown.rs" },
            ReaderSite { scope: CONFORMANCE, role: "freeze raw OS input before the existing dead-f32 assertion", site: "crates/ny-cli/examples/convtranspose_cuda_conformance.rs:main" },
        ],
    };

    /// Existing untrimmed positive-u128 cost-model override.
    pub FORWARD_LINEAR_MACS_PER_SEC = LeverDecl {
        name: "NY_FORWARD_LINEAR_MACS_PER_SEC",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "Raw Unicode compatibility input for the existing positive-u128 \
              throughput override. Parsing does not trim; zero, overflow, malformed \
              and non-Unicode input fall through to the original cached calibration, \
              explicit probe or conservative fallback. Both initializing and \
              no-initialization readers retain that same existing parser. It changes \
              admission estimates, never an enclosure, and remains excluded from \
              automated search. No calibration/probe or deadline policy changes.",
        provenance: Provenance::Unmeasured { why_ok: "governs the existing manual rate override with byte-identical parsing and absence behavior; no numerical constant or admission policy changes" },
        owner: FL_RATE,
        readers: &[
            ReaderSite { scope: FL_RATE, role: "initializing env-over-probe calibration selection", site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:forward_linear_rate_calibration" },
            ReaderSite { scope: FL_RATE, role: "noninitializing env-over-cache/fallback observation", site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:forward_linear_cached_rate" },
        ],
    };
}
