// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CUDA startup and transport selection.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const CUDA_TRANSPORT_SCOPE: Scope = Scope {
    package: "ny-cuda",
    subsystem: "gemm-transport",
};

const CLI_STARTUP_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "cuda-startup-admission",
};

const CUDA_GEMM_TRANSPORT_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CLI_STARTUP_SCOPE,
    role: "startup admission hint: a set transport request is a reason to attempt CUDA accelerator installation",
    site: "crates/ny-cli/src/main.rs:run",
}];

const CUDA_DISCRETE_MODE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: CLI_STARTUP_SCOPE,
        role: "startup admission hint only: treat the request as a reason to attempt CUDA accelerator installation",
        site: "crates/ny-cli/src/main.rs:run",
    },
    ReaderSite {
        scope: CUDA_TRANSPORT_SCOPE,
        role: "AUTHORITATIVE parser: select ExplicitDeviceCopy, or reject an out-of-contract value by failing engine construction",
        site: "crates/ny-cuda/src/lib.rs:cuda_discrete_mode_requested",
    },
];

declare_levers! {
    registry CUDA_LEVERS;

    /// `NY_CUDA_GEMM_TRANSPORT` — PRESENCE hint that a transport was requested.
    pub CUDA_GEMM_TRANSPORT = LeverDecl {
        name: "NY_CUDA_GEMM_TRANSPORT",
        kind: LeverKind::Presence,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Startup admission hint: if a CUDA GEMM transport has been named at all, that is \
a reason to attempt installing the accelerator. `ny-cuda` remains AUTHORITATIVE \
over parsing the value, checking it against the detected topology, and making \
the final transport selection — this reader never interprets the string.

PRESENCE, NOT EXACT-\"1\", and the declaration says so rather than rounding it \
to `Bool`. The reader is `var_os(..).is_some()`, so `NY_CUDA_GEMM_TRANSPORT=0` \
ARMS the hint; declaring it `Bool` would publish `false` in the flight receipt \
for a run where the hint was in fact taken. Naming a transport and expecting it \
not to be attempted is not a coherent request, so presence is the right parser \
here — it just has to be declared as the parser it is.

MoatRisk::Low: it can only cause an installation ATTEMPT. Admission failure is \
a no-op, and every transport that does get installed is still qualified by \
selected-transport IEEE known-answer tests before it can carry a result.",
        provenance: Provenance::Unmeasured {
            why_ok: "unset on every scored run here (no CUDA device on this host); it \
                     gates an installation attempt whose own qualification is unchanged",
        },
        owner: CUDA_TRANSPORT_SCOPE,
        readers: CUDA_GEMM_TRANSPORT_READERS,
    };

    /// `NY_CUDA_DISCRETE_MODE` — explicit-copy transport for discrete GPUs.
    pub CUDA_DISCRETE_MODE = LeverDecl {
        name: "NY_CUDA_DISCRETE_MODE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects cached device allocations with ordered H2D/GEMM/D2H work for ordinary \
f32/f64 GEMMs and the tensor-core proposal lane. It is the legacy force-explicit \
override; when unset or zero, `ny-cuda` now selects direct host-page-table, \
unified-memory, or explicit-device-copy transport automatically from live CUDA \
capabilities. Discrete GPUs need the explicit route because CPU access to \
managed allocations can trigger HMM page migration.

TWO READERS WITH DELIBERATELY DIFFERENT STRICTNESS, which is why this \
declaration names both sites. `ny-cuda` is AUTHORITATIVE and accepts exactly \
\"0\" or \"1\", failing engine construction on anything else rather than \
silently running a transport the user did not request; it keeps its own parser \
and is not routed through this crate's Boolean chokepoint, because that \
chokepoint's contract is that an out-of-contract value reads as the default, \
and here that would be the silent selection the ny-cuda parser exists to \
prevent. The `ny-cli` reader is only a startup admission HINT — should we even \
attempt to install the accelerator — so exact-\"1\" is the whole of its \
contract, and it is routed through the chokepoint.

MoatRisk::High because a transport switch changes which numerical path carries \
a sound result. What keeps that honest is not this lever: engine construction \
runs IEEE f32/f64 known-answer tests through the ACTUALLY SELECTED transport \
before it can carry one.",
        provenance: Provenance::Unmeasured {
            why_ok: "the legacy override ships off; automatic topology selection is \
                     qualified by selected-transport IEEE known-answer tests and emits \
                     its exact policy, reason, and capabilities in the runtime receipt",
        },
        owner: CUDA_TRANSPORT_SCOPE,
        readers: CUDA_DISCRETE_MODE_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, Source};

    #[test]
    fn discrete_mode_startup_parser_preserves_exact_one_contract() {
        for (raw, requested, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(""), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&CUDA_DISCRETE_MODE, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), requested, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }
}
