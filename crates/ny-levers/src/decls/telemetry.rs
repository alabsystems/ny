// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Declarations for the true-gradient selector plus telemetry/diagnostic
//! latches (Phase 2, batch B1 prep).
//!
//! These declarations and the dynamic-name environment chokepoint reduce the
//! direct-literal ratchet, but they do not complete the `OnceLock` -> per-run
//! [`crate::LeverSet`] migration. Hot readers still cache the raw process
//! environment in `OnceLock`; cold readers still sample the live process
//! environment. No runtime reader consumes a frozen `LeverSet` yet.
//!
//! The five telemetry values are diagnostic, but arming them is not
//! value-neutral: clocks, reductions, formatting, allocations, atomics, and
//! logging can perturb a deadline-sensitive run. Those five declarations are
//! therefore `MoatRisk::Low`, not `None`, and remain
//! `Provenance::Unmeasured` until a value/verdict parity experiment says
//! otherwise. `TRUE_GRAD_GPU_REPLAY` is a separate legacy-armed steering
//! selector: its default stays behavior-compatible, while High/Unmeasured
//! tracking prevents the no-ops parity test from being mistaken for live-GPU
//! value neutrality.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const PHASE_TELEMETRY_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "phase-telemetry",
};

const BETA_GPU_PROBE_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "beta-gpu-probe",
};

const SEG_PROBE_SCOPE: Scope = Scope {
    package: "ny-gpu",
    subsystem: "segment-probe",
};

const BETA_GPU_PROBE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: BETA_GPU_PROBE_SCOPE,
        role: "gate the existing beta/GPU diagnostic markers throughout graph \
               propagation and branching; these legacy sites still sample the \
               process environment directly and remain visible to the raw-read \
               migration ratchet",
        site: "crates/ny-propagate/src/beta_crown/engine and \
               crates/ny-propagate/src/network/graph_alpha",
    },
    ReaderSite {
        scope: Scope {
            package: "ny-cli",
            subsystem: "vnncomp-wide-lane-readout",
        },
        role: "gate both terminal `[wide-lane]` publication-count readouts \
               through the declared environment chokepoint",
        site: "crates/ny-cli/src/commands/vnncomp.rs:beta_gpu_probe_armed",
    },
];

/// `NY_SEG_PROBE` gates the three complementary diagnostics emitted by the
/// resident GPU segment-composition fold.
const SEG_PROBE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: SEG_PROBE_SCOPE,
        role: "emit the resident-fold concretization eligibility marker",
        site: "crates/ny-gpu/src/wgpu_device/ops/\
               crown_backward_sound_resident.rs:seg_probe_armed",
    },
    ReaderSite {
        scope: SEG_PROBE_SCOPE,
        role: "emit the resident-stream eligibility marker",
        site: "crates/ny-gpu/src/wgpu_device/ops/\
               crown_backward_sound_resident.rs:seg_probe_armed",
    },
    ReaderSite {
        scope: SEG_PROBE_SCOPE,
        role: "emit per-segment coefficient and certified-error magnitudes",
        site: "crates/ny-gpu/src/wgpu_device/ops/\
               crown_backward_sound_resident.rs:seg_probe_armed",
    },
];

/// `NY_PHASE_TELEMETRY` is the one deliberately-shared name in this batch:
/// five read sites across three packages, all printing complementary phase
/// markers under the SAME switch so one variable arms the whole timing lane.
const PHASE_TELEMETRY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: PHASE_TELEMETRY_SCOPE,
        role: "gate every `[phase]` marker and `[frontier]` frame in the root \
               pipeline; the current hot-path implementation caches the raw \
               process-environment string in a process-wide `OnceLock`",
        site: "crates/ny-propagate/src/phase_telemetry.rs:58",
    },
    ReaderSite {
        scope: Scope {
            package: "ny-cli",
            subsystem: "beta-crown-verify",
        },
        role: "emit the phase-budget ledger allocation lines \
               (`PhaseBudgetLedger::emit_telemetry`); currently samples the \
               live process environment through the chokepoint per emission",
        site: "crates/ny-cli/src/commands/beta_crown/verify/phase_budget.rs:82",
    },
    ReaderSite {
        scope: Scope {
            package: "ny-mip",
            subsystem: "ay-lib",
        },
        role: "SafeNLP shared-prefix session start/completion markers; \
               currently samples the live process environment through the \
               raw chokepoint view per session",
        site: "crates/ny-mip/src/ay_lib.rs:1521",
    },
    ReaderSite {
        scope: Scope {
            package: "ny-cli",
            subsystem: "main",
        },
        role: "post-command CUDA deadline-GEMM aggregate line (still a raw \
               `env::var` read; migrates with batch B3)",
        site: "crates/ny-cli/src/main.rs:178",
    },
    ReaderSite {
        scope: Scope {
            package: "ny-cli",
            subsystem: "beta-crown-dispatch",
        },
        role: "SafeNLP direct-first route markers (still a raw `env::var_os` \
               read; migrates with batch B3)",
        site: "crates/ny-cli/src/commands/beta_crown/dispatch.rs:460",
    },
];

declare_levers! {
    registry TELEMETRY_LEVERS;

    /// `NY_BETA_GPU_PROBE` — dark beta/GPU diagnostic output.
    pub BETA_GPU_PROBE = LeverDecl {
        name: "NY_BETA_GPU_PROBE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables diagnostic stderr markers for the beta/GPU propagation lanes and the \
terminal `[wide-lane]` publication count. Exact \"1\" arms it; absence and all \
other byte strings leave it dark. The markers do not feed a bound, but their \
clock reads, counters, formatting, and stderr writes can perturb a \
deadline-sensitive run. The newly added CLI completion readouts consume this \
declaration through the central chokepoint; the older propagation readers are \
declared here but remain explicit raw-read migration debt tracked by the exact \
ratchet.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and diagnostic only; armed-vs-unarmed \
                     deadline and verdict parity has not been measured",
        },
        owner: BETA_GPU_PROBE_SCOPE,
        readers: BETA_GPU_PROBE_READERS,
    };

    /// `NY_SEG_PROBE` — dark segment-composition diagnostics.
    pub SEG_PROBE = LeverDecl {
        name: "NY_SEG_PROBE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables the `[conc-gate]`, `[seg-resident]`, and per-segment coefficient/error \
diagnostic lines in the resident GPU fold. Exact \"1\" arms it; absence and \
every other value leave it dark. The output does not feed a bound, but \
formatting and stderr traffic can perturb a deadline-sensitive run. All \
production reads go through the declared live environment chokepoint so scoped \
diagnostic tests retain their historical ability to toggle the probe within \
one process. A future Phase-2 migration must thread a frozen LeverSet into the \
GPU package.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and diagnostic only; armed-vs-unarmed \
                     deadline and verdict parity has not been measured",
        },
        owner: SEG_PROBE_SCOPE,
        readers: SEG_PROBE_READERS,
    };

    /// `NY_TRUE_GRAD_GPU_REPLAY` — opt out of advisory GPU replay steering.
    pub TRUE_GRAD_GPU_REPLAY = LeverDecl {
        name: "NY_TRUE_GRAD_GPU_REPLAY",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the qualified sound-GPU implementation of the advisory true-gradient \
replay when GPU operands are available. Exact \"0\" opts out; absent or exact \
\"1\" selects the shipped GPU face. The replay's trajectory bounds are \
discarded: only alpha-steering gradients flow out, every alpha remains a valid \
lower-relaxation slope, and backend, shape, finite-value, tolerance, or deadline \
refusal falls back to the CPU replay. With no GPU operands a focused test is \
bit-identical, but the live GPU lane deliberately tolerates small `nu` and \
replay-bound differences. Those gradients steer later alpha iterates and can \
therefore move authoritative bounds or deadline verdicts. \
LEGACY-ARMED-UNQUALIFIED: the shipped default remains on for behavior \
compatibility while this compact schema lacks `DefaultStatus`; Bucket::Debug \
classifies the exact-0 diagnostic opt-out, not approval of the underlying \
default.",
        provenance: Provenance::Unmeasured {
            why_ok: "tracked LEGACY-ARMED-UNQUALIFIED steering path; the no-ops \
                     parity test does not qualify live GPU operands, and a \
                     current authoritative A/B is still required",
        },
        owner: Scope {
            package: "ny-propagate",
            subsystem: "true-gradient-gpu-replay",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-propagate",
                subsystem: "true-gradient-gpu-replay",
            },
            role: "legacy-armed selection gate for the qualified GPU replay; \
                   exact 0 opts out and every refusal takes the CPU replay",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/\
                   batched/wide_alpha_true.rs:true_grad_gpu_replay_enabled",
        }],
    };

    /// `NY_PHASE_TELEMETRY` — dark, print-only phase markers (#phase-telemetry).
    pub PHASE_TELEMETRY = LeverDecl {
        name: "NY_PHASE_TELEMETRY",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables one stderr line per phase boundary (`[phase] <name> t=<secs>s`), the \
per-depth `[frontier]` frames of the batched BaB lane, the CLI phase-budget \
ledger lines, and the SafeNLP MIP session markers — all under this ONE \
switch, because lever pricing needs phase boundaries from every layer of the \
same run (docs/BANKING_SWEEP_2026-07-18.md: single-row wall-time deltas are \
unpriceable across builds). Armed by the exact string \"1\" and nothing \
else; the declared `false` default emits nothing. The marker values do not \
feed a bound, but \
arming reads clocks, formats strings, and writes stderr in a deadline-sensitive \
run.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; existing tests cover gate semantics and \
                     formatting only, not armed-vs-unarmed value or verdict \
                     parity under scored deadlines",
        },
        owner: PHASE_TELEMETRY_SCOPE,
        readers: PHASE_TELEMETRY_READERS,
    };

    /// `NY_MIP_TRACE` — presence-gated MIP harvest/report tracing.
    pub MIP_TRACE = LeverDecl {
        name: "NY_MIP_TRACE",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables the `NY_MIP_TRACE ...` stderr report lines of the certified linear \
lower bound solver: relaxation/root/split harvest summaries, adaptive \
comb target-FSB reports, and proof/fallback budget lines. This is the one \
PRESENCE gate in the telemetry batch: the historical sites tested \
`env::var_os(..).is_some()`, so ANY present value arms it — including `0`, \
the empty string, and non-UTF-8 — which is why it is declared `Text` with an \
`Unset` default and read through `read_with` over a `var_os` lookup rather \
than as a `Bool`. Absent emits nothing. Every line reports a value already \
computed, but the extra formatting and stderr traffic can perturb timing.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; output is diagnostic, but armed-vs-unarmed \
                     value and deadline-verdict parity has not been measured",
        },
        owner: Scope {
            package: "ny-mip",
            subsystem: "certified-linear-lower",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-mip",
                subsystem: "certified-linear-lower",
            },
            role: "presence gate for the ~14 trace-print sites, collapsed into \
                   the single `mip_trace_armed` helper; currently samples the \
                   live process environment per call",
            site: "crates/ny-mip/src/certified_linear_lower.rs:1856",
        }],
    };

    /// `NY_ITER0_PARITY_TRACE` — per-node A-matrix telemetry (#iter0-alpha-parity).
    pub ITER0_PARITY_TRACE = LeverDecl {
        name: "NY_ITER0_PARITY_TRACE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables one stderr line per node visited by a backward walk, reporting the \
accumulated coefficient magnitudes of BOTH backward folds over the same run \
(`[iter0-parity] walk=.. pass=.. node=..`), for localizing where the \
iteration-0 bound explosion enters \
(docs/ROOT_ALPHA_STEP_EXPLODES_AND_STALLS_2026-07-29.md). Armed by the exact \
string \"1\" and nothing else; the declared `false` default emits nothing \
and skips the \
O(nnz) stat reductions entirely. The armed reductions and formatting can \
perturb timing even though their values are diagnostic.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; the gate is checked before the O(nnz) \
                     reductions, but armed-vs-unarmed value and deadline-verdict \
                     parity has not been measured",
        },
        owner: Scope {
            package: "ny-propagate",
            subsystem: "iter0-parity-trace",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-propagate",
                subsystem: "iter0-parity-trace",
            },
            role: "gate the per-node trace lines; the current hot-path \
                   implementation caches the raw process-environment string \
                   in a process-wide `OnceLock`",
            site: "crates/ny-propagate/src/iter0_parity_trace.rs:53",
        }],
    };

    /// `NY_PATCHES_CARRIER_TRACE` — Patches->Dense carrier transitions (#patches-drop).
    pub PATCHES_CARRIER_TRACE = LeverDecl {
        name: "NY_PATCHES_CARRIER_TRACE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables one stderr line per full Patches->Dense carrier materialization ATTEMPT \
(`[patches-drop] scope=.. node=.. site=<file:line> purpose=.. \
outcome=<ok|refused-deadline|refused-memory|refused-semantic> \
deadline=<none|live|expired> rows=.. unstable=.. coeff_err=..`) plus the \
alpha-CROWN target walk's carrier decision tuple (`[patches-carrier] .. \
repr_in=.. allow_patches=.. hard=.. handled=..`), for naming the site that \
densifies a conv carrier under finite authority \
(docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md). Armed by the exact string \
\"1\" and nothing else; the declared `false` default emits nothing and reaches \
neither the clock nor `Location::caller()`. `site=` comes from the one \
`#[track_caller]` materialization funnel and is exact; `scope=`/`node=` come \
from a thread-local set at each instrumented walk's node head and are EMPTY or \
stale for a conversion reached outside those walks. FILTER ON `node=`: three of \
the four walks publish a walk-kind literal as `scope=` (`dag-alpha`, \
`graph-crown`, `spec-crown`) and the fourth publishes the alpha-CROWN TARGET \
node, so `node=` is the only field naming the walked node. Only `outcome=ok` names a \
site that dropped the carrier: every refusal consumer is transactional and \
leaves the carrier Patches. `deadline=` is classified BEFORE the work, so it is \
the authority the materialization started under, not a post-hoc clock read; \
`repr_in=` is the step's INCOMING carrier representation, not its result. The \
armed clock reads and stderr traffic can perturb a deadline-sensitive run even \
though no line feeds a bound.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; the gate is checked before any clock read, \
                     caller lookup, or formatting, but armed-vs-unarmed value and \
                     deadline-verdict parity has not been measured",
        },
        owner: Scope {
            package: "ny-propagate",
            subsystem: "patches-carrier-trace",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-propagate",
                subsystem: "patches-carrier-trace",
            },
            role: "gate the `[patches-drop]` materialization lines and the \
                   `[patches-carrier]` decision lines; the current hot-path \
                   implementation caches the raw process-environment string in a \
                   process-wide `OnceLock`",
            site: "crates/ny-propagate/src/patches_carrier_trace.rs:enabled",
        }],
    };

    /// `NY_MARGIN_ROW_PROFILE` — margin-row BaB phase profiler (#twinwall).
    pub MARGIN_ROW_PROFILE = LeverDecl {
        name: "NY_MARGIN_ROW_PROFILE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables coarse wall-clock accounting of the margin-row tree-loop hot phases \
(RAII timers + event counters aggregated into fixed atomic arrays, dumped as \
a human-readable breakdown). Armed by the exact string \"1\" and nothing \
else; with the declared `false` default, `Timer::start` returns `None` and allocates \
nothing. The armed path reads clocks and updates atomics, so it can perturb \
timing even though it never directly changes a coefficient or bound.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; the profiler records durations and \
                     counts only, but armed-vs-unarmed value and deadline-verdict \
                     parity has not been measured",
        },
        owner: Scope {
            package: "ny-propagate",
            subsystem: "margin-row",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-propagate",
                subsystem: "margin-row",
            },
            role: "gate `Timer::start`/`bump` in the tree-loop driver; the \
                   current hot-path implementation caches the raw process \
                   environment string in a process-wide `OnceLock`",
            site: "crates/ny-propagate/src/margin_row/prof.rs:82",
        }],
    };

    /// `NY_GPU_MEM_TRACE` — per-label GPU allocation attribution (#gpu-pool-highwater).
    pub GPU_MEM_TRACE = LeverDecl {
        name: "NY_GPU_MEM_TRACE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables per-label high-water attribution in the GPU memory ledger (a \
`String` key per allocation label plus growth-curve log lines every 256 MiB \
of new peak) and the per-label section of `summary()`. Armed by the exact \
string \"1\" and nothing else; the declared `false` default keeps only the \
two process-wide \
atomic byte counters — no allocation, no logging. The ledger records sizes; \
nothing here directly supplies a bound. The armed bookkeeping allocates and \
logs on the GPU allocation path, so it can perturb timing and memory use.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; arming adds label bookkeeping and log \
                     lines, and armed-vs-unarmed value and deadline-verdict \
                     parity has not been measured",
        },
        owner: Scope {
            package: "ny-gpu",
            subsystem: "gpu-memory-ledger",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-gpu",
                subsystem: "gpu-memory-ledger",
            },
            role: "gate label attribution in `record_alloc` and the per-label \
                   section of `summary`; the current hot-path implementation \
                   caches the raw process-environment string in a process-wide \
                   `OnceLock`",
            site: "crates/ny-gpu/src/gpu_memory_ledger.rs:83",
        }],
    };
}
