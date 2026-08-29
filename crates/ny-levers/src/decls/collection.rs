// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Collection and intermediate-tightening scheduling overrides.
//!
//! These declarations centralize legacy parsers and make the values visible in
//! the flight receipt. The root-joint execution family is also delivered
//! through typed preset cells; older collection controls in this module remain
//! migration-preparation readers until the per-run `LeverSet` reaches them.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const JOINT_ALPHA_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-joint-interm-alpha",
};

const JOINT_ALPHA_CLI_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "beta-crown-engine-selection",
};

const WIDE_DEMANDED_INTERM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-wide-demanded-interm-crown",
};

const COMPREHENSIVE_GPU_INTERM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-comprehensive-gpu-interm-crown",
};

const PHASE_RESIDENT_CROWN_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-phase-resident-crown",
};

const CPU_PARALLEL_INTERM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-cpu-parallel-interm-crown",
};

const WALK_ADMISSION_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "graph-crown-ibp-budget",
};

const COMPLETE_CLIP_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "complete-clip-host-reconstruct",
};

const ROOT_JOINT_MAX_DIM_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "override the legacy selector's 2,048-element contextual scope",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:scoped_joint_alpha_targets",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "override the demand-ranked sound-GPU selector's 32,768-element \
               contextual scope",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:scoped_joint_alpha_targets_demand_ranked",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "pin the effective input-split contextual ceiling in its execution receipt",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_input_split_root_node_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "pin the effective graph-route contextual ceiling in its execution receipt",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:root_joint_interm_alpha",
    },
];

const ROOT_JOINT_MASTER_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "admit the input-split root-map tightening transaction",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_root_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "admit the graph multi-objective root-map tightening transaction",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:root_joint_interm_alpha",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "require the master arm before exposing the finite-deadline GPU lane",
        site: "crates/ny-propagate/src/sound_gpu_gate.rs:\
               root_joint_deadline_lane_requested",
    },
    ReaderSite {
        scope: JOINT_ALPHA_CLI_SCOPE,
        role: "suppress the optional CPU engine handle when the root-joint treatment owns it",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:handle_beta_crown_command",
    },
];

const CPU_COMPUTE_ENGINE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: JOINT_ALPHA_CLI_SCOPE,
    role: "retain or suppress the optional CPU compute-engine presence handle without \
           discarding a qualified WGPU proof device",
    site: "crates/ny-cli/src/commands/beta_crown/mod.rs:handle_beta_crown_command",
}];

const ROOT_JOINT_SECS_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "size the input-split root-joint authority slice",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_root_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "cap the graph-lane root-joint authority slice",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:root_joint_interm_alpha",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "price the root-joint phase for scheduler admission",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:RootTightenPhase::predicted_cost",
    },
];

const ROOT_JOINT_ITERS_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "set the input-split joint-alpha ascent iteration ceiling",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_root_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "set the graph-lane joint-alpha ascent iteration ceiling",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:root_joint_interm_alpha",
    },
];

const ROOT_JOINT_LR_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "set the input-split joint-alpha Adam step",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_root_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "set the graph-lane joint-alpha Adam step",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:root_joint_interm_alpha",
    },
];

const ROOT_JOINT_DEADLINE_ASCENT_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "complete the two-gate finite-deadline GPU admission",
        site: "crates/ny-propagate/src/sound_gpu_gate.rs:\
               root_joint_deadline_lane_requested",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "permit alpha ascent only through the exact bounded-deadline capability",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:RootJointDeadlineAscentPolicy::from_governed",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "pin the input-split bounded-ascent request in its execution receipt",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_input_split_root_node_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "pin the graph-lane bounded-ascent request in its execution receipt",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:root_joint_interm_alpha",
    },
];

const ROOT_JOINT_OPTIONS_READER: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "construct the bounded root-joint target/fold options",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:root_joint_tighten_options",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "pin the configured input-split treatment in its execution receipt",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_input_split_root_node_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "pin the configured graph-lane treatment in its execution receipt",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               root_phases.rs:root_joint_interm_alpha",
    },
];

const ROOT_JOINT_LAYERS_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "override automatic target selection for the legacy selector",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:scoped_joint_alpha_targets",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "override automatic target selection for the demand-ranked selector",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:scoped_joint_alpha_targets_demand_ranked",
    },
];

const ROOT_JOINT_PROBE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "print the input-split frozen-bound width profile",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               root_bounds.rs:collect_input_split_root_node_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "print input-split root-bound fallback diagnostics",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/\
               bounds_eval.rs:evaluate_root_bounds",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "print root-joint preparation and optimizer diagnostics",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:root_joint_tighten_options",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "print graph CROWN demanded-target diagnostics",
        site: "crates/ny-propagate/src/network/graph_crown/spec_propagation/\
               core.rs:propagate_specifications",
    },
];

declare_levers! {
    registry COLLECTION_LEVERS;

    /// `NY_CPU_COMPUTE_ENGINE` — legacy-armed optional CPU engine handle.
    pub CPU_COMPUTE_ENGINE = LeverDecl {
        name: "NY_CPU_COMPUTE_ENGINE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `0` suppresses the optional CPU compute-engine presence handle when no \
qualified WGPU proof device is active. Exact `1`, absence, malformed input and \
non-UTF-8 input retain the handle, preserving the historical default. This \
selector may change which verifier work is available and therefore which sound \
verdict is reached before the deadline, but it can never discard an already \
qualified WGPU proof device. LEGACY-ARMED-UNQUALIFIED: the enabled-by-absence \
policy predates governed measurement; Bucket::Debug classifies its explicit \
kill switch rather than claiming evidence for the shipped arm.",
        provenance: Provenance::Unmeasured {
            why_ok: "legacy-armed optional engine-selection policy; exact-zero \
                     rollback is useful for controlled search, while current \
                     category-wide conversion and moat evidence remain outstanding",
        },
        owner: JOINT_ALPHA_CLI_SCOPE,
        readers: CPU_COMPUTE_ENGINE_READERS,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA` — master arm for the shrink-only root-map
    /// joint intermediate-alpha transaction on both graph entry routes.
    pub ROOT_JOINT_INTERM_ALPHA = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` admits the root joint intermediate-alpha transaction before the root \
bound map is frozen; exact `0`, absence and malformed input keep it dark. The \
same governed answer owns the input-split and graph multi-objective gates, the \
finite-deadline GPU admission, and the CLI's optional CPU-engine suppression, so \
a typed preset cannot half-enable the mechanism. Every publication is an \
intersection with an already-sound bound, but selection and wall use can move \
which certified bounds reach BaB and therefore the verdict. Width-only research \
measurements moved cGAN/CIFAR bounds; no current official-budget conversion and \
moat sweep qualifies a shipped arm, so the declaration and every shipped preset \
remain OFF.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark Debug treatment; the implemented transaction is \
                     shrink-only, while scored conversion and category-wide moat \
                     evidence remain outstanding",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_MASTER_READERS,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_SECS` — local wall request for the pass.
    pub ROOT_JOINT_INTERM_ALPHA_SECS = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(30),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Trimmed unsigned seconds requested by the root-joint pass, default 30. It sizes \
both live route slices and the root scheduler's predicted cost. The graph route \
caps it by half the real remaining instance budget; the scored input-split route \
caps it by the strictest route-local BaB boundary and scored wall, independently \
of restart-cache availability. Only a standalone direct collector call without \
either authority retains the historical grace. Malformed or non-UTF-8 \
environment input suppresses any preset and restores 30.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert behind the default-dark root-joint master and not a \
                     separately qualified scored-budget policy",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_SECS_READERS,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_ITERS` — joint-alpha ascent iteration cap.
    pub ROOT_JOINT_INTERM_ALPHA_ITERS = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(100),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Trimmed platform-sized iteration ceiling for both root-joint consumers, default \
100. A finite-deadline request may reduce the effective count to zero unless the \
separate bounded-ascent policy and backend capability are both present. It is \
inert while the master is dark; when live it changes the sound bound reached \
before the authority expires.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert behind the default-dark root-joint master",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_ITERS_READERS,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_LR` — joint-alpha Adam step.
    pub ROOT_JOINT_INTERM_ALPHA_LR = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_LR",
        // Both legacy readers trimmed and parsed f32. Governance keeps the
        // finite f32 numeric range; the runtime narrows the resolved f64 back
        // to f32 before entering the unchanged optimizer.
        kind: LeverKind::F64ClosedTrimmed {
            min: -3.402_823_466_385_288_6e38,
            max: 3.402_823_466_385_288_6e38,
        },
        default: DefaultSpec::F64(0.1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Trimmed finite f32-range Adam step for both root-joint consumers, default 0.1. \
The governed value is represented as f64 in receipts and narrowed to f32 at the \
unchanged optimizer boundary, matching ordinary legacy tokens exactly. It is \
inert behind the dark master; when live it changes the optimization trajectory \
and thus which shrink-only bounds are published.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert behind the default-dark root-joint master",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_LR_READERS,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT` — exact bounded-ascent arm.
    pub ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1`, together with the root-joint master, requests alpha ascent under a \
finite deadline. The request still executes only when the selected sound GPU \
backend exposes NY's audited bounded joint-adjoint and sound-fold capability; \
otherwise the pass retains the zero-iteration certified base fold. Absence, \
exact `0`, malformed input and every shipped preset keep this policy OFF.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark capability request; unsupported engines fail \
                     closed to the certified base fold",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_DEADLINE_ASCENT_READERS,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL` — crossing-row ceiling per target.
    pub ROOT_JOINT_INTERM_ALPHA_MAX_SEL = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(512),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Trimmed platform-sized ceiling on crossing rows seeded for each selected target, \
default 512. Zero makes the tightener an immediate sound no-op. Smaller values \
reduce backend work but can discard decisive rows; larger values increase seed \
memory and may exceed a finite-deadline backend's audited row capacity. It is \
inert behind the default-dark master and never expands the instance deadline.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert behind the default-dark root-joint master; the shipped \
                     512-row research ceiling remains unchanged",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_OPTIONS_READER,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP` — truncated stop-box extraction.
    pub ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` permits root-joint extraction to stop at a finite frozen interior \
node and concretize the certified suffix fold against that node's sound box. \
This also admits the resulting pure-chain suffix. Exact `0`, absence and \
malformed input preserve the historical requirement to reach the network input. \
The stop can make otherwise unreachable cGAN targets executable, so it changes \
which shrink-only bounds reach BaB and remains default-dark.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark research extraction extension; publication is \
                     still certified and shrink-only, but scored conversion and \
                     moat evidence remain outstanding",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_OPTIONS_READER,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN` — non-residual suffix admission.
    pub ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` permits an otherwise valid GPU suffix containing no residual block to \
be represented as a single certified Chain segment. Exact `0`, absence and \
malformed input retain the residual-required extraction contract. Frozen-stop \
independently implies pure-chain admission, so execution receipts credit this \
specific knob only when it was the necessary admitting authority.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark selection extension behind the root-joint \
                     master; unsupported extraction fails closed",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_OPTIONS_READER,
    };

    /// `NY_BN_GPU_EXTRACT` — exact BatchNorm-to-certified-Conv extraction arm.
    pub BN_GPU_EXTRACT = LeverDecl {
        name: "NY_BN_GPU_EXTRACT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` lets the root-joint extractor represent a surviving BatchNorm as an \
exact diagonal 1x1 convolution and discharge its certified precompute error \
into the feeding ReLU. This is currently owned and consumed only by the \
root-joint lane. Exact `0`, absence and malformed input refuse BatchNorm during \
that extraction, preserving the shipped default. The July cGAN recipe required \
this arm together with frozen-stop, but no scored conversion qualifies it ON.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark root-joint extraction authority; the certified \
                     transform fails closed but remains unqualified for shipping",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_OPTIONS_READER,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_LAYERS` — instance-specific target override.
    pub ROOT_JOINT_INTERM_ALPHA_LAYERS = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_LAYERS",
        kind: LeverKind::Text,
        default: DefaultSpec::Text(""),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Free-form comma-separated ReLU node names. Empty text, whitespace-only text or \
case-insensitive `all` retains automatic structural selection; any other value \
restricts selection to matching node names after trimming each list member. It \
is typed-preset deliverable for reproducible instance research but deliberately \
has no global search axis: graph-local node names do not define a portable \
VNN-COMP treatment and unmatched text would replay the baseline mislabeled.",
        provenance: Provenance::Unmeasured {
            why_ok: "instance-specific Debug override behind the default-dark \
                     master; no portable finite domain exists for global search",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_LAYERS_READERS,
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_PROBE` — print-only root-joint diagnostics.
    pub ROOT_JOINT_INTERM_ALPHA_PROBE = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Exact `1` enables root-joint width, preparation, fallback and optimizer trace \
lines at four diagnostic sites. No bound arithmetic, target selection or \
publication predicate reads the value. It remains environment-only and is \
intentionally excluded from the competition search space; logging overhead is \
not a solver treatment.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark print-only instrumentation",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_PROBE_READERS,
    };

    /// `NY_ROOT_JOINT_MIN_REMAINING_SECS` — graph-route admission floor.
    pub ROOT_JOINT_MIN_REMAINING_SECS = LeverDecl {
        name: "NY_ROOT_JOINT_MIN_REMAINING_SECS",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(240),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Minimum real remaining instance seconds required before the graph-lane \
root-joint pass starts, default 240; explicit zero disables the floor for \
research. Below the floor the armed route is a byte-identical no-op. This is a \
schedule/admission control rather than proof authority, but it decides whether \
the shrink-only transaction publishes any bound before BaB and can therefore \
move deadline verdicts.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert behind the default-dark master; the conservative \
                     shipped floor remains unchanged",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: &[ReaderSite {
            scope: JOINT_ALPHA_SCOPE,
            role: "decline the graph-lane pass when its measured tax cannot fit",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   root_phases.rs:root_joint_interm_alpha",
        }],
    };

    /// `NY_ROOT_PHASE_RESIDENT_CROWN` — exact-one opt-in to the deferred,
    /// unified dense-head plus comprehensive resident root transaction.
    pub ROOT_PHASE_RESIDENT_CROWN = LeverDecl {
        name: "NY_ROOT_PHASE_RESIDENT_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one phase-resident sound-GPU root intermediate transaction whose ownership \
is resolved at the comprehensive slot and whose execution is deferred to the \
established dense-head slot after intervening prerequisites. The frozen request \
contains every structurally selected dense-head pre-activation and every row of \
those targets, plus the complete bounded comprehensive target census. Clean \
predispatch capacity declines may lower only comprehensive row ceilings through \
32, 16, and 8; dense rows and both target censuses remain fixed. The retained \
backend's exact typed preflight is final under an eight-GiB caller cap. Role-bound \
target identity, exact receipt validation, staged rowwise shrink-only publication, \
and one allocation-free commit make accepted execution atomic. Admission or a \
clean predispatch decline may use the established dense-head fallback once; no \
accepted failure may retry or fall back. Exact `1` opts in. Missing, exact `0`, \
malformed UTF-8, and non-UTF-8 values retain the shipped routes.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark exact-one Debug treatment; it reuses the retained sound-GPU \
                     request/receipt authority and has no claimed scored conversion",
        },
        owner: PHASE_RESIDENT_CROWN_SCOPE,
        readers: &[ReaderSite {
            scope: PHASE_RESIDENT_CROWN_SCOPE,
            role: "resolve exclusive ownership before deferring one unified root transaction",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   root.rs:root_phase_resident_crown_policy",
        }],
    };

    /// `NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN` — exact-one opt-in to the
    /// atomic all-target sound-WGPU root sweep.
    pub ROOT_COMPREHENSIVE_GPU_INTERM_CROWN = LeverDecl {
        name: "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one comprehensive sound-GPU root intermediate sweep over every eligible \
demanded ReLU pre-activation in the bounded 2,048..=32,768 dimension class. \
Exact `1` opts in; absent, exact `0`, malformed UTF-8, and non-UTF-8 values \
retain the shipped OFF behavior. More than sixteen eligible targets, any \
structurally unpreparable target, or an unavailable automatic device resource \
profile refuses the whole route. Every predispatch retry retains every target \
and only lowers a common per-target row ceiling; both crossing and sign-stable \
non-point rows are retained when present. One frozen graph/bound transcript, \
one typed backend request, exact all-target result validation, and one staged \
shrink-only commit make every accepted result atomic. The backend profile is \
derived from live device class, granted buffer limits, and its existing memory \
budget rather than adapter names. This comprehensive gate owns the phase when \
armed, so refusal never falls through to the legacy one-target route. It has a \
twenty-second local authority slice and remains an unmeasured diagnostic: no \
speedup or scored-row conversion is claimed.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; complete typed requests and \
                     publication are fail-closed and atomic, but no retained \
                     CIFAR value/verdict/deadline A/B exists yet",
        },
        owner: COMPREHENSIVE_GPU_INTERM_SCOPE,
        readers: &[ReaderSite {
            scope: COMPREHENSIVE_GPU_INTERM_SCOPE,
            role: "default-dark admission gate and owner of the comprehensive GPU root phase",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   root.rs:root_comprehensive_gpu_interm_crown_policy",
        }],
    };

    /// `NY_ROOT_CPU_PARALLEL_INTERM_CROWN` — exact-one opt-in to the atomic
    /// comprehensive host-CPU intermediate sweep.
    pub ROOT_CPU_PARALLEL_INTERM_CROWN = LeverDecl {
        name: "NY_ROOT_CPU_PARALLEL_INTERM_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one comprehensive root intermediate-bound CROWN transaction over every \
eligible ReLU pre-activation (dimension at most 16,384; at most ten targets). \
Exact `1` opts in; absent, exact `0`, malformed UTF-8, and non-UTF-8 values \
retain the shipped OFF behavior. More than ten eligible targets refuses the \
whole transaction rather than selecting a prefix. Independent target backwards \
share one immutable frozen map and one absolute deadline on a private host-only \
Rayon pool capped at four workers. Checked aggregate memory admission precedes \
snapshot allocation. CUDA/WGPU process-global f64 admission is structurally \
bypassed inside the pool; existing deadline-aware faer CPU folds remain \
authoritative. Every result is collected in canonical graph order and all \
L2-preserving shrink-only intersections are staged before one all-or-none map \
commit. Any refusal, expiry, disjoint interval, stale live map, or malformed \
result is a byte-identical no-op. The measured serial all-target diagnostic \
tightened 9/10 nodes in 349.1 s, reduced aggregate width 24,696.35 to 7,258.45, \
and moved the root census 0/99 to 3/99; the new parallel treatment itself has \
not yet completed a retained scored A/B, so it remains dark. Its local two-minute \
deadline is intentionally shorter than that 349.1 s serial measurement: this is \
a bounded diagnostic bridge, not a claim that CPU outer parallelism completes the \
all-target scored treatment or supplies the required speedup.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; the complete transaction is \
                     fail-closed and shrink-only, but the CPU-parallel treatment \
                     has no retained scored all-target A/B yet",
        },
        owner: CPU_PARALLEL_INTERM_SCOPE,
        readers: &[ReaderSite {
            scope: CPU_PARALLEL_INTERM_SCOPE,
            role: "default-dark admission gate for the comprehensive atomic CPU sweep",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
                   comprehensive_cpu.rs:comprehensive_cpu_enabled",
        }],
    };

    /// `NY_ROOT_WIDE_DEMANDED_INTERM_CROWN` — exact-one opt-in to one bounded
    /// demanded wide intermediate sound fold.
    pub ROOT_WIDE_DEMANDED_INTERM_CROWN = LeverDecl {
        name: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one root intermediate-bound base CROWN fold for the highest-impact wide \
demanded ReLU pre-activation. Exact `1` opts in; absent, exact `0`, malformed \
UTF-8, and non-UTF-8 values retain the shipped OFF behavior. The reader fixes \
the first-slice limits at one target, dimensions 2,048 through 32,768, at most \
512 crossing rows, and eight seconds of dispatch/publication authority. Ranking \
and extraction poll that authority and a late preparation cannot dispatch or \
publish, but one synchronous extractor work unit is not preemptible. Ranking is selection-only: \
crossing unstable mass multiplied by the number of reachable unstable ReLU \
layers. Publication remains the existing certified zero-iteration sound-GPU \
fold followed by shrink-only intersection. The local backend must expose sound \
GpuCrownBackward and finite-deadline authority; every refusal is a no-op. This \
can move verdict-authoritative root boxes and spend scored wall time, so it ships \
dark until retained A/B evidence justifies promotion.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; the shipped OFF arm is a \
                     bound/verdict-identical no-op, while the armed arm reuses certified \
                     shrink-only publication but has no retained track A/B yet",
        },
        owner: WIDE_DEMANDED_INTERM_SCOPE,
        readers: &[ReaderSite {
            scope: WIDE_DEMANDED_INTERM_SCOPE,
            role: "default-dark admission gate for the bounded one-target wide fold",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   root.rs:root_wide_demanded_interm_crown_policy",
        }],
    };

    /// `NY_CLIP_HOST_MEAN_LA` — exact-one opt-in to host mean-lA reconstruction.
    pub CLIP_HOST_MEAN_LA = LeverDecl {
        name: "NY_CLIP_HOST_MEAN_LA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the host mean-lA reconstruction used to rank Complete Clipping work \
when no admissible sound-GPU CROWN backend exists. Exact `1` opts in; absent, \
exact `0`, malformed UTF-8, and non-UTF-8 values retain the shipped OFF \
behavior. A contemporaneous relusplitter narrative reported that restored \
supply did not improve the 220-row result and cost branch throughput, but its \
machine-readable A/B receipts were not committed. That observation is useful \
diagnosis, not admissible Measured provenance. The snapshot is selection-only \
and published tightening remains independently certified, but the extra host \
CROWN passes consume the same absolute deadline and can therefore move bounds \
or verdicts; keep the default OFF and opt in only to investigate the downstream \
consumer when the current run has no admitted sound-GPU backend.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug override; the shipped OFF arm avoids \
                     all host reconstruction work, and no retained current-path \
                     A/B qualifies promotion",
        },
        owner: COMPLETE_CLIP_SCOPE,
        readers: &[ReaderSite {
            scope: COMPLETE_CLIP_SCOPE,
            role: "default-dark admission gate for host mean-lA reconstruction",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
                   interm_refine.rs:complete_clip_host_mean_la_enabled",
        }],
    };

    /// `NY_CLIP_INTERM_CERTIFIED` — exact-one opt-in to the certified batched clip.
    pub CLIP_INTERM_CERTIFIED = LeverDecl {
        name: "NY_CLIP_INTERM_CERTIFIED",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the certified batched intermediate-bound clip (interm_refine umbrella: \
`clip_resnet` + runtime `clip_guard`). Exact `1` opts in; absent, exact `0`, \
and every malformed spelling stay OFF. The armed path was reviewed 2026-08-13 \
(docs/CLIP_APPLY_ENCLOSURE_PROOF_DESIGN_2026-08-12.md): applied rows are \
minted CertifiedLayerCapture tokens, scope-validated per source and target, \
split half-spaces use the sound necessary-condition side in both directions, \
the coordinate solve carries an independent outward checker, and the unminted \
legacy lane is unreachable (zero production callers). Tightened caches feed \
verdict-authoritative CROWN bounds, so the blast radius is High; ships dark \
until the armed 220-row moat sweep lands its receipts in-tree. The legacy \
`NY_CLIP_INTERM`/`NY_CLIP_INTERM_RESNET` spellings remain permanently inert.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; the shipped OFF arm is \
                     byte-identical to the historical quarantine, and the armed \
                     arm is gated on the reviewed certified bank path with the \
                     runtime guard retained as defense in depth",
        },
        owner: COMPLETE_CLIP_SCOPE,
        readers: &[ReaderSite {
            scope: COMPLETE_CLIP_SCOPE,
            role: "authority gate for the certified batched intermediate clip umbrella",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
                   interm_refine.rs:clip_interm_umbrella_enabled",
        }],
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM` — explicit selector-width override.
    pub ROOT_JOINT_INTERM_ALPHA_MAX_DIM = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
        kind: LeverKind::UsizeTrimmed,
        // Absence means "use the reader's policy default": 2,048 for the
        // legacy selector and 32,768 for the demand-ranked GPU selector. A
        // single numeric declaration default would falsify one of those
        // contexts, so the receipt records this as an absent override.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Overrides the maximum pre-activation dimension eligible for root joint \
intermediate-alpha selection. The legacy selector's contextual default is \
2,048; the demand-ranked sound-GPU selector's contextual default is 32,768. \
Surrounding whitespace accepted by Rust's `str::trim` is removed before \
parsing as `usize`, exactly \
matching both legacy readers. Malformed, negative, non-UTF-8, or \
platform-overflowing input is rejected and leaves the relevant contextual \
default in force. This changes only which already-sound tightening targets \
are attempted, but selection can move published bounds and deadline verdicts, \
so the blast radius is High. `DefaultSpec::Unset` deliberately describes an \
override rather than pretending the two consumers share one fallback.",
        provenance: Provenance::Unmeasured {
            why_ok: "explicit Debug override; selection is sound but no current \
                     authoritative A/B qualifies a shipped non-contextual value",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_MAX_DIM_READERS,
    };

    /// `NY_NO_WALK_RECORD_ADMISSION` — exact-one opt-out of record consultation.
    pub NO_WALK_RECORD_ADMISSION = LeverDecl {
        name: "NY_NO_WALK_RECORD_ADMISSION",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` disables consultation of passive node-walk timing records by the \
Graph CROWN-IBP admission policy. Absent and exact `0` leave the default \
record-aware admission engaged; every other UTF-8 spelling is rejected and \
therefore also leaves that default engaged. Recording remains passive and \
always on. LEGACY-ARMED-UNQUALIFIED: the underlying record-aware policy ships \
engaged but has no discriminating current-path A/B; Phase 0 has no \
`DefaultStatus` field yet, so this exact marker keeps that debt explicit until \
the target schema can encode `LegacyArmedUnqualified`. Bucket::Debug classifies \
the explicit opt-out, not the evidence status of the underlying policy. The \
opt-out is useful for differential diagnosis, but changing a \
deadline admission decision can move published bounds or verdicts, so this \
remains a High-risk Debug lever.",
        provenance: Provenance::Unmeasured {
            why_ok: "explicit diagnostic opt-out around a tracked \
                     LEGACY-ARMED-UNQUALIFIED policy; both scheduling faces \
                     retain certified fallback behavior, but verdict/timing \
                     parity has not been established on the current sound path",
        },
        owner: WALK_ADMISSION_SCOPE,
        readers: &[ReaderSite {
            scope: WALK_ADMISSION_SCOPE,
            role: "process-latched opt-out of consulting completed/aborted walk records",
            site: "crates/ny-propagate/src/network/graph_alpha/bounds/\
                   budget_policy.rs:walk_record_admission_enabled",
        }],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    #[test]
    fn max_dim_parser_preserves_trimmed_usize_contract() {
        for (raw, expected) in [("2048", 2_048_u64), (" 32768\t", 32_768_u64), ("+7", 7_u64)] {
            let resolved = read_with(&ROOT_JOINT_INTERM_ALPHA_MAX_DIM, |_| Some(raw.to_owned()));
            assert_eq!(resolved.value, LeverValue::U64(expected), "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "{raw:?}");
        }

        for raw in ["", "-1", "1.5", "all"] {
            let resolved = read_with(&ROOT_JOINT_INTERM_ALPHA_MAX_DIM, |_| Some(raw.to_owned()));
            assert_eq!(resolved.value, LeverValue::Unset, "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{raw:?}");
        }
    }

    #[test]
    fn root_joint_family_parsers_preserve_the_governed_contracts() {
        for decl in [
            &ROOT_JOINT_INTERM_ALPHA,
            &ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT,
            &ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP,
            &ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN,
            &BN_GPU_EXTRACT,
            &ROOT_JOINT_INTERM_ALPHA_PROBE,
        ] {
            for (raw, enabled, source) in [
                (None, false, Source::Default),
                (Some("0"), false, Source::LegacyEnv),
                (Some("1"), true, Source::LegacyEnv),
                (Some("true"), false, Source::LegacyEnvRejected),
                (Some(" 1 "), false, Source::LegacyEnvRejected),
            ] {
                let resolved = read_with(decl, |_| raw.map(str::to_owned));
                assert_eq!(resolved.value.as_bool(), enabled, "{} {raw:?}", decl.name);
                assert_eq!(resolved.source, source, "{} {raw:?}", decl.name);
            }
        }

        for (decl, raw, expected) in [
            (&ROOT_JOINT_INTERM_ALPHA_SECS, " 60\t", 60_u64),
            (&ROOT_JOINT_INTERM_ALPHA_ITERS, " 12\t", 12_u64),
            (&ROOT_JOINT_INTERM_ALPHA_MAX_SEL, " 8\t", 8_u64),
            (&ROOT_JOINT_MIN_REMAINING_SECS, " 0\t", 0_u64),
        ] {
            let resolved = read_with(decl, |_| Some(raw.to_owned()));
            assert_eq!(resolved.value.as_u64(), Some(expected), "{}", decl.name);
            assert_eq!(resolved.source, Source::LegacyEnv, "{}", decl.name);
        }

        let lr = read_with(&ROOT_JOINT_INTERM_ALPHA_LR, |_| Some(" 0.25\t".to_owned()));
        assert_eq!(lr.value.as_f64(), Some(0.25));
        assert_eq!(lr.source, Source::LegacyEnv);

        for (decl, default) in [
            (&ROOT_JOINT_INTERM_ALPHA_SECS, LeverValue::U64(30)),
            (&ROOT_JOINT_INTERM_ALPHA_ITERS, LeverValue::U64(100)),
            (&ROOT_JOINT_INTERM_ALPHA_MAX_SEL, LeverValue::U64(512)),
            (&ROOT_JOINT_MIN_REMAINING_SECS, LeverValue::U64(240)),
        ] {
            let resolved = read_with(decl, |_| Some("malformed".to_owned()));
            assert_eq!(resolved.value, default, "{}", decl.name);
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{}", decl.name);
        }

        let lr = read_with(&ROOT_JOINT_INTERM_ALPHA_LR, |_| Some("nan".to_owned()));
        assert_eq!(lr.value.as_f64(), Some(0.1));
        assert_eq!(lr.source, Source::LegacyEnvRejected);

        let layers = read_with(&ROOT_JOINT_INTERM_ALPHA_LAYERS, |_| {
            Some("  Relu_7, Relu_11  ".to_owned())
        });
        assert_eq!(layers.value.as_str(), Some("  Relu_7, Relu_11  "));
        assert_eq!(layers.source, Source::LegacyEnv);
    }

    #[test]
    fn cpu_compute_engine_kill_switch_preserves_legacy_default_on_contract() {
        for (raw, enabled, source) in [
            (None, true, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), true, Source::LegacyEnvRejected),
            (Some(" 0"), true, Source::LegacyEnvRejected),
            (Some(""), true, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&CPU_COMPUTE_ENGINE, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn walk_record_opt_out_parser_is_exact() {
        for (raw, disabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&NO_WALK_RECORD_ADMISSION, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), disabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn host_mean_la_opt_in_preserves_default_off_exact_one_contract() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            // Malformed values fall back to the declared default (OFF), never
            // arm the lane by accident.
            (Some("true"), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&CLIP_HOST_MEAN_LA, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn wide_demanded_interm_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_WIDE_DEMANDED_INTERM_CROWN, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn comprehensive_gpu_interm_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(" 1"), false, Source::LegacyEnvRejected),
            (Some("1 "), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_COMPREHENSIVE_GPU_INTERM_CROWN, |_| {
                raw.map(str::to_owned)
            });
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn phase_resident_crown_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(" 1"), false, Source::LegacyEnvRejected),
            (Some("1 "), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_PHASE_RESIDENT_CROWN, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn cpu_parallel_interm_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(" 1"), false, Source::LegacyEnvRejected),
            (Some("1 "), false, Source::LegacyEnvRejected),
            (Some("01"), false, Source::LegacyEnvRejected),
            (Some("yes"), false, Source::LegacyEnvRejected),
            (Some(""), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_CPU_PARALLEL_INTERM_CROWN, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn cpu_parallel_interm_non_utf8_value_is_present_but_rejected_off() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let registry = crate::collect(&[&COLLECTION_LEVERS]).expect("collection registry");
        let raw = crate::RawLeverInputs::capture_with(&registry, |name| {
            (name == ROOT_CPU_PARALLEL_INTERM_CROWN.name)
                .then(|| OsString::from_vec(vec![b'1', 0xff]))
        });
        let set = crate::LeverSet::resolve(&registry, &raw);
        let resolved = set
            .resolved(&ROOT_CPU_PARALLEL_INTERM_CROWN)
            .expect("CPU diagnostic lever registered");
        assert!(!resolved.value.as_bool());
        assert_eq!(resolved.source, Source::LegacyEnvRejected);
        assert_eq!(resolved.env_utf8, Some(false));
    }
}
