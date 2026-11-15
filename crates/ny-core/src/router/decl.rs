// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The capability-declaration vocabulary (ROUTER_PROGRAM §3.1). DATA ONLY.
//!
//! One [`CapabilityDecl`] per routable capability: a legacy lane, an
//! ENGINE_PROGRAM tier / phase / variant, or an acting router rule. The
//! declarations are compiled-in tables owned by `ny-cli::router::registry`;
//! nothing here decides anything. There is deliberately no verdict type, no
//! proof-profile type, no name and no path anywhere in this module (the
//! `router_ratchet` source checks enforce it).
//!
//! Admission is a REFERENCE to the lane's own predicate ([`AdmissionRef`]),
//! never a copy of it: the registry calls the same function the lane calls,
//! and the lane re-checks its own admission whatever a plan says (§3.2 C2/C3).
//! An admission that cannot be evaluated is [`Admission::Unknown`], which is
//! a refusal, never a guess.

use serde::{Deserialize, Serialize};

/// A capability identity. Equal to the `id` of the capability's entry in
/// `docs/engine_program_capabilities.yaml` whenever one exists (§3.2 C1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CapId(pub &'static str);

impl CapId {
    /// The identity as a string slice.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// Capability identities that progress-bus TAP SITES name (the lanes push
/// under these ids; the registry in `ny-cli::router::registry` declares them
/// with the same constants, so a tap and its declaration cannot drift).
pub mod ids {
    use super::CapId;

    /// The margin-row lane's tree (expansions).
    pub const MARGIN_ROW: CapId = CapId("margin-row");
    /// The margin-row root alpha ascent (per-iteration gain).
    pub const MARGIN_ROW_ALPHA: CapId = CapId("margin-row.alpha");
    /// The frozen-dual BaB tier (per BFS level and per class).
    pub const FROZEN_DUAL_BAB: CapId = CapId("frozen-dual-bab");
    /// The wave beta-BaB tree (per wave).
    pub const WAVE_BETA_BAB: CapId = CapId("wave-beta-bab");
    /// The graph ReLU-split BaB (its stop line).
    pub const RELU_SPLIT_GRAPH: CapId = CapId("relu_split.graph");
    /// #router-r4: the low-dimensional input-split prover (per wave: its
    /// certified volume fraction and open queue; a CR-F source at its own
    /// wave boundary).
    pub const LOWDIM: CapId = CapId("lowdim");
    /// #router-r4: the margin-row `backward_interm` phase whose interm-alpha
    /// candidate loop `rule.ia_yield` may end early (declared so the
    /// safe-point rule can be enforced on it).
    pub const MARGIN_ROW_BACKWARD_INTERM: CapId = CapId("margin-row.backward_interm");
    /// #router-r5: the wave beta-BaB tree searched on the device
    /// (`margin_row.tree_mode: wave_cuda`).
    pub const WAVE_BETA_BAB_CUDA: CapId = CapId("wave-beta-bab-cuda");
    /// #router-r5: per-child certified zonotope intermediates in the
    /// input-split deferred rebound.
    pub const CGAN_CHILD_ZONO: CapId = CapId("cgan-child-zono");
    /// #router-r5: the certified forward root screen of the disjunctive route.
    pub const FWD_ROOT_SCREEN: CapId = CapId("fwd-root-screen");
    /// #router-r5: the receptive-field window of the margin-row
    /// `backward_interm` passes (and the joint refresh that re-runs them).
    pub const BI_WINDOW_JOINT_REFRESH: CapId = CapId("bi-window-joint-refresh");
    /// #router-r5: the frontier-complete lA warm start of the graph lane.
    pub const LA_WARM_FRONTIER: CapId = CapId("la-warm-frontier");
    /// #router-r5: the multi-objective cover certification of the graph lane
    /// (the per-row cover ledger with its certify-once replay).
    pub const VIT_COVER_CERTIFY: CapId = CapId("vit-cover-certify");
    /// #kfsb-production: the single-shot scorer of the shared graph kFSB
    /// selector (an objective-seeded prescore and one constrained CROWN pass
    /// per candidate child, simulated in a shared-budget fan-out).
    pub const KFSB_SINGLE_SHOT: CapId = CapId("kfsb-single-shot");
}

/// A router or optimizer rule identity (`CR-C`, `rule.governor`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RuleId(pub &'static str);

impl RuleId {
    /// The identity as a string slice.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// A capability's own O-STOP rule identity (§5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StopRuleId(pub &'static str);

/// A first-rung cost model identity (setup + one progress window, §3.2 C6).
/// Resolved against the compiled priors table; v1 priors are uniform, so every
/// model resolves to "unknown" and the router gives the lane today's cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CostModelId(pub &'static str);

/// A memory model identity (exact bytes where known, §3.2 C7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MemModelId(pub &'static str);

/// What a lane is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneRole {
    /// Searches for a counterexample; its sats go through the sat gate.
    Attack,
    /// Proves; `terminal_ok` means it may be the terminal lane that takes the
    /// remainder of the window.
    Prover {
        /// Whether the lane may be terminal.
        terminal_ok: bool,
    },
    /// A cheap screen that decides only obvious rows.
    Screen,
}

/// The kind of a routable capability (§3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapKind {
    /// A lane.
    Lane {
        /// The lane's role.
        role: LaneRole,
    },
    /// A tier inside a parent lane (frozen-dual inside margin-row).
    Tier {
        /// The parent lane.
        parent: CapId,
    },
    /// A phase feeding a lane that cannot decide the row by itself.
    Phase {
        /// The lane the phase feeds.
        lane: CapId,
        /// The slot the phase fills.
        slot: &'static str,
    },
    /// An alternative implementation of a slot.
    Variant {
        /// The slot.
        slot: &'static str,
        /// The value selecting this variant.
        value: &'static str,
    },
    /// An acting router or optimizer rule.
    Rule {
        /// The ENGINE_PROGRAM milestone step that may flip it to acting.
        milestone: &'static str,
    },
}

/// How a capability reports progress, in its own work units (§4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressKind {
    /// A binary split frontier (margin-row, graph ReLU split).
    FrontierBinary,
    /// A k-ary / wave frontier.
    FrontierKary,
    /// A level-synchronous BFS (frozen-dual, per class).
    Bfs,
    /// A certified-volume fraction (input split, lowdim).
    Volume,
    /// A best attack margin.
    Margin,
    /// A screen's roots per second.
    Screen,
    /// A dual bound (MIP).
    Dual,
    /// A phase's per-iteration yield (alpha ascent).
    Yield,
    /// No progress signal.
    Opaque,
}

impl ProgressKind {
    /// Stable token for logs.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::FrontierBinary => "frontier_binary",
            Self::FrontierKary => "frontier_kary",
            Self::Bfs => "bfs",
            Self::Volume => "volume",
            Self::Margin => "margin",
            Self::Screen => "screen",
            Self::Dual => "dual",
            Self::Yield => "yield",
            Self::Opaque => "opaque",
        }
    }
}

/// The unit a capability's work is counted in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkUnit {
    /// One tree expansion.
    Expansion,
    /// One search node.
    Node,
    /// One batch of domains.
    Batch,
    /// One BFS level.
    Level,
    /// One attack restart.
    Restart,
    /// One optimizer iteration.
    Iteration,
    /// One wave of domains.
    Wave,
    /// One root evaluation.
    Root,
    /// One whole pass.
    Pass,
    /// One LP solve.
    Solve,
    /// One annealing step.
    Step,
}

/// The first rung: what a lane needs to produce its first progress window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FirstRung {
    /// Priced by a cost model.
    Model(CostModelId),
    /// Not priced: the router gives the lane today's cap.
    #[default]
    Unknown,
}

/// What device a capability may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceUse {
    /// CPU only.
    #[default]
    None,
    /// Untrusted search on a device.
    Search,
    /// A verdict-qualified device path.
    VerdictQualified,
}

/// Resources a capability needs (§3.2 C7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceDecl {
    /// Minimum worker threads.
    pub threads_min: u32,
    /// Maximum worker threads (`None` = every available thread).
    pub threads_max: Option<u32>,
    /// Device use.
    pub device: DeviceUse,
    /// Memory model.
    pub mem: MemModelId,
}

/// Where the router may suspend or shorten a lane (§3.2 C8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafePoints {
    /// Uncancellable: the lane runs to its cap and is never shortened.
    None,
    /// Between work units.
    BetweenUnits,
}

/// Which amendments a capability honours at a safe point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct AmendmentSet {
    /// Accepts a new deadline from its cell.
    pub deadline: bool,
    /// May hand its unspent contract on.
    pub hand_on_source: bool,
    /// May receive a handed-on contract.
    pub hand_on_sink: bool,
    /// May defer the unspent part of its cap to a late reserve.
    pub defer: bool,
}

impl AmendmentSet {
    /// Honours nothing.
    pub const NONE: Self = Self {
        deadline: false,
        hand_on_source: false,
        hand_on_sink: false,
        defer: false,
    };
}

/// Whether a pinned knob is a ceiling or frozen (§2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    /// A pin is a ceiling: rules may use less, never more.
    Ceiling,
    /// A pin is frozen.
    Frozen,
}

/// Clamp-soundness audit status of a knob (§5.4, premise (i)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClampAudit {
    /// Nobody has audited the clamp; no rule may move the knob.
    Unaudited,
    /// Audited; the string names the audit record.
    Audited(&'static str),
}

/// A registry axis a capability exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KnobRef {
    /// The typed key (`margin_row.alpha_iters`).
    pub key: &'static str,
    /// Ceiling or frozen when pinned.
    pub effort: Effort,
    /// Clamp-soundness audit status.
    pub clamp: ClampAudit,
}

/// A capability's ledger status (§3.2 C10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LedgerStatus {
    /// Not routable; the string says why (which S-stage is missing).
    Open(&'static str),
    /// Runs in shadow only.
    Shadow,
    /// Today's shipped route already runs it on the shapes in its envelope.
    SealedAsShipped,
    /// S7 + S8 done on its envelope.
    Sealed,
}

impl LedgerStatus {
    /// Whether the router may act on this capability inside its envelope.
    #[must_use]
    pub const fn is_sealed(self) -> bool {
        matches!(self, Self::SealedAsShipped | Self::Sealed)
    }
}

/// Operator kinds, as bits of an [`OpSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    /// Gemm / MatMul / linear.
    Linear,
    /// Convolution.
    Conv,
    /// Transposed convolution.
    ConvTranspose,
    /// ReLU.
    Relu,
    /// Sigmoid.
    Sigmoid,
    /// Tanh.
    Tanh,
    /// Softmax.
    Softmax,
    /// Batch normalization.
    BatchNorm,
    /// Residual add.
    Residual,
    /// Concat.
    Concat,
    /// Pooling.
    Pool,
    /// Sign (binarized networks).
    Sign,
    /// Resize / upsample.
    Resize,
    /// Bilinear products (attention).
    Bilinear,
    /// Layer / group normalization.
    Norm,
    /// Anything else.
    Other,
}

impl OpKind {
    /// Every kind, in bit order.
    pub const ALL: [Self; 16] = [
        Self::Linear,
        Self::Conv,
        Self::ConvTranspose,
        Self::Relu,
        Self::Sigmoid,
        Self::Tanh,
        Self::Softmax,
        Self::BatchNorm,
        Self::Residual,
        Self::Concat,
        Self::Pool,
        Self::Sign,
        Self::Resize,
        Self::Bilinear,
        Self::Norm,
        Self::Other,
    ];

    /// This kind's bit.
    #[must_use]
    pub const fn bit(self) -> u64 {
        1u64 << (self as u32)
    }

    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Conv => "conv",
            Self::ConvTranspose => "conv_transpose",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Softmax => "softmax",
            Self::BatchNorm => "bn",
            Self::Residual => "residual",
            Self::Concat => "concat",
            Self::Pool => "pool",
            Self::Sign => "sign",
            Self::Resize => "resize",
            Self::Bilinear => "bilinear",
            Self::Norm => "norm",
            Self::Other => "other",
        }
    }
}

/// A set of operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct OpSet(pub u64);

impl OpSet {
    /// No kinds.
    pub const EMPTY: Self = Self(0);

    /// This set plus `op`.
    #[must_use]
    pub const fn with(self, op: OpKind) -> Self {
        Self(self.0 | op.bit())
    }

    /// Whether `op` is in the set.
    #[must_use]
    pub const fn contains(self, op: OpKind) -> bool {
        self.0 & op.bit() != 0
    }

    /// Whether every kind of `self` is in `other`.
    #[must_use]
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }
}

/// Specification archetypes, as bits of a [`SpecSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecArchetype {
    /// Argmax-margin robustness.
    Robustness,
    /// A conjunction of output half-spaces.
    Halfspace,
    /// A disjunction of clauses.
    Disjunctive,
    /// Two or more networks (relational).
    Relational,
    /// Outside the affine fragment.
    Nonlinear,
}

impl SpecArchetype {
    /// This archetype's bit.
    #[must_use]
    pub const fn bit(self) -> u32 {
        1u32 << (self as u32)
    }

    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Robustness => "robustness",
            Self::Halfspace => "halfspace",
            Self::Disjunctive => "disjunctive",
            Self::Relational => "relational",
            Self::Nonlinear => "nonlinear",
        }
    }
}

/// A set of specification archetypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SpecSet(pub u32);

impl SpecSet {
    /// No archetypes.
    pub const EMPTY: Self = Self(0);

    /// This set plus `spec`.
    #[must_use]
    pub const fn with(self, spec: SpecArchetype) -> Self {
        Self(self.0 | spec.bit())
    }

    /// Whether `spec` is in the set.
    #[must_use]
    pub const fn contains(self, spec: SpecArchetype) -> bool {
        self.0 & spec.bit() != 0
    }

    /// #router-r5: every archetype.
    pub const ALL: Self = Self(
        SpecArchetype::Robustness.bit()
            | SpecArchetype::Halfspace.bit()
            | SpecArchetype::Disjunctive.bit()
            | SpecArchetype::Relational.bit()
            | SpecArchetype::Nonlinear.bit(),
    );

    /// #router-r5: whether every archetype of `self` is in `other`.
    #[must_use]
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    /// #admission-precision: the archetypes of an affine single-network
    /// property (every archetype the β-CROWN lanes parse: relational and
    /// nonlinear properties take their own exclusive routes).
    pub const AFFINE: Self = Self(
        SpecArchetype::Robustness.bit()
            | SpecArchetype::Halfspace.bit()
            | SpecArchetype::Disjunctive.bit(),
    );
}

/// The operator envelope a capability's S7/S8 evidence covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct OpEnvelope {
    /// Operator kinds covered.
    pub ops: OpSet,
    /// Specification archetypes covered.
    pub specs: SpecSet,
}

impl OpEnvelope {
    /// No evidence: covers nothing.
    pub const NONE: Self = Self {
        ops: OpSet::EMPTY,
        specs: SpecSet::EMPTY,
    };

    /// Whether the envelope covers an instance with operator signature `ops`
    /// and specification `spec`. An unknown specification is not covered.
    #[must_use]
    pub const fn covers(self, ops: OpSet, spec: Option<SpecArchetype>) -> bool {
        match spec {
            Some(spec) => ops.is_subset_of(self.ops) && self.specs.contains(spec),
            None => false,
        }
    }
}

/// Where a capability's admission predicate lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdmissionRef {
    /// Admitted on every row (policy rules, the always-on fast path).
    Always,
    /// The lane's OWN predicate, by function name; `ny-cli` calls it and the
    /// lane re-checks it.
    Predicate {
        /// The predicate's function name.
        function: &'static str,
    },
    /// The admission is inline in a lane body and cannot be called yet: the
    /// registry reports `Unknown("inline-admission")` and the capability is
    /// admission debt (`router_admission_debt.tsv`).
    Inline {
        /// Where the inline admission lives.
        site: &'static str,
    },
    /// Admitted exactly when its parent lane is.
    Parent(CapId),
    /// Never routable in its current form (the string says why).
    NotRoutable(&'static str),
}

/// A capability's answer to "does your own predicate admit this row?".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Admission {
    /// Admitted.
    Admit,
    /// Declined, with the predicate's reason.
    Decline(&'static str),
    /// Could not be evaluated. A refusal, never a guess.
    Unknown(&'static str),
}

impl Admission {
    /// Whether the row is admitted.
    #[must_use]
    pub const fn is_admit(self) -> bool {
        matches!(self, Self::Admit)
    }
}

/// #router-r5: a capability's routability level, stage SR of the checklist
/// (ROUTER_PROGRAM §3.5), DERIVED from the capability registry's S-ladder by
/// [`SrLadder::level`], never asserted on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SrLevel {
    /// Declared: the registry row exists (its admission is called where the
    /// lane calls it; its progress tap, if any, is on a production line).
    Decl,
    /// Observed in shadow on a real row (a recorded `[router] cap=<id>` line
    /// and a progress sample).
    Obs,
    /// Sealed: S7 (negative controls, synthetic sats, knife-edge) and S8 (the
    /// both-year moat in both profiles) are done and the envelope those
    /// evidence rows cover is recorded. The ONLY level the router routes.
    Seal,
}

impl SrLevel {
    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Decl => "decl",
            Self::Obs => "obs",
            Self::Seal => "seal",
        }
    }
}

/// #router-r5: the registry stages SR is derived from (mirrors the
/// capability's `record:` in `docs/engine_program_capabilities.yaml`; a test
/// re-derives it from that file, so the compiled ladder cannot drift).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SrLadder {
    /// S7 done: negative controls, synthetic sats and the knife-edge ladder.
    pub s7: bool,
    /// S8 done: the both-year moat in both profiles, 0 flips.
    pub s8: bool,
    /// SR.obs recorded: a shadow run on a real row showed the capability.
    pub obs: bool,
}

impl SrLadder {
    /// Nothing recorded beyond the declaration.
    pub const DECLARED: Self = Self {
        s7: false,
        s8: false,
        obs: false,
    };

    /// The SR level: `Seal` needs S7 AND S8 AND a recorded (non-empty)
    /// envelope; otherwise `Obs` when a shadow observation is recorded, else
    /// `Decl`.
    #[must_use]
    pub const fn level(self, envelope: OpEnvelope) -> SrLevel {
        let recorded = envelope.ops.0 != 0 && envelope.specs.0 != 0;
        if self.s7 && self.s8 && recorded {
            SrLevel::Seal
        } else if self.obs {
            SrLevel::Obs
        } else {
            SrLevel::Decl
        }
    }
}

/// #router-r5: the moat risk of the typed keys a route sets. `High` (a key
/// that can move a verdict-bearing computation's reach) is never selected
/// automatically unless the capability is sealed; a key with no declaration
/// counts as `High` (unknown risk is high risk).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnobRisk {
    /// Cannot move a verdict (telemetry, width of an order-independent sum).
    Low,
    /// Can cost or (if the capability were unsound) mint a verdict.
    High,
}

impl KnobRisk {
    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
        }
    }
}

/// #router-r5: one typed key a route sets, with the value the capability's
/// S7/S8 evidence arm ran (`margin_row.tree_mode = wave`). The key is the
/// dotted typed-key path the capability's own consumer reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteSetting {
    /// The typed key path.
    pub key: &'static str,
    /// The value its evidence arm ran.
    pub value: &'static str,
}

/// #router-r5: how the router CONFIGURES a capability it routes: the
/// settings of its evidence arm, their moat risk, and the exclusive slot it
/// fills (two variants of one slot are never routed together).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteDecl {
    /// The settings a route applies where the row did not pin them.
    pub settings: &'static [RouteSetting],
    /// The highest moat risk among those keys.
    pub risk: KnobRisk,
    /// The exclusive slot (`tree_mode`), if any.
    pub slot: Option<&'static str>,
}

/// #admission-precision: WHERE inside its parent lane a routable consumer
/// sits, as a named site whose reachability is the lanes' own route checks
/// composed (`super::admission::site_reachable`): the router asks it at start
/// on the row's operator signature and specification shape, and the lanes
/// call the same checks inline. Data only here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmitSite {
    /// `verify_multi_clause_disjunction` on a `GraphNetwork` (its forward root
    /// screen): `fwd-root-screen`.
    DisjunctiveLaneGraph,
    /// A non-conjunctive graph multi-objective BaB (the cover certification
    /// wraps it): `vit-cover-certify`.
    DisjunctiveMultiObjective,
    /// The graph input-split lane (its deferred rebound): `cgan-child-zono`.
    GraphInputSplit,
    /// A graph BaB above an n-ary node (the lA warm-start frontier):
    /// `la-warm-frontier`.
    DagGraph,
    /// #kfsb-production: a graph ReLU-split BaB (the shared graph kFSB
    /// selector: the DomainList lane on a graph model, and the conjunctive
    /// multi-objective lane's aggregation-critical route, which a sequential
    /// model reaches through the CLI's conjunction upgrade or the disjunctive
    /// lane's per-disjunct graph BaB): `kfsb-single-shot`.
    GraphReluSplit,
}

impl AdmitSite {
    /// Every site, in declaration order.
    pub const ALL: [Self; 5] = [
        Self::DisjunctiveLaneGraph,
        Self::DisjunctiveMultiObjective,
        Self::GraphInputSplit,
        Self::DagGraph,
        Self::GraphReluSplit,
    ];

    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::DisjunctiveLaneGraph => "disjunctive_lane_graph",
            Self::DisjunctiveMultiObjective => "disjunctive_multi_objective",
            Self::GraphInputSplit => "graph_input_split",
            Self::DagGraph => "dag_graph",
            Self::GraphReluSplit => "graph_relu_split",
        }
    }

    /// The decline reason when the row cannot reach the site.
    #[must_use]
    pub const fn decline(self) -> &'static str {
        match self {
            Self::DisjunctiveLaneGraph => "site-unreached:disjunctive_lane_graph",
            Self::DisjunctiveMultiObjective => "site-unreached:disjunctive_multi_objective",
            Self::GraphInputSplit => "site-unreached:graph_input_split",
            Self::DagGraph => "site-unreached:dag_graph",
            Self::GraphReluSplit => "site-unreached:graph_relu_split",
        }
    }
}

/// #router-r5: what a capability needs from a row before the router may
/// route it — the ROUTING PRECONDITION, data only, evaluated name-blind on the
/// row's static facts and the host envelope ([`AdmitWhen::evaluate`]).
///
/// It is not a copy of the lane's admission (§3.2 C2, JIG F20): it states
/// where the capability's consumer sits (the specification archetypes that
/// call site serves), what its code needs to be useful at all (an operator it
/// acts on), size ceilings its own code declares, and the host it needs. The
/// lane's own admission still runs inside the lane whatever the router routes
/// (§3.2 C3), and routing additionally requires the SEALED envelope to cover
/// the row's operator signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdmitWhen {
    /// At least one of these operator kinds must be present (EMPTY: none
    /// required).
    pub ops_any: OpSet,
    /// The specification archetypes the consumer's call site serves. A row
    /// whose archetype is unknown at router start defers this check to that
    /// call site (the parent route re-checks its own admission there).
    pub specs: SpecSet,
    /// Parameter ceiling its own code declares (`None`: none declared).
    pub max_params: Option<u64>,
    /// Perturbed-input-dimension ceiling its own code declares.
    pub max_p_eff: Option<u64>,
    /// Device it needs.
    pub device: DeviceUse,
    /// Worker threads it needs (0: any).
    pub min_threads: u32,
    /// #admission-precision: the site inside its parent lane the consumer
    /// sits at (`None`: no site check). A row whose operator signature and
    /// specification shape cannot reach it is declined
    /// ([`AdmitSite::decline`]); an unknown shape the operators do not decide
    /// is deferred to that site, exactly as an unknown archetype is.
    #[serde(default)]
    pub site: Option<AdmitSite>,
}

/// Decline: the row has none of the operators the capability acts on.
pub const DECLINE_NO_OPERATOR: &str = "no-operator-it-acts-on";
/// Decline: the row's specification archetype is not served at the call site.
pub const DECLINE_SPEC: &str = "spec-not-served";
/// Decline: more parameters than its own code declares it handles.
pub const DECLINE_PARAMS: &str = "params-over-ceiling";
/// Decline: more perturbed dimensions than its own code declares.
pub const DECLINE_P_EFF: &str = "p-eff-over-ceiling";
/// Decline: the host has no qualified device.
pub const DECLINE_DEVICE: &str = "no-device";
/// Decline: the host has fewer threads than it needs.
pub const DECLINE_THREADS: &str = "threads-under-floor";
/// Unknown: the admission is inline in the lane body (the lane re-checks it
/// at its own site; the registry reports this token for it).
pub const UNKNOWN_INLINE: &str = "inline-admission";
/// Unknown: the facts could not be extracted.
pub const UNKNOWN_FACTS: &str = "facts-unknown";
/// Unknown: the operator signature is only a lower bound.
pub const UNKNOWN_OPS: &str = "ops-incomplete";
/// Unknown: a size fact its ceiling needs was not measured.
pub const UNKNOWN_SIZE: &str = "size-unmeasured";

impl AdmitWhen {
    /// No requirement at all (admits every row with known facts).
    pub const ANY: Self = Self {
        ops_any: OpSet::EMPTY,
        specs: SpecSet::EMPTY,
        max_params: None,
        max_p_eff: None,
        device: DeviceUse::None,
        min_threads: 0,
        site: None,
    };

    /// Evaluate the precondition on `facts` and `host`. Pure and total:
    /// unknown facts are `Unknown` (a refusal), never a guess; an unknown
    /// specification archetype is NOT a refusal (it is deferred to the
    /// consumer's call site, see [`AdmitWhen::specs`]), and neither is a site
    /// the facts cannot decide (see [`AdmitWhen::site`]).
    #[must_use]
    pub fn evaluate(
        &self,
        facts: &super::facts::InstanceFacts,
        host: &super::facts::HostEnvelope,
    ) -> Admission {
        if !facts.is_known() {
            return Admission::Unknown(UNKNOWN_FACTS);
        }
        if !facts.ops_complete {
            return Admission::Unknown(UNKNOWN_OPS);
        }
        if self.ops_any.0 != 0 && facts.ops.0 & self.ops_any.0 == 0 {
            return Admission::Decline(DECLINE_NO_OPERATOR);
        }
        if let Some(spec) = facts.spec {
            if self.specs.0 != 0 && !self.specs.contains(spec) {
                return Admission::Decline(DECLINE_SPEC);
            }
        }
        if let Some(site) = self.site {
            let reachable =
                super::admission::site_reachable(site, facts.ops, facts.spec_shape.as_ref());
            if reachable == Some(false) {
                return Admission::Decline(site.decline());
            }
        }
        if let Some(ceiling) = self.max_params {
            match facts.params {
                Some(params) if params > ceiling => return Admission::Decline(DECLINE_PARAMS),
                Some(_) => {}
                None => return Admission::Unknown(UNKNOWN_SIZE),
            }
        }
        if let Some(ceiling) = self.max_p_eff {
            match facts.p_eff {
                Some(p_eff) if p_eff > ceiling => return Admission::Decline(DECLINE_P_EFF),
                Some(_) => {}
                None => return Admission::Unknown(UNKNOWN_SIZE),
            }
        }
        let has_device = !matches!(host.device, super::facts::DeviceQual::None);
        if !matches!(self.device, DeviceUse::None) && !has_device {
            return Admission::Decline(DECLINE_DEVICE);
        }
        if host.threads < self.min_threads {
            return Admission::Decline(DECLINE_THREADS);
        }
        Admission::Admit
    }
}

/// #router-r5: everything the router needs to ROUTE a capability (its
/// precondition, its SR ladder, its route). `None` on every capability the
/// router never routes (lanes, rules, today's shipped phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Routable {
    /// The routing precondition.
    pub admit_when: AdmitWhen,
    /// The registry stages SR is derived from.
    pub ladder: SrLadder,
    /// How a route configures it.
    pub route: RouteDecl,
}

/// One routable capability (§3.1). Plain data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CapabilityDecl {
    /// Identity (== the capability registry `id` when one exists).
    pub id: CapId,
    /// Kind.
    pub kind: CapKind,
    /// The lane's own admission predicate.
    pub admission: AdmissionRef,
    /// Operator envelope covered by its S7/S8 evidence.
    pub envelope: OpEnvelope,
    /// Ledger status.
    pub ledger: LedgerStatus,
    /// Progress kind.
    pub progress: ProgressKind,
    /// Work unit.
    pub unit: WorkUnit,
    /// First-rung cost.
    pub first_rung: FirstRung,
    /// Resources.
    pub resources: ResourceDecl,
    /// Safe points.
    pub safe_points: SafePoints,
    /// Amendments honoured at a safe point.
    pub accepts: AmendmentSet,
    /// Registry axes it exposes.
    pub knobs: &'static [KnobRef],
    /// Every stop reason an exit path may record.
    pub stop_reasons: &'static [&'static str],
    /// Its own O-STOP rules.
    pub stop_rules: &'static [StopRuleId],
    /// #router-r5: how the router may route it (`None`: never routed).
    pub routable: Option<Routable>,
}

impl CapabilityDecl {
    /// #router-r5: the capability's SR level (`Decl` when it is not routable).
    #[must_use]
    pub const fn sr_level(&self) -> SrLevel {
        match self.routable {
            Some(routable) => routable.ladder.level(self.envelope),
            None => SrLevel::Decl,
        }
    }
}
