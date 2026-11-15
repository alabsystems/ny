// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The capability registry (ROUTER_PROGRAM §3.3, §3.4): every routable
//! capability as a [`CapabilityDecl`], and [`admit`], which asks each one's
//! OWN admission predicate about the row.
//!
//! `admit` calls EXISTING predicates only and never invents a static
//! threshold (§3.2 C2, JIG F20). A lane whose admission is inline in a lane
//! body, or whose predicate needs an input the router must never build (a
//! loaded graph, a parsed specification), reports `Unknown` — a refusal, so
//! today's route runs — and is listed in `router_admission_debt.tsv`, a
//! ratchet that may only fall.
//!
//! Ids equal the `id` of the capability's entry in
//! `docs/engine_program_capabilities.yaml` when one exists (its `routable:`
//! block), else an entry of that file's `routable_lanes:` list
//! (`router_registry_ids_match_capability_registry`). Ledger statuses are
//! today's: `SealedAsShipped` only where the shipped route already runs the
//! lane (derived from receipts, never from yaml keys), `Open` with the missing
//! stage for every ENGINE_PROGRAM tier, `Shadow` for the rules.

use ny_core::router::decl::{
    ids, Admission, AdmissionRef, AdmitSite, AdmitWhen, AmendmentSet, CapId, CapKind,
    CapabilityDecl, DeviceUse, FirstRung, KnobRef, KnobRisk, LaneRole, LedgerStatus, MemModelId,
    OpEnvelope, OpKind, OpSet, ProgressKind, ResourceDecl, Routable, RouteDecl, RouteSetting,
    SafePoints, SpecArchetype, SpecSet, SrLadder, StopRuleId, WorkUnit, UNKNOWN_INLINE,
};
use ny_core::router::facts::InstanceFacts;

const CPU_ALL: ResourceDecl = ResourceDecl {
    threads_min: 1,
    threads_max: None,
    device: DeviceUse::None,
    mem: MemModelId("unmodelled"),
};

const DEVICE_SEARCH: ResourceDecl = ResourceDecl {
    threads_min: 1,
    threads_max: None,
    device: DeviceUse::Search,
    mem: MemModelId("unmodelled"),
};

const ACCEPTS_UNITS: AmendmentSet = AmendmentSet {
    deadline: true,
    hand_on_source: true,
    hand_on_sink: true,
    defer: false,
};

const ACCEPTS_ATTACK: AmendmentSet = AmendmentSet {
    deadline: true,
    hand_on_source: true,
    hand_on_sink: false,
    defer: true,
};

const NO_KNOBS: &[KnobRef] = &[];
const NO_STOP_RULES: &[StopRuleId] = &[];
const STOP_DECIDED: &[&str] = &["decided", "deadline", "none-recorded"];

/// A capability's row in the registry, with the fields most rows share.
const fn row(
    id: CapId,
    kind: CapKind,
    admission: AdmissionRef,
    ledger: LedgerStatus,
    progress: ProgressKind,
    unit: WorkUnit,
    safe_points: SafePoints,
) -> CapabilityDecl {
    CapabilityDecl {
        id,
        kind,
        admission,
        envelope: OpEnvelope::NONE,
        ledger,
        progress,
        unit,
        first_rung: FirstRung::Unknown,
        resources: CPU_ALL,
        safe_points,
        accepts: match safe_points {
            SafePoints::BetweenUnits => ACCEPTS_UNITS,
            SafePoints::None => AmendmentSet::NONE,
        },
        knobs: NO_KNOBS,
        stop_reasons: STOP_DECIDED,
        stop_rules: NO_STOP_RULES,
        routable: None,
    }
}

// ---- #router-r5: routing data (ROUTER_PROGRAM §3.5, §10 R5) -----------------
//
// Each routable capability declares its ROUTING PRECONDITION (`AdmitWhen`:
// the specification archetypes its consumer's call site serves, an operator
// it needs to act on at all, size ceilings its own code declares, the host it
// needs), its SR ladder (mirroring the S7 / S8 / SR.obs record of its entry
// in `docs/engine_program_capabilities.yaml`; re-derived from that file by
// `router_r5_registry_ladders_match_the_capability_registry`), and the route:
// the typed keys its evidence arm set, their moat risk, its exclusive slot.
// Today no ladder has S7 or S8, so every routable capability is at `decl`
// and the router routes NOTHING — the expected state, pinned by a test.

/// The specification archetypes a consumer inside the margin-row lane serves
/// (the lane runs only on argmax-margin robustness rows).
const SERVES_ROBUSTNESS: SpecSet = SpecSet::EMPTY.with(SpecArchetype::Robustness);
/// #admission-precision: the archetypes the β-CROWN lanes parse (relational
/// and nonlinear properties take their own exclusive routes and never reach
/// them). The input-split, disjunctive and graph consumers below serve all
/// three: an argmax-robustness property IS a multi-clause disjunction of
/// output half-spaces to those lanes, and a boxed single clause enters the
/// disjunctive lane. Their precision is their SITE (`AdmitSite`), the lanes'
/// own route checks on the row's routing shape and operator signature.
/// (Before #admission-precision the input-split consumer declared
/// halfspace + disjunctive, and the disjunctive-route and cover-certification
/// consumers disjunctive only: with the archetype known at start those would
/// have refused every robustness row — the vit_2023 rows the cover
/// certification exists for among them.)
const SERVES_AFFINE: SpecSet = SpecSet::AFFINE;

/// A precondition that only names the archetypes the call site serves.
const fn serves(specs: SpecSet) -> AdmitWhen {
    AdmitWhen {
        specs,
        ..AdmitWhen::ANY
    }
}

/// #admission-precision: a precondition that names the archetypes the call
/// site serves AND the site itself.
const fn sits_at(specs: SpecSet, site: AdmitSite) -> AdmitWhen {
    AdmitWhen {
        specs,
        site: Some(site),
        ..AdmitWhen::ANY
    }
}

/// A route that sets `settings` (all `High`: every key below can move what a
/// verdict-bearing lane reaches, and keys with no declaration count `High`).
const fn route(settings: &'static [RouteSetting], slot: Option<&'static str>) -> RouteDecl {
    RouteDecl {
        settings,
        risk: KnobRisk::High,
        slot,
    }
}

/// Routable data with nothing recorded beyond the declaration.
const fn routable(admit_when: AdmitWhen, route: RouteDecl) -> Option<Routable> {
    Some(Routable {
        admit_when,
        ladder: SrLadder::DECLARED,
        route,
    })
}

const ROUTE_FROZEN_DUAL: &[RouteSetting] = &[RouteSetting {
    key: "margin_row.frozen_dual_bab",
    value: "true",
}];
const ROUTE_WAVE: &[RouteSetting] = &[RouteSetting {
    key: "margin_row.tree_mode",
    value: "wave",
}];
const ROUTE_WAVE_CUDA: &[RouteSetting] = &[RouteSetting {
    key: "margin_row.tree_mode",
    value: "wave_cuda",
}];
const ROUTE_BI_WINDOW: &[RouteSetting] = &[RouteSetting {
    key: "margin_row.bi_window",
    value: "true",
}];
const ROUTE_CGAN_CHILD_ZONO: &[RouteSetting] = &[RouteSetting {
    key: "solver.alpha_crown.cgan_child_zono_refresh.enabled",
    value: "true",
}];
const ROUTE_FWD_ROOT_SCREEN: &[RouteSetting] = &[RouteSetting {
    key: "bab.forward_root_screen",
    value: "true",
}];
const ROUTE_LA_WARM_FRONTIER: &[RouteSetting] = &[RouteSetting {
    key: "bab.la_warm_frontier",
    value: "true",
}];
/// The cover certification routes only as its bank arm RF (VIT_BANKABILITY.md
/// §3.1): `replay` (certify-once replay of every proved row) with the
/// containment fence and the root ORT adjudication. `ledger` alone mints a
/// token on the lane's arithmetic plus completeness only and must never be
/// armed on a banked shape (its registry entry's own gap).
const ROUTE_VIT_COVER: &[RouteSetting] = &[
    RouteSetting {
        key: "bab.mo_cover_certify",
        value: "replay",
    },
    RouteSetting {
        key: "bab.mo_containment_fence",
        value: "true",
    },
    RouteSetting {
        key: "bab.mo_root_ort_adjudication",
        value: "true",
    },
];

/// #kfsb-production: the single-shot kFSB scorer routes as its evidence arm
/// ran (`bab.kfsb_single_shot: true`).
const ROUTE_KFSB_SINGLE_SHOT: &[RouteSetting] = &[RouteSetting {
    key: "bab.kfsb_single_shot",
    value: "true",
}];

/// #prepared-root-route: the prepared-root route has no typed key: it runs by
/// default wherever its own admission admits the row, so a route sets
/// nothing (CR-RS can neither arm nor disarm it). It is routable so the
/// admission census scopes its moat and the plan prices it.
const ROUTE_NONE: &[RouteSetting] = &[];

/// The exclusive slot of the margin-row tree implementation.
const SLOT_TREE_MODE: Option<&str> = Some("tree_mode");

const PROVER: CapKind = CapKind::Lane {
    role: LaneRole::Prover { terminal_ok: true },
};
/// A prover that publishes `unsat` on its own but never takes the remainder
/// (it falls through to the lanes after it).
const PROVER_NOT_TERMINAL: CapKind = CapKind::Lane {
    role: LaneRole::Prover { terminal_ok: false },
};
const ATTACK: CapKind = CapKind::Lane {
    role: LaneRole::Attack,
};
const SCREEN: CapKind = CapKind::Lane {
    role: LaneRole::Screen,
};
const OPEN_S7_S8: LedgerStatus = LedgerStatus::Open("S7+S8 empty (checklist)");

const fn rule(id: &'static str, milestone: &'static str) -> CapabilityDecl {
    row(
        CapId(id),
        CapKind::Rule { milestone },
        AdmissionRef::Always,
        LedgerStatus::Shadow,
        ProgressKind::Opaque,
        WorkUnit::Pass,
        SafePoints::None,
    )
}

/// Every routable capability, in structural order within each kind.
pub(crate) static REGISTRY: &[CapabilityDecl] = &[
    // ---- screens -------------------------------------------------------------
    // The IBP fast path decides only obvious rows; ranked as a screen so CR-P
    // orders it first, as today's dispatch does.
    row(
        CapId("ibp-fast"),
        SCREEN,
        AdmissionRef::Always,
        LedgerStatus::SealedAsShipped,
        ProgressKind::Opaque,
        WorkUnit::Pass,
        SafePoints::None,
    ),
    row(
        CapId("box_refine.screen"),
        SCREEN,
        AdmissionRef::Inline {
            site: "ny-cli model_load.rs: >=512-clause box disjunction",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::Opaque,
        WorkUnit::Root,
        SafePoints::BetweenUnits,
    ),
    // ---- attacks -------------------------------------------------------------
    CapabilityDecl {
        accepts: ACCEPTS_ATTACK,
        ..row(
            CapId("attack.upfront_apgd"),
            ATTACK,
            AdmissionRef::Inline {
                site: "vnncomp.rs D14: differentiable graph, <=5-input common-RHS conjunction",
            },
            LedgerStatus::SealedAsShipped,
            ProgressKind::Margin,
            WorkUnit::Restart,
            SafePoints::BetweenUnits,
        )
    },
    CapabilityDecl {
        accepts: ACCEPTS_ATTACK,
        ..row(
            CapId("attack.disjunctive_pgd"),
            ATTACK,
            AdmissionRef::Inline {
                site: "verify/disjunctive.rs: disjunctive specification",
            },
            LedgerStatus::SealedAsShipped,
            ProgressKind::Margin,
            WorkUnit::Restart,
            SafePoints::BetweenUnits,
        )
    },
    CapabilityDecl {
        accepts: ACCEPTS_ATTACK,
        ..row(
            CapId("attack.bnn_lp"),
            ATTACK,
            AdmissionRef::Inline {
                site: "vnncomp.rs D17: Sign present (sign-space LP lane)",
            },
            LedgerStatus::SealedAsShipped,
            ProgressKind::Margin,
            WorkUnit::Solve,
            SafePoints::BetweenUnits,
        )
    },
    row(
        CapId("attack.bnn_ste"),
        ATTACK,
        AdmissionRef::Inline {
            site: "vnncomp.rs D17: Sign present (STE-PGD lane, fixed anneal)",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::Margin,
        WorkUnit::Step,
        SafePoints::None,
    ),
    // ---- provers -------------------------------------------------------------
    row(
        CapId("relational"),
        PROVER,
        AdmissionRef::Predicate {
            function: "RelationalRouteDecision::selected",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::Opaque,
        WorkUnit::Pass,
        SafePoints::None,
    ),
    // #prepared-root-route: the original-file prepared engine (a certified
    // root proof, else a <= 32-expansion paid tree cover, published as
    // `UnsatReceipt::from_original_prepared`). It runs BY DEFAULT on every
    // row its own admission admits (`prepared_root_route::admission`: the
    // frozen margin-row policy is the default one it executes), after the
    // upfront attack and bounded original-input Source attempt, before the
    // remaining generic provers, and falls through on an open tree. Its ledger
    // is open: origin moated it on SafeNLP only (26
    // instances); the both-year moat (S8) of every census-admitted row is
    // planned, not run.
    CapabilityDecl {
        routable: routable(serves(SERVES_ROBUSTNESS), route(ROUTE_NONE, None)),
        ..row(
            CapId("prepared-root"),
            PROVER_NOT_TERMINAL,
            AdmissionRef::Predicate {
                function: "prepared_root_route::admission",
            },
            LedgerStatus::Open("S7+S8 empty: default-on, moated on SafeNLP only"),
            ProgressKind::Opaque,
            WorkUnit::Expansion,
            SafePoints::None,
        )
    },
    row(
        CapId("nonlinear_vnnlib"),
        PROVER,
        AdmissionRef::Inline {
            site: "vnncomp.rs D45: the affine parser refused the specification",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::Opaque,
        WorkUnit::Pass,
        SafePoints::None,
    ),
    row(
        CapId("cz_hybrid"),
        PROVER,
        AdmissionRef::NotRoutable(
            "instance-id allowlist (vnncomp.rs cz_hybrid only_onnx/only_vnnlib)",
        ),
        LedgerStatus::Open("admitted by an instance allowlist, not structurally"),
        ProgressKind::Opaque,
        WorkUnit::Pass,
        SafePoints::None,
    ),
    row(
        ids::LOWDIM,
        PROVER,
        AdmissionRef::Predicate {
            function: "ny_propagate::lowdim::admit",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::Volume,
        WorkUnit::Level,
        SafePoints::BetweenUnits,
    ),
    row(
        CapId("mip.ay"),
        PROVER,
        AdmissionRef::Inline {
            site: "vnncomp.rs D27: encodable, <=5M params, unstable budget",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::Dual,
        WorkUnit::Node,
        SafePoints::None,
    ),
    row(
        CapId("input_split"),
        PROVER,
        AdmissionRef::Inline {
            site: "preset/branching.rs resolve_branching + engine input-split route",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::Volume,
        WorkUnit::Batch,
        SafePoints::BetweenUnits,
    ),
    row(
        ids::RELU_SPLIT_GRAPH,
        PROVER,
        AdmissionRef::Inline {
            site: "loader lowers the graph (run_bab_with_fallback)",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::FrontierBinary,
        WorkUnit::Batch,
        SafePoints::None,
    ),
    row(
        ids::MARGIN_ROW,
        PROVER,
        AdmissionRef::Inline {
            site:
                "vnncomp.rs D20: robustness spec, extract_twin_spec, head compiles, tableau <= 2^30",
        },
        LedgerStatus::SealedAsShipped,
        ProgressKind::FrontierBinary,
        WorkUnit::Expansion,
        SafePoints::None,
    ),
    // ---- tiers / phases / variants (ENGINE_PROGRAM capabilities) ------------
    row(
        ids::MARGIN_ROW_ALPHA,
        CapKind::Phase {
            lane: ids::MARGIN_ROW,
            slot: "root_alpha_ascent",
        },
        AdmissionRef::Parent(ids::MARGIN_ROW),
        LedgerStatus::SealedAsShipped,
        ProgressKind::Yield,
        WorkUnit::Iteration,
        SafePoints::BetweenUnits,
    ),
    // #router-r4: today's `backward_interm` phase of the margin-row lane,
    // declared so the safe-point rule can be ENFORCED on the stop that ends
    // its interm-alpha candidate loop early (`rule.ia_yield`, R3): it stops
    // between layers (its units), never inside one. Shipped wherever its
    // parent runs.
    row(
        ids::MARGIN_ROW_BACKWARD_INTERM,
        CapKind::Phase {
            lane: ids::MARGIN_ROW,
            slot: "backward_interm",
        },
        AdmissionRef::Parent(ids::MARGIN_ROW),
        LedgerStatus::SealedAsShipped,
        ProgressKind::Yield,
        WorkUnit::Pass,
        SafePoints::BetweenUnits,
    ),
    // #router-r5: routable. The tier's consumer sits inside the margin-row
    // lane's tree loop (robustness rows only).
    CapabilityDecl {
        routable: routable(serves(SERVES_ROBUSTNESS), route(ROUTE_FROZEN_DUAL, None)),
        ..row(
            ids::FROZEN_DUAL_BAB,
            CapKind::Tier {
                parent: ids::MARGIN_ROW,
            },
            AdmissionRef::Parent(ids::MARGIN_ROW),
            OPEN_S7_S8,
            ProgressKind::Bfs,
            WorkUnit::Level,
            SafePoints::BetweenUnits,
        )
    },
    // #router-r5: the device half of the wave tree (first of the two
    // variants of the margin-row tree slot). It needs a device to be more
    // than the CPU wave (without one the lane falls through to the CPU f32
    // search), so its precondition asks the host for one.
    CapabilityDecl {
        resources: DEVICE_SEARCH,
        routable: routable(
            AdmitWhen {
                device: DeviceUse::Search,
                ..serves(SERVES_ROBUSTNESS)
            },
            route(ROUTE_WAVE_CUDA, SLOT_TREE_MODE),
        ),
        ..row(
            ids::WAVE_BETA_BAB_CUDA,
            CapKind::Variant {
                slot: "tree_mode",
                value: "wave_cuda",
            },
            AdmissionRef::Parent(ids::MARGIN_ROW),
            OPEN_S7_S8,
            ProgressKind::FrontierKary,
            WorkUnit::Wave,
            SafePoints::BetweenUnits,
        )
    },
    // #router-r5: routable; the CPU variant of the margin-row tree slot. It
    // follows the device variant, so a host with a device routes the device
    // tree when both are sealed and admitted; the two never route together.
    CapabilityDecl {
        routable: routable(serves(SERVES_ROBUSTNESS), route(ROUTE_WAVE, SLOT_TREE_MODE)),
        ..row(
            ids::WAVE_BETA_BAB,
            CapKind::Variant {
                slot: "tree_mode",
                value: "wave",
            },
            AdmissionRef::Parent(ids::MARGIN_ROW),
            OPEN_S7_S8,
            ProgressKind::FrontierKary,
            WorkUnit::Wave,
            SafePoints::BetweenUnits,
        )
    },
    row(
        CapId("root-certify-once"),
        CapKind::Phase {
            lane: ids::MARGIN_ROW,
            slot: "root_alpha",
        },
        AdmissionRef::Parent(ids::MARGIN_ROW),
        OPEN_S7_S8,
        ProgressKind::Yield,
        WorkUnit::Iteration,
        SafePoints::BetweenUnits,
    ),
    CapabilityDecl {
        resources: DEVICE_SEARCH,
        ..row(
            CapId("bbe-joint-root"),
            CapKind::Variant {
                slot: "root_search",
                value: "joint_f32",
            },
            AdmissionRef::Parent(ids::MARGIN_ROW),
            OPEN_S7_S8,
            ProgressKind::Yield,
            WorkUnit::Iteration,
            SafePoints::BetweenUnits,
        )
    },
    row(
        CapId("root-box-source"),
        CapKind::Variant {
            slot: "root_box_source",
            value: "patch_backward",
        },
        AdmissionRef::Parent(ids::MARGIN_ROW),
        OPEN_S7_S8,
        ProgressKind::Opaque,
        WorkUnit::Root,
        SafePoints::None,
    ),
    // #router-r5: routable. The window crops the receptive field of a
    // convolution; on a net with none the crop is the whole net, so the
    // precondition asks for a convolution.
    CapabilityDecl {
        routable: routable(
            AdmitWhen {
                ops_any: OpSet::EMPTY.with(OpKind::Conv),
                ..serves(SERVES_ROBUSTNESS)
            },
            route(ROUTE_BI_WINDOW, None),
        ),
        ..row(
            ids::BI_WINDOW_JOINT_REFRESH,
            CapKind::Phase {
                lane: ids::MARGIN_ROW,
                slot: "backward_interm",
            },
            AdmissionRef::Parent(ids::MARGIN_ROW),
            OPEN_S7_S8,
            ProgressKind::Yield,
            WorkUnit::Pass,
            SafePoints::BetweenUnits,
        )
    },
    // #router-r5: routable; its consumer is the input-split lane's deferred
    // rebound. #admission-precision: it sits in the GRAPH input-split lane —
    // a model the loader lowers to a graph, or a sequential one the CLI
    // converts (the input-split conjunction upgrade, the disjunctive lane's
    // unified BaB).
    CapabilityDecl {
        routable: routable(
            sits_at(SERVES_AFFINE, AdmitSite::GraphInputSplit),
            route(ROUTE_CGAN_CHILD_ZONO, None),
        ),
        ..row(
            ids::CGAN_CHILD_ZONO,
            CapKind::Tier {
                parent: CapId("input_split"),
            },
            AdmissionRef::Parent(CapId("input_split")),
            OPEN_S7_S8,
            ProgressKind::Volume,
            WorkUnit::Batch,
            SafePoints::BetweenUnits,
        )
    },
    // #router-r5: routable; its consumer is the disjunctive route's root
    // precheck. #admission-precision: it sits in
    // `verify_multi_clause_disjunction` on a graph model.
    CapabilityDecl {
        routable: routable(
            sits_at(SERVES_AFFINE, AdmitSite::DisjunctiveLaneGraph),
            route(ROUTE_FWD_ROOT_SCREEN, None),
        ),
        ..row(
            ids::FWD_ROOT_SCREEN,
            CapKind::Phase {
                lane: CapId("input_split"),
                slot: "root_screen",
            },
            AdmissionRef::Parent(CapId("input_split")),
            OPEN_S7_S8,
            ProgressKind::Opaque,
            WorkUnit::Root,
            SafePoints::None,
        )
    },
    // #router-r5: the frontier-complete lA warm start of the graph lane. It
    // restores the warm start the #warm-dag guard removes from residual /
    // multi-consumer graphs; on a chain the guard already accepts the single
    // seed (bit-identical), so the precondition asks for a skip path.
    // #admission-precision: the skip path is its site, a graph BaB above an
    // n-ary node (`admission::dag_possible`): the residual / concat kinds the
    // operator precondition named, plus every kind that lowers to an n-ary
    // node — a bilinear product, a decomposed normalization, an unrecognised
    // operator — or a yaml rewrite makes one (the VGG MaxPool treatment, the
    // softmax decomposition on a graph model).
    CapabilityDecl {
        routable: routable(
            sits_at(SERVES_AFFINE, AdmitSite::DagGraph),
            route(ROUTE_LA_WARM_FRONTIER, None),
        ),
        ..row(
            ids::LA_WARM_FRONTIER,
            CapKind::Phase {
                lane: ids::RELU_SPLIT_GRAPH,
                slot: "warm_start",
            },
            AdmissionRef::Parent(ids::RELU_SPLIT_GRAPH),
            OPEN_S7_S8,
            ProgressKind::Opaque,
            WorkUnit::Pass,
            SafePoints::None,
        )
    },
    // #router-r5: the multi-objective cover certification (per-row cover
    // ledger + certify-once replay) of the graph lane's disjunctive
    // multi-objective BaB (conjunctive calls never arm it).
    // #admission-precision: it sits at a non-conjunctive graph
    // multi-objective BaB (`verify_graph_relational` over a top-level OR of
    // two or more atoms on a graph model, or the per-disjunct BaB inside the
    // disjunctive lane).
    CapabilityDecl {
        routable: routable(
            sits_at(SERVES_AFFINE, AdmitSite::DisjunctiveMultiObjective),
            route(ROUTE_VIT_COVER, None),
        ),
        ..row(
            ids::VIT_COVER_CERTIFY,
            CapKind::Phase {
                lane: ids::RELU_SPLIT_GRAPH,
                slot: "cover_certify",
            },
            AdmissionRef::Parent(ids::RELU_SPLIT_GRAPH),
            OPEN_S7_S8,
            ProgressKind::Opaque,
            WorkUnit::Pass,
            SafePoints::None,
        )
    },
    // #kfsb-production: the single-shot scorer of the shared graph kFSB
    // selector — an alternative implementation of the kFSB candidate scorer
    // inside the graph ReLU-split BaB (the DomainList lane and the
    // conjunctive multi-objective lane's aggregation-critical route). It
    // sits in a graph ReLU-split BaB on a model with a ReLU to split
    // (`AdmitSite::GraphReluSplit`); its simulations may use the device as
    // UNTRUSTED search (a branching pick), never as a published bound.
    CapabilityDecl {
        resources: DEVICE_SEARCH,
        routable: routable(
            sits_at(SERVES_AFFINE, AdmitSite::GraphReluSplit),
            route(ROUTE_KFSB_SINGLE_SHOT, None),
        ),
        ..row(
            ids::KFSB_SINGLE_SHOT,
            CapKind::Variant {
                slot: "kfsb_scorer",
                value: "single_shot",
            },
            AdmissionRef::Parent(ids::RELU_SPLIT_GRAPH),
            OPEN_S7_S8,
            ProgressKind::Opaque,
            WorkUnit::Pass,
            SafePoints::None,
        )
    },
    // ---- rules (ledger Shadow until each milestone's moat) --------------------
    // #router-r2: the seven R2 rules ACT under `router.mode: on` (acting set 1,
    // `ny_core::router::ACTING_RULES`), each ablated by its own
    // `router.rules.*` key; their ledger stays `Shadow` until the both-year
    // moat records their evidence.
    rule("rule.cap_authority", "R2"),
    rule("rule.publish_grace", "R2"),
    rule("rule.derived_caps", "R2"),
    rule("rule.hand_on", "R2"),
    rule("rule.admission", "R2"),
    rule("rule.contracts", "R2"),
    rule("rule.terminal", "R2"),
    // #router-r3: the four O-STOP rules ACT under `optimizer.mode: on`
    // (acting set 2, `ny_core::router::ACTING_OPTIMIZER_RULES`), each ablated
    // by its own `optimizer.rules.*` key; ledger `Shadow` until their moat.
    rule("rule.governor", "R3"),
    rule("rule.ia_yield", "R3"),
    rule("rule.attack_defer", "R3"),
    rule("rule.root_admission", "R3"),
    // #router-r4: acting set 3, CR-F (FAR hand-on to a strictly sealed
    // recipient) and CR-L (the memory lease), each ablated by its own
    // `router.rules.*` key; ledger `Shadow` until their moat.
    rule("rule.far_hand_on", "R4"),
    rule("rule.memory_lease", "R4"),
    // #router-r5: acting set 4, CR-RS (capabilities routed only at SR.seal)
    // and CR-G (gap-order contracts in multi-objective lanes), each ablated
    // by its own `router.rules.*` key; ledger `Shadow` until their moat.
    rule("rule.route", "R5"),
    rule("rule.gap_contract", "R5"),
    // #router-r6: the four O-TRIAL rules ACT under `optimizer.mode: on`
    // (`ny_core::router::otrial::R6_RULES`), each ablated by its own
    // `optimizer.rules.trial_*` key; ledger `Shadow` until their moat. Each
    // acts only at its sites' declared safe points (`wave-beta-bab`,
    // `input_split`, `lowdim`, `margin-row.alpha`, `attack.upfront_apgd`).
    rule("rule.trial_width", "R6"),
    rule("rule.trial_lr_backoff", "R6"),
    rule("rule.trial_split_restart", "R6"),
    rule("rule.trial_attack_halving", "R6"),
    // #router-r7: the offline cost-model prior read for planning, under
    // `optimizer.priors: true` (default false); ledger `Shadow` until its
    // moat and its held-out board.
    rule("rule.priors", "R7"),
];

/// What the registry may consult when asking a predicate. Facts only — never
/// a name or a path — plus the results of predicates the entry point already
/// ran (so the registry calls them, it does not copy them).
pub(crate) struct Context<'a> {
    /// The row's static facts.
    pub(crate) facts: &'a InstanceFacts,
    /// `RelationalRouteDecision::selected` as the entry point computed it
    /// (`None` when the entry point did not run it).
    pub(crate) relational_selected: Option<bool>,
    /// #prepared-root-route: the frozen lever set the router started with
    /// (`None`: no authority), from which a predicate whose input is the
    /// row's frozen policy resolves it.
    pub(crate) levers: Option<&'a ny_levers::LeverSet>,
}

/// The registry entry for `id`.
pub(crate) fn find(id: &str) -> Option<&'static CapabilityDecl> {
    REGISTRY.iter().find(|decl| decl.id.as_str() == id)
}

/// #router-r4: the safe points a governed deadline SLOT may be shortened at:
/// the strictest declaration among the capabilities the slot stands for
/// (`ny_core::router::acting::SLOT_CAPS`); an undeclared capability counts as
/// `None` (never).
pub(crate) fn slot_safe_points(slot: ny_core::deadline_authority::LaneSlot) -> SafePoints {
    use ny_core::router::safepoint::{declared_or_never, strictest};
    let caps = ny_core::router::acting::SLOT_CAPS
        .iter()
        .find(|(candidate, _)| *candidate == slot)
        .map_or(&[][..], |(_, caps)| *caps);
    if caps.is_empty() {
        return SafePoints::None;
    }
    caps.iter().fold(SafePoints::BetweenUnits, |acc, cap| {
        strictest(
            acc,
            declared_or_never(find(cap).map(|decl| decl.safe_points)),
        )
    })
}

/// Predicates the registry can evaluate at router start.
const CALLABLE_PREDICATES: &[&str] = &["relational", "prepared-root"];

/// Ask `decl`'s own admission predicate about the row.
pub(crate) fn admit(decl: &CapabilityDecl, ctx: &Context<'_>) -> Admission {
    admit_depth(decl, ctx, 0)
}

fn admit_depth(decl: &CapabilityDecl, ctx: &Context<'_>, depth: u8) -> Admission {
    if depth > 4 {
        return Admission::Unknown("parent-chain-too-deep");
    }
    match decl.admission {
        AdmissionRef::Always => Admission::Admit,
        AdmissionRef::NotRoutable(why) => Admission::Decline(why),
        AdmissionRef::Inline { .. } => Admission::Unknown(UNKNOWN_INLINE),
        AdmissionRef::Parent(parent) => match find(parent.as_str()) {
            Some(parent_decl) if parent_decl.id != decl.id => {
                match admit_depth(parent_decl, ctx, depth + 1) {
                    Admission::Admit => Admission::Admit,
                    Admission::Decline(_) => Admission::Decline("parent-declined"),
                    Admission::Unknown(_) => Admission::Unknown("parent-unknown"),
                }
            }
            _ => Admission::Unknown("parent-unregistered"),
        },
        AdmissionRef::Predicate { .. } => predicate(decl.id, ctx),
    }
}

fn predicate(id: CapId, ctx: &Context<'_>) -> Admission {
    match id.as_str() {
        // The relational route's own predicate: two or more declared networks
        // with a validated relation, computed by the entry point from the
        // property itself (#name-blind-routing).
        "relational" => match ctx.relational_selected {
            Some(true) => Admission::Admit,
            Some(false) => Admission::Decline("single-network-property"),
            None => Admission::Unknown("route-undecided"),
        },
        // #prepared-root-route: the route's OWN admission, the one function
        // the scored entry point calls before it runs the route, asked on the
        // row's frozen lever set.
        "prepared-root" => match ctx.levers {
            Some(levers) => crate::commands::prepared_root_route::router_admission(levers),
            None => Admission::Unknown("no-frozen-levers"),
        },
        // `ny_propagate::lowdim::admit(graph, input)` needs the loaded graph
        // and the input box; the router never loads a model.
        "lowdim" => {
            if ctx.facts.is_known() {
                Admission::Unknown("needs-loaded-graph")
            } else {
                Admission::Unknown("facts-unknown")
            }
        }
        _ => Admission::Unknown("predicate-not-callable-at-start"),
    }
}

/// `(capability, kind, where)` for every capability whose admission the
/// registry cannot evaluate: the admission debt, which may only fall.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn admission_debt() -> Vec<(&'static str, &'static str, &'static str)> {
    REGISTRY
        .iter()
        .filter_map(|decl| match decl.admission {
            AdmissionRef::Inline { site } => Some((decl.id.as_str(), "inline", site)),
            AdmissionRef::Predicate { function }
                if !CALLABLE_PREDICATES.contains(&decl.id.as_str()) =>
            {
                Some((decl.id.as_str(), "uncallable", function))
            }
            _ => None,
        })
        .collect()
}

/// A short digest of the compiled registry (every declaration's data), so a
/// replay can refuse a flight record made by a different registry.
pub(crate) fn registry_version() -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for decl in REGISTRY {
        hash.update(format!("{decl:?}\n").as_bytes());
    }
    let digest: [u8; 32] = hash.finalize().into();
    ny_core::router::hash::sha8(&digest)
}
