// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-VJP PGD routing and search controls.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const POLICY_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "exact-vjp-pgd-policy",
};
const CONJUNCTIVE_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "graph-pgd-exact-vjp-conjunctive",
};
const DISJUNCTIVE_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "graph-pgd-exact-vjp-disjunctive",
};
const SEQUENTIAL_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "graph-pgd-exact-gradient-sequential",
};

const ROUTE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: POLICY_SCOPE,
        role: "resolve the entry-time frozen lever into the typed per-run PGD policy",
        site: "crates/ny-propagate/src/pgd_attack/config.rs (ExactVjpPgdConfig)",
    },
    ReaderSite {
        scope: CONJUNCTIVE_SCOPE,
        role: "choose exact-gradient routing over the generic restart-batched SPSA route",
        site: "crates/ny-cli/src/commands/beta_crown/verify/graph_pgd.rs (try_graph_pgd_upfront_with_config)",
    },
];
const SHARED_BATCH_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: POLICY_SCOPE,
        role: "resolve the entry-time frozen lever into the typed per-run PGD policy",
        site: "crates/ny-propagate/src/pgd_attack/config.rs (ExactVjpPgdConfig)",
    },
    ReaderSite {
        scope: CONJUNCTIVE_SCOPE,
        role: "configure the conjunctive batched exact-VJP attack",
        site: "crates/ny-cli/src/commands/beta_crown/verify/graph_pgd_vjp_batched.rs (try_graph_pgd_vjp_batched)",
    },
    ReaderSite {
        scope: DISJUNCTIVE_SCOPE,
        role: "configure the disjunctive batched exact-VJP attack",
        site: "crates/ny-cli/src/commands/beta_crown/verify/graph_pgd_vjp_batched_disj.rs (try_graph_disjunctive_pgd_vjp_batched)",
    },
];
const CONJUNCTIVE_POLICY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: POLICY_SCOPE,
        role: "resolve the entry-time frozen lever into the typed per-run PGD policy",
        site: "crates/ny-propagate/src/pgd_attack/config.rs (ExactVjpPgdConfig)",
    },
    ReaderSite {
        scope: CONJUNCTIVE_SCOPE,
        role: "configure the conjunctive batched exact-VJP attack",
        site: "crates/ny-cli/src/commands/beta_crown/verify/graph_pgd_vjp_batched.rs (try_graph_pgd_vjp_batched)",
    },
];
const DISJUNCTIVE_POLICY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: POLICY_SCOPE,
        role: "resolve the entry-time frozen lever into the typed per-run PGD policy",
        site: "crates/ny-propagate/src/pgd_attack/config.rs (ExactVjpPgdConfig)",
    },
    ReaderSite {
        scope: DISJUNCTIVE_SCOPE,
        role: "configure the disjunctive batched exact-VJP attack's ODI phase and dangerous-class focus wave",
        site: "crates/ny-cli/src/commands/beta_crown/verify/graph_pgd_vjp_disj_odi.rs (DisjOdiPolicy::from_config)",
    },
];
const FAST_GRAD_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: POLICY_SCOPE,
        role: "resolve the entry-time frozen lever into the typed per-run PGD policy",
        site: "crates/ny-propagate/src/pgd_attack/config.rs (ExactVjpPgdConfig)",
    },
    ReaderSite {
        scope: SEQUENTIAL_SCOPE,
        role: "choose the point-Jacobian fast gradient before certified CROWN and SPSA fallbacks",
        site: "crates/ny-cli/src/commands/beta_crown/verify/graph_pgd_exact.rs (exact_graph_spec_gradient)",
    },
];

declare_levers! {
    registry EXACT_VJP_PGD_LEVERS;

    /// Default-on selector for exact-gradient routing on eligible conjunctions.
    pub EXACT_BATCHED = LeverDecl {
        name: "NY_PGD_EXACT_BATCHED",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "Selects the joint exact-gradient route instead of generic restart-batched SPSA on structurally exact-gradient-eligible conjunctive graphs. The historical rule is ON unless the process value is exactly `0`: exact `1` arms, exact `0` kills, and every malformed or non-Unicode token rejects to this true declaration default. Attack-only candidates still pass the independent revalidation and trusted publication gates, but route selection reallocates scored attack work and changes which SAT verdicts are reachable.",
        provenance: Provenance::Measured {
            commit: "6d4c52622",
            date: "2026-07-08",
            artifact: "commit 6d4c52622 measurement and soundnessbench model_18 replay",
            delta: "the exact joint-gradient route converted soundnessbench model_18 from timeout to trusted-gated sat in 62 seconds; the exact-0 generic-batched baseline timed out",
        },
        owner: POLICY_SCOPE,
        readers: ROUTE_READERS,
    };

    /// Default-on point-Jacobian gradient attempt in sequential graph PGD.
    pub FAST_GRAD = LeverDecl {
        name: "NY_PGD_FAST_GRAD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "Attempts the exact point-Jacobian gradient before the certified two-sided CROWN extraction in sequential conjunctive and disjunctive graph PGD. The historical rule is ON unless the process value is exactly `0`: exact `1` arms, exact `0` kills, and malformed or non-Unicode input rejects to this true declaration default. If the point path declines, certified extraction and then SPSA remain available. The route is attack-only and every candidate still passes independent revalidation, but changing gradient cost and quality changes which SAT verdicts are reachable inside the scored wall.",
        provenance: Provenance::Measured {
            commit: "8697699e9",
            date: "2026-07-07",
            artifact: "commits 406bd3c3a (initial point-VJP route) and 8697699e9 (default-on CIFAR100 medium sweep)",
            delta: "the default-on point-gradient policy converted 13 of 14 CIFAR100 medium SAT rows versus 0 of 14 for the certified-gradient baseline, with every witness independently confirmed by ONNX Runtime",
        },
        owner: POLICY_SCOPE,
        readers: FAST_GRAD_READERS,
    };

    /// Shared kill switch for the two batched exact-VJP drivers.
    pub VJP_BATCH = LeverDecl {
        name: "NY_PGD_VJP_BATCH",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "Admits wide batched exact point-VJP waves in both conjunctive and disjunctive graph PGD. It preserves the historical ON-unless-exact-0 parser, including malformed and non-Unicode values resolving to ON. Exact `0` selects the established sequential fallback. Every candidate remains attack-only and independently revalidated, while the choice changes restart throughput inside the scored slice.",
        provenance: Provenance::Measured {
            commit: "9676e9908",
            date: "2026-07-16",
            artifact: "commit bodies 986d99275, 320875d0e, and 9676e9908",
            delta: "batched exact VJP converted MetaRoom specs 129/148, five soundnessbench hard basins, and all seven measured CIFAR100/TinyImageNet robustness misses to trusted-gated sat",
        },
        owner: POLICY_SCOPE,
        readers: SHARED_BATCH_READERS,
    };

    /// Contextual restart width: 64 conjunctive, 32 disjunctive.
    pub VJP_K = LeverDecl {
        name: "NY_PGD_VJP_K",
        kind: LeverKind::Usize { min: 1 },
        default: DefaultSpec::Unset,
        bucket: Bucket::Auto,
        moat: MoatRisk::High,
        doc: "Overrides the wide exact-VJP restart width. Parsing is the historical untrimmed platform `usize` parser followed by `>= 1`; invalid input falls back contextually to 64 lanes in the conjunctive driver or 32 in the disjunctive driver. The conjunctive driver may halve the effective width after an adapter capacity refusal, which execution evidence records separately from the request.",
        provenance: Provenance::Measured {
            commit: "320875d0e",
            date: "2026-07-16",
            artifact: "commit 320875d0e wide-wave throughput table and official soundnessbench sweep",
            delta: "K=64 cost 3.0ms per restart-step versus 4.3ms at K=32 and, composed with ODI/exploit, produced a net +3 soundnessbench conversions; the disjunctive K=32 default converted MetaRoom specs 129/148",
        },
        owner: POLICY_SCOPE,
        readers: SHARED_BATCH_READERS,
    };

    /// Default-off uniform replay lanes in the first conjunctive wave.
    pub LEGACY_LANES = LeverDecl {
        name: "NY_PGD_LEGACY_LANES",
        kind: LeverKind::Usize { min: 0 },
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Number of trailing lanes in the first conjunctive wave that replay the pre-ODI uniform seed sequence. Zero disables. Parsing is the exact untrimmed platform `usize` grammar. Values saturate at one fewer than the active wave width. This is an attack trajectory probe, not a proof authority.",
        provenance: Provenance::Measured {
            commit: "320875d0e",
            date: "2026-07-16",
            artifact: "commit 320875d0e soundnessbench legacy-lane Pareto sweep",
            delta: "20 legacy lanes with K=52 recovered model_0 but lost models 6/8/30, a net -2 versus the retained pure-ODI K=64 policy; zero remains the measured winner",
        },
        owner: POLICY_SCOPE,
        readers: CONJUNCTIVE_POLICY_READERS,
    };

    /// Output-diversified-initialization step count.
    pub ODI_STEPS = LeverDecl {
        name: "NY_PGD_ODI_STEPS",
        kind: LeverKind::Usize { min: 0 },
        default: DefaultSpec::U64(8),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "Number of conjunctive batched exact-VJP steps spent on output-diversified initialization before joint-margin ascent. Zero disables. The parser is untrimmed platform `usize`, preserving every legacy token and the measured eight-step default.",
        provenance: Provenance::Measured {
            commit: "320875d0e",
            date: "2026-07-16",
            artifact: "commit 320875d0e official soundnessbench sweep",
            delta: "eight ODI steps reached basins uniform K=64/96 missed and participated in five trusted-gated flips; more uniform raw steps were measured negative",
        },
        owner: POLICY_SCOPE,
        readers: CONJUNCTIVE_POLICY_READERS,
    };

    /// Output-diversified-initialization step scale.
    pub ODI_ETA = LeverDecl {
        name: "NY_PGD_ODI_ETA",
        kind: LeverKind::F32Positive,
        default: DefaultSpec::F64(1.0),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "Positive finite `f32` ODI step scale in units of each input coordinate's half-range. Parsing is deliberately untrimmed and occurs in `f32`, not `f64`; invalid, zero, negative, nonfinite, overflow, and underflow-to-zero tokens restore the measured 1.0 full-radius scale.",
        provenance: Provenance::Measured {
            commit: "320875d0e",
            date: "2026-07-16",
            artifact: "commit 320875d0e ODI implementation and official soundnessbench sweep",
            delta: "the retained full-radius eta=1.0 ODI policy, composed with eight steps and K=64, participated in five trusted-gated hard-basin flips",
        },
        owner: POLICY_SCOPE,
        readers: CONJUNCTIVE_POLICY_READERS,
    };

    /// Optional per-wave joint-margin step cap.
    pub VJP_STEPS = LeverDecl {
        name: "NY_PGD_VJP_STEPS",
        kind: LeverKind::Usize { min: 1 },
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Optional positive conjunctive joint-margin steps per exact-VJP wave. Absence is contextual and retains the typed PGD step count; a value creates more, shorter ODI waves. Parsing is the exact untrimmed platform `usize` grammar. Execution evidence records both the request and contextual effective value.",
        provenance: Provenance::Unmeasured {
            why_ok: "absence preserves the measured deep-wave schedule exactly; the explicit many-short-wave treatments are attack-only research values and have no retained scored A/B result yet",
        },
        owner: POLICY_SCOPE,
        readers: CONJUNCTIVE_POLICY_READERS,
    };

    /// Default-off rate-projection lane recycling window.
    pub LANE_WINDOW = LeverDecl {
        name: "NY_PGD_LANE_WINDOW",
        kind: LeverKind::Usize { min: 0 },
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Conjunctive lane-recycling assessment window in steps; zero disables. The exact untrimmed platform `usize` parser is retained. A positive value may replace projected non-climbers with fresh uniform+ODI trajectories, so it changes which counterexamples remain reachable inside the attack deadline.",
        provenance: Provenance::Measured {
            commit: "320875d0e",
            date: "2026-07-16",
            artifact: "commit 320875d0e soundnessbench rate-recycling probe",
            delta: "rate recycling moved model_29's best joint margin from about -5e-6 to -1.1e-4 without flipping models 8/29/37; zero remains the measured winner",
        },
        owner: POLICY_SCOPE,
        readers: CONJUNCTIVE_POLICY_READERS,
    };

    /// Default-on endgame exploit recycling.
    pub EXPLOIT = LeverDecl {
        name: "NY_PGD_EXPLOIT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "Enables endgame respawning of the worst conjunctive lanes as small jittered clones of the best near-boundary point. The historical rule is ON unless the raw value is exactly `0`; malformed and non-Unicode values reject to ON. This is attack-only and cannot authorize a candidate, but it changes late trajectory coverage.",
        provenance: Provenance::Measured {
            commit: "320875d0e",
            date: "2026-07-16",
            artifact: "commit 320875d0e exploit-recycle timing and soundnessbench sweep",
            delta: "endgame-only exploit recycling flipped model_8 and produced an internal crossing on model_35; earlier firing lost the model_6 climber, supporting the retained bounded default-on policy",
        },
        owner: POLICY_SCOPE,
        readers: CONJUNCTIVE_POLICY_READERS,
    };

    /// Default-off tau-hinge ensemble threshold.
    pub HINGE_TAU = LeverDecl {
        name: "NY_PGD_HINGE_TAU",
        kind: LeverKind::F32NonNegative,
        default: DefaultSpec::F64(0.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Finite non-negative `f32` tau for odd conjunctive lanes' hinge ensemble; zero keeps the plain hinge in every lane. Parsing is the historical untrimmed `f32` grammar, including admissible negative zero. The raw counterexample screen remains unchanged and independently revalidated.",
        provenance: Provenance::Measured {
            commit: "320875d0e",
            date: "2026-07-16",
            artifact: "commit 320875d0e soundnessbench tau-hinge probe",
            delta: "the positive tau ensemble did not break the boundary jam and regressed model_8's best margin from about -8e-5 to -1.3e-4; zero remains the measured winner",
        },
        owner: POLICY_SCOPE,
        readers: CONJUNCTIVE_POLICY_READERS,
    };

    /// #disj-odi: default-off ODI phase on the disjunctive driver's dynamic lanes.
    pub DISJ_ODI = LeverDecl {
        name: "NY_PGD_DISJ_ODI",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "#disj-odi. Arms output-diversified initialization inside the batched disjunctive exact-VJP driver: before the step loop of every restart wave, the wave's DYNAMIC (uniform-init, best-disjunct) lanes take `NY_PGD_ODI_STEPS` box-clamped sign steps of scale `NY_PGD_ODI_ETA` half-ranges up a fixed random output direction, through one batched exact VJP per step, with a free candidate screen on the ODI trajectory; targeted lanes keep their clean init. Until now the disjunctive driver never read the ODI fields, so every cifar100/tinyimagenet row reported `odi_steps_configured=8 odi_vjp_steps=0` (configured, not arrived). Default false: the driver never enters the phase and is byte-identical. Attack-only: a candidate must still clear the driver's exact re-screen (`noise_scaled_margin`), `re_evaluate_and_confirm` and the trusted-ORT publication gate, and nothing here can produce `unsat`. High because it changes which counterexamples the attack slice reaches and spends ODI VJP passes of that slice.",
        provenance: Provenance::Unmeasured {
            why_ok: "default false and armed by no shipped yaml, so promotion changes no scored run; the dark driver is byte-identical (named test disj_odi_dark_is_byte_identical) and an armed run counts its ODI passes (disj_odi_moves_lanes_and_counts_vjp_steps); the calibration it ports (tinyimagenet GAP idx_1831/idx_5529, ORT-confirmed in torch) is not yet measured through ny",
        },
        owner: POLICY_SCOPE,
        readers: DISJUNCTIVE_POLICY_READERS,
    };

    /// #disj-odi: default-off dangerous-class focus lanes of the disjunctive driver.
    pub DISJ_FOCUS_LANES = LeverDecl {
        name: "NY_PGD_DISJ_FOCUS_LANES",
        kind: LeverKind::Usize { min: 0 },
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "#disj-odi. Number of dangerous-class focus lanes the batched disjunctive exact-VJP driver runs AFTER its restart waves finish without a candidate and before the attack deadline. The two closest-to-violation clauses seen so far (clean point plus every restart-wave point; rule `wave0`, or `clean` when no wave point was observed) are pinned round-robin across the lanes; each lane starts uniform plus `NY_PGD_ODI_STEPS` ODI steps, ascends its clause's margin row under the `NY_PGD_HINGE_TAU` hinge (a lane at or above tau contributes no gradient) with bias-corrected-Adam sign steps on the cosine schedule h*(0.15*(1+cos(pi*k/200))+0.003) for up to 200 steps. The wave is RSS-admitted against half the observed memory headroom (4 GiB when unobservable), capped at 256 lanes, and halves its width on a wide-VJP refusal. Zero disables and the driver is byte-identical. Attack-only, same candidate funnel as every other lane; it never sets the restart-sequence deadline flag, so the attack extension decision is unchanged. High because it spends attack-slice wall and changes which counterexamples are reachable.",
        provenance: Provenance::Unmeasured {
            why_ok: "zero by default and armed by no shipped yaml, so promotion changes no scored run; the dark driver is byte-identical (disj_odi_dark_is_byte_identical), focus lanes pin the ranked classes and respect the hinge (disj_focus_lanes_pin_the_selected_class_and_respect_the_hinge), and a below-floor near miss never becomes a candidate (disj_odi_near_miss_below_floor_never_returns_a_candidate); the conversion A/B on tinyimagenet idx_1831/idx_5529 is the pending measurement",
        },
        owner: POLICY_SCOPE,
        readers: DISJUNCTIVE_POLICY_READERS,
    };
}
