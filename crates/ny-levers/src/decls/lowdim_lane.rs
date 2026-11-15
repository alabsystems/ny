// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The `#lowdim-prover` lane's lsnc stack (#lsnc-lane).
//!
//! `lsnc_relu` (80 rows in each year, a 6-input Lyapunov certificate) is
//! unboundable on the real graph: the exact loader fails four constant-only
//! `MatMul` cones closed to `OpaqueSkip`, so every bound is `[-inf, +inf]`. The
//! owner decision (2026-09-24) is a LANE-PRIVATE rigorous enclosure of those
//! cones: a second, interval-folded load of the model that only the lowdim
//! lane can bound on (`ny_propagate::lowdim::LaneTwin`). The global carrier
//! `NY_CONST_MATMUL_INTERVAL` stays `Deliver::EnvOnly` and dark.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const LOWDIM_LANE_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "lowdim-lane-twin",
};

const LOWDIM_INTERVAL_TWIN_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_interval_twin into the frozen \
               scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the twin admission from the frozen LeverSet (scored) or the typed \
               preset over the environment (interactive), then build and attach the \
               lane-private interval twin to the freshly loaded, unmodified graph",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_interval_twin, \
               attach_lowdim_interval_twin",
    },
];

const LOWDIM_JOINT_QUADRATIC_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_joint_quadratic into the frozen \
               scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the joint quadratic bound from the frozen LeverSet (scored) or the \
               typed preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_joint_quadratic, which lowdim_config \
               delivers to every lowdim entry as LowDimConfig::joint_quadratic",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_joint_quadratic",
    },
];

const LOWDIM_SPLIT_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_split into the frozen scored-run \
               LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's split rule from the frozen LeverSet (scored) or the typed \
               preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_split, which lowdim_config delivers to \
               every lowdim entry as LowDimConfig::split",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_split",
    },
];

const LOWDIM_CLIP_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_clip into the frozen scored-run \
               LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's relaxed clip from the frozen LeverSet (scored) or the \
               typed preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_clip, which lowdim_config delivers as \
               LowDimConfig::clip",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_clip",
    },
];

const LOWDIM_PARALLEL_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_parallel into the frozen scored-run \
               LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's wave mode from the frozen LeverSet (scored) or the typed \
               preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_parallel, which lowdim_config delivers as \
               LowDimConfig::parallel (Full when on, the historical Bound when off)",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_parallel",
    },
];

const LOWDIM_VARYING_DIMS_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_varying_dims into the frozen \
               scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's varying-dims admission (and Sigmoid lowering) from the \
               frozen LeverSet (scored) or the typed preset over the environment \
               (interactive) into BetaCrownConfig::input_split_lowdim_varying_dims, which \
               lowdim_config delivers as LowDimConfig::varying_dims",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_varying_dims",
    },
];

const LOWDIM_ALPHA_ITERS_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_alpha_iters into the frozen \
               scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's boundary-alpha iteration count from the frozen LeverSet \
               (scored) or the typed preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_alpha_iters, which lowdim_config \
               delivers as LowDimConfig::alpha_iters",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_alpha_iters",
    },
];

const LOWDIM_LEAF_ATTACK_EVERY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_leaf_attack_every into the frozen \
               scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's in-lane leaf-attack period from the frozen LeverSet \
               (scored) or the typed preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_leaf_attack_every, which lowdim_config \
               delivers as LowDimConfig::leaf_attack_every",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_leaf_attack_every",
    },
];

const LOWDIM_SHARED_WORKLIST_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_shared_worklist into the frozen \
               scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the fold's shared worklist from the frozen LeverSet (scored) or the \
               typed preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_shared_worklist, which lowdim_config \
               delivers as LowDimConfig::shared_worklist",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_shared_worklist",
    },
];

declare_levers! {
    registry LOWDIM_LANE_LEVERS;

    /// `NY_LOWDIM_INTERVAL_TWIN` — build the `#lowdim-prover` lane's private
    /// interval twin of a model whose real graph carries `OpaqueSkip` nodes.
    pub LOWDIM_INTERVAL_TWIN = LeverDecl {
        name: "NY_LOWDIM_INTERVAL_TWIN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Admits the #lowdim-prover lane's LANE-PRIVATE interval twin. When the loaded \
graph carries OpaqueSkip nodes (the exact loader declined a constant-only cone \
whose rounded binary32 product is not the authored product), and the graph is \
the unmodified model (no terminal peel, no load-time rewrite, one declared \
input), the model is loaded a second time with the loader's audited interval \
fallback enabled for that one load only (a thread-scoped guard; \
NY_CONST_MATMUL_INTERVAL is NOT read or armed), and the resulting graph is \
moved into ny_propagate::lowdim::LaneTwin, whose graph only the lowdim lane \
can read. The lane then lowers the twin instead of the real graph: every \
IntervalConst is lowered WITH its radius (coefficient errors charged against \
the operand's rigorous magnitude, offsets as outward interval constants), and \
two-operand products use McCormick planes built from rigorous operand boxes, \
so every lane bound holds for every member of the enclosed constant set. The \
binding to the real graph is invalidated by any structural mutation. Exact \
\"1\" arms it and exact \"0\" disarms it; absence defers to the typed key \
bab.branching.input_split.lowdim_interval_twin, whose absence is false. \
MoatRisk::High because it makes a new family's rows provable by the lowdim \
lane — a bound path on lsnc_relu, which has published false unsat verdicts \
before (807bf511).",
        provenance: Provenance::Unmeasured {
            why_ok: "default-false and set by no shipped preset; armed only in a \
                     measurement config until the lane's per-row soundness evidence \
                     (sampled containment against an independent exact f64 forward of \
                     the real ONNX, the row-2 root bound against ORT samples, and an \
                     independent adjudication of every unsat) lands; with it unset no \
                     twin is built and every graph lowers exactly as before",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_INTERVAL_TWIN_READERS,
    };

    /// `NY_LOWDIM_JOINT_QUADRATIC` — bound the `#lowdim-prover` lane's
    /// `ReduceSum(Mul(a, b))` sites by the joint quadratic bound.
    pub LOWDIM_JOINT_QUADRATIC = LeverDecl {
        name: "NY_LOWDIM_JOINT_QUADRATIC",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lsnc-lane stage 2. Bounds every ReduceSum(Mul(a, b)) site of the #lowdim-prover \
lane's lowered program (lsnc_relu's Lyapunov value V and its decrease dV) by the \
JOINT QUADRATIC bound instead of per-product McCormick alone: each operand \
element is reduced to a midpoint affine plus a rigorous radius from its forward \
forms and two CROWN certificates (backward passes seeded at the operand, +e_i \
and -e_i, over the same forward pass); the sum of midpoint products is ONE \
quadratic over the input box; its matrix is split by a Jacobi sweep with the \
residual E = H - U diag(lambda) U' charged in full; the convex part is maximised \
exactly over the box vertices and the concave part replaced by a tangent. The \
site's interval is intersected with the result, and the tangent-at-centre planes \
plus the rigorous remainder (from the same pass, never carried to another box) \
are substituted in a second backward pass. Every step is f64 with outward \
rounding. Exact \"1\" arms it and exact \"0\" disarms it; absence defers to \
the typed key bab.branching.input_split.lowdim_joint_quadratic, whose absence is \
false. A program without such a site is unaffected. MoatRisk::High because it \
makes a family with a false-unsat history (807bf511) provable by a new bound.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-false and set by no shipped preset; armed only in the \
                     scratchpad/lsnclane measurement configs until the lane's \
                     throughput micro-benchmark, per-row soundness evidence (sampled \
                     containment of the joint sites, planes and published bounds against \
                     an independent exact f64 forward of the real ONNX; the row-2 root pin \
                     against the numpy prototype) and an independent adjudication of every \
                     unsat land; with it unset the lane bounds exactly as before",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_JOINT_QUADRATIC_READERS,
    };

    /// `NY_LOWDIM_SPLIT` — the `#lowdim-prover` lane's split rule.
    pub LOWDIM_SPLIT = LeverDecl {
        name: "NY_LOWDIM_SPLIT",
        kind: LeverKind::Enum(&["widest", "gradient", "sb", "sbwr"]),
        default: DefaultSpec::Text("widest"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lsnc-lane stage 3. Chooses the input coordinate the #lowdim-prover lane bisects. \
\"widest\" is the historical rule (the widest coordinate); \"gradient\" maximises \
|grad| * width for the objective closest to closing; \"sb\" is alpha-beta-CROWN's \
sb_sum score (sum over the clause's objectives of max(|lA|, 0.01) * width / 2, plus \
0.1 * width when the interval touches 0); \"sbwr\" is the numpy decision gate's \
allmax split (|lA| * width of the nearest objective, max-normalised, plus the \
max-normalised width relative to the root box). Both smart rules are restricted by an \
ASPECT GUARD: a coordinate is eligible only when its width relative to the root box \
is at least 1/8 of the largest relative width (plain SB builds 1000-deep chains on \
lsnc_relu). Every rule keeps BOTH halves of the box, so it can only shape the tree, \
never drop a point. An admissible token arms it; absence defers to the typed key \
bab.branching.input_split.lowdim_split, whose absence is \"widest\". MoatRisk::High \
because the tree shape moves a deadline-bounded verdict on a family with a \
false-unsat history (807bf511).",
        provenance: Provenance::Unmeasured {
            why_ok: "default \"widest\" is the historical rule, bit for bit (pinned by \
                     ny-propagate lowdim::search_tests), and no shipped preset sets the key; \
                     armed only in the scratchpad/lsnclane measurement configs until the \
                     lane's throughput benchmark and lsnc_relu measurement land",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_SPLIT_READERS,
    };

    /// `NY_LOWDIM_CLIP` — the `#lowdim-prover` lane's relaxed Clip-and-Verify.
    pub LOWDIM_CLIP = LeverDecl {
        name: "NY_LOWDIM_CLIP",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lsnc-lane stage 3. RELAXED CLIP-AND-VERIFY inside the #lowdim-prover lane: every \
open domain's box is shrunk to the smallest axis box containing each point that \
satisfies every conjunct's plane (a . x + c <= threshold), where each plane is one \
backward pass's OWN input multiplier and constant over THAT box (never rebuilt from \
a maximised lower bound, never carried to another box — a child is bounded and \
clipped with its own planes). Each new face is certified by one plane over the \
unclipped box, with the numerator raised by a rigorous rounding charge plus a 1e-12 \
relative guard and the face rounded outward; a box the planes empty is closed. \
Exact \"1\" arms it and exact \"0\" disarms it; absence defers to the typed key \
bab.branching.input_split.lowdim_clip, whose absence is false. MoatRisk::High \
because a false plane closes a box that holds a counterexample — the 807bf511 \
mechanism on lsnc_relu was clipping with reused, stacked planes.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-false and set by no shipped preset; armed only in the \
                     scratchpad/lsnclane measurement configs until the lane's measurement \
                     lands; soundness evidence is an exact-rational falsifier of every face \
                     (with a raw-float negative control), sampled plane containment and \
                     never-cut-a-counterexample checks against independent f64 references \
                     (synthetic every-member and the real lsnc ONNX); with it unset no box \
                     is clipped",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_CLIP_READERS,
    };

    /// `NY_LOWDIM_PARALLEL` — decide the lane's domains inside the rayon tasks.
    pub LOWDIM_PARALLEL = LeverDecl {
        name: "NY_LOWDIM_PARALLEL",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lsnc-lane stage 3. Where the #lowdim-prover lane's per-domain work runs. Off (the \
historical lane), each wave's chunks are bounded on rayon and every domain is then \
decided (close test, clip, violation-candidate test, split) serially; on, each domain \
is decided inside its chunk's rayon task and only the in-order merge is serial. Both \
cut a wave into the same chunks and decide a domain with the same function from its \
own box and bounds alone, so the outcome, every counter and any candidate point are \
identical (pinned by ny-propagate lowdim::lsnc_tests); only wall time changes. Exact \
\"1\" arms it and exact \"0\" disarms it; absence defers to the typed key \
bab.branching.input_split.lowdim_parallel, whose absence is false. MoatRisk::High \
because throughput moves a deadline-bounded verdict.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-false and set by no shipped preset; outcome-identical to the \
                     historical path by construction and by test; armed only in the \
                     scratchpad/lsnclane measurement configs until the lane's throughput \
                     benchmark lands",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_PARALLEL_READERS,
    };

    /// `NY_LOWDIM_VARYING_DIMS` — admit the `#lowdim-prover` lane by varying
    /// input dimensions, and lower `Sigmoid` (#lowdim-bundle (1)).
    pub LOWDIM_VARYING_DIMS = LeverDecl {
        name: "NY_LOWDIM_VARYING_DIMS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lowdim-bundle (1). Admits the #lowdim-prover lane by the number of VARYING input \
coordinates instead of the total: a root box wider than the lane's 16-dimension cap \
whose other coordinates are pinned (lo == hi; dist_shift_2023 mnist_concat has 784 \
pinned image pixels beside 8 latent coordinates) is re-expressed over its free \
coordinates. An embed node reproduces the full input exactly (free rows are exact \
unit terms, pinned rows the degenerate box interval); a Linear that reads the input \
keeps its free columns' exact weights and folds its pinned columns into an interval \
bias accumulated in f64 and widened by a Higham charge, rounded outward; the folded \
program is cross-checked against the unfolded one at three interior points before \
it is admitted, and a candidate is expanded back to the full input. The same key \
lowers Sigmoid to rigorous planes: slopes are heuristic (chord / tangent / zero), \
intercepts are rigorous extremes of sigmoid(z) - a z over the source box (convex / \
concave split at 0, endpoint or tangent-at-stationary-point bounds with the \
derivative enclosed and every rounding charged), intersected with the monotone \
interval image. Exact \"1\" arms it and exact \"0\" disarms it; absence defers to \
the typed key bab.branching.input_split.lowdim_varying_dims, whose absence is false. \
MoatRisk::High because it makes a new family's rows provable by a new bound path.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-false and set by no shipped preset; soundness evidence is \
                     the ny-propagate lowdim::bundle_tests suite (exact-rational plane \
                     containment against a rigorous sigmoid enclosure, sampled \
                     containment of folded bounds, never verifying a sampled-violated \
                     threshold, dark-by-default declines); with it unset every box and \
                     graph is admitted or declined exactly as before",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_VARYING_DIMS_READERS,
    };

    /// `NY_LOWDIM_ALPHA_ITERS` — boundary-alpha iterations of the lane's
    /// objective pass (#lowdim-bundle (2)).
    pub LOWDIM_ALPHA_ITERS = LeverDecl {
        name: "NY_LOWDIM_ALPHA_ITERS",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lowdim-bundle (2). Per-leaf BOUNDARY ALPHA in the #lowdim-prover lane: for open \
domains whose objective bound lies within 10 below its threshold (Vibecheck's acasxu \
wide band), this many sign-projected steps on the lower ReLU slopes alpha in [0, 1] \
of the objective's backward pass (clamped to 200). Any alpha in [0, 1] is a valid \
lower relaxation (relu(z) >= alpha z for every z); each alpha pass is the lane's own \
backward pass with every multiplier product charged, so the published bound is the \
MAXIMUM of rigorous bounds, and a clip plane from alpha is one pass's own plane on its \
own box. Intermediate bounds are untouched. 0 (the default) is the historical {0, 1} \
slope rule exactly. Absence defers to the typed key \
bab.branching.input_split.lowdim_alpha_iters, whose absence is 0. MoatRisk::High \
because it closes domains the historical bound leaves open.",
        provenance: Provenance::Unmeasured {
            why_ok: "default 0 and set by no shipped preset; soundness evidence is the \
                     ny-propagate lowdim::bundle_tests suite (alpha bounds never exceed a \
                     sampled value, the lane never verifies a sampled-violated threshold \
                     with alpha armed, alpha never lowers a published bound); with it 0 \
                     the backward pass is the historical one",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_ALPHA_ITERS_READERS,
    };

    /// `NY_LOWDIM_LEAF_ATTACK_EVERY` — in-lane worst-leaf PGD period, in
    /// waves (#lowdim-bundle (3)).
    pub LOWDIM_LEAF_ATTACK_EVERY = LeverDecl {
        name: "NY_LOWDIM_LEAF_ATTACK_EVERY",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lowdim-bundle (3). In-lane WORST-LEAF PGD: every this many waves the 64 open leaves \
of the wave furthest from closing get 256 restarts x 50 signed-gradient steps on the \
lane's f64 lowering, confined to each leaf box, within a wall share of 20% of the \
lane. ATTACK-ONLY: a hit is a point where the lowering violates every conjunct by the \
lane's candidate margin (re-checked by the lane's own point evaluation); it leaves the \
lane as a Candidate witness that the Sequential entry re-checks on the network's own \
f32 forward and the grouped entry on the trusted ONNX-Runtime forward of the original \
model, and only a point that passes there reaches the unchanged trusted-oracle gate. \
0 (the default) runs no attack. Absence defers to the typed key \
bab.branching.input_split.lowdim_leaf_attack_every, whose absence is 0. \
MoatRisk::High because a confirmed hit turns an unknown row into sat and a rejected \
one ends the lane for that clause.",
        provenance: Provenance::Unmeasured {
            why_ok: "default 0 and set by no shipped preset; attack-only (every candidate \
                     through the unchanged confirmers and trusted-oracle gate); the \
                     ny-propagate lowdim::bundle_tests suite pins that a hit is inside its \
                     leaf and violates on the lowering and that a proved clause is never \
                     given a candidate; with it 0 no attack runs",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_LEAF_ATTACK_EVERY_READERS,
    };

    /// `NY_LOWDIM_SHARED_WORKLIST` — the fold's families on one shared
    /// worklist (#lowdim-bundle (4)).
    pub LOWDIM_SHARED_WORKLIST = LeverDecl {
        name: "NY_LOWDIM_SHARED_WORKLIST",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#lowdim-bundle (4). Proves the #lowdim-fold families (nn4sys mscn dual clauses) on ONE \
shared worklist instead of one family after another: every clause keeps its own LIFO \
queue, each wave gives every live clause an equal share of the batch, the wave's \
leaves are bounded grouped by family (one forward per chunk per family, families in \
parallel), and the 200,000-domain per-clause cap is dropped (fairness replaces it). \
With lowdim_clip also on, single-objective families take stage 3's relaxed clip with \
each leaf's own planes. The per-leaf bound is the family core's; a clause is verified \
only when every one of its leaves closed. Queued boxes are capped by the lane's queue \
cap and a byte budget (1 GiB, at most a quarter of the live memory headroom). Exact \
\"1\" arms it and exact \"0\" disarms it; absence defers to the typed key \
bab.branching.input_split.lowdim_shared_worklist, whose absence is false. \
MoatRisk::High because it changes which clauses a deadline-bounded run proves.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-false and set by no shipped preset; soundness evidence is the \
                     ny-propagate lowdim::bundle_tests suite (the shared run agrees with \
                     one clause at a time and never verifies a sampled-violated threshold, \
                     with and without the clip); with it unset the sequential family loop \
                     runs exactly as before",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_SHARED_WORKLIST_READERS,
    };

    /// M8a (`#m8-input-split-batch`): the lowdim lane's wave BOUNDER.
    pub LOWDIM_BATCH_ENGINE = LeverDecl {
        name: "NY_LOWDIM_BATCH_ENGINE",
        kind: LeverKind::Enum(&["legacy", "bbe"]),
        default: DefaultSpec::Text("legacy"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
M8a (#m8-input-split-batch, ENGINE_PROGRAM section 2 M8). Chooses how the #lowdim-prover \
lane bounds each wave of input-split domains. \"legacy\" is the lane's own certified f64 \
forward-linear forms and backward objective pass on every domain (bit-identical to the \
lane before this key). \"bbe\" hands every wave to ONE call of the injected UNTRUSTED \
forward-linear search backend (ny-bbe's CPU f32 kernel, registered by ny-cli): its \
per-domain hints only CLAIM a leaf (the clause's own predicate on the hint shifted by the \
probe's tau) or pick a split dimension; a claimed leaf is re-bounded by the lane's own f64 \
forms over its own box and closes only on the unchanged strict lb > thr, and every other \
leaf is split into its exact halves. A program the backend does not mirror (refinement, \
joint quadratic, alpha, clip, products, global max), a missing backend, any refusal or \
panic, a failed probe and a refutation storm all run the legacy bounder. An admissible \
token arms it; absence defers to the typed key \
bab.branching.input_split.lowdim_batch_engine, whose absence is \"legacy\". \
MoatRisk::High because throughput and tree shape move a deadline-bounded verdict.",
        provenance: Provenance::Unmeasured {
            why_ok: "default \"legacy\" is the historical bounder, bit for bit (golden \
                     counters recorded at ce4f09170, pinned by ny-propagate \
                     lowdim::bbe_tests); set by no shipped preset (the router decides, \
                     user directive 2026-09-29); soundness evidence is the bbe_tests SAT \
                     moats (garbage backends never verify a planted violation), sampled \
                     containment of every certified bound and the one-ingest-gate source \
                     ratchet",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: LOWDIM_BATCH_ENGINE_READERS,
    };

    /// M8a (`#m8-input-split-batch`): the lowdim lane's MULTI-CLAUSE scheduler.
    pub INPUT_SPLIT_ENGINE = LeverDecl {
        name: "NY_INPUT_SPLIT_ENGINE",
        kind: LeverKind::Enum(&["legacy", "bbe"]),
        default: DefaultSpec::Text("legacy"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
M8a (#m8-input-split-batch, ENGINE_PROGRAM section 2 M8). The #lowdim-prover lane's \
multi-clause scheduler. \"legacy\" runs one lane tree per clause and the fold's sequential \
family loop (unless lowdim_shared_worklist). \"bbe\" proves same-box clauses (the \
grouped-disjunctive prepass: one lowering, one box, K clauses) on ONE disjunction tree - \
every leaf carries the set of clauses still open on it, one forward pass per node serves \
all of them, a clause is dropped from a leaf only on its own strict f64 close or its own \
clip planes emptying the leaf, and a clause is verified only when no leaf holds it any \
more - and runs the fold's families on the shared worklist (fold_shared = engine == bbe \
or lowdim_shared_worklist). The graph input-split drivers are untouched. An admissible \
token arms it; absence defers to the typed key bab.branching.input_split.engine, whose \
absence is \"legacy\". MoatRisk::High because it changes which clauses a \
deadline-bounded run proves.",
        provenance: Provenance::Unmeasured {
            why_ok: "default \"legacy\" is today's schedulers exactly (golden counters \
                     recorded at ce4f09170, pinned by ny-propagate lowdim::bbe_tests); set \
                     by no shipped preset (the router decides); soundness evidence is the \
                     disjunction tests (per-clause parity with one tree per clause, a \
                     clause dropped only on its own strict close, the clip hull keeps every \
                     open clause's violation, a sampled-violated clause never verifies)",
        },
        owner: LOWDIM_LANE_SCOPE,
        readers: INPUT_SPLIT_ENGINE_READERS,
    };
}

// M8a readers (kept after the registry so no existing declaration's
// neighbourhood moves).
const LOWDIM_BATCH_ENGINE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.lowdim_batch_engine into the frozen \
               scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's wave bounder from the frozen LeverSet (scored) or the \
               typed preset over the environment (interactive) into \
               BetaCrownConfig::input_split_lowdim_batch_engine, which lowdim_config \
               delivers to every lowdim entry as LowDimConfig::batch_engine",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_lowdim_batch_engine",
    },
];

const INPUT_SPLIT_ENGINE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "project bab.branching.input_split.engine into the frozen scored-run \
               LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LOWDIM_LANE_SCOPE,
        role: "resolve the lane's multi-clause scheduler from the frozen LeverSet (scored) \
               or the typed preset over the environment (interactive) into \
               BetaCrownConfig::input_split_engine, which lowdim_config delivers as \
               LowDimConfig::engine (the grouped prepass's disjunction tree, the \
               singleton route, and the fold's shared branch)",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_input_split_engine",
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, LeverValue, Source};

    /// Absent everything the twin is dark; the typed preset value arrives
    /// with `Source::Config` (the route a scored run takes, since the harness
    /// exports no `NY_*`); an exact `0` still kills it.
    #[test]
    fn interval_twin_is_dark_by_default_and_the_typed_key_arms_it() {
        let absent = read_over_config_with(&LOWDIM_INTERVAL_TWIN, |_| None, None)
            .expect("absence is admissible");
        assert_eq!(absent.value, LeverValue::Bool(false));
        assert_eq!(absent.source, Source::Default);

        let typed = read_over_config_with(
            &LOWDIM_INTERVAL_TWIN,
            |_| None,
            Some(LeverValue::Bool(true)),
        )
        .expect("the typed key is admissible");
        assert_eq!(typed.value, LeverValue::Bool(true));
        assert_eq!(typed.source, Source::Config);
        assert_ne!(typed.source, Source::LegacyEnv);

        let killed = read_over_config_with(
            &LOWDIM_INTERVAL_TWIN,
            |_| Some("0".to_owned()),
            Some(LeverValue::Bool(true)),
        )
        .expect("an exact 0 is an admissible disarm");
        assert_eq!(killed.value, LeverValue::Bool(false));
        assert_eq!(killed.source, Source::LegacyEnv);
    }

    /// #lsnc-lane stage 3's two Boolean keys: dark by default, armed by the
    /// typed key with `Source::Config` (never `LegacyEnv`), killed by an
    /// exact `0`.
    #[test]
    fn clip_and_parallel_are_dark_by_default_and_the_typed_key_arms_them() {
        for decl in [&LOWDIM_CLIP, &LOWDIM_PARALLEL] {
            let absent = read_over_config_with(decl, |_| None, None).expect("absence");
            assert_eq!(absent.value, LeverValue::Bool(false), "{}", decl.name);
            assert_eq!(absent.source, Source::Default, "{}", decl.name);
            let typed = read_over_config_with(decl, |_| None, Some(LeverValue::Bool(true)))
                .expect("the typed key is admissible");
            assert_eq!(typed.value, LeverValue::Bool(true), "{}", decl.name);
            assert_eq!(typed.source, Source::Config, "{}", decl.name);
            let killed =
                read_over_config_with(decl, |_| Some("0".to_owned()), Some(LeverValue::Bool(true)))
                    .expect("an exact 0 is an admissible disarm");
            assert_eq!(killed.value, LeverValue::Bool(false), "{}", decl.name);
            assert_eq!(killed.source, Source::LegacyEnv, "{}", decl.name);
        }
    }

    /// The split rule: absent it is the historical `widest`; every lane token
    /// arrives by the typed key with `Source::Config`; a token outside the
    /// four is inadmissible from the typed layer.
    #[test]
    fn the_split_key_defaults_to_widest_and_admits_exactly_the_lane_tokens() {
        let absent = read_over_config_with(&LOWDIM_SPLIT, |_| None, None).expect("absence");
        assert_eq!(absent.value, LeverValue::Text("widest".to_owned()));
        assert_eq!(absent.source, Source::Default);
        for token in ["widest", "gradient", "sb", "sbwr"] {
            let typed = read_over_config_with(
                &LOWDIM_SPLIT,
                |_| None,
                Some(LeverValue::Text(token.to_owned())),
            )
            .expect("a lane token is admissible");
            assert_eq!(typed.value, LeverValue::Text(token.to_owned()));
            assert_eq!(typed.source, Source::Config);
        }
        assert!(
            read_over_config_with(
                &LOWDIM_SPLIT,
                |_| None,
                Some(LeverValue::Text("sb_sum".to_owned())),
            )
            .is_err(),
            "a token the lane does not parse must not be admitted"
        );
    }

    /// #lowdim-bundle: the two Boolean keys are dark by default, armed by the
    /// typed key with `Source::Config` (never `LegacyEnv`), killed by an exact
    /// `0`; the two count keys default to 0, arrive by the typed key with
    /// `Source::Config`, and an exact `0` in the environment disarms them.
    #[test]
    fn the_bundle_keys_are_dark_by_default_and_the_typed_key_arms_them() {
        for decl in [&LOWDIM_VARYING_DIMS, &LOWDIM_SHARED_WORKLIST] {
            let absent = read_over_config_with(decl, |_| None, None).expect("absence");
            assert_eq!(absent.value, LeverValue::Bool(false), "{}", decl.name);
            assert_eq!(absent.source, Source::Default, "{}", decl.name);
            let typed = read_over_config_with(decl, |_| None, Some(LeverValue::Bool(true)))
                .expect("the typed key is admissible");
            assert_eq!(typed.value, LeverValue::Bool(true), "{}", decl.name);
            assert_eq!(typed.source, Source::Config, "{}", decl.name);
            let killed =
                read_over_config_with(decl, |_| Some("0".to_owned()), Some(LeverValue::Bool(true)))
                    .expect("an exact 0 is an admissible disarm");
            assert_eq!(killed.value, LeverValue::Bool(false), "{}", decl.name);
            assert_eq!(killed.source, Source::LegacyEnv, "{}", decl.name);
        }
        for decl in [&LOWDIM_ALPHA_ITERS, &LOWDIM_LEAF_ATTACK_EVERY] {
            let absent = read_over_config_with(decl, |_| None, None).expect("absence");
            assert_eq!(absent.value, LeverValue::U64(0), "{}", decl.name);
            assert_eq!(absent.source, Source::Default, "{}", decl.name);
            let typed = read_over_config_with(decl, |_| None, Some(LeverValue::U64(20)))
                .expect("the typed key is admissible");
            assert_eq!(typed.value, LeverValue::U64(20), "{}", decl.name);
            assert_eq!(typed.source, Source::Config, "{}", decl.name);
            assert_ne!(typed.source, Source::LegacyEnv, "{}", decl.name);
            let killed =
                read_over_config_with(decl, |_| Some("0".to_owned()), Some(LeverValue::U64(20)))
                    .expect("an exact 0 is an admissible disarm");
            assert_eq!(killed.value, LeverValue::U64(0), "{}", decl.name);
            assert_eq!(killed.source, Source::LegacyEnv, "{}", decl.name);
        }
    }

    /// M8a: the wave bounder is `legacy` absent everything; each admissible
    /// token arrives by the typed key with `Source::Config` (never
    /// `LegacyEnv`); a token outside the two is inadmissible.
    #[test]
    fn lowdim_batch_engine_is_dark_by_default_and_the_typed_key_arms_it() {
        let absent = read_over_config_with(&LOWDIM_BATCH_ENGINE, |_| None, None).expect("absence");
        assert_eq!(absent.value, LeverValue::Text("legacy".to_owned()));
        assert_eq!(absent.source, Source::Default);
        for token in ["legacy", "bbe"] {
            let typed = read_over_config_with(
                &LOWDIM_BATCH_ENGINE,
                |_| None,
                Some(LeverValue::Text(token.to_owned())),
            )
            .expect("a declared token is admissible");
            assert_eq!(typed.value, LeverValue::Text(token.to_owned()));
            assert_eq!(typed.source, Source::Config);
            assert_ne!(typed.source, Source::LegacyEnv);
        }
        assert!(
            read_over_config_with(
                &LOWDIM_BATCH_ENGINE,
                |_| None,
                Some(LeverValue::Text("cuda".to_owned())),
            )
            .is_err(),
            "a token the lane does not parse must not be admitted"
        );
    }

    /// M8a: the multi-clause scheduler, the same shape.
    #[test]
    fn input_split_engine_is_dark_by_default_and_the_typed_key_arms_it() {
        let absent = read_over_config_with(&INPUT_SPLIT_ENGINE, |_| None, None).expect("absence");
        assert_eq!(absent.value, LeverValue::Text("legacy".to_owned()));
        assert_eq!(absent.source, Source::Default);
        for token in ["legacy", "bbe"] {
            let typed = read_over_config_with(
                &INPUT_SPLIT_ENGINE,
                |_| None,
                Some(LeverValue::Text(token.to_owned())),
            )
            .expect("a declared token is admissible");
            assert_eq!(typed.value, LeverValue::Text(token.to_owned()));
            assert_eq!(typed.source, Source::Config);
            assert_ne!(typed.source, Source::LegacyEnv);
        }
        assert!(read_over_config_with(
            &INPUT_SPLIT_ENGINE,
            |_| None,
            Some(LeverValue::Text("graph".to_owned())),
        )
        .is_err());
    }

    /// The joint quadratic bound: dark by default, armed by the typed key
    /// with `Source::Config` (never `LegacyEnv`), killed by an exact `0`.
    #[test]
    fn joint_quadratic_is_dark_by_default_and_the_typed_key_arms_it() {
        let absent = read_over_config_with(&LOWDIM_JOINT_QUADRATIC, |_| None, None)
            .expect("absence is admissible");
        assert_eq!(absent.value, LeverValue::Bool(false));
        assert_eq!(absent.source, Source::Default);

        let typed = read_over_config_with(
            &LOWDIM_JOINT_QUADRATIC,
            |_| None,
            Some(LeverValue::Bool(true)),
        )
        .expect("the typed key is admissible");
        assert_eq!(typed.value, LeverValue::Bool(true));
        assert_eq!(typed.source, Source::Config);
        assert_ne!(typed.source, Source::LegacyEnv);

        let killed = read_over_config_with(
            &LOWDIM_JOINT_QUADRATIC,
            |_| Some("0".to_owned()),
            Some(LeverValue::Bool(true)),
        )
        .expect("an exact 0 is an admissible disarm");
        assert_eq!(killed.value, LeverValue::Bool(false));
        assert_eq!(killed.source, Source::LegacyEnv);
    }
}
