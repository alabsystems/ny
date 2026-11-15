// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The declarations themselves, one module per subsystem.
//!
//! Adding a module here is the only way a lever becomes visible to
//! [`crate::all`], to the receipt, and (later) to `ny levers`.
//!
//! These Phase-0 modules make the declared surface enumerable and auditable.
//! [`root_alpha`] and [`sound_channel`] still name original ad-hoc reads.
//! [`collection`], [`graph_mip`], and [`telemetry`] centralize several compatibility parsers,
//! so their literal reads no longer appear in the narrow ratchet count, but
//! their runtime sites still use process-wide latches or live environment
//! reads. They are therefore preparatory plumbing, not completed per-run
//! [`crate::LeverSet`] migrations; that claim begins only when the same frozen
//! set is threaded to both the reader and the scored receipt.

pub mod bab_budget;
pub mod bab_diagnostics;
pub mod certified_root_cut;
pub mod cgan_stacked;
pub mod collection;
pub mod comprehensive_rows;
pub mod conv_ibp_blocked;
pub mod crown_ibp_budget;
pub mod crown_partial_seed;
pub mod cuda;
pub mod dark_probes;
pub mod diagnostics;
pub mod exact_vjp_pgd;
pub mod forward_linear;
pub mod graph_mip;
pub mod kernel_inputs;
pub mod kfsb;
pub mod lowdim_lane;
pub mod margin_row;
pub mod measurement;
pub mod mo_alpha_heuristic;
pub mod mo_cover;
pub mod mo_optimizer;
pub mod nn4sys_1d_cover;
pub mod nn4sys_ctx_group;
pub mod no_cell_enum;
pub mod onnx;
pub mod pollable_faer;
pub mod root_alpha;
pub mod router;
pub mod safenlp_cycle;
pub mod seq_weighted_budget;
pub mod sound_channel;
pub mod star;
pub mod telemetry;
pub mod vnncomp_entry;
pub mod wide_lane;

use crate::Registry;

static REGISTRIES: &[&Registry] = &[
    &bab_budget::BAB_BUDGET_LEVERS,
    &bab_diagnostics::BAB_DIAGNOSTIC_LEVERS,
    &cgan_stacked::CGAN_STACKED_LEVERS,
    &certified_root_cut::ROOT_CERTIFIED_CUT_LEVERS,
    &collection::COLLECTION_LEVERS,
    &comprehensive_rows::COMPREHENSIVE_ROWS_LEVERS,
    &conv_ibp_blocked::CONV_IBP_BLOCKED_LEVERS,
    &crown_ibp_budget::CROWN_IBP_BUDGET_LEVERS,
    &crown_partial_seed::CROWN_PARTIAL_SEED_LEVERS,
    &cuda::CUDA_LEVERS,
    &dark_probes::DARK_PROBE_LEVERS,
    &diagnostics::DIAGNOSTIC_LEVERS,
    &exact_vjp_pgd::EXACT_VJP_PGD_LEVERS,
    &forward_linear::FORWARD_LINEAR_LEVERS,
    &graph_mip::GRAPH_MIP_LEVERS,
    &kfsb::KFSB_LEVERS,
    &kernel_inputs::KERNEL_INPUT_LEVERS,
    &lowdim_lane::LOWDIM_LANE_LEVERS,
    &margin_row::MARGIN_ROW_LEVERS,
    &measurement::MEASUREMENT_LEVERS,
    &mo_alpha_heuristic::MO_ALPHA_HEURISTIC_LEVERS,
    &mo_cover::MO_COVER_LEVERS,
    &mo_optimizer::MO_OPTIMIZER_LEVERS,
    &nn4sys_1d_cover::NN4SYS_1D_COVER_LEVERS,
    &nn4sys_ctx_group::NN4SYS_CTX_GROUP_LEVERS,
    &no_cell_enum::NO_CELL_ENUM_LEVERS,
    &onnx::ONNX_LEVERS,
    &pollable_faer::POLLABLE_FAER_LEVERS,
    &root_alpha::ROOT_ALPHA_LEVERS,
    &router::ROUTER_LEVERS,
    &safenlp_cycle::SAFENLP_CYCLE_LEVERS,
    &seq_weighted_budget::SEQ_WEIGHTED_BUDGET_LEVERS,
    &sound_channel::SOUND_CHANNEL_LEVERS,
    &star::STAR_LEVERS,
    &telemetry::TELEMETRY_LEVERS,
    &vnncomp_entry::VNNCOMP_ENTRY_LEVERS,
    &wide_lane::WIDE_LANE_LEVERS,
];

/// Every module registry in the crate.
pub fn registries() -> &'static [&'static Registry] {
    REGISTRIES
}

#[cfg(test)]
mod tests {
    use crate::{Bucket, DefaultSpec, MoatRisk, Provenance};

    #[test]
    fn registry_merges() {
        let all = crate::all();
        assert_eq!(
            all.len(),
            304,
            "Three existing kernel/calibration inputs are now governed, with unchanged defaults and raw compatibility predicates. The owner-directed same-source root cut adds one governed default-on declaration. Phase 0 declared two levers; batch B1 prep adds five telemetry declarations; \
             true-gradient GPU replay adds one governed legacy-armed selector; collection \
             policy prep adds three centrally parsed compatibility overrides; terminal-Softmax \
             work adds one governed transform gate and two test-only corpus selectors; the \
             beta/GPU diagnostics add one shared declared probe; wide-lane routing adds three \
             governed High-risk selectors; the authoritative margin-row GPU seam adds one \
             verdict-carrying sound-channel gate, its DOMAIN-BATCHING sibling adds one more, \
             Graph-MIP leaf SAT publication adds one governed legacy-armed rollback, and the \
             experimental alpha-envelope gradient adds one governed default-dark selector, \
             exact-VJP PGD adds eleven typed per-run controls, and \
             resident segment-composition diagnostics add one governed probe; the post-823 dark \
             probes add three diagnostic selectors and one armed A/B shape switch; the certified \
             batched-clip umbrella adds one governed opt-in; legacy convolution-Patches \
             diagnostics add one centrally declared text lever; six dark-star measurement \
             controls are governed without hiding their readers behind a helper, and the \
             discrete-CUDA transport opt-in is a declared two-package selector; the shared \
             full-measurement expansion is centrally declared for its three test readers, and \
             the one-target wide demanded root fold adds one governed opt-in, and the \
             comprehensive CPU-parallel root sweep adds one atomic default-dark opt-in, and the \
             comprehensive sound-GPU root sweep adds one separate atomic default-dark opt-in, and \
             the phase-resident root sweep adds one deferred unified default-dark opt-in, and the BNN sign-space falsification lane adds one governed default-dark High-risk probe; the advisory kFSB simulation budget adds one MEASURED closed-interval fraction, the first `F64ClosedTrimmed` axis; \
             the #comprehensive-rows-probe measurement program adds five High-risk dark \
             overrides, and two presence-parser gates are declared as such rather than \
             rounded to exact-\"1\" booleans; the finite-Patches authority repair adds \
             one High-risk gate that now SHIPS ARMED on measured evidence, with exact \
             \"0\" as its kill switch; and the \
             sign-space minimal-move lever adds one dark Low-risk A/B shape switch, whose \
             null result is explained by one more presence-gated row-generation trace, and \
             whose SIDEWAYS half is A/B'd by one more dark Low-risk enum selecting the \
             realizability LP's pixel-column bounds; and the hunt for the sites that \
             densify a conv carrier adds one dark Low-risk print-only Patches->Dense \
             carrier trace; and the STE-PGD \
             falsification lane over that same admitted fragment adds one dark \
             High-risk env-only attack gate; and the #bab-floor root-window arbitration adds \
             three dark High-risk closed-interval fractions -- one arming reservation for \
             branch-and-bound plus the two shares (root objective pass, root-alpha bootstrap \
             ceiling) that have to be named for that reservation to be reachable; and the pre-Softmax attack objective adds one \
             dark High-risk scoring gate; and the ported falsification portfolio adds one \
             dark High-risk strategy gate; and the cross-lane marginal-value \
             scheduler over the attack slice adds one dark High-risk env-only \
             admission; and LAYER A of the per-instance lane budget allocator -- the \
             multiple-choice knapsack that commits every attack-slice cap jointly and \
             up front -- adds one more dark High-risk env-only admission beside it; \
             and the margin-row joint forward re-bake (#rebake) adds \
             one dark High-risk gate plus its round cap and wall slice; and the cgan \
             budget-reclaim set (#cgan-reclaim) adds one dark High-risk master arm \
             for its three retention sites; and the SafeNLP full-cycle A/B \
             (#safenlp-cycle) adds one dark High-risk master arm composing the \
             conditional MIP handoff, the AY margin reframe and the serial solve; \
             and the cgan near-miss survivor alpha polish (#cgan-polish) adds one \
             dark High-risk master arm for the eager lane's knife-edge re-bound; \
             and the ratchet conversion of two committed raw reads declares the \
             nn4sys context-grouped disjunctive coverage gate (#ctx-group) and the \
             pollable chunked-faer bounded-CROWN restoration (#pollable-faer), the \
             latter kept on its legacy truthy Text parser rather than rounded to an \
             exact-\"1\" Boolean; and the wgpu CROWN-backward wide-kernel \
             diagnostics add one governed probe, declared so its eight open-coded \
             readers stop accepting only the exact spelling 1; and the GPU \
             backward's B2 perf slot adds the discard-only per-step shadow \
             profiler (#margin-row-gpu-eft, NY_MARGIN_ROW_GPU_PROF); and its B4/B5/B6 \
             multi-site landing adds the per-site shadow SELECTOR that keeps the \
             measurement phase at one site per cycle (NY_MARGIN_ROW_GPU_SHADOW_SITES, \
             defaulting to the root-only shape the M2 protocol was banked on); and \
             the upfront probe-first pass adds one dark High-risk ordering gate; and \
             the sequential CROWN-IBP cost-weighted budget (#seq-weighted-budget) \
             declares its OPT-OUT, kept on the legacy truthy parser because the \
             shipped arm is engaged-by-absence; and the fixed attack-slice frame \
             (#lane-budget-frame) adds one more dark High-risk env-only admission -- the \
             THIRD and weakest cap authority, which commits every lane to exactly today's \
             own window but decides it ONCE, so a declining lane's seconds reach \
             branch-and-bound instead of the next lane that reads the live clock; and the \
             #scored-wall-clamp adds one dark High-risk gate that hands branch-and-bound \
             an ABSOLUTE stop derived from the scored budget, separate from the effort \
             horizon its timeout shapes, so a lane may keep inflating its deadline to hold \
             a non-monotonic search on a converging trajectory without buying that \
             trajectory with wall-clock past the official per-instance budget; and \
             the blocked-pollable certified Conv2d IBP kernel (#pollable-conv-ibp) adds \
             the registry's first ValueNeutral DefaultOn selector -- three arms that are \
             byte-identical by construction, so the shipped arm is qualified by a named \
             parity test rather than by a verdict A/B; and two BaB diagnostics that \
             arrived as raw reads are declared with their DISAGREEING parsers \
             recorded (#bab-diagnostics: NY_ACASXU_PROF is presence-armed, \
             NY_BAB_CHILD_TRACE is exact-\"1\"); and #closure-adaptive -- the lane \
             selecting its OWN closure mechanisms at runtime from the child-closure \
             fraction it measures about itself -- adds FIVE: one three-state override of \
             the preset-chosen policy (NY_MARGIN_ROW_CLOSURE_ADAPTIVE, Unset default so \
             absence defers to `margin_row.closure_adaptive`) plus the four numbers the \
             policy is defined by -- the escalation threshold taken from \
             alpha-beta-CROWN's measured win/loss bands, the warm-up, the observation \
             window, and the ladder cap that exists so a measurement can isolate one \
             mechanism; and the ratchet reconciliation of the six inherited raw readers of \
             NY_RESNET_ERR_CONCRETIZE declares that gate as one High-risk exact-\"1\" \
             Bool, DELIBERATELY UNCACHED because this file's own differential tests \
             flip it mid-process with ScopedEnvVar, plus its PER-RELU sibling \
             NY_RESNET_ERR_CONCRETIZE_FINE declared separately because the two arm \
             different granularities and the fine one carries an extra admission; and the branching reduceop anti-parity adds one dark Low-risk RANKER-ONLY switch (NY_MARGIN_ROW_BRANCH_REDUCE_MAX) -- the registry's first non-probe Low, because selecting WHICH candidate to split cannot manufacture a proof, only reach a different set of domains within the budget; and the interval-valued \
             frozen-constant carrier (#const-interval) adds one dark High-risk BOUND \
             gate -- the first declared lever that changes what a frozen CONSTANT is \
             rather than which search runs, admitting an outward-rounded enclosure \
             where the bit-exact fold declines instead of failing the node closed to \
             an unbounded skip; and the joint bilinear relaxation \
             (#joint-bilinear) adds one dark High-risk BOUND gate beside it -- \
             the first that changes how a PRODUCT of two functions of the same \
             input is relaxed rather than what a constant is, replacing the \
             factor-wise McCormick decorrelation of `ReduceSum(A * B)` with one \
             quadratic form maximized jointly over the input box; and the \
             margin-row incremental parent-frame epoch adds one dark High-risk \
             top-K selector with an explicit finite nesting budget; and the \
             multi-objective kFSB lane adds its canonical governed master plus \
             its candidate-count, reduce-policy, and proof-carrying certificate-reuse \
             controls plus the separately governed legacy compound child-simulation switch; \
             and current-box multi-objective alpha refresh adds one governed, \
             default-dark measured candidate; and multi-objective optimizer \
             governance adds the measured default-on beta SPSA selector plus \
             the measured-negative default-dark joint alpha step; and the \
             #bi-adaptive-chunk pass sizer adds one governed default-dark \
             measured per-pass wall target beside the backward-interm budget, \
             shipped at 0 (= the historical fixed grain) because its A/B lost \
             a banked wall-hugger to affine per-pass overhead; and the root-alpha \
             optimizer adds governed Adam step, CVaR steering-width, and \
             published-frame guard controls; and the \
             margin-row y-pack LRU plus parallel-frontier width are declared \
             at their measured July defaults with typed search delivery; and \
             the pre-existing per-node CROWN-IBP floor/cap preset controls are \
             finally declared and exposed to governed optimization; and the VNN-COMP \
             scored entry freezes the upfront-attack master, ordinary/automatic wall caps, \
             and budget fraction in the same per-run LeverSet used by its receipt; and the \
             post-BaB fixed reserve adds one governed default-zero scored-budget axis, while its \
             independent wrapper admission is governed through the same frozen receipt; and the \
             root-joint family declares eleven verdict-affecting admission, budget, optimizer, \
             extraction, and selection controls plus one print-only diagnostic probe; and the \
             optional CPU compute-engine presence handle adds one governed legacy-armed \
             High-risk kill switch \
             and the #tableau-f32 f32 STORAGE lane of the root tableau walk adds \
             one dark High-risk gate (NY_MARGIN_ROW_TABLEAU_F32), delivered by \
             margin_row.tableau_f32, its ad-hoc parser narrowed to exact-\"1\"; and the \
             pre-existing root-compute precision switch is now separately governed as \
             NY_MARGIN_ROW_ROOT_F32 while preserving its legacy trimmed truthy grammar; and \
             the measured margin-row head/trunk branch shortlist pair adds two governed \
             positive trimmed-usize overrides with typed search delivery; and the outer \
             margin-row scheduler adds eight governed controls whose distinct override, \
             Unicode-ownership, and trimmed-exact-one grammars preserve the existing typed \
             preset layering while one immutable policy replaces repeated process reads; and \
             the four objective-chunk width/abort/growth/wave controls are centrally parsed \
             and excluded as coupled unsafe measurement-only axes; and the adjacent \
             Patches flat-bias kernel plus honest-provenance guard now give execution and \
             truncated-cache identity the same effective typed decisions; and the \
             #margin-row-domain-alpha SCHEDULE adds its three dark High-risk knobs \
             (depth floor, depth stride, closure-gap window) deciding WHICH domains pay \
             for the measured-NULL per-domain ascent, carried resolved on the frozen \
             DomainAlphaRuntimePolicy; and the #domain-stack cross-domain \
             scoring lane adds one dark High-risk SCHEDULE gate \
             (NY_MARGIN_ROW_DOMAIN_STACK), delivered by margin_row.domain_stack \
             -- promoted out of a raw env read because run_instance.sh exports \
             no NY_*, so the mechanism could not be armed on a scored binary at \
             all and its wall-clock effect was unmeasurable through the route \
             that competes; it scores every domain in a frontier batch in ONE \
             wide certified pass instead of one pass per domain, and its \
             reductions are column-local, so a stacked score is bit-identical to \
             the independent one and the gate selects a SCHEDULE rather than a \
             relaxation; and #interm-alpha adds one dark High-risk \
             per-START-NODE prefix-alpha candidate dose inside the backward-intermediate \
             phase (NY_MARGIN_ROW_INTERM_ALPHA, delivered by margin_row.interm_alpha), \
             every candidate box intersected shrink-only into the published box, so the \
             dose prices wall against the BI budget rather than admitting a looser bound \
             ; and STATEFUL Clip-and-Verify (#clip-stateful) adds one \
             dark High-risk gate (NY_MARGIN_ROW_CLIP_STATEFUL) that makes the margin-row \
             clip's intermediate tightening ACCUMULATE per domain and be inherited parent \
             -> child -- the alpha-beta-CROWN 2024->2025 cifar100 diff, which added \
             clip_interm_domain and had to REMOVE interm_transfer:False in the same two \
             lines, because clipping makes per-domain intermediate bounds mutate and a \
             config that refuses to transfer them is only valid while they never change; and \
             #branch-parent-lift adds one dark High-risk Debug gate (NY_MARGIN_ROW_PARENT_LIFT) \
             that folds the branching scores the expansion ALREADY paid for into the child \
             bound floor: `score_candidates` certifies an (l_i, r_i) pair for every \
             shortlisted candidate and the expansion discards ~31 of them, though each \
             still certifies something about the PARENT; `false` keeps the historical discard; \
             and the #flat-ladder-depth selective Hadamard refinement cap adds one governed \
             default-zero SCHEDULE selector (NY_FLAT_SURROGATE_MAX_BLOCKS, delivered by \
             attack.flat_surrogate_max_blocks) — 0 keeps the split-all ladder byte-identical, \
             cap>=2 freezes low-effect blocks so deeper levels stay affordable inside a fixed \
             official budget; and post-BaB restart diversity (#postbab-restart-diversity) adds one \
             governed default-zero High-risk restart count (NY_POSTBAB_RESTARTS, delivered by \
             attack.post_bab_restarts) — the 2026-08-31 soundnessbench window probe measured the \
             stage-0 attack stack CONVERGENT, bit-identical fixed points at 11 s and 49.6 s windows, \
             so the open axis is basin DIVERSITY: re-enter the converged alternation from \
             deterministically row-seeded uniform/corner/orthogonal-jitter basins, every candidate \
             still passing the unchanged trusted-ORT gate; and the #flat-surrogate-transfer \
             surrogate->real transfer stage adds one governed default-false Debug gate \
             (NY_FLAT_SURROGATE_TRANSFER, delivered by attack.flat_surrogate_transfer) — false \
             keeps the surrogate search byte-identical, true walks the real output's quantizer \
             staircase from the surrogate optimum through the unchanged acceptance stack; \
             and the cct engine-wiring pair (#patches-engine-refusal-containment / \
             #fast-f32-prewarm-all-hosts, reports/measured-2026/cct_bound_gap_2026-08-31) adds \
             two governed default-false High-risk gates: NY_PATCHES_ENGINE_REFUSAL_CONTAINMENT \
             (delivered by bab.patches_engine_refusal_containment) treats a POLICY-refusing GEMM \
             engine at the Conv2d Patches compose seam as an absent accelerator so the walk \
             keeps its byte-identical CPU scatter in patches, and NY_FAST_F32_PREWARM \
             (delivered by bab.fast_f32_prewarm) materializes the registered fast-f32 GEMM \
             factory on the vnncomp route for every build so finite-deadline consumers stop \
             failing closed to the quarantined proof adapter, plus one governed declaration of the \
             seam's pre-existing legacy-truthy kill switch NY_PATCHES_GPU, whose live reads \
             remain counted reader-frontier debt; and #metaroom-mip-governor adds TWO \
             governed default-dark SCHEDULE levers for the two MIP lanes measured spending \
             110 s of a 210 s metaroom budget to decide nothing: \
             NY_MIP_LEAF_UNDECIDED_STREAK_CAP (delivered by \
             bab.mip.leaf_undecided_streak_cap, 0 = today's no-cap behaviour) stops the \
             Graph-MIP leaf oracle after N consecutive undecided consults, and \
             NY_MIP_ESCALATE_SKIP_CERTIFIED (delivered by \
             bab.mip.escalation_skips_certified_clauses) seeds the disjunctive escalation's \
             pending set from the UNCERTIFIED clauses only -- both MoatRisk::Low because \
             declining a leaf is what its Undecided already does and a skipped clause \
             carries the same certified infeasibility bound ny publishes unsat on; and \
             #nn4sys-1d-cover promotes the \
             exact one-axis complete-cover prepass as THREE coupled declarations rather \
             than one -- NY_NN4SYS_1D_COMPLETE_COVER (delivered by bab.nn4sys_1d_cover) \
             is the master, NY_NN4SYS_1D_COVER_DEPTH (bab.nn4sys_1d_cover_depth) is not \
             optional beside it because the pass is ALL-OR-NOTHING and the master alone \
             would deliver the reader's DEFAULT_DEPTH of 8 into it, and \
             NY_NN4SYS_1D_COVER_SLICE (bab.nn4sys_1d_cover_slice) is a closed-interval \
             FRACTION of the caller's remaining deadline rather than the legacy \
             millisecond constant, because nn4sys budgets span 20s to 800s in one \
             benchmark and an unfenced prepass can consume the whole screen slice and \
             return nothing; and \
             #clip-head adds one dark High-risk gate \
             (NY_MARGIN_ROW_CLIP_HEAD, delivered by margin_row.clip_head) that builds \
             Clip-and-Verify halfspaces from a domain\'s HEAD sign fixes: halfspaces_for_splits \
             reads LayerGates::clip_rows and the root gates carry TRUNK relus only, so a head fix \
             has always clamped ONE interval and tightened nothing else, while on the traced \
             alpha-beta-CROWN rows the head layer is where 10 of 12 splits land and where \
             clip_interm_domain earns most of its per-domain improvement; it needs no clip_rows \
             retention because its certified lines are columns of the y-pack the eval already \
             holds, which is what lets a category that retains nothing arm it; and the \
             crossing-only sweep allocation adds one dark High-risk selector that \
             spends the chunk row budget only where a crossing row remains; and \
             #dense-sparse-seed adds one High-risk closed-unit-interval threshold \
             (NY_CROWN_PARTIAL_DENSE_MIN_STABLE, delivered by \
             bab.crown_partial_dense_min_stable) whose 0.9 default is the historical \
             literal of the sequential CROWN-IBP partial walk's Dense unstable-rows \
             seed, so absence is byte-identical, and whose armed value applies only to \
             ReLU-successor targets; and #lsnc-attack-twin adds one dark High-risk post-BaB \
             attack gate (NY_ATTACK_POINT_FOLDED_TWIN, delivered by attack.point_folded_twin) \
             that runs exact-gradient PGD on an ONNX-Runtime point-folded ATTACK twin when \
             the exact loader leaves the real net outside the point-VJP fragment, every \
             candidate still passing the unchanged trusted-ORT gate; and #lsnc-lane adds \
             one dark High-risk loader gate (delivered by \
             bab.branching.input_split.lowdim_interval_twin) that builds the lowdim \
             lane's PRIVATE interval twin of a model whose exact load left OpaqueSkip \
             nodes, the global interval carrier staying EnvOnly and dark; and #lsnc-lane \
             stage 2 adds one dark High-risk bound gate (delivered by \
             bab.branching.input_split.lowdim_joint_quadratic) that bounds the lowdim \
             lane's ReduceSum(Mul) sites by the joint quadratic bound; and #lsnc-lane \
             stage 3 adds three dark High-risk search gates (delivered by \
             bab.branching.input_split.lowdim_split / lowdim_clip / lowdim_parallel): the \
             lane's split rule, its relaxed clip with each box's own planes, and \
             deciding its domains inside the rayon tasks; and #vit-margin-alpha \
             promotes the env-only NY_ROOT_ALPHA_MARGIN_GRADIENT child gate to a declared \
             typed key (bab.alpha_crown.root_alpha_margin_gradient) and adds \
             NY_ROOT_ALPHA_MATMUL (bab.alpha_crown.matmul_alpha), both dark High-risk \
             selectors of the certified C-row margin ascent; and #mr-errlane \
             (margin-row throughput plan B3) adds one dark High-risk Debug \
             ENUM selector (NY_MARGIN_ROW_ERR_ENVELOPE, delivered by \
             margin_row.err_envelope: off | envelope | per_channel | audit) choosing how \
             the backward engine carries its CERTIFIED coefficient-error lane -- per \
             column (today, byte-identical), or as one envelope per row or per channel \
             that dominates it by monotone induction, which only loosens the certified \
             penalty; `audit` publishes `off` and shadows both envelopes to measure \
             their looseness; and #graph-free-candidates (#smart-turn-x86-harvest) adds \
             one dark Low-risk Debug INSTRUMENT (NY_GRAPH_FREE_CANDIDATE_SIDEFILE, \
             delivered by attack.graph_free_candidate_sidefile) that writes every \
             counterexample candidate the graph-free lane offers to its unchanged \
             acceptance gates, with the gate that decided it, to a per-row JSONL for \
             offline replay on a checker-parity x86 configuration -- report-only, \
             nothing it writes is read back; and \
             #mr-gpu-eval adds one dark High-risk Debug gate \
             (NY_MARGIN_ROW_GPU_CONV, delivered by margin_row.gpu_conv) that runs the \
             certified pass's conv backward products as full-contraction f64 GEMMs on the \
             process-global sound f64 engine, certified by the lane's unchanged \
             order-independent charges; and #frozen-dual-state adds one High-risk \
             Debug gate (NY_MARGIN_ROW_FROZEN_DUAL_BAB, delivered by \
             margin_row.frozen_dual_bab) arming V1 stage A, the certified frozen \
             alpha-zonotope query state of the frozen dual-ascent BaB, which stage A only \
             builds, reports and drops; and #frozen-dual-bab (V1 stage B) makes that gate arm \
             the certified search tier over the state and adds its two dark High-risk Debug \
             knobs (NY_MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP and \
             NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS, delivered by \
             margin_row.frozen_dual_bab_node_cap / margin_row.frozen_dual_bab_secs); and #lowdim-bundle adds four dark High-risk \
             #lowdim-prover keys (delivered by bab.branching.input_split.lowdim_varying_dims \
             / lowdim_alpha_iters / lowdim_leaf_attack_every / lowdim_shared_worklist): \
             varying-dims admission with rigorous Sigmoid planes, per-leaf boundary alpha \
             on the objective pass, an attack-only worst-leaf PGD whose hits the trusted \
             confirmers decide, and one shared worklist across the fold's families; and \
             #root-certify-once adds one dark High-risk Debug gate and three dark \
             High-risk Debug knobs (NY_MARGIN_ROW_ROOT_CERTIFY_ONCE, _ITERS, _REFRESH, \
             _SECS, delivered by margin_row.root_certify_once / _iters / _refresh / \
             _secs): the Phase-0.5 root alpha ascent run UNTRUSTED on a Parity scratch \
             root, then ONE certified evaluation kept only if it strictly beats the \
             certified incumbent. \
             and #bi-gpu adds one dark High-risk Debug gate (NY_MARGIN_ROW_BI_GPU, \
             delivered by margin_row.bi_gpu) that runs the backward-interm and \
             joint-refresh prefix pass pairs on #mr-gpu-eval's certified sound f64 conv \
             route through a scoped, per-pair-accounted, RSS-admitted route; \
             and #bi-patch adds one dark High-risk Debug gate and two dark High-risk \
             Debug knobs (NY_MARGIN_ROW_BI_PATCH, _RATIO, _MB, delivered by \
             margin_row.bi_patch / _ratio / _mb) that run the backward-interm and \
             joint-refresh prefix pass pairs as per-column receptive-field walks. \
             And #bbe-joint-root (ENGINE_PROGRAM M5a) adds one dark High-risk Debug \
             ENUM selector (NY_MARGIN_ROW_ROOT_SEARCH, delivered by \
             margin_row.root_search: ascent | joint_f32) choosing which UNTRUSTED search \
             proposes the #root-certify-once slopes -- today's ascent (bit-identical) or \
             the injected ny-bbe joint-alpha root search -- whose slopes still reach the \
             certified root only through the existing ingest gate. \
             And #breakpoint-dual adds one High-risk Debug knob \
             (NY_MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS, delivered by \
             margin_row.frozen_dual_refine_steps, default 0) whose untrusted breakpoint \
             proposals are retained only after strict improvement in the unchanged \
             certified frozen-dual node evaluator; its tier cover ledger is unchanged. \
             The parent and four refinement steps are enabled by the user-directed \
             CIFAR100/TinyImageNet category presets in bounded automatic mode, while \
             generic defaults remain false/zero and competition gains remain unmeasured. \
             And #fd-shared-state (ENGINE_PROGRAM M3) adds one dark High-risk Debug ENUM \
             selector (NY_MARGIN_ROW_FROZEN_DUAL_BAB_STATE, delivered by \
             margin_row.frozen_dual_bab_state: per_class | shared) choosing whether the \
             frozen-dual tier builds each class's state inside its share (bit-identical) or \
             ONE shared trunk of rows before the shares plus a one-column tail per class. \
             And #root-box-source (ENGINE_PROGRAM M5b) adds one dark High-risk Debug \
             ENUM selector (NY_MARGIN_ROW_ROOT_BOX_SOURCE, delivered by \
             margin_row.root_box_source: tableau | patch_backward) choosing how every \
             root walk builds its trunk boxes -- the forward tableau (bit-identical) or \
             a certified IBP reference plus a certified patch-backward pass over the \
             reference-unstable neurons, intersected shrink-only. \
             And #wide-fd-search (ENGINE_PROGRAM M4) adds one dark High-risk Debug ENUM \
             selector (NY_MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH, delivered by \
             margin_row.frozen_dual_bab_search: f64 | cpu_f32 | cuda | auto) choosing which \
             search scores the frozen-dual tier's nodes -- its own f64 search (bit-identical) \
             or an injected UNTRUSTED backend whose proposals the tier certifies in f64 and \
             splits otherwise. \
             And #wave-beta-bab (ENGINE_PROGRAM M6) adds one dark High-risk Debug ENUM \
             selector (NY_MARGIN_ROW_TREE_MODE, delivered by margin_row.tree_mode: legacy | \
             wave | wave_cuda) choosing the margin-row outer frame's tree engine -- today's \
             best-first tree (bit-identical) or waves of injected UNTRUSTED beta-CROWN search \
             (on the CPU, or on the device with the CPU as fall-through) whose claims a \
             private pool certifies with the lane's own evaluator -- and two dark \
             High-risk Debug knobs (NY_MARGIN_ROW_TREE_WAVE_BATCH, \
             NY_MARGIN_ROW_TREE_WAVE_CERT_THREADS, delivered by margin_row.tree_wave_batch \
             / margin_row.tree_wave_cert_threads); and #router-r1 adds the two \
             SafeToSearch shadow-mode selectors NY_ROUTER_MODE / NY_OPTIMIZER_MODE \
             (router.mode / optimizer.mode). \
             And #m8-input-split-batch (ENGINE_PROGRAM M8a) adds two dark High-risk Debug \
             ENUM selectors of the lowdim lane (delivered by \
             bab.branching.input_split.lowdim_batch_engine and \
             bab.branching.input_split.engine: legacy | bbe) -- the wave bounder \
             (the lane's own f64 forms, bit-identical, or an injected UNTRUSTED \
             forward-linear search whose claims the lane certifies) and the multi-clause \
             scheduler (one tree per clause, or one disjunction tree per shared box and \
             the fold's shared worklist); and the #vit-v3-bankable multi-objective \
             cover certification adds one dark High-risk enum (NY_MO_COVER_CERTIFY: \
             off | ledger | replay), downgrade-only in both armed modes. \
             And #router-r2 (ROUTER_PROGRAM R2) adds the seven acting-rule ablation \
             switches NY_ROUTER_RULE_ADMISSION / _CONTRACTS / _CAP_AUTHORITY / _TERMINAL / \
             _HAND_ON / _PUBLISH_GRACE / _DERIVED_CAPS (router.rules.*: BoolOverride, default \
             true = participates only when router.mode is on; UNSAFE_AXES, never searched). \
             And #router-r3 (ROUTER_PROGRAM R3) adds the four O-STOP optimizer-rule ablation \
             switches for rule.governor, rule.ia_yield, rule.attack_defer and \
             rule.root_admission (optimizer.rules.*: BoolOverride, default true = participates \
             only when optimizer.mode is on; UNSAFE_AXES, never searched). \
             And #router-r4 (ROUTER_PROGRAM R4) adds acting set 3's two ablation switches \
             NY_ROUTER_RULE_LEASE (CR-L) and NY_ROUTER_RULE_FAR_HAND_ON (CR-F) (router.rules.*: \
             BoolOverride, default true = participates only when router.mode is on; UNSAFE_AXES, \
             never searched). \
             And #router-r5 (ROUTER_PROGRAM R5) adds acting set 4's two ablation switches \
             NY_ROUTER_RULE_ROUTE (CR-RS) and NY_ROUTER_RULE_GAP_CONTRACT (CR-G) \
             (router.rules.*: BoolOverride, default true = participates only when router.mode \
             is on; UNSAFE_AXES, never searched). \
             And #router-r6 (ROUTER_PROGRAM R6) adds the four O-TRIAL optimizer-rule \
             ablation switches NY_OPTIMIZER_RULE_TRIAL_WIDTH / _TRIAL_LR_BACKOFF / \
             _TRIAL_SPLIT_RESTART / _TRIAL_ATTACK_HALVING (optimizer.rules.trial_*: \
             BoolOverride, default true = participates only when optimizer.mode is on; \
             UNSAFE_AXES, never searched). \
             And #router-r7 (ROUTER_PROGRAM R7) adds NY_OPTIMIZER_PRIORS (optimizer.priors: \
             Bool, default false = the router reads no offline prior; UNSAFE_AXES, never \
             searched). \
             And #head-groups (d0f718885) adds one High-risk Bool \
             (NY_MARGIN_ROW_HEAD_GROUPS, delivered by margin_row.head_groups) arming one \
             bounded complete head-phase cover pass on the outer margin-row root. \
             The user-directed directional-cut route adds one High-risk Bool \
             (NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS, margin_row.head_directional_cuts); \
             both this stage and its parent groups are now default true. \
             And #head-phase-cover adds one dark High-risk Debug count \
             (NY_MARGIN_ROW_HEAD_PHASE_COVER, delivered by margin_row.head_phase_cover) \
             for a complete Source phase cover of at most seven selected heads. \
             And #fl-alpha-composition declares the four forward-linear reference \
             switches its tests scrub (forward_linear, +4): the shared \
             NY_NO_FORWARD_LINEAR_REF and NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF kill \
             switches (both tracked LEGACY-ARMED-UNQUALIFIED), the dark \
             NY_FORWARD_LINEAR_SEQ_CONV_REF research gate, and NY_FORWARD_LINEAR_F32, the \
             legacy env route of the model.forward_linear_f32_seam key -- all High-risk \
             Debug exact-\"1\" Booleans whose raw readers stay counted reader debt. \
             And the environment-reader cleanup declares three test-only corpus/request \
             paths and the legacy classwise research schedule without promoting it. \
             The count is re-derived from the declarations at each merge."
        );
        assert!(all
            .get(crate::decls::margin_row::MARGIN_ROW_HEAD_GROUPS.name)
            .is_some());
        assert!(all
            .get(crate::decls::margin_row::MARGIN_ROW_HEAD_DIRECTIONAL_CUTS.name)
            .is_some());
        assert!(all
            .get(crate::decls::margin_row::MARGIN_ROW_HEAD_PHASE_COVER.name)
            .is_some());
        assert!(all
            .get(crate::decls::dark_probes::GRAPH_FREE_CANDIDATE_SIDEFILE.name)
            .is_some());
        assert!(all.get("NY_SWEEP_CROSSING_TARGETS_ONLY").is_some());
        assert!(all.get("NY_MIP_LEAF_UNDECIDED_STREAK_CAP").is_some());
        assert!(all.get("NY_MIP_ESCALATE_SKIP_CERTIFIED").is_some());
        assert!(all.get("NY_MO_ALPHA_HEURISTIC").is_some());
        assert!(all.get("NY_MO_BETA_SPSA").is_some());
        assert!(all.get("NY_MO_ALPHA_STEP").is_some());
        assert!(all.get("NY_MO_KFSB").is_some());
        assert!(all.get("NY_MO_KFSB_K").is_some());
        assert!(all.get("NY_MO_KFSB_REDUCE").is_some());
        assert!(all.get("NY_BRANCH_KFSB_CHILDSIM").is_some());
        assert!(all.get("NY_MARGIN_ROW_EPOCH_PARENT_TOPK").is_some());
        assert!(all.get("NY_JOINT_BILINEAR_QUAD").is_some());
        assert!(all.get("NY_MARGIN_ROW_CLIP_STATEFUL").is_some());
        assert!(all.get("NY_MARGIN_ROW_PARENT_LIFT").is_some());
        assert!(all
            .get(crate::decls::margin_row::MARGIN_ROW_ERR_ENVELOPE.name)
            .is_some());
        assert!(all.get("NY_CONST_MATMUL_INTERVAL").is_some());
        assert!(all.get("NY_SAFENLP_MIP_CYCLE").is_some());
        assert!(all.get("NY_NN4SYS_CTX_GROUP").is_some());
        assert!(all.get("NY_POLLABLE_FAER").is_some());
        assert!(all.get("NY_CONV_IBP_BLOCKED").is_some());
        assert!(all.get("NY_CGAN_RECLAIM").is_some());
        assert!(all.get("NY_CGAN_POLISH").is_some());
        assert!(all.get("NY_UPFRONT_ATTACK_PROBE_FIRST").is_some());
        assert!(all.get("NY_LANE_VALUE_SCHEDULER").is_some());
        assert!(all.get("NY_LANE_BUDGET_ALLOCATOR").is_some());
        assert!(all.get("NY_BNN_SIGN_SPACE_MINIMAL_MOVE").is_some());
        assert!(all.get("NY_BNN_SIGN_SPACE_TRUST_REGION").is_some());
        assert!(all.get("NY_BNN_SIGN_SPACE_TRACE").is_some());
        assert!(all.get("NY_EFT_ERR").is_some());
        assert!(all.get("NY_ALPHA_ZERO_YIELD_FRAC").is_some());
        assert!(all.get("NY_ALPHA_ENVELOPE_GRAD").is_some());
        assert!(all.get("NY_PHASE_TELEMETRY").is_some());
        assert!(all.get("NY_MIP_TRACE").is_some());
        assert!(all.get("NY_ITER0_PARITY_TRACE").is_some());
        assert!(all.get("NY_PATCHES_CARRIER_TRACE").is_some());
        assert!(all.get("NY_MARGIN_ROW_PROFILE").is_some());
        assert!(all.get("NY_GPU_MEM_TRACE").is_some());
        assert!(all.get("NY_BETA_GPU_PROBE").is_some());
        assert!(all.get("NY_SEG_PROBE").is_some());
        assert!(all.get("NY_TRUE_GRAD_GPU_REPLAY").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA").is_some());
        assert!(all
            .get("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT")
            .is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_SECS").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_ITERS").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_LR").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN").is_some());
        assert!(all.get("NY_BN_GPU_EXTRACT").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_LAYERS").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_PROBE").is_some());
        assert!(all.get("NY_ROOT_JOINT_MIN_REMAINING_SECS").is_some());
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM").is_some());
        assert!(all.get("NY_CPU_COMPUTE_ENGINE").is_some());
        assert!(all.get("NY_MARGIN_ROW_ROOT_F32").is_some());
        assert!(all.get("NY_NO_WALK_RECORD_ADMISSION").is_some());
        assert!(all.get("NY_CLIP_HOST_MEAN_LA").is_some());
        assert!(all.get("NY_CLIP_INTERM_CERTIFIED").is_some());
        assert!(all.get("NY_ROOT_WIDE_DEMANDED_INTERM_CROWN").is_some());
        assert!(all.get("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN").is_some());
        assert!(all.get("NY_ROOT_PHASE_RESIDENT_CROWN").is_some());
        assert!(all.get("NY_ROOT_CPU_PARALLEL_INTERM_CROWN").is_some());
        assert!(all.get("NY_STRIP_TERMINAL_SOFTMAX").is_some());
        assert!(all.get("NY_BENCH_ROOT").is_some());
        assert!(all.get("NY_BENCH_ROOT_2026").is_some());
        assert!(all.get("NY_BAB_RESNET_WIDE").is_some());
        assert!(all.get("NY_BAB_RESNET_WIDE_SUBGROUP").is_some());
        assert!(all.get("NY_MO_GPU_CHUNK_DEADLINE").is_some());
        assert!(all.get("NY_KFSB_SIM_SHARE").is_some());
        assert!(all.get("NY_MARGIN_ROW_GPU").is_some());
        assert!(all.get("NY_MARGIN_ROW_GPU_BATCH").is_some());
        assert!(all.get("NY_MARGIN_ROW_REBAKE").is_some());
        assert!(all.get("NY_MARGIN_ROW_REBAKE_ROUNDS").is_some());
        assert!(all.get("NY_MARGIN_ROW_REBAKE_SECS").is_some());
        assert!(all.get("NY_MARGIN_ROW_LRU").is_some());
        assert!(all.get("NY_MARGIN_ROW_PARALLEL").is_some());
        assert!(all.get("NY_GRAPH_MIP_LEAF_SAT").is_some());
        assert!(all.get("NY_ENVELOPE_XSTAR_PROBE").is_some());
        assert!(all.get("NY_ENVELOPE_RESCALE_PROBE").is_some());
        assert!(all.get("NY_INPUT_SPLIT_PROBE").is_some());
        assert!(all.get("NY_INPUT_SPLIT_NESTED_DEADLINE").is_some());
        assert!(all.get("NY_CONV_PATCHES_DEBUG").is_some());
        assert!(all.get("NY_STAR_DARK_SECONDS").is_some());
        assert!(all.get("NY_STAR_DARK_MAX_STARS").is_some());
        assert!(all.get("NY_STAR_DARK_MAX_DEPTH").is_some());
        assert!(all.get("NY_STAR_DARK_DUAL_ITERS").is_some());
        assert!(all.get("NY_STAR_DARK_INPUT_SPLIT").is_some());
        assert!(all.get("NY_STAR_DARK_EXACT_BELOW").is_some());
        assert!(all.get("NY_CUDA_DISCRETE_MODE").is_some());
        assert!(all.get("NY_FULL_MEASUREMENTS").is_some());
        assert!(all.get("NY_BAB_RESERVE_FRAC").is_some());
        assert!(all.get("NY_ROOT_SPEC_FRAC").is_some());
        assert!(all.get("NY_ROOT_ALPHA_FRAC").is_some());
        assert!(all.get("NY_UPFRONT_ATTACK").is_some());
        assert!(all.get("NY_UPFRONT_ATTACK_CAP").is_some());
        assert!(all.get("NY_UPFRONT_ATTACK_AUTO_CAP").is_some());
        assert!(all.get("NY_UPFRONT_ATTACK_FRAC").is_some());
        assert!(all.get("NY_POSTBAB_RESERVE_SECS").is_some());
        assert!(all.get("NY_POSTBAB_ATTACK").is_some());
        assert!(all.get("NY_CROWN_OBJ_CHUNK").is_some());
        assert!(all.get("NY_NO_CHUNK_ABORT").is_some());
        assert!(all.get("NY_NO_CHUNK_GROW").is_some());
        assert!(all.get("NY_NO_CHUNK_WAVE_PAR").is_some());
        assert!(all.get("NY_PATCHES_DEADLINE_FLAT_BIAS").is_some());
        assert!(all.get("NY_CROWN_HONEST_PROVENANCE").is_some());
        assert!(all
            .get(crate::decls::forward_linear::NO_FORWARD_LINEAR_REF.name)
            .is_some());
        assert!(all
            .get(crate::decls::forward_linear::FORWARD_LINEAR_F32.name)
            .is_some());
    }

    #[test]
    fn unmeasured_levers_are_never_default_on() {
        // The moat rule in its smallest form. The full evidence suite arrives
        // with Phase 3; this much can be enforced from day one.
        for decl in crate::all().all() {
            if matches!(decl.provenance, Provenance::Unmeasured { .. }) {
                assert_ne!(
                    decl.bucket,
                    Bucket::DefaultOn,
                    "{}: an armed default requires evidence",
                    decl.name
                );
            }
        }
    }

    const OWNER_DIRECTED_DEFAULTS: &[&str] = &[
        "NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS",
        "NY_MARGIN_ROW_HEAD_GROUPS",
        "NY_ROOT_CERTIFIED_CUT",
    ];

    fn admissible_owner_direction(name: &str, provenance: Provenance) -> bool {
        let Provenance::OwnerDirected {
            directive,
            limitations,
        } = provenance
        else {
            return false;
        };
        OWNER_DIRECTED_DEFAULTS.contains(&name)
            && !directive.trim().is_empty()
            && !limitations.trim().is_empty()
    }

    #[test]
    fn every_default_on_lever_has_admissible_evidence_or_scoped_owner_direction() {
        for decl in crate::all().all() {
            if decl.bucket == Bucket::DefaultOn {
                assert!(
                    matches!(
                        decl.provenance,
                        Provenance::ValueNeutral { .. }
                            | Provenance::Measured { .. }
                            | Provenance::Guard { .. }
                    ) || admissible_owner_direction(decl.name, decl.provenance),
                    "{}: a shipped arm needs admissible evidence or exact scoped user direction",
                    decl.name
                );
            }
        }
    }

    #[test]
    fn owner_directed_defaults_have_exact_scope_and_honest_receipt_tags() {
        let all = crate::all();
        let names: Vec<_> = all
            .all()
            .iter()
            .filter(|decl| matches!(decl.provenance, Provenance::OwnerDirected { .. }))
            .map(|decl| decl.name)
            .collect();
        assert_eq!(names, OWNER_DIRECTED_DEFAULTS);
        let levers = crate::LeverSet::freeze_with(all, |_| None);
        let receipt = levers.receipt();
        for &name in OWNER_DIRECTED_DEFAULTS {
            let decl = all.get(name).unwrap();
            assert!(admissible_owner_direction(name, decl.provenance));
            assert_eq!(decl.bucket, Bucket::DefaultOn);
            assert_eq!(decl.default, DefaultSpec::Bool(true));
            assert_eq!(decl.moat, MoatRisk::High);
            let row = receipt["levers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["name"].as_str() == Some(name))
                .unwrap();
            assert_eq!(row["provenance"], "owner_directed");
            assert_eq!(row["value"], true);
            assert!(!admissible_owner_direction(
                "NY_UNAUTHORIZED_DEFAULT",
                decl.provenance
            ));
            for (directive, limitations) in [("", "limits"), ("  ", "limits"), ("request", "")] {
                assert!(!admissible_owner_direction(
                    name,
                    Provenance::OwnerDirected {
                        directive,
                        limitations
                    }
                ));
            }
        }
    }

    #[test]
    fn legacy_armed_unqualified_defaults_stay_on_an_exact_tracking_list() {
        // Phase 0's compact schema does not yet carry the target design's
        // DefaultStatus field. Do not let an already-engaged negative
        // kill-switch default masquerade as an ordinary dark Debug lever in
        // the meantime. This list is exact and should shrink as declarations
        // gain evidence or the target schema lands.
        // NY_CLIP_HOST_MEAN_LA is not on this list because its positive gate
        // ships OFF; it remains Unmeasured until a retained current-path A/B
        // qualifies a promotion.
        let tracked = [
            "NY_BAB_RESNET_WIDE",
            "NY_CPU_COMPUTE_ENGINE",
            "NY_GRAPH_MIP_LEAF_SAT",
            "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
            "NY_NO_FORWARD_LINEAR_REF",
            "NY_NO_WALK_RECORD_ADMISSION",
            "NY_TRUE_GRAD_GPU_REPLAY",
        ];
        let observed: Vec<&str> = crate::all()
            .all()
            .iter()
            .filter(|decl| decl.doc.contains("LEGACY-ARMED-UNQUALIFIED"))
            .map(|decl| decl.name)
            .collect();
        assert_eq!(observed, tracked, "legacy-armed tracking list drifted");
        for name in tracked {
            let decl = crate::all().get(name).expect("tracked declaration");
            assert!(
                decl.doc.contains("LEGACY-ARMED-UNQUALIFIED"),
                "{name}: tracking marker must remain visible"
            );
            assert!(matches!(decl.provenance, Provenance::Unmeasured { .. }));
            assert_ne!(decl.bucket, Bucket::DefaultOn);
        }
        let no_walk = crate::all()
            .get("NY_NO_WALK_RECORD_ADMISSION")
            .expect("tracked no-walk declaration");
        assert!(matches!(no_walk.default, DefaultSpec::Bool(false)));
        assert_eq!(no_walk.bucket, Bucket::Debug);
        assert_eq!(no_walk.moat, MoatRisk::High);

        let graph_mip_leaf_sat = crate::all()
            .get("NY_GRAPH_MIP_LEAF_SAT")
            .expect("tracked Graph-MIP leaf SAT declaration");
        assert!(matches!(
            graph_mip_leaf_sat.default,
            DefaultSpec::Bool(true)
        ));
        assert_eq!(graph_mip_leaf_sat.bucket, Bucket::Debug);
        assert_eq!(graph_mip_leaf_sat.moat, MoatRisk::High);

        let bab_resnet_wide = crate::all()
            .get("NY_BAB_RESNET_WIDE")
            .expect("tracked wide-kernel declaration");
        assert!(matches!(bab_resnet_wide.default, DefaultSpec::Bool(true)));
        assert_eq!(bab_resnet_wide.bucket, Bucket::Debug);
        assert_eq!(bab_resnet_wide.moat, MoatRisk::High);

        let cpu_compute_engine = crate::all()
            .get("NY_CPU_COMPUTE_ENGINE")
            .expect("tracked optional CPU compute-engine selector");
        assert!(matches!(
            cpu_compute_engine.default,
            DefaultSpec::Bool(true)
        ));
        assert_eq!(cpu_compute_engine.bucket, Bucket::Debug);
        assert_eq!(cpu_compute_engine.moat, MoatRisk::High);

        // NY_CLIP_HOST_MEAN_LA stays dark and explicitly Unmeasured; its parser
        // and default are pinned by `host_mean_la_opt_in_preserves_default_off_...`.
        let clip_host = crate::all()
            .get("NY_CLIP_HOST_MEAN_LA")
            .expect("host mean-lA declaration");
        assert!(matches!(clip_host.default, DefaultSpec::Bool(false)));
        assert!(matches!(
            clip_host.provenance,
            Provenance::Unmeasured { .. }
        ));

        let true_grad = crate::all()
            .get("NY_TRUE_GRAD_GPU_REPLAY")
            .expect("tracked true-gradient declaration");
        assert!(matches!(true_grad.default, DefaultSpec::Bool(true)));
        assert_eq!(true_grad.bucket, Bucket::Debug);
        assert_eq!(true_grad.moat, MoatRisk::High);
    }

    #[test]
    fn alpha_zero_yield_decl_records_the_measured_unshipped_candidate() {
        let decl = crate::all()
            .get("NY_ALPHA_ZERO_YIELD_FRAC")
            .expect("alpha-zero-yield declaration");
        assert!(matches!(decl.default, DefaultSpec::Unset));
        assert_eq!(decl.bucket, Bucket::Debug);
        assert_eq!(decl.moat, MoatRisk::High);
        assert_eq!(
            decl.readers.len(),
            6,
            "validation, config delivery, receipt projection, env capture, resolver, and consumer"
        );
        let Provenance::Measured {
            commit,
            artifact,
            delta,
            ..
        } = decl.provenance
        else {
            panic!("the measured alpha-zero-yield candidate needs measured provenance");
        };
        assert_eq!(commit, "a5bc1e73");
        assert!(artifact.contains("section 8"));
        assert!(delta.contains("15/15 timeout rows"));
        assert!(decl.doc.contains("promotion was retracted"));
        assert!(decl.doc.contains("all 200"));
        assert!(decl.doc.contains("DefaultSpec::Unset"));
    }

    #[test]
    fn multi_package_readers_are_declared_as_such() {
        for decl in crate::all().all() {
            if decl.reader_packages().len() > 1 {
                assert!(
                    decl.is_multi_reader(),
                    "{}: cross-package sharing must be declared",
                    decl.name
                );
            }
        }
    }
}
