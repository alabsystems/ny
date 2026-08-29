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
pub mod cgan_stacked;
pub mod collection;
pub mod comprehensive_rows;
pub mod conv_ibp_blocked;
pub mod crown_ibp_budget;
pub mod cuda;
pub mod dark_probes;
pub mod diagnostics;
pub mod exact_vjp_pgd;
pub mod graph_mip;
pub mod kfsb;
pub mod margin_row;
pub mod measurement;
pub mod mo_alpha_heuristic;
pub mod mo_optimizer;
pub mod nn4sys_ctx_group;
pub mod onnx;
pub mod pollable_faer;
pub mod root_alpha;
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
    &collection::COLLECTION_LEVERS,
    &comprehensive_rows::COMPREHENSIVE_ROWS_LEVERS,
    &conv_ibp_blocked::CONV_IBP_BLOCKED_LEVERS,
    &crown_ibp_budget::CROWN_IBP_BUDGET_LEVERS,
    &cuda::CUDA_LEVERS,
    &dark_probes::DARK_PROBE_LEVERS,
    &diagnostics::DIAGNOSTIC_LEVERS,
    &exact_vjp_pgd::EXACT_VJP_PGD_LEVERS,
    &graph_mip::GRAPH_MIP_LEVERS,
    &kfsb::KFSB_LEVERS,
    &margin_row::MARGIN_ROW_LEVERS,
    &measurement::MEASUREMENT_LEVERS,
    &mo_alpha_heuristic::MO_ALPHA_HEURISTIC_LEVERS,
    &mo_optimizer::MO_OPTIMIZER_LEVERS,
    &nn4sys_ctx_group::NN4SYS_CTX_GROUP_LEVERS,
    &onnx::ONNX_LEVERS,
    &pollable_faer::POLLABLE_FAER_LEVERS,
    &root_alpha::ROOT_ALPHA_LEVERS,
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
            200,
            "Phase 0 declared two levers; batch B1 prep adds five telemetry declarations; \
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
             DomainAlphaRuntimePolicy"
        );
        assert!(all.get("NY_MO_ALPHA_HEURISTIC").is_some());
        assert!(all.get("NY_MO_BETA_SPSA").is_some());
        assert!(all.get("NY_MO_ALPHA_STEP").is_some());
        assert!(all.get("NY_MO_KFSB").is_some());
        assert!(all.get("NY_MO_KFSB_K").is_some());
        assert!(all.get("NY_MO_KFSB_REDUCE").is_some());
        assert!(all.get("NY_BRANCH_KFSB_CHILDSIM").is_some());
        assert!(all.get("NY_MARGIN_ROW_EPOCH_PARENT_TOPK").is_some());
        assert!(all.get("NY_JOINT_BILINEAR_QUAD").is_some());
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

    #[test]
    fn every_default_on_lever_has_admissible_evidence() {
        for decl in crate::all().all() {
            if decl.bucket == Bucket::DefaultOn {
                assert!(
                    matches!(
                        decl.provenance,
                        Provenance::ValueNeutral { .. }
                            | Provenance::Measured { .. }
                            | Provenance::Guard { .. }
                    ),
                    "{}: a shipped arm needs admissible evidence MERGE RECONCILIATION 2026-08-24: this total is the UNION of two sessions that promoted levers concurrently, and it is the third time this assertion has caught a merge — auto-merge keeps one side's integer while both sides' declarations land, so the number must be re-derived from the registry rather than taken from either branch. 125 -> 134.",
                    decl.name
                );
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
