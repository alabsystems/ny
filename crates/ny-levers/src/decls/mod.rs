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

pub mod collection;
pub mod comprehensive_rows;
pub mod cuda;
pub mod dark_probes;
pub mod diagnostics;
pub mod graph_mip;
pub mod measurement;
pub mod onnx;
pub mod root_alpha;
pub mod sound_channel;
pub mod star;
pub mod telemetry;
pub mod wide_lane;

use crate::Registry;

static REGISTRIES: &[&Registry] = &[
    &collection::COLLECTION_LEVERS,
    &comprehensive_rows::COMPREHENSIVE_ROWS_LEVERS,
    &cuda::CUDA_LEVERS,
    &dark_probes::DARK_PROBE_LEVERS,
    &diagnostics::DIAGNOSTIC_LEVERS,
    &graph_mip::GRAPH_MIP_LEVERS,
    &measurement::MEASUREMENT_LEVERS,
    &onnx::ONNX_LEVERS,
    &root_alpha::ROOT_ALPHA_LEVERS,
    &sound_channel::SOUND_CHANNEL_LEVERS,
    &star::STAR_LEVERS,
    &telemetry::TELEMETRY_LEVERS,
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
            57,
            "Phase 0 declared two levers; batch B1 prep adds five telemetry declarations; \
             true-gradient GPU replay adds one governed legacy-armed selector; collection \
             policy prep adds three centrally parsed compatibility overrides; terminal-Softmax \
             work adds one governed transform gate and two test-only corpus selectors; the \
             beta/GPU diagnostics add one shared declared probe; wide-lane routing adds three \
             governed High-risk selectors; the authoritative margin-row GPU seam adds one \
             verdict-carrying sound-channel gate, its DOMAIN-BATCHING sibling adds one more, \
             Graph-MIP leaf SAT publication adds one governed legacy-armed rollback, and the \
             experimental alpha-envelope gradient adds one governed default-dark selector, and \
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
             one dark High-risk gate whose armed arm is measured verdict-neutral; and the \
             sign-space minimal-move lever adds one dark Low-risk A/B shape switch, whose \
             null result is explained by one more presence-gated row-generation trace, and \
             whose SIDEWAYS half is A/B'd by one more dark Low-risk enum selecting the \
             realizability LP's pixel-column bounds; and the hunt for the sites that \
             densify a conv carrier adds one dark Low-risk print-only Patches->Dense \
             carrier trace"
        );
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
        assert!(all.get("NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM").is_some());
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
                    "{}: a shipped arm needs admissible evidence",
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
