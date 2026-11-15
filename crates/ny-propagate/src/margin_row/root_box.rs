// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #root-box-source (ENGINE_PROGRAM M5b): a CERTIFIED patch-backward box
//! source for the margin-row root, selected by the typed key
//! `margin_row.root_box_source: tableau | patch_backward`.
//!
//! The declaration default `tableau` never enters this module: the root build
//! (`root.rs::build_retaining_inner`) compares one enum and runs the unchanged
//! forward-tableau walk, bit for bit and silently.
//!
//! # Why
//!
//! The root's trunk boxes come from the dense forward DeepPoly (M, D) tableau:
//! `naug = n_in + 1` columns (9409 on tinyimagenet) pushed through every trunk
//! op, for every neuron, stable or not. That is 20.4-22.0 s of a 36 s root on
//! tiny and the whole 36 s `root_gate_build` on cifar100 resnet_large, which
//! never completes (ENGINE_PROGRAM §2 M5, GW P2). abc builds the same boxes
//! from a cheap interval reference plus a sparse CROWN backward over only the
//! neurons that reference leaves unstable, in patches. This module is that
//! recipe on ny's certified engines.
//!
//! # What
//!
//! The walk visits the trunk ops in order, up to the last trunk ReLU, and
//! carries ONE certified interval (IBP) REFERENCE box per live tensor, built by
//! `#mr-ibp-seed`'s reviewed certified builders (`ibp_seed::conv_box`,
//! `affine_box`, `add_box`; every rounding and fold charge, outward), starting
//! from the declared input box. At trunk ReLU layer `li` (op `k`):
//!
//! 1. the reference box of its pre-activation is taken;
//! 2. CANDIDATES are the neurons UNSTABLE in that certified reference
//!    (`l < 0 < u`); every other neuron is certified stable and keeps its
//!    reference box (its gate is the exact fixed line, so a looser box costs
//!    nothing downstream);
//! 3. for the candidates, one fused Lower+Upper certified backward prefix pass
//!    per column chunk through the FROZEN gates of layers `0..li`, concretized
//!    over the root box: `#bi-patch`'s per-column receptive-field walk
//!    (`PatchPlan::prefix_bounds`, the dense arm's certified arithmetic
//!    restricted to its support; reviewed sound on main), and the unchanged
//!    dense `BackwardEngine::run_prefix` pair wherever the plan refuses the
//!    layer or cannot express a chunk. Every pass runs on the CPU
//!    (`GpuConvRoute::Off`): the device never certifies (R2);
//! 4. the published box is the SHRINK-ONLY intersection of the two certified
//!    enclosures (reference and backward);
//! 5. the gates are derived from the PUBLISHED box exactly as the tableau walk
//!    derives them (`gates_from_box`, `repair_upper_lines`, the `#rebake`
//!    alpha seed on unstable neurons only, the baked epoch splits, `ms =
//!    max(alpha, s)`, `unst` from the published box);
//! 6. the ReLU's output reference is `[max(l, 0), max(u, 0)]` of that FINAL
//!    box, so the next layer's reference is one interval hop from the tightest
//!    certified box available (a residual skip likewise carries a ReLU
//!    output's final box).
//!
//! # Soundness
//!
//! Claim: every published trunk box encloses the exact real-arithmetic
//! pre-activation of the stored network for every input in the root box (and,
//! on a Tier-2 epoch rebuild, every input in the baked split subdomain; for
//! every member of the certified fold ball where a folded op carries one).
//! Induction over the trunk layers:
//!
//! * the reference box is certified by `ibp_seed`'s builders (sign-safe
//!   centre/radius charges, the fold ball, outward closes), from certified
//!   input boxes: the declared box, and ReLU outputs of certified
//!   pre-activation boxes (`relu` is monotone, `max` exact);
//! * the backward bound of a candidate is a certified outward bound over the
//!   root box because the prefix gates are valid DeepPoly relaxations on
//!   layers `0..li`'s published boxes (induction hypothesis), and the engine's
//!   certified error lane and outward concretize cover every rounding
//!   (`engine.rs::run_prefix`; the patch walk is the dense pass restricted to
//!   its support with `E_patch <= E_dense`, `patch.rs` SOUNDNESS);
//! * the intersection of two certified enclosures is a certified enclosure,
//!   and `max` / `min` are exact. Two certified enclosures of a non-empty
//!   range cannot cross, so a crossing is a DEFECT SIGNAL (or an empty baked
//!   split subdomain): it is never an emptiness certificate. One crossing
//!   DECLINES the whole build and the unchanged tableau build runs instead
//!   (fail closed);
//! * a candidate whose certified pass failed closed (the non-finite firewall,
//!   an engine refusal) keeps its reference box, which is certified;
//! * the unstable set of every layer is read from the PUBLISHED certified box,
//!   never from a hint; the only untrusted input is the `#rebake` alpha seed,
//!   which is admitted exactly as the tableau walk admits it (finite, clamped
//!   to `[0, 1]`, unstable neurons only, `ms` re-derived), so any seed is a
//!   valid lower line.
//!
//! The historical root leak (#root-rounding: a coefficient-column rounding
//! charge that moved a wall INWARD at negative inputs) cannot arise here: the
//! reference charges are non-negative radii, and the backward engine's error
//! lane is an elementwise coefficient ERROR bound concretized as `E . xabs`
//! with `xabs = |mid| + rad`, which is sign-safe by construction. Both are
//! pinned in exact rationals on the asymmetric-box, active-ReLU cancelling
//! nets that hid the leak (`root_box_tests`), with a mutation pin that runs
//! the same passes without the error lane and must escape.
//!
//! # What the tableau produced that this source does not — REFUSED, never dropped
//!
//! Three tableau products have no patch-backward counterpart. Building the
//! source anyway would silently switch each one OFF on a run whose preset asked
//! for it (and all three are armed on cifar100/tiny today), so an armed build
//! that REQUESTS any of them is REFUSED before the source runs
//! ([`PatchBackwardConflicts`]): the unchanged tableau build runs with the SAME
//! arguments — bit for bit the `tableau` default's root — and the build prints
//! `[root-box] refused reason=<a+b> source=tableau ...`. Preset validation
//! refuses the same combination up front (`preset::apply`), so a yaml cannot
//! ship it; this runtime check covers every other route (environment,
//! compatibility cells, epoch rebuilds).
//!
//! * `clip_rows` — `LayerGates::clip_rows` (#clip-rows, the input-relative
//!   sandwich rows every Clip-and-Verify consumer and the M6 wave's clip rung
//!   need) are tableau rows;
//! * `retain` — Tier-0 retained rows (#epoch-bab, ranker-only) are tableau
//!   rows;
//! * `root_f32` / `tableau_f32` — both select tableau kernels.
//!
//! Composing them for real (clip rows and retained rows from the backward
//! pass's own coefficient rows; an f32 reference walk) is new TCB and needs its
//! own containment pins; until then the combination is refused, fail closed.
//!
//! Unrequested, the source's published root carries no clip rows and no
//! retained rows (`None`), exactly as a tableau build that was not asked for
//! them.
//!
//! * `#backward-interm`'s baseline pass IS this source's pass (the same
//!   certified prefix pair over the same frozen prefix, for EVERY candidate
//!   rather than the widest `topk`), so the phase is subsumed and not run;
//!   its `#interm-alpha` candidates are not run either. Its receipt reports
//!   the configured policy with zero work.
//! * `#mr-ibp-seed` is subsumed (its box IS the reference, restarted at every
//!   layer from the final box, residual trunks included).
//!
//! # Cost and memory
//!
//! Per layer, `2 x candidates` pass columns through the prefix, where the
//! tableau pushes `naug` columns through every op. Patch chunks run
//! concurrently on the caller's rayon pool, their width and concurrency sized
//! so that all concurrent chunks together stay inside `WORKING_SET_BYTES`
//! (narrow legacy nets may use a quarter of observed memory headroom). Newly
//! admitted wide nets freeze the full existing ceiling before construction;
//! insufficient or unobservable headroom refuses their whole route, rather
//! than silently omitting a candidate's tightening. Dense-fallback chunks run
//! one at a time. Every column's arithmetic is column-local (the fused pair
//! and the conv kernels are bit-identical across column counts), and every
//! multi-column chunk that fails or that the plan cannot express is retried
//! column by column before anything runs dense, so chunk widths and thread
//! counts never change a published bit.
//!
//! # Observable (every armed Outward build, fixed schema)
//!
//! ```text
//! [root-box] layer=<li> op=<k> neurons=<n> ref_unst=<a> cols=<c> patch_cols=<p> dense_cols=<d> unavailable=<x> plan=<patch|dense(<why>)|-> tightened=<t> stabilized=<s> unst=<u> secs=<f>
//! [root-box] source=patch_backward complete=<0|1> layers=<L> ref_unst=<A> cols=<C> patch_cols=<P> dense_cols=<D> unavailable=<X> plan_refused=<R> stabilized=<S> unst=<U> crossed=0 retain=<-|empty> clip_rows=none bi=<-|subsumed> ibp_seed=<-|subsumed> root_f32=<-|inert> certify_s=<f>
//! [root-box] source=tableau declined=<parity|reference:t<id>|crossed:layer<li>x<n>> patch_s=<f> tableau_ok=<0|1> certify_s=<f>
//! [root-box] refused reason=<clip_rows|retain|root_f32|tableau_f32 joined by +> source=tableau tableau_ok=<0|1> certify_s=<f>
//! ```
//!
//! On a built line `retain=` and `root_f32=` always read `-`: the `empty` and
//! `inert` values are unreachable through the root build, which refuses those
//! requests (the fields stay for schema stability).
//!
//! `certify_s` is the whole build's wall (reference, passes, gates): every
//! second of it is certification. An armed run without a `[root-box]
//! source=` line is VOID; a `refused` line means the combination was never
//! measured (the tableau ran), which is VOID for M5b, not a negative.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{NyError, Result};
use rayon::prelude::*;

use super::backward_interm::BackwardIntermExecutionEvidence;
use super::engine::{BackwardEngine, LaneDir, Seed};
use super::gpu_conv::GpuConvRoute;
use super::ibp_seed::{self, IBox};
use super::net::{TwinNet, TwinOp};
use super::patch::{PatchPlan, DEFAULT_RATIO};
use super::root::{
    gates_from_box, repair_upper_lines, LayerGates, RetainCfg, RetainedLayer, RetainedRows,
    RootGates, RootPrecisionExecutionEvidence,
};
use super::rounding::{next_up, RoundMode};

/// The existing dense entry ceiling admits two f64 value/deviation matrices.
/// Source selection uses that same byte ceiling; it never raises it.
pub const ROOT_SOURCE_BYTES_CAP: usize = super::spec::MAX_ROOT_TABLEAU_ENTRIES * 16;

/// Actual root-source products, with conservative capacity/temporary pricing.
/// Tree candidate width is supplied separately because it depends on surviving
/// objectives; a root-only estimate cannot authorize a full BaB allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootSourceCensus {
    pub tableau_entries: usize,
    pub compiled_and_plan_bytes: usize,
    pub gate_and_reference_bytes: usize,
    pub dense_backward_column_bytes: usize,
    pub consumer_column_bytes: usize,
    pub head_columns: usize,
    pub head_pair_bytes: usize,
    pub root_peak_bytes: usize,
}

impl RootSourceCensus {
    pub fn consumer_peak_bytes(self, columns: usize) -> Option<usize> {
        // Dense backward coefficient/error lanes and temporaries over all
        // prefix tensors, plus the retained lower/upper pair. Deliberately
        // conservative: consumer admission may refuse useful wide batches.
        self.consumer_column_bytes
            .checked_mul(columns)?
            .checked_add(self.compiled_and_plan_bytes)?
            .checked_add(self.gate_and_reference_bytes)?
            .checked_add(self.head_pair_bytes)
    }
}

/// Worst-case dense scoring and y-pack residency for the requested tree. The
/// classwise schedule is sequential and retirement only narrows columns, so
/// pricing all original objectives also covers those two routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootConsumerCensus {
    pub columns: usize,
    pub workers: usize,
    pub cache_bytes: usize,
    pub peak_bytes: usize,
}

pub fn consumer_census(
    census: RootSourceCensus,
    policy: &super::MarginRowRuntimePolicy,
    objectives: usize,
    workers: usize,
    lru_cap: usize,
) -> Option<RootConsumerCensus> {
    let workers = workers.max(1);
    // Candidate pairs plus the incumbent/main margin pass that can remain
    // live during scoring. Contextual widths cannot exceed this reservation.
    let candidates = policy
        .width
        .k_head
        .max(policy.width.contextual_k_head)
        .checked_add(policy.width.k_trunk.max(policy.width.contextual_k_trunk))?;
    let columns = candidates
        .checked_mul(2)?
        .checked_add(2)?
        .checked_mul(objectives)?;
    let cache_bytes = census
        .head_pair_bytes
        .checked_mul(lru_cap)?
        .checked_mul(workers)?;
    let peak_bytes = census
        .consumer_peak_bytes(columns)?
        .checked_mul(workers)?
        .checked_add(cache_bytes)?
        .checked_add(objectives.checked_mul(workers)?.checked_mul(2048)?)?;
    Some(RootConsumerCensus {
        columns,
        workers,
        cache_bytes,
        peak_bytes,
    })
}

#[cfg(test)]
thread_local! {
    static WIDE_MEMORY_FOR_TEST: std::cell::Cell<Option<crate::network::crown_memory::ProcessMemoryEnvelope>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn with_wide_memory_for_test<T>(
    envelope: crate::network::crown_memory::ProcessMemoryEnvelope,
    f: impl FnOnce() -> T,
) -> T {
    struct Reset(Option<crate::network::crown_memory::ProcessMemoryEnvelope>);
    impl Drop for Reset {
        fn drop(&mut self) {
            WIDE_MEMORY_FOR_TEST.with(|cell| cell.set(self.0));
        }
    }
    let old = WIDE_MEMORY_FOR_TEST.with(|cell| cell.replace(Some(envelope)));
    let _reset = Reset(old);
    f()
}

/// Wide-source admission freezes the existing full working-set ceiling.
/// Host memory may refuse the whole route, never silently change its boxes.
pub fn wide_memory_admission(required_bytes: usize) -> Result<()> {
    use crate::network::crown_memory::{process_memory_envelope, ProcessMemoryEnvelope};
    if required_bytes > ROOT_SOURCE_BYTES_CAP {
        return Err(NyError::InvalidSpec(format!(
            "wide patch source: whole-route memory refusal required_bytes={required_bytes} cap={ROOT_SOURCE_BYTES_CAP}"
        )));
    }
    #[cfg(test)]
    let envelope = WIDE_MEMORY_FOR_TEST
        .with(|cell| cell.get())
        .unwrap_or_else(process_memory_envelope);
    #[cfg(not(test))]
    let envelope = process_memory_envelope();
    match envelope {
        ProcessMemoryEnvelope::Bounded { headroom_bytes }
            if (required_bytes as u128) > u128::from(headroom_bytes) => Err(NyError::InvalidSpec(format!(
                "wide patch source: whole-route memory refusal required_bytes={required_bytes} headroom_bytes={headroom_bytes} working_set_bytes={WORKING_SET_BYTES}"
            ))),
        ProcessMemoryEnvelope::Unavailable => Err(NyError::InvalidSpec(
            "wide patch source: whole-route memory envelope unavailable".into()
        )),
        _ => Ok(()),
    }
}

/// Pure geometry census before compiled tap tables or root matrices exist.
pub fn source_census(spec: &super::spec::TwinSpec) -> Option<RootSourceCensus> {
    use super::spec::TwinOpSpec;
    let sizes = spec.root_tensor_sizes()?;
    let head = spec
        .ops
        .iter()
        .position(|op| matches!(op, TwinOpSpec::Gemm { .. }))?;
    let head_columns = match &spec.ops[head] {
        TwinOpSpec::Gemm { shape, .. } => shape.0,
        _ => return None,
    };
    let widest = sizes.iter().copied().max()?;
    let mut compiled = spec.ops.len().checked_mul(1024)?;
    let mut gate_rows = 0usize;
    for (k, op) in spec.ops.iter().enumerate() {
        let bytes = match op {
            TwinOpSpec::Conv {
                weight,
                kernel: (_, ci, kh, kw),
                ishape,
                oshape,
                ..
            }
            | TwinOpSpec::ConvTranspose {
                weight,
                kernel: (_, ci, kh, kw),
                ishape,
                oshape,
                ..
            } => {
                let plane = oshape.1.checked_mul(oshape.2)?;
                let taps = kh.checked_mul(*kw)?;
                // Original weights, compiled forward/transposed weights, and
                // the patch plan's copies/tables. Three complete compiled
                // copies conservatively bound virtual receptive-field plans.
                let tables = plane
                    .checked_mul(*ci)?
                    .checked_mul(taps)?
                    .checked_mul(8)?
                    .checked_add(plane.checked_mul(taps)?.checked_mul(24)?)?
                    .checked_add(ishape.1.checked_mul(ishape.2)?.checked_mul(24)?)?;
                weight
                    .len()
                    .checked_mul(24)?
                    .checked_add(tables)?
                    .checked_mul(3)?
                    .checked_add(oshape.0.checked_mul(64)?)?
            }
            TwinOpSpec::ChannelAffine { .. } => sizes[k + 1].checked_mul(32)?,
            TwinOpSpec::Gemm { weight, bias, .. } => weight
                .len()
                .checked_mul(16)?
                .checked_add(bias.len().checked_mul(32)?)?,
            TwinOpSpec::Relu { input } if k < head => {
                gate_rows = gate_rows.checked_add(sizes[*input])?;
                0
            }
            _ => 0,
        };
        compiled = compiled.checked_add(bytes)?;
    }
    let prefix_rows = sizes[..=head]
        .iter()
        .try_fold(0usize, |sum, &rows| sum.checked_add(rows))?;
    let gate_and_reference_bytes = gate_rows
        .checked_mul(128)?
        .checked_add(prefix_rows.checked_mul(16)?)?
        .checked_add(widest.checked_mul(64)?)?;
    let dense_backward_column_bytes = prefix_rows.checked_mul(DENSE_BYTES_PER_ROW)?;
    // Reverse liveness over the original DAG: include the removed current
    // tensor AND every live/next tensor. Forty-eight bytes per row covers the
    // coefficient/error lanes, conv result and combined-error temporaries.
    let mut live = vec![false; sizes.len()];
    let first_input = match &spec.ops[head] {
        TwinOpSpec::Gemm { input, .. } => *input,
        _ => return None,
    };
    live[first_input] = true;
    let mut max_live_rows = 0usize;
    for k in (0..head).rev() {
        if !live[k + 1] {
            continue;
        }
        let current = sizes[k + 1];
        live[k + 1] = false;
        match &spec.ops[k] {
            TwinOpSpec::Conv { input, .. }
            | TwinOpSpec::ConvTranspose { input, .. }
            | TwinOpSpec::ChannelAffine { input, .. }
            | TwinOpSpec::Relu { input }
            | TwinOpSpec::Flatten { input } => live[*input] = true,
            TwinOpSpec::Add { lhs, rhs } => {
                live[*lhs] = true;
                live[*rhs] = true;
            }
            TwinOpSpec::Gemm { .. } => return None,
        }
        let rows = live
            .iter()
            .zip(&sizes)
            .try_fold(current, |sum, (&is_live, &rows)| {
                if is_live {
                    sum.checked_add(rows)
                } else {
                    Some(sum)
                }
            })?;
        max_live_rows = max_live_rows.max(rows);
    }
    let consumer_column_bytes = max_live_rows.checked_mul(DENSE_BYTES_PER_ROW)?;
    let head_pair_bytes = spec
        .n_in
        .checked_mul(head_columns)?
        .checked_mul(32)?
        .checked_add(head_columns.checked_mul(head_columns)?.checked_mul(32)?)?;
    let source_peak = compiled
        .checked_add(gate_and_reference_bytes)?
        .checked_add(WORKING_SET_BYTES as usize)?;
    let head_peak = consumer_column_bytes
        .checked_mul(head_columns)?
        .checked_add(compiled)?
        .checked_add(gate_and_reference_bytes)?
        .checked_add(head_pair_bytes)?;
    Some(RootSourceCensus {
        tableau_entries: spec.root_tableau_entries()?,
        compiled_and_plan_bytes: compiled,
        gate_and_reference_bytes,
        dense_backward_column_bytes,
        consumer_column_bytes,
        head_columns,
        head_pair_bytes,
        root_peak_bytes: source_peak.max(head_peak),
    })
}

/// Automatically recover only a formerly refused dense-width route, when
/// every requested tableau product is absent and the actual patch census fits.
pub fn select_source(
    spec: &super::spec::TwinSpec,
    requested: RootBoxSource,
    conflicts: PatchBackwardConflicts,
) -> RootBoxSource {
    if requested == RootBoxSource::Tableau
        && !spec.root_tableau_within_cap()
        && !conflicts.any()
        && source_geometry_admission(spec, RootBoxSource::PatchBackward, conflicts).is_ok()
    {
        RootBoxSource::PatchBackward
    } else {
        requested
    }
}

pub fn policy_conflicts(policy: &super::MarginRowRuntimePolicy) -> PatchBackwardConflicts {
    PatchBackwardConflicts {
        clip_rows: crate::preset_gates::margin_row_clip_rows_enabled(),
        retain: super::epoch_retention_requested(
            policy.epoch.tier0_exact,
            policy.epoch.depth,
            policy.epoch.parent_topk,
        ),
        root_f32: policy.root_precision.root_f32,
        tableau_f32: policy.root_precision.tableau_f32,
    }
}

pub fn wide_consumers_supported(policy: &super::MarginRowRuntimePolicy) -> bool {
    !(policy.root_alpha.alpha_enabled
        || policy.root_alpha.rebake_enabled
        || policy.root_alpha.spec_alpha_enabled
        || policy.root_certify_once.enabled
        || policy.joint_refresh.rounds > 0
        || policy.domain_alpha.enabled
        || policy.parallel_prebuild
        // Beta gradients retain a full linearized prefix, not only reverse
        // liveness. Both shared and per-column ascent need separate pricing.
        || super::beta::enabled()
        || super::closure::armed()
        // The frozen tier has its own state/shared-trunk allocations. Their
        // overlap with a newly admitted wide root has not been reserved here.
        || super::zono_state::armed()
        || policy.epoch.depth > 0
        || policy.epoch.parent_topk > 0
        || policy.width.adaptive
        || policy.wave != super::WavePolicy::default()
        || super::gpu_seam::enabled()
        || super::gpu_backward::enabled()
        || !super::gpu_conv::lane_route().is_off()
        || super::hyperplane_probe::enabled())
}

#[cfg(test)]
mod source_admission_tests {
    use super::*;
    use crate::margin_row::{
        run_margin_row_lane_with_policy, MarginRowOutcome, MarginRowRuntimePolicy, TwinOpSpec,
        TwinSpec,
    };

    fn wide_cnn_head(unsafe_head: bool) -> TwinSpec {
        let n = 40_000;
        TwinSpec {
            n_in: n,
            ops: vec![
                TwinOpSpec::Conv {
                    input: 0,
                    weight: vec![1.0],
                    bias: vec![0.0],
                    bias_err: vec![0.0],
                    weight_rel_err: 0.0,
                    kernel: (1, 1, 1, 1),
                    stride: (1, 1),
                    pads: (0, 0, 0, 0),
                    ishape: (1, 200, 200),
                    oshape: (1, 200, 200),
                },
                TwinOpSpec::Relu { input: 1 },
                TwinOpSpec::Flatten { input: 2 },
                TwinOpSpec::Gemm {
                    input: 3,
                    weight: vec![0.0; 2 * n],
                    bias: vec![2.0, 3.0],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (2, n),
                },
                TwinOpSpec::Relu { input: 4 },
                TwinOpSpec::Gemm {
                    input: 5,
                    weight: vec![1.0, 0.0, 0.0, 0.0],
                    bias: vec![0.0, if unsafe_head { 4.0 } else { -1.0 }],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (2, 2),
                },
            ],
        }
    }

    #[test]
    fn wide_patch_source_automatic_arrival_completes_real_certified_root() {
        let _env = ny_test_utils::env::lock_env_shared();
        let spec = wide_cnn_head(false);
        assert!(!spec.root_tableau_within_cap());
        assert!(TwinNet::compile(&spec).is_err());
        let policy = MarginRowRuntimePolicy::default();
        let conflicts = policy_conflicts(&policy);
        assert!(
            !conflicts.any(),
            "fixture requires ordinary clip-free defaults"
        );
        assert_eq!(
            select_source(&spec, RootBoxSource::Tableau, conflicts),
            RootBoxSource::PatchBackward
        );
        let census = source_census(&spec).unwrap();
        assert!(census.root_peak_bytes < ROOT_SOURCE_BYTES_CAP);
        let mut lo = vec![1.0; spec.n_in];
        lo[0] = -1.0; // A genuinely unstable conv row buys a completed prefix pass.
        let hi = vec![1.0; spec.n_in];
        match run_margin_row_lane_with_policy(&spec, &lo, &hi, 0, &[1], None, 0, policy) {
            MarginRowOutcome::Unsat(stats, _) => {
                assert_eq!(stats.root_closed_classes, 1);
                assert_eq!(stats.ledger_ok, Some(true));
                // A patch root completes without any dense forward conv/store
                // work; the oversized dense route would have refused above.
                assert_eq!(stats.root_precision_execution.eligible_conv_ops, 0);
                assert_eq!(stats.root_precision_execution.eligible_store_ops, 0);
            }
            MarginRowOutcome::Unknown { reason, .. } => {
                panic!("wide automatic patch root must close the constant safe head: {reason}")
            }
        }
    }

    #[test]
    fn wide_patch_declines_never_allocate_dense_tableau_or_close_sat_control() {
        let _env = ny_test_utils::env::lock_env_shared();
        let spec = wide_cnn_head(true);
        let net = TwinNet::compile_with_root_source(
            &spec,
            RootBoxSource::PatchBackward,
            PatchBackwardConflicts::default(),
        )
        .unwrap();
        let lo = vec![1.0; spec.n_in];
        let hi = lo.clone();
        let policy = super::super::RootPrecisionRuntimePolicy {
            box_source: RootBoxSource::PatchBackward,
            ..Default::default()
        };
        // Parity declines the patch source, then the standalone dense guard
        // refuses before its n_in-by-n_in identity/tableau is allocated.
        let error = RootGates::build_retaining_with_policy(
            &net,
            &lo,
            &hi,
            RoundMode::Parity,
            None,
            None,
            &[],
            policy,
            super::super::BackwardIntermRuntimePolicy::default(),
            None,
        )
        .err()
        .expect("dense fallback must refuse the oversized forward tableau");
        assert!(format!("{error}").contains("original entry cap"));
        assert!(matches!(
            run_margin_row_lane_with_policy(
                &spec,
                &lo,
                &hi,
                0,
                &[1],
                None,
                0,
                MarginRowRuntimePolicy::default()
            ),
            MarginRowOutcome::Unknown { .. }
        ));
    }

    #[test]
    fn wide_source_keeps_tableau_product_and_consumer_refusals() {
        let _env = ny_test_utils::env::lock_env_shared();
        let spec = wide_cnn_head(false);
        for conflicts in [
            PatchBackwardConflicts {
                clip_rows: true,
                ..Default::default()
            },
            PatchBackwardConflicts {
                retain: true,
                ..Default::default()
            },
            PatchBackwardConflicts {
                root_f32: true,
                ..Default::default()
            },
            PatchBackwardConflicts {
                tableau_f32: true,
                ..Default::default()
            },
        ] {
            assert_eq!(
                select_source(&spec, RootBoxSource::Tableau, conflicts),
                RootBoxSource::Tableau
            );
            assert!(source_admission(&spec, RootBoxSource::PatchBackward, conflicts).is_err());
        }
        let policy = MarginRowRuntimePolicy {
            parallel_prebuild: true,
            ..MarginRowRuntimePolicy::default()
        };
        assert!(!wide_consumers_supported(&policy));
        let census = source_census(&spec).unwrap();
        assert!(census.consumer_peak_bytes(usize::MAX).is_none());
    }

    #[test]
    fn wide_working_set_low_headroom_refuses_whole_route_without_published_boxes() {
        use crate::network::crown_memory::ProcessMemoryEnvelope;
        let _env = ny_test_utils::env::lock_env_shared();
        let spec = wide_cnn_head(false);
        let policy = MarginRowRuntimePolicy::default();
        let conflicts = policy_conflicts(&policy);
        let census = source_census(&spec).unwrap();
        let low = ProcessMemoryEnvelope::Bounded { headroom_bytes: 1 };
        let error = with_wide_memory_for_test(low, || {
            TwinNet::compile_with_root_source(&spec, RootBoxSource::PatchBackward, conflicts)
        })
        .err()
        .expect("low headroom must refuse before compiled tap allocation");
        assert!(format!("{error}").contains("whole-route memory refusal"));
        let net = with_wide_memory_for_test(ProcessMemoryEnvelope::Unbounded, || {
            TwinNet::compile_with_root_source(&spec, RootBoxSource::PatchBackward, conflicts)
        })
        .unwrap();
        let mut lo = vec![1.0; spec.n_in];
        lo[0] = -1.0;
        let hi = vec![1.0; spec.n_in];
        let root_policy = super::super::RootPrecisionRuntimePolicy {
            box_source: RootBoxSource::PatchBackward,
            ..Default::default()
        };
        let build = || {
            RootGates::build_retaining_with_policy(
                &net,
                &lo,
                &hi,
                RoundMode::Outward,
                None,
                None,
                &[],
                root_policy,
                super::super::BackwardIntermRuntimePolicy::default(),
                None,
            )
        };
        let (refused, taps) = with_root_box_tap_for_test(|| with_wide_memory_for_test(low, build));
        assert!(refused.is_err());
        assert!(
            taps.is_empty(),
            "no weakened root may be published under pressure"
        );
        let (first, _) =
            with_wide_memory_for_test(ProcessMemoryEnvelope::Unbounded, build).unwrap();
        let (second, _) = with_wide_memory_for_test(
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: (census.root_peak_bytes as u64).saturating_mul(2),
            },
            build,
        )
        .unwrap();
        assert_eq!(first.layers.len(), second.layers.len());
        for (a, b) in first.layers.iter().zip(&second.layers) {
            assert_eq!(
                a.l.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.l.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                a.u.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.u.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                a.alpha
                    .iter()
                    .chain(&a.s)
                    .chain(&a.c)
                    .chain(&a.ms)
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                b.alpha
                    .iter()
                    .chain(&b.s)
                    .chain(&b.c)
                    .chain(&b.ms)
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(a.unst, b.unst);
        }
        let outcome = with_wide_memory_for_test(low, || {
            run_margin_row_lane_with_policy(&spec, &lo, &hi, 0, &[1], None, 0, policy)
        });
        match outcome {
            MarginRowOutcome::Unknown { reason, .. } => {
                assert!(reason.contains("whole-route memory refusal"), "{reason}")
            }
            MarginRowOutcome::Unsat(_, _) => {
                panic!("low memory must not publish a downgraded wide proof")
            }
        }
    }

    #[test]
    fn wide_consumer_census_prices_full_columns_workers_and_cache_without_cap_raise() {
        let _env = ny_test_utils::env::lock_env_shared();
        let census = source_census(&wide_cnn_head(false)).unwrap();
        let policy = MarginRowRuntimePolicy::default();
        let one = consumer_census(census, &policy, 1, 1, 1).unwrap();
        let many = consumer_census(census, &policy, 200, 8, 64).unwrap();
        assert_eq!(one.columns, 34);
        assert!(one.peak_bytes < ROOT_SOURCE_BYTES_CAP);
        assert!(many.peak_bytes > ROOT_SOURCE_BYTES_CAP);
        assert_eq!(many.cache_bytes, census.head_pair_bytes * 8 * 64);
        assert!(consumer_census(census, &policy, usize::MAX, 1, 1).is_none());
        let mut product = policy;
        product.beta_per_column = true;
        product.beta_columns = 32;
        product.column_retire = true;
        assert_eq!(
            consumer_census(census, &product, 1, 1, 1),
            Some(one),
            "inert beta columns and retirement do not invent additional dense work"
        );
    }
}

pub fn source_admission(
    spec: &super::spec::TwinSpec,
    source: RootBoxSource,
    conflicts: PatchBackwardConflicts,
) -> Result<Option<RootSourceCensus>> {
    let census = source_geometry_admission(spec, source, conflicts)?;
    if !spec.root_tableau_within_cap() {
        if let Some(census) = census {
            wide_memory_admission(census.root_peak_bytes)?;
        }
    }
    Ok(census)
}

fn source_geometry_admission(
    spec: &super::spec::TwinSpec,
    source: RootBoxSource,
    conflicts: PatchBackwardConflicts,
) -> Result<Option<RootSourceCensus>> {
    if source == RootBoxSource::Tableau || conflicts.any() {
        if !spec.root_tableau_within_cap() {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: root tableau needs {:?} entries, above the original {} cap",
                spec.root_tableau_entries(),
                super::spec::MAX_ROOT_TABLEAU_ENTRIES
            )));
        }
        return Ok(None);
    }
    let census = source_census(spec).ok_or_else(|| {
        NyError::InvalidSpec(
            "margin_row: root source geometry overflow or malformed tensors".into(),
        )
    })?;
    if census.root_peak_bytes > ROOT_SOURCE_BYTES_CAP
        || census.dense_backward_column_bytes > WORKING_SET_BYTES as usize
    {
        return Err(NyError::InvalidSpec(format!(
            "margin_row: patch source/consumer peak {} exceeds original {} byte cap",
            census.root_peak_bytes, ROOT_SOURCE_BYTES_CAP
        )));
    }
    Ok(Some(census))
}

// ---------------------------------------------------------------------------
//  The typed key
// ---------------------------------------------------------------------------

/// Which certified construction supplies the root's trunk boxes
/// (`margin_row.root_box_source`).
///
/// The serde tokens are the typed preset key's values and the lever
/// declaration's `Enum` members ([`Self::TOKENS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RootBoxSource {
    /// The dense forward DeepPoly tableau (`tableau`, the default): today's
    /// root build, bit for bit.
    #[default]
    #[serde(rename = "tableau")]
    Tableau,
    /// This module (`patch_backward`, M5b): certified IBP reference plus a
    /// certified patch-backward pass over the reference-unstable neurons.
    #[serde(rename = "patch_backward")]
    PatchBackward,
}

impl RootBoxSource {
    /// The preset / environment vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 2] = ["tableau", "patch_backward"];

    /// Parse one canonical token (exact spelling).
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "tableau" => Some(Self::Tableau),
            "patch_backward" => Some(Self::PatchBackward),
            _ => None,
        }
    }

    /// The canonical token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Tableau => "tableau",
            Self::PatchBackward => "patch_backward",
        }
    }
}

// ---------------------------------------------------------------------------
//  Composition: what the source refuses rather than silently drops
// ---------------------------------------------------------------------------

/// Tableau products an ARMED patch-backward build was asked for but cannot
/// produce (see the module docs, "REFUSED, never dropped"). Any one of them
/// refuses the source: the unchanged tableau build runs instead, bit for bit.
///
/// Pure and profile-blind; the root build (runtime) and `preset::apply`
/// (validation) both decide through [`Self::any`], so the two refusals cannot
/// drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PatchBackwardConflicts {
    /// `#clip-rows` retention is requested (`margin_row.clip_rows`).
    pub clip_rows: bool,
    /// Tier-0 row retention is requested (`#epoch-bab`: a `RetainCfg` reached
    /// the root build; at validation, `epoch_retention_requested`).
    pub retain: bool,
    /// The f32 tableau compute kernel is requested (`margin_row.root_f32`).
    pub root_f32: bool,
    /// The f32 tableau storage is requested (`margin_row.tableau_f32`).
    pub tableau_f32: bool,
}

impl PatchBackwardConflicts {
    /// The stable reason tokens, in declaration order.
    pub const TOKENS: [&'static str; 4] = ["clip_rows", "retain", "root_f32", "tableau_f32"];

    /// Whether any requested product would be dropped (the source refuses).
    #[must_use]
    pub const fn any(self) -> bool {
        self.clip_rows || self.retain || self.root_f32 || self.tableau_f32
    }

    /// `a+b` over the requested products in declaration order (`-` if none).
    #[must_use]
    pub fn reason(self) -> String {
        let hit = [self.clip_rows, self.retain, self.root_f32, self.tableau_f32];
        let joined = Self::TOKENS
            .iter()
            .zip(hit)
            .filter_map(|(token, on)| on.then_some(*token))
            .collect::<Vec<_>>()
            .join("+");
        if joined.is_empty() {
            "-".to_string()
        } else {
            joined
        }
    }
}

// ---------------------------------------------------------------------------
//  Scheduling constants (none of them can change a published bit)
// ---------------------------------------------------------------------------

/// Crossover ratio of every layer's patch plan: `#bi-patch`'s declaration
/// default (walking backward, a frame larger than half its tensor runs as the
/// dense tail).
const PATCH_RATIO: f64 = DEFAULT_RATIO;
/// Working-set ceiling, in bytes, of all concurrently running chunks of one
/// layer. Narrow legacy nets may cap it further by observed headroom; wide
/// sources admit and freeze this full ceiling before allocating their root.
const WORKING_SET_BYTES: u64 = 4 << 30;
/// Fixed, admitted working-set ceiling for newly accepted wide sources.
pub const fn wide_working_set_bytes() -> u64 {
    WORKING_SET_BYTES
}
/// Working-set floor, so an observed-but-tiny headroom still admits a chunk.
const WORKING_SET_FLOOR: u64 = 64 << 20;
/// Widest patch chunk, in requested columns (each a fused Lower+Upper pair):
/// narrow enough to balance the rayon pool, wide enough for the conv kernel's
/// length-`R` inner loops.
const PATCH_CHUNK_MAX: usize = 128;
/// Widest dense-fallback chunk (the `#backward-interm` default grain).
const DENSE_CHUNK_MAX: usize = super::DEFAULT_BACKWARD_INTERM_CHUNK;
/// Bytes per dense-fallback column per prefix-tensor neuron: the coefficient
/// and error lanes plus the conv and combined-error temporaries.
const DENSE_BYTES_PER_ROW: usize = 48;

// ---------------------------------------------------------------------------
//  Outcomes
// ---------------------------------------------------------------------------

/// Why an ARMED build declined to the tableau.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decline {
    /// Not the certified-outward verdict mode (the source is an Outward
    /// certificate; Parity builds keep the tableau).
    Parity,
    /// A certified reference box was non-finite, missing or malformed at
    /// tensor `tensor` (fail closed rather than guess).
    Reference { tensor: usize },
    /// `count` candidates of trunk layer `layer` had crossing certified
    /// enclosures: a defect signal (or an empty baked split subdomain).
    Crossed { layer: usize, count: usize },
}

impl Decline {
    fn token(self) -> String {
        match self {
            Self::Parity => "parity".to_string(),
            Self::Reference { tensor } => format!("reference:t{tensor}"),
            Self::Crossed { layer, count } => format!("crossed:layer{layer}x{count}"),
        }
    }
}

/// A decline and the wall it spent before declining.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Declined {
    pub(crate) why: Decline,
    pub(crate) secs: f64,
}

/// The result of an armed build attempt.
pub(crate) enum Attempt {
    /// The patch-backward source built the root (or failed the way the tableau
    /// fails: a malformed box or spec, or the deadline). Boxed: the built
    /// root is 488 bytes against `Declined`'s 32 (clippy::large_enum_variant,
    /// workspace threshold 64).
    Built(Box<Result<(RootGates, Option<RetainedRows>)>>),
    /// Declined: the caller runs the unchanged tableau build.
    Declined(Declined),
}

/// One trunk layer's counters (the per-layer observable).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LayerStats {
    /// Layer width.
    pub(crate) neurons: usize,
    /// Neurons unstable in the certified reference box (the candidates).
    pub(crate) ref_unst: usize,
    /// Candidate columns whose bounds a PATCH walk certified.
    pub(crate) patch_cols: usize,
    /// Candidate columns whose bounds a DENSE prefix pair certified.
    pub(crate) dense_cols: usize,
    /// Candidates whose certified pass failed closed (the reference box
    /// stands).
    pub(crate) unavailable: usize,
    /// The layer's patch-plan refusal, when the whole layer ran dense.
    pub(crate) plan_refused: Option<&'static str>,
    /// Candidates whose box strictly shrank.
    pub(crate) tightened: usize,
    /// Candidates the backward bound proved stable.
    pub(crate) stabilized: usize,
    /// Unstable neurons in the PUBLISHED box.
    pub(crate) unst: usize,
}

#[derive(Default)]
struct Totals {
    layers: usize,
    ref_unst: usize,
    patch_cols: usize,
    dense_cols: usize,
    unavailable: usize,
    plan_refused: usize,
    stabilized: usize,
    unst: usize,
}

impl Totals {
    fn absorb(&mut self, st: &LayerStats) {
        self.layers += 1;
        self.ref_unst += st.ref_unst;
        self.patch_cols += st.patch_cols;
        self.dense_cols += st.dense_cols;
        self.unavailable += st.unavailable;
        self.plan_refused += usize::from(st.plan_refused.is_some());
        self.stabilized += st.stabilized;
        self.unst += st.unst;
    }
}

// ---------------------------------------------------------------------------
//  TEST-ONLY mutation switch and per-layer tap
// ---------------------------------------------------------------------------

/// A code mutation the containment tests must catch (TEST-ONLY; outside
/// `cfg(test)` `mutation` is a `const fn` returning `None`, so no
/// production build can reach one).
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Mutation {
    /// Run every backward pass on a PARITY scratch root: no error lane, raw
    /// round-to-nearest concretize — the uncharged-rounding class of the
    /// historical root leak.
    ParityScratch,
    /// Carry a ReLU's output reference as its RAW pre-activation box (no
    /// `max(., 0)`).
    UnclampedReluRef,
    /// Move every backward bound INWARD by this amount before the
    /// intersection (an under-charged certified pass).
    ShiftBackward(f64),
}

#[cfg(test)]
thread_local! {
    /// TEST-ONLY mutation (see [`with_root_box_mutation_for_test`]).
    static MUTATION: std::cell::Cell<Option<Mutation>> = const { std::cell::Cell::new(None) };
    /// TEST-ONLY per-layer tap (see [`with_root_box_tap_for_test`]).
    static TAP: std::cell::RefCell<Option<Vec<LayerTap>>> = const { std::cell::RefCell::new(None) };
    /// TEST-ONLY chunk-width cap (see [`with_root_box_chunk_cap_for_test`]).
    static CHUNK_CAP: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn mutation() -> Option<Mutation> {
    MUTATION.with(std::cell::Cell::get)
}

#[cfg(not(test))]
const fn mutation() -> Option<Mutation> {
    None
}

#[cfg(test)]
fn chunk_cap() -> Option<usize> {
    CHUNK_CAP.with(std::cell::Cell::get)
}

#[cfg(not(test))]
const fn chunk_cap() -> Option<usize> {
    None
}

/// TEST-ONLY: run `f` with every patch and dense-fallback chunk of a
/// patch-backward build on THIS thread capped at `cap` (at least 1) columns
/// (read once per build, on the calling thread; never inside rayon).
/// Scheduling only, like the working-set sizing it narrows: it exists so the
/// thread-count test also varies chunk COMPOSITION, which the working-set
/// sizing alone never does on the small fixtures. Reset on exit, unwind
/// included. Outside `cfg(test)` the cap is a `const fn` returning `None`.
#[cfg(test)]
pub(crate) fn with_root_box_chunk_cap_for_test<Out>(cap: usize, f: impl FnOnce() -> Out) -> Out {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CHUNK_CAP.with(|c| c.set(None));
        }
    }
    CHUNK_CAP.with(|c| c.set(Some(cap.max(1))));
    let _reset = Reset;
    f()
}

/// TEST-ONLY: run `f` with `m` applied to every patch-backward build on THIS
/// thread (read once per build, on the calling thread; never inside rayon).
/// Reset on exit, unwind included.
#[cfg(test)]
pub(crate) fn with_root_box_mutation_for_test<Out>(m: Mutation, f: impl FnOnce() -> Out) -> Out {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            MUTATION.with(|c| c.set(None));
        }
    }
    MUTATION.with(|c| c.set(Some(m)));
    let _reset = Reset;
    f()
}

/// TEST-ONLY record of one trunk layer's step.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct LayerTap {
    /// Trunk layer index.
    pub(crate) layer: usize,
    /// Relu op index.
    pub(crate) op: usize,
    /// The certified reference box handed in.
    pub(crate) ref_l: Vec<f64>,
    /// Upper side of the reference box.
    pub(crate) ref_u: Vec<f64>,
    /// The candidates (reference-unstable), ascending.
    pub(crate) cand: Vec<usize>,
    /// The published box.
    pub(crate) out_l: Vec<f64>,
    /// Upper side of the published box.
    pub(crate) out_u: Vec<f64>,
    /// The printed counters.
    pub(crate) stats: LayerStats,
}

/// TEST-ONLY: run `f` and return every layer step a patch-backward build on
/// THIS thread took (the per-layer bookkeeping runs on the calling thread).
#[cfg(test)]
pub(crate) fn with_root_box_tap_for_test<Out>(f: impl FnOnce() -> Out) -> (Out, Vec<LayerTap>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TAP.with(|t| t.borrow_mut().take());
        }
    }
    TAP.with(|t| *t.borrow_mut() = Some(Vec::new()));
    let reset = Reset;
    let out = f();
    let taps = TAP.with(|t| t.borrow_mut().take()).unwrap_or_default();
    drop(reset);
    (out, taps)
}

// ---------------------------------------------------------------------------
//  The build
// ---------------------------------------------------------------------------

/// Frozen inputs of one build.
struct Ctx<'a> {
    net: &'a TwinNet,
    mid: Vec<f64>,
    rad: Vec<f64>,
    xabs: Vec<f64>,
    lo: &'a [f64],
    hi: &'a [f64],
    deadline: Option<Instant>,
    mutation: Option<Mutation>,
    /// TEST-ONLY chunk-width cap (`None` outside `cfg(test)`); scheduling only.
    chunk_cap: Option<usize>,
    threads: usize,
    budget: usize,
}

/// Certified reference box per tensor id, dropped after its last in-walk
/// consumer.
struct RefBoxes {
    boxes: Vec<Option<IBox>>,
    left: Vec<usize>,
}

impl RefBoxes {
    fn get(&self, id: usize) -> Option<&IBox> {
        self.boxes.get(id).and_then(Option::as_ref)
    }

    fn release(&mut self, id: usize) {
        if let Some(n) = self.left.get_mut(id) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                if let Some(b) = self.boxes.get_mut(id) {
                    *b = None;
                }
            }
        }
    }
}

/// How a layer step stopped the build.
enum Stop {
    Declined(Decline),
    Error(NyError),
}

/// Build the root with the certified patch-backward box source (see the
/// module docs). Same contract as the tableau walk for everything it returns:
/// `Err` on a malformed input box or spec and on the deadline; a
/// [`Attempt::Declined`] hands the build back to the tableau.
///
/// Precondition (enforced by its one caller, `RootGates::build_retaining_inner`):
/// no [`PatchBackwardConflicts`] — a build that requests clip rows, Tier-0
/// retention or an f32 tableau kernel never reaches this function.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn try_build(
    net: &TwinNet,
    lo: &[f64],
    hi: &[f64],
    mode: RoundMode,
    deadline: Option<Instant>,
    retain: Option<&RetainCfg>,
    splits: &[(usize, usize, i8)],
    precision_policy: super::RootPrecisionRuntimePolicy,
    bi_policy: super::BackwardIntermRuntimePolicy,
    alpha_seed: Option<&[Vec<f64>]>,
) -> Attempt {
    let t0 = Instant::now();
    if !mode.outward() {
        return Attempt::Declined(Declined {
            why: Decline::Parity,
            secs: 0.0,
        });
    }
    // Formerly rejected wide nets use one frozen working set. Dynamic
    // headroom is an admission check, never a reason to omit tightening.
    let budget = if !net.root_tableau_within_cap() {
        let required = net
            .tsize
            .iter()
            .take(net.i_gemm1 + 1)
            .try_fold(0usize, |sum, &rows| sum.checked_add(rows))
            .and_then(|rows| rows.checked_mul(16))
            .and_then(|bytes| bytes.checked_add(net.tsize.iter().copied().max()?.checked_mul(64)?))
            .and_then(|bytes| {
                let gate_rows = net
                    .trunk_relus
                    .iter()
                    .try_fold(0usize, |sum, &op| sum.checked_add(*net.tsize.get(op + 1)?))?;
                bytes.checked_add(gate_rows.checked_mul(128)?)
            })
            .and_then(|bytes| bytes.checked_add(WORKING_SET_BYTES as usize));
        let Some(required) = required else {
            return Attempt::Built(Box::new(Err(NyError::InvalidSpec(
                "wide patch source: working-set geometry overflow".into(),
            ))));
        };
        if let Err(error) = wide_memory_admission(required) {
            return Attempt::Built(Box::new(Err(error)));
        }
        WORKING_SET_BYTES as usize
    } else {
        working_set_bytes()
    };
    let n_in = net.n_in;
    if lo.len() != n_in || hi.len() != n_in {
        return Attempt::Built(Box::new(Err(NyError::shape_mismatch(
            vec![n_in],
            vec![lo.len()],
        ))));
    }
    // The input box: the tableau walk's Outward formulas, verbatim.
    let mut mid = vec![0.0; n_in];
    let mut rad = vec![0.0; n_in];
    for i in 0..n_in {
        if !(lo[i].is_finite() && hi[i].is_finite() && lo[i] <= hi[i]) {
            return Attempt::Built(Box::new(Err(NyError::InvalidSpec(format!(
                "margin_row: bad input box at {i}: [{}, {}]",
                lo[i], hi[i]
            )))));
        }
        // Parity-critical formula shared with `root.rs` (`mid = (lo+hi)/2`).
        #[allow(clippy::manual_midpoint)]
        {
            mid[i] = (lo[i] + hi[i]) / 2.0;
        }
        rad[i] = next_up((hi[i] - mid[i]).max(mid[i] - lo[i]));
    }
    let xabs: Vec<f64> = mid.iter().zip(&rad).map(|(m, r)| m.abs() + r).collect();
    let Some(&last_relu) = net.trunk_relus.last() else {
        return Attempt::Built(Box::new(Err(NyError::InvalidSpec(
            "margin_row: no trunk relu".into(),
        ))));
    };
    // In-walk consumer counts over the trunk prefix (the tableau walk's).
    let mut consumers = vec![0usize; net.ops.len() + 1];
    for op in &net.ops[..=last_relu] {
        match op {
            TwinOp::Conv(c) => consumers[c.input] += 1,
            TwinOp::Relu { input, .. }
            | TwinOp::Flatten { input }
            | TwinOp::ChannelAffine { input, .. } => consumers[*input] += 1,
            TwinOp::Add { lhs, rhs } => {
                consumers[*lhs] += 1;
                consumers[*rhs] += 1;
            }
            TwinOp::Gemm { .. } => {
                return Attempt::Built(Box::new(Err(NyError::InvalidSpec(
                    "margin_row: gemm before last trunk relu".into(),
                ))))
            }
        }
    }
    let ctx = Ctx {
        net,
        mid,
        rad,
        xabs,
        lo,
        hi,
        deadline,
        mutation: mutation(),
        chunk_cap: chunk_cap(),
        threads: rayon::current_num_threads().max(1),
        budget,
    };
    let mut refs = RefBoxes {
        boxes: vec![None; net.ops.len() + 1],
        left: consumers,
    };
    refs.boxes[0] = Some((lo.to_vec(), hi.to_vec()));
    let naug = n_in + 1;
    // Baked splits per trunk layer (#epoch-bab Tier 2), as the tableau walk.
    let mut split_by_layer: BTreeMap<usize, Vec<(usize, i8)>> = BTreeMap::new();
    for &(li, idx, dir) in splits {
        split_by_layer.entry(li).or_default().push((idx, dir));
    }
    let mut retained = retain.map(|_| RetainedRows { layers: Vec::new() });
    let mut layers: Vec<LayerGates> = Vec::with_capacity(net.trunk_relus.len());
    let mut totals = Totals::default();
    let done = |totals: &Totals, complete: bool| {
        print_done(
            totals,
            precision_policy,
            bi_policy,
            retain.is_some(),
            complete,
            t0,
        );
    };
    let declined = |why: Decline| {
        if let Decline::Crossed { layer, count } = why {
            eprintln!(
                "[root-box] CROSSED layer={layer} count={count}: the certified reference and \
                 backward enclosures disagree — DEFECT SIGNAL (or an empty baked split \
                 subdomain); the patch-backward build is DECLINED and the tableau runs"
            );
        }
        Attempt::Declined(Declined {
            why,
            secs: t0.elapsed().as_secs_f64(),
        })
    };
    for (k, op) in net.ops.iter().enumerate().take(last_relu + 1) {
        if let Err(e) = check_deadline(deadline) {
            done(&totals, false);
            return Attempt::Built(Box::new(Err(e)));
        }
        let next: Option<IBox> = match op {
            TwinOp::Conv(c) => {
                let out = refs
                    .get(c.input)
                    .and_then(|(l, u)| ibp_seed::conv_box(c, l, u, net.tsize[k + 1]));
                refs.release(c.input);
                out
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
            } => {
                let out = refs.get(*input).and_then(|(l, u)| {
                    ibp_seed::affine_box(scale, shift, *scale_rel_err, shift_err, l, u)
                });
                refs.release(*input);
                out
            }
            TwinOp::Flatten { input } => {
                let out = refs.get(*input).cloned();
                refs.release(*input);
                out
            }
            TwinOp::Add { lhs, rhs } => {
                let out = match (refs.get(*lhs), refs.get(*rhs)) {
                    (Some(a), Some(b)) => ibp_seed::add_box(a, b),
                    _ => None,
                };
                refs.release(*lhs);
                refs.release(*rhs);
                out
            }
            TwinOp::Relu { input, layer } => {
                let li = layers.len();
                let n = net.tsize[k + 1];
                if *layer != li {
                    done(&totals, false);
                    return Attempt::Built(Box::new(Err(NyError::InvalidSpec(format!(
                        "margin_row: trunk relu op {k} carries layer {layer}, walk expected {li}"
                    )))));
                }
                let Some((rl, ru)) = refs.get(*input).cloned() else {
                    return declined(Decline::Reference { tensor: *input });
                };
                refs.release(*input);
                if rl.len() != n || ru.len() != n {
                    return declined(Decline::Reference { tensor: *input });
                }
                let layer_t0 = Instant::now();
                let (l, u, mut st) = match layer_box(&ctx, &mut layers, k, li, &rl, &ru) {
                    Ok(v) => v,
                    Err(Stop::Declined(why)) => return declined(why),
                    Err(Stop::Error(e)) => {
                        done(&totals, false);
                        return Attempt::Built(Box::new(Err(e)));
                    }
                };
                // Gates from the PUBLISHED box — the tableau walk's
                // derivation verbatim (`root.rs::walk_ops`, relu arm).
                let mut gates = gates_from_box(&l, &u);
                repair_upper_lines(&l, &u, &mut gates.1, &mut gates.2);
                let (mut alpha, mut s, mut c) = gates;
                // #rebake seed: UNSTABLE neurons only, finite, clamped to
                // [0, 1]; shape mismatches fail open to the heuristic.
                if let Some(seed) = alpha_seed {
                    if let Some(sa) = seed.get(li) {
                        if sa.len() == n {
                            for j in 0..n {
                                if l[j] < 0.0 && u[j] > 0.0 && sa[j].is_finite() {
                                    alpha[j] = sa[j].clamp(0.0, 1.0);
                                }
                            }
                        }
                    }
                }
                // Baked splits: exact fixed lines, valid on the split
                // halfspaces (the calling subtree's domain).
                let mut baked: BTreeSet<usize> = BTreeSet::new();
                if let Some(list) = split_by_layer.get(&li) {
                    for &(idx, dir) in list {
                        if idx >= n {
                            done(&totals, false);
                            return Attempt::Built(Box::new(Err(NyError::InvalidSpec(format!(
                                "margin_row: baked split neuron {idx} out of range {n}"
                            )))));
                        }
                        if dir > 0 {
                            alpha[idx] = 1.0;
                            s[idx] = 1.0;
                            c[idx] = 0.0;
                        } else {
                            alpha[idx] = 0.0;
                            s[idx] = 0.0;
                            c[idx] = 0.0;
                        }
                        baked.insert(idx);
                    }
                }
                let ms: Vec<f64> = alpha.iter().zip(&s).map(|(a, b)| a.max(*b)).collect();
                let unst: Vec<usize> = (0..n)
                    .filter(|&j| l[j] < 0.0 && u[j] > 0.0 && !baked.contains(&j))
                    .collect();
                st.unst = unst.len();
                if let Some(ret) = retained.as_mut() {
                    ret.layers.push(RetainedLayer {
                        eligible: unst.len(),
                        idx: Vec::new(),
                        unst_pos: Vec::new(),
                        a_l: Vec::new(),
                        a_u: Vec::new(),
                        naug,
                    });
                }
                print_layer(li, k, &st, layer_t0.elapsed().as_secs_f64());
                totals.absorb(&st);
                // The ReLU output reference from the FINAL published box (the
                // last trunk relu's output has no in-walk consumer).
                let out = if k == last_relu {
                    None
                } else if matches!(ctx.mutation, Some(Mutation::UnclampedReluRef)) {
                    Some((l.clone(), u.clone()))
                } else {
                    Some(ibp_seed::relu_box(&l, &u))
                };
                #[cfg(test)]
                TAP.with(|t| {
                    if let Some(v) = t.borrow_mut().as_mut() {
                        v.push(LayerTap {
                            layer: li,
                            op: k,
                            ref_l: rl.clone(),
                            ref_u: ru.clone(),
                            cand: (0..n).filter(|&j| rl[j] < 0.0 && ru[j] > 0.0).collect(),
                            out_l: l.clone(),
                            out_u: u.clone(),
                            stats: st,
                        });
                    }
                });
                layers.push(LayerGates {
                    op: k,
                    n,
                    l,
                    u,
                    alpha,
                    s,
                    c,
                    ms,
                    unst,
                    // Tableau rows; see the module docs.
                    clip_rows: None,
                });
                let Some(out) = out else {
                    break;
                };
                Some(out)
            }
            // Refused by the consumer count above.
            TwinOp::Gemm { .. } => None,
        };
        let Some(b) = next else {
            return declined(Decline::Reference { tensor: k + 1 });
        };
        if refs.left.get(k + 1).copied().unwrap_or(0) > 0 {
            refs.boxes[k + 1] = Some(b);
        }
    }
    done(&totals, true);
    let Ctx { mid, rad, xabs, .. } = ctx;
    // The tableau's precision work did not run: zero eligible ops, the
    // configured policy honoured (never refused, never mismatched).
    let expected_f32 = precision_policy.root_f32 && mode.outward();
    let root_precision_execution = RootPrecisionExecutionEvidence::completed(
        precision_policy,
        mode,
        expected_f32,
        precision_policy.tableau_f32 && expected_f32,
        0,
        0,
    );
    Attempt::Built(Box::new(Ok((
        RootGates {
            mode,
            mid,
            rad,
            xabs,
            layers,
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            // Subsumed, not run: the configured policy with zero work.
            backward_interm_execution: BackwardIntermExecutionEvidence::from_policy(bi_policy),
            // Carried into every descendant root (rebake rounds, epoch
            // rebuilds), which therefore use this source too.
            root_precision_policy: precision_policy,
            root_precision_execution,
            // Installed by the root driver after the build, never by it.
            spec_alpha_policy: None,
            head_alpha: None,
            root_eval_memo: super::bab::RootEvalMemo::default(),
            head_group_source: None,
        },
        retained,
    ))))
}

/// One trunk layer's published box: the certified reference, intersected
/// shrink-only with the certified backward bounds of its candidates.
fn layer_box(
    ctx: &Ctx<'_>,
    layers: &mut Vec<LayerGates>,
    relu_op: usize,
    li: usize,
    rl: &[f64],
    ru: &[f64],
) -> std::result::Result<(Vec<f64>, Vec<f64>, LayerStats), Stop> {
    let n = rl.len();
    let mut l = rl.to_vec();
    let mut u = ru.to_vec();
    let cand: Vec<usize> = (0..n).filter(|&j| rl[j] < 0.0 && ru[j] > 0.0).collect();
    let mut st = LayerStats {
        neurons: n,
        ref_unst: cand.len(),
        ..LayerStats::default()
    };
    if cand.is_empty() {
        return Ok((l, u, st));
    }
    // The frozen prefix (layers 0..li) in a scratch root; always restored.
    let tmp = scratch_root(ctx, std::mem::take(layers));
    let bounds = candidate_bounds(ctx, &tmp, relu_op, n, &cand, &mut st);
    *layers = tmp.layers;
    let (lb, ub) = bounds.map_err(Stop::Error)?;
    let mut crossed = 0usize;
    for (col, &j) in cand.iter().enumerate() {
        let (mut lbj, mut ubj) = (lb[col], ub[col]);
        if let Some(Mutation::ShiftBackward(d)) = ctx.mutation {
            lbj += d;
            ubj -= d;
        }
        if !(lbj.is_finite() && ubj.is_finite()) {
            // Failed closed (or never computed): the reference box stands.
            st.unavailable += 1;
            continue;
        }
        let nl = l[j].max(lbj);
        let nu = u[j].min(ubj);
        if nl > nu {
            crossed += 1;
            continue;
        }
        if nl > l[j] || nu < u[j] {
            st.tightened += 1;
            if nl >= 0.0 || nu <= 0.0 {
                st.stabilized += 1;
            }
            l[j] = nl;
            u[j] = nu;
        }
    }
    if crossed > 0 {
        return Err(Stop::Declined(Decline::Crossed {
            layer: li,
            count: crossed,
        }));
    }
    Ok((l, u, st))
}

/// The certified backward bounds `(lb, ub)` of the candidates `cand` (flat
/// indices on relu op `relu_op`'s pre-activation, width `n_pre`), in `cand`
/// order. An entry the passes could not certify is NaN (the caller keeps the
/// reference box there). `Err` only on the deadline.
///
/// COLUMN-LOCAL, failures included: a multi-column chunk that fails closed
/// (the walk's non-finite firewall, an engine refusal) OR that the patch plan
/// cannot express is retried one column at a time, and only a single column
/// the plan cannot express runs dense. So which candidates a patch walk
/// certifies, which run dense and which end up unavailable never depends on
/// the chunk width, the working set or the thread count (R6). (The walk's
/// `None` is column-dependent in principle: its embed check runs per column
/// anchor. Sending a whole multi-column `None` chunk dense would let one
/// column's refusal move its chunk-mates from the patch bound to the looser
/// dense bound, i.e. make published bits depend on the grouping.)
fn candidate_bounds(
    ctx: &Ctx<'_>,
    tmp: &RootGates,
    relu_op: usize,
    n_pre: usize,
    cand: &[usize],
    st: &mut LayerStats,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let m = cand.len();
    let mut lb = vec![f64::NAN; m];
    let mut ub = vec![f64::NAN; m];
    // Even a single dense fallback column must fit. Reference boxes remain
    // certified when tightening is unavailable; a minimum chunk width of one
    // is not permission to exceed the admitted working set.
    let rows = ctx.net.tsize[..=relu_op.min(ctx.net.tsize.len() - 1)]
        .iter()
        .try_fold(0usize, |sum, &rows| sum.checked_add(rows));
    if rows
        .and_then(|n| n.checked_mul(DENSE_BYTES_PER_ROW))
        .is_none_or(|n| n > ctx.budget)
    {
        st.plan_refused = Some("single_column_memory");
        if !ctx.net.root_tableau_within_cap() {
            return Err(NyError::InvalidSpec(
                "wide patch source: whole-route single-column working-set refusal".into(),
            ));
        }
        return Ok((lb, ub));
    }
    let mut dense: Vec<Range<usize>> = Vec::new();
    match PatchPlan::build(ctx.net, relu_op, PATCH_RATIO) {
        Ok(plan) => {
            // Width and concurrency so that all concurrent chunks together
            // stay inside the working set. Scheduling only: every column's
            // arithmetic is column-local.
            let col_bytes = plan.bytes_per_fused_col();
            if col_bytes > ctx.budget {
                st.plan_refused = Some("patch_column_memory");
                if !ctx.net.root_tableau_within_cap() {
                    return Err(NyError::InvalidSpec(
                        "wide patch source: whole-route patch-column working-set refusal".into(),
                    ));
                }
                dense.push(0..m);
            } else {
                let width = (ctx.budget / ctx.threads / col_bytes).clamp(1, PATCH_CHUNK_MAX);
                let width = ctx.chunk_cap.map_or(width, |cap| width.min(cap));
                let par =
                    (ctx.budget / width.saturating_mul(col_bytes).max(1)).clamp(1, ctx.threads);
                let mut ranges = split_range(0..m, width);
                // Two rounds: the chunks, then (column by column) every chunk of
                // more than one column that failed closed or was not expressed.
                for round in 0..2 {
                    let mut failed: Vec<Range<usize>> = Vec::new();
                    for wave in ranges.chunks(par) {
                        check_deadline(ctx.deadline)?;
                        let outs: Vec<Option<Option<(Vec<f64>, Vec<f64>)>>> = wave
                            .par_iter()
                            .map(|r| {
                                // CPU route: the device never certifies (R2).
                                let off = GpuConvRoute::Off;
                                plan.prefix_bounds(ctx.net, tmp, &cand[r.clone()], None, Some(&off))
                                    .map(|res| res.ok().map(|pair| (pair.lb, pair.ub)))
                            })
                            .collect();
                        for (r, out) in wave.iter().zip(outs) {
                            match out {
                                Some(Some((l, u))) if l.len() == r.len() && u.len() == r.len() => {
                                    lb[r.clone()].copy_from_slice(&l);
                                    ub[r.clone()].copy_from_slice(&u);
                                    st.patch_cols += r.len();
                                }
                                // Failed closed: retry its columns singly; a
                                // single column that fails keeps its reference.
                                Some(_) => {
                                    if round == 0 && r.len() > 1 {
                                        failed.extend(split_range(r.clone(), 1));
                                    }
                                }
                                // The plan cannot express this chunk: its columns
                                // singly first (R6, see the docs above), then a
                                // single column it cannot express runs dense below.
                                None => {
                                    if round == 0 && r.len() > 1 {
                                        failed.extend(split_range(r.clone(), 1));
                                    } else {
                                        dense.push(r.clone());
                                    }
                                }
                            }
                        }
                    }
                    if failed.is_empty() {
                        break;
                    }
                    ranges = failed;
                }
            }
        }
        Err(why) => {
            st.plan_refused = Some(why);
            dense.push(0..m);
        }
    }
    if !dense.is_empty() {
        let eng = BackwardEngine::new(ctx.net, tmp).with_gpu_conv_route(GpuConvRoute::Off);
        let width = dense_width(ctx, relu_op);
        for big in dense {
            for r in split_range(big, width) {
                check_deadline(ctx.deadline)?;
                let pieces: Vec<(Range<usize>, Result<(Vec<f64>, Vec<f64>)>)> =
                    match dense_pair(&eng, relu_op, n_pre, &cand[r.clone()]) {
                        Ok(pair) => vec![(r, Ok(pair))],
                        // Column by column, as the patch rounds above, polling
                        // the deadline between columns (up to a whole dense
                        // chunk of single passes would otherwise run unpolled).
                        Err(_) if r.len() > 1 => {
                            let mut singles = Vec::with_capacity(r.len());
                            for one in split_range(r, 1) {
                                check_deadline(ctx.deadline)?;
                                let pair = dense_pair(&eng, relu_op, n_pre, &cand[one.clone()]);
                                singles.push((one, pair));
                            }
                            singles
                        }
                        Err(e) => vec![(r, Err(e))],
                    };
                for (piece, pair) in pieces {
                    if let Ok((l, u)) = pair {
                        if l.len() == piece.len() && u.len() == piece.len() {
                            lb[piece.clone()].copy_from_slice(&l);
                            ub[piece.clone()].copy_from_slice(&u);
                            st.dense_cols += piece.len();
                        }
                    }
                }
            }
        }
    }
    Ok((lb, ub))
}

/// One certified dense Lower+Upper prefix pair on identity seeds at `cols`
/// (the `#backward-interm` dense arm, unchanged).
fn dense_pair(
    eng: &BackwardEngine<'_>,
    relu_op: usize,
    n_pre: usize,
    cols: &[usize],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut s = Array2::<f64>::zeros((n_pre, cols.len()));
    for (col, &j) in cols.iter().enumerate() {
        s[[j, col]] = 1.0;
    }
    // Identity seed is exact: the engine zero-fills its error lane.
    let seed = Seed { s, e: None };
    let plo = eng.run_prefix(&seed, relu_op, LaneDir::Lower)?;
    let pup = eng.run_prefix(&seed, relu_op, LaneDir::Upper)?;
    Ok((eng.concretize_lower(&plo), eng.concretize_upper(&pup)))
}

/// The frozen prefix in a scratch root the engines read: the build's box and
/// mode, the per-column error lane, no tableau, no policy.
fn scratch_root(ctx: &Ctx<'_>, layers: Vec<LayerGates>) -> RootGates {
    let mode = if matches!(ctx.mutation, Some(Mutation::ParityScratch)) {
        RoundMode::Parity
    } else {
        RoundMode::Outward
    };
    RootGates {
        mode,
        mid: ctx.mid.clone(),
        rad: ctx.rad.clone(),
        xabs: ctx.xabs.clone(),
        layers,
        lo: ctx.lo.to_vec(),
        hi: ctx.hi.to_vec(),
        backward_interm_execution: BackwardIntermExecutionEvidence::default(),
        root_precision_policy: super::RootPrecisionRuntimePolicy::default(),
        root_precision_execution: RootPrecisionExecutionEvidence::default(),
        // Prefix-only scratch root: never reaches the head or `root_eval`.
        spec_alpha_policy: None,
        head_alpha: None,
        root_eval_memo: super::bab::RootEvalMemo::default(),
        head_group_source: None,
    }
}

/// `Err(DeadlineExceeded)` once the deadline has passed.
fn check_deadline(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|dl| Instant::now() > dl) {
        return Err(NyError::DeadlineExceeded(
            "margin_row root patch-backward deadline".into(),
        ));
    }
    Ok(())
}

/// `r` in consecutive pieces of at most `width`.
fn split_range(r: Range<usize>, width: usize) -> Vec<Range<usize>> {
    let w = width.max(1);
    let end = r.end;
    r.step_by(w).map(|s| s..(s + w).min(end)).collect()
}

/// The layer working set: `WORKING_SET_BYTES`, or a quarter of the observed
/// memory headroom when that is smaller (never below `WORKING_SET_FLOOR`).
fn working_set_bytes() -> usize {
    let cap = match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(headroom) => (headroom / 4).clamp(WORKING_SET_FLOOR, WORKING_SET_BYTES),
        None => WORKING_SET_BYTES,
    };
    usize::try_from(cap).unwrap_or(usize::MAX)
}

/// Dense-fallback chunk width: the working set over the prefix tensors' rows.
fn dense_width(ctx: &Ctx<'_>, relu_op: usize) -> usize {
    let last = relu_op.min(ctx.net.tsize.len().saturating_sub(1));
    let rows: usize = ctx.net.tsize[..=last].iter().sum();
    let per_col = rows.saturating_mul(DENSE_BYTES_PER_ROW).max(1);
    let width = (ctx.budget / per_col).clamp(1, DENSE_CHUNK_MAX);
    ctx.chunk_cap.map_or(width, |cap| width.min(cap))
}

// ---------------------------------------------------------------------------
//  Observables
// ---------------------------------------------------------------------------

fn print_layer(li: usize, k: usize, st: &LayerStats, secs: f64) {
    let plan = if st.ref_unst == 0 {
        "-".to_string()
    } else {
        st.plan_refused
            .map_or_else(|| "patch".to_string(), |why| format!("dense({why})"))
    };
    eprintln!(
        "[root-box] layer={li} op={k} neurons={} ref_unst={} cols={} patch_cols={} dense_cols={} \
         unavailable={} plan={plan} tightened={} stabilized={} unst={} secs={secs:.3}",
        st.neurons,
        st.ref_unst,
        st.ref_unst,
        st.patch_cols,
        st.dense_cols,
        st.unavailable,
        st.tightened,
        st.stabilized,
        st.unst,
    );
}

fn print_done(
    totals: &Totals,
    precision_policy: super::RootPrecisionRuntimePolicy,
    bi_policy: super::BackwardIntermRuntimePolicy,
    retain: bool,
    complete: bool,
    t0: Instant,
) {
    let mark = |on: bool, token: &'static str| if on { token } else { "-" };
    eprintln!(
        "[root-box] source=patch_backward complete={} layers={} ref_unst={} cols={} \
         patch_cols={} dense_cols={} unavailable={} plan_refused={} stabilized={} unst={} \
         crossed=0 retain={} clip_rows=none bi={} ibp_seed={} root_f32={} certify_s={:.3}",
        u8::from(complete),
        totals.layers,
        totals.ref_unst,
        totals.ref_unst,
        totals.patch_cols,
        totals.dense_cols,
        totals.unavailable,
        totals.plan_refused,
        totals.stabilized,
        totals.unst,
        mark(retain, "empty"),
        mark(bi_policy.enabled, "subsumed"),
        mark(precision_policy.ibp_seed, "subsumed"),
        mark(
            precision_policy.root_f32 || precision_policy.tableau_f32,
            "inert"
        ),
        t0.elapsed().as_secs_f64(),
    );
}

/// The terminal line of an armed build REFUSED for a composition conflict
/// (printed by the root build after the tableau ran, so `certify_s` is its
/// wall). The source never ran: `patch_s` is absent by construction.
pub(crate) fn report_refused(conflicts: PatchBackwardConflicts, tableau_secs: f64, ok: bool) {
    eprintln!(
        "[root-box] refused reason={} source=tableau tableau_ok={} certify_s={tableau_secs:.3}",
        conflicts.reason(),
        u8::from(ok),
    );
}

/// The terminal line of an armed build that DECLINED to the tableau (printed
/// by the root build after the tableau ran, so `certify_s` is its wall).
pub(crate) fn report_declined(declined: &Declined, tableau_secs: f64, ok: bool) {
    eprintln!(
        "[root-box] source=tableau declined={} patch_s={:.3} tableau_ok={} \
         certify_s={tableau_secs:.3}",
        declined.why.token(),
        declined.secs,
        u8::from(ok),
    );
}
