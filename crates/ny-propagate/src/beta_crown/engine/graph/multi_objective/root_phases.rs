// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #root-phases — step 1 of decomposing `evaluate_root`: make the sequence
//! DATA.
//!
//! Plan and soundness argument: `docs/DECOMPOSE_EVALUATE_ROOT_2026-08-10.md`.
//!
//! `evaluate_root` is 2,271 lines of straight-line gated stages mutating one
//! `bootstrap`. The marginal-value scheduler ([`ny_core::phase_scheduler`]) has
//! nowhere to stand until that implicit sequence is an explicit list. This
//! module is that list.
//!
//! The table began as a declarative-only step. Several bodies are now extracted
//! and dispatched in place; producer phases that remain interleaved with other
//! root state are still called at their exact root.rs site. `dispatch()` being a
//! no-op for one of those variants means “not wired through the generic table,”
//! not “its body has not been extracted.”
//!
//! ## Why only these eleven
//!
//! Not a preference — a consequence. Every phase here publishes through a
//! **shrink-only** intersect (`intersect_interm_into_stored`,
//! `shrink_only_intersect`), and the decomposition theorem says:
//!
//! > Every ordering of shrink-only tighteners yields a sound enclosure, because
//! > each intersects a certified candidate into an already-certified box.
//! > Orderings differ only in TIGHTNESS, never in soundness.
//!
//! That asymmetry is what lets a scheduler reorder them without a
//! per-permutation soundness proof. Stages that are *not* shrink-only — the
//! mandatory root objective pass, the RAII scope publishers, the dd-zonotope
//! early exit that can return `Verified`, the installers consuming its output —
//! are excluded by the theorem, not by taste. See [`RootTightenPhase`].

/// A shrink-only intermediate tightener in the root pipeline.
///
/// Ordered as they appear in `evaluate_root` today. [`ORDER`] is the contract:
/// walking it reproduces the current sequence exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RootTightenPhase {
    /// `NY_STABILIZE=<secs>` — per-neuron ReLU fixing.
    StabilizeAndFix,
    /// `NY_DD_ZONO_INTERM=1` — intersect the dd-zonotope intermediate map.
    DdZonoIntermIntersect,
    /// `bab.mip_refine_intermediates` (#relusplitter-lp-refine, wave-6 C7) —
    /// intersect the attached refiner's RIGOROUS LP pre-activation bounds.
    /// Needs the verifier's runtime-attached refiner, so it runs through
    /// [`RootTightenPhase::run_mip_refine_in_place`], never the generic table.
    MipRefineInterm,
    /// `NY_FCHEAD_TIGHTEN=1` — dense-head pre-activation tighten.
    FcHeadTighten,
    /// `NY_ROOT_INTERM_ALPHA=1` — broad per-node α-CROWN re-bound.
    RootIntermAlpha,
    /// `NY_ROOT_JOINT_INTERM_ALPHA=1` — joint per-target α, with an admission
    /// floor on remaining budget.
    RootJointIntermAlpha,
    /// `NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN=1` — one atomic sweep over the
    /// complete bounded demanded target census.
    ComprehensiveGpuIntermCrown,
    /// `NY_ROOT_WIDE_DEMANDED_INTERM_CROWN=1` — one demanded wide target,
    /// ranked by crossing mass and downstream unstable-ReLU leverage.
    WideDemandedIntermCrown,
    /// `config.root_sparse_interm_crown` — sparse crossing-row CROWN. Armed in
    /// production via the margin-row reserve route.
    SparseIntermCrown,
    /// `NY_ROOT_PHASE_RESIDENT_CROWN=1` — deferred unified dense-head plus
    /// comprehensive resident transaction. Ownership is resolved at the old
    /// comprehensive site; execution remains here after sparse prerequisites.
    PhaseResidentCrown,
    /// `root_crown_interm_dense_head` — the dense-head CROWN tightener.
    /// Measured at ~0.5 s to take the cifar100 root from 94/99 to 98/99 rows
    /// (`Gemm_56` width 327.98 → 164.15). Armed by the preset.
    DenseHeadTighten,
}

/// Today's execution order. Walking this must reproduce `evaluate_root`'s
/// current sequence exactly — that is what makes step 1 behaviour-preserving
/// and what the extraction will be checked against.
pub(crate) const ORDER: &[RootTightenPhase] = &[
    RootTightenPhase::StabilizeAndFix,
    RootTightenPhase::DdZonoIntermIntersect,
    RootTightenPhase::MipRefineInterm,
    RootTightenPhase::FcHeadTighten,
    RootTightenPhase::RootIntermAlpha,
    RootTightenPhase::RootJointIntermAlpha,
    RootTightenPhase::ComprehensiveGpuIntermCrown,
    RootTightenPhase::WideDemandedIntermCrown,
    RootTightenPhase::SparseIntermCrown,
    RootTightenPhase::PhaseResidentCrown,
    RootTightenPhase::DenseHeadTighten,
];

impl RootTightenPhase {
    /// Stable identifier, used for telemetry and as the scheduler's phase name.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::StabilizeAndFix => "stabilize_and_fix",
            Self::DdZonoIntermIntersect => "ddzono_interm_intersect",
            Self::MipRefineInterm => "mip_refine_interm",
            Self::FcHeadTighten => "fchead_tighten",
            Self::RootIntermAlpha => "root_interm_alpha",
            Self::RootJointIntermAlpha => "root_joint_interm_alpha",
            Self::ComprehensiveGpuIntermCrown => "comprehensive_gpu_interm_crown",
            Self::WideDemandedIntermCrown => "wide_demanded_interm_crown",
            Self::SparseIntermCrown => "sparse_interm_crown",
            Self::PhaseResidentCrown => "phase_resident_crown",
            Self::DenseHeadTighten => "dense_head_tighten",
        }
    }

    /// Whether this phase is armed in the shipped cifar100 configuration.
    ///
    /// Recorded because it bounds what scheduling them can be worth: the
    /// schedulable set is eleven phases, but only the armed ones cost or buy
    /// anything today. Documenting that here keeps the next reader from
    /// over-estimating the payoff, which the plan doc is explicit about.
    pub(crate) const fn armed_in_cifar100_preset(self) -> bool {
        matches!(self, Self::SparseIntermCrown | Self::DenseHeadTighten)
    }

    /// Every phase in this enum publishes through a shrink-only intersect.
    ///
    /// This is the decomposition theorem's precondition, stated as code so a
    /// future variant cannot be added without confronting it. A phase that is
    /// not shrink-only does not belong in this enum at all — it belongs in the
    /// non-schedulable set.
    pub(crate) const fn is_shrink_only(self) -> bool {
        true
    }
}

// ===========================================================================
// Step 2: the context and the dispatch
// ===========================================================================

use crate::beta_crown::engine::graph::propagation::batched::interm_refine::{
    root_joint_tighten_relu_preactivations_with_deadline_gpu, RootJointTightenReport,
};
use crate::network::GraphNetwork;
// Imported by NAME so extracted bodies compile verbatim -- no path rewriting,
// which is what corrupted a fully-qualified path on the previous attempt.
use super::root::{
    bounded_root_crown_interm_deadline, provides_usable_sound_root_sparse_crown,
    root_comprehensive_gpu_interm_crown_policy, root_interm_engine_route,
    root_sparse_interm_crown_policy_from_env, root_wide_demanded_interm_crown_policy,
    RootIntermEngineRoute, RootPhaseResidentCrownPolicy,
};
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

/// What a phase produces besides its bound mutation.
///
/// Some tighteners are shrink-only in their BOUND MUTATION and still
/// **producers** in their control flow: dense and resident phases set the
/// dense-head selection plus realized element/target counts. `evaluate_root`
/// folds that one output into the shared stale-state summary used by the root
/// objective, paired-alpha builders, and root-domain handoff.
///
/// Returning `()` from `dispatch` would have dropped both silently — leaving
/// the flag `false` and the count `0`, changing which downstream passes run,
/// **and compiling**. This channel is what makes those phases extractable at
/// all.
///
/// It is also exactly what the scheduler's value model needs: the `tightened_*`
/// counts are the realised-value signal. The return channel is not overhead
/// added for the extraction; it is the thing that was going to be required.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhaseOutput {
    /// Set by the dense-head stage; gates later passes.
    pub(crate) dense_head_stage_selected: bool,
    /// Elements tightened by a CROWN-interm stage.
    pub(crate) tightened_elements: usize,
    /// Targets tightened by a per-target stage.
    pub(crate) tightened_targets: usize,
}

impl PhaseOutput {
    /// Fold one phase's output into the running total, so the driver can walk
    /// the table and hand `evaluate_root` the same locals it keeps today.
    pub(crate) fn merge(&mut self, other: Self) {
        self.dense_head_stage_selected |= other.dense_head_stage_selected;
        self.tightened_elements += other.tightened_elements;
        self.tightened_targets += other.tightened_targets;
    }

    /// One realized-value summary shared by telemetry and stale-state routing.
    pub(crate) const fn bounds_changed(self) -> bool {
        self.tightened_elements > 0 || self.tightened_targets > 0
    }
}

/// Outcome of the deferred phase-resident slot.
///
/// Only admission and a clean predispatch decline may reach the established
/// dense fallback. `Completed` means one request was accepted and validated,
/// including a legitimate zero-tightening result; `Failed` means an accepted
/// or publication transaction failed and therefore forbids a second route.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhaseResidentRun {
    AdmissionDeclined,
    CleanDecline,
    Completed(PhaseOutput),
    #[default]
    Failed,
}

impl PhaseResidentRun {
    pub(crate) const fn permits_legacy_dense_fallback(self) -> bool {
        matches!(self, Self::AdmissionDeclined | Self::CleanDecline)
    }

    pub(crate) const fn output(self) -> PhaseOutput {
        match self {
            Self::Completed(output) => output,
            Self::AdmissionDeclined | Self::CleanDecline | Self::Failed => PhaseOutput {
                dense_head_stage_selected: false,
                tightened_elements: 0,
                tightened_targets: 0,
            },
        }
    }
}

/// Everything a shrink-only tightener needs from `evaluate_root`.
///
/// One `&mut` to the bootstrap, held by the context rather than by each phase.
/// That is what makes the dispatch borrow-check: only one phase runs at a time,
/// so only one `&mut` exists at a time. A `Vec<Box<dyn Phase>>` cannot express
/// this — each trait object would have to capture `&mut bootstrap` and they
/// would overlap.
///
/// **The split borrow is load-bearing.** `stabilize_and_fix` takes
/// `&bootstrap.root_alpha_state` and `&mut bootstrap.initial_node_bounds` in a
/// single call. That compiles only because the two field paths are disjoint,
/// and it keeps compiling through this context only while both stay *direct
/// field paths off the same place expression*. Adding an accessor
/// (`ctx.alpha()`) collapses them into one borrow of `*self` and breaks it.
/// Do not restructure the call.
pub(crate) struct RootTightenCtx<'a> {
    pub(crate) graph: &'a GraphNetwork,
    pub(crate) input: &'a BoundedTensor,
    pub(crate) objectives: &'a [Vec<f32>],
    pub(crate) engine: Option<&'a dyn GemmEngine>,
    pub(crate) deadline: Option<std::time::Instant>,
    pub(crate) bootstrap: &'a mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
    /// Whether the root-interm factory engine route was requested.
    pub(crate) factory_requested: bool,
    /// The verifier config, for phases whose policy resolves from it.
    pub(crate) config: &'a crate::beta_crown::BetaCrownConfig,
    /// The dd-zonotope root result, when one was produced.
    pub(crate) dd_zono: Option<&'a super::dd_zono_root::DdZonoRootResult>,
}

impl RootTightenPhase {
    /// Run one phase IN PLACE, building a short-lived context around the call.
    ///
    /// This is the shape the decomposition needs, and the shape a hoisted walk
    /// got wrong. `ORDER` is a *description* of the sequence; it is not a place
    /// to execute it from. The phases are separated in `evaluate_root` by code
    /// that reads and writes the same state they do, so relocating a phase to a
    /// single walk site changes the program — measured: bounds identical
    /// (624.0831, the tighteners commute) but the verdict moved timeout →
    /// unknown and elapsed 82.7 → 55.1 against a baseline reproducible to
    /// ±0.03 s.
    ///
    /// The theorem licenses reordering tighteners AMONG THEMSELVES, not moving
    /// one across unrelated code. Dispatching in place keeps that distinction.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_in_place(
        self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
        bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
        config: &crate::beta_crown::BetaCrownConfig,
        factory_requested: bool,
        dd_zono: Option<&super::dd_zono_root::DdZonoRootResult>,
    ) -> PhaseOutput {
        let mut ctx = RootTightenCtx {
            graph,
            input,
            objectives,
            engine,
            deadline,
            bootstrap,
            factory_requested,
            config,
            dd_zono,
        };
        if !self.admitted() {
            return PhaseOutput::default();
        }
        let started = std::time::Instant::now();
        let out = self.dispatch(&mut ctx);
        self.report_value(out, started.elapsed());
        out
    }

    /// Run the deferred unified transaction through the same admission and
    /// realized-value machinery as every extracted root phase.
    pub(crate) fn run_phase_resident_in_place(
        self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
        bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
        policy: RootPhaseResidentCrownPolicy,
    ) -> PhaseResidentRun {
        debug_assert_eq!(self, Self::PhaseResidentCrown);
        if self != Self::PhaseResidentCrown {
            return PhaseResidentRun::Failed;
        }
        if !self.admitted() {
            return PhaseResidentRun::AdmissionDeclined;
        }
        let started = std::time::Instant::now();
        let outcome = phase_resident_crown(graph, input, engine, deadline, bootstrap, policy);
        self.report_value(outcome.output(), started.elapsed());
        outcome
    }

    /// #relusplitter-lp-refine (wave-6 C7): run the verifier's attached root
    /// intermediate-bound refiner through the same admission and
    /// realized-value machinery as every other phase. A producer: its output
    /// is folded into the root's stale-box summary by `evaluate_root`.
    pub(crate) fn run_mip_refine_in_place(
        self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<std::time::Instant>,
        bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
        refiner: &dyn crate::beta_crown::root_interm_refine::RootIntermRefiner,
    ) -> PhaseOutput {
        debug_assert_eq!(self, Self::MipRefineInterm);
        if self != Self::MipRefineInterm || !self.admitted() {
            return PhaseOutput::default();
        }
        let started = std::time::Instant::now();
        let out = mip_refine_interm(graph, input, deadline, bootstrap, refiner);
        self.report_value(out, started.elapsed());
        out
    }

    /// Run one phase. Byte-identical to the inline block it replaces: the gate,
    /// the argument order, and the telemetry are all preserved verbatim.
    ///
    /// A phase whose gate is off is a no-op that does not touch the bounds —
    /// which is what makes running the table equivalent to the straight-line
    /// sequence.
    pub(crate) fn dispatch(self, ctx: &mut RootTightenCtx<'_>) -> PhaseOutput {
        match self {
            Self::StabilizeAndFix => stabilize_and_fix(ctx),
            Self::DdZonoIntermIntersect => ddzono_interm_intersect(ctx),
            Self::FcHeadTighten => fchead_tighten(ctx),
            Self::RootIntermAlpha => root_interm_alpha(ctx),
            Self::DenseHeadTighten => dense_head_tighten(ctx),
            // Extracted bodies whose execution is still pinned to an exact
            // interleaved root.rs site. Generic-table dispatch is intentionally
            // a no-op until their producer outputs and intervening state are
            // carried through the table driver.
            Self::RootJointIntermAlpha
            | Self::ComprehensiveGpuIntermCrown
            | Self::WideDemandedIntermCrown
            | Self::SparseIntermCrown
            | Self::PhaseResidentCrown => PhaseOutput::default(),
            // Needs the verifier's attached refiner, which the generic context
            // does not carry: `run_mip_refine_in_place` is its only runner.
            Self::MipRefineInterm => PhaseOutput::default(),
        }
    }

    /// Whether this phase produces values `evaluate_root` branches on, beyond
    /// its bound mutation. Extracting a producer without carrying its output
    /// compiles and silently changes control flow, so this is recorded per
    /// phase rather than rediscovered.
    pub(crate) const fn is_producer(self) -> bool {
        matches!(
            self,
            Self::MipRefineInterm
                | Self::DenseHeadTighten
                | Self::PhaseResidentCrown
                | Self::ComprehensiveGpuIntermCrown
                | Self::WideDemandedIntermCrown
                | Self::SparseIntermCrown
                | Self::RootJointIntermAlpha
        )
    }

    /// Whether this phase's body has been moved out of `evaluate_root` yet.
    ///
    /// Now true for every variant: the decomposition is complete. Kept rather
    /// than deleted because it is what the progress test asserts against, and a
    /// future phase added to the enum starts out `false` until it is moved.
    pub(crate) const fn is_extracted(self) -> bool {
        matches!(
            self,
            Self::StabilizeAndFix
                | Self::DdZonoIntermIntersect
                | Self::MipRefineInterm
                | Self::FcHeadTighten
                | Self::RootIntermAlpha
                | Self::DenseHeadTighten
                | Self::PhaseResidentCrown
                | Self::ComprehensiveGpuIntermCrown
                | Self::WideDemandedIntermCrown
                | Self::SparseIntermCrown
                | Self::RootJointIntermAlpha
        )
    }
}

/// Verbatim move of `root.rs`'s stabilize block.
///
/// NOTE the gate: there is none at this site. The call is unconditional and the
/// `if let` guards only the log — the real gate lives inside the callee, which
/// returns `None` without touching the bounds when `NY_STABILIZE` is unset. A
/// dispatch that invented a call-site gate would diverge from today.
fn stabilize_and_fix(ctx: &mut RootTightenCtx<'_>) -> PhaseOutput {
    if let Some(stabilize_report) =
        crate::beta_crown::engine::graph::shared::stabilize::stabilize_and_fix_from_env(
            ctx.graph,
            ctx.input,
            ctx.objectives,
            ctx.engine,
            ctx.deadline,
            ctx.bootstrap.root_alpha_state.as_ref(),
            &mut ctx.bootstrap.initial_node_bounds,
        )
    {
        tracing::info!(
            "stabilize-and-fix: {} round(s), {} neuron-fix(es)",
            stabilize_report.rounds,
            stabilize_report.fixed.len()
        );
    }
    // Not a producer: `stabilize_report` is consumed by its own log and no
    // outer local survives the block. Verified against root.rs.
    PhaseOutput::default()
}

/// Verbatim move of the `#dd-zono-interm` block.
///
/// The gate at this site is the presence of a dd-zonotope result with a
/// non-empty intermediate map — `NY_DD_ZONO_INTERM` lives upstream and only
/// decides whether that map gets populated at all.
fn ddzono_interm_intersect(ctx: &mut RootTightenCtx<'_>) -> PhaseOutput {
    if let Some(result) = ctx.dd_zono {
        if !result.margin.interm.is_empty() {
            let tightened = super::root::intersect_interm_into_stored(
                &mut ctx.bootstrap.initial_node_bounds,
                &result.margin.interm,
            );
            tracing::info!(
                "#dd-zono-interm: intersected {} certified node enclosure(s) into the stored bounds",
                tightened
            );
        }
    }
    // Not a producer: `tightened` is consumed by its own log.
    PhaseOutput::default()
}

/// #relusplitter-lp-refine (wave-6 C7): ask the attached refiner for certified
/// pre-activation enclosures over the stored boxes, intersect them SHRINK-ONLY
/// through the `#dd-zono-interm` primitive (per node all-or-nothing: arity,
/// non-finite or crossed candidates are refused and the stored box is kept),
/// and print the fixed-schema `[mip-refine] root ...` line on every call.
///
/// A producer: `tightened_targets` is the number of stored nodes that actually
/// shrank, which `evaluate_root` folds into `root_intermediate_bounds_changed`
/// so the root objective pass re-runs against the tightened boxes instead of
/// reusing state built for the old ones.
fn mip_refine_interm(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    deadline: Option<std::time::Instant>,
    bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
    refiner: &dyn crate::beta_crown::root_interm_refine::RootIntermRefiner,
) -> PhaseOutput {
    use crate::beta_crown::root_interm_refine::{root_interm_refine_line, RootIntermRefineRequest};
    let outcome = refiner.refine(&RootIntermRefineRequest {
        graph,
        input_bounds: input,
        node_bounds: &bootstrap.initial_node_bounds,
        deadline,
    });
    let nodes_intersected = if outcome.candidates.is_empty() {
        0
    } else {
        super::root::intersect_interm_into_stored(
            &mut bootstrap.initial_node_bounds,
            &outcome.candidates,
        )
    };
    eprintln!(
        "{}",
        root_interm_refine_line(refiner.name(), &outcome.telemetry, nodes_intersected)
    );
    PhaseOutput {
        dense_head_stage_selected: false,
        tightened_elements: 0,
        tightened_targets: nodes_intersected,
    }
}

/// Verbatim move of the `NY_FCHEAD_TIGHTEN` block. Non-producer.
fn fchead_tighten(ctx: &mut RootTightenCtx<'_>) -> PhaseOutput {
    if std::env::var("NY_FCHEAD_TIGHTEN").ok().as_deref() == Some("1") {
        if let Some(alpha) = ctx.bootstrap.root_alpha_state.as_ref() {
            let now = std::time::Instant::now();
            // Grace cap (env-tunable for measurement; default 12s). The single
            // dense-head backward is ~1-2s on the sound GPU resnet path; cap it
            // and leave the bulk of the remaining budget for the root spec pass
            // and BaB. Only fires while the global wall-clock still has room.
            let grace_cap = std::env::var("NY_FCHEAD_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(12);
            let global_remaining = ctx
                .deadline
                .map(|g| g.saturating_duration_since(now))
                .unwrap_or_else(|| std::time::Duration::from_secs(grace_cap));
            // Reserve at least half the remaining budget for root-spec + BaB.
            let slice =
                std::time::Duration::from_secs(grace_cap).min(global_remaining.mul_f32(0.5));
            if slice >= std::time::Duration::from_secs(2) {
                let fc_deadline = Some(now + slice);
                if let Ok(exec_order) = ctx.graph.exec_order() {
                    let exec_order = exec_order.to_vec();
                    ctx.graph.tighten_fc_head_preactivations(
                        ctx.input,
                        &exec_order,
                        alpha,
                        ctx.engine,
                        fc_deadline,
                        &mut ctx.bootstrap.initial_node_bounds,
                    );
                }
            }
        }
    }
    PhaseOutput::default()
}

/// Verbatim move of the `NY_ROOT_INTERM_ALPHA` block. Non-producer.
fn root_interm_alpha(ctx: &mut RootTightenCtx<'_>) -> PhaseOutput {
    if std::env::var("NY_ROOT_INTERM_ALPHA").ok().as_deref() == Some("1") {
        eprintln!(
            "[root-interm-alpha] gate ON, root_alpha_state={}",
            if ctx.bootstrap.root_alpha_state.is_some() {
                "Some"
            } else {
                "None"
            }
        );
        if let Some(alpha) = ctx.bootstrap.root_alpha_state.as_ref() {
            let now = std::time::Instant::now();
            // Grace cap (env-tunable for measurement; default 120s — this pass is
            // the full O(L²) per-node root sweep over ~20 ReLU pre-activations on
            // a deep conv ResNet, far heavier than the single FC-head backward).
            let grace_cap = std::env::var("NY_ROOT_INTERM_ALPHA_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(120);
            let global_remaining = ctx
                .deadline
                .map(|g| g.saturating_duration_since(now))
                .unwrap_or_else(|| std::time::Duration::from_secs(grace_cap));
            // Reserve at least half the remaining budget for root-spec + BaB.
            let slice =
                std::time::Duration::from_secs(grace_cap).min(global_remaining.mul_f32(0.5));
            if slice >= std::time::Duration::from_secs(2) {
                let ria_deadline = Some(now + slice);
                if let Ok(exec_order) = ctx.graph.exec_order() {
                    let exec_order = exec_order.to_vec();
                    ctx.graph.tighten_all_relu_preactivations(
                        ctx.input,
                        &exec_order,
                        alpha,
                        ctx.engine,
                        ria_deadline,
                        &mut ctx.bootstrap.initial_node_bounds,
                    );
                }
            }
        }
    }
    PhaseOutput::default()
}

/// Verbatim move of the dense-head CROWN tightener — the first PRODUCER moved.
///
/// Measured at ~0.5 s to take the cifar100 root from 94/99 to 98/99 rows
/// (`Gemm_56` width 327.98 → 164.15). Its two outputs are read at five later
/// sites in `evaluate_root`, so they leave through `PhaseOutput` rather than by
/// assigning outer locals — which is exactly what §8 said made it unmovable
/// before the channel existed.
fn dense_head_tighten(ctx: &mut RootTightenCtx<'_>) -> PhaseOutput {
    // Default-dark comprehensive CPU treatment. When explicitly armed it owns
    // this root slot completely: a declined all-target transaction does not
    // degrade into the legacy partial/dense-only publication route.
    if crate::beta_crown::engine::graph::propagation::batched::comprehensive_cpu::comprehensive_cpu_enabled()
    {
        return crate::beta_crown::engine::graph::propagation::batched::comprehensive_cpu::run_comprehensive_cpu_intermediate_tighten(
            ctx.graph,
            ctx.input,
            &mut ctx.bootstrap.initial_node_bounds,
            ctx.deadline,
        )
        .map_or_else(PhaseOutput::default, |report| PhaseOutput {
            dense_head_stage_selected: true,
            tightened_elements: report.tightened_elements,
            tightened_targets: report.tightened_targets,
        });
    }
    let mut dense_head_stage_selected = false;
    let tightened_elements = if let Some(policy) =
        super::root::root_crown_interm_policy_from_env(ctx.config)
    {
        dense_head_stage_selected = matches!(
            policy.selection,
            super::root::RootCrownIntermSelection::DenseHead
        );
        let now = std::time::Instant::now();
        if let Some(pass_deadline) =
            bounded_root_crown_interm_deadline(now, ctx.deadline, policy.max_secs)
        {
            match root_interm_engine_route(ctx.engine.is_some(), ctx.factory_requested) {
                RootIntermEngineRoute::Local => super::root::run_root_crown_interm_tighten(
                    ctx.graph,
                    ctx.input,
                    ctx.engine
                        .expect("local intermediate route requires an engine"),
                    ctx.bootstrap,
                    &policy,
                    pass_deadline,
                ),
                RootIntermEngineRoute::Factory => {
                    match crate::sound_f64_gemm::with_engine_deadline(
                        pass_deadline,
                        |factory_engine| {
                            super::root::run_root_crown_interm_tighten(
                                ctx.graph,
                                ctx.input,
                                factory_engine,
                                ctx.bootstrap,
                                &policy,
                                pass_deadline,
                            )
                        },
                    ) {
                        Ok(Some(n_tightened)) => n_tightened,
                        Ok(None) => {
                            eprintln!(
                                "[root-crown-interm-tighten] sound CUDA factory unavailable; \
                             skipping (bounds unchanged)"
                            );
                            0
                        }
                        Err(error) => {
                            eprintln!(
                            "[root-crown-interm-tighten] sound CUDA factory admission refused: \
                             {error}; skipping (bounds unchanged)"
                        );
                            0
                        }
                    }
                }
                RootIntermEngineRoute::Unavailable => {
                    eprintln!("[root-crown-interm-tighten] no GPU engine (need the `ny vnncomp` GPU preset); skipping (bounds unchanged)");
                    0
                }
            }
        } else {
            eprintln!(
            "[root-crown-interm-tighten] no safe ctx.deadline slice remains; skipping (bounds unchanged)"
        );
            0
        }
    } else {
        0
    };
    PhaseOutput {
        dense_head_stage_selected,
        tightened_elements,
        tightened_targets: 0,
    }
}

/// Default-dark comprehensive GPU sweep over the complete eligible census.
///
/// This phase has exclusive ownership when armed. It never falls through to
/// the legacy one-target route, even when the retained backend cleanly declines
/// its complete request, because doing so would silently change an atomic
/// all-target experiment into a different serial verdict schedule.
/// `None` means the lever was unarmed; `Some(count)` records ownership even
/// when the complete route produced no tightening.
/// Fraction of the comprehensive-sweep slice the objective-influence backward
/// may consume. One coefficient backward is cheap next to a 64-chunk sweep, but
/// it runs on a deadline that is already tight, so it is capped rather than
/// trusted: over-run leaves the sweep with the width ordering it always had.
const OBJECTIVE_INFLUENCE_SHARE: f64 = 0.15;

/// Objective-directed row ranking is DEFAULT-ON: the scored entry point exports
/// no `NY_*` variables, so an env-gated improvement cannot fire in
/// competition however well it measures. `NY_ROOT_OBJECTIVE_DIRECTED_ROWS=0`
/// exists to hold the A/B arm that produced the measurement, nothing else.
///
/// DELIVERY SEAM (#lever-reachability): because the shipped default is ON, the
/// typed `bab.root_objective_directed_rows` key is a per-category KILL SWITCH
/// rather than an opt-in — the A/B arm that only an environment variable could
/// hold is now expressible in a preset. Env still wins in both directions.
fn objective_directed_rows_enabled() -> bool {
    crate::preset_gates::root_objective_directed_rows_enabled()
}

/// #joint-interm-grad: per-target `df/dl` weights for the objective-weighted
/// joint tightening lane — the PRODUCER that closes the loop.
///
/// # The loop it closes
///
/// The joint lane ascends a per-start-node alpha to tighten each intermediate
/// target. Unweighted it tightens every target for its own sake, counting a
/// neuron the objective barely reads the same as one the margin turns on. These
/// weights make it tighten each target in proportion to `df/dl` — the indirect
/// gradient term — so the ascent optimizes the FINAL objective through the
/// intermediate bounds rather than optimizing the intermediates as an end.
///
/// Two facts make this cheap rather than a new kernel:
///
/// * `df/dl` is host arithmetic over the objective adjoint at the target, and
///   that adjoint is already materialised by the BaBSR walk — a sink keeps it
///   instead of dropping it, so there is no extra propagation.
/// * `dl/dalpha` comes from the existing device call, because its per-row
///   harvest is degree-1 positive-homogeneous; seeding with `w >= 0` instead of
///   `1.0` returns `SUM_j w_j dl_j/dalpha_k` exactly.
///
/// # Seed choice, stated honestly
///
/// The adjoint is seeded with the MEAN of the objective rows, not the worst
/// straggler: this runs before the root objective pass, so which margin is
/// binding is not yet known here. The mean weights every margin the verdict
/// depends on and costs one backward instead of 99. That makes the weights a
/// principled approximation of `df/dl`, not an exact one — which is acceptable
/// because they are steering data, and a target that is mis-weighted is merely
/// tightened in the wrong proportion, never tightened unsoundly.
///
/// Returns `None` on any failure, which restores the historical unit seed.
fn joint_interm_objective_sensitivity(
    verifier: &crate::beta_crown::BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    objectives: &[Vec<f32>],
    deadline: std::time::Instant,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    let output_dim = objectives.first()?.len();
    if output_dim == 0 || objectives.iter().any(|row| row.len() != output_dim) {
        return None;
    }
    let mut seed = vec![0.0f32; output_dim];
    for row in objectives {
        for (accumulator, value) in seed.iter_mut().zip(row) {
            *accumulator += *value;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = objectives.len() as f32;
    for accumulator in &mut seed {
        *accumulator /= count;
    }

    let shared =
        crate::beta_crown::engine::graph::shared::setup::build_initial_node_bounds_arc(node_bounds);
    let adjoints = verifier
        .objective_adjoints_at_preactivations(
            graph,
            &shared,
            input,
            crate::beta_crown::config::KfsbReduceOp::Max,
            Some(&seed),
            deadline,
        )
        .ok()?;

    let mut out: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::with_capacity(adjoints.len());
    for (pre_name, adjoint) in &adjoints {
        let Some(bounds) = node_bounds.get(pre_name) else {
            continue;
        };
        let flat = bounds.flatten();
        let lower = ndarray::Array1::from_iter(flat.lower().iter().copied());
        let upper = ndarray::Array1::from_iter(flat.upper().iter().copied());
        let (w_l, _w_u) = GraphNetwork::interm_sensitivity_weights(adjoint, &lower, &upper, None);
        // A target whose weights are all zero carries no objective signal; leave
        // it out so the seed site keeps the unit diagonal there rather than
        // seeding an all-zero row, which would harvest nothing at all.
        if w_l.iter().any(|w| *w > 0.0) {
            out.insert(pre_name.clone(), w_l.to_vec());
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// #root-objective-directed-rows: per-neuron influence of each PRE-ACTIVATION
/// node on the objective ensemble, for ranking the sweep's bounded row budget.
///
/// The comprehensive sweep can afford ~128 rows per target out of thousands, and
/// it has always spent that budget on the WIDEST intervals. Width answers "how
/// much slack is here", which is not the question — the question is "how much
/// does the margin care". This replaces the proxy with a measurement: one
/// objective-seeded coefficient backward gives every candidate neuron its signed
/// influence on the objective rows, and the sweep then ranks by
/// `|influence| * width`.
///
/// The seed is the MEAN of the objective rows, not the worst straggler: this runs
/// BEFORE the root objective pass, so which objective is worst is not yet known
/// here. The mean is the cheapest single direction that still weights every
/// margin the verdict depends on; one backward per objective would be 99
/// backwards and cost more than the sweep it is steering.
///
/// Advisory-only, and structurally so: the result reorders a row list and touches
/// nothing else. Every selected row is bounded by the same sound backend sweep
/// and committed by the same shrink-only intersect, so a wrong or stale influence
/// vector can only waste rows. Any failure returns `None`, and the sweep then
/// runs exactly as it did before this existed.
fn objective_row_influence(
    verifier: &crate::beta_crown::BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    objectives: &[Vec<f32>],
    deadline: std::time::Instant,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    let output_dim = objectives.first()?.len();
    if output_dim == 0 || objectives.iter().any(|row| row.len() != output_dim) {
        return None;
    }
    let mut seed = vec![0.0f32; output_dim];
    for row in objectives {
        for (accumulator, value) in seed.iter_mut().zip(row) {
            *accumulator += *value;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = objectives.len() as f32;
    for accumulator in &mut seed {
        *accumulator /= count;
    }

    // The scorer reads an `Arc` view; the root map is plain at this point, so it
    // is lifted once here. Counted inside this phase's own budget and reported by
    // the `[root-objective-directed-rows]` timer below.
    let shared_bounds =
        crate::beta_crown::engine::graph::shared::setup::build_initial_node_bounds_arc(node_bounds);
    let scores = verifier
        .compute_graph_babsr_scores_from_bounds_until(
            graph,
            &shared_bounds,
            input,
            crate::beta_crown::config::KfsbReduceOp::Max,
            Some(&seed),
            None,
            deadline,
        )
        .ok()?;

    // The scorer keys by RELU node; the sweep tightens the PRE-ACTIVATION that
    // feeds it. Remap through the graph so the two agree on what a target is.
    let mut influence: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    for ((relu_name, neuron), parts) in &scores {
        let Some(node) = graph.node(relu_name) else {
            continue;
        };
        let Some(pre_name) = node.inputs.first() else {
            continue;
        };
        let Some(pre_bounds) = node_bounds.get(pre_name.as_str()) else {
            continue;
        };
        let entry = influence
            .entry(pre_name.clone())
            .or_insert_with(|| vec![0.0; pre_bounds.len()]);
        if let Some(slot) = entry.get_mut(*neuron) {
            *slot = parts.main_score.abs().max(parts.backup_score.abs());
        }
    }
    if influence.is_empty() {
        None
    } else {
        Some(influence)
    }
}

pub(super) fn comprehensive_gpu_interm_crown(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
    verifier: &crate::beta_crown::BetaCrownVerifier,
    objectives: &[Vec<f32>],
) -> Option<usize> {
    let config = &verifier.config;
    let policy = root_comprehensive_gpu_interm_crown_policy(config)?;
    let Some(engine) = engine else {
        eprintln!(
            "[root-comprehensive-gpu-interm-crown] no local proof engine; skipping \
             (bounds unchanged)"
        );
        return Some(0);
    };
    let now = std::time::Instant::now();
    let Some(pass_deadline) = bounded_root_crown_interm_deadline(now, deadline, policy.max_secs)
    else {
        eprintln!(
            "[root-comprehensive-gpu-interm-crown] no safe deadline slice remains; \
             skipping (bounds unchanged)"
        );
        return Some(0);
    };
    let Some(gpu) = engine
        .as_gpu_crown_backward()
        .filter(|gpu| gpu.provides_sound_gpu_crown() && gpu.provides_sound_intermediate_sweep())
    else {
        eprintln!(
            "[root-comprehensive-gpu-interm-crown] local engine lacks the authorized typed \
             sweep; skipping (bounds unchanged)"
        );
        return Some(0);
    };

    use crate::beta_crown::engine::graph::propagation::batched::intermediate_sweep::{
        root_comprehensive_gpu_intermediate_sweep, RootIntermediateSweepAttempt,
    };

    // #root-objective-directed-rows. Budgeted out of THIS phase's slice, not on
    // top of it, and capped: steering the row budget must not eat the sweep it
    // steers. Expiry inside the scorer fails closed to `None` = width ordering.
    let influence_deadline = pass_deadline.min(
        now + std::time::Duration::from_secs_f64(
            (pass_deadline - now).as_secs_f64() * OBJECTIVE_INFLUENCE_SHARE,
        ),
    );
    let influence_t0 = std::time::Instant::now();
    let influence = objective_directed_rows_enabled()
        .then(|| {
            objective_row_influence(
                verifier,
                graph,
                input,
                &bootstrap.initial_node_bounds,
                objectives,
                influence_deadline,
            )
        })
        .flatten();
    eprintln!(
        "[root-objective-directed-rows] targets={} elapsed={:.3}s budget={:.3}s \
         (none => width ordering)",
        influence.as_ref().map_or(0, std::collections::HashMap::len),
        influence_t0.elapsed().as_secs_f64(),
        (influence_deadline - now).as_secs_f64(),
    );

    Some(
        match root_comprehensive_gpu_intermediate_sweep(
            graph,
            input,
            gpu,
            pass_deadline,
            policy.min_dim,
            policy.max_dim,
            policy.max_rows_per_target,
            policy.max_targets,
            policy.max_device_bytes,
            policy.chunks,
            influence.as_ref(),
            &mut bootstrap.initial_node_bounds,
        ) {
            RootIntermediateSweepAttempt::Completed(tightened) => tightened,
            RootIntermediateSweepAttempt::CleanDecline => {
                eprintln!(
                    "[root-comprehensive-gpu-interm-crown] complete request cleanly declined; \
                 exclusive phase ownership leaves bounds unchanged"
                );
                0
            }
            RootIntermediateSweepAttempt::Failed => 0,
        },
    )
}

fn phase_resident_crown(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
    policy: RootPhaseResidentCrownPolicy,
) -> PhaseResidentRun {
    let Some(engine) = engine else {
        eprintln!("[root-phase-resident-crown] no local proof engine; clean decline");
        return PhaseResidentRun::CleanDecline;
    };
    let now = std::time::Instant::now();
    let Some(pass_deadline) = bounded_root_crown_interm_deadline(now, deadline, policy.max_secs)
    else {
        eprintln!("[root-phase-resident-crown] no safe deadline slice; clean decline");
        return PhaseResidentRun::CleanDecline;
    };
    let Some(gpu) = engine
        .as_gpu_crown_backward()
        .filter(|gpu| gpu.provides_sound_gpu_crown() && gpu.provides_sound_intermediate_sweep())
    else {
        eprintln!("[root-phase-resident-crown] retained engine lacks typed sweep; clean decline");
        return PhaseResidentRun::CleanDecline;
    };
    use crate::beta_crown::engine::graph::propagation::batched::intermediate_sweep::{
        root_phase_resident_gpu_intermediate_sweep, RootIntermediateSweepAttempt,
    };
    match root_phase_resident_gpu_intermediate_sweep(
        graph,
        input,
        gpu,
        pass_deadline,
        policy.min_comprehensive_dim,
        policy.max_comprehensive_dim,
        policy.max_comprehensive_rows_per_target,
        policy.max_comprehensive_targets,
        policy.max_dense_rows,
        policy.max_device_bytes,
        &mut bootstrap.initial_node_bounds,
    ) {
        RootIntermediateSweepAttempt::Completed(tightened_targets) => {
            PhaseResidentRun::Completed(PhaseOutput {
                dense_head_stage_selected: true,
                tightened_elements: 0,
                tightened_targets,
            })
        }
        RootIntermediateSweepAttempt::CleanDecline => PhaseResidentRun::CleanDecline,
        RootIntermediateSweepAttempt::Failed => PhaseResidentRun::Failed,
    }
}

/// Dark one-target vertical slice for wide demanded root pre-activations.
///
/// The local engine is the sole owner: this deliberately does not fall back to
/// a factory backend. The typed multi-depth sweep is attempted through the
/// exact retained `GpuCrownBackward` accessor and both of its sound capability
/// predicates. Only a clean predispatch decline may fall through to the legacy
/// single-target fold, which must independently satisfy its finite-deadline
/// capability predicate. Any accepted-request or publication failure is a
/// sound no-op and ends the phase.
pub(super) fn wide_demanded_interm_crown(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
) -> usize {
    let Some(policy) = root_wide_demanded_interm_crown_policy() else {
        return 0;
    };
    let Some(engine) = engine else {
        eprintln!(
            "[root-wide-demanded-interm-crown] no local proof engine; skipping \
             (bounds unchanged)"
        );
        return 0;
    };
    let now = std::time::Instant::now();
    let Some(pass_deadline) = bounded_root_crown_interm_deadline(now, deadline, policy.max_secs)
    else {
        eprintln!(
            "[root-wide-demanded-interm-crown] no safe deadline slice remains; \
             skipping (bounds unchanged)"
        );
        return 0;
    };
    if policy.max_rows == 0 || policy.max_targets == 0 {
        return 0;
    }

    if let Some(gpu) = engine
        .as_gpu_crown_backward()
        .filter(|gpu| gpu.provides_sound_gpu_crown() && gpu.provides_sound_intermediate_sweep())
    {
        use crate::beta_crown::engine::graph::propagation::batched::intermediate_sweep::{
            root_wide_demanded_intermediate_sweep, RootIntermediateSweepAttempt,
        };
        match root_wide_demanded_intermediate_sweep(
            graph,
            input,
            gpu,
            pass_deadline,
            policy.min_dim,
            policy.max_dim,
            policy.max_rows,
            policy.max_targets,
            policy.max_preflights,
            policy.max_device_bytes,
            &mut bootstrap.initial_node_bounds,
        ) {
            RootIntermediateSweepAttempt::Completed(tightened) => return tightened,
            RootIntermediateSweepAttempt::Failed => return 0,
            RootIntermediateSweepAttempt::CleanDecline => {
                eprintln!(
                    "[root-wide-demanded-interm-crown] typed sweep cleanly declined; \
                     considering finite-deadline legacy fallback"
                );
            }
        }
    }

    let Some(engine) =
        Some(engine).filter(|engine| provides_usable_sound_root_sparse_crown(*engine))
    else {
        eprintln!(
            "[root-wide-demanded-interm-crown] local engine lacks an authorized typed sweep or \
             legacy finite-deadline sound fold; skipping (bounds unchanged)"
        );
        return 0;
    };
    let targets =
        crate::beta_crown::engine::graph::propagation::batched::interm_refine::
            scoped_preparable_wide_demanded_crown_targets(
                graph,
                input,
                &bootstrap.initial_node_bounds,
                policy.min_dim,
                policy.max_dim,
                policy.max_targets,
                policy.max_preflights,
                pass_deadline,
            );
    let target_names: Vec<_> = targets.iter().map(|target| target.name.as_str()).collect();
    eprintln!(
        "[root-wide-demanded-interm-crown] legacy-fallback targets={target_names:?} \
         dim={}..={} max_rows={} max_targets={} max_preflights={} \
         rank=crossing-mass*x-downstream-unstable-relu budget={:.3}s",
        policy.min_dim,
        policy.max_dim,
        policy.max_rows,
        policy.max_targets,
        policy.max_preflights,
        pass_deadline.saturating_duration_since(now).as_secs_f32(),
    );
    crate::beta_crown::engine::graph::propagation::batched::interm_refine::
        root_wide_demanded_tighten_relu_preactivations(
            graph,
            input,
            targets,
            Some(engine),
            Some(pass_deadline),
            policy.max_rows,
            &mut bootstrap.initial_node_bounds,
        )
}

/// Verbatim move of the sparse crossing-row CROWN tightener (PRODUCER).
///
/// Extracted by the technique the previous attempt should have used: the
/// parameters are named EXACTLY like the locals the block referenced, so the
/// body is copied byte-for-byte with no identifier rewriting. The one edit is
/// renaming the bound result. Scripted rewriting of bare `engine`/`graph`
/// corrupted a fully-qualified path here last time
/// (`crate::beta_crown::engine::graph::propagation::…`); nothing is rewritten
/// now, so that class of corruption is impossible.
pub(super) fn sparse_interm_crown(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    root_interm_factory_requested: bool,
    verifier: &crate::beta_crown::BetaCrownVerifier,
    bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
) -> usize {
    let tightened_targets = if let Some(policy) =
        root_sparse_interm_crown_policy_from_env(&verifier.config)
    {
        let now = std::time::Instant::now();
        if let Some(pass_deadline) =
            bounded_root_crown_interm_deadline(now, deadline, policy.max_secs)
        {
            let targets =
                crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                    scoped_sparse_crown_targets(
                        graph,
                        &bootstrap.initial_node_bounds,
                        policy.max_dim,
                        policy.max_rows,
                        policy.max_targets,
                    );
            eprintln!(
                "[root-sparse-interm-crown] targets={} max_dim={} max_rows={} max_targets={} budget={:.3}s",
                targets.len(),
                policy.max_dim,
                policy.max_rows,
                policy.max_targets,
                pass_deadline.saturating_duration_since(now).as_secs_f32(),
            );
            let route = root_interm_engine_route(
                engine.is_some_and(provides_usable_sound_root_sparse_crown),
                root_interm_factory_requested,
            );
            match route {
                RootIntermEngineRoute::Local | RootIntermEngineRoute::Unavailable => {
                    crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                        root_sparse_tighten_relu_preactivations(
                            graph,
                            input,
                            &targets,
                            engine,
                            Some(pass_deadline),
                            policy.max_rows,
                            &mut bootstrap.initial_node_bounds,
                        )
                }
                RootIntermEngineRoute::Factory => {
                    let routed = crate::sound_f64_gemm::with_engine_deadline(
                        pass_deadline,
                        |factory_engine| {
                            provides_usable_sound_root_sparse_crown(factory_engine).then(|| {
                                crate::beta_crown::engine::graph::propagation::batched::
                                    interm_refine::root_sparse_tighten_relu_preactivations(
                                        graph,
                                        input,
                                        &targets,
                                        Some(factory_engine),
                                        Some(pass_deadline),
                                        policy.max_rows,
                                        &mut bootstrap.initial_node_bounds,
                                    )
                            })
                        },
                    );
                    match routed {
                        Ok(Some(Some(n_tightened))) => n_tightened,
                        Ok(Some(None)) => {
                            eprintln!(
                                "[root-sparse-interm-crown] sound CUDA factory engine lacks the \
                                 sparse CROWN capability; skipping (bounds unchanged)"
                            );
                            0
                        }
                        Ok(None) => {
                            eprintln!(
                                "[root-sparse-interm-crown] sound CUDA factory unavailable; \
                                 skipping (bounds unchanged)"
                            );
                            0
                        }
                        Err(error) => {
                            eprintln!(
                                "[root-sparse-interm-crown] sound CUDA factory admission refused: \
                                 {error}; skipping (bounds unchanged)"
                            );
                            0
                        }
                    }
                }
            }
        } else {
            eprintln!(
                "[root-sparse-interm-crown] no safe deadline slice remains; skipping (bounds unchanged)"
            );
            0
        }
    } else {
        0
    };
    tightened_targets
}

/// Verbatim move of the joint per-target interm-α tightener (PRODUCER).
///
/// The §14 worry that this phase's binding site and assignment site were
/// different statements turned out to be wrong in the way that matters: the
/// `let mut … = 0usize` and the gated block that assigns it are CONTIGUOUS
/// (root.rs:2137-2278, nothing between). So the declaration moves inside the
/// function and becomes its return value, and the body still copies verbatim.
///
/// Dark by default (`NY_ROOT_JOINT_INTERM_ALPHA=1`), so this contributes
/// nothing to the shipped configuration — a completeness item, not a behaviour
/// one.
pub(super) fn root_joint_interm_alpha(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    verifier: &crate::beta_crown::BetaCrownVerifier,
    // #joint-interm-grad: needed to seed the objective adjoint that produces the
    // per-target `df/dl` weights.
    objectives: &[Vec<f32>],
    bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
) -> usize {
    let mut root_joint_interm_tightened_targets = 0usize;
    let configured_master = crate::preset_gates::root_joint_interm_alpha_enabled();
    let configured_deadline_ascent =
        crate::preset_gates::root_joint_interm_alpha_deadline_ascent_enabled();
    let configured_secs = crate::preset_gates::root_joint_interm_alpha_secs();
    let configured_iters = crate::preset_gates::root_joint_interm_alpha_iters();
    let configured_lr = crate::preset_gates::root_joint_interm_alpha_lr() as f32;
    let configured_max_sel = crate::preset_gates::root_joint_interm_alpha_max_sel();
    let configured_frozen_stop = crate::preset_gates::root_joint_interm_alpha_frozen_stop_enabled();
    let configured_pure_chain = crate::preset_gates::root_joint_interm_alpha_pure_chain_enabled();
    let configured_bn_gpu_extract = crate::preset_gates::bn_gpu_extract_enabled();
    let configured_min_remaining_secs = crate::preset_gates::root_joint_min_remaining_secs();
    let deadline_lane_requested = crate::sound_gpu_gate::root_joint_deadline_lane_requested();
    debug_assert_eq!(
        deadline_lane_requested,
        configured_master && configured_deadline_ascent,
        "the compound GPU admission must use the same governed answers as the receipt"
    );
    let configured_max_dim_override =
        crate::preset_gates::root_joint_interm_alpha_max_dim_override();
    let configured_max_dim = configured_max_dim_override.unwrap_or(if deadline_lane_requested {
        32_768
    } else {
        2_048
    });
    let invocation_started = std::time::Instant::now();
    let mut target_count = 0usize;
    let mut available_budget = deadline
        .map(|limit| limit.saturating_duration_since(invocation_started))
        .unwrap_or_else(|| std::time::Duration::from_secs(configured_secs));
    let mut effective_slice = std::time::Duration::ZERO;
    let mut attempts = 0usize;
    let mut work_report = RootJointTightenReport::default();
    let mut outcome = if configured_master {
        crate::execution_telemetry::RootJointIntermAlphaOutcome::ExecutionRefused
    } else {
        crate::execution_telemetry::RootJointIntermAlphaOutcome::MasterDisabled
    };

    if configured_master {
        let now = std::time::Instant::now();
        let grace_cap = configured_secs;
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or_else(|| std::time::Duration::from_secs(grace_cap));
        available_budget = global_remaining;
        // #root-joint-admission (promotion prerequisite): the pass's measured
        // cost is ~+18s bab-start — free at the 700s research tier, FATAL on
        // scored 100s rows banked at 90-96s runtime (the tax alone flips them
        // to timeout). Admission floor: run only when the remaining budget can
        // absorb the slice with room for BaB — default 240s, override with
        // NY_ROOT_JOINT_MIN_REMAINING_SECS (0 disables the floor; research
        // probes at 400-900s budgets are unaffected). Below the floor the
        // gate-ON path is byte-identical to gate-OFF (skip, no map change).
        let min_remaining = configured_min_remaining_secs;
        let admitted = !(deadline.is_some()
            && global_remaining < std::time::Duration::from_secs(min_remaining));
        if !admitted {
            eprintln!(
                "[root-joint-interm-alpha] admission floor: remaining {:.1}s < {min_remaining}s — pass skipped (byte-identical)",
                global_remaining.as_secs_f32()
            );
        }
        // Reserve at least half the remaining budget for root-spec + BaB.
        let slice = std::time::Duration::from_secs(grace_cap).min(global_remaining.mul_f32(0.5));
        effective_slice = slice;
        // #root-joint-demand-rank: with the sound-GPU deadline lane armed, the
        // measured 23.6s/2-target slice tightened only the legacy ≤2048-dim
        // scope (FC head + last residual block) while the tree kept running on
        // the DEMANDED wide targets' fallback-grade boxes. The armed selector
        // widens the scope default and ranks demanded-first, so the GPU slice
        // is spent on the decisive nodes and the per-target deadline loop cuts
        // the tail. Unarmed path: the legacy selector, byte-identical.
        let targets = if deadline_lane_requested {
            crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                scoped_joint_alpha_targets_demand_ranked(graph, &bootstrap.initial_node_bounds)
        } else {
            crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                scoped_joint_alpha_targets(graph, &bootstrap.initial_node_bounds)
        };
        target_count = targets.len();
        eprintln!(
            "[root-joint-interm-alpha] gate ON, targets={} slice={:.1}s{}",
            targets.len(),
            slice.as_secs_f32(),
            if deadline_lane_requested {
                " rank=demand-first"
            } else {
                ""
            }
        );
        // #boxlift phase mirror (dark, NY_PHASE_TELEMETRY=1, print-only): the
        // same summary as a `[phase]` line on the shared epoch clock, so the
        // pass shows up in phase logs alongside `[frontier]` frames. The bare
        // eprintln above is unchanged; gate-off skips the format entirely.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            crate::phase_telemetry::phase_marker(&format!(
                "root-joint-interm start targets={} slice={:.1}s",
                targets.len(),
                slice.as_secs_f32()
            ));
        }
        outcome = if !admitted {
            crate::execution_telemetry::RootJointIntermAlphaOutcome::AdmissionFloor
        } else if !deadline_lane_requested {
            crate::execution_telemetry::RootJointIntermAlphaOutcome::DeadlineRouteDisabled
        } else if slice < std::time::Duration::from_secs(2) {
            crate::execution_telemetry::RootJointIntermAlphaOutcome::SliceTooSmall
        } else if targets.is_empty() {
            crate::execution_telemetry::RootJointIntermAlphaOutcome::NoTargets
        } else {
            crate::execution_telemetry::RootJointIntermAlphaOutcome::ExecutionRefused
        };
        if admitted
            && deadline_lane_requested
            && slice >= std::time::Duration::from_secs(2)
            && !targets.is_empty()
        {
            let iters = configured_iters;
            let lr = configured_lr;
            let pass_deadline = now + slice;
            // #joint-interm-grad: harvest `df/dl` per target ONCE, before either
            // routing attempt, so both share it and neither pays twice. `None`
            // restores the historical unit seed, so a failure here costs the
            // weighting and nothing else.
            let sensitivity = joint_interm_objective_sensitivity(
                verifier,
                graph,
                input,
                &bootstrap.initial_node_bounds,
                objectives,
                pass_deadline,
            );
            eprintln!(
                "[joint-interm-grad] weighted-seed targets={} (none => unit diagonal)",
                sensitivity
                    .as_ref()
                    .map_or(0, std::collections::HashMap::len),
            );
            let local = engine.and_then(|local_engine| {
                crate::sound_gpu_gate::with_root_joint_deadline_gpu(local_engine, |gpu| {
                    root_joint_tighten_relu_preactivations_with_deadline_gpu(
                        graph,
                        input,
                        &targets,
                        gpu,
                        pass_deadline,
                        iters,
                        lr,
                        sensitivity.as_ref(),
                        &mut bootstrap.initial_node_bounds,
                    )
                })
            });
            let (report, terminal_outcome) = if let Some(report) = local {
                // A completed local attempt, including a sound zero-result,
                // owns this invocation. Never retry it on another backend.
                attempts = 1;
                (
                    report,
                    crate::execution_telemetry::RootJointIntermAlphaOutcome::Completed,
                )
            } else {
                let routed =
                    crate::sound_f64_gemm::with_engine_deadline(pass_deadline, |factory_engine| {
                        crate::sound_gpu_gate::with_root_joint_deadline_gpu(factory_engine, |gpu| {
                            root_joint_tighten_relu_preactivations_with_deadline_gpu(
                                graph,
                                input,
                                &targets,
                                gpu,
                                pass_deadline,
                                iters,
                                lr,
                                sensitivity.as_ref(),
                                &mut bootstrap.initial_node_bounds,
                            )
                        })
                    });
                match routed {
                    Ok(Some(Some(report))) => {
                        attempts = 1;
                        (
                            report,
                            crate::execution_telemetry::RootJointIntermAlphaOutcome::Completed,
                        )
                    }
                    Ok(Some(None)) => {
                        eprintln!(
                            "[root-joint-interm-alpha] sound CUDA factory engine lacks the exact \
                             bounded root-joint capability; skipping (sound no-op)"
                        );
                        (
                            RootJointTightenReport::default(),
                            crate::execution_telemetry::RootJointIntermAlphaOutcome::CapabilityUnavailable,
                        )
                    }
                    Ok(None) => {
                        eprintln!(
                            "[root-joint-interm-alpha] sound CUDA factory unavailable; skipping \
                             (sound no-op)"
                        );
                        (
                            RootJointTightenReport::default(),
                            crate::execution_telemetry::RootJointIntermAlphaOutcome::CapabilityUnavailable,
                        )
                    }
                    Err(error) => {
                        eprintln!(
                            "[root-joint-interm-alpha] sound CUDA factory admission refused: \
                             {error}; skipping (sound no-op)"
                        );
                        (
                            RootJointTightenReport::default(),
                            crate::execution_telemetry::RootJointIntermAlphaOutcome::ExecutionRefused,
                        )
                    }
                }
            };
            work_report = report;
            root_joint_interm_tightened_targets = report.tightened_count;
            outcome = terminal_outcome;
            eprintln!(
                "[root-joint-interm-alpha] done: {}/{} target(s) tightened",
                root_joint_interm_tightened_targets,
                targets.len()
            );
            // #boxlift phase mirror (dark, print-only): pass outcome + wall on
            // the shared epoch clock; the eprintln above is unchanged.
            if crate::phase_telemetry::phase_telemetry_enabled() {
                crate::phase_telemetry::phase_marker(&format!(
                    "root-joint-interm done tightened={}/{} wall={:.1}s",
                    root_joint_interm_tightened_targets,
                    targets.len(),
                    now.elapsed().as_secs_f32()
                ));
            }
        }
    }
    crate::execution_telemetry::record_root_joint_interm_alpha_invocation(
        crate::execution_telemetry::RootJointIntermAlphaInvocation {
            route: crate::execution_telemetry::RootJointIntermAlphaRoute::MultiObjective,
            configured_master,
            configured_max_dim_override,
            configured_max_dim,
            configured_secs,
            configured_iters: u64::try_from(configured_iters).unwrap_or(u64::MAX),
            configured_lr,
            configured_max_sel: u64::try_from(configured_max_sel).unwrap_or(u64::MAX),
            configured_frozen_stop,
            configured_pure_chain,
            configured_bn_gpu_extract,
            configured_deadline_ascent,
            configured_min_remaining_secs,
            target_count,
            available_budget,
            effective_slice,
            attempts,
            tightened_count: root_joint_interm_tightened_targets,
            outlier_tightened_count: 0,
            target_attempts: work_report.target_attempts,
            base_folds_completed: work_report.base_folds_completed,
            effective_iters: work_report.effective_iters,
            ascent_iterations_attempted: work_report.ascent_iterations_attempted,
            max_ascent_iterations_attempted: work_report.max_ascent_iterations_attempted,
            optimizer_steps: work_report.optimizer_steps,
            ascent_folds_completed: work_report.ascent_folds_completed,
            max_selected_rows: work_report.max_selected_rows,
            selection_cap_bound_targets: work_report.selection_cap_bound_targets,
            frozen_stop_targets: work_report.frozen_stop_targets,
            pure_chain_targets: work_report.pure_chain_targets,
            bn_extract_targets: work_report.bn_extract_targets,
            outcome,
        },
    );
    root_joint_interm_tightened_targets
}

// ===========================================================================
// Scheduler machinery attached to the production phases (I1 + I3)
// ===========================================================================

impl RootTightenPhase {
    /// Predicted cost of this phase on this host, in seconds.
    ///
    /// Each phase already owns a budget it would spend if it runs — its grace
    /// cap or policy `max_secs`. That IS the prediction; today nothing compares
    /// it against what remains before starting. Returns `None` where a phase
    /// has no declared budget, and a phase with no prediction is never declined
    /// (absence of information must not make a gate stricter OR looser).
    pub(crate) fn predicted_cost(self) -> Option<std::time::Duration> {
        let secs = match self {
            Self::FcHeadTighten => std::env::var("NY_FCHEAD_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(12),
            Self::RootIntermAlpha => std::env::var("NY_ROOT_INTERM_ALPHA_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(120),
            Self::RootJointIntermAlpha => crate::preset_gates::root_joint_interm_alpha_secs(),
            Self::ComprehensiveGpuIntermCrown | Self::PhaseResidentCrown => 20,
            // The remaining phases are sub-second or their budget lives inside
            // a policy struct resolved at run time; no prediction is offered
            // rather than a guessed one.
            _ => return None,
        };
        Some(std::time::Duration::from_secs(secs))
    }

    /// Invariant I1 applied to a production phase: would starting this phase
    /// fit inside a fraction of the REAL remaining instance budget?
    ///
    /// This is the scheduler's admission rule
    /// ([`ny_core::phase_window::admit`]) reaching a real cost centre. It is
    /// deliberately NOT a reordering loop: the phases cannot be hoisted to a
    /// single site without changing behaviour (measured — bounds bit-identical,
    /// verdict changed), so admission and retirement are the parts of the
    /// scheduler that apply here, applied where each phase already sits.
    ///
    /// Fail-closed: with no published instance budget or no prediction, the
    /// answer is "run it", i.e. exactly today's behaviour.
    ///
    /// #router-r3 `rule.root_admission`: when the acting optimizer armed the
    /// rule, the same predicate runs without the legacy gate, against the
    /// window the phase would actually run in — the verifier's R2 deadline
    /// cell when the router governs it, else the instance remaining — and
    /// every consult logs its inputs, for every phase the router consults
    /// ([`Self::router_consults`]). Unarmed, this is today's function.
    #[must_use]
    pub(crate) fn admitted(self) -> bool {
        let authority = ny_core::stop_authority::global();
        if authority.root_admission() {
            let cell = ny_core::deadline_authority::global().remaining(
                ny_core::deadline_authority::LaneSlot::Verifier,
                std::time::Instant::now(),
            );
            if let Some(admitted) =
                self.router_admission(authority, ny_core::instance_budget::remaining(), cell)
            {
                return admitted;
            }
        }
        if !root_phase_admission_enabled() {
            return true;
        }
        let (Some(predicted), Some(remaining)) =
            (self.predicted_cost(), ny_core::instance_budget::remaining())
        else {
            return true;
        };
        let admission = ny_core::phase_window::admit(
            predicted,
            remaining,
            ny_core::phase_window::WindowPolicy::default(),
        );
        if !admission.is_admitted() {
            tracing::info!(
                phase = self.name(),
                predicted_s = predicted.as_secs_f64(),
                remaining_s = remaining.as_secs_f64(),
                "#root-phase-admission (I1): declining a phase that does not fit; \
                 the budget goes to what follows instead of to a pass that would be cut off"
            );
        }
        admission.is_admitted()
    }
}

impl RootTightenPhase {
    /// #router-r3 review: whether `rule.root_admission` consults this phase.
    ///
    /// NOT the two phases whose own run gate is a raw-environment read INSIDE
    /// [`Self::dispatch`] (`fchead_tighten`: `NY_FCHEAD_TIGHTEN`,
    /// `root_interm_alpha`: `NY_ROOT_INTERM_ALPHA`). No scored run sets that
    /// environment, so on every scored row the phase is a no-op, and a router
    /// consult there would book and print `admit=0` for a phase that was never
    /// going to run — a phantom decline that makes the rule read as acting
    /// when it changed nothing. Those two keep today's (legacy) admission.
    /// Every other phase's run is decided by typed state at its call site
    /// (`run_phase_resident_in_place` runs only with a resolved policy,
    /// `run_mip_refine_in_place` only with an attached refiner), so a consult
    /// there is a real decision. The extracted bodies `dispatch` no-ops
    /// (`root_joint_interm_alpha`, the comprehensive / wide / sparse CROWN
    /// sweeps) never call `admitted` at all.
    pub(crate) const fn router_consults(self) -> bool {
        !matches!(self, Self::FcHeadTighten | Self::RootIntermAlpha)
    }

    /// #router-r3 `rule.root_admission`: `phase_window::admit` against the
    /// window this phase would run in (`instance` remaining, the verifier's
    /// R2 `cell` when governed), with the predicate's inputs logged and
    /// ledgered on `authority`. `None` — the caller keeps today's admission —
    /// unless `authority` armed the rule and the router consults this phase
    /// ([`Self::router_consults`]). A phase without a prediction, or a row
    /// without any window, runs (today's answer).
    pub(crate) fn router_admission(
        self,
        authority: &ny_core::stop_authority::StopAuthority,
        instance: Option<std::time::Duration>,
        cell: Option<std::time::Duration>,
    ) -> Option<bool> {
        if !authority.root_admission() || !self.router_consults() {
            return None;
        }
        let Some(predicted) = self.predicted_cost() else {
            return Some(true);
        };
        let Some(event) = route_admission(self.name(), predicted, instance, cell) else {
            return Some(true);
        };
        eprintln!(
            "[optimizer] root-admission rule=rule.root_admission phase={} predicted_ms={} \
             padded_ms={} remaining_ms={} source={} affordable_ms={} admit={}",
            event.phase,
            event.predicted_ms,
            event.padded_ms,
            event.remaining_ms,
            event.source,
            event.affordable_ms,
            u8::from(event.admitted)
        );
        let admitted = event.admitted;
        authority.record_root_admission(event);
        Some(admitted)
    }
}

/// #router-r3 `rule.root_admission`, PURE: the admission of a phase that
/// predicts `predicted` against the window it would run in — the verifier's
/// R2 cell when the router governs it (`cell`), never more than the instance
/// remaining (`instance`). `None` when there is no window at all (the phase
/// runs: today's answer). The event carries every input of the predicate.
fn route_admission(
    phase: &str,
    predicted: std::time::Duration,
    instance: Option<std::time::Duration>,
    cell: Option<std::time::Duration>,
) -> Option<ny_core::stop_authority::RootAdmissionEvent> {
    let (remaining, source) = match (cell, instance) {
        (Some(cell), Some(instance)) => (cell.min(instance), "verifier_cell"),
        (Some(cell), None) => (cell, "verifier_cell"),
        (None, Some(instance)) => (instance, "instance"),
        (None, None) => return None,
    };
    let policy = ny_core::phase_window::WindowPolicy::default();
    let admitted = ny_core::phase_window::admit(predicted, remaining, policy).is_admitted();
    let padded = predicted
        .checked_mul(policy.margin_num)
        .map_or(std::time::Duration::MAX, |d| d / policy.margin_den.max(1));
    let affordable = remaining.mul_f64(if policy.max_frac.is_finite() {
        policy.max_frac.clamp(0.0, 1.0)
    } else {
        0.0
    });
    let ms = |d: std::time::Duration| u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
    Some(ny_core::stop_authority::RootAdmissionEvent {
        phase: phase.to_owned(),
        predicted_ms: ms(predicted),
        padded_ms: ms(padded),
        remaining_ms: ms(remaining),
        source: source.to_owned(),
        affordable_ms: ms(affordable),
        admitted,
    })
}

/// `NY_ROOT_PHASE_ADMISSION=1`, exact, default off => byte-identical.
fn root_phase_admission_enabled() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| std::env::var("NY_ROOT_PHASE_ADMISSION").is_ok_and(|v| v == "1"))
}

/// `NY_ROOT_PHASE_VALUE=1`, exact, default off. Telemetry only.
fn root_phase_value_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NY_ROOT_PHASE_VALUE").is_ok_and(|v| v == "1"))
}

impl RootTightenPhase {
    /// Report what a phase actually cost and actually produced.
    ///
    /// This is §2.2 of the scheduler design — the VALUE MODEL — reaching the
    /// production phases. The counts already existed inside each block and were
    /// consumed by its own log; `PhaseOutput` now carries them to a common
    /// caller, so cost and yield can be read side by side for the first time.
    ///
    /// Reporting rather than acting: a root phase runs ONCE per instance, so
    /// I3's "retire after k zero-yield blocks" has no k to count. Zero yield
    /// here is a fact about this instance, not a trend — acting on it would be
    /// inventing a policy the measurement does not support. What it does give
    /// is the per-phase cost/yield record the design asks for, which is what a
    /// scheduler would need and what nothing in this tree had.
    fn report_value(self, out: PhaseOutput, elapsed: std::time::Duration) {
        if !root_phase_value_enabled() {
            return;
        }
        let produced = out.tightened_elements + out.tightened_targets;
        tracing::info!(
            phase = self.name(),
            elapsed_s = elapsed.as_secs_f64(),
            tightened_elements = out.tightened_elements,
            tightened_targets = out.tightened_targets,
            selected = out.dense_head_stage_selected,
            yielded = produced > 0,
            "#root-phase-value: cost and realised yield"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// #router-r3 `rule.root_admission`: the predicate skips a phase whose
    /// PADDED prediction does not fit the affordable half of its window, runs
    /// one that does, reads the verifier's cell when it is tighter than the
    /// instance remaining, logs every input, and runs the phase when there is
    /// no window at all (today's answer).
    #[test]
    fn router_r3_root_admission_skips_a_phase_that_cannot_fit_and_logs_its_inputs() {
        use std::time::Duration;
        let secs = Duration::from_secs;
        // 12 s padded to 15 s against half of 20 s: declined.
        let declined =
            route_admission("fchead_tighten", secs(12), Some(secs(20)), None).expect("window");
        assert!(!declined.admitted);
        assert_eq!(declined.predicted_ms, 12_000);
        assert_eq!(declined.padded_ms, 15_000);
        assert_eq!(declined.remaining_ms, 20_000);
        assert_eq!(declined.affordable_ms, 10_000);
        assert_eq!(declined.source, "instance");
        // ... against half of 40 s: admitted.
        let admitted =
            route_admission("fchead_tighten", secs(12), Some(secs(40)), None).expect("window");
        assert!(admitted.admitted);
        // The verifier's cell, when tighter, is the window that counts.
        let cell = route_admission(
            "root_interm_alpha",
            secs(12),
            Some(secs(40)),
            Some(secs(20)),
        )
        .expect("window");
        assert!(!cell.admitted);
        assert_eq!(cell.source, "verifier_cell");
        assert_eq!(cell.remaining_ms, 20_000);
        // A cell never widens the instance remaining.
        let wide = route_admission(
            "root_interm_alpha",
            secs(12),
            Some(secs(20)),
            Some(secs(90)),
        )
        .expect("window");
        assert_eq!(wide.remaining_ms, 20_000);
        assert!(!wide.admitted);
        // No window at all: the phase runs.
        assert!(route_admission("x", secs(12), None, None).is_none());
        // Extremes never panic.
        let _ = route_admission(
            "x",
            Duration::MAX,
            Some(Duration::MAX),
            Some(Duration::ZERO),
        );
        let _ = route_admission("x", Duration::ZERO, Some(Duration::ZERO), None);
        // Unarmed, the production entry is today's: the phase-level answer of
        // every phase is unchanged (the legacy gate is off in tests).
        assert!(!ny_core::stop_authority::global().root_admission());
        for p in ORDER {
            assert!(p.admitted(), "{} declined on an unarmed row", p.name());
        }
    }

    /// #router-r3 review: `rule.root_admission` consults only phases whose run
    /// is decided by typed state at their call site. The two env-gated phases
    /// (a no-op on every scored row) are never consulted, so the rule cannot
    /// book a phantom decline for them; a typed-reachable phase that cannot
    /// fit is declined and ledgered; a phase without a prediction runs and
    /// books nothing; an unarmed authority consults nothing (today's gate).
    #[test]
    fn router_r3_root_admission_consults_only_phases_that_would_run() {
        use ny_core::stop_authority::{StopAuthority, StopDirectives};
        use std::time::Duration;
        let armed = StopAuthority::new();
        armed
            .arm(StopDirectives {
                root_admission: true,
                ..StopDirectives::NONE
            })
            .expect("arm");
        let tight = Some(Duration::from_secs(10));
        for phase in [
            RootTightenPhase::FcHeadTighten,
            RootTightenPhase::RootIntermAlpha,
        ] {
            assert!(phase.predicted_cost().is_some(), "{}", phase.name());
            assert!(!phase.router_consults(), "{}", phase.name());
            assert_eq!(
                phase.router_admission(&armed, tight, None),
                None,
                "{}: env-gated inside dispatch, never consulted",
                phase.name()
            );
        }
        assert!(
            armed.snapshot().root_admissions.is_empty(),
            "no phantom decline is booked for a phase that would not run"
        );
        // Positive control: the phase-resident transaction (20 s, padded 25 s)
        // cannot fit half of a 10 s window.
        assert_eq!(
            RootTightenPhase::PhaseResidentCrown.router_admission(&armed, tight, None),
            Some(false)
        );
        let booked = armed.snapshot().root_admissions;
        assert_eq!(booked.len(), 1);
        assert_eq!(booked[0].phase, RootTightenPhase::PhaseResidentCrown.name());
        assert!(!booked[0].admitted);
        // No prediction: runs, nothing booked.
        assert_eq!(
            RootTightenPhase::StabilizeAndFix.router_admission(&armed, tight, None),
            Some(true)
        );
        assert_eq!(armed.snapshot().root_admissions.len(), 1);
        // Unarmed: the router consults nothing.
        let unarmed = StopAuthority::new();
        for phase in ORDER {
            assert_eq!(
                phase.router_admission(&unarmed, tight, None),
                None,
                "{}",
                phase.name()
            );
        }
    }

    #[test]
    fn order_contains_every_variant_exactly_once() {
        // If a variant is added without placing it in ORDER, the sequence data
        // silently stops describing the function it is meant to describe.
        let seen: HashSet<_> = ORDER.iter().copied().collect();
        assert_eq!(seen.len(), ORDER.len(), "ORDER must not repeat a phase");
        let all = [
            RootTightenPhase::StabilizeAndFix,
            RootTightenPhase::DdZonoIntermIntersect,
            RootTightenPhase::MipRefineInterm,
            RootTightenPhase::FcHeadTighten,
            RootTightenPhase::RootIntermAlpha,
            RootTightenPhase::RootJointIntermAlpha,
            RootTightenPhase::ComprehensiveGpuIntermCrown,
            RootTightenPhase::WideDemandedIntermCrown,
            RootTightenPhase::SparseIntermCrown,
            RootTightenPhase::PhaseResidentCrown,
            RootTightenPhase::DenseHeadTighten,
        ];
        for p in all {
            assert!(seen.contains(&p), "{} missing from ORDER", p.name());
        }
        assert_eq!(ORDER.len(), all.len());
    }

    #[test]
    fn the_order_is_the_functions_order() {
        // Pinned deliberately. Step 1 is behaviour-preserving ONLY if walking
        // ORDER reproduces evaluate_root's current sequence; if someone
        // reorders this list they are changing behaviour, and this test is
        // where they should find that out.
        let names: Vec<_> = ORDER.iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            vec![
                "stabilize_and_fix",
                "ddzono_interm_intersect",
                "mip_refine_interm",
                "fchead_tighten",
                "root_interm_alpha",
                "root_joint_interm_alpha",
                "comprehensive_gpu_interm_crown",
                "wide_demanded_interm_crown",
                "sparse_interm_crown",
                "phase_resident_crown",
                "dense_head_tighten",
            ]
        );
    }

    #[test]
    fn names_are_unique() {
        let names: HashSet<_> = ORDER.iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), ORDER.len(), "phase names must be distinct");
    }

    #[test]
    fn every_schedulable_phase_is_shrink_only() {
        // The theorem's precondition, as a test. A variant that is not
        // shrink-only cannot be reordered soundly and does not belong here.
        for p in ORDER {
            assert!(p.is_shrink_only(), "{} is not shrink-only", p.name());
        }
    }

    #[test]
    fn extraction_progress_is_explicit() {
        // Step 2 proceeds one phase at a time, each verified byte-identical.
        // This pins which bodies have actually moved, so "the table exists" is
        // never mistaken for "the function is decomposed".
        let done: Vec<_> = ORDER
            .iter()
            .filter(|p| p.is_extracted())
            .map(|p| p.name())
            .collect();
        assert_eq!(
            done,
            vec![
                "stabilize_and_fix",
                "ddzono_interm_intersect",
                "mip_refine_interm",
                "fchead_tighten",
                "root_interm_alpha",
                "root_joint_interm_alpha",
                "comprehensive_gpu_interm_crown",
                "wide_demanded_interm_crown",
                "sparse_interm_crown",
                "phase_resident_crown",
                "dense_head_tighten"
            ]
        );
    }

    #[test]
    fn producers_are_recorded_so_extraction_cannot_drop_an_output() {
        // A phase that is shrink-only in its bound mutation can still be a
        // PRODUCER in its control flow. Extracting one without carrying its
        // output compiles and silently changes which downstream passes run --
        // measured on dense_head_tighten, whose two outer locals are read at
        // five later sites in evaluate_root. Recorded per phase so it is never
        // rediscovered the hard way.
        let producers: Vec<_> = ORDER
            .iter()
            .filter(|p| p.is_producer())
            .map(|p| p.name())
            .collect();
        assert_eq!(
            producers,
            vec![
                "mip_refine_interm",
                "root_joint_interm_alpha",
                "comprehensive_gpu_interm_crown",
                "wide_demanded_interm_crown",
                "sparse_interm_crown",
                "phase_resident_crown",
                "dense_head_tighten"
            ]
        );
        // NOTE: this assertion used to be "every extracted phase is a
        // non-producer", which held only while there was NO WAY to carry
        // outputs. `PhaseOutput` changed that, and the test caught the change
        // the moment the first producer was extracted -- which is the behaviour
        // wanted from it.
        //
        // The invariant now is the one that still matters: a producer may be
        // extracted only once the channel exists. Extracted producers must
        // therefore be a subset of the audited producer list, so a phase can
        // never be moved as a producer without having been classified as one.
        for p in ORDER.iter().filter(|p| p.is_extracted() && p.is_producer()) {
            assert!(
                producers.contains(&p.name()),
                "{} was extracted as a producer but is not in the audited list",
                p.name()
            );
        }
    }

    #[test]
    fn phase_output_merges_monotonically() {
        let mut acc = PhaseOutput::default();
        acc.merge(PhaseOutput {
            dense_head_stage_selected: false,
            tightened_elements: 3,
            tightened_targets: 1,
        });
        acc.merge(PhaseOutput {
            dense_head_stage_selected: true,
            tightened_elements: 4,
            tightened_targets: 2,
        });
        assert!(
            acc.dense_head_stage_selected,
            "the flag must latch, not toggle"
        );
        assert_eq!(acc.tightened_elements, 7);
        assert_eq!(acc.tightened_targets, 3);
        assert!(acc.bounds_changed());
        // A later non-selecting phase must not clear an earlier selection.
        acc.merge(PhaseOutput::default());
        assert!(acc.dense_head_stage_selected);
    }

    #[test]
    fn phases_with_a_declared_budget_predict_it_and_others_abstain() {
        // A prediction is only offered where the phase actually declares a
        // budget. Guessing one for the rest would be the fixed-constant defect
        // this whole system exists to remove, wearing a new hat.
        assert_eq!(
            RootTightenPhase::FcHeadTighten.predicted_cost(),
            Some(std::time::Duration::from_secs(12))
        );
        assert_eq!(
            RootTightenPhase::RootIntermAlpha.predicted_cost(),
            Some(std::time::Duration::from_mins(2))
        );
        assert_eq!(RootTightenPhase::StabilizeAndFix.predicted_cost(), None);
        assert_eq!(
            RootTightenPhase::PhaseResidentCrown.predicted_cost(),
            Some(std::time::Duration::from_secs(20))
        );
        assert_eq!(RootTightenPhase::DenseHeadTighten.predicted_cost(), None);
    }

    #[test]
    fn resident_phase_fallback_is_preaccept_only() {
        assert!(PhaseResidentRun::AdmissionDeclined.permits_legacy_dense_fallback());
        assert!(PhaseResidentRun::CleanDecline.permits_legacy_dense_fallback());
        assert!(!PhaseResidentRun::Failed.permits_legacy_dense_fallback());
        assert!(
            !PhaseResidentRun::Completed(PhaseOutput::default()).permits_legacy_dense_fallback()
        );

        let target_only = PhaseOutput {
            dense_head_stage_selected: true,
            tightened_elements: 0,
            tightened_targets: 1,
        };
        assert!(target_only.bounds_changed());
        assert_eq!(
            PhaseResidentRun::Completed(target_only).output(),
            target_only
        );
    }

    #[test]
    fn admission_is_permissive_when_it_has_no_information() {
        // Gate off, or no published instance budget, or no prediction => run,
        // i.e. exactly today's behaviour. Absence of information must never
        // make a gate stricter; that is how a working phase gets silently
        // dropped.
        for p in ORDER {
            assert!(p.admitted(), "{} declined with the gate off", p.name());
        }
    }

    // -----------------------------------------------------------------------
    // #relusplitter-lp-refine (wave-6 C7): the refiner phase publishes only
    // through the shrink-only primitive, and is reached by the real root.
    // -----------------------------------------------------------------------

    use crate::beta_crown::root_interm_refine::{
        RootIntermRefineOutcome, RootIntermRefineRequest, RootIntermRefineTelemetry,
        RootIntermRefiner,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Returns fixed candidates and counts its calls.
    struct MockRefiner {
        candidates: HashMap<String, (Vec<f64>, Vec<f64>)>,
        calls: AtomicUsize,
    }

    impl MockRefiner {
        fn new(candidates: HashMap<String, (Vec<f64>, Vec<f64>)>) -> Self {
            Self {
                candidates,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl RootIntermRefiner for MockRefiner {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn refine(&self, req: &RootIntermRefineRequest<'_>) -> RootIntermRefineOutcome {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(
                req.deadline.is_some(),
                "the root must hand the refiner a bounded window"
            );
            RootIntermRefineOutcome {
                candidates: self.candidates.clone(),
                telemetry: RootIntermRefineTelemetry {
                    engaged: true,
                    reason: "ok",
                    ..RootIntermRefineTelemetry::default()
                },
            }
        }
    }

    /// `x in [-1,1]^2 -> Linear(2->2) -> ReLU -> Linear(2->1)`.
    fn tiny_relu_graph() -> (GraphNetwork, BoundedTensor) {
        use crate::layers::{Layer, LinearLayer, ReLULayer};
        use crate::network::Network;
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                ndarray::arr2(&[[1.0_f32, 1.0], [1.0, -1.0]]),
                Some(ndarray::arr1(&[0.25_f32, -0.25])),
            )
            .expect("first affine"),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                ndarray::arr2(&[[1.0_f32, -1.0]]),
                Some(ndarray::arr1(&[0.0_f32])),
            )
            .expect("head affine"),
        ));
        let graph = GraphNetwork::from_sequential(&network).expect("tiny graph");
        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32, -1.0]).into_dyn(),
            ndarray::arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("input box");
        (graph, input)
    }

    fn tiny_bootstrap(
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap {
        let config = crate::beta_crown::BetaCrownConfig {
            use_alpha_crown: false,
            ..crate::beta_crown::BetaCrownConfig::default()
        };
        crate::beta_crown::engine::graph::shared::init::compute_graph_bab_bootstrap(
            graph, input, &config, None, None,
        )
        .expect("tiny bootstrap")
    }

    fn box_of(
        bootstrap: &crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
        name: &str,
    ) -> (Vec<f32>, Vec<f32>) {
        let stored = bootstrap
            .initial_node_bounds
            .get(name)
            .unwrap_or_else(|| panic!("the root stores `{name}` (pre-activation key assumption)"));
        (
            stored.lower().iter().copied().collect(),
            stored.upper().iter().copied().collect(),
        )
    }

    fn run_mock(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        bootstrap: &mut crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap,
        refiner: &MockRefiner,
    ) -> PhaseOutput {
        RootTightenPhase::MipRefineInterm.run_mip_refine_in_place(
            graph,
            input,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
            bootstrap,
            refiner,
        )
    }

    #[test]
    fn mip_refine_phase_intersects_a_tighter_candidate_shrink_only() {
        let (graph, input) = tiny_relu_graph();
        let mut bootstrap = tiny_bootstrap(&graph, &input);
        let (lo, hi) = box_of(&bootstrap, "layer_0");
        // Mechanics only (the refiner owns soundness by contract): neuron 0
        // gets a strictly tighter box, neuron 1 a strictly WIDER one.
        let candidate_lo = vec![f64::from(lo[0]) + 0.5, f64::from(lo[1]) - 1.0];
        let candidate_hi = vec![f64::from(hi[0]) - 0.5, f64::from(hi[1]) + 1.0];
        let refiner = MockRefiner::new(HashMap::from([(
            "layer_0".to_string(),
            (candidate_lo.clone(), candidate_hi.clone()),
        )]));
        let out = run_mock(&graph, &input, &mut bootstrap, &refiner);
        assert_eq!(refiner.calls.load(Ordering::SeqCst), 1);
        let (new_lo, new_hi) = box_of(&bootstrap, "layer_0");
        for i in 0..2 {
            assert!(
                new_lo[i] >= lo[i] && new_hi[i] <= hi[i],
                "widened neuron {i}"
            );
        }
        // Neuron 0 shrank to the candidate, rounded OUTWARD (never tighter).
        assert!(f64::from(new_lo[0]) <= candidate_lo[0] && new_lo[0] > lo[0]);
        assert!(f64::from(new_hi[0]) >= candidate_hi[0] && new_hi[0] < hi[0]);
        // Neuron 1's candidate was wider: its stored box is untouched.
        assert_eq!((new_lo[1], new_hi[1]), (lo[1], hi[1]));
        assert_eq!(out.tightened_targets, 1);
        assert!(out.bounds_changed());
    }

    #[test]
    fn mip_refine_phase_refuses_crossed_and_malformed_candidates() {
        let (graph, input) = tiny_relu_graph();
        for candidate in [
            // Disjoint from the stored box: the intersection would be empty.
            (vec![1.0e6, 1.0e6], vec![2.0e6, 2.0e6]),
            // Non-finite.
            (vec![f64::NAN, 0.0], vec![1.0, 1.0]),
            // Arity mismatch.
            (vec![0.0], vec![1.0]),
        ] {
            let mut bootstrap = tiny_bootstrap(&graph, &input);
            let before = box_of(&bootstrap, "layer_0");
            let refiner = MockRefiner::new(HashMap::from([("layer_0".to_string(), candidate)]));
            let out = run_mock(&graph, &input, &mut bootstrap, &refiner);
            assert_eq!(box_of(&bootstrap, "layer_0"), before);
            assert!(!out.bounds_changed());
        }
    }

    #[test]
    fn mip_refine_phase_with_no_candidates_changes_nothing() {
        let (graph, input) = tiny_relu_graph();
        let mut bootstrap = tiny_bootstrap(&graph, &input);
        let before: Vec<(String, (Vec<f32>, Vec<f32>))> = {
            let mut names: Vec<String> = bootstrap.initial_node_bounds.keys().cloned().collect();
            names.sort();
            names
                .into_iter()
                .map(|name| {
                    let b = box_of(&bootstrap, &name);
                    (name, b)
                })
                .collect()
        };
        let refiner = MockRefiner::new(HashMap::new());
        let out = run_mock(&graph, &input, &mut bootstrap, &refiner);
        assert_eq!(out, PhaseOutput::default());
        for (name, b) in before {
            assert_eq!(box_of(&bootstrap, &name), b, "{name} changed");
        }
    }

    /// Arrival: the REAL multi-objective root consults an attached refiner
    /// exactly once, and a verifier derived through `with_config_from` keeps
    /// it; with nothing attached the getter is `None` (dark by default).
    #[test]
    #[ntest::timeout(60000)]
    fn attached_refiner_reaches_the_multi_objective_root_once() {
        let (graph, input) = tiny_relu_graph();
        let dark = crate::beta_crown::BetaCrownVerifier::new(
            crate::beta_crown::BetaCrownConfig::default(),
        );
        assert!(dark.root_interm_refiner().is_none());

        let refiner = Arc::new(MockRefiner::new(HashMap::new()));
        let config = crate::beta_crown::BetaCrownConfig {
            timeout: std::time::Duration::from_secs(10),
            ..crate::beta_crown::BetaCrownConfig::default()
        };
        let verifier = crate::beta_crown::BetaCrownVerifier::new(config.clone())
            .with_root_interm_refiner(refiner.clone());
        let derived = verifier.with_config_from(config);
        assert!(derived.root_interm_refiner().is_some());

        // y = relu(x0+x1+0.25) - relu(x0-x1-0.25) ranges over roughly
        // [-1.5, 2.25], so `y > 0` is neither verified nor refuted by any box
        // before the tightening window.
        let _ = derived
            .verify_graph_relu_split_multi_objective(&graph, &input, &[vec![1.0_f32]], &[0.0])
            .expect("tiny multi-objective run");
        assert_eq!(
            refiner.calls.load(Ordering::SeqCst),
            1,
            "the root must consult the attached refiner exactly once"
        );
    }

    #[test]
    fn exactly_two_phases_are_armed_in_the_shipped_preset() {
        // Bounds the payoff honestly: ten schedulable phases, two live.
        // Scheduling eight phases that never run is worth nothing, and the plan
        // doc says so -- this keeps the code agreeing with it.
        let armed: Vec<_> = ORDER
            .iter()
            .filter(|p| p.armed_in_cifar100_preset())
            .map(|p| p.name())
            .collect();
        assert_eq!(armed, vec!["sparse_interm_crown", "dense_head_tighten"]);
    }
}
