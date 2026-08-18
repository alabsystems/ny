// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Preset/engine CONTRACT validation (`docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md` §S3).
//!
//! `load_preset` already rejects a key the schema does not know. This module
//! closes the *other* half of the same bug class: a key the schema knows, that
//! parses, that lands in `BetaCrownConfig` — and that no engine code acts on.
//! Such a field is indistinguishable from a working one at every point except
//! the scoreboard, which is why the design document found four of them in a
//! single session.
//!
//! # What this is and is not
//!
//! It is a REPORT, not a repair. Nothing here changes a bound, a certificate or
//! a verdict; it names the fields the engine will drop and, under
//! [`StrictMode`], refuses to start rather than run a configuration that is
//! quietly not the configuration that was asked for.
//!
//! # Direction of failure
//!
//! Every entry currently in the registry fails in the SAFE direction: the
//! engine does less work (skips an optional tightening, runs a slower backend,
//! runs an unscheduled attack), so a bound can only come out LOOSER. A loose
//! bound costs points; it cannot manufacture a proof. That fact is what makes
//! "warn, keep running" the right default here — aborting would convert weak
//! verdicts into no verdicts and buy no soundness. It is recorded per entry as
//! [`Unhonoured::accepted`], and it is a claim that has to be re-argued for any
//! new entry, because a field that failed in the NARROWING direction would be a
//! false-proof generator and must abort (see [`ContractSeverity`]).
//!
//! # Scope limit — read before trusting this as coverage
//!
//! The registry is a hand-maintained list. A field that is silently ignored and
//! that nobody has noticed yet is, by construction, absent from it. This module
//! therefore proves "these known mismatches are reported", never "there are no
//! others". The standing defences against the unknown remainder are
//! `load_preset`'s unknown-key rejection and the per-field propagation tests in
//! `preset::tests`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use tracing::{debug, warn};

use super::PresetConfig;

/// Environment override for how loudly an unhonoured field is reported.
pub(crate) const STRICT_ENV: &str = "NY_PRESET_STRICT";

/// How badly it matters that the engine drops a requested field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContractSeverity {
    /// Dropping the field changes WHICH VERDICTS ARE REACHABLE.
    ///
    /// Every such entry must state, in [`Unhonoured::engine_behaviour`], which
    /// direction it fails in. A field whose omission leaves bounds LOOSER only
    /// loses proofs; a field whose omission could leave a bound NARROWER than
    /// the truth is a false-`unsat` generator and must never be accepted debt —
    /// it is fatal under the default [`StrictMode::Default`].
    VerdictAffecting,
    /// Dropping the field changes throughput, scheduling, or which
    /// counterexample is found first — never which verdicts are reachable given
    /// unbounded time. Acceptance gates are untouched.
    PerformanceOnly,
}

/// Where in a run the request gets dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnhonouredScope {
    /// Dropped on every instance this preset can run.
    Always,
    /// Dropped only on instances that take a particular route; the string names
    /// the condition so the operator can tell whether their rows are affected.
    Conditional(&'static str),
}

/// One requested-but-unhonoured preset field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unhonoured {
    /// Dotted preset path exactly as it is written in the YAML.
    pub(crate) field: &'static str,
    /// The value the preset asked for, rendered for the operator.
    pub(crate) requested: String,
    pub(crate) severity: ContractSeverity,
    pub(crate) scope: UnhonouredScope,
    /// What the engine does instead, and in which direction that fails.
    pub(crate) engine_behaviour: &'static str,
    /// Where in the source the request is dropped. Symbol names, not line
    /// numbers, so the citation survives edits above it.
    pub(crate) citation: &'static str,
    /// `Some(reason)` records this as ACCEPTED DEBT: a mismatch that is known,
    /// argued safe, and tracked. Accepted entries warn. `None` means nobody has
    /// made that argument, so the default strict mode refuses to start.
    pub(crate) accepted: Option<&'static str>,
}

impl Unhonoured {
    /// One operator-readable line.
    fn render(&self) -> String {
        let scope = match self.scope {
            UnhonouredScope::Always => "always".to_string(),
            UnhonouredScope::Conditional(when) => format!("when {when}"),
        };
        let severity = match self.severity {
            ContractSeverity::VerdictAffecting => "verdict-affecting",
            ContractSeverity::PerformanceOnly => "performance-only",
        };
        let tracking = self
            .accepted
            .map_or_else(String::new, |why| format!(" [accepted debt: {why}]"));
        format!(
            "preset field {} requested '{}' is NOT honoured ({scope}, {severity}): {} \
             [dropped at: {}]{tracking}",
            self.field, self.requested, self.engine_behaviour, self.citation
        )
    }

    /// Does this entry stop the run under `mode`?
    fn is_fatal(&self, mode: StrictMode) -> bool {
        match mode {
            StrictMode::Off => false,
            // Verdict-affecting AND unargued. An accepted entry carries a
            // written safe-direction argument, so it warns instead.
            StrictMode::Default => {
                self.severity == ContractSeverity::VerdictAffecting && self.accepted.is_none()
            }
            StrictMode::All => true,
        }
    }
}

/// How the contract check reacts to an unhonoured field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StrictMode {
    /// `NY_PRESET_STRICT=0` — the escape hatch. Report at `debug` level, never
    /// fail. For experiments that deliberately run a preset the engine cannot
    /// fully honour.
    Off,
    /// Default. Loud warning for every unhonoured field; hard error only for a
    /// VERDICT-AFFECTING field with no accepted-debt record, because that is the
    /// class whose safe direction nobody has argued.
    Default,
    /// `NY_PRESET_STRICT=all` — every unhonoured field is a hard error,
    /// accepted or not. The ratchet position: run it in CI over a preset set to
    /// prove the accepted list is empty.
    All,
}

/// Parse the strict-mode value. Pure so tests never touch process env.
///
/// Anything unrecognised (including unset and `1`) is [`StrictMode::Default`]:
/// a typo in the escape hatch must not silently disable the check.
fn strict_mode_from_value(value: Option<&str>) -> StrictMode {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("0" | "off" | "false" | "no") => StrictMode::Off,
        Some("all" | "2" | "everything") => StrictMode::All,
        _ => StrictMode::Default,
    }
}

/// Process-wide strict mode, read once.
fn strict_mode() -> StrictMode {
    static MODE: OnceLock<StrictMode> = OnceLock::new();
    *MODE.get_or_init(|| strict_mode_from_value(std::env::var(STRICT_ENV).ok().as_deref()))
}

/// The engine capabilities the registry's predicates consult.
///
/// Every field is read from the owning crate's own seam rather than restated
/// here, so this struct cannot claim a capability the engine does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineContract {
    /// May WGPU carry verdict-bearing execution?
    /// (`ny_gpu::wgpu_proof_authority`.)
    pub(crate) wgpu_proof_authority: bool,
    /// May the SEQUENTIAL engine honour `bab.clip.interm_domain`?
    /// (`ny_propagate::sequential_clip_interm_domain_supported`.)
    pub(crate) sequential_clip_interm_domain: bool,
    /// Is the reference `middle` (attack INTERLEAVED with BaB) PGD placement
    /// implemented? `before` (upfront) and `after` (deferred post-BaB) are
    /// honoured end-to-end and are no longer part of this capability.
    pub(crate) pgd_order_middle_scheduling: bool,
    /// Does any engine code read `bab.pruning_in_iteration`?
    pub(crate) pruning_in_iteration: bool,
    /// Does any engine code read the `nonlinear_split` filter knobs?
    pub(crate) nonlinear_split_filters: bool,
    /// Does any engine code read the INVPROP preset knobs?
    pub(crate) invprop_preset_knobs: bool,
    /// Does any engine code read `attack.attack_tolerance`?
    pub(crate) attack_tolerance: bool,
    /// Does any engine code read `general.loss_reduction_func`?
    pub(crate) loss_reduction_func: bool,
    /// Does any solver code read `solver.mip.parallel_solvers`?
    pub(crate) mip_parallel_solvers: bool,
    /// Does the UPFRONT attack lane read the graph-PGD-only attack knobs
    /// (`attack.surrogate_sign_gradient`, GAMA `attack.attack_mode`)?
    pub(crate) upfront_lane_reads_graph_pgd_knobs: bool,
}

/// ny decodes `attack.pgd_order` to PGD ENABLEMENT plus PLACEMENT: `before`
/// runs the upfront schedule, `input_bab` suppresses the upfront stage, and
/// `after` is implemented end-to-end as DEFERRED placement
/// (`preset::apply::apply_attack_preset` maps compat-free `after` to
/// `ResolvedInitialPgdSchedule::Deferred`; `commands::vnncomp` honours it by
/// suppressing the upfront wrapper so the post-BaB stage spends the slice).
/// Only alpha-beta-CROWN's `middle` — the attack interleaved WITH BaB — has no
/// scheduler; `middle` loads solely through the explicit
/// `attack.ny_pgd_order_compat: upfront` contract, which maps it back onto the
/// upfront schedule.
const PGD_ORDER_MIDDLE_SCHEDULING_IMPLEMENTED: bool = false;

/// `AlphaCrownConfig` declares the field; no engine code reads it. Running
/// without in-iteration pruning is conservative: more work, never a looser
/// bound (`preset::apply`, historical `warn_unimplemented_fields`).
const PRUNING_IN_ITERATION_IMPLEMENTED: bool = false;

/// GenBaB candidate filtering (`bab.branching.nonlinear_split.filter`,
/// `filter_beta`) has no engine counterpart; the section still selects the
/// GenBaB path via `NonlinearSplitPreset::requests_genbab`.
const NONLINEAR_SPLIT_FILTERS_IMPLEMENTED: bool = false;

/// `bab.invprop.*` is parsed for alpha-beta-CROWN key compatibility; ny's
/// INVPROP configuration is driven by CLI flags (`--invprop-*`), not by these
/// preset keys.
const INVPROP_PRESET_KNOBS_IMPLEMENTED: bool = false;

/// A counterexample is admitted only by the zero-tolerance trusted-oracle gate,
/// so a requested attack slack has nowhere to land.
const ATTACK_TOLERANCE_IMPLEMENTED: bool = false;

/// alpha-beta-CROWN's attack loss reduction (`sum`/`max`/`min`) has no ny
/// counterpart; the attack uses its own fixed objective.
const LOSS_REDUCTION_FUNC_IMPLEMENTED: bool = false;

/// `solver.mip.parallel_solvers` is reserved for the phase-split racing mode
/// (`designs/scip.md` Phase C) and parsed for key compatibility only.
const MIP_PARALLEL_SOLVERS_IMPLEMENTED: bool = false;

/// `attack.surrogate_sign_gradient` and GAMA `attack.attack_mode` are read ONLY
/// inside the graph disjunctive PGD loop (`beta_crown::verify::graph_pgd`). A
/// category whose budget goes to the wrapper's own upfront DLR-APGD lane
/// (`commands::vnncomp`, which builds no `PgdConfig`) never reaches that loop,
/// so both keys are inert there.
///
/// MEASURED, not inferred: on a shipped `traffic_signs_recognition_2023` row at
/// the full 480s budget, "Running graph disjunctive PGD" occurred 0 times and
/// case-insensitive "gama" 0 times. That preset arms both keys — the right
/// techniques for a binarized `Sign` network — and buys nothing.
///
/// Routing them into the upfront lane was attempted and MEASURED to convert
/// nothing (8 rows at 480s: 6 sat + 2 timeout in both arms, byte-identical work
/// on the rows that converge), so the honest state is to REPORT the gap rather
/// than to claim it is closed. See
/// `docs/CANDIDATE_BRANCH_FINDINGS_2026-08-13.md` §3.
const UPFRONT_LANE_READS_GRAPH_PGD_KNOBS_IMPLEMENTED: bool = false;

impl EngineContract {
    /// The capabilities of THIS build.
    pub(crate) const fn current() -> Self {
        Self {
            wgpu_proof_authority: ny_gpu::wgpu_proof_authority(),
            sequential_clip_interm_domain: ny_propagate::sequential_clip_interm_domain_supported(),
            pgd_order_middle_scheduling: PGD_ORDER_MIDDLE_SCHEDULING_IMPLEMENTED,
            pruning_in_iteration: PRUNING_IN_ITERATION_IMPLEMENTED,
            nonlinear_split_filters: NONLINEAR_SPLIT_FILTERS_IMPLEMENTED,
            invprop_preset_knobs: INVPROP_PRESET_KNOBS_IMPLEMENTED,
            attack_tolerance: ATTACK_TOLERANCE_IMPLEMENTED,
            loss_reduction_func: LOSS_REDUCTION_FUNC_IMPLEMENTED,
            mip_parallel_solvers: MIP_PARALLEL_SOLVERS_IMPLEMENTED,
            upfront_lane_reads_graph_pgd_knobs: UPFRONT_LANE_READS_GRAPH_PGD_KNOBS_IMPLEMENTED,
        }
    }

    /// A build that honours everything — the state the registry is trying to
    /// reach. Used by tests to prove each predicate is capability-gated rather
    /// than unconditional, so an entry disappears the moment it is implemented.
    #[cfg(test)]
    pub(crate) const fn all_honoured() -> Self {
        Self {
            wgpu_proof_authority: true,
            sequential_clip_interm_domain: true,
            pgd_order_middle_scheduling: true,
            pruning_in_iteration: true,
            nonlinear_split_filters: true,
            invprop_preset_knobs: true,
            attack_tolerance: true,
            loss_reduction_func: true,
            mip_parallel_solvers: true,
            upfront_lane_reads_graph_pgd_knobs: true,
        }
    }
}

/// Every field this preset requests that `contract` says the engine will drop.
///
/// Ordered as the YAML is (general, attack, solver, bab) so a report reads in
/// file order.
pub(crate) fn validate_preset(preset: &PresetConfig, contract: &EngineContract) -> Vec<Unhonoured> {
    let mut out = Vec::new();

    // --- general -----------------------------------------------------------
    if let Some(device) = preset.general.device.as_deref() {
        if device.eq_ignore_ascii_case("wgpu") && !contract.wgpu_proof_authority {
            out.push(Unhonoured {
                field: "general.device",
                requested: device.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour:
                    "this build has no public WGPU proof constructor, so the requested \
                     backend uses the sound CPU verifier and emits a fallback receipt",
                citation: "ny_gpu::wgpu_proof_authority; commands::backend::resolve_proof_backend",
                accepted: Some(
                    "a build without the WGPU proof feature has only the sound CPU fallback; \
                     default ny-cli builds enable WGPU and do not report this debt",
                ),
            });
        }
    }
    if let Some(func) = preset.general.loss_reduction_func.as_deref() {
        if !contract.loss_reduction_func {
            out.push(Unhonoured {
                field: "general.loss_reduction_func",
                requested: func.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "ignored; the attack keeps its own fixed objective. \
                                   Attack-side only — every candidate still passes the \
                                   unchanged trusted-oracle gate.",
                citation: "preset::apply — no assignment into BetaCrownConfig",
                accepted: Some("alpha-beta-CROWN key compatibility; attack-direction only"),
            });
        }
    }

    // --- attack ------------------------------------------------------------
    if let Some(order) = preset.attack.pgd_order.as_deref() {
        // Split predicate (#pgd-order-after): `after` IS honoured — compat-free
        // `after` resolves to `ResolvedInitialPgdSchedule::Deferred`
        // (`preset::apply`), and `commands::vnncomp` honours it by suppressing
        // the upfront wrapper so the post-BaB stage spends the slice. Only
        // `middle` (attack interleaved with BaB) still requests a schedule no
        // engine code implements.
        let requests_interleaved = order.trim().eq_ignore_ascii_case("middle");
        if requests_interleaved && !contract.pgd_order_middle_scheduling {
            out.push(Unhonoured {
                field: "attack.pgd_order",
                requested: order.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "PGD enablement, 'before' (upfront) and 'after' \
                                   (deferred post-BaB) ARE honoured; only the \
                                   reference's interleaved 'middle' placement is not, \
                                   so PGD runs on the upfront schedule via the explicit \
                                   ny_pgd_order_compat contract. Scheduling only — \
                                   acceptance is the unchanged trusted-oracle gate.",
                citation: "preset::apply::apply_attack_preset — 'middle' arm",
                accepted: Some(
                    "ny implements enablement plus the 'before' and deferred 'after' \
                     placements; interleaved 'middle' has no engine counterpart",
                ),
            });
        }
    }
    if let Some(tolerance) = preset.attack.attack_tolerance {
        if !contract.attack_tolerance {
            out.push(Unhonoured {
                field: "attack.attack_tolerance",
                requested: tolerance.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "ignored; acceptance stays at ZERO tolerance. Ignoring a \
                                   requested slack is strictly STRICTER than honouring it, \
                                   so it can only cost `sat` rows, never admit a false one.",
                citation: "preset::apply — no assignment into BetaCrownConfig",
                accepted: Some("stricter than requested; safe by construction"),
            });
        }
    }
    // Both of the next two ARE applied into `BetaCrownConfig` — which is exactly
    // why they look honoured — but the only reader of either is the graph
    // disjunctive PGD loop. A category that spends its budget in the wrapper's
    // upfront DLR-APGD lane never enters that loop, so the request is dropped
    // with no trace at all. Conditional scope, because a preset whose rows DO
    // take the graph-PGD route gets both knobs honoured in full.
    if preset.attack.surrogate_sign_gradient == Some(true)
        && !contract.upfront_lane_reads_graph_pgd_knobs
    {
        out.push(Unhonoured {
            field: "attack.surrogate_sign_gradient",
            requested: "true".to_string(),
            severity: ContractSeverity::PerformanceOnly,
            scope: UnhonouredScope::Conditional(
                "the row's budget goes to the upfront DLR-APGD lane rather than graph \
                 disjunctive PGD",
            ),
            engine_behaviour:
                "ignored on that route; the attack runs without the straight-through \
                               sign-gradient surrogate. Losing an attack technique can only cost \
                               `sat` rows — acceptance is the unchanged trusted-oracle gate — so \
                               it fails in the safe direction.",
            citation: "beta_crown::verify::graph_pgd — sole reader of pgd_surrogate_sign_gradient",
            accepted: Some(
                "measured inert on traffic_signs_recognition_2023 (0 graph-PGD entries at the \
                 480s budget); routing it upfront was measured to convert 0 rows",
            ),
        });
    }
    if let Some(mode) = preset.attack.attack_mode.as_deref() {
        if mode.to_lowercase().contains("gama") && !contract.upfront_lane_reads_graph_pgd_knobs {
            out.push(Unhonoured {
                field: "attack.attack_mode",
                requested: mode.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Conditional(
                    "the row's budget goes to the upfront DLR-APGD lane rather than graph \
                     disjunctive PGD",
                ),
                engine_behaviour:
                    "the GAMA half is ignored on that route; the diversified-restart \
                                   half still applies. Same safe direction as above — a weaker \
                                   attack loses falsifications, it cannot admit a false one.",
                citation: "beta_crown::verify::graph_pgd — sole reader of the GAMA arm",
                accepted: Some(
                    "measured inert on traffic_signs_recognition_2023 (0 'gama' occurrences at \
                     the 480s budget)",
                ),
            });
        }
    }

    // --- solver ------------------------------------------------------------
    if let Some(solvers) = preset.solver.mip.parallel_solvers {
        if !contract.mip_parallel_solvers {
            out.push(Unhonoured {
                field: "solver.mip.parallel_solvers",
                requested: solvers.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "ignored; the MIP solver runs a single lane. Throughput \
                                   only — the certificate a lane emits is unchanged.",
                citation: "preset::MipPreset::parallel_solvers — reserved, no consumer",
                accepted: Some("reserved for the phase-split racing mode (designs/scip.md)"),
            });
        }
    }

    // --- bab ---------------------------------------------------------------
    if preset.bab.clip.interm_domain == Some(true) && !contract.sequential_clip_interm_domain {
        out.push(Unhonoured {
            field: "bab.clip.interm_domain",
            requested: "true".to_string(),
            severity: ContractSeverity::VerdictAffecting,
            scope: UnhonouredScope::Conditional(
                "the SEQUENTIAL engine runs (non-conv nets); the GRAPH engine reads \
                 enable_clip_interm_domain and IS unaffected",
            ),
            engine_behaviour: "the optional tightening is SKIPPED and the caller's already-valid \
                 enclosure is returned unchanged, so bounds are LOOSER — more `unknown`, \
                 never a narrower bound and never a false `unsat`. This used to raise \
                 SoundnessRefusal for every child, which collapsed BaB at the root; that \
                 is fixed, and the skip is the fix.",
            citation: "ny_propagate::sequential_clip_interm_domain_supported \
                       (beta_crown::engine::domain::clip)",
            accepted: Some(
                "sequential IntermediateLinearBounds are layer-output-relative and lack \
                 certified input-relative provenance; feeding them to the input-space \
                 clipping solver would intersect against constraints in the WRONG frame, \
                 which is a false-`unsat` generator. Skipping is the sound direction.",
            ),
        });
    }
    // #contract-bool-polarity: report only `Some(true)`. These fields are
    // ignored by the engine, so a preset asking for `false` is getting exactly
    // what it requested — flagging that is a false positive, and under
    // `NY_PRESET_STRICT=all` it would abort runs that are perfectly correct
    // (`vnncomp24/ml4acopf.yaml:54`, `vnncomp25/ml4acopf_2024.yaml:122`).
    if let Some(pruning) = preset.bab.pruning_in_iteration.filter(|v| *v) {
        if !contract.pruning_in_iteration {
            out.push(Unhonoured {
                field: "bab.pruning_in_iteration",
                requested: pruning.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "ignored; the optimizer keeps every domain in the \
                                   iteration. More work, never a looser bound.",
                citation: "preset::apply — AlphaCrownConfig field declared but unread",
                accepted: Some("conservative direction; alpha-beta-CROWN key compatibility"),
            });
        }
    }
    let nonlinear = &preset.bab.branching.nonlinear_split;
    if let Some(filter) = nonlinear.filter.filter(|v| *v) {
        if !contract.nonlinear_split_filters {
            out.push(Unhonoured {
                field: "bab.branching.nonlinear_split.filter",
                requested: filter.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "ignored; GenBaB branching evaluates its unfiltered \
                                   candidate set. Search order and cost only — a branch \
                                   decision cannot make a child's bound unsound.",
                citation: "preset::apply — no assignment into NonlinearBranchingConfig",
                accepted: Some("branching heuristic only; cannot affect enclosure validity"),
            });
        }
    }
    if let Some(filter_beta) = nonlinear.filter_beta.filter(|v| *v) {
        if !contract.nonlinear_split_filters {
            out.push(Unhonoured {
                field: "bab.branching.nonlinear_split.filter_beta",
                requested: filter_beta.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "ignored; GenBaB branching evaluates its unfiltered \
                                   candidate set. Search order and cost only.",
                citation: "preset::apply — no assignment into NonlinearBranchingConfig",
                accepted: Some("branching heuristic only; cannot affect enclosure validity"),
            });
        }
    }
    if let Some(nodes) = preset.bab.invprop.apply_output_constraints_to.as_ref() {
        if !contract.invprop_preset_knobs {
            out.push(Unhonoured {
                field: "bab.invprop.apply_output_constraints_to",
                requested: format!("{} node(s)", nodes.len()),
                severity: ContractSeverity::VerdictAffecting,
                scope: UnhonouredScope::Always,
                engine_behaviour:
                    "ignored; no output constraint is propagated backward for these nodes. \
                     INVPROP is a bound TIGHTENER, so dropping it leaves bounds LOOSER — \
                     lost proofs, never a narrower bound. ny's INVPROP is driven by the \
                     --invprop-* CLI flags instead.",
                citation: "preset::apply — InvpropPreset parsed, not applied",
                accepted: Some(
                    "omitting a tightener is the safe direction; the capability exists \
                     behind CLI flags, so this is a plumbing gap, not a missing algorithm",
                ),
            });
        }
    }
    if let Some(share) = preset.bab.invprop.share_gammas.filter(|v| *v) {
        if !contract.invprop_preset_knobs {
            out.push(Unhonoured {
                field: "bab.invprop.share_gammas",
                requested: share.to_string(),
                severity: ContractSeverity::PerformanceOnly,
                scope: UnhonouredScope::Always,
                engine_behaviour: "ignored; γ parameterization comes from the --invprop-* \
                                   CLI flags. A parameterization choice changes the \
                                   optimum reached, not the validity of the relaxation.",
                citation: "preset::apply — InvpropPreset parsed, not applied",
                accepted: Some("CLI flag carries the same knob"),
            });
        }
    }

    out
}

/// Validate `preset` against this build and report.
///
/// Warnings are emitted once per preset PATH so a category that loads its
/// preset per instance does not flood the log; the fatal decision is recomputed
/// every call, so deduplication can never swallow a refusal.
pub(crate) fn enforce_preset_contract(path: &Path, preset: &PresetConfig) -> Result<()> {
    let unhonoured = validate_preset(preset, &EngineContract::current());
    if unhonoured.is_empty() {
        return Ok(());
    }
    let mode = strict_mode();

    if first_report_for(path) {
        for entry in &unhonoured {
            let line = entry.render();
            if mode == StrictMode::Off {
                debug!("preset contract ({}): {line}", path.display());
            } else {
                warn!("preset contract ({}): {line}", path.display());
            }
        }
    }

    let fatal: Vec<&Unhonoured> = unhonoured.iter().filter(|u| u.is_fatal(mode)).collect();
    if fatal.is_empty() {
        return Ok(());
    }
    let detail = fatal
        .iter()
        .map(|entry| format!("  - {}", entry.render()))
        .collect::<Vec<_>>()
        .join("\n");
    bail!(
        "preset {} requests {} field(s) this build does not honour:\n{}\n\
         Refusing to start: a configuration that is silently not the configuration you \
         asked for is the bug class this check exists to end. Fix the preset, implement \
         the field, or set {}=0 to run anyway.",
        path.display(),
        fatal.len(),
        detail,
        STRICT_ENV,
    );
}

/// Has this preset path already had its report emitted in this process?
///
/// Fails OPEN (reports again) if the mutex is poisoned: a duplicate warning is
/// harmless, a swallowed one is not.
fn first_report_for(path: &Path) -> bool {
    static REPORTED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    let reported = REPORTED.get_or_init(|| Mutex::new(HashSet::new()));
    match reported.lock() {
        Ok(mut seen) => seen.insert(path.to_path_buf()),
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::{ClipPreset, GeneralPreset, InvpropPreset, PresetConfig};

    fn preset_with_device(device: &str) -> PresetConfig {
        PresetConfig {
            general: GeneralPreset {
                device: Some(device.to_string()),
                ..GeneralPreset::default()
            },
            ..PresetConfig::default()
        }
    }

    #[test]
    fn strict_mode_defaults_to_warn_and_only_zero_escapes() {
        assert_eq!(strict_mode_from_value(None), StrictMode::Default);
        assert_eq!(strict_mode_from_value(Some("1")), StrictMode::Default);
        assert_eq!(strict_mode_from_value(Some("0")), StrictMode::Off);
        assert_eq!(strict_mode_from_value(Some(" OFF ")), StrictMode::Off);
        assert_eq!(strict_mode_from_value(Some("all")), StrictMode::All);
        // A typo must NOT silently disable the check.
        assert_eq!(strict_mode_from_value(Some("of")), StrictMode::Default);
        assert_eq!(strict_mode_from_value(Some("")), StrictMode::Default);
    }

    #[test]
    fn empty_preset_reports_nothing() {
        let preset = PresetConfig::default();
        assert!(validate_preset(&preset, &EngineContract::current()).is_empty());
    }

    #[test]
    fn current_build_exposes_the_wgpu_proof_qualification_path() {
        let preset = preset_with_device("wgpu");
        assert!(validate_preset(&preset, &EngineContract::current()).is_empty());
    }

    #[test]
    fn wgpu_device_is_reported_when_proof_authority_is_withheld() {
        let preset = preset_with_device("wgpu");
        let contract = EngineContract {
            wgpu_proof_authority: false,
            ..EngineContract::all_honoured()
        };
        let reported = validate_preset(&preset, &contract);
        assert_eq!(
            reported.len(),
            1,
            "expected exactly one entry: {reported:?}"
        );
        assert_eq!(reported[0].field, "general.device");
        assert_eq!(reported[0].severity, ContractSeverity::PerformanceOnly);
    }

    #[test]
    fn cpu_device_is_always_honoured() {
        let preset = preset_with_device("cpu");
        assert!(validate_preset(&preset, &EngineContract::current()).is_empty());
    }

    /// Each predicate must be gated on the CAPABILITY, not written as an
    /// unconditional complaint — otherwise the entry would survive the fix it
    /// is asking for.
    #[test]
    fn every_entry_disappears_when_the_engine_honours_it() {
        let mut preset = PresetConfig {
            general: GeneralPreset {
                device: Some("wgpu".to_string()),
                loss_reduction_func: Some("max".to_string()),
                ..GeneralPreset::default()
            },
            ..PresetConfig::default()
        };
        preset.attack.pgd_order = Some("middle".to_string());
        preset.attack.attack_tolerance = Some(1e-4);
        preset.solver.mip.parallel_solvers = Some(4);
        preset.bab.pruning_in_iteration = Some(true);
        preset.bab.branching.nonlinear_split.filter = Some(true);
        preset.bab.branching.nonlinear_split.filter_beta = Some(true);
        preset.bab.invprop = InvpropPreset {
            apply_output_constraints_to: Some(vec!["node".to_string()]),
            share_gammas: Some(true),
        };
        preset.bab.clip = ClipPreset {
            interm_domain: Some(true),
            ..ClipPreset::default()
        };

        let known_debt = EngineContract {
            wgpu_proof_authority: false,
            ..EngineContract::current()
        };
        let now: Vec<&str> = validate_preset(&preset, &known_debt)
            .iter()
            .map(|entry| entry.field)
            .collect();
        assert_eq!(
            now,
            vec![
                "general.device",
                "general.loss_reduction_func",
                "attack.pgd_order",
                "attack.attack_tolerance",
                "solver.mip.parallel_solvers",
                "bab.clip.interm_domain",
                "bab.pruning_in_iteration",
                "bab.branching.nonlinear_split.filter",
                "bab.branching.nonlinear_split.filter_beta",
                "bab.invprop.apply_output_constraints_to",
                "bab.invprop.share_gammas",
            ],
            "the registry must report every knob this preset sets, in YAML order"
        );
        let fixed = validate_preset(&preset, &EngineContract::all_honoured());
        assert!(
            fixed.is_empty(),
            "an entry that survives a fully-capable engine is an unconditional complaint, \
             not a contract check: {fixed:?}"
        );
    }

    /// `pgd_order` decides ENABLEMENT plus PLACEMENT, and ny honours every
    /// placement except the interleaved `middle`: `before` runs upfront,
    /// `after` runs DEFERRED end-to-end (`preset::apply` resolves compat-free
    /// `after` to `Deferred`; `commands::vnncomp` suppresses the upfront
    /// wrapper and the post-BaB stage spends the slice). Reporting `after` as
    /// unhonoured was a false soundness-of-configuration warning — and a hard
    /// abort under `NY_PRESET_STRICT=all` — for a correctly configured preset.
    /// Only `middle` may be reported.
    #[test]
    fn only_the_interleaved_middle_pgd_order_is_reported() {
        for honoured in [
            "before",
            "after",
            "AFTER",
            "skip",
            "none",
            "disabled",
            "input_bab",
        ] {
            let mut preset = PresetConfig::default();
            preset.attack.pgd_order = Some(honoured.to_string());
            assert!(
                validate_preset(&preset, &EngineContract::current()).is_empty(),
                "pgd_order '{honoured}' is honoured and must not be reported"
            );
        }
        for reported in ["middle", "MIDDLE"] {
            let mut preset = PresetConfig::default();
            preset.attack.pgd_order = Some(reported.to_string());
            let out = validate_preset(&preset, &EngineContract::current());
            assert_eq!(
                out.len(),
                1,
                "pgd_order '{reported}' requests the unimplemented interleaved schedule"
            );
            assert_eq!(out[0].field, "attack.pgd_order");
        }
    }

    /// `interm_domain: false` asks for nothing, so there is nothing to drop.
    #[test]
    fn interm_domain_false_is_not_a_mismatch() {
        let mut preset = PresetConfig::default();
        preset.bab.clip.interm_domain = Some(false);
        assert!(validate_preset(&preset, &EngineContract::current()).is_empty());
    }

    /// The escape hatch must never be able to turn a mismatch into an abort,
    /// and `all` must never let one through.
    #[test]
    fn strict_mode_orders_the_three_positions() {
        let mut preset = PresetConfig::default();
        preset.bab.clip.interm_domain = Some(true);
        let entries = validate_preset(&preset, &EngineContract::current());
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.severity, ContractSeverity::VerdictAffecting);
        assert!(!entry.is_fatal(StrictMode::Off));
        assert!(
            !entry.is_fatal(StrictMode::Default),
            "an ACCEPTED verdict-affecting entry warns; aborting would trade weak \
             verdicts for no verdicts and buy no soundness"
        );
        assert!(entry.is_fatal(StrictMode::All));
    }

    /// An unaccepted verdict-affecting entry is the one class that stops a run
    /// by default. This is the ratchet: adding such a field without arguing its
    /// failure direction fails loudly at load.
    #[test]
    fn unaccepted_verdict_affecting_entry_is_fatal_by_default() {
        let entry = Unhonoured {
            field: "bab.hypothetical",
            requested: "true".to_string(),
            severity: ContractSeverity::VerdictAffecting,
            scope: UnhonouredScope::Always,
            engine_behaviour: "ignored",
            citation: "nowhere",
            accepted: None,
        };
        assert!(entry.is_fatal(StrictMode::Default));
        assert!(entry.is_fatal(StrictMode::All));
        assert!(!entry.is_fatal(StrictMode::Off));
    }

    /// The rendered line has to carry everything an operator needs to act:
    /// which key, what it asked for, what happened instead, and where.
    #[test]
    fn rendered_line_names_field_value_behaviour_and_citation() {
        let preset = preset_with_device("wgpu");
        let contract = EngineContract {
            wgpu_proof_authority: false,
            ..EngineContract::all_honoured()
        };
        let entries = validate_preset(&preset, &contract);
        let line = entries[0].render();
        assert!(line.contains("general.device"), "{line}");
        assert!(line.contains("wgpu"), "{line}");
        assert!(line.contains("CPU verifier"), "{line}");
        assert!(line.contains("dropped at:"), "{line}");
        assert!(line.contains("accepted debt:"), "{line}");
    }
}
