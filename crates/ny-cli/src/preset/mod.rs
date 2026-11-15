// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP benchmark preset configuration loader.
//! Loads per-benchmark YAML presets that override `BetaCrownConfig` defaults.
//! CLI flags take precedence over preset values.

use anyhow::{bail, Context, Result};
use ny_propagate::DepthTwoBranchLookaheadConfig;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
mod apply;
mod branching;
mod contract;
mod value_parse;
pub(crate) use apply::{
    apply_preset, apply_relational_point_vjp_policy, effective_alpha_zero_yield_frac,
    resolve_initial_pgd_schedule, resolve_use_alpha_from_bound_prop_method, validate_preset,
    ResolvedInitialPgdSchedule,
};
pub(crate) use branching::resolve_branching;
pub(crate) use contract::enforce_preset_contract;

#[cfg(test)]
mod alpha_preset_tests;
#[cfg(test)]
mod attack_concurrent_preset_tests;
#[cfg(test)]
mod attack_mode_tests;
#[cfg(test)]
mod attack_open_disjunct_tests;
#[cfg(test)]
mod backend_capability_tests;
#[cfg(test)]
mod bound_prop_mode_tests;
#[cfg(test)]
mod contract_tests;
#[cfg(test)]
mod conv_mode_tests;
#[cfg(test)]
mod cut_authority_tests;
#[cfg(test)]
mod edge_wave_cap_tests;
#[cfg(test)]
mod gpu_bab_sidecar_tests;
#[cfg(test)]
mod graph_search_preset_tests;
#[cfg(test)]
mod kfsb_single_shot_preset_tests;
#[cfg(test)]
mod la_warm_frontier_preset_tests;
/// #lever-reachability: PROMOTION IS NOT ARMING. Pins that no shipped yaml sets
/// any key from the 2026-08-23 delivery batch, so every promoted lever still
/// resolves through an empty cell and the batch is byte-identical.
#[cfg(test)]
mod lever_delivery_pin_tests;
#[cfg(test)]
mod linearizenn_2024_tests;
#[cfg(test)]
mod mip_governor_tests;
#[cfg(test)]
mod model_load_smoke_tests;
#[cfg(test)]
mod phase_budget_tests;
#[cfg(test)]
mod preset_resolution_pin_tests;
#[cfg(test)]
mod pwl_lookup_preset_tests;
#[cfg(test)]
mod relusplitter_biasfield_input_split_tests;
#[cfg(test)]
mod relusplitter_rsplitter_matrix_input_split_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod vnncomp_preset_tests;

/// Top-level preset configuration file structure.
///
/// Mirrors alpha-beta-CROWN's YAML config format for compatibility.
/// Supports both alpha-beta-CROWN's structure (solver: for batch_size, bab: for branching)
/// and ny's simplified structure (all under bab:). Every mapping in this
/// schema is closed: compatibility spellings are declared as Serde aliases,
/// while unknown keys are rejected instead of being silently ignored. There is
/// intentionally no flattened extension map whose contents could bypass that
/// fail-closed contract.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct PresetConfig {
    /// General settings (root_path, csv_name, device).
    #[serde(default)]
    pub(crate) general: GeneralPreset,

    /// Model-loading options (alpha-beta-CROWN compatibility).
    #[serde(default)]
    pub(crate) model: ModelPreset,

    /// Attack/counterexample settings.
    #[serde(default)]
    pub(crate) attack: AttackPreset,

    /// Solver configuration (alpha-beta-CROWN uses this for batch_size, alpha/beta-crown settings).
    /// These values are merged into bab during apply_preset.
    #[serde(default)]
    pub(crate) solver: SolverPreset,

    /// Branch-and-bound configuration (core BetaCrownConfig overrides).
    #[serde(default)]
    pub(crate) bab: BabPreset,

    /// Margin-row twin-wall lane settings (#twinwall / #epoch-bab).
    #[serde(default)]
    pub(crate) margin_row: MarginRowPreset,

    /// Certified double-double zonotope admission overrides
    /// (#dd-zonotope / #metaroom-ddzono).
    #[serde(default)]
    pub(crate) dd_zonotope: DdZonotopePreset,

    /// Exact-instance admission for the independently certified MetaRoom
    /// constrained-zonotope verdict lane (#cz-hybrid).
    #[serde(default)]
    pub(crate) cz_hybrid: CzHybridPreset,

    /// `#nonlinear-vnnlib` unsat-half settings (#acc-strict-intervals).
    #[serde(default)]
    pub(crate) nonlinear_vnnlib: NonlinearVnnlibPreset,

    /// #router-r1: the runtime router's typed mode (`router.mode`).
    #[serde(default)]
    pub(crate) router: RouterPreset,

    /// #router-r1: the runtime optimizer's typed mode (`optimizer.mode`).
    #[serde(default)]
    pub(crate) optimizer: OptimizerPreset,
}

/// #router-r1 (ROUTER_PROGRAM §11): the runtime router.
///
/// An OUTER key like `nonlinear_vnnlib`: it configures no engine field, so it
/// is deliberately absent from `preset::apply`'s `BetaCrownConfig` mapping.
/// It reaches the run through the generic axis projection
/// (`validate_delivered_lever_values` checks the token against
/// `NY_ROUTER_MODE`'s declaration; `commands::vnncomp::
/// preset_delivered_config_value` projects it into the frozen `LeverSet`), and
/// ONE reader consumes it: `crate::router::RouterPolicy::from_levers`.
///
/// `None` (every shipped preset; pinned by `lever_delivery_pin_tests`) is
/// `off`: dark and byte-identical. No category yaml may arm it (§2.5).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct RouterPreset {
    /// `off | shadow | on` (#router-r2: `on` acts through acting set 1).
    #[serde(default)]
    pub(crate) mode: Option<ny_core::router::RouterMode>,
    /// #router-r2: the seven acting-rule ablation switches (`router.rules.*`);
    /// #router-r4: and acting set 3's two (`lease`, `far_hand_on`);
    /// #router-r5: and acting set 4's two (`route`, `gap_contract`).
    #[serde(default)]
    pub(crate) rules: RouterRulesPreset,
}

/// #router-r2 (ROUTER_PROGRAM §10 R2): one ablation switch per acting rule,
/// delivered by the same generic axis projection as `router.mode` (declared
/// `NY_ROUTER_RULE_*`, `UNSAFE_AXES`: never searched). `None` (every shipped
/// preset) resolves to the declaration default `true` — the rule participates
/// when the router acts, which it never does unless `router.mode: on`.
/// Reader: `crate::router::RouterPolicy::from_levers`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct RouterRulesPreset {
    /// CR-A: which lanes the deadline authority governs.
    #[serde(default)]
    pub(crate) admission: Option<bool>,
    /// CR-C: the attack slice committed once, up front.
    #[serde(default)]
    pub(crate) contracts: Option<bool>,
    /// CR-1: the one deadline authority.
    #[serde(default)]
    pub(crate) cap_authority: Option<bool>,
    /// CR-T: the verifier is the terminal.
    #[serde(default)]
    pub(crate) terminal: Option<bool>,
    /// CR-H: hand-on of unspent time.
    #[serde(default)]
    pub(crate) hand_on: Option<bool>,
    /// CR-R: the publish grace.
    #[serde(default)]
    pub(crate) publish_grace: Option<bool>,
    /// CR-K: derived caps.
    #[serde(default)]
    pub(crate) derived_caps: Option<bool>,
    /// #router-r4 CR-L: the memory lease against the declared host's static
    /// envelope.
    #[serde(default)]
    pub(crate) lease: Option<bool>,
    /// #router-r4 CR-F: the FAR hand-on to a strictly sealed recipient.
    #[serde(default)]
    pub(crate) far_hand_on: Option<bool>,
    /// #router-r5 CR-RS: capabilities routed per instance, only at SR.seal.
    #[serde(default)]
    pub(crate) route: Option<bool>,
    /// #router-r5 CR-G: gap-order contracts inside multi-objective lanes.
    #[serde(default)]
    pub(crate) gap_contract: Option<bool>,
}

/// #router-r1 (ROUTER_PROGRAM §5): the runtime optimizer. Same delivery as
/// [`RouterPreset`], reader `crate::router::RouterPolicy::from_levers`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct OptimizerPreset {
    /// `off | shadow | on` (#router-r3: `on` acts through acting set 2).
    #[serde(default)]
    pub(crate) mode: Option<ny_core::router::OptimizerMode>,
    /// #router-r3: the four O-STOP rule ablation switches (`optimizer.rules.*`).
    #[serde(default)]
    pub(crate) rules: OptimizerRulesPreset,
    /// #router-r7: read the compiled offline cost-model prior for planning
    /// (`NY_OPTIMIZER_PRIORS`, `UNSAFE_AXES`: never searched). `None` (every
    /// shipped preset) is the declaration default `false`: no prior is read.
    #[serde(default)]
    pub(crate) priors: Option<bool>,
}

/// #router-r3 (ROUTER_PROGRAM §10 R3): one ablation switch per O-STOP rule
/// (and, #router-r6, per O-TRIAL rule: `trial_*`),
/// delivered by the same generic axis projection as `optimizer.mode`
/// (declared `NY_OPTIMIZER_RULE_*`, `UNSAFE_AXES`: never searched). `None`
/// (every shipped preset) resolves to the declaration default `true` — the
/// rule participates when the optimizer acts, which it never does unless
/// `optimizer.mode: on`. Reader: `crate::router::RouterPolicy::from_levers`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct OptimizerRulesPreset {
    /// `rule.governor`: the alpha ascent's relative self-decay stop.
    #[serde(default)]
    pub(crate) governor: Option<bool>,
    /// `rule.ia_yield`: the interm-alpha candidates' relative self-decay stop.
    #[serde(default)]
    pub(crate) ia_yield: Option<bool>,
    /// `rule.attack_defer` (CR-ATK): defer a flat upfront attack.
    #[serde(default)]
    pub(crate) attack_defer: Option<bool>,
    /// `rule.root_admission`: skip a root phase that does not fit.
    #[serde(default)]
    pub(crate) root_admission: Option<bool>,
    /// #router-r6 `rule.trial_width`: AIMD on a fast lane's width.
    #[serde(default)]
    pub(crate) trial_width: Option<bool>,
    /// #router-r6 `rule.trial_lr_backoff`: the root ascent's step backoff.
    #[serde(default)]
    pub(crate) trial_lr_backoff: Option<bool>,
    /// #router-r6 `rule.trial_split_restart`: one restart with the other
    /// split heuristic.
    #[serde(default)]
    pub(crate) trial_split_restart: Option<bool>,
    /// #router-r6 `rule.trial_attack_halving`: successive halving over the
    /// upfront attack's step pool.
    #[serde(default)]
    pub(crate) trial_attack_halving: Option<bool>,
}

/// `#nonlinear-vnnlib` unsat-half settings (#acc-strict-intervals).
///
/// Like `bab.phase_budget.results_flush_grace_secs`, this is an OUTER
/// WRAPPER key: the non-linear lane runs in the VNN-COMP router's
/// model-load/parse-refusal arm, not in the β-CROWN engine, so it is
/// deliberately absent from `preset::apply` (and from `BetaCrownConfig`) and
/// is consumed directly from the immutable `PresetConfig` snapshot by
/// `commands::vnncomp::nonlinear_case_split_from_snapshot`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct NonlinearVnnlibPreset {
    /// Refute interval-UNDECIDED cells by an exact pivot case split with
    /// open-endpoint interval logic
    /// (`ny_onnx::vnnlib::nonlinear::NonLinearFormula::holds_over_box_split`).
    ///
    /// `None`/`false` (every shipped preset) is today's lane, byte for byte:
    /// no split is attempted, no guard is applied and no receipt is printed.
    ///
    /// When `true`, the lane ALSO applies an fp32-replay guard to every cell
    /// (closed and split alike): the network enclosure is computed over the
    /// f32 hull of the cell and widened by `2^-16 * max(1, |Y|)`. Without it
    /// the split would remove the only obstacle — the Y = 0 straddle — between
    /// the real-arithmetic proof and four field-SAT ACC rows (4, 16, 28, 39)
    /// whose only violation is an fp32 replay artifact 5.5e-7 above the real
    /// clamp (see `commands::vnncomp::try_nonlinear_unsat`).
    ///
    /// Target: adaptive_cruise_control_non_linear_2026 rows 15 and 38, whose
    /// formula is false for every real point but only through strict
    /// inequalities. Arm only behind that category's own A/B and full moat.
    pub(crate) case_split: Option<bool>,
}

/// Fail-closed exact-basename allowlist for the wired MetaRoom CZ lane. Both
/// fields must be present and match the harness arguments; a partial or absent
/// section never engages the lane.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct CzHybridPreset {
    pub(crate) only_onnx: Option<String>,
    pub(crate) only_vnnlib: Option<String>,

    /// Largest share of the INSTANCE budget the wired CZ lane may consume
    /// before it is cut off and the ordinary pipeline is allowed to run
    /// (#cz-hybrid-budget). `None` (every shipped preset) means 1.0, i.e.
    /// today's behaviour: the lane is handed `instance_deadline - 5s`.
    ///
    /// MEASURED, metaroom_2023 6cnn_ry_39_6 spec_idx_119 at its official 210 s
    /// budget: the lane consumed 188.7 s for `proved_unsat=false,
    /// strictly_positive=0/19`, the fall-through pipeline was never told, it
    /// re-derived its own budget from scratch (`total_s=148.000
    /// elapsed_s=0.000`) with ~21 s actually left, overran, and ny's own
    /// watchdog SIGKILLed the verifier at 216 s — 6 s past the official budget
    /// and with NO `Result:` line printed at all.
    ///
    /// SOUNDNESS: it can only SHORTEN a lane that publishes exactly one
    /// verdict, a fully certified `unsat`. A shortened lane falls through
    /// unchanged, so a wrong value costs a proof and can never produce one.
    /// Values outside `(0.0, 1.0]` and non-finite values are REFUSED by
    /// semantic validation rather than silently clamped.
    #[serde(default)]
    pub(crate) max_budget_fraction: Option<f64>,
}

/// Per-category ADMISSION-override plumbing for the certified double-double
/// zonotope (`#dd-zonotope`). A future category can use this to make the lane
/// reachable from the scored `ny vnncomp v1` entry point without environment
/// variables; no shipped preset currently supplies a section.
///
/// Every field is an `Option`: an absent section (every existing preset) is
/// byte-identical to today. These knobs resize only the fail-closed detector's
/// blast-radius/resource caps; the lane's soundness gates (self-policing
/// precision gate, rounding-channel safety factor, `dd_selfcheck`, the
/// FP-environment probe, outward f64→f32 narrowing at the verdict) are not
/// preset-reachable, and explicitly set `NY_DD_ZONOTOPE_*` /
/// `NY_DD_ZONO_INTERM` environment knobs keep precedence
/// (`DdZonoConfig::with_admission_overrides`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct DdZonotopePreset {
    /// Minimum input volume the detector admits
    /// (`DdZonoConfig::min_input_numel`, built-in default 50,000).
    #[serde(default)]
    pub(crate) min_input_numel: Option<usize>,
    /// Perturbed-input-coordinate cap (`DdZonoConfig::max_k`, default 128).
    #[serde(default)]
    pub(crate) max_k: Option<usize>,
    /// Live generator-column cap (`DdZonoConfig::max_generators`, default 512).
    /// Exceeding it still FAILS CLOSED mid-pass.
    #[serde(default)]
    pub(crate) max_generators: Option<usize>,
    /// `#dd-zono-interm`: also collect the certified per-node enclosures so
    /// the multi-objective root intersects them into its stored intermediate
    /// bounds (intersection of certified enclosures can only tighten).
    #[serde(default)]
    pub(crate) interm_intersect: Option<bool>,
}

/// Margin-row lane budget policy, per benchmark category (#epoch-bab).
///
/// The lane normally runs after the internal verifier returns unknown/timeout.
/// Its historical runtime reserve default is 45 seconds, but that reserve is
/// held only after exact structural admission; an explicit typed zero releases
/// it. Category presets may instead run the lane concurrently, or enable the
/// adaptive budget-share route that releases the reserve and grants sparse
/// root CROWN authority. Those controls are one coupled schedule: configuration
/// alone is not evidence that time was held, released, or overlapped.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct MarginRowPreset {
    /// Seconds reserved from the internal verifier for the lane.
    pub(crate) reserve_secs: Option<u64>,

    /// Release a structurally admitted reserve when it reaches the configured
    /// share of the internal verifier budget, and route that case through the
    /// sparse-root alpha/beta policy. Default `None`/`false` preserves the
    /// historical fixed-reserve policy. This is budget arithmetic, not an
    /// instance-name allowlist.
    pub(crate) adaptive_reserve: Option<bool>,

    /// Ceiling on the reserve, as a fraction of the INTERNAL verifier budget
    /// (not the scored budget). Typed form of `NY_MARGIN_ROW_RESERVE_MAX_FRAC`.
    ///
    /// `reserve_secs` is a fixed number of seconds, so its share of the budget
    /// GROWS as the budget shrinks — the shipped 45 s is 47% of a 95 s internal
    /// tier but only 24% of a 190 s one, which is why a scored 100 s run
    /// behaves nothing like the first 100 s of a 200 s run (measured; see
    /// `capped_reserve_secs` in `commands::margin_row_bab`). This key makes the
    /// reserve proportional instead.
    ///
    /// Only finite values strictly inside `(0, 1)` arm the ceiling; anything
    /// else — absent, `0`, `>= 1`, non-finite — declines it and keeps the
    /// shipped fixed-seconds policy byte-identically. `1.0` is rejected on
    /// purpose: it is a no-op ceiling, and accepting it would invite "0 means
    /// release" confusion with `reserve_secs`, which is the full-release knob.
    ///
    /// The env var still wins wherever it is PRESENT, including as a kill
    /// switch: `NY_MARGIN_ROW_RESERVE_MAX_FRAC=0` (or any value the parser
    /// declines) disables a ceiling a preset asked for.
    ///
    /// No shipped yaml sets this; arming it needs a measured A/B.
    pub(crate) reserve_max_frac: Option<f32>,

    /// Share of the INTERNAL verifier budget at or above which
    /// `adaptive_reserve` releases the reserve entirely (default `0.40`).
    /// Typed form of `NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC`.
    ///
    /// This is the structural successor to the removed seven-filename
    /// CIFAR100 allowlist: the release now fires on the pathology the
    /// allowlist stood in for — a fixed reserve eating a disproportionate
    /// share of a short budget — rather than on instance identity.
    ///
    /// Same admission rule as `reserve_max_frac`: only finite values strictly
    /// inside `(0, 1)`. Anything else falls back to the shipped default.
    pub(crate) release_frac: Option<f32>,

    /// Run the lane CONCURRENTLY with the internal verifier instead of on the
    /// leftover tail. Typed form of `NY_MARGIN_ROW_CONCURRENT=1`; the env var
    /// still wins where it is present.
    ///
    /// WHY THIS KEY EXISTS (#twinwall-provenance, measured 2026-08-03): the
    /// `tinyimagenet_2024` +67 UNSAT bank (adee6117) and the `cifar100_2024`
    /// +23 bank (c4a8821b) were BOTH produced by the margin-row batch sweep
    /// (`sweep_targets`, `ny vnncomp-research margin-row sweep`), which calls
    /// `run_margin_row_lane` directly with the FULL per-instance budget and no
    /// harness. Their ledger diffs show the harness verdict being overwritten:
    /// `timeout,92 -> unsat,53`. Through `ny vnncomp` the lane only ever gets
    /// `reserve_secs`, and EVERY one of those 67 rows recorded >= 50 s of lane
    /// time, so the shipped 45 s reserve cannot fit even one of them — the
    /// loss is arithmetic, not a regression. This key is the only shipped way
    /// to give the lane what its bank measured it with.
    ///
    /// Requires an EFFECTIVE held reserve of zero: TinyImageNet requests that
    /// directly with `reserve_secs: 0`, while CIFAR100 reaches it through the
    /// adaptive release calculation. Concurrent + a held reserve taxes the
    /// verifier twice (it loses the reserve AND contends), the combination
    /// 31949bcc refused for want of evidence.
    pub(crate) concurrent: Option<bool>,
    /// Arm the SOUND f32 root tableau for this category (typed form of
    /// `NY_MARGIN_ROW_ROOT_F32=1`; the env var still wins where present).
    ///
    /// The root forward conv is memory-bandwidth-bound, and on the 9409-wide
    /// TinyImageNet tableau it is ~99% of the lane's wall time (measured
    /// 2026-08-03: lane 128.8 s of which the BaB tree is 0.5 s / 2
    /// expansions). Halving the streamed width is therefore nearly a 2x lane
    /// speedup.
    ///
    /// SOUND BY CONSTRUCTION: the f32 rounding is charged only into a
    /// certified additive per-neuron concretize slack consumed as pure
    /// loosening (see `margin_row::root::root_f32_requested`). It can lose a
    /// proof, never invent one, and the lane stays fail-closed.
    pub(crate) root_f32: Option<bool>,

    /// `#tableau-f32`: f32 STORAGE for the root tableau walk (typed form of
    /// `NY_MARGIN_ROW_TABLEAU_F32`; the env var wins where present, exact
    /// `1`/`0`).
    ///
    /// `root_f32` made the conv COMPUTE f32 but left every tensor STORED f64,
    /// so the bandwidth-bound walk still streamed 8-byte lanes plus a full
    /// f64->f32 cast pass per conv layer. This key stores the M and D lanes as
    /// `Array2<f32>` end-to-end, halving the streamed bytes of the walk that
    /// is 97% of the tinyimagenet root build (39.93 s of 41.35 s, measured
    /// 2026-08-24 under `NY_MARGIN_ROW_ROOT_TIMING`).
    ///
    /// SOUND BY CONSTRUCTION: conv-lane stores are bit-exact, radius-role
    /// stores are directed UP, and every remaining store perturbation is
    /// charged into the same certified `g_err` concretize slack `root_f32`
    /// uses — pure loosening, oracle-tested per site
    /// (`tableau_f32_store_per_site_containment_oracle`). Requires `root_f32`;
    /// requested without it the build refuses LOUDLY (`[tableau-f32] INERT`)
    /// and runs f64 storage.
    pub(crate) tableau_f32: Option<bool>,

    /// `#domain-stack`: score the candidate children of every domain in a
    /// frontier batch in ONE wide certified backward pass instead of one pass
    /// per domain. The certified pass is linear above R~64 columns on a
    /// 45-60 ms fixed cost, but the lane runs it at R = nf = 3-11 (~13x the
    /// marginal cost) and scoring is 83% of the lane — so this puts the prover
    /// on the linear part of its own cost curve, and is the affordability
    /// precondition for a per-domain joint alpha+beta re-optimisation.
    ///
    /// SCHEDULE ONLY: reductions are column-local, so a stacked score is
    /// bit-identical to the independent one
    /// (`domain_stacked_candidate_scores_match_independent_frontier_bit_exact`).
    /// It cannot move a published bound — only what the lane finishes in budget.
    ///
    /// Since #mr-stack (margin-row throughput plan B5) the same key also runs
    /// the batch's MAIN eval passes (R = nf = 1-3 on cifar100) as one stacked
    /// pass, equally bit-identical (`stacked_main_eval_bit_identical_to_per_domain`,
    /// `expand_batch_trajectory_hash_unchanged`); receipt `[mr-stack]`.
    pub(crate) domain_stack: Option<bool>,

    /// `#mr-errlane` (margin-row throughput plan B3): how the backward engine
    /// carries its CERTIFIED coefficient-error lane — `off` (default, today's
    /// per-column `n x R` matrix, byte-identical), `envelope` (one value per
    /// row on passes with `R >= 16`), `per_channel` (one value per channel on
    /// every pass) or `audit` (publish `off`, shadow both envelopes and record
    /// their looseness; a measurement arm).
    ///
    /// The error lane is measured at exactly half of every certified pass
    /// (B1 probe, `outward_over_parity` 2.00-2.04 on all three nets); an
    /// envelope dominates the per-column lane by monotone induction, so a wrong
    /// value costs proofs (looseness) and cannot manufacture one. Scored runs
    /// consume the key from the frozen LeverSet through its `space.rs` axis;
    /// no shipped yaml sets it.
    pub(crate) err_envelope: Option<ny_propagate::margin_row::ErrLane>,

    /// Deterministic y-pack LRU capacity (typed form of `NY_MARGIN_ROW_LRU`,
    /// historical default 64). A cache hit returns exactly the pack a miss
    /// rebuilds, so this changes memory/recomputation only. The legacy
    /// untrimmed environment value retains precedence, and a present malformed
    /// value suppresses this key and restores 64.
    ///
    /// No shipped yaml sets this key; it exists for canonical scored search.
    pub(crate) lru: Option<usize>,

    /// Parallel best-first frontier width (typed form of
    /// `NY_MARGIN_ROW_PARALLEL`). Canonical typed tokens are exactly
    /// `0|2|4|8|16`: `0` selects the serial frontier and positive values pin
    /// that width without clamping to available Rayon workers. Omission keeps
    /// the measured all-worker default.
    ///
    /// Typed `1` is deliberately invalid: the legacy environment grammar uses
    /// `1` to mean ALL available workers, not width one. The legacy grammar
    /// (`true|on|yes`, `false|off|no`, arbitrary usize, trimming and malformed
    /// fallback) remains authoritative wherever the environment is present.
    /// No shipped yaml sets this key.
    pub(crate) parallel: Option<String>,

    /// #margin-row-branch-width: candidate budget per expansion (head rows).
    ///
    /// Default 8 + 8. MEASURED 2026-08-19 on cifar100 resnet_medium at the
    /// official 100 s budget: `idx_8600_sidx_2721` — the row that timed out at
    /// k=8 through a 4x budget (~3000 expansions), Clip-and-Verify tightening,
    /// and eight eliminated hypotheses — PROVES `unsat` at k=16 in 186
    /// expansions and at k=32 in 173, with the frontier DRAINING (424 open ->
    /// 2) instead of exploding. Narrowing was already known to lose proofs
    /// (k=4 does 5x the search and drops a banked row); widening was never
    /// tested until now. Width only chooses WHICH domains to split — every
    /// candidate is scored by the same certified pass — so a wrong value costs
    /// proofs, never manufactures one.
    pub(crate) k_head: Option<usize>,

    /// #margin-row-branch-width: candidate budget per expansion (trunk rows).
    /// See `k_head`.
    pub(crate) k_trunk: Option<usize>,

    /// #margin-row-adaptive-width: escalate the candidate budget in-flight when
    /// the frontier shows the measured explosion signature (open >= 32 -> k 16,
    /// open >= 256 -> k 32; one-way ratchet, never narrows the configured
    /// base). Rows the base width serves run bit-identically — the trigger
    /// sits above every measured proving-row frontier peak (~26). This is the
    /// non-overfit alternative to pinning `k_head`/`k_trunk` per category.
    pub(crate) k_adaptive: Option<bool>,

    // #parameter-governance: complete typed delivery for the margin-row
    // search surface.  These keys are deliberately absent from every shipped
    // preset; they make existing, default-preserving A/B controls reachable
    // through the scored entry point without qualifying any treatment.
    /// Authoritative margin-row GPU seam (`NY_MARGIN_ROW_GPU`).
    pub(crate) gpu: Option<bool>,

    /// Domain batching for the GPU seam (`NY_MARGIN_ROW_GPU_BATCH`).
    pub(crate) gpu_batch: Option<bool>,

    /// Certified GPU backward transaction (`NY_MARGIN_ROW_GPU_BACKWARD`).
    pub(crate) gpu_backward: Option<bool>,

    /// RETIRED 2026-09-29 (ENGINE_PROGRAM R1 kill line + R2): kept in the
    /// deserializer only so an old yaml is REFUSED with the reason at
    /// validation (`preset::apply`, any value); it is no longer a delivered
    /// key and the lane never arms the route. Measured on the yaml route with
    /// zero `NY_*`: cifar100 c2050 unsat -> timeout and c1590 unsat -> unknown.
    /// What it was:
    ///
    /// `#mr-gpu-eval`: run the certified pass's CONV BACKWARD products (the
    /// signed coefficient product, the per-column error lane's `|W|` product
    /// and the row envelope's `R = 1` product) on the process-global sound f64
    /// GEMM engine — cuBLAS `Dgemm` on a CUDA build — as one full-contraction
    /// GEMM per spatial tile (typed form of `NY_MARGIN_ROW_GPU_CONV`).
    ///
    /// Those products are the O(`k_bwd`) part of every margin-row pass (main
    /// eval, stacked eval, scoring, y-refresh, root eval, backward-interm), and
    /// the GPU otherwise sits idle on this lane.
    ///
    /// SOUND BY CONSTRUCTION: the device consumes the stored f64 operands
    /// bit-for-bit and the lane's UNCHANGED `gamma(k_bwd + 2)` / `certify_up`
    /// charges are summation-order independent, so they cover the device sum
    /// exactly as they cover the CPU kernel's; beta, exceptions and gates stay
    /// host-side. Any refusal (no CUDA engine, deadline, non-finite output, a
    /// transposed conv) runs the unchanged CPU kernel. Receipt
    /// `[mr-gpu-conv] done ... gpu=`; armed with `gpu=0` is VOID. No shipped
    /// yaml sets this key.
    pub(crate) gpu_conv: Option<bool>,

    /// `#frozen-dual-bab` (V1 stage B of the frozen alpha-zonotope dual-ascent
    /// BaB, on stage A's `#frozen-dual-state` query state): run the certified
    /// frozen-dual search as a TIER between the margin-row root and its tree,
    /// on admitted classes the root leaves open. Per class it builds the certified
    /// state (binary64, explicit gamma charges), sorts the unstable rows by the
    /// static box + halfspace score and runs a level-synchronous BFS whose
    /// nodes are pruned ONLY on a certified node bound `> 0` at clamped dual
    /// points. A class leaves the tree's conjunction only when every leaf is
    /// certified and the level ledger closes; every other class is handed to
    /// the tree unchanged.
    ///
    /// Observable on every armed run that reaches the tree driver: one
    /// `[frozen-dual] class=...` line per open class and `[frozen-dual]
    /// armed=1 objectives= closed= nodes= depth_max= secs= ...`. Armed without
    /// the summary is VOID. Shipped CIFAR100 and TinyImageNet presets enable
    /// the bounded automatic tier; the generic declaration remains false.
    pub(crate) frozen_dual_bab: Option<bool>,

    /// `#frozen-dual-bab`: nodes one objective's search may evaluate before it
    /// is handed back open (`NY_MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP`, default
    /// 2,000,000, reader clamp `1..=200M`). Automatic mode additionally caps
    /// nodes at 8192 per objective. Inert unless `frozen_dual_bab`.
    pub(crate) frozen_dual_bab_node_cap: Option<usize>,

    /// `#frozen-dual-bab`: the tier's wall request in seconds
    /// (`NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS`); 0 = bounded automatic mode:
    /// at most 4 s / 5% of remaining time, 8192 nodes per objective and
    /// 256 MiB of state. Dense row estimates >=64 MiB select one objective,
    /// otherwise at most two. A positive request selects manual
    /// mode, capped to three quarters of remaining time. Inert unless
    /// `frozen_dual_bab`.
    pub(crate) frozen_dual_bab_secs: Option<f64>,

    /// `#breakpoint-dual`: untrusted multi-cut and coordinate refinement steps
    /// per frozen-dual node, certified by the unchanged node evaluator. Generic
    /// default 0 keeps the historical path; CIFAR100 and TinyImageNet presets
    /// request 4 steps. The tier clamps to `0..=32`. Inert unless `frozen_dual_bab`.
    pub(crate) frozen_dual_refine_steps: Option<usize>,

    /// `#fd-shared-state` (ENGINE_PROGRAM M3): how the tier builds its
    /// per-class states (`NY_MARGIN_ROW_FROZEN_DUAL_BAB_STATE`) — `per_class`
    /// (the declaration default: one full build per class inside its fair
    /// share, bit-identical) or `shared` (ONE trunk of every unstable row under
    /// slopes shared across the open classes, built before the shares, plus a
    /// one-column margin tail per class). Every value is sound. Observable
    /// `[frozen-dual] trunk state=shared lambda= ...`; an armed `shared` run
    /// without it is VOID. Inert unless `frozen_dual_bab`. No shipped yaml sets
    /// this key.
    pub(crate) frozen_dual_bab_state:
        Option<ny_propagate::margin_row::zono_bab::FrozenDualStateMode>,

    /// #wide-fd-search (ENGINE_PROGRAM M4): which search SCORES the tier's
    /// nodes — `f64` (the declaration default: the tier's own search,
    /// bit-identical), `cpu_f32` (the injected ny-bbe CPU f32 SIMD level
    /// search), `cuda` (ny-cuda's device search, bit-identical to it) or
    /// `auto`. A backend only PROPOSES dual points: the tier certifies every
    /// prune in f64 with its own evaluator and splits every node no proposal
    /// certifies. Inert unless `frozen_dual_bab`. Observables `[bbe] armed=1
    /// backend= probe= lease_mb=` and `search= claimed= refuted=` on the class
    /// lines; an armed non-`f64` run without them is VOID. No shipped yaml
    /// sets this key.
    pub(crate) frozen_dual_bab_search: Option<ny_propagate::margin_row::FrozenDualSearch>,

    /// #wave-beta-bab (ENGINE_PROGRAM M6): the tree engine of the lane's
    /// OUTER frame — `legacy` (the declaration default: today's best-first
    /// tree, bit-identical) or `wave` (waves of injected UNTRUSTED output-only
    /// beta-CROWN search over one host pool of every open class; claims are
    /// certified on a private pool by the lane's own Outward domain evaluation
    /// through the frozen -> clip -> y-pack ladder, everything else is split;
    /// the verdict passes every class's path-based cover proof and the frame's
    /// authorization). A missing backend or a refused upload runs `legacy`.
    /// Observable `[bbe] phase=wave armed=1 backend= ... wave-beta-bab/v1`;
    /// an armed `wave` run without it is VOID. No shipped yaml sets this key.
    pub(crate) tree_mode: Option<ny_propagate::margin_row::TreeMode>,

    /// #wave-beta-bab: domains per wave (default 256, reader clamp
    /// `1..=16384`). Inert unless `tree_mode` is `wave` or `wave_cuda`.
    pub(crate) tree_wave_batch: Option<usize>,

    /// #wave-beta-bab: threads of the private certify pool (default 0 = half
    /// of the lane's rayon threads, reader clamp `..=1024`). Inert unless
    /// `tree_mode` is `wave` or `wave_cuda`.
    pub(crate) tree_wave_cert_threads: Option<usize>,

    /// Lane-local alpha ascent (`NY_MARGIN_ROW_ALPHA_OPT`).
    pub(crate) alpha_opt: Option<bool>,

    /// Alpha-ascent iteration cap (`NY_MARGIN_ROW_ALPHA_ITERS`).
    pub(crate) alpha_iters: Option<usize>,

    /// Alpha-ascent wall-clock request (`NY_MARGIN_ROW_ALPHA_SECS`).
    pub(crate) alpha_secs: Option<f64>,

    /// Adam step for the root alpha ascent (`NY_MARGIN_ROW_ALPHA_LR`). A value
    /// of zero selects the historical normalized-supergradient update.
    pub(crate) alpha_lr: Option<f64>,

    /// Number of worst objective columns used to steer the root alpha ascent
    /// (`NY_MARGIN_ROW_ALPHA_COLS`).
    pub(crate) alpha_cols: Option<usize>,

    /// Published-frame monotonicity guard for the root alpha ascent
    /// (`NY_MARGIN_ROW_ALPHA_GUARD`).
    pub(crate) alpha_guard: Option<bool>,

    /// #spec-alpha: per-START-NODE (spec) alpha axis at the root direct pass
    /// (`NY_MARGIN_ROW_SPEC_ALPHA`). Absent/`false` is byte-identical to the
    /// pre-#spec-alpha root.
    pub(crate) spec_alpha: Option<bool>,

    /// #spec-alpha hardest-class width (`NY_MARGIN_ROW_SPEC_ALPHA_WIDTH`).
    pub(crate) spec_alpha_width: Option<usize>,

    /// #spec-alpha candidates per class (`NY_MARGIN_ROW_SPEC_ALPHA_CANDS`).
    pub(crate) spec_alpha_cands: Option<usize>,

    /// #spec-alpha gradient rounds (`NY_MARGIN_ROW_SPEC_ALPHA_ROUNDS`).
    pub(crate) spec_alpha_rounds: Option<usize>,

    /// #spec-alpha committed-exception cap
    /// (`NY_MARGIN_ROW_SPEC_ALPHA_ENTRIES`).
    pub(crate) spec_alpha_entries: Option<usize>,

    /// Clip-and-Verify y-box tightening (`NY_MARGIN_ROW_CLIP`).
    pub(crate) clip: Option<bool>,

    /// Intermediate Clip-and-Verify tightening (`NY_MARGIN_ROW_CLIP_INTERM`).
    pub(crate) clip_interm: Option<bool>,

    /// #clip-head: HEAD-split halfspaces in the Clip-and-Verify half
    /// (`NY_MARGIN_ROW_CLIP_HEAD`).
    ///
    /// `halfspaces_for_splits` reads `LayerGates::clip_rows`, and the root gates
    /// carry TRUNK relus only, so a head sign fix builds no constraint at all: it
    /// clamps ONE interval and nothing else in the domain moves. On the traced
    /// alpha-beta-CROWN rows the head layer is where 10 of 12 splits land and
    /// where `clip_interm_domain` earns most of its per-domain improvement
    /// (13-17 other head neurons re-concretized, ~0.07-0.11 each).
    ///
    /// Unlike the trunk keys this needs NO `clip_rows` retention — the certified
    /// head lines are columns of the y-pack the eval already holds — so a
    /// category that retains nothing can still arm it. Absent = dark =
    /// byte-identical. Ships dark pending its own measurement.
    pub(crate) clip_head: Option<bool>,

    /// Retain the certified clip halfspace rows required by both clip paths
    /// (`NY_MARGIN_ROW_CLIP_ROWS`).
    pub(crate) clip_rows: Option<bool>,

    /// Per-layer Clip-and-Verify width (`NY_MARGIN_ROW_CLIP_TOPK`).
    pub(crate) clip_topk: Option<usize>,

    /// Closure fraction that triggers the adaptive ladder
    /// (`NY_MARGIN_ROW_CLOSURE_THRESH`).
    pub(crate) closure_thresh: Option<f64>,

    /// Expansion warm-up before the first closure decision
    /// (`NY_MARGIN_ROW_CLOSURE_WARMUP`).
    pub(crate) closure_warmup: Option<usize>,

    /// Expansion window between closure decisions
    /// (`NY_MARGIN_ROW_CLOSURE_WINDOW`).
    pub(crate) closure_window: Option<usize>,

    /// Highest closure-adaptive tier (`NY_MARGIN_ROW_CLOSURE_MAX_TIER`).
    pub(crate) closure_max_tier: Option<usize>,

    /// Backward-intermediate chunk width (`NY_MARGIN_ROW_BI_CHUNK`).
    pub(crate) bi_chunk: Option<usize>,

    /// Target wall seconds for one backward-intermediate prefix pass
    /// (`NY_MARGIN_ROW_BI_CHUNK_SECS`). Zero preserves fixed-width chunking.
    pub(crate) bi_chunk_secs: Option<f64>,

    /// Backward-intermediate per-layer selection width
    /// (`NY_MARGIN_ROW_BI_TOPK`).
    pub(crate) bi_topk: Option<usize>,

    /// #interm-alpha: per-START-NODE prefix alpha candidate dose
    /// (`NY_MARGIN_ROW_INTERM_ALPHA`). `0`/absent is byte-identical to the
    /// pre-#interm-alpha backward-intermediate phase.
    pub(crate) interm_alpha: Option<usize>,

    /// Split-dual beta ascent (`NY_MARGIN_ROW_BETA`).
    pub(crate) beta: Option<bool>,

    /// Legacy sign-step size used when Polyak is disabled
    /// (`NY_MARGIN_ROW_BETA_ETA`).
    pub(crate) beta_eta: Option<f64>,

    /// Beta-ascent trial cap (`NY_MARGIN_ROW_BETA_ITERS`).
    pub(crate) beta_iters: Option<usize>,

    /// Give each selected failing output column independent split-dual state
    /// (`NY_MARGIN_ROW_BETA_PERCOL`). Default off; inert unless beta is live.
    pub(crate) beta_per_column: Option<bool>,

    /// Maximum failing columns carrying independent beta state
    /// (`NY_MARGIN_ROW_BETA_COLS`, default 8, clamped by the runtime to 1..=32).
    pub(crate) beta_columns: Option<usize>,

    /// Inherit certificate-closed output columns and omit them from descendant
    /// passes (`NY_MARGIN_ROW_COL_RETIRE`). Default off.
    pub(crate) column_retire: Option<bool>,

    /// Prebuild deterministic y-packs for executable multi-domain frontier
    /// batches (`NY_MARGIN_ROW_PARALLEL_PREBUILD`). Default off.
    pub(crate) parallel_prebuild: Option<bool>,

    /// Polyak gap fraction (`NY_MARGIN_ROW_BETA_LAMBDA`).
    pub(crate) beta_lambda: Option<f64>,

    /// Opt-out selector for Polyak sizing (`NY_MARGIN_ROW_BETA_POLYAK`).
    pub(crate) beta_polyak: Option<bool>,

    /// Opt-out selector for head-split beta terms
    /// (`NY_MARGIN_ROW_BETA_HEADS`).
    pub(crate) beta_heads: Option<bool>,

    /// Joint forward re-bake (`NY_MARGIN_ROW_REBAKE`).
    pub(crate) rebake: Option<bool>,

    /// Re-bake round cap (`NY_MARGIN_ROW_REBAKE_ROUNDS`).
    pub(crate) rebake_rounds: Option<usize>,

    /// Re-bake wall-clock request (`NY_MARGIN_ROW_REBAKE_SECS`).
    pub(crate) rebake_secs: Option<f64>,

    /// #bi-window: run the backward-intermediate prefix passes (and the
    /// joint refresh's) on receptive-field CROPS of the prefix net
    /// (`NY_MARGIN_ROW_BI_WINDOW`). Absent/`false` is byte-identical. Each
    /// cropped pair is a certified pass of the unchanged engine on a net that
    /// computes exactly the seeded pre-activations, never looser than the dense
    /// pair; what it changes is how many boxes the phase can afford.
    pub(crate) bi_window: Option<bool>,

    /// #joint-refresh: rounds of the backward-only joint intermediate refresh
    /// (`NY_MARGIN_ROW_JOINT_REFRESH`, clamped to 16). Absent/`0` is dark and
    /// byte-identical. Per round: re-derive every trunk box from the current
    /// gates shrink-only, re-steer alpha on the refreshed head frame, accept on
    /// a strict certified root-bound improvement. Requires `backward_interm`.
    pub(crate) joint_refresh: Option<usize>,

    /// #joint-refresh wall request in seconds
    /// (`NY_MARGIN_ROW_JOINT_REFRESH_SECS`, default 30, capped to 40% of the
    /// remaining deadline).
    pub(crate) joint_refresh_secs: Option<f64>,

    /// Source-owned bounded head-root refinement (`NY_MARGIN_ROW_HEAD_GROUPS`).
    /// Absence inherits the governed true default; explicit `false` disables
    /// the whole route. The fixed five-second slice remains inside the original
    /// deadline. Scored runs freeze this choice in their receipt-owning LeverSet.
    pub(crate) head_groups: Option<bool>,

    /// Combine Source-certified directional cuts in one root relaxation
    /// (`NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS`, default true). Explicit `false`
    /// disables only this stage; `head_groups: false` disables the parent route.
    pub(crate) head_directional_cuts: Option<bool>,

    /// Optional complete Source head-phase cover. `0` or absence is off;
    /// `1..=7` bounds the selected heads (at most 128 closed leaves).
    /// Requires `head_groups: true` and shares its original deadline.
    #[serde(default, deserialize_with = "head_phase_cover_count")]
    pub(crate) head_phase_cover: Option<usize>,

    /// #root-certify-once: Phase-0.5 root alpha, SEARCH UNTRUSTED / CERTIFY
    /// ONCE. Absent/`false` is dark and byte-identical. Armed, the root alpha
    /// ascent runs uncertified on a Parity scratch root and ONE certified
    /// `root_eval` of the chosen alpha is published only if it strictly beats
    /// the certified incumbent (today's root).
    pub(crate) root_certify_once: Option<bool>,

    /// #root-certify-once untrusted spec-phase iterations (default 20,
    /// clamped to 1..=128).
    pub(crate) root_certify_once_iters: Option<usize>,

    /// #root-certify-once untrusted joint-intermediate rounds before the spec
    /// phase (default 0, clamped to 16). Requires `backward_interm`.
    pub(crate) root_certify_once_refresh: Option<usize>,

    /// #root-certify-once wall request in seconds (default 20, capped to 40%
    /// of the remaining deadline).
    pub(crate) root_certify_once_secs: Option<f64>,

    /// #bbe-joint-root (ENGINE_PROGRAM M5a): which UNTRUSTED search proposes
    /// the #root-certify-once slopes — `ascent` (the declaration default:
    /// today's Parity-scratch ascent, bit-identical) or `joint_f32` (the
    /// injected ny-bbe joint-alpha root search: CROWN init, spec prune,
    /// per-start-node alpha with in-loop refresh, early exits). Either way the
    /// slopes reach the certified root only through the existing ingest gate and
    /// ONE certified evaluation, kept only if strictly better. Inert unless
    /// `root_certify_once` is armed. Observable `[bbe] phase=joint_root iters=
    /// search_s= certify_s= root_lb=`; an armed run without it is VOID. Scored
    /// runs consume the key from the frozen LeverSet through its `space.rs`
    /// axis; no shipped yaml sets it.
    pub(crate) root_search: Option<ny_propagate::margin_row::RootSearch>,

    /// #root-box-source (ENGINE_PROGRAM M5b): which certified construction
    /// supplies the root's trunk boxes — `tableau` (the declaration default:
    /// the forward DeepPoly tableau, bit-identical) or `patch_backward` (a
    /// certified IBP reference per tensor, then a certified #bi-patch
    /// receptive-field backward pass, dense fallback, over the
    /// reference-unstable neurons only, intersected shrink-only; CPU only).
    /// Rides the root-precision policy, so the initial root, every #rebake round
    /// and every Tier-2 epoch rebuild use the same source. The source cannot
    /// produce clip rows, Tier-0 retained rows or an f32 tableau walk, so
    /// `patch_backward` together with `clip_rows`, a retention-requesting epoch
    /// key, `root_f32` or `tableau_f32` is REFUSED at validation (and at runtime
    /// on any other route, `[root-box] refused reason=`, tableau bit for bit)
    /// rather than silently switching those off; #backward-interm /
    /// #mr-ibp-seed are subsumed. Observable `[root-box]
    /// source=patch_backward ... certify_s=` (`source=tableau declined=` on a
    /// decline); an armed run without `source=patch_backward` is VOID. Scored
    /// runs consume the key from the frozen LeverSet through its `space.rs`
    /// axis; no shipped yaml sets it.
    pub(crate) root_box_source: Option<ny_propagate::margin_row::RootBoxSource>,

    /// RETIRED 2026-09-29 (ENGINE_PROGRAM R2, the device never certifies):
    /// kept in the deserializer only so an old yaml is REFUSED with the reason
    /// at validation (any value); an environment request is refused at runtime
    /// (`[bi-gpu] refused reason=r2_device_never_certifies`). What it was:
    ///
    /// #bi-gpu: run the backward-intermediate prefix pass pairs (baseline,
    /// `#bi-window` crops, `#interm-alpha` candidates) and the joint refresh's
    /// on the certified sound-f64 GPU conv route of `#mr-gpu-eval`
    /// (`NY_MARGIN_ROW_BI_GPU`; independent of `gpu_conv`). Absent/`false` is
    /// byte-identical. The lane's unchanged order-independent charges certify
    /// the device sum; every refused product runs the unchanged CPU kernel and
    /// each pair is RSS-admitted. Receipt `[bi-gpu] done armed=1 pairs= gpu=`;
    /// armed with `gpu=0` is VOID. Requires `backward_interm`. No shipped yaml
    /// sets this key.
    pub(crate) bi_gpu: Option<bool>,

    /// #bi-patch: run the backward-intermediate prefix pass pairs (baseline
    /// and `#interm-alpha` candidates) and the joint refresh's as PER-COLUMN
    /// RECEPTIVE-FIELD walks (`NY_MARGIN_ROW_BI_PATCH`): each trunk-seeded
    /// column walks only its own receptive field in virtual per-column frames,
    /// with the dense arm's certified arithmetic restricted to that support, so
    /// each pair is never looser than the dense pair. Precedence over
    /// `bi_window`; anything it cannot plan falls back to the crop or the dense
    /// pair. Absent/`false` is byte-identical. Receipt `[bi-patch] armed`, a
    /// per-layer `[bi-patch] plan` line, then `[bi-patch] done: pairs=`; armed
    /// with `pairs=0` is VOID. `#auto-alloc` prices the phase at the patch MACs
    /// when this key is armed. Requires `backward_interm`. No shipped yaml sets
    /// this key.
    pub(crate) bi_patch: Option<bool>,

    /// #bi-patch crossover ratio (`NY_MARGIN_ROW_BI_PATCH_RATIO`, default 0.5,
    /// closed range [0.05, 1.0]): walking backward, the first tensor whose
    /// per-column frame exceeds `ratio * n` rows runs as the dense tail.
    /// Requires `bi_patch`.
    pub(crate) bi_patch_ratio: Option<f64>,

    /// #bi-patch per-pass working set in MiB (`NY_MARGIN_ROW_BI_PATCH_MB`,
    /// default 256) that caps a patch chunk's width. Requires `bi_patch`.
    pub(crate) bi_patch_mb: Option<usize>,

    /// #mr-ibp-seed: intersect every trunk box of the root tableau walk
    /// SHRINK-ONLY with its certified IBP box (`NY_MARGIN_ROW_IBP_SEED`), after
    /// the forward-only `bi_cal` capture and before `backward_interm`.
    /// Absent/`false` is dark and byte-identical. Residual trunks decline
    /// (`declined=residual`) and stay byte-identical when armed. Receipt
    /// `[mr-ibp-seed] layer=<k> ... tightened= ... declined=` per trunk relu;
    /// an armed run without it is VOID. No shipped yaml sets this key.
    pub(crate) ibp_seed: Option<bool>,

    /// #closure-adaptive: let the lane pick its OWN closure mechanisms at
    /// runtime from an observable it computes about itself.
    ///
    /// WHY ONE POLICY KEY IN ADDITION TO THE INDIVIDUAL KEYS. Every closure
    /// mechanism now has typed delivery for attributable A/B search. The
    /// composite key serves a different purpose: it lets one row-local policy
    /// choose among β, clip, retained rows, re-bake and per-domain α from an
    /// observable, instead of requiring a human to pin six category-wide
    /// treatments. No shipped preset arms either route merely because it is
    /// deliverable.
    ///
    /// THE OBSERVABLE. Measured 2026-08-22 on the 16 `cifar100_2024` rows this
    /// lane fails and alpha-beta-CROWN proves: the ONLY variable separating
    /// abc's wins from its losses, with ZERO overlap, is the per-batch
    /// child-closure fraction — wins 0.50-1.00, losses 0.18-0.34. Node count
    /// does not predict it (abc loses at 33 618 nodes, wins at 16) and neither
    /// does the root bound (wins at post-alpha -0.765, loses at -0.423). ny
    /// closes ZERO domains on 12 of the 16. That fraction is free to compute in
    /// a best-first loop, so the lane can detect that its children are not
    /// closing and escalate, re-measuring after each step.
    ///
    /// WHICH MECHANISMS, and this is measured rather than guessed. A 24-run
    /// matrix on one pinned binary, engagement telemetry on every arm: β moves
    /// closure 3-8x and is the ONLY thing that does; β with 8 ascent trials is
    /// the largest single movement in the matrix; Clip-and-Verify is
    /// BIT-IDENTICAL to baseline with the mechanism provably armed
    /// (`mean_rel_shrink` 8.4e-16 to 1.6e-4 — one split is one halfspace in 3072
    /// dimensions, and it cuts the split neuron 63% and everything
    /// near-orthogonal to it by nothing). So the ladder is β, then deeper β,
    /// then the Tier-2 epoch rebuild — dual/objective coupling, not box
    /// tightening — and the clip rungs were cut after measurement rather than
    /// shipped as decoration.
    ///
    /// SOUND BY CONSTRUCTION at every tier: both β rungs are valid for any
    /// `β >= 0` by weak duality and keep only ACCEPTED improvements scored by
    /// the certified pass; the epoch's gates are valid exactly on the
    /// split-halfspace intersection and the parent intervals it inherits are
    /// shrink-only. The policy writes one integer and never a bound. A row whose
    /// closure is healthy escalates nothing and runs bit-identically — measured
    /// on idx_5308 (closure 0.957) and idx_7554 (0.557-0.775), both held at
    /// base by the 0.40 threshold.
    ///
    /// Default `None`/`false`: ships dark pending the measurement.
    pub(crate) closure_adaptive: Option<bool>,

    /// #branch-reduceop: reduce a candidate's two child bounds by `max`
    /// (BaBSR s_SR, De Palma eq. 22) instead of `min` (FSB balanced, eq. 23).
    /// alpha-beta-CROWN's TUNED cifar100 and tinyimagenet configs both set
    /// `reduceop: max`, overriding its own global `min` default on exactly the
    /// two benchmarks ny fails. Ranker-only: it picks WHICH candidate to split,
    /// every child bound still comes from the same certified pass. Absent =
    /// `min` = byte-identical to today.
    pub(crate) branch_reduce_max: Option<bool>,

    /// #branch-parent-lift: fold `max_i min(l_i, r_i)` over the scored
    /// branching candidates into the child bound floor. Each candidate's two
    /// children COVER the parent, so its worse child's certified bound already
    /// lower-bounds the parent; ny scores ~32 candidates and discards all but
    /// the one it splits on. Absent = discard = byte-identical to today.
    pub(crate) parent_lift: Option<bool>,

    /// #backward-interm: recompute each trunk ReLU's input bounds via the
    /// lane's OWN backward engine through the already-frozen prefix gates,
    /// shrink-only intersected with the forward tableau BEFORE gate
    /// derivation. MEASURED 2026-08-19 (cycles 1-3): cifar deep-band roots
    /// move ~0.18-0.20 (idx_5242 -0.613 -> -0.438, INTO the proven band);
    /// tiny idx_4330 timeout -> UNSAT 3/3; and proving rows get 5-10x
    /// CHEAPER (idx_6659 211 exp/72s -> 19 exp/3.8s; idx_8600 186/45s ->
    /// 35/12s). Intersecting two valid enclosures and skipping/crossing
    /// fail-open are tightness-only; the clip-calibration ordering invariant is
    /// still soundness-critical, and parity mode refuses the phase.
    pub(crate) backward_interm: Option<bool>,

    /// #margin-row-domain-alpha: re-optimize the lane's lower-line slope PER
    /// BaB DOMAIN — against that domain's own gates, Clip-and-Verify tightening
    /// included — instead of freezing the root value for the whole tree, with
    /// each child warm-starting from its parent's accepted alpha.
    ///
    /// WHY IT IS THE REMAINING CANDIDATE. The frontier-worst bound at a given
    /// DEPTH is bit-identical across every arm tried (base, β, β×8, stateless
    /// clip, stateful clip, both branching reductions), improving ~0.0096-0.0120
    /// per level against a gap of 0.155-0.203 — 13-21 more levels, i.e.
    /// 2^13..2^21 domains where the budget affords ~90. Per-domain alpha is the
    /// only mechanism whose deficit GROWS WITH DEPTH by construction (at depth
    /// 30 every unsplit unstable neuron still carries a root-box slope), and it
    /// is the missing CONSUMER for clip: abc ships the two as a compounding
    /// pair, which is why a measured 1.5-2.7e-2 stateful-clip shrink moved the
    /// gating curve by exactly zero. Measure as a 2x2, never one at a time.
    ///
    /// Tightness-only: any `alpha in [0, 1]` is a valid lower line pointwise on
    /// all of R, `ms = max(alpha, s)` is re-derived at the single write site,
    /// and the certified pass is the only scorer. Default `None`/`false`: ships
    /// dark pending the measurement.
    pub(crate) domain_alpha: Option<bool>,

    /// Per-domain alpha ascent trials per expanded domain (typed form of
    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS`, default 2, clamped `1..=8`; env wins
    /// in both directions). EACH TRIAL IS ONE MORE CERTIFIED PASS on every
    /// expanded domain with a movable neuron — the whole cost model. Inert
    /// unless `domain_alpha` is armed.
    pub(crate) domain_alpha_iters: Option<u64>,

    /// Initial normalized step for the per-domain alpha ascent (typed form of
    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_ETA`, default 0.25, clamped to
    /// `[0.01, 1.0]`; env wins in both directions). Literally how far the
    /// largest-gradient neuron moves through the legal `[0, 1]` slope range in
    /// one trial. Inert unless `domain_alpha` is armed.
    pub(crate) domain_alpha_eta: Option<f64>,

    /// SCHEDULE: skip the per-domain alpha ascent at domains shallower than
    /// this split depth (typed form of `NY_MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH`,
    /// default 0 = every depth pays, capped at 64; env wins in both
    /// directions). The ascent costs `domain_alpha_iters` certified passes
    /// uniformly in depth while the measured gain GROWS with depth, so a floor
    /// buys the deep half of the curve without paying the shallow half — at the
    /// risk of declining exactly the shallow alphas every descendant inherits.
    /// Inert unless `domain_alpha` is armed.
    pub(crate) domain_alpha_min_depth: Option<u64>,

    /// SCHEDULE: pay the per-domain alpha ascent only where
    /// `depth % stride == 0` (typed form of
    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_STRIDE`, default 1 = every depth pays,
    /// clamped `1..=16`; env wins in both directions). The WARM START is free,
    /// so an unpaid level still carries its ancestors' accepted alpha; this is
    /// the arm that says whether the gain is carried by inheritance (cost/k,
    /// gain kept) or by each node's own trials (both divided). Inert unless
    /// `domain_alpha` is armed.
    pub(crate) domain_alpha_stride: Option<u64>,

    /// SCHEDULE: pay the per-domain alpha ascent only at domains within this
    /// much of closure, i.e. where the pre-ascent bound has `b >= -gap` (typed
    /// form of `NY_MARGIN_ROW_DOMAIN_ALPHA_GAP`, default 0 = UNLIMITED; env
    /// wins in both directions). cifar100 fails by a counting problem, so the
    /// passes are worth most where they can retire a domain outright rather
    /// than where they buy +0.002 against a -0.30 gap. Inert unless
    /// `domain_alpha` is armed.
    pub(crate) domain_alpha_gap: Option<f64>,
    /// #clip-stateful: make that tightening ACCUMULATE per domain and be
    /// inherited parent -> child, with the per-layer top-k rescored against the
    /// current intervals instead of the root ones.
    ///
    /// This is the alpha-beta-CROWN 2024 -> 2025 cifar100 diff: it added
    /// `clip_interm_domain` and REMOVED `interm_transfer: False`, and the second
    /// line is what makes the first one compound. ny's stateless port measured a
    /// FLAT dose-response (2.1x more tightening, identical bound, identical
    /// 429-domain frontier) because it recomputed from the root frame every call
    /// and threw the result away. Inert unless `clip_interm` is also armed.
    /// Absent = dark = byte-identical.
    pub(crate) clip_stateful: Option<bool>,

    /// Seconds cap for the backward-intermediate phase (typed form of
    /// `NY_MARGIN_ROW_BI_SECS`; env wins in both directions).
    ///
    /// MEASURED 2026-08-22 over 64 banked-timeout resnet_large rows: at the
    /// shipped 20 s default the phase is TRUNCATED (`exhausted=true`) on 10/10
    /// resnet_LARGE rows, which need 24-35 s, and on 0/9 resnet_MEDIUM rows,
    /// which finish in 6.6-8.4 s. Raising it is therefore free on medium (a
    /// byte-identical no-op) and strictly better on large: the tighter root
    /// yields a SMALLER tree (idx_5410 39 -> 28 expansions, idx_496 2 -> 1) at
    /// no wall cost. Converts idx_9883 (unknown 3/3 -> unsat 3/3); moat 7/7
    /// preserved; no row converts at 20 and fails at 40. The effective value
    /// is `min(secs, 0.4 * remaining)`, so 40 resolves to ~37.2 s at a scored
    /// 100 s budget — and 63/64 rows then finish inside it, so there is no
    /// headroom left in this lever.
    pub(crate) bi_secs: Option<u64>,

    /// #epoch-bab Tier-0 ranker shortlist width. Typed form of
    /// `NY_EPOCH_TIER0`; `0` keeps the established shortlist path, while a
    /// positive value sends that many rank-1-ranked candidates to the exact
    /// certified pass. A present legacy environment value still wins, and a
    /// malformed present value suppresses this key and restores the historical
    /// default `0`.
    ///
    /// No shipped preset sets this key: promotion makes the measured ranker
    /// reachable from a clean scorecard without arming it.
    pub(crate) epoch_tier0: Option<usize>,

    /// Number of trunk candidates scored by the Tier-0 ranker before its exact
    /// shortlist is selected. Typed form of `NY_EPOCH_UNIVERSE`, with the
    /// historical default `64` and the same untrimmed environment precedence.
    /// Inert unless `epoch_tier0` is positive.
    pub(crate) epoch_universe: Option<usize>,

    /// Retained f32 pre-activation rows per trunk ReLU layer for the Tier-0
    /// ranker. Typed form of `NY_EPOCH_RETAIN`, with historical default `128`;
    /// the separate 256 MiB global retention cap is unchanged. The runtime
    /// only consults this value when root-build retention is requested.
    pub(crate) epoch_retain: Option<usize>,

    /// #epoch-parent: number of NEW trunk splits between Tier-2 epoch
    /// refreshes. `None` and `0` keep epochs disabled. Typed form of the
    /// legacy `NY_EPOCH_DEPTH`; a present environment value still wins.
    ///
    /// This key only makes the mechanism reachable by canonical scored
    /// measurement. No shipped preset sets it pending a sealed A/B.
    pub(crate) epoch_depth: Option<usize>,

    /// Maximum Unknown-ending Tier-2 epoch attempts per BaB driver. Typed form
    /// of `NY_EPOCH_ATTEMPTS` (built-in default 2); a present environment value
    /// still wins. `0` is an explicit kill switch.
    ///
    /// No shipped preset sets this key.
    pub(crate) epoch_attempts: Option<usize>,

    /// #epoch-parent: maximum number of widest downstream preactivation
    /// columns refreshed per ReLU layer. `None`/`0` preserves the established
    /// full-tableau epoch path. Typed form of
    /// `NY_MARGIN_ROW_EPOCH_PARENT_TOPK`; a present governed environment value
    /// wins, including `0` as a kill switch.
    ///
    /// This changes an authoritative bound path when paired with a positive
    /// `epoch_depth`, so it ships dark until sealed soundness and wall-time
    /// measurements justify a preset value.
    pub(crate) epoch_parent_topk: Option<usize>,

    /// #twinwall-early-exit: publish RESULTS_FILE and end the instance the
    /// moment the CONCURRENT lane certifies `unsat`, instead of letting the
    /// internal verifier run its remaining budget out on an already-proved row.
    ///
    /// DEFAULT ON (`None` == `true`); this key exists as the typed kill switch,
    /// alongside `NY_MARGIN_ROW_EARLY_EXIT`, which wins where present.
    ///
    /// Verdict-neutral by construction: the lane is fail-closed (`Unsat` or
    /// `Unknown`, never `Sat`), both existing consumers already `return` its
    /// verdict verbatim with no refinement, and the attack lanes it skips can
    /// only ADD `sat` on rows the lane did NOT prove. MEASURED waste it
    /// removes on cifar100_2024: `resnet_medium prop_idx_7704_sidx_3701`
    /// proved at 10.4 s of a 100 s budget and published at 89.7 s.
    pub(crate) early_exit: Option<bool>,

    /// #verifier-thread-cap: request this width for Rayon's process-global pool
    /// before the INTERNAL alpha/beta verifier or any optional Rayon work starts.
    ///
    /// `NY_MARGIN_ROW_CONCURRENT_THREADS` already bounds the lane; nothing
    /// bounded the verifier, so on a 20-core box the lane that produces every
    /// UNSAT on this category competes with the lane that produces none.
    ///
    /// Absent or `0` preserves today's behavior: this policy does not initialize
    /// or inspect the global pool, and the verifier remains on the main thread.
    /// A positive value makes one main-thread `build_global` attempt. An already
    /// initialized pool is recorded with its observed actual width and voids the
    /// cap A/B; any other build failure stops scored work with a sound `unknown`.
    /// The verifier is never moved into a private pool because competition-result
    /// capture is main-thread-local. `NY_VERIFIER_THREADS` wins where present,
    /// including explicit zero.
    ///
    /// Scheduling only — a thread count cannot change a bound or a verdict, it
    /// can only change what finishes inside the budget.
    pub(crate) verifier_threads: Option<usize>,

    /// #scheduler-resource-governance: optional Rayon width for the CONCURRENT
    /// margin-row lane (typed form of `NY_MARGIN_ROW_CONCURRENT_THREADS`).
    ///
    /// `None` and `0` are intentionally different. Without a positive verifier
    /// request, `None` uses the shared global pool. With one, `None` derives a
    /// private lane width only as `host_parallelism - actual_global_threads`;
    /// missing observations or no positive remainder refuse the lane. It never
    /// derives from the requested verifier cap. Explicit `0` suppresses the
    /// derivation and requests the shared global pool. A positive value requests
    /// a private pool of exactly that width; a build failure refuses the lane,
    /// with no global-pool fallback. The environment override remains untrimmed
    /// and wins only when it parses as a platform `usize`, including zero.
    ///
    /// Scheduling only. No shipped preset sets this unmeasured resource seam.
    pub(crate) concurrent_threads: Option<usize>,
}

/// Model-loading configuration (alpha-beta-CROWN compatibility).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelPreset {
    /// ONNX conversion-time optimization flags.
    ///
    /// Accepts either a single string (`merge_linear`) or a YAML sequence.
    #[serde(default, deserialize_with = "string_or_vec_string")]
    pub(crate) onnx_optimization_flags: Vec<String>,

    /// BatchNormalization conversion policy.
    ///
    /// `preserve_raw` is required by proof lanes that claim authority over the
    /// authored ONNX graph. Unset preserves the historical loader policy.
    pub(crate) batch_norm_folding: Option<ny_onnx::BatchNormFoldingPolicy>,

    /// Fail loading unless every raw ONNX FLOAT initializer remains unchanged.
    ///
    /// This enables loader-sealed provenance and is required by the
    /// forward-linear spec-alpha admission check.
    pub(crate) require_authored_float32_initializers: Option<bool>,

    /// Default-off forward-linear spec-alpha surrogate.
    ///
    /// The surrogate only chooses ReLU lower slopes; a certified rebuild owns
    /// every bound. It is nevertheless admitted only with `preserve_raw` and
    /// authored FLOAT initializer provenance, so a tighter UNSAT result cannot
    /// silently certify a loader-rewritten graph. The old cGAN-named key is an
    /// input-only compatibility alias for existing sealed presets.
    #[serde(alias = "cgan_forward_alpha_surrogate")]
    pub(crate) forward_linear_spec_alpha: Option<bool>,

    /// #fl-seq-no-transpose: admit the SEQUENTIAL forward-linear intermediate
    /// reference on a Conv2d graph that carries NO ConvTranspose2d.
    ///
    /// `should_collect_forward_linear_intermediate_reference` admits sequential
    /// graphs only when they hold BOTH Conv2d and ConvTranspose2d — a clause
    /// written to target the cGAN image surface. Its own doc gives the soundness
    /// position: "Tightening a reference bound is sound-by-construction here:
    /// the value is an enclosure either way, and a narrower enclosure can only
    /// remove spurious relaxations." So the ConvTranspose requirement is
    /// SCOPING (it avoids shapes where the collector may fail closed), not
    /// correctness.
    ///
    /// MEASURED 2026-09-14: `cGAN_imgSz32_nCh_3_upsample` builds with
    /// Upsample+Conv and has no ConvTranspose, so it can never satisfy the
    /// clause. At its official 1200 s budget it logs `[cgan-root] DECLINED:
    /// ... forward_linear_reference=false`, the entire typed cGAN root cascade
    /// (dense collector, unstable rows, crown root map) is inert, and its
    /// timeout is therefore VOID rather than a capability result. The row is
    /// worth 4.75 normalized points.
    ///
    /// DELIVERY: the scored entry point exports no `NY_*`, so this is a typed
    /// key rather than an env lever. Absent (the shipped state on every yaml) is
    /// byte-identical to the historical admission. Requires GraphNetwork
    /// routing; a Sequential model with it set is refused rather than silently
    /// ignored.
    pub(crate) forward_linear_sequential_without_conv_transpose: Option<bool>,

    /// `#fl-s-shaped-untyped`: admit Tanh/Sigmoid on the ORDINARY (untyped)
    /// forward-linear surface, so `GraphNetwork spec-CROWN` stops refusing
    /// fail-closed on graphs carrying them.
    ///
    /// Measured motivation: on cgan's two open rows the typed ROOT cascade now
    /// completes (`mode=CganComplete`) while spec-CROWN still logs
    /// "forward-linear intermediates unavailable ... falling back (fail-closed)"
    /// five times per run. 4.75 normalized points per row.
    ///
    /// ⚠ The SIGN IS UNMEASURED. The path this replaces is per-node CROWN-IBP,
    /// not plain IBP, and neither enclosure dominates — arming it can loosen
    /// spec-CROWN's intermediates. It also risks EVICTING the root forward-linear
    /// map, whose cache is a single box-keyed slot that per-domain builds would
    /// now overwrite instead of failing at the allowlist. Default absent = the
    /// shipped admission, byte-identical. Ship only on measured evidence.
    pub(crate) forward_linear_s_shaped_untyped: Option<bool>,

    /// `#fl-f32-seam-key`: route the forward-linear composition's big *value*
    /// GEMMs (`A.W`, `|A|.|W|`) through the sound f32 fast path.
    ///
    /// Promotes what was an ENV-ONLY lever (`NY_FORWARD_LINEAR_F32`) to a
    /// deliverable key. `vnncomp_scripts/run_instance.sh` exports ZERO `NY_*`,
    /// so the seam could never fire in competition however it was measured —
    /// dead capability with a **1.9x** measured end-to-end speedup, and ~40x on
    /// the value GEMMs themselves (cuBLAS Sgemm vs the GB10's 0.41 TF/s Dgemm,
    /// which f64 is stuck on because wgpu has no f64).
    ///
    /// SOUND, and LOOSER. The larger f32 accumulation error is bounded by the
    /// Higham factor `gamma_{K+4}^f32 * S` plus an FTZ underflow guard, and is
    /// discharged into the bias through the SAME certified-error channel the
    /// f64 path already uses; the S-base GEMMs stay f64 so `S` is never
    /// under-estimated. But looser intermediates can LOSE rows:
    /// `docs/CIFAR100_BOUND_PARITY_ORACLE.md` measured **0 gains, 1 regression**
    /// on cifar100 and `docs/WHY_BETA_CROWN_FORFEITS_205S_2026-08-05.md:415`
    /// records a standing "do not arm".
    ///
    /// So this key exists to make the axis REACHABLE and per-preset; it is not
    /// an endorsement. Absent = the shipped f64 path, byte-identical.
    pub(crate) forward_linear_f32_seam: Option<bool>,

    /// Default-off graph-generic forward-map alpha surrogate
    /// (#fl-alpha-composition, consult #8 Days 1-3).
    ///
    /// Arms the SAME typed graph lever as `cgan_forward_alpha_surrogate`
    /// (`#w4-root-alpha-opt`: the optimizer only proposes ReLU lower slopes;
    /// ONE certified alpha-fed forward-linear rebuild owns every bound, and
    /// spec propagation intersects it element-wise with the fixed-slope FL
    /// candidate, so the fixed FL bound is a monotone floor — never weaker).
    ///
    /// UNLIKE the cGAN key this one claims authority over the LOADED graph —
    /// the same graph every other root candidate (CROWN backward, fixed FL
    /// C-margin, GPU resnet root) on these conv-DAG categories already
    /// certifies — so it deliberately does NOT require
    /// `batch_norm_folding: preserve_raw`. Requiring it would CHANGE the
    /// loaded graph on cifar100 (19 raw BatchNorm nodes kept instead of
    /// folded) and invalidate the measured FL cost/margin chain
    /// (docs/FL_FIRST_MEASUREMENT_2026-08-02.md) that licenses arming it.
    pub(crate) forward_alpha_surrogate: Option<bool>,

    /// Opt into the default-off alpha-beta-CROWN VGG treatment:
    /// exact 2x2 MaxPool decomposition plus property-size policy.
    ///
    /// This is intentionally model/preset scoped rather than a global solver
    /// default. Ineligible MaxPool nodes remain unchanged.
    pub(crate) vgg_abcrown_treatment: Option<bool>,

    /// `model.qdq_activation_relaxation` — DARK typed gate for the certified
    /// QDQ mechanism pair (#smart-turn-qdq, report
    /// `smart_turn_bucket_bisect_2026-08-31`). Default OFF preserves both of
    /// today's refusals byte-identical:
    ///
    /// 1. LOADER: a `Cast(to=FLOAT)` on a PROVEN-STATIC INT64 structural value
    ///    (transformer attention's `Shape -> Slice -> Cast -> Sqrt` head-dim
    ///    scaling) is admitted as a deferred-fold obligation — the original
    ///    refusal re-raises unless constant folding erases the Cast into a
    ///    literal, so nothing integer ever reaches the f32 runtime graph.
    /// 2. CONVERTER: each activation `QuantizeLinear -> DequantizeLinear` pair
    ///    with identical parameters is fused into the certified
    ///    `QdqPerturbationLayer` enclosure (dequant∘quant(x) within
    ///    `clamp(x ± (s/2 + rounding terms), saturation range) ± G`, every
    ///    step outward-rounded; containment suite in ny-propagate).
    ///
    /// REPORT-ONLY THIS ROUND: arming it additionally suppresses any `unsat`
    /// this lane would publish on a graph that contains a QDQ relaxation
    /// layer — the bound is printed as a finding; the verdict stays dark until
    /// the mechanism survives review and the family adjudication policy is
    /// settled. No shipped preset sets this key.
    pub(crate) qdq_activation_relaxation: Option<bool>,

    /// `model.multi_input_concat` — DARK typed gate for multi-input graph
    /// propagation (#multi-input-propagation, follow-up to the QDQ round's
    /// named next mechanism). Default OFF preserves today's fail-closed
    /// refusal byte-identical: a model with more than one declared graph
    /// input aliases every input to the single `_input` node, the input box
    /// is sized from the FIRST input only, and the VNN-LIB coordinate-count
    /// validation rejects the spec before any bound is computed.
    ///
    /// ON, the graph builder concatenates every declared input, in DECLARED
    /// ORDER (the same order VNN-LIB `X_i` variables enumerate multi-input
    /// models), into one flat network input, carves each input back out with
    /// exact Slice(+Reshape) view nodes, and the CLI sizes the joint input
    /// box to the concatenated element count. CONTAINMENT: the joint box is
    /// exactly the product of the per-input boxes — the views are coordinate
    /// selections, no widening, and the layout is partition-verified
    /// fail-closed before any node is emitted. Single-input models take the
    /// identical legacy path in both states. No shipped preset sets this key.
    pub(crate) multi_input_concat: Option<bool>,
}

/// General configuration options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct GeneralPreset {
    /// Root path for benchmark data (relative to config file).
    pub(crate) root_path: Option<PathBuf>,

    /// CSV file with instance list.
    pub(crate) csv_name: Option<String>,

    /// Compute device (cpu, wgpu).
    pub(crate) device: Option<String>,

    /// Loss reduction function (sum, max, min).
    pub(crate) loss_reduction_func: Option<String>,

    /// Convolution backward mode: auto, patches, or matrix.
    /// Reference: alpha-beta-CROWN `general.conv_mode` (`abcrown.py:228-231`).
    pub(crate) conv_mode: Option<ny_propagate::ConvMode>,

    /// Complete verifier selection: "auto", "bab", or "mip".
    /// Reference: alpha-beta-CROWN `general.complete_verifier`. Categories whose
    /// nets are MIP-exact and CROWN-loose (sat_relu, malbeware) route straight to
    /// the MIP solver with the full budget instead of burning it in BaB first.
    /// An explicit `--complete-verifier` CLI choice still wins over the preset.
    pub(crate) complete_verifier: Option<String>,
}

/// #sbench-f64-attack: smallest admissible `attack.active_set_repair_secs`.
pub(crate) const ACTIVE_SET_REPAIR_SECS_MIN: f64 = 1.0;
/// #sbench-f64-attack: largest admissible `attack.active_set_repair_secs`.
pub(crate) const ACTIVE_SET_REPAIR_SECS_MAX: f64 = 600.0;
/// #sbench-f64-attack: largest admissible `attack.f64_target_margin_ulps`.
pub(crate) const F64_TARGET_MARGIN_ULPS_MAX: u32 = 1 << 16;
/// #sbench-f64-maximin: largest admissible `attack.f64_basin_hops`.
pub(crate) const F64_BASIN_HOPS_MAX: usize = 100_000;

/// PGD attack configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct AttackPreset {
    /// Reference alpha-beta-CROWN PGD order.
    ///
    /// NY implements `before`, `after`, the disabled `skip` family, and
    /// `input_bab`. Reference `middle` placement still requires the explicit
    /// [`Self::ny_pgd_order_compat`] contract instead of silently masquerading
    /// as scheduler parity.
    pub(crate) pgd_order: Option<String>,
    /// NY-only compatibility contract for a reference `pgd_order` of `middle`
    /// or `after`.
    ///
    /// `upfront` explicitly preserves NY's measured historical *initial* PGD
    /// placement. It does not control either the engine's post-BaB fraction or
    /// the independent VNN-COMP post-BaB wrapper attack. Omission fails closed
    /// for `middle`/`after`; the field is rejected for every other order.
    pub(crate) ny_pgd_order_compat: Option<NyPgdOrderCompat>,
    /// Number of PGD restarts.
    pub(crate) pgd_restarts: Option<usize>,
    /// Number of PGD steps per restart.
    pub(crate) pgd_steps: Option<usize>,
    /// Prefer the exact-gradient conjunctive route over generic batched SPSA.
    pub(crate) pgd_exact_batched: Option<bool>,
    /// Attempt the point-Jacobian fast gradient in sequential graph PGD.
    pub(crate) pgd_fast_grad: Option<bool>,
    /// Admit the batched exact point-VJP graph attack drivers.
    pub(crate) pgd_vjp_batch: Option<bool>,
    /// Shared exact-VJP restart width (route-specific default when absent).
    pub(crate) pgd_vjp_k: Option<usize>,
    /// Uniform seed-replay lanes in the first conjunctive exact-VJP wave.
    pub(crate) pgd_legacy_lanes: Option<usize>,
    /// Output-diversified initialization steps per conjunctive wave.
    pub(crate) pgd_odi_steps: Option<usize>,
    /// Positive finite f32 ODI step scale in input half-range units.
    pub(crate) pgd_odi_eta: Option<f32>,
    /// Optional positive joint-margin steps per conjunctive exact-VJP wave.
    pub(crate) pgd_vjp_steps: Option<usize>,
    /// Rate-projection lane-recycling window; zero disables.
    pub(crate) pgd_lane_window: Option<usize>,
    /// Enable conjunctive endgame exploit recycling.
    pub(crate) pgd_exploit: Option<bool>,
    /// Non-negative finite f32 odd-lane tau-hinge threshold. The disjunctive
    /// focus wave ([`Self::pgd_disj_focus_lanes`]) reuses it as its hinge.
    pub(crate) pgd_hinge_tau: Option<f32>,
    /// #disj-odi (`NY_PGD_DISJ_ODI`): run `pgd_odi_steps` ODI steps of scale
    /// `pgd_odi_eta` on the dynamic lanes of every DISJUNCTIVE exact-VJP wave.
    /// Absent/false keeps the disjunctive driver byte-identical. Dark: no
    /// shipped yaml sets it until the commit that carries its measurement.
    pub(crate) pgd_disj_odi: Option<bool>,
    /// #disj-odi (`NY_PGD_DISJ_FOCUS_LANES`): dangerous-class focus lanes the
    /// disjunctive exact-VJP driver appends after its restart waves; zero or
    /// absent disables. Dark in every shipped yaml.
    pub(crate) pgd_disj_focus_lanes: Option<usize>,
    /// PGD alpha/step size. Accepts numeric YAML values or the string `auto`.
    #[serde(default, deserialize_with = "value_parse::option_string_or_number")]
    pub(crate) pgd_alpha: Option<String>,
    /// Whether `pgd_alpha` should scale by the input range.
    pub(crate) pgd_alpha_scale: Option<bool>,
    /// Per-step exponential decay for the PGD/Adam learning rate.
    /// Maps to `BetaCrownConfig::pgd_lr_decay` (→ `AdamClippingParams::lr_decay`).
    /// Reference: alpha-beta-CROWN `attack.pgd_lr_decay`.
    pub(crate) pgd_lr_decay: Option<f32>,
    pub(crate) attack_tolerance: Option<f32>,
    /// Restart PGD when a projected step leaves the point unchanged.
    pub(crate) pgd_restart_when_stuck: Option<bool>,
    /// Force the UPFRONT falsification lane on for every instance of this
    /// category (`#upfront-preset`).
    ///
    /// The lane auto-enables only for MULTI-CLAUSE DISJUNCTIONS. safenlp's specs
    /// are single-constraint (`(assert (<= Y_0 Y_1))`), so the auto rule skips
    /// them and the lane has to be forced. Historically that happened ONLY via
    /// a wrapper-only `NY_UPFRONT_ATTACK=1` export; the scored wrapper no longer
    /// injects any `NY_*` values.
    /// That made the submission path behave differently from every other
    /// invocation: `ny benchmarks run`, `measure_ny_scorecard.sh` and a direct
    /// `ny vnncomp` all missed it. MEASURED on safenlp
    /// medical/hyperrectangle_2096 at the official 20 s budget: `sat` with the
    /// flag, `timeout` without it — a real solve that no measurement path could
    /// see. Putting it in the preset makes the behaviour travel with the
    /// category. `NY_UPFRONT_ATTACK` still overrides (0 kills, 1 forces).
    ///
    /// CONSUMER: `commands::vnncomp::upfront_wrapper_route` — `true` resolves
    /// to `UpfrontWrapperRoute::ForcedByPreset` (the environment force's
    /// arming), `false` to `DisabledByPreset`. The key was a parsed
    /// no-op until that route read it, which silently reintroduced the exact
    /// measurement/submission divergence described above for every non-wrapper
    /// entry point.
    ///
    /// SCOPE (#ml4acopf-upfront-single): the key WIDENS admission to the
    /// single-clause shapes the automatic rule skips and runs them under
    /// `upfront_attack_cap_secs`. A multi-clause disjunction the automatic rule
    /// already admits keeps its automatic admission and its tighter
    /// `upfront_attack_auto_cap_secs` clamp (unlike `NY_UPFRONT_ATTACK=1`), so
    /// arming a category for its single-clause rows leaves every multi-clause
    /// row's budget unchanged. A single-output constant threshold (ml4acopf's
    /// `(assert (>= Y_2382 0.99))`) admitted this way ascends the raw margin on
    /// every restart instead of the margin/DLR ensemble.
    pub(crate) upfront_attack: Option<bool>,
    /// Whole-second wall cap for a forced or typed-relational upfront attack.
    ///
    /// Typed delivery for `NY_UPFRONT_ATTACK_CAP`. The environment override is
    /// resolved once at scored-run entry and wins when admissible; absence
    /// retains this value or the measured eight-second declaration default.
    pub(crate) upfront_attack_cap_secs: Option<u64>,
    /// Whole-second wall cap for automatic multi-clause-disjunction admission.
    ///
    /// Typed delivery for `NY_UPFRONT_ATTACK_AUTO_CAP`. This clamp is applied
    /// to every `auto_disjunction` admission: the contextual route, and (since
    /// #ml4acopf-upfront-single) a multi-clause disjunction under the typed
    /// `attack.upfront_attack: true` force. A `forced` admission (the
    /// environment force, or a single-clause row under the typed force) and a
    /// typed relational admission use [`Self::upfront_attack_cap_secs`].
    pub(crate) upfront_attack_auto_cap_secs: Option<u64>,
    /// Fraction of live scored wall offered to the upfront lane before its cap.
    ///
    /// Typed delivery for `NY_UPFRONT_ATTACK_FRAC`. Validation is driven by the
    /// declaration and requires a finite value in `0 < fraction <= 1`.
    pub(crate) upfront_attack_fraction: Option<f64>,
    /// Fraction of live scored wall offered to the upfront lane on Sign nets.
    ///
    /// Typed delivery for `NY_BNN_ATTACK_FRAC`. The environment override wins
    /// when admissible; absence retains this value or the measured 0.5 default.
    /// Declaration-driven validation requires a finite value in
    /// `0 < fraction <= 1`.
    pub(crate) bnn_attack_fraction: Option<f64>,
    /// Whole-second wall cap for the upfront lane on Sign nets.
    ///
    /// Typed delivery for `NY_BNN_ATTACK_CAP`. The environment override wins
    /// when admissible; absence retains this value or the measured 240-second
    /// default.
    pub(crate) bnn_attack_cap_secs: Option<u64>,
    /// Diversified-restart count for the converged post-BaB attack stack
    /// (#postbab-restart-diversity).
    ///
    /// Typed delivery for `NY_POSTBAB_RESTARTS`. After the post-BaB stage-0
    /// alternation self-terminates (`no improvement; stopping`) with attack
    /// window remaining, re-enter it up to this many times from
    /// deterministically seeded diversified basins (uniform / corner /
    /// orthogonal-jitter away from the converged point), keeping the best ORT
    /// margin across restarts. Arming reserves 2 s per requested restart (at
    /// most half the window) out of the seeded descent's share of the SAME
    /// attack window — the standard window is oversubscribed, so restarts must
    /// own a slice of it to run at all; the verifier grant is untouched.
    /// The RNG seed derives from the row's model and
    /// property file names — never ambient Date/random state — so a verdict
    /// stays a pure function of the instance. Absence/zero preserves the
    /// historical single-descent stack byte-for-byte. Motivation:
    /// `reports/measured-2026/soundnessbench_window_probe_2026-08-31/` measured
    /// that stack CONVERGENT (bit-identical fixed points at 11 s and 49.6 s
    /// windows), so basin diversity, not window length, is the open axis.
    ///
    /// CONSUMER: `commands::vnncomp::PostBabRestartPolicy::from_levers` via the
    /// frozen scored-run LeverSet, then `try_postbab_falsify`'s restart block.
    pub(crate) post_bab_restarts: Option<u64>,
    /// Exact-gradient falsification on a point-folded ATTACK twin
    /// (#lsnc-attack-twin).
    ///
    /// Typed delivery for `NY_ATTACK_POINT_FOLDED_TWIN`. Armed, the post-BaB
    /// tail on a net whose real graph is STATICALLY outside the exact point-VJP
    /// fragment — because the exact loader failed a constant-only cone closed —
    /// builds `ny_onnx::attack_twin::PointFoldedAttackTwin` (constant cones
    /// evaluated once by ONNX Runtime and baked in; a distinct type with no route
    /// to any bound API) and runs projected Adam PGD with face-biased restarts on
    /// it, ORT-checking every iterate against the ORIGINAL model. Only an
    /// ORT-confirmed violation leaves the lane, and the caller re-runs the
    /// unchanged `gate_sat_with_trusted_oracle`, so the key can only ADD `sat`
    /// verdicts. Absence/false keeps the historical tail byte-for-byte.
    ///
    /// CONSUMER: `commands::vnncomp::attack_twin_lane::PointFoldedTwinPolicy`
    /// via the frozen scored-run LeverSet, then `try_postbab_falsify`.
    pub(crate) point_folded_twin: Option<bool>,
    /// `attack.active_set_repair_secs` — total wall cap, in seconds, of the
    /// post-BaB trusted-ORT active-set repair phase (#sbench-f64-attack), and
    /// of the float64 target-margin phase when
    /// [`Self::f64_target_margin_ulps`] arms it.
    ///
    /// Absent (every shipped preset) keeps the historical constants
    /// byte-for-byte: 6 s per phase, 3 s per seed lineage, 28/32 Newton
    /// iterations. Present, the per-seed cap is half the total (so `6.0`
    /// reproduces the constants exactly) and the iteration caps scale by
    /// `ceil(total / 6 s)`. Every other rule is unchanged: the first lineage's
    /// slice, the three-second downstream reserve for later lineages, and the
    /// frozen attack deadline. Measured motivation: on soundnessbench
    /// `property_054` the repair's first lineage stopped at its 28-iteration cap
    /// and the two later lineages at their 1.5 s slices
    /// (`scratchpad/refine2/sbs_16.raw`).
    ///
    /// No environment twin, so no search-space axis: validated in
    /// `preset::apply::validate_delivered_lever_values` (finite, within
    /// [`ACTIVE_SET_REPAIR_SECS_MIN`]..=[`ACTIVE_SET_REPAIR_SECS_MAX`]).
    ///
    /// SOUNDNESS: budget only. The phase returns a claimed witness at most,
    /// which the caller re-runs through the unchanged
    /// `gate_sat_with_trusted_oracle`; it cannot produce `unsat`.
    ///
    /// CONSUMER: `commands::vnncomp::f64_target_attack::PostBabActiveSetPolicy`,
    /// read from the frozen preset snapshot in `run_and_translate`, then
    /// `try_postbab_falsify`.
    pub(crate) active_set_repair_secs: Option<f64>,
    /// `attack.f64_target_margin_ulps` — arm the post-BaB float64
    /// target-margin repair (#sbench-f64-attack) with a target margin of this
    /// many f32 ULPs at the constrained outputs' magnitude.
    ///
    /// WHY. SoundnessBench's planted counterexample regions are thinner than
    /// f32 output resolution: the field's accepted witnesses have ONNX Runtime
    /// margins of 1e-7..1e-5, and ny's own ORT-surface repair found an
    /// ORT-confirmed point on `property_054` (+6.8e-8) that the true-f64 gate
    /// correctly refused (-1.3e-6). Searching the f32 surface chases rounding
    /// noise. This stage lifts every conjunct of the property to an EXACT-f64
    /// margin of `delta = ulps * ulp_f32(max |Y_j|)` (the f64 twin of the
    /// batched point-VJP plan supplies the margins and their exact gradient),
    /// rounds each coordinate toward the side that helps, and asks the trusted
    /// forward only at points whose exact margin is already non-negative. When
    /// f64 clears `delta` but the trusted forward still does not, `delta`
    /// doubles (at most four times).
    ///
    /// Absent (every shipped preset) the stage never runs, nothing is loaded,
    /// and the post-BaB tail is byte-identical. No environment twin, so no
    /// search-space axis: validated in
    /// `preset::apply::validate_delivered_lever_values`
    /// (1..=[`F64_TARGET_MARGIN_ULPS_MAX`]).
    ///
    /// SOUNDNESS: attack-only. A candidate leaves the stage only when the
    /// trusted ORT forward confirms the violation at zero tolerance AND the
    /// exact-f64 margin is non-negative; it then passes the same local true-f64
    /// precheck as every other post-BaB lineage and the caller's unchanged
    /// `gate_sat_with_trusted_oracle`. The stage cannot produce `unsat`.
    ///
    /// OBSERVABLE: `[f64-attack] policy ...` at scored-run entry and one
    /// `[f64-attack] armed=1 engaged=...` line per post-BaB tail.
    ///
    /// CONSUMER: `commands::vnncomp::f64_target_attack::PostBabActiveSetPolicy`,
    /// read from the frozen preset snapshot in `run_and_translate`, then
    /// `try_postbab_falsify`.
    pub(crate) f64_target_margin_ulps: Option<u32>,
    /// `attack.f64_lift_step` — the step rule of the post-BaB float64 lift that
    /// [`Self::f64_target_margin_ulps`] arms (#sbench-f64-maximin).
    ///
    /// `min_norm` (absent) is #sbench-f64-attack's active-set min-norm Newton
    /// row lift, byte-for-byte. `lp_maximin` replaces it with a sequential LP
    /// that MAXIMISES THE MINIMUM exact-f64 margin over every conjunct inside
    /// a box trust region (accepting a step only when the true exact-f64
    /// minimum margin improves), plus [`Self::f64_basin_hops`] basin hops;
    /// under `lp_maximin` the trusted-ORT active-set phase keeps its
    /// historical 3 s / 6 s caps and [`Self::active_set_repair_secs`] funds
    /// the f64 lift instead.
    ///
    /// WHY (MEASURED off-box, `scratchpad/vbs/frontier/rest.md` §2):
    /// SoundnessBench `property_033` is truth-SAT; both witnesses that replay
    /// as violated on the organisers' x86-64 ONNX Runtime (+4.9e-6, +3.5e-6)
    /// came from an f64 LP-maximin, the from-scratch one only after 60 basin
    /// hops, while ny's post-BaB tail stalls ~9e-6 short.
    ///
    /// Valid only together with [`Self::f64_target_margin_ulps`]: the
    /// validator REFUSES the key without it (a treatment the run would not
    /// get). No environment twin, so no search-space axis.
    ///
    /// SOUNDNESS: attack-only, and the stage's exit is unchanged — a candidate
    /// leaves only when the trusted ORT forward confirms it at zero tolerance
    /// AND its exact-f64 margin is `>= 0`, then passes the local true-f64
    /// precheck and the caller's unchanged `gate_sat_with_trusted_oracle`.
    ///
    /// OBSERVABLE: `[f64-lift] policy step=lp_maximin ...` at scored-run entry
    /// and one `[f64-lift] step=lp_maximin iters=...` line per post-BaB tail.
    ///
    /// CONSUMER: `commands::vnncomp::f64_target_attack::PostBabActiveSetPolicy`,
    /// then `f64_target_attack::f64_lift_lineage` in `try_postbab_falsify`.
    pub(crate) f64_lift_step: Option<F64LiftStep>,
    /// `attack.f64_basin_hops` — basin hops of the `lp_maximin` lift
    /// (#sbench-f64-maximin): when an SLP run stalls, restart it from the
    /// incumbent kicked by uniform noise of a log-uniform radius `1e-5..1e-2`
    /// of each coordinate's width, keeping the better point; at most this many
    /// times per lineage, inside the lineage's deadline.
    ///
    /// Absent/0 = no hops. A positive value requires
    /// `f64_lift_step: lp_maximin` (refused otherwise), and at most
    /// [`F64_BASIN_HOPS_MAX`]. No environment twin, so no search-space axis.
    ///
    /// SOUNDNESS: attack-only (a kick only changes which points are tried).
    ///
    /// CONSUMER: as [`Self::f64_lift_step`].
    pub(crate) f64_basin_hops: Option<usize>,
    /// Attack mode: "PGD" or "diversed_PGD" (OSI init).
    pub(crate) attack_mode: Option<String>,
    /// OSI initialization steps; only with `attack_mode: diversed_PGD`.
    pub(crate) osi_steps: Option<usize>,
    /// Straight-through-estimator surrogate gradient for Sign layers during
    /// ATTACK gradient estimation (#surrogate-sign). For binarized nets
    /// (traffic_signs QConv/Sign) the default tanh smooth relaxation
    /// saturates to a zero gradient; STE keeps the signal at any scale.
    pub(crate) surrogate_sign_gradient: Option<bool>,
    /// Dense deterministic grid sweep over low-effective-dimension input
    /// boxes as a pre-PGD attack phase (#dense-sweep).
    pub(crate) dense_low_dim_sweep: Option<bool>,
    /// Effective-dimension gate for the dense sweep (#dense-sweep).
    pub(crate) dense_sweep_max_dims: Option<usize>,
    /// Forward-evaluation budget for the dense sweep (#dense-sweep).
    pub(crate) dense_sweep_points: Option<usize>,
    /// Allow the VNN-COMP wrapper's trusted exact-gradient attack to consider
    /// low-dimensional, single-clause relational conjunctions.
    ///
    /// This is deliberately default-off and narrower than the existing
    /// automatic multi-clause-disjunction route. It only changes
    /// counterexample candidate generation; every candidate still uses the
    /// inward-rounded input box and the unchanged ONNX Runtime + true-f64
    /// terminal gate.
    pub(crate) vnncomp_upfront_relational_exact_gradient: Option<bool>,
    /// Step geometry of the wrapper's upfront exact-gradient lane on a
    /// single-output threshold objective (#attack-vertex-geometry).
    ///
    /// `apgd` (absent) is today's per-dimension APGD rule: restart 0 at the
    /// box centre, a step of `0.25 * width_d` blended with Nesterov momentum,
    /// halved after 6 stalls. `global_max_eps` is alpha-beta-CROWN's
    /// `pgd_order: before` geometry: every restart starts uniformly in the
    /// box, and every dimension moves by ONE scalar step
    /// `alpha_t = 0.99^t * max_d(width_d) / 8` (abc's `max_eps / 4`) along the
    /// sign of the exact gradient, projected onto the box, for at most 40
    /// steps per restart, with no stall-halving and no re-centring. Because
    /// that scalar exceeds most dimensions' width, narrow dimensions flip
    /// vertex-to-vertex while wide ones move gradually. MEASURED off-box on
    /// `300_ieee` prop2 (a plateau at `Y = -9800.7` from the centre to
    /// `t = 0.5` toward the witness): 17-18 of 40 restarts hit with this rule
    /// against 0-2 of 25 with the APGD rule.
    ///
    /// Engages only where the lane ascends the raw margin
    /// (`AttackLossSchedule::MarginOnly`, i.e. a single-output constant
    /// threshold admitted by `upfront_attack: true`) on a non-`Sign` net;
    /// every other row keeps the APGD rule byte-for-byte.
    ///
    /// SOUNDNESS: attack-only. It changes which candidate points the search
    /// proposes, never how one is accepted: every candidate still passes the
    /// trusted ORT forward + zero-tolerance `property_violated_f64`, and the
    /// caller's `gate_sat_with_trusted_oracle`. It cannot produce `unsat`.
    ///
    /// CONSUMER: `commands::vnncomp::UpfrontAttackShape::from_snapshot`, read
    /// from the frozen preset snapshot like `upfront_attack_probe_first`'s
    /// lower layer. No environment variable exists for it.
    pub(crate) upfront_step_geometry: Option<UpfrontStepGeometry>,
    /// Number of attack restarts seeded at box corners
    /// (#attack-vertex-geometry): the first corner restart at the lower
    /// corner, the second at the upper corner, the rest at random box
    /// vertices; every other restart keeps its historical seed.
    ///
    /// Reaches three restart schedules: the wrapper's upfront exact-gradient
    /// lane (restart 0 keeps its centre/witness seed; the corners take
    /// restarts `1..=N`, displacing first the centre replay at restart 1),
    /// the engine's upfront conjunctive sampling+SPSA attack (restarts
    /// `0..N`), and the classified `PgdAttacker` restarts (restarts `0..N`,
    /// via `BetaCrownConfig::pgd_corner_seed_restarts`). Absent/0 keeps all
    /// three byte-for-byte.
    ///
    /// MEASURED off-box on safenlp `ruarobot/hyperrectangle_1800` (field
    /// `sat`, ny `timeout`): descents from the `lo`/`hi` corners find the
    /// counterexample deterministically, uniform interior seeds 7/20, the
    /// box centre 0/20 (a strict local minimum)
    /// (`reports/measured-2026/safenlp_undecided_2026-08-30/NOTES.md` §5).
    ///
    /// SOUNDNESS: attack-only; a corner is a point of the input box, and a
    /// candidate becomes `sat` only through the unchanged trusted gate.
    pub(crate) corner_seed_restarts: Option<usize>,
    /// `attack.concurrent` — run the verifier's conjunctive / single-clause
    /// UPFRONT PGD concurrently with the proof instead of before it
    /// (#attack-concurrent).
    ///
    /// WHY. The upfront PGD slice (`phase_budget.upfront_pgd_fraction` of the
    /// verifier's wall, 0.20 by default and 0.4 on acasxu) runs to its deadline
    /// on every row it does not falsify, BEFORE any proof work: the cost census
    /// (`review/cost-census` 4fb15b19d) measured attack time as 0.614 of the
    /// wall on unsat rows, of which at least 0.85 was provably serial
    /// (dist_shift 57.0 of 57.5 s). Armed, the attack runs on a sibling thread
    /// in a private Rayon pool while the proof runs on the calling thread with
    /// the global pool; the attack's window is stretched so it keeps at least
    /// today's CPU-seconds, a decided proof cancels it, and an attack
    /// counterexample goes through the scored path's unchanged sat pipeline.
    ///
    /// Absent/false (every shipped preset): the sites run today's serial code
    /// byte-for-byte and print nothing. No environment variable exists for it;
    /// the router routes it as the `attack-concurrent` capability once sealed.
    ///
    /// SOUNDNESS: scheduling only. The proof is the unchanged code under the
    /// unchanged absolute deadlines; a `sat` is published only after the
    /// unchanged `gate_sat_with_trusted_oracle` and VNN-LIB 2.0 normalization,
    /// first-verdict-wins under the #sat-precedence lock.
    ///
    /// CONSUMER: `BetaCrownConfig::attack_concurrent`, read at the two race
    /// sites (`verify/graph.rs`, `verify/sequential.rs`).
    pub(crate) concurrent: Option<bool>,
    /// Opt-in CPU point/VJP advice over both original shared-input towers.
    /// Consumed by the relational early route's frozen policy projection;
    /// absent/false preserves the legacy falsifier. Candidate acceptance stays
    /// with the original dual-model gate. No environment override exists.
    pub(crate) relational_point_vjp: Option<bool>,
    /// Arm the root-pruned, per-disjunct, many-restart attack stage
    /// (#attack-open-disjuncts, VIBECHECK_PLAN §6.V3).
    ///
    /// On a multi-clause disjunction, after the disjunctive attack phase, the
    /// stage reads the margin-row lane's certified ROOT per-class bounds
    /// (waiting at most `open_disjunct_root_wait_secs` for them), DROPS every
    /// disjunct the root closed, orders the rest by root lower bound (lowest
    /// first), and assigns restart `r` to open disjunct `r % n_open`; each
    /// restart ascends its own disjunct's margin with the exact batched VJP,
    /// from uniform or OSI starts. Without a published root every disjunct is
    /// open and the order is the clean-point margin. This is Vibecheck's
    /// `restrict_disj` + `per_restart_disj` (cct route C) and its Phase 2.6
    /// per-open-class attack (tinyimagenet).
    ///
    /// MEASURED MOTIVATION (field, not ny): Vibecheck finds cct
    /// `eps8_cnn7 idx8090` / `eps8_wide idx8397` at 4.0 / 5.9 s and
    /// tinyimagenet `idx_1831` / `idx_5529` at 16.4-16.8 s this way, where
    /// ny's joint low-restart disjunctive PGD ends `exhausted`
    /// (`scratchpad/vbs/vibecheck/study_{1,3}.md`). Unmeasured in ny.
    ///
    /// SOUNDNESS: attack-only. Every candidate still passes the unchanged
    /// `re_evaluate_and_confirm` (noise-scaled floor `max(1e-4,
    /// 1e-5*max|y|)`, attack box, per-clause box) and then the trusted-ORT
    /// `gate_sat_with_trusted_oracle`. It cannot produce `unsat`; skipping a
    /// root-closed disjunct cannot lose a real `sat` because a certified
    /// positive root bound excludes every counterexample of that disjunct.
    ///
    /// Absent/false: the stage does not exist and the lane never publishes
    /// (byte-identical). No environment variable exists for it.
    /// CONSUMER: `BetaCrownConfig::open_disjunct_attack` via
    /// [`open_disjunct_attack_from_preset`]; `vnncomp::run_and_translate`
    /// arms the lane's root publication from the same mapping.
    pub(crate) open_disjunct_restrict: Option<bool>,
    /// Floor on each attacked disjunct's share of the stage wall, seconds
    /// (#attack-open-disjuncts; default 0.5). When `n_open * min_secs` does
    /// not fit the stage budget only the lowest-root-bound disjuncts that fit
    /// are attacked. Must be > 0.
    pub(crate) open_disjunct_min_secs: Option<f32>,
    /// Wall cap of the stage's ATTACK, seconds (#attack-open-disjuncts;
    /// default 10), counted from the end of the root wait; the whole stage
    /// (wait + attack) is further clamped by `open_disjunct_max_frac` and the
    /// instance deadline.
    pub(crate) open_disjunct_max_secs: Option<f32>,
    /// Restarts per attacked disjunct (#attack-open-disjuncts; default 100).
    /// The stage ends early once they have all run.
    pub(crate) open_disjunct_restarts: Option<usize>,
    /// Longest wait for the margin-row lane's certified root, seconds
    /// (#attack-open-disjuncts; default 0 = use a root only if it is already
    /// published). A lane that is not running is never waited on.
    pub(crate) open_disjunct_root_wait_secs: Option<f32>,
    /// Share, in `(0, 1]`, of the time remaining at stage start that the
    /// WHOLE stage (root wait + attack) may use (#attack-open-disjuncts;
    /// default 0.25). Keeps one arming safe across budget tiers: the absolute
    /// seconds above are caps, and `1 - max_frac` of the clock is what the BaB
    /// hand-off always keeps. The root wait always leaves one
    /// `open_disjunct_min_secs` floor of this share for the attack.
    pub(crate) open_disjunct_max_frac: Option<f32>,
    /// Arm the LP-guided sign-space falsification lane (#bnn-sign-space).
    ///
    /// Typed preset form of the `NY_BNN_SIGN_SPACE` dark lever, in the shape
    /// `attack.upfront_attack` and `margin_row.root_f32` already use: the
    /// environment variable still WINS wherever it is present, in both
    /// directions (`1` arms whatever the preset says, `0` disarms it, and any
    /// other token is a recorded rejection that falls back to the DECLARATION
    /// default — `ny_levers::read_over_config`). Absent, this key decides.
    ///
    /// WHY THIS KEY EXISTS. An environment variable does not reach a scored
    /// competition run, so the three `model_30` eps=1 rows the lane captures —
    /// open in every result bank inspected, 0 verified / 36 falsified by every
    /// entrant across three competition years — were a capability result and
    /// not a score. The measured armed sweep is 30/45 versus 27/45 unarmed
    /// (`reports/measured-2026/traffic_signs_recognition_2023_NOTES.md`).
    ///
    /// SOUNDNESS IS NOT DELEGATED TO THIS KEY. `ny_mip::SignSpaceOutcome` has
    /// no verified/unsat variant BY CONSTRUCTION, so the lane cannot cause a
    /// false `unsat` on any setting; a candidate becomes a `sat` only by
    /// passing the unchanged `gate_sat_with_trusted_oracle` (real ONNX Runtime
    /// forward on the ORIGINAL model plus the true-f64 recheck); and admission
    /// is STRUCTURAL — a net outside the binarized `Conv -> B -> Conv -> B ->
    /// Dense` fragment is refused and falls through with the verdict
    /// unchanged. What this key really decides is BUDGET: armed, the lane may
    /// take `min((remaining - 45 s) / 2, 4 min)` inside the attack slice before
    /// the ordinary lanes run. That is why it is scoped to the category where
    /// that spend is measured rather than shipped on globally.
    ///
    /// CONSUMER: `commands::vnncomp::run_and_translate` projects the typed key
    /// into the entry-time `LeverSet`; `BnnFalsificationPolicy::from_levers`
    /// then freezes the resolved value before any private, allocated, or
    /// scheduled lane can run.
    pub(crate) bnn_sign_space: Option<bool>,

    /// Adopt the minimum feasible segment move instead of the LP vertex in
    /// sign-space falsification (`NY_BNN_SIGN_SPACE_MINIMAL_MOVE`).
    pub(crate) bnn_sign_space_minimal_move: Option<bool>,

    /// Pixel trust-region arm for sign-space falsification: `box`, `tight`, or
    /// `linf` (`NY_BNN_SIGN_SPACE_TRUST_REGION`).
    pub(crate) bnn_sign_space_trust_region: Option<String>,

    /// Arm the STE-PGD falsification lane (`#bnn-ste-pgd`) for this category.
    ///
    /// The sibling of [`Self::bnn_sign_space`], over the SAME structurally
    /// admitted binarized fragment and with the same soundness story:
    /// `ny_mip::SignSpaceOutcome` has no verified/unsat variant BY
    /// CONSTRUCTION, so the lane cannot cause a false `unsat` on any setting,
    /// and a candidate becomes a `sat` only by passing the unchanged
    /// `gate_sat_with_trusted_oracle`. What the key really decides is BUDGET:
    /// armed, the lane may take everything left to it after the publication
    /// margin and the downstream reserve, capped at 4 minutes.
    ///
    /// WHY IT IS A SEPARATE KEY. The two lanes are COMPLEMENTARY, not
    /// redundant, and measurably so: the LP search takes the three `model_30`
    /// eps=1 rows this one cannot (measured `exhausted`, best margin -222,
    /// 97 flips), and this one takes seven 48x48/64x64 rows the LP search
    /// cannot, whose witnesses sit 483-1483 first-layer flips from the box
    /// centre.
    ///
    /// CONSUMER: `commands::vnncomp::run_and_translate` projects the typed key
    /// into the entry-time `LeverSet`; `BnnFalsificationPolicy::from_levers`
    /// then freezes the resolved value before any private, allocated, or
    /// scheduled lane can run.
    pub(crate) bnn_ste_pgd: Option<bool>,

    /// `attack.upfront_attack_probe_first` — consult the EXISTING upfront
    /// DLR-APGD falsification lane once at the TOP of the attack slice, under
    /// the small fixed `UPFRONT_ATTACK_PROBE_CAP` (5 s), ahead of the two BNN
    /// lanes, instead of only behind them (#upfront-probe-first).
    ///
    /// WHAT IT DECIDES IS ORDER, NOT CAPABILITY. Nothing is added and nothing
    /// is removed: on a miss the LP sign-space and STE-PGD lanes run
    /// afterwards in today's order under today's windows, and the ordinary
    /// (uncapped) upfront consultation still happens behind them. The probe's
    /// seconds are charged against the same `attack_start` subtraction every
    /// attack-slice lane pays, so arming it cannot push branch-and-bound past
    /// the scored deadline.
    ///
    /// MEASURED, 45/45 rows of `traffic_signs_recognition_2023` at the official
    /// 480 s budget, arms interleaved per row (2026-08-21, see
    /// `reports/measured-2026/traffic_signs_recognition_2023_NOTES.md`):
    /// **0 rows lost, 15 rows won by the probe**, 12 of them the `model_30`
    /// eps>=3 instances that otherwise land at t=242-336 s and now land at
    /// t=1.4-5.6 s. Summed family solve time 7558 s -> 5335 s. On the 30 rows
    /// the probe CANNOT win it costs the cap and nothing more — measured
    /// 5.00-5.03 s on all 30, including the three `model_30` eps=1 rows the LP
    /// walk owns, which keep their winner and their bit-identical trajectory.
    /// Worst-case wall is unmoved (dark 468.65 s, armed 468.70 s of 480 s),
    /// because on rows that exhaust their budget the probe DISPLACES seconds
    /// rather than adding them.
    ///
    /// SOUNDNESS. Identical to the lane's ordinary consultation, because it IS
    /// that lane: it returns a claimed counterexample INPUT and never a
    /// verdict, so it cannot produce an `unsat`, and a candidate becomes a
    /// `sat` only by passing the EXISTING, UNCHANGED
    /// `gate_sat_with_trusted_oracle`. A candidate the gate drops falls
    /// through to the ordinary slice (#upfront-gate-fallthrough).
    ///
    /// `NY_UPFRONT_ATTACK_PROBE_FIRST` still WINS wherever it is present, in
    /// both directions (exact `"1"` arms, exact `"0"` disarms, anything else is
    /// a recorded rejection falling back to the declaration default).
    ///
    /// CONSUMER: `commands::vnncomp::preset_delivered_config_value`, which
    /// projects it into the shared entry-time LeverSet consumed by
    /// `UpfrontAttackPolicy::from_levers`.
    pub(crate) upfront_attack_probe_first: Option<bool>,

    // ---------------------------------------------------------------------
    // #lever-reachability (2026-08-23). The six keys below were promoted in
    // one batch from environment-only gates. Each is DEFAULT OFF and set by no
    // shipped yaml — pinned by
    // `no_shipped_preset_arms_a_promoted_lever_delivery_key` — so this batch is
    // byte-identical on every scored run. Promotion makes a lever
    // DELIVERABLE; arming one stays a separate, measured decision.
    //
    // WHY THEY ALL BELONG TO THE ATTACK SLICE. That slice is where ny earns
    // `sat`, the half of the score branch-and-bound cannot reach, and every
    // lane in it was reachable only through an environment variable that
    // `vnncomp_scripts/run_instance.sh` does not export. So the entire
    // falsification portfolio was, in competition, dead code.
    //
    // SOUNDNESS, once for the group: no lane in the attack slice has a
    // verdict-bearing return type. Each returns a CLAIMED counterexample
    // INPUT, and a claim becomes `sat` only by passing the unchanged
    // `gate_sat_with_trusted_oracle`; none can produce `unsat`. Arming any of
    // these can therefore lose time (hence proofs) but cannot manufacture a
    // verdict.
    // ---------------------------------------------------------------------
    /// Run the falsification portfolio (`NY_FALSIFY_PORTFOLIO`).
    ///
    /// The lane's own reader used to document the defect verbatim: "Environment
    /// only, and deliberately: there is no typed `attack.*` preset key, so a
    /// competition harness (which exports no `NY_*`) cannot reach this lane at
    /// all." This key is that missing route. Env wins in both directions.
    pub(crate) falsify_portfolio: Option<bool>,

    /// Wall cap in seconds for the portfolio above
    /// (`NY_FALSIFY_PORTFOLIO_SECONDS`). `0` or absent selects the derived
    /// rule. Taken from the same instance window the sound lanes spend, so it
    /// is a real budget trade rather than free time.
    pub(crate) falsify_portfolio_secs: Option<u64>,

    /// Active-block cap for the flat-objective surrogate Hadamard ladder
    /// (`NY_FLAT_SURROGATE_MAX_BLOCKS`, #flat-ladder-depth). `0` or absent
    /// keeps the historical split-all schedule byte-identically — block count
    /// doubles per level, so on smart_turn the ladder can never afford level
    /// 13 under the official 100 s budget. A cap `>= 2` switches to selective
    /// refinement: once a split would exceed the cap, only the top cap/2
    /// blocks by recovered effect stay active and the rest freeze at their
    /// incumbent faces, so refinement continues deeper at ~cap probes per
    /// level. Steering only; the acceptance stack (trusted-oracle gate
    /// included) is unchanged and the ladder cannot produce `unsat`.
    pub(crate) flat_surrogate_max_blocks: Option<u64>,

    /// Arm the surrogate->real transfer stage of the flat-objective surrogate
    /// search (`NY_FLAT_SURROGATE_TRANSFER`, #flat-surrogate-transfer). `false`
    /// or absent keeps the surrogate search byte-identical: on a
    /// quantizer-pinned row the ladder can drive the pre-saturation SURROGATE
    /// margin past zero while the REAL property margin stays at exactly +0.0
    /// (measured on smart_turn_multimodal_2026 row 11, 2026-08-31), and the
    /// search ends there. `true` hands over once the surrogate crosses zero:
    /// refinement keeps half of the remaining budget for a deeper seed, and
    /// the transfer stage spends the other half walking the REAL output's
    /// quantizer staircase from the surrogate optimum (surrogate-flat-subspace
    /// block moves first — the coordinates the surrogate cannot see are the
    /// ones free to tip a boundary-pinned rounding). Steering only; the
    /// acceptance stack (trusted-oracle gate included) is unchanged and the
    /// stage cannot produce `unsat`.
    pub(crate) flat_surrogate_transfer: Option<bool>,

    /// DIAGNOSTIC-ONLY side file of the graph-free falsification lane
    /// (`NY_GRAPH_FREE_CANDIDATE_SIDEFILE`, #graph-free-candidates,
    /// #smart-turn-x86-harvest). `false` or absent constructs no recorder and
    /// the lane is byte-identical. `true` writes every candidate the lane
    /// offers to its acceptance gates — accepted or rejected, with the gate
    /// that decided it, the exact f32 input tensor that ran and ny's own
    /// default-level ONNX Runtime outputs — plus a bounded sample of the
    /// flat-objective surrogate's re-scored incumbents, to
    /// `<RESULTS_FILE>.graph_free_candidates.jsonl`, for OFFLINE replay on a
    /// checker-parity x86-64 configuration (arm64 ONNX Runtime is not the
    /// organisers' checker on smart_turn's quantized model, measured
    /// 2026-09-26). Report-only: no gate changes, nothing in the file is
    /// published, and the key can produce neither `sat` nor `unsat`. It is an
    /// INSTRUMENT (space.rs `INSTRUMENT_ONLY`), never a treatment arm.
    pub(crate) graph_free_candidate_sidefile: Option<bool>,

    /// Score the attack against the PRE-Softmax objective
    /// (`NY_ATTACK_PRE_SOFTMAX_OBJECTIVE`) instead of the post-Softmax one.
    ///
    /// Disarmed — the shipped default — the objective, direction row and DLR
    /// denominator are the historical post-Softmax ones and the path is
    /// byte-identical. It changes which counterexamples are findable.
    pub(crate) pre_softmax_objective: Option<bool>,

    /// Size every attack-slice lane once, up front, from its own unchanged rule
    /// (`NY_LANE_BUDGET_FRAME`), so a lane that declines cannot have its unspent
    /// seconds absorbed by a downstream lane that re-derives its window from the
    /// live remaining budget.
    pub(crate) lane_budget_frame: Option<bool>,

    /// Commit every attack-slice lane's cap jointly and up front by solving a
    /// multiple-choice knapsack (`NY_LANE_BUDGET_ALLOCATOR`, Layer A).
    ///
    /// Can take a structurally blind lane to zero seconds, so it changes which
    /// lane runs under what cap and therefore which counterexamples are
    /// reachable within the budget.
    pub(crate) lane_budget_allocator: Option<bool>,

    /// Reallocate a stalled lane's remaining seconds to a later lane
    /// (`NY_LANE_VALUE_SCHEDULER`). Scheduling only; see the module Soundness
    /// note in `commands::lane_schedule`.
    pub(crate) lane_value_scheduler: Option<bool>,
}

/// Explicit NY compatibility choices for unimplemented reference PGD orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NyPgdOrderCompat {
    /// Preserve NY's historical initial/upfront attack placement.
    Upfront,
}

/// Step rule of the post-BaB float64 lift (`attack.f64_lift_step`,
/// #sbench-f64-maximin).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum F64LiftStep {
    /// #sbench-f64-attack's active-set min-norm Newton row lift (the default;
    /// absent is identical).
    #[default]
    MinNorm,
    /// The sequential LP-maximin step over every conjunct, with
    /// `attack.f64_basin_hops` basin hops.
    LpMaximin,
}

impl F64LiftStep {
    /// Stable token for receipts; equals the yaml spelling.
    pub(crate) const fn stable_token(self) -> &'static str {
        match self {
            Self::MinNorm => "min_norm",
            Self::LpMaximin => "lp_maximin",
        }
    }
}

/// Step geometry of the wrapper's upfront exact-gradient lane
/// (`attack.upfront_step_geometry`, #attack-vertex-geometry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UpfrontStepGeometry {
    /// Today's per-dimension APGD rule (the default; absent is identical).
    #[default]
    Apgd,
    /// alpha-beta-CROWN's global scalar sign step, `max_eps / 4` decayed by
    /// 0.99 per step, from uniform starts, at most 40 steps per restart.
    GlobalMaxEps,
}

impl UpfrontStepGeometry {
    /// Stable token for receipts; equals the yaml spelling.
    pub(crate) const fn stable_token(self) -> &'static str {
        match self {
            Self::Apgd => "apgd",
            Self::GlobalMaxEps => "global_max_eps",
        }
    }
}

/// MIP complete-verifier configuration (`solver.mip` in alpha-beta-CROWN).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct MipPreset {
    /// MIP backend: "ay" (the only solver — SOLVER POLICY, ny-mip
    /// docs/SOLVER_POLICY.md). Legacy values "highs"/"scip" and
    /// alpha-beta-CROWN's "gurobi" resolve to ay with a warning.
    /// An explicit `--mip-solver` CLI choice wins.
    pub(crate) mip_solver: Option<String>,
    /// Number of parallel MIP solver processes/splits. Reserved for the
    /// phase-split racing mode; parsed for alpha-beta-CROWN key compatibility
    /// (`solver.mip.parallel_solvers`).
    pub(crate) parallel_solvers: Option<usize>,
}

/// Solver configuration (`solver:` in alpha-beta-CROWN).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct SolverPreset {
    /// Opt-in paired-ReLU certifier on the shared-input relational route.
    /// Absent keeps historical scheduling. No legacy environment alias exists.
    pub(crate) paired_difference: Option<PairedDifferencePreset>,
    /// Batch size for parallel domain processing.
    pub(crate) batch_size: Option<usize>,
    /// Maximum root spec rows per build batch (`solver.build_batch_size`).
    pub(crate) build_batch_size: Option<usize>,
    /// Automatically enlarge batch size based on GPU memory.
    pub(crate) auto_enlarge_batch_size: Option<bool>,

    /// Minimum batch size ratio when auto-enlarging.
    pub(crate) min_batch_size_ratio: Option<f32>,

    /// Bound propagation method: `crown`, `alpha-crown`, `forward+backward`, or `forward+crown`.
    /// Maps to `BetaCrownConfig::{use_alpha_crown,use_forward_bounds}` and rejects
    /// unsupported alpha-beta-CROWN modes instead of silently coercing to another setting.
    /// Alpha-beta-CROWN reference key: `solver.bound_prop_method`.
    #[serde(alias = "bound-prop-method")]
    pub(crate) bound_prop_method: Option<String>,

    /// α-CROWN configuration under solver (alpha-beta-CROWN naming).
    #[serde(default, alias = "alpha-crown")]
    pub(crate) alpha_crown: AlphaCrownPreset,

    /// β-CROWN configuration under solver (alpha-beta-CROWN naming).
    #[serde(default, alias = "beta-crown")]
    pub(crate) beta_crown: BetaCrownPreset,

    /// MIP complete-verifier configuration (alpha-beta-CROWN `solver.mip`).
    #[serde(default)]
    pub(crate) mip: MipPreset,

    /// Retired typed spelling for the quarantined
    /// `NY_CONST_MATMUL_INTERVAL` research carrier.
    ///
    /// Kept only so an older research YAML receives the explicit quarantine
    /// error from semantic validation. Any `Some` value, including `false`, is
    /// rejected: the Unsafe axis is deliberately EnvOnly because the carrier
    /// exposed a false-proof propagation path and may not be armed by a scored
    /// preset.
    pub(crate) const_matmul_interval: Option<bool>,

    /// `solver.joint_bilinear_quad` — typed delivery for the
    /// `NY_JOINT_BILINEAR_QUAD` dark correlation-preserving relaxation.
    /// `None` preserves the historical default-OFF behavior; no shipped preset
    /// sets this key. A present environment value still wins in both directions
    /// through the central lever resolver.
    pub(crate) joint_bilinear_quad: Option<bool>,

    /// `solver.scored_wall_clamp` — stop branch-and-bound at an ABSOLUTE
    /// instant derived from the scored budget, independent of the effort
    /// horizon its timeout shapes (#scored-wall-clamp).
    ///
    /// WHAT IT SEPARATES. The disjunctive input-split lane sizes every
    /// per-node CROWN-IBP grant as a fraction of the budget it believes still
    /// REMAINS, so a nearer deadline buys looser bounds, more splits, and a
    /// non-converging tree — the search is measurably NON-MONOTONIC in its own
    /// timeout. The relational lane exploits that by inflating its BaB
    /// deadline 1.4x (`DEFAULT_REL_BAB_DEADLINE_MULT`). Before this key, the
    /// inflated number was ALSO the stop, so the trajectory was bought with
    /// wall-clock the process did not own.
    ///
    /// MEASURED, `isomorphic_acasxu_2026` at the official 100 s budget: on the
    /// banked 50-row sweep EIGHTEEN rows ran past the 95 s internal tier and
    /// were SIGKILLed by the in-process watchdog at 105.0-105.1 s — 105 % of
    /// the official budget, and on the competition harness an external kill at
    /// 100 s leaves them with no verdict token at all.
    ///
    /// WHAT IT CHANGES. Only the loop's termination checks. `crown_deadline`,
    /// the initial-bounds fraction and every per-node collector deadline still
    /// derive from the UNCHANGED horizon, so the trajectory up to the wall is
    /// the trajectory the multiplier bought. Clamping the horizon instead
    /// (equivalently `NY_REL_BAB_DEADLINE_MULT=1.0`) also stops the overrun but
    /// degrades the bounds and loses rows.
    ///
    /// SOUNDNESS. It can only ever SHORTEN a search, and a shortened sound
    /// search returns `timeout`, never a verdict. A wall beyond the horizon is
    /// ignored, so it cannot extend a configured timeout either.
    ///
    /// `NY_SCORED_WALL_CLAMP` still WINS wherever it is present, in both
    /// directions (exact `"1"` arms, exact `"0"` disarms, anything else is a
    /// recorded rejection falling back to the declaration default).
    ///
    /// CONSUMER: `commands::vnncomp::handle_vnncomp_command`, which passes it
    /// to `scored_wall_clamp_armed` as the lever's config layer.
    pub(crate) scored_wall_clamp: Option<bool>,
}

/// Numerical LP proposal provider. Both backends require NY source replay;
/// neither provider's status is neural verdict authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PairedLpBackend {
    Native,
    /// Approximate equality elimination and scaling for AY proposals, followed
    /// by independent replay against the complete original source relaxation.
    #[default]
    Ay,
}

impl PairedLpBackend {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Ay => "ay",
        }
    }
}

/// Budget for the structurally admitted paired-network proof-only prepass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PairedDifferencePreset {
    /// Absolute local cap; live relational and scored deadlines also apply.
    pub(crate) max_seconds: f64,
    /// Resource cap. Reaching it returns no proof with all open boxes retained.
    pub(crate) max_domains: u64,
    /// Optional joint relaxation before input splitting. This time is
    /// charged inside max_seconds; a missing value leaves the prepass disabled.
    #[serde(default)]
    pub(crate) max_lp_seconds: f64,
    /// Proposal provider; AY uses its reduced floating proposal path, requires
    /// `mip`, and supplies no proof authority. NY checks full source rows.
    #[serde(default)]
    pub(crate) lp_backend: PairedLpBackend,
    /// AY proposal workspace cap in MiB, validated in [16,2048]. Native
    /// proposals retain their existing structural resource caps.
    #[serde(default = "paired_difference_default_lp_memory_mib")]
    pub(crate) lp_memory_mib: usize,
    /// Native numerical proposal steps per output sign. Retained and validated
    /// for preset compatibility; AY does not interpret this optimizer budget.
    #[serde(default = "paired_difference_default_lp_steps")]
    pub(crate) lp_steps: usize,
    /// Bounded proof workers; a missing value uses one worker.
    #[serde(default = "paired_difference_default_workers")]
    pub(crate) workers: usize,
}

const fn paired_difference_default_workers() -> usize {
    1
}

const fn paired_difference_default_lp_steps() -> usize {
    2_000
}

const fn paired_difference_default_lp_memory_mib() -> usize {
    2_048
}

/// MIP SCHEDULE governors for the two lanes that can eat a branch-and-bound
/// budget without deciding anything (#metaroom-mip-governor).
///
/// Both are schedule choices between arms that are each independently sound:
/// the leaf oracle can only UPGRADE a leaf (its `Undecided` is what declining
/// already produces), and the escalation skip only removes clauses a certified
/// bound has already refuted. A wrong value here costs proofs; it cannot
/// manufacture one.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct BabMipPreset {
    /// Stop consulting the Graph-MIP leaf oracle after this many CONSECUTIVE
    /// `Undecided` replies (`NY_MIP_LEAF_UNDECIDED_STREAK_CAP`). `None` and
    /// `0` both mean NO CAP, which is the shipped behaviour: today only a
    /// confirmed SAT witness or the cumulative-wall fraction
    /// (`NY_GRAPH_MIP_LEAF_TOTAL_FRAC`, default 0.5 — half the BaB budget)
    /// can disarm the oracle.
    #[serde(default)]
    pub(crate) leaf_undecided_streak_cap: Option<usize>,

    /// Seed the disjunctive MIP escalation's pending clause set from the
    /// UNCERTIFIED clauses only (`NY_MIP_ESCALATE_SKIP_CERTIFIED`). `None`
    /// and `false` keep the historical all-clauses schedule.
    ///
    /// A disjunctive `Unsat` still requires every clause to be infeasible; a
    /// skipped clause is one a sound certified bound has already refuted, and
    /// it is booked as certified-unsat, never as undecided. Every step of the
    /// prescreen fails closed to the all-clauses set.
    #[serde(default)]
    pub(crate) escalation_skips_certified_clauses: Option<bool>,
}

/// Branch-and-bound (BaB) configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct BabPreset {
    /// Batch size for parallel domain processing.
    pub(crate) batch_size: Option<usize>,

    /// MIP schedule governors (`bab.mip.*`, #metaroom-mip-governor). An absent
    /// section is byte-identical to the pre-promotion tree.
    #[serde(default)]
    pub(crate) mip: BabMipPreset,

    /// Maximum number of output-adjacent backward layers/nodes for fixed-slope CROWN.
    pub(crate) crown_backward_layers: Option<usize>,

    /// Automatically enlarge batch size based on GPU memory.
    pub(crate) auto_enlarge_batch_size: Option<bool>,

    /// Minimum batch size ratio when auto-enlarging.
    pub(crate) min_batch_size_ratio: Option<f32>,

    /// Maximum simultaneous ReLU split depth for one parent domain.
    ///
    /// This is NY's defensive cap around alpha-beta-CROWN's adaptive
    /// `get_split_depth()` policy. The reference has no corresponding cap.
    #[serde(alias = "max_relu_split_depth")]
    pub(crate) max_split_depth: Option<usize>,

    /// Timeout in seconds.
    pub(crate) timeout: Option<u64>,

    /// Maximum domains to explore.
    pub(crate) max_domains: Option<usize>,

    /// Estimated payload cap in BYTES for supported graph BaB frontiers:
    /// ordinary/precomputed ReLU heaps and GPU DomainList ReLU/input split
    /// (#ml4acopf-bab-queue-mem). `max_domains` is count-based and model-blind.
    /// Over-budget domains are evicted lowest-priority-first and force the run
    /// to `unknown`, never `Verified`. Unset (or 0) keeps the unlimited queue.
    /// Grouped-disjunctive DomainList rejects a nonzero byte cap.
    pub(crate) max_queue_bytes: Option<usize>,

    /// Maximum tree depth.
    pub(crate) max_depth: Option<usize>,

    /// #budget-abandonment FIX A: recover from the initial-bound warmup cap
    /// instead of abandoning the rest of the budget.
    ///
    /// `bab_loop.rs` runs the mandatory node sweep on the FULL BaB deadline and
    /// the iterative root alpha-CROWN pass on `initial_bounds_fraction` of it
    /// (0.20). When only the ITERATIVE pass blows its cap, the engine returns
    /// `Unknown` and BaB never starts, even though the bootstrap's certified
    /// output-node enclosure is already in hand. Measured at HEAD on
    /// 14_ieee_ml4acopf: bootstrap 0 -> 121.0 s against a 114 s cap, 449 s of
    /// the 600 s budget abandoned.
    ///
    /// Armed, the engine falls back to that enclosure (after one bounded CROWN
    /// retry on a fresh slice) and lets BaB run on the remaining ~80%. A wider
    /// root can only make `domain_is_verified` / `domain_is_violation` HARDER to
    /// satisfy, so it can mint neither an `unsat` nor a spurious violation.
    ///
    /// Absent (the shipped state on every yaml) = dark = byte-identical.
    pub(crate) initial_bounds_warmup_fallback: Option<bool>,

    /// #budget-abandonment FIX B: extra passes over the OR-clauses a single
    /// pass could not decide.
    ///
    /// `verify/disjunctive.rs` allocates `remaining / remaining_clauses` in ONE
    /// pass, so the FIRST clauses get the SMALLEST slice and the failures
    /// concentrate there. Measured on 14_ieee_ml4acopf prop1 (56 clauses):
    /// clause 1 got 10 s and died, clause 56 got 442 s and used 2.1 s, 470 s of
    /// 600 unspent.
    ///
    /// Absent or `0` (the shipped state on every yaml) = today's single pass =
    /// byte-identical.
    pub(crate) disjunctive_clause_retry_passes: Option<usize>,

    /// #disjunctive-shared-bootstrap: compute the graph BaB bootstrap
    /// (intermediate node bounds, root alpha, identity root box) ONCE per
    /// serial multi-clause disjunction and lend it to every clause.
    ///
    /// The serial loop re-ran the whole bootstrap per OR-clause. Measured
    /// 2026-09-12 on 14_ieee_ml4acopf prop4 (28 clauses, 600 s): `Computing
    /// α-CROWN initial bounds` once PER CLAUSE, `[warmup-fallback]` firing on
    /// every clause at 6.3-7.5 s against a 5.0 s cap -- ~7 s x 28 spent
    /// recomputing the same enclosures of the same network over the same box.
    ///
    /// Absent (the shipped state on every yaml) = dark = byte-identical.
    pub(crate) disjunctive_shared_bootstrap: Option<bool>,

    /// #genbab-precomputed-lane: let the precomputed-bounds BaB lane honour a
    /// configured GenBaB nonlinear heuristic, as every other lane does. It is
    /// the only lane that never reads `branching_heuristic`, so on ml4acopf --
    /// `method: nonlinear`, all ReLUs stable -- it finds nothing to split and
    /// returns in microseconds holding the whole budget.
    ///
    /// Absent or `false` (the shipped state on every yaml) = byte-identical.
    pub(crate) genbab_on_precomputed_lane: Option<bool>,

    /// Early stopping patience (iterations without improvement).
    pub(crate) early_stop_patience: Option<usize>,

    /// Floor (seconds) for CROWN-IBP collectors' per-node time budget
    /// (#cgan-bn11-budget). Unset keeps the built-in 2.0 s constant.
    pub(crate) crown_ibp_per_node_floor_secs: Option<f64>,

    /// #per-node-floor-dim-scale: scale the CROWN-IBP per-node floor by
    /// `min(1, raw_node_dims / 28,800)` before the below-floor comparison.
    /// Composes with `crown_ibp_per_node_floor_secs` (whatever floor resolves
    /// is what gets scaled): a 128-wide dense target is held to ~0.009 s, a
    /// 28,800-row conv keeps the full 2.0 s.
    ///
    /// WHY: the share is cost-proportional and the floor is a constant
    /// calibrated for 28,800-row conv targets, so the comparison refuses
    /// exactly the CHEAP nodes. MEASURED on 11/11 retained ml4acopf early-quit
    /// logs (2026-09-12): 20/118/128-wide dense targets — including the first
    /// two Gemm layers of the 118-bus net — carried
    /// `PerNodeBudgetBelowFloor` before any work while the whole cascade used
    /// 20 s of 600, and the downstream α ascent then sat on an IBP-grade frame.
    ///
    /// SOUND by construction: an admitted target publishes CROWN-IBP instead
    /// of IBP, both sound, the former tighter-or-equal — `unknown -> unsat` is
    /// reachable, `unsat -> sat` is not. Absent (the shipped state on every
    /// yaml) = dark = the constant floor, byte-identical. Observable:
    /// `[per-node-floor] refused ... floor_secs=<held> floor_scaled=<0|1>`,
    /// printed on both arms.
    pub(crate) crown_ibp_per_node_floor_dim_scaled: Option<bool>,

    /// #pwl-lookup: intersect every ReLU LOOKUP TABLE's IBP box
    /// (`Unsqueeze → Sub(b[K]) → Relu → MatMul(w[K])`, and abc's
    /// `ReduceSum(Mul(Relu(Sub(·, b)), w), -1)`) with its exact 1-D range.
    ///
    /// WHY: the six ml4acopf `-linear-*` nets spell the output Sigmoid and the
    /// Cos/Sin corrections as 18/54/55-hinge tables; hinge-by-hinge IBP loses
    /// ~1e-6..1e-2 there, against prop3/prop4 margins of 1e-6..6.5e-6. The
    /// exact range refutes 100% of those clauses off-box (all 12 rows, both
    /// corpora); the native-Sigmoid twins bank the same properties in ~5 s.
    ///
    /// SOUND by construction: shrink-only intersection of two enclosures, with
    /// outward-rounded f64 evaluation; an empty intersection keeps the plain box
    /// and prints a `CONTRADICTION` line. Absent (the shipped state on every
    /// yaml) = dark = byte-identical. Observable: `[pwl-lookup] armed=1 census
    /// tables=<n> …` once per run, then `[pwl-lookup] armed=1 calls=<n> …
    /// tightened=<n> contradiction=<n>` (rate-limited), and `[precheck-ibp]`.
    pub(crate) pwl_lookup_exact_range: Option<bool>,

    /// #fwd-root-screen (V4, route (b)): before the disjunctive CROWN
    /// precheck, run ONE certified correlation-preserving zonotope forward
    /// pass (explicit symbol IDs, ID-aligned bilinear products, rigorous
    /// Sigmoid/Sin/Cos/Floor/lookup-table bands, a fresh symbol per unstable
    /// ReLU; every relaxation interval intersected with ny's IBP box and C1's
    /// exact range) and refute the clauses its certified output box rules out.
    ///
    /// WHY: F1 (scratchpad/vbs/vibecheck/gates/F1) verified 4 ml4acopf rows ny
    /// has open with the certified f64 replica — 14 base prop1 (+5.75e-5,
    /// 56/56), 118 lin-res prop5 (+6.09e-3), 300 lin-nonres prop105
    /// (+8.80e-3), 300 lin-res prop105 (+8.08e-3) — where ny's IBP-grade
    /// precheck is off by 1e-1..1e4.
    ///
    /// SOUND by construction: the verdict is the UNION with the existing
    /// precheck (the screen never replaces IBP/C1); every refusal fails closed.
    /// Absent (the shipped state on every yaml) = dark = byte-identical.
    /// Observable: `[fwd-root] ops=<list> admitted=<0|1> symbols=<n>
    /// refuted=<k>/<n> min_margin=<m> slack_max=<s> elapsed_ms=<t> …`, a
    /// `[fwd-root] union …` line, and a `fwd_root_screen` flight note.
    pub(crate) forward_root_screen: Option<bool>,

    /// #fwd-root-screen live-state ceiling in MiB (`64..=1048576`, default
    /// 8192; `0` is refused, never "unlimited"). Clamped at run time to half
    /// the process memory headroom.
    pub(crate) forward_root_screen_max_mb: Option<u32>,

    /// #vit-zono-root (VIBECHECK I1, the critical path to the vit_2023 pool):
    /// compute ONE certified forward-zonotope ROOT map over the input box —
    /// explicit symbol IDs, the live-live attention MatMul as an ID-aligned
    /// bilinear product with a certified remainder symbol, rigorous Exp /
    /// Reciprocal (wave-6's `[l, +inf)` case) / Clip bands, ReduceSum /
    /// ReduceMean, the direct-Softmax pairwise-difference box, BatchNorm with
    /// its certified radii, the Conv2d patch embedding and the probed
    /// Concat(CLS) / Transpose / Reshape — and intersect its per-node boxes
    /// SHRINK-ONLY into the alpha reference map the root CROWN / alpha-margin
    /// pass reads (and, on `fix_interm_bounds: false`, into the CROWN-IBP
    /// collector's baseline).
    ///
    /// WHY: ny's vit reference map is IBP or a truncated CROWN-IBP collection;
    /// the shift `x_i - x_{j*}` of the fixed-index softmax and `Q K^T` keep
    /// their shared symbols in a zonotope, which ny's own DeepZ oracle put at
    /// the full-collection root on ibp for one forward pass
    /// (scratchpad/vit_frontier_2026-08-06/RESULTS.txt §2).
    ///
    /// SOUND by construction: certified binary64 zonotope ∩ IBP per node, a
    /// runtime self-check (point semantics vs ny's pointwise forward, sampled
    /// containment) before publication, shrink-only merges (a disjoint
    /// element keeps the current value), every refusal publishes nothing.
    /// Absent (the shipped state on every yaml) = dark = byte-identical.
    /// Observable: `[vit-zono-root] armed=1 stage=stamp …` once per run,
    /// `[vit-zono-root] armed=1 admitted=<0|1> reason=… nodes=… census=…
    /// symbols=… …` per computation, `[vit-zono-root] stage=… elems_tightened=…`
    /// per reference collection, and a `vit_zono_root` flight note.
    pub(crate) vit_zono_root: Option<bool>,

    /// #vit-zono-root live-state ceiling in MiB (`64..=1048576`, default 8192;
    /// `0` is refused, never "unlimited"). Clamped at run time to half the
    /// process memory headroom.
    pub(crate) vit_zono_root_max_mb: Option<u32>,

    /// #vit-zono-root wall cap of one root-map computation in seconds
    /// (`(0, 3600]`, default 30), also bounded by the collection deadline.
    pub(crate) vit_zono_root_max_secs: Option<f32>,

    /// #vit-collector-throughput P2 step 1: run the var×var `MatMul` /
    /// `BilinearCrown` McCormick backward (midpoint and α arms) row-parallel,
    /// with each element's McCormick planes computed once per chunk of rows
    /// instead of once per row.
    ///
    /// WHY: the serial nest is `rows × batch × m × n × k` scalar iterations on
    /// one thread and recomputes the same plane for every row; on vit_2023's
    /// ibp model that is 13,872 iterations per row per attention crossing, and
    /// no intermediate collection target completes inside its share
    /// (scratchpad/vbs/diag6/vit.md §2).
    ///
    /// SOUND by construction: BIT-IDENTICAL to the serial nest (rows are
    /// independent; each row keeps its exact floating-point operation
    /// sequence). Throughput only. Absent (the shipped state on every yaml) =
    /// dark = the serial nest, byte-identical. Observable:
    /// `[bilinear-kernel] stage=… armed=1 form=rowpar calls=… rows=… secs=…`
    /// plus `[collector-kernels] armed …`.
    pub(crate) bilinear_rowpar_kernel: Option<bool>,

    /// #vit-collector-throughput P3: run a sequence Linear's certified f64
    /// `A·W` as ONE product over all token positions instead of one per
    /// position (vit: 17 positions; the stacked product crosses the sound f64
    /// engine gate that each per-position product sits under).
    ///
    /// SOUND: bit-identical on the CPU reducer; on the sound f64 engine the
    /// same summation-order-independent `γ_n·S` certificate covers it (not
    /// bit-identical). Absent (the shipped state on every yaml) = dark = the
    /// per-position loop. Observable: `[linear-pos-stack] stage=… armed=1
    /// calls=… positions=… rows=… route=cpu|engine …`.
    pub(crate) linear_position_stack: Option<bool>,

    /// #vit-collector-throughput P4: relax `1/x` on `[l, +inf)` (`l > 0`
    /// finite) by `0 <= y <= next_up(1/l)` instead of refusing, and refuse
    /// every remaining out-of-domain Reciprocal pre-activation with
    /// `UnsupportedOp` (a per-target IBP fallback in every collector) instead
    /// of `InvalidSpec`, which aborted vit_2023's pgd fixed-index-softmax rows
    /// (`unknown@9.71s`).
    ///
    /// SOUND: the relaxation encloses `1/x` on the whole half-line (checked in
    /// exact rationals), and the error class only reroutes a refusal to the
    /// certified IBP box. Absent (the shipped state on every yaml) = dark =
    /// the historical guard and error class. Observable:
    /// `[reciprocal-unbounded] stage=… armed=1 calls=… elems=…
    /// route=const-upper(lower=zero) refusals_unsupported_op=…`.
    pub(crate) reciprocal_unbounded_upper: Option<bool>,

    /// #vit-batched-bilinear (ENGINE_PROGRAM V1): run the var×var `MatMul` /
    /// `BilinearCrown` McCormick backward (midpoint and α arms) as the batched,
    /// GEMM-shaped kernel: every element's McCormick planes computed ONCE per
    /// call into a table, rows in parallel chunks and tiles of 8, each weight a
    /// sign-selected rank-1 update of its `k` contiguous coefficient cells,
    /// and the tile's f64 constant chains interleaved. Tried before
    /// `bilinear_rowpar_kernel` when both are set.
    ///
    /// WHY: the serial nest is `rows × batch × m × n × k` scalar iterations on
    /// one thread (13,872 per row per attention crossing on vit_2023's ibp
    /// model), the collector completes 1 of 19 CROWN targets inside its
    /// shares, and S7 asks for 19/19 in <= 5 s (scratchpad/vbs/diag6/vit.md
    /// §2, ENGINE_PROGRAM §2 V).
    ///
    /// SOUND by construction: BIT-IDENTICAL to the serial nest (every
    /// coefficient, certified-S and constant accumulator receives the nest's
    /// operations in the nest's order; the certified error accounting is the
    /// nest's). Throughput only. Absent (the shipped state on every yaml) =
    /// dark = the historical choice, byte-identical. Observable:
    /// `[bilinear-kernel] stage=… armed=1 form=gemm calls=… rows=… secs=…
    /// table_secs=…` plus `[collector-kernels] armed … bilinear_gemm=1`.
    pub(crate) bilinear_gemm_kernel: Option<bool>,

    /// Explicit base cap (seconds) on CROWN-IBP collectors' per-node time
    /// budget (#cgan-bn11-budget). Unset selects an adaptive cap from
    /// the remaining collection budget (25%, clamped to 12–600 seconds).
    /// Explicit caps are dimension-scaled above 28,800 rows.
    pub(crate) crown_ibp_per_node_cap_secs: Option<f64>,

    /// #target-cost-weight: weight each CROWN-IBP collection target by the
    /// MACs of the backward walk it will run (`rows x prefix`) instead of by
    /// its objective-row count. Unset/false keeps the historical row weights.
    /// Scheduling only — a declined target publishes exactly what it publishes
    /// today.
    pub(crate) crown_ibp_prefix_cost_weighted: Option<bool>,

    /// #target-cost-weight: cap on the fraction of the remaining CROWN-IBP
    /// collection window any single non-final target may be granted. Unset
    /// applies no cap. A present value outside `(0, 1)` is REJECTED at preset
    /// load (`apply.rs` bails) rather than silently ignored — a yaml that asks
    /// for a cap and gets none is worse than one that fails to load. The
    /// resolver additionally ignores such a value if one ever reaches it by
    /// another route.
    pub(crate) crown_ibp_max_target_share_frac: Option<f64>,

    /// #cgan-reclaim (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md): the typed,
    /// competition-reachable route for the cgan budget-reclaim set. The set
    /// is one gate over five retention sites — cross-mode CROWN-IBP
    /// collection-cache serve, crown-partial expiry retention, sequential
    /// CROWN-IBP expiry salvage, ConvTranspose2d mid-step expiry salvage, and
    /// forward-fallback publication past expiry — deliberately not separable,
    /// because the standing audit finding is that set-mates switch together
    /// and partial fixes measure exactly zero.
    ///
    /// WHY THE KEY EXISTS AT ALL. `vnncomp_scripts/run_instance.sh` exports no
    /// `NY_*`, so `NY_CGAN_RECLAIM` can never arm on a scored run:
    /// the mechanism was built, correct and unreachable, the sixth in that
    /// family after `NY_EPOCH_DEPTH` and `NY_MARGIN_ROW_CLIP_ROWS`. Env still
    /// wins in both directions where present.
    ///
    /// Absent (the shipped state on every config) = dark = byte-identical.
    pub(crate) cgan_reclaim: Option<bool>,

    // ---------------------------------------------------------------------
    // #lever-reachability (2026-08-23). Fifteen keys promoted in one batch
    // from environment-only gates. Every one is DEFAULT OFF (or, where the
    // declaration ships armed, a typed KILL SWITCH that changes nothing while
    // absent) and set by no shipped yaml — pinned by
    // `no_shipped_preset_arms_a_promoted_lever_delivery_key`. The batch is
    // byte-identical on every scored run: promotion makes a lever
    // DELIVERABLE, it does not arm it.
    //
    // The environment name keeps precedence in BOTH directions wherever it is
    // present, so every banked A/B recipe still works unchanged; a present but
    // malformed value is a kill switch that resolves to the declaration
    // default rather than silently revealing the preset.
    // ---------------------------------------------------------------------
    /// Root PHASE-RESIDENT sweep (`NY_ROOT_PHASE_RESIDENT_CROWN`).
    ///
    /// OWNS the root slot when armed, suppressing the comprehensive and
    /// wide-demanded phases entirely, so it decides which intermediate
    /// tightening the whole instance gets. Shrink-only and fail-closed: a
    /// wrong choice costs tightness and can lose a proof, never invent one.
    pub(crate) root_phase_resident_crown: Option<bool>,

    /// One-target wide DEMANDED intermediate CROWN
    /// (`NY_ROOT_WIDE_DEMANDED_INTERM_CROWN`). Reachable only when both
    /// earlier root phases are unarmed; same shrink-only contract.
    pub(crate) root_wide_demanded_interm_crown: Option<bool>,

    /// Comprehensive CPU-parallel root sweep
    /// (`NY_ROOT_CPU_PARALLEL_INTERM_CROWN`). REPLACES the dense-head
    /// tightener rather than adding to it, so it changes which intermediate
    /// bounds are published.
    pub(crate) root_cpu_parallel_interm_crown: Option<bool>,

    /// Per-target row ceiling for the comprehensive sound-GPU root sweep
    /// (`NY_ROOT_COMP_GPU_INTERM_ROWS`). Absent keeps the shipped cap. The
    /// device-class policy may still cap the request below the requested
    /// value, and the backend refuses an over-large one — which is exactly the
    /// signal the #comprehensive-rows-probe program measures.
    pub(crate) root_comprehensive_gpu_interm_rows: Option<u64>,

    /// Local authority slice, in seconds, for that sweep
    /// (`NY_ROOT_COMP_GPU_INTERM_SECS`). Capped in turn by half the remaining
    /// global budget, so large values saturate. Every chunk stays atomic,
    /// deadline-bounded and shrink-only.
    pub(crate) root_comprehensive_gpu_interm_secs: Option<u64>,

    /// Rank the root objective pass at the rows that gate the verdict
    /// (`NY_ROOT_OBJECTIVE_DIRECTED_ROWS`). SHIPS ARMED, so this key is the
    /// typed per-category KILL SWITCH — the A/B control arm that previously
    /// existed only as an environment variable a scored run cannot set.
    pub(crate) root_objective_directed_rows: Option<bool>,

    /// Retain the optional CPU compute-engine presence handle
    /// (`NY_CPU_COMPUTE_ENGINE`). The historical policy is enabled by absence;
    /// `false` is its typed kill switch. A qualified WGPU proof device always
    /// survives independently of this value. No shipped preset sets this key.
    pub(crate) cpu_compute_engine: Option<bool>,

    /// Master arm for the root joint intermediate-alpha transaction
    /// (`NY_ROOT_JOINT_INTERM_ALPHA`). The same resolved value owns both root
    /// entry routes, finite-deadline GPU admission and the CLI's optional CPU
    /// engine suppression, so this key cannot produce a half-enabled run.
    /// Absent keeps the measured research mechanism dark.
    pub(crate) root_joint_interm_alpha: Option<bool>,

    /// Permit root-joint alpha ascent under a finite deadline
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT`). This is a second,
    /// default-dark arm: the selected GPU must also expose the audited bounded
    /// joint-adjoint and sound-fold capability or execution keeps the certified
    /// zero-iteration base fold.
    pub(crate) root_joint_interm_alpha_deadline_ascent: Option<bool>,

    /// Local root-joint wall request in seconds
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_SECS`), default 30 when absent. This is a
    /// request, not extra authority: the graph route caps it by remaining
    /// budget and the scored input-split route caps it by the overall instance
    /// deadline. Standalone research calls without that authority retain the
    /// historical grace behavior.
    pub(crate) root_joint_interm_alpha_secs: Option<u64>,

    /// Joint-alpha ascent iteration ceiling
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_ITERS`), default 100 when absent.
    pub(crate) root_joint_interm_alpha_iters: Option<usize>,

    /// Joint-alpha Adam step (`NY_ROOT_JOINT_INTERM_ALPHA_LR`), default 0.1
    /// when absent. The runtime narrows the governed finite f32-range value at
    /// the unchanged optimizer boundary.
    pub(crate) root_joint_interm_alpha_lr: Option<f64>,

    /// Per-target crossing-row ceiling
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL`), default 512 when absent.
    /// Zero makes the bounded tightener a sound no-op; the execution receipt
    /// distinguishes a configured ceiling from one that actually truncated a
    /// target's crossing-row selection.
    pub(crate) root_joint_interm_alpha_max_sel: Option<usize>,

    /// Permit root-joint extraction to stop at a finite frozen interior box
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP`). Default OFF. The receipt
    /// credits the arm only when an accepted target actually used a stop box.
    pub(crate) root_joint_interm_alpha_frozen_stop: Option<bool>,

    /// Admit a certified pure-chain suffix with no residual block
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN`). Default OFF. Frozen-stop can
    /// independently imply this admission, so attributed execution requires a
    /// target for which this specific arm was necessary.
    pub(crate) root_joint_interm_alpha_pure_chain: Option<bool>,

    /// Admit the certified BatchNorm-to-diagonal-Conv extraction used by the
    /// root-joint lane (`NY_BN_GPU_EXTRACT`). Default OFF. The July cGAN
    /// research recipe required this together with frozen-stop, but no shipped
    /// preset arms either treatment.
    pub(crate) bn_gpu_extract: Option<bool>,

    /// Instance-specific comma-separated ReLU selector
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_LAYERS`). Empty or `all` keeps automatic
    /// selection. This key is deliverable for reproducible targeted research
    /// but deliberately has no global search axis because node names are not
    /// portable across VNN-COMP graphs.
    pub(crate) root_joint_interm_alpha_layers: Option<String>,

    /// Minimum real remaining seconds for graph-route admission
    /// (`NY_ROOT_JOINT_MIN_REMAINING_SECS`), default 240 when absent. Explicit
    /// zero preserves the research recipe that disables the floor.
    pub(crate) root_joint_min_remaining_secs: Option<u64>,

    /// Dimension ceiling for the root joint intermediate-alpha selector
    /// (`NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM`). Absent keeps each selector's own
    /// contextual default (2,048 legacy / 32,768 demand-ranked), which is why
    /// the declaration carries no number of its own. Selection-only: a target
    /// that is selected but not tightened keeps its reference bound bit-for-bit.
    pub(crate) root_joint_interm_alpha_max_dim: Option<u64>,

    /// Host mean-lA reconstruct for the per-domain clip lane
    /// (`NY_CLIP_HOST_MEAN_LA`). The 2026-08-12 pilot behind this lane was run
    /// entirely through its env half, i.e. through a route no scored run can
    /// take, so it could not have been qualified even had it come back positive.
    pub(crate) clip_host_mean_la: Option<bool>,

    /// Certified batched intermediate clip (`NY_CLIP_INTERM_CERTIFIED`).
    /// Certified end-to-end (docs/CLIP_APPLY_ENCLOSURE_PROOF_DESIGN_2026-08-12.md)
    /// and shrink-only, but it feeds verdict-authoritative CROWN bounds, so it
    /// is exactly the class of lever that must be preset-deliverable.
    pub(crate) clip_interm_certified: Option<bool>,

    /// Give each multi-objective GPU chunk its own deadline rather than one
    /// deadline for the batch (`NY_MO_GPU_CHUNK_DEADLINE`), letting the wide
    /// batch width rise under an authoritative deadline. Default OFF.
    pub(crate) mo_gpu_chunk_deadline: Option<bool>,

    /// Replace stored ancestor graph-domain alpha with the sound current-box
    /// 0/1 heuristic in multi-objective propagation
    /// (`NY_MO_ALPHA_HEURISTIC`). The retained trace moved a binding deep-child
    /// lower bound, but did not establish a row conversion or category-wide
    /// non-regression. This key is therefore deliverable for sealed searches
    /// and remains absent from every shipped preset.
    pub(crate) mo_alpha_heuristic: Option<bool>,

    /// Use the measured two-sided SPSA gradient in the multi-objective graph
    /// beta optimizer (`NY_MO_BETA_SPSA`). The declaration ships armed because
    /// the analytical rollback differentiates the wrong quantity; this typed
    /// key is therefore both a per-category arm and a kill switch. No shipped
    /// preset sets it, so omission preserves the measured default-on policy.
    pub(crate) mo_beta_spsa: Option<bool>,

    /// Jointly step stored graph alpha with beta through the same SPSA samples
    /// (`NY_MO_ALPHA_STEP`). The first isolated measurement regressed the probe
    /// bound, so this is a sealed-search route only and remains absent from
    /// every shipped preset. It is inert when beta SPSA is off, and the
    /// current-box alpha heuristic supersedes its stored alpha when armed.
    pub(crate) mo_alpha_step: Option<bool>,

    /// Share of the kFSB branching budget spent on candidate simulation
    /// (`NY_KFSB_SIM_SHARE`, closed `[0, 1]`, declaration default 0.35; `0`
    /// removes the cap). Sound by construction — stopping early ranks over
    /// fewer candidates, so the wave may commit a WORSE split, but every
    /// committed child is still evaluated by the ordinary sound backward.
    pub(crate) kfsb_sim_share: Option<f64>,

    /// Publish the Graph-MIP leaf oracle's SAT witness as the run's verdict
    /// candidate (`NY_GRAPH_MIP_LEAF_SAT`). SHIPS ARMED, so this key is the
    /// typed kill switch back to the pre-change ADVISORY behaviour. The witness
    /// is graph-forward-confirmed against every objective row before it is
    /// published either way.
    pub(crate) graph_mip_leaf_sat: Option<bool>,

    /// GPU-evaluated replay for the true-gradient lane
    /// (`NY_TRUE_GRAD_GPU_REPLAY`). SHIPS ARMED where a caller supplies
    /// operands, so this key is the typed kill switch to the CPU path.
    /// Gradients only steer alpha in [0, 1] — any value there is a valid lower
    /// relaxation slope — and no bound from this lane is ever published.
    pub(crate) true_grad_gpu_replay: Option<bool>,

    /// SafeNLP full-cycle master arm (`NY_SAFENLP_MIP_CYCLE`,
    /// docs/SAFENLP_CYCLE_2026-08-20.md): conditional MIP handoff with the
    /// historical BaB floor kept, serial one-model solve, and AY's
    /// verdict-preserving margin reframe, as one gate so the composed cycle
    /// cannot be measured split.
    ///
    /// This one changes WHICH prover answers, so it deserves its own soundness
    /// sentence: both provers are verdict authorities in their own right, and
    /// the reframe banks UNSAT only on a strict bound crossing with
    /// original-model proof replay. Arming it can lose a row to the schedule;
    /// it cannot certify a false one.
    pub(crate) safenlp_mip_cycle: Option<bool>,

    /// #hull-root / #hull-bab (wave-6 C4, `scratchpad/vbs/diag6/safenlp.md`
    /// Plans 1-2): the certified single-hidden-layer per-neuron hull bound,
    /// run in the sequential relational lane after the upfront attack and
    /// before the same-LHS reduction. `off` | `root` | `bab`. Absent (every
    /// shipped yaml) = `off` = byte-identical. `bab` adds a deadline-bounded
    /// best-first ReLU-split tree over the concave neurons (60% of the
    /// remaining wall, capped at 60 s; undecided rows keep the rest).
    ///
    /// SOUND for any value: the verdict authority is a closed-form,
    /// outward-rounded affine minorant valid for ARBITRARY multipliers (and,
    /// per tree node, ARBITRARY nonnegative split multipliers); the AY float
    /// LP only proposes them. It can turn a timeout into `unsat` and can
    /// never publish `sat` or flip a verdict. Observables:
    /// `[hull-root] armed engaged=<0|1> verdict=... cert_lb=...` (`root`) /
    /// `[hull-bab] armed engaged=<0|1> verdict=... nodes=... worst=...`
    /// (`bab`) on every armed run (stderr) plus a `hull_root` / `hull_bab`
    /// flight note.
    pub(crate) single_hidden_hull: Option<ny_propagate::SingleHiddenHullMode>,

    /// #graph-lane-search (ENGINE_PROGRAM §2 M7): which UNTRUSTED batched α/β
    /// search the graph β-CROWN multi-objective lane consults
    /// (`BetaCrownConfig::graph_search`): `off` (absent, every shipped yaml —
    /// byte-identical) or `cpu_f32` (the injected `ny_bbe::CpuF32Backend`
    /// over the lane's self-checked `GraphSearchNet` copy, at the root and on
    /// each shared-executor wave's open children).
    ///
    /// SOUND for any value: every hinted slope / multiplier is clamped into its
    /// valid set at the lane's ingest gate and certified ONCE by the lane's own
    /// evaluator (on the CPU); a row's LOWER bound is raised only by a strictly
    /// better certified candidate; a disjoint certified root pair downgrades the
    /// root to `unknown`. The device never certifies. Observables:
    /// `[bbe] phase=graph_bab stage=root …` / `stage=wave …` (stderr).
    pub(crate) graph_search: Option<ny_propagate::GraphLaneSearch>,

    /// #la-warm-frontier: the FRONTIER-COMPLETE lA warm start on DAG graphs
    /// (`BetaCrownConfig::la_warm_frontier`). Armed, every capturing backward
    /// pass also records, at each node a split can name, the rest of its
    /// backward state at that instant (pending skip-path blocks and the
    /// network-input channel, error-folded), and a child whose single-block
    /// seed the #warm-dag guard refuses resumes from its branch node's block
    /// plus that frontier instead of the output. Absent (every shipped yaml) =
    /// dark = the guard-only tree, byte-identical.
    ///
    /// SOUND for either value: the resumed bound bounds the objective over the
    /// child's region (child region ⊆ capturing region; proof in
    /// `ny-propagate/.../propagation/warm_frontier.rs`); a missing or invalid
    /// frontier takes the cold pass. Throughput only (fewer layers per child
    /// on residual nets). Observable: `[la-warm-frontier] lane=… armed=1
    /// captured=… resumed=…` once per armed multi-objective / DomainList run.
    pub(crate) la_warm_frontier: Option<bool>,

    /// #kfsb-single-shot: score graph kFSB candidates the αβ-CROWN way
    /// (`BetaCrownConfig::kfsb_single_shot`) — an objective-seeded,
    /// stop-early BaBSR prescore and ONE constrained CROWN pass per candidate
    /// child with inherited α/β — instead of the identity-seeded prescore over
    /// the whole graph output and the full β(+α) optimizer per candidate child.
    /// Absent (every shipped yaml) = dark = the historical scorer,
    /// byte-identical.
    ///
    /// SOUND for either value: the selector only chooses which ReLU to
    /// partition; committed children are bounded by the unchanged pipeline.
    /// Throughput only (cheaper branching decisions on graph lanes).
    /// Observable: `[kfsb-single-shot] armed=1 …` once per armed decision.
    pub(crate) kfsb_single_shot: Option<bool>,

    /// #relusplitter-lp-refine (wave-6 C7, `scratchpad/vbs/diag10/
    /// build_relusplitter-lp-refine.md`): rigorous LP-refined intermediate
    /// bounds at the multi-objective graph BaB root, `{layers, budget_frac}`
    /// (defaults 3 and 0.3; unknown sub-keys are refused). Presence arms it;
    /// absent (every shipped yaml) attaches no refiner = byte-identical.
    ///
    /// SOUND for any value: every published bound is a rigorous (Neumaier–
    /// Shcherbina or exact-rational) bound on a triangle LP whose feasible set
    /// contains the true ReLU graph, rounded outward and INTERSECTED into the
    /// stored boxes. It can turn a timeout into `unsat` and can never publish
    /// `sat` or flip a verdict. Observables: `[mip-refine] armed ...` at attach
    /// and `[mip-refine] root ...` whenever the root runs it.
    pub(crate) mip_refine_intermediates: Option<ny_propagate::MipRefineIntermediates>,

    /// #vgg-gpu-crown (`scratchpad/vbs/diag12/build_vgg-gpu-crown.md`): a
    /// certified patches-mode CROWN root pass on the sound GPU GEMM engine,
    /// run before the graph BaB — CROWN intermediates for every ReLU layer
    /// (intersected shrink-only into certified one-step screen boxes), then
    /// the spec rows. `{chunk_rows, max_mb, headroom_frac, gpu_tile_mb,
    /// precision: f64|f32, depth, screen, budget_frac, max_secs,
    /// min_perturbed}`, every sub-key defaulted, unknown sub-keys refused.
    /// Presence arms it; absent (every shipped yaml) runs nothing.
    ///
    /// SOUND for any value: products carry the a-posteriori `gamma * |A||B|`
    /// certificate charged against the multiplied value's box, relaxations
    /// are directed-rounded, the result is rounded toward `-inf`, and a GPU
    /// refusal is a bit-identical CPU product. It can only certify every
    /// objective (`unknown -> unsat`) or change nothing. Observable:
    /// `[gpu-patches-crown] armed=1 layers= gpu_passes= cpu_fallback= secs=
    /// peak_mb= ...` on every armed run.
    pub(crate) gpu_patches_crown: Option<ny_propagate::GpuPatchesCrownConfig>,

    /// Context-grouped disjunctive refinement covers on nn4sys
    /// (`NY_NN4SYS_CTX_GROUP`): one shared tree over the hull of bit-identical
    /// clause boxes instead of one tree per box. The cover and disjointness
    /// proofs are FAIL-CLOSED — any doubt keeps the obligation — so it changes
    /// which boxes get bounded, and therefore coverage, without weakening what
    /// a discharged obligation means.
    pub(crate) nn4sys_ctx_group: Option<bool>,

    /// #nn4sys-1d-cover MASTER (`NY_NN4SYS_1D_COMPLETE_COVER`): run the exact
    /// one-axis f64 complete cover as a prepass in front of the
    /// box-refinement clause screen.
    ///
    /// Soundness: the prepass is ALL-OR-NOTHING and fail-closed. It publishes
    /// only obligations it discharged itself, through the same certified
    /// outward-rounded f64 machinery the screen uses; one undecided eligible
    /// leaf, propagation error, unsupported schema, cap or deadline discards
    /// the entire pass and the unchanged screen decides every clause. A
    /// `false` seed always means unproven and proceeds downstream unchanged.
    pub(crate) nn4sys_1d_cover: Option<bool>,

    /// #nn4sys-1d-cover DEPTH (`NY_NN4SYS_1D_COVER_DEPTH`), default 8.
    ///
    /// Shipped beside the master deliberately. Because the pass is
    /// all-or-nothing, a depth too small for the property does not degrade
    /// the answer — it DISCARDS the prepass, so the master alone would be a
    /// reachable, correct, inert mechanism. The reader keeps its own
    /// `HARD_MAX_DEPTH = 24` clamp, so an out-of-range value falls back to
    /// the shipped default rather than widening any bound.
    pub(crate) nn4sys_1d_cover_depth: Option<u64>,

    /// #nn4sys-1d-cover SLICE (`NY_NN4SYS_1D_COVER_SLICE`), default 0.25,
    /// admissible `[0.02, 0.5]`: the fraction of the caller's REMAINING
    /// deadline the cover may consume, and the selector for the bounded
    /// partial-publication mode.
    ///
    /// A fraction rather than the legacy millisecond constant because nn4sys
    /// budgets span 20 s to 800 s inside one benchmark. It can only move
    /// proof time between the prepass and the screen; it cannot change what
    /// either lane certifies.
    pub(crate) nn4sys_1d_cover_slice: Option<f64>,

    // ---------------------------------------------------------------------
    // #lever-reachability batch 2 (2026-08-23). Nine more, same contract as
    // the block above: DEFAULT OFF (or a typed kill switch where the
    // declaration ships armed), set by no shipped yaml, pinned by
    // `no_shipped_preset_arms_a_promoted_lever_delivery_key`.
    //
    // Six of them are read in `ny-gpu`, whose cells live in
    // `ny_propagate::preset_gates` because `ny-gpu` depends on `ny-propagate`
    // — one place the bridge writes, one place a reviewer looks.
    // ---------------------------------------------------------------------
    /// The wide ResNet CROWN dispatch (`NY_BAB_RESNET_WIDE`). SHIPS ARMED, so
    /// this key is the typed per-category KILL SWITCH — the `0` arm, which was
    /// expressible only through an environment variable the scored harness does
    /// not export.
    pub(crate) bab_resnet_wide: Option<bool>,

    /// Split a HETEROGENEOUS wide batch into maximal homogeneous runs and fold
    /// it run-by-run instead of abandoning the wave to the serial path
    /// (`NY_BAB_RESNET_WIDE_SUBGROUP`). It changes WHICH kernel produces a
    /// verdict-bearing bound, so arming it stays a measured-session decision;
    /// this key makes that decision expressible at all.
    pub(crate) bab_resnet_wide_subgroup: Option<bool>,

    /// Row count above which a sound-GPU sweep is classified LARGE
    /// (`NY_SWEEP_CLASS_ROWS`). Absent keeps the shipped class policy. The
    /// backend preflight still computes exact simultaneous liveness and refuses
    /// anything it cannot honour.
    pub(crate) sweep_class_rows: Option<u64>,

    /// The memory half of the same classification, in whole MiB
    /// (`NY_SWEEP_CLASS_MIB`). Rows and MiB can disagree, and which one binds
    /// is the thing being measured. NOTE the GB10 hazard recorded at the read
    /// site: the 121 GiB is shared with the host and an earlier
    /// over-allocation caused a global OOM, so this is an explicit opt-in.
    pub(crate) sweep_class_mib: Option<u64>,

    /// Discharge the sound GPU-resident ResNet walk's accumulated coefficient
    /// error PER SEGMENT against the frontier magnitudes instead of carrying it
    /// forward as coefficient radii (`NY_RESNET_ERR_CONCRETIZE`). It is on the
    /// CERTIFIED lane and feeds the main-bound auto-fallback, so it changes
    /// which bound is published — not a plumbing switch despite looking like
    /// one.
    pub(crate) resnet_err_concretize: Option<bool>,

    /// Per-RELU rather than per-segment error concretization on the same
    /// certified walk (`NY_RESNET_ERR_CONCRETIZE_FINE`); a refinement of the
    /// key above and inert unless that one is armed.
    pub(crate) resnet_err_concretize_fine: Option<bool>,

    /// Cost-weighted vs count-based per-candidate grants for the sequential
    /// CROWN-IBP collection budget (`NY_SEQ_WEIGHTED_BUDGET`).
    ///
    /// SHIPS ARMED, and armed by ABSENCE on a legacy `!= "0"` opt-out parser,
    /// so this key is the typed count-based CONTROL ARM. A candidate that loses
    /// its grant degrades to IBP and that degraded intermediate is published
    /// into every downstream bound, so the choice moves bounds in both
    /// directions — which is exactly why the control arm needed a route a
    /// scored run can take.
    pub(crate) seq_weighted_budget: Option<bool>,

    /// One stacked backward walk in place of per-target walks in the cgan
    /// collector (`NY_CGAN_STACKED_BACKWARD`). Default dark; the fallback
    /// computes the same bounds.
    pub(crate) cgan_stacked_backward: Option<bool>,

    /// Host-memory admission budget in MiB for the stacked cGAN walk
    /// (`NY_CGAN_STACKED_BUDGET_MB`). Absence preserves the host-adaptive
    /// budget; explicit zero remains a real control arm.
    pub(crate) cgan_stacked_budget_mb: Option<u64>,

    /// Kernel selector for finite-deadline certified Conv2d IBP: `1`
    /// (parallel blocked), `seq`, or `0` (scalar oracle)
    /// (`NY_CONV_IBP_BLOCKED`).
    pub(crate) conv_ibp_blocked: Option<String>,

    /// Decide native Patches finite authority by deadline expiry rather than
    /// deadline presence (`NY_PATCHES_FINITE_EXPIRY`). Ships armed; this is
    /// also the typed kill-switch route.
    pub(crate) patches_finite_expiry: Option<bool>,

    /// Disable consultation of passive graph-walk timing records
    /// (`NY_NO_WALK_RECORD_ADMISSION`).
    pub(crate) no_walk_record_admission: Option<bool>,

    /// Pollable chunked-faer bounded Linear CROWN (`NY_POLLABLE_FAER`). The
    /// environment retains its historical truthy parser; this typed route is
    /// canonical Boolean delivery for scored search.
    pub(crate) pollable_faer: Option<bool>,

    /// Near-miss survivor alpha polish on the grouped-disjunctive eager lane
    /// (`NY_CGAN_POLISH`): children inside the knife-edge band get one
    /// certified warm-alpha re-bound with real SPSA steps, FLOOR-MERGED so
    /// bounds only tighten.
    pub(crate) cgan_polish: Option<bool>,

    /// Switch the two per-node partial CROWN sites from deadline PRESENCE to
    /// deadline EXPIRY (`NY_CROWN_PARTIAL_FINITE_EXPIRY`): commit what the walk
    /// already holds instead of declining the moment a deadline merely EXISTS.
    ///
    /// Worth delivering precisely because the standing audit predicts a NULL
    /// unless the whole root-cause-D set switches together — a zero measured
    /// through this key is evidence about the split, not about the fix, and
    /// until now there was no way to measure it on a scored run at all. Both
    /// arms publish sound enclosures; the walk's existing entry/exit expiry
    /// checks still refuse any late publication.
    pub(crate) crown_partial_finite_expiry: Option<bool>,

    /// #dense-sparse-seed: minimum IBP-stable fraction at which a Dense-seeded
    /// sequential CROWN-IBP partial walk whose successor layer is a ReLU seeds
    /// only its IBP-unstable rows (`NY_CROWN_PARTIAL_DENSE_MIN_STABLE`,
    /// admissible [0.0, 1.0], default 0.9 = the historical literal). `0.1` is
    /// alpha-beta-CROWN's `minimum_sparsity: 0.9` rule. Untracked stable
    /// neurons publish the certified IBP box, so every value stays an
    /// enclosure; walks without a ReLU successor always keep 0.9.
    pub(crate) crown_partial_dense_min_stable: Option<f64>,

    /// Hard wall-clock cap (seconds) on the root alpha-CROWN warmup.
    /// See `BetaCrownConfig::root_alpha_cap_secs` for the measurement.
    #[serde(default)]
    pub(crate) root_alpha_cap_secs: Option<f64>,

    /// Retain a completed DAG-alpha artifact when the root warmup reaches its
    /// local cap, then continue through certified multi-objective root
    /// evaluation under the outer deadline. Omission is default-off. An absent
    /// env value inherits this setting, exact `NY_ROOT_ALPHA_PHASE_CHECKPOINT=1`
    /// arms, and every other present value is a fail-closed kill switch.
    pub(crate) root_alpha_phase_checkpoint: Option<bool>,

    /// Additional exact full-`C` root margin-Adam iterations. Absent/zero is
    /// default-dark; values above the solver's hard cap of eight are rejected
    /// during preset application.
    pub(crate) atomic_root_c_margin_iterations: Option<usize>,

    /// In-iteration verified-domain pruning (alpha-beta-CROWN
    /// `pruning_in_iteration`). Parsed for reference-config compatibility but
    /// NOT implemented — no engine code reads it; `apply_preset` warns and
    /// ignores it.
    pub(crate) pruning_in_iteration: Option<bool>,

    /// Enable intermediate bound transfer.
    pub(crate) interm_transfer: Option<bool>,

    /// Allow root intermediate CROWN passes to fall back to the sound CUDA
    /// engine factory when no usable local engine was supplied.
    pub(crate) root_interm_cuda_factory: Option<bool>,

    /// Allow post-root multi-objective graph BaB to reuse an already-materialized
    /// engine that advertises deadline-safe support for the complete handoff
    /// surface, sound GPU CROWN, and cooperative cancellation when no local
    /// engine was supplied. Omission preserves the default-dark typed policy.
    pub(crate) mo_cuda_factory_engine_handoff: Option<bool>,

    /// Allow the post-root multi-objective shared executor to use a local,
    /// deadline-aware CPU GEMM facade while the already-materialized CUDA-wide
    /// backend remains confined to its call-local bounded β-CROWN API.
    /// Omission preserves the default-dark typed policy.
    pub(crate) mo_cuda_bounded_shared_executor: Option<bool>,

    /// #vit-bank-gate: arm the downgrade-only containment fence on the
    /// multi-objective graph BaB (`BetaCrownConfig::mo_containment_fence`).
    /// Omission keeps it off; no shipped preset arms it. Scoped: only
    /// `vit_2023` may ever set it (`mo_containment_fence_is_scoped_to_vit_2023`).
    pub(crate) mo_containment_fence: Option<bool>,

    /// #vit-margin-alpha-complex (P8): adjudicate a ROOT-terminal
    /// multi-objective `Verified` against an ONNX Runtime forward of the
    /// original model at the fence's samples
    /// (`BetaCrownConfig::mo_root_ort_adjudication`). Inert unless
    /// `mo_containment_fence` is also armed; downgrade-only; fails closed when
    /// armed and no sample could be adjudicated. Observable: the
    /// `[mo-containment-fence] root-verified … ort=…` fields. Omission keeps it
    /// off; only `vit_2023` may set it; no shipped preset does.
    pub(crate) mo_root_ort_adjudication: Option<bool>,

    /// #vit-v3-bankable: the multi-objective COVER CERTIFICATION policy of
    /// the disjunctive graph multi-objective BaB
    /// (`BetaCrownConfig::mo_cover_certify`, governed by `NY_MO_COVER_CERTIFY`):
    /// `off` (absent, every shipped yaml — byte-identical), `ledger` (per-row
    /// completeness ledger: a cover gap withdraws `Verified`, a proved cover
    /// publishes from an `UnsatToken`) or `replay` (the ledger plus a
    /// certify-once replay of every witness leaf with no reused lane state;
    /// `Verified` stands only when both pass). Downgrade-only. Observables:
    /// `[mo-cover] armed=1 mode=…`, `[cover-gate] graph-frame lane=graph_mo …`,
    /// `[mo-cover-replay] … outcome=…` (stderr).
    pub(crate) mo_cover_certify: Option<ny_propagate::MoCoverCertify>,

    /// Enable the one-time, structurally selected dense-head CROWN intermediate
    /// shrink-intersect at the graph root (#cifar-head-crown).
    pub(crate) root_crown_interm_dense_head: Option<bool>,

    /// Wall-clock cap in seconds for `root_crown_interm_dense_head`.
    pub(crate) root_crown_interm_max_secs: Option<u64>,

    /// Maximum selected dense-head pre-activation width.
    pub(crate) root_crown_interm_max_dim: Option<usize>,

    /// Run the comprehensive all-target sound-GPU root intermediate sweep.
    ///
    /// DELIVERY: the scored entry point exports no `NY_*` variables, so
    /// the env lever this was measured through is dead in competition. This typed
    /// key is how the result reaches a scored run
    /// (`crates/ny-cli/tests/measured_gate_delivery.rs`).
    pub(crate) root_comprehensive_gpu_interm: Option<bool>,

    /// `#bab-floor`: BaB's share of the root window, subtracted before any root
    /// phase sizes itself.
    ///
    /// DELIVERY: the scored entry point exports no `NY_*` variables, so
    /// the env levers these were measured through are dead in competition.
    /// These three typed keys are how a search result reaches a scored run.
    pub(crate) root_bab_reserve_frac: Option<f64>,

    /// `#bab-floor`: the root objective pass's share of that window.
    pub(crate) root_spec_frac: Option<f64>,

    /// `#bab-floor`: the bootstrap ascent's share of that window.
    pub(crate) root_alpha_frac: Option<f64>,

    /// Disjoint row windows the comprehensive sweep may accumulate. `1` is
    /// byte-identical to the historical single sweep; higher values trade wall
    /// clock for root coverage at constant peak device memory.
    pub(crate) root_comprehensive_gpu_interm_chunks: Option<usize>,

    /// Enable the one-time structurally selected sparse crossing-row CROWN fold
    /// for non-dense ReLU pre-activations at the graph root.
    pub(crate) root_sparse_interm_crown: Option<bool>,

    /// Wall-clock cap in seconds for `root_sparse_interm_crown`.
    pub(crate) root_sparse_interm_crown_max_secs: Option<u64>,

    /// Maximum flattened target width for the sparse pass.
    pub(crate) root_sparse_interm_crown_max_dim: Option<usize>,

    /// Maximum crossing rows seeded per sparse target.
    pub(crate) root_sparse_interm_crown_max_rows: Option<usize>,

    /// Maximum sparse targets processed deepest-first.
    pub(crate) root_sparse_interm_crown_max_targets: Option<usize>,

    /// Enable the β/α-ascent graft on the multi-objective dense-spec lane
    /// (#mo-beta-graft): the wide GPU segment-lane ascent optimizes the split
    /// β/α multipliers and the tight dense-spec primitive evaluates with them
    /// folded in (elementwise-tightest composition). Env `NY_MO_BETA_GRAFT`
    /// overrides in both directions.
    pub(crate) beta_graft: Option<bool>,

    /// Admit pure Linear/ReLU chains (FC networks) to the wide GPU stacker
    /// routing (#chain-wide): lifts the conv-only lane gates (the
    /// multi-objective GPU single-pass admission and the wide entry's
    /// `EntryGraphNotConv` decline) so the `beta_graft` composition can reach
    /// the wide seam on FC graphs. Arm together with `beta_graft`: on FC the
    /// wide-chain bound is only ever composed elementwise-tightest with the
    /// dense bound, never a replacement (its base relaxation is measured
    /// 2.4-2.9x looser than the dense backward).
    pub(crate) chain_wide: Option<bool>,

    /// Contain an engine POLICY refusal (`UnsupportedConfiguration` — the
    /// WGPU proof-adapter GEMM quarantine) at the Conv2d Patches compose GEMM
    /// seam by treating the refusing engine as an absent accelerator, so the
    /// walk keeps its byte-identical CPU scatter IN PATCHES instead of dying
    /// into a dense retry that blows the memory budget and leaves the target
    /// IBP-grade (`NY_PATCHES_ENGINE_REFUSAL_CONTAINMENT`,
    /// #patches-engine-refusal-containment,
    /// reports/measured-2026/cct_bound_gap_2026-08-31). Default OFF; genuine
    /// engine failures always propagate. CONTAINMENT, not finiteness: no bound
    /// value changes on any path that works today.
    pub(crate) patches_engine_refusal_containment: Option<bool>,

    /// Materialize the registered process-global fast-f32 GEMM factory on the
    /// vnncomp route for ALL builds, not only CUDA
    /// (`NY_FAST_F32_PREWARM`, #fast-f32-prewarm-all-hosts, same report).
    /// Finite-deadline consumers deliberately fail closed on a cold slot, so
    /// without this every non-CUDA host runs the whole instance with
    /// `is_preinitialized()` false and the Patches compose's GEMM of last
    /// resort is the quarantined proof adapter. Default OFF; scheduling-only
    /// (an engine materialization, no bound math).
    pub(crate) fast_f32_prewarm: Option<bool>,

    /// Branching configuration.
    #[serde(default)]
    pub(crate) branching: BranchingPreset,

    /// α-CROWN configuration overrides.
    /// Supports both "alpha_crown" and "alpha-crown" naming.
    #[serde(default, alias = "alpha-crown")]
    pub(crate) alpha_crown: AlphaCrownPreset,

    /// β-CROWN configuration overrides.
    /// Supports both "beta_crown" and "beta-crown" naming.
    #[serde(default, alias = "beta-crown")]
    pub(crate) beta_crown: BetaCrownPreset,

    /// Reject the easy-to-miss sibling spelling of the per-disjunct α knob.
    ///
    /// The implemented alpha-beta-CROWN-compatible location is
    /// `bab.beta_crown.optimize_disjuncts_separately`. The global unknown-field
    /// policy now rejects every other typo too; this declared trap remains to
    /// give the historically common misplaced key an actionable location error.
    #[serde(
        default,
        rename = "optimize_disjuncts_separately",
        alias = "optimize-disjuncts-separately",
        deserialize_with = "reject_misplaced_optimize_disjuncts_separately",
        skip_serializing
    )]
    #[allow(dead_code)]
    pub(crate) rejected_misplaced_optimize_disjuncts_separately: Option<bool>,

    /// GCP-CROWN cutting planes configuration.
    #[serde(default)]
    pub(crate) cuts: CutsPreset,

    /// INVPROP configuration (output constraint propagation).
    #[serde(default)]
    pub(crate) invprop: InvpropPreset,

    /// Clip-and-verify configuration.
    #[serde(default)]
    pub(crate) clip: ClipPreset,

    /// Phase-level time budget overrides.
    /// Only explicitly-set fields override `PhaseBudgetConfig` defaults.
    #[serde(default)]
    pub(crate) phase_budget: PhaseBudgetPreset,
}

/// Phase-level time budget preset overrides.
///
/// Each field is `Option` so only explicitly-set values override the
/// `PhaseBudgetConfig` defaults. This follows the same pattern as other
/// preset structs (e.g., `ClipPreset`, `CutsPreset`).
///
/// Source fractions and their defaults are documented in
/// `ny_propagate::PhaseBudgetConfig`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhaseBudgetPreset {
    /// Fraction of the BaB budget for the iterative root alpha-CROWN warmup
    /// (the foundational IBP/CROWN-IBP node-bounds sweep is never capped).
    /// Default: 0.20. Recommended competitive: 0.15; set 1.0 to uncap.
    pub(crate) initial_bounds_fraction: Option<f32>,

    /// Fraction of total timeout for upfront PGD attack.
    /// Default: 0.20.
    pub(crate) upfront_pgd_fraction: Option<f32>,

    /// Fraction of total timeout for reduced verification (sequential path).
    /// Default: 0.40.
    pub(crate) reduced_verification_fraction: Option<f32>,

    /// Fraction of total timeout for disjunctive global PGD.
    /// Default: 0.50.
    pub(crate) disjunctive_pgd_fraction: Option<f32>,

    /// Fraction of total timeout for disjunctive CROWN/alpha precheck.
    /// Default: 0.20.
    pub(crate) disjunctive_precheck_fraction: Option<f32>,

    /// Minimum fraction of total timeout guaranteed for MIP fallback.
    /// Default: 0.25.
    pub(crate) mip_min_fraction: Option<f32>,

    /// Minimum MIP timeout in seconds (floor clamp).
    /// Default: 5.
    pub(crate) mip_min_secs: Option<u64>,

    /// Maximum MIP timeout in seconds (ceiling clamp).
    /// Default: 30.
    pub(crate) mip_max_secs: Option<u64>,

    /// Fraction of total timeout reserved for the engine's post-BaB PGD attack
    /// (BaB stops at `timeout * (1 - fraction)` so the fallback attack gets the
    /// rest). Default: 0.10.
    pub(crate) post_bab_pgd_fraction: Option<f32>,

    /// #reserve-demand — size the post-BaB reservation from the lane that
    /// actually follows BaB instead of always subtracting
    /// `post_bab_pgd_fraction`.
    ///
    /// Default `true`. A reserve taken BEFORE BaB can only be justified by a
    /// capability known BEFORE BaB, and the post-BaB falsifier's only such
    /// capability is the network gradient. On a net outside the exact-gradient
    /// fragment the pre-payment buys nothing: `#deadlane` hands the lane the
    /// window and it declines in ~0.03 s, so the seconds are discarded with the
    /// row still undecided (measured on vit_2023 at the official 100 s budget:
    /// BaB stopped at 81.0 s of a 95 s internal tier and the row published
    /// `timeout` while its own ledger read `remaining_s=14.198`).
    ///
    /// `false` restores the historical unconditional subtraction exactly. This
    /// is the kill switch for the sizing rule, not for the lane.
    pub(crate) demand_sized_post_bab_reserve: Option<bool>,

    /// #seq-single-reserve — base the sequential engine's warmup cap on the
    /// caller ledger's already-reserved BaB boundary instead of subtracting
    /// the post-BaB PGD reserve a SECOND time.
    ///
    /// Default `true`. The ledger's `bab_deadline` has already carved out
    /// `post_bab_pgd_fraction` (and the MIP reserve) when it hands the
    /// engine its deadline; the sequential initial phase then multiplied by
    /// `(1 - post_bab_pgd_fraction)` again — a double reserve every graph
    /// lane explicitly refuses. MEASURED 2026-08-31 to 0.1 s on two arms
    /// (tll 600 s rows: 483.0 ledger boundary -> 434.7 double-reserved ->
    /// 369.5 warmup cap; walls 372.38/372.36 vs 372.3 predicted, and
    /// 437.44/437.46 vs 437.4 at fraction 1.00) — see
    /// `reports/measured-2026/tll_n64_attack_starvation_2026-08-31/`.
    ///
    /// Warmup-cap-only: the BaB search boundary keeps the engine's single
    /// reserve in both dispositions, because that window is the internal
    /// post-BaB PGD fallback's only home (it produced tll `N=56_3`'s banked
    /// sat). `false` restores the historical double-reserved cap
    /// byte-identically.
    pub(crate) sequential_warmup_single_reserve: Option<bool>,

    /// #seq-early-attack — route a sequential warmup-cap expiry through the
    /// engine's post-BaB PGD fallback instead of forfeiting the attack.
    ///
    /// Default `true`. The historical `Early(Timeout)` return escaped the
    /// engine before the PGD fallback closure was even constructed, so an
    /// early-quit row ran ZERO attack steps (tll N=64: 100% of 600 s on a
    /// bounds phase that provably cannot finish, 0 s attacking two rows the
    /// field falsifies 11-0; the same attack found `N=56_3`'s banked sat in
    /// 2.24 s). The fallback can only produce an independently re-validated
    /// `Violated`, still oracle-gated at the CLI; a row it cannot falsify
    /// returns the identical Timeout. `false` restores the historical
    /// forfeit byte-identically.
    pub(crate) sequential_early_quit_attack: Option<bool>,

    /// #mip-demand — size the Graph-MIP escalation reservation from the
    /// escalation's own STATIC admission test instead of always subtracting
    /// `mip_min_fraction`'s slice.
    ///
    /// Default `true`, and the same argument as `demand_sized_post_bab_reserve`
    /// above: the slice is carved out of the SCORED budget before BaB runs, so
    /// it can only be justified by a capability known before BaB. Two of the
    /// escalation gate's own predicates are exactly that — the whole-net
    /// encode-nnz cap (`nnz` is a property of the MODEL, identical on every row
    /// of a family) and the minimum-slice floor (a reserve under the floor is
    /// refused before the model is looked at).
    ///
    /// MEASURED 2026-08-30 (`reports/measured-2026/stop_census_2026-08-30/`,
    /// 7/7 rows produced the `[stop]` line): `challenging_certified` idx1031
    /// reserved 40 s of a 114 s solver deadline and the escalation declined in
    /// ~2 ms with `estimated nnz 150936750 > cap 5000000`; `metaroom_2023`
    /// idx125 reserved 54 s of 200 s on `nnz 50355927`; `cora_2024` img141
    /// reserved 6 s of 25 s and declined at the 20 s slice floor.
    ///
    /// RE-DERIVED: those figures are TWO reserves. The part this key governs is
    /// the MIP slice alone — 28 s / 30 s / 6 s respectively, i.e. 15–25% of the
    /// solver deadline and **+20% to +37% more BaB search**, not 24–35%.
    ///
    /// `false` restores the historical unconditional reservation exactly. This
    /// is the kill switch for the sizing rule, not for the escalation.
    pub(crate) demand_sized_mip_reserve: Option<bool>,

    /// #disj-demand — zero the disjunctive global PGD slice when the net is
    /// statically outside the exact-gradient fragment (the same static probe
    /// as `demand_sized_post_bab_reserve`), so the phase could only spend the
    /// slice on SPSA over point-IBP forwards.
    ///
    /// Default `false` (historical schedule everywhere): the SPSA fallback is
    /// a real searcher and gradient-ineligible categories may score through it
    /// (ml4acopf_2024 has a banked 256 s `sat`), so each category arms this
    /// only behind its own A/B. Measured for vit_2023 at the official 100 s
    /// budget: the phase spends its full 4.75 s slice and has never produced a
    /// counterexample on the family. On a net INSIDE the fragment the armed
    /// key is inert — the static probe decides, not the yaml.
    pub(crate) demand_sized_disjunctive_pgd: Option<bool>,

    /// Whether the VNN-COMP wrapper may run its independent exact-gradient
    /// falsifier after the in-process verifier returns undecided.
    ///
    /// This is deliberately separate from `post_bab_pgd_fraction`: that
    /// fraction controls the engine's own phase schedule and historically said
    /// nothing about the outer wrapper lane. `None`/`true` preserves the
    /// default-enabled wrapper. An explicit `false` disables it. When paired
    /// with an explicit zero `post_bab_pgd_fraction`, the VNN-COMP router may
    /// also give the scalable historical tail back to proof while retaining
    /// its fixed results-publication margin.
    pub(crate) vnncomp_post_bab_attack: Option<bool>,

    /// Fixed whole-second reserve for that outer wrapper falsifier
    /// (`NY_POSTBAB_RESERVE_SECS`). Zero or absence preserves the historical
    /// schedule. A positive value never reduces a verifier window from ten or
    /// more seconds to below ten, and never extends an already-shorter window;
    /// the outer lane still has to pass its ordinary route, memory,
    /// executable-stage, and live-budget admissions before it can receive
    /// optimizer engagement credit.
    pub(crate) post_bab_reserve_secs: Option<u64>,

    /// #flush-grace — whole-second RESULTS_FILE publication reserve for the
    /// VNN-COMP wrapper, REPLACING the historical `max(5, budget/20)` grace
    /// outright. Default: `None` = today's rule, byte-identical.
    ///
    /// Like `vnncomp_post_bab_attack` and `post_bab_reserve_secs` this is an
    /// OUTER WRAPPER key, not part of the reusable engine schedule, so it is
    /// deliberately absent from `preset::apply` and is consumed directly from
    /// the immutable `PresetConfig` snapshot by the VNN-COMP router
    /// (`commands::vnncomp::results_flush_grace_from_snapshot`).
    ///
    /// WHY IT REPLACES THE WHOLE EXPRESSION RATHER THAN THE `5` FLOOR.
    /// Substituting only the floor (`max(g, budget/20)`) would be a no-op on
    /// every budget at or above `20 * g` — i.e. on precisely the long rows
    /// where the scalable 5 % tail is largest — so it would generalise
    /// nothing. The grace a family declares here is a MEASUREMENT CLAIM about
    /// that family's own publication tail, and it is stated absolutely.
    ///
    /// WHAT THE RESERVE BUYS, measured (see
    /// `commands::vnncomp::SCORED_WALL_STOP_RESERVE_SECS`): verdict
    /// translation, trusted-oracle gating for `sat`, counterexample-witness
    /// output, and the atomic RESULTS_FILE rename — 9-18 ms of publication
    /// plus the error in the BaB stop estimate. On the one family that has a
    /// `BabWall`, that error measured 0.00-0.19 s across 70 armed runs. A
    /// family WITHOUT a wall (everything but the relational categories) stops
    /// only at its own cooperative deadline polls, so its tail is the largest
    /// unbracketed unit of work between two polls and it must be MEASURED for
    /// that family before this key is armed for it.
    ///
    /// SETTING IT ALSO ANCHORS THE DEADLINE ABSOLUTELY. Like the proof-tail
    /// route, an explicit grace makes the verifier deadline
    /// `instance_start + budget - grace` rather than a bare duration, so model
    /// load and spec parse are charged once instead of being forgotten by a
    /// restarted clock. That is strictly safer, and it is why arming this with
    /// the historical value is NOT a no-op.
    ///
    /// The router clamps the result so the internal timeout is at least one
    /// second and never exceeds the scored budget; it never touches the scored
    /// instance deadline, either watchdog tier, or the pre-written sound
    /// `unknown`.
    pub(crate) results_flush_grace_secs: Option<u64>,

    /// Fraction of the REMAINING budget for the single adaptive attack
    /// extension (#attack-extend). Default: 0.15. Set 0.0 to disable for
    /// categories where the promising-margin gate cannot discriminate
    /// (e.g. cgan_2023 band properties).
    pub(crate) attack_extension_fraction: Option<f32>,
    /// Optional ABSOLUTE ceiling (seconds) on the disjunctive global PGD phase,
    /// on top of `disjunctive_pgd_fraction`. Default: None (pure fraction).
    /// Recommended for hold-heavy conv benchmarks (cifar100/tinyimagenet) where
    /// PGD beyond a few seconds is wasted and the seconds are better spent in
    /// BaB (which re-bases on remaining time).
    pub(crate) disjunctive_pgd_max_secs: Option<u64>,

    /// Optional ABSOLUTE FLOOR (seconds) on the disjunctive global PGD phase,
    /// applied after the tiny-budget 15% attack cap and clamped to half the
    /// scored budget (#attack-floor). Default: None (pure cap behavior).
    /// For categories whose BaB provably cannot decide, so the cap is pure loss
    /// — measured on lsnc_relu (see `PhaseBudgetConfig::disjunctive_pgd_min_secs`).
    pub(crate) disjunctive_pgd_min_secs: Option<u64>,

    /// #attack-anchor: spend `disjunctive_pgd_fraction` from the PHASE START
    /// rather than the ledger start. Default: None (= historical
    /// ledger-start anchoring, byte-identical).
    ///
    /// Ledger-start anchoring charges model load / graph build / VNN-LIB parse
    /// against the falsifier's own slice. MEASURED on cifar100_2024
    /// `CIFAR100_resnet_large` at the official 100 s budget with the shipped
    /// `disjunctive_pgd_fraction: 0.05`: the batched exact-VJP lane got
    /// **0.1 s of its 5 s slice and took ZERO steps**
    /// (`[pgd-vjp-disj] deadline (0.1s): wave_steps=0`). See
    /// `ny_propagate::PhaseBudgetConfig::disjunctive_pgd_from_phase_start`.
    pub(crate) disjunctive_pgd_from_phase_start: Option<bool>,

    /// Optional ABSOLUTE ceiling (seconds) on the disjunctive CROWN/alpha
    /// PRECHECK phase, on top of `disjunctive_precheck_fraction`
    /// (#precheck-abs-cap). Default: None (pure fraction).
    ///
    /// The fraction slice is granted from the PHASE START, so it bounds nothing
    /// relative to what the attack phase already spent, and the work it sizes
    /// (the CROWN-IBP root collection) grows with the model — a fraction tuned
    /// when the collection was cheap silently becomes a BaB-starvation lever.
    /// Set this to the measured full-collection cost plus headroom; unspent
    /// seconds are reclaimed by BaB.
    pub(crate) disjunctive_precheck_max_secs: Option<u64>,

    /// Optional ADAPTIVE stall cutoff for the disjunctive global PGD phase, as
    /// a fraction of that phase's own slice (#attack-stall). Default: None
    /// (no cutoff), and NO shipped preset sets it.
    ///
    /// Cuts the attack when its best margin has not improved beyond the
    /// confirmation noise floor for a whole `w * attack_slice` window — the
    /// per-instance counterpart of `disjunctive_pgd_max_secs`. Arming it for a
    /// category needs an A/B for THAT category that covers its SAT rows: this
    /// lane is what finds them on tinyimagenet (b61b5f10), and the reclaim it
    /// buys converted zero rows on the GT-unsat oval21 rows it was designed
    /// for. See `ny_propagate::PhaseBudgetConfig::disjunctive_pgd_stall_window_fraction`.
    pub(crate) disjunctive_pgd_stall_window_fraction: Option<f32>,

    /// #mip-handoff — ENFORCE the MIP reservation `mip_min_fraction` /
    /// `mip_min_secs` already declare. Default: `false` (unchanged behaviour).
    ///
    /// Without this, the same-LHS relational reduction gets NO absolute deadline
    /// and owns the whole remaining wall clock (ACAS-Xu prop_2 needs that), so
    /// the reserved slice is planned but never handed over and the escalation
    /// gate sees zero seconds. Set `true` only for categories whose CLOSER is
    /// the exact-MIP complete verifier rather than BaB — measured on
    /// safenlp_2024, where the reservation was inert on every row.
    ///
    /// Budget-only, hence verdict-neutral: it moves a deadline inside the same
    /// scored budget, never past it. See
    /// `ny_propagate::PhaseBudgetConfig::enforce_mip_handoff`.
    pub(crate) enforce_mip_handoff: Option<bool>,
}

/// Branching heuristic configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct BranchingPreset {
    /// Branching method: "width", "impact", "babsr", "fsb", "kfsb", "input",
    /// "relu", "nonlinear" (alias "genbab" — GenBaB general-nonlinearity
    /// branching, mirrors alpha-beta-CROWN's `method: nonlinear`).
    pub(crate) method: Option<String>,

    /// Number of candidates for ordinary FSB/kFSB branching.
    pub(crate) candidates: Option<usize>,

    /// Reduce operation for ordinary kFSB: "min", "max", "mean".
    pub(crate) reduceop: Option<String>,

    /// Candidate count for the wave-batched multi-objective kFSB selector
    /// only. This is the typed layer beneath governed `NY_MO_KFSB_K`; keeping
    /// it separate from `candidates` makes a scored treatment equivalent to
    /// the legacy environment arm instead of also retuning ordinary FSB/kFSB.
    pub(crate) kfsb_multi_candidates: Option<usize>,

    /// Two-child reduction for the wave-batched multi-objective selector only.
    /// The governed `NY_MO_KFSB_REDUCE` environment override retains its
    /// narrower exact `min|max` parser; the typed layer also accepts `mean`.
    pub(crate) kfsb_multi_reduceop: Option<String>,

    /// Arm the multi-objective wave-batched kFSB selector (#kfsb-multi). When
    /// `Some(true)`, sets `BetaCrownConfig::use_kfsb_multi_branching = true`.
    /// Measured Pareto on CIFAR-100 and scoped to those presets only. Env
    /// `NY_MO_KFSB` overrides the resulting arming in either direction (kill
    /// switch `NY_MO_KFSB=0`).
    pub(crate) kfsb_multi: Option<bool>,

    /// Reuse a strictly-authorized scalar lower certificate from the
    /// wave-batched kFSB child simulations. Omission preserves the default-off
    /// solver policy. Exact `NY_MO_KFSB_CERT_REUSE=1` force-arms it; every
    /// other present value is a fail-closed kill switch.
    pub(crate) kfsb_cert_reuse: Option<bool>,

    /// Default-off, bounded branch-specific depth-2 lookahead policy.
    ///
    /// Omission leaves `BetaCrownConfig` at its inert default. A present map
    /// may specify only the fields it changes; the remaining fields use the
    /// published 15-candidate / five-round / λ=0.5 defaults.
    pub(crate) depth2_lookahead: Option<DepthTwoBranchLookaheadConfig>,

    /// Input-splitting SB tuning.
    #[serde(default)]
    pub(crate) input_split: InputSplitPreset,

    /// Nonlinear split configuration (for networks with nonlinear operations).
    #[serde(default)]
    pub(crate) nonlinear_split: NonlinearSplitPreset,
}

/// Self-calibrating per-child bound allowance (#cost-observer):
/// `bab.branching.input_split.child_bound_calibration`.
///
/// Every field optional AND every field DIMENSIONLESS; the block itself is
/// optional — absent keeps `child_bound_max_secs` in charge (byte-identical).
/// The absence of any `*_secs` field is the point of the key: the seconds are
/// measured at runtime, not declared here.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputSplitChildBoundCalibrationPreset {
    /// Multiple of this row's MEASURED median completed-call cost that a call
    /// may run before being cut off. Omitted = 160.
    pub(crate) headroom: Option<f64>,
    /// Largest share of the REMAINING wall one call may take. Omitted = 0.25.
    pub(crate) max_share_of_remaining: Option<f64>,
    /// Share of the remaining wall spent on the regime-finding probe before any
    /// call has completed. Omitted = 0.05.
    pub(crate) probe_share: Option<f64>,
    /// Completed observations before the measured term governs. Omitted = 8.
    pub(crate) min_samples: Option<usize>,
    /// Sliding window of retained completed observations. Omitted = 128.
    pub(crate) window: Option<usize>,
    /// Growth per CONSECUTIVE truncated call. Omitted = 2.0.
    pub(crate) growth: Option<f64>,
}

/// Per-wave edge-work allowance (#iso-edge-cap):
/// `bab.branching.input_split.edge_wave_cap`.
///
/// Every field optional; the block itself is optional — absent keeps the edge
/// work ungoverned (byte-identical). An empty block (`edge_wave_cap: {}`) arms
/// it with the defaults.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputSplitEdgeWaveCapPreset {
    /// Ceiling on one wave's edge wall, seconds. Omitted = 2.0. Zero means
    /// "no edge work at all".
    pub(crate) max_secs: Option<f64>,
    /// Share of the batched rebound preceding the wave, dimensionless.
    /// Omitted = 0.25.
    pub(crate) rebound_share: Option<f64>,
}

impl InputSplitEdgeWaveCapPreset {
    /// The engine-side config this block delivers. The ONE mapping, shared by
    /// `preset::apply` and the relational route's own projection, so the two
    /// deliveries cannot disagree about what a block means. Unnamed fields stay
    /// unnamed; the lane resolves (and sanitizes) the defaults.
    pub(crate) fn to_config(&self) -> ny_propagate::beta_crown::EdgeWaveCapConfig {
        ny_propagate::beta_crown::EdgeWaveCapConfig {
            max_secs: self.max_secs,
            rebound_share: self.rebound_share,
        }
    }
}

/// Certified f64 tail escalation sub-key (#cgan-f64-tail, design B step B5):
/// `bab.branching.input_split.f64_tail`. Every field optional; the block
/// itself is optional — absent keeps the mechanism dark (byte-identical).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputSplitF64TailPreset {
    /// Arm both escalation seams (batch seam + survivor-polish hook).
    pub(crate) enabled: Option<bool>,
    /// Escalation guard band (gap >= -band escalates). Omitted = legacy 5e-3.
    pub(crate) band: Option<f32>,
    /// Hard per-run escalation cap shared by both seams (0 = unlimited).
    pub(crate) max_escalations: Option<usize>,
}

/// Input-splitting SB configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputSplitPreset {
    /// Enable input splitting for alpha-beta-CROWN-compatible preset imports.
    pub(crate) enable: Option<bool>,

    /// Opt into sequential single-objective DomainList searches for a canonical
    /// exactly-two-singleton disjunction. NY-specific and default false.
    pub(crate) independent_singleton_disjunction: Option<bool>,

    /// Try the sound `#lowdim-prover` lane first at the DomainList input-split
    /// entry (small-input graphs only). NY-specific and default false.
    pub(crate) lowdim_prover: Option<bool>,

    /// Budget share for that lane before it falls through (clamped to
    /// [0.05, 0.95] by the engine; default 0.5).
    pub(crate) lowdim_prover_share: Option<f32>,

    /// Backward-CROWN refinement of pre-activation layers no wider than this
    /// inside that lane (0 = off; default off).
    pub(crate) lowdim_prover_refine_width: Option<usize>,

    /// `#lowdim-fold`: prove per-clause-box disjuncts on their free
    /// coordinates before the box screen (graph models; default off).
    pub(crate) lowdim_fold: Option<bool>,

    /// #lsnc-lane: build the `#lowdim-prover` lane's LANE-PRIVATE interval
    /// twin of a model whose exact load left `OpaqueSkip` nodes (constant-only
    /// cones the exact folder declined), so the lane can bound it.
    ///
    /// Typed delivery for `NY_LOWDIM_INTERVAL_TWIN` (`MoatRisk::High`). The
    /// twin is a second load of the same model with the loader's interval
    /// fallback enabled for that one load, readable only by the lowdim lane
    /// (`ny_propagate::lowdim::LaneTwin`); the global interval carrier stays
    /// dark. Absence/false builds no twin, byte-identical to before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_interval_twin` (frozen
    /// scored-run LeverSet, or this key over the environment interactively),
    /// then `attach_lowdim_interval_twin` right after the model loads.
    pub(crate) lowdim_interval_twin: Option<bool>,

    /// #lsnc-lane stage 2: bound every `ReduceSum(Mul(a, b))` site of the
    /// `#lowdim-prover` lane's program by the JOINT QUADRATIC bound (two CROWN
    /// certificates per operand element, one quadratic over the input box,
    /// exact vertex maximisation of its convex part, the eigen residual
    /// charged in full), and substitute its planes in a second backward pass.
    ///
    /// Typed delivery for `NY_LOWDIM_JOINT_QUADRATIC` (`MoatRisk::High`).
    /// Absence/false: the lane bounds exactly as without it.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_joint_quadratic` (frozen
    /// scored-run LeverSet, or this key over the environment interactively)
    /// writes `BetaCrownConfig::input_split_lowdim_joint_quadratic`, which
    /// `BetaCrownConfig::lowdim_config` delivers to every lowdim entry.
    pub(crate) lowdim_joint_quadratic: Option<bool>,

    /// #lsnc-lane stage 3: the `#lowdim-prover` lane's split rule — `widest`
    /// (the historical rule), `gradient`, `sb` (alpha-beta-CROWN's `sb_sum`
    /// under an aspect guard) or `sbwr` (the numpy gate's `allmax` split).
    /// Any rule keeps both halves of the box, so it shapes the tree only.
    ///
    /// Typed delivery for `NY_LOWDIM_SPLIT` (`MoatRisk::High`). Absence:
    /// `widest`, exactly as before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_split` (frozen scored-run
    /// LeverSet, or this key over the environment interactively) writes
    /// `BetaCrownConfig::input_split_lowdim_split`, which
    /// `BetaCrownConfig::lowdim_config` delivers to every lowdim entry.
    pub(crate) lowdim_split: Option<ny_propagate::lowdim::LowDimSplit>,

    /// #lsnc-lane stage 3: RELAXED CLIP-AND-VERIFY inside the `#lowdim-prover`
    /// lane — each open box shrunk by the halfspaces of its OWN backward
    /// passes' planes and the clause's thresholds, rounded outward; an emptied
    /// box is closed.
    ///
    /// Typed delivery for `NY_LOWDIM_CLIP` (`MoatRisk::High`). Absence/false:
    /// no clip, exactly as before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_clip` writes
    /// `BetaCrownConfig::input_split_lowdim_clip` (-> `lowdim_config`).
    pub(crate) lowdim_clip: Option<bool>,

    /// #lsnc-lane stage 3: decide each lane domain (close test, clip,
    /// candidate test, split) inside its rayon chunk task instead of serially
    /// after the bound. Outcome-identical by construction; throughput only.
    ///
    /// Typed delivery for `NY_LOWDIM_PARALLEL` (`MoatRisk::High`: a deadline-
    /// bounded verdict moves with throughput). Absence/false: the historical
    /// serial decision.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_parallel` writes
    /// `BetaCrownConfig::input_split_lowdim_parallel` (-> `lowdim_config`).
    pub(crate) lowdim_parallel: Option<bool>,

    /// #lowdim-bundle (1): admit the `#lowdim-prover` lane by VARYING input
    /// dimensions — a box wider than the lane's 16-dimension cap whose other
    /// coordinates are pinned (`lo == hi`, dist_shift_2023's 784 image pixels
    /// beside 8 latent coordinates) is re-expressed over its free ones, the
    /// pinned columns folded into rigorous interval biases — and lower
    /// `Sigmoid` to rigorous tangent/chord planes (rounding charged).
    ///
    /// Typed delivery for `NY_LOWDIM_VARYING_DIMS` (`MoatRisk::High`).
    /// Absence/false: such boxes and graphs are declined exactly as before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_varying_dims` writes
    /// `BetaCrownConfig::input_split_lowdim_varying_dims` (-> `lowdim_config`).
    pub(crate) lowdim_varying_dims: Option<bool>,

    /// #lowdim-bundle (2): BOUNDARY ALPHA inside the `#lowdim-prover` lane —
    /// this many projected steps on the lower ReLU slopes `alpha in [0, 1]`
    /// of the objective pass, for open leaves within 10 of the threshold
    /// (Vibecheck's acasxu wide band); the published bound is the maximum of
    /// rigorous bounds. 0 is off.
    ///
    /// Typed delivery for `NY_LOWDIM_ALPHA_ITERS` (`MoatRisk::High`).
    /// Absence/0: the historical `{0, 1}` lower slopes, exactly as before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_alpha_iters` writes
    /// `BetaCrownConfig::input_split_lowdim_alpha_iters` (-> `lowdim_config`).
    pub(crate) lowdim_alpha_iters: Option<usize>,

    /// #lowdim-bundle (3): IN-LANE WORST-LEAF PGD — every this many waves the
    /// 64 open leaves furthest from closing get 256 restarts x 50 signed steps
    /// on the lane's lowering, confined to each leaf box (wall share capped
    /// at 20% of the lane). Attack-only: a hit is a CANDIDATE that the entry's
    /// trusted forward re-checks before the unchanged trusted-oracle gate.
    ///
    /// Typed delivery for `NY_LOWDIM_LEAF_ATTACK_EVERY` (`MoatRisk::High`:
    /// a confirmed hit turns an unknown row into `sat`).
    /// Absence/0: no attack, exactly as before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_leaf_attack_every`
    /// writes `BetaCrownConfig::input_split_lowdim_leaf_attack_every`.
    pub(crate) lowdim_leaf_attack_every: Option<usize>,

    /// #lowdim-bundle (4): prove the `#lowdim-fold` families on ONE shared
    /// worklist (per-clause round-robin waves, no per-clause domain cap, the
    /// relaxed clip on single-objective families when `lowdim_clip` is on)
    /// instead of one family after another.
    ///
    /// Typed delivery for `NY_LOWDIM_SHARED_WORKLIST` (`MoatRisk::High`).
    /// Absence/false: the sequential family loop, exactly as before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_shared_worklist` writes
    /// `BetaCrownConfig::input_split_lowdim_shared_worklist`.
    pub(crate) lowdim_shared_worklist: Option<bool>,

    /// M8a (`#m8-input-split-batch`): the `#lowdim-prover` lane's wave
    /// BOUNDER — `legacy` (the lane's own f64 forms on every domain) or `bbe`
    /// (one call per wave of the injected UNTRUSTED forward-linear search;
    /// only its CLAIMS are re-bounded by the lane's f64 forms and close on the
    /// strict `lb > thr`, every other leaf is split into its exact halves).
    ///
    /// Typed delivery for the `lowdim_batch_engine` lever (`MoatRisk::High`).
    /// Absence: `legacy`, exactly as before. No shipped yaml sets it: the
    /// router decides (user directive 2026-09-29).
    ///
    /// CONSUMER: `commands::beta_crown::resolve_lowdim_batch_engine` writes
    /// `BetaCrownConfig::input_split_lowdim_batch_engine` (-> `lowdim_config`).
    pub(crate) lowdim_batch_engine: Option<ny_propagate::lowdim::LowDimBatchEngine>,

    /// M8a (`#m8-input-split-batch`): the `#lowdim-prover` lane's
    /// MULTI-CLAUSE scheduler — `legacy` (one tree per clause, the fold's
    /// family loop) or `bbe` (one disjunction tree for same-box clauses, the
    /// fold's shared worklist). The graph input-split drivers are untouched.
    ///
    /// Typed delivery for the `input_split_engine` lever (`MoatRisk::High`).
    /// Absence: `legacy`, exactly as before.
    ///
    /// CONSUMER: `commands::beta_crown::resolve_input_split_engine` writes
    /// `BetaCrownConfig::input_split_engine` (-> `lowdim_config`).
    pub(crate) engine: Option<ny_propagate::lowdim::InputSplitEngine>,

    /// Coefficient clamp threshold for SB input split scoring.
    #[serde(alias = "coeff_thresh")]
    pub(crate) sb_coeff_thresh: Option<f32>,

    /// Bonus score for intervals that touch zero.
    pub(crate) touch_zero_score: Option<f32>,

    /// Margin weight for SB input split scoring.
    #[serde(alias = "margin_weight")]
    pub(crate) sb_margin_weight: Option<f32>,

    /// Sum across spec rows instead of taking the max.
    pub(crate) sb_sum: Option<bool>,

    /// Restrict SB scoring to a single specification row.
    #[serde(alias = "primary_spec")]
    pub(crate) sb_primary_spec: Option<usize>,

    /// Enable IBP enhancement for the input-split BaB loop.
    /// When true, each domain (root + children) is screened with fast IBP before
    /// expensive CROWN backward. Domains verified by IBP skip CROWN entirely.
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.ibp_enhancement`
    pub(crate) ibp_enhancement: Option<bool>,

    /// Admit ny's bounded proof-only nonnegative conic closure for conjunctive graph
    /// input splitting. The current structural detector accepts only the two
    /// signed-zero-threshold rows `[1, 0] <= +0` and `[0, -1] <= -0`; omitted
    /// is default-dark.
    pub(crate) conic_objective: Option<bool>,

    /// Maximum deferred-rebound tranche before an authenticated affine-conic
    /// search returns newly bounded domains to its priority heap.
    pub(crate) conic_queue_refresh_batch_size: Option<usize>,

    /// Enable the domain-stacked dense-spec batched rebound
    /// (#cgan-batched-stack): one conv/BN backward call per node across the
    /// whole domain batch, plus fresh per-domain IBP re-anchoring when
    /// `ibp_enhancement` is also set. ny-specific (no alpha-beta-CROWN
    /// counterpart; the reference batches all layers on GPU natively).
    pub(crate) stacked_rebound: Option<bool>,

    /// Certified f64 tail re-bounding of knife-edge survivors on the
    /// grouped-disjunctive input-split lane (#cgan-f64-tail, design B).
    /// Sub-key `bab.branching.input_split.f64_tail.{enabled, band,
    /// max_escalations}` maps onto `BetaCrownConfig::input_split_f64_tail_*`.
    /// Omitted = dark everywhere (byte-identical); the env levers
    /// `NY_F64_TAIL` / `NY_F64_TAIL_BAND` override at the ny-propagate
    /// resolution chokepoint (`resolve_f64_tail_settings`). ny-specific.
    pub(crate) f64_tail: Option<InputSplitF64TailPreset>,

    /// Nested-rebound deadline authority for the input-split dense-spec
    /// rebound (#dist-shift-5291; the `6f49a660` deadline-presence cliff).
    /// `false` = the nested rebound verifier runs its bound computations
    /// WITHOUT the inner finite deadline (the pre-`6f49a660` shape): wall-
    /// clock safety comes from the OUTER BaB loop's own deadline poll between
    /// rebound batches plus the harness per-instance budget; what is ceded is
    /// intra-rebound interruptibility, bounded by one rebound batch — so only
    /// categories whose per-batch rebound cost is known-small should set it.
    /// Soundness is unaffected either way (the nested deadline selects routes
    /// and fallbacks, never bound math). Omitted/`true` keeps the shipped
    /// deadline-carrying arm. Env `NY_INPUT_SPLIT_NESTED_DEADLINE` overrides
    /// in both directions (`0` drop / `1` keep). ny-specific.
    pub(crate) nested_deadline: Option<bool>,

    /// Per-CHILD wall cap in SECONDS for the grouped-disjunctive input-split
    /// per-child bound calls (#cgan-dark-tail FIX-1). Maps onto
    /// `BetaCrownConfig::input_split_child_bound_max_secs`.
    ///
    /// OMITTED — the shipped state for every preset — is NO CAP: each per-child
    /// bound keeps the absolute BaB `crown_deadline` it has today and the route
    /// is byte-for-byte identical. Naming the key makes each call run against
    /// `min(crown_deadline, now + secs)`; the absolute ceiling is never
    /// extended.
    ///
    /// The defect it addresses: one child bound is entitled to the whole
    /// remaining budget, and the trajectory only prints between batches, so it
    /// spends it invisibly (cgan row 2: ~290 s / 32% of the 900 s budget on
    /// 0-10 domains, 13/13 logs; tail calls ~70 s to >=285 s against a 0.13 s
    /// median). Soundness is unaffected: a per-call deadline can only make a
    /// bound LOOSER, and the frozen-root-α retry that catches the `Err` arm
    /// already exists and is already exercised. Config-only, no env override.
    /// ny-specific.
    pub(crate) child_bound_max_secs: Option<f64>,

    /// #cost-observer: SELF-CALIBRATING replacement for
    /// `child_bound_max_secs`. Maps onto
    /// `BetaCrownConfig::input_split_child_bound_calibration`.
    ///
    /// OMITTED — the shipped state for every preset — leaves the constant above
    /// in charge and the route byte-for-byte identical. Naming the key derives
    /// each call's allowance at runtime from the median cost of the calls that
    /// have already completed ON THIS ROW and the wall actually remaining.
    ///
    /// The defect it addresses is not that the constant is mis-tuned; it is
    /// that no single value exists. A/B'd 2026-08-26 at the official 1200 s on
    /// the two 1200 s cgan rows at `child_bound_max_secs: 20.0`:
    /// `nonlinear_activations` explored `2 -> 957` with zero collection
    /// truncations (20 s CORRECT), `upsample` took 37 truncations cutting a
    /// 27-layer collection to 5-6 with the gap unmoved (20 s an order of
    /// magnitude TOO SHORT). Their per-call costs differ by ~3 orders of
    /// magnitude — `0.13 s` median versus `>= 300 s` — and nothing in the
    /// instance declares that.
    ///
    /// Soundness is unaffected and for the same reason as the constant: the
    /// derived allowance is composed by the same `min(ceiling, now + allowance)`
    /// law, so a bound can only ever get LOOSER, never wrong. Config-only, no
    /// env override. ny-specific.
    pub(crate) child_bound_calibration: Option<InputSplitChildBoundCalibrationPreset>,

    /// #iso-edge-cap: bound the SERIAL edge work of the grouped-disjunctive
    /// input-split BaB — the α edge pass and the edge-MILP escalation — by the
    /// scored stop and a per-wave wall allowance of
    /// `min(max_secs, rebound_share x the rebound that precedes the wave)`.
    /// Maps onto `BetaCrownConfig::input_split_edge_wave_cap`.
    ///
    /// OMITTED — the shipped state for every preset — leaves the edge work
    /// ungoverned, byte-for-byte as before. The block's PRESENCE arms it;
    /// every field is optional (`max_secs` 2.0, `rebound_share` 0.25).
    ///
    /// The edge work is armed today only by the RELATIONAL lane's inline
    /// config, which never runs `preset::apply`; that route projects this
    /// block from the category preset itself (`RelationalPresetKeys` in
    /// `commands/vnncomp.rs`). On every other lane edge-α and edge-MILP are
    /// off, so the key is inert there.
    ///
    /// Soundness: it can only REMOVE edge work, turning a would-be verify into
    /// a split. Config-only, no env override. ny-specific.
    pub(crate) edge_wave_cap: Option<InputSplitEdgeWaveCapPreset>,

    /// Parallelize independent per-domain warm-alpha refinements in the
    /// deferred reordered rebound. ny-specific and default false; a preset must
    /// opt in before `NY_INPUT_SPLIT_WARM_PARALLEL` may select the parallel arm.
    pub(crate) warm_parallel: Option<bool>,

    /// Evaluate complete-clip child-local CROWN rebounds with a fixed two-way
    /// executor when a conservative workload estimate fits within one eighth
    /// of a live kernel-enforced process envelope. This limits incremental OOM
    /// risk rather than guaranteeing a whole-call peak. ny-specific, default
    /// false, and preset-scoped so a global environment setting cannot arm
    /// unrelated categories.
    pub(crate) override_parallel: Option<bool>,

    /// Arm Saturation-Escape Branching for this category (#nn4sys-seb-dark).
    ///
    /// Typed form of `NY_SAT_ESCAPE_BRANCH` — maps to
    /// `BetaCrownConfig::sat_escape_branch`, which arms BOTH consumers the env
    /// flag arms: the advisory input-split SEB dim scorer (ny-propagate
    /// `sat_escape.rs`) and the disjunctive precheck-fraction cap that reserves
    /// per-clause BaB budget so the brancher is actually reached
    /// (`verify/disjunctive.rs`). The env var still overrides either way
    /// (`0` kill switch, `1` force-on). Absent keeps today's dark default —
    /// every preset that does not name this key is byte-identical.
    ///
    /// ny-specific (no alpha-beta-CROWN counterpart). Shipped only in
    /// nn4sys.yaml; see the probe numbers cited there.
    pub(crate) sat_escape_branch: Option<bool>,

    /// Use reordered BaB loop: bound before split (bound → filter → split → clip).
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.reorder_bab`
    pub(crate) reorder_bab: Option<bool>,

    /// #cap-extend (DEFER-dont-DROP): when the disjunctive input-split BaB
    /// loop trips `max_domains` with the frontier NEARLY CLOSED (head gap
    /// grazed zero AND ≥ 1/4 of explored domains verified), extend the search
    /// once (bounded ×8, wall still governs) instead of discarding the open
    /// frontier for a from-scratch restart re-trace. Omitted — the shipped
    /// state for every preset — is byte-identical to today; a cap that trips
    /// far from closure stops exactly as today either way. Schedule-only
    /// (never bound math). Maps onto
    /// `BetaCrownConfig::input_split_cap_extend_near_closure`. ny-specific.
    pub(crate) cap_extend_near_closure: Option<bool>,

    /// #sb-plane: while the fresh current-domain clip is armed, keep the popped
    /// domain's rebound plane for the SB split-dimension scorer ONLY (the fresh
    /// clip keeps computing its own plane; nothing verdict-bearing sees the
    /// kept one). Without it, arming the fresh clip makes SB width-only. Omitted
    /// — the shipped state for every preset — is byte-identical to today.
    /// Heuristic-only (never bound math). Maps onto
    /// `BetaCrownConfig::input_split_sb_keep_plane_under_fresh_clip`.
    /// ny-specific.
    pub(crate) sb_keep_plane_under_fresh_clip: Option<bool>,

    /// Domain-count threshold for adversarial checking during BaB.
    /// -1 = disabled, 0 = from first iteration, N = after N domains explored.
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.adv_check`
    pub(crate) adv_check: Option<i32>,

    /// Number of input dimensions to split per parent (multi-dimensional input
    /// split). Each parent is midpoint-split on the top-`depth` SB-scored dims,
    /// producing up to 2^depth children that exactly cover the parent (BaB
    /// completeness preserved). Default 1 = the classic 1-dim → 2-child split.
    /// Larger values fill the GPU batch from fewer parents.
    /// Reference: alpha-beta-CROWN `storage_depth` (fills `batch_size`).
    #[serde(alias = "storage_depth")]
    pub(crate) depth: Option<usize>,

    /// Per-sub-domain α refinement iterations in the input-split BaB loop.
    /// When > 0 AND alpha-CROWN is enabled, each sub-domain warm-starts from its
    /// parent's optimized alphas and re-optimizes them for this many SPSA
    /// iterations against the sub-domain's tighter box. Default 0 (off).
    /// Reference: alpha-beta-CROWN `solver.alpha-crown.input_split_alpha_iteration`.
    #[serde(alias = "input_split_alpha_iteration")]
    pub(crate) alpha_iteration: Option<usize>,

    /// Learning rate for per-sub-domain α refinement (see `alpha_iteration`).
    /// Only used when `alpha_iteration > 0`. Default 0.05.
    /// Reference: alpha-beta-CROWN `solver.alpha-crown.input_split_lr_alpha`.
    #[serde(alias = "input_split_lr_alpha")]
    pub(crate) lr_alpha: Option<f32>,
}

/// Nonlinear split configuration.
///
/// In alpha-beta-CROWN this section selects the GenBaB branching path for networks
/// with general nonlinearities (bounded `Mul`/`MatMul`, Sigmoid, Sin/Cos, …). ny
/// treats a configured `nonlinear_split` (any field present, or `enable: true`) as a
/// request for [`ny_propagate::BranchingHeuristic::GenBaB`], so the BaB loop splits
/// the product / activation inputs and tightens the McCormick relaxation, rather than
/// falling back to pure input splitting which cannot touch the nonlinear frontier.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct NonlinearSplitPreset {
    /// Explicitly enable GenBaB nonlinear branching. When unset, GenBaB is still
    /// selected if `filter` / `filter_beta` are present and no `method` is pinned.
    pub(crate) enable: Option<bool>,

    /// Enable filtering for nonlinear splits.
    pub(crate) filter: Option<bool>,

    /// Enable beta filtering for nonlinear splits.
    pub(crate) filter_beta: Option<bool>,

    /// #genbab-candidates: how many scored candidates the GenBaB selector keeps
    /// per split (`selector.rs` sorts by score, then
    /// `candidates.truncate(num_candidates)`).
    ///
    /// The engine default is **1**, so GenBaB branching has always considered a
    /// single candidate, and until this key existed no yaml could change that.
    /// ml4acopf's `branching.candidates: 100` does NOT reach here — that key
    /// lands on `fsb_candidates`, which only the sequential selector and the two
    /// kFSB lanes read, none of which ml4acopf runs.
    ///
    /// Absent keeps the engine default. `configs/vnncomp25/ml4acopf_2024.yaml`
    /// sets 100 (#genbab-selector-convention); value-neutral until a consumer
    /// reads past `decisions[0]` (see `preset::branching::genbab_config`).
    pub(crate) num_candidates: Option<usize>,
}

impl NonlinearSplitPreset {
    /// Whether this preset section requests GenBaB nonlinear branching.
    ///
    /// True when explicitly enabled, or when any nonlinear-split tuning field is
    /// present (mirrors alpha-beta-CROWN, where a populated `nonlinear_split`
    /// section is itself the GenBaB directive). A `Some(false)` `enable` opts out.
    pub(crate) fn requests_genbab(&self) -> bool {
        match self.enable {
            Some(enabled) => enabled,
            None => {
                self.filter.is_some() || self.filter_beta.is_some() || self.num_candidates.is_some()
            }
        }
    }
}

/// `solver.alpha-crown.softmax: complex` — the constant-shift softmax
/// decomposition (`GraphNetwork::decompose_softmax_complex`).
pub(crate) const SOFTMAX_MODE_COMPLEX: &str = "complex";

/// `solver.alpha-crown.softmax: complex_fixed_index` (#softmax-fixed-index) —
/// the fixed-index shift (`GraphNetwork::decompose_softmax_complex_fixed_index`).
pub(crate) const SOFTMAX_MODE_COMPLEX_FIXED_INDEX: &str = "complex_fixed_index";

/// Every value `AlphaCrownPreset::softmax` accepts (ASCII case-insensitive).
/// Anything else is refused at preset load (`validate_alpha_preset`).
pub(crate) const SOFTMAX_PRESET_MODES: [&str; 2] =
    [SOFTMAX_MODE_COMPLEX, SOFTMAX_MODE_COMPLEX_FIXED_INDEX];

/// `bab.alpha_crown.cgan_child_zono_refresh` (`#cgan-child-zono`). Each
/// absent field keeps the engine default (`enabled: false`,
/// `max_generators: 16384`, no per-child cap).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct CganChildZonoRefreshPreset {
    /// Arm the per-child zonotope refresh.
    pub(crate) enabled: Option<bool>,
    /// Generator budget per child, `8..=1048576`; rows beyond it fold soundly
    /// into the remainder. `0` is refused, never read as "unlimited".
    pub(crate) max_generators: Option<u32>,
    /// Per-child wall budget in seconds (finite, positive). Expiry leaves that
    /// child on its previous map.
    pub(crate) max_secs: Option<f32>,
}

/// α-CROWN configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct AlphaCrownPreset {
    /// Learning rate for α parameters.
    pub(crate) lr_alpha: Option<f32>,

    /// #root-alpha-margin: rank the root warmup's α iterates by the SPEC objective and return
    /// the best-scoring one, instead of the last iterate.
    /// Typed form of `NY_ROOT_ALPHA_MARGIN`.
    ///
    /// The warmup ascends a sum over RAW output dims while a multi-class property is a
    /// conjunction of MARGIN rows, and it keeps no best-α snapshot — so iterations spent on
    /// the wrong objective can hand the downstream spec pass an α worse for the margins.
    ///
    /// HISTORICAL: a 2026-07-26 Metal/WGPU run before the `1ede1d30` proof-adapter quarantine
    /// reported 43/99 verified specs on `CIFAR100_resnet_medium prop_idx_7704`. That result is
    /// not admissible evidence for the current sound verdict path. On HEAD `3b803c19`, a
    /// sound-path remeasurement found 0/99 both at baseline and with the env gate armed: no
    /// current verified-spec-count gain on this row. The lever therefore remains experimental.
    ///
    /// SOUND: selection only. The score never decides a verdict and never feeds a bound; it
    /// picks WHICH α to keep, and every α ∈ [0,1] yields a valid bound. The worst case is a
    /// weaker bound, never a wrong one.
    ///
    /// The env var still wins wherever it is PRESENT, including as a kill switch:
    /// `NY_ROOT_ALPHA_MARGIN=0` disables ranking a preset asked for.
    ///
    /// No shipped yaml sets this; arming it requires a current sound-path positive A/B.
    pub(crate) root_alpha_margin: Option<bool>,

    /// #alpha-zero-yield: retire the root α ascent after this fraction of its own window
    /// passes with no improvement over the best iterate, returning the remainder to search.
    /// Valid range `(0.0, 0.9)`.
    ///
    /// SOUND by construction: the early exit returns the already-certified elementwise best
    /// (`propagate_dag/mod.rs`, the same `should_save_best` route as ordinary convergence).
    /// Stopping sooner can return a looser certified enclosure; it cannot manufacture an
    /// invalid bound.
    ///
    /// MEASURED (2026-08-11, official 100 s budget,
    /// docs/LEVER_CENSUS_AND_ROOT_ALPHA_REMEASURE_2026-08-11.md §8 + addendum): at 0.25 the
    /// 16-row medium sample fired on 15/15 timeout rows, returned 8.4–14.8 s of root time per
    /// row (mean ~10.1 s), improved root-verified objectives on 3 rows (+15/+1/+1), had 0
    /// regressions and 0 verdict changes, and kept the sat row's counterexample byte-identical.
    /// A subsequent 16-row large sample did not engage the gate. No conversions were observed.
    /// The run retained no complete per-row receipt, commands, or input hashes, so this remains
    /// candidate evidence rather than an approved shipped default.
    ///
    /// `NY_ALPHA_ZERO_YIELD_FRAC` still wins wherever it is PRESENT, including as a kill
    /// switch: any invalid value (e.g. `0`) disarms a preset-armed fraction.
    /// No shipped preset sets this key. The typed seam remains available for controlled runs.
    pub(crate) alpha_zero_yield_frac: Option<f64>,

    /// #spec-axis-alpha: number of per-spec δ slots (worst-K margins get
    /// private α corrections). Absent/0 = shared-α behavior, byte-identical.
    /// Updates additionally require the margin-gradient lane
    /// (#root-alpha-margin); without it a nonzero K is inert by construction
    /// (`docs/SPEC_AXIS_ALPHA_DESIGN.md` §4–5).
    pub(crate) spec_slots: Option<usize>,

    /// Learning rate decay factor (exponential).
    pub(crate) lr_decay: Option<f32>,

    /// Number of optimization iterations.
    /// Supports both "iterations" (ny) and "iteration" (alpha-beta-CROWN).
    #[serde(alias = "iteration")]
    pub(crate) iterations: Option<usize>,

    /// Share α parameters across batch.
    pub(crate) share_alphas: Option<bool>,

    /// Softmax bound mode. `"complex"` decomposes each Softmax node into the
    /// alpha-optimizable Exp/ReduceSum/Reciprocal/MulBinary primitive subgraph
    /// at model load (vit_2023), shifting every Exp input by a CONSTANT.
    /// `"complex_fixed_index"` (#softmax-fixed-index) instead shifts by the
    /// logit at a frozen per-row index `j*` (the argmax at the box centre), so
    /// each Exp input is a RELATIVE logit `x_i - x_{j*}` whose `j*` entry is
    /// exactly 0 and the denominator is at least 1 — alpha-beta-CROWN's
    /// `softmax: complex` + `fixed_reducemax_index: True` (vnncomp23 vit winner
    /// recipe). Observable: `[softmax-complex] stage=rewrite decomposed=a/b
    /// shift=fixed-index …`. Any other value is REJECTED at preset load
    /// (`validate_alpha_preset`, [`SOFTMAX_PRESET_MODES`]), so a misspelled
    /// mode cannot silently measure the default direct-LSE relaxation. Runtime
    /// kill-switch for both values: `NY_NO_SOFTMAX_COMPLEX=1`.
    pub(crate) softmax: Option<String>,

    /// Use full convolution alpha (memory intensive).
    pub(crate) full_conv_alpha: Option<bool>,

    /// Fraction of the root α loop's remaining deadline assigned to the
    /// aggregate intermediate-reference refresh pool. Valid values are finite
    /// and in `[0.01, 1.0]`; unset preserves the built-in `0.25`.
    pub(crate) reference_refresh_fraction: Option<f32>,

    /// #joint-interm-alpha: rebuild the INTERMEDIATE bounds at the current alpha
    /// every `k` root-alpha iterations, turning the ascent into block-coordinate
    /// optimization over alpha AND the relaxation. `0`/unset keeps the legacy
    /// `improved_output`-gated refresh, which is measured dead on cifar100.
    ///
    /// This is a preset key rather than an env lever on purpose: the scored entry
    /// point exports no `NY_*` variables, so an env-only setting cannot
    /// fire in competition however well it measures
    /// (`crates/ny-cli/tests/measured_gate_delivery.rs`).
    pub(crate) joint_interm_alpha_every: Option<usize>,

    /// #envelope-grad: replace the sign-definite local alpha-gradient rule with
    /// the concretization-argmin (envelope) rule, so the ascent can raise alpha
    /// instead of only walking it to the 0 clamp. Unset keeps the shipped local
    /// rule. `NY_ALPHA_ENVELOPE_GRAD` still overrides in both directions for
    /// A/B, but env cannot fire in competition — this key is the delivery.
    pub(crate) alpha_envelope_grad: Option<bool>,

    /// Optional absolute ceiling, in seconds, on that same aggregate refresh
    /// pool. `0` explicitly disables refresh work. Unset preserves the
    /// fraction-only default.
    pub(crate) reference_refresh_max_secs: Option<u64>,

    /// When the preferred forward-linear intermediate collector refuses on
    /// deadline, use its sound plain-IBP fallback directly instead of spending
    /// the remaining slice in CROWN-IBP. Unset preserves the historical
    /// fallback chain.
    pub(crate) forward_linear_deadline_fallback_to_ibp: Option<bool>,

    /// Skip saving best bounds during warmup. Matches α,β-CROWN's `start_save_best`.
    pub(crate) start_save_best: Option<f32>,

    /// Reuse the cheap IBP forward pass for INTERMEDIATE node bounds instead of
    /// recomputing them with CROWN-IBP inside the α-CROWN pass. Maps 1:1 to
    /// `AlphaCrownConfig::fix_interm_bounds` and to auto_LiRPA's
    /// `bound_opts`/`compute_bounds(..., interm_bounds=…)` `fix_interm_bounds`
    /// kwarg that α,β-CROWN threads through `init_alpha`.
    ///
    /// `true` (the ny default) is the cheap O(N) path. `false` pays an O(N²)
    /// per-node CROWN-IBP sweep for much tighter intermediate — and therefore
    /// root — bounds. Unset keeps the built-in default, so every preset that
    /// does not name this key is byte-identical.
    ///
    /// The equivalent CLI flag is `--crown-ibp-intermediates` (= this key set
    /// to `false`); an explicit flag still wins over the preset.
    ///
    /// ml4acopf_2024 sets `false`: on 14_ieee prop9 the IBP-intermediate root
    /// bound on Y_159 is [-2.2110631, 3.6205366] against a true range of
    /// [-0.00236, -0.00068] (ORT, 4000 samples) — ~3,400x too wide for a
    /// property with a ±0.01 threshold, and no amount of branching closes it.
    /// With CROWN-IBP intermediates the same root tightens to
    /// [-0.00013195918, 0.003164109].
    pub(crate) fix_interm_bounds: Option<bool>,

    /// Default-dark cGAN root collector. With `fix_interm_bounds: false`, keep
    /// the certified forward-linear map as a baseline and spend the root
    /// collection budget on one atomic sparse ReLU-preactivation target.
    ///
    /// The propagation layer additionally requires a sequential ConvTranspose
    /// graph. Child warm starts clear the policy and retain their cheap
    /// forward-linear reference route.
    pub(crate) cgan_sparse_target_complete_root: Option<bool>,

    /// Default-dark complete cGAN root cascade. Starts from the certified
    /// forward-linear map and tightens every demanded CROWN-IBP target. Child
    /// warm starts clear the policy and retain the cheap forward-linear map.
    pub(crate) cgan_complete_crown_ibp_root: Option<bool>,
    /// Default-dark consumer for the complete cGAN cascade on the
    /// `bound_prop_method: crown` input-split lane (`#crown-root-map`). The
    /// root collector's `crown` arm publishes the cascade map to the BaB
    /// children — under the two publication guards (`ibp_enhancement` and
    /// `stacked_rebound`) and only on the collector's own typed provenance.
    /// Without it that arm returns no intermediate map, and the cascade
    /// `cgan_complete_crown_ibp_root` pays for is consulted by nothing.
    pub(crate) cgan_crown_root_map: Option<bool>,
    /// Default-dark E5 experiment key (`#cgan-collector-dense`): run the typed
    /// cGAN CROWN-IBP root cascade in DENSE (matrix) conv mode instead of the
    /// Patches mode the collector forces on both of its routes (the graph
    /// side ORs in `collector_patches_override`; the sequential shortcut has
    /// no conv-mode input). Changes the cost and the tightness of the
    /// collected map, never its soundness. Observable, armed or not:
    /// `[collector-mode] cgan_collector_dense=… patches_override=…
    /// use_patches_mode=…`.
    pub(crate) cgan_collector_dense: Option<bool>,
    /// Default-dark `#cgan-unstable-rows`: inside the typed, shrink-only cGAN
    /// root cascade, seed each demanded ReLU-preactivation target's CROWN
    /// backward with ONLY the rows the certified forward-linear baseline leaves
    /// sign-unstable or unresolved (alpha-beta-CROWN's
    /// `sparse_intermediate_bounds`). The mechanism already existed on this
    /// collector path behind the raw `NY_CROWN_IBP_SPARSE_RELU_ROWS`
    /// environment variable, which `run_instance.sh` never exports; this key is
    /// the only route that can fire on a scored run. Cost, not authority:
    /// omitted rows keep the certified baseline. Observable, armed or not:
    /// `[unstable-rows] root armed=… source=…`.
    pub(crate) cgan_unstable_rows: Option<bool>,
    /// Companion ceiling for `cgan_unstable_rows` (abc `minimum_sparsity`,
    /// default 0.9): the maximum fraction of a target's rows that may be
    /// selected before that target takes the existing full-width walk instead.
    /// Scheduling-only; a declined target keeps the certified baseline.
    pub(crate) cgan_unstable_rows_max_frac: Option<f32>,
    /// Row ceiling for demanded CROWN targets inside the complete cGAN root
    /// cascade (`#cgan-narrow-target-budget`). Absent/None keeps the historical
    /// all-demanded candidate set. Scheduling-only: a declined target retains
    /// the certified forward-linear baseline the cascade was seeded with.
    pub(crate) cgan_narrow_target_max_rows: Option<usize>,

    /// Per-child wall budget, in seconds, for child-side intermediate
    /// refinement (`#cgan-child-interm-refine`). Absent keeps today's
    /// root-only cascade. Requires `cgan_complete_crown_ibp_root`; the budget
    /// is the only dial and is not optional, because an unbounded per-child
    /// cascade is the O(N^2) pass the child path exists to avoid.
    pub(crate) cgan_child_interm_refine_max_secs: Option<f64>,

    /// `#cgan-child-seed`: seed each child cascade's forward-linear baseline
    /// with the domain's inherited map (frozen root CROWN map intersected with
    /// any child-local override) so the sparse ReLU-row selector walks only
    /// the rows the ROOT map leaves unstable. Absent keeps the unseeded child
    /// cascade. Inert unless `cgan_child_interm_refine_max_secs` is armed.
    /// Shrink-only intersection of certified enclosures of the same child box;
    /// a mismatched, NaN or disjoint entry keeps its baseline. Observable:
    /// `[cgan-child-seed] armed …` and the per-batch `[cgan-child-seed] batch=…`.
    pub(crate) cgan_child_interm_refine_seed_from_parent: Option<bool>,

    /// `#cgan-child-seed`: run up to this many child cascades of one deferred
    /// batch concurrently (`0`/`1` = serial, today's behaviour). A ceiling,
    /// not a grant: each tranche is re-admitted against a per-worker RSS
    /// estimate, the live process memory headroom (MemAvailable-derived when
    /// no cgroup exists) and an absolute byte cap. Scheduling only. Inert
    /// unless `cgan_child_interm_refine_max_secs` is armed.
    pub(crate) cgan_child_interm_refine_parallel: Option<usize>,

    /// `#cgan-child-seed`: hand a popped domain's own refined intermediate map
    /// to the fresh exact-domain clip instead of the root map alone. Absent
    /// keeps today's clip. Inert unless `cgan_child_interm_refine_max_secs` is
    /// armed (no refined map exists otherwise).
    pub(crate) cgan_child_interm_refine_fresh_clip_map: Option<bool>,

    /// `#cgan-child-zono` (V2): certified per-child DeepZ zonotope
    /// intermediates in the input-split deferred rebound. Each deferred child
    /// runs the shared certified f64 zonotope core over its own box, every
    /// ReLU's pre-activation box intersected with the inherited map (root
    /// CROWN map, then the domain's own override); the refreshed boxes are
    /// merged shrink-only into the map the batched CROWN rebound and the fresh
    /// exact-domain clip consume. Absent (or `enabled: false`) keeps the
    /// historical lane byte-identically. Observable: `[cgan-child-zono]
    /// armed=1 …` once, then the per-batch `[cgan-child-zono] armed=1
    /// children=… installed=…` line, plus a `cgan_child_zono` flight note.
    pub(crate) cgan_child_zono_refresh: Option<CganChildZonoRefreshPreset>,

    /// Consecutive non-improving α iterations tolerated before optimization
    /// stops early. Maps 1:1 to `AlphaCrownConfig::early_stop_patience` and to
    /// α,β-CROWN's `early_stop_patience` (`optimized_bounds.py:75-77`). Unset
    /// keeps the reference default of 10, so every preset that does not name
    /// this key is byte-identical.
    ///
    /// WHY it needs a preset path: `no_improve_iters` starts accumulating at
    /// iteration 0, where the bound is still the CROWN init (the first α step
    /// is applied at the END of iteration 0), so that iteration contributes a
    /// structurally-zero improvement. On a plateaued run the counter therefore
    /// reaches 10 at iteration 9 and the loop breaks — a preset asking for
    /// `iterations: 20` silently gets 10 unless it also raises the patience.
    /// The two knobs are not independent. (The counter RESETS on any iteration
    /// improving by more than `tolerance`, so this bites plateaus, not every
    /// run.)
    ///
    /// `0` is passed through unchanged: it is the reference's own "stop at the
    /// first non-improving iteration" value, not a sentinel for "unset".
    pub(crate) early_stop_patience: Option<usize>,

    /// #inert-ascent-exit: retire the DAG α ascent after this many CONSECUTIVE
    /// iterations whose best improvement was EXACTLY zero. Maps 1:1 to
    /// `AlphaCrownConfig::inert_ascent_exit_iters`; the historical
    /// `early_stop_patience` rule keeps running alongside and whichever fires
    /// first wins.
    ///
    /// MEASURED (2026-09-12, ml4acopf 118_ieee-linear-nonresidual/prop2): on an
    /// IBP-grade root frame (`lower_sum = -8645509120`) the ascent logged
    /// `best_impr = 0.000e0` from iter 5 and still ran to `Converged at
    /// iteration 10` — ten iterations buying nothing, which blew the 114 s
    /// warmup cap and (unarmed) abandoned BaB with 448 s of 600 left.
    ///
    /// SOUND by construction: the exit takes the same `should_save_best` route
    /// as ordinary convergence and returns the already-certified elementwise
    /// best; stopping sooner is looser-or-equal, never a minted bound.
    ///
    /// Must be `>= 1`; `0` is REJECTED at preset load (`apply.rs` bails).
    /// Absent (the shipped state on every yaml) = dark = the ten-iteration
    /// patience, byte-identical. Observable: `[alpha-ascent] armed=<0|1> ...
    /// exit=<converged|inert|budget|...>`, printed on both arms.
    pub(crate) inert_ascent_exit_iters: Option<usize>,

    /// #vit-crown-interm: whole-second wall cap on the `fix_interm_bounds:
    /// false` reference route. Maps 1:1 to
    /// `AlphaCrownConfig::crown_ibp_reference_cap_secs`; inert unless
    /// `fix_interm_bounds: false` is also set (a config that carries `true`, as
    /// every input-split warm-start child does, resolves no cap).
    ///
    /// WHY: that route's per-node CROWN-IBP collection cost 46-50 s of vit's
    /// 100 s budget on `ibp_3_3_8` and turned `ibp_3_3_8_7322` (decided in ~10 s
    /// with plain-IBP intermediates) into a timeout
    /// (`docs/VIT_INTERM_FLOOR_CONVERSION_2026-08-08.md` §5). A preset key cannot
    /// scope per model, so the admission is a COST cap instead: targets the cap
    /// does not reach keep their certified IBP box (the map is always complete,
    /// never an error), the loop reuses that map instead of re-collecting it,
    /// and the post-loop all-node alpha pass stops at the same cap.
    ///
    /// SOUND by construction: every published box is a completed CROWN
    /// enclosure intersected with its IBP box, or that IBP box.
    ///
    /// Must be in `1..=3600`; anything else is REJECTED at preset load. Absent
    /// (every shipped yaml) = the uncapped historical route, byte-identical.
    /// Observable: `[ref-map] stage=collect … route=fix_interm … cap_secs=N` and
    /// `[ref-map] stage=publish … source=artifact`.
    pub(crate) crown_ibp_reference_cap_secs: Option<u64>,

    /// #vit-margin-alpha (B1–B3): make the root α loss the property's margin
    /// C-rows. Typed form of the former env-only `NY_ROOT_ALPHA_MARGIN_GRADIENT`
    /// child gate (lever `ROOT_ALPHA_MARGIN_GRADIENT`); inert unless the parent
    /// `root_alpha_margin` is also armed, since the parent supplies the rows.
    ///
    /// On a root whose optimizer state is ReLU-only the resident joint-adjoint
    /// lane keeps its historical behavior. On a root carrying bilinear α state
    /// (every vit_2023 attention product) the certified C-row margin ascent
    /// runs and publishes ONE fresh certified margin bound that the
    /// multi-objective root intersects (`[alpha-margin]`,
    /// `[alpha-margin-publish]`).
    ///
    /// SOUND: every selectable relaxation is valid and the published bound is
    /// a certified pass; the root merge is identity-checked and all-or-nothing.
    /// No shipped yaml sets this key.
    pub(crate) root_alpha_margin_gradient: Option<bool>,

    /// #vit-margin-alpha (B4): inside the certified C-row margin ascent, also
    /// optimize the bilinear McCormick interpolation parameters of every
    /// `BilinearCrown` / var×var `MatMul` node (lever `ROOT_ALPHA_MATMUL`).
    /// Inert unless the margin lane runs. No shipped yaml sets this key.
    pub(crate) matmul_alpha: Option<bool>,

    /// #vit-margin-alpha-complex (P6): let the certified C-row margin ascent
    /// take a root that also carries MulBinary / Reciprocal α state (the
    /// fixed-index softmax, `softmax: complex_fixed_index`), with those two
    /// families' α FROZEN at their DAG-init values for the whole lane
    /// (`AlphaCrownConfig::margin_alpha_freeze_mul_reciprocal`). Without it the
    /// lane refuses such roots (`[alpha-margin] armed=0
    /// reason=unsupported_alpha_family`).
    ///
    /// SOUND: any fixed relaxation is valid and the published bound is one
    /// fresh certified pass with exactly the committed state. Inert unless the
    /// margin lane is requested. Observable: `[alpha-margin-frozen] armed=1 …`
    /// and `[alpha-margin] armed=1 … frozen_families=…`. No shipped yaml sets
    /// this key.
    pub(crate) margin_alpha_freeze_mul_reciprocal: Option<bool>,

    /// #vit-root-alpha (ENGINE_PROGRAM §2 V2): which search the certified C-row
    /// margin lane runs (`AlphaCrownConfig::margin_alpha_search`): `ascent`
    /// (the default, today's certified ascent) or `joint_f32` (the injected
    /// untrusted joint search over the lane's self-checked graph copy — ReLU
    /// slopes, attention McCormick planes, fixed-index softmax MulBinary
    /// planes — clamped at an ingest gate, certified ONCE, kept only when
    /// strictly better, published as the intersection with the incumbent).
    ///
    /// SOUND: every published number is a certified fold of a clamped, valid
    /// state; the search only chooses which relaxation is certified. Inert
    /// unless the margin lane runs. Observable: `[bbe] phase=vit_root …` and
    /// `[alpha-margin] armed=1 … exit=joint_f32_…`. No shipped yaml sets this
    /// key.
    pub(crate) margin_alpha_search: Option<ny_propagate::bounds::MarginAlphaSearch>,
}

/// β-CROWN configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct BetaCrownPreset {
    /// Learning rate for α parameters during β-CROWN.
    pub(crate) lr_alpha: Option<f32>,

    /// Learning rate for β parameters.
    pub(crate) lr_beta: Option<f32>,

    /// Learning rate decay factor.
    pub(crate) lr_decay: Option<f32>,

    /// Number of optimization iterations.
    /// Supports both "iterations" (ny) and "iteration" (alpha-beta-CROWN).
    #[serde(alias = "iteration")]
    pub(crate) iterations: Option<usize>,

    /// Maximum depth for β optimization.
    pub(crate) max_depth: Option<usize>,
    pub(crate) optimize_disjuncts_separately: Option<bool>, // #4355
}

/// GCP-CROWN cuts configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct CutsPreset {
    /// Enable cutting planes.
    pub(crate) enabled: Option<bool>,

    /// Maximum number of cuts to maintain.
    pub(crate) max_cuts: Option<usize>,

    /// Minimum depth for cut generation.
    pub(crate) min_cut_depth: Option<usize>,

    /// Enable near-miss cut generation.
    pub(crate) near_miss: Option<bool>,

    /// Near-miss margin threshold.
    pub(crate) near_miss_margin: Option<f32>,

    /// Enable proactive cut generation (BICCOS-lite).
    pub(crate) proactive: Option<bool>,

    /// Maximum proactive cuts.
    pub(crate) max_proactive: Option<usize>,
}

/// INVPROP configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct InvpropPreset {
    /// Node names to apply output constraints to.
    pub(crate) apply_output_constraints_to: Option<Vec<String>>,

    /// Share ny parameters.
    pub(crate) share_gammas: Option<bool>,
}

/// Clip-and-verify configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ClipPreset {
    /// Enable relaxed clipping.
    pub(crate) relaxed: Option<bool>,

    /// Relaxed clipping iterations.
    pub(crate) relaxed_iterations: Option<usize>,

    /// Enable the default-dark, exact-current-domain clip used by the
    /// batch-stack-unsafe grouped input-split route. The engine validates the
    /// required relaxed/reordered/IBP-enhanced lifecycle before execution.
    pub(crate) input_split_fresh_domain_clip: Option<bool>,

    /// Clip type: "relaxed" (default) or "complete" (LP-optimal via Lagrangian dual).
    /// Reference: alpha-beta-CROWN `clip_input_domain.clip_type`
    pub(crate) clip_type: Option<String>,

    /// Fraction of unstable neurons for complete clipping neuron selection.
    /// -1.0 = all neurons (default). Only used with clip_type: complete.
    /// Reference: alpha-beta-CROWN `clip_neuron_selection_type` + `clip_neuron_selection_value`
    pub(crate) neuron_selection_ratio: Option<f32>,
    /// Enable intermediate domain clipping.
    pub(crate) interm_domain: Option<bool>,

    /// Top-k neurons for intermediate clipping.
    pub(crate) interm_topk: Option<usize>,

    /// Apply clipping during α-CROWN.
    pub(crate) in_alpha_crown: Option<bool>,
    /// Enable pruning of infeasible domains.
    pub(crate) prune: Option<bool>,

    /// Use final layer constraints for pruning.
    pub(crate) use_final_layer: Option<bool>,
}

/// Load a preset configuration from a YAML file.
///
/// Unknown keys are ERRORS, not silence (#preset-strict). Dropping an
/// unrecognized YAML path without a word is the structural enabler of the
/// repo's worst bug class: a capability that looks configured but never
/// fires (safenlp's wrapper-only attack lane, cgan's unreachable CZ pipeline,
/// cifar100's never-running alpha). A typo'd
/// or renamed key must fail the run with the exact path, so misconfiguration
/// is loud at load time instead of invisible at scoring time. Every preset
/// shipped under `configs/` is loaded by a test below, so an in-repo key can
/// only become unknown by failing CI first.
/// This is soundness/performance authority, not a best-effort import surface.
pub(crate) fn load_preset(path: &Path) -> Result<PresetConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read preset config: {}", path.display()))?;

    let mut unknown = Vec::new();
    let deserializer = serde_yaml::Deserializer::from_str(&contents);
    let preset: PresetConfig =
        serde_ignored::deserialize(deserializer, |path| unknown.push(path.to_string()))
            .with_context(|| format!("Failed to parse preset YAML: {}", path.display()))?;
    if !unknown.is_empty() {
        bail!(
            "preset {} contains unrecognized key(s): {}. Every key must name a field the \
             verifier actually reads — an ignored key is a capability that silently never \
             fires. Fix the spelling, or remove the key.",
            path.display(),
            unknown.join(", ")
        );
    }

    Ok(preset)
}

/// Resolve the typed `bab.cgan_reclaim` key (#cgan-reclaim).
///
/// WITHOUT THIS READER AND ITS CALL SITE THE WHOLE SET IS DEAD CODE IN
/// COMPETITION, exactly like `NY_EPOCH_DEPTH` and the closure-adaptive ladder
/// before it: `run_instance.sh` exports no `NY_*`, so the env half
/// (`NY_CGAN_RECLAIM`) cannot arm on a scored run. Returning `None` for an
/// unreadable or key-less preset keeps the shipped configs byte-identical.
pub(crate) fn cgan_reclaim_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|path| load_preset(path).ok())
        .and_then(|config| config.bab.cgan_reclaim)
}

/// Resolve the typed `bab.mo_alpha_heuristic` key (#stale-alpha).
///
/// Same reason the reader above exists: without a call site the lever is dead
/// code on a scored run, because `run_instance.sh` exports no `NY_*` values.
pub(crate) fn mo_alpha_heuristic_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|path| load_preset(path).ok())
        .and_then(|config| config.bab.mo_alpha_heuristic)
}

/// Resolve the six `attack.open_disjunct_*` keys (#attack-open-disjuncts)
/// into the engine policy. The ONE mapping: `apply` writes its result into
/// `BetaCrownConfig::open_disjunct_attack`, and `vnncomp::run_and_translate`
/// arms the margin-row root publication from the same result.
///
/// `Ok(None)` when no key is present (the shipped state: the engine default,
/// `restrict: false`, stands). An omitted key takes the engine default; an
/// out-of-range value is a load error, never a silently dark stage.
pub(crate) fn open_disjunct_attack_from_preset(
    preset: &PresetConfig,
) -> Result<Option<ny_propagate::OpenDisjunctAttackConfig>> {
    let attack = &preset.attack;
    if attack.open_disjunct_restrict.is_none()
        && attack.open_disjunct_min_secs.is_none()
        && attack.open_disjunct_max_secs.is_none()
        && attack.open_disjunct_restarts.is_none()
        && attack.open_disjunct_root_wait_secs.is_none()
        && attack.open_disjunct_max_frac.is_none()
    {
        return Ok(None);
    }
    let defaults = ny_propagate::OpenDisjunctAttackConfig::default();
    let policy = ny_propagate::OpenDisjunctAttackConfig {
        restrict: attack.open_disjunct_restrict.unwrap_or(defaults.restrict),
        min_secs: attack.open_disjunct_min_secs.unwrap_or(defaults.min_secs),
        max_secs: attack.open_disjunct_max_secs.unwrap_or(defaults.max_secs),
        restarts: attack.open_disjunct_restarts.unwrap_or(defaults.restarts),
        root_wait_secs: attack
            .open_disjunct_root_wait_secs
            .unwrap_or(defaults.root_wait_secs),
        max_frac: attack.open_disjunct_max_frac.unwrap_or(defaults.max_frac),
    };
    policy
        .validate()
        .map_err(|error| anyhow::anyhow!("#attack-open-disjuncts: {error}"))?;
    Ok(Some(policy))
}

pub(crate) fn build_onnx_load_config(preset: &PresetConfig) -> Result<ny_onnx::OnnxLoadConfig> {
    let flags = resolve_onnx_optimization_flags(preset)?;
    let batch_norm_folding = preset.model.batch_norm_folding.unwrap_or_default();
    let require_authored_float32_initializers = preset
        .model
        .require_authored_float32_initializers
        .unwrap_or(false);
    let config = ny_onnx::OnnxLoadConfig::default()
        .with_optimization_flags(flags)
        .with_batch_norm_folding_policy(batch_norm_folding)
        .with_require_authored_float32_initializers(require_authored_float32_initializers)
        .with_qdq_activation_relaxation(preset.model.qdq_activation_relaxation.unwrap_or(false))
        .with_multi_input_concat(preset.model.multi_input_concat.unwrap_or(false));
    // #soundness-spine M2e: a lane that claims proof authority over the
    // AUTHORED graph must load through the certified lowering (no BN fold
    // with a dropped coefficient enclosure) — the same predicate every
    // certified GraphNetwork path reads, not a private copy of it.
    if preset.model.forward_linear_spec_alpha == Some(true)
        && (!config.admits_certified_graph_lowering()
            || !config.require_authored_float32_initializers())
    {
        bail!(
            "model.forward_linear_spec_alpha (legacy alias: \
             model.cgan_forward_alpha_surrogate) requires \
             model.batch_norm_folding: preserve_raw and \
             model.require_authored_float32_initializers: true; refusing proof authority \
             without loader-sealed authored weights"
        );
    }
    Ok(config)
}

pub(crate) fn resolve_onnx_optimization_flags(
    preset: &PresetConfig,
) -> Result<Vec<ny_onnx::OnnxOptimizationFlag>> {
    preset
        .model
        .onnx_optimization_flags
        .iter()
        .map(|flag| match normalize_flag_name(flag).as_str() {
            "merge_linear" => Ok(ny_onnx::OnnxOptimizationFlag::MergeLinear),
            _ => anyhow::bail!(
                "unsupported model.onnx_optimization_flags entry '{flag}': ny currently supports only 'merge_linear'"
            ),
        })
        .collect()
}

fn normalize_flag_name(flag: &str) -> String {
    flag.trim().to_ascii_lowercase().replace('-', "_")
}

fn string_or_vec_string<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Single(String),
        Multiple(Vec<String>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::Single(value) => Ok(vec![value]),
        StringOrVec::Multiple(values) => Ok(values),
    }
}

fn head_phase_cover_count<'de, D>(deserializer: D) -> std::result::Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<usize>::deserialize(deserializer)?;
    if value.is_some_and(|count| count > 7) {
        return Err(<D::Error as serde::de::Error>::custom(
            "margin_row.head_phase_cover must be 0 (off) or 1..=7",
        ));
    }
    Ok(value)
}

fn reject_misplaced_optimize_disjuncts_separately<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    // Consume the value first so every YAML type gets the same actionable
    // location error instead of an unrelated bool type error.
    serde::de::IgnoredAny::deserialize(deserializer)?;
    Err(<D::Error as serde::de::Error>::custom(
        "misplaced bab.optimize_disjuncts_separately; use \
         bab.beta_crown.optimize_disjuncts_separately",
    ))
}
