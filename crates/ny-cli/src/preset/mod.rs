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
    apply_preset, effective_alpha_zero_yield_frac, resolve_initial_pgd_schedule,
    resolve_use_alpha_from_bound_prop_method, validate_preset, ResolvedInitialPgdSchedule,
};
pub(crate) use branching::resolve_branching;
pub(crate) use contract::enforce_preset_contract;

#[cfg(test)]
mod alpha_preset_tests;
#[cfg(test)]
mod attack_mode_tests;
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
mod gpu_bab_sidecar_tests;
#[cfg(test)]
mod linearizenn_2024_tests;
#[cfg(test)]
mod model_load_smoke_tests;
#[cfg(test)]
mod phase_budget_tests;
#[cfg(test)]
mod preset_resolution_pin_tests;
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
/// The lane runs AFTER the internal verifier returns unknown/timeout, so by
/// default it lives on leftover budget only and is strictly additive. A
/// category may opt into a RESERVE — seconds held back from the internal
/// verifier — but only where the measured production solve-time
/// distribution shows that tail is genuinely unused, because on a category
/// whose solves crowd the budget wall a reserve trades away real points
/// (measured: a 45 s reserve would forfeit 28 `sat_relu` and 10
/// `cifar100_2024` solves). Default 0 = no reserve.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct MarginRowPreset {
    /// Seconds reserved from the internal verifier for the lane.
    pub(crate) reserve_secs: Option<u64>,

    /// Release the reserve only for the sealed exact open-row allowlist and
    /// route those rows to the internal alpha/beta verifier. Unknown rows keep
    /// the configured reserve. Default `None`/`false` preserves the historical
    /// fixed-reserve policy.
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
    /// Pairs with `reserve_secs: 0`: concurrent + a reserve taxes the verifier
    /// twice (it loses the reserve AND contends), which is the combination
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

    /// #backward-interm: recompute each trunk ReLU's input bounds via the
    /// lane's OWN backward engine through the already-frozen prefix gates,
    /// shrink-only intersected with the forward tableau BEFORE gate
    /// derivation. MEASURED 2026-08-19 (cycles 1-3): cifar deep-band roots
    /// move ~0.18-0.20 (idx_5242 -0.613 -> -0.438, INTO the proven band);
    /// tiny idx_4330 timeout -> UNSAT 3/3; and proving rows get 5-10x
    /// CHEAPER (idx_6659 211 exp/72s -> 19 exp/3.8s; idx_8600 186/45s ->
    /// 35/12s). Shrink-only by construction — a wrong bound costs tightness,
    /// never soundness; parity mode refuses.
    pub(crate) backward_interm: Option<bool>,
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
    /// Restart PGD when a projected step leaves the point unchanged (#4278).
    pub(crate) pgd_restart_when_stuck: Option<bool>,
    /// Force the UPFRONT falsification lane on for every instance of this
    /// category (`#upfront-preset`).
    ///
    /// The lane auto-enables only for MULTI-CLAUSE DISJUNCTIONS. safenlp's specs
    /// are single-constraint (`(assert (<= Y_0 Y_1))`), so the auto rule skips
    /// them and the lane has to be forced — which until now happened ONLY via
    /// `NY_UPFRONT_ATTACK=1` exported from `vnncomp_scripts/run_instance.sh`.
    /// That made the submission path behave differently from every other
    /// invocation: `ny benchmarks run`, `measure_ny_scorecard.sh` and a direct
    /// `ny vnncomp` all missed it. MEASURED on safenlp
    /// medical/hyperrectangle_2096 at the official 20 s budget: `sat` with the
    /// flag, `timeout` without it — a real solve that no measurement path could
    /// see. Putting it in the preset makes the behaviour travel with the
    /// category. `NY_UPFRONT_ATTACK` still overrides (0 kills, 1 forces).
    ///
    /// CONSUMER: `commands::vnncomp::upfront_wrapper_route` — `true` resolves
    /// to `UpfrontWrapperRoute::ForcedByPreset` (identical arming to the
    /// environment force), `false` to `DisabledByPreset`. The key was a parsed
    /// no-op until that route read it, which silently reintroduced the exact
    /// measurement/submission divergence described above for every non-wrapper
    /// entry point.
    pub(crate) upfront_attack: Option<bool>,
    /// Attack mode: "PGD" or "diversed_PGD" (OSI init, #1449).
    pub(crate) attack_mode: Option<String>,
    /// OSI initialization steps; only with `attack_mode: diversed_PGD` (#1449).
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
    /// CONSUMER: `commands::vnncomp::run_and_translate`, which passes it to
    /// `try_sign_space_falsify` as the lever's config layer.
    pub(crate) bnn_sign_space: Option<bool>,

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
    /// CONSUMER: `commands::vnncomp::run_and_translate`, which passes it to
    /// `try_ste_pgd_falsify` as the lever's config layer.
    pub(crate) bnn_ste_pgd: Option<bool>,
}

/// Explicit NY compatibility choices for unimplemented reference PGD orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NyPgdOrderCompat {
    /// Preserve NY's historical initial/upfront attack placement.
    Upfront,
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
    /// phase-split racing mode (designs/scip.md Phase C); parsed for
    /// alpha-beta-CROWN key compatibility (`solver.mip.parallel_solvers`).
    pub(crate) parallel_solvers: Option<usize>,
}

/// Solver configuration (`solver:` in alpha-beta-CROWN).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct SolverPreset {
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
}

/// Branch-and-bound (BaB) configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct BabPreset {
    /// Batch size for parallel domain processing.
    pub(crate) batch_size: Option<usize>,

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

    /// Early stopping patience (iterations without improvement).
    pub(crate) early_stop_patience: Option<usize>,

    /// Floor (seconds) for CROWN-IBP collectors' per-node time budget
    /// (#4413, #cgan-bn11-budget). Unset keeps the built-in 2.0 s constant.
    pub(crate) crown_ibp_per_node_floor_secs: Option<f64>,

    /// Explicit base cap (seconds) on CROWN-IBP collectors' per-node time
    /// budget (#4413, #cgan-bn11-budget). Unset selects an adaptive cap from
    /// the remaining collection budget (25%, clamped to 12–600 seconds).
    /// Explicit caps are dimension-scaled above 28,800 rows.
    pub(crate) crown_ibp_per_node_cap_secs: Option<f64>,
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

    /// Enable the one-time, structurally selected dense-head CROWN intermediate
    /// shrink-intersect at the graph root (#cifar-head-crown).
    pub(crate) root_crown_interm_dense_head: Option<bool>,

    /// Wall-clock cap in seconds for `root_crown_interm_dense_head`.
    pub(crate) root_crown_interm_max_secs: Option<u64>,

    /// Maximum selected dense-head pre-activation width.
    pub(crate) root_crown_interm_max_dim: Option<usize>,

    /// Run the comprehensive all-target sound-GPU root intermediate sweep.
    ///
    /// DELIVERY: the scored entry point exports exactly one `NY_*` variable, so
    /// the env lever this was measured through is dead in competition. This typed
    /// key is how the result reaches a scored run
    /// (`crates/ny-cli/tests/measured_gate_delivery.rs`).
    pub(crate) root_comprehensive_gpu_interm: Option<bool>,

    /// `#bab-floor`: BaB's share of the root window, subtracted before any root
    /// phase sizes itself.
    ///
    /// DELIVERY: the scored entry point exports exactly one `NY_*` variable, so
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

    /// Phase-level time budget overrides (#2206).
    /// Only explicitly-set fields override `PhaseBudgetConfig` defaults.
    #[serde(default)]
    pub(crate) phase_budget: PhaseBudgetPreset,
}

/// Phase-level time budget preset overrides (#2206 Packet E).
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

    /// Number of candidates for FSB/kFSB.
    pub(crate) candidates: Option<usize>,

    /// Reduce operation for kFSB: "min", "max", "mean".
    pub(crate) reduceop: Option<String>,

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

/// Input-splitting SB configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputSplitPreset {
    /// Enable input splitting for alpha-beta-CROWN-compatible preset imports.
    pub(crate) enable: Option<bool>,

    /// Opt into sequential single-objective DomainList searches for a canonical
    /// exactly-two-singleton disjunction. NY-specific and default false.
    pub(crate) independent_singleton_disjunction: Option<bool>,

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
            None => self.filter.is_some() || self.filter_beta.is_some(),
        }
    }
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
    /// at model load (vit_2023). Analog of alpha-beta-CROWN's
    /// `bound_opts={'softmax': 'complex'}` (vnncomp23 vit winner recipe,
    /// `custom_adhoc_tuning.py`). Any other value warns and keeps the default
    /// direct-LSE softmax relaxation. Runtime kill-switch:
    /// `NY_NO_SOFTMAX_COMPLEX=1`.
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
    /// point exports exactly one `NY_*` variable, so an env-only setting cannot
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
/// Unknown keys are ERRORS, not silence (#preset-strict). Until 2026-07-30
/// this parser dropped unrecognized YAML paths without a word, which is the
/// structural enabler of the repo's worst bug class: a capability that
/// looks configured but never fires (safenlp's wrapper-only attack lane,
/// cgan's unreachable CZ pipeline, cifar100's never-running alpha). A typo'd
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

pub(crate) fn build_onnx_load_config(preset: &PresetConfig) -> Result<ny_onnx::OnnxLoadConfig> {
    let flags = resolve_onnx_optimization_flags(preset)?;
    let batch_norm_folding = preset.model.batch_norm_folding.unwrap_or_default();
    let require_authored_float32_initializers = preset
        .model
        .require_authored_float32_initializers
        .unwrap_or(false);
    if preset.model.forward_linear_spec_alpha == Some(true)
        && (batch_norm_folding != ny_onnx::BatchNormFoldingPolicy::PreserveRaw
            || !require_authored_float32_initializers)
    {
        bail!(
            "model.forward_linear_spec_alpha (legacy alias: \
             model.cgan_forward_alpha_surrogate) requires \
             model.batch_norm_folding: preserve_raw and \
             model.require_authored_float32_initializers: true; refusing proof authority \
             without loader-sealed authored weights"
        );
    }
    Ok(ny_onnx::OnnxLoadConfig::default()
        .with_optimization_flags(flags)
        .with_batch_norm_folding_policy(batch_norm_folding)
        .with_require_authored_float32_initializers(require_authored_float32_initializers))
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
