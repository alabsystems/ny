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
/// #lever-reachability: PROMOTION IS NOT ARMING. Pins that no shipped yaml sets
/// any key from the 2026-08-23 delivery batch, so every promoted lever still
/// resolves through an empty cell and the batch is byte-identical.
#[cfg(test)]
mod lever_delivery_pin_tests;
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

    /// Exact-instance admission for the independently certified MetaRoom
    /// constrained-zonotope verdict lane (#cz-hybrid).
    #[serde(default)]
    pub(crate) cz_hybrid: CzHybridPreset,
}

/// Fail-closed exact-basename allowlist for the wired MetaRoom CZ lane. Both
/// fields must be present and match the harness arguments; a partial or absent
/// section never engages the lane.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct CzHybridPreset {
    pub(crate) only_onnx: Option<String>,
    pub(crate) only_vnnlib: Option<String>,
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

    /// Clip-and-Verify y-box tightening (`NY_MARGIN_ROW_CLIP`).
    pub(crate) clip: Option<bool>,

    /// Intermediate Clip-and-Verify tightening (`NY_MARGIN_ROW_CLIP_INTERM`).
    pub(crate) clip_interm: Option<bool>,

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
    /// Non-negative finite f32 odd-lane tau-hinge threshold.
    pub(crate) pgd_hinge_tau: Option<f32>,
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
    /// to `UpfrontWrapperRoute::ForcedByPreset` (identical arming to the
    /// environment force), `false` to `DisabledByPreset`. The key was a parsed
    /// no-op until that route read it, which silently reintroduced the exact
    /// measurement/submission divergence described above for every non-wrapper
    /// entry point.
    pub(crate) upfront_attack: Option<bool>,
    /// Whole-second wall cap for a forced or typed-relational upfront attack.
    ///
    /// Typed delivery for `NY_UPFRONT_ATTACK_CAP`. The environment override is
    /// resolved once at scored-run entry and wins when admissible; absence
    /// retains this value or the measured eight-second declaration default.
    pub(crate) upfront_attack_cap_secs: Option<u64>,
    /// Whole-second wall cap for contextual automatic multi-clause-disjunction
    /// admission.
    ///
    /// Typed delivery for `NY_UPFRONT_ATTACK_AUTO_CAP`. This clamp is applied
    /// only on the automatic route; a force or typed relational admission uses
    /// [`Self::upfront_attack_cap_secs`].
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

    /// Context-grouped disjunctive refinement covers on nn4sys
    /// (`NY_NN4SYS_CTX_GROUP`): one shared tree over the hull of bit-identical
    /// clause boxes instead of one tree per box. The cover and disjointness
    /// proofs are FAIL-CLOSED — any doubt keeps the obligation — so it changes
    /// which boxes get bounded, and therefore coverage, without weakening what
    /// a discharged obligation means.
    pub(crate) nn4sys_ctx_group: Option<bool>,

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
    /// Row ceiling for demanded CROWN targets inside the complete cGAN root
    /// cascade (`#cgan-narrow-target-budget`). Absent/None keeps the historical
    /// all-demanded candidate set. Scheduling-only: a declined target retains
    /// the certified forward-linear baseline the cascade was seeded with.
    pub(crate) cgan_narrow_target_max_rows: Option<usize>,

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
