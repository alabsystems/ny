// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use super::{
    resolve_branching, AlphaCrownPreset, BetaCrownPreset, ClipPreset, CutsPreset, NyPgdOrderCompat,
    PhaseBudgetPreset, PresetConfig,
};
use anyhow::{bail, Context, Result};
use ny_propagate::{
    AlphaCrownConfig, BetaCrownConfig, InputClipType, KfsbReduceOp, PgdAlphaMode,
    PgdInitialization, PgdOptimizer, PhaseBudgetConfig, ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedBoundPropMode {
    pub(crate) use_alpha_crown: bool,
    pub(crate) use_forward_bounds: bool,
}

/// Executable initial PGD placement resolved from the preset contract.
///
/// Post-BaB attack policy is intentionally absent: it is owned by independent
/// engine/wrapper fields and must not be inferred from reference `pgd_order`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolvedInitialPgdSchedule {
    Disabled,
    Upfront,
    InputBab,
    /// Reference `after`: no upfront stage; BaB runs first against a deadline that RESERVES
    /// the attack's slice, and the attack spends that slice afterwards.
    ///
    /// Why this matters: with the upfront schedule the attack's share is spent
    /// unconditionally BEFORE branch-and-bound, so on an UNSAT row it hunts a counterexample
    /// that provably cannot exist. On ACAS Xu that is 46s of a 116s budget, and the preset
    /// had to RAISE the fraction to 0.4 to make SAT rows reliable — trading away exactly the
    /// hardest UNSAT rows (prop_2 on 3_3 and 4_2, which NY misses and five other tools solve).
    ///
    /// Deferring inverts the trade: BaB gets `total - reserve` (90s instead of 66s, +36%)
    /// while the attack keeps the same slice it needs for SAT rows, just spent last. A SAT
    /// row still gets its counterexample, only later in the budget.
    Deferred,
}

/// Resolve `attack.pgd_order` without silently substituting a different
/// scheduler.
///
/// NY implements `after` as deferred placement when the selected route has a
/// post-BaB consumer. Shipped presets that intentionally retain the measured
/// historical upfront behavior say so with `ny_pgd_order_compat: upfront`.
/// `middle` remains unimplemented; an uncontracted or malformed schedule is a
/// preset error, which also prevents wrapper attack lanes from running first.
pub(crate) fn resolve_initial_pgd_schedule(
    preset: &PresetConfig,
) -> Result<Option<ResolvedInitialPgdSchedule>> {
    let order = preset.attack.pgd_order.as_deref();
    let compatibility = preset.attack.ny_pgd_order_compat;

    let Some(order) = order else {
        if compatibility.is_some() {
            bail!("attack.ny_pgd_order_compat requires attack.pgd_order to be 'middle' or 'after'");
        }
        return Ok(None);
    };

    match order.to_ascii_lowercase().as_str() {
        "skip" | "none" | "disabled" => {
            if compatibility.is_some() {
                bail!("attack.ny_pgd_order_compat is invalid with disabled pgd_order '{order}'");
            }
            Ok(Some(ResolvedInitialPgdSchedule::Disabled))
        }
        "before" => {
            if compatibility.is_some() {
                bail!(
                    "attack.ny_pgd_order_compat is only valid for unimplemented pgd_order \
                     'middle' or 'after', not 'before'"
                );
            }
            Ok(Some(ResolvedInitialPgdSchedule::Upfront))
        }
        "input_bab" => {
            if compatibility.is_some() {
                bail!(
                    "attack.ny_pgd_order_compat is only valid for unimplemented pgd_order \
                     'middle' or 'after', not 'input_bab'"
                );
            }
            Ok(Some(ResolvedInitialPgdSchedule::InputBab))
        }
        // `after` is now implemented; `middle` (attack interleaved with BaB) still is not.
        "after" => match compatibility {
            Some(NyPgdOrderCompat::Upfront) => Ok(Some(ResolvedInitialPgdSchedule::Upfront)),
            None => Ok(Some(ResolvedInitialPgdSchedule::Deferred)),
        },
        "middle" => match compatibility {
            Some(NyPgdOrderCompat::Upfront) => Ok(Some(ResolvedInitialPgdSchedule::Upfront)),
            None => bail!(
                "attack.pgd_order 'middle' (attack interleaved with BaB) is not implemented by \
                 NY; use 'after' for deferred placement, or add \
                 attack.ny_pgd_order_compat: upfront to keep the historical upfront stage"
            ),
        },
        other => bail!(
            "unknown attack.pgd_order '{other}': supported values are 'before', 'middle', \
             'after', 'input_bab', and 'skip'/'none'/'disabled'; reference 'middle' requires \
             attack.ny_pgd_order_compat: upfront"
        ),
    }
}

/// Apply preset configuration to a BetaCrownConfig.
///
/// CLI flags take precedence over preset values (apply preset first, then CLI).
/// Supports both alpha-beta-CROWN structure (solver: + bab:) and ny (bab: only).
///
/// Returns an error if preset contains an unsupported bound propagation method,
/// unknown branching method, or unknown reduce operation string.
pub(crate) fn apply_preset(config: &mut BetaCrownConfig, preset: &PresetConfig) -> Result<()> {
    apply_preset_values(config, preset)?;
    Ok(())
}

/// Pure semantic validation for a frozen preset.
///
/// This applies every typed preset value to a disposable default config and
/// validates the resulting engine configuration without emitting compatibility
/// or ignored-field warnings. VNN-COMP uses it before any outer verdict lane;
/// the real handler later applies the same values once and emits normal user
/// diagnostics.
pub(crate) fn validate_preset(preset: &PresetConfig) -> Result<()> {
    let mut config = BetaCrownConfig::default();
    apply_preset_values(&mut config, preset)?;
    config.validate()?;
    Ok(())
}

fn apply_preset_values(config: &mut BetaCrownConfig, preset: &PresetConfig) -> Result<()> {
    // #parameter-governance: every search-space axis that claims a typed
    // delivery route must pass its declaration's config validator before this
    // function mutates even a disposable `BetaCrownConfig`.  The runtime gate
    // resolvers deliberately fail closed to their declaration defaults, but a
    // typo'd or out-of-range PRESET must not reach a scored run labelled as a
    // treatment while silently executing that fallback.
    validate_delivered_lever_values(preset)?;

    // Resolve the scheduling contract before mutating config. This matters for
    // VNN-COMP's fail-closed preset snapshot: invalid scheduling must not leave
    // a partially applied verifier configuration.
    let initial_pgd_schedule = resolve_initial_pgd_schedule(preset)?;
    validate_alpha_preset(&preset.solver.alpha_crown, "solver.alpha_crown")?;
    validate_alpha_preset(&preset.bab.alpha_crown, "bab.alpha_crown")?;
    // Validate the effective phase policy BEFORE mutating `config`. In
    // particular, a NaN post-BaB fraction must be a preset error, not a panic
    // later in Duration arithmetic, and an invalid preset must not partially
    // install a different wrapper/engine schedule.
    let mut phase_budget = config.phase_budget.clone();
    apply_phase_budget_preset(&mut phase_budget, &preset.bab.phase_budget);
    // `pgd_order: after` moves the attack's slice from before BaB to after it. Applied here,
    // after the preset's own fractions, so the deferral acts on the values the preset chose.
    // Resolution errors surface from `resolve_initial_pgd_schedule` on the same snapshot.
    if matches!(
        resolve_initial_pgd_schedule(preset)?,
        Some(ResolvedInitialPgdSchedule::Deferred)
    ) {
        defer_attack_budget(&mut phase_budget);
    }
    phase_budget.validate()?;

    apply_solver_and_bab_settings(config, preset)?;
    apply_branching_preset(config, preset)?;

    apply_alpha_preset(&mut config.alpha_config, &preset.solver.alpha_crown);
    apply_alpha_preset(&mut config.alpha_config, &preset.bab.alpha_crown);

    apply_beta_preset(config, &preset.solver.beta_crown);
    apply_beta_preset(config, &preset.bab.beta_crown);

    apply_cuts_preset(config, &preset.bab.cuts);
    apply_clip_preset(config, &preset.bab.clip);
    config.phase_budget = phase_budget;
    apply_attack_preset(config, preset, initial_pgd_schedule)?;
    // General preset: conv_mode. Applied after cuts so auto-mode resolves correctly.
    if let Some(conv_mode) = preset.general.conv_mode {
        config.conv_mode = conv_mode;
    }
    // #dd-zonotope admission overrides (#metaroom-ddzono). Absent fields leave
    // the config `None`, which is byte-identical to today; env knobs keep
    // precedence at the consumption site
    // (`DdZonoConfig::with_admission_overrides`).
    if preset.dd_zonotope.min_input_numel.is_some() {
        config.dd_zonotope_min_input_numel = preset.dd_zonotope.min_input_numel;
    }
    if preset.dd_zonotope.max_k.is_some() {
        config.dd_zonotope_max_k = preset.dd_zonotope.max_k;
    }
    if preset.dd_zonotope.max_generators.is_some() {
        config.dd_zonotope_max_generators = preset.dd_zonotope.max_generators;
    }
    if preset.dd_zonotope.interm_intersect.is_some() {
        config.dd_zonotope_collect_interm = preset.dd_zonotope.interm_intersect;
    }
    // The "parsed but nothing reads it" warnings that used to live here are now
    // the preset/engine CONTRACT registry (`super::contract`), reported ONCE per
    // preset path at LOAD time — before any budget is spent — with the value
    // requested, what the engine does instead, which direction that fails in,
    // and where in the source the request is dropped. Keeping a second copy of
    // that list here is exactly the drift this module is meant to end: there is
    // one list, and `preset::contract_tests` holds it to the shipped presets.
    //
    // share_alphas is deliberately in NEITHER list: ny's graph alpha state
    // already implements the shared semantics alpha-beta-CROWN opts into with
    // share_alphas=True (GraphAlphaState keys one Array1<f32> of alphas per
    // node; bilinear alphas are [4,m,n,k] per MatMul node) — there is no
    // per-spec-row alpha dimension anywhere to share, so the request IS honoured.
    Ok(())
}

/// Validate the typed layer for every compiled `Deliver::PresetKey` axis.
///
/// This is driven by `ny_levers::space`, the same delivery map used by search
/// and the early flight receipt, rather than by another list of promoted keys.
/// `serde_yaml::Value` is intentional: unlike JSON's value serializer it keeps
/// NaN and infinities as numbers, so the declaration can reject them instead of
/// collapsing `Some(NaN)` into the same `null` representation as `None`.
fn validate_delivered_lever_values(preset: &PresetConfig) -> Result<()> {
    // #interval-carrier-preset-quarantine. This field predates the typed-route
    // quarantine and is deliberately left in the deserializer so an old research YAML fails
    // with an explicit safety explanation instead of an opaque unknown-key
    // error. It is NOT a delivery route: the corresponding axis is Unsafe +
    // EnvOnly because this bound path has produced false `unsat` on the live
    // graph walk. Accepting `Some(false)` would be equally misleading—the
    // presence of this key claims a governed scored treatment that no longer
    // exists.
    if preset.solver.const_matmul_interval.is_some() {
        bail!(
            "solver.const_matmul_interval is quarantined and cannot be delivered by a preset; \
             NY_CONST_MATMUL_INTERVAL is Unsafe + EnvOnly after the interval carrier exposed a \
             live false-proof propagation path. Remove the key. Audited armed research must use \
             the explicit environment gate so it cannot enter a scored preset."
        );
    }

    let serialized = serde_yaml::to_value(preset)
        .context("serialize frozen preset for governed-lever validation")?;

    for axis in ny_levers::space::axes()
        .iter()
        .chain(ny_levers::space::unsafe_axes())
    {
        let ny_levers::space::Deliver::PresetKey(key) = axis.deliver else {
            continue;
        };
        let raw = key
            .split('.')
            .try_fold(&serialized, |node, part| node.get(part))
            .with_context(|| {
                format!(
                    "compiled lever {} names missing preset delivery key {key}",
                    axis.name
                )
            })?;
        if raw.is_null() {
            continue;
        }

        let decl = ny_levers::all().get(axis.name).with_context(|| {
            format!(
                "compiled preset delivery {key} names undeclared lever {}",
                axis.name
            )
        })?;

        // The legacy environment selector is deliberately free-form Text:
        // unknown tokens historically select the parallel baseline. The typed
        // search/shipping surface is narrower. Rejecting a typo HERE preserves
        // the legacy parser while preventing `conv_ibp_blocked: seqq` from being
        // recorded as a treatment that actually ran the baseline.
        if decl.name == ny_levers::decls::conv_ibp_blocked::CONV_IBP_BLOCKED.name {
            let Some(token) = raw.as_str() else {
                bail!(
                    "preset delivery `{key}` for {} must be one of `0`, `1`, or `seq`",
                    decl.name
                );
            };
            if !matches!(token, "0" | "1" | "seq") {
                bail!(
                    "preset delivery `{key}` for {} has noncanonical token {token:?}; \
                     expected one of `0`, `1`, or `seq`",
                    decl.name
                );
            }
        }

        // The legacy environment grammar uses `1` and friendly truthy tokens
        // for ALL available workers, accepts arbitrary usize widths, and maps
        // malformed text to serial. It therefore remains a free-form Text
        // declaration. The typed treatment surface must be narrower: these
        // canonical tokens are the exact search domain, and excluding `1`
        // prevents a preset that appears to request one worker from actually
        // selecting every worker.
        if decl.name == ny_levers::decls::margin_row::MARGIN_ROW_PARALLEL.name {
            let Some(token) = raw.as_str() else {
                bail!(
                    "preset delivery `{key}` for {} must be one of `0`, `2`, `4`, `8`, or `16`",
                    decl.name
                );
            };
            if !matches!(token, "0" | "2" | "4" | "8" | "16") {
                bail!(
                    "preset delivery `{key}` for {} has noncanonical token {token:?}; \
                     expected one of `0`, `2`, `4`, `8`, or `16` (legacy `1` means all workers)",
                    decl.name
                );
            }
        }

        // These two typed fields predate registry delivery and historically
        // decline out-of-range values to their lower scheduler layer. Preserve
        // that contract while validating every admitted value against the
        // declaration, and widen the parsed f32 exactly instead of
        // round-tripping it through serde's generic numeric representation.
        let value = if std::ptr::eq(
            decl,
            &raw const ny_levers::decls::margin_row::MARGIN_ROW_RESERVE_MAX_FRAC,
        ) {
            let Some(value) = preset.margin_row.reserve_max_frac else {
                continue;
            };
            if !value.is_finite() || value <= 0.0 || value >= 1.0 {
                continue;
            }
            ny_levers::LeverValue::F64(f64::from(value))
        } else if std::ptr::eq(
            decl,
            &raw const ny_levers::decls::margin_row::MARGIN_ROW_ADAPTIVE_RELEASE_FRAC,
        ) {
            let Some(value) = preset.margin_row.release_frac else {
                continue;
            };
            if !value.is_finite() || value <= 0.0 || value >= 1.0 {
                continue;
            }
            ny_levers::LeverValue::F64(f64::from(value))
        } else {
            delivered_lever_value(key, decl, raw)?
        };
        ny_levers::read_over_config_with(decl, |_| None, Some(value)).map_err(|_| {
            anyhow::anyhow!(
                "preset delivery `{key}` value {raw:?} is not admissible for {} ({:?})",
                decl.name,
                decl.kind
            )
        })?;
    }

    Ok(())
}

/// Project one serialized typed preset field into the declaration's value
/// vocabulary. Type mismatch is a preset-authority error, not an absent key.
fn delivered_lever_value(
    key: &str,
    decl: &'static ny_levers::LeverDecl,
    raw: &serde_yaml::Value,
) -> Result<ny_levers::LeverValue> {
    use ny_levers::{LeverKind, LeverValue};

    let projected = match decl.kind {
        LeverKind::Bool
        | LeverKind::BoolOverride
        | LeverKind::BoolKillOverride
        | LeverKind::TruthyTrimmed
        | LeverKind::TrimmedExactOne
        | LeverKind::Presence => raw.as_bool().map(LeverValue::Bool),
        LeverKind::U64
        | LeverKind::U64Override
        | LeverKind::U64Trimmed
        | LeverKind::UsizeTrimmed
        | LeverKind::UsizeTrimmedUnicodeOwned
        | LeverKind::UsizeTrimmedOverride { .. }
        | LeverKind::UsizeOverride { .. }
        | LeverKind::Usize { .. } => raw.as_u64().map(LeverValue::U64),
        LeverKind::F32Positive | LeverKind::F32NonNegative => raw
            .as_f64()
            .map(|value| LeverValue::F64(f64::from(value as f32))),
        LeverKind::F64Open { .. }
        | LeverKind::F64OpenOverride { .. }
        | LeverKind::F64OpenUnicodeOwned { .. }
        | LeverKind::F64ClosedTrimmed { .. } => raw.as_f64().map(LeverValue::F64),
        LeverKind::Secs => raw.as_f64().map(LeverValue::Secs),
        LeverKind::Enum(_) | LeverKind::LayeredEnum { .. } => {
            raw.as_str().map(|text| LeverValue::Text(text.to_owned()))
        }
        LeverKind::Text => raw
            .as_str()
            .map(str::to_owned)
            // Legacy raw-predicate declarations retain Text parsing while the
            // typed surface is intentionally Boolean. Project that canonical
            // answer to the exact token the read-site predicate represents.
            .or_else(|| {
                raw.as_bool()
                    .map(|on| if on { "1" } else { "0" }.to_owned())
            })
            .map(LeverValue::Text),
    };

    projected.with_context(|| {
        format!(
            "preset delivery `{key}` value {raw:?} has the wrong type for {} ({:?})",
            decl.name, decl.kind
        )
    })
}

/// Effective typed `alpha_zero_yield_frac` supplied by one validated preset.
///
/// Preset application visits `solver.alpha_crown` first and
/// `bab.alpha_crown` second, and an absent field leaves the previously applied
/// value unchanged. Keep the receipt adapter on that exact precedence instead
/// of rediscovering it from the YAML path.
pub(crate) fn effective_alpha_zero_yield_frac(preset: &PresetConfig) -> Option<f64> {
    preset
        .bab
        .alpha_crown
        .alpha_zero_yield_frac
        .or(preset.solver.alpha_crown.alpha_zero_yield_frac)
}

pub(crate) fn resolve_use_alpha_from_bound_prop_method(
    method: Option<&str>,
) -> Result<Option<bool>> {
    Ok(resolve_bound_prop_mode(method)?.map(|mode| mode.use_alpha_crown))
}

pub(crate) fn resolve_bound_prop_mode(
    method: Option<&str>,
) -> Result<Option<ResolvedBoundPropMode>> {
    let Some(method) = method.map(str::trim).filter(|method| !method.is_empty()) else {
        return Ok(None);
    };

    match method.to_ascii_lowercase().as_str() {
        "crown" => Ok(Some(ResolvedBoundPropMode {
            use_alpha_crown: false,
            use_forward_bounds: false,
        })),
        "alpha-crown" | "alpha_crown" => Ok(Some(ResolvedBoundPropMode {
            use_alpha_crown: true,
            use_forward_bounds: false,
        })),
        "forward+backward" | "forward+crown" => Ok(Some(ResolvedBoundPropMode {
            use_alpha_crown: false,
            use_forward_bounds: true,
        })),
        _ => bail!(
            "unsupported solver.bound_prop_method '{method}': ny currently supports only 'crown', 'alpha-crown', 'forward+backward', and 'forward+crown' preset modes"
        ),
    }
}

fn apply_solver_and_bab_settings(
    config: &mut BetaCrownConfig,
    preset: &PresetConfig,
) -> Result<()> {
    let bab = &preset.bab;
    let solver = &preset.solver;

    if let Some(mode) = resolve_bound_prop_mode(solver.bound_prop_method.as_deref())? {
        config.use_alpha_crown = mode.use_alpha_crown;
        config.use_forward_bounds = mode.use_forward_bounds;
    }

    if let Some(batch_size) = solver.batch_size {
        config.batch_size = batch_size;
    }
    if let Some(build_batch_size) = solver.build_batch_size {
        config.build_batch_size = Some(build_batch_size);
    }
    if let Some(batch_size) = bab.batch_size {
        config.batch_size = batch_size;
    }
    if let Some(crown_backward_layers) = bab.crown_backward_layers {
        config.crown_backward_layers = Some(crown_backward_layers);
    }
    if let Some(timeout) = bab.timeout {
        config.timeout = Duration::from_secs(timeout);
    }
    if let Some(max_domains) = bab.max_domains {
        config.max_domains = max_domains;
    }
    if let Some(max_queue_bytes) = bab.max_queue_bytes {
        config.max_queue_bytes = max_queue_bytes; // #ml4acopf-bab-queue-mem
    }
    if let Some(max_depth) = bab.max_depth {
        config.max_depth = max_depth;
    }
    if let Some(interm_transfer) = bab.interm_transfer {
        config.enable_interm_transfer = interm_transfer;
    }
    if let Some(enabled) = bab.root_interm_cuda_factory {
        config.root_interm_cuda_factory = enabled;
    }
    if let Some(enabled) = bab.mo_cuda_factory_engine_handoff {
        config.mo_cuda_factory_engine_handoff = enabled;
    }
    if let Some(enabled) = bab.mo_cuda_bounded_shared_executor {
        config.mo_cuda_bounded_shared_executor = enabled;
    }
    if let Some(enabled) = bab.root_crown_interm_dense_head {
        config.root_crown_interm_dense_head = enabled;
    }
    if let Some(max_secs) = bab.root_crown_interm_max_secs {
        config.root_crown_interm_max_secs = max_secs;
    }
    if let Some(max_dim) = bab.root_crown_interm_max_dim {
        config.root_crown_interm_max_dim = max_dim;
    }
    if let Some(enabled) = bab.root_comprehensive_gpu_interm {
        config.root_comprehensive_gpu_interm = enabled;
    }
    // #bab-floor: carried as Option so "absent" stays distinguishable from
    // "0.0". Absent means the arbitration does not exist; 0.0 is the explicit
    // kill switch that keeps the other two parseable.
    if let Some(frac) = bab.root_bab_reserve_frac {
        config.root_bab_reserve_frac = Some(frac);
    }
    if let Some(frac) = bab.root_spec_frac {
        config.root_spec_frac = Some(frac);
    }
    if let Some(frac) = bab.root_alpha_frac {
        config.root_alpha_frac = Some(frac);
    }
    if let Some(chunks) = bab.root_comprehensive_gpu_interm_chunks {
        config.root_comprehensive_gpu_interm_chunks = chunks;
    }
    if let Some(enabled) = bab.root_sparse_interm_crown {
        config.root_sparse_interm_crown = enabled;
    }
    if let Some(max_secs) = bab.root_sparse_interm_crown_max_secs {
        config.root_sparse_interm_crown_max_secs = max_secs;
    }
    if let Some(max_dim) = bab.root_sparse_interm_crown_max_dim {
        config.root_sparse_interm_crown_max_dim = max_dim;
    }
    if let Some(max_rows) = bab.root_sparse_interm_crown_max_rows {
        config.root_sparse_interm_crown_max_rows = max_rows;
    }
    if let Some(max_targets) = bab.root_sparse_interm_crown_max_targets {
        config.root_sparse_interm_crown_max_targets = max_targets;
    }
    if let Some(beta_graft) = bab.beta_graft {
        config.mo_beta_graft = beta_graft; // #mo-beta-graft
    }
    if let Some(ratio) = solver.min_batch_size_ratio {
        config.min_batch_fill_ratio = ratio;
    }
    if let Some(ratio) = bab.min_batch_size_ratio {
        config.min_batch_fill_ratio = ratio;
    }
    if let Some(depth) = bab.max_split_depth {
        config.max_relu_split_depth = depth;
    }
    // auto_enlarge_batch_size: solver takes precedence, bab overrides (#4303).
    if let Some(auto_enlarge) = solver.auto_enlarge_batch_size {
        config.auto_enlarge_batch_size = auto_enlarge;
    }
    if let Some(auto_enlarge) = bab.auto_enlarge_batch_size {
        config.auto_enlarge_batch_size = auto_enlarge;
    }
    if let Some(patience) = bab.early_stop_patience {
        config.early_stop_patience = patience;
    }
    // Per-node CROWN-IBP time-budget overrides (#4413, #cgan-bn11-budget).
    // Only explicitly-set preset values override; unset keeps the 2.0 s floor
    // and adaptive remaining-budget cap.
    if let Some(floor_secs) = bab.crown_ibp_per_node_floor_secs {
        config.crown_ibp_per_node_floor_secs = Some(floor_secs);
    }
    if let Some(cap_secs) = bab.root_alpha_cap_secs {
        config.root_alpha_cap_secs = Some(cap_secs);
    }
    if let Some(enabled) = bab.root_alpha_phase_checkpoint {
        config.root_alpha_phase_checkpoint = enabled;
    }
    if let Some(iterations) = bab.atomic_root_c_margin_iterations {
        if iterations > ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS {
            bail!(
                "bab.atomic_root_c_margin_iterations must be <= {}, got {}",
                ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS,
                iterations
            );
        }
        config.atomic_root_c_margin_iterations = iterations;
    }
    if let Some(cap_secs) = bab.crown_ibp_per_node_cap_secs {
        config.crown_ibp_per_node_cap_secs = Some(cap_secs);
    }
    if let Some(prefix_cost_weighted) = bab.crown_ibp_prefix_cost_weighted {
        config.crown_ibp_prefix_cost_weighted = prefix_cost_weighted;
    }
    if let Some(frac) = bab.crown_ibp_max_target_share_frac {
        if !frac.is_finite() || frac <= 0.0 || frac >= 1.0 {
            bail!(
                "bab.crown_ibp_max_target_share_frac must be a finite fraction in (0, 1), got {}",
                frac
            );
        }
        config.crown_ibp_max_target_share_frac = Some(frac);
    }

    Ok(())
}

fn apply_branching_preset(config: &mut BetaCrownConfig, preset: &PresetConfig) -> Result<()> {
    let branching = &preset.bab.branching;

    if let Some(branching) = resolve_branching(preset)? {
        config.branching_heuristic = branching.heuristic;
    }
    if let Some(candidates) = branching.candidates {
        config.fsb_candidates = candidates;
    }
    if let Some(ref reduceop) = branching.reduceop {
        config.kfsb_reduce_op = parse_reduce_op(reduceop)?;
    }
    if let Some(candidates) = branching.kfsb_multi_candidates {
        config.kfsb_multi_candidates = Some(candidates);
    }
    if let Some(ref reduceop) = branching.kfsb_multi_reduceop {
        config.kfsb_multi_reduce_op = Some(parse_reduce_op(reduceop)?);
    }
    if let Some(kfsb_multi) = branching.kfsb_multi {
        // #kfsb-multi: preset-scoped opt-in for the wave-batched selector.
        config.use_kfsb_multi_branching = kfsb_multi;
    }
    if let Some(kfsb_cert_reuse) = branching.kfsb_cert_reuse {
        config.kfsb_cert_reuse = kfsb_cert_reuse;
    }
    if let Some(depth2_lookahead) = branching.depth2_lookahead {
        config.depth_two_branch_lookahead = depth2_lookahead;
    }
    if let Some(coeff_thresh) = branching.input_split.sb_coeff_thresh {
        config.input_split_coeff_thresh = coeff_thresh;
    }
    if let Some(touch_zero_score) = branching.input_split.touch_zero_score {
        config.input_split_touch_zero_score = touch_zero_score;
    }
    if let Some(sb_margin_weight) = branching.input_split.sb_margin_weight {
        config.input_split_sb_margin_weight = sb_margin_weight;
    }
    if let Some(sb_sum) = branching.input_split.sb_sum {
        config.input_split_sb_sum = sb_sum;
    }
    if let Some(sb_primary_spec) = branching.input_split.sb_primary_spec {
        config.input_split_sb_primary_spec = Some(sb_primary_spec);
    }
    if let Some(ibp_enhancement) = branching.input_split.ibp_enhancement {
        config.input_split_ibp_enhancement = ibp_enhancement;
    }
    if let Some(conic_objective) = branching.input_split.conic_objective {
        config.input_split_conic_objective = conic_objective;
    }
    if let Some(batch_size) = branching.input_split.conic_queue_refresh_batch_size {
        config.input_split_conic_queue_refresh_batch_size = batch_size;
    }
    if let Some(independent_singletons) = branching.input_split.independent_singleton_disjunction {
        config.input_split_independent_singleton_disjunction = independent_singletons;
    }
    if let Some(stacked_rebound) = branching.input_split.stacked_rebound {
        config.input_split_stacked_rebound = stacked_rebound;
    }
    if let Some(nested_deadline) = branching.input_split.nested_deadline {
        // #dist-shift-5291: typed nested-deadline authority; env
        // NY_INPUT_SPLIT_NESTED_DEADLINE still overrides in both directions
        // at the shared_specs.rs chokepoint.
        config.input_split_nested_deadline = Some(nested_deadline);
    }
    if let Some(child_bound_max_secs) = branching.input_split.child_bound_max_secs {
        // #cgan-dark-tail FIX-1: typed per-child bound wall cap. Config-only
        // (no env override — a new raw env read would raise the ny-levers
        // ratchet). Absent leaves `None` = NO CAP = today's behaviour.
        config.input_split_child_bound_max_secs = Some(child_bound_max_secs);
    }
    if let Some(calibration) = &branching.input_split.child_bound_calibration {
        // #cost-observer: typed self-calibrating per-child allowance. Config-only
        // (no env override — a new raw env read would raise the ny-levers
        // ratchet). Absent leaves `None`, i.e. the constant above governs.
        // Every field is optional; unnamed fields take the SlicePolicy default.
        config.input_split_child_bound_calibration =
            Some(ny_propagate::ChildBoundCalibrationConfig {
                headroom: calibration.headroom,
                max_share_of_remaining: calibration.max_share_of_remaining,
                probe_share: calibration.probe_share,
                min_samples: calibration.min_samples,
                window: calibration.window,
                growth: calibration.growth,
            });
    }
    if let Some(f64_tail) = &branching.input_split.f64_tail {
        // #cgan-f64-tail: typed f64-tail escalation authority; the env levers
        // NY_F64_TAIL / NY_F64_TAIL_BAND still override at the ny-propagate
        // `resolve_f64_tail_settings` chokepoint (both directions for the
        // gate; band env wins when set+valid).
        if let Some(enabled) = f64_tail.enabled {
            config.input_split_f64_tail_enabled = Some(enabled);
        }
        if let Some(band) = f64_tail.band {
            config.input_split_f64_tail_band = Some(band);
        }
        if let Some(max_escalations) = f64_tail.max_escalations {
            config.input_split_f64_tail_max_escalations = Some(max_escalations);
        }
    }
    if let Some(warm_parallel) = branching.input_split.warm_parallel {
        config.input_split_warm_parallel = warm_parallel;
    }
    if let Some(override_parallel) = branching.input_split.override_parallel {
        config.input_split_override_parallel = override_parallel;
    }
    if let Some(sat_escape_branch) = branching.input_split.sat_escape_branch {
        // #nn4sys-seb-dark: preset-scoped opt-in; env NY_SAT_ESCAPE_BRANCH
        // still overrides either way (BetaCrownConfig::sat_escape_branch_armed).
        config.sat_escape_branch = sat_escape_branch;
    }
    if let Some(reorder_bab) = branching.input_split.reorder_bab {
        config.reorder_bab = reorder_bab;
    }
    if let Some(adv_check) = branching.input_split.adv_check {
        config.adv_check = adv_check;
    }
    if let Some(depth) = branching.input_split.depth {
        config.input_split_depth = depth;
    }
    if let Some(alpha_iteration) = branching.input_split.alpha_iteration {
        config.input_split_alpha_iteration = alpha_iteration;
    }
    if let Some(lr_alpha) = branching.input_split.lr_alpha {
        config.input_split_lr_alpha = lr_alpha;
    }
    Ok(())
}

fn apply_attack_preset(
    config: &mut BetaCrownConfig,
    preset: &PresetConfig,
    initial_pgd_schedule: Option<ResolvedInitialPgdSchedule>,
) -> Result<()> {
    if let Some(pgd_restarts) = preset.attack.pgd_restarts {
        config.pgd_restarts = pgd_restarts;
    }
    if let Some(pgd_steps) = preset.attack.pgd_steps {
        config.pgd_steps = pgd_steps;
    }
    if let Some(value) = preset.attack.pgd_exact_batched {
        config.pgd_exact_vjp.prefer_exact_batched = value;
    }
    if let Some(value) = preset.attack.pgd_fast_grad {
        config.pgd_exact_vjp.fast_gradient = value;
    }
    if let Some(value) = preset.attack.pgd_vjp_batch {
        config.pgd_exact_vjp.batch_enabled = value;
    }
    if let Some(value) = preset.attack.pgd_vjp_k {
        config.pgd_exact_vjp.batch_width = Some(value);
    }
    if let Some(value) = preset.attack.pgd_legacy_lanes {
        config.pgd_exact_vjp.legacy_lanes = value;
    }
    if let Some(value) = preset.attack.pgd_odi_steps {
        config.pgd_exact_vjp.odi_steps = value;
    }
    if let Some(value) = preset.attack.pgd_odi_eta {
        config.pgd_exact_vjp.odi_eta = value;
    }
    if let Some(value) = preset.attack.pgd_vjp_steps {
        config.pgd_exact_vjp.wave_steps = Some(value);
    }
    if let Some(value) = preset.attack.pgd_lane_window {
        config.pgd_exact_vjp.lane_window = value;
    }
    if let Some(value) = preset.attack.pgd_exploit {
        config.pgd_exact_vjp.exploit = value;
    }
    if let Some(value) = preset.attack.pgd_hinge_tau {
        config.pgd_exact_vjp.hinge_tau = value;
    }
    if let Some(schedule) = initial_pgd_schedule {
        config.enable_pgd_attack = !matches!(schedule, ResolvedInitialPgdSchedule::Disabled);
    }
    if let Some(restart_when_stuck) = preset.attack.pgd_restart_when_stuck {
        config.pgd_restart_when_stuck = restart_when_stuck;
    }
    if let Some(ref attack_mode) = preset.attack.attack_mode {
        match attack_mode.to_lowercase().as_str() {
            "pgd" => {
                config.pgd_initialization = PgdInitialization::Uniform;
            }
            "diversed_pgd" => {
                config.pgd_initialization = PgdInitialization::Osi;
            }
            "diversed_gama_pgd" => {
                // OSI initialization + the GAMA guidance loss (#1449).
                // Reference: alpha-beta-CROWN `attack_mode: diversed_GAMA_PGD`
                // → `initialization="osi", GAMA_loss=True`
                // (`attack_interface.py:29-35`). Attack-only: candidates are
                // re-validated before any `sat`, never affects soundness.
                config.pgd_initialization = PgdInitialization::Osi;
                config.pgd_gama = true;
            }
            "boundary" => {
                bail!("attack_mode 'boundary' is not supported in ny");
            }
            other => {
                bail!(
                    "unknown attack_mode '{}': supported modes are 'PGD', \
                     'diversed_PGD', and 'diversed_GAMA_PGD'",
                    other
                );
            }
        }
    }
    if let Some(osi_steps) = preset.attack.osi_steps {
        config.pgd_osi_steps = osi_steps;
    }
    if let Some(pgd_lr_decay) = preset.attack.pgd_lr_decay {
        config.pgd_lr_decay = pgd_lr_decay;
    }
    // STE surrogate for Sign layers during attack gradients (#surrogate-sign)
    // and the dense low-effective-dimension sweep pre-phase (#dense-sweep).
    // Both are attack-only: candidates are re-validated before any `sat`.
    if let Some(surrogate_sign_gradient) = preset.attack.surrogate_sign_gradient {
        config.pgd_surrogate_sign_gradient = surrogate_sign_gradient;
    }
    if let Some(dense_low_dim_sweep) = preset.attack.dense_low_dim_sweep {
        config.pgd_dense_low_dim_sweep = dense_low_dim_sweep;
    }
    if let Some(dense_sweep_max_dims) = preset.attack.dense_sweep_max_dims {
        config.pgd_dense_sweep_max_dims = dense_sweep_max_dims;
    }
    if let Some(dense_sweep_points) = preset.attack.dense_sweep_points {
        config.pgd_dense_sweep_points = dense_sweep_points;
    }
    if preset.attack.pgd_alpha_scale.unwrap_or(false) {
        let alpha = preset
            .attack
            .pgd_alpha
            .as_deref()
            .unwrap_or("0.01")
            .parse::<f32>()
            .map_err(|_| {
                anyhow::anyhow!(
                    "attack.pgd_alpha_scale=true requires a numeric attack.pgd_alpha value"
                )
            })?;
        config.pgd_optimizer = PgdOptimizer::SignedGradient;
        config.pgd_alpha_mode = PgdAlphaMode::InputRangeScaled(alpha);
    } else if let Some(alpha) = preset.attack.pgd_alpha.as_deref() {
        if alpha.eq_ignore_ascii_case("auto") {
            config.pgd_alpha_mode = PgdAlphaMode::Auto;
        } else {
            config.pgd_alpha_mode = PgdAlphaMode::Scalar(alpha.parse::<f32>().map_err(|_| {
                anyhow::anyhow!("attack.pgd_alpha must be a numeric value or 'auto', got '{alpha}'")
            })?);
        }
    }
    Ok(())
}

fn apply_alpha_preset(config: &mut AlphaCrownConfig, preset: &AlphaCrownPreset) {
    if let Some(lr_alpha) = preset.lr_alpha {
        config.learning_rate = lr_alpha;
    }
    if let Some(spec_slots) = preset.spec_slots {
        config.alpha_spec_slots = spec_slots;
    }
    if let Some(iterations) = preset.iterations {
        config.iterations = iterations;
    }
    if let Some(lr_decay) = preset.lr_decay {
        config.lr_decay = lr_decay;
    }
    if let Some(start_save_best) = preset.start_save_best {
        config.start_save_best = start_save_best;
    }
    if let Some(full_conv_alpha) = preset.full_conv_alpha {
        config.full_conv_alpha = full_conv_alpha;
    }
    if let Some(every) = preset.joint_interm_alpha_every {
        config.joint_interm_alpha_every = every;
    }
    if let Some(enabled) = preset.alpha_envelope_grad {
        config.alpha_envelope_grad = enabled;
    }
    if let Some(fraction) = preset.reference_refresh_fraction {
        config.reference_refresh_fraction = fraction;
    }
    if let Some(max_secs) = preset.reference_refresh_max_secs {
        config.reference_refresh_max_secs = Some(max_secs);
    }
    if let Some(enabled) = preset.forward_linear_deadline_fallback_to_ibp {
        config.forward_linear_deadline_fallback_to_ibp = enabled;
    }
    // Intermediate-bounds mode. UNSET keeps the built-in default, so a preset
    // that never names the key is byte-identical. `false` buys the O(N²)
    // per-node CROWN-IBP sweep (the `--crown-ibp-intermediates` behaviour);
    // an explicit CLI flag still wins, because `handle_beta_crown_command`
    // applies it AFTER this and only when the flag was actually passed.
    if let Some(fix_interm_bounds) = preset.fix_interm_bounds {
        config.fix_interm_bounds = fix_interm_bounds;
    }
    if let Some(enabled) = preset.cgan_sparse_target_complete_root {
        config.cgan_sparse_target_complete_root = enabled;
    }
    if let Some(max_rows) = preset.cgan_narrow_target_max_rows {
        config.cgan_narrow_target_max_rows = Some(max_rows);
    }
    if let Some(enabled) = preset.cgan_complete_crown_ibp_root {
        config.cgan_complete_crown_ibp_root = enabled;
    }
    // α early-stop patience. UNSET keeps the built-in reference default (10),
    // so a preset that never names the key is byte-identical. NOTE this is the
    // ALPHA knob (`AlphaCrownConfig::early_stop_patience`), distinct from
    // `bab.early_stop_patience` above, which targets the BaB-level
    // `BetaCrownConfig::early_stop_patience`.
    if let Some(patience) = preset.early_stop_patience {
        config.early_stop_patience = patience;
    }
    // #root-alpha-margin delivery: the typed key is how this measured lever reaches a scored
    // run at all (see crates/ny-cli/tests/measured_gate_delivery.rs). The env var still wins.
    if let Some(rank) = preset.root_alpha_margin {
        config.root_alpha_margin = rank;
    }
    // #alpha-zero-yield delivery: same contract. `validate_alpha_preset` has already rejected
    // out-of-range fractions, so this is a plain move; the env var still wins at read time.
    if let Some(frac) = preset.alpha_zero_yield_frac {
        config.alpha_zero_yield_frac = Some(frac);
    }
}

fn validate_alpha_preset(preset: &AlphaCrownPreset, location: &str) -> Result<()> {
    if let Some(fraction) = preset.reference_refresh_fraction {
        if !AlphaCrownConfig::reference_refresh_fraction_is_valid(fraction) {
            bail!(
                "{location}.reference_refresh_fraction must be finite and in [0.01, 1.0], got {fraction}"
            );
        }
    }
    if let Some(fraction) = preset.alpha_zero_yield_frac {
        if !AlphaCrownConfig::alpha_zero_yield_frac_is_valid(fraction) {
            bail!(
                "{location}.alpha_zero_yield_frac must be finite and in (0.0, 0.9), got {fraction}"
            );
        }
    }
    Ok(())
}

fn apply_beta_preset(config: &mut BetaCrownConfig, preset: &BetaCrownPreset) {
    if let Some(lr_alpha) = preset.lr_alpha {
        config.alpha_lr = lr_alpha;
    }
    if let Some(lr_beta) = preset.lr_beta {
        config.beta_lr = lr_beta;
    }
    if let Some(iterations) = preset.iterations {
        config.beta_iterations = iterations;
    }
    if let Some(max_depth) = preset.max_depth {
        config.beta_max_depth = max_depth;
    }
    if let Some(optimize_disjuncts_separately) = preset.optimize_disjuncts_separately {
        config.optimize_disjuncts_separately = optimize_disjuncts_separately;
    }
    if let Some(lr_decay) = preset.lr_decay {
        // Beta lr_decay shares AlphaCrownConfig::lr_decay.
        // In alpha-beta-CROWN, solver.beta-crown.lr_decay overrides solver.alpha-crown.lr_decay
        // when both are set. Apply beta last so it takes precedence.
        config.alpha_config.lr_decay = lr_decay;
    }
}

fn apply_cuts_preset(config: &mut BetaCrownConfig, preset: &CutsPreset) {
    if let Some(enabled) = preset.enabled {
        config.enable_cuts = enabled;
    }
    if let Some(max_cuts) = preset.max_cuts {
        config.max_cuts = max_cuts;
    }
    if let Some(min_cut_depth) = preset.min_cut_depth {
        config.min_cut_depth = min_cut_depth;
    }
    if let Some(near_miss) = preset.near_miss {
        config.enable_near_miss_cuts = near_miss;
    }
    if let Some(near_miss_margin) = preset.near_miss_margin {
        config.near_miss_margin = near_miss_margin;
    }
    if let Some(proactive) = preset.proactive {
        config.enable_proactive_cuts = proactive;
    }
    if let Some(max_proactive) = preset.max_proactive {
        config.max_proactive_cuts = max_proactive;
    }
}

pub(crate) fn apply_clip_preset(config: &mut BetaCrownConfig, preset: &ClipPreset) {
    if let Some(relaxed) = preset.relaxed {
        config.enable_relaxed_clip = relaxed;
    }
    if let Some(relaxed_iterations) = preset.relaxed_iterations {
        config.relaxed_clip_iterations = relaxed_iterations;
    }
    if let Some(fresh_domain_clip) = preset.input_split_fresh_domain_clip {
        config.input_split_fresh_domain_clip = fresh_domain_clip;
    }
    if let Some(ref clip_type) = preset.clip_type {
        config.input_clip_type = match clip_type.to_lowercase().as_str() {
            "complete" => InputClipType::Complete,
            _ => InputClipType::Relaxed,
        };
    }
    if let Some(ratio) = preset.neuron_selection_ratio {
        config.clip_neuron_selection_ratio = ratio;
    }
    if let Some(interm_domain) = preset.interm_domain {
        config.enable_clip_interm_domain = interm_domain;
    }
    if let Some(interm_topk) = preset.interm_topk {
        config.clip_interm_topk = interm_topk;
    }
    if let Some(in_alpha_crown) = preset.in_alpha_crown {
        config.clip_in_alpha_crown = in_alpha_crown;
    }
    if let Some(prune) = preset.prune {
        config.clip_interm_prune = prune;
    }
    if let Some(use_final_layer) = preset.use_final_layer {
        config.clip_interm_use_final_layer = use_final_layer;
    }
}

/// Re-point the attack's budget from BEFORE branch-and-bound to AFTER it.
///
/// `pgd_order: after` is a SCHEDULING statement, and the schedule lives in the phase-budget
/// fractions: `upfront_pgd_fraction` is spent before BaB, `post_bab_pgd_fraction` after. So
/// deferring moves the slice from one to the other. BaB's deadline subtracts
/// the post-BaB share (`phase_budget.rs` `bab_deadline`). Standard,
/// non-relational Sequential dispatch reaches `verify_standard` and owns the
/// engine's internal deferred fallback when no outer route is available; when
/// the VNN-COMP outer route is available it becomes the exclusive owner and
/// internal PGD is disabled. Relational Sequential, Graph
/// (including late Graph upgrades), and MIP-only dispatch reject this
/// compat-free schedule rather than reserving and dropping the time, unless
/// the VNN-COMP wrapper's frozen outer post-BaB route is actually available.
///
/// The attack keeps the SAME slice it was tuned for; only its position moves. On ACAS Xu that
/// turns 46s-before / 66s-BaB into 90s-BaB / 26s-after: unsat rows gain 36% more search, and a
/// sat row still gets its counterexample, just later in the budget.
///
/// `max` rather than assignment so a preset that also names `post_bab_pgd_fraction` keeps
/// whichever reserve is larger; the fraction is clamped downstream.
fn defer_attack_budget(pb: &mut PhaseBudgetConfig) {
    let reserve = pb.upfront_pgd_fraction.max(pb.post_bab_pgd_fraction);
    pb.upfront_pgd_fraction = 0.0;
    pb.post_bab_pgd_fraction = reserve;
}

fn apply_phase_budget_preset(pb: &mut PhaseBudgetConfig, preset: &PhaseBudgetPreset) {
    if let Some(v) = preset.initial_bounds_fraction {
        pb.initial_bounds_fraction = v;
    }
    if let Some(v) = preset.upfront_pgd_fraction {
        pb.upfront_pgd_fraction = v;
    }
    if let Some(v) = preset.reduced_verification_fraction {
        pb.reduced_verification_fraction = v;
    }
    if let Some(v) = preset.disjunctive_pgd_fraction {
        pb.disjunctive_pgd_fraction = v;
    }
    if let Some(v) = preset.disjunctive_precheck_fraction {
        pb.disjunctive_precheck_fraction = v;
    }
    if let Some(v) = preset.mip_min_fraction {
        pb.mip_min_fraction = v;
    }
    if let Some(v) = preset.mip_min_secs {
        pb.mip_min_secs = v;
    }
    if let Some(v) = preset.mip_max_secs {
        pb.mip_max_secs = v;
    }
    if let Some(v) = preset.post_bab_pgd_fraction {
        pb.post_bab_pgd_fraction = v;
    }
    if let Some(v) = preset.demand_sized_post_bab_reserve {
        pb.demand_sized_post_bab_reserve = v;
    }
    if let Some(v) = preset.demand_sized_disjunctive_pgd {
        pb.demand_sized_disjunctive_pgd = v;
    }
    // `vnncomp_post_bab_attack` and `post_bab_reserve_secs` belong to the outer
    // competition wrapper, not the reusable engine schedule. The VNN-COMP
    // router consumes them directly from its immutable PresetConfig snapshot.
    if let Some(v) = preset.attack_extension_fraction {
        pb.attack_extension_fraction = v;
    }
    if preset.disjunctive_pgd_max_secs.is_some() {
        pb.disjunctive_pgd_max_secs = preset.disjunctive_pgd_max_secs;
    }
    if preset.disjunctive_pgd_min_secs.is_some() {
        pb.disjunctive_pgd_min_secs = preset.disjunctive_pgd_min_secs;
    }
    if let Some(v) = preset.disjunctive_pgd_from_phase_start {
        pb.disjunctive_pgd_from_phase_start = v;
    }
    if preset.disjunctive_precheck_max_secs.is_some() {
        pb.disjunctive_precheck_max_secs = preset.disjunctive_precheck_max_secs;
    }
    if preset.disjunctive_pgd_stall_window_fraction.is_some() {
        pb.disjunctive_pgd_stall_window_fraction = preset.disjunctive_pgd_stall_window_fraction;
    }
    if let Some(v) = preset.enforce_mip_handoff {
        pb.enforce_mip_handoff = v;
    }
}

pub(crate) fn parse_reduce_op(op: &str) -> Result<KfsbReduceOp> {
    match op.to_lowercase().as_str() {
        "min" => Ok(KfsbReduceOp::Min),
        "max" => Ok(KfsbReduceOp::Max),
        "mean" => Ok(KfsbReduceOp::Mean),
        _ => anyhow::bail!("Unknown kFSB reduce operation: '{op}'. Use: min, max, or mean"),
    }
}
