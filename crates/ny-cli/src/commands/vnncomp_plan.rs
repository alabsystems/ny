// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny vnncomp plan` — the I2 explicit-plan printer.
//!
//! Prints what a scored run of the same (category, onnx, vnnlib, budget)
//! would resolve: detected backend/host, loaded-model facts, every setting
//! as `name = value  [source]`, and the I6 nominal budget snapshot. Dynamic
//! elapsed-time/reservation deductions are intentionally not predicted. The
//! printer owns NO decisions — it loads the same inputs the scored path
//! loads (category preset via the same configs-dir seam, model via the same
//! preset-informed loader config), decodes the same schedule-affecting runtime
//! inputs, and hands them to the pure resolver. A routing change therefore
//! shows up here as a reviewed diff before it shows up in a scorecard.
//!
//! Docs: docs/PLAN_RESOLVER_V1_2026-08-01.md.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use crate::plan_resolver::{ModelFacts, ResolvedPlan};

/// Handle `ny vnncomp plan CATEGORY ONNX VNNLIB BUDGET [--configs-dir DIR] [--json]`.
pub(crate) fn handle_vnncomp_plan_command(
    category: &str,
    onnx: &Path,
    vnnlib: &Path,
    budget_secs: u64,
    configs_dir: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let backend = crate::compute_backend::detect();

    // Preset resolution: byte-for-byte the scored path's seam
    // (`handle_vnncomp_command`): explicit flag, then NY_CONFIGS_DIR (still
    // honored because the scored path honors it — the plan predicts THAT
    // run; retiring the env var is I10 work, not the printer's), then
    // auto-derivation from binary/ONNX/cwd ancestors.
    let configs_dir = configs_dir
        .or_else(|| {
            std::env::var_os("NY_CONFIGS_DIR")
                .map(PathBuf::from)
                .filter(|dir| dir.is_dir())
        })
        .or_else(|| {
            let mut starts = Vec::new();
            if let Ok(exe) = std::env::current_exe() {
                starts.push(exe);
            }
            starts.push(fs::canonicalize(onnx).unwrap_or_else(|_| onnx.to_path_buf()));
            if let Ok(cwd) = std::env::current_dir() {
                starts.push(cwd);
            }
            super::vnncomp::auto_derive_configs_dir(&starts)
        });
    let preset_path = configs_dir
        .as_deref()
        .and_then(|dir| super::vnncomp::resolve_preset_path(dir, category));
    let preset = preset_path
        .as_deref()
        .map(crate::preset::load_preset)
        .transpose()
        .with_context(|| format!("loading preset for category '{category}'"))?;

    // Model facts come from the LOADED model through the same loader config
    // the scored run would use (preset model-loading options included), so
    // the facts here are the facts the verifier would see.
    let file_size_bytes = fs::metadata(onnx)
        .with_context(|| format!("reading ONNX metadata: {}", onnx.display()))?
        .len();
    let load_config = preset
        .as_ref()
        .map(crate::preset::build_onnx_load_config)
        .transpose()?
        .unwrap_or_default();
    let model = ny_onnx::load_onnx_with_config(onnx, &load_config)
        .with_context(|| format!("loading ONNX model: {}", onnx.display()))?;
    let facts = ModelFacts::from_loaded_model(&model, file_size_bytes);

    // #fl-phase-budget (I10): the printer predicts the scored run, so it pays
    // the same measured-rate probe under the same scope gate (medium conv
    // class, both pair keys preset-absent) and hands the observation to the
    // same pure resolver.
    let fl_rate = if crate::plan_resolver::fl_rate_scope_applies(&facts, preset.as_ref()) {
        crate::plan_resolver::probe_fl_rate()
    } else {
        None
    };
    let runtime = crate::plan_resolver::PlanRuntimeOverrides::from_env_values(
        std::env::var_os(crate::plan_resolver::PGD_TIME_CAP_DISABLE_ENV).as_deref(),
        std::env::var_os(crate::plan_resolver::DISJUNCTIVE_PGD_SKIP_ENV).as_deref(),
    );
    let plan = crate::plan_resolver::resolve_plan_with_fl_rate_and_runtime(
        &facts,
        budget_secs,
        backend,
        preset.as_ref(),
        fl_rate.as_ref(),
        runtime,
    );

    if json {
        print_json(
            category,
            onnx,
            vnnlib,
            preset_path.as_deref(),
            &facts,
            &plan,
        )?;
    } else {
        print_human(
            category,
            onnx,
            vnnlib,
            preset_path.as_deref(),
            &facts,
            &plan,
        );
    }
    Ok(())
}

fn print_human(
    category: &str,
    onnx: &Path,
    vnnlib: &Path,
    preset_path: Option<&Path>,
    facts: &ModelFacts,
    plan: &ResolvedPlan,
) {
    println!("=== ny vnncomp plan (I2) ===");
    println!("category: {category}");
    println!("onnx:     {}", onnx.display());
    println!(
        "vnnlib:   {}{}",
        vnnlib.display(),
        if vnnlib.exists() { "" } else { "  (MISSING)" }
    );
    match preset_path {
        Some(path) => println!("preset:   {}", path.display()),
        None => println!("preset:   none (auto defaults — see the scored path's warning)"),
    }
    println!("host:     {} [{}]", plan.backend_kind, plan.backend_summary);
    println!();
    println!("model facts:");
    println!("  param_count           = {}", facts.param_count);
    println!("  conv_layers           = {}", facts.conv_layers);
    println!("  max_conv_out_channels = {}", facts.max_conv_out_channels);
    println!("  file_size_bytes       = {}", facts.file_size_bytes);
    println!("  conv_scale            = {}", plan.conv_scale);
    println!();
    // The snapshot-test surface (`render_settings`) IS the printed surface:
    // what the tests pin is exactly what a human reads here.
    println!("settings:");
    for line in plan.render_settings().lines() {
        println!("  {line}");
    }
    println!();
    println!("nominal budget ledger (before dynamic wrapper deductions/elapsed time):");
    println!("  scored_budget_secs  = {}", plan.ledger.scored_budget_secs);
    println!(
        "  nominal_internal_tier_secs = {}",
        plan.ledger.nominal_internal_tier_secs
    );
    println!(
        "  nominal_attack_slice_secs  = {}",
        plan.ledger
            .nominal_attack_slice_secs
            .map_or_else(|| "disabled".to_string(), |secs| format!("{secs:.1}"))
    );
    println!(
        "  root_alpha_cap_secs = {}",
        plan.ledger.root_alpha_cap_secs.map_or_else(
            || "none (initial-bounds fraction governs)".to_string(),
            |secs| { format!("{secs:.0}") }
        )
    );
}

fn print_json(
    category: &str,
    onnx: &Path,
    vnnlib: &Path,
    preset_path: Option<&Path>,
    facts: &ModelFacts,
    plan: &ResolvedPlan,
) -> Result<()> {
    let settings: Vec<serde_json::Value> = plan
        .iter()
        .map(|(name, value, source)| {
            serde_json::json!({ "name": name, "value": value, "source": source })
        })
        .collect();
    let output = serde_json::json!({
        "category": category,
        "onnx": onnx.display().to_string(),
        "vnnlib": vnnlib.display().to_string(),
        "preset_path": preset_path.map(|p| p.display().to_string()),
        "backend": {
            "kind": plan.backend_kind,
            "summary": &plan.backend_summary,
        },
        "model_facts": {
            "param_count": facts.param_count,
            "conv_layers": facts.conv_layers,
            "max_conv_out_channels": facts.max_conv_out_channels,
            "file_size_bytes": facts.file_size_bytes,
            "conv_scale": plan.conv_scale.to_string(),
        },
        "settings": settings,
        "budget_ledger": &plan.ledger,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
