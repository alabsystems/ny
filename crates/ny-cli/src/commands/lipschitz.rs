// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny lipschitz`: sound certified global Lipschitz upper bound (NY ext 2).
//!
//! Prints the SOUND exact-rational certificate from
//! `ny_api::lipschitz::certify_upper_bound` next to the optimistic
//! spectral-norm estimate from
//! `ny_api::probabilistic::estimate_lipschitz_from_network`, so the two are
//! never confused: the sound bound fails closed outside the certified
//! Linear/Conv/ReLU fragment, while the estimate silently treats unhandled
//! layers as 1-Lipschitz and only flags itself via `is_sound`.

use anyhow::{Context, Result};
use ny_api::lipschitz::{certify_upper_bound, SoundLipschitz};
use ny_api::probabilistic::estimate_lipschitz_from_network;
use ny_onnx::load_onnx;
use serde_json::{json, Value};
use std::path::Path;

fn sound_certificate_json(sound: &SoundLipschitz) -> Result<Value> {
    let bound_exact = sound
        .bound
        .to_clean_string()
        .context("exact Lipschitz bound is unavailable")?;
    let per_layer = sound
        .per_layer
        .iter()
        .map(|layer| -> Result<Value> {
            let squared_bound_exact = layer.squared_bound.to_clean_string().with_context(|| {
                format!(
                    "exact squared Lipschitz bound is unavailable for layer {}",
                    layer.index
                )
            })?;
            Ok(json!({
                "index": layer.index,
                "layer_type": layer.layer_type,
                "norm_kind": format!("{:?}", layer.norm_kind),
                "squared_bound_exact": squared_bound_exact,
                "bound_approx": layer.bound.to_f64_approx(),
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(json!({
        "bound_exact": bound_exact,
        "bound_approx": sound.bound_approx(),
        "per_layer": per_layer,
    }))
}

/// Handle the `ny lipschitz` command.
pub(crate) fn handle_lipschitz_command(model: &Path, json_output: bool) -> Result<()> {
    let onnx_model = load_onnx(model)
        .with_context(|| format!("Failed to load ONNX model: {}", model.display()))?;
    let network = onnx_model
        .to_propagate_network()
        .context("Failed to build propagation network from ONNX model")?;

    let sound = certify_upper_bound(&network);
    let estimate = estimate_lipschitz_from_network(&network);

    if json_output {
        let sound_json = match &sound {
            Ok(s) => sound_certificate_json(s).unwrap_or_else(|e| {
                json!({
                    "error": format!(
                        "exact certificate serialization failed (fails closed): {e:#}"
                    )
                })
            }),
            Err(e) => json!({ "error": e.to_string() }),
        };
        let estimate_json = match &estimate {
            Ok(e) => json!({
                "value": e.value,
                "is_sound": e.is_sound,
                "unhandled_layers": e.unhandled_layers,
            }),
            Err(e) => json!({ "error": e.to_string() }),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "model": model.display().to_string(),
                "sound_certificate": sound_json,
                "optimistic_estimate": estimate_json,
            }))?
        );
        return Ok(());
    }

    println!("Model: {}", model.display());
    match &sound {
        Ok(s) => match s.bound.to_clean_string() {
            Ok(exact) => {
                println!("SOUND certified Lipschitz upper bound (l2 -> l2):");
                println!(
                    "  bound  ~= {:.6e}  (certified upper bound)",
                    s.bound_approx()
                );
                if exact.len() <= 64 {
                    println!("  bound   = {exact}  (exact rational)");
                }
                println!("  per-layer:");
                for l in &s.per_layer {
                    println!(
                        "    [{:>3}] {:<12} {:?}: bound ~= {:.6e}",
                        l.index,
                        l.layer_type,
                        l.norm_kind,
                        l.bound.to_f64_approx()
                    );
                }
            }
            Err(e) => {
                println!(
                    "SOUND certificate: unavailable (fails closed): \
                     exact rational serialization failed: {e}"
                );
            }
        },
        Err(e) => {
            println!("SOUND certificate: unavailable (fails closed): {e}");
        }
    }
    match &estimate {
        Ok(e) => {
            println!(
                "Optimistic spectral-norm estimate: {} ({})",
                e.value,
                if e.is_sound {
                    "flagged sound".to_string()
                } else {
                    format!("NOT SOUND; unhandled layers: {:?}", e.unhandled_layers)
                }
            );
        }
        Err(e) => println!("Optimistic estimate: unavailable: {e}"),
    }
    Ok(())
}
