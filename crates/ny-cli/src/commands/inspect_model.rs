// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Model inspection formatting helpers.

use anyhow::{Context, Result};
use ny_onnx::{estimate_model_timing, load_onnx, TensorSpec, TimingProfile, WeightStore};
use serde_json::{json, Value};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tracing::info;

pub(crate) enum InspectOutput {
    Text(String),
    Json(Value),
}

pub(crate) fn render_inspect_output(
    model: &Path,
    native: bool,
    cost: bool,
    json_output: bool,
    timing_profile: Option<&Path>,
) -> Result<InspectOutput> {
    info!("Inspecting model: {}", model.display());

    if timing_profile.is_some() && !cost {
        anyhow::bail!("Timing estimation requires `ny inspect --cost --timing-profile ...`");
    }

    let use_native = native || model.is_dir() || {
        let ext = model
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        matches!(
            ext.as_str(),
            "pt" | "pth" | "bin" | "safetensors" | "gguf" | "mlmodel" | "mlpackage"
        )
    };

    if use_native {
        if timing_profile.is_some() {
            anyhow::bail!(
                "Timing estimation requires an ONNX model; export the model to ONNX and rerun \
                 `ny inspect --cost --timing-profile ...`"
            );
        }
        inspect_native_model(model, cost, json_output)
    } else {
        inspect_onnx_model(model, cost, json_output, timing_profile)
    }
}

fn inspect_native_model(model: &Path, cost: bool, json_output: bool) -> Result<InspectOutput> {
    use ny_onnx::native::NativeModel;

    let native_model = NativeModel::load(model)?;
    let network = &native_model.network;
    let config = &native_model.config;

    if cost {
        anyhow::bail!(
            "Static cost analysis requires an ONNX model with fully-known tensor shapes; \
             export the model to ONNX and rerun `ny inspect --cost`"
        );
    }

    if json_output {
        let output = json!({
            "name": network.name,
            "architecture": format!("{:?}", config.architecture),
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "parameters": network.param_count,
            "weights": native_model.weights.len(),
            "inputs": network.inputs.iter().map(|i| {
                json!({
                    "name": i.name,
                    "shape": i.shape,
                    "dtype": format!("{:?}", i.dtype)
                })
            }).collect::<Vec<_>>(),
            "outputs": network.outputs.iter().map(|o| {
                json!({
                    "name": o.name,
                    "shape": o.shape,
                    "dtype": format!("{:?}", o.dtype)
                })
            }).collect::<Vec<_>>(),
            "layer_count": network.layers.len()
        });
        return Ok(InspectOutput::Json(output));
    }

    let mut lines = vec![
        format!("Network: {}", network.name),
        format!("Architecture: {:?}", config.architecture),
        format!("Hidden dim: {}", config.hidden_dim),
    ];
    if let Some(heads) = config.num_heads {
        lines.push(format!("Attention heads: {}", heads));
    }
    if let Some(layers) = config.num_layers {
        lines.push(format!("Layers: {}", layers));
    }
    lines.push(format!("Parameters: {}", network.param_count));
    lines.push(format!("Weight tensors: {}", native_model.weights.len()));
    push_tensor_specs(&mut lines, "Inputs:", &network.inputs);
    push_tensor_specs(&mut lines, "Outputs:", &network.outputs);
    lines.push(String::new());
    lines.push(format!("Network layers: {}", network.layers.len()));
    push_weight_preview(&mut lines, &native_model.weights);

    Ok(InspectOutput::Text(lines.join("\n")))
}

fn inspect_onnx_model(
    model: &Path,
    cost: bool,
    json_output: bool,
    timing_profile: Option<&Path>,
) -> Result<InspectOutput> {
    let onnx_model = load_onnx(model)?;
    let network = &onnx_model.network;
    let known_shapes = onnx_model.tensor_shapes().len();
    let constant_tensors = onnx_model.constant_tensors().len();
    let cost_result = if cost {
        Some(
            ny_onnx::estimate_model_cost(&onnx_model)
                .context("Failed to estimate static model cost")?,
        )
    } else {
        None
    };
    let timing_result = if let Some(profile_path) = timing_profile {
        let cost_result = cost_result
            .as_ref()
            .context("Timing estimation requires `ny inspect --cost --timing-profile ...`")?;
        let profile = load_timing_profile(profile_path)?;
        Some(
            estimate_model_timing(cost_result, &profile)
                .context("Failed to estimate model timing")?,
        )
    } else {
        None
    };

    if json_output {
        let mut output = json!({
            "name": network.name,
            "parameters": network.param_count,
            "known_tensor_shapes": known_shapes,
            "constant_tensors": constant_tensors,
            "inputs": network.inputs.iter().map(|i| {
                json!({
                    "name": i.name,
                    "shape": i.shape,
                    "dtype": format!("{:?}", i.dtype)
                })
            }).collect::<Vec<_>>(),
            "outputs": network.outputs.iter().map(|o| {
                json!({
                    "name": o.name,
                    "shape": o.shape,
                    "dtype": format!("{:?}", o.dtype)
                })
            }).collect::<Vec<_>>(),
            "layer_count": network.layers.len()
        });
        if let Some(cost) = &cost_result {
            output["cost"] = serde_json::to_value(cost)?;
        }
        if let Some(timing) = &timing_result {
            output["timing"] = serde_json::to_value(timing)?;
        }
        return Ok(InspectOutput::Json(output));
    }

    let mut lines = vec![
        format!("Network: {}", network.name),
        format!("Parameters: {}", network.param_count),
        format!(
            "Known tensor shapes: {} (includes ONNX Runtime inference when available)",
            known_shapes
        ),
        format!("Constant tensors: {}", constant_tensors),
    ];
    push_tensor_specs(&mut lines, "Inputs:", &network.inputs);
    push_tensor_specs(&mut lines, "Outputs:", &network.outputs);
    lines.push(String::new());
    lines.push(format!("Layers: {}", network.layers.len()));
    if let Some(cost) = &cost_result {
        lines.push(String::new());
        lines.push(cost.summary());
    }
    if let Some(timing) = &timing_result {
        lines.push(String::new());
        lines.push(timing.summary());
    }

    Ok(InspectOutput::Text(lines.join("\n")))
}

fn load_timing_profile(path: &Path) -> Result<TimingProfile> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open timing profile {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("Failed to parse timing profile {}", path.display()))
}

fn push_tensor_specs(lines: &mut Vec<String>, label: &str, tensors: &[TensorSpec]) {
    lines.push(String::new());
    lines.push(label.to_string());
    for tensor in tensors {
        lines.push(format!(
            "  {}: {:?} ({:?})",
            tensor.name, tensor.shape, tensor.dtype
        ));
    }
}

fn push_weight_preview(lines: &mut Vec<String>, weights: &WeightStore) {
    lines.push(String::new());
    lines.push("First 10 weight tensors:".to_string());
    for (name, weight) in weights.iter().take(10) {
        lines.push(format!("  {}: shape {:?}", name, weight.shape()));
    }
    if weights.len() > 10 {
        lines.push(format!("  ... and {} more", weights.len() - 10));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ny_onnx::FamilyTimingCalibration;
    use ny_test_utils::{require_model, test_models_dir};
    use tempfile::tempdir;

    use super::{render_inspect_output, InspectOutput, TimingProfile};

    #[test]
    fn render_inspect_output_json_contains_cost_and_timing() {
        let model_path = test_models_dir().join("single_linear.onnx");
        require_model(&model_path);

        let tempdir = tempdir().expect("tempdir should be created");
        let profile_path = tempdir.path().join("timing-profile.json");
        let profile = TimingProfile {
            schema_version: 1,
            profile_name: "unit-test".to_string(),
            backend: "wgpu".to_string(),
            device_info: "unit-test-device".to_string(),
            workspace_slack_bytes: 128,
            families: BTreeMap::from([(
                "dense_mac".to_string(),
                FamilyTimingCalibration {
                    min_effective_flops_per_ns: 10.0,
                    min_effective_bytes_per_ns: 10.0,
                    launch_overhead_ns: 5,
                },
            )]),
        };
        std::fs::write(
            &profile_path,
            serde_json::to_vec_pretty(&profile).expect("profile should serialize"),
        )
        .expect("profile should be written");

        let output =
            render_inspect_output(&model_path, false, true, true, Some(profile_path.as_path()))
                .expect("inspect output should succeed");

        let InspectOutput::Json(output) = output else {
            panic!("expected JSON inspect output");
        };
        assert!(
            output.get("cost").is_some(),
            "expected top-level cost payload"
        );
        assert!(
            output.get("timing").is_some(),
            "expected top-level timing payload"
        );
        assert!(
            output["timing"]["total_time_ns"]
                .as_u64()
                .expect("timing total should be u64")
                > 0,
            "timing total should be positive"
        );
    }
}
