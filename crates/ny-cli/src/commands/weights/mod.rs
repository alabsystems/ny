// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Weight inspection, comparison, and norm analysis commands.
//!
//! Provides CLI handlers for:
//! - `weights info` - Show information about weights in a file
//! - `weights diff` - Compare weights between two files
//! - `weights norms` - Compute per-block weight norms

use anyhow::{Context, Result};
use clap::Subcommand;
use ny_onnx::load_onnx;
use ny_onnx::safetensors::{load_safetensors, safetensors_info};
use serde_json::json;
use std::path::PathBuf;

mod norms;

#[cfg(test)]
mod tests;

/// Weights subcommand actions
#[derive(Subcommand)]
pub(crate) enum WeightsAction {
    /// Show information about weights in a file
    Info {
        /// Path to weights file (ONNX, SafeTensors, PyTorch, CoreML, or GGUF)
        #[arg(short, long)]
        file: PathBuf,

        /// Show detailed per-tensor info
        #[arg(long, default_value_t = false)]
        detailed: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compare weights between two files
    Diff {
        /// Path to first weights file
        #[arg(long)]
        file_a: PathBuf,

        /// Path to second weights file
        #[arg(long)]
        file_b: PathBuf,

        /// Maximum allowed absolute difference (tighter than compare's 0.001
        /// because weights diff checks exact stored tensor values, not propagated
        /// bounds which have inherent approximation error)
        #[arg(short, long, default_value = "1e-5")]
        tolerance: f32,

        /// Show all differing tensors (not just first)
        #[arg(long, default_value_t = false)]
        show_all: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compute per-block weight norms (for sensitivity correlation analysis)
    Norms {
        /// Path to weights file (GGUF, SafeTensors, ONNX, etc.)
        #[arg(short, long)]
        file: PathBuf,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

/// Handle the Weights subcommand
pub(crate) fn handle_weights_command(action: WeightsAction) -> Result<()> {
    match action {
        WeightsAction::Info {
            file,
            detailed,
            json,
        } => {
            let ext = file
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if ext == "safetensors" {
                // SafeTensors file
                let info = safetensors_info(&file).with_context(|| {
                    format!("Failed to read SafeTensors file: {}", file.display())
                })?;

                if json {
                    let output = json!({
                        "format": "safetensors",
                        "file": file.to_string_lossy(),
                        "tensor_count": info.tensor_count,
                        "param_count": info.param_count,
                        "tensors": if detailed {
                            info.tensors.iter().map(|(name, shape, dtype)| {
                                json!({
                                    "name": name,
                                    "shape": shape,
                                    "dtype": dtype,
                                    "elements": shape.iter().product::<usize>()
                                })
                            }).collect::<Vec<_>>()
                        } else {
                            vec![]
                        }
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    println!("Format: SafeTensors");
                    println!("File: {}", file.display());
                    println!("Tensors: {}", info.tensor_count);
                    println!(
                        "Parameters: {} ({:.2}M)",
                        info.param_count,
                        info.param_count as f64 / 1e6
                    );

                    if detailed {
                        println!("\nTensor Details:");
                        for (name, shape, dtype) in &info.tensors {
                            let elements: usize = shape.iter().product();
                            println!(
                                "  {}: {:?} ({}) - {} elements",
                                name, shape, dtype, elements
                            );
                        }
                    }
                }
            } else if ext == "onnx" {
                // ONNX file - extract weight info
                let model = load_onnx(&file)
                    .with_context(|| format!("Failed to load ONNX file: {}", file.display()))?;

                let weight_count = model.weights.len();
                let param_count: usize = model.weights.iter().map(|(_, w)| w.len()).sum();

                if json {
                    let output = json!({
                        "format": "onnx",
                        "file": file.to_string_lossy(),
                        "tensor_count": weight_count,
                        "param_count": param_count,
                        "tensors": if detailed {
                            model.weights.iter().map(|(name, w)| {
                                json!({
                                    "name": name,
                                    "shape": w.shape(),
                                    "dtype": "F32",
                                    "elements": w.len()
                                })
                            }).collect::<Vec<_>>()
                        } else {
                            vec![]
                        }
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    println!("Format: ONNX");
                    println!("File: {}", file.display());
                    println!("Tensors: {}", weight_count);
                    println!(
                        "Parameters: {} ({:.2}M)",
                        param_count,
                        param_count as f64 / 1e6
                    );

                    if detailed {
                        println!("\nTensor Details:");
                        for (name, w) in model.weights.iter() {
                            println!("  {}: {:?} - {} elements", name, w.shape(), w.len());
                        }
                    }
                }
            } else if ext == "pt" || ext == "pth" || ext == "bin" {
                #[cfg(feature = "pytorch")]
                {
                    use ny_onnx::pytorch::pytorch_info;
                    // PyTorch file
                    let info = pytorch_info(&file).with_context(|| {
                        format!("Failed to read PyTorch file: {}", file.display())
                    })?;

                    if json {
                        let output = json!({
                            "format": "pytorch",
                            "file": file.to_string_lossy(),
                            "tensor_count": info.tensor_count,
                            "param_count": info.param_count,
                            "tensors": if detailed {
                                info.tensors.iter().map(|(name, shape, dtype)| {
                                    json!({
                                        "name": name,
                                        "shape": shape,
                                        "dtype": dtype,
                                        "elements": shape.iter().product::<usize>()
                                    })
                                }).collect::<Vec<_>>()
                            } else {
                                vec![]
                            }
                        });
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        println!("Format: PyTorch");
                        println!("File: {}", file.display());
                        println!("Tensors: {}", info.tensor_count);
                        println!(
                            "Parameters: {} ({:.2}M)",
                            info.param_count,
                            info.param_count as f64 / 1e6
                        );

                        if detailed {
                            println!("\nTensor Details:");
                            for (name, shape, dtype) in &info.tensors {
                                let elements: usize = shape.iter().product();
                                println!(
                                    "  {}: {:?} ({}) - {} elements",
                                    name, shape, dtype, elements
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "pytorch"))]
                {
                    anyhow::bail!("PyTorch support not enabled. Rebuild with --features pytorch");
                }
            } else if ext == "mlmodel"
                || ext == "mlpackage"
                || file.to_string_lossy().ends_with(".mlpackage")
            {
                #[cfg(feature = "coreml")]
                {
                    use ny_onnx::coreml::coreml_info;
                    // CoreML file
                    let info = coreml_info(&file).with_context(|| {
                        format!("Failed to read CoreML file: {}", file.display())
                    })?;

                    if json {
                        let output = json!({
                            "format": "coreml",
                            "file": file.to_string_lossy(),
                            "spec_version": info.spec_version,
                            "tensor_count": info.tensor_count,
                            "param_count": info.param_count,
                            "tensors": if detailed {
                                info.tensors.iter().map(|(name, shape, dtype)| {
                                    json!({
                                        "name": name,
                                        "shape": shape,
                                        "dtype": dtype,
                                        "elements": shape.iter().product::<usize>()
                                    })
                                }).collect::<Vec<_>>()
                            } else {
                                vec![]
                            }
                        });
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        println!("Format: CoreML");
                        println!("File: {}", file.display());
                        println!("Spec Version: {}", info.spec_version);
                        println!("Tensors: {}", info.tensor_count);
                        println!(
                            "Parameters: {} ({:.2}M)",
                            info.param_count,
                            info.param_count as f64 / 1e6
                        );

                        if detailed {
                            println!("\nTensor Details:");
                            for (name, shape, dtype) in &info.tensors {
                                let elements: usize = shape.iter().product();
                                println!(
                                    "  {}: {:?} ({}) - {} elements",
                                    name, shape, dtype, elements
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "coreml"))]
                {
                    anyhow::bail!("CoreML support not enabled. Rebuild with --features coreml");
                }
            } else if ext == "gguf" {
                #[cfg(feature = "gguf")]
                {
                    use ny_onnx::gguf::gguf_info;
                    // GGUF file (llama.cpp format)
                    let info = gguf_info(&file)
                        .with_context(|| format!("Failed to read GGUF file: {}", file.display()))?;

                    if json {
                        let output = json!({
                            "format": "gguf",
                            "file": file.to_string_lossy(),
                            "version": info.version,
                            "architecture": info.architecture,
                            "model_name": info.model_name,
                            "tensor_count": info.tensor_count,
                            "param_count": info.param_count,
                            "metadata": info.metadata.iter().map(|(k, v)| json!({"key": k, "value": v})).collect::<Vec<_>>(),
                            "tensors": if detailed {
                                info.tensors.iter().map(|(name, shape, dtype, is_quantized)| {
                                    json!({
                                        "name": name,
                                        "shape": shape,
                                        "dtype": dtype,
                                        "quantized": is_quantized,
                                        "elements": shape.iter().product::<u64>()
                                    })
                                }).collect::<Vec<_>>()
                            } else {
                                vec![]
                            }
                        });
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        println!("Format: GGUF (llama.cpp)");
                        println!("File: {}", file.display());
                        println!("Version: {}", info.version);
                        if let Some(arch) = &info.architecture {
                            println!("Architecture: {}", arch);
                        }
                        if let Some(name) = &info.model_name {
                            println!("Model Name: {}", name);
                        }
                        println!("Tensors: {}", info.tensor_count);
                        println!(
                            "Parameters: {} ({:.2}M)",
                            info.param_count,
                            info.param_count as f64 / 1e6
                        );

                        // Count quantized vs non-quantized
                        let quantized_count = info.tensors.iter().filter(|(_, _, _, q)| *q).count();
                        if quantized_count > 0 {
                            println!(
                                "Quantized tensors: {} (not loadable for diff)",
                                quantized_count
                            );
                        }

                        if !info.metadata.is_empty() {
                            println!("\nMetadata:");
                            for (key, value) in &info.metadata {
                                println!("  {}: {}", key, value);
                            }
                        }

                        if detailed {
                            println!("\nTensor Details:");
                            for (name, shape, dtype, is_quantized) in &info.tensors {
                                let elements: u64 = shape.iter().product();
                                let quant_marker = if *is_quantized { " [Q]" } else { "" };
                                println!(
                                    "  {}: {:?} ({}{}) - {} elements",
                                    name, shape, dtype, quant_marker, elements
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "gguf"))]
                {
                    anyhow::bail!("GGUF support not enabled. Rebuild with --features gguf");
                }
            } else {
                anyhow::bail!("Unsupported file format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .mlmodel, .mlpackage, or .gguf", ext);
            }
        }

        WeightsAction::Diff {
            file_a,
            file_b,
            tolerance,
            show_all,
            json,
        } => {
            // Load weights from both files
            let weights_a = load_weights_from_file(&file_a)?;
            let weights_b = load_weights_from_file(&file_b)?;

            // Compare
            let mut comparisons = Vec::new();
            let mut max_diff = 0.0f32;
            let mut differing_count = 0;

            // Find common tensors
            for (name, tensor_a) in weights_a.iter() {
                if let Some(tensor_b) = weights_b.get(name) {
                    // Compare shapes
                    if tensor_a.shape() != tensor_b.shape() {
                        comparisons.push(json!({
                            "name": name,
                            "status": "shape_mismatch",
                            "shape_a": tensor_a.shape(),
                            "shape_b": tensor_b.shape()
                        }));
                        differing_count += 1;
                        continue;
                    }

                    // Compare values
                    let diff = tensor_a
                        .iter()
                        .zip(tensor_b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);

                    max_diff = max_diff.max(diff);

                    if diff > tolerance {
                        differing_count += 1;
                        comparisons.push(json!({
                            "name": name,
                            "status": "differs",
                            "max_diff": diff,
                            "shape": tensor_a.shape()
                        }));
                    } else if show_all {
                        comparisons.push(json!({
                            "name": name,
                            "status": "match",
                            "max_diff": diff,
                            "shape": tensor_a.shape()
                        }));
                    }
                } else {
                    comparisons.push(json!({
                        "name": name,
                        "status": "missing_in_b"
                    }));
                    differing_count += 1;
                }
            }

            // Check for tensors only in B
            for name in weights_b.keys() {
                if weights_a.get(name).is_none() {
                    comparisons.push(json!({
                        "name": name,
                        "status": "missing_in_a"
                    }));
                    differing_count += 1;
                }
            }

            let is_match = differing_count == 0;

            if json {
                let output = json!({
                    "file_a": file_a.to_string_lossy(),
                    "file_b": file_b.to_string_lossy(),
                    "tolerance": tolerance,
                    "result": if is_match { "match" } else { "differs" },
                    "max_difference": max_diff,
                    "differing_tensors": differing_count,
                    "total_tensors_a": weights_a.len(),
                    "total_tensors_b": weights_b.len(),
                    "comparisons": comparisons
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("File A: {}", file_a.display());
                println!("File B: {}", file_b.display());
                println!("Tolerance: {}", tolerance);
                println!();

                if is_match {
                    println!("Result: MATCH");
                    println!("Max difference: {:.6e}", max_diff);
                } else {
                    println!("Result: DIFFERS");
                    println!("Max difference: {:.6e}", max_diff);
                    println!("Differing tensors: {}", differing_count);
                    println!();

                    for comp in &comparisons {
                        let status = comp["status"].as_str().unwrap_or("");
                        let name = comp["name"].as_str().unwrap_or("");
                        match status {
                            "differs" => {
                                let diff = comp["max_diff"].as_f64().unwrap_or(0.0);
                                println!("  {} - max diff: {:.6e}", name, diff);
                            }
                            "shape_mismatch" => {
                                println!(
                                    "  {} - SHAPE MISMATCH: {:?} vs {:?}",
                                    name, comp["shape_a"], comp["shape_b"]
                                );
                            }
                            "missing_in_a" => {
                                println!("  {} - only in file B", name);
                            }
                            "missing_in_b" => {
                                println!("  {} - only in file A", name);
                            }
                            _ => {}
                        }
                        if !show_all && differing_count > 10 && comparisons.len() > 10 {
                            println!("  ... and {} more", differing_count - 10);
                            break;
                        }
                    }
                }
            }
        }

        WeightsAction::Norms { file, json } => {
            norms::handle_weights_norms(&file, json)?;
        }
    }

    Ok(())
}

/// Load weights from a file (SafeTensors, ONNX, PyTorch, or CoreML)
fn load_weights_from_file(path: &std::path::Path) -> Result<ny_onnx::WeightStore> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check for .mlpackage directory (CoreML)
    if ext == "mlpackage" || (path.is_dir() && path.to_string_lossy().ends_with(".mlpackage")) {
        #[cfg(feature = "coreml")]
        {
            use ny_onnx::coreml::load_coreml;
            return load_coreml(path)
                .map_err(|e| anyhow::anyhow!("Failed to load CoreML package: {}", e));
        }
        #[cfg(not(feature = "coreml"))]
        {
            anyhow::bail!("CoreML support not enabled. Rebuild with --features coreml");
        }
    }

    match ext.as_str() {
        "safetensors" => {
            load_safetensors(path)
                .map_err(|e| anyhow::anyhow!("Failed to load SafeTensors: {}", e))
        }
        "onnx" => {
            let model = load_onnx(path)?;
            Ok(model.weights)
        }
        #[cfg(feature = "pytorch")]
        "pt" | "pth" | "bin" => {
            use ny_onnx::pytorch::load_pytorch;
            load_pytorch(path)
                .map_err(|e| anyhow::anyhow!("Failed to load PyTorch file: {}", e))
        }
        #[cfg(not(feature = "pytorch"))]
        "pt" | "pth" | "bin" => {
            anyhow::bail!("PyTorch support not enabled. Rebuild with --features pytorch")
        }
        #[cfg(feature = "coreml")]
        "mlmodel" => {
            use ny_onnx::coreml::load_coreml;
            load_coreml(path)
                .map_err(|e| anyhow::anyhow!("Failed to load CoreML model: {}", e))
        }
        #[cfg(not(feature = "coreml"))]
        "mlmodel" => {
            anyhow::bail!("CoreML support not enabled. Rebuild with --features coreml")
        }
        #[cfg(feature = "gguf")]
        "gguf" => {
            use ny_onnx::gguf::load_gguf;
            load_gguf(path)
                .map_err(|e| anyhow::anyhow!("Failed to load GGUF file: {}", e))
        }
        #[cfg(not(feature = "gguf"))]
        "gguf" => {
            anyhow::bail!("GGUF support not enabled. Rebuild with --features gguf")
        }
        _ => anyhow::bail!("Unsupported file format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .mlmodel, .mlpackage, or .gguf", ext),
    }
}
