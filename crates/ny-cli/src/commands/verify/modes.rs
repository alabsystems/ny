// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Layer-by-layer and block-wise verification execution modes.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::{
    compute_model_hash, BlockProgress, GraphNetwork, LayerProgress, VerificationCheckpoint,
};
use ny_tensor::BoundedTensor;
use std::path::Path;
use std::time::Instant;

use super::super::JsonCliError;
use super::model_load::VerifiableNetwork;
use crate::BackendArg;

/// Run layer-by-layer mode, streaming per-node bounds.
pub(crate) fn run_layer_by_layer(
    network: &VerifiableNetwork,
    spec_input_shape: Option<&[usize]>,
    epsilon: f32,
    progress: bool,
    progress_json: bool,
    json: bool,
) -> Result<()> {
    let effective_shape = spec_input_shape.ok_or_else(|| {
        JsonCliError::new(
            "missing_input_shape",
            "Layer-by-layer mode requires input_shape in verification spec.",
        )
    })?;
    let input_data = ArrayD::zeros(IxDyn(effective_shape));
    let input_bounded = BoundedTensor::from_epsilon(input_data, epsilon)?;

    match network {
        VerifiableNetwork::Sequential(_net) => {
            let message =
                "Layer-by-layer mode is only supported for native models (GraphNetwork). \
                Hint: use --native flag with PyTorch/SafeTensors/GGUF models.";
            Err(JsonCliError::new("unsupported_model_format", message).into())
        }
        VerifiableNetwork::Graph(graph) => run_layer_by_layer_graph(
            graph,
            &input_bounded,
            epsilon,
            progress,
            progress_json,
            json,
        ),
    }
}

fn run_layer_by_layer_graph(
    graph: &GraphNetwork,
    input_bounded: &BoundedTensor,
    epsilon: f32,
    progress: bool,
    progress_json: bool,
    json: bool,
) -> Result<()> {
    let start = Instant::now();
    let show_progress = progress || progress_json;
    let result = if show_progress {
        let progress_callback = |p: LayerProgress| {
            let pct = ((p.node_index + 1) as f32 / p.total_nodes as f32 * 100.0) as u32;
            let eta = p.estimated_remaining();
            if progress_json {
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "type": "layer_progress",
                        "node_index": p.node_index,
                        "total_nodes": p.total_nodes,
                        "node_name": p.node_name,
                        "layer_type": p.layer_type,
                        "percent": pct,
                        "current_max_sensitivity": p.current_max_sensitivity,
                        "degraded_so_far": p.degraded_so_far,
                        "elapsed_ms": p.elapsed.as_millis(),
                        "eta_ms": eta.as_millis()
                    })
                );
            } else {
                eprint!(
                    "\r[{:3}%] Node {}/{}: {} ({}) | max_sens: {:.2e} | elapsed: {:.1?} | ETA: {:.1?}  ",
                    pct,
                    p.node_index + 1,
                    p.total_nodes,
                    p.node_name,
                    p.layer_type,
                    p.current_max_sensitivity,
                    p.elapsed,
                    eta
                );
            }
        };
        let result = graph.propagate_ibp_detailed_with_progress(
            input_bounded,
            epsilon,
            Some(progress_callback),
        )?;
        if !progress_json {
            eprintln!();
        }
        result
    } else {
        graph.propagate_ibp_detailed(input_bounded, epsilon)?
    };
    let elapsed = start.elapsed();

    if json {
        let nodes_json: Vec<_> = result
            .nodes
            .iter()
            .map(|n| {
                serde_json::json!({
                    "name": n.name,
                    "layer_type": n.layer_type,
                    "input_width": n.input_width,
                    "output_width": n.output_width,
                    "sensitivity": n.sensitivity,
                    "output_shape": n.output_shape,
                    "min_bound": n.min_bound,
                    "max_bound": n.max_bound,
                    "saturated": n.saturated,
                    "has_nan": n.has_nan,
                    "has_infinite": n.has_infinite,
                    "status": n.status()
                })
            })
            .collect();

        println!(
            "{}",
            serde_json::json!({
                "mode": "layer_by_layer",
                "method": "ibp",
                "input_epsilon": result.input_epsilon,
                "final_width": result.final_width,
                "total_nodes": result.total_nodes,
                "degraded_at_node": result.degraded_at_node,
                "elapsed_ms": elapsed.as_millis(),
                "nodes": nodes_json
            })
        );
    } else {
        println!("{}", result.summary());
        println!("\nElapsed: {:.2?}", elapsed);
    }
    Ok(())
}

/// Run block-wise mode with optional checkpoint support.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_block_wise(
    network: &VerifiableNetwork,
    spec_input_shape: Option<&[usize]>,
    epsilon: f32,
    effective_method: ny_propagate::PropagationMethod,
    effective_backend: BackendArg,
    model: &Path,
    progress: bool,
    progress_json: bool,
    max_blocks: usize,
    checkpoint: Option<&Path>,
    json: bool,
) -> Result<()> {
    let effective_shape = spec_input_shape.ok_or_else(|| {
        JsonCliError::new(
            "missing_input_shape",
            "Block-wise mode requires input_shape in verification spec.",
        )
    })?;
    let input_data = ArrayD::zeros(IxDyn(effective_shape));
    let input_bounded = BoundedTensor::from_epsilon(input_data, epsilon)?;

    match network {
        VerifiableNetwork::Sequential(_net) => {
            let message = "Block-wise mode is only supported for native models (GraphNetwork). \
                Hint: use --native flag with PyTorch/SafeTensors/GGUF models.";
            Err(JsonCliError::new("unsupported_model_format", message).into())
        }
        VerifiableNetwork::Graph(graph) => run_block_wise_graph(
            graph,
            &input_bounded,
            epsilon,
            effective_method,
            effective_backend,
            model,
            progress,
            progress_json,
            max_blocks,
            checkpoint,
            json,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_block_wise_graph(
    graph: &GraphNetwork,
    input_bounded: &BoundedTensor,
    epsilon: f32,
    effective_method: ny_propagate::PropagationMethod,
    effective_backend: BackendArg,
    model: &Path,
    progress: bool,
    progress_json: bool,
    max_blocks: usize,
    checkpoint: Option<&Path>,
    json: bool,
) -> Result<()> {
    let start = Instant::now();
    let show_progress = progress || progress_json;

    let method_str = format!("{:?}", effective_method).to_lowercase();
    let backend_str = format!("{}", effective_backend);
    let model_hash = compute_model_hash(model)?;

    let (existing_checkpoint, checkpoint_path) = if let Some(ckpt_path) = checkpoint {
        if ckpt_path.exists() {
            let ckpt = VerificationCheckpoint::load(ckpt_path)?;
            ckpt.validate(model, &model_hash, epsilon, &method_str, &backend_str)?;

            if ckpt.is_complete() {
                if !json {
                    println!("Checkpoint is already complete. Showing previous results:");
                }
                let result = ckpt.into_result()?;
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "mode": "block_wise",
                            "method": "ibp+zonotope",
                            "resumed_from_checkpoint": true,
                            "total_blocks": result.total_blocks,
                            "max_sensitivity": result.max_sensitivity,
                            "degraded_blocks": result.degraded_blocks
                        })
                    );
                } else {
                    println!("{}", result.summary());
                }
                return Ok(());
            }

            if !json {
                eprintln!(
                    "Resuming from checkpoint: {}/{} blocks complete",
                    ckpt.next_block_index, ckpt.total_blocks
                );
            }
            (Some(ckpt), Some(ckpt_path.to_path_buf()))
        } else {
            (None, Some(ckpt_path.to_path_buf()))
        }
    } else {
        (None, None)
    };

    use std::sync::Mutex;

    let checkpoint_state = Mutex::new(existing_checkpoint.clone());

    let result = if show_progress || max_blocks > 0 || checkpoint.is_some() {
        let progress_callback = if show_progress {
            Some(|p: BlockProgress| {
                let pct = ((p.block_index + 1) as f32 / p.total_blocks as f32 * 100.0) as u32;
                let eta = p.estimated_remaining();
                if progress_json {
                    eprintln!(
                        "{}",
                        serde_json::json!({
                            "type": "block_progress",
                            "block_index": p.block_index,
                            "total_blocks": p.total_blocks,
                            "block_name": p.block_name,
                            "percent": pct,
                            "current_max_sensitivity": p.current_max_sensitivity,
                            "degraded_so_far": p.degraded_so_far,
                            "elapsed_ms": p.elapsed.as_millis(),
                            "eta_ms": eta.as_millis()
                        })
                    );
                } else {
                    eprint!(
                        "\r[{:3}%] Block {}/{}: {} | max_sens: {:.2e} | elapsed: {:.1?} | ETA: {:.1?}  ",
                        pct,
                        p.block_index + 1,
                        p.total_blocks,
                        p.block_name,
                        p.current_max_sensitivity,
                        p.elapsed,
                        eta
                    );
                }
            })
        } else {
            None
        };

        let model_path = model.to_path_buf();
        let model_hash_clone = model_hash;
        let method_str_clone = method_str;
        let backend_str_clone = backend_str;

        let checkpoint_callback = if checkpoint.is_some() {
            Some(
                move |block: &ny_propagate::BlockBoundsInfo,
                      elapsed_ms: u64,
                      total_blocks: usize| {
                    let mut state = checkpoint_state
                        .lock()
                        .expect("invariant: checkpoint state mutex not poisoned");

                    if state.is_none() {
                        *state = Some(VerificationCheckpoint::new(
                            model_path.clone(),
                            model_hash_clone.clone(),
                            epsilon,
                            &method_str_clone,
                            &backend_str_clone,
                            total_blocks,
                        ));
                    }

                    if let Some(ref mut ckpt) = *state {
                        ckpt.total_blocks = total_blocks;
                        ckpt.update(block.clone(), elapsed_ms);

                        if let Some(ref path) = checkpoint_path {
                            if let Err(e) = ckpt.save(path) {
                                if !json {
                                    eprintln!("\nWarning: Failed to save checkpoint: {}", e);
                                }
                            }
                        }
                    }
                },
            )
        } else {
            None
        };

        let result = graph.propagate_ibp_block_wise_with_checkpoint(
            input_bounded,
            epsilon,
            progress_callback,
            checkpoint_callback,
            max_blocks,
            existing_checkpoint.as_ref(),
        )?;
        if show_progress && !progress_json {
            eprintln!();
        }
        result
    } else {
        graph.propagate_ibp_block_wise(input_bounded, epsilon)?
    };
    let elapsed = start.elapsed();

    if json {
        print_block_wise_json(&result, elapsed);
    } else {
        println!("{}", result.summary());
        println!("\nElapsed: {:.2?}", elapsed);
    }
    Ok(())
}

fn print_block_wise_json(result: &ny_propagate::BlockWiseResult, elapsed: std::time::Duration) {
    let blocks_json: Vec<_> = result
        .blocks
        .iter()
        .map(|b| {
            let nodes_json: Vec<_> = b
                .nodes
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "name": n.name,
                        "layer_type": n.layer_type,
                        "input_width": n.input_width,
                        "output_width": n.output_width,
                        "sensitivity": n.sensitivity,
                        "saturated": n.saturated,
                        "has_nan": n.has_nan,
                        "has_infinite": n.has_infinite
                    })
                })
                .collect();
            serde_json::json!({
                "block_index": b.block_index,
                "block_name": b.block_name,
                "input_width": b.input_width,
                "output_width": b.output_width,
                "sensitivity": b.sensitivity,
                "qk_matmul_width": b.qk_matmul_width,
                "swiglu_width": b.swiglu_width,
                "degraded": b.degraded,
                "status": b.status(),
                "num_nodes": b.nodes.len(),
                "nodes": nodes_json
            })
        })
        .collect();

    let worst_5: Vec<_> = result
        .worst_k_blocks(5)
        .iter()
        .map(|(idx, name, sens, out_width)| {
            serde_json::json!({
                "block_index": idx,
                "block_name": name,
                "sensitivity": sens,
                "output_width": out_width
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::json!({
            "mode": "block_wise",
            "method": "ibp+zonotope",
            "block_epsilon": result.block_epsilon,
            "total_blocks": result.total_blocks,
            "summary": {
                "max_sensitivity": result.max_sensitivity,
                "min_sensitivity": result.min_sensitivity(),
                "median_sensitivity": result.median_sensitivity(),
                "sensitivity_range": result.sensitivity_range(),
                "degraded_blocks": result.degraded_blocks,
                "worst_5_blocks": worst_5
            },
            "elapsed_ms": elapsed.as_millis(),
            "blocks": blocks_json
        })
    );
}

#[cfg(test)]
#[path = "modes_tests.rs"]
mod tests;
