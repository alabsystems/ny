// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Export actual lowered graph observations for an independent pinned runtime.
//!
//! Run with `targo --unverified run -p ny-onnx --example decoder_parity -- request.json`.
//! Request:
//! ```json
//! {"model_directory":"/absolute/pinned/model","output_directory":"/absolute/archive/ny",
//!  "prompts":[{"name":"explicit-ids","token_ids":[1,2,3],"positions":[0,1,2]}]}
//! ```
//! This performs checked original token lookup then evaluates the actual
//! ny-build GraphNetwork through its concrete point route. It is a parity
//! observation, never an execution certificate. No alternate decoder evaluator
//! or top-K vocabulary path is used.
//!
//! Every binary tensor stores little-endian F32 values in C row-major order:
//! `[sequence_position, hidden_coordinate]` for every `block.N` and `hidden`,
//! `[sequence_position, original_vocabulary_id]` for `logits`. The singleton
//! batch axis is removed only for the export; `graph_shape` records the actual
//! graph result shape. `hidden` is post-final-normalization. `block.N` is the
//! raw decoder block output before final normalization, including the final
//! block; HF's last `hidden_states` entry alone is not that final block output.

use ny_onnx::native::decoder_contract::{
    load_decoder_safetensors, lower_decoder, DecoderContract, DecoderOptions,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model_directory: PathBuf,
    output_directory: PathBuf,
    prompts: Vec<Prompt>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Prompt {
    name: String,
    token_ids: Vec<usize>,
    positions: Vec<u64>,
}
#[derive(Debug, Serialize)]
struct TensorExport {
    output_alias: String,
    file: String,
    dtype: &'static str,
    byte_order: &'static str,
    shape: Vec<usize>,
    graph_shape: Vec<usize>,
    axis_names: [&'static str; 2],
    elements: usize,
    bytes: usize,
}
#[derive(Debug, Serialize)]
struct PromptExport {
    name: String,
    token_ids: Vec<usize>,
    positions: Vec<u64>,
    manifest_file: String,
    lowering_digest: [u8; 32],
    tensors: Vec<TensorExport>,
}
#[derive(Debug, Serialize)]
struct RunExport {
    schema_version: u32,
    observation_contract: &'static str,
    comparison_frontier: &'static str,
    model_directory: PathBuf,
    prompts: Vec<PromptExport>,
}

fn bounded_read(path: &Path, max: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() > max as u64 {
        return Err(format!("{} exceeds {max} bytes", path.display()).into());
    }
    let mut bytes = Vec::new();
    file.take(max as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max {
        return Err("file grew past admitted byte cap".into());
    }
    Ok(bytes)
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(std::fs::File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: decoder_parity request.json".into());
    }
    let request: Request = serde_json::from_slice(&bounded_read(Path::new(&args[0]), 1 << 20)?)?;
    if request.prompts.is_empty() || request.prompts.len() > 16 {
        return Err("request requires 1..=16 explicit prompts".into());
    }
    let options = DecoderOptions::default();
    let contract = DecoderContract::from_json(
        &bounded_read(
            &request.model_directory.join("config.json"),
            options.max_config_bytes,
        )?,
        &options,
    )?;
    std::fs::create_dir_all(&request.output_directory)?;
    let mut records = Vec::with_capacity(request.prompts.len());
    for (prompt_index, prompt) in request.prompts.iter().enumerate() {
        if prompt.token_ids.len() != prompt.positions.len() {
            return Err(format!("{}: explicit token/position lengths differ", prompt.name).into());
        }
        let weights =
            load_decoder_safetensors(request.model_directory.join("model.safetensors"), &options)?;
        let model = lower_decoder(weights, &contract, &prompt.positions, &options)?;
        let input = model.embedded_tokens(&prompt.token_ids, &options)?;
        let manifest_file = format!("prompt-{prompt_index:02}.lowering.json");
        write_json(
            &request.output_directory.join(&manifest_file),
            model.manifest(),
        )?;
        let mut tensors = Vec::with_capacity(model.block_outputs().len() + 2);
        let aliases: Vec<_> = (0..model.block_outputs().len())
            .map(|index| format!("block.{index}"))
            .chain(["hidden".to_string(), "logits".to_string()])
            .collect();
        for alias in aliases {
            let graph = model.graph_for(&alias, &options)?;
            let observation = graph.point(input.values(), &options)?;
            let values = observation.values();
            let width = if alias == "logits" {
                contract.parameters().vocab_size
            } else {
                contract.parameters().hidden_size
            };
            let shape = [prompt.token_ids.len(), width];
            if values.shape() != shape && values.shape() != [1, shape[0], shape[1]] {
                return Err(format!(
                    "{alias}: unexpected actual graph shape {:?}",
                    values.shape()
                )
                .into());
            }
            let file_name = format!("prompt-{prompt_index:02}.{}.f32", alias.replace('.', "-"));
            let mut file = BufWriter::new(std::fs::File::create(
                request.output_directory.join(&file_name),
            )?);
            // ndarray iteration is logical row-major even if a producer returns
            // a nonstandard memory layout; no full-tensor export clone.
            for &value in values {
                if !value.is_finite() {
                    return Err("nonfinite parity observation".into());
                }
                file.write_all(&value.to_bits().to_le_bytes())?;
            }
            file.flush()?;
            tensors.push(TensorExport {
                output_alias: alias.clone(),
                file: file_name,
                dtype: "F32",
                byte_order: "little",
                shape: shape.to_vec(),
                graph_shape: values.shape().to_vec(),
                axis_names: [
                    "sequence_position",
                    if alias == "logits" {
                        "original_vocabulary_id"
                    } else {
                        "hidden_coordinate"
                    },
                ],
                elements: values.len(),
                bytes: values
                    .len()
                    .checked_mul(4)
                    .ok_or("tensor byte count overflow")?,
            });
        }
        eprintln!(
            "[decoder-parity] prompt={} tokens={} blocks={} complete_vocab={} digest={:02x?}",
            prompt.name,
            prompt.token_ids.len(),
            model.block_outputs().len(),
            contract.parameters().vocab_size,
            model.manifest().lowering_digest
        );
        records.push(PromptExport {
            name: prompt.name.clone(),
            token_ids: prompt.token_ids.clone(),
            positions: prompt.positions.clone(),
            manifest_file,
            lowering_digest: model.manifest().lowering_digest,
            tensors,
        });
    }
    write_json(&request.output_directory.join("run.json"), &RunExport {
        schema_version: 1,
        observation_contract: "actual ny-build GraphNetwork concrete-point observation, decoded original coefficients, hashed F32 positional tables",
        comparison_frontier: "compare each raw block plus post-final-norm hidden and complete logits with pinned original CPU-eager runtime; point parity is not a numerical proof",
        model_directory: request.model_directory, prompts: records,
    })?;
    Ok(())
}
