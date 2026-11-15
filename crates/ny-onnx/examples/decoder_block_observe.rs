// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Export all original stages of one decoder block in one graph traversal.
//!
//! `targo --unverified run -p ny-onnx --example decoder_block_observe -- request.json`
//! Request fields: model_directory, output_directory, block_index, fixtures.
//! Each fixture has name, positions, shape, input_f32 (little-endian row-major).
//! The input is an archived RAW previous-block output, not final normalized
//! hidden states. Original shapes and actual graph shapes are both recorded.
//! Diagnostic observations do not carry a numerical certificate.

use ndarray::{ArrayD, IxDyn};
use ny_onnx::native::decoder_contract::{
    load_decoder_safetensors, lower_decoder, DecoderContract, DecoderOptions, LoweringManifest,
    ObservationBoundary,
};
use ny_propagate::certify::work::WorkReceipt;
use ny_propagate::network::GraphPointTrace;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model_directory: PathBuf,
    output_directory: PathBuf,
    block_index: usize,
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    positions: Vec<u64>,
    shape: Vec<usize>,
    input_f32: PathBuf,
}
#[derive(Serialize)]
struct ExportedTensor {
    node: String,
    tensor: String,
    original_shape: Vec<usize>,
    shape: Vec<usize>,
    file: String,
    value_digest: [u8; 32],
    file_sha256: [u8; 32],
}
#[derive(Serialize)]
struct FixtureExport {
    fixture: Fixture,
    metadata_file: String,
    input_file_sha256: [u8; 32],
    input_digest: [u8; 32],
    outputs_digest: [u8; 32],
    execution_digest: [u8; 32],
    tensors: Vec<ExportedTensor>,
    constants: Vec<ExportedConstant>,
}
#[derive(Serialize)]
struct ExportedConstant {
    name: &'static str,
    shape: Vec<usize>,
    file: String,
    file_sha256: [u8; 32],
}
#[derive(Serialize)]
struct ReceiptExport {
    attempted: u64,
    admitted: u64,
    refused: u64,
    completed: u64,
    work_reserved: u64,
    proposals_reserved: u64,
    live_scratch_bytes: usize,
    live_persistent_bytes: usize,
    peak_leased_bytes: usize,
}
impl From<WorkReceipt> for ReceiptExport {
    fn from(r: WorkReceipt) -> Self {
        Self {
            attempted: r.attempted,
            admitted: r.admitted,
            refused: r.refused,
            completed: r.completed,
            work_reserved: r.work_reserved,
            proposals_reserved: r.proposals_reserved,
            live_scratch_bytes: r.live_scratch_bytes,
            live_persistent_bytes: r.live_persistent_bytes,
            peak_leased_bytes: r.peak_leased_bytes,
        }
    }
}
#[derive(Serialize)]
struct Metadata<'a> {
    schema_version: u32,
    certificate_authority: bool,
    dtype: &'static str,
    byte_order: &'static str,
    layout: &'static str,
    boundary: &'a ObservationBoundary,
    trace: &'a GraphPointTrace,
    input_digest: &'a [u8; 32],
    outputs_digest: &'a [u8; 32],
    execution_digest: &'a [u8; 32],
    original_lowering_manifest: &'a LoweringManifest,
    work_receipt: ReceiptExport,
    receipt_semantics: &'static str,
}

fn bounded_read(path: &Path, cap: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() > cap as u64 {
        return Err("file exceeds byte cap".into());
    }
    let mut bytes = Vec::new();
    file.take(cap as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > cap {
        return Err("file grew past byte cap".into());
    }
    Ok(bytes)
}
fn json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut out = BufWriter::new(std::fs::File::create(path)?);
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 1 {
        return Err("usage: decoder_block_observe request.json".into());
    }
    let request: Request =
        serde_json::from_slice(&bounded_read(Path::new(&arguments[0]), 1 << 20)?)?;
    if request.fixtures.is_empty() || request.fixtures.len() > 16 {
        return Err("requires1..=16 fixtures".into());
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
    let mut exports = Vec::new();
    for (fixture_index, fixture) in request.fixtures.into_iter().enumerate() {
        let expected = [
            1,
            fixture.positions.len(),
            contract.parameters().hidden_size,
        ];
        if fixture.shape != expected && fixture.shape != expected[1..] {
            return Err("archived block input shape mismatch".into());
        }
        let count = fixture
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or("input element count overflow")?;
        let size = count.checked_mul(4).ok_or("input byte count overflow")?;
        let input_bytes = bounded_read(&fixture.input_f32, size)?;
        if input_bytes.len() != size {
            return Err("wrong archived F32 input length".into());
        }
        let input_file_sha256: [u8; 32] = Sha256::digest(&input_bytes).into();
        let values: Vec<_> = input_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from_le_bytes(*b)))
            .collect();
        let input = ArrayD::from_shape_vec(IxDyn(&fixture.shape), values)?;
        let weights =
            load_decoder_safetensors(request.model_directory.join("model.safetensors"), &options)?;
        let model = lower_decoder(weights, &contract, &fixture.positions, &options)?;
        let observer = model.observe_block(request.block_index, &options)?;
        let observation = observer.point(&input, &options)?;
        let mut constants = Vec::new();
        // Export the already resident, immutable original pinned tables by
        // logical iteration; no regeneration and no tensor-sized clone.
        for name in ["_ny_decoder.cos", "_ny_decoder.sin"] {
            let values = model
                .graph_model()
                .weights
                .get(name)
                .ok_or("missing original table")?;
            let file = format!(
                "fixture-{fixture_index:02}.{}.f32",
                name.rsplit('.').next().unwrap()
            );
            let mut output =
                BufWriter::new(std::fs::File::create(request.output_directory.join(&file))?);
            let mut hash = Sha256::new();
            for &value in values {
                let bytes = value.to_bits().to_le_bytes();
                output.write_all(&bytes)?;
                hash.update(bytes);
            }
            output.flush()?;
            constants.push(ExportedConstant {
                name,
                shape: values.shape().to_vec(),
                file,
                file_sha256: hash.finalize().into(),
            });
        }
        let metadata_file = format!("fixture-{fixture_index:02}.observation.json");
        json(
            &request.output_directory.join(&metadata_file),
            &Metadata {
                schema_version: 1,
                certificate_authority: false,
                dtype: "F32",
                byte_order: "little",
                layout: "logical C row-major",
                boundary: observation.boundary(),
                trace: observation.execution_trace(),
                input_digest: observation.input_digest(),
                outputs_digest: observation.outputs_digest(),
                execution_digest: observation.execution_digest(),
                original_lowering_manifest: model.manifest(),
                work_receipt: model.budget().receipt()?.into(),
                receipt_semantics: "reserved work and leased capacities; not runtime/RSS",
            },
        )?;
        let mut tensors = Vec::new();
        for (stage_index, tensor) in observation.tensors().iter().enumerate() {
            let file = format!("fixture-{fixture_index:02}.stage-{stage_index:03}.f32");
            let mut output =
                BufWriter::new(std::fs::File::create(request.output_directory.join(&file))?);
            let mut hash = Sha256::new();
            for &value in tensor.values() {
                let bytes = value.to_bits().to_le_bytes();
                output.write_all(&bytes)?;
                hash.update(bytes);
            }
            output.flush()?;
            tensors.push(ExportedTensor {
                node: tensor.identity().node.clone(),
                tensor: tensor.identity().tensor.clone(),
                original_shape: tensor.identity().original_shape.clone(),
                shape: tensor.values().shape().to_vec(),
                file,
                value_digest: *tensor.value_digest(),
                file_sha256: hash.finalize().into(),
            });
        }
        eprintln!(
            "[decoder-block-observe] fixture={} block={} stages={} one_traversal=true",
            fixture.name,
            request.block_index,
            tensors.len()
        );
        exports.push(FixtureExport {
            fixture,
            metadata_file,
            input_file_sha256,
            input_digest: *observation.input_digest(),
            outputs_digest: *observation.outputs_digest(),
            execution_digest: *observation.execution_digest(),
            tensors,
            constants,
        });
    }
    json(
        &request.output_directory.join("run.json"),
        &serde_json::json!({
            "schema_version":1, "certificate_authority":false,
            "claim":"diagnostic_original_lowered_block_observations",
            "model_directory":request.model_directory, "block_index":request.block_index,
            "fixtures":exports,
        }),
    )?;
    Ok(())
}
