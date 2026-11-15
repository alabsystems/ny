// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One managed full decoder forward with the unchanged default resource caps.
//!
//! `targo --unverified run -p ny-onnx --example decoder_compact_probe -- request.json`
//! Closed request: model_directory, output_directory, token_ids, positions.
//! This diagnostic exports every position and original vocabulary coordinate.
//! It is neither a numerical certificate nor a runtime or process-RSS claim.

use ny_onnx::native::decoder_contract::{
    load_decoder_safetensors, lower_decoder, DecoderContract, DecoderOptions,
};
use ny_propagate::certify::work::{WorkLease, WorkReceipt, WorkRequest};
use ny_propagate::Layer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const REQUEST_CAP: usize = 1 << 20;
const JSON_CAP: usize = 32 << 20;
const LOGITS_CAP: usize = 64 << 20;
const BUFFER_BYTES: usize = 16 << 10;
const POLL_BYTES: usize = 4096;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model_directory: PathBuf,
    output_directory: PathBuf,
    token_ids: Vec<usize>,
    positions: Vec<u64>,
}

#[derive(Serialize)]
struct Receipt {
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
impl From<WorkReceipt> for Receipt {
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
struct Capacity {
    scratch_bytes: usize,
    persistent_bytes: usize,
    work: u64,
    proposals: u64,
}
impl From<WorkRequest> for Capacity {
    fn from(r: WorkRequest) -> Self {
        Self {
            scratch_bytes: r.scratch_bytes,
            persistent_bytes: r.persistent_bytes,
            work: r.work,
            proposals: r.proposals,
        }
    }
}

#[derive(Serialize)]
struct DefaultCaps {
    deadline: Option<&'static str>,
    max_file_bytes: usize,
    max_resident_bytes: usize,
    max_scratch_bytes: usize,
    max_terms: u64,
    max_proposals: u64,
}

#[derive(Serialize)]
struct ArchiveFile {
    file: &'static str,
    bytes: usize,
    sha256: [u8; 32],
}

#[derive(Serialize)]
struct Report<'a> {
    schema_version: u32,
    certificate_authority: bool,
    observation_contract: &'static str,
    receipt_semantics: &'static str,
    receipt_snapshot: &'static str,
    default_caps: DefaultCaps,
    request: &'a Request,
    reference_profile: &'a str,
    driver_source_sha256: [u8; 32],
    lowering_digest: &'a [u8; 32],
    input_digest: &'a [u8; 32],
    value_digest: &'a [u8; 32],
    execution_digest: &'a [u8; 32],
    selected_output: &'a str,
    graph_shape: &'a [usize],
    export_shape: [usize; 2],
    axis_names: [&'static str; 2],
    dtype: &'static str,
    byte_order: &'static str,
    layout: &'static str,
    expected_blocks: usize,
    actual_completed_blocks: usize,
    original_lowered_nodes: usize,
    constructed_graph_nodes: usize,
    actual_completed_nodes: usize,
    point_linear_nodes: usize,
    proof_eligibility_refusal: &'a str,
    source_owned_requested_bytes: usize,
    embedded_input_requested_bytes: usize,
    graph_admission: Capacity,
    graph_owned_requested_bytes: usize,
    original_linear_retained_bytes: usize,
    point_scratch_requested_bytes: usize,
    point_work_reserved_bound: u64,
    point_observation_requested_bytes: usize,
    export_admission: Capacity,
    original_budget_receipt: Receipt,
    files: [ArchiveFile; 3],
}

fn bounded_read(path: &Path, cap: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let file = File::open(path)?;
    if file.metadata()?.len() > cap as u64 {
        return Err("input file exceeds byte cap".into());
    }
    let mut bytes = Vec::new();
    file.take(cap as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > cap {
        return Err("input file grew beyond byte cap".into());
    }
    Ok(bytes)
}

/// Stream writes, hashing and serialization share the original model lease.
/// The hard cap is checked before a write; every4KiB chunk polls that lease.
struct PolledWriter<'a, W> {
    inner: W,
    lease: &'a WorkLease,
    bytes: usize,
    cap: usize,
    hash: Option<Sha256>,
}
impl<'a, W: Write> PolledWriter<'a, W> {
    fn new(inner: W, lease: &'a WorkLease, cap: usize, hash: bool) -> Self {
        Self {
            inner,
            lease,
            bytes: 0,
            cap,
            hash: hash.then(Sha256::new),
        }
    }
    fn finish(mut self, file: &'static str) -> Result<ArchiveFile, Box<dyn Error>> {
        self.flush()?;
        let hash = self.hash.take().ok_or("missing output hash")?;
        Ok(ArchiveFile {
            file,
            bytes: self.bytes,
            sha256: hash.finalize().into(),
        })
    }
}
impl<W: Write> Write for PolledWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let total = self
            .bytes
            .checked_add(bytes.len())
            .filter(|&n| n <= self.cap)
            .ok_or_else(|| io::Error::other("output byte cap"))?;
        for part in bytes.chunks(POLL_BYTES) {
            poll(self.lease)?;
            self.inner.write_all(part)?;
            if let Some(hash) = &mut self.hash {
                hash.update(part);
            }
        }
        self.bytes = total;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        poll(self.lease)?;
        self.inner.flush()
    }
}

fn poll(lease: &WorkLease) -> io::Result<()> {
    lease.checkpoint().map_err(io::Error::other)?;
    if ny_core::instance_budget::deadline().is_some_and(|limit| std::time::Instant::now() >= limit)
    {
        return Err(io::Error::other("original instance deadline"));
    }
    Ok(())
}

fn json_size(value: &impl Serialize, lease: &WorkLease) -> Result<usize, Box<dyn Error>> {
    let mut count = PolledWriter::new(io::sink(), lease, JSON_CAP, false);
    serde_json::to_writer(&mut count, value)?;
    count.write_all(b"\n")?;
    count.flush()?;
    Ok(count.bytes)
}

fn output_writer<'a>(
    directory: &Path,
    file: &'static str,
    lease: &'a WorkLease,
    cap: usize,
) -> Result<PolledWriter<'a, BufWriter<File>>, Box<dyn Error>> {
    poll(lease)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(file))?;
    Ok(PolledWriter::new(
        BufWriter::with_capacity(BUFFER_BYTES, file),
        lease,
        cap,
        true,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: decoder_compact_probe request.json".into());
    }
    let request: Request =
        serde_json::from_slice(&bounded_read(Path::new(&args[0]), REQUEST_CAP)?)?;
    if !(1..=4).contains(&request.token_ids.len())
        || request.token_ids.len() != request.positions.len()
    {
        return Err("requires1..=4 explicit token IDs and matching positions".into());
    }
    // No deadline or resource override: the ordinary managed defaults qualify.
    let options = DecoderOptions::default();
    let contract = DecoderContract::from_json(
        &bounded_read(
            &request.model_directory.join("config.json"),
            options.max_config_bytes,
        )?,
        &options,
    )?;
    let weights =
        load_decoder_safetensors(request.model_directory.join("model.safetensors"), &options)?;
    let model = lower_decoder(weights, &contract, &request.positions, &options)?;
    let embedded = model.embedded_tokens(&request.token_ids, &options)?;
    let admission = model.graph_admission(&options)?;
    let graph = model.graph_for("logits", &options)?;
    let observation = graph.point(embedded.values(), &options)?;
    let values = observation.values();
    let shape = [request.token_ids.len(), contract.parameters().vocab_size];
    if values.shape() != shape && values.shape() != [1, shape[0], shape[1]] {
        return Err("full logits graph shape mismatch".into());
    }
    let logits_bytes = values
        .len()
        .checked_mul(4)
        .filter(|&n| n <= LOGITS_CAP)
        .ok_or("full logits output byte cap")?;
    let proof_refusal = graph
        .graph()
        .ensure_proof_eligible()
        .err()
        .ok_or("compact diagnostic graph unexpectedly proof eligible")?
        .to_string();
    if !proof_refusal.contains("PointLinear") {
        return Err("unexpected proof eligibility refusal".into());
    }
    let budget = model.budget();
    // The paid inspection/counting pass precedes its metadata allocations.
    // Both serializations are capped in aggregate, without a JSON tree clone.
    let constructed_nodes = graph.graph().node_names().len();
    let inspection_scratch = constructed_nodes
        .checked_mul(128)
        .and_then(|n| n.checked_add(64 << 10))
        .ok_or("inspection scratch overflow")?;
    let mut inspect = budget.lease(WorkRequest {
        scratch_bytes: inspection_scratch,
        persistent_bytes: 0,
        work: (JSON_CAP as u64) * 4,
        proposals: 0,
    })?;
    let trace = observation.execution_trace();
    let mut completed = BTreeSet::new();
    for step in trace.steps() {
        poll(&inspect)?;
        if !completed.insert(step.node.as_str()) {
            return Err("a graph node executed more than once".into());
        }
    }
    if completed.len() != constructed_nodes {
        return Err("full graph did not execute every constructed node".into());
    }
    let mut point_linear_nodes = 0usize;
    for name in graph.graph().node_names() {
        poll(&inspect)?;
        if !completed.contains(name.as_str()) {
            return Err("constructed graph node omitted from trace".into());
        }
        let node = graph.graph().node(name).ok_or("graph node disappeared")?;
        point_linear_nodes += usize::from(matches!(node.layer(), Layer::PointLinear(_)));
    }
    let mut completed_blocks = 0usize;
    for tensor in model.block_outputs() {
        poll(&inspect)?;
        let producer = model
            .manifest()
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find(|op| op.output == *tensor)
            .ok_or("original block output producer missing")?;
        if !completed.contains(producer.name.as_str()) {
            return Err("original block output was not executed".into());
        }
        completed_blocks += 1;
    }
    if completed_blocks != contract.parameters().num_layers || point_linear_nodes == 0 {
        return Err("complete original compact topology missing".into());
    }
    let manifest_bytes = json_size(model.manifest(), &inspect)?;
    let trace_bytes = json_size(trace, &inspect)?;
    if manifest_bytes
        .checked_add(trace_bytes)
        .ok_or("JSON byte overflow")?
        > JSON_CAP
    {
        return Err("aggregate JSON byte cap".into());
    }
    drop(completed);
    inspect.complete()?;
    drop(inspect);
    let export_bytes = logits_bytes
        .checked_add(manifest_bytes)
        .and_then(|n| n.checked_add(trace_bytes))
        .and_then(|n| n.checked_add(64 << 10))
        .ok_or("export byte overflow")?;
    let export_request = WorkRequest {
        scratch_bytes: 2 * BUFFER_BYTES,
        persistent_bytes: 0,
        work: (export_bytes as u64)
            .checked_mul(16)
            .ok_or("export work overflow")?,
        proposals: 0,
    };
    let mut export = budget.lease(export_request)?;
    // A newly created directory and create_new files prevent archive overwrite.
    std::fs::create_dir(&request.output_directory)?;
    let mut logits = output_writer(
        &request.output_directory,
        "logits.f32",
        &export,
        logits_bytes,
    )?;
    for &value in values {
        if !value.is_finite() {
            return Err("nonfinite point observation".into());
        }
        logits.write_all(&value.to_bits().to_le_bytes())?;
    }
    let logits_file = logits.finish("logits.f32")?;
    let mut manifest = output_writer(
        &request.output_directory,
        "lowering.json",
        &export,
        manifest_bytes,
    )?;
    serde_json::to_writer(&mut manifest, model.manifest())?;
    manifest.write_all(b"\n")?;
    let manifest_file = manifest.finish("lowering.json")?;
    let mut trace_file = output_writer(
        &request.output_directory,
        "trace.json",
        &export,
        trace_bytes,
    )?;
    serde_json::to_writer(&mut trace_file, trace)?;
    trace_file.write_all(b"\n")?;
    let trace_file = trace_file.finish("trace.json")?;
    let report = Report {
        schema_version: 1,
        certificate_authority: false,
        observation_contract: "one original model load/lowering; one compact logits graph build; one managed full concrete forward; mixed diagnostic point arithmetic and hashed pinned positional tables",
        receipt_semantics: "work/proposals are reserved allowances; byte counters are requested leased capacities, not process RSS; point parity is not a numerical proof or runtime claim",
        receipt_snapshot: "original shared ledger after logits, manifest and trace writes, before the final report write; final export completion is printed to the run log",
        default_caps: DefaultCaps {
            deadline: None,
            max_file_bytes: options.max_file_bytes,
            max_resident_bytes: options.max_resident_bytes,
            max_scratch_bytes: options.max_scratch_bytes,
            max_terms: options.max_terms,
            max_proposals: options.max_proposals,
        },
        request: &request,
        reference_profile: &model.manifest().parameters.reference_profile,
        driver_source_sha256: Sha256::digest(include_bytes!("decoder_compact_probe.rs")).into(),
        lowering_digest: observation.lowering_digest(),
        input_digest: observation.input_digest(),
        value_digest: observation.value_digest(),
        execution_digest: observation.execution_digest(),
        selected_output: observation.selected_output(),
        graph_shape: values.shape(),
        export_shape: shape,
        axis_names: ["sequence_position", "original_vocabulary_id"],
        dtype: "F32",
        byte_order: "little",
        layout: "logical C row-major; optional singleton leading batch is recorded in graph_shape",
        expected_blocks: contract.parameters().num_layers,
        actual_completed_blocks: completed_blocks,
        original_lowered_nodes: model.graph_model().network.layers.len(),
        constructed_graph_nodes: constructed_nodes,
        actual_completed_nodes: trace.steps().len(),
        point_linear_nodes,
        proof_eligibility_refusal: &proof_refusal,
        source_owned_requested_bytes: model.reserved_bytes(),
        embedded_input_requested_bytes: embedded.reserved_bytes(),
        graph_admission: admission.into(),
        graph_owned_requested_bytes: graph.reserved_bytes(),
        original_linear_retained_bytes: graph.linear_retained_bytes()?,
        point_scratch_requested_bytes: graph.point_scratch_bytes(),
        point_work_reserved_bound: graph.point_work_bound(),
        point_observation_requested_bytes: observation.reserved_bytes(),
        export_admission: export_request.into(),
        original_budget_receipt: budget.receipt()?.into(),
        files: [logits_file, manifest_file, trace_file],
    };
    let mut report_file = output_writer(&request.output_directory, "run.json", &export, 64 << 10)?;
    serde_json::to_writer_pretty(&mut report_file, &report)?;
    report_file.write_all(b"\n")?;
    report_file.finish("run.json")?;
    export.complete()?;
    drop(export);
    eprintln!(
        "[decoder-compact-probe] blocks={} nodes={} vocabulary={} original_budget_receipt={:?}",
        completed_blocks,
        trace.steps().len(),
        shape[1],
        budget.receipt()?
    );
    Ok(())
}
