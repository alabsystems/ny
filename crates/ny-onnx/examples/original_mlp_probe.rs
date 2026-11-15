// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Qualify the original Pythia block-5 affine/GELU-erf/affine fragment.
//!
//! `targo --unverified run -p ny-onnx --example original_mlp_probe -- request.json`
//! Closed request: controls_manifest, controls_manifest_sha256, output_directory.
//! The pinned nine boxes are explicit post_attention_norm assumptions. This
//! driver neither executes a point decoder nor certifies an upstream trajectory,
//! a deployed floating-point kernel, UNSAT, or complete vocabulary coverage.
//! F32LE exports are C-order and retain BOTH coefficient uncertainty planes.
//! Replay prepares fresh contexts over the same immutable original owners;
//! it is an original-source replay, not a second independent arithmetic method.

use ndarray::{ArrayBase, ArrayD, Data, Dimension, IxDyn};
use ny_onnx::native::decoder_contract::{
    load_decoder_safetensors, lower_decoder, DecoderArchitecture, DecoderContract, DecoderOptions,
    DecoderWeights, LoweringManifest, OperationIdentity,
};
use ny_propagate::certify::scope::{Digest, ScopeIdentity};
use ny_propagate::certify::work::{LeaseClass, WorkBudget, WorkLease, WorkReceipt, WorkRequest};
use ny_propagate::network::original_affine_chain::{
    OriginalChainNode, PreparedOriginalAffineChain, ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE,
};
use ny_propagate::network::original_affine_proof::{
    OriginalAffineIds, OriginalAffineOptions, OriginalLogitDifference, PreparedOriginalAffine,
};
use ny_propagate::{GraphNetwork, Layer, LinearBounds};
use ny_tensor::BoundedTensor;
use safetensors::SafeTensors;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

type AnyError = Box<dyn Error + Send + Sync>;
type AnyResult<T> = Result<T, AnyError>;
const MANIFEST_SHA: &str = "8e151b8cd1fa178864005432ace36b72aea9b56affbd42d7e0a41735fa3bec62";
const MODEL_SHA: &str = "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1";
const CONFIG_SHA: &str = "f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17";
const REVISION: &str = "cf967c0a9a04383db6f7b1108d86b2962634b4ac";
const JSON_CAP: usize = 1 << 20;
const OUTPUT_CAP: usize = 16 << 20;
const MODEL_CAP: usize = 32 << 20;
const POLL: usize = 1024;
const DRIVER_REQUEST: WorkRequest = WorkRequest {
    scratch_bytes: 8 << 20,
    persistent_bytes: 64 << 20,
    work: 1 << 30,
    proposals: 1,
};
const MLP_NAMES: [&str; 3] = [
    "decoder.block.5.mlp_up",
    "decoder.block.5.gelu_erf",
    "decoder.block.5.mlp_down",
];
const BOUNDARY: &str = "decoder.block.5.post_attention_norm:out";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    controls_manifest: PathBuf,
    controls_manifest_sha256: String,
    output_directory: PathBuf,
}
#[derive(Deserialize, Serialize)]
struct InputFile {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
struct TensorControl {
    original_name: String,
    original_dtype: String,
    shape: Vec<usize>,
    original_data_offsets: [usize; 2],
    original_tensor_bytes_sha256: String,
    decoded_f32le: InputFile,
}
#[derive(Deserialize)]
struct Case {
    observation_index: usize,
    input_shape: Vec<usize>,
    adjacent_binary32_steps_each_side: usize,
    original_observation: InputFile,
    original_observed_input: InputFile,
    original_lowering_digest: [u8; 32],
    input_lower_f32le: InputFile,
    input_upper_f32le: InputFile,
}
#[derive(Deserialize)]
struct Controls {
    schema: String,
    certificate_authority: bool,
    model_repository: String,
    model_revision: String,
    original_model: InputFile,
    original_config: InputFile,
    block_index: usize,
    operator_sequence: Vec<String>,
    domain: String,
    tensors: BTreeMap<String, TensorControl>,
    cases: Vec<Case>,
}
// Only bounded metadata is retained; unrelated archived tensor/trace values
// are skipped by serde rather than materialized as a large Value tree.
#[derive(Deserialize)]
struct ArchivedObservation {
    boundary: ArchivedBoundary,
}
#[derive(Deserialize)]
struct ArchivedBoundary {
    original_lowering_digest: [u8; 32],
    reference_profile: String,
    block_index: usize,
    positions: Vec<u64>,
    outputs: Vec<ArchivedTensor>,
}
#[derive(Deserialize)]
struct ArchivedTensor {
    node: String,
    tensor: String,
    original_shape: Vec<usize>,
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
struct Export {
    file: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: usize,
    sha256: [u8; 32],
}
#[derive(Serialize)]
struct RowExport {
    lower_a: Export,
    lower_b: Export,
    upper_a: Export,
    upper_b: Export,
    lower_a_err: Option<Export>,
    upper_a_err: Option<Export>,
    nonzero_lower_error_cells: usize,
    nonzero_upper_error_cells: usize,
}
#[derive(Serialize)]
struct Demand {
    target: usize,
    competitor: usize,
    position: usize,
}
impl From<OriginalLogitDifference> for Demand {
    fn from(d: OriginalLogitDifference) -> Self {
        Self {
            target: d.target,
            competitor: d.competitor,
            position: d.position,
        }
    }
}
#[derive(Serialize)]
struct Binding {
    query: Digest,
    output: Digest,
}
#[derive(Serialize)]
struct CaseReport {
    observation_index: usize,
    adjacent_binary32_steps_each_side: usize,
    input_shape: Vec<usize>,
    archived_lowering_digest: [u8; 32],
    archived_reference_profile: String,
    current_lowering_digest: [u8; 32],
    current_reference_profile: String,
    exact_real_chain_profile: &'static str,
    scope: ScopeIdentity,
    source_head_scope: ScopeIdentity,
    actual_checked_nodes: usize,
    exact_erf_gelu_facts: usize,
    interval_binding: Binding,
    signed_binding: Binding,
    sparse_binding: Binding,
    source_head_binding: Binding,
    sparse_feature_differences: Vec<Demand>,
    source_head_differences: Vec<Demand>,
    signed_demand_semantics: &'static str,
    interval_lower: Export,
    interval_upper: Export,
    incoming_signed_rows: RowExport,
    signed_input_rows: RowExport,
    sparse_input_rows: RowExport,
    original_source_replay_bit_identical: bool,
    replay_algorithm: &'static str,
    replay_admitted_phases: u64,
    forward_requested_bytes: usize,
    signed_requested_bytes: usize,
    sparse_requested_bytes: usize,
    shared_original_owner_requested_bytes: usize,
    same_ledger_before: Receipt,
    same_ledger_after: Receipt,
}
#[derive(Serialize)]
struct GroupReport {
    observation_index: usize,
    positions: Vec<u64>,
    current_lowering_digest: [u8; 32],
    manifest_file: Export,
    graph_admission: Capacity,
    source_requested_bytes: usize,
    graph_requested_bytes: usize,
    constructed_original_graph_nodes: usize,
    selected_original_node_names: [&'static str; 3],
    ordinary_graph_proof_eligibility_refusal: String,
}
#[derive(Serialize)]
struct RunReport {
    schema: &'static str,
    network_certificate_authority: bool,
    checked_local_fragment: bool,
    input_assumption: &'static str,
    numerical_profile: &'static str,
    point_decoder_executions: usize,
    upstream_decoder_claim: bool,
    deployed_kernel_claim: bool,
    full_vocabulary_claim: bool,
    runtime_claim: bool,
    capacity_semantics: &'static str,
    receipt_snapshot: &'static str,
    controls_manifest_sha256: &'static str,
    original_model_sha256: &'static str,
    original_config_sha256: &'static str,
    driver_source_sha256: [u8; 32],
    default_max_resident_bytes: usize,
    default_max_scratch_bytes: usize,
    default_max_work: u64,
    default_max_proposals: u64,
    deadline: Option<&'static str>,
    driver_admission: Capacity,
    driver_units_spent: u64,
    groups: Vec<GroupReport>,
    cases: Vec<CaseReport>,
    original_budget_receipt: Receipt,
}

/// The driver's fixed prepaid allowance is part of the ORIGINAL caller ledger.
/// Its metadata/raw source/output buffers stay under this lease until exit.
/// Inspection, copies, SHA input and JSON serialization spend bounded credits;
/// no failed query or sequence group receives a replacement allowance.
struct Driver {
    lease: WorkLease,
    left: Cell<u64>,
}
impl Driver {
    fn poll(&self) -> AnyResult<()> {
        self.lease.checkpoint()?;
        if ny_core::instance_budget::deadline().is_some_and(|d| std::time::Instant::now() >= d) {
            return Err("original instance deadline".into());
        }
        Ok(())
    }
    fn spend(&self, units: usize) -> AnyResult<()> {
        self.poll()?;
        let left = self
            .left
            .get()
            .checked_sub(u64::try_from(units)?)
            .ok_or("prepaid driver work exhausted")?;
        self.left.set(left);
        Ok(())
    }
    fn read(&self, path: &Path, cap: usize) -> AnyResult<Vec<u8>> {
        self.poll()?;
        let mut file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())?;
        if len > cap {
            return Err("source exceeds driver byte cap".into());
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len)?;
        if bytes.capacity() > len.saturating_add(128) {
            return Err("source allocation exceeds driver capacity prediction".into());
        }
        let mut part = [0u8; 4096];
        loop {
            self.poll()?;
            let n = file.read(&mut part)?;
            if n == 0 {
                break;
            }
            if bytes.len().checked_add(n).is_none_or(|n| n > len) {
                return Err("source grew during admitted read".into());
            }
            self.spend(n)?;
            bytes.extend_from_slice(&part[..n]);
        }
        if bytes.len() != len {
            return Err("source shrank during admitted read".into());
        }
        Ok(bytes)
    }
    fn hash(&self, bytes: &[u8]) -> AnyResult<[u8; 32]> {
        let mut hash = Sha256::new();
        for part in bytes.chunks(4096) {
            self.spend(part.len())?;
            hash.update(part);
        }
        self.poll()?;
        Ok(hash.finalize().into())
    }
    fn json<T: DeserializeOwned>(&self, bytes: &[u8]) -> AnyResult<T> {
        // Closed/pinned metadata is <=1 MiB. The 32*bytes allowance covers
        // bounded parsing and string/array copies; a final poll precedes use.
        self.spend(bytes.len().checked_mul(32).ok_or("JSON work overflow")?)?;
        let result = serde_json::from_slice(bytes)?;
        self.poll()?;
        Ok(result)
    }
    fn checked_read(&self, source: &InputFile, cap: usize) -> AnyResult<Vec<u8>> {
        if source.bytes > cap {
            return Err("manifest source size exceeds cap".into());
        }
        let bytes = self.read(&source.path, cap)?;
        if bytes.len() != source.bytes || self.hash(&bytes)? != digest_hex(&source.sha256)? {
            return Err(format!("source hash/size mismatch: {}", source.path.display()).into());
        }
        Ok(bytes)
    }
}

fn digest_hex(text: &str) -> AnyResult<[u8; 32]> {
    if text.len() != 64 || !text.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid SHA256 spelling".into());
    }
    let mut out = [0; 32];
    for (i, cell) in out.iter_mut().enumerate() {
        *cell = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)?;
    }
    Ok(out)
}

struct PolledWriter<'a> {
    inner: BufWriter<File>,
    driver: &'a Driver,
    bytes: usize,
    hash: Sha256,
}
impl Write for PolledWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self
            .bytes
            .checked_add(bytes.len())
            .filter(|&n| n <= OUTPUT_CAP)
            .ok_or_else(|| io::Error::other("output byte cap"))?;
        for part in bytes.chunks(4096) {
            self.driver
                .spend(part.len().saturating_mul(8))
                .map_err(io::Error::other)?;
            self.inner.write_all(part)?;
            self.hash.update(part);
        }
        self.bytes = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.driver.poll().map_err(io::Error::other)?;
        self.inner.flush()
    }
}
fn writer<'a>(directory: &Path, name: &str, driver: &'a Driver) -> AnyResult<PolledWriter<'a>> {
    driver.poll()?;
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(directory.join(name))?;
    Ok(PolledWriter {
        inner: BufWriter::with_capacity(16 << 10, file),
        driver,
        bytes: 0,
        hash: Sha256::new(),
    })
}
fn write_json(
    directory: &Path,
    name: String,
    value: &impl Serialize,
    driver: &Driver,
) -> AnyResult<Export> {
    let mut out = writer(directory, &name, driver)?;
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(Export {
        file: name,
        shape: Vec::new(),
        elements: 0,
        bytes: out.bytes,
        sha256: out.hash.finalize().into(),
    })
}
fn export_array<S: Data<Elem = f32>, D: Dimension>(
    directory: &Path,
    name: String,
    array: &ArrayBase<S, D>,
    driver: &Driver,
) -> AnyResult<Export> {
    if array.len().checked_mul(4).is_none_or(|n| n > OUTPUT_CAP) {
        return Err("tensor output cap".into());
    }
    let mut out = writer(directory, &name, driver)?;
    let mut part = [0u8; POLL * 4];
    let mut used = 0;
    for &value in array {
        if used == 0 {
            driver.poll()?;
        }
        if !value.is_finite() {
            return Err("nonfinite checked result cannot qualify this probe".into());
        }
        part[used * 4..used * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        used += 1;
        if used == POLL {
            out.write_all(&part)?;
            used = 0;
        }
    }
    out.write_all(&part[..used * 4])?;
    out.flush()?;
    Ok(Export {
        file: name,
        shape: array.shape().to_vec(),
        elements: array.len(),
        bytes: out.bytes,
        sha256: out.hash.finalize().into(),
    })
}
fn count_error<S: Data<Elem = f32>, D: Dimension>(
    error: Option<&ArrayBase<S, D>>,
    driver: &Driver,
) -> AnyResult<usize> {
    let mut count = 0;
    if let Some(error) = error {
        for (i, &value) in error.iter().enumerate() {
            if i.is_multiple_of(POLL) {
                driver.spend((error.len() - i).min(POLL))?;
            }
            if !value.is_finite() || value < 0.0 || (value.is_sign_negative() && value != 0.0) {
                return Err("invalid checked error plane".into());
            }
            count += usize::from(value != 0.0);
        }
    }
    Ok(count)
}
fn export_rows(
    directory: &Path,
    prefix: &str,
    rows: &LinearBounds,
    driver: &Driver,
) -> AnyResult<RowExport> {
    Ok(RowExport {
        lower_a: export_array(
            directory,
            format!("{prefix}.lower_a.f32le"),
            rows.lower_a(),
            driver,
        )?,
        lower_b: export_array(
            directory,
            format!("{prefix}.lower_b.f32le"),
            rows.lower_b(),
            driver,
        )?,
        upper_a: export_array(
            directory,
            format!("{prefix}.upper_a.f32le"),
            rows.upper_a(),
            driver,
        )?,
        upper_b: export_array(
            directory,
            format!("{prefix}.upper_b.f32le"),
            rows.upper_b(),
            driver,
        )?,
        lower_a_err: rows
            .lower_a_err()
            .map(|e| export_array(directory, format!("{prefix}.lower_a_err.f32le"), e, driver))
            .transpose()?,
        upper_a_err: rows
            .upper_a_err()
            .map(|e| export_array(directory, format!("{prefix}.upper_a_err.f32le"), e, driver))
            .transpose()?,
        nonzero_lower_error_cells: count_error(rows.lower_a_err(), driver)?,
        nonzero_upper_error_cells: count_error(rows.upper_a_err(), driver)?,
    })
}

fn equal_array<S: Data<Elem = f32>, T: Data<Elem = f32>, D: Dimension>(
    a: &ArrayBase<S, D>,
    b: &ArrayBase<T, D>,
    driver: &Driver,
) -> AnyResult<()> {
    if a.shape() != b.shape() {
        return Err("original replay shape mismatch".into());
    }
    let cells = a.len();
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        if i.is_multiple_of(POLL) {
            driver.spend((cells - i).min(POLL) * 2)?;
        }
        if a.to_bits() != b.to_bits() {
            return Err("original replay differs in stored bits".into());
        }
    }
    Ok(())
}
fn equal_rows(a: &LinearBounds, b: &LinearBounds, driver: &Driver) -> AnyResult<()> {
    equal_array(a.lower_a(), b.lower_a(), driver)?;
    equal_array(a.upper_a(), b.upper_a(), driver)?;
    equal_array(a.lower_b(), b.lower_b(), driver)?;
    equal_array(a.upper_b(), b.upper_b(), driver)?;
    for (a, b) in [
        (a.lower_a_err(), b.lower_a_err()),
        (a.upper_a_err(), b.upper_a_err()),
    ] {
        match (a, b) {
            (None, None) => {}
            (Some(a), Some(b)) => equal_array(a, b, driver)?,
            _ => return Err("original replay lost an error plane".into()),
        }
    }
    Ok(())
}

fn has_both_signs(rows: &LinearBounds, driver: &Driver) -> AnyResult<bool> {
    let a = rows.lower_a();
    let mut negative = false;
    let mut positive = false;
    for (i, &value) in a.iter().enumerate() {
        if i.is_multiple_of(POLL) {
            driver.spend((a.len() - i).min(POLL))?;
        }
        negative |= value < 0.0;
        positive |= value > 0.0;
    }
    Ok(negative && positive)
}

fn decode_f32(bytes: &[u8], driver: &Driver) -> AnyResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) || bytes.len() > JSON_CAP {
        return Err("invalid bounded F32LE input".into());
    }
    let mut result = Vec::new();
    result.try_reserve_exact(bytes.len() / 4)?;
    if result.capacity().saturating_mul(4) > bytes.len().saturating_add(128) {
        return Err("endpoint capacity exceeded prediction".into());
    }
    for (i, part) in bytes.as_chunks::<4>().0.iter().enumerate() {
        if i.is_multiple_of(POLL) {
            driver.spend((bytes.len() / 4 - i).min(POLL) * 8)?;
        }
        let value = f32::from_bits(u32::from_le_bytes(*part));
        if !value.is_finite() {
            return Err("nonfinite boundary assumption".into());
        }
        result.push(value);
    }
    Ok(result)
}
fn adjacent(mut value: f32, steps: usize, up: bool) -> f32 {
    for _ in 0..steps {
        value = if value == 0.0 {
            f32::from_bits(if up { 1 } else { 0x8000_0001 })
        } else {
            let bits = value.to_bits();
            let increment = up == value.is_sign_positive();
            f32::from_bits(if increment { bits + 1 } else { bits - 1 })
        };
    }
    value
}
fn input_box(case: &Case, driver: &Driver) -> AnyResult<BoundedTensor> {
    let base = decode_f32(
        &driver.checked_read(&case.original_observed_input, JSON_CAP)?,
        driver,
    )?;
    let lo = decode_f32(
        &driver.checked_read(&case.input_lower_f32le, JSON_CAP)?,
        driver,
    )?;
    let hi = decode_f32(
        &driver.checked_read(&case.input_upper_f32le, JSON_CAP)?,
        driver,
    )?;
    let cells = case
        .input_shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("boundary shape overflow")?;
    if base.len() != cells || lo.len() != cells || hi.len() != cells {
        return Err("boundary storage shape mismatch".into());
    }
    for (i, ((&base, &lo), &hi)) in base.iter().zip(&lo).zip(&hi).enumerate() {
        if i.is_multiple_of(POLL) {
            driver
                .spend((cells - i).min(POLL) * (8 + case.adjacent_binary32_steps_each_side * 2))?;
        }
        if lo.to_bits() != adjacent(base, case.adjacent_binary32_steps_each_side, false).to_bits()
            || hi.to_bits()
                != adjacent(base, case.adjacent_binary32_steps_each_side, true).to_bits()
            || lo > hi
        {
            return Err("staged box is not the declared original adjacent-bit box".into());
        }
    }
    driver.spend(cells.checked_mul(8).ok_or("box validation overflow")?)?;
    let shape = IxDyn(&case.input_shape);
    let lo = ArrayD::from_shape_vec(shape.clone(), lo)?;
    let hi = ArrayD::from_shape_vec(shape, hi)?;
    Ok(BoundedTensor::new_allow_infinite_with_poll(lo, hi, || {
        driver
            .poll()
            .map_err(|e| ny_core::NyError::DeadlineExceeded(e.to_string()))
    })?)
}

fn validate_controls(c: &Controls) -> AnyResult<()> {
    if c.schema != "ny.original_pythia_mlp_controls.v1"
        || c.certificate_authority
        || c.model_repository != "EleutherAI/pythia-14m"
        || c.model_revision != REVISION
        || c.original_model.sha256 != MODEL_SHA
        || c.original_config.sha256 != CONFIG_SHA
        || c.block_index != 5
        || c.operator_sequence != ["OriginalAffine", "ExactErfGELU", "OriginalAffine"]
        || c.domain != "explicit conditional post_attention_norm binary32 endpoint box; ExactReal original decoded coefficients"
        || c.tensors.len() != 4
        || c.cases.len() != 9
    {
        return Err("pinned control manifest contract mismatch".into());
    }
    let mut cover = BTreeSet::new();
    for case in &c.cases {
        let sequence = match case.observation_index {
            0 => 5,
            2 => 7,
            4 => 4,
            _ => 0,
        };
        if sequence == 0
            || case.input_shape != [1, sequence, 128]
            || ![0, 2, 16].contains(&case.adjacent_binary32_steps_each_side)
            || !cover.insert((
                case.observation_index,
                case.adjacent_binary32_steps_each_side,
            ))
        {
            return Err("incomplete or duplicated nine-box control cover".into());
        }
    }
    Ok(())
}

fn source_identity_hash(name: &str, shape: &[usize], dtype: &str) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-decoder-tensor/v1");
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((dtype.len() as u64).to_le_bytes());
    hash.update(dtype.as_bytes());
    hash.update((shape.len() as u64).to_le_bytes());
    for &dim in shape {
        hash.update((dim as u64).to_le_bytes());
    }
    hash
}
fn validate_loaded_sources(
    weights: &DecoderWeights,
    frozen: &SafeTensors<'_>,
    driver: &Driver,
) -> AnyResult<()> {
    let mut count = 0;
    for identity in weights.identities() {
        driver.spend(64 + identity.name.len())?;
        let tensor = frozen.tensor(&identity.name)?;
        let dtype = format!("{:?}", tensor.dtype());
        if identity.shape != tensor.shape() || identity.source_dtype != dtype {
            return Err("native original tensor identity differs from pinned source".into());
        }
        let mut hash = source_identity_hash(&identity.name, &identity.shape, &dtype);
        for part in tensor.data().chunks(4096) {
            driver.spend(part.len())?;
            hash.update(part);
        }
        if identity.source_digest != <[u8; 32]>::from(hash.finalize()) {
            return Err("native loader source differs from immutable pinned archive".into());
        }
        count += 1;
    }
    if count != frozen.len() {
        return Err("native source tensor coverage mismatch".into());
    }
    Ok(())
}
fn verify_staged_tensors(
    c: &Controls,
    frozen: &SafeTensors<'_>,
    source: &[u8],
    driver: &Driver,
) -> AnyResult<()> {
    let header = usize::try_from(u64::from_le_bytes(
        source.get(..8).ok_or("short original header")?.try_into()?,
    ))?;
    let data_start = header.checked_add(8).ok_or("source header overflow")?;
    for control in c.tensors.values() {
        let tensor = frozen.tensor(&control.original_name)?;
        let start = data_start
            .checked_add(control.original_data_offsets[0])
            .ok_or("offset overflow")?;
        let end = data_start
            .checked_add(control.original_data_offsets[1])
            .ok_or("offset overflow")?;
        let bytes = source
            .get(start..end)
            .ok_or("original tensor offset range")?;
        if control.original_dtype != format!("{:?}", tensor.dtype())
            || control.shape != tensor.shape()
            || tensor.data() != bytes
            || driver.hash(bytes)? != digest_hex(&control.original_tensor_bytes_sha256)?
        {
            return Err("original tensor slice/shape/type hash mismatch".into());
        }
        let _ = driver.checked_read(&control.decoded_f32le, JSON_CAP)?;
    }
    Ok(())
}
fn compare_original_array<S: Data<Elem = f32>, D: Dimension>(
    array: &ArrayBase<S, D>,
    control: &TensorControl,
    driver: &Driver,
) -> AnyResult<()> {
    if array.shape() != control.shape
        || array.len().checked_mul(4) != Some(control.decoded_f32le.bytes)
    {
        return Err("actual compact original coefficient shape mismatch".into());
    }
    let mut hash = Sha256::new();
    for (i, &v) in array.iter().enumerate() {
        if i.is_multiple_of(POLL) {
            driver.spend((array.len() - i).min(POLL) * 8)?;
        }
        if !v.is_finite() {
            return Err("nonfinite original coefficient".into());
        }
        hash.update(v.to_bits().to_le_bytes());
    }
    if <[u8; 32]>::from(hash.finalize()) != digest_hex(&control.decoded_f32le.sha256)? {
        return Err("actual compact coefficients differ from pinned decoded F32 bits".into());
    }
    Ok(())
}

fn operation<'a>(manifest: &'a LoweringManifest, name: &str) -> AnyResult<&'a OperationIdentity> {
    let mut matches = manifest
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| op.name == name);
    let op = matches
        .next()
        .ok_or("missing original operation identity")?;
    if matches.next().is_some() {
        return Err("duplicate original operation identity".into());
    }
    Ok(op)
}
fn ids(
    manifest: &LoweringManifest,
    name: &str,
    graph: &GraphNetwork,
    driver: &Driver,
) -> AnyResult<OriginalAffineIds> {
    driver.spend(4096)?;
    let op = operation(manifest, name)?;
    let node = graph
        .node(name)
        .ok_or("missing actual authored graph node")?;
    // Lowering manifests retain immutable weight/bias inputs. The constructed
    // graph has one activation edge because those parameters are owned by the
    // checked layer. Authenticate that correspondence before using the edge.
    let parameter_count = match node.layer() {
        Layer::PointLinear(affine)
            if op.operator == "Linear"
                && op.parameters.get("transB") == Some(&serde_json::json!(1)) =>
        {
            1 + usize::from(affine.bias().is_some())
        }
        Layer::GELU(gelu)
            if op.operator == "GELU"
                && gelu.approximation == ny_propagate::layers::GeluApproximation::Erf
                && op.parameters.get("approximate") == Some(&serde_json::json!("none")) =>
        {
            0
        }
        _ => return Err("unsupported original unary manifest profile".into()),
    };
    if op.inputs.len() != parameter_count + 1
        || op.tensors.len() != parameter_count
        || !op.inputs[1..]
            .iter()
            .zip(&op.tensors)
            .all(|(input, tensor)| input == &tensor.name)
        || node.inputs().len() != 1
        || node.name() != op.name
    {
        return Err("authored unary chain topology mismatch".into());
    }
    let producer = manifest
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find(|p| p.output == op.inputs[0]);
    let expected = producer.map_or(op.inputs[0].as_str(), |p| p.name.as_str());
    if node.inputs()[0] != expected {
        return Err("actual original authored producer edge mismatch".into());
    }
    Ok(OriginalAffineIds {
        lowering: Digest(manifest.lowering_digest),
        node: Digest::of_bytes(b"original-mlp-authored-node/v1", op.name.as_bytes()),
        input: Digest::of_bytes(b"original-mlp-authored-tensor/v1", op.inputs[0].as_bytes()),
        output: Digest::of_bytes(b"original-mlp-authored-tensor/v1", op.output.as_bytes()),
    })
}

fn main() -> AnyResult<()> {
    let defaults = DecoderOptions::default();
    // No new duration/allowance is created for a load, group, or replay.
    let inherited_deadline = ny_core::instance_budget::deadline();
    let budget = WorkBudget::new(
        inherited_deadline,
        defaults.max_scratch_bytes,
        defaults.max_resident_bytes,
        defaults.max_terms,
        defaults.max_proposals,
        Duration::ZERO,
    )?;
    let mut driver = Driver {
        lease: budget.lease(DRIVER_REQUEST)?,
        left: Cell::new(DRIVER_REQUEST.work),
    };
    let mut args = std::env::args_os().skip(1);
    let request_file = args
        .next()
        .ok_or("usage: original_mlp_probe request.json")?;
    if args.next().is_some() {
        return Err("usage: original_mlp_probe request.json".into());
    }
    let request: Request = driver.json(&driver.read(Path::new(&request_file), JSON_CAP)?)?;
    if request.controls_manifest_sha256 != MANIFEST_SHA {
        return Err("requires pinned nine-box manifest SHA".into());
    }
    let bytes = driver.read(&request.controls_manifest, JSON_CAP)?;
    if driver.hash(&bytes)? != digest_hex(MANIFEST_SHA)? {
        return Err("control manifest SHA mismatch".into());
    }
    let controls: Controls = driver.json(&bytes)?;
    validate_controls(&controls)?;
    let source = driver.checked_read(&controls.original_model, MODEL_CAP)?;
    driver.spend(JSON_CAP * 16)?; // original SafeTensors header metadata is separately bounded below
    let header = usize::try_from(u64::from_le_bytes(
        source.get(..8).ok_or("short model")?.try_into()?,
    ))?;
    if header > JSON_CAP {
        return Err("pinned model header cap".into());
    }
    let frozen = SafeTensors::deserialize(&source)?;
    verify_staged_tensors(&controls, &frozen, &source, &driver)?;
    let config = driver.checked_read(&controls.original_config, defaults.max_config_bytes)?;
    let options = DecoderOptions {
        budget: Some(budget.clone()),
        deadline: inherited_deadline,
        ..defaults
    };
    let contract = DecoderContract::from_json(&config, &options)?;
    let p = contract.parameters();
    if p.architecture != DecoderArchitecture::GptNeoX
        || p.hidden_size != 128
        || p.intermediate_size != 512
        || p.num_layers != 6
        || p.vocab_size != 50304
        || !p.parallel_residual
    {
        return Err("actual pinned Pythia config contract mismatch".into());
    }
    driver.poll()?;
    std::fs::create_dir(&request.output_directory)?;
    let mut groups = Vec::with_capacity(3);
    let mut cases = Vec::with_capacity(9);
    for observation_index in [0, 2, 4] {
        let group: Vec<_> = controls
            .cases
            .iter()
            .filter(|c| c.observation_index == observation_index)
            .collect();
        let archived: ArchivedObservation =
            driver.json(&driver.checked_read(&group[0].original_observation, JSON_CAP)?)?;
        let boundary = &archived.boundary;
        let sequence = group[0].input_shape[1];
        if boundary.block_index != 5
            || boundary.positions.len() != sequence
            || boundary
                .positions
                .iter()
                .enumerate()
                .any(|(i, &p)| p != i as u64)
            || boundary.original_lowering_digest != group[0].original_lowering_digest
            || !boundary.outputs.iter().any(|o| {
                o.node == "decoder.block.5.post_attention_norm"
                    && o.tensor == BOUNDARY
                    && o.original_shape == group[0].input_shape
            })
        {
            return Err("archived original post-normalization boundary mismatch".into());
        }
        let weights = load_decoder_safetensors(&controls.original_model.path, &options)?;
        validate_loaded_sources(&weights, &frozen, &driver)?;
        let model = lower_decoder(weights, &contract, &boundary.positions, &options)?;
        if !model.budget().shares_ledger(&budget) {
            return Err("native model replaced caller ledger".into());
        }
        let graph_admission = model.graph_admission(&options)?;
        let graph = model.graph_for("logits", &options)?; // ONE construction; no point execution
        let ordinary_refusal = graph
            .graph()
            .ensure_proof_eligible()
            .expect_err("PointLinear quarantine must remain")
            .to_string();
        if !ordinary_refusal.contains("PointLinear") {
            return Err("unexpected ordinary proof eligibility result".into());
        }
        let manifest = model.manifest();
        let nodes = MLP_NAMES.map(|name| {
            Ok::<_, AnyError>(OriginalChainNode {
                ids: ids(manifest, name, graph.graph(), &driver)?,
                layer: graph
                    .graph()
                    .node(name)
                    .ok_or("missing MLP original node")?
                    .layer(),
            })
        });
        let [up, gelu, down] = nodes;
        let nodes = [up?, gelu?, down?];
        let [Layer::PointLinear(up), Layer::GELU(gelu), Layer::PointLinear(down)] =
            nodes.map(|n| n.layer)
        else {
            return Err("actual compact original MLP layer types differ".into());
        };
        if gelu.approximation != ny_propagate::layers::GeluApproximation::Erf
            || operation(manifest, MLP_NAMES[0])?.inputs[0] != BOUNDARY
            || operation(manifest, MLP_NAMES[1])?.inputs[0]
                != operation(manifest, MLP_NAMES[0])?.output
            || operation(manifest, MLP_NAMES[2])?.inputs[0]
                != operation(manifest, MLP_NAMES[1])?.output
        {
            return Err("original authored exact-erf MLP connection mismatch".into());
        }
        compare_original_array(
            up.weight(),
            controls
                .tensors
                .get("up_weight")
                .ok_or("missing up weights")?,
            &driver,
        )?;
        compare_original_array(
            up.bias().ok_or("original up bias absent")?,
            controls.tensors.get("up_bias").ok_or("missing up bias")?,
            &driver,
        )?;
        compare_original_array(
            down.weight(),
            controls
                .tensors
                .get("down_weight")
                .ok_or("missing down weights")?,
            &driver,
        )?;
        compare_original_array(
            down.bias().ok_or("original down bias absent")?,
            controls
                .tensors
                .get("down_bias")
                .ok_or("missing down bias")?,
            &driver,
        )?;
        let head_name = "decoder.logits";
        let head_ids = ids(manifest, head_name, graph.graph(), &driver)?;
        let Layer::PointLinear(head) = graph
            .graph()
            .node(head_name)
            .ok_or("original head missing")?
            .layer()
        else {
            return Err("original head is not compact immutable affine owner".into());
        };
        let optional = OriginalAffineOptions {
            budget: &budget,
            class: LeaseClass::Optional,
            deadline: options.deadline,
        };
        let replay = OriginalAffineOptions {
            class: LeaseClass::Replay,
            ..optional
        };
        let chain = PreparedOriginalAffineChain::prepare(&nodes, optional)?;
        let head_context = PreparedOriginalAffine::prepare(head, head_ids, optional)?;
        let manifest_file = write_json(
            &request.output_directory,
            format!("group-{observation_index:02}.lowering.json"),
            manifest,
            &driver,
        )?;
        groups.push(GroupReport {
            observation_index,
            positions: boundary.positions.clone(),
            current_lowering_digest: manifest.lowering_digest,
            manifest_file,
            graph_admission: graph_admission.into(),
            source_requested_bytes: model.reserved_bytes(),
            graph_requested_bytes: graph.reserved_bytes(),
            constructed_original_graph_nodes: graph.graph().node_names().len(),
            selected_original_node_names: MLP_NAMES,
            ordinary_graph_proof_eligibility_refusal: ordinary_refusal,
        });
        for case in group {
            if case.original_lowering_digest != boundary.original_lowering_digest
                || case.original_observation.sha256
                    != group_observation_sha(&controls, observation_index)?
            {
                return Err("group archived identity changed between boxes".into());
            }
            let before = budget.receipt()?;
            let input = input_box(case, &driver)?;
            let facts = chain.forward(&input, optional)?;
            if facts.num_nodes() != 3 || facts.bounds().shape() != case.input_shape {
                return Err("original three-node forward did not engage".into());
            }
            let feature_differences = [
                OriginalLogitDifference {
                    target: 0,
                    competitor: 1,
                    position: 0,
                },
                OriginalLogitDifference {
                    target: 127,
                    competitor: 126,
                    position: sequence - 1,
                },
                OriginalLogitDifference {
                    target: 1,
                    competitor: 0,
                    position: sequence - 1,
                },
            ];
            let head_differences = [
                OriginalLogitDifference {
                    target: 0,
                    competitor: 1,
                    position: 0,
                },
                OriginalLogitDifference {
                    target: 1,
                    competitor: 0,
                    position: sequence - 1,
                },
                OriginalLogitDifference {
                    target: 50303,
                    competitor: 50299,
                    position: sequence - 1,
                },
            ];
            // A separately certified ORIGINAL head row is used as a signed
            // linear demand on the MLP output. This is hypothetical composition,
            // not the missing final norm/decoder residual or a vocabulary claim.
            let source_rows = head_context.sparse_margins(sequence, &head_differences, optional)?;
            if count_error(source_rows.bounds().lower_a_err(), &driver)? == 0
                || count_error(source_rows.bounds().upper_a_err(), &driver)? == 0
                || !has_both_signs(source_rows.bounds(), &driver)?
            {
                return Err(
                    "actual source demand did not exercise signed nonzero error planes".into(),
                );
            }
            let signed = chain.signed_linear(&facts, source_rows.bounds(), optional)?;
            let sparse = chain.sparse_margins(&facts, &feature_differences, optional)?;
            let replay_before = budget.receipt()?;
            let fresh_chain = PreparedOriginalAffineChain::prepare(&nodes, replay)?;
            let fresh_head = PreparedOriginalAffine::prepare(head, head_ids, replay)?;
            let fresh_facts = fresh_chain.forward(&input, replay)?;
            let fresh_source = fresh_head.sparse_margins(sequence, &head_differences, replay)?;
            let fresh_signed =
                fresh_chain.signed_linear(&fresh_facts, fresh_source.bounds(), replay)?;
            let fresh_sparse =
                fresh_chain.sparse_margins(&fresh_facts, &feature_differences, replay)?;
            equal_array(
                facts.bounds().lower(),
                fresh_facts.bounds().lower(),
                &driver,
            )?;
            equal_array(
                facts.bounds().upper(),
                fresh_facts.bounds().upper(),
                &driver,
            )?;
            equal_rows(source_rows.bounds(), fresh_source.bounds(), &driver)?;
            equal_rows(signed.bounds(), fresh_signed.bounds(), &driver)?;
            equal_rows(sparse.bounds(), fresh_sparse.bounds(), &driver)?;
            if facts.binding() != fresh_facts.binding()
                || signed.binding() != fresh_signed.binding()
                || sparse.binding() != fresh_sparse.binding()
                || source_rows.binding() != fresh_source.binding()
                || facts.scope() != fresh_facts.scope()
                || source_rows.scope() != fresh_source.scope()
            {
                return Err("fresh original replay scope/query binding mismatch".into());
            }
            let replay_after = budget.receipt()?;
            if replay_after.admitted <= replay_before.admitted
                || replay_after.work_reserved <= replay_before.work_reserved
            {
                return Err("fresh replay did not spend original cumulative allowance".into());
            }
            let prefix = format!(
                "fixture-{observation_index:02}-ulp-{}",
                case.adjacent_binary32_steps_each_side
            );
            let report = CaseReport {
                observation_index, adjacent_binary32_steps_each_side: case.adjacent_binary32_steps_each_side,
                input_shape: case.input_shape.clone(), archived_lowering_digest: case.original_lowering_digest,
                archived_reference_profile: boundary.reference_profile.clone(), current_lowering_digest: manifest.lowering_digest,
                current_reference_profile: manifest.parameters.reference_profile.clone(), exact_real_chain_profile: ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE,
                scope: facts.scope().clone(), source_head_scope: source_rows.scope().clone(), actual_checked_nodes: facts.num_nodes(), exact_erf_gelu_facts: 1,
                interval_binding: Binding { query: facts.binding().query, output: facts.binding().output },
                signed_binding: Binding { query: signed.binding().query, output: signed.binding().output },
                sparse_binding: Binding { query: sparse.binding().query, output: sparse.binding().output },
                source_head_binding: Binding { query: source_rows.binding().query, output: source_rows.binding().output },
                sparse_feature_differences: feature_differences.into_iter().map(Demand::from).collect(),
                source_head_differences: head_differences.into_iter().map(Demand::from).collect(),
                signed_demand_semantics: "original lm-head selected target-minus-competitor rows applied to THIS local MLP output; hypothetical affine demand, not decoder/vocabulary closure",
                interval_lower: export_array(&request.output_directory, format!("{prefix}.interval.lower.f32le"), facts.bounds().lower(), &driver)?,
                interval_upper: export_array(&request.output_directory, format!("{prefix}.interval.upper.f32le"), facts.bounds().upper(), &driver)?,
                incoming_signed_rows: export_rows(&request.output_directory, &format!("{prefix}.incoming"), source_rows.bounds(), &driver)?,
                signed_input_rows: export_rows(&request.output_directory, &format!("{prefix}.signed"), signed.bounds(), &driver)?,
                sparse_input_rows: export_rows(&request.output_directory, &format!("{prefix}.sparse"), sparse.bounds(), &driver)?,
                original_source_replay_bit_identical: true, replay_algorithm: "fresh contexts; same original immutable native owners and existing checked exact-real kernels; all six planes and bindings compared",
                replay_admitted_phases: replay_after.admitted - replay_before.admitted,
                forward_requested_bytes: facts.reserved_bytes(), signed_requested_bytes: signed.reserved_bytes(), sparse_requested_bytes: sparse.reserved_bytes(),
                shared_original_owner_requested_bytes: facts.original_owner_bytes(), same_ledger_before: before.into(), same_ledger_after: budget.receipt()?.into(),
            };
            write_json(
                &request.output_directory,
                format!("{prefix}.report.json"),
                &report,
                &driver,
            )?;
            cases.push(report);
        }
    }
    if cases.len() != 9 || groups.len() != 3 {
        return Err("incomplete original control execution".into());
    }
    let driver_source_sha256 = driver.hash(include_bytes!("original_mlp_probe.rs"))?;
    let report = RunReport {
        schema: "ny.original_pythia_mlp_checked_probe.v1", network_certificate_authority: false,
        checked_local_fragment: true, input_assumption: "explicit conditional post_attention_norm endpoint boxes; no upstream proof",
        numerical_profile: ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE, point_decoder_executions: 0, upstream_decoder_claim: false,
        deployed_kernel_claim: false, full_vocabulary_claim: false, runtime_claim: false,
        capacity_semantics: "same-ledger requested leased capacities and reserved work; shared report/owner bytes overlap; not measured RSS or runtime",
        receipt_snapshot: "after all nine controls and their case exports, before final report serialization and driver completion/release",
        controls_manifest_sha256: MANIFEST_SHA, original_model_sha256: MODEL_SHA, original_config_sha256: CONFIG_SHA,
        driver_source_sha256, default_max_resident_bytes: options.max_resident_bytes, default_max_scratch_bytes: options.max_scratch_bytes,
        default_max_work: options.max_terms, default_max_proposals: options.max_proposals,
        deadline: options.deadline.map(|_| "inherited original instance deadline"),
        driver_admission: DRIVER_REQUEST.into(), driver_units_spent: DRIVER_REQUEST.work - driver.left.get(), groups, cases,
        original_budget_receipt: budget.receipt()?.into(),
    };
    write_json(
        &request.output_directory,
        "report.json".into(),
        &report,
        &driver,
    )?;
    driver.poll()?;
    driver.lease.complete()?;
    println!(
        "Qualified 9 conditional original MLP controls; fresh six-plane replay; report={}",
        request.output_directory.join("report.json").display()
    );
    Ok(())
}

fn group_observation_sha(c: &Controls, observation_index: usize) -> AnyResult<&str> {
    c.cases
        .iter()
        .find(|case| case.observation_index == observation_index)
        .map(|case| case.original_observation.sha256.as_str())
        .ok_or_else(|| "missing original observation group".into())
}
