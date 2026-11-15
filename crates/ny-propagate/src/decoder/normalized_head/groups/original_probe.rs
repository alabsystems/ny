// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Explicit original-model conformance for the private grouped prototype.
//!
//! A supplied pre-normalization box is a conditional premise. This driver does
//! not execute a decoder, prove that box contains its trajectory, qualify a
//! deployed kernel, export a group wire certificate, or measure runtime/RSS.
//! Explicit execution requires NY_GROUPED_ORIGINAL_PROBE_REQUEST; missing or
//! changed pinned inputs are errors, never an omitted/shortlisted vocabulary.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority, MemoryReceipt};
use crate::certify::work::{WorkLimits, WorkReceipt, WorkReserve};
use anyhow::{anyhow, ensure, Context};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::Sha256;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

type AnyResult<T> = anyhow::Result<T>;
const VOCAB: usize = 50_304;
const HIDDEN: usize = 128;
const WIDTH: usize = 64;
const BLOCK: usize = 8192;
const JSON_CAP: usize = 64 << 10;
const OUTPUT_CAP: usize = 256 << 10;
const DRIVER_PERSISTENT: usize = 4 << 20;
const DRIVER_WORK: u64 = 256 << 20;
const MODEL_BYTES: u64 = 28_143_920;
const MODEL_SHA: &str = "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1";
const CONFIG_SHA: &str = "f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17";
const MANIFEST_SHA: &str = "ae8412227ee6d51c985a6dcd0c8cd46a6e25e28cfd83611adddef8c0b12ccc93";
const WEIGHTS_SHA: &str = "28bfb79d00cd0dd29918e95795f89923c938121425c169bd38fde883b8359be5";
const BIAS_SHA: &str = "ca4a66e71de062d98549a7eb755b9e2b327dc1051bd53eec1d19c11228ab6312";
const GAMMA_SHA: &str = "a7500251f611a27a041fe15c1fe5873ab7ffeba9f6c1894ab8b3130692763cd8";
const BETA_SHA: &str = "b6eccf224d2e4bf16c4b7724452fabae188eb679f4e33ff55a252cf40269ce16";
const CASES: [(usize, &str); 3] = [
    (
        253,
        "3a075097dc5e7a705748b081d9d7eb866558f2b7c4eeef827a6df91ef77a674a",
    ),
    (
        187,
        "168bb091349eea05a8cd7704842956e4b0dae6fcf34f8ef8b5f7a693acbd5570",
    ),
    (
        13,
        "2ad02508fb530c1b0d00592a69ce17fa71f78f25b483a11dabb97ad95cc0220f",
    ),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
enum Schema {
    #[serde(rename = "ny.original_grouped_normalized_probe.v1")]
    V1,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeRequest {
    schema: Schema,
    original_model: Asset,
    original_config: Asset,
    fixture_manifest: Asset,
    weights_f64le: Asset,
    bias_f64le: Asset,
    gamma_f64le: Asset,
    beta_f64le: Asset,
    cases: [Asset; 3],
    output_directory: PathBuf,
}
#[derive(Deserialize)]
enum ExactReal {
    ExactReal,
}
#[derive(Deserialize)]
enum PreNormalization {
    #[serde(rename = "pre_normalization_box")]
    Box,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalNorm {
    kind: FinalNormalizationKind,
    arithmetic: ExactReal,
    input_domain: PreNormalization,
    gamma_f64le: PathBuf,
    beta_f64le: PathBuf,
    epsilon: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalCase {
    vocab_size: usize,
    hidden_size: usize,
    weights_f64le: PathBuf,
    bias_f64le: PathBuf,
    arithmetic: ExactReal,
    target: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    anchor: Vec<f64>,
    final_normalization: OriginalNorm,
}

/// Credits belong to this existing lease. No helper constructs a work budget,
/// refreshes a deadline or returns a proportional unleased decoded buffer.
struct IoPhase {
    lease: WorkLease,
    remaining: u64,
}
impl IoPhase {
    fn admit(budget: &WorkBudget, request: WorkRequest) -> AnyResult<Self> {
        let mut lease = budget.lease_for(LeaseClass::Replay, request)?;
        lease.mark_materialized()?;
        Ok(Self {
            lease,
            remaining: request.work,
        })
    }
    fn check(&self) -> AnyResult<()> {
        self.lease.checkpoint()?;
        ensure!(
            ny_core::instance_budget::deadline().is_none_or(|d| Instant::now() < d),
            "original instance deadline"
        );
        Ok(())
    }
    fn take(&mut self, n: u64) -> AnyResult<()> {
        self.remaining = self
            .remaining
            .checked_sub(n)
            .context("prepaid IO credits exhausted")?;
        self.check()
    }
    fn small(&mut self, path: &Path, pin: Option<&str>) -> AnyResult<(Vec<u8>, String)> {
        self.check()?;
        valid_path(path)?;
        let mut file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())?;
        ensure!(len <= JSON_CAP, "bounded metadata file exceeds 64 KiB");
        // Includes decoding/hash visits and bounded diagnostic metadata copies.
        self.take(
            (len as u64)
                .checked_mul(32)
                .context("metadata work overflow")?
                + 1024,
        )?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len)?;
        ensure!(bytes.capacity() <= JSON_CAP + 4096, "metadata capacity");
        let mut block = [0u8; BLOCK];
        while bytes.len() < len {
            self.check()?;
            let n = (len - bytes.len()).min(BLOCK);
            file.read_exact(&mut block[..n])?;
            bytes.extend_from_slice(&block[..n]);
        }
        ensure!(
            file.read(&mut block[..1])? == 0,
            "metadata grew during read"
        );
        let sha = self.hash(&bytes)?;
        if let Some(pin) = pin {
            ensure!(
                sha == pin,
                "pinned metadata SHA changed: {}",
                path.display()
            );
        }
        Ok((bytes, sha))
    }
    fn json<T: DeserializeOwned>(&mut self, asset: &Asset, pin: &str) -> AnyResult<T> {
        ensure!(
            asset.sha256 == pin,
            "request substituted pinned metadata identity"
        );
        let (bytes, _) = self.small(&asset.path, Some(pin))?;
        self.check()?;
        let value = serde_json::from_slice(&bytes)?;
        self.check()?;
        Ok(value)
    }
    fn hash(&mut self, bytes: &[u8]) -> AnyResult<String> {
        let mut hash = Sha256::new();
        for chunk in bytes.chunks(BLOCK) {
            self.take(chunk.len() as u64)?;
            hash.update(chunk);
        }
        self.check()?;
        Ok(format!("{:x}", hash.finalize()))
    }
    fn model_hash(&mut self, file: &mut File) -> AnyResult<String> {
        ensure!(
            file.metadata()?.len() == MODEL_BYTES,
            "original model length changed"
        );
        file.seek(SeekFrom::Start(0))?;
        let mut remaining = MODEL_BYTES;
        let mut block = [0u8; BLOCK];
        let mut hash = Sha256::new();
        while remaining != 0 {
            let n = usize::try_from(remaining.min(BLOCK as u64))?;
            self.take((n as u64) * 2)?;
            file.read_exact(&mut block[..n])?;
            hash.update(&block[..n]);
            remaining -= n as u64;
        }
        self.take(1)?;
        ensure!(file.read(&mut block[..1])? == 0, "original model grew");
        self.check()?;
        Ok(format!("{:x}", hash.finalize()))
    }
    fn write_report(&mut self, directory: &Path, report: &Value) -> AnyResult<String> {
        self.take(OUTPUT_CAP as u64 * 8)?;
        self.check()?;
        let bytes = serde_json::to_vec_pretty(report)?;
        ensure!(
            bytes.len() <= OUTPUT_CAP && bytes.capacity() <= 2 * OUTPUT_CAP,
            "bounded output allocation"
        );
        let sha = self.hash(&bytes)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("report.json"))?;
        for chunk in bytes.chunks(BLOCK) {
            self.take(chunk.len() as u64)?;
            output.write_all(chunk)?;
        }
        self.check()?;
        output.flush()?;
        self.check()?;
        Ok(sha)
    }
}
fn valid_path(path: &Path) -> AnyResult<()> {
    ensure!(
        path.is_absolute() && path.as_os_str().len() <= 4096,
        "bounded absolute asset path required"
    );
    Ok(())
}
fn load_request(budget: &WorkBudget) -> AnyResult<WorkRequest> {
    let count = VOCAB
        .checked_mul(HIDDEN)
        .and_then(|n| n.checked_add(VOCAB))
        .and_then(|n| n.checked_add(2 * HIDDEN))
        .context("head shape overflow")?;
    let bytes = count.checked_mul(8).context("head size overflow")?;
    let request = WorkRequest {
        scratch_bytes: BLOCK,
        persistent_bytes: bytes
            .checked_add(16384)
            .context("loader capacity overflow")?,
        work: (bytes as u64)
            .checked_mul(3)
            .and_then(|n| n.checked_add(4096))
            .context("loader work overflow")?,
        proposals: 1,
    };
    budget.checkpoint_for(LeaseClass::Replay)?;
    Ok(request)
}
fn load_f64(asset: &Asset, pin: &str, count: usize, phase: &mut IoPhase) -> AnyResult<Vec<f64>> {
    ensure!(
        asset.sha256 == pin,
        "original parameter identity substituted"
    );
    valid_path(&asset.path)?;
    phase.check()?;
    let expected = count
        .checked_mul(8)
        .context("original parameter length overflow")?;
    let mut file = File::open(&asset.path)?;
    ensure!(
        file.metadata()?.len() == expected as u64,
        "original coefficient length"
    );
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    ensure!(
        values.capacity() <= count + 512,
        "original coefficient capacity"
    );
    let mut block = [0u8; BLOCK];
    let mut hash = Sha256::new();
    let mut remaining = expected;
    while remaining != 0 {
        let n = remaining.min(BLOCK);
        phase.take((n as u64) * 2 + (n / 8) as u64)?;
        file.read_exact(&mut block[..n])?;
        hash.update(&block[..n]);
        for bytes in block[..n].as_chunks::<8>().0 {
            let value = f64::from_bits(u64::from_le_bytes(*bytes));
            ensure!(value.is_finite(), "original parameter nonfinite");
            values.push(value);
        }
        remaining -= n;
    }
    phase.take(1)?;
    ensure!(file.read(&mut block[..1])? == 0, "coefficient file grew");
    ensure!(
        format!("{:x}", hash.finalize()) == pin,
        "original parameter SHA changed"
    );
    phase.check()?;
    Ok(values)
}

/// Binary16 data decoded through integer bits into its exact binary64 value;
/// no host cast or neural-network execution arithmetic grants authority here.
fn original_half(bits: u16) -> AnyResult<f64> {
    let sign = u64::from(bits & 0x8000) << 48;
    let exponent = u64::from((bits >> 10) & 31);
    let fraction = u64::from(bits & 1023);
    ensure!(exponent != 31, "nonfinite original binary16 coefficient");
    let value = if exponent != 0 {
        sign | ((exponent + 1008) << 52) | (fraction << 42)
    } else if fraction == 0 {
        sign
    } else {
        let leading = fraction.ilog2();
        sign | (u64::from(leading + 999) << 52) | ((fraction - (1u64 << leading)) << (52 - leading))
    };
    Ok(f64::from_bits(value))
}
fn original_tensor(
    file: &mut File,
    metadata: &Value,
    data_start: u64,
    name: &str,
    shape: &[usize],
    expected: &[f64],
    phase: &mut IoPhase,
) -> AnyResult<()> {
    phase.take(1024)?;
    let tensor = metadata.get(name).context("missing original tensor")?;
    ensure!(tensor["dtype"] == "F16", "authored original tensor type");
    ensure!(
        tensor["shape"] == serde_json::to_value(shape)?,
        "authored original tensor shape"
    );
    let offsets = tensor["data_offsets"]
        .as_array()
        .context("original tensor offsets")?;
    ensure!(offsets.len() == 2, "original tensor offset arity");
    let start = offsets[0].as_u64().context("original tensor start")?;
    let end = offsets[1].as_u64().context("original tensor end")?;
    ensure!(
        end.checked_sub(start) == Some((expected.len() as u64) * 2),
        "original tensor span"
    );
    ensure!(
        data_start
            .checked_add(end)
            .is_some_and(|n| n <= MODEL_BYTES),
        "original tensor outside file"
    );
    file.seek(SeekFrom::Start(
        data_start
            .checked_add(start)
            .context("tensor address overflow")?,
    ))?;
    let mut block = [0u8; BLOCK];
    let mut index = 0;
    while index < expected.len() {
        let count = (expected.len() - index).min(BLOCK / 2);
        phase.take((count as u64) * 4)?;
        file.read_exact(&mut block[..count * 2])?;
        for (at, value) in block[..count * 2].as_chunks::<2>().0.iter().enumerate() {
            let decoded = original_half(u16::from_le_bytes(*value))?;
            ensure!(
                decoded.to_bits() == expected[index + at].to_bits(),
                "original tensor bit mismatch: {name}"
            );
        }
        index += count;
    }
    phase.check()
}
fn receipt(r: WorkReceipt) -> Value {
    json!({"attempted":r.attempted,"admitted":r.admitted,"refused":r.refused,"completed":r.completed,
        "work_reserved":r.work_reserved,"proposals_reserved":r.proposals_reserved,
        "live_scratch_bytes":r.live_scratch_bytes,"live_persistent_bytes":r.live_persistent_bytes,
        "peak_leased_capacity_bytes":r.peak_leased_bytes})
}
fn memory_receipt(r: MemoryReceipt) -> Value {
    json!({"attempted":r.attempted,"admitted":r.admitted,"refused":r.refused,
        "live_pending_bytes":r.live_pending_bytes,"live_materialized_bytes":r.live_materialized_bytes,
        "peak_leased_capacity_bytes":r.peak_leased_bytes})
}
fn finish_visits(s: &NormalizedHeadStats) -> u64 {
    s.original_head_visits
        + s.normalization_parameter_visits
        + s.region_visits
        + s.kernel_coordinate_visits
        + s.coverage_visits
}
fn group_visits(s: &NormalizedGroupedHeadStats) -> u64 {
    finish_visits(&s.original)
        + s.proposal_coverage_visits
        + s.completed_group_coordinate_visits
        + s.member_rescue_coordinate_visits
}
fn direct_proof(report: &NormalizedHeadReport) -> AnyResult<&CertifiedNormalizedNextToken> {
    match &report.outcome {
        NormalizedNextTokenOutcome::Certified(proof) => Ok(proof),
        NormalizedNextTokenOutcome::Unknown => {
            Err(anyhow!("pinned original direct control did not close"))
        }
    }
}
fn grouped_proof(
    report: &NormalizedGroupedHeadReport,
) -> AnyResult<&CertifiedGroupedNormalizedNextToken> {
    match &report.outcome {
        NormalizedGroupedNextTokenOutcome::Certified(proof) => Ok(proof),
        NormalizedGroupedNextTokenOutcome::Unknown => {
            Err(anyhow!("grouped original control did not close"))
        }
    }
}
fn direct_hash(report: &NormalizedHeadReport, phase: &mut IoPhase) -> AnyResult<String> {
    let proof = direct_proof(report)?;
    ensure!(
        proof.target() == report.evidence.target
            && report.evidence.vocab_size == VOCAB
            && report.evidence.hidden_size == HIDDEN,
        "direct sealed original header"
    );
    ensure!(
        proof.covered_competitors() == VOCAB - 1
            && proof.maximum_competitor_upper().is_some_and(|u| u < 0.0),
        "direct strict full cover"
    );
    ensure!(
        report.evidence.rows.len() == VOCAB - 1,
        "direct complete evidence"
    );
    let mut hash = Sha256::new();
    hash.update(b"ny.original-group-probe.direct-evidence.v1\0");
    hash.update(report.evidence.scope_digest);
    let mut next = 0;
    for row in &report.evidence.rows {
        phase.take(16)?;
        if next == report.evidence.target {
            next += 1;
        }
        ensure!(
            row.competitor == next && row.upper.is_finite() && row.upper < 0.0,
            "direct original ID/upper"
        );
        hash.update((next as u64).to_le_bytes());
        hash.update(row.upper.to_bits().to_le_bytes());
        hash.update([match row.method {
            NormalizedSupportMethod::ProjectedDense => 0,
            NormalizedSupportMethod::RadialTaylor => 1,
        }]);
        next += 1;
    }
    if next == report.evidence.target {
        next += 1;
    }
    ensure!(
        next == VOCAB && proof.checked_stats() == &report.stats,
        "direct checked receipt"
    );
    Ok(format!("{:x}", hash.finalize()))
}
fn grouped_hash(report: &NormalizedGroupedHeadReport, phase: &mut IoPhase) -> AnyResult<String> {
    let proof = grouped_proof(report)?;
    ensure!(
        proof.target() == report.evidence.target
            && report.evidence.vocab_size == VOCAB
            && report.evidence.hidden_size == HIDDEN
            && report.evidence.schema_version == GROUP_SCHEMA
            && report.evidence.group_width == WIDTH,
        "group sealed original header"
    );
    ensure!(
        proof.covered_competitors() == VOCAB - 1
            && proof.maximum_competitor_upper().is_some_and(|u| u < 0.0),
        "group strict full cover"
    );
    ensure!(
        proof.checked_stats() == &report.stats,
        "group checked receipt"
    );
    let mut hash = Sha256::new();
    hash.update(b"ny.original-group-probe.group-evidence.v1\0");
    hash.update(report.evidence.scope_digest);
    let mut next = 0;
    for entry in &report.evidence.entries {
        phase.take(32)?;
        match entry {
            NormalizedTerminalEvidence::OriginalGroup { start, end, upper } => {
                ensure!(
                    upper.is_finite() && *upper < 0.0,
                    "group strictly finite upper"
                );
                hash.update([0]);
                hash.update((*start as u64).to_le_bytes());
                hash.update((*end as u64).to_le_bytes());
                hash.update(upper.to_bits().to_le_bytes());
                for competitor in *start..*end {
                    phase.take(4)?;
                    if competitor == proof.target() {
                        continue;
                    }
                    if next == proof.target() {
                        next += 1;
                    }
                    ensure!(
                        competitor == next,
                        "group complete ascending original membership"
                    );
                    next += 1;
                }
            }
            NormalizedTerminalEvidence::OriginalRow {
                competitor,
                upper,
                method,
            } => {
                ensure!(
                    upper.is_finite() && *upper < 0.0,
                    "row strictly finite upper"
                );
                if next == proof.target() {
                    next += 1;
                }
                ensure!(*competitor == next, "group fallback exact original ID");
                hash.update([1]);
                hash.update((*competitor as u64).to_le_bytes());
                hash.update(upper.to_bits().to_le_bytes());
                hash.update([match method {
                    NormalizedSupportMethod::ProjectedDense => 0,
                    NormalizedSupportMethod::RadialTaylor => 1,
                }]);
                next += 1;
            }
        }
    }
    if next == proof.target() {
        next += 1;
    }
    ensure!(next == VOCAB, "group omitted competitor");
    Ok(format!("{:x}", hash.finalize()))
}

fn run_original() -> AnyResult<()> {
    let mut limits = VocabOptions::default();
    let scratch = usize::try_from(limits.max_scratch_bytes)?;
    let persistent = usize::try_from(limits.max_persistent_bytes)?;
    let memory = MemoryAuthority::new(
        scratch
            .checked_add(persistent)
            .context("default capacity overflow")?,
    );
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: limits.deadline,
            scratch_bytes: scratch,
            persistent_bytes: persistent,
            max_work: limits.max_terms,
            max_proposals: limits.max_proposals,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )?;
    limits.budget = Some(budget.clone());
    let loader_request = load_request(&budget)?;
    let baseline = normalized_work_bound(VOCAB, HIDDEN)?;
    let grouped_worst = normalized_group_work_bound(VOCAB, HIDDEN, WIDTH)?;
    // Two original direct phases and two grouped phases per case, with the
    // grouped prediction covering BOTH old/member phases plus direct fallback.
    let per_case = baseline
        .checked_mul(2)
        .and_then(|n| grouped_worst.checked_mul(2)?.checked_add(n))
        .context("case work overflow")?;
    let owner_work = (VOCAB * (HIDDEN + 1)) as u64;
    let worst = per_case
        .checked_mul(3)
        .and_then(|n| n.checked_add(DRIVER_WORK))
        .and_then(|n| n.checked_add(loader_request.work))
        .and_then(|n| n.checked_add(owner_work))
        .context("complete probe work overflow")?;
    ensure!(
        worst <= limits.max_terms,
        "all original query/replay/IO predictions must fit unchanged default work"
    );
    {
        // Everything proportional below, including source/config/header/box
        // metadata and bounded output copies, remains inside this driver lease.
        let mut driver = IoPhase::admit(
            &budget,
            WorkRequest {
                scratch_bytes: BLOCK,
                persistent_bytes: DRIVER_PERSISTENT,
                work: DRIVER_WORK,
                proposals: 1,
            },
        )?;
        let inputs = ny_levers::RawLeverInputs::capture(ny_levers::all());
        let path = inputs
            .get(&ny_levers::decls::diagnostics::GROUPED_ORIGINAL_PROBE_REQUEST)
            .context(
                "NY_GROUPED_ORIGINAL_PROBE_REQUEST is required; explicit missing inputs never skip",
            )?;
        ensure!(path.len() <= 4096, "request path length");
        let (request_bytes, request_sha) = driver.small(Path::new(&path), None)?;
        let request: ProbeRequest = serde_json::from_slice(&request_bytes)?;
        driver.check()?;
        match request.schema {
            Schema::V1 => {}
        }
        for (asset, pin) in [
            (&request.original_model, MODEL_SHA),
            (&request.original_config, CONFIG_SHA),
            (&request.fixture_manifest, MANIFEST_SHA),
            (&request.weights_f64le, WEIGHTS_SHA),
            (&request.bias_f64le, BIAS_SHA),
            (&request.gamma_f64le, GAMMA_SHA),
            (&request.beta_f64le, BETA_SHA),
        ] {
            ensure!(asset.sha256 == pin, "closed request original identity");
            valid_path(&asset.path)?;
        }
        valid_path(&request.output_directory)?;
        let manifest: Value = driver.json(&request.fixture_manifest, MANIFEST_SHA)?;
        ensure!(
            manifest["repository"] == "EleutherAI/pythia-14m"
                && manifest["revision"] == "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
                && manifest["full_vocabulary"] == VOCAB
                && manifest["hidden_size"] == HIDDEN
                && manifest["certificate_authority"] == false
                && manifest["performance_claim"] == false,
            "known original fixture corpus"
        );
        let config: Value = driver.json(&request.original_config, CONFIG_SHA)?;
        ensure!(
            config["hidden_size"] == HIDDEN
                && config["num_hidden_layers"] == 6
                && config["vocab_size"] == VOCAB
                && config["layer_norm_eps"].as_f64() == Some(1e-5),
            "authored original config/epsilon"
        );
        let mut cases = Vec::with_capacity(3);
        for (index, asset) in request.cases.iter().enumerate() {
            let case: OriginalCase = driver.json(asset, CASES[index].1)?;
            ensure!(
                case.vocab_size == VOCAB
                    && case.hidden_size == HIDDEN
                    && case.target == CASES[index].0,
                "pinned original case identity"
            );
            ensure!(
                case.weights_f64le == request.weights_f64le.path
                    && case.bias_f64le == request.bias_f64le.path
                    && case.final_normalization.gamma_f64le == request.gamma_f64le.path
                    && case.final_normalization.beta_f64le == request.beta_f64le.path,
                "original case parameter paths"
            );
            ensure!(
                case.final_normalization.kind == FinalNormalizationKind::LayerNorm
                    && case.final_normalization.epsilon.to_bits() == 1e-5f64.to_bits(),
                "authored original normalization"
            );
            match (
                &case.arithmetic,
                &case.final_normalization.arithmetic,
                &case.final_normalization.input_domain,
            ) {
                (ExactReal::ExactReal, ExactReal::ExactReal, PreNormalization::Box) => {}
            }
            ensure!(
                case.lo.len() == HIDDEN && case.hi.len() == HIDDEN && case.anchor.len() == HIDDEN,
                "original box shape"
            );
            for ((lo, hi), anchor) in case.lo.iter().zip(&case.hi).zip(&case.anchor) {
                driver.take(16)?;
                ensure!(
                    lo.is_finite()
                        && hi.is_finite()
                        && anchor.is_finite()
                        && lo < hi
                        && lo <= anchor
                        && anchor <= hi,
                    "pinned nonzero finite box and legacy anchor metadata"
                );
            }
            cases.push(case);
        }
        driver.check()?;
        let mut original_file = File::open(&request.original_model.path)?;
        ensure!(
            driver.model_hash(&mut original_file)? == MODEL_SHA,
            "full original model SHA"
        );
        original_file.seek(SeekFrom::Start(0))?;
        driver.take(8)?;
        let mut header_length = [0u8; 8];
        original_file.read_exact(&mut header_length)?;
        let header_len = usize::try_from(u64::from_le_bytes(header_length))?;
        ensure!(
            header_len <= JSON_CAP && header_len > 0,
            "bounded SafeTensors metadata"
        );
        driver.take((header_len as u64) * 32)?;
        let mut header = Vec::new();
        header.try_reserve_exact(header_len)?;
        ensure!(header.capacity() <= JSON_CAP + 4096, "header capacity");
        for index in 0..header_len {
            if index % 1024 == 0 {
                driver.check()?;
            }
            header.push(0);
        }
        for chunk in header.chunks_mut(BLOCK) {
            driver.check()?;
            original_file.read_exact(chunk)?;
        }
        let tensors: Value = serde_json::from_slice(&header)?;
        let data_start = (header_len as u64)
            .checked_add(8)
            .context("original data offset overflow")?;
        let mut loading = IoPhase::admit(&budget, loader_request)?;
        let weights = load_f64(
            &request.weights_f64le,
            WEIGHTS_SHA,
            VOCAB * HIDDEN,
            &mut loading,
        )?;
        let bias = load_f64(&request.bias_f64le, BIAS_SHA, VOCAB, &mut loading)?;
        let gamma = load_f64(&request.gamma_f64le, GAMMA_SHA, HIDDEN, &mut loading)?;
        let beta = load_f64(&request.beta_f64le, BETA_SHA, HIDDEN, &mut loading)?;
        original_tensor(
            &mut original_file,
            &tensors,
            data_start,
            "embed_out.weight",
            &[VOCAB, HIDDEN],
            &weights,
            &mut driver,
        )?;
        original_tensor(
            &mut original_file,
            &tensors,
            data_start,
            "gpt_neox.final_layer_norm.weight",
            &[HIDDEN],
            &gamma,
            &mut driver,
        )?;
        original_tensor(
            &mut original_file,
            &tensors,
            data_start,
            "gpt_neox.final_layer_norm.bias",
            &[HIDDEN],
            &beta,
            &mut driver,
        )?;
        for block in bias.chunks(256) {
            driver.take(block.len() as u64)?;
            ensure!(
                block.iter().all(|v| v.to_bits() == 0),
                "original bias-free lm-head must retain exact zero bias"
            );
        }
        // Rehash the very same open original source after every original-data
        // tensor comparison; changed source cannot be labeled the pinned model.
        ensure!(
            driver.model_hash(&mut original_file)? == MODEL_SHA,
            "original source changed during bit binding"
        );
        loading.check()?;
        let owner = SharedAffineHead::new(
            VOCAB,
            HIDDEN,
            weights,
            bias,
            HeadArithmetic::ExactReal,
            &limits,
        )?;
        ensure!(
            owner.budget().shares_ledger(&budget),
            "original owner ledger"
        );
        let transfer_overlap = memory_receipt(memory.receipt()?);
        loading.lease.complete()?;
        drop(loading); // Ownership transfer, no second coefficient allocation.
        let after_transfer = memory_receipt(memory.receipt()?);
        std::fs::create_dir(&request.output_directory)?;
        let mut records = Vec::with_capacity(3);
        let mut retained: Option<CertifiedGroupedNormalizedNextToken> = None;
        for (index, case) in cases.iter().enumerate() {
            driver.check()?;
            let norm = FinalNormalization {
                kind: FinalNormalizationKind::LayerNorm,
                gamma: &gamma,
                beta: Some(&beta),
                epsilon: case.final_normalization.epsilon,
            };
            let pre = HiddenBox {
                lo: &case.lo,
                hi: &case.hi,
            };
            let before = budget.receipt()?;
            let direct = shared::verify_shared_normalized_next_token(
                &owner,
                norm,
                pre,
                case.target,
                &limits,
            )?;
            let after_direct = budget.receipt()?;
            ensure!(
                after_direct.work_reserved - before.work_reserved == direct.stats.reserved_work,
                "direct original work receipt"
            );
            let direct_evidence_sha = direct_hash(&direct, &mut driver)?;
            let direct_replay = shared::replay_shared_normalized_next_token(
                &owner,
                norm,
                pre,
                case.target,
                &direct.evidence,
                &limits,
            )?;
            let after_direct_replay = budget.receipt()?;
            ensure!(
                after_direct_replay.work_reserved - after_direct.work_reserved
                    == direct_replay.stats.reserved_work,
                "direct fresh original replay work"
            );
            ensure!(
                direct_replay.provenance() == NextTokenProvenance::OriginalReplay
                    && direct_replay.evidence == direct.evidence,
                "direct original replay"
            );
            ensure!(
                direct_hash(&direct_replay, &mut driver)? == direct_evidence_sha,
                "direct fresh original evidence bits"
            );
            let grouped = verify_shared_grouped_normalized_next_token(
                &owner,
                norm,
                pre,
                case.target,
                NormalizedGroupPolicy { group_width: WIDTH },
                NormalizedHeadOptions {
                    budget: &budget,
                    limits: &limits,
                },
            )?;
            let after_group = budget.receipt()?;
            ensure!(
                after_group.work_reserved - after_direct_replay.work_reserved
                    == grouped.stats.reserved_work,
                "group admitted original work"
            );
            let group_evidence_sha = grouped_hash(&grouped, &mut driver)?;
            let group_replay = grouped.replay(norm, pre, case.target, &limits)?;
            let after_replay = budget.receipt()?;
            ensure!(
                after_replay.work_reserved - after_group.work_reserved
                    == group_replay.stats.reserved_work,
                "group fresh original replay work"
            );
            ensure!(
                group_replay.provenance() == NextTokenProvenance::OriginalReplay
                    && group_replay.evidence == grouped.evidence,
                "fresh complete original group replay"
            );
            ensure!(
                grouped_hash(&group_replay, &mut driver)? == group_evidence_sha,
                "fresh original group evidence bits"
            );
            let dp = direct_proof(&direct)?;
            let gp = grouped_proof(&grouped)?;
            let rp = grouped_proof(&group_replay)?;
            ensure!(
                dp.scope() == gp.scope()
                    && gp.scope() == rp.scope()
                    && direct.evidence.scope_digest == grouped.evidence.scope_digest,
                "exact original query/normalization/box/property scope"
            );
            for stats in [
                &direct.stats,
                &direct_replay.stats,
                &grouped.stats.original,
                &group_replay.stats.original,
            ] {
                ensure!(stats.original_head_visits == (VOCAB * (HIDDEN + 1)) as u64 && stats.coverage_visits == (VOCAB - 1) as u64,
                    "full original coefficients, biases and competitor cover must actually be visited");
            }
            let observed_query = group_visits(&grouped.stats);
            let observed_replay = group_visits(&group_replay.stats);
            let direct_query = finish_visits(&direct.stats);
            let direct_replayed = finish_visits(&direct_replay.stats);
            let comparable = grouped.stats.unobserved_failed_group_attempts == 0
                && group_replay.stats.unobserved_failed_group_attempts == 0;
            driver.take(16 << 10)?;
            records.push(json!({
                "case":index,"target":case.target,"fixture_sha256":CASES[index].1,
                "conditional_domain":"original raw block5 last-position box, radius1e-4; explicit ExactReal premise",
                "legacy_anchor_used_as_proof_geometry":false,"original_scope":gp.scope(),"scope_digest":grouped.evidence.scope_digest,
                "covered_competitors":gp.covered_competitors(),"full_vocabulary":VOCAB,
                "direct_closed":true,"grouped_closed":true,"independent_original_replay_closed":true,
                "direct_maximum_upper":dp.maximum_competitor_upper(),"grouped_maximum_upper":gp.maximum_competitor_upper(),
                "direct_evidence_sha256":direct_evidence_sha,"group_evidence_sha256":group_evidence_sha,
                "direct_stats":direct.stats,"direct_replay_stats":direct_replay.stats,
                "group_stats":grouped.stats,"group_replay_stats":group_replay.stats,
                "reported_completed_visit_comparison_valid":comparable,
                "direct_query_completed_visits":direct_query,"group_query_completed_visits":observed_query,
                "direct_replay_completed_visits":direct_replayed,"group_replay_completed_visits":observed_replay,
                "group_query_fewer_completed_visits":comparable && observed_query < direct_query,
                "group_replay_fewer_completed_visits":comparable && observed_replay < direct_replayed,
                "reserved_work_comparison_is_not_observed_runtime":true,
                "same_ledger_before":receipt(before),"same_ledger_after":receipt(after_replay),
            }));
            retained = Some(rp.clone());
            drop(group_replay);
            drop(grouped);
            drop(direct_replay);
            drop(direct);
        }
        let proof = retained.take().context("last original proof")?;
        let retained_bytes = proof
            .reserved_bytes()
            .checked_add(proof.shared_head_bytes())
            .context("retention overflow")?;
        drop(owner);
        let retained_receipt = budget.receipt()?;
        ensure!(
            retained_receipt.live_persistent_bytes == DRIVER_PERSISTENT + retained_bytes
                && retained_receipt.live_scratch_bytes == BLOCK,
            "last original proof must retain exactly its checked capacity plus driver"
        );
        let last = proof.clone();
        drop(proof);
        ensure!(
            budget.receipt()?.live_persistent_bytes == DRIVER_PERSISTENT + retained_bytes,
            "intermediate proof clone cannot release owner"
        );
        drop(last);
        let after_last_clone = budget.receipt()?;
        ensure!(
            after_last_clone.live_persistent_bytes == DRIVER_PERSISTENT
                && after_last_clone.live_scratch_bytes == BLOCK,
            "last proof must release original and query claims without refunding work"
        );
        ensure!(
            after_last_clone.work_reserved == retained_receipt.work_reserved,
            "last clone must not refill spent work"
        );
        driver.take(32 << 10)?;
        let report = json!({
            "schema":"ny.original_grouped_normalized_probe.report.v1",
            "driver_source_sha256":driver.hash(include_bytes!("original_probe.rs"))?,
            "request_sha256":request_sha,
            "repository":"EleutherAI/pythia-14m","revision":"cf967c0a9a04383db6f7b1108d86b2962634b4ac",
            "original_model_sha256":MODEL_SHA,"original_config_sha256":CONFIG_SHA,"fixture_manifest_sha256":MANIFEST_SHA,
            "weights_f64le_sha256":WEIGHTS_SHA,"bias_f64le_sha256":BIAS_SHA,"gamma_f64le_sha256":GAMMA_SHA,"beta_f64le_sha256":BETA_SHA,
            "original_model_head_gamma_beta_bits_checked":true,"original_tensor_dtype":"F16 decoded exactly by bits into binary64",
            "original_head_owners_created":1,"full_vocabulary":VOCAB,"hidden_size":HIDDEN,
            "private_group_width":WIDTH,"default_route_claim":false,"bounded_wire_claim":false,
            "arithmetic":"ExactReal original head and LayerNorm","upstream_trajectory_certificate":false,"execution_kernel_certificate":false,
            "point_decoder_executions":0,"runtime_claim":false,"rss_claim":false,"competition_score_claim":false,
            "default_caps":{"scratch_bytes":limits.max_scratch_bytes,"persistent_bytes":limits.max_persistent_bytes,"work":limits.max_terms,
                "proposals":limits.max_proposals,"deadline":"unchanged default None; original inherited instance deadline always checked"},
            "work_prediction":{"direct_phase":baseline,"group_phase_worst":grouped_worst,"per_case_two_direct_two_group_phases":per_case,
                "case_count":3,"driver":DRIVER_WORK,"loader":loader_request.work,"single_head_owner":owner_work,
                "complete_original_probe_worst":worst,"no_refill_or_nested_budget":true},
            "head_transfer_overlap":transfer_overlap,"head_after_transfer":after_transfer,
            "cases":records,"retained_last_clone_capacity":retained_bytes,"before_last_clone_drop":receipt(retained_receipt),
            "after_last_clone_drop":receipt(after_last_clone),"memory_after_last_clone":memory_receipt(memory.receipt()?),
            "remaining_driver_claim_bytes":DRIVER_PERSISTENT+BLOCK,
            "post_driver_zero_capacity_gate":"external conformance asserts zero original-ledger and shared-memory live capacities after all driver buffers drop",
            "driver_work_consumed_before_output":DRIVER_WORK-driver.remaining,
        });
        driver.write_report(&request.output_directory, &report)?;
        driver.check()?;
        driver.lease.complete()?;
        // All buffer owners above drop within this block before the final
        // zero-live-capacity check. There is no raw result or owner escape.
    }
    let released = budget.receipt()?;
    ensure!(
        released.live_scratch_bytes == 0 && released.live_persistent_bytes == 0,
        "all original and driver capacity must release"
    );
    let released_memory = memory.receipt()?;
    ensure!(
        released_memory.live_pending_bytes == 0 && released_memory.live_materialized_bytes == 0,
        "shared byte authority cleanup"
    );
    ensure!(
        released.work_reserved <= limits.max_terms && released.work_reserved >= owner_work,
        "finite cumulative original work"
    );
    Ok(())
}

#[test]
fn decoder_n12_groups_original_pythia_probe() {
    run_original()
        .expect("original grouped query/replay probe must execute its full closed request");
}
