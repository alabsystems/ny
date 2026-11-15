// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original resident-head integration diagnostic. The supplied boxes are
//! conditional premises, not decoder-trajectory or deployed-kernel proofs.
//! Participating allocation capacities are leased; process RSS is not measured.
//! Run the separately pinned external runner after building this example with
//! `targo --unverified build -p ny-cli --example resident_head_probe`.

use anyhow::{bail, ensure, Context, Result};
use ny_api::decoder::{
    CancellationToken, CertifiedNormalizedNextToken, DecoderRefusal, FinalNormalization,
    FinalNormalizationKind, HeadArithmetic, HiddenBox, LeaseClass, MemoryAuthority, MemoryReceipt,
    NextTokenProvenance, NormalizedHeadRefusal, NormalizedHeadReport, NormalizedNextTokenOutcome,
    PersistentReservation, ResidentAffineHead, VocabOptions, WorkBudget, WorkLease, WorkLimits,
    WorkReceipt, WorkRefusal, WorkRequest, WorkReserve,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const GLOBAL_BYTES: usize = 128 << 20;
const REQUEST_BYTES: usize = 64 << 10;
const REQUEST_CAPACITY: usize = 1 << 20;
const OUTPUT_CAPACITY: usize = 256 << 10;
const QUERY_WORK: u64 = 1 << 30;
const CHUNK: usize = 8192;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
enum Schema {
    #[serde(rename = "ny.resident_head_probe.v1")]
    V1,
}
#[derive(Deserialize)]
enum Arithmetic {
    ExactReal,
}
#[derive(Deserialize)]
enum Domain {
    #[serde(rename = "pre_normalization_box")]
    PreNormalizationBox,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Norm {
    kind: FinalNormalizationKind,
    arithmetic: Arithmetic,
    input_domain: Domain,
    gamma_f64le: Asset,
    beta_f64le: Option<Asset>,
    epsilon: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    target: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: Schema,
    arithmetic: Arithmetic,
    vocab_size: usize,
    hidden_size: usize,
    weights_f64le: Asset,
    bias_f64le: Asset,
    final_normalization: Norm,
    timeout_secs: u64,
    cases: [Case; 3],
}

fn ledger(
    memory: &MemoryAuthority,
    cancel: &CancellationToken,
    seconds: u64,
    work: u64,
) -> Result<WorkBudget> {
    WorkBudget::new_in(
        memory,
        cancel,
        WorkLimits {
            deadline: Some(
                Instant::now()
                    .checked_add(Duration::from_secs(seconds))
                    .context("deadline overflow")?,
            ),
            scratch_bytes: 16 << 20,
            persistent_bytes: GLOBAL_BYTES,
            max_work: work,
            max_proposals: 64,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .map_err(Into::into)
}
fn options(budget: &WorkBudget) -> VocabOptions {
    VocabOptions {
        budget: Some(budget.clone()),
        max_vocab: 50304,
        max_hidden: 128,
        max_persistent_bytes: GLOBAL_BYTES as u64,
        max_scratch_bytes: 16 << 20,
        max_terms: QUERY_WORK,
        max_nodes: 50304,
        max_proposals: 64,
        max_evidence_bytes: 8 << 20,
        ..VocabOptions::default()
    }
}
fn budget_record(r: WorkReceipt) -> Value {
    json!({"attempted":r.attempted,"admitted":r.admitted,"refused":r.refused,
        "completed":r.completed,"reserved_work":r.work_reserved,"reserved_proposals":r.proposals_reserved,
        "live_scratch_bytes":r.live_scratch_bytes,"live_persistent_bytes":r.live_persistent_bytes,
        "peak_participating_capacity_bytes":r.peak_leased_bytes})
}
fn memory_record(r: MemoryReceipt) -> Value {
    json!({"attempted":r.attempted,"admitted":r.admitted,"refused":r.refused,
        "live_pending_bytes":r.live_pending_bytes,"live_materialized_bytes":r.live_materialized_bytes,
        "peak_participating_capacity_bytes":r.peak_leased_bytes})
}
fn read_request(
    path: &Path,
    budget: &WorkBudget,
) -> Result<(Request, PersistentReservation, String)> {
    let mut file = File::open(path)?;
    let len = usize::try_from(file.metadata()?.len())?;
    ensure!(len <= REQUEST_BYTES, "request exceeds 64 KiB");
    let mut lease = budget.lease_for(
        LeaseClass::Replay,
        WorkRequest {
            persistent_bytes: REQUEST_CAPACITY,
            scratch_bytes: CHUNK,
            work: (REQUEST_BYTES as u64) * 32,
            proposals: 1,
        },
    )?;
    lease.checkpoint()?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len)?;
    ensure!(
        bytes.capacity() <= REQUEST_CAPACITY,
        "request buffer capacity"
    );
    let mut block = [0u8; CHUNK];
    while bytes.len() < len {
        lease.checkpoint()?;
        let count = (len - bytes.len()).min(CHUNK);
        let got = file.read(&mut block[..count])?;
        ensure!(got != 0, "request shrank");
        bytes.extend_from_slice(&block[..got]);
    }
    ensure!(file.read(&mut block[..1])? == 0, "request grew");
    lease.checkpoint()?;
    let request: Request = serde_json::from_slice(&bytes)?;
    // Raw JSON decoding is a diagnostic driver allocation, explicitly covered
    // by the fixed 1 MiB lease and bounded 64 KiB input, not a library wire claim.
    ensure!(
        request.vocab_size == 50304 && request.hidden_size == 128,
        "original Pythia-14m shape required"
    );
    ensure!(
        (1..=1800).contains(&request.timeout_secs),
        "timeout must be 1..1800 seconds"
    );
    ensure!(
        request.final_normalization.epsilon.is_finite()
            && request.final_normalization.epsilon >= 1e-12_f32 as f64,
        "unchanged authored epsilon must be in the checked domain"
    );
    for case in &request.cases {
        ensure!(
            case.target < request.vocab_size
                && case.lo.len() == request.hidden_size
                && case.hi.len() == request.hidden_size,
            "case shape"
        );
        ensure!(
            case.lo
                .iter()
                .zip(&case.hi)
                .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u),
            "finite ordered box required"
        );
    }
    match (
        &request.schema,
        &request.arithmetic,
        &request.final_normalization.arithmetic,
        &request.final_normalization.input_domain,
    ) {
        (Schema::V1, Arithmetic::ExactReal, Arithmetic::ExactReal, Domain::PreNormalizationBox) => {
        }
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    lease.checkpoint()?;
    lease.complete()?;
    Ok((request, lease.commit_persistent()?, digest))
}
fn load(asset: &Asset, count: usize, lease: &WorkLease) -> Result<Vec<f64>> {
    ensure!(
        asset.sha256.len() == 64
            && asset
                .sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "lowercase SHA-256 required"
    );
    let bytes = count.checked_mul(8).context("file size overflow")?;
    let mut file = File::open(&asset.path)?;
    ensure!(
        file.metadata()?.len() == bytes as u64,
        "original coefficient file length"
    );
    lease.checkpoint()?;
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    ensure!(
        result
            .capacity()
            .checked_mul(8)
            .is_some_and(|n| n <= bytes + 4096),
        "coefficient capacity"
    );
    let mut hash = Sha256::new();
    let mut block = [0u8; CHUNK];
    let mut remaining = bytes;
    while remaining > 0 {
        lease.checkpoint()?;
        let take = remaining.min(CHUNK);
        file.read_exact(&mut block[..take])?;
        hash.update(&block[..take]);
        for cell in block[..take].as_chunks::<8>().0 {
            let value = f64::from_le_bytes(*cell);
            ensure!(value.is_finite(), "nonfinite original coefficient");
            result.push(value);
        }
        remaining -= take;
    }
    ensure!(file.read(&mut block[..1])? == 0, "coefficient file grew");
    ensure!(
        format!("{:x}", hash.finalize()) == asset.sha256,
        "original asset SHA-256 mismatch"
    );
    lease.checkpoint()?;
    Ok(result)
}
fn file_request(count: usize) -> Result<WorkRequest> {
    let bytes = count.checked_mul(8).context("file capacity overflow")?;
    Ok(WorkRequest {
        persistent_bytes: bytes.checked_add(8192).context("file metadata overflow")?,
        scratch_bytes: CHUNK,
        work: (bytes as u64)
            .checked_mul(2)
            .and_then(|w| w.checked_add((count as u64) * 4))
            .context("file work overflow")?,
        proposals: 1,
    })
}
fn certified(
    report: &NormalizedHeadReport,
    target: usize,
    vocab: usize,
    lease: &WorkLease,
) -> Result<CertifiedNormalizedNextToken> {
    let NormalizedNextTokenOutcome::Certified(proof) = &report.outcome else {
        bail!("original narrow-box positive control stayed unknown")
    };
    ensure!(
        proof.target() == target
            && proof.covered_competitors() == vocab - 1
            && report.evidence.rows.len() == vocab - 1,
        "complete original cover"
    );
    ensure!(
        proof.maximum_competitor_upper().is_some_and(|v| v < 0.),
        "strict original decision"
    );
    let mut next = 0;
    for id in 0..vocab {
        if id % 256 == 0 {
            lease.checkpoint()?;
        }
        if id == target {
            continue;
        }
        let row = &report.evidence.rows[next];
        ensure!(
            row.competitor == id && row.upper.is_finite() && row.upper < 0.,
            "canonical strictly certified original ID"
        );
        next += 1;
    }
    ensure!(
        proof.checked_stats().coverage_visits == (vocab - 1) as u64,
        "checked coverage receipt"
    );
    ensure!(
        proof.checked_stats().original_head_visits == (vocab * 129) as u64,
        "every original coefficient and bias must be rehashed"
    );
    Ok(proof.clone())
}
fn sha_polled(bytes: &[u8], lease: &WorkLease) -> Result<String> {
    let mut hash = Sha256::new();
    for chunk in bytes.chunks(CHUNK) {
        lease.checkpoint()?;
        hash.update(chunk);
    }
    lease.checkpoint()?;
    Ok(format!("{:x}", hash.finalize()))
}

fn run(path: &Path) -> Result<()> {
    let memory = MemoryAuthority::new(GLOBAL_BYTES);
    let construction = ledger(&memory, &CancellationToken::default(), 1800, QUERY_WORK)?;
    let (request, request_storage, request_sha) = read_request(path, &construction)?;
    // This finite driver lease prices three records, canonical-ID checks,
    // wire-byte hashing and bounded result serialization. Library query work
    // is independently reserved on each query's original ledger.
    let mut driver = construction.lease_for(
        LeaseClass::Replay,
        WorkRequest {
            persistent_bytes: OUTPUT_CAPACITY,
            scratch_bytes: 16 << 10,
            work: 64 << 20,
            proposals: 1,
        },
    )?;
    let values = request
        .vocab_size
        .checked_mul(request.hidden_size)
        .and_then(|n| n.checked_add(request.vocab_size))
        .context("original head shape overflow")?;
    let head_load_request = file_request(values)?;
    let head_loading = construction.lease_for(LeaseClass::Replay, head_load_request)?;
    let weights = load(
        &request.weights_f64le,
        request.vocab_size * request.hidden_size,
        &head_loading,
    )?;
    let bias = load(&request.bias_f64le, request.vocab_size, &head_loading)?;
    let mut parameter_loading = construction.lease_for(
        LeaseClass::Replay,
        file_request(
            request.hidden_size
                * (1 + usize::from(request.final_normalization.beta_f64le.is_some())),
        )?,
    )?;
    let gamma = load(
        &request.final_normalization.gamma_f64le,
        request.hidden_size,
        &parameter_loading,
    )?;
    let beta = request
        .final_normalization
        .beta_f64le
        .as_ref()
        .map(|asset| load(asset, request.hidden_size, &parameter_loading))
        .transpose()?;
    parameter_loading.complete()?;
    let parameter_storage = parameter_loading.commit_persistent()?;
    let head = ResidentAffineHead::new(
        request.vocab_size,
        request.hidden_size,
        weights,
        bias,
        HeadArithmetic::ExactReal,
        &options(&construction),
    )?;
    let original_bytes = head.resident_bytes();
    let transfer_overlap = memory.receipt()?;
    // There was one physical vector transfer; the loader claim and owner's
    // claim deliberately overlap until the caller releases its loader lease.
    drop(head_loading);
    let after_transfer = memory.receipt()?;
    ensure!(
        transfer_overlap
            .live_pending_bytes
            .checked_sub(after_transfer.live_pending_bytes)
            == Some(head_load_request.persistent_bytes + head_load_request.scratch_bytes),
        "explicit conservative transfer overlap"
    );
    let norm = FinalNormalization {
        kind: request.final_normalization.kind,
        gamma: &gamma,
        beta: beta.as_deref(),
        epsilon: request.final_normalization.epsilon,
    };
    let first = &request.cases[0];
    let first_box = HiddenBox {
        lo: &first.lo,
        hi: &first.hi,
    };

    let cancelled = CancellationToken::default();
    let cancelled_budget = ledger(&memory, &cancelled, request.timeout_secs, QUERY_WORK)?;
    let cancelled_context = head.bind_query(&cancelled_budget)?;
    cancelled.cancel();
    ensure!(
        matches!(
            cancelled_context.verify_normalized_next_token(
                norm,
                first_box,
                first.target,
                &options(&cancelled_budget)
            ),
            Err(NormalizedHeadRefusal::Invalid(DecoderRefusal::Budget(
                WorkRefusal::Cancelled
            )))
        ),
        "cancelled query must refuse"
    );
    ensure!(
        head.bind_query(&cancelled_budget).is_err(),
        "cancelled query cannot be rebound"
    );
    drop(cancelled_context);
    let spent_budget = ledger(
        &memory,
        &CancellationToken::default(),
        request.timeout_secs,
        1024,
    )?;
    let spent_context = head.bind_query(&spent_budget)?;
    ensure!(
        matches!(
            spent_context.verify_normalized_next_token(
                norm,
                first_box,
                first.target,
                &options(&spent_budget)
            ),
            Err(NormalizedHeadRefusal::Invalid(DecoderRefusal::Budget(
                WorkRefusal::WorkLimit
            )))
        ),
        "spent query must refuse"
    );
    ensure!(
        head.bind_query(&spent_budget).is_err(),
        "spent query cannot be rebound"
    );
    drop(spent_context);
    let foreign_memory = MemoryAuthority::new(GLOBAL_BYTES);
    let foreign = ledger(
        &foreign_memory,
        &CancellationToken::default(),
        request.timeout_secs,
        QUERY_WORK,
    )?;
    ensure!(
        head.bind_query(&foreign).is_err() && foreign.receipt()?.work_reserved == 0,
        "equal-limit foreign authority must refuse before work"
    );

    let mut records = Vec::with_capacity(3);
    let mut retained = None;
    for case in &request.cases {
        driver.checkpoint()?;
        let query = ledger(
            &memory,
            &CancellationToken::default(),
            request.timeout_secs,
            QUERY_WORK,
        )?;
        let context = head.bind_query(&query)?;
        let replacement = ledger(
            &memory,
            &CancellationToken::default(),
            request.timeout_secs,
            QUERY_WORK,
        )?;
        ensure!(
            context
                .verify_normalized_next_token(
                    norm,
                    HiddenBox {
                        lo: &case.lo,
                        hi: &case.hi
                    },
                    case.target,
                    &options(&replacement)
                )
                .is_err()
                && replacement.receipt()?.work_reserved == 0,
            "nested substitute ledger must refuse"
        );
        let pre = HiddenBox {
            lo: &case.lo,
            hi: &case.hi,
        };
        let report =
            context.verify_normalized_next_token(norm, pre, case.target, &options(&query))?;
        let proof = certified(&report, case.target, request.vocab_size, &driver)?;
        ensure!(
            proof.shared_head_bytes() == original_bytes
                && proof.provenance() == NextTokenProvenance::CheckedQuery,
            "original resident query provenance"
        );
        let wire = context.export_normalized_json(&report.evidence, &options(&query))?;
        let wire_sha = sha_polled(wire.as_bytes(), &driver)?;
        let replay_budget = ledger(
            &memory,
            &CancellationToken::default(),
            request.timeout_secs,
            QUERY_WORK,
        )?;
        let replay_context = head.bind_query(&replay_budget)?;
        let replay = replay_context.replay_normalized_json(
            norm,
            pre,
            case.target,
            wire.as_bytes(),
            &options(&replay_budget),
        )?;
        let replay_proof = certified(&replay.report, case.target, request.vocab_size, &driver)?;
        ensure!(
            replay_proof.provenance() == NextTokenProvenance::OriginalReplay
                && replay.report.evidence.scope_digest == report.evidence.scope_digest,
            "independent complete original replay"
        );
        let replay_checked_stats = replay_proof.checked_stats().clone();
        if retained.is_none() {
            retained = Some(replay_proof);
        }
        records.push(json!({"target":case.target,"status":"certified","covered_competitors":request.vocab_size-1,
            "canonical_original_ids_checked":true,"scope_digest":report.evidence.scope_digest,
            "wire_bytes":wire.as_bytes().len(),"wire_sha256":wire_sha,
            "query_provenance":"CheckedQuery","replay_provenance":"OriginalReplay",
            "query_checked_stages":proof.checked_stats(),"replay_checked_stages":replay_checked_stats,
            "query_budget":budget_record(query.receipt()?),"replay_budget":budget_record(replay_budget.receipt()?),
            "replay_decoded_row_visits":replay.stats.parsed_row_visits,
            "replay_wire_original_identity_visits":replay.stats.original_identity_visits,
            "original_head_bytes":original_bytes}));
    }
    let cancelled_receipt = budget_record(cancelled_budget.receipt()?);
    let spent_receipt = budget_record(spent_budget.receipt()?);
    let retained = retained.context("retained original replay proof")?;
    let retained_clone = retained.clone();
    drop(head);
    drop(gamma);
    drop(beta);
    drop(parameter_storage);
    drop(request);
    drop(request_storage);
    let with_last_clones = memory.receipt()?;
    ensure!(
        with_last_clones.live_materialized_bytes >= original_bytes
            && retained.shared_head_bytes() == original_bytes,
        "sealed proof retains original model"
    );
    drop(retained);
    let after_first_clone = memory.receipt()?;
    ensure!(
        after_first_clone.live_materialized_bytes == with_last_clones.live_materialized_bytes,
        "first clone cannot release original or query claims"
    );
    drop(retained_clone);
    let after_last_clone = memory.receipt()?;
    ensure!(
        after_last_clone.live_materialized_bytes == 0,
        "last sealed proof releases model and query claims"
    );
    let result = json!({"schema":"ny.resident_head_probe_result.v1","status":"passed",
        "build_commit":env!("NY_BUILD_COMMIT"),"build_dirty":env!("NY_BUILD_DIRTY"),
        "build_source_digest":env!("NY_BUILD_SOURCE_DIGEST"),"build_profile":env!("NY_BUILD_PROFILE"),
        "build_target":env!("NY_BUILD_TARGET"),
        "claim":"conditional_mathematical_real_final_normalization_and_complete_original_head",
        "upstream_trajectory_certificate":"not_provided","normalization_execution_certificate":"not_provided",
        "performance_claim":false,"process_rss_measured":false,"global_participating_capacity_cap":GLOBAL_BYTES,
        "capacity_profile":"leased_driver_and_library_capacities;temporary_transfer_and_borrowed_norm_parameters_conservatively_overlap;not_whole_process_RSS",
        "request_sha256":request_sha,"original_head_bytes":original_bytes,"cases":records,
        "controls":{"cancelled_query_refused":true,"spent_query_refused":true,"fresh_queries_after_refusal_certified":true,
            "foreign_memory_refused_without_work":true,"substitute_ledger_refused_without_work":true,
            "last_clone_retains_original_storage":true},
        "cancelled_query_receipt":cancelled_receipt,"spent_query_receipt":spent_receipt,
        "construction_budget":budget_record(construction.receipt()?),
        "transfer_overlap":memory_record(transfer_overlap),"after_transfer":memory_record(after_transfer),
        "after_public_owner_drop":memory_record(with_last_clones),"after_first_proof_drop":memory_record(after_first_clone),
        "after_last_proof_drop":memory_record(after_last_clone)});
    driver.checkpoint()?;
    let encoded = serde_json::to_vec(&result)?;
    ensure!(
        encoded.capacity() <= OUTPUT_CAPACITY,
        "bounded diagnostic result capacity"
    );
    driver.checkpoint()?;
    driver.complete()?;
    // No authority claims are published after a failed checkpoint.
    println!("{}", std::str::from_utf8(&encoded)?);
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args_os();
    let _ = args.next();
    let path = args
        .next()
        .context("usage: resident_head_probe <closed-probe-request.json>")?;
    ensure!(
        args.next().is_none(),
        "exactly one request path is required"
    );
    run(Path::new(&path))
}
