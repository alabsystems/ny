// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Explicit pre-normalization input and same-session normalized publication.
use super::*;
use ny_api::decoder::{
    export_shared_normalized_json, normalized_wire_size_bound, replay_shared_normalized_json,
    verify_shared_normalized_next_token, FinalNormalization, FinalNormalizationKind, LeaseClass,
    NormalizedNextTokenOutcome, PersistentReservation, WorkLease, WorkRequest,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FinalNormalizationRequest {
    pub kind: FinalNormalizationKind,
    pub arithmetic: NormalizationArithmetic,
    pub input_domain: PreNormalizationDomain,
    pub gamma_f64le: PathBuf,
    pub beta_f64le: Option<PathBuf>,
    pub epsilon: f64,
}
/// No deployed floating-point norm contract is inferred from a real head.
#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) enum NormalizationArithmetic {
    ExactReal,
}
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PreNormalizationDomain {
    PreNormalizationBox,
}

pub(super) fn preflight(request: &Request, limits: &VocabOptions) -> Result<()> {
    let Some(norm) = &request.final_normalization else {
        return Ok(());
    };
    if request.arithmetic != HeadArithmetic::ExactReal {
        bail!("final_normalization requires ExactReal head arithmetic");
    }
    // These closed enums are required rather than guessed from paths/names.
    match (norm.arithmetic, norm.input_domain) {
        (NormalizationArithmetic::ExactReal, PreNormalizationDomain::PreNormalizationBox) => {}
    }
    if !norm.epsilon.is_finite() || norm.epsilon < 1e-12f32 as f64 {
        bail!("final_normalization epsilon is outside the checked unchanged domain");
    }
    if normalized_wire_size_bound(request.vocab_size)? > limits.max_evidence_bytes {
        bail!("normalized publication exceeds the evidence cap");
    }
    let scratch = request
        .hidden_size
        .checked_mul(512)
        .and_then(|n| n.checked_add(16 << 10))
        .context("normalized scratch size overflow")?;
    if scratch as u64 > limits.max_scratch_bytes {
        bail!("normalized scratch exceeds its cap");
    }
    Ok(())
}

fn poll(ledger: &WorkBudget, limits: &VocabOptions) -> Result<()> {
    ledger.checkpoint_for(LeaseClass::Replay)?;
    if limits.deadline.is_some_and(|d| Instant::now() >= d)
        || ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d)
    {
        bail!("normalized publication deadline");
    }
    Ok(())
}
fn lease_poll(lease: &WorkLease, ledger: &WorkBudget, limits: &VocabOptions) -> Result<()> {
    lease.checkpoint()?;
    poll(ledger, limits)
}
fn admit_live(ledger: &WorkBudget, request: WorkRequest, limits: &VocabOptions) -> Result<()> {
    let live = ledger.receipt()?;
    if live
        .live_persistent_bytes
        .checked_add(request.persistent_bytes)
        .is_none_or(|n| n as u64 > limits.max_persistent_bytes)
        || live
            .live_scratch_bytes
            .checked_add(request.scratch_bytes)
            .is_none_or(|n| n as u64 > limits.max_scratch_bytes)
    {
        bail!("normalized CLI aggregate resident cap");
    }
    Ok(())
}
struct Parameters {
    gamma: Vec<f64>,
    beta: Option<Vec<f64>>,
    _reservation: PersistentReservation,
}
fn load_parameters(
    owner: &SharedAffineHead,
    request: &FinalNormalizationRequest,
    limits: &VocabOptions,
) -> Result<Parameters> {
    let ledger = owner.budget();
    let count = owner.as_head().hidden_size;
    let values = count
        .checked_mul(1 + usize::from(request.beta_f64le.is_some()))
        .context("normalization parameter shape overflow")?;
    let bytes = values
        .checked_mul(8)
        .and_then(|n| n.checked_add(4096))
        .context("normalization parameter byte overflow")?;
    let prediction = WorkRequest {
        persistent_bytes: bytes,
        scratch_bytes: 16 << 10,
        work: (values as u64)
            .checked_add(4096)
            .context("parameter work overflow")?,
        proposals: 1,
    };
    admit_live(&ledger, prediction, limits)?;
    let mut lease = ledger.lease_for(LeaseClass::Replay, prediction)?;
    lease_poll(&lease, &ledger, limits)?;
    // H<=65536. The existing loader polls every4096 coefficients, validates
    // finite exact-length originals and catches concurrent file growth.
    let deadline = limits
        .deadline
        .context("CLI normalized deadline required")?;
    let gamma = coefficients_polled(&request.gamma_f64le, count, deadline, || {
        lease_poll(&lease, &ledger, limits)
    })?;
    lease_poll(&lease, &ledger, limits)?;
    let beta = request
        .beta_f64le
        .as_ref()
        .map(|p| coefficients_polled(p, count, deadline, || lease_poll(&lease, &ledger, limits)))
        .transpose()?;
    lease_poll(&lease, &ledger, limits)?;
    let actual = gamma
        .capacity()
        .checked_add(beta.as_ref().map_or(0, Vec::capacity))
        .and_then(|n| n.checked_mul(8))
        .context("parameter actual capacity overflow")?;
    if actual > bytes {
        bail!("normalization actual parameter capacity");
    }
    lease.complete()?;
    let reservation = lease.commit_persistent()?;
    Ok(Parameters {
        gamma,
        beta,
        _reservation: reservation,
    })
}
struct Input {
    bytes: Vec<u8>,
    _reservation: PersistentReservation,
}
/// Evidence source storage is separately leased before allocation; bounded
/// streaming decoding keeps its expansion charged through original replay.
fn read_evidence(owner: &SharedAffineHead, path: &Path, limits: &VocabOptions) -> Result<Input> {
    let ledger = owner.budget();
    let mut file = File::open(path)?;
    let len = usize::try_from(file.metadata()?.len()).context("evidence size overflow")?;
    if len > limits.max_evidence_bytes {
        bail!("normalized input evidence cap");
    }
    let cap = len
        .checked_add(1)
        .context("evidence growth sentinel overflow")?;
    let prediction = WorkRequest {
        persistent_bytes: cap.checked_add(1024).context("input capacity overflow")?,
        scratch_bytes: 4096,
        work: (cap as u64)
            .checked_add(4096)
            .context("input work overflow")?,
        proposals: 1,
    };
    admit_live(&ledger, prediction, limits)?;
    let mut lease = ledger.lease_for(LeaseClass::Replay, prediction)?;
    lease_poll(&lease, &ledger, limits)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(cap)?;
    let mut block = [0u8; 4096];
    loop {
        lease_poll(&lease, &ledger, limits)?;
        let remaining = cap - bytes.len();
        if remaining == 0 {
            bail!("normalized evidence file grew beyond declared size");
        }
        let got = file.read(&mut block[..remaining.min(4096)])?;
        if got == 0 {
            break;
        }
        bytes.extend_from_slice(&block[..got]);
        if bytes.len() > len {
            bail!("normalized evidence file grew beyond declared size");
        }
    }
    if bytes.len() != len {
        bail!("normalized evidence file changed size");
    }
    if bytes.capacity() > prediction.persistent_bytes {
        bail!("normalized input actual capacity");
    }
    lease_poll(&lease, &ledger, limits)?;
    lease.complete()?;
    Ok(Input {
        bytes,
        _reservation: lease.commit_persistent()?,
    })
}

pub(super) fn run(
    request: &Request,
    owner: &SharedAffineHead,
    limits: &VocabOptions,
    replay_path: Option<&Path>,
    evidence_path: Option<&Path>,
    start: Instant,
) -> Result<()> {
    let payload = request
        .final_normalization
        .as_ref()
        .context("missing final normalization")?;
    preflight(request, limits)?;
    let parameters = load_parameters(owner, payload, limits)?;
    let norm = FinalNormalization {
        kind: payload.kind,
        gamma: &parameters.gamma,
        beta: parameters.beta.as_deref(),
        epsilon: payload.epsilon,
    };
    let pre = HiddenBox {
        lo: &request.lo,
        hi: &request.hi,
    };
    let (report, input_wire_stats) = if let Some(path) = replay_path {
        let input = read_evidence(owner, path, limits)?;
        let replay =
            replay_shared_normalized_json(owner, norm, pre, request.target, &input.bytes, limits)?;
        (replay.report, Some(replay.stats))
    } else {
        (
            verify_shared_normalized_next_token(owner, norm, pre, request.target, limits)?,
            None,
        )
    };
    // Plain serialized proposals acquire no authority. Always reconstruct the
    // original norm, head and complete competitor cover before publication.
    let serialized = export_shared_normalized_json(owner, &report.evidence, limits)?;
    let replay = replay_shared_normalized_json(
        owner,
        norm,
        pre,
        request.target,
        serialized.as_bytes(),
        limits,
    )?;
    let replayed = &replay.report;
    if matches!(report.outcome, NormalizedNextTokenOutcome::Certified(_))
        != matches!(replayed.outcome, NormalizedNextTokenOutcome::Certified(_))
    {
        bail!("normalized query and independent original replay disagree");
    }
    if replayed.provenance() != NextTokenProvenance::OriginalReplay {
        bail!("normalized publication requires original replay provenance");
    }
    let ledger = owner.budget();
    poll(&ledger, limits)?;
    // Final file-copy and fixed-size JSON result metadata are a separate
    // publication phase, funded on the original ledger before its allocations.
    let publication = WorkRequest {
        persistent_bytes: 32 << 10,
        scratch_bytes: 4096,
        work: (if evidence_path.is_some() {
            serialized.as_bytes().len()
        } else {
            0
        })
        .checked_add(32 << 10)
        .context("publication work overflow")? as u64,
        proposals: 1,
    };
    admit_live(&ledger, publication, limits)?;
    let mut publication_lease = ledger.lease_for(LeaseClass::Replay, publication)?;
    lease_poll(&publication_lease, &ledger, limits)?;
    if let Some(path) = evidence_path {
        let mut file = File::options().write(true).create_new(true).open(path)?;
        for block in serialized.as_bytes().chunks(4096) {
            poll(&ledger, limits)?;
            file.write_all(block)?;
        }
        file.write_all(b"\n")?;
    }
    poll(&ledger, limits)?;
    let (status, covered) = match &replayed.outcome {
        NormalizedNextTokenOutcome::Certified(proof) => ("certified", proof.covered_competitors()),
        NormalizedNextTokenOutcome::Unknown => ("unknown", 0),
    };
    let result = serde_json::json!({
        "schema": "ny.normalized_head_result.v1", "evidence_schema_version": 1,
        "status": status, "claim": "strict_target_argmax_over_supplied_pre_normalization_box",
        "upstream_trajectory_status": "not_provided", "normalization_execution_status": "not_provided",
        "normalization": payload.kind, "epsilon": payload.epsilon, "arithmetic": "ExactReal",
        "hidden_input_domain": "real_pre_normalization_coordinates_in_supplied_box",
        "target": request.target, "vocab_size": request.vocab_size,
        "shared_head_bytes": owner.reserved_bytes(), "shared_head_identity_terms": owner.identity_terms(),
        "covered_competitors": covered, "independent_replay": true,
        "decision_provenance": report.provenance(), "replay_provenance": replayed.provenance(),
        "scope_digest": replayed.evidence.scope_digest, "query_stats": report.stats,
        "replay_stats": replayed.stats, "input_wire_stats": input_wire_stats,
        "wire_replay_stats": replay.stats, "wire_export_stats": serialized.stats(),
        "whole_wall_seconds": start.elapsed().as_secs_f64(),
    });
    // Only fixed schema fields, bounded numeric receipts and a32-byte digest
    // enter this result: no vocabulary rows, parameter arrays or source paths.
    let result = serde_json::to_string(&result)?;
    if result.capacity() > 16 << 10 {
        bail!("normalized result metadata cap");
    }
    lease_poll(&publication_lease, &ledger, limits)?;
    publication_lease.complete()?;
    // No post-publication budget failure: the prepared result is emitted only
    // after the final checked completion, with its lease still alive.
    println!("{result}");
    Ok(())
}

#[cfg(test)]
mod tests;
