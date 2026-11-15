// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original head/box entry, optionally composing an explicit real final norm.
//! Both routes independently replay original arithmetic before publication.
use anyhow::{bail, Context, Result};
use ny_api::decoder::{
    HeadArithmetic, HiddenBox, NextTokenOutcome, NextTokenProvenance, SharedAffineHead,
    VocabOptions, WorkBudget, WorkReserve,
};
use serde::Deserialize;
mod normalized;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    vocab_size: usize,
    hidden_size: usize,
    /// Original row-major binary64 little-endian coefficients.
    weights_f64le: PathBuf,
    /// Absent bias means exactly zero, rather than a guessed model bias.
    bias_f64le: Option<PathBuf>,
    arithmetic: HeadArithmetic,
    target: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    anchor: Vec<f64>,
    /// Presence explicitly changes lo/hi to pre-normalization coordinates.
    #[serde(default)]
    final_normalization: Option<normalized::FinalNormalizationRequest>,
}

fn bounded_bytes(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    if file.metadata()?.len() > cap {
        bail!("input exceeds its byte limit");
    }
    // A growing file cannot bypass the metadata preflight.
    let mut bytes = Vec::new();
    file.take(cap.checked_add(1).context("input byte limit overflow")?)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        bail!("input grew beyond its byte limit");
    }
    Ok(bytes)
}

fn coefficients(path: &Path, terms: usize, deadline: Instant) -> Result<Vec<f64>> {
    coefficients_polled(path, terms, deadline, || Ok(()))
}
fn coefficients_polled(
    path: &Path,
    terms: usize,
    deadline: Instant,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<Vec<f64>> {
    checkpoint()?;
    let mut file = BufReader::new(File::open(path)?);
    let bytes = terms.checked_mul(8).context("coefficient size overflow")?;
    if file.get_ref().metadata()?.len() != bytes as u64 {
        bail!("coefficient file size does not match the declared shape");
    }
    let mut values = Vec::new();
    values.try_reserve_exact(terms)?;
    for i in 0..terms {
        if i & 4095 == 0 {
            if Instant::now() >= deadline {
                bail!("head loading deadline");
            }
            checkpoint()?;
        }
        let mut word = [0u8; 8];
        file.read_exact(&mut word)?;
        let value = f64::from_le_bytes(word);
        if !value.is_finite() {
            bail!("nonfinite original coefficient");
        }
        values.push(value);
    }
    if file.read(&mut [0u8; 1])? != 0 {
        bail!("coefficient file grew beyond the declared shape");
    }
    checkpoint()?;
    Ok(values)
}

pub(crate) fn run(
    request_path: &Path,
    replay_path: Option<&Path>,
    evidence_path: Option<&Path>,
    timeout: u64,
    max_head_mb: usize,
) -> Result<()> {
    let start = Instant::now();
    let deadline = start
        .checked_add(Duration::from_secs(timeout))
        .context("timeout overflow")?;
    if timeout == 0 || max_head_mb == 0 {
        bail!("timeout and head memory limit must be positive");
    }
    let request: Request = serde_json::from_slice(&bounded_bytes(request_path, 4 << 20)?)?;
    if request.vocab_size == 0 || request.hidden_size == 0 {
        bail!("head dimensions must be positive");
    }
    let terms = request
        .vocab_size
        .checked_mul(request.hidden_size)
        .context("head shape overflow")?;
    let cap = max_head_mb
        .checked_mul(1 << 20)
        .context("head cap overflow")?;
    let head_bytes = terms
        .checked_add(request.vocab_size)
        .and_then(|n| n.checked_mul(8))
        .context("head byte size overflow")?;
    if head_bytes > cap {
        bail!("original head exceeds --max-head-mb");
    }
    if request.vocab_size > 1_000_000
        || request.hidden_size > 65_536
        || request.target >= request.vocab_size
        || request.lo.len() != request.hidden_size
        || request.hi.len() != request.hidden_size
        || request.anchor.len() != request.hidden_size
        || request
            .lo
            .iter()
            .zip(&request.hi)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        || request.anchor.iter().any(|x| !x.is_finite())
    {
        bail!("invalid target or hidden-state enclosure shape");
    }
    normalized::preflight(&request, &VocabOptions::default())?;
    let weights = coefficients(&request.weights_f64le, terms, deadline)?;
    let bias = match &request.bias_f64le {
        Some(path) => coefficients(path, request.vocab_size, deadline)?,
        None => vec![0.0; request.vocab_size],
    };
    let mut options = VocabOptions {
        deadline: Some(deadline),
        max_persistent_bytes: (cap as u64).saturating_add(128 << 20),
        ..VocabOptions::default()
    };
    let replay_work = if request.final_normalization.is_some() {
        ny_api::decoder::normalized_publication_work_bound(request.vocab_size, request.hidden_size)?
            .checked_add(if evidence_path.is_some() {
                ny_api::decoder::normalized_wire_size_bound(request.vocab_size)? as u64
            } else {
                0
            })
            .and_then(|n| n.checked_add(32 << 10))
            .context("normalized final publication work overflow")?
    } else {
        (terms as u64)
            .checked_add(request.vocab_size as u64)
            .and_then(|n| n.checked_mul(24))
            .context("replay work prediction overflow")?
    };
    options.budget = Some(WorkBudget::new_reserved(
        Some(deadline),
        options.max_scratch_bytes as usize,
        options.max_persistent_bytes as usize,
        options.max_terms,
        options.max_proposals,
        Duration::from_secs(timeout / 10),
        WorkReserve {
            persistent_bytes: 0,
            work: replay_work,
            proposals: 6,
        },
    )?);
    let head = SharedAffineHead::new(
        request.vocab_size,
        request.hidden_size,
        weights,
        bias,
        request.arithmetic,
        &options,
    )?;
    if request.final_normalization.is_some() {
        return normalized::run(&request, &head, &options, replay_path, evidence_path, start);
    }
    let hidden = HiddenBox {
        lo: &request.lo,
        hi: &request.hi,
    };
    let report = if let Some(path) = replay_path {
        let evidence = bounded_bytes(path, options.max_evidence_bytes as u64)?;
        head.replay_next_token_json(hidden, &request.anchor, request.target, &evidence, &options)?
    } else {
        // This command handles one supplied box. Dense original support avoids
        // cold per-target tree construction; prepared trees remain the warm API.
        // The dense API and the emitted wire are both independently replayed.
        head.verify_dense_next_token(hidden, &request.anchor, request.target, &options)?
    };
    // Default executable arrival always independently replays generated evidence.
    let serialized = head.export_json(&report.evidence, &options)?;
    let replayed = head.replay_next_token_json(
        hidden,
        &request.anchor,
        request.target,
        serialized.as_bytes(),
        &options,
    )?;
    if report.is_certified() != replayed.is_certified() {
        bail!("head query and independent replay disagree");
    }
    if replayed.provenance() != NextTokenProvenance::OriginalReplay {
        bail!("head publication requires original replay provenance");
    }
    if let Some(path) = evidence_path {
        let mut file = File::options().write(true).create_new(true).open(path)?;
        file.write_all(serialized.as_bytes())?;
        file.write_all(b"\n")?;
    }
    let (status, covered) = match &replayed.outcome {
        NextTokenOutcome::Certified(proof) => ("certified", proof.covered_competitors()),
        NextTokenOutcome::Unknown => ("unknown", 0),
    };
    println!(
        "{}",
        serde_json::json!({
            "schema": "ny.affine_head_result.v1",
            "evidence_schema_version": 2,
            "status": status,
            "claim": "strict_target_argmax_over_supplied_hidden_box",
            "upstream_trajectory_status": "not_provided",
            "target": request.target,
            "arithmetic": request.arithmetic,
            "hidden_input_domain": if matches!(request.arithmetic, HeadArithmetic::Binary32 { .. }) {
                "finite_binary32_coordinates_in_supplied_box"
            } else { "real_coordinates_in_supplied_box" },
            "shared_head_bytes": head.reserved_bytes(),
            "shared_head_identity_terms": head.identity_terms(),
            "covered_competitors": covered,
            "vocab_size": request.vocab_size,
            "independent_replay": true,
            "decision_provenance": report.provenance(),
            "replay_provenance": replayed.provenance(),
            "scope_digest": replayed.evidence.scope_digest,
            "query_stats": report.stats,
            "replay_stats": replayed.stats,
            "wire_export_stats": serialized.stats(),
            "whole_wall_seconds": start.elapsed().as_secs_f64(),
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_cli_exports_complete_vocabulary_and_refuses_changed_original_head() {
        let directory = tempfile::tempdir().unwrap();
        let weights = directory.path().join("weights.f64le");
        let biases = directory.path().join("bias.f64le");
        std::fs::write(&weights, vec![0; 256 * 2 * 8]).unwrap();
        let mut bias = vec![0.0f64; 256];
        bias[17] = 1.0;
        std::fs::write(
            &biases,
            bias.iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let request = directory.path().join("request.json");
        let evidence = directory.path().join("evidence.json");
        std::fs::write(
            &request,
            serde_json::to_vec(&serde_json::json!({
                "vocab_size": 256, "hidden_size": 2,
                "weights_f64le": weights, "bias_f64le": biases,
                "arithmetic": "ExactReal", "target": 17,
                "anchor": [0.0, 0.0], "lo": [-1.0, -0.25], "hi": [0.5, 2.0],
            }))
            .unwrap(),
        )
        .unwrap();
        run(&request, None, Some(&evidence), 60, 1).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&evidence).unwrap()).unwrap();
        assert_eq!(parsed["schema_version"], 2);
        let mut members = Vec::new();
        for group in parsed["groups"].as_array().unwrap() {
            assert!(group.get("members").is_none());
            for range in group["member_ranges"].as_array().unwrap() {
                members.extend(
                    range[0].as_u64().unwrap() as usize..range[1].as_u64().unwrap() as usize,
                );
            }
            assert!(group["upper"].as_f64().unwrap() < 0.0);
        }
        members.sort_unstable();
        assert_eq!(members, (0..256).filter(|id| *id != 17).collect::<Vec<_>>());
        run(&request, Some(&evidence), None, 60, 1).unwrap();
        // The same dimensions and target cannot license an altered head.
        bias[18] = 2.0;
        std::fs::write(
            &biases,
            bias.iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(run(&request, Some(&evidence), None, 60, 1).is_err());
    }

    #[test]
    fn head_cli_admits_shape_and_file_size_before_reading_coefficients() {
        let directory = tempfile::tempdir().unwrap();
        let request = directory.path().join("request.json");
        // File is intentionally absent: the head cap must refuse first.
        std::fs::write(
            &request,
            serde_json::to_vec(&serde_json::json!({
                "vocab_size": 1_000_000, "hidden_size": 1_000_000,
                "weights_f64le": directory.path().join("absent.f64le"),
                "arithmetic": "ExactReal", "target": 0,
                "anchor": [0.0], "lo": [0.0], "hi": [0.0],
            }))
            .unwrap(),
        )
        .unwrap();
        let error = run(&request, None, None, 60, 1).unwrap_err();
        assert!(error.to_string().contains("exceeds --max-head-mb"));
        let wrong = directory.path().join("wrong.f64le");
        std::fs::write(&wrong, [0u8; 8]).unwrap();
        assert!(
            coefficients(&wrong, 2, Instant::now() + Duration::from_secs(60))
                .unwrap_err()
                .to_string()
                .contains("file size")
        );
    }
}
