// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Capability-based admission of the original-source dual-predicate owner.
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Instant;

use ny_onnx::vnnlib::nonlinear::{
    f32_chain::{Float32ChainAttempt, PreparedFloat32Chain},
    NonLinearFormula,
};
use sha2::{Digest, Sha256};

const MODEL_CAP: usize = 32 * 1024 * 1024;
const FORMULA_CAP: usize = 1024 * 1024;

fn bounded_bytes(path: &Path, cap: usize) -> Option<Vec<u8>> {
    let mut input = BufReader::new(File::open(path).ok()?);
    let gzip = input.fill_buf().ok()?.starts_with(&[0x1f, 0x8b]);
    let reader: Box<dyn Read> = if gzip {
        Box::new(flate2::read::GzDecoder::new(input))
    } else {
        Box::new(input)
    };
    let mut bytes = Vec::new();
    reader.take((cap + 1) as u64).read_to_end(&mut bytes).ok()?;
    (bytes.len() <= cap).then_some(bytes)
}

#[derive(Debug, PartialEq, Eq)]
enum Admission {
    Declined,
    Admitted(bool),
}

fn attempt(onnx: &Path, vnnlib: &Path, deadline: Instant) -> Admission {
    let _phase = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::Exact);
    let prepare = || {
        if Instant::now() >= deadline {
            return None;
        }
        let raw = bounded_bytes(onnx, MODEL_CAP)?;
        let spec = bounded_bytes(vnnlib, FORMULA_CAP)?;
        let formula = NonLinearFormula::parse(std::str::from_utf8(&spec).ok()?).ok()?;
        if !formula.is_nonlinear() {
            return None;
        }
        let chain = PreparedFloat32Chain::prepare(&raw, &formula, deadline, &mut || true)?;
        Some((raw, spec, formula, chain))
    };
    let Some((raw, spec, formula, chain)) = prepare() else {
        return Admission::Declined;
    };
    let model_sha = Sha256::digest(&raw);
    let spec_sha = Sha256::digest(&spec);
    let optimized_sha =
        chain
            .optimized_sha256()
            .iter()
            .fold(String::with_capacity(64), |mut text, b| {
                use std::fmt::Write as _;
                write!(&mut text, "{b:02x}").expect("writing to String is infallible");
                text
            });
    eprintln!("#nonlinear-f32-chain: admitted=1 model_sha256={model_sha:x} formula_sha256={spec_sha:x} optimized_sha256={optimized_sha}");
    let mut stats = Float32ChainAttempt::default();
    let proof = chain.refute_formula(&formula, deadline, &mut || true, &mut stats);
    // Complete-cover authority also belongs to these ORIGINAL source bytes
    // and this original deadline. A changed or unreadable file cannot reuse it.
    let fresh = proof.is_some()
        && bounded_bytes(onnx, MODEL_CAP).is_some_and(|now| now == raw)
        && bounded_bytes(vnnlib, FORMULA_CAP).is_some_and(|now| now == spec);
    let authorized = fresh && stats.completed && Instant::now() < deadline;
    eprintln!("#nonlinear-f32-chain: completed={} visited={} split={} dual_refuted={} real_open_refuted={} f32_pieces={} shared_scalar_heads={} source_fresh={}",u8::from(authorized),stats.visited,stats.split,stats.dual_refuted,stats.real_open_refuted,stats.f32_pieces,stats.shared_scalar_heads,u8::from(fresh));
    Admission::Admitted(authorized)
}

pub(super) fn try_source_first(
    onnx: &Path,
    vnnlib: &Path,
    timeout_secs: u64,
    work_deadline: Option<Instant>,
    case_split: &mut super::NonlinearCaseSplitStats,
) -> Option<bool> {
    let deadline = super::nonlinear_lane_deadline(timeout_secs, work_deadline)?;
    match attempt(onnx, vnnlib, deadline) {
        Admission::Declined => {
            super::try_nonlinear_unsat(onnx, vnnlib, timeout_secs, work_deadline, case_split)
        }
        // Once admitted, this owner keeps its original cover and deadline;
        // an incomplete dual proof grants no fallback proof authority.
        Admission::Admitted(complete) => complete.then_some(true),
    }
}

#[cfg(test)]
mod tests {
    use super::super::nonlinear_case_split_tests::write_scalar_gemm_and_spec;
    use super::*;
    use std::time::Duration;
    const SAFE:&str="(assert (and (>= X[0,0] 0) (<= X[0,0] 1))) (assert (>= (* X[0,0] X[0,0]) 0)) (assert (> Y[0,0] 2))";

    #[test]
    fn nonlinear_float32_default_source_route_refutes_through_the_original_fallback_bridge() {
        let (_dir, onnx, vnnlib) = write_scalar_gemm_and_spec(0., 0., SAFE);
        assert_eq!(
            attempt(&onnx, &vnnlib, Instant::now() + Duration::from_secs(5)),
            Admission::Admitted(true)
        );
        let mut stats = super::super::NonlinearCaseSplitStats::new(false);
        assert_eq!(
            try_source_first(&onnx, &vnnlib, 5, None, &mut stats),
            Some(true)
        );
        assert_eq!(stats.render(), None);
    }

    #[test]
    fn nonlinear_float32_unsupported_predicate_profile_retains_legacy_refusal_or_proof() {
        let spec="(assert (and (>= X[0,0] 0) (<= X[0,0] 1))) (assert (>= (* X[0,0] X[0,0]) 0)) (assert (> (/ Y[0,0] 2) 1))";
        let (_dir, onnx, vnnlib) = write_scalar_gemm_and_spec(0., 0., spec);
        assert_eq!(
            attempt(&onnx, &vnnlib, Instant::now() + Duration::from_secs(5)),
            Admission::Declined
        );
        let mut stats = super::super::NonlinearCaseSplitStats::new(false);
        assert_eq!(
            try_source_first(&onnx, &vnnlib, 5, None, &mut stats),
            Some(true)
        );
    }

    #[test]
    fn nonlinear_float32_source_route_cannot_publish_real_only_underflow_proofs() {
        let spec="(assert (and (>= X[0,0] 0) (<= X[0,0] 1.1754943508222875e-38))) (assert (and (> X[0,0] 0) (> Y[0,0] 0) (<= (* X[0,0] Y[0,0]) 0)))";
        let (_dir, onnx, vnnlib) = write_scalar_gemm_and_spec(1., 0., spec);
        assert_eq!(
            attempt(&onnx, &vnnlib, Instant::now() + Duration::from_millis(250)),
            Admission::Admitted(false)
        );
    }

    #[test]
    fn nonlinear_float32_expired_shared_deadline_refuses_before_source_reads() {
        let missing = Path::new("/nonexistent/ny-f32-source");
        let mut stats = super::super::NonlinearCaseSplitStats::new(false);
        assert_eq!(
            try_source_first(missing, missing, 100, Some(Instant::now()), &mut stats),
            None
        );
        assert_eq!(stats.render(), None);
    }
}
