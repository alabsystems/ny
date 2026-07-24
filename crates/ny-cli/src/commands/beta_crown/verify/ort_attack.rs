// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ORT-routed attack forwards (#four-walls attack-phase fix).
//!
//! The PGD / sampling attack phases score candidate inputs with concrete
//! forward passes. The internal graph point-forward is exact but slow on
//! large conv nets (per-layer GEMM dispatches with host<->GPU transfers:
//! ~45ms/eval on traffic_signs, 180s+ for one vggnet16 restart batch); the
//! ONNX-Runtime session evaluates the same model in milliseconds. This module
//! lets attack phases route candidate SCORING through a lazily-built ORT
//! session for the instance model.
//!
//! Soundness: attack scoring can never create a false `sat` — a candidate only
//! ever produces a violation CLAIM, and every claimed witness is still
//! re-checked by the independent trusted-oracle ORT gate in the vnncomp
//! translator (a separate session built from the raw ONNX protobuf) before
//! any `sat` is scored. A failure to find a counterexample leaves the verdict
//! to the (untouched) bound-propagation path. Disable with `NY_ORT_ATTACK=0`.

use ndarray::ArrayD;
use std::path::PathBuf;
use std::sync::Mutex;

enum OracleState {
    /// Model registered; session not built yet (built lazily on first use so
    /// non-attack runs pay nothing).
    Registered { path: PathBuf, input_len: usize },
    /// Session built and usable.
    Ready(Box<ny_onnx::diff::OrtForward>),
    /// Construction failed or the flag disabled it — never retried.
    Unavailable,
}

static ATTACK_ORACLE: Mutex<Option<OracleState>> = Mutex::new(None);

/// Whether ORT-routed attack scoring is enabled (`NY_ORT_ATTACK=0` disables).
fn ort_attack_enabled() -> bool {
    std::env::var("NY_ORT_ATTACK").map_or(true, |v| v != "0")
}

/// Register the instance model for ORT-routed attack scoring.
///
/// Called once per beta-crown command with the model path and the property's
/// flat input element count; the ORT session is built lazily on the first
/// attack forward. `input_len == None` (epsilon-ball mode, no VNN-LIB spec)
/// clears any previous registration.
pub(in crate::commands::beta_crown) fn register_attack_model(
    path: PathBuf,
    input_len: Option<usize>,
) {
    let mut guard = match ATTACK_ORACLE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    *guard = match input_len {
        Some(input_len) if ort_attack_enabled() => {
            Some(OracleState::Registered { path, input_len })
        }
        _ => None,
    };
}

/// Run one ORT forward on a flat candidate point.
///
/// Returns `None` when the oracle is disabled, unavailable, or the point
/// length does not match the registered model input (callers fall back to the
/// internal graph forward). The output is the flattened first model output.
pub(in crate::commands::beta_crown) fn ort_forward_point(
    point: &ArrayD<f32>,
) -> Option<ArrayD<f32>> {
    let flat: Vec<f32>;
    let slice: &[f32] = match point.as_slice() {
        Some(s) => s,
        None => {
            flat = point.iter().copied().collect();
            &flat
        }
    };
    let out = ort_forward_flat(slice)?;
    ArrayD::from_shape_vec(ndarray::IxDyn(&[out.len()]), out).ok()
}

/// Run one ORT forward on a flat slice, building the session on first use.
pub(in crate::commands::beta_crown) fn ort_forward_flat(flat_input: &[f32]) -> Option<Vec<f32>> {
    let mut guard = match ATTACK_ORACLE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    loop {
        match guard.as_mut()? {
            OracleState::Registered { path, input_len } => {
                if *input_len != flat_input.len() {
                    // Layout mismatch between graph input and ONNX input —
                    // fail closed to the internal forward, permanently.
                    tracing::debug!(
                        registered = *input_len,
                        got = flat_input.len(),
                        "ORT attack oracle: input length mismatch; disabling for this instance"
                    );
                    *guard = Some(OracleState::Unavailable);
                    return None;
                }
                let path = path.clone();
                let input_len = *input_len;
                match ny_onnx::diff::OrtForward::from_path(&path, input_len) {
                    Ok(forward) if forward.input_len() == input_len => {
                        tracing::info!(
                            model = %path.display(),
                            input_len,
                            "Attack scoring routed through ONNX Runtime (NY_ORT_ATTACK=0 to disable)"
                        );
                        *guard = Some(OracleState::Ready(Box::new(forward)));
                    }
                    Ok(forward) => {
                        tracing::debug!(
                            declared = forward.input_len(),
                            expected = input_len,
                            "ORT attack oracle: declared model input length mismatch; \
                             falling back to internal forwards"
                        );
                        *guard = Some(OracleState::Unavailable);
                        return None;
                    }
                    Err(err) => {
                        tracing::debug!(
                            error = %err,
                            "ORT attack oracle unavailable; falling back to internal forwards"
                        );
                        *guard = Some(OracleState::Unavailable);
                        return None;
                    }
                }
                // Loop around into the Ready arm.
            }
            OracleState::Ready(forward) => {
                if forward.input_len() != flat_input.len() {
                    return None;
                }
                return match forward.run(flat_input) {
                    Ok(out) if !out.is_empty() => Some(out),
                    Ok(_) => None,
                    Err(err) => {
                        // A failing session will keep failing — disable rather
                        // than paying the error path per candidate.
                        tracing::debug!(
                            error = %err,
                            "ORT attack forward failed; disabling oracle for this instance"
                        );
                        *guard = Some(OracleState::Unavailable);
                        None
                    }
                };
            }
            OracleState::Unavailable => return None,
        }
    }
}

/// Whether an ORT oracle is registered/usable for candidate scoring right now
/// (without forcing session construction).
pub(in crate::commands::beta_crown) fn ort_attack_registered_for_len(input_len: usize) -> bool {
    let guard = match ATTACK_ORACLE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    match guard.as_ref() {
        Some(OracleState::Registered { input_len: l, .. }) => *l == input_len,
        Some(OracleState::Ready(forward)) => forward.input_len() == input_len,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    // NOTE: tests share the process-global oracle; each test fully installs
    // the state it needs first. Serialize via a local lock to keep them from
    // interleaving under the parallel test runner.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn unregistered_oracle_returns_none() {
        let _l = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        register_attack_model(PathBuf::from("/nonexistent.onnx"), None);
        assert!(ort_forward_flat(&[0.0_f32; 4]).is_none());
        assert!(!ort_attack_registered_for_len(4));
    }

    #[test]
    fn missing_model_fails_closed_and_stays_unavailable() {
        let _l = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        register_attack_model(PathBuf::from("/nonexistent-model.onnx"), Some(4));
        assert!(ort_attack_registered_for_len(4));
        // First call attempts (and fails) session construction.
        assert!(ort_forward_flat(&[0.0_f32; 4]).is_none());
        // Now permanently unavailable — no repeated construction attempts.
        assert!(!ort_attack_registered_for_len(4));
        assert!(ort_forward_flat(&[0.0_f32; 4]).is_none());
        register_attack_model(PathBuf::from("x"), None);
    }

    #[test]
    fn input_length_mismatch_fails_closed() {
        let _l = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        register_attack_model(PathBuf::from("/nonexistent-model.onnx"), Some(8));
        let point = ArrayD::<f32>::zeros(IxDyn(&[2, 2]));
        // 4 != 8: mismatch short-circuits before any session build.
        assert!(ort_forward_point(&point).is_none());
        assert!(!ort_attack_registered_for_len(8));
        register_attack_model(PathBuf::from("x"), None);
    }
}
