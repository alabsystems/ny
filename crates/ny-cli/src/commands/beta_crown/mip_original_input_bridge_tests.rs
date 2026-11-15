// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! These are typed publication controls; none runs or simulates a MIP proof.
use super::*;

fn verified() -> VerificationResult {
    VerificationResult::Verified {
        provenance: Default::default(),
        output_bounds: vec![Bound::new_allow_infinite(0.25, 1.0)],
        proof: None,
        actual_method: Some(mip_unsat_method_tag()),
    }
}
struct Capture;
impl Capture {
    fn begin() -> Self {
        super::super::output::begin_capture();
        Self
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        super::super::output::end_capture();
    }
}

#[test]
fn bridge_mip_receipt_refuses_unknown_timeout_sat_and_uncertified_unsat_even_with_stale_json() {
    let _capture = Capture::begin();
    super::super::output::emit_competition_json(
        "stale verified-looking capture is not typed evidence",
    );
    let unknown = map_mip_nonsat_result(MipResult::Unsat { certified: false }, 1);
    let timeout = timeout_verification_result(1);
    let sat = VerificationResult::Violated {
        provenance: Default::default(),
        counterexample: vec![1.0],
        output: vec![0.0],
        details: None,
        actual_method: Some(ny_core::MethodUsed::MipHiGHS),
    };
    for result in [unknown, timeout, sat] {
        assert!(bridge_unsat_publication(&result, false, true).is_none());
    }
}

#[test]
fn bridge_mip_receipt_requires_successful_json_publication_and_active_capture() {
    super::super::output::end_capture();
    assert!(bridge_unsat_publication(&verified(), false, true).is_none());
    let _capture = Capture::begin();
    assert!(
        bridge_unsat_publication(&verified(), false, true).is_none(),
        "an empty capture is not publication"
    );
    super::super::output::emit_competition_json("exact new MIP emission");
    assert!(
        bridge_unsat_publication(&verified(), true, true).is_none(),
        "publication backend refusal is mandatory"
    );
    assert!(bridge_unsat_publication(&verified(), false, false).is_none());
    let receipt = bridge_unsat_publication(&verified(), false, true).unwrap();
    assert_eq!(receipt.into_captured_json(), "exact new MIP emission");
}

#[test]
fn bridge_mip_receipt_binds_exact_last_writer_and_capture_reset() {
    let _capture = Capture::begin();
    super::super::output::emit_competition_json("new MIP emission");
    let receipt = bridge_unsat_publication(&verified(), false, true)
        .unwrap()
        .into_captured_json();
    assert_eq!(
        super::super::output::snapshot_captured_json().as_deref(),
        Some(receipt.as_str())
    );
    super::super::output::emit_competition_json("later inconclusive BaB emission");
    assert_ne!(
        super::super::output::snapshot_captured_json().as_deref(),
        Some(receipt.as_str())
    );
    super::super::output::end_capture();
    assert!(super::super::output::snapshot_captured_json().is_none());
    super::super::output::begin_capture();
    assert!(super::super::output::snapshot_captured_json().is_none());
}

#[test]
fn late_typed_mip_verified_is_timeout_before_bridge_publication_receipt() {
    let _capture = Capture::begin();
    super::super::output::emit_competition_json("prior capture");
    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap();
    let late = gate_mip_result_at_deadline(verified(), 1, expired);
    assert!(matches!(late, VerificationResult::Timeout { .. }));
    assert!(bridge_unsat_publication(&late, false, true).is_none());
}
