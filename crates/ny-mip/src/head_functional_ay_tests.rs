// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::head_functional_advice::{
    HeadFunctionalLpRow, FUNCTIONAL_MAX_NONZEROS, FUNCTIONAL_MAX_ROWS, FUNCTIONAL_MAX_VARIABLES,
};
use ny_core::head_phase_advice::MAX_MAGNITUDE;

fn six_coordinate_request() -> HeadFunctionalLpRequest {
    // The surrogate is exact for this source: z_i=x_i+2 in [1,3],
    // y_i=z_i-2, stable y7=sum(z_i)-5. Source weights are six 1s
    // followed by -3/8; B=43/16 gives exact minimum 1/16.
    let mut rows = Vec::new();
    let mut slope_rows = Vec::new();
    for i in 0..6 {
        rows.push(HeadFunctionalLpRow {
            coefficients: vec![(6 + i, 1.0)],
            lower: 0.0,
        });
        slope_rows.push(Some(rows.len()));
        rows.push(HeadFunctionalLpRow {
            coefficients: vec![(i, -1.0), (6 + i, 1.0)],
            lower: -2.0,
        });
    }
    slope_rows.push(None); // stable negative head stays in owned scalar replay
    HeadFunctionalLpRequest {
        protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
        feature_count: 6,
        bounds: vec![[1.0, 3.0]; 6]
            .into_iter()
            .chain(vec![[0.0, 1.0]; 6])
            .collect(),
        objective: vec![-0.375; 6].into_iter().chain(vec![1.0; 6]).collect(),
        rows,
        slope_rows,
        phase_rows: vec![],
    }
}

fn functional_response(request: &HeadFunctionalLpRequest, dual: f64) -> Vec<u8> {
    encode_bounded(
        &FunctionalHelperResponse {
            protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
            ay_revision: AY_REVISION.into(),
            hint: HeadFunctionalLpHint {
                row_duals: vec![dual; request.rows.len()],
            },
        },
        FUNCTIONAL_MAX_RESPONSE_BYTES,
        None,
    )
    .unwrap()
}

fn scalar_intercept(q: f64, slope: f64, lower: f64, upper: f64) -> f64 {
    let mut result = f64::INFINITY;
    for y in [lower, upper] {
        result = result.min(q * y.max(0.0) - slope * y);
    }
    if lower <= 0.0 && upper >= 0.0 {
        result = result.min(0.0);
    }
    result
}

fn full_source_replay(
    request: &HeadFunctionalLpRequest,
    hint: &HeadFunctionalLpHint,
    b: f64,
) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for sign in [1.0, -1.0] {
        // All seven source heads remain present. The negative stable slope
        // cannot be replaced by a raw dual or dropped from the remainder.
        let mut slopes = [-0.375; 7];
        for (head, row) in request.slope_rows.iter().enumerate() {
            if let Some(row) = row {
                slopes[head] = (sign * hint.row_duals[*row]).clamp(0.0, 1.0);
            }
        }
        let mut scalar = 0.0;
        for &slope in &slopes[..6] {
            scalar += scalar_intercept(1.0, slope, -1.0, 1.0);
        }
        scalar += scalar_intercept(-0.375, slopes[6], 1.0, 13.0);
        // Native source-functional reduction in the original x box:
        // sum(v_i*x_i) -3/8*(7+sum(x_i)). Every coefficient is kept.
        let lower = b + scalar + 7.0 * slopes[6]
            - slopes[..6]
                .iter()
                .map(|&v| (v + slopes[6]).abs())
                .sum::<f64>();
        best = best.max(lower);
    }
    best
}

#[test]
fn pinned_float_global_advice_closes_six_independent_cancellation_coordinates() {
    assert!(include_str!("../Cargo.toml").contains(AY_REVISION));
    let request = six_coordinate_request();
    assert!(request.validate());
    let hint = functional_float_advice(&request, Instant::now() + Duration::from_secs(10)).unwrap();
    assert!(hint.validate(&request));
    // Assert the authored source replay, rather than one prescribed vector
    // or any AY objective, status, primal or certificate assertion.
    let lower = full_source_replay(&request, &hint, 43.0 / 16.0);
    assert!(lower > 0.0, "actual AY hints must improve the full source");
    assert!(
        lower <= 1.0 / 16.0,
        "exact minimum bounds every valid replay"
    );
    // Same slopes cannot certify the changed source whose exact minimum is
    // -1/16; this retains the complete scalar remainder and source bias.
    let negative = full_source_replay(&request, &hint, 41.0 / 16.0);
    assert!(negative <= -1.0 / 16.0);
    // A four-neuron group with frozen 1/2 slopes outside the group retains
    // two independent 1/8 residuals, so its bound is at most -3/16.
    assert_eq!(1.0 / 16.0 - 2.0 * (0.5 - 0.375), -3.0 / 16.0);
}

#[test]
fn arbitrary_global_hints_never_override_the_negative_authored_remainder() {
    let request = six_coordinate_request();
    for dual in [-MAX_MAGNITUDE, -0.375, 0.0, 0.375, 1.0, MAX_MAGNITUDE] {
        let hint = HeadFunctionalLpHint {
            row_duals: vec![dual; request.rows.len()],
        };
        assert!(hint.validate(&request));
        assert!(full_source_replay(&request, &hint, 41.0 / 16.0) <= -1.0 / 16.0);
    }
}

#[test]
fn functional_transport_rejects_wrong_versions_pins_lengths_caps_and_authority_fields() {
    let request = six_coordinate_request();
    let valid = functional_response(&request, -0.375);
    assert_eq!(
        decode_functional_hint(&valid, &request).unwrap().row_duals[1],
        -0.375
    );
    assert!(decode_functional_hint(b"{}", &request).is_none());
    assert!(
        decode_functional_hint(&vec![b' '; FUNCTIONAL_MAX_RESPONSE_BYTES + 1], &request).is_none()
    );
    let mut body: serde_json::Value = serde_json::from_slice(&valid).unwrap();
    for version in [PROTOCOL_VERSION, 99] {
        body["protocol_version"] = serde_json::json!(version);
        assert!(decode_functional_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    }
    body["protocol_version"] = serde_json::json!(FUNCTIONAL_PROTOCOL_VERSION);
    body["ay_revision"] = serde_json::json!("wrong-pin");
    assert!(decode_functional_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    body["ay_revision"] = serde_json::json!(AY_REVISION);
    body["hint"]["row_duals"] = serde_json::json!([0]);
    assert!(decode_functional_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    body["hint"]["row_duals"] = serde_json::json!(vec![2.0 * MAX_MAGNITUDE; request.rows.len()]);
    assert!(decode_functional_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    body["hint"]["row_duals"] = serde_json::json!(vec![0.0; request.rows.len()]);
    body["hint"]["lower"] = serde_json::json!(1.0);
    assert!(decode_functional_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    assert!(encode_bounded(&request, 4, None).is_err());
    assert!(encode_bounded(&request, FUNCTIONAL_MAX_MESSAGE_BYTES, Some(Instant::now())).is_err());
}

#[test]
fn malformed_sparse_global_requests_decline_before_ay_model_work() {
    let request = six_coordinate_request();
    assert!(functional_float_advice(&request, Instant::now()).is_err());
    let mut bad = request.clone();
    bad.rows[1].coefficients.reverse();
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request.clone();
    bad.rows[1].coefficients.push((6, 1.0));
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request.clone();
    bad.rows[0].coefficients[0] = (FUNCTIONAL_MAX_VARIABLES, 1.0);
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request.clone();
    bad.rows[0].coefficients[0].1 = 0.0;
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request.clone();
    bad.rows[0].lower = f64::NAN;
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request.clone();
    bad.feature_count = 129;
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request.clone();
    bad.slope_rows[1] = bad.slope_rows[0];
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request;
    bad.phase_rows.push(1); // role alias cannot select an epigraph row
    assert!(functional_float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
}

#[test]
fn maximum_sparse_global_request_has_bounded_wire_and_workspace_estimates() {
    let mut request = HeadFunctionalLpRequest {
        protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
        feature_count: 128,
        bounds: vec![[-1.0, 1.0]; FUNCTIONAL_MAX_VARIABLES],
        objective: vec![0.0; FUNCTIONAL_MAX_VARIABLES],
        rows: vec![],
        slope_rows: vec![],
        phase_rows: vec![],
    };
    for head in 0..512 {
        request.rows.push(HeadFunctionalLpRow {
            coefficients: vec![(128 + head, 1.0)],
            lower: 0.0,
        });
        request.slope_rows.push(Some(request.rows.len()));
        let mut coefficients: Vec<_> = (0..128).map(|i| (i, -0.5)).collect();
        coefficients.push((128 + head, 1.0));
        request.rows.push(HeadFunctionalLpRow {
            coefficients,
            lower: 0.0,
        });
    }
    for _ in 0..4 {
        request.phase_rows.push(request.rows.len());
        request.rows.push(HeadFunctionalLpRow {
            coefficients: (0..128).map(|i| (i, 1.0)).collect(),
            lower: 0.0,
        });
    }
    assert_eq!(request.rows.len(), FUNCTIONAL_MAX_ROWS);
    assert_eq!(
        request
            .rows
            .iter()
            .map(|row| row.coefficients.len())
            .sum::<usize>(),
        FUNCTIONAL_MAX_NONZEROS
    );
    assert!(request.validate());
    let bytes = encode_bounded(&request, FUNCTIONAL_MAX_MESSAGE_BYTES, None).unwrap();
    assert!(bytes.len() <= request.encoded_size_upper_bound().unwrap());
    let estimate = functional_workspace_admission(&request).unwrap();
    assert!(estimate <= FUNCTIONAL_ADVICE_MEMORY_BYTES);
    // An oversized request cannot exploit this estimate by bypassing the
    // structural admission, which is performed before any Model is copied.
    request.bounds.push([-1.0, 1.0]);
    request.objective.push(0.0);
    assert!(functional_float_advice(&request, Instant::now() + LOCAL_SLICE).is_err());
}

#[test]
fn functional_provider_advertises_capability_and_refuses_before_spawn() {
    let provider = HeadPhaseAyAdvice::new(std::env::current_exe().unwrap()).unwrap();
    assert!(provider.supports_functional_advice());
    let request = six_coordinate_request();
    assert!(provider
        .propose_functional(&request, Instant::now())
        .is_none());
    let mut bad = request;
    bad.bounds[0][0] = f64::NEG_INFINITY;
    assert!(provider
        .propose_functional(&bad, Instant::now() + LOCAL_SLICE)
        .is_none());
}

#[cfg(unix)]
fn functional_shell(script: &str) -> Command {
    let mut command = Command::new("/bin/sh");
    command.args(["-c", script]);
    command
}

#[cfg(unix)]
#[test]
fn functional_transport_accepts_its_larger_cap_and_v1_keeps_its_original_cap() {
    let active = AtomicBool::new(false);
    let mut permit = ChildPermit::acquire(&active).unwrap();
    let request = vec![b'x'; MAX_MESSAGE_BYTES + 1];
    let mut v1 = functional_shell("exit 7");
    assert!(exchange(
        &mut v1,
        request.clone(),
        Instant::now() + Duration::from_secs(2),
        &mut permit
    )
    .is_none());
    assert!(permit.cleanup_confirmed, "V1 rejects before spawning");
    let mut v2 = functional_shell("cat >/dev/null; printf '%s' '{\"ok\":true}'");
    assert_eq!(
        exchange_bounded(
            &mut v2,
            request,
            Instant::now() + Duration::from_secs(2),
            &mut permit,
            FUNCTIONAL_MAX_MESSAGE_BYTES,
            FUNCTIONAL_MAX_RESPONSE_BYTES
        )
        .unwrap(),
        b"{\"ok\":true}"
    );
    assert!(permit.cleanup_confirmed);
    drop(permit);
    assert!(!active.load(Ordering::Acquire));
}

#[cfg(unix)]
#[test]
fn functional_expiry_and_response_overflow_reap_before_admission_reopens() {
    let active = AtomicBool::new(false);
    let mut permit = ChildPermit::acquire(&active).unwrap();
    let started = Instant::now();
    let mut blocked = functional_shell("while :; do :; done");
    assert!(exchange_bounded(
        &mut blocked,
        vec![b'x'; FUNCTIONAL_MAX_MESSAGE_BYTES],
        started + Duration::from_millis(30),
        &mut permit,
        FUNCTIONAL_MAX_MESSAGE_BYTES,
        FUNCTIONAL_MAX_RESPONSE_BYTES
    )
    .is_none());
    assert!(permit.cleanup_confirmed);
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(
        active.load(Ordering::Acquire),
        "ownership lasts through pipe joins"
    );
    let mut flooded = functional_shell("while :; do printf '0123456789012345678901234567890123456789012345678901234567890123456789'; done");
    assert!(exchange_bounded(
        &mut flooded,
        Vec::new(),
        Instant::now() + Duration::from_secs(2),
        &mut permit,
        FUNCTIONAL_MAX_MESSAGE_BYTES,
        FUNCTIONAL_MAX_RESPONSE_BYTES
    )
    .is_none());
    assert!(permit.cleanup_confirmed);
    drop(permit);
    assert!(!active.load(Ordering::Acquire));
}

#[test]
fn shared_unconfirmed_cleanup_blocks_both_v1_and_v2_admission() {
    struct UnconfirmedChild;
    impl ChildControl for UnconfirmedChild {
        fn kill(&mut self) -> io::Result<()> {
            Err(io::Error::other("injected V2 kill failure"))
        }
        fn wait(&mut self) -> io::Result<()> {
            Err(io::Error::other("injected V2 unconfirmed reap"))
        }
    }
    let active = AtomicBool::new(false);
    let mut permit = ChildPermit::acquire(&active).unwrap();
    let mut process = ChildReaper::new(UnconfirmedChild, &mut permit);
    assert!(!process.stop());
    drop(process); // shared Drop retries, but cannot confirm this injected exit
    drop(permit);
    assert!(active.load(Ordering::Acquire));
    // V1 and V2 both acquire precisely this same shared permit. Neither
    // protocol may admit another child while its predecessor is unresolved.
    assert!(ChildPermit::acquire(&active).is_none());
    assert!(ChildPermit::acquire(&active).is_none());
}
