// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::head_functional_advice::HeadFunctionalLpRow;
use ny_core::head_phase_cover_advice::{
    PHASE_COVER_MAX_LEAVES, PHASE_COVER_MAX_ROWS, PHASE_COVER_MAX_VARIABLES,
};

fn two_leaf_request() -> PhaseCoverLpRequest {
    PhaseCoverLpRequest {
        protocol_version: PHASE_COVER_PROTOCOL_VERSION,
        selected_heads: 1,
        bounds: vec![[0.0, 1.0]],
        objective: vec![1.0],
        rows: vec![HeadFunctionalLpRow {
            coefficients: vec![(0, 1.0)],
            lower: 0.0,
        }],
        leaves: vec![
            PhaseCoverLeaf {
                mask: 0,
                rows: vec![HeadFunctionalLpRow {
                    coefficients: vec![(0, -1.0)],
                    lower: -0.25,
                }],
            },
            PhaseCoverLeaf {
                mask: 1,
                rows: vec![HeadFunctionalLpRow {
                    coefficients: vec![(0, 1.0)],
                    lower: 0.5,
                }],
            },
        ],
    }
}

fn response(request: &PhaseCoverLpRequest) -> Vec<u8> {
    encode_bounded(
        &PhaseCoverHelperResponse {
            protocol_version: PHASE_COVER_PROTOCOL_VERSION,
            ay_revision: AY_REVISION.into(),
            hint: PhaseCoverLpHint {
                leaves: request
                    .leaves
                    .iter()
                    .map(|leaf| PhaseCoverLeafHint {
                        mask: leaf.mask,
                        row_duals: vec![1.0; request.rows.len() + leaf.rows.len()],
                    })
                    .collect(),
            },
        },
        PHASE_COVER_MAX_RESPONSE_BYTES,
        None,
    )
    .unwrap()
}

#[test]
fn actual_cold_batch_hints_replay_against_each_original_leaf() {
    let request = two_leaf_request();
    let hint =
        phase_cover_float_advice(&request, Instant::now() + Duration::from_secs(10)).unwrap();
    assert!(hint.validate(&request));
    assert_eq!(hint.leaves.len(), 2);
    for h in &hint.leaves {
        let leaf = &request.leaves[h.mask];
        let mut best = f64::NEG_INFINITY;
        for sign in [1.0, -1.0] {
            let mut constant = 0.0;
            let mut coefficient = request.objective[0];
            for (row, &raw) in request.rows.iter().chain(&leaf.rows).zip(&h.row_duals) {
                let dual = (sign * raw).max(0.0);
                constant += dual * row.lower;
                coefficient -= dual * row.coefficients[0].1;
            }
            let lower = constant + coefficient.min(0.0);
            let exact = if h.mask == 0 { 0.0 } else { 0.5 };
            assert!(
                lower <= exact,
                "raw AY multipliers cannot overrule authored leaf bounds"
            );
            best = best.max(lower);
        }
        if h.mask == 1 {
            assert!(
                best > 0.25,
                "actual AY advice must improve the scalar floor"
            );
        }
    }
    assert!(include_str!("../Cargo.toml").contains(AY_REVISION));
}

#[test]
fn infeasible_leaf_supplies_no_hint_and_no_infeasibility_authority() {
    let mut request = two_leaf_request();
    request.leaves[0].rows[0] = HeadFunctionalLpRow {
        coefficients: vec![(0, 1.0)],
        lower: 2.0,
    };
    let hint =
        phase_cover_float_advice(&request, Instant::now() + Duration::from_secs(10)).unwrap();
    assert!(hint.validate(&request));
    assert!(hint.leaves.iter().any(|h| h.mask == 1));
    assert!(
        hint.leaves.iter().all(|h| h.mask != 0),
        "AY infeasibility is omission only, never a status or floor"
    );
    let encoded = serde_json::to_string(&hint).unwrap();
    assert!(
        !encoded.contains("infeasible")
            && !encoded.contains("status")
            && !encoded.contains("lower")
    );
}

#[test]
fn actual_ay_batch_closes_all_128_cycle_leaves_with_exact_unselected_positive_replay() {
    use num_rational::BigRational;
    use num_traits::Zero;
    fn row(mut coefficients: Vec<(usize, f64)>) -> HeadFunctionalLpRow {
        coefficients.sort_by_key(|&(i, _)| i);
        HeadFunctionalLpRow {
            coefficients,
            lower: 0.0,
        }
    }
    let mut request = PhaseCoverLpRequest {
        protocol_version: PHASE_COVER_PROTOCOL_VERSION,
        selected_heads: 7,
        bounds: [vec![[-1.0, 1.0]; 7], vec![[0.0, 2.0]; 14]].concat(),
        objective: [vec![0.0; 7], vec![-1.0; 7], vec![0.25; 7]].concat(),
        rows: Vec::new(),
        leaves: Vec::new(),
    };
    for head in 0..14 {
        let j = head % 7;
        request.rows.push(row(vec![(7 + head, 1.0)]));
        request
            .rows
            .push(row(vec![(j, -1.0), ((j + 1) % 7, 1.0), (7 + head, 1.0)]));
    }
    for mask in 0..128 {
        let mut rows = Vec::new();
        for j in 0..7 {
            if mask & (1 << j) != 0 {
                rows.push(row(vec![(j, 1.0), ((j + 1) % 7, -1.0)]));
                rows.push(row(vec![(j, 1.0), ((j + 1) % 7, -1.0), (7 + j, -1.0)]));
            } else {
                rows.push(row(vec![(j, -1.0), ((j + 1) % 7, 1.0)]));
                rows.push(row(vec![(7 + j, -1.0)]));
            }
        }
        request.leaves.push(PhaseCoverLeaf { mask, rows });
    }
    assert!(request.validate());
    // This tests actual pinned AY, not a synthetic provider. Production
    // supervision still keeps its separate <=750ms whole-batch ceiling.
    let hint =
        phase_cover_float_advice(&request, Instant::now() + Duration::from_secs(10)).unwrap();
    assert_eq!(
        hint.leaves.len(),
        128,
        "all closed masks are feasible (constant x belongs to each)"
    );
    let q = |x| BigRational::from_float(x).unwrap();
    let mut cover_floor = None::<BigRational>;
    for h in hint.leaves {
        let leaf = &request.leaves[h.mask];
        let mut best = None::<BigRational>;
        for sign in [1.0, -1.0] {
            let mut lower = q(4.75);
            let mut residual: Vec<_> = request.objective.iter().map(|&x| q(x)).collect();
            for (row, &raw) in request.rows.iter().chain(&leaf.rows).zip(&h.row_duals) {
                let multiplier = q((sign * raw).max(0.0));
                lower += &multiplier * q(row.lower);
                for &(i, a) in &row.coefficients {
                    residual[i] -= &multiplier * q(a);
                }
            }
            for (coefficient, &[lo, hi]) in residual.into_iter().zip(&request.bounds) {
                lower += (&coefficient * q(lo)).min(coefficient * q(hi));
            }
            best = Some(match best {
                Some(old) => old.max(lower),
                None => lower,
            });
        }
        let best = best.unwrap();
        cover_floor = Some(match cover_floor {
            Some(old) => old.min(best),
            None => best,
        });
    }
    let floor = cover_floor.unwrap();
    assert!(
        floor > BigRational::zero(),
        "AY multipliers must recover the positive duplicate coupling"
    );
    assert!(
        floor <= q(0.25),
        "alternating endpoints attain the original exact minimum"
    );
    assert!(
        floor - q(0.5) <= q(-0.25),
        "the same hint cannot certify the changed unsafe bias"
    );
}

#[test]
fn v3_wire_rejects_pin_version_authority_duplicate_and_wrong_leaf_shapes() {
    let request = two_leaf_request();
    let bytes = response(&request);
    assert_eq!(
        decode_phase_cover_hint(&bytes, &request)
            .unwrap()
            .leaves
            .len(),
        2
    );
    assert!(decode_phase_cover_hint(b"{}", &request).is_none());
    for fault in 0..7 {
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        match fault {
            0 => value["protocol_version"] = serde_json::json!(FUNCTIONAL_PROTOCOL_VERSION),
            1 => value["ay_revision"] = serde_json::json!("not-the-pin"),
            2 => value["hint"]["status"] = serde_json::json!("infeasible"),
            3 => value["hint"]["leaves"][1]["mask"] = serde_json::json!(0),
            4 => value["hint"]["leaves"][0]["row_duals"] = serde_json::json!([1.0]),
            5 => value["hint"]["leaves"][0]["row_duals"] = serde_json::json!([1e101, 0.0]),
            _ => value["hint"]["leaves"][0]["mask"] = serde_json::json!(128),
        }
        assert!(
            decode_phase_cover_hint(&serde_json::to_vec(&value).unwrap(), &request).is_none(),
            "fault={fault}"
        );
    }
    assert!(
        decode_phase_cover_hint(&vec![b' '; PHASE_COVER_MAX_RESPONSE_BYTES + 1], &request)
            .is_none()
    );
    let mut subset: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    subset["hint"]["leaves"].as_array_mut().unwrap().pop();
    assert_eq!(
        decode_phase_cover_hint(&serde_json::to_vec(&subset).unwrap(), &request)
            .unwrap()
            .leaves
            .len(),
        1
    );
}

#[test]
fn full_128_leaf_dense_response_exceeds_legacy_cap_and_stays_v3_bounded() {
    let mut request = two_leaf_request();
    request.selected_heads = 7;
    request.bounds = vec![[-1.0, 1.0]; PHASE_COVER_MAX_VARIABLES];
    request.objective = vec![0.0; PHASE_COVER_MAX_VARIABLES];
    request.rows = vec![
        HeadFunctionalLpRow {
            coefficients: vec![],
            lower: 0.0
        };
        PHASE_COVER_MAX_ROWS
    ];
    request.leaves = (0..PHASE_COVER_MAX_LEAVES)
        .map(|mask| PhaseCoverLeaf { mask, rows: vec![] })
        .collect();
    assert!(request.validate());
    assert!(phase_cover_workspace_admission(&request).is_some());
    let bytes = response(&request);
    assert!(bytes.len() > FUNCTIONAL_MAX_RESPONSE_BYTES);
    assert!(bytes.len() < PHASE_COVER_MAX_RESPONSE_BYTES);
    assert!(decode_phase_cover_hint(&bytes, &request).is_some());
    assert!(encode_bounded(&request, 16, None).is_err());
    assert!(encode_bounded(
        &request,
        PHASE_COVER_MAX_MESSAGE_BYTES,
        Some(Instant::now())
    )
    .is_err());
}

#[test]
fn v3_advertises_capability_and_refuses_invalid_or_expired_before_spawn() {
    let provider = HeadPhaseAyAdvice::new(std::env::current_exe().unwrap()).unwrap();
    assert!(provider.supports_phase_cover_advice());
    let request = two_leaf_request();
    assert!(provider
        .propose_phase_cover(&request, Instant::now())
        .is_none());
    let mut invalid = request;
    invalid.leaves[1].mask = 0;
    assert!(provider
        .propose_phase_cover(&invalid, Instant::now() + PHASE_COVER_LOCAL_SLICE)
        .is_none());
    assert!(phase_cover_float_advice(&invalid, Instant::now() + PHASE_COVER_LOCAL_SLICE).is_err());
}

#[cfg(unix)]
#[test]
fn v3_expiry_overflow_and_legacy_caps_share_confirmed_reap() {
    let active = AtomicBool::new(false);
    let mut permit = ChildPermit::acquire(&active).unwrap();
    let mut legacy = Command::new("/bin/sh");
    legacy.args(["-c", "exit 7"]);
    let oversized_legacy = vec![b'x'; FUNCTIONAL_MAX_MESSAGE_BYTES + 1];
    assert!(exchange_bounded(
        &mut legacy,
        oversized_legacy.clone(),
        Instant::now() + Duration::from_secs(2),
        &mut permit,
        FUNCTIONAL_MAX_MESSAGE_BYTES,
        FUNCTIONAL_MAX_RESPONSE_BYTES
    )
    .is_none());
    assert!(permit.cleanup_confirmed);
    let mut cover = Command::new("/bin/sh");
    cover.args(["-c", "cat >/dev/null; printf ok"]);
    assert_eq!(
        exchange_bounded(
            &mut cover,
            oversized_legacy,
            Instant::now() + Duration::from_secs(2),
            &mut permit,
            PHASE_COVER_MAX_MESSAGE_BYTES,
            PHASE_COVER_MAX_RESPONSE_BYTES
        )
        .unwrap(),
        b"ok"
    );
    let mut blocked = Command::new("/bin/sh");
    blocked.args(["-c", "while :; do :; done"]);
    assert!(exchange_bounded(
        &mut blocked,
        vec![b'x'; 4096],
        Instant::now() + Duration::from_millis(30),
        &mut permit,
        PHASE_COVER_MAX_MESSAGE_BYTES,
        PHASE_COVER_MAX_RESPONSE_BYTES
    )
    .is_none());
    assert!(permit.cleanup_confirmed);
    let mut flood = Command::new("/bin/sh");
    flood.args([
        "-c",
        "while :; do printf '0123456789012345678901234567890123456789'; done",
    ]);
    assert!(exchange_bounded(
        &mut flood,
        Vec::new(),
        Instant::now() + Duration::from_secs(2),
        &mut permit,
        PHASE_COVER_MAX_MESSAGE_BYTES,
        512
    )
    .is_none());
    assert!(permit.cleanup_confirmed);
    drop(permit);
    assert!(!active.load(Ordering::Acquire));
}
