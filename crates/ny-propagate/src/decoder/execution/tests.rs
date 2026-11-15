// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent exact-dyadic execution oracle; no host binary32/FMA evaluation.
use super::*;
use crate::decoder::{
    prepare_model_vocab_tree, prepare_vocab_tree, replay_next_token, verify_dense_next_token,
    verify_next_token, AffineHead, VocabOptions,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

fn exact32(bits: u32) -> BigRational {
    let exponent = ((bits >> 23) & 255) as i32;
    assert!(exponent < 255, "oracle finite encoding");
    let fraction = bits & 0x7f_ffff;
    let (significand, power) = if exponent == 0 {
        (fraction, -149)
    } else {
        ((1 << 23) | fraction, exponent - 150)
    };
    let mut numerator = BigInt::from(significand);
    if bits >> 31 != 0 {
        numerator = -numerator;
    }
    if power >= 0 {
        BigRational::from_integer(numerator << power as usize)
    } else {
        BigRational::new(numerator, BigInt::from(1) << (-power) as usize)
    }
}
fn q(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite exact source")
}

/// Binary-search the *exact rational* distances between adjacent encodings.
/// The production checker never uses this costly oracle or host casts.
fn rn32(value: &BigRational) -> Option<u32> {
    let sign = if value.is_negative() { 0x8000_0000 } else { 0 };
    let magnitude = value.abs();
    let maximum = exact32(0x7f7f_ffff);
    if magnitude > maximum {
        let threshold = maximum + BigRational::from_integer(BigInt::from(1) << 103);
        return (magnitude < threshold).then_some(sign | 0x7f7f_ffff);
    }
    let (mut low, mut high) = (0u32, 0x7f7f_ffffu32);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if exact32(middle) <= magnitude {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let lower = exact32(low);
    if lower == magnitude {
        return Some(sign | low);
    }
    let upper_bits = low + 1;
    let lower_distance = &magnitude - lower;
    let upper_distance = exact32(upper_bits) - magnitude;
    let choose_lower =
        lower_distance < upper_distance || (lower_distance == upper_distance && low & 1 == 0);
    Some(sign | if choose_lower { low } else { upper_bits })
}
fn rounded(value: BigRational) -> BigRational {
    exact32(rn32(&value).expect("oracle finite operation"))
}
fn pairwise(values: &[BigRational]) -> BigRational {
    if values.len() == 1 {
        values[0].clone()
    } else {
        let middle = values.len() / 2;
        rounded(pairwise(&values[..middle]) + pairwise(&values[middle..]))
    }
}
fn observed(
    head: AffineHead<'_>,
    point: &[f64],
    id: usize,
    reduction: Binary32Reduction,
) -> BigRational {
    let products: Vec<_> = head
        .row(id)
        .iter()
        .zip(point)
        .map(|(&w, &x)| q(w) * q(x))
        .collect();
    match reduction {
        Binary32Reduction::SequentialSeparate => products
            .into_iter()
            .fold(q(head.bias[id]), |acc, p| rounded(acc + rounded(p))),
        Binary32Reduction::SequentialFma => products
            .into_iter()
            .fold(q(head.bias[id]), |acc, p| rounded(acc + p)),
        Binary32Reduction::BalancedPairwiseSeparate => {
            let mut leaves: Vec<_> = products.into_iter().map(rounded).collect();
            leaves.push(q(head.bias[id]));
            pairwise(&leaves)
        }
        Binary32Reduction::ReadOnceTreeFamily => {
            panic!("family is an enclosure, not one evaluator")
        }
    }
}
const CONCRETE: [Binary32Reduction; 3] = [
    Binary32Reduction::SequentialSeparate,
    Binary32Reduction::BalancedPairwiseSeparate,
    Binary32Reduction::SequentialFma,
];
fn h<'a>(
    weights: &'a [f64],
    bias: &'a [f64],
    dim: usize,
    reduction: Binary32Reduction,
) -> AffineHead<'a> {
    AffineHead {
        vocab_size: bias.len(),
        hidden_size: dim,
        weights,
        bias,
        arithmetic: HeadArithmetic::Binary32 { reduction },
    }
}
fn opts() -> VocabOptions {
    VocabOptions {
        leaf_size: 2,
        ..VocabOptions::default()
    }
}

#[test]
fn decoder_n7_exact_representation_and_nonvacuous_input_domain() {
    for bits in [
        0,
        0x8000_0000,
        1,
        2,
        0x7f_ffff,
        0x80_0000,
        0x3dcc_cccd,
        0x7f7f_ffff,
        0xff7f_ffff,
    ] {
        let value = binary32_value(bits);
        assert_eq!(q(value), exact32(bits));
        assert_eq!(binary32_bits(value), Some(bits));
    }
    assert_eq!(binary32_bits(0.1), None);
    assert_eq!(binary32_bits(ETA32 * 0.5), None);
    assert_eq!(binary32_bits(f64::INFINITY), None);
    let weights = [binary32_value(0x3dcc_cccd), 0.0];
    let bias = [1.0, 0.0];
    let head = h(&weights, &bias, 1, CONCRETE[0]);
    let hidden = HiddenBox {
        lo: &[0.1],
        hi: &[0.2],
    };
    assert!(verify_dense_next_token(head, hidden, &[0.123], 0, &opts())
        .unwrap()
        .is_certified());
    assert!(verify_dense_next_token(
        head,
        HiddenBox {
            lo: &[0.1],
            hi: &[0.1]
        },
        &[0.123],
        0,
        &opts()
    )
    .is_err());
    let poison = [0.1, 0.0];
    assert!(prepare_vocab_tree(h(&poison, &bias, 1, CONCRETE[0]), &[0.0], 0, &opts()).is_err());
    let arithmetic = head.arithmetic;
    assert!(validate_input_domain(
        arithmetic,
        HiddenBox {
            lo: &[-ETA32 * 0.75],
            hi: &[-ETA32 * 0.25]
        }
    )
    .is_err());
    assert!(validate_input_domain(
        arithmetic,
        HiddenBox {
            lo: &[-ETA32 * 1.25],
            hi: &[-ETA32 * 0.75]
        }
    )
    .is_ok());
    assert!(validate_input_domain(
        arithmetic,
        HiddenBox {
            lo: &[-f64::MAX],
            hi: &[-MAX32]
        }
    )
    .is_ok());
    assert!(validate_input_domain(
        arithmetic,
        HiddenBox {
            lo: &[next_up(MAX32)],
            hi: &[f64::MAX]
        }
    )
    .is_err());
}

#[test]
fn decoder_n7_gradual_underflow_and_fma_ties_use_exact_dyadics() {
    assert_eq!(rn32(&(q(ETA32) / BigInt::from(2))), Some(0));
    assert_eq!(
        rn32(&(q(ETA32) * BigRational::new(3.into(), 2.into()))),
        Some(2)
    );
    let weights = [ETA32, 0.0];
    let bias = [ETA32, 0.0];
    let head = h(&weights, &bias, 1, CONCRETE[0]);
    assert_eq!(observed(head, &[0.5], 0, CONCRETE[0]), q(ETA32));
    assert_eq!(observed(head, &[0.5], 0, CONCRETE[2]), q(2.0 * ETA32));
    let bias = [0.0, 0.0];
    let hidden = HiddenBox {
        lo: &[0.5],
        hi: &[0.5],
    };
    let head = h(&weights, &bias, 1, CONCRETE[0]);
    let actual_margin =
        observed(head, &[0.5], 1, CONCRETE[0]) - observed(head, &[0.5], 0, CONCRETE[0]);
    assert!(actual_margin.is_zero(), "rounded target and competitor tie");
    assert!(!verify_next_token(head, hidden, &[0.5], 0, &opts())
        .unwrap()
        .is_certified());
    assert!(q(crate::decoder::numeric::dense_upper(head, hidden, 0, 1).unwrap()) >= actual_margin);
    assert!(binary32_error(0.0, 1).unwrap() >= 3.0 * ETA32);
}

#[test]
fn decoder_n7_cancellation_cannot_close_and_reduction_scopes_differ() {
    let weights = [16_777_216.0, 1.0, -16_777_216.0, 0.0, 0.0, 0.0];
    let bias = [0.0, 0.25];
    let point = [1.0; 3];
    let hidden = HiddenBox {
        lo: &point,
        hi: &point,
    };
    let head = h(&weights, &bias, 3, CONCRETE[0]);
    assert_eq!(observed(head, &point, 0, CONCRETE[0]), q(0.0));
    let checked = verify_next_token(head, hidden, &point, 0, &opts()).unwrap();
    assert!(
        !checked.is_certified(),
        "competitor wins in the specified sequential kernel"
    );
    let mut real = head;
    real.arithmetic = HeadArithmetic::ExactReal;
    assert!(verify_dense_next_token(real, hidden, &point, 0, &opts())
        .unwrap()
        .is_certified());
    let different = h(&weights, &bias, 3, CONCRETE[1]);
    assert!(replay_next_token(different, hidden, &point, 0, &checked.evidence, &opts()).is_err());

    let delta = binary32_value(0x3400_0000); // 2^-23
    let weights = [1.0 + delta, 0.0];
    let bias = [-1.0, -2.0];
    let head = h(&weights, &bias, 1, CONCRETE[0]);
    let point = [1.0 - delta];
    assert!(observed(head, &point, 0, CONCRETE[0]).is_zero());
    assert_eq!(observed(head, &point, 0, CONCRETE[2]), -q(delta) * q(delta));
}

#[test]
fn decoder_n7_partial_overflow_and_resource_caps_refuse_before_certificate() {
    let weights = [MAX32, -MAX32, 0.0, 0.0];
    let bias = [0.0, -1.0];
    let hidden = HiddenBox {
        lo: &[2.0, 2.0],
        hi: &[2.0, 2.0],
    };
    for reduction in CONCRETE
        .into_iter()
        .chain([Binary32Reduction::ReadOnceTreeFamily])
    {
        let head = h(&weights, &bias, 2, reduction);
        assert!(matches!(
            verify_next_token(head, hidden, &[0.0, 0.0], 0, &opts()),
            Err(DecoderRefusal::NonFinite(_))
        ));
    }
    assert!(binary32_error(1.0, 1 << 22).is_err());
    let weights = [0.0; 8];
    let bias = [2.0, 0.0, 0.0, 0.0];
    let head = h(&weights, &bias, 2, CONCRETE[0]);
    let reference = prepare_model_vocab_tree(head, &opts()).unwrap();
    let bytes = reference.reserved_bytes();
    let cap = VocabOptions {
        max_persistent_bytes: bytes as u64 - 1,
        ..opts()
    };
    assert!(prepare_model_vocab_tree(head, &cap).is_err());
    let low_work = VocabOptions {
        max_terms: 1,
        ..opts()
    };
    assert!(prepare_model_vocab_tree(head, &low_work).is_err());
}

#[test]
fn decoder_n7_family_envelope_dominates_exact_execution_at_every_box_vertex() {
    let dim = 3;
    for seed in 0..24 {
        let weights: Vec<_> = (0..15)
            .map(|k| (((k * 13 + seed * 7) % 17) as f64 - 8.0) / 4.0)
            .collect();
        let bias: Vec<_> = (0..5).map(|k| ((k * 3 + seed) % 9) as f64 / 8.0).collect();
        let head = h(&weights, &bias, dim, Binary32Reduction::ReadOnceTreeFamily);
        let lo = [-2.0, -0.5, 0.25];
        let hi = [0.25, 1.5, 2.0];
        let hidden = HiddenBox { lo: &lo, hi: &hi };
        for vertex in 0..8 {
            let point: Vec<_> = (0..dim)
                .map(|k| if vertex & (1 << k) == 0 { lo[k] } else { hi[k] })
                .collect();
            for reduction in CONCRETE {
                for row in 1..5 {
                    let actual = observed(head, &point, row, reduction)
                        - observed(head, &point, 0, reduction);
                    let bound = crate::decoder::numeric::dense_upper(head, hidden, 0, row).unwrap();
                    assert!(
                        actual <= q(bound),
                        "seed={seed} vertex={vertex} row={row} reduction={reduction:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn decoder_n7_model_tree_changing_targets_replays_full_cover_and_execution_error() {
    let mut weights = vec![0.0; 16 * 2];
    weights[0] = 1.0;
    weights[2] = -1.0;
    for id in 2..16 {
        weights[2 * id + 1] = (id % 3) as f64 / 8.0;
    }
    let mut bias = vec![0.0; 16];
    bias[0] = 2.0;
    bias[1] = 2.0;
    let head = h(&weights, &bias, 2, Binary32Reduction::ReadOnceTreeFamily);
    let tree = prepare_model_vocab_tree(head, &opts()).unwrap();
    let mut real = head;
    real.arithmetic = HeadArithmetic::ExactReal;
    assert!(
        tree.reserved_bytes()
            > prepare_model_vocab_tree(real, &opts())
                .unwrap()
                .reserved_bytes(),
        "absolute geometry is admitted for every executed contract"
    );
    for (target, lo, hi) in [
        (0, [0.75, -0.5], [1.0, 0.25]),
        (1, [-1.0, -0.5], [-0.75, 0.25]),
    ] {
        let hidden = HiddenBox { lo: &lo, hi: &hi };
        let anchor = [0.123, -0.375]; // Outside the x box and not binary32.
        let report = tree.verify(hidden, &anchor, target, &opts()).unwrap();
        assert!(report.is_certified());
        assert!(report.stats.cached_geometry_reuses > 0);
        assert!(report.report.evidence.groups.iter().any(|g| g
            .group_bounds
            .as_ref()
            .is_some_and(|b| b.execution_error > 0.0)));
        for vertex in 0..4 {
            let point = [
                if vertex & 1 == 0 { lo[0] } else { hi[0] },
                if vertex & 2 == 0 { lo[1] } else { hi[1] },
            ];
            for reduction in CONCRETE {
                for group in &report.report.evidence.groups {
                    for &id in &group.members {
                        let actual = observed(head, &point, id, reduction)
                            - observed(head, &point, target, reduction);
                        assert!(
                            actual <= q(group.upper),
                            "independent original-row execution exceeds a group summary"
                        );
                    }
                }
            }
        }
        let mut membership = [0; 16];
        for group in &report.report.evidence.groups {
            for &id in &group.members {
                membership[id] += 1;
            }
        }
        assert_eq!(membership[target], 0);
        assert!(membership
            .iter()
            .enumerate()
            .all(|(id, &n)| n == usize::from(id != target)));
        let replay = tree
            .replay(hidden, &anchor, target, &report.report.evidence, &opts())
            .unwrap();
        assert!(replay.is_certified());
        assert!(
            replay.stats.query_terms > 0,
            "independent replay scans are retained"
        );
        let mut poisoned = report.report.evidence.clone();
        let summary = poisoned
            .groups
            .iter_mut()
            .find_map(|g| g.group_bounds.as_mut())
            .unwrap();
        summary.execution_error = 0.0;
        assert!(tree
            .replay(hidden, &anchor, target, &poisoned, &opts())
            .is_err());
        let mut poisoned = report.report.evidence;
        poisoned.groups[0].members.push(target);
        assert!(tree
            .replay(hidden, &anchor, target, &poisoned, &opts())
            .is_err());
    }
    assert_ne!(
        tree.head_identity().assumptions,
        prepare_model_vocab_tree(real, &opts())
            .unwrap()
            .head_identity()
            .assumptions,
        "scope explicitly binds finite binary32 input domain"
    );
}

#[test]
fn decoder_n7_exact_execution_ties_stay_unknown_in_every_contract() {
    let weights = [0.25, -0.5, 0.25, -0.5, 0.0, 0.0];
    let bias = [1.0, 1.0, -1.0];
    let hidden = HiddenBox {
        lo: &[-0.75, 0.25],
        hi: &[0.5, 1.0],
    };
    for reduction in CONCRETE
        .into_iter()
        .chain([Binary32Reduction::ReadOnceTreeFamily])
    {
        let head = h(&weights, &bias, 2, reduction);
        let report = verify_next_token(head, hidden, &[0.123, 0.1], 0, &opts()).unwrap();
        assert!(!report.is_certified());
        assert!(report.unresolved.contains(&1));
        assert!(
            !replay_next_token(head, hidden, &[0.123, 0.1], 0, &report.evidence, &opts())
                .unwrap()
                .is_certified()
        );
    }
}
