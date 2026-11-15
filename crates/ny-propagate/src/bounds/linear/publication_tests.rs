// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::amortise::{NoPoll, WeightedPoll};
use ndarray::s;
use ny_core::NyError;

type Parts = (
    Array2<f32>,
    Array1<f32>,
    Array2<f32>,
    Array1<f32>,
    Array2<f32>,
    Array2<f32>,
);

fn fixture(rows: usize, cols: usize, layout: usize) -> Parts {
    let matrix = |shift: usize| {
        let value = |i: usize, j: usize| ((i * 13 + j * 7 + shift) % 29) as f32 * 0.125 - 1.0;
        match layout {
            0 => Array2::from_shape_fn((rows, cols), |(i, j)| value(i, j)),
            1 => Array2::from_shape_fn((cols, rows), |(j, i)| value(i, j)).reversed_axes(),
            _ => Array2::from_shape_fn((rows * 2, cols * 2), |(i, j)| value(i / 2, j / 2))
                .slice_move(s![..;2, ..;2]),
        }
    };
    let vector = |shift: usize| {
        let value = |i: usize| ((i * 3 + shift) % 17) as f32 * 0.25 - 0.5;
        if layout == 2 {
            Array1::from_shape_fn(rows * 2, |i| value(i / 2)).slice_move(s![..;2])
        } else {
            Array1::from_shape_fn(rows, value)
        }
    };
    (
        matrix(1),
        vector(3),
        matrix(5),
        vector(7),
        matrix(9),
        matrix(11),
    )
}

fn original(p: Parts) -> Result<LinearBounds> {
    LinearBounds::new_or_conservative_with_err(p.0, p.1, p.2, p.3, p.4, p.5)
}

fn polled<P: CooperativeWork>(p: Parts, work: &mut P) -> Result<LinearBounds> {
    LinearBounds::new_or_conservative_with_err_polled_in_place(p.0, p.1, p.2, p.3, p.4, p.5, work)
}

fn bits(b: &LinearBounds) -> (Vec<usize>, Vec<u32>) {
    let mut shape = vec![b.num_outputs(), b.num_inputs()];
    let mut values = Vec::new();
    values.extend(b.lower_a().iter().map(|v| v.to_bits()));
    values.extend(b.lower_b().iter().map(|v| v.to_bits()));
    values.extend(b.upper_a().iter().map(|v| v.to_bits()));
    values.extend(b.upper_b().iter().map(|v| v.to_bits()));
    for error in [b.lower_a_err(), b.upper_a_err()] {
        shape.push(usize::from(error.is_some()));
        if let Some(error) = error {
            shape.extend_from_slice(error.shape());
            values.extend(error.iter().map(|v| v.to_bits()));
        }
    }
    (shape, values)
}

fn same_as_original(p: Parts) -> LinearBounds {
    let expected = original(p.clone()).unwrap();
    let actual = polled(p, &mut NoPoll).unwrap();
    assert_eq!(bits(&actual), bits(&expected));
    actual
}

#[test]
fn private_publication_matches_original_bits_for_empty_and_strided_finite_inputs() {
    for (rows, cols) in [(0, 0), (0, 7), (3, 0), (1, 1), (5, 7), (33, 65)] {
        for layout in 0..3 {
            let _ = same_as_original(fixture(rows, cols, layout));
        }
    }
}

#[test]
fn private_publication_preserves_original_global_firewall_and_infinite_bias_policy() {
    for layout in 0..3 {
        for role in 0..4 {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut p = fixture(3, 5, layout);
                match role {
                    0 => p.0[[1, 2]] = value,
                    1 => p.1[1] = value,
                    2 => p.2[[1, 2]] = value,
                    _ => p.3[1] = value,
                }
                let actual = same_as_original(p);
                if role % 2 == 0 || value.is_nan() {
                    assert!(actual.lower_a().iter().all(|v| v.to_bits() == 0));
                    assert!(actual.upper_a().iter().all(|v| v.to_bits() == 0));
                    assert!(actual.lower_b().iter().all(|&v| v == f32::NEG_INFINITY));
                    assert!(actual.upper_b().iter().all(|&v| v == f32::INFINITY));
                }
                assert!(actual.lower_a_err().is_some() && actual.upper_a_err().is_some());
            }
        }
    }
}

#[test]
fn private_publication_error_sanitization_keeps_signed_zero_and_subnormal_bits() {
    let patterns = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x7f7f_ffff,
        0x8000_0001,
        0xbf80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0xffc0_5678,
    ];
    for layout in 0..3 {
        let mut p = fixture(3, patterns.len(), layout);
        for row in 0..3 {
            for (col, &pattern) in patterns.iter().enumerate() {
                p.4[[row, col]] = f32::from_bits(pattern);
                p.5[[row, col]] = f32::from_bits(patterns[patterns.len() - col - 1]);
            }
        }
        let actual = same_as_original(p);
        let error = actual.lower_a_err().unwrap();
        assert_eq!(error[[0, 1]].to_bits(), 0x8000_0000);
        assert_eq!(error[[0, 2]].to_bits(), 1);
        assert_eq!(error[[0, 3]].to_bits(), 0x007f_ffff);
        for col in 7..patterns.len() {
            assert_eq!(error[[0, col]], f32::INFINITY);
        }
    }
}

#[test]
fn private_publication_malformed_error_planes_degrade_and_drop_both_carriers() {
    for layout in 0..3 {
        for lower_bad in [false, true] {
            let mut p = fixture(3, 5, layout);
            if lower_bad {
                p.4 = Array2::zeros((3, 6));
            } else {
                p.5 = Array2::zeros((2, 5));
            }
            let actual = same_as_original(p);
            assert!(actual.lower_a_err().is_none() && actual.upper_a_err().is_none());
            assert!(actual.lower_a().iter().all(|v| v.to_bits() == 0));
            assert!(actual.upper_a().iter().all(|v| v.to_bits() == 0));
            assert!(actual.lower_b().iter().all(|&v| v == f32::NEG_INFINITY));
            assert!(actual.upper_b().iter().all(|&v| v == f32::INFINITY));
        }
    }
}

#[test]
fn private_publication_base_shape_errors_keep_original_hard_error() {
    for bad in 0..3 {
        let mut p = fixture(3, 5, 0);
        match bad {
            0 => p.1 = Array1::zeros(4),
            1 => p.2 = Array2::zeros((3, 4)),
            _ => p.3 = Array1::zeros(2),
        }
        let expected = original(p.clone()).unwrap_err();
        let actual = polled(p, &mut NoPoll).unwrap_err();
        assert_eq!(actual.to_string(), expected.to_string());
        assert!(matches!(
            actual,
            NyError::InvalidSpec(_) | NyError::ShapeMismatch { .. }
        ));
    }
}

#[derive(Default)]
struct Trace {
    calls: Vec<Option<usize>>,
    fail_at: Option<usize>,
}
impl Trace {
    fn call(&mut self, units: Option<usize>) -> Result<()> {
        let index = self.calls.len();
        self.calls.push(units);
        if self.fail_at == Some(index) {
            Err(NyError::DeadlineExceeded(
                "publication original-authority refusal".into(),
            ))
        } else {
            Ok(())
        }
    }
}
impl CooperativeWork for Trace {
    fn checkpoint(&mut self) -> Result<()> {
        self.call(None)
    }
    fn before(&mut self, units: usize) -> Result<()> {
        assert!(units <= POLL_CHUNK_UNITS);
        self.call(Some(units))
    }
}

#[test]
fn private_publication_every_refusal_including_after_partial_repair_blocks_return() {
    for repair in [false, true] {
        let mut p = fixture(3, 5, 0);
        if repair {
            p.0[[0, 0]] = f32::NAN;
        }
        let mut success = Trace::default();
        let output = polled(p.clone(), &mut success).unwrap();
        assert_eq!(bits(&output), bits(&original(p.clone()).unwrap()));
        assert_eq!(success.calls.last(), Some(&None));
        for index in 0..success.calls.len() {
            let mut fail = Trace {
                fail_at: Some(index),
                ..Trace::default()
            };
            let result = polled(p.clone(), &mut fail);
            assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
            assert_eq!(fail.calls, success.calls[..=index]);
        }
    }
}

#[test]
fn private_publication_declared_scan_and_sanitize_units_have_no_unpolled_tail() {
    let rows = 32;
    let cols = 40;
    let mut delivered = Vec::new();
    let mut poll = WeightedPoll::new(|units| {
        delivered.push(units);
        Ok(())
    });
    let p = fixture(rows, cols, 0);
    let expected = original(p.clone()).unwrap();
    let actual = polled(p, &mut poll).unwrap();
    assert_eq!(bits(&actual), bits(&expected));
    assert!(delivered.iter().all(|&n| n <= POLL_CHUNK_UNITS));
    // Two coefficient scans, two bias scans, and two error sanitizations.
    assert_eq!(
        delivered.iter().sum::<usize>(),
        64 + 8 * (4 * rows * cols + 2 * rows)
    );
    assert!(delivered.last().is_some_and(|&units| units > 0));
}

#[test]
fn private_publication_keeps_original_payload_owners_offsets_and_spare_capacities() {
    for repair in [false, true] {
        let rows = 3;
        let cols = 5;
        let matrix = || {
            let mut raw = Vec::with_capacity((rows + 2) * cols + 31);
            raw.resize((rows + 2) * cols, 0.25_f32);
            let capacity = raw.capacity();
            let array = Array2::from_shape_vec((rows + 2, cols), raw)
                .unwrap()
                .slice_move(s![1..rows + 1, ..]);
            let ptr = array.as_ptr() as usize;
            (array, (ptr, capacity, Some(cols)))
        };
        let vector = || {
            let mut raw = Vec::with_capacity(rows + 19);
            raw.resize(rows + 2, -0.0_f32);
            let capacity = raw.capacity();
            let array = Array1::from_vec(raw).slice_move(s![1..rows + 1]);
            let ptr = array.as_ptr() as usize;
            (array, (ptr, capacity, Some(1)))
        };
        let (mut la, la_id) = matrix();
        let (lb, lb_id) = vector();
        let (ua, ua_id) = matrix();
        let (ub, ub_id) = vector();
        let (le, le_id) = matrix();
        let (ue, ue_id) = matrix();
        if repair {
            la[[0, 0]] = f32::NAN;
        }
        let p = (la, lb, ua, ub, le, ue);
        let expected = original(p.clone()).unwrap();
        let actual = polled(p, &mut NoPoll).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        let LinearBounds {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err,
            upper_a_err,
        } = actual;
        fn identity<D: Dimension>(array: Array<f32, D>) -> (usize, usize, Option<usize>) {
            let ptr = array.as_ptr() as usize;
            let (raw, offset) = array.into_raw_vec_and_offset();
            (ptr, raw.capacity(), offset)
        }
        assert_eq!(identity(lower_a), la_id);
        assert_eq!(identity(lower_b), lb_id);
        assert_eq!(identity(upper_a), ua_id);
        assert_eq!(identity(upper_b), ub_id);
        assert_eq!(identity(lower_a_err.unwrap()), le_id);
        assert_eq!(identity(upper_a_err.unwrap()), ue_id);
    }
}
