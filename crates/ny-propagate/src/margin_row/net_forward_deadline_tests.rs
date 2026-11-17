// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn fixture(columns: usize) -> (ConvOp, Array2<f32>) {
    let eps = 2f32.powi(-23);
    let c = compile_conv(
        0,
        &[
            1.0,
            f64::from(1.0 + eps),
            0.0,
            -1.0,
            0.5,
            2.0,
            0.0,
            -2.0,
            -1.0,
        ],
        &[0.0; 3],
        &[0.0; 3],
        0.0,
        (3, 1, 1, 3),
        (1, 1),
        (0, 1, 0, 1),
        (1, 1, 3),
        (3, 1, 3),
    );
    let src = Array2::from_shape_fn((3, columns), |(row, col)| match col % 5 {
        0 => [-1.0, 1.0 - eps, 0.0][row],
        1 => [f32::from_bits(1), -f32::from_bits(1), -0.0][row],
        _ => (row as f32 - col as f32 % 19.0) / 16.0,
    });
    (c, src)
}

#[test]
fn checked_forward_preserves_mask_blocks_padding_signed_d_and_nonfusion() {
    // Each mask block exceeds the polling tile. A changed mask stride would
    // silently compute or skip the wrong original-frame columns.
    let columns = 40_003;
    let (c, src) = fixture(columns);
    let src64 = src.mapv(f64::from);
    let input = [u64::MAX; 3];
    let output = [1, 2, 3, 1, 2, 3, 1, 2, 3];
    for masked in [false, true] {
        let mask = masked.then_some((&input[..], &output[..], 2));
        for abs_w in [false, true] {
            let mut old = Array2::from_elem((9, columns), 9.25f32);
            let mut timed = old.clone();
            let mut wide = old.mapv(f64::from);
            conv_forward_f32_native(&c, &src, &mut old, abs_w, mask);
            conv_forward_blocked_f32(&c, &src64, &mut wide, abs_w, mask);
            conv_forward_f32_native_checked(
                &c,
                &src,
                &mut timed,
                abs_w,
                mask,
                Some(Instant::now() + Duration::from_secs(60)),
            )
            .unwrap();
            for ((a, b), w) in old.iter().zip(&timed).zip(&wide) {
                assert_eq!(a.to_bits(), b.to_bits());
                assert_eq!(f64::from(*b).to_bits(), w.to_bits());
            }
            for sp in 0..3 {
                for oc in 0..3 {
                    for col in [0, 1, 16_383, 16_384, 20_001, 20_002, 32_768, columns - 1] {
                        let block = col / columns.div_ceil(2);
                        let mut expected = 9.25f32;
                        if !masked || output[sp] & (1 << block) != 0 {
                            expected = 0.0;
                            for tap in 0..3 {
                                let source = sp as isize + tap as isize - 1;
                                if !(0..3).contains(&source) {
                                    continue;
                                }
                                let w = c.wmat[oc * 3 + tap] as f32;
                                let w = if abs_w { w.abs() } else { w };
                                if w != 0.0 {
                                    expected += w * src[[source as usize, col]];
                                }
                            }
                        }
                        assert_eq!(timed[[oc * 3 + sp, col]].to_bits(), expected.to_bits());
                    }
                }
            }
            if !masked && !abs_w {
                assert_eq!(timed[[1, 0]].to_bits(), 0.0f32.to_bits());
                assert_ne!(
                    (1.0 + 2f32.powi(-23))
                        .mul_add(1.0 - 2f32.powi(-23), -1.0)
                        .to_bits(),
                    0.0f32.to_bits()
                );
            }
        }
    }
}

#[test]
fn checked_forward_expired_entry_preserves_destination() {
    let (c, src) = fixture(11);
    let mut dst = Array2::from_elem((9, 11), 123.5f32);
    let result =
        conv_forward_f32_native_checked(&c, &src, &mut dst, false, None, Some(Instant::now()));
    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
    assert!(dst.iter().all(|v| v.to_bits() == 123.5f32.to_bits()));
}

#[test]
fn checked_forward_mid_contraction_cancellation_joins_workers() {
    let (c, src) = fixture(40_003);
    let (blk, nblk) = conv_f32_blocking(None, src.ncols(), c.kernel.0);
    let mut dst = Array2::from_elem((9, src.ncols()), 123.5f32);
    let calls = AtomicUsize::new(0);
    let checkpoint = || {
        if calls.fetch_add(1, Ordering::SeqCst) >= 50 {
            Err(NyError::DeadlineExceeded(
                "injected inside forward contraction".into(),
            ))
        } else {
            Ok(())
        }
    };
    // One worker makes the injected checkpoint deterministic. Cancellation is
    // in the core, after work starts, without racing the wall clock.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let result = pool.install(|| {
        conv_f32_core_checked::<f32, true, true>(
            &c,
            src.as_slice().unwrap(),
            src.ncols(),
            &mut dst,
            false,
            None,
            blk,
            nblk,
            Some(Instant::now() + Duration::from_secs(60)),
            Some(&checkpoint),
        )
    });
    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
    assert!(
        dst.iter().any(|v| v.to_bits() != 123.5f32.to_bits()),
        "work must precede cancellation"
    );
    assert!(
        dst.iter().any(|v| v.to_bits() == 123.5f32.to_bits()),
        "partial result must be discarded"
    );
    let joined = calls.load(Ordering::SeqCst);
    pool.install(|| {});
    assert_eq!(calls.load(Ordering::SeqCst), joined);
}
