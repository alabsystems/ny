// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use super::*;

#[test]
fn owned_storage_keeps_cached_operand_bits_and_original_poll_sequence() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        assert!(rayon::current_thread_index().is_some());
        for nested in [false, true] {
            let _guard = nested.then(crate::faer_parallelism::NestedFaerParGuard::new);
            for (m, k, p, rows) in [(70, 33, 45, 17), (9, 5, 7, 4), (1, 3, 7, 1)] {
                let a = Mat::from_fn(m, k, |i, j| ((i * 7 + j * 13) % 37) as f32 * 0.125 - 2.0);
                let w = Mat::from_fn(k, p, |i, j| ((i * 11 + j * 3) % 29) as f32 * 0.0625 - 0.75);
                let cached = WidenedWeightF64::new(&w).unwrap();
                let mut expected_polls = 0;
                let (expected_a, expected_s, expected_blocks) =
                    aw_f64_with_abssum_chunked_poll_widened(
                        &a,
                        &w,
                        None,
                        rows * k * p,
                        &mut || {
                            expected_polls += 1;
                            Ok(())
                        },
                    )
                    .unwrap()
                    .unwrap();
                let mut actual_polls = 0;
                let (actual_a, actual_s, actual_blocks) =
                    aw_f64_with_abssum_chunked_poll_with_storage(
                        a.as_ref(),
                        w.as_ref(),
                        Some(&cached),
                        rows * k * p,
                        &mut || {
                            actual_polls += 1;
                            Ok(())
                        },
                        &mut OwnedAwStorage,
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(actual_a.mapv(f64::to_bits), expected_a.mapv(f64::to_bits));
                assert_eq!(actual_s.mapv(f64::to_bits), expected_s.mapv(f64::to_bits));
                assert_eq!(actual_blocks, expected_blocks);
                assert_eq!(actual_polls, expected_polls);
            }
        }
    });
}
