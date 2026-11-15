// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use faer::MatMut;

#[test]
fn fixed_f64_views_match_direct_dispatch_bits_layout_and_nested_parallelism() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        assert!(rayon::current_thread_index().is_some());
        for nested in [false, true] {
            let _guard = nested.then(NestedFaerParGuard::new);
            let par = current_par();
            assert_eq!(matches!(par, Par::Seq), !nested);
            for (m, k, n) in [(1, 31, 7), (7, 31, 1), (9, 33, 35), (17, 5, 9), (4, 9, 7)] {
                let a = Mat::from_fn(m, k, |i, j| ((i * 13 + j * 7) % 31) as f64 * 0.125 - 1.0);
                let b = Mat::from_fn(k, n, |i, j| ((i * 3 + j * 11) % 23) as f64 * 0.0625 - 0.75);
                let mut expected = Mat::<f64>::zeros(m, n);
                matmul(&mut expected, Accum::Replace, &a, &b, 1.0, par);
                let actual = mat_mul_f64(&a, &b);
                for j in 0..n {
                    for i in 0..m {
                        assert_eq!(actual[(i, j)].to_bits(), expected[(i, j)].to_bits());
                    }
                }
                // Also preserve the existing into helper's strided output
                // semantics. Both dispatches get the SAME row-major layout.
                let stride = n + 3;
                let sentinel = f64::from_bits(0x7ff8_0000_1234_5678);
                let mut expected = vec![sentinel; m * stride];
                let mut actual = expected.clone();
                // faer 0.24.0's mutable row-major strided constructor
                // returns column-major strides. Reuse the safe composition
                // already used by forward_linear/image.rs; assert layout
                // before either dispatch can touch this destination.
                let expected_view =
                    MatMut::from_column_major_slice_with_stride_mut(&mut expected, n, m, stride)
                        .transpose_mut();
                assert_eq!(expected_view.row_stride(), stride as isize);
                assert_eq!(expected_view.col_stride(), 1);
                matmul(expected_view, Accum::Replace, &a, &b, 1.0, par);
                let actual_view =
                    MatMut::from_column_major_slice_with_stride_mut(&mut actual, n, m, stride)
                        .transpose_mut();
                assert_eq!(actual_view.row_stride(), stride as isize);
                assert_eq!(actual_view.col_stride(), 1);
                mat_mul_f64_views_into(actual_view, a.as_ref(), b.as_ref());
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                for i in 0..m {
                    assert!(actual[i * stride + n..(i + 1) * stride]
                        .iter()
                        .all(|v| v.to_bits() == sentinel.to_bits()));
                }
            }
        }
    });
}
