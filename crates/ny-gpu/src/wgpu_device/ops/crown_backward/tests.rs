// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{
    max_specs_for_gemm_dispatch, max_specs_per_dispatch, wide_domain_table_chunk,
    wide_max_safe_stacked_rows, wide_safe_domain_count,
};
use ny_core::GpuCrownLayer;

#[test]
fn test_max_specs_for_gemm_dispatch_small_k_conv_keeps_full_soundnessbench_batch_3599() {
    let limit = max_specs_for_gemm_dispatch(24, 24, 4_096, 384);
    assert_eq!(
        limit, 384,
        "small-K conv GEMM itself should fit the full batch"
    );
}

#[test]
fn test_max_specs_per_dispatch_caps_soundnessbench_conv_workgroups_3599() {
    let layers = vec![GpuCrownLayer::Conv2d {
        weight_col: vec![0.0; 24 * 24].into(),
        bias_expanded: None,
        out_channels: 24,
        in_channels: 24,
        kernel_h: 1,
        kernel_w: 1,
        out_h: 64,
        out_w: 64,
        in_h: 64,
        in_w: 64,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    }];

    let batch_limit = max_specs_per_dispatch(&layers, 384);
    assert_eq!(
        batch_limit, 170,
        "conv reshape/col2im must force spec batching"
    );
}

#[test]
fn test_max_specs_per_dispatch_keeps_maxpool_batch_uncapped_4211() {
    let layers = vec![GpuCrownLayer::MaxPool2d {
        routing: vec![0, 3, u32::MAX, 7],
        ibp_lower: vec![0.1, 0.2, -0.3, 0.4],
        ibp_upper: vec![0.5, 0.6, 0.7, 0.8],
        input_dim: 16,
        output_dim: 4,
    }];

    let batch_limit = max_specs_per_dispatch(&layers, 384);
    assert_eq!(
        batch_limit, 384,
        "maxpool backward uses one workgroup per spec row and should not reduce the batch"
    );
}

#[test]
fn test_wide_domain_limit_rejects_an_oversized_single_domain() {
    // width=2049 needs nine 256-thread workgroups per row, exceeding max_wg=8.
    assert_eq!(wide_max_safe_stacked_rows(8, 2_049), 0);
    assert_eq!(wide_safe_domain_count(8, 2_049, 1), None);

    // Even when individual rows fit, all rows belonging to one domain must fit
    // together; the wrapper cannot split a domain's relaxation block.
    assert_eq!(wide_max_safe_stacked_rows(8, 512), 4);
    assert_eq!(wide_safe_domain_count(8, 512, 5), None);
    assert_eq!(wide_safe_domain_count(8, 512, 2), Some(2));
}

#[test]
fn test_wide_domain_auxiliary_table_tracks_each_subchunk() {
    let table = [10, 20, 30, 40, 50];
    assert_eq!(
        wide_domain_table_chunk(&table, table.len(), 2, 4),
        Some(&table[2..4])
    );
    assert_eq!(wide_domain_table_chunk(&table, 6, 2, 4), None);

    let empty: [u8; 0] = [];
    assert_eq!(wide_domain_table_chunk(&empty, 5, 4, 5), Some(&[][..]));
}
