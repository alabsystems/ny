// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{
    automatic_intermediate_sweep_resource_policy, max_specs_for_gemm_dispatch,
    max_specs_per_dispatch, wide_domain_table_chunk, wide_max_safe_stacked_rows,
    wide_resnet_enabled, wide_safe_domain_count, wide_subgroup_enabled,
    WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
};
use ny_core::GpuCrownLayer;

#[test]
fn automatic_sweep_policy_uses_device_class_and_granted_limits_not_names() {
    let gib = 1024_u64 * 1024 * 1024;
    let discrete = automatic_intermediate_sweep_resource_policy(
        wgpu::DeviceType::DiscreteGpu,
        4 * gib,
        4 * gib,
        8 * gib as usize,
    )
    .expect("capable discrete GPU");
    assert_eq!(discrete.max_device_bytes, 8 * gib as usize);
    assert_eq!(discrete.preferred_rows_per_target, 32);
    assert_eq!(discrete.minimum_rows_per_target, 8);

    let integrated = automatic_intermediate_sweep_resource_policy(
        wgpu::DeviceType::IntegratedGpu,
        128 * 1024 * 1024,
        128 * 1024 * 1024,
        8 * gib as usize,
    )
    .expect("capable integrated GPU");
    assert_eq!(integrated.max_device_bytes, 2 * 1024 * 1024 * 1024);
    assert_eq!(integrated.preferred_rows_per_target, 16);

    let virtual_gpu = automatic_intermediate_sweep_resource_policy(
        wgpu::DeviceType::VirtualGpu,
        512 * 1024 * 1024,
        512 * 1024 * 1024,
        8 * gib as usize,
    )
    .expect("capable virtual GPU");
    assert_eq!(virtual_gpu.max_device_bytes, 1024 * 1024 * 1024);
    assert_eq!(virtual_gpu.preferred_rows_per_target, 8);

    assert!(automatic_intermediate_sweep_resource_policy(
        wgpu::DeviceType::Cpu,
        4 * gib,
        4 * gib,
        8 * gib as usize,
    )
    .is_none());
}

#[test]
fn automatic_sweep_policy_downshifts_with_small_granted_bindings() {
    let policy = automatic_intermediate_sweep_resource_policy(
        wgpu::DeviceType::DiscreteGpu,
        32 * 1024 * 1024,
        32 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
    )
    .expect("small but usable binding tier");
    assert_eq!(policy.preferred_rows_per_target, 32);
    assert_eq!(policy.max_device_bytes, 512 * 1024 * 1024);

    assert!(automatic_intermediate_sweep_resource_policy(
        wgpu::DeviceType::Other,
        512 * 1024,
        512 * 1024,
        8 * 1024 * 1024 * 1024,
    )
    .is_none());
}

/// CPU-buildable pin of the batched-BaB arming diagnosis (#batched-bab,
/// 2026-08-11): WgpuDevice historically inherited ny-core's capability defaults
/// — `provides_deadline_bounded_single_row_resnet_sound() == false` ⇒
/// `deadline_bounded_resnet_sound_max_rows() == 0` (ny-core/src/gemm.rs:2087) —
/// so every K≤8 bounded-rows admission seam
/// (`RootJointDeadlineGpu::from_engine` requires `1..=8`, resnet_decompose's
/// `DeadlineBoundedRows` dispatch requires `capacity >= num_specs`, the
/// active-set / bounded-shared selectors require `2..=8`) refused the prewarmed
/// sound WGPU backend. The honest override must therefore be non-zero, within
/// the audited contract cap, and — since the resident sound fold is row-batched
/// far past 8 rows in the wide lane — exactly the full K=8 cap.
#[test]
fn wgpu_deadline_bounded_resnet_capacity_is_the_full_audited_contract() {
    assert_eq!(
        WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
        ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
        "the honest capacity is the full audited K=8 contract cap"
    );
    // The exact admission windows every consumer enforces (a capacity of 0 —
    // the pre-fix default — fails all of them; the override passes all).
    assert!(
        (1..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS)
            .contains(&WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS),
        "root-joint deadline admission window (sound_gpu_gate.rs:239)"
    );
    assert!(
        (2..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS)
            .contains(&WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS),
        "active-set / bounded-shared K<=8 admission window"
    );
    assert!(
        !(1..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&0usize),
        "the pre-fix default of 0 is refused by every admission seam (the decline point)"
    );
}

/// #batched-bab HOLE-7 SUB-GROUPING must stay DARK.
///
/// This lane changes WHICH kernel produces a verdict-bearing bound on a
/// heterogeneous wave, so a stray default-on (or an accidental
/// `env_gate_default_on` conversion) would arm an unmeasured verdict path in
/// scored runs. Pin the exact opt-in string, and pin that anything else — unset,
/// empty, "0", "true", "yes" — leaves it off (fail closed).
#[test]
fn wide_subgroup_lane_is_dark_unless_explicitly_armed() {
    // Review defect 1: this variable is also written by the DEVICE tests in
    // crown_backward_sound_resident.rs, which hold a DIFFERENT mutex
    // (gpu_test_serial_guard) in the SAME test binary under --features
    // gpu-tests. Take both guards, always in this order, so the two suites
    // cannot cross-write each other's NY_BAB_RESNET_WIDE_SUBGROUP value.
    #[cfg(feature = "gpu-tests")]
    let _gpu_guard = crate::wgpu_device::test_support::gpu_test_serial_guard();
    use ny_test_utils::env::{lock_env, ScopedEnvVar};
    let _env = lock_env();
    {
        let _unset = ScopedEnvVar::unset("NY_BAB_RESNET_WIDE_SUBGROUP");
        assert!(
            !wide_subgroup_enabled(),
            "unset must leave the sub-group lane dark (the scored default)"
        );
    }
    for value in ["", "0", "true", "yes", "TRUE", "on", "2"] {
        let _set = ScopedEnvVar::set("NY_BAB_RESNET_WIDE_SUBGROUP", value);
        assert!(
            !wide_subgroup_enabled(),
            "{value:?} must NOT arm the sub-group lane (only the literal \"1\" does)"
        );
    }
    let _on = ScopedEnvVar::set("NY_BAB_RESNET_WIDE_SUBGROUP", "1");
    assert!(wide_subgroup_enabled(), "\"1\" is the documented opt-in");
}

/// The global wide-kernel gate predates the dark subgroup lane and is an
/// exact-zero kill switch. Central governance must not accidentally turn it
/// into an exact-one opt-in.
#[test]
fn wide_resnet_kernel_preserves_legacy_armed_exact_zero_contract() {
    #[cfg(feature = "gpu-tests")]
    let _gpu_guard = crate::wgpu_device::test_support::gpu_test_serial_guard();
    use ny_test_utils::env::{lock_env, ScopedEnvVar};
    let _env = lock_env();

    {
        let _unset = ScopedEnvVar::unset("NY_BAB_RESNET_WIDE");
        assert!(wide_resnet_enabled(), "unset preserves the shipped ON lane");
    }
    {
        let _off = ScopedEnvVar::set("NY_BAB_RESNET_WIDE", "0");
        assert!(!wide_resnet_enabled(), "exact zero is the kill switch");
    }
    for value in ["1", "true", "", "yes"] {
        let _set = ScopedEnvVar::set("NY_BAB_RESNET_WIDE", value);
        assert!(
            wide_resnet_enabled(),
            "{value:?} must preserve the legacy default-on behavior"
        );
    }
}

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
        cert_err: Default::default(),
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
