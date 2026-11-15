// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Real binary32 allocations and independent rational conversion controls.
//! Raw conversion fixtures below test arithmetic/storage only; integrated
//! row proof controls obtain their scope through the real original builder.

use super::*;

#[test]
fn input_rows_storage_native_owner_layout_observation() {
    assert!(INPUT_ROWS_OWNER_BYTES >= size_of::<InputRowsF32Transfer<'static>>() + 256);
    println!(
        "{{\"phase\":\"input_rows_layout\",\"rows_bytes\":{},\"storage_bytes\":{},\"pair_bytes\":{},\"transfer_bytes\":{},\"owner_envelope_bytes\":{},\"f32_transfer_fixed_bytes\":{}}}",
        size_of::<Rows>(),
        size_of::<Storage>(),
        size_of::<InputRowPair<'static>>(),
        size_of::<InputRowsF32Transfer<'static>>(),
        INPUT_ROWS_OWNER_BYTES,
        F32_TRANSFER_FIXED_BYTES,
    );
}

use num_bigint::BigInt;
use num_traits::Signed;

fn q32(value: f32) -> Q {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 255;
    assert_ne!(exponent, 255);
    let fraction = bits & 0x007f_ffff;
    let (significand, power) = if exponent == 0 {
        (fraction, -149)
    } else {
        ((1 << 23) | fraction, exponent as i32 - 127 - 23)
    };
    let integer = BigInt::from(significand) * if bits >> 31 == 0 { 1 } else { -1 };
    if power >= 0 {
        Q::from_integer(integer << power as usize)
    } else {
        Q::new(integer, BigInt::from(1) << (-power) as usize)
    }
}

fn conversion_admission(cells: usize) -> StorageAdmission {
    let bytes = cells * size_of::<f32>();
    StorageAdmission {
        peak: bytes + 4096,
        remaining: bytes,
        allocated: 0,
        budget: bytes + 8192,
        changed: 0,
        original_budget: None,
    }
}

#[test]
fn input_rows_storage_signed_gap_and_subtraction_residual_cover_exact_rational_difference() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    let tiny = f64::from_bits((1023_u64 - 100) << 52);
    for (source, stored) in [
        (-tiny, 1.0_f32),
        (tiny, -1.0_f32),
        (1.0, -1.0_f32),
        (-1.0, 1.0_f32),
        (f64::from_bits(1), 0.0_f32),
        (-f64::from_bits(1), -0.0_f32),
        (0.0, f32::from_bits(1)),
        (0.0, f32::from_bits(0x8000_0001)),
    ] {
        let gap = f32_storage_gap(source, stored).unwrap();
        let exact = (q(source) - q32(stored)).abs();
        assert!(q(gap) >= exact);
    }
    let (head, residual) = two_sum(-tiny, -1.0);
    assert_eq!(head, -1.0);
    assert_eq!(residual, -tiny);
    let exact = (q(-tiny) - q32(1.0)).abs();
    assert!(
        q(head.abs()) < exact,
        "dropping the subtraction residual must fail"
    );
    assert!(q(f32_storage_gap(-tiny, 1.0).unwrap()) >= exact);
}

#[test]
fn input_rows_storage_real_f32_arrays_reuse_f64_errors_and_enclose_signed_source_centers() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    let min32 = f64::from_bits((1023_u64 - 149) << 52);
    let values = [
        0.0,
        -0.0,
        1.0 + 2.0_f64.powi(-25),
        -1.0 - 2.0_f64.powi(-25),
        min32,
        -min32,
        min32 / 2.0,
        -min32 / 2.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from(f32::MAX),
        -f64::from(f32::MAX),
    ];
    let errors: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, _)| if i % 3 == 0 { 0.0 } else { 1e-300 })
        .collect();
    let a = Array2::from_shape_vec((1, values.len()), values.to_vec()).unwrap();
    let e = Array2::from_shape_vec((1, values.len()), errors.clone()).unwrap();
    let rows = seal_rows(a, e, values.to_vec(), errors.clone(), deadline()).unwrap();
    let error_pointer = rows.e.as_ptr();
    let bias_error_pointer = rows.eb.as_ptr();
    let mut paid = conversion_admission(2 * values.len());
    let stored = store_rows_f32(rows, &mut paid, deadline()).unwrap();
    assert_eq!(stored.e.as_ptr(), error_pointer);
    assert_eq!(stored.eb.as_ptr(), bias_error_pointer);
    let Coefficients::F32(ref coefficients) = stored.a else {
        panic!("actual contiguous binary32 coefficient buffer required")
    };
    let Biases::F32(ref biases) = stored.b else {
        panic!("actual Vec<f32> required")
    };
    assert_eq!(coefficients.dim(), (1, values.len()));
    assert_eq!(coefficients[[0, 1]].to_bits(), (-0.0_f32).to_bits());
    for (i, &source) in values.iter().enumerate() {
        let cast = source as f32;
        assert_eq!(coefficients[[0, i]].to_bits(), cast.to_bits());
        assert_eq!(biases[i].to_bits(), cast.to_bits());
        let exact_error = q(errors[i]) + (q(source) - q32(cast)).abs();
        assert!(q(stored.e[[0, i]]) >= exact_error);
        assert!(q(stored.eb[i]) >= exact_error);
        assert_eq!(q(stored.a.value(0, i)), q32(cast));
        assert_eq!(q(stored.b.value(i)), q32(cast));
    }
    assert_eq!(paid.remaining, 0);
    assert_eq!(paid.allocated, 2 * values.len() * size_of::<f32>());
    assert!(paid.changed > 0);
}

#[test]
fn input_rows_storage_binary32_subnormal_views_widen_bits_exactly() {
    let coefficients = Coefficients::F32(
        ContiguousMatrix::from_array(ndarray::arr2(&[[
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::from_bits(0x007f_ffff),
            -0.0_f32,
        ]]))
        .unwrap(),
    );
    let biases = Biases::F32(vec![
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
        -0.0_f32,
    ]);
    let Coefficients::F32(ref array) = coefficients else {
        unreachable!()
    };
    for i in 0..4 {
        assert_eq!(q(coefficients.value(0, i)), q32(array[[0, i]]));
        assert_eq!(q(biases.value(i)), q32(array[[0, i]]));
    }
    assert_eq!(coefficients.value(0, 3).to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn input_rows_storage_authenticated_patch_transfer_preserves_scope_mapping_and_rational_proof() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(true);
    let root = root(&net, &[-0.75; 25], &[1.0; 25], None);
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [24, 0, 12];
    let at = deadline();
    let original = products(&plan, &net, &root, &targets, at).rows;
    assert_eq!(original.backing(), InputRowsBacking::F64);
    let (mapping, error_pointer, bias_error_pointer) = match &original.storage {
        Storage::Fused { rows, .. } => (
            original
                .lower(0)
                .unwrap()
                .entries()
                .map(|(i, _, _)| i)
                .collect::<Vec<_>>(),
            rows.e.as_ptr(),
            rows.eb.as_ptr(),
        ),
        Storage::Dense { .. } => panic!("patch producer must retain fused rows"),
    };
    let old_floor = original.resident_floor(at).unwrap();
    let required = original.f32_storage_required_bytes(old_floor, at).unwrap();
    let transfer = original.into_f32(old_floor, required, at).unwrap();
    let pair = transfer.rows;
    let receipt = transfer.receipt;
    assert_eq!(pair.backing(), InputRowsBacking::F32);
    pair.validate_scope(&net, &root, 4, &targets).unwrap();
    assert_eq!(pair.deadline(), at);
    assert_eq!(receipt.deadline, at);
    assert_eq!(receipt.peak_admitted_bytes, required);
    assert_eq!(receipt.coefficient_cells_stored, 9 * 2 * targets.len());
    assert_eq!(receipt.bias_cells_stored, 2 * targets.len());
    assert!(receipt.changed_cells > 0);
    assert!(receipt.new_payload_bytes < receipt.old_payload_bytes);
    assert!(pair.resident_floor(at).unwrap() < old_floor);
    assert_eq!(
        pair.lower(0)
            .unwrap()
            .entries()
            .map(|(i, _, _)| i)
            .collect::<Vec<_>>(),
        mapping
    );
    let Storage::Fused { rows, .. } = &pair.storage else {
        unreachable!()
    };
    assert_eq!(rows.e.as_ptr(), error_pointer);
    assert_eq!(rows.eb.as_ptr(), bias_error_pointer);
    assert!(matches!(rows.a, Coefficients::F32(_)) && matches!(rows.b, Biases::F32(_)));
    assert_eq!(
        receipt.new_nominal_allocation_bytes,
        (receipt.coefficient_cells_stored + receipt.bias_cells_stored) * size_of::<f32>()
    );
    for pattern in 0..4 {
        let input: Vec<_> = (0..25)
            .map(|i| match pattern {
                0 => -0.75,
                1 => 1.0,
                2 => {
                    if i % 2 == 0 {
                        -0.75
                    } else {
                        1.0
                    }
                }
                _ => {
                    if i % 3 == 0 {
                        -0.25
                    } else {
                        0.5
                    }
                }
            })
            .collect();
        for (col, &target) in targets.iter().enumerate() {
            let original = original_residual_preactivation(&input, target);
            check_line(pair.lower(col).unwrap(), &input, &original, false, at);
            check_line(pair.upper(col).unwrap(), &input, &original, true, at);
        }
    }
    assert!(pair.validate_scope(&net, &root, 4, &[0, 24, 12]).is_err());
    assert!(
        pair.into_f32(0, 1 << 24, at).is_err(),
        "a repeat cannot mint a zero-op receipt"
    );
}

#[test]
fn input_rows_storage_dense_lower_upper_constants_survive_actual_f32_delivery() {
    let _env = crate::tests::lock_env_shared();
    let net = pointwise_fixture(false);
    let root = root(&net, &[-2.0; 25], &[-1.0; 25], None);
    let targets = [0, 24];
    let at = deadline();
    let mut seed = Array2::zeros((25, targets.len()));
    for (col, &target) in targets.iter().enumerate() {
        seed[[target, col]] = 1.0;
    }
    let rows = crate::margin_row::engine::BackwardEngine::new(&net, &root)
        .with_err_lane(crate::margin_row::errlane::ErrLane::PerCol)
        .with_gpu_conv_route(crate::margin_row::gpu_conv::GpuConvRoute::Off)
        .run_prefix_input_rows(
            &crate::margin_row::engine::Seed { s: seed, e: None },
            3,
            &targets,
            0,
            1 << 24,
            at,
        )
        .unwrap();
    let budget = rows.f32_storage_required_bytes(0, at).unwrap();
    let transfer = rows.into_f32(0, budget, at).unwrap();
    assert_eq!(
        transfer.receipt.coefficient_cells_stored,
        2 * 25 * targets.len()
    );
    assert_eq!(transfer.receipt.bias_cells_stored, 2 * targets.len());
    assert!(matches!(transfer.rows.storage, Storage::Dense { .. }));
    for col in 0..targets.len() {
        assert!(
            transfer.rows.upper(col).unwrap().bias() < transfer.rows.lower(col).unwrap().bias()
        );
        for x in [-2.0, -1.75, -1.5, -1.25, -1.0] {
            let original = (q(x) + q(1.5)).max(Q::zero());
            check_line(
                transfer.rows.lower(col).unwrap(),
                &[x; 25],
                &original,
                false,
                at,
            );
            check_line(
                transfer.rows.upper(col).unwrap(),
                &[x; 25],
                &original,
                true,
                at,
            );
        }
    }
}

#[test]
fn input_rows_storage_live_overlap_cap_and_earliest_deadline_refuse_before_publication() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(false);
    let original_deadline = deadline();
    let root = root(&net, &[-0.75; 25], &[1.0; 25], Some(original_deadline));
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [0, 12];
    let fresh = original_deadline + Duration::from_secs(600);
    let rows = products(&plan, &net, &root, &targets, fresh).rows;
    let floor = rows.resident_floor(fresh).unwrap();
    let required = rows.f32_storage_required_bytes(floor, fresh).unwrap();
    assert!(matches!(
        rows.into_f32(floor, required - 1, fresh),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    let rows = products(&plan, &net, &root, &targets, fresh).rows;
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(rows.into_f32(floor, required, expired).is_err());
    let rows = products(&plan, &net, &root, &targets, fresh).rows;
    assert!(rows.f32_storage_required_bytes(usize::MAX, fresh).is_err());
    let shorter = original_deadline
        .checked_sub(Duration::from_secs(1))
        .unwrap();
    let delivered = rows.into_f32(floor, required, shorter).unwrap();
    assert_eq!(delivered.rows.deadline(), shorter);
    assert_eq!(delivered.receipt.deadline, shorter);
    assert!(delivered
        .rows
        .lower(0)
        .unwrap()
        .pointwise_error(expired)
        .is_err());
}

#[test]
fn input_rows_storage_identity_centers_are_still_actual_f32_cells_with_nonzero_receipt() {
    let _env = crate::tests::lock_env_shared();
    let net = pointwise_fixture(false);
    let root = root(&net, &[0.0; 25], &[1.0; 25], None);
    let plan = PatchPlan::build(&net, 1, 0.5).unwrap();
    let targets = [0, 24];
    let at = deadline();
    let original = products(&plan, &net, &root, &targets, at).rows;
    let budget = original.f32_storage_required_bytes(0, at).unwrap();
    let result = original.into_f32(0, budget, at).unwrap();
    assert_eq!(result.receipt.changed_cells, 0);
    assert!(result.receipt.coefficient_cells_stored > 0 && result.receipt.bias_cells_stored > 0);
    assert_eq!(result.rows.backing(), InputRowsBacking::F32);
    assert!(result.receipt.new_payload_bytes < result.receipt.old_payload_bytes);
}

#[test]
fn input_rows_storage_overflow_nonfinite_carried_error_and_expired_allocator_refuse() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    for source in [
        f64::MAX,
        -f64::MAX,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let mut e = 0.0;
        let mut paid = conversion_admission(1);
        assert!(store_cell(source, &mut e, &mut paid).is_err());
    }
    for error in [f64::NAN, f64::INFINITY, -1.0] {
        let mut e = error;
        let mut paid = conversion_admission(1);
        assert!(store_cell(1.0, &mut e, &mut paid).is_err());
    }
    assert!(add_error_up(f64::MAX, f64::MAX).is_err());
    let mut paid = conversion_admission(1);
    assert!(paid
        .allocate(
            1,
            Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
        )
        .is_err());
    assert!(paid.allocate(usize::MAX, deadline()).is_err());
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct DirectedRounding {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl DirectedRounding {
    // The register is thread-local. This test-only guard saves it before
    // alteration and restores it during normal return and unwinding.
    #[allow(unsafe_code)]
    fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr", saved = out(reg) saved, options(nomem, nostack));
            let directed = (saved & !(3_u64 << 22)) | (1_u64 << 22);
            std::arch::asm!("msr fpcr, {directed}", "isb", directed = in(reg) directed, options(nomem, nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr((saved & !(3 << 13)) | (1 << 13));
            Self { saved }
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for DirectedRounding {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}", "isb", saved = in(reg) self.saved, options(nomem, nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn input_rows_storage_unqualified_rounding_environment_cannot_publish_f32_backing() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(false);
    let root = root(&net, &[-0.75; 25], &[1.0; 25], None);
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [0];
    let at = deadline();
    let rows = products(&plan, &net, &root, &targets, at).rows;
    let budget = rows.f32_storage_required_bytes(0, at).unwrap();
    {
        let _directed = DirectedRounding::enter();
        assert!(!ny_core::has_f64_interval_proof_environment());
        assert!(matches!(
            rows.into_f32(0, budget, at),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    ny_core::require_f64_interval_proof_environment().unwrap();
}

// External instrumentation for the original and compact row representations.
// Append to input_rows_storage_tests.rs only in a quiet, owned source window.

fn observe_native_row_accounting(label: &str, pair: InputRowPair<'_>, at: Instant) {
    let source_bytes = crate::margin_row::engine::input_product_resident_floor_with_deadline(
        pair.scope.net(),
        pair.root(),
        at,
    )
    .unwrap();
    let target_bytes = size_of_val(pair.scope.targets());
    let pair_bytes = size_of::<InputRowPair<'static>>();
    let payload_before = pair.owned_payload_bytes().unwrap();
    let floor_before = pair.resident_floor(at).unwrap();
    assert_eq!(
        floor_before,
        source_bytes + payload_before + pair_bytes + target_bytes
    );
    let (coefficient_cells, bias_cells) = pair.nominal_cells().unwrap();
    let nominal_bytes = (coefficient_cells + bias_cells) * size_of::<f32>();
    let required = pair.f32_storage_required_bytes(floor_before, at).unwrap();
    assert_eq!(
        required,
        floor_before + nominal_bytes + F32_TRANSFER_FIXED_BYTES
    );
    let transfer = pair.into_f32(floor_before, required, at).unwrap();
    let payload_after = transfer.rows.owned_payload_bytes().unwrap();
    let floor_after = transfer.rows.resident_floor(at).unwrap();
    assert_eq!(
        floor_after,
        source_bytes + payload_after + pair_bytes + target_bytes
    );
    assert_eq!(transfer.receipt.old_payload_bytes, payload_before);
    assert_eq!(transfer.receipt.new_payload_bytes, payload_after);
    assert_eq!(transfer.receipt.new_nominal_allocation_bytes, nominal_bytes);
    assert_eq!(transfer.receipt.peak_admitted_bytes, required);
    assert_eq!(transfer.rows.backing(), InputRowsBacking::F32);
    println!(
        "{{\"phase\":\"input_rows_accounting_{label}\",\"rows_bytes\":{},\"storage_bytes\":{},\"pair_bytes\":{},\"transfer_bytes\":{},\"owner_envelope_bytes\":{},\"f32_transfer_fixed_bytes\":{},\"source_bytes\":{source_bytes},\"target_bytes\":{target_bytes},\"payload_before\":{payload_before},\"payload_after\":{payload_after},\"floor_before\":{floor_before},\"floor_after\":{floor_after},\"coefficient_cells\":{coefficient_cells},\"bias_cells\":{bias_cells},\"nominal_allocation_bytes\":{nominal_bytes},\"peak_admitted_bytes\":{required},\"changed_cells\":{}}}",
        size_of::<Rows>(),
        size_of::<Storage>(),
        pair_bytes,
        size_of::<InputRowsF32Transfer<'static>>(),
        INPUT_ROWS_OWNER_BYTES,
        F32_TRANSFER_FIXED_BYTES,
        transfer.receipt.changed_cells,
    );
}

#[test]
fn input_rows_storage_native_complete_accounting_observation() {
    let _env = crate::tests::lock_env_shared();
    {
        let net = residual_fixture(true);
        let root = root(&net, &[-0.75; 25], &[1.0; 25], None);
        let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
        let targets = [24, 0, 12];
        let at = deadline();
        let pair = products(&plan, &net, &root, &targets, at).rows;
        assert!(matches!(pair.storage, Storage::Fused { .. }));
        observe_native_row_accounting("fused", pair, at);
    }
    {
        let net = pointwise_fixture(false);
        let root = root(&net, &[-2.0; 25], &[-1.0; 25], None);
        let targets = [0, 24];
        let at = deadline();
        let mut seed = Array2::zeros((25, targets.len()));
        for (col, &target) in targets.iter().enumerate() {
            seed[[target, col]] = 1.0;
        }
        let pair = crate::margin_row::engine::BackwardEngine::new(&net, &root)
            .with_err_lane(crate::margin_row::errlane::ErrLane::PerCol)
            .with_gpu_conv_route(crate::margin_row::gpu_conv::GpuConvRoute::Off)
            .run_prefix_input_rows(
                &crate::margin_row::engine::Seed { s: seed, e: None },
                3,
                &targets,
                0,
                1 << 24,
                at,
            )
            .unwrap();
        assert!(matches!(pair.storage, Storage::Dense { .. }));
        observe_native_row_accounting("dense", pair, at);
    }
}
