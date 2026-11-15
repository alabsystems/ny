// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cooperative cached-plan deadlines; no GPU verdict authority is added.
use super::WgpuIbpModelPlan;
use ny_core::{GpuIbpResult, NyError, Result};
use std::mem::size_of;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[cfg(all(test, feature = "gpu-tests"))]
thread_local! {
    static EXPIRE_AFTER_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A hardware regression can deterministically expire the call only after a
/// real queue submission. This seam is absent from every production build.
#[cfg(all(test, feature = "gpu-tests"))]
pub(super) fn expiry_after_submit(
    deadline: Option<Instant>,
) -> Option<crate::wgpu_device::CallLocalCrownDeadlineScope> {
    if deadline.is_some() && EXPIRE_AFTER_SUBMIT.with(|armed| armed.replace(false)) {
        return Some(crate::wgpu_device::CallLocalCrownDeadlineScope::arm(
            Instant::now(),
        ));
    }
    None
}

pub(in crate::wgpu_device::ops) fn poll_plan_deadline(deadline: Option<Instant>) -> Result<()> {
    let effective = deadline.map(effective_plan_deadline);
    if effective.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "resident IBP plan: original deadline expired".into(),
        ));
    }
    Ok(())
}

pub(in crate::wgpu_device::ops) fn effective_plan_deadline(deadline: Instant) -> Instant {
    let instance_capped =
        ny_core::instance_budget::deadline().map_or(deadline, |instance| instance.min(deadline));
    crate::wgpu_device::CALL_LOCAL_CROWN_DEADLINE
        .with(std::cell::Cell::get)
        .map_or(instance_capped, |active| active.min(instance_capped))
}

pub(super) fn copy_plan_values(
    values: &[f32],
    transform: fn(f32) -> f32,
    deadline: Option<Instant>,
) -> Result<Vec<f32>> {
    if deadline.is_none() {
        return Ok(values.iter().copied().map(transform).collect());
    }
    poll_plan_deadline(deadline)?;
    let mut out = Vec::with_capacity(values.len());
    for chunk in values.chunks(4096) {
        poll_plan_deadline(deadline)?;
        out.extend(chunk.iter().copied().map(transform));
    }
    poll_plan_deadline(deadline)?;
    Ok(out)
}

pub(super) fn upload_plan_values(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    values: &[f32],
    deadline: Option<Instant>,
) -> Result<()> {
    if deadline.is_none() {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(values));
        return Ok(());
    }
    for (i, chunk) in values.chunks(4096).enumerate() {
        poll_plan_deadline(deadline)?;
        queue.write_buffer(
            buffer,
            (i * 4096 * size_of::<f32>()) as u64,
            bytemuck::cast_slice(chunk),
        );
    }
    poll_plan_deadline(deadline)
}

impl WgpuIbpModelPlan {
    pub(super) fn execute_until(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
        deadline: Instant,
    ) -> Result<GpuIbpResult> {
        poll_plan_deadline(Some(deadline))?;
        let _scope =
            crate::wgpu_device::CallLocalCrownDeadlineScope::arm(effective_plan_deadline(deadline));
        let deadline = crate::wgpu_device::CALL_LOCAL_CROWN_DEADLINE
            .with(std::cell::Cell::get)
            .unwrap_or(deadline);
        let mut transaction =
            crate::wgpu_device::error_scope::begin_gpu_checked_transaction_on_device(
                &self.device,
                &self.serialize,
                &self.uncaptured_errors,
                "ibp_forward_cached",
                deadline,
            )?;
        let _guard = loop {
            match self.execution_serialize.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(NyError::InternalError(
                        "cached IBP execution lock poisoned".into(),
                    ))
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    poll_plan_deadline(Some(deadline))?;
                    std::thread::yield_now();
                }
            }
        };
        if self.poisoned.load(Ordering::Acquire) {
            return Err(NyError::InternalError(
                "cached IBP plan discarded after an incomplete GPU submission".into(),
            ));
        }
        let mut submitted = false;
        let result = (|| {
            let value = self.ibp_forward_cached_inner_until(
                input_lower,
                input_upper,
                input_shape,
                Some(deadline),
                &mut submitted,
            )?;
            transaction.finish_bounded("ibp_forward_cached")?;
            poll_plan_deadline(Some(deadline))?;
            Ok(value)
        })();
        // A timeout may leave queued GPU work alive. Do not reuse these mutable
        // buffers after such an incomplete submission; no partial value escapes.
        if submitted && result.is_err() {
            self.poisoned.store(true, Ordering::Release);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    static COPIED: AtomicUsize = AtomicUsize::new(0);
    fn counted_identity(value: f32) -> f32 {
        COPIED.fetch_add(1, Ordering::Relaxed);
        value
    }

    #[test]
    fn plan_copy_expiry_precedes_transform_and_live_copy_preserves_bits() {
        COPIED.store(0, Ordering::Relaxed);
        let values = [
            0.0_f32,
            -0.0,
            f32::from_bits(1),
            f32::INFINITY,
            f32::from_bits(0x7fc12345),
        ];
        assert!(matches!(
            copy_plan_values(&values, counted_identity, Some(Instant::now())),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(COPIED.load(Ordering::Relaxed), 0);
        let live = Instant::now() + Duration::from_secs(10);
        let got = copy_plan_values(&values, counted_identity, Some(live)).unwrap();
        assert_eq!(COPIED.load(Ordering::Relaxed), values.len());
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn plan_deadline_composes_with_tighter_outer_scope() {
        let live = Instant::now() + Duration::from_secs(10);
        let _outer = crate::wgpu_device::CallLocalCrownDeadlineScope::arm(Instant::now());
        assert!(matches!(
            poll_plan_deadline(Some(live)),
            Err(NyError::DeadlineExceeded(_))
        ));
        // None retains the historical unbounded plan behavior.
        assert!(poll_plan_deadline(None).is_ok());
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use ny_core::{GpuIbpForwardExt, GpuIbpLayer};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn deadline_cached_gpu_plan_reuses_static_layers_and_refuses_expired_entry() {
        let _serial = crate::wgpu_device::test_support::gpu_test_serial_guard();
        let device = crate::wgpu_device::test_support::require_device();
        let layers = vec![
            GpuIbpLayer::Linear {
                weight: Arc::from(vec![1.0_f32, -1.0, 0.5, 0.5]),
                bias: Some(Arc::from(vec![0.0, 0.0])),
                out_features: 2,
                in_features: 2,
            },
            GpuIbpLayer::ReLU { num_elements: 2 },
            GpuIbpLayer::Linear {
                weight: Arc::from(vec![1.0_f32, 2.0]),
                bias: Some(Arc::from(vec![0.25])),
                out_features: 1,
                in_features: 2,
            },
        ];
        let live = Instant::now() + Duration::from_secs(30);
        let plan = device
            .prepare_model_plan_with_deadline(&layers, &[2], live)
            .expect("deadline-capable WGPU preparation must complete on the required adapter")
            .expect("nonempty dense chain must retain one cached plan");
        assert!(plan.supports_deadline_execution());
        for (point, expected) in [([1.0_f32, -1.0], 2.25_f32), ([0.5, 0.5], 1.25)] {
            let bounded = plan
                .ibp_forward_cached_with_deadline(&point, &point, &[2], live)
                .expect("same cached plan must execute within its original deadline");
            let legacy = plan.ibp_forward_cached(&point, &point, &[2]).unwrap();
            assert_eq!(bounded.output_shape, [1]);
            assert_eq!(
                bounded
                    .lower_bounds
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                legacy
                    .lower_bounds
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                bounded
                    .upper_bounds
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                legacy
                    .upper_bounds
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(bounded.lower_bounds, [expected]);
            assert_eq!(bounded.upper_bounds, [expected]);
        }
        assert!(matches!(
            plan.ibp_forward_cached_with_deadline(&[1.0, -1.0], &[1.0, -1.0], &[2], Instant::now()),
            Err(NyError::DeadlineExceeded(_))
        ));
        // Entry refusal precedes submission and does not poison this plan.
        assert_eq!(
            plan.ibp_forward_cached_with_deadline(&[1.0, -1.0], &[1.0, -1.0], &[2], live)
                .unwrap()
                .lower_bounds,
            [2.25]
        );
        assert!(matches!(
            device.prepare_model_plan_with_deadline(&layers, &[2], Instant::now()),
            Err(NyError::DeadlineExceeded(_))
        ));
        eprintln!("[pgd-deadline-gpu] required_adapter=1 bounded_prepare=1 bounded_calls=3 legacy_calls=2 expired_entry_refused=1 expired_prepare_refused=1");
    }

    #[test]
    fn submitted_deadline_failure_returns_no_output_and_poisons_cached_plan() {
        let _serial = crate::wgpu_device::test_support::gpu_test_serial_guard();
        let device = crate::wgpu_device::test_support::require_device();
        let layers = [GpuIbpLayer::Linear {
            weight: Arc::from(vec![1.0_f32]),
            bias: None,
            out_features: 1,
            in_features: 1,
        }];
        let live = Instant::now() + Duration::from_secs(30);
        let plan = device
            .prepare_model_plan_with_deadline(&layers, &[1], live)
            .unwrap()
            .unwrap();
        EXPIRE_AFTER_SUBMIT.with(|armed| armed.set(true));
        let submitted_result = plan.ibp_forward_cached_with_deadline(&[1.0], &[1.0], &[1], live);
        let hook_consumed = EXPIRE_AFTER_SUBMIT.with(|armed| !armed.replace(false));
        assert!(
            hook_consumed,
            "expiry hook must have been reached after an actual submission"
        );
        assert!(matches!(
            submitted_result,
            Err(NyError::DeadlineExceeded(_))
        ));
        let retry = plan.ibp_forward_cached_with_deadline(&[1.0], &[1.0], &[1], live);
        assert!(
            matches!(retry, Err(NyError::InternalError(message)) if message.contains("discarded after an incomplete GPU submission"))
        );
        let legacy_retry = plan.ibp_forward_cached(&[1.0], &[1.0], &[1]);
        assert!(
            matches!(legacy_retry, Err(NyError::InternalError(message)) if message.contains("discarded after an incomplete GPU submission"))
        );
        eprintln!("[pgd-deadline-gpu-poison] required_adapter=1 actual_submit=1 deterministic_expiry=1 no_output=1 bounded_reuse_refused=1 legacy_reuse_refused=1");
    }
}
