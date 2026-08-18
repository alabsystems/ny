// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Spec-batching helpers for GPU CROWN backward.

use ny_core::{GpuCrownResult, GpuCrownSeed, NyError, Result};

use super::{record_host_phase, CrownGpuTimingProfile, CrownHostTimingProfile, WgpuDevice};

impl WgpuDevice {
    /// Run CROWN backward in spec-batches when buffers exceed wgpu limits.
    ///
    /// Each batch processes `batch_size` specs independently (CROWN backward
    /// is per-spec-row) and the results are concatenated.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn crown_backward_gpu_batched_seeded(
        &self,
        layers: &[ny_core::GpuCrownLayer],
        seed: &GpuCrownSeed,
        first_dim: usize,
        batch_size: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        profiling_enabled: bool,
        host_profiling_enabled: bool,
    ) -> Result<(
        GpuCrownResult,
        Option<CrownGpuTimingProfile>,
        Option<CrownHostTimingProfile>,
    )> {
        let num_specs = seed.num_specs;
        let mut all_lower = Vec::with_capacity(num_specs);
        let mut all_upper = Vec::with_capacity(num_specs);
        let mut offset = 0;
        let mut profile: Option<CrownGpuTimingProfile> = None;
        let mut host_profile = host_profiling_enabled.then(CrownHostTimingProfile::default);

        while offset < num_specs {
            // Cooperative cancellation (#w4-refresh-deadline): a wide per-target
            // refresh (e.g. 14400 specs on cifar100) splits into dozens of
            // batches totalling ~26s; without this check the whole loop was
            // un-interruptible past the verifier budget. Between batches is the
            // safe stop point — the caller treats DeadlineExceeded as a sound
            // reference/IBP fallback.
            if self.crown_backward_deadline_expired() {
                return Err(NyError::DeadlineExceeded(format!(
                    "GPU CROWN spec-batch deadline exceeded at spec offset {offset}/{num_specs}"
                )));
            }
            let batch_specs = batch_size.min(num_specs - offset);
            let batch_seed = record_host_phase(&mut host_profile, "batch_slice", || {
                let row_start = offset * first_dim;
                let row_end = row_start + batch_specs * first_dim;
                GpuCrownSeed {
                    lower_a: seed.lower_a[row_start..row_end].to_vec().into(),
                    upper_a: seed.upper_a[row_start..row_end].to_vec().into(),
                    lower_b: seed.lower_b[offset..offset + batch_specs].to_vec().into(),
                    upper_b: seed.upper_b[offset..offset + batch_specs].to_vec().into(),
                    num_specs: batch_specs,
                    current_dim: first_dim,
                }
            });

            // DEADLOCK FIX: call the *inner* backward, NOT the `GpuCrownBackward`
            // trait method. This batching loop only runs from
            // `crown_backward_gpu_seeded_inner`, i.e. already inside
            // `run_gpu_checked`'s non-reentrant `gpu_serialize` lock (and its
            // thread-local wgpu error scopes) — re-entering the trait method
            // re-locked `gpu_serialize` on the same thread and hung the process
            // (observed: `test_gpu_crown_memory_budget_batching_matches_manual_chunks_3515`
            // deadlocked 30+ min at 0% CPU). Same bug class as the resnet
            // per-segment deadlock fixed earlier (see
            // SESSION_REPORT_RESNET_WIRING_AND_SAT_ATTACK.md).
            let batch_result = self.crown_backward_gpu_seeded_inner(
                layers,
                &batch_seed,
                input_lower,
                input_upper,
            )?;

            if profiling_enabled {
                let batch_profile = self.take_last_crown_timestamp_profile()?.ok_or_else(|| {
                    NyError::InternalError(format!(
                        "missing GPU CROWN timestamp profile for spec batch offset {offset}"
                    ))
                })?;
                if let Some(aggregate) = profile.as_mut() {
                    aggregate.extend_from(batch_profile)?;
                } else {
                    profile = Some(batch_profile);
                }
            }

            if host_profiling_enabled {
                let batch_profile =
                    self.take_last_crown_host_timing_profile()?.ok_or_else(|| {
                        NyError::InternalError(format!(
                            "missing GPU CROWN host timing profile for spec batch offset {offset}"
                        ))
                    })?;
                if let Some(aggregate) = host_profile.as_mut() {
                    aggregate.extend_from(batch_profile);
                }
            }

            record_host_phase(&mut host_profile, "batch_stitch", || {
                all_lower.extend_from_slice(&batch_result.lower_bounds);
                all_upper.extend_from_slice(&batch_result.upper_bounds);
            });
            offset += batch_specs;
        }

        Ok((
            GpuCrownResult {
                lower_bounds: all_lower,
                upper_bounds: all_upper,
            },
            profile,
            host_profile,
        ))
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use crate::wgpu_device::test_support::{
        gpu_test_serial_guard, require_device, require_verdict_device,
    };
    use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuCrownSeed, NyError};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn tiny_linear_layers() -> Vec<GpuCrownLayer> {
        // One 2→2 Linear layer (backward order).
        let w: Vec<f32> = vec![0.5, -0.25, 0.75, 0.1];
        vec![GpuCrownLayer::Linear {
            weight: Arc::from(w.into_boxed_slice()),
            bias: None,
            out_features: 2,
            in_features: 2,
            cert_err: Default::default(),
        }]
    }

    fn identity_seed() -> GpuCrownSeed {
        GpuCrownSeed {
            lower_a: vec![1.0, 0.0, 0.0, 1.0].into(),
            upper_a: vec![1.0, 0.0, 0.0, 1.0].into(),
            lower_b: vec![0.0, 0.0].into(),
            upper_b: vec![0.0, 0.0].into(),
            num_specs: 2,
            current_dim: 2,
        }
    }

    /// #w4-refresh-deadline: an expired cooperative deadline stops the spec-batch
    /// loop between batches with DeadlineExceeded; clearing it restores the
    /// pre-existing run-to-completion behavior (same call succeeds).
    #[test]
    fn spec_batch_loop_honors_cooperative_deadline() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let layers = tiny_linear_layers();
        let seed = identity_seed();
        let (lo, hi) = (vec![-1.0f32, -1.0], vec![1.0f32, 1.0]);

        device.set_crown_backward_deadline(Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("1s before now is representable"),
        ));
        let expired = device.crown_backward_gpu_batched_seeded(
            &layers, &seed, 2, /* batch_size */ 1, &lo, &hi, false, false,
        );
        match expired {
            Err(NyError::DeadlineExceeded(_)) => {}
            Err(other) => panic!("expected DeadlineExceeded from the batch loop, got {other}"),
            Ok(_) => panic!("expired deadline must stop the batch loop, but it completed"),
        }

        device.set_crown_backward_deadline(None);
        let ok = device
            .crown_backward_gpu_batched_seeded(&layers, &seed, 2, 1, &lo, &hi, false, false)
            .expect("cleared deadline must run to completion");
        assert_eq!(ok.0.lower_bounds.len(), 2);
        assert_eq!(ok.0.upper_bounds.len(), 2);
        for (l, u) in ok.0.lower_bounds.iter().zip(ok.0.upper_bounds.iter()) {
            assert!(l.is_finite() && u.is_finite() && l <= u);
        }
    }

    /// #w4-refresh-deadline: the sound resident layer walk checks the same
    /// deadline between layer folds (covers the un-batched sound seeded path).
    #[test]
    fn sound_resident_walk_honors_cooperative_deadline() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let layers = tiny_linear_layers();
        let seed = identity_seed();
        let (lo, hi) = (vec![-1.0f32, -1.0], vec![1.0f32, 1.0]);

        device.set_crown_backward_deadline(Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("1s before now is representable"),
        ));
        let expired = device.crown_backward_gpu_seeded_sound(&layers, &seed, &lo, &hi);
        match expired {
            Err(NyError::DeadlineExceeded(_)) => {}
            Err(other) => panic!("expected DeadlineExceeded from the resident walk, got {other}"),
            Ok(_) => panic!("expired deadline must stop the resident walk, but it completed"),
        }

        device.set_crown_backward_deadline(None);
        let ok = device
            .crown_backward_gpu_seeded_sound(&layers, &seed, &lo, &hi)
            .expect("cleared deadline must run to completion");
        for (l, u) in ok.lower_bounds.iter().zip(ok.upper_bounds.iter()) {
            assert!(l.is_finite() && u.is_finite() && l <= u);
        }
    }
}
