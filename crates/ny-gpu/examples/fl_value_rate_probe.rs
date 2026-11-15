// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![recursion_limit = "256"]

//! #fl-value-gpu-tier evidence executable: run the REAL forward-linear rate
//! probe (`ny_propagate::forward_linear_measured_rate`, the exact production
//! build path) with the wgpu FL-value factory installed exactly as
//! `beta_crown` installs it, and report the derated GMAC/s plus the FL-value
//! telemetry that shows whether the GPU tier actually served the probe's
//! value GEMMs.
//!
//! ```text
//! targo --unverified run --release -p ny-gpu --features gpu-tests \
//!   --example fl_value_rate_probe -- --gpu --seam on
//! ```
//!
//! `--gpu` installs the FL-value factory; `--seam on|off` (default `off`, the
//! shipped setting) sets the forward-linear f32 value seam through its typed
//! override, the same call a preset's `forward_linear_f32_seam` key makes.
//! With the seam off the tier is never consulted, whatever `--gpu` says.
//!
//! A measurement, not a pass/fail invariant (`TEST_CONFORMANCE.md`,
//! "Measurements and exhaustive audits"). It exits nonzero when the result
//! would be VOID: `--gpu` without an adapter, a rate that did not come from
//! the probe, or `--gpu --seam on` with no value GEMM SERVED by the GPU tier.
//! "Served" is the telemetry's `hits` (results the seam validated and
//! published), not `calls`: `calls` also counts every refusal (size, memory,
//! dispatch, deadline, non-finite), each of which falls through to the CPU
//! tiers, so `calls > 0, hits == 0` is a CPU rate.
//!
//! "Did not come from the probe" means a calibration rep overran the probe's
//! 5 s rep deadline and the shipped fallback constant was reported instead.
//! Measured on an M5 Max (release, load ~20, 2026-09-30): seam off runs
//! 19.3 G cold-build MACs in ~1.3-1.5 s per rep; seam on uses the GPU-regime
//! fixture (64 out-channels, 77.3 G MACs) and runs ~2.7-3.9 s per rep with
//! `--gpu`. `--seam on` WITHOUT `--gpu` sends those 77.3 G MACs to the CPU f32
//! tier, which did not finish rep 0 inside 5 s (5.0 s wall, 11.5 s user +
//! 34 s sys), so that mode is VOID on this host. A debug build overruns in
//! every mode, and a saturated host (load ~94 measured) pushes every mode past
//! the deadline: re-run on a quiet machine before reading a VOID as a
//! regression.
//!
//! Flags, not raw environment variables, select the mode, so the lever
//! ratchets stay flat.

use std::process::ExitCode;
use std::sync::Arc;

fn usage() -> ExitCode {
    eprintln!("usage: fl_value_rate_probe [--gpu] [--seam on|off]");
    ExitCode::from(64)
}

fn main() -> ExitCode {
    let mut install_gpu = false;
    let mut seam = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gpu" => install_gpu = true,
            "--seam" => match args.next().as_deref() {
                Some("on") => seam = true,
                Some("off") => seam = false,
                _ => return usage(),
            },
            "-h" | "--help" => {
                let _ = usage(); // Help prints usage and deliberately succeeds.
                return ExitCode::SUCCESS;
            }
            _ => return usage(),
        }
    }

    if install_gpu {
        if !ny_gpu::wgpu_adapter_available() {
            eprintln!("FL_PROBE_VOID reason=no-wgpu-adapter (--gpu needs one)");
            return ExitCode::from(2);
        }
        // Identical to the beta_crown registration (#fl-value-gpu-tier).
        ny_propagate::fl_value_gemm::set_fl_value_gemm_factory(|| {
            match ny_gpu::FlValueGemmDevice::new_wgpu() {
                Ok(device) => Some(Arc::new(device) as Arc<dyn ny_core::GemmEngine>),
                Err(_) => None,
            }
        });
    }
    ny_propagate::set_forward_linear_f32_seam(seam);

    let obs = ny_propagate::forward_linear_measured_rate();
    let telem = ny_propagate::fl_value_gemm::telemetry_snapshot();
    println!(
        "FL_PROBE_RESULT gpu_factory_installed={install_gpu} seam_requested={seam} source={} \
         seam_f32={} macs_per_sec={} derated_gmacs_per_sec={:.3} probe_secs={:.4} \
         fl_calls={} fl_hits={} fl_backend={:?}",
        obs.source,
        obs.seam_f32,
        obs.macs_per_sec,
        obs.macs_per_sec as f64 / 1e9,
        obs.probe_secs,
        telem.calls,
        telem.hits,
        telem.backend,
    );

    if obs.source != "probe" {
        let why = match obs.source {
            "env" => "the manual MACs-per-second override replaced the probe; unset it",
            "fallback" => {
                "a calibration rep failed or overran the 5 s deadline; \
                 use --release on a quiet host"
            }
            _ => "the rate did not come from the probe",
        };
        eprintln!(
            "FL_PROBE_VOID reason=rate-source-{} (not a measurement: {why})",
            obs.source
        );
        return ExitCode::from(3);
    }
    if obs.seam_f32 != seam {
        eprintln!(
            "FL_PROBE_VOID reason=seam-mode-mismatch measured={}",
            obs.seam_f32
        );
        return ExitCode::from(3);
    }
    if install_gpu && seam && telem.hits == 0 {
        let reason = if telem.calls == 0 {
            "gpu-tier-never-consulted"
        } else {
            "gpu-tier-refused-every-call"
        };
        eprintln!(
            "FL_PROBE_VOID reason={reason} fl_calls={} fl_hits=0 (the rate is the CPU tiers')",
            telem.calls
        );
        return ExitCode::from(3);
    }
    ExitCode::SUCCESS
}
