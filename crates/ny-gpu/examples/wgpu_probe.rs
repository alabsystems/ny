// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! WGPU proof diagnostic: enumerate an adapter, show that an ordinary device is
//! unarmed, then make one explicit typed qualification request and print its
//! five-rung result. No environment variable can grant authority.
fn main() {
    use ny_core::{GemmEngine, GpuCrownBackward};

    let prov = ny_gpu::wgpu_adapter_provenance();
    println!(
        "adapter_provenance: hardware_available={} description={}",
        prov.hardware_available, prov.description
    );
    println!(
        "wgpu_backend_compiled = {}",
        ny_gpu::wgpu_backend_compiled()
    );
    println!(
        "wgpu_proof_path_compiled = {}",
        ny_gpu::wgpu_proof_authority()
    );
    match ny_gpu::WgpuDevice::new() {
        Ok(d) => {
            println!("WgpuDevice::new: OK (ordinary/unarmed)");
            println!(
                "ordinary_provides_sound_gpu_crown = {}",
                d.provides_sound_gpu_crown()
            );
            println!(
                "ordinary_as_gpu_crown_backward = {}",
                d.as_gpu_crown_backward().is_some()
            );
        }
        Err(e) => println!("WgpuDevice::new: ERR {e}"),
    }

    match ny_gpu::WgpuDevice::new_for_verdict(ny_gpu::WgpuVerdictRequest::new()) {
        Ok(d) => {
            let report = d
                .verdict_report()
                .expect("typed constructor returned without its qualification report");
            println!(
                "WgpuDevice::new_for_verdict: QUALIFIED adapter={:?} reason={}",
                report.adapter(),
                report.reason()
            );
            println!(
                "qualified_as_gpu_crown_backward = {}",
                d.as_gpu_crown_backward().is_some()
            );
            println!(
                "qualified_as_gpu_ibp_forward = {} (non-CROWN remains closed)",
                d.as_gpu_ibp_forward().is_some()
            );
        }
        Err(error) => {
            let report = error.report();
            println!(
                "WgpuDevice::new_for_verdict: REFUSED adapter={:?} failed_rung={:?} reason={}",
                report.adapter(),
                report.failed_rung(),
                report.reason()
            );
            println!("fallback = cpu");
        }
    }
}
