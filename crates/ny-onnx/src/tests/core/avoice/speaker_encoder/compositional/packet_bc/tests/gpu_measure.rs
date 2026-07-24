// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::boundary::{
    discover_ecapa_composition_boundary, EcapaCompositionBoundary,
};
use super::*;

struct StageMeasurement {
    tightened: usize,
    node_count: usize,
    fallback: usize,
    elapsed_secs: f64,
    output_width: f32,
}

fn measure_stage_a_crown_ibp(
    stage: &GraphNetwork,
    input: &BoundedTensor,
    label: &str,
    deadline_secs: u64,
    engine: Option<&dyn ny_core::GemmEngine>,
) -> StageMeasurement {
    let start = Instant::now();
    let result = collect_ecapa_stage_local_crown_ibp(stage, input, label, deadline_secs, engine)
        .unwrap_or_else(|e| panic!("{label} Stage A should succeed: {e}"));
    let elapsed = start.elapsed();
    let prov = stage_provenance(&result);
    let output = output_bounds_from_crown_result(&result, stage.output_name(), label)
        .unwrap_or_else(|e| panic!("{label} output bounds should exist: {e}"));
    assert_finite_and_ordered(&output, &format!("{label} Stage A output"));
    StageMeasurement {
        tightened: prov.crown_ibp_tightened_count,
        node_count: prov.node_count,
        fallback: prov.ibp_fallback_count,
        elapsed_secs: elapsed.as_secs_f64(),
        output_width: output.max_width(),
    }
}

fn report_gpu_vs_cpu(cpu: &StageMeasurement, gpu: Option<&StageMeasurement>, deadline_secs: u64) {
    eprintln!("Stage A GPU vs CPU measurement ({deadline_secs}s deadline):");
    eprintln!(
        "  CPU: tightened={}/{}, fallback={}, elapsed={:.1}s, output_width={:.6}",
        cpu.tightened, cpu.node_count, cpu.fallback, cpu.elapsed_secs, cpu.output_width,
    );
    if let Some(gpu) = gpu {
        eprintln!(
            "  GPU: tightened={}/{}, fallback={}, elapsed={:.1}s, output_width={:.6}",
            gpu.tightened, gpu.node_count, gpu.fallback, gpu.elapsed_secs, gpu.output_width,
        );
        match gpu.tightened.cmp(&cpu.tightened) {
            std::cmp::Ordering::Greater => eprintln!(
                "  GPU tightened {} more nodes ({} vs {})",
                gpu.tightened - cpu.tightened,
                gpu.tightened,
                cpu.tightened,
            ),
            std::cmp::Ordering::Equal => {
                eprintln!(
                    "  GPU and CPU tightened the same number of nodes ({})",
                    cpu.tightened
                )
            }
            std::cmp::Ordering::Less => {
                eprintln!(
                    "  GPU tightened FEWER nodes ({} vs {}) — unexpected",
                    gpu.tightened, cpu.tightened
                )
            }
        }
    } else {
        eprintln!("  SKIP: wgpu device not available — GPU measurement unavailable");
    }
}

/// Measurement: compare CROWN-IBP Stage A tightening depth with and without
/// a real GPU engine on the same 60s deadline. Answers: does GPU acceleration
/// push CROWN backward past the 4/52-node envelope seen with CPU-only?
/// Part of #3499 — GPU threading measurement lane.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_ecapa_stage_a_gpu_vs_cpu_tightening_depth_3499() {
    crate::test_fixtures::require_test_model_or_skip!("speaker_encoder.onnx");
    use ny_gpu::{Backend, ComputeDevice};
    const MEASUREMENT_DEADLINE_SECS: u64 = 60;

    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let boundary =
        discover_ecapa_composition_boundary(graph).expect("MFA boundary discovery should succeed");
    let [stage_a, _, _] = extract_ecapa_stage_graphs(graph, &boundary)
        .expect("stage graph extraction should succeed");

    let cpu = measure_stage_a_crown_ibp(
        &stage_a,
        &input,
        "cpu_baseline",
        MEASUREMENT_DEADLINE_SECS,
        None,
    );
    let gpu_device = ComputeDevice::new(Backend::Wgpu).ok();
    let gpu = gpu_device.as_ref().map(|device| {
        let engine: &dyn ny_core::GemmEngine = device;
        measure_stage_a_crown_ibp(
            &stage_a,
            &input,
            "gpu_engine",
            MEASUREMENT_DEADLINE_SECS,
            Some(engine),
        )
    });

    report_gpu_vs_cpu(&cpu, gpu.as_ref(), MEASUREMENT_DEADLINE_SECS);
    assert!(
        cpu.tightened > 0,
        "CPU baseline should tighten at least 1 node"
    );
}

fn stage_max_width(result: &EcapaStageResult, label: &str) -> f32 {
    match label {
        "x2" => result.x2_bounds.max_width(),
        "x3" => result.x3_bounds.max_width(),
        _ => result.x4_bounds.max_width(),
    }
}

fn report_alpha_gpu_vs_cpu(
    boundary: &EcapaCompositionBoundary,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    cpu: &EcapaStageResult,
    cpu_elapsed: f64,
    gpu: Option<(&EcapaStageResult, f64)>,
    deadline_secs: u64,
    iterations: usize,
) {
    eprintln!("Alpha-CROWN GPU vs CPU ({deadline_secs}s deadline, {iterations} iter):");
    for (label, output_name) in [
        ("x2", &boundary.block_outputs[0]),
        ("x3", &boundary.block_outputs[1]),
        ("x4", &boundary.block_outputs[2]),
    ] {
        let ibp_w = ibp_bounds
            .get(output_name)
            .map(|b| b.max_width())
            .unwrap_or(f32::NAN);
        let cpu_w = stage_max_width(cpu, label);
        eprintln!(
            "  {label}: IBP={ibp_w:.6e}, CPU={cpu_w:.6e} ({:.1}%)",
            width_reduction_pct(ibp_w, cpu_w),
        );
        if let Some((gpu_r, _)) = &gpu {
            let gpu_w = stage_max_width(gpu_r, label);
            eprintln!(
                "         GPU={gpu_w:.6e} ({:.1}% vs IBP, {:.1}% vs CPU)",
                width_reduction_pct(ibp_w, gpu_w),
                width_reduction_pct(cpu_w, gpu_w),
            );
        }
    }
    eprintln!("  CPU runtime: {cpu_elapsed:.1}s");
    if let Some((_, gpu_e)) = &gpu {
        eprintln!("  GPU runtime: {gpu_e:.1}s");
    } else {
        eprintln!("  SKIP: wgpu device not available");
    }
}

/// Level 0.5 measurement for #3499: alpha-CROWN GPU vs CPU stage widths.
///
/// Reference: `designs/2026-03-13-issue-3499-res2net-bound-widening-root-cause.md`
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_ecapa_alpha_crown_gpu_vs_cpu_stage_widths_3499() {
    crate::test_fixtures::require_test_model_or_skip!("speaker_encoder.onnx");
    use ny_gpu::{Backend, ComputeDevice};
    const DEADLINE: u64 = 60;
    const ITERS: usize = 1;

    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let boundary = discover_ecapa_composition_boundary(graph).expect("MFA boundary discovery");
    let ibp = graph
        .collect_node_bounds(&input)
        .expect("IBP should succeed");

    let mut config = alpha_crown_config_for_stage(DEADLINE);
    config.iterations = ITERS;
    let cpu_t = Instant::now();
    let cpu = run_ecapa_stage_local_alpha_crown(graph, &input, &config, None)
        .expect("CPU alpha-CROWN should succeed");
    let cpu_e = cpu_t.elapsed().as_secs_f64();

    let gpu_device = ComputeDevice::new(Backend::Wgpu).ok();
    let gpu = gpu_device.as_ref().map(|device| {
        let engine: &dyn ny_core::GemmEngine = device;
        let mut gc = alpha_crown_config_for_stage(DEADLINE);
        gc.iterations = ITERS;
        let t = Instant::now();
        let r = run_ecapa_stage_local_alpha_crown(graph, &input, &gc, Some(engine))
            .expect("GPU alpha-CROWN should succeed");
        (r, t.elapsed().as_secs_f64())
    });

    report_alpha_gpu_vs_cpu(
        &boundary,
        &ibp,
        &cpu,
        cpu_e,
        gpu.as_ref().map(|(r, e)| (r, *e)),
        DEADLINE,
        ITERS,
    );
    assert_finite_and_ordered(&cpu.mfa_bounds, "CPU alpha MFA");
    if let Some((ref g, _)) = gpu {
        assert_finite_and_ordered(&g.mfa_bounds, "GPU alpha MFA");
    }
}
