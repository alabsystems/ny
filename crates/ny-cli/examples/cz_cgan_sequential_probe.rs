// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-off constrained-zonotope generator/discriminator handoff probe for
//! the VNN-COMP `cGAN_imgSz32_nCh_1` sequential model.

#![deny(unsafe_code)]

#[path = "../src/commands/cz_cgan_sequential_unwired.rs"]
#[allow(dead_code)]
mod cz_cgan_sequential_unwired;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cz_cgan_sequential_unwired::{
    cgan_nch1_generator_discriminator_handoff_qualification_limits,
    probe_cgan_nch1_generator_discriminator_handoff_unwired, CganCzProbeStatus,
};
use ny_mip::ConstrainedZonotopeCallBudget;
use ny_onnx::vnnlib::load_vnnlib_with_certified_scalar_moat;
use ny_onnx::{load_onnx_with_config, BatchNormFoldingPolicy, OnnxLoadConfig};

fn benchmark_root() -> PathBuf {
    std::env::var_os("NY_CGAN_NCH1_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../benchmarks/vnncomp2025/benchmarks/cgan_2023")
        })
}

fn main() -> anyhow::Result<()> {
    let root = benchmark_root();
    let onnx = root.join("onnx/cGAN_imgSz32_nCh_1.onnx");
    let vnnlib =
        root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib");
    anyhow::ensure!(onnx.is_file(), "missing model {}", onnx.display());
    anyhow::ensure!(vnnlib.is_file(), "missing property {}", vnnlib.display());

    let config = OnnxLoadConfig::default()
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
        .with_raw_float32_initializer_provenance(true);
    let model = load_onnx_with_config(&onnx, &config)?;
    let graph = model.to_graph_network()?;
    let (spec, input, moat) = load_vnnlib_with_certified_scalar_moat(&vnnlib)?;
    anyhow::ensure!(
        spec.num_inputs == 5 && spec.num_outputs == 1,
        "unexpected property boundary"
    );

    let limits = cgan_nch1_generator_discriminator_handoff_qualification_limits();
    let budget = ConstrainedZonotopeCallBudget::new(
        Instant::now() + Duration::from_mins(1),
        64 << 20,
        2 << 30,
    );
    let report = probe_cgan_nch1_generator_discriminator_handoff_unwired(
        &model, &graph, &input, moat, limits, budget,
    );

    println!(
        "authority={:?} topology_work={} parameter_elements={} protected_latents={} peak_live_bytes={} charged_items={} deadline_polls={}",
        report.authority,
        report.topology_work_items,
        report.parameter_elements,
        report.protected_latent_symbols,
        report.peak_live_bytes,
        report.charged_items,
        report.deadline_polls,
    );
    for (index, stage) in report.stages.iter().enumerate() {
        println!(
            "stage[{index:02}] node={} kind={:?} shape={:?} alpha={}->{} nnz={}->{} unstable={} discarded={} peak={} items={} polls={}",
            stage.node,
            stage.kind,
            stage.output_shape,
            stage.input_alpha_dim,
            stage.output_alpha_dim,
            stage.input_generator_nonzeros,
            stage.output_generator_nonzeros,
            stage.unstable_coordinates,
            stage.discarded_generators,
            stage.peak_live_bytes,
            stage.charged_items,
            stage.deadline_polls,
        );
    }
    match &report.status {
        CganCzProbeStatus::PrefixCompleted(completed) => {
            anyhow::ensure!(
                completed.last_node == "Relu_15",
                "handoff probe stopped at unexpected node {}",
                completed.last_node
            );
            println!(
                "prefix_complete last_node={} shape={:?} value_dim={} alpha_dim={} nnz={} max_width={:.17e} mean_width={:.17e} max_box_remainder={:.17e} unsafe_low={:.17e} unsafe_high={:.17e}",
                completed.last_node,
                completed.output_shape,
                completed.value_dim,
                completed.alpha_dim,
                completed.generator_nonzeros,
                completed.maximum_coordinate_width,
                completed.mean_coordinate_width,
                completed.maximum_box_remainder,
                completed.low_unsafe_threshold,
                completed.high_unsafe_threshold,
            );
        }
        CganCzProbeStatus::Declined { node, reason } => {
            println!("declined node={node} reason={reason}");
            anyhow::bail!("handoff probe declined");
        }
        CganCzProbeStatus::Completed(_) => {
            anyhow::bail!("handoff probe unexpectedly returned a full output bound");
        }
    }
    Ok(())
}
