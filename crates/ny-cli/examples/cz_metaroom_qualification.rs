// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit real-model qualification lane for the unwired Metaroom CZ work.
//!
//! Run this example under `ny-safe-gpu-run`, one selector at a time. Every
//! algorithm retains the hard memory, work, solve-count, and wall-time caps
//! sealed in the shared qualification source.

#![deny(unsafe_code)]

mod cz_metaroom_unwired {
    include!("../src/commands/cz_metaroom_unwired_impl.rs");

    pub(crate) mod qualification {
        include!("../src/commands/cz_metaroom_qualification.rs");
    }
}

use std::path::{Path, PathBuf};

fn benchmark_root() -> PathBuf {
    std::env::var_os("NY_METAROOM_119_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../benchmarks/vnncomp2025/benchmarks/metaroom_2023")
        })
}

fn require_assets() -> anyhow::Result<()> {
    let root = benchmark_root();
    for relative in [
        "onnx/6cnn_ry_39_6_no_custom_OP.onnx",
        "vnnlib/spec_idx_119_eps_0.00000436.vnnlib",
    ] {
        let path = root.join(relative);
        anyhow::ensure!(
            path.is_file(),
            "missing Metaroom119 prerequisite {}",
            path.display()
        );
    }
    Ok(())
}

fn run_named(name: &str) -> anyhow::Result<()> {
    use cz_metaroom_unwired::qualification as probe;

    require_assets()?;
    match name {
        "trunk" => probe::real_metaroom_119_measures_full_conv_relu_trunk_resources(),
        "projected-trunk" => {
            probe::real_metaroom_119_measures_projected_full_conv_relu_trunk_resources();
        }
        "seal-tail" => probe::real_metaroom_119_seals_affine_tail_topology_and_float32_bits(),
        "full-tail" => {
            probe::real_metaroom_119_measures_full_affine_relu_output_tail_resources();
        }
        #[cfg(feature = "cuda")]
        "cuda-dual" => probe::real_metaroom_119_measures_projected_full_output_cuda_dual(),
        #[cfg(not(feature = "cuda"))]
        "cuda-dual" => anyhow::bail!("cuda-dual requires --features mip,cuda"),
        "box-advantage" => {
            probe::real_metaroom_119_quantifies_certified_box_advantage_over_projected_cz_radii();
        }
        "box-cz-hybrid" => {
            probe::real_metaroom_119_measures_inductive_box_cz_hybrid_resources();
        }
        "hybrid-tail" => {
            let stage = std::env::var("NY_CZ_HYBRID_TAIL_DIAGNOSTIC").map_err(|_| {
                anyhow::anyhow!(
                    "hybrid-tail requires NY_CZ_HYBRID_TAIL_DIAGNOSTIC=smoke0|one8|all8|cascade"
                )
            })?;
            anyhow::ensure!(
                matches!(stage.as_str(), "smoke0" | "one8" | "all8" | "cascade"),
                "invalid NY_CZ_HYBRID_TAIL_DIAGNOSTIC={stage:?}"
            );
            probe::real_metaroom_119_diagnoses_inductive_hybrid_tail_cascade();
        }
        "ay-lp-tail" => {
            anyhow::ensure!(
                std::env::var("NY_CZ_TAIL_AY_LP_DIAGNOSTIC").as_deref() == Ok("1"),
                "ay-lp-tail requires NY_CZ_TAIL_AY_LP_DIAGNOSTIC=1"
            );
            probe::real_metaroom_119_diagnoses_target_6_tail_with_exact_ay_lp();
        }
        _ => anyhow::bail!(
            "unknown selector {name:?}; expected trunk, projected-trunk, seal-tail, \
             full-tail, cuda-dual, box-advantage, box-cz-hybrid, hybrid-tail, or ay-lp-tail"
        ),
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let selector = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing Metaroom qualification selector"))?;
    anyhow::ensure!(args.next().is_none(), "expected exactly one selector");
    run_named(&selector)
}
