// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Real-corpus qualification for the extended cGAN BatchNorm folds.
//!
//! Usage:
//! `NY_CGAN_PROBE_ONNX=... NY_CGAN_PROBE_VNNLIB=... cargo run -p ny-onnx \
//!  --release --example cgan_bn_fold_probe`
//!
//! Hermetic default coverage remains in the 1,500 fixed-seed ONNX Runtime
//! enclosure cases in `loader::fusion::tests::batch_norm_ort_prop`.

use ndarray::ArrayD;
use ny_core::LayerType;
use ny_onnx::{
    diff, load_onnx_bytes, load_onnx_bytes_with_config, vnnlib, BatchNormFoldingPolicy,
    OnnxLoadConfig, OnnxModel,
};
use ny_tensor::BoundedTensor;

const FOLD_TOL_REL: f32 = 1e-4;

fn main() {
    let onnx_path = std::env::var("NY_CGAN_PROBE_ONNX")
        .expect("NY_CGAN_PROBE_ONNX must name a cgan_2023 generator ONNX");
    let vnnlib_path = std::env::var("NY_CGAN_PROBE_VNNLIB")
        .expect("NY_CGAN_PROBE_VNNLIB must name its VNN-LIB property");
    let bytes = std::fs::read(&onnx_path).expect("failed to read NY_CGAN_PROBE_ONNX");
    let spec = vnnlib::load_vnnlib(&vnnlib_path).expect("failed to parse NY_CGAN_PROBE_VNNLIB");

    let bn_count = |model: &OnnxModel| {
        model
            .network
            .layers
            .iter()
            .filter(|layer| layer.layer_type == LayerType::BatchNorm)
            .count()
    };

    let raw_config = OnnxLoadConfig::default()
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw);
    let model_raw = load_onnx_bytes_with_config("cgan_probe_raw", &bytes, &raw_config)
        .expect("load with raw BatchNorm preserved");
    let model_folded =
        load_onnx_bytes("cgan_probe_folded", &bytes).expect("load with fold enabled");

    let raw_bns = bn_count(&model_raw);
    let folded_bns = bn_count(&model_folded);
    println!(
        "Layers: raw = {}, folded = {}; BatchNorm layers: raw = {raw_bns}, folded = {folded_bns}",
        model_raw.network.layers.len(),
        model_folded.network.layers.len()
    );
    assert!(
        folded_bns < raw_bns,
        "extended folds should remove generator BN layers ({folded_bns} vs {raw_bns})"
    );

    let num_inputs = spec.num_inputs;
    let lower: Vec<f32> = spec.input_bounds.iter().map(|(l, _)| *l as f32).collect();
    let upper: Vec<f32> = spec.input_bounds.iter().map(|(_, u)| *u as f32).collect();
    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[num_inputs]), lower).expect("lower shape");
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[num_inputs]), upper).expect("upper shape");
    let mid: ArrayD<f32> = (&lower + &upper) * 0.5;
    let box_input = BoundedTensor::new(lower, upper).expect("input box");

    let mid_batched = mid
        .clone()
        .into_shape_with_order(ndarray::IxDyn(&[1, num_inputs]))
        .expect("batched midpoint");
    let ort_y = diff::run_inference_bytes(&bytes, &mid_batched).expect("ORT forward");
    let ort_y = ort_y.first().expect("ORT output");

    let graph_folded = model_folded
        .to_graph_network()
        .expect("graph with fold enabled");
    let graph_raw = model_raw
        .to_graph_network()
        .expect("graph with raw BatchNorm");

    let point_box = BoundedTensor::new(mid.clone(), mid).expect("point box");
    let point_out = graph_folded
        .propagate_ibp(&point_box)
        .expect("point IBP with fold enabled");
    for (idx, ((l, u), y)) in point_out
        .lower()
        .iter()
        .zip(point_out.upper().iter())
        .zip(ort_y.iter())
        .enumerate()
    {
        let tolerance = FOLD_TOL_REL * y.abs().max(1.0);
        assert!(
            (l - tolerance) <= *y && *y <= (u + tolerance),
            "elem {idx}: ORT {y} escapes folded point-IBP [{l}, {u}] (tol {tolerance})"
        );
    }
    println!("point enclosure vs ORT: OK ({} outputs)", ort_y.len());

    let root_folded = graph_folded
        .propagate_ibp(&box_input)
        .expect("root IBP with fold enabled");
    let root_raw = graph_raw
        .propagate_ibp(&box_input)
        .expect("root IBP with raw BatchNorm");
    for (idx, (((l_folded, u_folded), l_raw), u_raw)) in root_folded
        .lower()
        .iter()
        .zip(root_folded.upper().iter())
        .zip(root_raw.lower().iter())
        .zip(root_raw.upper().iter())
        .enumerate()
    {
        println!(
            "root IBP Y_{idx}: folded [{l_folded:.6e}, {u_folded:.6e}]  \
             raw [{l_raw:.6e}, {u_raw:.6e}]"
        );
        let slack_l = FOLD_TOL_REL * l_raw.abs().max(1.0);
        let slack_u = FOLD_TOL_REL * u_raw.abs().max(1.0);
        assert!(
            *l_folded >= l_raw - slack_l && *u_folded <= u_raw + slack_u,
            "elem {idx}: folded root bounds looser than unfolded beyond fold tolerance"
        );
    }
}
