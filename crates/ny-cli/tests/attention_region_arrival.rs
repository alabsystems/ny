// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original-loader arrival gate for the default directional region consumer.
//!
//! This is not a VNN-COMP solve, runtime comparison or numerical-execution
//! contract qualification. It checks the pinned 2026 2.0 occurrence 169 input
//! geometry, ordinary graph builder, and actual production backward consumer.
//! Missing/changed fixtures and no-arrival are failures, never ignored passes.
//! `NY_VIT_BENCH` may point to the staged category's exact `2.0` directory.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::{parse_vnnlib, OutputConstraint};
use ny_propagate::Layer;
use ny_tensor::BoundedTensor;
use sha2::{Digest, Sha256};

const COMMIT: &str = "303f2327a8e8411efc4457da03d2016ce42733cf";
const MODEL: &str = "onnx/ibp_3_3_8.onnx";
const MODEL_SHA: &str = "9cc53b9edb35d40d70de1d816008a434c7e60b0404cf3cb38b9aefc189692883";
const PROPERTY: &str = "vnnlib/ibp_3_3_8_5899.vnnlib";
const PROPERTY_SHA: &str = "d3d9bc9c9ee61bd48bfd7378ed770875422e65d612cdf412c6e84005d3b1001c";

fn fixture_dir() -> PathBuf {
    let inputs = ny_levers::RawLeverInputs::capture(ny_levers::all());
    inputs
        .get(&ny_levers::decls::onnx::VIT_BENCH)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new("<home>/ny-model-cache/high-impact/vnncomp2026-vit")
                .join(COMMIT)
                .join("2.0")
        })
}

fn original_bytes(path: &Path, expected_bytes: u64, expected_sha: &str) -> Vec<u8> {
    let metadata = fs::metadata(path).unwrap_or_else(|error| panic!(
        "explicit original-fixture refusal: {} unavailable ({error}); stage the pinned official {COMMIT} files before this gate",path.display()));
    assert!(
        metadata.is_file(),
        "fixture must be a regular original file: {}",
        path.display()
    );
    assert_eq!(
        metadata.len(),
        expected_bytes,
        "original fixture size changed: {}",
        path.display()
    );
    let bytes = fs::read(path).expect("bounded original fixture read");
    assert_eq!(
        bytes.len() as u64,
        expected_bytes,
        "original fixture changed during read"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        expected_sha,
        "original fixture hash changed: {}",
        path.display()
    );
    bytes
}

fn authored_pv_count(graph: &ny_propagate::GraphNetwork) -> usize {
    graph
        .node_names()
        .iter()
        .filter(|name| {
            let node = graph.node(name).unwrap();
            if node.inputs().len() != 2
                || !matches!(node.layer(), Layer::MatMul(_) | Layer::BilinearCrown(_))
            {
                return false;
            }
            graph
                .node(&node.inputs()[0])
                .is_some_and(|producer| matches!(producer.layer(), Layer::Softmax(_)))
        })
        .count()
}

#[test]
#[ntest::timeout(120000)]
fn attention_region_original_2026_ibp5899_default_batched_consumer_arrives() {
    // Preserve the original query and its budget while retaining the exact
    // optional-corridor refusal that otherwise precedes a layout error.
    // This subscriber changes observation only; no environment arms a route.
    let _ = tracing_subscriber::fmt()
        .with_env_filter("ny_propagate::network::graph_crown::attention_region::global=debug")
        .with_test_writer()
        .try_init();
    let fixture = fixture_dir();
    let model_path = fixture.join(MODEL);
    let property_path = fixture.join(PROPERTY);
    let model_bytes = original_bytes(&model_path, 303176, MODEL_SHA);
    let property_bytes = original_bytes(&property_path, 286418, PROPERTY_SHA);
    let model = ny_onnx::load_onnx_bytes("pinned-vit2026-ibp-3-3-8", &model_bytes)
        .expect("original model loader");
    let spec = parse_vnnlib(std::str::from_utf8(&property_bytes).expect("original vnnlib UTF8"))
        .expect("original 2.0 property parser");
    // The original parser retains the authored angle-bracket version marker.
    assert_eq!(spec.version.as_deref(), Some("<2.0>"));
    assert_eq!((spec.num_inputs, spec.num_outputs), (3072, 10));
    assert!(spec.is_disjunction);
    assert_eq!(spec.output_constraint_clauses.len(), 9);
    let mut competitors = Vec::new();
    for clause in &spec.output_constraint_clauses {
        assert_eq!(clause.len(), 1);
        match &clause[0] {
            OutputConstraint::GreaterEq(c, 1) => competitors.push(*c),
            OutputConstraint::LessEq(1, c) => competitors.push(*c),
            other => panic!("pinned original unsafe alternative changed: {other:?}"),
        }
    }
    competitors.sort_unstable();
    assert_eq!(competitors, (0..10).filter(|&i| i != 1).collect::<Vec<_>>());
    let graph = model
        .to_graph_network()
        .expect("ordinary original graph builder");
    assert_eq!(
        authored_pv_count(&graph),
        3,
        "the original graph must retain all3 decomposed producers"
    );
    assert!(
        graph
            .node_names()
            .iter()
            .all(|name| !matches!(graph.node(name).unwrap().layer(), Layer::SelfAttention(_))),
        "this gate exercises original decomposed operators, not inference fusion"
    );
    let (lower, upper) = spec.split_input_bounds_f32();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), upper).unwrap(),
    )
    .unwrap();
    // 100s is the exact official ordered occurrence's limit. This bounded
    // correctness probe does not run the official launcher or assert a verdict.
    let deadline = Instant::now() + Duration::from_secs(100);
    let (result, receipt) = graph
        .propagate_crown_batched_with_attention_region_receipt_and_deadline(&input, Some(deadline))
        .expect("bounded default original graph propagation");
    eprintln!("[attention-region original-arrival] commit={COMMIT} occurrence=169zeroBased model={MODEL} model_sha={MODEL_SHA} property={PROPERTY} property_sha={PROPERTY_SHA} timeout_s=100 receipt={receipt:?} provenance={:?}",result.provenance);
    assert!(receipt.batched>0 && receipt.composed>0,"the actual original default consumer must arrive, rather than merely recognize metadata: {receipt:?}");
    assert!(
        receipt.rows >= 10,
        "ordinary logit property rows must not be reinterpreted as heads: {receipt:?}"
    );
    assert_eq!(result.bounds.len(), 10);
    for (&lo, &hi) in result.bounds.lower().iter().zip(result.bounds.upper()) {
        assert!(!lo.is_nan() && !hi.is_nan() && lo <= hi);
    }
    // Recheck the immutable original files after the query; no stale or replaced
    // input can silently acquire a matched-fixture receipt.
    assert_eq!(original_bytes(&model_path, 303176, MODEL_SHA), model_bytes);
    assert_eq!(
        original_bytes(&property_path, 286418, PROPERTY_SHA),
        property_bytes
    );
}

#[test]
fn attention_region_original_2026_pgd_loader_retains_authored_producers() {
    let path = fixture_dir().join("onnx/pgd_2_3_16.onnx");
    let hash = "246326387574617a274b6f73f7f52771df1195e03c99065ce4ad18c46b8e0984";
    let bytes = original_bytes(&path, 325647, hash);
    let model = ny_onnx::load_onnx_bytes("pinned-vit2026-pgd-2-3-16", &bytes)
        .expect("original pgd model loader");
    let graph = model
        .to_graph_network()
        .expect("ordinary original pgd graph builder");
    assert_eq!(authored_pv_count(&graph), 2);
    assert_eq!(original_bytes(&path, 325647, hash), bytes);
    // No matching PGD property is staged here; topology alone is not production
    // consumer arrival, an official decision, or a transferable score claim.
}
