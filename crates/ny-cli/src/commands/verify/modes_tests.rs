// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::commands::backend::{BackendRequest, BackendRequestSource, ProofBackendReceipt};
use ndarray::arr1;
use ny_propagate::layers::ReLULayer;
use ny_propagate::{GraphNode, Layer, PropagationMethod};
use ny_tensor::BoundedTensor;
use std::fs;
use std::path::{Path, PathBuf};

fn bounded_input() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input")
}

fn cpu_backend_receipt() -> ProofBackendReceipt {
    ProofBackendReceipt::cpu(
        BackendRequest {
            backend: BackendArg::Cpu,
            source: BackendRequestSource::DefaultedCliValue,
            selection_reason: None,
        },
        "cpu",
    )
}

fn write_model_file(dir: &Path, name: &str) -> PathBuf {
    let model_path = dir.join(name);
    fs::write(&model_path, b"ny-cli modes checkpoint test")
        .expect("model fixture should be writable");
    model_path
}

fn single_block_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("layer0_relu", Layer::ReLU(ReLULayer)));
    graph.set_output("layer0_relu");
    graph
}

fn two_block_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("layer0_relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "layer1_relu",
        Layer::ReLU(ReLULayer),
        vec!["layer0_relu".to_string()],
    ));
    graph.set_output("layer1_relu");
    graph
}

#[test]
fn test_run_block_wise_graph_writes_checkpoint_file_4314() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let model_path = write_model_file(tempdir.path(), "model.onnx");
    let checkpoint_path = tempdir.path().join("checkpoint.json");
    let graph = single_block_graph();
    let input = bounded_input();

    run_block_wise_graph(
        &graph,
        &input,
        0.1,
        PropagationMethod::Ibp,
        BackendArg::Cpu,
        &model_path,
        false,
        false,
        0,
        Some(checkpoint_path.as_path()),
        false,
        &cpu_backend_receipt(),
    )
    .expect("block-wise CLI mode should write a checkpoint");

    let checkpoint =
        VerificationCheckpoint::load(&checkpoint_path).expect("checkpoint file should be readable");
    assert_eq!(checkpoint.model_path, model_path);
    assert_eq!(checkpoint.method, "ibp");
    assert_eq!(checkpoint.backend, "cpu");
    assert_eq!(checkpoint.total_blocks, 1);
    assert_eq!(checkpoint.next_block_index, 1);
    assert_eq!(checkpoint.completed_blocks.len(), 1);
    assert_eq!(checkpoint.completed_blocks[0].block_name, "layer0");
}

#[test]
fn test_run_block_wise_graph_resumes_existing_checkpoint_without_resetting_4314() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let model_path = write_model_file(tempdir.path(), "resume-model.onnx");
    let checkpoint_path = tempdir.path().join("resume-checkpoint.json");
    let graph = two_block_graph();
    let input = bounded_input();
    let full_result = graph
        .propagate_ibp_block_wise(&input, 0.1)
        .expect("graph fixture should support block-wise IBP");
    assert_eq!(
        full_result.total_blocks, 2,
        "fixture must expose two blocks"
    );

    let mut checkpoint = VerificationCheckpoint::new(
        model_path.clone(),
        compute_model_hash(&model_path).expect("model hash"),
        0.1,
        "ibp",
        "cpu",
        full_result.total_blocks,
    );
    checkpoint.update(full_result.blocks[0].clone(), 12);
    checkpoint
        .save(&checkpoint_path)
        .expect("seed checkpoint should save");

    run_block_wise_graph(
        &graph,
        &input,
        0.1,
        PropagationMethod::Ibp,
        BackendArg::Cpu,
        &model_path,
        false,
        false,
        0,
        Some(checkpoint_path.as_path()),
        false,
        &cpu_backend_receipt(),
    )
    .expect("block-wise CLI mode should resume from the saved checkpoint");

    let resumed_checkpoint = VerificationCheckpoint::load(&checkpoint_path)
        .expect("resumed checkpoint should still be readable");
    assert_eq!(resumed_checkpoint.total_blocks, 2);
    assert_eq!(resumed_checkpoint.next_block_index, 2);
    assert_eq!(resumed_checkpoint.completed_blocks.len(), 2);
    assert_eq!(resumed_checkpoint.completed_blocks[0].block_name, "layer0");
    assert_eq!(resumed_checkpoint.completed_blocks[1].block_name, "layer1");
}
