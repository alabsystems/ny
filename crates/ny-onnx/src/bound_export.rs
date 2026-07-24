// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bound export: serialize per-node IBP bounds to safetensors format.
//!
//! This module lets external verifier consumers ingest ny's computed bounds
//! as training signals (verification-guided fine-tuning).
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ny_onnx::bound_export::export_bounds_safetensors;
//! use ny_onnx::load_onnx;
//!
//! let model = load_onnx("model.onnx")?;
//! let graph = model.to_graph_network()?;
//! // Input box [-1, 1] on every pixel: center 0 with epsilon 1.
//! let center = ndarray::ArrayD::zeros(ndarray::IxDyn(&[1, 3, 224, 224]));
//! let input = ny_tensor::BoundedTensor::from_epsilon(center, 1.0)?;
//! let node_bounds = graph.collect_node_bounds(&input)?;
//! export_bounds_safetensors(&node_bounds, "bounds.safetensors")?;
//! # Ok::<(), ny_core::NyError>(())
//! ```

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use safetensors::tensor::{serialize, TensorView};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Export per-node IBP bounds to safetensors format.
///
/// Each node produces two tensors in the output file:
/// - `{node_name}.lower` — lower bound array (f32)
/// - `{node_name}.upper` — upper bound array (f32)
///
/// The output file can be loaded by Python via `safetensors.numpy.load_file()`
/// or `safetensors.torch.load_file()`.
pub fn export_bounds_safetensors(
    node_bounds: &HashMap<String, BoundedTensor>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let bytes = serialize_bounds_to_bytes(node_bounds)?;

    let mut file = std::fs::File::create(path.as_ref()).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to create bounds file {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    file.write_all(&bytes).map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to write bounds file {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    Ok(())
}

/// Serialize per-node IBP bounds to safetensors bytes (in-memory).
///
/// Useful for Python bindings or network transfer without touching disk.
pub fn serialize_bounds_to_bytes(node_bounds: &HashMap<String, BoundedTensor>) -> Result<Vec<u8>> {
    // Collect all tensor data upfront so references remain valid.
    let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    // Sort node names for deterministic output.
    let mut node_names: Vec<&String> = node_bounds.keys().collect();
    node_names.sort();

    for name in &node_names {
        let bounds = &node_bounds[*name];
        let shape: Vec<usize> = bounds.shape().to_vec();

        let lower_bytes: Vec<u8> = bounds
            .lower()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let upper_bytes: Vec<u8> = bounds
            .upper()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        tensor_data.push((format!("{}.lower", name), lower_bytes, shape.clone()));
        tensor_data.push((format!("{}.upper", name), upper_bytes, shape));
    }

    // Build TensorView references.
    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, data, shape)| {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), data)
                .map_err(|e| NyError::ModelLoad(format!("Safetensors view error: {}", e)))?;
            Ok((name.as_str(), view))
        })
        .collect::<Result<Vec<_>>>()?;

    // The safetensors::tensor::serialize function takes Vec<(impl AsRef<str>, TensorView)>.
    let serializable: Vec<(&str, &TensorView<'_>)> =
        views.iter().map(|(name, view)| (*name, view)).collect();

    serialize(serializable, None)
        .map_err(|e| NyError::ModelLoad(format!("Safetensors serialize error: {}", e)))
}

/// Summary of exported bounds: node count, total tensors, and per-node width stats.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BoundExportSummary {
    pub num_nodes: usize,
    pub num_tensors: usize,
    pub total_elements: usize,
    pub total_bytes: usize,
    pub nodes: Vec<NodeBoundSummary>,
}

/// Per-node summary of exported bounds.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeBoundSummary {
    pub name: String,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub max_width: f32,
    pub mean_width: f32,
}

/// Generate a summary of per-node bounds without writing to disk.
pub fn summarize_bounds(node_bounds: &HashMap<String, BoundedTensor>) -> BoundExportSummary {
    let mut nodes: Vec<NodeBoundSummary> = node_bounds
        .iter()
        .map(|(name, bounds)| {
            let widths = bounds.width();
            let num_elements = widths.len();
            let max_width = bounds.max_width();
            let mean_width = if num_elements > 0 {
                widths.iter().sum::<f32>() / num_elements as f32
            } else {
                0.0
            };
            NodeBoundSummary {
                name: name.clone(),
                shape: bounds.shape().to_vec(),
                num_elements,
                max_width,
                mean_width,
            }
        })
        .collect();
    nodes.sort_by(|a, b| a.name.cmp(&b.name));

    let total_elements: usize = nodes.iter().map(|n| n.num_elements).sum();
    // 2 tensors per node (lower + upper), f32 = 4 bytes per element.
    let total_bytes = total_elements * 4 * 2;

    BoundExportSummary {
        num_nodes: nodes.len(),
        num_tensors: nodes.len() * 2,
        total_elements,
        total_bytes,
        nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn make_test_bounds() -> HashMap<String, BoundedTensor> {
        let mut bounds = HashMap::new();
        bounds.insert(
            "layer_0".to_string(),
            BoundedTensor::new(arr1(&[-1.5, -0.5]).into_dyn(), arr1(&[1.5, 0.5]).into_dyn())
                .unwrap(),
        );
        bounds.insert(
            "layer_1".to_string(),
            BoundedTensor::new(
                arr1(&[-2.0, -1.0, 0.0]).into_dyn(),
                arr1(&[2.0, 1.0, 0.5]).into_dyn(),
            )
            .unwrap(),
        );
        bounds
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_export_bounds_roundtrip() {
        let node_bounds = make_test_bounds();

        let bytes = serialize_bounds_to_bytes(&node_bounds).expect("serialize");
        assert!(!bytes.is_empty(), "serialized bytes should be non-empty");

        // Verify we can parse back with safetensors.
        let loaded = safetensors::SafeTensors::deserialize(&bytes).expect("deserialize");
        for (name, bounds) in &node_bounds {
            let lower_name = format!("{}.lower", name);
            let upper_name = format!("{}.upper", name);
            let lower_view = loaded.tensor(&lower_name).expect("lower tensor");
            let upper_view = loaded.tensor(&upper_name).expect("upper tensor");
            assert_eq!(lower_view.shape(), bounds.shape());
            assert_eq!(upper_view.shape(), bounds.shape());

            // Verify actual values roundtrip correctly.
            let lower_data: Vec<f32> = lower_view
                .data()
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            for (got, expected) in lower_data.iter().zip(bounds.lower().iter()) {
                assert!(
                    (got - expected).abs() < f32::EPSILON,
                    "lower mismatch: {} vs {}",
                    got,
                    expected
                );
            }
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_export_bounds_to_file() {
        let node_bounds = make_test_bounds();

        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        let path = tmp.path().to_path_buf();
        export_bounds_safetensors(&node_bounds, &path).expect("export");

        let file_bytes = std::fs::read(&path).expect("read file");
        assert!(!file_bytes.is_empty());
        let loaded = safetensors::SafeTensors::deserialize(&file_bytes).expect("deserialize");
        // 2 nodes × 2 tensors (lower + upper) = 4 tensors.
        assert_eq!(loaded.len(), 4, "expected 4 tensors in file");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_summarize_bounds() {
        let node_bounds = make_test_bounds();
        let summary = summarize_bounds(&node_bounds);

        assert_eq!(summary.num_nodes, 2);
        assert_eq!(summary.num_tensors, 4);
        assert_eq!(summary.total_elements, 5); // 2 + 3
        assert_eq!(summary.total_bytes, 5 * 4 * 2); // 5 elements × 4 bytes × 2 (lower+upper)
        for node in &summary.nodes {
            assert!(node.max_width >= 0.0);
            assert!(node.mean_width >= 0.0);
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_export_empty_bounds() {
        let bounds: HashMap<String, BoundedTensor> = HashMap::new();
        let bytes = serialize_bounds_to_bytes(&bounds).expect("serialize empty");
        let loaded = safetensors::SafeTensors::deserialize(&bytes).expect("deserialize empty");
        assert_eq!(loaded.len(), 0);
    }
}
