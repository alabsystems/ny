// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::repr::repr_string;
use crate::utils::validate_tolerance;
use ny_core::nan_propagating_max;
#[cfg(feature = "coreml")]
use ny_onnx::coreml::load_coreml;
#[cfg(feature = "gguf")]
use ny_onnx::gguf::load_gguf;
use ny_onnx::load_onnx;
use ny_onnx::native::load_weights;
#[cfg(feature = "pytorch")]
use ny_onnx::pytorch::load_pytorch;
use ny_onnx::safetensors::load_safetensors;
use ny_onnx::WeightStore;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Information about a tensor in a weight file.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct TensorInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub elements: usize,
}

#[pymethods]
impl TensorInfo {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "TensorInfo(name={}, shape={:?}, elements={})",
            repr_string(&self.name),
            self.shape,
            self.elements
        )
    }
}

/// Result of weight file inspection.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct WeightsInfo {
    #[pyo3(get)]
    pub format: String,
    #[pyo3(get)]
    pub tensor_count: usize,
    #[pyo3(get)]
    pub total_params: usize,
    #[pyo3(get)]
    pub tensors: Vec<TensorInfo>,
}

#[pymethods]
impl WeightsInfo {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "WeightsInfo(format={}, tensors={}, params={})",
            repr_string(&self.format),
            self.tensor_count,
            self.total_params
        )
    }

    /// Get a formatted summary.
    pub(crate) fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Weights Info".to_string());
        lines.push("============".to_string());
        lines.push(format!("Format: {}", self.format));
        lines.push(format!("Tensors: {}", self.tensor_count));
        lines.push(format!(
            "Parameters: {} ({:.2}M)",
            self.total_params,
            self.total_params as f64 / 1e6
        ));
        lines.push("\nTensors:".to_string());
        for t in self.tensors.iter().take(20) {
            lines.push(format!(
                "  {}: {:?} ({} elements)",
                t.name, t.shape, t.elements
            ));
        }
        if self.tensors.len() > 20 {
            lines.push(format!("  ... and {} more", self.tensors.len() - 20));
        }
        lines.join("\n")
    }
}

/// Get information about weights in a file.
///
/// Supports ONNX (.onnx), SafeTensors (.safetensors), PyTorch (.pt, .pth, .bin),
/// GGUF (.gguf), and CoreML (.mlmodel, .mlpackage) formats.
/// Also supports directories containing sharded SafeTensors or HuggingFace PyTorch checkpoints.
///
/// Args:
///     path: Path to weights file
///
/// Returns:
///     WeightsInfo with tensor information
///
/// Example:
///     >>> info = ny.weights_info("model.safetensors")
///     >>> print(f"Total params: {info.total_params:,}")
///     >>> for t in info.tensors[:5]:
///     ...     print(f"  {t.name}: {t.shape}")
fn has_sharded_pytorch_bins(path: &std::path::Path) -> bool {
    std::fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.file_name().to_str().is_some_and(|n| {
                    n.starts_with("pytorch_model-")
                        && std::path::Path::new(n)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
                })
            })
        })
        .unwrap_or(false)
}

#[pyfunction]
pub fn weights_info(py: Python<'_>, path: &str) -> PyResult<WeightsInfo> {
    let path = std::path::Path::new(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check if it's an mlpackage directory (no extension check needed)
    let is_mlpackage = path.is_dir()
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(".mlpackage"))
            .unwrap_or(false);

    // Check if it's a directory (for sharded SafeTensors, HuggingFace model directories, etc.)
    let is_directory = path.is_dir() && !is_mlpackage;

    let directory_format = if is_directory {
        // Mirror ny_onnx::native directory format detection for better UX.
        let config_json = path.join("config.json");
        let has_safetensors = std::fs::read_dir(path)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
            })
            .unwrap_or(false);

        // Check for sharded PyTorch models (index json or numbered bin files)
        let has_sharded_pytorch =
            path.join("pytorch_model.bin.index.json").exists() || has_sharded_pytorch_bins(path);

        if config_json.exists() && has_safetensors {
            "SafeTensors (sharded)".to_string()
        } else if has_sharded_pytorch {
            "PyTorch (sharded)".to_string()
        } else if path.join("pytorch_model.bin").exists() || path.join("model.pt").exists() {
            "PyTorch (checkpoint)".to_string()
        } else {
            // Best-effort; ny_onnx::native::load_weights will provide the real error if unsupported.
            "SafeTensors (sharded)".to_string()
        }
    } else {
        String::new()
    };

    let (format, weights) = Python::detach(py, || -> ny_core::Result<(String, WeightStore)> {
        // Handle mlpackage directories first
        if is_mlpackage {
            #[cfg(feature = "coreml")]
            {
                let weights = load_coreml(path)?;
                return Ok(("CoreML".to_string(), weights));
            }
            #[cfg(not(feature = "coreml"))]
            {
                return Err(ny_core::NyError::ModelLoad(
                    "CoreML support not enabled. Rebuild with --features coreml".to_string(),
                ));
            }
        }

        // Handle directories (sharded SafeTensors, HuggingFace model directories, HuggingFace checkpoints)
        if is_directory {
            let weights = load_weights(path)?;
            return Ok((directory_format.clone(), weights));
        }

        match ext.as_str() {
            "safetensors" => {
                let weights = load_safetensors(path)?;
                Ok(("SafeTensors".to_string(), weights))
            }
            "onnx" => {
                let model = load_onnx(path)?;
                Ok(("ONNX".to_string(), model.weights))
            }
            #[cfg(feature = "pytorch")]
            "pt" | "pth" | "bin" => {
                let weights = load_pytorch(path)?;
                Ok(("PyTorch".to_string(), weights))
            }
            #[cfg(not(feature = "pytorch"))]
            "pt" | "pth" | "bin" => Err(ny_core::NyError::ModelLoad(
                "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
            )),
            #[cfg(feature = "gguf")]
            "gguf" => {
                let weights = load_gguf(path)?;
                Ok(("GGUF".to_string(), weights))
            }
            #[cfg(not(feature = "gguf"))]
            "gguf" => Err(ny_core::NyError::ModelLoad(
                "GGUF support not enabled. Rebuild with --features gguf".to_string(),
            )),
            #[cfg(feature = "coreml")]
            "mlmodel" => {
                let weights = load_coreml(path)?;
                Ok(("CoreML".to_string(), weights))
            }
            #[cfg(not(feature = "coreml"))]
            "mlmodel" => Err(ny_core::NyError::ModelLoad(
                "CoreML support not enabled. Rebuild with --features coreml".to_string(),
            )),
            _ => Err(ny_core::NyError::ModelLoad(format!(
                "Unsupported format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .gguf, .mlmodel, .mlpackage, or a directory with SafeTensors shards",
                ext
            ))),
        }
    })
    .map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;

    let mut tensors = Vec::new();
    let mut total_params = 0usize;

    for (name, tensor) in weights.iter() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let elements = shape.iter().product();
        total_params += elements;
        tensors.push(TensorInfo {
            name: name.to_string(),
            shape,
            elements,
        });
    }

    // Sort by name for consistent output
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(WeightsInfo {
        format,
        tensor_count: tensors.len(),
        total_params,
        tensors,
    })
}

/// Result of comparing a single tensor between two weight files.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TensorComparisonStatus {
    Match,
    Differs,
    ShapeMismatch,
    MissingInA,
    MissingInB,
}

#[pymethods]
impl TensorComparisonStatus {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            TensorComparisonStatus::Match => "TensorComparisonStatus.Match".to_string(),
            TensorComparisonStatus::Differs => "TensorComparisonStatus.Differs".to_string(),
            TensorComparisonStatus::ShapeMismatch => {
                "TensorComparisonStatus.ShapeMismatch".to_string()
            }
            TensorComparisonStatus::MissingInA => "TensorComparisonStatus.MissingInA".to_string(),
            TensorComparisonStatus::MissingInB => "TensorComparisonStatus.MissingInB".to_string(),
        }
    }

    pub(crate) fn __str__(&self) -> String {
        self.as_str().to_string()
    }
}

impl TensorComparisonStatus {
    fn as_str(&self) -> &'static str {
        match self {
            TensorComparisonStatus::Match => "match",
            TensorComparisonStatus::Differs => "differs",
            TensorComparisonStatus::ShapeMismatch => "shape_mismatch",
            TensorComparisonStatus::MissingInA => "missing_in_a",
            TensorComparisonStatus::MissingInB => "missing_in_b",
        }
    }
}

/// Result of comparing a single tensor between two weight files.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct TensorComparison {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub status: TensorComparisonStatus,
    #[pyo3(get)]
    pub max_diff: Option<f32>,
    #[pyo3(get)]
    pub shape_a: Option<Vec<usize>>,
    #[pyo3(get)]
    pub shape_b: Option<Vec<usize>>,
}

#[pymethods]
impl TensorComparison {
    pub(crate) fn __repr__(&self) -> String {
        match &self.max_diff {
            Some(diff) => format!(
                "TensorComparison(name={}, status={}, max_diff={:.2e})",
                repr_string(&self.name),
                repr_string(self.status.as_str()),
                diff
            ),
            None => format!(
                "TensorComparison(name={}, status={})",
                repr_string(&self.name),
                repr_string(self.status.as_str())
            ),
        }
    }
}

/// Result of comparing two weight files.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct WeightsDiffResult {
    #[pyo3(get)]
    pub is_match: bool,
    #[pyo3(get)]
    pub max_diff: f32,
    #[pyo3(get)]
    pub tolerance: f32,
    #[pyo3(get)]
    pub differing_count: usize,
    #[pyo3(get)]
    pub total_tensors_a: usize,
    #[pyo3(get)]
    pub total_tensors_b: usize,
    #[pyo3(get)]
    pub comparisons: Vec<TensorComparison>,
}

#[pymethods]
impl WeightsDiffResult {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "WeightsDiffResult(is_match={}, max_diff={:.2e}, differing={})",
            self.is_match, self.max_diff, self.differing_count
        )
    }

    /// Get a formatted summary.
    pub(crate) fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Weights Diff Result".to_string());
        lines.push("===================".to_string());
        lines.push(format!(
            "Result: {}",
            if self.is_match { "MATCH" } else { "DIFFERS" }
        ));
        lines.push(format!("Max difference: {:.6e}", self.max_diff));
        lines.push(format!("Tolerance: {:.6e}", self.tolerance));
        lines.push(format!("Differing tensors: {}", self.differing_count));
        lines.push(format!("Tensors in A: {}", self.total_tensors_a));
        lines.push(format!("Tensors in B: {}", self.total_tensors_b));

        if self.differing_count > 0 {
            lines.push("\nDifferences:".to_string());
            for c in self
                .comparisons
                .iter()
                .filter(|c| c.status != TensorComparisonStatus::Match)
                .take(20)
            {
                match &c.max_diff {
                    Some(diff) => lines.push(format!(
                        "  {}: {} (diff={:.2e})",
                        c.name,
                        c.status.as_str(),
                        diff
                    )),
                    None => lines.push(format!("  {}: {}", c.name, c.status.as_str())),
                }
            }
        }

        lines.join("\n")
    }

    /// Get matching tensors.
    fn matching_tensors(&self) -> Vec<TensorComparison> {
        self.comparisons
            .iter()
            .filter(|c| c.status == TensorComparisonStatus::Match)
            .cloned()
            .collect()
    }

    /// Get differing tensors.
    fn differing_tensors(&self) -> Vec<TensorComparison> {
        self.comparisons
            .iter()
            .filter(|c| c.status != TensorComparisonStatus::Match)
            .cloned()
            .collect()
    }
}

/// Load weights from a file or directory (helper for weights_diff).
///
/// Supports single files (.safetensors, .onnx, .pt, .pth, .bin, .gguf, .mlmodel, .mlpackage)
/// and directories containing sharded SafeTensors files, HuggingFace model directories, or HuggingFace PyTorch checkpoints.
fn load_weights_from_file(path: &str) -> ny_core::Result<WeightStore> {
    let path = std::path::Path::new(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check if it's an mlpackage directory
    let is_mlpackage = path.is_dir()
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(".mlpackage"))
            .unwrap_or(false);

    // Check if it's a directory (for sharded SafeTensors, HuggingFace model directories, etc.)
    let is_directory = path.is_dir() && !is_mlpackage;

    if is_mlpackage {
        #[cfg(feature = "coreml")]
        {
            return load_coreml(path);
        }
        #[cfg(not(feature = "coreml"))]
        {
            return Err(ny_core::NyError::ModelLoad(
                "CoreML support not enabled. Rebuild with --features coreml".to_string(),
            ));
        }
    }

    // Handle directories (sharded SafeTensors, HuggingFace model directories, HuggingFace checkpoints)
    if is_directory {
        return load_weights(path);
    }

    match ext.as_str() {
        "safetensors" => load_safetensors(path),
        "onnx" => {
            let model = load_onnx(path)?;
            Ok(model.weights)
        }
        #[cfg(feature = "pytorch")]
        "pt" | "pth" | "bin" => load_pytorch(path),
        #[cfg(not(feature = "pytorch"))]
        "pt" | "pth" | "bin" => Err(ny_core::NyError::ModelLoad(
            "PyTorch support not enabled. Rebuild with --features pytorch".to_string(),
        )),
        #[cfg(feature = "gguf")]
        "gguf" => load_gguf(path),
        #[cfg(not(feature = "gguf"))]
        "gguf" => Err(ny_core::NyError::ModelLoad(
            "GGUF support not enabled. Rebuild with --features gguf".to_string(),
        )),
        #[cfg(feature = "coreml")]
        "mlmodel" => load_coreml(path),
        #[cfg(not(feature = "coreml"))]
        "mlmodel" => Err(ny_core::NyError::ModelLoad(
            "CoreML support not enabled. Rebuild with --features coreml".to_string(),
        )),
        _ => Err(ny_core::NyError::ModelLoad(format!(
            "Unsupported format: {}. Use .safetensors, .onnx, .pt, .pth, .bin, .gguf, .mlmodel, .mlpackage, or a directory with SafeTensors/PyTorch shards",
            ext
        ))),
    }
}

/// Treat non-finite arithmetic as a difference, never as a match.
///
/// In particular, `NaN > tolerance` is false, so a plain comparison would
/// otherwise fail open for tensors containing NaN (including `inf - inf`).
fn difference_exceeds_tolerance(diff: f32, tolerance: f32) -> bool {
    !diff.is_finite() || diff > tolerance
}

/// Compare weights between two files.
///
/// Supports ONNX (.onnx), SafeTensors (.safetensors), PyTorch (.pt, .pth, .bin),
/// GGUF (.gguf), and CoreML (.mlmodel, .mlpackage) formats.
///
/// Args:
///     file_a: Path to first weights file
///     file_b: Path to second weights file
///     tolerance: Maximum allowed absolute difference (default: 1e-6)
///
/// Returns:
///     WeightsDiffResult with comparison results
///
/// Example:
///     >>> result = ny.weights_diff("model_a.safetensors", "model_b.safetensors")
///     >>> assert result.is_match, f"Max diff: {result.max_diff:.2e}"
///     >>> for diff in result.differing_tensors():
///     ...     print(f"  {diff.name}: {diff.status}")
#[pyfunction]
#[pyo3(signature = (file_a, file_b, tolerance=1e-6))]
pub fn weights_diff(
    py: Python<'_>,
    file_a: &str,
    file_b: &str,
    tolerance: f32,
) -> PyResult<WeightsDiffResult> {
    validate_tolerance(tolerance)?;

    let result = Python::detach(py, || -> ny_core::Result<WeightsDiffResult> {
        let weights_a = load_weights_from_file(file_a)?;
        let weights_b = load_weights_from_file(file_b)?;

        let mut comparisons = Vec::new();
        let mut max_diff = 0.0f32;
        let mut differing_count = 0;

        // Compare tensors in A
        for (name, tensor_a) in weights_a.iter() {
            if let Some(tensor_b) = weights_b.get(name) {
                // Compare shapes
                if tensor_a.shape() != tensor_b.shape() {
                    comparisons.push(TensorComparison {
                        name: name.to_string(),
                        status: TensorComparisonStatus::ShapeMismatch,
                        max_diff: None,
                        shape_a: Some(tensor_a.shape().to_vec()),
                        shape_b: Some(tensor_b.shape().to_vec()),
                    });
                    differing_count += 1;
                    continue;
                }

                // Compare values
                let diff = tensor_a
                    .iter()
                    .zip(tensor_b.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, nan_propagating_max);

                max_diff = nan_propagating_max(max_diff, diff);

                if difference_exceeds_tolerance(diff, tolerance) {
                    differing_count += 1;
                    comparisons.push(TensorComparison {
                        name: name.to_string(),
                        status: TensorComparisonStatus::Differs,
                        max_diff: Some(diff),
                        shape_a: Some(tensor_a.shape().to_vec()),
                        shape_b: Some(tensor_b.shape().to_vec()),
                    });
                } else {
                    comparisons.push(TensorComparison {
                        name: name.to_string(),
                        status: TensorComparisonStatus::Match,
                        max_diff: Some(diff),
                        shape_a: Some(tensor_a.shape().to_vec()),
                        shape_b: Some(tensor_b.shape().to_vec()),
                    });
                }
            } else {
                comparisons.push(TensorComparison {
                    name: name.to_string(),
                    status: TensorComparisonStatus::MissingInB,
                    max_diff: None,
                    shape_a: Some(tensor_a.shape().to_vec()),
                    shape_b: None,
                });
                differing_count += 1;
            }
        }

        // Check for tensors only in B
        for name in weights_b.keys() {
            if weights_a.get(name).is_none() {
                comparisons.push(TensorComparison {
                    name: name.to_string(),
                    status: TensorComparisonStatus::MissingInA,
                    max_diff: None,
                    shape_a: None,
                    shape_b: weights_b.get(name).map(|t| t.shape().to_vec()),
                });
                differing_count += 1;
            }
        }

        let is_match = differing_count == 0;

        Ok(WeightsDiffResult {
            is_match,
            max_diff,
            tolerance,
            differing_count,
            total_tensors_a: weights_a.len(),
            total_tensors_b: weights_b.len(),
            comparisons,
        })
    })
    .map_err(|e| PyValueError::new_err(format!("Weights diff error: {}", e)))?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "pytorch")]
    use super::weights_info;
    use super::{difference_exceeds_tolerance, weights_diff, TensorComparisonStatus};
    use pyo3::Python;

    #[cfg(feature = "pytorch")]
    use std::path::{Path, PathBuf};
    #[cfg(feature = "pytorch")]
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A tiny `torch.save(OrderedDict(test=tensor([[1, ..., 8]])))` archive.
    ///
    /// Keeping the 1.2 KiB fixture as hex makes fixture generation Cargo-owned
    /// and hermetic: neither torch nor Python-side setup is needed to assert
    /// checkpoint-directory behavior through the PyO3 surface.
    #[cfg(feature = "pytorch")]
    const PYTORCH_TEST_PT_HEX: &str = concat!(
        "504b0304000008080000000000000000000000000000000000000d001500746573742f646174612e706b6c464211005a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
        "800263636f6c6c656374696f6e730a4f726465726564446963740a710029527101580400000074657374710263746f7263682e5f7574696c730a5f7265627569",
        "6c645f74656e736f725f76320a71032828580700000073746f72616765710463746f7263680a4c6f6e6753746f726167650a7105580100000030710658030000",
        "0063707571074b08747108514b004b024b048671094b044b0186710a8968002952710b74710c52710d732e504b07083e6a9759ab000000ab000000504b030400",
        "0008080000000000000000000000000000000000000e001900746573742f627974656f72646572464215005a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
        "6c6974746c65504b0708853de3190600000006000000504b0304000008080000000000000000000000000000000000000b004100746573742f646174612f3046",
        "423d005a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
        "5a",
        "01000000000000000200000000000000030000000000000004000000000000000500000000000000060000000000000007000000000000000800000000000000",
        "504b0708a6b1b0494000000040000000504b0304000008080000000000000000000000000000000000000c000600746573742f76657273696f6e464202005a5a",
        "330a504b0708d19e67550200000002000000504b0304000008080000000000000000000000000000000000001b003500746573742f2e646174612f7365726961",
        "6c697a6174696f6e5f6964464231005a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
        "31313436343432373233383439333535303431353030303031303932373032323034353532313731504b0708b2d616d22800000028000000504b010200000000",
        "08080000000000003e6a9759ab000000ab0000000d0000000000000000000000000000000000746573742f646174612e706b6c504b0102000000000808000000",
        "000000853de31906000000060000000e00000000000000000000000000fb000000746573742f627974656f72646572504b0102000000000808000000000000a6",
        "b1b04940000000400000000b0000000000000000000000000056010000746573742f646174612f30504b0102000000000808000000000000d19e675502000000",
        "020000000c0000000000000000000000000010020000746573742f76657273696f6e504b0102000000000808000000000000b2d616d228000000280000001b00",
        "00000000000000000000000052020000746573742f2e646174612f73657269616c697a6174696f6e5f6964504b06062c000000000000001e032d000000000000",
        "000000050000000000000005000000000000003301000000000000f802000000000000504b0607000000002b0400000000000001000000504b05060000000005",
        "00050033010000f80200000000",
    );

    #[cfg(feature = "pytorch")]
    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[cfg(feature = "pytorch")]
    struct TestDir(PathBuf);

    #[cfg(feature = "pytorch")]
    impl TestDir {
        fn new(label: &str) -> Self {
            let serial = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("ny-python-{label}-{}-{serial}", std::process::id()));
            std::fs::create_dir(&path).expect("create hermetic PyTorch test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    #[cfg(feature = "pytorch")]
    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).expect("remove hermetic PyTorch test directory");
        }
    }

    #[cfg(feature = "pytorch")]
    fn decode_pytorch_fixture() -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("invalid hex byte in embedded PyTorch fixture"),
            }
        }

        let hex = PYTORCH_TEST_PT_HEX.as_bytes();
        assert_eq!(hex.len() % 2, 0, "fixture hex length must be even");
        hex.chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }

    #[cfg(feature = "pytorch")]
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = u32::MAX;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let low_bit_mask = 0u32.wrapping_sub(crc & 1);
                crc = (crc >> 1) ^ (0xedb8_8320 & low_bit_mask);
            }
        }
        !crc
    }

    #[cfg(feature = "pytorch")]
    fn pytorch_fixture_with_name(name: &[u8; 4]) -> Vec<u8> {
        let mut archive = decode_pytorch_fixture();
        if name == b"test" {
            return archive;
        }

        // Change only the four-byte state-dict key in data.pkl. The ZIP stores
        // this entry uncompressed, so updating its data-descriptor and central
        // directory CRC is sufficient; sizes and every subsequent offset stay
        // identical. This yields a genuinely distinct second shard without
        // invoking torch during a Cargo test.
        let pickle_key = b"X\x04\0\0\0testq\x02";
        let key_offsets: Vec<usize> = archive
            .windows(pickle_key.len())
            .enumerate()
            .filter_map(|(offset, window)| (window == pickle_key).then_some(offset))
            .collect();
        assert_eq!(key_offsets.len(), 1, "fixture has one state-dict key");
        let key_start = key_offsets[0] + 5;
        archive[key_start..key_start + 4].copy_from_slice(name);

        assert_eq!(&archive[..4], b"PK\x03\x04");
        let file_name_len = usize::from(u16::from_le_bytes([archive[26], archive[27]]));
        let extra_len = usize::from(u16::from_le_bytes([archive[28], archive[29]]));
        let data_start = 30 + file_name_len + extra_len;
        let descriptor = data_start
            + archive[data_start..]
                .windows(4)
                .position(|window| window == b"PK\x07\x08")
                .expect("fixture data descriptor");
        let checksum = crc32(&archive[data_start..descriptor]).to_le_bytes();
        archive[descriptor + 4..descriptor + 8].copy_from_slice(&checksum);

        let central = descriptor
            + archive[descriptor..]
                .windows(4)
                .position(|window| window == b"PK\x01\x02")
                .expect("fixture central directory");
        archive[central + 16..central + 20].copy_from_slice(&checksum);
        archive
    }

    #[cfg(feature = "pytorch")]
    fn write_pytorch_fixture(path: &Path, name: &[u8; 4]) {
        std::fs::write(path, pytorch_fixture_with_name(name))
            .expect("write embedded PyTorch fixture");
    }

    #[cfg(feature = "pytorch")]
    fn write_sharded_pytorch_fixture(dir: &Path) {
        let first = "pytorch_model-00001-of-00002.bin";
        let second = "pytorch_model-00002-of-00002.bin";
        write_pytorch_fixture(&dir.join(first), b"test");
        write_pytorch_fixture(&dir.join(second), b"tes2");
        std::fs::write(
            dir.join("pytorch_model.bin.index.json"),
            format!(r#"{{"weight_map":{{"test":"{first}","tes2":"{second}"}}}}"#),
        )
        .expect("write hermetic PyTorch shard index");
    }

    fn simple_mlp_path() -> String {
        format!(
            "{}/../../tests/models/simple_mlp.onnx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[test]
    fn test_tensor_comparison_status_str_3942() {
        assert_eq!(TensorComparisonStatus::Match.__str__(), "match");
        assert_eq!(TensorComparisonStatus::Differs.__str__(), "differs");
        assert_eq!(
            TensorComparisonStatus::ShapeMismatch.__str__(),
            "shape_mismatch"
        );
        assert_eq!(TensorComparisonStatus::MissingInA.__str__(), "missing_in_a");
        assert_eq!(TensorComparisonStatus::MissingInB.__str__(), "missing_in_b");
    }

    #[test]
    fn non_finite_weight_differences_fail_closed() {
        assert!(difference_exceeds_tolerance(f32::NAN, 1e-6));
        assert!(difference_exceeds_tolerance(f32::INFINITY, 1e-6));
        assert!(difference_exceeds_tolerance(f32::NEG_INFINITY, 1e-6));
        assert!(!difference_exceeds_tolerance(1e-7, 1e-6));
    }

    #[test]
    fn test_weights_diff_returns_typed_status_3942() {
        Python::initialize();
        Python::attach(|py| {
            let result = weights_diff(py, &simple_mlp_path(), &simple_mlp_path(), 1e-6)
                .expect("weights diff should succeed");

            assert!(
                result
                    .comparisons
                    .iter()
                    .all(|comparison| comparison.status == TensorComparisonStatus::Match),
                "same-file weights diff should only emit typed Match statuses"
            );
        });
    }

    #[cfg(feature = "pytorch")]
    #[test]
    fn pytorch_checkpoint_directory_info_is_cargo_owned() {
        let dir = TestDir::new("checkpoint-directory");
        write_pytorch_fixture(&dir.path().join("pytorch_model.bin"), b"test");

        Python::initialize();
        Python::attach(|py| {
            let info = weights_info(py, dir.path().to_str().expect("UTF-8 temporary path"))
                .expect("load hermetic PyTorch checkpoint directory");
            assert_eq!(info.format, "PyTorch (checkpoint)");
            assert_eq!(info.tensor_count, 1);
            assert_eq!(info.total_params, 8);
            assert_eq!(info.tensors[0].name, "test");
            assert_eq!(info.tensors[0].shape, [2, 4]);
        });
    }

    #[cfg(feature = "pytorch")]
    #[test]
    fn pytorch_sharded_directory_info_and_diff_are_cargo_owned() {
        let dir = TestDir::new("sharded-directory");
        let mirror = TestDir::new("sharded-directory-mirror");
        write_sharded_pytorch_fixture(dir.path());
        write_sharded_pytorch_fixture(mirror.path());

        Python::initialize();
        Python::attach(|py| {
            let info = weights_info(py, dir.path().to_str().expect("UTF-8 temporary path"))
                .expect("load hermetic sharded PyTorch directory");
            assert_eq!(info.format, "PyTorch (sharded)");
            assert_eq!(info.tensor_count, 2);
            assert_eq!(info.total_params, 16);
            assert_eq!(
                info.tensors
                    .iter()
                    .map(|tensor| tensor.name.as_str())
                    .collect::<Vec<_>>(),
                ["tes2", "test"]
            );

            let result = weights_diff(
                py,
                dir.path().to_str().expect("UTF-8 temporary path"),
                mirror.path().to_str().expect("UTF-8 temporary path"),
                1e-6,
            )
            .expect("compare independently loaded sharded PyTorch checkpoints");
            assert!(result.is_match);
            assert_eq!(result.max_diff, 0.0);
            assert_eq!(result.total_tensors_a, 2);
            assert_eq!(result.total_tensors_b, 2);
        });
    }
}
