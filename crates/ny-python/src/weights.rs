// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::repr::repr_string;
use crate::utils::validate_tolerance;
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
                    .fold(0.0f32, ny_core::nan_propagating_max);

                max_diff = max_diff.max(diff);

                if diff > tolerance {
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
    use super::{weights_diff, TensorComparisonStatus};
    use pyo3::Python;

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
}
