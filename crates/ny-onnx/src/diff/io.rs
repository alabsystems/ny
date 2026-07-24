// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{DiffError, ModelInfo};
use crate::loader::numeric_cast::f64_to_f32_checked;
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use std::path::Path;
use tracing::warn;

/// Load model info from an ONNX file.
pub fn load_model_info(path: impl AsRef<Path>) -> Result<ModelInfo, DiffError> {
    let onnx_model =
        crate::load_onnx(path.as_ref()).map_err(|e| DiffError::LoadError(format!("{}", e)))?;

    let network = &onnx_model.network;

    // Collect all intermediate tensor names from layer outputs
    let mut intermediate_names = Vec::new();
    for layer in &network.layers {
        for output in &layer.outputs {
            intermediate_names.push(output.clone());
        }
    }

    Ok(ModelInfo {
        inputs: network.inputs.clone(),
        outputs: network.outputs.clone(),
        intermediate_names,
        layers: network.layers.clone(),
    })
}

/// Load model info from in-memory ONNX bytes.
pub fn load_model_info_bytes(name: &str, data: &[u8]) -> Result<ModelInfo, DiffError> {
    let onnx_model =
        crate::load_onnx_bytes(name, data).map_err(|e| DiffError::LoadError(format!("{}", e)))?;

    let network = &onnx_model.network;

    // Collect all intermediate tensor names from layer outputs
    let mut intermediate_names = Vec::new();
    for layer in &network.layers {
        for output in &layer.outputs {
            intermediate_names.push(output.clone());
        }
    }

    Ok(ModelInfo {
        inputs: network.inputs.clone(),
        outputs: network.outputs.clone(),
        intermediate_names,
        layers: network.layers.clone(),
    })
}

/// Load a numpy array from a .npy file.
pub fn load_npy(path: impl AsRef<Path>) -> Result<ArrayD<f32>, DiffError> {
    fn shape_to_usize(path: &Path, shape: &[u64]) -> Result<Vec<usize>, DiffError> {
        shape
            .iter()
            .map(|&dim| {
                usize::try_from(dim).map_err(|_| {
                    DiffError::NpyError(format!(
                        "Dimension {dim} in {} does not fit into usize",
                        path.display()
                    ))
                })
            })
            .collect()
    }

    fn build_array(
        path: &Path,
        shape: &[u64],
        order: npyz::Order,
        data: Vec<f32>,
    ) -> Result<ArrayD<f32>, DiffError> {
        let shape = shape_to_usize(path, shape)?;
        match order {
            npyz::Order::C => ArrayD::from_shape_vec(IxDyn(&shape), data),
            npyz::Order::Fortran => ArrayD::from_shape_vec(IxDyn(&shape).f(), data),
        }
        .map_err(|e| {
            DiffError::NpyError(format!(
                "Failed to build ndarray from {}: {}",
                path.display(),
                e
            ))
        })
    }

    let path = path.as_ref();

    let file = std::fs::File::open(path)?;
    let npy = npyz::NpyFile::new(std::io::BufReader::new(file))?;
    let order = npy.order();
    let shape = npy.shape().to_vec();

    match npy.into_vec::<f32>() {
        Ok(data) => build_array(path, &shape, order, data),
        Err(_) => {
            let file = std::fs::File::open(path)?;
            let npy = npyz::NpyFile::new(std::io::BufReader::new(file))?;
            let order = npy.order();
            let shape = npy.shape().to_vec();
            match npy.into_vec::<f64>() {
                Ok(data) => {
                    let context = path.display().to_string();
                    let mut precision_loss_count = 0usize;
                    let mut first_precision_loss = None;
                    let mut converted = Vec::with_capacity(data.len());
                    for value in data {
                        let (downcast, loses_precision) = f64_to_f32_checked(value, &context)
                            .map_err(|e| DiffError::NpyError(e.to_string()))?;
                        if loses_precision {
                            precision_loss_count += 1;
                            if first_precision_loss.is_none() {
                                first_precision_loss = Some((value, downcast));
                            }
                        }
                        converted.push(downcast);
                    }
                    if let Some((original, downcast)) = first_precision_loss {
                        warn!(
                            "NPY file {} loses precision on {} values during f64→f32 downcast; \
                             first example: {} -> {}",
                            path.display(),
                            precision_loss_count,
                            original,
                            downcast
                        );
                    }
                    build_array(path, &shape, order, converted)
                }
                Err(_) => Err(DiffError::NpyError(format!(
                    "Could not read numpy file as f32 or f64: {}",
                    path.display()
                ))),
            }
        }
    }
}
