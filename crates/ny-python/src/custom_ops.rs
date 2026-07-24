// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Python-facing custom operator registration API.

use ny_core::CustomOpAttributeType as CoreCustomOpAttributeType;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyclass(name = "CustomOpAttributeType", from_py_object)]
#[derive(Clone, Copy, Debug)]
pub enum PyCustomOpAttributeType {
    Float,
    Int,
    String,
    Tensor,
    Graph,
    SparseTensor,
    TypeProto,
    Floats,
    Ints,
    Strings,
    Tensors,
    Graphs,
    SparseTensors,
    TypeProtos,
}

#[pymethods]
impl PyCustomOpAttributeType {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            PyCustomOpAttributeType::Float => "CustomOpAttributeType.Float",
            PyCustomOpAttributeType::Int => "CustomOpAttributeType.Int",
            PyCustomOpAttributeType::String => "CustomOpAttributeType.String",
            PyCustomOpAttributeType::Tensor => "CustomOpAttributeType.Tensor",
            PyCustomOpAttributeType::Graph => "CustomOpAttributeType.Graph",
            PyCustomOpAttributeType::SparseTensor => "CustomOpAttributeType.SparseTensor",
            PyCustomOpAttributeType::TypeProto => "CustomOpAttributeType.TypeProto",
            PyCustomOpAttributeType::Floats => "CustomOpAttributeType.Floats",
            PyCustomOpAttributeType::Ints => "CustomOpAttributeType.Ints",
            PyCustomOpAttributeType::Strings => "CustomOpAttributeType.Strings",
            PyCustomOpAttributeType::Tensors => "CustomOpAttributeType.Tensors",
            PyCustomOpAttributeType::Graphs => "CustomOpAttributeType.Graphs",
            PyCustomOpAttributeType::SparseTensors => "CustomOpAttributeType.SparseTensors",
            PyCustomOpAttributeType::TypeProtos => "CustomOpAttributeType.TypeProtos",
        }
        .to_string()
    }
}

impl From<PyCustomOpAttributeType> for CoreCustomOpAttributeType {
    fn from(value: PyCustomOpAttributeType) -> Self {
        match value {
            PyCustomOpAttributeType::Float => CoreCustomOpAttributeType::Float,
            PyCustomOpAttributeType::Int => CoreCustomOpAttributeType::Int,
            PyCustomOpAttributeType::String => CoreCustomOpAttributeType::String,
            PyCustomOpAttributeType::Tensor => CoreCustomOpAttributeType::Tensor,
            PyCustomOpAttributeType::Graph => CoreCustomOpAttributeType::Graph,
            PyCustomOpAttributeType::SparseTensor => CoreCustomOpAttributeType::SparseTensor,
            PyCustomOpAttributeType::TypeProto => CoreCustomOpAttributeType::TypeProto,
            PyCustomOpAttributeType::Floats => CoreCustomOpAttributeType::Floats,
            PyCustomOpAttributeType::Ints => CoreCustomOpAttributeType::Ints,
            PyCustomOpAttributeType::Strings => CoreCustomOpAttributeType::Strings,
            PyCustomOpAttributeType::Tensors => CoreCustomOpAttributeType::Tensors,
            PyCustomOpAttributeType::Graphs => CoreCustomOpAttributeType::Graphs,
            PyCustomOpAttributeType::SparseTensors => CoreCustomOpAttributeType::SparseTensors,
            PyCustomOpAttributeType::TypeProtos => CoreCustomOpAttributeType::TypeProtos,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug)]
pub struct CustomOpAttribute {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub attr_type: PyCustomOpAttributeType,
    #[pyo3(get)]
    pub required: bool,
    pub default_value: Option<Py<PyAny>>,
}

#[pymethods]
impl CustomOpAttribute {
    #[new]
    #[pyo3(signature = (name, attr_type, required=false, default_value=None))]
    fn new(
        name: String,
        attr_type: PyCustomOpAttributeType,
        required: bool,
        default_value: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if required && default_value.is_some() {
            return Err(PyValueError::new_err(
                "Custom op attribute cannot be required when a default value is provided",
            ));
        }
        Ok(Self {
            name,
            attr_type,
            required,
            default_value,
        })
    }

    pub(crate) fn __repr__(&self) -> String {
        let default_tag = if self.default_value.is_some() {
            ", default_value=..."
        } else {
            ""
        };
        format!(
            "CustomOpAttribute(name='{}', attr_type={}, required={}{})",
            self.name,
            self.attr_type.__repr__(),
            self.required,
            default_tag
        )
    }

    #[getter]
    fn default_value(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.default_value.as_ref().map(|value| value.clone_ref(py))
    }
}

impl Clone for CustomOpAttribute {
    fn clone(&self) -> Self {
        let default_value = self
            .default_value
            .as_ref()
            .map(|value| Python::attach(|py| value.clone_ref(py)));
        Self {
            name: self.name.clone(),
            attr_type: self.attr_type,
            required: self.required,
            default_value,
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct CustomOpSchema {
    #[pyo3(get)]
    pub min_inputs: usize,
    #[pyo3(get)]
    pub max_inputs: Option<usize>,
    #[pyo3(get)]
    pub min_outputs: usize,
    #[pyo3(get)]
    pub max_outputs: Option<usize>,
    #[pyo3(get)]
    pub attributes: Vec<CustomOpAttribute>,
}

#[pymethods]
impl CustomOpSchema {
    #[new]
    #[pyo3(signature = (min_inputs=0, max_inputs=None, min_outputs=0, max_outputs=None, attributes=None))]
    fn new(
        py: Python<'_>,
        min_inputs: usize,
        max_inputs: Option<usize>,
        min_outputs: usize,
        max_outputs: Option<usize>,
        attributes: Option<Vec<Py<CustomOpAttribute>>>,
    ) -> PyResult<Self> {
        if let Some(max_inputs) = max_inputs {
            if min_inputs > max_inputs {
                return Err(PyValueError::new_err(format!(
                    "Custom op schema min_inputs ({}) exceeds max_inputs ({})",
                    min_inputs, max_inputs
                )));
            }
        }
        if let Some(max_outputs) = max_outputs {
            if min_outputs > max_outputs {
                return Err(PyValueError::new_err(format!(
                    "Custom op schema min_outputs ({}) exceeds max_outputs ({})",
                    min_outputs, max_outputs
                )));
            }
        }
        let attributes = match attributes {
            Some(attributes) => attributes
                .into_iter()
                .map(|attr| attr.bind(py).borrow().clone())
                .collect(),
            None => Vec::new(),
        };
        Ok(Self {
            min_inputs,
            max_inputs,
            min_outputs,
            max_outputs,
            attributes,
        })
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "CustomOpSchema(min_inputs={}, max_inputs={:?}, min_outputs={}, max_outputs={:?}, attributes={})",
            self.min_inputs,
            self.max_inputs,
            self.min_outputs,
            self.max_outputs,
            self.attributes.len()
        )
    }
}

/// Register a custom operator for bound propagation.
///
/// Custom-op dispatch from Python is not yet wired into the verification runtime
/// (tracking issue: #1820, part of #1696). To avoid misleading no-op behavior,
/// this API rejects all registrations until runtime dispatch exists.
#[pyfunction]
fn register_custom_op(
    _py: Python<'_>,
    _domain: String,
    _op_type: String,
    _schema: Option<PyRef<'_, CustomOpSchema>>,
    _bound_impl: Py<PyAny>,
) -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "register_custom_op is not implemented yet because Python custom-op dispatch is not wired into the verification runtime (tracked by #1820).",
    ))
}

pub fn register_custom_ops_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CustomOpSchema>()?;
    m.add_class::<CustomOpAttribute>()?;
    m.add_class::<PyCustomOpAttributeType>()?;
    m.add_function(wrap_pyfunction!(register_custom_op, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_custom_op_returns_not_implemented() {
        Python::initialize();
        Python::attach(|py| {
            let err = register_custom_op(
                py,
                "com.example".to_string(),
                "MyCustomOp".to_string(),
                None,
                py.None(),
            )
            .expect_err("register_custom_op should reject until dispatch exists");

            assert!(err.is_instance_of::<PyNotImplementedError>(py));
            let message = err.to_string();
            assert!(message.contains("not implemented"));
            assert!(message.contains("#1820"));
        });
    }
}
