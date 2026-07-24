// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Custom operator metadata and registration for ny.

use crate::{NyError, Result};
use std::collections::HashSet;

/// Supported attribute kinds for custom operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomOpAttributeType {
    /// Scalar `f32` value.
    Float,
    /// Scalar `i64` value.
    Int,
    /// UTF-8 string value.
    String,
    /// Serialized tensor (raw bytes).
    Tensor,
    /// Serialized computation graph (raw bytes).
    Graph,
    /// Serialized sparse tensor (raw bytes).
    SparseTensor,
    /// Serialized ONNX type proto (raw bytes).
    TypeProto,
    /// List of `f32` values.
    Floats,
    /// List of `i64` values.
    Ints,
    /// List of UTF-8 strings.
    Strings,
    /// List of serialized tensors.
    Tensors,
    /// List of serialized computation graphs.
    Graphs,
    /// List of serialized sparse tensors.
    SparseTensors,
    /// List of serialized ONNX type protos.
    TypeProtos,
}

impl CustomOpAttributeType {
    /// Returns the string name of this attribute type (e.g., `"Float"`, `"Tensor"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            CustomOpAttributeType::Float => "Float",
            CustomOpAttributeType::Int => "Int",
            CustomOpAttributeType::String => "String",
            CustomOpAttributeType::Tensor => "Tensor",
            CustomOpAttributeType::Graph => "Graph",
            CustomOpAttributeType::SparseTensor => "SparseTensor",
            CustomOpAttributeType::TypeProto => "TypeProto",
            CustomOpAttributeType::Floats => "Floats",
            CustomOpAttributeType::Ints => "Ints",
            CustomOpAttributeType::Strings => "Strings",
            CustomOpAttributeType::Tensors => "Tensors",
            CustomOpAttributeType::Graphs => "Graphs",
            CustomOpAttributeType::SparseTensors => "SparseTensors",
            CustomOpAttributeType::TypeProtos => "TypeProtos",
        }
    }
}

/// Typed default value for a custom operator attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum CustomOpAttributeValue {
    /// Scalar `f32` value.
    Float(f32),
    /// Scalar `i64` value.
    Int(i64),
    /// UTF-8 string value.
    String(String),
    /// Serialized tensor (raw bytes).
    Tensor(Vec<u8>),
    /// Serialized computation graph (raw bytes).
    Graph(Vec<u8>),
    /// Serialized sparse tensor (raw bytes).
    SparseTensor(Vec<u8>),
    /// Serialized ONNX type proto (raw bytes).
    TypeProto(Vec<u8>),
    /// List of `f32` values.
    Floats(Vec<f32>),
    /// List of `i64` values.
    Ints(Vec<i64>),
    /// List of UTF-8 strings.
    Strings(Vec<String>),
    /// List of serialized tensors.
    Tensors(Vec<Vec<u8>>),
    /// List of serialized computation graphs.
    Graphs(Vec<Vec<u8>>),
    /// List of serialized sparse tensors.
    SparseTensors(Vec<Vec<u8>>),
    /// List of serialized ONNX type protos.
    TypeProtos(Vec<Vec<u8>>),
}

impl CustomOpAttributeValue {
    fn matches_type(&self, attr_type: &CustomOpAttributeType) -> bool {
        matches!(
            (self, attr_type),
            (
                CustomOpAttributeValue::Float(_),
                CustomOpAttributeType::Float
            ) | (CustomOpAttributeValue::Int(_), CustomOpAttributeType::Int)
                | (
                    CustomOpAttributeValue::String(_),
                    CustomOpAttributeType::String
                )
                | (
                    CustomOpAttributeValue::Tensor(_),
                    CustomOpAttributeType::Tensor
                )
                | (
                    CustomOpAttributeValue::Graph(_),
                    CustomOpAttributeType::Graph
                )
                | (
                    CustomOpAttributeValue::SparseTensor(_),
                    CustomOpAttributeType::SparseTensor
                )
                | (
                    CustomOpAttributeValue::TypeProto(_),
                    CustomOpAttributeType::TypeProto
                )
                | (
                    CustomOpAttributeValue::Floats(_),
                    CustomOpAttributeType::Floats
                )
                | (CustomOpAttributeValue::Ints(_), CustomOpAttributeType::Ints)
                | (
                    CustomOpAttributeValue::Strings(_),
                    CustomOpAttributeType::Strings
                )
                | (
                    CustomOpAttributeValue::Tensors(_),
                    CustomOpAttributeType::Tensors
                )
                | (
                    CustomOpAttributeValue::Graphs(_),
                    CustomOpAttributeType::Graphs
                )
                | (
                    CustomOpAttributeValue::SparseTensors(_),
                    CustomOpAttributeType::SparseTensors
                )
                | (
                    CustomOpAttributeValue::TypeProtos(_),
                    CustomOpAttributeType::TypeProtos
                )
        )
    }

    fn type_name(&self) -> &'static str {
        match self {
            CustomOpAttributeValue::Float(_) => "Float",
            CustomOpAttributeValue::Int(_) => "Int",
            CustomOpAttributeValue::String(_) => "String",
            CustomOpAttributeValue::Tensor(_) => "Tensor",
            CustomOpAttributeValue::Graph(_) => "Graph",
            CustomOpAttributeValue::SparseTensor(_) => "SparseTensor",
            CustomOpAttributeValue::TypeProto(_) => "TypeProto",
            CustomOpAttributeValue::Floats(_) => "Floats",
            CustomOpAttributeValue::Ints(_) => "Ints",
            CustomOpAttributeValue::Strings(_) => "Strings",
            CustomOpAttributeValue::Tensors(_) => "Tensors",
            CustomOpAttributeValue::Graphs(_) => "Graphs",
            CustomOpAttributeValue::SparseTensors(_) => "SparseTensors",
            CustomOpAttributeValue::TypeProtos(_) => "TypeProtos",
        }
    }
}

/// Attribute schema for a custom operator.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomOpAttribute {
    /// Attribute name (must be non-empty and unique within the schema).
    pub(crate) name: String,
    /// Expected value type for this attribute.
    pub(crate) attr_type: CustomOpAttributeType,
    /// Whether the attribute must be present in every operator instance.
    pub(crate) required: bool,
    /// Default value used when the attribute is absent. Must match `attr_type`.
    pub(crate) default_value: Option<CustomOpAttributeValue>,
}

impl CustomOpAttribute {
    /// Creates a new optional attribute with the given name and type (no default value).
    pub fn new(name: impl Into<String>, attr_type: CustomOpAttributeType) -> Self {
        Self {
            name: name.into(),
            attr_type,
            required: false,
            default_value: None,
        }
    }

    /// Returns the attribute name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the expected attribute type.
    pub fn attr_type(&self) -> &CustomOpAttributeType {
        &self.attr_type
    }

    /// Returns whether this attribute is required.
    pub fn required(&self) -> bool {
        self.required
    }

    /// Returns the default value, if any.
    pub fn default_value(&self) -> Option<&CustomOpAttributeValue> {
        self.default_value.as_ref()
    }
}

/// Schema describing the arity and attributes for a custom operator.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CustomOpSchema {
    /// Minimum number of inputs the operator accepts.
    pub(crate) min_inputs: usize,
    /// Maximum number of inputs (`None` = unbounded).
    pub(crate) max_inputs: Option<usize>,
    /// Minimum number of outputs the operator produces.
    pub(crate) min_outputs: usize,
    /// Maximum number of outputs (`None` = unbounded).
    pub(crate) max_outputs: Option<usize>,
    /// Attribute definitions for this operator.
    pub(crate) attributes: Vec<CustomOpAttribute>,
}

impl CustomOpSchema {
    /// Creates a schema with explicit input/output arity and attribute definitions.
    pub fn new(
        min_inputs: usize,
        max_inputs: Option<usize>,
        min_outputs: usize,
        max_outputs: Option<usize>,
        attributes: Vec<CustomOpAttribute>,
    ) -> Self {
        Self {
            min_inputs,
            max_inputs,
            min_outputs,
            max_outputs,
            attributes,
        }
    }

    /// Returns the minimum number of inputs.
    pub fn min_inputs(&self) -> usize {
        self.min_inputs
    }

    /// Returns the maximum number of inputs, if bounded.
    pub fn max_inputs(&self) -> Option<usize> {
        self.max_inputs
    }

    /// Returns the minimum number of outputs.
    pub fn min_outputs(&self) -> usize {
        self.min_outputs
    }

    /// Returns the maximum number of outputs, if bounded.
    pub fn max_outputs(&self) -> Option<usize> {
        self.max_outputs
    }

    /// Returns the attribute definitions.
    pub fn attributes(&self) -> &[CustomOpAttribute] {
        &self.attributes
    }
}

/// Metadata describing a custom operator.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomOpSpec {
    /// Operator domain (empty string for ONNX default domain).
    pub(crate) domain: String,
    /// Operator type name (e.g., "MyCustomOp").
    pub(crate) op_type: String,
    /// Optional opset version the spec targets.
    pub(crate) opset_version: Option<i64>,
    /// Schema describing inputs/outputs and attributes.
    pub(crate) schema: CustomOpSchema,
}

impl CustomOpSpec {
    /// Creates a custom op spec with a default (empty) schema.
    pub fn new(
        domain: impl Into<String>,
        op_type: impl Into<String>,
        opset_version: Option<i64>,
    ) -> Self {
        Self {
            domain: domain.into(),
            op_type: op_type.into(),
            opset_version,
            schema: CustomOpSchema::default(),
        }
    }

    /// Creates a custom op spec with an explicit schema.
    pub fn with_schema(
        domain: impl Into<String>,
        op_type: impl Into<String>,
        opset_version: Option<i64>,
        schema: CustomOpSchema,
    ) -> Self {
        Self {
            domain: domain.into(),
            op_type: op_type.into(),
            opset_version,
            schema,
        }
    }

    /// Returns the operator domain.
    pub fn domain(&self) -> &str {
        &self.domain
    }

    /// Returns the operator type name.
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// Returns the opset version, if specified.
    pub fn opset_version(&self) -> Option<i64> {
        self.opset_version
    }

    /// Returns the operator schema.
    pub fn schema(&self) -> &CustomOpSchema {
        &self.schema
    }
}

/// Registry of custom operator schemas, keyed by (domain, op_type, opset_version).
#[derive(Debug, Clone, Default)]
pub struct CustomOpSchemaRegistry {
    entries: Vec<CustomOpSpec>,
}

impl CustomOpSchemaRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a custom op spec. Returns an error on duplicate registrations
    /// or invalid schemas.
    pub fn register(&mut self, spec: CustomOpSpec) -> Result<()> {
        self.validate_registration(&spec)?;
        self.entries.push(spec);
        Ok(())
    }

    /// Registers multiple custom op specs. Stops on the first error.
    pub fn extend<I>(&mut self, specs: I) -> Result<()>
    where
        I: IntoIterator<Item = CustomOpSpec>,
    {
        for spec in specs {
            self.register(spec)?;
        }
        Ok(())
    }

    /// Returns all registered custom op specs.
    pub fn entries(&self) -> &[CustomOpSpec] {
        &self.entries
    }

    /// Resolve a custom op spec for the given domain/op_type and opset version.
    ///
    /// Precedence rules:
    /// - Exact opset match wins.
    /// - Otherwise, choose the highest version <= requested (if any).
    /// - Otherwise, fall back to an unversioned spec (if any).
    /// - If no opset version is provided, prefer unversioned, then highest versioned.
    pub fn resolve(
        &self,
        domain: &str,
        op_type: &str,
        opset_version: Option<i64>,
    ) -> Option<&CustomOpSpec> {
        let mut unversioned: Option<&CustomOpSpec> = None;
        let mut best_version: Option<(i64, &CustomOpSpec)> = None;

        for spec in self
            .entries
            .iter()
            .filter(|spec| spec.domain == domain && spec.op_type == op_type)
        {
            match (spec.opset_version, opset_version) {
                (Some(spec_version), Some(requested)) => {
                    if spec_version == requested {
                        return Some(spec);
                    }
                    if spec_version <= requested {
                        let should_update = match best_version {
                            Some((best, _)) => spec_version > best,
                            None => true,
                        };
                        if should_update {
                            best_version = Some((spec_version, spec));
                        }
                    }
                }
                (Some(spec_version), None) => {
                    let should_update = match best_version {
                        Some((best, _)) => spec_version > best,
                        None => true,
                    };
                    if should_update {
                        best_version = Some((spec_version, spec));
                    }
                }
                (None, _) => {
                    if unversioned.is_none() {
                        unversioned = Some(spec);
                    }
                }
            }
        }

        match opset_version {
            Some(_) => best_version.map(|(_, spec)| spec).or(unversioned),
            None => unversioned.or_else(|| best_version.map(|(_, spec)| spec)),
        }
    }

    /// Return a human-readable listing for diagnostics.
    pub fn debug_listing(&self) -> String {
        if self.entries.is_empty() {
            return "CustomOpSchemaRegistry(empty)".to_string();
        }

        let mut lines = Vec::with_capacity(self.entries.len() + 1);
        lines.push("CustomOpSchemaRegistry:".to_string());
        for spec in &self.entries {
            let opset = spec
                .opset_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "any".to_string());
            lines.push(format!(
                "- domain=\"{}\" op_type=\"{}\" opset_version={}",
                spec.domain, spec.op_type, opset
            ));
        }
        lines.join("\n")
    }

    fn validate_registration(&self, spec: &CustomOpSpec) -> Result<()> {
        if self.entries.iter().any(|entry| {
            entry.domain == spec.domain
                && entry.op_type == spec.op_type
                && entry.opset_version == spec.opset_version
        }) {
            return Err(NyError::InvalidSpec(format!(
                "Duplicate custom op registration for domain=\"{}\" op_type=\"{}\" opset_version={}",
                spec.domain,
                spec.op_type,
                spec.opset_version
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "any".to_string())
            )));
        }

        let schema = &spec.schema;
        if let Some(max_inputs) = schema.max_inputs {
            if schema.min_inputs > max_inputs {
                return Err(NyError::InvalidSpec(format!(
                    "Custom op schema for {}/{} has min_inputs {} > max_inputs {}",
                    spec.domain, spec.op_type, schema.min_inputs, max_inputs
                )));
            }
        }

        if let Some(max_outputs) = schema.max_outputs {
            if schema.min_outputs > max_outputs {
                return Err(NyError::InvalidSpec(format!(
                    "Custom op schema for {}/{} has min_outputs {} > max_outputs {}",
                    spec.domain, spec.op_type, schema.min_outputs, max_outputs
                )));
            }
        }

        let mut seen = HashSet::new();
        for attr in &schema.attributes {
            if attr.name.trim().is_empty() {
                return Err(NyError::InvalidSpec(format!(
                    "Custom op schema for {}/{} has empty attribute name",
                    spec.domain, spec.op_type
                )));
            }

            if !seen.insert(attr.name.clone()) {
                return Err(NyError::InvalidSpec(format!(
                    "Custom op schema for {}/{} has duplicate attribute \"{}\"",
                    spec.domain, spec.op_type, attr.name
                )));
            }

            if let Some(default_value) = &attr.default_value {
                if !default_value.matches_type(&attr.attr_type) {
                    return Err(NyError::InvalidSpec(format!(
                        "Custom op schema for {}/{} attribute \"{}\" default type mismatch: expected {}, got {}",
                        spec.domain,
                        spec.op_type,
                        attr.name,
                        attr.attr_type.as_str(),
                        default_value.type_name()
                    )));
                }
            }
        }

        Ok(())
    }
}
