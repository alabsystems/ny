// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_propagate::layers::{
    AddConstantLayer, AddLayer, DivConstantLayer, DivLayer, MaxBinaryLayer, MinBinaryLayer,
    MulBinaryLayer, MulConstantLayer, PowConstantLayer, SubConstantLayer, SubLayer,
};
use ny_propagate::network::broadcast_shapes;
use ny_propagate::Layer;
use tracing::{debug, warn};

use super::{i64_to_f32_checked, AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_add(&self, spec: &LayerSpec) -> Result<Layer> {
        // Add in ONNX: C = A + B
        // If one input is a constant with a retrievable value, treat as AddConstant (unary).
        // Otherwise, treat as binary Add (e.g., residual connections).
        //
        // A tensor may be in constant_tensors (known constant) but have no retrievable
        // value in weights or evaluated_constants (e.g., produced by an unevaluable
        // constant chain like ConstantOfShape → Unsqueeze). In that case, the producing
        // graph node stays in the network and we fall through to binary Add, where
        // bounds propagate through the graph node (#3186).

        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "Add {} requires exactly 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        // Retrieve constant values. Returns None for constant tensors whose value
        // was not pre-evaluated — fall through to binary in that case.
        let a_value = self.constant_value(input_a);
        let b_value = self.constant_value(input_b);

        match (a_value, b_value) {
            (Some(_), Some(_)) => {
                // Both are constants - should be constant folded upstream.
                Err(NyError::UnsupportedOp(format!(
                    "Add {} has both constant inputs — should be constant-folded \
                     by the ONNX optimizer before verification",
                    spec.name
                )))
            }
            (Some(constant), None) => {
                // First input is constant: C = const + B
                debug!("Add {} is bias addition (constant first input)", spec.name);
                self.add_constant_layer(constant, &spec.name)
                    .map(Layer::AddConstant)
            }
            (None, Some(constant)) => {
                // Second input is constant: C = A + const
                debug!("Add {} is bias addition (constant second input)", spec.name);
                self.add_constant_layer(constant, &spec.name)
                    .map(Layer::AddConstant)
            }
            (None, None) => {
                // Neither input has a retrievable constant value — either both are
                // activations, or one/both are constant tensors without values.
                // In the latter case, the producing graph node propagates bounds.
                debug!(
                    "Add {} is binary operation (both inputs are activations)",
                    spec.name
                );
                Ok(Layer::Add(AddLayer))
            }
        }
    }
    pub(crate) fn convert_div(&self, spec: &LayerSpec) -> Result<Layer> {
        // Div in ONNX: C = A / B
        // If the divisor (B) is a constant with a retrievable value, treat as DivConstant.
        // Otherwise, use binary DivLayer (e.g., for LayerNorm normalization).
        // See #3186 for constant-without-value fallback rationale.

        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "Div {} requires exactly 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        let a_value = self.constant_value(input_a);
        let b_value = self.constant_value(input_b);

        if a_value.is_some() && b_value.is_some() {
            // Both constants - should be constant folded upstream.
            return Err(NyError::UnsupportedOp(format!(
                "Div {} has both constant inputs — should be constant-folded \
                 by the ONNX optimizer before verification",
                spec.name
            )));
        }

        if let Some(constant) = b_value {
            // Divisor is constant: C = A / const
            debug!("Div {} is division by constant", spec.name);
            Ok(Layer::DivConstant(
                self.div_constant_layer(constant, input_a, &spec.name)?,
            ))
        } else {
            // Binary division: either both activations or divisor is a constant
            // tensor without a retrievable value (producing node stays in graph).
            debug!("Div {} is binary division of two activations", spec.name);
            Ok(Layer::Div(DivLayer))
        }
    }
    pub(crate) fn convert_sub(&self, spec: &LayerSpec) -> Result<Layer> {
        // Sub in ONNX: C = A - B
        // If one input is a constant with a retrievable value, treat as SubConstant.
        // Otherwise, use binary SubLayer. See #3186 for constant-without-value fallback.

        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "Sub {} requires exactly 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        let a_value = self.constant_value(input_a);
        let b_value = self.constant_value(input_b);

        match (a_value, b_value) {
            (Some(_), Some(_)) => {
                // Both are constants - should be constant folded
                warn!(
                    "Sub {} has both constant inputs - should be constant folded",
                    spec.name
                );
                Err(NyError::UnsupportedOp(format!(
                    "Sub {} has both constant inputs",
                    spec.name
                )))
            }
            (Some(constant), None) => {
                // First input is constant: C = const - B (reversed subtraction)
                debug!("Sub {} is subtraction from constant", spec.name);
                self.sub_constant_layer(constant, true, &spec.name)
                    .map(Layer::SubConstant)
            }
            (None, Some(constant)) => {
                // Second input is constant: C = A - const (normal subtraction)
                debug!("Sub {} is subtraction of constant", spec.name);
                self.sub_constant_layer(constant, false, &spec.name)
                    .map(Layer::SubConstant)
            }
            (None, None) => {
                // Neither has a retrievable value — binary subtraction.
                debug!("Sub {} is binary subtraction of two activations", spec.name);
                Ok(Layer::Sub(SubLayer))
            }
        }
    }
    pub(crate) fn convert_pow(&self, spec: &LayerSpec) -> Result<Layer> {
        // Pow in ONNX: C = A^B
        // We only support constant exponent (B is a scalar constant)

        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "Pow {} requires exactly 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        // Check if both inputs are constants (pure constant op - should be folded)
        let a_is_const = self.is_constant(input_a);
        let b_is_const = self.is_constant(input_b);

        if a_is_const && b_is_const {
            return Err(NyError::UnsupportedOp(format!(
                "Pow {} has both constant inputs — should be constant-folded \
                 by the ONNX optimizer before verification",
                spec.name
            )));
        }

        // Check if exponent is a constant (weight or evaluated constant)
        if let Some(exp_tensor) = self.constant_value(input_b).as_ref() {
            let Some(&exponent) = exp_tensor.iter().next() else {
                return Err(NyError::UnsupportedOp(format!(
                    "Pow {} has an empty exponent tensor",
                    spec.name
                )));
            };
            let layer = self.pow_constant_layer(exponent, &spec.name)?;

            // PowConstant applies one scalar exponent without changing the base
            // tensor's shape.  A homogeneous ONNX exponent tensor has the same
            // values, but reducing it to a scalar is only equivalent when ONNX
            // broadcasting also leaves the base shape unchanged.  In particular,
            // base [1] with exponent [3] produces [3] in ONNX, not [1].
            if exp_tensor.ndim() != 0 {
                if !exp_tensor.iter().all(|&value| value == exponent) {
                    return Err(NyError::UnsupportedOp(format!(
                        "Pow {} has non-scalar heterogeneous exponent (shape={:?}) — \
                         per-element exponentiation not supported",
                        spec.name,
                        exp_tensor.shape()
                    )));
                }

                let Some(base_shape) = self.tensor_shape_usize(input_a) else {
                    return Err(NyError::UnsupportedOp(format!(
                        "Pow {} cannot authenticate whether exponent shape {:?} \
                         preserves the unknown base shape",
                        spec.name,
                        exp_tensor.shape()
                    )));
                };
                let broadcast_shape = broadcast_shapes(&base_shape, exp_tensor.shape());
                if broadcast_shape.as_deref() != Some(base_shape.as_slice()) {
                    return Err(NyError::UnsupportedOp(format!(
                        "Pow {} exponent shape {:?} would change or is incompatible with \
                         base shape {:?} under ONNX broadcasting",
                        spec.name,
                        exp_tensor.shape(),
                        base_shape
                    )));
                }

                debug!(
                    "Pow {} has shape-preserving homogeneous exponent (shape={:?}, value={})",
                    spec.name,
                    exp_tensor.shape(),
                    exponent
                );
            }
            debug!("Pow {} with constant exponent {}", spec.name, exponent);
            Ok(Layer::PowConstant(layer))
        } else {
            // Exponent is not constant - not supported
            Err(NyError::UnsupportedOp(format!(
                "Pow {} with non-constant exponent not supported",
                spec.name
            )))
        }
    }
    fn tensor_shape_usize(&self, name: &str) -> Option<Vec<usize>> {
        let shape = self.tensor_shapes.get(name)?;
        let mut out = Vec::with_capacity(shape.len());
        for &dim in shape {
            if dim <= 0 {
                return None;
            }
            out.push(dim as usize);
        }
        Some(out)
    }

    fn add_constant_layer(
        &self,
        constant: ArrayD<f32>,
        spec_name: &str,
    ) -> Result<AddConstantLayer> {
        AddConstantLayer::try_new(constant)
            .map_err(|err| NyError::InvalidSpec(format!("Add {spec_name} constant invalid: {err}")))
    }

    fn sub_constant_layer(
        &self,
        constant: ArrayD<f32>,
        reverse: bool,
        spec_name: &str,
    ) -> Result<SubConstantLayer> {
        let layer = if reverse {
            SubConstantLayer::try_new_reverse(constant)
        } else {
            SubConstantLayer::try_new(constant)
        };
        layer
            .map_err(|err| NyError::InvalidSpec(format!("Sub {spec_name} constant invalid: {err}")))
    }

    fn pow_constant_layer(&self, exponent: f32, spec_name: &str) -> Result<PowConstantLayer> {
        PowConstantLayer::try_new(exponent)
            .map_err(|err| NyError::InvalidSpec(format!("Pow {spec_name} exponent invalid: {err}")))
    }

    fn mul_constant_layer(
        &self,
        constant: ArrayD<f32>,
        input_name: Option<&str>,
        spec_name: &str,
    ) -> Result<MulConstantLayer> {
        let layer = match input_name.and_then(|name| self.tensor_shape_usize(name)) {
            Some(input_shape) => MulConstantLayer::try_with_input_shape(constant, input_shape),
            None => MulConstantLayer::try_new(constant),
        };
        layer
            .map_err(|err| NyError::InvalidSpec(format!("Mul {spec_name} constant invalid: {err}")))
    }

    fn div_constant_layer(
        &self,
        constant: ArrayD<f32>,
        input_name: &str,
        spec_name: &str,
    ) -> Result<DivConstantLayer> {
        let layer = match self.tensor_shape_usize(input_name) {
            Some(input_shape) => DivConstantLayer::try_with_input_shape(constant, input_shape),
            None => DivConstantLayer::try_new(constant),
        };
        layer
            .map_err(|err| NyError::InvalidSpec(format!("Div {spec_name} constant invalid: {err}")))
    }

    pub(crate) fn check_mul_broadcast(
        &self,
        spec: &LayerSpec,
        input_a: &str,
        input_b: &str,
    ) -> Result<()> {
        let shape_a = self.tensor_shape_usize(input_a);
        let shape_b = self.tensor_shape_usize(input_b);
        if let (Some(a), Some(b)) = (shape_a, shape_b) {
            if broadcast_shapes(&a, &b).is_none() {
                return Err(NyError::ModelLoad(format!(
                    "Mul {} inputs are not broadcast-compatible: {:?} vs {:?}",
                    spec.name, a, b
                )));
            }
        }
        Ok(())
    }
    /// Try to convert Mul to a layer. Returns None if inputs are insufficient.
    pub(crate) fn try_convert_mul(&self, spec: &LayerSpec) -> Result<Option<Layer>> {
        if spec.outputs.len() != 1 {
            debug!(
                "Skipping Mul {} constant folding with {} outputs",
                spec.name,
                spec.outputs.len()
            );
            return Ok(None);
        }
        if spec.inputs.is_empty() {
            debug!("Skipping Mul {} with no inputs", spec.name);
            return Ok(None);
        }
        // ONNX Mul: C = A * B (element-wise)
        // If one input is a constant, create MulConstantLayer.
        // For native models: check for "scale" attribute (attention scaling).
        // Otherwise, treat the operation as MulBinary between two activation inputs.

        // Check for scale attribute first (from native model decomposed attention)
        if let Some(scale_attr) = spec.attributes.get("scale") {
            if spec.inputs.len() != 1 {
                debug!(
                    "Skipping Mul {} with scale attribute and {} inputs",
                    spec.name,
                    spec.inputs.len()
                );
                return Ok(None);
            }
            let scale = match scale_attr {
                AttributeValue::Float(value) => *value,
                AttributeValue::Int(value) => {
                    i64_to_f32_checked(*value, &format!("Mul {} scale attribute", spec.name))?
                }
                _ => {
                    return Ok(None);
                }
            };
            return self
                .mul_constant_layer(ArrayD::from_elem(IxDyn(&[]), scale), None, &spec.name)
                .map(|layer| Some(Layer::MulConstant(layer)));
        }

        if spec.inputs.len() < 2 {
            // Only one input and no scale attribute - not supported
            return Ok(None);
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        // Use retrievable values to determine constant vs activation.
        // A tensor in constant_tensors without a retrievable value (e.g., produced
        // by an unevaluable constant chain) stays as a graph activation node (#3186).
        let a_value = self.constant_value(input_a);
        let b_value = self.constant_value(input_b);

        match (a_value, b_value) {
            (None, Some(constant)) => {
                // x * c
                debug!(
                    "Mul {} is constant multiply (second input '{}', shape {:?})",
                    spec.name,
                    input_b,
                    constant.shape()
                );
                Ok(Some(Layer::MulConstant(self.mul_constant_layer(
                    constant,
                    Some(input_a),
                    &spec.name,
                )?)))
            }
            (Some(constant), None) => {
                // c * x (commutative)
                debug!(
                    "Mul {} is constant multiply (first input '{}', shape {:?})",
                    spec.name,
                    input_a,
                    constant.shape()
                );
                Ok(Some(Layer::MulConstant(self.mul_constant_layer(
                    constant,
                    Some(input_b),
                    &spec.name,
                )?)))
            }
            (Some(a), Some(b)) => {
                // Both constants - fold to the true product.
                let product = if a.len() == 1 && b.len() == 1 {
                    let scalar_a = a.iter().next().copied().unwrap_or(1.0);
                    let scalar_b = b.iter().next().copied().unwrap_or(1.0);
                    ArrayD::from_elem(IxDyn(&[]), scalar_a * scalar_b)
                } else if a.len() == 1 {
                    let scalar = a.iter().next().copied().unwrap_or(1.0);
                    b.mapv(|v| v * scalar)
                } else if b.len() == 1 {
                    let scalar = b.iter().next().copied().unwrap_or(1.0);
                    a.mapv(|v| v * scalar)
                } else {
                    let Some(output_shape) = broadcast_shapes(a.shape(), b.shape()) else {
                        return Ok(None);
                    };
                    let Some(a) = a.broadcast(IxDyn(&output_shape)) else {
                        return Ok(None);
                    };
                    let Some(b) = b.broadcast(IxDyn(&output_shape)) else {
                        return Ok(None);
                    };
                    (&a * &b).into_owned()
                };
                Ok(Some(Layer::MulConstant(
                    self.mul_constant_layer(product, None, &spec.name)?,
                )))
            }
            (None, None) => {
                // Binary multiply of two bounded tensors (e.g., SwiGLU: up * silu(gate))
                Ok(Some(Layer::MulBinary(MulBinaryLayer)))
            }
        }
    }

    /// Convert ONNX Min to MinBinaryLayer.
    ///
    /// ONNX Min is variadic (can take 2+ inputs), but conversion currently emits
    /// exactly one binary `MinBinaryLayer`.
    pub(crate) fn convert_min(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "Min {} requires exactly 2 inputs (got {})",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        // Check if both inputs are constants (should be constant folded upstream)
        let a_is_const = self.weights.get(input_a).is_some();
        let b_is_const = self.weights.get(input_b).is_some();

        if a_is_const && b_is_const {
            // Both constants - should be constant folded upstream.
            // Returning MinBinaryLayer here is unsound: it takes min of the
            // input activation and one constant, ignoring the other (#3021).
            return Err(NyError::UnsupportedOp(format!(
                "Min {} has both constant inputs — should be constant-folded \
                 by the ONNX optimizer before verification",
                spec.name
            )));
        }

        // Check broadcast compatibility if shapes are known
        let shape_a = self.tensor_shape_usize(input_a);
        let shape_b = self.tensor_shape_usize(input_b);
        if let (Some(a), Some(b)) = (&shape_a, &shape_b) {
            if broadcast_shapes(a, b).is_none() {
                return Err(NyError::ModelLoad(format!(
                    "Min {} inputs are not broadcast-compatible: {:?} vs {:?}",
                    spec.name, a, b
                )));
            }
        }

        debug!("Min {} is binary minimum of two activations", spec.name);
        Ok(Layer::MinBinary(MinBinaryLayer))
    }

    /// Convert ONNX Max to MaxBinaryLayer.
    ///
    /// ONNX Max is variadic (can take 2+ inputs), but conversion currently emits
    /// exactly one binary `MaxBinaryLayer`.
    pub(crate) fn convert_max(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "Max {} requires exactly 2 inputs (got {})",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        // Check if both inputs are constants (should be constant folded upstream)
        let a_is_const = self.weights.get(input_a).is_some();
        let b_is_const = self.weights.get(input_b).is_some();

        if a_is_const && b_is_const {
            // Both constants - should be constant folded upstream.
            // Returning MaxBinaryLayer here is unsound: it takes max of the
            // input activation and one constant, ignoring the other (#3021).
            return Err(NyError::UnsupportedOp(format!(
                "Max {} has both constant inputs — should be constant-folded \
                 by the ONNX optimizer before verification",
                spec.name
            )));
        }

        // Check broadcast compatibility if shapes are known
        let shape_a = self.tensor_shape_usize(input_a);
        let shape_b = self.tensor_shape_usize(input_b);
        if let (Some(a), Some(b)) = (&shape_a, &shape_b) {
            if broadcast_shapes(a, b).is_none() {
                return Err(NyError::ModelLoad(format!(
                    "Max {} inputs are not broadcast-compatible: {:?} vs {:?}",
                    spec.name, a, b
                )));
            }
        }

        debug!("Max {} is binary maximum of two activations", spec.name);
        Ok(Layer::MaxBinary(MaxBinaryLayer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    fn make_context() -> (WeightStore, HashMap<String, Vec<i64>>, HashSet<String>) {
        let weights = WeightStore::new();
        let tensor_shapes = HashMap::from([
            ("input".to_string(), vec![1]),
            ("output".to_string(), vec![1]),
        ]);
        let constant_tensors = HashSet::new();
        (weights, tensor_shapes, constant_tensors)
    }

    fn make_spec(layer_type: LayerType, inputs: &[&str]) -> LayerSpec {
        LayerSpec {
            name: "op".to_string(),
            layer_type,
            inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn convert_min_rejects_variadic_inputs() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Min, &["a", "b", "c"]);

        let error = context
            .convert_min(&spec)
            .expect_err("variadic Min should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(message) if message.contains("requires exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_max_rejects_variadic_inputs() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Max, &["a", "b", "c"]);

        let error = context
            .convert_max(&spec)
            .expect_err("variadic Max should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(message) if message.contains("requires exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #2969: non-scalar heterogeneous exponent must return error.
    #[test]
    fn convert_pow_rejects_heterogeneous_exponent_2969() {
        let (mut weights, shapes, constants) = make_context();
        // Insert a non-scalar exponent with different values per element
        weights.insert(
            "exp".to_string(),
            ndarray::arr1(&[2.0f32, 3.0, 4.0]).into_dyn(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Pow, &["input", "exp"]);

        let error = context
            .convert_pow(&spec)
            .expect_err("heterogeneous exponent should be rejected");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(msg) if msg.contains("heterogeneous exponent")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// A homogeneous exponent is safe when broadcasting preserves the base shape.
    #[test]
    fn convert_pow_accepts_shape_preserving_homogeneous_exponent() {
        let (mut weights, mut shapes, constants) = make_context();
        shapes.insert("input".to_string(), vec![3]);
        // Insert a non-scalar exponent where all values are the same
        weights.insert(
            "exp".to_string(),
            ndarray::arr1(&[2.0f32, 2.0, 2.0]).into_dyn(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Pow, &["input", "exp"]);

        let layer = context
            .convert_pow(&spec)
            .expect("homogeneous exponent should succeed");
        assert!(
            matches!(layer, Layer::PowConstant(_)),
            "expected PowConstant, got {layer:?}"
        );
    }

    /// Regression: scalarizing a homogeneous exponent must not discard an ONNX
    /// broadcast that expands the output shape.
    #[test]
    fn convert_pow_rejects_homogeneous_exponent_that_expands_base_shape() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert(
            "exp".to_string(),
            ndarray::arr1(&[2.0f32, 2.0, 2.0]).into_dyn(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Pow, &["input", "exp"]);

        let error = context
            .convert_pow(&spec)
            .expect_err("shape-expanding exponent must be rejected");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(message)
                    if message.contains("would change")
                        && message.contains("base shape [1]")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #2969: scalar exponent should work as before.
    #[test]
    fn convert_pow_accepts_scalar_exponent() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("exp".to_string(), ndarray::arr1(&[2.0f32]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Pow, &["input", "exp"]);

        let layer = context
            .convert_pow(&spec)
            .expect("scalar exponent should succeed");
        assert!(
            matches!(layer, Layer::PowConstant(_)),
            "expected PowConstant, got {layer:?}"
        );
    }

    #[test]
    fn convert_pow_rejects_non_finite_exponent_4307() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("exp".to_string(), ndarray::arr1(&[f32::NAN]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Pow, &["input", "exp"]);

        let error = context
            .convert_pow(&spec)
            .expect_err("non-finite exponent should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg)
                    if msg.contains("Pow op exponent invalid")
                        && msg.contains("must be finite")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #2956: Pow with both constant inputs must return error,
    /// not identity (which silently discards the computation).
    #[test]
    fn convert_pow_rejects_both_constant_inputs_2956() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("base".to_string(), ndarray::arr1(&[4.0f32]).into_dyn());
        weights.insert("exp".to_string(), ndarray::arr1(&[0.5f32]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Pow, &["base", "exp"]);

        let error = context
            .convert_pow(&spec)
            .expect_err("both-constant Pow should be rejected");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(msg) if msg.contains("constant-folded")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #3021: Add with both constant inputs must return error,
    /// not AddLayer (which silently passes input through without adding).
    #[test]
    fn convert_add_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), ndarray::arr1(&[1.0f32]).into_dyn());
        weights.insert("b".to_string(), ndarray::arr1(&[2.0f32]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Add, &["a", "b"]);

        let error = context
            .convert_add(&spec)
            .expect_err("both-constant Add should be rejected");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(msg) if msg.contains("constant-folded")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #3021: Min with both constant inputs must return error,
    /// not MinBinaryLayer (which ignores one constant).
    #[test]
    fn convert_min_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), ndarray::arr1(&[3.0f32]).into_dyn());
        weights.insert("b".to_string(), ndarray::arr1(&[5.0f32]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Min, &["a", "b"]);

        let error = context
            .convert_min(&spec)
            .expect_err("both-constant Min should be rejected");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(msg) if msg.contains("constant-folded")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #3021: Max with both constant inputs must return error,
    /// not MaxBinaryLayer (which ignores one constant).
    #[test]
    fn convert_max_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), ndarray::arr1(&[3.0f32]).into_dyn());
        weights.insert("b".to_string(), ndarray::arr1(&[5.0f32]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Max, &["a", "b"]);

        let error = context
            .convert_max(&spec)
            .expect_err("both-constant Max should be rejected");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(msg) if msg.contains("constant-folded")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #3021: Div with both constant inputs must return error,
    /// not DivConstantLayer (which ignores the constant numerator).
    #[test]
    fn convert_div_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), ndarray::arr1(&[10.0f32]).into_dyn());
        weights.insert("b".to_string(), ndarray::arr1(&[2.0f32]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Div, &["a", "b"]);

        let error = context
            .convert_div(&spec)
            .expect_err("both-constant Div should be rejected");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(msg) if msg.contains("constant-folded")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_add_rejects_non_finite_constant_4307() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert(
            "const".to_string(),
            ndarray::arr1(&[1.0f32, f32::NAN]).into_dyn(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Add, &["input", "const"]);

        let error = context
            .convert_add(&spec)
            .expect_err("non-finite add constant should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg)
                    if msg.contains("Add op constant invalid")
                        && msg.contains("non-finite")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_div_rejects_near_zero_constant_4307() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("divisor".to_string(), ndarray::arr1(&[0.0f32]).into_dyn());
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Div, &["input", "divisor"]);

        let error = context
            .convert_div(&spec)
            .expect_err("near-zero divisor should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg)
                    if msg.contains("Div op constant invalid")
                        && msg.contains("near-zero divisor")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn try_convert_mul_rejects_non_finite_scale_4307() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "mul_scale".to_string(),
            layer_type: LayerType::Mul,
            inputs: vec!["input".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::from([("scale".to_string(), AttributeValue::Float(f32::NAN))]),
        };

        let error = context
            .try_convert_mul(&spec)
            .expect_err("non-finite Mul scale should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg)
                    if msg.contains("Mul mul_scale constant invalid")
                        && msg.contains("non-finite")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #3186: Add with one constant tensor (no value) and one
    /// activation should produce binary AddLayer, not error.
    #[test]
    fn convert_add_constant_tensor_without_value_falls_back_to_binary_3186() {
        let (weights, shapes, _) = make_context();
        // "const_input" is in constant_tensors but has no value in weights
        let constant_tensors = HashSet::from(["const_input".to_string()]);
        let context = ConvertContext::new(&weights, &shapes, &constant_tensors);
        let spec = make_spec(LayerType::Add, &["const_input", "input"]);

        let layer = context
            .convert_add(&spec)
            .expect("Add with value-unavailable constant should succeed as binary");
        assert!(
            matches!(layer, Layer::Add(_)),
            "expected binary AddLayer, got {layer:?}"
        );
    }

    /// Regression test for #3186: Sub with constant tensor (no value) as second
    /// input should produce binary SubLayer.
    #[test]
    fn convert_sub_constant_tensor_without_value_falls_back_to_binary_3186() {
        let (weights, shapes, _) = make_context();
        let constant_tensors = HashSet::from(["const_input".to_string()]);
        let context = ConvertContext::new(&weights, &shapes, &constant_tensors);
        let spec = make_spec(LayerType::Sub, &["input", "const_input"]);

        let layer = context
            .convert_sub(&spec)
            .expect("Sub with value-unavailable constant should succeed as binary");
        assert!(
            matches!(layer, Layer::Sub(_)),
            "expected binary SubLayer, got {layer:?}"
        );
    }

    /// Regression test for #3186: Div with constant-tensor divisor (no value)
    /// should produce binary DivLayer.
    #[test]
    fn convert_div_constant_tensor_without_value_falls_back_to_binary_3186() {
        let (weights, shapes, _) = make_context();
        let constant_tensors = HashSet::from(["const_divisor".to_string()]);
        let context = ConvertContext::new(&weights, &shapes, &constant_tensors);
        let spec = make_spec(LayerType::Div, &["input", "const_divisor"]);

        let layer = context
            .convert_div(&spec)
            .expect("Div with value-unavailable constant divisor should succeed as binary");
        assert!(
            matches!(layer, Layer::Div(_)),
            "expected binary DivLayer, got {layer:?}"
        );
    }

    /// Regression test for #3186: Mul with constant tensor (no value) should
    /// produce binary MulBinaryLayer.
    #[test]
    fn try_convert_mul_constant_tensor_without_value_falls_back_to_binary_3186() {
        let (weights, shapes, _) = make_context();
        let constant_tensors = HashSet::from(["const_scale".to_string()]);
        let context = ConvertContext::new(&weights, &shapes, &constant_tensors);
        let spec = make_spec(LayerType::Mul, &["input", "const_scale"]);

        let layer = context
            .try_convert_mul(&spec)
            .expect("Mul with value-unavailable constant should not error")
            .expect("Mul with value-unavailable constant should succeed as binary");
        assert!(
            matches!(layer, Layer::MulBinary(_)),
            "expected binary MulBinaryLayer, got {layer:?}"
        );
    }

    #[test]
    fn try_convert_mul_broadcasted_constant_product_3500() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert(
            "style_gate".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![2.0, 3.0]).unwrap(),
        );
        weights.insert(
            "inst_norm".to_string(),
            ArrayD::from_shape_vec(
                IxDyn(&[1, 2, 3]),
                vec![
                    1.0, 2.0, 3.0, //
                    4.0, 5.0, 6.0,
                ],
            )
            .unwrap(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Mul, &["style_gate", "inst_norm"]);

        let layer = context
            .try_convert_mul(&spec)
            .expect("broadcasted constant Mul should not error")
            .expect("broadcasted constant Mul should fold to MulConstant");

        let Layer::MulConstant(layer) = layer else {
            panic!("expected MulConstantLayer");
        };
        let expected = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 3]),
            vec![
                2.0, 4.0, 6.0, //
                12.0, 15.0, 18.0,
            ],
        )
        .unwrap();
        assert_eq!(layer.constant(), &expected);
        assert_eq!(layer.input_shape(), None);
    }

    #[test]
    fn try_convert_mul_threads_input_shape_3896() {
        let (mut weights, mut shapes, constants) = make_context();
        shapes.insert("input".to_string(), vec![2, 3]);
        weights.insert(
            "scale".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 3.0]).unwrap(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Mul, &["input", "scale"]);

        let layer = context
            .try_convert_mul(&spec)
            .expect("Mul with retrievable constant should not error")
            .expect("Mul with retrievable constant should convert");

        let Layer::MulConstant(layer) = layer else {
            panic!("expected MulConstantLayer");
        };
        assert_eq!(layer.input_shape(), Some(&[2, 3][..]));
    }

    #[test]
    fn convert_div_threads_input_shape_3896() {
        let (mut weights, mut shapes, constants) = make_context();
        shapes.insert("input".to_string(), vec![2, 3]);
        weights.insert(
            "divisor".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 4.0]).unwrap(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Div, &["input", "divisor"]);

        let layer = context
            .convert_div(&spec)
            .expect("Div with retrievable constant should convert");

        let Layer::DivConstant(layer) = layer else {
            panic!("expected DivConstantLayer");
        };
        assert_eq!(layer.input_shape(), Some(&[2, 3][..]));
    }

    #[test]
    fn convert_layer_rejects_mul_scale_precision_loss_4149() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let mut spec = make_spec(LayerType::Mul, &["input"]);
        spec.attributes
            .insert("scale".to_string(), AttributeValue::Int(16_777_217));

        let error = context
            .convert_layer(&spec)
            .expect_err("precision-losing Mul scale should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg)
                    if msg.contains("precision loss")
                        && msg.contains("Mul op scale attribute")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #3186: Add with one constant tensor (no value) and
    /// one weight (has value) should produce unary AddConstant using the weight.
    #[test]
    fn convert_add_mixed_constant_tensor_and_weight_uses_available_value_3186() {
        let (mut weights, shapes, _) = make_context();
        weights.insert("bias".to_string(), ndarray::arr1(&[1.0f32]).into_dyn());
        let constant_tensors = HashSet::from(["const_input".to_string()]);
        let context = ConvertContext::new(&weights, &shapes, &constant_tensors);

        // const_input (no value) + bias (has value) -> AddConstant(bias)
        let spec = make_spec(LayerType::Add, &["const_input", "bias"]);
        let layer = context
            .convert_add(&spec)
            .expect("Add with one available constant should succeed as unary");
        assert!(
            matches!(layer, Layer::AddConstant(_)),
            "expected AddConstantLayer, got {layer:?}"
        );
    }

    /// Regression test for #2666: Add with extra inputs must be rejected.
    #[test]
    fn convert_add_rejects_extra_inputs_2666() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Add, &["a", "b", "c"]);

        let error = context
            .convert_add(&spec)
            .expect_err("Add with 3 inputs should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #2666: Sub with extra inputs must be rejected.
    #[test]
    fn convert_sub_rejects_extra_inputs_2666() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Sub, &["a", "b", "c"]);

        let error = context
            .convert_sub(&spec)
            .expect_err("Sub with 3 inputs should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #2666: Div with extra inputs must be rejected.
    #[test]
    fn convert_div_rejects_extra_inputs_2666() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Div, &["a", "b", "c"]);

        let error = context
            .convert_div(&spec)
            .expect_err("Div with 3 inputs should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #2666: Pow with extra inputs must be rejected.
    #[test]
    fn convert_pow_rejects_extra_inputs_2666() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Pow, &["input", "exp", "extra"]);

        let error = context
            .convert_pow(&spec)
            .expect_err("Pow with 3 inputs should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }
}
