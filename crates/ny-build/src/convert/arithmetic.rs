// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_propagate::layers::{
    AddConstantLayer, AddLayer, Atan2Layer, ClipLayer, DivConstantLayer, DivLayer,
    IntervalConstLayer, MaxBinaryLayer, MinBinaryLayer, MulBinaryLayer, MulConstantLayer,
    PowConstantLayer, SubConstantLayer, SubLayer,
};
use ny_propagate::network::broadcast_shapes;
use ny_propagate::Layer;
use tracing::{debug, warn};

use super::{i64_to_f32_checked, AttributeValue, ConvertContext, LayerSpec};

/// Which end of the output range a *constant* operand of ONNX `Max`/`Min` pins.
///
/// `max(x, c)` and `min(x, c)` against a constant `c` are not binary ops at all
/// — they are one-sided clamps, and they differ only in which end they clamp.
/// [`ConvertContext::convert_extremum`] is therefore written once, over this
/// distinction, instead of twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Extremum {
    /// `max(x, c)`: `c` is a FLOOR — the output never falls below it.
    Max,
    /// `min(x, c)`: `c` is a CEILING — the output never rises above it.
    Min,
}

impl Extremum {
    /// The ONNX op name, for diagnostics.
    fn op_name(self) -> &'static str {
        match self {
            Extremum::Max => "Max",
            Extremum::Min => "Min",
        }
    }

    /// The genuine binary layer, for the both-operands-are-activations case.
    fn binary_layer(self) -> Layer {
        match self {
            Extremum::Max => Layer::MaxBinary(MaxBinaryLayer),
            Extremum::Min => Layer::MinBinary(MinBinaryLayer),
        }
    }

    /// The `(min, max)` pair for [`ClipLayer`] that reproduces this extremum
    /// against the scalar constant `c` — EXACTLY, not as a relaxation.
    ///
    /// `clip(x, lo, hi) = max(lo, min(hi, x))`, so the two identities are
    ///
    /// ```text
    /// max(x, c) == clip(x,    c, +inf)   // upper end left open
    /// min(x, c) == clip(x, -inf,    c)   // lower end left open
    /// ```
    ///
    /// # Why the open end is a true infinity and not a finite sentinel
    ///
    /// 1. `validate_clip_bounds` (ny-propagate `layers/activations/validate.rs`)
    ///    rejects only NaN and `min > max`. ±Inf is explicitly admitted.
    /// 2. It is already a live, exercised path: `convert_clip` in
    ///    `elementwise.rs` defaults a *missing* ONNX `Clip` bound to exactly
    ///    `f32::NEG_INFINITY` / `f32::INFINITY` and hands it to this same
    ///    constructor, so every one-sided ONNX `Clip` in the corpus already
    ///    builds this layer shape.
    /// 3. A sentinel would be the *unsound* option, not the safe one. Any
    ///    finite stand-in for `+inf` is a real clamp, so an input above it gets
    ///    silently pulled DOWN and the layer bounds a smaller function than
    ///    `max(x, c)` — an under-approximation of the upper bound, i.e. a false
    ///    proof. No finite f32 is safe for every reachable input, because the
    ///    reachable range is precisely what verification is trying to establish.
    ///
    /// # Soundness against `ClipLayer`'s two propagation paths
    ///
    /// * **IBP** (`ClipLayer::propagate_ibp`) is `f32::clamp` per endpoint.
    ///   `clamp` is pure comparison, never arithmetic, so none of the
    ///   NaN-producing infinity patterns (`inf - inf`, `0 * inf`, `inf / inf`)
    ///   can arise; `clamp(v, c, +inf) == max(v, c)` and
    ///   `clamp(v, -inf, c) == min(v, c)` hold exactly for every `v`, finite or
    ///   not. The layer builds its result with
    ///   `BoundedTensor::new_allow_infinite`, which is what makes an infinite
    ///   endpoint legal there in the first place.
    /// * **CROWN** (`clip_linear_relaxation`) never lets the open end reach the
    ///   arithmetic. Its entry point is guarded by `non_finite_domain_guard`,
    ///   so `l` and `u` are finite. With `max = +inf` every branch that reads
    ///   `max_val` (`l >= max_val`, the crosses-both-boundaries case, and the
    ///   crosses-upper-only case) needs `l >= +inf` or `u > +inf` and is
    ///   unreachable — leaving the exact `u <= min_val` constant branch, the
    ///   exact identity branch, and the crosses-lower-only chord, none of which
    ///   mentions `max_val`. `min = -inf` is the mirror image.
    fn clip_bounds(self, constant: f32) -> (f32, f32) {
        match self {
            Extremum::Max => (constant, f32::INFINITY),
            Extremum::Min => (f32::NEG_INFINITY, constant),
        }
    }
}

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

        // An interval-valued constant addend is a pure additive ENCLOSURE, so it
        // lowers to the offset form of the interval carrier. Only the
        // `activation + interval` shape is admitted; anything else refuses and
        // degrades to a sound unbounded skip.
        if let Some(layer) =
            self.try_convert_interval_offset(spec, input_a, input_b, false, "Add")?
        {
            return Ok(layer);
        }

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
        // A constant without a retrievable value also takes the binary path.

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
        // Otherwise (including a constant without a retrievable value), use binary SubLayer.

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

        // `activation - interval` is an additive enclosure with the constant
        // negated (negation is exact in binary32). `interval - activation` is
        // NOT carried: it negates the coefficient as well, which the offset form
        // does not do — it refuses inside the helper.
        if let Some(layer) =
            self.try_convert_interval_offset(spec, input_a, input_b, true, "Sub")?
        {
            return Ok(layer);
        }

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

    /// Lower `activation +- interval_constant` to the offset form of
    /// [`Layer::IntervalConst`], or refuse.
    ///
    /// Returns `Ok(None)` when neither operand is interval-valued, so the
    /// caller's existing point logic runs untouched. Returns
    /// `Err(UnsupportedOp)` — a sound unbounded skip — for every interval shape
    /// this carrier does not handle: the constant on the LEFT (which would have
    /// to negate the activation's coefficient too, for `Sub`), or both operands
    /// interval-valued (which the const-folder should have folded).
    fn try_convert_interval_offset(
        &self,
        spec: &LayerSpec,
        input_a: &str,
        input_b: &str,
        subtract: bool,
        op: &str,
    ) -> Result<Option<Layer>> {
        let a_interval = self.weights.get_interval(input_a).is_some();
        let b_interval = self.weights.get_interval(input_b).is_some();
        if !a_interval && !b_interval {
            return Ok(None);
        }
        if a_interval || (b_interval && self.constant_value(input_a).is_some()) {
            warn!(
                "{op} {} has an interval-valued constant in a position the carrier does not \
                 support (a_interval={a_interval}, b_interval={b_interval}); failing closed \
                 to a sound skip",
                spec.name
            );
            return Err(NyError::UnsupportedOp(format!(
                "{op} {} has an interval-valued constant operand in an unsupported position",
                spec.name
            )));
        }
        let (lower, upper) = self.weights.get_interval(input_b).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "{op} {} missing interval constant input '{input_b}'",
                spec.name
            ))
        })?;
        let context = format!("{op} {}", spec.name);
        let (center, radius) = super::interval_const::center_and_radius(lower, upper, &context)?;
        // `x - c` is `x + (-c)`; negation is exact and leaves the radius alone.
        let center = if subtract { -center } else { center };
        // The carrier's backward step keeps the coefficients unchanged, which is
        // an identity only if the constant broadcasts INTO the activation. Pin
        // the activation's flat width here, from its declared shape, and refuse
        // when that shape is unknown — a guess would be a silent expansion.
        let input_len = self
            .tensor_shapes
            .get(input_a)
            .and_then(|shape| {
                shape.iter().try_fold(1usize, |acc, &d| {
                    usize::try_from(d).ok().map(|d| acc * d.max(1))
                })
            })
            .ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "{op} {} reads an interval-valued constant but the activation's shape is \
                     not statically known; refusing rather than guessing the broadcast",
                    spec.name
                ))
            })?;
        let layer = IntervalConstLayer::try_new_offset(center, radius, input_len)?;
        debug!(
            "{op} {} with an interval-valued constant converted to {} (max radius {:e})",
            spec.name,
            layer.kind_name(),
            layer.max_radius()
        );
        Ok(Some(Layer::IntervalConst(layer)))
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

    /// Convert an Atan2 spec to [`Atan2Layer`].
    ///
    /// Inputs are `(y, x)` — the `f32::atan2` / nn tracer order, which
    /// `ny-trace-bridge`'s `translate_atan2` emits.
    ///
    /// Without this arm the layer was UNREACHABLE from any front end that
    /// builds through `GraphModel`: `convert_layer` fell through to its
    /// catch-all `UnsupportedOp`, which `graph::builder` then swallows into a
    /// conservative `OpaqueSkip` — so an Atan2 node silently produced
    /// `[-inf, +inf]` instead of `Atan2Layer`'s quadrant-aware enclosure, with
    /// no error anywhere to say verification had gone vacuous. The bridge test
    /// that "covered" it only asserted the graph BUILT, which an OpaqueSkip
    /// substitution satisfies.
    pub(crate) fn convert_atan2(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "Atan2 {} requires exactly 2 inputs (y, x), got {}",
                spec.name,
                spec.inputs.len()
            )));
        }
        Ok(Layer::Atan2(Atan2Layer))
    }

    /// Convert ONNX `Min` — see [`ConvertContext::convert_extremum`].
    pub(crate) fn convert_min(&self, spec: &LayerSpec) -> Result<Layer> {
        self.convert_extremum(spec, Extremum::Min)
    }

    /// Convert ONNX `Max` — see [`ConvertContext::convert_extremum`].
    pub(crate) fn convert_max(&self, spec: &LayerSpec) -> Result<Layer> {
        self.convert_extremum(spec, Extremum::Max)
    }

    /// Convert a 2-input ONNX `Max`/`Min` node.
    ///
    /// Three operand shapes are handled, in this order:
    ///
    /// * **both operands constant** — refused. Emitting a binary layer
    ///   would take the extremum of one constant against whatever activation
    ///   the graph builder wired in the other slot, silently ignoring the
    ///   second constant. Constant-folding is the ONNX optimizer's job.
    /// * **one operand constant** — lowered to [`Layer::Clip`], a one-sided
    ///   clamp. `maximum(x, eps)` is a real and common op: it is how models
    ///   floor a value before a `sqrt` or a `log` for numerical stability
    ///   (ECAPA-TDNN's attentive-stat-pooling variance does exactly this). See
    ///   [`Extremum::clip_bounds`] for the identity and its soundness argument,
    ///   and [`ConvertContext::extremum_clamp_scalar`] for the two conditions
    ///   the constant must meet before it may be absorbed.
    /// * **neither operand constant** — the genuine binary extremum.
    ///
    /// # Why the constant test is `constant_value`, not `weights` alone
    ///
    /// It has to agree with `find_graph_input_nodes` (`graph/inputs.rs`), which
    /// drops an operand from a layer's activation-input list when EITHER
    /// `weights` OR `evaluated_constants` holds it. A converter that consults
    /// only `weights` can emit a *binary* layer whose second operand the graph
    /// builder then removes, and the load dies downstream with "expects 2
    /// activation inputs for binary op, got 1" — an error that names the graph
    /// builder rather than the converter that actually mis-classified the
    /// operand. `inputs.rs` states the contract in its own comment: the
    /// converter "should have already created unary variants for binary ops
    /// with constant tensor inputs via constant_value()".
    ///
    /// A constant tensor whose value could NOT be materialized still yields
    /// `None` here, and that is correct rather than a gap: `inputs.rs` keeps
    /// such an operand as an activation (its producing node stays in the graph
    /// and propagates bounds), so the binary lowering is the matching one.
    /// Interval-valued frozen constants likewise stay `None` — `WeightStore`
    /// keeps point and interval maps disjoint — and so keep their existing
    /// behaviour untouched by this lowering.
    ///
    /// ONNX `Max`/`Min` are variadic; conversion still admits exactly 2 inputs.
    fn convert_extremum(&self, spec: &LayerSpec, extremum: Extremum) -> Result<Layer> {
        let op = extremum.op_name();
        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "{op} {} requires exactly 2 inputs (got {})",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        let a_value = self.constant_value(input_a);
        let b_value = self.constant_value(input_b);

        if a_value.is_some() && b_value.is_some() {
            return Err(NyError::UnsupportedOp(format!(
                "{op} {} has both constant inputs — should be constant-folded \
                 by the ONNX optimizer before verification",
                spec.name
            )));
        }

        // Broadcast compatibility, checked before committing to a lowering so a
        // malformed node is reported as malformed rather than as an
        // unsupported clamp.
        let shape_a = self.tensor_shape_usize(input_a);
        let shape_b = self.tensor_shape_usize(input_b);
        if let (Some(a), Some(b)) = (&shape_a, &shape_b) {
            if broadcast_shapes(a, b).is_none() {
                return Err(NyError::ModelLoad(format!(
                    "{op} {} inputs are not broadcast-compatible: {:?} vs {:?}",
                    spec.name, a, b
                )));
            }
        }

        // `max`/`min` are commutative, so which side carries the constant does
        // not change the clamp — only which operand remains an activation.
        let constant_operand = match (&a_value, &b_value) {
            (Some(value), None) => Some((value, input_a.as_str(), input_b.as_str())),
            (None, Some(value)) => Some((value, input_b.as_str(), input_a.as_str())),
            _ => None,
        };

        if let Some((value, constant_name, activation_name)) = constant_operand {
            let threshold =
                self.extremum_clamp_scalar(spec, op, value, constant_name, activation_name)?;
            let (min, max) = extremum.clip_bounds(threshold);
            let layer = ClipLayer::try_new(min, max).map_err(|error| {
                NyError::ModelLoad(format!(
                    "{op} {} could not be lowered to a one-sided clamp at {threshold}: {error}",
                    spec.name
                ))
            })?;
            debug!(
                "{op} {} is a one-sided clamp against constant '{constant_name}' = {threshold} \
                 (lowered to Clip[{min}, {max}])",
                spec.name
            );
            return Ok(Layer::Clip(layer));
        }

        debug!("{op} {} is a binary extremum of two activations", spec.name);
        Ok(extremum.binary_layer())
    }

    /// Reduce a constant `Max`/`Min` operand to the single scalar threshold a
    /// [`ClipLayer`] can carry — or refuse.
    ///
    /// `ClipLayer` holds SCALAR `min`/`max`, so this collapse is admissible only
    /// when it is exact. Two independent conditions must hold, and both are
    /// checked rather than assumed: this is a bound-propagation path, where
    /// bounding the wrong function is a false proof, not merely a loose bound.
    ///
    /// 1. **Every element of the constant is the same value.** A genuinely
    ///    elementwise constant — a per-channel floor, say — has no scalar
    ///    representative at all. Collapsing it to one value (a mean, a min, the
    ///    first element) would clamp every channel against the wrong threshold
    ///    and bound a DIFFERENT function, so it is refused, never approximated.
    ///
    /// 2. **The constant cannot enlarge the activation's shape.** ONNX
    ///    `Max`/`Min` broadcast, so `max(x[4], c[8])` legitimately produces an
    ///    `[8]` output, whereas `ClipLayer` is elementwise and would keep `[4]`.
    ///    Condition 1 does not imply this one — an all-equal `c[8]` still
    ///    expands — so the broadcast of the two operand shapes is required to
    ///    equal the ACTIVATION's shape.
    ///
    ///    A rank-0 constant is exempt and needs no shape lookup at all: `[]`
    ///    broadcasts against every shape to that same shape, by definition.
    ///    Otherwise both shapes must be statically known and the broadcast must
    ///    be shape-preserving. An unknown shape is refused rather than guessed,
    ///    because a wrong guess here is a silent shape change — the same stance
    ///    [`ConvertContext::try_convert_interval_offset`] takes above.
    ///
    /// Non-finite constants are refused as well. `validate_clip_bounds` would
    /// admit ±Inf, and `max(x, -inf) = x` is even exact, but a non-finite
    /// threshold in a real model signals a broken export rather than an intended
    /// clamp, and `max(x, +inf)` would emit an all-`+inf` layer output that
    /// poisons every downstream bound. Failing closed keeps that out of the
    /// graph; the caller's refusal degrades to a sound skip.
    fn extremum_clamp_scalar(
        &self,
        spec: &LayerSpec,
        op: &str,
        constant: &ArrayD<f32>,
        constant_name: &str,
        activation_name: &str,
    ) -> Result<f32> {
        let Some(&threshold) = constant.iter().next() else {
            return Err(NyError::ModelLoad(format!(
                "{op} {} constant input '{constant_name}' is empty, so it carries no \
                 clamp threshold",
                spec.name
            )));
        };

        if !threshold.is_finite() || constant.iter().any(|value| !value.is_finite()) {
            return Err(NyError::UnsupportedOp(format!(
                "{op} {} clamps against non-finite constant '{constant_name}'; refusing \
                 rather than admitting a threshold that cannot soundly bound anything",
                spec.name
            )));
        }

        // Condition 1. Plain value equality, not bitwise: `-0.0` and `0.0` are
        // the same clamp, and NaN — the only value where `!=` would misbehave —
        // is already excluded by the finiteness check above.
        if constant.iter().any(|&value| value != threshold) {
            return Err(NyError::UnsupportedOp(format!(
                "{op} {} has an elementwise constant operand '{constant_name}' (shape \
                 {:?}, elements are not all equal); ny can only absorb a UNIFORM \
                 threshold into the scalar-bounded Clip layer, and collapsing distinct \
                 per-element thresholds to one value would bound a different function",
                spec.name,
                constant.shape()
            )));
        }

        // Condition 2. `tensor_shapes` is the authoritative ONNX shape record
        // and is the convention the activation's shape is read in too, so both
        // sides of the broadcast comparison come from it. Fall back to the
        // materialized array only to recognise rank-0, where no comparison is
        // needed.
        let constant_shape = self.tensor_shape_usize(constant_name);
        let is_rank_zero = match &constant_shape {
            Some(shape) => shape.is_empty(),
            None => constant.ndim() == 0,
        };
        if is_rank_zero {
            return Ok(threshold);
        }

        let (Some(constant_shape), Some(activation_shape)) =
            (constant_shape, self.tensor_shape_usize(activation_name))
        else {
            return Err(NyError::UnsupportedOp(format!(
                "{op} {} clamps against non-scalar constant '{constant_name}', but the \
                 operand shapes are not both statically known; refusing rather than \
                 guessing whether the broadcast enlarges the activation",
                spec.name
            )));
        };

        let broadcast = broadcast_shapes(&activation_shape, &constant_shape).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "{op} {} inputs are not broadcast-compatible: {activation_shape:?} vs \
                 {constant_shape:?}",
                spec.name
            ))
        })?;
        if broadcast != activation_shape {
            return Err(NyError::UnsupportedOp(format!(
                "{op} {} broadcasts activation '{activation_name}' {activation_shape:?} \
                 against constant '{constant_name}' {constant_shape:?} to {broadcast:?}, \
                 but the elementwise Clip lowering would keep {activation_shape:?}; \
                 refusing rather than silently reshaping the graph",
                spec.name
            )));
        }

        Ok(threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ndarray::arr1;
    use ny_core::LayerType;
    use ny_propagate::BoundPropagation;
    use ny_tensor::BoundedTensor;
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

    /// Regression test: non-scalar heterogeneous exponent must return error.
    #[test]
    fn convert_pow_rejects_heterogeneous_exponent_2969() {
        let (mut weights, shapes, constants) = make_context();
        // Insert a non-scalar exponent with different values per element
        weights.insert("exp".to_string(), arr1(&[2.0f32, 3.0, 4.0]).into_dyn());
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
        weights.insert("exp".to_string(), arr1(&[2.0f32, 2.0, 2.0]).into_dyn());
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
        weights.insert("exp".to_string(), arr1(&[2.0f32, 2.0, 2.0]).into_dyn());
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

    /// Regression test: scalar exponent should work as before.
    #[test]
    fn convert_pow_accepts_scalar_exponent() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("exp".to_string(), arr1(&[2.0f32]).into_dyn());
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
        weights.insert("exp".to_string(), arr1(&[f32::NAN]).into_dyn());
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

    /// Regression test: Pow with both constant inputs must return error,
    /// not identity (which silently discards the computation).
    #[test]
    fn convert_pow_rejects_both_constant_inputs_2956() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("base".to_string(), arr1(&[4.0f32]).into_dyn());
        weights.insert("exp".to_string(), arr1(&[0.5f32]).into_dyn());
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

    /// Regression test: Add with both constant inputs must return error,
    /// not AddLayer (which silently passes input through without adding).
    #[test]
    fn convert_add_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), arr1(&[1.0f32]).into_dyn());
        weights.insert("b".to_string(), arr1(&[2.0f32]).into_dyn());
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

    /// Regression test: Min with both constant inputs must return error,
    /// not MinBinaryLayer (which ignores one constant).
    #[test]
    fn convert_min_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), arr1(&[3.0f32]).into_dyn());
        weights.insert("b".to_string(), arr1(&[5.0f32]).into_dyn());
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

    /// Regression test: Max with both constant inputs must return error,
    /// not MaxBinaryLayer (which ignores one constant).
    #[test]
    fn convert_max_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), arr1(&[3.0f32]).into_dyn());
        weights.insert("b".to_string(), arr1(&[5.0f32]).into_dyn());
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

    /// Sample points spanning each element's interval, for enclosure checks.
    fn interval_samples(lower: f32, upper: f32) -> [f32; 3] {
        [lower, f32::midpoint(lower, upper), upper]
    }

    /// `max(x, eps)` with a scalar constant lowers to a one-sided clamp, and the
    /// resulting layer's IBP ENCLOSES the concrete `max(x, eps)`.
    ///
    /// Before this lowering existed, one constant operand still produced
    /// `Layer::MaxBinary`; `find_graph_input_nodes` then filtered the constant
    /// out of the activation list and failed the whole load with "expects 2
    /// activation inputs for binary op, got 1". ECAPA-TDNN hits this with the
    /// `maximum(x, 1e-7)` that floors its pooled variance before a `sqrt`.
    #[test]
    fn convert_max_with_scalar_constant_lowers_to_one_sided_clip_and_encloses() {
        let (mut weights, shapes, constants) = make_context();
        // Rank-0, exactly as the ECAPA trace serializes it.
        weights.insert("eps".to_string(), ArrayD::from_elem(IxDyn(&[]), 1e-7f32));
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Max, &["input", "eps"]);

        let layer = context
            .convert_max(&spec)
            .expect("max(x, eps) must lower to a clamp, not a binary op");
        let Layer::Clip(clip) = &layer else {
            panic!("expected a one-sided Clip, got {layer:?}");
        };
        assert_eq!(clip.min, 1e-7f32, "max(x, c) floors the output at c");
        assert_eq!(
            clip.max,
            f32::INFINITY,
            "max(x, c) must leave the UPPER end open; any finite ceiling would \
             clamp values the true function passes through unchanged, bounding \
             a smaller function than max(x, c)"
        );

        // Enclosure: intervals that sit below, across, and above the threshold.
        let lower = arr1(&[-2.0f32, -1.0, -1e-9, 1.0]).into_dyn();
        let upper = arr1(&[-1.0f32, 0.5, 1e-9, 3.0]).into_dyn();
        let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("input bounds");
        let output = clip.propagate_ibp(&input).expect("Clip IBP");

        let (lower, upper) = (
            lower.as_slice().expect("contiguous lower"),
            upper.as_slice().expect("contiguous upper"),
        );
        let out_lower = output.lower().as_slice().expect("contiguous out lower");
        let out_upper = output.upper().as_slice().expect("contiguous out upper");
        for index in 0..lower.len() {
            for sample in interval_samples(lower[index], upper[index]) {
                let concrete = sample.max(1e-7f32);
                assert!(
                    out_lower[index] <= concrete && concrete <= out_upper[index],
                    "IBP bound [{}, {}] at index {index} must enclose \
                     max({sample}, 1e-7) = {concrete}",
                    out_lower[index],
                    out_upper[index]
                );
            }
        }
    }

    /// The `min` mirror: `min(x, c)` lowers to `Clip(-inf, c)` and encloses.
    #[test]
    fn convert_min_with_scalar_constant_lowers_to_one_sided_clip_and_encloses() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("ceiling".to_string(), ArrayD::from_elem(IxDyn(&[]), 0.5f32));
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Min, &["input", "ceiling"]);

        let layer = context
            .convert_min(&spec)
            .expect("min(x, c) must lower to a clamp, not a binary op");
        let Layer::Clip(clip) = &layer else {
            panic!("expected a one-sided Clip, got {layer:?}");
        };
        assert_eq!(clip.max, 0.5f32, "min(x, c) caps the output at c");
        assert_eq!(
            clip.min,
            f32::NEG_INFINITY,
            "min(x, c) must leave the LOWER end open"
        );

        let lower = arr1(&[-2.0f32, 0.0, 0.5, 1.0]).into_dyn();
        let upper = arr1(&[-1.0f32, 1.0, 0.5, 3.0]).into_dyn();
        let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("input bounds");
        let output = clip.propagate_ibp(&input).expect("Clip IBP");

        let (lower, upper) = (
            lower.as_slice().expect("contiguous lower"),
            upper.as_slice().expect("contiguous upper"),
        );
        let out_lower = output.lower().as_slice().expect("contiguous out lower");
        let out_upper = output.upper().as_slice().expect("contiguous out upper");
        for index in 0..lower.len() {
            for sample in interval_samples(lower[index], upper[index]) {
                let concrete = sample.min(0.5f32);
                assert!(
                    out_lower[index] <= concrete && concrete <= out_upper[index],
                    "IBP bound [{}, {}] at index {index} must enclose \
                     min({sample}, 0.5) = {concrete}",
                    out_lower[index],
                    out_upper[index]
                );
            }
        }
    }

    /// The constant test must agree with `find_graph_input_nodes`, which filters
    /// on `weights` OR `evaluated_constants`. A constant reachable only through
    /// `evaluated_constants` must therefore lower to a clamp too — consulting
    /// `weights` alone would emit a binary layer whose operand the graph builder
    /// then deletes.
    #[test]
    fn convert_max_absorbs_a_constant_that_lives_only_in_evaluated_constants() {
        let (weights, shapes, constants) = make_context();
        let evaluated =
            HashMap::from([("folded".to_string(), ArrayD::from_elem(IxDyn(&[]), 2.0f32))]);
        let context =
            ConvertContext::with_evaluated_constants(&weights, &shapes, &constants, &evaluated);
        let spec = make_spec(LayerType::Max, &["input", "folded"]);

        let layer = context
            .convert_max(&spec)
            .expect("a const-folded operand must lower to a clamp");
        let Layer::Clip(clip) = &layer else {
            panic!("expected a one-sided Clip, got {layer:?}");
        };
        assert_eq!(clip.min, 2.0f32);
    }

    /// A genuinely ELEMENTWISE constant has no scalar representative, so it must
    /// be refused rather than collapsed. Collapsing it would clamp every element
    /// against the wrong threshold — a different function, i.e. a false proof.
    #[test]
    fn convert_max_refuses_an_elementwise_constant_rather_than_collapsing_it() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert(
            "per_channel_floor".to_string(),
            arr1(&[0.0f32, 1.0, 2.0]).into_dyn(),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Max, &["input", "per_channel_floor"]);

        let error = context
            .convert_max(&spec)
            .expect_err("a non-uniform constant must be refused");
        assert!(
            matches!(&error, NyError::UnsupportedOp(msg) if msg.contains("elementwise")),
            "unexpected error: {error:?}"
        );
    }

    /// An all-equal constant is still refused when broadcasting would ENLARGE
    /// the activation: `ClipLayer` is elementwise and would silently keep the
    /// smaller shape. Uniform values are not sufficient — the shape must be too.
    #[test]
    fn convert_max_refuses_a_constant_that_broadcasts_the_activation_larger() {
        let (mut weights, mut shapes, constants) = make_context();
        // Activation "input" is [1]; an all-equal [4] constant broadcasts the
        // ONNX output to [4], which the Clip lowering could not reproduce.
        weights.insert(
            "wide".to_string(),
            arr1(&[2.0f32, 2.0, 2.0, 2.0]).into_dyn(),
        );
        shapes.insert("wide".to_string(), vec![4]);
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Max, &["input", "wide"]);

        let error = context
            .convert_max(&spec)
            .expect_err("a shape-enlarging constant must be refused");
        assert!(
            matches!(&error, NyError::UnsupportedOp(msg) if msg.contains("broadcasts")),
            "unexpected error: {error:?}"
        );
    }

    /// A non-finite threshold is refused. `validate_clip_bounds` would admit it,
    /// but `max(x, +inf)` emits an all-`+inf` output that poisons every
    /// downstream bound, and a non-finite constant signals a broken export.
    #[test]
    fn convert_max_refuses_a_non_finite_constant() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert(
            "inf".to_string(),
            ArrayD::from_elem(IxDyn(&[]), f32::INFINITY),
        );
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Max, &["input", "inf"]);

        let error = context
            .convert_max(&spec)
            .expect_err("a non-finite threshold must be refused");
        assert!(
            matches!(&error, NyError::UnsupportedOp(msg) if msg.contains("non-finite")),
            "unexpected error: {error:?}"
        );
    }

    /// Two ACTIVATION operands must still produce the genuine binary layer —
    /// the clamp lowering must not swallow the ordinary case.
    #[test]
    fn convert_max_with_two_activations_stays_binary() {
        let (weights, shapes, constants) = make_context();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = make_spec(LayerType::Max, &["input", "output"]);

        let layer = context.convert_max(&spec).expect("binary max must lower");
        assert!(
            matches!(layer, Layer::MaxBinary(_)),
            "expected MaxBinary, got {layer:?}"
        );
    }

    /// Regression test: Div with both constant inputs must return error,
    /// not DivConstantLayer (which ignores the constant numerator).
    #[test]
    fn convert_div_rejects_both_constant_inputs_3021() {
        let (mut weights, shapes, constants) = make_context();
        weights.insert("a".to_string(), arr1(&[10.0f32]).into_dyn());
        weights.insert("b".to_string(), arr1(&[2.0f32]).into_dyn());
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
        weights.insert("const".to_string(), arr1(&[1.0f32, f32::NAN]).into_dyn());
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
        weights.insert("divisor".to_string(), arr1(&[0.0f32]).into_dyn());
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

    /// Regression test: Add with one constant tensor (no value) and one
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

    /// Regression test: Sub with constant tensor (no value) as second
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

    /// Regression test: Div with constant-tensor divisor (no value)
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

    /// Regression test: Mul with constant tensor (no value) should
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

    /// Regression test: Add with one constant tensor (no value) and
    /// one weight (has value) should produce unary AddConstant using the weight.
    #[test]
    fn convert_add_mixed_constant_tensor_and_weight_uses_available_value_3186() {
        let (mut weights, shapes, _) = make_context();
        weights.insert("bias".to_string(), arr1(&[1.0f32]).into_dyn());
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

    /// Regression test: Add with extra inputs must be rejected.
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

    /// Regression test: Sub with extra inputs must be rejected.
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

    /// Regression test: Div with extra inputs must be rejected.
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

    /// Regression test: Pow with extra inputs must be rejected.
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
