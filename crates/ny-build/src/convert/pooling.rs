// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{AttributeValue, ConvertContext, LayerSpec};
use ny_core::{NyError, Result};
use ny_propagate::layers::{AveragePoolLayer, MaxPool2dLayer};
use ny_propagate::Layer;

fn parse_non_negative_usize(
    value: i64,
    op_name: &str,
    layer_name: &str,
    attr_name: &str,
) -> Result<usize> {
    if value < 0 {
        return Err(NyError::ModelLoad(format!(
            "{} layer {} has negative {} value {}",
            op_name, layer_name, attr_name, value
        )));
    }
    Ok(value as usize)
}

/// Validate that pool strides are >= 1 to prevent divide-by-zero in output-size
/// formulas. Part of #2872 (pool stride=0 panic cliff).
fn validate_pool_strides(
    strides: (usize, usize),
    op_name: &str,
    layer_name: &str,
) -> Result<(usize, usize)> {
    if strides.0 == 0 || strides.1 == 0 {
        return Err(NyError::ModelLoad(format!(
            "{} layer {} has zero stride {:?} (must be >= 1)",
            op_name, layer_name, strides
        )));
    }
    Ok(strides)
}

fn parse_pair_attr(
    spec: &LayerSpec,
    op_name: &str,
    attr_name: &str,
    default: (usize, usize),
) -> Result<(usize, usize)> {
    match spec.attributes.get(attr_name) {
        None => Ok(default),
        Some(AttributeValue::Int(v)) => {
            let value = parse_non_negative_usize(*v, op_name, &spec.name, attr_name)?;
            Ok((value, value))
        }
        Some(AttributeValue::Ints(values)) if values.is_empty() => Ok(default),
        Some(AttributeValue::Ints(values)) if values.len() == 1 => {
            let value = parse_non_negative_usize(values[0], op_name, &spec.name, attr_name)?;
            Ok((value, value))
        }
        Some(AttributeValue::Ints(values)) if values.len() == 2 => Ok((
            parse_non_negative_usize(values[0], op_name, &spec.name, attr_name)?,
            parse_non_negative_usize(values[1], op_name, &spec.name, attr_name)?,
        )),
        Some(AttributeValue::Ints(values)) => Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid {} length {}, expected 1 or 2",
            op_name,
            spec.name,
            attr_name,
            values.len()
        ))),
        Some(_) => Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid {} attribute type",
            op_name, spec.name, attr_name
        ))),
    }
}

/// Reject max-pool padding that meets or exceeds the kernel extent: such
/// geometry admits pooling windows made entirely of padding, whose max (over
/// an empty set of inputs) is undefined, and ONNX Runtime refuses to run it.
/// ny-propagate's MaxPool2d output_size rejects it too; failing here surfaces
/// the malformed model at load time instead of at first propagation.
fn validate_max_pool_padding(
    pads: (usize, usize),
    kernel_shape: (usize, usize),
    layer_name: &str,
) -> Result<(usize, usize)> {
    if pads.0 >= kernel_shape.0 || pads.1 >= kernel_shape.1 {
        return Err(NyError::ModelLoad(format!(
            "MaxPool layer {} has pads {:?} not smaller than kernel_shape {:?} (each pad must be < kernel)",
            layer_name, pads, kernel_shape
        )));
    }
    Ok(pads)
}

fn parse_symmetric_pool_padding(spec: &LayerSpec, op_name: &str) -> Result<(usize, usize)> {
    match spec.attributes.get("pads") {
        None => Ok((0, 0)),
        Some(AttributeValue::Int(v)) => {
            let pad = parse_non_negative_usize(*v, op_name, &spec.name, "pads")?;
            Ok((pad, pad))
        }
        Some(AttributeValue::Ints(values)) if values.is_empty() => Ok((0, 0)),
        Some(AttributeValue::Ints(values)) if values.len() == 1 => {
            let pad = parse_non_negative_usize(values[0], op_name, &spec.name, "pads")?;
            Ok((pad, pad))
        }
        Some(AttributeValue::Ints(values)) if values.len() == 2 => Ok((
            parse_non_negative_usize(values[0], op_name, &spec.name, "pads")?,
            parse_non_negative_usize(values[1], op_name, &spec.name, "pads")?,
        )),
        Some(AttributeValue::Ints(values)) if values.len() == 4 => {
            if values[0] != values[2] || values[1] != values[3] {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{} layer {} uses asymmetric pads {:?}, which is not supported",
                    op_name, spec.name, values
                )));
            }
            Ok((
                parse_non_negative_usize(values[0], op_name, &spec.name, "pads")?,
                parse_non_negative_usize(values[1], op_name, &spec.name, "pads")?,
            ))
        }
        Some(AttributeValue::Ints(values)) => Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid pads length {}, expected 1, 2, or 4",
            op_name,
            spec.name,
            values.len()
        ))),
        Some(_) => Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid pads attribute type",
            op_name, spec.name
        ))),
    }
}

impl ConvertContext<'_> {
    fn ensure_supported_pool_auto_pad(&self, spec: &LayerSpec, op_name: &str) -> Result<()> {
        if let Some(attr) = spec.attributes.get("auto_pad") {
            let auto_pad = match attr {
                AttributeValue::String(s) => s.trim().to_ascii_uppercase(),
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "{} layer {} has invalid auto_pad attribute type",
                        op_name, spec.name
                    )));
                }
            };
            if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{} layer {} uses auto_pad={}, which is not supported",
                    op_name, spec.name, auto_pad
                )));
            }
        }
        Ok(())
    }

    /// Reject unsupported ONNX pooling attributes that would silently produce
    /// wrong bounds if ignored. See ONNX MaxPool/AveragePool specs.
    fn ensure_supported_pool_attributes(
        &self,
        spec: &LayerSpec,
        op_name: &str,
        is_max_pool: bool,
    ) -> Result<()> {
        // ceil_mode (MaxPool, AveragePool): default 0. When 1, output dims use
        // ceiling division, changing output shape and which inputs are pooled.
        if let Some(attr) = spec.attributes.get("ceil_mode") {
            let ceil_mode = match attr {
                AttributeValue::Int(v) => *v,
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "{} layer {} has invalid ceil_mode attribute type",
                        op_name, spec.name
                    )));
                }
            };
            if ceil_mode != 0 {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{} layer {} uses ceil_mode={}, which is not supported",
                    op_name, spec.name, ceil_mode
                )));
            }
        }

        if is_max_pool {
            // dilations (MaxPool only, opset 10+): default [1,1]. Non-unit
            // dilations change which input elements participate in each window.
            if let Some(attr) = spec.attributes.get("dilations") {
                let dilations = match attr {
                    AttributeValue::Int(v) => vec![*v],
                    AttributeValue::Ints(v) if !v.is_empty() => v.clone(),
                    AttributeValue::Ints(_) => vec![1, 1],
                    _ => {
                        return Err(NyError::ModelLoad(format!(
                            "{} layer {} has invalid dilations attribute type",
                            op_name, spec.name
                        )));
                    }
                };
                if dilations.iter().any(|&d| d != 1) {
                    return Err(NyError::UnsupportedConfiguration(format!(
                        "{} layer {} uses dilations {:?}, which is not supported",
                        op_name, spec.name, dilations
                    )));
                }
            }

            // storage_order (MaxPool only): default 0 (row-major). When 1,
            // output is column-major, changing element ordering.
            if let Some(attr) = spec.attributes.get("storage_order") {
                let storage_order = match attr {
                    AttributeValue::Int(v) => *v,
                    _ => {
                        return Err(NyError::ModelLoad(format!(
                            "{} layer {} has invalid storage_order attribute type",
                            op_name, spec.name
                        )));
                    }
                };
                if storage_order != 0 {
                    return Err(NyError::UnsupportedConfiguration(format!(
                        "{} layer {} uses storage_order={}, which is not supported",
                        op_name, spec.name, storage_order
                    )));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn convert_average_pool(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX AveragePool attributes: kernel_shape, strides, pads, count_include_pad, ceil_mode
        // GlobalAveragePool has no kernel_shape - we handle it by using the input shape
        self.ensure_supported_pool_auto_pad(spec, "AveragePool")?;
        self.ensure_supported_pool_attributes(spec, "AveragePool", false)?;
        let kernel_shape = match spec.attributes.get("kernel_shape") {
            Some(_) => parse_pair_attr(spec, "AveragePool", "kernel_shape", (0, 0))?,
            None => {
                // GlobalAveragePool: will use full spatial dimensions
                // For now, use a placeholder that will need runtime shape info
                (0, 0) // Indicates global pooling
            }
        };
        // ONNX defaults every spatial stride to one. GlobalAveragePool also
        // uses this harmless placeholder value because its full-spatial kernel
        // produces one output regardless of stride.
        let strides = parse_pair_attr(spec, "AveragePool", "strides", (1, 1))?;
        let strides = validate_pool_strides(strides, "AveragePool", &spec.name)?;
        let pads = parse_symmetric_pool_padding(spec, "AveragePool")?;

        let count_include_pad = match spec.attributes.get("count_include_pad") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => false,
        };

        Ok(Layer::AveragePool(AveragePoolLayer::new(
            kernel_shape,
            strides,
            pads,
            count_include_pad,
        )))
    }
    pub(crate) fn convert_max_pool(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX MaxPool attributes: kernel_shape, strides, pads, dilations, ceil_mode, storage_order
        self.ensure_supported_pool_auto_pad(spec, "MaxPool")?;
        self.ensure_supported_pool_attributes(spec, "MaxPool", true)?;
        let kernel_shape = if spec.attributes.contains_key("kernel_shape") {
            parse_pair_attr(spec, "MaxPool", "kernel_shape", (0, 0))?
        } else {
            return Err(NyError::ModelLoad(format!(
                "MaxPool {} requires kernel_shape attribute",
                spec.name
            )));
        };
        // ONNX MaxPool defaults every spatial stride to one, not to the kernel
        // extent. Using kernel_shape here silently changed overlapping pools.
        let strides = parse_pair_attr(spec, "MaxPool", "strides", (1, 1))?;
        let strides = validate_pool_strides(strides, "MaxPool", &spec.name)?;
        let pads = parse_symmetric_pool_padding(spec, "MaxPool")?;
        let pads = validate_max_pool_padding(pads, kernel_shape, &spec.name)?;

        Ok(Layer::MaxPool2d(MaxPool2dLayer::new(
            kernel_shape,
            strides,
            pads,
        )))
    }
}

#[cfg(test)]
mod tests {
    use ny_core::{LayerType, NyError};
    use ny_propagate::Layer;
    use std::collections::{HashMap, HashSet};

    use super::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};

    fn make_context() -> (WeightStore, HashMap<String, Vec<i64>>, HashSet<String>) {
        (WeightStore::new(), HashMap::new(), HashSet::new())
    }

    fn average_pool_spec(attrs: HashMap<String, AttributeValue>) -> LayerSpec {
        LayerSpec {
            name: "avg_pool".to_string(),
            layer_type: LayerType::AveragePool,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        }
    }

    fn max_pool_spec(attrs: HashMap<String, AttributeValue>) -> LayerSpec {
        LayerSpec {
            name: "max_pool".to_string(),
            layer_type: LayerType::MaxPool,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn average_pool_non_default_auto_pad_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "auto_pad".to_string(),
            AttributeValue::String("SAME_LOWER".to_string()),
        );
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![2, 2]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = average_pool_spec(attrs);
        let err = ctx
            .convert_average_pool(&spec)
            .expect_err("non-default auto_pad must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("auto_pad")),
            "expected UnsupportedConfiguration mentioning auto_pad, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_asymmetric_padding_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![1, 0, 0, 0]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let err = ctx
            .convert_max_pool(&spec)
            .expect_err("asymmetric max-pool pads must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("asymmetric pads")),
            "expected UnsupportedConfiguration mentioning asymmetric pads, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_non_unit_dilations_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![2, 2]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let err = ctx
            .convert_max_pool(&spec)
            .expect_err("non-unit dilations must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("dilations")),
            "expected UnsupportedConfiguration mentioning dilations, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_ceil_mode_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("ceil_mode".to_string(), AttributeValue::Int(1));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let err = ctx
            .convert_max_pool(&spec)
            .expect_err("ceil_mode=1 must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("ceil_mode")),
            "expected UnsupportedConfiguration mentioning ceil_mode, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn average_pool_ceil_mode_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("ceil_mode".to_string(), AttributeValue::Int(1));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = average_pool_spec(attrs);
        let err = ctx
            .convert_average_pool(&spec)
            .expect_err("ceil_mode=1 must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("ceil_mode")),
            "expected UnsupportedConfiguration mentioning ceil_mode, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_non_zero_storage_order_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("storage_order".to_string(), AttributeValue::Int(1));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let err = ctx
            .convert_max_pool(&spec)
            .expect_err("storage_order=1 must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("storage_order")),
            "expected UnsupportedConfiguration mentioning storage_order, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_default_attributes_accepted() {
        // Default values (dilations=[1,1], ceil_mode=0, storage_order=0) must not be rejected
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![1, 1]));
        attrs.insert("ceil_mode".to_string(), AttributeValue::Int(0));
        attrs.insert("storage_order".to_string(), AttributeValue::Int(0));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        ctx.convert_max_pool(&spec)
            .expect("default attribute values must be accepted");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn standard_pool_default_stride_is_one_not_kernel_extent() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 2]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);

        match ctx
            .convert_max_pool(&max_pool_spec(attrs.clone()))
            .expect("MaxPool with omitted strides")
        {
            Layer::MaxPool2d(pool) => assert_eq!(pool.stride, (1, 1)),
            other => panic!("expected MaxPool2d, got {other:?}"),
        }
        match ctx
            .convert_average_pool(&average_pool_spec(attrs))
            .expect("AveragePool with omitted strides")
        {
            Layer::AveragePool(pool) => assert_eq!(pool.stride, (1, 1)),
            other => panic!("expected AveragePool, got {other:?}"),
        }
    }

    // Regression tests for #2872: pooling stride=0 panic cliff

    #[ntest::timeout(10000)]
    #[test]
    fn average_pool_zero_stride_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![0, 1]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = average_pool_spec(attrs);
        let err = ctx
            .convert_average_pool(&spec)
            .expect_err("stride=0 must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("zero stride")),
            "expected ModelLoad mentioning zero stride, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_zero_stride_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![1, 0]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let err = ctx
            .convert_max_pool(&spec)
            .expect_err("stride=0 must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("zero stride")),
            "expected ModelLoad mentioning zero stride, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_pads_equal_to_kernel_rejected() {
        // pads >= kernel_shape admits all-padding windows with no defined max;
        // ONNX Runtime rejects this geometry, so it must fail at model load.
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![2, 2, 2, 2]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let err = ctx
            .convert_max_pool(&spec)
            .expect_err("pads equal to kernel_shape must be rejected at load");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("kernel_shape")),
            "expected ModelLoad mentioning kernel_shape, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_pads_exceeding_kernel_in_one_dimension_rejected() {
        // The pad-vs-kernel constraint is per dimension: one oversized pad is
        // enough to admit an all-padding window.
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![0, 3, 0, 3]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let err = ctx
            .convert_max_pool(&spec)
            .expect_err("a single pad >= kernel must be rejected at load");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("kernel_shape")),
            "expected ModelLoad mentioning kernel_shape, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_pads_just_below_kernel_accepted() {
        // pads = kernel - 1 is the largest geometry every window still touches
        // at least one real input element; it must convert.
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![2, 2, 2, 2]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let layer = ctx
            .convert_max_pool(&spec)
            .expect("pads strictly smaller than kernel must be accepted");
        match layer {
            Layer::MaxPool2d(mp) => {
                assert_eq!(mp.kernel_size, (3, 3));
                assert_eq!(mp.padding, (2, 2));
            }
            other => panic!("expected MaxPool2d, got: {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn average_pool_global_no_kernel_no_stride_accepted() {
        // GlobalAveragePool with no kernel_shape and no strides: strides default
        // to (1,1) (not to kernel_shape=(0,0)) so the stride=0 guard is not triggered.
        let attrs = HashMap::new();
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = average_pool_spec(attrs);
        ctx.convert_average_pool(&spec)
            .expect("GlobalAveragePool with default strides=(1,1) must be accepted");
    }

    // Positive-path tests: verify correct value extraction (Part of #2311)

    #[ntest::timeout(10000)]
    #[test]
    fn max_pool_extracts_correct_parameters() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![1, 1, 1, 1]));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = max_pool_spec(attrs);
        let layer = ctx
            .convert_max_pool(&spec)
            .expect("valid max pool spec must convert");
        match layer {
            Layer::MaxPool2d(mp) => {
                assert_eq!(mp.kernel_size, (3, 3));
                assert_eq!(mp.stride, (2, 2));
                assert_eq!(mp.padding, (1, 1));
            }
            other => panic!("expected MaxPool2d, got: {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn average_pool_extracts_correct_parameters_with_count_include_pad() {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![1, 1]));
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![1, 0]));
        attrs.insert("count_include_pad".to_string(), AttributeValue::Int(1));
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = average_pool_spec(attrs);
        let layer = ctx
            .convert_average_pool(&spec)
            .expect("valid average pool spec must convert");
        match layer {
            Layer::AveragePool(ap) => {
                assert_eq!(ap.kernel_size, (2, 2));
                assert_eq!(ap.stride, (1, 1));
                assert_eq!(ap.padding, (1, 0));
                assert!(ap.count_include_pad, "count_include_pad should be true");
            }
            other => panic!("expected AveragePool, got: {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn global_average_pool_produces_zero_kernel_placeholder() {
        // GlobalAveragePool: no kernel_shape → (0,0) sentinel, strides default
        // to (1,1) (not to kernel_shape=(0,0)), padding defaults to (0,0).
        let attrs = HashMap::new();
        let (ws, shapes, constants) = make_context();
        let ctx = ConvertContext::new(&ws, &shapes, &constants);
        let spec = average_pool_spec(attrs);
        let layer = ctx
            .convert_average_pool(&spec)
            .expect("GlobalAveragePool with default strides=(1,1) must be accepted");
        match layer {
            Layer::AveragePool(ap) => {
                assert_eq!(ap.kernel_size, (0, 0), "global pool uses (0,0) sentinel");
                assert_eq!(ap.stride, (1, 1), "global pool defaults stride to (1,1)");
                assert_eq!(ap.padding, (0, 0));
                assert!(!ap.count_include_pad, "default count_include_pad is false");
            }
            other => panic!("expected AveragePool, got: {other:?}"),
        }
    }
}
