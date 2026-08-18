// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto::{self, attribute_type};
use crate::WeightStore;
use ndarray::Dimension;

use super::super::common::read_tensor_i64s;
use super::super::FoldedTensor;

/// Fold a standard ONNX Trilu whose data and optional diagonal are exact
/// constants.  This lets finite additive attention masks stay as ordinary
/// Add -> Softmax graphs instead of relying on the unauthenticated hard-causal
/// semantic fusion.
pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    if node.op_type != "Trilu"
        || !matches!(node.input.len(), 1 | 2)
        || node.input[0].is_empty()
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return None;
    }

    let mut upper = None;
    for attribute in &node.attribute {
        if attribute.name != "upper"
            || upper.is_some()
            || attribute.r#type != attribute_type::INT
            || !matches!(attribute.i_value(), 0 | 1)
        {
            return None;
        }
        upper = Some(attribute.i_value() == 1);
    }
    let upper = upper.unwrap_or(true);

    let diagonal = match node.input.get(1).filter(|name| !name.is_empty()) {
        Some(name) => {
            // Trilu's diagonal operand is INT64.  A numerically integral FLOAT
            // constant is not dtype-equivalent and must not gain discrete
            // authority through the compatibility view.
            weights.get_integers(name)?;
            let values = read_tensor_i64s(weights, name)?;
            (values.len() == 1).then_some(values[0])?
        }
        None => 0,
    };

    let input_name = node.input.first()?;
    let input = weights.get(input_name)?;
    if input.ndim() < 2 {
        return None;
    }
    let mut float_data = input.clone();
    apply_mask_f32(&mut float_data, diagonal, upper)?;

    let integer_data = match weights.get_integers(input_name) {
        Some(integers) => {
            if integers.shape() != input.shape() {
                return None;
            }
            let mut integers = integers.clone();
            apply_mask_i64(&mut integers, diagonal, upper)?;
            Some(integers)
        }
        None => None,
    };
    let integer_range = integer_data
        .as_ref()
        .and_then(|_| weights.get_integer_range(input_name));

    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range,
    })
}

fn keep_entry(index: &[usize], diagonal: i64, upper: bool) -> Option<bool> {
    let row = i64::try_from(index[index.len().checked_sub(2)?]).ok()?;
    let column = i64::try_from(index[index.len().checked_sub(1)?]).ok()?;
    let relative = column.checked_sub(row)?;
    Some(if upper {
        relative >= diagonal
    } else {
        relative <= diagonal
    })
}

fn apply_mask_f32(tensor: &mut ndarray::ArrayD<f32>, diagonal: i64, upper: bool) -> Option<()> {
    for (index, value) in tensor.indexed_iter_mut() {
        if !keep_entry(index.slice(), diagonal, upper)? {
            *value = 0.0;
        }
    }
    Some(())
}

fn apply_mask_i64(tensor: &mut ndarray::ArrayD<i64>, diagonal: i64, upper: bool) -> Option<()> {
    for (index, value) in tensor.indexed_iter_mut() {
        if !keep_entry(index.slice(), diagonal, upper)? {
            *value = 0;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, ArrayD, IxDyn};

    fn node(inputs: &[&str], upper: Option<i64>) -> onnx_proto::NodeProto {
        let mut node = onnx_proto::NodeProto {
            op_type: "Trilu".to_string(),
            input: inputs.iter().map(|input| (*input).to_string()).collect(),
            output: vec!["out".to_string()],
            ..Default::default()
        };
        if let Some(upper) = upper {
            node.attribute.push(onnx_proto::AttributeProto {
                name: "upper".to_string(),
                i: Some(upper),
                r#type: attribute_type::INT,
                ..Default::default()
            });
        }
        node
    }

    fn matrix_weights() -> WeightStore {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        weights
    }

    #[test]
    fn folds_upper_and_lower_diagonals_exactly() {
        let mut weights = matrix_weights();
        weights.insert("k".to_string(), arr0(1.0).into_dyn());
        weights.insert_integers("k".to_string(), arr0(1_i64).into_dyn());
        let upper = try_fold(&node(&["data", "k"], Some(1)), &weights).unwrap();
        assert_eq!(
            upper.float_data.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0]
        );

        let lower = try_fold(&node(&["data"], Some(0)), &weights).unwrap();
        assert_eq!(
            lower.float_data.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0]
        );

        let omitted = try_fold(&node(&["data", ""], None), &weights).unwrap();
        assert_eq!(
            omitted.float_data.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0],
            "an empty optional diagonal input means the default k=0"
        );
    }

    #[test]
    fn declines_inexact_or_malformed_discrete_semantics() {
        let mut weights = matrix_weights();
        weights.insert("k".to_string(), arr0(0.5).into_dyn());
        assert!(try_fold(&node(&["data", "k"], None), &weights).is_none());
        assert!(try_fold(&node(&["data"], Some(2)), &weights).is_none());
    }
}
