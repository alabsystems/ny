// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared BaBSR scoring helpers.

use ndarray::{ArrayD, ArrayView1, IxDyn};
use ny_core::checked_shape_product;

use super::*;

#[derive(Debug, Clone, Copy, Default)]
pub(in crate::beta_crown::engine) struct BabsrScoreParts {
    pub main_score: f32,
    pub backup_score: f32,
}

#[inline]
fn reduce_branch_scores(lhs: f32, rhs: f32, reduce_op: KfsbReduceOp) -> f32 {
    match reduce_op {
        KfsbReduceOp::Min => lhs.min(rhs),
        KfsbReduceOp::Max => lhs.max(rhs),
        KfsbReduceOp::Mean => f32::midpoint(lhs, rhs),
    }
}

#[inline]
fn babsr_ratio_terms(lower: f32, upper: f32) -> Option<(f32, f32)> {
    if !lower.is_finite() || !upper.is_finite() {
        return None;
    }

    let lower_temp = lower.min(0.0);
    let upper_temp = upper.max(0.0);
    let width = upper_temp - lower_temp;
    if width.abs() <= RELU_INTERCEPT_MIN_WIDTH {
        return None;
    }

    let ratio_0 = upper_temp / width;
    let ratio_1 = -lower_temp * ratio_0;
    if ratio_0.is_finite() && ratio_1.is_finite() {
        Some((ratio_0, ratio_1))
    } else {
        None
    }
}

pub(in crate::beta_crown::engine) fn compute_babsr_score_parts(
    coeff_column: ArrayView1<'_, f32>,
    lower: f32,
    upper: f32,
    bias: f32,
    reduce_op: KfsbReduceOp,
) -> BabsrScoreParts {
    let Some((ratio_0, ratio_1)) = babsr_ratio_terms(lower, upper) else {
        return BabsrScoreParts::default();
    };
    if coeff_column.is_empty() {
        return BabsrScoreParts::default();
    }

    let mut main_sum = 0.0_f64;
    let mut backup_sum = 0.0_f64;
    let mut count = 0usize;

    for coeff in coeff_column.iter().copied() {
        if !coeff.is_finite() {
            return BabsrScoreParts::default();
        }

        let intercept_candidate = coeff.min(0.0) * ratio_1;
        let bias_scale = bias * coeff;
        let bias_candidate = reduce_branch_scores(
            bias_scale * (ratio_0 - 1.0),
            bias_scale * ratio_0,
            reduce_op,
        );
        let main_candidate = (bias_candidate + intercept_candidate).abs();
        if !intercept_candidate.is_finite() || !main_candidate.is_finite() {
            return BabsrScoreParts::default();
        }

        main_sum += f64::from(main_candidate);
        backup_sum += f64::from(intercept_candidate);
        count += 1;
    }

    if count == 0 {
        return BabsrScoreParts::default();
    }

    BabsrScoreParts {
        main_score: (main_sum / count as f64) as f32,
        backup_score: (backup_sum / count as f64) as f32,
    }
}

pub(in crate::beta_crown::engine) fn compute_babsr_intercept_only_score(
    coeff_column: ArrayView1<'_, f32>,
    lower: f32,
    upper: f32,
) -> f32 {
    let Some((_, ratio_1)) = babsr_ratio_terms(lower, upper) else {
        return 0.0;
    };
    if coeff_column.is_empty() {
        return 0.0;
    }

    let mean_coeff =
        coeff_column.iter().copied().map(f64::from).sum::<f64>() / coeff_column.len() as f64;
    if !mean_coeff.is_finite() {
        return 0.0;
    }

    let score = (-(mean_coeff as f32)).max(0.0) * ratio_1;
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

fn broadcast_bias_to_shape(bias: &ArrayD<f32>, target_shape: &[usize]) -> Option<ArrayD<f32>> {
    if bias.shape() == target_shape {
        return Some(bias.clone());
    }

    let target_len = checked_shape_product(target_shape)?;

    if bias.len() == target_len {
        if let Ok(reshaped) = bias.clone().into_shape_with_order(IxDyn(target_shape)) {
            return Some(reshaped);
        }
    }

    if bias.ndim() == 1 && !target_shape.is_empty() && bias.len() == target_shape[0] {
        let mut reshape = vec![bias.len()];
        reshape.resize(target_shape.len(), 1);
        let reshaped = bias.clone().into_shape_with_order(IxDyn(&reshape)).ok()?;
        if let Some(view) = reshaped.broadcast(IxDyn(target_shape)) {
            return Some(view.to_owned());
        }
    }

    let output_shape = crate::shape::broadcast_shapes(bias.shape(), target_shape)?;
    if output_shape != target_shape {
        return None;
    }
    bias.broadcast(IxDyn(target_shape))
        .map(|view| view.to_owned())
}

fn layer_bias_tensor(layer: &Layer, target_shape: &[usize]) -> Option<ArrayD<f32>> {
    match layer {
        Layer::Linear(linear) => linear
            .bias
            .as_ref()
            .map(|bias| bias.clone().into_dyn())
            .and_then(|bias| broadcast_bias_to_shape(&bias, target_shape)),
        Layer::Conv1d(conv) => conv
            .bias
            .as_ref()
            .map(|bias| bias.clone().into_dyn())
            .and_then(|bias| broadcast_bias_to_shape(&bias, target_shape)),
        Layer::Conv2d(conv) => conv
            .bias
            .as_ref()
            .map(|bias| bias.clone().into_dyn())
            .and_then(|bias| broadcast_bias_to_shape(&bias, target_shape)),
        Layer::ConvTranspose1d(conv) => conv
            .bias
            .as_ref()
            .map(|bias| bias.clone().into_dyn())
            .and_then(|bias| broadcast_bias_to_shape(&bias, target_shape)),
        Layer::ConvTranspose2d(conv) => conv
            .bias
            .as_ref()
            .map(|bias| bias.clone().into_dyn())
            .and_then(|bias| broadcast_bias_to_shape(&bias, target_shape)),
        Layer::BatchNorm(batch_norm) => broadcast_bias_to_shape(&batch_norm.bias, target_shape),
        Layer::LayerNorm(layer_norm) => {
            broadcast_bias_to_shape(&layer_norm.beta.clone().into_dyn(), target_shape)
        }
        Layer::GroupNorm(group_norm) => {
            broadcast_bias_to_shape(&group_norm.beta.clone().into_dyn(), target_shape)
        }
        Layer::InstanceNorm1d(instance_norm) => {
            broadcast_bias_to_shape(&instance_norm.beta.clone().into_dyn(), target_shape)
        }
        Layer::AdaIN1d(adain) => adain
            .effective_instance_norm()
            .ok()
            .map(|instance_norm| instance_norm.beta.into_dyn())
            .and_then(|bias| broadcast_bias_to_shape(&bias, target_shape)),
        Layer::AddConstant(add) => broadcast_bias_to_shape(&add.constant, target_shape),
        Layer::SubConstant(sub) => {
            let signed_constant = if sub.reverse {
                sub.constant.clone()
            } else {
                sub.constant.mapv(|value| -value)
            };
            broadcast_bias_to_shape(&signed_constant, target_shape)
        }
        _ => None,
    }
}

impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine) fn sequential_preact_bias(
        &self,
        network: &Network,
        activation_layer_idx: usize,
        target_shape: &[usize],
    ) -> Option<ArrayD<f32>> {
        if activation_layer_idx == 0 {
            return None;
        }
        layer_bias_tensor(&network.layers[activation_layer_idx - 1], target_shape)
    }

    pub(in crate::beta_crown::engine) fn graph_preact_bias(
        &self,
        graph: &GraphNetwork,
        producer_name: &str,
        target_shape: &[usize],
    ) -> Option<ArrayD<f32>> {
        if producer_name == NETWORK_INPUT {
            return None;
        }

        let producer = graph.nodes.get(producer_name)?;
        if matches!(&producer.layer, Layer::Add(_)) {
            let mut recovered: Option<ArrayD<f32>> = None;
            for input_name in &producer.inputs {
                let Some(input_bias) = self.graph_preact_bias(graph, input_name, target_shape)
                else {
                    continue;
                };
                match &mut recovered {
                    Some(total) => *total = &*total + &input_bias,
                    None => recovered = Some(input_bias),
                }
            }
            return recovered;
        }

        layer_bias_tensor(&producer.layer, target_shape)
    }
}
