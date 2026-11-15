// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Same-query immutable head source and already authenticated ReLU descriptors.
//! This resolver produces no numerical bound or advisory score.
use super::*;
use crate::network::NETWORK_INPUT;

fn scalar(v: f32) -> Option<f64> {
    ny_core::is_crown_coeff_safe(v).then_some(f64::from(v))
}
fn poll(guard: &mut Poll<'_>, i: usize) -> Option<()> {
    (!i.is_multiple_of(32) || guard.check()).then_some(())
}
fn vector(b: &BoundedTensor, n: usize) -> bool {
    b.len() == n && (matches!(b.shape(), [d] if *d == n) || matches!(b.shape(), [1,d] if *d == n))
}

#[derive(Clone, Copy)]
pub(super) struct Bn<'a> {
    pub(super) scale: &'a [f32],
    pub(super) shift: &'a [f32],
    pub(super) scale_error: &'a [f32],
    pub(super) bias_error: &'a [f32],
    pub(super) lower: &'a [f32],
    pub(super) upper: &'a [f32],
}
pub(super) struct Source<'a> {
    pub(super) matrix: &'a [f32],
    pub(super) bias: Option<&'a [f32]>,
    pub(super) lower: &'a [f32],
    pub(super) upper: &'a [f32],
    pub(super) final_matrix: &'a [f32],
    pub(super) final_bias: Option<&'a [f32]>,
    pub(super) objective: &'a [f32],
    pub(super) bn: Option<Bn<'a>>,
}
pub(super) fn resolve<'a>(
    b: &RawRootCutQuote<'a>,
    dimensions: [usize; 3],
    guard: &mut Poll<'_>,
) -> Option<Source<'a>> {
    if !guard.check() {
        return None;
    }
    let [e, d, o] = dimensions;
    let relu = b.graph.node(b.target)?;
    if !matches!(relu.layer(), Layer::ReLU(_)) || relu.require_unary_input().ok()? != b.pre_node {
        return None;
    }
    let pre = b.graph.node(b.pre_node)?;
    let (linear, parent, bn) = match pre.layer() {
        Layer::Linear(l) => (l, pre.require_unary_input().ok()?, None),
        Layer::BatchNorm(v) => {
            let name = pre.require_unary_input().ok()?;
            let node = b.graph.node(name)?;
            let Layer::Linear(l) = node.layer() else {
                return None;
            };
            let bounds = b.bounds.get(name)?;
            if v.num_channels != d
                || !vector(bounds, d)
                || [
                    v.scale.shape(),
                    v.bias.shape(),
                    v.scale_err.shape(),
                    v.bias_err.shape(),
                ]
                .iter()
                .any(|s| *s != [d])
            {
                return None;
            }
            // Parent extraction has already authenticated this raw layout.
            // No hidden layout/helper scan is repeated by this advice layer.
            let bn = Bn {
                scale: v.scale.as_slice()?,
                shift: v.bias.as_slice()?,
                scale_error: v.scale_err.as_slice()?,
                bias_error: v.bias_err.as_slice()?,
                lower: bounds.lower().as_slice()?,
                upper: bounds.upper().as_slice()?,
            };
            (l, node.require_unary_input().ok()?, Some(bn))
        }
        _ => return None,
    };
    let parent_box = if parent == NETWORK_INPUT {
        b.input
    } else {
        b.bounds.get(parent)?
    };
    if !vector(parent_box, e)
        || !vector(b.bounds.get(b.pre_node)?, d)
        || linear.in_features() != e
        || linear.out_features() != d
        || linear.weight.shape() != [d, e]
        || linear.bias.as_ref().is_some_and(|v| v.shape() != [d])
    {
        return None;
    }
    let last = b.graph.node(b.graph.output_name())?;
    let Layer::Linear(final_layer) = last.layer() else {
        return None;
    };
    if final_layer.in_features() != d
        || final_layer.out_features() != o
        || final_layer.weight.shape() != [o, d]
        || final_layer.bias.as_ref().is_some_and(|v| v.shape() != [o])
        || b.objective.len() != o
    {
        return None;
    }
    let mut name = last.require_unary_input().ok()?;
    for _ in 0..b.extraction.max_nodes {
        if !guard.check() || name.len() > 1024 {
            return None;
        }
        if name == b.target {
            break;
        }
        let node = b.graph.node(name)?;
        if !matches!(node.layer(), Layer::Flatten(_) | Layer::Reshape(_))
            || b.bounds.get(name)?.len() != d
        {
            return None;
        }
        name = node.require_unary_input().ok()?;
    }
    if name != b.target {
        return None;
    }
    Some(Source {
        matrix: linear.weight.as_slice()?,
        bias: match &linear.bias {
            Some(v) => Some(v.as_slice()?),
            None => None,
        },
        lower: parent_box.lower().as_slice()?,
        upper: parent_box.upper().as_slice()?,
        final_matrix: final_layer.weight.as_slice()?,
        final_bias: match &final_layer.bias {
            Some(v) => Some(v.as_slice()?),
            None => None,
        },
        objective: b.objective,
        bn,
    })
}

#[derive(Clone, Copy)]
pub(super) struct Slopes<'a> {
    pub(super) positive: &'a [f32],
    pub(super) negative: &'a [f32],
    pub(super) positive_bias: Option<&'a [f32]>,
    pub(super) negative_bias: &'a [f32],
}
impl<'a> Slopes<'a> {
    pub(super) fn from_layer(layer: &'a GpuCrownLayer, d: usize) -> Option<Self> {
        let out = match layer {
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } if *num_neurons == d => Self {
                positive: lower_slope,
                negative: upper_slope,
                positive_bias: Some(lower_intercept),
                negative_bias: upper_intercept,
            },
            GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
                num_neurons,
            } if *num_neurons == d && upper_neg_slope.len() == d => Self {
                positive: lower_pos_slope,
                negative: cross_slope,
                positive_bias: None,
                negative_bias: cross_intercept,
            },
            _ => return None,
        };
        (out.positive.len() == d
            && out.negative.len() == d
            && out.negative_bias.len() == d
            && out.positive_bias.is_none_or(|v| v.len() == d))
        .then_some(out)
    }
    pub(super) fn branch(self, j: usize, c: f64) -> Option<(f64, f64)> {
        if c < 0. {
            Some((scalar(self.negative[j])?, scalar(self.negative_bias[j])?))
        } else {
            Some((
                scalar(self.positive[j])?,
                self.positive_bias.map_or(Some(0.), |v| scalar(v[j]))?,
            ))
        }
    }
}
pub(super) fn target_slopes<'a>(
    context: &'a TwoLocalContext<'_, '_>,
    d: usize,
    guard: &mut Poll<'_>,
) -> Option<Slopes<'a>> {
    // The certified original unary extraction is a single backward-order chain.
    let [GpuResnetSegment::Chain(layers)] = context.base.segments else {
        return None;
    };
    let mut ordinal = 0;
    for (index, layer) in layers.iter().enumerate() {
        poll(guard, index)?;
        if matches!(
            layer,
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. }
        ) {
            if ordinal == context.target_activation {
                return Slopes::from_layer(layer, d);
            }
            ordinal += 1;
        }
    }
    None
}
