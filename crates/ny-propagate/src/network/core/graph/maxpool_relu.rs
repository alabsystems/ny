// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact, preset-gated VGG MaxPool decomposition.
//!
//! A 2x2, stride-2, zero-padding max pool can be written using convolutions
//! and ReLUs. This exposes the max operations as ordinary ReLUs to the bound
//! engine, matching the treatment used by alpha-beta-CROWN's VGG recipe.
//! The residual/CROWN form uses depthwise grouped convolutions: every channel
//! is independent, so dense mostly-zero `C x C` kernels would only waste
//! memory and compute. The forward-bounds form remains ungrouped because NY's
//! certified forward-linear image pass currently supports `groups == 1`.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};

use crate::layers::{AddLayer, Conv2dLayer, Layer, MaxPool2dLayer, ReLULayer};

use super::{GraphNetwork, GraphNode};

/// MaxPool decomposition selected by the effective bound-propagation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VggMaxPoolRewriteMode {
    /// Conv/ReLU/Add decomposition used with plain backward CROWN.
    Residual,
    /// Add-free Conv/ReLU chain used with forward bounds.
    ///
    /// Unlike the benchmark-specific upstream construction, this encoding is
    /// exact for arbitrary finite inputs, not only non-negative activations.
    Sequential,
}

/// Outcome of [`GraphNetwork::rewrite_vgg_maxpool2x2`].
#[derive(Debug, Default)]
pub struct VggMaxPoolRewriteReport {
    /// Eligible MaxPool nodes replaced by exact primitive subgraphs.
    pub rewritten: Vec<String>,
    /// MaxPool nodes retained unchanged, with a fail-closed reason.
    pub skipped: Vec<(String, String)>,
}

struct RewritePlan {
    node_name: String,
    helpers: Vec<GraphNode>,
    helper_shapes: Vec<(String, Vec<usize>)>,
    replacement_layer: Layer,
    replacement_inputs: Vec<String>,
}

impl GraphNetwork {
    /// Replace eligible MaxPool nodes with an exact Conv/ReLU decomposition.
    ///
    /// Only `(kernel, stride, padding) = ((2,2), (2,2), (0,0))` and a known
    /// unbatched `[C,H,W]` input shape are accepted. Every other MaxPool is
    /// retained byte-for-byte. Plans are fully validated before mutation; an
    /// unexpected apply-time invariant failure rolls the complete graph back.
    pub fn rewrite_vgg_maxpool2x2(
        &mut self,
        mode: VggMaxPoolRewriteMode,
    ) -> Result<VggMaxPoolRewriteReport> {
        let candidates: Vec<String> = self
            .node_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .is_some_and(|node| matches!(node.layer, Layer::MaxPool2d(_)))
            })
            .cloned()
            .collect();

        let mut report = VggMaxPoolRewriteReport::default();
        let mut plans = Vec::new();
        for name in candidates {
            match self.plan_vgg_maxpool_rewrite(&name, mode) {
                Ok(plan) => plans.push(plan),
                Err(reason) => report.skipped.push((name, reason)),
            }
        }

        // Validate helper-name uniqueness across every plan before touching the
        // graph. This makes ordinary ineligibility a pure no-op.
        let mut planned_names = std::collections::HashSet::new();
        let mut collision_nodes = std::collections::HashSet::new();
        for plan in &plans {
            for helper in &plan.helpers {
                if self.nodes.contains_key(helper.name())
                    || !planned_names.insert(helper.name().to_string())
                {
                    report.skipped.push((
                        plan.node_name.clone(),
                        format!("helper node name '{}' already exists", helper.name()),
                    ));
                    collision_nodes.insert(plan.node_name.clone());
                    break;
                }
            }
        }
        plans.retain(|plan| !collision_nodes.contains(&plan.node_name));

        if plans.is_empty() {
            return Ok(report);
        }

        let original = self.clone();
        for plan in plans {
            let rewritten_name = plan.node_name.clone();
            if let Err(error) = self.apply_vgg_maxpool_plan(plan) {
                *self = original;
                return Err(error);
            }
            report.rewritten.push(rewritten_name);
        }
        self.invalidate_forward_linear_cache();
        self.invalidate_exec_order_cache();
        Ok(report)
    }

    fn plan_vgg_maxpool_rewrite(
        &self,
        name: &str,
        mode: VggMaxPoolRewriteMode,
    ) -> std::result::Result<RewritePlan, String> {
        let node = self
            .nodes
            .get(name)
            .ok_or_else(|| "node disappeared while planning".to_string())?;
        let Layer::MaxPool2d(pool) = &node.layer else {
            return Err("node is not MaxPool2d".to_string());
        };
        validate_pool_geometry(pool)?;
        let input_name = node
            .require_unary_input()
            .map_err(|error| error.to_string())?
            .to_string();
        let input_shape = self
            .declared_shape(&input_name)
            .ok_or_else(|| format!("input '{input_name}' has no declared shape"))?;
        let [channels, input_h, input_w] = input_shape else {
            return Err(format!(
                "input '{input_name}' must have unbatched [C,H,W] shape, got {input_shape:?}"
            ));
        };
        if *channels == 0 {
            return Err("input has zero channels".to_string());
        }
        let (output_h, output_w) = pool
            .output_size(*input_h, *input_w)
            .map_err(|error| error.to_string())?;
        let expected_output = vec![*channels, output_h, output_w];
        if let Some(declared) = self.declared_shape(name) {
            if declared != expected_output {
                return Err(format!(
                    "declared output shape {declared:?} does not match expected {expected_output:?}"
                ));
            }
        }

        match mode {
            VggMaxPoolRewriteMode::Residual => residual_plan(
                name,
                &input_name,
                *channels,
                *input_h,
                *input_w,
                output_h,
                output_w,
            ),
            VggMaxPoolRewriteMode::Sequential => sequential_plan(
                name,
                &input_name,
                *channels,
                *input_h,
                *input_w,
                output_h,
                output_w,
            ),
        }
        .map_err(|error| error.to_string())
    }

    fn apply_vgg_maxpool_plan(&mut self, plan: RewritePlan) -> Result<()> {
        let position = self
            .node_order
            .iter()
            .position(|name| name == &plan.node_name)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "VGG MaxPool rewrite: node '{}' missing from node_order",
                    plan.node_name
                ))
            })?;

        for (offset, helper) in plan.helpers.into_iter().enumerate() {
            let helper_name = helper.name.clone();
            if self.nodes.insert(helper_name.clone(), helper).is_some() {
                return Err(NyError::InternalError(format!(
                    "VGG MaxPool rewrite: helper '{helper_name}' appeared at apply time"
                )));
            }
            self.node_order.insert(position + offset, helper_name);
        }
        for (name, shape) in plan.helper_shapes {
            self.declared_shapes.insert(name, shape);
        }

        let node = self.nodes.get_mut(&plan.node_name).ok_or_else(|| {
            NyError::InternalError(format!(
                "VGG MaxPool rewrite: node '{}' missing at apply time",
                plan.node_name
            ))
        })?;
        node.layer = plan.replacement_layer;
        node.inputs = plan.replacement_inputs;
        Ok(())
    }
}

fn validate_pool_geometry(pool: &MaxPool2dLayer) -> std::result::Result<(), String> {
    if pool.kernel_size != (2, 2) || pool.stride != (2, 2) || pool.padding != (0, 0) {
        return Err(format!(
            "requires kernel=(2,2), stride=(2,2), padding=(0,0); got kernel={:?}, stride={:?}, padding={:?}",
            pool.kernel_size, pool.stride, pool.padding
        ));
    }
    Ok(())
}

fn conv(
    kernel: ArrayD<f32>,
    stride: (usize, usize),
    groups: usize,
    input_h: usize,
    input_w: usize,
) -> Result<Layer> {
    Ok(Layer::Conv2d(Conv2dLayer::with_input_shape_full(
        kernel,
        None,
        stride,
        (0, 0),
        groups,
        input_h,
        input_w,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn residual_plan(
    name: &str,
    input_name: &str,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
) -> Result<RewritePlan> {
    let diff_name = format!("{name}/vgg_pair_diff_2x2");
    let diff_relu_name = format!("{name}/vgg_pair_diff_relu");
    let right_name = format!("{name}/vgg_pair_right_2x2");
    let pair_max_name = format!("{name}/vgg_pair_max");
    let final_diff_name = format!("{name}/vgg_final_diff");
    let final_relu_name = format!("{name}/vgg_final_diff_relu");
    let final_right_name = format!("{name}/vgg_final_right");

    let mut pair_diff = ArrayD::zeros(IxDyn(&[channels * 2, 1, 2, 2]));
    let mut pair_right = ArrayD::zeros(IxDyn(&[channels * 2, 1, 2, 2]));
    for channel in 0..channels {
        let base = channel * 2;
        pair_diff[[base, 0, 0, 0]] = 1.0;
        pair_diff[[base, 0, 0, 1]] = -1.0;
        pair_diff[[base + 1, 0, 1, 0]] = 1.0;
        pair_diff[[base + 1, 0, 1, 1]] = -1.0;
        pair_right[[base, 0, 0, 1]] = 1.0;
        pair_right[[base + 1, 0, 1, 1]] = 1.0;
    }

    let mut final_diff = ArrayD::zeros(IxDyn(&[channels, 2, 1, 1]));
    let mut final_right = ArrayD::zeros(IxDyn(&[channels, 2, 1, 1]));
    for channel in 0..channels {
        final_diff[[channel, 0, 0, 0]] = 1.0;
        final_diff[[channel, 1, 0, 0]] = -1.0;
        final_right[[channel, 1, 0, 0]] = 1.0;
    }

    let pair_shape = vec![channels * 2, output_h, output_w];
    let output_shape = vec![channels, output_h, output_w];
    let helpers = vec![
        GraphNode::new(
            &diff_name,
            conv(pair_diff, (2, 2), channels, input_h, input_w)?,
            vec![input_name.to_string()],
        ),
        GraphNode::new(
            &diff_relu_name,
            Layer::ReLU(ReLULayer::new()),
            vec![diff_name.clone()],
        ),
        GraphNode::new(
            &right_name,
            conv(pair_right, (2, 2), channels, input_h, input_w)?,
            vec![input_name.to_string()],
        ),
        GraphNode::binary(
            &pair_max_name,
            Layer::Add(AddLayer),
            &diff_relu_name,
            &right_name,
        ),
        GraphNode::new(
            &final_diff_name,
            conv(final_diff, (1, 1), channels, output_h, output_w)?,
            vec![pair_max_name.clone()],
        ),
        GraphNode::new(
            &final_relu_name,
            Layer::ReLU(ReLULayer::new()),
            vec![final_diff_name.clone()],
        ),
        GraphNode::new(
            &final_right_name,
            conv(final_right, (1, 1), channels, output_h, output_w)?,
            vec![pair_max_name.clone()],
        ),
    ];
    let helper_shapes = vec![
        (diff_name, pair_shape.clone()),
        (diff_relu_name, pair_shape.clone()),
        (right_name, pair_shape.clone()),
        (pair_max_name, pair_shape),
        (final_diff_name, output_shape.clone()),
        (final_relu_name.clone(), output_shape.clone()),
        (final_right_name.clone(), output_shape),
    ];
    Ok(RewritePlan {
        node_name: name.to_string(),
        helpers,
        helper_shapes,
        replacement_layer: Layer::Add(AddLayer),
        replacement_inputs: vec![final_relu_name, final_right_name],
    })
}

#[allow(clippy::too_many_arguments)]
fn sequential_plan(
    name: &str,
    input_name: &str,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
) -> Result<RewritePlan> {
    let encode_name = format!("{name}/vgg_signed_pair_encode");
    let encode_relu_name = format!("{name}/vgg_signed_pair_relu");
    let pair_name = format!("{name}/vgg_signed_max_encode");
    let pair_relu_name = format!("{name}/vgg_signed_max_relu");

    // Per channel: [a-b, b, -b, c-d, d, -d].
    let mut encode = ArrayD::zeros(IxDyn(&[channels * 6, channels, 2, 2]));
    for channel in 0..channels {
        let base = channel * 6;
        encode[[base, channel, 0, 0]] = 1.0;
        encode[[base, channel, 0, 1]] = -1.0;
        encode[[base + 1, channel, 0, 1]] = 1.0;
        encode[[base + 2, channel, 0, 1]] = -1.0;
        encode[[base + 3, channel, 1, 0]] = 1.0;
        encode[[base + 3, channel, 1, 1]] = -1.0;
        encode[[base + 4, channel, 1, 1]] = 1.0;
        encode[[base + 5, channel, 1, 1]] = -1.0;
    }

    // Let m1=max(a,b), m2=max(c,d). Encode [m1-m2, m2, -m2].
    let mut pair = ArrayD::zeros(IxDyn(&[channels * 3, channels * 6, 1, 1]));
    let first = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0];
    let second = [0.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    for channel in 0..channels {
        let base = channel * 3;
        for index in 0..6 {
            let input = channel * 6 + index;
            pair[[base, input, 0, 0]] = first[index];
            pair[[base + 1, input, 0, 0]] = second[index];
            pair[[base + 2, input, 0, 0]] = -second[index];
        }
    }

    // ReLU(m1-m2) + ReLU(m2) - ReLU(-m2) = max(m1,m2).
    let mut finish = ArrayD::zeros(IxDyn(&[channels, channels * 3, 1, 1]));
    for channel in 0..channels {
        let input = channel * 3;
        finish[[channel, input, 0, 0]] = 1.0;
        finish[[channel, input + 1, 0, 0]] = 1.0;
        finish[[channel, input + 2, 0, 0]] = -1.0;
    }

    let encoded_shape = vec![channels * 6, output_h, output_w];
    let pair_shape = vec![channels * 3, output_h, output_w];
    let helpers = vec![
        GraphNode::new(
            &encode_name,
            conv(encode, (2, 2), 1, input_h, input_w)?,
            vec![input_name.to_string()],
        ),
        GraphNode::new(
            &encode_relu_name,
            Layer::ReLU(ReLULayer::new()),
            vec![encode_name.clone()],
        ),
        GraphNode::new(
            &pair_name,
            conv(pair, (1, 1), 1, output_h, output_w)?,
            vec![encode_relu_name.clone()],
        ),
        GraphNode::new(
            &pair_relu_name,
            Layer::ReLU(ReLULayer::new()),
            vec![pair_name.clone()],
        ),
    ];
    let helper_shapes = vec![
        (encode_name, encoded_shape.clone()),
        (encode_relu_name, encoded_shape),
        (pair_name, pair_shape.clone()),
        (pair_relu_name.clone(), pair_shape),
    ];
    Ok(RewritePlan {
        node_name: name.to_string(),
        helpers,
        helper_shapes,
        replacement_layer: conv(finish, (1, 1), 1, output_h, output_w)?,
        replacement_inputs: vec![pair_relu_name],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_tensor::BoundedTensor;

    fn pool_graph(shape: [usize; 3], pool: MaxPool2dLayer) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pool", Layer::MaxPool2d(pool)));
        graph.set_declared_shape(super::super::NETWORK_INPUT, shape.to_vec());
        if let Ok((height, width)) =
            MaxPool2dLayer::new((2, 2), (2, 2), (0, 0)).output_size(shape[1], shape[2])
        {
            graph.set_declared_shape("pool", vec![shape[0], height, width]);
        }
        graph.set_output("pool");
        graph
    }

    fn point(values: Vec<f32>, shape: &[usize]) -> BoundedTensor {
        BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(shape), values).unwrap()).unwrap()
    }

    fn assert_point_equivalent(mode: VggMaxPoolRewriteMode) {
        let original = pool_graph([2, 4, 4], MaxPool2dLayer::new((2, 2), (2, 2), (0, 0)));
        // Includes negative maxima and exact ties (ReLU pre-activation zero).
        let values = vec![
            -3.0, -2.0, 4.0, 4.0, -5.0, -2.0, 1.0, 0.0, 7.0, 7.0, -8.0, -9.0, 6.0, 2.0, -8.0, -8.0,
            0.0, 0.0, 3.0, -1.0, 0.0, -4.0, 3.0, 3.0, -6.0, 2.0, 5.0, 1.0, -6.0, 2.0, 5.0, 5.0,
        ];
        let input = point(values, &[2, 4, 4]);
        let expected = original.propagate_ibp(&input).unwrap();

        let mut rewritten = original;
        let report = rewritten.rewrite_vgg_maxpool2x2(mode).unwrap();
        assert_eq!(report.rewritten, vec!["pool"]);
        assert!(report.skipped.is_empty());
        let actual = rewritten.propagate_ibp(&input).unwrap();
        assert_eq!(actual.lower(), expected.lower());
        assert_eq!(actual.upper(), expected.upper());
    }

    #[test]
    fn residual_forward_matches_maxpool_with_negative_and_tied_values() {
        assert_point_equivalent(VggMaxPoolRewriteMode::Residual);
    }

    #[test]
    fn sequential_forward_matches_maxpool_with_negative_and_tied_values() {
        assert_point_equivalent(VggMaxPoolRewriteMode::Sequential);
    }

    #[test]
    fn rewritten_ibp_soundly_contains_direct_maxpool_bounds() {
        let original = pool_graph([2, 4, 4], MaxPool2dLayer::new((2, 2), (2, 2), (0, 0)));
        let lower: Vec<f32> = (0..32).map(|i| -4.0 + i as f32 * 0.125).collect();
        let upper: Vec<f32> = lower
            .iter()
            .enumerate()
            .map(|(i, value)| value + 0.25 + (i % 5) as f32 * 0.125)
            .collect();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 4, 4]), lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 4, 4]), upper).unwrap(),
        )
        .unwrap();
        let direct = original.propagate_ibp(&input).unwrap();

        for mode in [
            VggMaxPoolRewriteMode::Residual,
            VggMaxPoolRewriteMode::Sequential,
        ] {
            let mut rewritten = original.clone();
            rewritten.rewrite_vgg_maxpool2x2(mode).unwrap();
            let bounds = rewritten.propagate_ibp(&input).unwrap();
            for ((&lower, &upper), (&direct_lower, &direct_upper)) in bounds
                .lower()
                .iter()
                .zip(bounds.upper())
                .zip(direct.lower().iter().zip(direct.upper()))
            {
                assert!(
                    lower <= direct_lower,
                    "{mode:?} lower {lower} excluded direct lower {direct_lower}"
                );
                assert!(
                    upper >= direct_upper,
                    "{mode:?} upper {upper} excluded direct upper {direct_upper}"
                );
            }
        }
    }

    #[test]
    fn residual_crown_bounds_enclose_exact_maxpool_samples() {
        let original = pool_graph([1, 4, 4], MaxPool2dLayer::new((2, 2), (2, 2), (0, 0)));
        let lower = vec![
            -3.0, -2.0, 0.0, 1.0, -4.0, -1.0, -2.0, 0.0, 1.0, 2.0, -5.0, -3.0, 0.0, 2.0, -4.0, -1.0,
        ];
        let upper = vec![
            -1.0, 1.0, 3.0, 3.0, -2.0, 2.0, 0.0, 4.0, 4.0, 5.0, -1.0, 1.0, 2.0, 5.0, 0.0, 2.0,
        ];
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), upper.clone()).unwrap(),
        )
        .unwrap();

        let mut rewritten = original.clone();
        rewritten
            .rewrite_vgg_maxpool2x2(VggMaxPoolRewriteMode::Residual)
            .unwrap();
        let crown = rewritten.propagate_crown_batched(&input).unwrap();

        let center: Vec<f32> = lower
            .iter()
            .zip(&upper)
            .map(|(lower, upper)| (lower + upper) * 0.5)
            .collect();
        for sample in [lower, center, upper] {
            let exact = original.propagate_ibp(&point(sample, &[1, 4, 4])).unwrap();
            for ((&value, &bound_lower), &bound_upper) in
                exact.lower().iter().zip(crown.lower()).zip(crown.upper())
            {
                assert!(
                    bound_lower <= value && value <= bound_upper,
                    "exact value {value} escaped CROWN interval [{bound_lower}, {bound_upper}]"
                );
            }
        }
    }

    #[test]
    fn ineligible_pool_is_retained_unchanged() {
        let mut graph = pool_graph([1, 4, 4], MaxPool2dLayer::new((3, 3), (1, 1), (0, 0)));
        let before_order = graph.node_names().to_vec();
        let report = graph
            .rewrite_vgg_maxpool2x2(VggMaxPoolRewriteMode::Residual)
            .unwrap();
        assert!(report.rewritten.is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(graph.node_names(), before_order.as_slice());
        assert!(matches!(
            graph.node("pool").unwrap().layer(),
            Layer::MaxPool2d(_)
        ));
    }

    #[test]
    fn residual_rewrite_uses_depthwise_groups_instead_of_dense_zero_kernels() {
        let mut graph = pool_graph([3, 4, 4], MaxPool2dLayer::new((2, 2), (2, 2), (0, 0)));
        graph
            .rewrite_vgg_maxpool2x2(VggMaxPoolRewriteMode::Residual)
            .unwrap();
        let node = graph.node("pool/vgg_pair_diff_2x2").unwrap();
        let Layer::Conv2d(conv) = node.layer() else {
            panic!("expected Conv2d helper");
        };
        assert_eq!(conv.groups, 3);
        assert_eq!(conv.kernel.shape(), &[6, 1, 2, 2]);
    }
}
