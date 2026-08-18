// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verdict-neutral structural recognizer for exact one-free-axis graph algebra.
//!
//! This module does not evaluate a network and is not called by any verifier
//! path.  It is a source-only prerequisite for a future complete 1-D checker:
//! given one structurally free scalar input, it follows that dependency through
//! a deliberately small graph fragment and rejects the first operation that
//! could create genuine nonlinear interaction.
//!
//! The recognizer is sufficient, not necessary.  In particular, it treats a
//! Linear output as depending on every structurally dynamic input even when a
//! weight happens to be zero.  A positive result is therefore safe against
//! missed dependencies; a negative result can be a conservative refusal.
//!
//! `Div(dynamic, constant)` is reported as piecewise-affine only modulo an
//! explicit nonzero-divisor obligation.  The caller must discharge every
//! [`OneAxisAlgebraReport::constant_divisor_nodes`] entry before treating the
//! structural class as exact.  No diagnostic phase-event proposal is consumed
//! here: local point-JVP roots are incomplete and cannot certify a partition.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::layers::Layer;

use super::{GraphNetwork, NETWORK_INPUT};

/// Hard graph-size cap for the source-only one-axis recognizer.
///
/// The official NN4SYS 2048-dual post-loader graph has 97 nodes.
pub const ONE_AXIS_MAX_NODES: usize = 128;
/// Hard edge cap checked before topological sorting.
pub const ONE_AXIS_MAX_EDGES: usize = 512;
/// Hard rank cap for every declared abstract tensor.
pub const ONE_AXIS_MAX_RANK: usize = 16;
/// Hard cap for any one abstract tensor dependency mask.
pub const ONE_AXIS_MAX_TENSOR_ELEMENTS: usize = 1 << 20;
/// Hard cap for all dependency-mask elements retained during one walk.
pub const ONE_AXIS_MAX_TOTAL_ELEMENTS: usize = 16 << 20;

/// Axis-local algebra recognized at the graph output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneAxisAlgebraClass {
    /// The output is structurally independent of the selected input axis.
    Constant,
    /// The output is piecewise-affine along the selected axis.
    PiecewiseAffine,
    /// A scalar monotone sigmoid wraps a piecewise-affine core, possibly with
    /// an axis-constant addend and sign from a final `Sub`.
    ///
    /// A future verifier may peel this wrapper only after soundly reducing the
    /// concrete output constraint (including any axis-constant sigmoid branch).
    PeelableMonotoneSigmoid,
}

/// Fail-closed reason why the structural walk did not recognize an exact
/// one-axis algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneAxisDeclineReason {
    Deadline,
    EmptyGraph,
    NodeLimit,
    EdgeLimit,
    InvalidInputShape,
    FreeAxisOutOfRange,
    ExecutionOrder,
    MissingNode,
    ForwardOrMissingInput,
    MissingDeclaredShape,
    ShapeMismatch,
    TensorElementLimit,
    TotalElementLimit,
    UnsupportedLayer,
    DynamicMulOperands,
    DynamicDivisor,
    NonAffineComposition,
    NonScalarSigmoidOutput,
}

/// Location and kind of a fail-closed refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisDecline {
    pub reason: OneAxisDeclineReason,
    pub node: Option<String>,
    pub layer: Option<&'static str>,
}

/// Bounded, verdict-neutral result of a one-free-axis structural walk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisAlgebraReport {
    pub free_axis: usize,
    pub class: Option<OneAxisAlgebraClass>,
    pub decline: Option<OneAxisDecline>,
    pub nodes_examined: usize,
    /// ReLUs whose input is structurally axis-dependent.
    pub dynamic_relu_nodes: usize,
    /// `MulBinary` nodes with exactly one axis-dependent operand.
    pub constant_sided_mul_nodes: usize,
    /// `Div` nodes whose numerator is axis-dependent and denominator is
    /// axis-constant.  Each name is an outstanding proof obligation that the
    /// fixed denominator is nonzero.
    pub constant_divisor_nodes: Vec<String>,
    /// Sigmoids whose input is structurally axis-dependent.
    pub dynamic_sigmoid_nodes: usize,
    /// Sigmoids whose input is structurally axis-constant.  Their concrete
    /// values still require a sound enclosure when they shift a peeled output
    /// threshold (the NN4SYS dual case).
    pub static_sigmoid_nodes: usize,
}

impl OneAxisAlgebraReport {
    fn new(free_axis: usize) -> Self {
        Self {
            free_axis,
            class: None,
            decline: None,
            nodes_examined: 0,
            dynamic_relu_nodes: 0,
            constant_sided_mul_nodes: 0,
            constant_divisor_nodes: Vec::new(),
            dynamic_sigmoid_nodes: 0,
            static_sigmoid_nodes: 0,
        }
    }

    fn decline(
        &mut self,
        reason: OneAxisDeclineReason,
        node: Option<&str>,
        layer: Option<&'static str>,
    ) {
        self.class = None;
        self.decline = Some(OneAxisDecline {
            reason,
            node: node.map(str::to_owned),
            layer,
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisExpression {
    Constant,
    PiecewiseAffine,
    SigmoidPiecewiseAffine,
}

#[derive(Clone, Debug)]
struct AxisState {
    shape: Vec<usize>,
    dependency: Arc<[bool]>,
    expression: AxisExpression,
}

impl AxisState {
    fn depends_on_axis(&self) -> bool {
        self.dependency.iter().any(|&depends| depends)
    }
}

fn checked_elements(shape: &[usize]) -> Option<usize> {
    if shape.len() > ONE_AXIS_MAX_RANK {
        return None;
    }
    if shape.is_empty() {
        return Some(1);
    }
    shape.iter().try_fold(1usize, |product, &dimension| {
        if dimension == 0 {
            None
        } else {
            product.checked_mul(dimension)
        }
    })
}

fn all_dependency(shape: &[usize], depends: bool) -> Option<Arc<[bool]>> {
    let elements = checked_elements(shape)?;
    if elements > ONE_AXIS_MAX_TENSOR_ELEMENTS {
        return None;
    }
    Some(vec![depends; elements].into())
}

fn deadline_expired(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

fn slice_dependency(
    input: &AxisState,
    layer: &crate::layers::SliceLayer,
    output_shape: &[usize],
    deadline: Instant,
) -> Option<Arc<[bool]>> {
    let (axis, start, end) = layer.resolved_range(&input.shape).ok()?;
    let mut expected_shape = input.shape.clone();
    expected_shape[axis] = end.checked_sub(start)?;
    if expected_shape != output_shape {
        return None;
    }
    let output_elements = checked_elements(output_shape)?;
    if output_elements > ONE_AXIS_MAX_TENSOR_ELEMENTS {
        return None;
    }

    let dimensions = input.shape.len();
    let mut input_strides = vec![1usize; dimensions];
    let mut output_strides = vec![1usize; dimensions];
    for index in (0..dimensions.saturating_sub(1)).rev() {
        input_strides[index] = input_strides[index + 1].checked_mul(input.shape[index + 1])?;
        output_strides[index] = output_strides[index + 1].checked_mul(output_shape[index + 1])?;
    }

    let mut dependency = Vec::with_capacity(output_elements);
    for output_flat in 0..output_elements {
        if output_flat % 4096 == 0 && deadline_expired(deadline) {
            return None;
        }
        let mut remaining = output_flat;
        let mut input_flat = 0usize;
        for dimension in 0..dimensions {
            let coordinate = remaining / output_strides[dimension];
            remaining %= output_strides[dimension];
            let input_coordinate = if dimension == axis {
                coordinate.checked_add(start)?
            } else {
                coordinate
            };
            input_flat =
                input_flat.checked_add(input_coordinate.checked_mul(input_strides[dimension])?)?;
        }
        dependency.push(*input.dependency.get(input_flat)?);
    }
    Some(dependency.into())
}

fn input_state<'a>(
    states: &'a HashMap<String, AxisState>,
    network_input: &'a AxisState,
    name: &str,
) -> Option<&'a AxisState> {
    if name == NETWORK_INPUT {
        Some(network_input)
    } else {
        states.get(name)
    }
}

fn reduce_sum_shape(input_shape: &[usize], axes: &[i64], keepdims: bool) -> Option<Vec<usize>> {
    let rank = input_shape.len();
    if axes.len() > rank {
        return None;
    }
    let raw_axes: Vec<i64> = if axes.is_empty() {
        (0..rank).map(|axis| axis as i64).collect()
    } else {
        axes.to_vec()
    };
    let mut resolved = Vec::with_capacity(raw_axes.len());
    for axis in raw_axes {
        let rank_i64 = i64::try_from(rank).ok()?;
        let axis = if axis < 0 {
            axis.checked_add(rank_i64)?
        } else {
            axis
        };
        let axis = usize::try_from(axis).ok()?;
        if axis >= rank || resolved.contains(&axis) {
            return None;
        }
        resolved.push(axis);
    }
    if keepdims {
        let mut output = input_shape.to_vec();
        for axis in resolved {
            output[axis] = 1;
        }
        Some(output)
    } else {
        Some(
            input_shape
                .iter()
                .enumerate()
                .filter_map(|(axis, &dimension)| (!resolved.contains(&axis)).then_some(dimension))
                .collect(),
        )
    }
}

fn concat_shape(inputs: &[&AxisState], layer: &crate::layers::ConcatLayer) -> Option<Vec<usize>> {
    let first = inputs.first()?;
    let axis = layer.normalize_axis(first.shape.len()).ok()?;
    let mut output = first.shape.clone();
    for input in &inputs[1..] {
        if input.shape.len() != output.len() {
            return None;
        }
        for (dimension, (&left, &right)) in output.iter().zip(&input.shape).enumerate() {
            if dimension != axis && left != right {
                return None;
            }
        }
        output[axis] = output[axis].checked_add(input.shape[axis])?;
    }
    Some(output)
}

impl GraphNetwork {
    /// Recognize a bounded exact-one-free-axis algebra without evaluating the
    /// network or mutating verifier state.
    ///
    /// `input_shape` is the graph's internal, batch-stripped input shape and
    /// `free_axis` is its row-major flattened coordinate.  The caller is
    /// responsible for proving that exactly this coordinate has unequal exact
    /// property endpoints.  The walk is hard-capped and must finish before
    /// `deadline`; every cap, malformed graph, unsupported operation, or
    /// nonlinear interaction returns a report with `class == None`.
    ///
    /// This method has no verdict authority.  Even a positive structural class
    /// leaves the constant-divisor and sigmoid-threshold obligations documented
    /// on [`OneAxisAlgebraReport`].
    pub fn recognize_one_free_axis_algebra_until(
        &self,
        input_shape: &[usize],
        free_axis: usize,
        deadline: Instant,
    ) -> OneAxisAlgebraReport {
        let mut report = OneAxisAlgebraReport::new(free_axis);
        if deadline_expired(deadline) {
            report.decline(OneAxisDeclineReason::Deadline, None, None);
            return report;
        }
        if self.num_nodes() == 0 {
            report.decline(OneAxisDeclineReason::EmptyGraph, None, None);
            return report;
        }
        if self.num_nodes() > ONE_AXIS_MAX_NODES || self.node_names().len() != self.num_nodes() {
            report.decline(OneAxisDeclineReason::NodeLimit, None, None);
            return report;
        }

        let mut edges = 0usize;
        for name in self.node_names() {
            if deadline_expired(deadline) {
                report.decline(OneAxisDeclineReason::Deadline, None, None);
                return report;
            }
            let Some(node) = self.node(name) else {
                report.decline(OneAxisDeclineReason::MissingNode, Some(name), None);
                return report;
            };
            let Some(next_edges) = edges.checked_add(node.inputs().len()) else {
                report.decline(OneAxisDeclineReason::EdgeLimit, Some(name), None);
                return report;
            };
            edges = next_edges;
            if edges > ONE_AXIS_MAX_EDGES {
                report.decline(OneAxisDeclineReason::EdgeLimit, Some(name), None);
                return report;
            }
        }

        let Some(input_elements) = checked_elements(input_shape) else {
            report.decline(OneAxisDeclineReason::InvalidInputShape, None, None);
            return report;
        };
        if input_elements > ONE_AXIS_MAX_TENSOR_ELEMENTS {
            report.decline(OneAxisDeclineReason::TensorElementLimit, None, None);
            return report;
        }
        if free_axis >= input_elements {
            report.decline(OneAxisDeclineReason::FreeAxisOutOfRange, None, None);
            return report;
        }
        if let Some(declared) = self.declared_shape(NETWORK_INPUT) {
            if declared != input_shape {
                report.decline(
                    OneAxisDeclineReason::ShapeMismatch,
                    Some(NETWORK_INPUT),
                    None,
                );
                return report;
            }
        }

        let mut input_dependency = vec![false; input_elements];
        input_dependency[free_axis] = true;
        let network_input = AxisState {
            shape: input_shape.to_vec(),
            dependency: input_dependency.into(),
            expression: AxisExpression::PiecewiseAffine,
        };
        let order = match self.exec_order() {
            Ok(order) => order,
            Err(_) => {
                report.decline(OneAxisDeclineReason::ExecutionOrder, None, None);
                return report;
            }
        };
        if deadline_expired(deadline) {
            report.decline(OneAxisDeclineReason::Deadline, None, None);
            return report;
        }
        if order.len() != self.num_nodes() {
            report.decline(OneAxisDeclineReason::ExecutionOrder, None, None);
            return report;
        }

        let mut states: HashMap<String, AxisState> = HashMap::with_capacity(order.len());
        let mut retained_elements = input_elements;
        for name in order {
            if deadline_expired(deadline) {
                report.decline(OneAxisDeclineReason::Deadline, Some(name), None);
                return report;
            }
            let Some(node) = self.node(name) else {
                report.decline(OneAxisDeclineReason::MissingNode, Some(name), None);
                return report;
            };
            let layer_type = node.layer().layer_type();
            let Some(declared_output_shape) = self.declared_shape(name) else {
                report.decline(
                    OneAxisDeclineReason::MissingDeclaredShape,
                    Some(name),
                    Some(layer_type),
                );
                return report;
            };
            if declared_output_shape.len() > ONE_AXIS_MAX_RANK {
                report.decline(
                    OneAxisDeclineReason::InvalidInputShape,
                    Some(name),
                    Some(layer_type),
                );
                return report;
            }
            let output_shape = declared_output_shape.to_vec();
            let Some(output_elements) = checked_elements(&output_shape) else {
                report.decline(
                    OneAxisDeclineReason::InvalidInputShape,
                    Some(name),
                    Some(layer_type),
                );
                return report;
            };
            if output_elements > ONE_AXIS_MAX_TENSOR_ELEMENTS {
                report.decline(
                    OneAxisDeclineReason::TensorElementLimit,
                    Some(name),
                    Some(layer_type),
                );
                return report;
            }
            let Some(next_retained) = retained_elements.checked_add(output_elements) else {
                report.decline(
                    OneAxisDeclineReason::TotalElementLimit,
                    Some(name),
                    Some(layer_type),
                );
                return report;
            };
            if next_retained > ONE_AXIS_MAX_TOTAL_ELEMENTS {
                report.decline(
                    OneAxisDeclineReason::TotalElementLimit,
                    Some(name),
                    Some(layer_type),
                );
                return report;
            }
            retained_elements = next_retained;

            let mut inputs = Vec::with_capacity(node.inputs().len());
            for input_name in node.inputs() {
                let Some(state) = input_state(&states, &network_input, input_name) else {
                    report.decline(
                        OneAxisDeclineReason::ForwardOrMissingInput,
                        Some(name),
                        Some(layer_type),
                    );
                    return report;
                };
                inputs.push(state);
            }

            let state = match node.layer() {
                Layer::Slice(layer) => {
                    let [input] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    let Some(dependency) = slice_dependency(input, layer, &output_shape, deadline)
                    else {
                        let reason = if deadline_expired(deadline) {
                            OneAxisDeclineReason::Deadline
                        } else {
                            OneAxisDeclineReason::ShapeMismatch
                        };
                        report.decline(reason, Some(name), Some(layer_type));
                        return report;
                    };
                    let expression = if dependency.iter().any(|&depends| depends) {
                        input.expression
                    } else {
                        AxisExpression::Constant
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression,
                    }
                }
                Layer::Linear(layer) => {
                    let [input] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    let Some((&last, leading)) = input.shape.split_last() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    let mut expected_shape = leading.to_vec();
                    expected_shape.push(layer.out_features());
                    if last != layer.in_features() || output_shape != expected_shape {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    if input.expression == AxisExpression::SigmoidPiecewiseAffine {
                        report.decline(
                            OneAxisDeclineReason::NonAffineComposition,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = input.depends_on_axis();
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression: if depends {
                            AxisExpression::PiecewiseAffine
                        } else {
                            AxisExpression::Constant
                        },
                    }
                }
                Layer::AddConstant(layer) => {
                    let [input] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    if layer.constant().ndim() > ONE_AXIS_MAX_RANK
                        || layer.constant().len() > ONE_AXIS_MAX_TENSOR_ELEMENTS
                        || crate::shape::broadcast_shapes(&input.shape, layer.constant().shape())
                            .as_deref()
                            != Some(output_shape.as_slice())
                        || input.expression == AxisExpression::SigmoidPiecewiseAffine
                    {
                        report.decline(
                            if input.expression == AxisExpression::SigmoidPiecewiseAffine {
                                OneAxisDeclineReason::NonAffineComposition
                            } else {
                                OneAxisDeclineReason::ShapeMismatch
                            },
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = input.depends_on_axis();
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression: if depends {
                            AxisExpression::PiecewiseAffine
                        } else {
                            AxisExpression::Constant
                        },
                    }
                }
                Layer::ReduceSum(layer) => {
                    let [input] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    if reduce_sum_shape(&input.shape, &layer.axes, layer.keepdims).as_deref()
                        != Some(output_shape.as_slice())
                        || input.expression == AxisExpression::SigmoidPiecewiseAffine
                    {
                        report.decline(
                            if input.expression == AxisExpression::SigmoidPiecewiseAffine {
                                OneAxisDeclineReason::NonAffineComposition
                            } else {
                                OneAxisDeclineReason::ShapeMismatch
                            },
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = input.depends_on_axis();
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression: if depends {
                            AxisExpression::PiecewiseAffine
                        } else {
                            AxisExpression::Constant
                        },
                    }
                }
                Layer::ReLU(_) => {
                    let [input] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    if input.shape != output_shape {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    if input.expression == AxisExpression::SigmoidPiecewiseAffine {
                        report.decline(
                            OneAxisDeclineReason::NonAffineComposition,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = input.depends_on_axis();
                    report.dynamic_relu_nodes += usize::from(depends);
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression: if depends {
                            AxisExpression::PiecewiseAffine
                        } else {
                            AxisExpression::Constant
                        },
                    }
                }
                Layer::MulBinary(_) => {
                    let [left, right] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    if crate::shape::broadcast_shapes(&left.shape, &right.shape).as_deref()
                        != Some(output_shape.as_slice())
                    {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let left_depends = left.depends_on_axis();
                    let right_depends = right.depends_on_axis();
                    if left_depends && right_depends {
                        report.decline(
                            OneAxisDeclineReason::DynamicMulOperands,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let expression = if left_depends {
                        left.expression
                    } else if right_depends {
                        right.expression
                    } else {
                        AxisExpression::Constant
                    };
                    if expression == AxisExpression::SigmoidPiecewiseAffine {
                        report.decline(
                            OneAxisDeclineReason::NonAffineComposition,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = left_depends || right_depends;
                    report.constant_sided_mul_nodes += usize::from(depends);
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression,
                    }
                }
                Layer::Div(_) => {
                    let [numerator, denominator] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    if crate::shape::broadcast_shapes(&numerator.shape, &denominator.shape)
                        .as_deref()
                        != Some(output_shape.as_slice())
                    {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    if denominator.depends_on_axis() {
                        report.decline(
                            OneAxisDeclineReason::DynamicDivisor,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = numerator.depends_on_axis();
                    if depends && numerator.expression == AxisExpression::SigmoidPiecewiseAffine {
                        report.decline(
                            OneAxisDeclineReason::NonAffineComposition,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    if depends {
                        report.constant_divisor_nodes.push(name.clone());
                    }
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression: if depends {
                            numerator.expression
                        } else {
                            AxisExpression::Constant
                        },
                    }
                }
                Layer::Concat(layer) => {
                    if inputs.is_empty() {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    if concat_shape(&inputs, layer).as_deref() != Some(output_shape.as_slice()) {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    if inputs
                        .iter()
                        .any(|input| input.expression == AxisExpression::SigmoidPiecewiseAffine)
                    {
                        report.decline(
                            OneAxisDeclineReason::NonAffineComposition,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = inputs.iter().any(|input| input.depends_on_axis());
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression: if depends {
                            AxisExpression::PiecewiseAffine
                        } else {
                            AxisExpression::Constant
                        },
                    }
                }
                Layer::Sigmoid(_) => {
                    let [input] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    if input.shape != output_shape {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    if input.expression == AxisExpression::SigmoidPiecewiseAffine {
                        report.decline(
                            OneAxisDeclineReason::NonAffineComposition,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let depends = input.depends_on_axis();
                    if depends {
                        report.dynamic_sigmoid_nodes += 1;
                    } else {
                        report.static_sigmoid_nodes += 1;
                    }
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression: if depends {
                            AxisExpression::SigmoidPiecewiseAffine
                        } else {
                            AxisExpression::Constant
                        },
                    }
                }
                Layer::Sub(_) => {
                    let [left, right] = inputs.as_slice() else {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    if crate::shape::broadcast_shapes(&left.shape, &right.shape).as_deref()
                        != Some(output_shape.as_slice())
                    {
                        report.decline(
                            OneAxisDeclineReason::ShapeMismatch,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    }
                    let expression = match (left.expression, right.expression) {
                        (AxisExpression::Constant, AxisExpression::Constant) => {
                            AxisExpression::Constant
                        }
                        (
                            AxisExpression::Constant | AxisExpression::PiecewiseAffine,
                            AxisExpression::Constant | AxisExpression::PiecewiseAffine,
                        ) => AxisExpression::PiecewiseAffine,
                        (AxisExpression::SigmoidPiecewiseAffine, AxisExpression::Constant)
                        | (AxisExpression::Constant, AxisExpression::SigmoidPiecewiseAffine) => {
                            AxisExpression::SigmoidPiecewiseAffine
                        }
                        _ => {
                            report.decline(
                                OneAxisDeclineReason::NonAffineComposition,
                                Some(name),
                                Some(layer_type),
                            );
                            return report;
                        }
                    };
                    let depends = left.depends_on_axis() || right.depends_on_axis();
                    let Some(dependency) = all_dependency(&output_shape, depends) else {
                        report.decline(
                            OneAxisDeclineReason::TensorElementLimit,
                            Some(name),
                            Some(layer_type),
                        );
                        return report;
                    };
                    AxisState {
                        shape: output_shape,
                        dependency,
                        expression,
                    }
                }
                _ => {
                    report.decline(
                        OneAxisDeclineReason::UnsupportedLayer,
                        Some(name),
                        Some(layer_type),
                    );
                    return report;
                }
            };

            report.nodes_examined += 1;
            states.insert(name.clone(), state);
        }

        let Some(output) = states.get(self.output_name()) else {
            report.decline(
                OneAxisDeclineReason::ForwardOrMissingInput,
                Some(self.output_name()),
                None,
            );
            return report;
        };
        let class = match output.expression {
            AxisExpression::Constant => OneAxisAlgebraClass::Constant,
            AxisExpression::PiecewiseAffine => OneAxisAlgebraClass::PiecewiseAffine,
            AxisExpression::SigmoidPiecewiseAffine => {
                if output.dependency.len() != 1 {
                    report.decline(
                        OneAxisDeclineReason::NonScalarSigmoidOutput,
                        Some(self.output_name()),
                        self.node(self.output_name())
                            .map(|node| node.layer().layer_type()),
                    );
                    return report;
                }
                OneAxisAlgebraClass::PeelableMonotoneSigmoid
            }
        };
        report.class = Some(class);
        report
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::layers::{
        DivLayer, MulBinaryLayer, ReLULayer, SigmoidLayer, SliceLayer, SubLayer, TanhLayer,
    };
    use crate::network::GraphNode;

    fn set_shape(graph: &mut GraphNetwork, name: &str, shape: &[usize]) {
        graph.set_declared_shape(name, shape.to_vec());
    }

    fn add_slice(graph: &mut GraphNetwork, name: &str, input: &str, start: usize, end: usize) {
        graph.add_node(GraphNode::new(
            name,
            Layer::Slice(SliceLayer::new(0, start, end)),
            vec![input.to_string()],
        ));
        set_shape(graph, name, &[end - start]);
    }

    #[test]
    fn recognizes_constant_sided_mul_div_and_peelable_sigmoid() {
        let mut graph = GraphNetwork::new();
        set_shape(&mut graph, NETWORK_INPUT, &[2]);
        add_slice(&mut graph, "x", NETWORK_INPUT, 0, 1);
        add_slice(&mut graph, "weight", NETWORK_INPUT, 1, 2);
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["x".to_string()],
        ));
        set_shape(&mut graph, "relu", &[1]);
        graph.add_node(GraphNode::binary(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            "relu",
            "weight",
        ));
        set_shape(&mut graph, "mul", &[1]);
        graph.add_node(GraphNode::binary(
            "div",
            Layer::Div(DivLayer),
            "mul",
            "weight",
        ));
        set_shape(&mut graph, "div", &[1]);
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["div".to_string()],
        ));
        set_shape(&mut graph, "sigmoid", &[1]);
        graph.set_output("sigmoid");

        let report = graph.recognize_one_free_axis_algebra_until(
            &[2],
            0,
            Instant::now() + Duration::from_secs(1),
        );
        assert_eq!(
            report.class,
            Some(OneAxisAlgebraClass::PeelableMonotoneSigmoid)
        );
        assert_eq!(report.decline, None);
        assert_eq!(report.dynamic_relu_nodes, 1);
        assert_eq!(report.constant_sided_mul_nodes, 1);
        assert_eq!(report.constant_divisor_nodes, ["div"]);
        assert_eq!(report.dynamic_sigmoid_nodes, 1);
        assert_eq!(report.static_sigmoid_nodes, 0);
    }

    #[test]
    fn recognizes_axis_constant_minus_dynamic_sigmoid() {
        let mut graph = GraphNetwork::new();
        set_shape(&mut graph, NETWORK_INPUT, &[2]);
        add_slice(&mut graph, "dynamic", NETWORK_INPUT, 0, 1);
        add_slice(&mut graph, "fixed", NETWORK_INPUT, 1, 2);
        graph.add_node(GraphNode::new(
            "dynamic_sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["dynamic".to_string()],
        ));
        set_shape(&mut graph, "dynamic_sigmoid", &[1]);
        graph.add_node(GraphNode::new(
            "fixed_sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["fixed".to_string()],
        ));
        set_shape(&mut graph, "fixed_sigmoid", &[1]);
        graph.add_node(GraphNode::binary(
            "difference",
            Layer::Sub(SubLayer),
            "fixed_sigmoid",
            "dynamic_sigmoid",
        ));
        set_shape(&mut graph, "difference", &[1]);
        graph.set_output("difference");

        let report = graph.recognize_one_free_axis_algebra_until(
            &[2],
            0,
            Instant::now() + Duration::from_secs(1),
        );
        assert_eq!(
            report.class,
            Some(OneAxisAlgebraClass::PeelableMonotoneSigmoid)
        );
        assert_eq!(report.dynamic_sigmoid_nodes, 1);
        assert_eq!(report.static_sigmoid_nodes, 1);
    }

    #[test]
    fn rejects_varying_times_varying_and_dynamic_divisor() {
        let mut mul_graph = GraphNetwork::new();
        set_shape(&mut mul_graph, NETWORK_INPUT, &[1]);
        add_slice(&mut mul_graph, "left", NETWORK_INPUT, 0, 1);
        add_slice(&mut mul_graph, "right", NETWORK_INPUT, 0, 1);
        mul_graph.add_node(GraphNode::binary(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            "left",
            "right",
        ));
        set_shape(&mut mul_graph, "mul", &[1]);
        mul_graph.set_output("mul");
        let mul = mul_graph.recognize_one_free_axis_algebra_until(
            &[1],
            0,
            Instant::now() + Duration::from_secs(1),
        );
        assert_eq!(
            mul.decline.as_ref().map(|decline| decline.reason),
            Some(OneAxisDeclineReason::DynamicMulOperands)
        );

        let mut div_graph = GraphNetwork::new();
        set_shape(&mut div_graph, NETWORK_INPUT, &[1]);
        add_slice(&mut div_graph, "numerator", NETWORK_INPUT, 0, 1);
        add_slice(&mut div_graph, "denominator", NETWORK_INPUT, 0, 1);
        div_graph.add_node(GraphNode::binary(
            "div",
            Layer::Div(DivLayer),
            "numerator",
            "denominator",
        ));
        set_shape(&mut div_graph, "div", &[1]);
        div_graph.set_output("div");
        let div = div_graph.recognize_one_free_axis_algebra_until(
            &[1],
            0,
            Instant::now() + Duration::from_secs(1),
        );
        assert_eq!(
            div.decline.as_ref().map(|decline| decline.reason),
            Some(OneAxisDeclineReason::DynamicDivisor)
        );
    }

    #[test]
    fn rejects_unsupported_nonlinearity_and_expired_deadline() {
        let mut graph = GraphNetwork::new();
        set_shape(&mut graph, NETWORK_INPUT, &[1]);
        graph.add_node(GraphNode::from_input("tanh", Layer::Tanh(TanhLayer)));
        set_shape(&mut graph, "tanh", &[1]);
        graph.set_output("tanh");

        let unsupported = graph.recognize_one_free_axis_algebra_until(
            &[1],
            0,
            Instant::now() + Duration::from_secs(1),
        );
        assert_eq!(
            unsupported.decline.as_ref().map(|decline| decline.reason),
            Some(OneAxisDeclineReason::UnsupportedLayer)
        );

        let expired = graph.recognize_one_free_axis_algebra_until(&[1], 0, Instant::now());
        assert_eq!(
            expired.decline.as_ref().map(|decline| decline.reason),
            Some(OneAxisDeclineReason::Deadline)
        );
        assert_eq!(expired.nodes_examined, 0);

        let oversized_rank = graph.recognize_one_free_axis_algebra_until(
            &[1; ONE_AXIS_MAX_RANK + 1],
            0,
            Instant::now() + Duration::from_secs(1),
        );
        assert_eq!(
            oversized_rank
                .decline
                .as_ref()
                .map(|decline| decline.reason),
            Some(OneAxisDeclineReason::InvalidInputShape)
        );

        let out_of_range = graph.recognize_one_free_axis_algebra_until(
            &[1],
            1,
            Instant::now() + Duration::from_secs(1),
        );
        assert_eq!(
            out_of_range.decline.as_ref().map(|decline| decline.reason),
            Some(OneAxisDeclineReason::FreeAxisOutOfRange)
        );
    }
}
