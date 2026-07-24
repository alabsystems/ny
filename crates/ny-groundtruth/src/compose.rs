// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Composition of ground-truth graphs: affine pre-transforms (pose) and
//! min/max over a fixed set of primitives (plan §2, `compose`).
//!
//! - [`with_pose`] builds `x ↦ g(Ax + b)` by prepending one Linear node —
//!   evaluate a primitive residual in a transformed frame (e.g. a fitted
//!   primitive's local coordinates).
//! - [`min_of`] / [`max_of`] merge several primitive graphs sharing the
//!   network input and fold their outputs with `MinBinary` / `MaxBinary`
//!   layers. For residuals that are negative inside the solid, `min` is the
//!   residual of the *union* and `max` of the *intersection* (CSG-style
//!   compound models).
//!
//! Pose constants follow the same §2.3 contract as builder parameters: each
//! entry of `A` and `b` must be finite and f64 → f32 exact (no derived
//! products arise — the pose is its own graph node, evaluated by the sound
//! propagation arithmetic). Note the semantic caveat: `with_pose` composes
//! *exactly* the affine map you supply; if you mean a rigid motion, `A` must
//! be orthonormal, and exactly-representable orthonormal f32 matrices are the
//! signed permutations (same descent argument as unit axes — general
//! rotations await the plan §2.3 interval-widening follow-up).

use ndarray::{Array1, Array2};

use ny_propagate::layers::{LinearLayer, MaxBinaryLayer, MinBinaryLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};

use crate::error::{GroundTruthError, Result};
use crate::exact::require_exact_vec3;

/// An affine pre-transform `x ↦ Ax + b` with exactly-representable constants.
#[derive(Debug, Clone)]
pub struct Pose {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl Pose {
    /// Validate and create a pose from row-major `A` and translation `b`.
    ///
    /// Every entry must be finite and round-trip f64 → f32 exactly
    /// (plan §2.3 — nothing is silently rounded).
    pub fn new(linear: [[f64; 3]; 3], translation: [f64; 3]) -> Result<Self> {
        let mut weight = Array2::<f32>::zeros((3, 3));
        for (i, row) in linear.iter().enumerate() {
            let exact = require_exact_vec3(&format!("linear[{i}]"), *row)?;
            for (j, &v) in exact.iter().enumerate() {
                weight[[i, j]] = v;
            }
        }
        let b = require_exact_vec3("translation", translation)?;
        Ok(Self {
            weight,
            bias: Array1::from(b.to_vec()),
        })
    }

    /// A pure translation pose `x ↦ x + t`.
    pub fn translation(t: [f64; 3]) -> Result<Self> {
        Self::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t)
    }
}

/// Build `x ↦ g(Ax + b)`: the primitive graph re-rooted on a Linear pose
/// node. The primitive itself is copied unchanged (its node names are
/// preserved; the pose node picks a fresh name).
pub fn with_pose(primitive: &GraphNetwork, pose: &Pose) -> Result<GraphNetwork> {
    if primitive.output_name().is_empty() {
        return Err(GroundTruthError::InvalidComposition(
            "primitive has no output node set".to_string(),
        ));
    }

    // Pick a pose node name that cannot collide with the primitive's nodes.
    let mut pose_name = "gt_pose".to_string();
    let mut suffix = 0_usize;
    while primitive.contains_node(&pose_name) {
        pose_name = format!("gt_pose_{suffix}");
        suffix += 1;
    }

    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input(
        pose_name.clone(),
        Layer::Linear(LinearLayer::new(
            pose.weight.clone(),
            Some(pose.bias.clone()),
        )?),
    ))?;
    for name in primitive.node_names() {
        let node = primitive.node(name).ok_or_else(|| {
            GroundTruthError::InvalidComposition(format!("node '{name}' missing"))
        })?;
        let inputs: Vec<String> = node
            .inputs()
            .iter()
            .map(|input| {
                if input == NETWORK_INPUT {
                    pose_name.clone()
                } else {
                    input.clone()
                }
            })
            .collect();
        g.try_add_node(GraphNode::new(name.clone(), node.layer().clone(), inputs))?;
    }
    g.set_output(primitive.output_name().to_string());
    Ok(g)
}

/// Element-wise minimum of several primitive residuals (union of solids for
/// negative-inside residuals). Requires at least one primitive; all must
/// share the same output shape (checked at propagation time).
pub fn min_of(primitives: &[GraphNetwork]) -> Result<GraphNetwork> {
    combine(primitives, true)
}

/// Element-wise maximum of several primitive residuals (intersection of
/// solids for negative-inside residuals).
pub fn max_of(primitives: &[GraphNetwork]) -> Result<GraphNetwork> {
    combine(primitives, false)
}

fn combine(primitives: &[GraphNetwork], is_min: bool) -> Result<GraphNetwork> {
    if primitives.is_empty() {
        return Err(GroundTruthError::InvalidComposition(
            "min/max composition needs at least one primitive".to_string(),
        ));
    }

    let mut g = GraphNetwork::new();
    let mut outputs = Vec::with_capacity(primitives.len());
    for (index, primitive) in primitives.iter().enumerate() {
        if primitive.output_name().is_empty() {
            return Err(GroundTruthError::InvalidComposition(format!(
                "primitive {index} has no output node set"
            )));
        }
        // Prefix every node so the merged DAG stays collision-free; all
        // primitives keep sharing NETWORK_INPUT. try_add_node still catches
        // any residual collision.
        let prefix = format!("m{index}_");
        for name in primitive.node_names() {
            let node = primitive.node(name).ok_or_else(|| {
                GroundTruthError::InvalidComposition(format!("node '{name}' missing"))
            })?;
            let inputs: Vec<String> = node
                .inputs()
                .iter()
                .map(|input| {
                    if input == NETWORK_INPUT {
                        NETWORK_INPUT.to_string()
                    } else {
                        format!("{prefix}{input}")
                    }
                })
                .collect();
            g.try_add_node(GraphNode::new(
                format!("{prefix}{name}"),
                node.layer().clone(),
                inputs,
            ))?;
        }
        outputs.push(format!("{prefix}{}", primitive.output_name()));
    }

    let mut acc = outputs[0].clone();
    for (index, output) in outputs.iter().enumerate().skip(1) {
        let layer = if is_min {
            Layer::MinBinary(MinBinaryLayer)
        } else {
            Layer::MaxBinary(MaxBinaryLayer)
        };
        let name = format!("gt_combine_{index}");
        g.try_add_node(GraphNode::binary(name.clone(), layer, acc, output.clone()))?;
        acc = name;
    }
    g.set_output(acc);
    Ok(g)
}
