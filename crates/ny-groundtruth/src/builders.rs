// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Builders for geometric ground-truth graphs in squared/residual polynomial
//! form (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md` §2, Route A).
//!
//! Every builder returns a [`GraphNetwork`] mapping a 3-D point
//! `x = (x0, x1, x2)` (the `NETWORK_INPUT`) to a single residual output:
//!
//! | builder | residual g(x) | zero set |
//! |---------|----------------|----------|
//! | [`signed_plane_distance`] | `n·x + d` | plane `n·x + d = 0` |
//! | [`sphere_residual`] | `‖x−c‖² − r²` | sphere of radius `r` about `c` |
//! | [`cylinder_residual`] | `‖(x−p) − ((x−p)·a)a‖² − r²` | infinite cylinder, axis `a` through `p` |
//! | [`cone_residual`] | `cos²α·‖x−q‖² − ((x−q)·a)²` | infinite double cone, apex `q`, half-angle `α` |
//! | [`torus_residual`] | `(‖x−p‖² + R² − r²)² − 4R²·‖(x−p) − ((x−p)·a)a‖²` | torus, axis `a` about `p` |
//!
//! **Rule (plan §2): squared/residual forms only — no sqrt, no division.**
//! Each residual is negative strictly inside the solid, zero on the surface,
//! and positive outside (for the plane: the signed side; for the cone:
//! negative inside either nappe of the solid double cone).
//!
//! # Constant-handling contract (plan §2.3)
//!
//! Parameters are f64 for API convenience, but a ground-truth graph must
//! denote its residual *exactly*; constants are stored as f32 (the
//! `ny-propagate` layer width). Therefore:
//!
//! - every parameter must be finite and round-trip f64 → f32 exactly, else
//!   the builder rejects it ([`GroundTruthError::InexactParameter`] /
//!   [`GroundTruthError::NonFiniteParameter`]);
//! - every derived constant (`r²`, `R² − r²`, `4R²`, projection entries
//!   `δ_ij − a_i a_j`, biases `−(I − a aᵀ)p`) is computed in exact
//!   arbitrary-precision rational arithmetic and must itself be exactly
//!   f32-representable, else the builder rejects it
//!   ([`GroundTruthError::InexactDerivedConstant`]);
//! - axes must be *exactly* unit length under exact rational arithmetic
//!   ([`GroundTruthError::AxisNotUnit`]); in f32 that means the signed
//!   standard basis vectors (a sum-of-three-squares descent shows
//!   `x² + y² + z² = 4^k` forces an axis-aligned solution). General fitted
//!   axes await the plan's interval-widened constants (§2.3 follow-up) —
//!   nothing is ever silently rounded.
//!
//! At propagation time the layers apply the usual sound directed-rounding
//! FP bound arithmetic, so evaluating a ground-truth graph at a zero-width
//! point yields a sound enclosure of the exact rational residual (this is
//! what the golden tests check against [`crate::reference`]).

use ndarray::{Array1, Array2};
use num_rational::BigRational;
use num_traits::{One, Zero};

use ny_propagate::layers::{AddLayer, LinearLayer, PowConstantLayer, SubLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

use crate::error::{GroundTruthError, Result};
use crate::exact::{
    rational, rational_to_exact_f32, require_exact_f32, require_exact_vec3, require_unit_axis,
};

/// Signed distance to a plane: `g(x) = n·x + d` (a single Linear layer).
///
/// The zero set is the plane `n·x + d = 0` for any nonzero `n`. The value is
/// the *metric* signed distance exactly when `‖n‖ = 1`; for other `n` it is
/// the signed algebraic distance scaled by `‖n‖` (the zero set is unchanged).
/// `n` is deliberately not required to be unit: unlike the quadric builders,
/// a non-unit `n` still denotes exactly the intended plane.
pub fn signed_plane_distance(normal: [f64; 3], offset: f64) -> Result<GraphNetwork> {
    let n = require_exact_vec3("normal", normal)?;
    let d = require_exact_f32("offset", offset)?;
    if n == [0.0; 3] {
        return Err(GroundTruthError::DegenerateParameter {
            name: "normal".to_string(),
            reason: "must be nonzero".to_string(),
        });
    }

    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input("gt_plane", row_layer(n, Some(d))?))?;
    g.set_output("gt_plane");
    Ok(g)
}

/// Sphere residual: `g(x) = ‖x − c‖² − r²`.
///
/// Negative inside the sphere, zero on it, positive outside. Requires
/// `r > 0`; `r²` is derived in exact rational arithmetic.
pub fn sphere_residual(center: [f64; 3], radius: f64) -> Result<GraphNetwork> {
    let c = require_exact_vec3("center", center)?;
    let r = require_positive_exact("radius", radius)?;
    let r_sq = rational_to_exact_f32("radius^2", &(rational(r) * rational(r)))?;

    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input("gt_centered", translate_layer(c)?))?;
    g.try_add_node(GraphNode::new(
        "gt_centered_sq",
        square_layer(),
        vec!["gt_centered".to_string()],
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_residual",
        row_layer([1.0, 1.0, 1.0], Some(-r_sq))?,
        vec!["gt_centered_sq".to_string()],
    ))?;
    g.set_output("gt_residual");
    Ok(g)
}

/// Infinite-cylinder residual: `g(x) = ‖(x−p) − ((x−p)·a)a‖² − r²`.
///
/// `a` is the (exactly unit) axis direction, `p` a point on the axis and
/// `r > 0` the radius. `(x−p) − ((x−p)·a)a = M(x−p)` with `M = I − a aᵀ` is
/// linear in `x`, so the whole residual is one Linear layer, an element-wise
/// square, and a summing Linear layer. Negative strictly inside the
/// cylinder, zero on its surface, positive outside.
pub fn cylinder_residual(axis: [f64; 3], point: [f64; 3], radius: f64) -> Result<GraphNetwork> {
    let a = require_unit_axis("axis", axis)?;
    let p = require_exact_vec3("point", point)?;
    let r = require_positive_exact("radius", radius)?;
    let r_sq = rational_to_exact_f32("radius^2", &(rational(r) * rational(r)))?;

    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input(
        "gt_perp",
        perp_projection_layer(a, p)?,
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_perp_sq",
        square_layer(),
        vec!["gt_perp".to_string()],
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_residual",
        row_layer([1.0, 1.0, 1.0], Some(-r_sq))?,
        vec!["gt_perp_sq".to_string()],
    ))?;
    g.set_output("gt_residual");
    Ok(g)
}

/// Infinite double-cone residual in squared form:
/// `g(x) = cos²α · ‖x − q‖² − ((x − q)·a)²`.
///
/// This is the standard quadric form of the right circular cone with apex
/// `q`, (exactly unit) axis `a`, and half-angle `α`: a point lies on the
/// cone surface iff the angle `θ` between `x − q` and the axis satisfies
/// `cos²θ = cos²α`, i.e. `((x−q)·a)² = cos²α‖x−q‖²`. Since
/// `g(x) = ‖x−q‖²(cos²α − cos²θ)`, the residual is negative strictly inside
/// either nappe of the solid double cone, zero on the surface (and at the
/// apex), and positive outside. The squared form cannot distinguish the two
/// nappes — that is inherent to the plan's no-sqrt rule.
///
/// The half-angle enters as `cos_half_angle_sq = cos²α ∈ (0, 1)` so that the
/// parameter is rational (no transcendental evaluation at build time).
pub fn cone_residual(
    axis: [f64; 3],
    apex: [f64; 3],
    cos_half_angle_sq: f64,
) -> Result<GraphNetwork> {
    let a = require_unit_axis("axis", axis)?;
    let q = require_exact_vec3("apex", apex)?;
    let k = require_exact_f32("cos_half_angle_sq", cos_half_angle_sq)?;
    if !(k > 0.0 && k < 1.0) {
        return Err(GroundTruthError::DegenerateParameter {
            name: "cos_half_angle_sq".to_string(),
            reason: format!("must lie strictly inside (0, 1), got {k}"),
        });
    }

    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input("gt_centered", translate_layer(q)?))?;
    g.try_add_node(GraphNode::new(
        "gt_centered_sq",
        square_layer(),
        vec!["gt_centered".to_string()],
    ))?;
    // cos²α · ‖x − q‖²
    g.try_add_node(GraphNode::new(
        "gt_radial",
        row_layer([k, k, k], None)?,
        vec!["gt_centered_sq".to_string()],
    ))?;
    // (x − q)·a, then its square
    g.try_add_node(GraphNode::new(
        "gt_axial",
        row_layer(a, None)?,
        vec!["gt_centered".to_string()],
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_axial_sq",
        square_layer(),
        vec!["gt_axial".to_string()],
    ))?;
    g.try_add_node(GraphNode::binary(
        "gt_residual",
        Layer::Sub(SubLayer),
        "gt_radial",
        "gt_axial_sq",
    ))?;
    g.set_output("gt_residual");
    Ok(g)
}

/// Torus residual in polynomial (quartic) form:
/// `g(x) = (‖x−p‖² + R² − r²)² − 4R² · ‖(x−p) − ((x−p)·a)a‖²`.
///
/// `a` is the (exactly unit) axis, `p` the torus center, `R > 0` the major
/// radius and `r > 0` the minor radius. This is the standard implicit torus
/// quartic (expand `(√(x₁²+x₂²) − R)² + x₃² = r²` and square away the root).
/// For a ring torus (`r < R`) the residual is negative strictly inside the
/// tube, zero on the surface, and positive outside. Derived constants
/// `R² − r²` and `4R²` are computed in exact rational arithmetic.
pub fn torus_residual(
    axis: [f64; 3],
    center: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
) -> Result<GraphNetwork> {
    let a = require_unit_axis("axis", axis)?;
    let p = require_exact_vec3("center", center)?;
    let big_r = require_positive_exact("major_radius", major_radius)?;
    let r = require_positive_exact("minor_radius", minor_radius)?;
    let ring_bias = rational_to_exact_f32(
        "major_radius^2 - minor_radius^2",
        &(rational(big_r) * rational(big_r) - rational(r) * rational(r)),
    )?;
    let neg_4r_sq = rational_to_exact_f32(
        "-4 * major_radius^2",
        &(-rational(4.0) * rational(big_r) * rational(big_r)),
    )?;

    let mut g = GraphNetwork::new();
    // s(x) = ‖x − p‖² + R² − r², then s²
    g.try_add_node(GraphNode::from_input("gt_centered", translate_layer(p)?))?;
    g.try_add_node(GraphNode::new(
        "gt_centered_sq",
        square_layer(),
        vec!["gt_centered".to_string()],
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_ring",
        row_layer([1.0, 1.0, 1.0], Some(ring_bias))?,
        vec!["gt_centered_sq".to_string()],
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_ring_sq",
        square_layer(),
        vec!["gt_ring".to_string()],
    ))?;
    // −4R² · ‖(x−p) − ((x−p)·a)a‖²
    g.try_add_node(GraphNode::from_input(
        "gt_perp",
        perp_projection_layer(a, p)?,
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_perp_sq",
        square_layer(),
        vec!["gt_perp".to_string()],
    ))?;
    g.try_add_node(GraphNode::new(
        "gt_perp_scaled",
        row_layer([neg_4r_sq, neg_4r_sq, neg_4r_sq], None)?,
        vec!["gt_perp_sq".to_string()],
    ))?;
    g.try_add_node(GraphNode::binary(
        "gt_residual",
        Layer::Add(AddLayer),
        "gt_ring_sq",
        "gt_perp_scaled",
    ))?;
    g.set_output("gt_residual");
    Ok(g)
}

// --- shared layer helpers -------------------------------------------------

/// Element-wise square as a `PowConstant(2)` layer (exact exponent; its CROWN
/// relaxation is the dedicated `pow2` envelope, tighter and cheaper than a
/// duplicated-input `MulBinary`).
fn square_layer() -> Layer {
    Layer::PowConstant(PowConstantLayer::square())
}

/// `x ↦ x − p` as a Linear layer (identity weight, bias `−p`; negation of a
/// finite f32 is exact).
fn translate_layer(p: [f32; 3]) -> Result<Layer> {
    let weight = Array2::<f32>::eye(3);
    let bias = Array1::from(vec![-p[0], -p[1], -p[2]]);
    Ok(Layer::Linear(LinearLayer::new(weight, Some(bias))?))
}

/// A single-row Linear layer `v ↦ coeffs·v (+ bias)` producing one output.
fn row_layer(coeffs: [f32; 3], bias: Option<f32>) -> Result<Layer> {
    let weight =
        Array2::from_shape_vec((1, 3), coeffs.to_vec()).expect("1x3 shape matches 3 items");
    let bias = bias.map(|b| Array1::from(vec![b]));
    Ok(Layer::Linear(LinearLayer::new(weight, bias)?))
}

/// `x ↦ M(x − p)` with `M = I − a aᵀ` (orthogonal projection onto the plane
/// normal to the unit axis `a`), as a Linear layer with weight `M` and bias
/// `−Mp`. All nine entries of `M` and the three bias components are derived
/// in exact rational arithmetic and must be exactly f32-representable.
fn perp_projection_layer(a: [f32; 3], p: [f32; 3]) -> Result<Layer> {
    let a_q: Vec<BigRational> = a.iter().map(|&ai| rational(ai)).collect();
    let mut m_q = vec![vec![BigRational::zero(); 3]; 3];
    let mut weight = Array2::<f32>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut entry = -(&a_q[i] * &a_q[j]);
            if i == j {
                entry += BigRational::one();
            }
            weight[[i, j]] = rational_to_exact_f32(&format!("projection[{i}][{j}]"), &entry)?;
            m_q[i][j] = entry;
        }
    }
    let mut bias = Array1::<f32>::zeros(3);
    for i in 0..3 {
        let b_i = -(0..3)
            .map(|j| &m_q[i][j] * rational(p[j]))
            .fold(BigRational::zero(), |acc, q| acc + q);
        bias[i] = rational_to_exact_f32(&format!("-(projection * point)[{i}]"), &b_i)?;
    }
    Ok(Layer::Linear(LinearLayer::new(weight, Some(bias))?))
}

/// Validate a strictly positive, finite, f32-exact scalar parameter.
fn require_positive_exact(name: &str, value: f64) -> Result<f32> {
    let v = require_exact_f32(name, value)?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(GroundTruthError::DegenerateParameter {
            name: name.to_string(),
            reason: format!("must be strictly positive, got {v}"),
        })
    }
}
