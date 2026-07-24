// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The `.gt.json` sidecar format — portable serialization of a ground-truth
//! spec (plan §2, M2 deliverable).
//!
//! A sidecar names an M1 builder with its exact `f64` parameters, or a
//! `min`/`max` composition of such specs, optionally pre-transformed by a
//! pose:
//!
//! ```json
//! {
//!   "format": "gt/1",
//!   "builder": "cylinder",
//!   "params": { "axis": [0.0, 0.0, 1.0], "point": [1.0, -2.0, 0.5], "radius": 1.5 }
//! }
//! ```
//!
//! ```json
//! {
//!   "format": "gt/1",
//!   "compose": {
//!     "op": "min",
//!     "parts": [
//!       { "builder": "sphere", "params": { "center": [0.0, 0.0, 0.0], "radius": 1.0 } },
//!       { "builder": "plane",  "params": { "normal": [0.0, 0.0, 1.0], "offset": -0.25 } }
//!     ]
//!   }
//! }
//! ```
//!
//! `format` is required (and must be `"gt/1"`) at the root; nested `parts`
//! may omit it (when present it must match). Exactly one of `builder`+`params`
//! or `compose` must be given per node; `pose` is optional on any node and
//! applies `x ↦ g(Ax + b)` to that node's result.
//!
//! **Validation goes through the M1 builders** ([`GroundTruthSpec::build`]):
//! every parameter is subject to the plan §2.3 exact-constant contract, and a
//! rejection (inexact constant, non-unit axis, degenerate parameter)
//! propagates as the builder's typed [`GroundTruthError`] — a sidecar that
//! parses is not yet a sidecar that denotes an exact ground truth.

use std::path::Path;

use num_rational::BigRational;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use ny_propagate::GraphNetwork;

use crate::builders::{
    cone_residual, cylinder_residual, signed_plane_distance, sphere_residual, torus_residual,
};
use crate::compose::{max_of, min_of, with_pose, Pose};
use crate::error::{GroundTruthError, Result};
use crate::reference;

/// The sidecar format tag this loader accepts.
pub const GT_FORMAT: &str = "gt/1";

/// Which M1 primitive builder a spec node names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BuilderKind {
    /// [`signed_plane_distance`] — params `{normal, offset}`.
    Plane,
    /// [`sphere_residual`] — params `{center, radius}`.
    Sphere,
    /// [`cylinder_residual`] — params `{axis, point, radius}`.
    Cylinder,
    /// [`cone_residual`] — params `{axis, apex, cos_half_angle_sq}`.
    Cone,
    /// [`torus_residual`] — params `{axis, center, major_radius, minor_radius}`.
    Torus,
}

/// An affine pre-transform in sidecar form (`x ↦ Ax + b`), validated through
/// [`Pose::new`] (every entry f64 → f32 exact).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseSpec {
    /// Row-major `A`.
    pub linear: [[f64; 3]; 3],
    /// Translation `b`.
    pub translation: [f64; 3],
}

/// A `min`/`max` composition over sub-specs (CSG union/intersection for
/// negative-inside residuals).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComposeSpec {
    /// `"min"` (union) or `"max"` (intersection).
    pub op: ComposeOp,
    /// The composed sub-specs (at least one).
    pub parts: Vec<GroundTruthSpec>,
}

/// The composition operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComposeOp {
    /// Element-wise minimum ([`min_of`]).
    Min,
    /// Element-wise maximum ([`max_of`]).
    Max,
}

// Per-builder parameter shapes. Kept private: the public surface is the
// `builder`/`params` pair, mirroring the JSON.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PlaneParams {
    normal: [f64; 3],
    offset: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SphereParams {
    center: [f64; 3],
    radius: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CylinderParams {
    axis: [f64; 3],
    point: [f64; 3],
    radius: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ConeParams {
    axis: [f64; 3],
    apex: [f64; 3],
    cos_half_angle_sq: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TorusParams {
    axis: [f64; 3],
    center: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
}

/// A parsed (but not yet builder-validated) `.gt.json` ground-truth spec.
///
/// Obtain one with [`GroundTruthSpec::load`] / [`GroundTruthSpec::from_json_str`]
/// or the typed constructors ([`GroundTruthSpec::cylinder`], …), then call
/// [`GroundTruthSpec::build`] to validate through the M1 builders and obtain
/// the [`GraphNetwork`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundTruthSpec {
    /// Format tag; required (`"gt/1"`) at the root, optional on nested parts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Primitive builder name (mutually exclusive with `compose`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub builder: Option<BuilderKind>,
    /// The builder's parameters, exactly as the M1 builders take them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// Optional affine pre-transform applied to this node's result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<PoseSpec>,
    /// `min`/`max` composition (mutually exclusive with `builder`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compose: Option<ComposeSpec>,
}

fn params_value<T: Serialize>(builder: &str, params: &T) -> serde_json::Value {
    serde_json::to_value(params)
        .unwrap_or_else(|e| unreachable!("{builder} params always serialize: {e}"))
}

impl GroundTruthSpec {
    fn primitive(builder: BuilderKind, params: serde_json::Value) -> Self {
        GroundTruthSpec {
            format: Some(GT_FORMAT.to_string()),
            builder: Some(builder),
            params: Some(params),
            pose: None,
            compose: None,
        }
    }

    /// Spec for [`signed_plane_distance`] with the given `normal`/`offset`.
    #[must_use]
    pub fn plane(normal: [f64; 3], offset: f64) -> Self {
        Self::primitive(
            BuilderKind::Plane,
            params_value("plane", &PlaneParams { normal, offset }),
        )
    }

    /// Spec for [`sphere_residual`] with the given `center`/`radius`.
    #[must_use]
    pub fn sphere(center: [f64; 3], radius: f64) -> Self {
        Self::primitive(
            BuilderKind::Sphere,
            params_value("sphere", &SphereParams { center, radius }),
        )
    }

    /// Spec for [`cylinder_residual`] with the given `axis`/`point`/`radius`.
    #[must_use]
    pub fn cylinder(axis: [f64; 3], point: [f64; 3], radius: f64) -> Self {
        Self::primitive(
            BuilderKind::Cylinder,
            params_value(
                "cylinder",
                &CylinderParams {
                    axis,
                    point,
                    radius,
                },
            ),
        )
    }

    /// Spec for [`cone_residual`] with the given `axis`/`apex`/`cos_half_angle_sq`.
    #[must_use]
    pub fn cone(axis: [f64; 3], apex: [f64; 3], cos_half_angle_sq: f64) -> Self {
        Self::primitive(
            BuilderKind::Cone,
            params_value(
                "cone",
                &ConeParams {
                    axis,
                    apex,
                    cos_half_angle_sq,
                },
            ),
        )
    }

    /// Spec for [`torus_residual`] with the given `axis`/`center`/radii.
    #[must_use]
    pub fn torus(axis: [f64; 3], center: [f64; 3], major_radius: f64, minor_radius: f64) -> Self {
        Self::primitive(
            BuilderKind::Torus,
            params_value(
                "torus",
                &TorusParams {
                    axis,
                    center,
                    major_radius,
                    minor_radius,
                },
            ),
        )
    }

    /// A `min`/`max` composition of `parts`.
    #[must_use]
    pub fn composed(op: ComposeOp, parts: Vec<GroundTruthSpec>) -> Self {
        GroundTruthSpec {
            format: Some(GT_FORMAT.to_string()),
            builder: None,
            params: None,
            pose: None,
            compose: Some(ComposeSpec { op, parts }),
        }
    }

    /// This spec with an affine pre-transform (validated at build time).
    #[must_use]
    pub fn with_pose(mut self, pose: PoseSpec) -> Self {
        self.pose = Some(pose);
        self
    }

    /// Parse a spec from JSON text. The root `format` must be `"gt/1"`.
    ///
    /// # Errors
    /// [`GroundTruthError::InvalidSidecar`] on JSON syntax errors, unknown
    /// fields, or a missing/unsupported root `format`.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let spec: GroundTruthSpec = serde_json::from_str(json)
            .map_err(|e| GroundTruthError::InvalidSidecar(e.to_string()))?;
        match spec.format.as_deref() {
            Some(GT_FORMAT) => Ok(spec),
            Some(other) => Err(GroundTruthError::InvalidSidecar(format!(
                "unsupported format `{other}` (expected `{GT_FORMAT}`)"
            ))),
            None => Err(GroundTruthError::InvalidSidecar(format!(
                "missing required root field `format` (expected `{GT_FORMAT}`)"
            ))),
        }
    }

    /// Load a spec from a `.gt.json` file.
    ///
    /// # Errors
    /// [`GroundTruthError::SidecarIo`] when the file cannot be read, otherwise
    /// as [`GroundTruthSpec::from_json_str`].
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| GroundTruthError::SidecarIo {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        Self::from_json_str(&text)
    }

    /// Serialize to pretty JSON (the canonical sidecar body).
    ///
    /// # Errors
    /// [`GroundTruthError::InvalidSidecar`] if serialization fails (only
    /// possible for a hand-built spec carrying non-serializable params).
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| GroundTruthError::InvalidSidecar(e.to_string()))
    }

    /// Build the ground-truth [`GraphNetwork`], validating every constant
    /// through the M1 builders' exact-constant contract (plan §2.3). Builder
    /// rejections propagate unchanged.
    ///
    /// # Errors
    /// [`GroundTruthError::InvalidSidecar`] for structural problems (missing
    /// or ambiguous `builder`/`compose`, params of the wrong shape, wrong
    /// nested `format`); the M1 builder errors for constant-contract
    /// violations.
    pub fn build(&self) -> Result<GraphNetwork> {
        if let Some(format) = self.format.as_deref() {
            if format != GT_FORMAT {
                return Err(GroundTruthError::InvalidSidecar(format!(
                    "unsupported format `{format}` (expected `{GT_FORMAT}`)"
                )));
            }
        }
        let base = match (&self.builder, &self.compose) {
            (Some(builder), None) => {
                let params = self.params.as_ref().ok_or_else(|| {
                    GroundTruthError::InvalidSidecar(format!(
                        "builder `{builder:?}` requires a `params` object"
                    ))
                })?;
                build_primitive(*builder, params)?
            }
            (None, Some(compose)) => {
                if self.params.is_some() {
                    return Err(GroundTruthError::InvalidSidecar(
                        "`params` is only valid together with `builder`".to_string(),
                    ));
                }
                if compose.parts.is_empty() {
                    return Err(GroundTruthError::InvalidSidecar(
                        "`compose.parts` must contain at least one spec".to_string(),
                    ));
                }
                let parts: Vec<GraphNetwork> = compose
                    .parts
                    .iter()
                    .map(GroundTruthSpec::build)
                    .collect::<Result<_>>()?;
                match compose.op {
                    ComposeOp::Min => min_of(&parts)?,
                    ComposeOp::Max => max_of(&parts)?,
                }
            }
            (Some(_), Some(_)) => {
                return Err(GroundTruthError::InvalidSidecar(
                    "`builder` and `compose` are mutually exclusive".to_string(),
                ));
            }
            (None, None) => {
                return Err(GroundTruthError::InvalidSidecar(
                    "spec needs exactly one of `builder` or `compose`".to_string(),
                ));
            }
        };
        match &self.pose {
            Some(p) => with_pose(&base, &Pose::new(p.linear, p.translation)?),
            None => Ok(base),
        }
    }

    /// Evaluate the EXACT rational reference residual of this spec at `x`,
    /// rounded once (correctly) to f64 on return — the `ny gt eval` oracle.
    ///
    /// The spec is first validated via [`GroundTruthSpec::build`] (so the
    /// reference and the graph always denote the same exact function); the
    /// evaluation itself mirrors [`crate::reference`] in exact
    /// arbitrary-precision rationals, with poses applied as exact affine maps
    /// and `min`/`max` as exact rational comparisons.
    ///
    /// # Errors
    /// As [`GroundTruthSpec::build`]; additionally rejects a non-finite `x`
    /// via [`GroundTruthError::NonFiniteParameter`].
    pub fn reference_eval(&self, x: [f64; 3]) -> Result<f64> {
        self.build()?;
        for (i, v) in x.iter().enumerate() {
            if !v.is_finite() {
                return Err(GroundTruthError::NonFiniteParameter {
                    name: format!("x[{i}]"),
                    value: *v,
                });
            }
        }
        let xq = reference::rat3("x", x);
        let value = self.reference_rational(&xq)?;
        value.to_f64().ok_or_else(|| {
            GroundTruthError::InvalidSidecar("reference value exceeds f64 range".to_string())
        })
    }

    /// Exact rational reference evaluation (assumes the spec already built OK).
    fn reference_rational(&self, x: &[BigRational; 3]) -> Result<BigRational> {
        // Apply the pose FIRST: the graph computes g(Ax + b).
        let posed;
        let x = match &self.pose {
            Some(p) => {
                let a: Vec<[BigRational; 3]> = p
                    .linear
                    .iter()
                    .enumerate()
                    .map(|(i, row)| reference::rat3(&format!("pose.linear[{i}]"), *row))
                    .collect();
                let b = reference::rat3("pose.translation", p.translation);
                posed = [
                    &a[0][0] * &x[0] + &a[0][1] * &x[1] + &a[0][2] * &x[2] + &b[0],
                    &a[1][0] * &x[0] + &a[1][1] * &x[1] + &a[1][2] * &x[2] + &b[1],
                    &a[2][0] * &x[0] + &a[2][1] * &x[1] + &a[2][2] * &x[2] + &b[2],
                ];
                &posed
            }
            None => x,
        };
        match (&self.builder, &self.compose) {
            (Some(builder), None) => {
                let params = self.params.as_ref().ok_or_else(|| {
                    GroundTruthError::InvalidSidecar("missing `params`".to_string())
                })?;
                primitive_reference(*builder, params, x)
            }
            (None, Some(compose)) => {
                let mut acc: Option<BigRational> = None;
                for part in &compose.parts {
                    let v = part.reference_rational(x)?;
                    acc = Some(match acc {
                        None => v,
                        Some(a) => match compose.op {
                            ComposeOp::Min => a.min(v),
                            ComposeOp::Max => a.max(v),
                        },
                    });
                }
                acc.ok_or_else(|| {
                    GroundTruthError::InvalidSidecar("empty `compose.parts`".to_string())
                })
            }
            _ => Err(GroundTruthError::InvalidSidecar(
                "spec needs exactly one of `builder` or `compose`".to_string(),
            )),
        }
    }
}

fn typed_params<T: serde::de::DeserializeOwned>(
    builder: BuilderKind,
    params: &serde_json::Value,
) -> Result<T> {
    serde_json::from_value(params.clone()).map_err(|e| {
        GroundTruthError::InvalidSidecar(format!("bad params for builder `{builder:?}`: {e}"))
    })
}

fn build_primitive(builder: BuilderKind, params: &serde_json::Value) -> Result<GraphNetwork> {
    match builder {
        BuilderKind::Plane => {
            let p: PlaneParams = typed_params(builder, params)?;
            signed_plane_distance(p.normal, p.offset)
        }
        BuilderKind::Sphere => {
            let p: SphereParams = typed_params(builder, params)?;
            sphere_residual(p.center, p.radius)
        }
        BuilderKind::Cylinder => {
            let p: CylinderParams = typed_params(builder, params)?;
            cylinder_residual(p.axis, p.point, p.radius)
        }
        BuilderKind::Cone => {
            let p: ConeParams = typed_params(builder, params)?;
            cone_residual(p.axis, p.apex, p.cos_half_angle_sq)
        }
        BuilderKind::Torus => {
            let p: TorusParams = typed_params(builder, params)?;
            torus_residual(p.axis, p.center, p.major_radius, p.minor_radius)
        }
    }
}

fn primitive_reference(
    builder: BuilderKind,
    params: &serde_json::Value,
    x: &[BigRational; 3],
) -> Result<BigRational> {
    match builder {
        BuilderKind::Plane => {
            let p: PlaneParams = typed_params(builder, params)?;
            Ok(reference::signed_plane_distance_rat(
                &reference::rat3("normal", p.normal),
                &reference::rat("offset", p.offset),
                x,
            ))
        }
        BuilderKind::Sphere => {
            let p: SphereParams = typed_params(builder, params)?;
            Ok(reference::sphere_residual_rat(
                &reference::rat3("center", p.center),
                &reference::rat("radius", p.radius),
                x,
            ))
        }
        BuilderKind::Cylinder => {
            let p: CylinderParams = typed_params(builder, params)?;
            Ok(reference::cylinder_residual_rat(
                &reference::rat3("axis", p.axis),
                &reference::rat3("point", p.point),
                &reference::rat("radius", p.radius),
                x,
            ))
        }
        BuilderKind::Cone => {
            let p: ConeParams = typed_params(builder, params)?;
            Ok(reference::cone_residual_rat(
                &reference::rat3("axis", p.axis),
                &reference::rat3("apex", p.apex),
                &reference::rat("cos_half_angle_sq", p.cos_half_angle_sq),
                x,
            ))
        }
        BuilderKind::Torus => {
            let p: TorusParams = typed_params(builder, params)?;
            Ok(reference::torus_residual_rat(
                &reference::rat3("axis", p.axis),
                &reference::rat3("center", p.center),
                &reference::rat("major_radius", p.major_radius),
                &reference::rat("minor_radius", p.minor_radius),
                x,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference as reff;
    use ndarray::Array1;
    use ny_tensor::BoundedTensor;

    fn enclosure_at(g: &GraphNetwork, x: [f64; 3]) -> (f32, f32) {
        let arr = Array1::from(vec![x[0] as f32, x[1] as f32, x[2] as f32]).into_dyn();
        let t = BoundedTensor::new(arr.clone(), arr).unwrap();
        let out = g.propagate_ibp(&t).unwrap();
        (out.lower()[0], out.upper()[0])
    }

    #[test]
    fn round_trips_every_primitive_through_json() {
        let specs = [
            GroundTruthSpec::plane([0.0, 0.0, 1.0], -0.25),
            GroundTruthSpec::sphere([1.0, -2.0, 0.5], 1.5),
            GroundTruthSpec::cylinder([0.0, 0.0, 1.0], [1.0, -2.0, 0.5], 1.5),
            GroundTruthSpec::cone([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 0.75),
            GroundTruthSpec::torus([0.0, 0.0, 1.0], [0.5, 0.5, 0.0], 2.0, 0.5),
        ];
        for spec in specs {
            let json = spec.to_json_string().unwrap();
            let parsed = GroundTruthSpec::from_json_str(&json).unwrap();
            assert_eq!(parsed, spec, "round trip changed the spec:\n{json}");
            assert_eq!(parsed.to_json_string().unwrap(), json, "JSON not stable");
            parsed.build().expect("round-tripped spec builds");
        }
    }

    #[test]
    fn round_trips_pose_and_compose() {
        let spec = GroundTruthSpec::composed(
            ComposeOp::Min,
            vec![
                GroundTruthSpec::sphere([0.0, 0.0, 0.0], 1.0),
                GroundTruthSpec::plane([0.0, 0.0, 1.0], -0.25).with_pose(PoseSpec {
                    linear: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    translation: [0.5, 0.0, 0.0],
                }),
            ],
        );
        let json = spec.to_json_string().unwrap();
        let parsed = GroundTruthSpec::from_json_str(&json).unwrap();
        assert_eq!(parsed, spec);
        parsed.build().expect("composed spec builds");
    }

    #[test]
    fn loader_validates_through_the_builders() {
        // 0.1 is not f64 -> f32 exact: the M1 contract rejection propagates.
        let json = r#"{
            "format": "gt/1",
            "builder": "sphere",
            "params": { "center": [0.0, 0.0, 0.0], "radius": 0.1 }
        }"#;
        let spec = GroundTruthSpec::from_json_str(json).unwrap();
        assert!(matches!(
            spec.build(),
            Err(GroundTruthError::InexactParameter { .. })
        ));
        // A non-unit axis is rejected by the builder, not silently rounded.
        let bad_axis = GroundTruthSpec::cylinder([1.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1.5);
        assert!(matches!(
            bad_axis.build(),
            Err(GroundTruthError::AxisNotUnit { .. })
        ));
    }

    #[test]
    fn rejects_malformed_sidecars() {
        // Wrong format tag.
        assert!(matches!(
            GroundTruthSpec::from_json_str(r#"{"format": "gt/999", "builder": "plane"}"#),
            Err(GroundTruthError::InvalidSidecar(_))
        ));
        // Missing format at the root.
        assert!(matches!(
            GroundTruthSpec::from_json_str(
                r#"{"builder": "plane", "params": {"normal": [0,0,1], "offset": 0}}"#
            ),
            Err(GroundTruthError::InvalidSidecar(_))
        ));
        // Unknown field.
        assert!(matches!(
            GroundTruthSpec::from_json_str(r#"{"format": "gt/1", "wat": 1}"#),
            Err(GroundTruthError::InvalidSidecar(_))
        ));
        // Neither builder nor compose.
        let empty = GroundTruthSpec::from_json_str(r#"{"format": "gt/1"}"#).unwrap();
        assert!(matches!(
            empty.build(),
            Err(GroundTruthError::InvalidSidecar(_))
        ));
        // Both builder and compose.
        let both = GroundTruthSpec::from_json_str(
            r#"{
                "format": "gt/1",
                "builder": "plane",
                "params": {"normal": [0.0, 0.0, 1.0], "offset": 0.0},
                "compose": {"op": "min", "parts": [
                    {"builder": "plane", "params": {"normal": [0.0, 0.0, 1.0], "offset": 0.0}}
                ]}
            }"#,
        )
        .unwrap();
        assert!(matches!(
            both.build(),
            Err(GroundTruthError::InvalidSidecar(_))
        ));
        // Wrong params shape for the named builder.
        let wrong = GroundTruthSpec::from_json_str(
            r#"{"format": "gt/1", "builder": "sphere",
                "params": {"axis": [0.0, 0.0, 1.0], "point": [0.0, 0.0, 0.0], "radius": 1.5}}"#,
        )
        .unwrap();
        assert!(matches!(
            wrong.build(),
            Err(GroundTruthError::InvalidSidecar(_))
        ));
        // Empty compose parts.
        let empty_parts = GroundTruthSpec::from_json_str(
            r#"{"format": "gt/1", "compose": {"op": "max", "parts": []}}"#,
        )
        .unwrap();
        assert!(matches!(
            empty_parts.build(),
            Err(GroundTruthError::InvalidSidecar(_))
        ));
    }

    #[test]
    fn reference_eval_matches_module_oracles_and_graph() {
        let x = [2.5, -1.0, 0.25];
        let spec = GroundTruthSpec::cylinder([0.0, 0.0, 1.0], [1.0, -2.0, 0.5], 1.5);
        let want = reff::cylinder_residual([0.0, 0.0, 1.0], [1.0, -2.0, 0.5], 1.5, x);
        let got = spec.reference_eval(x).unwrap();
        assert_eq!(got, want, "sidecar reference must equal the M1 oracle");
        // The graph's zero-width enclosure contains the exact reference.
        let g = spec.build().unwrap();
        let (lo, hi) = enclosure_at(&g, x);
        assert!(
            f64::from(lo) <= want && want <= f64::from(hi),
            "enclosure [{lo}, {hi}] must contain {want}"
        );
    }

    #[test]
    fn reference_eval_handles_pose_and_min_max() {
        // Pose: swap x/y then translate; residual of the plane z - 0.25... the
        // plane only reads x2, so pose with a swap of x0/x1 leaves it alone —
        // use a plane on x0 instead to see the swap.
        let posed = GroundTruthSpec::plane([1.0, 0.0, 0.0], 0.0).with_pose(PoseSpec {
            linear: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.5, 0.0, 0.0],
        });
        // g(x) = plane residual at (x1 + 1/2, x0, x2) = x1 + 1/2.
        let x = [3.0, -0.75, 2.0];
        assert_eq!(posed.reference_eval(x).unwrap(), -0.25);
        let g = posed.build().unwrap();
        let (lo, hi) = enclosure_at(&g, x);
        assert!(f64::from(lo) <= -0.25 && -0.25 <= f64::from(hi));

        // min/max of two spheres at a point where the values differ.
        let a = GroundTruthSpec::sphere([0.0, 0.0, 0.0], 1.0);
        let b = GroundTruthSpec::sphere([2.0, 0.0, 0.0], 1.0);
        let p = [0.5, 0.0, 0.0];
        let va = reff::sphere_residual([0.0, 0.0, 0.0], 1.0, p);
        let vb = reff::sphere_residual([2.0, 0.0, 0.0], 1.0, p);
        let min_spec = GroundTruthSpec::composed(ComposeOp::Min, vec![a.clone(), b.clone()]);
        let max_spec = GroundTruthSpec::composed(ComposeOp::Max, vec![a, b]);
        assert_eq!(min_spec.reference_eval(p).unwrap(), va.min(vb));
        assert_eq!(max_spec.reference_eval(p).unwrap(), va.max(vb));
    }

    #[test]
    fn load_reads_files_and_reports_io_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cyl.gt.json");
        let spec = GroundTruthSpec::cylinder([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 3.0);
        std::fs::write(&path, spec.to_json_string().unwrap()).unwrap();
        let loaded = GroundTruthSpec::load(&path).unwrap();
        assert_eq!(loaded, spec);
        assert!(matches!(
            GroundTruthSpec::load(&dir.path().join("missing.gt.json")),
            Err(GroundTruthError::SidecarIo { .. })
        ));
    }
}
