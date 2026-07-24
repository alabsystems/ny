// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Rich output-property constraints and the encoder that turns them into a
//! verifiable *margin* network.
//!
//! A [`VerificationSpec`] can carry [`OutputConstraint`]s beyond the legacy
//! per-output interval bounds: halfspace ([`OutputConstraint::Linear`]) and
//! robustness ([`OutputConstraint::ArgmaxMargin`]) properties. This module
//! reduces each such constraint to a single sound check of the form
//! `margin > 0`:
//!
//! - `Linear { a, b, Le }`  (`a·y <= b`) => margin `m = b - a·y`.
//! - `Linear { a, b, Ge }`  (`a·y >= b`) => margin `m = a·y - b`.
//! - `ArgmaxMargin { class }` => one margin per competing class `j != class`,
//!   `m_j = y[class] - y[j]`; the property holds iff *every* `m_j > 0`.
//!
//! The reduction appends a final affine ([`ny_propagate::layers::LinearLayer`])
//! node to the network whose single (or per-competitor) output *is* the margin,
//! mirroring the difference-network technique used by
//! [`ny_propagate::build_difference_network`]. Verification then proves the
//! margin's lower bound is strictly positive.
//!
//! The encoder is **fail-closed**: any constraint that cannot be soundly
//! reduced (e.g. an out-of-range argmax class) yields an error from
//! [`augment_for_constraint`], and [`verify_with_constraints`] turns
//! unencodable / inconclusive constraints into `Unknown` rather than silently
//! reporting success.

use ny_core::{
    Bound, ConstraintKind, NyError, OutputConstraint, Result, SoundnessProvenance, UnknownReason,
    VerificationResult, VerificationSpec,
};
use ny_propagate::layers::{Layer, LinearLayer};
use ny_propagate::{GraphNetwork, GraphNode, Verifier};

/// Name of the appended margin node in an augmented network.
const MARGIN_NODE: &str = "_constraint_margin";

/// A flat affine margin map: row-major `weight` of shape
/// `(out_features, in_features)` plus a `bias` of length `out_features`.
struct MarginMap {
    weight: Vec<f32>,
    bias: Vec<f32>,
    out_features: usize,
    in_features: usize,
}

/// Determine the number of outputs the network produces over `spec`'s input
/// region by running a cheap IBP forward pass.
///
/// Mirrors the approach in [`ny_propagate::verify_equivalence`], which uses an
/// IBP pass to discover the (otherwise statically-unknown) output width.
fn output_width(net: &GraphNetwork, spec: &VerificationSpec) -> Result<usize> {
    let input = Verifier::bounds_to_tensor(spec.input_bounds(), spec.input_shape())?;
    let ibp = net.propagate_ibp(&input)?;
    Ok(ibp.lower().len())
}

/// Build a network whose output is the *margin(s)* `m` for `constraint`, such
/// that the property holds iff every margin output is strictly positive
/// (`m > 0`).
///
/// The returned network shares `net`'s input and reuses every existing node;
/// a single final affine [`ny_propagate::layers::LinearLayer`] node consumes
/// `net`'s output and emits the margin vector.
///
/// # Margin encoding
/// - [`OutputConstraint::Linear`] with [`ConstraintKind::Le`] (`a·y <= b`):
///   margin `m = b - a·y` (one output).
/// - [`OutputConstraint::Linear`] with [`ConstraintKind::Ge`] (`a·y >= b`):
///   margin `m = a·y - b` (one output).
/// - [`OutputConstraint::ArgmaxMargin`] `{ class }`: one margin per competing
///   class `j != class`, row `e_class - e_j`, so `m_j = y[class] - y[j]`.
///
/// # Errors
/// - [`NyError::InvalidSpec`] if `net` has no output node set.
/// - [`NyError::InvalidSpec`] for [`OutputConstraint::Bounds`], which is a
///   per-output interval property that is *not* a single-margin check; verify
///   it through the legacy `output_bounds` path (see [`verify_with_constraints`]).
/// - [`NyError::ShapeMismatch`] / [`NyError::InvalidSpec`] if an argmax `class`
///   is out of range or the output width cannot be determined.
///
/// For [`OutputConstraint::ArgmaxMargin`] the output width is derived from the
/// network's output node when it is a `Linear` layer; otherwise an error is
/// returned. [`verify_with_constraints`] handles the general case via an IBP
/// forward pass.
pub fn augment_for_constraint(
    net: &GraphNetwork,
    constraint: &OutputConstraint,
) -> Result<GraphNetwork> {
    if net.output_name().is_empty() {
        return Err(NyError::InvalidSpec(
            "augment_for_constraint: network has no output node set".to_string(),
        ));
    }

    // Validate structural shape up front (non-empty coeffs, finite bias, etc.).
    constraint.validate()?;

    let map = match constraint {
        OutputConstraint::Bounds(_) => {
            return Err(NyError::InvalidSpec(
                "augment_for_constraint: OutputConstraint::Bounds is a per-output interval \
                 property, not a single-margin check; verify it via the legacy output_bounds \
                 path (verify_with_constraints handles this)."
                    .to_string(),
            ));
        }
        OutputConstraint::Linear { coeffs, bias, kind } => margin_for_linear(coeffs, *bias, *kind),
        OutputConstraint::ArgmaxMargin { class } => {
            let width = static_output_width(net).ok_or_else(|| {
                NyError::InvalidSpec(
                    "augment_for_constraint: cannot determine output width for ArgmaxMargin; \
                     the network's output node must be a Linear layer (use \
                     verify_with_constraints, which derives the width via an IBP forward pass)."
                        .to_string(),
                )
            })?;
            build_argmax_margin(*class, width)?
        }
    };

    append_margin_layer(net, &map)
}

/// Build the flat affine margin map for a `Linear` constraint.
///
/// `Le` (`a·y <= b`): `m = b - a·y` => weight row `-a`, bias `b`.
/// `Ge` (`a·y >= b`): `m = a·y - b` => weight row `a`, bias `-b`.
fn margin_for_linear(coeffs: &[f32], bias: f32, kind: ConstraintKind) -> MarginMap {
    let (row, b): (Vec<f32>, f32) = match kind {
        ConstraintKind::Le => (coeffs.iter().map(|c| -c).collect(), bias),
        ConstraintKind::Ge => (coeffs.to_vec(), -bias),
    };
    let in_features = row.len();
    MarginMap {
        weight: row,
        bias: vec![b],
        out_features: 1,
        in_features,
    }
}

/// Build the argmax margin map: one row `e_class - e_j` per competing class
/// `j != class`, with zero bias.
///
/// Row `k` corresponds to competitor `j` (the `k`-th index `!= class`) and is
/// `e_class - e_j`, so margin `m_k = y[class] - y[j]`.
fn build_argmax_margin(class: usize, width: usize) -> Result<MarginMap> {
    if class >= width {
        return Err(NyError::ShapeMismatch {
            expected: vec![width],
            got: vec![class + 1],
        });
    }
    if width < 2 {
        return Err(NyError::InvalidSpec(format!(
            "ArgmaxMargin requires at least 2 outputs to have a competing class, got {width}"
        )));
    }
    let num_competitors = width - 1;
    // Row-major (num_competitors, width) matrix.
    let mut weight = vec![0.0f32; num_competitors * width];
    let mut row = 0usize;
    for j in 0..width {
        if j == class {
            continue;
        }
        weight[row * width + class] = 1.0;
        weight[row * width + j] = -1.0;
        row += 1;
    }
    Ok(MarginMap {
        weight,
        bias: vec![0.0; num_competitors],
        out_features: num_competitors,
        in_features: width,
    })
}

/// Best-effort static output width: if the output node is a `Linear` layer,
/// its `out_features` is the width. Returns `None` otherwise.
fn static_output_width(net: &GraphNetwork) -> Option<usize> {
    let node = net.node(net.output_name())?;
    match node.layer() {
        Layer::Linear(lin) => Some(lin.out_features()),
        _ => None,
    }
}

/// Deep-copy a [`GraphNetwork`] node-by-node, preserving insertion order, names,
/// inputs, and the output target.
///
/// `GraphNetwork` is not `Clone` in the public surface, so we rebuild it via the
/// public node API (the same technique `build_difference_network` uses).
fn clone_network(net: &GraphNetwork) -> Result<GraphNetwork> {
    let mut out = GraphNetwork::new();
    for name in net.node_names() {
        let node = net
            .node(name)
            .ok_or_else(|| NyError::InternalError(format!("node '{name}' missing during clone")))?;
        let copied = GraphNode::new(
            node.name().to_string(),
            node.layer().clone(),
            node.inputs().to_vec(),
        );
        out.try_add_node(copied)?;
    }
    out.set_output(net.output_name());
    Ok(out)
}

/// Append a margin [`ny_propagate::layers::LinearLayer`] (built from a flat
/// [`MarginMap`]) onto a copy of `net`, retargeting the output to the new margin
/// node.
fn append_margin_layer(net: &GraphNetwork, map: &MarginMap) -> Result<GraphNetwork> {
    let margin_layer = LinearLayer::from_flat(
        map.weight.clone(),
        map.out_features,
        map.in_features,
        Some(map.bias.clone()),
    )?;
    let prev_output = net.output_name().to_string();
    let mut augmented = clone_network(net)?;
    augmented.try_add_node(GraphNode::new(
        MARGIN_NODE,
        Layer::Linear(margin_layer),
        vec![prev_output],
    ))?;
    augmented.set_output(MARGIN_NODE);
    Ok(augmented)
}

/// Verify a network against every [`OutputConstraint`] carried by `spec`,
/// combining the verdicts conjunctively (all constraints must hold).
///
/// Semantics:
/// - If `spec` carries no [`OutputConstraint`]s, this delegates to the legacy
///   per-output interval check ([`Verifier::verify_graph`] against
///   `spec.output_bounds()`), preserving existing behavior exactly.
/// - [`OutputConstraint::Bounds`] is verified through the legacy interval path.
/// - [`OutputConstraint::Linear`] / [`OutputConstraint::ArgmaxMargin`] are
///   reduced via the margin encoding and proven by checking that every margin
///   output's lower bound is **strictly positive**.
///
/// The result is fail-closed: the combined verdict is `Verified` only if *all*
/// constraints are individually verified; any inconclusive, unencodable, or
/// violated constraint yields a non-`Verified` result. Soundness provenance is
/// combined across the per-constraint runs via [`SoundnessProvenance::combine`].
///
/// # Errors
/// Propagates hard errors (invalid spec, propagation failure) from the
/// underlying verifier. Inconclusive or unencodable constraints do *not* error;
/// they downgrade the verdict to `Unknown`.
pub fn verify_with_constraints(
    verifier: &Verifier,
    net: &GraphNetwork,
    spec: &VerificationSpec,
) -> Result<VerificationResult> {
    // Legacy path: no rich constraints — verify against output_bounds directly.
    if spec.output_constraints().is_empty() {
        return verifier.verify_graph(net, spec);
    }

    let mut combined_provenance = SoundnessProvenance::sound();
    let mut first_non_verified: Option<VerificationResult> = None;

    for constraint in spec.output_constraints() {
        let result = verify_single_constraint(verifier, net, spec, constraint)?;
        combined_provenance = combined_provenance.combine(result.provenance());

        if !result.is_verified() && first_non_verified.is_none() {
            first_non_verified = Some(result);
        }
    }

    if let Some(non_verified) = first_non_verified {
        // Reflect the combined provenance over all constraints inspected so far.
        return Ok(non_verified.with_provenance(combined_provenance));
    }

    // All constraints verified: report the conjunction as Verified with combined
    // provenance. No single output-bounds vector is meaningful for a conjunction
    // of heterogeneous constraints, so we report an empty (no-op) bounds vector.
    Ok(VerificationResult::Verified {
        provenance: combined_provenance,
        output_bounds: Vec::new(),
        proof: None,
        actual_method: None,
    })
}

/// Verify a single output constraint, returning a per-constraint verdict.
fn verify_single_constraint(
    verifier: &Verifier,
    net: &GraphNetwork,
    spec: &VerificationSpec,
    constraint: &OutputConstraint,
) -> Result<VerificationResult> {
    match constraint {
        OutputConstraint::Bounds(bounds) => {
            // Legacy interval property: verify net's outputs against `bounds`.
            let bounds_spec = VerificationSpec::from_parts(
                spec.input_bounds().to_vec(),
                bounds.clone(),
                spec.timeout_ms(),
                spec.input_shape().map(<[usize]>::to_vec),
            )?;
            verifier.verify_graph(net, &bounds_spec)
        }
        OutputConstraint::Linear { coeffs, bias, kind } => {
            let map = margin_for_linear(coeffs, *bias, *kind);
            let augmented = append_margin_layer(net, &map)?;
            verify_margin(verifier, net, spec, augmented)
        }
        OutputConstraint::ArgmaxMargin { class } => {
            // Derive the output width via IBP (more general than the static check
            // used by augment_for_constraint), then build the margin network.
            let width = output_width(net, spec)?;
            let map = match build_argmax_margin(*class, width) {
                Ok(m) => m,
                // Unencodable (e.g. class out of range): fail closed to Unknown.
                Err(_) => {
                    return Ok(unknown_unencodable(
                        net,
                        spec,
                        format!(
                            "ArgmaxMargin class {class} is not encodable for output width {width}"
                        ),
                    ))
                }
            };
            let augmented = append_margin_layer(net, &map)?;
            verify_margin(verifier, net, spec, augmented)
        }
    }
}

/// Verify that every output of an augmented *margin* network is strictly
/// positive over `spec`'s input region.
///
/// Builds a spec requiring each margin output to lie in `[0, +inf)` and runs the
/// verifier, then strengthens `Verified` to require *strict* positivity (a
/// margin whose certified lower bound is exactly `0` does not establish the
/// strict property and is downgraded to `Unknown`).
fn verify_margin(
    verifier: &Verifier,
    net: &GraphNetwork,
    spec: &VerificationSpec,
    augmented: GraphNetwork,
) -> Result<VerificationResult> {
    let width = output_width(&augmented, spec)?;
    if width == 0 {
        return Ok(unknown_unencodable(
            net,
            spec,
            "margin network produced zero outputs".to_string(),
        ));
    }

    // Require every margin output to be >= 0; we post-check for strict > 0.
    let margin_bounds: Vec<Bound> = (0..width)
        .map(|_| Bound::new_allow_infinite(0.0, f32::INFINITY))
        .collect();
    let margin_spec = VerificationSpec::from_parts(
        spec.input_bounds().to_vec(),
        margin_bounds,
        spec.timeout_ms(),
        spec.input_shape().map(<[usize]>::to_vec),
    )?;

    let result = verifier.verify_graph(&augmented, &margin_spec)?;

    // Strengthen to strict positivity: a Verified result whose certified bounds
    // include a non-positive lower bound does not prove `margin > 0`.
    if let VerificationResult::Verified {
        provenance,
        output_bounds,
        ..
    } = &result
    {
        let all_strictly_positive = output_bounds.iter().all(|b| b.lower() > 0.0);
        if !all_strictly_positive {
            let worst = output_bounds
                .iter()
                .map(|b| b.lower())
                .fold(f32::INFINITY, f32::min);
            return Ok(VerificationResult::Unknown {
                provenance: provenance.clone(),
                bounds: output_bounds.clone(),
                reason: UnknownReason::BoundsTooLoose {
                    gap: if worst <= 0.0 { Some(-worst) } else { None },
                },
                actual_method: None,
            });
        }
    }

    Ok(result)
}

/// Build an `Unknown` verdict for a constraint that cannot be soundly encoded.
///
/// Fail-closed: never report success for an unencodable constraint. The returned
/// `bounds` are the network's IBP output bounds when available (purely
/// informational), else empty.
fn unknown_unencodable(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    message: String,
) -> VerificationResult {
    let bounds = Verifier::bounds_to_tensor(spec.input_bounds(), spec.input_shape())
        .and_then(|input| net.propagate_ibp(&input))
        .map(|ibp| {
            ibp.lower()
                .iter()
                .zip(ibp.upper().iter())
                .map(|(&l, &u)| Bound::new_allow_infinite(l, u))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    VerificationResult::Unknown {
        provenance: SoundnessProvenance::sound(),
        bounds,
        reason: UnknownReason::Other { message },
        actual_method: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::{PropagationConfig, PropagationMethod};

    /// Build a tiny single-output-node network: y = W x + b with the output
    /// node being a Linear layer. `weight` is row-major (out, in), `bias` len out.
    fn linear_network(
        weight: Vec<f32>,
        out_features: usize,
        in_features: usize,
        bias: Vec<f32>,
    ) -> GraphNetwork {
        let layer = Layer::Linear(
            LinearLayer::from_flat(weight, out_features, in_features, Some(bias)).unwrap(),
        );
        let mut net = GraphNetwork::new();
        net.add_node(GraphNode::from_input("out", layer));
        net.set_output("out");
        net
    }

    fn ibp_config() -> PropagationConfig {
        PropagationConfig {
            method: PropagationMethod::Ibp,
            ..Default::default()
        }
    }

    fn spec_for(width_in: usize) -> VerificationSpec {
        let input = (0..width_in).map(|_| Bound::new(0.0, 1.0)).collect();
        // output_bounds is required to be non-empty but is unused for constraint
        // verification; supply a trivial wide interval.
        VerificationSpec::from_parts(
            input,
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn margin_for_linear_le_negates_coeffs() {
        // Constraint 1*y <= 5  => margin = 5 - y => weight [-1], bias 5.
        let m = margin_for_linear(&[1.0], 5.0, ConstraintKind::Le);
        assert_eq!(m.weight, vec![-1.0]);
        assert_eq!(m.bias, vec![5.0]);
        assert_eq!((m.out_features, m.in_features), (1, 1));
    }

    #[test]
    fn margin_for_linear_ge_keeps_coeffs_negates_bias() {
        // Constraint 2*y >= 3  => margin = 2*y - 3 => weight [2], bias -3.
        let m = margin_for_linear(&[2.0], 3.0, ConstraintKind::Ge);
        assert_eq!(m.weight, vec![2.0]);
        assert_eq!(m.bias, vec![-3.0]);
    }

    #[test]
    fn build_argmax_margin_builds_difference_rows() {
        // class 0 vs width 3 => 2 rows: e0-e1, e0-e2 (row-major 2x3).
        let m = build_argmax_margin(0, 3).unwrap();
        assert_eq!((m.out_features, m.in_features), (2, 3));
        // Row 0: [1, -1, 0]; Row 1: [1, 0, -1].
        assert_eq!(m.weight, vec![1.0, -1.0, 0.0, 1.0, 0.0, -1.0]);
        assert_eq!(m.bias, vec![0.0, 0.0]);
    }

    #[test]
    fn build_argmax_margin_middle_class() {
        // class 1 vs width 3 => rows e1-e0, e1-e2.
        let m = build_argmax_margin(1, 3).unwrap();
        assert_eq!(m.weight, vec![-1.0, 1.0, 0.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn build_argmax_margin_rejects_out_of_range_class() {
        assert!(build_argmax_margin(3, 3).is_err());
        assert!(build_argmax_margin(0, 1).is_err());
        assert!(build_argmax_margin(0, 3).is_ok());
    }

    #[test]
    fn augment_appends_margin_node_and_retargets_output() {
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let c = OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: 5.0,
            kind: ConstraintKind::Le,
        };
        let aug = augment_for_constraint(&net, &c).unwrap();
        assert_eq!(aug.output_name(), MARGIN_NODE);
        // Original "out" node is preserved and feeds the margin node.
        assert!(aug.node("out").is_some());
        let margin = aug.node(MARGIN_NODE).unwrap();
        assert_eq!(margin.inputs(), &["out".to_string()]);
        if let Layer::Linear(lin) = margin.layer() {
            assert_eq!(lin.out_features(), 1);
            assert_eq!(lin.in_features(), 1);
        } else {
            panic!("expected Linear margin layer");
        }
    }

    #[test]
    fn augment_argmax_has_competitor_outputs() {
        let net = linear_network(vec![1.0, 1.0, 1.0], 3, 1, vec![0.0, 0.0, 0.0]);
        let aug =
            augment_for_constraint(&net, &OutputConstraint::ArgmaxMargin { class: 0 }).unwrap();
        if let Layer::Linear(lin) = aug.node(MARGIN_NODE).unwrap().layer() {
            assert_eq!(lin.out_features(), 2); // 2 competitors
            assert_eq!(lin.in_features(), 3); // consumes 3-wide output
        } else {
            panic!("expected Linear margin layer");
        }
    }

    #[test]
    fn augment_bounds_is_rejected() {
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let c = OutputConstraint::Bounds(vec![Bound::new(0.0, 1.0)]);
        let err = augment_for_constraint(&net, &c).unwrap_err();
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    #[test]
    fn verify_linear_le_holds_for_bounded_output() {
        // y = x over x in [0,1] => y in [0,1]. Constraint 1*y <= 5 holds
        // (margin = 5 - y in [4,5] > 0).
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![OutputConstraint::Linear {
                coeffs: vec![1.0],
                bias: 5.0,
                kind: ConstraintKind::Le,
            }])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(result.is_verified(), "expected Verified, got {result:?}");
    }

    #[test]
    fn verify_linear_le_fails_when_violated() {
        // y = x over [0,1]. Constraint 1*y <= 0.5: margin = 0.5 - y in [-0.5, 0.5],
        // not strictly positive => Unknown (fail-closed, never Verified).
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![OutputConstraint::Linear {
                coeffs: vec![1.0],
                bias: 0.5,
                kind: ConstraintKind::Le,
            }])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(
            !result.is_verified(),
            "must not verify a violated constraint"
        );
    }

    #[test]
    fn verify_linear_ge_holds() {
        // y = x in [0,1]. Constraint 1*y >= -5 holds (margin = y + 5 in [5,6] > 0).
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![OutputConstraint::Linear {
                coeffs: vec![1.0],
                bias: -5.0,
                kind: ConstraintKind::Ge,
            }])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(result.is_verified(), "expected Verified, got {result:?}");
    }

    #[test]
    fn verify_argmax_holds_when_class_dominates() {
        // y0 = x + 10 in [10,11]; y1 = x in [0,1]; y2 = -x in [-1,0].
        // Class 0 strictly dominates.
        let net = linear_network(vec![1.0, 1.0, -1.0], 3, 1, vec![10.0, 0.0, 0.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(result.is_verified(), "expected Verified, got {result:?}");
    }

    #[test]
    fn verify_argmax_fails_when_class_does_not_dominate() {
        // y0 = x in [0,1]; y1 = x + 10 in [10,11]. Class 0 does NOT dominate.
        let net = linear_network(vec![1.0, 1.0], 2, 1, vec![0.0, 10.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(
            !result.is_verified(),
            "class 0 does not dominate; must not verify"
        );
    }

    #[test]
    fn verify_argmax_out_of_range_class_is_unknown() {
        // 2-output network, class index 5 is out of range => fail-closed Unknown.
        let net = linear_network(vec![1.0, 1.0], 2, 1, vec![0.0, 0.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 5 }])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(matches!(result, VerificationResult::Unknown { .. }));
    }

    #[test]
    fn verify_conjunction_all_must_hold() {
        // Two constraints, both satisfiable: y in [0,1]. c1: y <= 5; c2: y >= -5.
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![
                OutputConstraint::Linear {
                    coeffs: vec![1.0],
                    bias: 5.0,
                    kind: ConstraintKind::Le,
                },
                OutputConstraint::Linear {
                    coeffs: vec![1.0],
                    bias: -5.0,
                    kind: ConstraintKind::Ge,
                },
            ])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(result.is_verified(), "both constraints hold: {result:?}");
    }

    #[test]
    fn verify_conjunction_one_failing_blocks() {
        // c1 holds (y <= 5), c2 fails (y >= 5 but y in [0,1]).
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let spec = spec_for(1)
            .with_output_constraints(vec![
                OutputConstraint::Linear {
                    coeffs: vec![1.0],
                    bias: 5.0,
                    kind: ConstraintKind::Le,
                },
                OutputConstraint::Linear {
                    coeffs: vec![1.0],
                    bias: 5.0,
                    kind: ConstraintKind::Ge,
                },
            ])
            .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(!result.is_verified(), "one failing constraint must block");
    }

    #[test]
    fn verify_no_constraints_uses_legacy_path() {
        // No output constraints: falls back to output_bounds check.
        let net = linear_network(vec![1.0], 1, 1, vec![0.0]);
        let input = vec![Bound::new(0.0, 1.0)];
        let spec = VerificationSpec::from_parts(
            input,
            vec![Bound::new(-1.0, 2.0)], // y in [0,1] is within [-1,2] => Verified
            None,
            None,
        )
        .unwrap();
        let verifier = Verifier::new(ibp_config());
        let result = verify_with_constraints(&verifier, &net, &spec).unwrap();
        assert!(result.is_verified());
    }
}
