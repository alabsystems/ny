// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::boundary::discover_ecapa_composition_boundary;
use super::super::subgraph::extract_single_input_subgraph;
use super::*;
use ny_propagate::LinearBounds;
use ny_tensor::{next_down_f32, next_up_f32};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// CROWN-IBP tightening budget for suffix intermediates in the composed pipeline.
const COMPOSED_SUFFIX_CROWN_IBP_BUDGET_SECS: u64 = 10;

/// Compose two `LinearBounds`: inner maps input -> intermediate,
/// outer maps intermediate -> output.
///
/// Standard CROWN backward composition. Given:
///
///   inner:  m_j in [pla_j * x + plb_j,  pua_j * x + pub_j]
///   outer:  y_i in [ola_i * m + olb_i,  oua_i * m + oub_i]
///
/// For the lower bound on y_i, when ola_{i,j} >= 0 we use the lower bound
/// on m_j, and when ola_{i,j} < 0 we use the upper bound on m_j:
///
///   composed_la[i] = sum_j max(ola_{i,j}, 0) * pla_j + min(ola_{i,j}, 0) * pua_j
///   composed_lb[i] = sum_j max(ola_{i,j}, 0) * plb_j + min(ola_{i,j}, 0) * pub_j + olb_i
///
/// Symmetrically for the upper bound with oua:
///
///   composed_ua[i] = sum_j max(oua_{i,j}, 0) * pua_j + min(oua_{i,j}, 0) * pla_j
///   composed_ub[i] = sum_j max(oua_{i,j}, 0) * pub_j + min(oua_{i,j}, 0) * plb_j + oub_i
///
/// Sound because the positive/negative decomposition correctly selects the
/// tightest bound direction for each coefficient sign.
fn compose_linear_bounds(
    inner: &LinearBounds,
    outer: &LinearBounds,
) -> Result<LinearBounds, String> {
    if outer.num_inputs() != inner.num_outputs() {
        return Err(format!(
            "LinearBounds composition dimension mismatch: \
             outer.num_inputs()={} != inner.num_outputs()={}",
            outer.num_inputs(),
            inner.num_outputs(),
        ));
    }

    let out_dim = outer.num_outputs();
    let mid_dim = outer.num_inputs();
    let in_dim = inner.num_inputs();
    let outer_lower_a = outer.lower_a();
    let outer_upper_a = outer.upper_a();
    let outer_lower_b = outer.lower_b();
    let outer_upper_b = outer.upper_b();
    let inner_lower_a = inner.lower_a();
    let inner_upper_a = inner.upper_a();
    let inner_lower_b = inner.lower_b();
    let inner_upper_b = inner.upper_b();

    let mut composed_la = ndarray::Array2::<f32>::zeros((out_dim, in_dim));
    let mut composed_ua = ndarray::Array2::<f32>::zeros((out_dim, in_dim));
    let mut composed_lb = ndarray::Array1::<f32>::zeros(out_dim);
    let mut composed_ub = ndarray::Array1::<f32>::zeros(out_dim);

    for i in 0..out_dim {
        for j in 0..in_dim {
            let mut lower_sum = 0.0_f64;
            let mut upper_sum = 0.0_f64;
            for k in 0..mid_dim {
                let lower_coeff = outer_lower_a[[i, k]] as f64;
                let upper_coeff = outer_upper_a[[i, k]] as f64;
                let lower_inner = if lower_coeff >= 0.0 {
                    inner_lower_a[[k, j]] as f64
                } else {
                    inner_upper_a[[k, j]] as f64
                };
                let upper_inner = if upper_coeff >= 0.0 {
                    inner_upper_a[[k, j]] as f64
                } else {
                    inner_lower_a[[k, j]] as f64
                };
                lower_sum += lower_coeff * lower_inner;
                upper_sum += upper_coeff * upper_inner;
            }
            if !lower_sum.is_finite() {
                return Err(format!(
                    "composed lower_a[{i},{j}] became non-finite during LinearBounds composition",
                ));
            }
            if !upper_sum.is_finite() {
                return Err(format!(
                    "composed upper_a[{i},{j}] became non-finite during LinearBounds composition",
                ));
            }
            let lower_coeff = lower_sum as f32;
            let upper_coeff = upper_sum as f32;
            if !lower_coeff.is_finite() {
                return Err(format!(
                    "composed lower_a[{i},{j}] overflowed f32 during LinearBounds composition",
                ));
            }
            if !upper_coeff.is_finite() {
                return Err(format!(
                    "composed upper_a[{i},{j}] overflowed f32 during LinearBounds composition",
                ));
            }
            // Keep coefficient casts unbiased. Directed rounding on A-matrix
            // entries is not monotone once later concretization can see
            // negative inputs; we only direct-round scalar bias/concretization
            // boundaries, matching the existing CROWN coefficient pattern.
            composed_la[[i, j]] = lower_coeff;
            composed_ua[[i, j]] = upper_coeff;
        }

        let mut lower_bias = outer_lower_b[i] as f64;
        let mut upper_bias = outer_upper_b[i] as f64;
        for k in 0..mid_dim {
            let lower_coeff = outer_lower_a[[i, k]] as f64;
            let upper_coeff = outer_upper_a[[i, k]] as f64;
            let lower_inner = if lower_coeff >= 0.0 {
                inner_lower_b[k] as f64
            } else {
                inner_upper_b[k] as f64
            };
            let upper_inner = if upper_coeff >= 0.0 {
                inner_upper_b[k] as f64
            } else {
                inner_lower_b[k] as f64
            };
            lower_bias += lower_coeff * lower_inner;
            upper_bias += upper_coeff * upper_inner;
        }
        composed_lb[i] = if lower_bias.is_nan() {
            f32::NEG_INFINITY
        } else {
            next_down_f32(lower_bias as f32)
        };
        composed_ub[i] = if upper_bias.is_nan() {
            f32::INFINITY
        } else {
            next_up_f32(upper_bias as f32)
        };
    }

    LinearBounds::new(composed_la, composed_lb, composed_ua, composed_ub)
        .map_err(|e| format!("composed LinearBounds validation failed: {e}"))
}

/// Run spec-guided CROWN on a suffix graph requesting both concrete bounds
/// and optional `LinearBounds` over the suffix input (MFA domain).
fn scalar_spec_bounds_with_linear(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    label: &str,
    deadline_secs: u64,
) -> Result<(f32, f32, Option<LinearBounds>), String> {
    let spec = ndarray::arr2(&[[1.0_f32]]);
    let deadline = Instant::now() + Duration::from_secs(deadline_secs);
    let (crown, linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            input,
            &spec,
            None,
            node_bounds,
            Some(deadline),
        )
        .map_err(|e| format!("{label}: spec-guided CROWN (linear) failed: {e}"))?;
    let flat = crown.flatten();
    if flat.lower().len() != 1 {
        return Err(format!(
            "{label}: expected scalar output, got shape {:?}",
            flat.lower().shape()
        ));
    }
    let lower = flat.lower()[0];
    let upper = flat.upper()[0];
    if !lower.is_finite() || !upper.is_finite() {
        return Err(format!(
            "{label}: non-finite scalar bounds [{lower}, {upper}]"
        ));
    }
    if lower > upper {
        return Err(format!(
            "{label}: inverted scalar bounds [{lower}, {upper}]"
        ));
    }
    Ok((lower, upper, linear))
}

/// Run the cosine suffix pipeline with LinearBounds composition (Packet 2).
///
/// Instead of concretizing the MFA LinearBounds to a BoundedTensor and feeding
/// that into the suffix CROWN (which loses cross-dimensional correlation),
/// this function:
/// 1. Uses the concretized MFA bounds for suffix IBP/CROWN-IBP tightening
/// 2. Runs suffix spec-guided CROWN requesting LinearBounds over MFA
/// 3. Composes suffix LinearBounds with prefix MFA LinearBounds
/// 4. Concretizes the composed end-to-end LinearBounds on the raw input domain
///
/// The improvement: suffix CROWN linearizes dot/normsq into affine functions of
/// MFA. Composing with the MFA LinearBounds preserves the cross-dimensional
/// correlation that is lost when concretizing MFA to an interval box.
pub(super) fn cosine_bounds_with_linear_composition(
    mfa_linear: &LinearBounds,
    raw_input: &BoundedTensor,
    mfa_bounds: &BoundedTensor,
    spec_deadline_secs: u64,
    label_prefix: &str,
) -> Result<(f32, f32, f32, f32), String> {
    let (dot_graph, norm_sq_graph, _) = build_speaker_cosine_component_graphs();
    let dot_boundary = discover_ecapa_composition_boundary(&dot_graph)?;
    let normsq_boundary = discover_ecapa_composition_boundary(&norm_sq_graph)?;
    if dot_boundary.mfa_concat != normsq_boundary.mfa_concat {
        return Err(format!(
            "{label_prefix}: dot and normsq MFA concat mismatch: '{}' vs '{}'",
            dot_boundary.mfa_concat, normsq_boundary.mfa_concat
        ));
    }
    let dot_suffix = extract_single_input_subgraph(
        &dot_graph,
        &dot_boundary.mfa_concat,
        dot_graph.output_name(),
    )?;
    let normsq_suffix = extract_single_input_subgraph(
        &norm_sq_graph,
        &normsq_boundary.mfa_concat,
        norm_sq_graph.output_name(),
    )?;

    let dot_ibp_bounds = dot_suffix
        .collect_node_bounds(mfa_bounds)
        .map_err(|e| format!("{label_prefix} dot suffix IBP failed: {e}"))?;
    let normsq_ibp_bounds = normsq_suffix
        .collect_node_bounds(mfa_bounds)
        .map_err(|e| format!("{label_prefix} normsq suffix IBP failed: {e}"))?;

    let dot_tighten_deadline =
        Instant::now() + Duration::from_secs(COMPOSED_SUFFIX_CROWN_IBP_BUDGET_SECS);
    let dot_node_bounds = dot_suffix
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            mfa_bounds,
            dot_ibp_bounds,
            Some(dot_tighten_deadline),
        )
        .map_err(|e| format!("{label_prefix} dot suffix CROWN-IBP failed: {e}"))?
        .bounds;
    let normsq_tighten_deadline =
        Instant::now() + Duration::from_secs(COMPOSED_SUFFIX_CROWN_IBP_BUDGET_SECS);
    let normsq_node_bounds = normsq_suffix
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            mfa_bounds,
            normsq_ibp_bounds,
            Some(normsq_tighten_deadline),
        )
        .map_err(|e| format!("{label_prefix} normsq suffix CROWN-IBP failed: {e}"))?
        .bounds;

    let (dot_lower_std, dot_upper_std, dot_suffix_linear) = scalar_spec_bounds_with_linear(
        &dot_suffix,
        mfa_bounds,
        &dot_node_bounds,
        &format!("{label_prefix} dot"),
        spec_deadline_secs,
    )?;
    let (normsq_lower_std, normsq_upper_std, normsq_suffix_linear) =
        scalar_spec_bounds_with_linear(
            &normsq_suffix,
            mfa_bounds,
            &normsq_node_bounds,
            &format!("{label_prefix} normsq"),
            spec_deadline_secs,
        )?;

    let (dot_lower, dot_upper) = match dot_suffix_linear {
        Some(ref suffix_lb) => {
            let composed = compose_linear_bounds(mfa_linear, suffix_lb)?;
            let composed_bounds = composed.concretize_sound(raw_input).flatten();
            let cmp = (composed_bounds.lower()[0], composed_bounds.upper()[0]);
            let std = (dot_lower_std, dot_upper_std);
            // Both composed and standard are sound over-approximations.
            // Intersect: take max of lowers, min of uppers for tightest result.
            let best = (cmp.0.max(std.0), cmp.1.min(std.1));
            eprintln!(
                "{label_prefix} dot: composed=[{}, {}], std=[{}, {}], best=[{}, {}]",
                cmp.0, cmp.1, std.0, std.1, best.0, best.1,
            );
            best
        }
        None => {
            eprintln!("{label_prefix} dot: no suffix LinearBounds — using standard");
            (dot_lower_std, dot_upper_std)
        }
    };
    let (normsq_lower, normsq_upper) = match normsq_suffix_linear {
        Some(ref suffix_lb) => {
            let composed = compose_linear_bounds(mfa_linear, suffix_lb)?;
            let composed_bounds = composed.concretize_sound(raw_input).flatten();
            let cmp = (composed_bounds.lower()[0], composed_bounds.upper()[0]);
            let std = (normsq_lower_std, normsq_upper_std);
            let best = (cmp.0.max(std.0), cmp.1.min(std.1));
            eprintln!(
                "{label_prefix} normsq: composed=[{}, {}], std=[{}, {}], best=[{}, {}]",
                cmp.0, cmp.1, std.0, std.1, best.0, best.1,
            );
            best
        }
        None => {
            eprintln!("{label_prefix} normsq: no suffix LinearBounds — using standard");
            (normsq_lower_std, normsq_upper_std)
        }
    };

    Ok((dot_lower, dot_upper, normsq_lower, normsq_upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compose_linear_bounds_keeps_negative_point_sound_3499() {
        let inner = LinearBounds::new(
            array![[0.1_f32], [0.2_f32]],
            array![0.0_f32, 0.0_f32],
            array![[0.1_f32], [0.2_f32]],
            array![0.0_f32, 0.0_f32],
        )
        .expect("inner linear bounds should build");
        let outer = LinearBounds::new(
            array![[1.0_f32, 1.0_f32]],
            array![0.0_f32],
            array![[1.0_f32, 1.0_f32]],
            array![0.0_f32],
        )
        .expect("outer linear bounds should build");

        let composed = compose_linear_bounds(&inner, &outer)
            .expect("composition should preserve finite coefficients");
        let exact_coeff = inner.lower_a()[[0, 0]] as f64 + inner.lower_a()[[1, 0]] as f64;
        assert_eq!(
            composed.lower_a()[[0, 0]],
            exact_coeff as f32,
            "coefficient composition should not apply directed rounding",
        );

        let point = BoundedTensor::new(array![-1.0_f32].into_dyn(), array![-1.0_f32].into_dyn())
            .expect("point input should build");
        let concretized = composed.concretize_sound(&point).flatten();
        let exact_value = -exact_coeff;
        assert!(
            concretized.lower()[0] as f64 <= exact_value,
            "lower bound {} must stay <= exact value {} for negative point input",
            concretized.lower()[0],
            exact_value,
        );
        assert!(
            concretized.upper()[0] as f64 >= exact_value,
            "upper bound {} must stay >= exact value {} for negative point input",
            concretized.upper()[0],
            exact_value,
        );
    }
}
