// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #rel-diff-coupling — DIFFERENCE-COUPLING relaxation strengthening for the
// isomorphic-network whole-net MILP finisher.
//
// The difference net (`build_difference_network`) stitches two ISOMORPHIC
// towers f and g — identical architecture, near-equal weights — as two disjoint
// node sets prefixed `a_` (f) and `b_` (g), sharing `NETWORK_INPUT`, joined by a
// final `Sub` at `diff_output = a_out - b_out`. The per-node TRIANGLE ReLU
// relaxation the MILP encoder emits treats every neuron INDEPENDENTLY, so it
// DISCARDS the cross-tower correlation that makes f and g near-equal: each
// intermediate `a_X` and its partner `b_X` are each free to roam their full
// (order-1e4-wide) forward-reachable box, and a single output-band row cannot
// propagate back through that loose relaxation (measured: OBBT shrinks the
// big-M by exactly zero).
//
// THE LEVER (idea A): for each PAIRED pre-/post-activation neuron i, the
// per-neuron DIFFERENCE h_i = a_X_i - b_X_i is provably SMALL — bounded
// layer-by-layer by the weight difference ‖W_a - W_b‖ propagated with the input
// box and the prior layer's difference bounds. We compute a RIGOROUS
// outward-rounded δ_i on |h_i| and add the COUPLING ROWS `-δ_i <= a_X_i - b_X_i
// <= δ_i` to the encoded MILP. These rows carry exactly the correlation the
// triangle relaxation throws away, letting the output band bite back on the
// intermediates.
//
// SOUNDNESS: a coupling row is a VALID relaxation constraint iff it holds for
// EVERY point of the EXACT violation region — i.e. iff δ_i is a rigorous UPPER
// bound on |f_X_i(x) - g_X_i(x)| over the whole input box. We derive δ from the
// EXACT difference identity per layer (below), bounded with interval arithmetic
// and OUTWARD-rounded finalisation, so δ_i is never under-tight. An added
// constraint that holds for every exact point can only remove SPURIOUS relaxed
// points — it never removes a real (exact-feasible) point, so the MILP feasible
// set stays a valid over-approximation of the reachable set. Any doubt (unknown
// op, shape mismatch, missing box) FAILS OPEN to δ_i = +∞ (no row emitted),
// which keeps the current sound relaxation. A too-tight δ would be the only way
// to certify a false property; the derivation and rounding preclude it.
//
// Per-layer EXACT difference identities (a = f, b = g; e_U = a_U - b_U the input
// difference, B_U a magnitude bound on the g-tower input |b_U|):
//   * Linear  y = W·u (+bias):
//       d = a_out - b_out = W_a·e_U + (W_a - W_b)·b_U + (bias_a - bias_b)
//       |d_i| <= Σ_j |W_a_ij|·|e_U_j| + Σ_j |W_a_ij - W_b_ij|·B_U_j + |Δbias_i|
//   * AddConstant y = u + c:   |d_i| <= |e_U_i| + |c_a_i - c_b_i|
//   * SubConstant y = u - c:   |d_i| <= |e_U_i| + |c_a_i - c_b_i|  (same `reverse`)
//   * MulConstant y = u * s:   |d_i| <= |s_a_i|·|e_U_i| + |s_a_i - s_b_i|·B_U_i
//   * ReLU:                    |d_i| <= |e_U_i|            (ReLU is 1-Lipschitz)
//   * Flatten / Reshape:       |d_i| =  |e_U_i|            (index permutation)
// The shared NETWORK_INPUT feeds both towers, so its difference is EXACTLY 0.

use std::collections::HashMap;

use ny_core::Bound;
use ny_propagate::{GraphNetwork, Layer, NETWORK_INPUT};
use tracing::debug;

use super::graph_mip::GraphMipEncoding;

/// Gate: `NY_REL_DIFF_COUPLING=1` arms the coupling-row emission. Default-off.
pub(super) fn diff_coupling_enabled() -> bool {
    matches!(
        std::env::var("NY_REL_DIFF_COUPLING").ok().as_deref(),
        Some("1")
    )
}

/// The prefix pairs `build_difference_network` chooses between (kept in the same
/// preference order so detection matches construction).
const PREFIX_CANDIDATES: &[(&str, &str)] = &[
    ("a_", "b_"),
    ("net_a_", "net_b_"),
    ("left_", "right_"),
    ("first_", "second_"),
    ("__diff_a_", "__diff_b_"),
];

/// Reserved sentinel node names that belong to neither tower.
const SENTINELS: &[&str] = &[NETWORK_INPUT, "diff_output"];

/// Whether the two input lists describe the same tower-local edges.  Merely
/// having the same prefixed node names is not enough: a malformed graph could
/// wire `a_x` from `a_u` but `b_x` from some other `b_v`.  Reusing `a_u-b_u` as
/// a bound for that pair would then be unsound.  The shared network input is the
/// only input that is intentionally unprefixed.
fn inputs_are_paired(a_inputs: &[String], b_inputs: &[String], pa: &str, pb: &str) -> bool {
    a_inputs.len() == b_inputs.len()
        && a_inputs.iter().zip(b_inputs).all(|(a, b)| {
            if a == NETWORK_INPUT {
                b == NETWORK_INPUT
            } else {
                a.strip_prefix(pa)
                    .is_some_and(|suffix| b.strip_prefix(pb) == Some(suffix))
            }
        })
}

/// Detect which `(prefix_a, prefix_b)` pair partitions this diff net's nodes
/// into two isomorphic towers (every non-sentinel node starts with exactly one
/// prefix and the stripped suffix-sets match). Returns `None` if the graph is
/// not a stitched difference net (fail-open: no coupling).
pub(super) fn detect_prefixes(graph: &GraphNetwork) -> Option<(&'static str, &'static str)> {
    for &(pa, pb) in PREFIX_CANDIDATES {
        let mut suf_a: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        let mut suf_b: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        let mut ok = true;
        for name in graph.node_names() {
            let n = name.as_str();
            if SENTINELS.contains(&n) {
                continue;
            }
            if let Some(s) = n.strip_prefix(pa) {
                suf_a.insert(s);
            } else if let Some(s) = n.strip_prefix(pb) {
                suf_b.insert(s);
            } else {
                ok = false;
                break;
            }
        }
        if ok && !suf_a.is_empty() && suf_a == suf_b {
            // Authenticate the paired topology as well as the names.  Every
            // tower-local edge must map to the corresponding edge in the other
            // tower; otherwise fail open and emit no coupling rows.
            let topology_is_paired = suf_a.iter().all(|suffix| {
                let Some(a_node) = graph.node(&format!("{pa}{suffix}")) else {
                    return false;
                };
                let Some(b_node) = graph.node(&format!("{pb}{suffix}")) else {
                    return false;
                };
                inputs_are_paired(a_node.inputs(), b_node.inputs(), pa, pb)
            });
            if topology_is_paired {
                return Some((pa, pb));
            }
        }
    }
    None
}

/// `max(|lo|, |hi|)` per element — an exact (f32→f64) magnitude bound.
fn box_mag(bounds: &[Bound]) -> Vec<f64> {
    bounds
        .iter()
        .map(|b| {
            let lo = f64::from(b.lower()).abs();
            let hi = f64::from(b.upper()).abs();
            lo.max(hi)
        })
        .collect()
}

/// Outward (upward) finalisation of a non-negative interval-arithmetic bound
/// accumulated over `n_ops` f64 fused mul/adds. Inflating an UPPER bound upward
/// is always sound (a looser coupling band never excludes a real point); this
/// covers accumulated f64 rounding with a generous relative + absolute margin.
fn outward_up(x: f64, n_ops: usize) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    // Rounding error of n_ops nonneg fused operations is bounded by ~n·u·|x|
    // with u = 2^-53 = EPSILON/2; use 8·(n+2)·EPSILON as a safe over-estimate.
    let rel = 8.0 * (n_ops as f64 + 2.0) * f64::EPSILON;
    x * (1.0 + rel) + 16.0 * f64::MIN_POSITIVE
}

/// `coeff · val` where `coeff >= 0`, avoiding the `0 · ∞ = NaN` trap (a zero
/// weight contributes zero even against an unbounded input difference).
#[inline]
fn nonneg_prod(coeff: f64, val: f64) -> f64 {
    if coeff == 0.0 {
        0.0
    } else {
        coeff * val
    }
}

/// Flatten a broadcastable constant tensor to a per-output-element f64 vector of
/// length `out_len` (scalar broadcasts; exact-length passes through; any other
/// shape fails open with `None`).
fn const_to_vec(constant: &ndarray::ArrayD<f32>, out_len: usize) -> Option<Vec<f64>> {
    let flat: Vec<f64> = constant.iter().map(|&v| f64::from(v)).collect();
    if flat.len() == 1 {
        Some(vec![flat[0]; out_len])
    } else if flat.len() == out_len {
        Some(flat)
    } else {
        None
    }
}

/// Per-suffix (original-node) rigorous difference bound δ, keyed by the suffix
/// shared by the `a_`/`b_` towers. `δ[suffix][i]` is an OUTWARD-rounded upper
/// bound on `|f_node_i(x) - g_node_i(x)|` over the whole input box; `+∞` entries
/// mean "no valid finite bound" (fail-open — no coupling row for that neuron).
pub(super) fn compute_difference_bounds(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    flat_bounds: &HashMap<String, Vec<Bound>>,
    pa: &str,
    pb: &str,
) -> Option<HashMap<String, Vec<f64>>> {
    let exec = graph.exec_order().ok()?;
    let input_len = input_bounds.len();
    let input_mag = box_mag(input_bounds);

    // suffix -> difference bound on that node's OUTPUT tensor.
    let mut diffb: HashMap<String, Vec<f64>> = HashMap::new();

    // For an a-tower node's single input: its difference bound (ε) and the
    // g-tower magnitude bound (B). `NETWORK_INPUT` is shared → ε = 0, B = input
    // magnitudes. Unknown/missing → +∞ vectors (fail-open).
    let eps_and_mag =
        |input_name: &str, diffb: &HashMap<String, Vec<f64>>| -> (Vec<f64>, Vec<f64>) {
            if input_name == NETWORK_INPUT {
                return (vec![0.0; input_len], input_mag.clone());
            }
            let suffix = input_name.strip_prefix(pa).unwrap_or(input_name);
            let eps = diffb
                .get(suffix)
                .cloned()
                .unwrap_or_else(|| vec![f64::INFINITY; 1]);
            let b_name = format!("{pb}{suffix}");
            let mag = flat_bounds
                .get(&b_name)
                .map(|b| box_mag(b))
                .unwrap_or_else(|| vec![f64::INFINITY; eps.len()]);
            (eps, mag)
        };

    for name in exec {
        // Only walk the a-tower; the b-tower is its mirror image.
        let Some(suffix) = name.strip_prefix(pa) else {
            continue;
        };
        if SENTINELS.contains(&name.as_str()) {
            continue;
        }
        let a_node = graph.node(name)?;
        let b_name = format!("{pb}{suffix}");
        let Some(b_node) = graph.node(&b_name) else {
            continue;
        };
        // Output length from the encoded box (aligns with node_cols later).
        let out_len = flat_bounds.get(name).map(Vec::len);

        let inf_vec = |n: Option<usize>| vec![f64::INFINITY; n.unwrap_or(1)];

        // Every currently supported transfer is unary.  A graph node with an
        // unexpected arity must not silently ignore extra operands: doing so
        // could under-bound its true paired difference.  Fail open here and let
        // the resulting infinity propagate downstream.
        let delta: Vec<f64> = if a_node.inputs().len() != 1 || b_node.inputs().len() != 1 {
            inf_vec(out_len)
        } else {
            let input_name = &a_node.inputs()[0];
            match (a_node.layer(), b_node.layer()) {
                (Layer::Linear(la), Layer::Linear(lb)) => {
                    let (out_a, in_a) = la.weight.dim();
                    let (out_b, in_b) = lb.weight.dim();
                    if out_a != out_b || in_a != in_b {
                        inf_vec(out_len)
                    } else {
                        let (eps_u, mag_u) = eps_and_mag(input_name, &diffb);
                        if eps_u.len() != in_a || mag_u.len() != in_a {
                            inf_vec(Some(out_a))
                        } else {
                            let bias_a = la.bias.as_ref();
                            let bias_b = lb.bias.as_ref();
                            (0..out_a)
                                .map(|i| {
                                    let mut acc = 0.0f64;
                                    for j in 0..in_a {
                                        let wa = f64::from(la.weight[[i, j]]);
                                        let wb = f64::from(lb.weight[[i, j]]);
                                        // W_a·e_U term (uses f's weight magnitude).
                                        acc += nonneg_prod(wa.abs(), eps_u[j]);
                                        // ΔW·b_U term (exact f32→f64 difference).
                                        acc += nonneg_prod((wa - wb).abs(), mag_u[j]);
                                    }
                                    let dbias = match (bias_a, bias_b) {
                                        (Some(ba), Some(bb)) => {
                                            (f64::from(ba[i]) - f64::from(bb[i])).abs()
                                        }
                                        (Some(ba), None) => f64::from(ba[i]).abs(),
                                        (None, Some(bb)) => f64::from(bb[i]).abs(),
                                        (None, None) => 0.0,
                                    };
                                    acc += dbias;
                                    outward_up(acc, 2 * in_a + 1)
                                })
                                .collect()
                        }
                    }
                }
                (Layer::ReLU(_), Layer::ReLU(_)) => {
                    // 1-Lipschitz: |ReLU(a)-ReLU(b)| <= |a-b|.
                    let (eps_u, _) = eps_and_mag(input_name, &diffb);
                    eps_u
                }
                (Layer::Flatten(_), Layer::Flatten(_)) | (Layer::Reshape(_), Layer::Reshape(_)) => {
                    let (eps_u, _) = eps_and_mag(input_name, &diffb);
                    eps_u
                }
                (Layer::AddConstant(ca), Layer::AddConstant(cb)) => {
                    let (eps_u, _) = eps_and_mag(input_name, &diffb);
                    let n = out_len.unwrap_or(eps_u.len());
                    match (
                        const_to_vec(ca.constant(), n),
                        const_to_vec(cb.constant(), n),
                    ) {
                        (Some(va), Some(vb)) if eps_u.len() == n => (0..n)
                            .map(|i| outward_up(eps_u[i] + (va[i] - vb[i]).abs(), 2))
                            .collect(),
                        _ => inf_vec(out_len),
                    }
                }
                (Layer::SubConstant(ca), Layer::SubConstant(cb)) => {
                    // Both towers must subtract in the SAME direction; a `reverse`
                    // mismatch changes the function (y=c-x vs y=x-c) and breaks the
                    // 1-Lipschitz-in-x bound → fail open.
                    if ca.reverse != cb.reverse {
                        inf_vec(out_len)
                    } else {
                        let (eps_u, _) = eps_and_mag(input_name, &diffb);
                        let n = out_len.unwrap_or(eps_u.len());
                        match (
                            const_to_vec(ca.constant(), n),
                            const_to_vec(cb.constant(), n),
                        ) {
                            (Some(va), Some(vb)) if eps_u.len() == n => (0..n)
                                .map(|i| outward_up(eps_u[i] + (va[i] - vb[i]).abs(), 2))
                                .collect(),
                            _ => inf_vec(out_len),
                        }
                    }
                }
                (Layer::MulConstant(ca), Layer::MulConstant(cb)) => {
                    let (eps_u, mag_u) = eps_and_mag(input_name, &diffb);
                    let n = out_len.unwrap_or(eps_u.len());
                    match (
                        const_to_vec(ca.constant(), n),
                        const_to_vec(cb.constant(), n),
                    ) {
                        (Some(sa), Some(sb)) if eps_u.len() == n && mag_u.len() == n => (0..n)
                            .map(|i| {
                                let t = nonneg_prod(sa[i].abs(), eps_u[i])
                                    + nonneg_prod((sa[i] - sb[i]).abs(), mag_u[i]);
                                outward_up(t, 3)
                            })
                            .collect(),
                        _ => inf_vec(out_len),
                    }
                }
                // Any other op (incl. DivConstant, whose reciprocal rounding we do
                // not certify here) fails open: δ = +∞ blocks coupling for this
                // neuron AND everything downstream of it — sound, just no benefit.
                _ => inf_vec(out_len),
            }
        };

        diffb.insert(suffix.to_string(), delta);
    }

    Some(diffb)
}

/// Add the coupling rows `-δ_i <= a_X_i - b_X_i <= δ_i` to `enc` for every
/// paired neuron with a FINITE δ and matching column vectors. Returns the number
/// of rows added. Only ever RESTRICTS the feasible set (by valid inequalities),
/// so it is strictly sound; non-paired / infinite / shape-mismatched neurons are
/// skipped.
pub(super) fn apply_diff_coupling(
    enc: &mut GraphMipEncoding,
    diffb: &HashMap<String, Vec<f64>>,
    pa: &str,
    pb: &str,
) -> usize {
    let mut added = 0usize;
    for (suffix, delta) in diffb {
        let a_name = format!("{pa}{suffix}");
        let b_name = format!("{pb}{suffix}");
        let (Some(acols), Some(bcols)) = (enc.node_cols.get(&a_name), enc.node_cols.get(&b_name))
        else {
            continue;
        };
        if acols.len() != bcols.len() || acols.len() != delta.len() {
            continue;
        }
        for (k, &d) in delta.iter().enumerate() {
            if !d.is_finite() || d < 0.0 {
                continue;
            }
            enc.problem
                .add_row(-d, d, [(acols[k], 1.0), (bcols[k], -1.0)]);
            added += 1;
        }
    }
    added
}

/// Convenience: detect prefixes, compute δ, and apply the coupling rows to an
/// already-encoded difference net. Returns `(rows_added, output_delta)` where
/// `output_delta` is the bound on the FINAL paired layer feeding `diff_output`
/// (the interval difference bound on `f - g`, for logging). Fails open (0, None)
/// if the graph is not a stitched diff net or δ cannot be derived.
pub(super) fn attach_diff_coupling(
    enc: &mut GraphMipEncoding,
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    flat_bounds: &HashMap<String, Vec<Bound>>,
) -> (usize, Option<Vec<f64>>) {
    let Some((pa, pb)) = detect_prefixes(graph) else {
        debug!("rel diff-coupling: not a stitched difference net; no coupling");
        return (0, None);
    };
    let Some(diffb) = compute_difference_bounds(graph, input_bounds, flat_bounds, pa, pb) else {
        debug!("rel diff-coupling: could not derive difference bounds; no coupling");
        return (0, None);
    };
    // The operand of the final Sub is a paired node; report its δ (the interval
    // bound on |f - g|) for measurement.
    let out_delta = graph
        .node("diff_output")
        .and_then(|n| n.inputs().first().cloned())
        .and_then(|opnd| opnd.strip_prefix(pa).map(str::to_string))
        .and_then(|suffix| diffb.get(&suffix).cloned());
    let added = apply_diff_coupling(enc, &diffb, pa, pb);
    (added, out_delta)
}

#[cfg(test)]
#[path = "graph_mip_diff_coupling_tests.rs"]
mod tests;
