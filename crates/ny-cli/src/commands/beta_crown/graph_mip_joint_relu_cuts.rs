// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #rel-joint-relu-cuts — JOINT PAIRED-ReLU MULTI-NEURON CUTS for the isomorphic
// difference-net whole-MILP finisher.
//
// THE LEVER. The per-node TRIANGLE relaxation the MILP encoder emits treats the
// two towers' partner neurons (a_X_i = f, b_X_i = g) INDEPENDENTLY: two disjoint
// triangles let (z_f, z_g, relu(z_f), relu(z_g)) roam the FULL PRODUCT of two
// triangles, so relu(z_f) and relu(z_g) may diverge freely even though f ≈ g
// forces the PAIR onto the diagonal z_f ≈ z_g. The rel-diff-coupling δ bound
// proves |z_f − z_g| ≤ δ_i rigorously; over the polygon
//     P = [l_f,u_f] × [l_g,u_g] ∩ { |z_f − z_g| ≤ δ }
// the CONVEX HULL of { (z_f, z_g, relu(z_f), relu(z_g)) } is dramatically tighter
// than the product of two triangles — it forces relu(z_f) ≈ relu(z_g) whenever
// z_f ≈ z_g, exactly the correlation that makes |f − g| small propagate through
// the ReLUs. We emit the DIAGONAL-COUPLING facets of that hull — valid linear
// inequalities in (z_f, z_g, y_f, y_g) — into the MILP, on top of the existing
// triangles.
//
// The facets we emit are the CONCAVE / CONVEX ENVELOPES over P of the coupled
// heights
//     h_{-}(z_f,z_g) = relu(z_f) − relu(z_g)      (the diagonal / difference cut)
//     h_{+}(z_f,z_g) = relu(z_f) + relu(z_g)      (the co-activation / sum cut)
// i.e. the tightest linear over/under-estimators c_f·y_f + c_g·y_g ⋛ α z_f +
// β z_g + γ valid for every exact point of P. h_{-} is the load-bearing one: it
// is what the two independent triangles cannot express.
//
// SOUNDNESS (absolute — a false "property HOLDS" certifies a false safety
// property). Each emitted row is a VALID inequality of the EXACT paired-ReLU set
// intersected with the (rigorous, δ-coupled, box) feasible region: it holds for
// EVERY exact-feasible (z_f, z_g, relu(z_f), relu(z_g)). The derivation:
//   * P is a superset of the reachable (z_f, z_g): [l_f,u_f]×[l_g,u_g] is the
//     sound CROWN-IBP box (contains every reachable pre-activation pair) and
//     |z_f − z_g| ≤ δ is the rel-diff-coupling RIGOROUS outward bound. So every
//     exact point lies in P.
//   * h_{±} is PIECEWISE-LINEAR over P with breaklines only at z_f = 0 and
//     z_g = 0. The MAX (resp. MIN) of h_{±} − (α z_f + β z_g) over the polygon P
//     is attained at a vertex of P's subdivision by {z_f = 0, z_g = 0}. We
//     enumerate every such subdivision vertex (every pairwise
//     intersection of the lines z_f ∈ {l_f,0,u_f}, z_g ∈ {l_g,0,u_g},
//     z_f−z_g = ±δ that lies inside P) and set γ = max (resp. min) of
//     h_{±} − (α z_f + β z_g) over them. Numerically identical candidates alone
//     are deduplicated: scale-relative approximate dedup is forbidden because it
//     can erase a soundness-critical off-diagonal vertex when δ is tiny.
//   * OUTWARD ROUNDING: vertex residuals are evaluated with directed-up f64
//     operations. Every committed γ is additionally inflated by a Lipschitz
//     allowance for the conservative coordinate-membership tolerance used while
//     enumerating line intersections. Upper bounds move up and lower bounds move
//     down; overflow fails open by omitting the envelope row.
//   * The (α, β) directions come from an f64 hull heuristic, but a WRONG
//     direction only yields a valid-but-loose cut (γ is still the rigorous
//     extremum over P for THAT direction) — never an invalid one. Soundness does
//     NOT depend on the hull enumeration being correct, only on the γ extremum +
//     outward rounding.
// Any doubt (non-finite box, non-finite δ, stable neuron, mismatched columns)
// FAILS OPEN: no cut for that neuron. The known-VIOLATED test must still return
// SAT — these cuts only remove SPURIOUS relaxed points, never a real one.

use std::collections::HashMap;

use ny_core::Bound;
use ny_mip::ir::Col;
use ny_propagate::{GraphNetwork, Layer};
use tracing::debug;

use super::graph_mip::GraphMipEncoding;
use super::graph_mip_diff_coupling::{compute_difference_bounds, detect_prefixes};

/// Gate: `NY_REL_JOINT_RELU_CUTS=1` arms the joint paired-ReLU cut emission.
pub(super) fn joint_relu_cuts_enabled() -> bool {
    matches!(
        std::env::var("NY_REL_JOINT_RELU_CUTS").ok().as_deref(),
        Some("1")
    )
}

/// Whether the co-activation (sum, h_{+}) envelope cuts are emitted in addition
/// to the load-bearing difference (h_{-}) cuts. Default ON; `=0` restricts to the
/// diagonal difference cuts only (fewer rows, faster LP).
fn sum_cuts_enabled() -> bool {
    !matches!(
        std::env::var("NY_REL_JOINT_RELU_CUTS_SUM").ok().as_deref(),
        Some("0")
    )
}

/// What a joint cut encodes (for diagnostics only — soundness is identical).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CutKind {
    /// `-δ ≤ zf − zg ≤ δ` pre-activation coupling.
    DeltaBox,
    /// A concave/convex envelope facet of `relu(zf) − relu(zg)` (the diagonal).
    Difference,
    /// A concave/convex envelope facet of `relu(zf) + relu(zg)` (co-activation).
    Sum,
}

/// A single joint cut: `lb ≤ Σ terms ≤ ub` over the four paired columns.
#[derive(Debug, Clone)]
pub(super) struct JointCut {
    pub terms: Vec<(Col, f64)>,
    pub lb: f64,
    pub ub: f64,
    pub kind: CutKind,
}

/// Diagnostics from one [`attach_joint_relu_cuts`] pass (for the measurement).
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct JointCutDiag {
    /// Paired unstable neurons that got at least one joint cut.
    pub paired_unstable: usize,
    /// Difference (h_{-}) rows emitted.
    pub diff_rows: usize,
    /// Sum (h_{+}) rows emitted.
    pub sum_rows: usize,
    /// Pre-activation δ-box coupling rows emitted (|z_f − z_g| ≤ δ).
    pub delta_box_rows: usize,
    /// Max, over emitted difference cuts, of the tightening vs the
    /// product-of-triangles bound on `y_f − y_g` at the polygon vertices
    /// (a coarse "how much does the diagonal facet buy" gauge).
    pub max_diff_tighten: f64,
}

/// The least representable float strictly greater than `x` (or `x` itself for
/// NaN/+∞). Used to turn round-to-nearest primitive results into upper bounds.
fn next_up(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn add_up(left: f64, right: f64) -> f64 {
    let value = left + right;
    if value.is_nan() {
        f64::INFINITY
    } else {
        next_up(value)
    }
}

fn mul_up(left: f64, right: f64) -> f64 {
    let value = left * right;
    if value.is_nan() {
        f64::INFINITY
    } else {
        next_up(value)
    }
}

/// Directed-up evaluation of
/// `cf*relu(a) + cg*relu(b) - alpha*a - beta*b` at one candidate vertex.
fn upper_vertex_residual(cf: f64, cg: f64, a: f64, b: f64, alpha: f64, beta: f64) -> f64 {
    let value = add_up(mul_up(cf, relu(a)), mul_up(cg, relu(b)));
    let value = add_up(value, mul_up(-alpha, a));
    add_up(value, mul_up(-beta, b))
}

/// Upper bound on objective drift when an exact line intersection is represented
/// by a candidate whose two coordinates are each within `tolerance`.
fn geometry_allowance(cf: f64, cg: f64, alpha: f64, beta: f64, tolerance: f64) -> f64 {
    let a_lipschitz = add_up(cf.abs(), alpha.abs());
    let b_lipschitz = add_up(cg.abs(), beta.abs());
    add_up(
        mul_up(a_lipschitz, tolerance),
        mul_up(b_lipschitz, tolerance),
    )
}

fn vertex_tolerance(lf: f64, uf: f64, lg: f64, ug: f64, delta: f64) -> f64 {
    let scale = lf
        .abs()
        .max(uf.abs())
        .max(lg.abs())
        .max(ug.abs())
        .max((uf - lf).abs())
        .max((ug - lg).abs())
        .max(delta.abs())
        .max(1.0);
    add_up(mul_up(scale, 1e-9), 1e-12)
}

/// The critical (z_f, z_g) vertices: a SUPERSET of the vertices of P's
/// subdivision by {z_f = 0, z_g = 0}. Every pairwise intersection of the lines
///   z_f ∈ {l_f, 0, u_f},  z_g ∈ {l_g, 0, u_g},  z_f − z_g = ±δ
/// that lies inside P = `[l_f, u_f] × [l_g, u_g] ∩ {|z_f − z_g| ≤ δ}`. Membership uses
/// a conservative tolerance; deduplication is exact. See the module SOUNDNESS note.
fn critical_vertices(lf: f64, uf: f64, lg: f64, ug: f64, delta: f64) -> (Vec<(f64, f64)>, f64) {
    let af = [lf, 0.0, uf];
    let ag = [lg, 0.0, ug];
    let mut pts: Vec<(f64, f64)> = Vec::new();
    // A generous membership tolerance: we would rather ADMIT a slightly-out
    // point (loosening γ, sound) than drop a true vertex.
    let tol = vertex_tolerance(lf, uf, lg, ug, delta);
    let push = |a: f64, b: f64, pts: &mut Vec<(f64, f64)>| {
        if !a.is_finite() || !b.is_finite() {
            return;
        }
        if a < lf - tol || a > uf + tol || b < lg - tol || b > ug + tol {
            return;
        }
        if (a - b).abs() > delta + tol {
            return;
        }
        // Clamp into the box. A tolerance-admitted point that remains slightly
        // outside the diagonal band only enlarges the candidate set and loosens γ.
        let a = a.clamp(lf, uf);
        let b = b.clamp(lg, ug);
        // Approximate dedup is unsound here: if `tol > delta`, it can merge
        // every off-diagonal vertex into the diagonal and manufacture the false
        // equality `relu(a) = relu(b)`. Exact numeric duplicates are harmless to
        // remove; all distinct candidates must remain.
        if pts.iter().any(|&(pa, pb)| pa == a && pb == b) {
            return;
        }
        pts.push((a, b));
    };
    // grid: z_f-line × z_g-line
    for &a in &af {
        for &b in &ag {
            push(a, b, &mut pts);
        }
    }
    // z_f-line × diagonal (b = a ∓ δ)
    for &a in &af {
        push(a, a - delta, &mut pts);
        push(a, a + delta, &mut pts);
    }
    // z_g-line × diagonal (a = b ± δ)
    for &b in &ag {
        push(b + delta, b, &mut pts);
        push(b - delta, b, &mut pts);
    }
    (pts, tol)
}

#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Solve the plane `α·a + β·b + γ = z` through three points; `None` if the
/// (a, b) projection is (near-)degenerate.
fn plane_through(
    p0: (f64, f64, f64),
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
) -> Option<(f64, f64, f64)> {
    let (a0, b0, z0) = p0;
    let (a1, b1, z1) = p1;
    let (a2, b2, z2) = p2;
    // Cross product of the two in-plane edges gives the (unnormalised) normal.
    let e1 = (a1 - a0, b1 - b0, z1 - z0);
    let e2 = (a2 - a0, b2 - b0, z2 - z0);
    let nx = e1.1 * e2.2 - e1.2 * e2.1;
    let ny = e1.2 * e2.0 - e1.0 * e2.2;
    let nz = e1.0 * e2.1 - e1.1 * e2.0;
    if nz.abs() < 1e-12 {
        return None; // vertical plane in z — not a graph over (a, b)
    }
    // n·(x − p0) = 0  ⇒  z = −(nx·a + ny·b)/nz + const.
    let alpha = -nx / nz;
    let beta = -ny / nz;
    let gamma = z0 - alpha * a0 - beta * b0;
    if !alpha.is_finite() || !beta.is_finite() || !gamma.is_finite() {
        return None;
    }
    Some((alpha, beta, gamma))
}

/// Candidate facet DIRECTIONS (α, β) for the UPPER envelope of the lifted points
/// `pts = {(a, b, h(a,b))}`: every triple whose plane sits weakly ABOVE all
/// points. Also seeds the four region-slope directions so a degenerate hull
/// still yields (looser) cuts. Deduped by (α, β). Tightness only; γ is recomputed
/// rigorously by the caller, so a spurious direction cannot break soundness.
fn upper_hull_directions(pts: &[(f64, f64, f64)], region_slopes: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut dirs: Vec<(f64, f64)> = Vec::new();
    let push_dir = |alpha: f64, beta: f64, dirs: &mut Vec<(f64, f64)>| {
        if !alpha.is_finite() || !beta.is_finite() {
            return;
        }
        if dirs
            .iter()
            .any(|&(a, b)| (a - alpha).abs() < 1e-7 && (b - beta).abs() < 1e-7)
        {
            return;
        }
        dirs.push((alpha, beta));
    };
    for &(a, b) in region_slopes {
        push_dir(a, b, &mut dirs);
    }
    let n = pts.len();
    if n >= 3 {
        // Scale tolerance to the height spread so the "all below" test is stable.
        let zmax = pts.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);
        let zmin = pts.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
        let tol = (zmax - zmin).abs().max(1.0) * 1e-7;
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let Some((alpha, beta, gamma)) = plane_through(pts[i], pts[j], pts[k]) else {
                        continue;
                    };
                    let all_below = pts
                        .iter()
                        .all(|&(a, b, z)| z <= alpha * a + beta * b + gamma + tol);
                    if all_below {
                        push_dir(alpha, beta, &mut dirs);
                    }
                }
            }
        }
    }
    dirs
}

/// Build the joint cuts for ONE paired unstable neuron. `(zf, zg, yf, yg)` are
/// the four columns; the ReLU pre-activation boxes are `[lf,uf]`, `[lg,ug]`; the
/// coupling bound is `|zf − zg| ≤ delta`. Returns the cuts (difference envelope,
/// optional sum envelope, plus the δ-box coupling on the pre-activations).
///
/// Every returned cut is a VALID inequality of the exact paired-ReLU set ∩ P (see
/// the module SOUNDNESS note).
#[allow(clippy::too_many_arguments)]
pub(super) fn neuron_joint_cuts(
    zf: Col,
    zg: Col,
    yf: Col,
    yg: Col,
    lf: f64,
    uf: f64,
    lg: f64,
    ug: f64,
    delta: f64,
    want_sum: bool,
) -> (Vec<JointCut>, f64) {
    let mut cuts: Vec<JointCut> = Vec::new();
    let mut max_tighten = 0.0f64;
    if !(lf.is_finite() && uf.is_finite() && lg.is_finite() && ug.is_finite() && delta.is_finite())
    {
        return (cuts, 0.0);
    }
    if delta < 0.0 || lf >= uf || lg >= ug {
        return (cuts, 0.0);
    }
    let (verts, vertex_tol) = critical_vertices(lf, uf, lg, ug, delta);
    if verts.len() < 3 {
        return (cuts, 0.0);
    }

    // δ-box coupling on the pre-activations: -δ ≤ zf − zg ≤ δ. This is exactly
    // the rel-diff-coupling row on the pre-activation pair; emitting it here
    // makes the gate self-contained (the diagonal envelope only BITES once the
    // LP is confined to P). Sound: |zf − zg| ≤ δ holds at every exact point.
    if delta.is_finite() {
        cuts.push(JointCut {
            terms: vec![(zf, 1.0), (zg, -1.0)],
            lb: -delta,
            ub: delta,
            kind: CutKind::DeltaBox,
        });
    }

    // Emit the concave (upper) and convex (lower) envelope facets of
    // h(a,b) = cf·relu(a) + cg·relu(b) over P, for each requested (cf, cg).
    let mut pairs: Vec<(f64, f64)> = vec![(1.0, -1.0)]; // difference (diagonal)
    if want_sum {
        pairs.push((1.0, 1.0)); // co-activation (sum)
    }
    for (idx, &(cf, cg)) in pairs.iter().enumerate() {
        let is_diff = idx == 0;
        let kind = if is_diff {
            CutKind::Difference
        } else {
            CutKind::Sum
        };
        // Lifted points for the height h.
        let lifted: Vec<(f64, f64, f64)> = verts
            .iter()
            .map(|&(a, b)| (a, b, cf * relu(a) + cg * relu(b)))
            .collect();
        // Region slopes: h is (cf·a + cg·b) on ++, cf·a on +-, cg·b on -+, 0 on --.
        let region_slopes = [(cf, cg), (cf, 0.0), (0.0, cg), (0.0, 0.0)];

        // UPPER envelope: cf·yf + cg·yg ≤ α zf + β zg + γ.
        for (alpha, beta) in upper_hull_directions(&lifted, &region_slopes) {
            // Rigorous γ = max over P vertices of (h − α a − β b), outward-UP.
            let mut gamma = f64::NEG_INFINITY;
            for &(a, b) in &verts {
                gamma = gamma.max(upper_vertex_residual(cf, cg, a, b, alpha, beta));
            }
            gamma = add_up(gamma, geometry_allowance(cf, cg, alpha, beta, vertex_tol));
            if !gamma.is_finite() {
                continue;
            }
            cuts.push(JointCut {
                terms: vec![(yf, cf), (yg, cg), (zf, -alpha), (zg, -beta)],
                lb: f64::NEG_INFINITY,
                ub: gamma,
                kind,
            });
        }

        // LOWER envelope: cf·yf + cg·yg ≥ α zf + β zg + γ. Run the upper-hull on
        // the NEGATED heights, then negate: the concave envelope of −h gives the
        // convex envelope of h.
        let neg: Vec<(f64, f64, f64)> = lifted.iter().map(|&(a, b, z)| (a, b, -z)).collect();
        let neg_slopes = [(-cf, -cg), (-cf, 0.0), (0.0, -cg), (0.0, 0.0)];
        for (alpha, beta) in upper_hull_directions(&neg, &neg_slopes) {
            let mut gamma_neg = f64::NEG_INFINITY;
            for &(a, b) in &verts {
                gamma_neg = gamma_neg.max(upper_vertex_residual(-cf, -cg, a, b, alpha, beta));
            }
            gamma_neg = add_up(
                gamma_neg,
                geometry_allowance(-cf, -cg, alpha, beta, vertex_tol),
            );
            if !gamma_neg.is_finite() {
                continue;
            }
            // −h ≤ α a + β b + γ_neg  ⇒  h ≥ (−α) a + (−β) b + (−γ_neg).
            let a2 = -alpha;
            let b2 = -beta;
            // Negating the directed-up bound moves the resulting lower bound
            // outward (down).
            let g2 = -gamma_neg;
            cuts.push(JointCut {
                terms: vec![(yf, cf), (yg, cg), (zf, -a2), (zg, -b2)],
                lb: g2,
                ub: f64::INFINITY,
                kind,
            });
        }

        // Diagnostic: how much the diagonal facet tightens y_f − y_g vs the
        // product-of-triangles bound (upper_tri_f(a) − max(0,b) etc.) at the
        // vertices. Coarse — informative only.
        if is_diff {
            // product-of-triangles upper bound on (yf − yg): yf ≤ uf(a−lf)/(uf−lf),
            // yg ≥ max(0, b). So (yf − yg) ≤ uf(a−lf)/(uf−lf) − max(0, b).
            let denom_f = uf - lf;
            for &(a, b, z) in &lifted {
                let tri = if denom_f > 0.0 {
                    uf * (a - lf) / denom_f
                } else {
                    a.max(0.0)
                } - b.max(0.0);
                // z here = relu(a) − relu(b) = the TRUE height (cf=1,cg=−1).
                let slack = tri - z;
                if slack.is_finite() {
                    max_tighten = max_tighten.max(slack);
                }
            }
        }
    }
    (cuts, max_tighten)
}

/// Detect the paired towers + rigorous per-neuron δ, then emit joint paired-ReLU
/// cuts for every paired UNSTABLE ReLU. Returns `(rows_added, diag)`. Fails open
/// (0 rows) when the graph is not a stitched diff net or δ cannot be derived.
pub(super) fn attach_joint_relu_cuts(
    enc: &mut GraphMipEncoding,
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    flat_bounds: &HashMap<String, Vec<Bound>>,
) -> (usize, JointCutDiag) {
    let mut diag = JointCutDiag::default();
    let Some((pa, pb)) = detect_prefixes(graph) else {
        debug!("rel joint-relu-cuts: not a stitched diff net; no cuts");
        return (0, diag);
    };
    let Some(diffb) = compute_difference_bounds(graph, input_bounds, flat_bounds, pa, pb) else {
        debug!("rel joint-relu-cuts: could not derive difference bounds; no cuts");
        return (0, diag);
    };
    let Ok(exec) = graph.exec_order() else {
        return (0, diag);
    };
    let want_sum = sum_cuts_enabled();
    let mut added = 0usize;

    for name in exec {
        // a-tower ReLU nodes only; the b-tower is the mirror.
        let Some(suffix) = name.strip_prefix(pa) else {
            continue;
        };
        let Some(a_relu) = graph.node(name) else {
            continue;
        };
        if !matches!(a_relu.layer(), Layer::ReLU(_)) {
            continue;
        }
        let b_relu_name = format!("{pb}{suffix}");
        let Some(b_relu) = graph.node(&b_relu_name) else {
            continue;
        };
        if !matches!(b_relu.layer(), Layer::ReLU(_)) {
            continue;
        }
        // Pre-activation (input) nodes of both towers' ReLUs.
        let Some(a_in) = a_relu.inputs().first().cloned() else {
            continue;
        };
        let Some(b_in) = b_relu.inputs().first().cloned() else {
            continue;
        };
        let Some(in_suffix) = a_in.strip_prefix(pa) else {
            continue;
        };
        // Rigorous δ on the pre-activation difference |zf − zg| for this layer.
        let Some(delta_pre) = diffb.get(in_suffix) else {
            continue;
        };
        // Columns: zf/zg = pre-activation (input node) cols; yf/yg = post cols.
        let (Some(zf_cols), Some(zg_cols)) = (enc.node_cols.get(&a_in), enc.node_cols.get(&b_in))
        else {
            continue;
        };
        let (Some(yf_cols), Some(yg_cols)) =
            (enc.node_cols.get(name), enc.node_cols.get(&b_relu_name))
        else {
            continue;
        };
        // Boxes on the pre-activations.
        let (Some(fbox), Some(gbox)) = (flat_bounds.get(&a_in), flat_bounds.get(&b_in)) else {
            continue;
        };
        let n = zf_cols.len();
        if zg_cols.len() != n
            || yf_cols.len() != n
            || yg_cols.len() != n
            || fbox.len() != n
            || gbox.len() != n
            || delta_pre.len() != n
        {
            continue;
        }

        let mut new_cuts: Vec<JointCut> = Vec::new();
        for i in 0..n {
            let lf = f64::from(fbox[i].lower());
            let uf = f64::from(fbox[i].upper());
            let lg = f64::from(gbox[i].lower());
            let ug = f64::from(gbox[i].upper());
            // Only PAIRED UNSTABLE neurons: both towers straddle 0 (the triangle
            // slack the joint hull tightens). Stable neurons are exact — skip.
            if !(lf < 0.0 && uf > 0.0 && lg < 0.0 && ug > 0.0) {
                continue;
            }
            let delta = delta_pre[i];
            if !delta.is_finite() || delta < 0.0 {
                continue;
            }
            let (cuts, tighten) = neuron_joint_cuts(
                zf_cols[i], zg_cols[i], yf_cols[i], yg_cols[i], lf, uf, lg, ug, delta, want_sum,
            );
            if cuts.is_empty() {
                continue;
            }
            diag.paired_unstable += 1;
            diag.max_diff_tighten = diag.max_diff_tighten.max(tighten);
            new_cuts.extend(cuts);
        }
        for cut in new_cuts {
            match cut.kind {
                CutKind::DeltaBox => diag.delta_box_rows += 1,
                CutKind::Difference => diag.diff_rows += 1,
                CutKind::Sum => diag.sum_rows += 1,
            }
            enc.problem.add_row(cut.lb, cut.ub, cut.terms);
            added += 1;
        }
    }
    (added, diag)
}

#[path = "graph_mip_joint_relu_cuts_tests.rs"]
pub(crate) mod research;
