// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PRODUCTION coupled-δ leaf oracle (#rel-coupled-relu integration).
//!
//! A sound, CROWN-style LINEAR bound on the isomorphic difference
//! `h(x) = f(x) - g(x)` that is TIGHTER than the plain difference-network CROWN
//! backward because it keeps the paired pre-activations symbolically coupled.
//! It attaches to the relational input-split BaB through the
//! [`GraphMipLeafOracle`] seam (exactly like the Graph-MIP edge oracle): on a
//! near-verified DEEP subdomain the engine hands us the exact input box and the
//! still-undecided spec rows, and we return
//! [`GraphMipLeafVerdict::VerifiedAllRows`] only after an independent,
//! directed-rounded source-network IBP replay certifies every row. The coupled
//! f64 calculation is a candidate filter, never the verdict authority. On any
//! replay miss the result is `Undecided` and the domain proceeds through the
//! unchanged split path. The oracle is therefore strictly additive (converts
//! "requeue" into "verified", never flips or discards a BaB decision),
//! preserving the 0-wrong moat.
//!
//! # Soundness of the joint ReLU-difference relaxation
//!
//! For any reals with `d = z_f - z_g`, relu is SUBADDITIVE
//! (`relu(a+b) ≤ relu(a)+relu(b)`, relu convex with relu(0)=0), so:
//!   * `relu(z_f) - relu(z_g) = relu(z_g + d) - relu(z_g) ≤ relu(d)`, and
//!   * `relu(z_g) - relu(z_f) ≤ relu(-d)` ⇒ `relu(z_f) - relu(z_g) ≥ -relu(-d)`.
//!
//! Hence `relu(z_f) - relu(z_g) ∈ [-relu(-d), relu(d)]` — a VALID
//! over-approximation depending only on the coupling `d`. We relax `relu(d)`
//! linearly over `d`'s TIGHT symmetric interval `[-cap, cap]` (small because the
//! nets are isomorphic; `λ = 1/2`, constant `cap/2`) while keeping
//! `d = z_f - z_g` symbolic — so the input-box concretization stays tight where
//! the two independent ReLU triangles would blow the difference up. The joint
//! form is applied ONLY per-neuron where its constant `cap/2` provably beats the
//! two independent triangle constants `μ_f + μ_g`, so the fixed point tightens
//! monotonically and never loses to plain coupled CROWN.
//!
//! # The oracle's soundness obligation (see [`GraphMipLeafOracle`])
//!
//! `VerifiedAllRows` requires `obj·y > threshold` for EVERY reachable output `y`
//! on the subdomain, for every requested row. We compute a sound LOWER bound `L`
//! on `min obj·(f-g)` over the box and return `VerifiedAllRows` only when
//! `L > threshold` for all rows. Since ordinary nearest-rounded f64 arithmetic
//! does not itself constitute a bound certificate, `L` is used only to select
//! promising rows. The authoritative replay propagates the exact subdomain box
//! through both source graphs using their outward-rounded IBP implementations,
//! forms `f-g` intervals with directed f64 rounding, and evaluates each row
//! downward. Since the diff network's output is
//! `f - g` coordinate-wise over a SHARED input (`build_difference_network`), and
//! the input-split subdomain is described EXACTLY by its input box (no ReLU-split
//! premises to honor — ignoring premises only enlarges the reachable set, which
//! is conservative), the replay's directed lower endpoint is a valid lower
//! bound and the verdict is sound. The affine-chain topology is proven
//! structurally and then spot-checked against the source graphs' forward at
//! construction; any mismatch disarms the oracle (fail-closed).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2};

// Observability counters (the production tracing subscriber defaults to WARN, so
// `info!` is invisible; these report via eprintln like the rest of the lane).
static COUPLED_CONSULTS: AtomicUsize = AtomicUsize::new(0);
static COUPLED_VERIFIED: AtomicUsize = AtomicUsize::new(0);

use ny_core::Bound;
use ny_propagate::beta_crown::graph_mip_leaf::{
    GraphInputLeafRequest, GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
};
use ny_propagate::{GraphNetwork, Layer, Verifier, NETWORK_INPUT};

/// A composed affine layer `y = W x + b` (f64 for bound precision).
struct Affine {
    w: Array2<f64>,
    b: Array1<f64>,
}

/// Prove that `exec` is the whole graph and is a single source-to-output chain.
///
/// A topological ordering alone is not a path: composing its nodes in sequence
/// would silently change the semantics of a branched DAG.  The coupled oracle
/// is verdict-bearing, so sampling cannot stand in for this structural proof.
fn is_single_input_chain(graph: &GraphNetwork, exec: &[String]) -> bool {
    if exec.is_empty()
        || exec.len() != graph.num_nodes()
        || exec.last().map(String::as_str) != Some(graph.output_name())
    {
        return false;
    }

    let mut expected_input = NETWORK_INPUT;
    for name in exec {
        let Some(node) = graph.node(name) else {
            return false;
        };
        if node.inputs().len() != 1 || node.inputs()[0] != expected_input {
            return false;
        }
        expected_input = name;
    }
    true
}

fn affine_is_finite(layer: &Affine) -> bool {
    layer.w.iter().chain(layer.b.iter()).all(|v| v.is_finite())
}

fn compatible_affine_chains(f: &[Affine], g: &[Affine], input_dim: usize) -> bool {
    if f.is_empty() || f.len() != g.len() || input_dim == 0 {
        return false;
    }
    let mut expected_width = input_dim;
    for (fl, gl) in f.iter().zip(g) {
        if !affine_is_finite(fl)
            || !affine_is_finite(gl)
            || fl.w.ncols() != expected_width
            || gl.w.ncols() != expected_width
            || fl.w.nrows() != fl.b.len()
            || gl.w.nrows() != gl.b.len()
            || fl.w.raw_dim() != gl.w.raw_dim()
            || fl.b.len() != gl.b.len()
        {
            return false;
        }
        expected_width = fl.w.nrows();
    }
    true
}

/// Extract a branch's affine+ReLU chain: the affine blocks BETWEEN ReLUs
/// (`n` affines ⇒ `n-1` ReLUs). Folds Linear / Add|Sub|Mul|DivConstant /
/// Flatten|Reshape into a running affine; splits at each ReLU. Returns `None`
/// on any op it cannot fold (fail-closed — the oracle disarms for that pair).
fn extract_affine_chain(graph: &GraphNetwork, input_dim: usize) -> Option<Vec<Affine>> {
    let exec = graph.exec_order().ok()?;
    if input_dim == 0 || !is_single_input_chain(graph, exec) {
        return None;
    }
    let mut layers: Vec<Affine> = Vec::new();
    let mut w: Array2<f64> = Array2::eye(input_dim);
    let mut b: Array1<f64> = Array1::zeros(input_dim);
    let scalar_or_vec = |c: &ndarray::ArrayD<f32>, n: usize| -> Option<Array1<f64>> {
        if c.len() == 1 {
            Some(Array1::from_elem(n, f64::from(*c.iter().next().unwrap())))
        } else if c.len() == n {
            Some(Array1::from_iter(c.iter().map(|&v| f64::from(v))))
        } else {
            None
        }
    };
    for name in exec {
        let node = graph.node(name)?;
        match node.layer() {
            Layer::Flatten(_) | Layer::Reshape(_) => {}
            Layer::Linear(lin) => {
                let lw = lin.weight().mapv(f64::from);
                if lw.ncols() != w.nrows()
                    || lin.bias().is_some_and(|bias| bias.len() != lw.nrows())
                    || !lw.iter().all(|v| v.is_finite())
                    || lin
                        .bias()
                        .is_some_and(|bias| bias.iter().any(|v| !v.is_finite()))
                {
                    return None;
                }
                w = lw.dot(&w);
                let mut nb = lw.dot(&b);
                if let Some(bias) = lin.bias() {
                    nb = nb + bias.mapv(f64::from);
                }
                b = nb;
            }
            Layer::AddConstant(ac) => {
                let c = scalar_or_vec(ac.constant(), b.len())?;
                if c.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                b = b + c;
            }
            Layer::SubConstant(sc) => {
                let c = scalar_or_vec(sc.constant(), b.len())?;
                if c.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                if sc.reverse {
                    w = w.mapv(|v| -v);
                    b = c - b;
                } else {
                    b = b - c;
                }
            }
            Layer::MulConstant(mc) => {
                let c = scalar_or_vec(mc.constant(), b.len())?;
                if c.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                for (i, s) in c.iter().enumerate() {
                    w.row_mut(i).mapv_inplace(|v| v * s);
                    b[i] *= s;
                }
            }
            Layer::DivConstant(dc) => {
                let c = scalar_or_vec(dc.constant(), b.len())?;
                if c.iter().any(|&v| !v.is_finite() || v == 0.0) {
                    return None;
                }
                for (i, s) in c.iter().enumerate() {
                    w.row_mut(i).mapv_inplace(|v| v / s);
                    b[i] /= s;
                }
            }
            Layer::ReLU(_) => {
                let n = w.nrows();
                let candidate = Affine { w, b };
                if !affine_is_finite(&candidate) {
                    return None;
                }
                layers.push(candidate);
                w = Array2::eye(n);
                b = Array1::zeros(n);
            }
            _ => return None, // unsupported op — fail closed
        }
        if w.nrows() != b.len() || w.iter().chain(b.iter()).any(|v| !v.is_finite()) {
            return None;
        }
    }
    let output = Affine { w, b };
    if !affine_is_finite(&output) {
        return None;
    }
    layers.push(output);
    Some(layers)
}

/// Concrete forward of an affine+ReLU chain at a point (f64).
fn forward_chain(layers: &[Affine], x: &Array1<f64>) -> Array1<f64> {
    let mut v = x.clone();
    for (li, layer) in layers.iter().enumerate() {
        v = layer.w.dot(&v) + &layer.b;
        if li + 1 < layers.len() {
            v.mapv_inplace(|t| t.max(0.0));
        }
    }
    v
}

/// One coupled backward pass over the box `[xlo, xhi]`, applying the JOINT
/// subadditive ReLU-difference relaxation at each hidden ReLU where beneficial
/// (`cap` = per-neuron symmetric δ range from the previous round; `None` = plain
/// coupled CROWN). Returns the per-hidden-stage per-neuron δ magnitudes (the
/// next round's `cap`) and, for each supplied objective over the diff output, a
/// sound LOWER bound on `obj·(f-g)` over the box.
///
/// This is the validated coupled-ReLU prototype backward, promoted to
/// production with the output stage generalized from `max|h|` to arbitrary
/// output objectives. The superseded offline probe was removed once this path
/// gained its own hermetic soundness coverage below.
fn coupled_pass(
    f: &[Affine],
    g: &[Affine],
    xlo: &Array1<f64>,
    xhi: &Array1<f64>,
    cap: Option<&Vec<Array1<f64>>>,
    objs: &[Array1<f64>],
) -> (Vec<Array1<f64>>, Vec<f64>) {
    let l = f.len();
    let conc_x = |cx: &Array1<f64>, k: f64, upper: bool| -> f64 {
        let mut acc = k;
        for j in 0..cx.len() {
            acc += if upper == (cx[j] >= 0.0) {
                cx[j] * xhi[j]
            } else {
                cx[j] * xlo[j]
            };
        }
        acc
    };
    let mut pf: Vec<(Array1<f64>, Array1<f64>)> = Vec::new();
    let mut pg: Vec<(Array1<f64>, Array1<f64>)> = Vec::new();
    // Back-substitute a coupled form (cf over post_f, cg over post_g) + const k
    // from layer `from` down to the input, applying the joint cap at each hidden
    // ReLU. Returns the concrete extremum in `upper` direction.
    let backprop = |mut cf: Array1<f64>,
                    mut cg: Array1<f64>,
                    mut k: f64,
                    from: usize,
                    upper: bool,
                    pf: &Vec<(Array1<f64>, Array1<f64>)>,
                    pg: &Vec<(Array1<f64>, Array1<f64>)>|
     -> f64 {
        let mut s = from;
        loop {
            let w = cf.len();
            let mut qf = Array1::<f64>::zeros(w);
            let mut qg = Array1::<f64>::zeros(w);
            let (flo, fhi) = &pf[s - 1];
            let (glo, ghi) = &pg[s - 1];
            for j in 0..w {
                let (mut cfj, mut cgj) = (cf[j], cg[j]);
                if let Some(caps) = cap {
                    if s - 1 < caps.len() {
                        let capj = caps[s - 1][j];
                        let mu_f = if flo[j] < 0.0 && fhi[j] > 0.0 {
                            fhi[j] * (-flo[j]) / (fhi[j] - flo[j])
                        } else {
                            0.0
                        };
                        let mu_g = if glo[j] < 0.0 && ghi[j] > 0.0 {
                            ghi[j] * (-glo[j]) / (ghi[j] - glo[j])
                        } else {
                            0.0
                        };
                        // Apply the joint relaxation ONLY where it is provably
                        // tighter than the two independent triangles.
                        if capj.is_finite()
                            && capj > 0.0
                            && cfj * cgj < 0.0
                            && capj / 2.0 < mu_f + mu_g
                        {
                            let coupled = cfj.abs().min(cgj.abs());
                            let c = coupled * cfj.signum();
                            qf[j] += c / 2.0;
                            qg[j] += -c / 2.0;
                            k += if upper {
                                c.abs() * capj / 2.0
                            } else {
                                -c.abs() * capj / 2.0
                            };
                            cfj -= c; // residual over relu_f
                            cgj += c; // residual over relu_g
                        }
                    }
                }
                for (c, q, lo, hi) in [
                    (cfj, &mut qf, flo[j], fhi[j]),
                    (cgj, &mut qg, glo[j], ghi[j]),
                ] {
                    if c == 0.0 {
                        continue;
                    }
                    if lo >= 0.0 {
                        q[j] += c;
                    } else if hi <= 0.0 {
                        // 0
                    } else {
                        let lam = hi / (hi - lo);
                        let mu = -lam * lo;
                        let sigma = if hi >= -lo { 1.0 } else { 0.0 };
                        if (c >= 0.0) == upper {
                            q[j] += c * lam;
                            k += c * mu;
                        } else {
                            q[j] += c * sigma;
                        }
                    }
                }
            }
            let li = s - 1;
            k += qf.dot(&f[li].b) + qg.dot(&g[li].b);
            if s == 1 {
                let cx = qf.dot(&f[li].w) + qg.dot(&g[li].w);
                return conc_x(&cx, k, upper);
            }
            cf = qf.dot(&f[li].w);
            cg = qg.dot(&g[li].w);
            s -= 1;
        }
    };
    // Forward: hidden pre-activation bounds pf/pg (via the capped backward of
    // each neuron), plus the per-stage per-neuron δ magnitudes (next cap).
    let mut deltas: Vec<Array1<f64>> = Vec::new();
    for s in 1..l {
        let li = s - 1;
        let rows = f[li].w.nrows();
        let mut flo = Array1::zeros(rows);
        let mut fhi = Array1::zeros(rows);
        let mut glo = Array1::zeros(rows);
        let mut ghi = Array1::zeros(rows);
        let mut dvec = Array1::<f64>::zeros(rows);
        for j in 0..rows {
            if s == 1 {
                let cxf = f[li].w.row(j).to_owned();
                fhi[j] = conc_x(&cxf, f[li].b[j], true);
                flo[j] = conc_x(&cxf, f[li].b[j], false);
                let cxg = g[li].w.row(j).to_owned();
                ghi[j] = conc_x(&cxg, g[li].b[j], true);
                glo[j] = conc_x(&cxg, g[li].b[j], false);
                let cxd = &cxf - &cxg;
                let dh = conc_x(&cxd, f[li].b[j] - g[li].b[j], true);
                let dl = conc_x(&cxd, f[li].b[j] - g[li].b[j], false);
                dvec[j] = dh.abs().max(dl.abs());
            } else {
                let zero = Array1::<f64>::zeros(f[li].w.ncols());
                let cf = f[li].w.row(j).to_owned();
                fhi[j] = backprop(cf.clone(), zero.clone(), f[li].b[j], s - 1, true, &pf, &pg);
                flo[j] = backprop(cf, zero.clone(), f[li].b[j], s - 1, false, &pf, &pg);
                let cg = g[li].w.row(j).to_owned();
                ghi[j] = backprop(zero.clone(), cg.clone(), g[li].b[j], s - 1, true, &pf, &pg);
                glo[j] = backprop(zero, cg, g[li].b[j], s - 1, false, &pf, &pg);
                let tf = f[li].w.row(j).to_owned();
                let tg = g[li].w.row(j).mapv(|v| -v);
                let bk = f[li].b[j] - g[li].b[j];
                let dh = backprop(tf.clone(), tg.clone(), bk, s - 1, true, &pf, &pg);
                let dl = backprop(tf, tg, bk, s - 1, false, &pf, &pg);
                dvec[j] = dh.abs().max(dl.abs());
            }
        }
        pf.push((flo, fhi));
        pg.push((glo, ghi));
        deltas.push(dvec);
    }
    // Output stage: for each objective, a sound LOWER bound on obj·(f-g).
    //   obj·y = (objᵀ W_f)·post_f - (objᵀ W_g)·post_g + obj·(b_f - b_g)
    // where W_f/b_f = f[l-1] (the output affine). Back-substitute from layer
    // l-1 in the min (`upper=false`) direction.
    let out = &f[l - 1];
    let outg = &g[l - 1];
    let mut obj_lowers = Vec::with_capacity(objs.len());
    if l == 1 {
        // No hidden ReLU: obj·(f-g) is affine in x; concretize directly.
        for obj in objs {
            let cf = obj.dot(&out.w);
            let cg = obj.dot(&outg.w);
            let k = obj.dot(&out.b) - obj.dot(&outg.b);
            let cx = &cf - &cg;
            obj_lowers.push(conc_x(&cx, k, false));
        }
    } else {
        for obj in objs {
            let cf = obj.dot(&out.w); // over post_{l-2}
            let cg = obj.dot(&outg.w).mapv(|v| -v);
            let k = obj.dot(&out.b) - obj.dot(&outg.b);
            obj_lowers.push(backprop(cf, cg, k, l - 1, false, &pf, &pg));
        }
    }
    (deltas, obj_lowers)
}

/// Sound LOWER bounds on `obj·(f-g)` over `[xlo, xhi]` for each objective, using
/// the iterated coupling-δ engine (fixed-point cap tightening). A row `(obj,
/// threshold)` is certified iff its returned lower bound `> threshold`.
fn iterated_row_lowers(
    f: &[Affine],
    g: &[Affine],
    xlo: &Array1<f64>,
    xhi: &Array1<f64>,
    objs: &[Array1<f64>],
    iters: usize,
) -> Vec<f64> {
    // Converge the per-neuron δ cap (objective-independent), then one final pass
    // computes the row lower bounds with the tightest cap.
    let (mut cap, _) = coupled_pass(f, g, xlo, xhi, None, &[]);
    for _ in 0..iters {
        let (next, _) = coupled_pass(f, g, xlo, xhi, Some(&cap), &[]);
        for (c, n) in cap.iter_mut().zip(next.iter()) {
            for (ci, ni) in c.iter_mut().zip(n.iter()) {
                *ci = ci.min(*ni);
            }
        }
    }
    let (_, lowers) = coupled_pass(f, g, xlo, xhi, Some(&cap), objs);
    lowers
}

/// The production coupled-δ leaf oracle. Holds the two extracted affine chains
/// (`f`, `g`) of the isomorphic pair; verified against the source forwards at
/// construction. Default-OFF: only armed via [`coupled_delta_oracle_from_env`].
pub(crate) struct CoupledDeltaOracle {
    f: Vec<Affine>,
    g: Vec<Affine>,
    graph_a: GraphNetwork,
    graph_b: GraphNetwork,
    expected_diff_graph: GraphNetwork,
    root_xlo: Array1<f64>,
    root_xhi: Array1<f64>,
    input_dim: usize,
    iters: usize,
}

impl CoupledDeltaOracle {
    /// Build from the two ISOMORPHIC source graphs (before stitching). Extracts
    /// each affine+ReLU chain and SPOT-CHECKS it against the graph forward at a
    /// handful of in-box points; any extraction or forward mismatch returns
    /// `None` (fail-closed, the oracle disarms).
    pub(crate) fn new(
        graph_a: &GraphNetwork,
        graph_b: &GraphNetwork,
        expected_diff_graph: &GraphNetwork,
        input_dim: usize,
        xlo: &Array1<f64>,
        xhi: &Array1<f64>,
        iters: usize,
    ) -> Option<Self> {
        let f = extract_affine_chain(graph_a, input_dim)?;
        let g = extract_affine_chain(graph_b, input_dim)?;
        // Isomorphic pair must share every affine/ReLU boundary and width.
        if !compatible_affine_chains(&f, &g, input_dim)
            || xlo.len() != input_dim
            || xhi.len() != input_dim
            || xlo
                .iter()
                .zip(xhi)
                .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return None;
        }
        // Spot-check the extracted chains reproduce the source graph forwards.
        if !spot_check_chain(graph_a, &f, input_dim, xlo, xhi)
            || !spot_check_chain(graph_b, &g, input_dim, xlo, xhi)
        {
            return None;
        }
        Some(Self {
            f,
            g,
            graph_a: graph_a.clone(),
            graph_b: graph_b.clone(),
            expected_diff_graph: expected_diff_graph.clone(),
            root_xlo: xlo.clone(),
            root_xhi: xhi.clone(),
            input_dim,
            iters,
        })
    }

    /// Authoritative verdict replay. Source IBP bounds are outward-rounded;
    /// every difference, product, and accumulation below is rounded down/up in
    /// the direction needed for a row LOWER bound. Any non-finite or shape miss
    /// declines rather than manufacturing a certificate.
    fn certified_ibp_confirms_all(&self, req: &GraphMipLeafRequest<'_>) -> bool {
        let Ok(a_out) = self.graph_a.propagate_ibp(req.input_bounds) else {
            return false;
        };
        let Ok(b_out) = self.graph_b.propagate_ibp(req.input_bounds) else {
            return false;
        };
        let (a_flat, b_flat) = (a_out.flatten(), b_out.flatten());
        let (Some(a_lo), Some(a_hi), Some(b_lo), Some(b_hi)) = (
            a_flat.lower().as_slice(),
            a_flat.upper().as_slice(),
            b_flat.lower().as_slice(),
            b_flat.upper().as_slice(),
        ) else {
            return false;
        };
        if a_lo.len() != a_hi.len()
            || a_lo.len() != b_lo.len()
            || a_lo.len() != b_hi.len()
            || a_lo
                .iter()
                .chain(a_hi)
                .chain(b_lo)
                .chain(b_hi)
                .any(|v| !v.is_finite())
        {
            return false;
        }

        req.rows.iter().all(|(coeffs, threshold)| {
            if coeffs.len() != a_lo.len()
                || !threshold.is_finite()
                || coeffs.iter().any(|v| !v.is_finite())
            {
                return false;
            }
            let mut lower = 0.0_f64;
            for (j, &coeff_f32) in coeffs.iter().enumerate() {
                if coeff_f32 == 0.0 {
                    continue;
                }
                let coeff = f64::from(coeff_f32);
                let diff_endpoint = if coeff > 0.0 {
                    (f64::from(a_lo[j]) - f64::from(b_hi[j])).next_down()
                } else {
                    (f64::from(a_hi[j]) - f64::from(b_lo[j])).next_up()
                };
                let term = (coeff * diff_endpoint).next_down();
                lower = (lower + term).next_down();
                if !lower.is_finite() {
                    return false;
                }
            }
            lower > f64::from(*threshold)
        })
    }
}

/// Confirm the extracted affine chain reproduces the source graph forward at a
/// set of deterministic in-box points (fail-closed extraction guard).
fn spot_check_chain(
    graph: &GraphNetwork,
    chain: &[Affine],
    input_dim: usize,
    xlo: &Array1<f64>,
    xhi: &Array1<f64>,
) -> bool {
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / ((1u64 << 53) as f64)
    };
    for t in 0..8 {
        let x = Array1::from_iter((0..input_dim).map(|j| {
            if t == 0 {
                xlo[j]
            } else if t == 1 {
                xhi[j]
            } else {
                xlo[j] + (xhi[j] - xlo[j]) * rnd()
            }
        }));
        let chain_y = forward_chain(chain, &x);
        // Concrete-point forward through the SOURCE graph (interval center after
        // every node), matched against the extracted chain.
        let degenerate: Vec<Bound> = x.iter().map(|&v| Bound::new(v as f32, v as f32)).collect();
        let input = match Verifier::bounds_to_tensor(&degenerate, None) {
            Ok(t) => t,
            Err(_) => return false,
        };
        let out = match graph.propagate_concrete_point(&input, None, None) {
            Ok(o) => o,
            Err(_) => return false,
        };
        let flat = out.flatten();
        let gy = match flat.lower().as_slice() {
            Some(s) => s,
            None => return false,
        };
        if gy.len() != chain_y.len() {
            return false;
        }
        for (a, b) in chain_y.iter().zip(gy.iter()) {
            if (a - f64::from(*b)).abs() > 1e-3 * (1.0 + a.abs()) {
                return false;
            }
        }
    }
    true
}

impl GraphMipLeafOracle for CoupledDeltaOracle {
    fn solve_leaf(&self, req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
        // Bind the proof object to the exact stitched difference graph for
        // which its source pair was constructed. Configured clones retain the
        // scope; structural mutations and output retargets mint a fresh one.
        if req.graph.cut_fold_scope() != self.expected_diff_graph.cut_fold_scope()
            || req.rows.is_empty()
            || req
                .deadline
                .is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
            return GraphMipLeafVerdict::Undecided;
        }
        // The input-split subdomain is described EXACTLY by its input box.
        let flat = req.input_bounds.flatten();
        let (Some(lo), Some(hi)) = (flat.lower().as_slice(), flat.upper().as_slice()) else {
            return GraphMipLeafVerdict::Undecided;
        };
        if lo.len() != self.input_dim
            || hi.len() != self.input_dim
            || lo.iter().zip(hi).enumerate().any(|(j, (&lo, &hi))| {
                !lo.is_finite()
                    || !hi.is_finite()
                    || lo > hi
                    || f64::from(lo) < self.root_xlo[j]
                    || f64::from(hi) > self.root_xhi[j]
            })
        {
            return GraphMipLeafVerdict::Undecided;
        }
        let xlo = Array1::from_iter(lo.iter().map(|&v| f64::from(v)));
        let xhi = Array1::from_iter(hi.iter().map(|&v| f64::from(v)));
        // Each row: objective coefficients over the diff-net output + threshold.
        // Verified iff min(obj·(f-g)) > threshold over the box.
        let objs: Vec<Array1<f64>> = req
            .rows
            .iter()
            .map(|(coeffs, _)| Array1::from_iter(coeffs.iter().map(|&v| f64::from(v))))
            .collect();
        if objs.iter().any(|o| {
            o.len() != self.f[self.f.len() - 1].w.nrows() || o.iter().any(|v| !v.is_finite())
        }) || req.rows.iter().any(|(_, threshold)| !threshold.is_finite())
        {
            return GraphMipLeafVerdict::Undecided;
        }
        let n = COUPLED_CONSULTS.fetch_add(1, Ordering::Relaxed) + 1;
        let lowers = iterated_row_lowers(&self.f, &self.g, &xlo, &xhi, &objs, self.iters);
        // Nearest-rounded f64 is useful as a fast candidate filter but is not a
        // certificate. A candidate becomes authoritative only if the separate
        // directed-rounded source-IBP replay below also proves every row.
        let mut candidate = true;
        for ((_, threshold), &low) in req.rows.iter().zip(lowers.iter()) {
            if !(low.is_finite() && low > f64::from(*threshold) + 1e-6) {
                candidate = false;
                break;
            }
        }
        if candidate && self.certified_ibp_confirms_all(req) {
            let v = COUPLED_VERIFIED.fetch_add(1, Ordering::Relaxed) + 1;
            // Periodic progress (survives a watchdog process-exit in the log).
            if v == 1 || v.is_multiple_of(50) {
                eprintln!(
                    "[coupled-δ] IBP-confirmed {v} edge domains ({n} consulted, depth={})",
                    req.depth
                );
            }
            return GraphMipLeafVerdict::VerifiedAllRows;
        }
        if n == 1 || n.is_multiple_of(200) {
            eprintln!(
                "[coupled-δ] {n} consulted, {} verified so far",
                COUPLED_VERIFIED.load(Ordering::Relaxed)
            );
        }
        GraphMipLeafVerdict::Undecided
    }
}

/// A composite leaf oracle: on either oracle surface, consults each inner oracle
/// in order and returns the first authoritative verdict. Used to stack the cheap
/// coupled-δ bound BEFORE the exact Graph-MIP edge solver (coupled-δ verifies
/// most edge domains without a MIP solve; the MIP handles the residual).
///
/// `Violated` authority belongs to the inner oracle that produced that exact
/// witness. An advisory violation from one inner must not borrow publication
/// authority from a different sibling. The loops below therefore filter every
/// unauthorized `Violated` before it crosses the composite boundary. As a
/// result, any `Violated` returned by the composite is publishable by
/// construction, while `VerifiedAllRows` retains its existing proof authority.
pub(crate) struct CompositeLeafOracle {
    oracles: Vec<Arc<dyn GraphMipLeafOracle>>,
}

impl CompositeLeafOracle {
    pub(crate) fn new(oracles: Vec<Arc<dyn GraphMipLeafOracle>>) -> Self {
        Self { oracles }
    }
}

impl GraphMipLeafOracle for CompositeLeafOracle {
    fn solve_input_leaf(&self, req: &GraphInputLeafRequest<'_>) -> GraphMipLeafVerdict {
        for oracle in &self.oracles {
            match oracle.solve_input_leaf(req) {
                GraphMipLeafVerdict::Undecided => continue,
                GraphMipLeafVerdict::Violated { .. } if !oracle.may_publish_violation_witness() => {
                    continue;
                }
                other => return other,
            }
        }
        GraphMipLeafVerdict::Undecided
    }

    fn solve_leaf(&self, req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
        for oracle in &self.oracles {
            match oracle.solve_leaf(req) {
                GraphMipLeafVerdict::Undecided => continue,
                GraphMipLeafVerdict::Violated { .. } if !oracle.may_publish_violation_witness() => {
                    continue;
                }
                other => return other,
            }
        }
        GraphMipLeafVerdict::Undecided
    }

    fn may_publish_violation_witness(&self) -> bool {
        // This does not aggregate sibling permissions. It describes the
        // composite's own output contract: both solve surfaces above suppress
        // every unauthorized inner violation before returning.
        true
    }
}

/// Number of fixed-point tightening rounds for the coupled-δ bound
/// (`NY_REL_COUPLED_DELTA_ITERS`, default 15). The probe converges by ~6 rounds;
/// 15 is a safe margin at negligible per-edge-domain cost.
fn coupled_delta_iters() -> usize {
    std::env::var("NY_REL_COUPLED_DELTA_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0 && n <= 200)
        .unwrap_or(15)
}

/// Arm the coupled-δ leaf oracle for the isomorphic relational lane. DEFAULT
/// OFF — opt in with `NY_REL_COUPLED_DELTA=1` (the 0-wrong discipline: validate
/// against the exact oracle + no corpus regression before any default flip).
/// `None` when disarmed, on any extraction/spot-check failure, or non-affine
/// pair — leaving the attached oracle stack exactly as without it.
pub(crate) fn coupled_delta_oracle_from_env(
    graph_a: &GraphNetwork,
    graph_b: &GraphNetwork,
    diff_graph: &GraphNetwork,
    input_bounds: &[Bound],
) -> Option<Arc<dyn GraphMipLeafOracle>> {
    if std::env::var("NY_REL_COUPLED_DELTA").ok().as_deref() != Some("1") {
        return None;
    }
    let input_dim = input_bounds.len();
    if input_dim == 0 {
        return None;
    }
    let xlo = Array1::from_iter(input_bounds.iter().map(|b| f64::from(b.lower())));
    let xhi = Array1::from_iter(input_bounds.iter().map(|b| f64::from(b.upper())));
    let iters = coupled_delta_iters();
    match CoupledDeltaOracle::new(graph_a, graph_b, diff_graph, input_dim, &xlo, &xhi, iters) {
        Some(oracle) => {
            eprintln!(
                "coupled-δ leaf oracle ARMED (input_dim={input_dim}, layers={}, iters={iters})",
                oracle.f.len()
            );
            Some(Arc::new(oracle))
        }
        None => {
            // Fail-closed: extraction or the source-forward spot-check declined.
            eprintln!(
                "coupled-δ leaf oracle NOT armed (chain extraction / spot-check declined); \
                 lane unchanged"
            );
            None
        }
    }
}

#[cfg(test)]
mod extraction_tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_propagate::layers::{LinearLayer, ReLULayer};
    use ny_propagate::GraphNode;
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer::new()), vec![input.to_string()])
    }

    #[test]
    fn affine_extraction_requires_a_whole_source_to_output_chain() {
        let mut chain = GraphNetwork::new();
        chain.add_node(relu("r1", NETWORK_INPUT));
        chain.add_node(relu("r2", "r1"));
        chain.set_output("r2");
        assert!(extract_affine_chain(&chain, 2).is_some());

        let mut branched = GraphNetwork::new();
        branched.add_node(relu("root", NETWORK_INPUT));
        branched.add_node(relu("left", "root"));
        branched.add_node(relu("right", "root"));
        branched.set_output("left");
        assert!(extract_affine_chain(&branched, 2).is_none());

        let mut skipped = GraphNetwork::new();
        skipped.add_node(relu("r1", NETWORK_INPUT));
        skipped.add_node(relu("r2", NETWORK_INPUT));
        skipped.set_output("r2");
        assert!(extract_affine_chain(&skipped, 2).is_none());

        let mut retargeted = chain.clone();
        retargeted.set_output("r1");
        assert!(extract_affine_chain(&retargeted, 2).is_none());
    }

    fn scalar_linear_graph(name: &str, weight: f32) -> GraphNetwork {
        let linear = LinearLayer::new(arr2(&[[weight]]), Some(arr1(&[0.0]))).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(name, Layer::Linear(linear)));
        graph.set_output(name);
        graph
    }

    struct CompositeInputProbe {
        consults: Arc<AtomicUsize>,
        verifies: bool,
    }

    impl GraphMipLeafOracle for CompositeInputProbe {
        fn solve_input_leaf(&self, _req: &GraphInputLeafRequest<'_>) -> GraphMipLeafVerdict {
            self.consults.fetch_add(1, AtomicOrdering::SeqCst);
            if self.verifies {
                GraphMipLeafVerdict::VerifiedAllRows
            } else {
                GraphMipLeafVerdict::Undecided
            }
        }

        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            panic!("input-leaf composite forwarding must not call the legacy surface")
        }
    }

    #[test]
    fn composite_forwards_input_leaf_until_first_authoritative_verdict() {
        let first_consults = Arc::new(AtomicUsize::new(0));
        let second_consults = Arc::new(AtomicUsize::new(0));
        let third_consults = Arc::new(AtomicUsize::new(0));
        let composite = CompositeLeafOracle::new(vec![
            Arc::new(CompositeInputProbe {
                consults: first_consults.clone(),
                verifies: false,
            }),
            Arc::new(CompositeInputProbe {
                consults: second_consults.clone(),
                verifies: true,
            }),
            Arc::new(CompositeInputProbe {
                consults: third_consults.clone(),
                verifies: true,
            }),
        ]);
        let graph = scalar_linear_graph("out", 1.0);
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input box");
        let objectives = arr2(&[[1.0_f32]]);
        let thresholds = [-0.1_f32];
        let advisory_objective_bounds = [(-1.0_f32, 1.0_f32)];
        let clause_sizes = [1usize];
        let request = GraphInputLeafRequest {
            graph: &graph,
            input_bounds: &input,
            objectives: &objectives,
            thresholds: &thresholds,
            advisory_objective_bounds: &advisory_objective_bounds,
            clause_sizes: &clause_sizes,
            depth: 0,
            deadline: None,
        };

        assert!(matches!(
            composite.solve_input_leaf(&request),
            GraphMipLeafVerdict::VerifiedAllRows
        ));
        assert_eq!(first_consults.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(second_consults.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(third_consults.load(AtomicOrdering::SeqCst), 0);
    }

    struct CompositeViolationProbe {
        consults: Arc<AtomicUsize>,
        marker: Option<f32>,
        may_publish: bool,
    }

    impl GraphMipLeafOracle for CompositeViolationProbe {
        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            self.consults.fetch_add(1, AtomicOrdering::SeqCst);
            match self.marker {
                Some(marker) => GraphMipLeafVerdict::Violated {
                    witness: vec![marker],
                    output: vec![marker],
                },
                None => GraphMipLeafVerdict::Undecided,
            }
        }

        fn may_publish_violation_witness(&self) -> bool {
            self.may_publish
        }
    }

    #[test]
    fn composite_binds_violation_authority_to_the_producing_oracle() {
        let advisory_consults = Arc::new(AtomicUsize::new(0));
        let publishing_consults = Arc::new(AtomicUsize::new(0));
        let composite = CompositeLeafOracle::new(vec![
            Arc::new(CompositeViolationProbe {
                consults: advisory_consults.clone(),
                marker: Some(1.0),
                may_publish: false,
            }),
            Arc::new(CompositeViolationProbe {
                consults: publishing_consults.clone(),
                marker: Some(2.0),
                may_publish: true,
            }),
        ]);
        let graph = scalar_linear_graph("out", 1.0);
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input box");
        let node_bounds = HashMap::new();
        let request = GraphMipLeafRequest {
            graph: &graph,
            input_bounds: &input,
            node_bounds: &node_bounds,
            splits: Vec::new(),
            rows: vec![(vec![1.0], 0.0)],
            depth: 1,
            deadline: None,
        };

        match composite.solve_leaf(&request) {
            GraphMipLeafVerdict::Violated { witness, output } => {
                assert_eq!(witness, vec![2.0]);
                assert_eq!(output, vec![2.0]);
            }
            other => panic!("expected the publishing sibling's violation, got {other:?}"),
        }
        assert!(composite.may_publish_violation_witness());
        assert_eq!(advisory_consults.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(publishing_consults.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn composite_does_not_export_an_advisory_only_violation() {
        let advisory_consults = Arc::new(AtomicUsize::new(0));
        let fallback_consults = Arc::new(AtomicUsize::new(0));
        let composite = CompositeLeafOracle::new(vec![
            Arc::new(CompositeViolationProbe {
                consults: advisory_consults.clone(),
                marker: Some(3.0),
                may_publish: false,
            }),
            Arc::new(CompositeViolationProbe {
                consults: fallback_consults.clone(),
                marker: None,
                may_publish: true,
            }),
        ]);
        let graph = scalar_linear_graph("out", 1.0);
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input box");
        let node_bounds = HashMap::new();
        let request = GraphMipLeafRequest {
            graph: &graph,
            input_bounds: &input,
            node_bounds: &node_bounds,
            splits: Vec::new(),
            rows: vec![(vec![1.0], 0.0)],
            depth: 1,
            deadline: None,
        };

        assert!(matches!(
            composite.solve_leaf(&request),
            GraphMipLeafVerdict::Undecided
        ));
        assert_eq!(advisory_consults.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(fallback_consults.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn verdict_requires_directed_ibp_replay_and_exact_graph_scope() {
        let graph_a = scalar_linear_graph("a", 1.0);
        let graph_b = scalar_linear_graph("b", 0.0);
        let diff = scalar_linear_graph("diff", 1.0);
        let xlo = arr1(&[1.0_f64]);
        let xhi = arr1(&[2.0_f64]);
        let oracle = CoupledDeltaOracle::new(&graph_a, &graph_b, &diff, 1, &xlo, &xhi, 1)
            .expect("valid scalar pair");
        let input = BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0]).into_dyn())
            .expect("valid input box");
        let node_bounds = HashMap::new();
        let request = GraphMipLeafRequest {
            graph: &diff,
            input_bounds: &input,
            node_bounds: &node_bounds,
            splits: Vec::new(),
            rows: vec![(vec![1.0], 0.5)],
            depth: 1,
            deadline: None,
        };
        assert!(matches!(
            oracle.solve_leaf(&request),
            GraphMipLeafVerdict::VerifiedAllRows
        ));

        // Even a corrupted/over-optimistic coupled candidate cannot authorize
        // a verdict that the independent outward IBP replay does not prove.
        let mut forged_candidate =
            CoupledDeltaOracle::new(&graph_a, &graph_b, &diff, 1, &xlo, &xhi, 1).unwrap();
        forged_candidate.f[0].b[0] = 10.0;
        let forged_request = GraphMipLeafRequest {
            graph: &diff,
            input_bounds: &input,
            node_bounds: &node_bounds,
            splits: Vec::new(),
            rows: vec![(vec![1.0], 5.0)],
            depth: 1,
            deadline: None,
        };
        assert!(matches!(
            forged_candidate.solve_leaf(&forged_request),
            GraphMipLeafVerdict::Undecided
        ));

        let unrelated = scalar_linear_graph("other", 1.0);
        let wrong_graph_request = GraphMipLeafRequest {
            graph: &unrelated,
            ..request
        };
        assert!(matches!(
            oracle.solve_leaf(&wrong_graph_request),
            GraphMipLeafVerdict::Undecided
        ));

        let expanded =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
        let expanded_request = GraphMipLeafRequest {
            graph: &diff,
            input_bounds: &expanded,
            node_bounds: &node_bounds,
            splits: Vec::new(),
            rows: vec![(vec![1.0], 0.5)],
            depth: 1,
            deadline: None,
        };
        assert!(matches!(
            oracle.solve_leaf(&expanded_request),
            GraphMipLeafVerdict::Undecided
        ));
    }
}

#[cfg(all(test, feature = "mip"))]
mod tests {
    use super::*;

    /// The last-holdout isomorphic pairs (network + perturbation).
    const PAIRS: &[(&str, &str, &str)] = &[
        (
            "instance_5",
            "ACASXU_run2a_4_5_batch_2000.onnx",
            "ACASXU_run2a_4_5_batch_2000_perturbed_5.onnx",
        ),
        (
            "instance_10",
            "ACASXU_run2a_3_7_batch_2000.onnx",
            "ACASXU_run2a_3_7_batch_2000_perturbed_10.onnx",
        ),
        (
            "instance_26",
            "ACASXU_run2a_5_1_batch_2000.onnx",
            "ACASXU_run2a_5_1_batch_2000_perturbed_26.onnx",
        ),
        (
            "instance_28",
            "ACASXU_run2a_4_1_batch_2000.onnx",
            "ACASXU_run2a_4_1_batch_2000_perturbed_28.onnx",
        ),
        (
            "instance_39",
            "ACASXU_run2a_3_1_batch_2000.onnx",
            "ACASXU_run2a_3_1_batch_2000_perturbed_39.onnx",
        ),
    ];

    /// Sampled minimum of `obj·(f-g)` over the box (the sound bound must sit at
    /// or below this for every objective).
    fn sampled_min_obj(
        f: &[Affine],
        g: &[Affine],
        xlo: &Array1<f64>,
        xhi: &Array1<f64>,
        obj: &Array1<f64>,
        seed: u64,
    ) -> f64 {
        let dim = xlo.len();
        let mut state = seed | 1;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / ((1u64 << 53) as f64)
        };
        let mut best = f64::INFINITY;
        for t in 0..200_000usize {
            let x = Array1::from_iter((0..dim).map(|j| {
                // Corners get heavy weight (extrema of an affine-ish response
                // often sit there), plus uniform interior.
                if t < (1 << dim) {
                    if (t >> j) & 1 == 1 {
                        xhi[j]
                    } else {
                        xlo[j]
                    }
                } else {
                    xlo[j] + (xhi[j] - xlo[j]) * rnd()
                }
            }));
            let yf = forward_chain(f, &x);
            let yg = forward_chain(g, &x);
            let val: f64 = obj
                .iter()
                .zip(yf.iter().zip(yg.iter()))
                .map(|(o, (a, b))| o * (a - b))
                .sum();
            best = best.min(val);
        }
        best
    }

    /// SOUNDNESS (the 0-wrong moat): the iterated coupled-δ directional bound
    /// must be a valid LOWER bound on `min obj·(f-g)` for EVERY band objective at
    /// EVERY box scale — otherwise the oracle could certify a violated domain.
    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn coupled_row_lower_is_sound() {
        let base = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/vnncomp2026_benchmarks/benchmarks/isomorphic_acasxu_2026/2.0",
        ));
        assert!(
            base.is_dir(),
            "external VNN-COMP 2026 relational fixtures missing at {}; run \
             benchmarks/vnncomp2026_benchmarks/setup.sh",
            base.display()
        );
        let mut checked = 0usize;
        for (stem, f_name, g_name) in PAIRS {
            let fp = base.join("onnx/original").join(f_name);
            let gp = base.join("onnx/perturbed").join(g_name);
            assert!(
                fp.is_file() && gp.is_file(),
                "{stem}: external ONNX fixtures missing (f={}, g={}); run \
                 benchmarks/vnncomp2026_benchmarks/setup.sh",
                fp.display(),
                gp.display()
            );
            let graph_f = super::super::vnncomp::load_graph_network(&fp).expect("load f");
            let graph_g = super::super::vnncomp::load_graph_network(&gp).expect("load g");
            let vnnlib = base.join("vnnlib").join(format!("{stem}.vnnlib"));
            let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib).expect("vnnlib");
            let dual = spec.dual_network.expect("dual");
            let bounds =
                super::super::vnncomp::bounds_from_f64(&dual.f_input_bounds).expect("bounds");
            let dim = bounds.len();
            let full_lo = Array1::from_iter(bounds.iter().map(|b| f64::from(b.lower())));
            let full_hi = Array1::from_iter(bounds.iter().map(|b| f64::from(b.upper())));
            let f = extract_affine_chain(&graph_f, dim).expect("f chain");
            let g = extract_affine_chain(&graph_g, dim).expect("g chain");
            let out_dim = f[f.len() - 1].w.nrows();
            // Extraction must reproduce the source forwards (the oracle's guard).
            assert!(
                spot_check_chain(&graph_f, &f, dim, &full_lo, &full_hi),
                "{stem}: f chain spot-check failed"
            );
            assert!(
                spot_check_chain(&graph_g, &g, dim, &full_lo, &full_hi),
                "{stem}: g chain spot-check failed"
            );
            // Band objectives: ±e_j over the diff-net output.
            let mut objs = Vec::new();
            for j in 0..out_dim {
                let mut a = Array1::<f64>::zeros(out_dim);
                a[j] = 1.0;
                objs.push(a.clone());
                objs.push(a.mapv(|v| -v));
            }
            // Center sub-boxes at several scales; soundness must hold at each.
            for &denom in &[1.0_f64, 8.0, 64.0] {
                let mut xlo = Array1::zeros(dim);
                let mut xhi = Array1::zeros(dim);
                for d in 0..dim {
                    let c = 0.5 * (full_lo[d] + full_hi[d]);
                    let half = 0.5 * (full_hi[d] - full_lo[d]) / denom;
                    xlo[d] = c - half;
                    xhi[d] = c + half;
                }
                let lowers = iterated_row_lowers(&f, &g, &xlo, &xhi, &objs, 15);
                for (oi, obj) in objs.iter().enumerate() {
                    let smin = sampled_min_obj(&f, &g, &xlo, &xhi, obj, 0x1234 + oi as u64);
                    let low = lowers[oi];
                    // Sound lower bound: low <= true min. Allow a tiny numeric
                    // slack for the sampler (it only APPROACHES the true min).
                    assert!(
                        low.is_finite() && low <= smin + 1e-6 * (1.0 + smin.abs()),
                        "{stem} box=1/{denom} obj#{oi}: UNSOUND lower={low} > sampled_min={smin}"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 0, "no holdout pairs exercised");
        eprintln!("[coupled-δ soundness] {checked} (pair,box) configs — all sound");
    }
}
