// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Margin functionals over the head (#twinwall).
//!
//! Port of `core.py::MarginBatch` + the head-gate refresh semantics verified
//! in `verify2/a2_code_trace.md`: the y-box always comes from the DOMAIN'S OWN
//! backward pass (or a parent superset box), head gates are DeepPoly gates on
//! that box (repaired outward in certified mode), and three sound lower-bound
//! paths are combined per class: the interval-exact head path `m1`, the
//! row-composed via-y path `m2v`, and the direct margin-seeded backward bound
//! (computed by the caller through [`super::engine::BackwardEngine`]).

use ndarray::Array2;
use ny_core::{NyError, Result};

use super::engine::{BackwardEngine, PassOut, Seed};
use super::net::TwinNet;
use super::root::{gates_from_box, repair_upper_lines, RetainedLayer, RootGates};
use super::rounding::{certify_up, gamma_n, next_down, next_up, slack16, RoundMode, UNIT};

/// Margin rows `m_r = Y_t - Y_{adv[r]}` split into positive/negative parts.
pub struct MarginBatch {
    /// True class.
    pub t: usize,
    /// Adversarial classes (row order).
    pub adv: Vec<usize>,
    /// `max(w, 0)` — (nf, n_y) row-major.
    pub wp: Vec<f64>,
    /// `min(w, 0)` — (nf, n_y) row-major.
    pub wn: Vec<f64>,
    /// Bias `b2[t] - b2[j]` per row.
    pub cst: Vec<f64>,
    /// Head width.
    pub n_y: usize,
}

impl MarginBatch {
    /// Build for a class set.
    pub fn new(net: &TwinNet, t: usize, adv: &[usize]) -> Result<Self> {
        let (w2, b2, (n_out, n_y)) = net.gemm2();
        if t >= n_out || adv.iter().any(|&j| j >= n_out) {
            return Err(NyError::InvalidSpec(
                "margin_row: class out of range".into(),
            ));
        }
        let nf = adv.len();
        let mut wp = vec![0.0; nf * n_y];
        let mut wn = vec![0.0; nf * n_y];
        let mut cst = vec![0.0; nf];
        for (r, &j) in adv.iter().enumerate() {
            for k in 0..n_y {
                let w = w2[t * n_y + k] - w2[j * n_y + k];
                wp[r * n_y + k] = w.max(0.0);
                wn[r * n_y + k] = w.min(0.0);
            }
            cst[r] = b2[t] - b2[j];
        }
        Ok(Self {
            t,
            adv: adv.to_vec(),
            wp,
            wn,
            cst,
            n_y,
        })
    }

    /// Number of margin rows.
    pub fn nf(&self) -> usize {
        self.adv.len()
    }
}

/// A y-box (per head neuron pre-activation bounds).
#[derive(Clone)]
pub struct YBox {
    /// Lower.
    pub ly: Vec<f64>,
    /// Upper.
    pub uy: Vec<f64>,
}

impl YBox {
    /// Concretize a y-row pack into a box.
    pub fn from_rows(eng: &BackwardEngine<'_>, al: &PassOut, au: &PassOut) -> Self {
        Self {
            ly: eng.concretize_lower(al),
            uy: eng.concretize_upper(au),
        }
    }

    /// Apply head sign-fix clamps (`(neuron, dir)`), Python-parity semantics.
    pub fn clamp(&mut self, head_clamp: &[(usize, i8)]) {
        for &(i, d) in head_clamp {
            if d > 0 {
                self.ly[i] = self.ly[i].max(0.0);
            } else {
                self.uy[i] = self.uy[i].min(0.0);
            }
        }
    }

    /// Intersect with another valid box for the same domain.
    pub fn intersect(&mut self, other_ly: &[f64], other_uy: &[f64]) {
        for i in 0..self.ly.len() {
            self.ly[i] = self.ly[i].max(other_ly[i]);
            self.uy[i] = self.uy[i].min(other_uy[i]);
        }
    }

    /// Python-parity emptiness test (`ly > uy + 1e-12` anywhere). With
    /// certified outward boxes this is CONSERVATIVE (declares empty only when
    /// provably empty by > 1e-12).
    pub fn is_empty(&self) -> bool {
        self.ly.iter().zip(&self.uy).any(|(l, u)| *l > *u + 1e-12)
    }
}

/// Head gates on a y-box (repaired in outward mode).
pub struct HeadGates {
    /// Lower slopes (0/1).
    pub alpha: Vec<f64>,
    /// Upper slopes.
    pub s: Vec<f64>,
    /// Upper intercepts.
    pub c: Vec<f64>,
}

/// DeepPoly head gates from a (clamped) y-box.
pub fn head_gates(ybox: &YBox, mode: RoundMode) -> HeadGates {
    let (alpha, mut s, mut c) = gates_from_box(&ybox.ly, &ybox.uy);
    if mode.outward() {
        repair_upper_lines(&ybox.ly, &ybox.uy, &mut s, &mut c);
    }
    HeadGates { alpha, s, c }
}

/// Margin seed through head gates + the two head-level bound paths.
pub struct MarginSeed {
    /// Seed for the backward engine (`(n_y, nf)` + certified error).
    pub seed: Seed,
    /// Per-row constant `wn @ c + cst`.
    pub cst: Vec<f64>,
    /// Certified error on `cst` (outward).
    pub cst_err: Vec<f64>,
    /// Interval-exact head path lower bound per row (certified lower in
    /// outward mode).
    pub m1: Vec<f64>,
    /// `max(uy, 0)` (head-child m1 adjustments need it).
    pub zu: Vec<f64>,
}

/// Build the margin seed `S = Wp*alpha + Wn*s` (+ const, + m1) for gates over
/// a y-box. Mirrors `probe_backward.py::margin_seed` + `MarginBatch.bounds`
/// path (i).
pub fn margin_seed(
    mb: &MarginBatch,
    gates: &HeadGates,
    ybox: &YBox,
    mode: RoundMode,
) -> MarginSeed {
    let nf = mb.nf();
    let n_y = mb.n_y;
    let mut s_mat = Array2::<f64>::zeros((n_y, nf));
    let mut e_mat = mode.outward().then(|| Array2::<f64>::zeros((n_y, nf)));
    for j in 0..n_y {
        for r0 in 0..nf {
            let v = mb.wp[r0 * n_y + j] * gates.alpha[j] + mb.wn[r0 * n_y + j] * gates.s[j];
            s_mat[[j, r0]] = v;
            if let Some(e) = e_mat.as_mut() {
                let mag = (mb.wp[r0 * n_y + j] * gates.alpha[j]).abs()
                    + (mb.wn[r0 * n_y + j] * gates.s[j]).abs();
                e[[j, r0]] = slack16(3.0 * UNIT * mag);
            }
        }
    }
    let gam = gamma_n(n_y + 8);
    let mut cst = vec![0.0; nf];
    let mut cst_err = vec![0.0; nf];
    let mut m1 = vec![0.0; nf];
    let zl: Vec<f64> = ybox.ly.iter().map(|v| v.max(0.0)).collect();
    let zu: Vec<f64> = ybox.uy.iter().map(|v| v.max(0.0)).collect();
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        let mut acc_c = 0.0;
        let mut abs_c = 0.0;
        let mut acc_m = 0.0;
        let mut abs_m = 0.0;
        for j in 0..n_y {
            acc_c += wn[j] * gates.c[j];
            abs_c += wn[j].abs() * gates.c[j];
            acc_m += wp[j] * zl[j] + wn[j] * zu[j];
            abs_m += wp[j] * zl[j] + wn[j].abs() * zu[j];
        }
        cst[r0] = acc_c + mb.cst[r0];
        m1[r0] = acc_m + mb.cst[r0];
        if mode.outward() {
            cst_err[r0] = slack16(gam * (abs_c + mb.cst[r0].abs()));
            let sl = next_up(gam * (abs_m + mb.cst[r0].abs()));
            m1[r0] = next_down(next_down(m1[r0] - sl));
        }
    }
    MarginSeed {
        seed: Seed { s: s_mat, e: e_mat },
        cst,
        cst_err,
        m1,
        zu,
    }
}

/// Direct-path per-class bounds from a completed margin-seeded pass:
/// `concretize_lower(pass) + cst` (certified in outward mode).
pub fn per_class_direct(
    eng: &BackwardEngine<'_>,
    pass: &PassOut,
    ms: &MarginSeed,
    rows: std::ops::Range<usize>,
) -> Vec<f64> {
    let low = eng.concretize_lower(pass);
    rows.map(|r0| {
        let v = low[r0] + ms.cst[r0];
        if eng.root.mode.outward() {
            next_down(next_down(v - next_up(ms.cst_err[r0])))
        } else {
            v
        }
    })
    .collect()
}

/// Precomputed per-y-row magnitude/error dots for the via-y composition:
/// `abs_dot[j] = |row_j| . xabs + |bias_j|`, `err_dot[j] = E_j . xabs + eb_j`.
pub struct RowDots {
    /// Magnitude dot per y-row.
    pub abs_dot: Vec<f64>,
    /// Error-penalty dot per y-row.
    pub err_dot: Vec<f64>,
}

/// Compute [`RowDots`] for a y-row pack lane.
pub fn row_dots(root: &RootGates, pack: &PassOut) -> RowDots {
    let n_in = root.mid.len();
    let r = pack.a.ncols();
    let asl = pack.a.as_slice().expect("standard layout");
    let esl = pack.e.as_ref().map(|e| e.as_slice().expect("layout"));
    let mut abs_dot = vec![0.0; r];
    let mut err_dot = vec![0.0; r];
    for i in 0..n_in {
        let xa = root.xabs[i];
        let arow = &asl[i * r..(i + 1) * r];
        for (j, av) in arow.iter().enumerate() {
            abs_dot[j] += av.abs() * xa;
        }
        if let Some(es) = esl {
            let erow = &es[i * r..(i + 1) * r];
            for (j, ev) in erow.iter().enumerate() {
                err_dot[j] += ev * xa;
            }
        }
    }
    for j in 0..r {
        abs_dot[j] += pack.b[j].abs();
        err_dot[j] += pack.eb[j];
    }
    RowDots { abs_dot, err_dot }
}

/// Via-y row-composed lower bounds (`MarginBatch.bounds` path (ii)): compose
/// the margin rows with the pack's y-rows through the head gates, concretize.
/// Certified in outward mode via the pack's error lanes + gamma envelopes.
#[allow(clippy::too_many_arguments)]
pub fn compose_viay(
    eng: &BackwardEngine<'_>,
    mb: &MarginBatch,
    gates: &HeadGates,
    al: &PassOut,
    au: &PassOut,
    al_dots: &RowDots,
    au_dots: &RowDots,
    mode: RoundMode,
) -> Vec<f64> {
    let nf = mb.nf();
    let n_y = mb.n_y;
    let root = eng.root;
    let n_in = root.mid.len();
    let asl = al.a.as_slice().expect("standard layout");
    let usl = au.a.as_slice().expect("standard layout");
    let gam_row = gamma_n(2 * n_y + 8);
    let gam_con = gamma_n(n_in + 16);
    let mut out = vec![0.0; nf];
    let mut rowbuf = vec![0.0; n_in];
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        // Composition coefficients cA = wp*alpha (>=0 side on lower rows),
        // cB = wn*s (<=0 side on upper rows).
        let ca: Vec<f64> = (0..n_y).map(|j| wp[j] * gates.alpha[j]).collect();
        let cb: Vec<f64> = (0..n_y).map(|j| wn[j] * gates.s[j]).collect();
        // rows_r[k] = sum_j ca[j]*Al[k][j] + cb[j]*Au[k][j]
        for (k, rb) in rowbuf.iter_mut().enumerate() {
            let arow = &asl[k * n_y..(k + 1) * n_y];
            let urow = &usl[k * n_y..(k + 1) * n_y];
            let mut acc = 0.0;
            for j in 0..n_y {
                acc += ca[j] * arow[j] + cb[j] * urow[j];
            }
            *rb = acc;
        }
        let mut bias = 0.0;
        let mut cterm = 0.0;
        for j in 0..n_y {
            bias += ca[j] * al.b[j] + cb[j] * au.b[j];
            cterm += wn[j] * gates.c[j];
        }
        // Concretize rowbuf over the box.
        let mut v = 0.0;
        let mut rr = 0.0;
        for (k, rb) in rowbuf.iter().enumerate() {
            v += rb * root.mid[k];
            rr += rb.abs() * root.rad[k];
        }
        let raw = v - rr + bias + cterm + mb.cst[r0];
        if !mode.outward() {
            out[r0] = raw;
            continue;
        }
        // Certified penalty: pack errors + seed-coefficient rounding +
        // row-computation gamma + concretization gamma + cterm/cst rounding.
        let mut pen = 0.0;
        let mut tmag = 0.0;
        for j in 0..n_y {
            let ea = slack16(3.0 * UNIT * ca[j].abs());
            let ebc = slack16(3.0 * UNIT * cb[j].abs());
            pen += (ca[j].abs() + ea) * al_dots.err_dot[j]
                + ea * al_dots.abs_dot[j]
                + (cb[j].abs() + ebc) * au_dots.err_dot[j]
                + ebc * au_dots.abs_dot[j];
            tmag += ca[j].abs() * al_dots.abs_dot[j]
                + cb[j].abs() * au_dots.abs_dot[j]
                + wn[j].abs() * gates.c[j];
        }
        tmag += mb.cst[r0].abs();
        let slack = next_up(
            certify_up(pen, 1e-13) + next_up((gam_row + gam_con) * certify_up(tmag, 1e-13)),
        );
        out[r0] = next_down(next_down(raw - slack));
    }
    out
}

/// Exact single-head-gate variant bound (RANKER ONLY — round-to-nearest in
/// both modes; candidate ordering never affects soundness). Port of
/// `core.py::MarginBatch.head_variant` on materialized margin rows.
pub struct VariantState {
    /// Materialized margin rows (nf x n_in), row-major.
    pub rows: Vec<f64>,
    /// Row biases (incl. composed y-row biases).
    pub rowb: Vec<f64>,
    /// m1 per row (nearest).
    pub m1: Vec<f64>,
    /// `wn @ c` per row.
    pub cterm: Vec<f64>,
}

/// Materialize the variant-ranker state (`rows = (wp*alpha)@Al + (wn*s)@Au`).
pub fn variant_state(
    mb: &MarginBatch,
    gates: &HeadGates,
    ybox: &YBox,
    al: &PassOut,
    au: &PassOut,
) -> VariantState {
    let nf = mb.nf();
    let n_y = mb.n_y;
    let n_in = al.a.nrows();
    let asl = al.a.as_slice().expect("standard layout");
    let usl = au.a.as_slice().expect("standard layout");
    let mut rows = vec![0.0; nf * n_in];
    let mut rowb = vec![0.0; nf];
    let mut m1 = vec![0.0; nf];
    let mut cterm = vec![0.0; nf];
    let zl: Vec<f64> = ybox.ly.iter().map(|v| v.max(0.0)).collect();
    let zu: Vec<f64> = ybox.uy.iter().map(|v| v.max(0.0)).collect();
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        let ca: Vec<f64> = (0..n_y).map(|j| wp[j] * gates.alpha[j]).collect();
        let cb: Vec<f64> = (0..n_y).map(|j| wn[j] * gates.s[j]).collect();
        let rslice = &mut rows[r0 * n_in..(r0 + 1) * n_in];
        for (k, rv) in rslice.iter_mut().enumerate() {
            let arow = &asl[k * n_y..(k + 1) * n_y];
            let urow = &usl[k * n_y..(k + 1) * n_y];
            let mut acc = 0.0;
            for j in 0..n_y {
                acc += ca[j] * arow[j] + cb[j] * urow[j];
            }
            *rv = acc;
        }
        for j in 0..n_y {
            rowb[r0] += ca[j] * al.b[j] + cb[j] * au.b[j];
            m1[r0] += wp[j] * zl[j] + wn[j] * zu[j];
            cterm[r0] += wn[j] * gates.c[j];
        }
        m1[r0] += mb.cst[r0];
    }
    VariantState {
        rows,
        rowb,
        m1,
        cterm,
    }
}

/// Tier-0 trunk variant bound (#epoch-bab; RANKER ONLY — nearest mode, the
/// trunk sibling of [`head_variant`]): estimated `min_f` child bound after
/// piece-fixing trunk neuron `ret.idx[ri]` of layer `li` in direction `dir`.
///
/// Mechanism (design doc `docs/EPOCH_BAB_DESIGN.md` §2.2): in the parent's
/// direct margin-seeded pass, the neuron's relu contributed through the lane
/// `g1(z) = (v⁺α + v⁻s)·z + v⁻c` per margin row (lower lane). The fixed
/// child replaces it with `g2` (identity for `dir > 0`, zero for `dir < 0`).
/// The coefficient delta `d = coeff(g2) − coeff(g1)` multiplies the neuron's
/// PRE-activation `z`, which the retained tableau rows sandwich over the
/// input (`A_l·x̂ ≤ z ≤ A_u·x̂`), so `d·z ≥ d·(A_side·x̂)` pointwise with
/// `A_side = A_l` when `d ≥ 0` else `A_u`. Substituting yields a valid
/// (real-arithmetic) child lower-bound row that is re-concretized JOINTLY
/// with the parent row — the joint |·| cancellation is what makes the score
/// discriminative (the single-term interval estimate is identically zero by
/// the chord identities; see the design doc). O(nf·n_in) per candidate.
///
/// The estimate never feeds a verdict: it only orders candidates for the
/// exact Tier-1 pass (`bab.rs::score_candidates`), exactly like
/// [`head_variant`] for head candidates.
#[allow(clippy::too_many_arguments)]
pub fn trunk_variant(
    root: &RootGates,
    ret: &RetainedLayer,
    li: usize,
    ri: usize,
    vrow: &[f64],
    pass: &PassOut,
    ms: &MarginSeed,
    dir: i8,
) -> f64 {
    let nf = vrow.len();
    let n_in = root.mid.len();
    let naug = ret.naug;
    debug_assert_eq!(naug, n_in + 1);
    let idx = ret.idx[ri];
    let rec = &root.layers[li];
    let (a0, s0, c0) = (rec.alpha[idx], rec.s[idx], rec.c[idx]);
    // Per-row lane deltas.
    let mut d = vec![0.0f64; nf];
    let mut db = vec![0.0f64; nf];
    for f in 0..nf {
        let v = vrow[f];
        let vp = v.max(0.0);
        let vn = v - vp;
        if dir > 0 {
            d[f] = vp * (1.0 - a0) + vn * (1.0 - s0);
        } else {
            d[f] = -(vp * a0 + vn * s0);
        }
        db[f] = -vn * c0;
    }
    let asl = pass.a.as_slice().expect("standard layout");
    debug_assert_eq!(pass.a.ncols(), nf);
    let al_row = &ret.a_l[ri * naug..(ri + 1) * naug];
    let au_row = &ret.a_u[ri * naug..(ri + 1) * naug];
    let mut acc_v = vec![0.0f64; nf];
    let mut acc_r = vec![0.0f64; nf];
    for k in 0..n_in {
        let alv = f64::from(al_row[k]);
        let auv = f64::from(au_row[k]);
        let m = root.mid[k];
        let rd = root.rad[k];
        let arow = &asl[k * nf..(k + 1) * nf];
        for f in 0..nf {
            let side = if d[f] >= 0.0 { alv } else { auv };
            let val = arow[f] + d[f] * side;
            acc_v[f] += val * m;
            acc_r[f] += val.abs() * rd;
        }
    }
    let bias_l = f64::from(al_row[n_in]);
    let bias_u = f64::from(au_row[n_in]);
    let mut best = f64::INFINITY;
    for f in 0..nf {
        let side_bias = if d[f] >= 0.0 { bias_l } else { bias_u };
        let b2 = pass.b[f] + db[f] + d[f] * side_bias;
        let est = (acc_v[f] - acc_r[f] + b2 + ms.cst[f]).max(ms.m1[f]);
        if est < best {
            best = est;
        }
    }
    best
}

/// Tier-0 head variant anchored on the parent's DIRECT pass (#epoch-bab;
/// RANKER ONLY — nearest mode): estimated `min_f` child bound after fixing
/// head neuron `i` in direction `dir`, evaluated as the parent's direct
/// margin-seeded rows plus the seed-coefficient delta routed through the
/// y-pack rows, jointly re-concretized. Unlike [`head_variant`] (which is
/// anchored on the looser via-y composed rows), this scores on the SAME
/// scale as [`trunk_variant`], so head and trunk candidates pool into one
/// comparable Tier-0 ranking (the via-y anchor made trunk scores dominate
/// the pool and starved the tree of head splits — measured on prop_1498).
#[allow(clippy::too_many_arguments)]
pub fn head_variant_direct(
    mb: &MarginBatch,
    gates: &HeadGates,
    ms: &MarginSeed,
    pass: &PassOut,
    al: &PassOut,
    au: &PassOut,
    root: &RootGates,
    i: usize,
    dir: i8,
) -> f64 {
    let nf = mb.nf();
    let n_y = mb.n_y;
    let n_in = root.mid.len();
    let (a0, s0, c0) = (gates.alpha[i], gates.s[i], gates.c[i]);
    let (a1, s1, dzu) = if dir > 0 {
        (1.0, 1.0, 0.0)
    } else {
        (0.0, 0.0, -ms.zu[i])
    };
    let asl = al.a.as_slice().expect("standard layout");
    let usl = au.a.as_slice().expect("standard layout");
    let psl = pass.a.as_slice().expect("standard layout");
    debug_assert_eq!(pass.a.ncols(), nf);
    // Each delta coefficient pairs with the sandwich side matching ITS OWN
    // sign (a positive coefficient on y_i lower-bounds via the Al row, a
    // negative one via Au) — the deltas flip sign with `dir`, so the pairing
    // is per-(row, delta), unlike `head_variant`'s recompute-the-whole-row
    // fixed wp->Al / wn->Au structure.
    let mut dca = vec![0.0f64; nf];
    let mut dcb = vec![0.0f64; nf];
    for f in 0..nf {
        dca[f] = mb.wp[f * n_y + i] * (a1 - a0);
        dcb[f] = mb.wn[f * n_y + i] * (s1 - s0);
    }
    let mut acc_v = vec![0.0f64; nf];
    let mut acc_r = vec![0.0f64; nf];
    for k in 0..n_in {
        let alv = asl[k * n_y + i];
        let auv = usl[k * n_y + i];
        let m = root.mid[k];
        let rd = root.rad[k];
        let prow = &psl[k * nf..(k + 1) * nf];
        for f in 0..nf {
            let r1 = if dca[f] >= 0.0 { alv } else { auv };
            let r2 = if dcb[f] >= 0.0 { alv } else { auv };
            let val = prow[f] + dca[f] * r1 + dcb[f] * r2;
            acc_v[f] += val * m;
            acc_r[f] += val.abs() * rd;
        }
    }
    let mut best = f64::INFINITY;
    for f in 0..nf {
        let wn_i = mb.wn[f * n_y + i];
        let b1 = if dca[f] >= 0.0 { al.b[i] } else { au.b[i] };
        let b2s = if dcb[f] >= 0.0 { al.b[i] } else { au.b[i] };
        let b2 = pass.b[f] + dca[f] * b1 + dcb[f] * b2s + wn_i * (0.0 - c0);
        let m1c = ms.m1[f] + wn_i * dzu;
        let est = (acc_v[f] - acc_r[f] + b2 + ms.cst[f]).max(m1c);
        if est < best {
            best = est;
        }
    }
    best
}

/// `min_j` bound after additionally head-fixing neuron `i` in direction
/// `dir` (exact recomputation with only that gate changed; nearest-mode
/// ranker).
#[allow(clippy::too_many_arguments)]
pub fn head_variant(
    mb: &MarginBatch,
    st: &VariantState,
    gates: &HeadGates,
    ybox: &YBox,
    al: &PassOut,
    au: &PassOut,
    root: &RootGates,
    i: usize,
    dir: i8,
) -> f64 {
    let nf = mb.nf();
    let n_y = mb.n_y;
    let n_in = al.a.nrows();
    let (a0, s0, c0) = (gates.alpha[i], gates.s[i], gates.c[i]);
    let (a1, s1, c1, dzu) = if dir > 0 {
        (1.0, 1.0, 0.0, 0.0)
    } else {
        (0.0, 0.0, 0.0, -ybox.uy[i].max(0.0))
    };
    let asl = al.a.as_slice().expect("standard layout");
    let usl = au.a.as_slice().expect("standard layout");
    let mut best = f64::INFINITY;
    for r0 in 0..nf {
        let wp_i = mb.wp[r0 * n_y + i];
        let wn_i = mb.wn[r0 * n_y + i];
        let m1 = st.m1[r0] + wn_i * dzu;
        let dca = wp_i * (a1 - a0);
        let dcb = wn_i * (s1 - s0);
        let rslice = &st.rows[r0 * n_in..(r0 + 1) * n_in];
        let mut v = 0.0;
        let mut rr = 0.0;
        for (k, rv) in rslice.iter().enumerate() {
            let val = rv + dca * asl[k * n_y + i] + dcb * usl[k * n_y + i];
            v += val * root.mid[k];
            rr += val.abs() * root.rad[k];
        }
        let bias = st.rowb[r0] + dca * al.b[i] + dcb * au.b[i];
        let m2 = v - rr + bias + st.cterm[r0] + wn_i * (c1 - c0) + mb.cst[r0];
        let m = m1.max(m2);
        if m < best {
            best = m;
        }
    }
    best
}
