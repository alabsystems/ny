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
use super::net::{nominal_product_dust, TwinNet};
use super::root::{gates_from_box, repair_upper_lines, RetainedLayer, RootGates};
use super::rounding::{certify_up, gamma_n, next_down, next_up, slack16, RoundMode, UNIT};

#[path = "bounds_confirm.rs"]
mod confirm;
#[path = "bounds_tiled.rs"]
mod tiled;

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

    /// #margin-row-col-retire: row-subset copy. `keep` must be strictly
    /// increasing indices into this batch's rows. The kept rows (`wp`, `wn`,
    /// `cst`, `adv`) are copied VERBATIM — bit-identical to the corresponding
    /// rows of `self` — so every per-row quantity downstream (`margin_seed`,
    /// `per_class_direct`, `compose_viay`, the variant rankers) reproduces the
    /// full-width batch's value for a surviving row exactly (all of those
    /// loops are row-local by construction).
    pub fn subset(&self, keep: &[usize]) -> Result<Self> {
        let n_y = self.n_y;
        let mut wp = Vec::with_capacity(keep.len() * n_y);
        let mut wn = Vec::with_capacity(keep.len() * n_y);
        let mut cst = Vec::with_capacity(keep.len());
        let mut adv = Vec::with_capacity(keep.len());
        let mut prev: Option<usize> = None;
        for &r in keep {
            if r >= self.nf() || prev.is_some_and(|p| p >= r) {
                return Err(NyError::InvalidSpec(
                    "margin_row: invalid column-retirement survivor set".into(),
                ));
            }
            prev = Some(r);
            wp.extend_from_slice(&self.wp[r * n_y..(r + 1) * n_y]);
            wn.extend_from_slice(&self.wn[r * n_y..(r + 1) * n_y]);
            cst.push(self.cst[r]);
            adv.push(self.adv[r]);
        }
        Ok(Self {
            t: self.t,
            adv,
            wp,
            wn,
            cst,
            n_y,
        })
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
    head_gates_with(ybox, mode, None)
}

/// #head-alpha: [`head_gates`] with the root's optimized head slopes
/// (`RootGates::head_alpha`) substituted for the binary area heuristic on every
/// head neuron that is UNSTABLE in this box (`ly < 0 < uy`).
///
/// Sound for any box: an unstable relu admits `alpha * y` as a lower line for
/// every `alpha` in `[0, 1]`, and the slope is clamped into that interval (a
/// non-finite slope keeps the heuristic). Stable neurons keep their exact lines
/// (`(1,1,0)` / `(0,0,0)`), and the upper chord `(s, c)` is untouched. The seed
/// arithmetic in `margin_seed` / `compose_viay` already charges the rounding of
/// `wp * alpha` for an arbitrary f64 `alpha`. `None`, or a length mismatch, is
/// the heuristic bit-identically.
pub fn head_gates_with(ybox: &YBox, mode: RoundMode, head_alpha: Option<&[f64]>) -> HeadGates {
    let (mut alpha, mut s, mut c) = gates_from_box(&ybox.ly, &ybox.uy);
    if let Some(ha) = head_alpha.filter(|ha| ha.len() == alpha.len()) {
        for (j, a) in alpha.iter_mut().enumerate() {
            if ybox.ly[j] < 0.0 && ybox.uy[j] > 0.0 && ha[j].is_finite() {
                *a = ha[j].clamp(0.0, 1.0);
            }
        }
    }
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
                if head_underflow(mb.wp[r0 * n_y + j], gates.alpha[j])
                    || head_underflow(mb.wn[r0 * n_y + j], gates.s[j])
                {
                    let lo = head_sum_lo(
                        head_prod_lo(mb.wp[r0 * n_y + j], gates.alpha[j]),
                        head_prod_lo(mb.wn[r0 * n_y + j], gates.s[j]),
                    );
                    let hi = head_sum_hi(
                        head_prod_hi(mb.wp[r0 * n_y + j], gates.alpha[j]),
                        head_prod_hi(mb.wn[r0 * n_y + j], gates.s[j]),
                    );
                    e[[j, r0]] = e[[j, r0]].max(head_gap(v, lo, hi));
                }
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
        let (mut underflow_c, mut underflow_m) = (false, false);
        for j in 0..n_y {
            acc_c += wn[j] * gates.c[j];
            abs_c += wn[j].abs() * gates.c[j];
            acc_m += wp[j] * zl[j] + wn[j] * zu[j];
            abs_m += wp[j] * zl[j] + wn[j].abs() * zu[j];
            if mode.outward() {
                underflow_c |= head_underflow(wn[j], gates.c[j]);
                underflow_m |= head_underflow(wp[j], zl[j]) || head_underflow(wn[j], zu[j]);
            }
        }
        cst[r0] = acc_c + mb.cst[r0];
        m1[r0] = acc_m + mb.cst[r0];
        if mode.outward() {
            cst_err[r0] = slack16(gam * (abs_c + mb.cst[r0].abs()));
            let sl = next_up(gam * (abs_m + mb.cst[r0].abs()));
            m1[r0] = next_down(next_down(m1[r0] - sl));
            if underflow_c {
                let (mut lo, mut hi) = (0.0, 0.0);
                for j in 0..n_y {
                    lo = head_sum_lo(lo, head_prod_lo(wn[j], gates.c[j]));
                    hi = head_sum_hi(hi, head_prod_hi(wn[j], gates.c[j]));
                }
                cst_err[r0] = cst_err[r0].max(head_gap(
                    cst[r0],
                    head_sum_lo(lo, mb.cst[r0]),
                    head_sum_hi(hi, mb.cst[r0]),
                ));
            }
            if underflow_m {
                let mut lo = 0.0;
                for j in 0..n_y {
                    lo = head_sum_lo(
                        lo,
                        head_sum_lo(head_prod_lo(wp[j], zl[j]), head_prod_lo(wn[j], zu[j])),
                    );
                }
                let directed = head_sum_lo(lo, mb.cst[r0]);
                // NaN must remain visible to the caller's finite firewall;
                // f64::min itself would hide a failed confirmation.
                m1[r0] = if directed.is_finite() {
                    m1[r0].min(directed)
                } else {
                    f64::NAN
                };
            }
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
            if root.mode.outward() && head_underflow(av.abs(), xa) {
                abs_dot[j] = head_sum_hi(abs_dot[j], head_prod_hi(av.abs(), xa));
            } else {
                abs_dot[j] += av.abs() * xa;
            }
        }
        if let Some(es) = esl {
            let erow = &es[i * r..(i + 1) * r];
            for (j, ev) in erow.iter().enumerate() {
                if root.mode.outward() && head_underflow(*ev, xa) {
                    err_dot[j] = head_sum_hi(err_dot[j], head_prod_hi(*ev, xa));
                } else {
                    err_dot[j] += ev * xa;
                }
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
    // Exact-zero coefficients can legitimately avoid every confirmation.
    // Qualify this outward bound itself; incidental repair is not an FP gate.
    // The root caller rejects nonfinite lanes before combining proof bounds.
    if mode.outward() && ny_core::require_f64_interval_proof_environment().is_err() {
        return vec![f64::NAN; mb.nf()];
    }
    compose_viay_impl::<true, true>(eng, mb, gates, al, au, al_dots, au_dots, mode)
}

#[allow(clippy::too_many_arguments)]
fn compose_viay_impl<const TILED: bool, const EXACT_ZERO: bool>(
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
    let tiled = (TILED
        && (!mode.outward() || ny_core::require_f64_interval_proof_environment().is_ok()))
    .then(|| tiled::TiledHeadRows::for_composition(asl, usl, n_in, n_y, nf))
    .flatten();
    let gam_row = gamma_n(2 * n_y + 8);
    let gam_con = gamma_n(n_in + 16);
    let mut out = vec![0.0; nf];
    let mut rowbuf = vec![0.0; n_in];
    // Invocation-local immutable sources; build only when repair is needed.
    // Both flags false retain the complete original calculation for controls.
    let mut confirmation_cache = None;
    let mut tried_confirmation_cache = false;
    let mut confirmation_trace = confirm::Trace::new(n_in, n_y, nf);
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        let mut underflow = false;
        // Composition coefficients cA = wp*alpha (>=0 side on lower rows),
        // cB = wn*s (<=0 side on upper rows).
        let ca: Vec<f64> = (0..n_y).map(|j| wp[j] * gates.alpha[j]).collect();
        let cb: Vec<f64> = (0..n_y).map(|j| wn[j] * gates.s[j]).collect();
        if mode.outward() {
            for j in 0..n_y {
                underflow |=
                    head_underflow(wp[j], gates.alpha[j]) || head_underflow(wn[j], gates.s[j]);
            }
        }
        // rows_r[k] = sum_j ca[j]*Al[k][j] + cb[j]*Au[k][j]
        let tiled_row = tiled
            .as_ref()
            .and_then(|rows| rows.product_underflow(&ca, &cb).map(|dust| (rows, dust)));
        if let Some((rows, dust)) = tiled_row {
            rows.compose(&ca, &cb, &mut rowbuf);
            underflow |= mode.outward() && dust;
        } else {
            for (k, rb) in rowbuf.iter_mut().enumerate() {
                let arow = &asl[k * n_y..(k + 1) * n_y];
                let urow = &usl[k * n_y..(k + 1) * n_y];
                let mut acc = 0.0;
                for j in 0..n_y {
                    acc += ca[j] * arow[j] + cb[j] * urow[j];
                    if mode.outward() {
                        underflow |=
                            head_underflow(ca[j], arow[j]) || head_underflow(cb[j], urow[j]);
                    }
                }
                *rb = acc;
            }
        }
        let mut bias = 0.0;
        let mut cterm = 0.0;
        for j in 0..n_y {
            bias += ca[j] * al.b[j] + cb[j] * au.b[j];
            cterm += wn[j] * gates.c[j];
            if mode.outward() {
                underflow |= head_underflow(ca[j], al.b[j])
                    || head_underflow(cb[j], au.b[j])
                    || head_underflow(wn[j], gates.c[j]);
            }
        }
        // Concretize rowbuf over the box.
        let mut v = 0.0;
        let mut rr = 0.0;
        for (k, rb) in rowbuf.iter().enumerate() {
            v += rb * root.mid[k];
            rr += rb.abs() * root.rad[k];
            if mode.outward() {
                underflow |=
                    head_underflow(*rb, root.mid[k]) || head_underflow(rb.abs(), root.rad[k]);
            }
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
            let ea = if EXACT_ZERO {
                head_seed_error(wp[j], gates.alpha[j], ca[j])
            } else {
                slack16(3.0 * UNIT * ca[j].abs())
            };
            let ebc = if EXACT_ZERO {
                head_seed_error(wn[j], gates.s[j], cb[j])
            } else {
                slack16(3.0 * UNIT * cb[j].abs())
            };
            underflow |= head_underflow(3.0 * UNIT, ca[j].abs())
                || head_underflow(3.0 * UNIT, cb[j].abs())
                || head_underflow(ca[j].abs() + ea, al_dots.err_dot[j])
                || head_underflow(ea, al_dots.abs_dot[j])
                || head_underflow(cb[j].abs() + ebc, au_dots.err_dot[j])
                || head_underflow(ebc, au_dots.abs_dot[j])
                || head_underflow(ca[j].abs(), al_dots.abs_dot[j])
                || head_underflow(cb[j].abs(), au_dots.abs_dot[j])
                || head_underflow(wn[j].abs(), gates.c[j]);
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
        underflow |= head_underflow(gam_row + gam_con, certify_up(tmag, 1e-13));
        if underflow {
            if TILED && !tried_confirmation_cache {
                tried_confirmation_cache = true;
                confirmation_trace.event("prepare_begin", r0, false);
                confirmation_cache = confirm::Prepared::new(root, mb, gates, al, au).ok();
                confirmation_trace.event("prepare_end", r0, confirmation_cache.is_some());
            }
            let cached = confirmation_cache.is_some();
            confirmation_trace.event("row_begin", r0, cached);
            let check = || Ok(());
            let confirmation = if let Some(prepared) = &mut confirmation_cache {
                prepared.row(r0)
            } else {
                HeadPoll::new(&check).and_then(|mut poll| {
                    head_confirm_viay_row(root, mb, gates, al, au, r0, &mut poll)
                })
            };
            confirmation_trace.row_end(r0, cached, confirmation.is_ok());
            out[r0] = match confirmation {
                Ok(directed) if directed.is_finite() => out[r0].min(directed),
                _ => f64::NAN,
            };
        }
    }
    confirmation_trace.finish();
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

// These are numerical adapters, not proof producers. The prepared ordinary
// evaluation owner holds the original source, row planes and SAME-ledger
// admission for the quoted overlap before calling them. A callback cannot
// authenticate supplied numerical premises or create a new proof scope.
const HEAD_POLL: usize = 256;
const HEAD_META: usize = 4096;

/// Conservative complete local arithmetic buffers, including transient
/// overlap. The caller retains the corresponding original WorkLease while
/// any returned numerical payload remains live; the owned evaluation prices
/// all simultaneously retained helper outputs, not just their maximum.
#[derive(Clone, Copy, Debug)]
pub(super) struct HeadArithmeticRequest {
    pub(super) scratch_bytes: usize,
    pub(super) work: u64,
}

fn head_invalid(message: &'static str) -> NyError {
    NyError::InvalidSpec(message.into())
}
fn head_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| head_invalid("cooperative head size overflow"))
}
fn head_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| head_invalid("cooperative head size overflow"))
}
fn head_request(cells: usize, ids: usize, units: usize) -> Result<HeadArithmeticRequest> {
    Ok(HeadArithmeticRequest {
        scratch_bytes: head_add(
            HEAD_META,
            head_add(head_mul(cells, 8)?, head_mul(ids, size_of::<usize>())?)?,
        )?,
        work: u64::try_from(units).map_err(|_| head_invalid("cooperative head work overflow"))?,
    })
}
pub(super) fn margin_batch_request(n_y: usize, nf: usize) -> Result<HeadArithmeticRequest> {
    let cells = head_mul(n_y, nf)?;
    head_request(
        head_add(head_mul(2, cells)?, nf)?,
        nf,
        head_add(head_mul(cells, 32)?, head_add(head_mul(nf, 16)?, 256)?)?,
    )
}
pub(super) fn head_gates_request(n_y: usize) -> Result<HeadArithmeticRequest> {
    head_request(
        head_mul(3, head_add(n_y, n_y.min(HEAD_POLL))?)?,
        0,
        head_add(head_mul(n_y, 256)?, 256)?,
    )
}
pub(super) fn margin_seed_request(
    n_y: usize,
    nf: usize,
    mode: RoundMode,
) -> Result<HeadArithmeticRequest> {
    let cells = head_mul(n_y, nf)?;
    let matrix_count = if mode.outward() { 2 } else { 1 };
    head_request(
        head_add(
            head_mul(matrix_count, cells)?,
            head_add(head_mul(3, nf)?, head_mul(2, n_y)?)?,
        )?,
        0,
        head_add(
            head_mul(cells, 256)?,
            head_add(head_mul(n_y, 32)?, head_add(head_mul(nf, 64)?, 256)?)?,
        )?,
    )
}
pub(super) fn row_dots_request(n_in: usize, columns: usize) -> Result<HeadArithmeticRequest> {
    head_request(
        head_mul(2, columns)?,
        0,
        head_add(
            head_mul(head_mul(n_in, columns)?, 64)?,
            head_add(head_mul(columns, 32)?, 256)?,
        )?,
    )
}
pub(super) fn compose_viay_request(
    n_in: usize,
    n_y: usize,
    nf: usize,
) -> Result<HeadArithmeticRequest> {
    head_request(
        head_add(nf, head_add(head_mul(3, n_in)?, head_mul(6, n_y)?)?)?,
        0,
        head_add(
            head_mul(head_mul(head_mul(nf, head_add(n_in, 1)?)?, n_y)?, 512)?,
            head_add(
                head_mul(head_mul(head_add(nf, 1)?, head_add(n_in, n_y)?)?, 128)?,
                256,
            )?,
        )?,
    )
}
pub(super) fn direct_from_lower_request(nf: usize) -> Result<HeadArithmeticRequest> {
    head_request(nf, 0, head_add(head_mul(nf, 32)?, 256)?)
}

struct HeadPoll<'a> {
    check: &'a dyn Fn() -> Result<()>,
    cells: usize,
}
impl<'a> HeadPoll<'a> {
    fn new(check: &'a dyn Fn() -> Result<()>) -> Result<Self> {
        check()?;
        Ok(Self { check, cells: 0 })
    }
    #[inline]
    fn tick(&mut self) -> Result<()> {
        self.cells += 1;
        if self.cells == HEAD_POLL {
            self.cells = 0;
            (self.check)()?;
        }
        Ok(())
    }
    fn finish(&self) -> Result<()> {
        (self.check)()
    }
}
fn head_buffer<T: Copy>(n: usize, value: T, poll: &mut HeadPoll<'_>) -> Result<Vec<T>> {
    poll.finish()?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| head_invalid("cooperative head buffer allocation refused"))?;
    poll.finish()?;
    if out.capacity() != n {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: out.capacity().saturating_mul(size_of::<T>()),
            budget_bytes: n.saturating_mul(size_of::<T>()),
            site: "cooperative head buffer actual capacity",
        });
    }
    for _ in 0..n {
        poll.tick()?;
        out.push(value);
    }
    poll.finish()?;
    Ok(out)
}
fn head_matrix(rows: usize, cols: usize, poll: &mut HeadPoll<'_>) -> Result<Array2<f64>> {
    Array2::from_shape_vec((rows, cols), head_buffer(head_mul(rows, cols)?, 0.0, poll)?)
        .map_err(|_| head_invalid("cooperative head matrix shape"))
}
fn head_qualify(mode: RoundMode) -> Result<()> {
    if mode.outward() {
        ny_core::require_f64_interval_proof_environment()?;
    }
    Ok(())
}
fn head_finite(x: f64) -> Result<()> {
    if !x.is_finite() {
        return Err(head_invalid("cooperative head nonfinite arithmetic"));
    }
    Ok(())
}
// Qualified RN+gradual-underflow operations lie between their adjacent
// representable values. Exact zero/additive identity and +/-1 products are
// preserved, avoiding artificial dust on genuine zero error lanes.
#[inline]
fn head_underflow(a: f64, b: f64) -> bool {
    nominal_product_dust(a, b) != 0.0
}
// A finite zero operand proves this seed coefficient has exactly zero error.
// Testing only the rounded product would erase genuine underflow (eta * 0.5).
// Inspect bits: DAZ can make a nonzero subnormal compare equal to zero. This
// local exemption does not qualify the surrounding arithmetic under DAZ.
// For MarginBatch::new, a zero split weight also has original-network
// provenance: finite binary64 differences lie on the eta lattice, so a
// nonzero exact difference cannot round to zero under RN+gradual underflow.
// Nonzero differences retain their existing relative-error envelope.
#[inline]
fn head_seed_error(a: f64, b: f64, product: f64) -> f64 {
    let zero = |x: f64| x.to_bits() & 0x7fff_ffff_ffff_ffff == 0;
    if a.is_finite() && b.is_finite() && (zero(a) || zero(b)) && zero(product) {
        0.0
    } else {
        slack16(3.0 * UNIT * product.abs())
    }
}
#[inline]
fn head_endpoint(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}
#[inline]
fn head_prod_lo(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }
    let p = a * b;
    head_endpoint(
        if a == 0.0 || b == 0.0 || a.abs() == 1.0 || b.abs() == 1.0 {
            p
        } else {
            next_down(p)
        },
    )
}
#[inline]
fn head_prod_hi(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }
    let p = a * b;
    head_endpoint(
        if a == 0.0 || b == 0.0 || a.abs() == 1.0 || b.abs() == 1.0 {
            p
        } else {
            next_up(p)
        },
    )
}
#[inline]
fn head_sum_lo(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }
    head_endpoint(if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        next_down(a + b)
    })
}
#[inline]
fn head_sum_hi(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }
    head_endpoint(if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        next_up(a + b)
    })
}
#[inline]
fn head_gap(center: f64, lo: f64, hi: f64) -> f64 {
    if !center.is_finite() || !lo.is_finite() || !hi.is_finite() {
        return f64::INFINITY;
    }
    let left = head_sum_hi(center, -lo);
    let right = head_sum_hi(hi, -center);
    if !left.is_finite() || !right.is_finite() {
        return f64::INFINITY;
    }
    left.max(right).max(0.0)
}
#[inline]
fn head_interval_product(al: f64, au: f64, bl: f64, bu: f64) -> (f64, f64) {
    let low = [
        head_prod_lo(al, bl),
        head_prod_lo(al, bu),
        head_prod_lo(au, bl),
        head_prod_lo(au, bu),
    ];
    let high = [
        head_prod_hi(al, bl),
        head_prod_hi(al, bu),
        head_prod_hi(au, bl),
        head_prod_hi(au, bu),
    ];
    // f64::min/max ignore NaN: check every endpoint BEFORE selecting, so an
    // overflowed interval can never be hidden by another finite corner.
    if low.iter().chain(&high).any(|v| !v.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    (
        low[0].min(low[1]).min(low[2]).min(low[3]),
        high[0].max(high[1]).max(high[2]).max(high[3]),
    )
}

/// Paid caller numerical adapter; identical nominal head-difference order.
pub(super) fn margin_batch_with_checkpoint(
    net: &TwinNet,
    t: usize,
    adv: &[usize],
    check: &dyn Fn() -> Result<()>,
) -> Result<MarginBatch> {
    let mut poll = HeadPoll::new(check)?;
    ny_core::require_f64_interval_proof_environment()?;
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    if t >= n_out {
        return Err(head_invalid("margin_row: class out of range"));
    }
    for &j in adv {
        poll.tick()?;
        if j >= n_out {
            return Err(head_invalid("margin_row: class out of range"));
        }
    }
    let nf = adv.len();
    let mut wp = head_buffer(head_mul(nf, n_y)?, 0.0, &mut poll)?;
    let mut wn = head_buffer(head_mul(nf, n_y)?, 0.0, &mut poll)?;
    let mut cst = head_buffer(nf, 0.0, &mut poll)?;
    let mut classes = head_buffer(nf, 0usize, &mut poll)?;
    for (r, &j) in adv.iter().enumerate() {
        for k in 0..n_y {
            poll.tick()?;
            let w = w2[t * n_y + k] - w2[j * n_y + k];
            head_finite(w)?;
            wp[r * n_y + k] = w.max(0.0);
            wn[r * n_y + k] = w.min(0.0);
        }
        poll.tick()?;
        cst[r] = b2[t] - b2[j];
        head_finite(cst[r])?;
        classes[r] = j;
    }
    poll.finish()?;
    Ok(MarginBatch {
        t,
        adv: classes,
        wp,
        wn,
        cst,
        n_y,
    })
}

/// Same original gate formulas and repair, bounded to a paid 256-row chunk.
pub(super) fn head_gates_with_checkpoint(
    ybox: &YBox,
    mode: RoundMode,
    head_alpha: Option<&[f64]>,
    check: &dyn Fn() -> Result<()>,
) -> Result<HeadGates> {
    let mut poll = HeadPoll::new(check)?;
    head_qualify(mode)?;
    let n = ybox.ly.len();
    if ybox.uy.len() != n {
        return Err(head_invalid("cooperative head box shape"));
    }
    let mut alpha = head_buffer(n, 0.0, &mut poll)?;
    let mut s = head_buffer(n, 0.0, &mut poll)?;
    let mut c = head_buffer(n, 0.0, &mut poll)?;
    for first in (0..n).step_by(HEAD_POLL) {
        let end = (first + HEAD_POLL).min(n);
        for j in first..end {
            poll.tick()?;
            if !ybox.ly[j].is_finite() || !ybox.uy[j].is_finite() || ybox.ly[j] > ybox.uy[j] {
                return Err(head_invalid("cooperative head invalid box"));
            }
        }
        poll.finish()?;
        let (mut a, mut ss, mut cc) = gates_from_box(&ybox.ly[first..end], &ybox.uy[first..end]);
        if a.capacity() != end - first
            || ss.capacity() != end - first
            || cc.capacity() != end - first
        {
            return Err(head_invalid("cooperative head gate chunk capacity"));
        }
        poll.finish()?;
        if let Some(ha) = head_alpha.filter(|ha| ha.len() == n) {
            for (offset, av) in a.iter_mut().enumerate() {
                poll.tick()?;
                let j = first + offset;
                if ybox.ly[j] < 0.0 && ybox.uy[j] > 0.0 && ha[j].is_finite() {
                    *av = ha[j].clamp(0.0, 1.0);
                }
            }
        }
        poll.finish()?;
        if mode.outward() {
            repair_upper_lines(&ybox.ly[first..end], &ybox.uy[first..end], &mut ss, &mut cc);
        }
        poll.finish()?;
        for offset in 0..end - first {
            poll.tick()?;
            alpha[first + offset] = a[offset];
            s[first + offset] = ss[offset];
            c[first + offset] = cc[offset];
            head_finite(a[offset])?;
            head_finite(ss[offset])?;
            head_finite(cc[offset])?;
        }
    }
    poll.finish()?;
    Ok(HeadGates { alpha, s, c })
}
fn head_shapes(mb: &MarginBatch, gates: &HeadGates) -> Result<()> {
    let n = mb.n_y;
    let cells = head_mul(mb.nf(), n)?;
    if mb.wp.len() != cells
        || mb.wn.len() != cells
        || mb.cst.len() != mb.nf()
        || gates.alpha.len() != n
        || gates.s.len() != n
        || gates.c.len() != n
    {
        return Err(head_invalid("cooperative head margin/gate shape"));
    }
    Ok(())
}

/// Nominal seed is legacy-bit-identical. Its errors and scalar bound include
/// independent directed confirmations, including every subnormal product and
/// cancellation; no relative-only positive m1 can enter a prepared close.
pub(super) fn margin_seed_with_checkpoint(
    mb: &MarginBatch,
    gates: &HeadGates,
    ybox: &YBox,
    mode: RoundMode,
    check: &dyn Fn() -> Result<()>,
) -> Result<MarginSeed> {
    let mut poll = HeadPoll::new(check)?;
    head_qualify(mode)?;
    head_shapes(mb, gates)?;
    let (nf, n_y) = (mb.nf(), mb.n_y);
    if ybox.ly.len() != n_y || ybox.uy.len() != n_y {
        return Err(head_invalid("cooperative head seed box shape"));
    }
    let mut s_mat = head_matrix(n_y, nf, &mut poll)?;
    let mut e_mat = if mode.outward() {
        Some(head_matrix(n_y, nf, &mut poll)?)
    } else {
        None
    };
    for j in 0..n_y {
        poll.tick()?;
        if !ybox.ly[j].is_finite()
            || !ybox.uy[j].is_finite()
            || ybox.ly[j] > ybox.uy[j]
            || !gates.alpha[j].is_finite()
            || !(0.0..=1.0).contains(&gates.alpha[j])
            || !gates.s[j].is_finite()
            || !(0.0..=1.0).contains(&gates.s[j])
            || !gates.c[j].is_finite()
            || gates.c[j] < 0.0
        {
            return Err(head_invalid("cooperative head seed invalid gates/box"));
        }
        for r0 in 0..nf {
            poll.tick()?;
            let wp = mb.wp[r0 * n_y + j];
            let wn = mb.wn[r0 * n_y + j];
            if !wp.is_finite() || wp < 0.0 || !wn.is_finite() || wn > 0.0 {
                return Err(head_invalid("cooperative head margin signs"));
            }
            let v = wp * gates.alpha[j] + wn * gates.s[j];
            head_finite(v)?;
            s_mat[[j, r0]] = v;
            if let Some(e) = e_mat.as_mut() {
                let mag = (wp * gates.alpha[j]).abs() + (wn * gates.s[j]).abs();
                let lo = head_sum_lo(
                    head_prod_lo(wp, gates.alpha[j]),
                    head_prod_lo(wn, gates.s[j]),
                );
                let hi = head_sum_hi(
                    head_prod_hi(wp, gates.alpha[j]),
                    head_prod_hi(wn, gates.s[j]),
                );
                e[[j, r0]] = slack16(3.0 * UNIT * mag).max(head_gap(v, lo, hi));
                head_finite(e[[j, r0]])?;
            }
        }
    }
    let gam = gamma_n(head_add(n_y, 8)?);
    let mut cst = head_buffer(nf, 0.0, &mut poll)?;
    let mut cst_err = head_buffer(nf, 0.0, &mut poll)?;
    let mut m1 = head_buffer(nf, 0.0, &mut poll)?;
    let mut zl = head_buffer(n_y, 0.0, &mut poll)?;
    let mut zu = head_buffer(n_y, 0.0, &mut poll)?;
    for j in 0..n_y {
        poll.tick()?;
        zl[j] = ybox.ly[j].max(0.0);
    }
    for j in 0..n_y {
        poll.tick()?;
        zu[j] = ybox.uy[j].max(0.0);
    }
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        let (mut acc_c, mut abs_c, mut acc_m, mut abs_m) = (0.0, 0.0, 0.0, 0.0);
        let (mut cl, mut cu, mut ml) = (0.0, 0.0, 0.0);
        for j in 0..n_y {
            poll.tick()?;
            acc_c += wn[j] * gates.c[j];
            abs_c += wn[j].abs() * gates.c[j];
            acc_m += wp[j] * zl[j] + wn[j] * zu[j];
            abs_m += wp[j] * zl[j] + wn[j].abs() * zu[j];
            if mode.outward() {
                cl = head_sum_lo(cl, head_prod_lo(wn[j], gates.c[j]));
                cu = head_sum_hi(cu, head_prod_hi(wn[j], gates.c[j]));
                ml = head_sum_lo(
                    ml,
                    head_sum_lo(head_prod_lo(wp[j], zl[j]), head_prod_lo(wn[j], zu[j])),
                );
            }
        }
        poll.tick()?;
        head_finite(mb.cst[r0])?;
        cst[r0] = acc_c + mb.cst[r0];
        m1[r0] = acc_m + mb.cst[r0];
        if mode.outward() {
            cl = head_sum_lo(cl, mb.cst[r0]);
            cu = head_sum_hi(cu, mb.cst[r0]);
            cst_err[r0] = slack16(gam * (abs_c + mb.cst[r0].abs())).max(head_gap(cst[r0], cl, cu));
            let sl = next_up(gam * (abs_m + mb.cst[r0].abs()));
            let directed_m1 = head_sum_lo(ml, mb.cst[r0]);
            head_finite(directed_m1)?;
            m1[r0] = next_down(next_down(m1[r0] - sl)).min(directed_m1);
        }
        head_finite(cst[r0])?;
        head_finite(cst_err[r0])?;
        head_finite(m1[r0])?;
    }
    poll.finish()?;
    Ok(MarginSeed {
        seed: Seed { s: s_mat, e: e_mat },
        cst,
        cst_err,
        m1,
        zu,
    })
}

fn head_pack_shape(root: &RootGates, pack: &PassOut, columns: usize) -> Result<()> {
    let n = root.mid.len();
    if root.xabs.len() != n
        || root.lo.len() != n
        || root.hi.len() != n
        || root.rad.len() != n
        || pack.a.dim() != (n, columns)
        || !pack.a.is_standard_layout()
        || pack.b.len() != columns
        || pack.eb.len() != columns
        || pack
            .e
            .as_ref()
            .is_some_and(|e| e.dim() != (n, columns) || !e.is_standard_layout())
    {
        return Err(head_invalid("cooperative head row pack shape"));
    }
    if root.mode.outward() && pack.e.is_none() {
        return Err(head_invalid("cooperative head missing row error planes"));
    }
    Ok(())
}
/// Fully directed magnitude/error dots, not RN error products rounded to zero.
pub(super) fn row_dots_with_checkpoint(
    root: &RootGates,
    pack: &PassOut,
    check: &dyn Fn() -> Result<()>,
) -> Result<RowDots> {
    let mut poll = HeadPoll::new(check)?;
    head_qualify(root.mode)?;
    let (n, r) = (root.mid.len(), pack.a.ncols());
    head_pack_shape(root, pack, r)?;
    let asl = pack
        .a
        .as_slice()
        .ok_or_else(|| head_invalid("cooperative head row layout"))?;
    let esl = pack.e.as_ref().and_then(|e| e.as_slice());
    let mut abs_dot = head_buffer(r, 0.0, &mut poll)?;
    let mut err_dot = head_buffer(r, 0.0, &mut poll)?;
    for i in 0..n {
        let xa = root.xabs[i].max(root.lo[i].abs().max(root.hi[i].abs()));
        if !root.lo[i].is_finite()
            || !root.hi[i].is_finite()
            || root.lo[i] > root.hi[i]
            || !xa.is_finite()
            || xa < 0.0
        {
            return Err(head_invalid("cooperative head input magnitude"));
        }
        let arow = &asl[i * r..(i + 1) * r];
        for (j, av) in arow.iter().enumerate() {
            poll.tick()?;
            head_finite(*av)?;
            abs_dot[j] = if root.mode.outward() {
                head_sum_hi(abs_dot[j], head_prod_hi(av.abs(), xa))
            } else {
                abs_dot[j] + av.abs() * xa
            };
        }
        if let Some(es) = esl {
            let erow = &es[i * r..(i + 1) * r];
            for (j, ev) in erow.iter().enumerate() {
                poll.tick()?;
                if !ev.is_finite() || *ev < 0.0 {
                    return Err(head_invalid("cooperative head row error"));
                }
                err_dot[j] = if root.mode.outward() {
                    head_sum_hi(err_dot[j], head_prod_hi(*ev, xa))
                } else {
                    err_dot[j] + ev * xa
                };
            }
        }
    }
    for j in 0..r {
        poll.tick()?;
        head_finite(pack.b[j])?;
        if !pack.eb[j].is_finite() || pack.eb[j] < 0.0 {
            return Err(head_invalid("cooperative head bias error"));
        }
        if root.mode.outward() {
            abs_dot[j] = head_sum_hi(abs_dot[j], pack.b[j].abs());
            err_dot[j] = head_sum_hi(err_dot[j], pack.eb[j]);
        } else {
            abs_dot[j] += pack.b[j].abs();
            err_dot[j] += pack.eb[j];
        }
        head_finite(abs_dot[j])?;
        head_finite(err_dot[j])?;
    }
    poll.finish()?;
    Ok(RowDots { abs_dot, err_dot })
}

/// Precomputed direct concretization plus its own complete constant error.
pub(super) fn per_class_direct_from_lower_with_checkpoint(
    low: &[f64],
    ms: &MarginSeed,
    rows: std::ops::Range<usize>,
    mode: RoundMode,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<f64>> {
    let mut poll = HeadPoll::new(check)?;
    head_qualify(mode)?;
    if rows.start > rows.end
        || rows.end > low.len()
        || rows.end > ms.cst.len()
        || rows.end > ms.cst_err.len()
    {
        return Err(head_invalid("cooperative head direct shape"));
    }
    let mut out = head_buffer(rows.len(), 0.0, &mut poll)?;
    for (k, r0) in rows.enumerate() {
        poll.tick()?;
        head_finite(low[r0])?;
        head_finite(ms.cst[r0])?;
        if !ms.cst_err[r0].is_finite() || ms.cst_err[r0] < 0.0 {
            return Err(head_invalid("cooperative head direct constant error"));
        }
        let v = low[r0] + ms.cst[r0];
        out[k] = if mode.outward() {
            next_down(next_down(v - next_up(ms.cst_err[r0])))
        } else {
            v
        };
        head_finite(out[k])?;
    }
    poll.finish()?;
    Ok(out)
}

/// Same nominal composition order plus an independent fully directed lower
/// confirmation. The original margin-difference residual remains the sealed
/// caller's responsibility. Each row uses BOTH complete coefficient/bias
/// error planes over the authentic original input magnitude/domain.
#[allow(clippy::too_many_arguments)]
pub(super) fn compose_viay_with_checkpoint(
    eng: &BackwardEngine<'_>,
    mb: &MarginBatch,
    gates: &HeadGates,
    al: &PassOut,
    au: &PassOut,
    al_dots: &RowDots,
    au_dots: &RowDots,
    mode: RoundMode,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<f64>> {
    let mut poll = HeadPoll::new(check)?;
    head_qualify(mode)?;
    head_shapes(mb, gates)?;
    let (nf, n_y) = (mb.nf(), mb.n_y);
    let root = eng.root;
    let n_in = root.mid.len();
    head_pack_shape(root, al, n_y)?;
    head_pack_shape(root, au, n_y)?;
    if root.mode != mode
        || root.lo.len() != n_in
        || root.hi.len() != n_in
        || al_dots.abs_dot.len() != n_y
        || al_dots.err_dot.len() != n_y
        || au_dots.abs_dot.len() != n_y
        || au_dots.err_dot.len() != n_y
    {
        return Err(head_invalid("cooperative head composition shape/mode"));
    }
    let asl =
        al.a.as_slice()
            .ok_or_else(|| head_invalid("cooperative head row layout"))?;
    let usl =
        au.a.as_slice()
            .ok_or_else(|| head_invalid("cooperative head row layout"))?;
    let gam_row = gamma_n(head_add(head_mul(2, n_y)?, 8)?);
    let gam_con = gamma_n(head_add(n_in, 16)?);
    let mut out = head_buffer(nf, 0.0, &mut poll)?;
    let mut rowbuf = head_buffer(n_in, 0.0, &mut poll)?;
    let mut rowlo = head_buffer(n_in, 0.0, &mut poll)?;
    let mut rowhi = head_buffer(n_in, 0.0, &mut poll)?;
    let mut ca = head_buffer(n_y, 0.0, &mut poll)?;
    let mut cb = head_buffer(n_y, 0.0, &mut poll)?;
    let mut cal = head_buffer(n_y, 0.0, &mut poll)?;
    let mut cau = head_buffer(n_y, 0.0, &mut poll)?;
    let mut cbl = head_buffer(n_y, 0.0, &mut poll)?;
    let mut cbu = head_buffer(n_y, 0.0, &mut poll)?;
    for j in 0..n_y {
        poll.tick()?;
        if !gates.alpha[j].is_finite()
            || !(0.0..=1.0).contains(&gates.alpha[j])
            || !gates.s[j].is_finite()
            || !(0.0..=1.0).contains(&gates.s[j])
            || !gates.c[j].is_finite()
            || gates.c[j] < 0.0
            || !al_dots.abs_dot[j].is_finite()
            || al_dots.abs_dot[j] < 0.0
            || !al_dots.err_dot[j].is_finite()
            || al_dots.err_dot[j] < 0.0
            || !au_dots.abs_dot[j].is_finite()
            || au_dots.abs_dot[j] < 0.0
            || !au_dots.err_dot[j].is_finite()
            || au_dots.err_dot[j] < 0.0
        {
            return Err(head_invalid(
                "cooperative head composition invalid gates/dots",
            ));
        }
    }
    for k in 0..n_in {
        poll.tick()?;
        if !root.lo[k].is_finite()
            || !root.hi[k].is_finite()
            || root.lo[k] > root.hi[k]
            || !root.mid[k].is_finite()
            || !root.rad[k].is_finite()
            || root.rad[k] < 0.0
        {
            return Err(head_invalid("cooperative head composition domain"));
        }
    }
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        for j in 0..n_y {
            poll.tick()?;
            if !wp[j].is_finite() || wp[j] < 0.0 || !wn[j].is_finite() || wn[j] > 0.0 {
                return Err(head_invalid("cooperative head composition margin signs"));
            }
            ca[j] = wp[j] * gates.alpha[j];
            cal[j] = head_prod_lo(wp[j], gates.alpha[j]);
            cau[j] = head_prod_hi(wp[j], gates.alpha[j]);
        }
        for j in 0..n_y {
            poll.tick()?;
            cb[j] = wn[j] * gates.s[j];
            cbl[j] = head_prod_lo(wn[j], gates.s[j]);
            cbu[j] = head_prod_hi(wn[j], gates.s[j]);
        }
        for (k, rb) in rowbuf.iter_mut().enumerate() {
            let arow = &asl[k * n_y..(k + 1) * n_y];
            let urow = &usl[k * n_y..(k + 1) * n_y];
            let (mut acc, mut lo, mut hi) = (0.0, 0.0, 0.0);
            for j in 0..n_y {
                poll.tick()?;
                head_finite(arow[j])?;
                head_finite(urow[j])?;
                acc += ca[j] * arow[j] + cb[j] * urow[j];
                if mode.outward() {
                    let (a, b) = head_interval_product(cal[j], cau[j], arow[j], arow[j]);
                    let (c, d) = head_interval_product(cbl[j], cbu[j], urow[j], urow[j]);
                    lo = head_sum_lo(lo, head_sum_lo(a, c));
                    hi = head_sum_hi(hi, head_sum_hi(b, d));
                }
            }
            *rb = acc;
            rowlo[k] = lo;
            rowhi[k] = hi;
        }
        let (mut bias, mut cterm, mut directed_bias, mut directed_c, mut directed_pen) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        for j in 0..n_y {
            poll.tick()?;
            head_finite(al.b[j])?;
            head_finite(au.b[j])?;
            bias += ca[j] * al.b[j] + cb[j] * au.b[j];
            cterm += wn[j] * gates.c[j];
            if mode.outward() {
                let (a, _) = head_interval_product(cal[j], cau[j], al.b[j], al.b[j]);
                let (b, _) = head_interval_product(cbl[j], cbu[j], au.b[j], au.b[j]);
                directed_bias = head_sum_lo(directed_bias, head_sum_lo(a, b));
                directed_c = head_sum_lo(directed_c, head_prod_lo(wn[j], gates.c[j]));
                let ea = head_prod_hi(cal[j].abs().max(cau[j].abs()), al_dots.err_dot[j]);
                let eb = head_prod_hi(cbl[j].abs().max(cbu[j].abs()), au_dots.err_dot[j]);
                directed_pen = head_sum_hi(directed_pen, head_sum_hi(ea, eb));
            }
        }
        let (mut v, mut rr, mut directed_input) = (0.0, 0.0, 0.0);
        for (k, rb) in rowbuf.iter().enumerate() {
            poll.tick()?;
            v += rb * root.mid[k];
            rr += rb.abs() * root.rad[k];
            if mode.outward() {
                let (l, _) = head_interval_product(rowlo[k], rowhi[k], root.lo[k], root.hi[k]);
                directed_input = head_sum_lo(directed_input, l);
            }
        }
        let raw = v - rr + bias + cterm + mb.cst[r0];
        head_finite(raw)?;
        if !mode.outward() {
            out[r0] = raw;
            continue;
        }
        let (mut pen, mut tmag) = (0.0, 0.0);
        for j in 0..n_y {
            poll.tick()?;
            let ea = head_seed_error(wp[j], gates.alpha[j], ca[j]);
            let ebc = head_seed_error(wn[j], gates.s[j], cb[j]);
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
        let legacy = next_down(next_down(raw - slack));
        let directed = head_sum_lo(
            head_sum_lo(
                head_sum_lo(head_sum_lo(directed_input, directed_bias), directed_c),
                mb.cst[r0],
            ),
            -directed_pen,
        );
        head_finite(legacy)?;
        head_finite(directed)?;
        let confirmed = head_confirm_viay_row(root, mb, gates, al, au, r0, &mut poll)?;
        out[r0] = legacy.min(directed).min(confirmed);
    }
    poll.finish()?;
    Ok(out)
}

// Allocation-free confirmation shared by the prepared owner and subnormal
// legacy repair. It reads the ACTUAL full row E/eb carriers, not RN cached
// magnitude/error summaries. Exact original endpoint magnitudes also cover
// legacy xabs whose abs(mid)+rad construction was nearest-rounded.
#[allow(clippy::too_many_arguments)]
fn head_confirm_viay_row(
    root: &RootGates,
    mb: &MarginBatch,
    gates: &HeadGates,
    al: &PassOut,
    au: &PassOut,
    column: usize,
    poll: &mut HeadPoll<'_>,
) -> Result<f64> {
    head_qualify(RoundMode::Outward)?;
    head_shapes(mb, gates)?;
    let (n, ny) = (root.mid.len(), mb.n_y);
    head_pack_shape(root, al, ny)?;
    head_pack_shape(root, au, ny)?;
    if column >= mb.nf() || root.lo.len() != n || root.hi.len() != n {
        return Err(head_invalid(
            "head confirmation original domain/column shape",
        ));
    }
    let ae =
        al.e.as_ref()
            .and_then(|e| e.as_slice())
            .ok_or_else(|| head_invalid("head confirmation lower error plane"))?;
    let ue =
        au.e.as_ref()
            .and_then(|e| e.as_slice())
            .ok_or_else(|| head_invalid("head confirmation upper error plane"))?;
    let a =
        al.a.as_slice()
            .ok_or_else(|| head_invalid("head confirmation lower layout"))?;
    let u =
        au.a.as_slice()
            .ok_or_else(|| head_invalid("head confirmation upper layout"))?;
    let wp = &mb.wp[column * ny..(column + 1) * ny];
    let wn = &mb.wn[column * ny..(column + 1) * ny];
    let mut input = 0.0;
    for k in 0..n {
        poll.tick()?;
        if !root.lo[k].is_finite() || !root.hi[k].is_finite() || root.lo[k] > root.hi[k] {
            return Err(head_invalid("head confirmation original input endpoints"));
        }
        let (mut lo, mut hi) = (0.0, 0.0);
        for j in 0..ny {
            poll.tick()?;
            let cal = head_prod_lo(wp[j], gates.alpha[j]);
            let cau = head_prod_hi(wp[j], gates.alpha[j]);
            let cbl = head_prod_lo(wn[j], gates.s[j]);
            let cbu = head_prod_hi(wn[j], gates.s[j]);
            let (ll, lh) = head_interval_product(cal, cau, a[k * ny + j], a[k * ny + j]);
            let (ul, uh) = head_interval_product(cbl, cbu, u[k * ny + j], u[k * ny + j]);
            lo = head_sum_lo(lo, head_sum_lo(ll, ul));
            hi = head_sum_hi(hi, head_sum_hi(lh, uh));
        }
        let (lower, _) = head_interval_product(lo, hi, root.lo[k], root.hi[k]);
        input = head_sum_lo(input, lower);
    }
    let (mut bias, mut constant, mut penalty) = (0.0, 0.0, 0.0);
    for j in 0..ny {
        poll.tick()?;
        if !wp[j].is_finite()
            || wp[j] < 0.0
            || !wn[j].is_finite()
            || wn[j] > 0.0
            || !gates.alpha[j].is_finite()
            || !(0.0..=1.0).contains(&gates.alpha[j])
            || !gates.s[j].is_finite()
            || !(0.0..=1.0).contains(&gates.s[j])
            || !gates.c[j].is_finite()
            || gates.c[j] < 0.0
            || !al.eb[j].is_finite()
            || al.eb[j] < 0.0
            || !au.eb[j].is_finite()
            || au.eb[j] < 0.0
        {
            return Err(head_invalid("head confirmation invalid source lanes"));
        }
        let cal = head_prod_lo(wp[j], gates.alpha[j]);
        let cau = head_prod_hi(wp[j], gates.alpha[j]);
        let cbl = head_prod_lo(wn[j], gates.s[j]);
        let cbu = head_prod_hi(wn[j], gates.s[j]);
        let (lower_bias, _) = head_interval_product(cal, cau, al.b[j], al.b[j]);
        let (upper_bias, _) = head_interval_product(cbl, cbu, au.b[j], au.b[j]);
        bias = head_sum_lo(bias, head_sum_lo(lower_bias, upper_bias));
        constant = head_sum_lo(constant, head_prod_lo(wn[j], gates.c[j]));
        let (mut eta_l, mut eta_u) = (al.eb[j], au.eb[j]);
        for k in 0..n {
            poll.tick()?;
            let (el, eu) = (ae[k * ny + j], ue[k * ny + j]);
            if !el.is_finite() || el < 0.0 || !eu.is_finite() || eu < 0.0 {
                return Err(head_invalid("head confirmation coefficient error"));
            }
            let xa = root.lo[k].abs().max(root.hi[k].abs());
            eta_l = head_sum_hi(eta_l, head_prod_hi(el, xa));
            eta_u = head_sum_hi(eta_u, head_prod_hi(eu, xa));
        }
        let el = head_prod_hi(cal.abs().max(cau.abs()), eta_l);
        let eu = head_prod_hi(cbl.abs().max(cbu.abs()), eta_u);
        penalty = head_sum_hi(penalty, head_sum_hi(el, eu));
    }
    let lower = head_sum_lo(
        head_sum_lo(
            head_sum_lo(head_sum_lo(input, bias), constant),
            mb.cst[column],
        ),
        -penalty,
    );
    head_finite(lower)?;
    poll.finish()?;
    Ok(lower)
}

#[cfg(test)]
#[path = "bounds_cooperative_tests.rs"]
mod cooperative_tests;

#[cfg(test)]
#[path = "bounds_zero_tests.rs"]
mod zero_tests;
