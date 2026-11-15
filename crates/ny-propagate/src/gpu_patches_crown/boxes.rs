// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#vgg-gpu-crown`: certified per-value boxes and the one-step screen.
//!
//! Every chain value carries a certified enclosure `[lo, hi]` (binary64) of its
//! reachable set over the input box. Affine outputs and max-pool outputs are
//! stored; a ReLU output is the elementwise `max(0, .)` of its input's box and
//! a reshape output IS its input's box, so neither is materialised.
//!
//! The SCREEN is the certified one-step interval image of an affine op over
//! its input's box: `z = W x + b` with `x` in `[c - r, c + r]` gives
//! `z in W c + b +- |W| r`. Its rounding is charged by Higham's
//! summation-order-independent `gamma_n` bound on the center sum and by
//! `certify_up` on the radius sum; the endpoints are rounded outward. It is
//! both the starting enclosure of every affine value and the reference the
//! CROWN rows are intersected with (shrink-only).

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::Instant;

use ny_core::f32_to_f64_exact;
use rayon::prelude::*;

use super::plan::{ChainOp, OpKind, PoolSpec, Shape3};
use super::product::{gamma_up, ETA_F64};
use crate::margin_row::rounding::{certify_up, next_down, next_up, slack16, UNIT};

/// How one value's box is represented.
pub(crate) enum ValueBox {
    /// A materialised enclosure.
    Stored { lo: Vec<f64>, hi: Vec<f64> },
    /// `max(0, box(p))` elementwise (a ReLU output).
    ReluOf(usize),
    /// Exactly `box(p)` (a reshape output: same row-major data).
    Same(usize),
}

/// Boxes of every value produced so far (index = value index).
#[derive(Default)]
pub(crate) struct Boxes {
    vals: Vec<ValueBox>,
}

/// A resolved read-only view of one value's box.
#[derive(Clone, Copy)]
pub(crate) struct BoxView<'a> {
    lo: &'a [f64],
    hi: &'a [f64],
    relu: bool,
}

impl BoxView<'_> {
    /// Lower endpoint of element `j`.
    #[inline]
    pub(crate) fn lo(&self, j: usize) -> f64 {
        let v = self.lo[j];
        if self.relu {
            v.max(0.0)
        } else {
            v
        }
    }

    /// Upper endpoint of element `j`.
    #[inline]
    pub(crate) fn hi(&self, j: usize) -> f64 {
        let v = self.hi[j];
        if self.relu {
            v.max(0.0)
        } else {
            v
        }
    }

    /// `max(|lo|, |hi|)`: a bound on `|value_j|` over the box.
    #[inline]
    pub(crate) fn mag(&self, j: usize) -> f64 {
        self.lo(j).abs().max(self.hi(j).abs())
    }

    /// Number of elements.
    pub(crate) fn len(&self) -> usize {
        self.lo.len()
    }
}

impl Boxes {
    /// Append the box of the next value.
    pub(crate) fn push(&mut self, b: ValueBox) {
        self.vals.push(b);
    }

    /// Number of values with a box.
    pub(crate) fn len(&self) -> usize {
        self.vals.len()
    }

    /// Resolve value `v`'s box. Aliases always point to an EARLIER value, so
    /// the walk terminates.
    pub(crate) fn view(&self, v: usize) -> BoxView<'_> {
        let mut idx = v;
        let mut relu = false;
        loop {
            match &self.vals[idx] {
                ValueBox::Stored { lo, hi } => {
                    return BoxView {
                        lo: lo.as_slice(),
                        hi: hi.as_slice(),
                        relu,
                    }
                }
                ValueBox::ReluOf(p) => {
                    relu = true;
                    idx = *p;
                }
                ValueBox::Same(p) => idx = *p,
            }
        }
    }

    /// Bytes held by the materialised boxes.
    pub(crate) fn stored_bytes(&self) -> usize {
        self.vals
            .iter()
            .map(|b| match b {
                ValueBox::Stored { lo, hi } => (lo.len() + hi.len()) * 8,
                _ => 0,
            })
            .sum()
    }
}

/// `true` iff every endpoint is finite and `lo <= hi`.
pub(crate) fn box_is_valid(lo: &[f64], hi: &[f64]) -> bool {
    lo.len() == hi.len()
        && lo
            .iter()
            .zip(hi.iter())
            .all(|(&l, &h)| l.is_finite() && h.is_finite() && l <= h)
}

/// Center / radius form of a box: `[lo, hi] subset [c - r, c + r]`.
fn center_radius(view: BoxView<'_>) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = view.len();
    let mut cen = Vec::with_capacity(n);
    let mut rad = Vec::with_capacity(n);
    for j in 0..n {
        let (l, h) = (view.lo(j), view.hi(j));
        if !l.is_finite() || !h.is_finite() || l > h {
            return None;
        }
        // Any `c` works: the radius is taken against the stored `c`, and
        // `next_up` of a nearest-rounded difference dominates the exact one.
        let c = 0.5 * l + 0.5 * h;
        let r = next_up((h - c).max(c - l));
        if !c.is_finite() || !r.is_finite() {
            return None;
        }
        cen.push(c);
        rad.push(r);
    }
    Some((cen, rad))
}

/// Certified endpoints from one output's three nearest-rounded sums.
///
/// `zc = fl(sum w c + b)`, `ta = fl(sum |w||c| + |b|)`, `zr = fl(sum |w| r)`,
/// each over `terms` products (the bias counted). The exact image is within
/// `sum |w| r_exact <= certify_up(zr) + eta` of the exact center, which is
/// within `gamma_terms * ta_exact + eta` of `zc` (Higham, any order).
#[inline]
fn endpoints(zc: f64, ta: f64, zr: f64, g: f64, eta: f64) -> (f64, f64) {
    let t_up = certify_up(ta, g) + eta;
    let r_up = certify_up(zr, g) + eta;
    let dev = slack16(r_up + slack16(g * t_up) + eta);
    (next_down(zc - dev), next_up(zc + dev))
}

/// Certified one-step interval image of an affine op over its input box.
/// `None` on a non-finite input or an expired deadline.
pub(crate) fn screen_affine(
    op: &ChainOp<'_>,
    input: BoxView<'_>,
    deadline: Instant,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let (cen, rad) = center_radius(input)?;
    let expired = AtomicBool::new(false);
    let out = op.output;
    let (lo, hi) = match &op.kind {
        OpKind::Conv(spec) => {
            let inp = op.input;
            let terms = spec.cin * spec.kh * spec.kw + 1;
            let g = gamma_up(terms as u64 + 2, UNIT)?;
            let eta = ETA_F64 * (terms as f64 + 2.0);
            let plane = out.plane();
            let mut lo = vec![0.0_f64; out.numel()];
            let mut hi = vec![0.0_f64; out.numel()];
            lo.par_chunks_mut(plane)
                .zip(hi.par_chunks_mut(plane))
                .enumerate()
                .for_each(|(co, (lo_c, hi_c))| {
                    if expired.load(Relaxed) {
                        return;
                    }
                    if Instant::now() >= deadline {
                        expired.store(true, Relaxed);
                        return;
                    }
                    let mut zc = vec![0.0_f64; plane];
                    let mut ta = vec![0.0_f64; plane];
                    let mut zr = vec![0.0_f64; plane];
                    for ci in 0..spec.cin {
                        let base = ci * inp.plane();
                        for ky in 0..spec.kh {
                            for kx in 0..spec.kw {
                                let wv = spec.weight(co, ci, ky, kx);
                                if wv == 0.0 {
                                    continue;
                                }
                                let wa = wv.abs();
                                for oy in 0..out.h {
                                    let iy = (oy * spec.sh + ky) as isize - spec.ph as isize;
                                    if iy < 0 || iy >= inp.h as isize {
                                        continue;
                                    }
                                    let row = base + (iy as usize) * inp.w;
                                    for ox in 0..out.w {
                                        let ix = (ox * spec.sw + kx) as isize - spec.pw as isize;
                                        if ix < 0 || ix >= inp.w as isize {
                                            continue;
                                        }
                                        let j = row + ix as usize;
                                        let o = oy * out.w + ox;
                                        zc[o] += wv * cen[j];
                                        ta[o] += wa * cen[j].abs();
                                        zr[o] += wa * rad[j];
                                    }
                                }
                            }
                        }
                    }
                    let b = spec.bias[co];
                    for o in 0..plane {
                        let (l, h) = endpoints(zc[o] + b, ta[o] + b.abs(), zr[o], g, eta);
                        lo_c[o] = l;
                        hi_c[o] = h;
                    }
                });
            (lo, hi)
        }
        OpKind::Linear(spec) => {
            let terms = spec.nin + 1;
            let g = gamma_up(terms as u64 + 2, UNIT)?;
            let eta = ETA_F64 * (terms as f64 + 2.0);
            let mut lo = vec![0.0_f64; spec.nout];
            let mut hi = vec![0.0_f64; spec.nout];
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .enumerate()
                .for_each(|(o, (lo_o, hi_o))| {
                    if expired.load(Relaxed) {
                        return;
                    }
                    if Instant::now() >= deadline {
                        expired.store(true, Relaxed);
                        return;
                    }
                    let row = &spec.w[o * spec.nin..(o + 1) * spec.nin];
                    let (mut zc, mut ta, mut zr) = (0.0_f64, 0.0_f64, 0.0_f64);
                    for (j, &wf) in row.iter().enumerate() {
                        let wv = f32_to_f64_exact(wf);
                        if wv == 0.0 {
                            continue;
                        }
                        let wa = wv.abs();
                        zc += wv * cen[j];
                        ta += wa * cen[j].abs();
                        zr += wa * rad[j];
                    }
                    let b = spec.bias[o];
                    let (l, h) = endpoints(zc + b, ta + b.abs(), zr, g, eta);
                    *lo_o = l;
                    *hi_o = h;
                });
            (lo, hi)
        }
        _ => return None,
    };
    if expired.load(Relaxed) || !box_is_valid(&lo, &hi) {
        return None;
    }
    Some((lo, hi))
}

/// Exact max-pool image: `[max lo, max hi]` over each window (max is exact).
pub(crate) fn pool_box(
    spec: &PoolSpec,
    inp: Shape3,
    out: Shape3,
    view: BoxView<'_>,
) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![f64::NEG_INFINITY; out.numel()];
    let mut hi = vec![f64::NEG_INFINITY; out.numel()];
    for c in 0..out.c {
        for oy in 0..out.h {
            for ox in 0..out.w {
                let o = (c * out.h + oy) * out.w + ox;
                for dy in 0..spec.kh {
                    for dx in 0..spec.kw {
                        let j = c * inp.plane() + (oy * spec.sh + dy) * inp.w + ox * spec.sw + dx;
                        lo[o] = lo[o].max(view.lo(j));
                        hi[o] = hi[o].max(view.hi(j));
                    }
                }
            }
        }
    }
    (lo, hi)
}
