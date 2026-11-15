// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lowdim-bundle (1) — admission by VARYING input dimensions.
//!
//! # Why
//!
//! dist_shift_2023 (`mnist_concat`): 792 inputs of which 784 are PINNED
//! (`lo == hi`) and 8 are free latent coordinates. The lane's admission
//! counted TOTAL dimensions (`MAX_INPUT_DIM = 16`) and declined the row, while
//! the field (alpha-beta-CROWN, Vibecheck `input_split_max_dims: 800`) splits
//! only the 8 varying dims. This module re-expresses a program lowered over
//! the FULL input as an equivalent program over the free coordinates only.
//!
//! # Which coordinates are pinned (#m8-cgan-leaf-batch)
//!
//! A coordinate is PINNED when its interval is at most a few `f32` ulps wide
//! ([`pin_kind`]): EXACT (`lo == hi`) or NARROW (`lo < hi` within the
//! tolerance). The lane's entry box is an `f32` tensor built by the VNN-LIB
//! loader's `split_input_bounds_f32`, which widens EVERY endpoint outward by
//! one `f32` ulp (#2658, deliberate) and snaps a subnormal endpoint outward
//! to `0` / `+-f32::MIN_POSITIVE`; so a declared point `X_j == v` arrives as
//! `[next_down(v), next_up(v)]` and never as `lo == hi`. On dist_shift that
//! made all 792 coordinates look free (`d_free=792 pinned=0`, W10B
//! `LDB_d2734_dist_r1`). The tolerance decides only which coordinates the
//! lane may SPLIT (completeness): a pinned coordinate is folded as its
//! INTERVAL, never as a point, so the enclosure holds for any width.
//!
//! ROUTER (R1 pending; data only): `lowdim-dist-shift` is an ADMISSION
//! change of the lowdim lane, not a new unit of work. Admission: `admit_with`
//! / `PinnedFold::of` ([`pin_kind`]) under `lowdim_varying_dims`; progress,
//! safe points and stop reasons are the lane's own (Volume / Wave,
//! BetweenUnits at the wave top; deadline, queue, resolution, candidate,
//! nonfinite); resources: the lane's (`MAX_TOTAL_NEURONS`, `CHUNK_BYTES`);
//! knobs: none (the pin tolerance is a classification, not a tuning knob).
//!
//! # Soundness
//!
//! * A new input node carries the free coordinates; an EMBED node (an
//!   [`Op::Sparse`] with one row per full coordinate) reproduces the full
//!   input: a free row is `1 * x_free[slot]` (coefficient exact, no error), a
//!   pinned row is the constant interval `[lo_j, hi_j]` (the box itself,
//!   exact or narrow). Every other op is the original op with its sources
//!   renumbered, so the program encloses the same function of the full input
//!   on the box.
//! * A `Linear` that reads the INPUT (directly or through exact flat
//!   identities) is folded instead: its free
//!   columns keep their exact weights, and its pinned columns move into the
//!   bias INTERVAL `[b_lo + sum min(W_ij lo_j, W_ij hi_j), b_hi + sum
//!   max(W_ij lo_j, W_ij hi_j)]`, accumulated in f64 and widened by
//!   `gslack(n + 2, |b| + sum max|W_ij c_j|)` then rounded outward — a
//!   rigorous enclosure of the exact sum over every point of the pinned
//!   intervals (Higham, any order).
//! * Admission CROSS-CHECKS the folded program against the unfolded one at
//!   three interior points (a guard against a mapping bug, not a proof), and
//!   declines on any disagreement.
//! * A violation CANDIDATE found on the folded program is EXPANDED back to the
//!   full input before it leaves the lane, so every caller's confirmer sees a
//!   full-dimension point: an exact pin at its value, a narrow pin at its
//!   interval's midpoint (inside the box; for a one-ulp outward widening of a
//!   declared `f32` point that is the declared point itself, which the
//!   confirmers' declared-bounds check needs). The trusted forward and the
//!   ORT gate still decide any `sat`.

use std::sync::Arc;

use super::{
    gslack, linear_parts, next_down, next_up, point_values, sparse_op, term, Op, Program,
    SparseTerm, MAX_INPUT_DIM, MAX_TOTAL_NEURONS,
};

/// How one input coordinate of a root box is treated (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinKind {
    /// `lo == hi`.
    Exact,
    /// `lo < hi`, at most [`pin_tolerance`] wide: folded as its interval.
    Narrow,
    /// Wider: a lane coordinate the search may split.
    Free,
}

/// Width of one `f32` ulp at magnitude `x` (an `f32`-representable value
/// carried in f64), or `0` when `x` is outside the finite `f32` range. The
/// spacing ABOVE `|x|` except at `f32::MAX`, where it is the spacing below.
fn f32_ulp_at(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() || a > f64::from(f32::MAX) {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let v = a as f32;
    if v == f32::MAX {
        return f64::from(v) - f64::from(f32::from_bits(v.to_bits() - 1));
    }
    f64::from(f32::from_bits(v.to_bits() + 1)) - f64::from(v)
}

/// The widest interval classified [`PinKind::Narrow`] at `[lo, hi]`:
/// `4 f32 ulps at max(|lo|, |hi|) + 2 f32::MIN_POSITIVE`. The loader's
/// widening is `2` ulps (one each side; `1.5` at a binade edge), or at most
/// `2 f32::MIN_POSITIVE` for a declared `0` after the subnormal snap
/// (`dist_shift_entry_box_widening_is_ulp_scale` measures it against the
/// real loader). Sums of powers of two: exact in f64.
#[must_use]
pub fn pin_tolerance(lo: f64, hi: f64) -> f64 {
    4.0 * f32_ulp_at(lo.abs().max(hi.abs())) + 2.0 * f64::from(f32::MIN_POSITIVE)
}

/// Classify one coordinate `[lo, hi]` (finite, `lo <= hi`; anything else is
/// [`PinKind::Free`], and the admission's own checks refuse it).
#[must_use]
pub fn pin_kind(lo: f64, hi: f64) -> PinKind {
    if !(lo.is_finite() && hi.is_finite()) || lo > hi {
        return PinKind::Free;
    }
    if lo == hi {
        PinKind::Exact
    } else if hi - lo <= pin_tolerance(lo, hi) {
        PinKind::Narrow
    } else {
        PinKind::Free
    }
}

/// Per-kind coordinate counts of a box (the `[lowdim-varying]` fields).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PinCounts {
    /// [`PinKind::Exact`] coordinates.
    pub exact: usize,
    /// [`PinKind::Narrow`] coordinates.
    pub narrow: usize,
    /// [`PinKind::Free`] coordinates.
    pub free: usize,
}

/// Classify every coordinate of `[lo, hi]`.
#[must_use]
pub fn pin_counts(lo: &[f64], hi: &[f64]) -> PinCounts {
    let mut counts = PinCounts::default();
    for (&l, &h) in lo.iter().zip(hi) {
        match pin_kind(l, h) {
            PinKind::Exact => counts.exact += 1,
            PinKind::Narrow => counts.narrow += 1,
            PinKind::Free => counts.free += 1,
        }
    }
    counts
}

/// The free coordinates of a root box whose other coordinates are pinned.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PinnedFold {
    /// Flat indices of the [`PinKind::Free`] coordinates, ascending.
    free: Vec<usize>,
    /// The full root box (pinned coordinates are exact or narrow).
    full_lo: Vec<f64>,
    full_hi: Vec<f64>,
    /// Coordinate counts by kind.
    counts: PinCounts,
}

/// Why a box was or was not folded — the `[lowdim-varying]` status token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FoldStatus {
    /// Folded to its free coordinates.
    Folded,
    /// The full dimension is already admitted: nothing to do.
    WithinCap,
    /// No coordinate is pinned, or the free count is 0 or above the cap.
    Declined,
}

impl FoldStatus {
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::Folded => "folded",
            Self::WithinCap => "within_cap",
            Self::Declined => "declined",
        }
    }
}

impl PinnedFold {
    /// The fold of an f64 root box whose FULL dimension exceeds
    /// [`MAX_INPUT_DIM`]: `Ok(Some(fold))` when 1..=MAX coordinates vary,
    /// `Ok(None)` when the full dimension is already admitted, `Err` (a
    /// decline reason) otherwise. The box must be finite and ordered.
    pub(crate) fn of(lo: &[f64], hi: &[f64]) -> Result<Option<Self>, String> {
        let d = lo.len();
        if d != hi.len() || d == 0 {
            return Err(format!("box of {d}/{} dims", hi.len()));
        }
        if d <= MAX_INPUT_DIM {
            return Ok(None);
        }
        if lo.iter().chain(hi).any(|v| !v.is_finite()) || lo.iter().zip(hi).any(|(l, h)| l > h) {
            return Err("input box is not a finite, ordered box".into());
        }
        let free: Vec<usize> = (0..d)
            .filter(|&i| pin_kind(lo[i], hi[i]) == PinKind::Free)
            .collect();
        if free.is_empty() || free.len() > MAX_INPUT_DIM {
            return Err(format!(
                "{} varying of {d} input dimensions (the lane admits 1..={MAX_INPUT_DIM})",
                free.len()
            ));
        }
        Ok(Some(Self {
            free,
            full_lo: lo.to_vec(),
            full_hi: hi.to_vec(),
            counts: pin_counts(lo, hi),
        }))
    }

    /// Coordinate counts by kind.
    pub(crate) fn counts(&self) -> PinCounts {
        self.counts
    }

    /// Full input dimension.
    pub(crate) fn full_dim(&self) -> usize {
        self.full_lo.len()
    }

    /// Free (lane) dimension.
    pub(crate) fn free_dim(&self) -> usize {
        self.free.len()
    }

    /// B2.2: the FULL box of a free-coordinate box: free coordinates from
    /// `[lo, hi]`, every pinned coordinate over its whole root range (the
    /// folded program's interval biases hold for every value in it, so a
    /// certificate over the free box holds over this full box).
    pub(crate) fn embed_box(&self, lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut flo = self.full_lo.clone();
        let mut fhi = self.full_hi.clone();
        for (k, &i) in self.free.iter().enumerate() {
            if let (Some(&l), Some(&h)) = (lo.get(k), hi.get(k)) {
                flo[i] = l;
                fhi[i] = h;
            }
        }
        (flo, fhi)
    }

    /// The full root box.
    pub(crate) fn full_box(&self) -> (&[f64], &[f64]) {
        (&self.full_lo, &self.full_hi)
    }

    /// The free coordinates of a full vector.
    pub(crate) fn restrict(&self, v: &[f64]) -> Vec<f64> {
        self.free.iter().map(|&i| v[i]).collect()
    }

    /// The full point of a free-coordinate point: pinned coordinates at their
    /// interval's midpoint (an exact pin's own value; inside the box for a
    /// narrow one), free ones clamped into the root box.
    pub(crate) fn expand(&self, x_free: &[f64]) -> Vec<f64> {
        let mut x: Vec<f64> = self
            .full_lo
            .iter()
            .zip(&self.full_hi)
            .map(|(&l, &h)| (l + 0.5 * (h - l)).max(l).min(h))
            .collect();
        for (k, &i) in self.free.iter().enumerate() {
            if let Some(&v) = x_free.get(k) {
                x[i] = v.max(self.full_lo[i]).min(self.full_hi[i]);
            }
        }
        x
    }

    /// Re-express `prog` (lowered over the full input) over the free
    /// coordinates. See the module docs for why the result computes the
    /// same function on the box. Also returns how many input `Linear`s were
    /// folded (observable only).
    pub(crate) fn apply(&self, prog: Program) -> Result<(Program, usize), String> {
        let d_full = self.full_dim();
        let d_free = self.free_dim();
        if prog.d != d_full || prog.sizes.first() != Some(&d_full) {
            return Err(format!(
                "program of {} inputs for a {d_full}-dimension box",
                prog.d
            ));
        }
        let mut slot = vec![usize::MAX; d_full];
        for (k, &i) in self.free.iter().enumerate() {
            slot[i] = k;
        }
        // Old index i >= 1 becomes i + 1; old 0 (the input) is read through
        // the embed node, index 1.
        let map = |i: usize| if i == 0 { 1 } else { i + 1 };
        let map_u32 = |i: u32| -> Result<u32, String> {
            u32::try_from(map(i as usize)).map_err(|_| "renumbered index exceeds u32".to_string())
        };

        let mut ops: Vec<Op> = Vec::with_capacity(prog.ops.len() + 1);
        let mut sizes: Vec<usize> = Vec::with_capacity(prog.sizes.len() + 1);
        ops.push(Op::Input);
        sizes.push(d_free);
        // The embed: element j of the full input.
        let mut terms: Vec<Vec<SparseTerm>> = Vec::with_capacity(d_full);
        let (mut k_lo, mut k_hi) = (Vec::with_capacity(d_full), Vec::with_capacity(d_full));
        for j in 0..d_full {
            if slot[j] == usize::MAX {
                terms.push(Vec::new());
                k_lo.push(self.full_lo[j]);
                k_hi.push(self.full_hi[j]);
            } else {
                terms.push(vec![term(0, slot[j], 1.0, 0.0)?]);
                k_lo.push(0.0);
                k_hi.push(0.0);
            }
        }
        ops.push(sparse_op(terms, k_lo, k_hi));
        sizes.push(d_full);

        // A node that IS the input: the input itself, or a chain of exact
        // flat identities (Reshape / Flatten / Squeeze) of it.
        let mut input_alias = vec![false; prog.ops.len()];
        for (i, op) in prog.ops.iter().enumerate() {
            input_alias[i] = match op {
                Op::Input => true,
                Op::Identity { src } => input_alias.get(*src).copied().unwrap_or(false),
                _ => false,
            };
        }
        let mut folded_linears = 0usize;
        for (old, op) in prog.ops.into_iter().enumerate().skip(1) {
            let new = match op {
                Op::Input => return Err(format!("a second input op at {old}")),
                Op::Linear {
                    src,
                    n_in,
                    w,
                    bias_lo,
                    bias_hi,
                    ..
                } if input_alias.get(src).copied().unwrap_or(false) => {
                    folded_linears += 1;
                    self.fold_linear(&w, n_in, &bias_lo, &bias_hi)?
                }
                Op::Linear {
                    src,
                    n_in,
                    w_pos,
                    w_neg,
                    w,
                    bias_lo,
                    bias_hi,
                    row_abs,
                } => Op::Linear {
                    src: map(src),
                    n_in,
                    w_pos,
                    w_neg,
                    w,
                    bias_lo,
                    bias_hi,
                    row_abs,
                },
                Op::Sparse {
                    terms,
                    k_lo,
                    k_hi,
                    fanin,
                } => {
                    let mut renamed = Vec::with_capacity(terms.len());
                    for row in terms {
                        let mut r = Vec::with_capacity(row.len());
                        for t in row {
                            r.push(SparseTerm {
                                src: map_u32(t.src)?,
                                ..t
                            });
                        }
                        renamed.push(r);
                    }
                    Op::Sparse {
                        terms: renamed,
                        k_lo,
                        k_hi,
                        fanin,
                    }
                }
                Op::Relu { src } => Op::Relu { src: map(src) },
                Op::Sigmoid { src } => Op::Sigmoid { src: map(src) },
                Op::Slice { src, start } => Op::Slice {
                    src: map(src),
                    start,
                },
                Op::Concat { srcs } => Op::Concat {
                    srcs: srcs.into_iter().map(map).collect(),
                },
                Op::Identity { src } => Op::Identity { src: map(src) },
                Op::Add { a, b } => Op::Add {
                    a: map(a),
                    b: map(b),
                },
                Op::Sub { a, b } => Op::Sub {
                    a: map(a),
                    b: map(b),
                },
                Op::AddConst { src, k } => Op::AddConst { src: map(src), k },
                Op::RevSubConst { src, k } => Op::RevSubConst { src: map(src), k },
                Op::MulConst { src, k } => Op::MulConst { src: map(src), k },
                Op::MaxGlobal { src } => Op::MaxGlobal { src: map(src) },
                Op::Mul { a, b } => Op::Mul {
                    a: map(a),
                    b: map(b),
                },
            };
            ops.push(new);
            sizes.push(prog.sizes[old]);
        }
        let total: usize = sizes.iter().sum();
        if total > MAX_TOTAL_NEURONS {
            return Err(format!(
                "{total} neurons exceed the lane cap {MAX_TOTAL_NEURONS}"
            ));
        }
        let old_names: Vec<(usize, String)> = prog
            .sites
            .iter()
            .map(|s| (map(s.node), s.name.clone()))
            .collect();
        let mut out = Program {
            ops,
            sizes,
            out: map(prog.out),
            d: d_free,
            const_terms: 4 * total + 64,
            total,
            seed: map(prog.seed),
            refine_max: prog.refine_max,
            feeds_relu: Vec::new(),
            mul_operand: Vec::new(),
            bilinear: false,
            sites: Vec::new(),
            site_of: Vec::new(),
            site_operand: Vec::new(),
            joint: prog.joint,
        };
        out.derive_structure();
        for site in &mut out.sites {
            if let Some((_, name)) = old_names.iter().find(|(node, _)| *node == site.node) {
                site.name.clone_from(name);
            }
        }
        Ok((out, folded_linears))
    }

    /// A `Linear` reading the full input, folded onto the free coordinates:
    /// the free columns keep their exact weights; the pinned columns join the
    /// bias interval (rigorous, see the module docs).
    fn fold_linear(
        &self,
        w: &Arc<[f64]>,
        n_in: usize,
        bias_lo: &[f64],
        bias_hi: &[f64],
    ) -> Result<Op, String> {
        let d_free = self.free_dim();
        if n_in != self.full_dim() || n_in == 0 || !w.len().is_multiple_of(n_in) {
            return Err("an input Linear does not span the full input".into());
        }
        let n_out = w.len() / n_in;
        if bias_lo.len() != n_out || bias_hi.len() != n_out {
            return Err("an input Linear has a malformed bias".into());
        }
        let pinned: Vec<usize> = (0..n_in).filter(|i| !self.free.contains(i)).collect();
        let mut sub = vec![0.0f64; n_out * d_free];
        let (mut new_lo, mut new_hi) = (Vec::with_capacity(n_out), Vec::with_capacity(n_out));
        for i in 0..n_out {
            let row = &w[i * n_in..(i + 1) * n_in];
            for (k, &f) in self.free.iter().enumerate() {
                sub[i * d_free + k] = row[f];
            }
            let (mut vlo, mut vhi) = (bias_lo[i], bias_hi[i]);
            let mut mag = bias_lo[i].abs().max(bias_hi[i].abs());
            for &j in &pinned {
                // The pinned coordinate's whole interval (exact: `lo == hi`;
                // narrow: a few f32 ulps), never a point of it.
                let (cl, ch) = (self.full_lo[j], self.full_hi[j]);
                let wj = row[j];
                let (pl, ph) = if wj >= 0.0 {
                    (wj * cl, wj * ch)
                } else {
                    (wj * ch, wj * cl)
                };
                vlo += pl;
                vhi += ph;
                mag += pl.abs().max(ph.abs());
            }
            let e = gslack(pinned.len() + 2, mag);
            let (lo, hi) = (next_down(vlo - e), next_up(vhi + e));
            if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
                return Err("an input Linear's folded bias is not finite".into());
            }
            new_lo.push(lo);
            new_hi.push(hi);
        }
        let (w_pos, w_neg, row_abs) = linear_parts(&sub, n_out, d_free);
        Ok(Op::Linear {
            src: 0,
            n_in: d_free,
            w_pos,
            w_neg,
            w: sub.into(),
            bias_lo: new_lo,
            bias_hi: new_hi,
            row_abs,
        })
    }

    /// Admission guard: the folded program and the unfolded one agree at the
    /// free box's centre and two interior points (relative `1e-6`). Not a
    /// proof — a guard against a fold that computes a different function.
    pub(crate) fn cross_check(&self, full: &Program, folded: &Program) -> Result<(), String> {
        const FRACS: [f64; 3] = [0.5, 0.3125, 0.6875];
        for (p, &f) in FRACS.iter().enumerate() {
            let x_free: Vec<f64> = self
                .free
                .iter()
                .enumerate()
                .map(|(k, &i)| {
                    let g = if p == 0 {
                        f
                    } else {
                        FRACS[(p + k) % FRACS.len()]
                    };
                    self.full_lo[i] + (self.full_hi[i] - self.full_lo[i]) * g
                })
                .collect();
            let a = point_values(folded, &x_free);
            let b = point_values(full, &self.expand(&x_free));
            if a.len() != b.len() {
                return Err(format!(
                    "varying-dims cross-check: {} vs {} outputs",
                    a.len(),
                    b.len()
                ));
            }
            for (j, (u, v)) in a.iter().zip(&b).enumerate() {
                let tol = 1e-6 * (1.0 + u.abs().max(v.abs()));
                if !(u.is_finite() && v.is_finite()) || (u - v).abs() > tol {
                    return Err(format!(
                        "varying-dims cross-check failed at output {j}, point {p}: folded {u:e} vs \
                         full {v:e}"
                    ));
                }
            }
        }
        Ok(())
    }
}
