// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane stage 3 unit tests for the search step: the relaxed clip
//! against an EXACT-RATIONAL falsifier (with a raw-float negative control
//! showing the falsifier bites), and the split rules' coverage, aspect guard
//! and bit-for-bit replication of the historical rules. The lane-level tests
//! (clip planes on real programs, parallel == serial, end to end) are in
//! `lsnc_tests`.

use num_rational::BigRational;

use super::search::{choose_split, clip_box, plane_faces, ClipPlane, Face, SB_ASPECT};
use super::LowDimSplit;

/// Deterministic LCG in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n - 1)
    }
}

fn q(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite test value")
}

/// One random clip instance, shaped like the numpy gate's falsifier
/// (`lsnc_gate.py --selftest`): planes of assorted scale with some exact
/// zeros, thresholds placed inside each plane's range over the box so the
/// clip bites, some degenerate (zero-width) coordinates.
struct Instance {
    lo: Vec<f64>,
    hi: Vec<f64>,
    planes: Vec<(Vec<f64>, f64, f64)>,
}

fn instance(rng: &mut Lcg) -> Instance {
    let d = 1 + rng.below(8);
    let p = 1 + rng.below(6);
    let sc = 10f64.powf(rng.range(-3.0, 2.0));
    let lo: Vec<f64> = (0..d).map(|_| rng.range(-3.0, 3.0)).collect();
    let hi: Vec<f64> = lo
        .iter()
        .map(|&l| {
            if rng.unit() < 0.9 {
                l + rng.range(1e-6, 2.0)
            } else {
                l
            }
        })
        .collect();
    let mut planes = Vec::with_capacity(p);
    for _ in 0..p {
        let a: Vec<f64> = (0..d)
            .map(|_| {
                if rng.unit() < 0.1 {
                    0.0
                } else {
                    rng.normal() * sc * 10f64.powf(rng.range(-6.0, 0.0))
                }
            })
            .collect();
        let c = rng.normal() * sc * 3.0;
        let (mut mn, mut mx) = (c, c);
        for k in 0..d {
            let (x, y) = (a[k] * lo[k], a[k] * hi[k]);
            mn += x.min(y);
            mx += x.max(y);
        }
        let t = mn + rng.range(-0.1, 1.0) * (mx - mn);
        planes.push((a, c, t));
    }
    Instance { lo, hi, planes }
}

/// EXACT: `c + sum_{m != skip} min(a_m lo_m, a_m hi_m)` over the box.
fn exact_rest(a: &[f64], c: f64, lo: &[f64], hi: &[f64], skip: Option<usize>) -> BigRational {
    let mut r = q(c);
    for m in 0..a.len() {
        if Some(m) == skip {
            continue;
        }
        let (x, y) = (q(a[m]) * q(lo[m]), q(a[m]) * q(hi[m]));
        r += if x < y { x } else { y };
    }
    r
}

/// EXACT certificate of one face: every point of the box beyond it violates
/// the plane, i.e. `a_n v + rest_n >= t` (then `x_n > v` with `a_n > 0`, or
/// `x_n < v` with `a_n < 0`, gives `a . x + c > t`).
fn face_is_certified(a: &[f64], c: f64, t: f64, lo: &[f64], hi: &[f64], f: Face) -> bool {
    let an = a[f.dim];
    if (an > 0.0) != f.upper || an == 0.0 {
        return false;
    }
    q(an) * q(f.value) + exact_rest(a, c, lo, hi, Some(f.dim)) >= q(t)
}

/// THE CLIP FALSIFIER. On 600 random instances: every face every plane
/// certifies, and every "this plane alone empties the box" claim, holds in
/// EXACT rational arithmetic over the unclipped box; `clip_box` returns
/// exactly the intersection of those faces (so the certified faces are the
/// ones the lane uses), never grows the box, and reports empty exactly when a
/// plane is empty or two certified faces cross. Non-vacuity: at least 500
/// faces are checked and at least 50 boxes are really emptied or shrunk.
#[test]
fn every_clip_face_is_exact_rational_sound_and_clip_box_is_their_intersection() {
    let mut rng = Lcg(0x636c_6970_5f71_6564);
    let (mut faces_checked, mut shrunk_or_empty, mut empties) = (0usize, 0usize, 0usize);
    for trial in 0..600 {
        let inst = instance(&mut rng);
        let d = inst.lo.len();
        let (mut nl, mut nh) = (inst.lo.clone(), inst.hi.clone());
        let mut any_empty_plane = false;
        for (a, c, t) in &inst.planes {
            let plane = ClipPlane {
                a: a.as_slice(),
                c: *c,
                t: *t,
            };
            let mut faces = Vec::new();
            let empty = plane_faces(&inst.lo, &inst.hi, &plane, |f| faces.push(f));
            if empty {
                assert!(
                    exact_rest(a, *c, &inst.lo, &inst.hi, None) > q(*t),
                    "trial {trial}: plane claimed EMPTY but its exact minimum over the box does \
                     not exceed t = {t}"
                );
                any_empty_plane = true;
                continue;
            }
            for f in faces {
                faces_checked += 1;
                assert!(
                    face_is_certified(a, *c, *t, &inst.lo, &inst.hi, f),
                    "trial {trial}: face {f:?} of plane a={a:?} c={c} t={t} on box {:?}..{:?} \
                     removes a point of the halfspace (exact-rational falsifier)",
                    inst.lo,
                    inst.hi
                );
                if f.upper {
                    nh[f.dim] = nh[f.dim].min(f.value);
                } else {
                    nl[f.dim] = nl[f.dim].max(f.value);
                }
            }
        }
        let crossed = (0..d).any(|n| nl[n] > nh[n]);
        let planes: Vec<ClipPlane<'_>> = inst
            .planes
            .iter()
            .map(|(a, c, t)| ClipPlane {
                a: a.as_slice(),
                c: *c,
                t: *t,
            })
            .collect();
        let (mut bl, mut bh) = (inst.lo.clone(), inst.hi.clone());
        match clip_box(&mut bl, &mut bh, &planes) {
            None => {
                assert!(
                    any_empty_plane || crossed,
                    "trial {trial}: clip_box reported EMPTY without an empty plane or a crossing"
                );
                assert_eq!(
                    (bl.as_slice(), bh.as_slice()),
                    (inst.lo.as_slice(), inst.hi.as_slice()),
                    "an emptied box is left unchanged"
                );
                empties += 1;
                shrunk_or_empty += 1;
            }
            Some(shrunk) => {
                assert!(
                    !any_empty_plane && !crossed,
                    "trial {trial}: clip_box kept a box a certified face or plane empties"
                );
                assert_eq!(
                    bl, nl,
                    "trial {trial}: lower faces are not the intersection"
                );
                assert_eq!(
                    bh, nh,
                    "trial {trial}: upper faces are not the intersection"
                );
                for k in 0..d {
                    assert!(
                        inst.lo[k] <= bl[k] && bl[k] <= bh[k] && bh[k] <= inst.hi[k],
                        "trial {trial}: the clip grew or inverted coordinate {k}"
                    );
                }
                let moved = (0..d).any(|k| bl[k] > inst.lo[k] || bh[k] < inst.hi[k]);
                assert_eq!(shrunk, moved, "trial {trial}: `shrunk` misreports");
                shrunk_or_empty += usize::from(shrunk);
            }
        }
    }
    eprintln!(
        "[lsnc-clip-test] faces_checked={faces_checked} shrunk_or_empty={shrunk_or_empty} \
         empties={empties}"
    );
    assert!(
        faces_checked >= 500,
        "VOID: only {faces_checked} faces checked"
    );
    assert!(
        shrunk_or_empty >= 50,
        "VOID: the clip bit on only {shrunk_or_empty} boxes"
    );
}

/// NEGATIVE CONTROL: the same falsifier FAILS the raw float clip (abc's
/// closed form, no numerator guard, no outward rounding) on a real fraction
/// of faces — the numpy gate measured 753 of 1525. A falsifier that could
/// not catch that would prove nothing above.
#[test]
fn the_falsifier_catches_the_raw_float_clip() {
    let mut rng = Lcg(0x7261_775f_636c_6970);
    let (mut raw_faces, mut raw_bad) = (0usize, 0usize);
    for _ in 0..300 {
        let inst = instance(&mut rng);
        let d = inst.lo.len();
        for (a, c, t) in &inst.planes {
            for n in 0..d {
                if a[n] == 0.0 {
                    continue;
                }
                let mut num = t - c;
                for m in 0..d {
                    if m != n {
                        num -= (a[m] * inst.lo[m]).min(a[m] * inst.hi[m]);
                    }
                }
                let v = num / a[n];
                if !v.is_finite() {
                    continue;
                }
                let f = Face {
                    dim: n,
                    upper: a[n] > 0.0,
                    value: v,
                };
                raw_faces += 1;
                if !face_is_certified(a, *c, *t, &inst.lo, &inst.hi, f) {
                    raw_bad += 1;
                }
            }
        }
    }
    eprintln!("[lsnc-clip-test] raw float clip: {raw_bad} of {raw_faces} faces fail the falsifier");
    assert!(raw_faces >= 200, "VOID: only {raw_faces} raw faces");
    assert!(
        raw_bad > 0,
        "the exact-rational falsifier never caught the unguarded clip: it cannot fail, so the \
         soundness test above proves nothing"
    );
}

/// Degenerate planes are ignored, never trusted: a non-finite coefficient,
/// constant or threshold, or a length mismatch, certifies nothing.
#[test]
fn a_malformed_plane_certifies_nothing() {
    let (lo, hi) = (vec![0.0, -1.0], vec![1.0, 1.0]);
    for (a, c, t) in [
        (vec![f64::NAN, 1.0], 0.0, 0.0),
        (vec![1.0, f64::INFINITY], 0.0, 0.0),
        (vec![1.0, 1.0], f64::NAN, 0.0),
        (vec![1.0, 1.0], 0.0, f64::NEG_INFINITY),
        (vec![1.0], 0.0, -10.0),
    ] {
        let plane = ClipPlane {
            a: a.as_slice(),
            c,
            t,
        };
        let mut faces = 0usize;
        let empty = plane_faces(&lo, &hi, &plane, |_| faces += 1);
        assert!(
            !empty && faces == 0,
            "a malformed plane {a:?} {c} {t} certified something"
        );
        let (mut bl, mut bh) = (lo.clone(), hi.clone());
        assert_eq!(clip_box(&mut bl, &mut bh, &[plane]), Some(false));
        assert_eq!((bl, bh), (lo.clone(), hi.clone()));
    }
    // A well-formed plane whose minimum over the box exceeds t empties it.
    let a = vec![1.0, 0.0];
    let plane = ClipPlane {
        a: a.as_slice(),
        c: 5.0,
        t: 1.0,
    };
    let (mut bl, mut bh) = (lo, hi);
    assert_eq!(clip_box(&mut bl, &mut bh, &[plane]), None);
}

/// The historical split loops, verbatim from `prove_conjunction_core` before
/// #lsnc-lane stage 3.
fn historical(split: LowDimSplit, l: &[f64], h: &[f64], grad: &[f64]) -> usize {
    let mut best = 0;
    let mut score = f64::NEG_INFINITY;
    for k in 0..l.len() {
        let w = h[k] - l[k];
        let sc = match split {
            LowDimSplit::Widest => w,
            LowDimSplit::Gradient => w * grad[k].abs().max(1e-12),
            LowDimSplit::SbSum | LowDimSplit::SbWr => unreachable!(),
        };
        if sc > score {
            score = sc;
            best = k;
        }
    }
    best
}

/// Random split cases: a root box, a sub-box of it (some coordinates
/// degenerate, some touching zero), gradients for 1-4 objectives.
fn split_case(rng: &mut Lcg) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let d = 1 + rng.below(8);
    let w0: Vec<f64> = (0..d).map(|_| rng.range(0.1, 6.0)).collect();
    let lo0: Vec<f64> = w0.iter().map(|w| -0.5 * w).collect();
    let mut l = Vec::with_capacity(d);
    let mut h = Vec::with_capacity(d);
    for k in 0..d {
        let frac = [1.0, 0.5, 0.1, 1e-3, 0.0][rng.below(5)];
        let w = w0[k] * frac;
        let start = lo0[k] + rng.unit() * (w0[k] - w);
        let (a, b) = if rng.unit() < 0.1 {
            (0.0, w.min(0.5 * w0[k]))
        } else {
            (start, start + w)
        };
        l.push(a);
        h.push(b);
    }
    let k_obj = 1 + rng.below(4);
    let grads: Vec<f64> = (0..k_obj * d)
        .map(|_| {
            if rng.unit() < 0.15 {
                0.0
            } else {
                rng.normal() * 10f64.powf(rng.range(-4.0, 2.0))
            }
        })
        .collect();
    let nearest = rng.below(k_obj);
    (l, h, w0, grads, nearest)
}

/// SPLIT COVERAGE AND GUARD. For every rule, on 4000 random cases: the chosen
/// coordinate is in range; the two children `[l, mid]` / `[mid, h]` along it
/// cover the parent (every sampled point and every corner of the parent lies
/// in one of them) and lie inside it; the smart rules obey the aspect guard
/// (relative width at least 1/8 of the largest); and Widest / Gradient choose
/// EXACTLY what the historical loops chose.
#[test]
fn every_split_rule_covers_the_parent_and_the_smart_rules_obey_the_aspect_guard() {
    let mut rng = Lcg(0x7370_6c69_7473_6221);
    let mut checked = 0usize;
    for case in 0..4000 {
        let (l, h, w0, grads, nearest) = split_case(&mut rng);
        let d = l.len();
        for split in [
            LowDimSplit::Widest,
            LowDimSplit::Gradient,
            LowDimSplit::SbSum,
            LowDimSplit::SbWr,
        ] {
            let dim = choose_split(split, &l, &h, &w0, &grads, nearest);
            assert!(
                dim < d,
                "case {case} {split:?}: coordinate {dim} out of range"
            );
            if matches!(split, LowDimSplit::Widest | LowDimSplit::Gradient) {
                let g = &grads[nearest * d..(nearest + 1) * d];
                assert_eq!(
                    dim,
                    historical(split, &l, &h, g),
                    "case {case} {split:?}: not the historical choice"
                );
            } else {
                let rw: Vec<f64> = (0..d)
                    .map(|k| {
                        let w = h[k] - l[k];
                        if w > 0.0 {
                            w / w0[k]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let rmax = rw.iter().copied().fold(0.0f64, f64::max);
                assert!(
                    rw[dim] >= rmax / SB_ASPECT,
                    "case {case} {split:?}: coordinate {dim} (relative width {}) violates the \
                     aspect guard (max {rmax})",
                    rw[dim]
                );
                if rmax > 0.0 {
                    assert!(
                        h[dim] > l[dim],
                        "case {case} {split:?}: split a zero-width side"
                    );
                }
            }
            let mid = f64::midpoint(l[dim], h[dim]);
            if !(mid > l[dim] && mid < h[dim]) {
                continue; // the lane stops on Resolution here; nothing to cover
            }
            let (mut lh, mut rl) = (h.clone(), l.clone());
            lh[dim] = mid;
            rl[dim] = mid;
            let inside =
                |x: &[f64], lo: &[f64], hi: &[f64]| (0..d).all(|k| lo[k] <= x[k] && x[k] <= hi[k]);
            assert!(
                (0..d).all(|k| l[k] <= lh[k] && rl[k] <= h[k]),
                "case {case}: a child leaves the parent"
            );
            let mut points: Vec<Vec<f64>> = (0..48)
                .map(|_| {
                    (0..d)
                        .map(|k| (l[k] + (h[k] - l[k]) * rng.unit()).clamp(l[k], h[k]))
                        .collect()
                })
                .collect();
            for corner in 0..(1usize << d.min(6)) {
                points.push(
                    (0..d)
                        .map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] })
                        .collect(),
                );
            }
            let mut x = l.clone();
            x[dim] = mid;
            points.push(x);
            for x in &points {
                assert!(
                    inside(x, &l, &lh) || inside(x, &rl, &h),
                    "case {case} {split:?}: point {x:?} is in neither child"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 100_000, "VOID: only {checked} coverage points");
}

/// The smart rules' intent, pinned on hand-built cases.
#[test]
fn the_smart_rules_follow_the_gradient_the_touch_bonus_and_the_guard() {
    let w0 = [2.0, 2.0, 2.0];
    // Equal widths, gradient only on coordinate 1: both smart rules pick it.
    let (l, h) = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let g = [0.0, 10.0, 0.0];
    assert_eq!(choose_split(LowDimSplit::SbSum, &l, &h, &w0, &g, 0), 1);
    assert_eq!(choose_split(LowDimSplit::SbWr, &l, &h, &w0, &g, 0), 1);
    // No gradient, equal widths: sb_sum's touch-zero bonus decides (the box
    // touches 0 on coordinate 2 only).
    let (l, h) = ([0.25, -1.0, 0.0], [1.25, 0.0 - 1e-9, 1.0]);
    let zero = [0.0, 0.0, 0.0];
    let w0b = [1.0, 1.0, 1.0];
    assert_eq!(choose_split(LowDimSplit::SbSum, &l, &h, &w0b, &zero, 0), 2);
    // The aspect guard: a huge gradient on a coordinate already 16x narrower
    // (relative to the root) than the widest is NOT chosen.
    let (l, h) = ([-1.0, 0.0, -1.0], [1.0, 0.125, 1.0]);
    let g = [0.0, 1e6, 1e-3];
    for split in [LowDimSplit::SbSum, LowDimSplit::SbWr] {
        let dim = choose_split(split, &l, &h, &w0, &g, 0);
        assert_ne!(
            dim, 1,
            "{split:?} split a coordinate the aspect guard excludes"
        );
    }
    // NaN gradients never win: the widest eligible coordinate is chosen.
    let (l, h) = ([-1.0, -0.5, -1.0], [1.0, 0.5, 0.5]);
    let g = [f64::NAN, f64::NAN, f64::NAN];
    assert_eq!(choose_split(LowDimSplit::SbWr, &l, &h, &w0, &g, 0), 0);
}
