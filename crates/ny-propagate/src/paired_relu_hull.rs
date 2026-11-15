// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified affine supports for a pair of coupled ReLUs.
//!
//! The caller supplies rigorous enclosures of `zf`, `zg`, and their SIGNED
//! difference. For any finite proposed slopes `af, ag`, this module returns a
//! downward-rounded intercept `k` such that
//!
//! ```text
//! cf * relu(zf) + cg * relu(zg) >= af * zf + ag * zg + k
//! ```
//!
//! throughout the supplied rectangle intersected with the difference strip.
//! Choosing slopes is a heuristic; repairing their intercept is the authority.
//! This is a support oracle for a small joint hull, rather than an independent
//! pair of triangle relaxations. Differential convex planes and multi-neuron
//! hulls have prior art (NeuroDiff and PRIMA); this module makes no priority
//! claim for those ideas.
//!
//! # Geometric and rounding proof
//!
//! Split the clipped rectangle at `zf = 0` and `zg = 0`. The residual of the
//! proposed affine support is affine on each of the four resulting polygons.
//! Its minimum is attained at a vertex, including in degenerate segments and
//! singleton polygons. Every vertex is an intersection of two nonparallel
//! lines from `zf = lf, uf, 0`, `zg = lg, ug, 0`, `zf - zg = dl, du`.
//! There are only 21 nonparallel pairs among these eight lines.
//!
//! Intersections use copies and single sums/differences of f64 endpoints,
//! enclosed with `next_down`/`next_up`. A candidate is discarded only when its
//! interval enclosure is disjoint from a rigorous constraint; interval
//! contraction preserves every exact intersection. The residual is evaluated
//! with outward interval arithmetic, and the minimum of its lower endpoints
//! encloses the true minimum from below. Computing ordinary nearest-rounded
//! candidate slopes therefore cannot invalidate the returned support. All
//! supplied floats and slopes denote exact real numbers, and the arithmetic
//! assumes IEEE-754 binary64 with gradual underflow, as elsewhere in NY.
//!
//! Once both activation signs are fixed, residuals are also evaluated in
//! common/difference coordinates. Their coefficient sums are rounded outward
//! before multiplication. Taking the maximum of several rigorous lower
//! enclosures preserves validity and substantially reduces cancellation loss.

/// Rigorous preactivation and signed difference enclosures.
#[derive(Clone, Copy, Debug)]
pub struct PairedReluRanges {
    pub lf: f64,
    pub uf: f64,
    pub lg: f64,
    pub ug: f64,
    pub dl: f64,
    pub du: f64,
}

/// Prepared clipped vertices, reusable for many objective/slopes proposals on
/// the same neuron and domain. Preparation is objective independent.
#[derive(Clone, Debug)]
pub struct PairedReluHull {
    ranges: PairedReluRanges,
    vertices: Vec<(Interval, Interval)>,
}

impl PairedReluHull {
    /// Number of conservatively retained intersections (duplicates allowed).
    #[cfg(test)]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Certify arbitrary finite slopes against the prepared vertices.
    pub fn certified_intercept(&self, cf: f64, cg: f64, af: f64, ag: f64) -> Option<f64> {
        if [cf, cg, af, ag].iter().any(|x| !x.is_finite()) {
            return None;
        }
        let mut best = f64::INFINITY;
        for &(x, y) in &self.vertices {
            best = best.min(self.ranges.residual_lower(x, y, cf, cg, af, ag)?);
        }
        best.is_finite().then_some(best)
    }
}

#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn invalid() -> Self {
        Self {
            lo: f64::NAN,
            hi: f64::NAN,
        }
    }

    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }

    fn add(self, rhs: Self) -> Self {
        if !self.finite() || !rhs.finite() {
            return Self::invalid();
        }
        Self {
            lo: add_down(self.lo, rhs.lo),
            hi: add_up(self.hi, rhs.hi),
        }
    }

    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    fn mul(self, rhs: Self) -> Self {
        if !self.finite() || !rhs.finite() {
            return Self::invalid();
        }
        if (self.lo == 0.0 && self.hi == 0.0) || (rhs.lo == 0.0 && rhs.hi == 0.0) {
            return Self::point(0.0);
        }
        let pairs = [
            (self.lo, rhs.lo),
            (self.lo, rhs.hi),
            (self.hi, rhs.lo),
            (self.hi, rhs.hi),
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (a, b) in pairs {
            lo = lo.min(mul_down(a, b));
            hi = hi.max(mul_up(a, b));
        }
        Self { lo, hi }
    }

    fn relu(self) -> Self {
        if !self.finite() {
            return Self::invalid();
        }
        Self {
            lo: self.lo.max(0.0),
            hi: self.hi.max(0.0),
        }
    }

    fn intersect(self, rhs: Self) -> Option<Self> {
        if !self.finite() || !rhs.finite() {
            return None;
        }
        let result = Self {
            lo: self.lo.max(rhs.lo),
            hi: self.hi.min(rhs.hi),
        };
        (result.lo <= result.hi).then_some(result)
    }
}

// Exact identities below preserve exact zero cancellation without adding a
// subnormal at every operation. Otherwise one adjacent float safely encloses
// the exact real operation, including underflow.
fn add_down(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        (a + b).next_down()
    }
}

fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        (a + b).next_up()
    }
}

fn mul_down(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else if a == 1.0 {
        b
    } else if b == 1.0 {
        a
    } else if a == -1.0 {
        -b
    } else if b == -1.0 {
        -a
    } else {
        (a * b).next_down()
    }
}

fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else if a == 1.0 {
        b
    } else if b == 1.0 {
        a
    } else if a == -1.0 {
        -b
    } else if b == -1.0 {
        -a
    } else {
        (a * b).next_up()
    }
}

/// Exact comparison of the real difference `a - b` against an f64 endpoint.
/// If the rounded subtraction differs from `c`, monotonicity of rounding
/// resolves the comparison. At equality, an error-free TwoSum recovers which
/// side of `c` the exact subtraction lies on. This makes an empty strip fail
/// closed even if its infeasibility is below one ulp of the rounded difference.
fn difference_cmp(a: f64, b: f64, c: f64) -> std::cmp::Ordering {
    let sum = a - b;
    if sum < c {
        return std::cmp::Ordering::Less;
    }
    if sum > c {
        return std::cmp::Ordering::Greater;
    }
    let negative_b = -b;
    let virtual_b = sum - a;
    let error = (a - (sum - virtual_b)) + (negative_b - virtual_b);
    error.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
}

#[derive(Clone, Copy)]
enum Line {
    F(f64),
    G(f64),
    Delta(f64),
}

fn intersection(a: Line, b: Line) -> Option<(Interval, Interval)> {
    match (a, b) {
        (Line::F(x), Line::G(y)) | (Line::G(y), Line::F(x)) => {
            Some((Interval::point(x), Interval::point(y)))
        }
        (Line::F(x), Line::Delta(d)) | (Line::Delta(d), Line::F(x)) => Some((
            Interval::point(x),
            Interval::point(x).sub(Interval::point(d)),
        )),
        (Line::G(y), Line::Delta(d)) | (Line::Delta(d), Line::G(y)) => Some((
            Interval::point(y).add(Interval::point(d)),
            Interval::point(y),
        )),
        _ => None,
    }
}

impl PairedReluRanges {
    /// Validate finite, ordered intervals and exact nonemptiness of their
    /// intersection. Rejecting an empty domain avoids vacuous certificates.
    pub fn is_valid(self) -> bool {
        [self.lf, self.uf, self.lg, self.ug, self.dl, self.du]
            .iter()
            .all(|x| x.is_finite())
            && self.lf <= self.uf
            && self.lg <= self.ug
            && self.dl <= self.du
            && difference_cmp(self.uf, self.lg, self.dl) != std::cmp::Ordering::Less
            && difference_cmp(self.lf, self.ug, self.du) != std::cmp::Ordering::Greater
    }

    fn clip_vertex(self, mut x: Interval, mut y: Interval) -> Option<(Interval, Interval)> {
        x = x.intersect(Interval {
            lo: self.lf,
            hi: self.uf,
        })?;
        y = y.intersect(Interval {
            lo: self.lg,
            hi: self.ug,
        })?;
        let delta = Interval {
            lo: self.dl,
            hi: self.du,
        };
        // Contract only by outward-rounded implications of x = y + delta.
        // Two rounds keep interval intersections near clipped corners tight.
        for _ in 0..2 {
            x = x.intersect(y.add(delta))?;
            y = y.intersect(x.sub(delta))?;
        }
        x.sub(y).intersect(delta)?;
        Some((x, y))
    }

    /// Rigorous intercept for arbitrary proposed finite slopes. `None` means
    /// malformed, empty, or numerically unbounded, and must never certify a row.
    #[cfg(test)]
    pub fn certified_intercept(self, cf: f64, cg: f64, af: f64, ag: f64) -> Option<f64> {
        self.prepare()?.certified_intercept(cf, cg, af, ag)
    }

    /// Prepare geometry once before back-substituting many specification rows.
    /// Only rigorous ranges are accepted; numerical overflow fails closed.
    pub fn prepare(self) -> Option<PairedReluHull> {
        if !self.is_valid() {
            return None;
        }
        let lines = [
            Line::F(self.lf),
            Line::F(self.uf),
            Line::F(0.0),
            Line::G(self.lg),
            Line::G(self.ug),
            Line::G(0.0),
            Line::Delta(self.dl),
            Line::Delta(self.du),
        ];
        let mut vertices = Vec::with_capacity(21);
        for i in 0..lines.len() {
            for j in i + 1..lines.len() {
                let Some((x, y)) = intersection(lines[i], lines[j]) else {
                    continue;
                };
                if !x.finite() || !y.finite() {
                    return None;
                }
                let Some((x, y)) = self.clip_vertex(x, y) else {
                    continue;
                };
                vertices.push((x, y));
            }
        }
        (!vertices.is_empty()).then_some(PairedReluHull {
            ranges: self,
            vertices,
        })
    }

    fn residual_lower(
        self,
        x: Interval,
        y: Interval,
        cf: f64,
        cg: f64,
        af: f64,
        ag: f64,
    ) -> Option<f64> {
        let scalar = Interval::point;
        let direct = scalar(cf)
            .mul(x.relu())
            .add(scalar(cg).mul(y.relu()))
            .sub(scalar(af).mul(x))
            .sub(scalar(ag).mul(y));
        if !direct.finite() {
            return None;
        }
        let mut lower = direct.lo;
        let sign_f = if x.lo >= 0.0 {
            Some(cf)
        } else if x.hi <= 0.0 {
            Some(0.0)
        } else {
            None
        };
        let sign_g = if y.lo >= 0.0 {
            Some(cg)
        } else if y.hi <= 0.0 {
            Some(0.0)
        } else {
            None
        };
        if let (Some(sf), Some(sg)) = (sign_f, sign_g) {
            let rf = scalar(sf).sub(scalar(af));
            let rg = scalar(sg).sub(scalar(ag));
            // Sum the original coefficients first: exact cf=-cg and af=-ag
            // remain exact zero, preserving common-mode cancellation.
            let common = scalar(sf).add(scalar(sg)).sub(scalar(af).add(scalar(ag)));
            let delta = x.sub(y).intersect(Interval {
                lo: self.dl,
                hi: self.du,
            })?;
            for bound in [
                rf.mul(x).add(rg.mul(y)),
                common.mul(y).add(rf.mul(delta)),
                common.mul(x).sub(rg.mul(delta)),
            ] {
                if bound.finite() {
                    lower = lower.max(bound.lo);
                }
            }
        }
        lower.is_finite().then_some(lower)
    }

    /// Small proposal portfolio: independent triangles, signed difference
    /// secants, and mixed common/difference slopes. These are ordinary rounded
    /// proposals; callers must certify EACH chosen pair with
    /// `certified_intercept` before using it in a lower bound.
    pub fn candidate_slopes(self, cf: f64, cg: f64) -> Vec<(f64, f64)> {
        if !self.is_valid() || !cf.is_finite() || !cg.is_finite() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut push = |af: f64, ag: f64| {
            if af.is_finite() && ag.is_finite() && !result.iter().any(|&(a, b)| a == af && b == ag)
            {
                result.push((af, ag));
            }
        };
        let independent_f = ordinary_slopes(cf, self.lf, self.uf);
        let independent_g = ordinary_slopes(cg, self.lg, self.ug);
        for &af in &independent_f {
            for &ag in &independent_g {
                push(af, ag);
            }
        }
        if (cf > 0.0 && cg < 0.0) || (cf < 0.0 && cg > 0.0) {
            let cancelled = cf.abs().min(cg.abs()).copysign(cf);
            let residual_f = ordinary_slopes(cf - cancelled, self.lf, self.uf);
            let residual_g = ordinary_slopes(cg + cancelled, self.lg, self.ug);
            let signed_secant = if cancelled > 0.0 {
                if self.dl >= 0.0 {
                    0.0
                } else if self.du <= 0.0 {
                    1.0
                } else {
                    -self.dl / (self.du - self.dl)
                }
            } else if self.dl >= 0.0 {
                1.0
            } else if self.du <= 0.0 {
                0.0
            } else {
                self.du / (self.du - self.dl)
            };
            for fraction in [signed_secant, 0.0, 0.25, 0.5, 0.75, 1.0] {
                let a = cancelled * fraction;
                for &rf in &residual_f {
                    for &rg in &residual_g {
                        push(rf + a, rg - a);
                    }
                }
            }
        }
        result
    }
}

fn ordinary_slopes(c: f64, lo: f64, hi: f64) -> Vec<f64> {
    if c == 0.0 || hi <= 0.0 {
        vec![0.0]
    } else if lo >= 0.0 {
        vec![c]
    } else if c > 0.0 {
        vec![0.0, c]
    } else {
        vec![c * (hi / (hi - lo))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric(cap: f64) -> PairedReluRanges {
        PairedReluRanges {
            lf: -2.0,
            uf: 2.0,
            lg: -2.0,
            ug: 2.0,
            dl: -cap,
            du: cap,
        }
    }

    fn assert_grid_sound(r: PairedReluRanges, cf: f64, cg: f64, af: f64, ag: f64) {
        let k = r.certified_intercept(cf, cg, af, ag).unwrap();
        for i in 0..=80 {
            let x = r.lf + (r.uf - r.lf) * (i as f64 / 80.0);
            for j in 0..=80 {
                let y = r.lg + (r.ug - r.lg) * (j as f64 / 80.0);
                if x - y < r.dl || x - y > r.du {
                    continue;
                }
                let true_value = cf * x.max(0.0) + cg * y.max(0.0);
                let affine = af * x + ag * y + k;
                let evaluation_error =
                    16.0 * f64::EPSILON * (true_value.abs() + affine.abs() + 1.0);
                assert!(
                    affine <= true_value + evaluation_error,
                    "{r:?}: ({x}, {y}), ({cf}, {cg}; {af}, {ag}, {k})"
                );
            }
        }
    }

    #[test]
    fn symmetric_difference_secant_has_analytical_intercept() {
        // min(relu(x)-relu(y) - (x-y)/2) is exactly -cap/2,
        // attained both in the negative quadrant at d=cap and in the
        // positive quadrant at d=-cap.
        let r = symmetric(0.25);
        let k = r.certified_intercept(1.0, -1.0, 0.5, -0.5).unwrap();
        assert!((-0.125 - 1e-14..=-0.125).contains(&k));
        assert_grid_sound(r, 1.0, -1.0, 0.5, -0.5);
    }

    #[test]
    fn asymmetric_difference_secant_beats_symmetric_cap() {
        // Lower secant of min(d,0) on [-1/8,3/8] has slope 1/4
        // and intercept -3/32. The old symmetric cap gives -3/16.
        let mut r = symmetric(0.375);
        r.dl = -0.125;
        let k = r.certified_intercept(1.0, -1.0, 0.25, -0.25).unwrap();
        assert!((-0.09375 - 1e-14..=-0.09375).contains(&k));
        assert!(k > -0.1875);
        assert_grid_sound(r, 1.0, -1.0, 0.25, -0.25);
    }

    #[test]
    fn shifted_signed_strip_certifies_monotone_relu_order() {
        let mut r = symmetric(0.5);
        r.dl = 0.125;
        let k = r.certified_intercept(1.0, -1.0, 0.0, 0.0).unwrap();
        assert!((-1e-14..=0.0).contains(&k));
        assert_grid_sound(r, 1.0, -1.0, 0.0, 0.0);
        assert!(r.candidate_slopes(1.0, -1.0).contains(&(0.0, 0.0)));
    }

    #[test]
    fn identical_activations_preserve_exact_common_cancellation() {
        let mut r = symmetric(0.0);
        r.lf = -1e12;
        r.uf = 1e12;
        r.lg = r.lf;
        r.ug = r.uf;
        assert_eq!(r.certified_intercept(1.0, -1.0, 0.5, -0.5), Some(0.0));
    }

    #[test]
    fn zero_width_and_touching_domains_are_retained() {
        let singleton = PairedReluRanges {
            lf: 0.0,
            uf: 0.0,
            lg: -0.25,
            ug: -0.25,
            dl: 0.25,
            du: 0.25,
        };
        assert!(singleton.is_valid());
        let k = singleton.certified_intercept(3.0, -2.0, 1.0, -1.0).unwrap();
        assert!((-0.25 - 1e-14..=-0.25).contains(&k));
        let segment = PairedReluRanges {
            lf: 0.0,
            uf: 0.0,
            lg: -1.0,
            ug: 1.0,
            dl: -1.0,
            du: 1.0,
        };
        assert_grid_sound(segment, 2.0, -1.0, 0.75, -0.5);
        let touching = PairedReluRanges {
            lf: -1.0,
            uf: 0.0,
            lg: 0.0,
            ug: 1.0,
            dl: 0.0,
            du: 0.0,
        };
        assert!(touching.is_valid());
        assert_eq!(
            touching.certified_intercept(4.0, -3.0, 2.0, -2.0),
            Some(0.0)
        );
    }

    #[test]
    fn empty_and_nonfinite_inputs_fail_closed() {
        let mut r = symmetric(0.5);
        r.dl = 5.0;
        r.du = 6.0;
        assert!(!r.is_valid());
        assert_eq!(r.certified_intercept(1.0, -1.0, 0.0, 0.0), None);
        r = symmetric(0.5);
        r.lf = f64::NAN;
        assert_eq!(r.certified_intercept(1.0, -1.0, 0.0, 0.0), None);
        r = symmetric(0.5);
        assert_eq!(r.certified_intercept(f64::INFINITY, 1.0, 0.0, 0.0), None);
        r.uf = r.lf.next_down();
        assert!(!r.is_valid());
    }

    #[test]
    fn infeasible_strip_below_one_ulp_is_rejected_exactly() {
        // Exact 1 - 2^-54 lies below the rounded value 1.0. A strip
        // requiring difference >=1 must be rejected, not treated as touching.
        let r = PairedReluRanges {
            lf: 1.0,
            uf: 1.0,
            lg: 2f64.powi(-54),
            ug: 2f64.powi(-54),
            dl: 1.0,
            du: 1.0,
        };
        assert_eq!(r.uf - r.lg, 1.0);
        assert!(!r.is_valid());
        assert_eq!(r.certified_intercept(1.0, -1.0, 0.5, -0.5), None);
    }

    #[test]
    fn mixed_coefficients_and_arbitrary_slopes_are_sound() {
        let cases = [
            (symmetric(0.25), 2.0, -3.0),
            (symmetric(0.75), -2.0, 3.0),
            (symmetric(1.25), 1.5, 0.75),
            (symmetric(0.0), -1.5, -0.75),
            (
                PairedReluRanges {
                    lf: -0.125,
                    uf: 2.0,
                    lg: -1.0,
                    ug: 0.5,
                    dl: -0.25,
                    du: 1.75,
                },
                0.125,
                -4.0,
            ),
        ];
        for (r, cf, cg) in cases {
            for (af, ag) in r.candidate_slopes(cf, cg).into_iter().chain([
                (3.25, -7.0),
                (-0.625, 2.75),
                (0.0, 0.0),
            ]) {
                assert_grid_sound(r, cf, cg, af, ag);
            }
        }
    }

    #[test]
    fn certified_support_is_below_exact_integer_lattice_oracle() {
        // All endpoints are integer multiples of 1/16. The rectangle/strip
        // normals are unimodular, so every vertex, including ReLU kink
        // intersections, lies on this lattice. Enumerating ALL lattice
        // points therefore computes the exact continuous residual minimum;
        // i64 arithmetic below is independent of floating interval arithmetic.
        for case in 0i64..160 {
            let lf = -20 + case % 11;
            let uf = 4 + case % 17;
            let lg = -18 + case % 13;
            let ug = 3 + case % 19;
            let dl = -12 + case % 9;
            let du = 2 + case % 14;
            let cf = -16 + case % 33;
            let cg = -16 + (case * 7) % 33;
            let af = -24 + (case * 11) % 49;
            let ag = -24 + (case * 13) % 49;
            let r = PairedReluRanges {
                lf: lf as f64 / 16.0,
                uf: uf as f64 / 16.0,
                lg: lg as f64 / 16.0,
                ug: ug as f64 / 16.0,
                dl: dl as f64 / 16.0,
                du: du as f64 / 16.0,
            };
            let mut exact_numerator = i64::MAX;
            for x in lf..=uf {
                for y in lg..=ug {
                    if x - y >= dl && x - y <= du {
                        let residual = cf * x.max(0) + cg * y.max(0) - af * x - ag * y;
                        exact_numerator = exact_numerator.min(residual);
                    }
                }
            }
            if exact_numerator == i64::MAX {
                assert_eq!(
                    r.certified_intercept(
                        cf as f64 / 8.0,
                        cg as f64 / 8.0,
                        af as f64 / 8.0,
                        ag as f64 / 8.0
                    ),
                    None
                );
                continue;
            }
            let exact_minimum = exact_numerator as f64 / 128.0;
            let k = r
                .certified_intercept(
                    cf as f64 / 8.0,
                    cg as f64 / 8.0,
                    af as f64 / 8.0,
                    ag as f64 / 8.0,
                )
                .unwrap();
            assert!(
                k <= exact_minimum,
                "case {case}: {k} > exact {exact_minimum}"
            );
            assert!(exact_minimum - k < 1e-12 * (1.0 + exact_minimum.abs()));
        }
    }

    #[test]
    fn prepared_geometry_matches_one_shot_certification() {
        let r = symmetric(0.375);
        let hull = r.prepare().unwrap();
        assert!(hull.vertex_count() > 0 && hull.vertex_count() <= 21);
        for (af, ag) in r.candidate_slopes(2.0, -3.0) {
            assert_eq!(
                hull.certified_intercept(2.0, -3.0, af, ag),
                r.certified_intercept(2.0, -3.0, af, ag)
            );
        }
    }

    #[test]
    fn rectangle_and_strip_vertices_both_matter() {
        // A triangle-only support loses a full unit although the paired
        // network is identical. A strip-only relaxation also misses that
        // both activations are always positive in the second example.
        assert_eq!(
            symmetric(0.0).certified_intercept(1.0, -1.0, 0.0, 0.0),
            Some(0.0)
        );
        let positive = PairedReluRanges {
            lf: 1.0,
            uf: 2.0,
            lg: 0.5,
            ug: 1.5,
            dl: -0.5,
            du: 0.5,
        };
        assert_eq!(
            positive.certified_intercept(1.0, -1.0, 1.0, -1.0),
            Some(0.0)
        );
    }

    #[test]
    fn relu_kink_vertices_are_required_for_a_valid_support() {
        // Checking only rectangle corners would report intercept +1/2 for
        // relu(x)-x/2 on [-1,1], which is false at x=0. The activation
        // partition lines must participate in vertex enumeration.
        let r = PairedReluRanges {
            lf: -1.0,
            uf: 1.0,
            lg: -1.0,
            ug: 1.0,
            dl: -2.0,
            du: 2.0,
        };
        assert_eq!(r.certified_intercept(1.0, 0.0, 0.5, 0.0), Some(0.0));
    }
}
