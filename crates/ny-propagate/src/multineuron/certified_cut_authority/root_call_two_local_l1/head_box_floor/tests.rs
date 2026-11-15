// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact rational contraction oracles; no benchmark or production-engagement credit.
use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive};
use std::time::Duration;

const IDS: [[usize; 8]; 2] = [[0, 1, 2, 0, 0, 0, 0, 0], [3, 4, 5, 0, 0, 0, 0, 0]];
const WEIGHTS: [[f32; 8]; 2] = [[1., 1., 1., 0., 0., 0., 0., 0.]; 2];
fn rat(v: f32) -> Q {
    Q::from_float(f64::from(v)).unwrap()
}
fn zero() -> Q {
    rat(0.)
}
struct Data {
    matrix: Vec<f32>,
    bias: Option<Vec<f32>>,
    lo: Vec<f32>,
    hi: Vec<f32>,
    final_matrix: Vec<f32>,
    final_bias: Vec<f32>,
    objective: Vec<f32>,
    slopes: [Vec<f32>; 4],
    bn: Option<[Vec<f32>; 6]>,
}
impl Data {
    fn source(&self) -> Source<'_> {
        Source {
            matrix: &self.matrix,
            bias: self.bias.as_deref(),
            lower: &self.lo,
            upper: &self.hi,
            final_matrix: &self.final_matrix,
            final_bias: Some(&self.final_bias),
            objective: &self.objective,
            bn: self.bn.as_ref().map(|b| head_source::Bn {
                scale: &b[0],
                shift: &b[1],
                scale_error: &b[2],
                bias_error: &b[3],
                lower: &b[4],
                upper: &b[5],
            }),
        }
    }
    fn slopes(&self) -> Slopes<'_> {
        Slopes {
            positive: &self.slopes[0],
            negative: &self.slopes[1],
            positive_bias: Some(&self.slopes[2]),
            negative_bias: &self.slopes[3],
        }
    }
    fn run(&self, lambda: [f32; 2], support: [f32; 2], guard: &mut Poll<'_>) -> Option<f32> {
        lower(
            &self.source(),
            self.slopes(),
            quote(self.lo.len(), 9, self.objective.len())?,
            [3, 3],
            IDS,
            WEIGHTS,
            lambda,
            support,
            guard,
        )
    }
    // Independent exact affine surrogate including a separate additive BN
    // uncertainty interval for every original row. No implementation intervals.
    fn exact(&self, lambda: [f32; 2], support: [f32; 2]) -> Q {
        let e = self.lo.len();
        let mut a = vec![zero(); e];
        let mut b = zero();
        for k in 0..self.objective.len() {
            b += rat(self.objective[k]) * rat(self.final_bias[k]);
        }
        for g in 0..2 {
            b -= rat(lambda[g]) * rat(support[g]);
        }
        for j in 0..9 {
            let mut c = zero();
            for k in 0..self.objective.len() {
                c += rat(self.objective[k]) * rat(self.final_matrix[k * 9 + j]);
            }
            let mut pre = zero();
            for g in 0..2 {
                for i in 0..3 {
                    if IDS[g][i] == j {
                        let v = rat(lambda[g]) * rat(WEIGHTS[g][i]);
                        pre -= v.clone();
                        c += rat(2.) * v;
                    }
                }
            }
            let (s, t) = if c < zero() {
                (self.slopes[1][j], self.slopes[3][j])
            } else {
                (self.slopes[0][j], self.slopes[2][j])
            };
            let q = &c * rat(s) + pre;
            b += c * rat(t);
            let (scale, shift) = if let Some(bn) = self.bn.as_ref() {
                let radius =
                    rat(bn[2][j]) * rat(bn[4][j]).abs().max(rat(bn[5][j]).abs()) + rat(bn[3][j]);
                b -= q.abs() * radius;
                (rat(bn[0][j]), rat(bn[1][j]))
            } else {
                (rat(1.), zero())
            };
            let bias = self.bias.as_ref().map_or_else(zero, |v| rat(v[j]));
            b += &q * (&scale * bias + shift);
            for k in 0..e {
                a[k] += &q * &scale * rat(self.matrix[j * e + k]);
            }
        }
        for k in 0..e {
            b += (&a[k] * rat(self.lo[k])).min(&a[k] * rat(self.hi[k]));
        }
        b
    }
}
fn fixture() -> Data {
    let mut matrix = vec![0.; 18];
    for j in 0..9 {
        matrix[j * 2] = 0.5;
        matrix[j * 2 + 1] = if j % 2 == 0 { 0.5 } else { -0.5 };
    }
    Data {
        matrix,
        bias: None,
        lo: vec![-1.; 2],
        hi: vec![1.; 2],
        final_matrix: [vec![-2.; 9], vec![0.5; 9]].concat(),
        final_bias: vec![9., 1.],
        objective: vec![0.5, -0.25],
        slopes: [vec![1.; 9], vec![0.5; 9], vec![0.; 9], vec![0.5; 9]],
        bn: None,
    }
}
fn with_guard<T>(f: impl FnOnce(&mut Poll<'_>) -> T) -> T {
    let mut yes = || true;
    let mut guard = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut yes,
        alive: true,
    };
    f(&mut guard)
}
fn near_exact(d: &Data, lambda: [f32; 2], support: [f32; 2]) {
    let expected = d.exact(lambda, support);
    let observed = with_guard(|g| d.run(lambda, support, g)).unwrap();
    assert!(rat(observed) <= expected, "{observed} exceeds {expected}");
    assert!(expected.to_f64().unwrap() - f64::from(observed) < 1e-4);
}

#[test]
fn head_box_floor_interval_sum_preserves_small_positive_terms() {
    let mut total = I::point(1.).unwrap();
    let small = 2f64.powi(-54);
    for _ in 0..128 {
        total = total.add(I::checked(small, small).unwrap()).unwrap();
    }
    let exact =
        Q::from_float(1.).unwrap() + Q::from_float(small).unwrap() * Q::from_integer(128.into());
    assert!(Q::from_float(total.lo).unwrap() <= exact && Q::from_float(total.hi).unwrap() >= exact);
    assert!(total.hi > 1.);
    assert!(I::point(0.)
        .unwrap()
        .mul(I::point(1.).unwrap())
        .unwrap()
        .zero());
}

#[test]
fn head_box_floor_exact_multioutput_contraction_and_sign_uncertainty() {
    let d = fixture();
    for lambda in [[0.25, 0.5], [0.5625, 0.5625], [1., 0.75]] {
        near_exact(&d, lambda, [3., 3.]);
    }
    // Exactly zero post coefficient is enclosed through both sign branches.
    let slopes = d.slopes();
    let (a, b) = relax(slopes, 0, I::checked(-1e-12, 1e-12).unwrap()).unwrap();
    assert!(a.lo <= 0. && a.hi >= 0. && b.lo <= 0. && b.hi >= 0.);
}

#[test]
fn head_box_floor_negative_raw_bn_uncertainty_is_charged_against_original_box() {
    let mut d = fixture();
    d.bn = Some([
        vec![-1. / 3.; 9],
        vec![0.25; 9],
        vec![2f32.powi(-24); 9],
        vec![2f32.powi(-25); 9],
        vec![-1.; 9],
        vec![1.; 9],
    ]);
    near_exact(&d, [0.25, 0.5], [3., 3.]);
    let wide = with_guard(|g| d.run([0.25, 0.5], [3., 3.], g)).unwrap();
    d.bn.as_mut().unwrap()[2].fill(0.);
    d.bn.as_mut().unwrap()[3].fill(0.);
    let exact = with_guard(|g| d.run([0.25, 0.5], [3., 3.], g)).unwrap();
    assert!(wide < exact);
}

#[test]
fn head_box_floor_positive_zero_and_unsafe_original_parent_ranges() {
    for offset in [0.5, 0., -0.5] {
        let mut d = fixture();
        d.matrix = vec![0.; 9];
        d.matrix[6] = 1.;
        d.lo = vec![0.];
        d.hi = vec![2.];
        d.final_matrix = vec![-2.; 9];
        d.final_matrix[6] = 1.;
        d.final_matrix[7..].fill(0.);
        d.final_bias = vec![offset];
        d.objective = vec![1.];
        d.slopes[1].fill(1.);
        d.slopes[3].fill(0.);
        let observed = with_guard(|g| d.run([1.; 2], [0.; 2], g)).unwrap();
        assert!(observed <= offset && observed > offset - 1e-5);
        assert_eq!(observed > 0., offset > 0.);
        near_exact(&d, [1.; 2], [0.; 2]);
    }
}

#[test]
fn head_box_floor_cancellation_bad_payloads_and_quote_overflow_refuse() {
    let d = fixture();
    let mut polls = 0;
    let mut count = || {
        polls += 1;
        true
    };
    let mut counted = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut count,
        alive: true,
    };
    assert!(d.run([0.25; 2], [3.; 2], &mut counted).is_some());
    assert!(polls > 0);
    for cutoff in 0..polls {
        let mut seen = 0;
        let mut callback = || {
            let allowed = seen < cutoff;
            seen += 1;
            allowed
        };
        let mut g = Poll {
            end: Instant::now() + Duration::from_secs(30),
            callback: &mut callback,
            alive: true,
        };
        assert!(d.run([0.25; 2], [3.; 2], &mut g).is_none(), "poll {cutoff}");
    }
    for which in 0..6 {
        let mut bad = fixture();
        match which {
            0 => bad.matrix[0] = f32::NAN,
            1 => bad.lo[0] = 2.,
            2 => bad.final_matrix.pop().map(|_| ()).unwrap(),
            3 => bad.slopes[1].clear(),
            4 => bad.objective[0] = f32::INFINITY,
            _ => bad.hi[1] = f32::NEG_INFINITY,
        }
        assert!(with_guard(|g| bad.run([0.25; 2], [3.; 2], g)).is_none());
    }
    for dimensions in [
        [0, 9, 1],
        [1, 1, 1],
        [1, 9, 0],
        [usize::MAX, 9, 1],
        [1, usize::MAX, 1],
        [1, 9, usize::MAX],
    ] {
        assert!(quote(dimensions[0], dimensions[1], dimensions[2]).is_none());
    }
}
