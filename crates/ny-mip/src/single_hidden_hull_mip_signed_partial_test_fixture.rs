// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original D15/H17 encoder fixture; all authored controls are unexecuted.
pub(super) use super::super::test_fixture::{check_rows, check_rows_except, end, lp, q};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

pub(super) const D: usize = 15;
pub(super) const H: usize = 17;
pub(super) const EPS: f64 = 1. / 128.;

pub(super) fn coordinate(j: usize) -> usize {
    match j {
        0 | 1 | 9 => 0,
        2..=8 => j - 1,
        10..=16 => j - 2,
        _ => panic!("head"),
    }
}
pub(super) struct Fixture {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub dims: Vec<usize>,
    pub bounds: Vec<Vec<Bound>>,
    exact_source: bool,
}
impl Fixture {
    pub fn new(bias: f64) -> Self {
        let mut first = vec![0.; D * H];
        for j in 0..H {
            first[j * D + coordinate(j)] = 1.;
        }
        let mut output = vec![0.; 2 * H];
        for j in 0..H {
            let c = match j {
                0 => 4_f64,
                1 => -1_f64,
                9 => -3_f64,
                _ => -EPS,
            };
            output[j] = c.next_up().next_up();
        }
        Self {
            w: vec![first, output],
            b: vec![vec![0.; H], vec![bias, 0.]],
            dims: vec![D, H, 2],
            bounds: vec![vec![Bound::new(-1., 1.); H]],
            exact_source: true,
        }
    }
    pub fn graph_control(bias: f64) -> Self {
        let mut f = Self::new(bias);
        // An exact original tie needs exact original padding coefficients.
        // Compensated weights would make bias14eps very slightly positive.
        for j in 2..H {
            if j != 9 {
                f.w[1][j] = -EPS;
            }
        }
        f.exact_source = false;
        f
    }
    pub fn source(&self) -> HullMarginProblem {
        let p = HullMarginProblem::new(
            self.w[0].clone(),
            self.b[0].clone(),
            D,
            &self.w[1],
            &self.b[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; D],
            vec![1.; D],
        )
        .unwrap();
        assert_eq!((p.input_dim, p.hidden_dim), (D, H));
        assert_eq!((p.c[0], p.c[1], p.c[9]), (4., -1., -3.));
        for j in 2..H {
            if j != 9 {
                if self.exact_source {
                    assert_eq!(p.c[j], -EPS);
                } else {
                    assert!(p.c[j] <= -EPS && p.c[j] > -EPS - 1e-15);
                }
            }
        }
        p
    }
    pub fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.w,
            &self.b,
            &self.dims,
            &[Bound::new(-1., 1.); D],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    pub fn parts(&self) -> MipParts {
        let mut p = self.original();
        let positive = p.binary_vars[0];
        assert!(p.problem.relax_binary_integrality(&[positive]));
        p.binary_vars.remove(0);
        p.binary_widths.remove(0);
        p
    }
    pub fn encoding(&self, p: &MipParts) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.w,
            biases: &self.b,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(p.problem.num_rows() - 1),
        }
    }
}
pub(super) fn plan(p: &HullMarginProblem, units: u8) -> Plan {
    assert!(units <= 2);
    let mut components = Vec::new();
    for (start, count) in [(1, units), (9, 2 - units)] {
        let positive = if count == 0 { vec![] } else { vec![0] };
        let mut alpha = Vec::new();
        if count != 0 {
            let active = (-p.c[start] / (p.c[0] * f64::from(count) * 0.5)).clamp(0., 1.);
            for mask in 0..256 {
                alpha.push(if mask & 1 == 0 { 0. } else { active });
            }
        }
        components.push(Component {
            negative: (start..start + 8).collect(),
            positive,
            positive_units: if count == 0 { vec![] } else { vec![count] },
            g: vec![0.; D],
            alpha,
        });
    }
    Plan {
        components,
        leftover: vec![],
    }
}
pub(super) fn cuts(p: &HullMarginProblem, units: u8) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    for j in 2..H {
        if j != 9 {
            for i in 0..D {
                m.g[j * D + i] = 0.5 * p.w_row(j)[i];
            }
        }
    }
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let mut c = super::super::super::super::prepare(p, m, lower, end(), &mut || true).unwrap();
    let plan = plan(p, units);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked =
        super::super::super::certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(&c).collect());
    let row = row(&c, &plan, checked, cost, &b, &seal);
    component_rows::admit(&c, &plan, &b).unwrap();
    c.signed = Some(row);
    c
}
pub(super) fn append(f: &Fixture, units: u8) -> (MipParts, HullMipCutReport, usize) {
    let p = f.source();
    let c = cuts(&p, units);
    let mut parts = f.parts();
    let before = parts.problem.num_rows();
    let encoding = f.encoding(&parts);
    let report = c.append_to(&mut parts, &encoding, end()).unwrap();
    let signed = report.signed.as_ref().unwrap();
    let joint = &signed.component_rows.joint;
    assert_eq!(signed.source_replays, 1);
    assert_eq!(joint.producer_attempts, 1);
    assert_eq!(joint.partial_paid_pairs, 1);
    assert_eq!(joint.partial_completed_pairs, 1);
    assert_eq!(joint.partial_selected_heads, 3);
    assert_eq!(joint.partial_tail_heads, 13);
    assert_eq!(joint.partial_checked_masks, 8);
    assert_eq!(joint.checker_slots, 1);
    assert_eq!(joint.appended_rows, 1);
    assert!(joint.appended_rows <= 2);
    assert_eq!(report.proposal_lp_calls, 0);
    assert!(signed.work_units <= MAX_WORK && signed.admitted_bytes <= MAX_BYTES);
    (parts, report, before)
}
pub(super) fn point(
    f: &Fixture,
    parts: &MipParts,
    inputs: &[Q],
    heads: &[Q],
    zero_mask: usize,
) -> Vec<Q> {
    assert_eq!((inputs.len(), heads.len()), (D, H));
    let original = f.original();
    let enc = f.encoding(&original);
    let columns = binding::head_columns(&enc, &original).unwrap();
    let mut v = vec![Q::zero(); parts.num_cols];
    for i in 0..D {
        assert_eq!(original.input_vars[i].0, i);
        v[parts.input_vars[i].0] = inputs[i].clone();
    }
    for j in 0..H {
        let z = &inputs[coordinate(j)];
        v[D + j] = z.clone();
        v[columns[j].0] = heads[j].clone();
        let phase = if z.is_zero() {
            (zero_mask >> j) & 1
        } else if z > &Q::zero() {
            1
        } else {
            0
        };
        v[original.binary_vars[j].0] = q(phase as f64);
    }
    v[parts.output_vars[0].0] = q(f.b[1][0]) + (0..H).map(|j| q(f.w[1][j]) * &heads[j]).sum::<Q>();
    v
}
pub(super) fn fractional(f: &Fixture, parts: &MipParts, dyadic: bool) -> Vec<Q> {
    let mut inputs = vec![q(-1.); D];
    inputs[0] = Q::zero();
    let mut heads = vec![Q::zero(); H];
    if dyadic {
        heads[0] = q(3. / 16.);
        heads[1] = q(3. / 8.);
        heads[9] = q(1. / 4.);
    } else {
        heads[0] = q(1. / 4.);
        heads[1] = q(1. / 2.);
        heads[9] = Q::new(1.into(), 3.into());
    }
    let mut v = point(f, parts, &inputs, &heads, 0);
    let original = f.original();
    for j in [0, 1, 9] {
        v[original.binary_vars[j].0] = q(0.5);
    }
    v
}
pub(super) fn graph(f: &Fixture, parts: &MipParts, inputs: &[Q], zero_mask: usize) -> Vec<Q> {
    let heads: Vec<Q> = (0..H)
        .map(|j| {
            let z = &inputs[coordinate(j)];
            if z > &Q::zero() {
                z.clone()
            } else {
                Q::zero()
            }
        })
        .collect();
    point(f, parts, inputs, &heads, zero_mask)
}
