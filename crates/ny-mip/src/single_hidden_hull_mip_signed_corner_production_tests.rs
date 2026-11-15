// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Genuine Source, encoder and two-replay cache integration controls.
use super::super::tests::{end, rat};
use super::*;
use crate::MipParts;
use num_rational::BigRational as Q;
use num_traits::Zero;
use ny_core::Bound;

struct CrossFixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dimensions: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl CrossFixture {
    const D: usize = 16;
    fn new() -> Self {
        let mut first = vec![0.; 6 * Self::D];
        for (j, [x, y]) in [
            [1., 1.],
            [-1., -1.],
            [1., -1.],
            [-1., 1.],
            [1., 1.],
            [-1., -1.],
        ]
        .into_iter()
        .enumerate()
        {
            first[j * Self::D] = x;
            first[j * Self::D + 1] = y;
        }
        let bias = vec![
            1. / 16.,
            -1. / 16.,
            -1. / 16.,
            1. / 16.,
            -1. / 16.,
            1. / 16.,
        ];
        let bounds = vec![bias
            .iter()
            .map(|&b| Bound::new((b - 2.) as f32, (b + 2.) as f32))
            .collect()];
        Self {
            weights: vec![
                first,
                vec![-1., -1., -1., -1., -0.125, -0.125, 0., 0., 0., 0., 0., 0.],
            ],
            biases: vec![bias, vec![2.5, 0.]],
            dimensions: vec![Self::D, 6, 2],
            bounds,
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            Self::D,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; Self::D],
            vec![1.; Self::D],
        )
        .unwrap()
    }
    fn plan(&self) -> Plan {
        Plan {
            components: (0..3)
                .map(|c| Component {
                    negative: vec![2 * c, 2 * c + 1],
                    positive: vec![],
                    positive_units: vec![],
                    g: vec![0.; Self::D],
                    alpha: vec![],
                })
                .collect(),
            leftover: vec![],
        }
    }
    fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dimensions,
            &[Bound::new(-1., 1.); Self::D],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &self.dimensions,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
}
fn satisfies(parts: &MipParts, point: &[Q], rows: usize) -> bool {
    point.len() == parts.num_cols
        && parts.problem.cols().iter().zip(point).all(|(c, x)| {
            (!c.lb.is_finite() || *x >= rat(c.lb)) && (!c.ub.is_finite() || *x <= rat(c.ub))
        })
        && parts.problem.rows()[..rows].iter().all(|r| {
            let value: Q = r.coeffs.iter().map(|&(j, a)| rat(a) * &point[j]).sum();
            (!r.lb.is_finite() || value >= rat(r.lb)) && (!r.ub.is_finite() || value <= rat(r.ub))
        })
}
fn genuine_owner<'p>(
    p: &'p HullMarginProblem,
    f: &CrossFixture,
    enable: bool,
) -> HullRootMipCuts<'p> {
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    let mut cuts = propose_root_mip_cuts_inner(p, until, Some(&mut attempt)).unwrap();
    assert_eq!(cuts.proposal_lp_calls, 1);
    let mut b = Budget::new_with_portfolio_conditions(&cuts, until, Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let plan = f.plan();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    component_rows::admit(&cuts, &plan, &b).unwrap();
    native_publication::fund(&cuts, &b).unwrap();
    child::prepay(&cuts, &plan, &b).unwrap();
    if enable {
        prepay(&cuts, &plan, &b).unwrap();
    }
    assert_eq!(native_publication::funding(&b).funded_calls, 1);
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, &plan, checked, cost, &b, &seal));
    cuts
}

#[test]
fn corner_authentic_constructor_encoder_and_two_replay_cache_close_cross_fixture() {
    let f = CrossFixture::new();
    let p = f.source(); // Deliberately never replace rounded p.c or p.d.
    let initial = f.original();
    let encoding = f.encoding();
    let heads = binding::head_columns(&encoding, &initial).unwrap();
    assert_eq!(initial.binary_vars.len(), 6);
    let mut relaxed = vec![Q::zero(); initial.num_cols];
    for j in 0..6 {
        // The original affine layer allocates D+j; the complete original IR
        // check below verifies this mapping instead of accepting it on trust.
        relaxed[CrossFixture::D + j] = rat(f.biases[0][j]);
        relaxed[heads[j].0] = rat(1.) + rat(f.biases[0][j]) / rat(2.);
        relaxed[initial.binary_vars[j].0] = rat(0.5);
    }
    relaxed[initial.output_vars[0].0] = rat(-1.75);
    assert!(satisfies(&initial, &relaxed, initial.problem.num_rows()));
    // This is explicitly an LP-relaxation witness: half selectors are allowed.

    // Independent exact original graph oracle, including every closed-phase
    // selector at actual zero preactivations. Safe graph points intentionally
    // violate ONLY the final unsafe row, so check the whole prefix before it.
    let mut exact_min = None::<Q>;
    for x in [-1., -1. / 16., 0., 1. / 16., 1.] {
        for y in [-1., -1. / 16., 0., 1. / 16., 1.] {
            let pre: Vec<Q> = (0..6)
                .map(|j| {
                    rat(f.biases[0][j])
                        + rat(f.weights[0][j * CrossFixture::D]) * rat(x)
                        + rat(f.weights[0][j * CrossFixture::D + 1]) * rat(y)
                })
                .collect();
            let zeros: Vec<_> = (0..6).filter(|&j| pre[j].is_zero()).collect();
            let output = rat(2.5)
                + (0..6)
                    .map(|j| rat(f.weights[1][j]) * pre[j].clone().max(Q::zero()))
                    .sum::<Q>();
            assert!(output >= rat(17. / 128.));
            exact_min = Some(exact_min.map_or(output.clone(), |v| v.min(output.clone())));
            for ties in 0..1usize << zeros.len() {
                let mut point = vec![Q::zero(); initial.num_cols];
                point[initial.input_vars[0].0] = rat(x);
                point[initial.input_vars[1].0] = rat(y);
                for j in 0..6 {
                    point[CrossFixture::D + j] = pre[j].clone();
                    point[heads[j].0] = pre[j].clone().max(Q::zero());
                    let active = if let Some(bit) = zeros.iter().position(|&head| head == j) {
                        (ties >> bit) & 1
                    } else {
                        usize::from(pre[j] > Q::zero())
                    };
                    point[initial.binary_vars[j].0] = rat(active as f64);
                }
                point[initial.output_vars[0].0] = output.clone();
                assert!(satisfies(&initial, &point, initial.problem.num_rows() - 1));
            }
        }
    }
    assert_eq!(exact_min.unwrap(), rat(17. / 128.));

    let mut old_bits = None;
    for enabled in [false, true] {
        let cuts = genuine_owner(&p, &f, enabled);
        let row = cuts.signed.as_ref().unwrap();
        assert!(row.lower < -1.88 && row.lower > -1.89);
        let tuple = (
            row.lower.to_bits(),
            row.constant.to_bits(),
            row.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
        if let Some(old) = &old_bits {
            assert_eq!(&tuple, old);
        } else {
            old_bits = Some(tuple.clone());
        }
        let mut parts = f.original();
        let original = parts.problem.clone();
        cuts.append_to(&mut parts, &encoding, end()).unwrap();
        assert_eq!(
            &parts.problem.rows()[..original.num_rows()],
            original.rows()
        );
        assert_eq!(parts.problem.cols(), original.cols());
        assert_eq!(row.stats().source_replays, 1);
        if enabled {
            let state = row.budget.state.lock().unwrap();
            let axes = state
                .corner
                .grant
                .as_ref()
                .unwrap()
                .axes
                .as_ref()
                .unwrap()
                .selected();
            assert!(axes.contains(&0) && axes.contains(&1));
        }
        // The original method installs the child cache even if its signed
        // proof route stays nonpositive. This tests the REAL installed channel.
        let _proof = cuts.best_signed_root_proof(end());
        assert_eq!(row.stats().source_replays, 2);
        let cache = cuts
            .signed_child_cache
            .get()
            .expect("second original replay must install cache");
        let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
        let root = bound.root_lower().unwrap();
        let child = bound.lower_zero(&[], end(), &mut || true).unwrap();
        if enabled {
            assert!(root > 0.13 && child > 0.13);
            assert!(rat(root) <= rat(17. / 128.) && rat(child) <= rat(17. / 128.));
        } else {
            assert_eq!(root.to_bits(), row.lower.to_bits());
            assert!(child < -1.88);
        }
        assert_eq!(
            tuple,
            (
                row.lower.to_bits(),
                row.constant.to_bits(),
                row.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            )
        );
        assert_eq!(cuts.proposal_lp_calls, 1);
        assert_eq!(native_publication::funding(&row.budget).funded_calls, 1);
        assert!(row.replay_with_cache(&cuts, end(), &mut || true).is_none());
    }
}

#[path = "single_hidden_hull_mip_signed_corner_continuation_tests.rs"]
mod continuation_tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_corner_wrapper_tests.rs"]
mod wrapper_tests;
