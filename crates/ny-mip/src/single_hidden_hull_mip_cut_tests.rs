// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};
use num_rational::BigRational as Q;
use num_traits::Zero;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(v: f64) -> Q {
    Q::from_float(v).unwrap()
}

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}

impl Fixture {
    // .25+x+1-ReLU(x+y)>=.25. Scalar triangle permits -.75 at(-1,1),
    // whereas the coordinate plane ReLU(x+y)<=x+1 closes the real margin.
    fn new(extra: bool) -> Self {
        let (first, bias, coefficients, bounds) = if extra {
            (
                vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![-1.0, 1.0, 1.0, 0.0],
                vec![
                    Bound::new(-2.0, 2.0),
                    Bound::new(1.0, 3.0),
                    Bound::new(-1.0, 1.0),
                    Bound::new(-2.0, 2.0),
                ],
            )
        } else {
            (
                vec![1.0, 1.0, 1.0, 0.0],
                vec![0.0, 2.0],
                vec![-1.0, 1.0],
                vec![Bound::new(-2.0, 2.0), Bound::new(1.0, 3.0)],
            )
        };
        let h = bias.len();
        let mut output = coefficients;
        output.extend(vec![0.0; h]);
        Self {
            weights: vec![first, output],
            biases: vec![bias, vec![-0.75, 0.0]],
            dims: vec![2, h, 2],
            bounds: vec![bounds],
        }
    }

    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            2,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.0), (1, -1.0)],
            0.0,
            vec![-1.0; 2],
            vec![1.0; 2],
        )
        .unwrap()
    }

    fn parts(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &[Bound::new(-1.0, 1.0); 2],
            &self.bounds,
        )
        .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
    }

    fn binding(&self, parts: &MipParts) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(parts.problem.num_rows() - 1),
        }
    }
}

fn manual(problem: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers {
        alpha: vec![0.0; problem.hidden_dim],
        g: vec![0.0; problem.hidden_dim * problem.input_dim],
    };
    m.alpha[1] = 1.0;
    m.g[0] = 1.0;
    let lower = certify_hull_lower_bound(problem, &m).unwrap();
    prepare(problem, m, lower, end(), &mut || true).unwrap()
}

fn assert_refusal(cuts: &HullRootMipCuts<'_>, parts: &mut MipParts, f: &Fixture) {
    let before = parts.problem.clone();
    let vars = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let encoding = f.binding(parts);
    assert!(cuts.append_to(parts, &encoding, end()).is_none());
    assert_eq!(parts.problem, before);
    assert_eq!(parts.binary_vars, vars);
    assert_eq!(parts.binary_widths, widths);
}

#[test]
fn root_mip_majorant_matches_exact_rational_all_corner_support_and_interior_samples() {
    let w = [0.5, -1.25, 0.75];
    let b = -0.375;
    let lo = [-1.0, 0.25, -0.5];
    let hi = [1.0, 1.5, 2.0];
    for g in [[1.0, -0.25, 0.125], [-0.5, 2.0, -1.0], [0.0; 3]] {
        let upper = majorant_offset_upper(&w, b, &g, &lo, &hi).unwrap();
        let mut exact = None::<Q>;
        for mask in 0..8 {
            let x: Vec<f64> = (0..3)
                .map(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] })
                .collect();
            let t = q(b) + (0..3).map(|i| q(w[i]) * q(x[i])).sum::<Q>();
            let value = t.max(Q::zero()) - (0..3).map(|i| q(g[i]) * q(x[i])).sum::<Q>();
            exact = Some(exact.map_or(value.clone(), |old| old.max(value)));
        }
        assert!(q(upper) >= exact.unwrap());
        // A max of two affine forms is convex: its box maximum occurs at a
        // vertex. The separate interior grid additionally checks row signs.
        for a in 0..5 {
            for c in 0..5 {
                for d in 0..5 {
                    let k = [a, c, d];
                    let x: Vec<Q> = (0..3)
                        .map(|i| q(lo[i]) + (q(hi[i]) - q(lo[i])) * Q::new(k[i].into(), 4.into()))
                        .collect();
                    let t = q(b) + (0..3).map(|i| q(w[i]) * &x[i]).sum::<Q>();
                    let residual = t.max(Q::zero()) - (0..3).map(|i| q(g[i]) * &x[i]).sum::<Q>();
                    assert!(residual <= q(upper));
                }
            }
        }
    }
}

#[test]
fn root_mip_coordinate_plane_is_stronger_than_scalar_triangle_on_authored_corner() {
    let x = [-1.0, 1.0];
    let coordinate =
        majorant_offset_upper(&[1.0, 1.0], 0.0, &[1.0, 0.0], &[-1.0; 2], &[1.0; 2]).unwrap();
    let plane = q(x[0]) + q(coordinate);
    let triangle = Q::new(1.into(), 2.into()) * (q(x[0]) + q(x[1])) + q(1.0);
    assert!(plane < q(0.0000001));
    assert_eq!(triangle, q(1.0));
    assert!(q(0.25) + q(x[0]) + q(1.0) - plane > q(0.249));
}

#[test]
fn root_mip_last_input_nonzero_bias_and_empty_branch_are_in_support() {
    let w = [0.0, 0.0, 2.0];
    let g = [0.0, 0.0, 0.0];
    let upper = majorant_offset_upper(&w, -0.5, &g, &[-1.0; 3], &[1.0; 3]).unwrap();
    assert!(q(upper) >= q(1.5));
    let inactive = majorant_offset_upper(&[0.0], -2.0, &[1.0], &[-1.0], &[1.0]).unwrap();
    assert!(q(inactive) >= q(1.0), "zero-ReLU branch must survive max");
}

#[test]
fn root_mip_completed_bound_must_match_actual_exported_vector() {
    let f = Fixture::new(false);
    let p = f.source();
    let cuts = manual(&p);
    assert!(cuts.checked_lower() > 0.249 && cuts.checked_lower() <= 0.25);
    assert!(prepare(
        &p,
        cuts.multipliers.clone(),
        cuts.checked_lower() + 1.0,
        end(),
        &mut || true
    )
    .is_none());
}

#[test]
fn root_mip_injection_succeeds_after_mixed_convex_binary_metadata_filtering() {
    let f = Fixture::new(true);
    let p = f.source();
    let cuts = manual(&p);
    let mut parts = f.parts();
    let vars = parts.binary_vars.clone();
    assert_eq!(vars.len(), 3);
    assert!(parts.problem.relax_binary_integrality(&vars[1..]));
    parts.binary_vars = vec![vars[0]];
    parts.binary_widths = vec![parts.binary_widths[0]];
    parts
        .problem
        .mark_margin_row(Row(parts.problem.num_rows() - 1))
        .unwrap();
    let original = parts.problem.clone();
    let rows = original.num_rows();
    let encoding = f.binding(&parts);
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert!(report.cuts >= 1);
    assert!(report.checked_hull_lower > 0.249);
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(&parts.problem.rows()[..rows], original.rows());
    assert_eq!(parts.problem.margin_row(), original.margin_row());
    assert_eq!(parts.binary_vars, vec![vars[0]]);
    // The appended exact row contains every original graph grid point.
    for x in [-1.0f64, 0.0, 1.0] {
        for y in [-1.0f64, 0.0, 1.0] {
            for row in &parts.problem.rows()[rows..] {
                let mut values = vec![Q::zero(); parts.num_cols];
                values[0] = q(x);
                values[1] = q(y);
                for j in 0..p.hidden_dim {
                    let t = q(p.b1[j]) + q(p.w_row(j)[0]) * q(x) + q(p.w_row(j)[1]) * q(y);
                    let columns = binding::head_columns(&encoding, &f.parts()).unwrap();
                    values[columns[j].0] = t.max(Q::zero());
                }
                let lhs = row
                    .coeffs
                    .iter()
                    .map(|(i, c)| q(*c) * &values[*i])
                    .sum::<Q>();
                assert!(lhs <= q(row.ub));
            }
        }
    }
}

#[test]
fn root_mip_wrong_first_weight_or_bias_refuses_without_appending() {
    let original = Fixture::new(false);
    let p = original.source();
    let cuts = manual(&p);
    for bias in [false, true] {
        let mut foreign = Fixture::new(false);
        if bias {
            foreign.biases[0][0] = 0.125;
        } else {
            foreign.weights[0][1] = 0.5;
        }
        let mut parts = foreign.parts();
        assert_refusal(&cuts, &mut parts, &foreign);
    }
}

#[test]
fn root_mip_wrong_box_output_objective_or_head_layout_refuses() {
    let f = Fixture::new(false);
    let p = f.source();
    let cuts = manual(&p);
    let mut parts = f.parts();
    parts.problem.fix_col(parts.input_vars[1], 0.0);
    assert_refusal(&cuts, &mut parts, &f);
    let mut foreign = Fixture::new(false);
    foreign.weights[1][0] = -0.5;
    let mut parts = foreign.parts();
    assert_refusal(&cuts, &mut parts, &foreign);
    let mut parts = f.parts();
    parts.output_vars.swap(0, 1);
    assert_refusal(&cuts, &mut parts, &f);
    let mut parts = f.parts();
    parts.binary_widths[0] = 3.0;
    assert_refusal(&cuts, &mut parts, &f);
}

#[test]
fn root_mip_plane_head_coefficient_and_offset_mutation_refuse() {
    let f = Fixture::new(false);
    let p = f.source();
    for mutation in 0..3 {
        let mut cuts = manual(&p);
        match mutation {
            0 => cuts.planes[0].head = 1,
            1 => cuts.planes[0].g[0] = 0.5,
            _ => cuts.planes[0].upper = f64::from_bits(cuts.planes[0].upper.to_bits() + 1),
        }
        assert_refusal(&cuts, &mut f.parts(), &f);
    }
}

#[test]
fn root_mip_expiry_and_partial_preparation_keep_original_ir_unchanged() {
    let f = Fixture::new(true);
    let p = f.source();
    let cuts = manual(&p);
    let mut parts = f.parts();
    let before = parts.problem.clone();
    let encoding = f.binding(&parts);
    assert!(cuts
        .append_to(
            &mut parts,
            &encoding,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("overflow when subtracting duration from instant")
        )
        .is_none());
    assert_eq!(parts.problem, before);
    for allowance in [1, 8, 24, 48] {
        let mut calls = 0;
        let result = cuts.append_with_poll(&mut parts, &encoding, end(), &mut || {
            calls += 1;
            calls <= allowance
        });
        assert!(result.is_none());
        assert_eq!(parts.problem, before);
    }
    let mut stale = manual(&p);
    stale.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(stale.append_to(&mut parts, &encoding, end()).is_none());
    assert_eq!(
        parts.problem, before,
        "later appender deadline renewed stored cut-stage clock"
    );
}

#[test]
fn root_mip_second_append_and_foreign_margin_marker_are_refused() {
    let f = Fixture::new(false);
    let p = f.source();
    let cuts = manual(&p);
    let mut parts = f.parts();
    let encoding = f.binding(&parts);
    cuts.append_to(&mut parts, &encoding, end()).unwrap();
    let once = parts.problem.clone();
    assert!(cuts.append_to(&mut parts, &encoding, end()).is_none());
    assert_eq!(parts.problem, once);
    let mut parts = f.parts();
    let foreign = Row(f.dims[1]); // Existing first one-sided ReLU row: no census mismatch.
    parts.problem.mark_margin_row(foreign).unwrap();
    assert_refusal(&cuts, &mut parts, &f);
}

#[test]
fn root_mip_capacity_and_expired_proposal_decline_before_lp() {
    let p = HullMarginProblem::new(
        vec![1.0; 129],
        vec![0.0],
        129,
        &[-1.0, 0.0],
        &[1.0, 0.0],
        2,
        &[(0, 1.0), (1, -1.0)],
        0.0,
        vec![-1.0; 129],
        vec![1.0; 129],
    )
    .unwrap();
    assert!(!capacity(&p));
    assert!(propose_root_mip_cuts(&p, end()).is_none());
    let f = Fixture::new(false);
    let p = f.source();
    assert!(propose_root_mip_cuts(
        &p,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant")
    )
    .is_none());
}

fn actual_root_lower(parts: &MipParts, unsafe_row: Row) -> f64 {
    // This is a diagnostic LP ONLY. Exclude the unsafe decision row so the
    // original signed output can be optimized, while retaining every cut.
    let mut ir = crate::ir::MilpProblem::new();
    for col in parts.problem.cols() {
        ir.add_col(col.obj, col.lb, col.ub);
    }
    for (index, row) in parts.problem.rows().iter().enumerate() {
        if index != unsafe_row.0 {
            ir.add_row(
                row.lb,
                row.ub,
                row.coeffs.iter().map(|(i, c)| (crate::ir::Col(*i), *c)),
            );
        }
    }
    let model = crate::ay_lib::to_ay_model(&ir).unwrap();
    let opts = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut session = ay_milp::LpSession::new(&model, &opts).unwrap();
    let result = session
        .rigorous_bound_expr(
            &[(model.col_at(parts.output_vars[0].0).unwrap(), 1.0)],
            ay_milp::Sense::Minimize,
        )
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&result).unwrap()
}

#[test]
fn root_mip_actual_lp_export_strengthens_triangle_and_actual_ay_checks_augmented_unsat() {
    let f = Fixture::new(false);
    let p = f.source();
    let mut parts = f.parts();
    let encoding = f.binding(&parts);
    let before = actual_root_lower(&parts, encoding.unsafe_row);
    assert!(before <= -0.74);
    let cuts = propose_root_mip_cuts(&p, end()).expect("actual completed hull LP vector required");
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert!(report.checked_hull_lower > 0.249 && report.checked_hull_lower <= 0.25);
    assert_eq!(report.proposal_lp_calls, 1);
    assert!(report.cuts >= 1);
    let after = actual_root_lower(&parts, encoding.unsafe_row);
    assert!(after > 0.249 && after <= 0.25 && after > before + 0.9);
    let solver = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.0,
            ay_hard_deadline: Some(end()),
            ..Default::default()
        },
    );
    let started = Instant::now();
    let result = solver.check_feasibility().unwrap();
    eprintln!(
        "original augmented AY result={result:?} elapsed={:?}",
        started.elapsed()
    );
    assert!(
        matches!(result, MipResult::Unsat { certified: true }),
        "original augmented AY graph requires certified UNSAT, got {result:?}"
    );
}
