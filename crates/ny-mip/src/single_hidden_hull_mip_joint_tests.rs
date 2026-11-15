// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};
use num_rational::BigRational as Q;
use num_traits::Zero;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(value: f64) -> Q {
    Q::from_float(value).unwrap()
}

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}

impl Fixture {
    // r1=ReLU(x+y), r2=ReLU(x-y), p=x+2. The exact joint relation
    // r1+r2-x<=1 gives minimum +.25 / -.25 for the two authored biases.
    fn new(bias: f64, extra: bool) -> Self {
        let mut w = vec![1.0, 1.0, 1.0, -1.0, 1.0, 0.0];
        let mut b = vec![0.0, 0.0, 2.0];
        let mut c = vec![-1.0, -1.0, 1.0];
        let mut bounds = vec![Bound::new(-2.0, 2.0); 2];
        bounds.push(Bound::new(1.0, 3.0));
        if extra {
            w.extend([0.0, 1.0]);
            b.push(0.0);
            c.push(0.25);
            bounds.push(Bound::new(-1.0, 1.0));
        }
        let h = b.len();
        c.extend(vec![0.0; h]);
        Self {
            weights: vec![w, c],
            biases: vec![b, vec![bias, 0.0]],
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
        let mut e = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &[Bound::new(-1.0, 1.0); 2],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
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

fn manual(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    m.alpha[2] = 1.0;
    m.g[..4].copy_from_slice(&[0.5, 0.5, 0.5, -0.5]);
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    prepare(p, m, lower, end(), &mut || true).unwrap()
}

fn separated(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut cuts = manual(p);
    discover_impl(&mut cuts, end(), &mut || true, false);
    assert_eq!(cuts.joint.len(), 1);
    cuts
}

fn actual_lower(parts: &MipParts, unsafe_row: Row) -> f64 {
    let mut ir = crate::ir::MilpProblem::new();
    for c in parts.problem.cols() {
        ir.add_col(c.obj, c.lb, c.ub);
    }
    for (i, r) in parts.problem.rows().iter().enumerate() {
        if i != unsafe_row.0 {
            ir.add_row(r.lb, r.ub, r.coeffs.iter().map(|(i, c)| (IrCol(*i), *c)));
        }
    }
    let model = crate::ay_lib::to_ay_model(&ir).unwrap();
    let opts = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut session = ay_milp::LpSession::new(&model, &opts).unwrap();
    let objective = [
        (model.col_at(parts.output_vars[0].0).unwrap(), 1.0),
        (model.col_at(parts.output_vars[1].0).unwrap(), -1.0),
    ];
    let result = session
        .rigorous_bound_expr(&objective, ay_milp::Sense::Minimize)
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&result).unwrap()
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
fn joint_mip_actual_pair_beats_full_individual_ideal_root_and_ay_certifies_unsat() {
    let f = Fixture::new(-0.75, false);
    let p = f.source();
    let old = hull_root_bound(&p, end());
    assert!(
        old.certified.is_some_and(|v| v > -0.751 && v <= -0.749),
        "{old:?}"
    );
    // At x=y=0, each individual graph hull separately admits r1=r2=1;
    // their shared first moments do not force a shared graph distribution.
    let mut parts = f.parts();
    let encoding = f.binding(&parts);
    assert!(actual_lower(&parts, encoding.unsafe_row) <= -0.749);
    let cuts = propose_root_mip_cuts(&p, end()).expect("actual attaining LP vector required");
    assert!(!cuts.joint.is_empty());
    let original = parts.problem.clone();
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(report.individual_cuts, cuts.planes.len());
    assert_eq!(report.joint_cuts, 1);
    assert_eq!(report.cuts, report.individual_cuts + report.joint_cuts);
    assert_eq!(
        &parts.problem.rows()[..original.num_rows()],
        original.rows()
    );
    assert_eq!(parts.problem.cols(), original.cols());
    assert!(
        report.checked_hull_lower < 0.0,
        "pair count must not replace old LP certificate"
    );
    assert_eq!(report.proposal_lp_calls, 1);
    let after = actual_lower(&parts, encoding.unsafe_row);
    assert!(after > 0.249 && after <= 0.25, "{after}");
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
    assert!(matches!(
        solver.check_feasibility().unwrap(),
        MipResult::Unsat { certified: true }
    ));
}

#[test]
fn joint_mip_unsafe_changed_bias_stays_sat_with_original_rational_margin() {
    let f = Fixture::new(-1.25, false);
    let p = f.source();
    let cuts = propose_root_mip_cuts(&p, end()).unwrap();
    let mut parts = f.parts();
    let encoding = f.binding(&parts);
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(report.joint_cuts, 1);
    assert!(actual_lower(&parts, encoding.unsafe_row) <= -0.25);
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
    let MipResult::Sat { input_values, .. } = solver.check_feasibility().unwrap() else {
        panic!("unsafe graph was excluded")
    };
    let (x, y) = (q(input_values[0]), q(input_values[1]));
    assert!(x >= q(-1.0) && x <= q(1.0) && y >= q(-1.0) && y <= q(1.0));
    let margin = q(0.75) + &x - (&x + &y).max(Q::zero()) - (&x - &y).max(Q::zero());
    assert!(margin <= Q::zero());
}

#[test]
fn joint_mip_all_four_masks_and_rational_graph_grid_include_bias_and_last_input() {
    let p = HullMarginProblem::new(
        vec![0.5, -1.25, 0.75, -0.25, 0.5, 1.5],
        vec![-0.375, 0.625],
        3,
        &[-1.0, -2.0],
        &[0.0],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0, 0.25, -0.5],
        vec![1.0, 1.5, 2.0],
    )
    .unwrap();
    let a = [0.5, 1.25];
    let g = [0.125, -0.25, 0.375];
    let upper = support(&p, &[0, 1], &a, &g, end(), &mut || true).unwrap();
    let mut best = None::<Q>;
    for mask in 0..4 {
        for corner in 0..8 {
            let x: Vec<_> = (0..3)
                .map(|i| {
                    q(if corner & (1 << i) == 0 {
                        p.lower[i]
                    } else {
                        p.upper[i]
                    })
                })
                .collect();
            let mut affine = Q::zero();
            for j in 0..2 {
                if mask & (1 << j) != 0 {
                    affine += q(a[j])
                        * (q(p.b1[j]) + (0..3).map(|i| q(p.w_row(j)[i]) * &x[i]).sum::<Q>());
                }
            }
            affine -= (0..3).map(|i| q(g[i]) * &x[i]).sum::<Q>();
            best = Some(best.map_or(affine.clone(), |old| old.max(affine)));
        }
    }
    assert!(q(upper) >= best.unwrap());
    for k0 in 0..5 {
        for k1 in 0..5 {
            for k2 in 0..5 {
                let k = [k0, k1, k2];
                let x: Vec<_> = (0..3)
                    .map(|i| {
                        q(p.lower[i])
                            + (q(p.upper[i]) - q(p.lower[i])) * Q::new(k[i].into(), 4.into())
                    })
                    .collect();
                let value = (0..2)
                    .map(|j| {
                        q(a[j])
                            * (q(p.b1[j]) + (0..3).map(|i| q(p.w_row(j)[i]) * &x[i]).sum::<Q>())
                                .max(Q::zero())
                    })
                    .sum::<Q>()
                    - (0..3).map(|i| q(g[i]) * &x[i]).sum::<Q>();
                assert!(value <= q(upper));
            }
        }
    }
}

#[test]
fn joint_mip_empty_support_branch_and_invalid_roles_cannot_be_skipped() {
    let p = HullMarginProblem::new(
        vec![0.0; 6],
        vec![-100.0; 2],
        3,
        &[-1.0, -1.0],
        &[0.0],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0; 3],
        vec![1.0; 3],
    )
    .unwrap();
    let upper = support(
        &p,
        &[0, 1],
        &[1.0, 1.0],
        &[0.0, 0.0, 2.0],
        end(),
        &mut || true,
    )
    .unwrap();
    assert!((2.0..2.000001).contains(&upper));
    for a in [
        [0.0, 1.0],
        [-1.0, 1.0],
        [f64::NAN, 1.0],
        [f64::INFINITY, 1.0],
    ] {
        assert!(support(&p, &[0, 1], &a, &[0.0; 3], end(), &mut || true).is_none());
    }
    for heads in [[0, 0], [1, 0], [0, 2]] {
        assert!(support(&p, &heads, &[1.0; 2], &[0.0; 3], end(), &mut || true).is_none());
    }
    assert!(support(&p, &[0, 1], &[1.0; 2], &[0.0; 2], end(), &mut || true).is_none());
}

#[test]
fn joint_mip_uniform_gain_prices_the_last_aggregate_coefficient_residual() {
    let f = Fixture::new(-0.75, false);
    let p = f.source();
    let cuts = manual(&p);
    let (a, mut g) = proposal(&cuts, &[0, 1]).unwrap();
    g[1] += 0.5 * a[0];
    let upper = support(&p, &[0, 1], &a, &g, end(), &mut || true).unwrap();
    let gain = improvement(&cuts, &[0, 1], &a, &g, upper, end(), &mut || true).unwrap();
    // H=1.5a, but the old pair's input residual has minimum -0.5a.
    // Ignoring that last input would invent a positive 0.5a improvement.
    assert!(gain <= 0.0, "{gain}");
}

#[test]
fn joint_mip_selected_tuple_mutation_refuses_the_whole_transaction() {
    let f = Fixture::new(-0.75, false);
    let p = f.source();
    for change in 0..5 {
        let mut cuts = separated(&p);
        let row = &mut cuts.joint[0];
        match change {
            0 => row.heads.swap(0, 1),
            1 => row.weights[0] = f64::from_bits(row.weights[0].to_bits() + 1),
            2 => row.g[1] = 0.125,
            3 => row.upper = f64::from_bits(row.upper.to_bits() + 1),
            _ => row.improvement = f64::from_bits(row.improvement.to_bits() + 1),
        }
        assert_refusal(&cuts, &mut f.parts(), &f);
    }
}

#[test]
fn joint_mip_foreign_weights_box_layout_and_output_bias_preserve_original_ir() {
    let f = Fixture::new(-0.75, false);
    let p = f.source();
    let cuts = separated(&p);
    let mut foreign = Fixture::new(-0.75, false);
    foreign.weights[0][1] = 0.5;
    assert_refusal(&cuts, &mut foreign.parts(), &foreign);
    let foreign = Fixture::new(-1.25, false);
    assert_refusal(&cuts, &mut foreign.parts(), &foreign);
    let mut parts = f.parts();
    parts.problem.fix_col(parts.input_vars[1], 0.0);
    assert_refusal(&cuts, &mut parts, &f);
    let mut parts = f.parts();
    parts.output_vars.swap(0, 1);
    assert_refusal(&cuts, &mut parts, &f);
}

#[test]
fn joint_mip_partial_support_preparation_and_expired_authority_never_append() {
    let f = Fixture::new(-0.75, false);
    let p = f.source();
    let cuts = separated(&p);
    for allowed in [0, 1, 4, 8] {
        let mut n = 0;
        assert!(support(
            &p,
            &[0, 1],
            &cuts.joint[0].weights,
            &cuts.joint[0].g,
            end(),
            &mut || {
                n += 1;
                n <= allowed
            }
        )
        .is_none());
        let mut parts = f.parts();
        let before = parts.problem.clone();
        let encoding = f.binding(&parts);
        let mut n = 0;
        assert!(cuts
            .append_with_poll(&mut parts, &encoding, end(), &mut || {
                n += 1;
                n <= allowed
            })
            .is_none());
        assert_eq!(parts.problem, before);
    }
    let mut stale = separated(&p);
    stale.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert_refusal(&stale, &mut f.parts(), &f);
}

#[test]
fn joint_mip_optional_discovery_refusal_retains_individual_planes_for_live_append() {
    let f = Fixture::new(-0.75, false);
    let p = f.source();
    let mut cuts = manual(&p);
    discover_impl(&mut cuts, end(), &mut || false, false);
    assert!(cuts.joint.is_empty());
    assert_eq!(cuts.planes.len(), 2);
    let mut parts = f.parts();
    let encoding = f.binding(&parts);
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(
        (report.individual_cuts, report.joint_cuts, report.cuts),
        (2, 0, 2)
    );
}

#[test]
fn joint_mip_overflowing_optional_pair_is_discarded_before_bundle_selection() {
    let eps = f64::from(f32::from_bits(1));
    let w = vec![2.0, 1.0, 2.0];
    let b = vec![0.0; 3];
    let second = vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0];
    let bias = vec![1.0, 0.0];
    let p = HullMarginProblem::new(
        w.clone(),
        b.clone(),
        1,
        &second,
        &bias,
        2,
        &[(0, 1.0), (1, -1.0)],
        0.0,
        vec![-eps],
        vec![eps],
    )
    .unwrap();
    let mut m = HullMultipliers::zeros(&p);
    m.g = vec![1e308, -1e308, 1e308];
    let lower = certify_hull_lower_bound(&p, &m).unwrap();
    let mut cuts = prepare(&p, m, lower, end(), &mut || true).unwrap();
    let mut n = 0;
    discover_impl(
        &mut cuts,
        end(),
        &mut || {
            n += 1;
            n <= 2
        },
        false,
    );
    assert_eq!(
        (cuts.joint_attempted, cuts.joint_completed, cuts.joint.len()),
        (1, 0, 0)
    );
    let weights = vec![w, second];
    let biases = vec![b, bias];
    let dims = vec![1, 3, 2];
    let bounds = vec![vec![
        Bound::new((-2.0 * eps) as f32, (2.0 * eps) as f32),
        Bound::new(-eps as f32, eps as f32),
        Bound::new((-2.0 * eps) as f32, (2.0 * eps) as f32),
    ]];
    let mut e = crate::encode_feedforward(
        &weights,
        &biases,
        &dims,
        &[Bound::new(-eps as f32, eps as f32)],
        &bounds,
    )
    .unwrap();
    assert!(cuts
        .planes
        .iter()
        .all(|r| r.upper.is_finite() && r.g.iter().all(|g| g.is_finite())));
    let unsafe_row = e.constrain_output_leq_row(0, 1).unwrap();
    let mut parts = e.into_parts();
    let encoding = HullMipEncoding {
        weights: &weights,
        biases: &biases,
        dimensions: &dims,
        hidden_bounds: &bounds,
        first_output: 0,
        second_output: 1,
        reverse: false,
        unsafe_row,
    };
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(
        (report.individual_cuts, report.joint_cuts, report.cuts),
        (3, 0, 3)
    );
}

#[test]
fn joint_mip_discovery_pair_row_caps_and_single_pass_are_shared() {
    let h = 66;
    let w: Vec<_> = (0..h)
        .flat_map(|j| [1.0, if j % 2 == 0 { 1.0 } else { -1.0 }])
        .collect();
    let p = HullMarginProblem::new(
        w.clone(),
        vec![0.0; h],
        2,
        &vec![-1.0; h],
        &[1.0],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0; 2],
        vec![1.0; 2],
    )
    .unwrap();
    let mut m = HullMultipliers::zeros(&p);
    m.g = w.iter().map(|v| 0.5 * v).collect();
    let lower = certify_hull_lower_bound(&p, &m).unwrap();
    let mut cuts = prepare(&p, m, lower, end(), &mut || true).unwrap();
    discover_impl(&mut cuts, end(), &mut || true, false);
    assert_eq!(cuts.joint_attempted, MAX_PAIRS);
    assert!(cuts.joint_completed <= MAX_PAIRS);
    // Cross-parity pairs have strictly positive gain, so this also exercises
    // actual truncation of more than 64 admitted candidates, not an empty set.
    assert_eq!(cuts.joint.len(), MAX_ROWS);
    assert_eq!(cuts.planes.len(), h);
    let before = (cuts.joint_attempted, cuts.joint_completed, cuts.joint.len());
    discover_impl(&mut cuts, end(), &mut || true, false);
    assert_eq!(
        (cuts.joint_attempted, cuts.joint_completed, cuts.joint.len()),
        before
    );
}

#[test]
fn joint_mip_full_binding_keeps_sign_reduced_binary_metadata_and_individual_rows() {
    let f = Fixture::new(-0.75, true);
    let p = f.source();
    let cuts = separated(&p);
    let mut parts = f.parts();
    let positive = *parts.binary_vars.last().unwrap();
    assert!(parts.problem.relax_binary_integrality(&[positive]));
    parts.binary_vars.pop();
    parts.binary_widths.pop();
    let original = parts.problem.clone();
    let variables = parts.binary_vars.clone();
    let encoding = f.binding(&parts);
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(report.individual_cuts, cuts.planes.len());
    assert_eq!(report.joint_cuts, 1);
    assert_eq!(
        &parts.problem.rows()[..original.num_rows()],
        original.rows()
    );
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(parts.binary_vars, variables);
    assert_eq!(parts.problem.num_rows(), original.num_rows() + report.cuts);
}
