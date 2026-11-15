// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};

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
        (model.col_at(parts.output_vars[0].0).unwrap(), 1.),
        (model.col_at(parts.output_vars[1].0).unwrap(), -1.),
    ];
    let out = session
        .rigorous_bound_expr(&objective, ay_milp::Sense::Minimize)
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&out).unwrap()
}
#[test]
fn compact_pair_actual_lp_beats_individual_root_and_ay_certifies_safe_original() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    let mut parts = f.parts();
    let binding = f.binding(&parts);
    let old = actual_lower(&parts, binding.unsafe_row);
    assert!(old <= -1.749, "{old}");
    let cuts = propose_root_mip_cuts(&p, end()).unwrap();
    cuts.append_to(&mut parts, &binding, end()).unwrap();
    let after = actual_lower(&parts, binding.unsafe_row);
    assert!(after > 0.249 && after <= 0.25, "{after}");
    let solver = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
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
fn compact_pair_unsafe_original_keeps_nonpositive_floor_and_actual_rational_sat_witness() {
    let f = Fixture::new(-1.75);
    let p = f.source();
    let cuts = compact(&p);
    assert!(cuts.compact_root_proof(end()).is_none());
    let row = replay(&cuts, end(), &mut || true).unwrap();
    assert!(root_lower(&cuts, &row, end(), &mut || true).unwrap() <= -0.25);
    let mut parts = f.parts();
    let binding = f.binding(&parts);
    cuts.append_to(&mut parts, &binding, end()).unwrap();
    let solver = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
            ay_hard_deadline: Some(end()),
            ..Default::default()
        },
    );
    let MipResult::Sat { input_values, .. } = solver.check_feasibility().unwrap() else {
        panic!("unsafe original excluded")
    };
    let x: Vec<_> = input_values.iter().take(4).map(|v| q(*v)).collect();
    assert!(x.iter().all(|v| *v >= q(-1.) && *v <= q(1.)));
    let mut margin = q(1.75) + &x[0] + &x[2];
    for (a, b) in [(0, 1), (2, 3)] {
        margin -= (&x[a] + &x[b]).max(Q::zero());
        margin -= (&x[a] - &x[b]).max(Q::zero());
    }
    assert!(
        margin <= Q::zero(),
        "original rational graph is safe at solver point"
    );
}
