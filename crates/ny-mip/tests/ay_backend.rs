// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end tests of the ay MIP backend (in-process ay-milp library since
// R3 — no external binary needed). Each positive case has its refuted
// negative twin (a3d methodology: a green without its refuted twin is
// vacuous). The differential gate (mip-diff) is the corpus-scale check;
// these are the smoke pair.

use ny_mip::{
    MilpProblem, MipBackend, MipConfig, MipParts, MipResult, MipSolver, OneSidedSatDecline,
    OneSidedSatProbe,
};

fn solve(problem: MilpProblem, binary_vars: Vec<ny_mip::ir::Col>) -> MipResult {
    let num_cols = problem.num_cols();
    let all: Vec<ny_mip::ir::Col> = (0..num_cols).map(ny_mip::ir::Col).collect();
    let parts = MipParts {
        problem,
        input_vars: all.clone(),
        output_vars: all,
        binary_widths: vec![1.0; binary_vars.len()],
        binary_vars,
        num_cols,
    };
    let config = MipConfig {
        backend: MipBackend::Ay,
        parallel_split: 1,
        timeout_secs: 30.0,
        ..MipConfig::default()
    };
    MipSolver::new(parts, config)
        .check_feasibility()
        .expect("ay solve should not error")
}

fn probe_solver(problem: MilpProblem, backend: MipBackend) -> MipSolver {
    let num_cols = problem.num_cols();
    let all: Vec<ny_mip::ir::Col> = (0..num_cols).map(ny_mip::ir::Col).collect();
    MipSolver::new(
        MipParts {
            problem,
            input_vars: all.clone(),
            output_vars: all,
            binary_vars: vec![],
            binary_widths: vec![],
            num_cols,
        },
        MipConfig {
            backend,
            parallel_split: 1,
            timeout_secs: 30.0,
            ..MipConfig::default()
        },
    )
}

#[test]
fn test_ay_backend_feasible_lp_returns_sat_witness() {
    // x in [0,1], y in [0,2], y = 2x, x >= 1/2  -> sat (x=1/2..1)
    let mut p = MilpProblem::new();
    let x = p.add_col(0.0, 0.0, 1.0);
    let y = p.add_col(0.0, 0.0, 2.0);
    p.add_row(0.0, 0.0, [(y, 1.0), (x, -2.0)]);
    p.add_row(0.5, f64::INFINITY, [(x, 1.0)]);
    match solve(p, vec![]) {
        MipResult::Sat { input_values, .. } => {
            let (xv, yv) = (input_values[0], input_values[1]);
            assert!((0.5 - 1e-9..=1.0 + 1e-9).contains(&xv), "x={xv}");
            assert!((yv - 2.0 * xv).abs() < 1e-9, "y={yv} != 2x={}", 2.0 * xv);
        }
        other => panic!("expected Sat, got {other:?}"),
    }
}

#[test]
fn test_ay_backend_infeasible_lp_returns_unsat() {
    // Negative twin of the feasible case: additionally require x <= 1/4.
    let mut p = MilpProblem::new();
    let x = p.add_col(0.0, 0.0, 1.0);
    let y = p.add_col(0.0, 0.0, 2.0);
    p.add_row(0.0, 0.0, [(y, 1.0), (x, -2.0)]);
    p.add_row(0.5, f64::INFINITY, [(x, 1.0)]);
    p.add_row(f64::NEG_INFINITY, 0.25, [(x, 1.0)]);
    assert!(
        matches!(solve(p, vec![]), MipResult::Unsat { .. }),
        "x>=1/2 and x<=1/4 must be unsat"
    );
}

fn result_tag(result: &MipResult) -> &'static str {
    match result {
        MipResult::Sat { .. } => "sat",
        MipResult::Unsat { .. } => "unsat",
        MipResult::Timeout => "timeout",
        MipResult::Error(_) => "error",
    }
}

/// End-to-end SAT parity for NY's explicit marker plumbing.  The constraints
/// are byte-identical; the marked copy merely opts AY into the equivalent
/// margin-optimization schedule.
#[test]
fn test_marked_margin_sat_matches_plain_feasibility() {
    let mut marked = MilpProblem::new();
    let x = marked.add_col(0.0, 0.0, 2.0);
    marked.add_row(0.5, f64::INFINITY, [(x, 1.0)]);
    let margin = marked.add_row(f64::NEG_INFINITY, 1.0, [(x, 1.0)]);
    let plain = marked.clone();
    marked.mark_margin_row(margin).expect("valid decision row");

    let marked_result = solve(marked, vec![]);
    let plain_result = solve(plain, vec![]);
    assert!(
        matches!(marked_result, MipResult::Sat { .. }),
        "marked reachable band must be SAT, got {}",
        result_tag(&marked_result)
    );
    assert!(
        matches!(plain_result, MipResult::Sat { .. }),
        "plain reachable band must be SAT, got {}",
        result_tag(&plain_result)
    );
}

/// End-to-end UNSAT parity and certificate admission.  `R` forces x>=1.5,
/// while the marked decision row asks x<=1; the reframed optimum must map back
/// to a certificate-verified UNSAT for the original feasibility problem.
#[test]
fn test_marked_margin_unsat_matches_plain_and_is_certified() {
    let mut marked = MilpProblem::new();
    let x = marked.add_col(0.0, 0.0, 2.0);
    marked.add_row(1.5, f64::INFINITY, [(x, 1.0)]);
    let margin = marked.add_row(f64::NEG_INFINITY, 1.0, [(x, 1.0)]);
    let plain = marked.clone();
    marked.mark_margin_row(margin).expect("valid decision row");

    let marked_result = solve(marked, vec![]);
    let plain_result = solve(plain, vec![]);
    assert!(
        matches!(marked_result, MipResult::Unsat { certified: true }),
        "marked unreachable band must be certified UNSAT, got {marked_result:?}"
    );
    assert!(
        matches!(plain_result, MipResult::Unsat { .. }),
        "plain unreachable band must be UNSAT, got {plain_result:?}"
    );
}

/// Big-M ReLU shape: z binary, pre-activation w in [-1, 1], post y = relu(w)
/// via the standard triangle + indicator rows.
fn relu_problem() -> (
    MilpProblem,
    ny_mip::ir::Col,
    ny_mip::ir::Col,
    ny_mip::ir::Col,
) {
    let mut p = MilpProblem::new();
    let w = p.add_col(0.0, -1.0, 1.0);
    let y = p.add_col(0.0, 0.0, 1.0);
    let z = p.add_integer_col(0.0, 0.0, 1.0);
    // y >= w ; y <= w + M(1-z) with M=1 ; y <= M*z with M=1
    p.add_row(0.0, f64::INFINITY, [(y, 1.0), (w, -1.0)]);
    p.add_row(f64::NEG_INFINITY, 1.0, [(y, 1.0), (w, -1.0), (z, 1.0)]);
    p.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (z, -1.0)]);
    (p, w, y, z)
}

#[test]
fn test_ay_backend_relu_binary_case_split_sat() {
    // relu(w) >= 1/2 is satisfiable (w >= 1/2, z = 1).
    let (mut p, _w, y, z) = relu_problem();
    p.add_row(0.5, f64::INFINITY, [(y, 1.0)]);
    match solve(p, vec![z]) {
        MipResult::Sat { input_values, .. } => {
            let zv = input_values[2];
            assert!((zv - 1.0).abs() < 1e-9, "active ReLU needs z=1, got {zv}");
        }
        other => panic!("expected Sat, got {other:?}"),
    }
}

#[test]
fn test_ay_backend_relu_binary_case_split_unsat() {
    // Negative twin: relu(w) >= 1/2 AND w <= -1/2 is unsatisfiable.
    let (mut p, w, y, z) = relu_problem();
    p.add_row(0.5, f64::INFINITY, [(y, 1.0)]);
    p.add_row(f64::NEG_INFINITY, -0.5, [(w, 1.0)]);
    assert!(
        matches!(solve(p, vec![z]), MipResult::Unsat { .. }),
        "relu(w) >= 1/2 with w <= -1/2 must be unsat"
    );
}

#[test]
fn test_ay_backend_minimize_output_reports_optimum() {
    // min y s.t. y = 2x, x in [1/4, 1] -> optimum y = 1/2 at x = 1/4.
    let mut p = MilpProblem::new();
    let x = p.add_col(0.0, 0.25, 1.0);
    let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
    p.add_row(0.0, 0.0, [(y, 1.0), (x, -2.0)]);
    let num_cols = p.num_cols();
    let parts = MipParts {
        problem: p,
        input_vars: vec![x],
        output_vars: vec![y],
        binary_vars: vec![],
        binary_widths: vec![],
        num_cols,
    };
    let config = MipConfig {
        backend: MipBackend::Ay,
        parallel_split: 1,
        timeout_secs: 30.0,
        ..MipConfig::default()
    };
    let solver = MipSolver::new(parts, config);
    match solver
        .minimize_output(0)
        .expect("optimize should not error")
    {
        MipResult::Sat {
            objective,
            dual_bound,
            ..
        } => {
            assert!(
                (objective - 0.5).abs() < 1e-9,
                "expected optimum 1/2, got {objective}"
            );
            // A COMPLETED optimization surfaces its proven optimum as a
            // rigorous dual bound (outward-rounded; 1/2 is exact in f64).
            let db = dual_bound.expect("completed optimum must carry a rigorous dual bound");
            assert!(
                (db - 0.5).abs() < 1e-9,
                "dual bound should equal the proven optimum, got {db}"
            );
            assert!(
                db <= objective,
                "Minimize dual bound must never exceed the primal objective"
            );
        }
        other => panic!("expected Sat with optimum, got {other:?}"),
    }
}

/// The twin of the test above, and the one that pins AY `f36f19a5b`'s
/// correction: the SAME outcome shape with the SAME certificate is `RimClosed`
/// on a CONTINUOUS model and `SearchTrusted` on an INTEGRAL one.
///
/// `LpSession::verify` asserts the certified dual bound MEETS the primal only
/// under `!model.has_integrality()`. On an integral model the certificate is
/// checked for NON-CROSSING only, and the integrality gap is closed by the
/// search's exhaustiveness, which is not certified. NY therefore withholds the
/// prunable `dual_bound` there — the Sat VERDICT is unchanged and still
/// revalidated downstream by the concrete forward pass.
///
/// This matters because every NY model carrying ReLU binaries is integral, so
/// this — not the continuous case above — is NY's own workload.
#[test]
fn integral_optimum_withholds_the_prunable_dual_bound() {
    // Same shape as the continuous twin, but driven by a ReLU BINARY (the only
    // integer column shape NY's lowering admits):
    // min y s.t. y = 2b, b binary, b = 1 -> optimum y = 2.
    let mut p = MilpProblem::new();
    let b = p.add_integer_col(0.0, 0.0, 1.0);
    let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
    p.add_row(0.0, 0.0, [(y, 1.0), (b, -2.0)]);
    p.add_row(1.0, 1.0, [(b, 1.0)]);
    let num_cols = p.num_cols();
    let parts = MipParts {
        problem: p,
        input_vars: vec![b],
        output_vars: vec![y],
        binary_vars: vec![],
        binary_widths: vec![],
        num_cols,
    };
    let config = MipConfig {
        backend: MipBackend::Ay,
        parallel_split: 1,
        timeout_secs: 30.0,
        ..MipConfig::default()
    };
    let solver = MipSolver::new(parts, config);
    match solver
        .minimize_output(0)
        .expect("optimize should not error")
    {
        MipResult::Sat {
            objective,
            dual_bound,
            ..
        } => {
            assert!(
                (objective - 2.0).abs() < 1e-9,
                "expected optimum 2, got {objective}"
            );
            assert!(
                dual_bound.is_none(),
                "an integral model's optimum is search-trusted, not rim-closed: \
                 the prunable dual bound must be withheld, got {dual_bound:?}"
            );
        }
        other => panic!("expected Sat with optimum, got {other:?}"),
    }
}

#[test]
fn one_sided_sat_probe_minimizes_inside_upper_unsafe_row() {
    // The unsafe system is x <= 1 over [-2,2].  Plain feasibility may stop at
    // the boundary; the direct objective must return the deepest point x=-2.
    let mut problem = MilpProblem::new();
    let x = problem.add_col(0.0, -2.0, 2.0);
    let unsafe_row = problem.add_row(f64::NEG_INFINITY, 1.0, [(x, 1.0)]);
    let solver = probe_solver(problem, MipBackend::Ay);

    let OneSidedSatProbe::Witness(witness) = solver.probe_one_sided_sat(unsafe_row, 5.0) else {
        panic!("reachable upper unsafe row must yield a SAT candidate");
    };
    assert_eq!(witness.objective.to_bits(), (-2.0_f64).to_bits());
    assert_eq!(witness.input_values, vec![-2.0]);
    assert_eq!(witness.output_values, vec![-2.0]);

    assert_eq!(
        solver.probe_one_sided_sat_until(unsafe_row, 5.0, std::time::Instant::now()),
        OneSidedSatProbe::Declined(OneSidedSatDecline::Deadline),
        "an expired caller deadline must reach AY instead of being re-anchored"
    );
}

#[test]
fn one_sided_sat_probe_maximizes_inside_lower_unsafe_row() {
    // Direction twin: x >= -1 must maximize to the opposite box face.
    let mut problem = MilpProblem::new();
    let x = problem.add_col(0.0, -2.0, 2.0);
    let unsafe_row = problem.add_row(-1.0, f64::INFINITY, [(x, 1.0)]);
    let solver = probe_solver(problem, MipBackend::Ay);

    let OneSidedSatProbe::Witness(witness) = solver.probe_one_sided_sat(unsafe_row, 5.0) else {
        panic!("reachable lower unsafe row must yield a SAT candidate");
    };
    assert_eq!(witness.objective.to_bits(), 2.0_f64.to_bits());
    assert_eq!(witness.input_values, vec![2.0]);
}

#[test]
fn one_sided_sat_probe_discards_infeasibility_without_unsat_authority() {
    let mut problem = MilpProblem::new();
    let x = problem.add_col(0.0, 0.0, 1.0);
    let impossible = problem.add_row(2.0, f64::INFINITY, [(x, 1.0)]);
    let solver = probe_solver(problem, MipBackend::Ay);

    assert_eq!(
        solver.probe_one_sided_sat(impossible, 5.0),
        OneSidedSatProbe::Declined(OneSidedSatDecline::InfeasibleIgnored),
        "even exact infeasibility must remain a verdict-neutral decline"
    );
}

#[test]
fn one_sided_sat_probe_refuses_invalid_row_timeout_and_backend() {
    let mut problem = MilpProblem::new();
    let x = problem.add_col(0.0, 0.0, 1.0);
    let equality = problem.add_row(0.5, 0.5, [(x, 1.0)]);
    let ay = probe_solver(problem.clone(), MipBackend::Ay);
    assert!(matches!(
        ay.probe_one_sided_sat(equality, 5.0),
        OneSidedSatProbe::Declined(OneSidedSatDecline::InvalidRow(_))
    ));
    assert_eq!(
        ay.probe_one_sided_sat(equality, 0.0),
        OneSidedSatProbe::Declined(OneSidedSatDecline::InvalidTimeout)
    );

    let subprocess = probe_solver(problem, MipBackend::AyProc);
    assert_eq!(
        subprocess.probe_one_sided_sat(equality, 5.0),
        OneSidedSatProbe::Declined(OneSidedSatDecline::UnsupportedBackend)
    );
}

#[test]
fn expired_ay_hard_deadline_refuses_serial_and_phase_split_work() {
    for parallel_split in [1, 2] {
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(0.0, 1.0, [(x, 1.0)]);
        let num_cols = problem.num_cols();
        let parts = MipParts {
            problem,
            input_vars: vec![x],
            output_vars: vec![x],
            binary_vars: vec![x],
            binary_widths: vec![1.0],
            num_cols,
        };
        let solver = MipSolver::new(
            parts,
            MipConfig {
                backend: MipBackend::Ay,
                parallel_split,
                timeout_secs: 100.0,
                ay_hard_deadline: Some(std::time::Instant::now()),
                ..MipConfig::default()
            },
        );
        let started = std::time::Instant::now();
        assert!(
            matches!(solver.check_feasibility(), Ok(MipResult::Timeout)),
            "expired hard deadline must be verdict-neutral (split={parallel_split})"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "expired hard deadline must not start fresh work (split={parallel_split})"
        );
    }
}
