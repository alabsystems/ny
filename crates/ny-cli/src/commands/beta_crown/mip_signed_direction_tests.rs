// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Same production helper, original tuples and proof ownership; no corpus claim.
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
use ny_onnx::vnnlib::OutputConstraint;

struct Fixture {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<usize>,
    input: Vec<Bound>,
    bounds: Vec<Vec<Bound>>,
    spec: VnnLibSpec,
}
impl Fixture {
    fn new(bias: f64, permuted: bool) -> Self {
        let d = 30;
        let h = 4;
        let mut first = vec![0.; d * h];
        let mut b = vec![0.; h];
        let mut c = vec![0.; h];
        for role in 0..h {
            let j = if permuted { (3 * role + 1) % h } else { role };
            let k = role % 2;
            for (i, v) in if k == 0 {
                [(0, 1.), (29, 0.5)]
            } else {
                [(1, 1.), (28, -0.25)]
            } {
                let column = if permuted { (7 * i + 3) % d } else { i };
                first[j * d + column] = v;
            }
            b[j] = if k == 0 { 0.125 } else { -0.125 };
            c[j] = if role < 2 { 1. } else { -1. };
        }
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = d;
        spec.num_outputs = 2;
        spec.input_bounds = vec![(-1., 1.); d];
        spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
        Self {
            w: vec![first, c.into_iter().chain(vec![0.; h]).collect()],
            b: vec![b, vec![bias, 0.]],
            dims: vec![d, h, 2],
            input: vec![Bound::new(-1., 1.); d],
            bounds: vec![vec![Bound::new(-2., 2.); h]],
            spec,
        }
    }
    fn parts(&self) -> (MipParts, Vec<Row>) {
        let mut encoded =
            encode_feedforward(&self.w, &self.b, &self.dims, &self.input, &self.bounds).unwrap();
        let rows = add_vnnlib_constraints(&mut encoded, &self.spec).unwrap();
        let mut parts = encoded.into_parts();
        let relaxed = convex_binary::relax(
            &mut parts,
            true,
            &self.w,
            &self.b,
            &self.dims,
            &self.bounds,
            &self.spec,
            &rows,
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(relaxed.relaxed_nonnegative, 2);
        assert_eq!(relaxed.negative_binaries, 2);
        (parts, rows)
    }
    fn attempt(
        &self,
        parts: &mut MipParts,
        rows: &[Row],
        allowed: bool,
        end: Instant,
    ) -> Option<StrengtheningOutcome> {
        // Exactly the helper called by the eligible default MIP production path.
        strengthen_with_signed_refinement(
            parts,
            true,
            &self.w,
            &self.b,
            &self.dims,
            &self.bounds,
            &self.input,
            &self.spec,
            rows,
            end,
            allowed,
        )
    }
    fn run(&self, allowed: bool) -> (MipParts, StrengtheningOutcome) {
        let (mut parts, rows) = self.parts();
        let before = parts.problem.clone();
        let binary = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        let out = self
            .attempt(
                &mut parts,
                &rows,
                allowed,
                Instant::now() + Duration::from_secs(30),
            )
            .unwrap();
        assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
        assert_eq!(parts.problem.cols(), before.cols());
        assert_eq!(parts.binary_vars, binary);
        assert_eq!(parts.binary_widths, widths);
        (parts, out)
    }
    fn margin(&self, x: &[Q]) -> Q {
        let d = self.dims[0];
        let h = self.dims[1];
        Q::from_float(self.b[1][0]).unwrap()
            + (0..h)
                .map(|j| {
                    let z = Q::from_float(self.b[0][j]).unwrap()
                        + (0..d)
                            .map(|i| Q::from_float(self.w[0][j * d + i]).unwrap() * &x[i])
                            .sum::<Q>();
                    Q::from_float(self.w[1][j]).unwrap() * z.max(Q::zero())
                })
                .sum::<Q>()
    }
}
#[test]
fn signed_direction_cli_default_safe_permutations_keep_one_lp_and_opaque_original_proof() {
    for permuted in [false, true] {
        let (_, out) = Fixture::new(0.25, permuted).run(true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert!(out.cuts.checked_hull_lower < 0.);
        let s = out.cuts.signed.as_ref().unwrap();
        let d = &s.direction;
        assert!(
            s.admitted && s.source_replays > 0 && s.lower.is_some_and(|v| v > 0.249 && v <= 0.25)
        );
        assert_eq!(d.attempts, 1);
        assert_eq!(d.completed_sweeps, 2);
        assert!(d.cached_masks > 0 && d.coordinates > 0 && d.work_units > 0 && d.cache_bytes > 0);
        assert!(
            d.work_units <= s.work_units
                && s.work_units <= 64_000_000
                && s.admitted_bytes <= 8 << 20
        );
        let proof = out.aggregate.unwrap();
        assert_eq!(proof.route(), "signed-coordinate");
        assert!(proof.verified() && proof.lower() > 0.249 && proof.lower() <= 0.25);
        assert!(out.continuation.is_none());
    }
}
#[test]
fn signed_direction_cli_default_unsafe_tie_and_policy_refusal_retain_original_witness() {
    for bias in [-0.25, 0.] {
        let f = Fixture::new(bias, true);
        let (_, out) = f.run(false);
        let s = out.cuts.signed.as_ref().unwrap();
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert_eq!(s.direction.attempts, 1);
        assert_eq!(s.direction.completed_sweeps, 2);
        assert!(s.lower.is_some_and(|v| v <= bias));
        assert!(out.aggregate.is_none() && out.continuation.is_none());
        let zero = vec![Q::zero(); 30];
        assert_eq!(f.margin(&zero), Q::from_float(bias).unwrap());
        assert!(f.margin(&zero) <= Q::zero());
    }
    let (_, out) = Fixture::new(0.25, false).run(false);
    assert_eq!(out.cuts.signed.as_ref().unwrap().direction.attempts, 1);
    assert!(
        out.aggregate.is_none() && out.continuation.is_none(),
        "geometry cannot bypass ingress exclusions"
    );
}
#[test]
fn signed_direction_cli_expired_foreign_models_refuse_atomically_and_old_positive_skips_work() {
    for kind in 0..4 {
        let mut f = Fixture::new(0.25, false);
        let (mut parts, rows) = f.parts();
        if kind == 1 {
            parts.output_vars.swap(0, 1);
        }
        if kind == 2 {
            f.w[0][29] = 0.75;
        }
        if kind == 3 {
            f.input[29] = Bound::new(-0.5, 1.);
        }
        let before = parts.problem.clone();
        let binary = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        let end = if kind == 0 {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline")
        } else {
            Instant::now() + Duration::from_secs(30)
        };
        assert!(f.attempt(&mut parts, &rows, true, end).is_none());
        assert_eq!(parts.problem, before);
        assert_eq!(parts.binary_vars, binary);
        assert_eq!(parts.binary_widths, widths);
    }
    let (_, out) = Fixture::new(10., false).run(true);
    let s = out.cuts.signed.as_ref().unwrap();
    assert_eq!(s.stop_reason, "old_positive");
    assert_eq!(s.direction.attempts, 0);
    assert_eq!(s.direction.work_units, 0);
    assert!(!s.admitted && s.coordinate_proposals == 0 && s.completed_masks == 0);
    assert!(out.cuts.checked_hull_lower > 0. && s.lower.is_some_and(|v| v > 0.));
    // This fixture is already positive in the individual scalar hull. It has
    // no compact/global/signed aggregate owner; native reuse rechecks its root.
    assert!(out.aggregate.is_none());
    let proof = out
        .continuation
        .expect("opaque replay of the old positive scalar");
    let report = proof.report();
    assert!(proof.verified() && report.root_certified.is_some_and(|v| v > 0.));
    assert_eq!(report.root_lp_calls, 0);
    assert_eq!(report.proposal_lp_calls, 0);
    assert_eq!(report.open, 0);
}
