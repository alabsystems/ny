// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual eligible default helper and original IR; no corpus/timing claim.
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
                let col = if permuted { (7 * i + 3) % d } else { i };
                first[j * d + col] = v;
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
        let mut e =
            encode_feedforward(&self.w, &self.b, &self.dims, &self.input, &self.bounds).unwrap();
        let rows = add_vnnlib_constraints(&mut e, &self.spec).unwrap();
        let mut p = e.into_parts();
        let r = convex_binary::relax(
            &mut p,
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
        assert_eq!(r.relaxed_nonnegative, 2);
        assert_eq!(r.negative_binaries, 2);
        (p, rows)
    }
    fn attempt(
        &self,
        p: &mut MipParts,
        rows: &[Row],
        allowed: bool,
        end: Instant,
    ) -> Option<StrengtheningOutcome> {
        strengthen_with_signed_refinement(
            p,
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
        let (mut p, rows) = self.parts();
        let before = p.problem.clone();
        let binary = p.binary_vars.clone();
        let widths = p.binary_widths.clone();
        let out = self
            .attempt(
                &mut p,
                &rows,
                allowed,
                Instant::now() + Duration::from_secs(30),
            )
            .unwrap();
        assert_eq!(&p.problem.rows()[..before.num_rows()], before.rows());
        assert_eq!(p.problem.cols(), before.cols());
        assert_eq!(p.binary_vars, binary);
        assert_eq!(p.binary_widths, widths);
        (p, out)
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
fn signed_child_cli_default_unsafe_tie_collect_real_second_replay_and_retain_original_witness() {
    for bias in [-0.25, 0.] {
        let f = Fixture::new(bias, true);
        let (_, out) = f.run(true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert!(out.cuts.signed.as_ref().unwrap().admitted);
        assert!(out.aggregate.is_none());
        let proof = out
            .continuation
            .expect("actual native continuation with immutable signed cache");
        assert!(!proof.verified());
        assert_eq!(proof.report().root_lp_calls, 0);
        let child = proof
            .signed_child_report()
            .expect("attempt accounting survives native open result");
        assert_eq!(child.collection_attempts, 1);
        assert_eq!(child.cached_masks, 4);
        assert!(child.cache_bytes > 0 && child.work_units > 0 && child.attempted_calls > 0);
        assert!(child.completed_calls <= child.attempted_calls && child.attempted_calls <= 256);
        assert!(child.lp_skips <= child.positive_calls);
        assert_eq!(child.source_replays, 2);
        assert!(child.work_units <= child.total_work_units && child.total_work_units <= 64_000_000);
        assert!(out.cuts.signed.as_ref().unwrap().admitted_bytes + child.cache_bytes <= 8 << 20);
        let x = vec![Q::zero(); 30];
        assert_eq!(f.margin(&x), Q::from_float(bias).unwrap());
        assert!(f.margin(&x) <= Q::zero());
    }
}

#[test]
fn signed_child_cli_safe_default_one_lp_and_excluded_ingress_keep_original_owner() {
    for permuted in [false, true] {
        let f = Fixture::new(0.25, permuted);
        let (_, out) = f.run(true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        let proof = out
            .aggregate
            .expect("existing authenticated signed aggregate owner");
        assert!(proof.verified() && proof.lower() > 0.249 && proof.lower() <= 0.25);
        assert_eq!(proof.route(), "signed-coordinate");
        assert!(out.continuation.is_none());
        let (_, excluded) = f.run(false);
        assert!(excluded.aggregate.is_none() && excluded.continuation.is_none());
        assert_eq!(excluded.cuts.proposal_lp_calls, 1);
        assert!(excluded.cuts.signed.as_ref().unwrap().admitted);
    }
    for shared in [false, true] {
        for canary in [false, true] {
            for capture in [false, true] {
                assert_eq!(
                    short_proof_allowed(shared, canary, capture),
                    !shared && !canary && !capture
                );
            }
        }
    }
}

#[test]
fn signed_child_cli_foreign_original_model_and_expired_stage_refuse_atomic_append() {
    for kind in 0..4 {
        let mut f = Fixture::new(-0.25, false);
        let (mut p, rows) = f.parts();
        match kind {
            1 => p.output_vars.swap(0, 1),
            2 => f.w[0][29] = 0.75,
            3 => f.input[29] = Bound::new(-0.5, 1.),
            _ => {}
        }
        let before = p.problem.clone();
        let binary = p.binary_vars.clone();
        let widths = p.binary_widths.clone();
        let end = if kind == 0 {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline")
        } else {
            Instant::now() + Duration::from_secs(30)
        };
        assert!(f.attempt(&mut p, &rows, true, end).is_none());
        assert_eq!(p.problem, before);
        assert_eq!(p.binary_vars, binary);
        assert_eq!(p.binary_widths, widths);
    }
}
