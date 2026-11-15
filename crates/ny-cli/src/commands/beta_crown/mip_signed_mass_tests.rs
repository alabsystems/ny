// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-IR/native-owner controls; no default partition or target claim.
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
    fn new(bias: f64, coefficient: f64, permuted: bool) -> Self {
        let (d, h) = (30, 17);
        let mut first = vec![0.; d * h];
        let mut c = vec![0.; h];
        for j in 0..h {
            let head = if permuted { (5 * j + 3) % h } else { j };
            let input = if permuted { 7 } else { 29 };
            first[head * d + input] = 1.;
            c[head] = if j == 0 { coefficient } else { -1. };
        }
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = d;
        spec.num_outputs = 2;
        spec.input_bounds = vec![(-1., 1.); d];
        spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
        Self {
            w: vec![first, c.into_iter().chain(vec![0.; h]).collect()],
            b: vec![vec![0.; h], vec![bias, 0.]],
            dims: vec![d, h, 2],
            input: vec![Bound::new(-1., 1.); d],
            bounds: vec![vec![Bound::new(-1., 1.); h]],
            spec,
        }
    }
    fn parts(&self) -> (MipParts, Vec<Row>) {
        let mut e =
            encode_feedforward(&self.w, &self.b, &self.dims, &self.input, &self.bounds).unwrap();
        let rows = add_vnnlib_constraints(&mut e, &self.spec).unwrap();
        let mut parts = e.into_parts();
        let r = convex_binary::relax(
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
        assert_eq!(r.relaxed_nonnegative, 1);
        assert_eq!(r.negative_binaries, 16);
        (parts, rows)
    }
    fn attempt(
        &self,
        parts: &mut MipParts,
        rows: &[Row],
        allowed: bool,
        end: Instant,
    ) -> Option<StrengtheningOutcome> {
        // EXACT current production default helper, with no alternate optimizer,
        // manual partition, plan injection, additional LP or proof bypass.
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
        let (d, h) = (self.dims[0], self.dims[1]);
        let mut out = Q::from_float(self.b[1][0]).unwrap();
        for j in 0..h {
            let z = (0..d)
                .map(|i| Q::from_float(self.w[0][j * d + i]).unwrap() * &x[i])
                .sum::<Q>();
            out += Q::from_float(self.w[1][j]).unwrap() * z.max(Q::zero());
        }
        out
    }
}
#[test]
fn signed_mass_cli_actual_default_helper_selects_shared_geometry_and_preserves_original_ir() {
    for permuted in [false, true] {
        let f = Fixture::new(0.25, 16., permuted);
        let (_, out) = f.run(true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert!(out.cuts.checked_hull_lower < 0.);
        let s = out
            .cuts
            .signed
            .as_ref()
            .expect("completed helper must retain no-row/row attempt diagnostics");
        assert!(s.admitted && s.mass.attempted && s.mass.selected);
        assert_eq!(s.components, 11);
        let root = &s.root_coupling;
        assert!(
            root.requested && root.admitted && root.selected,
            "root={root:?}"
        );
        assert_eq!(root.parent_count, 6, "root={root:?}");
        assert!(root.parent_indices.windows(2).all(|w| w[0] < w[1]));
        assert_eq!((root.selected_negatives, root.tail_negatives), (6, 0));
        assert_eq!((root.old_masks, root.new_masks), (32, 84));
        assert_eq!((root.positive_heads, root.recipient_positive_heads), (1, 0));
        assert_eq!(
            (
                root.producer_attempts,
                root.support_attempts,
                root.completed_supports,
                root.checker_slots
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(root.work_units, 8_000_000);
        assert!(root.used_work_units > 0 && root.used_work_units <= root.work_units);
        assert!(root.actual_check_work > 0 && root.actual_check_work <= root.quoted_check_work);
        let checked_root = root.after_global_lower.unwrap();
        assert!(
            checked_root > root.before_global_lower.unwrap(),
            "root={root:?}"
        );
        assert_eq!(
            checked_root.to_bits(),
            root.proposed_lower.unwrap().to_bits()
        );
        let exact_root = 0.25 - 5.;
        assert!(
            checked_root > exact_root - 0.001 && checked_root <= exact_root,
            "six-parent checked root={root:?}"
        );

        assert_eq!(s.assigned_positive_heads, 1);
        assert_eq!(s.positive_appearances, 2);
        assert_eq!(s.split_positive_heads, 1);
        assert_eq!(s.mass.baseline_checks, 1);
        assert_eq!(s.mass.final_checks, 1);
        assert_eq!(s.mass.funded_visits, 2);
        assert_eq!(s.mass.completed_visits, 2);
        assert!(s.lower.is_some_and(|v| v > -4.251 && v <= -4.25));
        assert!(
            s.work_units <= 64_000_000 && s.admitted_bytes <= (8 << 20) && s.source_replays >= 1
        );
        // A safe graph alone does not guarantee the bounded original tree
        // closes all sixteen tied negative phases. No root token is minted.
        assert!(out.aggregate.is_none());
        if let Some(proof) = out.continuation {
            // The default native route reuses the proposal and permits one
            // additional root LP, checked against the complete old floor.
            let tree = proof.report();
            assert_eq!(tree.root_lp_calls, 1);
            let before = tree.root_before_native.expect("native root/cache floor");
            assert!(before.is_finite() && before >= s.lower.unwrap());
            // The standalone LP field excludes the separate cache bound;
            // every open region must inherit the effective root/cache floor.
            assert_eq!(tree.worst.is_none(), tree.open == 0);
            assert!(tree.worst.is_none_or(|after| after >= before));
            if proof.verified() {
                assert_eq!(proof.report().open, 0);
            }
        }
        assert_eq!(f.margin(&vec![Q::zero(); 30]), Q::from_float(0.25).unwrap());
    }
}
#[test]
fn signed_mass_cli_unsafe_and_tie_keep_original_margin_and_never_publish_positive_owner() {
    for bias in [-0.25, 0.] {
        let f = Fixture::new(bias, 16., false);
        let (_, out) = f.run(true);
        let s = out.cuts.signed.as_ref().unwrap();
        assert!(s.mass.attempted && s.mass.selected);
        assert!(s.lower.is_some_and(|v| v <= bias));
        assert!(out.aggregate.is_none());
        assert!(out
            .continuation
            .as_ref()
            .is_none_or(|proof| !proof.verified()));
        assert_eq!(f.margin(&vec![Q::zero(); 30]), Q::from_float(bias).unwrap());
    }
}
#[test]
fn signed_mass_cli_half_mass_cannot_double_an_insufficient_original_positive_coefficient() {
    let f = Fixture::new(0.25, 8., false);
    let (_, out) = f.run(true);
    assert!(out.aggregate.is_none());
    assert!(out
        .continuation
        .as_ref()
        .is_none_or(|proof| !proof.verified()));
    let mut witness = vec![Q::zero(); 30];
    witness[29] = Q::from_integer(1.into());
    assert_eq!(f.margin(&witness), Q::from_float(-7.75).unwrap());
    assert!(out
        .cuts
        .signed
        .as_ref()
        .unwrap()
        .lower
        .is_none_or(|v| v <= -7.75));
}
#[test]
fn signed_mass_cli_foreign_original_weights_bias_box_or_output_layout_refuse_atomically() {
    for kind in 0..4 {
        let mut f = Fixture::new(0.25, 16., false);
        let (mut parts, rows) = f.parts();
        match kind {
            0 => f.w[0][29] = 0.5,
            1 => f.b[0][0] = 0.125,
            2 => f.input[29] = Bound::new(-0.5, 1.),
            _ => parts.output_vars.swap(0, 1),
        }
        let before = parts.problem.clone();
        let binary = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        assert!(f
            .attempt(
                &mut parts,
                &rows,
                true,
                Instant::now() + Duration::from_secs(30)
            )
            .is_none());
        assert_eq!(parts.problem, before);
        assert_eq!(parts.binary_vars, binary);
        assert_eq!(parts.binary_widths, widths);
    }
}
#[test]
fn signed_mass_cli_disabled_publication_keeps_shared_rows_without_opaque_native_authority() {
    let f = Fixture::new(0.25, 16., false);
    let (_, out) = f.run(false);
    assert!(out
        .cuts
        .signed
        .as_ref()
        .is_some_and(|s| s.admitted && s.mass.selected && s.split_positive_heads == 1));
    assert!(out.aggregate.is_none() && out.continuation.is_none());
    for shared in [false, true] {
        for canary in [false, true] {
            for bridge in [false, true] {
                assert_eq!(
                    short_proof_allowed(shared, canary, bridge),
                    !shared && !canary && !bridge
                );
            }
        }
    }
}
#[test]
fn signed_mass_cli_actual_ay_keeps_the_original_unsafe_graph_after_split_rows() {
    let f = Fixture::new(-0.25, 16., false);
    let (parts, out) = f.run(false);
    assert!(out.cuts.signed.as_ref().is_some_and(|s| s.mass.selected));
    let solver = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
            ay_hard_deadline: Some(Instant::now() + Duration::from_secs(10)),
            ..Default::default()
        },
    );
    let MipResult::Sat { input_values, .. } = solver.check_feasibility().unwrap() else {
        panic!("original unsafe graph was excluded")
    };
    let x: Vec<_> = input_values
        .into_iter()
        .map(|v| Q::from_float(v).unwrap())
        .collect();
    assert!(x
        .iter()
        .all(|v| v >= &Q::from_integer((-1).into()) && v <= &Q::from_integer(1.into())));
    assert!(f.margin(&x) <= Q::zero());
}

#[test]
fn signed_mass_cli_actual_default_shared_root_mints_only_the_original_model_bound_native_owner() {
    for permuted in [false, true] {
        let f = Fixture::new(4.75, 16., permuted);
        let (_, out) = f.run(true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert!(out.cuts.checked_hull_lower > -3.251 && out.cuts.checked_hull_lower <= -3.25);
        let s = out.cuts.signed.as_ref().unwrap();
        // Keep the stronger default root stage genuinely open (-0.25), so
        // only the subsequent checked half-sharing improvement earns MASS credit.
        assert_eq!(s.components, 11);
        assert!(s.admitted && s.mass.attempted && s.mass.selected);
        let root = &s.root_coupling;
        assert!(
            root.requested && root.admitted && root.selected,
            "root={root:?}"
        );
        assert_eq!(root.parent_count, 6, "root={root:?}");
        assert!(root.parent_indices.windows(2).all(|w| w[0] < w[1]));
        assert_eq!((root.selected_negatives, root.tail_negatives), (6, 0));
        assert_eq!((root.old_masks, root.new_masks), (32, 84));
        assert_eq!((root.positive_heads, root.recipient_positive_heads), (1, 0));
        assert_eq!(
            (
                root.producer_attempts,
                root.support_attempts,
                root.completed_supports,
                root.checker_slots
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(root.work_units, 8_000_000);
        assert!(root.used_work_units > 0 && root.used_work_units <= root.work_units);
        assert!(root.actual_check_work > 0 && root.actual_check_work <= root.quoted_check_work);
        let checked_root = root.after_global_lower.unwrap();
        assert!(
            checked_root > root.before_global_lower.unwrap(),
            "root={root:?}"
        );
        assert_eq!(
            checked_root.to_bits(),
            root.proposed_lower.unwrap().to_bits()
        );
        let exact_root = -0.25;
        assert!(
            checked_root > exact_root - 0.001 && checked_root <= exact_root,
            "six-parent checked root={root:?}"
        );

        assert_eq!(s.split_positive_heads, 1);
        assert_eq!(s.positive_appearances, 2);
        assert!(s
            .mass
            .banked_lower
            .is_some_and(|v| v > -0.251 && v <= -0.25));
        assert!(s.mass.lower.is_some_and(|v| v > 0.249 && v <= 0.25));
        assert_eq!(s.mass.baseline_checks, 1);
        assert_eq!(s.mass.final_checks, 1);
        assert_eq!((s.mass.funded_visits, s.mass.completed_visits), (2, 2));
        assert!(s.work_units <= 64_000_000 && s.admitted_bytes <= (8 << 20));
        assert_eq!(s.source_replays, 1);
        let proof = out
            .aggregate
            .expect("positive checked shared row must use the original native root owner");
        assert_eq!(proof.route(), "signed-coordinate");
        assert!(proof.verified());
        assert!(proof.lower() > 0.249 && proof.lower() <= 0.25);
        assert!(out.continuation.is_none());
        assert_eq!(f.margin(&vec![Q::zero(); 30]), Q::from_float(4.75).unwrap());
    }
}
