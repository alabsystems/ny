// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual default helper/model ingress; no target or tree-closure claim.
use super::*;
use ny_onnx::vnnlib::OutputConstraint;

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    input: Vec<Bound>,
    bounds: Vec<Vec<Bound>>,
    spec: VnnLibSpec,
}
impl Fixture {
    fn new(bias: f64, permuted: bool) -> Self {
        let (d, h) = (30, 17);
        let mut first = vec![0.; d * h];
        let mut coefficients = vec![0.; h];
        for j in 0..h {
            let head = if permuted { (5 * j + 3) % h } else { j };
            first[head * d + if permuted { 7 } else { 29 }] = 1.;
            coefficients[head] = if j == 0 { 16. } else { -1. };
        }
        coefficients.extend(vec![0.; h]);
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = d;
        spec.num_outputs = 2;
        spec.input_bounds = vec![(-1., 1.); d];
        spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
        Self {
            weights: vec![first, coefficients],
            biases: vec![vec![0.; h], vec![bias, 0.]],
            dims: vec![d, h, 2],
            input: vec![Bound::new(-1., 1.); d],
            bounds: vec![vec![Bound::new(-1., 1.); h]],
            spec,
        }
    }
    fn parts(&self) -> (MipParts, Vec<Row>) {
        let mut encoder = encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &self.input,
            &self.bounds,
        )
        .unwrap();
        let rows = add_vnnlib_constraints(&mut encoder, &self.spec).unwrap();
        let mut parts = encoder.into_parts();
        let report = convex_binary::relax(
            &mut parts,
            true,
            &self.weights,
            &self.biases,
            &self.dims,
            &self.bounds,
            &self.spec,
            &rows,
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(report.relaxed_nonnegative, 1);
        assert_eq!(report.negative_binaries, 16);
        (parts, rows)
    }
    fn attempt(
        &self,
        parts: &mut MipParts,
        rows: &[Row],
        allowed: bool,
        end: Instant,
    ) -> Option<StrengtheningOutcome> {
        strengthen_with_signed_refinement(
            parts,
            true,
            &self.weights,
            &self.biases,
            &self.dims,
            &self.bounds,
            &self.input,
            &self.spec,
            rows,
            end,
            allowed,
        )
    }
}

#[test]
fn signed_component_rows_cli_default_open_root_exports_checked_components_into_original_ir() {
    for permuted in [false, true] {
        let f = Fixture::new(0.25, permuted);
        let (mut parts, rows) = f.parts();
        let old = parts.problem.clone();
        let binaries = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        // Publication is disabled to inspect the first append alone: the same
        // default optimizer and original model binder still run in production.
        let out = f
            .attempt(
                &mut parts,
                &rows,
                false,
                Instant::now() + Duration::from_secs(30),
            )
            .unwrap();
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        let signed = out.cuts.signed.as_ref().unwrap();
        assert!(signed.admitted && signed.mass.selected);
        let root = &signed.root_coupling;
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

        assert_eq!(signed.components, 11);
        assert!(signed.lower.is_some_and(|v| v > -4.251 && v <= -4.25));
        let report = &signed.component_rows;
        assert!(report.attempted && report.admitted);
        assert_eq!(report.collection_attempts, 1);
        assert_eq!(report.collected_components, 11);
        assert_eq!(report.prepared_rows, 11);
        assert_eq!(report.appended_rows, 11);
        assert_eq!(report.prepared_head_terms, 18);
        assert_eq!(report.appended_terms, report.prepared_terms);
        assert!(report.appended_terms >= 18 && report.appended_terms <= report.quoted_terms);
        assert!(report.work_units > 0 && report.reserved_work_units == 0);
        assert_eq!(report.stop_reason, "appended");
        assert_eq!(signed.source_replays, 1);
        assert!(signed.work_units <= 64_000_000 && signed.admitted_bytes <= (8 << 20));
        assert!(out.aggregate.is_none() && out.continuation.is_none());
        assert_eq!(parts.problem.num_rows(), old.num_rows() + out.cuts.cuts);
        assert!(out.cuts.cuts > report.appended_rows);
        assert_eq!(&parts.problem.rows()[..old.num_rows()], old.rows());
        assert_eq!(parts.problem.cols(), old.cols());
        assert_eq!(parts.problem.margin_row(), old.margin_row());
        assert_eq!(parts.binary_vars, binaries);
        assert_eq!(parts.binary_widths, widths);
    }
}

#[test]
fn signed_component_rows_cli_existing_positive_owner_skips_optional_component_export() {
    let f = Fixture::new(10., false);
    let (mut parts, rows) = f.parts();
    let out = f
        .attempt(
            &mut parts,
            &rows,
            true,
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
    let signed = out.cuts.signed.as_ref().unwrap();
    assert_eq!(signed.stop_reason, "old_positive");
    assert_eq!(signed.coordinate_proposals, 0);
    assert!(!signed.mass.attempted);
    assert!(!signed.component_rows.attempted);
    assert_eq!(signed.component_rows.appended_rows, 0);
    assert_eq!(signed.component_rows.work_units, 0);
    // The earlier retained root may be represented by the old compact owner
    // or the native scalar replay. This test requires actual opaque authority,
    // rather than assigning a proof route from positivity alone.
    let verified_old = out
        .aggregate
        .as_ref()
        .is_some_and(|p| p.verified() && p.lower() > 0.)
        || out
            .continuation
            .as_ref()
            .is_some_and(|p| p.verified() && p.report().root_certified.is_some_and(|v| v > 0.));
    assert!(
        verified_old,
        "a completed old positive proof must retain its genuine owner"
    );
    if let Some(proof) = out.continuation {
        assert_eq!(proof.report().root_lp_calls, 0);
        assert_eq!(proof.report().open, 0);
    }
}

#[test]
fn signed_component_rows_cli_unsafe_and_tie_export_geometry_without_positive_publication() {
    for bias in [-0.25, 0.] {
        let f = Fixture::new(bias, false);
        let (mut parts, rows) = f.parts();
        let out = f
            .attempt(
                &mut parts,
                &rows,
                false,
                Instant::now() + Duration::from_secs(30),
            )
            .unwrap();
        let signed = out.cuts.signed.as_ref().unwrap();
        assert!(signed.mass.selected);
        assert_eq!(signed.components, 11);
        let root = &signed.root_coupling;
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
        let exact_root = bias - 5.;
        assert!(
            checked_root > exact_root - 0.001 && checked_root <= exact_root,
            "six-parent checked root={root:?}"
        );

        assert_eq!(signed.component_rows.collected_components, 11);
        assert_eq!(signed.component_rows.prepared_rows, 11);
        assert_eq!(signed.component_rows.appended_rows, 11);
        assert_eq!(signed.component_rows.prepared_head_terms, 18);
        assert_eq!(
            signed.component_rows.appended_terms,
            signed.component_rows.prepared_terms
        );
        assert!(signed
            .lower
            .is_some_and(|v| v > bias - 4.501 && v <= bias - 4.5));
        assert!(signed.lower.is_some_and(|v| v <= bias));
        assert_eq!(signed.source_replays, 1);
        assert!(out.aggregate.is_none() && out.continuation.is_none());
        // All identical original ReLUs cancel, so zero is an exact unsafe/tie
        // original graph witness; accepted geometry may not exclude this point.
        let mut x = vec![0.; parts.problem.num_cols()];
        for (j, col) in parts.output_vars.iter().enumerate() {
            x[col.0] = f.biases[1][j];
        }
        for row in parts.problem.rows() {
            let value: f64 = row.coeffs.iter().map(|(j, a)| a * x[*j]).sum();
            assert!(value >= row.lb && value <= row.ub);
        }
    }
}

#[test]
fn signed_component_rows_cli_foreign_source_layout_or_marker_refuses_the_complete_append_atomically(
) {
    for kind in 0..5 {
        let mut f = Fixture::new(0.25, false);
        let (mut parts, rows) = f.parts();
        match kind {
            0 => f.weights[0][29] = 0.5,
            1 => f.biases[0][0] = 0.125,
            2 => f.input[29] = Bound::new(-0.5, 1.),
            3 => parts.output_vars.swap(0, 1),
            _ => {
                parts.problem.mark_margin_row(Row(30)).unwrap();
            }
        }
        let before = parts.problem.clone();
        let binaries = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        assert!(f
            .attempt(
                &mut parts,
                &rows,
                false,
                Instant::now() + Duration::from_secs(30)
            )
            .is_none());
        assert_eq!(parts.problem, before);
        assert_eq!(parts.binary_vars, binaries);
        assert_eq!(parts.binary_widths, widths);
    }
}

#[test]
fn signed_component_rows_cli_original_deadline_and_excluded_publication_are_preserved() {
    let f = Fixture::new(0.25, false);
    let (mut parts, rows) = f.parts();
    let before = parts.problem.clone();
    assert!(f
        .attempt(
            &mut parts,
            &rows,
            true,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline")
        )
        .is_none());
    assert_eq!(parts.problem, before);
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
    // Disabling native publication still permits globally valid checked rows,
    // while neither a native token nor AY/Source receipt can be fabricated.
    let out = f
        .attempt(
            &mut parts,
            &rows,
            false,
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
    assert!(
        out.cuts
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .appended_rows
            > 0
    );
    assert!(out.aggregate.is_none() && out.continuation.is_none());
}
