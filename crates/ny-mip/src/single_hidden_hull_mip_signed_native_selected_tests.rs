// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Paid additive third-row publication retaining the exact original pair. The explicit
//! checked Plan and optional target/activity are advice, not LP-returned facts.
//! Source, encoder, original LP receipt, funding and publication remain real.
use super::*;
use crate::single_hidden_hull::bab::HullBabOptions;
use phase::cube::cell::input_phase_resources;

struct SelectedFixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dimensions: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
    plan: Plan,
}
impl SelectedFixture {
    const D: usize = 16;
    fn sparse_weights(h: usize) -> Vec<f64> {
        // Fifteen zero-weight inputs preserve the exact one-dimensional
        // function while meeting the existing native sparse-LP guard. A true
        // one-input model is correctly refused by that unchanged dense gate.
        let mut weights = vec![0.; Self::D * h];
        for j in 0..h {
            weights[j * Self::D] = 1.;
        }
        weights
    }
    fn input(leftover: bool) -> Self {
        let h = if leftover { 4 } else { 3 };
        let mut bias = vec![0.25, -0.25, -0.25];
        let mut output = vec![1., -1., -1.];
        let mut bounds = vec![
            Bound::new(-0.75, 1.25),
            Bound::new(-1.25, 0.75),
            Bound::new(-1.25, 0.75),
        ];
        if leftover {
            bias.push(-0.75);
            output.push(2.);
            bounds.push(Bound::new(-1.75, 0.25));
        }
        output.resize(2 * h, 0.);
        Self {
            weights: vec![Self::sparse_weights(h), output],
            biases: vec![bias, vec![0.; 2]],
            dimensions: vec![Self::D, h, 2],
            bounds: vec![bounds],
            plan: Plan {
                components: vec![Component {
                    negative: vec![1, 2],
                    positive: vec![0],
                    positive_units: vec![2],
                    g: vec![0.; Self::D],
                    alpha: vec![0., 0.625, 0.625, 1.],
                }],
                leftover: if leftover { vec![3] } else { vec![] },
            },
        }
    }
    fn later_component() -> Self {
        // The second K3 component uniquely matches the paid maximum K; no
        // activity override or component-index mutation selects it.
        let mut f = Self::input(false);
        f.weights = vec![
            Self::sparse_weights(6),
            vec![1., -0.125, -0.125, -1., -1., -1., 0., 0., 0., 0., 0., 0.],
        ];
        f.biases = vec![vec![0.25, -0.5, -0.5, -0.25, -0.25, -0.25], vec![0.; 2]];
        f.dimensions = vec![Self::D, 6, 2];
        f.bounds = vec![vec![
            Bound::new(-0.75, 1.25),
            Bound::new(-1.5, 0.5),
            Bound::new(-1.5, 0.5),
            Bound::new(-1.25, 0.75),
            Bound::new(-1.25, 0.75),
            Bound::new(-1.25, 0.75),
        ]];
        f.plan = Plan {
            components: vec![
                Component {
                    negative: vec![1, 2],
                    positive: vec![],
                    positive_units: vec![],
                    g: vec![0.; Self::D],
                    alpha: vec![],
                },
                Component {
                    negative: vec![3, 4, 5],
                    positive: vec![0],
                    positive_units: vec![2],
                    g: vec![0.; Self::D],
                    alpha: vec![0., 0.625, 0.625, 1., 0.625, 1., 1., 1.],
                },
            ],
            leftover: vec![],
        };
        f
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

// A multi-component control keeps real captured advice. For the single
// component, Some(true) supplies deterministic paid advisory target/activity,
// Some(false) drops the optional capability, and None removes only the target.
// Every mode retains the already paid debt and every native/child hold.
fn selected_owner<'p>(
    p: &'p HullMarginProblem,
    f: &SelectedFixture,
    mode: Option<bool>,
) -> HullRootMipCuts<'p> {
    selected_owner_with(p, f, mode, |cuts, b| {
        upgrade(cuts, b).unwrap();
    })
}
fn selected_owner_with<'p>(
    p: &'p HullMarginProblem,
    f: &SelectedFixture,
    mode: Option<bool>,
    before_install: impl FnOnce(&HullRootMipCuts<'_>, &Budget),
) -> HullRootMipCuts<'p> {
    selected_owner_options(p, f, mode, false, before_install)
}
fn selected_owner_options<'p>(
    p: &'p HullMarginProblem,
    f: &SelectedFixture,
    mode: Option<bool>,
    root_ticket: bool,
    before_install: impl FnOnce(&HullRootMipCuts<'_>, &Budget),
) -> HullRootMipCuts<'p> {
    selected_owner_policy(p, f, mode, root_ticket, true, before_install)
}
fn selected_owner_policy<'p>(
    p: &'p HullMarginProblem,
    f: &SelectedFixture,
    mode: Option<bool>,
    root_ticket: bool,
    pay_input: bool,
    before_install: impl FnOnce(&HullRootMipCuts<'_>, &Budget),
) -> HullRootMipCuts<'p> {
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    let mut cuts = propose_root_mip_cuts_inner(p, until, Some(&mut attempt)).unwrap();
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert!(cuts.lp_primal_report().same_call && cuts.lp_primal_report().decoded);
    let mut b = Budget::new_with_portfolio_conditions(&cuts, until, Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let cost = f.plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &f.plan, &b, b.search_end, &mut || true).unwrap();
    if root_ticket {
        root_coupling::admit(&cuts, &f.plan, cost, &mut b).unwrap();
    }
    component_rows::admit(&cuts, &f.plan, &b).unwrap();
    fund(&cuts, &b).unwrap();
    child::prepay(&cuts, &f.plan, &b).unwrap();
    assert_eq!(funding(&b).funded_calls, 1);
    if pay_input {
        input_phase_resources::prepay(&cuts, &b).unwrap();
    }
    if pay_input && f.plan.components.len() == 1 {
        let target = mode.map(|_| {
            let mut input = vec![0.; SelectedFixture::D];
            input[0] = 3. / 8.;
            input
        });
        phase::cube::configure_paid_input_for_test(
            &b,
            target,
            &[(1, 0.75), (2, 0.75)],
            mode == Some(false),
        );
    }
    before_install(&cuts, &b);
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, &f.plan, checked, cost, &b, &seal));
    cuts
}

fn append_selected(cuts: &HullRootMipCuts<'_>, f: &SelectedFixture) -> Box<MipParts> {
    let mut parts = Box::new(f.original());
    let original = parts.problem.clone();
    let report = cuts
        .append_to(parts.as_mut(), &f.encoding(), end())
        .unwrap();
    assert_eq!(report.proposal_lp_calls, 1);
    assert_eq!(
        &parts.problem.rows()[..original.num_rows()],
        original.rows()
    );
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(parts.problem.num_rows(), original.num_rows() + report.cuts);
    let stats = cuts.signed.as_ref().unwrap().stats();
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
    let pair = published(cuts);
    assert_eq!(
        pair.payload.context.parts,
        std::ptr::from_ref(parts.as_ref()) as usize
    );
    let planes = pair.planes().unwrap();
    assert_eq!(planes.len(), 2);
    for plane in planes.into_iter().chain(pair.extra_plane()) {
        let mut expected: Vec<_> = plane
            .terms()
            .iter()
            .map(|t| (t.original_column, t.coefficient_bits))
            .collect();
        expected.sort_unstable();
        assert!(
            parts.problem.rows()[original.num_rows()..]
                .iter()
                .any(|row| {
                    let mut actual: Vec<_> =
                        row.coeffs.iter().map(|&(j, a)| (j, a.to_bits())).collect();
                    actual.sort_unstable();
                    row.lb.to_bits() == plane.lower_bits
                        && row.ub == f64::INFINITY
                        && actual == expected
                }),
            "every native slot must name an actually committed row"
        );
    }
    parts
}

// Preserve row order, semantic authority, original column and every coefficient
// bit. Equal numbers at a stale advisory point are not enough for this control.
fn old_pair_bits(pair: &Published) -> Vec<(u64, Vec<(Atom, usize, u64)>)> {
    pair.planes()
        .unwrap()
        .into_iter()
        .map(|plane| {
            (
                plane.lower_bits,
                plane
                    .terms()
                    .iter()
                    .map(|t| (t.atom, t.original_column, t.coefficient_bits))
                    .collect(),
            )
        })
        .collect()
}
fn stored_plane(parts: &MipParts, plane: &Plane) -> RowSpec {
    let row = parts
        .problem
        .rows()
        .iter()
        .find(|row| {
            row.lb.to_bits() == plane.lower_bits
                && row.coeffs.len() == plane.terms().len()
                && plane.terms().iter().all(|t| {
                    row.coeffs
                        .iter()
                        .any(|&(j, a)| j == t.original_column && a.to_bits() == t.coefficient_bits)
                })
        })
        .expect("native plane is an actually appended original-IR row");
    RowSpec {
        lb: row.lb,
        ub: row.ub,
        coeffs: row.coeffs.clone(),
    }
}

#[test]
fn native_additive_cell_keeps_original_pair_and_authenticates_later_component() {
    let _serial = native_test_guard();
    let f = SelectedFixture::later_component();
    let p = f.source();
    let baseline = selected_owner_with(&p, &f, Some(true), |_, _| {});
    let old_parts = append_selected(&baseline, &f);
    let old_bits = old_pair_bits(&published(&baseline));
    let cuts = selected_owner(&p, &f, Some(true));
    let parts = append_selected(&cuts, &f);
    assert_eq!(old_pair_bits(&published(&cuts)), old_bits);
    assert_eq!(parts.problem, old_parts.problem);
    let stats = cuts.signed.as_ref().unwrap().stats();
    assert_eq!(stats.component_rows.phase.cube.cell.component, Some(1));
    let pair = published(&cuts);
    assert_eq!(pair.component(), 0);
    assert_eq!(pair.extra_component(), Some(1));
    for plane in pair.planes().unwrap() {
        let heads: Vec<_> = plane
            .terms()
            .iter()
            .filter_map(|t| match t.atom {
                Atom::Head(j) => Some(j),
                _ => None,
            })
            .collect();
        assert_eq!(heads, [1, 2]);
    }
    let heads: Vec<_> = pair
        .extra_plane()
        .unwrap()
        .terms()
        .iter()
        .filter_map(|t| match t.atom {
            Atom::Head(j) => Some(j),
            _ => None,
        })
        .collect();
    assert_eq!(heads, [0, 3, 4, 5]);
    assert_eq!(cuts.native_funding_report().funded_calls, 1);
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
}

#[test]
fn native_additive_missing_cell_keeps_original_two_plane_fallback() {
    let _serial = native_test_guard();
    let f = SelectedFixture::later_component();
    let p = f.source();
    let cuts = selected_owner(&p, &f, Some(true));
    let owner = cuts.signed.as_ref().unwrap();
    let funding_before = cuts.native_funding_report();
    let debt = owner.stats();
    phase::cube::cell::omit_after_payment_for_test(&owner.budget);
    let _parts = append_selected(&cuts, &f);
    let pair = published(&cuts);
    assert_eq!(pair.component(), 0);
    assert!(pair.extra_plane().is_none());
    assert!(owner
        .stats()
        .component_rows
        .phase
        .cube
        .cell
        .component
        .is_none());
    for plane in pair.planes().unwrap() {
        let heads: Vec<_> = plane
            .terms()
            .iter()
            .filter_map(|t| match t.atom {
                Atom::Head(j) => Some(j),
                _ => None,
            })
            .collect();
        assert_eq!(heads, [1, 2]);
    }
    assert_eq!(cuts.native_funding_report(), funding_before);
    assert!(owner.stats().work_units >= debt.work_units);
    assert_eq!(owner.stats().admitted_bytes, debt.admitted_bytes);
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
}

#[test]
fn native_additive_input_support_keeps_exact_nonzero_slope_and_one_three_row_call() {
    let _serial = native_test_guard();
    // Preserve the measured dense refusal as a positive fallback control;
    // the genuine sparse fixture must still append exactly three native rows.
    for (leftover, expected_added) in [(false, 2), (true, 3)] {
        let f = SelectedFixture::input(leftover);
        let p = f.source();
        let cuts = selected_owner(&p, &f, Some(true));
        let _parts = append_selected(&cuts, &f);
        let owner = cuts.signed.as_ref().unwrap();
        let stats = owner.stats();
        assert!(stats
            .component_rows
            .phase
            .cube
            .cell
            .input_phase
            .selected_candidate
            .is_some());
        assert!(owner.plan.components[0].g.iter().all(|g| *g == 0.));
        let pair = published(&cuts);
        let planes = pair.planes().unwrap();
        assert!(!planes[0]
            .terms()
            .iter()
            .any(|t| matches!(t.atom, Atom::Input(_))));
        assert!(planes[1]
            .terms()
            .iter()
            .all(|t| !matches!(t.atom, Atom::Input(_))));
        let slopes: Vec<_> = pair
            .extra_plane()
            .unwrap()
            .terms()
            .iter()
            .filter_map(|t| match t.semantic() {
                (Atom::Input(i), a) => Some((i, a)),
                _ => None,
            })
            .collect();
        assert_eq!(slopes.len(), 1);
        assert_eq!(slopes[0].0, 0);
        assert!(slopes[0].1.is_finite() && slopes[0].1 != 0.);
        let funding_before = cuts.native_funding_report();
        let token = bind(owner, &cuts, pair.end(), &mut || true).unwrap();
        let bounds = preactivation_bounds(&p);
        let roles = classify(&p, &bounds);
        let mut calls = 0;
        for attempt in 0..2 {
            let mut lp = build_hull_lp(&p, &roles, &bounds).unwrap();
            let rows = lp.model.num_rows();
            let columns = lp.model.num_cols();
            let nonzeros: usize = (0..rows)
                .map(|i| {
                    let row = lp.model.row_at(i).unwrap();
                    lp.model.row(row).0.len()
                })
                .sum();
            // These are actual original Hull LP dimensions, not fabricated count
            // receipts. The extra positive leftover changes only this outer LP:
            // component geometry and its paid nonzero input slope are unchanged.
            assert_eq!(
                (columns, rows, nonzeros),
                if leftover { (22, 4, 10) } else { (21, 3, 8) }
            );
            let caps = &pair.payload.context.caps;
            assert_eq!(caps.row_capacity, 3);
            let expanded = |n: usize| (caps.expanded_pair / caps.row_capacity * n).min(columns * n);
            assert_eq!((expanded(2), expanded(3)), (2 * columns, 3 * columns));
            assert!((rows + 3) * columns <= (1 << 20));
            // The original conservative density guard must retain the old pair
            // in BOTH cases, and keep the third only in the sparse true case.
            assert!(2 * (nonzeros + expanded(2)) < (rows + 2) * columns);
            assert_eq!(
                2 * (nonzeros + expanded(3)) >= (rows + 3) * columns,
                !leftover
            );
            let result = token.with_node(
                &p,
                &[],
                &mut lp,
                &[],
                pair.end(),
                &mut || true,
                |_, _| None,
                |actual, certify| {
                    calls += 1;
                    if leftover {
                        assert_eq!(actual.model.num_rows(), rows + 3);
                    } else {
                        assert_eq!(actual.model.num_rows(), rows + 2);
                    }
                    assert_eq!(actual.model.num_rows(), rows + expected_added);
                    let _ = certify(&vec![0.; actual.model.num_rows()]);
                },
            );
            if attempt == 0 {
                assert!(matches!(result, bank::Visit::Committed(Some(()))));
            } else {
                assert!(matches!(result, bank::Visit::Declined));
                assert_eq!(lp.model.num_rows(), rows);
            }
        }
        assert_eq!(calls, 1);
        assert_eq!(cuts.native_funding_report(), funding_before);
        assert_eq!(cuts.proposal_lp_calls, 1); // the callback above is a hook stand-in
        assert_eq!(cuts.native_rows_report().committed, 1);
    }
}

#[test]
fn native_additive_input_capture_rejects_wrong_missing_and_nonfinite_slope() {
    let _serial = native_test_guard();
    let f = SelectedFixture::input(false);
    let p = f.source();
    let cuts = selected_owner(&p, &f, Some(true));
    let mut parts = append_selected(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let plane = pair.extra_plane().unwrap();
    let row = parts
        .problem
        .rows()
        .iter()
        .find(|row| {
            row.lb.to_bits() == plane.lower_bits
                && row.coeffs.len() == plane.terms().len()
                && plane.terms().iter().all(|t| {
                    row.coeffs
                        .iter()
                        .any(|&(j, a)| j == t.original_column && a.to_bits() == t.coefficient_bits)
                })
        })
        .unwrap();
    let stored = RowSpec {
        lb: row.lb,
        ub: row.ub,
        coeffs: row.coeffs.clone(),
    };
    // The ordinary global signed row has the same head set but unit
    // coefficients and head-index order. The native producer's component
    // base uses doubled allocated coefficients, negatives then positives.
    // Match that complete stored prefix, including order and coefficient bits.
    let component = &owner.plan.components[pair.component()];
    let head_count = component.negative.len() + component.positive.len();
    let head_terms: Vec<_> = stored.coeffs[..head_count]
        .iter()
        .map(|&(j, a)| (j, a.to_bits()))
        .collect();
    let base = parts.problem.rows()[pair.payload.context.original_rows..]
        .iter()
        .find(|row| {
            row.coeffs.len() == head_terms.len()
                && row
                    .coeffs
                    .iter()
                    .map(|&(j, a)| (j, a.to_bits()))
                    .eq(head_terms.iter().copied())
        })
        .expect("actual appended component base without selectors")
        .clone();
    let input = plane
        .terms()
        .iter()
        .find(|t| t.atom == Atom::Input(0))
        .unwrap();
    let mut g = vec![0.; SelectedFixture::D];
    g[0] = -0.5 * f64::from_bits(input.coefficient_bits);
    assert_ne!(g[0], 0.);
    let mut selectors = [0; 8];
    let mut n = 0;
    for &head in &owner.plan.components[0].negative {
        if plane.terms().iter().any(|t| t.atom == Atom::Selector(head)) {
            selectors[n] = head;
            n += 1;
        }
    }
    // Preserve the genuine Context's stable Parts address and restore the
    // original model for a fresh complete binding. Only these adversarial
    // descriptors are test-created; no Admission/Published/Source is forged.
    *parts = f.original();
    binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
        let inspect = |s: &RowSpec, advice: Option<&[f64]>| {
            let origin = phase::NativeRow::for_extra_test(0, selectors, n, s, advice);
            capture(
                owner,
                &cuts,
                &parts,
                binding,
                &origin,
                &pair.payload.context,
                &mut || true,
            )
        };
        assert!(inspect(&stored, Some(&g)).is_some());
        assert!(inspect(&stored, None).is_none());
        assert!(inspect(&stored, Some(&[])).is_none());
        let mut wrong = g.clone();
        wrong[0] = f64::NAN;
        assert!(inspect(&stored, Some(&wrong)).is_none());
        wrong[0] = -g[0];
        assert!(inspect(&stored, Some(&wrong)).is_none());
        // A genuinely emitted selector-free base row remains representable in
        // the additive slot. This adversarial descriptor does not publish it or
        // manufacture a production license; it checks capture's numeric shape.
        let zero = phase::NativeRow::for_extra_test(0, [0; 8], 0, &base, None);
        assert!(capture(
            owner,
            &cuts,
            &parts,
            binding,
            &zero,
            &pair.payload.context,
            &mut || true
        )
        .is_some());
        let mut changed = stored.clone();
        let term = changed
            .coeffs
            .iter_mut()
            .find(|(j, _)| *j == input.original_column)
            .unwrap();
        term.1 = f64::from_bits(term.1.to_bits() ^ 1);
        assert!(inspect(&changed, Some(&g)).is_none());
        changed.coeffs.retain(|(j, _)| *j != input.original_column);
        assert!(inspect(&changed, Some(&g)).is_none());
        let origin = phase::NativeRow::for_extra_test(0, selectors, n, &stored, Some(&g));
        assert!(capture(
            owner,
            &cuts,
            &parts,
            binding,
            &origin,
            &pair.payload.context,
            &mut || false
        )
        .is_none());
    })
    .unwrap();
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
}

#[test]
fn native_additive_missing_paid_target_keeps_old_pair_without_input_slope() {
    let _serial = native_test_guard();
    let f = SelectedFixture::input(false);
    let p = f.source();
    let cuts = selected_owner(&p, &f, None);
    let before = cuts.native_funding_report();
    let _parts = append_selected(&cuts, &f);
    let stats = cuts.signed.as_ref().unwrap().stats();
    let r = &stats.component_rows.phase.cube.cell.input_phase;
    assert!(r.admitted && r.selected_candidate.is_none());
    let pair = published(&cuts);
    assert!(pair.planes().unwrap().iter().all(|plane| !plane
        .terms()
        .iter()
        .any(|t| matches!(t.atom, Atom::Input(_)))));
    assert!(pair.extra_plane().is_none_or(|plane| !plane
        .terms()
        .iter()
        .any(|t| matches!(t.atom, Atom::Input(_)))));
    assert_eq!(cuts.native_funding_report(), before);
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
}

#[test]
fn native_additive_input_staging_cancel_rolls_back_ir_without_reopening_credit() {
    let _serial = native_test_guard();
    let f = SelectedFixture::input(false);
    let p = f.source();
    let cuts = selected_owner(&p, &f, Some(true));
    let owner = cuts.signed.as_ref().unwrap();
    let funding_before = cuts.native_funding_report();
    let mut parts = Box::new(f.original());
    let prefix = parts.problem.clone();
    let mut stopped = false;
    let result = cuts.append_with_poll(&mut parts, &f.encoding(), end(), &mut || {
        let counts = owner.budget.state.lock().unwrap();
        stopped |= counts.native_rows.pending.as_ref().is_some_and(|p| {
            p.extra.as_ref().is_some_and(|plane| {
                plane
                    .terms()
                    .iter()
                    .any(|t| matches!(t.atom, Atom::Input(_)))
            })
        });
        !stopped
    });
    assert!(stopped && result.is_none());
    assert_eq!(parts.problem, prefix);
    let counts = owner.budget.state.lock().unwrap();
    assert!(
        counts.native_rows.begun
            && counts.native_rows.pending.is_none()
            && counts.native_rows.published.is_none()
    );
    drop(counts);
    assert_eq!(cuts.native_funding_report(), funding_before);
    assert!(bind(owner, &cuts, owner.budget.authority, &mut || true).is_none());
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
}

#[test]
fn native_additive_input_support_improves_actual_original_root_lp() {
    let _serial = native_test_guard();
    let f = SelectedFixture::input(true);
    let p = f.source();
    let mut after = Vec::new();
    let mut original_pair = None;
    let mut original_ir = None;
    let mut original_floor = None;
    for enabled in [false, true] {
        // Identical checked Plan/input support/global rows. Only the native
        // additive capability differs; both original native planes survive.
        let cuts = selected_owner_with(&p, &f, Some(true), |cuts, b| {
            if enabled {
                upgrade(cuts, b).unwrap();
            }
        });
        let parts = append_selected(&cuts, &f);
        let pair = published(&cuts);
        let bits = old_pair_bits(&pair);
        if enabled {
            assert_eq!(Some(bits), original_pair);
            assert_eq!(Some(parts.problem.clone()), original_ir);
            assert!(pair.extra_plane().is_some());
        } else {
            original_pair = Some(bits);
            original_ir = Some(parts.problem.clone());
            assert!(pair.extra_plane().is_none());
        }
        assert!(cuts.best_signed_root_proof(end()).is_none());
        assert!(cuts.signed_child_cache.get().is_some());
        let result = cuts
            .continue_hull_bab_with_native_rows(
                end(),
                HullBabOptions {
                    workers: 1,
                    max_nodes: 1,
                },
            )
            .unwrap();
        let tree = result.report();
        assert_eq!(
            (tree.nodes, tree.root_lp_calls, tree.proposal_lp_calls),
            (1, 1, 1)
        );
        let before = tree.root_before_native.unwrap();
        if enabled {
            assert_eq!(Some(before.to_bits()), original_floor);
        } else {
            original_floor = Some(before.to_bits());
        }
        let bound = tree.root_certified.unwrap().max(before);
        assert!(bound.is_finite() && bound >= before);
        // Original graph x=-1 is feasible with exact margin zero.
        assert!(bound <= 0.);
        assert_eq!(cuts.native_rows_report().committed, 1);
        assert_eq!(cuts.native_funding_report().funded_calls, 1);
        assert_eq!(cuts.proposal_lp_calls, 1);
        eprintln!(
            "[native-additive-input-control] enabled={enabled} before={before:?} after={bound:?}"
        );
        after.push(bound);
    }
    assert!(
        after[1] > after[0] + 1e-8,
        "additive support did not improve the actual certified root floor: {after:?}"
    );
}

// These fill already-accounted ordinary ledger consumption synthetically. The
// native receipt, Source, checked Plan, child credits and input capability are
// real. They test exact admission arithmetic, not a measured runtime layout.
fn upgrade_edge(work_edge: bool, excess: usize) {
    let f = SelectedFixture::input(false);
    let p = f.source();
    let _cuts = selected_owner_with(&p, &f, Some(true), |cuts, b| {
        let (delta_work, delta_bytes) = bank::upgrade_quote(cuts, b).unwrap();
        assert!(delta_work > 1024 && delta_bytes > 992);
        let base_bytes = b.admitted_bytes;
        let before_funding = funding(b);
        let holds;
        let before_work;
        let before_component_bytes;
        {
            let mut c = b.state.lock().unwrap();
            holds = (
                c.reserved,
                c.component_reserved,
                c.prepaid_collection_work,
                c.prepaid_bind_work,
                c.child.cache_bytes,
            );
            assert!(holds.2 > 0 && holds.3 > 0 && holds.4 > 0);
            if work_edge {
                let padded = MAX_WORK - c.reserved - c.component_reserved - delta_work + excess;
                assert!(padded >= c.work);
                c.work = padded;
            } else {
                let total = b.admitted_bytes
                    + c.shared_bytes
                    + c.direction.cache_bytes
                    + c.child.cache_bytes
                    + c.component_rows.admitted_bytes;
                let padding = MAX_BYTES - delta_bytes + excess - total;
                c.component_rows.admitted_bytes += padding;
            }
            before_work = c.work;
            before_component_bytes = c.component_rows.admitted_bytes;
        }
        let admitted = upgrade(cuts, b).is_some();
        assert_eq!(admitted, excess == 0);
        assert_eq!(b.admitted_bytes, base_bytes);
        let after_funding = funding(b);
        assert_eq!(after_funding.funded_calls, 1);
        assert_eq!(
            after_funding.setup_work_units,
            before_funding.setup_work_units + 1024
        );
        {
            let c = b.state.lock().unwrap();
            assert_eq!(
                (
                    c.reserved,
                    c.component_reserved,
                    c.prepaid_collection_work,
                    c.prepaid_bind_work,
                    c.child.cache_bytes
                ),
                holds
            );
            assert_eq!((c.replay_attempts, c.source_replays), (0, 0));
            assert_eq!(
                c.work,
                before_work + if admitted { delta_work } else { 1024 }
            );
            assert_eq!(
                c.component_rows.admitted_bytes,
                before_component_bytes + if admitted { delta_bytes } else { 0 }
            );
            assert_eq!(
                c.native_rows.admission.as_ref().unwrap().caps.row_capacity,
                if admitted { 3 } else { 2 }
            );
            if admitted && work_edge {
                assert_eq!(c.work + c.reserved + c.component_reserved, MAX_WORK);
            }
            if admitted && !work_edge {
                assert_eq!(
                    b.admitted_bytes
                        + c.shared_bytes
                        + c.direction.cache_bytes
                        + c.child.cache_bytes
                        + c.component_rows.admitted_bytes,
                    MAX_BYTES
                );
            }
        }
        // Both success and refusal burn the only upgrade attempt. Refusal may
        // spend its bounded setup, but cannot steal child/Source credit.
        assert!(upgrade(cuts, b).is_none());
        assert_eq!(funding(b), after_funding);
    });
}

#[test]
fn native_additive_upgrade_exact_work_and_one_unit_refusal() {
    let _serial = native_test_guard();
    upgrade_edge(true, 0);
    upgrade_edge(true, 1);
}

#[test]
fn native_additive_upgrade_exact_bytes_and_one_byte_refusal() {
    let _serial = native_test_guard();
    upgrade_edge(false, 0);
    upgrade_edge(false, 1);
}

#[test]
fn native_additive_upgrade_992_byte_refusal_keeps_exact_old_pair() {
    let _serial = native_test_guard();
    let f = SelectedFixture::input(false);
    let p = f.source();
    let old = selected_owner_with(&p, &f, Some(true), |_, _| {});
    let old_parts = append_selected(&old, &f);
    let cuts = selected_owner_with(&p, &f, Some(true), |cuts, b| {
        let (_, delta_bytes) = bank::upgrade_quote(cuts, b).unwrap();
        assert!(delta_bytes > 992);
        let before = funding(b);
        {
            let mut c = b.state.lock().unwrap();
            let bytes = b.admitted_bytes
                + c.shared_bytes
                + c.direction.cache_bytes
                + c.child.cache_bytes
                + c.component_rows.admitted_bytes;
            c.component_rows.admitted_bytes += MAX_BYTES - 992 - bytes;
        }
        assert!(upgrade(cuts, b).is_none());
        assert_eq!(funding(b).funded_calls, before.funded_calls);
        assert_eq!(funding(b).admitted_bytes, before.admitted_bytes);
        assert_eq!(funding(b).setup_work_units, before.setup_work_units + 1024);
    });
    let parts = append_selected(&cuts, &f);
    let pair = published(&cuts);
    assert_eq!(old_pair_bits(&pair), old_pair_bits(&published(&old)));
    assert!(pair.extra_plane().is_none());
    assert_eq!(parts.problem, old_parts.problem);
    assert_eq!(cuts.native_funding_report().funded_calls, 1);
}

#[test]
fn native_additive_upgrade_preserves_real_root_ticket_base_identity() {
    let _serial = native_test_guard();
    let f = SelectedFixture::later_component();
    let p = f.source();
    let _cuts = selected_owner_options(&p, &f, Some(true), true, |cuts, b| {
        let ticket = b.root_coupling.as_ref().unwrap();
        assert!(ticket.owned(cuts, b));
        let base = b.admitted_bytes;
        upgrade(cuts, b).unwrap();
        assert_eq!(b.admitted_bytes, base);
        assert!(ticket.owned(cuts, b));
        assert_eq!(funding(b).funded_calls, 1);
    });
}

#[test]
fn native_additive_extra_component_descriptor_must_match_authenticated_source() {
    let _serial = native_test_guard();
    let f = SelectedFixture::later_component();
    let p = f.source();
    let cuts = selected_owner(&p, &f, Some(true));
    let mut parts = append_selected(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let plane = pair.extra_plane().unwrap();
    let stored = stored_plane(&parts, plane);
    let mut g = vec![0.; SelectedFixture::D];
    for t in plane.terms() {
        if let Atom::Input(i) = t.atom {
            g[i] = -0.5 * f64::from_bits(t.coefficient_bits);
        }
    }
    let mut selectors = [0; 8];
    let mut n = 0;
    for &head in &owner.plan.components[1].negative {
        if plane.terms().iter().any(|t| t.atom == Atom::Selector(head)) {
            selectors[n] = head;
            n += 1;
        }
    }
    *parts = f.original();
    binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
        for component in [1, 0, 2, usize::MAX] {
            let origin =
                phase::NativeRow::for_extra_test(component, selectors, n, &stored, Some(&g));
            assert_eq!(
                capture(
                    owner,
                    &cuts,
                    &parts,
                    binding,
                    &origin,
                    &pair.payload.context,
                    &mut || true
                )
                .is_some(),
                component == 1
            );
        }
        let old_slot = phase::NativeRow::for_test(1, true, selectors, n, &stored, Some(&g));
        assert!(capture(
            owner,
            &cuts,
            &parts,
            binding,
            &old_slot,
            &pair.payload.context,
            &mut || true
        )
        .is_none());
    })
    .unwrap();
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
}

#[test]
fn native_additive_full_signature_tamper_burns_handoff_without_native_call() {
    let _serial = native_test_guard();
    let f = SelectedFixture::input(false);
    let p = f.source();
    let mut cuts = selected_owner(&p, &f, Some(true));
    let _parts = append_selected(&cuts, &f);
    let pair = published(&cuts);
    assert!(pair.extra_plane().is_some());
    let before = cuts.native_funding_report();
    let old = cuts.multipliers.alpha[0];
    cuts.multipliers.alpha[0] = old + 0.25;
    let owner = cuts.signed.as_ref().unwrap();
    assert!(bind(owner, &cuts, pair.end(), &mut || true).is_none());
    cuts.multipliers.alpha[0] = old;
    let owner = cuts.signed.as_ref().unwrap();
    assert!(bind(owner, &cuts, pair.end(), &mut || true).is_none());
    assert_eq!(cuts.native_funding_report(), before);
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
}

#[test]
fn native_additive_cell_capacity_without_input_phase_pays_third_row_or_retains_pair() {
    let _serial = native_test_guard();
    let f = SelectedFixture::later_component();
    let p = f.source();
    let baseline = selected_owner_policy(&p, &f, None, false, false, |_, _| {});
    let baseline_parts = append_selected(&baseline, &f);
    let baseline_pair = old_pair_bits(&published(&baseline));
    for refuse in [false, true] {
        let cuts = selected_owner_policy(&p, &f, None, false, false, |cuts, b| {
            let mut c = b.state.lock().unwrap();
            assert!(c.component_rows.phase.cube.cell.admitted);
            assert!(!c.component_rows.phase.cube.cell.input_phase.admitted);
            let holds = (
                c.reserved,
                c.component_reserved,
                c.prepaid_collection_work,
                c.prepaid_bind_work,
                c.child.cache_bytes,
            );
            let base = b.admitted_bytes;
            if refuse {
                let used = base
                    + c.shared_bytes
                    + c.direction.cache_bytes
                    + c.component_rows.admitted_bytes
                    + c.child.cache_bytes;
                c.component_rows.admitted_bytes += MAX_BYTES - used;
            }
            drop(c);
            assert_eq!(upgrade(cuts, b).is_some(), !refuse);
            let c = b.state.lock().unwrap();
            assert!(!c.component_rows.phase.cube.cell.input_phase.admitted);
            assert_eq!(b.admitted_bytes, base);
            assert_eq!(
                (
                    c.reserved,
                    c.component_reserved,
                    c.prepaid_collection_work,
                    c.prepaid_bind_work,
                    c.child.cache_bytes
                ),
                holds
            );
            assert_eq!(
                c.native_rows.admission.as_ref().unwrap().caps.row_capacity,
                if refuse { 2 } else { 3 }
            );
        });
        let parts = append_selected(&cuts, &f);
        let pair = published(&cuts);
        assert_eq!(old_pair_bits(&pair), baseline_pair);
        assert_eq!(parts.problem, baseline_parts.problem);
        assert_eq!(pair.extra_plane().is_some(), !refuse);
        assert_eq!(cuts.native_funding_report().funded_calls, 1);
    }
}
