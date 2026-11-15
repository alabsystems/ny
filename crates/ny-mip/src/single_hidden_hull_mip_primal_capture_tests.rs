// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual same-LP projection and attempt-owned refusal controls.
use super::*;
use std::time::Duration;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn source(extra_roles: bool) -> HullMarginProblem {
    let (w, b, c) = if extra_roles {
        (
            vec![1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 3., 2.],
            vec![0., 0., 0., 2., -1., 0.],
            vec![1., 1., -1., 1., -1., 0.],
        )
    } else {
        (vec![1., 0., 0., 1., 1., 1.], vec![0.; 3], vec![1., 1., -1.])
    };
    let h = b.len();
    let mut out = c;
    out.resize(2 * h, 0.);
    HullMarginProblem::new(
        w,
        b,
        2,
        &out,
        &[0.25, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; 2],
        vec![1.; 2],
    )
    .unwrap()
}
fn mapped(p: &HullMarginProblem, until: Instant) -> (Attempt<'_>, HullLp) {
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    let mut attempt = Attempt::new(p, until);
    attempt.admit(&roles);
    let lp = build_hull_lp_with_capture(p, &roles, &bounds, attempt.capture.as_mut()).unwrap();
    assert!(attempt.enable_same_call(lp.model.num_cols()));
    (attempt, lp)
}
fn manual_account(p: &HullMarginProblem, raw: Vec<f64>, until: Instant) -> Captured<'_> {
    let (mut attempt, _) = mapped(p, until);
    attempt.finish(Some(raw.into_boxed_slice()), &mut || true);
    attempt.take_account().unwrap().into_inner().unwrap()
}

#[test]
fn same_lp_capture_actual_original_projection_retains_fractional_negative_head() {
    let p = source(false);
    let until = end();
    let (mut attempt, lp) = mapped(&p, until);
    let roles = classify(&p, &preactivation_bounds(&p));
    let opts = SolveOpts::new()
        .with_deadline(until)
        .with_time_limit(Duration::from_secs(30));
    let mut certify = |y: &[f64]| {
        let m = multipliers_from_duals(&p, &roles, &lp, y)?;
        certify_hull_lower_bound(&p, &m)
    };
    // Exactly the production sibling seam: the returned duals and point come
    // from one float solve; no optimize/rigorous replay is requested here.
    let (result, primal) = crate::lp_propose_certify::propose_then_certify_min_with_advice(
        &lp.model,
        &lp.objective,
        &opts,
        ReplayPolicy::Skip,
        &mut certify,
    )
    .unwrap();
    assert!(result.duals_available);
    assert!(result.closed_form.unwrap() > -0.750001);
    attempt.finish(primal, &mut || true);
    let report = attempt.snapshot();
    assert!(report.admitted && report.same_call && report.decoded);
    assert_eq!(
        (report.columns, report.positive_heads, report.negative_heads),
        (7, 2, 1)
    );
    assert_eq!(report.work_units, 8880);
    let mut account = attempt.take_account().unwrap().into_inner().unwrap();
    let ledger = Arc::new(());
    let bank_end = until
        .checked_sub(Duration::from_millis(1))
        .expect("overflow when subtracting duration from instant");
    let mut debits = Vec::new();
    account
        .bank_once(&p, bank_end, &ledger, &mut |work, bytes| {
            debits.push((work, bytes));
            true
        })
        .unwrap();
    assert_eq!(debits, vec![(report.work_units, report.peak_bytes)]);
    let point = account.point(&p, bank_end, &ledger).unwrap();
    assert_eq!(
        (&*point.positive, &*point.negative),
        (&[0, 1][..], &[2][..])
    );
    assert!(point.input.iter().all(|v| v.abs() < 1e-6));
    assert!(point.values[0].abs() < 1e-6 && point.values[1].abs() < 1e-6);
    assert!((point.values[2] - 1.).abs() < 1e-6);
    let forward = (point.input[0] + point.input[1]).max(0.);
    assert!(point.values[2] > forward + 0.999);
    // Replacing this projected negative value with a forward ReLU would erase
    // the actual LP violation which the new advice channel is meant to retain.
}

#[test]
fn same_lp_capture_transformed_columns_and_stable_placeholders_have_exact_role_lists() {
    let mut p = source(true);
    // The production margin constructor rounds even exact zero downward. This
    // synthetic projection control explicitly supplies the exact inert role;
    // it does not change production rounding or claim a source certificate.
    assert!(p.c[5] < 0.);
    p.c[5] = 0.;
    assert_eq!(
        classify(&p, &preactivation_bounds(&p)),
        vec![
            Role::PosUnstable,
            Role::PosUnstable,
            Role::NegUnstable,
            Role::Live,
            Role::Dead,
            Role::Inert,
        ]
    );
    let until = end();
    // The seven ORIGINAL LP columns are x,y,s0,s1,mu2,t20,t21.
    // This finite point is intentionally untrusted; its role here is to test
    // the independent original-column projection, not establish feasibility.
    let raw = vec![0.25, -0.5, 2., 3., 0.25, 0.5, 0.25];
    let mut account = manual_account(&p, raw, until);
    let ledger = Arc::new(17u8);
    let bank_end = until
        .checked_sub(Duration::from_millis(1))
        .expect("overflow when subtracting duration from instant");
    account
        .bank_once(&p, bank_end, &ledger, &mut |_, _| true)
        .unwrap();
    let point = account.point(&p, bank_end, &ledger).unwrap();
    assert_eq!(&*point.input, &[0.25, -0.5]);
    assert_eq!(&*point.positive, &[0, 1]);
    assert_eq!(&*point.negative, &[2]);
    assert_eq!(&point.values[..3], &[2., 3., 0.75]);
    assert_eq!(&point.values[3..], &[0.; 3]);
    assert_eq!(
        (p.w_row(2)[0] * point.input[0] + p.w_row(2)[1] * point.input[1]).max(0.),
        0.
    );
    // Live/dead/inert placeholders never become partners or targets.
}

#[test]
fn same_lp_capture_malformed_missing_or_interrupted_points_keep_the_full_paid_debt() {
    let p = source(false);
    for (raw, allow) in [
        (None, true),
        (Some(vec![0.; 6]), true),
        (Some(vec![f64::NAN; 7]), true),
        (Some(vec![0.; 7]), false),
    ] {
        let (mut attempt, _) = mapped(&p, end());
        let before = attempt.snapshot();
        attempt.finish(raw.map(Vec::into_boxed_slice), &mut || allow);
        let after = attempt.snapshot();
        assert!(!after.decoded);
        assert_eq!(
            (after.work_units, after.peak_bytes),
            (before.work_units, before.peak_bytes)
        );
        let account = attempt.take_account().unwrap().into_inner().unwrap();
        assert!(!account.has_point());
        assert_eq!(
            (account.work(), account.peak_bytes()),
            (before.work_units, before.peak_bytes)
        );
        assert!(attempt.take_account().is_none());
    }
}

#[test]
fn same_lp_capture_one_shot_finish_and_metadata_admission_cannot_reissue_paid_work() {
    let p = source(false);
    let (mut attempt, _) = mapped(&p, end());
    attempt.finish(None, &mut || true);
    let before = attempt.snapshot();
    attempt.admit(&classify(&p, &preactivation_bounds(&p)));
    attempt.finish(Some(vec![0.; 7].into_boxed_slice()), &mut || true);
    assert_eq!(attempt.snapshot(), before);
    assert!(!attempt
        .take_account()
        .unwrap()
        .into_inner()
        .unwrap()
        .has_point());
    let mut malformed = Attempt::new(&p, end());
    malformed.admit(&[]);
    let refused = malformed.snapshot();
    malformed.admit(&classify(&p, &preactivation_bounds(&p)));
    assert_eq!(malformed.snapshot(), refused);
    assert_eq!(refused.work_units, 128);
    assert!(!refused.admitted && !refused.same_call && !refused.decoded);
    // A mutable report cannot erase the PRIVATE paid prefix when no capture
    // constructor was admitted. A later bank must still charge that prefix.
    {
        let mut report = malformed.report.lock().unwrap();
        report.work_units = 0;
        report.peak_bytes = 0;
    }
    malformed.finish(None, &mut || true);
    let mut account = malformed.take_account().unwrap().into_inner().unwrap();
    assert_eq!((account.work(), account.peak_bytes()), (128, 4096));
    let mut charged = None;
    account
        .bank_once(
            &p,
            Instant::now() + Duration::from_secs(1),
            &Arc::new(()),
            &mut |work, bytes| {
                charged = Some((work, bytes));
                true
            },
        )
        .unwrap();
    assert_eq!(charged, Some((128, 4096)));
}

#[test]
fn same_lp_capture_quote_source_binding_precedes_allocation_and_foreign_bank_refuses() {
    let p = source(false);
    let foreign = source(false);
    let quote = primal_advice::quote(
        &p,
        RoleCounts {
            positive: 2,
            negative: 1,
        },
    )
    .unwrap();
    let debt = quote.debt();
    assert_eq!(debt.0, 8880);
    assert!(Capture::new(&foreign, quote, end()).is_none());
    let mut same_address = source(false);
    let quote = primal_advice::quote(
        &same_address,
        RoleCounts {
            positive: 2,
            negative: 1,
        },
    )
    .unwrap();
    same_address.input_dim = 1;
    assert!(Capture::new(&same_address, quote, end()).is_none());
    let until = end();
    let mut account = manual_account(&p, vec![0.; 7], until);
    let ledger = Arc::new(());
    let mut debits = 0;
    assert!(account
        .bank_once(
            &foreign,
            until
                .checked_sub(Duration::from_millis(1))
                .expect("overflow when subtracting duration from instant"),
            &ledger,
            &mut |_, _| {
                debits += 1;
                true
            }
        )
        .is_none());
    assert_eq!(debits, 0);
    account
        .bank_once(
            &p,
            until
                .checked_sub(Duration::from_millis(1))
                .expect("overflow when subtracting duration from instant"),
            &ledger,
            &mut |_, _| {
                debits += 1;
                true
            },
        )
        .unwrap();
    assert_eq!(debits, 1);
    assert!(account
        .point(
            &foreign,
            until
                .checked_sub(Duration::from_millis(1))
                .expect("overflow when subtracting duration from instant"),
            &ledger
        )
        .is_none());
    assert!(account
        .point(
            &p,
            until
                .checked_sub(Duration::from_millis(1))
                .expect("overflow when subtracting duration from instant"),
            &Arc::new(())
        )
        .is_none());
}

#[test]
fn same_lp_capture_refused_bank_and_shorter_clock_are_irreversible() {
    let p = source(false);
    let until = end();
    let ledger = Arc::new(());
    let mut declined = manual_account(&p, vec![0.; 7], until);
    let mut debits = 0;
    assert!(declined
        .bank_once(&p, until, &ledger, &mut |_, _| {
            debits += 1;
            false
        })
        .is_none());
    assert!(declined
        .bank_once(&p, until, &ledger, &mut |_, _| {
            debits += 1;
            true
        })
        .is_none());
    assert_eq!(debits, 1);
    assert!(declined.point(&p, until, &ledger).is_none());
    let mut account = manual_account(&p, vec![0.; 7], until);
    let short = until
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    account
        .bank_once(&p, short, &ledger, &mut |_, _| true)
        .unwrap();
    assert!(account.point(&p, short, &ledger).is_some());
    assert!(account.point(&p, until, &ledger).is_none());
    assert!(account
        .bank_once(&p, short, &ledger, &mut |_, _| true)
        .is_none());
}

#[test]
fn same_lp_capture_root_none_and_expiry_keep_distinct_attempt_receipts() {
    let mut p = source(false);
    p.c[2] = 1.;
    let answer = propose_root_mip_cuts_with_signed_refinement_attempt(&p, end(), Duration::ZERO);
    assert!(answer.cuts.is_none());
    assert!(answer.primal.attempted);
    assert_eq!(answer.primal.work_units, 128);
    assert!(answer.primal.peak_bytes > 0);
    assert!(!answer.primal.admitted && !answer.primal.same_call && !answer.primal.decoded);
    let expired =
        propose_root_mip_cuts_with_signed_refinement_attempt(&p, Instant::now(), Duration::ZERO);
    assert!(expired.cuts.is_none());
    assert!(!expired.primal.attempted);
    assert_eq!(
        (expired.primal.work_units, expired.primal.peak_bytes),
        (0, 0)
    );
    assert_eq!(expired.primal.stop_reason, "preflight");
    // Finite Source and full capture admission, then the ORIGINAL LP builder
    // declines abs(c)*upper overflow before requesting any same-call solve.
    let mut overflow = source(false);
    overflow.c[2] = -f64::MAX;
    let until = end();
    let mut attempt = Attempt::new(&overflow, until);
    assert!(propose_root_mip_cuts_inner(&overflow, until, Some(&mut attempt)).is_none());
    attempt.finish(None, &mut || true);
    let refused = attempt.snapshot();
    assert!(refused.attempted && refused.admitted);
    assert!(!refused.same_call && !refused.decoded);
    assert_eq!((refused.work_units, refused.columns), (8880, 0));
    let account = attempt.take_account().unwrap().into_inner().unwrap();
    assert_eq!(
        (account.work(), account.peak_bytes()),
        (refused.work_units, refused.peak_bytes)
    );
    assert!(!account.has_point());
    let public =
        propose_root_mip_cuts_with_signed_refinement_attempt(&overflow, end(), Duration::ZERO);
    assert!(public.cuts.is_none() && public.primal.admitted && !public.primal.same_call);
    assert_eq!(public.primal.work_units, 8880);
}

#[test]
fn same_lp_capture_original_dual_only_wrapper_does_not_request_or_mint_primal_advice() {
    let p = source(false);
    let old = propose_root_mip_cuts(&p, end()).unwrap();
    assert_eq!(old.proposal_lp_calls, 1);
    assert_eq!(old.lp_primal_report(), HullLpPrimalReport::default());
    assert!(old.primal.is_none());
    let mut attempt = Attempt::new(&p, end());
    let new = propose_root_mip_cuts_inner(&p, attempt.end, Some(&mut attempt)).unwrap();
    assert_eq!(new.proposal_lp_calls, 1);
    assert_eq!(old.checked_lower.to_bits(), new.checked_lower.to_bits());
    let report = new.lp_primal_report();
    assert!(report.same_call && report.decoded && report.admitted);
    assert_eq!(report.work_units, 8880);
}

#[test]
fn same_lp_capture_actual_builder_skips_zero_weight_and_fixed_input_negative_terms() {
    let p = HullMarginProblem::new(
        vec![1., 0., 0., 1., 5., 0.],
        vec![0., -5.],
        3,
        &[1., -1., 0., 0.],
        &[0.25, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-2., 1., -3.],
        vec![2., 1., 3.],
    )
    .unwrap();
    let until = end();
    let (mut attempt, lp) = mapped(&p, until);
    // Actual columns x0,x1,x2,s0,mu1,t10: the fixed nonzero wi=5 and
    // zero wi=0 contribute NO t column. x1's bias contribution cancels -5.
    assert_eq!(lp.model.num_cols(), 6);
    let raw = vec![0.5, 1., -2., 0.75, 0.25, 0.5];
    let exact: Vec<_> = raw
        .iter()
        .map(|&x| num_rational::BigRational::from_float(x).unwrap())
        .collect();
    assert!(lp.model.check_point(&exact).is_ok());
    attempt.finish(Some(raw.into_boxed_slice()), &mut || true);
    assert!(attempt.snapshot().decoded);
    assert_eq!(
        (attempt.snapshot().columns, attempt.snapshot().work_units),
        (6, 8928)
    );
    let mut account = attempt.take_account().unwrap().into_inner().unwrap();
    let ledger = Arc::new(());
    let short = until
        .checked_sub(Duration::from_millis(1))
        .expect("overflow when subtracting duration from instant");
    account
        .bank_once(&p, short, &ledger, &mut |_, _| true)
        .unwrap();
    let point = account.point(&p, short, &ledger).unwrap();
    assert_eq!(&*point.input, &[0.5, 1., -2.]);
    assert_eq!((&*point.positive, &*point.negative), (&[0][..], &[1][..]));
    assert_eq!(point.values[0], 0.75);
    assert!((point.values[1] - 1.).abs() < 1e-12);
    assert!(point.values[1] > 0.5 + 0.499);
}
