// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored Source routing control, not a latency or competition measurement.
use super::*;

#[test]
fn signed_default_policy_uses_production_root_clock_without_any_advice() {
    for (bias, safe) in [(0.5f32, true), (-0.5f32, false)] {
        let policy = HeadGroupPolicy::new(5.0, 2, 4).unwrap();
        assert!(policy.directional_cuts());
        assert!(policy.phase_cover().is_none());

        // Invoke the ordinary verifier rather than refine_with_prefix directly.
        // Its unchanged implementation imposes the real 750ms root-stage cap;
        // the outer allowance only permits source preparation and publication.
        let outcome = verify_with_provider(
            &dense_spec(bias),
            &[-1.0; 65],
            &[1.0; 65],
            0,
            1,
            policy,
            end(),
            || None,
        );
        let report = match outcome {
            InputHeadOutcome::Unsat(report, token) if safe => {
                assert_eq!(token.scope(), TokenScope::Instance);
                assert_eq!(token.classes(), &[1]);
                assert!(token.covers_all(&[1]));
                assert!(!token.covers_all(&[0, 1]));
                report
            }
            InputHeadOutcome::Unknown(report) if !safe => report,
            InputHeadOutcome::Unknown(report) => {
                panic!("safe default-on signed Source control stayed open: {report:?}")
            }
            InputHeadOutcome::Unsat(report, _) => {
                panic!("unsafe default-on signed Source control certified: {report:?}")
            }
        };
        let lower = report.lower.expect("completed original-margin replay");
        if safe {
            assert!(lower > 0.45 && lower <= 0.5, "{report:?}");
        } else {
            // At x=0 all original hidden values are zero and the margin is -0.5.
            assert!(lower <= -0.5, "{report:?}");
        }
        assert!(
            (1..=MAX_CUTS * MAX_ROUNDS).contains(&report.directional_cuts),
            "{report:?}"
        );
        assert!((1..=MAX_ROUNDS).contains(&report.directional_rounds));
        assert!((1..=MAX_ATTEMPTS).contains(&report.directional_attempted));
        assert!(
            (1..=report.directional_rounds).contains(&report.directional_native_replayed),
            "{report:?}"
        );
        assert!((256..=MAX_MASKS).contains(&report.directional_masks));
        assert!((1..=MAX_CELLS).contains(&report.directional_work_cells));

        // Missing providers are not helper executions or imported LP evidence.
        assert_eq!(report.advice_calls, 0);
        assert_eq!(report.triangle_advice_calls, 0);
        assert_eq!(report.directional_advice_calls, 0);
        assert_eq!(report.directional_ay_replayed, 0);
        assert_eq!(report.phase_cover_advice_calls, 0);
        assert_eq!(report.phase_cover_replayed_leaves, 0);
    }
}
