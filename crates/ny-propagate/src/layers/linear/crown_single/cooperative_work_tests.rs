// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::amortise::{CooperativeWork, NoPoll, WeightedPoll, POLL_CHUNK_UNITS};
use std::time::Duration;

fn fixture(with_error: bool, overflow: bool) -> (LinearLayer, LinearBounds) {
    let weight = Array2::from_shape_fn((33, 45), |(i, j)| {
        ((i * 13 + j * 7) % 19) as f32 * 0.125 - 1.0
    });
    let bias = Array1::from_shape_fn(33, |i| (i % 7) as f32 * 0.25 - 0.5);
    let layer = LinearLayer::new(weight, Some(bias)).unwrap();
    let lower = Array2::from_shape_fn((70, 33), |(i, j)| {
        if overflow && i == 3 {
            1.0e20
        } else {
            ((i * 5 + j * 11) % 23) as f32 * 0.125 - 1.25
        }
    });
    let upper = -&lower;
    let lower_b = Array1::from_shape_fn(70, |i| i as f32 * 0.0625);
    let upper_b = -&lower_b;
    let bounds = if with_error {
        LinearBounds::new_or_conservative_with_err(
            lower,
            lower_b,
            upper,
            upper_b,
            Array2::from_elem((70, 33), 0.03125),
            Array2::from_elem((70, 33), 0.0625),
        )
        .unwrap()
    } else {
        LinearBounds::new(lower, lower_b, upper, upper_b).unwrap()
    };
    (layer, bounds)
}

fn bits(bounds: &LinearBounds) -> Vec<u32> {
    let mut result = Vec::new();
    result.extend(bounds.lower_a().iter().map(|v| v.to_bits()));
    result.extend(bounds.upper_a().iter().map(|v| v.to_bits()));
    result.extend(bounds.lower_b().iter().map(|v| v.to_bits()));
    result.extend(bounds.upper_b().iter().map(|v| v.to_bits()));
    for error in [bounds.lower_a_err(), bounds.upper_a_err()] {
        result.push(u32::from(error.is_some()));
        if let Some(error) = error {
            result.extend(error.iter().map(|v| v.to_bits()));
        }
    }
    result
}

#[test]
fn shared_cpu_work_hooks_preserve_all_bound_bits_and_overflow_rows() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        assert!(
            rayon::current_thread_index().is_some(),
            "must exercise actual Rayon worker dispatch"
        );
        for nested in [false, true] {
            let _guard = nested.then(crate::faer_parallelism::NestedFaerParGuard::new);
            assert_eq!(
                matches!(crate::faer_parallelism::current_par(), faer::Par::Seq),
                !nested,
                "guard presence must actually switch sequential versus nested faer dispatch"
            );
            for with_error in [false, true] {
                for overflow in [false, true] {
                    let (layer, input) = fixture(with_error, overflow);
                    let original = bits(&input);
                    let reference = propagate_linear_cpu_with_deadline_stack_and_work(
                        &layer,
                        &input,
                        None,
                        false,
                        false,
                        &mut NoPoll,
                    )
                    .unwrap();
                    for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
                        let mut units = 0;
                        let mut poll = WeightedPoll::new(|n| {
                            units += n;
                            check_optional_aw_deadline(deadline, "cooperative CPU parity test")
                        });
                        let actual = propagate_linear_cpu_with_deadline_stack_and_work(
                            &layer, &input, deadline, false, false, &mut poll,
                        )
                        .unwrap();
                        assert_eq!(bits(&actual), bits(&reference));
                        assert!(units >= 70 * 45 * 128);
                        if overflow {
                            assert!(actual.lower_a().row(3).iter().all(|&v| v == 0.0));
                            assert!(actual.upper_a().row(3).iter().all(|&v| v == 0.0));
                            assert_eq!(actual.lower_b()[3], f32::NEG_INFINITY);
                            assert_eq!(actual.upper_b()[3], f32::INFINITY);
                        }
                        assert_eq!(bits(&input), original);
                    }
                }
            }
        }
    });
}

#[test]
fn shared_cpu_midwork_refusal_discards_the_result_and_preserves_original_input() {
    let (layer, input) = fixture(true, false);
    let original = bits(&input);
    for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
        let mut delivered = 0;
        let mut poll = WeightedPoll::new(|units| {
            delivered += units;
            if units > 0 {
                Err(NyError::DeadlineExceeded(
                    "CPU loop forced expiry after real publication work".into(),
                ))
            } else {
                Ok(())
            }
        });
        let error = propagate_linear_cpu_with_deadline_stack_and_work(
            &layer, &input, deadline, false, false, &mut poll,
        )
        .unwrap_err();
        assert!(error.is_deadline_exceeded());
        assert!(delivered > 0 && delivered <= POLL_CHUNK_UNITS);
        assert_eq!(bits(&input), original);
    }
}

#[test]
fn shared_cpu_checks_existing_authority_after_the_final_bounds_firewall() {
    struct Boundary {
        calls: usize,
        fail: Option<usize>,
    }
    impl CooperativeWork for Boundary {
        fn before(&mut self, _: usize) -> Result<()> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<()> {
            self.calls += 1;
            if self.fail == Some(self.calls) {
                Err(NyError::DeadlineExceeded(
                    "CPU final authority refused".into(),
                ))
            } else {
                Ok(())
            }
        }
    }
    let (layer, input) = fixture(true, false);
    let original = bits(&input);
    let mut observed = Boundary {
        calls: 0,
        fail: None,
    };
    propagate_linear_cpu_with_deadline_stack_and_work(
        &layer,
        &input,
        None,
        false,
        false,
        &mut observed,
    )
    .unwrap();
    // The all-position incoming-error bias sum has an entry and final check
    // for EACH row of EACH present lower/upper plane. Metadata/term pre-debits
    // do not call this fixture's checkpoint method. Keep the refusal at the
    // actual FINAL firewall checkpoint rather than the old sixth boundary.
    let final_boundary = 6 + 2 * input.num_outputs() * 2;
    assert_eq!(
        observed.calls, final_boundary,
        "six original outer checks plus two checks per incoming-bias row/side"
    );
    let mut refused = Boundary {
        calls: 0,
        fail: Some(final_boundary),
    };
    assert!(propagate_linear_cpu_with_deadline_stack_and_work(
        &layer,
        &input,
        None,
        false,
        false,
        &mut refused
    )
    .unwrap_err()
    .is_deadline_exceeded());
    assert_eq!(refused.calls, final_boundary);
    assert_eq!(bits(&input), original);
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(propagate_linear_cpu_with_deadline_stack_and_work(
        &layer,
        &input,
        Some(past),
        false,
        false,
        &mut NoPoll
    )
    .unwrap_err()
    .is_deadline_exceeded());
}
