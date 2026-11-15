// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::amortise::{WeightedPoll, POLL_CHUNK_UNITS};
use ndarray::s;
use ny_core::NyError;
use std::cell::Cell;
use std::time::{Duration, Instant};

fn observe(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        Err(NyError::DeadlineExceeded("original bias authority".into()))
    } else {
        Ok(())
    }
}

#[test]
fn cooperative_bias_keeps_ordered_accumulation_bits_in_both_deadline_arms() {
    let values = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        1.0e20,
        -1.0e20,
        3.0,
        -7.0,
        f32::MIN_POSITIVE,
    ];
    let bias = Array1::from_shape_fn(513, |i| values[i % values.len()]);
    let block = BiasBlockParams {
        num_outputs: 19,
        out_features: bias.len(),
        col_offset: 7,
    };
    let lower = |i: usize, j: usize| values[(i * 13 + j) % values.len()];
    let upper = |i: usize, j: usize| values[(i + j * 5) % values.len()];
    let mut expected_l = [0.0; 19];
    let mut expected_u = [-0.0; 19];
    // Frozen historical ordered reference, independent of the new adapter.
    for i in 0..19 {
        for j in 0..bias.len() {
            let b = f32_to_f64_exact(bias[j]);
            expected_l[i] = add_f64_down(expected_l[i], f32_to_f64_exact(lower(i, 7 + j)) * b);
            expected_u[i] = add_f64_up(expected_u[i], f32_to_f64_exact(upper(i, 7 + j)) * b);
        }
    }
    for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
        let mut actual_l = vec![0.0; 19];
        let mut actual_u = vec![-0.0; 19];
        let mut units = 0;
        let mut calls = 0;
        let mut poll = WeightedPoll::new(|n| {
            units += n;
            calls += 1;
            observe(deadline)
        });
        accumulate_bias_f64_with_work(
            &mut (&mut actual_l, &mut actual_u),
            lower,
            upper,
            &bias,
            &block,
            &mut poll,
        )
        .unwrap();
        assert_eq!(units, 19 * 513 * 32);
        assert!(calls > 2);
        for i in 0..19 {
            assert_eq!(actual_l[i].to_bits(), expected_l[i].to_bits());
            assert_eq!(actual_u[i].to_bits(), expected_u[i].to_bits());
        }
    }
}

#[test]
fn cooperative_bias_expiry_precedes_getters_and_midwork_refusal_keeps_inputs() {
    let bias = Array1::ones(513);
    let block = BiasBlockParams {
        num_outputs: 1,
        out_features: 513,
        col_offset: 0,
    };
    for expired_at_entry in [false, true] {
        let getters = Cell::new(0);
        let get = |_: usize, _: usize| {
            getters.set(getters.get() + 1);
            1.0
        };
        let mut lower = [0.0];
        let mut upper = [0.0];
        let mut units = 0;
        let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let mut poll = WeightedPoll::new(|n| {
            units += n;
            if expired_at_entry {
                observe(Some(past))
            } else if units >= POLL_CHUNK_UNITS {
                Err(NyError::DeadlineExceeded("bias forced expiry".into()))
            } else {
                Ok(())
            }
        });
        assert!(accumulate_bias_f64_with_work(
            &mut (&mut lower, &mut upper),
            get,
            get,
            &bias,
            &block,
            &mut poll
        )
        .unwrap_err()
        .is_deadline_exceeded());
        assert_eq!(getters.get(), if expired_at_entry { 0 } else { 256 });
        assert_eq!(
            units,
            if expired_at_entry {
                0
            } else {
                POLL_CHUNK_UNITS
            }
        );
        assert!(bias.iter().all(|&v| v == 1.0));
    }
}

#[test]
fn directed_bias_into_strided_views_matches_historical_bits_and_preserves_gaps() {
    let values = [
        0.0,
        -0.0,
        1.0e-300,
        -1.0e-300,
        1.0e50,
        -1.0e50,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0 / 3.0,
    ];
    let lower = Array1::from_shape_fn(40, |i| values[i % 10]);
    let upper = Array1::from_shape_fn(40, |i| values[(i * 7 + 3) % 10]);
    let old_l = Array1::from_shape_fn(40, |i| if i % 3 == 0 { f32::from_bits(1) } else { -2.0 });
    let old_u = -&old_l;
    let (expected_l, expected_u) = finalize_bias_directed(
        &lower.slice(s![..;2]).to_owned(),
        &upper.slice(s![..;2]).to_owned(),
        &old_l.slice(s![..;2]).to_owned(),
        &old_u.slice(s![..;2]).to_owned(),
    );
    let sentinel = f32::from_bits(0x7fc0_0042);
    for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
        let mut actual_l = Array1::from_elem(40, sentinel);
        let mut actual_u = Array1::from_elem(40, sentinel);
        let mut units = 0;
        let mut poll = WeightedPoll::new(|n| {
            units += n;
            observe(deadline)
        });
        finalize_bias_directed_into(
            lower.slice(s![..;2]),
            upper.slice(s![..;2]),
            old_l.slice(s![..;2]),
            old_u.slice(s![..;2]),
            actual_l.slice_mut(s![..;2]),
            actual_u.slice_mut(s![..;2]),
            &mut poll,
        )
        .unwrap();
        assert_eq!(units, 40 * 16);
        for i in 0..40 {
            assert_eq!(
                actual_l[i].to_bits(),
                if i % 2 == 0 {
                    expected_l[i / 2].to_bits()
                } else {
                    sentinel.to_bits()
                }
            );
            assert_eq!(
                actual_u[i].to_bits(),
                if i % 2 == 0 {
                    expected_u[i / 2].to_bits()
                } else {
                    sentinel.to_bits()
                }
            );
        }
    }
}

#[test]
fn directed_bias_into_refuses_after_real_work_and_at_final_commit() {
    for midwork in [false, true] {
        let len = if midwork { 300 } else { 1 };
        let accum = Array1::ones(len);
        let old = Array1::zeros(len);
        let mut lower = Array1::from_elem(len, 42.0);
        let mut upper = lower.clone();
        let mut units = 0;
        let mut calls = 0;
        let mut poll = WeightedPoll::new(|n| {
            units += n;
            calls += 1;
            if (midwork && units >= POLL_CHUNK_UNITS) || (!midwork && calls == 2) {
                Err(NyError::DeadlineExceeded("bias publication refused".into()))
            } else {
                Ok(())
            }
        });
        assert!(finalize_bias_directed_into(
            accum.view(),
            accum.view(),
            old.view(),
            old.view(),
            lower.view_mut(),
            upper.view_mut(),
            &mut poll
        )
        .unwrap_err()
        .is_deadline_exceeded());
        assert_eq!(
            lower.iter().filter(|&&v| v != 42.0).count(),
            if midwork { 256 } else { 1 }
        );
        assert_eq!(
            upper.iter().filter(|&&v| v != 42.0).count(),
            if midwork { 0 } else { 1 }
        );
    }
}
