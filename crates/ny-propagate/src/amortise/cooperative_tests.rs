// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::{CooperativeWork, NoPoll, WeightedPoll, POLL_CHUNK_UNITS};
use ny_core::NyError;

#[test]
fn weighted_ragged_batches_deliver_all_units_without_overshooting_quantum() {
    let batches = [64, 2048, 1984, 1, 4095, 7, 0, 23, 4096];
    let mut delivered = Vec::new();
    {
        let mut poll = WeightedPoll::new(|units| {
            delivered.push(units);
            Ok(())
        });
        poll.checkpoint().unwrap();
        for units in batches {
            poll.before(units).unwrap();
        }
        poll.checkpoint().unwrap();
        poll.checkpoint().unwrap();
    }
    assert_eq!(delivered, [0, 4096, 4096, 30, 4096, 0]);
    assert_eq!(
        delivered.iter().sum::<usize>(),
        batches.iter().sum::<usize>()
    );
    assert!(delivered.iter().all(|&units| units <= POLL_CHUNK_UNITS));
}

#[test]
fn weighted_refusal_precedes_the_next_real_batch_after_4096_units() {
    let mut actual = 0;
    let mut delivered = 0;
    let result = {
        let mut poll = WeightedPoll::new(|units| {
            delivered += units;
            if delivered >= POLL_CHUNK_UNITS {
                Err(NyError::DeadlineExceeded(
                    "injected shared authority".into(),
                ))
            } else {
                Ok(())
            }
        });
        (|| {
            poll.checkpoint()?;
            for _ in 0..257 {
                poll.before(16)?;
                actual += 16;
            }
            poll.checkpoint()
        })()
    };
    assert!(result.unwrap_err().is_deadline_exceeded());
    assert_eq!(actual, POLL_CHUNK_UNITS);
    assert_eq!(delivered, actual);
}

#[test]
fn weighted_entry_and_empty_publication_observe_the_same_authority() {
    let mut calls = 0;
    let mut poll = WeightedPoll::new(|units| {
        assert_eq!(units, 0);
        calls += 1;
        if calls == 2 {
            Err(NyError::DeadlineExceeded("publication refused".into()))
        } else {
            Ok(())
        }
    });
    poll.checkpoint().unwrap();
    poll.before(0).unwrap();
    assert!(poll.checkpoint().unwrap_err().is_deadline_exceeded());
}

#[test]
fn weighted_oversized_batch_is_refused_before_work_in_both_poll_arms() {
    let mut calls = 0;
    let mut poll = WeightedPoll::new(|_| {
        calls += 1;
        Ok(())
    });
    assert!(poll.before(POLL_CHUNK_UNITS + 1).is_err());
    assert_eq!(poll.pending, 0);
    assert!(NoPoll.before(POLL_CHUNK_UNITS + 1).is_err());
    assert_eq!(calls, 0);
}
