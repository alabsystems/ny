// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ny_propagate::{BabVerificationStatus, BetaCrownResult, GraphNetwork};

use super::{
    disjunctive::{finalize_disjunctive_result, run_with_optional_forward_linear_warmer},
    normalize_result_wall_time,
};

#[test]
fn short_budget_does_not_spawn_or_join_optional_forward_linear_warmer() {
    let short_deadline = Some(Instant::now() + Duration::from_secs(10));
    let admitted = GraphNetwork::forward_linear_cold_build_admitted(short_deadline);
    assert!(
        !admitted,
        "a 10s verifier slice cannot fit the cold image pass"
    );

    let worker_ran = AtomicBool::new(false);
    let result = run_with_optional_forward_linear_warmer(
        admitted.then_some({
            || {
                worker_ran.store(true, Ordering::SeqCst);
                panic!("refused optional warmer must never be spawned");
            }
        }),
        || 17usize,
    );

    assert_eq!(result, 17, "the foreground attack result must be preserved");
    assert!(
        !worker_ran.load(Ordering::SeqCst),
        "short-budget scope must have no warmer worker to join"
    );
}

#[test]
fn finalize_disjunctive_result_uses_overall_wall_time_3870() {
    let overall_start = Instant::now().checked_sub(Duration::from_secs(3)).unwrap();
    let aggregated = BetaCrownResult {
        result: BabVerificationStatus::Timeout,
        domains_explored: 11,
        time_elapsed: Duration::from_secs(1),
        max_depth_reached: 7,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 5,
    };

    let finalized = finalize_disjunctive_result(
        aggregated,
        overall_start,
        BabVerificationStatus::Unknown {
            reason: "Clause 1: timeout".to_string(),
        },
    );

    assert!(
        finalized.time_elapsed >= Duration::from_secs(3),
        "final disjunctive time should track overall wall time, not summed clause time"
    );
    assert!(
        finalized.time_elapsed < Duration::from_secs(4),
        "final disjunctive wall time should stay close to the injected overall start"
    );
    assert!(
        matches!(
            finalized.result,
            BabVerificationStatus::Unknown { ref reason } if reason == "Clause 1: timeout"
        ),
        "final status should be preserved while rewriting time_elapsed"
    );
}

#[test]
fn normalize_result_wall_time_keeps_larger_existing_elapsed_3870() {
    let overall_start = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let result = BetaCrownResult {
        result: BabVerificationStatus::Timeout,
        domains_explored: 0,
        time_elapsed: Duration::from_secs(3),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    let normalized = normalize_result_wall_time(result, overall_start);

    assert_eq!(
        normalized.time_elapsed,
        Duration::from_secs(3),
        "wall-time normalization should never shrink an already larger elapsed value"
    );
}
