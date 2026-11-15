// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #seq-single-reserve / #seq-early-attack — the sequential initial phase's
//! schedule and early-quit routing (2026-08-31).
//!
//! The schedule ARITHMETIC (double reserve, byte-identical kill switch, the
//! measured tll two-point validation) is pinned in
//! `config/phase_budget.rs::tests`. These tests exercise the ENGINE seam:
//! a warmup-cap expiry inside `evaluate_initial_phase` must consult the
//! post-BaB PGD fallback before publishing its Timeout (historically it
//! escaped `core.rs` before the fallback closure was constructed), must
//! return the ORIGINAL result when the attack finds nothing, and must
//! restore the historical forfeit byte-identically under the kill switch.

use std::time::Instant;

use super::super::cut_gate::CutGateState;
use super::super::verify_phases::InitialPhaseOutcome;
use super::prelude::*;
use crate::beta_crown::result::{
    reset_sequential_initial_schedule_receipt, take_sequential_initial_schedule_receipt,
    SequentialEarlyQuitDisposition,
};

/// Simple 2 -> 1 linear network: output = x0 + 2*x1, so with the unit box
/// the reachable output range is [0, 3].
fn linear_2_to_1() -> Network {
    let w = arr2(&[[1.0, 2.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network
}

fn unit_box_input() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap()
}

fn early_quit_config() -> BetaCrownConfig {
    BetaCrownConfig {
        enable_pgd_attack: true,
        pgd_restarts: 5,
        pgd_steps: 20,
        verify_upper_bound: true,
        enable_cuts: false,
        timeout: Duration::from_secs(600),
        ..BetaCrownConfig::default()
    }
}

/// Drive `evaluate_initial_phase` into its warmup-cap expiry deterministically:
/// the phase start is placed far enough in the past that the warmup cap
/// (default fractions: 20% of the window) is already expired at entry, while
/// the overall (caller) deadline is still live — exactly the tll N=64 shape,
/// where the cap expires at 369.5 s of a 483 s boundary with the PGD window
/// still ahead.
fn run_early_quit(config: BetaCrownConfig, threshold: f32) -> InitialPhaseOutcome {
    let network = linear_2_to_1();
    let input = unit_box_input();
    let verifier = BetaCrownVerifier::new(config);
    let cut_gate = CutGateState::new(&verifier.config);
    let mut cut_pool = CutPool::new(0);

    let start_time = Instant::now().checked_sub(Duration::from_secs(5)).unwrap();
    let overall_deadline = Some(start_time + Duration::from_secs(10));
    verifier
        .evaluate_initial_phase(
            &network,
            &input,
            threshold,
            None,
            start_time,
            &cut_gate,
            &mut cut_pool,
            overall_deadline,
        )
        .expect("an expired warmup cap is a normal outcome, not an error")
}

/// #seq-early-attack: a warmup-cap expiry on a falsifiable row must consult
/// the PGD fallback and publish the independently re-validated Violated —
/// the outcome the historical Early return forfeited.
#[ntest::timeout(20000)]
#[test]
fn early_quit_routes_through_the_pgd_fallback() {
    reset_sequential_initial_schedule_receipt();
    // Threshold 2.5 is reachable (max output 3), so the attack must convert.
    let outcome = run_early_quit(early_quit_config(), 2.5);
    match outcome {
        InitialPhaseOutcome::Early(result) => match result.result {
            BabVerificationStatus::Violated { ref output, .. } => {
                assert!(
                    output[0] >= 2.5,
                    "re-validated output {} should be >= threshold 2.5",
                    output[0]
                );
            }
            ref other => {
                panic!("#seq-early-attack must convert the falsifiable early quit, got {other:?}")
            }
        },
        InitialPhaseOutcome::Proceed { .. } => {
            panic!("an expired warmup cap must produce an Early outcome")
        }
    }
    let receipt = take_sequential_initial_schedule_receipt()
        .expect("the engine must leave a schedule receipt for the CLI flight note");
    assert_eq!(receipt.early_quit, SequentialEarlyQuitDisposition::Attacked);
    assert!(receipt.caller_deadline);
    assert!(
        receipt.warmup_single_reserve_applied,
        "#seq-single-reserve ships ARMED on the caller-deadline route"
    );
    assert!(
        (receipt.effective_total_secs - 10.0).abs() < 0.2,
        "the receipt must carry the caller window, got {}",
        receipt.effective_total_secs
    );
    assert!(
        receipt.warmup_cap_secs > receipt.effective_total_secs * 0.19
            && receipt.warmup_cap_secs <= receipt.effective_total_secs,
        "the corrected warmup cap must be ledger-true (~20% of 10 s, not 20% \
         of the double-reserved 9 s), got {}",
        receipt.warmup_cap_secs
    );
}

/// #seq-early-attack containment: when the attack finds nothing (threshold
/// unreachable), the ORIGINAL Timeout is returned unchanged.
#[ntest::timeout(20000)]
#[test]
fn early_quit_attack_returns_the_original_timeout_when_it_finds_nothing() {
    reset_sequential_initial_schedule_receipt();
    // Threshold 10 is unreachable (max output 3): the fallback must return
    // the original result.
    let outcome = run_early_quit(early_quit_config(), 10.0);
    match outcome {
        InitialPhaseOutcome::Early(result) => {
            assert!(
                matches!(result.result, BabVerificationStatus::Timeout),
                "a fruitless attack must not change the published verdict, got {:?}",
                result.result
            );
            assert_eq!(result.domains_explored, 0);
            assert_eq!(result.domains_verified, 0);
            assert!(result.output_bounds.is_none());
        }
        InitialPhaseOutcome::Proceed { .. } => {
            panic!("an expired warmup cap must produce an Early outcome")
        }
    }
    let receipt = take_sequential_initial_schedule_receipt().expect("receipt");
    assert_eq!(receipt.early_quit, SequentialEarlyQuitDisposition::Attacked);
}

/// #seq-early-attack kill switch: `sequential_early_quit_attack: false` must
/// restore the historical forfeit byte-identically — the SAME falsifiable
/// row that converts in `early_quit_routes_through_the_pgd_fallback` must
/// publish the bare Timeout with the attack never consulted. This is the
/// able-to-fail control for that test.
#[ntest::timeout(20000)]
#[test]
fn early_quit_kill_switch_restores_the_historical_forfeit() {
    reset_sequential_initial_schedule_receipt();
    let mut config = early_quit_config();
    config.phase_budget.sequential_early_quit_attack = false;
    let outcome = run_early_quit(config, 2.5);
    match outcome {
        InitialPhaseOutcome::Early(result) => {
            assert!(
                matches!(result.result, BabVerificationStatus::Timeout),
                "the kill switch must restore the historical forfeit, got {:?}",
                result.result
            );
            assert_eq!(result.domains_explored, 0);
        }
        InitialPhaseOutcome::Proceed { .. } => {
            panic!("an expired warmup cap must produce an Early outcome")
        }
    }
    let receipt = take_sequential_initial_schedule_receipt().expect("receipt");
    assert_eq!(
        receipt.early_quit,
        SequentialEarlyQuitDisposition::ForfeitedByKillSwitch
    );
}

/// #seq-single-reserve kill switch at the engine seam: with the key OFF the
/// warmup cap is the historical double-reserved one; the receipt records the
/// distinction so a banked row can prove which schedule it ran under.
#[ntest::timeout(20000)]
#[test]
fn warmup_kill_switch_reports_the_historical_double_reserved_cap() {
    reset_sequential_initial_schedule_receipt();
    let mut config = early_quit_config();
    config.phase_budget.sequential_warmup_single_reserve = false;
    let outcome = run_early_quit(config, 10.0);
    assert!(matches!(outcome, InitialPhaseOutcome::Early(_)));
    let receipt = take_sequential_initial_schedule_receipt().expect("receipt");
    assert!(!receipt.warmup_single_reserve_applied);
    assert!(
        (receipt.warmup_cap_secs - 10.0 * 0.9 * 0.2).abs() < 0.05,
        "the historical cap is effective_total * (1 - pgd_frac) * warmup_frac, got {}",
        receipt.warmup_cap_secs
    );
    assert!(
        (receipt.bab_timeout_secs - 9.0).abs() < 0.05,
        "the BaB boundary keeps the engine single reserve in every disposition, got {}",
        receipt.bab_timeout_secs
    );
}
