// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `ny_api::levers::install` writes PROCESS-GLOBAL cells and may run once.
//!
//! That is why this is a separate integration binary with exactly ONE test:
//! two `#[test]`s in one binary run on different threads of the same process,
//! so a "defaults are unchanged" assertion and an "installation took effect"
//! assertion would race over the same cells. Here the whole life cycle —
//! defaults, install, confirm, refuse the second install — is one sequence.

use ny_api::levers::{self, LeverPlan, RefusalReason, Setting};

#[test]
fn a_safe_lever_round_trips_and_the_one_shot_holds() {
    // --- 1. Nothing is installed, so every settable lever reads its shipped
    // default. `NY_EPOCH_TIER0` declares `DefaultSpec::U64(0)`.
    assert!(!levers::installed());
    let baselines: Vec<(&'static str, Option<Setting>)> = levers::settable()
        .into_iter()
        .map(|name| (name, levers::effective(name)))
        .collect();
    assert_eq!(
        levers::effective("NY_EPOCH_TIER0"),
        Some(Setting::Count(0)),
        "an unwritten cell must resolve to the declaration default"
    );

    // An empty plan is a no-op and must NOT consume the one-shot.
    let empty = levers::install(&LeverPlan::new()).expect("an empty plan installs trivially");
    assert!(empty.is_empty());
    assert!(
        !levers::installed(),
        "an empty plan must not claim the one-shot"
    );

    // --- 2. Build a plan over the SafeToSearch subset only.
    let mut plan = LeverPlan::new();
    plan.set("NY_EPOCH_TIER0", Setting::Count(7))
        .expect("a SafeToSearch count lever must be accepted")
        .set("NY_MARGIN_ROW_CLIP_ROWS", Setting::Bool(true))
        .expect("a SafeToSearch gate must be accepted")
        .set("NY_MARGIN_ROW_BETA_ETA", Setting::Real(0.25))
        .expect("a SafeToSearch fraction must be accepted");
    assert_eq!(plan.len(), 3);
    assert_eq!(
        plan.levers(),
        vec![
            "NY_EPOCH_TIER0",
            "NY_MARGIN_ROW_BETA_ETA",
            "NY_MARGIN_ROW_CLIP_ROWS"
        ]
    );

    // Building the plan alone must change nothing.
    for (name, baseline) in &baselines {
        assert_eq!(
            levers::effective(name),
            *baseline,
            "{name} must not move until install() runs"
        );
    }

    // --- 3. Install, and confirm through the LIVE resolver.
    let installed = match levers::install(&plan) {
        Ok(installed) => installed,
        Err(refusal) => {
            if matches!(
                refusal.reason,
                RefusalReason::OverriddenByProcessEnvironment { .. }
            ) {
                // The test host exported one of these NY_* names. Refusing
                // WITHOUT writing anything is the correct, fail-closed
                // behaviour, and that is what the rest of this test checks.
                assert!(
                    !levers::installed(),
                    "an environment-refused install must not claim the one-shot"
                );
                for (name, baseline) in &baselines {
                    assert_eq!(levers::effective(name), *baseline, "{name} must not move");
                }
                return;
            }
            panic!("install refused unexpectedly: {refusal}");
        }
    };
    assert!(levers::installed());
    assert_eq!(
        installed.effective(),
        &[
            ("NY_EPOCH_TIER0", Setting::Count(7)),
            ("NY_MARGIN_ROW_BETA_ETA", Setting::Real(0.25)),
            ("NY_MARGIN_ROW_CLIP_ROWS", Setting::Bool(true)),
        ]
    );
    assert_eq!(levers::effective("NY_EPOCH_TIER0"), Some(Setting::Count(7)));
    assert_eq!(
        levers::effective("NY_MARGIN_ROW_CLIP_ROWS"),
        Some(Setting::Bool(true))
    );

    // --- 4. Levers NOT in the plan keep their defaults: delivery is targeted,
    // not a wholesale reconfiguration.
    for (name, baseline) in &baselines {
        if plan.levers().contains(name) {
            continue;
        }
        assert_eq!(
            levers::effective(name),
            *baseline,
            "{name} was not in the plan and must not have moved"
        );
    }

    // --- 5. The one-shot holds. The readers latch, so a second, different
    // installation could not take effect; it is refused rather than silently
    // half-applied.
    let mut second = LeverPlan::new();
    second
        .set("NY_EPOCH_RETAIN", Setting::Count(11))
        .expect("a SafeToSearch lever is still plannable");
    let refusal = levers::install(&second).expect_err("a second install must be refused");
    assert_eq!(refusal.reason, RefusalReason::AlreadyInstalled);
    assert!(
        refusal.to_string().contains("latch"),
        "the refusal must explain the ordering rule: {refusal}"
    );
    assert_ne!(
        levers::effective("NY_EPOCH_RETAIN"),
        Some(Setting::Count(11)),
        "a refused install must write nothing"
    );
}
