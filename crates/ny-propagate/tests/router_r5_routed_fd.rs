// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r5: the frozen-dual tier's ROUTED arming, through the
//! PROCESS-GLOBAL route authority the tier's own latch consults
//! (`zono_state::armed() = key || routed()`).
//!
//! Its OWN test process on purpose: it arms the process-global route
//! authority, which a shared unit-test binary must never do (another test's
//! margin-row lane would read it and run the tier). Its one test runs
//! serially.

use ny_core::route_authority;
use ny_core::router::decl::ids;
use ny_propagate::margin_row::zono_state;

/// Unarmed, `routed()` is `false` (one atomic load: every row without
/// `router.mode: on`, and every `on` row today since nothing is sealed);
/// armed with a route for the tier it is `true` and the consult is counted
/// (the route's arrival evidence); a route for another capability, or a
/// disarmed authority, is `false` again.
#[test]
fn router_r5_frozen_dual_latch_reads_the_route_authority() {
    let authority = route_authority::global();
    assert!(!authority.is_armed());
    assert!(!zono_state::routed(), "unarmed: today's key alone");
    // The tier's own latch: no key in this process (dark), so it is the
    // route alone.
    assert!(
        !zono_state::armed(),
        "dark key, nothing routed: the tier is off"
    );

    authority.arm(
        vec![(
            ids::WAVE_BETA_BAB.as_str().to_owned(),
            vec![("margin_row.tree_mode".to_owned(), "wave".to_owned())],
        )],
        false,
    );
    assert!(!zono_state::routed(), "another capability's route");

    authority.arm(
        vec![(
            ids::FROZEN_DUAL_BAB.as_str().to_owned(),
            vec![("margin_row.frozen_dual_bab".to_owned(), "true".to_owned())],
        )],
        false,
    );
    assert!(zono_state::routed());
    assert!(
        zono_state::armed(),
        "the tier's own latch reads the route: key || routed"
    );
    let snapshot = authority.snapshot();
    assert_eq!(snapshot.routes[0].cap, ids::FROZEN_DUAL_BAB.as_str());
    assert_eq!(snapshot.routes[0].consults, 2, "each consult is observed");

    authority.disarm();
    assert!(!zono_state::routed(), "a disarmed authority is today's");
    assert!(!zono_state::armed(), "and the tier is off again");
}
