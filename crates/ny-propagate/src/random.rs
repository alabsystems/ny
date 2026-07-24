// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use rand::RngExt;

/// Process-global restart offset added to the base seed (task #36).
///
/// The deterministic multi-seed restart loop (input-split disjunctive BaB) sets
/// this before each restart so the SAME base seed yields a DIFFERENT — but still
/// fully deterministic — relaxation-parameter draw per restart index. Restarts
/// run strictly one-at-a-time (sequential BaB), so a single global cell is
/// sufficient and every RNG consumer (including rayon workers spawned inside a
/// restart) reads the offset for the restart currently executing. Default 0, so
/// with no restart wrapper the seed is exactly the base seed — the historical
/// single-seed behavior tests relied on.
static RESTART_OFFSET: AtomicU64 = AtomicU64::new(0);

/// Return a DETERMINISTICALLY-seeded RNG for the bound-optimization routines.
///
/// Every caller of this function (SPSA α-ascent, the α-CROWN gradient estimator,
/// the MulBinary α SPSA, the DAG-supplement gradients) uses the draw ONLY to
/// choose the VALUE of a relaxation parameter — a slope/α that yields a sound
/// enclosure for ANY choice in its box. So the seed can never weaken a bound; it
/// only decides WHICH (equally sound) relaxation gets picked. Neither the PGD
/// attack (its own fixed-seed `SimpleRng`, seed 42) nor the probabilistic
/// Monte-Carlo verifier (`rand::rng()` directly) route through here.
///
/// Determinism (task #36): with an entropy seed the SPSA-optimized alphas
/// differed at the ULP level from one process to the next; on razor-thin
/// disjunctive specs (lsnc `quadrotor2d_state_0`) those ULP shifts cascaded
/// through input-split BaB into different per-domain verification decisions and a
/// run-dependent verdict (unsat vs timeout). A fixed seed makes the SAME alphas —
/// hence the SAME sound bounds and the SAME BaB tree — recur every run. Tests
/// already relied on this deterministic seed; production now shares it. The
/// multi-seed restart loop then tries a FIXED sequence of seeds (base+0, base+1,
/// …) in a fixed order, keeping the first success — reproducible run-to-run yet
/// no longer gambling the verdict on a single lucky seed.
pub(crate) fn rng() -> impl RngExt {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(current_seed())
}

/// Base default seed. `NY_RNG_SEED` overrides it (still deterministic per
/// process) so a reproducibility sweep can A/B relaxation-parameter seeds without
/// a rebuild. Not read in a hot loop — every caller creates one RNG per
/// optimization, never per iteration.
fn base_seed() -> u64 {
    std::env::var("NY_RNG_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

/// The seed currently in effect: `base_seed + restart_offset`. The multi-seed
/// restart loop advances the offset; every other path leaves it at 0.
pub(crate) fn current_seed() -> u64 {
    base_seed().wrapping_add(RESTART_OFFSET.load(Ordering::Relaxed))
}

/// Set the restart offset for the restart about to execute and return a guard
/// that restores the offset to 0 on drop. The guard makes a panic or early
/// return inside a restart unable to leak a non-zero offset into unrelated code
/// (importantly, into other tests sharing the process). Part of task #36.
///
/// Re-exported from the crate root as `set_rng_restart_offset` for the
/// input-split disjunctive multi-seed restart loop in `ny-cli`. The enclosing
/// `random` module is private, so the machinery is reachable only through that
/// single explicit re-export.
#[must_use]
pub fn set_restart_offset(offset: u64) -> RestartOffsetGuard {
    RESTART_OFFSET.store(offset, Ordering::Relaxed);
    RestartOffsetGuard
}

/// RAII guard that resets the restart offset to 0 when dropped.
pub struct RestartOffsetGuard;

impl Drop for RestartOffsetGuard {
    fn drop(&mut self) {
        RESTART_OFFSET.store(0, Ordering::Relaxed);
    }
}
