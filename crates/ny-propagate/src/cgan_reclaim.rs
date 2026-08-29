// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cgan-reclaim — one gate for the cgan BUDGET-RECLAIM set
//! (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md).
//!
//! The diagnosed mechanism (scratchpad cgan_live_route.md, 2026-08-19): the
//! 900 s grouped-disjunctive-eager route is a CLOSING tree starved of budget —
//! ~630 expansions delivered where the near-closed holdout needed ~750 —
//! because ~68 % of the wall is spent before the first split. `NY_CGAN_RECLAIM=1`
//! (exact token) arms three retention sites together, so the set cannot be
//! measured split (the standing audit finding: set-mates switch together,
//! partial fixes measure exactly zero):
//!
//! 1. Cross-mode CROWN-IBP collection-cache serve
//!    (`network/graph_alpha/bounds/crown.rs`): the CLI precheck's COMPLETE
//!    Standard-mode root collection may serve a typed cGAN consumer over the
//!    bit-identical box, shrink-only intersected against the consumer's own
//!    certified forward-linear baseline.
//! 2. Crown-partial expiry retention
//!    (`network/ibp/crown_partial.rs:crown_partial_expiry_armed`): same
//!    semantics as `NY_CROWN_PARTIAL_FINITE_EXPIRY=1` for the root-cause-D
//!    pair (GPU partial route + sparse seed decide by EXPIRY, not PRESENCE).
//! 3. Sequential CROWN-IBP expiry salvage
//!    (`network/ibp/crown_ibp.rs:collect_core`): at mid-walk global expiry,
//!    commit-what-you-have instead of discarding every completed per-node
//!    tightening with an `Err`.
//! 4. ConvTranspose2d mid-step expiry salvage
//!    (`network/core/sequential/crown/patches_step.rs`, the ConvTranspose2d
//!    Patches arm): a deadline that crosses INSIDE the Anchored planner is
//!    absorbed as `CrownStepResult::IbpFallback` — the walk keeps the
//!    enclosure it already holds — instead of returning an `Err` that
//!    escapes to the CLI as "no verdict" with a third of the budget unspent
//!    (measured: `cGAN_imgSz64_nCh_3` prop_1/prop_3 stop at 826/825 s of
//!    900 s). The same walk already returns that enclosure when it finds the
//!    deadline expired at the step's ENTRY check.
//! 5. Forward-fallback PUBLICATION past expiry
//!    (`graph_crown/propagation.rs:plain_forward_fallback` and
//!    `core/sequential/crown.rs:clone_crown_forward_fallback`, via
//!    [`fallback_publication_deadline`]): item 4's partner. With item 4 alone
//!    the ConvTranspose expiry is absorbed and the walk then asks for its
//!    sound forward enclosure — and the CLONE of that enclosure is itself
//!    deadline-refused, so the row still dies (MEASURED on
//!    `cGAN_imgSz64_nCh_3` prop_1: the stopper simply moves from
//!    "ConvTranspose2d Patches backward: deadline exceeded" to "GraphNetwork
//!    DAG-CROWN: deadline exceeded before forward fallback clone", 826 s ->
//!    825 s of 900 s). Memory receipts unchanged; only the wall-clock refusal
//!    on a bounded memcpy of an already-certified enclosure is lifted.
//!
//! SOUNDNESS: every arm is a monotone intersect of independently certified
//! enclosures, a selection between two sound routes, or a retention of
//! already-certified bounds. No arm manufactures a bound value, and no arm
//! grants new wall-clock authority (all downstream expiry checks remain).
//!
//! Deliberately NOT latched in a `OnceLock`: the callers are once-per-collection
//! or once-per-expiry-event (never a hot loop), and an unlatched read keeps the
//! `with_env_edits` test harness able to exercise both arms in one process.
//! Only the R9 engagement announcement is latched, so it prints exactly once.

/// Pure arming rule: exact `"1"` arms, everything else (absence, `"0"`,
/// `"true"`, `" 1"`, empty) leaves the set dark — the campaign's exact-token
/// convention, pinned by `tests::token_is_exact_one`.
#[inline]
fn cgan_reclaim_token_arms(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Whether the cgan budget-reclaim set is armed (`NY_CGAN_RECLAIM=1`).
///
/// Env acquisition goes through the typed declaration's `read_raw` chokepoint
/// (`ny_levers::decls::cgan_stacked::CGAN_RECLAIM`) so the lever appears in
/// the flight receipt; the unit-tested pure predicate above IS the arming
/// rule, the sanctioned `read_raw` caller shape.
///
/// Prints the R9 engagement announcement once per process on first armed read;
/// a null measured without that line is indistinguishable from an inert lever.
pub(crate) fn cgan_reclaim_armed() -> bool {
    let raw = ny_levers::read_raw(&ny_levers::decls::cgan_stacked::CGAN_RECLAIM);
    // Env wins in BOTH directions where present — exact `1` arms, any other
    // present token is the fail-closed disarm. Absent, the typed preset key
    // decides. This is the shape that makes the set reachable in competition:
    // `vnncomp_scripts/run_instance.sh` exports no `NY_*`, so an
    // env-only gate can never fire on a scored run.
    let (armed, source) = match raw.as_deref() {
        Some(token) => (cgan_reclaim_token_arms(Some(token)), "NY_CGAN_RECLAIM"),
        None => (
            CGAN_RECLAIM_PRESET.load(std::sync::atomic::Ordering::Relaxed),
            "preset bab.cgan_reclaim",
        ),
    };
    if armed {
        static ANNOUNCED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        ANNOUNCED.get_or_init(|| {
            eprintln!(
                "[cgan-reclaim] ARMED (source: {source}): cross-mode collection-cache \
                 serve + crown-partial expiry retention + sequential CROWN-IBP \
                 expiry salvage + ConvTranspose2d mid-step expiry salvage + \
                 forward-fallback publication past expiry \
                 (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md)"
            );
        });
    }
    armed
}

/// Typed preset route for the set (`bab.cgan_reclaim`). Same bridge shape as
/// `margin_row::set_backward_interm_preset`: the CLI writes it once from the
/// resolved preset before any verification starts, and an explicit env value
/// still wins in both directions.
static CGAN_RECLAIM_PRESET: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Arm (or explicitly disarm) the cgan budget-reclaim set from the typed
/// preset key. Default `false` — omission leaves the set dark.
pub fn set_cgan_reclaim_preset(on: bool) {
    CGAN_RECLAIM_PRESET.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// #cgan-reclaim item 5 — a forward-fallback PUBLICATION may complete past
/// expiry.
///
/// MEASURED (`cGAN_imgSz64_nCh_3` prop_1, 900 s, revision 4882): with item 4
/// armed the ConvTranspose2d mid-step expiry is absorbed and the walk asks for
/// its sound forward enclosure — and then the run dies anyway on
/// `GraphNetwork DAG-CROWN: deadline exceeded before forward fallback clone`,
/// 75 s inside its budget. The FALLBACK ITSELF is deadline-refused.
///
/// That refusal cannot buy back any wall clock worth having. By the time these
/// clones run, the enclosure is ALREADY computed and ALREADY resident; the only
/// remaining work is copying `2 * len` f32 endpoints — 230 KB on cgan's widest
/// node (28,800 dims), microseconds. What the refusal actually costs is the
/// whole instance: the `Err` escapes the graph pass, the CLI prints
/// "Verification produced no verdict (sound unknown)" and the row scores zero
/// while holding a perfectly good sound bound.
///
/// Armed, the wall-clock refusal is lifted for that copy alone. Every MEMORY
/// receipt inside the clone is untouched, no bound value is computed or
/// changed (it is a bit-exact copy of an already-certified enclosure), and no
/// SEARCH gets extra time: the callers publish and return.
///
/// Returns the deadline the clone should be given.
pub(crate) fn fallback_publication_deadline(
    deadline: Option<std::time::Instant>,
    site: &'static str,
) -> Option<std::time::Instant> {
    if deadline.is_none() || !cgan_reclaim_armed() {
        return deadline;
    }
    // R9: the arm's own observable, rate limited to the first three events and
    // then powers of two. A null measured without this line cannot be told
    // apart from an inert lever.
    use std::sync::atomic::{AtomicU64, Ordering};
    static PUBLISHED: AtomicU64 = AtomicU64::new(0);
    let count = PUBLISHED.fetch_add(1, Ordering::Relaxed).saturating_add(1);
    if count <= 3 || count.is_power_of_two() {
        eprintln!(
            "[cgan-reclaim] fallback-publication site={site} count={count}: \
             publishing the already-certified forward enclosure past expiry \
             (memory receipt unchanged)"
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_exact_one() {
        assert!(cgan_reclaim_token_arms(Some("1")));
        assert!(!cgan_reclaim_token_arms(Some("0")));
        assert!(!cgan_reclaim_token_arms(Some("true")));
        assert!(!cgan_reclaim_token_arms(Some("")));
        assert!(!cgan_reclaim_token_arms(Some(" 1")));
        assert!(!cgan_reclaim_token_arms(Some("01")));
        assert!(!cgan_reclaim_token_arms(None));
    }
}
