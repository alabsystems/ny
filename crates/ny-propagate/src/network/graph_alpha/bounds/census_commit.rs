// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bug #19 (budget-monotonicity): shrink-only publication for the DAG-alpha
//! root intermediate map.
//!
//! MEASURED: the root census DEGRADES with more budget (idx_8600 83/99 at
//! 100 s -> 73/99 at 400 s; idx_885 72/99 -> 31/99), which is impossible when
//! every commit intersects into its incumbent — so something PUBLISHES
//! (overwrites) instead. The overwrite is in `alpha_dag_dispatch.rs`:
//!
//!   * Site A (`resolve_reference_bounds_publication`): the ordinary route
//!     REPLACES the optimizer's monotone artifact reference map (initial
//!     certified reference INTERSECTED with every mid-loop refresh candidate,
//!     see `reference_bounds.rs::merge_tighter_bounds`) with a fresh
//!     recollection, wholesale.
//!   * Site B (post-loop `collect_crown_bounds_with_alpha` publication): the
//!     all-node collection under the FINAL alpha state is published with only
//!     the output node intersected against the optimizer's monotone output
//!     box; every intermediate node takes the fresh collection verbatim. The
//!     final alpha degrades as iterations accumulate (the shipped alpha
//!     gradient is sign-definite <= 0, so alpha clamps toward 0 — see
//!     AlphaGradientDefect.lean), so MORE budget -> more iterations -> a
//!     LOOSER final-alpha collection replacing the same discarded artifact:
//!     the published map is anti-monotone in budget. (Mechanism INFERRED from
//!     the measured census inversion; the intersection fix is sound and
//!     monotone regardless of which site dominates.)
//!
//! SOUNDNESS of the fix: the artifact map and the freshly collected map are
//! INDEPENDENTLY CERTIFIED enclosures of the same nodes' reachable sets over
//! the SAME input box. The elementwise intersection `[max(l), min(u)]` of two
//! valid enclosures still contains every reachable point, and is at least as
//! tight as either operand. A rebuild under a different relaxation may be
//! validly looser for some neurons while tighter for others — intersection
//! keeps the tighter side of each, which is exactly why replacement (keeping
//! only ONE side for every neuron) loses certified tightenings. Elements
//! where f32 outward rounding of the two pipelines produces an empty
//! intersection are kept from the PUBLISHED candidate unchanged (fail-closed:
//! never manufacture a value); NaN candidate elements keep the incumbent.
//!
//! Gates (exact `"1"` arms, default OFF => byte-identical publication):
//!   * `NY_CENSUS_MONOTONE=1` — apply the shrink-only intersection at both
//!     sites, and emit `[census-commit]` telemetry.
//!   * `NY_CENSUS_COMMIT_TELEMETRY=1` — telemetry only: count, per phase, how
//!     many elements the published map is LOOSER than the incumbent artifact
//!     (`would_have_loosened` > 0 is the smoking gun) without touching any
//!     bound.

use ndarray::Zip;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

const CENSUS_MONOTONE_ENV: &str = "NY_CENSUS_MONOTONE";
const CENSUS_COMMIT_TELEMETRY_ENV: &str = "NY_CENSUS_COMMIT_TELEMETRY";

/// Exact-`"1"` arming test, shared by both gates. Any other value (including
/// `"true"`, `"0"`, empty) leaves the lever off — matching the campaign rule
/// that only the exact string arms a lever.
fn census_flag_armed(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// `NY_CENSUS_MONOTONE=1`: apply shrink-only commits (implies telemetry).
pub(in crate::network::graph_alpha) fn census_monotone_enabled() -> bool {
    census_flag_armed(std::env::var(CENSUS_MONOTONE_ENV).ok().as_deref())
}

/// `NY_CENSUS_COMMIT_TELEMETRY=1` (or the monotone lever): count and print,
/// never mutate on its own.
pub(in crate::network::graph_alpha) fn census_commit_observed() -> bool {
    census_flag_armed(std::env::var(CENSUS_COMMIT_TELEMETRY_ENV).ok().as_deref())
        || census_monotone_enabled()
}

/// Per-phase outcome of comparing (and optionally intersecting) a published
/// candidate map against the incumbent artifact map.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(in crate::network::graph_alpha) struct CensusCommitStats {
    /// Nodes present in both maps with matching shapes.
    pub(in crate::network::graph_alpha) nodes_compared: usize,
    /// Incumbent nodes skipped: absent from the published map or shape drift
    /// (pre- vs post-concat views, #4384). Skipping keeps the published entry
    /// — sound, merely un-tightened.
    pub(in crate::network::graph_alpha) nodes_skipped: usize,
    /// Elements where the published candidate is strictly tighter than the
    /// incumbent on at least one endpoint.
    pub(in crate::network::graph_alpha) elems_candidate_tighter: usize,
    /// Elements where the INCUMBENT is strictly tighter — i.e. the overwrite
    /// WOULD HAVE LOOSENED a previously certified bound. Nonzero proves the
    /// publish-instead-of-intersect bug.
    pub(in crate::network::graph_alpha) elems_would_loosen: usize,
    /// Elements whose f32 intersection came up empty; the published value was
    /// kept unchanged (fail-closed).
    pub(in crate::network::graph_alpha) elems_disjoint_kept: usize,
}

/// Compare `published` against `incumbent`; when `apply` is set, replace each
/// compared entry with the elementwise intersection (shrink-only commit).
///
/// See the module docs for the soundness argument at this, the load-bearing
/// commit site: both operands are independently certified enclosures over the
/// same input box, so `[max(l), min(u)]` is sound and at least as tight as
/// either; empty-after-rounding elements and NaN candidates fail closed.
pub(in crate::network::graph_alpha) fn shrink_only_commit(
    published: &mut HashMap<String, BoundedTensor>,
    incumbent: &HashMap<String, BoundedTensor>,
    apply: bool,
) -> CensusCommitStats {
    let mut stats = CensusCommitStats::default();
    for (name, incumbent_bounds) in incumbent {
        let Some(current) = published.get(name) else {
            stats.nodes_skipped += 1;
            continue;
        };
        if current.shape() != incumbent_bounds.shape() {
            stats.nodes_skipped += 1;
            continue;
        }
        stats.nodes_compared += 1;

        let mut lower = current.lower().clone();
        let mut upper = current.upper().clone();
        let mut node_changed = false;
        Zip::from(lower.view_mut())
            .and(upper.view_mut())
            .and(incumbent_bounds.lower())
            .and(incumbent_bounds.upper())
            .for_each(|pub_lower, pub_upper, &inc_lower, &inc_upper| {
                // NaN on the published side: the incumbent is the only
                // certified value; NaN on the incumbent side: keep published.
                // (f32::max/min already skip NaN, counts stay explicit.)
                if pub_lower.is_nan() || pub_upper.is_nan() {
                    if !inc_lower.is_nan() && !inc_upper.is_nan() {
                        stats.elems_would_loosen += 1;
                        if apply {
                            *pub_lower = inc_lower;
                            *pub_upper = inc_upper;
                            node_changed = true;
                        }
                    }
                    return;
                }
                if inc_lower.is_nan() || inc_upper.is_nan() {
                    stats.elems_candidate_tighter += 1;
                    return;
                }
                let merged_lower = pub_lower.max(inc_lower);
                let merged_upper = pub_upper.min(inc_upper);
                if merged_lower > merged_upper {
                    // Two sound outward-rounded pipelines can disagree by
                    // ulps; an empty f32 intersection is a rounding artifact,
                    // but intersecting anyway would MANUFACTURE a value.
                    // Fail closed: keep the published element unchanged.
                    stats.elems_disjoint_kept += 1;
                    return;
                }
                if inc_lower > *pub_lower || inc_upper < *pub_upper {
                    // The incumbent is strictly tighter on an endpoint: a
                    // plain publish would LOOSEN a certified bound here.
                    stats.elems_would_loosen += 1;
                    if apply {
                        *pub_lower = merged_lower;
                        *pub_upper = merged_upper;
                        node_changed = true;
                    }
                } else if *pub_lower > inc_lower || *pub_upper < inc_upper {
                    stats.elems_candidate_tighter += 1;
                }
            });

        if apply && node_changed {
            // `new_allow_infinite` mirrors `merge_tighter_bounds`: +/-inf
            // endpoints are sound overapproximations. A construction failure
            // keeps the original published entry (fail-closed, sound: it was
            // the certified candidate already headed for publication).
            if let Ok(mut merged) = BoundedTensor::new_allow_infinite(lower, upper) {
                // Intersecting endpoint boxes does not invalidate an
                // independently proven L2 ball that already encloses the same
                // live tensor values; preserve it exactly (same argument as
                // `publish_validated_batch` in intermediate_sweep.rs). If the
                // annotation cannot be carried, keep the original published
                // entry rather than dropping proof metadata.
                if let Some(l2) = current.l2_constraint().cloned() {
                    merged = merged.with_l2_constraint(l2);
                    if !merged.has_l2_constraint() {
                        continue;
                    }
                }
                published.insert(name.clone(), merged);
            }
        }
    }
    stats
}

/// Emit one rate-limited `[census-commit]` line for a phase's commit.
///
/// `would_have_loosened > 0` on an OBSERVE-only run is the smoking gun that
/// the phase publishes instead of intersecting. On an ARMED run the same
/// counter counts the loosenings the shrink-only commit PREVENTED — nonzero
/// there means the fix is engaged and doing work, which is the expected
/// healthy reading, not a defect. The R9 engagement check is `applied=true`
/// plus a nonzero `tightened + would_have_loosened` total; a permanently
/// zero counter on rows that reproduce the census regression would instead
/// mean the guarded phase is not the one publishing.
pub(in crate::network::graph_alpha) fn emit_census_commit(
    phase: &str,
    applied: bool,
    stats: &CensusCommitStats,
) {
    static EMITTED: AtomicUsize = AtomicUsize::new(0);
    const MAX_LINES: usize = 64;
    let n = EMITTED.fetch_add(1, Ordering::Relaxed);
    if n >= MAX_LINES {
        return;
    }
    eprintln!(
        "[census-commit] phase={phase} applied={applied} nodes={} skipped={} \
         tightened_elems={} would_have_loosened={} disjoint_kept={}",
        stats.nodes_compared,
        stats.nodes_skipped,
        stats.elems_candidate_tighter,
        stats.elems_would_loosen,
        stats.elems_disjoint_kept,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
            .expect("test tensor must be a valid interval")
    }

    fn map(entries: &[(&str, BoundedTensor)]) -> HashMap<String, BoundedTensor> {
        entries
            .iter()
            .map(|(name, bounds)| ((*name).to_string(), bounds.clone()))
            .collect()
    }

    #[test]
    fn exact_one_arms_and_everything_else_does_not() {
        assert!(census_flag_armed(Some("1")));
        assert!(!census_flag_armed(Some("0")));
        assert!(!census_flag_armed(Some("true")));
        assert!(!census_flag_armed(Some("")));
        assert!(!census_flag_armed(Some(" 1")));
        assert!(!census_flag_armed(None));
    }

    #[test]
    fn would_have_loosened_is_counted_and_eliminated_when_applied() {
        // Incumbent tighter on element 0 (both endpoints) and element 2
        // (upper only); candidate tighter on element 1.
        let incumbent = map(&[("n", bt(&[0.5, -2.0, 0.0], &[1.0, 2.0, 3.0]))]);
        let candidate = bt(&[0.0, -1.0, 0.0], &[2.0, 1.0, 4.0]);

        // Observe-only: counts reported, bounds untouched.
        let mut observed = map(&[("n", candidate.clone())]);
        let stats = shrink_only_commit(&mut observed, &incumbent, false);
        assert_eq!(stats.nodes_compared, 1);
        assert_eq!(stats.elems_would_loosen, 2);
        assert_eq!(stats.elems_candidate_tighter, 1);
        assert_eq!(stats.elems_disjoint_kept, 0);
        let untouched = observed.get("n").expect("entry retained");
        assert_eq!(untouched.lower(), candidate.lower());
        assert_eq!(untouched.upper(), candidate.upper());

        // Applied: every loosening is replaced by the intersection.
        let mut applied = map(&[("n", candidate)]);
        let stats = shrink_only_commit(&mut applied, &incumbent, true);
        assert_eq!(stats.elems_would_loosen, 2);
        let merged = applied.get("n").expect("entry retained");
        assert_eq!(merged.lower().as_slice().unwrap(), &[0.5, -1.0, 0.0]);
        assert_eq!(merged.upper().as_slice().unwrap(), &[1.0, 1.0, 3.0]);
    }

    #[test]
    fn intersection_is_never_wider_than_either_operand() {
        let incumbent = map(&[("n", bt(&[-3.0, 0.25], &[0.5, 4.0]))]);
        let mut published = map(&[("n", bt(&[-2.0, 0.0], &[1.0, 3.5]))]);
        shrink_only_commit(&mut published, &incumbent, true);
        let merged = published.get("n").expect("entry retained");
        let inc = incumbent.get("n").expect("incumbent");
        let published_before = bt(&[-2.0, 0.0], &[1.0, 3.5]);
        for i in 0..2 {
            let ml = merged.lower().as_slice().unwrap()[i];
            let mu = merged.upper().as_slice().unwrap()[i];
            let il = inc.lower().as_slice().unwrap()[i];
            let iu = inc.upper().as_slice().unwrap()[i];
            let pl = published_before.lower().as_slice().unwrap()[i];
            let pu = published_before.upper().as_slice().unwrap()[i];
            assert!(
                ml >= il.max(pl) && mu <= iu.min(pu),
                "never wider than either"
            );
            assert!(ml <= mu, "intersection must stay a valid interval");
        }
        assert_eq!(merged.lower().as_slice().unwrap(), &[-2.0, 0.25]);
        assert_eq!(merged.upper().as_slice().unwrap(), &[0.5, 3.5]);
    }

    #[test]
    fn disjoint_elements_fail_closed_to_the_published_value() {
        let incumbent = map(&[("n", bt(&[2.0], &[3.0]))]);
        let mut published = map(&[("n", bt(&[0.0], &[1.0]))]);
        let stats = shrink_only_commit(&mut published, &incumbent, true);
        assert_eq!(stats.elems_disjoint_kept, 1);
        assert_eq!(stats.elems_would_loosen, 0);
        let kept = published.get("n").expect("entry retained");
        assert_eq!(kept.lower().as_slice().unwrap(), &[0.0]);
        assert_eq!(kept.upper().as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn shape_mismatch_and_missing_nodes_are_skipped() {
        let incumbent = map(&[
            ("shape_drift", bt(&[0.0, 0.0], &[1.0, 1.0])),
            ("missing", bt(&[0.0], &[1.0])),
        ]);
        let mut published = map(&[("shape_drift", bt(&[-1.0], &[2.0]))]);
        let stats = shrink_only_commit(&mut published, &incumbent, true);
        assert_eq!(stats.nodes_compared, 0);
        assert_eq!(stats.nodes_skipped, 2);
        let kept = published.get("shape_drift").expect("entry retained");
        assert_eq!(kept.lower().as_slice().unwrap(), &[-1.0]);
        assert_eq!(kept.upper().as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn identical_maps_report_zero_movement() {
        let bounds = bt(&[-1.0, 0.0], &[1.0, 2.0]);
        let incumbent = map(&[("n", bounds.clone())]);
        let mut published = map(&[("n", bounds)]);
        let stats = shrink_only_commit(&mut published, &incumbent, true);
        assert_eq!(stats.nodes_compared, 1);
        assert_eq!(stats.elems_would_loosen, 0);
        assert_eq!(stats.elems_candidate_tighter, 0);
        assert_eq!(stats.elems_disjoint_kept, 0);
    }
}
