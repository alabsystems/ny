// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for multi-objective graph BaB orchestration.
//!
//! Extracted from `multi_objective.rs` to keep the top-level verification loop
//! focused on queue/domain flow while preserving behavior.

use ndarray::Array2;
use ny_tensor::BoundedTensor;
use std::sync::Arc;

use crate::batched_domain::CachedLinearBounds;

/// Batteries-included gate for the multi-objective GPU single-pass domain lane
/// (#w5-bab-throughput): route beta-opt-eligible BaB children through the
/// domain-batched single-pass adapter (whole-suffix sound GPU backward with the
/// inherited-β dual folded) instead of the ~3s-per-pass CPU per-child beta-opt,
/// and prune the adapter's dense spec matrix to the union of unverified
/// objectives. ON by default; opt out with `NY_MO_GPU_BATCH=0` (disable-flag
/// principle) to restore the legacy per-child lane byte-identically.
pub(in crate::beta_crown::engine::graph) fn multi_objective_gpu_single_pass_enabled() -> bool {
    !matches!(std::env::var("NY_MO_GPU_BATCH").ok().as_deref(), Some("0"))
}

/// Per-domain β OPTIMIZATION inside the GPU single-pass lane
/// (#w4-split-tightening). Default ON; `NY_MO_GPU_BETA=0` restores the
/// single-shot (inherited-β) GPU lane byte-identically.
pub(in crate::beta_crown::engine::graph) fn multi_objective_gpu_beta_enabled() -> bool {
    !matches!(std::env::var("NY_MO_GPU_BETA").ok().as_deref(), Some("0"))
}

/// #violdrop — may a freshly-bounded BaB CHILD be DISCARDED (its sub-region
/// abandoned) because its objective interval reads `upper < threshold`?
///
/// **Default: NO.** The criterion is not a proof on a β-CROWN BaB child, and
/// acting on it destroys the whole search tree.
///
/// MEASURED, vit_2023 `ibp_3_3_8_3005` (official 100 s budget, preset-only,
/// 2026-07-25): BaB is granted 90.25 s, splits the root once, and BOTH children
/// report an ORDERED (`l <= u`, not inverted) interval below the threshold —
/// `obj1=[-1.917245,-0.652679]`, `obj3=[-1.634678,-0.157248]` on one child and
/// `obj1=[-1.692895,-0.454421]`, `obj3=[-1.564326,-0.102122]` on the other,
/// every threshold `0` — against a root interval of `obj1=[-8.223450,9.859236]`.
/// Both children are dropped, the queue empties at `explored=1 verified=0
/// queue=0 max_depth=0 elapsed=2.58s`, and the run returns `unknown` with 97.1 %
/// of the BaB grant unused.
///
/// Why that cannot be a proof (GT-INDEPENDENT ARGUMENT): a ReLU split is
/// exhaustive — the active and inactive children partition the parent region —
/// so if both children's `upper` were valid, `max obj1` over the WHOLE root box
/// would be `<= -0.454 < 0`, i.e. clause 1 of the disjunction would hold at
/// EVERY point of the input box and every point would be a counterexample. The
/// same process's trusted-ORT falsification lanes evaluate the box centre (and
/// then search for 86.8 s) and find no violation, and the official result set
/// records `unsat` for this instance. So the child `upper` is not a certified
/// upper bound on the child's sub-region.
///
/// Mechanism: [`MultiObjectiveGraphBabDomain::with_constraint`] does NOT narrow
/// the child's `node_bounds` — a ReLU split is carried ONLY by the child's β
/// (Lagrangian) state. The relaxation `min f − β·s` with `β >= 0` certifies the
/// LOWER bound of the objective on the constrained sub-region; the UPPER element
/// of the same interval carries no valid certificate there.
///
/// Corroboration at scale (same instance, all five drop sites gated): BaB now
/// runs to its deadline and `2366` of the `2367` domains it explores read as
/// `any_violated` — 99.96 %, every one of them a split child, while the ROOT
/// (the one domain with no β term) does NOT. If those readings were genuine the
/// property would be violated on essentially the whole partition of the input
/// box. That "every child but never the root" signature is exactly what a
/// split-carrying β term corrupting the upper bound looks like.
///
/// Why refusing to drop cannot lose a verdict: the drop never produced `sat`.
/// It set `unresolved_due_to_violated_drop`, which forces the entire BaB run to
/// `Unknown` (`shared/state.rs::build_final_result`) even when every other
/// domain verified. Keeping the child instead leaves it a NORMAL unverified
/// frontier domain, subject to the same depth / `max_domains` / deadline caps as
/// any other domain — strictly more search, never a different verdict class, and
/// nothing is abandoned so no unresolved flag is raised.
///
/// `NY_BAB_DROP_VIOLATED_CHILD=1` restores the legacy drop for A/B measurement.
pub(in crate::beta_crown::engine::graph) fn bab_violated_child_drop_enabled() -> bool {
    parse_bab_violated_child_drop(std::env::var("NY_BAB_DROP_VIOLATED_CHILD").ok().as_deref())
}

/// Pure gate parse for [`bab_violated_child_drop_enabled`] (env-free, so the
/// decision is testable without touching process state). ONLY the exact string
/// `"1"` re-arms the legacy drop; unset, empty, `"0"` and anything malformed keep
/// the safe default (never drop).
fn parse_bab_violated_child_drop(raw: Option<&str>) -> bool {
    matches!(raw, Some("1"))
}

/// #violdrop — may a `upper < threshold` reading on THIS domain be trusted as a
/// CONCLUSIVE violation (abandoning its sub-region and poisoning the run's
/// verdict to `Unknown`)?
///
/// * `split_depth == 0` — the ROOT domain. Its objective interval comes from an
///   UNAUGMENTED CROWN / α-CROWN pass over its own region, so both ends are
///   certified and the reading is a genuine certificate. Answer: YES (unchanged
///   behavior).
/// * `split_depth > 0` — any BaB CHILD. `depth` counts split constraints, and
///   [`MultiObjectiveGraphBabDomain::with_constraint`] enforces a split ONLY
///   through the child's β (Lagrangian) state — it never narrows `node_bounds`.
///   That relaxation certifies the LOWER bound direction; the UPPER element
///   carries no valid certificate for the child's sub-region. Answer: NO. See
///   [`bab_violated_child_drop_enabled`] for the measurement and the
///   GT-independent soundness argument.
///
/// Consulted at ALL FIVE drop sites (enumerated in [`violdrop_site_probe`]) —
/// they read the same β-derived interval through the same predicate, and gating
/// only some of them just moves the queue collapse to the next one.
///
/// Scope note: the INPUT-SPLIT lanes are deliberately untouched. There the child
/// region is narrowed by actually shrinking the input box, so both ends of its
/// interval are certified and `upper < threshold` IS a proof. The one overlapping
/// sub-case — a `ReLU` whose pre-activation IS the network input, where
/// `with_constraint` narrows `input_bounds` as well as adding β — is covered
/// conservatively (not dropped), which costs search and never a verdict.
pub(in crate::beta_crown::engine::graph) fn violation_drop_is_certified(
    split_depth: usize,
) -> bool {
    split_depth == 0 || bab_violated_child_drop_enabled()
}

/// #violdrop probe (dark, `NY_VIOLDROP_PROBE=1`, print-only): announce WHICH of
/// the five drop sites fired, with the domain's split depth.
///
/// There are five independent copies of the `upper < threshold` drop in the
/// multi-objective graph lanes — the batched entry pre-filter, the batched child
/// assembly, the batched `NoUnstable` leaf, the pop-time `prefilter_batch`, and
/// `is_domain_dropped` (shared by the sequential and per-disjunct lanes). Fixing
/// them one at a time cost three full rebuild cycles because the run's exit
/// REASON is identical whichever one fires; this probe names the site so the next
/// investigation reads it off the log. Gate off => zero output.
pub(in crate::beta_crown::engine::graph) fn violdrop_site_probe(site: &str, split_depth: usize) {
    if std::env::var("NY_VIOLDROP_PROBE").ok().as_deref() == Some("1") {
        eprintln!("[violdrop-site] site={site} depth={split_depth}");
    }
}

/// Unverified objective subset for child-domain propagation.
///
/// alpha-beta-CROWN prunes verified OR-specs before later optimization passes
/// (`complete_verifier/prune.py:27-98`,
/// `complete_verifier/incomplete_verifier_func.py:277-370`). Ny keeps the
/// full bound vector on each domain for queue accounting, but child CROWN only
/// needs to revisit the still-unverified objectives.
pub(crate) struct PrunedMultiObjectiveTargets {
    pub(crate) active_indices: Vec<usize>,
    pub(crate) objectives: Vec<Vec<f32>>,
    pub(crate) thresholds: Vec<f32>,
    pub(crate) verified_mask: Vec<bool>,
}

/// Keep only the objectives that are not yet verified in the current domain.
pub(crate) fn prune_verified_multi_objective_targets(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    verified_mask: &[bool],
) -> PrunedMultiObjectiveTargets {
    debug_assert_eq!(
        objectives.len(),
        thresholds.len(),
        "prune_verified_multi_objective_targets(): objectives/thresholds mismatch"
    );
    debug_assert_eq!(
        objectives.len(),
        verified_mask.len(),
        "prune_verified_multi_objective_targets(): objectives/verified_mask mismatch"
    );

    let mut active_indices = Vec::new();
    let mut active_objectives = Vec::new();
    let mut active_thresholds = Vec::new();

    for (idx, ((objective, &threshold), &verified)) in objectives
        .iter()
        .zip(thresholds.iter())
        .zip(verified_mask.iter())
        .enumerate()
    {
        if verified {
            continue;
        }
        active_indices.push(idx);
        active_objectives.push(objective.clone());
        active_thresholds.push(threshold);
    }

    let active_verified_mask = vec![false; active_indices.len()];
    PrunedMultiObjectiveTargets {
        active_indices,
        objectives: active_objectives,
        thresholds: active_thresholds,
        verified_mask: active_verified_mask,
    }
}

/// Intersect fresh child objective bounds with the bounds inherited from the
/// parent domain (#w5-bab-throughput).
///
/// Sound: the child's sub-region is a subset of the parent's, so the parent's
/// per-objective interval also encloses the child's reachable objective values;
/// the per-objective intersection `[max(l), min(u)]` is a valid — and never
/// looser — enclosure. NaN in a fresh entry is preserved verbatim so the
/// existing NaN rejection in `update_bounds` (#2982) still fires. A numerically
/// inverted intersection (possible only from f32 slop between two sound
/// enclosures) keeps the fresh bound (sound; matches legacy behavior).
///
/// Moved here from `batched::batched_multi` (#bab-monotone-inherit) so the
/// sequential and per-disjunct BaB lanes can reuse the SAME helper instead of
/// growing a second copy; the batched call site is unchanged.
pub(in crate::beta_crown::engine::graph::multi_objective) fn tighten_child_bounds_with_parent(
    inherited: &[(f32, f32)],
    fresh: Vec<(f32, f32)>,
) -> Vec<(f32, f32)> {
    if inherited.len() != fresh.len() {
        return fresh;
    }
    fresh
        .into_iter()
        .zip(inherited.iter())
        .map(|((fl, fu), &(il, iu))| {
            if fl.is_nan() || fu.is_nan() {
                return (fl, fu);
            }
            let l = fl.max(il);
            let u = fu.min(iu);
            if l <= u {
                (l, u)
            } else {
                (fl, fu)
            }
        })
        .collect()
}

/// Lower-half-only parent inheritance, for the lanes newly covered by
/// #bab-monotone-inherit.
///
/// WHY NOT [`tighten_child_bounds_with_parent`]: that helper also does
/// `u = fu.min(iu)`, and this codebase states in writing that a BaB child's UPPER
/// end carries no certificate for the child's sub-region —
/// `multi_objective/sequential.rs`: *"A BaB child's objective interval is produced
/// with its split enforced ONLY through the β (Lagrangian) dual, which certifies
/// the LOWER bound direction — the upper end carries no certificate for the
/// child's sub-region."* That is exactly why `violation_drop_is_certified(depth)`
/// refuses every depth > 0. Inheriting a tightened upper into these lanes would
/// re-open the path `#violdrop` exists to close, and it would do so with a
/// *tighter* uncertified value — worse, not better.
///
/// The batched lane keeps calling the two-sided helper unchanged: its behaviour is
/// pre-existing and byte-identical, and narrowing it is a separate decision.
///
/// SOUND: `max(parent_l, child_l)` is valid by subset containment — the child's
/// region is a strict sub-region of the parent's, so any valid lower bound over the
/// parent is a valid lower bound over the child. The lower end is also the proof
/// side here (`verify_upper_bound` is pinned false on this subtree), so this
/// tightens exactly the direction that can convert a row.
pub(in crate::beta_crown::engine::graph::multi_objective) fn inherit_parent_lower_only(
    inherited: &[(f32, f32)],
    fresh: Vec<(f32, f32)>,
) -> Vec<(f32, f32)> {
    if inherited.len() != fresh.len() {
        return fresh;
    }
    fresh
        .into_iter()
        .zip(inherited.iter())
        .map(|((fl, fu), &(il, _iu))| {
            if fl.is_nan() || fu.is_nan() || !il.is_finite() {
                return (fl, fu);
            }
            let l = fl.max(il);
            // Never invert: if the inherited lower exceeds the child's own upper,
            // the child's interval is left exactly as computed.
            if l <= fu {
                (l, fu)
            } else {
                (fl, fu)
            }
        })
        .collect()
}

/// #bab-monotone-inherit (dark, `NY_BAB_MONOTONE_INHERIT=1`, default OFF):
/// extend the batched lane's monotone parent-bound inheritance
/// ([`tighten_child_bounds_with_parent`]) to the SEQUENTIAL and PER-DISJUNCT BaB
/// child lanes, which install their freshly-bounded objective intervals with no
/// reference to what the parent already proved.
///
/// SOUNDNESS (sub-region argument, stated explicitly): every child reaching
/// those two install sites was built by
/// [`MultiObjectiveGraphBabDomain::with_constraint`], which appends ONE ReLU/Sign
/// split constraint to the parent's history (and may only NARROW `input_bounds`).
/// The child's feasible region is therefore a subset of the parent's, so any
/// valid lower bound on the parent's region is also a valid lower bound on the
/// child's: `max(parent_l, child_l)` is sound, and symmetrically
/// `min(parent_u, child_u)` is sound. The intersection is never looser than the
/// fresh bound, so it can only help — it cannot flip a verdict from `unknown` to
/// a wrong answer, because both operands are sound enclosures of the same
/// sub-region.
///
/// Why it matters: the fresh bound source on these lanes can differ from the one
/// that produced the parent's interval (root bounds are `margin ∩ GPU ∩ IBP`),
/// so a child can REGRESS below its parent's already-proven bound and stall BaB
/// convergence — exactly the failure the batched lane already guards against.
///
/// ONLY the exact string `"1"` arms this. Unset, `"0"`, `"true"`, `" 1 "` and any
/// other malformed value keep the legacy install byte-identical.
pub(in crate::beta_crown::engine::graph::multi_objective) fn bab_monotone_inherit_enabled() -> bool
{
    parse_bab_monotone_inherit(std::env::var("NY_BAB_MONOTONE_INHERIT").ok().as_deref())
}

/// Pure gate parse for [`bab_monotone_inherit_enabled`] (env-free, so the
/// decision is testable without touching process state).
fn parse_bab_monotone_inherit(raw: Option<&str>) -> bool {
    matches!(raw, Some("1"))
}

/// Merge updated bounds for the active objective subset back into the full vector.
pub(crate) fn merge_pruned_objective_bounds(
    full_bounds: &[(f32, f32)],
    pruned_targets: &PrunedMultiObjectiveTargets,
    active_bounds: Vec<(f32, f32)>,
) -> Vec<(f32, f32)> {
    debug_assert_eq!(
        pruned_targets.active_indices.len(),
        active_bounds.len(),
        "merge_pruned_objective_bounds(): active index/bounds mismatch"
    );

    let mut merged_bounds = full_bounds.to_vec();
    for (idx, bounds) in pruned_targets
        .active_indices
        .iter()
        .copied()
        .zip(active_bounds)
    {
        merged_bounds[idx] = bounds;
    }
    merged_bounds
}

/// Select per-objective cached lA entries for the active objective subset.
pub(crate) fn prune_cached_las_for_targets<'a>(
    full_cached_las: &'a [Option<Arc<CachedLinearBounds>>],
    pruned_targets: &PrunedMultiObjectiveTargets,
) -> Vec<Option<&'a CachedLinearBounds>> {
    pruned_targets
        .active_indices
        .iter()
        .map(|&idx| full_cached_las.get(idx).and_then(Option::as_deref))
        .collect()
}

/// Merge updated caches for the active objective subset back into the full vector.
pub(crate) fn merge_pruned_cached_las(
    full_cached_las: &[Option<Arc<CachedLinearBounds>>],
    pruned_targets: &PrunedMultiObjectiveTargets,
    active_cached_las: Vec<Option<CachedLinearBounds>>,
) -> Vec<Option<Arc<CachedLinearBounds>>> {
    debug_assert_eq!(
        pruned_targets.active_indices.len(),
        active_cached_las.len(),
        "merge_pruned_cached_las(): active index/cache mismatch"
    );

    let mut merged_cached_las = full_cached_las.to_vec();
    for (idx, cache) in pruned_targets
        .active_indices
        .iter()
        .copied()
        .zip(active_cached_las)
    {
        merged_cached_las[idx] = cache.map(Arc::new);
    }
    merged_cached_las
}

/// Build a dense specification matrix for spec-guided CROWN.
///
/// Returns `None` when objective dimensions are inconsistent.
///
/// Visibility widened to `pub(in crate::beta_crown::engine::graph)` so the
/// domain-batched single-pass adapter (`batched::batched_dense_specs`) can build
/// one uniform spec matrix from the full objective set (#perf).
pub(in crate::beta_crown::engine::graph) fn build_spec_matrix(
    objectives: &[Vec<f32>],
) -> Option<Array2<f32>> {
    if objectives.is_empty() {
        return None;
    }
    let num_specs = objectives.len();
    let output_dim = objectives[0].len();
    let mut data = Vec::with_capacity(num_specs * output_dim);
    for obj in objectives {
        if obj.len() != output_dim {
            return None;
        }
        data.extend_from_slice(obj);
    }
    Array2::from_shape_vec((num_specs, output_dim), data).ok()
}

/// Convert scalar spec bounds to `(lower, upper)` tuples.
///
/// Visibility widened to `pub(in crate::beta_crown::engine::graph)` for the
/// domain-batched single-pass adapter (`batched::batched_dense_specs`).
pub(in crate::beta_crown::engine::graph) fn spec_bounds_to_vec(
    bounds: &BoundedTensor,
) -> Vec<(f32, f32)> {
    let flat = bounds.flatten();
    (0..flat.len())
        .map(|i| (flat.lower()[[i]], flat.upper()[[i]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        merge_pruned_cached_las, merge_pruned_objective_bounds, parse_bab_violated_child_drop,
        prune_cached_las_for_targets, prune_verified_multi_objective_targets,
        violation_drop_is_certified,
    };
    use crate::batched_domain::CachedLinearBounds;

    /// #violdrop: the ROOT keeps its certified drop; every β-constrained CHILD
    /// loses it. Both drop sites (batched child assembly and the pop-time
    /// prefilter) consult this one predicate, so the depth split is the whole
    /// decision.
    #[test]
    fn only_the_root_may_be_dropped_on_a_violation_reading() {
        assert!(
            violation_drop_is_certified(0),
            "the root's interval comes from an unaugmented CROWN pass — both ends certified"
        );
        for depth in 1..8 {
            assert!(
                !violation_drop_is_certified(depth),
                "a BaB child at depth {depth} carries a β-derived upper bound with no \
                 certificate for its sub-region"
            );
        }
    }

    /// #violdrop: the legacy "conclusive violation" child drop is OFF unless the
    /// gate is EXACTLY `"1"`. A malformed value must not silently re-arm a
    /// criterion that empties the BaB queue after one root split.
    #[test]
    fn violated_child_drop_is_off_unless_explicitly_armed() {
        assert!(!parse_bab_violated_child_drop(None), "unset => never drop");
        assert!(!parse_bab_violated_child_drop(Some("0")));
        assert!(!parse_bab_violated_child_drop(Some("")));
        assert!(!parse_bab_violated_child_drop(Some("true")));
        assert!(!parse_bab_violated_child_drop(Some("2")));
        assert!(
            parse_bab_violated_child_drop(Some("1")),
            "the explicit A/B opt-in restores the legacy drop"
        );
    }

    #[test]
    fn test_prune_verified_multi_objective_targets_keeps_unverified_order_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![true, false, true];

        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);

        assert_eq!(pruned.active_indices, vec![1]);
        assert_eq!(pruned.objectives, vec![vec![2.0]]);
        assert_eq!(pruned.thresholds, vec![0.2]);
        assert_eq!(pruned.verified_mask, vec![false]);
    }

    #[test]
    fn test_merge_pruned_objective_bounds_restores_verified_slots_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![true, false, true];
        let full_bounds = vec![(10.0, 11.0), (20.0, 21.0), (30.0, 31.0)];

        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);
        let merged = merge_pruned_objective_bounds(&full_bounds, &pruned, vec![(4.0, 5.0)]);

        assert_eq!(merged, vec![(10.0, 11.0), (4.0, 5.0), (30.0, 31.0)]);
    }

    #[test]
    fn test_prune_cached_las_for_targets_keeps_active_alignment_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![false, true, false];
        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);

        let mut cache0 = CachedLinearBounds::default();
        cache0
            .lower_b
            .insert("relu0".to_string(), ndarray::arr1(&[1.0]));
        let mut cache2 = CachedLinearBounds::default();
        cache2
            .lower_b
            .insert("relu2".to_string(), ndarray::arr1(&[3.0]));

        let full_cached_las = vec![Some(Arc::new(cache0)), None, Some(Arc::new(cache2))];
        let active_cached_las = prune_cached_las_for_targets(&full_cached_las, &pruned);

        assert_eq!(active_cached_las.len(), 2);
        assert_eq!(
            active_cached_las[0]
                .and_then(|cache| cache.lower_b.get("relu0"))
                .map(|bias| bias[0]),
            Some(1.0)
        );
        assert_eq!(
            active_cached_las[1]
                .and_then(|cache| cache.lower_b.get("relu2"))
                .map(|bias| bias[0]),
            Some(3.0)
        );
    }

    #[test]
    fn test_merge_pruned_cached_las_restores_verified_slots_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![true, false, true];
        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);

        let mut full_cache0 = CachedLinearBounds::default();
        full_cache0
            .lower_b
            .insert("relu0".to_string(), ndarray::arr1(&[10.0]));
        let mut active_cache = CachedLinearBounds::default();
        active_cache
            .lower_b
            .insert("relu1".to_string(), ndarray::arr1(&[20.0]));

        let inherited = Arc::new(full_cache0);
        let merged = merge_pruned_cached_las(
            &[Some(Arc::clone(&inherited)), None, None],
            &pruned,
            vec![Some(active_cache.clone())],
        );

        assert_eq!(merged.len(), 3);
        assert!(Arc::ptr_eq(
            merged[0].as_ref().expect("inherited cache should remain"),
            &inherited,
        ));
        assert_eq!(
            merged[0]
                .as_ref()
                .and_then(|cache| cache.lower_b.get("relu0"))
                .map(|bias| bias[0].to_bits()),
            Some(10.0_f32.to_bits())
        );
        assert_eq!(
            merged[1]
                .as_ref()
                .and_then(|cache| cache.lower_b.get("relu1"))
                .map(|bias| bias[0].to_bits()),
            Some(20.0_f32.to_bits())
        );
        assert!(merged[2].is_none());
    }
}

#[cfg(test)]
mod monotone_merge_tests {
    use super::{
        merge_pruned_objective_bounds, parse_bab_monotone_inherit,
        prune_verified_multi_objective_targets, tighten_child_bounds_with_parent,
    };

    /// #bab-monotone-inherit: the EXACT composition now installed on the
    /// sequential and per-disjunct lanes — `merge_pruned_objective_bounds` first,
    /// then `tighten_child_bounds_with_parent` against the same inherited vector.
    ///
    /// Two invariants the install depends on: (a) an already-VERIFIED slot, which
    /// `merge` restores verbatim from the parent, must come back bit-identical —
    /// intersecting an interval with itself is the identity, so the gate cannot
    /// perturb a verified objective; (b) an ACTIVE slot never gets LOOSER than the
    /// fresh bound.
    #[test]
    fn merge_then_tighten_preserves_verified_slots_and_never_loosens() {
        let objectives = vec![vec![1.0_f32], vec![2.0], vec![3.0]];
        let thresholds = vec![0.0_f32, 0.0, 0.0];
        // obj0 and obj2 already verified => only obj1 is re-bounded.
        let verified_mask = vec![true, false, true];
        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);

        let inherited = vec![(0.25_f32, 4.0_f32), (-0.06, 5.0), (-9.0, 9.0)];
        // The child's fresh pass REGRESSED below what the parent already proved.
        let active_bounds = vec![(-0.5_f32, 4.0_f32)];

        let merged = merge_pruned_objective_bounds(&inherited, &pruned, active_bounds);
        let tightened = tighten_child_bounds_with_parent(&inherited, merged);

        // (a) verified slots are untouched.
        assert_eq!(tightened[0], inherited[0]);
        assert_eq!(tightened[2], inherited[2]);
        // (b) the active slot recovers the parent's lower bound and keeps the
        // tighter fresh upper — never looser than either operand.
        assert_eq!(tightened[1], (-0.06, 4.0));
    }

    /// Sub-region argument (#w5-bab-throughput): the child's region is a subset
    /// of the parent's, so per-objective intersection with the inherited bounds
    /// is sound and never looser. Fresh-tighter, parent-tighter, and mixed
    /// entries must each resolve to the elementwise-tightest interval.
    #[test]
    fn tighten_child_bounds_takes_elementwise_tightest_w5() {
        let inherited = [(-0.06_f32, 5.0_f32), (-3.0, 2.0), (-1.0, 1.0)];
        let fresh = vec![(-0.5_f32, 4.0_f32), (-2.5, 3.0), (-1.0, 1.0)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh);
        // obj0: parent lower (-0.06) beats fresh (-0.5); fresh upper (4.0) beats parent.
        assert_eq!(merged[0], (-0.06, 4.0));
        // obj1: fresh lower tighter, parent upper tighter.
        assert_eq!(merged[1], (-2.5, 2.0));
        // obj2: identical stays identical.
        assert_eq!(merged[2], (-1.0, 1.0));
    }

    /// NaN in a fresh entry must survive verbatim so `update_bounds` (#2982)
    /// still rejects the child as numerically corrupted — silently masking a
    /// failed pass with the parent's bound would hide the corruption.
    #[test]
    fn tighten_child_bounds_preserves_fresh_nan_w5() {
        let inherited = [(0.0_f32, 1.0_f32)];
        let fresh = vec![(f32::NAN, 1.0_f32)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh);
        assert!(merged[0].0.is_nan(), "NaN must propagate to update_bounds");
    }

    /// A numerically inverted intersection (only possible from f32 slop between
    /// two sound enclosures) keeps the fresh bound — matching what the legacy
    /// lane would have reported.
    #[test]
    fn tighten_child_bounds_keeps_fresh_on_inverted_intersection_w5() {
        let inherited = [(0.5_f32, 0.6_f32)];
        let fresh = vec![(0.7_f32, 0.9_f32)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh);
        assert_eq!(merged[0], (0.7, 0.9));
    }

    /// Length mismatch (defensive) returns the fresh bounds unchanged.
    #[test]
    fn tighten_child_bounds_length_mismatch_returns_fresh_w5() {
        let inherited = [(0.0_f32, 1.0_f32)];
        let fresh = vec![(0.1_f32, 0.9_f32), (0.2, 0.8)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh.clone());
        assert_eq!(merged, fresh);
    }

    /// #bab-monotone-inherit: ONLY the exact string `"1"` arms the extension to
    /// the sequential / per-disjunct lanes.
    #[test]
    fn monotone_inherit_gate_arms_only_on_exact_one() {
        assert!(parse_bab_monotone_inherit(Some("1")));
    }

    /// Mandatory malformed-value test (house rule): no near-miss spelling may
    /// silently arm the gate, so an unset/garbled env keeps both lanes
    /// byte-identical to today.
    #[test]
    fn monotone_inherit_gate_does_not_arm_on_malformed_values() {
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("on"),
            Some(" 1"),
            Some("1 "),
            Some(" 1 "),
            Some("01"),
            Some("1.0"),
            Some("11"),
            Some("-1"),
        ] {
            assert!(
                !parse_bab_monotone_inherit(raw),
                "NY_BAB_MONOTONE_INHERIT={raw:?} must NOT arm the gate"
            );
        }
    }
}

#[cfg(test)]
mod lower_only_inherit_tests {
    use super::inherit_parent_lower_only;

    #[test]
    fn raises_the_lower_end_and_never_touches_the_upper() {
        // The upper end carries no certificate for a child's sub-region, so it must
        // pass through untouched even when the parent's upper is tighter.
        let parent = [(1.0f32, 2.0f32), (-1.0, 5.0)];
        let child = vec![(0.0f32, 9.0f32), (-3.0, 9.0)];
        let out = inherit_parent_lower_only(&parent, child);
        assert_eq!(
            out,
            vec![(1.0, 9.0), (-1.0, 9.0)],
            "lower raised, upper kept"
        );
    }

    #[test]
    fn never_inverts_an_interval() {
        // Inherited lower above the child's own upper must leave the child alone
        // rather than produce lo > hi.
        let parent = [(100.0f32, 200.0f32)];
        let child = vec![(0.0f32, 1.0f32)];
        assert_eq!(inherit_parent_lower_only(&parent, child), vec![(0.0, 1.0)]);
    }

    #[test]
    fn passes_through_on_nan_non_finite_and_length_mismatch() {
        let nan_child = vec![(f32::NAN, 1.0f32)];
        assert!(inherit_parent_lower_only(&[(0.0, 2.0)], nan_child)[0]
            .0
            .is_nan());

        let inf_parent = [(f32::NEG_INFINITY, 2.0f32)];
        assert_eq!(
            inherit_parent_lower_only(&inf_parent, vec![(0.0, 1.0)]),
            vec![(0.0, 1.0)],
            "non-finite inherited lower must not be adopted"
        );

        assert_eq!(
            inherit_parent_lower_only(&[(0.0, 1.0)], vec![(2.0, 3.0), (4.0, 5.0)]),
            vec![(2.0, 3.0), (4.0, 5.0)],
            "length mismatch falls through"
        );
    }

    #[test]
    fn is_weakly_tighter_than_the_child_alone() {
        // Selection property: the result's lower end is always >= the child's own,
        // so this can only ever help the proof side.
        let parent = [(0.5f32, 9.0f32), (-2.0, 9.0), (7.0, 9.0)];
        let child = vec![(0.0f32, 1.0f32), (0.0, 1.0), (0.0, 1.0)];
        for (i, (lo, _)) in inherit_parent_lower_only(&parent, child.clone())
            .into_iter()
            .enumerate()
        {
            assert!(lo >= child[i].0, "row {i}: {lo} must be >= {}", child[i].0);
        }
    }
}
