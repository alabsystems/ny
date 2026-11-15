// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-child-seed`: seed one BaB child's complete-cGAN-cascade baseline
//! with the domain's INHERITED intermediate map.
//!
//! The child cascade (`#cgan-child-interm-refine`) starts from the certified
//! forward-linear map on the child's own box. On the cGAN discriminator that
//! map is roughly ten times looser than CROWN, so the sparse ReLU-row selector
//! walks every row it leaves unstable (7,680 on cgan2026 row 7) instead of the
//! rows the ROOT CROWN map already leaves unstable (at most 694). Intersecting
//! the two before the cascade runs is what this module does.
//!
//! # Soundness
//!
//! Every operand is a certified enclosure of the SAME reachable set:
//!
//! * the forward-linear baseline is computed on the child box itself;
//! * the root CROWN map encloses the root box's image, and the root box
//!   contains the child box (BaB children only ever split or clip it);
//! * a child-local override is, by the contract its producers already rely
//!   on for the rebound, an enclosure of the child box's image.
//!
//! The elementwise intersection of enclosures of one set is an enclosure of
//! that set, and the cascade downstream only ever SHRINKS the baseline it is
//! handed. Merging is fail-closed per NODE: a shape mismatch, a NaN on either
//! side, or a single disjoint element (which two valid enclosures of a
//! non-empty set cannot produce, so it can only mean one operand is wrong)
//! leaves that node's baseline entry byte-identical. A node the baseline does
//! not carry is never added, so the cascade sees exactly the key set it would
//! have seen unseeded.

use std::collections::HashMap;

use ndarray::Zip;
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};

/// Telemetry for one seeded child cascade. Carries no proof authority.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChildSeedReport {
    /// Baseline nodes intersected with an inherited entry.
    pub(crate) nodes_seeded: usize,
    /// Baseline nodes whose inherited entry was mismatched, NaN, or disjoint
    /// and which therefore kept their baseline entry unchanged.
    pub(crate) nodes_declined: usize,
    /// Elements whose interval strictly shrank under the seed.
    pub(crate) elems_tightened: usize,
    /// Sign-unstable ReLU pre-activation rows (`l < 0 < u`) under the
    /// UNSEEDED forward-linear baseline: what the sparse selector would have
    /// walked without the seed.
    pub(crate) unstable_rows_unseeded: usize,
    /// The same count under the SEEDED baseline the cascade actually receives.
    pub(crate) unstable_rows_seeded: usize,
}

/// Outcome of one [`intersect_parent_seed_into_baseline`] call.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SeedMergeStats {
    pub(crate) nodes_seeded: usize,
    pub(crate) nodes_declined: usize,
    pub(crate) elems_tightened: usize,
}

/// Intersect `seed` into `baseline` in place, node by node, shrink-only.
///
/// Only keys already present in `baseline` are touched; see the module doc for
/// why every decline keeps the baseline entry byte-identical.
pub(crate) fn intersect_parent_seed_into_baseline(
    baseline: &mut HashMap<String, BoundedTensor>,
    seed: &HashMap<String, BoundedTensor>,
) -> SeedMergeStats {
    let mut stats = SeedMergeStats::default();
    for (name, current) in baseline.iter_mut() {
        let Some(candidate) = seed.get(name) else {
            continue;
        };
        match intersect_entry(current, candidate) {
            Some((merged, tightened)) => {
                *current = merged;
                stats.nodes_seeded += 1;
                stats.elems_tightened += tightened;
            }
            None => stats.nodes_declined += 1,
        }
    }
    stats
}

/// `[max(l_b, l_s), min(u_b, u_s)]` elementwise, or `None` (keep the baseline)
/// on a shape mismatch, any NaN, or any disjoint element.
fn intersect_entry(
    current: &BoundedTensor,
    candidate: &BoundedTensor,
) -> Option<(BoundedTensor, usize)> {
    if current.shape() != candidate.shape() {
        return None;
    }
    let mut lower = current.lower().clone();
    let mut upper = current.upper().clone();
    let mut tightened = 0usize;
    let mut admissible = true;
    Zip::from(lower.view_mut())
        .and(upper.view_mut())
        .and(candidate.lower())
        .and(candidate.upper())
        .for_each(|lo, hi, &cand_lo, &cand_hi| {
            if !admissible {
                return;
            }
            let (base_lo, base_hi) = (*lo, *hi);
            if base_lo.is_nan() || base_hi.is_nan() || cand_lo.is_nan() || cand_hi.is_nan() {
                admissible = false;
                return;
            }
            let merged_lo = base_lo.max(cand_lo);
            let merged_hi = base_hi.min(cand_hi);
            if merged_lo > merged_hi {
                admissible = false;
                return;
            }
            if merged_lo > base_lo || merged_hi < base_hi {
                tightened += 1;
            }
            *lo = merged_lo;
            *hi = merged_hi;
        });
    if !admissible {
        return None;
    }
    let merged = BoundedTensor::new_allow_infinite(lower, upper).ok()?;
    Some((merged, tightened))
}

/// Number of sign-unstable rows (`l < 0 < u`) over every ReLU's
/// pre-activation entry in `map`, in `exec_order`.
///
/// This is the row set the typed sparse ReLU-row selector seeds a target's
/// backward walk with when its plan engages; a target that exceeds the
/// selector's `max_frac` ceiling is walked full width instead, so this is a
/// count of what the SEED makes walkable, not a meter of walked rows.
pub(crate) fn count_unstable_relu_rows(
    graph: &GraphNetwork,
    map: &HashMap<String, BoundedTensor>,
    exec_order: &[String],
) -> usize {
    let mut seen: Vec<&str> = Vec::new();
    let mut total = 0usize;
    for name in exec_order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        let Ok(input) = node.require_unary_input() else {
            continue;
        };
        if input == NETWORK_INPUT || seen.contains(&input) {
            continue;
        }
        seen.push(input);
        let Some(bounds) = map.get(input) else {
            continue;
        };
        total += bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .filter(|(lo, hi)| **lo < 0.0 && **hi > 0.0)
            .count();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::{count_unstable_relu_rows, intersect_parent_seed_into_baseline, SeedMergeStats};
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::network::GraphNode;
    use crate::{GraphNetwork, Layer};
    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;

    fn interval(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("upper"),
        )
        .expect("interval")
    }

    fn bits(bound: &BoundedTensor) -> Vec<(u32, u32)> {
        bound
            .lower()
            .iter()
            .zip(bound.upper().iter())
            .map(|(lo, hi)| (lo.to_bits(), hi.to_bits()))
            .collect()
    }

    /// Every fail-closed arm keeps the baseline entry BYTE-identical, a node
    /// only the seed carries is never added, and the one admissible node is
    /// the exact elementwise intersection.
    #[test]
    fn child_seed_disjoint_or_mismatch_keeps_current_entry() {
        let mut baseline: HashMap<String, BoundedTensor> = HashMap::new();
        baseline.insert("ok".into(), interval(&[-1.0, -1.0, 0.0], &[1.0, 1.0, 2.0]));
        baseline.insert(
            "shape".into(),
            interval(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]),
        );
        baseline.insert("disjoint".into(), interval(&[-1.0, 0.0], &[1.0, 0.5]));
        baseline.insert("nan".into(), interval(&[-1.0, -1.0], &[1.0, 1.0]));
        baseline.insert("infeasible".into(), interval(&[-1.0, -1.0], &[1.0, 1.0]));
        baseline.insert("baseline_only".into(), interval(&[-3.0], &[3.0]));
        let before = baseline.clone();

        let mut seed: HashMap<String, BoundedTensor> = HashMap::new();
        seed.insert("ok".into(), interval(&[-0.5, -2.0, 0.5], &[0.5, 0.5, 1.0]));
        seed.insert("shape".into(), interval(&[-0.1, -0.1], &[0.1, 0.1]));
        // Element 1 is disjoint ([0.6, 0.9] vs [0.0, 0.5]); element 0 would
        // have tightened. The WHOLE node must be declined, not just element 1.
        seed.insert("disjoint".into(), interval(&[-0.2, 0.6], &[0.2, 0.9]));
        // NaN is unrepresentable through the checked constructors, so the
        // guard is exercised through the test-only unchecked one.
        seed.insert(
            "nan".into(),
            BoundedTensor::new_unchecked(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.5_f32, f32::NAN]).expect("nan lower"),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5_f32, 0.5]).expect("nan upper"),
            )
            .expect("unchecked NaN entry"),
        );
        // The canonical infeasible sentinel `(+inf, -inf)` in an inherited
        // entry must decline (keep the baseline), never publish an empty box.
        let mut infeasible = interval(&[-0.5, -0.5], &[0.5, 0.5]);
        infeasible.mark_infeasible_all();
        seed.insert("infeasible".into(), infeasible);
        seed.insert("seed_only".into(), interval(&[-0.1], &[0.1]));

        let stats = intersect_parent_seed_into_baseline(&mut baseline, &seed);

        assert!(
            !baseline.contains_key("seed_only"),
            "a node the baseline does not carry must never be ADDED by the seed"
        );
        for name in ["shape", "disjoint", "nan", "infeasible", "baseline_only"] {
            assert_eq!(
                bits(&baseline[name]),
                bits(&before[name]),
                "node '{name}' must keep its baseline entry byte-identical"
            );
        }
        let merged = &baseline["ok"];
        let lower: Vec<f32> = merged.lower().iter().copied().collect();
        let upper: Vec<f32> = merged.upper().iter().copied().collect();
        assert_eq!(lower, vec![-0.5, -1.0, 0.5]);
        assert_eq!(upper, vec![0.5, 0.5, 1.0]);
        assert_eq!(
            stats,
            SeedMergeStats {
                nodes_seeded: 1,
                nodes_declined: 4,
                elems_tightened: 3,
            }
        );
    }

    /// A vacuous (infinite) baseline entry is narrowed by a finite seed, and a
    /// finite baseline is never widened by a vacuous seed.
    #[test]
    fn child_seed_infinite_operands_only_ever_narrow() {
        let mut baseline: HashMap<String, BoundedTensor> = HashMap::new();
        baseline.insert(
            "vacuous".into(),
            interval(&[f32::NEG_INFINITY], &[f32::INFINITY]),
        );
        baseline.insert("finite".into(), interval(&[-1.0], &[2.0]));
        let mut seed: HashMap<String, BoundedTensor> = HashMap::new();
        seed.insert("vacuous".into(), interval(&[-1.5], &[0.25]));
        seed.insert(
            "finite".into(),
            interval(&[f32::NEG_INFINITY], &[f32::INFINITY]),
        );
        let stats = intersect_parent_seed_into_baseline(&mut baseline, &seed);
        assert_eq!(stats.nodes_seeded, 2);
        assert_eq!(stats.elems_tightened, 1);
        assert_eq!(baseline["vacuous"].lower()[[0]], -1.5);
        assert_eq!(baseline["vacuous"].upper()[[0]], 0.25);
        assert_eq!(baseline["finite"].lower()[[0]], -1.0);
        assert_eq!(baseline["finite"].upper()[[0]], 2.0);
    }

    /// Randomised shrink-only + containment: for every admissible node the
    /// result lies inside BOTH operands, and every point inside both operands
    /// (sampled) lies inside the result — i.e. it is exactly the intersection,
    /// so it can exclude no point either enclosure admits.
    #[test]
    fn child_seed_merge_is_exact_intersection_on_random_intervals() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        let mut checked_points = 0usize;
        for trial in 0..200 {
            let n = 1 + trial % 7;
            let mut base_lo = Vec::with_capacity(n);
            let mut base_hi = Vec::with_capacity(n);
            let mut seed_lo = Vec::with_capacity(n);
            let mut seed_hi = Vec::with_capacity(n);
            for _ in 0..n {
                // A common reachable point, so both operands are enclosures
                // of a non-empty set and are never disjoint.
                let x = 4.0 * next() - 2.0;
                base_lo.push(x - 3.0 * next());
                base_hi.push(x + 3.0 * next());
                seed_lo.push(x - 3.0 * next());
                seed_hi.push(x + 3.0 * next());
            }
            let mut baseline = HashMap::new();
            baseline.insert("t".to_string(), interval(&base_lo, &base_hi));
            let mut seed = HashMap::new();
            seed.insert("t".to_string(), interval(&seed_lo, &seed_hi));
            let stats = intersect_parent_seed_into_baseline(&mut baseline, &seed);
            assert_eq!(stats.nodes_seeded, 1, "overlapping operands must merge");
            let merged = &baseline["t"];
            for k in 0..n {
                let (lo, hi) = (merged.lower()[[k]], merged.upper()[[k]]);
                assert!(
                    lo >= base_lo[k] && hi <= base_hi[k],
                    "merge widened the baseline"
                );
                assert!(
                    lo >= seed_lo[k] && hi <= seed_hi[k],
                    "merge widened the seed"
                );
                for _ in 0..50 {
                    let p = base_lo[k].max(seed_lo[k])
                        + next() * (base_hi[k].min(seed_hi[k]) - base_lo[k].max(seed_lo[k]));
                    let p = p.clamp(base_lo[k].max(seed_lo[k]), base_hi[k].min(seed_hi[k]));
                    assert!(
                        lo <= p && p <= hi,
                        "point {p} admitted by BOTH enclosures was EXCLUDED by the merge \
                         [{lo}, {hi}] — a false proof, not a tighter bound"
                    );
                    checked_points += 1;
                }
            }
        }
        assert!(
            checked_points > 10_000,
            "sampling did not run ({checked_points})"
        );
    }

    /// Only ReLU PRE-activation entries count, only sign-straddling rows
    /// count, and a pre-activation shared by two ReLUs counts once.
    #[test]
    fn unstable_relu_rows_counts_only_sign_straddling_relu_inputs() {
        let fc = LinearLayer::new(
            Array2::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("weights"),
            Some(Array1::from_vec(vec![0.0_f32, 0.0, 0.0])),
        )
        .expect("linear layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("fc", Layer::Linear(fc)));
        graph.add_node(GraphNode::new(
            "fc_relu",
            Layer::ReLU(ReLULayer),
            vec!["fc".to_string()],
        ));
        graph.set_output("fc_relu");
        let exec_order = graph.exec_order().expect("exec order").to_vec();

        let mut map = HashMap::new();
        // Row 0 unstable, row 1 stable-active, row 2 unstable.
        map.insert(
            "fc".to_string(),
            interval(&[-1.0, 0.5, -0.1], &[1.0, 2.0, 0.1]),
        );
        // The ReLU OUTPUT entry must not be counted even if it straddled.
        map.insert(
            "fc_relu".to_string(),
            interval(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]),
        );
        assert_eq!(count_unstable_relu_rows(&graph, &map, &exec_order), 2);

        // A row touching zero on one side is NOT sign-unstable.
        map.insert(
            "fc".to_string(),
            interval(&[0.0, -1.0, -0.1], &[1.0, 0.0, 0.1]),
        );
        assert_eq!(count_unstable_relu_rows(&graph, &map, &exec_order), 1);
    }
}
