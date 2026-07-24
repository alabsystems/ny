// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN increment C1: split-strengthened k-ReLU cut bound
//! derivation (`docs/CERTIFIED_CUT_CROWN_DESIGN.md`).
//!
//! A *k-ReLU cut* over a group of same-layer neurons with affine
//! pre-activations `z_i(x) = w_i·x + r_i` on the input box `[xl, xu]` is
//!
//! ```text
//!   Σ_i cc_i · relu(z_i(x)) ≤ B        (cc_i ≥ 0)
//! ```
//!
//! whose bound `B` is DERIVED (not assumed) as the max over activation
//! patterns `S ⊆ G` of the box-max of the affine form `Σ_{i∈S} cc_i·z_i(x)`.
//! The box-max of an affine form is analytic per coordinate
//! (`Σ_j max(w_j·xl_j, w_j·xu_j) + r`), so no corner enumeration is needed.
//!
//! **Split strengthening (the novel lever):** on a BaB subdomain where split
//! constraints force `Act` active (`z_i ≥ 0`) and `Inact` inactive
//! (`z_i ≤ 0`), only split-CONSISTENT patterns (`Act ⊆ S`, `S ∩ Inact = ∅`)
//! need dominating — the re-derived `B` is (weakly, often strictly) tighter.
//!
//! Soundness schema (Lean, supplied by the exact Lake-pinned Clean module
//! `Crownproof.MultiReluCutK`):
//! `multiReluCut_pattern_dominance` / `multiReluCut_box_le` (proven, ∀k∀n)
//! ground the pattern-max derivation of `B`, and `multiReluCut_bridge` folds
//! the cut into the Farkas combination with λ≥0 multipliers. The
//! split-strengthened bound is the same dominance argument restricted to the
//! split-consistent patterns (`Act ⊆ S`, `S ∩ Inact = ∅`); its strictness
//! (B 1 → 1/2) is mirrored in the tests below.
//! Rounding: accumulation in f64 with a final `next_up`
//! outward step — `B` too LARGE is merely a looser (still valid) cut; only a
//! too-small `B` would be unsound, so all rounding here goes UP.

/// One neuron's affine pre-activation over the (flattened) input box:
/// `z(x) = Σ_j w[j].1 · x[w[j].0] + r` (sparse rows — conv receptive fields).
#[derive(Debug, Clone)]
pub struct AffineRow {
    /// Sparse weights as `(input_index, weight)`, indices strictly increasing.
    pub w: Vec<(u32, f32)>,
    /// Offset (bias) of the pre-activation.
    pub r: f32,
}

/// A derived k-ReLU cut over a neuron group (same ReLU layer).
#[derive(Debug, Clone)]
pub struct MultiReluCut {
    /// Flat neuron indices of the group within its ReLU layer.
    pub neurons: Vec<u32>,
    /// Nonnegative cut weights (`cc`), same length as `neurons`.
    pub cc: Vec<f32>,
    /// The derived bound `B` (outward-rounded).
    pub bound: f32,
}

/// Split state of a group neuron on the current subdomain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitState {
    /// Unsplit — both activation patterns admissible.
    Free,
    /// Split-active premise (`z ≥ 0`): patterns must CONTAIN this neuron.
    Active,
    /// Split-inactive premise (`z ≤ 0`): patterns must EXCLUDE this neuron.
    Inactive,
}

/// Box-max of the affine form `Σ_{i∈S} cc_i·z_i(x)` on `[xl, xu]`,
/// f64-accumulated, rounded UP (outward for an upper bound).
///
/// `S` is given as a bitmask over `rows` (k ≤ 32; practical k ≤ 4).
fn pattern_box_max(rows: &[AffineRow], cc: &[f32], mask: u32, xl: &[f32], xu: &[f32]) -> f64 {
    // Merged sparse weights of the pattern's combined affine form. Group
    // receptive fields overlap heavily (adjacent conv neurons), so merge via
    // a small sorted map.
    let mut merged: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    let mut offset = 0.0f64;
    for (i, row) in rows.iter().enumerate() {
        if mask & (1 << i) == 0 {
            continue;
        }
        let c = f64::from(cc[i]);
        offset += c * f64::from(row.r);
        for &(j, wj) in &row.w {
            *merged.entry(j).or_insert(0.0) += c * f64::from(wj);
        }
    }
    let mut total = offset;
    for (&j, &wj) in &merged {
        let (lo, hi) = (f64::from(xl[j as usize]), f64::from(xu[j as usize]));
        total += (wj * lo).max(wj * hi);
    }
    total
}

/// Derive the k-ReLU cut bound `B` over the box, restricted to
/// split-consistent activation patterns.
///
/// Returns the outward-rounded (`next_up`) f32 bound. The empty pattern is
/// admissible only when `Act = ∅`; its value is `0` (the ReLU sum's floor
/// when everything is inactive), which the pattern max naturally includes
/// via `mask = 0` (`pattern_box_max` returns `0.0` there) whenever no neuron
/// is forced active.
///
/// Soundness: mirrors `multiReluCut_split_box_le` — `B` dominates every
/// split-consistent pattern's affine form everywhere on the box (per-
/// coordinate corner max), hence dominates `Σ cc·relu(z)` on the subdomain.
pub fn derive_cut_bound(
    rows: &[AffineRow],
    cc: &[f32],
    splits: &[SplitState],
    xl: &[f32],
    xu: &[f32],
) -> f32 {
    let k = rows.len();
    debug_assert!(k <= 16, "cut group too large (k={k})");
    debug_assert_eq!(cc.len(), k);
    debug_assert_eq!(splits.len(), k);
    debug_assert!(cc.iter().all(|&c| c >= 0.0), "cut weights must be ≥ 0");

    let act_mask: u32 = splits
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == SplitState::Active)
        .fold(0, |m, (i, _)| m | (1 << i));
    let inact_mask: u32 = splits
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == SplitState::Inactive)
        .fold(0, |m, (i, _)| m | (1 << i));

    let mut best = f64::NEG_INFINITY;
    for mask in 0..(1u32 << k) {
        // Split-consistency: Act ⊆ S, S ∩ Inact = ∅.
        if mask & act_mask != act_mask || mask & inact_mask != 0 {
            continue;
        }
        let v = pattern_box_max(rows, cc, mask, xl, xu);
        if v > best {
            best = v;
        }
    }
    // Outward: a larger B is a looser but still valid cut.
    ny_tensor::next_up_f32(best as f32)
}

/// Convenience: the unconditional (root) bound — all neurons `Free`.
pub fn derive_cut_bound_root(rows: &[AffineRow], cc: &[f32], xl: &[f32], xu: &[f32]) -> f32 {
    derive_cut_bound(rows, cc, &vec![SplitState::Free; rows.len()], xl, xu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// Exact ReLU-sum evaluation at a point.
    fn relu_sum_at(rows: &[AffineRow], cc: &[f32], x: &[f64]) -> f64 {
        rows.iter()
            .zip(cc)
            .map(|(row, &c)| {
                let z: f64 = f64::from(row.r)
                    + row
                        .w
                        .iter()
                        .map(|&(j, w)| f64::from(w) * x[j as usize])
                        .sum::<f64>();
                f64::from(c) * relu(z)
            })
            .sum()
    }

    /// The Lean `split_strengthening_strict` witness, exactly:
    /// box x ∈ [-1,1], z0 = x, z1 = -x - 1/2, cc = (1,1).
    /// Unconditional B = 1; under Act = {1} the re-derived B = 1/2.
    #[test]
    fn strengthening_witness_matches_lean() {
        let rows = vec![
            AffineRow {
                w: vec![(0, 1.0)],
                r: 0.0,
            },
            AffineRow {
                w: vec![(0, -1.0)],
                r: -0.5,
            },
        ];
        let cc = [1.0, 1.0];
        let (xl, xu) = ([-1.0f32], [1.0f32]);

        let b_root = derive_cut_bound_root(&rows, &cc, &xl, &xu);
        // Patterns: {} → 0, {0} → max(x) = 1, {1} → max(-x) - 1/2 = 1/2,
        // {0,1} → -1/2.  Root B = 1 (+1 ulp outward).
        assert!((f64::from(b_root) - 1.0).abs() < 1e-6, "b_root = {b_root}");

        let b_act1 = derive_cut_bound(
            &rows,
            &cc,
            &[SplitState::Free, SplitState::Active],
            &xl,
            &xu,
        );
        // Admissible patterns: {1} → 1/2, {0,1} → -1/2.  B' = 1/2 < 1.
        assert!((f64::from(b_act1) - 0.5).abs() < 1e-6, "b_act1 = {b_act1}");
        assert!(b_act1 < b_root, "split strengthening must tighten");
    }

    /// Brute-force oracle: on a small box, the derived root bound dominates
    /// the exact ReLU sum at every corner and at a grid of interior points.
    #[test]
    fn root_bound_dominates_exact_relu_sum() {
        // 3 neurons over a 2-D box — the k=3 genuine-coupling demo geometry
        // from MultiReluCutK.lean: z1 = x1, z2 = -x1 + 2 x2, z3 = -x1 - 2 x2
        // on [-1,1]^2 (joint bound 3).
        let rows = vec![
            AffineRow {
                w: vec![(0, 1.0)],
                r: 0.0,
            },
            AffineRow {
                w: vec![(0, -1.0), (1, 2.0)],
                r: 0.0,
            },
            AffineRow {
                w: vec![(0, -1.0), (1, -2.0)],
                r: 0.0,
            },
        ];
        let cc = [1.0, 1.0, 1.0];
        let (xl, xu) = ([-1.0f32, -1.0], [1.0f32, 1.0]);
        let b = derive_cut_bound_root(&rows, &cc, &xl, &xu);
        assert!(
            (f64::from(b) - 3.0).abs() < 1e-6,
            "k=3 demo joint bound is 3, got {b}"
        );

        // Exhaustive grid (includes corners).
        let steps = 9;
        for a in 0..=steps {
            for bb in 0..=steps {
                let x = [
                    -1.0 + 2.0 * (a as f64) / (steps as f64),
                    -1.0 + 2.0 * (bb as f64) / (steps as f64),
                ];
                let v = relu_sum_at(&rows, &cc, &x);
                assert!(v <= f64::from(b) + 1e-9, "cut violated at {x:?}: {v} > {b}");
            }
        }
    }

    /// Split-restricted bounds dominate the ReLU sum on the premise-restricted
    /// region, and are monotone: more premises ⇒ tighter-or-equal B.
    #[test]
    fn split_bound_sound_on_subdomain_and_monotone() {
        let rows = vec![
            AffineRow {
                w: vec![(0, 1.0)],
                r: 0.0,
            },
            AffineRow {
                w: vec![(0, -1.0), (1, 2.0)],
                r: 0.0,
            },
            AffineRow {
                w: vec![(0, -1.0), (1, -2.0)],
                r: 0.0,
            },
        ];
        let cc = [1.0, 1.0, 1.0];
        let (xl, xu) = ([-1.0f32, -1.0], [1.0f32, 1.0]);
        let b_root = derive_cut_bound_root(&rows, &cc, &xl, &xu);

        // Premise: neuron 1 inactive (z2 ≤ 0) — patterns exclude 1.
        let b_in1 = derive_cut_bound(
            &rows,
            &cc,
            &[SplitState::Free, SplitState::Inactive, SplitState::Free],
            &xl,
            &xu,
        );
        assert!(
            b_in1 <= b_root,
            "restriction can only tighten: {b_in1} vs {b_root}"
        );

        // Soundness on the restricted region (z2 ≤ 0): grid-check.
        let steps = 9;
        for a in 0..=steps {
            for bb in 0..=steps {
                let x = [
                    -1.0 + 2.0 * (a as f64) / (steps as f64),
                    -1.0 + 2.0 * (bb as f64) / (steps as f64),
                ];
                let z2 = -x[0] + 2.0 * x[1];
                if z2 > 0.0 {
                    continue; // outside the premise region
                }
                let v = relu_sum_at(&rows, &cc, &x);
                assert!(
                    v <= f64::from(b_in1) + 1e-9,
                    "restricted cut violated at {x:?}: {v} > {b_in1}"
                );
            }
        }

        // Fully pinned: every neuron split ⇒ exactly one admissible pattern.
        let b_pinned = derive_cut_bound(
            &rows,
            &cc,
            &[
                SplitState::Active,
                SplitState::Inactive,
                SplitState::Inactive,
            ],
            &xl,
            &xu,
        );
        assert!(b_pinned <= b_in1);
        // Pattern {0} alone: box-max of x1 = 1.
        assert!(
            (f64::from(b_pinned) - 1.0).abs() < 1e-6,
            "b_pinned = {b_pinned}"
        );
    }

    /// Outward rounding: derived bound is never below the exact pattern max
    /// on dyadic inputs (f64 arithmetic exact here).
    #[test]
    fn bound_rounds_outward() {
        let rows = vec![AffineRow {
            w: vec![(0, 0.25), (1, -0.5)],
            r: 0.125,
        }];
        let cc = [1.0];
        let (xl, xu) = ([-0.5f32, -1.0], [1.5f32, 2.0]);
        let b = derive_cut_bound_root(&rows, &cc, &xl, &xu);
        // Exact: max(0.25·1.5, ...) = 0.375, max(-0.5·-1, -0.5·2) = 0.5,
        // + 0.125 = 1.0.  Empty pattern gives 0.  B = 1.0, next_up'd.
        assert!(f64::from(b) >= 1.0, "outward: {b} must be ≥ exact 1.0");
        assert!(f64::from(b) < 1.0 + 1e-5);
    }
}
