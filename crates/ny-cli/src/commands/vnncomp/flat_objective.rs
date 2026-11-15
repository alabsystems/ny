// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Structural sign-corner search for a FLAT falsification objective
//! (#flat-objective-surrogate) — the search half; the model surgery lives in
//! [`ny_onnx::tail_surrogate`].
//!
//! When the scored objective is bit-identical at every probed point (a
//! saturated terminal activation pins it — see the module doc over there), the
//! graph-free lane switches to scoring a pre-saturation surrogate tensor and
//! needs a search that extracts signal from FEW, EXPENSIVE, zeroth-order
//! probes. The box here is a per-coordinate interval, so the surrogate's
//! maximizer over a (locally) monotone network is expected at a SIGN CORNER —
//! every coordinate at one of its two faces — and the question is which face,
//! per coordinate, out of 2^n.
//!
//! The answer this module implements is multi-resolution HADAMARD SIGN
//! RECOVERY over structural tensor blocks:
//!
//! * Coordinates are grouped into blocks by recursively halving the ACTUAL
//!   tensor axes (the longest axis first — on an `[80, 800]` mel×time grid
//!   that alternates time/mel tiles), never by i.i.d. coordinate hashing, so a
//!   block is a contiguous time-window / frequency-band with coherent effect.
//! * At each level the incumbent corner is probed with the rows of a Sylvester
//!   Hadamard matrix: probe `k` flips exactly the blocks `b` whose Hadamard
//!   entry `H[k][b]` is −1. Under a local-affine model the per-block
//!   directional effects are recovered EXACTLY from `P = 2^⌈log₂ B⌉` probes —
//!   every probe informs every block, so each estimate averages `P`
//!   measurements instead of resting on one finite difference, which is what
//!   makes it robust to the quantization staircase this search exists for
//!   (318 Q/DQ pairs make the true gradient zero almost everywhere; a probe
//!   must step ACROSS quantization levels, and a whole-block face flip does).
//! * Block 0 shares the Hadamard's constant column, so its effect is
//!   confounded with the mean; both of its orientations are tried explicitly
//!   (2 evaluations) instead of spending a doubled probe budget on `B+1`
//!   informative columns.
//!
//! Derivative-free sign search buys gain ~ √(#probes) whatever the schedule;
//! Hadamard-at-`B = #probes` attains it. Measured on
//! `smart_turn_multimodal_2026` (two independent research lanes, 2026-08-25):
//! this family of search converts 13–16 of 35 flat rows where the previous
//! random contiguous-block search converted zero, and a freely re-optimized
//! (non-structural) block assignment OVERFITS the recovered model and does
//! worse than the incumbent.

/// Flat-vector layout of one ACTIVE (searched) input within the concatenated
/// active vector the surrogate session consumes.
#[derive(Debug, Clone)]
pub struct ActiveInputLayout {
    /// Offset of this input's first element in the active flat vector.
    pub offset: usize,
    /// Concrete resolved shape (batch axes included).
    pub shape: Vec<usize>,
    /// Row-major strides matching `shape`.
    pub strides: Vec<usize>,
}

impl ActiveInputLayout {
    pub fn new(offset: usize, shape: Vec<usize>) -> Self {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        Self {
            offset,
            shape,
            strides,
        }
    }
}

/// A contiguous axis-aligned tile of one active input's index space.
#[derive(Debug, Clone)]
pub struct TensorBlock {
    /// Index into the active-input layout table.
    pub input: usize,
    /// Half-open `[start, end)` range per axis.
    pub ranges: Vec<(usize, usize)>,
}

impl TensorBlock {
    /// One block covering an input's whole index space.
    pub fn covering(input: usize, shape: &[usize]) -> Self {
        Self {
            input,
            ranges: shape.iter().map(|&d| (0, d)).collect(),
        }
    }

    /// Consumed only by this module's test suite.
    #[cfg(test)]
    pub fn element_count(&self) -> usize {
        self.ranges.iter().map(|&(s, e)| e - s).product()
    }

    /// Split along the longest axis, or `None` for a single-element block.
    pub fn split(&self) -> Option<(TensorBlock, TensorBlock)> {
        let (axis, &(start, end)) = self
            .ranges
            .iter()
            .enumerate()
            .max_by_key(|(_, &(s, e))| e - s)?;
        if end - start < 2 {
            return None;
        }
        let mid = start + (end - start) / 2;
        let mut left = self.clone();
        let mut right = self.clone();
        left.ranges[axis] = (start, mid);
        right.ranges[axis] = (mid, end);
        Some((left, right))
    }

    /// Visit every element's offset in the ACTIVE flat vector.
    pub fn for_each_offset(&self, layout: &ActiveInputLayout, f: &mut impl FnMut(usize)) {
        // Odometer over the block's index ranges.
        let dims = self.ranges.len();
        if dims == 0 {
            f(layout.offset);
            return;
        }
        let mut idx: Vec<usize> = self.ranges.iter().map(|&(s, _)| s).collect();
        loop {
            let offset: usize = layout.offset
                + idx
                    .iter()
                    .zip(&layout.strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            f(offset);
            let mut axis = dims;
            loop {
                if axis == 0 {
                    return;
                }
                axis -= 1;
                idx[axis] += 1;
                if idx[axis] < self.ranges[axis].1 {
                    break;
                }
                idx[axis] = self.ranges[axis].0;
            }
        }
    }
}

/// `H[k][j]` of the Sylvester Hadamard matrix: `(-1)^popcount(k & j)`.
/// `true` = +1 (keep the incumbent orientation), `false` = −1 (flip).
#[inline]
pub fn hadamard_keeps(k: usize, j: usize) -> bool {
    (k & j).count_ones().is_multiple_of(2)
}

/// Write the sign corner `(blocks, signs)` into `out` (an active-flat vector):
/// every element of block `b` goes to its HI face when `signs[b] > 0`, LO face
/// otherwise. `blocks` partition the active index space, so `out` is fully
/// determined given it was seeded from any in-box template.
pub fn materialize_corner(
    blocks: &[TensorBlock],
    signs: &[i8],
    layouts: &[ActiveInputLayout],
    lo: &[f32],
    hi: &[f32],
    out: &mut [f32],
) {
    for (block, &sign) in blocks.iter().zip(signs) {
        let layout = &layouts[block.input];
        let face = if sign > 0 { hi } else { lo };
        block.for_each_offset(layout, &mut |offset| {
            out[offset] = face[offset];
        });
    }
}

/// One completed Hadamard level's outcome.
#[derive(Debug, Clone)]
pub struct LevelProposal {
    /// Proposed per-block signs (block 0 keeps the incumbent; the caller tries
    /// both of its orientations explicitly).
    pub signs: Vec<i8>,
    /// Recovered per-block directional effects. On a locally affine objective
    /// `effects[b] = P * w_b` for `b > 0` (Hadamard orthogonality over the
    /// `P` probes). `effects[0]` rides the constant column and is confounded
    /// with the probe mean, so it says nothing about block 0's orientation and
    /// MUST NOT be ranked against the others.
    pub effects: Vec<f64>,
}

/// One Hadamard level: estimate every block's orientation effect around the
/// incumbent corner and propose the per-block sign flips.
///
/// `probe(signs)` evaluates the surrogate objective at the corner given by the
/// signs (bigger = better); `None` aborts (deadline / evaluation failure) and
/// surfaces as `None` here. Returns the proposed signs plus the recovered
/// per-block effects; the CALLER evaluates the proposal and decides acceptance
/// — only measured values are ever kept.
pub fn hadamard_level_proposal(
    block_count: usize,
    incumbent_signs: &[i8],
    probe: &mut dyn FnMut(&[i8]) -> Option<f64>,
) -> Option<LevelProposal> {
    debug_assert_eq!(block_count, incumbent_signs.len());
    let probes = block_count.next_power_of_two().max(2);
    let mut effect = vec![0.0f64; block_count];
    let mut pattern = vec![0i8; block_count];
    for k in 0..probes {
        for (b, sign) in pattern.iter_mut().enumerate() {
            *sign = if hadamard_keeps(k, b) {
                incumbent_signs[b]
            } else {
                -incumbent_signs[b]
            };
        }
        let value = probe(&pattern)?;
        for (b, eff) in effect.iter_mut().enumerate() {
            if hadamard_keeps(k, b) {
                *eff += value;
            } else {
                *eff -= value;
            }
        }
    }
    // Block 0 rides the constant column: its "effect" is the probe mean, which
    // says nothing about its orientation. Keep the incumbent sign; the caller
    // tries both of its orientations explicitly.
    let signs = incumbent_signs
        .iter()
        .enumerate()
        .map(|(b, &s)| if b == 0 || effect[b] >= 0.0 { s } else { -s })
        .collect();
    Some(LevelProposal {
        signs,
        effects: effect,
    })
}

/// Keep the `keep` highest-|effect| blocks ACTIVE and move every other block,
/// with its incumbent sign, onto the frozen set (#flat-ladder-depth).
///
/// Block 0 ALWAYS stays active: its effect entry rides the Hadamard's constant
/// column and is confounded with the probe mean, so it cannot be ranked.
/// Retained blocks keep their relative order (so the ladder's positional
/// "block 0" identity is stable). Returns the retained count.
///
/// This is the mechanism behind the typed `attack.flat_surrogate_max_blocks`
/// cap: unarmed, the ladder never calls it and `split_all` doubles the block
/// count each level exactly as before.
pub fn retain_top_effect_blocks(
    blocks: &mut Vec<TensorBlock>,
    signs: &mut Vec<i8>,
    effects: &[f64],
    keep: usize,
    frozen_blocks: &mut Vec<TensorBlock>,
    frozen_signs: &mut Vec<i8>,
) -> usize {
    debug_assert_eq!(blocks.len(), signs.len());
    debug_assert_eq!(blocks.len(), effects.len());
    if blocks.len() <= keep {
        return blocks.len();
    }
    // Rank blocks 1.. by |effect| descending; block 0 is force-retained.
    let mut order: Vec<usize> = (1..blocks.len()).collect();
    order.sort_by(|&a, &b| {
        effects[b]
            .abs()
            .partial_cmp(&effects[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep_mask = vec![false; blocks.len()];
    keep_mask[0] = true;
    for &i in order.iter().take(keep.saturating_sub(1)) {
        keep_mask[i] = true;
    }
    let mut kept_blocks = Vec::with_capacity(keep);
    let mut kept_signs = Vec::with_capacity(keep);
    for (i, (block, sign)) in blocks.drain(..).zip(signs.drain(..)).enumerate() {
        if keep_mask[i] {
            kept_blocks.push(block);
            kept_signs.push(sign);
        } else {
            frozen_blocks.push(block);
            frozen_signs.push(sign);
        }
    }
    *blocks = kept_blocks;
    *signs = kept_signs;
    blocks.len()
}

/// Split every splittable block, children inheriting the parent's sign.
/// Returns `false` when nothing split (the partition is fully refined).
pub fn split_all(blocks: &mut Vec<TensorBlock>, signs: &mut Vec<i8>) -> bool {
    let mut next_blocks = Vec::with_capacity(blocks.len() * 2);
    let mut next_signs = Vec::with_capacity(blocks.len() * 2);
    let mut any = false;
    for (block, &sign) in blocks.iter().zip(signs.iter()) {
        match block.split() {
            Some((left, right)) => {
                any = true;
                next_blocks.push(left);
                next_signs.push(sign);
                next_blocks.push(right);
                next_signs.push(sign);
            }
            None => {
                next_blocks.push(block.clone());
                next_signs.push(sign);
            }
        }
    }
    *blocks = next_blocks;
    *signs = next_signs;
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covering_block_partitions_and_split_preserves_coverage() {
        let layout = ActiveInputLayout::new(0, vec![1, 6, 10]);
        let mut blocks = vec![TensorBlock::covering(0, &layout.shape)];
        let mut signs = vec![1i8];
        for _ in 0..4 {
            assert!(split_all(&mut blocks, &mut signs));
            let total: usize = blocks.iter().map(TensorBlock::element_count).sum();
            assert_eq!(total, 60, "splits must preserve total coverage");
            let mut seen = [0u8; 60];
            for block in &blocks {
                block.for_each_offset(&layout, &mut |offset| seen[offset] += 1);
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "every element covered exactly once"
            );
        }
    }

    #[test]
    fn longest_axis_splits_first() {
        let block = TensorBlock::covering(0, &[80, 800]);
        let (left, right) = block.split().expect("splittable");
        // Time (800) is longer than mel (80): the split halves time.
        assert_eq!(left.ranges, vec![(0, 80), (0, 400)]);
        assert_eq!(right.ranges, vec![(0, 80), (400, 800)]);
    }

    #[test]
    fn single_element_blocks_do_not_split() {
        let layout = ActiveInputLayout::new(0, vec![2]);
        let mut blocks = vec![TensorBlock::covering(0, &layout.shape)];
        let mut signs = vec![1i8];
        assert!(split_all(&mut blocks, &mut signs));
        assert_eq!(blocks.len(), 2);
        assert!(!split_all(&mut blocks, &mut signs), "fully refined");
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn hadamard_rows_are_orthogonal() {
        let p = 16usize;
        for a in 0..p {
            for b in 0..p {
                let dot: i64 = (0..p)
                    .map(|k| {
                        let va = if hadamard_keeps(k, a) { 1i64 } else { -1 };
                        let vb = if hadamard_keeps(k, b) { 1i64 } else { -1 };
                        va * vb
                    })
                    .sum();
                assert_eq!(dot, if a == b { p as i64 } else { 0 });
            }
        }
    }

    /// On an exactly-affine objective one Hadamard level recovers every block
    /// orientation except block 0's (the constant column), in P = 2^ceil(log2 B)
    /// probes.
    #[test]
    fn hadamard_level_recovers_affine_block_signs() {
        let weights = [3.0, -2.0, 5.0, -1.0, 0.5, -4.0, 2.5, -0.25];
        let objective = |signs: &[i8]| -> f64 {
            signs
                .iter()
                .zip(&weights)
                .map(|(&s, &w)| f64::from(s) * w)
                .sum()
        };
        let incumbent = vec![1i8; 8];
        let mut probes = 0usize;
        let proposal = hadamard_level_proposal(8, &incumbent, &mut |signs| {
            probes += 1;
            Some(objective(signs))
        })
        .expect("no aborts");
        assert_eq!(probes, 8);
        // Optimal signs are sign(w_b); block 0 keeps the incumbent.
        assert_eq!(proposal.signs, vec![1, -1, 1, -1, 1, -1, 1, -1]);
        assert!(objective(&proposal.signs) > objective(&incumbent));
        // On an exactly-affine objective the recovered effect of block b > 0
        // is P * w_b (orthogonality of the Hadamard rows over the 8 probes).
        for (b, &w) in weights.iter().enumerate().skip(1) {
            assert!(
                (proposal.effects[b] - 8.0 * w).abs() < 1e-9,
                "effects[{b}] = {}, expected {}",
                proposal.effects[b],
                8.0 * w
            );
        }
    }

    /// The multi-resolution ladder drives a nonlinear-but-monotone objective to
    /// its optimal corner from few probes.
    #[test]
    fn ladder_reaches_the_optimal_corner_on_a_monotone_objective() {
        // 64 coordinates on a [4, 16] grid, box [-1, 1] per coordinate,
        // objective = quantized (staircase) affine score, like the QDQ tail.
        let layout = ActiveInputLayout::new(0, vec![4, 16]);
        let mut weights = vec![0.0f64; 64];
        let mut state = 0x9E3779B97F4A7C15u64;
        for w in weights.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *w = ((state % 2000) as f64 - 1000.0) / 250.0;
        }
        let lo = vec![-1.0f32; 64];
        let hi = vec![1.0f32; 64];
        let staircase = |x: &[f32]| -> f64 {
            let raw: f64 = x
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| f64::from(v) * w)
                .sum();
            (raw / 0.5).floor() * 0.5 // 0.5-wide quantization steps
        };
        let optimum: f64 = weights.iter().map(|w| w.abs()).sum();

        let mut blocks = vec![TensorBlock::covering(0, &layout.shape)];
        let mut signs = vec![1i8];
        let mut x = vec![0.0f32; 64];
        let layouts = [layout];
        let mut best = f64::NEG_INFINITY;
        for _ in 0..7 {
            if !split_all(&mut blocks, &mut signs) {
                break;
            }
            let proposal = hadamard_level_proposal(blocks.len(), &signs, &mut |s| {
                materialize_corner(&blocks, s, &layouts, &lo, &hi, &mut x);
                Some(staircase(&x))
            })
            .expect("no aborts");
            // Try both orientations of block 0, keep the best measured corner.
            for flip0 in [false, true] {
                let mut candidate = proposal.signs.clone();
                if flip0 {
                    candidate[0] = -candidate[0];
                }
                materialize_corner(&blocks, &candidate, &layouts, &lo, &hi, &mut x);
                let value = staircase(&x);
                if value > best {
                    best = value;
                    signs = candidate;
                }
            }
        }
        assert!(
            best >= optimum - 0.5,
            "ladder reached {best}, optimum {optimum}"
        );
    }

    /// Retention keeps block 0 unconditionally, keeps the highest-|effect|
    /// blocks after it, preserves relative order, and moves everything else —
    /// with its sign — onto the frozen set.
    #[test]
    fn retain_top_effect_blocks_keeps_block0_and_the_largest_effects() {
        let layout = ActiveInputLayout::new(0, vec![8]);
        let mut blocks: Vec<TensorBlock> = (0..8)
            .map(|i| TensorBlock {
                input: 0,
                ranges: vec![(i, i + 1)],
            })
            .collect();
        let mut signs: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        // Block 0's effect is a HUGE decoy (the constant column would show
        // exactly this); it must be retained by position, not by rank.
        let effects = [1e9, 0.1, 7.0, 0.2, -9.0, 0.3, 2.0, -0.4];
        let mut frozen_blocks = Vec::new();
        let mut frozen_signs = Vec::new();
        let kept = retain_top_effect_blocks(
            &mut blocks,
            &mut signs,
            &effects,
            4,
            &mut frozen_blocks,
            &mut frozen_signs,
        );
        assert_eq!(kept, 4);
        // Retained: block 0 (forced) + blocks 4 (|-9|), 2 (7), 6 (2), in
        // original relative order.
        let retained: Vec<usize> = blocks.iter().map(|b| b.ranges[0].0).collect();
        assert_eq!(retained, vec![0, 2, 4, 6]);
        assert_eq!(signs, vec![1, 1, 1, 1]);
        let frozen: Vec<usize> = frozen_blocks.iter().map(|b| b.ranges[0].0).collect();
        assert_eq!(frozen, vec![1, 3, 5, 7]);
        assert_eq!(frozen_signs, vec![-1, -1, -1, -1]);
        // Active + frozen still partition the whole index space.
        let mut seen = [0u8; 8];
        for block in blocks.iter().chain(frozen_blocks.iter()) {
            block.for_each_offset(&layout, &mut |offset| seen[offset] += 1);
        }
        assert!(seen.iter().all(|&c| c == 1));
        // A set already within the cap is untouched.
        let kept = retain_top_effect_blocks(
            &mut blocks,
            &mut signs,
            &effects[..4],
            4,
            &mut frozen_blocks,
            &mut frozen_signs,
        );
        assert_eq!(kept, 4);
        assert_eq!(frozen_blocks.len(), 4);
    }

    /// The CAPPED ladder (#flat-ladder-depth): per-level probe cost stays
    /// bounded by the cap while refinement continues to single coordinates —
    /// the uncapped schedule would need 64-block levels to get there.
    #[test]
    fn capped_ladder_refines_deeper_at_bounded_per_level_cost() {
        let layout = ActiveInputLayout::new(0, vec![4, 16]);
        // Concentrate the mass on an 8-coordinate heavy run at the START of
        // the tensor (so the force-retained positional block 0 is part of it),
        // with MIXED signs but unequal magnitudes: every contiguous sub-run of
        // `5, -4, 5, -4, ...` has a nonzero sum, so a heavy-containing tile
        // outranks the 0.01 background at EVERY granularity and effect-ranked
        // retention keeps the right region. (An equal-magnitude alternation
        // would cancel over whole tiles and blind the selection — that is a
        // real property of the mechanism, not a bug in it.)
        let mut weights = vec![0.01f64; 64];
        for (i, w) in weights.iter_mut().enumerate().take(8) {
            *w = if i % 2 == 0 { 5.0 } else { -4.0 };
        }
        let lo = vec![-1.0f32; 64];
        let hi = vec![1.0f32; 64];
        let affine = |x: &[f32]| -> f64 {
            x.iter()
                .zip(&weights)
                .map(|(&v, &w)| f64::from(v) * w)
                .sum()
        };

        let cap = 16usize;
        let mut blocks = vec![TensorBlock::covering(0, &layout.shape)];
        let mut signs = vec![1i8];
        let mut frozen_blocks: Vec<TensorBlock> = Vec::new();
        let mut frozen_signs: Vec<i8> = Vec::new();
        let mut x = vec![0.0f32; 64];
        let layouts = [layout];
        let mut best = f64::NEG_INFINITY;
        let mut max_level_probes = 0usize;
        let mut last_effects: Option<Vec<f64>> = None;
        loop {
            if blocks.len().saturating_mul(2) > cap {
                if let Some(effects) = last_effects.take() {
                    retain_top_effect_blocks(
                        &mut blocks,
                        &mut signs,
                        &effects,
                        cap / 2,
                        &mut frozen_blocks,
                        &mut frozen_signs,
                    );
                    materialize_corner(&frozen_blocks, &frozen_signs, &layouts, &lo, &hi, &mut x);
                }
            }
            if !split_all(&mut blocks, &mut signs) {
                break;
            }
            let mut level_probes = 0usize;
            let proposal = hadamard_level_proposal(blocks.len(), &signs, &mut |s| {
                level_probes += 1;
                materialize_corner(&blocks, s, &layouts, &lo, &hi, &mut x);
                Some(affine(&x))
            })
            .expect("no aborts");
            max_level_probes = max_level_probes.max(level_probes);
            for flip0 in [false, true] {
                let mut candidate = proposal.signs.clone();
                if flip0 {
                    candidate[0] = -candidate[0];
                }
                materialize_corner(&blocks, &candidate, &layouts, &lo, &hi, &mut x);
                let value = affine(&x);
                if value > best {
                    best = value;
                    signs = candidate;
                }
            }
            last_effects = Some(proposal.effects);
        }
        // Per-level cost is capped: never more than `cap` probes per level.
        assert!(
            max_level_probes <= cap,
            "a level spent {max_level_probes} probes, cap {cap}"
        );
        // Refinement went ALL the way down inside the retained region: the
        // active blocks are single coordinates, which the uncapped schedule
        // only reaches with a 64-probe level.
        assert!(blocks
            .iter()
            .all(|b| b.ranges.iter().all(|&(s, e)| e - s == 1)));
        assert!(!frozen_blocks.is_empty(), "the cap must have engaged");
        // The heavy run is fully recovered: the 8 heavy coordinates fit inside
        // the cap/2 retained blocks, the final Hadamard level over single
        // coordinates recovers each sign exactly (the objective is exactly
        // affine), and the frozen positive background was already at its
        // optimal hi face when it froze.
        let heavy: f64 = weights[..8].iter().map(|w| w.abs()).sum();
        assert!(
            best >= heavy,
            "capped ladder reached {best}, heavy mass {heavy}"
        );
    }
}
