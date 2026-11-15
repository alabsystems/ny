// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #la-warm-frontier — the data half of the frontier-complete lA warm start.
//!
//! A [`WarmFrontier`] is the COMPLETE backward state a CROWN pass held at the
//! moment it popped one node `N`, minus `N`'s own block (which is the ordinary
//! [`CachedLinearBounds`] entry for `N`): every other node's block that was
//! still pending (skip-path / multi-consumer operands) and the network-input
//! channel (coefficients that already reached the input plus every bias that a
//! binary / n-ary node diverted there). A child whose last split is on `N`
//! resumes its backward pass from `N`'s block PLUS this frontier instead of
//! from the output. Why that resumed bound is a bound on the objective over the
//! child's region is proved in the engine half,
//! `beta_crown/engine/graph/propagation/warm_frontier.rs`; this file owns the
//! capture-time transformation that the proof relies on:
//!
//! * **Fold.** Every captured block has its certified coefficient error folded
//!   OUTWARD into its bias over the capturing domain's box for the block's
//!   columns ([`CachedLinearBounds::fold_capture_coeff_err`], the same primitive
//!   the single-block cache uses). The box encloses those values on the
//!   capturing region, which contains every descendant region the frontier is
//!   later read on, so each folded block is still a lower (upper) bound term
//!   there. A block that cannot be folded (no box, a box that does not match its
//!   columns, a non-finite penalty) refuses the WHOLE frontier: a frontier is
//!   usable only complete.
//! * **Finiteness.** Every stored coefficient and bias is finite. A capture that
//!   would store a non-finite value is refused (the child then takes the cold
//!   pass) rather than stored as a conservative ±inf, which would only ever
//!   make the resumed bound vacuous.
//! * **Shape.** Every member has the row count of `N`'s block, so the row
//!   slicing / stacking of multi-objective caches keeps row `i` of every member
//!   paired with row `i` of `N`'s block.
//! * **Compaction.** A member whose coefficients are identically zero (the
//!   input channel of a pass that has only diverted bias so far) stores its
//!   bias only; its `rows x input_dim` zeros are rebuilt at seed time.
//! * **Pairing.** A frontier is an identity only TOGETHER with the exact block
//!   `N` held at the same instant of the same pass. Caches are merged per node
//!   across passes (`multi_objective/shared.rs` #la-union-merge keeps an
//!   ancestor's entries and lets a fresh pass overwrite others), sliced per row
//!   and stacked, so the frontier records a 128-bit [`BlockDigest`] of its
//!   partner block's exact bits; attach, split, stack and every seed site
//!   recompute the digest of the block actually present and drop / refuse the
//!   frontier on any mismatch. A fresh block paired with an ancestor's frontier
//!   (a sum of terms from two different passes, NOT a bound) therefore can
//!   never be seeded.
//!
//! Nothing here decides whether a frontier may be USED: the seed sites check
//! the frontier against the child's graph, row count and node widths first.

use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

use ndarray::{Array1, Array2, Axis};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::types::CachedLinearBounds;
use crate::LinearBounds;

/// A 128-bit identity of one cache block's exact contents (shapes and every
/// bit of `lower_a`, `lower_b`, `upper_a`, `upper_b`, in logical order): two
/// independent 64-bit hashes (FNV-1a and a multiply-rotate mix). Used only to
/// prove that a frontier is seeded next to the very block it was captured
/// with; an accidental collision needs both hashes to agree (~2^-128).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockDigest(u64, u64);

impl BlockDigest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325, 0x9e37_79b9_7f4a_7c15)
    }

    fn word(&mut self, word: u64) {
        for byte in word.to_le_bytes() {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.1 = (self.1 ^ word)
            .wrapping_mul(0xff51_afd7_ed55_8ccd)
            .rotate_left(29);
    }

    fn values<'a>(&mut self, values: impl Iterator<Item = &'a f32>) {
        for value in values {
            self.word(u64::from(value.to_bits()));
        }
    }

    /// Digest of a block given by its four parts.
    pub(crate) fn of_parts(
        lower_a: &Array2<f32>,
        lower_b: &Array1<f32>,
        upper_a: &Array2<f32>,
        upper_b: &Array1<f32>,
    ) -> Self {
        let mut digest = Self::new();
        for dim in [
            lower_a.nrows(),
            lower_a.ncols(),
            lower_b.len(),
            upper_a.nrows(),
            upper_a.ncols(),
            upper_b.len(),
        ] {
            digest.word(dim as u64);
        }
        digest.values(lower_a.iter());
        digest.values(lower_b.iter());
        digest.values(upper_a.iter());
        digest.values(upper_b.iter());
        digest
    }

    /// Digest of a block held as [`LinearBounds`] (a cache block never
    /// carries coefficient error: it is folded before publication).
    pub(crate) fn of_block(block: &LinearBounds) -> Self {
        Self::of_parts(
            block.lower_a(),
            block.lower_b(),
            block.upper_a(),
            block.upper_b(),
        )
    }
}

/// One pending block of a captured frontier: error-free, finite, `rows x cols`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FrontierBlock {
    /// `(lower_a, upper_a)`, or `None` when both are identically zero.
    coefficients: Option<(Array2<f32>, Array2<f32>)>,
    lower_b: Array1<f32>,
    upper_b: Array1<f32>,
    /// Width of the values the coefficients multiply (the node's flattened
    /// output, or the network input for the input channel).
    cols: usize,
}

impl FrontierBlock {
    /// Adopt an already-folded block. `None` when it still carries coefficient
    /// error, is malformed, or holds any non-finite value.
    fn from_folded(block: LinearBounds) -> Option<Self> {
        if block.has_coeff_err() || block.validate_internal_shapes().is_err() {
            return None;
        }
        let cols = block.num_inputs();
        let (lower_a, lower_b, upper_a, upper_b) = block.into_parts();
        if !lower_a.iter().all(|v| v.is_finite())
            || !upper_a.iter().all(|v| v.is_finite())
            || !lower_b.iter().all(|v| v.is_finite())
            || !upper_b.iter().all(|v| v.is_finite())
        {
            return None;
        }
        let zero = lower_a.iter().all(|&v| v == 0.0) && upper_a.iter().all(|&v| v == 0.0);
        Some(Self {
            coefficients: (!zero).then_some((lower_a, upper_a)),
            lower_b,
            upper_b,
            cols,
        })
    }

    /// Row count (the objective rows the block bounds).
    pub(crate) fn rows(&self) -> usize {
        self.lower_b.len()
    }

    /// Column count (the width of the values the block multiplies).
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    /// Whether the block stores no coefficients (bias only).
    #[cfg(test)]
    pub(crate) fn is_bias_only(&self) -> bool {
        self.coefficients.is_none()
    }

    /// Rebuild the block as ordinary [`LinearBounds`] for a seed site. The
    /// constructor re-validates shapes and finiteness, so a corrupted entry is
    /// an `Err` (the caller refuses the frontier), never a seed.
    pub(crate) fn to_linear_bounds(&self) -> Result<LinearBounds> {
        let rows = self.rows();
        let (lower_a, upper_a) = match &self.coefficients {
            Some((lower_a, upper_a)) => (lower_a.clone(), upper_a.clone()),
            None => (
                Array2::zeros((rows, self.cols)),
                Array2::zeros((rows, self.cols)),
            ),
        };
        LinearBounds::new(lower_a, self.lower_b.clone(), upper_a, self.upper_b.clone())
    }

    fn owned_bytes(&self) -> usize {
        let coefficients = self.coefficients.as_ref().map_or(0, |(lower, upper)| {
            lower
                .len()
                .saturating_add(upper.len())
                .saturating_mul(size_of::<f32>())
        });
        coefficients
            .saturating_add(
                self.lower_b
                    .len()
                    .saturating_add(self.upper_b.len())
                    .saturating_mul(size_of::<f32>()),
            )
            .saturating_add(size_of::<Self>())
    }

    fn slice_row(&self, row: usize) -> Option<Self> {
        if row >= self.rows() {
            return None;
        }
        let pick = |a: &Array2<f32>| a.row(row).to_owned().insert_axis(Axis(0));
        Some(Self {
            coefficients: self
                .coefficients
                .as_ref()
                .map(|(lower, upper)| (pick(lower), pick(upper))),
            lower_b: ndarray::arr1(&[self.lower_b[row]]),
            upper_b: ndarray::arr1(&[self.upper_b[row]]),
            cols: self.cols,
        })
    }

    /// Stack single-row blocks of the same width, in order. A mix of
    /// bias-only and full blocks stacks as a full block (zeros filled in).
    fn stack(blocks: &[&Self]) -> Option<Self> {
        let first = blocks.first()?;
        let cols = first.cols;
        if blocks.iter().any(|b| b.rows() != 1 || b.cols != cols) {
            return None;
        }
        let lower_b = Array1::from_iter(blocks.iter().map(|b| b.lower_b[0]));
        let upper_b = Array1::from_iter(blocks.iter().map(|b| b.upper_b[0]));
        let coefficients = if blocks.iter().all(|b| b.coefficients.is_none()) {
            None
        } else {
            let mut lower = Array2::<f32>::zeros((blocks.len(), cols));
            let mut upper = Array2::<f32>::zeros((blocks.len(), cols));
            for (row, block) in blocks.iter().enumerate() {
                if let Some((l, u)) = &block.coefficients {
                    if l.shape() != [1, cols] || u.shape() != [1, cols] {
                        return None;
                    }
                    lower.row_mut(row).assign(&l.row(0));
                    upper.row_mut(row).assign(&u.row(0));
                }
            }
            Some((lower, upper))
        };
        Some(Self {
            coefficients,
            lower_b,
            upper_b,
            cols,
        })
    }
}

/// The complete pending backward state at one popped node, minus that node's
/// own block. Members are sorted by key and unique; the key is the node whose
/// OUTPUT the block's columns multiply, or `NETWORK_INPUT` for the input
/// channel. Never empty. Paired with its node's block by [`BlockDigest`].
#[derive(Debug, Clone, PartialEq)]
pub struct WarmFrontier {
    rows: usize,
    /// Digest of the popped node's own block as stored in the cache.
    branch: BlockDigest,
    members: Vec<(String, FrontierBlock)>,
}

/// Outcome of one frontier capture.
#[derive(Debug)]
pub(crate) enum FrontierCapture {
    /// A complete, folded, finite frontier.
    Recorded(WarmFrontier),
    /// Nothing else was pending: the popped node's block alone is the whole
    /// state (the unary-chain case the single-block seed already covers).
    Empty,
    /// The frontier could not be stored completely; the reason is a stable
    /// token for the observable.
    Refused(&'static str),
}

impl WarmFrontier {
    /// Capture the pending state `pending` (every entry except the popped
    /// node's own) of a pass whose popped block, AS STORED IN THE CACHE (after
    /// its own fold), is `branch_block`.
    ///
    /// `box_of(key)` must return the capturing domain's certified box for the
    /// values the block under `key` multiplies: the node's output box, or the
    /// domain's input box for `NETWORK_INPUT`. Deadline refusals of the fold
    /// are returned as `Err`; every other failure is `Refused`.
    pub(crate) fn capture<'b, F>(
        branch_block: &LinearBounds,
        pending: Vec<(String, LinearBounds)>,
        mut box_of: F,
        deadline: Option<Instant>,
    ) -> Result<FrontierCapture>
    where
        F: FnMut(&str) -> Option<&'b BoundedTensor>,
    {
        if pending.is_empty() {
            return Ok(FrontierCapture::Empty);
        }
        if branch_block.has_coeff_err() {
            return Ok(FrontierCapture::Refused("branch_unfolded"));
        }
        let rows = branch_block.num_outputs();
        let branch = BlockDigest::of_block(branch_block);
        let mut members = Vec::with_capacity(pending.len());
        for (key, mut block) in pending {
            if block.num_outputs() != rows {
                return Ok(FrontierCapture::Refused("rows"));
            }
            if !CachedLinearBounds::fold_capture_coeff_err(&mut block, box_of(&key), deadline)? {
                return Ok(FrontierCapture::Refused("unfoldable"));
            }
            let Some(stored) = FrontierBlock::from_folded(block) else {
                return Ok(FrontierCapture::Refused("non_finite"));
            };
            members.push((key, stored));
        }
        members.sort_by(|a, b| a.0.cmp(&b.0));
        if members.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Ok(FrontierCapture::Refused("duplicate"));
        }
        Ok(FrontierCapture::Recorded(Self {
            rows,
            branch,
            members,
        }))
    }

    /// Row count shared by every member (and by the popped node's block).
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Digest of the block this frontier must be seeded next to.
    pub(crate) fn branch_digest(&self) -> BlockDigest {
        self.branch
    }

    /// Number of pending members.
    pub(crate) fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether no member is stored (never true for a recorded frontier).
    pub(crate) fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The members, sorted by key.
    pub(crate) fn members(&self) -> impl Iterator<Item = (&str, &FrontierBlock)> {
        self.members
            .iter()
            .map(|(key, block)| (key.as_str(), block))
    }

    /// Conservative census of the bytes this frontier owns (queue budgets).
    pub(crate) fn estimated_owned_bytes(&self) -> usize {
        self.members
            .iter()
            .fold(size_of::<Self>(), |sum, (key, block)| {
                sum.saturating_add(key.len())
                    .saturating_add(block.owned_bytes())
            })
    }

    /// Row `row` of every member, as a single-row frontier paired with the
    /// block whose digest is `branch` (row `row` of the partner block).
    pub(crate) fn slice_row(&self, row: usize, branch: BlockDigest) -> Option<Self> {
        let members = self
            .members
            .iter()
            .map(|(key, block)| Some((key.clone(), block.slice_row(row)?)))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            rows: 1,
            branch,
            members,
        })
    }

    /// Stack single-row frontiers of the SAME structure (member keys and
    /// widths) into one multi-row frontier, rows in slice order, paired with
    /// the stacked partner block whose digest is `branch`. `None` on any
    /// structural difference: the caller then keeps no frontier for that node.
    pub(crate) fn stack_single_row(frontiers: &[&Self], branch: BlockDigest) -> Option<Self> {
        let first = frontiers.first()?;
        if frontiers.iter().any(|f| {
            f.rows != 1
                || f.members.len() != first.members.len()
                || f.members
                    .iter()
                    .zip(&first.members)
                    .any(|(a, b)| a.0 != b.0 || a.1.cols != b.1.cols)
        }) {
            return None;
        }
        let mut members = Vec::with_capacity(first.members.len());
        for (index, (key, _)) in first.members.iter().enumerate() {
            let column: Vec<&FrontierBlock> =
                frontiers.iter().map(|f| &f.members[index].1).collect();
            members.push((key.clone(), FrontierBlock::stack(&column)?));
        }
        Some(Self {
            rows: frontiers.len(),
            branch,
            members,
        })
    }
}

impl CachedLinearBounds {
    /// #la-warm-frontier: the frontier captured when the pass popped `node`.
    pub(crate) fn frontier(&self, node: &str) -> Option<&WarmFrontier> {
        self.frontiers.get(node)
    }

    /// Number of frontiers this cache carries.
    pub(crate) fn frontier_count(&self) -> usize {
        self.frontiers.len()
    }

    /// Digest of the block this cache holds for `node` (see [`BlockDigest`]).
    pub(crate) fn block_digest(&self, node: &str) -> Option<BlockDigest> {
        Some(BlockDigest::of_parts(
            self.lower_a.get(node)?,
            self.lower_b.get(node)?,
            self.upper_a.get(node)?,
            self.upper_b.get(node)?,
        ))
    }

    /// Whether `frontier` is paired with the block this cache holds for `node`.
    fn frontier_is_paired(&self, node: &str, frontier: &WarmFrontier) -> bool {
        !frontier.is_empty()
            && self
                .lower_b
                .get(node)
                .is_some_and(|b| b.len() == frontier.rows())
            && self.block_digest(node) == Some(frontier.branch_digest())
    }

    /// #la-warm-frontier: attach frontiers captured by the SAME pass that
    /// produced this cache's blocks. A frontier is kept only when the cache
    /// holds the exact block it was captured with (digest-checked). Returns
    /// how many were attached.
    pub(crate) fn attach_frontiers(&mut self, frontiers: HashMap<String, WarmFrontier>) -> usize {
        let mut attached = 0usize;
        for (node, frontier) in frontiers {
            if self.frontier_is_paired(&node, &frontier) {
                self.frontiers.insert(node, frontier);
                attached += 1;
            }
        }
        attached
    }

    /// Builder form of [`Self::attach_frontiers`].
    pub(crate) fn with_frontiers(mut self, frontiers: HashMap<String, WarmFrontier>) -> Self {
        self.attach_frontiers(frontiers);
        self
    }

    /// #la-union-merge with frontiers: overwrite this cache's blocks with
    /// `fresh`'s (fresh wins on collision; inherited entries at nodes `fresh`
    /// never visited stay), drop every inherited frontier whose partner block
    /// was overwritten, and adopt `fresh`'s frontiers. With no frontier on
    /// either side this is exactly the historical four-map `extend`.
    pub(crate) fn union_with_fresh(&mut self, fresh: CachedLinearBounds) {
        let CachedLinearBounds {
            lower_a,
            upper_a,
            lower_b,
            upper_b,
            frontiers,
        } = fresh;
        if !self.frontiers.is_empty() {
            self.frontiers.retain(|node, _| {
                !(lower_a.contains_key(node)
                    || upper_a.contains_key(node)
                    || lower_b.contains_key(node)
                    || upper_b.contains_key(node))
            });
        }
        self.lower_a.extend(lower_a);
        self.upper_a.extend(upper_a);
        self.lower_b.extend(lower_b);
        self.upper_b.extend(upper_b);
        if !frontiers.is_empty() {
            self.attach_frontiers(frontiers);
        }
    }

    /// Bytes owned by the attached frontiers (part of the queue census).
    pub(crate) fn frontier_owned_bytes(&self) -> usize {
        self.frontiers.iter().fold(0usize, |sum, (node, frontier)| {
            sum.saturating_add(node.len())
                .saturating_add(frontier.estimated_owned_bytes())
        })
    }

    /// Row `row` of every frontier paired in `self`, re-paired with the
    /// corresponding block of `row_cache` (row `row` of this cache, as built
    /// by `split_multi_row`).
    pub(crate) fn frontiers_for_row(
        &self,
        row: usize,
        row_cache: &CachedLinearBounds,
    ) -> HashMap<String, WarmFrontier> {
        self.frontiers
            .iter()
            .filter(|(node, frontier)| self.frontier_is_paired(node, frontier))
            .filter_map(|(node, frontier)| {
                let sliced = frontier.slice_row(row, row_cache.block_digest(node)?)?;
                Some((node.clone(), sliced))
            })
            .collect()
    }

    /// Frontiers present, paired and of one structure in EVERY single-row
    /// cache, stacked in cache order and re-paired with the stacked block of
    /// `stacked` (for `stack_single_row`).
    pub(crate) fn stacked_frontiers(
        caches: &[&CachedLinearBounds],
        stacked: &CachedLinearBounds,
    ) -> HashMap<String, WarmFrontier> {
        let Some(first) = caches.first() else {
            return HashMap::new();
        };
        first
            .frontiers
            .keys()
            .filter_map(|node| {
                let column = caches
                    .iter()
                    .map(|cache| {
                        cache
                            .frontiers
                            .get(node)
                            .filter(|frontier| cache.frontier_is_paired(node, frontier))
                    })
                    .collect::<Option<Vec<_>>>()?;
                let branch = stacked.block_digest(node)?;
                Some((
                    node.clone(),
                    WarmFrontier::stack_single_row(&column, branch)?,
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;

    fn block(a: [[f32; 2]; 1], b: f32) -> LinearBounds {
        LinearBounds::new(arr2(&a), arr1(&[b]), arr2(&a), arr1(&[b])).expect("finite block")
    }

    fn unit_box(width: usize) -> BoundedTensor {
        BoundedTensor::new(
            Array1::from_elem(width, -1.0f32).into_dyn(),
            Array1::from_elem(width, 1.0f32).into_dyn(),
        )
        .expect("box")
    }

    /// A one-row partner block for captures that do not care about pairing.
    fn partner() -> LinearBounds {
        block([[0.5, -0.5]], 0.25)
    }

    fn two_row_partner() -> LinearBounds {
        LinearBounds::new(
            arr2(&[[1.0, 0.0], [0.0, 1.0]]),
            arr1(&[0.0, 0.0]),
            arr2(&[[1.0, 0.0], [0.0, 1.0]]),
            arr1(&[0.0, 0.0]),
        )
        .expect("main")
    }

    fn recorded(capture: FrontierCapture) -> WarmFrontier {
        match capture {
            FrontierCapture::Recorded(frontier) => frontier,
            other => panic!("expected a recorded frontier, got {other:?}"),
        }
    }

    #[test]
    fn warm_frontier_capture_is_empty_without_pending_blocks() {
        let capture =
            WarmFrontier::capture(&partner(), Vec::new(), |_| None, None).expect("capture");
        assert!(matches!(capture, FrontierCapture::Empty));
    }

    /// The fold is outward: a block carrying certified coefficient error `E`
    /// over `y in [-1, 1]^2` loses `Σ E` on its lower bias and gains it on its
    /// upper bias, and is stored error-free.
    #[test]
    fn warm_frontier_capture_folds_coefficient_error_outward() {
        let mut carrying = block([[1.0, -1.0]], 0.5);
        carrying.set_coeff_err(arr2(&[[1e-3, 2e-3]]), arr2(&[[1e-3, 2e-3]]));
        let boxed = unit_box(2);
        let frontier = recorded(
            WarmFrontier::capture(
                &partner(),
                vec![("skip".to_string(), carrying)],
                |_| Some(&boxed),
                None,
            )
            .expect("capture"),
        );
        let (key, stored) = frontier.members().next().expect("one member");
        assert_eq!(key, "skip");
        let rebuilt = stored.to_linear_bounds().expect("rebuild");
        assert!(!rebuilt.has_coeff_err());
        assert!(rebuilt.lower_b()[0] <= 0.5 - 3e-3);
        assert!(rebuilt.upper_b()[0] >= 0.5 + 3e-3);
    }

    /// Completeness is all-or-nothing: one member that cannot be folded (no
    /// box) refuses the whole frontier instead of storing the others.
    #[test]
    fn warm_frontier_capture_refuses_the_whole_frontier_on_one_unfoldable_member() {
        let mut carrying = block([[1.0, -1.0]], 0.5);
        carrying.set_coeff_err(arr2(&[[1e-3, 2e-3]]), arr2(&[[1e-3, 2e-3]]));
        let exact = block([[0.25, 0.5]], 1.0);
        let capture = WarmFrontier::capture(
            &partner(),
            vec![
                ("exact".to_string(), exact),
                ("carrying".to_string(), carrying),
            ],
            |_| None,
            None,
        )
        .expect("capture");
        assert!(matches!(capture, FrontierCapture::Refused("unfoldable")));
    }

    #[test]
    fn warm_frontier_capture_refuses_row_mismatch_and_non_finite_values() {
        let capture = WarmFrontier::capture(
            &two_row_partner(),
            vec![("skip".to_string(), block([[1.0, 0.0]], 0.0))],
            |_| None,
            None,
        )
        .expect("capture");
        assert!(matches!(capture, FrontierCapture::Refused("rows")));
        let infinite = LinearBounds::new(
            arr2(&[[1.0, 0.0]]),
            arr1(&[f32::NEG_INFINITY]),
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
        )
        .expect("an infinite bias is a legal LinearBounds");
        let capture = WarmFrontier::capture(
            &partner(),
            vec![("skip".to_string(), infinite)],
            |_| None,
            None,
        )
        .expect("capture");
        assert!(matches!(capture, FrontierCapture::Refused("non_finite")));
    }

    /// A zero-coefficient carrier (the input channel holding only diverted
    /// bias) is stored bias-only and rebuilt with its zeros at full width.
    #[test]
    fn warm_frontier_bias_only_member_round_trips_at_full_width() {
        let bias_only = LinearBounds::new(
            Array2::zeros((1, 5)),
            arr1(&[-0.75]),
            Array2::zeros((1, 5)),
            arr1(&[1.25]),
        )
        .expect("bias carrier");
        let frontier = recorded(
            WarmFrontier::capture(
                &partner(),
                vec![(crate::NETWORK_INPUT.to_string(), bias_only)],
                |_| None,
                None,
            )
            .expect("capture"),
        );
        let (_, stored) = frontier.members().next().expect("member");
        assert!(stored.is_bias_only());
        assert_eq!(stored.cols(), 5);
        let rebuilt = stored.to_linear_bounds().expect("rebuild");
        assert_eq!(rebuilt.lower_a().shape(), &[1, 5]);
        assert!(rebuilt.lower_a().iter().all(|&v| v == 0.0));
        assert_eq!(rebuilt.lower_b()[0], -0.75);
        assert_eq!(rebuilt.upper_b()[0], 1.25);
    }

    /// Slicing rows and stacking them back is the identity, so a multi-row
    /// cache split per objective and re-stacked keeps every member row-paired.
    #[test]
    fn warm_frontier_rows_round_trip_through_slice_and_stack() {
        let two_rows = LinearBounds::new(
            arr2(&[[1.0, 2.0], [3.0, 4.0]]),
            arr1(&[0.5, -0.5]),
            arr2(&[[1.5, 2.5], [3.5, 4.5]]),
            arr1(&[1.0, 2.0]),
        )
        .expect("two-row block");
        let bias = LinearBounds::new(
            Array2::zeros((2, 3)),
            arr1(&[-1.0, -2.0]),
            Array2::zeros((2, 3)),
            arr1(&[1.0, 2.0]),
        )
        .expect("bias");
        let partner_block = two_row_partner();
        let frontier = recorded(
            WarmFrontier::capture(
                &partner_block,
                vec![
                    ("skip".to_string(), two_rows),
                    (crate::NETWORK_INPUT.to_string(), bias),
                ],
                |_| None,
                None,
            )
            .expect("capture"),
        );
        let digest = frontier.branch_digest();
        let row0 = frontier.slice_row(0, digest).expect("row 0");
        let row1 = frontier.slice_row(1, digest).expect("row 1");
        assert!(frontier.slice_row(2, digest).is_none());
        let stacked = WarmFrontier::stack_single_row(&[&row0, &row1], digest).expect("stack");
        assert_eq!(stacked, frontier);
        // A structural difference refuses the stack.
        let other = recorded(
            WarmFrontier::capture(
                &partner(),
                vec![("other".to_string(), block([[1.0, 0.0]], 0.0))],
                |_| None,
                None,
            )
            .expect("capture"),
        );
        assert!(WarmFrontier::stack_single_row(&[&row0, &other], digest).is_none());
    }

    /// `attach_frontiers` keeps a frontier only next to the EXACT block it was
    /// captured with (digest), and the cache's split / stack keep frontiers
    /// row-paired and re-paired with the sliced / stacked blocks.
    #[test]
    fn warm_frontier_cache_attach_split_and_stack_keep_frontiers_row_paired() {
        let main = two_row_partner();
        let mut cache = CachedLinearBounds::from_linear_bounds_map(HashMap::from([
            ("relu".to_string(), main.clone()),
            ("other".to_string(), main.clone()),
        ]));
        let skip = LinearBounds::new(
            arr2(&[[0.5, 0.5], [0.25, -0.25]]),
            arr1(&[0.1, 0.2]),
            arr2(&[[0.5, 0.5], [0.25, -0.25]]),
            arr1(&[0.3, 0.4]),
        )
        .expect("skip");
        let frontier = recorded(
            WarmFrontier::capture(&main, vec![("skip".to_string(), skip)], |_| None, None)
                .expect("capture"),
        );
        // `other` holds a DIFFERENT block than the one the frontier paired with.
        cache
            .lower_b
            .insert("other".to_string(), arr1(&[0.0, -1.0]));
        let attached = cache.attach_frontiers(HashMap::from([
            ("relu".to_string(), frontier.clone()),
            ("absent".to_string(), frontier.clone()),
            ("other".to_string(), frontier.clone()),
        ]));
        assert_eq!(attached, 1, "only the exact partner block may carry it");
        assert!(cache.frontier("relu").is_some());
        assert!(cache.frontier("absent").is_none());
        assert!(cache.frontier("other").is_none());
        assert!(cache.estimated_owned_bytes() > 0);

        let split = cache.split_multi_row(2).expect("split");
        assert_eq!(split.len(), 2);
        for part in &split {
            let sliced = part.frontier("relu").expect("sliced frontier");
            assert_eq!(Some(sliced.branch_digest()), part.block_digest("relu"));
        }
        let restacked =
            CachedLinearBounds::stack_single_row(&[&split[0], &split[1]]).expect("stack");
        assert_eq!(restacked.frontier("relu"), Some(&frontier));
    }

    /// #la-union-merge: a fresh pass that overwrites a node's block evicts the
    /// inherited frontier paired with the old block and brings its own; an
    /// untouched node keeps its inherited pair.
    #[test]
    fn warm_frontier_union_merge_never_pairs_a_fresh_block_with_an_inherited_frontier() {
        let old_block = partner();
        let new_block = block([[0.75, 0.0]], -0.5);
        let pending = || vec![("skip".to_string(), block([[1.0, 1.0]], 0.0))];
        let old_frontier = recorded(
            WarmFrontier::capture(&old_block, pending(), |_| None, None).expect("capture"),
        );
        let kept_frontier = recorded(
            WarmFrontier::capture(&old_block, pending(), |_| None, None).expect("capture"),
        );
        let mut inherited = CachedLinearBounds::from_linear_bounds_map(HashMap::from([
            ("n".to_string(), old_block.clone()),
            ("kept".to_string(), old_block),
        ]))
        .with_frontiers(HashMap::from([
            ("n".to_string(), old_frontier),
            ("kept".to_string(), kept_frontier.clone()),
        ]));
        assert_eq!(inherited.frontier_count(), 2);

        // Fresh pass without a frontier at `n`: the stale pair must go.
        let fresh = CachedLinearBounds::from_linear_bounds_map(HashMap::from([(
            "n".to_string(),
            new_block.clone(),
        )]));
        let mut merged = inherited.clone();
        merged.union_with_fresh(fresh);
        assert!(
            merged.frontier("n").is_none(),
            "stale inherited frontier kept"
        );
        assert_eq!(merged.frontier("kept"), Some(&kept_frontier));

        // Fresh pass WITH its own frontier at `n`: adopted.
        let fresh_frontier = recorded(
            WarmFrontier::capture(&new_block, pending(), |_| None, None).expect("capture"),
        );
        let fresh = CachedLinearBounds::from_linear_bounds_map(HashMap::from([(
            "n".to_string(),
            new_block,
        )]))
        .with_frontiers(HashMap::from([("n".to_string(), fresh_frontier.clone())]));
        inherited.union_with_fresh(fresh);
        assert_eq!(inherited.frontier("n"), Some(&fresh_frontier));
    }

    /// DARK BIT-IDENTITY of the queue census (review). `estimated_owned_bytes`
    /// feeds queue eviction and adaptive microbatch sizing, so a cache that
    /// carries no frontier must be charged EXACTLY the historical figure: the
    /// inline size of the four pre-frontier maps plus every key and payload.
    /// Only a cache that really carries a frontier pays for the fifth map and
    /// the frontier's own bytes.
    #[test]
    fn warm_frontier_dark_cache_census_is_the_historical_four_map_figure() {
        let main = two_row_partner();
        let mut cache = CachedLinearBounds::from_linear_bounds_map(HashMap::from([
            ("relu".to_string(), main.clone()),
            ("lin".to_string(), main.clone()),
        ]));
        assert_eq!(cache.frontier_count(), 0);
        let historical_inline = 2 * size_of::<HashMap<String, Array2<f32>>>()
            + 2 * size_of::<HashMap<String, Array1<f32>>>();
        // Per node: lower_a + upper_a (2x2 f32 each) and lower_b + upper_b
        // (2 f32 each), every map entry charged its key.
        let per_node = |name: &str| {
            2 * (name.len() + 4 * size_of::<f32>()) + 2 * (name.len() + 2 * size_of::<f32>())
        };
        let historical = historical_inline + per_node("relu") + per_node("lin");
        assert_eq!(
            cache.estimated_owned_bytes(),
            historical,
            "a dark cache must be charged the historical four-map census byte for byte"
        );

        let skip = LinearBounds::new(
            arr2(&[[0.5, 0.5], [0.25, -0.25]]),
            arr1(&[0.1, 0.2]),
            arr2(&[[0.5, 0.5], [0.25, -0.25]]),
            arr1(&[0.3, 0.4]),
        )
        .expect("skip");
        let frontier = recorded(
            WarmFrontier::capture(&main, vec![("skip".to_string(), skip)], |_| None, None)
                .expect("capture"),
        );
        assert_eq!(
            cache.attach_frontiers(HashMap::from([("relu".to_string(), frontier)])),
            1
        );
        assert_eq!(
            cache.estimated_owned_bytes(),
            historical + size_of::<HashMap<String, WarmFrontier>>() + cache.frontier_owned_bytes(),
            "an armed cache pays for the frontier map and every frontier byte"
        );
        assert!(cache.frontier_owned_bytes() > 0);
    }
}
