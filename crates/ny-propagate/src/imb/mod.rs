// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Input-Manifold Bound (IMB) partition proposer — default-OFF and fail-closed.
//!
//! The cGAN-specific decomposed `(p,q)` path, seam samples, and prefix
//! branch-and-bound are proposal and telemetry mechanisms. They may discover a
//! useful terminal input partition, but they do not carry verdict authority.
//!
//! # Stage discipline
//!
//! - **STAGE 0** (plumbing): the whole path is gated behind [`enabled`]
//!   (`NY_IMB=1`); with the gate off no proposal, allocation, or authority path
//!   runs. Bounds and verdict behavior remain identical; only the cheap exact
//!   environment-gate checks exist on the surrounding call path.
//! - **Proposal-only mode:** with `NY_IMB=1` and no `NY_IMB_WIRE=1`, the path logs
//!   its proposal and returns the caller's baseline verbatim.
//! - **Wired mode:** with both gates enabled, ny first reconstructs an exact cover
//!   from the terminal boxes, then either independently replays the original full
//!   network objective on every leaf or, under exact
//!   `NY_IMB_TAIL_CERT_AY=1`, uses NY's sound prefix lower either as an exact
//!   AY+ny-cert reachability premise for the original tail objective (regional
//!   default) or in the legacy residual composition. Only those certificate
//!   paths can raise a baseline bound.
//!
//! The replay certificate carries an absolute deadline. Invalid configuration,
//! incomplete coverage, unsupported operations, non-finite bounds, or expiry at
//! validation, evaluation, return, or consumption all leave the baseline unchanged.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::dd::{next_down_f64, next_up_f64};
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::GraphNetwork;

pub mod dual_seam_certificate;
mod raf;
pub mod root_inject;
mod tail_grad;

/// Independently checked AY proof for one fixed post-seam affine residual.
///
/// `q` certifies
/// `objective · tail(seam_value) - p · seam_value >= q`
/// throughout the supplied seam box. A proposal solve, when used, has no
/// authority. Region callers may instead supply the exact lower threshold they
/// need and skip optimization. In either mode the installed oracle may
/// construct this token only after exact AY proof validation and an independent
/// ny-cert replay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AyTailCertificate {
    /// Exact-proof-certified binary32 lower bound on the affine residual.
    q: f32,
    /// Number of leaves in AY's exact case-split proof (zero for root Farkas).
    ay_tree_leaves: usize,
    /// Number of Farkas obligations independently accepted by ny-cert.
    ny_cert_farkas_replays: usize,
}

impl AyTailCertificate {
    /// Construct the opaque result of an independently verified AY tail proof.
    ///
    /// Ordinary safe callers cannot mint verdict authority:
    ///
    /// ```compile_fail
    /// use ny_propagate::imb::AyTailCertificate;
    ///
    /// let _forged = AyTailCertificate {
    ///     q: f32::MAX,
    ///     ay_tree_leaves: 0,
    ///     ny_cert_farkas_replays: 1,
    /// };
    /// ```
    ///
    /// The only constructor is itself outside safe Rust:
    ///
    /// ```compile_fail
    /// let _forged =
    ///     ny_propagate::imb::AyTailCertificate::from_independently_verified_parts(
    ///         f32::MAX,
    ///         0,
    ///         1,
    ///     );
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must have established that `q` is a finite lower bound on
    /// `objective · tail(y) - p · y` for the *same* tail, seam box, objective,
    /// `p`, and optional requested threshold supplied to the registered oracle
    /// invocation. A proposal-derived `q` must be rounded outward; a requested
    /// binary32 `q` must be transported exactly into its decision row. AY's
    /// exact root/tree proof must have verified against that original decision
    /// model, every Farkas obligation must have been independently
    /// reconstructed and accepted by ny-cert, and the counters must describe
    /// those accepted obligations exactly.
    #[allow(unsafe_code)]
    pub unsafe fn from_independently_verified_parts(
        q: f32,
        ay_tree_leaves: usize,
        ny_cert_farkas_replays: usize,
    ) -> Self {
        Self {
            q,
            ay_tree_leaves,
            ny_cert_farkas_replays,
        }
    }

    /// Exact-proof-certified binary32 lower bound on the affine residual.
    #[must_use]
    pub fn q(self) -> f32 {
        self.q
    }

    /// Number of leaves in AY's case-split proof (zero for root Farkas).
    #[must_use]
    pub fn ay_tree_leaves(self) -> usize {
        self.ay_tree_leaves
    }

    /// Number of Farkas obligations independently accepted by ny-cert.
    #[must_use]
    pub fn ny_cert_farkas_replays(self) -> usize {
        self.ny_cert_farkas_replays
    }
}

/// Independently checked AY proof under one certified prefix-reachability fact.
///
/// The supplied prefix proof establishes `p · seam_value >= prefix_lower` for
/// every execution in one input region. This token certifies the original tail
/// objective throughout the intersection of the supplied seam box and that
/// affine premise. It is deliberately distinct from [`AyTailCertificate`]:
/// conditional reachability authority must never be consumed as an
/// unconditional residual lower bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AyTailReachabilityCertificate {
    /// Exact-proof-certified binary32 lower threshold on the original objective.
    lower: f32,
    /// Exact binary32 regional prefix premise used by the proof.
    prefix_lower: f32,
    /// Number of leaves in AY's exact case-split proof (zero for root Farkas).
    ay_tree_leaves: usize,
    /// Number of Farkas obligations independently accepted by ny-cert.
    ny_cert_farkas_replays: usize,
}

impl AyTailReachabilityCertificate {
    /// Construct the opaque result of an independently verified conditional
    /// tail proof.
    ///
    /// # Safety
    ///
    /// The caller must have established that `objective · tail(y) >
    /// requested_lower` for every `y` in the supplied seam box satisfying
    /// `p · y >= prefix_lower`, for the exact tail, objective, `p`, prefix
    /// lower, requested threshold, and deadline supplied to the registered
    /// oracle invocation. The affine premise and decision row must transport
    /// every binary32 argument exactly into the rational AY model. AY's exact
    /// root/tree proof must verify against that model, every Farkas obligation
    /// must be independently reconstructed and accepted by ny-cert, and the
    /// counters must describe those accepted obligations exactly.
    #[allow(unsafe_code)]
    pub unsafe fn from_independently_verified_parts(
        lower: f32,
        prefix_lower: f32,
        ay_tree_leaves: usize,
        ny_cert_farkas_replays: usize,
    ) -> Self {
        Self {
            lower,
            prefix_lower,
            ay_tree_leaves,
            ny_cert_farkas_replays,
        }
    }

    /// Exact-proof-certified binary32 lower threshold on the original objective.
    #[must_use]
    pub fn lower(self) -> f32 {
        self.lower
    }

    /// Exact binary32 regional prefix premise used by the proof.
    #[must_use]
    pub fn prefix_lower(self) -> f32 {
        self.prefix_lower
    }

    /// Number of leaves in AY's case-split proof (zero for root Farkas).
    #[must_use]
    pub fn ay_tree_leaves(self) -> usize {
        self.ay_tree_leaves
    }

    /// Number of Farkas obligations independently accepted by ny-cert.
    #[must_use]
    pub fn ny_cert_farkas_replays(self) -> usize {
        self.ny_cert_farkas_replays
    }
}

/// Fixed support-bank width for the first relational prefix-reachability lane.
///
/// Two rows cover the two distinct tail-relaxation modes observed on the sealed
/// cGAN row while keeping both the prefix backward and the exact tail model
/// narrow. This is an immutable authority cap, not an environment-tunable
/// resource knob.
pub const AY_TAIL_AFFINE_REACHABILITY_ROWS: usize = 2;

/// Closed, descending support widths admitted by the shared root-input lane.
///
/// These are authority constants rather than environment knobs. The shared
/// producer remains K4, while the selector route has distinct exact K2/K4
/// canaries; K8/K16 remain dark encoder/type capabilities for a later
/// cost-qualified canary. Malformed or lower-rank banks fail closed.
pub const AY_TAIL_SHARED_INPUT_SUPPORT_ROWS: [usize; 4] = [16, 8, 4, 2];

/// Maximum immutable public payload for one shared root-input support bank.
pub const AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES: usize = 256 * 1024;

/// Narrow, dark payload ceiling for the CIFAR100 compact K16 tail executor.
///
/// The ordinary cGAN shared-input lane retains
/// [`AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES`]. This larger ceiling is admitted
/// only for the exact `Gemm_56`, 3,072-input, 100-seam, K16 shape and merely
/// permits a live prefix-CROWN producer to hand an opaque bank to the
/// default-off executor.
pub const AY_TAIL_COMPACT_K16_MAX_BANK_BYTES: usize = 512 * 1024;
pub const AY_TAIL_COMPACT_K16_INPUTS: usize = 3_072;
pub const AY_TAIL_COMPACT_K16_SEAM_ELEMENTS: usize = 100;
pub const AY_TAIL_COMPACT_K16_SUPPORTS: usize = 16;
pub const AY_TAIL_COMPACT_K16_SEAM_NODE: &str = "Gemm_56";

/// Exact regional disjunct count admitted by the synthetic-selector lane.
pub const AY_TAIL_REGION_SELECTOR_REGIONS: usize = 16;

/// Binary selector width whose canonical assignments enumerate all 16 regions.
pub const AY_TAIL_REGION_SELECTOR_BITS: usize = 4;

/// Closed original-input shape admitted by the selector-conditioned K4 lift.
///
/// The sealed cGAN region grid has five latent coordinates. Its four
/// non-negligible-width coordinates are split in this exact order, which is
/// also the low-bit-first selector order used by [`AyTailRegionSelectorEnvelope`].
pub const AY_TAIL_REGION_SELECTOR_K4_INPUTS: usize = 5;
pub const AY_TAIL_REGION_SELECTOR_K2_SUPPORTS: usize = 2;
pub const AY_TAIL_REGION_SELECTOR_K4_SUPPORTS: usize = 4;
pub const AY_TAIL_REGION_SELECTOR_K4_SPLIT_DIMS: [usize; AY_TAIL_REGION_SELECTOR_BITS] =
    [0, 1, 2, 4];

/// Maximum number of objective-independent root tail anchors carried by one
/// selector request.
///
/// The sealed cGAN tail has exactly two non-input ReLU sources. Four leaves
/// narrow headroom for similarly small tails without turning the opaque
/// envelope into an unpriced full-graph node-bound transport.
pub const AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHORS: usize = 4;

/// Maximum total tensor elements across all selector root tail anchors.
///
/// The two measured cGAN anchors contain 1,536 elements. This immutable cap is
/// deliberately independent of environment configuration.
pub const AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_ELEMENTS: usize = 4_096;

/// Maximum immutable payload for selector root tail anchors.
///
/// This charges node-name bytes, shape dimensions, and both binary32
/// endpoints. The measured two-anchor cGAN payload is about 12 KiB.
pub const AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_BYTES: usize = 32 * 1024;

/// Maximum UTF-8 byte length of one request-bound tail-anchor node identity.
pub const AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_NAME_BYTES: usize = 256;

/// Selector regions whose certified tail pre-activation boxes are transported
/// into the exact model.
///
/// The closed target set covers the measured binding region zero plus the next
/// canonical/Gray-scheduled region one. Keeping this authority set immutable
/// makes the added payload and row budget independent of environment
/// configuration.
pub const AY_TAIL_REGION_SELECTOR_RELU_BOUND_REGIONS: [usize; 2] = [0, 1];

/// Maximum number of region/node pre-activation boxes in one selector request.
pub const AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_RECORDS: usize = 8;

/// Maximum aggregate tensor elements across selector regional ReLU boxes.
pub const AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_ELEMENTS: usize = 4_096;

/// Maximum immutable payload for selector regional ReLU boxes.
pub const AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_BYTES: usize = 32 * 1024;

/// Sound K=2 relational outer approximation of one regional prefix image.
///
/// For the exact regional prefix input `x`, seam value `y = h(x)`, and every
/// support row `j`, the prefix CROWN producer has established
///
/// ```text
/// lower_a[j] · x + lower_b[j]
///     <= directions[j] · y
///     <= upper_a[j] · x + upper_b[j].
/// ```
///
/// Keeping the same latent `x` in all four inequalities retains correlation
/// that is lost by independently concretizing the two support directions to
/// scalar floors. Fields are private so safe external code cannot construct or
/// mutate a purported prefix certificate.
#[derive(Clone, Debug)]
pub struct AyTailAffineReachabilityEnvelope {
    seam_node: String,
    region_input: BoundedTensor,
    directions: Array2<f32>,
    lower_a: Array2<f32>,
    lower_b: Array1<f32>,
    upper_a: Array2<f32>,
    upper_b: Array1<f32>,
}

impl AyTailAffineReachabilityEnvelope {
    /// Construct an envelope from a prefix CROWN backward whose certified
    /// coefficient errors have already been discharged outward into the
    /// corresponding biases over `region_input`.
    pub(crate) fn from_prefix_crown(
        seam_node: String,
        region_input: BoundedTensor,
        directions: Array2<f32>,
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
    ) -> Option<Self> {
        let rows = AY_TAIL_AFFINE_REACHABILITY_ROWS;
        let input_dim = region_input.flatten().len();
        let rank_two = if directions.nrows() == rows {
            let first = directions.row(0);
            let second = directions.row(1);
            let mut first_norm2 = 0.0_f64;
            let mut second_norm2 = 0.0_f64;
            let mut dot = 0.0_f64;
            for (&a, &b) in first.iter().zip(second.iter()) {
                first_norm2 += f64::from(a) * f64::from(a);
                second_norm2 += f64::from(b) * f64::from(b);
                dot += f64::from(a) * f64::from(b);
            }
            if first_norm2.is_finite()
                && second_norm2.is_finite()
                && dot.is_finite()
                && first_norm2 > 0.0
                && second_norm2 > 0.0
            {
                let cosine_abs = (dot / (first_norm2 * second_norm2).sqrt()).abs().min(1.0);
                1.0 - cosine_abs > 1.0e-10
            } else {
                false
            }
        } else {
            false
        };
        if seam_node.is_empty()
            || region_input.has_l2_constraint()
            || directions.nrows() != rows
            || directions.ncols() == 0
            || !rank_two
            || lower_a.shape() != [rows, input_dim]
            || upper_a.shape() != [rows, input_dim]
            || lower_b.len() != rows
            || upper_b.len() != rows
            || directions.iter().any(|value| !value.is_finite())
            || lower_a.iter().any(|value| !value.is_finite())
            || upper_a.iter().any(|value| !value.is_finite())
            || lower_b.iter().any(|value| !value.is_finite())
            || upper_b.iter().any(|value| !value.is_finite())
            || region_input
                .lower()
                .iter()
                .zip(region_input.upper())
                .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return None;
        }
        Some(Self {
            seam_node,
            region_input,
            directions,
            lower_a,
            lower_b,
            upper_a,
            upper_b,
        })
    }

    /// Exact seam node identity used when deriving the prefix rows.
    #[must_use]
    pub fn seam_node(&self) -> &str {
        &self.seam_node
    }

    /// Exact regional prefix input box shared by every support row.
    #[must_use]
    pub fn region_input(&self) -> &BoundedTensor {
        &self.region_input
    }

    /// K=2 support rows over the flattened seam value.
    #[must_use]
    pub fn directions(&self) -> &Array2<f32> {
        &self.directions
    }

    /// Input-linear lower coefficients, one row per support direction.
    #[must_use]
    pub fn lower_a(&self) -> &Array2<f32> {
        &self.lower_a
    }

    /// Input-linear lower biases, one per support direction.
    #[must_use]
    pub fn lower_b(&self) -> &Array1<f32> {
        &self.lower_b
    }

    /// Input-linear upper coefficients, one row per support direction.
    #[must_use]
    pub fn upper_a(&self) -> &Array2<f32> {
        &self.upper_a
    }

    /// Input-linear upper biases, one per support direction.
    #[must_use]
    pub fn upper_b(&self) -> &Array1<f32> {
        &self.upper_b
    }
}

fn finite_box_is_inside(root: &BoundedTensor, region: &BoundedTensor) -> bool {
    root.shape() == region.shape()
        && !root.has_l2_constraint()
        && !region.has_l2_constraint()
        && root.lower().len() == root.upper().len()
        && region.lower().len() == region.upper().len()
        && root.lower().len() == region.lower().len()
        && root
            .lower()
            .iter()
            .zip(root.upper())
            .zip(region.lower().iter().zip(region.upper()))
            .all(|((&root_lower, &root_upper), (&lower, &upper))| {
                root_lower.is_finite()
                    && root_upper.is_finite()
                    && lower.is_finite()
                    && upper.is_finite()
                    && root_lower <= lower
                    && lower <= upper
                    && upper <= root_upper
            })
}

fn same_tensor_bits(left: &BoundedTensor, right: &BoundedTensor) -> bool {
    left.shape() == right.shape()
        && left.has_l2_constraint() == right.has_l2_constraint()
        && left.lower().len() == right.lower().len()
        && left.upper().len() == right.upper().len()
        && left
            .lower()
            .iter()
            .zip(right.lower())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
        && left
            .upper()
            .iter()
            .zip(right.upper())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
}

fn support_rows_have_full_rank(directions: &Array2<f32>) -> bool {
    if directions.nrows() == 0 || directions.ncols() == 0 {
        return false;
    }
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(directions.nrows());
    for row in directions.rows() {
        let original: Vec<f64> = row.iter().copied().map(f64::from).collect();
        let original_norm2 = original.iter().map(|value| value * value).sum::<f64>();
        if !original_norm2.is_finite() || original_norm2 <= 0.0 {
            return false;
        }
        let mut residual = original;
        // Re-orthogonalize once. K is capped at 16, so this small deterministic
        // rank check is cheap and avoids admitting duplicated/near-duplicated
        // proposal rows as a purported support basis.
        for _ in 0..2 {
            for unit in &basis {
                let projection = residual
                    .iter()
                    .zip(unit)
                    .map(|(value, direction)| value * direction)
                    .sum::<f64>();
                for (value, direction) in residual.iter_mut().zip(unit) {
                    *value -= projection * direction;
                }
            }
        }
        let residual_norm2 = residual.iter().map(|value| value * value).sum::<f64>();
        if !residual_norm2.is_finite() || residual_norm2 <= original_norm2 * 1.0e-10 {
            return false;
        }
        let inverse_norm = residual_norm2.sqrt().recip();
        basis.push(
            residual
                .into_iter()
                .map(|value| value * inverse_norm)
                .collect(),
        );
    }
    true
}

#[derive(Clone, Debug)]
struct AyTailSharedInputReachabilityBank {
    seam_node: String,
    certified_root_input: BoundedTensor,
    support_indices: Box<[usize]>,
    directions: Array2<f32>,
    lower_a: Array2<f32>,
    lower_b: Array1<f32>,
    upper_a: Array2<f32>,
    upper_b: Array1<f32>,
}

/// Root-valid relational prefix image for an exact shared-input tail model.
///
/// For the exact root input `x`, seam value `y = h(x)`, and every selected
/// support row `j`, one batched prefix CROWN backward proves
///
/// ```text
/// lower_a[j] · x + lower_b[j]
///     <= directions[j] · y
///     <= upper_a[j] · x + upper_b[j].
/// ```
///
/// A model may narrow the latent `x` block to `region_input`; the production
/// global-root canary keeps it equal to `certified_root_input`. The
/// coefficient-error matrices are not stored: the producer must first widen
/// the biases outward over `certified_root_input`. Private fields and an
/// immutable `Arc` prevent safe callers from fabricating or mutating premises.
#[derive(Clone, Debug)]
pub struct AyTailSharedInputReachabilityEnvelope {
    bank: Arc<AyTailSharedInputReachabilityBank>,
    region_input: BoundedTensor,
}

impl AyTailSharedInputReachabilityEnvelope {
    /// Construct one root or subregion view of a root-valid support bank.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_prefix_crown(
        seam_node: String,
        certified_root_input: BoundedTensor,
        region_input: BoundedTensor,
        support_indices: Vec<usize>,
        directions: Array2<f32>,
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
    ) -> Option<Self> {
        let rows = directions.nrows();
        let input_dim = certified_root_input.flatten().len();
        let allowed_rows = AY_TAIL_SHARED_INPUT_SUPPORT_ROWS.contains(&rows);
        let unique_supports = support_indices.len() == rows
            && support_indices
                .iter()
                .enumerate()
                .all(|(idx, value)| !support_indices[..idx].contains(value));
        let scalar_count = directions
            .len()
            .checked_add(lower_a.len())?
            .checked_add(lower_b.len())?
            .checked_add(upper_a.len())?
            .checked_add(upper_b.len())?;
        let bank_bytes = scalar_count
            .checked_mul(size_of::<f32>())?
            .checked_add(support_indices.len().checked_mul(size_of::<usize>())?)?;
        let ordinary_payload = bank_bytes <= AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES;
        let compact_k16_payload = seam_node == AY_TAIL_COMPACT_K16_SEAM_NODE
            && rows == AY_TAIL_COMPACT_K16_SUPPORTS
            && input_dim == AY_TAIL_COMPACT_K16_INPUTS
            && directions.ncols() == AY_TAIL_COMPACT_K16_SEAM_ELEMENTS
            && bank_bytes <= AY_TAIL_COMPACT_K16_MAX_BANK_BYTES;
        if seam_node.is_empty()
            || !allowed_rows
            || !unique_supports
            || !(ordinary_payload || compact_k16_payload)
            || directions.ncols() == 0
            || !support_rows_have_full_rank(&directions)
            || lower_a.shape() != [rows, input_dim]
            || upper_a.shape() != [rows, input_dim]
            || lower_b.len() != rows
            || upper_b.len() != rows
            || directions.iter().any(|value| !value.is_finite())
            || lower_a.iter().any(|value| !value.is_finite())
            || upper_a.iter().any(|value| !value.is_finite())
            || lower_b.iter().any(|value| !value.is_finite())
            || upper_b.iter().any(|value| !value.is_finite())
            || !finite_box_is_inside(&certified_root_input, &region_input)
        {
            return None;
        }
        Some(Self {
            bank: Arc::new(AyTailSharedInputReachabilityBank {
                seam_node,
                certified_root_input,
                support_indices: support_indices.into_boxed_slice(),
                directions,
                lower_a,
                lower_b,
                upper_a,
                upper_b,
            }),
            region_input,
        })
    }

    /// Exact seam-node identity used by the root prefix backward.
    #[must_use]
    pub fn seam_node(&self) -> &str {
        &self.bank.seam_node
    }

    /// Exact root input box over which every support row is valid.
    #[must_use]
    pub fn certified_root_input(&self) -> &BoundedTensor {
        &self.bank.certified_root_input
    }

    /// Exact regional bounds imposed on the bank's shared latent input block.
    #[must_use]
    pub fn region_input(&self) -> &BoundedTensor {
        &self.region_input
    }

    /// Deterministic proposal indices that produced the support basis.
    #[must_use]
    pub fn support_indices(&self) -> &[usize] {
        &self.bank.support_indices
    }

    /// K support rows over the flattened seam value.
    #[must_use]
    pub fn directions(&self) -> &Array2<f32> {
        &self.bank.directions
    }

    /// Root-input-linear lower coefficients.
    #[must_use]
    pub fn lower_a(&self) -> &Array2<f32> {
        &self.bank.lower_a
    }

    /// Root-input-linear lower biases.
    #[must_use]
    pub fn lower_b(&self) -> &Array1<f32> {
        &self.bank.lower_b
    }

    /// Root-input-linear upper coefficients.
    #[must_use]
    pub fn upper_a(&self) -> &Array2<f32> {
        &self.bank.upper_a
    }

    /// Root-input-linear upper biases.
    #[must_use]
    pub fn upper_b(&self) -> &Array1<f32> {
        &self.bank.upper_b
    }

    /// Exact immutable public payload size checked at construction.
    #[must_use]
    pub fn bank_bytes(&self) -> usize {
        let scalar_count = self
            .directions()
            .len()
            .saturating_add(self.lower_a().len())
            .saturating_add(self.lower_b().len())
            .saturating_add(self.upper_a().len())
            .saturating_add(self.upper_b().len());
        scalar_count
            .saturating_mul(size_of::<f32>())
            .saturating_add(
                self.support_indices()
                    .len()
                    .saturating_mul(size_of::<usize>()),
            )
    }
}

fn shared_input_bank_byte_cap(envelope: &AyTailSharedInputReachabilityEnvelope) -> usize {
    if envelope.seam_node() == AY_TAIL_COMPACT_K16_SEAM_NODE
        && envelope.certified_root_input().flatten().len() == AY_TAIL_COMPACT_K16_INPUTS
        && envelope.region_input().flatten().len() == AY_TAIL_COMPACT_K16_INPUTS
        && envelope.directions().shape()
            == [
                AY_TAIL_COMPACT_K16_SUPPORTS,
                AY_TAIL_COMPACT_K16_SEAM_ELEMENTS,
            ]
    {
        AY_TAIL_COMPACT_K16_MAX_BANK_BYTES
    } else {
        AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES
    }
}

/// One certified, objective-independent root enclosure of a tail ReLU source.
///
/// The producer obtains this box by concretizing an identity-spec CROWN
/// backward over the original root input. It is therefore valid for every
/// selector region and may soundly be intersected with the tail-local seam-box
/// enclosure before Graph-MIP encoding. Private fields prevent safe external
/// code from changing either the node identity or the certified endpoints.
#[derive(Clone, Debug)]
pub struct AyTailRootAnchor {
    node_name: String,
    bounds: BoundedTensor,
}

impl AyTailRootAnchor {
    /// Construct one already-certified objective-independent root anchor.
    ///
    /// This remains crate-private: only the checked full-graph CROWN producer
    /// may introduce anchor authority into a selector envelope.
    pub(crate) fn from_certified_root_box(
        node_name: String,
        bounds: BoundedTensor,
    ) -> Option<Self> {
        if node_name.is_empty()
            || node_name.len() > AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_NAME_BYTES
            || bounds.is_empty()
            || bounds.len() > AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_ELEMENTS
            || !finite_box_is_inside(&bounds, &bounds)
        {
            return None;
        }
        Some(Self { node_name, bounds })
    }

    /// Exact tail-node identity whose output is a ReLU pre-activation.
    #[must_use]
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Certified objective-independent root enclosure for that node.
    #[must_use]
    pub fn bounds(&self) -> &BoundedTensor {
        &self.bounds
    }
}

/// One certified regional enclosure of a tail ReLU pre-activation.
///
/// The producer obtains this box by concretizing the same root-valid
/// input-linear map that produced the corresponding
/// [`AyTailRootAnchor`] over one exact selector input region. Private fields
/// prevent safe external code from changing the region, node identity, or
/// endpoints after certification.
#[derive(Clone, Debug)]
pub struct AyTailRegionReluBounds {
    region_index: usize,
    node_name: String,
    bounds: BoundedTensor,
}

impl AyTailRegionReluBounds {
    /// Construct one already-certified regional pre-activation box.
    ///
    /// This remains crate-private: only the checked root-valid CROWN
    /// coefficient path may introduce regional bound authority.
    pub(crate) fn from_certified_region_box(
        region_index: usize,
        node_name: String,
        bounds: BoundedTensor,
    ) -> Option<Self> {
        if !AY_TAIL_REGION_SELECTOR_RELU_BOUND_REGIONS.contains(&region_index)
            || node_name.is_empty()
            || node_name.len() > AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_NAME_BYTES
            || bounds.is_empty()
            || bounds.len() > AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_ELEMENTS
            || !finite_box_is_inside(&bounds, &bounds)
        {
            return None;
        }
        Some(Self {
            region_index,
            node_name,
            bounds,
        })
    }

    /// Canonical selector region whose input box certifies this enclosure.
    #[must_use]
    pub fn region_index(&self) -> usize {
        self.region_index
    }

    /// Exact tail-node identity whose output is a ReLU pre-activation.
    #[must_use]
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Certified regional pre-activation enclosure.
    #[must_use]
    pub fn bounds(&self) -> &BoundedTensor {
        &self.bounds
    }
}

fn checked_region_selector_root_anchor_payload(
    anchors: &[AyTailRootAnchor],
) -> Option<(usize, usize)> {
    if anchors.is_empty() || anchors.len() > AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHORS {
        return None;
    }
    let mut previous_name: Option<&str> = None;
    let mut total_elements = 0usize;
    let mut total_bytes = 0usize;
    for anchor in anchors {
        let name = anchor.node_name();
        let bounds = anchor.bounds();
        if name.is_empty()
            || name.len() > AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_NAME_BYTES
            || previous_name.is_some_and(|previous| previous >= name)
            || bounds.is_empty()
            || !finite_box_is_inside(bounds, bounds)
        {
            return None;
        }
        previous_name = Some(name);
        total_elements = total_elements.checked_add(bounds.len())?;
        let endpoint_bytes = bounds.len().checked_mul(2)?.checked_mul(size_of::<f32>())?;
        let shape_bytes = bounds.shape().len().checked_mul(size_of::<usize>())?;
        total_bytes = total_bytes
            .checked_add(name.len())?
            .checked_add(shape_bytes)?
            .checked_add(endpoint_bytes)?;
    }
    if total_elements > AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_ELEMENTS
        || total_bytes > AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_BYTES
    {
        return None;
    }
    Some((total_elements, total_bytes))
}

fn checked_region_selector_regional_relu_payload(
    records: &[AyTailRegionReluBounds],
    root_anchors: &[AyTailRootAnchor],
) -> Option<(usize, usize)> {
    let expected_records = AY_TAIL_REGION_SELECTOR_RELU_BOUND_REGIONS
        .len()
        .checked_mul(root_anchors.len())?;
    if root_anchors.is_empty()
        || records.len() != expected_records
        || records.len() > AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_RECORDS
    {
        return None;
    }

    let mut total_elements = 0usize;
    let mut total_bytes = 0usize;
    let mut record_index = 0usize;
    for &region_index in &AY_TAIL_REGION_SELECTOR_RELU_BOUND_REGIONS {
        for root_anchor in root_anchors {
            let record = records.get(record_index)?;
            if record.region_index() != region_index
                || record.node_name() != root_anchor.node_name()
                || !finite_box_is_inside(root_anchor.bounds(), record.bounds())
            {
                return None;
            }
            total_elements = total_elements.checked_add(record.bounds().len())?;
            let endpoint_bytes = record
                .bounds()
                .len()
                .checked_mul(2)?
                .checked_mul(size_of::<f32>())?;
            let shape_bytes = record
                .bounds()
                .shape()
                .len()
                .checked_mul(size_of::<usize>())?;
            total_bytes = total_bytes
                .checked_add(size_of::<usize>())?
                .checked_add(record.node_name().len())?
                .checked_add(shape_bytes)?
                .checked_add(endpoint_bytes)?;
            record_index = record_index.checked_add(1)?;
        }
    }
    if total_elements > AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_ELEMENTS
        || total_bytes > AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_BYTES
    {
        return None;
    }
    Some((total_elements, total_bytes))
}

/// One exact 16-region prefix partition encoded in a single tail model.
///
/// Region `r` has the canonical little-endian selector assignment
/// `z[j] = (r >> j) & 1`. Its certified prefix fact is
/// `directions[r] · y >= prefix_floors[r]`. For each row, the constructor
/// computes a directed-down root-seam-box lower `B_r` and a directed-up
/// `M_r >= max(0, prefix_floors[r] - B_r)`, then installs
///
/// ```text
/// directions[r] · y
///   + sum(bit_r[j] == 0 ? M_r : -M_r) z[j]
///   >= row_lowers[r].
/// ```
///
/// At the canonical assignment the row is the certified regional premise
/// (possibly weakened only outward by the directed-down row lower). Every
/// non-canonical assignment differs in at least one bit and therefore relaxes
/// the row to at most `B_r`, making it redundant over `root_seam_box`.
///
/// For the closed target set of regions zero and one, the envelope also
/// carries one root-contained pre-activation box per tail ReLU source. The MIP
/// consumer gates each corresponding ideal single-ReLU hull with these same
/// selector bits while reusing the existing activation binaries.
///
/// The selector is existential from the perspective of prefix reachability:
/// exact coverage guarantees that every real root input belongs to some region,
/// so choosing that region's canonical bits embeds its real seam value in this
/// augmented model. Proving the original objective for every feasible augmented
/// point is consequently sound. Private fields prevent safe callers from
/// replacing the certified boxes or the canonical gated rows.
#[derive(Clone, Debug)]
pub struct AyTailRegionSelectorEnvelope {
    seam_node: String,
    certified_root_input: BoundedTensor,
    region_inputs: Box<[BoundedTensor]>,
    root_seam_box: BoundedTensor,
    root_tail_anchors: Box<[AyTailRootAnchor]>,
    regional_relu_bounds: Box<[AyTailRegionReluBounds]>,
    selector_k2_lift: Option<AyTailSharedInputReachabilityEnvelope>,
    selector_k4_lift: Option<AyTailSharedInputReachabilityEnvelope>,
    directions: Array2<f32>,
    prefix_floors: Array1<f32>,
    global_seam_lowers: Array1<f64>,
    big_m: Array1<f64>,
    selector_coefficients: Array2<f64>,
    row_lowers: Array1<f64>,
}

fn checked_region_selector_input_lift_context(
    seam_node: &str,
    certified_root_input: &BoundedTensor,
    seam_dim: usize,
    expected_supports: usize,
    lift: &AyTailSharedInputReachabilityEnvelope,
) -> bool {
    lift.seam_node() == seam_node
        && matches!(
            expected_supports,
            AY_TAIL_REGION_SELECTOR_K2_SUPPORTS | AY_TAIL_REGION_SELECTOR_K4_SUPPORTS
        )
        && lift.directions().shape() == [expected_supports, seam_dim]
        && lift.support_indices().len() == expected_supports
        && lift
            .support_indices()
            .iter()
            .enumerate()
            .all(|(index, &support)| {
                support < AY_TAIL_REGION_SELECTOR_REGIONS
                    && !lift.support_indices()[..index].contains(&support)
            })
        && lift.certified_root_input().flatten().len() == AY_TAIL_REGION_SELECTOR_K4_INPUTS
        && same_tensor_bits(lift.certified_root_input(), certified_root_input)
        && same_tensor_bits(lift.region_input(), certified_root_input)
        && lift.lower_a().shape() == [expected_supports, AY_TAIL_REGION_SELECTOR_K4_INPUTS]
        && lift.upper_a().shape() == [expected_supports, AY_TAIL_REGION_SELECTOR_K4_INPUTS]
        && lift.lower_b().len() == expected_supports
        && lift.upper_b().len() == expected_supports
        && lift
            .directions()
            .iter()
            .chain(lift.lower_a())
            .chain(lift.lower_b())
            .chain(lift.upper_a())
            .chain(lift.upper_b())
            .all(|value| value.is_finite())
        && lift.bank_bytes() <= AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES
}

fn checked_region_selector_k2_lift_context(
    seam_node: &str,
    certified_root_input: &BoundedTensor,
    seam_dim: usize,
    lift: &AyTailSharedInputReachabilityEnvelope,
) -> bool {
    checked_region_selector_input_lift_context(
        seam_node,
        certified_root_input,
        seam_dim,
        AY_TAIL_REGION_SELECTOR_K2_SUPPORTS,
        lift,
    )
}

fn checked_region_selector_k4_lift_context(
    seam_node: &str,
    certified_root_input: &BoundedTensor,
    seam_dim: usize,
    lift: &AyTailSharedInputReachabilityEnvelope,
) -> bool {
    checked_region_selector_input_lift_context(
        seam_node,
        certified_root_input,
        seam_dim,
        AY_TAIL_REGION_SELECTOR_K4_SUPPORTS,
        lift,
    )
}

fn directed_region_selector_dot_lower(
    direction: ndarray::ArrayView1<'_, f32>,
    root_seam_box: &BoundedTensor,
) -> Option<f64> {
    let flat = root_seam_box.flatten();
    if direction.len() != flat.len() {
        return None;
    }
    let mut lower = 0.0_f64;
    for ((&coefficient, &box_lower), &box_upper) in
        direction.iter().zip(flat.lower()).zip(flat.upper())
    {
        if !coefficient.is_finite() || !box_lower.is_finite() || !box_upper.is_finite() {
            return None;
        }
        let endpoint = if coefficient >= 0.0 {
            box_lower
        } else {
            box_upper
        };
        // A binary32 product is exact in binary64. Only the accumulation needs
        // widening, and doing it after every addition remains sound without a
        // dimension-dependent error estimate.
        let sum = lower + f64::from(coefficient) * f64::from(endpoint);
        if !sum.is_finite() {
            return None;
        }
        lower = next_down_f64(sum);
    }
    lower.is_finite().then_some(lower)
}

fn directed_region_selector_penalty_up(big_m: f64, count: usize) -> Option<f64> {
    if count == 0 || big_m == 0.0 {
        return Some(0.0);
    }
    let mut penalty = 0.0_f64;
    for _ in 0..count {
        penalty = next_up_f64(penalty + big_m);
        if !penalty.is_finite() {
            return None;
        }
    }
    Some(penalty)
}

impl AyTailRegionSelectorEnvelope {
    /// Construct canonical gated rows from 16 independently certified regional
    /// prefix floors.
    ///
    /// The caller must first establish that `region_inputs` exactly cover
    /// `certified_root_input` and that each supplied floor is backed by an exact
    /// prefix frontier for the corresponding region. The constructor is
    /// crate-private so only that checked authority path can create an envelope.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_certified_prefix_frontiers(
        seam_node: String,
        certified_root_input: BoundedTensor,
        region_inputs: Vec<BoundedTensor>,
        root_seam_box: BoundedTensor,
        root_tail_anchors: Vec<AyTailRootAnchor>,
        regional_relu_bounds: Vec<AyTailRegionReluBounds>,
        directions: Array2<f32>,
        prefix_floors: Array1<f32>,
    ) -> Option<Self> {
        Self::from_certified_prefix_frontiers_impl(
            seam_node,
            certified_root_input,
            region_inputs,
            root_seam_box,
            root_tail_anchors,
            regional_relu_bounds,
            None,
            None,
            directions,
            prefix_floors,
        )
    }

    /// Construct the selector envelope with one fresh, objective-specific,
    /// root-valid K2 CROWN bank over the original input.
    ///
    /// This constructor is intentionally distinct from both legacy schema v3
    /// and K4 schema v4. The K2 bank receives schema v5 request identity, so
    /// enabling its exact canary cannot reinterpret an existing certificate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_certified_prefix_frontiers_with_selector_k2_lift(
        seam_node: String,
        certified_root_input: BoundedTensor,
        region_inputs: Vec<BoundedTensor>,
        root_seam_box: BoundedTensor,
        root_tail_anchors: Vec<AyTailRootAnchor>,
        regional_relu_bounds: Vec<AyTailRegionReluBounds>,
        selector_k2_lift: AyTailSharedInputReachabilityEnvelope,
        directions: Array2<f32>,
        prefix_floors: Array1<f32>,
    ) -> Option<Self> {
        Self::from_certified_prefix_frontiers_impl(
            seam_node,
            certified_root_input,
            region_inputs,
            root_seam_box,
            root_tail_anchors,
            regional_relu_bounds,
            Some(selector_k2_lift),
            None,
            directions,
            prefix_floors,
        )
    }

    /// Construct the selector envelope with one fresh, objective-specific,
    /// root-valid K4 CROWN bank over the original input.
    ///
    /// This constructor is intentionally separate from the legacy constructor:
    /// gate-off callers retain the schema-v3 payload and model byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_certified_prefix_frontiers_with_selector_k4_lift(
        seam_node: String,
        certified_root_input: BoundedTensor,
        region_inputs: Vec<BoundedTensor>,
        root_seam_box: BoundedTensor,
        root_tail_anchors: Vec<AyTailRootAnchor>,
        regional_relu_bounds: Vec<AyTailRegionReluBounds>,
        selector_k4_lift: AyTailSharedInputReachabilityEnvelope,
        directions: Array2<f32>,
        prefix_floors: Array1<f32>,
    ) -> Option<Self> {
        Self::from_certified_prefix_frontiers_impl(
            seam_node,
            certified_root_input,
            region_inputs,
            root_seam_box,
            root_tail_anchors,
            regional_relu_bounds,
            None,
            Some(selector_k4_lift),
            directions,
            prefix_floors,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_certified_prefix_frontiers_impl(
        seam_node: String,
        certified_root_input: BoundedTensor,
        region_inputs: Vec<BoundedTensor>,
        root_seam_box: BoundedTensor,
        mut root_tail_anchors: Vec<AyTailRootAnchor>,
        mut regional_relu_bounds: Vec<AyTailRegionReluBounds>,
        selector_k2_lift: Option<AyTailSharedInputReachabilityEnvelope>,
        selector_k4_lift: Option<AyTailSharedInputReachabilityEnvelope>,
        directions: Array2<f32>,
        prefix_floors: Array1<f32>,
    ) -> Option<Self> {
        root_tail_anchors.sort_unstable_by(|left, right| left.node_name.cmp(&right.node_name));
        regional_relu_bounds.sort_unstable_by(|left, right| {
            (left.region_index, left.node_name.as_str())
                .cmp(&(right.region_index, right.node_name.as_str()))
        });
        let seam_dim = root_seam_box.flatten().len();
        if seam_node.is_empty()
            || region_inputs.len() != AY_TAIL_REGION_SELECTOR_REGIONS
            || directions.shape() != [AY_TAIL_REGION_SELECTOR_REGIONS, seam_dim]
            || prefix_floors.len() != AY_TAIL_REGION_SELECTOR_REGIONS
            || seam_dim == 0
            || !finite_box_is_inside(&certified_root_input, &certified_root_input)
            || !finite_box_is_inside(&root_seam_box, &root_seam_box)
            || region_inputs
                .iter()
                .any(|region| !finite_box_is_inside(&certified_root_input, region))
            || directions.iter().any(|value| !value.is_finite())
            || prefix_floors.iter().any(|value| !value.is_finite())
            || (selector_k2_lift.is_some() && selector_k4_lift.is_some())
            || selector_k2_lift.as_ref().is_some_and(|lift| {
                !checked_region_selector_k2_lift_context(
                    &seam_node,
                    &certified_root_input,
                    seam_dim,
                    lift,
                )
            })
            || selector_k4_lift.as_ref().is_some_and(|lift| {
                !checked_region_selector_k4_lift_context(
                    &seam_node,
                    &certified_root_input,
                    seam_dim,
                    lift,
                )
            })
        {
            return None;
        }
        checked_region_selector_root_anchor_payload(&root_tail_anchors)?;
        checked_region_selector_regional_relu_payload(&regional_relu_bounds, &root_tail_anchors)?;

        let mut global_seam_lowers = Array1::zeros(AY_TAIL_REGION_SELECTOR_REGIONS);
        let mut big_m = Array1::zeros(AY_TAIL_REGION_SELECTOR_REGIONS);
        let mut selector_coefficients = Array2::zeros((
            AY_TAIL_REGION_SELECTOR_REGIONS,
            AY_TAIL_REGION_SELECTOR_BITS,
        ));
        let mut row_lowers = Array1::zeros(AY_TAIL_REGION_SELECTOR_REGIONS);

        for region_idx in 0..AY_TAIL_REGION_SELECTOR_REGIONS {
            let box_lower =
                directed_region_selector_dot_lower(directions.row(region_idx), &root_seam_box)?;
            let floor = f64::from(prefix_floors[region_idx]);
            let gap = floor - box_lower;
            if !gap.is_finite() {
                return None;
            }
            let row_big_m = if gap > 0.0 {
                let widened = next_up_f64(gap);
                widened.is_finite().then_some(widened)?
            } else {
                0.0
            };
            let mut one_bits = 0usize;
            for bit_idx in 0..AY_TAIL_REGION_SELECTOR_BITS {
                let desired_one = ((region_idx >> bit_idx) & 1) != 0;
                selector_coefficients[[region_idx, bit_idx]] = if desired_one {
                    one_bits += 1;
                    -row_big_m
                } else {
                    row_big_m
                };
            }
            let penalty = directed_region_selector_penalty_up(row_big_m, one_bits)?;
            let row_lower = if penalty == 0.0 {
                floor
            } else {
                let rounded = next_down_f64(floor - penalty);
                rounded.is_finite().then_some(rounded)?
            };

            global_seam_lowers[region_idx] = box_lower;
            big_m[region_idx] = row_big_m;
            row_lowers[region_idx] = row_lower;
        }

        Some(Self {
            seam_node,
            certified_root_input,
            region_inputs: region_inputs.into_boxed_slice(),
            root_seam_box,
            root_tail_anchors: root_tail_anchors.into_boxed_slice(),
            regional_relu_bounds: regional_relu_bounds.into_boxed_slice(),
            selector_k2_lift,
            selector_k4_lift,
            directions,
            prefix_floors,
            global_seam_lowers,
            big_m,
            selector_coefficients,
            row_lowers,
        })
    }

    /// Exact seam-node identity used by every regional prefix proof.
    #[must_use]
    pub fn seam_node(&self) -> &str {
        &self.seam_node
    }

    /// Exact root input box covered by the 16 regional prefix frontiers.
    #[must_use]
    pub fn certified_root_input(&self) -> &BoundedTensor {
        &self.certified_root_input
    }

    /// Canonically ordered exact regional input boxes.
    #[must_use]
    pub fn region_inputs(&self) -> &[BoundedTensor] {
        &self.region_inputs
    }

    /// Root seam enclosure used to derive every redundant-row lower `B_r`.
    #[must_use]
    pub fn root_seam_box(&self) -> &BoundedTensor {
        &self.root_seam_box
    }

    /// Canonically node-name-sorted objective-independent root tail anchors.
    #[must_use]
    pub fn root_tail_anchors(&self) -> &[AyTailRootAnchor] {
        &self.root_tail_anchors
    }

    /// Canonically `(region_index, node_name)`-sorted certified regional
    /// pre-activation boxes for the targeted selector regions.
    #[must_use]
    pub fn regional_relu_bounds(&self) -> &[AyTailRegionReluBounds] {
        &self.regional_relu_bounds
    }

    /// Fresh objective-specific root K4 bank gated to the canonical selector
    /// input cell, when the explicit production canary is enabled.
    #[must_use]
    pub fn selector_k4_lift(&self) -> Option<&AyTailSharedInputReachabilityEnvelope> {
        self.selector_k4_lift.as_ref()
    }

    /// Fresh objective-specific root K2 bank gated to the canonical selector
    /// input cell, when the exact default-off K2 canary is enabled.
    #[must_use]
    pub fn selector_k2_lift(&self) -> Option<&AyTailSharedInputReachabilityEnvelope> {
        self.selector_k2_lift.as_ref()
    }

    /// One regional prefix direction `p_r` per canonical selector assignment.
    #[must_use]
    pub fn directions(&self) -> &Array2<f32> {
        &self.directions
    }

    /// Independently certified regional prefix floors `L_r`.
    #[must_use]
    pub fn prefix_floors(&self) -> &Array1<f32> {
        &self.prefix_floors
    }

    /// Directed-down root seam-box lower bounds `B_r`.
    #[must_use]
    pub fn global_seam_lowers(&self) -> &Array1<f64> {
        &self.global_seam_lowers
    }

    /// Directed-up gating magnitudes `M_r`.
    #[must_use]
    pub fn big_m(&self) -> &Array1<f64> {
        &self.big_m
    }

    /// Canonical binary-selector coefficients, shaped `[16, 4]`.
    #[must_use]
    pub fn selector_coefficients(&self) -> &Array2<f64> {
        &self.selector_coefficients
    }

    /// Directed-down lower sides for the 16 gated rows.
    #[must_use]
    pub fn row_lowers(&self) -> &Array1<f64> {
        &self.row_lowers
    }
}

fn push_usize_bits(out: &mut Vec<u32>, value: usize) {
    let value = value as u64;
    out.push(value as u32);
    out.push((value >> 32) as u32);
}

fn push_f32_matrix_bits(out: &mut Vec<u32>, values: &Array2<f32>) {
    push_usize_bits(out, values.nrows());
    push_usize_bits(out, values.ncols());
    out.extend(values.iter().map(|value| value.to_bits()));
}

fn push_f32_vector_bits(out: &mut Vec<u32>, values: &Array1<f32>) {
    push_usize_bits(out, values.len());
    out.extend(values.iter().map(|value| value.to_bits()));
}

fn push_f64_bits(out: &mut Vec<u32>, value: f64) {
    let bits = value.to_bits();
    out.push(bits as u32);
    out.push((bits >> 32) as u32);
}

fn push_f64_matrix_bits(out: &mut Vec<u32>, values: &Array2<f64>) {
    push_usize_bits(out, values.nrows());
    push_usize_bits(out, values.ncols());
    for &value in values {
        push_f64_bits(out, value);
    }
}

fn push_f64_vector_bits(out: &mut Vec<u32>, values: &Array1<f64>) {
    push_usize_bits(out, values.len());
    for &value in values {
        push_f64_bits(out, value);
    }
}

fn push_tensor_bits(out: &mut Vec<u32>, tensor: &BoundedTensor) {
    push_usize_bits(out, tensor.shape().len());
    for &dim in tensor.shape() {
        push_usize_bits(out, dim);
    }
    push_usize_bits(out, tensor.lower().len());
    out.extend(tensor.lower().iter().map(|value| value.to_bits()));
    out.extend(tensor.upper().iter().map(|value| value.to_bits()));
}

fn affine_reachability_request_bits(
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailAffineReachabilityEnvelope,
    requested_lower: f32,
) -> Box<[u32]> {
    // Domain separator: "NYAR" / schema v1.
    let mut out = vec![0x4e59_4152, 1];
    push_usize_bits(&mut out, envelope.seam_node.len());
    out.extend(envelope.seam_node.bytes().map(u32::from));
    push_tensor_bits(&mut out, seam_box);
    push_usize_bits(&mut out, objective.len());
    out.extend(objective.iter().map(|value| value.to_bits()));
    push_tensor_bits(&mut out, &envelope.region_input);
    push_f32_matrix_bits(&mut out, &envelope.directions);
    push_f32_matrix_bits(&mut out, &envelope.lower_a);
    push_f32_vector_bits(&mut out, &envelope.lower_b);
    push_f32_matrix_bits(&mut out, &envelope.upper_a);
    push_f32_vector_bits(&mut out, &envelope.upper_b);
    out.push(requested_lower.to_bits());
    out.into_boxed_slice()
}

fn shared_input_reachability_request_bits(
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailSharedInputReachabilityEnvelope,
    requested_lower: f32,
) -> Box<[u32]> {
    // Domain separator: "NYSH" / schema v1.
    let mut out = vec![0x4e59_5348, 1];
    push_usize_bits(&mut out, envelope.seam_node().len());
    out.extend(envelope.seam_node().bytes().map(u32::from));
    push_tensor_bits(&mut out, seam_box);
    push_usize_bits(&mut out, objective.len());
    out.extend(objective.iter().map(|value| value.to_bits()));
    push_tensor_bits(&mut out, envelope.certified_root_input());
    push_tensor_bits(&mut out, envelope.region_input());
    push_usize_bits(&mut out, envelope.support_indices().len());
    for &support_idx in envelope.support_indices() {
        push_usize_bits(&mut out, support_idx);
    }
    push_f32_matrix_bits(&mut out, envelope.directions());
    push_f32_matrix_bits(&mut out, envelope.lower_a());
    push_f32_vector_bits(&mut out, envelope.lower_b());
    push_f32_matrix_bits(&mut out, envelope.upper_a());
    push_f32_vector_bits(&mut out, envelope.upper_b());
    out.push(requested_lower.to_bits());
    out.into_boxed_slice()
}

fn region_selector_request_bits(
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailRegionSelectorEnvelope,
    requested_lower: f32,
) -> Box<[u32]> {
    // Domain separator: "NYSR". Schema v3 remains byte-for-byte unchanged
    // when both canaries are off. Schema v4 continues to mean the complete
    // fresh selector-conditioned K4 bank; schema v5 distinctly binds K2.
    let (schema, selector_input_lift) =
        match (envelope.selector_k2_lift(), envelope.selector_k4_lift()) {
            (None, None) => (3, None),
            (None, Some(lift)) => (4, Some(lift)),
            (Some(lift), None) => (5, Some(lift)),
            // Safe constructors reject this. Retain an unmistakably invalid
            // request identity for mutation/provenance tests instead of
            // silently selecting either bank.
            (Some(_), Some(_)) => (0, None),
        };
    let mut out = vec![0x4e59_5352, schema];
    push_usize_bits(&mut out, AY_TAIL_REGION_SELECTOR_REGIONS);
    push_usize_bits(&mut out, AY_TAIL_REGION_SELECTOR_BITS);
    push_usize_bits(&mut out, envelope.seam_node().len());
    out.extend(envelope.seam_node().bytes().map(u32::from));
    push_tensor_bits(&mut out, seam_box);
    push_usize_bits(&mut out, objective.len());
    out.extend(objective.iter().map(|value| value.to_bits()));
    push_tensor_bits(&mut out, envelope.certified_root_input());
    push_usize_bits(&mut out, envelope.region_inputs().len());
    for region in envelope.region_inputs() {
        push_tensor_bits(&mut out, region);
    }
    push_tensor_bits(&mut out, envelope.root_seam_box());
    push_usize_bits(&mut out, envelope.root_tail_anchors().len());
    for anchor in envelope.root_tail_anchors() {
        push_usize_bits(&mut out, anchor.node_name().len());
        out.extend(anchor.node_name().bytes().map(u32::from));
        push_tensor_bits(&mut out, anchor.bounds());
    }
    push_usize_bits(&mut out, envelope.regional_relu_bounds().len());
    for regional in envelope.regional_relu_bounds() {
        push_usize_bits(&mut out, regional.region_index());
        push_usize_bits(&mut out, regional.node_name().len());
        out.extend(regional.node_name().bytes().map(u32::from));
        push_tensor_bits(&mut out, regional.bounds());
    }
    push_f32_matrix_bits(&mut out, envelope.directions());
    push_f32_vector_bits(&mut out, envelope.prefix_floors());
    push_f64_vector_bits(&mut out, envelope.global_seam_lowers());
    push_f64_vector_bits(&mut out, envelope.big_m());
    push_f64_matrix_bits(&mut out, envelope.selector_coefficients());
    push_f64_vector_bits(&mut out, envelope.row_lowers());
    if let Some(lift) = selector_input_lift {
        push_usize_bits(&mut out, lift.seam_node().len());
        out.extend(lift.seam_node().bytes().map(u32::from));
        push_tensor_bits(&mut out, lift.certified_root_input());
        push_tensor_bits(&mut out, lift.region_input());
        push_usize_bits(&mut out, lift.support_indices().len());
        for &support_idx in lift.support_indices() {
            push_usize_bits(&mut out, support_idx);
        }
        push_f32_matrix_bits(&mut out, lift.directions());
        push_f32_matrix_bits(&mut out, lift.lower_a());
        push_f32_vector_bits(&mut out, lift.lower_b());
        push_f32_matrix_bits(&mut out, lift.upper_a());
        push_f32_vector_bits(&mut out, lift.upper_b());
    }
    out.push(requested_lower.to_bits());
    out.into_boxed_slice()
}

/// Independently checked AY proof under a K=2 relational prefix envelope.
///
/// Unlike the scalar reachability token, this token bit-binds the complete
/// seam box, objective, region box, support bank, input-linear coefficients and
/// biases, requested threshold, and exact deadline, and address-binds the exact
/// run-local tail object for the synchronous oracle call. The snapshot is
/// compared byte-for-byte at the safe boundary; a hash is never proof identity.
#[derive(Clone, Debug)]
pub struct AyTailAffineReachabilityCertificate {
    lower: f32,
    /// Address identity of the exact run-local tail borrowed by the oracle.
    /// The certificate is consumed synchronously before that borrow ends.
    tail_identity: usize,
    request_bits: Box<[u32]>,
    deadline: Instant,
    ay_tree_leaves: usize,
    ny_cert_farkas_replays: usize,
}

impl AyTailAffineReachabilityCertificate {
    /// Construct the opaque result of an independently verified relational
    /// tail proof.
    ///
    /// # Safety
    ///
    /// The caller must have encoded the supplied `tail`, transported every
    /// supplied binary32 coefficient exactly into that immutable AY model,
    /// added the two lower and two upper relational rows against one shared
    /// regional input variable block, proved the requested original-objective
    /// decision row, verified AY's exact relaxation entailment or root/tree
    /// certificate against that same augmented model, and replayed every linear
    /// obligation independently through ny-cert.
    #[allow(unsafe_code)]
    pub unsafe fn from_independently_verified_parts(
        tail: &GraphNetwork,
        tail_seam_box: &BoundedTensor,
        objective: &[f32],
        envelope: &AyTailAffineReachabilityEnvelope,
        requested_lower: f32,
        deadline: Instant,
        lower: f32,
        ay_tree_leaves: usize,
        ny_cert_farkas_replays: usize,
    ) -> Self {
        Self {
            lower,
            tail_identity: std::ptr::from_ref(tail) as usize,
            request_bits: affine_reachability_request_bits(
                tail_seam_box,
                objective,
                envelope,
                requested_lower,
            ),
            deadline,
            ay_tree_leaves,
            ny_cert_farkas_replays,
        }
    }

    /// Exact-proof-certified binary32 lower threshold on the original objective.
    #[must_use]
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Number of leaves in AY's exact case-split proof (zero for a root proof
    /// or exact relaxation entailment).
    #[must_use]
    pub fn ay_tree_leaves(&self) -> usize {
        self.ay_tree_leaves
    }

    /// Number of independently accepted ny-cert linear-obligation replays.
    #[must_use]
    pub fn ny_cert_farkas_replays(&self) -> usize {
        self.ny_cert_farkas_replays
    }
}

/// Independently checked AY proof under one shared root-input support bank.
///
/// The opaque token bit-binds the tail seam box, objective, certified root box,
/// exact regional latent box, deterministic support identities, all support and
/// input-linear coefficients, requested threshold, and exact deadline. It also
/// address-binds the synchronous run-local tail object.
#[derive(Clone, Debug)]
pub struct AyTailSharedInputReachabilityCertificate {
    lower: f32,
    tail_identity: usize,
    request_bits: Box<[u32]>,
    deadline: Instant,
    ay_tree_leaves: usize,
    ny_cert_farkas_replays: usize,
}

impl AyTailSharedInputReachabilityCertificate {
    /// Construct the opaque result of an independently verified shared-input
    /// tail proof.
    ///
    /// # Safety
    ///
    /// The caller must have encoded the supplied `tail`, transported every
    /// binary32 bank coefficient exactly, created one latent block bounded by
    /// the exact `region_input`, added all lower and upper support rows, proved
    /// the requested original-objective decision row, verified AY's exact
    /// relaxation entailment or root/tree certificate against that immutable
    /// augmented model, and independently replayed every linear obligation
    /// through ny-cert.
    #[allow(unsafe_code)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_independently_verified_parts(
        tail: &GraphNetwork,
        tail_seam_box: &BoundedTensor,
        objective: &[f32],
        envelope: &AyTailSharedInputReachabilityEnvelope,
        requested_lower: f32,
        deadline: Instant,
        lower: f32,
        ay_tree_leaves: usize,
        ny_cert_farkas_replays: usize,
    ) -> Self {
        Self {
            lower,
            tail_identity: std::ptr::from_ref(tail) as usize,
            request_bits: shared_input_reachability_request_bits(
                tail_seam_box,
                objective,
                envelope,
                requested_lower,
            ),
            deadline,
            ay_tree_leaves,
            ny_cert_farkas_replays,
        }
    }

    /// Exact-proof-certified binary32 lower threshold on the original objective.
    #[must_use]
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Number of leaves in AY's exact case-split proof.
    #[must_use]
    pub fn ay_tree_leaves(&self) -> usize {
        self.ay_tree_leaves
    }

    /// Number of independently accepted ny-cert linear-obligation replays.
    #[must_use]
    pub fn ny_cert_farkas_replays(&self) -> usize {
        self.ny_cert_farkas_replays
    }
}

/// Independently checked AY proof under one 16-region synthetic selector.
///
/// The opaque token bit-binds the tail seam box, objective, exact root/region
/// input boxes, every certified objective-independent root tail anchor, every
/// targeted regional ReLU pre-activation box, all regional directions and
/// certified floors, every outward-rounded gating constant and row, and, for
/// schema v4/v5, the complete fresh selector-conditioned K4/K2 input bank
/// respectively. It also binds the requested threshold and deadline and
/// address-binds the synchronous run-local tail object.
#[derive(Clone, Debug)]
pub struct AyTailRegionSelectorCertificate {
    lower: f32,
    tail_identity: usize,
    request_bits: Box<[u32]>,
    deadline: Instant,
    ay_tree_leaves: usize,
    ny_cert_farkas_replays: usize,
}

impl AyTailRegionSelectorCertificate {
    /// Construct the opaque result of an independently verified selector proof.
    ///
    /// # Safety
    ///
    /// The caller must have encoded the supplied `tail`, transported every
    /// binary32 and binary64 envelope value exactly into one immutable model,
    /// validated that the request-bound root anchors exactly cover the tail's
    /// non-input ReLU sources, intersected those anchors into independently
    /// rebuilt seam-box bounds without a disjoint fallback, created exactly
    /// four binary selector variables, added exactly the 16 canonical gated
    /// prefix rows, added four selector-gated ideal facets for every globally
    /// unstable tail ReLU in each targeted region without creating another
    /// column, and proved the requested original-objective decision row. If
    /// schema-v4 K4 or schema-v5 K2 evidence is present, the caller must
    /// additionally have created exactly five root-bounded continuous latent
    /// columns. For K4, it must have transported four lower and four upper bank
    /// rows directly against that one shared block. For K2, it must also have
    /// created exactly two continuous support-value columns with finite,
    /// directed-outward enclosures of `P_j y` over the seam-column box, added
    /// two exact equalities `t_j = P_j y`, and transported two lower and two
    /// upper sparse rows `t_j - A_j x` against the same five latent columns.
    /// The K2 caller may additionally replace a dense regional direction by
    /// one support-value coefficient only after exact binary32-bit equality
    /// with that K2 support direction. Directions absent from the K2 bank may
    /// be shared only when their complete binary32 payloads are bit-identical;
    /// each distinct new direction requires one continuous support-value
    /// column with a finite directed-outward seam-box enclosure and one exact
    /// equality `t = P y`. Every one of the 16 regional premise rows must then
    /// retain its original binary64 selector coefficients and lower bound. If
    /// that exact sparse encoding is not selected, all 16 historical dense
    /// regional rows must be transported unchanged. In both cases the caller
    /// must have exact-checked the 2^4 little-endian region topology and added
    /// exactly eight outward selector-to-input bound rows without new
    /// binaries. The accepted proof must be either a root entailment with one
    /// independently replayed linear obligation or a complete 16-leaf
    /// selector tree with all 16 obligations independently replayed by
    /// ny-cert.
    #[allow(unsafe_code)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_independently_verified_parts(
        tail: &GraphNetwork,
        tail_seam_box: &BoundedTensor,
        objective: &[f32],
        envelope: &AyTailRegionSelectorEnvelope,
        requested_lower: f32,
        deadline: Instant,
        lower: f32,
        ay_tree_leaves: usize,
        ny_cert_farkas_replays: usize,
    ) -> Self {
        Self {
            lower,
            tail_identity: std::ptr::from_ref(tail) as usize,
            request_bits: region_selector_request_bits(
                tail_seam_box,
                objective,
                envelope,
                requested_lower,
            ),
            deadline,
            ay_tree_leaves,
            ny_cert_farkas_replays,
        }
    }

    /// Exact-proof-certified binary32 lower threshold on the original objective.
    #[must_use]
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Number of exact selector-tree leaves (zero for root entailment).
    #[must_use]
    pub fn ay_tree_leaves(&self) -> usize {
        self.ay_tree_leaves
    }

    /// Number of independently accepted ny-cert obligation replays.
    #[must_use]
    pub fn ny_cert_farkas_replays(&self) -> usize {
        self.ny_cert_farkas_replays
    }
}

/// Optional implementation seam for the small post-seam cGAN MILP.
///
/// `ny-propagate` deliberately does not depend on a solver crate.  The CLI's
/// MIP feature installs one process-global function at startup; library users
/// that do not install it simply decline the opt-in certificate lane.
pub type AyTailCertificateOracle = fn(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    objective: &[f32],
    p: &[f32],
    requested_lower: Option<f32>,
    deadline: Instant,
) -> Option<AyTailCertificate>;

static AY_TAIL_CERTIFICATE_ORACLE: OnceLock<AyTailCertificateOracle> = OnceLock::new();

/// Optional implementation seam for a post-seam cGAN MILP strengthened by one
/// certified prefix-reachability fact.
///
/// `prefix_lower` is the sound regional prefix lower on `p · seam_value`.
/// `requested_lower` is the exact binary32 threshold the oracle must prove for
/// the original `objective · tail(seam_value)` expression.
pub type AyTailReachabilityCertificateOracle = fn(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    p: &[f32],
    prefix_lower: f32,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailReachabilityCertificate>;

static AY_TAIL_REACHABILITY_CERTIFICATE_ORACLE: OnceLock<AyTailReachabilityCertificateOracle> =
    OnceLock::new();

/// Optional implementation seam for a post-seam cGAN MILP strengthened by a
/// K=2 input-relational prefix envelope.
pub type AyTailAffineReachabilityCertificateOracle =
    fn(
        tail: &GraphNetwork,
        seam_box: &BoundedTensor,
        objective: &[f32],
        envelope: &AyTailAffineReachabilityEnvelope,
        requested_lower: f32,
        deadline: Instant,
    ) -> Option<AyTailAffineReachabilityCertificate>;

static AY_TAIL_AFFINE_REACHABILITY_CERTIFICATE_ORACLE: OnceLock<
    AyTailAffineReachabilityCertificateOracle,
> = OnceLock::new();

/// Optional implementation seam for a post-seam cGAN MILP strengthened by one
/// root-valid support bank and a region-bounded shared prefix-input block.
pub type AyTailSharedInputReachabilityCertificateOracle =
    fn(
        tail: &GraphNetwork,
        seam_box: &BoundedTensor,
        objective: &[f32],
        envelope: &AyTailSharedInputReachabilityEnvelope,
        requested_lower: f32,
        deadline: Instant,
    ) -> Option<AyTailSharedInputReachabilityCertificate>;

static AY_TAIL_SHARED_INPUT_REACHABILITY_CERTIFICATE_ORACLE: OnceLock<
    AyTailSharedInputReachabilityCertificateOracle,
> = OnceLock::new();

/// Optional implementation seam for one post-seam exact model strengthened by
/// 16 regional prefix rows gated by four synthetic binary selectors.
pub type AyTailRegionSelectorCertificateOracle = fn(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailRegionSelectorEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailRegionSelectorCertificate>;

static AY_TAIL_REGION_SELECTOR_CERTIFICATE_ORACLE: OnceLock<AyTailRegionSelectorCertificateOracle> =
    OnceLock::new();

/// Install the process-global post-seam AY certificate oracle.
///
/// Returns `false` when an oracle was already installed.  Installation alone
/// changes no behavior: the call site additionally requires the exact runtime
/// gate `NY_IMB_TAIL_CERT_AY=1`.
///
/// This is deliberately an unsafe verdict-authority boundary. Safe external
/// code cannot register an arbitrary callback:
///
/// ```compile_fail
/// let forged: ny_propagate::imb::AyTailCertificateOracle =
///     |_, _, _, _, _, _, _| None;
/// let _ = ny_propagate::imb::install_ay_tail_certificate_oracle(forged);
/// ```
///
/// # Safety
///
/// The callback must return a certificate only when it was constructed under
/// [`AyTailCertificate::from_independently_verified_parts`]'s contract for the
/// exact arguments of that invocation. It must never convert an optimization
/// answer, sampled check, timeout, or proof for a different request into a
/// certificate.
///
/// This is a compile-time safe-Rust boundary, not a cryptographic capability:
/// any caller choosing to invoke these APIs through `unsafe` is explicitly
/// inside the verifier's trusted computing base.
#[allow(unsafe_code)]
pub unsafe fn install_ay_tail_certificate_oracle(oracle: AyTailCertificateOracle) -> bool {
    AY_TAIL_CERTIFICATE_ORACLE.set(oracle).is_ok()
}

/// Install the process-global conditional post-seam AY certificate oracle.
///
/// Installation alone changes no behavior: the call site additionally
/// requires the exact runtime gate `NY_IMB_TAIL_CERT_AY=1`.
///
/// # Safety
///
/// The callback must return a certificate only when it was constructed under
/// [`AyTailReachabilityCertificate::from_independently_verified_parts`]'s
/// contract for the exact arguments of that invocation. It must never turn an
/// optimization answer, sampled fact, unvalidated prefix proposal, timeout, or
/// proof for a different affine premise into a certificate.
#[allow(unsafe_code)]
pub unsafe fn install_ay_tail_reachability_certificate_oracle(
    oracle: AyTailReachabilityCertificateOracle,
) -> bool {
    AY_TAIL_REACHABILITY_CERTIFICATE_ORACLE.set(oracle).is_ok()
}

/// Install the process-global relational post-seam AY certificate oracle.
///
/// # Safety
///
/// The callback must return a certificate only after proving the exact
/// original-objective decision row over the exact tail model augmented with
/// the supplied immutable K=2 envelope. It must obey
/// [`AyTailAffineReachabilityCertificate::from_independently_verified_parts`]'s
/// contract for the exact invocation arguments.
#[allow(unsafe_code)]
pub unsafe fn install_ay_tail_affine_reachability_certificate_oracle(
    oracle: AyTailAffineReachabilityCertificateOracle,
) -> bool {
    AY_TAIL_AFFINE_REACHABILITY_CERTIFICATE_ORACLE
        .set(oracle)
        .is_ok()
}

/// Install the process-global shared-root-input AY certificate oracle.
///
/// # Safety
///
/// The callback must obey
/// [`AyTailSharedInputReachabilityCertificate::from_independently_verified_parts`]'s
/// contract for the exact invocation and may never authorize a proposal,
/// optimization answer, malformed/substitute bank, or expired proof.
#[allow(unsafe_code)]
pub unsafe fn install_ay_tail_shared_input_reachability_certificate_oracle(
    oracle: AyTailSharedInputReachabilityCertificateOracle,
) -> bool {
    AY_TAIL_SHARED_INPUT_REACHABILITY_CERTIFICATE_ORACLE
        .set(oracle)
        .is_ok()
}

/// Install the process-global 16-region synthetic-selector AY oracle.
///
/// # Safety
///
/// The callback must obey
/// [`AyTailRegionSelectorCertificate::from_independently_verified_parts`]'s
/// contract for the exact invocation. It may never authorize an optimization
/// answer, a non-canonical selector encoding, an incomplete selector tree, a
/// substitute envelope, or an expired proof.
#[allow(unsafe_code)]
pub unsafe fn install_ay_tail_region_selector_certificate_oracle(
    oracle: AyTailRegionSelectorCertificateOracle,
) -> bool {
    AY_TAIL_REGION_SELECTOR_CERTIFICATE_ORACLE
        .set(oracle)
        .is_ok()
}

fn ay_tail_certificate_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(crate) fn ay_tail_certificate_enabled() -> bool {
    ay_tail_certificate_enabled_from_value(std::env::var("NY_IMB_TAIL_CERT_AY").ok().as_deref())
}

fn ay_tail_certificate_matches_request(
    certificate: AyTailCertificate,
    requested_lower: Option<f32>,
) -> bool {
    certificate.q.is_finite()
        && requested_lower.is_none_or(|lower| certificate.q.to_bits() == lower.to_bits())
        && certificate.ny_cert_farkas_replays != 0
        && (certificate.ay_tree_leaves == 0
            || certificate.ay_tree_leaves == certificate.ny_cert_farkas_replays)
}

fn ay_tail_reachability_certificate_matches_request(
    certificate: AyTailReachabilityCertificate,
    prefix_lower: f32,
    requested_lower: f32,
) -> bool {
    certificate.lower.is_finite()
        && certificate.lower.to_bits() == requested_lower.to_bits()
        && certificate.prefix_lower.is_finite()
        && certificate.prefix_lower.to_bits() == prefix_lower.to_bits()
        && certificate.ny_cert_farkas_replays != 0
        && (certificate.ay_tree_leaves == 0
            || certificate.ay_tree_leaves == certificate.ny_cert_farkas_replays)
}

fn ay_tail_affine_reachability_certificate_matches_request(
    certificate: &AyTailAffineReachabilityCertificate,
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailAffineReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> bool {
    certificate.lower.is_finite()
        && certificate.lower.to_bits() == requested_lower.to_bits()
        && certificate.tail_identity == std::ptr::from_ref(tail) as usize
        && certificate.deadline == deadline
        && certificate.request_bits.as_ref()
            == affine_reachability_request_bits(seam_box, objective, envelope, requested_lower)
                .as_ref()
        && certificate.ny_cert_farkas_replays != 0
        && (certificate.ay_tree_leaves == 0
            || certificate.ay_tree_leaves == certificate.ny_cert_farkas_replays)
}

fn ay_tail_shared_input_reachability_certificate_matches_request(
    certificate: &AyTailSharedInputReachabilityCertificate,
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailSharedInputReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> bool {
    certificate.lower.is_finite()
        && certificate.lower.to_bits() == requested_lower.to_bits()
        && certificate.tail_identity == std::ptr::from_ref(tail) as usize
        && certificate.deadline == deadline
        && certificate.request_bits.as_ref()
            == shared_input_reachability_request_bits(
                seam_box,
                objective,
                envelope,
                requested_lower,
            )
            .as_ref()
        && certificate.ny_cert_farkas_replays != 0
        && (certificate.ay_tree_leaves == 0
            || certificate.ay_tree_leaves == certificate.ny_cert_farkas_replays)
}

fn ay_tail_region_selector_certificate_matches_request(
    certificate: &AyTailRegionSelectorCertificate,
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailRegionSelectorEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> bool {
    certificate.lower.is_finite()
        && certificate.lower.to_bits() == requested_lower.to_bits()
        && certificate.tail_identity == std::ptr::from_ref(tail) as usize
        && certificate.deadline == deadline
        && certificate.request_bits.as_ref()
            == region_selector_request_bits(seam_box, objective, envelope, requested_lower).as_ref()
        && matches!(
            (
                certificate.ay_tree_leaves,
                certificate.ny_cert_farkas_replays
            ),
            (0, 1)
                | (
                    AY_TAIL_REGION_SELECTOR_REGIONS,
                    AY_TAIL_REGION_SELECTOR_REGIONS
                )
        )
}

pub(crate) fn certify_tail_with_ay(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    objective: &[f32],
    p: &[f32],
    requested_lower: Option<f32>,
    deadline: Instant,
) -> Option<AyTailCertificate> {
    if !ay_tail_certificate_enabled()
        || Instant::now() >= deadline
        || requested_lower.is_some_and(|value| !value.is_finite())
    {
        return None;
    }
    let oracle = AY_TAIL_CERTIFICATE_ORACLE.get()?;
    let certificate = oracle(
        tail,
        seam_box,
        node_bounds,
        objective,
        p,
        requested_lower,
        deadline,
    )?;
    if Instant::now() >= deadline
        || !ay_tail_certificate_matches_request(certificate, requested_lower)
    {
        return None;
    }
    Some(certificate)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn certify_tail_with_ay_reachability(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    p: &[f32],
    prefix_lower: f32,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailReachabilityCertificate> {
    if !ay_tail_certificate_enabled()
        || Instant::now() >= deadline
        || !prefix_lower.is_finite()
        || !requested_lower.is_finite()
    {
        return None;
    }
    let oracle = AY_TAIL_REACHABILITY_CERTIFICATE_ORACLE.get()?;
    let certificate = oracle(
        tail,
        seam_box,
        objective,
        p,
        prefix_lower,
        requested_lower,
        deadline,
    )?;
    if Instant::now() >= deadline
        || !ay_tail_reachability_certificate_matches_request(
            certificate,
            prefix_lower,
            requested_lower,
        )
    {
        return None;
    }
    Some(certificate)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn certify_tail_with_ay_affine_reachability(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailAffineReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailAffineReachabilityCertificate> {
    if !ay_tail_certificate_enabled()
        || Instant::now() >= deadline
        || !requested_lower.is_finite()
        || envelope.directions.ncols() != seam_box.flatten().len()
    {
        return None;
    }
    let oracle = AY_TAIL_AFFINE_REACHABILITY_CERTIFICATE_ORACLE.get()?;
    let certificate = oracle(
        tail,
        seam_box,
        objective,
        envelope,
        requested_lower,
        deadline,
    )?;
    if Instant::now() >= deadline
        || !ay_tail_affine_reachability_certificate_matches_request(
            &certificate,
            tail,
            seam_box,
            objective,
            envelope,
            requested_lower,
            deadline,
        )
    {
        return None;
    }
    Some(certificate)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn certify_tail_with_ay_shared_input_reachability(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailSharedInputReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailSharedInputReachabilityCertificate> {
    if !ay_tail_certificate_enabled()
        || Instant::now() >= deadline
        || !requested_lower.is_finite()
        || envelope.directions().ncols() != seam_box.flatten().len()
        || !AY_TAIL_SHARED_INPUT_SUPPORT_ROWS.contains(&envelope.directions().nrows())
        || envelope.bank_bytes() > shared_input_bank_byte_cap(envelope)
    {
        return None;
    }
    let oracle = AY_TAIL_SHARED_INPUT_REACHABILITY_CERTIFICATE_ORACLE.get()?;
    let certificate = oracle(
        tail,
        seam_box,
        objective,
        envelope,
        requested_lower,
        deadline,
    )?;
    if Instant::now() >= deadline
        || !ay_tail_shared_input_reachability_certificate_matches_request(
            &certificate,
            tail,
            seam_box,
            objective,
            envelope,
            requested_lower,
            deadline,
        )
    {
        return None;
    }
    Some(certificate)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn certify_tail_with_ay_region_selector(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailRegionSelectorEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailRegionSelectorCertificate> {
    if !ay_tail_certificate_enabled()
        || Instant::now() >= deadline
        || !requested_lower.is_finite()
        || !same_tensor_bits(seam_box, envelope.root_seam_box())
        || checked_region_selector_root_anchor_payload(envelope.root_tail_anchors()).is_none()
        || checked_region_selector_regional_relu_payload(
            envelope.regional_relu_bounds(),
            envelope.root_tail_anchors(),
        )
        .is_none()
        || envelope.region_inputs().len() != AY_TAIL_REGION_SELECTOR_REGIONS
        || envelope.directions().shape()
            != [AY_TAIL_REGION_SELECTOR_REGIONS, seam_box.flatten().len()]
        || envelope.prefix_floors().len() != AY_TAIL_REGION_SELECTOR_REGIONS
        || envelope.global_seam_lowers().len() != AY_TAIL_REGION_SELECTOR_REGIONS
        || envelope.big_m().len() != AY_TAIL_REGION_SELECTOR_REGIONS
        || envelope.selector_coefficients().shape()
            != [
                AY_TAIL_REGION_SELECTOR_REGIONS,
                AY_TAIL_REGION_SELECTOR_BITS,
            ]
        || envelope.row_lowers().len() != AY_TAIL_REGION_SELECTOR_REGIONS
        || (envelope.selector_k2_lift().is_some() && envelope.selector_k4_lift().is_some())
        || envelope.selector_k2_lift().is_some_and(|lift| {
            !checked_region_selector_k2_lift_context(
                envelope.seam_node(),
                envelope.certified_root_input(),
                seam_box.flatten().len(),
                lift,
            )
        })
        || envelope.selector_k4_lift().is_some_and(|lift| {
            !checked_region_selector_k4_lift_context(
                envelope.seam_node(),
                envelope.certified_root_input(),
                seam_box.flatten().len(),
                lift,
            )
        })
        || envelope
            .global_seam_lowers()
            .iter()
            .chain(envelope.big_m())
            .chain(envelope.selector_coefficients())
            .chain(envelope.row_lowers())
            .any(|value| !value.is_finite())
    {
        return None;
    }
    let oracle = AY_TAIL_REGION_SELECTOR_CERTIFICATE_ORACLE.get()?;
    let certificate = oracle(
        tail,
        seam_box,
        objective,
        envelope,
        requested_lower,
        deadline,
    )?;
    if Instant::now() >= deadline
        || !ay_tail_region_selector_certificate_matches_request(
            &certificate,
            tail,
            seam_box,
            objective,
            envelope,
            requested_lower,
            deadline,
        )
    {
        return None;
    }
    Some(certificate)
}

/// Env gate: IMB is armed only with `NY_IMB=1` (default-OFF — a new
/// soundness-critical bounding mode is opt-in, mirroring `multineuron::enabled`).
pub fn enabled() -> bool {
    matches!(std::env::var("NY_IMB").ok().as_deref(), Some("1"))
}

/// Parse a `usize` env knob with a default (shared by the config sites).
pub(crate) fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Parse an `f64` env knob with a default.
pub(crate) fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

thread_local! {
    /// Re-entrancy guard: the nested per-leaf prefix-BaB bounding must NOT re-arm
    /// IMB at the disjunctive root (it would recurse forever). Set for the whole
    /// `tighten_root_objective_bounds_imb` body via [`scope`].
    static IN_PROGRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether an IMB root injection is already on the stack (re-entrancy guard).
pub(crate) fn in_progress() -> bool {
    IN_PROGRESS.with(|c| c.get())
}

/// RAII guard that marks IMB as in-progress until dropped.
pub(crate) struct ImbScopeGuard {
    _private: (),
}

impl Drop for ImbScopeGuard {
    fn drop(&mut self) {
        IN_PROGRESS.with(|c| c.set(false));
    }
}

/// Enter an IMB scope (sets the re-entrancy guard). Dropping the returned guard
/// clears it. Nested calls short-circuit via [`in_progress`] inside [`armed`].
pub(crate) fn scope() -> ImbScopeGuard {
    IN_PROGRESS.with(|c| c.set(true));
    ImbScopeGuard { _private: () }
}

thread_local! {
    /// When set (only by the IMB anchor build, via [`AnchorChunkParallelGuard`]), the
    /// objective-chunk driver in `propagate_crown_to_node_chunked` runs its
    /// row-independent chunks in PARALLEL (rayon). Default OFF ⇒ that driver stays
    /// sequential — byte-identical for EVERY non-IMB caller (the CROWN-IBP collector's
    /// over-budget chunking included). Read on the calling thread only (the decision
    /// to parallelize is made there, before the fan-out), so no propagation to rayon
    /// workers is needed.
    static ANCHOR_CHUNK_PARALLEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the current thread is inside an IMB anchor build that opts the
/// objective-chunk backward into parallel chunks (read by the core chunk driver).
pub(crate) fn anchor_chunk_parallel() -> bool {
    ANCHOR_CHUNK_PARALLEL.with(|c| c.get())
}

/// RAII guard: while held, `propagate_crown_to_node_chunked` parallelizes its
/// (row-independent, bound-equivalent) objective-row chunks. Wrap the IMB anchor's
/// `propagate_crown_to_node` calls with it. Restores the prior value on drop
/// (nestable / re-entrant-safe).
pub(crate) struct AnchorChunkParallelGuard(bool);

impl AnchorChunkParallelGuard {
    pub(crate) fn enable() -> Self {
        Self(ANCHOR_CHUNK_PARALLEL.with(|c| c.replace(true)))
    }
}

impl Drop for AnchorChunkParallelGuard {
    fn drop(&mut self) {
        ANCHOR_CHUNK_PARALLEL.with(|c| c.set(self.0));
    }
}

thread_local! {
    /// Set once the CLI-level early hook (`try_imb_early_disjunctive`, LEVER 1) has
    /// ATTEMPTED the IMB — whether it verified or fell through. The DOWNSTREAM in-lane
    /// `#imb-early` block (inside `verify_graph_input_split_multi_clause_disjunctive`)
    /// reads it and skips, so the (expensive) leaf-BaB / tail-opt never repeats when
    /// the hook already ran. If the hook never fires (non-Graph model, unsupported
    /// contract, IMB disabled), this stays false and the in-lane block runs as before.
    static IMB_EARLY_ATTEMPTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// One admitted replay-only diagnostic may run per verification. The early
    /// and late IMB entry points share this guard so they cannot evaluate the
    /// same selected leaf twice.
    static IMB_REPLAY_ONLY_ATTEMPTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Record that the CLI early hook attempted the IMB (LEVER 1).
pub(crate) fn mark_early_attempted() {
    IMB_EARLY_ATTEMPTED.with(|c| c.set(true));
}

/// Whether the CLI early hook already attempted the IMB (so the in-lane block skips).
pub(crate) fn early_attempted() -> bool {
    IMB_EARLY_ATTEMPTED.with(|c| c.get())
}

/// Whether an admitted replay-only diagnostic already ran in this verification.
pub(crate) fn replay_only_attempted() -> bool {
    IMB_REPLAY_ONLY_ATTEMPTED.with(|c| c.get())
}

/// Consume the one replay-only diagnostic attempt for this verification.
///
/// Call only after gate, selector, shape, exact-cover, and target admission.
/// Returns `true` only to the first admitted caller.
pub(crate) fn begin_replay_only_attempt() -> bool {
    IMB_REPLAY_ONLY_ATTEMPTED.with(|c| !c.replace(true))
}

/// Clear the early-attempted flag at the START of each disjunctive verify, so the
/// thread-local can't leak across instances if a process is ever reused for more than
/// one instance (ny runs one instance per process today, but this keeps both the normal
/// early suppression and replay-only single-attempt guard scoped to one verify).
/// The CLI calls this at its disjunctive verification boundary; direct reusable
/// lower-level API callers must do the same before starting a new verification.
pub fn reset_early_attempted() {
    IMB_EARLY_ATTEMPTED.with(|c| c.set(false));
    IMB_REPLAY_ONLY_ATTEMPTED.with(|c| c.set(false));
}

/// PROCESS-GLOBAL flag: set (only by [`RegionSeqGuard`], around the parallel region
/// par_iter with `NY_IMB_REGION_THREADS>1`) so every rayon fan-out INSIDE a region's
/// certification runs SEQUENTIALLY — the tail-opt (`rayon::join` over 2 inits + the
/// `sample_min_arg` reduce), the seam sampling, the prefix leaf-BaB batch, and the
/// ConvTranspose backward `par_chunks_mut`. That makes each region single-cored so the
/// N region workers get TRUE N-way parallelism instead of `N regions × inner-fan-out`
/// oversubscribing the box into effective serialism.
///
/// GLOBAL (not thread-local) BY NECESSITY: the fan-out sites run on nested rayon
/// workers (different threads from where the guard is set), which would not inherit a
/// thread-local. It is a FLAG the fan-out sites read to pick the sequential branch — no
/// nested `install()` (which rayon can serialize unpredictably). Set for the region
/// par_iter's whole duration — during which the ONLY parallel work is the per-region
/// certification we want single-cored — and cleared on drop. Default OFF ⇒ leaf-parallel
/// prop_0/prop_1 and every other path keep their exact threading.
static REGION_SEQ_INNER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether we are inside a parallel region loop (⇒ each region's inner work is
/// single-cored). Read by the rayon fan-out sites to pick their sequential branch.
pub(crate) fn region_seq_inner() -> bool {
    REGION_SEQ_INNER.load(std::sync::atomic::Ordering::Relaxed)
}

/// RAII guard enabling [`region_seq_inner`] for its scope. Held by `run_region_loop`
/// around the parallel region par_iter (only when `region_threads > 1`).
pub(crate) struct RegionSeqGuard;

impl RegionSeqGuard {
    pub(crate) fn enable() -> Self {
        REGION_SEQ_INNER.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}

impl Drop for RegionSeqGuard {
    fn drop(&mut self) {
        REGION_SEQ_INNER.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Number of free input scalars (`lower < upper`). The IMB prefix-BaB branches
/// over exactly these, so the arming gate caps it (a high-dim box is not the
/// cGAN-shaped 4-free-dim surface the certificate targets).
pub fn free_input_dims(input: &BoundedTensor) -> usize {
    let flat = input.flatten();
    let lo = flat.lower();
    let hi = flat.upper();
    lo.iter().zip(hi.iter()).filter(|(l, u)| **l < **u).count()
}

/// Whether the graph carries a transposed-convolution generator prefix — the
/// image-generator surface the IMB seam decomposition is designed for. Conv-only
/// discriminators without a ConvTranspose generator are out of scope.
pub fn has_conv_generator_prefix(graph: &GraphNetwork) -> bool {
    graph.nodes.values().any(|node| {
        matches!(
            node.layer,
            Layer::ConvTranspose2d(_) | Layer::ConvTranspose1d(_)
        )
    })
}

/// Arming predicate: IMB runs only when enabled, not already recursing, the free
/// input dimensionality is small enough to branch, and the graph has a
/// ConvTranspose generator prefix. Mirrors `multineuron::root_inject`'s
/// enabled+conv gating but adds the free-dim cap and the re-entrancy guard.
pub fn armed(graph: &GraphNetwork, input: &BoundedTensor) -> bool {
    enabled()
        && !in_progress()
        && free_input_dims(input) <= env_usize("NY_IMB_MAXDIM", 8)
        && has_conv_generator_prefix(graph)
}

#[cfg(test)]
mod ay_tail_tests {
    use super::{
        affine_reachability_request_bits, ay_tail_affine_reachability_certificate_matches_request,
        ay_tail_certificate_enabled_from_value, ay_tail_certificate_matches_request,
        ay_tail_reachability_certificate_matches_request,
        ay_tail_region_selector_certificate_matches_request,
        ay_tail_shared_input_reachability_certificate_matches_request, finite_box_is_inside,
        region_selector_request_bits, shared_input_reachability_request_bits,
        AyTailAffineReachabilityCertificate, AyTailAffineReachabilityEnvelope, AyTailCertificate,
        AyTailReachabilityCertificate, AyTailRegionReluBounds, AyTailRegionSelectorCertificate,
        AyTailRegionSelectorEnvelope, AyTailRootAnchor, AyTailSharedInputReachabilityCertificate,
        AyTailSharedInputReachabilityEnvelope, AY_TAIL_COMPACT_K16_INPUTS,
        AY_TAIL_COMPACT_K16_MAX_BANK_BYTES, AY_TAIL_COMPACT_K16_SEAM_ELEMENTS,
        AY_TAIL_COMPACT_K16_SEAM_NODE, AY_TAIL_COMPACT_K16_SUPPORTS, AY_TAIL_REGION_SELECTOR_BITS,
        AY_TAIL_REGION_SELECTOR_K2_SUPPORTS, AY_TAIL_REGION_SELECTOR_K4_INPUTS,
        AY_TAIL_REGION_SELECTOR_K4_SPLIT_DIMS, AY_TAIL_REGION_SELECTOR_K4_SUPPORTS,
        AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_ELEMENTS, AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHORS,
        AY_TAIL_REGION_SELECTOR_REGIONS, AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES,
    };
    use ndarray::{array, Array1, Array2};
    use ny_tensor::BoundedTensor;
    use std::mem::size_of;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::GraphNetwork;

    fn box2(lower: [f32; 2], upper: [f32; 2]) -> BoundedTensor {
        BoundedTensor::new(
            array![lower[0], lower[1]].into_dyn(),
            array![upper[0], upper[1]].into_dyn(),
        )
        .expect("valid two-dimensional box")
    }

    fn affine_envelope(directions: Array2<f32>) -> Option<AyTailAffineReachabilityEnvelope> {
        AyTailAffineReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            box2([-1.0, -2.0], [3.0, 4.0]),
            directions,
            array![[1.0, 0.0], [0.0, 1.0]],
            Array1::from_vec(vec![-0.25, -0.5]),
            array![[1.0, 0.0], [0.0, 1.0]],
            Array1::from_vec(vec![0.25, 0.5]),
        )
    }

    fn box4(lower: [f32; 4], upper: [f32; 4]) -> BoundedTensor {
        BoundedTensor::new(
            Array1::from_vec(lower.to_vec()).into_dyn(),
            Array1::from_vec(upper.to_vec()).into_dyn(),
        )
        .expect("valid four-dimensional box")
    }

    fn box5(lower: [f32; 5], upper: [f32; 5]) -> BoundedTensor {
        BoundedTensor::new(
            Array1::from_vec(lower.to_vec()).into_dyn(),
            Array1::from_vec(upper.to_vec()).into_dyn(),
        )
        .expect("valid five-dimensional box")
    }

    fn shared_envelope(
        root: BoundedTensor,
        region: BoundedTensor,
    ) -> Option<AyTailSharedInputReachabilityEnvelope> {
        AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            root,
            region,
            vec![0, 1, 2, 3],
            Array2::eye(4),
            array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
            Array1::from_vec(vec![-0.25, -0.5, -0.75, -1.0]),
            array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
            Array1::from_vec(vec![0.25, 0.5, 0.75, 1.0]),
        )
    }

    fn box1(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new(array![lower].into_dyn(), array![upper].into_dyn())
            .expect("valid one-dimensional box")
    }

    fn selector_regions() -> Vec<BoundedTensor> {
        (0..AY_TAIL_REGION_SELECTOR_REGIONS)
            .map(|region_idx| {
                box1(
                    region_idx as f32 / AY_TAIL_REGION_SELECTOR_REGIONS as f32,
                    (region_idx + 1) as f32 / AY_TAIL_REGION_SELECTOR_REGIONS as f32,
                )
            })
            .collect()
    }

    fn selector_directions() -> Array2<f32> {
        Array2::from_shape_fn(
            (AY_TAIL_REGION_SELECTOR_REGIONS, 2),
            |(region_idx, seam_idx)| {
                if seam_idx == 0 {
                    1.0 + region_idx as f32 / 16.0
                } else if region_idx % 2 == 0 {
                    -0.5
                } else {
                    0.75
                }
            },
        )
    }

    fn selector_root_anchors() -> Vec<AyTailRootAnchor> {
        vec![
            AyTailRootAnchor::from_certified_root_box(
                "pre_2".to_string(),
                box2([-0.75, -1.0], [0.5, 1.25]),
            )
            .expect("valid second root anchor"),
            AyTailRootAnchor::from_certified_root_box(
                "pre_1".to_string(),
                box2([-0.5, -0.25], [0.75, 1.0]),
            )
            .expect("valid first root anchor"),
        ]
    }

    fn selector_regional_relu_bounds() -> Vec<AyTailRegionReluBounds> {
        vec![
            AyTailRegionReluBounds::from_certified_region_box(
                1,
                "pre_2".to_string(),
                box2([-0.5, -0.75], [0.25, 1.0]),
            )
            .expect("valid region-one second ReLU box"),
            AyTailRegionReluBounds::from_certified_region_box(
                0,
                "pre_2".to_string(),
                box2([-0.625, -0.875], [0.375, 1.125]),
            )
            .expect("valid region-zero second ReLU box"),
            AyTailRegionReluBounds::from_certified_region_box(
                1,
                "pre_1".to_string(),
                box2([-0.25, -0.125], [0.625, 0.875]),
            )
            .expect("valid region-one first ReLU box"),
            AyTailRegionReluBounds::from_certified_region_box(
                0,
                "pre_1".to_string(),
                box2([-0.375, -0.25], [0.5, 0.75]),
            )
            .expect("valid region-zero first ReLU box"),
        ]
    }

    fn selector_envelope() -> AyTailRegionSelectorEnvelope {
        AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
            "seam".to_string(),
            box1(0.0, 1.0),
            selector_regions(),
            box2([-2.0, -3.0], [4.0, 5.0]),
            selector_root_anchors(),
            selector_regional_relu_bounds(),
            selector_directions(),
            Array1::from_iter(
                (0..AY_TAIL_REGION_SELECTOR_REGIONS)
                    .map(|region_idx| 4.0 + region_idx as f32 / 8.0),
            ),
        )
        .expect("valid 16-region selector envelope")
    }

    fn selector_k4_regions(root: &BoundedTensor) -> Vec<BoundedTensor> {
        let flat = root.flatten();
        let root_lower: [f32; 5] = flat
            .lower()
            .as_slice()
            .expect("contiguous")
            .try_into()
            .expect("five lowers");
        let root_upper: [f32; 5] = flat
            .upper()
            .as_slice()
            .expect("contiguous")
            .try_into()
            .expect("five uppers");
        let boundaries = [-1.0, 3.0, -2.0, 17.0];
        (0..AY_TAIL_REGION_SELECTOR_REGIONS)
            .map(|region_index| {
                let mut lower = root_lower;
                let mut upper = root_upper;
                for (selector_bit, &dim) in AY_TAIL_REGION_SELECTOR_K4_SPLIT_DIMS.iter().enumerate()
                {
                    if ((region_index >> selector_bit) & 1) == 0 {
                        upper[dim] = boundaries[selector_bit];
                    } else {
                        lower[dim] = boundaries[selector_bit];
                    }
                }
                box5(lower, upper)
            })
            .collect()
    }

    fn selector_input_bank(
        root: &BoundedTensor,
        support_indices: Vec<usize>,
    ) -> AyTailSharedInputReachabilityEnvelope {
        let supports = support_indices.len();
        AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            root.clone(),
            root.clone(),
            support_indices,
            Array2::from_shape_fn((supports, 4), |(row, col)| u8::from(row == col) as f32),
            Array2::from_shape_fn(
                (supports, AY_TAIL_REGION_SELECTOR_K4_INPUTS),
                |(row, col)| (1 + row * AY_TAIL_REGION_SELECTOR_K4_INPUTS + col) as f32,
            ),
            Array1::from_iter((1..=supports).map(|index| -(index as f32) / 4.0)),
            Array2::from_shape_fn(
                (supports, AY_TAIL_REGION_SELECTOR_K4_INPUTS),
                |(row, col)| -((1 + row * AY_TAIL_REGION_SELECTOR_K4_INPUTS + col) as f32),
            ),
            Array1::from_iter((1..=supports).map(|index| index as f32 / 4.0)),
        )
        .expect("valid five-input selector bank")
    }

    fn selector_k2_bank(
        root: &BoundedTensor,
        support_indices: Vec<usize>,
    ) -> AyTailSharedInputReachabilityEnvelope {
        assert_eq!(support_indices.len(), AY_TAIL_REGION_SELECTOR_K2_SUPPORTS);
        selector_input_bank(root, support_indices)
    }

    fn selector_k4_bank(
        root: &BoundedTensor,
        support_indices: Vec<usize>,
    ) -> AyTailSharedInputReachabilityEnvelope {
        assert_eq!(support_indices.len(), AY_TAIL_REGION_SELECTOR_K4_SUPPORTS);
        selector_input_bank(root, support_indices)
    }

    fn selector_k2_envelope() -> AyTailRegionSelectorEnvelope {
        let root = box5([-4.0, 2.0, -9.0, 11.0, 0.0], [8.0, 10.0, -1.0, 11.0, 20.0]);
        let regions = selector_k4_regions(&root);
        let bank = selector_k2_bank(&root, vec![0, 1]);
        AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers_with_selector_k2_lift(
            "seam".to_string(),
            root,
            regions,
            box4([-2.0, -3.0, -4.0, -5.0], [4.0, 5.0, 6.0, 7.0]),
            selector_root_anchors(),
            selector_regional_relu_bounds(),
            bank,
            Array2::from_shape_fn((AY_TAIL_REGION_SELECTOR_REGIONS, 4), |(region, col)| {
                0.25 + (region * 4 + col) as f32 / 32.0
            }),
            Array1::from_iter(
                (0..AY_TAIL_REGION_SELECTOR_REGIONS)
                    .map(|region_idx| 4.0 + region_idx as f32 / 8.0),
            ),
        )
        .expect("valid schema-v5 selector K2 envelope")
    }

    fn selector_k4_envelope() -> AyTailRegionSelectorEnvelope {
        let root = box5([-4.0, 2.0, -9.0, 11.0, 0.0], [8.0, 10.0, -1.0, 11.0, 20.0]);
        let regions = selector_k4_regions(&root);
        let bank = selector_k4_bank(&root, vec![0, 1, 2, 3]);
        AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers_with_selector_k4_lift(
            "seam".to_string(),
            root,
            regions,
            box4([-2.0, -3.0, -4.0, -5.0], [4.0, 5.0, 6.0, 7.0]),
            selector_root_anchors(),
            selector_regional_relu_bounds(),
            bank,
            Array2::from_shape_fn((AY_TAIL_REGION_SELECTOR_REGIONS, 4), |(region, col)| {
                0.25 + (region * 4 + col) as f32 / 32.0
            }),
            Array1::from_iter(
                (0..AY_TAIL_REGION_SELECTOR_REGIONS)
                    .map(|region_idx| 4.0 + region_idx as f32 / 8.0),
            ),
        )
        .expect("valid schema-v4 selector K4 envelope")
    }

    #[test]
    fn ay_tail_certificate_gate_is_exact_and_default_off() {
        assert!(!ay_tail_certificate_enabled_from_value(None));
        assert!(ay_tail_certificate_enabled_from_value(Some("1")));
        for malformed in ["", "0", "true", "yes", " 1", "1 "] {
            assert!(!ay_tail_certificate_enabled_from_value(Some(malformed)));
        }
    }

    #[test]
    fn decision_certificate_is_bit_bound_to_the_requested_threshold() {
        let certificate = AyTailCertificate {
            q: -0.0,
            ay_tree_leaves: 2,
            ny_cert_farkas_replays: 2,
        };
        assert!(ay_tail_certificate_matches_request(certificate, Some(-0.0)));
        assert!(
            !ay_tail_certificate_matches_request(certificate, Some(0.0)),
            "signed-zero thresholds have distinct exact decision-row identities"
        );
        assert!(
            !ay_tail_certificate_matches_request(certificate, Some(-1.0)),
            "even a numerically stronger certificate must match the exact request"
        );
        assert!(!ay_tail_certificate_matches_request(
            AyTailCertificate {
                q: f32::NAN,
                ay_tree_leaves: 0,
                ny_cert_farkas_replays: 1,
            },
            None,
        ));
    }

    #[test]
    fn reachability_certificate_is_a_distinct_bit_bound_token() {
        let certificate = AyTailReachabilityCertificate {
            lower: -0.0,
            prefix_lower: 0.25,
            ay_tree_leaves: 3,
            ny_cert_farkas_replays: 3,
        };
        assert!(ay_tail_reachability_certificate_matches_request(
            certificate,
            0.25,
            -0.0
        ));
        assert!(!ay_tail_reachability_certificate_matches_request(
            certificate,
            0.25,
            0.0
        ));
        assert!(!ay_tail_reachability_certificate_matches_request(
            certificate,
            ny_tensor::next_up_f32(0.25),
            -0.0,
        ));
        assert!(!ay_tail_reachability_certificate_matches_request(
            AyTailReachabilityCertificate {
                lower: -0.0,
                prefix_lower: 0.25,
                ay_tree_leaves: 2,
                ny_cert_farkas_replays: 1,
            },
            0.25,
            -0.0,
        ));
    }

    #[test]
    fn affine_envelope_requires_two_genuinely_independent_supports() {
        assert!(affine_envelope(array![[1.0, 0.0], [0.0, 1.0]]).is_some());
        for rejected in [
            array![[1.0, 0.0], [2.0, 0.0]],
            array![[1.0, 0.0], [-1.0, 0.0]],
            array![[1.0, 0.0], [0.0, 0.0]],
        ] {
            assert!(
                affine_envelope(rejected).is_none(),
                "zero or (anti-)collinear K=2 banks must fail closed"
            );
        }
    }

    #[test]
    fn affine_certificate_bit_binds_complete_request_and_deadline() {
        let tail = GraphNetwork::new();
        let other_tail = GraphNetwork::new();
        assert!(!std::ptr::eq(&raw const tail, &raw const other_tail));
        let seam = box2([-10.0, -20.0], [10.0, 20.0]);
        let objective = [-0.0_f32, 2.0];
        let envelope = affine_envelope(array![[1.0, 0.0], [0.0, 1.0]]).expect("rank-two envelope");
        let requested_lower = -0.0_f32;
        let deadline = Instant::now() + Duration::from_secs(10);
        let certificate = AyTailAffineReachabilityCertificate {
            lower: requested_lower,
            tail_identity: std::ptr::from_ref(&tail) as usize,
            request_bits: affine_reachability_request_bits(
                &seam,
                &objective,
                &envelope,
                requested_lower,
            ),
            deadline,
            ay_tree_leaves: 2,
            ny_cert_farkas_replays: 2,
        };

        assert!(ay_tail_affine_reachability_certificate_matches_request(
            &certificate,
            &tail,
            &seam,
            &objective,
            &envelope,
            requested_lower,
            deadline,
        ));
        assert!(
            !ay_tail_affine_reachability_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &[0.0, 2.0],
                &envelope,
                requested_lower,
                deadline,
            ),
            "signed-zero objective bits are part of proof identity"
        );
        let changed_seam = box2([-10.0, -20.0], [10.0, 21.0]);
        assert!(!ay_tail_affine_reachability_certificate_matches_request(
            &certificate,
            &tail,
            &changed_seam,
            &objective,
            &envelope,
            requested_lower,
            deadline,
        ));
        let mut changed_envelope = envelope.clone();
        changed_envelope.lower_b[0] = ny_tensor::next_up_f32(changed_envelope.lower_b[0]);
        assert!(
            !ay_tail_affine_reachability_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                &changed_envelope,
                requested_lower,
                deadline,
            ),
            "every transported affine coefficient and bias is request-bound"
        );
        assert!(!ay_tail_affine_reachability_certificate_matches_request(
            &certificate,
            &tail,
            &seam,
            &objective,
            &envelope,
            requested_lower,
            deadline + Duration::from_nanos(1),
        ));
        assert!(!ay_tail_affine_reachability_certificate_matches_request(
            &certificate,
            &tail,
            &seam,
            &objective,
            &envelope,
            0.0,
            deadline,
        ));
        assert!(
            !ay_tail_affine_reachability_certificate_matches_request(
                &certificate,
                &other_tail,
                &seam,
                &objective,
                &envelope,
                requested_lower,
                deadline,
            ),
            "a token cannot cross to a different run-local tail object"
        );
    }

    #[test]
    fn shared_envelope_requires_closed_full_rank_bank_and_root_subset() {
        let root = box2([-1.0, -2.0], [3.0, 4.0]);
        let region = box2([0.0, -1.0], [2.0, 3.0]);
        let envelope = shared_envelope(root.clone(), region.clone()).expect("valid K4 shared bank");
        assert_eq!(envelope.directions().shape(), &[4, 4]);
        assert_eq!(envelope.support_indices(), &[0, 1, 2, 3]);
        assert!(envelope.bank_bytes() <= AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES);

        assert!(
            shared_envelope(root.clone(), box2([-2.0, -1.0], [2.0, 3.0])).is_none(),
            "a regional latent box may not escape the certified root box"
        );
        let duplicate_support = AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            root.clone(),
            region.clone(),
            vec![0, 1, 1, 3],
            Array2::eye(4),
            Array2::zeros((4, 2)),
            Array1::zeros(4),
            Array2::zeros((4, 2)),
            Array1::zeros(4),
        );
        assert!(duplicate_support.is_none());
        let rank_three = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]
        ];
        assert!(AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            root,
            region,
            vec![0, 1, 2, 3],
            rank_three,
            Array2::zeros((4, 2)),
            Array1::zeros(4),
            Array2::zeros((4, 2)),
            Array1::zeros(4),
        )
        .is_none());
    }

    #[test]
    fn shared_envelope_enforces_public_payload_cap_before_authority() {
        let seam_dim = AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES / size_of::<f32>();
        let mut directions = Array2::zeros((4, seam_dim));
        for row in 0..4 {
            directions[[row, row]] = 1.0;
        }
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        assert!(AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            root.clone(),
            root,
            vec![0, 1, 2, 3],
            directions,
            Array2::zeros((4, 2)),
            Array1::zeros(4),
            Array2::zeros((4, 2)),
            Array1::zeros(4),
        )
        .is_none());
    }

    #[test]
    fn compact_k16_constructor_is_opaque_and_exact_shape_bounded() {
        let root = BoundedTensor::new(
            Array1::from_elem(AY_TAIL_COMPACT_K16_INPUTS, -1.0).into_dyn(),
            Array1::from_elem(AY_TAIL_COMPACT_K16_INPUTS, 1.0).into_dyn(),
        )
        .expect("finite compact root box");
        let directions = Array2::from_shape_fn(
            (
                AY_TAIL_COMPACT_K16_SUPPORTS,
                AY_TAIL_COMPACT_K16_SEAM_ELEMENTS,
            ),
            |(row, col)| f32::from(u8::from(row == col)),
        );
        let a = Array2::zeros((AY_TAIL_COMPACT_K16_SUPPORTS, AY_TAIL_COMPACT_K16_INPUTS));
        let b = Array1::zeros(AY_TAIL_COMPACT_K16_SUPPORTS);
        let envelope = AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            AY_TAIL_COMPACT_K16_SEAM_NODE.to_owned(),
            root.clone(),
            root.clone(),
            (0..AY_TAIL_COMPACT_K16_SUPPORTS).collect(),
            directions.clone(),
            a.clone(),
            b.clone(),
            a.clone(),
            b.clone(),
        )
        .expect("the sealed live-producer compact K16 shape fits its private cap");
        assert_eq!(envelope.bank_bytes(), 399_872);
        assert!(envelope.bank_bytes() > AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES);
        assert!(envelope.bank_bytes() <= AY_TAIL_COMPACT_K16_MAX_BANK_BYTES);

        assert!(
            AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
                "unbound_diagnostic_seam".to_owned(),
                root.clone(),
                root,
                (0..AY_TAIL_COMPACT_K16_SUPPORTS).collect(),
                directions,
                a.clone(),
                b.clone(),
                a,
                b,
            )
            .is_none(),
            "the enlarged payload cap is unavailable without the exact live seam identity"
        );
    }

    #[test]
    fn shared_certificate_bit_binds_root_region_supports_tail_and_deadline() {
        let tail = GraphNetwork::new();
        let other_tail = GraphNetwork::new();
        let seam = box4([-10.0, -20.0, -30.0, -40.0], [10.0, 20.0, 30.0, 40.0]);
        let objective = [-0.0_f32, 2.0, 0.0, -1.0];
        let root = box2([-1.0, -2.0], [3.0, 4.0]);
        let region = box2([0.0, -1.0], [2.0, 3.0]);
        let envelope = shared_envelope(root.clone(), region).expect("valid K4 shared bank");
        let requested_lower = -0.0_f32;
        let deadline = Instant::now() + Duration::from_secs(10);
        let certificate = AyTailSharedInputReachabilityCertificate {
            lower: requested_lower,
            tail_identity: std::ptr::from_ref(&tail) as usize,
            request_bits: shared_input_reachability_request_bits(
                &seam,
                &objective,
                &envelope,
                requested_lower,
            ),
            deadline,
            ay_tree_leaves: 2,
            ny_cert_farkas_replays: 2,
        };
        assert!(
            ay_tail_shared_input_reachability_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                &envelope,
                requested_lower,
                deadline,
            )
        );

        let changed_region =
            shared_envelope(root, box2([0.0, -1.0], [2.0, 2.0])).expect("valid narrower region");
        assert!(
            !ay_tail_shared_input_reachability_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                &changed_region,
                requested_lower,
                deadline,
            ),
            "the exact regional latent bounds are request identity"
        );
        let changed_root = shared_envelope(
            box2([-2.0, -2.0], [3.0, 4.0]),
            box2([0.0, -1.0], [2.0, 3.0]),
        )
        .expect("valid bank on a different root");
        assert!(
            !ay_tail_shared_input_reachability_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                &changed_root,
                requested_lower,
                deadline,
            ),
            "the exact certified root box is request identity"
        );
        let mut changed_support = envelope.clone();
        Arc::make_mut(&mut changed_support.bank).support_indices[0] = 9;
        assert!(
            !ay_tail_shared_input_reachability_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                &changed_support,
                requested_lower,
                deadline,
            ),
            "support identities are request-bound independently of coefficients"
        );
        assert!(
            !ay_tail_shared_input_reachability_certificate_matches_request(
                &certificate,
                &other_tail,
                &seam,
                &objective,
                &envelope,
                requested_lower,
                deadline,
            )
        );
        assert!(
            !ay_tail_shared_input_reachability_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                &envelope,
                requested_lower,
                deadline + Duration::from_nanos(1),
            )
        );
    }

    #[test]
    fn selector_envelope_rounds_outward_and_makes_every_inactive_row_redundant() {
        let envelope = selector_envelope();
        assert_eq!(
            envelope.directions().shape(),
            &[AY_TAIL_REGION_SELECTOR_REGIONS, 2]
        );
        assert_eq!(
            envelope.selector_coefficients().shape(),
            &[
                AY_TAIL_REGION_SELECTOR_REGIONS,
                AY_TAIL_REGION_SELECTOR_BITS
            ]
        );

        let seam = envelope.root_seam_box().flatten();
        let seam_lower: Vec<f32> = seam.lower().iter().copied().collect();
        let seam_upper: Vec<f32> = seam.upper().iter().copied().collect();
        for region_idx in 0..AY_TAIL_REGION_SELECTOR_REGIONS {
            let direction = envelope.directions().row(region_idx);
            let mut exact_corner_lower = f64::INFINITY;
            for corner in 0..4 {
                let value = f64::from(direction[0])
                    * f64::from(if corner & 1 == 0 {
                        seam_lower[0]
                    } else {
                        seam_upper[0]
                    })
                    + f64::from(direction[1])
                        * f64::from(if corner & 2 == 0 {
                            seam_lower[1]
                        } else {
                            seam_upper[1]
                        });
                exact_corner_lower = exact_corner_lower.min(value);
            }
            let box_lower = envelope.global_seam_lowers()[region_idx];
            let floor = f64::from(envelope.prefix_floors()[region_idx]);
            let big_m = envelope.big_m()[region_idx];
            assert!(
                box_lower <= exact_corner_lower,
                "B_{region_idx} must be directed down"
            );
            assert!(
                big_m >= (floor - box_lower).max(0.0),
                "M_{region_idx} must be directed up"
            );

            let desired_one_count = (0..AY_TAIL_REGION_SELECTOR_BITS)
                .filter(|&bit_idx| ((region_idx >> bit_idx) & 1) != 0)
                .count();
            assert!(
                envelope.row_lowers()[region_idx] <= floor - desired_one_count as f64 * big_m,
                "row lower must be rounded toward weaker feasibility"
            );

            for assignment in 0..AY_TAIL_REGION_SELECTOR_REGIONS {
                let selector_sum = (0..AY_TAIL_REGION_SELECTOR_BITS)
                    .filter(|&bit_idx| ((assignment >> bit_idx) & 1) != 0)
                    .map(|bit_idx| envelope.selector_coefficients()[[region_idx, bit_idx]])
                    .sum::<f64>();
                let seam_premise = if assignment == region_idx {
                    floor
                } else {
                    box_lower
                };
                assert!(
                    seam_premise + selector_sum >= envelope.row_lowers()[region_idx],
                    "row {region_idx} must be valid when active and redundant for assignment \
                     {assignment} when inactive"
                );
            }
        }
    }

    #[test]
    fn selector_rows_use_little_endian_canonical_region_bits() {
        let envelope = selector_envelope();
        for region_idx in 0..AY_TAIL_REGION_SELECTOR_REGIONS {
            let big_m = envelope.big_m()[region_idx];
            assert!(big_m > 0.0);
            for bit_idx in 0..AY_TAIL_REGION_SELECTOR_BITS {
                let expected = if ((region_idx >> bit_idx) & 1) == 0 {
                    big_m
                } else {
                    -big_m
                };
                assert_eq!(
                    envelope.selector_coefficients()[[region_idx, bit_idx]].to_bits(),
                    expected.to_bits(),
                    "selector bit 0 is the least-significant region-index bit"
                );
            }
        }
    }

    #[test]
    fn selector_envelope_rejects_bad_counts_shapes_boxes_and_nonfinite_values() {
        let root = box1(0.0, 1.0);
        let seam = box2([-2.0, -3.0], [4.0, 5.0]);
        let directions = selector_directions();
        let floors = Array1::from_elem(AY_TAIL_REGION_SELECTOR_REGIONS, 1.0);

        let mut too_few_regions = selector_regions();
        too_few_regions.pop();
        assert!(
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                "seam".to_string(),
                root.clone(),
                too_few_regions,
                seam.clone(),
                selector_root_anchors(),
                selector_regional_relu_bounds(),
                directions.clone(),
                floors.clone(),
            )
            .is_none()
        );

        let mut escaping_regions = selector_regions();
        escaping_regions[15] = box1(15.0 / 16.0, 2.0);
        assert!(
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                "seam".to_string(),
                root.clone(),
                escaping_regions,
                seam.clone(),
                selector_root_anchors(),
                selector_regional_relu_bounds(),
                directions.clone(),
                floors.clone(),
            )
            .is_none()
        );

        let mut nonfinite_directions = directions.clone();
        nonfinite_directions[[9, 1]] = f32::NAN;
        assert!(
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                "seam".to_string(),
                root.clone(),
                selector_regions(),
                seam.clone(),
                selector_root_anchors(),
                selector_regional_relu_bounds(),
                nonfinite_directions,
                floors.clone(),
            )
            .is_none()
        );

        let mut nonfinite_floors = floors.clone();
        nonfinite_floors[15] = f32::INFINITY;
        assert!(
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                "seam".to_string(),
                root.clone(),
                selector_regions(),
                seam.clone(),
                selector_root_anchors(),
                selector_regional_relu_bounds(),
                directions,
                nonfinite_floors,
            )
            .is_none()
        );
        assert!(
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                String::new(),
                root,
                selector_regions(),
                seam,
                selector_root_anchors(),
                selector_regional_relu_bounds(),
                Array2::zeros((AY_TAIL_REGION_SELECTOR_REGIONS, 1)),
                floors,
            )
            .is_none()
        );
    }

    #[test]
    fn selector_envelope_rejects_missing_duplicate_and_over_cap_root_anchors() {
        let root = box1(0.0, 1.0);
        let seam = box2([-2.0, -3.0], [4.0, 5.0]);
        let directions = selector_directions();
        let floors = Array1::from_elem(AY_TAIL_REGION_SELECTOR_REGIONS, 1.0);
        let build = |anchors| {
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                "seam".to_string(),
                root.clone(),
                selector_regions(),
                seam.clone(),
                anchors,
                selector_regional_relu_bounds(),
                directions.clone(),
                floors.clone(),
            )
        };

        assert!(build(Vec::new()).is_none());
        let duplicate =
            AyTailRootAnchor::from_certified_root_box("pre".to_string(), box1(-1.0, 1.0))
                .expect("valid individual anchor");
        assert!(build(vec![duplicate.clone(), duplicate]).is_none());
        assert!(
            AyTailRootAnchor::from_certified_root_box(
                "pre_inf".to_string(),
                BoundedTensor::new_conservative(&[1]),
            )
            .is_none(),
            "infinite root anchors are not exact-lane payload"
        );
        assert!(
            AyTailRootAnchor::from_certified_root_box(
                "pre_wide".to_string(),
                BoundedTensor::new(
                    Array1::zeros(super::AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_ELEMENTS + 1,)
                        .into_dyn(),
                    Array1::ones(super::AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHOR_ELEMENTS + 1,)
                        .into_dyn(),
                )
                .expect("finite over-cap box"),
            )
            .is_none(),
            "one anchor cannot bypass the aggregate element cap"
        );
        let over_cap: Vec<_> = (0..=AY_TAIL_REGION_SELECTOR_MAX_ROOT_ANCHORS)
            .map(|index| {
                AyTailRootAnchor::from_certified_root_box(format!("pre_{index}"), box1(-1.0, 1.0))
                    .expect("valid individual anchor")
            })
            .collect();
        assert!(build(over_cap).is_none());
    }

    #[test]
    fn selector_regional_relu_payload_requires_exact_target_source_cover_and_caps() {
        let envelope = selector_envelope();
        let identities: Vec<_> = envelope
            .regional_relu_bounds()
            .iter()
            .map(|record| (record.region_index(), record.node_name()))
            .collect();
        assert_eq!(
            identities,
            vec![(0, "pre_1"), (0, "pre_2"), (1, "pre_1"), (1, "pre_2")]
        );
        for record in envelope.regional_relu_bounds() {
            let root = envelope
                .root_tail_anchors()
                .iter()
                .find(|anchor| anchor.node_name() == record.node_name())
                .expect("each regional record has one root anchor");
            assert!(finite_box_is_inside(root.bounds(), record.bounds()));
        }

        let build = |regional_relu_bounds| {
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                "seam".to_string(),
                box1(0.0, 1.0),
                selector_regions(),
                box2([-2.0, -3.0], [4.0, 5.0]),
                selector_root_anchors(),
                regional_relu_bounds,
                selector_directions(),
                Array1::ones(AY_TAIL_REGION_SELECTOR_REGIONS),
            )
        };
        let mut missing = selector_regional_relu_bounds();
        missing.pop();
        assert!(build(missing).is_none());

        let mut duplicate = selector_regional_relu_bounds();
        duplicate[0].region_index = 0;
        assert!(build(duplicate).is_none());

        let mut escaping = selector_regional_relu_bounds();
        escaping[0].bounds = box2([-1.0, -0.25], [0.5, 0.75]);
        assert!(build(escaping).is_none());
        assert!(
            AyTailRegionReluBounds::from_certified_region_box(
                2,
                "pre_1".to_string(),
                box1(-0.5, 0.5),
            )
            .is_none(),
            "only the closed target-region set may carry authority"
        );

        let elements = AY_TAIL_REGION_SELECTOR_MAX_RELU_BOUND_ELEMENTS / 2 + 1;
        let root_box = BoundedTensor::new(
            Array1::from_elem(elements, -1.0).into_dyn(),
            Array1::from_elem(elements, 1.0).into_dyn(),
        )
        .expect("finite root box below the root-anchor aggregate cap");
        let root_anchor =
            AyTailRootAnchor::from_certified_root_box("pre".to_string(), root_box.clone())
                .expect("individual root anchor remains below its cap");
        let regional: Vec<_> = [0, 1]
            .into_iter()
            .map(|region_index| {
                AyTailRegionReluBounds::from_certified_region_box(
                    region_index,
                    "pre".to_string(),
                    root_box.clone(),
                )
                .expect("individual regional record remains below its cap")
            })
            .collect();
        assert!(
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers(
                "seam".to_string(),
                box1(0.0, 1.0),
                selector_regions(),
                box2([-2.0, -3.0], [4.0, 5.0]),
                vec![root_anchor],
                regional,
                selector_directions(),
                Array1::ones(AY_TAIL_REGION_SELECTOR_REGIONS),
            )
            .is_none(),
            "aggregate regional endpoints cannot exceed the immutable element cap"
        );
    }

    #[test]
    fn selector_k4_constructor_rejects_out_of_range_support_identity() {
        let root = box5([-4.0, 2.0, -9.0, 11.0, 0.0], [8.0, 10.0, -1.0, 11.0, 20.0]);
        let regions = selector_k4_regions(&root);
        let bank = selector_k4_bank(&root, vec![0, 1, 2, AY_TAIL_REGION_SELECTOR_REGIONS]);
        assert!(
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers_with_selector_k4_lift(
                "seam".to_string(),
                root,
                regions,
                box4([-2.0, -3.0, -4.0, -5.0], [4.0, 5.0, 6.0, 7.0]),
                selector_root_anchors(),
                selector_regional_relu_bounds(),
                bank,
                Array2::ones((AY_TAIL_REGION_SELECTOR_REGIONS, 4)),
                Array1::ones(AY_TAIL_REGION_SELECTOR_REGIONS),
            )
            .is_none(),
            "K4 support identities are canonical region indices, not arbitrary tags"
        );
    }

    #[test]
    fn selector_k2_constructor_requires_exact_rank_two_canonical_bank() {
        let root = box5([-4.0, 2.0, -9.0, 11.0, 0.0], [8.0, 10.0, -1.0, 11.0, 20.0]);
        let build = |bank| {
            AyTailRegionSelectorEnvelope::from_certified_prefix_frontiers_with_selector_k2_lift(
                "seam".to_string(),
                root.clone(),
                selector_k4_regions(&root),
                box4([-2.0, -3.0, -4.0, -5.0], [4.0, 5.0, 6.0, 7.0]),
                selector_root_anchors(),
                selector_regional_relu_bounds(),
                bank,
                Array2::ones((AY_TAIL_REGION_SELECTOR_REGIONS, 4)),
                Array1::ones(AY_TAIL_REGION_SELECTOR_REGIONS),
            )
        };

        assert!(
            build(selector_k4_bank(&root, vec![0, 1, 2, 3])).is_none(),
            "schema-v5 authority requires exactly two supports"
        );

        let mut duplicate = selector_k2_bank(&root, vec![0, 1]);
        Arc::make_mut(&mut duplicate.bank).support_indices[1] = 0;
        assert!(
            build(duplicate).is_none(),
            "K2 support identities must remain unique"
        );

        let mut out_of_range = selector_k2_bank(&root, vec![0, 1]);
        Arc::make_mut(&mut out_of_range.bank).support_indices[1] = AY_TAIL_REGION_SELECTOR_REGIONS;
        assert!(
            build(out_of_range).is_none(),
            "K2 support identities are canonical region indices"
        );

        assert!(
            AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
                "seam".to_string(),
                root.clone(),
                root,
                vec![0, 1],
                array![[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
                Array2::zeros((2, AY_TAIL_REGION_SELECTOR_K4_INPUTS)),
                Array1::zeros(2),
                Array2::zeros((2, AY_TAIL_REGION_SELECTOR_K4_INPUTS)),
                Array1::zeros(2),
            )
            .is_none(),
            "the public payload constructor rejects a rank-one K2 claim"
        );
    }

    #[test]
    fn selector_k2_production_shape_has_exact_payload_bytes() {
        let root = box5([-4.0, 2.0, -9.0, 11.0, 0.0], [8.0, 10.0, -1.0, 11.0, 20.0]);
        let directions =
            Array2::from_shape_fn((2, 2_048), |(row, col)| f32::from(u8::from(row == col)));
        let bank = AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            root.clone(),
            root,
            vec![0, 1],
            directions,
            Array2::ones((2, AY_TAIL_REGION_SELECTOR_K4_INPUTS)),
            Array1::zeros(2),
            Array2::ones((2, AY_TAIL_REGION_SELECTOR_K4_INPUTS)),
            Array1::zeros(2),
        )
        .expect("rank-two production-shape K2 bank");
        assert_eq!(
            bank.bank_bytes(),
            16_496,
            "4,120 binary32 scalars plus two usize support identities"
        );
        assert!(bank.bank_bytes() <= AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES);
    }

    #[test]
    fn selector_k4_schema_v4_binds_every_bank_component_while_legacy_stays_v3() {
        let legacy = selector_envelope();
        assert!(legacy.selector_k2_lift().is_none());
        assert!(legacy.selector_k4_lift().is_none());
        assert_eq!(
            region_selector_request_bits(legacy.root_seam_box(), &[1.0, -1.0], &legacy, 0.25,)[1],
            3,
            "gate-off selector request identity remains schema v3"
        );

        let tail = GraphNetwork::new();
        let envelope = selector_k4_envelope();
        let seam = envelope.root_seam_box().clone();
        let objective = [1.0, -1.0, 0.5, -0.25];
        let requested_lower = 0.25;
        let deadline = Instant::now() + Duration::from_secs(10);
        let request_bits =
            region_selector_request_bits(&seam, &objective, &envelope, requested_lower);
        assert_eq!(request_bits[1], 4);
        let certificate = AyTailRegionSelectorCertificate {
            lower: requested_lower,
            tail_identity: std::ptr::from_ref(&tail) as usize,
            request_bits,
            deadline,
            ay_tree_leaves: 0,
            ny_cert_farkas_replays: 1,
        };
        let matches = |candidate: &AyTailRegionSelectorEnvelope| {
            ay_tail_region_selector_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                candidate,
                requested_lower,
                deadline,
            )
        };
        assert!(matches(&envelope));

        let mut support = envelope.clone();
        Arc::make_mut(&mut support.selector_k4_lift.as_mut().unwrap().bank).support_indices[0] = 15;
        assert!(!matches(&support), "support identity is schema-v4 identity");

        let mut direction = envelope.clone();
        Arc::make_mut(&mut direction.selector_k4_lift.as_mut().unwrap().bank).directions[[0, 0]] =
            1.25;
        assert!(!matches(&direction), "K4 direction P is schema-v4 identity");

        let mut coefficient = envelope.clone();
        Arc::make_mut(&mut coefficient.selector_k4_lift.as_mut().unwrap().bank).lower_a[[1, 4]] =
            -3.5;
        assert!(
            !matches(&coefficient),
            "input-linear A is schema-v4 identity"
        );

        let mut bias = envelope.clone();
        Arc::make_mut(&mut bias.selector_k4_lift.as_mut().unwrap().bank).upper_b[2] = 9.0;
        assert!(!matches(&bias), "input-linear b is schema-v4 identity");

        let mut root = envelope.clone();
        Arc::make_mut(&mut root.selector_k4_lift.as_mut().unwrap().bank).certified_root_input =
            box5([-4.0, 2.0, -9.0, 11.0, -1.0], [8.0, 10.0, -1.0, 11.0, 20.0]);
        assert!(!matches(&root), "certified root box is schema-v4 identity");

        let mut region = envelope;
        region.selector_k4_lift.as_mut().unwrap().region_input =
            box5([-4.0, 2.0, -9.0, 11.0, 0.0], [8.0, 10.0, -1.0, 11.0, 19.0]);
        assert!(!matches(&region), "latent region box is schema-v4 identity");
    }

    #[test]
    fn selector_k2_schema_v5_binds_bank_provenance_and_conflicts_fail_closed() {
        let tail = GraphNetwork::new();
        let envelope = selector_k2_envelope();
        assert!(envelope.selector_k2_lift().is_some());
        assert!(envelope.selector_k4_lift().is_none());
        let seam = envelope.root_seam_box().clone();
        let objective = [1.0, -1.0, 0.5, -0.25];
        let requested_lower = 0.25;
        let deadline = Instant::now() + Duration::from_secs(10);
        let request_bits =
            region_selector_request_bits(&seam, &objective, &envelope, requested_lower);
        assert_eq!(request_bits[1], 5);
        assert_ne!(
            request_bits,
            region_selector_request_bits(
                selector_k4_envelope().root_seam_box(),
                &objective,
                &selector_k4_envelope(),
                requested_lower,
            ),
            "K2 cannot reuse a schema-v4 K4 proof identity"
        );
        let certificate = AyTailRegionSelectorCertificate {
            lower: requested_lower,
            tail_identity: std::ptr::from_ref(&tail) as usize,
            request_bits,
            deadline,
            ay_tree_leaves: 0,
            ny_cert_farkas_replays: 1,
        };
        let matches = |candidate: &AyTailRegionSelectorEnvelope| {
            ay_tail_region_selector_certificate_matches_request(
                &certificate,
                &tail,
                &seam,
                &objective,
                candidate,
                requested_lower,
                deadline,
            )
        };
        assert!(matches(&envelope));

        let mut support = envelope.clone();
        Arc::make_mut(&mut support.selector_k2_lift.as_mut().unwrap().bank).support_indices[0] = 15;
        assert!(!matches(&support), "support identity is schema-v5 identity");

        let mut direction = envelope.clone();
        Arc::make_mut(&mut direction.selector_k2_lift.as_mut().unwrap().bank).directions[[0, 0]] =
            1.25;
        assert!(!matches(&direction), "K2 direction P is schema-v5 identity");

        let mut coefficient = envelope.clone();
        Arc::make_mut(&mut coefficient.selector_k2_lift.as_mut().unwrap().bank).lower_a[[1, 4]] =
            -3.5;
        assert!(
            !matches(&coefficient),
            "input-linear A is schema-v5 identity"
        );

        let mut bias = envelope.clone();
        Arc::make_mut(&mut bias.selector_k2_lift.as_mut().unwrap().bank).upper_b[1] = 9.0;
        assert!(!matches(&bias), "input-linear b is schema-v5 identity");

        let mut root = envelope.clone();
        Arc::make_mut(&mut root.selector_k2_lift.as_mut().unwrap().bank).certified_root_input =
            box5([-4.0, 2.0, -9.0, 11.0, -1.0], [8.0, 10.0, -1.0, 11.0, 20.0]);
        assert!(!matches(&root), "certified root box is schema-v5 identity");

        let mut region = envelope.clone();
        region.selector_k2_lift.as_mut().unwrap().region_input =
            box5([-4.0, 2.0, -9.0, 11.0, 0.0], [8.0, 10.0, -1.0, 11.0, 19.0]);
        assert!(!matches(&region), "latent region box is schema-v5 identity");

        let mut conflicting = envelope;
        conflicting.selector_k4_lift = selector_k4_envelope().selector_k4_lift;
        assert_eq!(
            region_selector_request_bits(&seam, &objective, &conflicting, requested_lower)[1],
            0,
            "an impossible dual-bank mutation receives no valid schema identity"
        );
        assert!(
            !matches(&conflicting),
            "dual K2/K4 authority fails closed at certificate validation"
        );
    }

    #[test]
    fn selector_certificate_bit_binds_rows_and_accepts_only_complete_proof_shapes() {
        let tail = GraphNetwork::new();
        let other_tail = GraphNetwork::new();
        let envelope = selector_envelope();
        let seam = envelope.root_seam_box().clone();
        let objective = [-0.0_f32, 2.0];
        let requested_lower = -0.0_f32;
        let deadline = Instant::now() + Duration::from_secs(10);
        assert_eq!(
            region_selector_request_bits(&seam, &objective, &envelope, requested_lower)[1],
            3,
            "regional ReLU transport is bound under selector request schema v3"
        );
        let root_certificate = AyTailRegionSelectorCertificate {
            lower: requested_lower,
            tail_identity: std::ptr::from_ref(&tail) as usize,
            request_bits: region_selector_request_bits(
                &seam,
                &objective,
                &envelope,
                requested_lower,
            ),
            deadline,
            ay_tree_leaves: 0,
            ny_cert_farkas_replays: 1,
        };
        assert!(ay_tail_region_selector_certificate_matches_request(
            &root_certificate,
            &tail,
            &seam,
            &objective,
            &envelope,
            requested_lower,
            deadline,
        ));

        let mut complete_tree = root_certificate.clone();
        complete_tree.ay_tree_leaves = AY_TAIL_REGION_SELECTOR_REGIONS;
        complete_tree.ny_cert_farkas_replays = AY_TAIL_REGION_SELECTOR_REGIONS;
        assert!(ay_tail_region_selector_certificate_matches_request(
            &complete_tree,
            &tail,
            &seam,
            &objective,
            &envelope,
            requested_lower,
            deadline,
        ));

        for (leaves, replays) in [(0, 16), (1, 1), (16, 1), (15, 15)] {
            let mut malformed = root_certificate.clone();
            malformed.ay_tree_leaves = leaves;
            malformed.ny_cert_farkas_replays = replays;
            assert!(
                !ay_tail_region_selector_certificate_matches_request(
                    &malformed,
                    &tail,
                    &seam,
                    &objective,
                    &envelope,
                    requested_lower,
                    deadline,
                ),
                "partial or over-replayed selector proofs must fail closed"
            );
        }

        let mut mutated_bits = root_certificate.clone();
        mutated_bits.request_bits[3] ^= 1;
        assert!(!ay_tail_region_selector_certificate_matches_request(
            &mutated_bits,
            &tail,
            &seam,
            &objective,
            &envelope,
            requested_lower,
            deadline,
        ));

        let mut changed_envelope = envelope.clone();
        changed_envelope.selector_coefficients[[7, 2]] =
            ny_core::dd::next_up_f64(changed_envelope.selector_coefficients[[7, 2]]);
        assert!(!ay_tail_region_selector_certificate_matches_request(
            &root_certificate,
            &tail,
            &seam,
            &objective,
            &changed_envelope,
            requested_lower,
            deadline,
        ));
        let mut changed_anchor = envelope.clone();
        changed_anchor.root_tail_anchors[0].bounds = box2([-0.5, -0.25], [0.75, 1.25]);
        assert!(
            !ay_tail_region_selector_certificate_matches_request(
                &root_certificate,
                &tail,
                &seam,
                &objective,
                &changed_anchor,
                requested_lower,
                deadline,
            ),
            "root-anchor endpoints are selector request identity"
        );
        let mut changed_anchor_name = envelope.clone();
        changed_anchor_name.root_tail_anchors[0].node_name.push('x');
        assert!(
            !ay_tail_region_selector_certificate_matches_request(
                &root_certificate,
                &tail,
                &seam,
                &objective,
                &changed_anchor_name,
                requested_lower,
                deadline,
            ),
            "root-anchor node identity is selector request identity"
        );
        let mut changed_regional = envelope.clone();
        changed_regional.regional_relu_bounds[0].bounds = box2([-0.375, -0.25], [0.5, 0.875]);
        assert!(
            !ay_tail_region_selector_certificate_matches_request(
                &root_certificate,
                &tail,
                &seam,
                &objective,
                &changed_regional,
                requested_lower,
                deadline,
            ),
            "regional ReLU endpoints are selector schema-v3 request identity"
        );
        assert!(!ay_tail_region_selector_certificate_matches_request(
            &root_certificate,
            &other_tail,
            &seam,
            &objective,
            &envelope,
            requested_lower,
            deadline,
        ));
        assert!(!ay_tail_region_selector_certificate_matches_request(
            &root_certificate,
            &tail,
            &seam,
            &objective,
            &envelope,
            0.0,
            deadline,
        ));
    }
}
