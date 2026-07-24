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
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::GraphNetwork;

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
/// These are authority constants rather than environment knobs. The current
/// evidence-backed producer emits only K4; K8/K16 remain dark encoder/type
/// capabilities for a later cost-qualified canary. Malformed or lower-rank
/// banks fail closed.
pub const AY_TAIL_SHARED_INPUT_SUPPORT_ROWS: [usize; 3] = [16, 8, 4];

/// Maximum immutable public payload for one shared root-input support bank.
pub const AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES: usize = 256 * 1024;

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
        if seam_node.is_empty()
            || !allowed_rows
            || !unique_supports
            || bank_bytes > AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES
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
        || envelope.bank_bytes() > AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES
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
        ay_tail_shared_input_reachability_certificate_matches_request,
        shared_input_reachability_request_bits, AyTailAffineReachabilityCertificate,
        AyTailAffineReachabilityEnvelope, AyTailCertificate, AyTailReachabilityCertificate,
        AyTailSharedInputReachabilityCertificate, AyTailSharedInputReachabilityEnvelope,
        AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES,
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
}
