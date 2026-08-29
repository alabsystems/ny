// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Branch-tree (case-split) composition of per-cell certificates.
//!
//! The flat [`crate::selfcheck`] verifier checks ONE Farkas / entailment
//! combination — a single convex region. A whole-box verdict for a function that
//! is only piecewise bounded (e.g. the TLL `max_j min_{i∈S_j} L_i` lattice) is a
//! MIN over a finite partition of the box: split the box into cells, bound `y`
//! below by `bound_C` on each cell `C`, and conclude `min_x y ≥ min_C bound_C`.
//!
//! That composition is exactly the machine-checked β-CROWN combination rule
//! [`branch_split_min`] (`Crownproof.Branch` in the exact pinned Clean dependency,
//! kernel re-typechecked per `KERNEL_IMPORT.md`): applied recursively over an
//! axis-aligned grid, the min of the per-leaf bounds is a sound global bound.
//! This module verifies the SHAPE of such a proof tree exactly (partition is an
//! exact cover; every leaf's per-cell facts are valid entailments over that
//! cell; each leaf binds to its declared bound) and returns the composed global
//! bound. Each leaf's entailments remain individually Clean-kernel-checkable
//! (`verify_entailment_certificate`); the ONLY new trust element is the
//! composition rule, and it is `branch_split_min` — no new axiom.
//!
//! # Soundness
//!
//! For a leaf cell `C` with declared bound `m_C`, every supplied member
//! entailment proves `L_i(x) ≥ m_C` over `C` (checked: a valid non-negative
//! combination of `C`'s own box faces, whose conclusion `a_i·x ≥ m_C − b_i`
//! rearranges to `L_i(x) ≥ m_C`). Given the producer's decode `y(x) =
//! max_j min_{i∈S_j} L_i(x)` and that the entailments enumerate one selector
//! group `S_{j*}`, `y(x) ≥ min_{i∈S_{j*}} L_i(x) ≥ m_C` over `C`. Since the
//! leaves EXACTLY partition the box (each axis' edges strictly increase from
//! `lo` to `hi`; the leaf set equals the full product grid), `branch_split_min`
//! iterated gives `min_x y ≥ min_C m_C`. If that exceeds the `≤`-threshold, the
//! property `Y_0 ≤ threshold` is UNSAT. Everything is exact rational; no float
//! sits between the accepted cert and the verdict.
//!
//! The decode binding (`(a_i, b_i)` are the network's affine pieces; the members
//! are a real selector group; `y` is `max-min`) is the SAME trust boundary the
//! per-cell slice already relies on — established by NY's runtime forward
//! self-check, not by this linear checker.

use crate::json::{Json, Map};
use crate::rational::{ensure_healthy, poisoned, Rat, RatError};
use crate::schema::{entailment_to_json, ConstraintKind, EntailmentCertificate, LinearConstraint};
use crate::selfcheck::{check_entailment, CheckError};
#[cfg(trust_verify)]
use core::contracts::ensures;
#[cfg(not(trust_verify))]
use trust::ensures;

/// Largest product-cell count a branch-tree certificate may claim.
///
/// The partition cardinality is the product of the per-axis slab counts, all
/// read from an untrusted certificate, so nothing in the program text bounds it
/// — the allocation checker refutes the claim table with a counterexample at
/// `2^28` cells. This is the domain answer: a partition at or above 2^20 cells
/// is not a certificate anyone means to check. It must already carry that many
/// `BranchLeaf` values (the cardinality equality below), each holding two
/// coordinate `Vec`s and a proof, so ~2^20 cells is already hundreds of MB of
/// resident certificate; the largest shipped example is 64 cells (an 8x8 grid).
///
/// The value matches `exact::MAX_EXACT_SYSTEM_DIMENSION`, deliberately, so the
/// crate carries one size doctrine rather than a second unexplained constant.
/// Rejecting is fail-CLOSED: an oversized certificate is refused, never
/// truncated and silently checked at the wrong cardinality.
const MAX_PARTITION_CELLS: usize = 1_048_576;

/// A 1-D axis partition: `edges` strictly increasing, `edges[0]` = box lower,
/// `edges[last]` = box upper. The `k`-th slab is `[edges[k], edges[k+1]]`.
#[derive(Debug, Clone)]
pub struct AxisPartition {
    /// Variable name (e.g. `"x0"`).
    pub var: String,
    /// Sorted edges; `len = ncells + 1`.
    pub edges: Vec<Rat>,
}

/// One leaf (cell) of the branch tree: `y(x) ≥ bound` over the box cell
/// `[lo[k], hi[k]]` per axis, witnessed by one selector group.
#[derive(Debug, Clone)]
pub struct BranchLeaf {
    /// Per-axis cell lower corner (parallel to the certificate's axes).
    pub lo: Vec<Rat>,
    /// Per-axis cell upper corner.
    pub hi: Vec<Rat>,
    /// Certified lower bound: `y(x) ≥ bound` for all `x` in this cell.
    pub bound: Rat,
    /// One entailment per witness-group member `i∈S_{j*}` proving
    /// `L_i(x) ≥ bound` over the cell (`a_i·x ≥ bound − b_i`).
    pub member_entailments: Vec<EntailmentCertificate>,
    /// Bias `b_i` of each member (parallel to `member_entailments`): binds the
    /// entailment conclusion constant `bound − b_i` back to `bound`.
    pub member_biases: Vec<Rat>,
}

/// Whether the property threshold is an upper (`Y_0 ≤ t`) or lower (`Y_0 ≥ t`)
/// bound. The current lower-bound certificate format supports only `Le`;
/// `Ge` is retained in the wire model but rejected until upper-bound leaf
/// evidence has a direction-aware checker and composition rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreshDir {
    /// Unsafe region `Y_0 ≤ t`; a lower-bound branch tree refutes it.
    Le,
    /// Unsafe region `Y_0 ≥ t`; an upper-bound branch tree refutes it.
    Ge,
}

/// A whole-box branch-tree certificate: an exact cell partition, a per-cell
/// bound with its supporting entailments, and the refuted property threshold.
#[derive(Debug, Clone)]
pub struct BranchTreeCertificate {
    /// Axis partitions (product grid); `axes.len()` = input dimension.
    pub axes: Vec<AxisPartition>,
    /// The leaves — must exactly equal the product grid of `axes`.
    pub leaves: Vec<BranchLeaf>,
    /// The property threshold on `Y_0`.
    pub threshold: Rat,
    /// Threshold direction.
    pub dir: ThreshDir,
}

/// Why a branch-tree certificate failed verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchError {
    /// An axis had fewer than two edges (no cell).
    DegenerateAxis(usize),
    /// Axis edges are not strictly increasing (partition not an exact cover).
    NonMonotoneAxis(usize, usize),
    /// The leaf set is not exactly the product grid (gap, overlap, or extra).
    PartitionMismatch(String),
    /// The Cartesian product of per-axis cells cannot be represented by
    /// `usize`; attempting to enumerate it would otherwise wrap or saturate.
    PartitionSizeOverflow(usize),
    /// A leaf's dimension disagrees with the number of axes.
    LeafDimension(usize),
    /// A leaf has no supporting member entailments.
    EmptyLeaf(usize),
    /// `member_entailments` / `member_biases` length mismatch.
    MemberLengthMismatch(usize, usize, usize),
    /// A member entailment premise is not a face of this leaf's cell.
    PremiseNotCellFace(usize, usize),
    /// A member entailment conclusion is not `≥` over the cell variables.
    ConclusionNotGe(usize, usize),
    /// A member entailment does not bind to the leaf's declared bound.
    BoundBindingFailed(usize, usize),
    /// The certificate direction needs evidence this format cannot express.
    UnsupportedDirection(ThreshDir),
    /// The composed global bound does not clear the threshold.
    ThresholdNotCleared(String),
    /// A per-leaf entailment failed the flat check.
    Leaf(usize, usize, CheckError),
    /// Exact arithmetic failure.
    ///
    /// Transparent: `Display` and `source` forward to the inner [`RatError`].
    Rat(RatError),
}

// Hand-written replacements for the former `thiserror::Error` derive: the
// proc-macro dependency is removed so the whole dependency stack (including
// bignum) can be verified rather than assumed. Messages, `source()` polarity
// and the `From` conversion are byte-for-byte what the derive expanded to.
impl core::fmt::Display for BranchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BranchError::DegenerateAxis(axis) => write!(f, "axis {0} has < 2 edges", axis),
            BranchError::NonMonotoneAxis(axis, index) => write!(
                f,
                "axis {0} edges are not strictly increasing at index {1}",
                axis, index
            ),
            BranchError::PartitionMismatch(detail) => {
                write!(f, "leaf set is not the exact product grid: {0}", detail)
            }
            BranchError::PartitionSizeOverflow(axis) => {
                write!(f, "partition cell count overflows usize at axis {0}", axis)
            }
            BranchError::LeafDimension(leaf) => write!(f, "leaf {0} has wrong dimension", leaf),
            BranchError::EmptyLeaf(leaf) => write!(f, "leaf {0} has no member entailments", leaf),
            BranchError::MemberLengthMismatch(leaf, entailments, biases) => write!(
                f,
                "leaf {0}: {1} entailments vs {2} biases",
                leaf, entailments, biases
            ),
            BranchError::PremiseNotCellFace(leaf, member) => write!(
                f,
                "leaf {0} member {1}: premise is not a cell face",
                leaf, member
            ),
            BranchError::ConclusionNotGe(leaf, member) => write!(
                f,
                "leaf {0} member {1}: conclusion is not a Ge inequality",
                leaf, member
            ),
            BranchError::BoundBindingFailed(leaf, member) => write!(
                f,
                "leaf {0} member {1}: conclusion + bias does not equal the leaf bound",
                leaf, member
            ),
            BranchError::UnsupportedDirection(dir) => write!(
                f,
                "threshold direction {0:?} is unsupported by lower-bound branch certificates",
                dir
            ),
            BranchError::ThresholdNotCleared(detail) => {
                write!(f, "global bound does not clear the threshold ({0})", detail)
            }
            BranchError::Leaf(leaf, member, err) => {
                write!(f, "leaf {0} member {1}: {2}", leaf, member, err)
            }
            // `#[error(transparent)]`: delegate to the inner error's `Display`.
            BranchError::Rat(err) => core::fmt::Display::fmt(err, f),
        }
    }
}

impl core::error::Error for BranchError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            // `#[error(transparent)]` forwards `source()` to the inner error
            // (it does NOT report the inner error as its own source).
            BranchError::Rat(err) => core::error::Error::source(err),
            // No other variant carries a `#[from]`/`#[source]` field; the
            // `CheckError` in `Leaf` was a plain payload rendered by `{2}`.
            _ => None,
        }
    }
}

impl From<RatError> for BranchError {
    fn from(source: RatError) -> Self {
        BranchError::Rat(source)
    }
}

/// True iff `c` is the single-variable face `var (Ge) lo` or `var (Le) hi`
/// of the cell — i.e. a box face implied by `x ∈ cell` (coefficient exactly 1).
fn is_cell_face(c: &LinearConstraint, vars: &[String], lo: &[Rat], hi: &[Rat]) -> bool {
    if c.coefficients.len() != 1 {
        return false;
    }
    let Some((name, coeff)) = c.coefficients.iter().next() else {
        return false;
    };
    if *coeff != Rat::ONE {
        return false;
    }
    // Explicit indexed scan (not `.position(closure)`): keeps the axis lookup in
    // verified code (no absent-adapter `Iterator::position`/closure-Fn
    // obligation). Identical: first index whose var name matches, else `None`.
    let mut axis_opt = None;
    for (i, v) in vars.iter().enumerate() {
        if v.as_str() == name.as_str() {
            axis_opt = Some(i);
            break;
        }
    }
    let Some(axis) = axis_opt else {
        return false;
    };
    // total: `get` (not `lo[axis]`/`hi[axis]`): `axis` indexes `vars`, and the
    // caller passes corners parallel to `vars` (leaf dimensions are checked),
    // so the `None` arm is unreachable — and fails closed ("not a face", so
    // the premise is rejected) rather than index.
    match c.kind {
        ConstraintKind::Ge => lo.get(axis).is_some_and(|b| c.constant == *b),
        ConstraintKind::Le => hi.get(axis).is_some_and(|b| c.constant == *b),
        _ => false,
    }
}

/// Locate exactly one slab `[lo, hi]` in a strictly increasing edge list.
///
/// Manual lower-bound search keeps leaf lookup logarithmic without allocating
/// string keys. The caller has already validated edge monotonicity.
fn slab_index(edges: &[Rat], lo: Rat, hi: Rat) -> Option<usize> {
    let mut left = 0usize;
    let mut right = edges.len();
    while left < right {
        // Totalized to match the `checked_add` already used below: saturation is
        // UNREACHABLE on every path this loop admits, so the rewrite is
        // behaviour-identical and merely removes the MIR overflow asserts rather
        // than assuming them away.
        //   `right - left`     the loop guard IS `left < right`, so no underflow.
        //   `left + (r-l)/2`   `(right-left)/2 < right-left`, so `mid < right`.
        //   `mid + 1`          `mid < right <= edges.len() <= usize::MAX`.
        // The two arithmetic obligations here were blocked by the fresh-private-
        // consumer PDR replay wall — an invariant WAS found and could not be
        // admitted — so totalizing on the ny side sidesteps a research-grade
        // verifier blocker with no toolchain change.
        let mid = left.saturating_add(right.saturating_sub(left) / 2);
        let edge = *edges.get(mid)?;
        if edge < lo {
            left = mid.saturating_add(1);
        } else {
            right = mid;
        }
    }
    if edges.get(left) == Some(&lo) && edges.get(left.checked_add(1)?) == Some(&hi) {
        Some(left)
    } else {
        None
    }
}

/// Verify a branch-tree certificate and return the composed global bound.
///
/// On success returns `(global_bound, threshold)` where `global_bound =
/// min_C bound_C` is a sound bound on `y` over the whole box, and (for the `Le`
/// direction) `global_bound > threshold`, so the property `Y_0 ≤ threshold` is
/// UNSAT. `Ge` certificates are rejected because leaves currently contain only
/// lower-bound (`≥`) entailments, not the upper-bound evidence needed to refute
/// `Y_0 ≥ threshold`.
///
/// # Soundness contract (L1 — Trust = Clean fusion)
/// If this returns `Ok((g, t))`, then: (1) the leaves EXACTLY partition the box
/// (each axis' edges strictly increase `lo→hi`; the leaf set equals the product
/// grid); (2) every leaf's member entailments are individually valid
/// ([`check_entailment`]) over that leaf's own box faces and bind to the leaf's
/// declared bound; (3) `g` is the minimum of the per-leaf lower bounds and
/// clears the `Le` threshold. By the cited, kernel-re-typechecked `branch_split_min`
/// applied recursively over the grid, `g` is a sound bound on `y` over the whole
/// box — so the composed verdict is a proof, not a float claim. The leaf
/// entailments' grounding is [`check_entailment`]'s `farkas_premise_combination`;
/// the composition's is `branch_split_min`. No new axiom. L0 safety is
/// tRustc-checkable.
///
/// # Errors
/// [`BranchError`] on any malformed axis, non-exact partition, invalid or
/// mis-bound leaf entailment, or an uncleared threshold.
#[ensures(|r: &Result<(Rat, Rat), BranchError>| !matches!(r, Ok((g, t)) if g == t))]
#[trust::cite(crownproof::branch_split_min)]
// `?` here would desugar to `from_residual` return paths the verifier's
// ordering-witness grounding cannot aggregate over — the explicit match/early
// returns ARE the proof shape (see the extract-then-guard comment).
#[allow(clippy::question_mark)]
pub fn check_branch_tree(cert: &BranchTreeCertificate) -> Result<(Rat, Rat), BranchError> {
    if poisoned() {
        return Err(crate::err_barrier(BranchError::Rat(RatError::Poisoned)));
    }
    if cert.dir != ThreshDir::Le {
        return Err(crate::err_barrier(BranchError::UnsupportedDirection(
            cert.dir,
        )));
    }

    // Extract-then-guard: makes the `#[ensures]` locally provable. The match
    // only EXTRACTS (the Err arm returns early), the direction-matched
    // threshold-clearance guard is straight-line on the extracted pair, and
    // the tail is a plain `Ok((g, t))` — so every return path constructs its
    // `Ok`/`Err` in the direct predecessor of the return block and the
    // guard's cleared edge dominates the `Ok` (the verifier's
    // ordering-witness grounding window; `check_branch_tree_inner` owns
    // droppable locals whose end-of-body drops would split its own tail
    // construction out of that window). The `Rat` ordering witnesses (`>` /
    // `<`, NOT a handle `==`, whose derived `PartialEq` the extractor erases
    // to an ambiguous total-call sentinel) give the grounding lane
    // `g > t ∨ g < t` on the cleared edge — which entails the contract's
    // `g != t` in the arena total order. The guard is unreachable by
    // construction — the inner step-4 `cleared` check already admitted the
    // same strict inequality on the same pair — so this is
    // behavior-identical, fail-closed hardening.
    // `Option` carrier: see the note in `selfcheck::check_entailment`. A
    // `TRUST_LENWITNESS_DEBUG=1` probe traced one return path here (bb32->bb33)
    // whose `_0 def=NONE (Call-dest)`, against zero such paths in
    // `selfcheck::check_chain`, which proves. Because
    // `check_branch_tree_inner` returns the SAME `Result<(Rat, Rat),
    // BranchError>` this wrapper does, the optimizer forwards its Result rather
    // than rebuilding the tail `Ok`, so no in-body aggregate exists on that path
    // and the discr gate fails closed.
    let carried: Option<(Rat, Rat)> = match check_branch_tree_inner(cert) {
        Ok(pair) => Some(pair),
        // `crate::err_barrier` (identity, `#[inline(never)]`): a fresh in-body
        // `Err` aggregate, not a whole-`Result` forward the return-grounding
        // lane cannot see (nor a const-promoted+merged unit variant).
        Err(e) => {
            if poisoned() {
                return Err(crate::err_barrier(BranchError::Rat(RatError::Poisoned)));
            }
            return Err(crate::err_barrier(e));
        }
    };
    // UNREACHABLE: the `Err` arm returns, so every path here stored `Some`.
    // Reuses an error this function already constructs, adding no call and no
    // allocation for the verifier to discharge.
    let (g, t) = match carried {
        Some(pair) => pair,
        None => return Err(crate::err_barrier(BranchError::Rat(RatError::Poisoned))),
    };
    let cleared = g > t;
    if !cleared {
        if poisoned() {
            return Err(crate::err_barrier(BranchError::Rat(RatError::Poisoned)));
        }
        return Err(crate::err_barrier(BranchError::ThresholdNotCleared(
            "internal: strict-inequality invariant violated after compose (unreachable)".to_owned(),
        )));
    }
    if poisoned() {
        return Err(crate::err_barrier(BranchError::Rat(RatError::Poisoned)));
    }
    Ok((g, t))
}

/// The full verification behind [`check_branch_tree`]. Private and
/// contract-free: the ensures-bearing wrapper re-establishes the
/// strict-inequality invariant with an in-body direction-matched ordering
/// guard (this body's pervasive `?` returns are `from_residual` paths, and
/// its `Vec` end-of-body drops split the tail `Ok` construction out of the
/// local proof's grounding window).
fn check_branch_tree_inner(cert: &BranchTreeCertificate) -> Result<(Rat, Rat), BranchError> {
    // Every leaf below proves `y >= bound`. Such lower bounds can refute only
    // an unsafe `y <= threshold` region. Treating them as upper bounds for
    // `Ge` would be a false proof, so fail closed until the schema carries and
    // checks direction-aware upper-bound evidence.
    if cert.dir != ThreshDir::Le {
        return Err(BranchError::UnsupportedDirection(cert.dir));
    }

    // --- 1. Axis partitions are exact 1-D covers -------------------------------
    for (ai, axis) in cert.axes.iter().enumerate() {
        if axis.edges.len() < 2 {
            return Err(BranchError::DegenerateAxis(ai));
        }
        // total: adjacent-pair walk via zip (not `edges[k]`/`edges[k + 1]`):
        // identical pairs and indices, no slice-bounds obligation.
        for (k, (a, b)) in axis.edges.iter().zip(axis.edges.iter().skip(1)).enumerate() {
            if a >= b {
                return Err(BranchError::NonMonotoneAxis(ai, k));
            }
        }
    }
    // Explicit Vec::new()+push (not `.collect()`): the length is the
    // input-derived axis count, so a bulk `.collect()` raises an
    // unbounded-alloc obligation; identical elements and order.
    let mut vars: Vec<String> = Vec::new();
    for a in &cert.axes {
        vars.push(a.var.clone());
    }

    // --- 2. Leaves are EXACTLY the product grid --------------------------------
    // Every axis passed the degenerate-axis guard. Compute the product with
    // checked arithmetic, then require the compact certificate's cardinality to
    // agree before allocating claim flags or inspecting per-leaf proofs.
    let mut ncells: usize = 1;
    for (axis_index, a) in cert.axes.iter().enumerate() {
        let axis_cells = a.edges.len().saturating_sub(1);
        ncells = ncells
            .checked_mul(axis_cells)
            .ok_or(BranchError::PartitionSizeOverflow(axis_index))?;
    }
    // Reject a cardinality mismatch before enumerating the product. Otherwise
    // a compact certificate with many two-cell axes and no leaves can force an
    // enormous (but still representable) loop.
    if cert.leaves.len() != ncells {
        return Err(BranchError::PartitionMismatch(format!(
            "{} leaves vs {} product cells",
            cert.leaves.len(),
            ncells
        )));
    }
    // Dominating size cap (fail-closed), guarding the SAME operand the allocation
    // below consumes. Without it the claim table is bounded only by `ncells`, which
    // is derived from the parsed certificate's axis edge counts and is therefore
    // untrusted: the verifier refutes the allocation with a counterexample at
    // `ncells = 2^28`. `try_reserve_exact` alone does not answer that — under memory
    // overcommit it can return `Ok` on an unbacked mapping, leaving the `resize`
    // below to fault in every page.
    //
    // Deliberately the bare `if x >= MAX { return Err }` shape rather than an inline
    // `.min(MAX)`: truncation would silently check a SMALLER partition than the
    // certificate claims, which is fail-wrong for a proof checker. The bare form is
    // also the one the allocation checker demonstrably accepts (see `exact.rs`,
    // whose `MAX_EXACT_SYSTEM_DIMENSION` guard discharges the same obligation).
    // Rebind to a fresh single-assignment local before guarding. `ncells` is
    // assigned inside the `checked_mul` loop above, so it reads as
    // may-reassigned at every later program point and a guard written against it
    // is dropped by the checker's versioning kill -- measured in R40, where the
    // guard on `ncells` left this row refuted while the identically-shaped guard
    // in `fp_margin::deployed_fp_margin` (over a fresh `let dim = mags.len()`)
    // proved. `cells` is written once and never again, so the guard survives to
    // the allocation that consumes it.
    let cells = ncells;
    if cells >= MAX_PARTITION_CELLS {
        return Err(BranchError::PartitionMismatch(format!(
            "partition has {cells} product cells, at or above the {MAX_PARTITION_CELLS} \
             cell limit; refusing to allocate a claim table this large"
        )));
    }
    let mut used: Vec<bool> = Vec::new();
    used.try_reserve_exact(cells).map_err(|error| {
        BranchError::PartitionMismatch(format!(
            "claim table for {cells} product cells cannot be allocated: {error}"
        ))
    })?;
    used.resize(cells, false);
    for (li, leaf) in cert.leaves.iter().enumerate() {
        if leaf.lo.len() != cert.axes.len() || leaf.hi.len() != cert.axes.len() {
            return Err(BranchError::LeafDimension(li));
        }
        // Map the leaf directly to its mixed-radix grid index (axis 0 is the
        // fastest-changing digit). This avoids materializing every cell key and
        // the former O(leaves × cells) string scan.
        let mut pos = 0usize;
        let mut stride = 1usize;
        for (axis_index, ((axis, &lo), &hi)) in
            cert.axes.iter().zip(&leaf.lo).zip(&leaf.hi).enumerate()
        {
            let Some(slab) = slab_index(&axis.edges, lo, hi) else {
                return Err(BranchError::PartitionMismatch(format!(
                    "leaf {li} is not a product-grid cell on axis {axis_index} ({})",
                    axis.var
                )));
            };
            let offset = slab
                .checked_mul(stride)
                .ok_or(BranchError::PartitionSizeOverflow(axis_index))?;
            pos = pos
                .checked_add(offset)
                .ok_or(BranchError::PartitionSizeOverflow(axis_index))?;
            stride = stride
                .checked_mul(axis.edges.len().saturating_sub(1))
                .ok_or(BranchError::PartitionSizeOverflow(axis_index))?;
        }
        let Some(claimed) = used.get_mut(pos) else {
            return Err(BranchError::PartitionMismatch(
                "internal: claim flag missing for a grid cell (unreachable)".to_owned(),
            ));
        };
        if *claimed {
            return Err(BranchError::PartitionMismatch(format!(
                "leaf {li} duplicates a cell"
            )));
        }
        *claimed = true;
    }
    // Bijection: same size + every leaf in the grid + no dup ⇒ exact cover.

    // --- 3. Each leaf's per-cell facts are valid & bind to its bound -----------
    let mut global: Option<Rat> = None;
    for (li, leaf) in cert.leaves.iter().enumerate() {
        if leaf.member_entailments.is_empty() {
            return Err(BranchError::EmptyLeaf(li));
        }
        if leaf.member_entailments.len() != leaf.member_biases.len() {
            return Err(BranchError::MemberLengthMismatch(
                li,
                leaf.member_entailments.len(),
                leaf.member_biases.len(),
            ));
        }
        for (mi, (ent, bias)) in leaf
            .member_entailments
            .iter()
            .zip(&leaf.member_biases)
            .enumerate()
        {
            // (a) the entailment is a valid non-negative combination.
            check_entailment(ent).map_err(|e| BranchError::Leaf(li, mi, e))?;
            // (b) every premise is a box FACE of THIS cell (so the derived
            //     conclusion holds over the whole cell).
            for p in &ent.premises {
                if !is_cell_face(p, &vars, &leaf.lo, &leaf.hi) {
                    return Err(BranchError::PremiseNotCellFace(li, mi));
                }
            }
            // (c) the conclusion is `a·x ≥ bound − b_i`, binding to leaf.bound.
            if ent.conclusion.kind != ConstraintKind::Ge {
                return Err(BranchError::ConclusionNotGe(li, mi));
            }
            let reconstructed = ent.conclusion.constant.add(*bias)?;
            if reconstructed != leaf.bound {
                return Err(BranchError::BoundBindingFailed(li, mi));
            }
        }
        // (d) compose the global lower bound as the minimum over leaves.
        global = Some(match global {
            None => leaf.bound,
            Some(g) => {
                if leaf.bound < g {
                    leaf.bound
                } else {
                    g
                }
            }
        });
    }
    // total: fail-closed `else` (not `.expect`): `ncells >= 1` always (an empty
    // axis list has product 1; each axis contributes `edges.len() - 1 >= 1`
    // slabs) and `leaves.len() == ncells` was checked, so step 3's loop ran at
    // least once and `global` is `Some` — the reject arm is unreachable, and an
    // (impossible) empty leaf set now rejects instead of panicking.
    let Some(global) = global else {
        return Err(BranchError::PartitionMismatch(
            "internal: empty leaf set after exact-cover check".to_owned(),
        ));
    };

    // --- 4. The composed bound clears the threshold ----------------------------
    let cleared = global > cert.threshold;
    if !cleared {
        return Err(BranchError::ThresholdNotCleared(format!(
            "global={}/{} thresh={}/{} dir={:?}",
            global.num(),
            global.den(),
            cert.threshold.num(),
            cert.threshold.den(),
            cert.dir
        )));
    }
    Ok((global, cert.threshold))
}

fn dir_str(d: ThreshDir) -> &'static str {
    match d {
        ThreshDir::Le => "le",
        ThreshDir::Ge => "ge",
    }
}

/// Encode a slice of rationals as a JSON array of their clean strings.
///
/// A free `fn` (not a `|xs| ..` closure) with an explicit loop (not
/// `.map(closure).collect::<Result<Vec<_>, _>>()?`): keeps the encoding in
/// verified code (no absent-adapter `Iterator::map`/`Result::map`/closure-Fn
/// obligation). Identical elements, order, and early-`Err` propagation.
fn rat_arr_json(xs: &[Rat]) -> Result<Json, RatError> {
    let mut out: Vec<Json> = Vec::new();
    for x in xs {
        let s = x.to_clean_string()?;
        out.push(Json::Str(s));
    }
    Ok(Json::Arr(out))
}

/// Serialize a branch-tree certificate to a self-describing JSON envelope.
///
/// The envelope records the exact axis partition, the property threshold and
/// direction, the composition rule citation (`branch_split_min`), and — per leaf
/// — the cell corners, the certified `bound`, and every member entailment (as a
/// Clean-canonical `entailment_certificate`). The wire format retains
/// `"direction": "ge"` for compatibility, but the current leaves prove only
/// lower bounds and [`check_branch_tree`] rejects every `Ge` certificate. For a
/// supported `Le` certificate, a consumer re-checks the exact product partition,
/// verifies every embedded entailment, takes the minimum leaf lower bound, and
/// compares it to the threshold, backed by `branch_split_min`. Serialized data
/// alone is not proof evidence; consumers must apply those checks and must not
/// infer a `Ge` verdict from this lower-bound envelope.
///
/// # Errors
/// Propagates rational-encoding failures, including a poisoned rational arena.
pub fn branch_tree_to_json(cert: &BranchTreeCertificate) -> Result<Json, RatError> {
    ensure_healthy()?;
    // Explicit loops (not `.map(closure).collect::<Result<Vec<_>, _>>()?`): keeps
    // the per-axis / per-leaf JSON construction in verified code (no
    // absent-adapter `Iterator::map`/`Result::map`/closure-Fn obligations).
    // Identical elements, order, and early-`Err` propagation; `rat_arr_json` is a
    // free `fn` (not a closure) for the same reason.
    let mut axes: Vec<Json> = Vec::new();
    for a in &cert.axes {
        let mut edges: Vec<Json> = Vec::new();
        for e in &a.edges {
            let s = e.to_clean_string()?;
            edges.push(Json::Str(s));
        }
        let mut o = Map::new();
        o.insert("var".to_owned(), Json::Str(a.var.clone()));
        o.insert("edges".to_owned(), Json::Arr(edges));
        axes.push(Json::Obj(o));
    }

    let mut leaves: Vec<Json> = Vec::new();
    for leaf in &cert.leaves {
        // Explicit Vec::new()+push (not `.collect()`): the length is the
        // input-derived member count, so a bulk `.collect()` raises an
        // unbounded-alloc obligation; identical elements, order, and
        // early-`Err` propagation.
        let mut ents = Vec::new();
        for ent in &leaf.member_entailments {
            ents.push(entailment_to_json(ent)?);
        }
        let mut o = Map::new();
        o.insert("cell_lo".to_owned(), rat_arr_json(&leaf.lo)?);
        o.insert("cell_hi".to_owned(), rat_arr_json(&leaf.hi)?);
        o.insert("bound".to_owned(), Json::Str(leaf.bound.to_clean_string()?));
        o.insert(
            "member_biases".to_owned(),
            rat_arr_json(&leaf.member_biases)?,
        );
        o.insert("member_entailments".to_owned(), Json::Arr(ents));
        leaves.push(Json::Obj(o));
    }

    let mut root = Map::new();
    root.insert(
        "type".to_owned(),
        Json::Str("branch_tree_certificate".to_owned()),
    );
    root.insert("version".to_owned(), Json::Str("1.0".to_owned()));
    root.insert(
        "composition_rule".to_owned(),
        Json::Str("crownproof::branch_split_min".to_owned()),
    );
    root.insert("axes".to_owned(), Json::Arr(axes));
    root.insert(
        "threshold".to_owned(),
        Json::Str(cert.threshold.to_clean_string()?),
    );
    root.insert(
        "direction".to_owned(),
        Json::Str(dir_str(cert.dir).to_owned()),
    );
    root.insert("leaves".to_owned(), Json::Arr(leaves));
    let value = Json::Obj(root);
    ensure_healthy()?;
    Ok(value)
}

/// Serialize EVERY leaf member entailment as a flat JSON array of tagged
/// `entailment_certificate` objects — the exact input Clean's batch external-cert
/// endpoint (`BatchVerifyExternalCert`) consumes, so all per-cell facts are
/// kernel-re-checked in one call. The branch-tree envelope then only adds the
/// (arithmetically trivial, `branch_split_min`-backed) partition + min step.
///
/// # Errors
/// Propagates rational-encoding failures, including a poisoned rational arena
/// even when the leaf batch is empty.
pub fn branch_tree_leaf_batch_json(cert: &BranchTreeCertificate) -> Result<Json, RatError> {
    ensure_healthy()?;
    let mut items = Vec::new();
    for leaf in &cert.leaves {
        for ent in &leaf.member_entailments {
            items.push(entailment_to_json(ent)?);
        }
    }
    let value = Json::Arr(items);
    ensure_healthy()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PoisonReset;

    impl Drop for PoisonReset {
        fn drop(&mut self) {
            crate::rational::set_poisoned_for_test(false);
        }
    }

    fn r(n: i128, d: i128) -> Rat {
        Rat::new(n, d).unwrap()
    }

    /// Face `var (kind) bound` with unit coefficient.
    fn face(var: &str, kind: ConstraintKind, bound: Rat) -> LinearConstraint {
        LinearConstraint::with_kind(kind, &[(var, Rat::ONE)], bound)
    }

    /// Entailment `a0*x0 + a1*x1 >= bound - b` over cell, corner faces + |a|.
    fn ent(a0: Rat, a1: Rat, b: Rat, bound: Rat, lo: &[Rat], hi: &[Rat]) -> EntailmentCertificate {
        let (p0, mu0) = if !a0.is_negative() {
            (face("x0", ConstraintKind::Ge, lo[0]), a0)
        } else {
            (face("x0", ConstraintKind::Le, hi[0]), a0.neg())
        };
        let (p1, mu1) = if !a1.is_negative() {
            (face("x1", ConstraintKind::Ge, lo[1]), a1)
        } else {
            (face("x1", ConstraintKind::Le, hi[1]), a1.neg())
        };
        EntailmentCertificate {
            premises: vec![p0, p1],
            multipliers: vec![mu0, mu1],
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x0", a0), ("x1", a1)],
                bound.sub(b).unwrap(),
            ),
        }
    }

    /// A tiny 2-cell (split x0 at 0) partition of [-1,1]x[-1,1] with a single
    /// affine y = L_0(x) = x0 (a0=1). Per-cell min: left cell = -1, right = 0.
    fn two_cell_cert(threshold: Rat) -> BranchTreeCertificate {
        let lo = r(-1, 1);
        let mid = r(0, 1);
        let hi = r(1, 1);
        let a0 = r(1, 1);
        let a1 = r(0, 1);
        let b = r(0, 1);
        let mk = |x0lo: Rat, x0hi: Rat, bound: Rat| BranchLeaf {
            lo: vec![x0lo, lo],
            hi: vec![x0hi, hi],
            bound,
            member_entailments: vec![ent(a0, a1, b, bound, &[x0lo, lo], &[x0hi, hi])],
            member_biases: vec![b],
        };
        BranchTreeCertificate {
            axes: vec![
                AxisPartition {
                    var: "x0".to_owned(),
                    edges: vec![lo, mid, hi],
                },
                AxisPartition {
                    var: "x1".to_owned(),
                    edges: vec![lo, hi],
                },
            ],
            leaves: vec![mk(lo, mid, r(-1, 1)), mk(mid, hi, r(0, 1))],
            threshold,
            dir: ThreshDir::Le,
        }
    }

    #[test]
    fn accepts_exact_partition_clearing_threshold() {
        // global = min(-1, 0) = -1 > threshold -2.
        let cert = two_cell_cert(r(-2, 1));
        let (g, t) = check_branch_tree(&cert).expect("valid cert accepted");
        assert_eq!(g, r(-1, 1));
        assert_eq!(t, r(-2, 1));
    }

    #[test]
    fn rejects_uncleared_threshold() {
        // global = -1, threshold = -1: not strictly cleared.
        let cert = two_cell_cert(r(-1, 1));
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::ThresholdNotCleared(_))
        ));
    }

    #[test]
    fn rejects_ge_direction_without_upper_bound_evidence() {
        // These leaves validly prove only lower bounds on y=x0. The old Ge
        // path incorrectly treated max(lower bounds)=0 as an upper bound and
        // would refute y>=1 even though y=1 at x0=1.
        let mut cert = two_cell_cert(r(1, 1));
        cert.dir = ThreshDir::Ge;
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::UnsupportedDirection(ThreshDir::Ge))
        ));
    }

    #[test]
    fn rejects_partition_gap() {
        // Drop the second leaf: only 1 leaf vs 2 product cells -> mismatch.
        let mut cert = two_cell_cert(r(-2, 1));
        cert.leaves.pop();
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::PartitionMismatch(_))
        ));
    }

    #[test]
    fn rejects_partition_cell_count_overflow_before_enumeration() {
        // Each axis contributes two cells. One more axis than usize has bits
        // makes the Cartesian-product count overflow with only a tiny input.
        let axes = (0..=usize::BITS)
            .map(|i| AxisPartition {
                var: format!("x{i}"),
                edges: vec![Rat::ZERO, Rat::ONE, Rat::from_int(2)],
            })
            .collect();
        let cert = BranchTreeCertificate {
            axes,
            leaves: Vec::new(),
            threshold: Rat::ZERO,
            dir: ThreshDir::Le,
        };
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::PartitionSizeOverflow(_))
        ));
    }

    #[test]
    fn slab_lookup_handles_large_axes_without_linear_key_scans() {
        let edges: Vec<Rat> = (0..=10_000).map(Rat::from_int).collect();
        assert_eq!(
            slab_index(&edges, Rat::from_int(9_999), Rat::from_int(10_000)),
            Some(9_999)
        );
        assert_eq!(
            slab_index(&edges, Rat::from_int(9_998), Rat::from_int(10_000)),
            None,
            "a leaf may claim only one exact adjacent slab"
        );
    }

    #[test]
    fn rejects_leaf_count_mismatch_before_large_product_enumeration() {
        let axes = (0..40)
            .map(|i| AxisPartition {
                var: format!("x{i}"),
                edges: vec![Rat::ZERO, Rat::ONE, Rat::from_int(2)],
            })
            .collect();
        let cert = BranchTreeCertificate {
            axes,
            leaves: Vec::new(),
            threshold: Rat::ZERO,
            dir: ThreshDir::Le,
        };
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::PartitionMismatch(_))
        ));
    }

    #[test]
    fn rejects_non_covering_edges() {
        // Edges not spanning: make an axis non-monotone.
        let mut cert = two_cell_cert(r(-2, 1));
        cert.axes[0].edges = vec![r(-1, 1), r(-1, 1), r(1, 1)]; // duplicate edge
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::NonMonotoneAxis(0, _))
        ));
    }

    #[test]
    fn rejects_bound_overclaim() {
        // Inflate the left leaf's bound above its true corner min (-1): the
        // member entailment then fails to bind / verify.
        let mut cert = two_cell_cert(r(-2, 1));
        cert.leaves[0].bound = r(0, 1); // claims y>=0 on [-1,0] where min is -1
        let err = check_branch_tree(&cert).unwrap_err();
        // Either the entailment no longer verifies, or the bound-binding fails.
        assert!(
            matches!(err, BranchError::Leaf(0, 0, _))
                || matches!(err, BranchError::BoundBindingFailed(0, 0)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn rejects_premise_not_cell_face() {
        // Add a premise that is NOT a face of this cell (x0 <= 5; the left cell's
        // hi is 0) with a zero multiplier — the entailment still VERIFIES, but the
        // cell-face guard must reject it (only genuine box faces are admissible).
        let mut cert = two_cell_cert(r(-2, 1));
        let e = &mut cert.leaves[0].member_entailments[0];
        e.premises.push(face("x0", ConstraintKind::Le, r(5, 1)));
        e.multipliers.push(r(0, 1));
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::PremiseNotCellFace(0, 0))
        ));
    }

    #[test]
    fn checker_and_emitters_refuse_poison_on_degenerate_payloads() {
        let cert = BranchTreeCertificate {
            axes: Vec::new(),
            leaves: Vec::new(),
            threshold: Rat::ZERO,
            dir: ThreshDir::Ge,
        };
        crate::rational::set_poisoned_for_test(true);
        let _reset = PoisonReset;

        assert_eq!(
            check_branch_tree(&cert),
            Err(BranchError::Rat(RatError::Poisoned))
        );
        assert_eq!(branch_tree_to_json(&cert), Err(RatError::Poisoned));
        assert_eq!(branch_tree_leaf_batch_json(&cert), Err(RatError::Poisoned));
    }
}
