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

use crate::rational::{Rat, RatError};
use crate::schema::{entailment_to_json, ConstraintKind, EntailmentCertificate, LinearConstraint};
use crate::selfcheck::{check_entailment, CheckError};
#[cfg(trust_verify)]
use core::contracts::ensures;
#[cfg(not(trust_verify))]
use trust::ensures;

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
/// bound. UNSAT for `Le` iff `min y > t`; for `Ge` iff `max y < t`.
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
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BranchError {
    /// An axis had fewer than two edges (no cell).
    #[error("axis {0} has < 2 edges")]
    DegenerateAxis(usize),
    /// Axis edges are not strictly increasing (partition not an exact cover).
    #[error("axis {0} edges are not strictly increasing at index {1}")]
    NonMonotoneAxis(usize, usize),
    /// The leaf set is not exactly the product grid (gap, overlap, or extra).
    #[error("leaf set is not the exact product grid: {0}")]
    PartitionMismatch(String),
    /// A leaf's dimension disagrees with the number of axes.
    #[error("leaf {0} has wrong dimension")]
    LeafDimension(usize),
    /// A leaf has no supporting member entailments.
    #[error("leaf {0} has no member entailments")]
    EmptyLeaf(usize),
    /// `member_entailments` / `member_biases` length mismatch.
    #[error("leaf {0}: {1} entailments vs {2} biases")]
    MemberLengthMismatch(usize, usize, usize),
    /// A member entailment premise is not a face of this leaf's cell.
    #[error("leaf {0} member {1}: premise is not a cell face")]
    PremiseNotCellFace(usize, usize),
    /// A member entailment conclusion is not `≥` over the cell variables.
    #[error("leaf {0} member {1}: conclusion is not a Ge inequality")]
    ConclusionNotGe(usize, usize),
    /// A member entailment does not bind to the leaf's declared bound.
    #[error("leaf {0} member {1}: conclusion + bias does not equal the leaf bound")]
    BoundBindingFailed(usize, usize),
    /// The composed global bound does not clear the threshold.
    #[error("global bound does not clear the threshold ({0})")]
    ThresholdNotCleared(String),
    /// A per-leaf entailment failed the flat check.
    #[error("leaf {0} member {1}: {2}")]
    Leaf(usize, usize, CheckError),
    /// Exact arithmetic failure.
    #[error(transparent)]
    Rat(#[from] RatError),
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

/// Verify a branch-tree certificate and return the composed global bound.
///
/// On success returns `(global_bound, threshold)` where `global_bound =
/// min_C bound_C` is a sound bound on `y` over the whole box, and (for the `Le`
/// direction) `global_bound > threshold`, so the property `Y_0 ≤ threshold` is
/// UNSAT. For `Ge`, `global_bound = max_C bound_C < threshold`.
///
/// # Soundness contract (L1 — Trust = Clean fusion)
/// If this returns `Ok((g, t))`, then: (1) the leaves EXACTLY partition the box
/// (each axis' edges strictly increase `lo→hi`; the leaf set equals the product
/// grid); (2) every leaf's member entailments are individually valid
/// ([`check_entailment`]) over that leaf's own box faces and bind to the leaf's
/// declared bound; (3) `g` is the min (resp. max) of the per-leaf bounds and
/// clears the threshold. By the cited, kernel-re-typechecked `branch_split_min`
/// applied recursively over the grid, `g` is a sound bound on `y` over the whole
/// box — so the composed verdict is a proof, not a float claim. The leaf
/// entailments' grounding is [`check_entailment`]'s `farkas_premise_combination`;
/// the composition's is `branch_split_min`. No new axiom. L0 safety is
/// tRustc-checkable.
///
/// # Errors
/// [`BranchError`] on any malformed axis, non-exact partition, invalid or
/// mis-bound leaf entailment, or an uncleared threshold.
// CONTRACT FIX: the old predicate `!(… if g <= t)` was FALSIFIABLE — for
// `ThreshDir::Ge` the cleared verdict is `global < threshold`, so the Ok pair
// has g < t (masked as Unknown until the grounding lane could refute it). The
// direction-independent invariant BOTH arms guarantee is STRICT inequality.
#[ensures(|r: &Result<(Rat, Rat), BranchError>| !matches!(r, Ok((g, t)) if g == t))]
#[trust::cite(crownproof::branch_split_min)]
// `?` here would desugar to `from_residual` return paths the verifier's
// ordering-witness grounding cannot aggregate over — the explicit match/early
// returns ARE the proof shape (see the extract-then-guard comment).
#[allow(clippy::question_mark)]
pub fn check_branch_tree(cert: &BranchTreeCertificate) -> Result<(Rat, Rat), BranchError> {
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
    let (g, t) = match check_branch_tree_inner(cert) {
        Ok(pair) => pair,
        // `crate::err_barrier` (identity, `#[inline(never)]`): a fresh in-body
        // `Err` aggregate, not a whole-`Result` forward the return-grounding
        // lane cannot see (nor a const-promoted+merged unit variant).
        Err(e) => return Err(crate::err_barrier(e)),
    };
    let cleared = match cert.dir {
        ThreshDir::Le => g > t,
        ThreshDir::Ge => g < t,
    };
    if !cleared {
        return Err(crate::err_barrier(BranchError::ThresholdNotCleared(
            "internal: strict-inequality invariant violated after compose (unreachable)".to_owned(),
        )));
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
    // Build the expected cell keys (lo,hi corner rationals as strings). A plain
    // `Vec<String>` (not `BTreeSet::new`/`insert`/`contains`): keeps the
    // membership machinery in verified code (no absent std-collection
    // obligation). Grid keys are pairwise distinct (edges strictly increase and
    // the `|`/`;` separators never occur in clean rational strings, so the key
    // encoding is injective over distinct index tuples), and membership below is
    // a first-match scan — identical accept/reject to the old set.
    // Explicit loop (not `.map(closure).product()`): keeps the cell-count
    // product in verified code (no absent-adapter `Iterator::map`/`product` or
    // closure-Fn obligation). `saturating_sub`/`saturating_mul` (not `- 1`/`*`):
    // every axis passed the `edges.len() < 2` DegenerateAxis guard above, so
    // neither saturates — identical slab count, no underflow/overflow VC (same
    // totalization as the `dims` loop below).
    let mut ncells: usize = 1;
    for a in &cert.axes {
        ncells = ncells.saturating_mul(a.edges.len().saturating_sub(1));
    }
    let key = |lo: &[Rat], hi: &[Rat]| -> Result<String, BranchError> {
        // total: length guard + zip (not `lo[k]`/`hi[k]`): both call sites
        // pass equal-length corners (grid-built / LeafDimension-checked), so
        // the guard is unreachable and fails closed (reject) rather than
        // index or silently truncate.
        if lo.len() != hi.len() {
            return Err(BranchError::PartitionMismatch(
                "cell corner arity mismatch".to_owned(),
            ));
        }
        let mut s = String::new();
        for (l, h) in lo.iter().zip(hi) {
            s.push_str(&l.to_clean_string()?);
            s.push('|');
            s.push_str(&h.to_clean_string()?);
            s.push(';');
        }
        Ok(s)
    };
    let mut expected_keys: Vec<String> = Vec::new();
    // Cartesian product of the per-axis slabs.
    // Explicit Vec::new()+push (not `.collect()` / `vec![0; n]`): input-derived
    // bulk allocs raise unbounded-alloc obligations; the loop builds the
    // identical parallel (slab-count, cursor) vectors.
    let mut dims: Vec<usize> = Vec::new();
    let mut idx: Vec<usize> = Vec::new();
    for a in &cert.axes {
        // total: `saturating_sub` (not `- 1`): every axis passed the
        // `edges.len() < 2` DegenerateAxis guard above, so the subtraction
        // never saturates — identical slab count, no underflow VC.
        dims.push(a.edges.len().saturating_sub(1));
        idx.push(0);
    }
    for _ in 0..ncells {
        // total: zip + `get` (not `axes[d].edges[idx[d]]` collects): every
        // cursor stays `< edges.len() - 1` by the mixed-radix wrap below, so
        // the `None` arm is unreachable — and fails closed (reject) rather
        // than index.
        let mut lo: Vec<Rat> = Vec::new();
        let mut hi: Vec<Rat> = Vec::new();
        // total: `saturating_add` (not `k + 1`): `k < edges.len() - 1 <=
        // isize::MAX` by the mixed-radix wrap, so the add never saturates; a
        // (unreachable) saturated cursor makes `get` return `None` — the same
        // fail-closed reject arm below.
        for (axis, &k) in cert.axes.iter().zip(&idx) {
            let (Some(l), Some(h)) = (axis.edges.get(k), axis.edges.get(k.saturating_add(1)))
            else {
                return Err(BranchError::PartitionMismatch(format!(
                    "internal: slab cursor {k} outside axis {}",
                    axis.var
                )));
            };
            lo.push(*l);
            hi.push(*h);
        }
        expected_keys.push(key(&lo, &hi)?);
        // increment mixed-radix counter over the per-axis slab indices.
        // total: parallel walk via zip (not `idx[d]`/`dims[d]`): `idx` and
        // `dims` are built together (same length), so this is the identical
        // carry loop with no slice-bounds obligation.
        for (slot, &dim) in idx.iter_mut().zip(&dims) {
            // total: `saturating_add` (not `+= 1`): the cursor is `< dim <=
            // isize::MAX` on entry (reset below whenever it reaches `dim`), so
            // the add never saturates — identical carry, no overflow VC.
            *slot = (*slot).saturating_add(1);
            if *slot < dim {
                break;
            }
            *slot = 0;
        }
    }
    if cert.leaves.len() != ncells {
        return Err(BranchError::PartitionMismatch(format!(
            "{} leaves vs {} product cells",
            cert.leaves.len(),
            ncells
        )));
    }
    // Claim flags parallel to `expected_keys` (was `seen: BTreeSet<String>`):
    // `used[i]` marks grid cell `i` as already claimed by an earlier leaf.
    // Every key the old code inserted into `seen` had just passed the
    // `expected` membership test, so seen-membership ≡ the claim flag on the
    // (deterministic) first-match index — identical dup detection and error
    // order. Explicit Vec::new()+push (not `vec![false; n]`): input-derived
    // bulk allocs raise unbounded-alloc obligations; identical parallel vector.
    let mut used: Vec<bool> = Vec::new();
    // Push loop (not `vec![false; n]`): input-derived bulk fill raises an
    // unbounded-alloc obligation; identical parallel flag vector.
    #[allow(clippy::same_item_push)]
    for _ in 0..expected_keys.len() {
        used.push(false);
    }
    for (li, leaf) in cert.leaves.iter().enumerate() {
        if leaf.lo.len() != cert.axes.len() || leaf.hi.len() != cert.axes.len() {
            return Err(BranchError::LeafDimension(li));
        }
        let k = key(&leaf.lo, &leaf.hi)?;
        // Explicit first-match scan (not `BTreeSet::contains`): keeps the
        // membership test in verified code (no absent std-collection
        // obligation). Identical: found iff the key is a grid key.
        let mut pos_opt: Option<usize> = None;
        for (ei, e) in expected_keys.iter().enumerate() {
            if e.as_str() == k.as_str() {
                pos_opt = Some(ei);
                break;
            }
        }
        let Some(pos) = pos_opt else {
            return Err(BranchError::PartitionMismatch(format!(
                "leaf {li} cell {k} is not a product-grid cell"
            )));
        };
        // total: `get_mut` (not `used[pos]`): `pos` indexes `expected_keys`
        // and `used` was built parallel to it (same length), so the `None`
        // arm is unreachable — and fails closed (reject) rather than index.
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
        // (d) compose: global bound is the MIN (Le) / MAX (Ge) over leaves.
        global = Some(match (global, cert.dir) {
            (None, _) => leaf.bound,
            (Some(g), ThreshDir::Le) => {
                if leaf.bound < g {
                    leaf.bound
                } else {
                    g
                }
            }
            (Some(g), ThreshDir::Ge) => {
                if leaf.bound > g {
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
    let cleared = match cert.dir {
        ThreshDir::Le => global > cert.threshold,
        ThreshDir::Ge => global < cert.threshold,
    };
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
fn rat_arr_json(xs: &[Rat]) -> Result<serde_json::Value, RatError> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    for x in xs {
        let s = x.to_clean_string()?;
        out.push(serde_json::Value::String(s));
    }
    Ok(serde_json::Value::Array(out))
}

/// Serialize a branch-tree certificate to a self-describing JSON envelope.
///
/// The envelope records the exact axis partition, the property threshold and
/// direction, the composition rule citation (`branch_split_min`), and — per leaf
/// — the cell corners, the certified `bound`, and every member entailment (as a
/// Clean-canonical `entailment_certificate`). A consumer re-checks it by (a)
/// verifying the partition is the exact product grid, (b) batch-verifying every
/// embedded leaf entailment with Clean's `verify_entailment_certificate`, and
/// (c) taking the min/max of the leaf bounds and comparing to the threshold —
/// which is exactly what [`check_branch_tree`] does, backed by `branch_split_min`.
///
/// # Errors
/// Propagates rational-encoding failures (infallible in practice — full bignum).
pub fn branch_tree_to_json(cert: &BranchTreeCertificate) -> Result<serde_json::Value, RatError> {
    // Explicit loops (not `.map(closure).collect::<Result<Vec<_>, _>>()?`): keeps
    // the per-axis / per-leaf JSON construction in verified code (no
    // absent-adapter `Iterator::map`/`Result::map`/closure-Fn obligations).
    // Identical elements, order, and early-`Err` propagation; `rat_arr_json` is a
    // free `fn` (not a closure) for the same reason.
    let mut axes: Vec<serde_json::Value> = Vec::new();
    for a in &cert.axes {
        let mut edges: Vec<serde_json::Value> = Vec::new();
        for e in &a.edges {
            let s = e.to_clean_string()?;
            edges.push(serde_json::Value::String(s));
        }
        let mut o = serde_json::Map::new();
        o.insert("var".to_owned(), serde_json::Value::String(a.var.clone()));
        o.insert("edges".to_owned(), serde_json::Value::Array(edges));
        axes.push(serde_json::Value::Object(o));
    }

    let mut leaves: Vec<serde_json::Value> = Vec::new();
    for leaf in &cert.leaves {
        // Explicit Vec::new()+push (not `.collect()`): the length is the
        // input-derived member count, so a bulk `.collect()` raises an
        // unbounded-alloc obligation; identical elements, order, and
        // early-`Err` propagation.
        let mut ents = Vec::new();
        for ent in &leaf.member_entailments {
            ents.push(entailment_to_json(ent)?);
        }
        let mut o = serde_json::Map::new();
        o.insert("cell_lo".to_owned(), rat_arr_json(&leaf.lo)?);
        o.insert("cell_hi".to_owned(), rat_arr_json(&leaf.hi)?);
        o.insert(
            "bound".to_owned(),
            serde_json::Value::String(leaf.bound.to_clean_string()?),
        );
        o.insert(
            "member_biases".to_owned(),
            rat_arr_json(&leaf.member_biases)?,
        );
        o.insert(
            "member_entailments".to_owned(),
            serde_json::Value::Array(ents),
        );
        leaves.push(serde_json::Value::Object(o));
    }

    let mut root = serde_json::Map::new();
    root.insert(
        "type".to_owned(),
        serde_json::Value::String("branch_tree_certificate".to_owned()),
    );
    root.insert(
        "version".to_owned(),
        serde_json::Value::String("1.0".to_owned()),
    );
    root.insert(
        "composition_rule".to_owned(),
        serde_json::Value::String("crownproof::branch_split_min".to_owned()),
    );
    root.insert("axes".to_owned(), serde_json::Value::Array(axes));
    root.insert(
        "threshold".to_owned(),
        serde_json::Value::String(cert.threshold.to_clean_string()?),
    );
    root.insert(
        "direction".to_owned(),
        serde_json::Value::String(dir_str(cert.dir).to_owned()),
    );
    root.insert("leaves".to_owned(), serde_json::Value::Array(leaves));
    Ok(serde_json::Value::Object(root))
}

/// Serialize EVERY leaf member entailment as a flat JSON array of tagged
/// `entailment_certificate` objects — the exact input Clean's batch external-cert
/// endpoint (`BatchVerifyExternalCert`) consumes, so all per-cell facts are
/// kernel-re-checked in one call. The branch-tree envelope then only adds the
/// (arithmetically trivial, `branch_split_min`-backed) partition + min step.
///
/// # Errors
/// Propagates rational-encoding failures (infallible in practice).
pub fn branch_tree_leaf_batch_json(
    cert: &BranchTreeCertificate,
) -> Result<serde_json::Value, RatError> {
    let mut items = Vec::new();
    for leaf in &cert.leaves {
        for ent in &leaf.member_entailments {
            items.push(entailment_to_json(ent)?);
        }
    }
    Ok(serde_json::Value::Array(items))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
