// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array, Array1, Array2, Dimension};
use ny_core::{NyError, Result};
use ny_tensor::RepairStrategy;
use std::mem::size_of;
use tracing::{debug, warn};

// ---- NY_SLACK_PROBE (dark, print-only): total accumulated f32 soundness slack ----
//
// Sums, per objective row, the outward-rounding penalty
//   p_i = Σ_j lower_a_err[i,j] · max(|in_l[j]|, |in_u[j]|)
// discharged at EVERY eager fold (`fold_coeff_err_over_box_eager`, called after each
// node's backward at `backward_node_dispatch.rs`). The running per-row total is the
// exact margin-units the f32 soundness rounding subtracts from the certified lower
// bound over the whole backward. This lets us MEASURE whether the ~0.3 cifar100 margin
// gap vs α,β-CROWN is f32 soundness slack (killable by an f64 backward) or fundamental
// relaxation looseness (which f64 cannot touch). Default-OFF and byte-identical: the
// accumulation is guarded by the env read and NEVER mutates a bound, bias, or verdict.
thread_local! {
    static SLACK_ACCUM: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[inline]
pub fn slack_probe_enabled() -> bool {
    matches!(std::env::var("NY_SLACK_PROBE").ok().as_deref(), Some("1"))
}

/// Accumulate the lower-side penalty `p` (margin-units) discharged for objective `row`.
fn slack_probe_add(row: usize, p: f64) {
    SLACK_ACCUM.with(|a| {
        let mut v = a.borrow_mut();
        if row >= v.len() {
            v.resize(row + 1, 0.0);
        }
        v[row] += p;
    });
}

/// Drain and return the accumulated per-row f32 soundness slack (margin-units).
pub fn slack_probe_take() -> Vec<f64> {
    SLACK_ACCUM.with(|a| std::mem::take(&mut *a.borrow_mut()))
}

/// Fast scan for any non-finite (NaN or ±Inf) element in an `f32` array.
///
/// Behaviourally equivalent to `arr.iter().any(|v| !v.is_finite())`, but when
/// the array is in standard (contiguous, C-order) layout — the overwhelmingly
/// common case for freshly-constructed CROWN coefficient matrices — it scans a
/// flat `&[f32]` slice instead of ndarray's strided element iterator. The slice
/// loop has no per-element stride bookkeeping and autovectorizes, which matters
/// because this firewall runs on every `LinearBounds` construction (O(many) per
/// deep-net verification). Non-contiguous arrays fall back to the strided
/// iterator, preserving identical results.
///
/// `f32::is_finite()` returns `false` for both NaN and ±Inf, so this catches the
/// exact same set of values as the original predicate regardless of scan order
/// (`.any()` is order-independent for a pure predicate).
#[inline]
fn any_non_finite<D: Dimension>(arr: &Array<f32, D>) -> bool {
    match arr.as_slice() {
        Some(slice) => slice.iter().any(|v| !v.is_finite()),
        None => arr.iter().any(|v| !v.is_finite()),
    }
}

/// Fast scan for any NaN element in an `f32` array.
///
/// Behaviourally equivalent to `arr.iter().any(|v| v.is_nan())`, with the same
/// contiguous-slice fast path as [`any_non_finite`]. Used for bias vectors,
/// where ±Inf is permitted (conservative bounds) but NaN is not.
#[inline]
fn any_nan<D: Dimension>(arr: &Array<f32, D>) -> bool {
    match arr.as_slice() {
        Some(slice) => slice.iter().any(|v| v.is_nan()),
        None => arr.iter().any(|v| v.is_nan()),
    }
}

/// Linear bounds representation for CROWN-style propagation.
///
/// Represents bounds of the form: lA @ x + lb <= y <= uA @ x + ub
/// where x is the network input and y is some intermediate/output value.
///
/// Shape conventions:
/// - For N outputs and M inputs: lower_a/upper_a are (N, M), lower_b/upper_b are (N,)
///
/// # Validation
///
/// The validated constructor [`LinearBounds::new()`] enforces:
/// - Shape consistency between lower/upper coefficient matrices and biases
/// - Coefficient matrices (lower_a, upper_a) must be finite (no NaN or Inf)
/// - Bias vectors (lower_b, upper_b) must not contain NaN (±Inf is allowed
///   for conservative bounds)
///
/// Known-safe factories ([`identity()`](Self::identity),
/// [`conservative()`](Self::conservative)) bypass validation since their
/// outputs are NaN-free by construction.
///
/// For performance-critical inner loops where inputs are already validated,
/// [`from_parts_unchecked()`](Self::from_parts_unchecked) skips validation
/// (with `debug_assert` in debug builds).
///
/// # Two-Phase Invariant
///
/// During backward propagation, coefficient matrices are accumulated via
/// `safe_add` across multiple paths in a DAG network. This accumulation
/// may produce ±Inf coefficients when many paths converge (e.g., residual
/// connections). **This is expected and sound**: Inf coefficients represent
/// conservative (imprecise but valid) bounds.
///
/// The finiteness invariant from `new()` applies at **construction** time.
/// After construction, mutable accessors (`lower_a_mut()`, etc.) allow
/// modification without re-validation. The invariant is re-established
/// at **concretization** time (`concretize()`, `concretize_box()`), where
/// Inf coefficients produce valid ±Inf concrete bounds.
///
/// Code that consumes intermediate `LinearBounds` between accumulation and
/// concretization must **not** assume `coefficients.iter().all(|v| v.is_finite())`.
/// See: #3032 (safe_add Inf production), #3094 (this documentation).
#[derive(Debug, Clone)]
#[must_use = "LinearBounds from CROWN propagation should not be silently discarded"]
pub struct LinearBounds {
    /// Lower bound coefficient matrix: shape (num_outputs, num_inputs)
    pub(crate) lower_a: Array2<f32>,
    /// Lower bound bias: shape (num_outputs,)
    pub(crate) lower_b: Array1<f32>,
    /// Upper bound coefficient matrix: shape (num_outputs, num_inputs)
    pub(crate) upper_a: Array2<f32>,
    /// Upper bound bias: shape (num_outputs,)
    pub(crate) upper_b: Array1<f32>,
    /// Certified per-coefficient error bound on the *lower* relaxation's stored
    /// coefficients (`#vnncomp-aw-soundness`). Invariant: for every (i, j),
    /// `|lower_a[i,j] - true_coeff| <= lower_a_err[i,j]`, where `true_coeff` is
    /// the exact real-arithmetic coefficient the backward pass intends to store.
    /// All entries are non-negative and finite. `None` means "exact" (error 0),
    /// the default for identity / spec / clip / batched downcast construction
    /// sites whose coefficients carry no accumulated floating-point error.
    ///
    /// This is consumed at [`concretize`](Self::concretize) /
    /// [`concretize_sound`](Self::concretize_sound) via an S-scaled,
    /// `max(|in_l|,|in_u|)`-scaled penalty that is provably sound over the input
    /// box (the corner is no longer chosen by a single possibly-wrong f32 sign).
    /// See `crown_single::aw_f64_with_abssum` for how this is populated by the
    /// linear CROWN-backward `A·W` product accumulated in f64.
    pub(crate) lower_a_err: Option<Array2<f32>>,
    /// Certified per-coefficient error bound on the *upper* relaxation's stored
    /// coefficients. Mirror of [`lower_a_err`](Self::lower_a_err) for `upper_a`.
    pub(crate) upper_a_err: Option<Array2<f32>>,
}

impl LinearBounds {
    /// Create linear bounds with full validation: shapes and NaN rejection.
    ///
    /// Shape invariants:
    /// - `lower_a.shape() == upper_a.shape()`
    /// - `lower_b.len() == lower_a.nrows()`
    /// - `upper_b.len() == lower_a.nrows()`
    ///
    /// NaN/Inf invariants (see §Key Insight in designs/2026-02-25-validated-linear-bounds.md):
    /// - Coefficient matrices (lower_a, upper_a) must be finite (no NaN or Inf).
    ///   An infinite coefficient means the bound is unbounded in proportion to
    ///   input, which is not a valid linear relaxation.
    /// - Bias vectors (lower_b, upper_b) may contain ±Inf (conservative bounds
    ///   from CROWN backward widening) but not NaN.
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if any shape invariant is violated.
    /// Returns `NyError::NumericalInstability` if coefficients contain NaN/Inf
    /// or biases contain NaN.
    pub fn new(
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
    ) -> Result<Self> {
        let bounds = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        };
        bounds.validate_internal_shapes()?;
        bounds.validate_no_nan()?;
        Ok(bounds)
    }

    /// Validate that coefficients are finite and biases are not NaN.
    ///
    /// Coefficients (the "A" in Ax + b) must be finite — an infinite coefficient
    /// means the bound is unbounded in proportion to input, which is not a valid
    /// linear relaxation. Biases may be ±Inf (conservative bounds from CROWN
    /// backward widening) but not NaN.
    ///
    /// Reference: designs/2026-02-25-validated-linear-bounds.md §Step 1
    pub(crate) fn validate_no_nan(&self) -> Result<()> {
        if any_non_finite(&self.lower_a) {
            return Err(NyError::NumericalInstability(
                "LinearBounds lower_a coefficients contain NaN or Inf".into(),
            ));
        }
        if any_non_finite(&self.upper_a) {
            return Err(NyError::NumericalInstability(
                "LinearBounds upper_a coefficients contain NaN or Inf".into(),
            ));
        }
        if any_nan(&self.lower_b) {
            return Err(NyError::NumericalInstability(
                "LinearBounds lower_b bias contains NaN".into(),
            ));
        }
        if any_nan(&self.upper_b) {
            return Err(NyError::NumericalInstability(
                "LinearBounds upper_b bias contains NaN".into(),
            ));
        }
        Ok(())
    }

    /// Create identity linear bounds (output = input).
    ///
    /// Returns bounds where A = I (identity) and b = 0. Known-safe by
    /// construction: identity matrix and zero vector contain no NaN.
    ///
    /// ENSURES: `self.num_inputs() == dim` and `self.num_outputs() == dim`.
    /// ENSURES: `lower_a == I_dim`, `upper_a == I_dim`, and `lower_b == upper_b == 0`.
    pub fn identity(dim: usize) -> Self {
        Self {
            lower_a: Array2::eye(dim),
            lower_b: Array1::zeros(dim),
            upper_a: Array2::eye(dim),
            upper_b: Array1::zeros(dim),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Heap bytes needed for the lower/upper identity coefficient pair.
    pub(crate) fn identity_pair_bytes(dim: usize) -> Option<usize> {
        crate::network::crown_memory::identity_pair_bytes(dim)
    }

    /// Create a subset-identity seed: `rows.len()` rows of the `dim`-wide
    /// identity, one per referenced output index (#margin-subset-seed).
    ///
    /// Row `r` reads exactly output `rows[r]` (coefficient 1.0, bias 0).
    /// Each row is bit-identical in semantics to the corresponding row of
    /// [`Self::identity`]; the backward walk, per-row error terms, and per-row
    /// concretize are all row-local, so a k-row seed computes exactly the k
    /// referenced full-width rows at k/dim of the cost.
    ///
    /// REQUIRES: every index in `rows` is `< dim` (callers obtain the indices
    /// from `output_margin_seed::margin_subset_indices`, which enforces this).
    /// Known-safe by construction: 0/1 coefficients and zero biases contain
    /// no NaN.
    pub fn identity_rows(dim: usize, rows: &[usize]) -> Self {
        let k = rows.len();
        let mut lower_a = Array2::zeros((k, dim));
        for (r, &idx) in rows.iter().enumerate() {
            debug_assert!(idx < dim, "identity_rows index {idx} out of range {dim}");
            lower_a[[r, idx]] = 1.0;
        }
        let upper_a = lower_a.clone();
        Self {
            lower_a,
            lower_b: Array1::zeros(k),
            upper_a,
            upper_b: Array1::zeros(k),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Conservative fallback bounds: coefficients = 0, biases = ±Inf.
    ///
    /// Used when CROWN backward fails or produces numerically unstable results.
    /// The resulting bounds are `(-inf, +inf)` for all outputs — always sound
    /// but maximally imprecise.
    ///
    /// Known-safe by construction: zero coefficients and ±Inf biases contain
    /// no NaN.
    ///
    /// Reference: designs/2026-02-25-validated-linear-bounds.md §Step 2
    pub fn conservative(num_outputs: usize, num_inputs: usize) -> Self {
        Self {
            lower_a: Array2::zeros((num_outputs, num_inputs)),
            lower_b: Array1::from_elem(num_outputs, f32::NEG_INFINITY),
            upper_a: Array2::zeros((num_outputs, num_inputs)),
            upper_b: Array1::from_elem(num_outputs, f32::INFINITY),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Validated construction that falls back to conservative bounds on NaN.
    ///
    /// This is the CROWN backward NaN firewall (Tier 2 of the NaN strategy,
    /// designs/2026-02-25-nan-strategy-unification.md). When CROWN backward
    /// produces NaN/Inf coefficients due to numerical instability, instead of
    /// returning `Err(NumericalInstability)` (which aborts the entire backward
    /// chain), this returns conservative bounds (A=0, b=±∞) that are sound
    /// but maximally imprecise.
    ///
    /// Use this at CROWN backward output points where the caller would
    /// otherwise discard the entire verification attempt on NaN. For
    /// construction sites where NaN should be a hard error (e.g., spec matrix
    /// initialization), use [`new()`](Self::new) instead.
    ///
    /// Reference: #2812 (IBP-vs-CROWN NaN defense asymmetry)
    pub fn new_or_conservative(
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
    ) -> Result<Self> {
        let bounds = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        };
        bounds.validate_internal_shapes()?;
        match bounds.validate_no_nan() {
            Ok(()) => Ok(bounds),
            Err(_) => {
                let (n_out, n_in) = (bounds.lower_a.nrows(), bounds.lower_a.ncols());
                warn!(
                    "LinearBounds NaN firewall: CROWN backward produced non-finite \
                     coefficients ({n_out}x{n_in}), falling back to conservative bounds"
                );
                Ok(Self::conservative(n_out, n_in))
            }
        }
    }

    /// Construct LinearBounds with automatic NaN/Inf repair per strategy.
    ///
    /// Element-wise repair (more precise than `new_or_conservative` which does
    /// all-or-nothing fallback). Part of #3423.
    ///
    /// Coefficient repair: Conservative: NaN/Inf → 0.0; Widen: NaN → 0.0,
    /// Inf kept (two-phase invariant); Strict: error.
    /// Bias repair: NaN → ±Inf (Conservative/Widen), ±Inf kept as-is.
    pub fn new_repaired(
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
        strategy: RepairStrategy,
    ) -> Result<Self> {
        let bounds = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        };
        bounds.validate_internal_shapes()?;
        if matches!(strategy, RepairStrategy::Strict) {
            let (la, lb, ua, ub) = bounds.into_parts();
            return Self::new(la, lb, ua, ub);
        }
        let (lower_a, lower_b, upper_a, upper_b) = bounds.into_parts();
        let mut repair_count = 0usize;
        // Coefficient predicate: Conservative rejects NaN+Inf, Widen rejects NaN only
        let coeff_bad: fn(f32) -> bool = match strategy {
            RepairStrategy::Conservative => |x| !x.is_finite(),
            _ => |x| x.is_nan(),
        };
        let repair = |arr: Array2<f32>, count: &mut usize| {
            arr.mapv(|x| {
                if coeff_bad(x) {
                    *count += 1;
                    0.0
                } else {
                    x
                }
            })
        };
        let lower_a = repair(lower_a, &mut repair_count);
        let upper_a = repair(upper_a, &mut repair_count);
        let lower_b = lower_b.mapv(|x| {
            if x.is_nan() {
                repair_count += 1;
                f32::NEG_INFINITY
            } else {
                x
            }
        });
        let upper_b = upper_b.mapv(|x| {
            if x.is_nan() {
                repair_count += 1;
                f32::INFINITY
            } else {
                x
            }
        });
        if repair_count > 0 {
            debug!(
                repair_count, strategy = ?strategy,
                "LinearBounds::new_repaired: repaired {repair_count} non-finite elements",
            );
        }
        Ok(Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        })
    }

    /// Create linear bounds from a specification matrix for spec-guided CROWN.
    ///
    /// This initializes the CROWN backward pass with a specification matrix `C`
    /// instead of identity. The backward pass will then compute bounds on `C @ y`
    /// directly, preserving correlation information between outputs.
    ///
    /// # Arguments
    /// * `c` - Specification matrix of shape [num_specs, output_dim]. Must be finite.
    ///
    /// # Returns
    /// LinearBounds where A = C and b = 0
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if `c` contains NaN or Inf.
    ///
    /// # Example
    /// For verification property "output_0 > output_1", use C = [[1, -1, 0, ...]]
    /// to compute bounds on output_0 - output_1 directly.
    pub fn from_spec_matrix(c: Array2<f32>) -> Result<Self> {
        if any_non_finite(&c) {
            return Err(NyError::NumericalInstability(
                "LinearBounds::from_spec_matrix: specification matrix contains NaN or Inf".into(),
            ));
        }
        let num_specs = c.nrows();
        Ok(Self {
            lower_a: c.clone(),
            lower_b: Array1::zeros(num_specs),
            upper_a: c,
            upper_b: Array1::zeros(num_specs),
            lower_a_err: None,
            upper_a_err: None,
        })
    }

    /// Number of outputs (rows in coefficient matrix).
    pub fn num_outputs(&self) -> usize {
        self.lower_a.nrows()
    }

    /// Number of inputs (columns in coefficient matrix).
    pub fn num_inputs(&self) -> usize {
        self.lower_a.ncols()
    }

    /// Total heap memory used by this bounds struct, in bytes.
    ///
    /// Includes both A-matrices (coefficient matrices) and bias vectors.
    /// For N outputs and M inputs: `2 * N * M * 4 + 2 * N * 4` bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.lower_a.len() + self.upper_a.len() + self.lower_b.len() + self.upper_b.len())
            * size_of::<f32>()
    }

    // --- Read-only accessors ---

    /// Lower bound coefficient matrix: shape (num_outputs, num_inputs).
    pub fn lower_a(&self) -> &Array2<f32> {
        &self.lower_a
    }

    /// Upper bound coefficient matrix: shape (num_outputs, num_inputs).
    pub fn upper_a(&self) -> &Array2<f32> {
        &self.upper_a
    }

    /// Lower bound bias: shape (num_outputs,).
    pub fn lower_b(&self) -> &Array1<f32> {
        &self.lower_b
    }

    /// Upper bound bias: shape (num_outputs,).
    pub fn upper_b(&self) -> &Array1<f32> {
        &self.upper_b
    }

    // --- Mutable accessors ---
    //
    // WARNING: These return raw `&mut` references with no post-mutation validation.
    // Callers may write ±Inf (from safe_add accumulation) or other non-finite values.
    // This is sound — see "Two-Phase Invariant" in the struct-level docs.
    // Do NOT assume finiteness of coefficients between accumulation and concretization.

    /// Mutable reference to lower bound coefficient matrix.
    ///
    /// # Post-mutation invariant
    /// No validation runs after mutation. Values may include ±Inf after
    /// accumulation (see struct-level "Two-Phase Invariant" docs). NaN is
    /// never sound — callers must ensure no NaN is written.
    pub fn lower_a_mut(&mut self) -> &mut Array2<f32> {
        &mut self.lower_a
    }

    /// Mutable reference to upper bound coefficient matrix.
    ///
    /// # Post-mutation invariant
    /// Same as [`lower_a_mut()`](Self::lower_a_mut).
    pub fn upper_a_mut(&mut self) -> &mut Array2<f32> {
        &mut self.upper_a
    }

    /// Mutable reference to lower bound bias.
    ///
    /// # Post-mutation invariant
    /// ±Inf is allowed (conservative bounds). NaN is never sound.
    pub fn lower_b_mut(&mut self) -> &mut Array1<f32> {
        &mut self.lower_b
    }

    /// Mutable reference to upper bound bias.
    ///
    /// # Post-mutation invariant
    /// Same as [`lower_b_mut()`](Self::lower_b_mut).
    pub fn upper_b_mut(&mut self) -> &mut Array1<f32> {
        &mut self.upper_b
    }

    // --- Certified coefficient-error accessors (#vnncomp-aw-soundness) ---

    /// Certified per-coefficient error on the lower relaxation coefficients.
    ///
    /// `Some(err)` carries `|lower_a[i,j] - true_coeff| <= err[i,j]` (all >= 0,
    /// finite). `None` means exact (error 0). See struct field docs.
    pub(crate) fn lower_a_err(&self) -> Option<&Array2<f32>> {
        self.lower_a_err.as_ref()
    }

    /// Certified per-coefficient error on the upper relaxation coefficients.
    pub(crate) fn upper_a_err(&self) -> Option<&Array2<f32>> {
        self.upper_a_err.as_ref()
    }

    /// Whether this bounds object carries any nonzero certified coefficient error.
    ///
    /// Used by composition steps (ReLU/conv backward) to decide whether the
    /// error must be propagated/degraded rather than silently dropped.
    pub(crate) fn has_coeff_err(&self) -> bool {
        self.lower_a_err.is_some() || self.upper_a_err.is_some()
    }

    /// Attach certified coefficient-error matrices to this bounds object.
    ///
    /// REQUIRES the error matrices to be element-wise `>= 0`, finite, and the
    /// same shape as `lower_a`/`upper_a`. Entries are clamped to non-negative
    /// and any non-finite is treated as conservatively large (handled at
    /// concretize, which degrades the row to `[-inf, +inf]`).
    pub(crate) fn set_coeff_err(&mut self, lower_err: Array2<f32>, upper_err: Array2<f32>) {
        debug_assert_eq!(lower_err.shape(), self.lower_a.shape());
        debug_assert_eq!(upper_err.shape(), self.upper_a.shape());
        self.lower_a_err = Some(lower_err);
        self.upper_a_err = Some(upper_err);
    }

    /// Apply a β-CROWN split contribution to neuron column `neuron_idx`,
    /// folding the f32 rounding of the mutation into the certified coefficient
    /// error (#vnncomp-aw-soundness).
    ///
    /// β-CROWN's split-constraint term subtracts `signed_beta` from every lower
    /// coefficient and adds it to every upper coefficient in the given column:
    /// `lower_a[i,j] := fl32(lower_a[i,j] - signed_beta)` and
    /// `upper_a[i,j] := fl32(upper_a[i,j] + signed_beta)`. Each of these is a
    /// single f32 operation that introduces its own rounding gap. If this bounds
    /// object already carries a certified coefficient error, that gap MUST be
    /// folded into the error or the certificate under-counts the true distance
    /// between the stored f32 coefficient and the exact real coefficient — a
    /// false-proof risk (mirrors the conv f32/err mismatch fixed in becc501).
    ///
    /// For each mutated entry, the new certified error is
    /// `next_up_f32(old_err + |fl32(a ± β) - (a_f64 ± β_f64)|)`, which is sound
    /// by the triangle inequality: `old_err` already bounds
    /// `|a - true_coeff|`, and `signed_beta` is the same exact real subtracted
    /// from both the stored and the true coefficient, so the only new error is
    /// the f32 rounding of the mutation itself.
    ///
    /// `signed_beta` MUST be finite (callers filter non-finite betas upstream).
    /// When this object carries no coefficient error (exact coefficients), the
    /// mutation introduces a single correctly-rounded f32 op whose stored value
    /// is the exact-rounded result; such "freshly accumulated" f32 errors are
    /// certified elsewhere and this method preserves the `None` (exact-marker)
    /// state to avoid changing the err-tracking mode of an exact bounds object.
    pub(crate) fn apply_beta_split_to_column(&mut self, neuron_idx: usize, signed_beta: f32) {
        debug_assert!(signed_beta.is_finite());
        let n_out = self.num_outputs();
        let track_err = self.has_coeff_err();
        let beta64 = signed_beta as f64;
        for i in 0..n_out {
            let lo = self.lower_a[[i, neuron_idx]];
            let hi = self.upper_a[[i, neuron_idx]];
            let new_lo = lo - signed_beta;
            let new_hi = hi + signed_beta;
            if track_err {
                // Rounding gap of the single f32 mutation, measured against the
                // exact f64 result of the same operation (f32->f64 widening is
                // exact, so these f64 references are the true intended values).
                let lo_gap = ((new_lo as f64) - (lo as f64 - beta64)).abs() as f32;
                let hi_gap = ((new_hi as f64) - (hi as f64 + beta64)).abs() as f32;
                if let Some(le) = self.lower_a_err.as_mut() {
                    le[[i, neuron_idx]] = ny_tensor::next_up_f32(le[[i, neuron_idx]] + lo_gap);
                }
                if let Some(ue) = self.upper_a_err.as_mut() {
                    ue[[i, neuron_idx]] = ny_tensor::next_up_f32(ue[[i, neuron_idx]] + hi_gap);
                }
            }
            self.lower_a[[i, neuron_idx]] = new_lo;
            self.upper_a[[i, neuron_idx]] = new_hi;
        }
    }

    /// Inject a multi-neuron group facet's per-variable contribution `+coeff`
    /// into the **lower** backward coefficient column `col` (all output rows),
    /// folding the f32 rounding of each mutation into the certified coefficient
    /// error — the generalization of [`apply_beta_split_to_column`] to an
    /// arbitrary real coefficient on the lower (verification) side only
    /// (`docs/MULTI_NEURON_RELAXATION_DESIGN.md` §2.2).
    ///
    /// The Lagrangian embedding (§2.1) adds `+β_c·(a·x + g·y − b)` to the margin
    /// whose lower bound is computed; for each participating variable this lands
    /// as `+β_c·a_i` on that variable's column. Unlike the β-split (which mutates
    /// both lower `−β` and upper `+β` symmetrically), a facet contribution on the
    /// lower bound only touches `lower_a`. The rounding gap of the single f32
    /// add is folded into `lower_a_err` by the same triangle-inequality argument
    /// as the split (`#vnncomp-aw-soundness`): `old_err` already bounds
    /// `|a − true_coeff|`, and `coeff` is the same exact real added to both the
    /// stored and the true coefficient, so the only new error is the f32 rounding
    /// of the mutation. `coeff` MUST be finite (callers filter non-finite β·a).
    ///
    /// [`apply_beta_split_to_column`]: Self::apply_beta_split_to_column
    ///
    /// Exercised end-to-end by the `multineuron` soundness suite; the deep-net
    /// caller (`apply_group_constraint_contribution`) lands with the increment-2
    /// conv-stack group producer.
    #[allow(dead_code)]
    pub(crate) fn add_to_lower_column(&mut self, col: usize, coeff: f32) {
        debug_assert!(coeff.is_finite());
        if col >= self.lower_a.ncols() {
            return;
        }
        let track_err = self.lower_a_err.is_some();
        let coeff64 = coeff as f64;
        let n_out = self.num_outputs();
        for i in 0..n_out {
            let old = self.lower_a[[i, col]];
            let new = old + coeff;
            if track_err {
                let gap = ((new as f64) - (old as f64 + coeff64)).abs() as f32;
                if let Some(le) = self.lower_a_err.as_mut() {
                    le[[i, col]] = ny_tensor::next_up_f32(le[[i, col]] + gap);
                }
            }
            self.lower_a[[i, col]] = new;
        }
    }

    /// Add a constant `delta` to every output row's **lower** bias, rounded
    /// OUTWARD (`next_down_f32`), for a group facet's `−β_c·b_c` term (§2.2 step
    /// 3). Rounding the added constant DOWN can only lower the lower bound, so it
    /// never overstates it — sound regardless of the sign of `delta`.
    #[allow(dead_code)]
    pub(crate) fn add_lower_bias_outward(&mut self, delta: f32) {
        if delta == 0.0 {
            return;
        }
        for i in 0..self.lower_b.len() {
            self.lower_b[i] =
                ny_tensor::next_down_f32((self.lower_b[i] as f64 + delta as f64) as f32);
        }
    }

    /// Build a "carrier" [`LinearBounds`] whose coefficient matrices ARE this
    /// object's certified error matrices (`|err|`, non-negative) and whose bias
    /// is zero (#vnncomp-aw-soundness).
    ///
    /// This is the mechanism for propagating the certified coefficient error
    /// through an EXACT-linear graph op (Slice, Transpose, Tile, Gather, Concat,
    /// Add/Sub split, Conv1d, ConvTranspose, constant arithmetic, …). Such an op
    /// applies a fixed linear column transform `T` to the coefficient matrix:
    /// `A_out = T(A_in)` (plus an `A→bias` fold for the constant-shift ops). For
    /// a perturbed coefficient `A_in + δ` with `|δ| ≤ err_in`, the output error is
    /// bounded by `T_abs(err_in)`, where `T_abs` replaces every transform entry by
    /// its absolute value. Running the op's own `propagate_linear` transform on
    /// this all-non-negative carrier yields exactly that magnitude bound in the
    /// result's coefficients, and the result's bias magnitude bounds the error
    /// folded into the bias by the op (e.g. `err_in @ |c|` for AddConstant). The
    /// caller takes `|coeff|` of the carried result as the new coefficient error
    /// and `max(|lower_b|,|upper_b|)` as the per-row bias-error widening.
    ///
    /// Returns `None` when there is no error to carry (the op proceeds exactly).
    pub(crate) fn coeff_err_carrier(&self) -> Option<LinearBounds> {
        if !self.has_coeff_err() {
            return None;
        }
        let n = self.lower_a.nrows();
        let lower_carrier = self
            .lower_a_err
            .as_ref()
            .map(|e| e.mapv(|v| v.abs()))
            .unwrap_or_else(|| Array2::zeros(self.lower_a.raw_dim()));
        let upper_carrier = self
            .upper_a_err
            .as_ref()
            .map(|e| e.mapv(|v| v.abs()))
            .unwrap_or_else(|| Array2::zeros(self.upper_a.raw_dim()));
        Some(LinearBounds {
            lower_a: lower_carrier,
            lower_b: Array1::zeros(n),
            upper_a: upper_carrier,
            upper_b: Array1::zeros(n),
            lower_a_err: None,
            upper_a_err: None,
        })
    }

    /// Attach the certified error derived from running an exact-linear op on a
    /// [`coeff_err_carrier`](Self::coeff_err_carrier) (#vnncomp-aw-soundness).
    ///
    /// `carried` is the `LinearBounds` produced by feeding this object's error
    /// carrier through the same op transform that produced `self`'s coefficients.
    /// Its coefficient magnitudes are the new per-coefficient error (`|coeff|`,
    /// rounded UP to a sound f32), and its bias magnitudes are folded into `self`'s
    /// bias OUTWARD (lower decreases, upper increases) to cover any error the op
    /// moved from coefficients into the bias. Shapes must match `self`.
    pub(crate) fn attach_err_from_carried(&mut self, carried: &LinearBounds) {
        if carried.lower_a.shape() != self.lower_a.shape()
            || carried.upper_a.shape() != self.upper_a.shape()
            || carried.lower_b.len() != self.lower_b.len()
            || carried.upper_b.len() != self.upper_b.len()
        {
            // Shape mismatch: cannot soundly map the carried error; degrade this
            // object to conservative so the dropped penalty never reads as tight.
            let (n_out, n_in) = (self.lower_a.nrows(), self.lower_a.ncols());
            *self = Self::conservative(n_out, n_in);
            return;
        }
        let san = |v: f32| {
            let a = v.abs();
            if a.is_finite() {
                ny_tensor::next_up_f32(a)
            } else {
                f32::INFINITY
            }
        };
        self.lower_a_err = Some(carried.lower_a.mapv(san));
        self.upper_a_err = Some(carried.upper_a.mapv(san));
        // Fold the carried bias magnitude OUTWARD into self's bias. The op's
        // transform may have produced a directed [lower,upper] interval on the
        // carrier bias; max(|lower|,|upper|) bounds the magnitude either way.
        for i in 0..self.lower_b.len() {
            let mag = (carried.lower_b[i].abs()).max(carried.upper_b[i].abs());
            if mag != 0.0 && mag.is_finite() {
                self.lower_b[i] = ny_tensor::next_down_f32(self.lower_b[i] - mag);
                self.upper_b[i] = ny_tensor::next_up_f32(self.upper_b[i] + mag);
            } else if !mag.is_finite() {
                self.lower_b[i] = f32::NEG_INFINITY;
                self.upper_b[i] = f32::INFINITY;
            }
        }
    }

    /// Soundly discharge any certified coefficient error by folding it into the
    /// BIAS over a known input box, then clearing the error (#vnncomp-aw-soundness).
    ///
    /// This is the precise (tight) alternative to
    /// [`discharge_coeff_err_to_conservative`](Self::discharge_coeff_err_to_conservative):
    /// rather than degrading the whole affected row to `[-inf, +inf]`, it absorbs
    /// the certified coefficient uncertainty as a concrete bias widening evaluated
    /// against the node's input box `[in_l, in_u]`. For row `i`, the worst-case
    /// contribution of the coefficient interval `[A-err, A+err]` over the box is
    /// `penalty_i = Σ_j max(|in_l_j|, |in_u_j|)·err_ij` — exactly the term
    /// `concretize_sound` would apply — so subtracting it from `lower_b` and adding
    /// it to `upper_b` produces error-free bounds that remain sound when any
    /// downstream layer further transforms them. `in_l`/`in_u` are flattened to
    /// match the coefficient columns; a length mismatch falls back to the
    /// conservative row degrade.
    ///
    /// Returns silently if there is no error to discharge.
    pub(crate) fn fold_coeff_err_into_bias(&mut self, in_l: &[f32], in_u: &[f32]) {
        if !self.has_coeff_err() {
            return;
        }
        let n_in = self.lower_a.ncols();
        if in_l.len() != n_in || in_u.len() != n_in {
            // Cannot map the box onto the columns; fall back to the conservative
            // row degrade (still sound, just looser).
            self.discharge_coeff_err_to_conservative();
            return;
        }
        let n_out = self.lower_a.nrows();
        // Worst-case input magnitude per column, in f64.
        let mut mag = vec![0.0f64; n_in];
        for j in 0..n_in {
            mag[j] = (in_l[j] as f64).abs().max((in_u[j] as f64).abs());
        }
        if let Some(le) = self.lower_a_err.take() {
            for i in 0..n_out {
                let mut p = 0.0f64;
                for j in 0..n_in {
                    p += le[[i, j]] as f64 * mag[j];
                }
                if p != 0.0 {
                    if p.is_finite() {
                        self.lower_b[i] =
                            ny_tensor::next_down_f32((self.lower_b[i] as f64 - p) as f32);
                    } else {
                        self.lower_b[i] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        if let Some(ue) = self.upper_a_err.take() {
            for i in 0..n_out {
                let mut p = 0.0f64;
                for j in 0..n_in {
                    p += ue[[i, j]] as f64 * mag[j];
                }
                if p != 0.0 {
                    if p.is_finite() {
                        self.upper_b[i] =
                            ny_tensor::next_up_f32((self.upper_b[i] as f64 + p) as f32);
                    } else {
                        self.upper_b[i] = f32::INFINITY;
                    }
                }
            }
        }
    }

    /// EAGERLY discharge the certified coefficient error over a box, per row,
    /// keeping rows whose penalty is non-finite carried (#cgan-conv-err-compose).
    ///
    /// Same fold identity as [`fold_coeff_err_into_bias`](Self::fold_coeff_err_into_bias):
    /// for true coefficients `Ã ∈ [A−E, A+E]` over `y ∈ [in_l, in_u]` (the box the
    /// columns multiply), `Ã·y ∈ A·y ± Σ_j E_ij·max(|in_l_j|,|in_u_j|)`, so folding
    /// the penalty outward into the bias preserves the enclosure for every `Ã`.
    ///
    /// The difference is the POLICY: this variant is called right after an
    /// activation backward step, where the columns multiply the activation's
    /// (typically CROWN-tightened) PRE-ACTIVATION cut — the tightest box the
    /// error will ever see. Carrying the error further discharges it either at
    /// the network input after ABS-composition through every remaining layer
    /// (`T_abs(E)·|x|`, which equals `E` against the forward IBP-abs magnitudes
    /// — IBP-scale, exponentially loose on deep conv stacks) or at some later
    /// discharge box of IBP scale. The fold penalty here is `≤ u·Σ|A|·mag(box)`,
    /// which is `~2^-24` RELATIVE to the activation relaxation's own intercept
    /// mass over the very same box — always negligible where CROWN is useful at
    /// all. Measured on cgan_2023 (nCh_1 prop_1): carried-to-input penalties of
    /// 1.4e-2 (Conv_22 target, 18% of oracle width) drop to ~1e-5.
    ///
    /// Rows whose penalty is NON-FINITE (unbounded box or ∞-poisoned error) keep
    /// their error entries and continue to carry — byte-identical to the prior
    /// behavior for those rows, never a new degrade.
    pub(crate) fn fold_coeff_err_over_box_eager(&mut self, input_box: &ny_tensor::BoundedTensor) {
        if !self.has_coeff_err() {
            return;
        }
        let n_in = self.lower_a.ncols();
        let flat = input_box.flatten();
        let (Some(in_l), Some(in_u)) = (flat.lower().as_slice(), flat.upper().as_slice()) else {
            return; // non-contiguous: keep carrying (sound, prior behavior)
        };
        if in_l.len() != n_in || in_u.len() != n_in {
            return; // cannot map the box onto the columns: keep carrying
        }
        let n_out = self.lower_a.nrows();
        let mut mag = vec![0.0f64; n_in];
        for j in 0..n_in {
            mag[j] = (in_l[j] as f64).abs().max((in_u[j] as f64).abs());
        }
        let probe_on = slack_probe_enabled();
        let fold_side =
            |err: &mut Option<Array2<f32>>, bias: &mut Array1<f32>, lower_side: bool| {
                let Some(e) = err.as_mut() else { return };
                let mut any_kept = false;
                for i in 0..n_out {
                    let mut p = 0.0f64;
                    for j in 0..n_in {
                        p += e[[i, j]] as f64 * mag[j];
                    }
                    if p.is_finite() {
                        if p != 0.0 {
                            if lower_side {
                                // NY_SLACK_PROBE: record the margin-units this fold
                                // subtracts from objective row `i`'s lower bound.
                                if probe_on {
                                    slack_probe_add(i, p);
                                }
                                bias[i] = ny_tensor::next_down_f32((bias[i] as f64 - p) as f32);
                            } else {
                                bias[i] = ny_tensor::next_up_f32((bias[i] as f64 + p) as f32);
                            }
                        }
                        for j in 0..n_in {
                            e[[i, j]] = 0.0;
                        }
                    } else {
                        any_kept = true;
                    }
                }
                if !any_kept {
                    *err = None;
                }
            };
        let (lower_a_err, upper_a_err) = (&mut self.lower_a_err, &mut self.upper_a_err);
        fold_side(lower_a_err, &mut self.lower_b, true);
        fold_side(upper_a_err, &mut self.upper_b, false);
    }

    /// Soundly discharge any certified coefficient error by degrading every
    /// affected row to a conservative `[-inf, +inf]` bound, then clearing the
    /// error (#vnncomp-aw-soundness).
    ///
    /// Used by the CROWN backward dispatcher right before handing an
    /// error-carrying bounds object to a layer that does NOT propagate the
    /// error: for each row with any nonzero `lower_a_err` (resp. `upper_a_err`),
    /// the row's lower (resp. upper) coefficients are zeroed and the bias set to
    /// `-inf` (resp. `+inf`). This is the same maximally-loose-but-sound fallback
    /// used for non-finite coefficient rows, and it lets the downstream layer
    /// proceed on plain (error-free) bounds without ever silently dropping the
    /// penalty. Rows with zero error are left fully precise.
    pub(crate) fn discharge_coeff_err_to_conservative(&mut self) {
        if !self.has_coeff_err() {
            return;
        }
        let (n_out, n_in) = (self.lower_a.nrows(), self.lower_a.ncols());
        if let Some(le) = self.lower_a_err.take() {
            for i in 0..n_out {
                let row_has_err = (0..n_in).any(|j| le[[i, j]] != 0.0);
                if row_has_err {
                    for j in 0..n_in {
                        self.lower_a[[i, j]] = 0.0;
                    }
                    self.lower_b[i] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(ue) = self.upper_a_err.take() {
            for i in 0..n_out {
                let row_has_err = (0..n_in).any(|j| ue[[i, j]] != 0.0);
                if row_has_err {
                    for j in 0..n_in {
                        self.upper_a[[i, j]] = 0.0;
                    }
                    self.upper_b[i] = f32::INFINITY;
                }
            }
        }
    }

    // --- Destructuring ---

    /// Consume self and return the four components.
    ///
    /// Returns `(lower_a, lower_b, upper_a, upper_b)`.
    ///
    /// NOTE: this DROPS any certified coefficient-error matrices
    /// (#vnncomp-aw-soundness). On the default CROWN / α-CROWN verdict path the
    /// error is consumed at `concretize_sound` BEFORE any `into_parts`, and the
    /// sequential/graph backward dispatch degrades to IBP rather than routing an
    /// error-carrying bounds object through a layer that would drop it. Callers
    /// that decompose bounds for non-verdict bookkeeping (batched-domain picked
    /// conversion, OpaqueSkip rebias) use this and lose only a precision penalty
    /// for bounds that do not reach a Verified verdict through this object.
    pub fn into_parts(self) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
        (self.lower_a, self.lower_b, self.upper_a, self.upper_b)
    }

    // --- Factory methods ---

    /// Unchecked construction for performance-critical inner loops where
    /// inputs are already validated upstream.
    ///
    /// In debug builds, asserts that coefficients are finite and biases
    /// are not NaN (matching `validate_no_nan()` contract).
    /// In release builds, skips all validation.
    ///
    /// # Safety (logical)
    /// Caller must ensure:
    /// - Shapes are consistent (lower_a.shape() == upper_a.shape(), etc.)
    /// - Coefficients are finite (no NaN or Inf)
    /// - Biases do not contain NaN
    ///
    /// Tracked: grep for `from_parts_unchecked` to audit usage sites.
    /// Target: < 20 production uses. See designs/2026-02-25-validated-linear-bounds.md §Step 2.
    pub(crate) fn from_parts_unchecked(
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
    ) -> Self {
        // Coefficients: must be finite (no NaN, no Inf) — matches validate_no_nan()
        debug_assert!(
            !any_non_finite(&lower_a),
            "from_parts_unchecked: lower_a contains NaN or Inf"
        );
        debug_assert!(
            !any_non_finite(&upper_a),
            "from_parts_unchecked: upper_a contains NaN or Inf"
        );
        // Biases: must not be NaN (±Inf allowed for conservative bounds)
        debug_assert!(
            !any_nan(&lower_b),
            "from_parts_unchecked: lower_b contains NaN"
        );
        debug_assert!(
            !any_nan(&upper_b),
            "from_parts_unchecked: upper_b contains NaN"
        );
        Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Construct a `LinearBounds` from parts plus certified coefficient error.
    ///
    /// Internal constructor used by the linear CROWN-backward step to carry the
    /// certified `A·W` rounding/propagation error to concretize. Performs the
    /// NaN firewall on coefficients/biases like `new_or_conservative`; the error
    /// matrices are sanitized to non-negative finite values (non-finite or NaN
    /// error entries are mapped to `f32::INFINITY`, which makes concretize widen
    /// that row to `[-inf, +inf]` — sound).
    pub(crate) fn new_or_conservative_with_err(
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
        lower_a_err: Array2<f32>,
        upper_a_err: Array2<f32>,
    ) -> Result<Self> {
        let mut base = Self::new_or_conservative(lower_a, lower_b, upper_a, upper_b)?;
        // If the firewall degraded to conservative bounds (coeffs zeroed), the
        // error matrices may no longer match shape; only attach when shapes line up.
        let sanitize = |e: Array2<f32>| {
            e.mapv(|v| {
                if v.is_finite() && v >= 0.0 {
                    v
                } else {
                    f32::INFINITY
                }
            })
        };
        if lower_a_err.shape() == base.lower_a.shape()
            && upper_a_err.shape() == base.upper_a.shape()
        {
            base.lower_a_err = Some(sanitize(lower_a_err));
            base.upper_a_err = Some(sanitize(upper_a_err));
        }
        Ok(base)
    }

    /// Create symmetric linear bounds where lower == upper.
    ///
    /// Useful for layers that produce exact (non-relaxed) linear transforms,
    /// such as affine layers and convolutions.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if coefficients contain NaN/Inf
    /// or biases contain NaN.
    pub fn symmetric(a: Array2<f32>, b: Array1<f32>) -> Result<Self> {
        Self::new(a.clone(), b.clone(), a, b)
    }

    /// Create linear bounds from coefficient matrices with zero biases.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if coefficients contain NaN/Inf.
    pub fn from_coefficients(lower_a: Array2<f32>, upper_a: Array2<f32>) -> Result<Self> {
        let n = lower_a.nrows();
        Self::new(lower_a, Array1::zeros(n), upper_a, Array1::zeros(n))
    }
}
// Tests moved to bounds/tests/linear.rs
