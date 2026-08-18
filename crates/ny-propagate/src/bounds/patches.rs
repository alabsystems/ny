// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode linear bounds for CNN-optimized CROWN backward propagation.
//!
//! Instead of materializing a full dense [out_dim x in_dim] A-matrix,
//! Patches stores receptive field coefficients per output position:
//! O(out_c * out_h * out_w * in_c * kH * kW) instead of
//! O((out_c * out_h * out_w)^2).
//!
//! Reference: alpha-beta-CROWN `auto_LiRPA/patches.py` (Patches class)
//! Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::mem::size_of;
use std::time::Instant;

use super::LinearBounds;
use crate::execution_telemetry::PatchesMaterializationMemoryReceipt;
pub(crate) use crate::execution_telemetry::PatchesMaterializationPurpose;

mod crown_bounds;
mod eager_err;
mod merge;
mod scatter;
mod selected_columns;
mod sparse_concretize;
mod to_dense;
mod types;

pub(crate) use crown_bounds::CrownBounds;
pub(crate) use eager_err::eager_err_enabled;
/// #eager-err-test-override: test-only, so the equivalence moats can pin the
/// UNFOLDED kernel without widening `eager_err`'s visibility in normal builds.
#[cfg(test)]
pub(crate) use eager_err::test_override;
pub(crate) use types::{PatchGeometry, PatchesData, UnstableIdx};

/// Request-local cooperative deadline state for Patches materialization.
///
/// `None` is a zero-clock-read fast path used by the historical APIs. A finite
/// deadline is checked at explicit allocation/publication boundaries and after
/// bounded units of fill/scatter work. The test-only forced stage is local to
/// one call and cannot alter production or ordinary no-deadline behavior.
pub(crate) struct PatchesMaterializationDeadline {
    deadline: Option<Instant>,
    work_since_check: usize,
    #[cfg(test)]
    forced_stage: Option<&'static str>,
}

/// One request-local admission receipt for Patches proof allocations.
///
/// The CROWN dense budget is interpreted here as a ceiling on the request's
/// total live logical payload, not as an incremental allowance for each new
/// `Vec`. Callers therefore include the borrowed source
/// [`PatchesLinearBounds`], any operation-owned buffers retained across a
/// nested admission (for example chunked concretization outputs), and the
/// operation-local peak in `nominal_required_bytes`. Lower/upper carriers can
/// share anchored-axis Arcs, but [`PatchesLinearBounds::memory_bytes`]
/// deliberately charges both: that conservative double count is preferable to
/// admitting a process-OOM peak.
///
/// A shared ndarray exposes its logical length but not its backing `Vec`
/// capacity. Consequently pre-existing Patches/LinearBounds carriers are
/// charged by `len * size_of::<T>()`; this receipt does not claim to census
/// allocator slack retained before the request. The process memory envelope
/// caps the adaptive dense budget at one eighth of live cgroup/RLIMIT headroom,
/// leaving the other seven eighths for that unobservable slack, graph state,
/// and allocator overhead. Tracking retained capacity exactly would require a
/// capacity field on every array-producing schema, including external ndarray
/// constructors, rather than a truthful local materialization seam.
///
/// `Vec::try_reserve_exact` is permitted to return more capacity than was
/// requested. Every fallible allocator reconciles that overage into this same
/// receipt before touching the buffer, so independently rounded allocations
/// cannot each consume the full budget.
struct PatchesMemoryAdmission {
    nominal_required_bytes: usize,
    capacity_overage_bytes: usize,
    budget_bytes: usize,
}

impl PatchesMemoryAdmission {
    fn check(nominal_required_bytes: usize, site: &'static str) -> Result<Self> {
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        Self::check_with_budget(nominal_required_bytes, budget_bytes, site)
    }

    fn check_with_budget(
        nominal_required_bytes: usize,
        budget_bytes: usize,
        site: &'static str,
    ) -> Result<Self> {
        if nominal_required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: nominal_required_bytes,
                budget_bytes,
                site,
            });
        }
        Ok(Self {
            nominal_required_bytes,
            capacity_overage_bytes: 0,
            budget_bytes,
        })
    }

    #[inline]
    fn required_bytes(&self) -> usize {
        self.nominal_required_bytes
            .saturating_add(self.capacity_overage_bytes)
    }

    #[inline]
    fn capacity_overage_bytes(&self) -> usize {
        self.capacity_overage_bytes
    }

    #[inline]
    fn receipt(&self) -> PatchesMaterializationMemoryReceipt {
        PatchesMaterializationMemoryReceipt {
            nominal_required_bytes: self.nominal_required_bytes,
            capacity_overage_bytes: self.capacity_overage_bytes,
            admitted_bytes: self.required_bytes(),
            budget_bytes: self.budget_bytes,
        }
    }

    fn allocation_error(&self, site: &'static str) -> NyError {
        NyError::CpuMemoryExceeded {
            required_bytes: self.required_bytes(),
            budget_bytes: self.budget_bytes,
            site,
        }
    }

    /// Reconcile one newly reserved `Vec` against the capacity budgeted for it.
    /// This is called immediately after reserve and before resize/push/copy.
    fn reconcile_vec_capacity<T>(
        &mut self,
        requested_elements: usize,
        actual_capacity: usize,
        site: &'static str,
    ) -> Result<()> {
        let requested_bytes = requested_elements.saturating_mul(size_of::<T>());
        let actual_bytes = actual_capacity.saturating_mul(size_of::<T>());
        self.capacity_overage_bytes = self
            .capacity_overage_bytes
            .saturating_add(actual_bytes.saturating_sub(requested_bytes));
        if self.required_bytes() > self.budget_bytes {
            return Err(self.allocation_error(site));
        }
        Ok(())
    }
}

#[cfg(test)]
mod materialization_admission_tests {
    use super::{PatchesLinearBounds, PatchesMemoryAdmission};
    use ny_core::NyError;
    use std::{mem::size_of, time::Duration};

    #[test]
    fn total_live_receipt_accepts_exact_budget_and_refuses_budget_minus_one() {
        let required = 137usize;
        assert!(PatchesMemoryAdmission::check_with_budget(
            required,
            required,
            "exact receipt test"
        )
        .is_ok());
        assert!(matches!(
            PatchesMemoryAdmission::check_with_budget(
                required,
                required - 1,
                "budget-minus-one receipt test"
            ),
            Err(NyError::CpuMemoryExceeded {
                required_bytes: 137,
                budget_bytes: 136,
                site: "budget-minus-one receipt test",
            })
        ));

        let mut exact =
            PatchesMemoryAdmission::check_with_budget(100, 108, "capacity receipt test").unwrap();
        exact
            .reconcile_vec_capacity::<u8>(10, 18, "capacity receipt test")
            .unwrap();
        assert_eq!(exact.required_bytes(), 108);

        let mut one_short =
            PatchesMemoryAdmission::check_with_budget(100, 107, "capacity receipt test").unwrap();
        assert!(matches!(
            one_short.reconcile_vec_capacity::<u8>(10, 18, "capacity receipt test"),
            Err(NyError::CpuMemoryExceeded {
                required_bytes: 108,
                budget_bytes: 107,
                site: "capacity receipt test",
            })
        ));
    }

    fn assert_same_virtual_identity(actual: &PatchesLinearBounds, expected: &PatchesLinearBounds) {
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_eq!(actual.upper_b, expected.upper_b);
        for (actual, expected) in [
            (&actual.lower_a, &expected.lower_a),
            (&actual.upper_a, &expected.upper_a),
        ] {
            assert_eq!(actual.patches, expected.patches);
            assert_eq!(actual.geometry, expected.geometry);
            assert_eq!(actual.identity, expected.identity);
            assert_eq!(actual.output_shape, expected.output_shape);
            assert_eq!(actual.input_shape, expected.input_shape);
            assert!(actual.unstable_idx.is_none());
            assert!(actual.coeff_err.is_none());
        }
    }

    #[test]
    fn finite_identity_matches_legacy_at_exact_total_live_budget() {
        let shape = (2, 3, 4);
        let retained = 19usize;
        let required = retained + 2 * 24 * size_of::<f32>();
        let expected = PatchesLinearBounds::identity(shape, shape);
        let actual = PatchesLinearBounds::try_identity_with_deadline_and_budget(
            shape,
            shape,
            Some(std::time::Instant::now() + Duration::from_mins(1)),
            retained,
            required,
        )
        .expect("exact total-live budget admits the finite identity");
        assert_same_virtual_identity(&actual, &expected);

        let legacy = PatchesLinearBounds::try_identity_with_deadline(shape, shape, None, 0)
            .expect("no-deadline identity remains infallible");
        assert_same_virtual_identity(&legacy, &expected);
    }

    #[test]
    fn finite_identity_expired_deadline_is_typed_and_atomic() {
        let error = PatchesLinearBounds::try_identity_with_deadline_and_budget(
            (1, 2, 2),
            (1, 2, 2),
            Some(std::time::Instant::now()),
            0,
            usize::MAX,
        )
        .expect_err("expired authority must refuse before allocation");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
    }

    #[test]
    fn finite_identity_refuses_budget_minus_one_before_allocation() {
        let shape = (2, 3, 4);
        let retained = 19usize;
        let required = retained + 2 * 24 * size_of::<f32>();
        assert!(matches!(
            PatchesLinearBounds::try_identity_with_deadline_and_budget(
                shape,
                shape,
                Some(std::time::Instant::now() + Duration::from_mins(1)),
                retained,
                required - 1,
            ),
            Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "PatchesLinearBounds::try_identity_with_deadline",
            }) if required_bytes == required && budget_bytes == required - 1
        ));
    }
}

impl PatchesMaterializationDeadline {
    /// Maximum touch-heavy work between finite-deadline clock reads. Individual
    /// opaque operations (allocator calls) are bracketed by explicit checks.
    pub(super) const CHECK_STRIDE: usize = 4096;

    pub(crate) const fn new(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            work_since_check: 0,
            #[cfg(test)]
            forced_stage: None,
        }
    }

    #[inline]
    fn active(&self) -> bool {
        #[cfg(test)]
        let forced = self.forced_stage.is_some();
        #[cfg(not(test))]
        let forced = false;
        self.deadline.is_some() || forced
    }

    /// Check immediately at a named atomicity/resource boundary.
    #[inline]
    pub(crate) fn checkpoint(&mut self, stage: &'static str) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        #[cfg(test)]
        if self.forced_stage == Some(stage) {
            return Err(NyError::DeadlineExceeded(format!(
                "patches materialization: forced test deadline {stage}"
            )));
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(NyError::DeadlineExceeded(format!(
                "patches materialization: deadline exceeded {stage}"
            )));
        }
        Ok(())
    }

    /// Record touch-heavy work, checking after at most `CHECK_STRIDE` units.
    #[inline]
    pub(crate) fn work(&mut self, units: usize, stage: &'static str) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        #[cfg(test)]
        if self.forced_stage == Some(stage) {
            return self.checkpoint(stage);
        }
        self.work_since_check = self.work_since_check.saturating_add(units);
        if self.work_since_check >= Self::CHECK_STRIDE {
            self.work_since_check %= Self::CHECK_STRIDE;
            self.checkpoint(stage)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) const fn forced_at(stage: &'static str) -> Self {
        Self {
            deadline: None,
            work_since_check: 0,
            forced_stage: Some(stage),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
thread_local! {
    /// Per-thread patches→dense call-site recorder.
    ///
    /// This was previously a process-global `Mutex<Vec<String>>`, but cargo runs
    /// the test binary multi-threaded and `to_dense()` records on *every* call in
    /// `#[cfg(test)]` builds. Any other test that triggered a patches→dense
    /// conversion concurrently appended to the shared buffer and corrupted the
    /// reset→propagate→read window of a test that observes the recorder (#4138:
    /// passed in isolation, failed in the full parallel suite).
    ///
    /// Each test runs on its own thread and the CROWN propagation it exercises is
    /// synchronous on that thread, so a thread-local buffer captures exactly the
    /// observing test's own conversions and is immune to concurrent tests.
    static PATCHES_TO_DENSE_CALL_SITES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn reset_patches_to_dense_call_count() {
    PATCHES_TO_DENSE_CALL_SITES.with(|sites| sites.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn patches_to_dense_call_sites() -> Vec<String> {
    PATCHES_TO_DENSE_CALL_SITES.with(|sites| sites.borrow().clone())
}

#[cfg(test)]
pub(crate) fn record_patches_to_dense_call_site(site: String) {
    PATCHES_TO_DENSE_CALL_SITES.with(|sites| sites.borrow_mut().push(site));
}

/// Patches-mode linear bounds for CROWN backward propagation.
///
/// Analogous to LinearBounds but with structured sparse A-matrices.
/// The bias vectors remain dense Array1<f32> since they're per-output-neuron.
#[derive(Debug, Clone)]
pub(crate) struct PatchesLinearBounds {
    /// Logical number of Dense rows represented by this Patches object.
    ///
    /// Legacy spatial-output Patches use one logical row per output position
    /// (`out_c * out_h * out_w`). Dense->Patches re-entry uses arbitrary spec
    /// rows over the same spatial output grid, so the row count must be tracked
    /// explicitly instead of being inferred from `output_shape`.
    pub(crate) row_count: usize,
    pub(crate) lower_a: PatchesData,
    pub(crate) lower_b: Array1<f32>,
    pub(crate) upper_a: PatchesData,
    pub(crate) upper_b: Array1<f32>,
}

impl PatchesLinearBounds {
    /// Create identity Patches bounds for starting CROWN backward.
    ///
    /// The A-matrices are identity (each output position maps to itself).
    /// This is the Patches equivalent of `LinearBounds::identity(out_dim)`.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md
    pub(crate) fn identity(
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
    ) -> Self {
        let out_dim = output_shape.0 * output_shape.1 * output_shape.2;
        PatchesLinearBounds {
            row_count: out_dim,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None,
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: None,
            },
            lower_b: Array1::zeros(out_dim),
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: None,
            },
            upper_b: Array1::zeros(out_dim),
        }
    }

    /// Fallibly construct a virtual identity under one absolute deadline.
    ///
    /// `None` deliberately delegates to [`Self::identity`] so the historical
    /// allocation and exact result remain unchanged. A finite request validates
    /// both shape products, charges the caller-retained payload plus the two
    /// result bias vectors, reconciles actual `Vec` capacity, and polls every
    /// bounded unit of initialization before publishing the complete carrier.
    pub(crate) fn try_identity_with_deadline(
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<Self> {
        if deadline.is_none() {
            return Ok(Self::identity(output_shape, input_shape));
        }
        Self::try_identity_with_deadline_and_budget(
            output_shape,
            input_shape,
            deadline,
            retained_base_bytes,
            crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
        )
    }

    fn try_identity_with_deadline_and_budget(
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
        deadline: Option<Instant>,
        retained_base_bytes: usize,
        budget_bytes: usize,
    ) -> Result<Self> {
        const SITE: &str = "PatchesLinearBounds::try_identity_with_deadline";
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        deadline.checkpoint(SITE)?;
        let out_dim = checked_shape_product(&[output_shape.0, output_shape.1, output_shape.2])
            .ok_or_else(|| NyError::InvalidSpec("Patches identity output shape overflow".into()))?;
        checked_shape_product(&[input_shape.0, input_shape.1, input_shape.2])
            .ok_or_else(|| NyError::InvalidSpec("Patches identity input shape overflow".into()))?;
        let bias_pair_bytes = out_dim
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or_else(|| NyError::InvalidSpec("Patches identity bias size overflow".into()))?;
        let nominal_required_bytes = retained_base_bytes.saturating_add(bias_pair_bytes);
        let mut admission =
            PatchesMemoryAdmission::check_with_budget(nominal_required_bytes, budget_bytes, SITE)?;

        let (lower_b, upper_b) = {
            let mut make_bias = || -> Result<Array1<f32>> {
                deadline.checkpoint("before Patches identity bias allocation")?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(out_dim)
                    .map_err(|_| admission.allocation_error(SITE))?;
                admission.reconcile_vec_capacity::<f32>(out_dim, values.capacity(), SITE)?;
                deadline.checkpoint("after Patches identity bias allocation")?;
                while values.len() < out_dim {
                    let previous_len = values.len();
                    let next_len = previous_len
                        .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE)
                        .min(out_dim);
                    values.resize(next_len, 0.0);
                    deadline.work(
                        next_len - previous_len,
                        "during Patches identity bias initialization",
                    )?;
                }
                deadline.checkpoint("after Patches identity bias initialization")?;
                Ok(Array1::from_vec(values))
            };
            (make_bias()?, make_bias()?)
        };
        deadline.checkpoint("before Patches identity publication")?;

        Ok(PatchesLinearBounds {
            row_count: out_dim,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None,
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: None,
            },
            lower_b,
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: None,
            },
            upper_b,
        })
    }

    /// Create sparse identity Patches bounds tracking only unstable neurons.
    ///
    /// Like `identity()`, but only creates patches for the specified unstable
    /// output positions. The patches tensor is 4D `(unstable_size, in_c, 1, 1)`
    /// instead of 6D. Bias vectors have length `unstable_size`.
    ///
    /// This is the starting point for sparse CROWN backward: only computing
    /// bounds for neurons that are actually unstable (lower < 0 < upper).
    ///
    /// Reference: alpha-beta-CROWN `backward_bound.py` `get_sparse_C` + `Patches(identity=1, unstable_idx=...)`
    /// Part of #2613 Phase 4 step 19
    pub(crate) fn sparse_identity(
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
        unstable_idx: UnstableIdx,
    ) -> Self {
        let n = unstable_idx.len();
        let idx = Some(unstable_idx);
        PatchesLinearBounds {
            row_count: n,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None,
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: idx.clone(),
            },
            lower_b: Array1::zeros(n),
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: idx,
            },
            upper_b: Array1::zeros(n),
        }
    }

    /// Convert Dense rows over a known spatial output tensor into row-aware
    /// Patches coefficients with 1x1 receptive fields.
    pub(crate) fn from_dense_spatial_rows(
        bounds: &LinearBounds,
        output_shape: (usize, usize, usize),
    ) -> Result<Self> {
        let (out_c, out_h, out_w) = output_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec("PatchesLinearBounds output shape overflow".into())
        })?;
        if bounds.num_inputs() != out_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![bounds.num_inputs()],
            });
        }

        let row_count = bounds.num_outputs();
        // #patches-row-range: the 7D re-entry PAIR materializes
        // `2 x row_count x out_c^2 x out_h x out_w` f32 cells — a factor of
        // `out_c` MORE than the dense pair it replaces (250 rows over VGG16's
        // 64x224x224 conv1 grid is a 411 GB pair, which aborted the process on
        // allocation). Refuse an over-budget (or overflowing) re-entry with
        // the structured `CpuMemoryExceeded`: the only production caller
        // (`try_dense_spatial_patches_reentry`) treats any Err as "skip
        // re-entry, stay Dense" — sound AND more precise (per-cell err carried
        // natively there; see the err-carry notes below).
        let required = checked_shape_product(&[row_count, out_c, out_h, out_w, out_c])
            .and_then(|cells| cells.checked_mul(2 * size_of::<f32>()))
            .unwrap_or(usize::MAX);
        let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required > budget {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: required,
                budget_bytes: budget,
                site: "patches from_dense_spatial_rows 7D re-entry",
            });
        }
        let mut lower_patches =
            ArrayD::<f32>::zeros(IxDyn(&[row_count, out_c, out_h, out_w, out_c, 1, 1]));
        let mut upper_patches =
            ArrayD::<f32>::zeros(IxDyn(&[row_count, out_c, out_h, out_w, out_c, 1, 1]));

        for row in 0..row_count {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let flat = oc * out_h * out_w + oh * out_w + ow;
                        lower_patches[[row, oc, oh, ow, oc, 0, 0]] = bounds.lower_a()[[row, flat]];
                        upper_patches[[row, oc, oh, ow, oc, 0, 0]] = bounds.upper_a()[[row, flat]];
                    }
                }
            }
        }

        // Carry the source dense per-cell coefficient error into the 7D
        // re-entry as a per-spec-row bound (#patches-coeff-err-soundness,
        // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §10). The copy loop above writes
        // ONLY diagonal entries `P[r,oc,oh,ow,oc,0,0] = s_a[[r, flat]]` (bitwise;
        // every other 7D entry is a structural zero). So for each spec row `r`,
        // the per-row max of the source row's per-cell err over-bounds every
        // stored coefficient's true deviation: a copied diagonal entry deviates
        // by `<= E_s[r, flat] <= rowmax`; a structural zero has deviation 0 <=
        // any nonnegative row err. `sanitize` maps non-finite/negative to `+INF`
        // (an outward degrade — `set_coeff_err` does NOT sanitize despite its
        // doc); a wrong-shaped `E_s` returns `Err(ShapeMismatch)` so the caller
        // skips re-entry and stays Dense (per-cell err carried natively there:
        // sound AND more precise; never silently zero-filled).
        let row_max_err = |err: Option<&Array2<f32>>| -> Result<Option<Array1<f32>>> {
            let Some(e) = err else {
                return Ok(None); // exact source (unchanged)
            };
            if e.shape() != [row_count, out_dim] {
                return Err(NyError::ShapeMismatch {
                    expected: vec![row_count, out_dim],
                    got: e.shape().to_vec(),
                });
            }
            let mut out = Array1::<f32>::zeros(row_count);
            for r in 0..row_count {
                let mut rowmax = 0.0f32;
                for j in 0..out_dim {
                    let v = e[[r, j]];
                    // sanitize: non-finite or negative => +INF (poison outward).
                    // `-0.0` is finite and `>= 0.0`, so it does NOT poison.
                    let s = if v.is_finite() && v >= 0.0 {
                        v
                    } else {
                        f32::INFINITY
                    };
                    if s > rowmax {
                        rowmax = s;
                    }
                }
                // H1 (spec §14): an all-zero sanitized row max stays exactly 0.0
                // (tighter than `next_up(0)`); a nonzero row takes one outward
                // ULP of doc-conformance slack. `+INF` is a `next_up` fixed point.
                out[r] = if rowmax == 0.0 {
                    0.0
                } else {
                    ny_tensor::next_up_f32(rowmax)
                };
            }
            Ok(Some(out))
        };
        let lower_coeff_err = row_max_err(bounds.lower_a_err())?;
        let upper_coeff_err = row_max_err(bounds.upper_a_err())?;

        Ok(PatchesLinearBounds {
            row_count,
            lower_a: PatchesData {
                coeff_err: lower_coeff_err,
                patches: Some(lower_patches),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape,
                input_shape: output_shape,
                unstable_idx: None,
            },
            lower_b: bounds.lower_b().to_owned(),
            upper_a: PatchesData {
                coeff_err: upper_coeff_err,
                patches: Some(upper_patches),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape,
                input_shape: output_shape,
                unstable_idx: None,
            },
            upper_b: bounds.upper_b().to_owned(),
        })
    }

    fn validate_row_count(&self) -> Result<()> {
        if self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some() {
            return Ok(());
        }
        if self.lower_b.len() != self.row_count || self.upper_b.len() != self.row_count {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.row_count],
                got: vec![self.lower_b.len().max(self.upper_b.len())],
            });
        }
        Ok(())
    }

    /// Authenticate the complete lower/upper sparse relation before any
    /// consumer reuses lower-side metadata to read or scatter the upper side.
    ///
    /// Sparse materializers historically selected the layout, unfold map, and
    /// output rows solely from `lower_a`.  A missing or permuted upper index, a
    /// mismatched tensor prefix, or a different geometry could therefore
    /// silently attach upper coefficients to the wrong neuron (and malformed
    /// shapes could panic in the unchecked scatter kernels).  Keep those
    /// kernels infallible by checking the paired contract once, before their
    /// first allocation.
    fn validate_sparse_pair(&self) -> Result<&UnstableIdx> {
        let mut deadline = PatchesMaterializationDeadline::new(None);
        self.validate_sparse_pair_with_poll(&mut deadline)
    }

    fn validate_sparse_pair_with_poll(
        &self,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<&UnstableIdx> {
        self.lower_a
            .validate_common_geometry_with_poll(&self.upper_a, deadline)?;
        self.lower_a
            .geometry
            .require_affine("sparse 4D/5D patch validation")?;

        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "sparse patches output shape overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;
        checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "sparse patches input shape overflows: {in_c} * {in_h} * {in_w}"
            ))
        })?;

        let lower_idx = self.lower_a.unstable_idx.as_ref().ok_or_else(|| {
            NyError::InvalidSpec("sparse patches lower side has no unstable_idx".into())
        })?;
        let upper_idx = self.upper_a.unstable_idx.as_ref().ok_or_else(|| {
            NyError::InvalidSpec("sparse patches upper side has no unstable_idx".into())
        })?;
        let indices_match = {
            let mut slices_equal = |left: &[usize], right: &[usize]| -> Result<bool> {
                if left.len() != right.len() {
                    return Ok(false);
                }
                for (&left, &right) in left.iter().zip(right.iter()) {
                    if left != right {
                        return Ok(false);
                    }
                    deadline.work(1, "during sparse lower/upper index comparison")?;
                }
                Ok(true)
            };
            slices_equal(&lower_idx.channels, &upper_idx.channels)?
                && slices_equal(&lower_idx.heights, &upper_idx.heights)?
                && slices_equal(&lower_idx.widths, &upper_idx.widths)?
        };
        if !indices_match {
            return Err(NyError::InvalidSpec(
                "sparse patches lower/upper unstable_idx differ".into(),
            ));
        }
        lower_idx.validate_with_poll(out_c, out_h, out_w, None, deadline)?;
        let unstable_size = lower_idx.len();

        if self.lower_a.identity != self.upper_a.identity {
            return Err(NyError::InvalidSpec(
                "sparse patches lower/upper identity flags differ".into(),
            ));
        }
        if self.lower_a.identity {
            self.lower_a.validate_identity_geometry()?;
            self.upper_a.validate_identity_geometry()?;
            if self.lower_a.patches.is_some() || self.upper_a.patches.is_some() {
                return Err(NyError::InvalidSpec(
                    "sparse identity must not carry materialized patches".into(),
                ));
            }
            if self.row_count != unstable_size {
                return Err(NyError::ShapeMismatch {
                    expected: vec![unstable_size],
                    got: vec![self.row_count],
                });
            }
            if self.lower_b.len() != unstable_size || self.upper_b.len() != unstable_size {
                return Err(NyError::ShapeMismatch {
                    expected: vec![unstable_size, unstable_size],
                    got: vec![self.lower_b.len(), self.upper_b.len()],
                });
            }
            deadline.checkpoint("after sparse identity pair validation")?;
            return Ok(lower_idx);
        }

        let lower_patches = self.lower_a.patches.as_ref().ok_or_else(|| {
            NyError::InvalidSpec("non-identity sparse lower patches tensor is missing".into())
        })?;
        let upper_patches = self.upper_a.patches.as_ref().ok_or_else(|| {
            NyError::InvalidSpec("non-identity sparse upper patches tensor is missing".into())
        })?;
        let shape = lower_patches.shape();
        if shape != upper_patches.shape() {
            return Err(NyError::ShapeMismatch {
                expected: shape.to_vec(),
                got: upper_patches.shape().to_vec(),
            });
        }
        checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec(format!("sparse patches tensor shape overflows: {shape:?}"))
        })?;

        let (kh, kw, expected_bias_len) = match shape.len() {
            4 => {
                if self.row_count != unstable_size || shape[0] != unstable_size || shape[1] != in_c
                {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![unstable_size, unstable_size, in_c],
                        got: vec![self.row_count, shape[0], shape[1]],
                    });
                }
                (shape[2], shape[3], unstable_size)
            }
            5 => {
                if shape[0] != self.row_count || shape[1] != unstable_size || shape[2] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![self.row_count, unstable_size, in_c],
                        got: vec![shape[0], shape[1], shape[2]],
                    });
                }
                (shape[3], shape[4], self.row_count)
            }
            rank => {
                return Err(NyError::ShapeMismatch {
                    expected: vec![4, 5],
                    got: vec![rank],
                });
            }
        };
        checked_shape_product(&[in_c, kh, kw]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "sparse patches kernel block overflows: {in_c} * {kh} * {kw}"
            ))
        })?;
        let actual_output = self
            .lower_a
            .validated_geometry_for_with_poll((kh, kw), deadline)?
            .require_affine("sparse 4D/5D patch validation")?
            .output_size((in_h, in_w), (kh, kw))?;
        if actual_output != (out_h, out_w) {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_h, out_w],
                got: vec![actual_output.0, actual_output.1],
            });
        }
        if self.lower_b.len() != expected_bias_len || self.upper_b.len() != expected_bias_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![expected_bias_len, expected_bias_len],
                got: vec![self.lower_b.len(), self.upper_b.len()],
            });
        }

        deadline.checkpoint("after sparse patch pair validation")?;
        Ok(lower_idx)
    }

    /// Dense matrix shape that `to_dense()` would materialize.
    pub(crate) fn dense_pair_shape(&self) -> Result<(usize, usize)> {
        self.validate_row_count()?;
        if self.lower_a.output_shape != self.upper_a.output_shape
            || self.lower_a.input_shape != self.upper_a.input_shape
        {
            return Err(NyError::InternalError(
                "PatchesLinearBounds: lower/upper spatial shapes differ".into(),
            ));
        }
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec("PatchesLinearBounds input shape overflow".into())
        })?;
        let legacy_sparse_rows =
            self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some();
        if legacy_sparse_rows {
            // Sparse identity has one stored row per unstable coordinate, not
            // one row per full-grid output. Authenticate that compact pair,
            // then report the rows its dense materialization will actually
            // publish. Performing the ordinary identity check first made every
            // nontrivial sparse-identity fallback fail at the budget guard.
            self.validate_sparse_pair()?;
            let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
                NyError::InvalidSpec("PatchesLinearBounds output shape overflow".into())
            })?;
            return Ok((out_dim, in_dim));
        }
        // Dense identity bounds require one row per output position. Explicit
        // spec rows are represented with materialized patches.
        if self.lower_a.identity || self.upper_a.identity {
            let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
                NyError::InvalidSpec("PatchesLinearBounds output shape overflow".into())
            })?;
            if self.row_count != out_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_dim],
                    got: vec![self.row_count],
                });
            }
        }
        Ok((self.row_count, in_dim))
    }

    /// Heap bytes needed for the Dense lower/upper coefficient pair.
    pub(crate) fn dense_pair_bytes(&self) -> Result<usize> {
        let (rows, cols) = self.dense_pair_shape()?;
        Ok(crate::network::crown_memory::dense_pair_bytes(rows, cols).unwrap_or(usize::MAX))
    }

    /// Filter dense patches to only keep unstable output neurons (sparse mode).
    ///
    /// Given a boolean mask of unstable neurons over the full `(out_c, out_h, out_w)`
    /// grid, extracts only the patches and biases for unstable positions.
    ///
    /// Returns `None` if all neurons are unstable (no benefit from sparse mode).
    /// Returns `None` if fewer than `(1.0 - min_sparsity) * total` neurons are unstable
    /// (the default `min_sparsity` of 0.9 means at least 10% must be stable).
    ///
    /// **Precondition:** `self` must be in dense mode (no existing `unstable_idx`).
    ///
    /// Reference: alpha-beta-CROWN `backward_bound.py` `get_sparse_C`, `minimum_sparsity=0.9`
    /// Part of #2613 Phase 4 step 19 — currently test-only; remove #[cfg(test)] when
    /// wiring to CROWN backward engine.
    #[cfg(test)]
    pub(crate) fn filter_to_unstable(
        &self,
        unstable_mask: &ndarray::Array3<bool>,
        min_sparsity: f32,
    ) -> Option<PatchesLinearBounds> {
        debug_assert!(
            self.lower_a.unstable_idx.is_none(),
            "filter_to_unstable called on already-sparse patches"
        );
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let total = checked_shape_product(&[out_c, out_h, out_w])?;
        if self.row_count != total {
            return None;
        }

        // Collect unstable positions
        let mut channels = Vec::new();
        let mut heights = Vec::new();
        let mut widths = Vec::new();
        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    if unstable_mask[[c, h, w]] {
                        channels.push(c);
                        heights.push(h);
                        widths.push(w);
                    }
                }
            }
        }
        let unstable_size = channels.len();

        // Check sparsity threshold
        if unstable_size >= total || (unstable_size as f32) > min_sparsity * (total as f32) {
            return None;
        }
        if unstable_size == 0 {
            // All stable — no backward needed. Return empty sparse patches.
            let idx = UnstableIdx {
                channels: vec![],
                heights: vec![],
                widths: vec![],
            };
            return Some(PatchesLinearBounds::sparse_identity(
                self.lower_a.output_shape,
                self.lower_a.input_shape,
                idx,
            ));
        }

        let idx = UnstableIdx {
            channels,
            heights,
            widths,
        };

        // Extract sparse patches from dense 6D tensor
        let lower_a = Self::extract_sparse_patches(&self.lower_a, &idx)?;
        let upper_a = Self::extract_sparse_patches(&self.upper_a, &idx)?;

        // Extract sparse bias vectors
        let mut lower_b = Array1::zeros(unstable_size);
        let mut upper_b = Array1::zeros(unstable_size);
        for (i, flat) in idx
            .channels
            .iter()
            .zip(idx.heights.iter())
            .zip(idx.widths.iter())
            .map(|((c, h), w)| c * out_h * out_w + h * out_w + w)
            .enumerate()
        {
            lower_b[i] = self.lower_b[flat];
            upper_b[i] = self.upper_b[flat];
        }

        Some(PatchesLinearBounds {
            row_count: unstable_size,
            lower_a,
            lower_b,
            upper_a,
            upper_b,
        })
    }

    /// Extract sparse patches from a single PatchesData, keeping only unstable positions.
    #[cfg(test)]
    fn extract_sparse_patches(data: &PatchesData, idx: &UnstableIdx) -> Option<PatchesData> {
        let unstable_size = idx.len();
        let patches = match &data.patches {
            None => {
                // Identity: no tensor to extract from
                return Some(PatchesData {
                    coeff_err: None,
                    patches: None,
                    geometry: data.geometry.clone(),
                    identity: true,
                    output_shape: data.output_shape,
                    input_shape: data.input_shape,
                    unstable_idx: Some(idx.clone()),
                });
            }
            Some(p) => p,
        };
        let shape = patches.shape();
        let in_c = shape[3];
        let kh = shape[4];
        let kw = shape[5];

        // Sparse patches: (unstable_size, in_c, kH, kW) — 4D
        let mut sparse = ArrayD::zeros(IxDyn(&[unstable_size, in_c, kh, kw]));
        for (i, ((&c, &h), &w)) in idx
            .channels
            .iter()
            .zip(idx.heights.iter())
            .zip(idx.widths.iter())
            .enumerate()
        {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        sparse[[i, ic, ki, kj]] = patches[[c, h, w, ic, ki, kj]];
                    }
                }
            }
        }

        Some(PatchesData {
            coeff_err: None,
            patches: Some(sparse),
            geometry: data.geometry.clone(),
            identity: false,
            output_shape: data.output_shape,
            input_shape: data.input_shape,
            unstable_idx: Some(idx.clone()),
        })
    }

    /// Total logical heap payload used by this Patches bounds struct, in bytes.
    ///
    /// Includes both A-matrices, typed-geometry backing storage, and bias
    /// vectors. A virtual affine identity has no A/geometry heap allocation;
    /// Arc-backed anchored axes are counted even before coefficients exist.
    /// Lower/upper carriers normally share those Arcs, but this diagnostic
    /// deliberately charges each carrier's logical payload independently: it
    /// is a conservative sharing estimate, not a unique-allocation or backing
    /// capacity census. See [`PatchesMemoryAdmission`] for the retained-capacity
    /// boundary.
    pub(crate) fn memory_bytes(&self) -> usize {
        self.lower_a
            .memory_bytes()
            .saturating_add(self.lower_b.len().saturating_mul(size_of::<f32>()))
            .saturating_add(self.upper_a.memory_bytes())
            .saturating_add(self.upper_b.len().saturating_mul(size_of::<f32>()))
    }
}
