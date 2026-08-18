// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound, default-dark Clip-and-Verify primitive for one exact input domain.
//!
//! The historical disjunctive clip accepted the box and affine planes as
//! unrelated arguments.  That made it possible to clip a child with a reused
//! parent plane; on LSNC's batch-stack-unsafe graph this produced the real
//! `quadrotor2d_state_34` / `state_45` false UNSATs.  This module deliberately
//! is wired only through a typed, default-off dispatcher for batch-stack-unsafe
//! grouped input splitting.  Its entry point invokes one non-domain-stacked,
//! full-spec CROWN callback on the exact source box and keeps all resulting
//! planes paired with that box until clipping finishes.
//!
//! Soundness contract for `compute_full_spec_crown_on_exact_domain`:
//!
//! * it must bound exactly the `BoundedTensor` passed to it;
//! * it must perform one non-domain-stacked CROWN computation;
//! * it must not return a parent, ancestor, sibling, cached-batch, or otherwise
//!   inherited plane; and
//! * each returned lower (upper, in upper-verification mode) affine row must be
//!   a certified bound for the corresponding objective on that exact box.
//!
//! The primitive discharges certified coefficient error outward over the same
//! source box before any plane is consumed.  For an OR-of-AND property it clips
//! every clause independently from the source box and carries the bounding box
//! of the union of the surviving clause regions.  A terminal refutation is
//! published only when every clause is independently empty.  Any callback,
//! layout, shape, numerical, or clipping error returns the original box with
//! [`FreshDomainClipStatus::Skipped`], never a refutation.

use std::cell::Cell;

use ndarray::IxDyn;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;

use super::batched_clip::batched_relaxed_clip_from_flat;

/// Run-local, self-invalidating telemetry for the default-dark dispatcher.
///
/// Every verifier invocation owns its local counters and also forwards each
/// disposition to the command-scoped structured recorder. With the gate off
/// this object is silent and allocation-free beyond the five inline counters.
pub(super) struct FreshDomainClipTelemetry {
    enabled: bool,
    attempts: Cell<usize>,
    applied: Cell<usize>,
    all_clauses_refuted: Cell<usize>,
    skipped: Cell<usize>,
    tightened_dimensions: Cell<usize>,
}

impl FreshDomainClipTelemetry {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            attempts: Cell::new(0),
            applied: Cell::new(0),
            all_clauses_refuted: Cell::new(0),
            skipped: Cell::new(0),
            tightened_dimensions: Cell::new(0),
        }
    }

    #[inline]
    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn record(&self, source: &BoundedTensor, outcome: &FreshDomainClipResult) {
        if !self.enabled {
            return;
        }
        self.attempts.set(self.attempts.get().saturating_add(1));
        match outcome.status {
            FreshDomainClipStatus::Applied => {
                self.applied.set(self.applied.get().saturating_add(1));
                let tightened = source
                    .lower()
                    .iter()
                    .zip(source.upper().iter())
                    .zip(
                        outcome
                            .bounds
                            .lower()
                            .iter()
                            .zip(outcome.bounds.upper().iter()),
                    )
                    .filter(|&((&source_l, &source_u), (&clipped_l, &clipped_u))| {
                        clipped_l > source_l || clipped_u < source_u
                    })
                    .count();
                self.tightened_dimensions
                    .set(self.tightened_dimensions.get().saturating_add(tightened));
                crate::execution_telemetry::record_fresh_domain_clip_outcome(
                    crate::execution_telemetry::FreshDomainClipDisposition::Applied,
                    tightened,
                );
            }
            FreshDomainClipStatus::AllClausesRefuted => {
                self.all_clauses_refuted
                    .set(self.all_clauses_refuted.get().saturating_add(1));
                crate::execution_telemetry::record_fresh_domain_clip_outcome(
                    crate::execution_telemetry::FreshDomainClipDisposition::AllClausesRefuted,
                    0,
                );
            }
            FreshDomainClipStatus::Skipped => {
                self.skipped.set(self.skipped.get().saturating_add(1));
                crate::execution_telemetry::record_fresh_domain_clip_outcome(
                    crate::execution_telemetry::FreshDomainClipDisposition::Skipped,
                    0,
                );
            }
        }
        let attempts = self.attempts.get();
        if attempts == 1 || attempts.is_power_of_two() {
            eprintln!("{}", self.marker("progress"));
        }
    }

    fn marker(&self, status: &str) -> String {
        format!(
            "NY_FRESH_DOMAIN_CLIP route=grouped-disjunctive-current-domain \
             status={status} attempts={} applied={} all_clauses_refuted={} skipped={} \
             tightened_dimensions={}",
            self.attempts.get(),
            self.applied.get(),
            self.all_clauses_refuted.get(),
            self.skipped.get(),
            self.tightened_dimensions.get(),
        )
    }

    #[cfg(test)]
    pub(super) fn snapshot(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.attempts.get(),
            self.applied.get(),
            self.all_clauses_refuted.get(),
            self.skipped.get(),
            self.tightened_dimensions.get(),
        )
    }
}

impl Drop for FreshDomainClipTelemetry {
    fn drop(&mut self) {
        if self.enabled {
            eprintln!("{}", self.marker("final"));
        }
    }
}

/// Outcome of one exact-domain fresh-plane clip attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FreshDomainClipStatus {
    /// Fresh certified planes were consumed.  The returned box encloses every
    /// point that can satisfy at least one clause (it may equal the source box).
    Applied,
    /// Every clause's own affine half-space intersection was empty.
    AllClausesRefuted,
    /// The attempt failed validation or bounding.  The returned box is the
    /// original source box and this status has no verification authority.
    Skipped,
}

/// Fail-closed result: `bounds` is always valid and is bit-identical to the
/// source box for `Skipped` and `AllClausesRefuted`.
#[derive(Debug)]
pub(super) struct FreshDomainClipResult {
    pub(super) bounds: BoundedTensor,
    pub(super) status: FreshDomainClipStatus,
}

/// Fresh full-spec CROWN proof planes inseparably paired with their exact source
/// domain. Fields and constructor are private so clipping code cannot provide a
/// different (for example, child) box after the planes have been computed.
struct FoldedExactDomainPlanes {
    source: BoundedTensor,
    planes: LinearBounds,
}

impl FoldedExactDomainPlanes {
    fn compute<F>(source: &BoundedTensor, compute_full_spec_crown: F) -> Result<Self>
    where
        F: FnOnce(&BoundedTensor) -> Result<LinearBounds>,
    {
        // Own the exact callback argument and retain that same value as the only
        // box the resulting planes may clip.
        let source = source.clone();
        validate_finite_nonempty_box(&source)?;
        let mut planes = compute_full_spec_crown(&source)?;
        planes.validate_internal_shapes()?;
        planes.validate_no_nan()?;

        let flat = source.flatten();
        let in_l = flat.lower().as_slice().ok_or_else(|| {
            NyError::InternalError(
                "fresh-domain clip: flattened lower box is not contiguous".into(),
            )
        })?;
        let in_u = flat.upper().as_slice().ok_or_else(|| {
            NyError::InternalError(
                "fresh-domain clip: flattened upper box is not contiguous".into(),
            )
        })?;
        if planes.lower_a().ncols() != in_l.len() {
            return Err(NyError::InvalidSpec(format!(
                "fresh-domain clip: plane input dimension {} != exact box dimension {}",
                planes.lower_a().ncols(),
                in_l.len()
            )));
        }

        // The carrier E certifies |A_stored - A_real| <= E.  Folding
        // sum_j E_ij max(|l_j|, |u_j|) into b in the outward direction makes
        // the subsequently consumed f32 coefficients independently sound on
        // this exact source box.
        planes.fold_coeff_err_into_bias(in_l, in_u);
        planes.validate_internal_shapes()?;
        planes.validate_no_nan()?;
        if planes.has_coeff_err() {
            return Err(NyError::NumericalInstability(
                "fresh-domain clip: coefficient error was not fully discharged".into(),
            ));
        }

        Ok(Self { source, planes })
    }
}

/// Clip one exact domain using planes computed synchronously on that same
/// domain.  No shipped verifier preset enables its default-dark dispatcher.
///
/// `compute_full_spec_crown_on_exact_domain` is called exactly once. See the
/// module-level soundness contract; in particular, it may not domain-stack or
/// reuse planes. All errors fail closed to an unchanged, non-authoritative
/// result so deadline exhaustion can safely fall through to the existing BaB
/// path.
#[allow(clippy::too_many_arguments)]
pub(super) fn clip_with_fresh_exact_domain_planes<F>(
    exact_domain: &BoundedTensor,
    thresholds: &[f32],
    clause_sizes: &[usize],
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
    compute_full_spec_crown_on_exact_domain: F,
) -> FreshDomainClipResult
where
    F: FnOnce(&BoundedTensor) -> Result<LinearBounds>,
{
    let fallback = || FreshDomainClipResult {
        bounds: exact_domain.clone(),
        status: FreshDomainClipStatus::Skipped,
    };

    match clip_with_fresh_exact_domain_planes_checked(
        exact_domain,
        thresholds,
        clause_sizes,
        verify_upper_bound,
        relaxed_clip_iterations,
        compute_full_spec_crown_on_exact_domain,
    ) {
        Ok(result) => result,
        Err(_) => fallback(),
    }
}

#[allow(clippy::too_many_arguments)]
fn clip_with_fresh_exact_domain_planes_checked<F>(
    exact_domain: &BoundedTensor,
    thresholds: &[f32],
    clause_sizes: &[usize],
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
    compute_full_spec_crown_on_exact_domain: F,
) -> Result<FreshDomainClipResult>
where
    F: FnOnce(&BoundedTensor) -> Result<LinearBounds>,
{
    validate_layout(thresholds, clause_sizes, relaxed_clip_iterations)?;
    let fresh =
        FoldedExactDomainPlanes::compute(exact_domain, compute_full_spec_crown_on_exact_domain)?;
    let n_rows = fresh.planes.lower_a().nrows();
    if n_rows != thresholds.len() {
        return Err(NyError::InvalidSpec(format!(
            "fresh-domain clip: plane rows {} != threshold rows {}",
            n_rows,
            thresholds.len()
        )));
    }

    let flat = fresh.source.flatten();
    let result = batched_relaxed_clip_from_flat(
        std::slice::from_ref(flat.lower()),
        std::slice::from_ref(flat.upper()),
        &[&fresh.planes],
        thresholds,
        clause_sizes,
        verify_upper_bound,
        relaxed_clip_iterations,
    )?;

    let all_clauses_refuted = result.verified.first().copied().ok_or_else(|| {
        NyError::InternalError("fresh-domain clip: missing single-domain result".into())
    })?;
    if all_clauses_refuted {
        return Ok(FreshDomainClipResult {
            // The clipped sentinel is intentionally never published.  Keeping
            // the exact source makes accidental post-terminal reads harmless.
            bounds: fresh.source,
            status: FreshDomainClipStatus::AllClausesRefuted,
        });
    }

    let clipped_l = result
        .clipped_lowers
        .into_iter()
        .next()
        .ok_or_else(|| NyError::InternalError("fresh-domain clip: missing lower box".into()))?;
    let clipped_u = result
        .clipped_uppers
        .into_iter()
        .next()
        .ok_or_else(|| NyError::InternalError("fresh-domain clip: missing upper box".into()))?;
    let shape = IxDyn(fresh.source.lower().shape());
    let clipped_l = clipped_l
        .into_shape_with_order(shape.clone())
        .map_err(|error| {
            NyError::InternalError(format!(
                "fresh-domain clip: lower reshape to source {:?} failed: {error}",
                fresh.source.lower().shape()
            ))
        })?;
    let clipped_u = clipped_u.into_shape_with_order(shape).map_err(|error| {
        NyError::InternalError(format!(
            "fresh-domain clip: upper reshape to source {:?} failed: {error}",
            fresh.source.upper().shape()
        ))
    })?;
    let clipped = BoundedTensor::new(clipped_l, clipped_u)?;

    // Defense in depth: relaxed clipping may tighten but must never widen past
    // the exact source domain.  A numerical or implementation defect here is a
    // skipped optimization, not certificate authority.
    if !box_is_subset(&clipped, &fresh.source) {
        return Err(NyError::NumericalInstability(
            "fresh-domain clip widened outside its exact source box".into(),
        ));
    }

    Ok(FreshDomainClipResult {
        bounds: clipped,
        status: FreshDomainClipStatus::Applied,
    })
}

fn validate_finite_nonempty_box(domain: &BoundedTensor) -> Result<()> {
    if domain.lower().is_empty() {
        return Err(NyError::InvalidSpec(
            "fresh-domain clip: empty input domain".into(),
        ));
    }
    for (index, (&lower, &upper)) in domain.lower().iter().zip(domain.upper().iter()).enumerate() {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(NyError::NumericalInstability(format!(
                "fresh-domain clip: invalid exact box at flat index {index}: [{lower}, {upper}]"
            )));
        }
    }
    Ok(())
}

fn validate_layout(
    thresholds: &[f32],
    clause_sizes: &[usize],
    relaxed_clip_iterations: usize,
) -> Result<()> {
    if thresholds.is_empty() || clause_sizes.is_empty() {
        return Err(NyError::InvalidSpec(
            "fresh-domain clip: thresholds and clauses must be non-empty".into(),
        ));
    }
    if relaxed_clip_iterations == 0 {
        return Err(NyError::InvalidSpec(
            "fresh-domain clip: relaxed_clip_iterations must be positive".into(),
        ));
    }
    if clause_sizes.contains(&0) {
        return Err(NyError::InvalidSpec(
            "fresh-domain clip: zero-sized clause".into(),
        ));
    }
    let total = clause_sizes
        .iter()
        .try_fold(0usize, |sum, &size| sum.checked_add(size))
        .ok_or_else(|| NyError::InvalidSpec("fresh-domain clip: clause total overflow".into()))?;
    if total != thresholds.len() {
        return Err(NyError::InvalidSpec(format!(
            "fresh-domain clip: clause total {total} != threshold rows {}",
            thresholds.len()
        )));
    }
    if let Some((row, threshold)) = thresholds
        .iter()
        .copied()
        .enumerate()
        .find(|(_, threshold)| !threshold.is_finite())
    {
        return Err(NyError::NumericalInstability(format!(
            "fresh-domain clip: threshold row {row} is non-finite ({threshold})"
        )));
    }
    Ok(())
}

fn box_is_subset(inner: &BoundedTensor, outer: &BoundedTensor) -> bool {
    inner.lower().shape() == outer.lower().shape()
        && inner
            .lower()
            .iter()
            .zip(inner.upper().iter())
            .zip(outer.lower().iter().zip(outer.upper().iter()))
            .all(|((&inner_l, &inner_u), (&outer_l, &outer_u))| {
                inner_l >= outer_l && inner_u <= outer_u
            })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use ndarray::{Array1, Array2};
    use rand::{RngExt, SeedableRng};

    use super::*;

    fn exact_planes(a: Array2<f32>, b: Array1<f32>) -> LinearBounds {
        LinearBounds::new(a.clone(), b.clone(), a, b).expect("valid exact affine planes")
    }

    fn flat_box(lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
        BoundedTensor::new(
            Array1::from(lower).into_dyn(),
            Array1::from(upper).into_dyn(),
        )
        .expect("valid finite box")
    }

    fn assert_box_bit_identical(actual: &BoundedTensor, expected: &BoundedTensor) {
        assert_eq!(actual.lower().shape(), expected.lower().shape());
        assert_eq!(actual.upper().shape(), expected.upper().shape());
        for (&a, &e) in actual.lower().iter().zip(expected.lower().iter()) {
            assert_eq!(a.to_bits(), e.to_bits());
        }
        for (&a, &e) in actual.upper().iter().zip(expected.upper().iter()) {
            assert_eq!(a.to_bits(), e.to_bits());
        }
    }

    fn assert_point_in_box(point: &[f32], bounds: &BoundedTensor, tolerance: f32) {
        let flat = bounds.flatten();
        for (dimension, ((&x, &lower), &upper)) in point
            .iter()
            .zip(flat.lower().iter())
            .zip(flat.upper().iter())
            .enumerate()
        {
            assert!(
                x >= lower - tolerance && x <= upper + tolerance,
                "point escaped dimension {dimension}: x={x}, clip=[{lower}, {upper}]"
            );
        }
    }

    /// Structural reproduction of the 2025 LSNC state-property layout:
    /// thirteen OR clauses, each containing its unsafe row and the two common
    /// Y_1 shell rows.  Only clause zero has a witness; all other clauses are
    /// independently refuted.  The fresh clause-union primitive must therefore
    /// retain the witness and may not publish UNSAT.
    fn run_lsnc_state_moat(
        state: usize,
        witness: [f32; 6],
        witness_y0: f32,
        witness_y1: f32,
        shell_lower: f32,
        shell_upper: f32,
    ) {
        let source = flat_box(
            vec![-0.7, -0.7, -1.0, -3.0, -3.0, -2.0],
            vec![0.7, 0.7, 1.0, 3.0, 3.0, 2.0],
        );
        assert_point_in_box(&witness, &source, 0.0);
        assert!(witness_y0 >= 1.0e-6);
        assert!(witness_y1 >= shell_lower && witness_y1 <= shell_upper);

        // Planner-normalized first-row thresholds for the thirteen unsafe
        // alternatives in quadrotor2d_state_{34,45}.  Every objective is in the
        // lower-bound direction: a counterexample satisfies g_r(x) <= t_r.
        let unsafe_thresholds = [
            -1.0e-6, -0.700_001, -0.700_001, -0.700_001, -0.700_001, -1.000_001, -1.000_001,
            -3.000_001, -3.000_001, -3.000_001, -3.000_001, -2.000_001, -2.000_001,
        ];
        let mut thresholds = Vec::with_capacity(39);
        let mut biases = Vec::with_capacity(39);
        for (clause, unsafe_threshold) in unsafe_thresholds.into_iter().enumerate() {
            thresholds.extend([unsafe_threshold, shell_upper, -shell_lower]);
            // Exact constant affine objectives make the geometry independent of
            // an ONNX runtime in this unit moat.  Clause zero represents
            // -Y_0 <= -1e-6 and is satisfied by the recorded/shell witness;
            // every other unsafe alternative is deliberately infeasible.
            let unsafe_value = if clause == 0 {
                -witness_y0
            } else {
                unsafe_threshold + 0.25
            };
            biases.extend([unsafe_value, witness_y1, -witness_y1]);
        }
        let a = Array2::<f32>::zeros((39, 6));
        let b = Array1::from_vec(biases);
        let callback_called = Cell::new(false);
        let result = clip_with_fresh_exact_domain_planes(
            &source,
            &thresholds,
            &[3; 13],
            false,
            20,
            |callback_box| {
                callback_called.set(true);
                assert_box_bit_identical(callback_box, &source);
                Ok(exact_planes(a, b))
            },
        );

        assert!(callback_called.get(), "state_{state}: fresh CROWN callback");
        assert_eq!(
            result.status,
            FreshDomainClipStatus::Applied,
            "state_{state}: one feasible OR clause must prevent a terminal refutation"
        );
        assert_point_in_box(&witness, &result.bounds, 1.0e-6);
    }

    #[test]
    fn lsnc_state_34_known_counterexample_moat() {
        // Genuine in-box witness banked for the 2025 state_34 property.
        run_lsnc_state_moat(
            34,
            [
                -0.492_563_5,
                -0.174_982_25,
                0.160_729_75,
                0.417_165_94,
                2.608_798_3,
                1.591_176_5,
            ],
            0.000_566_918_5,
            0.763_050_6,
            0.713_956_5,
            0.763_956_5,
        );
    }

    #[test]
    fn lsnc_state_45_live_breach_layout_moat() {
        // state_45 was the live false-UNSAT breach.  Its historical CE was not
        // persisted, so use the commit's in-box LSNC witness coordinates with
        // an exact affine fixture at the real state_45 Y_1 shell.  This seals
        // the same 13x3 OR-of-AND geometry without claiming an ONNX replay.
        run_lsnc_state_moat(
            45,
            [-0.206, -0.336, 0.219, 0.061, 2.669, 1.456],
            0.001_19,
            0.599_619_87,
            0.574_619_9,
            0.624_619_9,
        );
    }

    #[test]
    fn terminal_status_requires_every_clause_to_be_refuted() {
        let source = flat_box(vec![0.0], vec![1.0]);
        // Clause 0: x <= 0.25 (feasible). Clause 1: x <= -1 (empty).
        let one_kept = exact_planes(
            Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(),
            Array1::zeros(2),
        );
        let result =
            clip_with_fresh_exact_domain_planes(&source, &[0.25, -1.0], &[1, 1], false, 3, |_| {
                Ok(one_kept)
            });
        assert_eq!(result.status, FreshDomainClipStatus::Applied);
        assert!(result.bounds.upper()[[0]] >= 0.25 - 1.0e-5);

        // Both single-row clauses are empty on [0,1].
        let all_empty = exact_planes(
            Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap(),
            Array1::zeros(2),
        );
        let result =
            clip_with_fresh_exact_domain_planes(&source, &[-1.0, -2.0], &[1, 1], false, 3, |_| {
                Ok(all_empty)
            });
        assert_eq!(result.status, FreshDomainClipStatus::AllClausesRefuted);
        assert_box_bit_identical(&result.bounds, &source);
    }

    #[test]
    fn upper_direction_selects_and_outward_folds_upper_planes() {
        let source = flat_box(vec![0.0], vec![1.0]);
        let upper_planes = || {
            // Lower rows are deliberately uninformative: this test must consume
            // the upper rows. A 0.1 coefficient-error radius on [0,1] must be
            // folded upward into upper_b before the clip negates the plane.
            let mut planes = LinearBounds::new(
                Array2::zeros((2, 1)),
                Array1::zeros(2),
                Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap(),
                Array1::zeros(2),
            )
            .expect("valid asymmetric upper-direction planes");
            planes.set_coeff_err(Array2::zeros((2, 1)), Array2::from_elem((2, 1), 0.1));
            planes
        };

        // Upper-side counterexamples satisfy upper(x) >= threshold. Clause 0
        // keeps x + 0.1 >= 0.75 (x >= 0.65 after outward folding); clause 1 is
        // empty. The feasible first clause must prevent terminal refutation and
        // retain x=0.7.
        let one_kept =
            clip_with_fresh_exact_domain_planes(&source, &[0.75, 2.0], &[1, 1], true, 3, |_| {
                Ok(upper_planes())
            });
        assert_eq!(one_kept.status, FreshDomainClipStatus::Applied);
        assert_point_in_box(&[0.7], &one_kept.bounds, 1.0e-6);
        assert!(
            one_kept.bounds.lower()[[0]] > 0.5 && one_kept.bounds.lower()[[0]] < 0.7,
            "upper plane should clip near the outward-folded x>=0.65 boundary, got {}",
            one_kept.bounds.lower()[[0]]
        );

        // Both upper-side clauses are impossible on [0,1], even after the
        // outward 0.1 bias fold, so terminal authority is permitted.
        let all_empty =
            clip_with_fresh_exact_domain_planes(&source, &[2.0, 2.0], &[1, 1], true, 3, |_| {
                Ok(upper_planes())
            });
        assert_eq!(all_empty.status, FreshDomainClipStatus::AllClausesRefuted);
        assert_box_bit_identical(&all_empty.bounds, &source);
    }

    #[test]
    fn callback_layout_shape_and_numerical_errors_fail_closed() {
        let source = flat_box(vec![-1.0, -2.0], vec![1.0, 2.0]);
        let valid = || exact_planes(Array2::zeros((1, 2)), Array1::zeros(1));

        let callback_error =
            clip_with_fresh_exact_domain_planes(&source, &[0.0], &[1], false, 1, |_| {
                Err(NyError::InternalError("deadline".into()))
            });
        assert_eq!(callback_error.status, FreshDomainClipStatus::Skipped);
        assert_box_bit_identical(&callback_error.bounds, &source);

        let bad_layout = clip_with_fresh_exact_domain_planes(
            &source,
            &[0.0],
            &[1, 1],
            false,
            1,
            |_| Ok(valid()),
        );
        assert_eq!(bad_layout.status, FreshDomainClipStatus::Skipped);
        assert_box_bit_identical(&bad_layout.bounds, &source);

        let wrong_input_dim =
            clip_with_fresh_exact_domain_planes(&source, &[0.0], &[1], false, 1, |_| {
                Ok(exact_planes(Array2::zeros((1, 1)), Array1::zeros(1)))
            });
        assert_eq!(wrong_input_dim.status, FreshDomainClipStatus::Skipped);
        assert_box_bit_identical(&wrong_input_dim.bounds, &source);

        let nonfinite_threshold =
            clip_with_fresh_exact_domain_planes(&source, &[f32::NAN], &[1], false, 1, |_| {
                Ok(valid())
            });
        assert_eq!(nonfinite_threshold.status, FreshDomainClipStatus::Skipped);
        assert_box_bit_identical(&nonfinite_threshold.bounds, &source);
    }

    #[test]
    fn telemetry_counters_saturate_without_affecting_dispatch() {
        let source = flat_box(vec![0.0], vec![1.0]);
        let outcome = FreshDomainClipResult {
            bounds: flat_box(vec![0.0], vec![0.5]),
            status: FreshDomainClipStatus::Applied,
        };
        let mut telemetry = FreshDomainClipTelemetry::new(true);
        telemetry.attempts.set(usize::MAX);
        telemetry.applied.set(usize::MAX);
        telemetry.all_clauses_refuted.set(usize::MAX);
        telemetry.skipped.set(usize::MAX);
        telemetry.tightened_dimensions.set(usize::MAX);

        telemetry.record(&source, &outcome);

        assert_eq!(
            telemetry.snapshot(),
            (usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX)
        );
        telemetry.enabled = false;
    }

    #[test]
    fn local_telemetry_forwards_actual_outcome_to_command_scope() {
        let _test_lock = crate::execution_telemetry::TEST_LOCK
            .lock()
            .expect("telemetry test lock");
        let _run = crate::execution_telemetry::begin_run();
        crate::execution_telemetry::record_fresh_domain_clip_route(true, true);

        let source = flat_box(vec![0.0, -1.0], vec![1.0, 1.0]);
        let outcome = FreshDomainClipResult {
            bounds: flat_box(vec![0.25, -1.0], vec![1.0, 0.5]),
            status: FreshDomainClipStatus::Applied,
        };
        let mut telemetry = FreshDomainClipTelemetry::new(true);
        telemetry.record(&source, &outcome);

        let observed = crate::execution_telemetry::snapshot();
        assert_eq!(observed.fresh_domain_clip.attempts, 1);
        assert_eq!(observed.fresh_domain_clip.applied, 1);
        assert_eq!(observed.fresh_domain_clip.tightened_dimensions, 2);
        assert!(!observed.fresh_domain_clip.attribution_conflict);
        telemetry.enabled = false;
    }

    /// Randomized inclusion oracle over exact affine objectives with a carried
    /// coefficient-error certificate.  Any point satisfying one true clause
    /// must survive in the union box, and its existence forbids a terminal
    /// refutation.  This jointly exercises exact-box callback binding, outward
    /// error discharge, within-clause intersection, and across-clause union.
    #[ntest::timeout(60000)]
    #[test]
    fn randomized_fresh_plane_clause_union_includes_every_sampled_witness() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xF2E5_4D0A_C11F_2026);
        let mut witnessed_nontrivial_tightening = false;

        for trial in 0..160 {
            let dimension = rng.random_range(1usize..=4);
            let clause_count = rng.random_range(2usize..=4);
            let clause_sizes: Vec<usize> = (0..clause_count)
                .map(|_| rng.random_range(1usize..=3))
                .collect();
            let rows: usize = clause_sizes.iter().sum();

            let mut lower = vec![0.0; dimension];
            let mut upper = vec![0.0; dimension];
            let mut designated_witness = vec![0.0; dimension];
            for d in 0..dimension {
                lower[d] = rng.random_range(-2.0f32..1.0);
                upper[d] = lower[d] + rng.random_range(0.2f32..2.0);
                designated_witness[d] = rng.random_range(lower[d]..upper[d]);
            }
            let source = flat_box(lower.clone(), upper.clone());

            let mut stored_a = Array2::<f32>::zeros((rows, dimension));
            let mut true_a = Array2::<f32>::zeros((rows, dimension));
            let mut coeff_err = Array2::<f32>::zeros((rows, dimension));
            let mut bias = Array1::<f32>::zeros(rows);
            let mut thresholds = vec![0.0; rows];
            let first_clause_rows = clause_sizes[0];
            for row in 0..rows {
                for d in 0..dimension {
                    let stored = rng.random_range(-2.0f32..2.0);
                    let error = rng.random_range(0.0f32..2.0e-3);
                    let delta = rng.random_range(-error..=error);
                    stored_a[[row, d]] = stored;
                    true_a[[row, d]] = stored + delta;
                    // Treat both f32 coefficients as exact reals.  Round their
                    // real-valued gap upward so this test's carrier actually
                    // certifies the post-add rounded `true_a` coefficient.
                    let exact_gap = (true_a[[row, d]] as f64 - stored_a[[row, d]] as f64).abs();
                    coeff_err[[row, d]] = ny_tensor::next_up_f32(exact_gap as f32);
                }
                bias[row] = rng.random_range(-1.0f32..1.0);
                let at_witness = bias[row]
                    + (0..dimension)
                        .map(|d| true_a[[row, d]] * designated_witness[d])
                        .sum::<f32>();
                thresholds[row] = if row < first_clause_rows {
                    // Clause zero is guaranteed strictly feasible at the
                    // designated witness in every trial.
                    at_witness + rng.random_range(0.02f32..0.4)
                } else {
                    at_witness + rng.random_range(-1.0f32..1.0)
                };
            }

            let planes = LinearBounds::new_or_conservative_with_err(
                stored_a.clone(),
                bias.clone(),
                stored_a,
                bias.clone(),
                coeff_err.clone(),
                coeff_err,
            )
            .expect("valid error-carrying affine bounds");
            let result = clip_with_fresh_exact_domain_planes(
                &source,
                &thresholds,
                &clause_sizes,
                false,
                4,
                |callback_box| {
                    assert_box_bit_identical(callback_box, &source);
                    Ok(planes)
                },
            );
            assert_eq!(
                result.status,
                FreshDomainClipStatus::Applied,
                "trial {trial}: designated strict witness forbids skip/refutation"
            );
            assert_point_in_box(&designated_witness, &result.bounds, 2.0e-4);
            if result
                .bounds
                .lower()
                .iter()
                .zip(source.lower().iter())
                .any(|(&clipped, &original)| clipped > original)
                || result
                    .bounds
                    .upper()
                    .iter()
                    .zip(source.upper().iter())
                    .any(|(&clipped, &original)| clipped < original)
            {
                witnessed_nontrivial_tightening = true;
            }

            for _sample in 0..160 {
                let point: Vec<f32> = (0..dimension)
                    .map(|d| rng.random_range(lower[d]..upper[d]))
                    .collect();
                let mut offset = 0usize;
                let mut satisfies_any_clause = false;
                for &size in &clause_sizes {
                    let clause_satisfied = (offset..offset + size).all(|row| {
                        let value = bias[row] as f64
                            + (0..dimension)
                                .map(|d| true_a[[row, d]] as f64 * point[d] as f64)
                                .sum::<f64>();
                        value <= thresholds[row] as f64 - 1.0e-5
                    });
                    if clause_satisfied {
                        satisfies_any_clause = true;
                        break;
                    }
                    offset += size;
                }
                if satisfies_any_clause {
                    assert_point_in_box(&point, &result.bounds, 2.0e-3);
                }
            }
        }

        assert!(
            witnessed_nontrivial_tightening,
            "oracle must exercise a genuinely clipped union box"
        );
    }
}
