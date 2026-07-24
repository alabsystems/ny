// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Specification helpers: bounds conversion, sanitization, and spec checking.

use super::Verifier;
use ndarray::ArrayD;
use ny_core::{
    checked_shape_product, Bound, MethodUsed, NyError, Result, SoundnessProvenance,
    VerificationResult,
};
use ny_tensor::BoundedTensor;

impl Verifier {
    pub fn bounds_to_tensor(bounds: &[Bound], shape: Option<&[usize]>) -> Result<BoundedTensor> {
        let lower: Vec<f32> = bounds.iter().map(|b| b.lower()).collect();
        let upper: Vec<f32> = bounds.iter().map(|b| b.upper()).collect();

        // Use provided shape or default to 1D
        let tensor_shape: Vec<usize> = match shape {
            Some(s) => {
                // Verify total elements match
                let total = checked_shape_product(s).ok_or_else(|| {
                    NyError::InvalidSpec(format!("Input shape {:?} overflows usize", s))
                })?;
                if total != bounds.len() {
                    return Err(NyError::InvalidSpec(format!(
                        "Input shape {:?} has {} elements but bounds has {}",
                        s,
                        total,
                        bounds.len()
                    )));
                }
                s.to_vec()
            }
            None => vec![bounds.len()],
        };

        BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&tensor_shape), lower)
                .map_err(|e| NyError::InvalidSpec(e.to_string()))?,
            ArrayD::from_shape_vec(ndarray::IxDyn(&tensor_shape), upper)
                .map_err(|e| NyError::InvalidSpec(e.to_string()))?,
        )
    }

    /// Sanitize output bounds: replace NaN values with conservative infinities.
    ///
    /// NaN bounds indicate numerical issues during propagation (e.g., inf + (-inf)).
    /// Sound verification requires converting these to conservative bounds:
    /// - NaN lower bound -> -inf (sound: any value >= -inf)
    /// - NaN upper bound -> +inf (sound: any value <= +inf)
    pub(super) fn sanitize_output_bounds(bounds: BoundedTensor) -> Result<BoundedTensor> {
        use ndarray::Zip;
        let (mut lower, mut upper) = bounds.into_parts();
        let mut nan_count = 0usize;
        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| {
            if l.is_nan() || u.is_nan() {
                nan_count += 1;
            }
            if l.is_nan() {
                *l = f32::NEG_INFINITY;
            }
            if u.is_nan() {
                *u = f32::INFINITY;
            }
            // Ensure lower <= upper (sanitize inverted bounds)
            if *l > *u {
                *l = f32::NEG_INFINITY;
                *u = f32::INFINITY;
            }
        });
        if nan_count > 0 {
            tracing::warn!(
                nan_count,
                total = lower.len(),
                "Output bounds contain NaN — numerical corruption in bound propagation. \
                 Replacing with conservative [-inf, +inf] (#2589)."
            );
        }
        // NaN replaced with ±Inf above; inverted bounds expanded to [-Inf, +Inf].
        BoundedTensor::new_allow_infinite(lower, upper).map_err(|e| {
            NyError::InternalError(format!(
                "sanitize_output_bounds failed to rebuild bounded tensor: {e}"
            ))
        })
    }

    pub(super) fn flatten_output_bounds(output: &BoundedTensor) -> Vec<Bound> {
        output
            .lower()
            .iter()
            .zip(output.upper().iter())
            .map(|(&l, &u)| Bound::new_allow_infinite(l, u))
            .collect()
    }

    /// Check computed output bounds against the spec's required bounds.
    ///
    /// `Verified` means exactly "every computed bound lies inside its required
    /// bound". When no requirement has a finite endpoint that containment is a
    /// vacuous truth — it attests that bound propagation completed, not that a
    /// property holds. The CLI standard path and the Python bindings submit
    /// exactly such specs for bounds-only runs and gate their property
    /// verdicts downstream (VNN-LIB property status / fold-to-Unknown), so the
    /// vacuous case is accepted here; any caller presenting `Verified` as a
    /// property claim must put at least one finite endpoint in its spec.
    pub(super) fn check_spec(
        &self,
        output: &BoundedTensor,
        required: &[Bound],
        actual_method: Option<MethodUsed>,
        provenance: SoundnessProvenance,
    ) -> Result<VerificationResult> {
        // Guard: empty output spec would trivially verify any network (#2238)
        if required.is_empty() {
            return Err(NyError::InvalidSpec(
                "empty output_bounds in specification — nothing to verify".to_string(),
            ));
        }

        // All-(-inf, +inf) specs are the same triviality class as the empty
        // spec, but they are the bounds-only sentinel the CLI and Python
        // front ends rely on (see doc comment) — note it rather than reject.
        if !required
            .iter()
            .any(|b| b.lower().is_finite() || b.upper().is_finite())
        {
            tracing::debug!(
                num_outputs = required.len(),
                "output spec has no finite constraint; Verified is vacuous \
                 (bounds computation only, no property checked)"
            );
        }

        let output_bounds = Self::flatten_output_bounds(output);

        // Guard: zip truncation would silently skip spec requirements (#2230)
        if output_bounds.len() < required.len() {
            return Err(NyError::InvalidSpec(format!(
                "Network produced {} output bounds but spec requires {}",
                output_bounds.len(),
                required.len()
            )));
        }

        // Check if computed bounds are within required bounds
        for (computed, req) in output_bounds.iter().zip(required.iter()) {
            if computed.lower() < req.lower() || computed.upper() > req.upper() {
                // Calculate the gap (how much the bound exceeds the requirement)
                let lower_gap = if computed.lower() < req.lower() {
                    req.lower() - computed.lower()
                } else {
                    0.0
                };
                let upper_gap = if computed.upper() > req.upper() {
                    computed.upper() - req.upper()
                } else {
                    0.0
                };
                let gap = lower_gap.max(upper_gap);

                return Ok(VerificationResult::Unknown {
                    provenance,
                    bounds: output_bounds.clone(),
                    reason: ny_core::UnknownReason::BoundsTooLoose {
                        gap: if gap > 0.0 { Some(gap) } else { None },
                    },
                    actual_method,
                });
            }
        }

        Ok(VerificationResult::Verified {
            provenance,
            output_bounds,
            proof: None,
            actual_method,
        })
    }
}
