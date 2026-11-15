// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original coefficients for a diagnostic point-only affine layer.
//!
//! This representation has no signed, transposed, spectral or CROWN caches.
//! Unsupported interval/certificate calls refuse explicitly. The private owner
//! retains its existing shared ledger reservation across every layer clone.
//! Its digest identifies coefficients; it is not arithmetic proof authority.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayD, ArrayView2, ArrayViewD, IxDyn};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use serde::Serialize;
use sha2::{Digest as _, Sha256};

use crate::certify::scope::Digest;
use crate::certify::work::{PersistentReservation, WorkBudget, WorkRequest};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

const POLL_TERMS: usize = 4_096;
const OWNER_EXTRA_BYTES: usize = 1_024;
const CONSTRUCTION_SCRATCH_BYTES: usize = 1_024;
const POINT_MAX_RANK: usize = 1_024;
const FINITE_POINT_MAX_ELEMENTS: usize = 4 * 1024 * 1024;

/// Completed original-coefficient diagnostic arm, not a certified kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PointLinearArm {
    ScalarF64OnceF32,
    OriginalRowMajorFaerMatvecF32,
}
impl PointLinearArm {
    pub fn identity_name(self) -> &'static str {
        match self {
            Self::ScalarF64OnceF32 => "point-linear-scalar-f64-one-f32-store/v1",
            Self::OriginalRowMajorFaerMatvecF32 => {
                "point-linear-original-row-major-faer-matvec-f32/v1"
            }
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PointLinearReceipt {
    pub arm: PointLinearArm,
    pub backend: &'static str,
}

/// Caller limits narrow an existing budget; this creates no allowance.
#[derive(Clone, Copy, Debug)]
pub struct PointLinearBuildPolicy<'a> {
    pub budget: &'a WorkBudget,
    pub deadline: Option<Instant>,
    pub max_resident_bytes: usize,
    pub max_scratch_bytes: usize,
    pub max_work: u64,
}

#[derive(Debug)]
struct OriginalOwner {
    weight: Array2<f32>,
    bias: Option<Array1<f32>>,
    digest: Digest,
    coefficient_terms: usize,
    capacity_bytes: usize,
    reservation: PersistentReservation,
    budget: WorkBudget,
    deadline: Option<Instant>,
    max_scratch_bytes: usize,
    max_work: u64,
}

/// Immutable original binary32 affine coefficients with point-only authority.
///
/// Cloning shares the actual coefficient owner and its reservation. Replacing
/// parameters constructs another owner; an older clone retains the old values
/// and lease. No empty derived matrix can be mistaken for valid weights.
#[derive(Clone, Debug)]
pub struct PointLinearLayer(Arc<OriginalOwner>);

fn overflow() -> NyError {
    NyError::InvalidSpec("PointLinear size/work overflow".into())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn budget_error(error: crate::certify::work::WorkRefusal) -> NyError {
    match error {
        crate::certify::work::WorkRefusal::Deadline => {
            NyError::DeadlineExceeded("PointLinear original budget deadline".into())
        }
        _ => NyError::UnsupportedConfiguration(format!(
            "PointLinear original shared budget refused: {error}"
        )),
    }
}
fn deadline_checkpoint(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "PointLinear explicit deadline".into(),
        ));
    }
    #[cfg(test)]
    construction_poll_for_test()?;
    Ok(())
}

impl PointLinearLayer {
    /// Pre-copy capacity and work, before any derived matrix is constructed.
    /// Work units are original-coefficient visits: copy, finite check and hash.
    /// Capacity includes both vectors, the Arc/private owner and reservation
    /// controls. Allocator slack/RSS are not inferred from requested capacity.
    pub fn construction_request(out: usize, inner: usize, bias: bool) -> Result<WorkRequest> {
        if out == 0 || inner == 0 {
            return Err(NyError::InvalidSpec(
                "PointLinear requires nonempty affine dimensions".into(),
            ));
        }
        let terms = add(mul(out, inner)?, if bias { out } else { 0 })?;
        let bytes = add(
            mul(terms, 4)?,
            add(size_of::<OriginalOwner>(), OWNER_EXTRA_BYTES)?,
        )?;
        let work = u64::try_from(add(mul(terms, 3)?, 32)?).map_err(|_| overflow())?;
        Ok(WorkRequest {
            scratch_bytes: CONSTRUCTION_SCRATCH_BYTES,
            persistent_bytes: bytes,
            work,
            proposals: 1,
        })
    }

    /// Canonical-copy a borrowed original matrix under the existing ledger.
    ///
    /// `transpose` applies the declared original affine orientation. Scalar,
    /// length-one, row-vector and output-vector biases use the same ONNX Gemm
    /// broadcast shapes as the eager builder. Strided source values are read
    /// logically into one C-layout owner; no clone-then-transpose overlap occurs.
    pub fn from_original_view(
        weight: ArrayView2<'_, f32>,
        transpose: bool,
        bias: Option<ArrayViewD<'_, f32>>,
        policy: PointLinearBuildPolicy<'_>,
    ) -> Result<Self> {
        deadline_checkpoint(policy.deadline)?;
        policy.budget.checkpoint().map_err(budget_error)?;
        let (out, inner) = if transpose {
            (weight.ncols(), weight.nrows())
        } else {
            weight.dim()
        };
        let bias_scalar = match bias.as_ref() {
            None => None,
            Some(b) => match b.shape() {
                [] if b.len() == 1 => Some(b[[]]),
                [one] if *one == 1 => Some(b[[0]]),
                [one, one_b] if *one == 1 && *one_b == 1 => Some(b[[0, 0]]),
                [features] if *features == out => None,
                [one, features] if *one == 1 && *features == out => None,
                _ => {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![out],
                        got: b.shape().to_vec(),
                    })
                }
            },
        };
        let request = Self::construction_request(out, inner, bias.is_some())?;
        if request.persistent_bytes > policy.max_resident_bytes
            || request.scratch_bytes > policy.max_scratch_bytes
            || request.work > policy.max_work
        {
            return Err(NyError::UnsupportedConfiguration(
                "PointLinear pre-copy local capacity/work refusal".into(),
            ));
        }
        // Admission occurs before either output vector is allocated.
        let mut lease = policy.budget.lease(request).map_err(budget_error)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(mul(out, inner)?)
            .map_err(|e| NyError::InvalidSpec(format!("PointLinear original allocation: {e}")))?;
        lease.checkpoint().map_err(budget_error)?;
        deadline_checkpoint(policy.deadline)?;
        let mut hash = Sha256::new();
        hash.update(b"ny-point-linear-original/v1");
        hash.update(u64::try_from(out).map_err(|_| overflow())?.to_le_bytes());
        hash.update(u64::try_from(inner).map_err(|_| overflow())?.to_le_bytes());
        hash.update([u8::from(bias.is_some())]);
        for row in 0..out {
            for column in 0..inner {
                if values.len().is_multiple_of(POLL_TERMS) {
                    lease.checkpoint().map_err(budget_error)?;
                    deadline_checkpoint(policy.deadline)?;
                }
                let value = if transpose {
                    weight[[column, row]]
                } else {
                    weight[[row, column]]
                };
                if !value.is_finite() {
                    return Err(NyError::NumericalInstability(
                        "PointLinear nonfinite original weight".into(),
                    ));
                }
                hash.update(value.to_bits().to_le_bytes());
                values.push(value);
            }
        }
        let mut bias_values = Vec::new();
        if let Some(b) = &bias {
            lease.checkpoint().map_err(budget_error)?;
            deadline_checkpoint(policy.deadline)?;
            bias_values.try_reserve_exact(out).map_err(|e| {
                NyError::InvalidSpec(format!("PointLinear original bias allocation: {e}"))
            })?;
            for index in 0..out {
                if index.is_multiple_of(POLL_TERMS) {
                    lease.checkpoint().map_err(budget_error)?;
                    deadline_checkpoint(policy.deadline)?;
                }
                let value = if let Some(value) = bias_scalar {
                    value
                } else if b.ndim() == 1 {
                    b[[index]]
                } else {
                    b[[0, index]]
                };
                if !value.is_finite() {
                    return Err(NyError::NumericalInstability(
                        "PointLinear nonfinite original bias".into(),
                    ));
                }
                hash.update(value.to_bits().to_le_bytes());
                bias_values.push(value);
            }
        }
        let capacity_bytes = add(
            mul(add(values.capacity(), bias_values.capacity())?, 4)?,
            add(size_of::<OriginalOwner>(), OWNER_EXTRA_BYTES)?,
        )?;
        if capacity_bytes > request.persistent_bytes {
            return Err(NyError::UnsupportedConfiguration(
                "PointLinear actual capacity exceeded pre-copy reservation".into(),
            ));
        }
        let weight = Array2::from_shape_vec((out, inner), values)
            .map_err(|e| NyError::InvalidSpec(format!("PointLinear original shape: {e}")))?;
        let bias = bias.map(|_| Array1::from_vec(bias_values));
        let coefficient_terms = add(mul(out, inner)?, bias.as_ref().map_or(0, |b| b.len()))?;
        let digest = Digest(hash.finalize().into());
        deadline_checkpoint(policy.deadline)?;
        lease.complete().map_err(budget_error)?;
        let inherited_deadline = earlier(policy.deadline, lease.deadline());
        let reservation = lease.commit_persistent().map_err(budget_error)?;
        // The fixed owner/control allowance also covers this final Arc allocation.
        let result = Self(Arc::new(OriginalOwner {
            weight,
            bias,
            digest,
            coefficient_terms,
            capacity_bytes,
            reservation,
            budget: policy.budget.clone(),
            deadline: inherited_deadline,
            max_scratch_bytes: policy.max_scratch_bytes,
            max_work: policy.max_work,
        }));
        result.point_checkpoint(inherited_deadline)?;
        Ok(result)
    }

    pub fn in_features(&self) -> usize {
        self.0.weight.ncols()
    }
    pub fn out_features(&self) -> usize {
        self.0.weight.nrows()
    }
    pub fn weight(&self) -> &Array2<f32> {
        &self.0.weight
    }
    pub fn bias(&self) -> Option<&Array1<f32>> {
        self.0.bias.as_ref()
    }
    pub fn original_digest(&self) -> Digest {
        self.0.digest
    }
    pub fn coefficient_terms(&self) -> usize {
        self.0.coefficient_terms
    }
    /// Actual requested capacities plus the explicitly priced owner controls.
    /// One clone does not allocate another coefficient matrix or reservation.
    pub fn retained_bytes(&self) -> usize {
        self.0.capacity_bytes
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.reservation.bytes()
    }
    pub fn original_budget(&self) -> WorkBudget {
        self.0.budget.clone()
    }
    /// Immutable earliest construction/owner deadline. A proof-only adapter
    /// may narrow it further, but cannot extend this original owner's time.
    pub fn original_deadline(&self) -> Option<Instant> {
        self.0.deadline
    }

    /// Bounded original-view execution capacity for either supported CPU arm.
    ///
    /// Work units are products/sums and scalar input/output visits. Original
    /// coefficients were checked once by the private immutable constructor;
    /// execution does not scan a derived signed matrix or rehash them. The
    /// pinned faer row-major single-column matvec bypasses GEMM packing/TLS;
    /// the fixed cushion prices scalar/SIMD stack and shape metadata. This
    /// predicts requested buffers, not allocator/RSS overhead.
    pub fn execution_request(
        out: usize,
        inner: usize,
        rows: usize,
        bias: bool,
    ) -> Result<WorkRequest> {
        if out == 0 || inner == 0 || rows == 0 {
            return Err(NyError::InvalidSpec(
                "PointLinear nonempty execution dimensions required".into(),
            ));
        }
        let input = mul(rows, inner)?;
        let output = mul(rows, out)?;
        let scratch = add(add(mul(input, 4)?, mul(output, 16)?)?, 32 * 1024)?;
        let work = add(
            mul(mul(mul(rows, out)?, inner)?, 2)?,
            add(mul(input, 3)?, mul(output, if bias { 7 } else { 6 })?)?,
        )?;
        Ok(WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: 0,
            work: u64::try_from(add(work, POINT_MAX_RANK * 4)?).map_err(|_| overflow())?,
            proposals: 0,
        })
    }

    // Geometry-only preflight used by the original-owner execution path.
    // This returns a buffer/work prediction, never arithmetic authority or a
    // phase lease. The caller retains the original checkpoint and ledger.
    fn execution_request_with_limits(
        out: usize,
        inner: usize,
        rows: usize,
        bias: bool,
        deadline: Option<Instant>,
        max_scratch_bytes: usize,
        max_work: u64,
    ) -> Result<WorkRequest> {
        let request = Self::execution_request(out, inner, rows, bias)?;
        if request.scratch_bytes > max_scratch_bytes || request.work > max_work {
            return Err(NyError::UnsupportedConfiguration(
                "PointLinear diagnostic execution exceeds original local phase limits".into(),
            ));
        }
        let elements = mul(rows, inner)?.max(mul(rows, out)?);
        if deadline.is_some() && elements > FINITE_POINT_MAX_ELEMENTS {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: mul(elements, 4)?,
                budget_bytes: FINITE_POINT_MAX_ELEMENTS * 4,
                site: "PointLinear finite-deadline concrete point buffer",
            });
        }
        Ok(request)
    }

    /// Original-coefficient point evaluation inside caller-owned phase admission.
    ///
    /// This approximate observation cannot certify intervals or runtime parity.
    /// Managed decoder execution holds its aggregate phase lease through graph
    /// cache/result lifetimes. Raw callers must price execution and returned
    /// tensors externally, as with the eager diagnostic helper. This method
    /// creates no fresh budget, per-layer allowance or proof/resource receipt.
    /// The existing coefficient owner budget and earliest deadline still apply.
    pub fn propagate_concrete_point_with_receipt(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(BoundedTensor, PointLinearReceipt)> {
        let deadline = earlier(deadline, self.0.deadline);
        self.point_checkpoint(deadline)?;
        if engine.is_some() {
            return Err(NyError::UnsupportedOp(
                "PointLinear engine/device execution needs a separately priced original-view adapter"
                    .into(),
            ));
        }
        let shape = input.shape();
        if shape.is_empty() || shape.len() > POINT_MAX_RANK || shape.contains(&0) {
            return Err(NyError::InvalidSpec("PointLinear input shape".into()));
        }
        if shape.last().copied() != Some(self.in_features()) {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.in_features()],
                got: shape.to_vec(),
            });
        }
        let input_elements = checked_shape_product(shape).ok_or_else(overflow)?;
        let rows = input_elements / self.in_features();
        let output_elements = mul(rows, self.out_features())?;
        let _request = Self::execution_request_with_limits(
            self.out_features(),
            self.in_features(),
            rows,
            self.bias().is_some(),
            deadline,
            self.0.max_scratch_bytes,
            self.0.max_work,
        )?;
        // Validate the entire original point before allocating output buffers.
        for (index, (&lo, &hi)) in input.lower().iter().zip(input.upper().iter()).enumerate() {
            if index.is_multiple_of(POLL_TERMS) {
                self.point_checkpoint(deadline)?;
            }
            if !lo.is_finite() || lo != hi {
                return Err(NyError::InvalidSpec(
                    "PointLinear requires a finite degenerate input".into(),
                ));
            }
        }
        let mut values = Vec::new();
        values.try_reserve_exact(input_elements).map_err(|e| {
            NyError::InvalidSpec(format!("PointLinear point input allocation: {e}"))
        })?;
        self.point_checkpoint(deadline)?;
        for (index, &value) in input.lower().iter().enumerate() {
            if index.is_multiple_of(POLL_TERMS) {
                self.point_checkpoint(deadline)?;
            }
            values.push(value);
        }
        let mut output = Vec::new();
        output.try_reserve_exact(output_elements).map_err(|e| {
            NyError::InvalidSpec(format!("PointLinear point output allocation: {e}"))
        })?;
        self.point_checkpoint(deadline)?;
        let receipt = if deadline.is_some() {
            use super::linear::bias::f32_to_f64_exact;
            let mut terms = 0usize;
            for row in 0..rows {
                for out in 0..self.out_features() {
                    let mut sum = 0.0_f64;
                    for column in 0..self.in_features() {
                        if terms == POLL_TERMS {
                            self.point_checkpoint(deadline)?;
                            terms = 0;
                        }
                        terms += 1;
                        sum += f32_to_f64_exact(self.0.weight[[out, column]])
                            * f32_to_f64_exact(values[row * self.in_features() + column]);
                    }
                    let bias = self.bias().map_or(0.0, |b| f32_to_f64_exact(b[out]));
                    output.push((sum + bias) as f32);
                }
            }
            PointLinearReceipt {
                arm: PointLinearArm::ScalarF64OnceF32,
                backend: "ny-cpu-scalar-f64",
            }
        } else {
            // Pinned faer0.24 matmul_imp dispatches this row-major ncols=1
            // shape to matvec_rowmajor before the GEMM packing/TLS branch.
            // Explicit sequential SIMD matvec avoids worker-dependent scratch.
            // Reduction order can differ from eager column-major/GEMM; the
            // executed identity is therefore distinct, never a bit-parity claim.
            use faer::{Accum, MatMut, MatRef, Par};
            output.resize(output_elements, 0.0);
            let weight = MatRef::from_row_major_slice(
                self.0.weight.as_slice().expect("private canonical C owner"),
                self.out_features(),
                self.in_features(),
            );
            for row in 0..rows {
                self.point_checkpoint(deadline)?;
                let input = MatRef::from_column_major_slice(
                    &values[row * self.in_features()..(row + 1) * self.in_features()],
                    self.in_features(),
                    1,
                );
                let result =
                    &mut output[row * self.out_features()..(row + 1) * self.out_features()];
                let destination =
                    MatMut::from_column_major_slice_mut(result, self.out_features(), 1);
                faer::linalg::matmul::matmul(
                    destination,
                    Accum::Replace,
                    weight,
                    input,
                    1.0,
                    Par::Seq,
                );
                if let Some(bias) = self.bias() {
                    for (value, &bias) in result.iter_mut().zip(bias.iter()) {
                        *value += bias;
                    }
                }
                self.point_checkpoint(deadline)?;
            }
            PointLinearReceipt {
                arm: PointLinearArm::OriginalRowMajorFaerMatvecF32,
                backend: "ny-cpu-faer-f32-row-major-matvec-seq",
            }
        };
        for (index, &value) in output.iter().enumerate() {
            if index.is_multiple_of(POLL_TERMS) {
                self.point_checkpoint(deadline)?;
            }
            if !value.is_finite() {
                return Err(NyError::NumericalInstability(
                    "PointLinear nonfinite diagnostic output".into(),
                ));
            }
        }
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features());
        let array: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
            .map_err(|e| NyError::InvalidSpec(format!("PointLinear point output shape: {e}")))?;
        let point = BoundedTensor::concrete_with_poll(array, || self.point_checkpoint(deadline))?;
        self.point_checkpoint(deadline)?;
        Ok((point, receipt))
    }

    fn point_checkpoint(&self, deadline: Option<Instant>) -> Result<()> {
        self.0.budget.checkpoint().map_err(budget_error)?;
        deadline_checkpoint(deadline)
    }
}

fn earlier(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn unsupported() -> NyError {
    NyError::UnsupportedOp(
        "PointLinear is diagnostic point-only: interval/CROWN/proof consumers require a separately admitted and qualified derived view".into(),
    )
}
impl BoundPropagation for PointLinearLayer {
    fn propagate_ibp(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        Err(unsupported())
    }
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(unsupported())
    }
    fn propagate_linear_with_bounds(
        &self,
        _bounds: &LinearBounds,
        _pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        Err(unsupported())
    }
    fn propagate_crown_backward(
        &self,
        _bounds: &LinearBounds,
        _pre_activation: Option<&BoundedTensor>,
    ) -> Result<LinearBounds> {
        Err(unsupported())
    }
}

#[cfg(test)]
thread_local! {
    static CONSTRUCTION_POLLS_FOR_TEST: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
fn construction_poll_for_test() -> Result<()> {
    CONSTRUCTION_POLLS_FOR_TEST.with(|polls| match polls.get() {
        Some(0) => Err(NyError::DeadlineExceeded(
            "PointLinear forced cooperative construction deadline".into(),
        )),
        Some(n) => {
            polls.set(Some(n - 1));
            Ok(())
        }
        None => Ok(()),
    })
}

#[cfg(test)]
mod tests;
