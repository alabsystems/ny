// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Noncertifying concrete Linear evaluation. Interval padding is never an
//! inference operation: its midpoint can move at binary32 binade boundaries.

use std::time::Instant;

use faer::Mat;
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use serde::Serialize;
use tracing::debug;

use super::{bias::f32_to_f64_exact, LinearLayer};

const POINT_POLL_OPS: usize = 4_096;
// Match the existing finite-deadline Linear IBP endpoint admission ceiling.
const POINT_MAX_ELEMENTS: usize = 4 * 1024 * 1024;
const POINT_MAX_RANK: usize = 1_024;

/// The completed diagnostic contraction, not a kernel qualification or proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearPointArm {
    ScalarF64,
    CpuFaerF32,
    EngineF32,
    CpuFaerF32AfterEngineRefusal,
}

impl LinearPointArm {
    pub fn identity_name(self) -> &'static str {
        match self {
            Self::ScalarF64 => "linear-scalar-f64-one-f32-store/v1",
            Self::CpuFaerF32 => "linear-faer-f32-original-weight/v1",
            Self::EngineF32 => "linear-engine-f32-original-weight/v1",
            Self::CpuFaerF32AfterEngineRefusal => "linear-faer-f32-after-engine-refusal/v1",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LinearPointReceipt {
    pub arm: LinearPointArm,
    /// Software provenance supplied by the actual successful backend. The
    /// engine trait does not identify hardware, driver or deployed kernel.
    pub backend: &'static str,
    pub declined_backend: Option<&'static str>,
}

fn checkpoint(deadline: Option<Instant>, stage: &str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "Linear concrete point: deadline exceeded {stage}"
        )));
    }
    Ok(())
}

impl LinearLayer {
    /// Evaluate a finite, degenerate input with the original `W` and bias.
    ///
    /// This returns an approximate point observation, NEVER a certified
    /// enclosure of the mathematical result or a particular inference kernel.
    /// Ordinary CPU evaluation uses one binary32 faer contraction; an optional
    /// engine uses its binary32 GEMM. Under finite deadline authority, a direct
    /// binary64 contraction polls bounded work chunks and narrows once per
    /// output. These diagnostic arithmetic choices do not assert Torch parity.
    ///
    /// Every interval/verdict caller must continue to use `propagate_ibp` or its
    /// certified variants. In particular, a degenerate input does not authorize
    /// substituting this routine into an interval propagation path.
    pub fn propagate_concrete_point_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.propagate_concrete_point_with_receipt(input, engine, deadline)
            .map(|(point, _)| point)
    }

    /// The compatible point evaluator with an actual completed-arm receipt.
    /// A failed/nonfinite/deadline result never produces a receipt.
    pub fn propagate_concrete_point_with_receipt(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(BoundedTensor, LinearPointReceipt)> {
        checkpoint(deadline, "before admission")?;
        let shape = input.shape();
        if shape.is_empty() || shape.len() > POINT_MAX_RANK || shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "Linear concrete point: invalid input rank or empty dimension".into(),
            ));
        }
        let inner = *shape.last().expect("nonempty shape admitted");
        if inner != self.in_features() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.in_features()],
                got: vec![inner],
            });
        }
        let elements = checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec("Linear concrete point input size overflow".into())
        })?;
        let rows = elements / inner;
        let outputs = rows.checked_mul(self.out_features()).ok_or_else(|| {
            NyError::InvalidSpec("Linear concrete point output size overflow".into())
        })?;
        if deadline.is_some() && (elements > POINT_MAX_ELEMENTS || outputs > POINT_MAX_ELEMENTS) {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: elements.max(outputs).saturating_mul(4),
                budget_bytes: POINT_MAX_ELEMENTS * 4,
                site: "Linear finite-deadline concrete point buffer",
            });
        }
        if deadline.is_none() && engine.is_some_and(|e| e.forbids_unbounded_cpu_fallback()) {
            return Err(NyError::UnsupportedOp(
                "bounded Linear point requires the explicit deadline-aware entry".into(),
            ));
        }
        for (index, (&lo, &hi)) in input.lower().iter().zip(input.upper().iter()).enumerate() {
            if index.is_multiple_of(POINT_POLL_OPS) {
                checkpoint(deadline, "while validating input")?;
            }
            if !lo.is_finite() || lo != hi {
                return Err(NyError::InvalidSpec(
                    "Linear concrete point requires finite degenerate input".into(),
                ));
            }
        }
        for (index, &coefficient) in self
            .weight()
            .iter()
            .chain(self.bias().into_iter().flat_map(|bias| bias.iter()))
            .enumerate()
        {
            if index.is_multiple_of(POINT_POLL_OPS) {
                checkpoint(deadline, "while validating coefficients")?;
            }
            if !coefficient.is_finite() {
                return Err(NyError::NumericalInstability(
                    "Linear concrete point: nonfinite coefficient".into(),
                ));
            }
        }
        checkpoint(deadline, "before input allocation")?;
        let mut values = Vec::new();
        values.try_reserve_exact(elements).map_err(|error| {
            NyError::InvalidSpec(format!("Linear concrete point input allocation: {error}"))
        })?;
        // ndarray iteration follows logical row-major order even for strided
        // point tensors; no layout assumptions escape this flattening boundary.
        for (index, &value) in input.lower().iter().enumerate() {
            if index.is_multiple_of(POINT_POLL_OPS) {
                checkpoint(deadline, "while flattening input")?;
            }
            values.push(value);
        }
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features());
        let (output, receipt) = if deadline.is_some() {
            let mut result = Vec::new();
            result.try_reserve_exact(outputs).map_err(|error| {
                NyError::InvalidSpec(format!("Linear concrete point output allocation: {error}"))
            })?;
            checkpoint(deadline, "after output allocation")?;
            let mut operations = 0usize;
            for row in 0..rows {
                for out in 0..self.out_features() {
                    // Bias is added after contraction, as on the binary32
                    // route. Exact source decoding avoids binary32 DAZ casts.
                    let mut sum = 0.0_f64;
                    for column in 0..inner {
                        if operations == POINT_POLL_OPS {
                            checkpoint(deadline, "during direct contraction")?;
                            operations = 0;
                        }
                        operations += 1;
                        sum += f32_to_f64_exact(self.weight[[out, column]])
                            * f32_to_f64_exact(values[row * inner + column]);
                    }
                    let bias = self.bias().map_or(0.0, |bias| f32_to_f64_exact(bias[out]));
                    result.push((sum + bias) as f32);
                }
            }
            (
                result,
                LinearPointReceipt {
                    arm: LinearPointArm::ScalarF64,
                    backend: "ny-cpu-scalar-f64",
                    declined_backend: None,
                },
            )
        } else if let Some(engine) = engine {
            match engine.gemm_f32(
                rows,
                inner,
                self.out_features(),
                &values,
                self.weight_t_row_major(),
            ) {
                Ok(mut result)
                    if result.len() == outputs && result.iter().all(|v| v.is_finite()) =>
                {
                    if let Some(bias) = self.bias() {
                        for row in 0..rows {
                            for out in 0..self.out_features() {
                                result[row * self.out_features() + out] += bias[out];
                            }
                        }
                    }
                    (
                        result,
                        LinearPointReceipt {
                            arm: LinearPointArm::EngineF32,
                            backend: engine.backend_provenance(),
                            declined_backend: None,
                        },
                    )
                }
                _ => {
                    debug!("Linear concrete GEMM declined or returned malformed output; using CPU point contraction");
                    (
                        self.point_faer(&values, rows, inner),
                        LinearPointReceipt {
                            arm: LinearPointArm::CpuFaerF32AfterEngineRefusal,
                            backend: "ny-cpu-faer-f32",
                            declined_backend: Some(engine.backend_provenance()),
                        },
                    )
                }
            }
        } else {
            (
                self.point_faer(&values, rows, inner),
                LinearPointReceipt {
                    arm: LinearPointArm::CpuFaerF32,
                    backend: "ny-cpu-faer-f32",
                    declined_backend: None,
                },
            )
        };
        if output.len() != outputs {
            return Err(NyError::NumericalInstability(
                "Linear concrete point: malformed output".into(),
            ));
        }
        for (index, value) in output.iter().enumerate() {
            if index.is_multiple_of(POINT_POLL_OPS) {
                checkpoint(deadline, "while validating output")?;
            }
            if !value.is_finite() {
                return Err(NyError::NumericalInstability(
                    "Linear concrete point: nonfinite output".into(),
                ));
            }
        }
        checkpoint(deadline, "before result publication")?;
        let array = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
            .map_err(|error| NyError::InvalidSpec(format!("Linear point shape: {error}")))?;
        let point = if deadline.is_some() {
            BoundedTensor::concrete_with_poll(array, || {
                checkpoint(deadline, "while publishing output")
            })
        } else {
            BoundedTensor::concrete(array)
        }?;
        Ok((point, receipt))
    }

    fn point_faer(&self, values: &[f32], rows: usize, inner: usize) -> Vec<f32> {
        // Use the existing immutable original-W matrix on the left so neither
        // a new coefficient matrix nor a retained transpose is needed.
        let input = Mat::from_fn(inner, rows, |column, row| values[row * inner + column]);
        let product = crate::faer_parallelism::mat_mul(&self.weight_faer, &input);
        let mut output = Vec::with_capacity(rows * self.out_features());
        for row in 0..rows {
            for out in 0..self.out_features() {
                let value = product[(out, row)];
                output.push(self.bias().map_or(value, |bias| value + bias[out]));
            }
        }
        output
    }
}

#[cfg(test)]
mod tests;
