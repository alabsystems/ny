// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Authority token for verdict-adjacent Complete Clipping affine rows.
//!
//! This module intentionally exposes no production constructor.  The current
//! CUDA resident-coefficient object is a public suggestion made of raw vectors;
//! accepting it here would allow a backend/layout bug to mint clipping authority.
//! A later milestone must make the sound CROWN output boundary return the opaque
//! pass stamp and this token together, after its independent enclosure oracle.
//! Until then the quarantined caller can only present `None` and inherits bounds.

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use crate::beta_crown::branching::GraphSplitHistory;

use super::{check_clip_deadline, validate_clip_work_budget};

const PROVENANCE_MAX_CONTEXT_BYTES: usize = 16 * 1024 * 1024;
const PROVENANCE_POLL_STRIDE: usize = 1024;

/// Opaque identity of one sound CROWN backward invocation.
///
/// The private representation prevents a raw coefficient producer from choosing
/// a pass identity.  Exact-domain equality alone is insufficient for model/node
/// freshness because two networks can use the same node names and boxes.
/// Deliberately neither `Clone` nor `Copy`: each sound backward invocation must
/// receive a freshly minted handle from the future sealed CROWN boundary.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CrownPassStamp {
    words: [u64; 2],
    model_identity: Arc<[u8]>,
}

/// Affine enclosure whose origin and outward error are bound to one exact CROWN
/// pass, input box, split history, node, and selected-objective layout.
///
/// All fields are private and there is no production raw-parts constructor.  In
/// particular, `GpuResidentCoeffBatched`, CUDA trajectory captures, and plain
/// `ndarray` values cannot be converted into authority by this module today.
#[derive(Debug)]
pub(crate) struct CertifiedAffineEnclosure {
    pass_words: [u64; 2],
    model_identity: Arc<[u8]>,
    input_lower_bits: Arc<[u32]>,
    input_upper_bits: Arc<[u32]>,
    history_identity: Arc<[u8]>,
    node_name: Arc<str>,
    selected_neurons: Arc<[usize]>,
    row_of_neuron: Arc<[usize]>,
    lower_a: Array2<f32>,
    upper_a: Array2<f32>,
    lower_a_error: Array2<f32>,
    upper_a_error: Array2<f32>,
    lower_bias_center: Array1<f32>,
    upper_bias_center: Array1<f32>,
    lower_bias_error: Array1<f32>,
    upper_bias_error: Array1<f32>,
}

/// Rows admitted after exact-context validation and outward error discharge.
/// This is the only affine-row type the quarantined clip compatibility seam may
/// consume.  It cannot be constructed outside this module.
#[derive(Debug)]
pub(crate) struct ValidatedAffineEnclosure {
    lower_a: Array2<f32>,
    upper_a: Array2<f32>,
    lower_b: Array1<f32>,
    upper_b: Array1<f32>,
}

impl ValidatedAffineEnclosure {
    pub(crate) fn dim(&self) -> usize {
        self.lower_a.ncols()
    }

    pub(crate) fn rows(&self) -> usize {
        self.lower_a.nrows()
    }

    pub(crate) fn lower_a(&self) -> &Array2<f32> {
        &self.lower_a
    }

    pub(crate) fn upper_a(&self) -> &Array2<f32> {
        &self.upper_a
    }

    pub(crate) fn lower_b(&self) -> &Array1<f32> {
        &self.lower_b
    }

    pub(crate) fn upper_b(&self) -> &Array1<f32> {
        &self.upper_b
    }
}

impl CertifiedAffineEnclosure {
    /// Validate exact provenance and discharge every stored coefficient/bias
    /// error outward over the exact current input box.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_for_clip(
        &self,
        pass: &CrownPassStamp,
        input_lower: &[f32],
        input_upper: &[f32],
        history: &GraphSplitHistory,
        node_name: &str,
        selected_neurons: &[usize],
        row_of_neuron: &[usize],
        deadline: Option<Instant>,
    ) -> Result<ValidatedAffineEnclosure> {
        check_clip_deadline(deadline, "affine provenance entry")?;
        if self.pass_words != pass.words
            || self.model_identity.as_ref() != pass.model_identity.as_ref()
        {
            return Err(invalid("stale CROWN pass stamp"));
        }
        let n_rows = self.lower_a.nrows();
        let x_dim = self.lower_a.ncols();
        validate_clip_work_budget(1, n_rows, 0, x_dim)?;
        validate_context_size(
            pass.model_identity.as_ref(),
            node_name,
            selected_neurons,
            row_of_neuron,
        )?;
        validate_payload_shapes_and_values(self)?;
        validate_subject_layout(selected_neurons, row_of_neuron, n_rows)?;

        if self.node_name.as_ref() != node_name
            || self.selected_neurons.as_ref() != selected_neurons
            || self.row_of_neuron.as_ref() != row_of_neuron
        {
            return Err(invalid("node/objective identity mismatch"));
        }
        if input_lower.len() != x_dim
            || input_upper.len() != x_dim
            || self.input_lower_bits.len() != x_dim
            || self.input_upper_bits.len() != x_dim
        {
            return Err(invalid("input domain shape mismatch"));
        }
        for j in 0..x_dim {
            if j.is_multiple_of(PROVENANCE_POLL_STRIDE) {
                check_clip_deadline(deadline, "affine provenance input identity")?;
            }
            let lower = input_lower[j];
            let upper = input_upper[j];
            if !lower.is_finite()
                || !upper.is_finite()
                || lower > upper
                || lower.to_bits() != self.input_lower_bits[j]
                || upper.to_bits() != self.input_upper_bits[j]
            {
                return Err(invalid("input domain identity mismatch"));
            }
        }
        let current_history = history
            .exact_provenance_identity()
            .ok_or_else(|| invalid("split-history identity resource refusal"))?;
        check_clip_deadline(deadline, "affine provenance history identity")?;
        if current_history.as_slice() != self.history_identity.as_ref() {
            return Err(invalid("split-history identity mismatch"));
        }

        let mut lower_a = Array2::<f32>::zeros((n_rows, x_dim));
        check_clip_deadline(deadline, "affine provenance upper allocation")?;
        let mut upper_a = Array2::<f32>::zeros((n_rows, x_dim));
        check_clip_deadline(deadline, "affine provenance bias allocation")?;
        let mut lower_b = Array1::<f32>::zeros(n_rows);
        let mut upper_b = Array1::<f32>::zeros(n_rows);

        let mut cells = 0usize;
        for row in 0..n_rows {
            check_clip_deadline(deadline, "affine provenance row")?;
            let mut lower_penalty = 0.0f64;
            let mut upper_penalty = 0.0f64;
            for j in 0..x_dim {
                if cells.is_multiple_of(PROVENANCE_POLL_STRIDE) {
                    check_clip_deadline(deadline, "affine provenance error fold")?;
                }
                lower_a[[row, j]] = self.lower_a[[row, j]];
                upper_a[[row, j]] = self.upper_a[[row, j]];
                let magnitude = f64::from(input_lower[j].abs().max(input_upper[j].abs()));
                lower_penalty = add_up(
                    lower_penalty,
                    mul_up(f64::from(self.lower_a_error[[row, j]]), magnitude),
                );
                upper_penalty = add_up(
                    upper_penalty,
                    mul_up(f64::from(self.upper_a_error[[row, j]]), magnitude),
                );
                cells = cells.saturating_add(1);
            }
            let lower_error = self.lower_bias_error[row];
            let upper_error = self.upper_bias_error[row];
            // Exact-zero errors need no arithmetic and preserve an already
            // outward source row bit-for-bit. This is both sound and required
            // for the legacy proposal compatibility comparison.
            let lower = if lower_error == 0.0 && lower_penalty == 0.0 {
                self.lower_bias_center[row]
            } else {
                to_f32_down(sub_down(
                    sub_down(
                        f64::from(self.lower_bias_center[row]),
                        f64::from(lower_error),
                    ),
                    lower_penalty,
                ))
            };
            let upper = if upper_error == 0.0 && upper_penalty == 0.0 {
                self.upper_bias_center[row]
            } else {
                to_f32_up(add_up(
                    add_up(
                        f64::from(self.upper_bias_center[row]),
                        f64::from(upper_error),
                    ),
                    upper_penalty,
                ))
            };
            if !lower.is_finite() || !upper.is_finite() {
                return Err(invalid("non-finite outward affine bias"));
            }
            lower_b[row] = lower;
            upper_b[row] = upper;
        }
        check_clip_deadline(deadline, "affine provenance return")?;
        Ok(ValidatedAffineEnclosure {
            lower_a,
            upper_a,
            lower_b,
            upper_b,
        })
    }
}

fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("complete-clip affine provenance: {message}"))
}

fn validate_context_size(
    model_identity: &[u8],
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
) -> Result<()> {
    let bytes = model_identity
        .len()
        .checked_add(node_name.len())
        .ok_or_else(|| invalid("model/node context size overflow"))?
        .checked_add(
            selected_neurons
                .len()
                .checked_mul(size_of::<usize>())
                .ok_or_else(|| invalid("selected-neuron size overflow"))?,
        )
        .and_then(|n| {
            row_of_neuron
                .len()
                .checked_mul(size_of::<usize>())
                .and_then(|m| n.checked_add(m))
        })
        .ok_or_else(|| invalid("objective context size overflow"))?;
    if bytes > PROVENANCE_MAX_CONTEXT_BYTES {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: bytes,
            budget_bytes: PROVENANCE_MAX_CONTEXT_BYTES,
            site: "complete_clip_affine_provenance",
        });
    }
    Ok(())
}

fn validate_subject_layout(
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    n_rows: usize,
) -> Result<()> {
    if selected_neurons.len() != n_rows {
        return Err(invalid("selected-objective row count mismatch"));
    }
    for (row, &neuron) in selected_neurons.iter().enumerate() {
        if row_of_neuron.get(neuron).copied() != Some(row) {
            return Err(invalid("selected-objective row mapping mismatch"));
        }
    }
    for (neuron, &row) in row_of_neuron.iter().enumerate() {
        if row != usize::MAX
            && (row >= n_rows || selected_neurons.get(row).copied() != Some(neuron))
        {
            return Err(invalid("objective row mapping is not an exact inverse"));
        }
    }
    Ok(())
}

fn validate_payload_shapes_and_values(token: &CertifiedAffineEnclosure) -> Result<()> {
    let shape = token.lower_a.raw_dim();
    let n_rows = token.lower_a.nrows();
    if token.upper_a.raw_dim() != shape
        || token.lower_a_error.raw_dim() != shape
        || token.upper_a_error.raw_dim() != shape
        || token.lower_bias_center.len() != n_rows
        || token.upper_bias_center.len() != n_rows
        || token.lower_bias_error.len() != n_rows
        || token.upper_bias_error.len() != n_rows
    {
        return Err(invalid("affine payload shape mismatch"));
    }
    if token.lower_a.iter().any(|v| !v.is_finite())
        || token.upper_a.iter().any(|v| !v.is_finite())
        || token.lower_bias_center.iter().any(|v| !v.is_finite())
        || token.upper_bias_center.iter().any(|v| !v.is_finite())
        || token
            .lower_a_error
            .iter()
            .chain(token.upper_a_error.iter())
            .chain(token.lower_bias_error.iter())
            .chain(token.upper_bias_error.iter())
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(invalid("non-finite or negative affine error payload"));
    }
    Ok(())
}

fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        next_up_f64(a + b)
    }
}

fn sub_down(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a
    } else {
        next_down_f64(a - b)
    }
}

fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        next_up_f64(a * b)
    }
}

fn next_down_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn to_f32_down(value: f64) -> f32 {
    let candidate = value as f32;
    if f64::from(candidate) <= value {
        candidate
    } else {
        ny_tensor::next_down_f32(candidate)
    }
}

fn to_f32_up(value: f64) -> f32 {
    let candidate = value as f32;
    if f64::from(candidate) >= value {
        candidate
    } else {
        ny_tensor::next_up_f32(candidate)
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct TestSoundCrownAffineParts {
    pub(crate) lower_a: Array2<f32>,
    pub(crate) upper_a: Array2<f32>,
    pub(crate) lower_a_error: Array2<f32>,
    pub(crate) upper_a_error: Array2<f32>,
    pub(crate) lower_bias_center: Array1<f32>,
    pub(crate) upper_bias_center: Array1<f32>,
    pub(crate) lower_bias_error: Array1<f32>,
    pub(crate) upper_bias_error: Array1<f32>,
}

/// Test-only stand-in for the future sealed sound-CROWN output constructor.
/// Raw-parts construction does not exist in non-test builds.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn test_certified_affine_fixture(
    pass_words: [u64; 2],
    model_identity: &[u8],
    input_lower: &[f32],
    input_upper: &[f32],
    history: &GraphSplitHistory,
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    parts: TestSoundCrownAffineParts,
) -> Result<(CrownPassStamp, CertifiedAffineEnclosure)> {
    if model_identity.is_empty() || model_identity.len() > PROVENANCE_MAX_CONTEXT_BYTES {
        return Err(invalid("fixture model identity invalid or oversized"));
    }
    let model_identity: Arc<[u8]> = Arc::from(model_identity);
    let pass = CrownPassStamp {
        words: pass_words,
        model_identity: Arc::clone(&model_identity),
    };
    validate_clip_work_budget(1, parts.lower_a.nrows(), 0, parts.lower_a.ncols())?;
    validate_context_size(
        model_identity.as_ref(),
        node_name,
        selected_neurons,
        row_of_neuron,
    )?;
    validate_subject_layout(selected_neurons, row_of_neuron, parts.lower_a.nrows())?;
    if input_lower.len() != parts.lower_a.ncols() || input_upper.len() != parts.lower_a.ncols() {
        return Err(invalid("fixture input shape mismatch"));
    }
    for (&lower, &upper) in input_lower.iter().zip(input_upper) {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid("fixture input domain invalid"));
        }
    }
    let history_identity = history
        .exact_provenance_identity()
        .ok_or_else(|| invalid("fixture split-history identity refusal"))?;
    let token = CertifiedAffineEnclosure {
        pass_words,
        model_identity,
        input_lower_bits: input_lower
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
            .into(),
        input_upper_bits: input_upper
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
            .into(),
        history_identity: history_identity.into(),
        node_name: Arc::from(node_name),
        selected_neurons: Arc::from(selected_neurons),
        row_of_neuron: Arc::from(row_of_neuron),
        lower_a: parts.lower_a,
        upper_a: parts.upper_a,
        lower_a_error: parts.lower_a_error,
        upper_a_error: parts.upper_a_error,
        lower_bias_center: parts.lower_bias_center,
        upper_bias_center: parts.upper_bias_center,
        lower_bias_error: parts.lower_bias_error,
        upper_bias_error: parts.upper_bias_error,
    };
    validate_payload_shapes_and_values(&token)?;
    Ok((pass, token))
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};

    use super::*;

    fn history(active: bool) -> GraphSplitHistory {
        GraphSplitHistory::new().with_constraint(
            GraphNeuronConstraint::new("relu".into(), 0, active, 0.25).expect("constraint"),
        )
    }

    fn parts(error: f32) -> TestSoundCrownAffineParts {
        TestSoundCrownAffineParts {
            lower_a: arr2(&[[1.0, -2.0]]),
            upper_a: arr2(&[[1.0, -2.0]]),
            lower_a_error: arr2(&[[error, error]]),
            upper_a_error: arr2(&[[error, error]]),
            lower_bias_center: arr1(&[0.5]),
            upper_bias_center: arr1(&[0.5]),
            lower_bias_error: arr1(&[error]),
            upper_bias_error: arr1(&[error]),
        }
    }

    fn fixture(
        pass: [u64; 2],
        h: &GraphSplitHistory,
    ) -> Result<(CrownPassStamp, CertifiedAffineEnclosure)> {
        test_certified_affine_fixture(
            pass,
            b"model-a",
            &[-1.0, -2.0],
            &[1.0, 2.0],
            h,
            "relu",
            &[0],
            &[0],
            parts(0.125),
        )
    }

    #[test]
    fn exact_context_and_stale_pass_fail_closed() {
        let h = history(true);
        let (pass, token) = fixture([7, 11], &h).expect("fixture");
        assert!(token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_ok());
        let stale = CrownPassStamp {
            words: [7, 12],
            model_identity: Arc::from(&b"model-a"[..]),
        };
        assert!(token
            .validate_for_clip(
                &stale,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());
        // Reusing the exact same invocation words against another exact model
        // identity is still a stale cross-model pass and must fail closed.
        let other_model = CrownPassStamp {
            words: [7, 11],
            model_identity: Arc::from(&b"model-b"[..]),
        };
        assert!(token
            .validate_for_clip(
                &other_model,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());
        let score_changed = GraphSplitHistory::new().with_constraint(
            GraphNeuronConstraint::new(
                "relu".into(),
                0,
                true,
                f32::from_bits(0.25f32.to_bits() + 1),
            )
            .expect("constraint"),
        );
        assert!(token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &score_changed,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());
        assert!(token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.25],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());
        assert!(token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &history(false),
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());
        assert!(token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "other",
                &[0],
                &[0],
                None,
            )
            .is_err());
        assert!(token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[1],
                &[usize::MAX, 0],
                None,
            )
            .is_err());
    }

    #[test]
    fn errors_are_discharged_outward_and_invalid_errors_refuse() {
        let h = history(true);
        let (pass, token) = fixture([3, 5], &h).expect("fixture");
        let rows = token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .expect("validated");
        // penalty = .125*1 + .125*2 = .375, plus .125 bias error.
        assert!(rows.lower_b()[0] <= 0.0);
        assert!(rows.upper_b()[0] >= 1.0);

        assert!(test_certified_affine_fixture(
            [1, 1],
            b"model-a",
            &[-1.0, -2.0],
            &[1.0, 2.0],
            &h,
            "relu",
            &[0],
            &[0],
            parts(f32::NAN),
        )
        .is_err());
        assert!(test_certified_affine_fixture(
            [1, 2],
            b"model-a",
            &[-1.0, -2.0],
            &[1.0, 2.0],
            &h,
            "relu",
            &[0],
            &[0],
            parts(-0.125),
        )
        .is_err());
        assert!(test_certified_affine_fixture(
            [1, 3],
            b"model-a",
            &[-1.0, -2.0],
            &[1.0, 2.0],
            &h,
            "relu",
            &[0],
            &[0, 0],
            parts(0.0),
        )
        .is_err());
    }

    #[test]
    fn exact_zero_errors_preserve_source_rows_bit_for_bit() {
        let h = history(true);
        let source = parts(0.0);
        let expected_lower_a = source.lower_a.clone();
        let expected_upper_a = source.upper_a.clone();
        let expected_lower_b = source.lower_bias_center.clone();
        let expected_upper_b = source.upper_bias_center.clone();
        let (pass, token) = test_certified_affine_fixture(
            [19, 23],
            b"model-a",
            &[-1.0, -2.0],
            &[1.0, 2.0],
            &h,
            "relu",
            &[0],
            &[0],
            source,
        )
        .expect("fixture");
        let rows = token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .expect("validated");
        assert!(rows
            .lower_a()
            .iter()
            .zip(expected_lower_a.iter())
            .all(|(got, expected)| got.to_bits() == expected.to_bits()));
        assert!(rows
            .upper_a()
            .iter()
            .zip(expected_upper_a.iter())
            .all(|(got, expected)| got.to_bits() == expected.to_bits()));
        assert!(rows
            .lower_b()
            .iter()
            .zip(expected_lower_b.iter())
            .all(|(got, expected)| got.to_bits() == expected.to_bits()));
        assert!(rows
            .upper_b()
            .iter()
            .zip(expected_upper_b.iter())
            .all(|(got, expected)| got.to_bits() == expected.to_bits()));
    }

    #[test]
    fn expired_deadline_refuses_before_replay() {
        let h = history(true);
        let (pass, token) = fixture([9, 9], &h).expect("fixture");
        assert!(token
            .validate_for_clip(
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                Some(Instant::now()),
            )
            .is_err());
    }

    #[test]
    fn signed_zero_input_bits_and_context_resource_cap_are_exact() {
        let h = history(true);
        let (pass, token) = test_certified_affine_fixture(
            [13, 17],
            b"model-a",
            &[-0.0, -2.0],
            &[1.0, 2.0],
            &h,
            "relu",
            &[0],
            &[0],
            parts(0.0),
        )
        .expect("fixture");
        assert!(token
            .validate_for_clip(
                &pass,
                &[0.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());

        let entries = PROVENANCE_MAX_CONTEXT_BYTES / size_of::<usize>() + 1;
        let oversized = vec![usize::MAX; entries];
        assert!(matches!(
            validate_context_size(b"model-a", "relu", &[], &oversized),
            Err(NyError::CpuMemoryExceeded {
                site: "complete_clip_affine_provenance",
                ..
            })
        ));
    }
}
