// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Authority token for verdict-adjacent Complete Clipping affine rows.
//!
//! Raw accelerator coefficients are suggestions, never authority. Release
//! builds create reusable root rows only through a fixed scalar host CROWN
//! replay or exact selected network-input identities, then bind those immutable
//! rows to one exact child history. The raw host/suggestion dominance seams are
//! compiled only for adversarial tests. Only a sealed value can reach the
//! factory that mints a fresh pass stamp and [`CertifiedAffineEnclosure`].

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{NaiveCpuGemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::CutFoldScope;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::engine::graph::propagation::batched::backward_selected_input_relative_bounds_at_node;
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::{GraphNetwork, LinearBounds, NETWORK_INPUT};

use super::{check_clip_deadline, validate_clip_work_budget};

const PROVENANCE_MAX_CONTEXT_BYTES: usize = 16 * 1024 * 1024;
const PROVENANCE_POLL_STRIDE: usize = 1024;
static NEXT_CROWN_PASS: Mutex<[u64; 2]> = Mutex::new([0, 0]);

/// Opaque identity of one sound CROWN backward invocation.
///
/// The private representation prevents a raw coefficient producer from choosing
/// a pass identity.  Exact-domain equality alone is insufficient for model/node
/// freshness because two networks can use the same node names and boxes.
/// Deliberately neither `Clone` nor `Copy`: each sound backward invocation must
/// receive a freshly minted handle from the sealed CROWN boundary.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CrownPassStamp {
    words: [u64; 2],
    model_scope: CutFoldScope,
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
    model_scope: CutFoldScope,
    input_lower_bits: Arc<[u32]>,
    input_upper_bits: Arc<[u32]>,
    history_identity: Arc<[u8]>,
    node_name: Arc<str>,
    selected_neurons: Arc<[usize]>,
    row_of_neuron: Arc<[usize]>,
    lower_a: Arc<Array2<f32>>,
    upper_a: Arc<Array2<f32>>,
    lower_a_error: Arc<Array2<f32>>,
    upper_a_error: Arc<Array2<f32>>,
    lower_bias_center: Arc<Array1<f32>>,
    upper_bias_center: Arc<Array1<f32>>,
    lower_bias_error: Arc<Array1<f32>>,
    upper_bias_error: Arc<Array1<f32>>,
}

/// Untrusted affine-row suggestion, including certified numerical error claimed
/// by its producer.
///
/// This raw carrier grants no authority: there is no release conversion from
/// this type to [`SoundCrownAffineRows`] or [`CertifiedAffineEnclosure`]. It is
/// also used internally to discharge a host terminal's coefficient errors.
#[derive(Debug)]
pub(crate) struct UntrustedCrownAffineRows {
    lower_a: Array2<f32>,
    upper_a: Array2<f32>,
    lower_a_error: Array2<f32>,
    upper_a_error: Array2<f32>,
    lower_bias_center: Array1<f32>,
    upper_bias_center: Array1<f32>,
    lower_bias_error: Array1<f32>,
    upper_bias_error: Array1<f32>,
}

impl UntrustedCrownAffineRows {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        lower_a: Array2<f32>,
        upper_a: Array2<f32>,
        lower_a_error: Array2<f32>,
        upper_a_error: Array2<f32>,
        lower_bias_center: Array1<f32>,
        upper_bias_center: Array1<f32>,
        lower_bias_error: Array1<f32>,
        upper_bias_error: Array1<f32>,
    ) -> Self {
        Self {
            lower_a,
            upper_a,
            lower_a_error,
            upper_a_error,
            lower_bias_center,
            upper_bias_center,
            lower_bias_error,
            upper_bias_error,
        }
    }
}

/// History-scoped sound affine rows accepted by the pass factory.
///
/// Every field is private and there is intentionally no raw-parts constructor,
/// `From` implementation, `Clone`, or `Copy`. Release construction occurs only
/// by binding an immutable root template to an exact child history; the
/// host-vs-suggestion constructor exists only in tests.
#[derive(Debug)]
pub(crate) struct SoundCrownAffineRows {
    model_scope: CutFoldScope,
    input_lower_bits: Arc<[u32]>,
    input_upper_bits: Arc<[u32]>,
    history_identity: Arc<[u8]>,
    node_name: Arc<str>,
    selected_neurons: Arc<[usize]>,
    row_of_neuron: Arc<[usize]>,
    lower_a: Arc<Array2<f32>>,
    upper_a: Arc<Array2<f32>>,
    lower_b: Arc<Array1<f32>>,
    upper_b: Arc<Array1<f32>>,
    zero_a_error: Arc<Array2<f32>>,
    zero_bias_error: Arc<Array1<f32>>,
}

/// Opaque independently computed host sound-CROWN terminal rows.
///
/// This is intentionally distinct from both raw suggestions and downstream
/// [`ValidatedAffineEnclosure`] values.  Its fields are private, it is neither
/// `Clone` nor `Copy`, and exists only in provenance tests.
#[derive(Debug)]
#[cfg(test)]
pub(crate) struct HostSoundCrownAffineRows {
    model_scope: CutFoldScope,
    input_lower_bits: Arc<[u32]>,
    input_upper_bits: Arc<[u32]>,
    history_identity: Arc<[u8]>,
    node_name: Arc<str>,
    selected_neurons: Arc<[usize]>,
    row_of_neuron: Arc<[usize]>,
    lower_a: Array2<f32>,
    upper_a: Array2<f32>,
    lower_b: Array1<f32>,
    upper_b: Array1<f32>,
}

/// Opaque host CROWN terminal computed once at the root input box.
///
/// This private transient deliberately carries no split history. Release code
/// obtains it only from the fixed selected host replay or exact input identity,
/// then moves it directly into an immutable root template.
#[derive(Debug)]
pub(crate) struct HostSoundCrownRootAffineRows {
    model_scope: CutFoldScope,
    input_lower_bits: Arc<[u32]>,
    input_upper_bits: Arc<[u32]>,
    node_name: Arc<str>,
    selected_neurons: Arc<[usize]>,
    row_of_neuron: Arc<[usize]>,
    lower_a: Array2<f32>,
    upper_a: Array2<f32>,
    lower_b: Array1<f32>,
    upper_b: Array1<f32>,
}

/// Sound affine rows reusable as an immutable root template.
///
/// This value contains no child identity and exposes no mutable row access.
/// [`bind_root_sound_crown_rows_to_history`] borrows it and creates a
/// history-scoped [`SoundCrownAffineRows`] for each child. Release construction
/// is confined to [`capture_sound_crown_root_rows_at_node`] and
/// [`capture_exact_root_input_rows`].
#[derive(Debug)]
pub(crate) struct SoundCrownRootAffineRows {
    model_scope: CutFoldScope,
    input_lower_bits: Arc<[u32]>,
    input_upper_bits: Arc<[u32]>,
    node_name: Arc<str>,
    selected_neurons: Arc<[usize]>,
    row_of_neuron: Arc<[usize]>,
    lower_a: Arc<Array2<f32>>,
    upper_a: Arc<Array2<f32>>,
    lower_b: Arc<Array1<f32>>,
    upper_b: Arc<Array1<f32>>,
    zero_a_error: Arc<Array2<f32>>,
    zero_bias_error: Arc<Array1<f32>>,
}

/// Ephemeral capability passed only while the host terminal callback is
/// executing.  Its private fields prevent fabrication outside this module; its
/// engine accessor forces the independent replay onto the scalar CPU engine
/// rather than the accelerator that produced the suggestion.
pub(crate) struct HostSoundCrownTerminalCapability<'a> {
    cpu_engine: &'a NaiveCpuGemmEngine,
    deadline: Option<Instant>,
}

impl HostSoundCrownTerminalCapability<'_> {
    pub(crate) fn cpu_engine(&self) -> &NaiveCpuGemmEngine {
        self.cpu_engine
    }

    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

/// Rows admitted after exact-context validation and outward error discharge.
/// This is the only affine-row type the quarantined clip compatibility seam may
/// consume.  It cannot be constructed outside this module.
#[derive(Debug)]
pub(crate) struct ValidatedAffineEnclosure {
    lower_a: Arc<Array2<f32>>,
    upper_a: Arc<Array2<f32>>,
    lower_b: Arc<Array1<f32>>,
    upper_b: Arc<Array1<f32>>,
}

impl ValidatedAffineEnclosure {
    // Kept for the quarantined raw-proposal compatibility check. Production
    // authority does not enter that seam until its checker boundary is armed.
    #[allow(dead_code)]
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

#[cfg(test)]
impl HostSoundCrownAffineRows {
    #[allow(clippy::too_many_arguments)]
    fn matches_exact_context(
        &self,
        model_scope: CutFoldScope,
        input_lower: &[f32],
        input_upper: &[f32],
        history_identity: &[u8],
        node_name: &str,
        selected_neurons: &[usize],
        row_of_neuron: &[usize],
    ) -> bool {
        self.model_scope == model_scope
            && exact_f32_bits_match(input_lower, &self.input_lower_bits)
            && exact_f32_bits_match(input_upper, &self.input_upper_bits)
            && self.history_identity.as_ref() == history_identity
            && self.node_name.as_ref() == node_name
            && self.selected_neurons.as_ref() == selected_neurons
            && self.row_of_neuron.as_ref() == row_of_neuron
    }
}

impl HostSoundCrownRootAffineRows {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn matches_exact_root_context(
        &self,
        model_scope: CutFoldScope,
        input_lower: &[f32],
        input_upper: &[f32],
        node_name: &str,
        selected_neurons: &[usize],
        row_of_neuron: &[usize],
    ) -> bool {
        self.model_scope == model_scope
            && exact_f32_bits_match(input_lower, &self.input_lower_bits)
            && exact_f32_bits_match(input_upper, &self.input_upper_bits)
            && self.node_name.as_ref() == node_name
            && self.selected_neurons.as_ref() == selected_neurons
            && self.row_of_neuron.as_ref() == row_of_neuron
    }
}

impl SoundCrownRootAffineRows {
    /// Checked payload bytes retained by this immutable root template.
    ///
    /// This counts each root-owned dense allocation once, even though bound
    /// history tokens subsequently share those allocations through `Arc`.
    /// It also includes the variable-sized context payloads used to bind and
    /// validate the template. `Arc` control blocks, ndarray metadata, and any
    /// allocator capacity beyond the logical array lengths are intentionally
    /// excluded.
    pub(crate) fn resident_payload_bytes(&self) -> Option<usize> {
        let payloads = [
            (self.lower_a.len(), size_of::<f32>()),
            (self.upper_a.len(), size_of::<f32>()),
            (self.zero_a_error.len(), size_of::<f32>()),
            (self.lower_b.len(), size_of::<f32>()),
            (self.upper_b.len(), size_of::<f32>()),
            (self.zero_bias_error.len(), size_of::<f32>()),
            (self.input_lower_bits.len(), size_of::<u32>()),
            (self.input_upper_bits.len(), size_of::<u32>()),
            (self.node_name.len(), size_of::<u8>()),
            (self.selected_neurons.len(), size_of::<usize>()),
            (self.row_of_neuron.len(), size_of::<usize>()),
        ];
        let mut total = 0usize;
        for (len, element_bytes) in payloads {
            total = total.checked_add(len.checked_mul(element_bytes)?)?;
        }
        Some(total)
    }

    #[allow(clippy::too_many_arguments)]
    fn matches_exact_root_context(
        &self,
        model_scope: CutFoldScope,
        input_lower: &[f32],
        input_upper: &[f32],
        node_name: &str,
        selected_neurons: &[usize],
        row_of_neuron: &[usize],
    ) -> bool {
        self.model_scope == model_scope
            && exact_f32_bits_match(input_lower, &self.input_lower_bits)
            && exact_f32_bits_match(input_upper, &self.input_upper_bits)
            && self.node_name.as_ref() == node_name
            && self.selected_neurons.as_ref() == selected_neurons
            && self.row_of_neuron.as_ref() == row_of_neuron
    }
}

impl CertifiedAffineEnclosure {
    /// Compatibility face for the currently quarantined caller.
    ///
    /// Production authority must use [`Self::validate_for_clip_in_scope`] with
    /// the live graph's scope.  This legacy face can only compare the token with
    /// its paired stamp and therefore cannot by itself prevent moving the pair
    /// to another graph.  The only non-test call site currently supplies no
    /// token at all.
    //
    // Keep this compatibility face compiled while authority is quarantined so
    // its provenance tests continue to exercise the future integration seam.
    #[allow(dead_code)]
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
        self.validate_for_clip_in_scope(
            pass.model_scope,
            pass,
            input_lower,
            input_upper,
            history,
            node_name,
            selected_neurons,
            row_of_neuron,
            deadline,
        )
    }

    /// Validate exact provenance and discharge every stored coefficient/bias
    /// error outward over the exact current input box.  `live_model_scope` must
    /// be read from the graph being evaluated, rather than recovered from the
    /// token or stamp.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_for_clip_in_scope(
        &self,
        live_model_scope: CutFoldScope,
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
            || self.model_scope != pass.model_scope
            || self.model_scope != live_model_scope
        {
            return Err(invalid("stale CROWN pass stamp"));
        }
        let n_rows = self.lower_a.nrows();
        let x_dim = self.lower_a.ncols();
        validate_clip_work_budget(1, n_rows, 0, x_dim)?;
        validate_context_size(node_name, selected_neurons, row_of_neuron)?;
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

        let validated = if payload_errors_are_exact_zero(self.payload()) {
            // Root-bank rows have already been sealed as outward sound and
            // carry shared exact-zero error arrays. Preserve those immutable
            // allocations across history validation instead of copying and
            // re-discharging every dense row for every target.
            ValidatedAffineEnclosure {
                lower_a: Arc::clone(&self.lower_a),
                upper_a: Arc::clone(&self.upper_a),
                lower_b: Arc::clone(&self.lower_bias_center),
                upper_b: Arc::clone(&self.upper_bias_center),
            }
        } else {
            let (lower_a, upper_a, lower_b, upper_b) =
                discharge_payload(self.payload(), input_lower, input_upper, deadline)?;
            ValidatedAffineEnclosure {
                lower_a: Arc::new(lower_a),
                upper_a: Arc::new(upper_a),
                lower_b: Arc::new(lower_b),
                upper_b: Arc::new(upper_b),
            }
        };
        check_clip_deadline(deadline, "affine provenance return")?;
        Ok(validated)
    }
}

fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("complete-clip affine provenance: {message}"))
}

fn validate_context_size(
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
) -> Result<()> {
    let bytes = size_of::<CutFoldScope>()
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

#[derive(Clone, Copy)]
struct AffinePayloadRef<'a> {
    lower_a: &'a Array2<f32>,
    upper_a: &'a Array2<f32>,
    lower_a_error: &'a Array2<f32>,
    upper_a_error: &'a Array2<f32>,
    lower_bias_center: &'a Array1<f32>,
    upper_bias_center: &'a Array1<f32>,
    lower_bias_error: &'a Array1<f32>,
    upper_bias_error: &'a Array1<f32>,
}

impl CertifiedAffineEnclosure {
    fn payload(&self) -> AffinePayloadRef<'_> {
        AffinePayloadRef {
            lower_a: &self.lower_a,
            upper_a: &self.upper_a,
            lower_a_error: &self.lower_a_error,
            upper_a_error: &self.upper_a_error,
            lower_bias_center: &self.lower_bias_center,
            upper_bias_center: &self.upper_bias_center,
            lower_bias_error: &self.lower_bias_error,
            upper_bias_error: &self.upper_bias_error,
        }
    }
}

impl UntrustedCrownAffineRows {
    fn payload(&self) -> AffinePayloadRef<'_> {
        AffinePayloadRef {
            lower_a: &self.lower_a,
            upper_a: &self.upper_a,
            lower_a_error: &self.lower_a_error,
            upper_a_error: &self.upper_a_error,
            lower_bias_center: &self.lower_bias_center,
            upper_bias_center: &self.upper_bias_center,
            lower_bias_error: &self.lower_bias_error,
            upper_bias_error: &self.upper_bias_error,
        }
    }
}

fn validate_payload_shapes_and_values(token: &CertifiedAffineEnclosure) -> Result<()> {
    validate_payload_ref(token.payload())
}

fn payload_errors_are_exact_zero(payload: AffinePayloadRef<'_>) -> bool {
    payload
        .lower_a_error
        .iter()
        .chain(payload.upper_a_error.iter())
        .chain(payload.lower_bias_error.iter())
        .chain(payload.upper_bias_error.iter())
        .all(|value| *value == 0.0)
}

fn validate_payload_ref(payload: AffinePayloadRef<'_>) -> Result<()> {
    let shape = payload.lower_a.raw_dim();
    let n_rows = payload.lower_a.nrows();
    if payload.upper_a.raw_dim() != shape
        || payload.lower_a_error.raw_dim() != shape
        || payload.upper_a_error.raw_dim() != shape
        || payload.lower_bias_center.len() != n_rows
        || payload.upper_bias_center.len() != n_rows
        || payload.lower_bias_error.len() != n_rows
        || payload.upper_bias_error.len() != n_rows
    {
        return Err(invalid("affine payload shape mismatch"));
    }
    if payload.lower_a.iter().any(|v| !v.is_finite())
        || payload.upper_a.iter().any(|v| !v.is_finite())
        || payload.lower_bias_center.iter().any(|v| !v.is_finite())
        || payload.upper_bias_center.iter().any(|v| !v.is_finite())
        || payload
            .lower_a_error
            .iter()
            .chain(payload.upper_a_error.iter())
            .chain(payload.lower_bias_error.iter())
            .chain(payload.upper_bias_error.iter())
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(invalid("non-finite or negative affine error payload"));
    }
    Ok(())
}

fn discharge_payload(
    payload: AffinePayloadRef<'_>,
    input_lower: &[f32],
    input_upper: &[f32],
    deadline: Option<Instant>,
) -> Result<(Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>)> {
    validate_payload_ref(payload)?;
    let n_rows = payload.lower_a.nrows();
    let x_dim = payload.lower_a.ncols();
    if input_lower.len() != x_dim || input_upper.len() != x_dim {
        return Err(invalid("input domain shape mismatch"));
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
            lower_a[[row, j]] = payload.lower_a[[row, j]];
            upper_a[[row, j]] = payload.upper_a[[row, j]];
            let magnitude = f64::from(input_lower[j].abs().max(input_upper[j].abs()));
            lower_penalty = add_up(
                lower_penalty,
                mul_up(f64::from(payload.lower_a_error[[row, j]]), magnitude),
            );
            upper_penalty = add_up(
                upper_penalty,
                mul_up(f64::from(payload.upper_a_error[[row, j]]), magnitude),
            );
            cells = cells.saturating_add(1);
        }
        let lower_error = payload.lower_bias_error[row];
        let upper_error = payload.upper_bias_error[row];
        // Exact-zero errors need no arithmetic and preserve an already
        // outward source row bit-for-bit.
        let lower = if lower_error == 0.0 && lower_penalty == 0.0 {
            payload.lower_bias_center[row]
        } else {
            to_f32_down(sub_down(
                sub_down(
                    f64::from(payload.lower_bias_center[row]),
                    f64::from(lower_error),
                ),
                lower_penalty,
            ))
        };
        let upper = if upper_error == 0.0 && upper_penalty == 0.0 {
            payload.upper_bias_center[row]
        } else {
            to_f32_up(add_up(
                add_up(
                    f64::from(payload.upper_bias_center[row]),
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
    Ok((lower_a, upper_a, lower_b, upper_b))
}

/// Run an independently supplied host sound-CROWN terminal callback and seal
/// its exact-context rows for the later accelerator differential check.
///
/// The integration callback receives only an ephemeral capability containing
/// the CPU-only GEMM engine and deadline.  The intended call site is the
/// node-by-node host CROWN terminal immediately after it takes the
/// `NETWORK_INPUT` `LinearBounds`; arbitrary `LinearBounds::new` values must
/// never be routed here.  Keeping this callback boundary in one function makes
/// that integration call auditable while preventing any raw affine array from
/// constructing [`HostSoundCrownAffineRows`] directly.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn capture_host_sound_crown_rows<F>(
    model_scope: CutFoldScope,
    input_lower: &[f32],
    input_upper: &[f32],
    history: &GraphSplitHistory,
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    deadline: Option<Instant>,
    run_host_terminal: F,
) -> Result<HostSoundCrownAffineRows>
where
    F: FnOnce(&HostSoundCrownTerminalCapability<'_>) -> Result<LinearBounds>,
{
    check_clip_deadline(deadline, "host affine terminal entry")?;
    validate_context_size(node_name, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, selected_neurons.len())?;

    let cpu_engine = NaiveCpuGemmEngine;
    let capability = HostSoundCrownTerminalCapability {
        cpu_engine: &cpu_engine,
        deadline,
    };
    let terminal = run_host_terminal(&capability)?;
    check_clip_deadline(deadline, "host affine terminal return")?;

    let n_rows = terminal.num_outputs();
    let x_dim = terminal.num_inputs();
    if n_rows != selected_neurons.len() {
        return Err(invalid("host affine terminal row count mismatch"));
    }
    validate_clip_work_budget(1, n_rows, 0, x_dim)?;
    validate_input_box(input_lower, input_upper, x_dim, deadline)?;

    let lower_a_error = terminal
        .lower_a_err()
        .cloned()
        .unwrap_or_else(|| Array2::zeros((n_rows, x_dim)));
    let upper_a_error = terminal
        .upper_a_err()
        .cloned()
        .unwrap_or_else(|| Array2::zeros((n_rows, x_dim)));
    let host_payload = UntrustedCrownAffineRows::new(
        terminal.lower_a().clone(),
        terminal.upper_a().clone(),
        lower_a_error,
        upper_a_error,
        terminal.lower_b().clone(),
        terminal.upper_b().clone(),
        Array1::zeros(n_rows),
        Array1::zeros(n_rows),
    );
    let (lower_a, upper_a, lower_b, upper_b) =
        discharge_payload(host_payload.payload(), input_lower, input_upper, deadline)?;
    let history_identity = history
        .exact_provenance_identity()
        .ok_or_else(|| invalid("host split-history identity resource refusal"))?;
    check_clip_deadline(deadline, "host affine terminal seal")?;

    // Sole production construction of the opaque host terminal type.
    Ok(HostSoundCrownAffineRows {
        model_scope,
        input_lower_bits: f32_bits(input_lower),
        input_upper_bits: f32_bits(input_upper),
        history_identity: history_identity.into(),
        node_name: Arc::from(node_name),
        selected_neurons: Arc::from(selected_neurons),
        row_of_neuron: Arc::from(row_of_neuron),
        lower_a,
        upper_a,
        lower_b,
        upper_b,
    })
}

/// Capture one host CROWN terminal over the exact root input box.
///
/// This generic callback boundary is deliberately private. Production callers
/// can obtain a reusable root template only through
/// [`capture_sound_crown_root_rows_at_node`] or
/// [`capture_exact_root_input_rows`], whose terminal producers are fixed in
/// this module.
#[allow(clippy::too_many_arguments)]
fn capture_host_sound_crown_root_rows_impl<F>(
    model_scope: CutFoldScope,
    input_lower: &[f32],
    input_upper: &[f32],
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    deadline: Option<Instant>,
    run_host_terminal: F,
) -> Result<HostSoundCrownRootAffineRows>
where
    F: FnOnce(&HostSoundCrownTerminalCapability<'_>) -> Result<LinearBounds>,
{
    check_clip_deadline(deadline, "root host affine terminal entry")?;
    validate_context_size(node_name, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, selected_neurons.len())?;

    let cpu_engine = NaiveCpuGemmEngine;
    let capability = HostSoundCrownTerminalCapability {
        cpu_engine: &cpu_engine,
        deadline,
    };
    let terminal = run_host_terminal(&capability)?;
    check_clip_deadline(deadline, "root host affine terminal return")?;

    let n_rows = terminal.num_outputs();
    let x_dim = terminal.num_inputs();
    if n_rows != selected_neurons.len() {
        return Err(invalid("root host affine terminal row count mismatch"));
    }
    validate_clip_work_budget(1, n_rows, 0, x_dim)?;
    validate_input_box(input_lower, input_upper, x_dim, deadline)?;

    let lower_a_error = terminal
        .lower_a_err()
        .cloned()
        .unwrap_or_else(|| Array2::zeros((n_rows, x_dim)));
    let upper_a_error = terminal
        .upper_a_err()
        .cloned()
        .unwrap_or_else(|| Array2::zeros((n_rows, x_dim)));
    let host_payload = UntrustedCrownAffineRows::new(
        terminal.lower_a().clone(),
        terminal.upper_a().clone(),
        lower_a_error,
        upper_a_error,
        terminal.lower_b().clone(),
        terminal.upper_b().clone(),
        Array1::zeros(n_rows),
        Array1::zeros(n_rows),
    );
    let (lower_a, upper_a, lower_b, upper_b) =
        discharge_payload(host_payload.payload(), input_lower, input_upper, deadline)?;
    check_clip_deadline(deadline, "root host affine terminal seal")?;

    // Sole production construction of the root host terminal.  In particular,
    // no conversion from `HostSoundCrownAffineRows` exists.
    Ok(HostSoundCrownRootAffineRows {
        model_scope,
        input_lower_bits: f32_bits(input_lower),
        input_upper_bits: f32_bits(input_upper),
        node_name: Arc::from(node_name),
        selected_neurons: Arc::from(selected_neurons),
        row_of_neuron: Arc::from(row_of_neuron),
        lower_a,
        upper_a,
        lower_b,
        upper_b,
    })
}

/// Capture sound input-relative CROWN rows at one graph node over the finalized
/// root box.
///
/// Unlike the test-only differential-checking seam, this production API accepts
/// no callback and no raw affine arrays. The only terminal it can seal is the
/// selected node-by-node host CROWN replay below, using the private scalar CPU
/// capability created by this module.
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_sound_crown_root_rows_at_node(
    graph: &GraphNetwork,
    seed_node: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    root_bounds: &HashMap<String, Arc<BoundedTensor>>,
    root_input: &BoundedTensor,
    alpha_state: Option<&GraphDomainAlphaState>,
    deadline: Option<Instant>,
) -> Result<SoundCrownRootAffineRows> {
    check_clip_deadline(deadline, "root graph affine capture entry")?;
    if root_input.l2_constraint().is_some() {
        return Err(invalid("root graph affine capture refuses L2 input"));
    }
    if selected_neurons.is_empty() {
        return Err(invalid("root graph affine capture has no selected rows"));
    }
    let flat = root_input.flatten();
    let input_lower = flat
        .lower()
        .as_slice_memory_order()
        .ok_or_else(|| invalid("non-contiguous root input lower bound"))?;
    let input_upper = flat
        .upper()
        .as_slice_memory_order()
        .ok_or_else(|| invalid("non-contiguous root input upper bound"))?;
    validate_context_size(seed_node, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, selected_neurons.len())?;
    validate_input_box(input_lower, input_upper, input_lower.len(), deadline)?;
    validate_clip_work_budget(1, selected_neurons.len(), 0, input_lower.len())?;
    check_clip_deadline(deadline, "root graph affine capture preflight")?;
    let host = capture_host_sound_crown_root_rows_impl(
        graph.cut_fold_scope(),
        input_lower,
        input_upper,
        seed_node,
        selected_neurons,
        row_of_neuron,
        deadline,
        |capability| {
            backward_selected_input_relative_bounds_at_node(
                graph,
                seed_node,
                selected_neurons,
                root_bounds,
                root_input,
                None,
                alpha_state,
                capability.cpu_engine(),
                capability.deadline(),
            )
            .ok_or_else(|| invalid("root host CROWN terminal refused"))
        },
    )?;
    seal_host_sound_crown_root_rows(host, deadline)
}

/// Capture exact selected identity rows for a ReLU fed directly by the network
/// input.
///
/// The affine terminal is constructed here from the selected input coordinates;
/// callers cannot substitute an arbitrary `LinearBounds` value.
pub(crate) fn capture_exact_root_input_rows(
    graph: &GraphNetwork,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    root_input: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<SoundCrownRootAffineRows> {
    check_clip_deadline(deadline, "root input identity capture entry")?;
    if root_input.l2_constraint().is_some() {
        return Err(invalid("root input identity capture refuses L2 input"));
    }
    if selected_neurons.is_empty() {
        return Err(invalid("root input identity capture has no selected rows"));
    }
    let flat = root_input.flatten();
    let input_lower = flat
        .lower()
        .as_slice_memory_order()
        .ok_or_else(|| invalid("non-contiguous root input lower bound"))?;
    let input_upper = flat
        .upper()
        .as_slice_memory_order()
        .ok_or_else(|| invalid("non-contiguous root input upper bound"))?;
    let input_dim = input_lower.len();
    validate_context_size(NETWORK_INPUT, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, selected_neurons.len())?;
    validate_input_box(input_lower, input_upper, input_dim, deadline)?;
    validate_clip_work_budget(1, selected_neurons.len(), 0, input_dim)?;
    check_clip_deadline(deadline, "root input identity capture preflight")?;
    if row_of_neuron.len() != input_dim {
        return Err(invalid("root input identity row map width mismatch"));
    }
    let mut spec = Array2::<f32>::zeros((selected_neurons.len(), input_dim));
    for (row, &neuron) in selected_neurons.iter().enumerate() {
        if neuron >= input_dim {
            return Err(invalid("root input identity neuron out of range"));
        }
        spec[[row, neuron]] = 1.0;
    }
    let host = capture_host_sound_crown_root_rows_impl(
        graph.cut_fold_scope(),
        input_lower,
        input_upper,
        NETWORK_INPUT,
        selected_neurons,
        row_of_neuron,
        deadline,
        |_capability| LinearBounds::from_spec_matrix(spec),
    )?;
    seal_host_sound_crown_root_rows(host, deadline)
}

fn seal_host_sound_crown_root_rows(
    host: HostSoundCrownRootAffineRows,
    deadline: Option<Instant>,
) -> Result<SoundCrownRootAffineRows> {
    check_clip_deadline(deadline, "root host affine direct seal")?;
    let n_rows = host.lower_a.nrows();
    let x_dim = host.lower_a.ncols();
    validate_root_host_rows(&host, n_rows, x_dim)?;
    let zero_a_error = Arc::new(Array2::zeros((n_rows, x_dim)));
    let zero_bias_error = Arc::new(Array1::zeros(n_rows));
    Ok(SoundCrownRootAffineRows {
        model_scope: host.model_scope,
        input_lower_bits: host.input_lower_bits,
        input_upper_bits: host.input_upper_bits,
        node_name: host.node_name,
        selected_neurons: host.selected_neurons,
        row_of_neuron: host.row_of_neuron,
        lower_a: Arc::new(host.lower_a),
        upper_a: Arc::new(host.upper_a),
        lower_b: Arc::new(host.lower_b),
        upper_b: Arc::new(host.upper_b),
        zero_a_error,
        zero_bias_error,
    })
}

/// Test-only raw-terminal capture seam for adversarial provenance tests.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_host_sound_crown_root_rows<F>(
    model_scope: CutFoldScope,
    input_lower: &[f32],
    input_upper: &[f32],
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    deadline: Option<Instant>,
    run_host_terminal: F,
) -> Result<HostSoundCrownRootAffineRows>
where
    F: FnOnce(&HostSoundCrownTerminalCapability<'_>) -> Result<LinearBounds>,
{
    capture_host_sound_crown_root_rows_impl(
        model_scope,
        input_lower,
        input_upper,
        node_name,
        selected_neurons,
        row_of_neuron,
        deadline,
        run_host_terminal,
    )
}

/// Independently check every suggested affine row against already validated
/// host sound-CROWN rows and seal the suggestion only on a pointwise proof.
///
/// For a lower suggestion `G_L` and host lower enclosure `H_L`, this proves
/// `max_{x in box}(G_L(x) - H_L(x)) <= 0`.  For the upper side it proves
/// `max_{x in box}(H_U(x) - G_U(x)) <= 0`.  Thus the suggestion is no tighter
/// than the independently sound host row at any point in the box.  This is
/// deliberately stronger than comparing scalar concretizations and uses no
/// numerical tolerance.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn check_affine_dominance_and_seal(
    model_scope: CutFoldScope,
    input_lower: &[f32],
    input_upper: &[f32],
    history: &GraphSplitHistory,
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    host: &HostSoundCrownAffineRows,
    suggestion: UntrustedCrownAffineRows,
    deadline: Option<Instant>,
) -> Result<SoundCrownAffineRows> {
    check_clip_deadline(deadline, "affine dominance entry")?;
    let payload = suggestion.payload();
    validate_payload_ref(payload)?;
    let n_rows = payload.lower_a.nrows();
    let x_dim = payload.lower_a.ncols();
    validate_clip_work_budget(1, n_rows, 0, x_dim)?;
    validate_context_size(node_name, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, n_rows)?;
    validate_input_box(input_lower, input_upper, x_dim, deadline)?;

    let history_identity = history
        .exact_provenance_identity()
        .ok_or_else(|| invalid("split-history identity resource refusal"))?;
    check_clip_deadline(deadline, "affine dominance history identity")?;
    if !host.matches_exact_context(
        model_scope,
        input_lower,
        input_upper,
        &history_identity,
        node_name,
        selected_neurons,
        row_of_neuron,
    ) {
        return Err(invalid("host affine enclosure context mismatch"));
    }
    validate_host_rows(host, n_rows, x_dim)?;

    let (lower_a, upper_a, lower_b, upper_b) =
        discharge_payload(payload, input_lower, input_upper, deadline)?;
    check_pointwise_affine_dominance(
        &host.lower_a,
        &host.upper_a,
        &host.lower_b,
        &host.upper_b,
        &lower_a,
        &upper_a,
        &lower_b,
        &upper_b,
        input_lower,
        input_upper,
        deadline,
    )?;
    check_clip_deadline(deadline, "affine dominance seal")?;

    // Sole production construction of the sealed type.  The raw suggestion is
    // inaccessible after this point; only its outward-folded, all-row-checked
    // affine forms are retained.
    let zero_a_error = Arc::new(Array2::zeros((n_rows, x_dim)));
    let zero_bias_error = Arc::new(Array1::zeros(n_rows));
    Ok(SoundCrownAffineRows {
        model_scope,
        input_lower_bits: f32_bits(input_lower),
        input_upper_bits: f32_bits(input_upper),
        history_identity: history_identity.into(),
        node_name: Arc::from(node_name),
        selected_neurons: Arc::from(selected_neurons),
        row_of_neuron: Arc::from(row_of_neuron),
        lower_a: Arc::new(lower_a),
        upper_a: Arc::new(upper_a),
        lower_b: Arc::new(lower_b),
        upper_b: Arc::new(upper_b),
        zero_a_error,
        zero_bias_error,
    })
}

/// Check an untrusted root-row suggestion against an independently captured
/// root host terminal and seal it as an immutable, history-free template.
///
/// The proof is the same directed all-row pointwise dominance proof used by
/// [`check_affine_dominance_and_seal`], but neither the inputs nor the output
/// carry a child history.  Consequently, the resulting template can acquire a
/// child identity only through [`bind_root_sound_crown_rows_to_history`].
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn check_root_affine_dominance_and_seal(
    model_scope: CutFoldScope,
    input_lower: &[f32],
    input_upper: &[f32],
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    host: &HostSoundCrownRootAffineRows,
    suggestion: UntrustedCrownAffineRows,
    deadline: Option<Instant>,
) -> Result<SoundCrownRootAffineRows> {
    check_clip_deadline(deadline, "root affine dominance entry")?;
    let payload = suggestion.payload();
    validate_payload_ref(payload)?;
    let n_rows = payload.lower_a.nrows();
    let x_dim = payload.lower_a.ncols();
    validate_clip_work_budget(1, n_rows, 0, x_dim)?;
    validate_context_size(node_name, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, n_rows)?;
    validate_input_box(input_lower, input_upper, x_dim, deadline)?;

    if !host.matches_exact_root_context(
        model_scope,
        input_lower,
        input_upper,
        node_name,
        selected_neurons,
        row_of_neuron,
    ) {
        return Err(invalid("root host affine enclosure context mismatch"));
    }
    validate_root_host_rows(host, n_rows, x_dim)?;

    let (lower_a, upper_a, lower_b, upper_b) =
        discharge_payload(payload, input_lower, input_upper, deadline)?;
    check_pointwise_affine_dominance(
        &host.lower_a,
        &host.upper_a,
        &host.lower_b,
        &host.upper_b,
        &lower_a,
        &upper_a,
        &lower_b,
        &upper_b,
        input_lower,
        input_upper,
        deadline,
    )?;
    check_clip_deadline(deadline, "root affine dominance seal")?;

    let zero_a_error = Arc::new(Array2::zeros((n_rows, x_dim)));
    let zero_bias_error = Arc::new(Array1::zeros(n_rows));
    Ok(SoundCrownRootAffineRows {
        model_scope,
        input_lower_bits: f32_bits(input_lower),
        input_upper_bits: f32_bits(input_upper),
        node_name: Arc::from(node_name),
        selected_neurons: Arc::from(selected_neurons),
        row_of_neuron: Arc::from(row_of_neuron),
        lower_a: Arc::new(lower_a),
        upper_a: Arc::new(upper_a),
        lower_b: Arc::new(lower_b),
        upper_b: Arc::new(upper_b),
        zero_a_error,
        zero_bias_error,
    })
}

/// Borrow a reusable root template and bind it to one exact child split
/// history, yielding the ordinary sealed input accepted by the pass factory.
///
/// Model scope, input bits, node, and objective layout are revalidated on every
/// binding.  The template remains available for subsequent children.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_root_sound_crown_rows_to_history(
    model_scope: CutFoldScope,
    input_lower: &[f32],
    input_upper: &[f32],
    history: &GraphSplitHistory,
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    root_rows: &SoundCrownRootAffineRows,
    deadline: Option<Instant>,
) -> Result<SoundCrownAffineRows> {
    check_clip_deadline(deadline, "root affine history bind entry")?;
    let n_rows = root_rows.lower_a.nrows();
    let x_dim = root_rows.lower_a.ncols();
    validate_clip_work_budget(1, n_rows, 0, x_dim)?;
    validate_context_size(node_name, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, n_rows)?;
    validate_input_box(input_lower, input_upper, x_dim, deadline)?;
    validate_validated_row_arrays(
        &root_rows.lower_a,
        &root_rows.upper_a,
        &root_rows.lower_b,
        &root_rows.upper_b,
        n_rows,
        x_dim,
    )?;

    if !root_rows.matches_exact_root_context(
        model_scope,
        input_lower,
        input_upper,
        node_name,
        selected_neurons,
        row_of_neuron,
    ) {
        return Err(invalid("sealed root affine rows context mismatch"));
    }
    let history_identity = history
        .exact_provenance_identity()
        .ok_or_else(|| invalid("root bind split-history identity resource refusal"))?;
    check_clip_deadline(deadline, "root affine history bind seal")?;

    Ok(SoundCrownAffineRows {
        model_scope,
        input_lower_bits: Arc::clone(&root_rows.input_lower_bits),
        input_upper_bits: Arc::clone(&root_rows.input_upper_bits),
        history_identity: history_identity.into(),
        node_name: Arc::clone(&root_rows.node_name),
        selected_neurons: Arc::clone(&root_rows.selected_neurons),
        row_of_neuron: Arc::clone(&root_rows.row_of_neuron),
        // Root coefficients are immutable and identical for every ReLU child.
        // Arc binding keeps one physical top-K bank per layer instead of four
        // dense matrices per domain.
        lower_a: Arc::clone(&root_rows.lower_a),
        upper_a: Arc::clone(&root_rows.upper_a),
        lower_b: Arc::clone(&root_rows.lower_b),
        upper_b: Arc::clone(&root_rows.upper_b),
        zero_a_error: Arc::clone(&root_rows.zero_a_error),
        zero_bias_error: Arc::clone(&root_rows.zero_bias_error),
    })
}

/// Consume independently checked rows and mint their one-use CROWN pass
/// authority.  No raw coefficient type can call this factory successfully
/// because [`SoundCrownAffineRows`] has no public or crate-visible constructor.
pub(crate) fn mint_certified_affine_enclosure(
    rows: SoundCrownAffineRows,
    deadline: Option<Instant>,
) -> Result<(CrownPassStamp, CertifiedAffineEnclosure)> {
    check_clip_deadline(deadline, "affine provenance factory entry")?;
    let n_rows = rows.lower_a.nrows();
    let x_dim = rows.lower_a.ncols();
    validate_clip_work_budget(1, n_rows, 0, x_dim)?;
    validate_context_size(
        rows.node_name.as_ref(),
        rows.selected_neurons.as_ref(),
        rows.row_of_neuron.as_ref(),
    )?;
    validate_subject_layout(
        rows.selected_neurons.as_ref(),
        rows.row_of_neuron.as_ref(),
        n_rows,
    )?;
    validate_validated_row_arrays(
        &rows.lower_a,
        &rows.upper_a,
        &rows.lower_b,
        &rows.upper_b,
        n_rows,
        x_dim,
    )?;
    check_clip_deadline(deadline, "affine provenance factory pass")?;
    let pass_words = fresh_pass_words()?;

    let pass = CrownPassStamp {
        words: pass_words,
        model_scope: rows.model_scope,
    };
    let token = CertifiedAffineEnclosure {
        pass_words,
        model_scope: rows.model_scope,
        input_lower_bits: rows.input_lower_bits,
        input_upper_bits: rows.input_upper_bits,
        history_identity: rows.history_identity,
        node_name: rows.node_name,
        selected_neurons: rows.selected_neurons,
        row_of_neuron: rows.row_of_neuron,
        lower_a: rows.lower_a,
        upper_a: rows.upper_a,
        lower_a_error: Arc::clone(&rows.zero_a_error),
        upper_a_error: rows.zero_a_error,
        lower_bias_center: rows.lower_b,
        upper_bias_center: rows.upper_b,
        lower_bias_error: Arc::clone(&rows.zero_bias_error),
        upper_bias_error: rows.zero_bias_error,
    };
    validate_payload_shapes_and_values(&token)?;
    Ok((pass, token))
}

fn validate_input_box(
    input_lower: &[f32],
    input_upper: &[f32],
    x_dim: usize,
    deadline: Option<Instant>,
) -> Result<()> {
    if input_lower.len() != x_dim || input_upper.len() != x_dim {
        return Err(invalid("input domain shape mismatch"));
    }
    for (j, (&lower, &upper)) in input_lower.iter().zip(input_upper).enumerate() {
        if j.is_multiple_of(PROVENANCE_POLL_STRIDE) {
            check_clip_deadline(deadline, "affine provenance input validation")?;
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid("input domain invalid"));
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_host_rows(rows: &HostSoundCrownAffineRows, n_rows: usize, x_dim: usize) -> Result<()> {
    validate_validated_row_arrays(
        &rows.lower_a,
        &rows.upper_a,
        &rows.lower_b,
        &rows.upper_b,
        n_rows,
        x_dim,
    )
}

fn validate_root_host_rows(
    rows: &HostSoundCrownRootAffineRows,
    n_rows: usize,
    x_dim: usize,
) -> Result<()> {
    validate_validated_row_arrays(
        &rows.lower_a,
        &rows.upper_a,
        &rows.lower_b,
        &rows.upper_b,
        n_rows,
        x_dim,
    )
}

fn validate_validated_row_arrays(
    lower_a: &Array2<f32>,
    upper_a: &Array2<f32>,
    lower_b: &Array1<f32>,
    upper_b: &Array1<f32>,
    n_rows: usize,
    x_dim: usize,
) -> Result<()> {
    if lower_a.dim() != (n_rows, x_dim)
        || upper_a.dim() != (n_rows, x_dim)
        || lower_b.len() != n_rows
        || upper_b.len() != n_rows
    {
        return Err(invalid("validated affine row shape mismatch"));
    }
    if lower_a
        .iter()
        .chain(upper_a.iter())
        .chain(lower_b.iter())
        .chain(upper_b.iter())
        .any(|value| !value.is_finite())
    {
        return Err(invalid("non-finite validated affine row"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn check_pointwise_affine_dominance(
    host_lower_a: &Array2<f32>,
    host_upper_a: &Array2<f32>,
    host_lower_b: &Array1<f32>,
    host_upper_b: &Array1<f32>,
    suggested_lower_a: &Array2<f32>,
    suggested_upper_a: &Array2<f32>,
    suggested_lower_b: &Array1<f32>,
    suggested_upper_b: &Array1<f32>,
    input_lower: &[f32],
    input_upper: &[f32],
    deadline: Option<Instant>,
) -> Result<()> {
    let n_rows = host_lower_a.nrows();
    let x_dim = host_lower_a.ncols();
    validate_validated_row_arrays(
        host_lower_a,
        host_upper_a,
        host_lower_b,
        host_upper_b,
        n_rows,
        x_dim,
    )?;
    validate_validated_row_arrays(
        suggested_lower_a,
        suggested_upper_a,
        suggested_lower_b,
        suggested_upper_b,
        n_rows,
        x_dim,
    )?;

    for row in 0..n_rows {
        check_clip_deadline(deadline, "affine dominance row")?;
        let lower_excess = affine_difference_box_max_up(
            suggested_lower_a,
            suggested_lower_b,
            host_lower_a,
            host_lower_b,
            row,
            input_lower,
            input_upper,
            deadline,
        )?;
        if !lower_excess.is_finite() || lower_excess > 0.0 {
            return Err(invalid("suggested lower affine row exceeds host row"));
        }
        let upper_shortfall = affine_difference_box_max_up(
            host_upper_a,
            host_upper_b,
            suggested_upper_a,
            suggested_upper_b,
            row,
            input_lower,
            input_upper,
            deadline,
        )?;
        if !upper_shortfall.is_finite() || upper_shortfall > 0.0 {
            return Err(invalid("suggested upper affine row is below host row"));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn affine_difference_box_max_up(
    lhs_a: &Array2<f32>,
    lhs_b: &Array1<f32>,
    rhs_a: &Array2<f32>,
    rhs_b: &Array1<f32>,
    row: usize,
    input_lower: &[f32],
    input_upper: &[f32],
    deadline: Option<Instant>,
) -> Result<f64> {
    let (_, mut maximum) = sub_interval(f64::from(lhs_b[row]), f64::from(rhs_b[row]));
    for j in 0..input_lower.len() {
        if j.is_multiple_of(PROVENANCE_POLL_STRIDE) {
            check_clip_deadline(deadline, "affine dominance coefficient")?;
        }
        let (delta_lower, delta_upper) =
            sub_interval(f64::from(lhs_a[[row, j]]), f64::from(rhs_a[[row, j]]));
        let box_lower = f64::from(input_lower[j]);
        let box_upper = f64::from(input_upper[j]);
        // The exact coefficient difference lies in [delta_lower, delta_upper].
        // A bilinear function on that rectangle reaches its maximum at a
        // corner.  Widen every product and the running sum upward.
        let term = mul_up(delta_lower, box_lower)
            .max(mul_up(delta_lower, box_upper))
            .max(mul_up(delta_upper, box_lower))
            .max(mul_up(delta_upper, box_upper));
        maximum = add_up(maximum, term);
    }
    Ok(maximum)
}

#[cfg(test)]
fn sub_interval(a: f64, b: f64) -> (f64, f64) {
    if a == b {
        (0.0, 0.0)
    } else if b == 0.0 {
        (a, a)
    } else {
        let difference = a - b;
        (next_down_f64(difference), next_up_f64(difference))
    }
}

fn f32_bits(values: &[f32]) -> Arc<[u32]> {
    values
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>()
        .into()
}

fn exact_f32_bits_match(values: &[f32], bits: &[u32]) -> bool {
    values.len() == bits.len()
        && values
            .iter()
            .zip(bits)
            .all(|(value, expected)| value.to_bits() == *expected)
}

fn increment_pass_words(words: &mut [u64; 2]) -> Result<[u64; 2]> {
    if words[1] != u64::MAX {
        words[1] += 1;
    } else if words[0] != u64::MAX {
        words[0] += 1;
        words[1] = 0;
    } else {
        return Err(invalid("CROWN pass identity space exhausted"));
    }
    Ok(*words)
}

fn fresh_pass_words() -> Result<[u64; 2]> {
    let mut words = NEXT_CROWN_PASS
        .lock()
        .map_err(|_| invalid("CROWN pass allocator poisoned"))?;
    increment_pass_words(&mut words)
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
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::NEG_INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return -f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn next_up_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
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
    test_certified_affine_fixture_in_scope(
        pass_words,
        CutFoldScope::fresh(),
        input_lower,
        input_upper,
        history,
        node_name,
        selected_neurons,
        row_of_neuron,
        parts,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn test_certified_affine_fixture_in_scope(
    pass_words: [u64; 2],
    model_scope: CutFoldScope,
    input_lower: &[f32],
    input_upper: &[f32],
    history: &GraphSplitHistory,
    node_name: &str,
    selected_neurons: &[usize],
    row_of_neuron: &[usize],
    parts: TestSoundCrownAffineParts,
) -> Result<(CrownPassStamp, CertifiedAffineEnclosure)> {
    let pass = CrownPassStamp {
        words: pass_words,
        model_scope,
    };
    validate_clip_work_budget(1, parts.lower_a.nrows(), 0, parts.lower_a.ncols())?;
    validate_context_size(node_name, selected_neurons, row_of_neuron)?;
    validate_subject_layout(selected_neurons, row_of_neuron, parts.lower_a.nrows())?;
    validate_input_box(input_lower, input_upper, parts.lower_a.ncols(), None)?;
    let history_identity = history
        .exact_provenance_identity()
        .ok_or_else(|| invalid("fixture split-history identity refusal"))?;
    let token = CertifiedAffineEnclosure {
        pass_words,
        model_scope,
        input_lower_bits: f32_bits(input_lower),
        input_upper_bits: f32_bits(input_upper),
        history_identity: history_identity.into(),
        node_name: Arc::from(node_name),
        selected_neurons: Arc::from(selected_neurons),
        row_of_neuron: Arc::from(row_of_neuron),
        lower_a: Arc::new(parts.lower_a),
        upper_a: Arc::new(parts.upper_a),
        lower_a_error: Arc::new(parts.lower_a_error),
        upper_a_error: Arc::new(parts.upper_a_error),
        lower_bias_center: Arc::new(parts.lower_bias_center),
        upper_bias_center: Arc::new(parts.upper_bias_center),
        lower_bias_error: Arc::new(parts.lower_bias_error),
        upper_bias_error: Arc::new(parts.upper_bias_error),
    };
    validate_payload_shapes_and_values(&token)?;
    Ok((pass, token))
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_tensor::L2Constraint;

    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
    use crate::layers::LinearLayer;
    use crate::{GraphNode, Layer};

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

    fn zero_parts(n_rows: usize, x_dim: usize) -> TestSoundCrownAffineParts {
        TestSoundCrownAffineParts {
            lower_a: Array2::zeros((n_rows, x_dim)),
            upper_a: Array2::zeros((n_rows, x_dim)),
            lower_a_error: Array2::zeros((n_rows, x_dim)),
            upper_a_error: Array2::zeros((n_rows, x_dim)),
            lower_bias_center: Array1::zeros(n_rows),
            upper_bias_center: Array1::zeros(n_rows),
            lower_bias_error: Array1::zeros(n_rows),
            upper_bias_error: Array1::zeros(n_rows),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn validated_host(
        model_scope: CutFoldScope,
        input_lower: &[f32],
        input_upper: &[f32],
        h: &GraphSplitHistory,
        selected: &[usize],
        row_of: &[usize],
        source: TestSoundCrownAffineParts,
    ) -> HostSoundCrownAffineRows {
        capture_host_sound_crown_rows(
            model_scope,
            input_lower,
            input_upper,
            h,
            "relu",
            selected,
            row_of,
            None,
            |capability| {
                // The real integration callback uses both values while running
                // the selected node-by-node host CROWN replay.
                let _cpu = capability.cpu_engine();
                let _deadline = capability.deadline();
                let mut bounds = LinearBounds::new(
                    source.lower_a,
                    source.lower_bias_center,
                    source.upper_a,
                    source.upper_bias_center,
                )?;
                bounds.set_coeff_err(source.lower_a_error, source.upper_a_error);
                Ok(bounds)
            },
        )
        .expect("host terminal")
    }

    #[allow(clippy::too_many_arguments)]
    fn validated_root_host(
        model_scope: CutFoldScope,
        input_lower: &[f32],
        input_upper: &[f32],
        selected: &[usize],
        row_of: &[usize],
        source: TestSoundCrownAffineParts,
    ) -> HostSoundCrownRootAffineRows {
        capture_host_sound_crown_root_rows(
            model_scope,
            input_lower,
            input_upper,
            "relu",
            selected,
            row_of,
            None,
            |capability| {
                let _cpu = capability.cpu_engine();
                let _deadline = capability.deadline();
                let mut bounds = LinearBounds::new(
                    source.lower_a,
                    source.lower_bias_center,
                    source.upper_a,
                    source.upper_bias_center,
                )?;
                bounds.set_coeff_err(source.lower_a_error, source.upper_a_error);
                Ok(bounds)
            },
        )
        .expect("root host terminal")
    }

    fn suggestion(
        lower_a: Array2<f32>,
        upper_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_b: Array1<f32>,
    ) -> UntrustedCrownAffineRows {
        let shape = lower_a.raw_dim();
        let n_rows = lower_a.nrows();
        UntrustedCrownAffineRows::new(
            lower_a,
            upper_a,
            Array2::zeros(shape),
            Array2::zeros(shape),
            lower_b,
            upper_b,
            Array1::zeros(n_rows),
            Array1::zeros(n_rows),
        )
    }

    #[test]
    fn root_template_binds_two_distinct_histories_with_fresh_exact_tokens() {
        let scope = CutFoldScope::fresh();
        let input_lower = [-1.0];
        let input_upper = [1.0];
        let active = history(true);
        let inactive = history(false);
        let host = validated_root_host(
            scope,
            &input_lower,
            &input_upper,
            &[0],
            &[0],
            zero_parts(1, 1),
        );
        let root = check_root_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0])),
            None,
        )
        .expect("sealed reusable root rows");

        let active_rows = bind_root_sound_crown_rows_to_history(
            scope,
            &input_lower,
            &input_upper,
            &active,
            "relu",
            &[0],
            &[0],
            &root,
            None,
        )
        .expect("active child bind");
        let inactive_rows = bind_root_sound_crown_rows_to_history(
            scope,
            &input_lower,
            &input_upper,
            &inactive,
            "relu",
            &[0],
            &[0],
            &root,
            None,
        )
        .expect("inactive child bind reuses same root");
        let expected_resident_bytes = 6usize
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(2 * size_of::<u32>()))
            .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>()))
            .and_then(|bytes| bytes.checked_add("relu".len()))
            .expect("small fixture byte count");
        assert_eq!(root.resident_payload_bytes(), Some(expected_resident_bytes));
        let (active_pass, active_token) =
            mint_certified_affine_enclosure(active_rows, None).expect("active token");
        let (inactive_pass, inactive_token) =
            mint_certified_affine_enclosure(inactive_rows, None).expect("inactive token");
        assert_ne!(active_pass.words, inactive_pass.words);
        assert_ne!(
            active_token.history_identity.as_ref(),
            inactive_token.history_identity.as_ref()
        );
        assert!(Arc::ptr_eq(&active_token.lower_a, &inactive_token.lower_a));
        assert!(Arc::ptr_eq(&active_token.upper_a, &inactive_token.upper_a));
        assert!(Arc::ptr_eq(
            &active_token.lower_a_error,
            &inactive_token.lower_a_error
        ));
        assert!(Arc::ptr_eq(
            &active_token.upper_a_error,
            &inactive_token.upper_a_error
        ));
        assert!(Arc::ptr_eq(
            &active_token.lower_bias_center,
            &inactive_token.lower_bias_center
        ));
        assert!(Arc::ptr_eq(
            &active_token.upper_bias_center,
            &inactive_token.upper_bias_center
        ));
        assert!(Arc::ptr_eq(
            &active_token.lower_bias_error,
            &inactive_token.lower_bias_error
        ));
        assert!(Arc::ptr_eq(
            &active_token.upper_bias_error,
            &inactive_token.upper_bias_error
        ));

        let active_validated = active_token
            .validate_for_clip_in_scope(
                scope,
                &active_pass,
                &input_lower,
                &input_upper,
                &active,
                "relu",
                &[0],
                &[0],
                None,
            )
            .expect("active root token validation");
        let inactive_validated = inactive_token
            .validate_for_clip_in_scope(
                scope,
                &inactive_pass,
                &input_lower,
                &input_upper,
                &inactive,
                "relu",
                &[0],
                &[0],
                None,
            )
            .expect("inactive root token validation");
        assert!(Arc::ptr_eq(
            &active_validated.lower_a,
            &inactive_validated.lower_a
        ));
        assert!(Arc::ptr_eq(
            &active_validated.upper_a,
            &inactive_validated.upper_a
        ));
        assert!(Arc::ptr_eq(
            &active_validated.lower_b,
            &inactive_validated.lower_b
        ));
        assert!(Arc::ptr_eq(
            &active_validated.upper_b,
            &inactive_validated.upper_b
        ));
        assert!(active_token
            .validate_for_clip_in_scope(
                scope,
                &active_pass,
                &input_lower,
                &input_upper,
                &inactive,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());
    }

    #[test]
    fn root_template_rejects_wrong_live_model_input_and_layout() {
        let scope = CutFoldScope::fresh();
        let input_lower = [-0.0];
        let input_upper = [1.0];
        let h = history(true);
        let host = validated_root_host(
            scope,
            &input_lower,
            &input_upper,
            &[0],
            &[0],
            zero_parts(1, 1),
        );
        let root = check_root_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0])),
            None,
        )
        .expect("sealed root");

        assert!(bind_root_sound_crown_rows_to_history(
            CutFoldScope::fresh(),
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &[0],
            &[0],
            &root,
            None,
        )
        .is_err());
        // Signed-zero bit identity is load-bearing.
        assert!(bind_root_sound_crown_rows_to_history(
            scope,
            &[0.0],
            &input_upper,
            &h,
            "relu",
            &[0],
            &[0],
            &root,
            None,
        )
        .is_err());
        assert!(bind_root_sound_crown_rows_to_history(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &[1],
            &[usize::MAX, 0],
            &root,
            None,
        )
        .is_err());
        assert!(bind_root_sound_crown_rows_to_history(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "other",
            &[0],
            &[0],
            &root,
            None,
        )
        .is_err());
    }

    #[test]
    fn root_checker_rejects_tight_nan_and_wrong_shaped_gpu_like_suggestions() {
        let scope = CutFoldScope::fresh();
        let input_lower = [-1.0];
        let input_upper = [1.0];
        let host = validated_root_host(
            scope,
            &input_lower,
            &input_upper,
            &[0],
            &[0],
            zero_parts(1, 1),
        );
        let one_ulp = f32::from_bits(1);
        assert!(check_root_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(
                arr2(&[[0.0]]),
                arr2(&[[0.0]]),
                arr1(&[one_ulp]),
                arr1(&[0.0]),
            ),
            None,
        )
        .is_err());

        assert!(check_root_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(
                arr2(&[[f32::NAN]]),
                arr2(&[[0.0]]),
                arr1(&[0.0]),
                arr1(&[0.0]),
            ),
            None,
        )
        .is_err());

        assert!(check_root_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(
                arr2(&[[0.0, 0.0]]),
                arr2(&[[0.0, 0.0]]),
                arr1(&[0.0]),
                arr1(&[0.0]),
            ),
            None,
        )
        .is_err());
    }

    #[test]
    fn root_template_deadlines_fail_closed_before_host_callback_and_binding() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let scope = CutFoldScope::fresh();
        let input_lower = [-1.0];
        let input_upper = [1.0];
        let callback_called = AtomicBool::new(false);
        let expired_capture = capture_host_sound_crown_root_rows(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            Some(Instant::now()),
            |_capability| {
                callback_called.store(true, Ordering::SeqCst);
                LinearBounds::new(arr2(&[[0.0]]), arr1(&[0.0]), arr2(&[[0.0]]), arr1(&[0.0]))
            },
        );
        assert!(expired_capture.is_err());
        assert!(!callback_called.load(Ordering::SeqCst));

        let host = validated_root_host(
            scope,
            &input_lower,
            &input_upper,
            &[0],
            &[0],
            zero_parts(1, 1),
        );
        assert!(check_root_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0]),),
            Some(Instant::now()),
        )
        .is_err());
        let root = check_root_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0])),
            None,
        )
        .expect("sealed root");
        assert!(bind_root_sound_crown_rows_to_history(
            scope,
            &input_lower,
            &input_upper,
            &history(true),
            "relu",
            &[0],
            &[0],
            &root,
            Some(Instant::now()),
        )
        .is_err());
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
            model_scope: pass.model_scope,
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
            model_scope: CutFoldScope::fresh(),
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
    fn live_model_scope_rejects_token_and_stamp_pair_replay() {
        let h = history(true);
        let graph = GraphNetwork::new();
        let same_model_clone = graph.clone();
        let other_graph = GraphNetwork::new();
        assert_eq!(graph.cut_fold_scope(), same_model_clone.cut_fold_scope());
        assert_ne!(graph.cut_fold_scope(), other_graph.cut_fold_scope());

        let (pass, token) = test_certified_affine_fixture_in_scope(
            [29, 31],
            graph.cut_fold_scope(),
            &[-1.0, -2.0],
            &[1.0, 2.0],
            &h,
            "relu",
            &[0],
            &[0],
            parts(0.0),
        )
        .expect("fixture");
        assert!(token
            .validate_for_clip_in_scope(
                same_model_clone.cut_fold_scope(),
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
        // Move the still-matching token+stamp pair together.  Only comparing
        // those two objects would accept; the live graph scope must reject it.
        assert!(token
            .validate_for_clip_in_scope(
                other_graph.cut_fold_scope(),
                &pass,
                &[-1.0, -2.0],
                &[1.0, 2.0],
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .is_err());
    }

    #[test]
    fn production_root_capture_refuses_l2_before_flatten_or_backward() {
        let graph = GraphNetwork::new();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let l2 = L2Constraint::new(
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::from_elem(IxDyn(&[]), 1.0),
            0,
            &[1],
        )
        .unwrap();
        let input = input.with_l2_constraint(l2);
        let root_bounds = HashMap::new();

        let graph_error = capture_sound_crown_root_rows_at_node(
            &graph,
            "missing",
            &[0],
            &[0],
            &root_bounds,
            &input,
            None,
            None,
        )
        .unwrap_err();
        assert!(graph_error.to_string().contains("refuses L2 input"));

        let identity_error =
            capture_exact_root_input_rows(&graph, &[0], &[0], &input, None).unwrap_err();
        assert!(identity_error.to_string().contains("refuses L2 input"));
    }

    #[test]
    fn production_root_capture_preflights_layout_before_terminal_work() {
        let graph = GraphNetwork::new();
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let root_bounds = HashMap::new();

        let graph_error = capture_sound_crown_root_rows_at_node(
            &graph,
            "missing",
            &[0],
            &[usize::MAX],
            &root_bounds,
            &input,
            None,
            None,
        )
        .unwrap_err();
        assert!(
            graph_error
                .to_string()
                .contains("selected-objective row mapping mismatch"),
            "unexpected refusal: {graph_error}"
        );

        let identity_error =
            capture_exact_root_input_rows(&graph, &[0], &[0, usize::MAX], &input, None)
                .unwrap_err();
        assert!(identity_error
            .to_string()
            .contains("root input identity row map width mismatch"));
    }

    #[test]
    fn production_root_capture_fixed_host_rows_round_trip() {
        // #explicit-cap-overrides-heuristic: this test drives a real CROWN root
        // capture, so it reads the Conv2d memory cap indirectly. `cargo` runs the
        // suite as THREADS OF ONE PROCESS, so a sibling test that sets
        // `NY_CROWN_MEM_CAP_MB` (see `tests::with_crown_mem_cap_mb`) leaks it into
        // this thread. That was harmless while an explicit cap could only LOWER
        // the ceiling through the `min`; now that it displaces the host-RAM
        // heuristic it can also RAISE it, which changes which materialization path
        // runs here. Pin the variable unset for the duration.
        ny_test_utils::env::with_serialized_env_vars_removed(&["NY_CROWN_MEM_CAP_MB"], || {
            production_root_capture_fixed_host_rows_round_trip_inner()
        });
    }

    fn production_root_capture_fixed_host_rows_round_trip_inner() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear",
            Layer::Linear(LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[1.0]))).unwrap()),
        ));
        graph.set_output("linear");
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let root_bounds: HashMap<String, Arc<BoundedTensor>> = graph
            .collect_node_bounds(&input)
            .unwrap()
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();
        let root = capture_sound_crown_root_rows_at_node(
            &graph,
            "linear",
            &[0],
            &[0],
            &root_bounds,
            &input,
            None,
            None,
        )
        .unwrap();
        let history = GraphSplitHistory::new();
        let sealed = bind_root_sound_crown_rows_to_history(
            graph.cut_fold_scope(),
            &[-1.0],
            &[1.0],
            &history,
            "linear",
            &[0],
            &[0],
            &root,
            None,
        )
        .unwrap();
        let (pass, token) = mint_certified_affine_enclosure(sealed, None).unwrap();
        let validated = token
            .validate_for_clip_in_scope(
                graph.cut_fold_scope(),
                &pass,
                &[-1.0],
                &[1.0],
                &history,
                "linear",
                &[0],
                &[0],
                None,
            )
            .unwrap();
        assert_eq!(validated.lower_a(), &arr2(&[[2.0]]));
        assert_eq!(validated.upper_a(), &arr2(&[[2.0]]));
        assert!(validated.lower_b()[0] <= 1.0);
        assert!(validated.upper_b()[0] >= 1.0);
        assert!((validated.lower_b()[0] - 1.0).abs() < 1e-5);
        assert!((validated.upper_b()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pointwise_checker_rejects_equal_scalar_extrema_with_interior_violation() {
        let h = history(true);
        let scope = CutFoldScope::fresh();
        let input_lower = [0.0];
        let input_upper = [1.0];
        let host = validated_host(
            scope,
            &input_lower,
            &input_upper,
            &h,
            &[0],
            &[0],
            zero_parts(1, 1),
        );

        // Host lower is 0. Candidate lower is x. Both concretize to the same
        // scalar lower minimum (0) over [0,1], but x > 0 in the interior.
        let lower_violation =
            suggestion(arr2(&[[1.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0]));
        assert!(check_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            lower_violation,
            None,
        )
        .is_err());

        // Host upper is 0. Candidate upper is -x. Both concretize to the same
        // scalar upper maximum (0), but -x < 0 in the interior.
        let upper_violation =
            suggestion(arr2(&[[0.0]]), arr2(&[[-1.0]]), arr1(&[0.0]), arr1(&[0.0]));
        assert!(check_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            upper_violation,
            None,
        )
        .is_err());
    }

    #[test]
    fn pointwise_checker_rejects_one_ulp_tightening_and_checks_every_row() {
        let h = history(true);
        let scope = CutFoldScope::fresh();
        let input_lower = [-1.0];
        let input_upper = [1.0];
        let selected = [0, 1];
        let row_of = [0, 1];
        let host = validated_host(
            scope,
            &input_lower,
            &input_upper,
            &h,
            &selected,
            &row_of,
            zero_parts(2, 1),
        );
        let one_ulp = f32::from_bits(1);

        // Row zero is exact; only the final row is poisoned.  A sampled-row
        // checker could miss this, while the all-row proof must refuse it.
        let lower_violation = suggestion(
            arr2(&[[0.0], [0.0]]),
            arr2(&[[0.0], [0.0]]),
            arr1(&[0.0, one_ulp]),
            arr1(&[0.0, 0.0]),
        );
        assert!(check_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &selected,
            &row_of,
            &host,
            lower_violation,
            None,
        )
        .is_err());

        let upper_violation = suggestion(
            arr2(&[[0.0], [0.0]]),
            arr2(&[[0.0], [0.0]]),
            arr1(&[0.0, 0.0]),
            arr1(&[0.0, -one_ulp]),
        );
        assert!(check_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &selected,
            &row_of,
            &host,
            upper_violation,
            None,
        )
        .is_err());
    }

    #[test]
    fn checked_rows_factory_mints_fresh_nonreused_passes() {
        let h = history(true);
        let scope = CutFoldScope::fresh();
        let input_lower = [-1.0];
        let input_upper = [1.0];
        let host = validated_host(
            scope,
            &input_lower,
            &input_upper,
            &h,
            &[0],
            &[0],
            zero_parts(1, 1),
        );
        let widened = || suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[-1.0]), arr1(&[1.0]));
        let sealed_a = check_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            widened(),
            None,
        )
        .expect("pointwise-wider suggestion");
        let sealed_b = check_affine_dominance_and_seal(
            scope,
            &input_lower,
            &input_upper,
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            widened(),
            None,
        )
        .expect("pointwise-wider suggestion");
        let (pass_a, token_a) =
            mint_certified_affine_enclosure(sealed_a, None).expect("first mint");
        let (pass_b, _token_b) =
            mint_certified_affine_enclosure(sealed_b, None).expect("second mint");
        assert_ne!(pass_a.words, pass_b.words);

        let checked = token_a
            .validate_for_clip_in_scope(
                scope,
                &pass_a,
                &input_lower,
                &input_upper,
                &h,
                "relu",
                &[0],
                &[0],
                None,
            )
            .expect("factory token validates");
        assert_eq!(checked.lower_b()[0].to_bits(), (-1.0f32).to_bits());
        assert_eq!(checked.upper_b()[0].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn checker_rejects_invalid_errors_context_and_deadlines() {
        let h = history(true);
        let scope = CutFoldScope::fresh();
        let host = validated_host(scope, &[-0.0], &[1.0], &h, &[0], &[0], zero_parts(1, 1));

        let mut negative_error =
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0]));
        negative_error.lower_a_error[[0, 0]] = -f32::EPSILON;
        assert!(check_affine_dominance_and_seal(
            scope,
            &[-0.0],
            &[1.0],
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            negative_error,
            None,
        )
        .is_err());

        let mut nan_error = suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0]));
        nan_error.upper_bias_error[0] = f32::NAN;
        assert!(check_affine_dominance_and_seal(
            scope,
            &[-0.0],
            &[1.0],
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            nan_error,
            None,
        )
        .is_err());

        // Signed zero is part of the exact host/checker context.
        assert!(check_affine_dominance_and_seal(
            scope,
            &[0.0],
            &[1.0],
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0]),),
            None,
        )
        .is_err());

        assert!(check_affine_dominance_and_seal(
            scope,
            &[-0.0],
            &[1.0],
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0]),),
            Some(Instant::now()),
        )
        .is_err());

        let sealed = check_affine_dominance_and_seal(
            scope,
            &[-0.0],
            &[1.0],
            &h,
            "relu",
            &[0],
            &[0],
            &host,
            suggestion(arr2(&[[0.0]]), arr2(&[[0.0]]), arr1(&[0.0]), arr1(&[0.0])),
            None,
        )
        .expect("exact suggestion");
        assert!(mint_certified_affine_enclosure(sealed, Some(Instant::now())).is_err());
    }

    #[test]
    fn pass_counter_carries_and_refuses_wrap() {
        let mut words = [7, u64::MAX];
        assert_eq!(increment_pass_words(&mut words).expect("carry"), [8, 0]);
        let mut exhausted = [u64::MAX, u64::MAX];
        assert!(increment_pass_words(&mut exhausted).is_err());
        assert_eq!(exhausted, [u64::MAX, u64::MAX]);
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
            validate_context_size("relu", &[], &oversized),
            Err(NyError::CpuMemoryExceeded {
                site: "complete_clip_affine_provenance",
                ..
            })
        ));
    }
}
