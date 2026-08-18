// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Invariance / equivariance property verification (NY ext 3).
//!
//! For point-cloud-style networks the scanner pose or point ordering must not
//! change the prediction. This module verifies two property forms, both by
//! reducing to a **difference-network equivalence query** (the same machinery
//! as [`crate::equivalence`] / `build_difference_network`):
//!
//! - [`verify_permutation_invariance`]: `|f(Px) − f(x)| ≤ ε` on a box, where
//!   `P` is a fixed permutation of the (flat) input coordinates, wired as an
//!   exact 0/1 `Linear` layer prepended to `f`. Use [`block_permutation`] to
//!   lift a permutation of *points* to the flat coordinate permutation.
//! - [`verify_rotation_invariance_finite`]: `|f((Iₙ⊗R)x) − f(x)| ≤ ε` for
//!   each rotation `R` in a **fixed finite set**, applied blockwise per
//!   point. v1 accepts exactly the rotations representable as **signed
//!   permutation matrices** (entries in {0, ±1}, one nonzero per row/column,
//!   determinant +1 — e.g. the 24 axis-aligned 90° rotations of
//!   [`octahedral_rotations`]), so the wiring layer and the input-box
//!   invariance check are float-exact. **Continuous SO(3) invariance is out
//!   of scope for v1** — it is not expressible as a finite wiring; certifying
//!   it would need a quantifier over the rotation manifold.
//!
//! The transformed input must stay inside the verified region for the claim
//! "invariant *on this box*" to be meaningful, so both entry points validate
//! that the box is setwise invariant under the transformation and reject
//! non-invariant boxes.
//!
//! Falsification follows the ny-groundtruth witness-search pattern
//! (`crates/ny-groundtruth/src/verify.rs`): when bound propagation cannot
//! prove the property, the difference network is evaluated at grid points via
//! zero-width IBP — a sound enclosure — and [`SymmetryOutcome::Falsified`] is
//! reported only when the enclosure **certainly** violates the tolerance, with
//! the concrete witness point.

use ndarray::{Array1, Array2};
use ny_core::{
    Bound, NyError, Result, VerificationResult, VerificationSoundnessMode, VerificationSpec,
};
use ny_propagate::layers::LinearLayer;
use ny_propagate::{
    build_difference_network, GraphNetwork, GraphNode, Layer, PropagationConfig, PropagationMethod,
    Verifier, NETWORK_INPUT,
};
use ny_tensor::{next_down_f32, BoundedTensor};

/// Outcome of a symmetry (invariance) verification query.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SymmetryOutcome {
    /// `|f(Tx) − f(x)| ≤ ε` is proved on the whole box; `difference_bounds`
    /// are the certified bounds on `f(Tx) − f(x)`.
    Verified {
        /// Certified output bounds of the difference network.
        difference_bounds: Vec<Bound>,
    },
    /// A concrete point in the box certainly violates invariance: the sound
    /// enclosure of `f(Tx*) − f(x*)` lies strictly outside `[−ε, ε]`.
    Falsified {
        /// The witness point `x*` (inside the input box).
        witness: Vec<f32>,
        /// Sound enclosure of the violating output difference at `x*`.
        difference: Bound,
    },
    /// Neither proved nor concretely falsified (bounds too loose and the
    /// witness grid found no certain violation).
    Unknown {
        /// Best achieved bounds on `f(Tx) − f(x)` over the box.
        difference_bounds: Vec<Bound>,
    },
}

impl SymmetryOutcome {
    /// True when the invariance property was proved.
    #[must_use]
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

/// Options for the `_with` variants.
#[derive(Debug, Clone)]
pub struct SymmetryOptions {
    /// Bound-propagation configuration for the difference network. Defaults
    /// to CROWN — plain IBP decorrelates the two copies of the shared input
    /// and cannot prove invariance even for exactly symmetric networks.
    pub config: PropagationConfig,
    /// Witness-search grid resolution per input dimension (min 2 unless the
    /// dimension is degenerate); the total point count is capped.
    pub witness_grid: usize,
}

impl Default for SymmetryOptions {
    fn default() -> Self {
        Self {
            config: PropagationConfig {
                method: PropagationMethod::Crown,
                ..Default::default()
            },
            witness_grid: 5,
        }
    }
}

/// Per-rotation outcomes for [`verify_rotation_invariance_finite`], aligned
/// with the input rotation slice.
#[derive(Debug, Clone)]
pub struct FiniteRotationOutcome {
    /// One outcome per rotation, in input order.
    pub per_rotation: Vec<SymmetryOutcome>,
}

impl FiniteRotationOutcome {
    /// True when invariance was proved for **every** rotation in the set.
    #[must_use]
    pub fn all_verified(&self) -> bool {
        self.per_rotation.iter().all(SymmetryOutcome::is_verified)
    }

    /// The first rotation index with a concrete counterexample, if any.
    #[must_use]
    pub fn first_falsified(&self) -> Option<(usize, &SymmetryOutcome)> {
        self.per_rotation
            .iter()
            .enumerate()
            .find(|(_, o)| matches!(o, SymmetryOutcome::Falsified { .. }))
    }
}

/// Cap on the total number of witness-grid evaluations (same policy as the
/// ny-groundtruth witness search).
const MAX_WITNESS_POINTS: usize = 20_000;

/// Resource ceiling for the dense wiring matrix (64 MiB of `f32` values).
/// Permutations and block rotations are sparse conceptually; until the graph
/// has a sparse wiring layer, reject dimensions that would amplify a compact
/// request into an unbounded quadratic allocation.
const MAX_DENSE_WIRING_ELEMENTS: usize = 16 * 1024 * 1024;

/// Name of the prepended wiring node inside the transformed copy of `f`.
const WIRE_NODE: &str = "sym_wire";

fn dense_wiring_elements(dimension: usize) -> Result<usize> {
    let elements = dimension.checked_mul(dimension).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "dense symmetry wiring size overflows usize: {dimension} × {dimension}"
        ))
    })?;
    if elements > MAX_DENSE_WIRING_ELEMENTS {
        return Err(NyError::InvalidSpec(format!(
            "dense symmetry wiring requires {elements} elements, exceeding the \
             {MAX_DENSE_WIRING_ELEMENTS}-element resource limit"
        )));
    }
    Ok(elements)
}

fn zeroed_square_wiring(dimension: usize) -> Result<Array2<f32>> {
    let elements = dense_wiring_elements(dimension)?;
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        NyError::InvalidSpec(format!(
            "dense symmetry wiring with {elements} elements cannot be allocated: {error}"
        ))
    })?;
    values.resize(elements, 0.0);
    Array2::from_shape_vec((dimension, dimension), values).map_err(|error| {
        NyError::InternalError(format!(
            "validated dense symmetry wiring shape was rejected: {error}"
        ))
    })
}

/// Lift a permutation of **points** to the flat input-coordinate permutation
/// for a point cloud stored as `n` consecutive blocks of `point_dim`
/// coordinates: flat index `p·point_dim + c` reads
/// `point_permutation[p]·point_dim + c`.
///
/// # Errors
///
/// Rejects `point_dim == 0`, slices that are not permutations, and flattened
/// sizes that overflow or cannot be represented by a `Vec`.
pub fn block_permutation(point_permutation: &[usize], point_dim: usize) -> Result<Vec<usize>> {
    if point_dim == 0 {
        return Err(NyError::InvalidSpec(
            "point_dim must be at least 1".to_string(),
        ));
    }
    validate_permutation(point_permutation)?;
    let flat_len = point_permutation
        .len()
        .checked_mul(point_dim)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "flattened permutation size overflows usize: {} points × {point_dim} coordinates",
                point_permutation.len()
            ))
        })?;
    let mut flat = Vec::new();
    flat.try_reserve_exact(flat_len).map_err(|e| {
        NyError::InvalidSpec(format!(
            "flattened permutation with {flat_len} entries cannot be represented: {e}"
        ))
    })?;
    for &src_point in point_permutation {
        let base = src_point.checked_mul(point_dim).ok_or_else(|| {
            NyError::InvalidSpec("flattened permutation index overflows usize".to_string())
        })?;
        for c in 0..point_dim {
            flat.push(base + c);
        }
    }
    Ok(flat)
}

/// The 24 proper axis-aligned rotations of 3-space: all signed permutation
/// matrices (entries in {0, ±1}) with determinant +1. This is the standard
/// finite rotation set for scanner-pose invariance checks; feed it to
/// [`verify_rotation_invariance_finite`].
#[must_use]
pub fn octahedral_rotations() -> Vec<Array2<f32>> {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
    ];
    // Parity of each permutation above (even = +1, odd = −1).
    const PERM_SIGNS: [f32; 6] = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

    let mut rotations = Vec::with_capacity(24);
    for (perm, perm_sign) in PERMS.iter().zip(PERM_SIGNS) {
        for signs_bits in 0..8_u8 {
            let signs = [
                if signs_bits & 1 == 0 { 1.0_f32 } else { -1.0 },
                if signs_bits & 2 == 0 { 1.0_f32 } else { -1.0 },
                if signs_bits & 4 == 0 { 1.0_f32 } else { -1.0 },
            ];
            let det = perm_sign * signs[0] * signs[1] * signs[2];
            if det != 1.0 {
                continue;
            }
            let mut m = Array2::<f32>::zeros((3, 3));
            for (row, (&col, &sign)) in perm.iter().zip(&signs).enumerate() {
                m[[row, col]] = sign;
            }
            rotations.push(m);
        }
    }
    debug_assert_eq!(rotations.len(), 24);
    rotations
}

/// Verify that `f` is invariant (up to `epsilon`) under a permutation of its
/// flat input coordinates, over the box `input_bounds`, with default options
/// (CROWN + 5-per-dimension witness grid).
///
/// `permutation[j] = i` means transformed coordinate `j` reads original
/// coordinate `i`, i.e. the property checked is
/// `∀x ∈ box: |f(x ∘ permutation) − f(x)| ≤ ε` (element-wise on outputs).
///
/// # Errors
///
/// Rejects invalid permutations, an `epsilon` that is not strictly positive
/// after sound `f32` rounding, and boxes that are not setwise invariant under
/// the permutation (the property would leave the verified region).
pub fn verify_permutation_invariance(
    network: &GraphNetwork,
    permutation: &[usize],
    input_bounds: &[Bound],
    epsilon: f64,
) -> Result<SymmetryOutcome> {
    verify_permutation_invariance_with(
        network,
        permutation,
        input_bounds,
        epsilon,
        &SymmetryOptions::default(),
    )
}

/// [`verify_permutation_invariance`] with explicit options.
pub fn verify_permutation_invariance_with(
    network: &GraphNetwork,
    permutation: &[usize],
    input_bounds: &[Bound],
    epsilon: f64,
    options: &SymmetryOptions,
) -> Result<SymmetryOutcome> {
    // Enforce the quadratic resource budget before validation allocates even
    // its linear-size bookkeeping for a caller-controlled dimension.
    dense_wiring_elements(permutation.len())?;
    validate_permutation(permutation)?;
    if permutation.len() != input_bounds.len() {
        return Err(NyError::InvalidSpec(format!(
            "permutation length {} does not match input dimension {}",
            permutation.len(),
            input_bounds.len()
        )));
    }
    // Box invariance: coordinate j of the permuted input ranges over the box
    // of source coordinate permutation[j]; it must equal coordinate j's box.
    for (j, &src) in permutation.iter().enumerate() {
        let (dst_b, src_b) = (&input_bounds[j], &input_bounds[src]);
        if dst_b.lower() != src_b.lower() || dst_b.upper() != src_b.upper() {
            return Err(NyError::InvalidSpec(format!(
                "input box is not invariant under the permutation: coordinate {j} has bounds \
                 [{}, {}] but reads coordinate {src} with bounds [{}, {}]; permutation \
                 invariance on this box is ill-posed",
                dst_b.lower(),
                dst_b.upper(),
                src_b.lower(),
                src_b.upper()
            )));
        }
    }

    let n = permutation.len();
    let mut weight = zeroed_square_wiring(n)?;
    for (j, &src) in permutation.iter().enumerate() {
        weight[[j, src]] = 1.0;
    }
    verify_wired_difference(network, weight, input_bounds, epsilon, options)
}

/// Verify that `f` is invariant (up to `epsilon`) under each rotation in a
/// fixed finite set, applied blockwise to consecutive point blocks of the
/// input, over the box `input_bounds`. Default options (CROWN + 5-point
/// witness grid).
///
/// Each rotation must be a **signed permutation matrix** (entries in
/// {0, ±1}, exactly one nonzero per row and column) with determinant +1 —
/// e.g. any subset of [`octahedral_rotations`]. General rotation matrices
/// (continuous SO(3)) are out of scope for v1; see the module docs.
///
/// # Errors
///
/// Rejects an empty rotation set, non-signed-permutation or reflection
/// (det = −1) matrices, an input dimension not divisible by the rotation
/// dimension, an `epsilon` that is not strictly positive after sound `f32`
/// rounding, and boxes not setwise invariant under a rotation.
pub fn verify_rotation_invariance_finite(
    network: &GraphNetwork,
    rotations: &[Array2<f32>],
    input_bounds: &[Bound],
    epsilon: f64,
) -> Result<FiniteRotationOutcome> {
    verify_rotation_invariance_finite_with(
        network,
        rotations,
        input_bounds,
        epsilon,
        &SymmetryOptions::default(),
    )
}

/// [`verify_rotation_invariance_finite`] with explicit options.
pub fn verify_rotation_invariance_finite_with(
    network: &GraphNetwork,
    rotations: &[Array2<f32>],
    input_bounds: &[Bound],
    epsilon: f64,
    options: &SymmetryOptions,
) -> Result<FiniteRotationOutcome> {
    if rotations.is_empty() {
        return Err(NyError::InvalidSpec(
            "rotation set must not be empty".to_string(),
        ));
    }
    let mut per_rotation = Vec::new();
    per_rotation
        .try_reserve_exact(rotations.len())
        .map_err(|error| {
            NyError::InvalidSpec(format!(
                "rotation result set with {} entries cannot be allocated: {error}",
                rotations.len()
            ))
        })?;
    for (r_idx, rotation) in rotations.iter().enumerate() {
        let signed = validate_signed_permutation_rotation(rotation, r_idx)?;
        let d = signed.len();
        if input_bounds.is_empty() || !input_bounds.len().is_multiple_of(d) {
            return Err(NyError::InvalidSpec(format!(
                "input dimension {} is not a positive multiple of the rotation dimension {d}",
                input_bounds.len()
            )));
        }
        let num_blocks = input_bounds.len() / d;

        // Box invariance, exactly: transformed coordinate p·d + row reads
        // ±(x at p·d + col); its exact image interval must equal the box.
        for p in 0..num_blocks {
            for (row, &(col, sign)) in signed.iter().enumerate() {
                let dst = &input_bounds[p * d + row];
                let src = &input_bounds[p * d + col];
                let (img_lo, img_hi) = if sign > 0.0 {
                    (src.lower(), src.upper())
                } else {
                    (-src.upper(), -src.lower())
                };
                if dst.lower() != img_lo || dst.upper() != img_hi {
                    return Err(NyError::InvalidSpec(format!(
                        "input box is not invariant under rotation {r_idx}: coordinate {} has \
                         bounds [{}, {}] but the rotated input ranges over [{img_lo}, {img_hi}]; \
                         rotation invariance on this box is ill-posed",
                        p * d + row,
                        dst.lower(),
                        dst.upper()
                    )));
                }
            }
        }

        // Block-diagonal wiring Iₙ ⊗ R (exact: entries are 0/±1).
        let n = input_bounds.len();
        let mut weight = zeroed_square_wiring(n)?;
        for p in 0..num_blocks {
            for (row, &(col, sign)) in signed.iter().enumerate() {
                weight[[p * d + row, p * d + col]] = sign;
            }
        }
        per_rotation.push(verify_wired_difference(
            network,
            weight,
            input_bounds,
            epsilon,
            options,
        )?);
    }
    Ok(FiniteRotationOutcome { per_rotation })
}

/// Check that `permutation` is a permutation of `0..len`.
fn validate_permutation(permutation: &[usize]) -> Result<()> {
    let n = permutation.len();
    if n == 0 {
        return Err(NyError::InvalidSpec(
            "permutation must not be empty".to_string(),
        ));
    }
    let mut seen = vec![false; n];
    for &i in permutation {
        if i >= n {
            return Err(NyError::InvalidSpec(format!(
                "permutation entry {i} is out of range for length {n}"
            )));
        }
        if seen[i] {
            return Err(NyError::InvalidSpec(format!(
                "permutation repeats entry {i}"
            )));
        }
        seen[i] = true;
    }
    Ok(())
}

/// Validate a v1 rotation: square, entries in {0, ±1}, exactly one nonzero
/// per row and per column, determinant +1. Returns per-row `(column, sign)`.
fn validate_signed_permutation_rotation(
    rotation: &Array2<f32>,
    r_idx: usize,
) -> Result<Vec<(usize, f32)>> {
    let (rows, cols) = rotation.dim();
    if rows == 0 || rows != cols {
        return Err(NyError::InvalidSpec(format!(
            "rotation {r_idx} must be a non-empty square matrix, got {rows}x{cols}"
        )));
    }
    let mut signed = Vec::with_capacity(rows);
    let mut col_used = vec![false; cols];
    for row in 0..rows {
        let mut entry: Option<(usize, f32)> = None;
        for col in 0..cols {
            let v = rotation[[row, col]];
            if v == 0.0 {
                continue;
            }
            if v != 1.0 && v != -1.0 {
                return Err(NyError::InvalidSpec(format!(
                    "rotation {r_idx} entry ({row}, {col}) is {v}; v1 supports only signed \
                     permutation matrices (entries in {{0, ±1}}) — continuous SO(3) rotations \
                     are out of scope"
                )));
            }
            if entry.is_some() {
                return Err(NyError::InvalidSpec(format!(
                    "rotation {r_idx} row {row} has more than one nonzero entry; v1 supports \
                     only signed permutation matrices"
                )));
            }
            entry = Some((col, v));
        }
        let (col, sign) = entry.ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "rotation {r_idx} row {row} is all zeros; not a rotation"
            ))
        })?;
        if col_used[col] {
            return Err(NyError::InvalidSpec(format!(
                "rotation {r_idx} column {col} is used by more than one row; not a rotation"
            )));
        }
        col_used[col] = true;
        signed.push((col, sign));
    }

    // Determinant = parity of the permutation × product of signs; must be +1
    // (−1 would be a reflection, which is not a rotation).
    let perm: Vec<usize> = signed.iter().map(|&(col, _)| col).collect();
    let parity = permutation_parity(&perm);
    let sign_product: f32 = signed.iter().map(|&(_, s)| s).product();
    if parity * sign_product != 1.0 {
        return Err(NyError::InvalidSpec(format!(
            "matrix {r_idx} has determinant -1 (a reflection, not a rotation)"
        )));
    }
    Ok(signed)
}

/// Parity (+1 even / −1 odd) of a permutation, by counting transpositions.
fn permutation_parity(perm: &[usize]) -> f32 {
    let mut perm = perm.to_vec();
    let mut parity = 1.0_f32;
    for i in 0..perm.len() {
        while perm[i] != i {
            let j = perm[i];
            perm.swap(i, j);
            parity = -parity;
        }
    }
    parity
}

/// Build the transformed copy `x ↦ f(Wx)`: a `Linear(W)` wiring node reading
/// `NETWORK_INPUT`, with every node of `f` re-rooted onto it (names prefixed
/// to avoid collisions, mirroring `build_difference_network`).
fn prepend_wiring(network: &GraphNetwork, weight: Array2<f32>) -> Result<GraphNetwork> {
    if network.output_name().is_empty() {
        return Err(NyError::InvalidSpec(
            "network has no output node set".to_string(),
        ));
    }
    let wire_layer = LinearLayer::new(weight, None)?;

    // Choose a prefix that cannot collide with the wiring node or the input
    // sentinel after prefixing.
    let prefix = ["t_", "wired_", "transformed_"]
        .into_iter()
        .find(|prefix| {
            network.node_names().iter().all(|name| {
                let prefixed = format!("{prefix}{name}");
                prefixed != WIRE_NODE && prefixed != NETWORK_INPUT
            })
        })
        .ok_or_else(|| {
            NyError::InvalidSpec("could not find a collision-free node prefix".to_string())
        })?;

    let mut wired = GraphNetwork::new();
    wired.try_add_node(GraphNode::from_input(WIRE_NODE, Layer::Linear(wire_layer)))?;
    for name in network.node_names() {
        let node = network
            .node(name)
            .ok_or_else(|| NyError::InternalError(format!("node '{name}' missing")))?;
        let inputs: Vec<String> = node
            .inputs()
            .iter()
            .map(|input| {
                if input == NETWORK_INPUT {
                    WIRE_NODE.to_string()
                } else {
                    format!("{prefix}{input}")
                }
            })
            .collect();
        wired.try_add_node(GraphNode::new(
            format!("{prefix}{name}"),
            node.layer().clone(),
            inputs,
        ))?;
    }
    wired.set_output(format!("{prefix}{}", network.output_name()));
    Ok(wired)
}

/// Shared engine: build `h(x) = f(Wx) − f(x)`, try to prove `|h| ≤ ε` on the
/// box, and fall back to the sound grid witness search on failure.
fn verify_wired_difference(
    network: &GraphNetwork,
    weight: Array2<f32>,
    input_bounds: &[Bound],
    epsilon: f64,
    options: &SymmetryOptions,
) -> Result<SymmetryOutcome> {
    let eps = validated_epsilon(epsilon)?;
    let wired = prepend_wiring(network, weight)?;
    let h = build_difference_network(&wired, network)?;

    // Cheap probe: determines the output dimension and catches shape
    // mismatches up front (same approach as verify_equivalence).
    let input_tensor = Verifier::bounds_to_tensor(input_bounds, None)?;
    let probe = h.propagate_ibp_sound(&input_tensor)?;
    let num_outputs = probe.lower().len();

    let spec = VerificationSpec::new(
        input_bounds.to_vec(),
        vec![Bound::new(-eps, eps); num_outputs],
    )?;
    let verifier = Verifier::new(options.config.clone());
    let result = verifier.verify_graph(&h, &spec)?;

    if let Some(output_bounds) = sound_verified_bounds(&result) {
        return Ok(SymmetryOutcome::Verified {
            difference_bounds: output_bounds.to_vec(),
        });
    }
    let best_bounds = match result {
        // Heuristic provenance cannot establish a universal symmetry. Retain
        // its bounds as a best effort and continue with sound witness search.
        VerificationResult::Verified { output_bounds, .. } => output_bounds,
        VerificationResult::Violated { counterexample, .. } => {
            // Trust but verify: only report Falsified when a sound concrete
            // evaluation confirms the violation.
            if point_in_box(&counterexample, input_bounds) {
                if let Some(outcome) = certain_violation_at(&h, &counterexample, eps)? {
                    return Ok(outcome);
                }
            }
            enclosure_bounds(&probe)
        }
        VerificationResult::Unknown { bounds, .. } => bounds,
        VerificationResult::Timeout { partial_bounds, .. } => {
            partial_bounds.unwrap_or_else(|| enclosure_bounds(&probe))
        }
    };

    if let Some(outcome) = grid_witness(&h, input_bounds, eps, options.witness_grid)? {
        return Ok(outcome);
    }
    Ok(SymmetryOutcome::Unknown {
        difference_bounds: best_bounds,
    })
}

fn sound_verified_bounds(result: &VerificationResult) -> Option<&[Bound]> {
    match result {
        VerificationResult::Verified {
            provenance,
            output_bounds,
            ..
        } if provenance.mode() == VerificationSoundnessMode::Sound => Some(output_bounds),
        _ => None,
    }
}

/// Validate ε and round it down to `f32` (sound direction: the checked
/// property is at least as strong as the `f64` request).
fn validated_epsilon(epsilon: f64) -> Result<f32> {
    if !epsilon.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "epsilon must be finite, got {epsilon}"
        )));
    }
    let mut eps32 = epsilon as f32;
    if f64::from(eps32) > epsilon {
        eps32 = next_down_f32(eps32);
    }
    if eps32 <= 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "epsilon must be strictly positive (after sound f32 rounding), got {epsilon}"
        )));
    }
    Ok(eps32)
}

/// Does the sound enclosure `[lo, hi]` certainly violate `|h| ≤ ε`?
fn certainly_violates(eps: f32, lo: f32, hi: f32) -> bool {
    lo > eps || hi < -eps
}

/// Evaluate `h` at a concrete point via zero-width IBP (sound enclosure) and
/// return a Falsified outcome if any output certainly violates the tolerance.
fn certain_violation_at(
    h: &GraphNetwork,
    point: &[f32],
    eps: f32,
) -> Result<Option<SymmetryOutcome>> {
    let arr = Array1::from(point.to_vec()).into_dyn();
    let tensor = BoundedTensor::new(arr.clone(), arr)?;
    let enclosure = h.propagate_ibp_sound(&tensor)?;
    for (&lo, &hi) in enclosure.lower().iter().zip(enclosure.upper().iter()) {
        if certainly_violates(eps, lo, hi) {
            return Ok(Some(SymmetryOutcome::Falsified {
                witness: point.to_vec(),
                difference: Bound::new_allow_infinite(lo, hi),
            }));
        }
    }
    Ok(None)
}

fn point_in_box(point: &[f32], input_bounds: &[Bound]) -> bool {
    point.len() == input_bounds.len()
        && point
            .iter()
            .zip(input_bounds.iter())
            .all(|(&x, b)| x >= b.lower() && x <= b.upper())
}

/// Convert an IBP output enclosure into per-output `Bound`s.
fn enclosure_bounds(enclosure: &BoundedTensor) -> Vec<Bound> {
    enclosure
        .lower()
        .iter()
        .zip(enclosure.upper().iter())
        .map(|(&lo, &hi)| Bound::new_allow_infinite(lo, hi))
        .collect()
}

/// Number of grid points to evaluate, capped even when the Cartesian product
/// overflows `usize` or the minimum two-point resolution is already too large.
fn grid_point_budget(counts: &[usize]) -> usize {
    counts
        .iter()
        .try_fold(1_usize, |acc, &count| acc.checked_mul(count))
        .unwrap_or(MAX_WITNESS_POINTS)
        .min(MAX_WITNESS_POINTS)
}

fn capped_grid_resolution(requested: usize, varying_dimensions: usize) -> usize {
    let requested = requested.clamp(2, MAX_WITNESS_POINTS);
    let fits = |resolution: usize| {
        (0..varying_dimensions)
            .try_fold(1_usize, |product, _| {
                product
                    .checked_mul(resolution)
                    .filter(|&next| next <= MAX_WITNESS_POINTS)
            })
            .is_some()
    };
    if varying_dimensions <= 1 || !fits(2) {
        return if varying_dimensions <= 1 {
            requested
        } else {
            2
        };
    }

    let (mut low, mut high) = (2, requested);
    while low < high {
        let midpoint = low + (high - low).div_ceil(2);
        if fits(midpoint) {
            low = midpoint;
        } else {
            high = midpoint - 1;
        }
    }
    low
}

/// Grid witness search over the input box (ny-groundtruth pattern): evaluate
/// `h` at evenly spaced points (endpoints included) and return the first
/// point whose sound enclosure certainly violates the tolerance.
fn grid_witness(
    h: &GraphNetwork,
    input_bounds: &[Bound],
    eps: f32,
    grid: usize,
) -> Result<Option<SymmetryOutcome>> {
    let dim = input_bounds.len();
    if dim == 0
        || input_bounds
            .iter()
            .any(|b| !b.lower().is_finite() || !b.upper().is_finite())
    {
        return Ok(None); // no finite box to sample
    }

    // Per-dimension point counts: degenerate dimensions get one point;
    // shrink the resolution until the total fits the cap.
    let varying_dimensions = input_bounds
        .iter()
        .filter(|bound| bound.lower() != bound.upper())
        .count();
    let resolution = capped_grid_resolution(grid, varying_dimensions);
    let counts: Vec<usize> = input_bounds
        .iter()
        .map(|bound| {
            if bound.lower() == bound.upper() {
                1
            } else {
                resolution
            }
        })
        .collect();

    let mut index = vec![0_usize; dim];
    for _ in 0..grid_point_budget(&counts) {
        let point: Vec<f32> = index
            .iter()
            .zip(input_bounds.iter())
            .zip(counts.iter())
            .map(|((&i, b), &n)| {
                if n == 1 {
                    b.lower()
                } else {
                    let t = i as f64 / (n - 1) as f64;
                    let width = f64::from(b.upper()) - f64::from(b.lower());
                    let x = f64::from(b.lower()) + t * width;
                    // Clamp so FP rounding cannot push the sample outside the box.
                    (x as f32).clamp(b.lower(), b.upper())
                }
            })
            .collect();
        if let Some(outcome) = certain_violation_at(h, &point, eps)? {
            return Ok(Some(outcome));
        }

        // Odometer increment.
        let mut carry = true;
        for (i, count) in index.iter_mut().zip(counts.iter()) {
            *i += 1;
            if *i < *count {
                carry = false;
                break;
            }
            *i = 0;
        }
        if carry {
            return Ok(None);
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::SoundnessProvenance;

    fn verified_with(provenance: SoundnessProvenance) -> VerificationResult {
        VerificationResult::Verified {
            provenance,
            output_bounds: vec![Bound::new(-1.0, 1.0)],
            proof: None,
            actual_method: None,
        }
    }

    #[test]
    fn only_sound_verified_results_are_treated_as_proofs() {
        assert!(sound_verified_bounds(&verified_with(SoundnessProvenance::sound())).is_some());
        assert!(
            sound_verified_bounds(&verified_with(SoundnessProvenance::heuristic())).is_none(),
            "heuristic bounds must not become an unqualified Verified outcome"
        );
    }

    #[test]
    fn block_permutation_expands_point_blocks() {
        assert_eq!(
            block_permutation(&[1, 0, 2], 2).unwrap(),
            vec![2, 3, 0, 1, 4, 5]
        );
        assert!(block_permutation(&[1, 1], 2).is_err());
        assert!(block_permutation(&[0, 1], 0).is_err());
        assert!(block_permutation(&[], 3).is_err());
        assert!(
            block_permutation(&[1, 0], usize::MAX).is_err(),
            "flattened-size overflow must be returned, not panic"
        );
        assert!(
            block_permutation(&[0], usize::MAX).is_err(),
            "unrepresentable Vec capacity must be returned, not panic"
        );
    }

    #[test]
    fn dense_wiring_rejects_quadratic_resource_amplification() {
        assert!(zeroed_square_wiring(usize::MAX).is_err());
        let over_limit_dimension = (MAX_DENSE_WIRING_ELEMENTS as f64).sqrt() as usize + 1;
        assert!(
            zeroed_square_wiring(over_limit_dimension).is_err(),
            "oversized dimensions must reject before ndarray allocation"
        );
    }

    #[test]
    fn witness_grid_budget_is_hard_capped() {
        assert_eq!(grid_point_budget(&[2, 3, 4]), 24);
        assert_eq!(
            grid_point_budget(&vec![2; usize::BITS as usize]),
            MAX_WITNESS_POINTS,
            "overflowing Cartesian products must still honor the cap"
        );
        assert_eq!(
            grid_point_budget(&[MAX_WITNESS_POINTS, 2]),
            MAX_WITNESS_POINTS
        );
        let two_dimensional = capped_grid_resolution(usize::MAX, 2);
        assert!(
            two_dimensional * two_dimensional <= MAX_WITNESS_POINTS
                && (two_dimensional + 1) * (two_dimensional + 1) > MAX_WITNESS_POINTS,
            "resolution must be the largest square grid within the hard cap"
        );
        assert_eq!(capped_grid_resolution(usize::MAX, 20), 2);
    }

    #[test]
    fn octahedral_rotation_set_is_the_24_proper_rotations() {
        let rotations = octahedral_rotations();
        assert_eq!(rotations.len(), 24);
        // All distinct, and all pass the signed-permutation + det +1 gate.
        for (i, r) in rotations.iter().enumerate() {
            assert!(
                validate_signed_permutation_rotation(r, i).is_ok(),
                "rotation {i} must validate"
            );
            for (j, s) in rotations.iter().enumerate() {
                if i != j {
                    assert_ne!(r, s, "rotations {i} and {j} must be distinct");
                }
            }
        }
    }

    #[test]
    fn reflections_and_non_signed_matrices_are_rejected() {
        // det = −1 reflection.
        let mut refl = Array2::<f32>::eye(3);
        refl[[2, 2]] = -1.0;
        let err = validate_signed_permutation_rotation(&refl, 0).unwrap_err();
        assert!(err.to_string().contains("reflection"), "{err}");
        // A continuous rotation (45° about z) is out of scope in v1.
        let c = std::f32::consts::FRAC_1_SQRT_2;
        let cont = ndarray::arr2(&[[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]]);
        assert!(validate_signed_permutation_rotation(&cont, 0).is_err());
        // Two nonzeros in one row.
        let bad = ndarray::arr2(&[[1.0_f32, 1.0], [0.0, 1.0]]);
        assert!(validate_signed_permutation_rotation(&bad, 0).is_err());
    }

    #[test]
    fn permutation_parity_matches_transposition_count() {
        assert_eq!(permutation_parity(&[0, 1, 2]), 1.0);
        assert_eq!(permutation_parity(&[1, 0, 2]), -1.0);
        assert_eq!(permutation_parity(&[1, 2, 0]), 1.0);
    }

    #[test]
    fn epsilon_is_rounded_toward_zero_and_validated() {
        let eps = validated_epsilon(1e-9).unwrap();
        assert!(f64::from(eps) <= 1e-9 && eps > 0.0);
        assert!(validated_epsilon(0.0).is_err());
        assert!(validated_epsilon(-1.0).is_err());
        assert!(validated_epsilon(f64::NAN).is_err());
    }

    #[test]
    fn violation_predicate_is_strict() {
        assert!(certainly_violates(1.0, 1.5, 2.0));
        assert!(certainly_violates(1.0, -3.0, -1.5));
        assert!(!certainly_violates(1.0, -0.5, 0.5));
        assert!(!certainly_violates(1.0, 0.5, 1.5));
    }
}
