// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Input-relative provenance conversion for the sequential engine.
//!
//! # The frame problem this module solves
//!
//! The sequential engine's [`IntermediateLinearBounds`] rows are stored in
//! LAYER-OUTPUT coordinates: `bounds_at_layer[k]` is the SPEC objective written
//! as an affine function of layer `k`'s output activation `h_k`
//! (`beta_crown::domain::sequential`, `optimization::intermediate_merge`). The
//! input-space Complete Clipping solver (`clip_interm_domain_full`) wants the
//! opposite direction and a different subject: for each neuron `j` of a layer it
//! needs
//!
//! ```text
//! lA_j · x + lb_j  ≤  h_k[j](x)  ≤  uA_j · x + ub_j        for all x in the box
//! ```
//!
//! i.e. PER-NEURON rows affine in the NETWORK INPUT. No rearrangement of a
//! spec-relative row can produce a per-neuron input-relative row: the two row
//! banks have different row spaces (`output_dim` vs `width(h_k)`) and different
//! column spaces (`width(h_k)` vs `x_dim`). That is why the adapter was
//! quarantined, and why this conversion RE-DERIVES the rows instead of
//! reinterpreting the stored ones.
//!
//! # How the rows are derived
//!
//! By composing an identity seed at the subject layer backward through the
//! certified linear bounds of the prefix — exactly the composition CROWN
//! backward already performs, via the same
//! [`BoundPropagation::propagate_crown_backward`] primitive the production
//! sequential backward pass and `complete_clip_intermediate` use. The shared
//! entry point is [`crown_backward_for_neurons`]; this module adds no new
//! floating-point arithmetic of its own.
//!
//! # Where the outward rounding lives
//!
//! Nowhere in this module: every directed-rounding step is inside machinery this
//! module CALLS, and each one is named here so the audit trail is explicit.
//!
//! 1. Each layer's backward step rounds outward inside
//!    `propagate_crown_backward` (`#2239` directed `f64`→`f32`) and accumulates a
//!    certified per-coefficient error envelope (`LinearBounds::lower_a_err` /
//!    `upper_a_err`, populated by `crown_single::aw_f64_with_abssum`).
//! 2. That envelope is discharged OUTWARD into the bias by
//!    [`BetaCrownVerifier::discharge_coeff_err_for_clip`] →
//!    `LinearBounds::fold_coeff_err_into_bias`, which subtracts the penalty from
//!    the lower bias with `next_down` and adds it to the upper bias with
//!    `next_up`, over the very box the columns multiply. Skipping this step is
//!    what makes a raw `f32` row possibly one ULP TIGHTER than the real-arithmetic
//!    row it claims to be — the exact shape of a false-`unsat`.
//! 3. The solver's own bias merge (`tighten_with_constraints`) closes with
//!    `add_f32_down` / `add_f32_up`.
//!
//! # Fail-closed policy
//!
//! Every failure mode returns `None`, and every caller treats `None` as "drop
//! this constraint / skip this layer". Dropping a split constraint only ENLARGES
//! the solver's feasible region (a weaker necessary condition), and skipping a
//! layer leaves its already-valid enclosure untouched. Both directions are
//! loose-not-wrong.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;

use crate::beta_crown::engine::complete_clip_intermediate::crown_backward_for_neurons;
use crate::{LinearBounds, Network};

use super::super::BetaCrownVerifier;

/// Refuse a conversion whose seed matrix would exceed this many cells
/// (`rows * columns`). 4 Mi cells is 16 MiB per coefficient matrix and four
/// matrices are live at once. A refusal is a skip, not an error.
const MAX_CONVERSION_CELLS: usize = 1 << 22;

/// WHICH quantity a converted row bank's subject is — the thing the rows bound,
/// not the thing they are affine in (they are always affine in the input).
///
/// Getting this wrong is the whole hazard: a row that bounds `h_k` used as if it
/// bounded `h_{k-1}` intersects the solver's constraints against the wrong
/// coordinates and can cut away a region containing the true solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProvenanceSubject {
    /// The network input `x` itself (the pre-activation of layer 0).
    NetworkInput,
    /// Layer `k`'s OUTPUT activation `h_k`, whose concrete box is
    /// `layer_bounds[k]` (see `IntermediateLinearBounds`' CONVENTION note).
    LayerOutput(usize),
}

/// The subject of a `SplitHistory` constraint recorded at `split_layer_idx`.
///
/// A `NeuronConstraint { layer_idx, neuron_idx, .. }` constrains the
/// PRE-ACTIVATION of layer `layer_idx`, i.e. that layer's INPUT — which is
/// layer `layer_idx - 1`'s output, or the network input when `layer_idx == 0`.
/// `create_child_domain` encodes exactly this by clamping
/// `layer_bounds[layer_idx - 1]`, and `optimization::validate` sizes the neuron
/// index against `layer_bounds[layer_idx - 1].len()`.
///
/// The OFF-BY-ONE HERE IS THE BUG CLASS: using `LayerOutput(layer_idx)` would
/// state the split condition about the post-activation instead of the
/// pre-activation, silently producing constraints that are not implied by the
/// branch and can therefore exclude feasible points.
#[inline]
pub(super) const fn split_subject(split_layer_idx: usize) -> ProvenanceSubject {
    match split_layer_idx {
        0 => ProvenanceSubject::NetworkInput,
        layer_idx => ProvenanceSubject::LayerOutput(layer_idx - 1),
    }
}

/// Certified input-relative rows for `neurons` of `subject`, or `None`.
///
/// On success the returned [`LinearBounds`] has one row per entry of `neurons`
/// (in the same order), `input.len()` columns, no residual coefficient-error
/// envelope (it has been folded outward into the bias over `input`), and only
/// finite entries.
///
/// `input` and `layer_bounds` must describe the SAME region: `input` is the
/// domain's effective input box and `layer_bounds[i]` is that domain's enclosure
/// of `h_i` over it. The composed rows are then valid pointwise for every `x` in
/// that box, hence for every `x` in the (smaller) split region the caller cares
/// about.
pub(super) fn input_relative_rows(
    network: &Network,
    input: &BoundedTensor,
    layer_bounds: &[Arc<BoundedTensor>],
    subject: ProvenanceSubject,
    neurons: &[usize],
) -> Option<LinearBounds> {
    if neurons.is_empty() {
        return None;
    }
    let x_dim = input.len();
    let n_rows = neurons.len();

    match subject {
        // The pre-activation of layer 0 IS an input coordinate, so the exact
        // affine identity `x[j] = e_j · x + 0` is the conversion. No composition,
        // no rounding, no envelope: representable exactly in f32.
        ProvenanceSubject::NetworkInput => {
            if n_rows.checked_mul(x_dim)? > MAX_CONVERSION_CELLS {
                return None;
            }
            let mut selector = Array2::<f32>::zeros((n_rows, x_dim));
            for (row, &col) in neurons.iter().enumerate() {
                if col >= x_dim {
                    return None;
                }
                selector[[row, col]] = 1.0;
            }
            LinearBounds::new(
                selector.clone(),
                Array1::zeros(n_rows),
                selector,
                Array1::zeros(n_rows),
            )
            .ok()
        }
        ProvenanceSubject::LayerOutput(target_layer) => {
            if target_layer >= layer_bounds.len() || target_layer >= network.layers.len() {
                return None;
            }
            let target_dim = layer_bounds[target_layer].len();
            if n_rows.checked_mul(target_dim.max(x_dim))? > MAX_CONVERSION_CELLS {
                return None;
            }
            if neurons.iter().any(|&col| col >= target_dim) {
                return None;
            }

            // THE COMPOSITION. Identity seed at `h_target_layer`, then the same
            // per-layer backward step the production CROWN pass runs, over this
            // domain's own pre-activation boxes. Any unsupported layer returns
            // `Err` here and becomes a skip.
            let mut rows =
                crown_backward_for_neurons(network, input, layer_bounds, target_layer, neurons)
                    .ok()?;

            // THE OUTWARD ROUNDING that makes the stored f32 coefficients
            // trustworthy: fold the certified per-coefficient error envelope into
            // the bias over the very box these columns multiply.
            BetaCrownVerifier::discharge_coeff_err_for_clip(&mut rows, input);

            if rows.num_outputs() != n_rows || rows.num_inputs() != x_dim {
                return None;
            }
            // A conversion that has drifted to a residual envelope, a non-finite
            // coefficient (an unsupported layer degraded to `conservative`) or a
            // non-finite bias carries no checker-backed meaning. Refuse it.
            if rows.has_coeff_err() || !rows_are_finite(&rows) {
                return None;
            }
            Some(rows)
        }
    }
}

fn rows_are_finite(rows: &LinearBounds) -> bool {
    rows.lower_a().iter().all(|v| v.is_finite())
        && rows.upper_a().iter().all(|v| v.is_finite())
        && rows.lower_b().iter().all(|v| v.is_finite())
        && rows.upper_b().iter().all(|v| v.is_finite())
}

/// Extract row `row_idx` in the `(lA, lbias, uA, ubias)` shape the split-
/// constraint builder consumes.
pub(super) fn row_at(
    rows: &LinearBounds,
    row_idx: usize,
) -> Option<(Array1<f32>, f32, Array1<f32>, f32)> {
    if row_idx >= rows.num_outputs() {
        return None;
    }
    Some((
        rows.lower_a().row(row_idx).to_owned(),
        rows.lower_b()[row_idx],
        rows.upper_a().row(row_idx).to_owned(),
        rows.upper_b()[row_idx],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2};

    /// `y = W2 · relu(W1 x + b1) + b2` as a 3-layer sequential net.
    ///
    /// Layer 0 = Linear(W1, b1)  -> h_0 (pre-activation)
    /// Layer 1 = ReLU            -> h_1 (post-activation)
    /// Layer 2 = Linear(W2, b2)  -> h_2 (output)
    fn two_layer_network() -> Network {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 1.0], [1.0, -1.0]]), Some(arr1(&[0.0, 0.0])))
                .expect("valid layer 0"),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 2.0]]), Some(arr1(&[0.5]))).expect("valid layer 2"),
        ));
        network
    }

    fn boxed(lower: &[f32], upper: &[f32]) -> Arc<BoundedTensor> {
        Arc::new(
            BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
                .expect("valid test box"),
        )
    }

    /// Root layer bounds for `two_layer_network()` over `x in [-1,1]^2`,
    /// computed by hand:
    ///   h_0 = (x0+x1, x0-x1)  -> both in [-2, 2]
    ///   h_1 = relu(h_0)       -> both in [0, 2]
    ///   h_2 = h_1[0] + 2 h_1[1] + 0.5 -> [0.5, 6.5]
    fn root_bounds() -> (BoundedTensor, Vec<Arc<BoundedTensor>>) {
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .expect("valid input box");
        let layer_bounds = vec![
            boxed(&[-2.0, -2.0], &[2.0, 2.0]),
            boxed(&[0.0, 0.0], &[2.0, 2.0]),
            boxed(&[0.5], &[6.5]),
        ];
        (input, layer_bounds)
    }

    /// HAND-COMPUTABLE ORACLE, layer 0. `h_0` is affine in `x` with no
    /// activation in between, so the converted rows must be EXACT:
    /// `h_0[0] = x0 + x1`, `h_0[1] = x0 - x1`, both biases 0, lower == upper.
    #[test]
    fn layer0_rows_are_the_exact_affine_map() {
        let network = two_layer_network();
        let (input, layer_bounds) = root_bounds();

        let rows = input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::LayerOutput(0),
            &[0, 1],
        )
        .expect("layer 0 conversion must succeed");

        assert_eq!(rows.lower_a().row(0).to_vec(), vec![1.0, 1.0]);
        assert_eq!(rows.upper_a().row(0).to_vec(), vec![1.0, 1.0]);
        assert_eq!(rows.lower_a().row(1).to_vec(), vec![1.0, -1.0]);
        assert_eq!(rows.upper_a().row(1).to_vec(), vec![1.0, -1.0]);
        for i in 0..2 {
            assert!(
                rows.lower_b()[i] <= 0.0 && rows.lower_b()[i] > -1e-6,
                "lower bias must be 0 up to outward rounding, got {}",
                rows.lower_b()[i]
            );
            assert!(
                rows.upper_b()[i] >= 0.0 && rows.upper_b()[i] < 1e-6,
                "upper bias must be 0 up to outward rounding, got {}",
                rows.upper_b()[i]
            );
        }
    }

    /// HAND-COMPUTABLE ORACLE, layer 1 (post-ReLU). Both `h_0` neurons are
    /// unstable over `[-2,2]`, so CROWN's ReLU relaxation with `l=-2, u=2` is
    ///   upper: `0.5*(z + 2)`   (chord through (-2,0) and (2,2))
    ///   lower: `alpha * z` with `alpha in {0,1}`
    /// Composing the upper chord through `h_0[0] = x0 + x1` gives
    /// `0.5*x0 + 0.5*x1 + 1`. Whatever `alpha` the engine picks, the LOWER row
    /// must be one of `0` or `x0 + x1`; both are checked by the containment
    /// property test below, so here we pin the exactly-derivable UPPER row.
    #[test]
    fn layer1_upper_row_is_the_crown_chord() {
        let network = two_layer_network();
        let (input, layer_bounds) = root_bounds();

        let rows = input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::LayerOutput(1),
            &[0],
        )
        .expect("layer 1 conversion must succeed");

        let ua = rows.upper_a().row(0).to_vec();
        assert!(
            (ua[0] - 0.5).abs() < 1e-5 && (ua[1] - 0.5).abs() < 1e-5,
            "expected the 0.5 chord slope composed through W1, got {ua:?}"
        );
        assert!(
            (rows.upper_b()[0] - 1.0).abs() < 1e-4,
            "expected chord intercept 1.0, got {}",
            rows.upper_b()[0]
        );
    }

    /// The split subject is the layer's PRE-activation, one index back. This is
    /// the off-by-one that turns a sound conversion into a false-`unsat`
    /// generator, so it is pinned as a value test rather than left implicit.
    #[test]
    fn split_subject_is_the_pre_activation() {
        assert_eq!(split_subject(0), ProvenanceSubject::NetworkInput);
        assert_eq!(split_subject(1), ProvenanceSubject::LayerOutput(0));
        assert_eq!(split_subject(4), ProvenanceSubject::LayerOutput(3));
    }

    /// A split recorded at layer 0 is a constraint on an input coordinate; the
    /// conversion must return the exact selector row.
    #[test]
    fn network_input_subject_is_the_exact_selector() {
        let network = two_layer_network();
        let (input, layer_bounds) = root_bounds();
        let rows = input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::NetworkInput,
            &[1],
        )
        .expect("input-subject conversion must succeed");
        assert_eq!(rows.lower_a().row(0).to_vec(), vec![0.0, 1.0]);
        assert_eq!(rows.upper_a().row(0).to_vec(), vec![0.0, 1.0]);
        assert_eq!(rows.lower_b()[0], 0.0);
        assert_eq!(rows.upper_b()[0], 0.0);
    }

    /// CONTAINMENT PROPERTY (gate (a) in miniature, at the row level).
    ///
    /// For every layer and every neuron, the converted row must enclose the true
    /// activation at DENSELY SAMPLED inputs. This is the property that fails the
    /// instant the subject index slips by one, the composition direction flips,
    /// or the outward rounding is dropped in a way that matters.
    #[test]
    fn converted_rows_enclose_the_true_activation_on_a_dense_grid() {
        let network = two_layer_network();
        let (input, layer_bounds) = root_bounds();

        const STEPS: usize = 41;
        for target_layer in 0..3 {
            let width = layer_bounds[target_layer].len();
            let neurons: Vec<usize> = (0..width).collect();
            let rows = input_relative_rows(
                &network,
                &input,
                &layer_bounds,
                ProvenanceSubject::LayerOutput(target_layer),
                &neurons,
            )
            .unwrap_or_else(|| panic!("conversion must succeed for layer {target_layer}"));

            for i in 0..STEPS {
                for j in 0..STEPS {
                    let x0 = -1.0 + 2.0 * (i as f32) / ((STEPS - 1) as f32);
                    let x1 = -1.0 + 2.0 * (j as f32) / ((STEPS - 1) as f32);
                    let truth = forward(&[x0, x1], target_layer);
                    for (neuron, &value) in truth.iter().enumerate() {
                        let lower = rows.lower_a()[[neuron, 0]] * x0
                            + rows.lower_a()[[neuron, 1]] * x1
                            + rows.lower_b()[neuron];
                        let upper = rows.upper_a()[[neuron, 0]] * x0
                            + rows.upper_a()[[neuron, 1]] * x1
                            + rows.upper_b()[neuron];
                        assert!(
                            lower <= value + 1e-5,
                            "layer {target_layer} neuron {neuron} at ({x0},{x1}): converted lower \
                             {lower} EXCEEDS the true activation {value} — the row cuts off a \
                             reachable point"
                        );
                        assert!(
                            upper >= value - 1e-5,
                            "layer {target_layer} neuron {neuron} at ({x0},{x1}): converted upper \
                             {upper} is BELOW the true activation {value} — the row cuts off a \
                             reachable point"
                        );
                    }
                }
            }
        }
    }

    /// TEETH for the containment property. Injecting the exact off-by-one this
    /// module exists to prevent — using layer `k`'s rows to bound layer `k+1`'s
    /// activation — must be CAUGHT by the same dense-grid check. A property test
    /// that cannot fail proves nothing.
    #[test]
    fn containment_check_catches_a_one_layer_subject_slip() {
        let network = two_layer_network();
        let (input, layer_bounds) = root_bounds();

        // Rows that bound h_0 (the pre-activation), misused as if they bounded
        // h_1 = relu(h_0). At x = (-1, -1): h_0[0] = -2 but h_1[0] = 0, and the
        // h_0 upper row evaluates to -2 < 0 — a reachable point excluded.
        let wrong = input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::LayerOutput(0),
            &[0],
        )
        .expect("conversion must succeed");

        let (x0, x1) = (-1.0_f32, -1.0_f32);
        let truth = forward(&[x0, x1], 1)[0];
        let upper =
            wrong.upper_a()[[0, 0]] * x0 + wrong.upper_a()[[0, 1]] * x1 + wrong.upper_b()[0];
        assert!(
            upper < truth - 1e-5,
            "the slipped-subject row must VIOLATE containment (upper {upper} should be below the \
             true h_1 value {truth}); if it does not, the containment test has no teeth"
        );
    }

    /// Concrete forward evaluation of `two_layer_network()` up to and including
    /// `target_layer`, written out by hand so the oracle is independent of the
    /// engine's own propagation code.
    fn forward(x: &[f32; 2], target_layer: usize) -> Vec<f32> {
        let h0 = vec![x[0] + x[1], x[0] - x[1]];
        if target_layer == 0 {
            return h0;
        }
        let h1: Vec<f32> = h0.iter().map(|v| v.max(0.0)).collect();
        if target_layer == 1 {
            return h1;
        }
        vec![h1[0] + 2.0 * h1[1] + 0.5]
    }

    /// Out-of-range subjects and neurons are refusals, never panics.
    #[test]
    fn out_of_range_requests_refuse() {
        let network = two_layer_network();
        let (input, layer_bounds) = root_bounds();
        assert!(input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::LayerOutput(99),
            &[0]
        )
        .is_none());
        assert!(input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::LayerOutput(0),
            &[99]
        )
        .is_none());
        assert!(input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::NetworkInput,
            &[99]
        )
        .is_none());
        assert!(input_relative_rows(
            &network,
            &input,
            &layer_bounds,
            ProvenanceSubject::LayerOutput(0),
            &[]
        )
        .is_none());
    }
}
