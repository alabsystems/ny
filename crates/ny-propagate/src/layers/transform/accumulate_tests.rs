// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{IndexAddLayer, ScatterAddLayer};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

fn make_bounded(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).expect("valid lower"),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).expect("valid upper"),
    )
    .expect("bounded tensor should construct")
}

#[test]
fn scatter_add_activation_input_count_tracks_embedded_constants() {
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::zeros(IxDyn(&[4]))),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 1]).unwrap()),
        None,
    );
    assert_eq!(layer.activation_input_count(), 1);
}

#[test]
fn scatter_add_static_indices_adds_exactly() {
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::from_shape_vec(IxDyn(&[4]), vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1_i64, 1]).unwrap()),
        None,
    );
    let src = make_bounded(&[2.0, -3.0], &[4.0, -1.0], &[2]);
    let output = layer
        .propagate_ibp(&src)
        .expect("ScatterAdd IBP should succeed");
    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[10.0, 19.0, 30.0, 40.0]
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        &[10.0, 23.0, 30.0, 40.0]
    );
}

#[test]
fn scatter_add_dynamic_indices_adds_global_contribution_range() {
    let layer = ScatterAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[3]))), None, None);
    let index = make_bounded(&[0.0, 0.0], &[2.0, 2.0], &[2]);
    let src = make_bounded(&[-2.0, 1.5], &[3.0, 4.0], &[2]);
    let output = layer
        .propagate_ibp_binary(&index, &src)
        .expect("ScatterAdd IBP should succeed");
    assert_eq!(output.lower().as_slice().unwrap(), &[-2.0, -2.0, -2.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[7.0, 7.0, 7.0]);
}

#[test]
fn index_add_static_indices_adds_selected_rows() {
    let layer = IndexAddLayer::new(
        1,
        Some(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
        Some(arr1(&[2_i64, 0]).into_dyn()),
        None,
    );
    let src = make_bounded(&[10.0, -1.0, 20.0, -2.0], &[10.0, 1.0, 20.0, 2.0], &[2, 2]);
    let output = layer
        .propagate_ibp(&src)
        .expect("IndexAdd IBP should succeed");
    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[0.0, 2.0, 13.0, 2.0, 5.0, 26.0]
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        &[2.0, 2.0, 13.0, 6.0, 5.0, 26.0]
    );
}

#[test]
fn index_add_dynamic_indices_adds_global_contribution_range() {
    let layer = IndexAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[3]))), None, None);
    let index = make_bounded(&[0.0, 0.0], &[2.0, 2.0], &[2]);
    let src = make_bounded(&[-1.0, 2.0], &[3.0, 4.0], &[2]);
    let output = layer
        .propagate_ibp_binary(&index, &src)
        .expect("IndexAdd IBP should succeed");
    assert_eq!(output.lower().as_slice().unwrap(), &[-1.0, -1.0, -1.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[7.0, 7.0, 7.0]);
}

#[test]
fn scatter_add_propagate_linear_returns_unsupported() {
    let layer = ScatterAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[1]))), None, None);
    let err = layer
        .propagate_linear(&LinearBounds::identity(1))
        .expect_err("ScatterAdd CROWN should be unsupported");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// === CROWN backward exactness/soundness ===
//
// Strategy: With `node_lb = identity(output_size)`, the CROWN backward produces
// linear bounds whose concretization over the variable input's box must equal
// the IBP forward output (since identity spec means "output == output"). Because
// these ops are exactly linear with constant indices, CROWN must reproduce IBP
// exactly. We also build the dense Jacobian explicitly and check the CROWN
// A-matrix equals it.

/// Assert the CROWN LOWER bias is SOUND (`<=` the exact real value) and within a
/// tiny absolute margin of it. The const-shift bias fold accumulates in f64 with
/// directed adds and lands with a directed cast (`cast_f64_to_f32_down`) for
/// soundness (#vnncomp-aw-soundness), so an exact integer expectation like `5.0`
/// may legitimately land one ULP below (e.g. `4.9999995`) and `0.0` at the
/// smallest negative subnormal. We therefore check sound-and-close rather than
/// bit-exact. (The directed helpers are exact when the arithmetic is exact, so
/// these particular small-integer cases do in fact land bit-exact — the margin
/// is here so the assertion stays about SOUNDNESS, not about representation.)
fn assert_lower_bias_sound_close(crown: &LinearBounds, expected: &[f32]) {
    let lb = crown.lower_b().as_slice().unwrap();
    assert_eq!(lb.len(), expected.len(), "bias length mismatch");
    for (i, (&got, &exp)) in lb.iter().zip(expected.iter()).enumerate() {
        // Sound: lower bias must not exceed the exact value (allow a sub-ULP
        // tolerance to absorb the exact value's own f32 representation).
        assert!(
            got <= exp + 1e-5,
            "lower_b[{i}] = {got} exceeds expected {exp} (unsound lower bias)"
        );
        assert!(
            (got - exp).abs() <= 1e-5,
            "lower_b[{i}] = {got} too far from expected {exp}"
        );
    }
}

fn concretize_eq_ibp(crown: &LinearBounds, var: &BoundedTensor, ibp: &BoundedTensor) {
    let c = crown.concretize(var);
    let cl = c.lower().as_slice().unwrap();
    let cu = c.upper().as_slice().unwrap();
    let il = ibp.lower().as_slice().unwrap();
    let iu = ibp.upper().as_slice().unwrap();
    assert_eq!(cl.len(), il.len(), "lower length mismatch");
    for i in 0..cl.len() {
        assert!(
            (cl[i] - il[i]).abs() < 1e-5,
            "lower[{i}]: crown {} vs ibp {}",
            cl[i],
            il[i]
        );
        assert!(
            (cu[i] - iu[i]).abs() < 1e-5,
            "upper[{i}]: crown {} vs ibp {}",
            cu[i],
            iu[i]
        );
    }
}

#[test]
fn index_add_crown_data_variable_matches_ibp_and_dense() {
    // y = data_var (3) + index_add(axis=0, src_const) at indices [2, 0].
    // src_const = [10, -5]: adds 10 to position 2, -5 to position 0.
    let layer = IndexAddLayer::new(
        0,
        None,
        Some(arr1(&[2_i64, 0]).into_dyn()),
        Some(arr1(&[10.0_f32, -5.0]).into_dyn()),
    );
    let data = make_bounded(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], &[3]);
    let ibp = layer.propagate_ibp(&data).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(3))
        .expect("crown backward");
    // Data-variable: Jacobian is identity, bias = [-5, 0, 10].
    assert_eq!(crown.lower_a(), &ndarray::Array2::<f32>::eye(3));
    assert_eq!(crown.upper_a(), &ndarray::Array2::<f32>::eye(3));
    assert_lower_bias_sound_close(&crown, &[-5.0, 0.0, 10.0]);
    concretize_eq_ibp(&crown, &data, &ibp);
}

#[test]
fn index_add_crown_src_variable_matches_ibp_and_dense() {
    // y = data_const (2x3) + index_add(axis=1, src_var) with indices [2, 0].
    // Mirrors the fixed IBP test but with src as the variable operand.
    let layer = IndexAddLayer::new(
        1,
        Some(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
        Some(arr1(&[2_i64, 0]).into_dyn()),
        None,
    );
    let src = make_bounded(&[10.0, -1.0, 20.0, -2.0], &[10.0, 1.0, 20.0, 2.0], &[2, 2]);
    let ibp = layer.propagate_ibp(&src).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(6))
        .expect("crown backward");
    // Build the dense Jacobian S (6 outputs x 4 src) explicitly and compare.
    // src layout [2,2] row-major: (r,c) -> r*2 + c.
    // output layout [2,3] row-major: (r,k) -> r*3 + k.
    // index_add axis=1: src col 0 -> out col 2, src col 1 -> out col 0.
    let mut dense = ndarray::Array2::<f32>::zeros((6, 4));
    for r in 0..2 {
        // src col0 (k=0) adds to out col2
        dense[[r * 3 + 2, r * 2]] = 1.0;
        // src col1 (k=1) adds to out col0
        dense[[r * 3, r * 2 + 1]] = 1.0;
    }
    assert_eq!(crown.lower_a(), &dense, "src Jacobian mismatch");
    assert_eq!(crown.upper_a(), &dense);
    // bias == flattened data_const (sound-and-close after outward directed round).
    assert_lower_bias_sound_close(&crown, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    concretize_eq_ibp(&crown, &src, &ibp);
}

#[test]
fn scatter_add_crown_src_variable_matches_ibp_and_dense() {
    // y = data_const (4) + scatter_add(axis=-1, src_var) at indices [1, 1].
    // Both src elements scatter into output position 1 (accumulate).
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::from_shape_vec(IxDyn(&[4]), vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1_i64, 1]).unwrap()),
        None,
    );
    let src = make_bounded(&[2.0, -3.0], &[4.0, -1.0], &[2]);
    let ibp = layer.propagate_ibp(&src).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(4))
        .expect("crown backward");
    // Dense S (4 outputs x 2 src): both src cols add into output row 1.
    let mut dense = ndarray::Array2::<f32>::zeros((4, 2));
    dense[[1, 0]] = 1.0;
    dense[[1, 1]] = 1.0;
    assert_eq!(crown.lower_a(), &dense);
    assert_eq!(crown.upper_a(), &dense);
    assert_lower_bias_sound_close(&crown, &[10.0, 20.0, 30.0, 40.0]);
    concretize_eq_ibp(&crown, &src, &ibp);
}

#[test]
fn scatter_add_crown_data_variable_matches_ibp() {
    // y = data_var (4) + scatter_add(axis=-1, src_const [5, -7]) at indices [0, 3].
    let layer = ScatterAddLayer::new(
        -1,
        None,
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 3]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0_f32, -7.0]).unwrap()),
    );
    let data = make_bounded(&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0], &[4]);
    let ibp = layer.propagate_ibp(&data).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(4))
        .expect("crown backward");
    assert_eq!(crown.lower_a(), &ndarray::Array2::<f32>::eye(4));
    assert_lower_bias_sound_close(&crown, &[5.0, 0.0, 0.0, -7.0]);
    concretize_eq_ibp(&crown, &data, &ibp);
}

#[test]
fn index_add_crown_dynamic_indices_unsupported() {
    // Dynamic indices (indices = None) must fall back: UnsupportedOp.
    let layer = IndexAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[3]))), None, None);
    let err = layer
        .propagate_linear(&LinearBounds::identity(3))
        .expect_err("dynamic indices should be unsupported");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// ===========================================================================
// Directed-rounding regression tests (#vnncomp-aw-soundness)
//
// These are the executed adversarial-review probes. Each one builds a case
// where a plain f32 `+=` ABSORBS the small terms — `1e8 + 1.0` rounds back to
// `1e8`, and it does so identically on BOTH endpoints, so the certified
// interval is TRANSLATED off the true value instead of widened around it.
//
// The assertion is always the same and is the only one that matters: the
// certified interval must ENCLOSE the exact value, computed independently in
// f64 (every quantity here is a small integer, so the f64 reference is exact).
// ===========================================================================

/// The absorbing probe payload: one huge term followed by `n` unit terms that a
/// round-to-nearest f32 accumulation drops entirely. `-1e8` and `-100001000`
/// are both exactly f32-representable (ULP is 8 in that binade and both are
/// multiples of 8), so the probe is not measuring representation error.
const PROBE_HEAD: f32 = -1e8;
const PROBE_TAIL_LEN: usize = 1000;

fn probe_src_values() -> Vec<f32> {
    let mut v = vec![PROBE_HEAD];
    v.extend(std::iter::repeat_n(-1.0_f32, PROBE_TAIL_LEN));
    v
}

/// The exact real sum of the probe payload, in f64 (exact: all integers).
fn probe_exact_sum() -> f64 {
    probe_src_values().iter().map(|&v| f64::from(v)).sum()
}

/// The one assertion that matters: `[lower, upper]` must contain `exact`.
#[track_caller]
fn assert_encloses(lower: f32, upper: f32, exact: f64, what: &str) {
    assert!(
        f64::from(lower) <= exact,
        "{what}: certified LOWER {lower} is ABOVE the exact value {exact} — false proof"
    );
    assert!(
        f64::from(upper) >= exact,
        "{what}: certified UPPER {upper} is BELOW the exact value {exact} — false proof"
    );
}

/// PROBE 1 (static indices). `data = zeros[1001]`, all indices 0, `src`
/// point-valued at the absorbing payload. Every src element lands on cell 0, so
/// the exact output there is `-100001000`. With a plain `+=` both endpoints came
/// back `-1e8`, missing the true value by 1000 on the LOW side.
#[test]
fn scatter_add_static_duplicate_accumulation_encloses_exact_sum() {
    let n = PROBE_TAIL_LEN + 1;
    let src_values = probe_src_values();
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::zeros(IxDyn(&[n]))),
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), vec![0_i64; n]).unwrap()),
        None,
    );
    let src = make_bounded(&src_values, &src_values, &[n]);
    let out = layer.propagate_ibp(&src).expect("ScatterAdd IBP");

    let exact = probe_exact_sum();
    assert_eq!(exact, -100_001_000.0, "probe reference value");
    assert_encloses(
        out.lower()[[0]],
        out.upper()[[0]],
        exact,
        "ScatterAdd static duplicate accumulation",
    );
    // The untouched cells stay exactly at `data`.
    for cell in 1..n {
        assert_eq!((out.lower()[[cell]], out.upper()[[cell]]), (0.0, 0.0));
    }
}

/// PROBE 1, IndexAdd twin. `IndexAdd`'s 1-D index of all zeros routes every src
/// row to destination row 0, giving the same absorbing accumulation.
#[test]
fn index_add_static_duplicate_accumulation_encloses_exact_sum() {
    let n = PROBE_TAIL_LEN + 1;
    let src_values = probe_src_values();
    let layer = IndexAddLayer::new(
        0,
        Some(ArrayD::zeros(IxDyn(&[n]))),
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), vec![0_i64; n]).unwrap()),
        None,
    );
    let src = make_bounded(&src_values, &src_values, &[n]);
    let out = layer.propagate_ibp(&src).expect("IndexAdd IBP");
    assert_encloses(
        out.lower()[[0]],
        out.upper()[[0]],
        probe_exact_sum(),
        "IndexAdd static duplicate accumulation",
    );
}

/// PROBE 2 (dynamic indices). The additive-hull fold was a bare f32
/// `fold(0.0, |acc, v| acc + v.min(0.0))`, which absorbed the 1000 unit terms
/// and certified `-1e8`. The f64 directed fold reproduces the exact hull.
#[test]
fn scatter_add_dynamic_hull_fold_encloses_exact_sum() {
    let n = PROBE_TAIL_LEN + 1;
    let src_values = probe_src_values();
    let layer = ScatterAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[n]))), None, None);
    let index = make_bounded(&vec![0.0; n], &vec![(n - 1) as f32; n], &[n]);
    let src = make_bounded(&src_values, &src_values, &[n]);
    let out = layer
        .propagate_ibp_binary(&index, &src)
        .expect("dynamic ScatterAdd IBP");

    let exact = probe_exact_sum();
    // Worst case for a cell: EVERY element routes to it (the all-zeros index).
    for cell in 0..n {
        assert_encloses(
            out.lower()[[cell]],
            out.upper()[[cell]],
            exact,
            "ScatterAdd dynamic additive hull",
        );
    }
    // Every element is non-positive, so the fold is exactly representable and
    // the hull is TIGHT, not merely sound.
    assert_eq!(out.lower()[[0]], -100_001_000.0);
    assert_eq!(out.upper()[[0]], 0.0);
}

/// PROBE 2, IndexAdd twin.
#[test]
fn index_add_dynamic_hull_fold_encloses_exact_sum() {
    let n = PROBE_TAIL_LEN + 1;
    let src_values = probe_src_values();
    let layer = IndexAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[n]))), None, None);
    let index = make_bounded(&vec![0.0; n], &vec![(n - 1) as f32; n], &[n]);
    let src = make_bounded(&src_values, &src_values, &[n]);
    let out = layer
        .propagate_ibp_binary(&index, &src)
        .expect("dynamic IndexAdd IBP");
    assert_encloses(
        out.lower()[[0]],
        out.upper()[[0]],
        probe_exact_sum(),
        "IndexAdd dynamic additive hull",
    );
    assert_eq!(out.lower()[[0]], -100_001_000.0);
}

/// The positive mirror of the dynamic fold: the UPPER hull must round UP.
#[test]
fn scatter_add_dynamic_hull_fold_encloses_positive_exact_sum() {
    let n = PROBE_TAIL_LEN + 1;
    let src_values: Vec<f32> = probe_src_values().iter().map(|v| -v).collect();
    let layer = ScatterAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[n]))), None, None);
    let index = make_bounded(&vec![0.0; n], &vec![(n - 1) as f32; n], &[n]);
    let src = make_bounded(&src_values, &src_values, &[n]);
    let out = layer
        .propagate_ibp_binary(&index, &src)
        .expect("dynamic ScatterAdd IBP");
    let exact: f64 = src_values.iter().map(|&v| f64::from(v)).sum();
    assert_eq!(exact, 100_001_000.0);
    assert_encloses(
        out.lower()[[0]],
        out.upper()[[0]],
        exact,
        "ScatterAdd dynamic additive hull (positive)",
    );
    assert_eq!(out.upper()[[0]], 100_001_000.0);
    assert_eq!(out.lower()[[0]], 0.0);
}

/// A duplicate-heavy case with THREE hits on one cell and genuinely interval
/// src values: the certified interval must contain the exact sum of every
/// endpoint combination, and in particular the two extreme ones.
#[test]
fn scatter_add_three_hits_on_one_cell_enclose_exact_endpoint_sums() {
    // indices [1, 1, 1, 0]: cell 1 is hit three times, cell 0 once.
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1e7_f32, 1e7, 1e7]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[4]), vec![1_i64, 1, 1, 0]).unwrap()),
        None,
    );
    // Magnitudes chosen so `1e7 + 0.3` is NOT representable (ULP is 1 at 1e7):
    // a round-to-nearest accumulation must lose the fractional parts.
    let lower = [0.3_f32, 0.3, 0.3, -0.7];
    let upper = [0.4_f32, 0.4, 0.4, -0.6];
    let src = make_bounded(&lower, &upper, &[4]);
    let out = layer.propagate_ibp(&src).expect("ScatterAdd IBP");

    let base = 1e7_f64;
    let exact_lo_cell1 = base + lower[0..3].iter().map(|&v| f64::from(v)).sum::<f64>();
    let exact_up_cell1 = base + upper[0..3].iter().map(|&v| f64::from(v)).sum::<f64>();
    assert_encloses(
        out.lower()[[1]],
        out.upper()[[1]],
        exact_lo_cell1,
        "ScatterAdd 3-hit cell (low corner)",
    );
    assert_encloses(
        out.lower()[[1]],
        out.upper()[[1]],
        exact_up_cell1,
        "ScatterAdd 3-hit cell (high corner)",
    );
    // The single-hit cell and the untouched cell must still be enclosed.
    assert_encloses(
        out.lower()[[0]],
        out.upper()[[0]],
        base + f64::from(lower[3]),
        "ScatterAdd single-hit cell",
    );
    assert_eq!((out.lower()[[2]], out.upper()[[2]]), (1e7, 1e7));
}

/// The IndexAdd twin of the 3-hit case, on a rank-2 tensor so the accumulation
/// walks whole slices rather than single elements.
#[test]
fn index_add_three_hits_on_one_row_enclose_exact_endpoint_sums() {
    // data [3, 2], indices [0, 0, 0] -> three src rows all land on row 0.
    let layer = IndexAddLayer::new(
        0,
        Some(ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1e7_f32; 6]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[3]), vec![0_i64, 0, 0]).unwrap()),
        None,
    );
    let lower = [0.3_f32, -0.3, 0.3, -0.3, 0.3, -0.3];
    let upper = [0.4_f32, -0.2, 0.4, -0.2, 0.4, -0.2];
    let src = make_bounded(&lower, &upper, &[3, 2]);
    let out = layer.propagate_ibp(&src).expect("IndexAdd IBP");

    for col in 0..2 {
        let exact_lo = 1e7_f64 + (0..3).map(|r| f64::from(lower[r * 2 + col])).sum::<f64>();
        let exact_up = 1e7_f64 + (0..3).map(|r| f64::from(upper[r * 2 + col])).sum::<f64>();
        assert_encloses(
            out.lower()[[0, col]],
            out.upper()[[0, col]],
            exact_lo,
            "IndexAdd 3-hit row (low corner)",
        );
        assert_encloses(
            out.lower()[[0, col]],
            out.upper()[[0, col]],
            exact_up,
            "IndexAdd 3-hit row (high corner)",
        );
    }
    // Rows 1 and 2 receive nothing and keep `data` exactly.
    for row in 1..3 {
        for col in 0..2 {
            assert_eq!(
                (out.lower()[[row, col]], out.upper()[[row, col]]),
                (1e7, 1e7)
            );
        }
    }
}

/// The CROWN data-variable path folds a CONSTANT scatter into the bias. When
/// that constant scatter has duplicate destinations it accumulates in f32 too,
/// so the folded shift is an INTERVAL; the bias must enclose the exact shift.
#[test]
fn scatter_add_crown_constant_shift_with_duplicates_encloses_exact_shift() {
    let n = PROBE_TAIL_LEN + 1;
    // Positive payload so the absorbed terms push the true shift UP: an
    // undirected fold returns 1e8 and the true value is 100001000.
    let src_values: Vec<f32> = probe_src_values().iter().map(|v| -v).collect();
    let layer = ScatterAddLayer::new(
        -1,
        None,
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), vec![0_i64; n]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), src_values.clone()).unwrap()),
    );
    let crown = layer
        .crown_backward(&LinearBounds::identity(n))
        .expect("crown backward");
    let exact: f64 = src_values.iter().map(|&v| f64::from(v)).sum();
    assert_eq!(exact, 100_001_000.0);
    assert_encloses(
        crown.lower_b()[0],
        crown.upper_b()[0],
        exact,
        "ScatterAdd CROWN constant shift (duplicates)",
    );
    // The Jacobian is still the identity and the other rows are untouched.
    assert_eq!(crown.lower_a(), &ndarray::Array2::<f32>::eye(n));
    assert_eq!(crown.lower_b()[1], 0.0);
    assert_eq!(crown.upper_b()[1], 0.0);
}

/// The IndexAdd twin of the CROWN constant-shift enclosure.
#[test]
fn index_add_crown_constant_shift_with_duplicates_encloses_exact_shift() {
    let n = PROBE_TAIL_LEN + 1;
    let src_values: Vec<f32> = probe_src_values().iter().map(|v| -v).collect();
    let layer = IndexAddLayer::new(
        0,
        None,
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), vec![0_i64; n]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), src_values.clone()).unwrap()),
    );
    let crown = layer
        .crown_backward(&LinearBounds::identity(n))
        .expect("crown backward");
    let exact: f64 = src_values.iter().map(|&v| f64::from(v)).sum();
    assert_encloses(
        crown.lower_b()[0],
        crown.upper_b()[0],
        exact,
        "IndexAdd CROWN constant shift (duplicates)",
    );
}

/// A NEGATIVE CROWN coefficient must pick the OTHER end of the interval shift:
/// with `A = -I` the lower bias needs the shift's UPPER endpoint. This is the
/// case a single-valued `const_shift` could not express.
#[test]
fn scatter_add_crown_negative_coefficient_picks_the_far_shift_endpoint() {
    let n = PROBE_TAIL_LEN + 1;
    let src_values: Vec<f32> = probe_src_values().iter().map(|v| -v).collect();
    let layer = ScatterAddLayer::new(
        -1,
        None,
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), vec![0_i64; n]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[n]), src_values.clone()).unwrap()),
    );
    // node_lb: y = -x, i.e. A = -I, b = 0.
    let neg_eye = -ndarray::Array2::<f32>::eye(n);
    let zeros_b = ndarray::Array1::<f32>::zeros(n);
    let node_lb = LinearBounds::new(neg_eye.clone(), zeros_b.clone(), neg_eye, zeros_b)
        .expect("negated identity spec");
    let crown = layer.crown_backward(&node_lb).expect("crown backward");
    let exact: f64 = -src_values.iter().map(|&v| f64::from(v)).sum::<f64>();
    assert_eq!(exact, -100_001_000.0);
    assert_encloses(
        crown.lower_b()[0],
        crown.upper_b()[0],
        exact,
        "ScatterAdd CROWN negated spec",
    );
}
