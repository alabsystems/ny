// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-errconv-screen: the per-row underflow screen on the ordinary paired
//! error contraction must be BIT-IDENTICAL to the per-element repaired kernel
//! on every input, engage on ordinary magnitudes, and decline exactly where a
//! repair can fire.

use super::{
    compile_conv, error_conv_without_deadline, error_conv_without_deadline_unscreened,
    error_product_with_dust, error_products_branch_free, error_products_widen_only,
    error_row_floors, screened_block_is_normal, screened_row_is_valid, ConvOp,
    ERRCONV_SCREEN_TEST_OFF, ERRCONV_TEST_SCREENED_CALLS,
};
use super::{next_up, nominal_product_dust, underflow_product_up};
use crate::margin_row::engine::{
    bias_row_charges_branch_free, bias_row_charges_branch_free_env, bias_row_is_valid,
    bias_row_is_valid_env, concretize_row_branch_free, concretize_row_is_valid, dust_flag,
    error_axpy_branch_free, nextup_pos, relu_error_update_branch_free, select_f64, up_mul_any,
    up_mul_valid, BackwardEngine, LaneDir, Seed,
};
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::rounding::{gamma_n, UNIT};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use ndarray::Array2;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Mutex;

/// Serializes the tests that flip the global oracle switch.
static SWITCH: Mutex<()> = Mutex::new(());

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Signed values with exact zeros, negative zeros and (when `hazards`) the
/// whole underflow / non-finite zoo, at deterministic positions.
fn signed(n: usize, seed: u64, hazards: bool) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| {
            let v = rng.next() * 4.0 - 2.0;
            match (hazards, i % 29) {
                (_, 0) => 0.0,
                (_, 1) => -0.0,
                (true, 2) => f64::from_bits(1),
                (true, 3) => -1e-310,
                (true, 4) => 1e-300,
                (true, 5) => -1e-160,
                (true, 6) if i % 3 == 0 => f64::NAN,
                (true, 7) if i % 5 == 0 => f64::INFINITY,
                _ => v,
            }
        })
        .collect()
}

/// Nonnegative error budgets (with `hazards`: subnormal / tiny normal /
/// non-finite / negative entries that the per-element arithmetic owns).
fn budgets(n: usize, seed: u64, hazards: bool) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| {
            let v = rng.next() * 1e-12;
            match (hazards, i % 31) {
                (_, 0) => 0.0,
                (_, 1) => -0.0,
                (true, 2) => f64::from_bits(3),
                (true, 3) => 1e-300,
                (true, 4) => 1e-170,
                (true, 5) if i % 4 == 0 => f64::NAN,
                (true, 6) if i % 7 == 0 => f64::INFINITY,
                (true, 7) if i % 9 == 0 => -1e-20,
                _ => v,
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn conv_op(
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pads: (usize, usize, usize, usize),
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
    seed: u64,
    tiny_weights: bool,
) -> ConvOp {
    let (co, ci, kh, kw) = kernel;
    let mut weights = signed(co * ci * kh * kw, seed, false);
    if tiny_weights {
        for (i, w) in weights.iter_mut().enumerate() {
            match i % 11 {
                3 => *w = 1e-200,
                5 => *w = -f64::from_bits(7),
                7 => *w = 1e-150,
                _ => {}
            }
        }
    }
    compile_conv(
        0,
        &weights,
        &vec![0.0; co],
        &vec![0.0; co],
        0.0,
        kernel,
        stride,
        pads,
        ishape,
        oshape,
    )
}

type Geometry = (
    (usize, usize, usize, usize),
    (usize, usize),
    (usize, usize, usize, usize),
    (usize, usize, usize),
    (usize, usize, usize),
);

fn geometries() -> Vec<Geometry> {
    vec![
        // cifar-class body conv, stride 1
        ((8, 6, 3, 3), (1, 1), (1, 1, 1, 1), (6, 8, 8), (8, 8, 8)),
        // input-adjacent ci=3 stride-2 downsample
        ((6, 3, 3, 3), (2, 2), (1, 1, 1, 1), (3, 9, 9), (6, 5, 5)),
        // odd asymmetric kernel / stride / pads
        ((4, 5, 2, 3), (1, 2), (0, 1, 1, 0), (5, 6, 7), (4, 6, 4)),
        // 1x1 kernel
        ((5, 7, 1, 1), (1, 1), (0, 0, 0, 0), (7, 4, 4), (5, 4, 4)),
    ]
}

fn run_pair(c: &ConvOp, err: &Array2<f64>, nom: &Array2<f64>) -> (Vec<u64>, Vec<u64>) {
    let rows_in = c.kernel.1 * c.ishape.1 * c.ishape.2;
    let mut screened = Array2::from_elem((rows_in, err.ncols()), 17.0);
    let mut oracle = Array2::from_elem((rows_in, err.ncols()), -3.0);
    error_conv_without_deadline(c, err, nom, &mut screened).unwrap();
    error_conv_without_deadline_unscreened(c, err, nom, &mut oracle).unwrap();
    (bits(screened.iter().copied()), bits(oracle.iter().copied()))
}

/// Ordinary magnitudes: the screen engages on EVERY block (each row's floor
/// clears every nonzero weight) and the result equals the oracle bit for bit,
/// across geometries and column widths that straddle the accumulator tile.
#[test]
fn screen_engages_and_is_bit_identical_on_ordinary_magnitudes() {
    for (gi, (kernel, stride, pads, ishape, oshape)) in geometries().into_iter().enumerate() {
        let c = conv_op(
            kernel,
            stride,
            pads,
            ishape,
            oshape,
            0xC0 + gi as u64,
            false,
        );
        let rows = kernel.0 * oshape.1 * oshape.2;
        for r in [1, 3, 64, 129] {
            let err =
                Array2::from_shape_vec((rows, r), budgets(rows * r, 7 + r as u64, false)).unwrap();
            let nom =
                Array2::from_shape_vec((rows, r), signed(rows * r, 11 + r as u64, false)).unwrap();
            let floors = error_row_floors(&err, &nom).expect("aligned sources");
            assert_eq!(floors.len(), rows);
            for (row, &[floor, lfloor]) in floors.iter().enumerate() {
                assert!(floor > 0.0, "geometry {gi} r={r} row {row}: floor {floor}");
                assert!(
                    lfloor >= floor,
                    "geometry {gi} r={r} row {row}: lfloor {lfloor}"
                );
                for &w in &c.wt {
                    if w != 0.0 {
                        assert!(
                            screened_block_is_normal(w.abs(), floor),
                            "geometry {gi} r={r} row {row}: weight {w} floor {floor} declined"
                        );
                    }
                }
            }
            let (screened, oracle) = run_pair(&c, &err, &nom);
            assert_eq!(screened, oracle, "geometry {gi} r={r}");
        }
    }
}

/// The hazard zoo (subnormal / tiny normal / non-finite / negative budgets,
/// subnormal and tiny nominal entries, tiny and subnormal weights): the screen
/// declines every row and weight where a repair or an invalid-budget refusal
/// can fire, and the result is still the oracle's, bit for bit.
#[test]
fn screen_is_bit_identical_on_underflow_and_non_finite_hazards() {
    for (gi, (kernel, stride, pads, ishape, oshape)) in geometries().into_iter().enumerate() {
        for tiny_weights in [false, true] {
            let c = conv_op(
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                0xA5 + gi as u64,
                tiny_weights,
            );
            let rows = kernel.0 * oshape.1 * oshape.2;
            for r in [1, 5, 64, 130] {
                let err = Array2::from_shape_vec((rows, r), budgets(rows * r, 3 + r as u64, true))
                    .unwrap();
                let nom = Array2::from_shape_vec((rows, r), signed(rows * r, 5 + r as u64, true))
                    .unwrap();
                let floors = error_row_floors(&err, &nom).expect("aligned sources");
                for (row, &[floor, lfloor]) in floors.iter().enumerate() {
                    let er = &err.as_slice().unwrap()[row * r..(row + 1) * r];
                    let nr = &nom.as_slice().unwrap()[row * r..(row + 1) * r];
                    let invalid = er.iter().any(|e| !e.is_finite() || *e < 0.0)
                        || nr.iter().any(|l| !l.is_finite());
                    if invalid {
                        assert_eq!(floor, 0.0, "geometry {gi} row {row}: invalid row screened");
                        assert_eq!(lfloor, 0.0, "geometry {gi} row {row}: invalid row screened");
                    }
                    for v in nr {
                        if v.is_finite() && *v != 0.0 {
                            assert!(lfloor <= v.abs() || invalid);
                        }
                    }
                    // The floor is a true lower bound of every nonzero magnitude.
                    for v in er.iter().chain(nr) {
                        if v.is_finite() && *v != 0.0 {
                            assert!(floor <= v.abs() || invalid);
                        }
                    }
                }
                let (screened, oracle) = run_pair(&c, &err, &nom);
                assert_eq!(
                    screened, oracle,
                    "geometry {gi} tiny_weights={tiny_weights} r={r}"
                );
            }
        }
    }
}

/// A block whose products underflow is declined, and the oracle's repair (the
/// lost product widened into the subnormal range plus nominal dust) survives.
#[test]
fn screen_declines_underflowing_blocks_and_keeps_the_repair() {
    let one_by_one = |weights: &[f64], co: usize| {
        compile_conv(
            0,
            weights,
            &vec![0.0; co],
            &vec![0.0; co],
            0.0,
            (co, 1, 1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 1, 1),
            (co, 1, 1),
        )
    };
    // Row 0 (weight 1e-160): e = 1e-160 makes w*e = 1e-320 (subnormal) and
    // the nominal 1e-160 makes w*l subnormal too. Row 1 (weight 0.5): normal.
    let c = one_by_one(&[1e-160, 0.5], 2);
    let err = Array2::from_shape_vec((2, 2), vec![1e-160, 0.0, 0.25, 0.125]).unwrap();
    let nom = Array2::from_shape_vec((2, 2), vec![-1e-160, 1e-160, 1.0, -2.0]).unwrap();
    let floors = error_row_floors(&err, &nom).unwrap();
    assert_eq!(
        bits(floors.iter().flatten().copied()),
        bits([1e-160, 1e-160, 0.125, 1.0])
    );
    assert!(!screened_block_is_normal(1e-160, floors[0][0]));
    assert!(screened_block_is_normal(0.5, floors[1][0]));
    let (screened, oracle) = run_pair(&c, &err, &nom);
    assert_eq!(screened, oracle);

    // The underflowing row alone: the repair is visible in the result.
    let c = one_by_one(&[1e-160], 1);
    let err = Array2::from_shape_vec((1, 2), vec![1e-160, 0.0]).unwrap();
    let nom = Array2::from_shape_vec((1, 2), vec![-1e-160, 1e-160]).unwrap();
    let mut out = Array2::zeros((1, 2));
    error_conv_without_deadline(&c, &err, &nom, &mut out).unwrap();
    let (screened, oracle) = run_pair(&c, &err, &nom);
    assert_eq!(screened, oracle);
    let plain = 1e-160_f64 * 1e-160_f64;
    assert!(plain < f64::MIN_POSITIVE);
    // Widened error product plus one unit of nominal dust, both retained.
    assert!(
        out[[0, 0]] >= plain + 2.0 * f64::from_bits(1),
        "{:e}",
        out[[0, 0]]
    );
    // Zero error, underflowing nominal product: the dust alone survives.
    assert!(out[[0, 1]] > 0.0, "{:e}", out[[0, 1]]);
}

/// The branch-free block kernel equals the scalar helper's accumulation bit
/// for bit over the whole underflow / overflow / signed-zero lattice its
/// preconditions admit (finite positive weight, finite nonnegative-or-`-0.0`
/// error, finite nominal), from several accumulator starts.
#[test]
fn branch_free_terms_match_the_scalar_helper_bit_for_bit() {
    let eta = f64::from_bits(1);
    let m = f64::MIN_POSITIVE;
    let weights = [
        eta,
        7.0 * eta,
        m / 3.0,
        m,
        1e-300,
        1e-160,
        0.375,
        0.5,
        1.0,
        3.0,
        1e160,
        1e300,
        f64::MAX,
    ];
    let errors = [
        0.0,
        -0.0,
        eta,
        3.0 * eta,
        m / 2.0,
        m - eta,
        m,
        2.0 * m,
        1e-300,
        1e-160,
        1e-12,
        1.0,
        1e300,
        f64::MAX,
    ];
    let mut nominals = vec![0.0, -0.0];
    for v in [eta, m / 3.0, m, 1e-300, 1e-160, 1.0, 1e300, f64::MAX] {
        nominals.push(v);
        nominals.push(-v);
    }
    for &w in &weights {
        assert!(screened_row_is_valid(w, 1.0));
        for start in [0.0, 1.0, eta, 1e-300] {
            for &l in &nominals {
                let err: Vec<f64> = errors.to_vec();
                let nom = vec![l; err.len()];
                let mut want = vec![start; err.len()];
                for ((a, &e), &n) in want.iter_mut().zip(&err).zip(&nom) {
                    *a += error_product_with_dust(w, e, n);
                }
                let mut got = vec![start; err.len()];
                error_products_branch_free(&mut got, w, &err, &nom);
                assert_eq!(
                    bits(got),
                    bits(want.iter().copied()),
                    "w={w:e} start={start:e} l={l:e}"
                );
                // The widen-only kernel is taken only when the nominal row
                // cannot produce dust for this weight; there it is exact too.
                if l == 0.0 || screened_block_is_normal(w, l.abs()) {
                    let mut widened = vec![start; err.len()];
                    error_products_widen_only(&mut widened, w, &err);
                    assert_eq!(
                        bits(widened),
                        bits(want.iter().copied()),
                        "widen-only w={w:e} start={start:e} l={l:e}"
                    );
                }
            }
        }
    }
    // The validity predicate refuses what the branch-free kernel cannot take.
    assert!(!screened_row_is_valid(1.0, 0.0));
    assert!(!screened_row_is_valid(0.0, 1.0));
    assert!(!screened_row_is_valid(-1.0, 1.0));
    assert!(!screened_row_is_valid(f64::INFINITY, 1.0));
    assert!(!screened_row_is_valid(f64::NAN, 1.0));
}

/// The predicate's edges: one binade of margin, and no non-finite, negative,
/// zero or NaN weight ever passes; an all-zero row (floor +inf) passes for a
/// finite positive weight because no term of it can fire.
#[test]
fn screen_predicate_edges() {
    let m = f64::MIN_POSITIVE;
    assert!(screened_block_is_normal(1.0, 2.0 * m));
    assert!(!screened_block_is_normal(1.0, 2.0 * m - f64::from_bits(1)));
    assert!(!screened_block_is_normal(1.0, m));
    assert!(!screened_block_is_normal(0.5, 3.0 * m));
    assert!(screened_block_is_normal(0.5, 4.0 * m));
    assert!(!screened_block_is_normal(1.0, 0.0));
    assert!(!screened_block_is_normal(f64::INFINITY, 1.0));
    assert!(!screened_block_is_normal(f64::NAN, 1.0));
    assert!(!screened_block_is_normal(-1.0, 1.0));
    assert!(!screened_block_is_normal(0.0, f64::INFINITY));
    assert!(screened_block_is_normal(1e-300, f64::INFINITY));
    // All-zero row: identical to the oracle (zeros, signed zeros preserved).
    let c = compile_conv(
        0,
        &[0.75],
        &[0.0],
        &[0.0],
        0.0,
        (1, 1, 1, 1),
        (1, 1),
        (0, 0, 0, 0),
        (1, 1, 1),
        (1, 1, 1),
    );
    let err = Array2::from_shape_vec((1, 3), vec![0.0, -0.0, 0.0]).unwrap();
    let nom = Array2::from_shape_vec((1, 3), vec![-0.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        error_row_floors(&err, &nom).unwrap(),
        vec![[f64::INFINITY, f64::INFINITY]]
    );
    let (screened, oracle) = run_pair(&c, &err, &nom);
    assert_eq!(screened, oracle);
    // Misaligned sources: no floors; the kernel refuses exactly as before.
    let short = Array2::zeros((1, 2));
    assert!(error_row_floors(&err, &short).is_none());
    let mut dst = Array2::zeros((1, 3));
    assert!(error_conv_without_deadline(&c, &err, &short, &mut dst).is_err());
}

/// Whole-engine bit identity: an Outward root build plus lower/upper passes
/// (identity y-seed, a dense seed WITH an incoming error matrix, and the
/// backward-interm prefix pass from the first trunk ReLU) produce the same
/// coefficients, errors, biases, bias errors and concretized bounds with the
/// screen on and with every contraction routed through the oracle. The tiny
/// variant mixes underflowing weights in, so both paths run inside one pass.
#[test]
fn engine_passes_bit_identical_with_and_without_errconv_screen() {
    let _guard = SWITCH.lock().unwrap_or_else(|p| p.into_inner());
    for tiny in [false, true] {
        let spec = engine_spec(0x5EED, tiny);
        let net = TwinNet::compile(&spec).expect("compile");
        let lo = vec![-0.3; spec.n_in];
        let hi = vec![0.45; spec.n_in];
        let on = engine_snapshot(&net, &lo, &hi);
        ERRCONV_SCREEN_TEST_OFF.store(true, SeqCst);
        let off = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine_snapshot(&net, &lo, &hi)
        }));
        ERRCONV_SCREEN_TEST_OFF.store(false, SeqCst);
        let off = off.expect("oracle snapshot");
        assert!(
            on.screened_calls > 0,
            "tiny={tiny}: the engine never reached the screen"
        );
        assert_eq!(
            off.screened_calls, 0,
            "tiny={tiny}: the oracle arm was screened"
        );
        assert_eq!(on.words.len(), off.words.len());
        for (i, (a, b)) in on.words.iter().zip(&off.words).enumerate() {
            assert_eq!(a, b, "tiny={tiny}: snapshot section {i} differs");
        }
    }
}

struct Snapshot {
    words: Vec<Vec<u64>>,
    screened_calls: u64,
}

fn engine_snapshot(net: &TwinNet, lo: &[f64], hi: &[f64]) -> Snapshot {
    let before = ERRCONV_TEST_SCREENED_CALLS.with(std::cell::Cell::get);
    let root = RootGates::build(net, lo, hi, RoundMode::Outward, None).expect("root");
    let mut words = Vec::new();
    for lg in &root.layers {
        words.push(bits(lg.l.iter().copied()));
        words.push(bits(lg.u.iter().copied()));
    }
    let eng = BackwardEngine::new(net, &root);
    let mut push = |out: &crate::margin_row::engine::PassOut, lower: bool| {
        words.push(bits(out.a.iter().copied()));
        words.push(bits(
            out.e.as_ref().expect("outward error lane").iter().copied(),
        ));
        words.push(bits(out.b.iter().copied()));
        words.push(bits(out.eb.iter().copied()));
        let conc = if lower {
            eng.concretize_lower(out)
        } else {
            eng.concretize_upper(out)
        };
        words.push(bits(conc));
    };
    let (al, au) = eng.y_rows(None).expect("y rows");
    push(&al, true);
    push(&au, false);
    let n_y = net.n_y;
    let cols = 9;
    let s = Array2::from_shape_vec((n_y, cols), signed(n_y * cols, 0xD1, false)).unwrap();
    let e = Array2::from_shape_vec((n_y, cols), budgets(n_y * cols, 0xD2, false)).unwrap();
    let seed = Seed { s, e: Some(e) };
    for dir in [LaneDir::Lower, LaneDir::Upper] {
        let out = eng.run(&seed, None, dir, None, false).expect("dense pass");
        push(&out, matches!(dir, LaneDir::Lower));
    }
    // backward-interm's own call: identity columns at the second trunk ReLU's
    // pre-activation, walked back through the first conv/ReLU to the input.
    let relu_layer = &root.layers[1];
    let n_pre = relu_layer.n;
    let w = n_pre.min(16);
    let mut s = Array2::<f64>::zeros((n_pre, w));
    for c in 0..w {
        s[[c, c]] = 1.0;
    }
    let seed = Seed { s, e: None };
    for dir in [LaneDir::Lower, LaneDir::Upper] {
        let out = eng
            .run_prefix_dom(&seed, relu_layer.op, dir, None)
            .expect("prefix pass");
        push(&out, matches!(dir, LaneDir::Lower));
    }
    let screened_calls = ERRCONV_TEST_SCREENED_CALLS.with(std::cell::Cell::get) - before;
    Snapshot {
        words,
        screened_calls,
    }
}

/// Conv(3->6, s1) -> ReLU -> Conv(6->6, s2) -> ReLU -> Flatten -> Gemm(5) ->
/// ReLU -> Gemm(3). With `tiny`, every 9th conv weight is 1e-170 so the
/// error products of those weights underflow and their blocks are declined.
fn engine_spec(seed: u64, tiny: bool) -> TwinSpec {
    let mut rng = Lcg(seed);
    let mut w = |n: usize, scale: f64| -> Vec<f64> {
        (0..n)
            .map(|i| {
                if tiny && i % 9 == 4 {
                    1e-170
                } else if i % 13 == 0 {
                    0.0
                } else {
                    (rng.next() * 2.0 - 1.0) * scale
                }
            })
            .collect()
    };
    let tiny_bias = |mut v: Vec<f64>| {
        if tiny {
            v[1] = 1e-300;
            v[4] = -1e-305;
        }
        v
    };
    let conv1 = TwinOpSpec::Conv {
        input: 0,
        weight: w(6 * 3 * 3 * 3, 0.6),
        bias: tiny_bias(w(6, 0.2)),
        bias_err: if tiny { vec![1e-310; 6] } else { vec![0.0; 6] },
        weight_rel_err: 1e-15,
        kernel: (6, 3, 3, 3),
        stride: (1, 1),
        pads: (1, 1, 1, 1),
        ishape: (3, 6, 6),
        oshape: (6, 6, 6),
    };
    let conv2 = TwinOpSpec::Conv {
        input: 2,
        weight: w(6 * 6 * 3 * 3, 0.5),
        bias: tiny_bias(w(6, 0.2)),
        bias_err: vec![1e-18; 6],
        weight_rel_err: 1e-15,
        kernel: (6, 6, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (6, 6, 6),
        oshape: (6, 3, 3),
    };
    let flat = 6 * 3 * 3;
    TwinSpec {
        n_in: 3 * 6 * 6,
        ops: vec![
            conv1,                            // t1
            TwinOpSpec::Relu { input: 1 },    // t2 (trunk relu 0)
            conv2,                            // t3
            TwinOpSpec::Relu { input: 3 },    // t4 (trunk relu 1)
            TwinOpSpec::Flatten { input: 4 }, // t5
            TwinOpSpec::Gemm {
                input: 5,
                weight: w(5 * flat, 0.4),
                bias: w(5, 0.2),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (5, flat),
            }, // t6 (y)
            TwinOpSpec::Relu { input: 6 },    // t7 (head relu)
            TwinOpSpec::Gemm {
                input: 7,
                weight: w(3 * 5, 0.5),
                bias: w(3, 0.1),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (3, 5),
            }, // t8
        ],
    }
}

/// The branch-free helpers equal the scalar ones over the special-value
/// lattice their callers admit: `up_mul_valid` = `underflow_product_up` for
/// finite non-negative operands, `dust_flag` = `nominal_product_dust != 0` for
/// EVERY input, `nextup_pos` = `next_up` for positive and `+inf` arguments.
#[test]
fn branch_free_helpers_match_the_scalar_ones() {
    let eta = f64::from_bits(1);
    let m = f64::MIN_POSITIVE;
    let nonneg = [
        0.0,
        -0.0,
        eta,
        3.0 * eta,
        m / 3.0,
        m - eta,
        m,
        2.0 * m,
        1e-300,
        1e-160,
        gamma_n(10),
        0.5,
        1.0,
        3.0,
        1e160,
        1e300,
        f64::MAX,
    ];
    for &a in &nonneg {
        for &b in &nonneg {
            assert_eq!(
                up_mul_valid(a, b).to_bits(),
                underflow_product_up(a, b).to_bits(),
                "{a:e} * {b:e}"
            );
        }
    }
    let mut any: Vec<f64> = nonneg.to_vec();
    any.extend(nonneg.iter().map(|v| -v));
    any.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    for &a in &any {
        for &b in &any {
            assert_eq!(
                dust_flag(a, b),
                nominal_product_dust(a, b) != 0.0,
                "{a:e} {b:e}"
            );
        }
        if a > 0.0 {
            assert_eq!(nextup_pos(a).to_bits(), next_up(a).to_bits(), "{a:e}");
        }
        assert_eq!(select_f64(true, a, 1.5).to_bits(), a.to_bits());
        assert_eq!(select_f64(false, 1.5, a).to_bits(), a.to_bits());
    }
}

/// The conv bias arm: on every VALID row the branch-free charges equal the
/// scalar repaired loop bit for bit (dust and carry included); invalid rows
/// are refused. Random rows mix ordinary magnitudes with the underflow /
/// signed-zero / huge / non-finite zoo; both outcomes occur.
#[test]
fn bias_row_branch_free_charges_equal_the_repaired_loop() {
    let gam_b = gamma_n(4096 + 2);
    let biases = [0.0, -0.0, 0.2, -3.5, 1e-300, -1e-160, 1e140, f64::NAN];
    let berrs = [0.0, 1e-18, 3.7e-14, 1e-300, 1e140, -1e-20, f64::INFINITY];
    let (mut admitted, mut refused) = (0usize, 0usize);
    for seed in 0..24u64 {
        let r = 1 + (seed as usize * 7) % 41;
        let lr = signed(r, 100 + seed, seed % 3 == 0);
        let er: Vec<f64> = budgets(r, 200 + seed, seed % 4 == 1)
            .into_iter()
            .map(|e| if seed % 5 == 2 { e * 1e-300 } else { e })
            .collect();
        let b = signed(r, 300 + seed, seed % 6 == 3);
        for &bias in &biases {
            for &berr in &berrs {
                let ab = bias.abs();
                let mut repaired = signed(r, 900 + seed, false);
                let mut fast = repaired.clone();
                for (ri, ebv) in repaired.iter_mut().enumerate() {
                    let mag = lr[ri].abs();
                    let charge =
                        underflow_product_up(er[ri] + underflow_product_up(gam_b, mag), ab)
                            + underflow_product_up(mag + er[ri], berr);
                    let dust = nominal_product_dust(bias, lr[ri]);
                    *ebv += if dust != 0.0 {
                        next_up(charge + dust)
                    } else {
                        charge
                    };
                    if bias != 0.0 {
                        *ebv += underflow_product_up(2.0 * UNIT, b[ri].abs());
                    }
                }
                if bias_row_is_valid(&lr, &er, &b, bias, berr) {
                    admitted += 1;
                    bias_row_charges_branch_free(&mut fast, &lr, &er, &b, bias, berr, gam_b);
                    assert_eq!(
                        bits(fast),
                        bits(repaired),
                        "seed {seed} bias {bias:e} berr {berr:e}"
                    );
                } else {
                    refused += 1;
                }
                // The envelope lane: one envelope value per row in place of
                // the per-column budgets.
                for &ev in &[
                    0.0,
                    -0.0,
                    f64::from_bits(2),
                    1e-300,
                    3e-9,
                    1e140,
                    -1.0,
                    f64::NAN,
                ] {
                    let mut env_repaired = signed(r, 950 + seed, false);
                    let mut env_fast = env_repaired.clone();
                    for (ri, ebv) in env_repaired.iter_mut().enumerate() {
                        let mag = lr[ri].abs();
                        let charge =
                            underflow_product_up(ev + underflow_product_up(gam_b, mag), ab)
                                + underflow_product_up(mag + ev, berr);
                        let dust = nominal_product_dust(bias, lr[ri]);
                        *ebv += if dust != 0.0 {
                            next_up(charge + dust)
                        } else {
                            charge
                        };
                        if bias != 0.0 {
                            *ebv += underflow_product_up(2.0 * UNIT, b[ri].abs());
                        }
                    }
                    if bias_row_is_valid_env(&lr, ev, &b, bias, berr) {
                        admitted += 1;
                        bias_row_charges_branch_free_env(
                            &mut env_fast,
                            &lr,
                            ev,
                            &b,
                            bias,
                            berr,
                            gam_b,
                        );
                        assert_eq!(
                            bits(env_fast),
                            bits(env_repaired),
                            "env seed {seed} bias {bias:e} berr {berr:e} ev {ev:e}"
                        );
                    } else {
                        refused += 1;
                    }
                }
            }
        }
    }
    assert!(
        admitted > 0 && refused > 0,
        "admitted {admitted} refused {refused}"
    );
}

/// Concretize: on every VALID input row the branch-free accumulation equals
/// the scalar repaired loop bit for bit in all five accumulators (dust
/// included); invalid rows are refused; both outcomes occur.
#[test]
fn concretize_row_branch_free_equals_the_repaired_loop() {
    let mids = [0.0, -0.0, 0.5, -1.25, 1e-300, 1e300, f64::NAN];
    let rads = [0.0, 0.01, 1e-310, 2.0];
    let xabs = [0.0, -0.0, 0.75, 1e-300, 3.0, -1.0, f64::INFINITY];
    let (mut admitted, mut refused) = (0usize, 0usize);
    for seed in 0..24u64 {
        let r = 1 + (seed as usize * 5) % 37;
        let arow: Vec<f64> = signed(r, 400 + seed, seed % 3 == 1)
            .into_iter()
            .map(|a| if seed % 4 == 3 { a * 1e-305 } else { a })
            .collect();
        let erow: Vec<f64> = budgets(r, 500 + seed, seed % 5 == 4)
            .into_iter()
            .map(|e| if seed % 3 == 2 { e * 1e-308 } else { e })
            .collect();
        for &m in &mids {
            for &rd in &rads {
                for &xa in &xabs {
                    for with_e in [false, true] {
                        let e = with_e.then_some(&erow[..]);
                        let start = signed(r, 700 + seed, false);
                        let (mut v, mut rr, mut tabs, mut pen) =
                            (start.clone(), start.clone(), start.clone(), start);
                        let mut dust: Vec<f64> = (0..r).map(|i| (i % 3) as f64 * eta()).collect();
                        let (mut fv, mut frr, mut ftabs, mut fpen, mut fdust) = (
                            v.clone(),
                            rr.clone(),
                            tabs.clone(),
                            pen.clone(),
                            dust.clone(),
                        );
                        for (ri, &av) in arow.iter().enumerate() {
                            v[ri] += av * m;
                            let aa = av.abs();
                            rr[ri] += aa * rd;
                            tabs[ri] += underflow_product_up(aa, xa);
                            let dm = nominal_product_dust(av, m);
                            let dr = nominal_product_dust(aa, rd);
                            if dm != 0.0 {
                                dust[ri] = next_up(dust[ri] + dm);
                            }
                            if dr != 0.0 {
                                dust[ri] = next_up(dust[ri] + dr);
                            }
                        }
                        if let Some(es) = e {
                            for (ri, &ev) in es.iter().enumerate() {
                                pen[ri] += underflow_product_up(ev, xa);
                            }
                        }
                        if concretize_row_is_valid(&arow, e, xa) {
                            admitted += 1;
                            concretize_row_branch_free(
                                &mut fv, &mut frr, &mut ftabs, &mut fdust, &mut fpen, &arow, e, m,
                                rd, xa,
                            );
                            let tag = format!("seed {seed} m {m:e} rd {rd:e} xa {xa:e} e {with_e}");
                            assert_eq!(bits(fv), bits(v), "v {tag}");
                            assert_eq!(bits(frr), bits(rr), "rr {tag}");
                            assert_eq!(bits(ftabs), bits(tabs), "tabs {tag}");
                            assert_eq!(bits(fdust), bits(dust), "dust {tag}");
                            assert_eq!(bits(fpen), bits(pen), "pen {tag}");
                        } else {
                            refused += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        admitted > 0 && refused > 0,
        "admitted {admitted} refused {refused}"
    );
}

fn eta() -> f64 {
    f64::from_bits(1)
}

/// `up_mul_any` equals `underflow_product_up` for EVERY input pair (signed
/// zeros, subnormals, the underflow edge, huge values, infinities, NaN,
/// negatives); `error_axpy_branch_free` equals the scalar repaired
/// `error_axpy_checked` loop; the branch-free ReLU error update equals the
/// scalar repaired update for every operand combination drawn from the same
/// lattice (both lanes' slope orientations).
#[test]
fn general_branch_free_forms_match_the_scalar_repairs() {
    let eta = f64::from_bits(1);
    let m = f64::MIN_POSITIVE;
    let base = [
        0.0,
        eta,
        3.0 * eta,
        m / 3.0,
        m - eta,
        m,
        2.0 * m,
        1e-300,
        1e-160,
        gamma_n(10),
        0.5,
        1.0,
        3.0,
        1e160,
        1e300,
        f64::MAX,
        f64::INFINITY,
    ];
    let mut any: Vec<f64> = base.to_vec();
    any.extend(base.iter().map(|v| -v));
    any.push(f64::NAN);
    for &a in &any {
        for &b in &any {
            assert_eq!(
                up_mul_any(a, b).to_bits(),
                underflow_product_up(a, b).to_bits(),
                "{a:e} * {b:e}"
            );
        }
    }
    // error_axpy: one weight (>= 0 in the engine, any here) against rows
    // drawn from the lattice.
    let n = any.len();
    for &w in &any {
        let error: Vec<f64> = (0..n).map(|i| any[(i * 7 + 3) % n]).collect();
        let nominal: Vec<f64> = (0..n).map(|i| any[(i * 11 + 5) % n]).collect();
        let mut want: Vec<f64> = (0..n).map(|i| any[(i * 13 + 1) % n]).collect();
        let mut got = want.clone();
        for ((a, &e), &v) in want.iter_mut().zip(&error).zip(&nominal) {
            let term = underflow_product_up(w, e);
            let dust = nominal_product_dust(w, v);
            *a += if dust != 0.0 {
                next_up(term + dust)
            } else {
                term
            };
        }
        error_axpy_branch_free(&mut got, w, &error, &nominal);
        assert_eq!(bits(got), bits(want), "w {w:e}");
    }
    // ReLU error update over (e, v, slopes, ms) drawn from the lattice.
    let slopes = [0.0, -0.0, eta, 1e-300, 0.25, 1.0, f64::NAN];
    let ms = [0.0, -0.0, 1e-300, 0.5, 1.0, 2.0, f64::INFINITY];
    for (ie, &e) in any.iter().enumerate() {
        for (iv, &v) in any.iter().enumerate().step_by(3) {
            for &aj in &slopes {
                for &sj in &slopes {
                    let mj = ms[(ie + iv) % ms.len()];
                    for lower in [true, false] {
                        let vp = v.max(0.0);
                        let vn = v - vp;
                        let lnew = if lower {
                            vp * aj + vn * sj
                        } else {
                            vp * sj + vn * aj
                        };
                        let (sp, sn) = if lower { (aj, sj) } else { (sj, aj) };
                        let charge =
                            underflow_product_up(e + underflow_product_up(4.0 * UNIT, v.abs()), mj)
                                + underflow_product_up(4.0 * UNIT, lnew.abs());
                        let d1 = nominal_product_dust(vp, sp);
                        let charge = if d1 != 0.0 {
                            next_up(charge + d1)
                        } else {
                            charge
                        };
                        let d2 = nominal_product_dust(vn, sn);
                        let charge = if d2 != 0.0 {
                            next_up(charge + d2)
                        } else {
                            charge
                        };
                        let want = next_up(charge * (1.0 + 16.0 * UNIT));
                        let got = relu_error_update_branch_free(e, v, lnew, vp, vn, sp, sn, mj);
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "e {e:e} v {v:e} aj {aj:e} sj {sj:e} mj {mj:e} lower {lower}"
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// review(mrcost): adversarial randomized differential checks.
//
// The tests above check the branch-free forms over hand-picked lattices, with
// oracles written alongside the forms. These draw MILLIONS of operands from
// the whole binary64 space (random bit patterns, NaN payloads, both signed
// zeros, subnormals, values within a few ulps of every threshold the proofs
// rely on: MIN_POSITIVE, 2*MIN_POSITIVE, sqrt(MIN_POSITIVE), the 2^400 bias
// cap, 2^511, f64::MAX) and compare against the ORIGINAL scalar compositions
// (`underflow_product_up` / `nominal_product_dust` / `next_up` / `slack16` /
// the engine's own `error_mul`), bit for bit. Each test also asserts that the
// interesting cases were actually drawn, so a generator regression cannot turn
// a check into a no-op.
// ---------------------------------------------------------------------------

use crate::margin_row::engine::{error_mul, ERRCONV_TEST_ROWS};
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::rounding::slack16;

/// splitmix64 over a hazard-biased binary64 distribution.
struct Mix(u64);

impl Mix {
    fn u(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.u() % n
    }

    /// `x` moved by up to 4 ulps either way in its bit pattern (sign kept;
    /// clamps at +0).
    fn near(&mut self, x: f64) -> f64 {
        let k = self.below(9) as i64 - 4;
        let sign = x.to_bits() & (1 << 63);
        let mag = (x.to_bits() & !(1 << 63)) as i64;
        f64::from_bits(sign | (mag + k).max(0) as u64)
    }

    /// A magnitude-like value: mostly nonnegative, but class 0 is ANY bit
    /// pattern (negative, NaN payloads, infinities included).
    fn mag(&mut self) -> f64 {
        let m = f64::MIN_POSITIVE;
        match self.below(17) {
            0 => f64::from_bits(self.u()),
            1 => 0.0,
            2 => -0.0,
            3 => f64::from_bits(1 + self.below(64)),
            4 => f64::from_bits(self.u() & 0x000F_FFFF_FFFF_FFFF),
            5 => self.near(m),
            6 => self.near(2.0 * m),
            7 => self.near(m.sqrt()),
            8 => self.near(2f64.powi(511)),
            9 => self.near(2f64.powi(400)),
            10 => self.near(f64::MAX),
            11 => f64::INFINITY,
            12 => f64::from_bits(0x7FF8_0000_0000_0000 | (self.u() & 0xF_FFFF)),
            13 => self.near(4.0 * UNIT),
            // Ordinary: a uniformly random exponent over all finite binades.
            _ => f64::from_bits((self.below(2046) << 52) | (self.u() & 0x000F_FFFF_FFFF_FFFF)),
        }
    }

    fn any(&mut self) -> f64 {
        let v = self.mag();
        if self.u() & 1 == 1 {
            -v
        } else {
            v
        }
    }

    /// `(a, b)` whose product lies within a few ulps of `target` (or
    /// underflows/overflows trying), `a` drawn over 1200 binades.
    fn pair_near(&mut self, target: f64) -> (f64, f64) {
        let a =
            f64::from_bits(((self.below(1200) + 423) << 52) | (self.u() & 0x000F_FFFF_FFFF_FFFF));
        let b = self.near(target / a);
        (a, b)
    }

    /// An operand pair for the product helpers: random, or aimed at the
    /// underflow edge / the screen's one-binade margin, random signs.
    fn product_pair(&mut self) -> (f64, f64) {
        let (a, b) = match self.below(4) {
            0 => self.pair_near(f64::MIN_POSITIVE),
            1 => self.pair_near(2.0 * f64::MIN_POSITIVE),
            _ => (self.mag(), self.mag()),
        };
        let sa = if self.below(5) == 0 { -a } else { a };
        let sb = if self.below(5) == 0 { -b } else { b };
        (sa, sb)
    }
}

const REVIEW_N: usize = 2_000_000;

/// The scalar dust step, exactly as `add_nominal_dust(charge, a, b, true)`.
fn ref_dust(charge: f64, a: f64, b: f64) -> f64 {
    let d = nominal_product_dust(a, b);
    if d != 0.0 {
        next_up(charge + d)
    } else {
        charge
    }
}

// `!(x < 0.0)` mirrors `underflow_product_up`'s refusal predicate exactly.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
#[test]
fn review_product_helpers_match_scalar_on_random_bits() {
    let mut g = Mix(0x5EED_0001);
    let (mut widened, mut refused, mut dusty, mut valid) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..REVIEW_N {
        let (a, b) = g.product_pair();
        let want = underflow_product_up(a, b);
        assert_eq!(
            up_mul_any(a, b).to_bits(),
            want.to_bits(),
            "up_mul_any({a:e}, {b:e})"
        );
        assert_eq!(
            error_mul(a, b, true).to_bits(),
            want.to_bits(),
            "error_mul({a:e}, {b:e})"
        );
        if want == f64::INFINITY && !(a * b).is_infinite() {
            refused += 1;
        }
        if a.is_finite() && b.is_finite() && !(a < 0.0) && !(b < 0.0) {
            valid += 1;
            assert_eq!(
                up_mul_valid(a, b).to_bits(),
                want.to_bits(),
                "up_mul_valid({a:e}, {b:e})"
            );
            if want.to_bits() != (a * b).to_bits() {
                widened += 1;
            }
        }
        let d = nominal_product_dust(a, b) != 0.0;
        dusty += usize::from(d);
        assert_eq!(dust_flag(a, b), d, "dust_flag({a:e}, {b:e})");
        let x = a.abs();
        if !x.is_nan() {
            assert_eq!(
                nextup_pos(x).to_bits(),
                next_up(x).to_bits(),
                "nextup_pos({x:e})"
            );
        }
    }
    // The draws reached every regime the proofs distinguish.
    assert!(widened > 10_000, "widened {widened}");
    assert!(refused > 10_000, "refused {refused}");
    assert!(dusty > 10_000, "dusty {dusty}");
    assert!(valid > REVIEW_N / 4, "valid {valid}");
    // -0.0 is the one input where `nextup_pos` and `next_up` differ; the
    // engine never feeds it one (every argument is `x + eta` with `x >= -0`,
    // or `charge * (1 + 16u)` with `charge >= +0`). Pin that this is known.
    assert_ne!(nextup_pos(-0.0).to_bits(), next_up(-0.0).to_bits());
}

#[test]
fn review_relu_update_and_error_axpy_match_scalar_on_random_bits() {
    let mut g = Mix(0x5EED_0002);
    let mut dust_hits = 0usize;
    for i in 0..REVIEW_N {
        let e = if i % 3 == 0 { g.any() } else { g.mag().abs() };
        let v = match g.below(3) {
            0 => g.pair_near(f64::MIN_POSITIVE).0,
            _ => g.any(),
        };
        let (aj, sj) = if g.below(3) == 0 {
            // Slopes aimed so v * slope sits on the underflow edge.
            let t = g.near(f64::MIN_POSITIVE / v.abs().max(f64::MIN_POSITIVE));
            (t, g.any())
        } else {
            (g.any(), g.any())
        };
        let mj = if g.below(2) == 0 {
            g.mag().abs()
        } else {
            g.any()
        };
        let lower = g.u() & 1 == 1;
        let vp = v.max(0.0);
        let vn = v - vp;
        let lnew = if g.below(4) == 0 {
            g.any()
        } else if lower {
            vp * aj + vn * sj
        } else {
            vp * sj + vn * aj
        };
        let (sp, sn) = if lower { (aj, sj) } else { (sj, aj) };
        // The engine's scalar repaired update, verbatim.
        let charge = error_mul(e + error_mul(4.0 * UNIT, v.abs(), true), mj, true)
            + error_mul(4.0 * UNIT, lnew.abs(), true);
        dust_hits += usize::from(nominal_product_dust(vp, sp) != 0.0)
            + usize::from(nominal_product_dust(vn, sn) != 0.0);
        let charge = ref_dust(charge, vp, sp);
        let want = slack16(ref_dust(charge, vn, sn));
        let got = relu_error_update_branch_free(e, v, lnew, vp, vn, sp, sn, mj);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "relu e={e:e} v={v:e} lnew={lnew:e} sp={sp:e} sn={sn:e} mj={mj:e}"
        );
    }
    assert!(dust_hits > 10_000, "dust_hits {dust_hits}");
    // error_axpy over random rows and weights (any sign, NaN, inf).
    for _ in 0..REVIEW_N / 16 {
        let n = 1 + g.below(24) as usize;
        let w = if g.below(4) == 0 {
            g.any()
        } else {
            g.mag().abs()
        };
        let error: Vec<f64> = (0..n).map(|_| g.mag()).collect();
        let nominal: Vec<f64> = (0..n)
            .map(|_| {
                if g.below(3) == 0 {
                    g.near(f64::MIN_POSITIVE / w.abs().max(f64::MIN_POSITIVE))
                } else {
                    g.any()
                }
            })
            .collect();
        let start: Vec<f64> = (0..n).map(|_| g.any()).collect();
        let mut want = start.clone();
        for ((a, &e), &v) in want.iter_mut().zip(&error).zip(&nominal) {
            let term = underflow_product_up(w, e);
            *a += ref_dust(term, w, v);
        }
        let mut got = start;
        error_axpy_branch_free(&mut got, w, &error, &nominal);
        assert_eq!(bits(got), bits(want), "error_axpy w={w:e}");
    }
}

/// A row operand for the bias / concretize fuzz: ordinary magnitudes with
/// the threshold zoo mixed in at a low rate, so a useful fraction of rows is
/// VALID (the branch-free path) while every edge still appears.
fn review_row(g: &mut Mix, n: usize, nonneg: bool, hazard_rate: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let v = if g.below(hazard_rate) == 0 {
                g.mag()
            } else {
                match g.below(12) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f64::from_bits(g.u() & 0x000F_FFFF_FFFF_FFFF),
                    3 => g.near(f64::MIN_POSITIVE),
                    4 => g.near(2f64.powi(400)),
                    5 => g.near(f64::MIN_POSITIVE.sqrt()),
                    _ => (g.u() >> 11) as f64 / (1u64 << 53) as f64 * 4.0,
                }
            };
            if nonneg || g.u() & 1 == 0 {
                v
            } else {
                -v
            }
        })
        .collect()
}

#[test]
fn review_bias_rows_match_the_engine_scalar_loop_on_random_rows() {
    let mut g = Mix(0x5EED_0003);
    let (mut admitted, mut admitted_env, mut refused) = (0usize, 0usize, 0usize);
    for it in 0..60_000usize {
        let r = 1 + g.below(40) as usize;
        let hz = [2, 8, 64, 1_000_000][it % 4];
        let lr = review_row(&mut g, r, false, hz);
        let er = review_row(&mut g, r, true, hz);
        let b = review_row(&mut g, r, false, hz);
        let bias = if g.below(6) == 0 {
            g.any()
        } else {
            review_row(&mut g, 1, false, hz)[0]
        };
        let berr = if g.below(6) == 0 {
            g.mag()
        } else {
            review_row(&mut g, 1, true, hz)[0]
        };
        let gam_b = gamma_n(1 + g.below(1 << 20) as usize);
        let ab = bias.abs();
        let start: Vec<f64> = (0..r).map(|_| g.any()).collect();
        // Per-column lane, the engine's scalar repaired loop verbatim.
        let mut want = start.clone();
        for (ri, ebv) in want.iter_mut().enumerate() {
            let mag = lr[ri].abs();
            let charge = error_mul(er[ri] + error_mul(gam_b, mag, true), ab, true)
                + error_mul(mag + er[ri], berr, true);
            *ebv += ref_dust(charge, bias, lr[ri]);
            if bias != 0.0 {
                *ebv += error_mul(2.0 * UNIT, b[ri].abs(), true);
            }
        }
        if bias_row_is_valid(&lr, &er, &b, bias, berr) {
            admitted += 1;
            let mut got = start.clone();
            bias_row_charges_branch_free(&mut got, &lr, &er, &b, bias, berr, gam_b);
            assert_eq!(
                bits(got),
                bits(want),
                "bias row it={it} bias={bias:e} berr={berr:e}"
            );
        } else {
            refused += 1;
        }
        // Envelope lane: one envelope value in place of the column budgets.
        let ev = if g.below(4) == 0 { g.mag() } else { er[0] };
        let mut want = start.clone();
        for (ri, ebv) in want.iter_mut().enumerate() {
            let mag = lr[ri].abs();
            let charge = error_mul(ev + error_mul(gam_b, mag, true), ab, true)
                + error_mul(mag + ev, berr, true);
            *ebv += ref_dust(charge, bias, lr[ri]);
            if bias != 0.0 {
                *ebv += error_mul(2.0 * UNIT, b[ri].abs(), true);
            }
        }
        if bias_row_is_valid_env(&lr, ev, &b, bias, berr) {
            admitted_env += 1;
            let mut got = start;
            bias_row_charges_branch_free_env(&mut got, &lr, ev, &b, bias, berr, gam_b);
            assert_eq!(bits(got), bits(want), "env bias row it={it} ev={ev:e}");
        } else {
            refused += 1;
        }
    }
    // The overflow edges the 2^400 magnitude cap exists for: a sum that
    // overflows to +inf times a ZERO factor is +inf (the refusal budget) in
    // the scalar helper but NaN in a bare product, so a row carrying one must
    // be refused or agree bit for bit. (`bias == 0 && berr == 0` rows never
    // reach the arm: the engine skips them first.)
    let big = 2f64.powi(1023);
    let edges: [(f64, f64, f64, f64, f64); 4] = [
        // e + gam_b*|l| overflows and |bias| = 0.
        (big, f64::MAX, 0.0, 1e-20, 2f64.powi(-40)),
        // |l| + e overflows and berr = 0.
        (big, big, 0.5, 0.0, 1e-12),
        (-big, big, -0.25, 0.0, 1e-12),
        // just above the cap, nothing overflows.
        (2f64.powi(401), 1.0, 1e-300, 0.0, 1e-12),
    ];
    let mut edge_checked = 0usize;
    for &(l, e, bias, berr, gam_b) in &edges {
        let lr = [l, 0.5];
        let er = [e, 0.0];
        let b = [1.0f64, -2.0];
        let ab = bias.abs();
        for env in [false, true] {
            let mut want = [0.25f64, 0.0];
            for (ri, ebv) in want.iter_mut().enumerate() {
                let ee = if env { e } else { er[ri] };
                let mag = lr[ri].abs();
                let charge = error_mul(ee + error_mul(gam_b, mag, true), ab, true)
                    + error_mul(mag + ee, berr, true);
                *ebv += ref_dust(charge, bias, lr[ri]);
                if bias != 0.0 {
                    *ebv += error_mul(2.0 * UNIT, b[ri].abs(), true);
                }
            }
            let admitted_row = if env {
                bias_row_is_valid_env(&lr, e, &b, bias, berr)
            } else {
                bias_row_is_valid(&lr, &er, &b, bias, berr)
            };
            if admitted_row {
                let mut got = [0.25f64, 0.0];
                if env {
                    bias_row_charges_branch_free_env(&mut got, &lr, e, &b, bias, berr, gam_b);
                } else {
                    bias_row_charges_branch_free(&mut got, &lr, &er, &b, bias, berr, gam_b);
                }
                assert_eq!(
                    bits(got),
                    bits(want.iter().copied()),
                    "overflow edge l={l:e} e={e:e} bias={bias:e} berr={berr:e} env={env}"
                );
            }
            edge_checked += 1;
            // The scalar helper's refusal budget is what a bare product would
            // turn into NaN: the edge really is an edge.
            assert!(want.iter().all(|v| !v.is_nan()));
        }
    }
    assert_eq!(edge_checked, 8);
    assert!(admitted > 2_000, "admitted {admitted}");
    assert!(admitted_env > 2_000, "admitted_env {admitted_env}");
    assert!(refused > 2_000, "refused {refused}");
}

#[test]
fn review_concretize_rows_match_the_engine_scalar_loop_on_random_rows() {
    let mut g = Mix(0x5EED_0004);
    let (mut admitted, mut refused) = (0usize, 0usize);
    for it in 0..60_000usize {
        let r = 1 + g.below(40) as usize;
        let hz = [2, 8, 64, 1_000_000][it % 4];
        let arow = review_row(&mut g, r, false, hz);
        let erow = review_row(&mut g, r, true, hz);
        let e = (g.below(3) != 0).then_some(&erow[..]);
        let m = if g.below(3) == 0 {
            g.any()
        } else {
            g.near(f64::MIN_POSITIVE)
        };
        let rd = if g.below(3) == 0 {
            g.mag()
        } else {
            review_row(&mut g, 1, true, hz)[0]
        };
        let xa = if g.below(5) == 0 {
            g.any()
        } else {
            review_row(&mut g, 1, true, hz)[0]
        };
        let start: Vec<f64> = (0..r).map(|_| g.any()).collect();
        // The engine's dust accumulator starts at +0 and only takes
        // `next_up(d + eta)` steps.
        let dust0: Vec<f64> = (0..r).map(|_| f64::from_bits(g.below(5))).collect();
        let (mut v, mut rr, mut tabs, mut pen, mut dust) = (
            start.clone(),
            start.clone(),
            start.clone(),
            start.clone(),
            dust0.clone(),
        );
        for (ri, av) in arow.iter().enumerate() {
            v[ri] += av * m;
            let aa = av.abs();
            rr[ri] += aa * rd;
            tabs[ri] += error_mul(aa, xa, true);
            let dm = nominal_product_dust(*av, m);
            let dr = nominal_product_dust(aa, rd);
            if dm != 0.0 {
                dust[ri] = next_up(dust[ri] + dm);
            }
            if dr != 0.0 {
                dust[ri] = next_up(dust[ri] + dr);
            }
        }
        if let Some(es) = e {
            for (ri, ev) in es.iter().enumerate() {
                pen[ri] += error_mul(*ev, xa, true);
            }
        }
        if concretize_row_is_valid(&arow, e, xa) {
            admitted += 1;
            let (mut fv, mut frr, mut ftabs, mut fpen, mut fdust) =
                (start.clone(), start.clone(), start.clone(), start, dust0);
            concretize_row_branch_free(
                &mut fv, &mut frr, &mut ftabs, &mut fdust, &mut fpen, &arow, e, m, rd, xa,
            );
            assert_eq!(bits(fv), bits(v), "v it={it}");
            assert_eq!(bits(frr), bits(rr), "rr it={it}");
            assert_eq!(bits(ftabs), bits(tabs), "tabs it={it}");
            assert_eq!(bits(fdust), bits(dust), "dust it={it}");
            assert_eq!(bits(fpen), bits(pen), "pen it={it}");
        } else {
            refused += 1;
        }
    }
    assert!(admitted > 2_000, "admitted {admitted}");
    assert!(refused > 2_000, "refused {refused}");
}

/// The conv screen on a contraction whose column count spans SEVERAL
/// accumulator blocks (`ci = 520` makes `r_blk = 64`, so `r = 150` runs as
/// 64 + 64 + 22: the existing kernel tests all fit one block), with row
/// floors and weights engineered so `w * floor` lands within a few ulps of
/// the screen's `2 * MIN_POSITIVE` threshold on both sides, plus rows whose
/// products genuinely underflow and invalid rows. Every class (fast, widen,
/// exact, scalar) must occur, and the result must equal the oracle.
#[test]
fn review_conv_screen_multiblock_threshold_rows_equal_the_oracle() {
    let mut g = Mix(0x5EED_0005);
    let (co, ci) = (3usize, 520usize);
    let (oh, ow) = (2usize, 2usize);
    let rows = co * oh * ow;
    let r = 150usize;
    // Row floors spanning the regimes; rows 9..12 get hazards.
    let floors_wanted: Vec<f64> = (0..rows)
        .map(|i| {
            [
                1e-300,
                1e-200,
                1e-160,
                f64::MIN_POSITIVE.sqrt(),
                1e-12,
                0.25,
            ][i % 6]
        })
        .collect();
    let mut err = Vec::with_capacity(rows * r);
    let mut nom = Vec::with_capacity(rows * r);
    for (i, &f) in floors_wanted.iter().enumerate() {
        // Odd rows carry nominal magnitudes far above their error floor, so a
        // weight the error floor declines can still clear the nominal floor
        // (the widen-only tier); even rows share one floor (the exact tier).
        let nscale = if i % 2 == 1 { 1e100 } else { 1.0 };
        for j in 0..r {
            // The floor itself at column j = 0 (error) and 1 (nominal); every
            // other nonzero entry at or above it (some a few ulps above).
            let e = match j % 7 {
                0 => f,
                1 | 4 => 0.0,
                2 => -0.0,
                3 => g.near(f * 1.0000001).max(f),
                _ => f * (1.0 + (g.below(1000) as f64)),
            };
            let l = nscale
                * match j % 5 {
                    1 => -f,
                    2 => 0.0,
                    3 => g.near(f * 2.0),
                    _ => f * (2.0 + (g.below(1000) as f64)) * if j % 2 == 0 { 1.0 } else { -1.0 },
                };
            err.push(e);
            nom.push(l);
        }
        if i >= 9 {
            // Hazard rows: an invalid entry somewhere forces the scalar path.
            let at = i * r + 1 + (i * 37) % (r - 1);
            match i {
                9 => err[at] = f64::NAN,
                10 => err[at] = -1e-20,
                _ => nom[at] = f64::INFINITY,
            }
        }
    }
    let err = Array2::from_shape_vec((rows, r), err).unwrap();
    let nom = Array2::from_shape_vec((rows, r), nom).unwrap();
    let floors = error_row_floors(&err, &nom).expect("aligned");
    // Weights: for each output channel, some within a few ulps of the
    // threshold weight 2*MIN/floor for that channel's rows, some below it
    // (so products underflow), some ordinary.
    let mut weights = vec![0.0; co * ci];
    for oc in 0..co {
        for icx in 0..ci {
            // Exactly one output channel feeds each input channel (blocks of
            // 12 = lcm(6 weight variants, 4 spatial rows)), so every
            // accumulator entry is a SINGLE term: a one-eta widening or dust
            // difference cannot be absorbed by a larger term of the sum.
            if oc != (icx / 12) % co {
                continue;
            }
            let f = floors[oc * oh * ow + (icx % (oh * ow))][0].max(f64::MIN_POSITIVE);
            let thr = 2.0 * f64::MIN_POSITIVE / f;
            let w = match icx % 6 {
                0 => g.near(thr),
                1 => g.near(thr) * 0.5,
                2 => g.near(thr) * 1e-20,
                3 => (g.below(1000) as f64 + 1.0) / 1000.0,
                4 => 0.0,
                _ => -g.near(thr),
            };
            // compile_conv takes [co][ci][kh][kw] = [co][ci] here.
            weights[oc * ci + icx] = if w.is_finite() { w } else { 1.0 };
        }
    }
    let c = compile_conv(
        0,
        &weights,
        &vec![0.0; co],
        &vec![0.0; co],
        0.0,
        (co, ci, 1, 1),
        (1, 1),
        (0, 0, 0, 0),
        (ci, oh, ow),
        (co, oh, ow),
    );
    // Classify every (weight, row) block the kernel will see, with the
    // kernel's own predicates, and require every class plus near-threshold
    // fast blocks.
    let (mut fast, mut widen, mut exact, mut slow, mut edge_fast, mut edge_declined) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for oc in 0..co {
        for icx in 0..ci {
            let w = c.wt[icx * co + oc].abs();
            if w == 0.0 {
                continue;
            }
            for osp in 0..oh * ow {
                let [floor, lfloor] = floors[oc * oh * ow + osp];
                let p = w * floor;
                let edge = (p - 2.0 * f64::MIN_POSITIVE).abs() <= 8.0 * f64::from_bits(1);
                if screened_block_is_normal(w, floor) {
                    fast += 1;
                    edge_fast += usize::from(edge);
                } else if screened_row_is_valid(w, floor) && screened_block_is_normal(w, lfloor) {
                    widen += 1;
                    edge_declined += usize::from(edge);
                } else if screened_row_is_valid(w, floor) {
                    exact += 1;
                    edge_declined += usize::from(edge);
                } else {
                    slow += 1;
                }
            }
        }
    }
    assert!(
        fast > 0 && widen > 0 && exact > 0 && slow > 0 && edge_fast > 0 && edge_declined > 0,
        "fast {fast} widen {widen} exact {exact} slow {slow} edge_fast {edge_fast} edge_declined {edge_declined}"
    );
    let (screened, oracle) = run_pair(&c, &err, &nom);
    assert_eq!(screened, oracle);
}

/// Whole-engine bit identity in the ENVELOPE lanes (row envelope and
/// per-channel envelope), which the screen also changes (the envelope bias
/// arm, concretize, the head axpy) and which the existing engine test does
/// not run: cifar100/tinyimagenet presets publish `err_envelope: envelope`.
#[test]
fn review_engine_envelope_lanes_bit_identical_with_and_without_screen() {
    let _guard = SWITCH.lock().unwrap_or_else(|p| p.into_inner());
    for lane in [ErrLane::Envelope, ErrLane::PerChannel, ErrLane::PerCol] {
        for tiny in [false, true] {
            let spec = engine_spec(0xE57, tiny);
            let net = TwinNet::compile(&spec).expect("compile");
            let lo = vec![-0.3; spec.n_in];
            let hi = vec![0.45; spec.n_in];
            let rows0 = ERRCONV_TEST_ROWS.with(std::cell::Cell::get);
            let on = review_lane_snapshot(&net, &lo, &hi, lane);
            let rows1 = ERRCONV_TEST_ROWS.with(std::cell::Cell::get);
            ERRCONV_SCREEN_TEST_OFF.store(true, SeqCst);
            let off = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                review_lane_snapshot(&net, &lo, &hi, lane)
            }));
            ERRCONV_SCREEN_TEST_OFF.store(false, SeqCst);
            let rows2 = ERRCONV_TEST_ROWS.with(std::cell::Cell::get);
            let off = off.expect("oracle snapshot");
            let on_rows: Vec<u64> = (0..3).map(|k| rows1[k] - rows0[k]).collect();
            let off_rows: Vec<u64> = (0..3).map(|k| rows2[k] - rows1[k]).collect();
            assert_eq!(on.0.len(), off.0.len());
            for (i, (a, b)) in on.0.iter().zip(&off.0).enumerate() {
                assert_eq!(a, b, "{lane:?} tiny={tiny}: snapshot section {i} differs");
            }
            // The screened row sites were reached in the ON arm and never in the
            // oracle arm. An envelope pass publishes its envelope BROADCAST
            // into `PassOut::e`, so the envelope lane's engagement is read
            // from its own bias-arm site (it runs only while `cur.env` is
            // set), not from the published matrix.
            assert_eq!(
                off_rows,
                vec![0, 0, 0],
                "{lane:?} tiny={tiny}: oracle arm screened"
            );
            assert!(
                on_rows[2] > 0,
                "{lane:?} tiny={tiny}: concretize screen never ran {on_rows:?}"
            );
            if lane == ErrLane::PerCol {
                assert!(
                    on_rows[0] > 0,
                    "{lane:?} tiny={tiny}: per-col bias screen never ran {on_rows:?}"
                );
                assert_eq!(
                    on_rows[1], 0,
                    "{lane:?} tiny={tiny}: an envelope bias row ran {on_rows:?}"
                );
            } else {
                assert!(
                    on_rows[1] > 0,
                    "{lane:?} tiny={tiny}: envelope bias screen never ran {on_rows:?}"
                );
            }
        }
    }
}

fn review_lane_snapshot(
    net: &TwinNet,
    lo: &[f64],
    hi: &[f64],
    lane: ErrLane,
) -> (Vec<Vec<u64>>, usize) {
    let root = RootGates::build(net, lo, hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(net, &root)
        .with_err_lane(lane)
        .with_err_min_cols(1);
    let mut words = Vec::new();
    let mut passes = 0usize;
    let mut push = |out: &crate::margin_row::engine::PassOut, lower: bool| {
        passes += 1;
        words.push(bits(out.a.iter().copied()));
        if let Some(e) = out.e.as_ref() {
            words.push(bits(e.iter().copied()));
        }
        words.push(bits(out.b.iter().copied()));
        words.push(bits(out.eb.iter().copied()));
        let conc = if lower {
            eng.concretize_lower(out)
        } else {
            eng.concretize_upper(out)
        };
        words.push(bits(conc));
    };
    let (al, au) = eng.y_rows(None).expect("y rows");
    push(&al, true);
    push(&au, false);
    let n_y = net.n_y;
    for cols in [1usize, 9, 40] {
        let s = Array2::from_shape_vec((n_y, cols), signed(n_y * cols, 0xD1 + cols as u64, false))
            .unwrap();
        let e = Array2::from_shape_vec((n_y, cols), budgets(n_y * cols, 0xD2 + cols as u64, false))
            .unwrap();
        for with_e in [false, true] {
            let seed = Seed {
                s: s.clone(),
                e: with_e.then(|| e.clone()),
            };
            for dir in [LaneDir::Lower, LaneDir::Upper] {
                let out = eng.run(&seed, None, dir, None, false).expect("dense pass");
                push(&out, matches!(dir, LaneDir::Lower));
            }
        }
    }
    for li in 0..root.layers.len() {
        let relu_layer = &root.layers[li];
        let n_pre = relu_layer.n;
        let w = n_pre.min(24);
        let mut s = Array2::<f64>::zeros((n_pre, w));
        for c in 0..w {
            s[[c, c]] = 1.0;
        }
        let seed = Seed { s, e: None };
        for dir in [LaneDir::Lower, LaneDir::Upper] {
            let out = eng
                .run_prefix_dom(&seed, relu_layer.op, dir, None)
                .expect("prefix pass");
            push(&out, matches!(dir, LaneDir::Lower));
        }
    }
    (words, passes)
}
