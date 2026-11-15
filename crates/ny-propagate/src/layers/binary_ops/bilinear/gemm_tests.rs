// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-batched-bilinear (ENGINE_PROGRAM §2 V1): tests of the GEMM-shaped
//! McCormick backward (`matmul/crown_gemm.rs`) on BOTH arms — the midpoint
//! `MatMul` arm and the interpolated-α `BilinearCrown` arm.
//!
//! * `bilinear_gemm_bit_identical_to_serial` — the soundness argument: every
//!   published bit (coefficients, halved constants, certified error planes) of
//!   the GEMM-shaped kernel equals the historical serial nest's, called
//!   directly and through every routed kernel choice, on vit_2023's attention
//!   shapes and on batch/2-D/`k = 1`/pathological-weight shapes, plus finite
//!   subnormal-weight cases whose relation stays live (the pathological ones
//!   collapse to the conservative relation in both paths). Any change to
//!   one floating-point operation, its operands, its order per accumulator, a
//!   plane selection or a guard flips bits on these inputs, so this is also
//!   the kernel's mutation pin. Because every fallback is ALSO bit-identical,
//!   each call additionally asserts its ENGAGEMENT through the test-only
//!   per-thread probe (`collector_kernels::gemm_calls_this_thread`): the
//!   SERIAL reference must record 0 GEMM-shaped completions and every direct
//!   or routed GEMM choice exactly 1, so a router that silently falls through
//!   to the row-parallel twin or the nest fails here instead of reading green.
//! * `bilinear_gemm_thread_count_independent` — the same bits under rayon
//!   pools of 1, 2 and 5 threads, row chunks of 1, 8, 13 and all rows, fanned
//!   out and sequential, and through the production schedule.
//! * `bilinear_gemm_contains_exact_product_on_random_attention_blocks` — the
//!   published relation (the SUM of the two operand shares, certified error
//!   charged outward) contains the exact `C·(s·A·B) + bias` at box vertices and
//!   interior points, in exact rational arithmetic, on random attention-shaped
//!   blocks including one vit-sized block.

use ndarray::{Array1, Array2, Array4, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_tensor::BoundedTensor;

use super::super::matmul::{parse_matmul_dims, GemmSchedule, MatMulLayer, McCormickRowparShape};
use super::BilinearCrownLayer;
use crate::layers::collector_kernels::{gemm_calls_this_thread, BilinearKernelChoice};
use crate::LinearBounds;

/// Run `call` and assert how many GEMM-shaped completions it recorded on this
/// thread. Every fallback (row-parallel twin, serial nest) is bit-identical,
/// so a routed result that matches the nest proves nothing about WHICH kernel
/// ran; this is the engagement half of the pin (`expected == 1`: the GEMM
/// kernel ran exactly once; `expected == 0`: it was never touched).
fn engaged<T>(expected: u64, label: &str, call: impl FnOnce() -> T) -> T {
    let before = gemm_calls_this_thread();
    let out = call();
    let recorded = gemm_calls_this_thread() - before;
    assert_eq!(
        recorded, expected,
        "{label}: expected {expected} GEMM-shaped completion(s) on this thread, recorded \
         {recorded} (a routed entry that silently falls through to a bit-identical sibling \
         is a VOID engagement, not a pass)"
    );
    out
}

/// SplitMix64: deterministic, dependency-free, bit-reproducible.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, exactly representable (24 random bits).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    fn pick(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// One endpoint pair: mixed sign, a point interval, a tiny endpoint beside a
/// large one, a same-sign interval away from zero, or a random decade.
fn interval(rng: &mut SplitMix) -> (f32, f32) {
    let (a, b) = match rng.pick(5) {
        0 => (rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)),
        1 => {
            let v = rng.range(-3.0, 3.0);
            (v, v)
        }
        2 => (rng.range(-1e-3, 1e-3), rng.range(0.5, 500.0)),
        3 => {
            let base = rng.range(1.0, 100.0);
            (base, base + rng.range(0.0, 20.0))
        }
        _ => {
            let scale = 10_f32.powi(rng.pick(7) as i32 - 3);
            (rng.range(-1.0, 1.0) * scale, rng.range(-1.0, 1.0) * scale)
        }
    };
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn random_box(rng: &mut SplitMix, shape: &[usize]) -> BoundedTensor {
    let len: usize = shape.iter().product();
    let (lower, upper): (Vec<f32>, Vec<f32>) = (0..len).map(|_| interval(rng)).unzip();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap(),
    )
    .unwrap()
}

/// How the incoming coefficients are drawn.
#[derive(Clone, Copy, Debug)]
enum Weights {
    /// Mixed sign with ~20% exact zeros and a few `-0.0` (the `w != 0.0`
    /// guards and the `w >= 0.0` plane choice).
    Ordinary,
    /// Subnormals, `-0.0`, huge magnitudes whose product with a non-unit scale
    /// overflows to `±inf` (so `inf · 0` NaN cells and the NaN firewall are
    /// exercised), and ordinary values. NOTE: one non-finite coefficient cell
    /// makes `new_or_conservative` replace the WHOLE share by the conservative
    /// relation in both paths, so a share that overflows anywhere pins the
    /// firewall's agreement, not the accumulation bits; [`Weights::Tiny`]
    /// carries the subnormal corner with the relation guaranteed live.
    Pathological,
    /// Subnormals, `±0.0`, `±MIN_POSITIVE` and ordinary values, nothing large:
    /// every published number stays finite, so the firewall does NOT collapse
    /// the relation and the underflow corner — `coefficient * scale` rounding
    /// to (or from) a subnormal, a product that underflows to `0` and is then
    /// skipped by `w != 0.0` in both paths — is compared bit for bit.
    Tiny,
}

fn random_bounds(rng: &mut SplitMix, rows: usize, cols: usize, weights: Weights) -> LinearBounds {
    let coeff = |rng: &mut SplitMix| match weights {
        Weights::Ordinary => match rng.pick(10) {
            0 | 1 => 0.0,
            2 => -0.0,
            _ => rng.range(-2.0, 2.0),
        },
        Weights::Pathological => match rng.pick(12) {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1 + (rng.next_u64() % 0x007f_ffff) as u32),
            3 => -f32::from_bits(1 + (rng.next_u64() % 0x007f_ffff) as u32),
            4 => 3.0e38,
            5 => -3.0e38,
            6 => f32::MIN_POSITIVE,
            _ => rng.range(-2.0, 2.0),
        },
        Weights::Tiny => match rng.pick(10) {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1 + (rng.next_u64() % 0x007f_ffff) as u32),
            3 => -f32::from_bits(1 + (rng.next_u64() % 0x007f_ffff) as u32),
            // The smallest subnormals: `* scale` (< 1) rounds these to zero.
            4 => f32::from_bits(1 + (rng.next_u64() % 3) as u32),
            5 => f32::MIN_POSITIVE,
            6 => -f32::MIN_POSITIVE,
            _ => rng.range(-2.0, 2.0),
        },
    };
    let lower_a = Array2::from_shape_fn((rows, cols), |_| coeff(rng));
    let upper_a = Array2::from_shape_fn((rows, cols), |_| coeff(rng));
    let lower_b = Array1::from_shape_fn(rows, |_| rng.range(-1.0, 1.0));
    let upper_b = Array1::from_shape_fn(rows, |_| rng.range(-1.0, 1.0));
    LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap()
}

fn shape_for(transpose_b: bool, a: &[usize], b: &[usize]) -> McCormickRowparShape {
    let dims = parse_matmul_dims(transpose_b, a, b).unwrap();
    let batch_size = dims.batch_size().unwrap();
    let c_size_per_batch = dims.c_size_per_batch().unwrap();
    let a_size_per_batch = dims.a_size_per_batch().unwrap();
    McCormickRowparShape {
        batch_size,
        m: dims.m,
        n: dims.n,
        k: dims.k,
        c_size_per_batch,
        a_size_per_batch,
        b_size_per_batch: dims.b_size_per_batch,
        total_a_size: batch_size * a_size_per_batch,
        total_b_size: batch_size * dims.b_size_per_batch,
        transpose_b,
    }
}

/// One f32 as compared bits. NaN is canonicalized: the kernel and the nest
/// perform the same operations on the same operands, but IEEE 754 leaves the
/// payload of a NaN produced from two NaN operands implementation-defined, so
/// only "is NaN" is part of the contract. (Every other value, including the
/// sign of zero, compares bit for bit.)
fn f32_bits(value: f32) -> u32 {
    if value.is_nan() {
        0x7fc0_0000
    } else {
        value.to_bits()
    }
}

/// Every published bit of one share, certified-error planes included.
fn bits(bounds: &LinearBounds) -> Vec<u32> {
    let mut out: Vec<u32> = bounds.lower_a().iter().map(|&v| f32_bits(v)).collect();
    out.extend(bounds.upper_a().iter().map(|&v| f32_bits(v)));
    out.extend(bounds.lower_b().iter().map(|&v| f32_bits(v)));
    out.extend(bounds.upper_b().iter().map(|&v| f32_bits(v)));
    out.push(u32::from(bounds.lower_a_err().is_some()));
    out.push(u32::from(bounds.upper_a_err().is_some()));
    for err in [bounds.lower_a_err(), bounds.upper_a_err()]
        .into_iter()
        .flatten()
    {
        out.extend(err.iter().map(|&v| f32_bits(v)));
    }
    out
}

fn assert_same(
    serial: &(LinearBounds, LinearBounds),
    other: &(LinearBounds, LinearBounds),
    label: &str,
) {
    assert_eq!(bits(&serial.0), bits(&other.0), "{label}: A share differs");
    assert_eq!(bits(&serial.1), bits(&other.1), "{label}: B share differs");
}

/// The production choices that must reach the GEMM-shaped kernel.
const GEMM_CHOICES: [BilinearKernelChoice; 2] = [
    BilinearKernelChoice {
        gemm: true,
        rowpar: false,
    },
    BilinearKernelChoice {
        gemm: true,
        rowpar: true,
    },
];

/// The soundness argument of `crown_gemm.rs`, pinned: the GEMM-shaped kernel
/// publishes EXACTLY the serial nest's bits on both arms, called directly (so
/// a silent decline cannot pass) and through every routed choice.
#[test]
fn bilinear_gemm_bit_identical_to_serial() {
    let mut rng = SplitMix(0x0B11_1EA5_6E33_2026);
    // (transpose_b, A shape, B shape, scale, incoming rows, weights)
    let cases: Vec<(bool, Vec<usize>, Vec<usize>, Option<f32>, usize, Weights)> = vec![
        // vit_2023 ibp attn·V: attn [3,17,17] · V [3,17,16].
        (
            false,
            vec![3, 17, 17],
            vec![3, 17, 16],
            None,
            37,
            Weights::Ordinary,
        ),
        // vit_2023 ibp Q·Kᵀ with the attention scale; 9 rows = one full tile + 1.
        (
            true,
            vec![3, 17, 16],
            vec![3, 17, 16],
            Some(0.25),
            9,
            Weights::Ordinary,
        ),
        // The brief's shape pair [3,17,16]·[3,16,17], non-power-of-two scale.
        (
            false,
            vec![3, 17, 16],
            vec![3, 16, 17],
            Some(0.37),
            16,
            Weights::Ordinary,
        ),
        // vit_2023 pgd Q·Kᵀ (5 tokens), many rows (many tiles and chunks).
        (
            true,
            vec![3, 5, 16],
            vec![3, 5, 16],
            Some(0.25),
            65,
            Weights::Ordinary,
        ),
        // Two batch dims.
        (
            false,
            vec![2, 3, 5, 4],
            vec![2, 3, 4, 6],
            Some(0.37),
            17,
            Weights::Ordinary,
        ),
        // Plain 2-D, one row.
        (true, vec![5, 7], vec![3, 7], None, 1, Weights::Ordinary),
        // k = 1.
        (
            false,
            vec![4, 1],
            vec![1, 3],
            Some(1.5),
            8,
            Weights::Ordinary,
        ),
        // Pathological weights on an attention shape: subnormals, -0.0,
        // overflow of `coefficient * scale` to ±inf.
        (
            true,
            vec![2, 6, 4],
            vec![2, 6, 4],
            Some(1.5),
            11,
            Weights::Pathological,
        ),
        (
            false,
            vec![2, 5, 3],
            vec![2, 3, 4],
            None,
            7,
            Weights::Pathological,
        ),
        // Tiny (finite, live relation): the subnormal / underflow-to-zero
        // corner compared bit for bit, with the vit attention scale and a
        // non-power-of-two scale.
        (
            true,
            vec![3, 17, 16],
            vec![3, 17, 16],
            Some(0.25),
            11,
            Weights::Tiny,
        ),
        (
            false,
            vec![2, 5, 3],
            vec![2, 3, 4],
            Some(0.37),
            9,
            Weights::Tiny,
        ),
    ];
    let mut compared = 0_usize;
    let mut nonzero_coefficients = 0_usize;
    for (case, (transpose_b, a_shape, b_shape, scale, rows, weights)) in cases.iter().enumerate() {
        let shape = shape_for(*transpose_b, a_shape, b_shape);
        let total_c = shape.batch_size * shape.c_size_per_batch;
        let raw_scale = scale.unwrap_or(1.0);
        let schedule = GemmSchedule {
            chunk_rows: 8,
            parallel: true,
        };
        for rep in 0..2 {
            let label = format!("case {case} rep {rep} ({weights:?})");
            let a = random_box(&mut rng, a_shape);
            let b = random_box(&mut rng, b_shape);
            let incoming = random_bounds(&mut rng, *rows, total_c, *weights);

            // Midpoint (alpha-free) MatMul arm.
            let matmul = MatMulLayer::new(*transpose_b, *scale);
            let serial = engaged(0, &format!("{label} midpoint SERIAL"), || {
                matmul
                    .propagate_linear_binary_with_kernels(
                        &incoming,
                        &a,
                        &b,
                        BilinearKernelChoice::SERIAL,
                    )
                    .unwrap()
            });
            let direct = engaged(1, &format!("{label} midpoint direct"), || {
                matmul
                    .propagate_linear_binary_gemm(
                        &incoming,
                        &a,
                        &b,
                        &shape,
                        raw_scale,
                        Some(schedule),
                    )
                    .unwrap()
                    .expect("the GEMM-shaped midpoint kernel must accept this shape")
            });
            assert_same(&serial, &direct, &format!("{label} midpoint direct"));
            for choice in GEMM_CHOICES {
                let routed = engaged(1, &format!("{label} midpoint routed {choice:?}"), || {
                    matmul
                        .propagate_linear_binary_with_kernels(&incoming, &a, &b, choice)
                        .unwrap()
                });
                assert_same(
                    &serial,
                    &routed,
                    &format!("{label} midpoint routed {choice:?}"),
                );
            }
            // Non-vacuity: the compared relations carry real coefficients.
            let case_nonzero = direct
                .0
                .lower_a()
                .iter()
                .chain(direct.1.upper_a().iter())
                .filter(|&&v| v != 0.0)
                .count();
            // A non-pathological case must NOT have been collapsed to the
            // conservative relation by the NaN firewall (all-zero
            // coefficients): that would compare two conservative objects and
            // pin nothing about the accumulation.
            if !matches!(weights, Weights::Pathological) {
                assert!(
                    case_nonzero > 0,
                    "{label}: the midpoint relation is all-zero (firewall-collapsed?), so its \
                     bit-identity would be vacuous"
                );
            }
            nonzero_coefficients += case_nonzero;

            // Interpolated alpha BilinearCrown arm; parameters outside [0, 1]
            // exercise the per-element clamp.
            let bilinear = BilinearCrownLayer::new(*transpose_b, *scale);
            let alphas =
                Array4::from_shape_fn((4, shape.m, shape.n, shape.k), |_| match rng.pick(6) {
                    0 => 0.0,
                    1 => 1.0,
                    _ => rng.range(-0.25, 1.25),
                });
            let serial = engaged(0, &format!("{label} alpha SERIAL"), || {
                bilinear
                    .propagate_linear_binary_with_alpha_kernels(
                        &incoming,
                        &a,
                        &b,
                        Some(&alphas),
                        BilinearKernelChoice::SERIAL,
                    )
                    .unwrap()
            });
            let direct = engaged(1, &format!("{label} alpha direct"), || {
                bilinear
                    .propagate_linear_binary_with_alpha_gemm(
                        &incoming,
                        &a,
                        &b,
                        &alphas,
                        &shape,
                        raw_scale,
                        Some(schedule),
                    )
                    .unwrap()
                    .expect("the GEMM-shaped alpha kernel must accept this shape")
            });
            assert_same(&serial, &direct, &format!("{label} alpha direct"));
            // The alpha entry's `None` arm delegates to the midpoint arm with
            // the SAME choice.
            let none_serial = engaged(0, &format!("{label} alpha None SERIAL"), || {
                bilinear
                    .propagate_linear_binary_with_alpha_kernels(
                        &incoming,
                        &a,
                        &b,
                        None,
                        BilinearKernelChoice::SERIAL,
                    )
                    .unwrap()
            });
            for choice in GEMM_CHOICES {
                let routed = engaged(1, &format!("{label} alpha routed {choice:?}"), || {
                    bilinear
                        .propagate_linear_binary_with_alpha_kernels(
                            &incoming,
                            &a,
                            &b,
                            Some(&alphas),
                            choice,
                        )
                        .unwrap()
                });
                assert_same(
                    &serial,
                    &routed,
                    &format!("{label} alpha routed {choice:?}"),
                );
                let none_routed = engaged(1, &format!("{label} alpha None {choice:?}"), || {
                    bilinear
                        .propagate_linear_binary_with_alpha_kernels(&incoming, &a, &b, None, choice)
                        .unwrap()
                });
                assert_same(
                    &none_serial,
                    &none_routed,
                    &format!("{label} alpha None {choice:?}"),
                );
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 11 * 2, "every case and rep must be compared");
    assert!(
        nonzero_coefficients > 0,
        "the compared relations must carry coefficients, or the bit-identity is vacuous"
    );
}

/// Thread-count and schedule independence: the kernel's bits do not depend on
/// the rayon pool size, the row-chunk size, whether chunks fan out, or the
/// production schedule — each is compared against the serial nest.
#[test]
fn bilinear_gemm_thread_count_independent() {
    let mut rng = SplitMix(0x7EAD_C0DE_2026_0929);
    // vit_2023 ibp Q·Kᵀ; 21 rows = two full tiles + 5.
    let (transpose_b, a_shape, b_shape, scale, rows) = (
        true,
        [3_usize, 17, 16],
        [3_usize, 17, 16],
        Some(0.25_f32),
        21,
    );
    let shape = shape_for(transpose_b, &a_shape, &b_shape);
    let total_c = shape.batch_size * shape.c_size_per_batch;
    let a = random_box(&mut rng, &a_shape);
    let b = random_box(&mut rng, &b_shape);
    let incoming = random_bounds(&mut rng, rows, total_c, Weights::Ordinary);
    let alphas = Array4::from_shape_fn((4, shape.m, shape.n, shape.k), |_| rng.range(-0.25, 1.25));
    let matmul = MatMulLayer::new(transpose_b, scale);
    let bilinear = BilinearCrownLayer::new(transpose_b, scale);
    let raw_scale = scale.unwrap_or(1.0);

    let reference_mid = matmul
        .propagate_linear_binary_with_kernels(&incoming, &a, &b, BilinearKernelChoice::SERIAL)
        .unwrap();
    let reference_alpha = bilinear
        .propagate_linear_binary_with_alpha_kernels(
            &incoming,
            &a,
            &b,
            Some(&alphas),
            BilinearKernelChoice::SERIAL,
        )
        .unwrap();

    let mut runs = 0_usize;
    for threads in [1_usize, 2, 5] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a rayon pool of the requested size");
        for chunk_rows in [1_usize, 8, 13, rows] {
            for parallel in [false, true] {
                let schedule = GemmSchedule {
                    chunk_rows,
                    parallel,
                };
                let (mid, alpha) = pool.install(|| {
                    let mid = matmul
                        .propagate_linear_binary_gemm(
                            &incoming,
                            &a,
                            &b,
                            &shape,
                            raw_scale,
                            Some(schedule),
                        )
                        .unwrap()
                        .expect("the midpoint kernel must accept the vit shape");
                    let alpha = bilinear
                        .propagate_linear_binary_with_alpha_gemm(
                            &incoming,
                            &a,
                            &b,
                            &alphas,
                            &shape,
                            raw_scale,
                            Some(schedule),
                        )
                        .unwrap()
                        .expect("the alpha kernel must accept the vit shape");
                    (mid, alpha)
                });
                let label =
                    format!("threads={threads} chunk_rows={chunk_rows} parallel={parallel}");
                assert_same(&reference_mid, &mid, &format!("{label} midpoint"));
                assert_same(&reference_alpha, &alpha, &format!("{label} alpha"));
                runs += 1;
            }
        }
        // The production schedule (chunk size from the pool's thread count,
        // fan-out by the nesting policy) through the routed entries.
        let gemm = BilinearKernelChoice {
            gemm: true,
            rowpar: false,
        };
        // The engagement probe runs INSIDE the pool: the kernel records on the
        // worker that called it, and a dedicated pool runs no foreign jobs.
        let (mid, alpha) = pool.install(|| {
            (
                engaged(1, &format!("threads={threads} production midpoint"), || {
                    matmul
                        .propagate_linear_binary_with_kernels(&incoming, &a, &b, gemm)
                        .unwrap()
                }),
                engaged(1, &format!("threads={threads} production alpha"), || {
                    bilinear
                        .propagate_linear_binary_with_alpha_kernels(
                            &incoming,
                            &a,
                            &b,
                            Some(&alphas),
                            gemm,
                        )
                        .unwrap()
                }),
            )
        });
        assert_same(
            &reference_mid,
            &mid,
            &format!("threads={threads} production midpoint"),
        );
        assert_same(
            &reference_alpha,
            &alpha,
            &format!("threads={threads} production alpha"),
        );
    }
    // Outside any pool: the driver thread may fan out on the global pool.
    let mid = matmul
        .propagate_linear_binary_gemm(
            &incoming,
            &a,
            &b,
            &shape,
            raw_scale,
            Some(GemmSchedule {
                chunk_rows: 8,
                parallel: true,
            }),
        )
        .unwrap()
        .expect("the midpoint kernel must accept the vit shape");
    assert_same(&reference_mid, &mid, "global pool midpoint");
    assert_eq!(runs, 3 * 4 * 2, "every schedule must be compared");
}

/// Exact value of an f32 (every finite f32 is a dyadic rational).
fn rat(value: f32) -> BigRational {
    BigRational::from_float(f64::from(value)).expect("finite f32 is an exact rational")
}

/// A point of `[lo, hi]` as an f32 that is guaranteed to lie inside it.
fn inside(lo: f32, hi: f32, t: f32) -> f32 {
    (lo + (hi - lo) * t).clamp(lo, hi)
}

/// One attention-shaped block: `C = s · A · B` (or `s · A · Bᵀ`), `A` is
/// `[batch, m, k]`, `B` is `[batch, k, n]` (or `[batch, n, k]` transposed).
#[derive(Clone, Copy, Debug)]
struct Block {
    transpose_b: bool,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    scale: Option<f32>,
    rows: usize,
    samples: usize,
}

impl Block {
    fn a_shape(&self) -> [usize; 3] {
        [self.batch, self.m, self.k]
    }

    fn b_shape(&self) -> [usize; 3] {
        if self.transpose_b {
            [self.batch, self.n, self.k]
        } else {
            [self.batch, self.k, self.n]
        }
    }
}

/// Exact `Σ_c coeff[c] · (s · A @ B)_c + bias` at one operand point.
fn exact_target(block: &Block, coeff_row: &[f32], bias: f32, a: &[f32], b: &[f32]) -> BigRational {
    let (batch, m, n, k) = (block.batch, block.m, block.n, block.k);
    let scale = rat(block.scale.unwrap_or(1.0));
    let mut total = rat(bias);
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let c = coeff_row[bi * m * n + i * n + j];
                if c == 0.0 {
                    continue;
                }
                let mut dot = rat(0.0);
                for l in 0..k {
                    let av = a[bi * m * k + i * k + l];
                    let bv = if block.transpose_b {
                        b[bi * n * k + j * k + l]
                    } else {
                        b[bi * k * n + l * n + j]
                    };
                    dot += rat(av) * rat(bv);
                }
                total += rat(c) * scale.clone() * dot;
            }
        }
    }
    total
}

/// `(lower, upper)` of ONE operand share's row at a point, evaluated EXACTLY:
/// its halved constant plus `Σ coeff·p`, with the certified coefficient error
/// charged outward (`∓ err·|p|`, the charge `concretize_sound` applies).
fn share_at(share: &LinearBounds, row: usize, point: &[f32]) -> (BigRational, BigRational) {
    assert_eq!(share.num_inputs(), point.len(), "share / point width");
    let mut lower = rat(share.lower_b()[row]);
    let mut upper = rat(share.upper_b()[row]);
    for (col, &p) in point.iter().enumerate() {
        let value = rat(p);
        let magnitude = rat(p.abs());
        lower += rat(share.lower_a()[[row, col]]) * value.clone();
        upper += rat(share.upper_a()[[row, col]]) * value;
        if let Some(err) = share.lower_a_err() {
            lower -= rat(err[[row, col]]) * magnitude.clone();
        }
        if let Some(err) = share.upper_a_err() {
            upper += rat(err[[row, col]]) * magnitude;
        }
    }
    (lower, upper)
}

/// Every published number of a pair is finite (so the exact evaluation is
/// meaningful and the containment check is not passed vacuously).
fn pair_is_finite(pair: &(LinearBounds, LinearBounds)) -> bool {
    [&pair.0, &pair.1].iter().all(|share| {
        share
            .lower_a()
            .iter()
            .chain(share.upper_a().iter())
            .chain(share.lower_b().iter())
            .chain(share.upper_b().iter())
            .chain(share.lower_a_err().into_iter().flatten())
            .chain(share.upper_a_err().into_iter().flatten())
            .all(|v| v.is_finite())
    })
}

/// CONTAINMENT, exactly: on random attention-shaped blocks the GEMM-shaped
/// kernel's published relation — the SUM of the two operand shares, which is
/// what production composes (`backward_dispatch` moves both halved constants
/// into one bias channel; a share alone is not an enclosure,
/// #concretize-vacuity) — satisfies, for every sampled operand point `(x, y)`
/// in the boxes (random vertices and interior points),
///
/// ```text
/// lower_A(x) + lower_B(y) <= Σ_c L[r,c]·(s·x@y)_c + lower_bias[r]
/// upper_A(x) + upper_B(y) >= Σ_c U[r,c]·(s·x@y)_c + upper_bias[r]
/// ```
///
/// with no floating tolerance anywhere, for both arms (α parameters drawn
/// outside `[0, 1]` too, so the clamp is on the path), including a vit-sized
/// Q·Kᵀ block.
#[test]
fn bilinear_gemm_contains_exact_product_on_random_attention_blocks() {
    let mut rng = SplitMix(0xC0DE_A77E_2026_0929);
    let blocks = [
        // Q·Kᵀ-shaped, attention scale.
        Block {
            transpose_b: true,
            batch: 2,
            m: 3,
            n: 3,
            k: 4,
            scale: Some(0.5),
            rows: 3,
            samples: 8,
        },
        // attn·V-shaped.
        Block {
            transpose_b: false,
            batch: 2,
            m: 3,
            n: 4,
            k: 3,
            scale: None,
            rows: 3,
            samples: 8,
        },
        // Non-power-of-two scale (the `w = fl32(coefficient · s)` rounding).
        Block {
            transpose_b: true,
            batch: 1,
            m: 4,
            n: 4,
            k: 2,
            scale: Some(0.37),
            rows: 9,
            samples: 6,
        },
        Block {
            transpose_b: false,
            batch: 3,
            m: 2,
            n: 2,
            k: 5,
            scale: Some(1.5),
            rows: 2,
            samples: 8,
        },
        // vit_2023 ibp Q·Kᵀ, full size (few rows and points: the exact
        // rational reference is 13,872 products per row).
        Block {
            transpose_b: true,
            batch: 3,
            m: 17,
            n: 17,
            k: 16,
            scale: Some(0.25),
            rows: 1,
            samples: 2,
        },
    ];
    let schedule = GemmSchedule {
        chunk_rows: 8,
        parallel: true,
    };
    let mut points_checked = 0_usize;
    let mut expected_points = 0_usize;
    for (block_idx, block) in blocks.iter().enumerate() {
        for rep in 0..2 {
            let a_shape = block.a_shape();
            let b_shape = block.b_shape();
            let shape = shape_for(block.transpose_b, &a_shape, &b_shape);
            let total_c = block.batch * block.m * block.n;
            assert_eq!(total_c, shape.batch_size * shape.c_size_per_batch);
            let a_box = random_box(&mut rng, &a_shape);
            let b_box = random_box(&mut rng, &b_shape);
            let incoming = random_bounds(&mut rng, block.rows, total_c, Weights::Ordinary);
            let alphas =
                Array4::from_shape_fn((4, block.m, block.n, block.k), |_| match rng.pick(6) {
                    0 => 0.0,
                    1 => 1.0,
                    _ => rng.range(-0.25, 1.25),
                });
            let raw_scale = block.scale.unwrap_or(1.0);
            let midpoint = MatMulLayer::new(block.transpose_b, block.scale)
                .propagate_linear_binary_gemm(
                    &incoming,
                    &a_box,
                    &b_box,
                    &shape,
                    raw_scale,
                    Some(schedule),
                )
                .unwrap()
                .expect("the GEMM-shaped midpoint kernel must accept this block");
            let alpha = BilinearCrownLayer::new(block.transpose_b, block.scale)
                .propagate_linear_binary_with_alpha_gemm(
                    &incoming,
                    &a_box,
                    &b_box,
                    &alphas,
                    &shape,
                    raw_scale,
                    Some(schedule),
                )
                .unwrap()
                .expect("the GEMM-shaped alpha kernel must accept this block");
            let arms = [("midpoint", &midpoint), ("alpha", &alpha)];
            for (arm, pair) in arms {
                assert!(
                    pair_is_finite(pair),
                    "block {block_idx} rep {rep} ({arm} arm): a relation over finite boxes \
                     published a non-finite number, so containment would pass vacuously"
                );
            }

            let a_lo: Vec<f32> = a_box.lower().iter().copied().collect();
            let a_hi: Vec<f32> = a_box.upper().iter().copied().collect();
            let b_lo: Vec<f32> = b_box.lower().iter().copied().collect();
            let b_hi: Vec<f32> = b_box.upper().iter().copied().collect();
            for sample in 0..block.samples {
                let vertex = sample % 2 == 0;
                let mut draw = |lo: &[f32], hi: &[f32]| -> Vec<f32> {
                    lo.iter()
                        .zip(hi)
                        .map(|(&l, &h)| {
                            if vertex {
                                if rng.pick(2) == 0 {
                                    l
                                } else {
                                    h
                                }
                            } else {
                                inside(l, h, rng.unit())
                            }
                        })
                        .collect()
                };
                let a_pt = draw(&a_lo, &a_hi);
                let b_pt = draw(&b_lo, &b_hi);
                for row in 0..block.rows {
                    let lower_row: Vec<f32> = incoming.lower_a().row(row).to_vec();
                    let upper_row: Vec<f32> = incoming.upper_a().row(row).to_vec();
                    let exact_lower =
                        exact_target(block, &lower_row, incoming.lower_b()[row], &a_pt, &b_pt);
                    let exact_upper =
                        exact_target(block, &upper_row, incoming.upper_b()[row], &a_pt, &b_pt);
                    for (arm, pair) in arms {
                        let (a_lower, a_upper) = share_at(&pair.0, row, &a_pt);
                        let (b_lower, b_upper) = share_at(&pair.1, row, &b_pt);
                        let lower = a_lower + b_lower;
                        let upper = a_upper + b_upper;
                        assert!(
                            lower <= exact_lower,
                            "block {block_idx} rep {rep} sample {sample} row {row} ({arm} arm, \
                             {block:?}): the published LOWER relation exceeds the exact \
                             C·(s·A@B) + bias — a false-proof mechanism"
                        );
                        assert!(
                            upper >= exact_upper,
                            "block {block_idx} rep {rep} sample {sample} row {row} ({arm} arm, \
                             {block:?}): the published UPPER relation is below the exact \
                             C·(s·A@B) + bias"
                        );
                        points_checked += 1;
                    }
                }
            }
            expected_points += block.samples * block.rows * 2;
        }
    }
    assert_eq!(
        points_checked, expected_points,
        "every (block, rep, sample, row, arm) must be checked"
    );
    assert!(points_checked > 0);
}
