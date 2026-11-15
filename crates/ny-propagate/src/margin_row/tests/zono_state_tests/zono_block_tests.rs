// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#fd-block-eval` (ENGINE_PROGRAM M1) tests: the block evaluator is
//! BIT-IDENTICAL to `eval_node`, and the block search walks the reference's
//! tree. A child of `zono_state_tests` (its fixtures, hostile multipliers,
//! cancellation state and exact-rational node value) and a sibling of
//! `zono_bab_tests` (its sweep, problem and option helpers).
//!
//! Bit identity (the whole soundness argument of M1: no new certificate
//! arithmetic, so equality with the reference IS the proof obligation):
//!
//! * `fd_block_eval_bit_identical_on_fixtures` / `_on_synthetic_shapes` —
//!   every node's `value`, `certified`, `charge`, `rc` row and subgradient
//!   against `eval_node`, over depths, block counts, geometries and seven
//!   multiplier flavours (refusing nodes mixed into every call);
//! * `fd_block_sum_seed_matches_iterator_sum` — the fold seed and the
//!   carried-across-tiles fold, including signed-zero-only problems;
//! * `fd_block_epilogue_refactor_is_bit_identical` — the refactored
//!   `eval_node_inner` against a frozen verbatim copy of the pre-M1 body;
//! * `fd_block_axpy_rows_matches_the_per_node_direction` — the blocked `da`;
//! * `fd_block_search_nodes_match_the_node_path` — `eval_search_block` against
//!   `eval_search_node`, node by node (bound, winner, children's warm start).
//!
//! The tree: `fd_block_search_equals_node_search` (whole report, incl. node
//! caps and memory stops), `fd_block_far_first_preserves_the_tree`,
//! `fd_block_search_is_thread_count_independent`.
//!
//! Pins: `fd_block_charge_is_load_bearing` (mutation),
//! `fd_block_far_probe_recomputed_at_clamped_point` (U4b),
//! `fd_prune_predicate_is_strict_and_unique`, `fd_block_has_no_fma`,
//! `fd_block_crosscheck_catches_injected_fault`,
//! `fd_block_crosscheck_sample_is_spread`,
//! `fd_block_level_bytes_covers_block_scratch`,
//! `fd_block_is_the_default_and_keyless`; and the integrator's
//! `fd_block_rate_probe` (behind the `benchmarks` feature).

use super::zono_bab_tests::{open_problem, opts, problem_of, raw_positive_state, sweep};
use super::*;

use crate::margin_row::zono_bab::{
    block_work_bytes, crosscheck_due, exercised_block_work_bytes, level_bytes, probe_nodes, search,
    search_with_test_hooks, Candidate, Evaluator, LevelTrace, NodeProbe, OpenReason, SearchOptions,
    SearchReport, SearchVerdict, FROZEN_DUAL_CROSSCHECK_EVERY, FROZEN_DUAL_DEFAULT_EVALUATOR,
};
#[cfg(feature = "benchmarks")]
use crate::margin_row::zono_bab::{phase_probe, rate_probe};
use crate::margin_row::zono_state::problem::block::{
    sum_seed, BlockGeom, BlockScratch, BlockWant, BLOCK_NODES,
};
use crate::margin_row::zono_state::NodeBound;

const B: usize = BLOCK_NODES;

const WANT_ALL: BlockWant = BlockWant {
    rc: true,
    subgrad: true,
};
const WANT_NONE: BlockWant = BlockWant {
    rc: false,
    subgrad: false,
};
const WANTS: [BlockWant; 4] = [
    WANT_ALL,
    WANT_NONE,
    BlockWant {
        rc: true,
        subgrad: false,
    },
    BlockWant {
        rc: false,
        subgrad: true,
    },
];

fn nb_bits(nb: &NodeBound) -> (u64, u64, u64) {
    (
        nb.value.to_bits(),
        nb.certified.to_bits(),
        nb.charge.to_bits(),
    )
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn random_sides(rng: &mut StdRng, d: usize) -> Vec<Side> {
    (0..d)
        .map(|_| {
            if rng.random_range(0..2u32) == 0 {
                Side::Off
            } else {
                Side::On
            }
        })
        .collect()
}

/// Multiplier flavours of the bit-identity sweeps (one per node).
#[derive(Clone, Copy, Debug)]
enum Flavour {
    Hostile,
    F32RoundTrip,
    Zeros,
    NegZero,
    AllNan,
    Negative,
    PosInf,
}

const FLAVOURS: [Flavour; 7] = [
    Flavour::Hostile,
    Flavour::F32RoundTrip,
    Flavour::Zeros,
    Flavour::NegZero,
    Flavour::AllNan,
    Flavour::Negative,
    Flavour::PosInf,
];

fn f32_round_trip(rng: &mut StdRng, hi: f64) -> f64 {
    let v: f64 = rng.random_range(0.0..hi);
    #[allow(clippy::cast_possible_truncation)]
    let w = v as f32;
    f64::from(w)
}

fn flavoured(rng: &mut StdRng, flavour: Flavour, d: usize) -> (Vec<f64>, Vec<f64>) {
    match flavour {
        Flavour::Hostile => hostile_multipliers(rng, d),
        Flavour::F32RoundTrip => {
            let a: Vec<f64> = (0..d).map(|_| f32_round_trip(rng, 4.0)).collect();
            let b: Vec<f64> = (0..d).map(|_| f32_round_trip(rng, 4.0)).collect();
            (a, b)
        }
        Flavour::Zeros => (vec![0.0; d], vec![0.0; d]),
        Flavour::NegZero => (vec![-0.0; d], vec![-0.0; d]),
        Flavour::AllNan => (vec![f64::NAN; d], vec![f64::NAN; d]),
        Flavour::Negative => {
            let a: Vec<f64> = (0..d)
                .map(|_| {
                    let v: f64 = rng.random_range(0.0..3.0);
                    -v
                })
                .collect();
            (a, vec![-1.5; d])
        }
        Flavour::PosInf => {
            let (mut a, b) = hostile_multipliers(rng, d);
            if d > 0 {
                let k = rng.random_range(0..d);
                a[k] = f64::INFINITY;
            }
            (a, b)
        }
    }
}

/// `count` nodes of `depth` splits, row-major, each with its own flavour
/// (rotating through all seven from a random offset, so every call mixes
/// refusing and finite nodes).
fn mixed_batch(rng: &mut StdRng, count: usize, depth: usize) -> (Vec<Side>, Vec<f64>, Vec<f64>) {
    let offset = rng.random_range(0..FLAVOURS.len());
    let mut sides = Vec::with_capacity(count * depth);
    let mut l0 = Vec::with_capacity(count * depth);
    let mut l1 = Vec::with_capacity(count * depth);
    for b in 0..count {
        sides.extend(random_sides(rng, depth));
        let (a, c) = flavoured(rng, FLAVOURS[(offset + b) % FLAVOURS.len()], depth);
        l0.extend(a);
        l1.extend(c);
    }
    (sides, l0, l1)
}

/// What the lane feeds the evaluator: widened `f32` search state,
/// non-negative and finite, a quarter exact zeros.
fn searchlike_batch(
    rng: &mut StdRng,
    count: usize,
    depth: usize,
) -> (Vec<Side>, Vec<f64>, Vec<f64>) {
    let sides = random_sides(rng, count * depth);
    let draw = |rng: &mut StdRng| {
        if rng.random_range(0..4u32) == 0 {
            0.0
        } else {
            f32_round_trip(rng, 2.0)
        }
    };
    let l0: Vec<f64> = (0..count * depth).map(|_| draw(rng)).collect();
    let l1: Vec<f64> = (0..count * depth).map(|_| draw(rng)).collect();
    (sides, l0, l1)
}

/// True iff `eval_node` refuses these multipliers in its prologue (a clamped
/// multiplier is non-finite: only `+inf` survives the clamp).
fn prologue_refuses(l0: &[f64], l1: &[f64]) -> bool {
    l0.iter().chain(l1).any(|v| !v.max(0.0).is_finite())
}

fn geoms(n: usize) -> [BlockGeom; 4] {
    [
        BlockGeom { nodes: 1, cols: 1 },
        BlockGeom { nodes: 3, cols: 7 },
        BlockGeom {
            nodes: 16,
            cols: 256,
        },
        BlockGeom {
            nodes: B,
            cols: n + 3,
        },
    ]
}

/// Run ONE `eval_block` call and compare every node against `eval_node`:
/// `value`, `certified`, `charge` by bits; the `rc` row when the reference's
/// value is finite; the subgradient unless the node refused in its prologue.
/// Returns `(nodes compared, nodes refused)`.
#[allow(clippy::too_many_arguments)]
fn assert_block_matches(
    prob: &FrozenDualProblem,
    depth: usize,
    count: usize,
    sides: &[Side],
    l0: &[f64],
    l1: &[f64],
    geom: BlockGeom,
    want: BlockWant,
    ws: &mut BlockScratch,
    what: &str,
) -> (usize, usize) {
    let n = prob.n();
    // Sentinels, so a node the block path forgot cannot pass.
    let sentinel = NodeBound {
        value: 7.0,
        certified: 7.0,
        charge: 7.0,
    };
    let mut out = vec![sentinel; count];
    let mut s0 = vec![f64::NAN; count * depth];
    let mut s1 = vec![f64::NAN; count * depth];
    prob.eval_block(
        depth, sides, l0, l1, geom, want, ws, &mut s0, &mut s1, &mut out,
    );
    let mut scratch = NodeScratch::default();
    let mut refused = 0usize;
    for b in 0..count {
        let r = b * depth..(b + 1) * depth;
        let mut r0 = vec![f64::NAN; depth];
        let mut r1 = vec![f64::NAN; depth];
        let reference = prob.eval_node(
            &sides[r.clone()],
            &l0[r.clone()],
            &l1[r.clone()],
            &mut scratch,
            Some(Subgradient {
                s0: &mut r0,
                s1: &mut r1,
            }),
        );
        assert_eq!(
            nb_bits(&out[b]),
            nb_bits(&reference),
            "{what}: node {b} of {count} (depth {depth}, {geom:?}, {want:?}): block {:?} vs \
             eval_node {reference:?}",
            out[b]
        );
        if !reference.value.is_finite() {
            refused += 1;
        }
        if want.rc && reference.value.is_finite() {
            assert_eq!(
                bits(ws.rc_row(b, n)),
                bits(scratch.rc()),
                "{what}: node {b} of {count} (depth {depth}, {geom:?}): the rc row differs"
            );
        }
        if want.subgrad && !prologue_refuses(&l0[r.clone()], &l1[r.clone()]) {
            assert_eq!(
                bits(&s0[r.clone()]),
                bits(&r0),
                "{what}: node {b} of {count} (depth {depth}, {geom:?}): s0 differs"
            );
            assert_eq!(
                bits(&s1[r]),
                bits(&r1),
                "{what}: node {b} of {count} (depth {depth}, {geom:?}): s1 differs"
            );
        }
    }
    (count, refused)
}

// ---------------------------------------------------------------------------
// Bit identity of the block evaluator
// ---------------------------------------------------------------------------

#[test]
fn fd_block_eval_bit_identical_on_fixtures() {
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(8));
    let counts = [1, B - 1, B, B + 1, 3 * B + 5];
    let mut ws = BlockScratch::default();
    let (mut nodes, mut refused, mut calls) = (0usize, 0usize, 0usize);
    let mut max_depth_seen = 0usize;
    for f in &fxs {
        let frame = frame_of(f);
        for &j in f.adv.iter().take(2) {
            let st = state_of(f, &frame, j);
            let prob = problem_of(&st);
            let max = prob.max_depth();
            let n = prob.n();
            let g = geoms(n);
            let mut depths: Vec<usize> = [0, 1, 2, 3, 7, max]
                .into_iter()
                .filter(|&d| d <= max)
                .collect();
            depths.sort_unstable();
            depths.dedup();
            let mut rng = StdRng::seed_from_u64(0xB10C ^ ((j as u64) << 7) ^ n as u64);
            let what = format!("{} class {j}", f.name);
            for &depth in &depths {
                max_depth_seen = max_depth_seen.max(depth);
                for (ci, &count) in counts.iter().enumerate() {
                    // Every (count, geometry, want) combination recurs across
                    // the fixtures; every call is also repeated on the lane
                    // geometry with every output requested.
                    let k = ci + depth;
                    for (geom, want) in
                        [(g[k % 4], WANTS[(k / 4) % 4]), (BlockGeom::LANE, WANT_ALL)]
                    {
                        let (sides, l0, l1) = mixed_batch(&mut rng, count, depth);
                        let (c, r) = assert_block_matches(
                            &prob, depth, count, &sides, &l0, &l1, geom, want, &mut ws, &what,
                        );
                        nodes += c;
                        refused += r;
                        calls += 1;
                    }
                }
            }
            // Whole-call refusals (the `eval_batch` rule): a depth beyond the
            // order, and a length mismatch. A wrong subgradient length only
            // suppresses the subgradient.
            let deep = max + 1;
            let (sides, l0, l1) = mixed_batch(&mut rng, 3, deep);
            let mut out = vec![NodeBound::refused(); 3];
            prob.eval_block(
                deep,
                &sides,
                &l0,
                &l1,
                BlockGeom::LANE,
                WANT_ALL,
                &mut ws,
                &mut [],
                &mut [],
                &mut out,
            );
            assert!(
                out.iter().all(|nb| nb.certified == f64::NEG_INFINITY),
                "{what}: a too-deep call must refuse every node"
            );
            if max > 0 {
                let (sides, l0, l1) = mixed_batch(&mut rng, 3, max);
                let mut out = vec![
                    NodeBound {
                        value: 1.0,
                        certified: 1.0,
                        charge: 0.0
                    };
                    3
                ];
                prob.eval_block(
                    max,
                    &sides[1..],
                    &l0,
                    &l1,
                    BlockGeom::LANE,
                    WANT_NONE,
                    &mut ws,
                    &mut [],
                    &mut [],
                    &mut out,
                );
                assert!(
                    out.iter().all(|nb| nb.certified == f64::NEG_INFINITY),
                    "{what}: a length mismatch must refuse every node"
                );
                let (sides, l0, l1) = searchlike_batch(&mut rng, 2, max);
                let mut out = vec![NodeBound::refused(); 2];
                let mut short0 = vec![f64::NAN; 1];
                let mut short1 = vec![f64::NAN; 1];
                prob.eval_block(
                    max,
                    &sides,
                    &l0,
                    &l1,
                    BlockGeom::LANE,
                    WANT_ALL,
                    &mut ws,
                    &mut short0,
                    &mut short1,
                    &mut out,
                );
                let mut scratch = NodeScratch::default();
                for b in 0..2 {
                    let r = b * max..(b + 1) * max;
                    let reference = prob.eval_node(
                        &sides[r.clone()],
                        &l0[r.clone()],
                        &l1[r],
                        &mut scratch,
                        None,
                    );
                    assert_eq!(
                        nb_bits(&out[b]),
                        nb_bits(&reference),
                        "{what}: short subgradient"
                    );
                }
                assert!(
                    short0[0].is_nan() && short1[0].is_nan(),
                    "{what}: a wrong subgradient length must leave it unwritten"
                );
            }
        }
    }
    eprintln!(
        "[fd-block-bits] fixtures={} calls={calls} nodes={nodes} refused={refused} \
         max_depth={max_depth_seen}",
        fxs.len()
    );
    assert!(
        nodes > 5_000 && refused > 0 && refused < nodes && max_depth_seen >= 7,
        "VOID: too few nodes ({nodes}), no refusal mixed in ({refused}), or only shallow \
         depths ({max_depth_seen})"
    );
}

#[test]
fn fd_block_eval_bit_identical_on_synthetic_shapes() {
    let mut ws = BlockScratch::default();
    let mut nodes = 0usize;
    let shapes: [(usize, &[usize], &[usize]); 6] = [
        (1, &[1], &[1, B + 1]),
        (255, &[1, 7, 28, 64], &[B - 1, B + 1]),
        (256, &[7, 64], &[B, 3 * B + 5]),
        (257, &[7, 64], &[1, B + 1]),
        (3448, &[45, 64], &[B + 1]),
        (11418, &[28, 64], &[B + 1]),
    ];
    for (n, depths, counts) in shapes {
        for &depth in depths {
            let degenerates: &[bool] = if n <= 3448 { &[false, true] } else { &[false] };
            for &degenerate in degenerates {
                let seed = 0x5A7 ^ (n as u64) << 8 ^ depth as u64;
                let prob = if degenerate {
                    FrozenDualProblem::synthetic_degenerate(n, depth, seed)
                } else {
                    FrozenDualProblem::synthetic(n, depth, seed)
                };
                let d = prob.max_depth();
                assert_eq!(d, depth.min(n), "synthetic depth");
                let mut rng = StdRng::seed_from_u64(seed);
                let what = format!("synthetic n={n} depth={d} degenerate={degenerate}");
                for &count in counts {
                    let mut gs = vec![
                        BlockGeom::LANE,
                        BlockGeom {
                            nodes: B,
                            cols: n + 3,
                        },
                    ];
                    if n < 3448 {
                        gs.push(BlockGeom { nodes: 3, cols: 7 });
                    }
                    for geom in gs {
                        let (sides, l0, l1) = mixed_batch(&mut rng, count, d);
                        let (c, _) = assert_block_matches(
                            &prob, d, count, &sides, &l0, &l1, geom, WANT_ALL, &mut ws, &what,
                        );
                        nodes += c;
                    }
                }
            }
        }
    }
    assert!(nodes > 200, "VOID: only {nodes} synthetic nodes compared");
}

#[test]
fn fd_block_sum_seed_matches_iterator_sum() {
    let empty: [f64; 0] = [];
    let seed = sum_seed();
    assert_eq!(
        seed.to_bits(),
        empty.iter().sum::<f64>().to_bits(),
        "the block folds' seed is not Iterator::sum's seed on this toolchain"
    );
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let random: Vec<f64> = (0..1000)
        .map(|k| match k % 5 {
            0 => -0.0,
            1 => 0.0,
            _ => rng.random_range(-1e3..1e3),
        })
        .collect();
    let cases: Vec<Vec<f64>> = vec![
        vec![],
        vec![-0.0; 7],
        vec![0.0, -0.0, -0.0],
        vec![-0.0, 0.0],
        vec![1.0, -1.0, 1e-300, -0.0, 3.5, -2.25, -1e-300],
        random,
    ];
    for xs in &cases {
        let want = xs.iter().sum::<f64>().to_bits();
        for t in [1usize, 3, 7, 256, 4096] {
            let mut acc = seed;
            for chunk in xs.chunks(t) {
                for &x in chunk {
                    acc += x;
                }
            }
            assert_eq!(
                acc.to_bits(),
                want,
                "the fold carried across tiles of {t} differs from .sum() on {xs:?}"
            );
        }
    }
    // Through the evaluator: problems whose every row, cut and constant is
    // zero and every margin coefficient -0.0, so the folds see only signed
    // zeros and a wrong seed would flip the sign of a zero subgradient.
    let mut ws = BlockScratch::default();
    let mut nodes = 0usize;
    for n in [1usize, 5, 300] {
        for depth in [0usize, 1, 4] {
            let prob = FrozenDualProblem::synthetic_degenerate(n, depth, 0xD0 + n as u64);
            let d = prob.max_depth();
            for count in [1usize, B + 2] {
                let sides = random_sides(&mut rng, count * d);
                for fill in [0.0f64, -0.0] {
                    let l = vec![fill; count * d];
                    for geom in [BlockGeom::LANE, BlockGeom { nodes: 1, cols: 1 }] {
                        let (c, _) = assert_block_matches(
                            &prob,
                            d,
                            count,
                            &sides,
                            &l,
                            &l,
                            geom,
                            WANT_ALL,
                            &mut ws,
                            &format!("degenerate n={n} depth={d} fill={fill:?}"),
                        );
                        nodes += c;
                    }
                }
            }
        }
    }
    assert!(nodes > 0, "VOID");
}

#[test]
fn fd_block_epilogue_refactor_is_bit_identical() {
    let mut probs: Vec<(String, FrozenDualProblem)> = Vec::new();
    for f in fixtures() {
        let frame = frame_of(&f);
        for &j in f.adv.iter().take(2) {
            probs.push((
                format!("{} class {j}", f.name),
                problem_of(&state_of(&f, &frame, j)),
            ));
        }
    }
    for seed in 0..2u64 {
        let st = cancellation_state(0xCA0 + seed);
        let order: Vec<usize> = (0..st.num_unstable()).collect();
        probs.push((
            format!("cancellation {seed}"),
            FrozenDualProblem::new(&st, &order).expect("problem"),
        ));
    }
    probs.push((
        "synthetic".to_string(),
        FrozenDualProblem::synthetic(257, 9, 3),
    ));
    probs.push((
        "degenerate".to_string(),
        FrozenDualProblem::synthetic_degenerate(40, 6, 4),
    ));
    let mut checked = 0usize;
    for (name, prob) in &probs {
        let mut rng = StdRng::seed_from_u64(0xE91 ^ prob.n() as u64);
        let mut new_scratch = NodeScratch::default();
        let mut old_scratch = NodeScratch::default();
        for k in 0..80 {
            let d = rng.random_range(0..=prob.max_depth());
            let sides = random_sides(&mut rng, d);
            let (l0, l1) = flavoured(&mut rng, FLAVOURS[k % FLAVOURS.len()], d);
            for charge_on in [true, false] {
                for with_sub in [true, false] {
                    let mut n0 = vec![f64::NAN; d];
                    let mut n1 = vec![f64::NAN; d];
                    let mut o0 = vec![f64::NAN; d];
                    let mut o1 = vec![f64::NAN; d];
                    let sg_new = if with_sub {
                        Some(Subgradient {
                            s0: &mut n0,
                            s1: &mut n1,
                        })
                    } else {
                        None
                    };
                    let new =
                        prob.eval_node_inner(&sides, &l0, &l1, &mut new_scratch, sg_new, charge_on);
                    let sg_old = if with_sub {
                        Some(Subgradient {
                            s0: &mut o0,
                            s1: &mut o1,
                        })
                    } else {
                        None
                    };
                    let old = prob.eval_node_pre_m1(
                        &sides,
                        &l0,
                        &l1,
                        &mut old_scratch,
                        sg_old,
                        charge_on,
                    );
                    assert_eq!(
                        nb_bits(&new),
                        nb_bits(&old),
                        "{name}: node {k} (charge {charge_on}): refactored {new:?} vs pre-M1 {old:?}"
                    );
                    if new.value.is_finite() {
                        assert_eq!(
                            bits(new_scratch.rc()),
                            bits(old_scratch.rc()),
                            "{name}: node {k}: rc differs"
                        );
                    }
                    if with_sub && !prologue_refuses(&l0, &l1) {
                        assert_eq!(bits(&n0), bits(&o0), "{name}: node {k}: s0 differs");
                        assert_eq!(bits(&n1), bits(&o1), "{name}: node {k}: s1 differs");
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1_000, "VOID: only {checked} nodes compared");
}

#[test]
fn fd_block_axpy_rows_matches_the_per_node_direction() {
    let mut ws = BlockScratch::default();
    let mut compared = 0usize;
    for (n, depth) in [(1usize, 1usize), (37, 5), (300, 28), (1000, 3)] {
        let prob = FrozenDualProblem::synthetic(n, depth, 0xDA ^ n as u64);
        let d = prob.max_depth();
        let mut rng = StdRng::seed_from_u64(0xDA0 + n as u64);
        for count in [1usize, 7, B, B + 5] {
            for geom in [BlockGeom::LANE, BlockGeom { nodes: 3, cols: 7 }] {
                // Multipliers with exact zeros of both signs (the per-node loop
                // skips them) and ordinary values.
                let mult: Vec<f64> = (0..count * d)
                    .map(|_| match rng.random_range(0..4u32) {
                        0 => 0.0,
                        1 => -0.0,
                        _ => rng.random_range(-5.0..5.0),
                    })
                    .collect();
                let mut out = vec![f64::NAN; count * n];
                assert!(prob.block_axpy_rows(d, count, &mult, geom, &mut ws, &mut out));
                for b in 0..count {
                    let mut v = vec![0.0f64; n];
                    for i in 0..d {
                        let m = mult[b * d + i];
                        if m != 0.0 {
                            for (acc, &a) in v.iter_mut().zip(prob.split_row(i).a) {
                                *acc += m * a;
                            }
                        }
                    }
                    assert_eq!(
                        bits(&out[b * n..(b + 1) * n]),
                        bits(&v),
                        "n={n} depth={d} count={count} {geom:?}: node {b}'s direction differs"
                    );
                    compared += 1;
                }
            }
        }
        // Shape mismatches write nothing and report it.
        let mut out = vec![f64::NAN; n];
        assert!(!prob.block_axpy_rows(d, 1, &vec![1.0; d + 1], BlockGeom::LANE, &mut ws, &mut out));
        assert!(out.iter().all(|v| v.is_nan()));
    }
    assert!(compared > 50, "VOID");
}

/// The search-visible part of one node's evaluation, by bits.
fn out_bits(p: &NodeProbe) -> (u64, u64, Candidate, bool, Vec<u64>, Vec<u64>) {
    (
        p.best.to_bits(),
        p.certified.to_bits(),
        p.winner,
        p.far_used,
        bits(&p.next0),
        bits(&p.next1),
    )
}

#[test]
fn fd_block_search_nodes_match_the_node_path() {
    let (mut compared, mut pruned, mut refused) = (0usize, 0usize, 0usize);
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(5));
    for f in &fxs {
        let frame = frame_of(f);
        for &j in f.adv.iter().take(2) {
            let prob = problem_of(&state_of(f, &frame, j));
            let max = prob.max_depth();
            if max == 0 {
                continue;
            }
            let mut rng = StdRng::seed_from_u64(0x5EA ^ (j as u64) << 9 ^ prob.n() as u64);
            let mut depths = vec![1, 2, max.min(5), max.min(12)];
            depths.sort_unstable();
            depths.dedup();
            for depth in depths {
                let count = 3 * B + 5;
                for (sides, l0, l1) in [
                    mixed_batch(&mut rng, count, depth),
                    searchlike_batch(&mut rng, count, depth),
                ] {
                    let node = probe_nodes(&prob, depth, &sides, &l0, &l1, Evaluator::Node);
                    let block = probe_nodes(
                        &prob,
                        depth,
                        &sides,
                        &l0,
                        &l1,
                        Evaluator::Block { far_first: false },
                    );
                    let first = probe_nodes(
                        &prob,
                        depth,
                        &sides,
                        &l0,
                        &l1,
                        Evaluator::Block { far_first: true },
                    );
                    for b in 0..count {
                        let (x, y, z) = (&node[b], &block[b], &first[b]);
                        let what = format!("{} class {j} depth {depth} node {b}", f.name);
                        assert_eq!(out_bits(x), out_bits(y), "{what}: block vs node path");
                        if x.certified > 0.0 {
                            pruned += 1;
                            assert!(
                                z.certified > 0.0 && z.certified <= x.certified,
                                "{what}: far-first certified {:e} outside (0, {:e}]",
                                z.certified,
                                x.certified
                            );
                        } else {
                            assert_eq!(
                                out_bits(x),
                                out_bits(z),
                                "{what}: far-first changed a node it did not prune"
                            );
                        }
                        if x.certified == f64::NEG_INFINITY {
                            refused += 1;
                        }
                        compared += 1;
                    }
                }
            }
        }
    }
    eprintln!("[fd-block-nodes] compared={compared} pruned={pruned} refused={refused}");
    assert!(
        compared > 1_000 && pruned > 0 && refused > 0,
        "VOID: compared={compared} pruned={pruned} refused={refused}"
    );
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

/// Everything far-first must preserve.
#[derive(Debug, PartialEq)]
struct Tree {
    verdict: SearchVerdict,
    root_certified: u64,
    nodes: u64,
    depth: usize,
    certified_prunes: u64,
    peak_frontier: u64,
    worst_open: u64,
    levels: Vec<LevelTrace>,
    pruned_sides: Vec<Vec<Side>>,
}

fn tree(r: &SearchReport) -> Tree {
    Tree {
        verdict: r.verdict,
        root_certified: r.root_certified.to_bits(),
        nodes: r.nodes,
        depth: r.depth,
        certified_prunes: r.certified_prunes,
        peak_frontier: r.peak_frontier,
        worst_open: r.worst_open.to_bits(),
        levels: r.levels.clone(),
        pruned_sides: r.pruned.iter().map(|p| p.sides.clone()).collect(),
    }
}

/// The whole report except the wall clock, the evaluator's name and the
/// block path's own cross-check counters.
#[derive(Debug, PartialEq)]
struct Full {
    tree: Tree,
    cert_min_slack: u64,
    far_used: u64,
    slack_hist: Vec<u64>,
    pruned: Vec<(Vec<u64>, Vec<u64>, u64, Candidate)>,
}

fn full(r: &SearchReport) -> Full {
    Full {
        tree: tree(r),
        cert_min_slack: r.cert_min_slack.to_bits(),
        far_used: r.far_used,
        slack_hist: r.slack_hist.to_vec(),
        pruned: r
            .pruned
            .iter()
            .map(|p| (bits(&p.l0), bits(&p.l1), p.certified.to_bits(), p.candidate))
            .collect(),
    }
}

/// The tree tests' problems: two classes of every stage-A fixture and of
/// four sweep nets spanning the box widths (0.3, 0.15, 0.08, 0.04), so some
/// close by splitting, some hit the node cap and every one prunes somewhere.
/// Debug-build test time scales with this list: keep it short.
fn tree_problems() -> Vec<(String, FrozenDualProblem)> {
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(5));
    let mut out = Vec::new();
    for f in &fxs {
        let frame = frame_of(f);
        for &j in f.adv.iter().take(2) {
            out.push((
                format!("{} class {j}", f.name),
                problem_of(&state_of(f, &frame, j)),
            ));
        }
    }
    out
}

#[test]
fn fd_block_search_equals_node_search() {
    let (mut compared, mut closed, mut with_prunes, mut capped, mut starved) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let block = Evaluator::Block { far_first: false };
    for (k, (name, prob)) in tree_problems().iter().enumerate() {
        // Admit level 1 exactly: level 2, if any, stops on memory.
        let tight = level_bytes(2, 1, prob.n());
        let mut bases = vec![opts(3_000, 64), opts(40, 64)];
        if k < 6 {
            bases.push(SearchOptions {
                max_frontier_bytes: tight,
                ..opts(3_000, 64)
            });
            bases.push(SearchOptions {
                max_frontier_bytes: 1,
                ..opts(3_000, 0)
            });
        }
        for base in bases {
            let a = search(
                prob,
                &SearchOptions {
                    evaluator: Evaluator::Node,
                    ..base
                },
            );
            let b = search(
                prob,
                &SearchOptions {
                    evaluator: block,
                    ..base
                },
            );
            assert_eq!((a.evaluator, b.evaluator), ("node", "block"), "VOID");
            assert_eq!(
                full(&a),
                full(&b),
                "{name}: the block search differs from the node search"
            );
            assert_eq!(b.crosscheck_failures, 0, "{name}");
            compared += 1;
            closed += usize::from(a.closed() && a.nodes > 0);
            with_prunes += usize::from(a.certified_prunes > 0);
            capped += usize::from(a.verdict == SearchVerdict::Open(OpenReason::NodeCap));
            starved += usize::from(a.verdict == SearchVerdict::Open(OpenReason::Memory));
        }
    }
    eprintln!(
        "[fd-block-search] compared={compared} closed_by_split={closed} with_prunes={with_prunes} \
         node_cap={capped} memory={starved}"
    );
    // Closure by splitting on the block path is asserted by the positive
    // control (`frozen_dual_bab_closed_classes_hold_at_sampled_points`, now on
    // the block default) and the ny-cli P1 replica tests; here it is reported.
    assert!(
        with_prunes > 0 && capped > 0 && starved > 0,
        "VOID: closed={closed} with_prunes={with_prunes} node_cap={capped} memory={starved}"
    );
}

#[test]
fn fd_block_far_first_preserves_the_tree() {
    let (mut compared, mut with_prunes, mut moved) = (0usize, 0usize, 0usize);
    for (name, prob) in &tree_problems() {
        for base in [opts(3_000, 64), opts(40, 64)] {
            let a = search(
                prob,
                &SearchOptions {
                    evaluator: Evaluator::Node,
                    ..base
                },
            );
            let b = search(
                prob,
                &SearchOptions {
                    evaluator: Evaluator::Block { far_first: true },
                    ..base
                },
            );
            assert!(b.far_first, "VOID: far-first did not run");
            assert_eq!(tree(&a), tree(&b), "{name}: far-first changed the tree");
            if a.certified_prunes > 0 {
                with_prunes += 1;
                assert!(
                    b.cert_min_slack > 0.0 && b.cert_min_slack <= a.cert_min_slack,
                    "{name}: far-first cert_min_slack {:e} outside (0, {:e}]",
                    b.cert_min_slack,
                    a.cert_min_slack
                );
            } else {
                assert_eq!(a.cert_min_slack.to_bits(), b.cert_min_slack.to_bits());
            }
            for (x, y) in a.pruned.iter().zip(&b.pruned) {
                assert!(
                    y.certified > 0.0 && y.certified <= x.certified,
                    "{name}: a far-first record certified {:e} outside (0, {:e}]",
                    y.certified,
                    x.certified
                );
            }
            moved += usize::from(
                a.far_used != b.far_used
                    || a.cert_min_slack.to_bits() != b.cert_min_slack.to_bits(),
            );
            compared += 1;
        }
    }
    eprintln!("[fd-block-far-first] compared={compared} with_prunes={with_prunes} moved={moved}");
    assert!(with_prunes > 0, "VOID: no search pruned a node");
}

#[test]
fn fd_block_search_is_thread_count_independent() {
    let (name, prob) = open_problem();
    for evaluator in [
        Evaluator::Block { far_first: true },
        Evaluator::Block { far_first: false },
    ] {
        let run = |threads: usize, chunk: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            pool.install(|| {
                search(
                    &prob,
                    &SearchOptions {
                        chunk,
                        evaluator,
                        ..opts(3_000, 16)
                    },
                )
            })
        };
        let a = run(1, 128);
        assert!(a.nodes > 0, "VOID: {name} evaluated no node");
        for (threads, chunk) in [(3, 128), (8, 128), (8, 7), (3, 1)] {
            let b = run(threads, chunk);
            assert_eq!(
                full(&a),
                full(&b),
                "{name} {evaluator:?}: {threads} threads / chunk {chunk} changed the search"
            );
            assert_eq!(
                (a.crosschecks, a.crosscheck_failures),
                (b.crosschecks, b.crosscheck_failures),
                "{name} {evaluator:?}: the cross-check sample depends on the schedule"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Pins
// ---------------------------------------------------------------------------

#[test]
fn fd_block_charge_is_load_bearing() {
    // MUTATION PIN (mirrors `frozen_dual_bab_prunes_only_on_the_certified_value`):
    // every node of this state has a POSITIVE raw value and a NEGATIVE
    // certified value, so the real block search never closes it, and the
    // same search with the charge disabled does.
    let st = raw_positive_state(6);
    let order: Vec<usize> = (0..st.num_unstable()).collect();
    let prob = FrozenDualProblem::new(&st, &order).expect("problem");
    let base = SearchOptions {
        evaluator: Evaluator::Block { far_first: true },
        // The uncharged values would (rightly) fail the scalar cross-check.
        crosscheck_every: 0,
        ..opts(100_000, 0)
    };
    let real = search_with_test_hooks(&prob, &base, false, false);
    assert!(
        !real.closed(),
        "the block search closed a raw-positive state"
    );
    assert_eq!(
        real.certified_prunes, 0,
        "a node pruned without a certificate"
    );
    let bare = search_with_test_hooks(&prob, &base, true, false);
    assert!(
        bare.closed() && bare.certified_prunes > 0,
        "with the charge disabled the block search must close this state ({:?} after {} nodes): \
         otherwise the pin is VOID",
        bare.verdict,
        bare.nodes
    );
    // Node level, against exact rationals: the uncharged block evaluator
    // certifies above the exact node value on the cancellation fixture; the
    // charged one never does.
    let mut over = 0usize;
    let mut checked = 0usize;
    let mut ws = BlockScratch::default();
    for seed in 0..4u64 {
        let st = cancellation_state(0xCA0 + seed);
        let order: Vec<usize> = (0..st.num_unstable()).collect();
        let prob = FrozenDualProblem::new(&st, &order).expect("problem");
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..40 {
            let d = rng.random_range(0..=prob.max_depth());
            let count = 4usize;
            let sides = random_sides(&mut rng, count * d);
            let l0: Vec<f64> = (0..count * d).map(|_| rng.random_range(0.0..3.0)).collect();
            let l1: Vec<f64> = (0..count * d).map(|_| rng.random_range(0.0..3.0)).collect();
            let mut charged = vec![NodeBound::refused(); count];
            let mut bare = vec![NodeBound::refused(); count];
            for (out, charge_on) in [(&mut charged, true), (&mut bare, false)] {
                prob.eval_block_inner(
                    d,
                    &sides,
                    &l0,
                    &l1,
                    BlockGeom::LANE,
                    WANT_NONE,
                    &mut ws,
                    &mut [],
                    &mut [],
                    out,
                    charge_on,
                );
            }
            for b in 0..count {
                let r = b * d..(b + 1) * d;
                let exact = exact_node_value(&prob, &sides[r.clone()], &l0[r.clone()], &l1[r]);
                if charged[b].certified.is_finite() {
                    assert!(
                        q(charged[b].certified) <= exact,
                        "the charged block certificate exceeds the exact node value"
                    );
                }
                if bare[b].certified.is_finite() && q(bare[b].certified) > exact {
                    over += 1;
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "VOID");
    assert!(
        over > 0,
        "with the charge disabled no block node exceeded its exact value: the mutation pin is VOID"
    );
}

#[test]
fn fd_block_far_probe_recomputed_at_clamped_point() {
    // U4b PIN: the far point is an ordinary CLAMPED dual point, its rc
    // recomputed from scratch; never Vibecheck's incremental rc + eta da
    // (unclamped multipliers under the rc, clamped under the b-term). The test
    // finds nodes where the far step crosses a multiplier's zero, where the two
    // forms visibly differ, so a block path that took the incremental form
    // would fail the bit-equality below.
    let (mut checked_far, mut crossing) = (0usize, 0usize);
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(5));
    for f in &fxs {
        let frame = frame_of(f);
        for &j in f.adv.iter().take(2) {
            let prob = problem_of(&state_of(f, &frame, j));
            let max = prob.max_depth();
            if max == 0 {
                continue;
            }
            let mut rng = StdRng::seed_from_u64(0x04B ^ (j as u64) << 6 ^ prob.n() as u64);
            let mut depths = vec![1, 2, max.min(6), max.min(12)];
            depths.sort_unstable();
            depths.dedup();
            for depth in depths {
                let count = 3 * B;
                let sides = random_sides(&mut rng, count * depth);
                let l0: Vec<f64> = (0..count * depth)
                    .map(|_| rng.random_range(0.0..3.0))
                    .collect();
                let l1: Vec<f64> = (0..count * depth)
                    .map(|_| rng.random_range(0.0..3.0))
                    .collect();
                let probes = probe_nodes(
                    &prob,
                    depth,
                    &sides,
                    &l0,
                    &l1,
                    Evaluator::Block { far_first: false },
                );
                let mut scratch = NodeScratch::default();
                for (b, p) in probes.iter().enumerate() {
                    let Some(far) = p.far else { continue };
                    if !p.far_differs {
                        continue;
                    }
                    let r = b * depth..(b + 1) * depth;
                    let reference =
                        prob.eval_node(&sides[r.clone()], &p.f0, &p.f1, &mut scratch, None);
                    assert_eq!(
                        nb_bits(&far),
                        nb_bits(&reference),
                        "{} class {j}: the block far bound is not eval_node at the clamped far point",
                        f.name
                    );
                    checked_far += 1;
                    let crossed = (0..depth).any(|i| {
                        l0[r.start + i] + p.eta_c * p.sp0[i] < -1e-6
                            || l1[r.start + i] + p.eta_c * p.sp1[i] < -1e-6
                    });
                    if !crossed || !reference.value.is_finite() {
                        continue;
                    }
                    let incremental: Vec<f64> = p
                        .rc_warm
                        .iter()
                        .zip(&p.da)
                        .map(|(&rv, &dv)| rv + p.eta_c * dv)
                        .collect();
                    let recomputed = scratch.rc();
                    let gap: f64 = incremental
                        .iter()
                        .zip(recomputed)
                        .map(|(x, y)| (x - y).abs())
                        .sum();
                    let sum_inc: f64 = incremental.iter().map(|v| v.abs()).sum();
                    let sum_rec: f64 = recomputed.iter().map(|v| v.abs()).sum();
                    if gap > 1e-9 && (sum_inc - sum_rec).abs() > 1e-9 * (1.0 + sum_rec) {
                        crossing += 1;
                    }
                }
            }
        }
    }
    eprintln!("[fd-block-u4b] far_points={checked_far} visible_crossings={crossing}");
    assert!(checked_far > 0, "VOID: no far point moved");
    assert!(
        crossing > 0,
        "VOID: no far step crossed a multiplier's zero with a visible rc difference, so this test \
         could not tell the clamped far point from the incremental (U4b) one"
    );
}

#[test]
fn fd_prune_predicate_is_strict_and_unique() {
    let nb = |c: f64| NodeBound {
        value: 1.0,
        certified: c,
        charge: 0.0,
    };
    assert!(!nb(0.0).prunes(), "certified == 0 must not prune (strict)");
    assert!(!nb(-0.0).prunes());
    assert!(!nb(f64::NAN).prunes());
    assert!(!nb(-f64::MIN_POSITIVE).prunes());
    assert!(!NodeBound::refused().prunes());
    assert!(
        nb(f64::from_bits(1)).prunes(),
        "the smallest positive value prunes"
    );
    // Source ratchets: the search compares a certified value against zero in
    // exactly one place (`NodeOut::prunes`), strictly, and so does the problem
    // (`NodeBound::prunes`); nothing uses `>=`.
    let needle = concat!("certified", " > ", "0.0");
    let weak = concat!("certified", " >= ", "0");
    let bab = include_str!("../../zono_bab.rs");
    let hits: Vec<usize> = bab.match_indices(needle).map(|(i, _)| i).collect();
    assert_eq!(
        hits.len(),
        1,
        "zono_bab.rs must compare a certified value against zero in exactly one predicate"
    );
    let pred = bab
        .find("fn prunes(&self) -> bool {")
        .expect("NodeOut::prunes exists");
    assert!(
        hits[0] > pred && hits[0] - pred < 80,
        "the one `certified > 0.0` in zono_bab.rs is not the body of NodeOut::prunes"
    );
    let problem = include_str!("../../zono_state/problem.rs");
    assert_eq!(
        problem.matches(needle).count(),
        1,
        "problem.rs: NodeBound::prunes only"
    );
    let block = include_str!("../../zono_state/problem/block.rs");
    assert_eq!(
        block.matches(needle).count(),
        0,
        "block.rs decides no prune"
    );
    for (name, src) in [
        ("zono_bab.rs", bab),
        ("problem.rs", problem),
        ("block.rs", block),
    ] {
        assert!(
            !src.contains(weak),
            "{name}: a non-strict certified comparison"
        );
    }
}

#[test]
fn fd_block_has_no_fma() {
    // The bit-identity of the block form needs the reference's separate
    // multiply and add; Rust never fuses them, so only an explicit fused call
    // could. Needles built from pieces so this file does not match itself.
    let fused = [concat!("mul", "_add"), concat!("f", "ma")];
    assert!("x.mul_add(y, z)".contains(fused[0]), "VOID needle");
    assert!("vfmaq_f64".contains(fused[1]), "VOID needle");
    for (name, src) in [
        (
            "block.rs",
            include_str!("../../zono_state/problem/block.rs"),
        ),
        ("problem.rs", include_str!("../../zono_state/problem.rs")),
    ] {
        for needle in fused {
            assert!(
                !src.contains(needle),
                "{name} contains `{needle}`: the certified evaluator must stay unfused"
            );
        }
    }
}

/// The first class of the sweep the default block search closes by
/// splitting (the positive control guarantees one).
fn closing_problem() -> (String, FrozenDualProblem) {
    for f in &sweep() {
        let frame = frame_of(f);
        for &j in &f.adv {
            let prob = problem_of(&state_of(f, &frame, j));
            let r = search(&prob, &opts(6_000, 0));
            if r.closed() && r.nodes > 0 {
                return (format!("{} class {j}", f.name), prob);
            }
        }
    }
    panic!("VOID: no class of the sweep closes by splitting");
}

#[test]
fn fd_block_crosscheck_catches_injected_fault() {
    let (name, prob) = closing_problem();
    let every_prune = SearchOptions {
        crosscheck_every: 1,
        ..opts(6_000, 0)
    };
    let clean = search_with_test_hooks(&prob, &every_prune, false, false);
    assert!(
        clean.closed() && clean.crosschecks > 0 && clean.crosscheck_failures == 0,
        "{name}: the clean block search must close with every prune re-checked and none failing \
         ({:?}, {} checks, {} failures)",
        clean.verdict,
        clean.crosschecks,
        clean.crosscheck_failures
    );
    assert_eq!(
        clean.crosschecks, clean.certified_prunes,
        "{name}: every prune re-checked"
    );
    // One ulp added to every block output: the scalar re-check sees it.
    let faulted = search_with_test_hooks(&prob, &every_prune, false, true);
    assert_eq!(
        faulted.verdict,
        SearchVerdict::Open(OpenReason::Refused),
        "{name}: a cross-check mismatch must stop the search OPEN"
    );
    assert!(
        faulted.crosscheck_failures >= 1,
        "{name}: the fault went unseen"
    );
    assert!(!faulted.closed());
    // At the lane's sampling rate the shallow prunes are always checked.
    let lane = search_with_test_hooks(&prob, &opts(6_000, 0), false, true);
    if lane.crosschecks > 0 {
        assert_eq!(
            lane.verdict,
            SearchVerdict::Open(OpenReason::Refused),
            "{name}"
        );
    }
    // With the cross-check off the same fault is invisible: the check is what
    // catches it.
    let blind = search_with_test_hooks(
        &prob,
        &SearchOptions {
            crosscheck_every: 0,
            ..opts(6_000, 0)
        },
        false,
        true,
    );
    assert_eq!((blind.crosschecks, blind.crosscheck_failures), (0, 0));
}

#[test]
fn fd_block_crosscheck_sample_is_spread() {
    // G4's 1-in-`every` sample below depth 4 must reach BOTH children of a
    // split (a level lists each kept parent's `Off` child at an even index,
    // its `On` child at the next odd one), move with the level, and keep its
    // rate. The brief's bare `idx * K % every` fails all three at the
    // power-of-two default: it selects exactly `idx % 256 == 0`.
    let every = FROZEN_DUAL_CROSSCHECK_EVERY;
    let span = 1usize << 16;
    let (mut picked, mut odd) = (0usize, 0usize);
    for idx in 0..span {
        if crosscheck_due(every, 9, idx) {
            picked += 1;
            odd += idx & 1;
        }
    }
    let want = span / every as usize;
    assert!(
        picked > want / 2 && picked < want * 2,
        "rate: {picked} of {span} picked, want about {want}"
    );
    assert!(
        odd > picked / 4 && odd < picked * 3 / 4,
        "parity: {odd} odd of {picked}: the sample misses one side of every split"
    );
    // Levels narrower than `every` are not all pinned to index 0.
    let narrow: std::collections::BTreeSet<usize> = (5..200)
        .flat_map(|d| (0..every as usize).filter(move |&i| crosscheck_due(every, d, i)))
        .collect();
    assert!(
        narrow.len() > 16,
        "narrow levels sample only indices {narrow:?}"
    );
    // Depth <= 4 is always checked; `every = 0` never; `every = 1` always.
    for i in 0..64 {
        for d in 0..=4 {
            assert!(crosscheck_due(every, d, i));
        }
        assert!(!crosscheck_due(0, 1, i) && !crosscheck_due(0, 9, i));
        assert!(crosscheck_due(1, 9, i));
    }
}

#[test]
fn fd_block_level_bytes_covers_block_scratch() {
    let threads = rayon::current_num_threads().max(1) as u64;
    for (n, depth) in [
        (1usize, 1usize),
        (40, 7),
        (257, 28),
        (600, 512),
        (3448, 45),
        (11418, 28),
    ] {
        let prob = FrozenDualProblem::synthetic(n, depth, 0x1B ^ n as u64);
        let d = prob.max_depth();
        let measured = exercised_block_work_bytes(&prob, d, 0x77 ^ n as u64);
        let budget = block_work_bytes(n, d);
        eprintln!("[fd-block-bytes] n={n} depth={d} measured={measured} budget={budget}");
        assert!(
            measured > 0 && measured <= budget,
            "n={n} depth={d}: the block buffers reserve {measured} bytes, above the {budget}-byte \
             per-thread estimate the memory admission uses"
        );
        assert!(
            level_bytes(1, d, n) >= threads.saturating_mul(budget),
            "n={n} depth={d}: level_bytes does not charge every thread's block buffers"
        );
    }
}

#[test]
fn fd_block_is_the_default_and_keyless() {
    // M1 adds no key: the block evaluator is the search default (inside the
    // dark `margin_row.frozen_dual_bab` tier), with the lane sampling rate.
    let d = SearchOptions::default();
    assert_eq!(d.evaluator, Evaluator::Block { far_first: true });
    assert_eq!(FROZEN_DUAL_DEFAULT_EVALUATOR, d.evaluator);
    assert_eq!(d.crosscheck_every, FROZEN_DUAL_CROSSCHECK_EVERY);
    assert_eq!(FROZEN_DUAL_CROSSCHECK_EVERY, 256);
    let (name, prob) = open_problem();
    let r = search(&prob, &opts(500, 0));
    assert_eq!(r.evaluator, "block", "{name}");
    assert!(r.far_first, "{name}");
}

// ---------------------------------------------------------------------------
// The integrator's rate probe
// ---------------------------------------------------------------------------

/// `raw` with its margin constant moved so the median raw value of random
/// nodes (at the probe's multiplier distribution) sits at zero: about half
/// the probe's nodes then prune, as on the real rows.
#[cfg(feature = "benchmarks")]
fn centred(raw: &FrozenDualProblem, seed: u64) -> FrozenDualProblem {
    let d = raw.max_depth();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut scratch = NodeScratch::default();
    let mut rest: Vec<f64> = (0..256)
        .map(|_| {
            let sides = random_sides(&mut rng, d);
            let l0: Vec<f64> = (0..d).map(|_| rng.random_range(0.0..2.0)).collect();
            let l1: Vec<f64> = (0..d).map(|_| rng.random_range(0.0..2.0)).collect();
            raw.eval_node(&sides, &l0, &l1, &mut scratch, None).value - raw.c0()
        })
        .collect();
    rest.sort_by(f64::total_cmp);
    raw.with_c0_for_test(-rest[rest.len() / 2])
}

/// The integrator's throughput probe (release, under `ny_box.sh measure`):
/// prints nodes/s per evaluator and asserts nothing about speed, so it is an
/// evidence job behind ny-propagate's `benchmarks` feature (wall-clock) rather
/// than `#[ignore]`: `targo --unverified test --release --locked --offline -p
/// ny-propagate --features benchmarks --lib fd_block_rate_probe -- --nocapture`.
#[cfg(feature = "benchmarks")]
#[test]
fn fd_block_rate_probe() {
    let all = rayon::current_num_threads().max(1);
    for (n, depth, tag) in [(11418usize, 28usize, "tiny-like"), (3448, 45, "cifar-like")] {
        let prob = centred(
            &FrozenDualProblem::synthetic(n, depth, 0xFD0 ^ n as u64),
            0xC3,
        );
        for threads in [1usize, all] {
            // One thread gets fewer nodes (the node path is slow there).
            let count = if threads == 1 { 4_096 } else { 32_768 };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            for evaluator in [
                Evaluator::Node,
                Evaluator::Block { far_first: false },
                Evaluator::Block { far_first: true },
            ] {
                let (secs, pruned, phases) =
                    pool.install(|| rate_probe(&prob, count, 0x5EED, evaluator));
                #[allow(clippy::cast_precision_loss)]
                let nps = if secs > 0.0 { count as f64 / secs } else { 0.0 };
                let split = phase_probe::PHASES
                    .iter()
                    .zip(phases)
                    .map(|(p, s)| format!("{p}={s:.3}"))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "[fd-block-rate] shape={tag} n={n} depth={depth} threads={threads} \
                     eval={} far_first={} nodes={count} secs={secs:.3} nps={nps:.0} \
                     pruned={pruned} phase_thread_secs={split}",
                    evaluator.token(),
                    u8::from(matches!(evaluator, Evaluator::Block { far_first: true })),
                );
            }
        }
    }
}
