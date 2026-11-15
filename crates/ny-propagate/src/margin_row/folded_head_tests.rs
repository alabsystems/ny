// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cct-margin-row-admission: the lane charges a BN-folded head's certified
//! fold error (`TwinOpSpec::Gemm { bias_err, weight_rel_err }`).
//!
//! The lane never sees the exact `Gemm -> BatchNormalization` map, only the
//! folded weights `W'`, `b'` plus the certified ball
//! `|W - W'| <= rel |W'|`, `|b - b'| <= bias_err` the exact map lies in. So a
//! sound y-box / margin bound must hold for EVERY net inside that ball, not
//! just for the stored one. At a trunk output `z >= 0` the ball's extreme head
//! pre-activations are exactly `y' +- (rel * |W'| z + bias_err)`; these tests
//! sample `z` and demand the lane's certified bounds cover those extremes.
//!
//! The error budgets here are INFLATED (percent-level, not the 1e-15 of a real
//! fold) and the input box is near-degenerate, so an engine that dropped the
//! charge fails by orders of magnitude rather than by rounding noise — the
//! tests discriminate.

use std::collections::BTreeMap;

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::bounds::{
    compose_viay, head_gates, margin_seed, per_class_direct, row_dots, MarginBatch, YBox,
};
use super::engine::{BackwardEngine, LaneDir};
use super::net::TwinNet;
use super::root::RootGates;
use super::rounding::RoundMode;
use super::spec::{TwinOpSpec, TwinSpec, MAX_ROOT_TABLEAU_ENTRIES};

const N_IN: usize = 9; // 1 x 3 x 3
const N_Y: usize = 4;
const N_OUT: usize = 3;
const REL: f64 = 0.02;
const BERR: f64 = 0.05;

/// `Conv(1->2, 3x3, pad 1) -> Relu -> Flatten -> Gemm(18->4) [folded] ->
/// Relu -> Gemm(4->3)`: the CNN7 head shape in miniature.
fn folded_spec(seed: u64, rel: f64, berr: f64) -> TwinSpec {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w =
        |n: usize, s: f64| -> Vec<f64> { (0..n).map(|_| rng.random_range(-s..s)).collect() };
    TwinSpec {
        n_in: N_IN,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: w(2 * 3 * 3, 0.8),
                bias: w(2, 0.3),
                bias_err: vec![0.0; 2],
                weight_rel_err: 0.0,
                kernel: (2, 1, 3, 3),
                stride: (1, 1),
                pads: (1, 1, 1, 1),
                ishape: (1, 3, 3),
                oshape: (2, 3, 3),
            }, // t1
            TwinOpSpec::Relu { input: 1 },    // t2 (trunk relu)
            TwinOpSpec::Flatten { input: 2 }, // t3
            TwinOpSpec::Gemm {
                input: 3,
                weight: w(N_Y * 18, 0.6),
                bias: w(N_Y, 0.4),
                bias_err: vec![berr; N_Y],
                weight_rel_err: rel,
                shape: (N_Y, 18),
            }, // t4 (y, BN folded)
            TwinOpSpec::Relu { input: 4 },    // t5 (head relu)
            TwinOpSpec::Gemm {
                input: 5,
                weight: w(N_OUT * N_Y, 1.0),
                bias: w(N_OUT, 0.2),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (N_OUT, N_Y),
            }, // t6
        ],
    }
}

/// Points in `[lo, hi]` as COLUMNS.
fn sample(rng: &mut StdRng, lo: &[f64], hi: &[f64], n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((lo.len(), n));
    for i in 0..lo.len() {
        for b in 0..n {
            x[[i, b]] = if hi[i] > lo[i] {
                rng.random_range(lo[i]..hi[i])
            } else {
                lo[i]
            };
        }
    }
    x
}

/// Stored-net head pre-activation `y'` and the trunk output `z` at each point.
fn forward_y_and_z(net: &TwinNet, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let last = *net.trunk_relus.last().expect("one trunk relu");
    let n_z = net.tsize[last + 1];
    let sel: BTreeMap<usize, Vec<usize>> = [(last, (0..n_z).collect())].into_iter().collect();
    let (y, pre) = net.forward_points(x, &sel).expect("forward");
    let z = pre[&last].mapv(|v| v.max(0.0));
    (y, z)
}

/// The error ball's extreme head pre-activations at each point:
/// `y' -+ (rel * sum_i |W'_ki| z_i + bias_err_k)`.
fn ball_extremes(net: &TwinNet, y: &Array2<f64>, z: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (w1, _, (n_y, n_h)) = net.gemm1();
    let (b1_err, rel) = net.gemm1_fold_err();
    let mut lo = y.clone();
    let mut hi = y.clone();
    for b in 0..y.ncols() {
        for k in 0..n_y {
            let mut spread = b1_err[k];
            for i in 0..n_h {
                spread += rel * w1[k * n_h + i].abs() * z[[i, b]];
            }
            lo[[k, b]] -= spread;
            hi[[k, b]] += spread;
        }
    }
    (lo, hi)
}

/// Per-class certified margin lower bounds `Y_t - Y_j`, combined exactly as
/// the lane's root evaluation combines its three sound paths.
fn certified_margins(net: &TwinNet, lo: &[f64], hi: &[f64], t: usize, adv: &[usize]) -> Vec<f64> {
    let mode = RoundMode::Outward;
    let root = RootGates::build(net, lo, hi, mode, None).expect("root");
    let eng = BackwardEngine::new(net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb = MarginBatch::new(net, t, adv).expect("mb");
    let gates = head_gates(&ybox, mode);
    let ms = margin_seed(&mb, &gates, &ybox, mode);
    let pass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .expect("pass");
    let direct = per_class_direct(&eng, &pass, &ms, 0..adv.len());
    let ald = row_dots(&root, &al);
    let aud = row_dots(&root, &au);
    let m2v = compose_viay(&eng, &mb, &gates, &al, &au, &ald, &aud, mode);
    (0..adv.len())
        .map(|k| direct[k].max(ms.m1[k]).max(m2v[k]))
        .collect()
}

fn outward_ybox(net: &TwinNet, lo: &[f64], hi: &[f64]) -> YBox {
    let root = RootGates::build(net, lo, hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    YBox::from_rows(&eng, &al, &au)
}

fn boxes(rng: &mut StdRng) -> Vec<(Vec<f64>, Vec<f64>)> {
    let x0: Vec<f64> = (0..N_IN).map(|_| rng.random_range(-1.0..1.0)).collect();
    // Near-degenerate (the discriminating case: the stored-net y-box is
    // ~1e-8 wide, the charged one ~BERR wide) and a genuinely wide box.
    [1e-9, 0.1]
        .iter()
        .map(|&r| {
            (
                x0.iter().map(|v| v - r).collect(),
                x0.iter().map(|v| v + r).collect(),
            )
        })
        .collect()
}

/// SOUNDNESS: the lane's certified head y-box encloses every head
/// pre-activation any net inside the certified fold-error ball can produce.
#[test]
fn folded_head_fold_error_is_charged_in_y_box() {
    let mut rng = StdRng::seed_from_u64(0xF01D_4EAD);
    let mut discriminating = 0usize;
    for seed in 0..6u64 {
        let net = TwinNet::compile(&folded_spec(seed, REL, BERR)).expect("compile");
        // The SAME weights with the fold error dropped: what an engine that
        // ignored `gemm1_fold_err` would certify.
        let uncharged = TwinNet::compile(&folded_spec(seed, 0.0, 0.0)).expect("compile");
        assert_eq!(net.gemm1_fold_err().1, REL);
        for (lo, hi) in boxes(&mut rng) {
            let ybox = outward_ybox(&net, &lo, &hi);
            let ybox0 = outward_ybox(&uncharged, &lo, &hi);
            let x = sample(&mut rng, &lo, &hi, 48);
            let (y, z) = forward_y_and_z(&net, &x);
            let (ylo, yhi) = ball_extremes(&net, &y, &z);
            for k in 0..N_Y {
                for b in 0..x.ncols() {
                    assert!(
                        ybox.ly[k] <= ylo[[k, b]] + 1e-12 && ybox.uy[k] >= yhi[[k, b]] - 1e-12,
                        "seed {seed} y[{k}] ball [{}, {}] escapes certified box [{}, {}]",
                        ylo[[k, b]],
                        yhi[[k, b]],
                        ybox.ly[k],
                        ybox.uy[k]
                    );
                    // Would the UNCHARGED box have been caught out here?
                    if ylo[[k, b]] < ybox0.ly[k] || yhi[[k, b]] > ybox0.uy[k] {
                        discriminating += 1;
                    }
                }
            }
        }
    }
    assert!(
        discriminating > 0,
        "no sample escaped the UNCHARGED box, so this test cannot tell a charged \
         engine from one that drops the fold error — it is vacuous"
    );
}

/// SOUNDNESS: the lane's certified per-class margin lower bound holds for the
/// WORST net inside the fold-error ball at every sampled point (each head
/// neuron pushed to whichever ball extreme lowers `Y_t - Y_j`).
#[test]
fn folded_head_margin_bounds_enclose_every_net_in_the_error_ball() {
    let mut rng = StdRng::seed_from_u64(0xB0_BA11);
    let (t, adv) = (0usize, vec![1usize, 2]);
    for seed in 10..16u64 {
        let net = TwinNet::compile(&folded_spec(seed, REL, BERR)).expect("compile");
        let (w2, b2, (_, n_y)) = net.gemm2();
        for (lo, hi) in boxes(&mut rng) {
            let bound = certified_margins(&net, &lo, &hi, t, &adv);
            let x = sample(&mut rng, &lo, &hi, 48);
            let (y, z) = forward_y_and_z(&net, &x);
            let (ylo, yhi) = ball_extremes(&net, &y, &z);
            for (r, &j) in adv.iter().enumerate() {
                for b in 0..x.ncols() {
                    let mut worst = b2[t] - b2[j];
                    for k in 0..n_y {
                        let wk = w2[t * n_y + k] - w2[j * n_y + k];
                        let yk = if wk >= 0.0 { ylo[[k, b]] } else { yhi[[k, b]] };
                        worst += wk * yk.max(0.0);
                    }
                    assert!(
                        bound[r] <= worst + 1e-12,
                        "seed {seed} class {j}: certified {} above the ball's worst margin {worst}",
                        bound[r]
                    );
                }
            }
        }
    }
}

/// Charging only ever LOOSENS the head: the folded net's y-box contains the
/// same weights' uncharged y-box (the value lanes are identical; only the
/// certified error lanes grow). An unfolded head — `bias_err` empty or all
/// zero, `rel` 0 — takes the historical arithmetic BIT-FOR-BIT.
#[test]
fn folded_head_charge_only_loosens() {
    let mut rng = StdRng::seed_from_u64(0x10_05E5);
    for seed in 20..24u64 {
        let charged = TwinNet::compile(&folded_spec(seed, REL, BERR)).expect("compile");
        let exact = TwinNet::compile(&folded_spec(seed, 0.0, 0.0)).expect("compile");
        let empty = {
            let mut spec = folded_spec(seed, 0.0, 0.0);
            let TwinOpSpec::Gemm { bias_err, .. } = &mut spec.ops[3] else {
                unreachable!("op 3 is the head gemm")
            };
            bias_err.clear();
            TwinNet::compile(&spec).expect("compile")
        };
        assert_eq!(exact.gemm1_fold_err(), (&[0.0; N_Y][..], 0.0));
        assert_eq!(empty.gemm1_fold_err(), (&[0.0; N_Y][..], 0.0));
        for (lo, hi) in boxes(&mut rng) {
            let yc = outward_ybox(&charged, &lo, &hi);
            let ye = outward_ybox(&exact, &lo, &hi);
            let y0 = outward_ybox(&empty, &lo, &hi);
            for k in 0..N_Y {
                assert!(yc.ly[k] <= ye.ly[k] && yc.uy[k] >= ye.uy[k]);
                assert_eq!(ye.ly[k].to_bits(), y0.ly[k].to_bits());
                assert_eq!(ye.uy[k].to_bits(), y0.uy[k].to_bits());
            }
        }
    }
}

/// The compiler refuses what the lane cannot charge: a fold error on the
/// FINAL Gemm (read as exact by every margin consumer), malformed budgets,
/// and a root tableau above the shared ceiling.
#[test]
fn compile_refuses_folded_final_gemm_and_bad_fold_budgets() {
    let final_folded = {
        let mut spec = folded_spec(1, 0.0, 0.0);
        let TwinOpSpec::Gemm { weight_rel_err, .. } = &mut spec.ops[5] else {
            unreachable!("op 5 is the final gemm")
        };
        *weight_rel_err = 1e-15;
        spec
    };
    assert!(final_folded.ops[5].is_folded_gemm());
    assert!(TwinNet::compile(&final_folded).is_err());

    for bad in [
        (vec![0.1; N_Y - 1], 0.0),       // wrong length
        (vec![-0.1; N_Y], 0.0),          // negative
        (vec![f64::NAN; N_Y], 0.0),      // non-finite
        (vec![0.0; N_Y], -1e-15),        // negative rel
        (vec![0.0; N_Y], f64::INFINITY), // non-finite rel
    ] {
        let mut spec = folded_spec(2, 0.0, 0.0);
        let TwinOpSpec::Gemm {
            bias_err,
            weight_rel_err,
            ..
        } = &mut spec.ops[3]
        else {
            unreachable!("op 3 is the head gemm")
        };
        *bias_err = bad.0.clone();
        *weight_rel_err = bad.1;
        assert!(
            TwinNet::compile(&spec).is_err(),
            "fold budget {bad:?} was accepted"
        );
    }
}

/// #cct-margin-row-admission memory guard: `root_tableau_entries` is
/// `widest tensor x (n_in + 1)` and the compiler refuses above the ceiling —
/// the cct TinyImageNet CNN7 pair's regime — while the small net passes.
#[test]
fn compile_refuses_root_tableau_above_cap() {
    let small = folded_spec(3, REL, BERR);
    // widest = conv out 18 (> n_in 9): 18 * 10.
    assert_eq!(small.root_tableau_entries(), Some(18 * 10));
    assert!(small.root_tableau_within_cap());

    // A dense-only net whose INPUT is the widest tensor: n_in = 40_000 gives
    // 40_000 * 40_001 = 1.6e9 entries > 2^30.
    let n_in = 40_000usize;
    let wide = TwinSpec {
        n_in,
        ops: vec![
            TwinOpSpec::Relu { input: 0 },
            TwinOpSpec::Gemm {
                input: 1,
                weight: vec![0.0; 2 * n_in],
                bias: vec![0.0; 2],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, n_in),
            },
            TwinOpSpec::Relu { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![0.0; 4],
                bias: vec![0.0; 2],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    };
    assert_eq!(wide.root_tableau_entries(), Some(n_in * (n_in + 1)));
    assert!(n_in * (n_in + 1) > MAX_ROOT_TABLEAU_ENTRIES);
    assert!(!wide.root_tableau_within_cap());
    let err = TwinNet::compile(&wide)
        .err()
        .expect("above-cap net must be refused");
    assert!(
        format!("{err}").contains("root tableau"),
        "refusal must name the tableau cap: {err}"
    );
}
