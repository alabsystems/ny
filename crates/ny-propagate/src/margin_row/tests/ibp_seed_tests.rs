// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-ibp-seed soundness, contract and dark-default tests.
//!
//! The oracle is EXACT: every value the stored network (or any member of its
//! certified fold ball) computes from an f64 input is a dyadic rational, and
//! the forward uses only `+`, `*` and `max`, so [`Dy`] (a `BigInt` mantissa and
//! a binary exponent) evaluates it with no rounding at all. Its arithmetic is
//! cross-checked against `BigRational` by `ibp_seed_dyadic_oracle_is_exact`.
//!
//! * `ibp_seed_box_contains_exact_rational_samples` — CNN7-shaped chains
//!   (conv -> relu -> strided conv [-> standalone BN] -> relu -> 1x1 conv ->
//!   relu, FC-BN head) with BN folds (`weight_rel_err` / `bias_err`, the
//!   standalone affine's `scale_rel_err` / `shift_err`), ASYMMETRIC input boxes
//!   straddling 0 and ACTIVE relus: 10^4+ exact samples (random vertices,
//!   interior points, and per-neuron targeted extreme vertices WITH the
//!   extreme fold-ball member) against the seeded boxes of every trunk layer,
//!   with backward-interm off and on and on the f32 compute lane; plus the pure
//!   IBP chain. Engagement (tightened, stabilized, active relus) is required,
//!   and the CODE-MUTATION PIN (the IBP fold charge dropped) must be caught.
//! * `ibp_seed_trunk0_box_contains_the_exact_fold_ball_hull` — the seeded
//!   trunk-0 box contains the EXACT hull over the whole box and the whole fold
//!   ball (`fold_ball_hull`, no sampling), and the dropped-fold-charge
//!   mutation misses it.
//! * `ibp_seed_is_shrink_only_and_refuses_crossings` — `intersect_layer` on
//!   adversarial arrays (crossings both ways, NaN / inf, shape mismatch), and
//!   every seeded walk step: published ⊆ forward, published = forward ∩ IBP
//!   exactly (backward-interm off), backward-interm only shrinks further.
//! * `ibp_seed_keeps_bi_cal_on_forward_only_lines` — with clip rows retained,
//!   every retained slack is recovered against the FORWARD-ONLY box (bit
//!   identical to `retain_clip_rows` over the tap's forward box), with the seed
//!   engaged on retained rows; the mutation that captures `bi_cal` AFTER the
//!   seed must be caught.
//! * `ibp_seed_dark_is_byte_identical` — the key off (and armed-but-declined)
//!   builds are bit-identical to the historical entries, root bound included.
//! * `ibp_seed_declines_residual_trunks` — a residual trunk armed declines
//!   (`Decline::Residual` on every layer), Parity declines, both bit-identical.
//! * `ibp_seed_never_verifies_a_false_bound` — the certified per-class root
//!   bound of the seeded root is `<=` the EXACT margin at every sample and fold
//!   ball member, and stays negative on a falsified target.
//! * `ibp_seed_is_dark_by_default_and_reachable_by_preset` — the typed key
//!   resolves `Source::Config` (not LegacyEnv), and the frozen route resolves
//!   it into the root-precision policy and nothing else.
//!
//! The seed produces no witness, so it cannot emit a `sat` of any kind.

use super::*;

use std::collections::HashMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::margin_row::bab::root_eval;
use crate::margin_row::backward_interm::BiCfg;
use crate::margin_row::engine::BackwardEngine as Engine;
use crate::margin_row::ibp_seed::{
    intersect_layer, intersect_layer_or_refuse, pure_ibp_trunk_boxes,
    with_ibp_fold_charge_dropped_for_test, with_ibp_seed_tap_for_test, Decline, LayerTap,
};
use crate::margin_row::rebake::certified_root_bound;
use crate::margin_row::root::{retain_clip_rows, with_bi_cal_after_seed_for_test};
use crate::margin_row::{ErrLane, MarginRowRuntimePolicy, RootPrecisionRuntimePolicy};

// ---------------------------------------------------------------------------
//  Exact dyadic arithmetic
// ---------------------------------------------------------------------------

/// An exact dyadic rational `m * 2^e`.
#[derive(Clone, Debug)]
struct Dy {
    m: BigInt,
    e: i64,
}

impl Dy {
    fn zero() -> Self {
        Self {
            m: BigInt::zero(),
            e: 0,
        }
    }

    /// The exact value of a finite f64.
    fn of(v: f64) -> Self {
        assert!(
            v.is_finite(),
            "non-finite value reached the exact oracle: {v}"
        );
        if v == 0.0 {
            return Self::zero();
        }
        let bits = v.to_bits();
        let neg = bits >> 63 == 1;
        let biased = i64::try_from((bits >> 52) & 0x7ff).expect("11-bit exponent");
        let frac = bits & ((1u64 << 52) - 1);
        let (mant, e) = if biased == 0 {
            (frac, -1074)
        } else {
            (frac | (1u64 << 52), biased - 1075)
        };
        let m = BigInt::from(mant);
        Self {
            m: if neg { -m } else { m },
            e,
        }
        .norm()
    }

    /// Strip trailing zero bits (canonical, keeps mantissas small).
    fn norm(mut self) -> Self {
        match self.m.trailing_zeros() {
            None => Self::zero(),
            Some(0) => self,
            Some(tz) => {
                self.m >>= tz;
                self.e += i64::try_from(tz).expect("shift fits i64");
                self
            }
        }
    }

    fn add(&self, o: &Self) -> Self {
        if self.m.is_zero() {
            return o.clone();
        }
        if o.m.is_zero() {
            return self.clone();
        }
        let e = self.e.min(o.e);
        let a = &self.m << usize::try_from(self.e - e).expect("non-negative shift");
        let b = &o.m << usize::try_from(o.e - e).expect("non-negative shift");
        Self { m: a + b, e }.norm()
    }

    fn neg(&self) -> Self {
        Self {
            m: -(self.m.clone()),
            e: self.e,
        }
    }

    fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }

    fn mul(&self, o: &Self) -> Self {
        if self.m.is_zero() || o.m.is_zero() {
            return Self::zero();
        }
        Self {
            m: &self.m * &o.m,
            e: self.e + o.e,
        }
        .norm()
    }

    fn is_neg(&self) -> bool {
        self.m.is_negative()
    }

    fn relu(&self) -> Self {
        if self.is_neg() {
            Self::zero()
        } else {
            self.clone()
        }
    }

    /// `self <= o`, exactly.
    fn le(&self, o: &Self) -> bool {
        !o.sub(self).is_neg()
    }

    /// The same value as a `BigRational` (the cross-check).
    fn to_q(&self) -> BigRational {
        if self.e >= 0 {
            BigRational::from_integer(&self.m << usize::try_from(self.e).expect("shift"))
        } else {
            let den = BigInt::from(1u8) << usize::try_from(-self.e).expect("shift");
            BigRational::new(self.m.clone(), den)
        }
    }

    /// Approximate f64 (messages only).
    fn approx(&self) -> f64 {
        self.to_q().to_f64().unwrap_or(f64::NAN)
    }
}

/// The oracle's arithmetic, cross-checked against `BigRational` on values
/// spanning signs, magnitudes and subnormals.
#[test]
fn ibp_seed_dyadic_oracle_is_exact() {
    let q = |v: f64| BigRational::from_float(v).expect("finite");
    let mut rng = StdRng::seed_from_u64(0xD7AD_1C00);
    let mut vals: Vec<f64> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        -0.3,
        1e-300,
        -5e-324,
        f64::MIN_POSITIVE,
        3.5e10,
    ];
    for _ in 0..200 {
        let mag: f64 = rng.random_range(-40.0..40.0);
        let sign = if rng.random_range(0.0..1.0) < 0.5 {
            -1.0
        } else {
            1.0
        };
        vals.push(sign * rng.random_range(0.5..1.0) * mag.exp2());
    }
    for (i, &a) in vals.iter().enumerate() {
        let b = vals[(i * 7 + 3) % vals.len()];
        let (da, db) = (Dy::of(a), Dy::of(b));
        assert_eq!(da.to_q(), q(a), "of({a})");
        assert_eq!(da.add(&db).to_q(), q(a) + q(b), "{a} + {b}");
        assert_eq!(da.sub(&db).to_q(), q(a) - q(b), "{a} - {b}");
        assert_eq!(da.mul(&db).to_q(), q(a) * q(b), "{a} * {b}");
        assert_eq!(da.le(&db), q(a) <= q(b), "{a} <= {b}");
        assert_eq!(
            da.relu().to_q(),
            if q(a) < BigRational::zero() {
                BigRational::zero()
            } else {
                q(a)
            },
            "relu({a})"
        );
    }
}

// ---------------------------------------------------------------------------
//  Fixtures: CNN7-shaped chains with BN folds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ChainCfg {
    /// Conv `weight_rel_err` (the conv+BN fold).
    rel: f64,
    /// Conv `bias_err`.
    berr: f64,
    /// Standalone BN `(scale_rel_err, shift_err)` after the strided conv.
    affine: Option<(f64, f64)>,
    /// FC-BN head fold `(weight_rel_err, bias_err)`.
    head: (f64, f64),
    seed: u64,
}

/// Large fold errors make the fold ball LOAD-BEARING in sampled checks (a
/// missing charge is ~1e-3, not an ulp); the small ones are production-sized.
const CONFIGS: &[ChainCfg] = &[
    ChainCfg {
        rel: 1e-3,
        berr: 1e-3,
        affine: None,
        head: (1e-15, 1e-12),
        seed: 0x1B5E_ED01,
    },
    ChainCfg {
        rel: 1e-3,
        berr: 0.0,
        affine: Some((1e-3, 1e-3)),
        head: (1e-15, 0.0),
        seed: 0x1B5E_ED02,
    },
    ChainCfg {
        rel: 1e-15,
        berr: 1e-12,
        affine: Some((1e-15, 1e-12)),
        head: (1e-15, 1e-12),
        seed: 0x1B5E_ED03,
    },
    ChainCfg {
        rel: 1e-13,
        berr: 0.0,
        affine: None,
        head: (0.0, 0.0),
        seed: 0x1B5E_ED04,
    },
    ChainCfg {
        rel: 0.0,
        berr: 0.0,
        affine: Some((0.0, 0.0)),
        head: (1e-15, 1e-12),
        seed: 0x1B5E_ED05,
    },
];

/// Trunk geometry.
const CIN: usize = 1;
const HW: usize = 4;
const C1: usize = 3;
const C2: usize = 3;
const C3: usize = 4;
const NY: usize = 6;
const NOUT: usize = 3;

/// Conv(1->3, 3x3, pad 1) -> Relu -> Conv(3->3, 3x3, stride 2, pad 1)
/// [-> ChannelAffine] -> Relu -> Conv(3->4, 1x1) -> Relu -> Flatten ->
/// Gemm(16->6, folded) -> Relu -> Gemm(6->3).
///
/// Weights are POSITIVE-SKEWED (the IBP-trained regime in which interval
/// boxes beat DeepPoly's alpha=1 lower lines), and output channel 0 of the
/// strided conv has all-non-negative weights and a small positive bias: its
/// IBP lower bound is `>= bias > 0` whenever its inputs are relu outputs, while
/// the tableau's alpha=1 lower lines leave it unstable — a guaranteed IBP
/// STABILIZATION on every fixture (the engagement the tests require).
fn cnn7ish_spec(cfg: &ChainCfg) -> TwinSpec {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let skew = |rng: &mut StdRng, n: usize, s: f64| -> Vec<f64> {
        (0..n).map(|_| rng.random_range(-0.35..1.0) * s).collect()
    };
    let signed = |rng: &mut StdRng, n: usize, s: f64| -> Vec<f64> {
        (0..n).map(|_| rng.random_range(-s..s)).collect()
    };
    let mut ops = Vec::new();
    // t1: conv1.
    ops.push(TwinOpSpec::Conv {
        input: 0,
        weight: skew(&mut rng, C1 * CIN * 9, 0.6),
        bias: (0..C1).map(|_| rng.random_range(-0.1..0.3)).collect(),
        bias_err: vec![cfg.berr; C1],
        weight_rel_err: cfg.rel,
        kernel: (C1, CIN, 3, 3),
        stride: (1, 1),
        pads: (1, 1, 1, 1),
        ishape: (CIN, HW, HW),
        oshape: (C1, HW, HW),
    });
    // t2: trunk relu 0.
    ops.push(TwinOpSpec::Relu { input: 1 });
    // t3: strided conv2, channel 0 structured (see the doc).
    let mut w2 = skew(&mut rng, C2 * C1 * 9, 0.4);
    for v in w2.iter_mut().take(C1 * 9) {
        *v = v.abs();
    }
    let mut b2: Vec<f64> = (0..C2).map(|_| rng.random_range(-0.1..0.4)).collect();
    b2[0] = 0.05;
    ops.push(TwinOpSpec::Conv {
        input: 2,
        weight: w2,
        bias: b2,
        bias_err: vec![cfg.berr; C2],
        weight_rel_err: cfg.rel,
        kernel: (C2, C1, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (C1, HW, HW),
        oshape: (C2, 2, 2),
    });
    let mut last = 3;
    if let Some((srel, serr)) = cfg.affine {
        // Positive scale on the structured channel (keeps its stabilization),
        // both signs elsewhere.
        let scale: Vec<f64> = (0..C2)
            .map(|c| {
                let v: f64 = rng.random_range(0.5..1.5);
                if c % 2 == 1 {
                    -v
                } else {
                    v
                }
            })
            .collect();
        let mut shift: Vec<f64> = (0..C2).map(|_| rng.random_range(0.0..0.3)).collect();
        shift[0] = 0.02;
        ops.push(TwinOpSpec::ChannelAffine {
            input: 3,
            scale,
            shift,
            scale_rel_err: srel,
            shift_err: vec![serr; C2],
            shape: (C2, 2, 2),
        });
        last = 4;
    }
    // Trunk relu 1.
    ops.push(TwinOpSpec::Relu { input: last });
    let r1 = last + 1;
    ops.push(TwinOpSpec::Conv {
        input: r1,
        weight: skew(&mut rng, C3 * C2, 0.8),
        bias: (0..C3).map(|_| rng.random_range(-0.2..0.3)).collect(),
        bias_err: vec![cfg.berr; C3],
        weight_rel_err: cfg.rel,
        kernel: (C3, C2, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (C2, 2, 2),
        oshape: (C3, 2, 2),
    });
    // Trunk relu 2.
    ops.push(TwinOpSpec::Relu { input: r1 + 1 });
    let r2 = r1 + 2;
    ops.push(TwinOpSpec::Flatten { input: r2 });
    let (head_rel, head_berr) = cfg.head;
    ops.push(TwinOpSpec::Gemm {
        input: r2 + 1,
        weight: signed(&mut rng, NY * C3 * 4, 0.5),
        bias: signed(&mut rng, NY, 0.3),
        bias_err: vec![head_berr; NY],
        weight_rel_err: head_rel,
        shape: (NY, C3 * 4),
    });
    ops.push(TwinOpSpec::Relu { input: r2 + 2 });
    ops.push(TwinOpSpec::Gemm {
        input: r2 + 3,
        weight: signed(&mut rng, NOUT * NY, 0.6),
        bias: signed(&mut rng, NOUT, 0.2),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (NOUT, NY),
    });
    TwinSpec {
        n_in: CIN * HW * HW,
        ops,
    }
}

/// A fixture: spec, compiled net and an ASYMMETRIC input box.
struct Fx {
    cfg: ChainCfg,
    spec: TwinSpec,
    net: TwinNet,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

fn chain_fixture(cfg: &ChainCfg) -> Fx {
    let spec = cnn7ish_spec(cfg);
    let net = TwinNet::compile(&spec).expect("CNN7-shaped fixture compiles");
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0xB0C5);
    let (lo, hi): (Vec<f64>, Vec<f64>) = (0..spec.n_in)
        .map(|_| {
            let c: f64 = rng.random_range(-0.3..0.3);
            let below: f64 = rng.random_range(0.1..0.6);
            let above: f64 = rng.random_range(0.02..0.35);
            (c - below, c + above)
        })
        .unzip();
    Fx {
        cfg: *cfg,
        spec,
        net,
        lo,
        hi,
    }
}

fn policy(ibp_seed: bool, root_f32: bool) -> RootPrecisionRuntimePolicy {
    RootPrecisionRuntimePolicy {
        root_f32,
        tableau_f32: false,
        err_lane: ErrLane::PerCol,
        ibp_seed,
        // #root-box-source: the seed is a TABLEAU-walk mechanism.
        box_source: crate::margin_row::RootBoxSource::Tableau,
    }
}

/// An always-affordable backward-interm config (every unstable neuron).
fn bi_cfg() -> BiCfg {
    BiCfg {
        secs: 1_000.0,
        chunk: 16,
        chunk_secs: 0.0,
        topk: 4_096,
        yield_window: 0,
        yield_ratio: 0.25,
        interm_alpha: 0,
    }
}

/// Build through the explicit-policy entry, recording every seed step.
fn build_tapped(
    fx: &Fx,
    pol: RootPrecisionRuntimePolicy,
    bi: Option<BiCfg>,
    mode: RoundMode,
) -> (RootGates, Vec<LayerTap>) {
    let (built, taps) = with_ibp_seed_tap_for_test(|| {
        RootGates::build_with_policy_for_test(&fx.net, &fx.lo, &fx.hi, mode, pol, bi)
    });
    (built.expect("root builds").0, taps)
}

// ---------------------------------------------------------------------------
//  The exact forward over a fold-ball member
// ---------------------------------------------------------------------------

/// One fold-ball member: per op index, the sign (-1, 0, +1) every folded
/// parameter takes inside its certified error ball (absent = stored value).
#[derive(Clone, Default)]
struct Member {
    signs: HashMap<usize, (Vec<i8>, Vec<i8>)>,
}

/// One op's exact parameters for one member.
// The conv geometry tuples are boxed so `Conv` stays within the workspace's
// 64-byte clippy::large_enum_variant threshold of `Gemm`.
enum XOp {
    Conv {
        input: usize,
        w: Vec<Dy>,
        b: Vec<Dy>,
        kernel: Box<(usize, usize, usize, usize)>,
        stride: (usize, usize),
        pads: Box<(usize, usize, usize, usize)>,
        ishape: Box<(usize, usize, usize)>,
        oshape: Box<(usize, usize, usize)>,
    },
    Affine {
        input: usize,
        s: Vec<Dy>,
        t: Vec<Dy>,
        hw: usize,
    },
    Relu {
        input: usize,
        trunk: bool,
    },
    Flatten {
        input: usize,
    },
    Add {
        lhs: usize,
        rhs: usize,
    },
    Gemm {
        input: usize,
        w: Vec<Dy>,
        b: Vec<Dy>,
        shape: (usize, usize),
    },
}

/// `v + sigma rel |v|` exactly (a relative-ball member).
fn ball_value(v: f64, rel: f64, sigma: i8) -> Dy {
    let base = Dy::of(v);
    if sigma == 0 || rel == 0.0 {
        return base;
    }
    let d = Dy::of(rel).mul(&Dy::of(v.abs()));
    if sigma > 0 {
        base.add(&d)
    } else {
        base.sub(&d)
    }
}

/// `v + tau err` exactly (an absolute-ball member).
fn ball_shift(v: f64, err: f64, tau: i8) -> Dy {
    let base = Dy::of(v);
    if tau == 0 || err == 0.0 {
        return base;
    }
    if tau > 0 {
        base.add(&Dy::of(err))
    } else {
        base.sub(&Dy::of(err))
    }
}

fn exact_ops(spec: &TwinSpec, member: &Member) -> Vec<XOp> {
    let mut seen_gemm = false;
    let mut out = Vec::with_capacity(spec.ops.len());
    for (k, op) in spec.ops.iter().enumerate() {
        let sg = member.signs.get(&k);
        let sw = |i: usize| sg.map_or(0, |(w, _)| w.get(i).copied().unwrap_or(0));
        let sb = |i: usize| sg.map_or(0, |(_, b)| b.get(i).copied().unwrap_or(0));
        out.push(match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
            } => XOp::Conv {
                input: *input,
                w: weight
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| ball_value(v, *weight_rel_err, sw(i)))
                    .collect(),
                b: bias
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| ball_shift(v, bias_err[i], sb(i)))
                    .collect(),
                kernel: Box::new(*kernel),
                stride: *stride,
                pads: Box::new(*pads),
                ishape: Box::new(*ishape),
                oshape: Box::new(*oshape),
            },
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
                shape,
            } => XOp::Affine {
                input: *input,
                s: scale
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| ball_value(v, *scale_rel_err, sw(i)))
                    .collect(),
                t: shift
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| ball_shift(v, shift_err[i], sb(i)))
                    .collect(),
                hw: shape.1 * shape.2,
            },
            TwinOpSpec::Relu { input } => XOp::Relu {
                input: *input,
                trunk: !seen_gemm,
            },
            TwinOpSpec::Flatten { input } => XOp::Flatten { input: *input },
            TwinOpSpec::Add { lhs, rhs } => XOp::Add {
                lhs: *lhs,
                rhs: *rhs,
            },
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                seen_gemm = true;
                XOp::Gemm {
                    input: *input,
                    w: weight
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| ball_value(v, *weight_rel_err, sw(i)))
                        .collect(),
                    b: bias
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            ball_shift(v, bias_err.get(i).copied().unwrap_or(0.0), sb(i))
                        })
                        .collect(),
                    shape: *shape,
                }
            }
            TwinOpSpec::ConvTranspose { .. } => {
                panic!("the ibp-seed fixtures use no ConvTranspose")
            }
        });
    }
    out
}

/// Exact trunk pre-activations (one vector per trunk relu) and, when
/// `head`, the exact logits.
struct Exact {
    trunk: Vec<Vec<Dy>>,
    logits: Vec<Dy>,
}

fn exact_forward(xops: &[XOp], x: &[f64], head: bool) -> Exact {
    let mut t: Vec<Vec<Dy>> = Vec::with_capacity(xops.len() + 1);
    t.push(x.iter().map(|&v| Dy::of(v)).collect());
    let mut trunk = Vec::new();
    for op in xops {
        let next: Vec<Dy> = match op {
            XOp::Conv {
                input,
                w,
                b,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
            } => {
                let (co, ci, kh, kw) = **kernel;
                let (_, ih, iw) = **ishape;
                let (_, oh, ow) = **oshape;
                let src = &t[*input];
                let mut out = Vec::with_capacity(co * oh * ow);
                for oc in 0..co {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = b[oc].clone();
                            for c in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oy * stride.0 + ky).checked_sub(pads.0);
                                        let ix = (ox * stride.1 + kx).checked_sub(pads.1);
                                        if let (Some(iy), Some(ix)) = (iy, ix) {
                                            if iy < ih && ix < iw {
                                                let wi = ((oc * ci + c) * kh + ky) * kw + kx;
                                                let xi = c * ih * iw + iy * iw + ix;
                                                acc = acc.add(&w[wi].mul(&src[xi]));
                                            }
                                        }
                                    }
                                }
                            }
                            out.push(acc);
                        }
                    }
                }
                out
            }
            XOp::Affine {
                input,
                s,
                t: sh,
                hw,
            } => t[*input]
                .iter()
                .enumerate()
                .map(|(j, v)| s[j / *hw].mul(v).add(&sh[j / *hw]))
                .collect(),
            XOp::Relu {
                input,
                trunk: is_trunk,
            } => {
                if *is_trunk {
                    trunk.push(t[*input].clone());
                } else if !head {
                    break;
                }
                t[*input].iter().map(Dy::relu).collect()
            }
            XOp::Flatten { input } => t[*input].clone(),
            XOp::Add { lhs, rhs } => t[*lhs]
                .iter()
                .zip(&t[*rhs])
                .map(|(a, b)| a.add(b))
                .collect(),
            XOp::Gemm { input, w, b, shape } => {
                if !head {
                    break;
                }
                let (no, ni) = *shape;
                let src = &t[*input];
                (0..no)
                    .map(|o| {
                        let mut acc = b[o].clone();
                        for i in 0..ni {
                            acc = acc.add(&w[o * ni + i].mul(&src[i]));
                        }
                        acc
                    })
                    .collect()
            }
        };
        t.push(next);
    }
    let logits = if head {
        t.last().cloned().unwrap_or_default()
    } else {
        Vec::new()
    };
    Exact { trunk, logits }
}

/// A random fold-ball VERTEX member (every folded parameter at +-1).
fn random_member(spec: &TwinSpec, rng: &mut StdRng) -> Member {
    let mut pm = |n: usize| -> Vec<i8> {
        (0..n)
            .map(|_| {
                if rng.random_range(0.0..1.0) < 0.5 {
                    -1
                } else {
                    1
                }
            })
            .collect()
    };
    let mut signs = HashMap::new();
    for (k, op) in spec.ops.iter().enumerate() {
        match op {
            TwinOpSpec::Conv { weight, bias, .. } | TwinOpSpec::Gemm { weight, bias, .. } => {
                signs.insert(k, (pm(weight.len()), pm(bias.len())));
            }
            TwinOpSpec::ChannelAffine { scale, shift, .. } => {
                signs.insert(k, (pm(scale.len()), pm(shift.len())));
            }
            _ => {}
        }
    }
    Member { signs }
}

/// A random box vertex (`vertex`) or interior point, exactly inside the box.
fn random_point(lo: &[f64], hi: &[f64], rng: &mut StdRng, vertex: bool) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(&l, &h)| {
            if vertex {
                if rng.random_range(0.0..1.0) < 0.5 {
                    l
                } else {
                    h
                }
            } else {
                let t: f64 = rng.random_range(0.0..1.0);
                (l + t * (h - l)).clamp(l, h)
            }
        })
        .collect()
}

/// The extreme vertex and fold-ball member of trunk-0 neuron `j` (conv1 is
/// the first op of every fixture): every tap's input at the face that pushes
/// `w' x` up (`upper`) or down, every tap weight moved by `+-rel |w'|` in the
/// direction of `x`'s sign, the bias at `+-bias_err`. Its exact value attains
/// the whole-ball, whole-box extreme of that neuron up to the certified box's
/// rounding charges, so an IBP box MISSING a fold charge excludes it.
fn targeted_extreme(fx: &Fx, j: usize, upper: bool) -> (Vec<f64>, Member) {
    let TwinOpSpec::Conv {
        weight,
        bias,
        kernel: (co, ci, kh, kw),
        stride,
        pads,
        ishape: (_, ih, iw),
        oshape: (_, oh, ow),
        ..
    } = &fx.spec.ops[0]
    else {
        panic!("every ibp-seed fixture starts with a conv");
    };
    let p = oh * ow;
    let (oc, sp) = (j / p, j % p);
    let (oy, ox) = (sp / ow, sp % ow);
    let mut x: Vec<f64> = fx
        .lo
        .iter()
        .zip(&fx.hi)
        .map(|(&l, &h)| (l + 0.5 * (h - l)).clamp(l, h))
        .collect();
    let mut wsign = vec![0i8; weight.len()];
    for c in 0..*ci {
        for ky in 0..*kh {
            for kx in 0..*kw {
                let iy = (oy * stride.0 + ky).checked_sub(pads.0);
                let ix = (ox * stride.1 + kx).checked_sub(pads.1);
                let (Some(iy), Some(ix)) = (iy, ix) else {
                    continue;
                };
                if iy >= *ih || ix >= *iw {
                    continue;
                }
                let wi = ((oc * ci + c) * kh + ky) * kw + kx;
                let xi = c * ih * iw + iy * iw + ix;
                let w = weight[wi];
                let up_face = (w >= 0.0) == upper;
                x[xi] = if up_face { fx.hi[xi] } else { fx.lo[xi] };
                // sigma rel |w'| x has the sign of sigma * x: push it the
                // extreme's way.
                let xs: i8 = if x[xi] >= 0.0 { 1 } else { -1 };
                wsign[wi] = if upper { xs } else { -xs };
            }
        }
    }
    let _ = co;
    let mut bsign = vec![0i8; bias.len()];
    bsign[oc] = if upper { 1 } else { -1 };
    let mut signs = HashMap::new();
    signs.insert(0usize, (wsign, bsign));
    (x, Member { signs })
}

/// One set of trunk boxes converted to exact dyadics ONCE (the containment
/// loops compare thousands of samples against the same boxes).
struct DyBoxes {
    tag: String,
    layers: Vec<Vec<(Dy, Dy, f64, f64)>>,
}

impl DyBoxes {
    fn of_pairs(tag: &str, pairs: &[(Vec<f64>, Vec<f64>)]) -> Self {
        Self {
            tag: tag.to_owned(),
            layers: pairs
                .iter()
                .map(|(l, u)| {
                    l.iter()
                        .zip(u)
                        .map(|(&a, &b)| (Dy::of(a), Dy::of(b), a, b))
                        .collect()
                })
                .collect(),
        }
    }

    fn of_root(tag: &str, root: &RootGates) -> Self {
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = root
            .layers
            .iter()
            .map(|g| (g.l.clone(), g.u.clone()))
            .collect();
        Self::of_pairs(tag, &pairs)
    }

    /// Count exact trunk samples outside these boxes (first few described).
    fn escapes(&self, ex: &Exact, ctx: &str, out: &mut Vec<String>) -> usize {
        assert_eq!(
            self.layers.len(),
            ex.trunk.len(),
            "{}: trunk depth",
            self.tag
        );
        let mut n = 0usize;
        for (li, (layer, pre)) in self.layers.iter().zip(&ex.trunk).enumerate() {
            assert_eq!(layer.len(), pre.len(), "{}: layer {li} width", self.tag);
            for (j, ((dl, du, l, u), v)) in layer.iter().zip(pre).enumerate() {
                if !(dl.le(v) && v.le(du)) {
                    n += 1;
                    if out.len() < 12 {
                        out.push(format!(
                            "{} {ctx}: layer {li} neuron {j}: box [{l:e}, {u:e}] excludes exact {:e}",
                            self.tag,
                            v.approx()
                        ));
                    }
                }
            }
        }
        n
    }
}

// ---------------------------------------------------------------------------
//  1. Exact containment
// ---------------------------------------------------------------------------

/// THE soundness gate: exact dyadic-rational containment of every seeded
/// trunk box, over 10^4+ samples including vertices and the targeted
/// whole-ball extremes, with the dropped-fold-charge mutation pin.
#[test]
fn ibp_seed_box_contains_exact_rational_samples() {
    let _env = ny_test_utils::env::lock_env_shared();
    let mut samples = 0usize;
    let mut checks = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut n_fail = 0usize;
    let mut dark_fail = 0usize;
    let (mut tightened, mut stabilized, mut crossed) = (0usize, 0usize, 0usize);
    let mut active_relus = 0usize;
    let mut straddles = 0usize;
    let mut mutation_escapes = 0usize;
    let mut mutation_armed_configs = 0usize;
    for cfg in CONFIGS {
        let fx = chain_fixture(cfg);
        straddles += fx
            .lo
            .iter()
            .zip(&fx.hi)
            .filter(|(l, h)| **l < 0.0 && **h > 0.0)
            .count();
        let big_fold = cfg.rel >= 1e-6;
        // Arm A: seeded, f64 lane, no backward-interm.
        let (root_a, taps_a) = build_tapped(&fx, policy(true, false), None, RoundMode::Outward);
        assert_eq!(
            taps_a.len(),
            root_a.layers.len(),
            "one seed step per trunk layer"
        );
        for tap in &taps_a {
            assert_eq!(tap.declined, None, "a chain trunk must not decline");
            tightened += tap.stats.tightened;
            stabilized += tap.stats.stabilized;
            crossed += tap.stats.crossed;
        }
        active_relus += root_a
            .layers
            .iter()
            .map(|l| l.l.iter().filter(|&&v| v >= 0.0).count())
            .sum::<usize>();
        // Arm B: seeded with backward-interm (production-sized folds only:
        // the backward engine is not the object under test here).
        let root_b = (!big_fold)
            .then(|| build_tapped(&fx, policy(true, false), Some(bi_cfg()), RoundMode::Outward).0);
        // Arm C: seeded on the f32 COMPUTE lane (one fixture).
        let root_c = (cfg.seed == CONFIGS[2].seed)
            .then(|| build_tapped(&fx, policy(true, true), None, RoundMode::Outward).0);
        // The pure IBP chain (the ny-cli real-ONNX oracle).
        let pure = pure_ibp_trunk_boxes(&fx.net, &fx.lo, &fx.hi).expect("pure IBP chain");
        let tag = format!("cfg {:?}", fx.cfg);
        let box_a = DyBoxes::of_root(&format!("{tag} arm A"), &root_a);
        let box_b = root_b
            .as_ref()
            .map(|r| DyBoxes::of_root(&format!("{tag} arm B (bi)"), r));
        let box_c = root_c
            .as_ref()
            .map(|r| DyBoxes::of_root(&format!("{tag} arm C (f32)"), r));
        let box_pure = DyBoxes::of_pairs(&format!("{tag} pure IBP"), &pure);
        // ATTRIBUTION arm: the DARK root (tableau only). The published seeded
        // box is `tableau ∩ IBP`, so an escape there is a defect on EITHER side;
        // this arm and the pure chain say which.
        let dark = RootGates::build_with_policy_for_test(
            &fx.net,
            &fx.lo,
            &fx.hi,
            RoundMode::Outward,
            policy(false, false),
            None,
        )
        .expect("dark root builds")
        .0;
        let box_dark = DyBoxes::of_root(&format!("{tag} DARK tableau"), &dark);
        // Mutation arm: the IBP fold charge DROPPED.
        let root_m = big_fold.then(|| {
            with_ibp_fold_charge_dropped_for_test(|| {
                RootGates::build_with_policy_for_test(
                    &fx.net,
                    &fx.lo,
                    &fx.hi,
                    RoundMode::Outward,
                    policy(true, false),
                    None,
                )
            })
            .expect("mutated root builds")
            .0
        });
        if root_m.is_some() {
            mutation_armed_configs += 1;
        }
        let box_m = root_m.as_ref().map(|r| DyBoxes::of_root("mutation", r));

        let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x5A3F_1E00);
        // (member, points) plan: stored + 15 random ball vertices, 128 points
        // each (64 vertices + 64 interior), then the targeted extremes.
        let mut plan: Vec<(Member, Vec<Vec<f64>>)> = Vec::new();
        for m in 0..16 {
            let member = if m == 0 {
                Member::default()
            } else {
                random_member(&fx.spec, &mut rng)
            };
            let pts: Vec<Vec<f64>> = (0..128)
                .map(|i| random_point(&fx.lo, &fx.hi, &mut rng, i % 2 == 0))
                .collect();
            plan.push((member, pts));
        }
        let n0 = root_a.layers[0].n;
        let mut targeted = Vec::new();
        for j in 0..n0 {
            for upper in [true, false] {
                targeted.push(targeted_extreme(&fx, j, upper));
            }
        }
        for (member, pts) in &plan {
            let xops = exact_ops(&fx.spec, member);
            for x in pts {
                let ex = exact_forward(&xops, x, false);
                samples += 1;
                checks += ex.trunk.iter().map(Vec::len).sum::<usize>();
                n_fail += box_a.escapes(&ex, "sample", &mut failures);
                if let Some(b) = box_b.as_ref() {
                    n_fail += b.escapes(&ex, "sample", &mut failures);
                }
                // The f32-lane arm and the pure chain are secondary oracles:
                // a strided subset keeps the exact loop affordable in debug.
                if samples.is_multiple_of(4) {
                    if let Some(b) = box_c.as_ref() {
                        n_fail += b.escapes(&ex, "sample", &mut failures);
                    }
                }
                if samples.is_multiple_of(8) {
                    n_fail += box_pure.escapes(&ex, "sample", &mut failures);
                }
                if samples % 4 == 1 {
                    dark_fail += box_dark.escapes(&ex, "sample", &mut failures);
                }
            }
        }
        for (x, member) in &targeted {
            let xops = exact_ops(&fx.spec, member);
            let ex = exact_forward(&xops, x, false);
            samples += 1;
            checks += ex.trunk.iter().map(Vec::len).sum::<usize>();
            n_fail += box_a.escapes(&ex, "targeted", &mut failures);
            if let Some(b) = box_b.as_ref() {
                n_fail += b.escapes(&ex, "targeted", &mut failures);
            }
            n_fail += box_pure.escapes(&ex, "targeted", &mut failures);
            dark_fail += box_dark.escapes(&ex, "targeted", &mut failures);
            if let Some(b) = box_m.as_ref() {
                let mut sink = Vec::new();
                mutation_escapes += b.escapes(&ex, "targeted", &mut sink);
            }
        }
    }
    eprintln!(
        "[mr-ibp-seed test] containment: samples={samples} checks={checks} failures={n_fail} \
         dark_failures={dark_fail} tightened={tightened} stabilized={stabilized} crossed={crossed} \
         active_relus={active_relus} straddles={straddles} \
         mutation_configs={mutation_armed_configs} mutation_escapes={mutation_escapes}"
    );
    assert!(
        samples >= 10_000 && checks > 0 && straddles > 0,
        "VOID: {samples} samples (need 10^4), {checks} checks, {straddles} straddling inputs"
    );
    assert!(
        tightened > 0 && stabilized > 0 && active_relus > 0,
        "VOID: the seed never engaged (tightened={tightened}, stabilized={stabilized}, \
         active relus={active_relus}) — a containment pass would prove nothing"
    );
    assert_eq!(
        crossed, 0,
        "two certified enclosures crossed on a clean fixture"
    );
    assert!(
        mutation_armed_configs > 0 && mutation_escapes > 0,
        "VOID: the dropped-fold-charge mutation escaped no targeted extreme on \
         {mutation_armed_configs} load-bearing configs — the oracle cannot see an \
         under-charged IBP radius"
    );
    assert_eq!(
        dark_fail, 0,
        "THE DARK (tableau-only) ROOT EXCLUDES AN EXACT SAMPLE — a pre-existing tableau \
         soundness defect, NOT the seed's, and still an emergency: {failures:#?}"
    );
    assert!(
        failures.is_empty() && n_fail == 0,
        "SEEDED ROOT BOX EXCLUDES AN EXACT SAMPLE (a false-unsat generator): {n_fail} \
         escapes, first: {failures:#?}"
    );
}

/// The seeded trunk-0 box contains the EXACT hull of the pre-activation over
/// the WHOLE input box and the WHOLE certified fold ball (`fold_ball_hull`,
/// no sampling), and the dropped-fold-charge mutation misses it.
#[test]
fn ibp_seed_trunk0_box_contains_the_exact_fold_ball_hull() {
    let _env = ny_test_utils::env::lock_env_shared();
    let q = |v: f64| BigRational::from_float(v).expect("finite");
    let configs: &[(f64, f64, Option<(f64, f64)>)] = &[
        (1e-13, 0.0, None),
        (1e-13, 1e-6, None),
        (1e-12, 1e-3, None),
        (0.0, 0.0, Some((1e-12, 0.0))),
        (1e-13, 1e-6, Some((1e-13, 1e-6))),
    ];
    let (mut checked, mut failures, mut mutation_missed_configs) = (0usize, Vec::new(), 0usize);
    let mut tightened = 0usize;
    for (ci, &(rel, berr, affine)) in configs.iter().enumerate() {
        let seed = 0x1B5E_F01D_u64 ^ ci as u64;
        let spec = fold_ball_spec(rel, berr, affine, seed);
        let net = TwinNet::compile(&spec).expect("compile");
        let mut rng = StdRng::seed_from_u64(seed ^ 0xA5A5);
        let (lo, hi): (Vec<f64>, Vec<f64>) = (0..spec.n_in)
            .map(|_| {
                let c: f64 = rng.random_range(-0.3..0.3);
                (
                    c - rng.random_range(0.4..1.0),
                    c + rng.random_range(0.2..0.8),
                )
            })
            .unzip();
        let fx_pol = policy(true, false);
        let ((built, _), taps) = {
            let (b, t) = with_ibp_seed_tap_for_test(|| {
                RootGates::build_with_policy_for_test(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    fx_pol,
                    None,
                )
            });
            (b.expect("seeded root"), t)
        };
        tightened += taps.iter().map(|t| t.stats.tightened).sum::<usize>();
        let mutated = with_ibp_fold_charge_dropped_for_test(|| {
            RootGates::build_with_policy_for_test(&net, &lo, &hi, RoundMode::Outward, fx_pol, None)
        })
        .expect("mutated root")
        .0;
        let hull = fold_ball_hull(&spec, &lo, &hi, true);
        let (l, u) = (&built.layers[0].l, &built.layers[0].u);
        assert_eq!(hull.len(), l.len(), "config {ci}: hull width");
        let mut missed = false;
        for (j, (mn, mx)) in hull.iter().enumerate() {
            checked += 1;
            if !(q(l[j]) <= *mn && q(u[j]) >= *mx) {
                failures.push(format!(
                    "config {ci} neuron {j}: seeded box [{:e}, {:e}] vs exact ball hull [{:e}, {:e}]",
                    l[j],
                    u[j],
                    mn.to_f64().unwrap_or(f64::NAN),
                    mx.to_f64().unwrap_or(f64::NAN)
                ));
            }
            let (ml, mu) = (mutated.layers[0].l[j], mutated.layers[0].u[j]);
            if !(q(ml) <= *mn && q(mu) >= *mx) {
                missed = true;
            }
        }
        mutation_missed_configs += usize::from(missed);
    }
    eprintln!(
        "[mr-ibp-seed test] trunk-0 fold-ball hull: checked={checked} failures={} \
         tightened={tightened} mutation_missed_configs={mutation_missed_configs}/{}",
        failures.len(),
        configs.len()
    );
    assert!(
        checked > 0 && tightened > 0,
        "VOID: nothing checked or the seed never moved a box"
    );
    // EVERY config (as the sibling `root_box_encloses_the_exact_fold_ball_hull`
    // pins its legacy mutation): each config makes a different charge
    // load-bearing — config 0 the conv `weight_rel_err` alone, config 3 the
    // affine `scale_rel_err` alone, configs 2 and 4 the absolute `bias_err` /
    // `shift_err` — and a charge the oracle cannot see missing is a charge whose
    // containment check above proves nothing.
    assert_eq!(
        mutation_missed_configs,
        configs.len(),
        "VOID: the dropped-fold-charge mutation contains the exact hull on {} of {} configs",
        configs.len() - mutation_missed_configs,
        configs.len()
    );
    assert!(
        failures.is_empty(),
        "SEEDED TRUNK-0 BOX MISSES THE EXACT FOLD-BALL HULL (false-unsat generator): {failures:#?}"
    );
}

// ---------------------------------------------------------------------------
//  2. Shrink-only and crossing refusal
// ---------------------------------------------------------------------------

#[test]
fn ibp_seed_is_shrink_only_and_refuses_crossings() {
    let _env = ny_test_utils::env::lock_env_shared();
    // (a) The intersection primitive on adversarial arrays.
    let l0 = vec![-1.0, -1.0, 0.5, -2.0, -1.0, -1.0, -1.0, -3.0, -1.0];
    let u0 = vec![1.0, 1.0, 2.0, -0.5, 1.0, 1.0, 1.0, 3.0, 1.0];
    let bl = vec![
        -0.5,
        2.0,
        0.0,
        -1.0,
        f64::NAN,
        -2.0,
        0.25,
        -1.0,
        f64::NEG_INFINITY,
    ];
    let bu = vec![0.5, 3.0, 1.0, -0.75, 0.5, -1.5, 0.75, 4.0, 0.5];
    let (mut l, mut u) = (l0.clone(), u0.clone());
    let st = intersect_layer(&mut l, &mut u, &bl, &bu).expect("shapes agree");
    // 0: tightened both sides, stays unstable.
    assert_eq!((l[0], u[0]), (-0.5, 0.5));
    // 1: IBP above the whole tableau box -> CROSSED, kept.
    assert_eq!((l[1], u[1]), (l0[1], u0[1]));
    // 2: tightened to [0.5, 1.0].
    assert_eq!((l[2], u[2]), (0.5, 1.0));
    // 3: tightened to [-1.0, -0.75].
    assert_eq!((l[3], u[3]), (-1.0, -0.75));
    // 4: NaN lower -> skipped, kept.
    assert_eq!((l[4], u[4]), (l0[4], u0[4]));
    // 5: IBP below the whole tableau box -> CROSSED, kept.
    assert_eq!((l[5], u[5]), (l0[5], u0[5]));
    // 6: tightened AND stabilized (active).
    assert_eq!((l[6], u[6]), (0.25, 0.75));
    // 7: lower tightened only (IBP upper looser).
    assert_eq!((l[7], u[7]), (-1.0, 3.0));
    // 8: -inf lower -> skipped (non-finite), kept.
    assert_eq!((l[8], u[8]), (l0[8], u0[8]));
    assert_eq!(st.neurons, 9);
    assert_eq!(st.tightened, 5, "{st:?}");
    assert_eq!(st.crossed, 2, "{st:?}");
    assert_eq!(st.nonfinite, 2, "{st:?}");
    assert_eq!(st.stabilized, 1, "{st:?}");
    assert_eq!(st.unst_before, 7, "{st:?}");
    assert_eq!(st.unst_after, 6, "{st:?}");
    for j in 0..9 {
        assert!(l[j] >= l0[j] && u[j] <= u0[j], "neuron {j} widened");
    }
    // Shape mismatch: refused, untouched.
    let (mut l2, mut u2) = (l0.clone(), u0.clone());
    assert!(intersect_layer(&mut l2, &mut u2, &bl[..3], &bu[..3]).is_none());
    assert_eq!((l2, u2), (l0.clone(), u0.clone()));
    // Randomized: published is always inside both boxes where not refused.
    let mut rng = StdRng::seed_from_u64(0x5A1F_0001);
    for _ in 0..200 {
        let n = 16;
        let mut l: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..1.0)).collect();
        let mut u: Vec<f64> = l.iter().map(|&a| a + rng.random_range(0.0..2.0)).collect();
        let bl: Vec<f64> = (0..n).map(|_| rng.random_range(-2.5..1.5)).collect();
        let bu: Vec<f64> = bl.iter().map(|&a| a + rng.random_range(0.0..2.0)).collect();
        let (lb, ub) = (l.clone(), u.clone());
        let st = intersect_layer(&mut l, &mut u, &bl, &bu).expect("shapes");
        let mut crossed = 0usize;
        for j in 0..n {
            assert!(l[j] >= lb[j] && u[j] <= ub[j] && l[j] <= u[j]);
            if lb[j].max(bl[j]) > ub[j].min(bu[j]) {
                crossed += 1;
                assert_eq!((l[j], u[j]), (lb[j], ub[j]));
            } else {
                assert_eq!(l[j].to_bits(), lb[j].max(bl[j]).to_bits());
                assert_eq!(u[j].to_bits(), ub[j].min(bu[j]).to_bits());
            }
        }
        assert_eq!(st.crossed, crossed);
    }

    // (a') The LAYER-LEVEL commit the walk publishes: one crossing refuses the
    // seed for the WHOLE layer (FRONTIER_PLAN §7.F2), every other neuron keeps
    // its tableau box too; with no crossing it is `intersect_layer` bit for bit.
    let (mut l, mut u) = (l0.clone(), u0.clone());
    let st = intersect_layer_or_refuse(&mut l, &mut u, &bl, &bu).expect("shapes agree");
    assert_eq!(
        bits(&l),
        bits(&l0),
        "a crossed layer must keep its tableau lower box"
    );
    assert_eq!(
        bits(&u),
        bits(&u0),
        "a crossed layer must keep its tableau upper box"
    );
    assert_eq!(st.crossed, 2, "{st:?}");
    assert_eq!(st.nonfinite, 2, "{st:?}");
    assert_eq!((st.tightened, st.stabilized), (0, 0), "{st:?}");
    assert_eq!((st.unst_before, st.unst_after), (7, 7), "{st:?}");
    // Drop the two crossing neurons (1 and 5): no crossing -> committed.
    fn pick(v: &[f64]) -> Vec<f64> {
        const KEEP: [usize; 7] = [0, 2, 3, 4, 6, 7, 8];
        KEEP.iter().map(|&j| v[j]).collect()
    }
    let (mut l, mut u) = (pick(&l0), pick(&u0));
    let (mut lr, mut ur) = (l.clone(), u.clone());
    let st_all = intersect_layer_or_refuse(&mut l, &mut u, &pick(&bl), &pick(&bu)).expect("shapes");
    let st_ref = intersect_layer(&mut lr, &mut ur, &pick(&bl), &pick(&bu)).expect("shapes");
    assert_eq!(st_all, st_ref);
    assert_eq!(st_all.crossed, 0);
    assert!(
        st_all.tightened > 0,
        "VOID: the uncrossed layer committed nothing"
    );
    assert_eq!((bits(&l), bits(&u)), (bits(&lr), bits(&ur)));
    // Shape mismatch: refused, untouched.
    let (mut l2, mut u2) = (l0.clone(), u0.clone());
    assert!(intersect_layer_or_refuse(&mut l2, &mut u2, &bl[..3], &bu[..3]).is_none());
    assert_eq!((bits(&l2), bits(&u2)), (bits(&l0), bits(&u0)));
    // Randomized: all-or-nothing per layer.
    let mut refused_layers = 0usize;
    for _ in 0..200 {
        let n = 16;
        let mut l: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..1.0)).collect();
        let mut u: Vec<f64> = l.iter().map(|&a| a + rng.random_range(0.0..2.0)).collect();
        let bl: Vec<f64> = (0..n).map(|_| rng.random_range(-2.5..1.5)).collect();
        let bu: Vec<f64> = bl.iter().map(|&a| a + rng.random_range(0.0..2.0)).collect();
        let (lb, ub) = (l.clone(), u.clone());
        let (mut lr, mut ur) = (l.clone(), u.clone());
        let st = intersect_layer_or_refuse(&mut l, &mut u, &bl, &bu).expect("shapes");
        let st_ref = intersect_layer(&mut lr, &mut ur, &bl, &bu).expect("shapes");
        assert_eq!(st.crossed, st_ref.crossed);
        if st_ref.crossed > 0 {
            refused_layers += 1;
            assert_eq!(
                (bits(&l), bits(&u)),
                (bits(&lb), bits(&ub)),
                "refused layer moved"
            );
        } else {
            assert_eq!(st, st_ref);
            assert_eq!((bits(&l), bits(&u)), (bits(&lr), bits(&ur)));
        }
    }
    assert!(refused_layers > 0, "VOID: no randomized layer crossed");

    // (b) Every seeded walk step.
    let mut steps = 0usize;
    let mut moved = 0usize;
    for cfg in CONFIGS {
        let fx = chain_fixture(cfg);
        let (root_a, taps_a) = build_tapped(&fx, policy(true, false), None, RoundMode::Outward);
        for (tap, layer) in taps_a.iter().zip(&root_a.layers) {
            steps += 1;
            assert_eq!(
                tap.ibp_l.len(),
                tap.fwd_l.len(),
                "layer {}: IBP width",
                tap.layer
            );
            for j in 0..tap.fwd_l.len() {
                let (fl, fu) = (tap.fwd_l[j], tap.fwd_u[j]);
                let (ol, ou) = (tap.out_l[j], tap.out_u[j]);
                assert!(
                    ol >= fl && ou <= fu,
                    "layer {} neuron {j}: seed widened",
                    tap.layer
                );
                assert_eq!(
                    ol.to_bits(),
                    fl.max(tap.ibp_l[j]).to_bits(),
                    "layer {} neuron {j}: published lower is not forward ∩ IBP",
                    tap.layer
                );
                assert_eq!(
                    ou.to_bits(),
                    fu.min(tap.ibp_u[j]).to_bits(),
                    "layer {} neuron {j}: published upper is not forward ∩ IBP",
                    tap.layer
                );
                // Backward-interm is off: the seeded box IS the published one.
                assert_eq!(layer.l[j].to_bits(), ol.to_bits());
                assert_eq!(layer.u[j].to_bits(), ou.to_bits());
                moved += usize::from(ol > fl || ou < fu);
            }
            assert_eq!(tap.stats.crossed, 0);
        }
        // With backward-interm (production-sized folds): it only shrinks the
        // seeded box further.
        if cfg.rel < 1e-6 {
            let (root_b, taps_b) =
                build_tapped(&fx, policy(true, false), Some(bi_cfg()), RoundMode::Outward);
            for (tap, layer) in taps_b.iter().zip(&root_b.layers) {
                for j in 0..tap.out_l.len() {
                    assert!(
                        layer.l[j] >= tap.out_l[j] && layer.u[j] <= tap.out_u[j],
                        "layer {} neuron {j}: backward-interm widened the seeded box",
                        tap.layer
                    );
                    assert!(tap.out_l[j] >= tap.fwd_l[j] && tap.out_u[j] <= tap.fwd_u[j]);
                }
            }
        }
    }
    assert!(steps > 0 && moved > 0, "VOID: no seed step moved a box");
}

// ---------------------------------------------------------------------------
//  3. The bi_cal ordering
// ---------------------------------------------------------------------------

/// Every retained clip row's slack, re-derived with `retain_clip_rows` from
/// the row itself and the given calibration box, bit-compared with what the
/// build retained. Returns `(rows compared, rows differing, rows whose neuron
/// the seed moved)`.
fn clip_calibration_diff(root: &RootGates, taps: &[LayerTap]) -> (usize, usize, usize) {
    let n_in = root.mid.len();
    let (mut rows, mut differ, mut engaged) = (0usize, 0usize, 0usize);
    for (layer, tap) in root.layers.iter().zip(taps) {
        let Some(cr) = layer.clip_rows.as_ref() else {
            continue;
        };
        let r = layer.unst.len();
        assert_eq!(cr.m.nrows(), r, "retained rows = unstable neurons");
        let mut mi = Array2::<f64>::zeros((r, n_in + 1));
        let mut di = Array2::<f64>::zeros((r, n_in + 1));
        let mut l_cal = Vec::with_capacity(r);
        let mut u_cal = Vec::with_capacity(r);
        for (row, &i) in layer.unst.iter().enumerate() {
            for c in 0..n_in {
                mi[[row, c]] = cr.m[[row, c]];
                di[[row, c]] = cr.d[[row, c]];
            }
            mi[[row, n_in]] = cr.bm[row];
            di[[row, n_in]] = cr.bd[row];
            // THE forward-only box: what the walk handed the seed.
            l_cal.push(tap.fwd_l[i]);
            u_cal.push(tap.fwd_u[i]);
            engaged += usize::from(tap.out_l[i] > tap.fwd_l[i] || tap.out_u[i] < tap.fwd_u[i]);
        }
        let idx: Vec<usize> = (0..r).collect();
        let want = retain_clip_rows(&mi, &di, &idx, &root.mid, &root.rad, &l_cal, &u_cal);
        for row in 0..r {
            rows += 1;
            if want.sl_lo[row].to_bits() != cr.sl_lo[row].to_bits()
                || want.sl_up[row].to_bits() != cr.sl_up[row].to_bits()
            {
                differ += 1;
            }
        }
    }
    (rows, differ, engaged)
}

/// `clip_rows` slack MUST be recovered against the forward-only lines (the
/// #backward-interm ordering trap, which the seed shares): with the seed
/// armed, every retained slack equals `retain_clip_rows` over the FORWARD box,
/// bit for bit, on rows whose box the seed moved; the mutation that captures
/// `bi_cal` after the seed must be caught.
#[test]
fn ibp_seed_keeps_bi_cal_on_forward_only_lines() {
    ny_test_utils::env::with_env_edits(|env| {
        env.set(
            ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
            "1",
        );
        let (mut rows, mut differ, mut engaged) = (0usize, 0usize, 0usize);
        let (mut m_rows, mut m_differ) = (0usize, 0usize);
        for cfg in CONFIGS {
            let fx = chain_fixture(cfg);
            // Backward-interm OFF: the seed alone must open the capture (the
            // new `|| seed_live` arm of the condition).
            let mut arms: Vec<Option<BiCfg>> = vec![None];
            if cfg.rel < 1e-6 {
                arms.push(Some(bi_cfg()));
            }
            for bi in arms {
                let (root, taps) = build_tapped(&fx, policy(true, false), bi, RoundMode::Outward);
                let (r, d, e) = clip_calibration_diff(&root, &taps);
                rows += r;
                differ += d;
                engaged += e;
                // The mutation: capture AFTER the seed.
                let (mutated, mtaps) = with_bi_cal_after_seed_for_test(|| {
                    build_tapped(&fx, policy(true, false), bi, RoundMode::Outward)
                });
                let (r, d, _) = clip_calibration_diff(&mutated, &mtaps);
                m_rows += r;
                m_differ += d;
            }
        }
        eprintln!(
            "[mr-ibp-seed test] bi_cal: rows={rows} differ={differ} engaged={engaged} \
             mutation_rows={m_rows} mutation_differ={m_differ}"
        );
        assert!(
            rows > 0 && engaged > 0,
            "VOID: no clip row retained ({rows}) or the seed moved none of their boxes ({engaged})"
        );
        assert!(
            m_rows > 0 && m_differ > 0,
            "VOID: capturing bi_cal AFTER the seed changed no retained slack — the test \
             cannot see the ordering defect it pins"
        );
        assert_eq!(
            differ, 0,
            "CLIP SLACK NOT CALIBRATED ON THE FORWARD-ONLY LINES ({differ} of {rows} rows): \
             the halfspaces would cut into the true subdomain (false-unsat generator)"
        );
    });
}

// ---------------------------------------------------------------------------
//  4. Dark byte identity and declines
// ---------------------------------------------------------------------------

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every published field of the two roots, bit for bit.
fn assert_roots_bit_identical(a: &RootGates, b: &RootGates, tag: &str) {
    assert_eq!(a.layers.len(), b.layers.len(), "{tag}: depth");
    assert_eq!(bits(&a.mid), bits(&b.mid), "{tag}: mid");
    assert_eq!(bits(&a.rad), bits(&b.rad), "{tag}: rad");
    for (li, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
        assert_eq!(bits(&x.l), bits(&y.l), "{tag}: layer {li} l");
        assert_eq!(bits(&x.u), bits(&y.u), "{tag}: layer {li} u");
        assert_eq!(bits(&x.alpha), bits(&y.alpha), "{tag}: layer {li} alpha");
        assert_eq!(bits(&x.s), bits(&y.s), "{tag}: layer {li} s");
        assert_eq!(bits(&x.c), bits(&y.c), "{tag}: layer {li} c");
        assert_eq!(bits(&x.ms), bits(&y.ms), "{tag}: layer {li} ms");
        assert_eq!(x.unst, y.unst, "{tag}: layer {li} unst");
    }
}

#[test]
fn ibp_seed_dark_is_byte_identical() {
    let _env = ny_test_utils::env::lock_env_shared();
    let mut compared = 0usize;
    let mut armed_moved = 0usize;
    for cfg in CONFIGS {
        let fx = chain_fixture(cfg);
        let (dark, taps) = with_ibp_seed_tap_for_test(|| {
            RootGates::build_ibp_seeded(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None, false)
                .expect("dark root")
        });
        assert!(taps.is_empty(), "the dark walk must take no seed step");
        let historical =
            RootGates::build_prec(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None, false)
                .expect("historical root");
        let explicit = RootGates::build_with_policy_for_test(
            &fx.net,
            &fx.lo,
            &fx.hi,
            RoundMode::Outward,
            policy(false, false),
            None,
        )
        .expect("explicit dark root")
        .0;
        assert!(!dark.root_precision_policy.ibp_seed);
        assert_roots_bit_identical(&dark, &historical, "dark vs build_prec");
        assert_roots_bit_identical(&dark, &explicit, "dark vs explicit policy");
        // With backward-interm too.
        let dark_bi = RootGates::build_with_policy_for_test(
            &fx.net,
            &fx.lo,
            &fx.hi,
            RoundMode::Outward,
            policy(false, false),
            Some(bi_cfg()),
        )
        .expect("dark bi root")
        .0;
        let hist_bi = RootGates::build_retaining_bi(
            &fx.net,
            &fx.lo,
            &fx.hi,
            RoundMode::Outward,
            None,
            Some(bi_cfg()),
        )
        .expect("historical bi root")
        .0;
        assert_roots_bit_identical(&dark_bi, &hist_bi, "dark vs build_retaining_bi");
        let t = 0usize;
        let adv: Vec<usize> = (1..NOUT).collect();
        let bd = certified_root_bound(&fx.net, &dark, t, &adv).expect("dark bound");
        let bh = certified_root_bound(&fx.net, &historical, t, &adv).expect("historical bound");
        assert_eq!(bd.to_bits(), bh.to_bits(), "dark root bound moved");
        // The armed root must differ somewhere (else "dark" proves nothing).
        let armed =
            RootGates::build_ibp_seeded(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None, true)
                .expect("armed root");
        assert!(armed.root_precision_policy.ibp_seed);
        let moved = armed
            .layers
            .iter()
            .zip(&dark.layers)
            .any(|(a, d)| bits(&a.l) != bits(&d.l) || bits(&a.u) != bits(&d.u));
        armed_moved += usize::from(moved);
        compared += 1;
    }
    assert!(
        compared > 0 && armed_moved > 0,
        "VOID: the armed root equals the dark one on every fixture"
    );
}

#[test]
fn ibp_seed_declines_residual_trunks() {
    let _env = ny_test_utils::env::lock_env_shared();
    let mut residual_layers = 0usize;
    for seed in [0x2E51_D001_u64, 0x2E51_D002, 0x2E51_D003] {
        let mut rng = StdRng::seed_from_u64(seed);
        let spec = tiny_spec(&mut rng, 0.6);
        assert!(
            spec.ops
                .iter()
                .any(|op| matches!(op, TwinOpSpec::Add { .. })),
            "premise: tiny_spec is residual"
        );
        let net = TwinNet::compile(&spec).expect("compile");
        let lo = vec![-0.35; 72];
        let hi = vec![0.4; 72];
        let (armed, taps) = with_ibp_seed_tap_for_test(|| {
            RootGates::build_ibp_seeded(&net, &lo, &hi, RoundMode::Outward, None, true)
                .expect("armed")
        });
        let dark = RootGates::build_ibp_seeded(&net, &lo, &hi, RoundMode::Outward, None, false)
            .expect("dark");
        assert_eq!(
            taps.len(),
            armed.layers.len(),
            "one (declined) line per trunk layer"
        );
        for tap in &taps {
            assert_eq!(tap.declined, Some(Decline::Residual), "layer {}", tap.layer);
            assert_eq!(tap.stats.tightened, 0);
            assert_eq!(tap.stats.crossed, 0);
            assert_eq!(bits(&tap.out_l), bits(&tap.fwd_l));
            assert_eq!(bits(&tap.out_u), bits(&tap.fwd_u));
            residual_layers += 1;
        }
        assert_roots_bit_identical(&armed, &dark, "residual armed vs dark");
    }
    // Parity mode declines on a chain trunk, bit-identically.
    let fx = chain_fixture(&CONFIGS[3]);
    let (armed_p, taps_p) = build_tapped(&fx, policy(true, false), None, RoundMode::Parity);
    let (dark_p, _) = build_tapped(&fx, policy(false, false), None, RoundMode::Parity);
    assert!(!taps_p.is_empty());
    for tap in &taps_p {
        assert_eq!(tap.declined, Some(Decline::Parity));
    }
    assert_roots_bit_identical(&armed_p, &dark_p, "parity armed vs dark");
    assert!(residual_layers > 0, "VOID: no residual layer observed");
}

// ---------------------------------------------------------------------------
//  5. Never verifies a false bound
// ---------------------------------------------------------------------------

/// The certified per-class root bound of the SEEDED root lower-bounds the
/// EXACT margin of every sampled input and fold-ball member (head fold
/// included), and a falsified target keeps a negative bound.
#[test]
fn ibp_seed_never_verifies_a_false_bound() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (mut checks, mut violations, mut falsified_rows, mut moved) =
        (0usize, Vec::new(), 0usize, 0usize);
    for cfg in CONFIGS.iter().filter(|c| c.rel < 1e-6) {
        let fx = chain_fixture(cfg);
        let seeded =
            RootGates::build_ibp_seeded(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None, true)
                .expect("seeded root");
        let seeded_bi =
            build_tapped(&fx, policy(true, false), Some(bi_cfg()), RoundMode::Outward).0;
        let dark =
            RootGates::build_ibp_seeded(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None, false)
                .expect("dark root");
        // The target set: the argmax and the ARGMIN class at the box centre
        // (the argmin target is falsified at the centre).
        let centre: Vec<f64> = fx
            .lo
            .iter()
            .zip(&fx.hi)
            .map(|(&l, &h)| (l + 0.5 * (h - l)).clamp(l, h))
            .collect();
        let stored = exact_ops(&fx.spec, &Member::default());
        let c_logits = exact_forward(&stored, &centre, true).logits;
        let order = |better: bool| -> usize {
            let mut best = 0usize;
            for k in 1..NOUT {
                let beats = if better {
                    c_logits[best].le(&c_logits[k]) && !c_logits[k].le(&c_logits[best])
                } else {
                    c_logits[k].le(&c_logits[best]) && !c_logits[best].le(&c_logits[k])
                };
                if beats {
                    best = k;
                }
            }
            best
        };
        let (t_top, t_low) = (order(true), order(false));
        let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x7E57_0001);
        let mut plan: Vec<(Member, Vec<Vec<f64>>)> = Vec::new();
        for m in 0..8 {
            let member = if m == 0 {
                Member::default()
            } else {
                random_member(&fx.spec, &mut rng)
            };
            let mut pts: Vec<Vec<f64>> = (0..64)
                .map(|i| random_point(&fx.lo, &fx.hi, &mut rng, i % 2 == 0))
                .collect();
            pts.push(centre.clone());
            plan.push((member, pts));
        }
        let exacts: Vec<Vec<Dy>> = plan
            .iter()
            .flat_map(|(member, pts)| {
                let xops = exact_ops(&fx.spec, member);
                pts.iter()
                    .map(|x| exact_forward(&xops, x, true).logits)
                    .collect::<Vec<_>>()
            })
            .collect();
        for t in [t_top, t_low] {
            let adv: Vec<usize> = (0..NOUT).filter(|&k| k != t).collect();
            for (tag, root) in [("seeded", &seeded), ("seeded+bi", &seeded_bi)] {
                let eng = Engine::new(&fx.net, root);
                let re = root_eval(&eng, &fx.net, t, &adv).expect("root_eval");
                assert_eq!(re.dj.len(), adv.len());
                for logits in &exacts {
                    for (r, &j) in adv.iter().enumerate() {
                        checks += 1;
                        let margin = logits[t].sub(&logits[j]);
                        let d = re.dj[r];
                        // `-inf` is a vacuous (sound) lower bound. `+inf` and
                        // NaN CLAIM a margin the exact oracle cannot confirm:
                        // both are violations, never skipped.
                        if d != f64::NEG_INFINITY && (!d.is_finite() || !Dy::of(d).le(&margin)) {
                            violations.push(format!(
                                "{tag} cfg {:?} t={t} j={j}: certified {d:e} > exact margin {:e}",
                                fx.cfg,
                                margin.approx()
                            ));
                        }
                    }
                }
                if t == t_low && t_low != t_top {
                    falsified_rows += 1;
                    let min = re.dj.iter().copied().fold(f64::INFINITY, f64::min);
                    assert!(
                        min < 0.0,
                        "{tag} cfg {:?}: the FALSIFIED target {t} verified (min bound {min:e})",
                        fx.cfg
                    );
                }
            }
            let bs = certified_root_bound(&fx.net, &seeded, t, &adv).expect("seeded bound");
            let bd = certified_root_bound(&fx.net, &dark, t, &adv).expect("dark bound");
            moved += usize::from(bs.to_bits() != bd.to_bits());
        }
    }
    eprintln!(
        "[mr-ibp-seed test] never-verifies: checks={checks} violations={} falsified_rows={falsified_rows} \
         moved_bounds={moved}",
        violations.len()
    );
    assert!(
        checks > 0 && falsified_rows > 0,
        "VOID: nothing checked / no falsified target"
    );
    assert!(
        moved > 0,
        "VOID: the seed never moved a certified root bound"
    );
    assert!(
        violations.is_empty(),
        "SEEDED ROOT BOUND EXCEEDS AN EXACT MARGIN (false unsat): {violations:#?}"
    );
}

// ---------------------------------------------------------------------------
//  6. The typed key
// ---------------------------------------------------------------------------

/// Promotion is not arming, and the preset route is what fires: the typed key
/// resolves `Source::Config` (never LegacyEnv), and the frozen scored route
/// resolves it into the root-precision policy — and nothing else — so every
/// root walk (initial, rebake, epoch) receives it.
#[test]
fn ibp_seed_is_dark_by_default_and_reachable_by_preset() {
    use ny_levers::{LeverValue, Source};

    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_IBP_SEED;
    let resolve = |preset: Option<bool>, raw: Option<&str>| {
        let raw = raw.map(str::to_owned);
        ny_levers::read_over_config_with(decl, move |_| raw, preset.map(LeverValue::Bool))
            .expect("Bool config is admissible for a Bool declaration")
    };
    let absent = resolve(None, None);
    assert!(!absent.value.as_bool());
    assert_eq!(absent.source, Source::Default);
    let armed = resolve(Some(true), None);
    assert!(armed.value.as_bool());
    assert_ne!(armed.source, Source::LegacyEnv);
    assert_eq!(armed.source, Source::Config);
    assert!(!resolve(Some(false), None).value.as_bool());

    let dark = MarginRowRuntimePolicy::from_levers(&ny_levers::LeverSet::freeze_with(
        ny_levers::all(),
        |_| None,
    ))
    .expect("dark policy");
    assert!(!dark.root_precision.ibp_seed, "#mr-ibp-seed must ship dark");
    let name = decl.name;
    let fed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |candidate| {
        (candidate == name).then(|| "1".to_owned())
    });
    let armed_policy = MarginRowRuntimePolicy::from_levers(&fed).expect("armed policy");
    assert!(armed_policy.root_precision.ibp_seed);
    assert_eq!(
        RootPrecisionRuntimePolicy {
            ibp_seed: false,
            ..armed_policy.root_precision
        },
        dark.root_precision,
        "the key moved another precision field"
    );
    assert_eq!(armed_policy.backward_interm, dark.backward_interm);
}
